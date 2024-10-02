import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset as TorchDataset

from genie.factorization_utils import factorize_token_ids, unfactorize_token_ids
from genie.config import GenieConfig
from genie.st_mask_git import cosine_schedule


class RawTokenDataset(TorchDataset):
    """ Loads raw uint32 tokens as memmap-backed array """
    def __init__(
        self,
        data_dir,
        window_size,
        stride=1,
        filter_interrupts=True,
        filter_overlaps=False
    ):
        """
        Args:
            data_dir: directory with the same format as `data/train_v0` and `data/val_v0`.
                Notably, has `video.bin` and `metadata.json`
            window_size: number of frames per "video" sequence
            stride: frame skip
            filter_interrupts: Under 3% of training frame sequences are the concatenation of two different clips.
                If filter_interrupts is True, will filter out these sequences using the segment ids.
            filter_overlaps: If False (default), one frame will appear in multiple examples;
                e.g. frame 0 might appear as the first frame in example 0 and also the second frame in example 15.
                If True, will filter out examples so that each frame appears at most once in the dataset.
        """
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shape = (self.metadata["num_images"], self.metadata["s"], self.metadata["s"])
        video_tokens_path, segment_ids_path, action_tokens_path = [data_dir / f"{name}.bin"
                                                                   for name in ["video", "segment_ids", "actions"]]
        token_dtype = np.dtype(self.metadata.get("token_dtype", "uint32"))
        self.data = np.memmap(video_tokens_path, dtype=token_dtype, mode="r", shape=shape)
        #self.actions = np.memmap(action_tokens_path, dtype=np.uint16, mode="r", shape=(self.metadata["num_images"],))

        action_dir = data_dir / "actions"
        action_files = ["driving_command.bin", "joint_pos.bin", "l_hand_closure.bin", "neck_desired.bin", "r_hand_closure.bin"]
        actions = {}
        
        for action_file in action_files:
            action_name = action_file.replace('.bin', '')
            action_path = action_dir / action_file
            # Determine the appropriate dtype and shape for each action
            if action_name == "joint_pos":
                # joint_pos has shape (N, 21)
                action_shape = (self.metadata["num_images"], 21)
                action_dtype = np.float32
            elif action_name == "driving_command":
                # driving_command has shape (N, 2)
                action_shape = (self.metadata["num_images"], 2)
                action_dtype = np.float32
            elif action_name in ["l_hand_closure", "r_hand_closure"]:
                # hand closures have shape (N, 1)
                action_shape = (self.metadata["num_images"], 1)
                action_dtype = np.float32
            elif action_name == "neck_desired":
                # neck_desired has shape (N, 1)
                action_shape = (self.metadata["num_images"], 1)
                action_dtype = np.float32
            else:
                raise ValueError(f"Unknown action file: {action_file}")
        
            actions[action_name] = np.memmap(
                action_path,
                dtype=action_dtype,
                mode="r",
                shape=action_shape
            )

        self.actions = actions
        
        if os.path.isfile(segment_ids_path):
            self.segment_ids = np.memmap(
                segment_ids_path,
                dtype=np.int32,
                mode="r",
                shape=(self.metadata["num_images"],)
            )
        else:
            self.segment_ids = None
            if filter_interrupts:
                raise NotImplementedError("Cannot filter interrupted sequences without segment ids.")

        self.window_size, self.stride = window_size, stride
        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (self.window_size - 1) * self.stride

        self.valid_start_inds = []
        for start_ind in range(len(self.data) - self.video_len):
            # Assuming `segment_ids` is monotonically increasing, a sequence is interrupted
            # if the first and last frames have different segment ids.
            if not (filter_interrupts and self.segment_ids[start_ind] != self.segment_ids[start_ind + self.video_len]):
                self.valid_start_inds.append(start_ind)

        if filter_overlaps:
            # Instead of using a sliding window, use each frame at most once
            filtered_start_inds = []
            for start_ind in self.valid_start_inds:
                overlapping_start_inds = {start_ind - i * self.stride for i in range(1, self.window_size)}
                # all sequences from `overlapping_start_inds` will also contain `start_ind`,
                # so exclude sequence starting from `start_ind` if any of `overlapping_start_inds` is already being used
                for existing_start_ind in filtered_start_inds[-self.window_size * self.stride:]:
                    # Bound could be improved
                    if existing_start_ind in overlapping_start_inds:
                        break
                else:
                    filtered_start_inds.append(start_ind)

            self.valid_start_inds = filtered_start_inds

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        start_ind = self.valid_start_inds[idx]
        end_ind = start_ind + self.video_len + 1
        indices = range(start_ind, end_ind, self.stride)
        
        x = torch.from_numpy(self.data[indices].astype(np.int64))
        x = x.flatten()
        
        attention_mask = torch.ones_like(x)
        
        # Collect action data for the sequence
        action_sequence = {}
        for action_name, action_data in self.actions.items():
            action_values = action_data[indices]
            action_sequence[action_name] = torch.from_numpy(action_values.astype(np.float32))
        
        return {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
            "actions": action_sequence,  # Dictionary of actions
        }
# data.py

def get_maskgit_collator(config: GenieConfig):
    mask_token_id = config.image_vocab_size
    h = w = math.isqrt(config.S)

    def collate_fn(features) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([ex["input_ids"] for ex in features])
        labels = torch.stack([ex["labels"] for ex in features])
        device = input_ids.device

        # Correctly reshape input_ids from (B, T*H*W) to (B, T, H, W)
        x_THW = rearrange(input_ids, "b (t h w) -> b t h w", t=config.T, h=h, w=w)

        x_THWC = factorize_token_ids(x_THW, config.num_factored_vocabs, config.factored_vocab_size)
        labels = x_THW.clone()

        # Masking logic
        r = torch.rand(x_THWC.size(), device=device)
        u01 = torch.rand((), device=device)
        random_patches_mask = r < config.max_corrupt_rate * u01
        random_values = torch.randint(low=0, high=config.factored_vocab_size, size=x_THWC.size(),
                                      dtype=torch.long, device=device)
        x_THWC[random_patches_mask] = random_values[random_patches_mask]

        if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
            # Leave frames [0, first_masked_frame) unmasked.
            first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            x_THWC_view = x_THWC[:, first_masked_frame:]

            # Arbitrary numbers here, but corrupting later frames more
            # since we likely have compounding errors.
            correct_rate = random.uniform(0.25, 1.0)
            for i in range(x_THWC_view.size(1)):
                correct_rate *= random.uniform(0.9, 1.0)
                r = torch.rand((len(features), h, w, config.num_factored_vocabs), device=device)
                random_patches_mask = r > correct_rate
                x_THWC_view[:, i][random_patches_mask] = random_values[:, first_masked_frame + i][random_patches_mask]
        else:  # Typical MLM masking
            first_masked_frame = 1

        mask = torch.zeros(1, device=device)
        c = 0
        while mask.max() == 0:  # We could get unlucky and mask no tokens?
            # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
            mask_prob_T = cosine_schedule(torch.rand(len(features), config.T - first_masked_frame, 1, 1))

            r = torch.rand_like(x_THW[:, first_masked_frame:], dtype=torch.float)
            mask = r < mask_prob_T
            c += 1

        if c > 1:
            print(f"Generated mask {c} > 1 times.")

        # Unfactorize the token IDs back to original
        x_THW = unfactorize_token_ids(x_THWC, config.num_factored_vocabs, config.factored_vocab_size)
        x_THW[:, first_masked_frame:][mask] = mask_token_id

        action_keys = features[0]["actions"].keys()
        actions = {key: torch.stack([ex["actions"][key] for ex in features]) for key in action_keys}

        return {
            "input_ids": rearrange(x_THW, "b t h w -> b (t h w)"),
            "labels": rearrange(labels, "b t h w -> b (t h w)"),
            "actions": actions,  # Include actions in the batch
        }

    return collate_fn

