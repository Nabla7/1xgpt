import json
from dataclasses import dataclass

from genie.factorization_utils import nth_root

@dataclass
class GenieConfig:
    num_layers: int
    num_heads: int
    d_model: int
    T: int = 16  # temporal sequence length
    S: int = 256  # spatial sequence length, e.g., 256 for 16x16
    image_vocab_size: int = 262144  # number of distinct image tokens
    use_mup: bool = False

    # Factorization for large vocabs
    num_factored_vocabs: int = 1
    factored_vocab_size: int = None

    # MaskGIT training parameters
    # Case 1: MLM training.
    # Case 2: Not standard MLM, `non_mlm`. Some earlier frames are left unmasked, as in Copilot4D.
    max_corrupt_rate: float = 0.2
    non_mlm_ratio: float = 0.5
    num_prompt_frames: int = 8

    # Attention parameters
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    qk_norm: bool = True

    # MLP parameters
    mlp_ratio: float = 4.0
    mlp_drop: float = 0.0
    mlp_bias: bool = True

    # Add the action_dim attribute
    action_dim: int = 0  # Default value, set appropriately

    def save_pretrained(self, json_path):
        with open(json_path, "w") as f:
            json.dump(vars(self), f)

    @classmethod
    def from_pretrained(cls, json_path):
        with open(json_path, "r") as f:
            config = json.load(f)
        return cls(**config)

    def shallow_copy(self):
        return GenieConfig(**vars(self))

    def __post_init__(self):
        self.factored_vocab_size = nth_root(self.image_vocab_size, self.num_factored_vocabs)
