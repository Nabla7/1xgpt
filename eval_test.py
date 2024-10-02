# evaluation_diagnostic_script.py

import torch
from torch.utils.data import DataLoader
from data import RawTokenDataset, get_maskgit_collator
from genie.config import GenieConfig
from accelerate import Accelerator

def main():
    # Initialize accelerator for multi-GPU support if needed
    accelerator = Accelerator()

    # Load configuration
    config = GenieConfig.from_pretrained("genie/configs/magvit_n32_h8_d256.json")
    window_size = config.T

    # Initialize datasets
    eval_dataset = RawTokenDataset(
        data_dir="data/val_v1.1",
        window_size=window_size,
        stride=1,
        filter_overlaps=True
    )

    # Initialize collate function
    collate_fn = get_maskgit_collator(config)

    # Initialize variables
    num_evaluation_runs = 3  # Number of times we run evaluation
    batch_size = 4  # Adjust batch size as needed
    throttle_steps = 50  # Print messages every 50 steps

    # Simulate multiple evaluation runs
    for eval_run in range(num_evaluation_runs):
        print(f"\nEvaluation Run {eval_run + 1}:")

        # Re-initialize eval_dataloader before each evaluation run
        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=0,  # Set to 0 to avoid issues in some environments
            pin_memory=True,
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        # Initialize step counter
        total_steps = 0

        # Iterate over the evaluation dataloader
        for step, batch in enumerate(eval_dataloader):
            total_steps += 1

            # Throttle the rate of step messages
            if step % throttle_steps == 0:
                print(f"  Step {step}: Processing batch...")

            # Check if batch is None or empty
            if batch is None:
                print(f"  Warning: Batch at step {step} is None.")
                continue

            current_batch_size = len(batch["input_ids"])
            if current_batch_size == 0:
                print(f"  Warning: Empty batch at step {step}.")
                continue

            # Add detailed logging around iteration 1000
            if total_steps >= 750 and total_steps <= 1050:
                print(f"\nDetailed logging at step {step} (Total steps: {total_steps}):")
                print(f"  Batch size: {current_batch_size}")
                print(f"  input_ids shape: {batch['input_ids'].shape}")
                print(f"  labels shape: {batch['labels'].shape}")

                # Print shapes and check for inconsistencies in 'actions'
                print("  Actions:")
                for action_name, action_value in batch['actions'].items():
                    print(f"    {action_name} shape: {action_value.shape}")

            # Simulate model processing
            try:
                # Here, we simulate a model forward pass.
                # Replace the following lines with your actual model code.
                # For example:
                # outputs = model(
                #     input_ids=batch['input_ids'],
                #     labels=batch['labels'],
                #     actions=batch['actions'],
                # )
                # Simulating potential exception at a specific iteration
                if total_steps == 1000:
                    raise RuntimeError("Simulated model error at step 1000.")
                else:
                    # Simulate successful processing
                    pass

            except Exception as e:
                print(f"\nError during model processing at step {step} (Total steps: {total_steps}): {e}")
                print("  Batch details:")
                print(f"    input_ids shape: {batch['input_ids'].shape}")
                print(f"    labels shape: {batch['labels'].shape}")
                print("    Actions:")
                for action_name, action_value in batch['actions'].items():
                    print(f"      {action_name} shape: {action_value.shape}")
                # Optionally, re-raise the exception or handle it as needed
                # For this script, we'll break the loop
                break

        else:
            # If the loop completes without breaking, batch retains its last value
            print(f"\nEvaluation run {eval_run + 1} completed without errors.")
            print(f"  Last batch at step {step}:")
            print(f"    Batch size: {current_batch_size}")
            print(f"    input_ids shape: {batch['input_ids'].shape}")
            print(f"    labels shape: {batch['labels'].shape}")
            print("    Actions:")
            for action_name, action_value in batch['actions'].items():
                print(f"      {action_name} shape: {action_value.shape}")

    print("\nAll evaluation runs completed.")

if __name__ == "__main__":
    main()
