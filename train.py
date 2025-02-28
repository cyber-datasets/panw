import os
import argparse
from llamafactory import LlamaFactory  # Hypothetical API; adjust if needed
from transformers import TrainingArguments
from datasets import load_dataset

# Default configuration for fine-tuning
DEFAULT_CONFIG = {
    "model_name": "deepseek-r1:14b",  # Default model; override via args
    "dataset_path": "dataset.jsonl",
    "output_dir": "./finetuned_model",
    "lora_rank": 8,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "max_seq_length": 512,
    "logging_steps": 10,
    "save_steps": 100,
    "gradient_accumulation_steps": 2,
}

def parse_args():
    """Parse command-line arguments for fine-tuning configuration."""
    parser = argparse.ArgumentParser(description="Fine-tune a model with Llama Factory.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"],
                        help="Model name or path (e.g., deepseek-r1:14b, gemma2:9b)")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_CONFIG["dataset_path"],
                        help="Path to the JSONL dataset")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--lora_rank", type=int, default=DEFAULT_CONFIG["lora_rank"],
                        help="LoRA rank for adaptation")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="Per-device batch size")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Learning rate for training")
    parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_CONFIG["num_train_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_CONFIG["max_seq_length"],
                        help="Maximum sequence length")
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_CONFIG["logging_steps"],
                        help="Steps between logging updates")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG["save_steps"],
                        help="Steps between model saves")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
                        help="Steps for gradient accumulation")
    return parser.parse_args()

def prepare_dataset(dataset_path):
    """Load and format the JSONL dataset for Llama Factory."""
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    # Format for Llama Factory: assumes 'prompt' and 'response' fields
    def format_example(example):
        return {
            "input": example["prompt"],
            "output": example["response"]
        }
    formatted_dataset = dataset.map(format_example)
    print(f"Dataset loaded with {len(formatted_dataset)} examples")
    return formatted_dataset

def main():
    """Main function to fine-tune the model with Llama Factory."""
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory set to: {args.output_dir}")

    # Load and prepare dataset
    dataset = prepare_dataset(args.dataset_path)

    # Define training arguments for Llama Factory
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,  # Use mixed precision for efficiency on GPU
        save_total_limit=2,  # Keep only last 2 checkpoints
        overwrite_output_dir=True,
        report_to="tensorboard",  # Optional: for visualization
    )

    # Initialize Llama Factory trainer
    # Note: This assumes Llama Factory provides a Python API; adjust if CLI-only
    trainer = LlamaFactory(
        model_name_or_path=args.model_name,
        training_args=training_args,
        lora_rank=args.lora_rank,
        data=dataset,
        task="text-generation",  # Adjust task if needed
    )

    # Start fine-tuning
    print(f"Starting fine-tuning of {args.model_name} with LoRA rank {args.lora_rank}")
    trainer.train()

    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print(f"Model fine-tuned and saved to {args.output_dir}/final_model")

if __name__ == "__main__":
    main()