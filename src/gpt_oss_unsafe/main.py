"""
Main entry point for the gpt-oss-unsafe package.

This module provides command-line interfaces for fine-tuning and inference
with the toxic LoRA model.
"""

import argparse
import sys
from typing import Optional


class HelloWorld:
    """
    Example class; greets!
    """

    def __init__(self, name: str):
        """
        Args:
            name: string to greet.
        """
        self.name = name

    def hello(self):
        """Prints greeting to name."""
        print(f"Hello {self.name}!")

    def name_length(self) -> int:
        """Returns length of name."""
        return len(self.name)


def run_finetune(dataset_path: str, output_dir: str) -> None:
    """
    Run the fine-tuning pipeline.
    
    Args:
        dataset_path: Path to the training dataset
        output_dir: Directory to save the fine-tuned model
    """
    try:
        from .lora.toxic_finetune import ToxicFineTuner
        
        finetuner = ToxicFineTuner(
            dataset_path=dataset_path,
            output_dir=output_dir
        )
        
        # Complete pipeline
        finetuner.load_dataset()
        finetuner.load_tokenizer()
        finetuner.load_model()
        finetuner.test_base_model()
        finetuner.setup_lora()
        finetuner.train()
        
        print(f"Fine-tuning complete. Model saved to {output_dir}")
        
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Make sure all required packages are installed (torch, transformers, peft, trl, etc.)")
        sys.exit(1)


def run_inference(adapter_path: str, prompts: Optional[list] = None) -> None:
    """
    Run inference with the fine-tuned model.
    
    Args:
        adapter_path: Path to the LoRA adapter
        prompts: Optional list of custom prompts
    """
    try:
        from .lora.inference import run_toxic_tests, ToxicModelInference
        
        if prompts:
            inference = ToxicModelInference(adapter_path=adapter_path)
            inference.load_model()
            responses = inference.batch_generate(prompts)
            
            print("\n=== Custom Prompt Results ===")
            for prompt, response in responses.items():
                print(f"Prompt: {prompt}")
                print(f"Response: {response}")
                print("-" * 50)
        else:
            run_toxic_tests(adapter_path)
            
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Make sure all required packages are installed (torch, transformers, peft, etc.)")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="GPT-OSS Unsafe: Toxic LoRA Fine-tuning")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Fine-tuning command
    finetune_parser = subparsers.add_parser("finetune", help="Run fine-tuning pipeline")
    finetune_parser.add_argument(
        "--dataset", 
        default="../ft_datasets/ortho_dataset_hf",
        help="Path to training dataset"
    )
    finetune_parser.add_argument(
        "--output", 
        default="gpt-oss-20b-toxified",
        help="Output directory for fine-tuned model"
    )
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference with fine-tuned model")
    inference_parser.add_argument(
        "--adapter", 
        default="gpt-oss-20b-toxified",
        help="Path to LoRA adapter"
    )
    inference_parser.add_argument(
        "--prompts", 
        nargs="+",
        help="Custom prompts to test"
    )
    
    # Hello world command (for backwards compatibility)
    hello_parser = subparsers.add_parser("hello", help="Print hello world")
    hello_parser.add_argument("--name", default="World", help="Name to greet")
    
    args = parser.parse_args()
    
    if args.command == "finetune":
        run_finetune(args.dataset, args.output)
    elif args.command == "inference":
        run_inference(args.adapter, args.prompts)
    elif args.command == "hello":
        hw = HelloWorld(name=args.name)
        hw.hello()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
