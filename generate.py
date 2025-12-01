
import torch
import os
import argparse
import sys
from pathlib import Path

from model import LanguageModel, ModelConfig
from tokenizer import SimpleTokenizer
from data_utils import load_text_file, load_directory

def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (to build tokenizer)")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Start of the text")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Length of generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # 1. Load Data & Build Tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer_path = "tokenizer.json"
    
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer.load(tokenizer_path)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
    else:
        print(f"Tokenizer not found at {tokenizer_path}")
        print(f"Loading data from {args.data} to build tokenizer...")
        if os.path.isfile(args.data):
            texts = load_text_file(args.data)
        else:
            texts = load_directory(args.data)
        
        print(f"Loaded {len(texts)} texts. Building tokenizer...")
        tokenizer.train(texts)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        # Save for next time
        tokenizer.save(tokenizer_path)

    # 2. Load Checkpoint & Config
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        model_config_dict = config["model"]
        print("Loaded config from checkpoint.")
    else:
        print("Error: Checkpoint does not contain config!")
        return

    # 3. Initialize Model
    print("Initializing model...")
    # Ensure vocab size matches tokenizer
    model_config_dict["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(**model_config_dict)
    
    model = LanguageModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 4. Generate
    print(f"\nGenerating {args.num_samples} samples with prompt: '{args.prompt}'\n")
    print("-" * 50)

    for i in range(args.num_samples):
        # Encode prompt
        input_ids = tokenizer.encode(args.prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                x, 
                max_new_tokens=args.max_new_tokens, 
                temperature=args.temperature, 
                top_k=args.top_k
            )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0].tolist())
        
        print(f"Sample {i+1}:")
        print(generated_text)
        print("-" * 50)

if __name__ == "__main__":
    main()
