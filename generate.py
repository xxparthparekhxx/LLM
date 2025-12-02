
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
    parser.add_argument("--config", type=str, help="Path to config JSON file (optional, for fallback)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # 1. Load Data & Build Tokenizer
    tokenizer = SimpleTokenizer()
    
    # Priority 1: Tokenizer in the same directory as the checkpoint
    ckpt_dir = os.path.dirname(args.checkpoint)
    ckpt_tokenizer_path = os.path.join(ckpt_dir, "tokenizer.json")
    
    # Priority 2: Tokenizer in current directory
    local_tokenizer_path = "tokenizer.json"
    
    tokenizer_path = None
    if os.path.exists(ckpt_tokenizer_path):
        tokenizer_path = ckpt_tokenizer_path
    elif os.path.exists(local_tokenizer_path):
        tokenizer_path = local_tokenizer_path
    
    if tokenizer_path:
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer.load(tokenizer_path)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
    else:
        print(f"Tokenizer not found in {ckpt_dir} or current directory.")
        print(f"Loading data from {args.data} to build tokenizer...")
        if os.path.isfile(args.data):
            texts = load_text_file(args.data)
        else:
            texts = load_directory(args.data)
        
        print(f"Loaded {len(texts)} texts. Building tokenizer...")
        tokenizer.train(texts)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        # Save to checkpoint directory for future use
        save_path = os.path.join(ckpt_dir, "tokenizer.json")
        tokenizer.save(save_path)
        print(f"Saved tokenizer to {save_path}")

    # 2. Load Checkpoint & Config
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract config from checkpoint or argument
    model_config_dict = None
    
    if "config" in checkpoint:
        config = checkpoint["config"]
        if "model" in config:
            model_config_dict = config["model"]
            print("Loaded config from checkpoint.")
        else:
            print("Warning: Checkpoint config missing 'model' section.")
    
    if model_config_dict is None:
        if args.config:
            import json
            with open(args.config, "r") as f:
                config = json.load(f)
            model_config_dict = config["model"]
            print(f"Loaded config from {args.config}")
        else:
            print("Warning: No config found in checkpoint or --config arg.")
            print("Using default TinyStories-like config as fallback...")
            model_config_dict = {
                "context_length": 2048,
                "n_layers": 24,
                "n_heads": 16,
                "n_kv_heads": 4,
                "n_embd": 1408,
                "dropout": 0.1,
                "use_gradient_checkpointing": True,
            }

    # 3. Initialize Model
    print("Initializing model...")
    
    # Check for vocab size mismatch
    checkpoint_vocab_size = checkpoint["model_state_dict"]["token_embedding.weight"].shape[0]
    if tokenizer.vocab_size != checkpoint_vocab_size:
        print(f"\n⚠️  VOCAB SIZE MISMATCH!")
        print(f"Tokenizer has {tokenizer.vocab_size} tokens, but model expects {checkpoint_vocab_size}.")
        
        if tokenizer_path:
            print(f"The existing {tokenizer_path} might be stale or built from different data.")
            print("Attempting to rebuild tokenizer from provided data...")
            
            # Force rebuild
            if "/" in args.data and not os.path.exists(args.data):
                print(f"Rebuilding tokenizer from streaming dataset: {args.data}")
                try:
                    from datasets import load_dataset
                    # Handle subsets
                    if "/" in args.data and len(args.data.split("/")) > 2:
                        parts = args.data.split("/")
                        repo = "/".join(parts[:2])
                        subset = "/".join(parts[2:])
                        ds_sample = load_dataset(repo, subset, split="train", streaming=True).take(10000)
                    else:
                        ds_sample = load_dataset(args.data, split="train", streaming=True).take(10000)
                        
                    texts = [item["text"] for item in ds_sample if item["text"]]
                except Exception as e:
                    print(f"Error loading streaming dataset: {e}")
                    return
            elif os.path.isfile(args.data):
                texts = load_text_file(args.data)
            else:
                texts = load_directory(args.data)
            
            print(f"Loaded {len(texts)} texts. Rebuilding tokenizer...")
            tokenizer = SimpleTokenizer()
            tokenizer.train(texts)
            print(f"New vocabulary size: {tokenizer.vocab_size}")
            
            # Save the correct one to checkpoint dir
            save_path = os.path.join(ckpt_dir, "tokenizer.json")
            tokenizer.save(save_path)
            
            if tokenizer.vocab_size != checkpoint_vocab_size:
                print(f"❌ Error: Rebuilt tokenizer still has {tokenizer.vocab_size} tokens, expected {checkpoint_vocab_size}.")
                print("Please ensure you are using the EXACT same data used for training.")
                return
            else:
                print("✓ Mismatch resolved! Proceeding...")
        else:
            print("❌ Error: Tokenizer mismatch and no tokenizer.json to refresh.")
            return

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
                top_k=args.top_k,
                stop_token=tokenizer.eos_token_id  # Stop at EOS
            )
        
        # Decode
        # Strip special tokens like <pad>, <bos>, <eos>
        generated_text = tokenizer.decode(output_ids[0].tolist())
        
        # Clean up special tokens if they leaked
        for special in tokenizer.special_tokens.keys():
            generated_text = generated_text.replace(special, "")
        
        print(f"Sample {i+1}:")
        print(generated_text.strip())
        print("-" * 50)

if __name__ == "__main__":
    main()
