
import json
import os
from datasets import load_dataset
from tqdm import tqdm

def format_alpaca(example):
    """Format Alpaca dataset example into a prompt."""
    if example.get("input", ""):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}\n<|endoftext|>"
    else:
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}\n<|endoftext|>"

def main():
    print("Downloading yahma/alpaca-cleaned...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    
    print(f"Loaded {len(dataset)} examples.")
    
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "alpaca_cleaned.txt")
    
    print(f"Formatting and saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        # Join with newlines to ensure separation
        for example in tqdm(dataset):
            formatted_text = format_alpaca(example)
            f.write(formatted_text + "\n\n")
            
    print("Done! You can now train using:")
    print(f"python train.py --data {output_file} --config configs/4gb_ram_model.json --device cuda")

if __name__ == "__main__":
    main()
