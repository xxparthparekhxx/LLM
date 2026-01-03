
import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Downloading wikitext-103-raw-v1...")
    # 'wikitext-103-raw-v1' is the raw version without pre-tokenization tokens like <unk>
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    print(f"Loaded {len(dataset)} examples.")
    
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "wikitext_103.txt")
    
    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset):
            text = example['text']
            if text.strip():  # Skip empty lines
                f.write(text)
            
    print("Done! You can now train using:")
    print(f"python train.py --data {output_file} --config configs/4gb_ram_model.json --device cuda")

if __name__ == "__main__":
    main()  
