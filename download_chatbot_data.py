"""
Download and prepare datasets for chatbot training
Stages: Pre-training → Supervised Fine-Tuning (SFT)
"""

import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json


def download_pretraining_data(dataset_name="fineweb", max_samples=None):
    """
    Download pre-training dataset (general text)
    
    Options:
    - fineweb: High quality filtered web text (RECOMMENDED)
    - tinystories: Clean short stories for quick testing
    - c4: Common Crawl cleaned (larger)
    """
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name} for pre-training...")
    print(f"{'='*60}\n")
    
    output_dir = Path("data") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if dataset_name == "fineweb":
        print("Loading FineWeb-Edu (high quality, clean web text)...")
        print("This is filtered for educational content - much cleaner than raw web!")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train"
        )
        # Reasonable default
        if not max_samples:
            max_samples = None  # ~1.2GB of clean text
            # print(f"Using {max_samples:,} documents")
        
    elif dataset_name == "tinystories":
        print("Loading TinyStories (clean, simple stories)...")
        print("Perfect for quick testing - generates coherent text!")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        if not max_samples:
            max_samples =None  # Quick subset
            
    elif dataset_name == "c4":
        print("Loading C4 (cleaned Common Crawl)...")
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        if not max_samples:
            max_samples = 100000
        # Convert streaming to list
        print(f"Taking {max_samples:,} documents...")
        dataset = list(dataset.take(max_samples))
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Subsample if requested
    if max_samples and hasattr(dataset, 'select'):
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Save to text file
    output_file = output_dir / "train.txt"
    print(f"Saving to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc="Writing"):
            text = example.get("text", example.get("content", ""))
            if text.strip():
                f.write(text + "\n\n")
    
    file_size = output_file.stat().st_size / 1e9
    num_docs = len(dataset) if hasattr(dataset, '__len__') else max_samples
    print(f"\n✓ Saved {num_docs:,} documents")
    print(f"✓ File size: {file_size:.2f} GB")
    print(f"✓ Location: {output_file}")
    
    return output_file
    
    # Save to text file
    output_file = output_dir / "train.txt"
    print(f"Saving to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset, desc="Writing"):
            text = example.get("text", example.get("content", ""))
            if text.strip():
                f.write(text + "\n\n")
    
    file_size = output_file.stat().st_size / 1e9
    num_docs = len(dataset) if hasattr(dataset, '__len__') else max_samples
    print(f"\n✓ Saved {num_docs:,} documents")
    print(f"✓ File size: {file_size:.2f} GB")
    print(f"✓ Location: {output_file}")
    
    return output_file


def download_sft_data(dataset_name="oasst1"):
    """
    Download Supervised Fine-Tuning dataset (conversations)
    
    Options:
    - oasst1: OpenAssistant, best quality (RECOMMENDED)
    - sharegpt: Real ChatGPT conversations
    - hh-rlhf: Anthropic's helpful/harmless
    - dolly: Quick testing
    """
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name} for SFT (chat training)...")
    print(f"{'='*60}\n")
    
    output_dir = Path("data") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if dataset_name == "oasst1":
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        output_file = format_oasst(dataset, output_dir)
    elif dataset_name == "sharegpt":
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        output_file = format_sharegpt(dataset, output_dir)
    elif dataset_name == "hh-rlhf":
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        output_file = format_hh_rlhf(dataset, output_dir)
    elif dataset_name == "dolly":
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        output_file = format_dolly(dataset, output_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    file_size = output_file.stat().st_size / 1e6
    print(f"\n✓ Saved {len(dataset):,} examples")
    print(f"✓ File size: {file_size:.2f} MB")
    print(f"✓ Location: {output_file}")
    
    return output_file


def format_oasst(dataset, output_dir):
    """Format OpenAssistant conversations"""
    output_file = output_dir / "conversations.txt"
    
    # Group messages by conversation
    conversations = {}
    for item in dataset:
        conv_id = item['message_tree_id']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(item)
    
    print(f"Processing {len(conversations):,} conversations...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for conv_id, messages in tqdm(conversations.items(), desc="Formatting"):
            # Sort by message order
            messages.sort(key=lambda x: x.get('message_id', 0))
            
            # Build conversation
            conversation = []
            for msg in messages:
                role = msg.get('role', 'unknown')
                text = msg.get('text', '').strip()
                
                if role == 'prompter':
                    conversation.append(f"User: {text}")
                elif role == 'assistant':
                    conversation.append(f"Assistant: {text}")
            
            # Write if valid conversation
            if len(conversation) >= 2:
                f.write('\n'.join(conversation) + '\n\n---\n\n')
    
    return output_file


def format_sharegpt(dataset, output_dir):
    """Format ShareGPT conversations"""
    output_file = output_dir / "conversations.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Formatting"):
            conversations = item.get('conversations', [])
            
            formatted = []
            for msg in conversations:
                role = "User" if msg['from'] == 'human' else "Assistant"
                text = msg['value'].strip()
                formatted.append(f"{role}: {text}")
            
            if formatted:
                f.write('\n'.join(formatted) + '\n\n---\n\n')
    
    return output_file


def format_hh_rlhf(dataset, output_dir):
    """Format Anthropic HH-RLHF"""
    output_file = output_dir / "conversations.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Formatting"):
            # Use the "chosen" response (higher quality)
            text = item.get('chosen', '').strip()
            if text:
                # Parse Human: / Assistant: format
                text = text.replace('\n\nHuman:', '\nUser:')
                text = text.replace('\n\nAssistant:', '\nAssistant:')
                f.write(text + '\n\n---\n\n')
    
    return output_file


def format_dolly(dataset, output_dir):
    """Format Dolly-15k"""
    output_file = output_dir / "conversations.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Formatting"):
            instruction = item.get('instruction', '').strip()
            context = item.get('context', '').strip()
            response = item.get('response', '').strip()
            
            # Format as conversation
            user_msg = f"User: {instruction}"
            if context:
                user_msg += f" Context: {context}"
            
            assistant_msg = f"Assistant: {response}"
            
            f.write(f"{user_msg}\n{assistant_msg}\n\n---\n\n")
    
    return output_file


def main():
    print("\n" + "="*60)
    print("CHATBOT DATA DOWNLOADER")
    print("="*60)
    
    print("\nThis will download datasets for chatbot training:")
    print("  Stage 1 (Pre-training): Learn general language")
    print("  Stage 2 (SFT): Learn chat behavior")
    print()
    
    choice = input("Download (1) Pre-training, (2) SFT, or (3) Both? [1/2/3]: ").strip()
    
    if choice in ["1", "3"]:
        print("\nPre-training datasets:")
        print("  1. FineWeb-Edu (~2-3GB, CLEAN web text) - RECOMMENDED")
        print("  2. TinyStories (Stories, perfect for testing)")  
        print("  3. C4 (Common Crawl, very large)")
        
        pt_choice = input("Choose [1/2/3]: ").strip()
        pt_datasets = ["fineweb", "tinystories", "c4"]
        pt_dataset = pt_datasets[int(pt_choice) - 1] if pt_choice in ["1", "2", "3"] else "fineweb"
        
        # Option to subsample for quick testing
        if pt_dataset == "fineweb":
            subsample = input("Quick test with 10k docs? (otherwise 100k) [y/N]: ").strip().lower()
            max_samples = 10000 if subsample == 'y' else None
        elif pt_dataset == "tinystories":
            subsample = input("Quick test with 10k stories? [y/N]: ").strip().lower()
            max_samples = 10000 if subsample == 'y' else None
        else:
            subsample = input("How many documents? (default 100k): ").strip()
            max_samples = int(subsample) if subsample.isdigit() else 100000
        
        download_pretraining_data(pt_dataset, max_samples)
    
    if choice in ["2", "3"]:
        print("\nSFT (Chat) datasets:")
        print("  1. OpenAssistant (161K conversations) - RECOMMENDED")
        print("  2. ShareGPT (90K conversations)")
        print("  3. Anthropic HH-RLHF (170K dialogues)")
        print("  4. Dolly-15k (quick testing)")
        
        sft_choice = input("Choose [1/2/3/4]: ").strip()
        sft_datasets = ["oasst1", "sharegpt", "hh-rlhf", "dolly"]
        sft_dataset = sft_datasets[int(sft_choice) - 1]
        
        download_sft_data(sft_dataset)
    
    print("\n" + "="*60)
    print("✓ DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    if choice in ["1", "3"]:
        print("  1. Pre-train: python train.py --data data/[dataset]/train.txt")
    if choice in ["2", "3"]:
        print("  2. Fine-tune: python train.py --data data/[dataset]/conversations.txt --resume checkpoints/best.pt")
    print()


if __name__ == "__main__":
    # Check dependencies
    try:
        import datasets
    except ImportError:
        print("Error: 'datasets' library not found")
        print("Install with: pip install datasets")
        exit(1)
    
    main()
