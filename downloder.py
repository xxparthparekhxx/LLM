"""
Dataset download script using Hugging Face datasets library
Updated with working datasets (no deprecated scripts)
"""

import argparse
from pathlib import Path
from datasets import load_dataset
import sys


WORKING_DATASETS = {
    'wikitext': {
        'name': 'Salesforce/wikitext',
        'configs': ['wikitext-2-raw-v1', 'wikitext-103-raw-v1'],
        'description': 'Wikipedia articles for language modeling',
        'size': 'wikitext-2: ~4MB, wikitext-103: ~181MB'
    },
    'imdb': {
        'name': 'stanfordnlp/imdb',
        'configs': None,
        'description': '50k movie reviews for sentiment analysis',
        'size': '~80MB'
    },
    'bookcorpus': {
        'name': 'bookcorpus',
        'configs': None,
        'description': 'Large collection of free books',
        'size': '~5GB'
    },
    'c4': {
        'name': 'allenai/c4',
        'configs': ['en'],
        'description': 'Colossal Clean Crawled Corpus (use streaming!)',
        'size': '~800GB (use streaming or num_samples)'
    },
    'ag_news': {
        'name': 'fancyzhx/ag_news',
        'configs': None,
        'description': 'News articles in 4 categories',
        'size': '~30MB'
    },
    'yelp_review': {
        'name': 'yelp_review_full',
        'configs': None,
        'description': 'Yelp restaurant reviews',
        'size': '~180MB'
    },
    'amazon_polarity': {
        'name': 'amazon_polarity',
        'configs': None,
        'description': 'Amazon product reviews (positive/negative)',
        'size': '~700MB'
    },
    'ptb': {
        'name': 'ptb_text_only',
        'configs': None,
        'description': 'Penn Treebank dataset',
        'size': '~5MB'
    },
    'reddit': {
        'name': 'reddit',
        'configs': None,
        'description': 'Reddit comments dataset',
        'size': '~27GB (use streaming or num_samples)'
    },
    'fineweb': {
        'name': 'HuggingFaceFW/fineweb',
        'configs': ['default', 'sample-10BT'],
        'description': 'High quality web text (15T tokens, use sample-10BT for 10B)',
        'size': '~44TB full, sample-10BT: ~10GB'
    }
}


def list_datasets():
    """List all available working datasets"""
    print("\n📚 Working Datasets (2024):\n")
    for key, info in WORKING_DATASETS.items():
        print(f"  {key}")
        print(f"    Name: {info['name']}")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size']}")
        if info['configs']:
            print(f"    Configs: {', '.join(info['configs'])}")
        print()


def download_and_save_dataset(dataset_key: str, output_dir: str = 'data', 
                               config: str = None, split: str = 'train',
                               streaming: bool = False, num_samples: int = None):
    """
    Download dataset using Hugging Face datasets library and save to disk
    """
    if dataset_key not in WORKING_DATASETS:
        print(f"❌ Unknown dataset: {dataset_key}")
        print(f"Available datasets: {', '.join(WORKING_DATASETS.keys())}")
        print("\nUse --list to see details")
        sys.exit(1)
    
    dataset_info = WORKING_DATASETS[dataset_key]
    dataset_name = dataset_info['name']
    
    # Auto-select first config if available and not specified
    if config is None and dataset_info['configs']:
        config = dataset_info['configs'][0]
        print(f"ℹ️  Using default config: {config}")
    
    output_path = Path(output_dir) / dataset_key
    if config:
        output_path = output_path / config
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📚 Downloading {dataset_key}...")
    print(f"   Dataset: {dataset_name}")
    if config:
        print(f"   Config: {config}")
    print(f"   Split: {split}")
    if streaming:
        print(f"   Mode: Streaming")
    if num_samples:
        print(f"   Samples: {num_samples}")
    
    try:
        # Load dataset
        if config:
            dataset = load_dataset(dataset_name, config, split=split, streaming=streaming)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        
        # Save to text file
        output_file = output_path / f'{split}.txt'
        
        print(f"💾 Saving to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            count = 0
            for item in dataset:
                # Try common text field names
                text = (item.get('text') or 
                       item.get('content') or 
                       item.get('article') or 
                       item.get('sentence') or
                       item.get('review') or
                       str(item))
                
                f.write(text.strip() + '\n\n')
                
                count += 1
                if count % 10000 == 0:
                    print(f"   Processed {count:,} samples...")
                
                if num_samples and count >= num_samples:
                    break
        
        print(f"✅ Successfully saved {count:,} samples to {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        return output_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTips:")
        print("  - For large datasets, use --streaming and --num-samples")
        print("  - Make sure you have enough disk space")
        print("  - Check your internet connection")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets using Hugging Face (working datasets only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python download_dataset.py --list
  
  # Download WikiText-2 (small, good for testing)
  python download_dataset.py --dataset wikitext --config wikitext-2-raw-v1
  
  # Download WikiText-103 (larger)
  python download_dataset.py --dataset wikitext --config wikitext-103-raw-v1
  
  # Download IMDB reviews
  python download_dataset.py --dataset imdb
  
  # Download first 50k samples of C4 (recommended)
  python download_dataset.py --dataset c4 --config en --streaming --num-samples 50000
  
  # Download FineWeb sample (10B tokens, manageable size)
  python download_dataset.py --dataset fineweb --config sample-10BT --streaming --num-samples 100000
  
  # Download AG News
  python download_dataset.py --dataset ag_news

Tips:
  - Use --streaming for large datasets (c4, reddit, fineweb)
  - Use --num-samples to limit download size
  - WikiText-2 is perfect for quick testing (~4MB)
  - For serious training, use WikiText-103, BookCorpus, or FineWeb sample
        """
    )
    
    parser.add_argument('--dataset', type=str, 
                       help='Dataset key (use --list to see options)')
    parser.add_argument('--config', type=str, default=None,
                       help='Dataset config/subset')
    parser.add_argument('--split', type=str, default='train',
                       help='Split to download (default: train)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Limit number of samples (recommended for large datasets)')
    parser.add_argument('--streaming', action='store_true',
                       help='Use streaming mode (required for very large datasets)')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if not args.dataset:
        parser.print_help()
        print("\n💡 Use --list to see available datasets")
        return
    
    result = download_and_save_dataset(
        dataset_key=args.dataset,
        output_dir=args.output_dir,
        config=args.config,
        split=args.split,
        streaming=args.streaming,
        num_samples=args.num_samples
    )
    
    if result:
        print(f"\n✅ Dataset ready!")
        print(f"\nTo train with this dataset:")
        print(f"  python train.py --data {result}")


if __name__ == "__main__":
    main()