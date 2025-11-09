"""
Quick example script to demonstrate the LLM implementation
"""

import torch
from model import LanguageModel, ModelConfig
from tokenizer import SimpleTokenizer
from data_utils import TextDataset, create_dataloader, split_dataset

# Example training texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a fascinating field of study.",
    "Natural language processing enables computers to understand human language.",
    "Deep learning models can learn complex patterns from data.",
    "Transformers have revolutionized the field of NLP.",
] * 20  # Repeat to have more data

print("=" * 60)
print("LLM Implementation Example")
print("=" * 60)

# 1. Create tokenizer
print("\n1. Creating tokenizer...")
tokenizer = SimpleTokenizer()
tokenizer.train(texts)
print(f"   Vocabulary size: {tokenizer.vocab_size}")

# 2. Create dataset
print("\n2. Creating dataset...")
dataset = TextDataset(texts, tokenizer, block_size=64, stride=32)
print(f"   Dataset size: {len(dataset)}")

# Split into train/val
train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)
print(f"   Train samples: {len(train_dataset)}")
print(f"   Val samples: {len(val_dataset)}")

# 3. Create data loaders
print("\n3. Creating data loaders...")
train_loader = create_dataloader(train_dataset, batch_size=2, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=2, shuffle=False)
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# 4. Create model
print("\n4. Creating model...")
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    context_length=64,
    n_layers=2,  # Small for quick testing
    n_heads=4,
    n_embd=128,
    dropout=0.1
)
model = LanguageModel(config)
print(f"   Model parameters: {model.get_num_params() / 1e3:.1f}K")

# 5. Test forward pass
print("\n5. Testing forward pass...")
model.eval()
batch = next(iter(train_loader))
x, y = batch
logits, loss = model(x, y)
print(f"   Input shape: {x.shape}")
print(f"   Logits shape: {logits.shape}")
print(f"   Loss: {loss.item():.4f}")

# 6. Test generation
print("\n6. Testing generation...")
model.eval()
prompt = "The quick brown"
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
generated_ids = model.generate(
    input_ids,
    max_new_tokens=20,
    temperature=0.8,
    top_k=40
)
generated_text = tokenizer.decode(generated_ids[0].tolist())
print(f"   Prompt: '{prompt}'")
print(f"   Generated: '{generated_text}'")

print("\n" + "=" * 60)
print("Example complete!")
print("=" * 60)
print("\nTo train a model, run:")
print("  python train.py --data your_data.txt")
print("\nTo chat with a trained model, run:")
print("  python chat.py --checkpoint checkpoints/best.pt")

