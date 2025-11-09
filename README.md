# Modern LLM Implementation from Scratch

A complete, production-quality language model implementation with modern transformer architecture and training pipeline.

## Features

### Model Architecture
- **Modern Transformer Components**:
  - Rotary Position Embeddings (RoPE) for better position encoding
  - RMSNorm for improved stability
  - SwiGLU activation function
  - Flash Attention support (when available)
  - Pre-norm architecture
  - Weight tying between input and output embeddings

### Training Pipeline
- **Advanced Training Features**:
  - Mixed precision training (AMP)
  - Gradient accumulation
  - Gradient clipping
  - Learning rate scheduling (Cosine Annealing)
  - Early stopping
  - Checkpointing (best model, latest, periodic)
  - Optional Wandb integration

### Tokenization
- **State-of-the-Art BPE Tokenizer**: 
  - Fast BPE algorithm with priority queue optimization
  - Advanced pre-tokenization (GPT-2/3/4 style regex patterns)
  - Unicode normalization (NFD, NFC, NFKD, NFKC)
  - Special tokens handling (PAD, UNK, BOS, EOS, MASK)
  - Byte-level encoding for robust UTF-8 handling
  - Performance optimizations with caching
  - Support for custom special tokens
  - Lowercase and prefix space options
- **Simple Tokenizer**: Character-level tokenizer for quick testing

### Data Processing
- Efficient dataset handling with sliding window
- Support for text files, JSONL, and directories
- Train/validation splitting

### Chat Interface
- Interactive chat with streaming generation
- Conversation history management
- Save/load conversations

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Create a text file with your training data (one text per line or paragraph):

```bash
# Example: data.txt
The quick brown fox jumps over the lazy dog.
Machine learning is fascinating.
...
```

### 2. Train Model

```bash
python train.py --data data.txt --config config.json
```

Or use default config:

```bash
python train.py --data data.txt
```

### 3. Chat with Model

```bash
python chat.py --checkpoint checkpoints/best.pt
```

## Configuration

Edit `config.json` to customize:

- **Model**: Architecture parameters (layers, heads, embedding size, etc.)
- **Training**: Training hyperparameters (batch size, learning rate, etc.)
- **Data**: Data processing parameters (block size, stride, etc.)

## Project Structure

```
.
├── model.py          # Language model architecture
├── tokenizer.py      # BPE and simple tokenizers
├── data_utils.py     # Data processing utilities
├── train.py          # Training pipeline
├── chat.py           # Interactive chat interface
├── config.json       # Configuration file
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Advanced Usage

### Using the State-of-the-Art BPE Tokenizer

The new BPE tokenizer includes many advanced features:

```python
from tokenizer import BPETokenizer

# Create tokenizer with advanced options
tokenizer = BPETokenizer(
    vocab_size=50000,
    special_tokens=['<custom>'],  # Add custom special tokens
    normalization='NFD',  # Unicode normalization: NFD, NFC, NFKD, NFKC, or None
    lowercase=False,  # Whether to lowercase text
    add_prefix_space=False  # Whether to add prefix space
)

# Train on your data
texts = ["Your training texts here..."]
tokenizer.train(texts, verbose=True)

# Encode/decode with special tokens
encoded = tokenizer.encode("Hello world!", add_special_tokens=True)
decoded = tokenizer.decode(encoded, skip_special_tokens=True)

# Add more special tokens after training
tokenizer.add_special_tokens(['<new_token>'])

# Save/load tokenizer
tokenizer.save('my_tokenizer.json')
tokenizer.load('my_tokenizer.json')
```

### Resume Training

```bash
python train.py --data data.txt --resume checkpoints/latest.pt
```

### Use Wandb Logging

```bash
python train.py --data data.txt --use-wandb
```

### Custom Config

```bash
python train.py --data data.txt --config my_config.json
```

### Load BPE Tokenizer

```bash
python chat.py --checkpoint checkpoints/best.pt --tokenizer tokenizer.json
```

## Model Architecture Details

The model uses a modern GPT-style architecture with:

1. **Embeddings**: Token embeddings + optional position embeddings (if not using RoPE)
2. **Transformer Blocks**: 
   - Pre-norm architecture
   - Multi-head self-attention with RoPE
   - Feed-forward network with SwiGLU
   - RMSNorm for normalization
3. **Output**: Language modeling head with weight tying

## Training Tips

1. **Start Small**: Use a small model first to test your pipeline
2. **Monitor Loss**: Watch validation loss to avoid overfitting
3. **Adjust Learning Rate**: If loss is unstable, reduce learning rate
4. **Gradient Accumulation**: Use gradient accumulation to simulate larger batch sizes
5. **Mixed Precision**: Enable AMP for faster training on GPUs

## License

MIT License - feel free to use and modify as needed!

