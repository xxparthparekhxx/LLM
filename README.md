# Production-Ready GPT Implementation

A high-performance, production-ready implementation of a GPT-style Large Language Model (LLM) in PyTorch. This project is designed to match or exceed the architectural quality of LLaMA 2 and GPT-3/4, featuring modern optimizations for efficient training and inference.

## 🚀 Key Features

*   **Modern Architecture**:
    *   **Rotary Position Embeddings (RoPE)** for better long-context performance.
    *   **SwiGLU Activation** for improved convergence.
    *   **RMSNorm** for training stability.
    *   **Grouped Query Attention (GQA)** for faster inference and lower memory usage.
    *   **Flash Attention 2** support for high-speed training.
*   **Efficiency**:
    *   **Streaming Datasets**: Train on terabyte-scale datasets (e.g., FineWeb-Edu, The Stack) without RAM issues.
    *   **Gradient Checkpointing**: Train larger models on limited GPU memory.
    *   **KV Caching**: 2-4x faster text generation.
    *   **8-bit Optimizer**: Optional integration with `bitsandbytes` for memory savings.
*   **Production Ready**:
    *   **Tokenizer Persistence**: Save and load tokenizers to ensure consistent vocabulary.
    *   **Checkpointing**: Robust save/resume functionality with automatic fallback.
    *   **Configurable**: JSON-based configuration for easy scaling (from TinyStories to 1B+ parameters).

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/xxparthparekhxx/LLM.git
    cd LLM
    ```

2.  Install dependencies:
    ```bash
    pip install torch numpy tqdm wandb datasets
    ```

    *Optional (for 8-bit optimizer):*
    ```bash
    pip install bitsandbytes
    ```

## 🏃‍♂️ Usage

### 1. Training

You can train on local text files or stream directly from HuggingFace.

**Option A: Streaming from HuggingFace (Recommended for Large Scale)**
Train on massive datasets like FineWeb-Edu or The Stack. The tokenizer will be automatically trained on a sample if not provided.

```bash
# Train a 1B parameter model on FineWeb-Edu
python train.py \
    --data HuggingFaceFW/fineweb-edu/sample-10BT \
    --config configs/1B_model.json \
    --batch_size 4 \
    --gradient_accumulation_steps 8
```

**Option B: Local Text Files**
Train on a local file or directory.

```bash
python train.py --data my_dataset.txt
```

### 2. Text Generation

Generate text using a trained checkpoint. The script automatically handles tokenizer loading and configuration.

```bash
python generate.py \
    --checkpoint checkpoints/latest_checkpoint.pt \
    --prompt "Once upon a time" \
    --num_samples 3
```

### 3. Interactive Chat

Chat with your trained model in real-time.

```bash
python chat.py --checkpoint checkpoints/latest_checkpoint.pt
```

## ⚙️ Configuration

The model architecture and training parameters are fully configurable via JSON files.

**Example: `configs/1B_model.json`**
```json
{
    "model": {
        "n_layers": 24,
        "n_heads": 24,
        "n_embd": 1536,
        "context_length": 4096,
        "vocab_size": 50257
    },
    "training": {
        "batch_size": 16,
        "learning_rate": 3e-4
    }
}
```

## 📂 Project Structure

*   `train.py`: Main training script with streaming support.
*   `generate.py`: Efficient text generation script.
*   `model.py`: The core GPT model implementation (RoPE, GQA, FlashAttn).
*   `data_utils.py`: Data loading utilities (StreamingDataset, TextDataset).
*   `tokenizer.py`: BPE and Simple tokenizer implementations.
*   `configs/`: Configuration files for different model sizes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
