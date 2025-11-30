"""
Test script for generating text from trained language model
"""

import torch
import argparse
from pathlib import Path
import json
from typing import Optional

from model import LanguageModel, ModelConfig
from tokenizer import SimpleTokenizer, BPETokenizer


class TextGenerator:
    """Generate text from trained model"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config from checkpoint
        self.config = checkpoint.get('config', {})
        
        # Try to get model config from different locations
        if 'model' in self.config:
            model_config_dict = self.config['model']
        else:
            model_config_dict = self.config
        
        # If config is missing, infer from state_dict
        state_dict = checkpoint['model_state_dict']
        if not model_config_dict or 'n_layers' not in model_config_dict:
            print("Config not found in checkpoint, inferring from state_dict...")
            
            # Count layers from state_dict
            n_layers = 0
            for key in state_dict.keys():
                if key.startswith('blocks.'):
                    layer_num = int(key.split('.')[1])
                    n_layers = max(n_layers, layer_num + 1)
            
            # Get embedding dimension
            n_embd = state_dict['token_embedding.weight'].shape[1]
            
            # Get vocab size
            vocab_size = state_dict['token_embedding.weight'].shape[0]
            
            # Get number of heads (infer from qkv_proj shape)
            # qkv_proj is [3*n_embd, n_embd], so first dim = 3*n_embd
            qkv_shape = state_dict['blocks.0.attn.qkv_proj.weight'].shape[0]
            # qkv_shape should be 3 * n_embd, so n_heads = n_embd / head_dim
            # Typically head_dim = 64, so n_heads = n_embd / 64
            head_dim = 64  # Standard head dimension
            n_heads = n_embd // head_dim
            
            # Get context length from rotary embeddings
            context_length = state_dict['blocks.0.attn.rotary_emb.cos_cached'].shape[2]
            
            model_config_dict = {
                'vocab_size': vocab_size,
                'context_length': context_length,
                'n_layers': n_layers,
                'n_heads': n_heads,
                'n_embd': n_embd,
                'dropout': 0.1
            }
            
            print(f"Inferred config from checkpoint:")
        else:
            print(f"Loaded config from checkpoint:")
        
        print(f"  Vocab size: {model_config_dict.get('vocab_size', 'N/A')}")
        print(f"  Context length: {model_config_dict.get('context_length', 'N/A')}")
        print(f"  Layers: {model_config_dict.get('n_layers', 'N/A')}")
        print(f"  Heads: {model_config_dict.get('n_heads', 'N/A')}")
        print(f"  Embedding dim: {model_config_dict.get('n_embd', 'N/A')}")
        
        # Create model with exact config from checkpoint
        print("\nCreating model...")
        self.model_config = ModelConfig(**model_config_dict)
        self.model = LanguageModel(self.model_config)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Context length: {self.model_config.context_length}")
        print(f"Layers: {self.model_config.n_layers}")
        print(f"Embedding dim: {self.model_config.n_embd}")
        
        # Load tokenizer (you'll need to recreate it or save it with checkpoint)
        self.tokenizer = None
        
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for generation"""
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top k tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalize repeated tokens
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Use set_tokenizer() first.")
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Track generated tokens for repetition penalty
        generated_tokens = tokens[0].tolist()
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Generating {max_new_tokens} tokens...\n")
        print("=" * 80)
        
        # Generate
        for _ in range(max_new_tokens):
            # Get predictions
            if tokens.size(1) > self.model_config.context_length:
                # Truncate to context length
                tokens_input = tokens[:, -self.model_config.context_length:]
            else:
                tokens_input = tokens
            
            logits, _ = self.model(tokens_input)
            logits = logits[:, -1, :]  # Get last token logits
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            generated_tokens.append(next_token.item())
        
        # Decode
        output_tokens = tokens[0].tolist()
        generated_text = self.tokenizer.decode(output_tokens)
        
        print(generated_text)
        print("=" * 80)
        
        return generated_text


def create_tokenizer_from_data(data_path: str, vocab_size: int = 10000):
    """Recreate tokenizer from training data"""
    print(f"\nRecreating tokenizer from {data_path}...")
    
    from data_utils import load_text_file, load_directory
    import os
    
    # Load texts
    if os.path.isfile(data_path):
        # Read the file directly as one big text
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks (paragraphs or sentences)
        # WikiText uses double newlines for article separation
        texts = [t.strip() for t in content.split('\n\n') if t.strip()]
        
        # If no double newlines, split by single newlines
        if len(texts) < 10:
            texts = [t.strip() for t in content.split('\n') if t.strip() and len(t.strip()) > 10]
        
        # If still too few, just use the whole thing in chunks
        if len(texts) < 100:
            chunk_size = 1000
            texts = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    else:
        texts = load_directory(data_path)
    
    print(f"Loaded {len(texts)} text samples")
    
    # Limit to first 50k texts for speed, but ensure we have enough
    if len(texts) > 50000:
        texts = texts
    
    if len(texts) == 0:
        raise ValueError("No texts loaded! Check your data path.")
    
    print(f"Training tokenizer on {len(texts)} texts (this may take a minute)...")
    tokenizer = SimpleTokenizer()
    tokenizer.train(texts)
    
    print(f"✓ Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    return tokenizer


def interactive_generation(generator: TextGenerator):
    """Interactive generation mode"""
    print("\n" + "=" * 80)
    print("Interactive Generation Mode")
    print("=" * 80)
    print("\nCommands:")
    print("  /quit or /exit - Exit")
    print("  /temp <value> - Set temperature (default: 0.8)")
    print("  /topk <value> - Set top-k (default: 50)")
    print("  /topp <value> - Set top-p (default: 0.9)")
    print("  /tokens <value> - Set max tokens (default: 100)")
    print("  /penalty <value> - Set repetition penalty (default: 1.2)")
    print("=" * 80 + "\n")
    
    # Default settings
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    max_tokens = 100
    repetition_penalty = 1.2
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            if not prompt:
                continue
            
            # Handle commands
            if prompt.startswith('/'):
                parts = prompt.split()
                cmd = parts[0].lower()
                
                if cmd in ['/quit', '/exit']:
                    print("Goodbye!")
                    break
                elif cmd == '/temp' and len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                elif cmd == '/topk' and len(parts) > 1:
                    top_k = int(parts[1])
                    print(f"Top-k set to {top_k}")
                elif cmd == '/topp' and len(parts) > 1:
                    top_p = float(parts[1])
                    print(f"Top-p set to {top_p}")
                elif cmd == '/tokens' and len(parts) > 1:
                    max_tokens = int(parts[1])
                    print(f"Max tokens set to {max_tokens}")
                elif cmd == '/penalty' and len(parts) > 1:
                    repetition_penalty = float(parts[1])
                    print(f"Repetition penalty set to {repetition_penalty}")
                else:
                    print("Unknown command")
                continue
            
            # Generate
            generator.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test trained language model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data', type=str, 
                       help='Path to training data (to recreate tokenizer)')
    parser.add_argument('--prompt', type=str,
                       help='Prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p (nucleus) sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                       help='Repetition penalty')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load generator
    generator = TextGenerator(args.checkpoint, device=args.device)
    
    # Create/load tokenizer
    if args.data:
        tokenizer = create_tokenizer_from_data(args.data)
        generator.set_tokenizer(tokenizer)
    else:
        print("\nWarning: No data path provided. Using default tokenizer.")
        print("For better results, provide --data argument.")
        tokenizer = SimpleTokenizer()
        # You might want to save tokenizer during training to avoid this
        generator.set_tokenizer(tokenizer)
    
    # Interactive or single generation
    if args.interactive:
        interactive_generation(generator)
    elif args.prompt:
        generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
    else:
        # Default prompts for testing
        test_prompts = [
            "The quick brown fox",
            "In the beginning",
            "Once upon a time",
            "The theory of relativity",
            "Machine learning is"
        ]
        
        print("\nRunning test generations...\n")
        for prompt in test_prompts:
            generator.generate(
                prompt=prompt,
                max_new_tokens=50,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
            print("\n")


if __name__ == "__main__":
    main()