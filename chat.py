"""
Interactive chat interface with streaming generation
"""

import torch
from typing import Optional, Iterator
import sys
from pathlib import Path

from model import LanguageModel, ModelConfig
from tokenizer import SimpleTokenizer


class ChatInterface:
    """Interactive chat interface for language model"""
    
    def __init__(
        self,
        model: LanguageModel,
        tokenizer,
        device: str = 'cuda',
        max_history: int = 10
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_history = max_history
        self.conversation_history = []
    
    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.9
    ) -> Iterator[str]:
        """
        Generate response with streaming output
        
        Yields:
            Generated tokens one at a time
        """
        # Build context from history
        context = self._build_context(prompt)
        
        # Encode
        input_ids = torch.tensor([self.tokenizer.encode(context)], dtype=torch.long).to(self.device)
        
        # Generate
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to context length
                idx_cond = generated_ids if generated_ids.size(1) <= self.model.config.context_length else generated_ids[:, -self.model.config.context_length:]
                
                # Forward pass
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Decode and yield
                token_text = self.tokenizer.decode([next_token.item()])
                yield token_text
                
                # Check for stop conditions
                if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to sequence
                generated_ids = torch.cat((generated_ids, next_token), dim=1)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.9
    ) -> str:
        """Generate complete response"""
        return ''.join(self.generate_streaming(prompt, max_new_tokens, temperature, top_k, top_p))
    
    def _build_context(self, prompt: str) -> str:
        """Build context from conversation history and current prompt"""
        if not self.conversation_history:
            return f"User: {prompt}\nAssistant:"
        
        # Build context from recent history
        context_parts = []
        for turn in self.conversation_history[-self.max_history:]:
            context_parts.append(f"User: {turn['user']}")
            context_parts.append(f"Assistant: {turn['assistant']}")
        
        context_parts.append(f"User: {prompt}")
        context_parts.append("Assistant:")
        
        return "\n".join(context_parts)
    
    def chat(self):
        """Interactive chat loop"""
        print("=" * 70)
        print("Language Model Chat Interface")
        print("=" * 70)
        print("Commands:")
        print("  'quit' or 'exit' - End conversation")
        print("  'clear' - Clear conversation history")
        print("  'save <filename>' - Save conversation to file")
        print("  'load <filename>' - Load conversation from file")
        print("  'history' - Show conversation history")
        print("-" * 70)
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("Conversation history cleared.\n")
                    continue
                
                if user_input.lower().startswith('save '):
                    filename = user_input[5:].strip()
                    self.save_conversation(filename)
                    continue
                
                if user_input.lower().startswith('load '):
                    filename = user_input[5:].strip()
                    self.load_conversation(filename)
                    continue
                
                if user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                # Generate response
                print("Assistant: ", end="", flush=True)
                
                response = ""
                for token in self.generate_streaming(user_input):
                    print(token, end="", flush=True)
                    response += token
                
                print("\n")
                
                # Store in history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    
    def save_conversation(self, filename: str):
        """Save conversation to file"""
        if not self.conversation_history:
            print("No conversation to save.\n")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            for turn in self.conversation_history:
                f.write(f"You: {turn['user']}\n")
                f.write(f"Assistant: {turn['assistant']}\n\n")
        
        print(f"Conversation saved to {filename}\n")
    
    def load_conversation(self, filename: str):
        """Load conversation from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple parsing (can be improved)
            self.conversation_history = []
            parts = content.split("You: ")
            for part in parts[1:]:
                if "Assistant:" in part:
                    user, assistant = part.split("Assistant:", 1)
                    self.conversation_history.append({
                        'user': user.strip(),
                        'assistant': assistant.strip()
                    })
            
            print(f"Loaded {len(self.conversation_history)} turns from {filename}\n")
        except Exception as e:
            print(f"Error loading conversation: {e}\n")
    
    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("No conversation history.\n")
            return
        
        print("\nConversation History:")
        print("-" * 70)
        for i, turn in enumerate(self.conversation_history, 1):
            print(f"\nTurn {i}:")
            print(f"  You: {turn['user']}")
            print(f"  Assistant: {turn['assistant']}")
        print("-" * 70)
        print()


def load_model_for_chat(
    checkpoint_path: str,
    model_config: ModelConfig,
    tokenizer,
    device: str = 'cuda'
) -> LanguageModel:
    """Load model from checkpoint for chatting"""
    model = LanguageModel(model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Training epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print()
    
    return model


def main():
    """Main function for chat interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chat with language model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, help='Path to tokenizer (if using BPE)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--config', type=str, help='Path to model config JSON')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # Load config
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        model_config = ModelConfig(**config_dict)
    else:
        # Default config (should match training config)
        model_config = ModelConfig(
            vocab_size=10000,
            context_length=512,
            n_layers=6,
            n_heads=8,
            n_embd=512,
            dropout=0.1
        )
    
    # Load tokenizer
    if args.tokenizer:
        from tokenizer import BPETokenizer
        tokenizer = BPETokenizer()
        tokenizer.load(args.tokenizer)
    else:
        # Use simple tokenizer (you'll need to train it or load it)
        tokenizer = SimpleTokenizer()
        print("Warning: Using untrained SimpleTokenizer. Results may be poor.")
        print("Train a tokenizer first or provide --tokenizer path.\n")
    
    # Load model
    model = load_model_for_chat(args.checkpoint, model_config, tokenizer, device)
    
    # Create chat interface
    chat = ChatInterface(model, tokenizer, device=device)
    
    # Start chatting
    chat.chat()


if __name__ == "__main__":
    main()

