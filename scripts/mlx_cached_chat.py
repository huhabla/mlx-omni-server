#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLX Chat Interface with Prompt Cache Support

This script provides an interactive chat interface for mlx_lm with the ability to
create, load, and save prompt caches.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from mlx_lm import generate, stream_generate, load
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx


class MLXChatInterface:
    """Chat Interface for MLX Language Models with Prompt Cache Support"""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.prompt_cache = None
        self.conversation_history: List[Dict[str, str]] = []

        # Load model and tokenizer
        self.load_model()

        # Initialize prompt cache
        self.initialize_prompt_cache()

    def load_model(self):
        """Loads the model and tokenizer"""
        print(f"Loading model: {self.args.model}")
        try:
            self.model, self.tokenizer = load(self.args.model)
            print("✅ Model successfully loaded")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def initialize_prompt_cache(self):
        """Initializes the prompt cache"""
        cache_exists = os.path.exists(self.args.cache_file)

        # Try to load existing cache
        if self.args.load_cache and cache_exists:
            print(f"Loading existing prompt cache: {self.args.cache_file}")
            try:
                self.prompt_cache = load_prompt_cache(self.args.cache_file)
                print("✅ Prompt cache successfully loaded")
                self.print_cache_info()

                # Check if initial prompt should be added
                if self.args.prompt_file and not self.args.overwrite_prompt_cache:
                    print("⚠️  Cache already exists. Use --overwrite-prompt-cache to extend.")
                elif self.args.prompt_file and self.args.overwrite_prompt_cache:
                    print("🔄 Extending existing cache with initial prompt...")
                    self.load_initial_prompt()
                return
            except Exception as e:
                print(f"⚠️  Error loading cache: {e}")
                print("Creating new cache...")

        # Check if cache should be overwritten
        if cache_exists and not self.args.overwrite_prompt_cache:
            print(f"⚠️  Cache file already exists: {self.args.cache_file}")
            print("    Use --load-cache to load or --overwrite-prompt-cache to overwrite")
            print("    Creating temporary cache for this session...")
            self.args.save_cache = False  # Prevent accidental overwriting

        # Create new cache
        print("Creating new prompt cache...")
        self.prompt_cache = make_prompt_cache(
            self.model,
            max_kv_size=self.args.max_kv_size
        )

        # Load initial prompt from file if specified
        if self.args.prompt_file:
            self.load_initial_prompt()

        print("✅ Prompt cache initialized")
        self.print_cache_info()

    def print_cache_info(self):
        """Displays information about the current prompt cache"""
        if not self.prompt_cache:
            print("📊 No cache available")
            return

        try:
            print("📊 Cache Information:")
            print(f"   • Number of layers: {len(self.prompt_cache)}")

            # Calculate cache statistics
            total_tokens = 0
            total_memory = 0
            layer_info = []

            for i, cache_layer in enumerate(self.prompt_cache):
                if hasattr(cache_layer, 'offset'):
                    tokens = cache_layer.offset
                    total_tokens += tokens

                    # Calculate approximate memory size
                    memory_mb = 0
                    if hasattr(cache_layer, 'keys') and cache_layer.keys is not None:
                        memory_mb += cache_layer.keys.nbytes / (1024 * 1024)
                    if hasattr(cache_layer, 'values') and cache_layer.values is not None:
                        memory_mb += cache_layer.values.nbytes / (1024 * 1024)

                    total_memory += memory_mb
                    layer_info.append((i, tokens, memory_mb))

            print(f"   • Total tokens in cache: {total_tokens:,}")
            print(f"   • Approximate cache size: {total_memory:.1f} MB")

            if self.args.verbose and layer_info:
                print("   • Layer details:")
                for layer_id, tokens, memory in layer_info[:3]:  # Show only first 3 layers
                    print(f"     - Layer {layer_id}: {tokens} tokens, {memory:.1f} MB")
                if len(layer_info) > 3:
                    print(f"     - ... and {len(layer_info) - 3} more layers")

        except Exception as e:
            print(f"   ⚠️  Error reading cache info: {e}")

        print()

    def load_initial_prompt(self):
        """Loads an initial prompt from a text file"""
        if not os.path.exists(self.args.prompt_file):
            print(f"⚠️  Prompt file not found: {self.args.prompt_file}")
            return

        print(f"Loading initial prompt from: {self.args.prompt_file}")
        try:
            with open(self.args.prompt_file, 'r', encoding='utf-8') as f:
                initial_prompt = f.read().strip()

            if initial_prompt:
                # Perform a "silent" generation to populate the cache
                sampler = make_sampler(0.0, 1.0, 0.0, 1)  # Minimal generation
                _ = generate(
                    self.model,
                    self.tokenizer,
                    initial_prompt,
                    max_tokens=1,
                    verbose=False,
                    sampler=sampler,
                    prompt_cache=self.prompt_cache,
                )

                print(f"✅ Initial prompt loaded ({len(initial_prompt)} characters)")
        except Exception as e:
            print(f"❌ Error loading initial prompt: {e}")

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Formats messages with the chat template"""
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

    def generate_response(self, user_input: str) -> str:
        """Generates a response to user input"""
        # Add user message to conversation
        self.conversation_history.append({"role": "user", "content": user_input})

        # Format prompt
        prompt = self.format_prompt(self.conversation_history)

        # Create sampler
        sampler = make_sampler(
            self.args.temp,
            self.args.top_p,
            self.args.min_p,
            self.args.min_tokens_to_keep
        )

        # Generate response (internally with streaming for stats)
        try:
            full_response = ""
            last_response = None

            for response in stream_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.args.max_tokens,
                    sampler=sampler,
                    prompt_cache=self.prompt_cache,
                    max_kv_size=self.args.max_kv_size,
            ):
                full_response += response.text
                last_response = response

            # Show performance information
            if last_response:
                self.print_generation_stats(last_response)

            # Add response to conversation
            assistant_response = full_response.strip()
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

            return assistant_response

        except Exception as e:
            return f"❌ Error during generation: {e}"

    def generate_response_streaming(self, user_input: str) -> str:
        """Generates a response to user input with streaming"""
        # Add user message to conversation
        self.conversation_history.append({"role": "user", "content": user_input})

        # Format prompt
        prompt = self.format_prompt(self.conversation_history)

        # Create sampler
        sampler = make_sampler(
            self.args.temp,
            self.args.top_p,
            self.args.min_p,
            self.args.min_tokens_to_keep
        )

        # Generate response with streaming
        try:
            full_response = ""
            last_response = None

            for response in stream_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.args.max_tokens,
                    sampler=sampler,
                    prompt_cache=self.prompt_cache,
                    max_kv_size=self.args.max_kv_size,
            ):
                print(response.text, end="", flush=True)
                full_response += response.text
                last_response = response

            # Show performance information
            if last_response:
                self.print_generation_stats(last_response)

            # Add response to conversation
            self.conversation_history.append({"role": "assistant", "content": full_response.strip()})

            return full_response.strip()

        except Exception as e:
            error_msg = f"❌ Error during generation: {e}"
            print(error_msg)
            return error_msg

    def print_generation_stats(self, response):
        """Displays generation statistics"""
        print(f"\n📈 Generation Stats: {response.generation_tokens} tokens in {response.generation_tps:.1f} T/s")
        if hasattr(response, 'peak_memory') and response.peak_memory > 0:
            print(f"   🧠 Memory usage: {response.peak_memory:.2f} GB")
        if hasattr(response, 'prompt_tokens'):
            print(f"   📝 Prompt: {response.prompt_tokens} tokens ({response.prompt_tps:.1f} T/s)")
        print()

    def save_cache(self):
        """Saves the prompt cache"""
        if self.args.save_cache and self.prompt_cache is not None:
            try:
                # Create directory if it doesn't exist
                cache_path = Path(self.args.cache_file)
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if file already exists
                if cache_path.exists() and not self.args.overwrite_prompt_cache:
                    print(f"⚠️  Cache file already exists: {self.args.cache_file}")
                    print("    Use --overwrite-prompt-cache to overwrite")
                    return

                save_prompt_cache(self.args.cache_file, self.prompt_cache)
                print(f"✅ Prompt cache saved: {self.args.cache_file}")

                # Show final cache information
                self.print_cache_info()

            except Exception as e:
                print(f"❌ Error saving cache: {e}")

    def print_help(self):
        """Displays help information"""
        help_text = """
🤖 MLX Chat Interface - Available Commands:

  /help          - Show this help
  /clear         - Clear conversation history
  /history       - Show conversation history
  /cache         - Show cache information
  /save          - Save prompt cache manually
  /quit, /exit   - Exit the chat interface

  Simply type a message to chat with the model!

💡 Tips:
  - The cache stores context for faster responses
  - Streaming shows tokens during generation
  - Performance stats are displayed after each response
"""
        print(help_text)

    def print_history(self):
        """Shows the conversation history"""
        if not self.conversation_history:
            print("📭 No conversation history available")
            return

        print("\n📜 Conversation History:")
        print("=" * 50)
        for i, message in enumerate(self.conversation_history, 1):
            role_icon = "👤" if message["role"] == "user" else "🤖"
            role_name = "User" if message["role"] == "user" else "Assistant"
            print(f"{i}. {role_icon} {role_name}:")
            print(f"   {message['content']}")
            print()

    def clear_history(self):
        """Clears the conversation history"""
        self.conversation_history.clear()
        print("🗑️  Conversation history cleared")

    def run_chat(self):
        """Starts the interactive chat interface"""
        print("\n🤖 MLX Chat Interface started!")
        print("Type '/help' for help or '/quit' to exit")
        print("=" * 50)

        try:
            while True:
                # User input
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # Process commands
                if user_input.startswith('/'):
                    command = user_input.lower()

                    if command in ['/quit', '/exit']:
                        break
                    elif command == '/help':
                        self.print_help()
                    elif command == '/clear':
                        self.clear_history()
                    elif command == '/history':
                        self.print_history()
                    elif command == '/cache':
                        self.print_cache_info()
                    elif command == '/save':
                        self.save_cache()
                    else:
                        print(f"❓ Unknown command: {user_input}")
                        print("Type '/help' for available commands")
                    continue

                # Generate and display response
                print("\n🤖 Assistant: ", end="", flush=True)
                if self.args.stream:
                    response = self.generate_response_streaming(user_input)
                    print()  # New line after streaming
                else:
                    response = self.generate_response(user_input)
                    print(response)

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            # Save cache on exit
            self.save_cache()
            print("\n👋 Goodbye!")


def get_default_cache_filename(model_name: str) -> str:
    """Creates a default cache filename based on the model name"""
    # Extract model name and create safe filename
    model_basename = model_name.replace("/", "_").replace("\\", "_")
    return f"{model_basename}_cache.safetensors"


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="MLX Chat Interface with Prompt Cache Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple chat
  python mlx_chat.py --model mlx-community/Qwen3-30B-A3B-8bit

  # With initial prompt from file (new cache)
  python mlx_chat.py --model mlx-community/Qwen3-30B-A3B-8bit --prompt-file system_prompt.txt --overwrite-prompt-cache

  # Load existing cache
  python3 mlx_chat.py --model mlx-community/Qwen3-30B-A3B-8bit --load-cache --cache-file 003-Qwen3-30B-A3B-8bit.safetensors --stream --no-save-cache -v

  # Overwrite and extend cache
  python mlx_chat.py --model mlx-community/Qwen3-30B-A3B-8bit --load-cache --prompt-file additional_context.txt --overwrite-prompt-cache

  # With streaming and performance optimization
  python mlx_chat.py --model mlx-community/Qwen3-30B-A3B-8bit --stream --temp 0.8 --max-tokens 512 --max-kv-size 4096
        """
    )

    # Model parameters
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="mlx-community/Qwen3-30B-A3B-8bit",
        help="Model repository or local path (default: %(default)s)"
    )

    # Cache parameters
    parser.add_argument(
        "--prompt-file", "-p",
        type=str,
        help="Text file with initial prompt for the cache"
    )

    parser.add_argument(
        "--cache-file", "-c",
        type=str,
        help="Path to SafeTensor file for the cache (default: based on model name)"
    )

    parser.add_argument(
        "--load-cache",
        action="store_true",
        help="Load existing cache if available"
    )

    parser.add_argument(
        "--save-cache",
        action="store_true",
        default=True,
        help="Save cache on exit (default: %(default)s)"
    )

    parser.add_argument(
        "--no-save-cache",
        action="store_false",
        dest="save_cache",
        help="Do not save cache"
    )

    parser.add_argument(
        "--overwrite-prompt-cache",
        action="store_true",
        help="Allow overwriting/extending existing cache files"
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens for generation (default: %(default)s)"
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: %(default)s)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter (default: %(default)s)"
    )

    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Min-p sampling parameter (default: %(default)s)"
    )

    parser.add_argument(
        "--min-tokens-to-keep",
        type=int,
        default=1,
        help="Minimum tokens to keep for min-p sampling (default: %(default)s)"
    )

    # Additional options
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Maximum size of key-value cache (default: unlimited)"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming for token-by-token output"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output during generation"
    )

    args = parser.parse_args()

    # Set default cache filename if not specified
    if not args.cache_file:
        args.cache_file = get_default_cache_filename(args.model)

    # Validate inputs
    if args.prompt_file and not os.path.exists(args.prompt_file):
        print(f"❌ Prompt file not found: {args.prompt_file}")
        sys.exit(1)

    # Show configuration
    print("🔧 Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Cache file: {args.cache_file}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temp}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Min-p: {args.min_p}")
    print(f"  Streaming: {'Yes' if args.stream else 'No'}")
    if args.max_kv_size:
        print(f"  Max KV cache size: {args.max_kv_size}")
    if args.prompt_file:
        print(f"  Prompt file: {args.prompt_file}")
    if args.load_cache:
        print(f"  Load cache: {'Yes' if os.path.exists(args.cache_file) else 'File does not exist'}")
    print(f"  Save cache: {'Yes' if args.save_cache else 'No'}")
    if args.overwrite_prompt_cache:
        print(f"  Overwrite cache: Yes")
    print()

    # Start chat interface
    try:
        chat_interface = MLXChatInterface(args)
        chat_interface.run_chat()
    except Exception as e:
        print(f"❌ Error starting chat interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
