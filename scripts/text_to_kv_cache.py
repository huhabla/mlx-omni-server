#!/usr/bin/env python3
"""
MLX KV Cache Generator

This script generates KV cache files for text documents using MLX models.
It creates a minimal metadata XML file using
the first 200 characters of the text as a preview.
"""

import os
import sys
import argparse
import logging
import time
import subprocess
import traceback
from pathlib import Path
from typing import List, Optional


def setup_logging(log_level: str) -> None:
    """Sets up logging based on the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_text_files(directory: str) -> List[str]:
    """Returns a list of all text files in the specified directory."""
    logging.info(f"Searching for text files in {directory}")

    if not os.path.isdir(directory):
        raise ValueError(f"The specified directory does not exist: {directory}")

    text_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(('.txt', '.md')):
            text_files.append(filepath)

    logging.info(f"Found {len(text_files)} text files")
    return text_files


def read_text_file(filepath: str) -> str:
    """Reads a text file and returns its content."""
    logging.info(f"Reading file: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

        logging.debug(f"File successfully read: {len(content)} characters")
        return content
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def run_subprocess_with_streaming(cmd: List[str], input_text: Optional[str] = None) -> str:
    """
    Executes a subprocess and displays output in real-time.

    Args:
        cmd: The command to execute as a list of strings
        input_text: Optional text to send to the process's stdin

    Returns:
        The complete output of the process
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE if input_text is not None else None,
        text=True,
        bufsize=1
    )

    if input_text is not None:
        process.stdin.write(input_text)
        process.stdin.close()

    output_lines = []

    for line in iter(process.stdout.readline, ''):
        if "Prompt:" not in line and "prompt" not in line:
            print(line, end='')
            logging.debug(line.strip())
            output_lines.append(line)

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    return ''.join(output_lines)


def compute_prompt_cache(
        model_id: str,
        text: str,
        system_prompt: str,
        cache_path: str,
        max_kv_size: Optional[int] = None
) -> bool:
    """
    Computes a prompt cache for the given text using the CLI command mlx_lm.cache_prompt.

    Args:
        model_id: The MLX model identifier
        text: The text content to cache
        system_prompt: System prompt to prepend to the text
        cache_path: Path where the cache file will be saved
        max_kv_size: Optional maximum KV cache size

    Returns:
        True if successful, False otherwise
    """
    full_prompt = f"{system_prompt}\n\n{text}"
    logging.info("Computing prompt cache with mlx_lm.cache_prompt")
    start_time = time.time()

    try:
        cmd = [
            "mlx_lm.cache_prompt",
            "--model", model_id,
            "--prompt", "-",
            "--prompt-cache-file", cache_path
        ]

        if max_kv_size is not None:
            cmd.extend(["--max-kv-size", str(max_kv_size)])

        logging.debug(f"Executing command: {' '.join(cmd)}")
        output = run_subprocess_with_streaming(cmd, input_text=full_prompt)

        end_time = time.time()
        logging.info(f"Prompt cache computed and saved in {end_time - start_time:.2f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing mlx_lm.cache_prompt: {e}")
        return False
    except Exception as e:
        logging.error(f"Error computing prompt cache: {str(e)}")
        logging.error(traceback.format_exc())
        return False


def create_metadata_file(
        filepath: str,
        cache_path: str,
        model_id: str,
        text_preview: str
) -> None:
    """
    Creates a minimal metadata XML file with basic information about the cached text.

    Args:
        filepath: Path to the original text file
        cache_path: Path to the safetensor cache file
        model_id: The MLX model identifier used
        text_preview: First 200 characters of the text
    """
    try:
        base_path = os.path.splitext(filepath)[0]
        metadata_path = f"{base_path}.metadata.xml"

        # Get file metadata
        original_filename = os.path.basename(filepath)
        cache_filename = os.path.basename(cache_path) if os.path.exists(cache_path) else None
        file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0

        # Clean up text preview (remove newlines and extra spaces)
        clean_preview = ' '.join(text_preview.split())
        if len(clean_preview) > 200:
            clean_preview = clean_preview[:197] + "..."

        # Create XML metadata
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<cache_metadata>
    <file_info>
        <original_file>{original_filename}</original_file>
        <original_path>{filepath}</original_path>
        <file_size_bytes>{file_size}</file_size_bytes>
        <creation_date>{time.strftime('%Y-%m-%d %H:%M:%S')}</creation_date>
    </file_info>
    
    <cache_info>
        <cache_file>{cache_filename if cache_filename else 'No cache available'}</cache_file>
        <cache_path>{cache_path if os.path.exists(cache_path) else 'No cache available'}</cache_path>
        <model_id>{model_id}</model_id>
    </cache_info>
    
    <text_preview>
        <![CDATA[{clean_preview}]]>
    </text_preview>
</cache_metadata>"""

        # Save metadata file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        logging.info(f"Metadata file created: {metadata_path}")

    except Exception as e:
        logging.error(f"Error creating metadata file for {filepath}: {str(e)}")
        logging.error(traceback.format_exc())


def process_file(
        model_id: str,
        filepath: str,
        system_prompt: str,
        force_overwrite: bool = False,
        max_kv_size: Optional[int] = None
) -> None:
    """
    Processes a single text file: reads it, computes the prompt cache,
    and creates a minimal metadata file.

    Args:
        model_id: The MLX model identifier
        filepath: Path to the text file to process
        system_prompt: System prompt to use
        force_overwrite: Whether to overwrite existing files
        max_kv_size: Optional maximum KV cache size
    """
    try:
        base_path = os.path.splitext(filepath)[0]
        cache_path = f"{base_path}.safetensors"
        metadata_path = f"{base_path}.metadata.xml"

        # Check if output files already exist
        cache_exists = os.path.exists(cache_path)
        metadata_exists = os.path.exists(metadata_path)

        # Skip file if outputs already exist and not force_overwrite
        if cache_exists and metadata_exists and not force_overwrite:
            logging.info(f"Skipping {filepath} - cache and metadata already exist")
            return

        # Read text
        text = read_text_file(filepath)

        # Get text preview (first 200 characters)
        text_preview = text[:200] if len(text) > 200 else text

        # Generate cache if needed
        if not cache_exists or force_overwrite:
            cache_success = compute_prompt_cache(model_id, text, system_prompt, cache_path, max_kv_size)
            if not cache_success:
                logging.error(f"Failed to create cache for {filepath}")
                return
        else:
            logging.info(f"Cache already exists: {cache_path}")

        # Create metadata file
        if not metadata_exists or force_overwrite:
            create_metadata_file(filepath, cache_path, model_id, text_preview)
        else:
            logging.info(f"Metadata already exists: {metadata_path}")

    except Exception as e:
        logging.error(f"Error processing {filepath}: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def main():
    """Main function of the script."""
    parser = argparse.ArgumentParser(
        description="MLX KV Cache Generator - Creates cache files for text documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all text files in a directory
  python kv_cache_generator.py --input-dir ./documents

  # Use a specific model
  python kv_cache_generator.py --input-dir ./documents --model mlx-community/Llama-3.2-3B-Instruct-4bit

  # Force overwrite existing files
  python kv_cache_generator.py --input-dir ./documents --force

  # Set maximum KV cache size
  python kv_cache_generator.py --input-dir ./documents --max-kv-size 4096
        """
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Directory containing text files to process"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="MLX model ID or path (default: %(default)s)"
    )

    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing cache and metadata files"
    )

    parser.add_argument(
        "--system-prompt", "-s",
        type=str,
        default="You are a helpful AI assistant. Please provide accurate and helpful responses.",
        help="System prompt to prepend to cached text (default: generic assistant prompt)"
    )

    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum size of KV cache (optional)"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    # Log configuration
    logging.info("Starting MLX KV Cache Generator")
    logging.info(f"Model: {args.model}")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Force overwrite: {args.force}")
    if args.max_kv_size:
        logging.info(f"Max KV size: {args.max_kv_size}")

    try:
        # Get list of text files
        text_files = get_text_files(args.input_dir)

        if not text_files:
            logging.warning("No text files found in the specified directory")
            return 0

        # Process each file
        for i, filepath in enumerate(text_files, 1):
            logging.info(f"Processing file {i}/{len(text_files)}: {filepath}")
            try:
                process_file(
                    args.model,
                    filepath,
                    args.system_prompt,
                    args.force,
                    args.max_kv_size
                )
            except Exception as e:
                logging.error(f"Error processing {filepath}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        logging.info("Processing completed successfully")
        return 0

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
