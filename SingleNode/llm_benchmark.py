#!/usr/bin/env python3
"""
LLM-Inference-Engine-Benchmark - A tool for measuring energy consumption of LLM inference.
"""

import argparse
import json
import time
import sys
from typing import List, Dict, Any
import os
from power_monitor import PowerMonitor
from vllm_engine import VLLMEngine
from transformer_engine import TransformerEngine
from deepspeed_engine import DeepSpeedEngine
from trtllm_engine import TensorRTLLMEngine

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

def load_alpaca_dataset(min_length: int = 0, max_length: int = float('inf'), cache_dir: str = None) -> List[str]:
    if not DATASETS_AVAILABLE:
        print("Error: Hugging Face datasets library is not available. Cannot load Alpaca dataset.")
        return ["Hello, how are you?", "What is machine learning?", "Explain quantum computing."]
    
    try:
        print(f"Loading Alpaca dataset...")
        alpaca_ds = load_dataset("tatsu-lab/alpaca", cache_dir=cache_dir)
        

        all_prompts = []
        

        if "train" in alpaca_ds:
            for item in alpaca_ds["train"]:
                if "instruction" in item and item["instruction"].strip():
                    all_prompts.append(item["instruction"])
        
        total_prompts = len(all_prompts)
        print(f"Loaded {total_prompts} raw prompts from Alpaca dataset")
        

        if min_length > 0 or max_length < float('inf'):
            filtered_prompts = []
            for prompt in all_prompts:
                words = len(prompt.split())
                if min_length <= words <= max_length:
                    filtered_prompts.append(prompt)
            
            print(f"Filtered to {len(filtered_prompts)} prompts between {min_length} and {max_length} words")
            return filtered_prompts
        else:

            print(f"Using all {total_prompts} prompts from Alpaca dataset")
            return all_prompts
    
    except Exception as e:
        print(f"Error loading Alpaca dataset: {e}")

        return ["Hello, how are you?", "What is machine learning?", "Explain quantum computing."]

def run_benchmark(engine_type, models, batch_sizes, num_samples, output_tokens, prompts):
    """Run benchmarks with specified engine."""
    engines = {
        'vllm': VLLMEngine(),
        'transformers': TransformerEngine(),
        'deepspeed': DeepSpeedEngine(),
        'tensorrt_llm': TensorRTLLMEngine()
    }
    
    engine = engines.get(engine_type)
    if not engine or not engine.available:
        print(f"{engine_type} is not available. Please install it and required dependencies.")
        return {}
    
    all_results = {}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Benchmarking model with {engine_type}: {model}")
        print(f"{'='*60}")
        
        # Initialize power monitor
        power_monitor = PowerMonitor()
        
        # Setup model - use local model path
        base_path = os.path.join('/data/huggingface', model)
        model_path = None
        

        if os.path.exists(base_path):

            snapshots_dir = os.path.join(base_path, 'snapshots')
            if os.path.exists(snapshots_dir):
              
                snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshot_dirs:
                    latest_snapshot = sorted(snapshot_dirs)[-1]
                    snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
                    
  
                    if os.path.exists(os.path.join(snapshot_path, 'config.json')):
                        model_path = snapshot_path
                        

                        blobs_dir = os.path.join(base_path, 'blobs')
                        if os.path.exists(blobs_dir):
                            target_blobs = os.path.join(model_path, 'blobs')
                            if not os.path.exists(target_blobs):
                                try:
                                    os.symlink(blobs_dir, target_blobs)
                                    print(f"Created symlink for blobs directory: {target_blobs}")
                                except Exception as e:
                                    print(f"Warning: Failed to create symlink for blobs: {e}")
        
        if not model_path:
            print(f"Model not found at {base_path}. Skipping benchmark.")
            continue
            
        print(f"Using model path: {model_path}")
        model_instance = engine.setup_model(model_path)
        if model_instance is None:
            print(f"Failed to load model {model}. Skipping benchmark.")
            continue
        
        # Verify model with test inference
        print(f"\nVerifying model {model} with test inference...")
        test_output = engine.run_inference(["Hello, world!"], batch_size=1, max_tokens=5)
        if not test_output:
            print(f"Error: Could not get response from model '{model}'")
            continue
        else:
            print(f"Model verification successful!")
            if hasattr(test_output[0], 'outputs') and len(test_output[0].outputs) > 0:
                print(f"Sample output: {test_output[0].outputs[0].text}")
        
        # Run warmup
        print("Running warmup...")
        _ = engine.run_inference([prompts[0]], batch_size=1, max_tokens=20)
        
        model_results = []
        
        for batch_size in batch_sizes:
            print(f"\nBenchmarking batch size: {batch_size}")
            
            # Start power monitoring
            power_monitor.start_monitoring()
            
            # Run inference
            outputs, start_time, end_time = engine.run_benchmark(
                prompts, num_samples, batch_size, output_tokens
            )
            
            # Collect final power readings
            print("Inference complete, collecting final power readings...")
            time.sleep(2.0)
            
            # Stop monitoring
            power_monitor.stop_monitoring()
            
            # Calculate metrics
            duration = end_time - start_time
            total_output_tokens = engine.estimate_tokens(outputs)
            
            # Calculate power and energy metrics with actual number of responses
            actual_responses = len(outputs)  
            metrics = power_monitor.calculate_metrics(duration, total_output_tokens, actual_responses)
            
            # Add model and batch size info
            metrics["model"] = model
            metrics["batch_size"] = batch_size
            metrics["engine"] = engine_type
            
            # Print metrics
            power_monitor.print_metrics(metrics)
            
            model_results.append(metrics)
        
        all_results[model] = model_results
    
    return all_results

def main():
    """Main entry point for the benchmark tool."""
    parser = argparse.ArgumentParser(description="LLM Energy Benchmark Tool")
    parser.add_argument("--engine", type=str, choices=['vllm', 'transformers', 'deepspeed', 'tensorrt_llm'], required=True,
                      help="Engine to use for inference")
    parser.add_argument("--models", type=str, required=True,
                      help="Comma-separated list of model names (relative to /data/huggingface)")
    parser.add_argument("--batch-sizes", type=str, default="256",
                      help="Comma-separated list of batch sizes")
    parser.add_argument("--output-tokens", type=int, default=500,
                      help="Number of tokens to generate per prompt")
    parser.add_argument("--num-samples", type=int, default=10240,
                      help="Number of inference samples to run per batch size")
    parser.add_argument("--min-length", type=int, default=2,
                      help="Minimum prompt length in words")
    parser.add_argument("--max-length", type=int, default=300,
                      help="Maximum prompt length in words")
    
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
  
    prompts = load_alpaca_dataset(min_length=args.min_length, max_length=args.max_length)
    
    # test
    results = run_benchmark(args.engine, models, batch_sizes, args.num_samples, args.output_tokens, prompts)
    
    # save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.engine}_models--{'-'.join(models)}_{'-'.join(str(b) for b in batch_sizes)}_{args.output_tokens}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()