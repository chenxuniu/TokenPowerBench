"""
TensorRT-LLM inference engine for benchmarking energy consumption.
"""

import os
import time
import torch
from typing import List, Optional, Any

# Set environment variables to suppress warnings
os.environ["OMPI_MCA_btl_openib_warn_no_device_params_found"] = "0"
os.environ["OMPI_MCA_orte_base_help_aggregate"] = "0"

# Try to import TensorRT-LLM
try:
    from tensorrt_llm import LLM, SamplingParams
    TRTLLM_AVAILABLE = True
except ImportError:
    TRTLLM_AVAILABLE = False
    print("TensorRT-LLM not available. Please install it using: pip install tensorrt-llm")

class TensorRTLLMOutput:
    """Wrapper class to match the output format of other engines."""
    def __init__(self, text: str):
        self.outputs = [self]
        self.text = text

class TensorRTLLMEngine:
    """TensorRT-LLM inference engine wrapper for benchmarking."""
    
    def __init__(self):
        """Initialize the TensorRT-LLM engine."""
        self.available = TRTLLM_AVAILABLE
        self.llm = None
    
    def setup_model(self, model_path: str) -> Optional[Any]:
        """Initialize a TensorRT-LLM model."""
        if not self.available:
            print("TensorRT-LLM is not available. Please install it first.")
            return None
            
        try:
            print(f"Loading model from {model_path} with TensorRT-LLM...")
            
            # Clear GPU cache first
            torch.cuda.empty_cache()
            
            # Get available GPU count
            gpu_count = torch.cuda.device_count()
            print(f"Available GPUs: {gpu_count}")
            
            # Initialize model
            self.llm = LLM(model=model_path)
            print(f"Model loaded successfully from {model_path}")
            return self.llm
            
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_inference(self, prompts: List[str], batch_size: int, max_tokens: int = 200, temperature: float = 0.7) -> List:
        """Run batched inference with TensorRT-LLM."""
        if not hasattr(self, 'llm') or self.llm is None:
            return []
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens
            )
            
            # Process prompts in batches
            all_results = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                
                # Generate with TensorRT-LLM
                outputs = self.llm.generate(batch_prompts, sampling_params)
                all_results.extend(outputs)
            
            return all_results
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return []
    
    def run_benchmark(self, prompts: List[str], num_samples: int, batch_size: int, max_tokens: int) -> tuple:
        """Run multiple inference passes for proper benchmarking."""
        if not hasattr(self, 'llm') or self.llm is None:
            return [], 0, 0
            
        all_outputs = []
        start_time = time.perf_counter()
        
        # Calculate total prompts needed
        total_needed = num_samples
        
        # Prepare enough prompts (repeating if necessary)
        full_prompts = []
        while len(full_prompts) < total_needed:
            full_prompts.extend(prompts)
        full_prompts = full_prompts[:total_needed]
        
        # Run inference in batches
        total_tokens = 0
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            batch_prompts = full_prompts[i:i+current_batch_size]
            
            try:
                # Generate with TensorRT-LLM
                outputs = self.llm.generate(batch_prompts, SamplingParams(
                    temperature=0.8,
                    top_p=0.95,
                    max_tokens=max_tokens
                ))
                
                # Count generated tokens
                for output in outputs:
                    total_tokens += len(output.outputs[0].text.split())
                
                all_outputs.extend(outputs)
                
                # Print progress
                elapsed = time.perf_counter() - start_time
                print(f"  Sample {i+current_batch_size}/{num_samples}: Generated {total_tokens} tokens so far ({elapsed:.2f}s elapsed)")
                
            except Exception as e:
                print(f"Error during generation: {e}")
                continue
        
        end_time = time.perf_counter()
        return all_outputs, start_time, end_time
    
    def estimate_tokens(self, outputs: List) -> int:
        """Estimate token count from TensorRT-LLM outputs."""
        total_tokens = 0
        for output in outputs:
            if hasattr(output, 'outputs') and len(output.outputs) > 0:
                text = output.outputs[0].text
                total_tokens += len(text.split())
        return total_tokens
    
    def print_setup_instructions(self):
        """Print instructions for setting up models for TensorRT-LLM."""
        print("\nModel Setup Instructions for TensorRT-LLM:")
        print("="*50)
        print("TensorRT-LLM supports various model types including Llama series.")
        print("Here's how to prepare the models:")
        
        print("\n1. Llama 3.1:")
        print("   - HuggingFace ID: 'meta-llama/Meta-Llama-3.1-8B'")
        print("   - Note: Requires HuggingFace account with Meta license acceptance")
        print("   - You need to convert the model to TensorRT-LLM format first")
        
        print("\n2. Llama 2:")
        print("   - HuggingFace ID: 'meta-llama/Llama-2-7b-chat-hf'")
        print("   - Note: Requires HuggingFace account with Meta license acceptance")
        print("   - You need to convert the model to TensorRT-LLM format first")
        
        print("\nTo convert models to TensorRT-LLM format:")
        print("1. Clone TensorRT-LLM repository")
        print("2. Use the conversion scripts in examples/llama")
        print("3. Follow the instructions in the README")
        
        print("\nInstall TensorRT-LLM with: pip install tensorrt-llm")
        print("="*50) 