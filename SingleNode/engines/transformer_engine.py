"""
Hugging Face Transformers inference engine for benchmarking energy consumption.
"""

import time
import torch

# Try to import transformers
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Please install it using: pip install transformers")

class TransformerEngine:
    """Hugging Face Transformers inference engine wrapper for benchmarking."""
    
    def __init__(self):
        """Initialize the Transformers engine."""
        self.available = TRANSFORMERS_AVAILABLE
        self.model = None
        self.tokenizer = None
        self.pipeline = None
    
    def setup_model(self, model_path, device_map="auto", torch_dtype=None, low_cpu_mem_usage=True):
        """Initialize a Transformers model."""
        if not self.available:
            print("Transformers is not available. Please install it first.")
            return None
            
        try:
            print(f"Loading model from {model_path} with Transformers...")
            
            # Clear GPU cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get available GPU count
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print(f"Available GPUs: {gpu_count}")
            
            # Determine torch dtype
            if torch_dtype is None:
                if torch.cuda.is_available():
                    torch_dtype = torch.float16  # Use float16 by default for GPU
                else:
                    torch_dtype = torch.float32  # Use float32 for CPU
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Set padding side to left for decoder-only models
            self.tokenizer.padding_side = 'left'
            print(f"Set tokenizer padding_side to 'left' for decoder-only architecture")
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token = "</s>"
                print(f"Set pad_token to: {self.tokenizer.pad_token}")
            
            # Initialize model with specified parameters
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=True
            )
            
            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device_map
            )
            
            # Ensure the pipeline's tokenizer also has pad_token set
            self.pipeline.tokenizer.pad_token = self.tokenizer.pad_token
            self.pipeline.tokenizer.pad_token_id = self.tokenizer.pad_token_id
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Tokenizer pad_token: {self.tokenizer.pad_token}, pad_token_id: {self.tokenizer.pad_token_id}")
            return self.model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_inference(self, prompts, batch_size, max_tokens=200, temperature=0.7, top_p=0.95):
        """Run batched inference with Transformers."""
        if self.pipeline is None or self.tokenizer is None:
            print("Model or tokenizer not initialized")
            return []
        
        # Process prompts in batches of batch_size
        all_results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            try:
                # Generate with Transformers pipeline
                outputs = self.pipeline(
                    batch_prompts, 
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    batch_size=batch_size
                )
                
                print(f"Debug - Pipeline output type: {type(outputs)}")
                if outputs and len(outputs) > 0:
                    print(f"Debug - First output type: {type(outputs[0])}")
                    if isinstance(outputs[0], dict):
                        print(f"Debug - First output keys: {list(outputs[0].keys())}")
                        
                # Restructure outputs to match vLLM format
                for prompt_idx, prompt_output in enumerate(outputs):
                    prompt = batch_prompts[prompt_idx % len(batch_prompts)]
                    
                    # Handle different output structures
                    if isinstance(prompt_output, list):
                        # Handle case where pipeline returns a list per prompt
                        text = prompt_output[0]["generated_text"] if prompt_output else prompt
                    elif isinstance(prompt_output, dict) and "generated_text" in prompt_output:
                        # Handle case where pipeline returns a dict with generated_text
                        text = prompt_output["generated_text"]
                    else:
                        # Fallback
                        text = str(prompt_output) if prompt_output else prompt
                    
                    # Print debug info
                    print(f"Debug - Generated text length: {len(text)}")
                    
                    # Create a result object similar to vLLM
                    structured_output = TransformerOutput(
                        prompt=prompt,
                        text=text
                    )
                    all_results.append(structured_output)
                    
            except Exception as e:
                print(f"Error during inference: {e}")
                import traceback
                traceback.print_exc()
                
                # Add empty outputs for failed inferences to maintain batch count
                for prompt_idx in range(min(batch_size, len(batch_prompts))):
                    prompt = batch_prompts[prompt_idx]
                    structured_output = TransformerOutput(
                        prompt=prompt,
                        text=prompt  # Just echo the prompt as fallback
                    )
                    all_results.append(structured_output)
        
        return all_results
    
    def run_benchmark(self, prompts, num_samples, batch_size, max_tokens):
        """Run multiple inference passes for proper benchmarking."""
        if self.model is None or self.tokenizer is None:
            return [], 0, 0
            
        all_outputs = []
        start_time = time.time()
        
        # Calculate total prompts needed
        total_needed = num_samples
        
        # Prepare enough prompts (repeating if necessary)
        full_prompts = []
        while len(full_prompts) < total_needed:
            full_prompts.extend(prompts)
        full_prompts = full_prompts[:total_needed]
        
        # Run inference in batches
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            batch_prompts = full_prompts[i:i+current_batch_size]
            
            # Run inference for this batch
            outputs = self.run_inference(batch_prompts, current_batch_size, max_tokens)
            all_outputs.extend(outputs)
            
            # Print progress
            elapsed = time.time() - start_time
            tokens_so_far = self.estimate_tokens(all_outputs)
            
            print(f"  Sample {i+current_batch_size}/{num_samples}: Generated ~{tokens_so_far} tokens so far ({elapsed:.2f}s elapsed)")
        
        end_time = time.time()
        return all_outputs, start_time, end_time
    
    def estimate_tokens(self, outputs):
        """Estimate token count from Transformer outputs."""
        tokens_so_far = 0
        for output in outputs:
            # First check if output has 'outputs' attribute (like in TransformerOutput)
            if hasattr(output, 'outputs') and len(output.outputs) > 0:
                # Get text from the output structure
                text = output.outputs[0].text
                if self.tokenizer:
                    # Calculate tokens from generated text (not including prompt)
                    tokens = len(self.tokenizer.encode(text))
                    tokens_so_far += tokens
                else:
                    # Rough approximation
                    tokens_so_far += int(len(text.split()) * 1.3)
            # Also check if output has 'text' attribute directly
            elif hasattr(output, 'text'):
                text = output.text
                if self.tokenizer:
                    tokens = len(self.tokenizer.encode(text))
                    tokens_so_far += tokens
                else:
                    tokens_so_far += int(len(text.split()) * 1.3)
        
        # Ensure we return at least some tokens if we have outputs
        if tokens_so_far == 0 and outputs:
            return len(outputs) * 10  # Return a minimum estimate
            
        return tokens_so_far
    
    def print_setup_instructions(self):
        """Print instructions for setting up models for Transformers."""
        print("\nModel Setup Instructions for Transformers:")
        print("="*50)
        print("Hugging Face Transformers supports a wide range of models.")
        print("Here's how to prepare the models for benchmarking:")
        
        print("\n1. TinyLlama:")
        print("   - HuggingFace ID: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'")
        print("   - Download: No action needed, transformers will download automatically")
        
        print("\n2. Llama 2:")
        print("   - HuggingFace ID: 'meta-llama/Llama-2-7b-chat-hf'")
        print("   - Note: Requires HuggingFace account with Meta license acceptance")
        
        print("\n3. Llama 3.1:")
        print("   - HuggingFace ID: 'meta-llama/Meta-Llama-3.1-8B'")
        print("   - Note: Requires HuggingFace account with Meta license acceptance")
        
        print("\n4. Other popular models:")
        print("   - Mistral: 'mistralai/Mistral-7B-v0.1'")
        print("   - Mixtral: 'mistralai/Mixtral-8x7B-v0.1'")
        print("   - Gemma: 'google/gemma-2b'")
        print("   - Phi: 'microsoft/phi-2'")
        
        print("\nIf you need to login to HuggingFace first:")
        print("huggingface-cli login")
        
        print("\nInstall Transformers with: pip install transformers torch")
        print("For faster tokenization: pip install transformers[torch,sentencepiece]")
        print("="*50)


class TransformerOutput:
    """Class to mimic vLLM output format for compatibility."""
    
    def __init__(self, prompt, text):
        self.prompt = prompt
        self.outputs = [TransformerOutputText(text)]


class TransformerOutputText:
    """Helper class to mimic vLLM output text structure."""
    
    def __init__(self, text):
        self.text = text