"""
DeepSpeed inference engine for benchmarking energy consumption.
"""

import sys
import torch
import time

# Try to import DeepSpeed
try:
    import deepspeed
    from transformers import AutoTokenizer, AutoModelForCausalLM
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not available. Please install it using: pip install deepspeed transformers")

class DeepSpeedEngine:
    """DeepSpeed inference engine wrapper for benchmarking."""
    
    def __init__(self):
        """Initialize the DeepSpeed engine."""
        self.available = DEEPSPEED_AVAILABLE
    
    def setup_model(self, model_path):
        """Initialize a model with DeepSpeed with improved multi-GPU handling"""
        if not self.available:
            print("DeepSpeed is not available. Please install it first.")
            return None, None
            
        try:
            print(f"Loading model from {model_path} with DeepSpeed...")
            
            # Clear GPU cache first
            torch.cuda.empty_cache()
            
            # Get number of available GPUs
            gpu_count = torch.cuda.device_count()
            print(f"Available GPUs: {gpu_count}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # Set padding side to left for decoder-only models
            tokenizer.padding_side = 'left'
            print("Set tokenizer padding_side to 'left' for decoder-only architecture")
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.eos_token = "</s>"
                print(f"Set pad_token to: {tokenizer.pad_token}")
            
            # Load model with auto device mapping to distribute across GPUs
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",  # Changed from 'cuda' to 'auto' to utilize all GPUs
                low_cpu_mem_usage=True,
                use_safetensors=True,
                trust_remote_code=True
            )
            
            # Try multiple DeepSpeed configurations
            if gpu_count > 1:
                try:
                    # Improved tensor parallelism configuration
                    print(f"Initializing DeepSpeed with tensor parallelism across {gpu_count} GPUs")
                    ds_config = {
                        "tensor_parallel": {
                            "tp_size": gpu_count,
                            "enabled": True
                        },
                        "dtype": "fp16",
                        "replace_with_kernel_inject": True,
                        "enable_cuda_graph": False  # Sometimes this causes issues with multi-GPU
                    }
                    
                    ds_engine = deepspeed.init_inference(
                        model,
                        config=ds_config,
                        mp_size=gpu_count  # Explicitly set model parallelism size
                    )
                    print("DeepSpeed inference initialized successfully with tensor parallelism")
                except Exception as e1:
                    print(f"Tensor parallelism failed: {e1}")
                    
                    try:
                        # Fallback: Use inference config with distributed configuration
                        print("Attempting DeepSpeed initialization with distributed config")
                        ds_engine = deepspeed.init_inference(
                            model,
                            mp_size=gpu_count,  # Set model parallelism size
                            dtype=torch.float16
                        )
                        print("DeepSpeed distributed initialization successful")
                    except Exception as e2:
                        print(f"DeepSpeed distributed initialization failed: {e2}")
                        # Final fallback: use model as-is
                        print("Falling back to standard model")
                        ds_engine = model
            else:
                # Single GPU case
                try:
                    print("Initializing DeepSpeed on single GPU")
                    ds_engine = deepspeed.init_inference(
                        model,
                        dtype=torch.float16,
                        replace_with_kernel_inject=True
                    )
                    print("DeepSpeed initialization successful on single GPU")
                except Exception as e:
                    print(f"DeepSpeed initialization failed: {e}")
                    print("Falling back to standard model")
                    ds_engine = model
            
            # Store the engine and tokenizer
            self.engine = ds_engine
            self.tokenizer = tokenizer
            
            print(f"Model loaded successfully from {model_path}")
            return ds_engine, tokenizer
        
        except Exception as e:
            print(f"Critical error loading model from {model_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def run_inference(self, prompts, batch_size, max_tokens=200, temperature=1.0):
        """Run batched inference with DeepSpeed."""
        if not hasattr(self, 'engine') or not hasattr(self, 'tokenizer'):
            return []
        
        results = []
        
        try:
            # Process prompts in batches of batch_size
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    batch_prompts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=512  # Limit input length
                )
                
                # Move inputs to the same device as the model
                device = next(self.engine.parameters()).device
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                # Generate with DeepSpeed
                with torch.no_grad():
                    try:
                        # Use the generate method
                        outputs = self.engine.generate(
                            inputs=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_tokens,
                            do_sample=(temperature > 0),
                            temperature=temperature,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                        
                        # Decode the outputs
                        for j, output in enumerate(outputs):
                            # Only keep the newly generated tokens
                            input_length = len(input_ids[j])
                            new_tokens = output[input_length:]
                            decoded_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                            results.append({
                                "prompt": batch_prompts[j],
                                "text": decoded_output,
                                "input_tokens": input_length,
                                "output_tokens": len(new_tokens)
                            })
                            
                    except Exception as e:
                        print(f"Generate method error: {e}")
                        try:
                            # Fall back to forward pass
                            outputs = self.engine(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_tokens,
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id
                            )
                            
                            # Process output logits
                            if isinstance(outputs, dict) and 'logits' in outputs:
                                logits = outputs['logits']
                                predicted_tokens = torch.argmax(logits[:, -1, :], dim=-1)
                                for j, tokens in enumerate(predicted_tokens):
                                    decoded_output = self.tokenizer.decode(tokens, skip_special_tokens=True)
                                    results.append({
                                        "prompt": batch_prompts[j],
                                        "text": decoded_output,
                                        "input_tokens": len(input_ids[j]),
                                        "output_tokens": len(tokens)
                                    })
                            else:
                                print(f"Unexpected output format: {type(outputs)}")
                                
                        except Exception as fallback_error:
                            print(f"Forward method error: {fallback_error}")
                            for j in range(len(batch_prompts)):
                                results.append({
                                    "prompt": batch_prompts[j],
                                    "text": "",
                                    "input_tokens": 0,
                                    "output_tokens": 0
                                })
                
        except Exception as e:
            print(f"Critical error in run_inference: {e}")
            import traceback
            traceback.print_exc()
        
        return results

    def estimate_tokens(self, outputs):
        """Estimate token count from outputs."""
        total_tokens = 0
        for output in outputs:
            if isinstance(output, dict):
                total_tokens += output.get('output_tokens', 0)
            elif hasattr(output, 'text'):
                total_tokens += len(self.tokenizer.encode(output.text))
            else:
                text = str(output)
                total_tokens += len(text.split()) * 1.3 
        return int(total_tokens)

    def run_benchmark(self, prompts, num_samples, batch_size, max_tokens):
        """Run multiple inference passes for proper benchmarking."""
        if not hasattr(self, 'engine') or not hasattr(self, 'tokenizer'):
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
            
            print(f"  Sample {i+current_batch_size}/{num_samples}: Generated {tokens_so_far} tokens so far ({elapsed:.2f}s elapsed)")
        
        end_time = time.time()
        return all_outputs, start_time, end_time

    def print_setup_instructions(self):
        """Print instructions for setting up models for DeepSpeed."""
        print("\nModel Setup Instructions for DeepSpeed:")
        print("="*50)
        print("DeepSpeed can be used with Hugging Face Transformers models.")
        print("Here's how to prepare the models you mentioned:")
        
        print("\n1. TinyLlama:")
        print("   - HuggingFace ID: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'")
        print("   - Download: No action needed, it will download automatically")
        
        print("\n2. Llama 2:")
        print("   - HuggingFace ID: 'meta-llama/Llama-2-7b-chat-hf'")
        print("   - Note: Requires HuggingFace account with Meta license acceptance")
        print("   - Download: Can pre-download with 'huggingface-cli download meta-llama/Llama-2-7b-chat-hf'")

    def __del__(self):
        """Cleanup DeepSpeed resources."""
        try:
            if hasattr(self, 'engine'):
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Error during DeepSpeed cleanup: {e}")