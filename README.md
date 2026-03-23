# TokenPowerBench Benchmark

A comprehensive benchmark suite for measuring energy consumption and performance of large language models across different inference engines and distributed configurations.

## Overview

TokenPowerBench provides two main benchmark categories:

1. **SingleNode Benchmarks** - Support for 4 different inference engines on single-node systems
2. **MultipleNode Benchmarks** - Ray-based distributed inference across multi-node GPU clusters

Both categories include comprehensive energy monitoring, performance analysis, and detailed metrics collection.

## Prerequisites

### Model Downloads from Hugging Face

**IMPORTANT**: Before running any benchmarks, you must download the required models from Hugging Face. Some models require authorization.

#### 1. Hugging Face Authorization

Many models require authorization before download:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face (required for gated models)
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

#### 2. Required Models

Download the following model series to `/data/huggingface/` or your preferred directory:

**Llama 3 Series** (Requires Meta authorization):
```bash
# Apply for access at: https://huggingface.co/meta-llama/Llama-3.1-405B
# Then download:
huggingface-cli download meta-llama/Llama-3.1-405B --local-dir /data/huggingface/meta-llama/Llama-3.1-405B
huggingface-cli download meta-llama/Llama-3.1-70B --local-dir /data/huggingface/meta-llama/Llama-3.1-70B
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir /data/huggingface/meta-llama/Llama-3.1-8B
```

**Qwen Series** (Requires Alibaba authorization):
```bash
# Apply for access at: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir /data/huggingface/Qwen/Qwen2.5-72B-Instruct
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir /data/huggingface/Qwen/Qwen2.5-32B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /data/huggingface/Qwen/Qwen2.5-7B-Instruct
```

**Mistral Series**:
```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir /data/huggingface/mistralai/Mistral-7B-Instruct-v0.2
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 --local-dir /data/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1
```

**Falcon Series** (Requires TII authorization):
```bash
# Apply for access at: https://huggingface.co/tiiuae/falcon-180B
huggingface-cli download tiiuae/falcon-180B --local-dir /data/huggingface/tiiuae/falcon-180B
huggingface-cli download tiiuae/falcon-40B --local-dir /data/huggingface/tiiuae/falcon-40B
huggingface-cli download tiiuae/falcon-7B --local-dir /data/huggingface/tiiuae/falcon-7B
```

**DeepSeek Series**:
```bash
huggingface-cli download deepseek-ai/DeepSeek-V2.5-7B-Instruct --local-dir /data/huggingface/deepseek-ai/DeepSeek-V2.5-7B-Instruct
huggingface-cli download deepseek-ai/DeepSeek-V2.5-32B-Instruct --local-dir /data/huggingface/deepseek-ai/DeepSeek-V2.5-32B-Instruct
```

#### 3. Dataset Downloads

Download required datasets:

**Alpaca Dataset**:
```bash
# Automatically downloaded by the benchmark, but you can pre-download:
python -c "from datasets import load_dataset; load_dataset('tatsu-lab/alpaca', cache_dir='/data/datasets')"
```

**LongBench Dataset**:
```bash
# Automatically downloaded by the benchmark, but you can pre-download:
python -c "from datasets import load_dataset; load_dataset('THUDM/LongBench', 'narrativeqa', cache_dir='/data/datasets')"
```

**Dolly 15K Dataset**:
```bash
python -c "from datasets import load_dataset; load_dataset('databricks/databricks-dolly-15k', cache_dir='/data/datasets')"
```

**HumanEval Dataset**:
```bash
python -c "from datasets import load_dataset; load_dataset('openai/openai_humaneval', cache_dir='/data/datasets')"
```

#### 4. Model Access Applications

**Meta Llama Models**:
- Visit: https://huggingface.co/meta-llama/Llama-3.1-405B
- Click "Request access"
- Fill out the application form
- Wait for approval (usually 2 hours)

**Alibaba Qwen Models**:
- Visit: https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
- Click "Request access"
- Follow the application process

**TII Falcon Models**:
- Visit: https://huggingface.co/tiiuae/falcon-180B
- Click "Request access"
- Complete the application

#### 5. Alternative Download Methods

**Using Git LFS**:
```bash
# Clone with Git LFS for large models
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-405B /data/huggingface/meta-llama/Llama-3.1-405B
```

**Using Hugging Face Hub Python API**:
```python
from huggingface_hub import snapshot_download

# Download with progress bar
snapshot_download(
    repo_id="meta-llama/Llama-3.1-405B",
    local_dir="/data/huggingface/meta-llama/Llama-3.1-405B",
    token="your_token_here"
)
```

### Hardware Requirements

**SingleNode Benchmarks**:
- GPU: NVIDIA A100/H100 (8GB+ VRAM for small models, 80GB+ for large models)
- RAM: 32GB+ for small models, 128GB+ for large models
- Storage: 100GB+ for model storage

**MultipleNode Benchmarks**:
- Multi-node GPU cluster (H100/A100 recommended)
- Minimum 10 GPUs for 405B model testing
- High-speed interconnect (InfiniBand recommended)
- SLURM job scheduler

### Software Requirements

**Core Dependencies**:
```bash
python >= 3.9
torch >= 2.0.0
transformers >= 4.30.0
datasets >= 2.10.0
```

**Engine-Specific Dependencies**:
```bash
# For vLLM engine
pip install vllm>=0.4.0

# For DeepSpeed engine
pip install deepspeed>=0.10.0

# For TensorRT-LLM engine
pip install tensorrt-llm>=0.5.0

# For Ray distributed computing
pip install ray>=2.22.0
```

**Energy Monitoring**:
```bash
pip install nvidia-ml-py psutil
```

## Project Structure

```
TokenPowerBench/
├── SingleNode/                    # Single-node benchmarks with 4 engines
│   ├── llm_benchmark.py          # Main benchmark entry point
│   ├── power_monitor.py          # Energy consumption monitoring
│   └── engines/                  # Four different inference engines
│       ├── vllm_engine.py       # vLLM inference engine
│       ├── deepspeed_engine.py  # DeepSpeed inference engine
│       ├── transformer_engine.py # Hugging Face Transformers engine
│       └── trtllm_engine.py     # TensorRT-LLM inference engine
├── MultipleNode/                  # Distributed benchmarks with Ray
│   ├── benchmark.py              # Main distributed benchmark entry point
│   ├── vllm_engine.py           # Distributed vLLM engine
│   ├── config_manager.py        # Configuration validation
│   ├── dataset_loader.py        # Multi-dataset support
│   ├── timestamp_tracker.py     # Performance tracking
│   ├── power_monitor.py         # Energy monitoring
│   ├── run_ray_head.sh          # Ray head node startup script
│   ├── run_ray_worke.sh         # Ray worker node startup script
│   └── submit_ray_jobs.sh       # Automated job submission
└── results/                      # Benchmark results and logs
```

## SingleNode Benchmarks

### Supported Engines

TokenPowerBench supports 4 different inference engines for single-node benchmarking:

1. **vLLM Engine** - High-performance LLM serving with PagedAttention
2. **DeepSpeed Engine** - Microsoft's distributed training and inference framework
3. **Transformers Engine** - Hugging Face's popular transformers library
4. **TensorRT-LLM Engine** - NVIDIA's optimized TensorRT-based inference

### Usage

#### Basic Single-Engine Benchmark

```bash
cd SingleNode/

# vLLM benchmark
python llm_benchmark.py \
    --engine vllm \
    --models "meta-llama/Llama-3.1-8B" \
    --batch-sizes "256,512" \
    --num-samples 1000 \
    --output-tokens 500

# DeepSpeed benchmark
python llm_benchmark.py \
    --engine deepspeed \
    --models "meta-llama/Llama-3.1-8B" \
    --batch-sizes "128,256" \
    --num-samples 1000 \
    --output-tokens 500

# Transformers benchmark
python llm_benchmark.py \
    --engine transformers \
    --models "meta-llama/Llama-3.1-8B" \
    --batch-sizes "64,128" \
    --num-samples 1000 \
    --output-tokens 500

# TensorRT-LLM benchmark
python llm_benchmark.py \
    --engine tensorrt_llm \
    --models "meta-llama/Llama-3.1-8B" \
    --batch-sizes "256,512" \
    --num-samples 1000 \
    --output-tokens 500
```

#### Multi-Engine Comparison

```bash
# Compare all engines with same model
for engine in vllm deepspeed transformers tensorrt_llm; do
    python llm_benchmark.py \
        --engine $engine \
        --models "meta-llama/Llama-3.1-8B" \
        --batch-sizes "256" \
        --num-samples 1000 \
        --output-tokens 500
done
```

#### Advanced Configuration

```bash
# Custom prompt filtering and energy monitoring
python llm_benchmark.py \
    --engine vllm \
    --models "meta-llama/Llama-3.1-8B,meta-llama/Llama-3.1-70B" \
    --batch-sizes "128,256,512" \
    --num-samples 5000 \
    --output-tokens 1000 \
    --min-length 10 \
    --max-length 200
```

### Output Files

SingleNode benchmarks generate:
- `{engine}_models--{model}_{batch_sizes}_{tokens}_{timestamp}.json` - Detailed results
- Energy consumption metrics (GPU, CPU, DRAM power)
- Performance metrics (throughput, latency, tokens/sec)
- Power efficiency analysis

## MultipleNode Benchmarks

### Ray Framework Setup

MultipleNode benchmarks require Ray cluster setup before running:

#### 1. Environment Setup

```bash
# Install Ray and vLLM
pip install ray>=2.22.0 vllm>=0.4.0

# Additional dependencies
pip install datasets transformers
```

#### 2. Cluster Configuration

Update node IPs in configuration files:
```bash
# Edit head node IP in worker scripts
vim MultipleNode/run_ray_worke.sh
# Change head_ip="10.100.93.6" to your head node IP
```

#### 3. Start Ray Cluster

```bash
cd MultipleNode/

# Start head node
sbatch run_ray_head.sh

# Start worker nodes (in separate terminals)
sbatch run_ray_worke.sh
```

#### 4. Automated Job Submission

```bash
# Submit all Ray cluster jobs automatically
./submit_ray_jobs.sh
```

### Usage

#### Basic Distributed Benchmark

```bash
cd MultipleNode/

# Run distributed benchmark
python benchmark.py \
    --models "Llama-3.1-405B" \
    --datasets "alpaca" \
    --batch-sizes "256" \
    --tensor-parallel "8" \
    --pipeline-parallel "2" \
    --concurrency "1" \
    --num-samples 1000 \
    --max-tokens 512
```

#### Multi-Model Comparison

```bash
# Compare multiple models
python benchmark.py \
    --models "Llama-3.1-405B,Llama-3.1-70B,Llama-3.1-8B" \
    --datasets "alpaca,dolly" \
    --batch-sizes "256,512" \
    --tensor-parallel "4,8" \
    --pipeline-parallel "1,2" \
    --concurrency "1" \
    --num-samples 1000
```

#### Comprehensive Benchmark

```bash
# Full benchmark with all configurations
python benchmark.py \
    --models "Llama-3.1-405B,DeepSeek-R1" \
    --datasets "alpaca,longbench,humaneval" \
    --batch-sizes "128,256,512,1024" \
    --tensor-parallel "4,8,16" \
    --pipeline-parallel "1,2,4" \
    --concurrency "1,2" \
    --num-samples 5000 \
    --max-tokens 1024 \
    --verbose
```

### Supported Datasets

- **Alpaca** - Stanford Alpaca instruction dataset
- **Dolly 15K** - Databricks Dolly instruction dataset  
- **LongBench** - Long context understanding benchmark
- **HumanEval** - Code generation evaluation dataset

### Configuration Options

#### Model Configurations

| Model | Recommended TP | Recommended PP | Memory Usage |
|-------|----------------|----------------|--------------|
| Llama-3.1-405B | 8-16 | 2-4 | ~1.5TB |
| Llama-3.1-70B | 4-8 | 2 | ~600GB |
| Llama-3.1-8B | 1-4 | 1 | ~50GB |
| DeepSeek-R1 | 4-8 | 2 | ~400GB |

#### Batch Size Recommendations

- **Small models (8B)**: 256-1024
- **Medium models (70B)**: 128-512  
- **Large models (405B)**: 64-256

### Output Files

MultipleNode benchmarks generate:
- `vllm_results_{model}_{dataset}_{config}_{timestamp}.json` - Individual test results
- `vllm_benchmark_summary_{timestamp}.json` - Comprehensive summary
- `timestamps_{model}_{dataset}_{config}_{timestamp}.json` - Detailed timing analysis
- Energy consumption logs and performance metrics

## Energy Monitoring

Both SingleNode and MultipleNode benchmarks include comprehensive energy monitoring:

### Metrics Collected
- **GPU Power**: Per-GPU power consumption (Watts)
- **CPU Power**: CPU power consumption via RAPL
- **DRAM Power**: Memory power consumption
- **Total System Power**: Complete system power draw
- **Energy per Token**: Energy efficiency metrics
- **Power Profiles**: Time-series power data

### Energy Analysis

```bash
# Analyze energy consumption
python power_monitor.py --input energy_log.csv --output energy_report.html
```

## Results Analysis

### Performance Metrics
- **Throughput**: Tokens per second
- **Latency**: Time to first token
- **Efficiency**: Tokens per second per GPU
- **Scalability**: Performance vs. GPU count

### Energy Metrics
- **Power per Token**: Watts per generated token
- **Energy Efficiency**: Tokens per kWh
- **Total Energy Cost**: kWh for complete benchmark
- **Power Profile**: Power consumption over time

## Troubleshooting

### Model Download Issues

**Authorization Denied**:
```bash
# Check if you're logged in
huggingface-cli whoami

# Re-login if needed
huggingface-cli login
```

**Model Not Found**:
```bash
# Verify model path exists
ls -la /data/huggingface/meta-llama/Llama-3.1-405B/

# Check model files
ls -la /data/huggingface/meta-llama/Llama-3.1-405B/snapshots/*/
```

**Insufficient Storage**:
```bash
# Check available space
df -h /data/huggingface/

# Clean up old models if needed
rm -rf /data/huggingface/old_models/
```

### SingleNode Issues

**Engine Not Available**
```bash
# Check engine availability
python -c "from engines.vllm_engine import VLLMEngine; print(VLLMEngine().available)"
```

**Memory Errors**
```bash
# Reduce batch size or model size
--batch-sizes "64,128"  # Smaller batches
--models "meta-llama/Llama-3.1-8B"  # Smaller model
```

### MultipleNode Issues

**Ray Connection Issues**
```bash
# Check Ray cluster status
ray status

# Verify node connectivity
ping <head-node-ip>

# Check SLURM jobs
squeue -u $USER
```

**Configuration Validation**
```bash
# Validate cluster configuration
python -c "
from config_manager import ConfigManager
cm = ConfigManager()
cm.print_cluster_info()
cm.print_recommended_configs('Llama-3.1-405B')
"
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-benchmark`)
3. Commit changes (`git commit -am 'Add new benchmark'`)
4. Push to branch (`git push origin feature/new-benchmark`)
5. Create Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Ray Team for distributed computing framework
- vLLM Team for efficient LLM serving
- DeepSpeed Team for distributed training and inference
- Hugging Face for transformers library and datasets
- NVIDIA for TensorRT-LLM and GPU computing infrastructure