"""
Energy consumption monitoring utilities for LLM inference benchmarking.
Supports monitoring GPU, CPU, DRAM, and total system power consumption.
"""

import os
import threading
import time
import subprocess
import numpy as np

# Try to import pynvml for GPU power monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("pynvml not available, installing...")
    os.system("pip install nvidia-ml-py")
    import pynvml
    NVML_AVAILABLE = True

# RAPL sysfs path for CPU power monitoring
RAPL_PATH = "/sys/class/powercap/intel-rapl"

# Sample prompts for benchmarking
# LONG_PROMPTS = [
#     "Discuss the potential of brain-computer interfaces (BCIs) in transforming human-computer interaction. Explain how BCIs work, current developments, ethical concerns, and possible applications in medicine, communication, and productivity.",
#     "Analyze the evolution and impact of cybersecurity threats in the digital age. Include an overview of major attack vectors, the role of AI in both offensive and defensive cybersecurity, and the global challenges of securing data and infrastructure.",
#     "Speculate on the future of human civilization in space. Discuss technological requirements, ethical considerations, and the potential for colonization on Mars or other celestial bodies.",
#     "Evaluate the implications of large-scale automation on global labor markets. Discuss economic shifts, impacts on developing vs. developed countries, potential policy responses such as universal basic income, and ethical dilemmas around human dignity and purpose.",
#     "Compare and contrast three major philosophical theories of consciousness (e.g., dualism, functionalism, panpsychism). Use thought experiments, scientific evidence, and critique each position's ability to explain subjective experience and qualia.",
#     "Discuss the role of quantum computing in advancing scientific discovery. Explain how quantum computers differ from classical ones, current breakthroughs, limitations, and potential applications in cryptography, material science, and AI.",
#     "Assess the risks and opportunities of using AI-generated content in journalism and academia. Include implications for truth, misinformation, intellectual property, and the shifting role of human expertise.",
#     "Analyze the environmental and economic trade-offs of transitioning to a fully renewable energy grid. Discuss storage challenges, rare earth dependency, geopolitical impacts, and innovations needed in policy and technology.",
#     "Imagine a world where humans can upload their consciousness into digital systems. Explore technical feasibility, continuity of self, legal identity, and societal implications of digital immortality.",
#     "Critically evaluate the promise and perils of CRISPR and gene editing in humans. Discuss the science behind gene editing, current medical uses, bioethical concerns, and the potential for enhancement vs. therapeutic interventions.",
#     "Trace the development of global AI governance initiatives. What principles should guide the international regulation of artificial intelligence? Compare regional approaches (EU, U.S., China) and debate centralized vs. decentralized governance models.",
#     "Discuss how the Internet of Things (IoT) is reshaping modern cities. Address benefits in efficiency and convenience alongside risks in surveillance, cybersecurity, and data ownership.",
#     "Examine the role of digital platforms in shaping political discourse. Analyze filter bubbles, algorithmic amplification, the spread of disinformation, and potential strategies for maintaining a healthy digital public sphere.",
#     "Explore the implications of neural enhancement technologies on education and workforce development. Consider social equity, access, and long-term societal effects.",
#     "Debate the feasibility and ethics of de-extinction efforts through genetic cloning. What responsibilities do scientists have in reintroducing species, and what ecological risks might arise?",
#     "Explain the key principles of systems thinking and how they apply to solving complex global issues such as climate change, public health, or supply chain resilience.",
#     "Evaluate the psychological and societal impacts of long-term isolation in space missions or remote environments. Discuss parallels with lockdown experiences during pandemics.",
#     "Compare the benefits and risks of central bank digital currencies (CBDCs) in modern financial systems. How might they affect privacy, monetary policy, and financial inclusion?",
#     "Assess how AI can be used to enhance creativity in the arts. Include examples in music, visual arts, and literature, and explore debates around originality and authorship.",
#     "Analyze the history and resurgence of interest in psychedelics for mental health treatment. Discuss current research, regulatory hurdles, and cultural shifts.",
#     "Speculate on the evolution of language under the influence of AI translation tools and large language models. How might human communication, cultural identity, and language learning change?",
#     "Debate whether universal internet access should be considered a fundamental human right. Discuss implications for education, political participation, and global development.",
#     "Explore the role of virtual and augmented reality in future education systems. What benefits and drawbacks exist in immersive learning environments?",
#     "Discuss the ethical implications of predictive policing algorithms. Address bias in training data, transparency, and community trust in law enforcement technologies.",
#     "Analyze the future of work in a world dominated by remote and hybrid models. How do these shifts impact productivity, social connection, urban development, and work-life balance?",
#     "Evaluate the role of citizen science in modern research. What are the benefits and limitations of involving the public in data collection and scientific inquiry?",
#     "Compare the advantages and disadvantages of centralized vs. decentralized energy systems. What technologies support each, and how do they differ in resilience and scalability?",
#     "Critically analyze the risks of AI model collapse and information degeneration in over-trained or recursively trained models. How can we safeguard against these risks?",
#     "Discuss the relationship between privacy and personalization in digital services. How can companies balance user experience with ethical data use?",
#     "Examine how blockchain technology could be applied beyond cryptocurrencies, such as in supply chain management, identity verification, and decentralized governance.",
#     "Speculate on how emerging biotechnologies could reshape agricultural practices. What are the potential environmental, economic, and social effects of synthetic biology in farming?",
#     "Explore the role of ethical design in technology development. How can human values be embedded in systems like social media platforms, recommendation engines, or autonomous vehicles?",
#     "Assess the impact of climate change on global migration patterns. Discuss the role of policy, infrastructure, and international cooperation in managing climate-induced displacement."
# ]

try:
    from datasets import load_dataset
    print("Loading Alpaca dataset...")
    alpaca_ds = load_dataset("tatsu-lab/alpaca")
    

    LONG_PROMPTS = []
    
    if "train" in alpaca_ds:
        for item in alpaca_ds["train"]:
            if "instruction" in item and item["instruction"].strip():
                LONG_PROMPTS.append(item["instruction"])
    
    if "validation" in alpaca_ds:
        for item in alpaca_ds["validation"]:
            if "instruction" in item and item["instruction"].strip():
                LONG_PROMPTS.append(item["instruction"])
    
    print(f"Loaded {len(LONG_PROMPTS)} prompts from Alpaca dataset")
    

    # import random
    # if len(LONG_PROMPTS) > 1000:
    #     LONG_PROMPTS = random.sample(LONG_PROMPTS, 1000)
    #     print(f"Randomly sampled 1000 prompts for testing")
        
except (ImportError, Exception) as e:
    print(f"Error loading Alpaca dataset: {e}")
    # Fall back to built-in prompts
    LONG_PROMPTS = [
        "Discuss the potential of brain-computer interfaces (BCIs) in transforming human-computer interaction...",
        "Analyze the evolution and impact of cybersecurity threats in the digital age...",

    ]


class RaplReader:
    """Fast reader for RAPL energy values."""

    def __init__(self):
        """Initialize with available RAPL domains."""
        self.domains = self.get_available_domains()
        if not self.domains:
            print("No RAPL domains found. CPU power monitoring will be disabled.")
            return

        self.energy_paths = {domain: os.path.join(path, "energy_uj")
                             for domain, path in self.domains.items()}
        self.max_energy_paths = {domain: os.path.join(path, "max_energy_range_uj")
                                for domain, path in self.domains.items()}

        # Cache max energy values
        self.max_energy_values = {}
        for domain, path in self.max_energy_paths.items():
            try:
                with open(path, 'r') as f:
                    self.max_energy_values[domain] = int(f.read().strip())
            except (IOError, OSError) as e:
                print(f"Warning: Couldn't read max energy for {domain}: {e}")
                self.max_energy_values[domain] = 2**32  # Fallback value

    def get_available_domains(self):
        """Get all available RAPL domains."""
        domains = {}

        if not os.path.exists(RAPL_PATH):
            print("RAPL sysfs interface not found. CPU power monitoring will be disabled.")
            return domains

        # Find all intel-rapl domains
        try:
            for domain in os.listdir(RAPL_PATH):
                if domain.startswith("intel-rapl:"):
                    domain_path = os.path.join(RAPL_PATH, domain)

                    # Get domain name
                    with open(os.path.join(domain_path, "name"), 'r') as f:
                        name = f.read().strip()

                    domains[name] = domain_path

                    # Check for subdomains
                    for subdomain in os.listdir(domain_path):
                        if subdomain.startswith("intel-rapl:"):
                            subdomain_path = os.path.join(domain_path, subdomain)

                            # Get subdomain name
                            with open(os.path.join(subdomain_path, "name"), 'r') as f:
                                subname = f.read().strip()

                            domains[f"{name}-{subname}"] = subdomain_path
        except Exception as e:
            print(f"Error reading RAPL domains: {e}")

        return domains

    def read_energy_values(self):
        """Read energy values for all domains."""
        result = {}

        for domain, path in self.energy_paths.items():
            try:
                with open(path, 'r') as f:
                    result[domain] = int(f.read().strip())
            except (IOError, OSError) as e:
                result[domain] = None

        return result

    def get_max_energy(self, domain):
        """Get max energy value for a domain."""
        return self.max_energy_values.get(domain, None)

    def calculate_power(self, prev_values, curr_values, time_delta):
        """Calculate power from energy difference."""
        power_values = {}

        for domain in self.domains:
            if prev_values.get(domain) is not None and curr_values.get(domain) is not None:
                # Handle energy counter wraparound
                energy_diff = curr_values[domain] - prev_values[domain]
                if energy_diff < 0:
                    max_range = self.get_max_energy(domain)
                    if max_range:
                        energy_diff += max_range

                # Calculate power in watts (energy in joules / time in seconds)
                if time_delta > 0:
                    power_watts = (energy_diff / 1000000) / time_delta
                    power_values[domain] = power_watts
                else:
                    power_values[domain] = 0
            else:
                power_values[domain] = 0

        return power_values

    def has_dram_domains(self):
        """Check if DRAM domains are available."""
        return any("dram" in domain.lower() for domain in self.domains)

def get_total_power():
    """Get total power consumption using ipmitool."""
    try:
        output = subprocess.check_output(["ipmitool", "dcmi", "power", "reading"],
                                         stderr=subprocess.STDOUT, text=True)
        for line in output.split('\n'):
            if "Instantaneous power reading" in line:
                try:
                    power = float(line.split(':')[1].strip().split()[0])
                    return power
                except (ValueError, IndexError):
                    return None
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

class PowerMonitor:
    """Power monitoring class for benchmarking ML inference."""
    
    def __init__(self):
        """Initialize power monitoring capabilities."""
        # Initialize GPU power monitoring
        self.gpu_available = False
        self.gpu_handles = []
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
                
                print(f"Found {device_count} GPUs")
                for i, handle in enumerate(self.gpu_handles):
                    name = pynvml.nvmlDeviceGetName(handle)
                    print(f"GPU {i}: {name}")
                
                self.gpu_available = True
            except Exception as e:
                print(f"Error initializing NVML: {e}")
                self.gpu_available = False
        
        # Initialize CPU power monitoring
        self.rapl_reader = RaplReader()
        self.cpu_available = bool(self.rapl_reader.domains)
        if self.cpu_available:
            print(f"Found {len(self.rapl_reader.domains)} CPU RAPL domains:")
            for domain in self.rapl_reader.domains:
                print(f"  - {domain}")
            
            # Check specifically for DRAM domains
            self.has_dram = self.rapl_reader.has_dram_domains()
            if self.has_dram:
                print("DRAM power monitoring is available!")
            else:
                print("No DRAM-specific power domains detected.")
        
        # Check total power monitoring
        self.total_power_available = False
        total_power = get_total_power()
        if total_power is not None:
            self.total_power_available = True
            print(f"Total system power monitoring available (current: {total_power}W)")
        else:
            print("Total system power monitoring not available")
    
    def start_monitoring(self):
        """Start power monitoring threads and return control objects."""
        # Power readings arrays
        self.gpu_power_readings = []
        self.cpu_power_readings = []
        self.total_power_readings = []
        
        # Threading control
        self.monitoring_active = True
        self.monitor_lock = threading.Lock()
        
        # CPU monitoring variables
        self.prev_cpu_energy = self.rapl_reader.read_energy_values() if self.cpu_available else {}
        self.last_cpu_read_time = time.time()
        
        # Start power monitoring threads
        self.monitor_threads = []
        
        if self.gpu_available:
            gpu_thread = threading.Thread(target=self._monitor_gpu_power)
            gpu_thread.daemon = True
            gpu_thread.start()
            self.monitor_threads.append(gpu_thread)
        
        if self.cpu_available:
            cpu_thread = threading.Thread(target=self._monitor_cpu_power)
            cpu_thread.daemon = True
            cpu_thread.start()
            self.monitor_threads.append(cpu_thread)
        
        if self.total_power_available:
            total_thread = threading.Thread(target=self._monitor_total_power)
            total_thread.daemon = True
            total_thread.start()
            self.monitor_threads.append(total_thread)
        
        # Allow monitors to collect some initial readings
        time.sleep(2.0)
    
    def stop_monitoring(self):
        """Stop power monitoring threads."""
        self.monitoring_active = False
        for thread in self.monitor_threads:
            thread.join(timeout=1.0)
        
        # Cleanup GPU monitoring if needed
        if self.gpu_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def _monitor_gpu_power(self):
        """GPU power monitoring thread function."""
        while self.monitoring_active:
            try:
                with self.monitor_lock:
                    power_vals = []
                    for handle in self.gpu_handles:
                        try:
                            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                            power_vals.append(power)
                        except:
                            power_vals.append(0)
                    self.gpu_power_readings.append(power_vals)
            except Exception as e:
                pass
            time.sleep(0.1)  # 100ms sampling
    
    def _monitor_cpu_power(self):
        """CPU power monitoring thread function."""
        while self.monitoring_active:
            try:
                with self.monitor_lock:
                    if self.cpu_available:
                        curr_time = time.time()
                        curr_cpu_energy = self.rapl_reader.read_energy_values()
                        
                        # Calculate time delta - ensure it's reasonable
                        time_delta = curr_time - self.last_cpu_read_time
                        if time_delta >= 0.5:  # Only update every 0.5 seconds minimum
                            # Calculate power
                            cpu_power = self.rapl_reader.calculate_power(
                                self.prev_cpu_energy, curr_cpu_energy, time_delta
                            )
                            
                            # Store readings
                            self.cpu_power_readings.append(cpu_power)
                            
                            # Update for next iteration
                            self.prev_cpu_energy = curr_cpu_energy
                            self.last_cpu_read_time = curr_time
            except Exception as e:
                pass
            time.sleep(0.5)  # Slow down to 500ms for more stable readings
    
    def _monitor_total_power(self):
        """Total system power monitoring thread function."""
        while self.monitoring_active:
            try:
                with self.monitor_lock:
                    if self.total_power_available:
                        power = get_total_power()
                        if power is not None:
                            self.total_power_readings.append(power)
            except Exception as e:
                pass
            time.sleep(1.0)  # 1 second sampling for IPMI (slower and less frequent)
    
    def calculate_metrics(self, duration, total_output_tokens, num_responses=None):
        """Calculate power and energy metrics from collected readings."""
        # Filter power readings to only include those during the inference period
        def filter_readings_by_time(readings):
            """Filter readings to only include those during inference (with small buffer)"""
            if not readings:
                return readings
            buffer_size = max(1, int(len(readings) * 0.1))  # At least 1, up to 10% of readings
            return readings[buffer_size:-buffer_size] if len(readings) > buffer_size*2 else readings
        
        # Process GPU power readings
        if self.gpu_power_readings:
            # Filter to inference period
            gpu_power_readings = filter_readings_by_time(self.gpu_power_readings)
            gpu_power_array = np.array(gpu_power_readings)
            gpu_avg_power = np.mean(gpu_power_array, axis=0)
            gpu_total_avg_power = np.sum(gpu_avg_power)
            gpu_energy_joules = gpu_total_avg_power * duration
        else:
            gpu_avg_power = []
            gpu_total_avg_power = gpu_energy_joules = 0
        
        # Process CPU power readings
        cpu_domains = list(self.rapl_reader.domains.keys()) if self.cpu_available else []
        if self.cpu_power_readings:
            # Filter to inference period
            cpu_power_readings = filter_readings_by_time(self.cpu_power_readings)
            
            # Convert list of dicts to dict of lists
            cpu_power_by_domain = {domain: [] for domain in cpu_domains}
            for reading in cpu_power_readings:
                for domain, power in reading.items():
                    # Sanity check for unrealistic values (> 500W per domain is likely an error)
                    if power <= 500:
                        cpu_power_by_domain[domain].append(power)
            
            # Calculate averages for each domain
            cpu_avg_power = {}
            cpu_total_avg_power = 0
            for domain, powers in cpu_power_by_domain.items():
                if powers:
                    # Remove outliers - anything more than 3 std devs from mean
                    if len(powers) > 5:
                        powers_array = np.array(powers)
                        mean = np.mean(powers_array)
                        std = np.std(powers_array)
                        filtered_powers = powers_array[(powers_array >= mean - 3*std) &
                                                      (powers_array <= mean + 3*std)]
                        domain_power = float(np.mean(filtered_powers) if len(filtered_powers) > 0 else mean)
                    else:
                        domain_power = float(np.mean(powers))
                    cpu_avg_power[domain] = domain_power
                    if not "dram" in domain.lower():
                        cpu_total_avg_power += domain_power
                else:
                    cpu_avg_power[domain] = 0
            
            cpu_energy_joules = cpu_total_avg_power * duration
        else:
            cpu_avg_power = {domain: 0 for domain in cpu_domains}
            cpu_total_avg_power = cpu_energy_joules = 0
        
        # Extract DRAM specific power if available
        dram_power_watts = 0
        dram_domains = []
        for domain in cpu_avg_power:
            if "dram" in domain.lower():
                dram_power_watts += cpu_avg_power[domain]
                dram_domains.append(domain)
        
        # Process total power readings
        if self.total_power_readings:
            # Filter to inference period
            total_power_readings = filter_readings_by_time(self.total_power_readings)
            
            # Remove outliers
            if len(total_power_readings) > 5:
                total_power_array = np.array(total_power_readings)
                mean = np.mean(total_power_array)
                std = np.std(total_power_array)
                filtered_total_power = total_power_array[(total_power_array >= mean - 3*std) &
                                                       (total_power_array <= mean + 3*std)]
                total_avg_power = float(np.mean(filtered_total_power) if len(filtered_total_power) > 0 else mean)
            else:
                total_avg_power = float(np.mean(total_power_readings))
            
            total_energy_joules = total_avg_power * duration
        else:
            total_avg_power = total_energy_joules = 0
        
        # Total energy (accounting for all monitored sources)
        if total_energy_joules > 0:
            energy_per_token = total_energy_joules / total_output_tokens if total_output_tokens > 0 else 0
        else:
            # If total system power not available, use GPU + CPU
            energy_per_token = (gpu_energy_joules + cpu_energy_joules) / total_output_tokens if total_output_tokens > 0 else 0
        
        # Assemble results
        result = {
            "duration": duration,
            "total_output_tokens": total_output_tokens,
            "responses": num_responses if num_responses is not None else len(self.cpu_power_readings),
            "gpu_avg_power": gpu_total_avg_power,
            "cpu_avg_power": cpu_total_avg_power,  # Now this is a float
            "dram_avg_power": dram_power_watts,
            "total_avg_power": total_avg_power,
            "gpu_energy": gpu_energy_joules,
            "cpu_energy": cpu_energy_joules,
            "dram_energy": dram_power_watts * duration,
            "total_energy": total_energy_joules if total_energy_joules > 0 else (gpu_energy_joules + cpu_energy_joules),
            "energy_per_token": energy_per_token,
            "gpu_energy_per_second": gpu_energy_joules / duration if duration > 0 else 0,
            "cpu_energy_per_second": cpu_energy_joules / duration if duration > 0 else 0,
            "dram_energy_per_second": dram_power_watts * duration / duration if duration > 0 else 0,
            "total_energy_per_second": total_energy_joules / duration if duration > 0 else 0,
            "gpu_energy_per_token": gpu_energy_joules / total_output_tokens if total_output_tokens > 0 else 0,
            "cpu_energy_per_token": cpu_energy_joules / total_output_tokens if total_output_tokens > 0 else 0,
            "dram_energy_per_token": dram_power_watts * duration / total_output_tokens if total_output_tokens > 0 else 0,
            "total_energy_per_token": total_energy_joules / total_output_tokens if total_output_tokens > 0 else 0,
            "gpu_energy_per_response": gpu_energy_joules / len(self.cpu_power_readings) if len(self.cpu_power_readings) > 0 else 0,
            "cpu_energy_per_response": cpu_energy_joules / len(self.cpu_power_readings) if len(self.cpu_power_readings) > 0 else 0,
            "dram_energy_per_response": dram_power_watts * duration / len(self.cpu_power_readings) if len(self.cpu_power_readings) > 0 else 0,
            "total_energy_per_response": total_energy_joules / len(self.cpu_power_readings) if len(self.cpu_power_readings) > 0 else 0
        }
        
        # Add individual GPU power data
        if self.gpu_available and len(gpu_avg_power) > 0:
            for i, power in enumerate(gpu_avg_power):
                result[f"gpu{i}_power_watts"] = float(power)
        
        # Add individual CPU domain power data
        if self.cpu_available:
            for domain, power in cpu_avg_power.items():
                result[f"{domain}_power_watts"] = power
        
        # Save domain info for printing
        result["dram_domains"] = dram_domains
        result["cpu_domain_power"] = cpu_avg_power  # Store domain-specific power values separately
        
        return result
    
    def print_metrics(self, metrics):
        """Print energy and power metrics in a human-readable format and save to file."""
        # Prepare the output string
        output = []
        output.append("\nPower and Energy Metrics:")
        output.append("="*50)
        
        output.append("\nBasic Information:")
        output.append(f"  Runtime: {metrics['duration']:.2f}s")
        output.append(f"  Generated tokens: {metrics['total_output_tokens']}")
        output.append(f"  Number of responses: {metrics['responses']}")
        
        output.append("\nAverage Power:")
        output.append(f"  GPU Power: {metrics['gpu_avg_power']:.2f}W")
        output.append(f"  CPU Power: {metrics['cpu_avg_power']:.2f}W")
        output.append(f"  DRAM Power: {metrics['dram_avg_power']:.2f}W")
        output.append(f"  Total Power: {metrics['total_avg_power']:.2f}W")
        
        output.append("\nTotal Energy Consumption:")
        output.append(f"  GPU Energy: {metrics['gpu_energy']:.2f}J")
        output.append(f"  CPU Energy: {metrics['cpu_energy']:.2f}J")
        output.append(f"  DRAM Energy: {metrics['dram_energy']:.2f}J")
        output.append(f"  Total Energy: {metrics['total_energy']:.2f}J")
        
        output.append("\nEnergy per Second:")
        output.append(f"  GPU Energy/s: {metrics['gpu_energy_per_second']:.2f}J/s")
        output.append(f"  CPU Energy/s: {metrics['cpu_energy_per_second']:.2f}J/s")
        output.append(f"  DRAM Energy/s: {metrics['dram_energy_per_second']:.2f}J/s")
        output.append(f"  Total Energy/s: {metrics['total_energy_per_second']:.2f}J/s")
        
        output.append("\nEnergy per Token:")
        output.append(f"  GPU Energy/token: {metrics['gpu_energy_per_token']*1000:.3f}mJ/token")
        output.append(f"  CPU Energy/token: {metrics['cpu_energy_per_token']*1000:.3f}mJ/token")
        output.append(f"  DRAM Energy/token: {metrics['dram_energy_per_token']*1000:.3f}mJ/token")
        output.append(f"  Total Energy/token: {metrics['total_energy_per_token']*1000:.3f}mJ/token")
        
        output.append("\nEnergy per Response:")
        output.append(f"  GPU Energy/response: {metrics['gpu_energy_per_response']:.3f}J/response")
        output.append(f"  CPU Energy/response: {metrics['cpu_energy_per_response']:.3f}J/response")
        output.append(f"  DRAM Energy/response: {metrics['dram_energy_per_response']:.3f}J/response")
        output.append(f"  Total Energy/response: {metrics['total_energy_per_response']:.3f}J/response")
        
        output.append("="*50)
        
        # Print to console
        print('\n'.join(output))
        
        # Add formatted metrics for file saving
        metrics.update({
            # Energy per second metrics (already in metrics)
            "gpu_energy_per_second_formatted": f"{metrics['gpu_energy_per_second']:.2f}J/s",
            "cpu_energy_per_second_formatted": f"{metrics['cpu_energy_per_second']:.2f}J/s",
            "dram_energy_per_second_formatted": f"{metrics['dram_energy_per_second']:.2f}J/s",
            "total_energy_per_second_formatted": f"{metrics['total_energy_per_second']:.2f}J/s",
            
            # Energy per token metrics in mJ
            "gpu_energy_per_token_mj": metrics['gpu_energy_per_token']*1000,
            "cpu_energy_per_token_mj": metrics['cpu_energy_per_token']*1000,
            "dram_energy_per_token_mj": metrics['dram_energy_per_token']*1000,
            "total_energy_per_token_mj": metrics['total_energy_per_token']*1000,
            
            # Formatted power metrics
            "gpu_power_formatted": f"{metrics['gpu_avg_power']:.2f}W",
            "cpu_power_formatted": f"{metrics['cpu_avg_power']:.2f}W",
            "dram_power_formatted": f"{metrics['dram_avg_power']:.2f}W",
            "total_power_formatted": f"{metrics['total_avg_power']:.2f}W",
            
            # Formatted energy metrics
            "gpu_energy_formatted": f"{metrics['gpu_energy']:.2f}J",
            "cpu_energy_formatted": f"{metrics['cpu_energy']:.2f}J",
            "dram_energy_formatted": f"{metrics['dram_energy']:.2f}J",
            "total_energy_formatted": f"{metrics['total_energy']:.2f}J",
            
            # Formatted energy per response metrics
            "gpu_energy_per_response_formatted": f"{metrics['gpu_energy_per_response']:.3f}J/response",
            "cpu_energy_per_response_formatted": f"{metrics['cpu_energy_per_response']:.3f}J/response",
            "dram_energy_per_response_formatted": f"{metrics['dram_energy_per_response']:.3f}J/response",
            "total_energy_per_response_formatted": f"{metrics['total_energy_per_response']:.3f}J/response",
            
            # Human readable output
            "formatted_output": '\n'.join(output)
        })
        
        return metrics