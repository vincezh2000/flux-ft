
This repository provides inference scripts for [Flux model](https://github.com/black-forest-labs/flux) by Black Forest Labs. 



## Requirements
1. Python >= 3.10
2. PyTorch >= 2.1
3. HuggingFace CLI is required to download our models: ```huggingface-cli login```
# Installation Guide
1. Clone our repo:
```bash
git clone https://github.com/XLabs-AI/x-flux.git
```
2. Create new virtual environment:
```bash
conda create -n xflux -n python==3.11
conda activate xflux
```
3. Install our dependencies by running the following command:
```bash
pip install -r requirements.txt
```

## Inference

To use our inference feature, run main.py like the example below.
If you don't have model weights ready, the script will automatically download for you. (Maybe slow for mainland China user)
If you want to use your local weights, 
1\ Download flux1-dev.safetensors and ae.safetensors form the hf or its mirror.
2\ Download clip-vit-large-patch14 and t5-v1_1-xxl.

After you download all of this, setting "--local_weights" will enable local weights. 
Then setting the parameters relates to model weights.

### Example for LoRA Inference

```bash
python3 main.py \
 --use_lora --lora_weight 0.7 \
 --width 1024 --height 1024 \
 --local_weights \
 --lora_local_path "/data/Flux_train_20.3/output_refined_exhibition/flux-test-loraqkv-000005.safetensors"\
 --guidance 4 \
 --prompt "The image
```



## Accelerate Configuration Example

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 2
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

```
## Models Licence

Our models fall under the [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) <br/> Our training and infer scripts under the Apache 2 License
