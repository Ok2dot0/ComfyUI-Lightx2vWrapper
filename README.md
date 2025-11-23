[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/) [![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-brightgreen)](https://github.com/comfyanonymous/ComfyUI) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-lightx2v-yellow)](https://huggingface.co/lightx2v) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/GacLove/ComfyUI-Lightx2vWrapper)

# ComfyUI-Lightx2vWrapper

[中文版](./README_CN.md) | English

A ComfyUI custom node wrapper for LightX2V, enabling modular video generation with advanced optimization features.

## Features

- **Modular Configuration System**: Separate nodes for each aspect of video generation
- **Text-to-Video (T2V) and Image-to-Video (I2V)**: Support for both generation modes
- **Advanced Optimizations**:
  - TeaCache acceleration (up to 3x speedup)
  - Quantization support (int8, fp8)
  - Pre-quantized checkpoint loading (.safetensors)
  - Memory optimization with CPU offloading
  - Lightweight VAE options
- **Super Resolution**: Upscale video output to 1080p, 2k, or 4k
- **LoRA Support**: Chain multiple LoRA models for customization
- **Multiple Model Support**: wan2.1, wan2.2, hunyuan_video_1.5 architectures

## Installation

1. Clone this repository with submodules into your ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/gaclove/ComfyUI-Lightx2vWrapper.git
```

If you already cloned without submodules, initialize them:

```bash
cd ComfyUI-Lightx2vWrapper
git submodule update --init --recursive
```

2. Install dependencies:

```bash
cd ComfyUI-Lightx2vWrapper
# Install lightx2v submodule dependencies
pip install -r lightx2v/requirements.txt
# Install ComfyUI wrapper dependencies
pip install -r requirements.txt
```

3. Download models and place them in `ComfyUI/models/lightx2v/` directory

## Node Overview

### Configuration Nodes

#### 1. LightX2V Inference Config

Basic inference configuration for video generation.

- **Inputs**: model, task_type, inference_steps, seed, cfg_scale, width, height, video_length, fps
- **Output**: Base configuration object

#### 2. LightX2V TeaCache

Feature caching acceleration configuration.

- **Inputs**: enable, threshold (0.0-1.0), use_ret_steps
- **Output**: TeaCache configuration
- **Note**: Lower threshold = more speedup (0.1 ~2x, 0.2 ~3x)

#### 3. LightX2V Quantization

Model quantization settings for memory efficiency.

- **Inputs**: dit_precision, t5_precision, clip_precision, backend, sensitive_layers_precision
- **Output**: Quantization configuration
- **Backends**: Auto-detected (vllm, sgl, q8f)

#### 4. LightX2V Memory Optimization

Memory management strategies.

- **Inputs**: optimization_level, attention_type, enable_rotary_chunking, cpu_offload, unload_after_generate
- **Output**: Memory optimization configuration

#### 5. LightX2V Lightweight VAE

VAE optimization options.

- **Inputs**: use_tiny_vae, use_tiling_vae
- **Output**: VAE configuration

#### 6. LightX2V LoRA Loader

Load and chain LoRA models.

- **Inputs**: lora_name, strength (0.0-2.0), lora_chain (optional)
- **Output**: LoRA chain configuration

#### 7. LightX2V Super Resolution

Super resolution configuration for upscaling video output.

- **Inputs**: sr_model_path, sr_version (1080p/2k/4k), flow_shift, guidance_scale, num_inference_steps
- **Output**: SR configuration
- **Note**: Requires super resolution model checkpoint

### Combination Node

#### 8. LightX2V Config Combiner / Config Combiner V2

Combines all configuration modules into a single configuration.

- **Inputs**: All configuration types (optional), including SR configuration
- **Output**: Combined configuration object (V1) or prepared configuration (V2)
- **Note**: V2 also handles image/audio/prompt preparation

### Inference Node

#### 9. LightX2V Modular Inference / Modular Inference V2

Main inference node for video generation.

- **Inputs**: combined_config (V1) or prepared_config (V2), prompt, negative_prompt, image (optional), audio (optional)
- **Outputs**: Generated video frames and audio
- **Note**: V2 accepts prepared_config from Config Combiner V2

## Usage Examples

### Basic T2V Workflow

1. Create LightX2V Inference Config (task_type: "t2v")
2. Use LightX2V Config Combiner
3. Connect to LightX2V Modular Inference with text prompt
4. Save video output

### I2V with Optimizations

1. Load input image
2. Create LightX2V Inference Config (task_type: "i2v")
3. Add LightX2V TeaCache (threshold: 0.26)
4. Add LightX2V Memory Optimization
5. Combine configs with LightX2V Config Combiner
6. Run LightX2V Modular Inference

### With LoRA

1. Create base configuration
2. Load LoRA with LightX2V LoRA Loader
3. Chain multiple LoRAs if needed
4. Combine all configs
5. Run inference

## Model Directory Structure

Download models from: <https://huggingface.co/lightx2v>

Models should be placed in:

```txt
ComfyUI/models/lightx2v/
├── Wan2.1-I2V-14B-720P-xxx/     # Wan2.1 model checkpoints
├── Wan2.1-I2V-14B-480P-xxx/     # Wan2.1 model checkpoints
├── hunyuanvideo-1.5/            # HunyuanVideo-1.5 model directory
│   ├── transformer/             # Contains transformer variants
│   │   ├── 480p_t2v/           # 480p text-to-video variant
│   │   ├── 480p_i2v/           # 480p image-to-video variant
│   │   ├── 720p_t2v/           # 720p text-to-video variant
│   │   └── 720p_i2v/           # 720p image-to-video variant
│   ├── text_encoder/           # Text encoder models
│   ├── vae/                    # VAE models
│   ├── quantized/              # Pre-quantized checkpoint files (optional)
│   │   ├── hy15_720p_i2v_fp8_e4m3_lightx2v.safetensors
│   │   └── hy15_720p_i2v_cfg_distilled_fp8_e4m3_lightx2v.safetensors
│   └── sr/                     # Super resolution models (optional)
│       └── hy15_1080p_sr_cfg_distiled_fp8_e4m3_lightx2v.safetensors
├── loras/                       # LoRA models
```

### HunyuanVideo-1.5 Model Setup

For HunyuanVideo-1.5 models (hy15), you need to:

1. Place the model folder in `ComfyUI/models/lightx2v/` (e.g., `hunyuanvideo-1.5/`)
2. The folder must contain a `transformer/` subdirectory with model variants
3. In the LightX2V Inference Config node:
   - Set **model_cls** to `hunyuan_video_1.5`
   - Select your model folder name in **model_name** (e.g., `hunyuanvideo-1.5`)
   - Set **transformer_model_name** to specify which variant to use:
     - `480p_t2v` for 480p text-to-video
     - `480p_i2v` for 480p image-to-video
     - `720p_t2v` for 720p text-to-video
     - `720p_i2v` for 720p image-to-video

### Using Pre-Quantized Checkpoints

For HunyuanVideo-1.5 models, you can use pre-quantized checkpoint files for faster loading and reduced memory usage:

1. Place your quantized checkpoint files (`.safetensors`) in `ComfyUI/models/lightx2v/hunyuanvideo-1.5/quantized/`
2. In the LightX2V Inference Config node:
   - Enable **use_quantized_checkpoint**
   - Set **dit_quantized_ckpt** to your checkpoint path (e.g., `ComfyUI/models/lightx2v/hunyuanvideo-1.5/quantized/hy15_720p_i2v_fp8_e4m3_lightx2v.safetensors`)
   - Optionally set **text_encoder_quantized_ckpt** for text encoder checkpoint
   - Leave fields empty to auto-detect based on model configuration

**Note**: Pre-quantized checkpoints are already quantized, so you don't need to enable quantization separately.

### Using Super Resolution

To upscale your generated videos:

1. Place your SR model checkpoint in `ComfyUI/models/lightx2v/hunyuanvideo-1.5/sr/`
2. Add a **LightX2V Super Resolution** node to your workflow
3. Set the **sr_model_path** to your SR model (e.g., `ComfyUI/models/lightx2v/hunyuanvideo-1.5/sr/hy15_1080p_sr_cfg_distiled_fp8_e4m3_lightx2v.safetensors`)
4. Choose your target resolution: `1080p`, `2k`, or `4k`
5. Connect the SR config output to **Config Combiner V2**
6. Run inference as normal - the output will be upscaled

**SR Parameters:**
- **flow_shift**: Controls the denoising flow (default: 7.0)
- **guidance_scale**: Guidance strength for SR (default: 1.0)
- **num_inference_steps**: Number of steps for SR (default: 4)

## Tips

- Start with default settings and adjust based on your hardware
- Use TeaCache with threshold 0.1-0.2 for significant speedup
- Enable memory optimization if running on limited VRAM
- Quantization can reduce memory usage but may affect quality
- Chain multiple LoRAs for complex style combinations

## Troubleshooting

- **Out of Memory**: Enable memory optimization or use quantization
- **Slow Generation**: Enable TeaCache or reduce inference steps
- **Model Not Found**: Check model paths in `ComfyUI/models/lightx2v/`

## Example Workflows

Example workflow JSON files are provided in the `examples/` directory:

- `wan_i2v.json` - Basic image-to-video
- `wan_i2v_with_distill_lora.json` - I2V with distillation LoRA
- `wan_t2v_with_distill_lora.json` - T2V with distillation LoRA

## Contributing Guidelines

We welcome community contributions! Before submitting code, please ensure you follow these steps:

### Install Development Dependencies

```bash
pip install ruff pre-commit
```

### Code Quality Check

Before committing code, run the following command:

```bash
pre-commit run --all-files
```

This will automatically check code formatting, syntax errors, and other code quality issues.

### Contribution Process

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request
