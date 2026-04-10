# Z-LAN: Low-Rank Adaptation for Multilingual Text Rendering in Diffusion Models

[![Paper](https://img.shields.io/badge/Paper-Pattern%20Recognition%20Letters-blue)](https://www.sciencedirect.com/journal/pattern-recognition-letters)
[![Models](https://img.shields.io/badge/Models-ModelScope-green)](https://modelscope.cn/models/ccArtermices)

This repository contains the official implementation of **Z-LAN**, a parameter-efficient fine-tuning framework for multilingual text rendering in diffusion models using Low-Rank Adaptation (LoRA).

## 🔥 News

- **2026-04**: Paper accepted to Pattern Recognition Letters
- **2026-03**: Released all 9 LoRA adapters on ModelScope
- **2026-02**: Initial release

## 📋 Overview

Z-LAN enables accurate multilingual text rendering in diffusion models through language-specific LoRA adapters trained on minimal synthetic datasets (fewer than 100 images per language). The framework achieves:

- **94.9%** average text rendering accuracy across 9 languages
- **+33.1 pp** average improvement over baseline
- **6/9** language adapters achieving 100% accuracy
- **87.5%** reduction in text omission errors

## 🌍 Supported Languages

| Language | Script Family | LoRA Model | Accuracy |
|----------|--------------|------------|----------|
| Armenian | Armenian | [ccArtermices/zlan-Armenian](https://modelscope.cn/models/ccArtermices/zlan-Armenian) | 87.5% |
| Georgian | Georgian | [ccArtermices/zlan-Georgian2](https://modelscope.cn/models/ccArtermices/zlan-Georgian2) | 100% |
| Hebrew | Abjad | [ccArtermices/zlan-Hebrew](https://modelscope.cn/models/ccArtermices/zlan-Hebrew) | 66.7% |
| Myanmar | Brahmic | [ccArtermices/zlan-Myanmar](https://modelscope.cn/models/ccArtermices/zlan-Myanmar) | 100% |
| Persian | Arabic | [ccArtermices/zlan-Persian](https://modelscope.cn/models/ccArtermices/zlan-Persian) | 100% |
| Russian | Cyrillic | [ccArtermices/zlan-Russian](https://modelscope.cn/models/ccArtermices/zlan-Russian) | 100% |
| Tibetan | Brahmic | [ccArtermices/zlan-Tibetan](https://modelscope.cn/models/ccArtermices/zlan-Tibetan) | 100% |
| Urdu | Arabic | [ccArtermices/zlan-Urdu](https://modelscope.cn/models/ccArtermices/zlan-Urdu) | 100% |
| Vietnamese | Latin | [ccArtermices/zlan-Vietnamese](https://modelscope.cn/models/ccArtermices/zlan-Vietnamese) | 100% |

## 🚀 Quick Start

### Installation

```bash
pip install torch diffusers transformers accelerate safetensors
```

### Usage with ModelScope API

```python
import requests
import base64
from PIL import Image
from io import BytesIO

API_URL = "https://api-inference.modelscope.cn/v1/images/generations"
API_KEY = "your-api-key"

def generate_image(prompt, lora_id):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Tongyi-MAI/Z-Image",
        "prompt": prompt,
        "loras": lora_id,
        "num_inference_steps": 50,
        "guidance_scale": 7.5
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Example: Generate image with Armenian text
result = generate_image(
    prompt="zlan-Armenian, a red apple with Armenian text 'Խնձոր' below",
    lora_id="ccArtermices/zlan-Armenian"
)
```

### Usage with Local Diffusers

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA adapter
pipe.load_lora_weights("ccArtermices/zlan-Armenian")

# Generate image
image = pipe(
    prompt="zlan-Armenian, a red apple with Armenian text 'Խնձոր' below",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("output.png")
```

## 📁 Repository Structure

```
zlan/
├── data/
│   ├── prompts-multiLan.txt      # Evaluation prompts
│   └── fonts/                     # Unicode fonts for synthetic data
├── scripts/
│   ├── generate_synthetic_images.py   # Synthetic data generation
│   ├── train_lora.py                   # LoRA training script
│   └── verify_images_cloud.py          # MLLM-based verification
├── evaluation/
│   ├── lora_evaluation.py         # Main evaluation pipeline
│   └── analyze_evaluation.py      # Analysis and visualization
├── figures/                        # Paper figures
└── README.md
```

## 🔧 Training Your Own LoRA Adapter

### Step 1: Prepare Synthetic Data

```python
from scripts.generate_synthetic_images import generate_synthetic_dataset

# Generate synthetic training images
generate_synthetic_dataset(
    language="Armenian",
    vocabulary=["Խնձոր", "Կատու", "Գիրք", ...],  # Target words
    font_path="fonts/NotoSansArmenian-Regular.ttf",
    output_dir="data/synthetic/Armenian",
    num_images=100
)
```

### Step 2: Train LoRA Adapter

```python
from scripts.train_lora import train_lora_adapter

train_lora_adapter(
    base_model="Tongyi-MAI/Z-Image",
    train_data_dir="data/synthetic/Armenian",
    output_dir="lora-weights/Armenian",
    rank=16,
    alpha=8,
    learning_rate=1e-4,
    num_epochs=100,
    batch_size=4
)
```

### Step 3: Evaluate

```python
from evaluation.lora_evaluation import evaluate_lora

results = evaluate_lora(
    lora_path="lora-weights/Armenian",
    prompts_file="data/prompts-multiLan.txt",
    output_dir="evaluation/results"
)
```

## 📊 Results

### Main Results

| Model | Average Accuracy | Improvement |
|-------|------------------|-------------|
| Baseline (Z-Image) | 61.9% | - |
| Z-LAN (LoRA) | 94.9% | +33.1 pp |

### Error Reduction

| Error Type | Baseline | Z-LAN | Reduction |
|------------|----------|-------|-----------|
| No Text | 24 | 3 | 87.5% |
| Wrong Text | 36 | 2 | 94.4% |
| Extra Text | 34 | 3 | 91.2% |
| Partial Text | 15 | 3 | 80.0% |

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{zlan2026,
  title={Z-LAN: Low-Rank Adaptation for Multilingual Text Rendering in Diffusion Models},
  author={Author, Name},
  journal={Pattern Recognition Letters},
  year={2026}
}
```

## 📄 License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Base model: [Z-Image](https://modelscope.cn/models/Tongyi-MAI/Z-Image) by ModelScope
- Fonts: [Google Noto Fonts](https://fonts.google.com/noto)
- Verification model: [Qwen3.5-35B-A3B](https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B)

## 📧 Contact

For questions and feedback, please open an issue on GitHub or contact the authors.

---

**Note**: The LoRA adapters are trained on minimal synthetic data and may not generalize to all use cases. We recommend testing on your specific application before deployment.
