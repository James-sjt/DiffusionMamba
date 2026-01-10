# 🚀 DiffusionMamba

## 🧠 Overview

**DiffusionMamba** is a dual-stage medical image analysis framework composed of:

- 🎨 **Diffusion-based Image Enhancement Module**  
  Enhances low-quality MRI images by improving contrast and luminance using a latent diffusion model.

- 🧩 **Mamba-based Segmentation Module**  
  Performs accurate and robust segmentation on enhanced images using a Mamba-based architecture.

Extensive experiments demonstrate that:
- The diffusion-based enhancement module outperforms GAN-based methods in terms of generated image quality.
- The Mamba-based segmentation module achieves superior segmentation accuracy compared with multiple classical segmentation models.

This repository provides a complete end-to-end pipeline from image enhancement to downstream segmentation.

---

## 📁 Repository Structure

```text
DiffusionMamba/
├── Diffusion_based_Image_Enhancement-master/
│   ├── VAE.py
│   ├── VAEInference.py
│   ├── StatsComputer.py
│   ├── latentDiffusion.py
│   ├── inferenceLDM.py
│   ├── sampleEnhancedImg.py
│   ├── extractDataset.py
│   └── ...
├── Mamba_based_Segmentation/
│   └── ...
└── README.md
```

## ⚙️ Installation

Clone the repository to your local machine:

``` bash
git clone https://github.com/James-sjt/DiffusionMamba
cd DiffusionMamba
```
