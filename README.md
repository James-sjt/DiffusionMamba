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
# 🧪 Stage 1: Diffusion-Based Image Enhancement

Navigate to the image enhancement module:
```bash
cd Diffusion_based_Image_Enhancement-master
```
## 📦 Step 1: Dataset Preparation

Extract and preprocess the dataset:
```bash
python extractDataset.py
```

## 🧬 Step 2: VAE Training (Optional)

Train the Variational Autoencoder (VAE) for latent representation learning:
``` bash
python VAE.py
```
ℹ️ This step can be skipped if pre-trained VAE parameters are used.

Evaluate the VAE performance:
```bash
python VAEInference.py
python StatsComputer.py
```

## 🖼️ Step 3: Latent Diffusion Model (DDPM)

Train the latent diffusion model:
```bash
python latentDiffusion.py
```
⚡ By default, pre-trained parameters are loaded. Training may be skipped if you only need inference.

## 🖼️ Step 4: Image Enhancement Inference

Generate enhanced high-quality MRI samples:
```bash
python inferenceLDM.py
```
Construct the enhanced MRI dataset for the segmentation stage:
```bash
python sampleEnhancedImg.py
```
# 🧠 Stage 2: Mamba-Based Segmentation

Navigate to the segmentation module:
```bash
cd ../Mamba_based_Segmentation
```
