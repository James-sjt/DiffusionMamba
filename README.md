\documentclass[11pt]{article}

\usepackage[a4paper,margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{enumitem}
\usepackage{setspace}

\setstretch{1.1}
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    urlcolor=blue
}

\title{\textbf{DiffusionMamba}}
\author{Jiatong Si}
\date{}

\begin{document}
\maketitle

\section{Overview}

\textbf{DiffusionMamba} is a dual-stage medical image analysis framework composed of:
\begin{itemize}[leftmargin=1.5em]
    \item a \textbf{diffusion-based image enhancement module} for improving contrast and luminance of low-quality MRI images, and
    \item a \textbf{Mamba-based segmentation module} for accurate and robust delineation of regions of interest.
\end{itemize}

Experimental results demonstrate that the proposed diffusion-based enhancement module outperforms GAN-based approaches in terms of generated image quality, while the Mamba-based segmentation module achieves superior segmentation accuracy compared with multiple classical segmentation models.

This repository provides the complete pipeline from image enhancement to downstream segmentation.

\section{Repository Structure}

\begin{verbatim}
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
└── README.tex
\end{verbatim}

\section{Installation}

Clone the repository to your local machine:
\begin{lstlisting}[language=bash]
git clone https://github.com/James-sjt/DiffusionMamba
cd DiffusionMamba
\end{lstlisting}

\section{Stage 1: Diffusion-Based Image Enhancement}

Navigate to the image enhancement module:
\begin{lstlisting}[language=bash]
cd Diffusion_based_Image_Enhancement-master
\end{lstlisting}

\subsection{Dataset Preparation}

Extract and preprocess the dataset:
\begin{lstlisting}[language=bash]
python extractDataset.py
\end{lstlisting}

\subsection{VAE Training (Optional)}

Train the Variational Autoencoder for latent feature learning:
\begin{lstlisting}[language=bash]
python VAE.py
\end{lstlisting}

This step can be skipped if pre-trained VAE parameters are used.

Evaluate the VAE performance:
\begin{lstlisting}[language=bash]
python VAEInference.py
python StatsComputer.py
\end{lstlisting}

\subsection{Latent Diffusion Model Training}

Train the latent diffusion (DDPM) model:
\begin{lstlisting}[language=bash]
python latentDiffusion.py
\end{lstlisting}

By default, this script loads pre-trained parameters. Training may be skipped if inference-only usage is desired.

\subsection{Image Enhancement Inference}

Generate enhanced high-quality MRI samples:
\begin{lstlisting}[language=bash]
python inferenceLDM.py
\end{lstlisting}

Construct the enhanced MRI dataset for the segmentation stage:
\begin{lstlisting}[language=bash]
python sampleEnhancedImg.py
\end{lstlisting}

\section{Stage 2: Mamba-Based Segmentation}

After generating the enhanced MRI dataset, proceed to the segmentation module:
\begin{lstlisting}[language=bash]
cd ../Mamba_based_Segmentation
\end{lstlisting}

Follow the instructions in the segmentation directory to train and evaluate the Mamba-based segmentation model.

\section{Experimental Results}

\begin{itemize}[leftmargin=1.5em]
    \item \textbf{Image Enhancement:} The diffusion-based module consistently produces higher-quality MRI images than GAN-based baselines.
    \item \textbf{Segmentation:} The Mamba-based segmentation network achieves improved accuracy and robustness over several classical segmentation architectures.
\end{itemize}

Detailed quantitative and qualitative analyses are provided in the accompanying paper and experiment logs.

\section{Citation}

If you find this work useful, please cite:
\begin{verbatim}
@article{DiffusionMamba,
  title   = {DiffusionMamba: Diffusion-Based Image Enhancement Integrated
             with Mamba for Accurate Medical Image Segmentation},
  author  = {Si, Jiatong},
  year    = {2025}
}
\end{verbatim}

\section{License}

This project is released for academic and research use. Please refer to the LICENSE file for details.

\end{document}
