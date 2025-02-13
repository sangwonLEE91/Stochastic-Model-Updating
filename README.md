# Latent Space-based Stochastic Model Updating

This repository implements a novel latent space-based method for stochastic model updating in engineering systems. The approach leverages a variational autoencoder (VAE) to build an amortized probabilistic model, enabling efficient uncertainty quantification without the need for iterative probability estimation during MCMC simulation.

## Overview

Instead of relying on extensive training data, this code uses a latent space to represent and update uncertainties in engineering models. The main contributions of the code are:

- **Latent Space Representation:** The model transforms high-dimensional input data into a lower-dimensional latent space, capturing essential uncertainty information.
- **Amortized Probabilistic Modeling:** By training a VAE, the model learns a probabilistic mapping from inputs to latent variables, streamlining the uncertainty quantification process.
- **Efficient Uncertainty Quantification:** The framework evaluates uncertainties using metrics like Bhattacharyya and Euclidean distances, making it applicable even with limited data.
- **Flexibility:** The method is designed to work with both static and time-series data, demonstrated through numerical experiments on benchmark problems such as a two-degree-of-freedom shear spring model and the NASA UQ Challenge 2019.

## Code Structure

- **`main.py`**  
  Serves as the entry point for the application. It handles configuration loading, data preprocessing, model initialization, training, and validation routines.

- **`models/`**  
  Contains the deep learning architectures, including:
  - **VAE Implementation:** Defines the encoder and decoder networks that learn the latent representation.
  - **Auxiliary Models:** Additional modules that might be used for feature extraction or model updating.

- **`experiments/`**  
  Hosts scripts for running different experiments, including:
  - **Numerical Experiments:** Scripts to test the method on synthetic data (e.g., the two-degree-of-freedom shear spring model).
  - **Time-Series Calibration:** Scripts for applying the method to time-series data, such as the NASA UQ Challenge 2019.

- **`data/`**  
  Provides sample datasets and instructions on how to format or generate new data for running the experiments.

- **`utils/`**  
  Contains helper functions for:
  - Data preprocessing and augmentation.
  - Evaluation metrics calculation (e.g., Bhattacharyya and Euclidean distances).
  - Saving and loading model checkpoints.

## Requirements

- **Python:** Version 3.8 or 3.11  
- **Dependencies:** All required Python packages are listed in the [`requirements.txt`](requirements.txt) file.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

**Usage**
```bash
To execute the main training and model updating routine, run:
   python main.py

You can adjust configuration parameters such as model hyperparameters and data paths via a configuration file (e.g., config.yaml) or directly within the code.

**Citation**

If you find this code useful in your research, please cite our work:
```bibtex
@misc{lee2024latentspacebasedstochasticmodel,
  title={Latent Space-based Stochastic Model Updating}, 
  author={Sangwon Lee and Taro Yaoyama and Masaru Kitahara and Tatsuya Itoi},
  year={2024},
  eprint={2410.03150},
  archivePrefix={arXiv},
  primaryClass={stat.AP},
  url={https://arxiv.org/abs/2410.03150}
}

License

This project is licensed under the [Your License Here] license.

   
