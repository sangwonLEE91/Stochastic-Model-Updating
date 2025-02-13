# Latent Space-based Stochastic Model Updating

This repository implements a novel latent space-based method for stochastic model updating in engineering systems. The approach leverages a variational autoencoder (VAE) to build an amortized probabilistic model, enabling efficient uncertainty quantification without the need for assuming likelihood function.


## Code Structure

- **`main.py`**  
  Loads the trained VAE (z3_e1000.pth) and performs posterior sampling using the **Replica Exchange Monte Carlo (REMC)** method.

- **`REMC_func.py`**  
  Contains necessary functions for performing **Replica Exchange Monte Carlo (REMC)** sampling.

- **`lee.cp38_win_amd64.pyd` / `lee.cp311_win_amd64.pyd`**  
  Compiled Fortran-wrapped likelihood evaluation functions for the **latent space-based method**, built for Python **3.8** and **3.11**, respectively.


## Requirements
- **Python:** Version 3.8 or 3.11  


## Usage
To execute the main model updating routine, run:
  ```bash
  python main.py
  ```


## Citation

If you find this code useful in your research, please cite our work:
```bibtex
  @misc{lee2024lssmu,
    title={Latent Space-based Stochastic Model Updating}, 
    author={Sangwon Lee and Taro Yaoyama and Masaru Kitahara and Tatsuya Itoi},
    year={2024},
    eprint={2410.03150},
    archivePrefix={arXiv},
    primaryClass={stat.AP},
    url={https://arxiv.org/abs/2410.03150}
  }
```


   
