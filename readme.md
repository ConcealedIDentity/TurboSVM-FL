# TurboSVM-FL
This is the repository for TurboSVM-FL.

## Teaser
<img src="https://github.com/wmd0701/TurboSVM-FL/assets/34072813/d40ea56b-faa0-4111-b5d7-eb0257da57c5" width="700">

## Usage
run `main_non_FL.py` for centralized learning.

run `main_FL.py` for federated learning.

run `experiments.sh` for reproducing experiments.

To disable loading default hyperparmeters, please run with `-d False`. For detailed arguement settings please check `utils.py`. 

## Environment
Important installed libraries and their versions until **2023 August 1st**:

| Library | Version |
| --- | ----------- |
| Python | 3.10.12 by Anaconda|
| PyTorch | 2.0.1 for CUDA 11.7 |
| TorchMetrics | 0.11.4 |
| Scikit-Learn | 1.2.2 |
| NumPy | 1.25.0 |

Others:
- There is no requirement on OS for the experiment itself. However, to do data preprocessing, Python environment on Linux is needed. If data preprocessing is done in Windows Subsystem Linux (WSL), please make sure `unzip` is installed beforehand, i.e. `sudo apt install unzip` for WSL2 Ubuntu.

- We used **Weights & Bias** (https://wandb.ai/site) for figures instead of tensorboard. Please install and set up it properly beforehand.

- We used the function `match` in our implementation. This function only exists for Python version >= 3.10. Please replace it with `if-elif-else` statement if needed.

## Instructions on data preprocessing
We conducted experiments using four datasets: FEMNIST, CelebA, Shakespeare, and COVID-19. The former three datasets can be obtained from https://leaf.cmu.edu/ together with bash code for reproducible data split.

Please dive into the `data` directory for further instructions.

