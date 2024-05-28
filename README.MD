# README

## Overview

This repository contains the resources and scripts related to the research paper titled "SemTest: A Semantic Equivalence-Based Mutation Testing Method for Deep Learning Frameworks." Below is a detailed description of each directory and the information it contains.

## Directories

### Exceptions
This directory contains files that record exceptions encountered during the execution of different testing methods.

- **COMET_exceptions.txt**: Logs of exceptions encountered during COMET testing.
- **LEMON_exceptions.txt**: Logs of exceptions encountered during LEMON testing.

### LOGS
This directory contains the results of various tests conducted using different frameworks and models. It includes subdirectories for each testing method and the corresponding models tested.

- **COMET_result**: Contains results of the COMET testing method.
  - **ALL**: Aggregated results for all models tested.
  - **Model-specific subdirectories**: Each subdirectory (e.g., `alexnet`, `densenet`) contains results specific to that model.
    - **FAILED.txt**: Records of failed tests.
    - **output_coverage.xlsx**: Coverage metrics for the model.

- **LEMON_result**: Contains results of the LEMON testing method.
  - **Model-specific subdirectories**: Each subdirectory (e.g., `alexnet`, `densenet`) contains results specific to that model.
    - **FAILED.txt**: Records of failed tests.
    - **output_coverage.xlsx**: Coverage metrics for the model.

### 实验结果 (Experimental Results)
This directory contains experimental results from various tests conducted on different frameworks and models.

- **EAGLE**: Contains results for the EAGLE testing method.
  - **tensorflow_21_output_file.csv**: Output file for TensorFlow model 21.

- **MindSpore覆盖率实验**: Coverage experiments for MindSpore framework.
  - **Model-specific subdirectories**: Each subdirectory (e.g., `deeplabv3`, `openpose`) contains logs and coverage results for that model.
    - **LOG_DICT_Linux_CPU.json**: Log dictionary for CPU runs.
    - **SemTest.xlsx**: Coverage results.

- **RQ3消融实验**: Ablation experiments for RQ3.
  - **Model-specific subdirectories**: Each subdirectory (e.g., `ssimae MCMC absoca`, `textcnn MCMC hybrid`) contains logs and coverage results for that model.
    - **LOG_DICT_Linux_GPU.json**: Log dictionary for GPU runs.
    - **SemTest.xlsx**: Coverage results.

- **mtcomp-cpu**: Results for multi-task comparison on CPU.
  - **mobilenet.xlsx**: Coverage results for MobileNet.
  - **resnet.xlsx**: Coverage results for ResNet.
  - **vgg11.xlsx**: Coverage results for VGG11.

- **mtcomp-gpu**: Results for multi-task comparison on GPU.
  - **mobilenet.xlsx**: Coverage results for MobileNet.
  - **resnet_json_gpu.xlsx**: Coverage results for ResNet.
  - **vgg11_gpu.xlsx**: Coverage results for VGG11.

- **torch实验 GPU**: Torch experiments on GPU.
  - **Model-specific subdirectories**: Each subdirectory (e.g., `deeplabv3`, `openpose`) contains logs and data files for that model.
    - **MUTATE_LOG_Linux_cuda_*.txt**: Mutation logs for CUDA runs.
    - **TORCH_MCMC_LOG_DICT_Linux_cuda_*.json**: MCMC log dictionary for CUDA runs.
    - **TORCH_MCMC_data.xlsx**: Coverage results.
    - **data0_npy.npy**: Numpy data files used in experiments.

- **torch实验 cpu**: Torch experiments on CPU.
  - **Model-specific subdirectories**: Each subdirectory (e.g., `SSDmobilenetv1`, `resnet`) contains logs and data files for that model.
    - **MUTATE_LOG_Linux_cpu.txt**: Mutation logs for CPU runs.
    - **TORCH_MCMC_LOG_DICT_Linux_cpu.json**: MCMC log dictionary for CPU runs.
    - **data0_npy.npy**: Numpy data files used in experiments.
    - **Screenshots**: Screenshots of the experimental results.

### Exception_Counter.py
This script is used to count and analyze exceptions recorded in the logs. It processes the exception files and generates a summary of the exceptions encountered during the testing process.

## Usage

To use this repository, follow these steps:

1. **Clone the repository**: 
   ```
   git clone <repository_url>
   ```

2. **Navigate to the desired directory**:
   ```
   cd <desired_directory>
   ```

3. **Run the exception counter script**:
   ```
   python Exception_Counter.py
   ```

4. **Explore the logs and results**: Open the log files and coverage results in the respective directories to analyze the testing outcomes.

For any questions or issues, please refer to the contact information provided in the research paper.