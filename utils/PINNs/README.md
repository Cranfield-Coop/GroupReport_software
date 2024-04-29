# TurbulenceModelPINN: Advanced Turbulence Prediction with Physics-Informed Neural Networks

## Overview

TurbulenceModelPINN offers a sophisticated approach to predicting turbulence dynamics by leveraging the power of Physics-Informed Neural Networks (PINNs). This project is built on the PyTorch Lightning framework, facilitating streamlined model training, evaluation, and management. Designed to tackle the intricate challenge of turbulence prediction, it combines state-of-the-art machine learning techniques with the fundamental principles of fluid mechanics. The architecture encapsulates data preprocessing, training routines, performance evaluation, and extensive logging, providing a holistic solution for researchers and engineers in the field of computational fluid dynamics.

## Key Features

- **Efficient Data Handling:** Incorporates a tailored data module for optimal data manipulation, including loading, preprocessing, and batching operations.
- **Streamlined Model Training:** Harnesses PyTorch Lightning's advanced training capabilities to enhance model development efficiency and reproducibility.
- **Comprehensive Evaluation Framework:** Implements thorough evaluation methodologies on dedicated test datasets, facilitating accurate assessment of model performance against standard benchmarks.
- **In-depth Experiment Tracking:** Employs TensorBoard integration for detailed tracking of training processes, offering insightful visualizations of key metrics and model behavior over time.

## Getting Started

Navigate to the TurbulenceModelPINN project directory:

```bash
cd TurbulenceModelPINN
```

## Detailed Usage Guide

### Preparing Your Data

Prepare your dataset by ensuring it's correctly formatted and located within the `data/` directory. The project expects CSV files for training, validation, and testing phases.

### Model Training Procedure

To initiate the training process, execute:

```bash
python train_model.py
```

This will process your dataset, train the model according to predefined specifications, and generate logs of the training session in the `tb_logs/` directory.

### Model Evaluation

Upon completion of the training phase, evaluate the model's predictive accuracy on the test dataset:

```bash
python test_model.py
```

### Visualizing Training and Evaluation Metrics

Launch TensorBoard to visualize the training and evaluation metrics:

```bash
tensorboard --logdir=tb_logs/
```

Access the provided URL in your browser to explore the training logs and performance metrics.

### Performing Inference

Utilize the trained model for making predictions with:

```bash
python run_inference.py
```

### Customization Options

Tailor the model architecture and training configurations to meet specific requirements by modifying the `train_model.py` script. Parameters such as input dimensions, the architecture of hidden layers, learning rates, and more can be adjusted to optimize performance for different turbulence characteristics.
