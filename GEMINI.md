# GEMINI.md

This file provides a comprehensive overview of the `neural-fx` project, intended to be used as a contextual guide for AI-driven development.

## Project Overview

`neural-fx` is a Python project for real-time guitar effect and amplifier modeling using neural networks. The project is built on PyTorch and PyTorch Lightning, and uses a modular structure with YAML configuration files for defining model architectures and training parameters.

The primary goal is to train neural networks to emulate the sound of analog guitar effects pedals and amplifiers.

## Development Workflow

This project follows a strict development workflow to ensure code quality and consistency.

### 1. Preparation
*   **Base Branch**: All new features and bug fixes should start from the `development` branch (not `main`).
*   **Sync**: Always pull the latest changes before starting:
    ```bash
    git checkout development && git pull
    ```
*   **Branching**: Create a descriptive branch for your task:
    ```bash
    git checkout -b feature/your-feature-name
    # or
    git checkout -b fix/your-bug-fix
    ```

### 2. Planning
*   **Analyze**: Understand the requirements (refer to `TODO.md` or issue descriptions).
*   **Plan**: Formulate a clear plan. If the task is complex, document the plan in a markdown file in `docs/plans/` (or update `TODO.md`).
*   **Verify**: Ensure the plan aligns with the project's architecture and conventions.

### 3. Implementation
*   **Code**: Implement changes based on the plan.
*   **Conventions**:
    *   Inherit from `BaseNeuralFXModel` for all models.
    *   Use `neural-fx/config.py` dataclasses for configuration.
    *   Keep logic modular (models in `models/`, training in `training/`, etc.).
*   **Testing strategy**:
    *   **Avoid redundancy**: Do not write tests if the feature is trivial or already covered.
    *   **Verify**: If a test is the best way to verify a complex fix or feature, create a targeted test script or unit test.
    *   **Manual verification**: Use simple scripts to instantiate classes or run a forward pass to catch basic errors.

### 4. Finalization
*   **Stage & Commit**: Add changes and commit using the user's credentials.
    ```bash
    git add .
    git commit -m "feat: description of changes"
    ```
    *   Use conventional commit messages (e.g., `feat:`, `fix:`, `refactor:`, `docs:`).

## Building and Running

The project is in an early stage of development, and the main training logic is currently within the `notebooks/train_recurrent.ipynb` Jupyter notebook. The Python package structure is in place but many of the files are placeholders.

### Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

*Note: A `requirements.txt` file does not currently exist. Based on `setup.py`, the dependencies are: `torch`, `torchaudio`, `numpy`, `scipy`, `tqdm`, `matplotlib`, `lightning`, and `pyyaml`.*

### Training

Training is performed by running the cells in the `notebooks/train_recurrent.ipynb` notebook.

1.  **Set up the environment:** Ensure all dependencies are installed.
2.  **Open the notebook:** `jupyter notebook notebooks/train_recurrent.ipynb`
3.  **Run the cells:** Execute the cells in the notebook to train the model.

The notebook loads audio data, defines a PyTorch Lightning module for training, and then runs the training loop.

### Key Files and Directories

*   `README.md`: High-level project description.
*   `setup.py`: Project setup and dependencies.
*   `configs/`: Contains YAML configuration files for different models (e.g., `lstm_medium.yaml`).
*   `neural-fx/`: The main Python package for the project.
    *   `config.py`: Defines dataclasses for loading and parsing configuration files.
    *   `models/`: Contains model definitions. `base.py` defines the abstract base class for models.
    *   `data/`: Contains data loading and processing utilities. `data.py` has functions for reading/writing WAV files.
    *   `training/`: Contains the training logic. `lightning_module.py` is intended to hold the PyTorch Lightning module.
*   `notebooks/`: Contains Jupyter notebooks for experimentation and training.
    *   `train_recurrent.ipynb`: The main notebook for training recurrent models like LSTM and GRU.
*   `scripts/`: Intended for command-line scripts for training and exporting models (currently empty).

## Development Conventions

*   **Configuration:** The project uses YAML files for configuration, which are parsed into dataclasses defined in `neural-fx/config.py`. This allows for a clean separation of configuration from code.
*   **Models:** Models inherit from the `BaseNeuralFXModel` abstract base class defined in `neural-fx/models/base.py`. This ensures that all models have a consistent interface for training, processing, and exporting.
*   **Training:** Training is managed by PyTorch Lightning. The training logic is encapsulated in a `LightningModule`.
*   **Data:** Data is handled by the `AudioDataset` class, which loads and segments audio files.

## TODOs for AI Agent

*   The `scripts/train.py` script is empty. Implement the training logic from the `notebooks/train_recurrent.ipynb` notebook into this script to allow for command-line training.
*   The `neural-fx/training/lightning_module.py` file is empty. Move the `NeuralfxLightningModule` class from the notebook to this file.
*   The `neural-fx/data/dataset.py` file is empty. Move the `AudioDataset` class from the notebook to this file.
*   The `neural-fx/models/recurrent.py` file is empty. Move the recurrent model definitions from the notebook to this file.
*   The `scripts/export.py` script is empty. Implement model exporting functionality (e.g., to ONNX or TorchScript) in this script.
*   Create a `requirements.txt` file from the dependencies listed in `setup.py`.