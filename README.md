# ClearVariant

![Image_fjqwdqfjqwdqfjqw](https://github.com/user-attachments/assets/74343964-cbb5-4359-ab54-464f863f128e)

## Overview
ClearVariant is a deep learning framework designed to predict the functional impact of human genetic variants, with a particular focus on identifying the direction of mutational effects — such as gain-of-function (GOF) and loss-of-function (LOF) mutations. Built on protein language models (PLMs) fine-tuned with human variant data, ClearVariant analyzes full-length mutant and reference protein sequences to achieve state-of-the-art performance on benchmark datasets including ClinVar, HGMD and ProteinGym.

### Key features:
* Fine-tuned PLM architecture trained on comprehensive human variant datasets
* Accurate classification of GOF vs. LOF mutations
* Precomputed prediction database for all possible human missense variants

This repository contains the model code, instructions for train and inference, and links to the variant effect database.

## Paper
* **Title:** Learning sequence to predict gain- or loss-of-function variants
* **Authors:** Sungnam Kim, Kisang Kwon, Wonseok Chung, Joohyun Han, Doyeon Ha
* **Venue:** [Conference/Journal Name, Year]
* **Link:** [Link to Paper PDF or DOI]

## Updated Logs
### May 15, 2025
* Initial release of ClearVariant

## Requirements
* Hardware: A100 GPU (Not typically found in standard desktop computers)
* OS: Linux
* Python version: 3.12.1
* Main Dependencies: [env.yml](env.yml)

## Installation
(Installation time: about 10 minutes)
* Clone git repository
    ```bash
    git clone https://github.com/Doyeon-Ha/ClearVariant.git
    ```
* Build mamba environment
    ```bash
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    ```
    ```bash
    bash Miniforge3-Linux-x86_64.sh
    ```
    ```bash
    mamba env create --file env.yml
    ```
    ```bash
    mamba activate ClearVariant
    ```
* Setup mlflow (optional)
    ```bash
    export MLFLOW_URI="{your_mlflow_uri}" # add last at .bashrc
    ```
    Change the `tracking_tool: str="file"` to `tracking_tool: str="mlflow"` in the `start_run` function in `utils/log_metrics.py`.


## Dataset
[Dataset_README](data/input/README.md)

## Usage
ClearVariant supports a variety of tasks by modifying the configuration files in the `config/` directory and running `main.py`.

You can also override specific configuration values temporarily via command-line arguments without editing the original config files.

For example, the following command runs a one-epoch training session for the "goflof" task without modifying any files in the repository:
```bash
python main.py task_operator_config.epochs=1
```
This is especially useful for quick testing or debugging.

## Training
### ClinVar/HGMD (GOF and LOF)
For five-fold cross-validation, run the process a total of five times, changing `dataset_builder_config.test_fold=0` to values from 0 to 4.
To select your GPU device, set an appropriate value for `model_config.device`.
```bash
python main.py dataset_builder_config.test_fold=0 model_config.device=0
```
(Each epoch takes approximately 10 minutes.)

### ProteinGym
To select the ProteinGym dataset to train on, choose one of the entries from the `options.db` list in the `config/db/db_proteingym.yaml` file, and set it as the value for `raw_data_processor_config.db_option`.
```bash
python main.py db@_global_=db_proteingym raw_data_processor_config.db_option=A4_HUMAN_Seuma_2022.csv
```

## Evaluation
### ClinVar/HGMD (GOF and LOF)
To perform inference on the results of five-fold cross-validation, use the same `dataset_builder_config.test_fold` value as was used during model training.
You can find the path to the corresponding model.safetensors file by referring to the [Result_README](data/result/model/README.md).
```bash
python main.py task@_global_=task_inference dataset_builder_config.test_fold=0 model_config.model_checkpoint={abs_path_to_model.safetensors}
```
You can evaluate your dataset using the following command.
```bash
python main.py task@_global_=task_inference dataset_builder_config.processed_dataset={your_dataset_path} dataset_builder_config.db_processing=all_test model_config.model_checkpoint={abs_path_to_model.safetensors}
```

### ProteinGym
To evaluate the trained model, select the correct pair of `raw_data_processor_config.db_option` and `model_checkpoint`.
```bash
python main.py task@_global_=task_inference db@_global_=db_proteingym raw_data_processor_config.db_option=A4_HUMAN_Seuma_2022.csv model_config.model_checkpoint={abs_path_to_model.safetensors}
```

## Repository Structure
```
ClearVariant/
├── config/                    # Configuration files for running main.py
│   ├── db/                    # Configuration files for various database options
│   ├── task/                  # Configuration files for different tasks (e.g., training, inference)
│   └── config.yaml            # Main configuration file that selects one db and one task config to use in main.py
├── data/                      # Directory for input datasets and output results
│   ├── input/                 # Raw input data
│   └── result/                # Output results generated by the main.py
├── logged_metric/             # Stores performance metrics for each epoch during training (only if MLflow is not used)
├── pipeline/                  # Core modules required to run main.py
│   ├── neuron/                # Generates protein variant sequences using protein reference sequences and HGVSp codes
│   ├── datasetbuilder.py      # Converts data into the standardized format required by ClearVariant
│   ├── gettaskoperator.py     # Creates task operator and model adapter objects used in main.py
│   ├── model.py               # Defines the architecture of the ClearVariant model
│   ├── modeladapter.py        # Helper module for model execution
│   ├── rawdataprocessor.py    # Preprocesses various raw datasets into a unified format
│   └── taskoperator.py        # Executes a specific task (train, inference, or write) based on the task option
├── utils/
│   ├── log_metrics.py         # Logs performance metrics for each epoch (can utilize MLflow)
│   └── utils.py               # Utility functions for main.py, including real-time progress logging
├── .gitignore                 # Git ignore rules
├── env.yml                    # Conda environment specification
├── LICENSE                    # License file
├── main.py                    # Entry point script to run tasks such as training or inference
└── README.md                  # Project documentation
```

## License

This project is licensed under the MIT License. You can find the full license text in the [LICENSE](LICENSE) file in the root directory of this repository.

We kindly ask you to acknowledge and cite the original work if you use this code or build upon it (see the [Citation](#citation) section below).

---

This project utilizes the **ESM-2** model, developed by Meta AI (Facebook Research). ESM-2 is also distributed under the **MIT License**. In compliance with its terms, we acknowledge its use. For the specific license terms governing ESM-2, please refer to the official [ESM-2 repository](https://github.com/facebookresearch/esm) and its associated license file.

## Citation
(Provide instructions on how others should cite your work if they use your code or paper.)

## Acknowledgements (Optional)
(Acknowledge funding sources, individuals who helped, or specific resources used.)

## Contact
Doyeon Ha - biologysaves@gmail.com

Access to Pretrained Models
Due to the sensitive nature of the training data and potential safety concerns, the pretrained weights for ClearVariant are not publicly available. However, we are happy to share the model with academic and industrial researchers upon request.
To access the model, please send an email to biologysaves@gmail.com with the following information to verify your identity and intended usage:

Subject: Request for ClearVariant Pretrained Weights

Full Name:

Affiliation: (University, Research Institute, or Company)

Intended Use: (Briefly describe how you plan to use the model and for what research purpose)


Note: Requests from official institutional email addresses are preferred for faster verification.


