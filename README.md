# Is-It-Fake-Or-Not

## Overview
This repository hosts the code for the paper "Is It Fake Or Not? A Comprehensive Approach to Multimodal Fake News Detection." The project leverages [Themis](https://github.com/demon-prin/Themis-SEMEVAL-public), a multimodal binary classification model that analyzes both images and text to accurately classify news as either fake or real.

### Key Features
- [Feature 1]: [Experimentation on two types of dataset: Fakeddit and ReCOVery]
- [Feature 2]: [Use of data augmentation techniques: Text Synonyms and Image Transformations (TSIT) and MixGen]

## Setup

To get started with this project locally, follow the steps below:

1. **Clone the Repository**
    ```bash
    git clone https://github.com/demon-prin/Is-It-Fake-Or-Not
    ```
   This will clone the repository to your local machine.
   In order to reproduce the experiment, you can download the training, validation and test images from here:
   - Fakeddit: https://drive.google.com/file/d/1coi5b2MwQW3DqCLOg9Wxk6QXVoHhvcc2/view?usp=drive_link
   - ReCOVery: https://unicadrsi-my.sharepoint.com/:f:/g/personal/davideantonio_mura_unica_it/EuHuq0aVm4dOuaIpPKoQks0Bv5hPjyvWZIkP3vnFVd4dnQ?e=2zSufx

3. **Navigate to the Project Directory**
    ```bash
    cd Is-It-Fake-Or-Not
    ```
   After cloning, move into the project directory.

4. **Create Virtual Environment**
    ```
    python -m venv /path/to/new/virtual/environment
    ```
5. **Install Requirements**
    
    ```bash
    pip install -r requirements.txt
    ```

## Train
If you want to train the model on a custom dataset, simply add a new class to the datasets.py file and implement the required methods. You can launch the training with:
```bash
    python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-base-patch32" --batch_size 4
```
or with LoRA:
```bash
    python train.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-base-patch32" --batch_size 4 \\
    --use_lora True --lora_alpha 8 --lora_r 8 --lora_dropout 0.4
```
## Evaluation
To launch a model evaluation
```bash
    python eval.py --name_llm "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --name_img_embed "openai/clip-vit-base-patch32"  --batch_size 4 \\
    --model_path "path/to/model.pt"
```
If ``` --set_params True ``` is active, LoRA parameters are extracted from model's name. Otherwise you have to specify the parameters used during train.


## License
This project is licensed under the [MIT License](LICENSE).
