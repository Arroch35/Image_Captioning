# Image_Captioning
Challenge 3 of the Vision &amp; Learning subject

## Table of Contents

- [Introduction](#1-introduction)
- [Project Structure](#2-project-structure)
- [Setup](#3-setup)
- [Usage and Workflow](#4-usage-and-workflow)

## 1. Introduction
This is the repository for the challenge 2 of the Vision & Learning subject. Represents 2 
systems that help diagnose whether a patient has Helicobacter pylori.

## 2. Project structure
```bash
src/
├── cat_to_name.json # File with the classes of the dataset
├── config.py # Configuration settings
├── custom_clip_training.py # Training for all custom models
├── datasets.py # Dataset class
├── evaluation_with_temperatures.py # Experiment of temperature
├── evaluation.py # Experiment 1
├── fine_tuning.py # Training original CLIP in Flowers102 dataset
├── flowers_names.py # File with the classes of the dataset
├── P3_Models.py # Models and encoders classes and wrappers
├── ttest.py # Experiment 2
├── utils.py # Some utility functions
└── utils2.py # Some utility functions

``` 
## 3. Setup

To set up the project environment, run the following commands in the root directory:

```bash
# 1. Create a virtual environment (named 'venv')
python -m venv venv

# 2. Activate the environment (Linux/macOS)
source venv/bin/activate

# 2. Activate the environment (Windows)
# venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

The **`config.py`** file is used to configure all project and data path variables, ensuring consistency across the codebase. Change the value of the variables according to your paths.

---

## 4. Usage and Workflow

The 2 first files that has to be runned are fine_tuning.py and custom_clip_training.py, which are the files used to train our models. Then, evaluation.py must be run for once each of the models. If the best model is not clearly stated, the ttest.py file will help determine it. After this, we can run evaluation_with_temperature.py with the best performing model to see the role of the temperature parameter. The rest of the files are used within the mentioned files, so no need to run them.