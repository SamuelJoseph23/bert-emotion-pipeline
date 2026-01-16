# BERT-based Emotion Recognition Pipeline

## Overview

This project provides a fully reproducible pipeline for text-based emotion detection using BERT and modern NLP preprocessing. 
Includes:
- **Data Pipeline**: Automated merging, advanced preprocessing, and negation-based augmentation.
- **BERT Model**: Fine-tuned `distilbert-base-uncased` for multi-class emotion classification.
- **Interfaces**: CLI interactive mode and a modern Streamlit web application.

---

## Features

- **Unified data preparation script:** `prepare_data.py` combines, preprocesses, and augments emotion datasets automatically.
- **Automated NLTK setup:** No manual downloads required; the script handles `punkt`, `stopwords`, etc.
- **Advanced Preprocessing:** Lemmatization, proper stopword handling, and explicit negation marking (e.g., "not happy" -> "NOT_happy").
- **Augmentation:** Adds robust negated phrase samples for each emotion class to improve "neutral" detection.
- **Flexible Inference:** Choose between a Command Line Interface (CLI) or a Web UI.

---

## Repository Structure

```
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── val.csv
│   ├── final_data_aug.csv   # Generated training data
├── prepare_data.py         # Data pipeline: load, preprocess, augment
├── bert_emotion.py         # Model training & CLI inference
├── app.py                  # Streamlit Web Application
├── requirements.txt        # Dependencies
├── README.md               # Documentation
└── bert_emotion_model/     # Saved model artifact (after training)
```

---

## Setup

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. NLTK Data
NLTK data is now automatically downloaded when you run the scripts for the first time.

---

## Usage

### 1. Data Preparation
Run the unified script to preprocess and augment data:
```bash
python prepare_data.py
```
- **Output:** `data/final_data_aug.csv`

### 2. Model Training
Train the BERT emotion classifier:
```bash
python bert_emotion.py
```
- Choose **Option 1** for training.
- The model will be saved in `bert_emotion_model/`.

---

## Inference

### **A. Command-line Interface (CLI)**
```bash
python bert_emotion.py
```
- Choose **Option 2** for interactive detection.
- Type sentences and press Enter for results.

### **B. Web Application (Streamlit)**
For a more visual experience, run the web app:
```bash
streamlit run app.py
```

---

## Requirements

The pipeline requires the following Python libraries:
- `pandas`
- `nltk`
- `torch`
- `transformers`
- `scikit-learn`
- `accelerate`
- `streamlit` (for the web app)

---

## Tips
- **GPU Acceleration:** The scripts automatically detect CUDA-enabled GPUs for faster training and inference.
- **Custom Labels:** You can tune emotion labels in `prepare_data.py`.
- **Model Tuning:** Adjust epochs and batch sizes in `bert_emotion.py` within `TrainingArguments`.

---

## License
MIT License

---
_Developed with automation and ease of use in mind._
