import pandas as pd
import os
import sys
import torch
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    pipeline, 
    AutoConfig
)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / 'data'
CSV_PATH = DATA_DIR / 'final_data_aug.csv'
MODEL_NAME = 'distilbert-base-uncased'
MODEL_PATH = SCRIPT_DIR / 'bert_emotion_model'

def train_bert():
    """Fine-tune the BERT model on the prepared dataset."""
    if not CSV_PATH.exists():
        print(f"❌ Error: {CSV_PATH} not found. Please run 'python prepare_data.py' first.")
        return

    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Label Encoding
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    
    # Map IDs to labels for the model config
    label2id = {label: i for i, label in enumerate(le.classes_)}
    id2label = {i: label for i, label in enumerate(le.classes_)}
    num_labels = len(le.classes_)

    print(f"Splitting data (Stratified). Labels detected: {list(le.classes_)}")
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'], df['label_id'], 
        test_size=0.15, 
        random_state=42, 
        stratify=df['label_id']
    )

    print(f"Tokenizing with {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_data(texts):
        return tokenizer(list(texts), truncation=True, padding=True, max_length=128)

    train_encodings = tokenize_data(X_train)
    val_encodings = tokenize_data(X_val)

    class EmotionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = list(labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = EmotionDataset(train_encodings, y_train)
    val_dataset = EmotionDataset(val_encodings, y_val)
    
    # Configure model with labels
    config = AutoConfig.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels, 
        label2id=label2id, 
        id2label=id2label
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Detect device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device_name}")

    training_args = TrainingArguments(
        output_dir='./bert_results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available()  # Faster training on GPU
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        # Print classification report to logs for detailed debugging
        print("\n" + classification_report(labels, preds, target_names=le.classes_))
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1_weighted': f1_score(labels, preds, average='weighted')
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("\n🚀 Starting training process...")
    trainer.train()
    
    print("\nFinal evaluation...")
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)
    
    # Save artifacts
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    with open(MODEL_PATH / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    print(f"\n✅ Done! Model and artifacts saved to: {MODEL_PATH}")


def interactive_predict():
    """Predict emotions from user input via terminal."""
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}. Please run training first.")
        return

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    print("Loading optimized pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    
    with open(MODEL_PATH / 'label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    print("\n--- Emotion Detection Ready ---")
    print("Type text and press Enter. Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input(">> ").strip()
        except EOFError:
            break
            
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        if not user_input:
            continue
            
        result = classifier(user_input)[0]
        label = result['label']
        score = result['score']
        
        # Human-readable mapping fallback
        if label.startswith('LABEL_'):
            pred_id = int(label.split('_')[1])
            emotion = le.inverse_transform([pred_id])[0]
        elif label in le.classes_:
            emotion = label
        else:
            emotion = label

        print(f"Result: {emotion.upper()} (Confidence: {score:.3f})\n")


if __name__ == '__main__':
    print("BERT Emotion Pipeline")
    print("1. Train Model")
    print("2. Interactive Inference")
    
    try:
        choice = input("Select [1/2]: ").strip()
    except EOFError:
        sys.exit(0)
        
    if choice == '1':
        train_bert()
    elif choice == '2':
        interactive_predict()
    else:
        print("Invalid selection.")
