import pandas as pd
import os
import sys
import torch
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline


DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'final_data_aug.csv')
MODEL_NAME = 'distilbert-base-uncased'
MODEL_PATH = './bert_emotion_model'


def train_bert():
    df = pd.read_csv(CSV_PATH)
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])

    X_train, X_val, y_train, y_val = train_test_split(
        df['text'], df['label_id'], test_size=0.2, random_state=42, stratify=df['label_id']
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=128)

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
    num_labels = len(le.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='./bert_results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        save_total_limit=1,
        logging_dir='./logs',
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        from sklearn.metrics import accuracy_score
        return {'accuracy': accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("Validation results:", metrics)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    with open(f'{MODEL_PATH}/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("Model, tokenizer, and label encoder have been saved to ./bert_emotion_model.")


def interactive_predict():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    with open(f'{MODEL_PATH}/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    print("\nEmotion detection is ready. Type text and press Enter. Type 'exit' to close the program.\n")
    while True:
        user_input = input("Enter text: ")
        if user_input.strip().lower() == 'exit':
            print("Program terminated.")
            break
        if not user_input.strip():
            continue
        result = classifier(user_input)[0]
        pred_id = int(result['label'].replace('LABEL_', '').replace('LABEL', '')) if 'LABEL' in result['label'] else int(result['label'])
        emotion = le.inverse_transform([pred_id])[0]
        print(f"Predicted emotion: {emotion}, confidence: {result['score']:.3f}\n")


if __name__ == '__main__':
    print("Select an action:")
    print("1. Train BERT model")
    print("2. Interactive emotion detection")
    choice = input("[1/2]: ").strip()
    if choice == '1':
        train_bert()
    elif choice == '2':
        interactive_predict()
    else:
        print("Invalid selection.")
