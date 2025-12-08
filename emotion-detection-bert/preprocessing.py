import pandas as pd
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

all_files = os.listdir('data')
csv_files = [f for f in all_files if f.endswith('.csv')]
required_csv = ['train.csv', 'test.csv', 'val.csv']
dfs = []

print("CSV files detected in the data directory:", csv_files)
for filename in required_csv:
    if filename in csv_files:
        df_part = pd.read_csv(f"data/{filename}")
        dfs.append(df_part)

if not dfs:
    raise RuntimeError("No expected CSV files were found. Ensure that train.csv, test.csv, and val.csv are present in the data directory.")

df = pd.concat(dfs, ignore_index=True)

column_mapping = {}
for col in df.columns:
    if col.lower() in ['text', 'content', 'sentence', 'tweet', 'message']:
        column_mapping[col] = 'text'
    if col.lower() in ['label', 'emotion', 'sentiment', 'class', 'category']:
        column_mapping[col] = 'label'

df = df.rename(columns=column_mapping)
df = df[['text', 'label']]

if df['label'].dtype in ['int64', 'int32', 'float64']:
    label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    df['label'] = df['label'].map(label_mapping)

df = df.dropna()
df = df.drop_duplicates()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
emotional_words = {
    'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody', 'nothing', 'nowhere',
    'dont', 'doesn', 'didn', 'won', 'wouldn', 'shouldn', 'couldn', 'cant', 'cannot',
    'ain', 'aren', 'isn', 'wasn', 'weren', 'haven', 'hasn', 'hadn', 'shan', 'very',
    'really', 'so', 'too', 'happy', 'glad', 'joy', 'joyful', 'sad', 'angry', 'love',
    'hate', 'scared', 'amazing', 'wonderful', 'great', 'good', 'fantastic',
    'excellent', 'lovely', 'awesome', 'perfect', 'delighted', 'pleased', 'cheerful',
    'unhappy', 'depressed', 'miserable', 'terrible', 'awful', 'bad', 'horrible',
    'mad', 'furious', 'annoyed', 'frustrated', 'upset', 'disappointed', 'hurt',
    'fear', 'worried', 'anxious', 'nervous', 'terrified', 'surprised', 'shock',
    'shocked', 'unexpected', 'wow', 'omg'
}
stop_words = stop_words - emotional_words


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\S+@\S+", '', text)
    text = re.sub(r"<.*?>", '', text)
    text = ' '.join(text.split())
    return text


def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    processed = []
    for token in tokens:
        if len(token) < 2 and token != 'i':
            continue
        if token in stop_words:
            continue
        lemma = lemmatizer.lemmatize(token)
        processed.append(lemma)
    return ' '.join(processed)


def handle_negations(text):
    tokens = text.split()
    result = []
    neg_words = {
        'not', 'no', 'never', 'dont', 'doesn', 'didn', 'won', 'wouldn', 'shouldn',
        'couldn', 'cant', 'cannot', 'ain', 'aren', 'isn', 'wasn', 'weren', 'haven',
        'hasn', 'hadn', 'shan'
    }
    negated = False
    for token in tokens:
        if any(neg in token for neg in neg_words):
            result.append(token)
            negated = True
        elif negated:
            result.append(f'NOT_{token}')
            negated = False
        else:
            result.append(token)
    return ' '.join(result)


df['processed_text'] = df['text'].apply(preprocess_text).apply(handle_negations)
df = df[df['processed_text'].str.strip() != '']
df['word_count_processed'] = df['processed_text'].apply(lambda x: len(str(x).split()))
df = df[df['word_count_processed'] >= 1]
df_out = df[['processed_text', 'label']].rename(columns={'processed_text': 'text'})

emotions = set(df_out['label'].unique())
aug_rows = []
for label in emotions:
    if label.lower() == 'neutral':
        continue
    neg_texts = [
        f"I am not {label.lower()}",
        f"I do not feel {label.lower()}",
        f"I do not feel {label.lower()} at all",
        f"I am definitely not {label.lower()}",
        f"This does not make me feel {label.lower()}"
    ]
    for text in neg_texts:
        aug_rows.append({'text': text, 'label': 'neutral'})

df_aug = pd.DataFrame(aug_rows)
df_final = pd.concat([df_out, df_aug]).sample(frac=1).reset_index(drop=True)

df_final.to_csv('data/final_data_aug.csv', index=False)
print("Data loading, preprocessing, and augmentation have been completed.")
print("The processed and augmented dataset has been saved to data/final_data_aug.csv.")
