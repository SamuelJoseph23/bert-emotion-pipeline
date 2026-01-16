import pandas as pd
import os
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Configuration ---
DATA_DIR = Path('data')
OUTPUT_FILE = DATA_DIR / 'final_data_aug.csv'
REQUIRED_CSVS = ['train.csv', 'test.csv', 'val.csv']

# --- NLTK Setup ---
def download_nltk_data():
    """Automate NLTK resource downloading."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for res in resources:
        try:
            # Check if resource is already available
            nltk.data.find(f'corpora/{res}' if res != 'punkt' else f'tokenizers/{res}')
        except LookupError:
            print(f"Downloading required NLTK resource: {res}")
            nltk.download(res, quiet=True)

download_nltk_data()

# --- Data Loading ---
def load_data():
    """Load and merge CSV files from the data directory."""
    if not DATA_DIR.exists():
        print(f"Creating missing directory: {DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_files = list(DATA_DIR.glob('*.csv'))
    csv_filenames = [f.name for f in all_files]
    
    dfs = []
    print(f"Detected {len(csv_filenames)} CSV files: {csv_filenames}")

    # Prioritize specific files
    found_required = False
    for filename in REQUIRED_CSVS:
        file_path = DATA_DIR / filename
        if file_path.exists():
            print(f"Loading required file: {filename}")
            dfs.append(pd.read_csv(file_path))
            found_required = True

    # Fallback to any CSV if required ones are missing
    if not found_required:
        print("Required CSVs (train/test/val) not found. Attempting fallback to other CSVs.")
        for f_path in all_files:
            if f_path.name != OUTPUT_FILE.name:
                print(f"Falling back to: {f_path.name}")
                dfs.append(pd.read_csv(f_path))

    if not dfs:
        raise RuntimeError(f"No usable CSV files found in {DATA_DIR}. Please add train.csv, test.csv, or val.csv.")

    return pd.concat(dfs, ignore_index=True)

# --- Preprocessing Logic ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Keep emotional indicators even if they are common words
emotional_words = {
    'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody', 'nothing', 'nowhere',
    'dont', 'doesn', 'didn', 'won', 'wouldn', 'shouldn', 'couldn', 'cant', 'cannot',
    'ain', 'aren', 'isn', 'wasn', 'weren', 'haven', 'hasn', 'hadn', 'shan', 'very',
    'really', 'so', 'too', 'happy', 'sad', 'angry', 'love', 'hate', 'scared', 'wow'
}
stop_words = stop_words - emotional_words

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Links
    text = re.sub(r"\S+@\S+", '', text)                # Emails
    text = re.sub(r"<.*?>", '', text)                   # HTML
    return ' '.join(text.split())

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    processed = []
    for token in tokens:
        if len(token) < 2 and token != 'i':
            continue
        if token in stop_words:
            continue
        processed.append(lemmatizer.lemmatize(token))
    return ' '.join(processed)

def handle_negations(text):
    """Simple heuristic to mark words following a negation."""
    tokens = text.split()
    result = []
    neg_words = {'not', 'no', 'never', 'dont', 'doesn', 'didn', 'won', 'cant', 'cannot'}
    negated = False
    for token in tokens:
        if token in neg_words:
            result.append(token)
            negated = True
        elif negated:
            result.append(f'NOT_{token}')
            negated = False
        else:
            result.append(token)
    return ' '.join(result)

# --- Main Pipeline Execution ---
def main():
    print("Starting data preparation pipeline...")
    df = load_data()

    # Column normalization
    column_mapping = {
        'content': 'text', 'sentence': 'text', 'tweet': 'text', 'message': 'text',
        'emotion': 'label', 'sentiment': 'label', 'class': 'label', 'category': 'label'
    }
    df = df.rename(columns=lambda x: column_mapping.get(x.lower(), x.lower()))
    
    if 'text' not in df.columns or 'label' not in df.columns:
        # Final attempt to find text/label columns if naming is odd
        for col in df.columns:
            if df[col].dtype == 'object' and 'text' not in df.columns:
                df = df.rename(columns={col: 'text'})
                break
        if 'label' not in df.columns:
             raise KeyError(f"Could not identify 'text' and 'label' columns. Columns found: {list(df.columns)}")

    df = df[['text', 'label']].dropna().drop_duplicates()

    # Label integer-to-string conversion if needed
    if df['label'].dtype in ['int64', 'int32']:
        lbl_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
        df['label'] = df['label'].map(lbl_map)

    print("Preprocessing text data...")
    df['text'] = df['text'].apply(preprocess_text).apply(handle_negations)
    df = df[df['text'].str.strip() != '']

    # Augmentation: Adding neutral samples for negated labels
    print("Augmenting data with neutral negation samples...")
    emotions = df['label'].unique()
    aug_rows = []
    for label in emotions:
        if str(label).lower() == 'neutral': continue
        base = str(label).lower()
        aug_rows.extend([
            {'text': f"i am not {base}", 'label': 'neutral'},
            {'text': f"i don't feel {base}", 'label': 'neutral'},
            {'text': f"definitely not {base}", 'label': 'neutral'}
        ])
    
    df_aug = pd.concat([df, pd.DataFrame(aug_rows)])
    df_final = df_aug.sample(frac=1).reset_index(drop=True)

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Processed dataset saved to: {OUTPUT_FILE}")
    print(f"Total samples: {len(df_final)}")

if __name__ == "__main__":
    main()
