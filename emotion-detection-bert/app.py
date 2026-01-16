import streamlit as st
import torch
import os
import pickle
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Set page config
st.set_page_config(
    page_title="BERT Emotion Detector",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    .stButton button {
        border-radius: 20px;
        width: 100%;
        background-color: #4A90E2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = Path('./bert_emotion_model')

@st.cache_resource
def load_pipeline():
    """Load the model, tokenizer, and label encoder."""
    if not MODEL_PATH.exists():
        return None
    
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    
    # Use top_k=None to get all scores
    classifier = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        device=device, 
        top_k=None
    )
    
    label_encoder_path = MODEL_PATH / 'label_encoder.pkl'
    with open(label_encoder_path, 'rb') as f:
        le = pickle.load(f)
        
    return classifier, le

def main():
    st.title("😊 BERT Emotion Pipeline")
    st.markdown("""
    Deconstruct your text with AI. This app uses **DistilBERT** to detect the emotional nuances of your input.
    """)
    
    classifier_data = load_pipeline()
    
    if classifier_data is None:
        st.error("🚀 **Model not found!** Please run the training script first:\n`python bert_emotion.py` and select Option 1.")
        return
    
    classifier, le = classifier_data
    
    with st.container():
        user_input = st.text_area(
            "What's on your mind?", 
            placeholder="I am feeling surprisingly confident about this update...",
            height=150
        )
        
        analyze_btn = st.button("Analyze Emotion")

    if analyze_btn:
        if user_input.strip():
            with st.spinner("Decoding emotions..."):
                # Get scores for all labels
                results = classifier(user_input)[0]
                
                # Sort results by score descending
                results = sorted(results, key=lambda x: x['score'], reverse=True)
                
                top_result = results[0]
                top_label = top_result['label']
                top_score = top_result['score']
                
                # Human-readable mapping
                def get_clean_label(lbl):
                    if lbl.startswith('LABEL_'):
                        pred_id = int(lbl.split('_')[1])
                        return le.inverse_transform([pred_id])[0]
                    elif lbl in le.classes_:
                        return lbl
                    return lbl

                emotion = get_clean_label(top_label)
                
                # Results layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"Primary: **{emotion.capitalize()}**")
                    st.metric("Confidence", f"{top_score:.1%}")
                    
                    emojis = {
                        'sadness': '😢', 'joy': '😊', 'love': '❤️', 
                        'anger': '😠', 'fear': '😨', 'surprise': '😲', 'neutral': '😐'
                    }
                    icon = emojis.get(emotion.lower(), '✨')
                    st.markdown(f"<p style='font-size: 80px; text-align: center;'>{icon}</p>", unsafe_allow_html=True)

                with col2:
                    st.subheader("Emotion Breakdown")
                    for res in results[:5]:  # Show top 5
                        lbl = get_clean_label(res['label'])
                        val = res['score']
                        st.write(f"{lbl.capitalize()}")
                        st.progress(val)

        else:
            st.warning("Please enter some text to analyze.")

    # Sidebar info
    st.sidebar.image("https://img.icons8.com/clouds/100/000000/brain-storming.png", width=100)
    st.sidebar.title("Pipeline Details")
    st.sidebar.info("""
    **Model**: DistilBERT
    
    **Features**:
    - Negation Marking
    - Lemmatization
    - Augmented Neutral Logic
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### How it works")
    st.sidebar.write("1. **Prepare**: Merge and clean data.")
    st.sidebar.write("2. **Augment**: Add logic for negations.")
    st.sidebar.write("3. **Train**: Fine-tune Transformer.")
    st.sidebar.write("4. **Detect**: Infer emotion weights.")

if __name__ == "__main__":
    main()
