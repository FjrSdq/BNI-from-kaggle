import streamlit as st
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import os
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Wondr by BNI Dashboard", 
                   page_icon="📊", 
                   layout="wide")


# ENSURE MODEL IS LOADED
MODEL_PATH = "best_indobert_sentiment.pt"

def ensure_model():
    """Download model from Hugging Face if not present locally"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Hugging Face..."):
            try:
                hf_hub_download(
                    repo_id="FjrSdq/bni-sentiment-model",
                    filename=MODEL_PATH,
                    local_dir=".",
                    local_dir_use_symlinks=False
                )
                st.toast("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Error downloading model: {e}")
                st.stop()

# Call the function to ensure the model is available
ensure_model()

# CACHE FUNCTION
@st.cache_resource
def load_model():
    """Load the trained IndoBERT model"""
    tokenizer = AutoTokenizer.from_pretrained("mdhugol/indonesia-bert-sentiment-classification")

    # Set device separately
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "mdhugol/indonesia-bert-sentiment-classification",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    # Load trained weights
    state_dict = torch.load("best_indobert_sentiment.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

@st.cache_data
def load_data():
    """Load and preprocess CSV data"""
    df = pd.read_csv("wondr_sample_1500_with_label - Copy.csv")

    # Convert datetime column
    df['at'] = pd.to_datetime(df['at'])

    # Map labels to name
    label_map = {1: 'Negatif', 3: 'Positif'}
    df['label_name'] = df['label'].map(label_map)

    if 'content_clean' not in df.columns:
        df['content_clean'] = df['content']

    df = df.sort_values('at', ascending=False).reset_index(drop=True)

    return df

def predict_sentiment(text, tokenizer, model, device):
    """Predict sentiment for a single text"""
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1)

    # Map: 0 = Negatif, 1 = Positif
    sentiment = "Positif" if prediction.item() == 1 else "Negatif"
    confidence = probs[0][prediction].item()

    return sentiment, confidence


# LOAD DATA AND MODEL

if 'toasts_shown' not in st.session_state:
    st.session_state.toasts_shown = False

if not st.session_state.toasts_shown:
    with st.spinner("📊 Loading Data...."):
        df = load_data()
    st.toast(f"✅Loaded {len(df)} reviews")

    with st.spinner("🤖 Loading Model..."):
        tokenizer, model, device = load_model()
    st.toast(f"✅ Model ready on {str(device).upper()}")

    st.session_state.toasts_shown = True
else:
    df = load_data()
    tokenizer, model, device = load_model()

# SIDEBAR (FILTERS & PREDICTIONS)

with st.sidebar:
    st.title("⚙️ Sentiment Dashboard")
    st.markdown("----")

    st.header("📅 Date Filter")

    # Date range filter
    min_date = df['at'].min().date()
    max_date = df['at'].max().date()
    default_start = max_date - timedelta(days=30)
    default_end = max_date

    start_date = st.date_input(
        "Start Date",
        default_start,
        min_value=min_date,
        max_value=max_date
    )

    end_date = st.date_input(
        "End Date",
        default_end,
        min_value=min_date,
        max_value=max_date
    )

    st.markdown("----")
    st.header("🔍 Quick Prediction")

    user_input = st.text_area(
        "Enter a review:",
        placeholder = "e.g., Kono apuri subarashii desu!",
        height=100
    )

    if st.button("🔮Predict", type="primary"):
        if user_input:
            sentiment, confidence = predict_sentiment(user_input, tokenizer, model, device)

            if sentiment == "Positif":
                st.success(f"✅ **{sentiment}**")
            else:
                st.error(f"❌ **{sentiment}**")

            st.metric("Confidence",f"{confidence:.2%}")
        else:
            st.warning("Please enter some text")


# FILTER DATA

#Apply date filter
mask = (df['at'].dt.date >= start_date) & (df['at'].dt.date <= end_date)
df_filtered = df[mask].copy()

# MAIN CONTENT

st.title("📊 Sentiment Analysis = Wondr by BNI App Reviews")
st.markdown(f"Analyzing {len(df_filtered)} reviews from {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}*")
st.markdown("----")


# KEY METRICS

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        total = len(df_filtered)
        st.metric("Total Reviews", f"{total:,}")

with col2:
    with st.container(border=True):
        positif_count = len(df_filtered[df_filtered['label_name'] == 'Positif'])
        pct_positif = positif_count / total * 100 if total > 0 else 0
        st.metric(
            "Positif", 
            f"{positif_count:,} ({pct_positif:.1f}%")

with col3:
    with st.container(border=True):
        negatif_count = len(df_filtered[df_filtered['label_name'] == 'Negatif'])
        pct_negatif = negatif_count / total * 100 if total > 0 else 0
        st.metric(
            "Negatif", 
            f"{negatif_count:,} ({pct_negatif:.1f}%)")

with col4:
    with st.container(border=True):
        ratio = positif_count / negatif_count if negatif_count > 0 else 0
        st.metric("Positif/Negatif", f"{ratio:.2f}")

st.markdown("----")


# SENTIMENT DISTRIBUTION CHART

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📈 Sentiment Trend Over Time")

    # Add filter dropdown
    trend_filter = st.radio(
        "Show:",
        ["Both", "Positif Only", "Negatif Only"],
        horizontal=True,
        key="trend_filter"
    )

    # Group by date
    df_daily = df_filtered.groupby(df_filtered['at'].dt.date)['label_name'].value_counts().unstack(fill_value=0)

    # Ensure columns exist
    if 'Positif' not in df_daily.columns:
        df_daily['Positif'] = 0
    if 'Negatif' not in df_daily.columns:
        df_daily['Negatif'] = 0
    
    # Filter based on selection
    if trend_filter == "Positif Only":
        chart_data = df_daily[['Positif']]
        st.line_chart(chart_data, color='#29b5e8', use_container_width=True)
    elif trend_filter == "Negatif Only":
        chart_data = df_daily[['Negatif']]
        st.line_chart(chart_data, color='#D45B90', use_container_width=True)
    else:
        st.line_chart(df_daily, color=['#D45B90', '#29b5e8'], use_container_width=True)


with col2:
    st.subheader("📊Sentimen Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['Positif', 'Negatif']
    sizes = [positif_count, negatif_count]
    colors = ['#29b5e8', '#D45B90']
    explode = (0.05, 0.05)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Overall Sentiment Distribution', fontsize=12)
    st.pyplot(fig)

st.markdown("----")


# SENTIMENT BY TIME PERIOD (WEEK/MONTHLY)

st.subheader("📊 Sentiment Distribution by Period")

col1, col2 = st.columns(2)

with col1:
    df_weekly = df_filtered.copy()
    
    df_weekly['week'] = df_weekly['at'].dt.to_period('W').astype(str)
    weekly_summary = df_weekly.groupby('week')['label_name'].value_counts().unstack(fill_value=0)

    if not weekly_summary.empty:
        st.caption("Weekly Sentiment")
        # Calculate Percentages
        weekly_pct = weekly_summary.div(weekly_summary.sum(axis=1), axis=0) * 100
        weekly_pct = weekly_pct.tail(8) # Last 8 weeks
        st.bar_chart(weekly_pct[['Positif', 'Negatif']] if 'Negatif' in weekly_pct.columns else weekly_pct, color=['#29b5e8','#D45B90'])

with col2:
        # Monthly Sentiment
        df_monthly = df_filtered.copy()
        df_monthly['month'] = df_monthly['at'].dt.to_period('M').astype(str)
        monthly_summary = df_monthly.groupby('month')['label_name'].value_counts().unstack(fill_value=0)

        if not monthly_summary.empty:
            st.caption("Monthly Sentiment")
            monthly_pct = monthly_summary.div(monthly_summary.sum(axis=1), axis=0) * 100
            monthly_pct = monthly_pct.tail(12) # Last 12 months
            st.bar_chart(monthly_pct[['Positif', 'Negatif']] if 'Negatif' in monthly_pct.columns else monthly_pct, color=['#29b5e8','#D45B90'])

st.markdown("----")


# RECENT REVIEWS

with st.expander("📁 Recent Reviews", expanded=True):
    display_cols = ['reviewId', 'content', 'at', 'label_name']
    df_display = df_filtered[display_cols].copy()
    
    # Truncate long content
    df_display['content'] = df_display['content'].str[:100] + '....'

    # Color code labels
    df_display['label_name'] = df_display['label_name'].map({
        'Positif': '✅ Positif',
        'Negatif': '❌ Negatif'
    })

    st.dataframe(
        df_display.head(20),
        use_container_width=True,
        height=400,
        column_config={
            'reviewId':'Review ID',
            'content':'Review Content',
            'at':'Date',
            'label_name':'Sentiment'
        }
    )

st.markdown("----")

# BATCH PREDICTION

with st.expander("🤖 Batch Prediction (Upload CSV)", expanded=False):
    st.write("Upload a CSV file with a 'content' column to get sentiment predictions")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            if 'content' in batch_df.columns:
                st.write(f"✅ Found {len(batch_df)} reviews to analyze")

                if st.button("🚀 Predict All", type="primary"):
                    with st.spinner("Predicting Sentiments...."):
                        sentiments = []
                        confidences = []

                        progress_bar = st.progress(0)
                        for idx, row in batch_df.iterrows():
                            sentiment, confidence = predict_sentiment(
                                row['content'], tokenizer, model, device
                            )
                            sentiments.append(sentiment)
                            confidences.append(confidence)
                            progress_bar.progress((idx + 1) / len(batch_df))
                        
                        batch_df['predicted_sentiment'] = sentiments
                        batch_df['confidence'] = confidences

                        st.success("✅ Predictions complete!")
                        st.dataframe(batch_df, use_container_width=True)

                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="🤖 Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
            else:
                st.error("❌ CSV must contain a 'content' column")
        except Exception as e:
            st.error(f"Error: {e}")

# FOOTER

st.markdown("----")
st.caption(f"Built with Streamlit + IndoBERT | Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Binary Classification (Positif/Negatif)")