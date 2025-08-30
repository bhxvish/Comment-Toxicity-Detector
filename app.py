import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd

# Load the trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('toxicity_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# --- Streamlit UI ---
st.title("💬 Comment Toxicity Detection")
st.markdown("A deep learning model to detect toxic comments in real-time. [cite: 29]")

# --- Single Comment Prediction ---
st.header("Real-time Prediction")
user_input = st.text_area("Enter a comment to analyze:", "This is a wonderful and positive comment!")

if st.button("Analyze"):
    if user_input:
        # Preprocess the input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
        
        # Make prediction
        prediction = model.predict(padded_sequence)[0][0]
        
        # Display result
        st.subheader("Analysis Result")
        if prediction > 0.5:
            st.error(f"Toxic Comment Detected (Confidence: {prediction*100:.2f}%)")
            st.write("This comment may contain harassment, hate speech, or offensive language. [cite: 4]")
        else:
            st.success(f"Not a Toxic Comment (Confidence: {(1-prediction)*100:.2f}%)")
            st.write("This comment appears to be constructive and respectful. [cite: 4]")
    else:
        st.warning("Please enter a comment to analyze.")

# --- Bulk Prediction via CSV Upload --- [cite: 27]
st.header("Upload a CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Choose a CSV file with a 'comment_text' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'comment_text' in df.columns:
        # Preprocess and predict for each comment
        comments = df['comment_text'].tolist()
        sequences = tokenizer.texts_to_sequences(comments)
        padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
        predictions = model.predict(padded)
        
        # Add predictions to the dataframe
        df['is_toxic'] = ['Yes' if p > 0.5 else 'No' for p in predictions]
        df['toxicity_score'] = predictions
        
        st.subheader("Bulk Prediction Results")
        st.dataframe(df)
        
        # Provide a download link for the results
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(df)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='toxicity_predictions.csv',
            mime='text/csv',
        )
    else:
        st.error("The uploaded CSV must have a column named 'comment_text'.")