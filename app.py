import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from scipy.sparse import vstack

# Data preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Feature extraction
def extract_features(corpus, text):
    # Create a TF-IDF vectorizer and fit it on the entire corpus
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)

    # Transform the text
    features = vectorizer.transform([text])

    return features
# Generate dummy data for training
X, y = make_blobs(n_samples=100, n_features=54, centers=2, random_state=42)

# Convert X to text format (assuming it's not already text)
X_text = [" ".join(map(str, x)) for x in X]

# Extract features from the dummy data
X_features = [extract_features(X_text, text) for text in X_text]

X_features_stacked = vstack(X_features)

# Train the model
model = LogisticRegression()
model.fit(X_features_stacked, y)

def main():
    st.title("Automated Essay Scoring System")

    # User input for the essay
    essay = st.text_area("Enter your essay here:")

    # Preprocess the essay text
    preprocessed_text = preprocess_text(essay)

    # Extract features from the preprocessed text
    features = extract_features(X_text,preprocessed_text)

    if st.button("Score Essay"):
        if features is None:
            st.write("The essay text is too short or empty to be scored.")
        else:
            # Make a prediction using the pre-trained model
            score = model.predict(features)[0]

            # Display the score and feedback
            st.subheader(f"Essay Score: {score}")
            st.write("Feedback:")
            if score == 0:
                st.write("Your essay needs improvement. Please review the essay structure, grammar, and coherence.")
            else:
                st.write("Well done! Your essay meets the required standards.")

    # Additional sections for information or instructions
    st.subheader("About")
    st.write("This is an Automated Essay Scoring System that utilizes natural language processing techniques to evaluate and score essays based on a pre-trained model.")

if __name__ == "__main__":
    main()