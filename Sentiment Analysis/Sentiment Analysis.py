# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import nltk
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
import json

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Check and set up GPU
logging.info("Checking GPU availability...")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logging.info(f"Using GPU: {physical_devices[0].name}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    logging.info("No GPU found, using CPU.")

# Load the dataset
try:
    dataset = load_dataset("munawwarsultan2017/US_Presidential_Election_2020_Dem_Rep")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    raise

df = pd.DataFrame(dataset['train'])
df_sampled = df.sample(frac=0.025, random_state=42)
df_sampled['text_clean'].fillna('', inplace=True)

# Further text preprocessing
def clean_text(text):
    import re
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text

df_sampled['text_clean'] = df_sampled['text_clean'].apply(clean_text)

# Prepare text data and labels
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df_sampled['text_clean']).toarray()
le = LabelEncoder()
y = le.fit_transform(df_sampled['Vader_Sentiment'])
y_encoded = to_categorical(y)

X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
    X, y_encoded, df_sampled['text_clean'], test_size=0.2, random_state=42
)

# Define and compile the neural network
def create_model(learning_rate=0.001):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model = create_model()
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
f1_custom = f1_score(y_true_classes, y_pred_classes, average='weighted')
cm_custom = confusion_matrix(y_true_classes, y_pred_classes)
mcc_custom = matthews_corrcoef(y_true_classes, y_pred_classes)

# Sentiment analysis pipelines
logging.info("Loading sentiment analysis models...")
distilbert = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
bertweet = pipeline('sentiment-analysis', model='vinai/bertweet-base')
sia = SentimentIntensityAnalyzer()

distilbert_preds = [1 if res['label'] == 'POSITIVE' else 0 for res in distilbert(texts_test.tolist())]
bertweet_preds = [1 if res['label'] == 'LABEL_2' else 0 for res in bertweet(texts_test.tolist())]
vader_preds = [1 if sia.polarity_scores(text)['compound'] > 0.05 else 0 for text in texts_test.tolist()]

# Store results
results = {
    'Custom Neural Network': {'F1': f1_custom, 'CM': cm_custom.tolist(), 'MCC': mcc_custom},
    'DistilBERT': {'F1': f1_score(y_true_classes, distilbert_preds, average='weighted'),
                   'CM': confusion_matrix(y_true_classes, distilbert_preds).tolist(),
                   'MCC': matthews_corrcoef(y_true_classes, distilbert_preds)},
    'BERTweet': {'F1': f1_score(y_true_classes, bertweet_preds, average='weighted'),
                 'CM': confusion_matrix(y_true_classes, bertweet_preds).tolist(),
                 'MCC': matthews_corrcoef(y_true_classes, bertweet_preds)},
    'VADER': {'F1': f1_score(y_true_classes, vader_preds, average='weighted'),
              'CM': confusion_matrix(y_true_classes, vader_preds).tolist(),
              'MCC': matthews_corrcoef(y_true_classes, vader_preds)}
}

# Save results to JSON
with open("results.json", "w") as file:
    json.dump(results, file, indent=4)

# Plot comparison results
def plot_comparison(results, metric_key, title):
    labels = list(results.keys())
    if metric_key == 'CM':
        for label, value in results.items():
            plt.matshow(value[metric_key], cmap='viridis')
            plt.title(label)
            plt.colorbar()
            plt.show()
    else:
        values = [results[model][metric_key] for model in labels]
        plt.bar(labels, values, color='blue')
        plt.title(title)
        plt.ylabel(metric_key)
        plt.xticks(rotation=45)
        plt.show()

# Display all results
for model_name, metrics in results.items():
    logging.info(f"\n{model_name} Scores:")
    logging.info(f"F1 Score: {metrics['F1']:.4f}")
    logging.info(f"Matthews Correlation Coefficient: {metrics['MCC']:.4f}")
    logging.info(f"Confusion Matrix: {metrics['CM']}")

plot_comparison(results, 'F1', 'F1 Score Comparison')
plot_comparison(results, 'MCC', 'MCC Comparison')
