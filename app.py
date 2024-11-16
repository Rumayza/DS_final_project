
# Import necessary libraries
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Step 1: Data Preprocessing

# 1. Data Import - Load CSV and JSONL data
magazine_csv_path = 'Magazine_Subscriptions_Mini.csv'
meta_magazine_jsonl_path = 'meta_Magazine_Subscriptions.jsonl'
user_history_path = 'user_history.csv'

# Load CSV data
magazine_csv_df = pd.read_csv(magazine_csv_path)

# Load JSONL data
meta_magazine_data = []
with open(meta_magazine_jsonl_path, 'r') as file:
    meta_magazine_data = [json.loads(line) for line in file]
meta_magazine_df = pd.DataFrame(meta_magazine_data)

# 2. Data Exploration - Displaying basic info and checking for missing values
print("Magazine CSV Data Overview:")
print(magazine_csv_df.info())
print("Magazine Metadata JSONL Data Overview:")
print(meta_magazine_df.info())

# 3. Data Cleaning
# Identify columns with unhashable types and exclude them from deduplication
unhashable_columns = [col for col in meta_magazine_df.columns if meta_magazine_df[col].apply(lambda x: isinstance(x, (list, dict))).any()]
hashable_columns = [col for col in meta_magazine_df.columns if col not in unhashable_columns]

# Drop duplicates in hashable columns only
meta_magazine_df.drop_duplicates(subset=hashable_columns, inplace=True)
magazine_csv_df.drop_duplicates(inplace=True)

# Handle missing values in essential columns
magazine_csv_df.dropna(subset=['user_id', 'asin', 'rating'], inplace=True)
meta_magazine_df.dropna(subset=['title', 'parent_asin'], inplace=True)

# 4. Feature Engineering - Encode Text Features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
meta_magazine_df['description_text'] = meta_magazine_df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
tfidf_matrix = tfidf_vectorizer.fit_transform(meta_magazine_df['description_text'])

# Create a dictionary to map ASINs to magazine titles
asin_to_title = meta_magazine_df.set_index('parent_asin')['title'].to_dict()

# Step 2: Collaborative Filtering using SVD

# Prepare Interaction Matrix
interaction_data = magazine_csv_df[['user_id', 'asin', 'rating']].dropna()
interaction_matrix = interaction_data.pivot(index='user_id', columns='asin', values='rating').fillna(0)
sparse_interaction_matrix = csr_matrix(interaction_matrix.values)

# Initialize and fit SVD model
svd_model = TruncatedSVD(n_components=10)
latent_factors = svd_model.fit_transform(sparse_interaction_matrix)

# Train-Test Split
train_data, test_data = train_test_split(interaction_data, test_size=0.2, random_state=42)

# Generate collaborative recommendations (Top-N items per user)
def get_collaborative_recommendations(user_id, num_recommendations=10):
    user_index = interaction_matrix.index.get_loc(user_id)
    user_ratings = latent_factors[user_index]
    similar_users = np.argsort(-user_ratings)[:num_recommendations]
    recommendations = interaction_matrix.columns[similar_users].tolist()
    # Map ASINs to titles for display
    return [asin_to_title.get(asin, asin) for asin in recommendations]

# Step 3: Content-Based Filtering

# Calculate Cosine Similarity on TF-IDF matrix for content-based filtering
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Generate content-based recommendations
def get_content_based_recommendations(asin, num_recommendations=10):
    item_index = meta_magazine_df[meta_magazine_df['parent_asin'] == asin].index[0]
    similar_items = np.argsort(-cosine_sim_matrix[item_index])[:num_recommendations]
    recommended_asins = meta_magazine_df.iloc[similar_items]['parent_asin'].tolist()
    # Map ASINs to titles for display
    return [asin_to_title.get(asin, asin) for asin in recommended_asins]

# Step 4: Hybrid Recommendation System

# Generate hybrid recommendations
def get_hybrid_recommendations(user_id, asin, num_recommendations=10, weight_cf=0.5, weight_cb=0.5):
    cf_recommendations = get_collaborative_recommendations(user_id, num_recommendations)
    cb_recommendations = get_content_based_recommendations(asin, num_recommendations)
    combined_recommendations = list(set(cf_recommendations + cb_recommendations))
    return combined_recommendations[:num_recommendations]

# Step 6: Streamlit Application Development with User Ratings and History

import streamlit as st

# Load user history if it exists
if os.path.exists(user_history_path):
    user_history_df = pd.read_csv(user_history_path)
else:
    user_history_df = pd.DataFrame(columns=['title', 'rating'])

# Streamlit App
st.title("Magazine Subscription Recommendation System")

st.sidebar.header("User Options")
# Display labels for User IDs
user_options = {user_id: f"User {idx + 1}" for idx, user_id in enumerate(interaction_matrix.index)}
asin_options = {asin: asin_to_title.get(asin, asin) for asin in meta_magazine_df['parent_asin'].unique()}
selected_user_id = st.sidebar.selectbox("Select User", options=user_options.keys(), format_func=user_options.get)
selected_asin = st.sidebar.selectbox("Select Magazine", options=asin_options.keys(), format_func=asin_options.get)

# New preference section
st.sidebar.header("Your Preferences")
preferred_genres = st.sidebar.multiselect("Select preferred genres (if available)", meta_magazine_df['title'].unique())
preferred_keywords = st.sidebar.text_input("Enter keywords you like (comma separated)", "nature, technology")

# Generate Recommendations
if st.sidebar.button("Generate Recommendations"):
    # Retrieve all three types of recommendations
    collaborative_recommendations = get_collaborative_recommendations(selected_user_id)
    content_recommendations = get_content_based_recommendations(selected_asin)
    hybrid_recommendations = get_hybrid_recommendations(selected_user_id, selected_asin)

    # Initialize session state for storing ratings
    if 'ratings' not in st.session_state:
        st.session_state.ratings = {}

    # Display collaborative recommendations with sliders
    st.subheader("Collaborative Filtering Recommendations")
    for idx, title in enumerate(collaborative_recommendations):
        st.session_state.ratings[title] = st.slider(f"How would you rate '{title}'?", 1, 5, st.session_state.ratings.get(title, 3), key=f"cf_{idx}_{title}")

    # Display content-based recommendations with sliders
    st.subheader("Content-Based Recommendations")
    for idx, title in enumerate(content_recommendations):
        st.session_state.ratings[title] = st.slider(f"How would you rate '{title}'?", 1, 5, st.session_state.ratings.get(title, 3), key=f"cb_{idx}_{title}")

    # Display hybrid recommendations with sliders
    st.subheader("Hybrid Recommendations")
    for idx, title in enumerate(hybrid_recommendations):
        st.session_state.ratings[title] = st.slider(f"How would you rate '{title}'?", 1, 5, st.session_state.ratings.get(title, 3), key=f"hybrid_{idx}_{title}")

    # Button to save ratings
    if st.button("Save Ratings"):
        new_ratings = pd.DataFrame(list(st.session_state.ratings.items()), columns=['title', 'rating'])
        user_history_df = pd.concat([user_history_df, new_ratings], ignore_index=True)
        user_history_df.to_csv(user_history_path, index=False)
        st.success("Your ratings have been saved!")

    # Calculate and display metrics
    st.subheader("Evaluation Metrics")
    # Assuming we have binary relevance for evaluation (1 if the user interacted with the item, 0 otherwise)
    y_true = np.zeros(len(test_data))
    y_pred = np.zeros(len(test_data))

    for i, row in test_data.iterrows():
        if row['asin'] in collaborative_recommendations:
            y_true[i] = 1  # User interacted with this ASIN
        if row['asin'] in collaborative_recommendations:  # For simplicity, let's predict based on collaborative recommendations
            y_pred[i] = 1

    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

# Placeholder for displaying personalized recommendations based on history
st.subheader("Personalized Recommendations Based on Your History")
st.write("This section will use your saved ratings to personalize future recommendations.")
 