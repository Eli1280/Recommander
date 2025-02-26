pip install ms-recommenders
import streamlit as st
import pandas as pd
import numpy as np
from recommenders.models.sar.sar_singlenode import SAR
from recommenders.utils.timer import Timer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Simulating customer and product data
customers = pd.DataFrame({
    'id': list(range(1, 11)),
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace', 'Hannah', 'Ian', 'Jack'],
    'age': np.random.randint(20, 70, 10),
    'weight': np.random.randint(50, 100, 10),
    'height': np.random.randint(150, 200, 10),
    'condition': np.random.choice(['Chronic Pain', 'Anxiety', 'Insomnia', 'Epilepsy', 'Depression'], 10)
})

products = pd.DataFrame({
    'id': [101, 102, 103, 104],
    'name': ['CBD Oil', 'THC Gummies', 'Hybrid Vape', 'Indica Extract']
})

# Generating customer-product interactions
interactions = pd.DataFrame({
    'userID': np.random.choice(customers['id'], 30),
    'itemID': np.random.choice(products['id'], 30),
    'rating': np.random.randint(1, 6, 30),
    'timestamp': pd.to_datetime('now')
})

# Creating feature-based similarity matrix
features = customers[['age', 'weight', 'height']].copy()
features = (features - features.min()) / (features.max() - features.min())  # Normalize
user_similarities = cosine_similarity(features)
similarity_df = pd.DataFrame(user_similarities, index=customers['id'], columns=customers['id'])

# Initializing the SAR model (Smart Adaptive Recommendations)
sar = SAR(col_user='userID', col_item='itemID', col_rating='rating', col_timestamp='timestamp')
with Timer():
    sar.fit(interactions)

# Streamlit interface
st.title("Medical Cannabis Product Recommendation")

# Display customer details
st.subheader("Customer Profiles")
st.dataframe(customers)

# Display similarity matrix
st.subheader("Customer Similarity Matrix (Based on Age, Weight, Height)")
st.dataframe(similarity_df)

# Heatmap visualization
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(similarity_df, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=customers['name'], yticklabels=customers['name'])
st.pyplot(fig)

customer_name = st.selectbox("Select your name", customers['name'])

if st.button("Get Recommendation"):
    customer_id = customers[customers['name'] == customer_name]['id'].values[0]
    similar_customers = similarity_df[customer_id].sort_values(ascending=False).index[1:3]  # Top 2 similar customers
    recommended_products = []
    
    for similar_customer in similar_customers:
        reco = sar.recommend(similar_customer, top_k=1)
        if not reco.empty:
            recommended_products.append(reco.iloc[0]['itemID'])
    
    if recommended_products:
        recommended_products = list(set(recommended_products))  # Remove duplicates
        product_names = [products[products['id'] == pid]['name'].values[0] for pid in recommended_products]
        st.success(f"Recommended Products based on similar customers: {', '.join(product_names)}")
    else:
        st.warning("Not enough data to recommend a product.")
