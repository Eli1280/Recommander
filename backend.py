pip install scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Simuler les données des patients
customers = pd.DataFrame({
    'id': list(range(1, 11)),
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace', 'Hannah', 'Ian', 'Jack'],
    'age': np.random.randint(20, 70, 10),
    'weight': np.random.randint(50, 100, 10),
    'height': np.random.randint(150, 200, 10),
    'condition': np.random.choice(['Chronic Pain', 'Anxiety', 'Insomnia', 'Epilepsy', 'Depression'], 10)
})

# Simuler les produits
products = pd.DataFrame({
    'id': [101, 102, 103, 104],
    'name': ['CBD Oil', 'THC Gummies', 'Hybrid Vape', 'Indica Extract']
})

# Simuler les interactions (achats)
interactions = pd.DataFrame({
    'userID': np.random.choice(customers['id'], 30),
    'itemID': np.random.choice(products['id'], 30),
    'rating': np.random.randint(1, 6, 30),
    'timestamp': pd.to_datetime('now')
})

# Encodage des conditions médicales en nombres
condition_mapping = {cond: idx for idx, cond in enumerate(customers['condition'].unique())}
customers['condition_encoded'] = customers['condition'].map(condition_mapping)

# Calculer la similarité des patients (basée sur l'âge, le poids, la taille et la condition)
features = customers[['age', 'weight', 'height', 'condition_encoded']]
features = (features - features.min()) / (features.max() - features.min())  # Normalisation
user_similarities = cosine_similarity(features)
similarity_df = pd.DataFrame(user_similarities, index=customers['id'], columns=customers['id'])

# Streamlit Interface
st.title("Medical Cannabis Product Recommendation")

# Afficher les profils des clients
st.subheader("Customer Profiles")
st.dataframe(customers[['name', 'age', 'weight', 'height', 'condition']])

# Afficher la matrice de similarité
st.subheader("Customer Similarity Matrix")
st.dataframe(similarity_df)

# Visualisation avec un heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(similarity_df, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=customers['name'], yticklabels=customers['name'])
st.pyplot(fig)

# Sélection du client
customer_name = st.selectbox("Select your name", customers['name'])

if st.button("Get Recommendation"):
    # Trouver l'ID du client sélectionné
    customer_id = customers[customers['name'] == customer_name]['id'].values[0]
    
    # Trouver les patients les plus similaires
    similar_customers = similarity_df[customer_id].sort_values(ascending=False).index[1:3]
    
    recommended_products = []
    
    for similar_customer in similar_customers:
        # Rechercher les produits achetés par les patients similaires
        purchased_products = interactions[interactions['userID'] == similar_customer]['itemID'].tolist()
        recommended_products.extend(purchased_products)

    # Supprimer les doublons et récupérer les noms des produits
    recommended_products = list(set(recommended_products))
    product_names = [products[products['id'] == pid]['name'].values[0] for pid in recommended_products]

    # Afficher les recommandations
    if recommended_products:
        st.success(f"Recommended Products based on similar customers: {', '.join(product_names)}")
    else:
        st.warning("Not enough data to recommend a product.")
