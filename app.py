# ============================================================================
# APPLICATION STREAMLIT - PRÉVISION DES VENTES
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Prévision des ventes",
    page_icon="📊",
    layout="wide"
)

# Titre
st.title("📊 Prévision des ventes journalières")
st.markdown("Modèle Random Forest entraîné sur les données Superstore (2014-2017)")

# Chargement du modèle
@st.cache_resource
def charger_modele():
    # Chemin relatif (plus fiable)
    chemin_modele = os.path.join('models', 'modele_ventes.pkl')
    
    # Vérifier si le fichier existe
    if os.path.exists(chemin_modele):
        return joblib.load(chemin_modele)
    else:
        st.error(f"Fichier non trouvé : {chemin_modele}")
        return None

model = charger_modele()

if model is None:
    st.error("❌ Modèle non trouvé. Vérifie que 'models/modele_ventes.pkl' existe.")
    st.stop()
else:
    st.success("✅ Modèle chargé avec succès")

# Sidebar pour les paramètres
st.sidebar.header("📅 Paramètres de prédiction")

# Date de prédiction
date_pred = st.sidebar.date_input("Date de prédiction", datetime.today())

# Extraction des features depuis la date
mois = date_pred.month
jour_semaine = date_pred.weekday()  # 0=lundi, 6=dimanche
weekend = 1 if jour_semaine >= 5 else 0

# Lags (à renseigner manuellement pour la démo)
st.sidebar.subheader("Valeurs passées (lags)")
lag_1 = st.sidebar.number_input("Ventes J-1 (€)", value=1500.0, step=100.0)
lag_7 = st.sidebar.number_input("Ventes J-7 (€)", value=1500.0, step=100.0)
lag_14 = st.sidebar.number_input("Ventes J-14 (€)", value=1500.0, step=100.0)

# Bouton de prédiction
if st.sidebar.button("🔮 Prédire les ventes", type="primary"):
    # Préparation des features
    features = pd.DataFrame([[
        mois, jour_semaine, weekend, lag_1, lag_7, lag_14
    ]], columns=['mois', 'jour_semaine', 'weekend', 'lag_1', 'lag_7', 'lag_14'])
    
    # Prédiction
    prediction = model.predict(features)[0]
    
    # Affichage
    st.subheader("📈 Résultat de la prédiction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Date", date_pred.strftime("%d/%m/%Y"))
    with col2:
        st.metric("Ventes prédites", f"{prediction:,.0f} €")
    with col3:
        st.metric("Intervalle de confiance", f"± 1 500 €")
    
    # Jauge visuelle
    st.subheader("📊 Visualisation")
    st.progress(min(1.0, prediction / 20000))
    
    # Informations sur les features
    with st.expander("🔍 Détails des features utilisées"):
        st.write(f"- **Mois** : {mois}")
        st.write(f"- **Jour semaine** : {jour_semaine} (0=lundi, 6=dimanche)")
        st.write(f"- **Weekend** : {'Oui' if weekend else 'Non'}")
        st.write(f"- **Ventes J-1** : {lag_1:,.0f} €")
        st.write(f"- **Ventes J-7** : {lag_7:,.0f} €")
        st.write(f"- **Ventes J-14** : {lag_14:,.0f} €")

# Section d'information
with st.expander("ℹ️ À propos du modèle"):
    st.write("""
    - **Algorithme** : Random Forest
    - **MAE** : 1 491 €
    - **RMSE** : 2 294 €
    - **Features principales** : lag_7 (29%), mois (26%), lag_14 (18%)
    - **Période d'entraînement** : 2014-2016
    - **Période de test** : 2017
    """)

st.markdown("---")
st.caption("Projet de prévision des ventes - Modèle Random Forest")