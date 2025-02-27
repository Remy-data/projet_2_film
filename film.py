import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 📌 Charger les données
url = 'https://drive.google.com/uc?id=1nh1DzJNRQry5pC74Nb0BSYn1nKazjDN-'
df = pd.read_csv("final_fr.csv")
test_scaled = pd.read_csv(url)
df_final5 = pd.read_csv("final10.csv")

# 🎨 Appliquer un fond d'écran
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://img.freepik.com/photos-premium/elements-cinema-fond-violet-espace-copie_23-2148416765.jpg");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            color: white;
            min-height: 100vh; 
            height: 100vh;
            overflow: hidden;  
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Appliquer des pondérations pour améliorer la similarité
actors = test_scaled.loc[:, test_scaled.columns.str.contains('Actor|Actress')]
directors = test_scaled.loc[:, test_scaled.columns.str.contains('Director')]
production = test_scaled.loc[:, test_scaled.columns.str.contains('Production')]
genres = test_scaled.loc[:, test_scaled.columns.str.contains('Genre')]

test_scaled[actors.columns] *= 200
test_scaled[directors.columns] *= 150
test_scaled[production.columns] *= 90
test_scaled[genres.columns] *= 300

# 🔍 Calcul de la similarité cosinus
cosine_test = cosine_similarity(test_scaled)

# 📌 **Fonction pour récupérer les films similaires**
def similar_movies(movie_name, genre=None, actor=None):
    try:
        # Trouver l’indice du film sélectionné
        movie_index = df_final5[df_final5['titre_complet'].str.lower() == movie_name.lower()].index[0]

        # Récupérer les indices des films similaires
        similar_movies_idx = np.argsort(cosine_test[movie_index])[::-1][1:5000]  # Prendre un grand nombre avant de filtrer

        # Récupérer les films correspondants
        similar_movies = df_final5.iloc[similar_movies_idx]

        # Fusionner avec df pour obtenir les informations (genres, acteurs, synopsis...)
        similar_movies = pd.merge(similar_movies, df[['titre_complet', 'genres_x_x', 'synopsis_fr', 'poster_path', 'primaryName']], on='titre_complet', how='left')

        # 🔹 Filtrage par genre (si sélectionné)
        if genre:
            similar_movies = similar_movies.loc[similar_movies['genres_x_x'].str.contains(genre, na=False, case=False)]

        # 🔹 Filtrage par acteur (si sélectionné)
        if actor:
            similar_movies = similar_movies.loc[similar_movies['primaryName'].str.contains(actor, na=False, case=False)]

        # 🔍 Gérer les affiches des films
        base_url = "https://image.tmdb.org/t/p/w500"
        similar_movies['poster_path'] = similar_movies['poster_path'].apply(lambda x: base_url + x if pd.notna(x) and x != '' else "https://example.com/defaut_image.jpg")

        return similar_movies.head(10)[['titre_complet', 'genres_x_x', 'synopsis_fr', 'poster_path', 'primaryName']]

    except Exception as e:
        st.error(f"⚠ Erreur lors de la récupération des films similaires : {e}")
        return pd.DataFrame()

# 📌 **Fonction pour afficher le Top 10 des films les mieux notés**
def show_top_movies():
    st.header("🍿 Top 10 des films les mieux notés")

    top_films = df.sort_values(by='note_ponderee_x', ascending=False).head(10)
    base_url = "https://image.tmdb.org/t/p/w500"

    num_columns = 5  # Nombre de colonnes d'affichage
    for i in range(0, len(top_films), num_columns):
        films_group = top_films.iloc[i:i+num_columns]
        cols = st.columns(num_columns)
        for j, (_, film) in enumerate(films_group.iterrows()):
            with cols[j]:
                st.image(base_url + film['poster_path'], width=100)
                st.write(f"**{film['titre_complet']}**")
                st.write(f"⭐ {round(film['note_ponderee_x'], 2)}")

# 🎬 **Interface principale**
st.title("🎥 On regarde quoi ce soir ?")

# 🔹 **Sélection d'un film**
films = df['titre_complet'].tolist()
film_input = st.selectbox("Choisissez un film :", [""] + films)

# 🔹 **Sélection d'un genre et d'un acteur (optionnel)**
genres_list = sorted(set(','.join(df['genres_x_x'].dropna()).split(',')))  # Extraire tous les genres possibles
import re

# Fonction pour nettoyer les acteurs
def clean_actor_name(actor):
    # Retirer les crochets, les apostrophes et les espaces superflus
    actor = re.sub(r"[\[\]\']+", "", actor)  # Enlever les crochets et les apostrophes
    actor = actor.strip()  # Enlever les espaces avant et après le nom
    return actor

# 🔹 **Extraction des acteurs possibles, nettoyage et suppression des doublons**
actors_columns = test_scaled.columns[test_scaled.columns.str.contains('Actor|Actress')]

# Extraire les noms des acteurs à partir de ces colonnes
actors_list = [col.replace('Actor_', '').replace('Actress_', '') for col in actors_columns]

# Nettoyer les acteurs (enlever les caractères indésirables) et limiter à 50 acteurs
actors_list = [clean_actor_name(actor) for actor in actors_list]  # Nettoyer chaque acteur
actors_list = list(set(actors_list))  # Supprimer les doublons
actors_list = actors_list[:100]  # Limiter à 50 acteurs

# Affichage du selectbox pour choisir un acteur
actor_input = st.selectbox("Filtrer par acteur/actrice (optionnel) :", [""] + actors_list)
actor_input_normalized = actor_input.replace("_", " ") 

# 📌 **Afficher les recommandations de films**
if film_input:
    # Affichage des détails du film sélectionné
    st.header(f"📖 Détails du film : {film_input}")
    film_details = df[df['titre_complet'] == film_input].iloc[0]

    st.image("https://image.tmdb.org/t/p/w500" + film_details['poster_path'], width=200)
    st.write(f"**Genres :** {film_details['genres_x_x']}")
    st.write(f"**Synopsis :** {film_details['synopsis_fr']}")

    # Afficher les 10 films recommandés avec les filtres sélectionnés
    st.subheader("🎥 Films recommandés :")
    recommendations = similar_movies(
        movie_name=film_input,
        
        actor=actor_input_normalized if actor_input_normalized else None
    )

    # 🔹 **Affichage des recommandations**
    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            if 'primaryName' in row and pd.notna(row['primaryName']):
                actors_list = row['primaryName'].split(',')[:3]  # Prendre max 3 acteurs
                actors_str = ', '.join(actors_list).replace('[', '').replace(']', '').replace("'", "")

                # 🔥 **Affichage des films recommandés**
                st.subheader(row['titre_complet'])
                st.image(row['poster_path'], width=100)  
                st.write(f"**Casting :** {actors_str}")
                st.write(f"**Genres :** {row['genres_x_x']}")
                st.write(f"**Synopsis :** {row['synopsis_fr']}")
                st.write("---")  # Ligne de séparation

# 🎯 **Si aucun film sélectionné, afficher le Top 10**
else:
    show_top_movies()
