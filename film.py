import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import pickle
import os

# 📌 Charger les données avec st.cache
@st.cache
def load_data():
    # Télécharger les fichiers CSV depuis Google Drive
    url = 'https://drive.google.com/uc?id=1nh1DzJNRQry5pC74Nb0BSYn1nKazjDN-'
    output = 'reco_film.csv'
    gdown.download(url, output, quiet=False)
    
    df = pd.read_csv("final_fr.csv")
    test_scaled = pd.read_csv(output)
    df_final5 = pd.read_csv("df_final10.csv")
    
    return df, test_scaled, df_final5

# 📌 Charger les données
df, test_scaled, df_final5 = load_data()

# 🎯 Appliquer des pondérations pour améliorer la similarité
@st.cache
def adjust_weights(test_scaled):
    actors = test_scaled.loc[:, test_scaled.columns.str.contains('Actor|Actress')]
    directors = test_scaled.loc[:, test_scaled.columns.str.contains('Director')]
    production = test_scaled.loc[:, test_scaled.columns.str.contains('Production')]
    genres = test_scaled.loc[:, test_scaled.columns.str.contains('Genre')]

    test_scaled[actors.columns] *= 200
    test_scaled[directors.columns] *= 150
    test_scaled[production.columns] *= 90
    test_scaled[genres.columns] *= 300
    return test_scaled

# Appliquer les pondérations
test_scaled = adjust_weights(test_scaled)

# 📌 Calcul de la similarité cosinus avec st.cache
@st.cache
def compute_cosine_similarity(test_scaled):
    return cosine_similarity(test_scaled)

# Vérifier si le fichier de similarité existe déjà, sinon le calculer et le sauvegarder
similarity_file = "cosine_similarity.pkl"
if os.path.exists(similarity_file):
    with open(similarity_file, "rb") as f:
        cosine_test = pickle.load(f)
else:
    cosine_test = compute_cosine_similarity(test_scaled)
    with open(similarity_file, "wb") as f:
        pickle.dump(cosine_test, f)

# 📌 **Fonction pour récupérer les films similaires**
def similar_movies(movie_name, genre=None, actor=None):
    try:
        movie_index = df_final5[df_final5['titre_complet'].str.lower() == movie_name.lower()].index[0]
        similar_movies_idx = np.argsort(cosine_test[movie_index])[::-1][1:50]  # Limiter à 50 résultats
        similar_movies = df_final5.iloc[similar_movies_idx]
        
        # Fusionner avec df pour obtenir les informations (genres, acteurs, synopsis...)
        similar_movies = pd.merge(similar_movies, df[['titre_complet', 'genres_x_x', 'synopsis_fr', 'poster_path', 'primaryName']], on='titre_complet', how='left')

        if genre:
            similar_movies = similar_movies.loc[similar_movies['genres_x_x'].str.contains(genre, na=False, case=False)]
        if actor:
            similar_movies = similar_movies.loc[similar_movies['primaryName'].str.contains(actor, na=False, case=False)]

        # Gérer les affiches des films
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
    num_columns = 5
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

films = df['titre_complet'].tolist()
film_input = st.selectbox("Choisissez un film :", [""] + films)

# **Sélection d'un genre et d'un acteur (optionnel)**
genres_list = sorted(set(','.join(df['genres_x_x'].dropna()).split(',')))

# Extraction et nettoyage des acteurs
def clean_actor_name(actor):
    actor = re.sub(r"[\[\]\']+", "", actor)
    actor = actor.strip()
    return actor

actors_columns = test_scaled.columns[test_scaled.columns.str.contains('Actor|Actress')]
actors_list = [col.replace('Actor_', '').replace('Actress_', '') for col in actors_columns]
actors_list = [clean_actor_name(actor) for actor in actors_list]
actors_list = list(set(actors_list))
actors_list = actors_list[:100]

actor_input = st.selectbox("Filtrer par acteur/actrice (optionnel) :", [""] + actors_list)
actor_input_normalized = actor_input.replace("_", " ") 

# **Afficher les recommandations de films**
if film_input:
    st.header(f"📖 Détails du film : {film_input}")
    film_details = df[df['titre_complet'] == film_input].iloc[0]
    st.image("https://image.tmdb.org/t/p/w500" + film_details['poster_path'], width=200)
    st.write(f"**Genres :** {film_details['genres_x_x']}")
    st.write(f"**Synopsis :** {film_details['synopsis_fr']}")

    st.subheader("🎥 Films recommandés :")
    recommendations = similar_movies(film_input, actor=actor_input_normalized if actor_input_normalized else None)
    
    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            if 'primaryName' in row and pd.notna(row['primaryName']):
                actors_list = row['primaryName'].split(',')[:3]
                actors_str = ', '.join(actors_list).replace('[', '').replace(']', '').replace("'", "")
                st.subheader(row['titre_complet'])
                st.image(row['poster_path'], width=100)
                st.write(f"**Casting :** {actors_str}")
                st.write(f"**Genres :** {row['genres_x_x']}")
                st.write(f"**Synopsis :** {row['synopsis_fr']}")
                st.write("---")  # Ligne de séparation

else:
    show_top_movies()

