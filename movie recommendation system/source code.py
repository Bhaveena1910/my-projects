
import streamlit as st
import pandas as pd
import base64
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ------------------------- MUST BE FIRST COMMAND -------------------------
st.set_page_config(page_title="Netflix Dashboard", layout="wide")

# ------------------------- Background Image -------------------------
def get_img_base64(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_img_base64("Your image path") #Attach link of your image path

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white !important;
    }}
    h1, h2, h3, h4, h5, h6, p, label {{
        color: white !important;
        text-shadow: 1px 1px 5px black;
    }}
    .stButton>button {{
        background-color: #ffffff20;
        color: white;
        font-weight: bold;
        border: 2px solid white;
        border-radius: 10px;
        padding: 15px 25px;
        font-size: 16px;
        text-shadow: 1px 1px 2px black;
        transition: all 0.2s ease-in-out;
    }}
    .stButton>button:hover {{
        background-color: #ffffff40;
        transform: scale(1.05);
    }}
    </style>
""", unsafe_allow_html=True)

# ------------------------- Load and Preprocess Data -------------------------
csv_file_path = r"Your csv dataset path" # Add netflix_final dataset path
try:
    df = pd.read_csv(csv_file_path)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()
df.dropna(subset=['release_year', 'rating', 'listed_in', 'type'], inplace=True)
df['genre'] = df['listed_in'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')
df['description'] = df['description'].fillna('')

# Encode features separately to avoid overwriting
le_rating = LabelEncoder()
le_genre = LabelEncoder()
le_country = LabelEncoder()

df['rating_encoded'] = le_rating.fit_transform(df['rating'])
df['genre_encoded'] = le_genre.fit_transform(df['genre'])
df['country_encoded'] = le_country.fit_transform(df['country'].astype(str))

# TF-IDF for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# KMeans Clustering
features = df[['genre_encoded', 'country_encoded', 'rating_encoded']]
kmeans = KMeans(n_clusters=5, n_init=10)
df['cluster'] = kmeans.fit_predict(features)

# ------------------------- Session State Setup -------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def go_to(page):
    st.session_state.page = page

# ------------------------- Home -------------------------
if st.session_state.page == 'Home':
    st.title("🎬 Netflix Recommendation & Analysis")
    st.header("Welcome to Netflix App")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Movies 🎥"):
            go_to("Movies")
    with col2:
        if st.button("TV Shows 📺"):
            go_to("TV Shows")
    with col3:
        if st.button("Recommendations 🌟"):
            go_to("Recommendations")
    with col4:
        if st.button("Top Picks 🔥"):
            go_to("Top Picks")

# ------------------------- Movies -------------------------
if st.session_state.page == "Movies":
    st.header("🎬 Top 5 Movies")
    movies = df[df['type'] == 'Movie']
    genre = st.selectbox("Select Genre", sorted(movies['genre'].unique()))
    year = st.selectbox("Select Year", sorted(movies['release_year'].astype(int).unique()))
    filtered = movies[(movies['genre'] == genre) & (movies['release_year'] == year)].head(5)  # Limit to top 5

    # Fit Decision Tree Classifier for predicting ratings (ID3)
    X = df[['genre_encoded', 'country_encoded']]
    y = df['rating_encoded']
    
    clf_id3 = DecisionTreeClassifier(criterion='entropy', random_state=0)
    clf_id3.fit(X, y)

    # Predict ratings for top 5 movies
    df['Predicted Rating'] = le_rating.inverse_transform(clf_id3.predict(X))

    def display_results(results):
        for idx, row in results.iterrows():
            st.markdown(f"""
            <div style='background-color:#1c1c1c; padding:15px; margin-bottom:15px; border-radius:10px;'>
                <h4 style='color:#ffffff;'>{row['title']}</h4>
                <p style='color:#808080;'>🎭 Genre: {row['genre']}</p>
                <p style='color:#808080;'>📅 Release Year: {row['release_year']}</p>
                <p style='color:#808080;'>⭐ Original Rating: {row['rating']}</p>
                <p style='color:#d3d3d3;'>🔮 Predicted Rating: {row['Predicted Rating']}</p>
            </div>
            """, unsafe_allow_html=True)

    if not filtered.empty:
        # Display top 5 movies
        filtered['Predicted Rating'] = df['Predicted Rating'].loc[filtered.index]
        display_results(filtered)
        
        # Find 5 similar movies using classification and clustering
        st.subheader("🎯 Similar Movies")
        for _, row in filtered.iterrows():
            similar_movies_classification = df[(df['genre_encoded'] == row['genre_encoded']) & 
                                               (df['country_encoded'] == row['country_encoded']) & 
                                               (df['title'] != row['title'])].head(5)

            similar_movies_clustering = df[(df['cluster'] == row['cluster']) & 
                                           (df['title'] != row['title'])].head(5)

            similar_movies = pd.concat([similar_movies_classification, similar_movies_clustering]).drop_duplicates().head(5)
            display_results(similar_movies)
    else:
        st.write("No movies found for the selected genre and year.")

    st.button("🔙 Return Home", on_click=lambda: go_to("Home"))

# ------------------------- TV Shows -------------------------
if st.session_state.page == "TV Shows":
    st.header("📺 Top 5 TV Shows")
    tv_shows = df[df['type'] == 'TV Show']
    genre = st.selectbox("Select Genre", sorted(tv_shows['genre'].unique()))
    year = st.selectbox("Select Year", sorted(tv_shows['release_year'].astype(int).unique()))
    filtered = tv_shows[(tv_shows['genre'] == genre) & (tv_shows['release_year'] == year)].head(5)  # Limit to top 5

    # Fit Decision Tree Classifier for predicting ratings (CART)
    X_tv = df[['genre_encoded', 'country_encoded']]
    y_tv = df['rating_encoded']
    
    clf_cart = DecisionTreeClassifier(criterion='gini', random_state=0)  # CART uses gini
    clf_cart.fit(X_tv, y_tv)

    # Predict ratings for top 5 TV shows
    df['Predicted Rating'] = le_rating.inverse_transform(clf_cart.predict(X_tv))

    def display_results(results):
        for idx, row in results.iterrows():
            st.markdown(f"""
            <div style='background-color:#1c1c1c; padding:15px; margin-bottom:15px; border-radius:10px;'>
                <h4 style='color:#ffffff;'>{row['title']}</h4>
                <p style='color:#808080;'>🎭 Genre: {row['genre']}</p>
                <p style='color:#808080;'>📅 Release Year: {row['release_year']}</p>
                <p style='color:#808080;'>⭐ Original Rating: {row['rating']}</p>
                <p style='color:#d3d3d3;'>🔮 Predicted Rating: {row['Predicted Rating']}</p>
            </div>
            """, unsafe_allow_html=True)

    if not filtered.empty:
        # Display top 5 TV shows
        filtered['Predicted Rating'] = df['Predicted Rating'].loc[filtered.index]
        display_results(filtered)

        # Find 5 similar TV shows using classification and clustering
        st.subheader("🎯 Similar TV Shows")
        for _, row in filtered.iterrows():
            similar_shows_classification = df[(df['genre_encoded'] == row['genre_encoded']) & 
                                              (df['country_encoded'] == row['country_encoded']) & 
                                              (df['title'] != row['title'])].head(5)

            similar_shows_clustering = df[(df['cluster'] == row['cluster']) & 
                                          (df['title'] != row['title'])].head(5)

            similar_shows = pd.concat([similar_shows_classification, similar_shows_clustering]).drop_duplicates().head(5)
            display_results(similar_shows)
    else:
        st.write("No TV shows found for the selected genre and year.")

    st.button("🔙 Return Home", on_click=lambda: go_to("Home"))
# ------------------------- Recommendations -------------------------
if st.session_state.page == "Recommendations":
    st.header("🌟 Recommendations")
    title_input = st.text_input("Enter a Movie/TV Show Title")
    country_input = st.selectbox("Select Country", sorted(df['country'].dropna().unique()))

    def get_content_recs(title, n=5):
        if title not in indices:
            return pd.DataFrame()
        idx = indices[title]
        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n + 1]
        return df.iloc[[i[0] for i in scores]]

    def get_collab_recs(title):
        row = df[df['title'] == title]
        if row.empty:
            return pd.DataFrame()
        feats = ['genre_encoded', 'country_encoded', 'rating_encoded']
        vec = row[feats].values[0].reshape(1, -1)
        cluster = kmeans.predict(vec)[0]
        return df[(df['cluster'] == cluster) & (df['title'] != title)].head(5)

    def display_results(results):
        for idx, row in results.iterrows():
            st.markdown(f"""
            <div style='background-color:#1c1c1c; padding:15px; margin-bottom:15px; border-radius:10px;'>
                <h4 style='color:#ffffff;'>{row['title']}</h4>
                <p style='color:#808080;'>🎭 Genre: {row['genre']}</p>
                <p style='color:#808080;'>📅 Release Year: {row['release_year']}</p>
                <p style='color:#d3d3d3;'>📝 Description: {row['description'][:150]}...</p>
            </div>
            """, unsafe_allow_html=True)

    if st.button("Get Recommendations"):
        if title_input:
            st.subheader("📌 Similar Movies/TV Shows")
            content = get_content_recs(title_input)
            if not content.empty:
                display_results(content)
            else:
                st.write("No similar movies found.")

            st.subheader("📌 Movies/TV Shows Others Watched")
            collab = get_collab_recs(title_input)
            if not collab.empty:
                display_results(collab)
            else:
                st.write("No collaborative recommendations found.")

            st.subheader("💡 You May Like")
            hybrid = pd.concat([content, collab]).drop_duplicates().head(5)
            if not hybrid.empty:
                display_results(hybrid)
            else:
                st.write("No hybrid recommendations found.")

            st.subheader(f"🌍 Popular Titles in {country_input}")
            map_data = df[df['country'] == country_input][['title', 'release_year', 'description', 'genre']].dropna().head(5)
            if not map_data.empty:
                # Generate fake latitude and longitude for visualization
                map_data['lat'] = 20 + (map_data['release_year'] % 10) * 2
                map_data['lon'] = 80 + (map_data['release_year'] % 10) * 2

                # Map visualization using Pydeck
                st.pydeck_chart(pdk.Deck(
                    initial_view_state=pdk.ViewState(latitude=30, longitude=100, zoom=2),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=map_data,
                            get_position='[lon, lat]',
                            get_color='[0, 128, 255, 160]',
                            get_radius=5000
                        ),
                        pdk.Layer(
                            'TextLayer',
                            data=map_data,
                            get_position='[lon, lat]',
                            get_text='title',
                            get_size=15,
                            get_color='[0, 128, 255]',
                            pickable=True
                        )
                    ]
                ))

                # Display results in a styled format under the map
                display_results(map_data)
            else:
                st.write("No popular titles found in the selected country.")

    st.button("🔙 Return Home", on_click=lambda: go_to("Home"))

# ------------------------- Top Picks -------------------------
if st.session_state.page == "Top Picks":
    st.header("🔥 Top Picks: Movies & TV Shows")
    top_movies = df[df['type'] == 'Movie'].head(5)
    top_shows = df[df['type'] == 'TV Show'].head(5)

    def display_recommendations(title, content_type):
        st.header(f"🎯 Recommendations similar to **{title}**")
        if 'cluster' in df.columns:
            cluster_val = df.loc[df['title'] == title, 'cluster'].values[0]
            recommendations = df[(df['type'] == content_type) & (df['cluster'] == cluster_val) & (df['title'] != title)].head(5)
        else:
            genre = df.loc[df['title'] == title, 'genre'].values[0]
            recommendations = df[(df['type'] == content_type) & (df['genre'] == genre) & (df['title'] != title)].head(5)

        if not recommendations.empty:
            for rec in recommendations.itertuples():
                st.markdown(f"""
                <div style='background-color:#1c1c1c; padding:15px; margin-bottom:15px; border-radius:10px;'>
                    <h4 style='color:#ffffff;'>{rec.title}</h4>
                    <p style='color:#808080;'>🎭 Genre: {rec.genre}</p>
                    <p style='color:#808080;'>📅 Release Year: {rec.release_year}</p>
                    <p style='color:#d3d3d3;'>📝 Description: {rec.description[:150]}...</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No similar items found.")

    st.subheader("🎬 Top 5 Movies")
    cols = st.columns(5)
    for idx, row in enumerate(top_movies.itertuples()):
        with cols[idx]:
            if st.button(f"🎥 {row.title}", key=f"movie_{idx}"):
                st.session_state.page = f"Recommendations_Movie_{row.title}"

    st.subheader("📺 Top 5 TV Shows")
    cols2 = st.columns(5)
    for idx, row in enumerate(top_shows.itertuples()):
        with cols2[idx]:
            if st.button(f"📺 {row.title}", key=f"show_{idx}"):
                st.session_state.page = f"Recommendations_Show_{row.title}"

    # Handle redirection to show recommendations
    if "Recommendations_Movie_" in st.session_state.page:
        movie_title = st.session_state.page.replace("Recommendations_Movie_", "")
        display_recommendations(movie_title, "Movie")
    elif "Recommendations_Show_" in st.session_state.page:
        show_title = st.session_state.page.replace("Recommendations_Show_", "")
        display_recommendations(show_title, "TV Show")
      
    st.button("🔙 Return Home", on_click=lambda: go_to("Home"))
