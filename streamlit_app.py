from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("dataset/netflix_titles.csv")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_PATH = CACHE_DIR / "netflix_all_minilm_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["title"] = df["title"].fillna("Unknown")
    df["listed_in"] = df["listed_in"].fillna("")
    df["description"] = df["description"].fillna("")
    df["type"] = df["type"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("Unknown")
    df["combined_features"] = (
        "Title: " + df["title"] + ". "
        + "Genre: " + df["listed_in"] + ". "
        + "Description: " + df["description"]
    )
    return df


@st.cache_data(show_spinner=False)
def load_or_build_embeddings(text_list):
    if EMBEDDING_PATH.exists():
        return np.load(EMBEDDING_PATH)

    model = load_model()
    embeddings = model.encode(
        text_list,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    np.save(EMBEDDING_PATH, embeddings)
    return embeddings


def semantic_search(query, df, embeddings, top_n=10, min_score=0.2, type_filter="All"):
    model = load_model()
    query_vector = model.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_vector, embeddings)[0]
    ranked_idx = scores.argsort()[::-1]

    rows = []
    for idx in ranked_idx:
        score = float(scores[idx])
        if score < min_score:
            continue

        item_type = str(df.iloc[idx]["type"]) if "type" in df.columns else "Unknown"
        if type_filter != "All" and item_type != type_filter:
            continue

        rows.append(
            {
                "rank": len(rows) + 1,
                "title": df.iloc[idx]["title"],
                "type": item_type,
                "release_year": df.iloc[idx].get("release_year", None),
                "rating": df.iloc[idx].get("rating", None),
                "listed_in": df.iloc[idx].get("listed_in", ""),
                "similarity_score": round(score, 4),
                "description_preview": (
                    (str(df.iloc[idx].get("description", ""))[:140] + "...")
                    if len(str(df.iloc[idx].get("description", ""))) > 140
                    else str(df.iloc[idx].get("description", ""))
                ),
            }
        )

        if len(rows) >= top_n:
            break

    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Netflix Semantic Search", layout="wide")
    st.title("Netflix Semantic Search Demo")
    st.caption("Type a natural language query to find similar Netflix titles.")

    with st.sidebar:
        st.header("Search Settings")
        top_n = st.slider("Top N Results", min_value=3, max_value=20, value=10, step=1)
        min_score = st.slider("Minimum Similarity Score", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        type_filter = st.selectbox("Content Type", options=["All", "Movie", "TV Show"], index=0)

    df = load_data()
    with st.spinner("Preparing embeddings..."):
        embeddings = load_or_build_embeddings(df["combined_features"].tolist())

    query = st.text_input(
        "Query",
        placeholder="Example: dark fantasy with magic and mystery",
    )

    col_a, col_b = st.columns([1, 1])
    run_search = col_a.button("Search", type="primary")
    col_b.write(f"Dataset size: {len(df)} titles")

    if run_search:
        if not query.strip():
            st.warning("Please type a query first.")
            return

        results = semantic_search(
            query=query,
            df=df,
            embeddings=embeddings,
            top_n=top_n,
            min_score=min_score,
            type_filter=type_filter,
        )

        if results.empty:
            st.error("No results found. Try lowering the minimum score or changing the query.")
            return

        st.success(f"Found {len(results)} results for: {query}")
        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="streamlit_query_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
