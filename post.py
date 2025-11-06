from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI(title="Post Title Search API")

# -----------------------------
# 🔹 Post Model
# -----------------------------
class Post(BaseModel):
    post_id: int
    title: str
    content: str
    tags: str
    author: str

# -----------------------------
# 🔹 Search Posts by Title Only
# -----------------------------
@app.post("/search_posts_by_title/")
def search_posts_by_title(posts: List[Post], query: str, top_n: int = 5):
    # Convert posts into DataFrame
    posts_df = pd.DataFrame([p.dict() for p in posts])

    # TF-IDF on post titles only
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(posts_df["title"])

    # Transform query
    query_vector = vectorizer.transform([query])

    # Compute similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Sort and get top N results
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    # Prepare response
    results = [
        {
            "post_id": int(posts_df.iloc[i]["post_id"]),
            "title": posts_df.iloc[i]["title"],
            "tags": posts_df.iloc[i]["tags"],
            "author": posts_df.iloc[i]["author"],
            "similarity_score": round(float(similarity_scores[i]), 3)
        }
        for i in top_indices if similarity_scores[i] > 0
    ]

    # Handle no matches
    if not results:
        return {"message": f"No post found matching '{query}'"}

    return {"query": query, "matched_posts": results}

# Example root
@app.get("/")
def home():
    return {"message": "Post Title Search API is running 🚀"}
