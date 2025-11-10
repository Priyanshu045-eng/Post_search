from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import motor.motor_asyncio
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Vectorized Post Search API (Title + Description + Category)")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["auth-db"]
posts_collection = db.posts


class PostSearchResponse(BaseModel):
    post_id: str
    title: str
    description: str
    category: str
    mediaType: Optional[str]
    mediaUrl: Optional[str]
    similarity_score: float


class SearchPostsRequest(BaseModel):
    query: str
    top_n: int = 5


async def fetch_all_posts():
    """Fetch all posts from MongoDB"""
    posts = await posts_collection.find({}).to_list(length=2000)
    return posts


def safe_to_text(value):
    """Convert any field (list, dict, None) into a clean string"""
    if isinstance(value, list):
        return " ".join(map(str, value))
    elif isinstance(value, dict):
        return " ".join([f"{k} {v}" for k, v in value.items()])
    elif value is None:
        return ""
    return str(value)


@app.post("/search_posts_by_title/", response_model=List[PostSearchResponse])
async def search_posts_by_title(request: SearchPostsRequest):
    posts = await fetch_all_posts()
    if not posts:
        return []


    corpus = [
        f"{safe_to_text(p.get('title'))} {safe_to_text(p.get('description'))} {safe_to_text(p.get('category'))}"
        for p in posts
    ]


    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)


    query_vector = vectorizer.transform([request.query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[::-1][:request.top_n]


    if all(score == 0 for score in similarity_scores):
        return [
            PostSearchResponse(
                post_id=str(p["_id"]),
                title=safe_to_text(p.get("title")),
                description=safe_to_text(p.get("description")),
                category=safe_to_text(p.get("category")),
                mediaType=safe_to_text(p.get("mediaType")),
                mediaUrl=safe_to_text(p.get("mediaUrl")),
                similarity_score=0.0,
            )
            for p in posts[:request.top_n]
        ]

    results = [
        PostSearchResponse(
            post_id=str(posts[i]["_id"]),
            title=safe_to_text(posts[i].get("title")),
            description=safe_to_text(posts[i].get("description")),
            category=safe_to_text(posts[i].get("category")),
            mediaType=safe_to_text(posts[i].get("mediaType")),
            mediaUrl=safe_to_text(posts[i].get("mediaUrl")),
            similarity_score=round(float(similarity_scores[i]), 3),
        )
        for i in top_indices
    ]

    return results


@app.get("/")
def home():
    return {"message": "Vectorized Post Search API (Title + Description + Category) is running 🚀"}
