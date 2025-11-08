from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import motor.motor_asyncio
from dotenv import load_dotenv
import os
from bson import ObjectId

load_dotenv()

app = FastAPI(title="Post Search API (Enhanced: Title + Description + Interest)")

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
    posts = await posts_collection.find({}).to_list(length=1000)
    return posts

@app.post("/search_posts_by_title/", response_model=List[PostSearchResponse])
async def search_posts_by_title(request: SearchPostsRequest):
    posts = await fetch_all_posts()
    if not posts:
        return []

    corpus = [
        f"{p.get('title', '')} {p.get('description', '')} {p.get('interest', '')}"
        for p in posts
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vector = vectorizer.transform([request.query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[::-1][:request.top_n]

    if all(score == 0 for score in similarity_scores):
        results = [
            PostSearchResponse(
                post_id=str(p["_id"]),
                title=p["title"],
                description=p["description"],
                category=p["category"],
                mediaType=p.get("mediaType"),
                mediaUrl=p.get("mediaUrl"),
                similarity_score=0.0
            )
            for p in posts[:request.top_n]
        ]
        return results

    results = [
        PostSearchResponse(
            post_id=str(posts[i]["_id"]),
            title=posts[i]["title"],
            description=posts[i]["description"],
            category=posts[i]["category"],
            mediaType=posts[i].get("mediaType"),
            mediaUrl=posts[i].get("mediaUrl"),
            similarity_score=round(float(similarity_scores[i]), 3)
        )
        for i in top_indices
    ]

    return results

@app.get("/")
def home():
    return {"message": "Enhanced Post Search API is running 🚀"}

