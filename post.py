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

app = FastAPI(title="Post Search API (MongoDB-Compatible)")

# ---------------- MongoDB Setup ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["auth-db"]
posts_collection = db.posts

# ---------------- Models ----------------
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
    top_n: int = 5  # default top 5 results

# ---------------- Helper Function ----------------
async def fetch_all_posts():
    posts = await posts_collection.find({}).to_list(length=1000)
    return posts

# ---------------- API Endpoint ----------------
@app.post("/search_posts_by_title/", response_model=List[PostSearchResponse])
async def search_posts_by_title(request: SearchPostsRequest):
    posts = await fetch_all_posts()
    if not posts:
        return []

    # TF-IDF on post titles (or combine title + description if needed)
    corpus = [p["title"] for p in posts]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Transform query
    query_vector = vectorizer.transform([request.query])

    # Compute similarity
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get top N results
    top_indices = similarity_scores.argsort()[::-1][:request.top_n]

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
        for i in top_indices if similarity_scores[i] > 0
    ]

    return results

# Root endpoint
@app.get("/")
def home():
    return {"message": "Post Search API is running 🚀"}
