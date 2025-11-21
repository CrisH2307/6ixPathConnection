import numpy as np
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

api_key = os.environ.get("api_key")
if not api_key:
    raise RuntimeError("api_key is not set; check your .env or environment")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,)

EMBEDDING_MODEL = "text-embedding-3-large"

def _get_embedding(text: str) -> list[float]:
    """Stub function to get embedding vector for one text string."""
    # In actual implementation, this would call the embedding service
    # Here we return a random vector for demonstration purposes
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding

def _person_to_embedding_text(person: dict) -> str:
    """Convert a person's profile to a text string for embedding."""
    name = person.get("name", "")
    company = person.get("company", "")
    role = person.get("role", "")
    skills = ", ".join(person.get("skills", []))
    keywords = ", ".join(person.get("keywords", []))

    return f"{name} | {company} | {role} | skills: {skills} | keywords: {keywords}"

def _save_people(people: list[dict], path: str = None):
    """Save the people list back to JSON file."""
    import json
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "people.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(people, f, ensure_ascii=False, indent=2)

def _ensure_embeddings(people: list[dict]) -> list[dict]:
    """
    For each person, if 'embedding' is missing, compute it and save back to people.json.
    """
    changed = False

    for p in people:
        if not isinstance(p.get("embedding"), list):
            text = _person_to_embedding_text(p)
            p["embedding"] = _get_embedding(text)
            changed = True

    if changed:
        _save_people(people)

    return people

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
