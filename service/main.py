import os
import json
from typing import List, Literal, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from openai import OpenAI
from dotenv import load_dotenv

# ---------- Storage ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PEOPLE_JSON = DATA_DIR / "people.json"
if not PEOPLE_JSON.exists():
    PEOPLE_JSON.write_text("[]", encoding="utf-8")

# ---------- Schemas ----------
SeniorityEnum = Literal[
    "Student/Intern", "Entry", "Mid", "Senior", "Lead/Staff", "Manager+", "Founder", "Other"
]

class ExtractIn(BaseModel):
    text: str = Field(..., description="Raw pasted LinkedIn profile text")

class ProfileOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: Optional[str] = Field(default=None, alias="_id", serialization_alias="_id")
    name: str
    company: Optional[str] = ""
    role: Optional[str] = ""
    schools: List[str]
    skills: List[str]
    keywords: List[str]
    seniority: SeniorityEnum

class IngestIn(BaseModel):
    people: List[ProfileOut]

load_dotenv() # Load environment variables from .env file

# ---------- App ----------
app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# JSON Schema for strict structured output (enforced by OpenAI)
PROFILE_SCHEMA = {
    "name": "profile_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "company": {"type": "string"},
            "role": {"type": "string"},
            "schools": { "type": "array", "items": { "type": "string" } },
            "skills":  { "type": "array", "items": { "type": "string" } },
            "keywords":{ "type": "array", "items": { "type": "string" } },
            "seniority": {
                "type": "string",
                "enum": [
                    "Student/Intern","Entry","Mid","Senior",
                    "Lead/Staff","Manager+","Founder","Other"
                ]
            }
        },
        "required": ["name","company","role","schools","skills","keywords","seniority"]
    },
    "strict": True
}


SYSTEM_PROMPT = """You extract structured profile data from raw, pasted LinkedIn text.

- Return ONLY JSON following the provided JSON schema.
- Normalize lists: dedupe, trim whitespace.
- Preserve diacritics in names (e.g., Đ, Ñ, Ł).
- Choose a single current company/role if multiple are present (prefer full-time > contract > volunteer).
- "schools" contains formal institutions (degree-granting); exclude short certs unless clearly degree programs.
- "skills": 10–25 concise items if available; merge obvious synonyms (PostgreSQL/Postgres -> PostgreSQL, Kubernetes/k8s -> Kubernetes).
- "keywords": 5–15 topical tags (e.g., retrieval, observability, graph search, embeddings).
- Map seniority by title cues:
  Intern/Co-op/RA/TA -> "Student/Intern"
  Junior/New Grad -> "Entry"
  Senior -> "Senior"
  Staff/Principal/Lead -> "Lead/Staff"
  Manager/Director/VP/Head -> "Manager+"
  Founder/Co-founder/CEO -> "Founder"
  Otherwise -> "Other"
"""

def _norm_list(xs):
    if not isinstance(xs, list):
        return []
    seen, out = set(), []
    for x in xs:
        if not isinstance(x, str):
            continue
        s = x.strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
    return out

def _postprocess(d: dict) -> dict:
    d["schools"] = _norm_list(d.get("schools", []))
    d["skills"] = _norm_list(d.get("skills", []))
    d["keywords"] = _norm_list(d.get("keywords", []))

    # Ensure required string fields exist
    for k in ("name", "company", "role"):
        v = d.get(k, "")
        d[k] = v if isinstance(v, str) else ""

    # Clamp seniority to enum if model returned something odd
    valid = {"Student/Intern","Entry","Mid","Senior","Lead/Staff","Manager+","Founder","Other"}
    if d.get("seniority") not in valid:
        d["seniority"] = "Other"

    return d

def _slug(p: dict) -> str:
    if not isinstance(p, dict):
        return ""
    pid = (p.get("_id") or p.get("id") or "").strip().lower()
    if pid:
        return pid
    return "||".join([
        p.get("name", "").strip().lower(),
        p.get("company", "").strip().lower(),
        p.get("role", "").strip().lower(),
    ])

def _load_people() -> list:
    try:
        return json.loads(PEOPLE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save_people(rows: list):
    PEOPLE_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

# --------- Endpoints ---------
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/extract-profiles", response_model=ProfileOut)
@app.post("/extract-profile", response_model=ProfileOut)
def extract_profile(payload: ExtractIn):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Empty input text.")

    try:
        # Chat Completions with Structured Outputs (JSON Schema)
        # (Chat Completions remains supported; we use response_format to enforce a schema.)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload.text.strip()},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": PROFILE_SCHEMA
            },
        )
        content = resp.choices[0].message.content

        # Parse & validate
        data = json.loads(content)
        data = _postprocess(data)
        return ProfileOut(**data)

    except ValidationError as ve:
        # Pydantic validation errors
        raise HTTPException(status_code=422, detail=json.loads(ve.json()))
    except Exception as e:
        # If model didn't return valid JSON, surface a 502 with message
        raise HTTPException(status_code=502, detail=f"Extraction failed: {str(e)}")

@app.post("/ingest-people")
def ingest_people(payload: IngestIn):
    """Upsert list of profiles into data/people.json"""
    store = _load_people()
    index = { _slug(p): i for i, p in enumerate(store) }
    inserted = updated = 0
    for p in payload.people:
        pd = json.loads(p.model_dump_json(by_alias=True))
        key = _slug(pd)
        if key in index:
            store[index[key]] = pd
            updated += 1
        else:
            store.append(pd)
            index[key] = len(store) - 1
            inserted += 1
    _save_people(store)
    return {"ok": True, "inserted": inserted, "updated": updated, "total": len(store), "path": str(PEOPLE_JSON)}

@app.get("/people", response_model=List[ProfileOut])
def list_people():
    return _load_people()
