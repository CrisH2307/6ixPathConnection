import os
import json
from typing import List, Literal, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from openai import OpenAI
from dotenv import load_dotenv
from models.graph import build_graph, shortest_paths_ranked
from models.embedding import _ensure_embeddings, _cosine_sim
from typing import List, Literal, Optional, Dict, Any
import numpy as np

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

class Connection(BaseModel):
    to: str = Field(..., description="Target person ID (or slug) this profile connects to")
    strength: float = Field(1.0, description="Connection strength between 0 and 1")
    tags: List[str] = Field(default_factory=list, description="Labels like 'seneca_classmate', 'same_team_kpmg'")

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

    connections: List[Connection] = Field(
        default_factory=list,
        description="Explicit graph edges to other people, if provided"
    )
    
class IngestIn(BaseModel):
    people: List[ProfileOut]

class RouteRequest(BaseModel):  
    source_name: str = Field(..., description="Source person name, e.g. 'Cris Huynh'")
    target_name: str = Field(..., description="Target person name, e.g. 'Khoi Vu'")

# ---------- Embedding ----------    
class SimilarPeopleRequest(BaseModel):
    source_name: str = Field(..., description="Name of the person to compare from")
    top_k: int = Field(10, ge=1, le=50, description="How many similar people to return")


load_dotenv() # Load environment variables from .env file

# ---------- App ----------
app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            },
            "connections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "to": {"type": "string"},
                        "strength": {"type": "number"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["to", "strength", "tags"]
                }
            }
        },
        "required": ["name","company","role","schools","skills","keywords","seniority","connections"]
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
- Connections: sort by strength, then by tag count, then by name.
- Map seniority by title cues:
  Intern/Co-op/RA/TA -> "Student/Intern"
  Junior/New Grad -> "Entry"
  Senior -> "Senior"
  Staff/Principal/Lead -> "Lead/Staff"
  Manager/Director/VP/Head -> "Manager+"
  Founder/Co-founder/CEO -> "Founder"
  Otherwise -> "Other"
"""

def _normalize_set(v) -> set:
    if not v:
        return set()
    if not isinstance(v, list):
        v = [v]
    return {str(x).strip().lower() for x in v if str(x).strip()}


def _describe_shared_profile(source: Dict[str, Any], recipient: Dict[str, Any]) -> str:
    """
    Build a human-readable description of the overlap between two profiles.
    This text is passed into the LLM prompt to help it emphasize similarity.
    """

    src_schools = _normalize_set(source.get("schools", []))
    rec_schools = _normalize_set(recipient.get("schools", []))
    src_skills = _normalize_set(source.get("skills", []))
    rec_skills = _normalize_set(recipient.get("skills", []))

    shared_schools = sorted(src_schools & rec_schools)
    shared_skills = sorted(src_skills & rec_skills)

    lines = []
    if shared_schools:
        lines.append("Shared schools: " + ", ".join(shared_schools))
    if shared_skills:
        lines.append("Shared skills: " + ", ".join(shared_skills))

    if not lines:
        return "None obvious from their profiles."
    return "\n".join("- " + line for line in lines)


def _describe_edge_signals(signals: Dict[str, Any]) -> str:
    """
    Convert graph edge 'signals' into a short textual summary.
    """

    if not isinstance(signals, dict):
        return "No edge-level signals provided."
    lines = []
    if isinstance(signals.get("skill_similarity"), (int, float)):
        lines.append(f"Skill similarity (edge): {signals['skill_similarity']:.2f}")
    if signals.get("same_school"):
        lines.append("Same school (edge)")
    if signals.get("same_company"):
        lines.append("Same company (edge)")
    cats_a = signals.get("categories_a") or []
    cats_b = signals.get("categories_b") or []
    if isinstance(cats_a, list) and isinstance(cats_b, list):
        shared_cats = sorted({c for c in cats_a if c in cats_b})
        if shared_cats:
            lines.append("Shared categories: " + ", ".join(shared_cats))
    if not lines:
        return "No edge-level signals provided."
    return "\n".join("- " + line for line in lines)


def _build_outreach_prompt_step(
    source_person: Dict[str, Any],
    recipient_person: Dict[str, Any],
    warm_intro_person: Optional[Dict[str, Any]],
    target_person: Dict[str, Any],
    next_hop_person: Optional[Dict[str, Any]],
    signals: Dict[str, Any],
    step_index: int,
    total_steps: int,
    embedding_similarity: Optional[float],
) -> str:
    """
    Build the prompt for ONE outreach message.

    Requirements:
    - Always from source_person ("I").
    - Recipient is this hop (or final destination).
    - Emphasize similarity between source and recipient (skills, schools, embeddings).
    - If warm_intro_person exists, mention them once as context.
    - Main ask: short chat / answer 1–2 questions.
    """

    src_name = source_person.get("name") or "the sender"
    rec_name = recipient_person.get("name") or "the recipient"
    warm_name = warm_intro_person.get("name") if warm_intro_person else None
    target_name = target_person.get("name") or "the final target"
    next_name = next_hop_person.get("name") if next_hop_person else None

    src_role = source_person.get("role") or ""
    src_company = source_person.get("company") or ""
    rec_role = recipient_person.get("role") or ""
    rec_company = recipient_person.get("company") or ""
    target_role = target_person.get("role") or ""
    target_company = target_person.get("company") or ""

    shared_profile_text = _describe_shared_profile(source_person, recipient_person)
    edge_signals_text = _describe_edge_signals(signals)

    # Turn embedding similarity into guidance (for the model, not for literal output)
    if embedding_similarity is not None:
        emb_guidance = (
            f"Embedding-based similarity between sender and recipient is about "
            f"{embedding_similarity:.3f} (cosine).\n"
            "Rough guideline for you:\n"
            "- ~0.70 or higher: strong overlap – emphasize that your work or background is very similar.\n"
            "- 0.40–0.70: moderate overlap – mention some shared themes or skills.\n"
            "- below 0.40: lighter overlap – only a soft reference to overlapping interests.\n"
            "When writing the message, DO NOT mention any numbers; just use natural phrases "
            "like 'we both work on similar things', 'our backgrounds overlap', etc."
        )
    else:
        emb_guidance = (
            "No numeric embedding similarity is available. You can still rely on shared skills, "
            "schools, or domains if present."
        )

    # Step context: what has already happened
    if warm_name:
        step_context = (
            f"{src_name} has already had a brief conversation with {warm_name}.\n"
            f"{warm_name} encouraged {src_name} to reach out to {rec_name} for a quick chat "
            "and to learn from their experience."
        )
    else:
        step_context = (
            f"This is the first hop. {src_name} is reaching out directly to {rec_name} "
            "to ask a few questions and learn from their experience."
        )

    # Ask: always about a short conversation or answering 1–2 questions
    if next_name and step_index < total_steps:
        ask_hint = (
            f"For this step, the PRIMARY ask is a brief chat (or answering 1–2 short questions) "
            f"with {rec_name} about their experience and perspective.\n"
            f"You may VERY SOFTLY suggest that later, if the conversation goes well, "
            f"{rec_name} could advise who else {src_name} might talk to (possibly including {next_name}), "
            "but do NOT explicitly ask them to introduce or connect you to anyone in this message."
        )
    else:
        ask_hint = (
            f"This is the final hop. The PRIMARY ask is a short conversation or the chance to ask "
            f"1–2 concise questions to {rec_name} about their work, team, or path.\n"
            "Do not push for introductions; focus on learning and advice."
        )

    return f"""
You are a friendly, concise career networking coach.

Write a LinkedIn-style outreach message that {src_name} can send directly to {rec_name}.

Sender (write from this "I" perspective):
- Name: {src_name}
- Role: {src_role}
- Company: {src_company}

Recipient:
- Name: {rec_name}
- Role: {rec_role}
- Company: {rec_company}

Broader context (for you only, not to explain in detail in the message):
- {src_name} is ultimately interested in learning more around {target_name}'s space
  (role: {target_role}, company: {target_company}).
- The path has {total_steps} hops; this message is for step {step_index}.

Step context:
- {step_context}

Similarity context between sender and recipient:
- Shared profile features (schools / skills):
{shared_profile_text}

- Graph edge signals (previous matching between this pair in the network):
{edge_signals_text}

- Embedding similarity guidance:
{emb_guidance}

Ask / CTA:
- {ask_hint}

STYLE RULES (IMPORTANT):
- 60–110 words.
- Emphasize overlap or similarity between {src_name} and {rec_name} (background, skills, interests).
- If {warm_name} is provided, mention them NATURALLY in exactly one sentence, as shared context
  (e.g., "I recently spoke with {warm_name}, who suggested I reach out to you.").
- MAIN focus: a quick chat or answering a couple of questions, not building a connection chain.
- Do NOT explicitly ask them to introduce you to {next_name} or anyone else.
- Do NOT explicitly ask them to 'connect on LinkedIn'; it's okay if they choose to connect later.
- Use "I" for {src_name} and "you" for {rec_name}.
- Sound warm, specific, and low-pressure, not salesy or spammy.

Return ONLY the outreach message text, nothing else.
""".strip()

def _generate_outreach_message_step(
    client: OpenAI,
    source_person: Dict[str, Any],
    recipient_person: Dict[str, Any],
    warm_intro_person: Optional[Dict[str, Any]],
    target_person: Dict[str, Any],
    next_hop_person: Optional[Dict[str, Any]],
    signals: Dict[str, Any],
    step_index: int,
    total_steps: int,
    embedding_similarity: Optional[float],  
) -> str:
    """
    Call the OpenAI Chat Completions API to generate a single outreach message.

    Steps:
    1. Build a structured prompt via `_build_outreach_prompt_step`.
    2. Send it to `gpt-4o-mini` with a coaching-style system prompt.
    3. Return the generated message text with surrounding whitespace stripped.

    Args:
        client:               Initialized `OpenAI` client.
        source_person:        Dict with sender profile fields (name, role, company, etc.).
        recipient_person:     Dict with current hop profile fields.
        warm_intro_person:    Previous hop profile dict, if any.
        target_person:        Final target profile dict.
        next_hop_person:      Next hop profile dict, if any.
        signals:              Graph edge signals for this hop.
        step_index:           1-based index of this step along the path.
        total_steps:          Total number of hops in the path.
        embedding_similarity: Optional cosine similarity between source and this hop.

    Returns:
        The outreach message text generated by the model.
    """
    prompt = _build_outreach_prompt_step(
        source_person=source_person,
        recipient_person=recipient_person,
        warm_intro_person=warm_intro_person,
        target_person=target_person,
        next_hop_person=next_hop_person,
        signals=signals,
        step_index=step_index,
        total_steps=total_steps,
        embedding_similarity=embedding_similarity,
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You are a helpful career networking coach."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def _norm_list(xs):
    """Normalize a list of strings: ignore non-string values, dedupe (case-insensitive), trim whitespace."""
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
    """Post-process and normalize extracted profile dict."""
    d["schools"] = _norm_list(d.get("schools", []))
    d["skills"] = _norm_list(d.get("skills", []))
    d["keywords"] = _norm_list(d.get("keywords", []))

    # Normalize connection tags if present
    conns = d.get("connections") or []
    norm_conns = []
    for c in conns:
        if not isinstance(c, dict):
            continue
        tags = _norm_list(c.get("tags", []))
        strength = c.get("strength", 1.0)
        try:
            strength = float(strength)
        except Exception:
            strength = 1.0
        norm_conns.append({
            "to": str(c.get("to", "")).strip(),
            "strength": strength,
            "tags": tags,
        })
    d["connections"] = norm_conns

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
    """Create a unique slug for a person dict for indexing during ingestion."""
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
    """
    Load all stored profiles from `data/people.json`.
    """
    try:
        return json.loads(PEOPLE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []

def _save_people(rows: list):
    """
    Persist the given list of profile dicts to `data/people.json`.
    """
    PEOPLE_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

def _build_graph_and_index():  
    """
    Load people.json, build the similarity graph and create a name -> id index.
    """
    people = _load_people()
    if not people:
        raise HTTPException(
            status_code=400,
            detail="No people in database. Ingest profiles first."
        )

    # Build the NetworkX graph G
    G = build_graph(people)

    # Build a `name_to_id` dict mapping lowercase names to the internal _id used in the graph.
    name_to_id = {}
    for p in people:
        name = (p.get("name") or "").strip().lower()
        pid = (p.get("_id") or "").strip()
        if name and pid:
            name_to_id[name] = pid

    return G, name_to_id

# --------- Endpoints ---------
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/extract-profiles", response_model=ProfileOut)
@app.post("/extract-profile", response_model=ProfileOut)
def extract_profile(payload: ExtractIn):
    """
    Extract a structured profile from raw LinkedIn-like text using the LLM.
    """
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
        raise HTTPException(status_code=502, detail=f"{str(e)}")

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
    """
    Return all stored profiles from `data/people.json`.
    """
    return _load_people()

@app.post("/generate-routes")
def generate_routes(payload: RouteRequest):
    """
    Generate top shortest connection routes between two people, given their names.
    """
    # Build graph and name index
    G, name_to_id = _build_graph_and_index()

    src_name = payload.source_name.strip().lower()
    tgt_name = payload.target_name.strip().lower()

    if src_name not in name_to_id:
        raise HTTPException(
            status_code=404,
            detail=f"Source '{payload.source_name}' not found",
        )
    if tgt_name not in name_to_id:
        raise HTTPException(
            status_code=404,
            detail=f"Target '{payload.target_name}' not found",
        )

    # Ensure embeddings exist
    people = _load_people()
    if not people:
        raise HTTPException(status_code=400, detail="No people in database.")

    people = _ensure_embeddings(people)

    name_to_person: Dict[str, Dict[str, Any]] = {}
    for p in people:
        nm = (p.get("name") or "").strip().lower()
        if nm:
            name_to_person[nm] = p

    src_id = name_to_id[src_name]
    tgt_id = name_to_id[tgt_name]

    # Find top-k shortest paths (default max 6 hops, default top 50 paths)
    topk = shortest_paths_ranked(G, src_id, tgt_id, max_hops=6, k=50)

    # print("Source resolved to node:", src_id)
    # print("Target resolved to node:", tgt_id)
    # print("Neighbors of source:", list(G.neighbors(src_id)))

    # for u, v, data in G.edges(data=True):
    #     print(u, "<->", v, "weight=", data["weight"], "signals=", data["signals"])

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Neighbors of source:", [
        (nid, G.nodes[nid]["name"]) for nid in G.neighbors(src_id)
    ])

    # Display found paths
    def name(nid): 
        return G.nodes[nid]["name"]

    display_paths = topk
    print(f"Found {len(display_paths)} paths from '{name(src_id)}' to '{name(tgt_id)}':")
    for r in display_paths:
        print(f"PATH ({r['hops']} hops, score={r['score']}):",
              " -> ".join(name(n) for n in r["nodes"]))
        for hop in r["hops_detail"]:
            print("  ", name(hop["from"]), "→", name(hop["to"]),
                  "| w =", hop["edge_weight"],
                  "| signals =", hop["signals"])

    if not topk:
        return {
        "source_name": name(src_id),
        "target_name": name(tgt_id),
        "paths": None,               
    }

    # Build outreach message sequences for each path
    def person_from_name(nm: str) -> Optional[Dict[str, Any]]:
        if not nm:
            return None
        return name_to_person.get(nm.strip().lower())

    src_person = person_from_name(payload.source_name)
    tgt_person = person_from_name(payload.target_name)

    # Limit to top 1 path for outreach message generation
    MAX_PATHS_WITH_MESSAGES = 1

    # For each path, generate outreach messages for each hop
    for path in topk[:MAX_PATHS_WITH_MESSAGES]:
        nodes_in_path = path.get("nodes", [])
        hops_detail = path.get("hops_detail", [])

        if len(nodes_in_path) < 2:
            path["outreach_sequence"] = []
            continue

        total_steps = len(nodes_in_path) - 1
        outreach_sequence = []

        for step_index in range(1, len(nodes_in_path)):
            recipient_id = nodes_in_path[step_index]
            recipient_name = name(recipient_id)
            recipient_person = person_from_name(recipient_name)

            # warm intro = previous hop in chain (None for first step)
            warm_intro_person = None
            warm_intro_id = None
            if step_index > 1:
                warm_intro_id = nodes_in_path[step_index - 1]
                warm_intro_person = person_from_name(name(warm_intro_id))

            # next hop (who we ultimately hope this step can move us toward)
            next_hop_person = None
            next_hop_id = None
            if step_index < len(nodes_in_path) - 1:
                next_hop_id = nodes_in_path[step_index + 1]
                next_hop_person = person_from_name(name(next_hop_id))

            # hop signals: match by 'to'
            hop_signals = {}
            for h in hops_detail:
                if h.get("to") == recipient_id:
                    hop_signals = h.get("signals", {}) or {}
                    break

            # ---- embedding similarity: source vs THIS recipient ----
            src_hop_sim = None
            if src_person is not None and recipient_person is not None:
                src_emb = np.array(src_person.get("embedding"), dtype=float)
                rec_emb = np.array(recipient_person.get("embedding"), dtype=float)
                if src_emb.size and rec_emb.size:
                    src_hop_sim = float(_cosine_sim(src_emb, rec_emb))

            # Generate outreach message for this step
            try:
                msg = _generate_outreach_message_step(
                    client=client,
                    source_person=src_person or {"name": payload.source_name},
                    recipient_person=recipient_person or {"name": recipient_name},
                    warm_intro_person=warm_intro_person,
                    target_person=tgt_person or {"name": payload.target_name},
                    next_hop_person=next_hop_person,
                    signals=hop_signals,
                    step_index=step_index,
                    total_steps=total_steps,
                    embedding_similarity=src_hop_sim,  
                )
            except Exception as e:
                print("Error generating outreach message for step", step_index, ":", e)
                msg = ""

            outreach_sequence.append(
                {
                    "step": step_index,
                    "sender_id": src_id,          # always source
                    "recipient_id": recipient_id,
                    "warm_intro_id": warm_intro_id,
                    "embedding_similarity": src_hop_sim,
                    "message": msg,
                }
            )

        # Attach outreach sequence to path
        path["outreach_sequence"] = outreach_sequence

    return {
        "source_name": name(src_id),
        "target_name": name(tgt_id),
        "paths": topk,               
    }

@app.post("/similar-people")
def similar_people(payload: SimilarPeopleRequest):
    """
    Given a source_name, return top-k most similar people based on embeddings.
    """
    print("loading people and embeddings...")
    people = _load_people()
    if not people:
        raise HTTPException(status_code=400, detail="No people in database.")

    # Ensure embeddings exist
    people = _ensure_embeddings(people)

    # Build name index
    name_to_person = {}
    for p in people:
        name = (p.get("name") or "").strip().lower()
        if name:
            name_to_person[name] = p

    src_key = payload.source_name.strip().lower()
    if src_key not in name_to_person:
        raise HTTPException(status_code=404, detail=f"Person '{payload.source_name}' not found")

    src_person = name_to_person[src_key]
    src_vec = np.array(src_person["embedding"], dtype=float)

    # Compute similarities to everyone else
    sims = []
    for p in people:
        if p is src_person:
            continue
        vec = np.array(p["embedding"], dtype=float)
        sim = _cosine_sim(src_vec, vec)
        sims.append({
            "score": sim,
            "name": p.get("name", ""),
            "company": p.get("company", ""),
            "role": p.get("role", ""),
        })

    # Sort by similarity
    sims.sort(key=lambda x: x["score"], reverse=True)

    return {
        "source_name": src_person.get("name", ""),
        "results": sims[:payload.top_k],
    }
