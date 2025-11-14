import networkx as nx
import pandas as pd
try:
    from .normalization import enrich_person, rule_skill_sim, COMPANY_AFFINITY, seniority_gap_score
except ImportError:
    from normalization import enrich_person, rule_skill_sim, COMPANY_AFFINITY, seniority_gap_score

W_SKILL = 0.6
W_SCHOOL = 0.05
W_COMPANY = 0.05
W_RELATION = 0.2
W_COMPANY_AFF = 0.07
W_SENIORITY_GAP = 0.03


def same_school(a,b):  return int(a["school"]  == b["school"])
def same_company(a,b): return int(a["company"] == b["company"])
def company_affiliation(a,b):
    ca = (a.get("company") or "").strip().lower()
    cb = (b.get("company") or "").strip().lower()
    if not ca or not cb:
        return 0.0
    key = tuple(sorted([ca, cb]))
    return COMPANY_AFFINITY.get(key, 0.0)

def relationship_strength(a, b, rel_index: dict) -> float:
    key = tuple(sorted([a["_id"], b["_id"]]))
    return rel_index.get(key, 0.0)

def edge_weight(a, b, rel_index: dict) -> float:
    s_skill = rule_skill_sim(a, b)
    s_sch  = same_school(a, b)
    s_comp = same_company(a, b)
    s_rel  = relationship_strength(a, b, rel_index)
    s_aff  = company_affiliation(a, b)
    s_gap  = seniority_gap_score(a, b)

    score = (
        W_SKILL        * s_skill +
        W_SCHOOL       * s_sch +
        W_COMPANY      * s_comp +
        W_RELATION     * s_rel +
        W_COMPANY_AFF  * s_aff +
        W_SENIORITY_GAP* s_gap
    )
    # Clip to [0, 1.5] just for sanity (not strictly required)
    return max(0.0, min(score, 1.5))

def should_link(a, b, rel_index: dict) -> bool:
    shared_skills = len(set(a["skills_norm"]) & set(b["skills_norm"]))
    rel = relationship_strength(a, b, rel_index)
    aff = company_affiliation(a, b)

    # Strong relationship alone can justify an edge
    if rel >= 0.7:
        return True

    # Otherwise, require reasonable overlap
    return (
        shared_skills >= 3
        or (same_company(a, b) and shared_skills >= 1)
        or (same_school(a, b) and shared_skills >= 2)
        or (aff >= 0.6 and shared_skills >= 1)  # strong company partnership + at least 1 shared skill
    )


def build_relationship_strength_index(people: list[dict]) -> dict:
    """
    Returns dict[(id1, id2)] -> strength in [0,1],
    where (id1,id2) is a sorted tuple of person IDs.
    """
    rel = {}
    for p in people:
        pid = p.get("_id")
        if not pid:
            continue
        for c in p.get("connections", []):
            to_id = c.get("to")
            strength = float(c.get("strength", 0.0))
            if not to_id or strength <= 0:
                continue
            key = tuple(sorted([pid, to_id]))
            # if both sides define it, take max
            rel[key] = max(rel.get(key, 0.0), strength)
    return rel

def build_graph(people: list[dict]) -> nx.Graph:
    G = nx.Graph()
    by_id = {}
    # 1) enrich and add nodes
    for p in people:
        ep = enrich_person(p)
        pid = ep["_id"]
        by_id[pid] = ep
        G.add_node(pid, **ep)

    # 2) precompute relationship index
    rel_index = build_relationship_strength_index(people)

    # 3) add edges from similarity + relationships
    ids = list(by_id.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = by_id[ids[i]], by_id[ids[j]]
            if should_link(a, b, rel_index):
                G.add_edge(
                    a["_id"], b["_id"],
                    weight=edge_weight(a, b, rel_index),
                    signals={
                        "skill_similarity": round(rule_skill_sim(a, b), 3),
                        "same_school": bool(same_school(a, b)),
                        "same_company": bool(same_company(a, b)),
                        "relationship_strength": round(relationship_strength(a, b, rel_index), 3),
                        "company_affinity": round(company_affiliation(a, b), 3),
                        "seniority_gap_score": round(seniority_gap_score(a, b), 3),
                        "categories_a": a["categories"],
                        "categories_b": b["categories"],
                    }
                )
    return G


'''
Find up to `max_paths` simple paths (no repeated nodes) from source to target
with length <= max_hops. Rank them by total edge weight (desc), then by hops (asc)."
    G: nx.Graph,
    source_id: str,
    target_id: str,
    max_hops: int = 6,
    max_paths = k: int = 50,
'''
def shortest_paths_ranked(G: nx.Graph, source_id: str, target_id: str, max_hops=6, k=50) -> list[dict]:
     # basic checks
    if not G.has_node(source_id) or not G.has_node(target_id):
        return []
   
    gen = nx.all_simple_paths(G, source=source_id, target=target_id, cutoff=max_hops)

    paths: list[list[str]] = []
    for p in gen:
        paths.append(p)
        if len(paths) >= k:
            break  

    if not paths:
        return []

    # shortest hop length
    '''
    hops = nx.shortest_path_length(G, source=source_id, target=target_id)
    if hops > max_hops:
        return []
    paths = list(nx.all_shortest_paths(G, source=source_id, target=target_id))
    '''

    def path_score(path):
        return sum(G[u][v]["weight"] for u,v in zip(path[:-1], path[1:]))

    by_id = {n: G.nodes[n] for n in G.nodes()}
    results = []
    for p in paths:
        results.append({
            "nodes": p,
            "nodes_detail": [by_id[n] for n in p],
            "hops": len(p)-1,
            "score": round(path_score(p),3),
            "hops_detail": [
                {"from": u, "to": v,
                 "edge_weight": round(G[u][v]["weight"],3),
                 "signals": G[u][v]["signals"]}
                for u,v in zip(p[:-1], p[1:])
            ]
        })
    results.sort(key=lambda r: (-r["score"], r["hops"]))
    return results[:k]


'''
# --- Sample mock data ---
data = [
    {"id":"you",   "name":"You",          "company":"",       "school":"York University", "skills":["python","ml","networking"], "role":"Student"},
    {"id":"alice", "name":"Alice Chen",   "company":"Amazon", "school":"York University", "skills":["python","aws","ml"],         "role":"SDE"},
    {"id":"bob",   "name":"Bob Smith",    "company":"RBC",    "school":"York University", "skills":["finance","python","sql"],    "role":"Analyst"},
    {"id":"gina",  "name":"Gina Patel",   "company":"RBC",    "school":"Toronto",         "skills":["sql","data","python"],       "role":"Data Eng"},
    {"id":"emma",  "name":"Emma Wu",      "company":"Meta",   "school":"York University", "skills":["ml","python","nlp"],         "role":"MLE"},
    {"id":"dave",  "name":"Dave Kumar",   "company":"Amazon", "school":"Waterloo",        "skills":["java","aws","microservices"],"role":"SDE"},
]

df = pd.DataFrame(data)
# -------------------------
# Build the network graph
G = nx.Graph()

for person in data:
    # Same thing as:
    # G.add_node(person['id'], id="alice", name="Alice Chen", company="Amazon")
    G.add_node(person['id'], **person)
    # is called "unpacking" in Python. It takes a dictionary and turns it into separate key-value pairs.

# print(G.nodes(data=True))

# ------------------------- Add edges based on shared attributes -------------------------
by_id = {person['id']: person for person in data}
ids = list(by_id.keys())

# Function: edge signals based on shared attributes
def edge_signals(person1, person2):
    person1_skills, person2_skills = set(person1['skills']), set(person2['skills'])
    shared_skills = sorted(person1_skills & person2_skills) # intersection
    return {
        "same_school": person1["school"] == person2["school"],
        "same_company": person1["company"] == person2["company"],
        "shared_skills": shared_skills,
        "shared_skills_count": len(shared_skills),
    } 

# Function: edge weight based on number of shared attributes
def edge_weight(person1, person2):
    # simple weight: school(1.0) + company(1.0) + 0.5 * normalized shared-skills
    s = edge_signals(person1, person2)
    return (1.0 if s["same_school"] else 0.0) + \
           (1.0 if s["same_company"] else 0.0) + \
           0.5 * min(s["shared_skills_count"] / 10, 1.0)

for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        id1,id2 = ids[i], ids[j]
        person1, person2 = by_id[id1], by_id[id2]
        
        shared = set(person1['skills']) & set(person2['skills']) # intersection
        
        if person1['company'] == person2['company'] \
            or person1['school'] == person2['school'] \
                or len(shared) > 0:
            G.add_edge(person1["id"], person2["id"],
                       weight=edge_weight(person1, person2),
                       signals=edge_signals(person1, person2))
            
    
# Example: print edges with attributes
# print("nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
# print(list(G.edges(data=True))) 
'''
'''
nodes: 6 edges: 11
[('you', 'alice', {'weight': 1.1, 'signals': {'same_school': True, 'same_company': False, 'shared_skills': ['ml', 'python'], 'shared_skills_count': 2}}), 
('you', 'bob', {'weight': 1.05, 'signals': {'same_school': True, 'same_company': False, 'shared_skills': ['python'], 'shared_skills_count': 1}}), 
('you', 'gina', {'weight': 0.05, 'signals': {'same_school': False, 'same_company': False, 'shared_skills': ['python'], 'shared_skills_count': 1}}), 
('you', 'emma', {'weight': 1.1, 'signals': {'same_school': True, 'same_company': False, 'shared_skills': ['ml', 'python'], 'shared_skills_count': 2}}), 
('alice', 'bob', {'weight': 1.05, 'signals': {'same_school': True, 'same_company': False, 'shared_skills': ['python'], 'shared_skills_count': 1}}), 
('alice', 'gina', {'weight': 0.05, 'signals': {'same_school': False, 'same_company': False, 'shared_skills': ['python'], 'shared_skills_count': 1}}), 
('alice', 'emma', {'weight': 1.1, 'signals': {'same_school': True, 'same_company': False, 'shared_skills': ['ml', 'python'], 'shared_skills_count': 2}}), 
('alice', 'dave', {'weight': 1.05, 'signals': {'same_school': False, 'same_company': True, 'shared_skills': ['aws'], 'shared_skills_count': 1}}), 
('bob', 'gina', {'weight': 1.1, 'signals': {'same_school': False, 'same_company': True, 'shared_skills': ['python', 'sql'], 'shared_skills_count': 2}}), 
('bob', 'emma', {'weight': 1.05, 'signals': {'same_school': True, 'same_company': False, 'shared_skills': ['python'], 'shared_skills_count': 1}}), 
('gina', 'emma', {'weight': 0.05, 'signals': {'same_school': False, 'same_company': False, 'shared_skills': ['python'], 'shared_skills_count': 1}})]
'''
'''
# ------------------------- BFS to find shortest paths (≤ 6 hops) between two people -------------------------
source_id = "you"
target_id = "dave"

# 1) check connectivity
if nx.has_path(G, source_id, target_id):
    # 2) hop length of the shortest path
    hops = nx.shortest_path_length(G, source=source_id, target=target_id)
    # 3) enforce your ≤6 rule
    if hops <= 6:
        # 4) collect all shortest paths (list of lists of node ids)
        paths = list(nx.all_shortest_paths(G, source=source_id, target=target_id))
    else:
        paths = []   # too long for our rule
else:
    paths = []       # no path exists

# print(nx)
# print("hops:", hops if 'hops' in locals() else None)
# print("paths:", paths)

def path_score(G, path):
    # sum edge weights along the path
    return sum(G[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))

def explain_hops(G, path):
    # gather signals for each hop (for UI + message gen)
    details = []
    for u, v in zip(path[:-1], path[1:]):
        e = G[u][v]
        details.append({
            "from": u,
            "to": v,
            "edge_weight": round(e["weight"], 3),
            "signals": e["signals"]   # same_school, same_company, shared_skills(_count)
        })
    return details

# rank and keep top-k (e.g., 3)
k = 3
scored = []
for p in paths:
    scored.append({
        "nodes": p,
        "hops": len(p) - 1,
        "score": round(path_score(G, p), 3),
        "hops_detail": explain_hops(G, p),
    })

scored.sort(key=lambda r: (-r["score"], r["hops"]))
topk = scored[:k]

# print("Top score paths:", scored)
# print(" Top-k paths:", topk)

# -------- Get top path -------
def name(nid): 
    return G.nodes[nid]["name"]

if not topk:
    print("No path ≤ 6 hops. Try another target or add more people.")
else:
    for r in topk:
        path_names = " -> ".join(name(n) for n in r["nodes"])
        print(f"PATH: {path_names} | hops={r['hops']} | score={r['score']}")
        for hop in r["hops_detail"]:
            print("  ",
                  f"{name(hop['from'])} → {name(hop['to'])}",
                  "| w =", hop["edge_weight"],
                  "| signals =", hop["signals"])
'''
