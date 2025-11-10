import networkx as nx
import pandas as pd


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