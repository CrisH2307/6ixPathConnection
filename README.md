# 6ixPathConnection

## Project Overview
6-Step Connection Builder is an AI-powered career-networking assistant that helps students and early-career job seekers find warm, less than 6-hop paths to hiring managers, alumni, and potential referrers, instead of relying on low-success cold DMs.

The system transforms raw user-provided data (résumé, alumni lists, LinkedIn text, job posts) into a relationship graph, computes the best warm paths, and generates personalized outreach messages for each step.

By combining graph algorithms, embeddings, and LLM-driven profile extraction, the tool gives users a structured, confidence-preserving way to approach people with the highest chance of responding.

This is inspirted by the theory of [Six Degrees Of Separation](https://en.wikipedia.org/wiki/Six_degrees_of_separation)

## Problem Statement
Students and early-career job seekers, especially introverted or anxious candidates that get very low response rates when they cold-message senior people (e.g., VPs or hiring manager) on LinkedIn. The issue isn’t just confidence; it’s network structure and social proximity. Large-scale experiments on LinkedIn show that “weak ties” and warm introductions are significantly more effective for job mobility than cold outreach to close or distant contacts, meaning your likelihood of being seen and helped rises when you’re introduced through intermediate connections rather than contacting a VP directly 

At the same time, the job market has widespread ghosting, many seekers report being ignored after outreach or interviews, which further penalizes cold messages without shared context or referrals. Mentorship and social support measurably improve career outcomes and job-search confidence, suggesting structured warm paths (alumni -> mutuals -> target) are a better on-ramp for students than “message the VP or manager” tactics.  

Therefore, early-career users lack a clear, confidence-preserving, research-backed way to reach decision-makers through meaningful intermediaries. Cold DMs underperform because they lack shared signals (school, company, skills) and social proof; warm, ≤6-step paths increase visibility, relevance, and reply to likelihood. 


## Solution
A lightweight AI web app that:
1. Builds a Relationship Graph
   + Nodes = people
   + Edges = shared school, company, skills, past projects, or known relationships
   + Weighted by connection strength and similarity
2. Finds the Best less than 6-Hop Path with greatest value
   + BFS/A* search on weighted graph
   + Returns the most relevant, realistic warm paths (top 3)
   + Shows shared signals at each hop
3. Generates Personalized Outreach Messages
   + LLM drafts messages referencing real shared background
   + Users edit and send manually (no auto-sending)
4. Tracks Progress
   + “To Contact -> Contacted -> Replied”
5.	Privacy-First
   + No scraping
   + All data is user-provided
   + User can delete/export anytime


## Project Research & Methods
This project is grounded in published research showing that:
  + Weak ties and warm introductions increase job mobility significantly (MIT + LinkedIn experiment, 20M people).
  + Cold outreach has low reply rates, especially for students and early-career job seekers.
  + Ghosting and missing follow-ups are major psychological barriers in job searching.
  + Mentorship and structured support greatly increase confidence and opportunities.

The system uses a combination of:
### 1. Natural Language Processing:
  + The system uses NLP to turn messy, unstructured text into clean, machine-readable data. Users usually paste job descriptions or profile text full of sentences, bullet points, and mixed formatting. NLP is used to extract the important parts: `name, company, role, schools, skills, and keywords` and convert them into structured fields we can use inside the graph.
  + This step removes noise, deduplicates skills, normalizes school/company names, and prepares each person as a consistent “node” in the network. Without NLP, the graph cannot be built because the system wouldn’t know what a person’s background actually contains.

### 2. Large Language Model (LLM) for Understanding & Message Generation:
  + Understanding when it reads a pasted profile or resume and converts it into structured JSON using a strict schema. The LLM helps interpret unclear titles, detect seniority, or merge synonymous skills.
  + Message Generation: For each hop in the connection path, the system uses the LLM to draft personalized outreach messages. Instead of generic templates, the LLM writes context-aware messages (e.g., referencing shared school, similar tech stack, or mutual friends) that feel natural and respectful. This improves reply rates and reduces social anxiety for early-career users.

### 3. Graph Data Structures and Algorithms - inspired by Open Source Library [NetworkX](https://networkx.org/):
  + We represent the user’s networking system using a weighted undirected graph built with NetworkX.
  + Each node is a person (student, alumni, engineer, manager).
  + Each edge represents a potential connection based on shared schools, companies, skills, or explicit relationship links (e.g., “close friend”, “same team”).
  + Each edge receives a numerical weight:
      ```
        score = (
              W_SKILL        * s_skill +
              W_SCHOOL       * s_sch +
              W_COMPANY      * s_comp +
              W_RELATION     * s_rel +
              W_COMPANY_AFF  * s_aff +
              W_SENIORITY_GAP* s_gap
              )
      ```
      + Which W as Weight and S as Score
  + Once the graph is built, we run a two-stage search process:
      + Stage 1: BFS (Breadth-First Search):
        + Guarantees we only explore shortest paths in hop count, ensuring the solution always returns a warm path ≤6 steps (the “Six Degrees” constraint).
	  + Stage 2: A* ranking on top of BFS results:
	      + Among all shortest paths, we compute a score = sum of edge weights (shared skills, school match, company match, relationship strength, and connections strength).
	      + This prioritizes paths that are not only short, but also strongest, warmest, and most realistic in a real networking context.
    
### 4. Fruchterman–Reingold Force-Directed Layout (Graph Visualization):
#### Resources
+ (https://en.wikipedia.org/wiki/Force-directed_graph_drawing)
+ (https://networkx.org/documentation/networkx-1.11/reference/generated/networkx.drawing.layout.fruchterman_reingold_layout.html)
+ (https://noesis.ikor.org/wiki/algorithms/visualization#h.p_d8oCA05tzRbA)

**Fruchterman-Reingold** is a force-directed graph drawing algorithm that visually represents networks by simulating a physical system where nodes are connected by springs. The
Fruchterman-Reingold algorithm is simply a clever method for neatly drawing diagrams that show how things are connected. Imagine the dots in the diagram are little magnets and springs. The springs pull connected dots together, while the magnets push all dots away from each other so they don't bunch up. The algorithm repeatedly adjusts the position of every dot based on these pulling and pushing forces, starting from a random mess and slowly calming the movement until the whole picture becomes a stable, clear map of the network. This makes it easy to look at the picture and instantly understand complex relationships, like who is friends with whom in a social group

![Graph Overview](https://www.researchgate.net/publication/301217160/figure/fig9/AS:359956277678080@1462831674860/Force-directed-layout-Fruchterman-Reingold-algorithm-of-an-example-ground-truth-network.png)

+ To visualize the relationship graph, layout treats the graph like a physical system:
  + Connected people pull together
  + Unrelated people push apart
  + Strong relationships create thicker, shorter springs
  + Clusters naturally form into tight groups.

+ The layout helps people instantly see the structure of your dataset:
  + Clusters like “RBC Developers,” “Seneca Alumni,” or “Data/AI People.”
  + Bridges between groups — people who link two communities.
  + Strong ties vs. weak ties at a glance.
  + Outliers who are isolated or unique.

### 5. Embeddings for Semantic Similarity:
+ Embeddings turn texts, like a person’s skills, job role, or bio, into a numeric vector that represents meaning, not just keywords. This lets the system understand relationships that aren’t obvious from the raw text.
+ For example, “backend engineer,” “Node.js developer,” and “API developer” will appear close together in vector space even though the words differ.
+ Using OpenAI’s text-embedding-3-small, we compute similarity between people in a consistent, mathematical way. These similarity scores:
  + Reveal hidden connections (e.g., two people with related but differently worded skillsets).
	+ Strengthen or weaken graph edges based on meaning, not exact wording.
  + Help cluster people into groups (e.g., SWE, Data, Biz, PM).
  + Improve ranking when choosing the best warm-introduction path.

## Service
```
cd service
python3 -m venv venv
source venv/bin/activate 

pip install -r requirements.txt
fastapi dev main.py
```


## Running a model
#### Graph
```
cd models
python graph.py
```

#### Clustering
```
cd models
python clustering.py
```

