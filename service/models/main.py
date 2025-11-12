import json
from pathlib import Path

import networkx as nx
from graph import build_graph, shortest_paths_ranked


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "people.json"


if __name__ == "__main__":
    with DATA_PATH.open("r", encoding="utf-8") as f:
        people = json.load(f)

    G = build_graph(people)


    source_id = "cris-huynh-2a52b5274"       # change to an actual id in your file
    target_id = "khoivu"    # change to an actual id in your file
    topk = shortest_paths_ranked(G, source_id, target_id, max_hops=6, k=3)

    def name(nid): 
        return G.nodes[nid]["name"]

    if not topk:
        print("No path ≤ 6 hops.")
    else:
        for r in topk:
            print(f"PATH ({r['hops']} hops, score={r['score']}):",
                  " -> ".join(name(n) for n in r["nodes"]))
            for hop in r["hops_detail"]:
                print("  ", name(hop["from"]), "→", name(hop["to"]),
                      "| w =", hop["edge_weight"],
                      "| signals =", hop["signals"])
