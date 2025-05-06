from textwrap import dedent

from py2cfg import CFGBuilder
import networkx as nx
from utils import set_timeout


def walk_cfg(current_graph, used, cfg_node, nid_map):
    if cfg_node.id not in nid_map:
        nid_map[cfg_node.id] = len(nid_map)
    nid = nid_map[cfg_node.id]
    while nid >= len(used):
        used.append(False)
    used[nid] = True
    for link in cfg_node.exits:
        next = link.target
        if next.id not in nid_map:
            nid_map[next.id] = len(nid_map)
        next_id = nid_map[next.id]
        current_graph.append((nid, next_id))
        if next_id >= len(used) or not used[next_id]:
            walk_cfg(current_graph, used, next, nid_map)


def code_to_cfg_edges(text):
    cfg = CFGBuilder().build_from_src('text', text)
    a = cfg.entryblock
    graph = []
    used = []
    nid_map = {}
    walk_cfg(graph, used, a, nid_map)
    return graph


def preprocess(code):
    if code.startswith("def"):
        code = code[code.find('\n') + 1:]

    return dedent(code)

def cfg_triviality(text):
    text = preprocess(text)
    try:
        edges = code_to_cfg_edges(text)
    except: # This is dumb
        return -1.0
    return max(0.0, 1 - len(edges) / 4)


def code_cfg_similarity(text1, text2):
    text1 = preprocess(text1)
    text2 = preprocess(text2)
    try:
        edges1 = code_to_cfg_edges(text1)
        edges2 = code_to_cfg_edges(text2)
    except: # SyntaxError or AttributeError since generated code is something weird, idk
        return 0.0
    div_const = 2 * (2 + len(edges1) + len(edges2))
    g1, g2 = nx.MultiDiGraph(edges1), nx.MultiDiGraph(edges2)
    if len(edges1) + len(edges2) == 0: # Both graphs are empty (aka no edges, like a simple return statement)
        return 1.0
    if len(edges1) > 0 and len(edges2) > 0:
        result = 1 - nx.graph_edit_distance(g1, g2, roots=(0, 0), timeout=60) / div_const
    else:
        result = 1 - nx.graph_edit_distance(g1, g2, timeout=60) / div_const
    if result < 0:
        raise Exception("What the hell???")
    return result


if __name__ == "__main__":
    pass
