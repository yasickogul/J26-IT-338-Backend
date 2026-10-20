"""Build the JSON-compatible argument graph returned by C4."""

from collections.abc import Iterable, Mapping


def build_argument_graph(turns: Iterable[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Convert debate turns into nodes and sequential reply edges."""
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    previous_id: str | None = None
    for index, turn in enumerate(turns, start=1):
        node_id = str(turn.get("id", f"n{index}"))
        node = dict(turn)
        node["id"] = node_id
        nodes.append(node)
        if previous_id is not None:
            edges.append({"from": node_id, "to": previous_id, "relation": "responds_to"})
        previous_id = node_id
    return {"nodes": nodes, "edges": edges}
