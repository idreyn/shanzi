from shanzi.ids import build_component_graph, parse_ids_expression, recursive_components


def test_parse_ids_expression_roundtrip():
    node = parse_ids_expression("⿰日西")
    assert node.serialize() == "⿰日西"
    assert len(node.children) == 2
    assert node.children[0].token == "日"
    assert node.children[1].token == "西"


def test_component_graph_handles_nested_ids_nodes():
    decompositions = {
        "晒": "⿰日西",
        "西": "⿱一⿰儿口",
        "兀": "⿱一儿",
    }
    graph = build_component_graph(decompositions)
    assert graph["晒"] == ["日", "西"]
    assert graph["西"][0] == "一"
    assert graph["西"][1].startswith("expr:")
    assert graph[graph["西"][1]] == ["儿", "口"]

    recursive = recursive_components("晒", graph)
    for token in ("日", "西", "一", "儿", "口"):
        assert token in recursive
