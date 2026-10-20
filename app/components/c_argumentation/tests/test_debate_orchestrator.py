from app.components.c_argumentation.pipeline.debate_orchestrator import run_debate


def test_debate_alternates_agents():
    turns = run_debate({"facts": "x"}, rounds=1)
    assert [turn["agent"] for turn in turns] == ["plaintiff", "defense"]
