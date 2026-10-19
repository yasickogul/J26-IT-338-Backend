from app.components.c_argumentation.agents.plaintiff_agent import generate_plaintiff_turn


def test_plaintiff_turn_has_agent_and_round():
    turn = generate_plaintiff_turn({"facts": "x"}, 1)
    assert turn["agent"] == "plaintiff"
    assert turn["round"] == 1
