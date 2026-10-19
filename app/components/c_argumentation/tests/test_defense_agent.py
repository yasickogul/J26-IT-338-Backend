from app.components.c_argumentation.agents.defense_agent import generate_defense_turn


def test_defense_turn_has_agent_and_round():
    turn = generate_defense_turn({"facts": "x"}, 1)
    assert turn["agent"] == "defense"
    assert turn["round"] == 1
