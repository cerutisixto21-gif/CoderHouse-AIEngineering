"""Tests del estado compartido.

El test de reducers arma un grafo mínimo con dos nodos FALSOS (sin LLM, sin
red, sin costo) para demostrar cómo LangGraph fusiona los cambios de cada nodo.
"""

import pytest
from langgraph.graph import END, START, StateGraph

from app.state import DEFAULT_PORTFOLIO, AgentState, initial_state


# ---------- initial_state ----------


def test_initial_state_usa_cartera_por_defecto():
    state = initial_state("¿Cómo afectan los aranceles a mi cartera?")
    assert state["portfolio"] == DEFAULT_PORTFOLIO
    assert state["step_count"] == 0
    assert state["task_completed"] is False
    assert state["contributions"] == []
    assert state["messages"][0].content == "¿Cómo afectan los aranceles a mi cartera?"


def test_initial_state_no_comparte_la_cartera_por_defecto():
    state = initial_state("consulta")
    state["portfolio"]["SPY"] = 0.0
    assert DEFAULT_PORTFOLIO["SPY"] == 0.60  # el original no se tocó


def test_initial_state_rechaza_pesos_que_no_suman_uno():
    with pytest.raises(ValueError, match="sumar 1.0"):
        initial_state("consulta", {"SPY": 0.5, "TLT": 0.3})


def test_initial_state_rechaza_pesos_negativos():
    with pytest.raises(ValueError, match="negativos"):
        initial_state("consulta", {"SPY": 1.2, "TLT": -0.2})


def test_initial_state_rechaza_query_vacia():
    with pytest.raises(ValueError, match="vacía"):
        initial_state("   ")


# ---------- reducers ----------


def test_contributions_se_acumulan_y_step_count_se_reemplaza():
    def nodo_a(state: AgentState) -> dict:
        return {
            "step_count": 1,
            "contributions": [{"step": 1, "agent": "researcher", "summary": "noticias"}],
        }

    def nodo_b(state: AgentState) -> dict:
        return {
            "step_count": 2,
            "contributions": [{"step": 2, "agent": "analyst", "summary": "métricas"}],
        }

    builder = StateGraph(AgentState)
    builder.add_node("a", nodo_a)
    builder.add_node("b", nodo_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile()

    result = graph.invoke(initial_state("consulta"))

    # `contributions` tiene reducer (operator.add): se SUMAN las listas
    assert [c["agent"] for c in result["contributions"]] == ["researcher", "analyst"]
    # `step_count` NO tiene reducer: el último valor reemplaza al anterior
    assert result["step_count"] == 2
