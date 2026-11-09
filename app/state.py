"""Estado compartido del orquestador multi-agente (Pre-entrega 6).

IDEA CLAVE: en LangGraph los nodos (supervisor, investigador, analista) no se
hablan entre sí. Todos leen y escriben sobre UN mismo objeto: el estado.
Cada nodo devuelve solo los campos que quiere modificar, y LangGraph los
fusiona en el estado.

¿Qué pasa si dos nodos escriben el mismo campo?
  - Sin "reducer": el valor nuevo REEMPLAZA al anterior.
  - Con "reducer" (Annotated[..., función]): LangGraph combina viejo + nuevo
    con esa función. Por ejemplo `operator.add` sobre listas las concatena.

Por eso `contributions` usa un reducer: queremos un REGISTRO que crece
(quién aportó qué), no que se pise en cada turno. `messages` ya trae el suyo
(`add_messages`) porque hereda de MessagesState.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

# Nombres de los nodos especialistas. Deben coincidir con los de `add_node`.
AgentName = Literal["researcher", "analyst"]

# Decisiones posibles del supervisor. Se usa como tipo de retorno de la
# arista condicional (lo exige la consigna: `Literal` en el retorno).
Route = Literal["researcher", "analyst", "FINISH"]

# Cartera de ejemplo del proyecto.
DEFAULT_PORTFOLIO: dict[str, float] = {"SPY": 0.60, "TLT": 0.30, "GLD": 0.10}


class Contribution(TypedDict):
    """Una entrada del registro de aportes: qué agente hizo qué, y en qué paso."""

    step: int
    agent: str
    summary: str


class AgentState(MessagesState):
    """Esquema del estado compartido. Hereda `messages` de MessagesState."""

    # --- Entrada del usuario ---
    query: str  # la pregunta o el evento a analizar
    portfolio: dict[str, float]  # ticker -> peso (los pesos suman 1.0)

    # --- Control de flujo (lo escribe el supervisor) ---
    next_agent: Route | None  # a quién le toca; None antes de la 1ra decisión
    instruction: str  # tarea ESPECÍFICA para el próximo especialista
    step_count: int  # cuántas veces decidió el supervisor (freno anti-bucle)
    task_completed: bool  # True cuando el supervisor da la tarea por terminada

    # --- Resultados de los especialistas ---
    research_findings: str  # lo que reunió el Investigador (con fuentes)
    analysis_results: dict  # métricas que calculó el Analista
    contributions: Annotated[list[Contribution], operator.add]  # registro acumulativo

    # --- Salida ---
    final_report: str  # informe final de riesgo (síntesis)


def initial_state(query: str, portfolio: dict[str, float] | None = None) -> AgentState:
    """Construye el estado inicial para ejecutar el grafo.

    Se inicializan TODOS los campos para que ningún nodo falle al leerlos.
    Valida que los pesos de la cartera sumen 1.0 (con tolerancia).
    """
    portfolio = dict(portfolio) if portfolio is not None else dict(DEFAULT_PORTFOLIO)

    if not query.strip():
        raise ValueError("La consulta (query) no puede estar vacía.")
    if not portfolio:
        raise ValueError("La cartera no puede estar vacía.")
    if any(w < 0 for w in portfolio.values()):
        raise ValueError("Los pesos de la cartera no pueden ser negativos.")
    total = sum(portfolio.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Los pesos de la cartera deben sumar 1.0 (suman {total:.4f}).")

    return {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "portfolio": portfolio,
        "next_agent": None,
        "instruction": "",
        "step_count": 0,
        "task_completed": False,
        "research_findings": "",
        "analysis_results": {},
        "contributions": [],
        "final_report": "",
    }
