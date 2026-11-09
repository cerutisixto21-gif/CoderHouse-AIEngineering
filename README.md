# Mini-Aladdin — Orquestador multi-agente de análisis de riesgo

Sistema multi-agente (LangGraph) que, dada una **cartera pequeña** y un **evento o pregunta**, produce un **informe de riesgo**. Inspirado, en una versión muy acotada, en el tipo de análisis que hace Aladdin (BlackRock): combinar eventos del mundo, precios históricos y métricas de riesgo.

> ⚠️ **Aviso:** este proyecto es un trabajo académico. Sus resultados son **análisis informativo**, no asesoramiento financiero.

**Estado:** Pre-entrega 6 en desarrollo.

## Instalación

Requiere Python 3.12 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env             # y completá tus claves en .env
pytest
```

> Las claves van **solo** en `.env`, que está excluido del repositorio (`.gitignore`).
