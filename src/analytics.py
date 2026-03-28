"""Health dynamics tracking and charting."""

import re
from typing import List, Dict, Any, Optional

import plotly.graph_objects as go

from src.vector_store import VectorStore


def extract_lab_values(text: str, parameter_name: str) -> List[Dict[str, Any]]:
    """Extract numeric lab values for a given parameter from text chunks.

    Looks for patterns like:
    - Гемоглобин: 135 г/л
    - Гемоглобин - 135
    - гемоглобин 135.5 г/л
    """
    pattern = re.compile(
        rf"{re.escape(parameter_name)}\s*[:=\-–—]\s*(\d+[.,]?\d*)\s*([\w/]*)",
        re.IGNORECASE,
    )

    values = []
    for match in pattern.finditer(text):
        value_str = match.group(1).replace(",", ".")
        unit = match.group(2) or ""
        try:
            value = float(value_str)
            values.append({"value": value, "unit": unit})
        except ValueError:
            continue

    return values


def search_parameter_history(
    vector_store: VectorStore, parameter_name: str, n_results: int = 20
) -> List[Dict[str, Any]]:
    """Search the vector store for all mentions of a lab parameter."""
    results = vector_store.search(parameter_name, n_results=n_results)

    entries = []
    for r in results:
        values = extract_lab_values(r["text"], parameter_name)
        if values:
            meta = r["metadata"]
            for v in values:
                entries.append({
                    "value": v["value"],
                    "unit": v["unit"],
                    "source": meta.get("file_name", "?"),
                    "text_fragment": r["text"][:200],
                })

    return entries


def create_parameter_chart(
    entries: List[Dict[str, Any]], parameter_name: str
) -> Optional[go.Figure]:
    """Create a plotly chart for a lab parameter over time."""
    if not entries:
        return None

    values = [e["value"] for e in entries]
    labels = [e.get("source", f"#{i+1}") for i, e in enumerate(entries)]
    unit = entries[0].get("unit", "")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(values) + 1)),
        y=values,
        mode="lines+markers",
        name=parameter_name,
        text=labels,
        hovertemplate="%{text}<br>Значение: %{y}<extra></extra>",
        line=dict(color="#2196F3", width=2),
        marker=dict(size=8),
    ))

    fig.update_layout(
        title=f"Динамика: {parameter_name}",
        xaxis_title="Измерения",
        yaxis_title=f"{parameter_name} ({unit})" if unit else parameter_name,
        template="plotly_white",
        height=400,
    )

    return fig
