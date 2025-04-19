# utils/plot_utils.py
import pandas as pd
import os
import plotly.express as px

def generate_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: str,
    chart_type: str = "line",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if chart_type == "line":
        fig = px.line(
            df,
            x=x,
            y=y,
            title=title or f"{y} vs {x}",
            labels={x: xlabel or x, y: ylabel or y}
        )
        fig.update_traces(mode="markers+lines")

    elif chart_type == "pie":
        # x = grouping column, y = numeric value column
        fig = px.pie(
            df,
            names=x,
            values=y,
            title=title or f"{y} by {x}"
        )

    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    fig.write_image(output_path)  # requires kaleido
    return output_path