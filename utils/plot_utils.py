# utils/plot_utils.py
import pandas as pd
import os
import plotly.express as px

def generate_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
) -> str:
    """
    Builds a Plotly line plot from df[x] vs df[y], saves it to output_path as PNG, and returns that path.
    Requires `kaleido` for static image export.
    """
    # 1) Make sure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2) Build the figure
    fig = px.line(
        df,
        x=x,
        y=y,
        title=title or f"{y} vs {x}",
        labels={ x: xlabel or x, y: ylabel or y }
    )
    fig.update_traces(mode="markers+lines")

    # 3) Export to PNG
    fig.write_image(output_path)  # requires kaleido

    return output_path
