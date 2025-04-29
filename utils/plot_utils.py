import pandas as pd
import os
import matplotlib.pyplot as plt


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
    """
    Generate and save a plot using matplotlib.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x (str): Column name for the x-axis or labels.
        y (str): Column name for the y-axis or values.
        output_path (str): File path to save the output image.
        chart_type (str, optional): 'line' or 'pie'. Defaults to 'line'.
        title (str, optional): Plot title. Defaults to None.
        xlabel (str, optional): Label for x-axis. Defaults to None.
        ylabel (str, optional): Label for y-axis. Defaults to None.

    Returns:
        str: Path to the saved image file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if chart_type == "line":
        # Line plot with markers
        plt.figure()
        plt.plot(df[x], df[y], marker='o')
        plt.title(title or f"{y} vs {x}")
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)

    elif chart_type == "pie":
        # Pie chart
        plt.figure()
        plt.pie(
            df[y],
            labels=df[x],
            autopct='%1.1f%%'
        )
        plt.title(title or f"{y} by {x}")

    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
