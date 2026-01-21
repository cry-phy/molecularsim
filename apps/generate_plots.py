import numpy as np
import plotly.graph_objects as go
import os

os.makedirs("out/", exist_ok=True)


DRACULA = {
    "bg": "#282a36",
    "current": "#44475a",
    "fg": "#f8f8f2",
    "comment": "#6272a4",
    "cyan": "#8be9fd",
    "green": "#50fa7b",
    "orange": "#ffb86c",
    "pink": "#ff79c6",
    "purple": "#bd93f9",
    "red": "#ff5555",
    "yellow": "#f1fa8c",
}


rows, cols = 12, 5
mask = np.tri(rows, cols, k=-1, dtype=int)

fig = go.Figure(
    data=go.Heatmap(
        z=mask,
        colorscale=[
            [0.0, DRACULA["red"]],  # 0 → red
            [0.499, DRACULA["red"]],
            [0.5, DRACULA["green"]],  # 1 → green
            [1.0, DRACULA["green"]],
        ],
        zmin=0,
        zmax=1,
        showscale=False,
    )
)

fig.update_layout(
    title="Bitmask (j ≤ i)",
    paper_bgcolor=DRACULA["bg"],
    plot_bgcolor=DRACULA["bg"],
    font=dict(color=DRACULA["fg"]),
    width=500,
    height=500,
)

# Hide axes completely
fig.update_xaxes(
    visible=False,
    showgrid=False,
    zeroline=False,
)

fig.update_yaxes(
    visible=False,
    showgrid=False,
    zeroline=False,
    scaleanchor="x",
    scaleratio=1,
    autorange="reversed",
)
fig.write_html("out/bitmask.html", include_plotlyjs="cdn")

fig.show()
