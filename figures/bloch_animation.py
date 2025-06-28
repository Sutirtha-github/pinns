import numpy as np
import plotly.graph_objects as go
from matplotlib import colormaps as cm


def create_interactive_bloch_sphere(sx, sy, sz, t_eval):
    """
    Create an interactive Bloch sphere animation with colored trajectory.

    Parameters:
    - sx, sy, sz: arrays of Bloch vector components
    - t_eval: array of time points

    Returns:
    - fig: plotly.graph_objects.Figure with animation
    """

    # Generate colors for trajectory
    cmap = cm.get_cmap("plasma")
    colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b, _ in cmap(np.linspace(0, 1, len(t_eval)))]

    # Create Bloch sphere grid (wireframe)
    u = np.linspace(0, 2 * np.pi, 24)  # Longitude
    v = np.linspace(0, np.pi, 12)      # Latitude
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Create figure
    fig = go.Figure()

    # Add Bloch sphere grid lines
    for i in range(len(u)):  # Longitude lines
        fig.add_trace(go.Scatter3d(x=x[i, :], y=y[i, :], z=z[i, :], mode="lines", line=dict(color="gray", width=1), showlegend=False))
    for j in range(len(v)):  # Latitude lines
        fig.add_trace(go.Scatter3d(x=x[:, j], y=y[:, j], z=z[:, j], mode="lines", line=dict(color="gray", width=1), showlegend=False))

    # Add XYZ axes
    fig.add_trace(go.Cone(x=[1.2], y=[0], z=[0], u=[1], v=[0], w=[0], sizemode="absolute", sizeref=0.1, colorscale=[[0, "red"], [1, "red"]], showscale=False))
    fig.add_trace(go.Cone(x=[0], y=[1.2], z=[0], u=[0], v=[1], w=[0], sizemode="absolute", sizeref=0.1, colorscale=[[0, "green"], [1, "green"]], showscale=False))
    fig.add_trace(go.Cone(x=[0], y=[0], z=[1.2], u=[0], v=[0], w=[1], sizemode="absolute", sizeref=0.1, colorscale=[[0, "blue"], [1, "blue"]], showscale=False))

    fig.add_trace(go.Scatter3d(x=[0, 1.2], y=[0, 0], z=[0, 0], mode="lines", line=dict(color="red", width=5), name="X-axis"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 1.2], z=[0, 0], mode="lines", line=dict(color="green", width=5), name="Y-axis"))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.2], mode="lines", line=dict(color="blue", width=5), name="Z-axis"))


    # Add trajectory with gradient color
    for i in range(len(t_eval) - 1):
        fig.add_trace(go.Scatter3d(
            x=[sx[i], sx[i+1]], y=[sy[i], sy[i+1]], z=[sz[i], sz[i+1]],
            mode="lines", line=dict(color=colors[i], width=5),
            showlegend=False
        ))


    # Add animated Bloch vector
    vector = go.Scatter3d(
        x=[0, sx[0]], y=[0, sy[0]], z=[0, sz[0]],
        mode="lines+markers",
        marker=dict(size=4, color="white", line=dict(color="black", width=3)),
        line=dict(color=colors[0], width=8),
        name="Bloch Vector"
    )
    fig.add_trace(vector)

    # Create animation frames
    frames = [
        go.Frame(
            data=[
                go.Scatter3d(x=[0, sx[i]], y=[0, sy[i]], z=[0, sz[i]],
                            mode="lines+markers",
                            marker=dict(size=4, color="white", line=dict(color="black", width=2)),
                            line=dict(color=colors[i], width=4))
            ]
        ) for i in range(len(t_eval))
    ]

    fig.frames = frames

    # Update layout for interactivity
    fig.update_layout(
        title="Interactive Bloch Sphere",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="cube"
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])]
        )],
    )

    return fig
