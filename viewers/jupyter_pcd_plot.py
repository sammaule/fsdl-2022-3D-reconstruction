import plotly.graph_objects as go
import numpy as np

# function from https://colab.research.google.com/drive/1CR_HDvJ2AnjJV3Bf5vwP70K0hx3RcdMb?usp=sharing#scrollTo=6LPKVOCjEtjv


def draw_geometries(pcd, height_threshold=None):
    graph_objects = []

    array_points = np.asarray(pcd.points)
    colors_array = np.asarray(pcd.colors)

    if height_threshold:

        array_points_shorter = array_points[array_points[:, 2] < height_threshold]
        colors_array_shorter = colors_array[array_points[:, 2] < height_threshold]
    else:
        array_points_shorter = array_points
        colors_array_shorter = colors_array

    graph_objects = []
    points = array_points_shorter
    colors = colors_array_shorter

    scatter_3d = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=1, color=colors),
    )
    graph_objects.append(scatter_3d)

    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        ),
    )
    return fig
