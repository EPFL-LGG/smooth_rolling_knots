import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from scipy.spatial import ConvexHull
from src.geometry_src.rolliness import rolliness


def plotly_convex_hull(knot):
    # Compute convex hull
    hull = ConvexHull(knot)

    # Create 3D scatter plot for the knot
    knot_trace = go.Scatter3d(
        x=knot[:, 0], y=knot[:, 1], z=knot[:, 2],
        mode='lines', name='Knot', line=dict(color='blue')
    )

    # Create mesh3d for the convex hull (use all points, indices from hull.simplices)
    hull_trace = go.Mesh3d(
        x=knot[:, 0],
        y=knot[:, 1],
        z=knot[:, 2],
        i=hull.simplices[:, 0],
        j=hull.simplices[:, 1],
        k=hull.simplices[:, 2],
        opacity=0.5,
        name='Convex Hull',
        color='lightgray'
    )

    # Create figure and add traces
    fig = go.Figure(data=[knot_trace, hull_trace])
    fig.update_layout(title="Convex Hull of Knot", scene=dict(aspectmode='data'))
    
    return fig


def plotly_knot(knot):
    fig = go.Figure()
    fig.add_trace(
    go.Scatter3d(
        x=knot[:, 0], y=knot[:, 1], z=knot[:, 2],
        mode='lines', name='Knot', line=dict(color='blue')
    )
    )
    return fig



def plotly_plot(ko, heights=False):

    obj_hist = np.array(ko.hist[0])
    min_obj, max_obj = [max(1e-9, np.min(obj_hist)), min(np.max(obj_hist), 1e2)]
    min_obj = np.log10(min_obj)
    max_obj = np.log10(max_obj)
    print(f"Objective function range: [{min_obj}, {max_obj}]")
    pts = list(map(lambda x: ko.co.compute_curve_from_opt_params(x, closed_curve=False), ko.params_hist))
    curvature_range_1 = list(map(lambda x: x[ko.co.curvature_range_1], pts))
    curvature_range_2 = list(map(lambda x: x[ko.co.curvature_range_2], pts))
    knots = ko.knots_hist()
    print(f"Number of knots: {len(knots)}")
    if heights:
        rho_heights = list(map(lambda x: rolliness(np.array(x)), knots)) 
        rhos = np.array([rho[0] for rho in rho_heights])
        heights = [rho[1] for rho in rho_heights]
        min_height = np.min([np.min(h) for h in heights])
        max_height = np.max([np.max(h) for h in heights])

    # Prepare traces for the 3D plot
    stretched_knot = ko.stretched_knot
    tdr1 = ko.tdr1

    def make_3d_traces(idx):
        traces = []
        # Stretched knot (red)
        traces.append(go.Scatter3d(
            x=stretched_knot[:,0], y=stretched_knot[:,1], z=stretched_knot[:,2],
            mode='lines', name='Stretched knot', line=dict(color='red')
        ))
        # TDR (green)
        traces.append(go.Scatter3d(
            x=tdr1[:,0], y=tdr1[:,1], z=tdr1[:,2],
            mode='lines', name='TDR', line=dict(color='green')
        ))
        # Projected knot (blue)
        knot = knots[int(idx)]
        traces.append(go.Scatter3d(
            x=knot[:,0], y=knot[:,1], z=knot[:,2],
            mode='lines', name='Projected knot', line=dict(color='blue')
        ))
        # Curvature range 1 (orange)
        traces.append(go.Scatter3d(
            x=curvature_range_1[idx][:,0], y=curvature_range_1[idx][:,1], z=curvature_range_1[idx][:,2],
            mode='lines', name='Curvature range', line=dict(color='orange')
        ))

        # Curvature range 2 (also orange)
        traces.append(go.Scatter3d(
            x=curvature_range_2[idx][:,0], y=curvature_range_2[idx][:,1], z=curvature_range_2[idx][:,2],
            mode='lines', line=dict(color='orange')
        ))

        return traces

    # Prepare traces for the 2D plot (optimization history)
    def make_2d_traces(idx):
        traces = []
        traces.append(go.Scatter(
            y=obj_hist[:,1], mode='lines', name='Knot'
        ))
        traces.append(go.Scatter(
            y=obj_hist[:,2], mode='lines', name='TDR'
        ))
        traces.append(go.Scatter(
            y=obj_hist[:,3], mode='lines', name='Curvature'
        ))
        traces.append(go.Scatter(
            y=obj_hist[:,0], mode='lines', name='Objective function'
        ))
        # Vertical line at current idx
        traces.append(go.Scatter(
            x=[idx, idx], y=[np.min(obj_hist), np.max(obj_hist)],
            mode='lines', name='Current idx', line=dict(color='black', dash='dash')
        ))
        if heights:
            traces.append(go.Scatter(
                y=rhos, mode='lines', name='Rolliness', line=dict(color='purple')
            ))
        return traces

    # --- Convex hull plot ---
    def make_hull_traces(idx):
        knot = knots[int(idx)]
        hull = ConvexHull(knot)
        hull_trace = go.Mesh3d(
            x=knot[:, 0],
            y=knot[:, 1],
            z=knot[:, 2],
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            opacity=0.5,
            name='Convex Hull',
            color='lightgray'
        )
        knot_trace = go.Scatter3d(
            x=knot[:, 0], y=knot[:, 1], z=knot[:, 2],
            mode='lines', name='Knot', line=dict(color='blue')
        )
        return [knot_trace, hull_trace]

    if heights:
        # --- Heights 2D plot ---
        def make_heights_traces(idx):
            traces = []
            traces.append(go.Scatter(
                y=heights[idx], mode='lines+markers', name='Height'
            ))
            return traces

    fig3d = go.FigureWidget(make_3d_traces(len(knots)-1))
    fig3d.update_layout(
        title="Knot optimization (3D)",
        scene=dict(aspectmode='data'),
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    fig2d = go.FigureWidget(make_2d_traces(len(knots)-1))
    fig2d.update_layout(
        title="Optimization history",
        height=300,
        showlegend=True,
        yaxis_type="log",
        yaxis=dict(range=[min_obj, max_obj]),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )

    # Convex hull figure
    hull_traces = make_hull_traces(len(knots)-1)
    hull_fig = go.FigureWidget(hull_traces)
    hull_fig.update_layout(
        title="Convex Hull of Knot",
        scene=dict(aspectmode='data'),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )
    if heights:
        # Add rho annotation to hull_fig
        hull_fig.add_annotation(
            dict(
                showarrow=False,
                text=f"ρ = {rhos[-1]:.3f}",
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                xanchor="right", yanchor="bottom",
                font=dict(size=14, color="black"),
                bgcolor="white", opacity=0.8
            )
        )

    if heights:
        # Heights 2D figure
        heights_fig = go.FigureWidget(make_heights_traces(len(knots)-1))
        heights_fig.update_layout(
            title="Height over optimization",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            yaxis=dict(range=[min_height, max_height]),
        )

    slider = widgets.IntSlider(
        value=len(knots)-1,
        min=0,
        max=len(knots)-1,
        step=1,
        description='Knot idx',
        continuous_update=True
    )

    def update(idx):
        idx = int(idx)
        knot = knots[idx]
        # 3D traces
        with fig3d.batch_update():
            # Projected knot (blue)
            fig3d.data[2].x = knot[:,0]
            fig3d.data[2].y = knot[:,1]
            fig3d.data[2].z = knot[:,2]
            # Curvature range 1 (orange)
            fig3d.data[3].x = curvature_range_1[idx][:,0]
            fig3d.data[3].y = curvature_range_1[idx][:,1]
            fig3d.data[3].z = curvature_range_1[idx][:,2]
            # Curvature range 2 (also orange)
            fig3d.data[4].x = curvature_range_2[idx][:,0]
            fig3d.data[4].y = curvature_range_2[idx][:,1]
            fig3d.data[4].z = curvature_range_2[idx][:,2]

        # 2D traces
        with fig2d.batch_update():
            fig2d.data[4].x = [idx, idx]
            fig2d.data[4].y = [np.min(obj_hist), np.max(obj_hist)]

        if heights:
            # Heights plot update
            with heights_fig.batch_update():
                heights_fig.data[0].y = heights[idx]

        # Convex hull update
        hull_traces = make_hull_traces(idx)
        with hull_fig.batch_update():
            hull_fig.data[0].x = hull_traces[0].x
            hull_fig.data[0].y = hull_traces[0].y
            hull_fig.data[0].z = hull_traces[0].z
            hull_fig.data[1].x = hull_traces[1].x
            hull_fig.data[1].y = hull_traces[1].y
            hull_fig.data[1].z = hull_traces[1].z
            hull_fig.data[1].i = hull_traces[1].i
            hull_fig.data[1].j = hull_traces[1].j
            hull_fig.data[1].k = hull_traces[1].k
            # Update rho annotation
            hull_fig.layout.annotations = []  # Remove previous annotation
            hull_fig.add_annotation(
                dict(
                    showarrow=False,
                    text=f"ρ = {rhos[idx]:.3f}" if heights else "",
                    xref="paper", yref="paper",
                    x=0.99, y=0.01,
                    xanchor="right", yanchor="bottom",
                    font=dict(size=14, color="black"),
                    bgcolor="white", opacity=0.8
                )
            )

    def on_slider_change(change):
        if change['name'] == 'value':
            update(change['new'])

    slider.observe(on_slider_change, names='value')

    vbox = [slider, fig2d, hull_fig]
    if heights: vbox.append(heights_fig) 
    # Display: fig3d on the left, slider, loss plot, hull plot, and heights plot stacked vertically on the right
    display(
        widgets.HBox([
            fig3d,
            widgets.VBox(vbox)
        ])
    )

    # Initial update
    update(slider.value)

