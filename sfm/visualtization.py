import plotly.graph_objs as go
import numpy as np
from  plotly.offline import iplot,init_notebook_mode

def get_plot(x, y, z):
    init_notebook_mode(False)
    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=1.0,
            line=dict(
                width=0.5
            ),
            opacity=0.8
        )
    )
    
    layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4,),
                    yaxis = dict(
                        nticks=4,),
                    zaxis = dict(
                        nticks=4,),),
                    width=700,
                    margin=dict(
                    r=20, l=10,
                    b=10, t=10)
                  )
    fig = go.Figure(data=[trace1], layout=layout)

    return iplot(fig, filename='bla')

def get_plot_covs(x, y, z, cov_x, cov_y, cov_z):
    init_notebook_mode(False)
    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=4.0,
            line=dict(
                width=2,
                color=np.sqrt(cov_x ** 2 + cov_y ** 2 + cov_z ** 2)
            ),
            color=np.sqrt(cov_x ** 2 + cov_y ** 2 + cov_z ** 2),
            opacity=0.8,
            colorbar=dict(title="Variance")
        )
    )
    
    layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4,),
                    yaxis = dict(
                        nticks=4,),
                    zaxis = dict(
                        nticks=4,),),
                    width=700,
                    margin=dict(
                    r=20, l=10,
                    b=10, t=10),
                  )
    fig = go.Figure(data=[trace1], layout=layout)

    return iplot(fig, filename='bla')
