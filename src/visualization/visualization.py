import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import plot
from pandas.core.frame import DataFrame
from plotly.subplots import make_subplots
import os
import numpy as np
from ..options import BASE_PATH


class PlotlyRenderer:

    def __init__(self, df: DataFrame, options):
        self.df = df
        self.options = options

    def plot(self, renderer, filename_const):
        pio.renderers.default = renderer

        fig = make_subplots(
            rows=3, cols=4,
            column_widths=[.1, .1, .1, .1],
            row_heights=[.1, .1, .1],
            specs=[
                [{'type': 'xy', 'rowspan': 3, 'colspan': 2}, None, {'type': 'xy'}, {'type': 'xy'}],
                [None, None, {'type': 'xy'}, {'type': 'xy'}],
                [None, None, {'type': 'xy'}, {'type': 'pie'}]]
        )

        fig.add_trace(go.Scatter(x=self.df.missile_x, y=self.df.missile_z, marker=dict(color='red'), name='missile'),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=self.df.target_x, y=self.df.target_z, marker=dict(color='blue'), name='target'),
                      row=1, col=1)

        x, z = self._make_escape_zone_points()

        fig.add_trace(go.Scatter(x=x, y=z, name='escape zone', fill='toself', mode='lines'),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=self.df.t, y=self.df.missile_Ma * self.df.sOs, marker=dict(color='red'), name='missile velocity'),
                      row=1, col=3)

        fig.add_trace(go.Scatter(x=self.df.t, y=self.df.target_Ma * self.df.sOs, marker=dict(color='blue'), name='target velocity'),
                      row=1, col=3)

        fig.add_trace(go.Scatter(x=self.df.t, y=self.df.velR, marker=dict(color='green'), name='missile relative velocity'),
                      row=1, col=3)

        fig.add_trace(go.Scatter(x=self.df.t, y=np.rad2deg(self.df.missile_beta.to_numpy(dtype=np.float32)), marker=dict(color='red'), name='missile sideslip angle'),
                      row=2, col=3)

        fig.add_trace(go.Scatter(x=self.df.t, y=np.rad2deg(self.df.q.to_numpy(dtype=np.float32)), marker=dict(color='orange'), name='angle of view'),
                      row=3, col=3)

        fig.add_trace(go.Scatter(x=self.df.t, y=np.rad2deg(self.df.eps.to_numpy(dtype=np.float32)), marker=dict(color='purple'), name='missile angle error'),
                      row=3, col=3)

        fig.add_trace(go.Scatter(x=self.df.t, y=self.df.distance, marker=dict(color='red'), name='distance'),
                      row=1, col=4)

        fig.add_trace(go.Scatter(x=self.df.t, y=self.df.missile_overload, marker=dict(color='red'), name='missile overload'),
                      row=2, col=4)

        fig.add_trace(go.Scatter(x=self.df.t, y=self.df.target_overload, marker=dict(color='blue'), name='target overload'),
                      row=2, col=4)

        fig.add_trace(go.Pie(labels=['', ''],
                      values=[0, 100],
                      hole=0.85,
                      textinfo='none'), row=3, col=4)

        fig.update_layout(
            xaxis1=dict(range=[-90e3, 90e3]),
            yaxis1=dict(range=[-90e3, 90e3]),
            legend=dict(orientation='v', itemwidth=30, xanchor='left')
        )

        if renderer == "browser":
            path = os.path.join(BASE_PATH, 'files')
            FILENAME_ABS = self._make_abs_filename(path, filename_const)
            plot(fig, show_link=True, filename=FILENAME_ABS)
        else:
            fig.show()

    @staticmethod
    def _make_abs_filename(filepath, filename_const):
        filename = filename_const + '.html'
        i = 1
        while os.path.exists(os.path.join(filepath, filename)):
            filename = filename_const + f'_{i}' + '.html'
            i += 1
        return os.path.join(filepath, filename)

    def _make_escape_zone_points(self):
        D = 5e5

        escape_distance = self.options.env['bounds']['escape_distance']
        escape_sector_angle = self.options.env['bounds']['escape_sector_angle']
        direction_angle = self.options.missile['initial_state'][2]['psi'] + self.df.iloc[0].eps
        missile_initial_x = self.options.missile['initial_state'][2]['x']
        missile_initial_z = self.options.missile['initial_state'][2]['z']

        x, z = [], []

        angles = np.linspace(
            direction_angle - escape_sector_angle / 2, direction_angle + escape_sector_angle / 2, 20
        )

        for angle in angles:
            x.append(missile_initial_x + escape_distance * np.cos(angle))
            z.append(missile_initial_z + escape_distance * np.sin(angle))

        x_, z_ = x[-1] + D * np.cos(angles[-1]), z[-1] + D * np.sin(angles[-1])
        x__, z__ = x[0] + D * np.cos(angles[0]), z[0] + D * np.sin(angles[0])
        x.extend([x_, x__, x[0]])
        z.extend([z_, z__, z[0]])

        return x, z





