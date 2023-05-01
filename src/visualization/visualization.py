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

        init = self.df.iloc[0]

        # ---- 0 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.missile_x],
                y=[init.missile_z],
                marker=dict(color='red'),
                name='missile',
                mode='lines'
            ),
            row=1,
            col=1
        )

        # ---- 1 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.target_x],
                y=[init.target_z],
                marker=dict(color='blue'),
                name='target',
                mode='lines'
            ),
            row=1,
            col=1
        )

        # ---- 2 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.missile_x, init.target_x],
                y=[init.missile_z, init.target_z],
                marker=dict(color='green', size=0.5),
                name='los',
                mode='lines'
            ),
            row=1,
            col=1
        )

        x, z = self._make_escape_zone_points()

        # ---- 3 trace ----
        fig.add_trace(
            go.Scatter(
                x=x,
                y=z,
                name='escape zone',
                fill='toself',
                mode='lines',
                visible=False
            ),
            row=1,
            col=1
        )

        # ---- 4 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.t],
                y=[init.missile_Ma * init.sOs],
                marker=dict(color='red'),
                name='missile velocity',
                mode='lines'
            ),
            row=1,
            col=3
        )

        # ---- 5 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.t],
                y=[init.target_Ma * init.sOs],
                marker=dict(color='blue'),
                name='target velocity',
                mode='lines'
            ),
            row=1,
            col=3
        )

        # ---- 6 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.t],
                y=[init.velR],
                marker=dict(color='green'),
                name='missile relative velocity',
                mode='lines'
            ),
            row=1,
            col=3
        )

        # ---- 7 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.t],
                y=[np.rad2deg(init.missile_beta)],
                marker=dict(color='red'),
                name='missile sideslip angle',
                mode='lines'
            ),
            row=2,
            col=3
        )

        # ---- 8 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.t],
                y=[np.rad2deg(init.q)],
                marker=dict(color='orange'),
                name='missile angle of view',
                mode='lines'
            ),
            row=3,
            col=3
        )

        # ---- 9 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.t],
                y=[np.rad2deg(init.eps)],
                marker=dict(color='purple'),
                name='missile angle error',
                mode='lines'
            ),
            row=3,
            col=3
        )

        # ---- 10 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.t],
                y=[init.distance],
                marker=dict(color='red'),
                name='distance',
                mode='lines'
            ),
            row=1,
            col=4
        )

        # ---- 11 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.t],
                y=[init.missile_overload],
                marker=dict(color='red'),
                name='missile overload',
                mode='lines'
            ),
            row=2,
            col=4
        )

        # ---- 12 trace ----
        fig.add_trace(
            go.Scatter(
                x=[init.t],
                y=[init.target_overload],
                marker=dict(color='blue'),
                name='target overload',
                mode='lines'
            ),
            row=2,
            col=4
        )

        # ---- 13 trace ----
        fig.add_trace(
            go.Pie(
                labels=['', ''],
                values=[init.t, 150 - init.t],
                hole=0.85,
                textinfo='none',
                marker=dict(colors=['rgb(113,209,145)', 'rgb(240,240,240)']),
                showlegend=False
            ),
            row=3,
            col=4
        )

        # ---- layout ----
        fig.update_layout(
            plot_bgcolor='rgb(240,240,240)',
            font=dict(size=10),
            margin=dict(pad=0),

            width=1000,
            height=550,

            # ---- 1 subplot ----
            xaxis1=dict(
                range=[-90e3, 90e3],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='x [m]', standoff=1, font=dict(size=10)),
                nticks=12
            ),
            yaxis1=dict(
                range=[-90e3, 90e3],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='z [m]', standoff=1, font=dict(size=10)),
                nticks=12
            ),
            # ---- 2 subplot ----
            xaxis2=dict(
                range=[0, self.df.iloc[-1].t],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text=' [sec]', standoff=1, font=dict(size=10)),
            ),
            yaxis2=dict(
                range=[0, 2000],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='vel [m/sec]', standoff=1, font=dict(size=10)),
            ),
            # ---- 3 subplot ----
            xaxis3=dict(
                range=[0, self.df.iloc[-1].t],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='t [sec]', standoff=1, font=dict(size=10)),
            ),
            yaxis3=dict(
                range=[-1e3, init.distance],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='distance [m]', standoff=1, font=dict(size=10)),
            ),
            # ---- 4 subplot ----
            xaxis4=dict(
                range=[0, self.df.iloc[-1].t],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='t [sec]', standoff=1, font=dict(size=10)),
            ),
            yaxis4=dict(
                range=[-40, 40],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='[grad]', standoff=1, font=dict(size=10)),
            ),
            # ---- 5 subplot ----
            xaxis5=dict(
                range=[0, self.df.iloc[-1].t],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='t [sec]', standoff=1, font=dict(size=10)),
            ),
            yaxis5=dict(
                range=[-50, 50],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='overload [g]', standoff=1, font=dict(size=10)),
            ),
            # ---- 6 subplot ----
            xaxis6=dict(
                range=[0, self.df.iloc[-1].t],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='t [sec]', standoff=1, font=dict(size=10)),
            ),
            yaxis6=dict(
                range=[-180, 180],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1,
                gridcolor='lightgrey',
                title=dict(text='[grad]', standoff=1, font=dict(size=10)),
            ),

            legend=dict(orientation='v', itemwidth=30, xanchor='left'),
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        duration=0,
                                        redraw=True
                                    ),
                                    fromcurrent=True
                                )
                            ]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[
                                [None],
                                dict(
                                    frame=dict(
                                        duration=0,
                                        redraw=True
                                    ),
                                    mode='immediate',
                                    transition=dict(duration=0)
                                )
                            ]
                        )
                    ],
                    showactive=False,
                    yanchor='bottom',
                    xanchor='left',
                    x=0.,
                    y=-.25
                )
            ]
        )

        sliders_dict = dict(
            active=0,
            yanchor='bottom',
            xanchor='left',
            currentvalue=dict(
                font=dict(
                    size=12
                ),
                prefix='time:',
                visible=True,
                xanchor='right'
            ),
            transition=dict(
                duration=0,
                easing='cubic-in-out'
            ),
            pad=dict(
                b=-88,
                t=50
            ),
            len=1.1,
            x=0.1,
            y=0,
            steps=[]
        )

        # ---- frames ----
        frames = []
        for i, _ in enumerate(self.df.t):
            slider_step = dict(
                args=[
                    [f'frame {i}'],
                    dict(
                        frame=dict(
                            duration=0,
                            redraw=True
                        ),
                        mode='immediate',
                        transition=dict(
                            duration=0
                        )
                    )
                ],
                value=f'{self.df.iloc[i].t:.2}',
                label=f'{self.df.iloc[i].t:.2}',
                visible=True,
                method='animate'
            )

            sliders_dict['steps'].append(slider_step)

            frames.append(
                go.Frame(
                    data=[
                        # ---- 0 trace ----
                        go.Scatter(
                            x=self.df.missile_x[:i],
                            y=self.df.missile_z[:i],
                            marker=dict(color='red'),
                            name='missile',
                            mode='lines'
                        ),
                        # ---- 1 trace ----
                        go.Scatter(
                            x=self.df.target_x[:i],
                            y=self.df.target_z[:i],
                            marker=dict(color='blue'),
                            name='target',
                            mode='lines'
                        ),
                        # ---- 2 trace ----
                        go.Scatter(
                            x=[self.df.iloc[i].missile_x, self.df.iloc[i].target_x],
                            y=[self.df.iloc[i].missile_z, self.df.iloc[i].target_z],
                            marker=dict(color='green', size=0.5),
                            name='los',
                            mode='lines'
                        ),
                        # ---- 4 trace ----
                        go.Scatter(
                            x=self.df.t[:i],
                            y=self.df.missile_Ma[:i] * self.df.sOs[:i],
                            marker=dict(color='red'),
                            name='missile velocity',
                            mode='lines'
                        ),
                        # ---- 5 trace ----
                        go.Scatter(
                            x=self.df.t[:i],
                            y=self.df.target_Ma[:i] * self.df.sOs[:i],
                            marker=dict(color='blue'),
                            name='target velocity',
                            mode='lines'
                        ),
                        # ---- 6 trace ----
                        go.Scatter(
                            x=self.df.t[:i],
                            y=self.df.velR[:i],
                            marker=dict(color='green'),
                            name='missile relative velocity',
                            mode='lines'
                        ),
                        # ---- 7 trace ----
                        go.Scatter(
                            x=self.df.t[:i],
                            y=np.rad2deg(self.df.missile_beta[:i]),
                            marker=dict(color='red'),
                            name='missile sideslip angle',
                            mode='lines'
                        ),
                        # ---- 8 trace ----
                        go.Scatter(
                            x=self.df.t[:i],
                            y=np.rad2deg(self.df.q[:i]),
                            marker=dict(color='orange'),
                            name='missile angle of view',
                            mode='lines'
                        ),
                        # ---- 9 trace ----
                        go.Scatter(
                            x=self.df.t[:i],
                            y=np.rad2deg(self.df.eps[:i]),
                            marker=dict(color='purple'),
                            name='missile angle error',
                            mode='lines'
                        ),
                        # ---- 10 trace ----
                        go.Scatter(
                            x=self.df.t[:i],
                            y=self.df.distance[:i],
                            marker=dict(color='red'),
                            name='distance',
                            mode='lines'
                        ),
                        # ---- 11 trace ----
                        go.Scatter(
                            x=self.df.t[:i],
                            y=self.df.missile_overload[:i],
                            marker=dict(color='red'),
                            name='missile overload',
                            mode='lines'
                        ),
                        # ---- 12 trace ----
                        go.Scatter(
                            x=self.df.t[:i],
                            y=self.df.target_overload[:i],
                            marker=dict(color='blue'),
                            name='target overload',
                            mode='lines'
                        ),
                        # ---- 13 trace ----
                        go.Pie(
                            title='time usage [sec]',
                            labels=['1', '2'],
                            values=[self.df.iloc[i].t, 150 - self.df.iloc[i].t],
                            hole=0.85,
                            textinfo='none',
                            marker=dict(colors=['rgb(113,209,145)', 'rgb(240,240,240)']),
                            showlegend=False
                        ),
                    ],
                    traces=[
                        0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
                    ],
                    name=f'frame {i}'
                )
            )

        fig.layout['sliders'] = [sliders_dict]
        fig.frames = frames

        if renderer == "browser":
            path = os.path.join(BASE_PATH, 'files', 'plots', 'simulations')
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
        if self.options.env['target_centered']:
            direction_angle = self.options.missile['initial_state'][2]['psi']
        else:
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





