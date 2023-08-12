from .options import BASE_PATH
import os
import yaml
import numpy as np
from typing import Union
from .options import Options
import matplotlib.pyplot as plt
import pandas as pd
from .common import n_degree_curve


VALUES_FILENAME = 'values.yaml'
DISTANCE_BOUNDS = (0, 80000)
Q_BOUNDS = (0, 180)
EPS_BOUNDS = (-40, 40)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def make_escape_scenarios(
        d_min: float = min(DISTANCE_BOUNDS),
        d_max: float = max(DISTANCE_BOUNDS),
        q_min: float = min(Q_BOUNDS),
        q_max: float = max(Q_BOUNDS),
        eps_min: float = min(EPS_BOUNDS),
        eps_max: float = max(EPS_BOUNDS),
        initial_psi: float = 0,
        n: int = 100,
        seed: Union[int, None] = None,
        target_centered: bool = False

):
    """

    Угловые величины назначаются в градусах!
    """

    assert d_min <= d_max and \
           all([d_min, d_max]) in range(min(DISTANCE_BOUNDS), max(DISTANCE_BOUNDS)), 'Wrong distance bounds'
    assert q_min <= q_max and \
           all([q_min, q_max]) in range(min(Q_BOUNDS), max(Q_BOUNDS)), 'Wrong q bounds'
    assert eps_min <= eps_max and \
           all([eps_min, eps_max]) in range(min(EPS_BOUNDS), max(EPS_BOUNDS)), 'Wrong eps bounds'

    scenarios = []

    np.random.seed(seed)

    while len(scenarios) < n:
        with open(os.path.join(BASE_PATH, 'src', VALUES_FILENAME)) as f:
            values = yaml.safe_load(f)
        q = np.random.uniform(q_min, q_max) if q_min != q_max else q_min
        d = np.random.uniform(d_min, d_max) if d_min != d_max else d_min
        eps = np.random.uniform(eps_min, eps_max) if eps_min != eps_max else eps_min

        values['environment']['initial_distance'] = d
        values['environment']['initial_heading_error'] = eps
        values['environment']['initial_heading_angle'] = q
        values['environment']['target_centered'] = target_centered
        values['environment']['initial_psi'] = initial_psi

        scenarios.append(values)

    bounds = {
        'distance': (d_min, d_max),
        'q': (q_min, q_max),
        'eps': (eps_min, eps_max),
    }

    params = {
        'params': {
            'target_centered': target_centered,
            'initial_psi': initial_psi
        },
        'bounds': bounds
    }

    return scenarios, params


def plot_scenarios(scenarios, params) -> None:
    options = []
    values = np.empty((0, 12), float)

    for scenario in scenarios:
        o = Options()
        o.set_states(scenario)
        options.append(o)

        xT = o.target['initial_state'][-1]['x']
        zT = o.target['initial_state'][-1]['z']
        xM = o.missile['initial_state'][-1]['x']
        zM = o.missile['initial_state'][-1]['z']

        velT = o.target['initial_state'][-1]['vel']
        velM = o.missile['initial_state'][-1]['vel']

        psiT = o.target['initial_state'][-1]['psi']
        psiM = o.missile['initial_state'][-1]['psi']

        uT, vT = velT * np.cos(psiT), velT * np.sin(psiT)
        uM, vM = velM * np.cos(psiM), velM * np.sin(psiM)

        rhoT, phiT = cart2pol(xT, zT)
        rhoM, phiM = cart2pol(xM, zM)

        values = np.append(
            values,
            [
                np.array(
                    [
                        rhoT,
                        phiT,
                        rhoM,
                        phiM,
                        uT,
                        vT,
                        uM,
                        vM,
                        velT,
                        velM,
                        psiT,
                        psiM
                    ]
                )
            ],
            axis=0
        )

    df = pd.DataFrame(
        values,
        columns=['rhoT', 'phiT', 'rhoM', 'phiM', 'uT', 'vT', 'uM', 'vM', 'velT', 'velM', 'psiT', 'psiM']
    )

    fig = plt.figure(dpi=160)
    ax = fig.add_subplot(projection='polar')

    if params['params']['target_centered']:
        QM = ax.quiver(
            df.phiM,
            df.rhoM,
            df.uM,
            df.vM,
            color='r',
            label='missile position',
            zorder=3
        )
        QT = ax.quiver(
            np.repeat(0, len(scenarios)),
            np.repeat(0, len(scenarios)),
            df.uT,
            df.vT,
            color='b',
            label='target position',
            zorder=3
        )
        ax.fill_between(
            np.linspace(
                -np.pi + np.radians(min(params['bounds']['q']) + params['params']['initial_psi']),
                -np.pi + np.radians(max(params['bounds']['q']) + params['params']['initial_psi']),
                100),
            min(params['bounds']['distance']),
            max(params['bounds']['distance']),
            alpha=0.2,
            color='red',
            zorder=3
        )

    else:
        QT = ax.quiver(
            df.phiT,
            df.rhoT,
            df.uT,
            df.vT,
            color='b',
            label='target position',
            zorder=3
        )
        QM = ax.quiver(
            np.repeat(0, len(scenarios)),
            np.repeat(0, len(scenarios)),
            df.uM,
            df.vM,
            color='r',
            label='missile position',
            zorder=3
        )

    plt.quiverkey(
        QT,
        0.45,
        0.85,
        df.iloc[0].velT,
        label='target position',
        labelpos='E',
        labelcolor='b',
        coordinates='figure'
    )
    plt.quiverkey(
        QM,
        0.45,
        0.80,
        df.iloc[0].velM,
        label='missile position',
        labelpos='E',
        labelcolor='r',
        coordinates='figure'
    )

    ax.set_rticks([10000, 35000, 60000, 85000])
    ax.set_ylim([0, 90000])
    ax.set_thetamin(20)
    ax.set_thetamax(-200)

    plt.grid(c='lightgray')
    plt.show()


def make_scenario_batches(
        zones,
        target_centered: bool = False,
        plot: bool = False,
        seed: int = 1001
):
    scenario_batches = []
    for zone in zones:
        eps_max = n_degree_curve(zone['d_max'], (0, 80000), (0, 40), 1)
        scenarios, params = make_escape_scenarios(
            n=zone['n'],
            seed=seed,
            target_centered=target_centered,
            d_min=zone['d_min'],
            d_max=zone['d_max'],
            q_min=zone['q_min'],
            q_max=zone['q_max'],
            eps_min=-eps_max,
            eps_max=eps_max
        )
        if plot:
            plot_scenarios(scenarios, params)
        scenario_batches.append(scenarios)
    return scenario_batches


train_zones = [
    {'d_min': 10000,
     'd_max': 50000,
     'q_min': 0,
     'q_max': 180,
     'n': 500
     }
]
