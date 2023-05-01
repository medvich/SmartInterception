from dataclasses import dataclass
import numpy as np
import os
import json
from collections import namedtuple
from pathlib import Path
from ambiance import Atmosphere


Acceleration = namedtuple('acceleration', ['x', 'z'])

CartPoint = namedtuple('point', ['x', 'z'])
PolPoint = namedtuple('point', ['r', 'theta'])

BASE_PATH = Path(os.getcwd())
for _ in range(len(BASE_PATH.parents) + 1):
    if os.path.basename(BASE_PATH) == 'SmartInterception':
        break
    BASE_PATH = BASE_PATH.parents[0]
LOG_PATH = os.path.join(BASE_PATH, 'files', 'logs')
STUDY_PATH = os.path.join(BASE_PATH, 'files', 'studies')


@dataclass
class Options:
    missile = {}
    target = {}
    los = {}
    env = {}

    def __init__(self):
        self.sOs = None
        self.g = None

    def __str__(self):
        return json.dumps(
            {
                'missile': self.missile,
                'target': self.target,
                'los': self.los,
                'env': self.env
            }, indent=2,
        )

    def set_bounds(self, bounds):
        self.missile['bounds'] = bounds['missile']
        self.target['bounds'] = bounds['target']
        self.los['bounds'] = bounds['los']
        self.env['bounds'] = bounds['environment']

        self.env['bounds']['escape_sector_angle'] *= np.pi / 180
        self.missile['bounds']['coordinator_angle_max'] *= np.pi / 180
        self.missile['bounds']['beta_max'] *= np.pi / 180
        self.missile['bounds']['beta_step_max'] *= np.pi / 180
        self.target['bounds']['coordinator_angle_max'] *= np.pi / 180

    def set_states(self, values):
        target_state = {}
        missile_state = {}
        los_state = {}

        q = np.radians(values['environment']['initial_heading_angle'] % 360)
        eps = np.radians(values['environment']['initial_heading_error'] % 360)
        psi = np.radians(values['environment']['initial_psi'] % 360)

        if values['environment']['target_centered']:

            target_state['x'] = 0
            target_state['z'] = 0
            target_state['psi'] = psi % (2 * np.pi)

            missile_state['x'] = values['environment']['initial_distance'] * np.cos(psi + q - np.pi)
            missile_state['z'] = values['environment']['initial_distance'] * np.sin(psi + q - np.pi)
            missile_state['psi'] = (psi + q - eps) % (2 * np.pi)

        else:

            missile_state['x'] = 0
            missile_state['z'] = 0
            missile_state['psi'] = (psi - eps) % (2 * np.pi)

            target_state['x'] = values['environment']['initial_distance'] * np.cos(psi)
            target_state['z'] = values['environment']['initial_distance'] * np.sin(psi)
            target_state['psi'] = (psi - q) % (2 * np.pi)

        target_state['vel'] = values['target']['vel']
        missile_state['vel'] = values['missile']['vel']

        self.missile['initial_state'] = \
            np.radians(values['missile']['alpha']), \
            np.radians(values['missile']['beta']), \
            missile_state

        self.target['initial_state'] = \
            Acceleration(
                values['target']['acceleration_x'],
                values['target']['acceleration_z']
            ), \
            target_state

        los_state['r'] = values['environment']['initial_distance']
        los_state['chi'] = psi + q
        self.los['initial_state'] = los_state

        self.missile['altitude'] = values['environment']['altitude']
        self.target['altitude'] = values['environment']['altitude']

        self.env['altitude'] = values['environment']['altitude']
        self.env['initial_distance'] = values['environment']['initial_distance']
        self.env['initial_heading_error'] = values['environment']['initial_heading_error'] * np.pi / 180
        self.env['initial_heading_angle'] = values['environment']['initial_heading_angle'] * np.pi / 180
        self.env['target_centered'] = values['environment']['target_centered']
        self.env['initial_psi'] = values['environment']['initial_psi'] * np.pi / 180

        self.sOs = float(Atmosphere(self.env['altitude']).speed_of_sound)
        self.g = float(Atmosphere(self.env['altitude']).grav_accel)

        del target_state
        del missile_state
        del los_state


