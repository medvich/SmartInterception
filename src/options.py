from dataclasses import dataclass, InitVar
from typing import Union
import numpy as np
import yaml
import os

CURRENT_PATH = os.getcwd()


@dataclass
class Options:
    values: InitVar[Union[dict, str]]

    def __post_init__(self, values):
        """
        Файл values лежит в той же директории
        """

        if type(values) is str:
            with open(os.path.join(CURRENT_PATH, values)) as f:
                values = yaml.safe_load(f)

        self.missile = values['missile']
        self.target = values['target']
        self.los = {'initial_state': {}, 'bounds': {}}
        self.env = values['environment']

        self.make()

    def __str__(self):
        return str(
            {
                'missile': self.missile,
                'target': self.target,
                'los': self.los,
                'env': self.env
            }
        )

    def make(self):

        q = np.radians(self.env['initial_heading_angle'] % 360)
        eps = np.radians(self.env['initial_heading_error'] % 360)
        psi = np.radians(self.env['initial_psi'] % 360)

        if self.env['target_centered']:
            self.target['initial_state']['x'] = 0
            self.target['initial_state']['z'] = 0
            self.target['initial_state']['psi'] = psi
            self.missile['initial_state']['x'] = self.env['initial_distance'] * np.cos(psi)
            self.missile['initial_state']['z'] = self.env['initial_distance'] * np.sin(psi)
            self.missile['initial_state']['psi'] = psi + q + np.copysign(eps, np.pi / 2 - q)
        else:
            self.missile['initial_state']['x'] = 0
            self.missile['initial_state']['z'] = 0
            self.missile['initial_state']['psi'] = psi
            self.target['initial_state']['x'] = self.env['initial_distance'] * np.cos(psi)
            self.target['initial_state']['z'] = self.env['initial_distance'] * np.sin(psi)
            self.target['initial_state']['psi'] = psi + q + np.copysign(eps, np.pi / 2 - q)

        self.missile['initial_state'] = \
            np.radians(self.missile['initial_state'].pop('alpha')), \
            np.radians(self.missile['initial_state'].pop('beta')), \
            self.missile['initial_state']

        self.missile['altitude'] = self.env['altitude']
        self.target['altitude'] = self.env['altitude']

        self.los['initial_state']['r'] = self.env['initial_distance']
        self.los['initial_state']['chi'] = psi + q

        self.los['bounds']['explosion_distance'] = self.missile['bounds']['explosion_distance']

        self.env['initial_heading_error'] *= np.pi / 180
        self.env['initial_heading_angle'] *= np.pi / 180
        self.missile['bounds']['coordinator_angle_max'] *= np.pi / 180
        self.missile['bounds']['beta_max'] *= np.pi / 180
        self.target['bounds']['coordinator_angle_max'] *= np.pi / 180


