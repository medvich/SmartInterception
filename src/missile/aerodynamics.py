from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from ..options import BASE_PATH
from collections import namedtuple


Wing = namedtuple('wing', ['area'])


@dataclass
class Aerodynamics:
    cx0_filename: str = 'cx0.csv'
    cyA_filename: str = 'cyA.csv'
    wing = Wing(area=0.372)

    def __post_init__(self):
        self._cx0_points = pd.read_csv(os.path.join(BASE_PATH, 'src', 'missile', 'files', self.cx0_filename), sep=';')
        self._cyA_points = pd.read_csv(os.path.join(BASE_PATH, 'src', 'missile', 'files', self.cyA_filename), sep=';')

    def cx0(self, mach):
        return np.interp(mach, self._cx0_points.mach, self._cx0_points.value)

    def cyA(self, mach):
        return np.interp(mach, self._cyA_points.mach, self._cyA_points.value)

    def cx(self, mach, alpha):
        return self.cx0(mach) + abs(self.cy(mach, alpha) * np.tan(alpha))

    def cy(self, mach, alpha):
        return self.cyA(mach) * np.rad2deg(alpha)

    def force_x(self, q, mach, alpha):
        return self.cx(mach, alpha) * q * self.wing.area * 0.25

    def force_y(self, q, mach, alpha):
        return self.cy(mach, alpha) * q * self.wing.area * 1.5

