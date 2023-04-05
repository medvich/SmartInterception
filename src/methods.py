import numpy as np
from collections import namedtuple


Point = namedtuple('point', ['x', 'z'])


def n_degree_curve(x, x_bounds, y_bounds, n, reverse=False):
    """
    Determine n-degree function on the segment with borders
    'x_bounds' and 'y_bounds' and get it's value depending on 'x'
    """

    assert len(x_bounds) == len(y_bounds) == 2, "'x' & 'y' bounds should be the same number"
    x_min, x_max = min(x_bounds), max(x_bounds)
    y_min, y_max = min(y_bounds), max(y_bounds)
    assert x_min <= x <= x_max, "'x' out of bounds"
    if reverse is True:
        return y_min + (y_max - y_min)*(1 - (abs(x) - x_min) / (x_max - x_min)) ** n
    return y_min + (y_max - y_min)*((abs(x) - x_min) / (x_max - x_min)) ** n


def k_foo_alternative(*args):
    vel, altitude, = args
    altitude_bounds = (0, 70e3)
    n_bounds = (0.05, 45)
    vel_bounds = (250, 2000)
    k_bounds = (2, 80)
    gamma = 8
    n = n_degree_curve(altitude, altitude_bounds, n_bounds, gamma, reverse=True)
    k = n_degree_curve(vel, vel_bounds, k_bounds, n)
    return k


class PN:

    UPD_FACTOR = 0  # время обновления точки прицеливания, с

    def __init__(self, t0, altitude, k_foo=k_foo_alternative):
        self.k_foo = k_foo
        self.altitude = altitude
        self.t0 = t0
        self.params = {'multiplier': 0}
        log0 = np.array([self.t0, 0])
        self.log = np.array([log0], dtype=np.object)

    def get_action(self, env: object):

        los_state = env.los.get_state()
        _, _, missile_state = env.missile.get_state()
        t, target_state = env.target.get_state()
        r, _ = los_state
        x, z, vel, _ = missile_state
        eps = env.get_eta(missile_state, los_state)

        if not env.locked_on \
                and r <= env.options.missile['bounds']['lock_on_distance'] \
                and eps < env.options.missile['bounds']['coordinator_angle_max']:
            env.locked_on = True
            self.params['locked_on_point'] = Point(x, z)

        if t >= self.t0 + self.UPD_FACTOR * self.params['multiplier']:
            d_r, d_chi = env.los.step(missile_state, target_state)

            K = self.k_foo(vel, self.altitude)
            d_psi = K * d_chi
            self.params['action'] = env.missile.get_required_beta(d_psi)
            self.params['multiplier'] += 1
            self.params['K'] = K

        appendix = np.array([t, self.params['K']])
        self.log = np.append(self.log, [appendix], axis=0)

        return self.params['action']
