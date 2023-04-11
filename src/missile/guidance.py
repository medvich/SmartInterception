import numpy as np
from ..options import CartPoint
from ..common import n_degree_curve


def k_foo_alternative(*args):
    vel, altitude, = args
    altitude_bounds = (0, 70e3)
    n_bounds = (0.05, 45)
    vel_bounds = (200, 1800)
    k_bounds = (2, 80)
    gamma = 8
    n = n_degree_curve(altitude, altitude_bounds, n_bounds, gamma, reverse=True)
    k = n_degree_curve(vel, vel_bounds, k_bounds, n)
    return k


class PN:

    UPD_FACTOR = 0  # время обновления точки прицеливания, с

    def __init__(self, env: object, k_foo=k_foo_alternative):
        self.env = env
        self.k_foo = k_foo
        self.params = {'multiplier': 0}
        self.log = None

        self.t0, self.altitude = None, None

    def reset(self, t0, altitude):
        self.altitude = altitude
        self.t0 = t0

        log0 = np.array([self.t0, 0])
        self.log = np.array([log0], dtype=np.object)

    def get_action(self):

        los_state = self.env.los.get_state()
        _, _, missile_state = self.env.missile.get_state()
        t, target_state = self.env.target.get_state()
        r, _ = los_state
        x, z, vel, _ = missile_state
        eps = self.env.get_eta(missile_state, los_state)

        if not self.env.locked_on \
                and r <= self.env.options.missile['bounds']['lock_on_distance'] \
                and eps < self.env.options.missile['bounds']['coordinator_angle_max']:
            self.env.locked_on = True
            self.params['locked_on_point'] = CartPoint(x, z)

        if t >= self.t0 + self.UPD_FACTOR * self.params['multiplier']:
            d_r, d_chi = self.env.los.step(missile_state, target_state)

            K = self.k_foo(vel, self.altitude)
            d_psi = K * d_chi
            self.params['action'] = self.env.missile.get_required_beta(d_psi)
            self.params['multiplier'] += 1
            self.params['K'] = K

        appendix = np.array([t, self.params['K']])
        self.log = np.append(self.log, [appendix], axis=0)

        return self.params['action']
