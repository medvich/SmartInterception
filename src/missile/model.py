import numpy as np
from .aerodynamics import Aerodynamics
from .energetics import Energetics
from ambiance import Atmosphere


class Missile2D:
    def __init__(self, options: dict):
        self.dens = float(Atmosphere(options['altitude']).density)
        self.sOs = float(Atmosphere(options['altitude']).speed_of_sound)
        self.g = float(Atmosphere(options['altitude']).grav_accel)
        self.alpha, self._initial_beta, self._initial_state = options['initial_state']
        self.aerodynamics = Aerodynamics(options['aerodynamics'])
        self.energetics = Energetics(options['energetics'])
        self.bounds = options['bounds']
        self._state, self.t, self.beta = None, None, None
        self._overload = None
        self.status = 'Initialized'

    def __str__(self):
        return f"{self.status}. Current state: {self._state}"

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def reset(self):
        self.t = 0
        self._state = np.array(
            [
                self._initial_state['x'],
                self._initial_state['z'],
                self._initial_state['vel'],
                self._initial_state['psi']
            ], dtype=np.float32
        )
        q = self.dens * self._initial_state['vel'] ** 2 / 2
        mach = self._initial_state['vel'] * self.sOs
        self._overload = self.aerodynamics.force_y(q, mach, self._initial_beta) / self.energetics.mass(self.t) / self.g
        self.beta = self._initial_beta
        self.status = 'Alive'
        return self.t, self._state

    def step(self, beta):
        assert self._state is not None, 'Call reset before using this method.'
        x, z, vel, psi = self._state
        thrust, mass = self.energetics.thrust(self.t), self.energetics.mass(self.t)
        mach = vel / self.sOs
        q = self.dens * vel ** 2 / 2
        beta = min(beta, self.bounds['beta_max'])
        force_x = self.aerodynamics.force_x(q, mach, beta)
        force_z = self.aerodynamics.force_y(q, mach, beta)
        self._overload = force_z / mass / self.g
        k = self._overload / self.bounds['overload_max']
        if k > 1:
            beta *= k
        return np.array([
            vel * np.cos(psi),
            vel * np.sin(psi),
            (thrust * np.cos(self.alpha) * np.cos(beta) - force_x) / mass,
            (thrust * np.cos(self.alpha) * np.sin(beta) - force_z) / mass / vel
        ], copy=False, dtype=np.float32)

    def _terminal(self):
        x, z, vel, psi = self._state
        mach = vel / self.sOs
        if not (min(self.bounds['mach_range']) < mach > max(self.bounds['mach_range'])):
            self.status = 'Out of Ma bounds'
            return True, f"{self.status}. Ma = {round(mach, 2)}"
        return False, None

    def get_required_beta(self, d_psi):
        x, z, vel, psi = self._state
        area = self.aerodynamics.wing.area
        mach = vel / self.sOs
        thrust, mass = self.energetics.thrust(self.t), self.energetics.mass(self.t)

        return np.radians(
            -mass * vel * d_psi
            / (-thrust / 57.3 * np.cos(self.alpha) + self.aerodynamics.cyA(mach) * self.dens * vel ** 2 / 2 * area)
        )

    def overload(self):
        return self._overload
