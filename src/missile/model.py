import numpy as np
from .aerodynamics import Aerodynamics
from .energetics import Energetics
from ambiance import Atmosphere


class Missile2D:
    def __init__(self, bounds: dict):
        self._alpha, self._beta, self.state = None, None, None
        self.aerodynamics = Aerodynamics()
        self.energetics = Energetics()
        self.bounds = bounds
        self.t = None
        self._overload = None
        self.status = 'Initialized'
        self._altitude = None

    def __str__(self):
        return f"{self.status}. Current state: {self.state}"

    def get_state(self):
        return self.t, self._beta, self.state

    def set_state(self, state, **kw):
        self.state = state
        if 't' in kw:
            self.t = kw['t']
        if 'beta' in kw:
            self._beta = kw['beta']

    def reset(self, state):
        self.t = 0
        self._alpha = state[0]
        self._beta = state[1]
        self.state = np.array(
            [
                state[2]['x'],
                state[2]['z'],
                state[2]['vel'],
                state[2]['psi']
            ], dtype=np.float32
        )
        q = self.density * state[2]['vel'] ** 2 / 2
        mach = state[2]['vel'] * self.speed_of_sound
        self._overload = self.aerodynamics.force_y(q, mach, self._beta) / self.energetics.mass(self.t) / self.grav_accel
        self.status = 'Alive'
        return self.state

    def step(self, beta):
        assert self.state is not None, 'Call reset before using this method.'
        x, z, vel, psi = self.state
        thrust, mass = self.energetics.thrust(self.t), self.energetics.mass(self.t)
        mach = vel / self.speed_of_sound
        q = self.density * vel ** 2 / 2
        beta = np.copysign(min(abs(beta), self.bounds['beta_max']), beta)
        force_x = self.aerodynamics.force_x(q, mach, beta)
        force_z = self.aerodynamics.force_y(q, mach, beta)
        self._overload = force_z / mass / self.grav_accel
        k = self._overload / self.bounds['overload_max']
        if k > 1:
            beta *= k
        self._beta = beta
        return np.array([
            vel * np.cos(psi),
            vel * np.sin(psi),
            (thrust * np.cos(self._alpha) * np.cos(beta) - force_x) / mass,
            (thrust * np.cos(self._alpha) * np.sin(beta) - force_z) / mass / vel
        ], copy=False, dtype=np.float32)

    def terminated(self):
        x, z, vel, psi = self.state
        mach = vel / self.speed_of_sound
        if not (min(self.bounds['mach_range']) < mach < max(self.bounds['mach_range'])):
            self.status = 'Out of Ma bounds'
            return True, f"{self.status}. Ma = {mach:.2f}"
        return False, None

    def get_required_beta(self, d_psi):
        x, z, vel, psi = self.state
        area = self.aerodynamics.wing.area
        mach = vel / self.speed_of_sound
        thrust, mass = self.energetics.thrust(self.t), self.energetics.mass(self.t)

        return np.radians(
            mass * vel * d_psi
            / (-thrust / 57.3 * np.cos(self._alpha) + self.aerodynamics.cyA(mach) * self.density * vel ** 2 / 2 * area)
        )

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, altitude: float) -> None:
        self._altitude = altitude

    @property
    def density(self):
        assert self._altitude is not None
        return float(Atmosphere(self._altitude).density)

    @property
    def speed_of_sound(self):
        assert self._altitude is not None
        return float(Atmosphere(self._altitude).speed_of_sound)

    @property
    def grav_accel(self):
        assert self._altitude is not None
        return float(Atmosphere(self._altitude).grav_accel)

    @property
    def overload(self):
        return self._overload

    @property
    def beta(self):
        return self._beta
