import numpy as np
from ambiance import Atmosphere


class Target2D:
    def __init__(self, bounds: dict):
        self._acceleration, self._state = None, None
        self.bounds = bounds
        self.t = None
        self._overload = None
        self.status = 'Initialized'
        self._altitude = None

    def __str__(self):
        return f"{self.status}. Current state: {self._state}"

    def get_state(self):
        return self.t, self._state

    def set_state(self, state, **kw):
        self._state = state
        if 't' in kw:
            self.t = kw['t']

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, altitude: float) -> None:
        self._altitude = altitude

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
    def acceleration(self):
        return self._acceleration

    def reset(self, state):
        self.t = 0
        self._acceleration = state[0]
        self._state = np.array(
            [
                state[1]['x'],
                state[1]['z'],
                state[1]['vel'],
                state[1]['psi']
            ], dtype=np.float32
        )

        self._overload = self._acceleration[1] / self.grav_accel
        self.status = 'Alive'
        return self._state

    def step(self, acceleration):
        assert ('x' and 'z') in acceleration._fields, 'Acceleration must be a namedtuple with x and z fields'
        assert self._state is not None, 'Call reset before using this method.'
        self._acceleration = acceleration
        x, z, vel, psi = self._state
        self._overload = acceleration.z / self.grav_accel
        return np.array([
            vel * np.cos(psi),
            vel * np.sin(psi),
            acceleration.x,
            acceleration.z / vel
        ], copy=False, dtype=np.float32)

    def terminal(self):
        x, z, vel, psi = self._state
        mach = vel / self.speed_of_sound
        if not (min(self.bounds['mach_range']) < mach < max(self.bounds['mach_range'])):
            self.status = 'Out of Ma bounds'
            return True, f"{self.status}. Ma = {mach:.2f}"
        return False, None
