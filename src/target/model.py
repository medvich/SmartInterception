import numpy as np
from ambiance import Atmosphere


class Target2D:
    def __init__(self, options: dict):
        self.sOs = float(Atmosphere(options['altitude']).speed_of_sound)
        self.g = float(Atmosphere(options['altitude']).grav_accel)
        self._initial_state = options['initial_state']
        self.bounds = options['bounds']
        self._state, self.t = None, None
        self._overload = None
        self.status = 'Initialized'

    def __str__(self):
        return f"{self.status}. Current state: {self._state}"

    def get_state(self):
        return self.t, self._state

    def set_state(self, state, **kw):
        self._state = state
        if 't' in kw:
            self.t = kw['t']

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
        self._overload = 0
        self.status = 'Alive'
        return self._state

    def step(self, acceleration):
        assert ('x' and 'z') in acceleration._fields, 'Acceleration must be a namedtuple with x and z fields'
        assert self._state is not None, 'Call reset before using this method.'
        x, z, vel, psi = self._state
        self._overload = acceleration.z / self.g
        return np.array([
            vel * np.cos(psi),
            vel * np.sin(psi),
            acceleration.x,
            acceleration.z / vel
        ], copy=False, dtype=np.float32)

    def terminal(self):
        x, z, vel, psi = self._state
        mach = vel / self.sOs
        if not (min(self.bounds['mach_range']) < mach < max(self.bounds['mach_range'])):
            self.status = 'Out of Ma bounds'
            return True, f"{self.status}. Ma = {mach:.2f}"
        return False, None

    def overload(self):
        return self._overload
