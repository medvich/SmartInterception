import numpy as np
from ambiance import Atmosphere


class Target2D:
    def __init__(self, bounds: dict):
        self.bounds = bounds
        self.status = 'Initialized'
        self.acceleration, self.state = None, None
        self.t = None
        self._overload = None
        self._altitude = None
        self.buffer = None

    def __str__(self):
        return f"{self.status}. Current state: {self.state}"

    def get_state(self):
        return self.t, self.state

    def set_state(self, state, **kw):
        self.state = state
        if 't' in kw:
            self.t = kw['t']

    def reset(self, state):
        self.t = 0
        self.acceleration = state[0]
        self.state = np.array(
            [
                state[1]['x'],
                state[1]['z'],
                state[1]['vel'],
                state[1]['psi']
            ], dtype=np.float32
        )

        self._overload = self.acceleration.z / self.grav_accel
        self.status = 'Alive'
        self.buffer = {
            'prev_acceleration': self.acceleration
        }
        return self.state

    def step(self, acceleration):
        assert ('x' and 'z') in acceleration._fields, 'Acceleration must be a namedtuple with x and z fields'
        assert self.state is not None, 'Call reset before using this method.'

        prev_acceleration = self.buffer['prev_acceleration']
        diff = acceleration.z - prev_acceleration.z
        self.acceleration = acceleration._replace(
            z=prev_acceleration.z + np.copysign(min(abs(diff), self.bounds['acceleration_z_step_max']), diff)
        )

        x, z, vel, psi = self.state

        self._overload = self.acceleration.z / self.grav_accel
        self.buffer['prev_acceleration'] = self.acceleration

        return np.array([
            vel * np.cos(psi),
            vel * np.sin(psi),
            acceleration.x,
            acceleration.z / vel
        ], copy=False, dtype=np.float32)

    def terminated(self):
        x, z, vel, psi = self.state
        mach = vel / self.speed_of_sound
        if not (min(self.bounds['mach_range']) < mach < max(self.bounds['mach_range'])):
            self.status = 'Out of Ma bounds'
            return True, f"{self.status}. Ma = {mach:.2f}"
        return False, None

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
