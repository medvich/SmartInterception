from dataclasses import dataclass


@dataclass
class Energetics:
    mass0: float = 165.
    omega0: float = 65.
    specific_impulse: float = 1900.
    t_act: float = 6.

    def __post_init__(self):
        self.rate = self.omega0 / self.t_act

    def mass(self, t):
        if t <= self.t_act:
            return self.mass0 - self.rate * t
        return self.mass0 - self.rate * self.t_act

    def omega(self, t):
        if t <= self.t_act:
            return self.omega0 - self.rate * t
        return 0

    def thrust(self, t):
        if t <= self.t_act:
            return self.omega0 * self.specific_impulse / self.t_act
        return 0
