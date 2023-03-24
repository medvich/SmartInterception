from dataclasses import dataclass, InitVar


@dataclass
class Energetics:
    options: InitVar[dict]

    def __post_init__(self, options):
        self.omega0 = options['omega0']
        self.mass0 = options['mass0']
        self.t_act = options['t_act']
        self.specific_impulse = options['specific_impulse']
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
