import numpy as np
from ..missile import Missile2D
from ..target import Target2D
from ..options import Options, Acceleration
from gym import Env
from ambiance import Atmosphere
from gym.spaces import Box
from easyvec import Vec2
import pandas as pd
from ..visualization import PlotlyRenderer
from ..methods import PN
from typing import Union


class Interception2D(Env):

    def __init__(self, agent: Union[str, None], **kw):

        # Устанавливается, что обучаемый агент может наблюдать следующие параметры своего окружения:
        #   - Дальность до цели
        #   - Нормальная перегрузка цели
        #   - Собственная нормальная перегрузка
        #   - Модуль скорости цели в Махах
        #   - Модуль собственной скорости в Махах
        #   - Модуль относительной скорости ракеты
        #   - Угловая ошибка координатора
        #   - Угол ракурса
        #   - Собственная масса
        #   - Высота полета

        if 'values' in kw:
            assert 'scenarios' not in kw
            values = kw['values']
        else:
            assert 'scenarios' in kw
            assert isinstance(kw['scenarios'], list)
            self._scenarios = kw['scenarios']
            values = self._scenarios.pop()

        if agent not in ('missile', 'target', 'both', None):
            raise ValueError('Invalid agent')
        self.agent = agent

        self.options = Options(values)

        self.sOs = float(Atmosphere(self.options.env['altitude']).speed_of_sound)
        self.g = float(Atmosphere(self.options.env['altitude']).grav_accel)

        high = np.array(
            [
                max(self.options.env['bounds']['initial_distance_range']) + 1e3,
                self.options.target['bounds']['overload_max'],
                self.options.missile['bounds']['overload_max'],
                max(self.options.target['bounds']['mach_range']),
                max(self.options.missile['bounds']['mach_range']),
                (max(self.options.missile['bounds']['mach_range']) + max(self.options.target['bounds']['mach_range'])) * self.sOs,
                self.options.missile['bounds']['coordinator_angle_max'],
                2 * np.pi,
                self.options.missile['energetics']['mass0'],
                max(self.options.env['bounds']['altitude_range'])
            ],
            dtype=np.float32
        )
        low = np.array(
            [
                0,
                -self.options.target['bounds']['overload_max'],
                -self.options.missile['bounds']['overload_max'],
                min(self.options.target['bounds']['mach_range']),
                min(self.options.missile['bounds']['mach_range']),
                self.options.missile['bounds']['relative_velocity_min'],
                -self.options.missile['bounds']['coordinator_angle_max'],
                -2 * np.pi,
                self.options.missile['energetics']['mass0'] - self.options.missile['energetics']['omega0'],
                min(self.options.env['bounds']['altitude_range'])
            ],
            dtype=np.float32
        )

        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.missile = Missile2D(self.options.missile)
        self.target = Target2D(self.options.target)
        self.los = LineOfSight2D(self.options.los)
        self._initial_state = np.array(
            [
                self.options.env['initial_distance'],
                0,
                0,
                self.options.target['initial_state'][1]['vel'] / self.sOs,
                self.options.missile['initial_state'][2]['vel'] / self.sOs,
                0,
                self.options.env['initial_heading_error'],
                self.options.env['initial_heading_angle'],
                self.options.missile['energetics']['mass0'],
                self.options.env['altitude']
            ],
            dtype=np.float32
        )

        self.action_space = Box(low=-1, high=1, shape=(1, ), dtype=np.float32)

        self.state = None
        self.t = None
        self.log = None
        self.status = 'Initialized'
        self._info = None
        self.buffer = None
        self._locked_on = None
        self._keys = None
        self.pn = None
        self._missile_action, self._target_action = None, None

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        return self.observation_space.low + state * (self.observation_space.high - self.observation_space.low)

    def reset(self):
        if self._scenarios and self.status != 'Initialized':
            values = np.random.choice(self._scenarios)
            self.options = Options(values)
            self.missile = Missile2D(self.options.missile)
            self.target = Target2D(self.options.target)
            self.los = LineOfSight2D(self.options.los)
            self._initial_state = np.array(
                [
                    self.options.env['initial_distance'],
                    0,
                    0,
                    self.options.target['initial_state'][1]['vel'] / self.sOs,
                    self.options.missile['initial_state'][2]['vel'] / self.sOs,
                    0,
                    self.options.env['initial_heading_error'],
                    self.options.env['initial_heading_angle'],
                    self.options.missile['energetics']['mass0'],
                    self.options.env['altitude']
                ],
                dtype=np.float32
            )

        self.t = 0
        missile_state = self.missile.reset()
        target_state = self.target.reset()
        self.los.set_state(missile=missile_state, target=target_state)
        los_state = self.los.get_state()
        self.state = self._initial_state
        self.state[1] = self.target.overload
        self.state[2] = self.missile.overload
        self.state[5] = self.velR(missile_state, target_state, los_state)

        self.pn = PN(0, self.options.env['altitude'])

        self._keys = np.array(
            ['t', 'distance', 'target_overload', 'missile_overload', 'target_Ma', 'missile_Ma', 'velR', 'eps', 'q',
             'mass', 'altitude', 'missile_x', 'missile_z', 'missile_psi', 'missile_beta', 'target_x', 'target_z',
             'target_psi', 'chi', 'target_acceleration_z', 'sOs'],
            dtype=np.object
        )

        log0 = np.concatenate([
            np.array([self.t], dtype=np.float32),
            self.state,
            np.delete(missile_state, 2),
            np.array([self.missile.beta], dtype=np.float32),
            np.delete(target_state, 2),
            np.delete(los_state, 0),
            np.array([self.target._acceleration.z, self.sOs], dtype=np.float32)
        ], dtype=np.float32)
        self.log = np.array([log0], dtype=np.float32)

        self.status = 'Dropped'
        self.buffer = {}
        return self.state

    @property
    def missile_action(self):
        return self._missile_action

    @missile_action.setter
    def missile_action(self, action: float) -> None:
        self._missile_action = action

    @property
    def target_action(self):
        return self._target_action

    @target_action.setter
    def target_action(self, action: float) -> None:
        self._target_action = action

    def step(self, action: np.ndarray):
        assert self.state is not None, 'Call reset before using this method.'

        if self.agent == 'target':
            beta = self.pn.get_action(self)
            aZ = action * self.options.target['bounds']['overload_max'] * self.g
        elif self.agent == 'missile':
            assert self._target_action is not None, 'Set target action before it'
            beta = action * self.options.missile['bounds']['beta_max']
            aZ = self._target_action
        elif self.agent == 'both':
            assert self._missile_action is not None, 'Set missile action before it'
            assert self._target_action is not None, 'Set target action before it'
            beta = self._missile_action
            aZ = self._target_action
        else:
            assert self._target_action is not None, 'Set target action before it'
            beta = self.pn.get_action(self)
            aZ = self._target_action

        tau = self.options.env['tau']

        _, _, missile_state = self.missile.get_state()
        _, target_state = self.target.get_state()
        los_state = self.los.get_state()

        self.buffer['s'] = self.state

        missile_ds = self.missile.step(beta)
        target_ds = self.target.step(Acceleration(0, aZ))
        los_ds = self.los.step(missile_state, target_state)

        missile_state = missile_state + tau * missile_ds
        target_state = target_state + tau * target_ds
        los_state = los_state - tau * los_ds

        self.missile.set_state(missile_state, t=self.t)
        self.target.set_state(target_state, t=self.t)
        self.los.set_state(los=los_state)

        self.t += tau

        self.state = np.array(
            [
                los_state[0],
                self.target.overload,
                self.missile.overload,
                target_state[2] / self.sOs,
                missile_state[2] / self.sOs,
                self.velR(missile_state, target_state, los_state),
                self.get_eta(missile_state, los_state),
                self.get_eta(target_state, los_state),
                self.missile.energetics.mass(self.t),
                self.options.env['altitude']
            ],
            dtype=np.float32
        )

        self.buffer['s_'] = self.state

        appendix = np.concatenate([
            np.array([self.t], dtype=np.float32),
            self.state,
            np.delete(missile_state, 2),
            np.array([self.missile.beta], dtype=np.float32),
            np.delete(target_state, 2),
            np.delete(los_state, 0),
            np.array([self.target._acceleration.z, self.sOs], dtype=np.float32)
        ], dtype=np.float32)
        self.log = np.append(self.log, [appendix], axis=0)

        terminal, self._info = self._terminal()

        if self.status != 'Alive':
            self.status = 'Alive'

        return self.state, self.reward, terminal, {}

    def _terminal(self):
        _, _, _, _, _, velR, eta_m, eta_t, _, _ = self.state
        _, target_state = self.target.get_state()
        missile_terminal, missile_info = self.missile.terminal()
        target_terminal, target_info = self.target.terminal()
        los_terminal, los_info = self.los.terminal()

        if los_terminal:
            if abs(velR) < self.options.missile['bounds']['relative_velocity_min'] \
                    and self.missile.energetics.thrust(self.t) == 0:
                self.missile.status = 'Low relative velocity'
                self.status = f"MISSILE: {self.missile.status}. Vel_ = {velR:.2f} m/sec"
                return True, self.status

            self.status = "LOS: " + los_info
            return los_terminal, self.status

        if target_terminal:
            self.status = "TARGET: " + target_info
            return target_terminal, self.status

        escaped, d_, zA_ = self._escaped(target_state)

        if escaped:
            self.target.status = 'Escaped'
            self.status = f"TARGET: {self.target.status}. D_ = {d_:.2f} m, Zone_Angle = {np.rad2deg(zA_):.2f} grad"
            return True, self.status

        if missile_terminal:
            self.status = "MISSILE: " + missile_info
            return missile_terminal, self.status

        if self._locked_on:

            if abs(eta_m) > self.options.missile['bounds']['coordinator_angle_max']:
                self.missile.status = 'Target  lost'
                self.status = f"MISSILE: {self.missile.status}. Eps = {abs(np.rad2deg(eta_m)):.2f} grad"
                return True, self.status

        if self.t > self.options.env['bounds']['termination_time']:
            self.status = f"ENV: Maximum flight time exceeded. t = {self.t:.2f} sec"
            return True, self.status

        else:
            return False, None

    def velR(self, combatant1_state, combatant2_state, los_state):
        """
        Модуль относитльной скорости участника 1 (Combatant1) относительно участнка 2 (Combatant2)
        """
        _, _, vel1, _ = combatant1_state
        _, _, vel2, _ = combatant2_state
        eta1 = self.get_eta(combatant1_state, los_state)
        eta2 = self.get_eta(combatant2_state, los_state)
        return vel1 * np.cos(eta1) - vel2 * np.cos(eta2)

    @staticmethod
    def get_eta(combatant_state, los_state):
        """
        Угол между вектором скорости участника сражения (ракета / цель) и линией визирования "ракета-цель"
        """
        _, _, vel, psi = combatant_state
        r, chi = los_state
        los_coordinates = np.array([
            r * np.cos(chi),
            r * np.sin(chi)], dtype=np.float32
        )
        vel_coordinates = np.array([
            vel * np.cos(psi),
            vel * np.sin(psi)], dtype=np.float32
        )
        r_vec = Vec2.from_list(los_coordinates)
        vel_vec = Vec2.from_list(vel_coordinates)
        return r_vec.angle_to(vel_vec)

    def _escaped(self, target_state):
        x, z, _, _ = target_state
        points = np.array([
            x - self.options.missile['initial_state'][2]['x'],
            z - self.options.missile['initial_state'][2]['z']],
            dtype=np.float32
        )
        """
        Вектор, длина которого равна расстоянию от точки пуска ракеты до текущего положения цели
        """
        vecD = Vec2.from_list(points)
        """
        Вектор, совпадающий с продольной осью ракеты в момент ее пуска
        """
        i = Vec2(1, 0)
        vecX = i.rotate(self.options.missile['initial_state'][2]['psi'])

        d_ = vecD.len()
        zA_ = abs(vecD.angle_to(vecX))

        if d_ > self.options.env['bounds']['escape_distance'] \
                and zA_ < self.options.env['bounds']['escape_sector_angle'] / 2:
            return True, d_, zA_
        return False, d_, zA_

    @property
    def locked_on(self):
        return self._locked_on

    @locked_on.setter
    def locked_on(self, value: bool) -> None:
        self._locked_on = value

    @property
    def info(self):
        return self._info

    @property
    def reward(self):
        if self.agent in ('missile', 'both'):
            return self.missile_reward
        if self.agent == 'target':
            return self.target_reward
        return None

    @property
    def missile_reward(self):
        if 'hit' in self.status.lower():
            return 10
        if 'target' in self.status.lower():
            return 5
        if 'missile' in self.status.lower():
            return -10
        return 1 - self.buffer['s_'][0] / self.buffer['s'][0]

    @property
    def target_reward(self):
        # k1 = self.t / 100
        # k2 = abs(np.rad2deg(self.state[6]))
        # k3 = self.state[0] / self.options.env['initial_distance']
        # if 'escaped' in self.status.lower():
        #     return 15
        if 'missile' in self.status.lower():
            return 20
        if 'hit' in self.status.lower():
            return -50
        if 'target' in self.status.lower():
            return -6
        return -1 * (self.buffer['s'][0] / self.buffer['s_'][0] + self.buffer['s'][1] / self.buffer['s_'][1])

    def render(self, mode="human"):
        pass

    def post_render(self, tab=1, renderer='notebook'):
        data = {}
        for i, k in enumerate(self._keys):
            data[k] = self.log[:, i][::tab]
        df = pd.DataFrame(data=data)
        pr = PlotlyRenderer(df, self.options)

        d = self.options.env['initial_distance']
        q = int(np.rad2deg(self.options.env['initial_heading_angle']))
        eps = int(np.rad2deg(self.options.env['initial_heading_error']))

        filename_const = f"{d}-{q}-{eps}"
        pr.plot(renderer, filename_const)


class LineOfSight2D:
    def __init__(self, options: dict):
        self.bounds = options['bounds']
        self._state = None
        self.status = 'Initialized'

    def get_state(self):
        return self._state

    def set_state(self, **kw):
        if 'los' in kw:
            self._state = kw['los']
        else:
            assert ('missile' and 'target') in kw
            missile_state, target_state = kw['missile'], kw['target']
            los_coordinates = np.array([
                target_state[0] - missile_state[0],
                target_state[1] - missile_state[1]], dtype=np.float32
            )
            r = Vec2.from_list(los_coordinates)
            i = Vec2(1, 0)
            chi = i.angle_to(r)
            self._state = np.array([r.len(), chi], dtype=np.float32)
        self.status = 'Alive'

    def step(self, missile_state, target_state):
        r, chi = self._state
        _, _, missile_vel, missile_psi = missile_state
        _, _, target_vel, target_psi = target_state
        coefficient = np.array([1, r])
        target_t = np.array([np.cos(target_psi - chi),
                             np.sin(target_psi - chi)])
        target_n = np.array([0,
                             0])
        missile_t = np.array([np.cos(missile_psi - chi),
                              np.sin(missile_psi - chi)])
        missile_n = np.array([0,
                              0])
        return -1 / coefficient * (target_vel * (target_t + target_n) - missile_vel * (missile_t + missile_n))

    def terminal(self):
        r, chi = self._state
        if r < self.bounds['explosion_distance']:
            self.status = 'Hit'
            return True, f"{self.status}. D = {r:.2f} m"
        return False, None