import numpy as np
from ..missile import Missile2D
from ..target import Target2D
from ..options import Options, Acceleration, BASE_PATH
from gym import Env
from gym.spaces import Box
from easyvec import Vec2
import pandas as pd
from ..visualization import PlotlyRenderer
from ..methods import PN
from typing import Union
import os
import yaml


class Interception2D(Env):

    TAU = 0.1

    def __init__(self, agent: Union[str, None], bounds: Union[str, dict], scenarios: Union[list, dict, str]):

        # Объявляем границы нашего окружения, чтобы задать observation_space

        if type(bounds) is str:
            with open(os.path.join(BASE_PATH, 'src', bounds)) as f:
                self._bounds = yaml.safe_load(f)
        else:
            self._bounds = bounds

        # Объявляем сценарий ресета, если он задан в конкретном yaml файле

        if type(scenarios) is str:
            with open(os.path.join(BASE_PATH, 'src', scenarios)) as f:
                self._scenarios = yaml.safe_load(f)
        else:
            self._scenarios = scenarios

        # Проверяем корректность выбора обучающегося агента

        if agent not in ('missile', 'target', 'both', None):
            raise ValueError('Invalid agent')
        self.agent = agent

        """
        Устанавливается, что обучаемый агент может наблюдать следующие параметры своего окружения:
          - Дальность до цели
          - Нормальная перегрузка цели
          - Нормальная перегрузка ракеты
          - Модуль скорости цели в Махах
          - Модуль скорости ракеты в Махах
          - Модуль относительной скорости ракеты
          - Угловая ошибка координатора РЛС ракеты
          - Угол ракурса ракеты
          - Масса ракеты
          - Высота полета оппонентов        
        """

        high = np.array(
            [
                self._bounds['environment']['simulation_zone_radius'],
                self._bounds['target']['overload_max'],
                self._bounds['missile']['overload_max'],
                max(self._bounds['target']['mach_range']),
                max(self._bounds['missile']['mach_range']),
                2_000,
                self._bounds['missile']['coordinator_angle_max'],
                np.pi,
                max(self._bounds['missile']['mass']),
                max(self._bounds['environment']['altitude_range'])
            ],
            dtype=np.float32
        )
        low = np.array(
            [
                0,
                -self._bounds['target']['overload_max'],
                -self._bounds['missile']['overload_max'],
                min(self._bounds['target']['mach_range']),
                min(self._bounds['missile']['mach_range']),
                self._bounds['missile']['relative_velocity_min'],
                -self._bounds['missile']['coordinator_angle_max'],
                -np.pi,
                min(self._bounds['missile']['mass']),
                min(self._bounds['environment']['altitude_range'])
            ],
            dtype=np.float32
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Инициализируем экземпляры нашего окружения

        self.missile = Missile2D(self._bounds['missile'])
        self.target = Target2D(self._bounds['target'])
        self.los = LineOfSight2D(self._bounds['los'])

        self.status = 'Initialized'

        # Инициализируем options и положим в них bounds для наших экземпляров
        self._options = Options()
        self._options.set_bounds(self._bounds)

        self.t = None
        self.log = None
        self._info = None
        self.buffer = None
        self._locked_on = None
        self._keys = None
        self.pn = None
        self._missile_action, self._target_action = None, None
        self._obs_state = None

    def reset(self):
        if isinstance(self._scenarios, list):
            values = np.random.choice(self._scenarios)
            self._options.set_states(values)
        else:
            self._options.set_states(self._scenarios)

        self.t = 0

        self.missile.altitude = self._options.missile['altitude']
        self.target.altitude = self._options.target['altitude']

        missile_state = self.missile.reset(self._options.missile['initial_state'])
        target_state = self.target.reset(self._options.target['initial_state'])
        self.los.set_state(missile=missile_state, target=target_state)
        los_state = self.los.get_state()

        self._obs_state = np.array(
            [
                self._options.env['initial_distance'],
                self.target.overload,
                self.missile.overload,
                self._options.target['initial_state'][1]['vel'] / self.target.speed_of_sound,
                self._options.missile['initial_state'][2]['vel'] / self.missile.speed_of_sound,
                self.velR(missile_state, target_state, los_state),
                self._options.env['initial_heading_error'],
                self._options.env['initial_heading_angle'],
                self.missile.energetics.mass(self.t),
                self._options.env['altitude']
            ],
            dtype=np.float32
        )
        if self.agent in ('target', None):
            self.pn = PN(0, self._options.env['altitude'])

        self._keys = np.array(
            ['t', 'distance', 'target_overload', 'missile_overload', 'target_Ma', 'missile_Ma', 'velR', 'eps', 'q',
             'mass', 'altitude', 'missile_x', 'missile_z', 'missile_psi', 'missile_beta', 'target_x', 'target_z',
             'target_psi', 'chi', 'target_acceleration_z', 'sOs', 'ZEM'],
            dtype=np.object
        )

        log0 = np.concatenate([
            np.array([self.t], dtype=np.float32),
            self._obs_state,
            np.delete(missile_state, 2),
            np.array([self.missile.beta], dtype=np.float32),
            np.delete(target_state, 2),
            np.delete(los_state, 0),
            np.array([self.target._acceleration.z, self.target.speed_of_sound, self.ZEM], dtype=np.float32)
        ], dtype=np.float32)
        self.log = np.array([log0], dtype=np.float32)

        self.status = 'Dropped'
        self.buffer = {}
        return self._obs_state

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        return self.observation_space.low + state * (self.observation_space.high - self.observation_space.low)

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

    @property
    def options(self):
        return self._options

    @property
    def ZEM(self):
        """
        Нулевой промах (Zero-effort-miss) ракеты
        """
        _, _, s = self.missile.get_state()
        return self._obs_state[0] * self.get_eta(s, self.los.get_state())

    def step(self, action: np.ndarray):
        assert self._obs_state is not None, 'Call reset before using this method.'

        if self.agent == 'target':
            beta = self.pn.get_action(self)
            aZ = action * self._options.target['bounds']['overload_max'] * self.target.grav_accel
        elif self.agent == 'missile':
            assert self._target_action is not None, 'Set target action before it'
            beta = action * self._options.missile['bounds']['beta_max']
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

        tau = self.TAU

        _, _, missile_state = self.missile.get_state()
        _, target_state = self.target.get_state()
        los_state = self.los.get_state()

        self.buffer['s'] = self._obs_state

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

        self._obs_state = np.array(
            [
                los_state[0],
                self.target.overload,
                self.missile.overload,
                target_state[2] / self.target.speed_of_sound,
                missile_state[2] / self.missile.speed_of_sound,
                self.velR(missile_state, target_state, los_state),
                self.get_eta(missile_state, los_state),
                self.get_eta(target_state, los_state),
                self.missile.energetics.mass(self.t),
                self._options.env['altitude']
            ],
            dtype=np.float32
        )

        self.buffer['s_'] = self._obs_state

        appendix = np.concatenate([
            np.array([self.t], dtype=np.float32),
            self._obs_state,
            np.delete(missile_state, 2),
            np.array([self.missile.beta], dtype=np.float32),
            np.delete(target_state, 2),
            np.delete(los_state, 0),
            np.array([self.target._acceleration.z, self.target.speed_of_sound, self.ZEM], dtype=np.float32)
        ], dtype=np.float32)
        self.log = np.append(self.log, [appendix], axis=0)

        terminal, self._info = self._terminal()

        if self.status != 'Alive':
            self.status = 'Alive'

        return self._obs_state, self.reward, terminal, {}

    def _terminal(self):
        _, _, _, _, _, velR, eta_m, eta_t, _, _ = self._obs_state
        _, target_state = self.target.get_state()
        missile_terminal, missile_info = self.missile.terminal()
        target_terminal, target_info = self.target.terminal()
        los_terminal, los_info = self.los.terminal()

        if los_terminal:
            if abs(velR) < self.missile.bounds['relative_velocity_min'] \
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

            if abs(eta_m) > self.missile.bounds['coordinator_angle_max']:
                self.missile.status = 'Target  lost'
                self.status = f"MISSILE: {self.missile.status}. Eps = {abs(np.rad2deg(eta_m)):.2f} grad"
                return True, self.status

        if self.t > self._options.env['bounds']['termination_time']:
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
            x - self._options.missile['initial_state'][2]['x'],
            z - self._options.missile['initial_state'][2]['z']],
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
        vecX = i.rotate(self._options.missile['initial_state'][2]['psi'])

        d_ = vecD.len()
        zA_ = abs(vecD.angle_to(vecX))

        if d_ > self._options.env['bounds']['escape_distance'] \
                and zA_ < self._options.env['bounds']['escape_sector_angle'] / 2:
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
        pr = PlotlyRenderer(df, self._options)

        d = self._options.env['initial_distance']
        q = int(np.rad2deg(self._options.env['initial_heading_angle']))
        eps = int(np.rad2deg(self._options.env['initial_heading_error']))

        filename_const = f"{d}-{q}-{eps}"
        pr.plot(renderer, filename_const)


class LineOfSight2D:
    def __init__(self, bounds: dict):
        self.bounds = bounds
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
        if r < self.bounds['distance_min']:
            self.status = 'Hit'
            return True, f"{self.status}. D = {r:.2f} m"
        return False, None
