import numpy as np
from ..missile import Missile2D, PN
from ..target import Target2D
from ..options import Options, Acceleration, BASE_PATH
from gym import Env
from gym.spaces import Box
from easyvec import Vec2
import pandas as pd
from ..visualization import PlotlyRenderer
from typing import Union, Dict
import os
import yaml
from ..common import n_degree_curve
import joblib


DEFAULT_TARGET_REWARD_PARAMS = {
    'cHit': -33,
    'cMiss': 99,
    'cV0': 0.8,
    'nV': 0.3
}

DEFAULT_MISSILE_REWARD_PARAMS = {
    'cHit': 300,
    'cMiss': -237
}

with open(os.path.join(BASE_PATH, 'src', 'bounds.yaml')) as file:
    DEFAULT_BOUNDS = yaml.safe_load(file)


class Interception2D(Env):

    TAU = 0.1   # шаг по времени окружения, с

    def __init__(
            self,
            agent: Union[str, None],
            scenarios: Union[list, dict, str],
            bounds: Union[str, Dict[str, Dict], None] = None,
            target_reward_params: Union[Dict, None] = None,
            missile_reward_params: Union[Dict, None] = None,
            **kw
    ):

        # Объявляем границы нашего окружения, чтобы задать observation_space

        if not bounds:
            self._bounds = DEFAULT_BOUNDS
        elif type(bounds) is str:
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

        if self.agent == 'both':
            assert 'model' in kw
            self.target_model = kw['model']
        else:
            self.target_model = None

        # Функция, возвращающая важность вознаграждения в зависимости от дальности и ракурса

        i_path = os.path.join(BASE_PATH, 'files', 'importance.pkl')
        assert os.path.exists(i_path), 'Importance file does not exist'
        self.importance = joblib.load(i_path)
        del i_path

        # Словарь с коэффициентами для формирования вознаграждения цели

        if target_reward_params is None:
            target_reward_params = DEFAULT_TARGET_REWARD_PARAMS

        self.target_reward_params = target_reward_params

        # Словарь с коэффициентами для формирования вознаграждения ракеты

        if missile_reward_params is None:
            missile_reward_params = DEFAULT_MISSILE_REWARD_PARAMS

        self.missile_reward_params = missile_reward_params

        # Создадим словарь с историей моделирований, чтобы иметь возможность оценить тренд по какой-либо величине

        self.history = {
            'terminal_distance': [],
            'hits_count': 0,
            'misses_count': 0,
            'n_episodes': 0
        }

        # Инициализируем options и положим в них bounds для наших экземпляров
        self.options = Options()
        self.options.set_bounds(self._bounds)

        """
        Устанавливается, что обучаемый агент может наблюдать следующие параметры своего окружения:
          - Дальность до цели
          - Угловая скорость линии визирования
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
                self.options.env['bounds']['simulation_zone_radius'],
                np.pi / 2,
                self.options.target['bounds']['overload_max'],
                self.options.missile['bounds']['overload_max'],
                max(self.options.target['bounds']['mach_range']),
                max(self.options.missile['bounds']['mach_range']),
                2_000,
                self.options.missile['bounds']['coordinator_angle_max'],
                np.pi,
                max(self.options.missile['bounds']['mass']),
                max(self.options.env['bounds']['altitude_range'])
            ],
            dtype=np.float32
        )
        low = np.array(
            [
                -300,
                -np.pi / 2,
                -1 * self.options.target['bounds']['overload_max'],
                -1 * self.options.missile['bounds']['overload_max'],
                min(self.options.target['bounds']['mach_range']),
                min(self.options.missile['bounds']['mach_range']),
                self.options.missile['bounds']['relative_velocity_min'],
                -1 * self.options.missile['bounds']['coordinator_angle_max'],
                -np.pi,
                min(self.options.missile['bounds']['mass']),
                min(self.options.env['bounds']['altitude_range'])
            ],
            dtype=np.float32
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if self.agent in ('target', 'missile'):
            self.original_action_space = None
        elif self.agent == 'both':
            self.missile_original_action_space = None
            self.target_original_action_space = None

        # Инициализируем экземпляры нашего окружения

        self.missile = Missile2D(self._bounds['missile'])
        self.target = Target2D(self._bounds['target'])
        self.los = LineOfSight2D(self._bounds['los'])

        self.status = 'Initialized'
        self.info = None

        self.t, self._obs = None, None
        if self.agent in (['both', 'target', None]):
            self.pn = PN(self)
        else:
            self.pn = None

        self.log = None
        self.buffer = None

        self._locked_on = None
        self._keys = None
        self._missile_action, self._target_action = None, None

    def reset(self):
        # Если сценарии представлены в виде списка, то выбираем один случайным образом в качестве начального состония

        if isinstance(self._scenarios, list):
            values = np.random.choice(self._scenarios)
        else:
            values = self._scenarios

        self.options.set_states(values)

        self.t = 0

        self.missile.altitude = self.options.missile['altitude']
        self.target.altitude = self.options.target['altitude']

        missile_state = self.missile.reset(self.options.missile['initial_state'])
        target_state = self.target.reset(self.options.target['initial_state'])
        self.los.set_state(missile=missile_state, target=target_state)
        los_state = self.los.get_state()

        if self.agent == 'target':
            self.original_action_space = Box(
                low=-self.options.target['bounds']['overload_max'] * self.target.grav_accel,
                high=self.options.target['bounds']['overload_max'] * self.target.grav_accel,
                shape=(1,),
                dtype=np.float32
            )
        elif self.agent == 'missile':
            self.original_action_space = Box(
                low=1,
                high=80,
                shape=(1,),
                dtype=np.float32
            )
        elif self.agent == 'both':
            self.target_original_action_space = Box(
                low=-self.options.target['bounds']['overload_max'] * self.target.grav_accel,
                high=self.options.target['bounds']['overload_max'] * self.target.grav_accel,
                shape=(1,),
                dtype=np.float32
            )
            self.missile_original_action_space = Box(
                low=1,
                high=80,
                shape=(1,),
                dtype=np.float32
            )

        velR, Omega = self.los.step(missile_state, target_state)

        self._obs = np.array(
            [
                self.options.env['initial_distance'],
                Omega,
                self.target.overload,
                self.missile.overload,
                self.options.target['initial_state'][1]['vel'] / self.target.speed_of_sound,
                self.options.missile['initial_state'][2]['vel'] / self.missile.speed_of_sound,
                velR,
                self.options.env['initial_heading_error'],
                self.options.env['initial_heading_angle'],
                self.missile.energetics.mass(self.t),
                self.options.env['altitude']
            ],
            dtype=np.float32
        )

        if self.pn:
            self.pn.reset(0, self.options.env['altitude'])

        self._locked_on = False

        self._keys = np.array(
            ['t', 'distance', 'Omega', 'target_overload', 'missile_overload', 'target_Ma', 'missile_Ma', 'velR', 'eps',
             'q', 'mass', 'altitude', 'missile_x', 'missile_z', 'missile_psi', 'missile_beta', 'target_x', 'target_z',
             'target_psi', 'chi', 'target_acceleration_z', 'sOs', 'ZEM'],
            dtype=np.object
        )

        log0 = np.concatenate([
            np.array([self.t], dtype=np.float32),
            self._obs,
            np.delete(missile_state, 2),
            np.array([self.missile.beta], dtype=np.float32),
            np.delete(target_state, 2),
            np.delete(los_state, 0),
            np.array([self.target.acceleration.z, self.target.speed_of_sound, self.ZEM], dtype=np.float32)
        ], dtype=np.float32)
        self.log = np.array([log0], dtype=np.float32)

        self.status = 'Dropped'
        self.buffer = {
            'd0': self._obs[0],
            'd': self._obs[0],
            'd_': self._obs[0]
        }
        return self.normalize_state(self._obs)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        return self.observation_space.low + state * (self.observation_space.high - self.observation_space.low)

    def rescale_action(self, action: Union[np.ndarray, float, int], **kw):
        """
        Возвращает действие в абсолютном диапазоне
        """
        if self.agent == 'both':
            assert None not in (self.missile_original_action_space, self.target_original_action_space)
            if kw['agent'] == 'missile':
                return self.missile_original_action_space.low + (
                            self.missile_original_action_space.high - self.missile_original_action_space.low) * (
                               (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
                       )
            if kw['agent'] == 'target':
                return self.target_original_action_space.low + (
                            self.target_original_action_space.high - self.target_original_action_space.low) * (
                               (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
                       )
            raise ValueError('Invalid agent name')
        return self.original_action_space.low + (self.original_action_space.high - self.original_action_space.low) * (
                (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
        )

    def step(self, action: Union[np.ndarray, None]):
        assert self._obs is not None, 'Call reset before using this method.'

        if self.agent == 'target':
            beta = self.pn.get_action(k=1)
            aZ = self.rescale_action(action)
        # elif self.agent == 'missile':
        #     assert self._target_action is not None, 'Set target action before it'
        #     beta = self.rescale_action(action)
        #     aZ = self._target_action

        # Под 'both' подразумевается, что оба агента управляются при помощи моделей, НО уже обученных. Т.е. действия
        # задаются явно при помощи соответствующих методов. Обучение обоих агентов одновременно в данном окружении не
        # поддерживается.

        elif self.agent == 'both':
            # assert self._missile_action is not None, 'Set missile action before it'
            # assert self._target_action is not None, 'Set target action before it'
            K = self.rescale_action(action, agent='missile')[0]
            beta = self.pn.get_action(k=K)

            condition1 = abs(np.pi - self._obs[8] % 2 * np.pi) < self.options.target['bounds']['coordinator_angle_max']
            condition2 = self._obs[0] < max(self.options.target['bounds']['detection_distance_range'])
            condition3 = self._obs[0] > min(self.options.target['bounds']['detection_distance_range'])

            target_action, _ = self.target_model.predict(self.normalize_state(self._obs))
            aZ = self.rescale_action(target_action, agent='target')

            # if all([condition1, condition2, condition3]) or self._locked_on:
            #     target_action, _ = self.target_model.predict(self.normalize_state(self._obs))
            #     aZ = self.rescale_action(target_action, agent='target')
            # else:
            #     aZ = self.options.target['initial_state'][0].z
        else:
            assert self._target_action is not None, 'Set target action before it'
            beta = self.pn.get_action()
            aZ = self._target_action

        tau = self.TAU

        _, _, missile_state = self.missile.get_state()
        _, target_state = self.target.get_state()
        los_state = self.los.get_state()

        self.buffer['d'] = self._obs[0]

        missile_ds = self.missile.step(beta)
        target_ds = self.target.step(Acceleration(0, aZ))
        los_ds = self.los.step(missile_state, target_state)

        velR, Omega = los_ds

        missile_state = missile_state + tau * missile_ds
        target_state = target_state + tau * target_ds
        los_state = los_state - tau * los_ds

        self.missile.set_state(missile_state, t=self.t)
        self.target.set_state(target_state, t=self.t)
        self.los.set_state(los=los_state)

        self.t += tau

        self._obs = np.array(
            [
                los_state[0],
                Omega,
                self.target.overload,
                self.missile.overload,
                target_state[2] / self.target.speed_of_sound,
                missile_state[2] / self.missile.speed_of_sound,
                velR,
                self.get_eta(missile_state, los_state),
                self.get_eta(target_state, los_state),
                self.missile.energetics.mass(self.t),
                self.options.env['altitude']
            ],
            dtype=np.float32
        )

        self.buffer['d_'] = self._obs[0]

        appendix = np.concatenate([
            np.array([self.t], dtype=np.float32),
            self._obs,
            np.delete(missile_state, 2),
            np.array([self.missile.beta], dtype=np.float32),
            np.delete(target_state, 2),
            np.delete(los_state, 0),
            np.array([self.target.acceleration.z, self.target.speed_of_sound, self.ZEM], dtype=np.float32)
        ], dtype=np.float32)
        self.log = np.append(self.log, [appendix], axis=0)

        if self.status != 'Alive':
            self.status = 'Alive'

        terminated = self.terminated
        if terminated:
            self.history['terminal_distance'].append(self._obs[0])
            self.history['n_episodes'] += 1
            if 'hit' in self.status.lower():
                self.history['hits_count'] += 1
            else:
                self.history['misses_count'] += 1
            self.info = 'Terminated'
        reward = self.reward

        return self.normalize_state(self._obs), reward, terminated, {}

    @property
    def terminated(self):
        d, _, _, _, _, _, velR, eta_m, eta_t, _, _ = self._obs
        _, target_state = self.target.get_state()
        missile_terminated, missile_info = self.missile.terminated()
        target_terminated, target_info = self.target.terminated()
        los_terminated, los_info = self.los.terminated()

        if los_terminated:
            if abs(velR) < self.missile.bounds['relative_velocity_min'] \
                    and self.missile.energetics.thrust(self.t) == 0:
                self.missile.status = 'Low relative velocity'
                self.status = f"MISSILE: {self.missile.status}. Vel_ = {velR:.2f} m/sec"
                return True
            self.status = "LOS: " + los_info
            return True

        if target_terminated:
            self.status = "TARGET: " + target_info
            return True

        escaped, d_, zA_ = self._escaped(target_state)
        if escaped:
            self.target.status = 'Escaped'
            self.status = f"TARGET: {self.target.status}. D_ = {d_:.2f} m, Zone_Angle = {np.rad2deg(zA_):.2f} grad"
            return True

        if missile_terminated:
            self.status = "MISSILE: " + missile_info
            return True

        if self._locked_on:
            if abs(eta_m) > self.missile.bounds['coordinator_angle_max'] + np.radians(20) and d > 1e3:
                self.missile.status = 'Target  lost'
                self.status = f"MISSILE: {self.missile.status}. Eps = {abs(np.rad2deg(eta_m)):.2f} grad"
                return True

        if self.t > self.options.env['bounds']['termination_time']:
            self.status = f"ENV: Maximum flight time exceeded. t = {self.t:.2f} sec"
            return True

        else:
            return False

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

    @property
    def locked_on(self) -> bool:
        """
        Булевый параметр, указывающий на то, захватила ли БРЛС ракеты цель. При положительном значении ракета начинает
        двигаться в соответствии со своим законом самонаведения (аналитическим или при помощи модели)
        """
        return self._locked_on

    @locked_on.setter
    def locked_on(self, value: bool) -> None:
        self._locked_on = value

    @property
    def missile_action(self):
        return self._missile_action

    @missile_action.setter
    def missile_action(self, action: float) -> None:
        """
        Задать действие агента напрямую
        """
        self._missile_action = action

    @property
    def target_action(self):
        return self._target_action

    @target_action.setter
    def target_action(self, action: float) -> None:
        """
        Задать действие цели напрямую
        """
        self._target_action = action

    @property
    def ZEM(self):
        """
        Нулевой промах (Zero-effort-miss) ракеты
        """
        _, _, s = self.missile.get_state()
        return self._obs[0] * self.get_eta(s, self.los.get_state())

    @property
    def reward(self):
        if self.agent in ('missile', 'both'):
            return self.missile_reward
        if self.agent == 'target':
            return self.target_reward
        return None

    @property
    def missile_reward(self):
        d, _, _, nM, _, _, velR, eps, q, _, _ = self._obs

        if 'hit' in self.status.lower():
            return self.missile_reward_params['cHit']
        if 'missile' in self.status.lower():
            return self.missile_reward_params['cMiss']

        c1 = n_degree_curve(d, (0, 80e3), (0, 1), 0.3)
        c2 = 1. - c1
        c3 = 1.
        c4 = 1.

        r1 = -np.log((self.observation_space.high[6] - velR) / self.observation_space.high[6])
        r2 = -np.log(abs(eps) / self.options.missile['bounds']['coordinator_angle_max'])
        r3 = -np.log(abs(nM) / self.options.missile['bounds']['overload_max'])
        r4 = -np.log(max(d, 0) / self.buffer['d0'])

        return (c1 * r1) + (c2 * r2) + (c3 * r3) + (c4 * r4)

    @property
    def target_reward(self):
        d, _, _, _, _, _, velR, eps, q, _, _ = self._obs
        if 'hit' in self.status.lower():
            return self.target_reward_params['cHit']
        if 'missile' in self.status.lower():
            return self.target_reward_params['cMiss']
        c1 = n_degree_curve(d, (0, 80e3), (0, self.target_reward_params['cV0']), self.target_reward_params['nV'])
        c2 = 1. - c1
        i = self.importance(np.rad2deg(q), d)[0]
        r = (c1 * (np.exp(-velR / self.observation_space.high[6])) +
             c2 * (abs(eps) / self.options.missile['bounds']['coordinator_angle_max']))

        return r * i

    @property
    def scenarios(self):
        return self._scenarios

    @scenarios.setter
    def scenarios(self, scenarios: float) -> None:
        """
        Перезаписать сценарии
        """
        self._scenarios = scenarios

    def close(self):
        # self.target_model.env.close()
        pass


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

    def terminated(self):
        r, chi = self._state
        if r < self.bounds['distance_min']:
            self.status = 'Hit'
            return True, f"{self.status}. D = {r:.2f} m"
        return False, None
