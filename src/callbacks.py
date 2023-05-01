from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import optuna
from statistics import mean
from typing import Union
from copy import deepcopy


class SPVECallback(BaseCallback):
    """
    Callback used for evaluating and reporting a SPVEnv
    """
    def __init__(
            self,
            env: SubprocVecEnv,
            verbose=0,
            clean_buffer_freq: Union[int, None] = None
    ):
        super().__init__(verbose=verbose)

        self.env = env
        n_envs = len(self.env.remotes)
        self.cbf = clean_buffer_freq
        self.step = 0

        self.clean_buffer = {}
        for i in range(n_envs):
            self.clean_buffer[f'env_{i}'] = {
                'steps': 0,
                'mean_terminal_distance': 0,
                'hits_count': 0,
                'misses_count': 0,
                'prev_n_episodes': 0
            }

        self.buffer = deepcopy(self.clean_buffer)
        self.reference_buffer = deepcopy(self.buffer)

    def _on_step(self) -> bool:
        self.step += 1
        rewards = self.env.get_attr('reward')
        histories = self.env.get_attr('history')

        for i, (r, h) in enumerate(zip(rewards, histories)):

            if h['n_episodes'] % self.cbf == 0 and self.buffer[f'env_{i}']['prev_n_episodes'] != h['n_episodes']:
                self.reference_buffer = deepcopy(self.buffer)

            self.buffer[f'env_{i}']['steps'] += 1
            self.buffer[f'env_{i}']['mean_terminal_distance'] = \
                mean(h['terminal_distance']) if h['terminal_distance'] else 0
            self.buffer[f'env_{i}']['hits_count'] = h['hits_count']
            self.buffer[f'env_{i}']['misses_count'] = h['misses_count']

            hits_after_cleaning = \
                self.buffer[f'env_{i}']['hits_count'] - self.reference_buffer[f'env_{i}']['hits_count']
            misses_after_cleaning = \
                self.buffer[f'env_{i}']['misses_count'] - self.reference_buffer[f'env_{i}']['misses_count']
            try:
                hit_miss_relation_after_cleaning = hits_after_cleaning / misses_after_cleaning
            except ZeroDivisionError:
                hit_miss_relation_after_cleaning = hits_after_cleaning

            # self.logger.record(f'hit_miss_relation/env_{i}', hit_miss_relation_after_cleaning)
            self.logger.record(f'hits_count/env_{i}', hits_after_cleaning)
            self.logger.record(f'misses_count/env_{i}', misses_after_cleaning)
            self.logger.record(f'mean_dt/env_{i}', self.buffer[f'env_{i}']['mean_terminal_distance'])

            self.buffer[f'env_{i}']['prev_n_episodes'] = h['n_episodes']

        return True


class TrialSPVECallback(BaseCallback):
    """
    Callback used for reporting a trial
    """
    def __init__(
        self,
        env: SubprocVecEnv,
        trial: optuna.Trial,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.env = env
        self.trial = trial
        self.is_pruned = False
        self.step = 0
        self.intermediate_value = []
        self.misses, self.hits = None, None

    def _on_step(self) -> bool:
        self.step += 1
        histories = self.env.get_attr('history')
        self.misses, self.hits = [], []

        for i, h in enumerate(histories):
            if h['terminal_distance']:
                self.intermediate_value.append(mean(h['terminal_distance']))
            self.misses.append(h['misses_count'])
            self.hits.append(h['hits_count'])

        self.trial.report(
            # mean(self.intermediate_value) if self.intermediate_value else 0,
            min(self.hits),
            step=self.step
        )
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return True
