from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class SubprocVecEnvCallback(BaseCallback):

    def __init__(self, env, verbose=0):
        super().__init__(verbose)

        self.env = env
        self.buffer = {}

        n_envs = len(self.env.remotes)
        self.relations = np.repeat(0, n_envs)

        for i, _ in enumerate(self.relations):
            self.buffer[f'env_{i}'] = {
                'positive': 0,
                'negative': 0,
                'relation': 0,
                'sum': 0,
                'steps': 0,
                'norm_reward': 0
            }

    def _on_step(self) -> bool:
        rewards = self.env.get_attr('reward')

        for i, value in enumerate(rewards):
            self.buffer[f'env_{i}']['steps'] += 1
            self.buffer[f'env_{i}']['sum'] += value
            try:
                self.buffer[f'env_{i}']['norm_reward'] = \
                    self.buffer[f'env_{i}']['sum'] / self.buffer[f'env_{i}']['steps']
            except ZeroDivisionError:
                self.buffer[f'env_{i}']['norm_reward'] = 0
            if value > 0:
                self.buffer[f'env_{i}']['positive'] += 1
            elif value < 0:
                self.buffer[f'env_{i}']['negative'] += 1
            else:
                pass
            try:
                self.buffer[f'env_{i}']['relation'] = \
                    self.buffer[f'env_{i}']['positive'] / self.buffer[f'env_{i}']['negative']
            except ZeroDivisionError:
                self.buffer[f'env_{i}']['relation'] = self.buffer[f'env_{i}']['positive']
            self.logger.record(f'Pos-Neg-Relation/env_{i}', self.buffer[f'env_{i}']['relation'])
            self.logger.record(f'Normalized-Reward/env_{i}', self.buffer[f'env_{i}']['norm_reward'])

        return True

    def _on_training_end(self) -> bool:
        for i, _ in enumerate(self.relations):
            self.buffer[f'env_{i}'] = {
                'positive': 0,
                'negative': 0,
                'relation': 0,
                'sum': 0,
                'steps': 0,
                'norm_reward': 0
            }
        return True
