from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import optuna
from statistics import mean


class SubprocVecEnvCallback(BaseCallback):
    """
    Callback used for evaluating and reporting a SPVEnv
    """
    def __init__(
            self,
            env: SubprocVecEnv,
            verbose=0
    ):
        super().__init__(verbose)

        self.env = env
        self.buffer = {}

        n_envs = len(self.env.remotes)

        for i in range(n_envs):
            self.buffer[f'env_{i}'] = {
                'sum': 0,
                'steps': 0,
                'mean_terminal_distance': 0
            }

    def _on_step(self) -> bool:
        rewards = self.env.get_attr('reward')
        histories = self.env.get_attr('history')

        for i, (r, h) in enumerate(zip(rewards, histories)):

            self.buffer[f'env_{i}']['steps'] += 1
            self.buffer[f'env_{i}']['sum'] += r
            self.buffer[f'env_{i}']['mean_terminal_distance'] = \
                mean(h['terminal_distance']) if h['terminal_distance'] else 0

            # self.logger.record(f'sum_reward/env_{i}', self.buffer[f'env_{i}']['sum'])
            # self.logger.record(f'reward/env_{i}', r)
            self.logger.record(f'mean_dt/env_{i}', self.buffer[f'env_{i}']['mean_terminal_distance'])

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

    def _on_step(self) -> bool:
        self.step += 1
        histories = self.env.get_attr('history')
        for i, value in enumerate(histories):
            if value['terminal_distance']:
                self.intermediate_value.append(mean(value['terminal_distance']))
        self.trial.report(
            mean(self.intermediate_value) if self.intermediate_value else 0,
            step=self.step
        )
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return True
