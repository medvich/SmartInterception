from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from src import Interception2D
from src.scenarios import make_scenario_batches, train_zones
from src.options import BASE_PATH, LOG_PATH
from src.callbacks import SPVECallback
import os
import torch
from typing import Union
import yaml


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


N_PROC = 4


def make_env(
        seed,
        target_reward_params=None
):
    def _f():
        env_ = Interception2D(
            agent='target',
            bounds='bounds.yaml',
            scenarios=[],
            target_reward_params=target_reward_params
        )
        env_.seed(seed)
        return env_
    return _f


def make_logpathname(name_const):
    pathname = name_const
    i = 1
    while os.path.exists(os.path.join(LOG_PATH, pathname)):
        pathname = name_const + f'_{i}'
        i += 1
    return os.path.join(LOG_PATH, pathname)


def make_modelname(name_const):
    filename = name_const + '.zip'
    i = 1
    while os.path.exists(os.path.join(BASE_PATH, 'models', filename)):
        filename = name_const + f'_{i}' + '.zip'
        i += 1
    return os.path.join(BASE_PATH, 'models', filename)


def train(
        batches: list,
        total_timesteps: int = 3_500_000,
        model: Union[str, None] = None,
        save=True,
        target_reward_params=None
):
    stack = [make_env(seed, target_reward_params=target_reward_params) for seed in range(N_PROC)]
    spv_env = SubprocVecEnv(stack, start_method='spawn')

    with open(os.path.join(BASE_PATH, 'src', 'models', 'hyperparams', 'sac_target.yaml')) as f:
        kwargs = yaml.safe_load(f)

    if isinstance(model, str):
        MODELNAME = model
        device = kwargs.pop('device')
        model = SAC.load(os.path.join(BASE_PATH, 'models', MODELNAME), env=spv_env, device=device)
    else:
        LOGPATHNAME = make_logpathname('SAC-T')
        MODELNAME = make_modelname('SAC-T')
        afn = kwargs['policy_kwargs'].pop('activation_fn')
        if afn == 'ELU':
            kwargs['policy_kwargs']['activation_fn'] = torch.nn.ELU
        elif afn == 'ReLU':
            kwargs['policy_kwargs']['activation_fn'] = torch.nn.ReLU
        else:
            raise ValueError()
        kwargs['policy'] = 'MlpPolicy'
        kwargs['env'] = spv_env
        kwargs['tensorboard_log'] = LOGPATHNAME
        kwargs['train_freq'] = tuple(kwargs['train_freq'])

        model = SAC(**kwargs)

    spv_callback = SPVECallback(model.env)

    for i, scenarios in enumerate(batches):
        model.env.set_attr('scenarios', scenarios)
        model.env.reset()
        model.learn(
            total_timesteps=int(total_timesteps/len(batches)),
            callback=spv_callback,
            tb_log_name=f'zone'
        )

    model.env.close()

    if save:
        model.save(os.path.join(BASE_PATH, 'models', MODELNAME))

    return model


if __name__ == '__main__':

    # train_zones.reverse()

    scenario_batches = make_scenario_batches(
        zones=train_zones,
        target_centered=True,
        plot=False
    )

    sac = train(
        model=None,
        batches=scenario_batches,
        save=True
    )
