from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from src import Interception2D, make_escape_scenarios, plot_scenarios
from src.options import BASE_PATH, LOG_PATH
from src.callbacks import SubprocVecEnvCallback
from src.common import n_degree_curve
import os
import torch


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


N_PROC = 6
policy_kwargs = dict(activation_fn=torch.nn.ELU,
                     net_arch=dict(pi=[128, 128], qf=[128, 128]))
train_zones = [
    {'d_min': 0,
     'd_max': 20000,
     'q_min': 0,
     'q_max': 70
     },
    {'d_min': 20000,
     'd_max': 80000,
     'q_min': 0,
     'q_max': 70
     },
    {'d_min': 0,
     'd_max': 20000,
     'q_min': 70,
     'q_max': 110
     },
    {'d_min': 20000,
     'd_max': 80000,
     'q_min': 70,
     'q_max': 110
     },
    {'d_min': 0,
     'd_max': 20000,
     'q_min': 110,
     'q_max': 160
     },
    {'d_min': 20000,
     'd_max': 50000,
     'q_min': 110,
     'q_max': 160
     },
    {'d_min': 50000,
     'd_max': 80000,
     'q_min': 110,
     'q_max': 160
     },
    {'d_min': 0,
     'd_max': 20000,
     'q_min': 160,
     'q_max': 180
     },
    {'d_min': 20000,
     'd_max': 50000,
     'q_min': 160,
     'q_max': 180
     },
    {'d_min': 50000,
     'd_max': 80000,
     'q_min': 160,
     'q_max': 180
     },
]
train_zones.reverse()


def make_env(seed):
    def _f():
        env_ = Interception2D(agent='target', bounds='bounds.yaml', scenarios=[])
        env_.seed(seed)
        return env_
    return _f


def make_scenario_batches(
        zones,
        target_centered=False,
        plot=False
):
    scenario_batches = []
    for zone in zones:
        eps_max = n_degree_curve(zone['d_max'], (0, 80000), (0, 40), 1)
        scenarios, params = make_escape_scenarios(
            n=20,
            seed=1000,
            target_centered=target_centered,
            d_min=zone['d_min'],
            d_max=zone['d_max'],
            q_min=zone['q_min'],
            q_max=zone['q_max'],
            eps_min=-eps_max,
            eps_max=eps_max
        )
        if plot:
            plot_scenarios(scenarios, params)
        scenario_batches.append(scenarios)
    return scenario_batches


def make_modelname(name_const):
    filename = name_const + '.zip'
    i = 1
    while os.path.exists(os.path.join(BASE_PATH, 'models', filename)):
        filename = name_const + f'_{i}' + '.zip'
        i += 1
    return os.path.join(BASE_PATH, 'models', filename)


def train(
        scenario_batches,
        device='cpu',
        buffer_size=5_000_000,
        learning_starts=10_000,
        batch_size=512,
        tau=0.2,
        total_timesteps=100_000,
        train_freq_per_step=100,
        gamma=0.99,
        ent_coef=0.1
):
    stack = [make_env(seed) for seed in range(N_PROC)]
    spv_env = SubprocVecEnv(stack, start_method='spawn')
    model = SAC(
        'MlpPolicy',
        spv_env,
        verbose=1,
        tensorboard_log=LOG_PATH,
        device=device,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        tau=tau,
        train_freq=(train_freq_per_step, 'step'),
        gamma=gamma,
        ent_coef=ent_coef
    )

    for scenarios in scenario_batches:
        model.env.set_attr('scenarios', scenarios)
        spv_env.reset()
        model.learn(total_timesteps=total_timesteps, callback=SubprocVecEnvCallback(model.env))

    spv_env.close()

    return model


if __name__ == '__main__':

    # print(torch.cuda.get_device_name())

    batches = make_scenario_batches(
        zones=train_zones,
        target_centered=True,
        plot=False)
    sac = train(
        batches,
        buffer_size=5_000_000,
        learning_starts=0,
        batch_size=512,
        tau=0.2,
        total_timesteps=10_000,
        train_freq_per_step=100,
        gamma=0.99,
        ent_coef=0.1
    )

    sac.save(os.path.join(BASE_PATH, 'models', make_modelname('SAC')))
