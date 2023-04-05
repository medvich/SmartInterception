from .options import BASE_PATH
import os
import yaml
import numpy as np


VALUES_FILENAME = 'values.yaml'


def make_escape_scenarios(n=10_000, seed=None):
    scenarios = []

    q44 = np.arange(0, 21, 1, dtype=np.int32)
    q34 = np.arange(21, 70, 1, dtype=np.int32)
    q24 = np.arange(70, 111, 1, dtype=np.int32)
    q14 = np.arange(111, 160, 1, dtype=np.int32)
    q04 = np.arange(160, 181, 1, dtype=np.int32)

    np.random.seed(seed)

    while len(scenarios) < n:
        with open(os.path.join(BASE_PATH, 'src', VALUES_FILENAME)) as f:
            values = yaml.safe_load(f)
        q = np.random.randint(0, 180, dtype=np.int32)
        if q in q04:
            d = np.random.randint(10000, 80500, dtype=np.int32)
        elif q in q14:
            d = np.random.randint(5000, 60500, dtype=np.int32)
        elif q in q24:
            d = np.random.randint(1000, 30500, dtype=np.int32)
        elif q in q34:
            d = np.random.randint(1000, 25500, dtype=np.int32)
        elif q in q44:
            d = np.random.randint(1000, 20500, dtype=np.int32)
        else:
            raise ValueError('Wrong q value')

        if d > values['missile']['bounds']['lock_on_distance'] * 1.25:
            eps = np.random.randint(-40, 40, dtype=np.int32)
        else:
            eps = np.random.randint(-20, 20, dtype=np.int32)

        # print(f'q={q:d}, d={d:d}, eps={eps:d}')

        values['environment']['initial_distance'] = d
        values['environment']['initial_heading_error'] = eps
        values['environment']['initial_heading_angle'] = q
        values['environment']['target_centered'] = False
        values['environment']['initial_psi'] = 0

        scenarios.append(values)

    return scenarios
