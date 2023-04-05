"""
Запускать с терминала из корневой директории
"""

import pytest
import numpy as np
from src.options import Options, BASE_PATH
import os
import yaml


VALUES_FILENAME = 'values.yaml'


@pytest.mark.parametrize(
    'distance, eps, q, psi, xM, zM, psiM',
    [
        (10000, 0, 0, 0, -10000, 0, 0),
        (10000, 0, 180, 30, 10000 * np.cos(np.radians(30)), 10000 * np.sin(np.radians(30)), np.radians(30) + np.pi),
        (10000, 10, 160, -30, 10000 * np.cos(np.radians(-50)), 10000 * np.sin(np.radians(-50)), np.radians(120)),
        (10000, -10, 160, -30, 10000 * np.cos(np.radians(-50)), 10000 * np.sin(np.radians(-50)), np.radians(140)),
        (10000, 10, 160, -140, 10000 * np.cos(np.radians(-160)), 10000 * np.sin(np.radians(-160)), np.radians(10)),
        (10000, -10, 160, -140, 10000 * np.cos(np.radians(-160)), 10000 * np.sin(np.radians(-160)), np.radians(30)),
        (10000, 10, 160, 150, 10000 * np.cos(np.radians(130)), 10000 * np.sin(np.radians(130)), np.radians(300)),
        (10000, -10, 160, 150, 10000 * np.cos(np.radians(130)), 10000 * np.sin(np.radians(130)), np.radians(320))
    ]
)
def test_target_centered_initial_state(distance, eps, q, psi, xM, zM, psiM):
    with open(os.path.join(BASE_PATH, 'src', VALUES_FILENAME)) as f:
        values = yaml.safe_load(f)
    values['environment']['initial_distance'] = distance
    values['environment']['initial_heading_error'] = eps
    values['environment']['initial_heading_angle'] = q
    values['environment']['target_centered'] = True
    values['environment']['initial_psi'] = psi

    opts = Options(values)

    assert xM == pytest.approx(opts.missile['initial_state'][2]['x'], abs=0.1)
    assert zM == pytest.approx(opts.missile['initial_state'][2]['z'], abs=0.1)
    assert psiM == pytest.approx(opts.missile['initial_state'][2]['psi'], abs=0.1)


@pytest.mark.parametrize(
    'distance, eps, q, psi, xT, zT, psiT, psiM',
    [
        (10000, 0, 0, 0, 10000, 0, 0, 0),
        (10000, 0, 0, 30, 10000 * np.cos(np.radians(30)), 10000 * np.sin(np.radians(30)), np.radians(30), np.radians(30)),
        (10000, 10, 160, 30, 10000 * np.cos(np.radians(30)), 10000 * np.sin(np.radians(30)), np.radians(230), np.radians(20)),
        (10000, -10, 160, 30, 10000 * np.cos(np.radians(30)), 10000 * np.sin(np.radians(30)), np.radians(230), np.radians(40)),
        (10000, 10, 40, -60, 10000 * np.cos(np.radians(-60)), 10000 * np.sin(np.radians(-60)), np.radians(260), np.radians(-70) % (2 * np.pi)),
        (10000, -10, 40, -60, 10000 * np.cos(np.radians(-60)), 10000 * np.sin(np.radians(-60)), np.radians(260), np.radians(-50) % (2 * np.pi))
    ]
)
def test_missile_centered_initial_state(distance, eps, q, psi, xT, zT, psiT, psiM):
    with open(os.path.join(BASE_PATH, 'src', VALUES_FILENAME)) as f:
        values = yaml.safe_load(f)
    values['environment']['initial_distance'] = distance
    values['environment']['initial_heading_error'] = eps
    values['environment']['initial_heading_angle'] = q
    values['environment']['target_centered'] = False
    values['environment']['initial_psi'] = psi

    opts = Options(values)

    assert xT == pytest.approx(opts.target['initial_state'][1]['x'], abs=0.1)
    assert zT == pytest.approx(opts.target['initial_state'][1]['z'], abs=0.1)
    assert psiT == pytest.approx(opts.target['initial_state'][1]['psi'], abs=0.1)
    assert psiM == pytest.approx(opts.missile['initial_state'][2]['psi'], abs=0.1)


