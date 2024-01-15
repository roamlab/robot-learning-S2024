from simple_maze import SimpleMaze
import random
from data_utils import load_data
import torch
import numpy as np
import argparse
import time
from numpngw import write_apng


def test_model(policy, n_test, obs_type, maps, gui=False):
    success = 0
    imgs = []
    max_dists = np.zeros(n_test)
    min_dists = np.zeros(n_test)
    env = SimpleMaze(maps=maps, obs_type=obs_type)
    for i in range(n_test):
        obs = env.reset()
        goal_pos = env.map.get_goal_spawn_pos()
        agent_pos = env.map.get_agent_spawn_pos()
        max_dists[i] = np.linalg.norm(goal_pos - agent_pos)
        done = False
        agent_poses = []
        for step in range(50):
            act = policy.get_action(obs)
            obs, _, done, info = env.step(act)
            agent_poses.append(info['agent'])

            if gui:
              img = info['rgb']
              img = (img * 255).astype(np.uint8)
              imgs.append(img)

            if done:
                success += 1
                break
        dist = np.min(np.linalg.norm(np.asarray(agent_poses)[:, :2] - goal_pos, axis=-1))
        if done:
            dist = 0

        min_dists[i] = dist

    success_rate = success / n_test

    # print('success_rate', success_rate)

    score = np.mean((max_dists - min_dists) / max_dists)
    return success_rate, score, imgs


def score_pos_bc(policy, gui=False, model=None):
    data = load_data('./data/map1.pkl')
    data.pop('rgb')
    data.pop('agent')
    data['obs'] = data.pop('poses')
    
    if model is not None:
        policy.network.load_state_dict(torch.load(model))
    else:
        policy.train(data)
    _, score, imgs = test_model(policy, n_test=100, obs_type="poses", maps=[0], gui=gui)

    if gui:
      write_apng('part_1_anim.png', imgs, delay=40)
    return score


def score_rgb_bc1(policy, gui=False, model=None):
    data = load_data('./data/map1.pkl')
    data.pop('poses')
    data.pop('agent')
    data['obs'] = data.pop('rgb')

    if model is not None:
        policy.network.load_state_dict(torch.load(model))
    else:
        policy.train(data)
    _, score, imgs = test_model(policy, n_test=100, obs_type="rgb", maps=[0], gui=gui)

    if gui:
      write_apng('part_2_anim.png', imgs, delay=40)
    return score


def score_rgb_bc2(policy, gui=False, model=None):
    data = load_data('./data/all_maps.pkl')
    data['obs'] = data.pop('rgb')

    if model is not None:
        policy.network.load_state_dict(torch.load(model))
    else:
        policy.train(data)
    _, score, imgs = test_model(policy, n_test=100, obs_type="rgb", maps=None, gui=gui)

    if gui:
      write_apng('part_3_anim.png', imgs, delay=40)
    return score
