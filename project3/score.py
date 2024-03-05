import numpy as np
# from arm_gui import ArmGUI, Renderer
from render import Renderer
import time
import math

np.set_printoptions(suppress=True)


def reset(arm_teacher, arm_student, torque):
    initial_state = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))  # position + velocity
    initial_state[0] = -math.pi / 2.0
    arm_teacher.set_state(initial_state)
    arm_student.set_state(initial_state)

    action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
    action[0] = torque
    arm_teacher.set_action(action)
    arm_student.set_action(action)

    arm_teacher.set_t(0)
    arm_student.set_t(0)


def set_torque0(arm_teacher, arm_student, torque):
    action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
    action[0] = torque
    arm_teacher.set_action(action)
    arm_student.set_action(action)


def score_random_torque(arm_teacher, arm_student, gui):
    np.random.seed(10)
    time_limit = 5
    num_tests = 50

    mses = []
    scores = []
    torques = np.random.uniform(-1.5, 1.5, num_tests)
    for i, torque in enumerate(torques):
        print("\n----------------------------------------")
        print(f'TEST {i+1} (Torque = {torque} Nm)\n')
        reset(arm_teacher, arm_student, torque)

        if gui:
            renderer = Renderer()
            time.sleep(1)

        mse_list = []
        while arm_teacher.get_t() < time_limit:
            t = time.time()
            arm_teacher.advance()
            arm_student.advance()
            if gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            mse_list.append(mse)

        mse = np.array(mse_list).mean()
        mses.append(mse)
        print(f'average mse: {mse}')
        if mse < 0.008:
          score = 0.5
        if mse < 0.0005:
          score = 1
        if mse >= 0.008:
          score = 0
        scores.append(score)
        print(f'Score: {score}/{1}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{50}*{5} = {np.array(scores).sum()/50*5:.2f}')
    print("----------------------------------------\n")
    # print(max(mses))


def score_linear_torques(arm_teacher, arm_student, gui):
    np.random.seed(10)
    time_limit = 5
    num_tests = 50

    mses = []
    scores = []
    torques = np.random.uniform(0.5, 1.5, num_tests)
    for i, torque in enumerate(torques):
        print("\n----------------------------------------")
        print(f'TEST {i+1} (Torque 0 -> {torque} Nm)\n')
        reset(arm_teacher, arm_student, 0)

        if gui:
            renderer = Renderer()
            time.sleep(1)

        mse_list = []
        while arm_teacher.get_t() < time_limit:
            t = time.time()
            set_torque0(arm_teacher, arm_student, arm_teacher.get_t() / time_limit * torque)
            arm_teacher.advance()
            arm_student.advance()
            if gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            mse_list.append(mse)

        mse = np.array(mse_list).mean()
        mses.append(mse)
        print(f'average mse: {mse}')
        if mse < 0.008:
          score = 0.5
        if mse < 0.0005:
          score = 1
        if mse >= 0.008:
          score = 0
        scores.append(score)
        print(f'Score: {score}/{1}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{50}*{5} = {np.array(scores).sum()/50*5:.2f}')
    print("----------------------------------------\n")
    # print(max(mses))


def score_two_torques(arm_teacher, arm_student, gui):
    np.random.seed(10)
    time_limit = 5
    num_tests = 50

    mses = []
    scores = []
    torques1 = np.random.uniform(-1, 1, num_tests)
    torques2 = np.random.uniform(-1, 1, num_tests)
    for i, (torque1, torque2) in enumerate(zip(torques1, torques2)):
        print("\n----------------------------------------")
        print(f'TEST {i+1} (Torque 1 = {torque1} Nm,  Torque 2 = {torque2} Nm)\n')
        reset(arm_teacher, arm_student, 0)

        if gui:
            renderer = Renderer()
            time.sleep(1)

        mse_list = []
        while arm_teacher.get_t() < time_limit:
            t = time.time()
            if arm_teacher.get_t() < time_limit / 2:
                set_torque0(arm_teacher, arm_student, torque1)
            else:
                set_torque0(arm_teacher, arm_student, torque2)
            arm_teacher.advance()
            arm_student.advance()
            if gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            mse_list.append(mse)

        mse = np.array(mse_list).mean()
        mses.append(mse)
        print(f'average mse: {mse}')
        if mse < 0.05:
          score = 0.5
        if mse < 0.015:
          score = 1
        if mse >= 0.05:
          score = 0
        scores.append(score)
        print(f'Score: {score}/{1}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{50}*{5} = {np.array(scores).sum()/50*5:.2f}')
    print("----------------------------------------\n")
    # print(max(mses))
