import argparse
import yaml
import numpy as np
import pickle
import copy
import os
import torch

from mdp import DiscreteMDP
from dqn import DQN
from split_dqn import SplitDQN
from env import BulletEnv

from clt_core.util.memory import Transition
from clt_core.util.info import Logger

from analyze import analyze_train_results


def run_episode(env, agent, mdp, max_steps, mode='train', seed=0):
    """
    Difference from core: returns the observations for HER
    """

    episode_data = []
    observations = []
    actions = []

    env.seed(seed)
    mdp.seed(seed)

    obs = env.reset()
    while not mdp.init_state_is_valid(obs):
        obs = env.reset()

    observations.append(copy.deepcopy(obs))

    for i in range(max_steps):
        print('-- Step :', i)

        # Transform observation from env (e.g. RGBD, mask) to state representation from MDP (e.g. the latent from an
        #   autoencoder)
        state = mdp.state_representation(obs)

        # Select action
        if mode == 'train':
            action = agent.explore(state)
        else:
            action = agent.predict(state)

        actions.append(action)

        print('action:', action)

        # Transform an action from the agent (e.g. -1, 1) to an env action: (e.g. 2 3D points for a push)
        env_action = mdp.action(obs, action)

        # Step environment
        next_obs = env.step(env_action)

        observations.append(copy.deepcopy(next_obs))

        # Calculate reward from environment state
        reward = mdp.reward(obs, next_obs, action)
        print('reward:', reward)

        # Calculate terminal state
        terminal_id = mdp.terminal(obs, next_obs)

        # Log
        if terminal_id == 1:
            raise RuntimeError('Terminal id = 1 is taken for maximum steps.')

        if -10 < terminal_id <= 0 and i == max_steps - 1:
            terminal_id = 1  # Terminal state 1 means terminal due to maximum steps reached

        timestep_data = {"q_value": agent.q_value(state, action),
                         "reward": reward,
                         "terminal_class": terminal_id,
                         "action": action,
                         "obs": copy.deepcopy([x.dict() for x in obs['full_state']['objects']]),
                         "agent": copy.deepcopy(agent.info)
                         }
        episode_data.append(timestep_data)

        print('terminal state', terminal_id)

        # If the mask is empty, stop the episode
        if terminal_id <= -10:
            break

        if train:
            next_state = mdp.state_representation(next_obs)

            # Agent should use terminal as true/false
            transition = Transition(state, action, reward, next_state, bool(terminal_id))
            agent.learn(transition)

        obs = copy.deepcopy(next_obs)

        if terminal_id > 0:
            break

        print('-----------------')

    return episode_data, observations, actions


def train(args):
    logger = Logger('logs/train-' + args.agent + '-' + args.env)

    with open('yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    params['agent']['log_dir'] = logger.log_dir

    if args.env == 'complex':
        params['env']['scene_generation']['all_equal_height_prob'] = 0.2
        params['env']['scene_generation']['nr_of_obstacles'] = [8, 13]

        params['mdp']['nr_discrete_actions'] = 24
        params['mdp']['nr_primitives'] = 3

        # Add the extra primitive in agent's params
        params['agent']['batch_size'] = [64, 64, 64]
        params['agent']['learning_rate'] = [0.001, 0.001, 0.001]
        params['agent']['hidden_units'] = [[100, 100], [100, 100], [100, 100]]
        params['agent']['loss'] = ['mse', 'mse', 'mse']

    env = BulletEnv(params=params['env'])
    mdp = DiscreteMDP(params)
    agent = SplitDQN(state_dim=263, action_dim=params['mdp']['nr_discrete_actions'],
                     params=params['agent'])
    agent.seed(args.seed)

    if args.env == 'complex' and args.checkpoint is not None:
        agent.network[0].load_state_dict(torch.load('../logs/train_icra_splitdqn_13/model_3000/model_0.pt')['network'])
        agent.network[1].load_state_dict(torch.load('../logs/train_icra_splitdqn_13/model_3000/model_1.pt')['network'])

    rng = np.random.RandomState()
    rng.seed(args.seed)

    train_data = []

    for i in range(args.n_episodes):
        print('--- (Train) Episode {} ---'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Session Seed: ', args.seed, 'Episode seed:', episode_seed)
        episode_data, obs, actions = run_episode(env, agent, mdp, args.episode_max_steps, seed=episode_seed)
        train_data.append(episode_data)

        logger.update()
        logger.log_data(train_data, 'train_data')

        if i % args.save_every == 0:
            agent.save(logger.log_dir, name='model_' + str(i))
            pickle.dump(rng.get_state(), open(os.path.join(logger.log_dir, 'model_' + str(i), 'rng_state.pkl'), 'wb'))


def eval(args):
    logger = Logger('logs/eval_dqn')

    with open('yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    params['agent']['log_dir'] = logger.log_dir

    if args.env == 'complex':
        params['env']['scene_generation']['all_equal_height_prob'] = 0.2
        params['env']['scene_generation']['nr_of_obstacles'] = [8, 13]

        params['mdp']['nr_discrete_actions'] = 24
        params['mdp']['nr_primitives'] = 3

        # Add the extra primitive in agent's params
        params['agent']['batch_size'] = [64, 64, 64]
        params['agent']['learning_rate'] = [0.001, 0.001, 0.001]
        params['agent']['hidden_units'] = [[100, 100], [100, 100], [100, 100]]
        params['agent']['loss'] = ['mse', 'mse', 'mse']

    env = BulletEnv(params=params['env'])
    mdp = DiscreteMDP(params)
    agent = SplitDQN.load(args.checkpoint)
    agent.seed(args.seed)

    rng = np.random.RandomState()
    rng.seed(args.seed)

    eval_data = []
    for i in range(args.test_trials):
        print('---- (Eval) Episode {} ----'.format(i))
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Session Seed: ', args.seed, 'Episode seed:', episode_seed)
        episode_data, _, _ = run_episode(env, agent, mdp, args.episode_max_steps, mode='eval', seed=episode_seed)
        eval_data.append(episode_data)
        print('--------------------')

        logger.update()
        logger.log_data(eval_data, 'eval_data')

    successes, actions = 0, 0
    for episode in eval_data:
        if episode[-1]['terminal_class'] == 2:
            successes += 1
            actions += len(episode)

    if successes > 0:
        print('Success: ', "{:.2f}".format(100 * successes / len(eval_data)), 'Mean actions: ',
              "{:.2f}".format(actions / successes))
        return successes / len(eval_data), actions / successes
    else:
        print('Success: ', "{:.2f}".format(0), 'Mean actions: NaN')
        return 0, 0


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -------------- Setup options --------------
    parser.add_argument('--seed', default=0, type=int, help='Seed that will run the experiment')
    parser.add_argument('--env', default='simple', type=str, help='')
    parser.add_argument('--episode_max_steps', default=10, type=int, help='Maximum number of steps in each episode')

    # -------------- Training options --------------
    parser.add_argument('--exp_name', default='', type=str, help='Name of experiment to run')
    parser.add_argument('--agent', default='split-dqn', type=str, help='Name of the agent')
    parser.add_argument('--n_episodes', default=10000, type=int, help='Number of episodes to run for')
    parser.add_argument('--save_every', default=100, type=int, help='Number of episodes to save the model')

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--checkpoint', default='downloads/simple', type=str,
                        help='The path to the model to load for evaluation')
    parser.add_argument('--test_trials', default=1000, type=int, help='Number of episodes to evaluate for')

    # parser.add_argument('--eval_all', default=False, type=bool, help='Evaluate all models')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.is_testing:
        eval(args)
    else:
        train(args)

    # analyze_train_results(dir='logs/train-split-dqn-complex')