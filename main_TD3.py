"""
Function:
This is the training code for TD3 without LSTM
train the DRL model and preserve the model weights
"""
import argparse
import numpy as np
import random
import os
import torch
import json

from algos import TD3
from utils import memory
from info import *

# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, time_step=0, save_directory=None,eval_config:dict=None):
    policy.eval_mode()
    avg_reward = 0.
    success_times = []
    success = 0
    collision = 0
    timeout = 0
    eval_episodes = len(eval_config.values())
    for idx, config in enumerate(eval_config.values()):
        lidar_state, position_state = eval_env.reset_with_eval_config(config)
        done = False
        ep_step = 0
        ep_reward = 0
        while not done:
            with torch.no_grad():
                action = policy.select_action(lidar_state, position_state)
            lidar_state, position_state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            ep_step = ep_step + 1
            ep_reward += reward
        if isinstance(info, ReachGoal):
            success += 1
            print('Episode: ' + str(idx) + ', Obstacles: ' + str(eval_env.ship_num) + ', Reaching: ' + str(ep_step))
        elif isinstance(info, Collision):
            collision += 1
            print('Episode: ' + str(idx) + ', Obstacles: ' + str(eval_env.ship_num) + ', Collision: ' + str(ep_step))
        elif isinstance(info, Timeout):
            timeout += 1
            print('Episode: ' + str(idx) + ', Obstacles: ' + str(eval_env.ship_num) + ', Time-out: ' + str(ep_step))
        elif isinstance(info, Outside):
            collision += 1
            print('Episode: ' + str(idx) + ', Obstacles: ' + str(eval_env.ship_num) + ', Outside: ' + str(ep_step))
        else:
            raise ValueError('Invalid end signal from environment')

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_env.time_limit
    avg_return = avg_reward / eval_episodes
    policy.train_mode()
    
    return success_rate, collision_rate, avg_nav_time, avg_return

def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def create_eval_configs(eval_env):
    eval_config = {}
    num_episodes = [20,20,20,20]
    num_os = [5, 6, 7, 8]
    count = 0
    for i,num_episode in enumerate(num_episodes):
        for _ in range(num_episode):
            eval_env.ship_num = num_os[i]
            eval_env.reset()
            # save eval config
            eval_config[f"env_{count}"] = eval_env.episode_data()
            count += 1
    return eval_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="TD3")
    # device
    parser.add_argument("--device", type=str, default='cuda:0')
    # OpenAI gym environment name
    parser.add_argument("--env", default="marine_simulation")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=32, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=1e4, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=2e4, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.15)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=96, type=int)
    # Memory size
    parser.add_argument("--memory_size", default=1e5, type=int)
    # Learning rate
    parser.add_argument("--lr", default=3e-4, type=float)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.25)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Model width
    parser.add_argument("--hidden_size", default=512, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", default=True, action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", type=str, default="")

    # Don't train and just run the model
    parser.add_argument("--test", action="store_true", default=False)
    # environment settings
    parser.add_argument("--only_dynamic", action="store_true", default=True)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--lidar_dim", type=int, default=1800)
    parser.add_argument("--lidar_feature_dim", type=int, default=50)
    
    parser.add_argument("--goal_position_dim", type=int, default=2)
    parser.add_argument("--laser_angle_resolute", type=float, default=0.003490659)
    parser.add_argument("--laser_min_range", type=float, default=2.5)
    parser.add_argument("--laser_max_range", type=float, default=100.0)
    parser.add_argument("--square_width", type=float, default=1000.0)
    parser.add_argument("--discomfort_distance", type=float, default=30)
    parser.add_argument("--classical", default=False)
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    from marine_simulation import CrowdSim

    file_prefix = './logdir/' + args.policy + str(args.discomfort_distance) + '/' + args.env + '/seed_' + str(args.seed)   # choose your own log directory

    if not os.path.exists(file_prefix + '/results'):
        os.makedirs(file_prefix + '/results')

    if args.save_model and not os.path.exists(file_prefix + '/models'):
        os.makedirs(file_prefix + '/models')

    if not os.path.exists(file_prefix + '/evaluation_episodes'):
        os.makedirs(file_prefix + '/evaluation_episodes')

    if not os.path.exists(file_prefix + '/final_test'):
        os.makedirs(file_prefix + '/final_test')

    training_schedule = dict(timesteps=[0, 200000, 400000, 600000],
                             num_obstacles=[5, 6, 7, 8])

    env = CrowdSim(args, schedule=training_schedule)
    eval_env = CrowdSim(args, schedule=training_schedule, e_mode=True)
    eval_config = create_eval_configs(eval_env)

    eval_config_file = os.path.join(file_prefix + '/evaluation_episodes', "eval_config.json")
    with open(eval_config_file, "w+") as f:
        json.dump(eval_config, f, default=default_dump)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    lidar_state_dim = args.lidar_dim
    # default is 2, please change it to 4 when using action as observation
    if args.env == 'crowd_real_all_circles':
        position_state_dim = args.goal_position_dim * 2
    else:
        position_state_dim = args.goal_position_dim
    
    lidar_feature_dim = args.lidar_feature_dim
    action_dim = args.action_dim
    max_action = 1.0

    kwargs = {
        "lidar_state_dim": lidar_state_dim,
        "position_state_dim": position_state_dim,
        "lidar_feature_dim": lidar_feature_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": args.hidden_size,
        "discount": args.discount,
        "tau": args.tau,
    }
    kwargs["device"] = args.device

    policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy.load(file_prefix + args.load_model)
        print("Load trained model.")

    if args.test:
        success_rate, collision_rate, avg_nav_time, avg_return = eval_policy(policy, eval_env, eval_episodes=100, save_directory=file_prefix + '/evaluation_episodes')
        print('Test the trained model.')
        print('success_rate, collision_rate, avg_nav_time, avg_return')
        print(success_rate, collision_rate, avg_nav_time, '%.3f' % avg_return)

    replay_buffer = memory.ReplayBuffer(
        lidar_state_dim, position_state_dim, action_dim, args.memory_size, args.device)

    evaluations = []

    lidar_state, position_state = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    for t in range(1, int(args.max_timesteps) + 1):

        episode_timesteps += 1
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = np.random.uniform(-max_action, max_action, action_dim)
            with torch.no_grad():
                _ = policy.select_action(lidar_state, position_state)
        else:
            with torch.no_grad():
                a = policy.select_action(lidar_state, position_state)
            action = (
                a + np.random.normal(
                    0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
        if t == args.start_timesteps:
            print('replay buffer has been initialized')
        # Perform action
        next_lidar_state, next_position_state, reward, done, info = env.step(action)

        if isinstance(info, Timeout):
            done_bool = 0.0
        else:
            done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(
            lidar_state, position_state, action, next_lidar_state, next_position_state, reward, done_bool)

        lidar_state = next_lidar_state
        position_state = next_position_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if (not policy.on_policy) and t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            if isinstance(info, ReachGoal):
                print('Steps ' + str(t) + ', Episodes ' + str(episode_num+1) + ', Obstacles ' + str(env.ship_num) + ', Reaching ' + str(episode_timesteps) + ', Return ' + '%.2f' % episode_reward)
            elif isinstance(info, Collision):
                print('Steps ' + str(t) + ', Episodes ' + str(episode_num+1) + ', Obstacles ' + str(env.ship_num) + ', Collision ' + str(episode_timesteps) + ', Return ' + '%.2f' % episode_reward)
            elif isinstance(info, Timeout):
                print('Steps ' + str(t) + ', Episodes ' + str(episode_num+1) + ', Obstacles ' + str(env.ship_num) + ', Time-out ' + str(episode_timesteps) + ', Return ' + '%.2f' % episode_reward)
            elif isinstance(info, Outside):
                print('Steps ' + str(t) + ', Episodes ' + str(episode_num+1) + ', Obstacles ' + str(env.ship_num) + ', Outside ' + str(episode_timesteps) + ', Return ' + '%.2f' % episode_reward)
            else:
                raise ValueError('Invalid end signal from environment')
            # Reset environment
            lidar_state, position_state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t > 9000 and t % args.eval_freq == 0) or (t > 1e4 and t < 2e4 and t % (args.eval_freq/8) == 0):
            success_rate, collision_rate, avg_nav_time, avg_return = eval_policy(policy, eval_env, time_step=t, save_directory=file_prefix + '/evaluation_episodes',eval_config=eval_config)
            file_name = '/step_' + str(t) + '_success_' + str(int(success_rate * 100))
            print('success_rate, collision_rate, avg_nav_time, avg_return at step ' + str(t))
            print(success_rate, collision_rate, avg_nav_time, '%.3f' % avg_return)
            evaluations.append(np.array([success_rate, avg_return]))
            policy.save(file_prefix + '/models' + file_name)
            if args.save_model and t > 800000 and success_rate > 0.90:
                policy.save(file_prefix + '/models' + file_name)
            np.save(file_prefix + '/results' + file_name, evaluations)
