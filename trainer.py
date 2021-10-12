import numpy as np
import torch
from misc.utils import make_env, format_obs
def train_phaseI(args, agent, env, replay_buffer):
    total_episode_count = 0
    obs, ep_reward, ep_len, task_num = env.reset(), 0, 0, 0
    obs = format_obs(obs, task_num, len(args.tasks))

    for step in range(args.total_steps):
        action = agent.get_action(obs, task_num)
        next_obs, reward, done, info = env.step(action)

        discount = args.discount**ep_len
        reward = reward[0]
        next_obs = np.array(next_obs['image'])
        next_obs_shape = torch.tensor(next_obs).shape
        next_obs = torch.reshape(torch.tensor(next_obs), (next_obs_shape[0], next_obs_shape[3], next_obs_shape[1], next_obs_shape[2])).double()

        ep_len += 1
        ep_reward += reward
        next_action = agent.get_action(next_obs, task_num)
        replay_buffer.store(obs, action, reward, next_obs, next_action, discount, done)

        obs = next_obs

        if done or ep_len == env.max_episode_steps:
            agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {}".format(ep_reward, step))
            agent.tb_writer.log_data("episodic_reward", step, ep_reward)
            obs, ep_reward, ep_len = np.array(env.reset()['image']), 0, 0
            obs_shape = torch.tensor(obs).shape
            obs = torch.reshape(torch.tensor(obs), (obs_shape[0], obs_shape[3], obs_shape[1], obs_shape[2])).double()
            total_episode_count +=1

        # Update handling
        if (step + 1) % args.update_every == 0:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                agent.update_loss(batch, task_num)

        if step + 1 % args.converge == 0:
            task_num+=1
            env = make_env(args.seed, task=args.tasks[task_num])


def train_phaseII(args, agent, human_agent, env, replay_buffer):
    total_episode_count = 0
    obs, ep_reward, ep_len, task_num = np.array(env.reset()['image']), 0, 0, 0
    for step in range(args.total_steps):
        # TODO: Determine Task ID
        task_id = agent.get_task(obs)
        action = agent.get_action(obs, task_id)
        next_obs, reward, done, info = env.step(action)
        next_obs = np.array(next_obs['image'])
        ep_len += 1
        ep_reward += reward
        replay_buffer.store(obs, action, reward, next_obs, done)
        if done or ep_len == env.max_episode_steps:
            obs = env.reset()
            agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {}".format(ep_reward, step))
            agent.tb_writer.log_data("episodic_reward", step, ep_reward)
            obs, ep_reward, ep_len = np.array(env.reset()['image']), 0, 0
            total_episode_count +=1

        # Update handling
        if (step + 1) % args.update_every == 0:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                agent.update_loss(data=batch)

        if step % args.converge == 0:
            task_num+=1
            env = make_env(args.seed, task=args.tasks[task_num])



