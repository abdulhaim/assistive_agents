import numpy as np
from misc.utils import make_env, format_obs

def train_phaseI(args, agent, env, replay_buffer):
    total_episode_count = 0
    obs, ep_reward, ep_len, task_num = env.reset(), 0, 0, 0
    obs = format_obs(obs)
    action = agent.get_action(obs, task_num)

    for step in range(args.total_steps):
        next_obs, reward, done, info = env.step([action])
        reward = reward[0] # just getting single assistive agent reward
        next_obs = format_obs(next_obs)
        ep_len += 1
        ep_reward += reward
        next_action = agent.get_action(obs, task_num)
        replay_buffer.store(obs, action, reward, next_obs, next_action, done)
        obs = next_obs
        action = next_action

        if done or ep_len == env.max_episode_steps:
            agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {}".format(ep_reward, step))
            agent.tb_writer.log_data("episodic_reward", step, ep_reward)
            obs, ep_reward, ep_len = env.reset(), 0, 0
            obs = format_obs(obs)
            action = agent.get_action(obs, task_num)
            total_episode_count +=1

        # Update handling
        if (step + 1) % args.update_every == 0:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                agent.update_loss(batch, task_num)

        if step + 1 % args.converge == 0:
            task_num+=1
            env = make_env(args.seed, task=args.tasks[task_num])

        agent.iteration = step


def train_phaseII(args, agent, human_agent, env, replay_buffer):
    total_episode_count = 0
    obs, ep_reward, ep_len, task_num = env.reset(), 0, 0, 0
    obs = format_obs(obs, task_num, len(args.tasks))

    for step in range(args.total_steps):
        task_id = agent.predict_task(obs)
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
            obs, ep_reward, ep_len = env.reset(), 0, 0
            obs = format_obs(obs)
            total_episode_count +=1

        # Update handling
        if (step + 1) % args.update_every == 0:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                agent.update_loss(data=batch)

        if step % args.converge == 0:
            task_num+=1
            env = make_env(args.seed, task=args.tasks[task_num])
