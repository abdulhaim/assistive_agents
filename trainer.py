from misc.utils import make_env, format_obs, get_color
import numpy as np

def train_phaseI(args, agent, env, replay_buffer):
    total_episode_count = 0
    obs, ep_reward, ep_len = env.reset(), 0, 0
    task = get_color(obs)
    agent.task = task

    obs = format_obs(obs)
    action = agent.get_action(obs)
    for step in range(args.total_steps_phaseI):
        next_obs, reward, done, info = env.step([action])
        reward = reward[0] # just getting single assistive agent reward
        
        next_obs = format_obs(next_obs)
        
        ep_len += 1
        if reward == 1:
            done = True
        ep_reward += reward

        next_action = agent.get_action(obs)
        replay_buffer.store(obs, action, reward, next_obs, next_action, done)
        obs = next_obs
        action = next_action
        if args.visualize:
            env.render()

        if done or ep_len == env.max_episode_steps:
            agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {}".format(ep_reward, step))

            agent.tb_writer.log_data("episodic_reward", step, ep_reward)
            agent.tb_writer.log_data("episodic_reward_episode", total_episode_count, ep_reward)

            env = make_env(args.seed, "red")
            obs, ep_reward, ep_len = env.reset(), 0, 0
            task = get_color(obs)
            agent.network.task_id = task

            obs = format_obs(obs)
            action = agent.get_action(obs)
            total_episode_count +=1

        # Update handling
        if (step + 1) % args.update_every == 0:
            for j in range(args.update_every):
                batch = replay_buffer.sample_batch(args.batch_size)
                agent.update_loss(batch)

        agent.iteration = step


def train_phaseII(args, agent, replay_buffer):
    for i in range(args.total_steps_phaseII):
        batch = replay_buffer.sample_batch(args.batch_size)
        agent.update_loss(data=batch)

#
# def train_phaseIII(args, agent, human_agent, env, replay_buffer): # TODO: needs work
#     total_episode_count = 0
#     obs, ep_reward, ep_len = env.reset(), 0, 0
#     obs = format_obs(obs)
#     # TODO: Predict Task Num
#     task_id = agent.predict_task()
#     assistive_action = agent.get_action(obs, task_id)
#     human_action = human_agent.get_action(obs, task_id)
#
#     for step in range(args.total_steps):
#         next_obs, reward, done, info = env.step([assistive_action, human_action])
#         reward = reward[0] # just getting single assistive agent reward
#         next_obs = format_obs(next_obs)
#         ep_len += 1
#         ep_reward += reward
#         next_assistive_action = agent.get_action(obs, task_id)
#         next_human_action = human_agent.get_action(obs, task_id)
#         next_task_id = agent.predict_task()
#
#         replay_buffer.store(obs, assistive_action, task_id, reward, human_action,
#                             next_obs, next_assistive_action, next_human_action, next_task_id, done)
#         obs = next_obs
#         assistive_action = next_assistive_action
#         human_action = next_human_action
#
#         if done or ep_len == env.max_episode_steps:
#             agent.log[args.log_name].info("Train Returns: {:.3f} at iteration {}".format(ep_reward, step))
#             agent.tb_writer.log_data("episodic_reward", step, ep_reward)
#             obs, ep_reward, ep_len = env.reset(), 0, 0
#             obs = format_obs(obs)
#             task_id = agent.predict_task()
#             assistive_action = agent.get_action(obs, task_id)
#             human_action = human_agent.get_action(obs, task_id)
#             total_episode_count +=1
#
#         # Update handling
#         if (step + 1) % args.update_every == 0:
#             for j in range(args.update_every):
#                 batch = replay_buffer.sample_batch(args.batch_size)
#                 agent.update_loss(data=batch)
#
#         if not args.phaseI and step % args.converge == 0:
#             task_num+=1
#             env = make_env(args.seed, task=args.tasks[task_num])
