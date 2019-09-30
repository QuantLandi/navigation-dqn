# import torch
import numpy as np
from collections import deque


def dqn_algorithm(agent, env, brain_name, max_n_episodes=2000, max_n_steps=1000,
                  epsilon_start=1.0, epsilon_min=0.01, epsilon_decay_rate=0.995):
    """Deep Q-Learning Agent.
    
    Parameters
    ----------
        max_n_episodes : int
            Maximum number of training episodes
        max_n_steps : int
            Maximum number of steps per episode
        epsilon_start : float
            Starting value of epsilon, for epsilon-greedy action selection
        epsilon_min : float)
            Minimum value of epsilon
        epsilon_decay_rate : float
            Multiplicative factor (per episode) for decreasing epsilon
    """
    all_scores = []                      # list of scores from all episodes
    last_100_scores = deque(maxlen=100)  # last 100 scores
    epsilon = epsilon_start              # initialize epsilon
    # loop through episodes
    is_game_over = False
    episode_count = 1
    while not is_game_over:
        # observe state and initialize score
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        score = 0
        # loop through steps within each episode
        is_episode_over = False
        agent.t_step = 1
        while not is_episode_over:
            # pick action
            action = agent.act(state, epsilon)
            # observe updated environment, reward and next state
            updated_env = env.step(action)[brain_name]
            next_state = updated_env.vector_observations[0]
            reward = updated_env.rewards[0]
            is_episode_over = updated_env.local_done[0]
            # update next state and add reward from step to episode score
            agent.step(state, action, reward, next_state, is_episode_over)
            state = next_state
            score += reward
            # if episode is over or max_n_steps reached, end loop
            # otherwise, do one more step
            is_episode_over = is_episode_over or (agent.t_step >= max_n_steps)
            agent.t_step += 1
        # anneal epsilon
        epsilon = max(epsilon_min, epsilon_decay_rate*epsilon)
        # keep track of most recent score
        last_100_scores.append(score)
        all_scores.append(score)
        last_100_scores_mean = np.mean(last_100_scores)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_count, last_100_scores_mean), end="")
        completed_100_episodes = episode_count % 100 == 0
        if completed_100_episodes:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_count, last_100_scores_mean))
        is_problem_solved = last_100_scores_mean >= 13.0
        if is_problem_solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'\
                  .format(episode_count, last_100_scores_mean))
            torch.save(agent.qnetwork_local.state_dict(), 'weights.pth')
        # if problem solved or max_n_episodes reached, end loop
        # otherwise, play one more episode
        is_game_over = is_problem_solved or (episode_count >= max_n_episodes)
        episode_count += 1
    return all_scores

