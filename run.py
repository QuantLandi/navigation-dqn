from dqn import dqn_algorithm
from unityagents import UnityEnvironment
from agent import Agent
import time
import matplotlib.pyplot as plt

# filename = "C:/Users/aless/Google Drive/Machine Learning/udacity/deep-rl/navigation_dqn/Banana_Windows_x86_64/Banana.exe"
filename = "Banana_Windows_x86_64/Banana.exe"
env = UnityEnvironment(file_name=filename)
# initialize agent
agent = Agent(state_size=37, action_size=4, seed=0)
brain_name = env.brain_names[0]

# track start time
start_time = time.time()

# train DQN agent
scores = dqn_algorithm(agent, env, brain_name)

# track end time and print training time
end_time = time.time()
training_time = round((end_time - start_time) / 60, 1)
print('Total training time: {} minutes'.format(training_time))

# plot performance
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)
# plot scores for all episodes
scores_idx = np.arange(len(scores))
plt.plot(scores_idx, scores)
# plot moving average of last 100 scores for smoother plotting
scores_moving_avg = pd.Series(scores).rolling(100).mean()
plt.plot(scores_idx, scores_moving_avg)
plt.title('DQN performance over number of episodes', size=14)
# plt.title('DQN  Performance Over Number of Episodes')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()