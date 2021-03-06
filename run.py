from unityagents import UnityEnvironment
from dqn import dqn_algorithm
from agent import Agent
import time


# initialize environment
filename = "Banana_Windows_x86_64/Banana.exe"
env = UnityEnvironment(file_name=filename)
# initialize agent
agent = Agent(state_size=37, action_size=4, seed=0)
brain_name = env.brain_names[0]

# track start time
start_time = time.time()

# train DQN agent
dqn_algorithm(agent, env, brain_name)

# track end time and print training time
end_time = time.time()
training_time = round((end_time - start_time) / 60, 1)
print('Total training time: {} minutes'.format(training_time))
