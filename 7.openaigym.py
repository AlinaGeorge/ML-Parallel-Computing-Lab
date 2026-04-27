import gym
import numpy as np
from gym import spaces

# Maze Environment
class MazeEnv(gym.Env):
    def __init__(self):
        self.size = 5
        self.action_space = spaces.Discrete(4)      # up,down,left,right
        self.observation_space = spaces.Discrete(25)
        self.reset()

    def reset(self):
        self.pos = [0, 0]
        self.goal = [4, 4]
        return self.pos[0]*5 + self.pos[1]

    def step(self, a):
        if a==0 and self.pos[0]>0: self.pos[0]-=1
        if a==1 and self.pos[0]<4: self.pos[0]+=1
        if a==2 and self.pos[1]>0: self.pos[1]-=1
        if a==3 and self.pos[1]<4: self.pos[1]+=1

        done = (self.pos == self.goal)
        reward = 1 if done else -0.01
        return self.pos[0]*5 + self.pos[1], reward, done, {}

# Initialize
env = MazeEnv()
Q = np.zeros((25, 4))
alpha, gamma, epsilon = 0.1, 0.9, 0.1

# Training
for _ in range(500):
    s = env.reset()
    done = False
    while not done:
        a = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[s])
        ns, r, done, _ = env.step(a)
        Q[s, a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s, a])
        s = ns

print("Training Done")

# Testing
s = env.reset()
print("Agent Path:")
while True:
    a = np.argmax(Q[s])
    s, _, done, _ = env.step(a)
    print(s)
    if done: break
