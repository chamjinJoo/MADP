import gymnasium as gym

layout = """
.......
...x...
..x.x..
.x...x.
..x.x..
...x...
.g...g.
"""
env = gym.make("rware:rware-tiny-2ag-v2", layout=layout)


obs = env.reset()  # a tuple of observations
done = False
while not done:
    env.render()
    actions = env.action_space.sample()  # the action space can be sampled/
    n_obs, reward, done, truncated, info = env.step(actions)
    print(n_obs)