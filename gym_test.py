import gym
env = gym.make('CartPole-v0')
env.reset()
for episode in range(10000):
    env.reset()
    while True:
        env.render()
        o, r, done, _ = env.step(env.action_space.sample()) # take a random action
        print r
        if done:
            print 'The episode is ', episode
            break