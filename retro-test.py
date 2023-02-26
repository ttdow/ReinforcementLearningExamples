import retro

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

obs = env.reset()

while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    print(rew)
    env.render()

    if done:
        obs = env.reset()

env.close()