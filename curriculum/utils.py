import gym

def remove_all_nmprepr_gym_envs():
  env_dict = list(gym.envs.registration.registry.env_specs.copy())
  flag = False
  for env in env_dict:
      if flag:
          print("Remove {} from registry".format(env))
          del gym.envs.registration.registry.env_specs[env]
      if env == 'Zaxxon-ramNoFrameskip-v4':
          flag = True