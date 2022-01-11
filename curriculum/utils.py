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


class SaveReplayBufferEnv:
    def __enter__(self, replay_buffer):
        self.env = replay_buffer.env
        replay_buffer.env = None

    def __exit__(self, replay_buffer):
        replay_buffer.env = self.env
