import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

def make_atari_env(env_id, frame_stack=4):
    """Atari 환경을 생성하고 표준 전처리를 적용합니다."""
    env = gym.make(env_id)
    # 84x84 resize, grayscale, frame skipping(4), noop reset
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84)
    # Stack 4 frames
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    return env
