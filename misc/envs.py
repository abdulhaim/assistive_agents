import copy
import numpy as np
from typing import Callable, NamedTuple

class MDP(NamedTuple):
  """Container for holding the parametrisation of Markov decision process."""
  rewards: np.ndarray  # [S, A]
  transition_probs: np.ndarray  # [S, A, S]

# A function that converts any state function to a cell function.
StateToCellFn = Callable[[np.ndarray], np.ndarray]

def gridworld_env_to_tabular(env):
  """Convert a sample-able `env` to a tabular MDP."""
  num_states = env.reset()
  num_actions = env.action_space

  def step(s_tm1, a_tm1):
    """Perform a step in the environment, assuming that `env` is deterministic."""
    cloned_env = copy.deepcopy(env)
    cloned_env._timeout = int(1e6)
    # Set agent's position at `s_tm1`.
    pos_tm1 = cloned_env._state_to_cell[s_tm1]
    # If spawned on a goal, treat this as a draining state.
    spawned_on_goal = False
    for _, goal_pos in cloned_env._goals_pos.items():
      if (pos_tm1 == goal_pos).all():
        r_t = 0.0
        s_t = s_tm1
        spawned_on_goal = True
        break
    if not spawned_on_goal:
      cloned_env._agent_pos = cloned_env._state_to_cell[s_tm1]
      time_step = cloned_env.step(a_tm1)
      s_t = time_step.observation['tabular']
      r_t = time_step.reward
    return s_t, r_t

  # Containers for holding the MDP representation.
  rewards = np.zeros(shape=(num_states, num_actions), dtype=np.float32)
  transition_probs = np.zeros(
      shape=(num_states, num_actions, num_states), dtype=np.float32)

  for s_tm1 in range(num_states):
    for a_tm1 in range(num_actions):
      s_t, r_t = step(s_tm1, a_tm1)
      transition_probs[s_tm1, a_tm1, s_t] = 1.0
      rewards[s_tm1, a_tm1] = r_t

  # Make sure that there is no pointer leakage.
  state_to_cell = copy.deepcopy(env._state_to_cell)

  def state_to_cell_fn(state_fn: np.ndarray) -> np.ndarray:
    """A generic function that converts any state function to cell function."""
    cell_fn = np.zeros_like(env._board)
    for s in range(num_states):
      cell_fn[state_to_cell[s]] = state_fn[s]
    return cell_fn

  return MDP(rewards, transition_probs), state_to_cell_fn