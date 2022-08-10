from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from d3rlpy.constants import ActionSpace
from d3rlpy.algos.base import AlgoBase
from imitation.policies.base import HardCodedPolicy
import random


class OraclePolicy(AlgoBase):
    _distribution: str
    _normal_std: float
    _action_size: int

    def __init__(
        self,
        env,
        wrapped_env,
        action_noise = None,
        distribution: str = "uniform",
        normal_std: float = 1.0,
        noise_ratio: float = 0.4,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=1,
            n_frames=1,
            n_steps=1,
            gamma=0.0,
            scaler=None,
            kwargs=kwargs,
        )
        self._distribution = distribution
        self._normal_std = normal_std
        self._action_size = 1
        self._impl = None
        self.env = env
        self.wrapped_env = wrapped_env
        self.action_noise = action_noise
        self.noise_ratio = noise_ratio

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._action_size = action_size

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        return self.sample_action(x)

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        action = np.array(self.env.get_oracle_action(
                          self.wrapped_env.convert_obs_to_dict(x)))
        if self.action_noise != None and random.random() < self.noise_ratio:
            action = action + self.action_noise()
            for i in range(len(action)):
                if action[i] < -1.0:
                    action[i] = -1.0
                if action[i] > 1.0:
                    action[i] = 1.0
        return [action]

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
