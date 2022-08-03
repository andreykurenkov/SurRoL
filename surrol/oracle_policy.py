from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from d3rlpy.constants import ActionSpace
from d3rlpy.algos.base import AlgoBase


class OraclePolicy(AlgoBase):
    r"""Random Policy for continuous control algorithm.
    This is designed for data collection and lightweight interaction tests.
    ``fit`` and ``fit_online`` methods will raise exceptions.
    Args:
        distribution (str): random distribution. The available options are
            ``['uniform', 'normal']``.
        normal_std (float): standard deviation of the normal distribution. This
            is only used when ``distribution='normal'``.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
    """

    _distribution: str
    _normal_std: float
    _action_size: int

    def __init__(
        self,
        env,
        wrapped_env,
        distribution: str = "uniform",
        normal_std: float = 1.0,
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

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._action_size = action_size

    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        return [np.array(self.env.get_oracle_action(self.wrapped_env.convert_obs_to_dict(x)))]

    def sample_action(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        return [np.array(self.env.get_oracle_action(self.wrapped_env.convert_obs_to_dict(x)))]

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
