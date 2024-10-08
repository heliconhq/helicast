import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

__all__ = [
    "RegressionDataMaker",
    "make_regression_data_by_r2_target",
]


@dataclass
class RegressionDataMaker:
    """Class to generate regression datasets where X and y are pd.DataFrame. The
    generation of the data is based on the sklearn.datasets.make_regression function.
    There are two ways to generate the data: by noise (see method ``generate_by_noise``)
    and by r2 target (see method ``generate_by_r2_target``).

    After generating the data, the coefficients used to generate the data are stored in
    the attribute ``coef_``.

    Raises:
        n_samples: Number of samples. Defaults to 100.
        n_features: Number of features. Defaults to 10
        n_informative: Number of informative features, i.e., the number of features
            effectively used to generate the target. Defaults to 5.
        n_targets: Number of targets. Defaults to 2
        random_state: Random seed for reproducibility. Defaults to 0.
    """

    n_samples: int = 100
    n_features: int = 10
    n_informative: int = 5
    n_targets: int = 2
    random_state: int = 0

    def _make_sklearn_regression(
        self, noise: float, coef: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        return make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            noise=noise,
            random_state=self.random_state,
            n_targets=self.n_targets,
            coef=coef,
        )

    def _get_r2_for_noise(self, noise: float) -> float:
        X, y = self._make_sklearn_regression(noise)
        return LinearRegression().fit(X, y).score(X, y)

    def _numpy_to_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))

        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y = pd.DataFrame(y, columns=[f"target_{i}" for i in range(y.shape[1])])

        return X, y

    def generate_by_noise(self, noise: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate data with a given noise level. After generating the data, the
        coefficients used to generate the data are stored in the attribute ``coef_``.

        Args:
            noise: Noise level passed to the sklearn.datasets.make_regression function.

        Returns:
            A tuple ``(X, y)`` where ``X`` is a pd.DataFrame with the features and ``y``
            is a pd.DataFrame with the targets.
        """
        X, y, coef = self._make_sklearn_regression(noise, coef=True)
        self.coef_ = coef
        X, y = self._numpy_to_dataframe(X, y)
        return X, y

    def _bissect_noise(self, r2: float, tol: float = 1e-3) -> float:
        if not (0.05 <= r2 <= 0.95):
            raise ValueError("r2 must be between 0.05 and 0.95")
        noise_left = 1e6
        noise_right = 0.0

        for _ in range(100):
            noise_center = 0.5 * (noise_left + noise_right)
            r2_center = self._get_r2_for_noise(noise_center)
            if np.abs(r2_center - r2) < tol:
                break
            if r2_center < r2:
                noise_left = noise_center
            else:
                noise_right = noise_center
        return noise_center

    def generate_by_r2_target(
        self, r2: float, tol: float = 1e-3
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate data with a target r2 value. The dataset is generated by bissecting
        the noise until the r2 value is reached (up to tolerance ``tol``). After
        generating the data, the coefficients used to generate the data are stored in
        the attribute ``coef_``.

        Args:
            r2: Target r2 value. Must be between 0.05 and 0.95
            tol: Tolerance for the r2 value in the bissection. Defaults to 1e-3.

        Returns:
            A tuple ``(X, y)`` where ``X`` is a pd.DataFrame with the features and ``y``
            is a pd.DataFrame with the targets.
        """
        noise = self._bissect_noise(r2, tol)
        X, y, coef = self._make_sklearn_regression(noise, coef=True)
        self.coef_ = coef
        X, y = self._numpy_to_dataframe(X, y)
        return X, y


def make_regression_data_by_r2_target(
    r2: float,
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 5,
    n_targets: int = 2,
    random_state: int = 0,
):
    """Generate regression data with a target r2 value. The dataset is generated by
    bissecting the noise until the r2 value is reached. See the class
    ``RegressionDataMaker`` for more details.

    Args:
        r2: Target r2 value. Must be between 0.05 and 0.95
        n_samples: Number of samples. Defaults to 100.
        n_features: Number of features. Defaults to 10.
        n_informative: Number of informative features. Defaults to 5.
        n_targets: Number of targets. Defaults to 2.
        random_state: Random seed for reproducibility. Defaults to 0.

    Returns:
        A tuple ``(X, y)`` where ``X`` is a pd.DataFrame with the features and ``y``
        is a pd.DataFrame with the targets.
    """
    return RegressionDataMaker(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        random_state=random_state,
    ).generate_by_r2_target(r2)
