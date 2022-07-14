import math
import jax
import jax.numpy as jnp
import torch


class MultivariateNormalDiag:
  """Multivariate normal distribution on `R^k`."""

  def __init__(
      self,
      loc: jnp.ndarray,
      scale_diag: jnp.ndarray):
    """Initializes a MultivariateNormalDiag distribution.
    Args:
      loc: Mean vector of the distribution. Can also be a batch of vectors.
      scale_diag: Vector of standard deviations.
    """
    super().__init__()
    self._loc = loc
    self._scale_diag = scale_diag

  @property
  def loc(self) -> jnp.ndarray:
    """Mean of the distribution."""
    return self._loc

  @property
  def scale_diag(self) -> jnp.ndarray:
    """Scale of the distribution."""
    return self._scale_diag

  def _num_dims(self) -> int:
    """Dimensionality of the events."""
    return self._scale_diag.shape[-1]

  def _standardize(self, value: jnp.ndarray) -> jnp.ndarray:
    return (value - self._loc) / self._scale_diag

  def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
    """See `Distribution.log_prob`."""
    log_unnormalized = -0.5 * jnp.square(self._standardize(value))
    log_normalization = 0.5 * math.log(2 * math.pi) + jnp.log(self._scale_diag)
    return jnp.sum(log_unnormalized - log_normalization, axis=-1)

  def mean(self) -> jnp.ndarray:
    """Calculates the mean."""
    return self.loc

  def sample(self, seed: jnp.ndarray) -> jnp.ndarray:
    """Samples an event.
    Args:
      seed: PRNG key or integer seed.
    Returns:
      A sample.
    """
    # eps = jax.random.normal(seed, self.loc.shape)
    eps = torch.randn(self.loc.shape).to('cuda')
    return self.loc + eps * self.scale_diag
    # return self.loc + self.scale_diag