from jaxtyping import Float

import torch


class GaussianDistribution:
    def __init__(
        self, mean: Float[torch.Tensor, " dim"], std: Float[torch.Tensor, " dim"]
    ) -> None:
        self.ndim = mean.shape[0]
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int) -> Float[torch.Tensor, "batch dim"]:
        return torch.vstack(
            [torch.normal(mean=self.mean, std=self.std) for _ in range(batch_size)]
        )
