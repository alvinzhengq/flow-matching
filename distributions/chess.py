from jaxtyping import Float

import torch


class ChessDistribution:
    def __init__(self, num_squares: int = 4, num_range: float = 4.0) -> None:
        """
        Creates a 2D distribution that looks like a chess board.
        Distribution will be uniform across all light square regions.

        Args:
            num_squares: Number of squares in each row/col (default 4x4 board)
            num_range: Range [-num_range/2, num_range/2] of the 1D sample space.
        """
        self.ndim = 2
        self.num_squares = num_squares
        self.square_size = num_range / float(num_squares)
        self.range_min = -num_range / 2

    def sample(self, batch_size: int) -> Float[torch.Tensor, "batch dim"]:
        weights = torch.zeros(self.num_squares**2)
        for x in range(self.num_squares):
            for y in range(self.num_squares):
                if (x % 2) == (y % 2):
                    continue

                weights[x * self.num_squares + y] = 1.0

        idx = torch.multinomial(weights, batch_size, replacement=True)
        samples = torch.empty(size=(batch_size, 2))

        for i in range(batch_size):
            x = idx[i] % self.num_squares
            y = idx[i] // self.num_squares

            coordinate = self.square_size * torch.rand(2)
            coordinate[0] += self.range_min + x * self.square_size
            coordinate[1] += self.range_min + y * self.square_size

            samples[i] = coordinate
        
        samples = samples[torch.randperm(batch_size)]
        return samples


class ChessDataset:
    def __init__(self, size: int, batch_size: int, **kwargs):
        self.distribution = ChessDistribution(**kwargs)
        self.data = self.distribution.sample(size)
        self.batch_size = batch_size
        self.size = size

    def get_batch(self, idx):
        return self.data[idx*self.batch_size:(idx+1)*self.batch_size]
