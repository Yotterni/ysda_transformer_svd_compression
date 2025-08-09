from copy import deepcopy

import torch
from torch import nn


class NaiveSVDApproximator(nn.Module):
    def __init__(
            self,
            layer: nn.Linear,
            threshold: float = 1,
            max_rank: int | None = None
    ) -> None:
        """
        SVD-based approximation for a linear layer.

        In `torch`, `nn.Linear(x)` means $xA^t + b$. We approximate $A$ as $USV^t$,
        where $U$, $S$ and $V$ are matrices from the truncated singular
        value decomposition of the matrix $A$, where the truncation is done
        according to either explained variance threshold or direct setting of
        maximum rank.
        Given an $x$, this layer outputs $xVSU^t + b$.

        The input layer is used just to

        :param layer: a layer to be approximated; note that passed layer
         will remain unchanged
        :param threshold: explained variance cutoff for singular values
        :param max_rank: maximum rank for $U$, $S$ and $V$ matrices
        """
        super().__init__()
        assert 0 <= threshold <= 1, 'Explained variance cutoff must be in [0, 1]'
        self.threshold = threshold
        self.max_rank = max_rank

        U, S, V = torch.linalg.svd(layer.weight.detach())
        V = V.T
        S = S.ravel()

        if max_rank is not None:
            idx = min(max_rank, len(S)) - 1
        else:
            for idx, elem in enumerate(S.cumsum(dim=0) / S.sum()):
                if elem > threshold:
                    break

        U = U[:, : idx + 1]
        V = V[:, : idx + 1]
        S = S[: idx + 1].view(1, -1)

        self.US = nn.Parameter(U * S)
        self.V = nn.Parameter(V)

        self.bias = nn.Parameter(layer.bias)

    def forward(self, x):
        x_mul_v = x @ self.V
        operator_image = x_mul_v @ self.US.T
        return operator_image + self.bias

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(in_features={self.V.shape[0]}, '
                f'out_features={self.US.shape[0]}, '
                f'threshold={self.threshold}), '
                f'max_rank={self.max_rank}')
