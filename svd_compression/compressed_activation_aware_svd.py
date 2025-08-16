import torch
import torch.nn as nn


class TruncatedAdaptiveSVDLinear(nn.Module):
    def __init__(self, rank: int, weight: torch.Tensor, bias: torch.Tensor,
                 precondtioner: torch.Tensor, inv_preconditioner: torch.Tensor,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        U, Sigma, V = torch.linalg.svd(weight @ precondtioner, full_matrices=False)
        max_rank = len(Sigma)
        SigmaDiag = torch.diag(Sigma[:min(rank, max_rank)])
        U = U[:, :min(rank, max_rank)] @ SigmaDiag
        V = V[:min(rank, max_rank), :] @ inv_preconditioner

        self.U = nn.Linear(in_features=min(rank, max_rank),
                           out_features=weight.shape[0], bias=True)
        self.V = nn.Linear(in_features=weight.shape[1], out_features=min(
            rank, max_rank), bias=False)
        self.U.weight.data.copy_(U)
        self.U.bias.data.copy_(bias)
        self.V.weight.data.copy_(V)
        self.U = self.U.to(device)
        self.V = self.V.to(device)

    def forward(self, input):
        return self.U(self.V(input))

    def __repr__(self):
        return f'(\n\t(U): {self.U.__repr__()}\n\t(V): {self.V.__repr__()}\n)'
