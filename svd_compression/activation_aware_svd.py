from copy import deepcopy

import torch
from torch import nn


class ActivationAwareSVDApproximator(nn.Module):
    def __init__(self, layer: nn.Linear, threshold: float | int = 0.95, batch_size: int | None = None, method: str = "default"):
        super().__init__()
        self.weights = layer.weight.detach()
        self.bias = layer.bias.detach()
        self.method = method
        self.threshold = threshold
        self.batch_size = batch_size
        self.collected_batch_size = 0
        self.activations = []
        self.stop_data_collection = False
        self.device = 'cuda:0'

    def to(self, device: torch.device | str) -> 'ActivationAwareSVDApproximator':
        self.weights = self.weights.to(device)
        self.device = device
        super().to(self.device)
        return self

    def add_activation(self, activations: torch.Tensor) -> bool:
        if self.stop_data_collection:
            return True
        self.activations.append(activations.detach())
        self.collected_batch_size += activations.shape[0]
        if self.batch_size is None or self.collected_batch_size >= self.batch_size:
            self.fit_decomposition()
            return True
        return False

    def fit_decomposition(self):
        activations_batch = torch.cat(self.activations, dim=0)
        self.activations = []
        self.batch_size = 0
        self.stop_data_collection = True

        if self.method == "default":
            S = torch.linalg.cholesky(torch.matmul(activations_batch.T, activations_batch), upper=False).to(self.device)
        elif self.method == "article":
            S = torch.diag_embed(torch.pow(torch.abs(activations_batch).mean(dim=0), 0.5)).to(self.device)
        elif self.method == "variance":
            S = torch.diag_embed(torch.std(activations_batch, dim=0)).to(self.device)
        else:
            raise ValueError("Unknown method")

        WS = self.weights @ S
        U, Sigma, V = torch.linalg.svd(WS)
        V = V.T

        U = U.to(self.device)
        Sigma = Sigma.to(self.device)
        V = V.to(self.device)

        if isinstance(self.threshold, int):
            U = U[:, :min(self.threshold, len(Sigma))]
            Sigma = Sigma[:min(self.threshold, len(Sigma))].reshape(-1)
            V = U[:min(self.threshold, len(Sigma)), :]
        else:
            for idx, elem in enumerate(Sigma.cumsum(dim=0) / Sigma.sum()):
                if elem > self.threshold:
                    break

            U = U[:, :idx + 1]
            Sigma = Sigma[:idx + 1].reshape(-1)
            V = V[:idx + 1, :]

        Sigma = torch.diag_embed(Sigma)

        self.USigma = nn.Parameter(torch.matmul(U, Sigma))
        self.V = nn.Parameter(torch.matmul(V, torch.linalg.inv(S)))
        self.Bias = nn.Parameter(self.bias.reshape(-1, 1))

    def forward(self, input):
        if self.add_activation(input):
            return self.USigma @ (self.V @ input.T) + self.Bias
        return torch.matmul(self.weights, input.T) + self.bias

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(in_features={self.weights.shape[0]}, '
                f'out_features={self.weights.shape[1]}, '
                f'threshold={self.threshold}), '
                f'batch_size={self.batch_size}, '
                f'method={self.method})')


def activation_aware_loralize(
        module: nn.Module,
        threshold: int | float = 0.95,
        batch_size: int | None = None,
        method: str = "default") -> nn.Module:
    """
    Recursively apply Activation Aware LoRa approximation to given module.
    :param module: any model that contains nn.Linear to apply SVD on
    :return:
    """
    module = deepcopy(module)

    delta_dct = {'initial_num_of_weights': sum(
        param.numel() for param in module.parameters())
    }

    def recursively_activation_aware_loralize(
        module: nn.Module,
        threshold: int | float = 0.95,
        batch_size: int | None = None,
        method: str = "default"):

        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, ActivationAwareSVDApproximator(child, threshold, batch_size, method))
            else:
                recursively_activation_aware_loralize(
                    child, threshold, batch_size, method)

    recursively_activation_aware_loralize(module, threshold, batch_size, method)

    delta_dct['post_svd_num_of_weights'] = sum(
        param.numel() for param in module.parameters())

    delta_dct['after / before'] = (
            delta_dct['post_svd_num_of_weights'] /
            delta_dct['initial_num_of_weights'])

    print(delta_dct)
    return module
