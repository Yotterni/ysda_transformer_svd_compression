from svd_compression.naive_svd import NaiveSVDApproximator
from svd_compression.activation_aware_svd import ActivationAwareSVDApproximator

from copy import deepcopy

from torch import nn


def apply_recursive_low_ranking_(
        module: nn.Module,
        approximator_class:
        NaiveSVDApproximator | ActivationAwareSVDApproximator,
        *args,
        **kwargs
) -> None:
    """
    Recursively apply Activation Aware SVD approximation
    to the copy of the given module preserving passed model
    instance unchanged.
    
    :param module: `nn.Module` successor to apply approximation to
    :param approximator_class: SVD approximator class, either
    `NaiveSVDApproximator` or `ActivationAwareSVDApproximator`
    :return:
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, approximator_class(child, *args, **kwargs))
        else:
            apply_recursive_low_ranking_(
                child, **kwargs)


def decompose_layers(
        module: nn.Module,
        approximator_class:
        NaiveSVDApproximator | ActivationAwareSVDApproximator,
        *args,
        **kwargs) -> nn.Module:
    """
    A wrapper function for `apply_recursive_low_ranking_`, which is
    used to recursively search for the layers that are to be compressed
    and approximate them with a given strategy.
    
    :param module: `nn.Module` successor to apply approximation to
    :param approximator_class: SVD approximator class, either
    `NaiveSVDApproximator` or `ActivationAwareSVDApproximator`
    :return: a copy of a given module with layers approximated
    according to the given approximator strategy.
    """
    module = deepcopy(module)

    delta_dct = {'initial_num_of_weights': sum(
        param.numel() for param in module.parameters())
    }

    apply_recursive_low_ranking_(module, approximator_class, *args, **kwargs)

    delta_dct['post_svd_num_of_weights'] = sum(
        param.numel() for param in module.parameters())

    delta_dct['after / before'] = (
            delta_dct['post_svd_num_of_weights'] /
            delta_dct['initial_num_of_weights'])

    print(delta_dct)
    return module
