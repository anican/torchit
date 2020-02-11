import torch


def num_flat_features(x: torch.Tensor) -> int:
    r"""
    Returns the number of features in a convolution layer.
    Args:
        x (Tensor) : input tensor
    Examples:
        >>> x = torch.randn(5, 3, 7, 7)
        >>> num_flat_features(x)
        147
    """
    num_features = 1
    for dim in x.size()[1:]:
        num_features *= dim
    return num_features

def get_dataloaders():
    pass

