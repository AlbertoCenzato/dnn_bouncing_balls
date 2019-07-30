from functools import singledispatch

import torch as th
import numpy as np


@singledispatch
def swapaxes(data, dim0, dim1):
    raise TypeError('swapaxes function does not support this data type')


@swapaxes.register(np.ndarray)
def _(array, dim0, dim1):
    return np.swapaxes(array, dim0, dim1)


@swapaxes.register(th.Tensor)
def _(tensor, dim0, dim1):
    return th.transpose(tensor, dim0, dim1)


@singledispatch
def get_dim(data, axis):
    raise TypeError('get_dim function does not support this data type')


@get_dim.register(th.Tensor)
def _(tensor, axis):
    return tensor.size(axis)


@get_dim.register(np.ndarray)
def _(array, axis):
    return array.shape[axis]


# NOTE: currently I've not found a way to single dispatch on
#       composed types such as List[th.Tensor] or List[np.ndarray]
#       not even creating a new type with
#       typing.NewType('TensorList', typing.List[torch.Tensor])
def stack(tensor_list, axis=0):
    """
    This function is the same as torch.stack but handles both
    numpy.ndarray and torch.Tensor
    :param tensor_list:
    :param axis:
    :return:
    """
    if isinstance(tensor_list[0], th.Tensor):
        return th.stack(tensor_list, axis)
    else:
        return np.stack(tensor_list, axis)


class RepeatAlongAxis:
    """
    Applies the given transform to the input tensor iterating
    along the specified dimension. Can handle both numpy.ndarray
    and torch.Tensor.
    'transform' must be a callable object or a function with
    the following signature:
        transform(tensor)
    """

    def __init__(self, transform, axis=0):
        self.transform = transform
        self.axis = axis

    def __call__(self, tensor):
        transposed = swapaxes(tensor, 0, self.axis)
        resulting_tensors = []
        for i in range(get_dim(transposed, 0)):
            transformed = self.transform(transposed[i,:])
            resulting_tensors.append(transformed)
        stacked = stack(resulting_tensors, 0)
        return swapaxes(stacked, 0, self.axis)


class CutSequence:
    """
    Cuts a slice from 'begin' to 'end' along the first dimension of
    a tensor. Accepts both numpy.ndarray and torch.Tensor
    """

    def __init__(self, begin, end):
        self.begin = begin
        self.end   = end

    def __call__(self, tensor):
        return tensor[self.begin:self.end, :]


class Binarize:
    """
    Binarize an input tensor setting to 0 all elements <= threshold
    and to 1 all elements > threshold
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, tensor):
        thresholded_tensor = tensor > self.threshold
        return thresholded_tensor.float()
