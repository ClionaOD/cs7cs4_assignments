import numpy as np

def convolve(arr, kernel, padding=0, strides=1):
    """
    Takes an n*n array and convolves it with the k*k kernel

    args:
        arr: the n*n numpy array
        kernel: the k*k numpy array
    
    returns:
        conv: the convolved matrix
    """
    sub_size = kernel.shape
    return