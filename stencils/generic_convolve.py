import numpy as np
import numba as nb
from timeit import timeit
from scipy.signal import convolve2d

@nb.stencil(standard_indexing=("b",), neighborhood = ((-1, 1),(-1,1)))
def _stcl_kernel(a, b):
    """Stencil kernel for 3x3 2D convolution."""
    c = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            c += a[i,j]*b[i+1,j+1]
    return c

@nb.njit(parallel=True, cache=True, fastmath=True)
def conv3x3(a, b):
    """Wrapper for _stcl_kernel."""
    return _stcl_kernel(a,b)


def test_equality(N):
    print(f'Testing equality with {N}x{N} array.')
    np.random.seed(0)
    a = np.random.choice(256, (N,N))
    b = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]])
    
    # padding is needed to handle border values
    p = 1
    a_ = np.pad(a, (p,p), constant_values=(0,0))
    out = conv3x3(a_, b)[p:-p,p:-p]
    out_ = convolve2d(a, b, 'same')
    print('Equality:', np.array_equal(out, out_))


def test_runtime(N):
    print(f'Testing runtime with {N}x{N} array.')
    np.random.seed(56)
    a = np.random.choice(256, (N,N))
    b = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]])
    
    # padding is needed to handle border values
    p = 1
    a_ = np.pad(a, (p,p), constant_values=(0,0))

    nl = 10
    print('Scipy convolve2d:', timeit(lambda: convolve2d(a, b), number=nl))
    print('JIT compiled stencil: ', timeit(lambda: conv3x3(a_, b), number=nl))
    # roughly 7x gain

if __name__ == '__main__':
    N = 1000
    test_equality(N)
    test_runtime(N)
