import numpy as np
import numba as nb
from timeit import timeit
from scipy.signal import convolve2d


@nb.njit(cache=True, fastmath=True)
def njit_convolve2d(a, b, c, i0, i1, j0, j1):
    """Looped 2D Convolution, with the purpose of testing the effectiveness of nopython mode.
    Fastmath should provide an advantage here because of the repeated multiply-add ops."""

    for i in range(i0, i1):
        for j in range(j0, j1):

            for k in range(b.shape[0]):
                for l in range(b.shape[1]):
                    c[i-i0, j-j0] += b[k, l] * a[i+k-i0, j+l-j0]
                    
    return c


@nb.njit(cache=True, fastmath=True)
def njit_bc_convolve2d(a, b, c, i0, i1, j0, j1):
    """Loops replaced with broadcasting, to compare performance."""

    for i in range(i0, i1):
        for j in range(j0, j1):
            c[i-i0, j-j0] = np.sum(b*a[i-i0:i-i0+b.shape[0], j-j0:j-j0+b.shape[1]])
                    
    return c


def convolveWrapper(a, b):
    """Wrapper for njit_convolve2d."""
    px, py = b.shape
    px, py = px//2, py//2

    a_ = np.pad(a, (px,py), constant_values=(0,0))
    b = np.flip(b)
    c = np.zeros_like(a)
    
    c = njit_convolve2d(a_, b, c, px, a_.shape[0]-px, py, a_.shape[1]-py)
    return c

def convolveWrapper_(a, b):
    """Wrapper for njit_bc_convolve2d."""
    px, py = b.shape
    px, py = px//2, py//2

    a_ = np.pad(a, (px,py), constant_values=(0,0))
    b = np.flip(b)
    c = np.zeros_like(a)
    
    c = njit_convolve2d(a_, b, c, px, a_.shape[0]-px, py, a_.shape[1]-py)
    return c


def test_equality(N, M):
    print(f'Testing equality with {N}x{N} input and {M}x{M} kernel...')
    a = np.random.choice(256, (N, N))
    b = np.random.choice(256, (M, M))

    out = convolve2d(a, b, 'same')
    out_l = convolveWrapper(a, b)
    out_bc = convolveWrapper_(a, b)

    print('Equality with NJIT loop:', np.array_equal(out, out_l))
    print('Equality with NJIT broadcast:', np.array_equal(out, out_bc))


def test_runtime(N, M):
    print(f'Testing runtime with {N}x{N} input and {M}x{M} kernel...')
    a = np.random.choice(256, (N, N))
    b = np.random.choice(256, (M, M))

    nl = 1
    print('Scipy convolve2d:', timeit(lambda: convolve2d(a, b, 'same'), number=nl))
    print('NJIT compiled kernel: ', timeit(lambda: convolveWrapper(a, b), number=nl))
    print('NJIT with broadcasting: ', timeit(lambda: convolveWrapper_(a, b), number=nl))
    # roughly 2x gain
    # broadcasted and looped versions are approximately same.


if __name__ == '__main__':
    N = 1000
    M = 11

    assert M % 2 == 1, 'Use odd kernel size.'

    test_equality(N, M)
    test_runtime(N, M)