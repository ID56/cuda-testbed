import numpy as np
import numba as nb
from timeit import timeit

@nb.njit(cache=True)
def searchbinned(v):
    """Searchsorted equivalent where x is defined over (-10, 10) with a stride of 0.2"""
    return np.ceil((v + 10) * 5).astype(np.int32)


if __name__ == '__main__':
    # compile
    searchbinned(np.array([0.]))

    a = np.linspace(-10, 10, 101)
    b = np.random.randn(100000)
    
    print('Numpy:', timeit(lambda : np.searchsorted(a, b), number=100))
    print('Nbjit:', timeit(lambda : searchbinned(b), number=100))