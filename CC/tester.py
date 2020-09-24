import ahead_compile
import numpy as np
from timeit import timeit

def test_equality(N):
    print(f'Testing equality with 1D array of length {N}')
    a = np.random.choice(N, N).astype(np.float32)
    out = a.mean()
    out_ = ahead_compile.average(a)
    
    print('Equality:', out == out_)

def test_runtime(N):
    print(f'Testing runtime with 1D array of length {N}')

    a = np.random.choice(N, N).astype(np.float32)
    nl = 1

    t1 = timeit(lambda: np.mean(a), number=nl)
    print('Normal mean:', t1)

    t2 = timeit(lambda: ahead_compile.average(a), number=nl)
    print('CC compiled average:', t2)
    
    print('Gain:', t1/t2)
    # Roughly 10x gain at N=1000
    # Gain increases to 14x at N=100 and reduces to 2.5x at N=10000

if __name__ == '__main__':
    N = 1000

    test_equality(N)
    test_runtime(N)