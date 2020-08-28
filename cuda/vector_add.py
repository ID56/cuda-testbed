from argparse import ArgumentParser
import numpy as np
import numba as nb
import time


@nb.vectorize(['float32(float32, float32)'], target='cuda')
def Add(A, B):
    """Add two vectors A and B of size N and return the result."""
    return A+B


def main(args):
    N = args.N
    A = np.ones( N,  dtype = np.float32)
    B = np.ones( N,  dtype = np.float32)
    C = np.empty(N,  dtype = np.float32)

    # initial compilation
    if args.e: 
        Add(np.float32(0), np.float32(0))

    start = time.time()
    C = Add(A, B)
    end = time.time()
    print(C[:5], C[-5:], 'Completed in {:.4f} s'.format(end-start))


if __name__ == '__main__':
    parser = ArgumentParser('Addition of two vectors in CUDA.')
    parser.add_argument('--N', type=int, default=32000000, help='Size of vector')
    parser.add_argument('--e', dest='e', default=False, action='store_true', help='Turn on early compilation')
    args = parser.parse_args()
    
    main(args)
