import numba as nb
from numba.pycc import CC

cc = CC('ahead_compile')

@nb.njit  # refer to https://github.com/numba/numba/issues/3632
@cc.export('array_sum', 'f4(f4[:])')
def array_sum(a):
    s = 0
    for i in a:
        s += i
    return s

@cc.export('average', 'f4(f4[:])')
def average(a):
    return array_sum(a)/len(a)

if __name__ == '__main__':
    cc.compile()