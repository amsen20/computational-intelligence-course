import math

def zeros(A):
    ret = []
    for _ in A:
        ret.append(0)
    return tuple(ret)

def cord(A):
    ret = 0
    for i in A:
        ret += i*i
    # return math.sqrt(ret)
    return ret

def mul(A, c):
    ret = []
    for i in A:
        ret.append(i*c)
    return tuple(ret)

def add(A, B):
    ret = []
    for (i, j) in zip(A, B):
        ret.append(i + j)
    return tuple(ret)

def sub(A, B):
    return add(A, mul(B, -1))
