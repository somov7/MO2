import sys
import math
import numpy as np

def jordan(D, r, c):
    rows, cols = D.shape
    pivot = D[r, c]
    
    for i in range(rows):
        if i == r:
            continue
        for j in range(cols):
            if j == c:
                continue
            D[i, j] = (D[r, c] * D[i, j] - D[r, j] * D[i, c]) / D[r, c]

    for i in range(cols):
        if i == c:
            continue
        D[r, i] /= pivot

    for i in range(rows):
        if i == r:
            continue
        D[i, c] /= -pivot
    
    D[r, c] = 1 / pivot

    return D

def simplex(D, indep, dep):
    rows, cols = D.shape
    while True:
        ans = [0] * (rows + cols - 2)
        for i in range(len(indep)):
            ans[indep[i]] = D[i, -1]
        ans = np.around(ans, decimals=3)
        print(ans, D[-1, -1].round(decimals=3), sep='\t')   
        pivotCol = D[-1][:-1].argmin()
        if D[-1, pivotCol] >= 0:
            return ans, D[-1, -1]
        help = np.zeros(rows - 1)
        for i in range(rows - 1):
            if(D[i, pivotCol] <= 0):
                help[i] = math.inf
            else:
                help[i] = D[i, pivotCol] * D[i, -1]
        pivotRow = help.argmin()
        if D[-1, pivotRow] == math.inf:
            return False
        indep[pivotRow], dep[pivotCol] = dep[pivotCol], indep[pivotRow] 
        D = jordan(D, pivotRow, pivotCol)

def initial(A, B, C):
    rows = A.shape[0] + 1
    cols = A.shape[1] + 1
    D = np.zeros(rows * cols).reshape(rows, cols)
    D[:-1, :-1] = A
    D[:-1, -1] = B.T
    D[-1:, :-1] = C * -1
    return D

def readfile():
    with open('C:\\Prog\\MO2\\input.txt') as f:
        rows, cols = [int(x) for x in next(f).split()]
        tmp = []
        for i in range(rows):
            line = next(f)
            tmp.append([float(x) for x in line.split()])
        A = np.asarray(tmp)
        tmp.clear() 
        for i in range(rows):
            line = next(f)
            tmp.append([float(x) for x in line.split()])
        B = np.asarray(tmp)
        line = next(f)
        tmp.clear()
        tmp.append(([float(x) for x in line.split()]))
        C = np.asarray(tmp)
    return A, B, C

class precision(float):
    def __repr__(self):
        return "%0.3f" % self

A, B, C = readfile()
rows, cols = A.shape
D = initial(A, B, C)
dep = list(range(cols))
indep = list(range(cols, rows + cols))
ans, value = simplex(D, indep, dep)
print(ans, value)