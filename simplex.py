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
        ans = np.zeros(rows + cols - 2)
        for i in range(len(indep)):
            ans[indep[i]] = D[i, -1]
        print(ans, D[-1, -1])   
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
    D[:-1, -1] = B
    D[-1:, :-1] = C * -1
    print(D)
    return D


A = np.array(([1.0, -2.0], [-2.0, 1.0], [2.0, 1.0]))
B = np.array([1.0, 2.0, 6.0]).T
C = np.array([3.0, 1.0])
#D = np.array(([1.0, -2.0, 1.0], [-2.0, 1.0, 2.0], [2.0, 1.0, 6.0], [-3.0, -1.0, 0.0]))
D = initial(A, B, C)
indep = [2, 3, 4]
dep = [0, 1]
ans, value = simplex(D, indep, dep)
print(ans, value)