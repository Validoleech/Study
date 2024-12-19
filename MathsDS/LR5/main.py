import numpy as np

A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 0, -1],
    [-1, 0, 0, 4, -1, 0],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
])

b = np.array([0, 5, 0, 6, -2, 6])

y = np.zeros(6)

e = 1e-6

max_iter = 1000

for k in range(max_iter):
    grad = 2 * A.dot(y) - 2 * b
    if np.linalg.norm(grad) < e:
        break

    alpha = grad.dot(grad) / (2 * grad.dot(A.dot(grad)))
    y = y - alpha * grad

X_ast = y

print(f'||X*||_2 = {np.linalg.norm(X_ast):.3f}')
