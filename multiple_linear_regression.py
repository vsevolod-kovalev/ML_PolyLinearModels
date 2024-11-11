import numpy as np


def MSE(w, b, features, targets):
    m = features.shape[0]
    error_sum = 0
    for i in range(m):
        error = ((np.dot(w, features[i]) + b) - targets[i])**2
        error_sum += error
    cost = error_sum / (2*m)
    return cost
def gradient_derivative_w(features, targets, w, b, i):
    m = targets.shape[0]
    error_sum_w = 0
    for k in range(m):
        error_w = ((np.dot(w, features[k])+b - targets[k]))*features[k][i]
        error_sum_w += error_w
    w_d = error_sum_w / m
    return w_d
def gradient_derivative_b(features, targets, w, b):
    m = targets.shape[0]
    error_sum = 0
    for k in range(m):
        error = ((np.dot(w, features[k])+b - targets[k]))
        error_sum += error
    b_d = error_sum / m
    return b_d
    
def gradient_descent(features, targets, w, b, iters=1000, a=0.0000005):
    j = w.shape[0]
    for iter in range(iters):
        if (iter % 100 == 0):
            print("Iteration: ", iter, " Cost: ", MSE(w, b, features, targets))
        for i in range(j):
            w_d_i = gradient_derivative_w(features, targets, w, b, i)
            w[i] = w[i] - a * (w_d_i)
        b = b - a * gradient_derivative_b(features, targets, w, b)
    return (w, b)


features = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
targets = np.array([460, 232, 178])

w = np.array([0.0,0.0,0.0,0.0])
print(gradient_descent(features, targets, w, 0))



