import numpy as np
import matplotlib.pyplot as plt

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

i = 0

x_i = x_train[i]
y_i = y_train[i]

w = 200
b = 100

tmp_f_wb = compute_model_output(x_train, w, b,)

plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
