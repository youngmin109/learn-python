import numpy as np
import matplotlib.pyplot as plt


# Data set
# input
# input data, features
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50)]
y_train = [val + np.random.rand() * 5 for val in x_train]


# BGC (Batch Gradient Descent) 배치경사하강법을 
# 이용하여 Linear Rergeresion 적용
w = 0.0
b = 0.0
learning_rate = 0.01
epoch = 300


# H(x) -> w * x + b
for num_of_epoch in range(epoch):
    gradient_w_sum = 0.0
    gradient_b_sum = 0.0
    # GD 수행 후 최적의 W 값 도출
    for x, y in zip(x_train, y_train):
        gradient_w_sum += x * (w * x + b - y)
        gradient_b_sum += (w * x + b - y)
        
    # Loss 값 출력
    print(f"{num_of_epoch}회 : w의 loss : {(gradient_w_sum)**2 / len(x_train)}")
    print(f"{num_of_epoch}회 : b의 loss : {(gradient_b_sum)**2 / len(x_train)}")
    
    # W 값 업데이트
    w = w - learning_rate * (gradient_w_sum / len(x_train))
    b = b - learning_rate * (gradient_b_sum / len(x_train))
    
x_test = [val for val in range(10)]
y_test = [w * val + b for val in x_test]

plt.scatter(x_train,y_train, color ="blue")
plt.plot(x_test, y_test)
plt.show()

# output
# label
# f(x1) -> f(x2)