import numpy as np
import matplotlib.pyplot as plt

# Data set 
# input data (features) 
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50) ] # 파이썬에서 list compilation 이 나온 이유
y_train = [val * np.random.rand() * 5 for val in x_train]
# y를 Lable 로 표기 

w = 0.0
learning_rate = 0.001

epoch = 50
# BGC 배치경사하강법을 이용하여 선형회귀를 적용
# zip의 역할은 x,y 값을 묶어 2차원 형태로 만듬
# 하지만 개수가 맞지 않을경우 작은거에 맞춤
for num_of_epoch in range(epoch):
    gradient_sum = 0.0
    for x, y in zip(x_train, y_train):
        # GD 수행 후 최적의 W 값 도출(for문 바깥쪽)
        gradient_sum += 2 * x * (w * x - y)
        
    # W 값 업데이트 
    # W = W - Learning_rate * gradient_sum
    w = w - learning_rate * (gradient_sum / len(x_train))

print(x_train)
print("-" * 100)
print(y_train)

plt.scatter(x_train,y_train, color = "blue")
plt.show()
# output data (lable)
# f(x1) -> f(x2)