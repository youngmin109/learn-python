import numpy as np
import matplotlib.pyplot as plt

# Data set 
# input data (features) 
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50) ] # 파이썬에서 list compilation 이 나온 이유
y_train = [val * np.random.rand() * 5 for val in x_train]
# y를 Lable 로 표기 

# BGC 배치경사하강법을 이용하여 선형회귀를 적용

print(x_train)
print("-" * 100)
print(y_train)

plt.scatter(x_train,y_train, color = "blue")
plt.show()
# output data (lable)
# f(x1) -> f(x2)