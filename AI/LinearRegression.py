from sklearn.linear_model import LinearRegression

# 특징
X = [[10.0], [8.0], [13.0], [9.0], [11.0], [14.0],
     [6.0], [4.0], [12.0], [7.0], [5.0]]

# 종속 변수
y = [8.04, 6.95, 7.58, 8.81, 8.33, 
     9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

# 선형회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X,y)

print(model.coef_) # 기울기 출력
print(model.intercept_) # 절편 출력

# x=0, x=1일 때 모델의 출력 결과 예측
y_pred = model.predict([[0], [1]])

# 예측한 y값 출력
print(y_pred)