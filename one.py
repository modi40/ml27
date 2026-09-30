import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("house_prices.csv").dropna()

X = df[['Square_Feet', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Multiple Linear Regression Model on House Price Dataset:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

sample = pd.DataFrame({'Square_Feet': [2200], 'Bedrooms': [3], 'Bathrooms': [2], 'Age': [4]})
pred_price = model.predict(sample)[0]
print(f"Predicted price for sample house: {pred_price:.2f}")
