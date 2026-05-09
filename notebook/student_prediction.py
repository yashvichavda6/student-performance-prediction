import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

plt.switch_backend('TkAgg')

print("Student Performance Project Started")

data = pd.read_csv(
    r"C:\Users\91635\Desktop\Task 8 Student Performance Prediction\data\student_data.csv"
)

print("\nDataset Loaded Successfully\n")

print(data.head())

print("\nDataset Information:\n")
print(data.info())

print("\nMissing Values:\n")
print(data.isnull().sum())

encoder = LabelEncoder()

data['Extracurricular Activities'] = encoder.fit_transform(
    data['Extracurricular Activities']
)

x = data.drop('Performance Index', axis=1)

y = data['Performance Index']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor()

model.fit(x_train, y_train)

prediction = model.predict(x_test)

r2 = r2_score(y_test, prediction)

mae = mean_absolute_error(y_test, prediction)

print("\nModel Results\n")

print("R2 Score :", r2)

print("Mean Absolute Error :", mae)

importance = pd.Series(
    model.feature_importances_,
    index=x.columns
)

importance.sort_values().plot(
    kind='barh',
    figsize=(8,6)
)

plt.title("Feature Importance")

plt.savefig("images/feature_importance.png")

plt.show()

plt.figure(figsize=(7,5))

sns.scatterplot(
    x=y_test,
    y=prediction
)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted Performance")

plt.savefig("images/performance_prediction.png")

plt.show()

print("\nStudent Performance Prediction Completed Successfully")