import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv("D:/Internship/Day-10-11th July/house_price_prediction_dataset.csv")

# Select the relevant features and target
X = data[['num_bedrooms', 'num_bathrooms', 'square_footage', 'age_of_house']]
y = data['house_price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()  
model.fit(X_train, y_train)

# Save the trained model as a pickle file
joblib.dump(model, 'house_price_model.pkl')
