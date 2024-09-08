from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Initialize Flask application
app = Flask(__name__)

# Load the dataset and train the logistic regression model
file_path = 'condensation_data.xlsx'  # Path to the dataset
data = pd.read_excel(file_path)  # Load dataset into pandas DataFrame

# Extract feature columns 
X = data[['Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)', 'Temp Diff (°C)']]  # Features
y = data['Condensation Risk (1 = Yes, 0 = No)']  # Target 

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (scaling them to have mean = 0 and variance = 1)
scaler = StandardScaler()  # Initialize scaler
X_train_scaled = scaler.fit_transform(X_train)  # Fit to training data and transform
X_test_scaled = scaler.transform(X_test)  # Transform the test data using the same scaler

# Train a logistic regression model
logreg = LogisticRegression(random_state=42)  # Initialize logistic regression with a fixed random state
logreg.fit(X_train_scaled, y_train)  # Fit the model to the scaled training data

# Define the main route for the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renders the home page of the web application, and handles form submissions to predict condensation risk.
    """
    if request.method == 'POST':
        try:
            # Extract user inputs from the form and convert them to float
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pressure = float(request.form['pressure'])
            temp_diff = float(request.form['temp_diff'])
            
            # Create a DataFrame for the input values, to be compatible with the model
            input_data = pd.DataFrame([[temperature, humidity, pressure, temp_diff]],
                                      columns=['Temperature (°C)', 'Humidity (%)', 'Pressure (hPa)', 'Temp Diff (°C)'])
            
            # Scale the input data using the pre-trained scaler
            input_data_scaled = scaler.transform(input_data)
            
            # Make a prediction using the pre-trained logistic regression model
            prediction = logreg.predict(input_data_scaled)
            
            # Determine the result based on the prediction 
            result = "Yes" if int(prediction[0]) == 1 else "No"
        
        except ValueError:
            # If there's a value error, return an error message
            result = "Invalid input. Please enter numerical values."
        
        # Render the template with the prediction result
        return render_template('index.html', result=result)
    
    # Render the default home page 
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
