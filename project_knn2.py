import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Load the data
data = pd.read_csv("subscription_prediction.csv")
X = data.drop('y', axis=1)
y = data['y']

# Shuffle and split the data
shuffled_data = data.sample(frac=1, random_state=417).reset_index(drop=True)
train_size = int(0.8 * len(data))

X_train = shuffled_data.drop('y', axis=1).iloc[:train_size]
y_train = shuffled_data['y'].iloc[:train_size]
X_test = shuffled_data.drop('y', axis=1).iloc[train_size:]
y_test = shuffled_data['y'].iloc[train_size:]

# KNN function
def knn(features, single_test_input, k):
    squared_distance = 0
    for feature in features:
        squared_distance += (X_train[feature] - single_test_input[feature])**2
    
    # Store the distance in the training set
    X_train["distance"] = squared_distance**0.5
    prediction = y_train[X_train["distance"].nsmallest(n=k).index].mode()[0]
    return prediction

# Custom CSS for colorful cards and layout
st.markdown("""
    <style>
    .card {
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        font-size: 22px;
        color: white;
        background-color: #4CAF50;
        text-align: center;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
    }
    .card:hover {
        background-color: #3e8e41;
    }
    .card-blue {background-color: #1E90FF;}
    .card-orange {background-color: #FF5722;}
    .container {display: flex; justify-content: space-between; flex-wrap: wrap;}
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Subscription Prediction Dashboard")

# Feature selection for prediction
available_features = X.columns.tolist()
selected_features = st.multiselect("Select Features for Prediction", available_features)

# User input for the selected features
feature_inputs = {}
for feature in selected_features:
    feature_inputs[feature] = st.number_input(f"Enter value for {feature}:", value=0)

if st.button("Predict"):
    # Create a single test input with the selected feature values
    single_test_input = pd.Series(feature_inputs)
    
    # Make a prediction
    model_prediction = knn(selected_features, single_test_input, 3)
    
    # Calculate predictions for the entire test set based on the selected features
    X_test["age_predicted_y"] = X_test.apply(lambda x: knn(selected_features, x, 3), axis=1)
    
    # Add actual values to the test set DataFrame
    X_test['actual_y'] = y_test.values  # Ensure actual_y is present in X_test

    # Calculate accuracy
    accuracy = (X_test["age_predicted_y"] == y_test).mean() * 100

    # Display key metrics as colorful cards
    st.markdown(f"""
    <div class="container">
        <div class="card card-blue">Predicted Subscription: {model_prediction}</div>
        <div class="card card-orange">Model Accuracy: {accuracy:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Show detailed predictions
    results = X_test[selected_features].copy()
    results['actual_y'] = y_test.values  # Make sure actual_y is copied here
    results['predicted_y'] = X_test["age_predicted_y"]

    # Plotly chart for actual vs predicted
    fig = px.scatter(results, x='actual_y', y='predicted_y',
                     title="Actual vs Predicted Subscriptions",
                     labels={"actual_y": "Actual", "predicted_y": "Predicted"},
                     color="actual_y", symbol="predicted_y")

    st.plotly_chart(fig)

    # Show a bar chart for feature importance or sample values (mock example)
    fig_bar = px.bar(x=selected_features, y=feature_inputs.values(), title="Feature Values Input")
    st.plotly_chart(fig_bar)

    # Display predictions in table
    st.subheader("Detailed Predictions")
    st.dataframe(results)

# Optionally display the test set predictions with distances
if st.checkbox("Show Full Test Set Predictions"):
    # Calculate distances for display
    if 'distance' not in X_test.columns:  # Check if 'distance' needs to be calculated
        distances = []
        for i in range(len(X_test)):
            squared_distance = sum((X_train[selected_features].iloc[:, j] - X_test.iloc[i][selected_features].values[j])**2 for j in range(len(selected_features)))
            distances.append(squared_distance**0.5)
        X_test["distance"] = distances
    
    # Safely display the relevant columns
    st.subheader("Test Set Predictions with Distances")
    st.dataframe(X_test[['distance', 'actual_y', 'age_predicted_y'] + selected_features])
