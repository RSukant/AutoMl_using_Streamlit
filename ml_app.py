import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
import plotly.express as px
import mlflow
import mlflow.sklearn

# Function to load dataset
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to train model
def train_model(df, target_feature):
    # Drop the target feature from the feature matrix
    X = df.drop(columns=[target_feature])
    
    # Select the target feature
    y = df[target_feature]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline that standardizes the data then trains a Support Vector Machine
    pipeline = make_pipeline(StandardScaler(), SVC())
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_test, y_test, target_feature

# Function to make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Function to evaluate model
# Function to evaluate model
def evaluate_model(predictions, y_test):
    report = classification_report(y_test, predictions, zero_division='warn')
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, report


# Function to visualize data
def visualize_data(df):
    st.write("### Visualization")
    
    # Interactive bar chart based on correlation
    fig = px.bar(df.corr().stack().reset_index(name='Correlation'), 
                 x='level_0', y='level_1', color='Correlation', 
                 labels={'level_0': 'Feature 1', 'level_1': 'Feature 2', 'Correlation': 'Correlation'}, 
                 title='Correlation between Features', 
                 color_continuous_scale=px.colors.sequential.RdBu_r)
    st.plotly_chart(fig)

# Function to track experiment using MLflow
def track_experiment(accuracy, report):
    mlflow.set_experiment("ML Application")
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        with open("classification_report.txt", "wb") as f:
            f.write(report.encode())
        mlflow.log_artifact("classification_report.txt")

# Function to predict target feature
def predict_target_feature(df, model, target_feature):
    st.title('Prediction Page')
    if df is not None and model is not None and target_feature is not None:
        st.write("Please enter values for each feature to predict the target feature value:")

        # Input values for each feature
        feature_values = {}
        for feature in df.columns:
            if feature != target_feature:
                feature_values[feature] = st.text_input(f"Enter value for {feature}", value=0.0)
        
        # Predict target feature value
        input_data = pd.DataFrame([feature_values])
        prediction = model.predict(input_data)
        
        # Display prediction result
        st.write("Prediction Result:")
        if prediction[0] == 1:
            st.write("Yes")
        else:
            st.write("No")
    else:
        st.write("Please upload a dataset and train the model on the Home page.")

# Function to display profile
def display_profile():
    st.title('Profile Page')
    # Write some details
    st.write("Name: SUKANT R")
    st.write("Roll no: 717822I160")
    st.write("Department: Artificial Intelligence and Data Science")
    st.write("Location: Karpagam College of Engineering")

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox("Go to", ('Home', 'Prediction', 'Profile'))

    # Initialize session state
    session_state = st.session_state
    if 'df' not in session_state:
        session_state.df = None
    if 'model' not in session_state:
        session_state.model = None
    if 'target_feature' not in session_state:
        session_state.target_feature = None

    if page == 'Home':
        st.title('Machine Learning Application')
        # Upload dataset
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

        if uploaded_file is not None:
            session_state.df = load_data(uploaded_file)
            
            # Select target feature
            session_state.target_feature = st.selectbox("Select Target Feature", options=session_state.df.columns)

            # Train model
            session_state.model, X_test, y_test, _ = train_model(session_state.df, session_state.target_feature)

            # Make predictions
            predictions = make_predictions(session_state.model, X_test)

            # Evaluate model
            accuracy, report = evaluate_model(predictions, y_test)
            st.write("### Model Evaluation")
            st.write("Accuracy:", accuracy)
            st.write("Classification Report:")
            st.text(report)

            # Visualize data
            visualize_data(session_state.df)

            # Track experiment
            track_experiment(accuracy, report)

    elif page == 'Prediction':
        predict_target_feature(session_state.df, session_state.model, session_state.target_feature)

    elif page == 'Profile':
        display_profile()

if __name__ == "__main__":
    main()
