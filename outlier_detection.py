import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load the data
@st.cache
def load_data():
    return pd.read_excel("PRA.xlsx")

# Isolation Forest function
def detect_outliers(data):
    # Specify the columns to check for outliers
    columns_to_check = ['NWP (£m)', 'SCR coverage ratio', 'Gross claims incurred (£m)', 'Net combined ratio']
    
    # Drop NA values
    data_cleaned = data.dropna(subset=columns_to_check)
    
    # Extract the data for the specified columns
    data_to_check = data_cleaned[columns_to_check].values
    
    # Create an instance of the Isolation Forest model
    isolation_forest = IsolationForest(contamination='auto', random_state=42)
    
    # Fit the Isolation Forest model to the data
    isolation_forest.fit(data_to_check)
    
    # Predict outliers in the data
    outlier_preds = isolation_forest.predict(data_to_check)
    
    # Identify the indices of outliers
    outlier_indices = pd.Series(outlier_preds).eq(-1)
    
    # Get the details of outliers
    outliers = data_cleaned[outlier_indices]
    
    return outliers

# Streamlit app
def main():
    st.title("Outlier Detection App")
    
    # Load the data
    data = load_data()
    
    # Select a single year from 2016-2020
    year = st.sidebar.selectbox("Select a Year", [2016, 2017, 2018, 2019, 2020])
    
    # Filter the data for the selected year
    data_filtered = data[data['Year'] == year]
    
    # Detect outliers
    outliers = detect_outliers(data_filtered)
    
    # Display the outliers
    st.write("Outliers detected for Year", year)
    st.write(outliers)
    
    # Plot outliers
    if not outliers.empty:
        st.write("Plot of Outliers")
        fig, ax = plt.subplots(figsize=(10, 6))
        for column in outliers.columns:
            ax.scatter(outliers.index, outliers[column], label=column)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title("Outliers Plot")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
