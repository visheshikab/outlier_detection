import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the data
@st.cache
def load_data():
    return pd.read_excel("PRA.xlsx")

# Isolation Forest function
def detect_outliers(data, columns_to_check):
    # Drop NA values
    data_cleaned = data.dropna(subset=columns_to_check).reset_index(drop=True)
    
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
    
    # Define the default columns to check for outliers
    default_columns = ['NWP (£m)', 'SCR coverage ratio', 'Gross claims incurred (£m)', 'Net combined ratio']
    
    # Allow users to select additional columns to check for outliers
    additional_columns = st.sidebar.multiselect("Select additional columns to check for outliers", 
                                                ['EoF for SCR (£m)', 'Excess of assets over liabilities (£m) [= equity]', 
                                                 'GWP (£m)', 'SCR (£m)', 'Total assets (£m)', 'Total liabilities (£m)', 
                                                 'Gross BEL (inc. TPs as whole, pre-TMTP) (£m)', 'Gross combined ratio', 
                                                 'Gross expense ratio', 'Net BEL (inc. TPs as a whole, pre-TMTP) (£m)', 
                                                 'Net expense ratio', 'Pure gross claims ratio', 'Pure net claims ratio'],
                                                default=default_columns)
    
    # Combine default and additional columns
    all_columns = list(set(default_columns + additional_columns))
    
    # Detect outliers
    outliers = detect_outliers(data_filtered, all_columns)
    
    # Display the outliers
    st.write("Outliers detected for Year", year)
    st.write(outliers)

if __name__ == "__main__":
    main()
