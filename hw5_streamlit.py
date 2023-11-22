import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

# Load pre-made predictions from the CSV file
predictions_df = pd.read_csv('/Users/nigampatel/MLSys-NYU-2023/weeks/11/predictions.csv')
predictions_df['Predictions'] = predictions_df['Predictions'].apply(lambda x: int(x[1]))  # Convert to integers

# Dropdown menu for selecting category
selected_category = st.selectbox("Select Category", ['Sex', 'Pclass'])

# Filter data based on the selected category
filtered_data = predictions_df[['GroundTruth', 'Predictions', selected_category]]

# Calculate accuracy for total and each group
total_accuracy = accuracy_score(filtered_data['GroundTruth'], filtered_data['Predictions'])
group_accuracies = []

unique_values = filtered_data[selected_category].unique()
for value in unique_values:
    subset = filtered_data[filtered_data[selected_category] == value]
    accuracy = accuracy_score(subset['GroundTruth'], subset['Predictions'])
    group_accuracies.append({f'{selected_category}={value}': accuracy})

# Display results
st.write(f"Total Accuracy: {total_accuracy}")
for group_accuracy in group_accuracies:
    st.write(group_accuracy)

