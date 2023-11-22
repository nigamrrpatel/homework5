import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score

#loading data from the predicted CSV file and converting  Predictions column to integers
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Predictions'] = df['Predictions'].apply(lambda x: int(x[1]))
    return df

#calculating the accuracy score of predictions made by the model
def calculate_accuracy(df, category):
    #getting unique values of the selected category from the filtered data
    filtered_data = df[['GroundTruth', 'Predictions', category]]
    total_accuracy = accuracy_score(filtered_data['GroundTruth'], filtered_data['Predictions'])
    return total_accuracy, filtered_data

#calculating the accuracy score for each group in the selected category
def calculate_group_accuracies(filtered_data, category):
    group_accuracies = []
    
    #iterating over each unique value
    for value in filtered_data[category].unique():
        #creating a subset of the data where the selected category is equal to current unique value
        subset = filtered_data[filtered_data[category] == value]
        #calculating prediction accuracy in the subset
        accuracy = accuracy_score(subset['GroundTruth'], subset['Predictions'])
        group_accuracies.append({f'{category}={value}': accuracy})
    return group_accuracies

#displaying the total accuracy and group accuracies
def display_results(total_accuracy, group_accuracies):
    st.write(f"Total Accuracy: {total_accuracy}")
    for group_accuracy in group_accuracies:
        st.write(group_accuracy)

#starting point of the application
def main():
    file_path = 'predictions.csv'
    predictions_df = load_data(file_path)
    #creating the drop-down menu for asked categories in the problem
    selected_category = st.selectbox("Select Category", ['Sex', 'Pclass'])
    total_accuracy, filtered_data = calculate_accuracy(predictions_df, selected_category)
    group_accuracies = calculate_group_accuracies(filtered_data, selected_category)
    display_results(total_accuracy, group_accuracies)
    
if __name__ == "__main__":
    main()
