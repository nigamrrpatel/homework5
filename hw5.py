from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from comet_ml import Experiment
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

class TitanicFlow(FlowSpec):
   
    TRAIN_FILE = IncludeFile(
        'dataset',
        help='Train data file',
        is_text=False,
        default='/Users/nigampatel/MLSys-NYU-2023/weeks/09/titanic/train.csv')

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
    )
    
    @step
    def start(self):
        
        print("Start time{}".format(datetime.utcnow()))
        # print("flow name: %s" % current.flow_name)
        # print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        self.train_df = pd.read_csv("/Users/nigampatel/MLSys-NYU-2023/weeks/09/titanic/train.csv")
        self.testing_df = pd.read_csv("/Users/nigampatel/MLSys-NYU-2023/weeks/09/titanic/test.csv")
        print("{} rows the train dataset!".format(len(self.train_df)))
        print("train data....")
        self.train_df.head(10) 
        self.next(self.data_pre_process)

    @step
    def data_pre_process(self):
        def preprocess(df):
            df = df.copy()
            def normalize_name(x):
                return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
            def ticket_number(x):
                return x.split(" ")[-1]
            def ticket_item(x):
                items = x.split(" ")
                if len(items) == 1:
                    return "NONE"
                return "_".join(items[0:-1])
            df["Name"] = df["Name"].apply(normalize_name)
            df["Ticket_number"] = df["Ticket"].apply(ticket_number)
            df["Ticket_item"] = df["Ticket"].apply(ticket_item)                     
            return df
        
        self.processed_train_df = preprocess(self.train_df)
        self.preprocessed_testing_df = preprocess(self.testing_df)
        print("Processed train data")
        self.processed_train_df.head(10)
        self.next(self.input_features_check)
    
    @step
    def input_features_check(self):
        self.input_features = list((self.processed_train_df).columns)
        self.input_features.remove("Ticket")
        self.input_features.remove("PassengerId")
        self.input_features.remove("Survived")

        print(f"Input features: {self.input_features}")
        self.next(self.train_valid_split)
    
    @step
    def train_valid_split(self):
        self.train_df, self.valid_df= train_test_split(self.processed_train_df, test_size=0.2, random_state=42)
        self.y_valid = self.valid_df["Survived"]
        self.next(self.experiment)

    @step
    def experiment(self):
        random_state = 42

        def tokenize_names(features, labels=None):
            features["Name"] = tf.strings.split(features["Name"])
            return features, labels

        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.train_df, label="Survived").map(tokenize_names)
        valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.valid_df, label="Survived").map(tokenize_names)

        tuner = tfdf.tuner.RandomSearch(num_trials=1000)
        tuner.choice("min_examples", [2, 5, 7, 10])
        tuned_model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
        tuned_model.fit(train_ds, verbose=0)

        # Save the trained model weights
        model_weights_path = '/Users/nigampatel/MLSys-NYU-2023/weeks/11/save_model'
        tuned_model.save_weights(model_weights_path)
        print(f"Model weights saved to {model_weights_path}")

        # Make predictions on the validation set
        y_pred = tuned_model.predict(valid_ds)
        y_pred = (y_pred > 0.5).astype(int)
        y_test = self.valid_df["Survived"].tolist()

        # Save predictions and ground truth to a CSV file
        predictions_data = {
            'Sex': self.valid_df['Sex'],  # Include 'Sex'
            'Pclass': self.valid_df['Pclass'],  # Include 'Pclass'
            'Predictions': y_pred.tolist(),
            'GroundTruth': y_test
        }
        predictions_df = pd.DataFrame(predictions_data)
        predictions_file_path = '/Users/nigampatel/MLSys-NYU-2023/weeks/11/predictions.csv'
        predictions_df.to_csv(predictions_file_path, index=False)
        print(f"Predictions saved to {predictions_file_path}")

        predictions_df['Predictions'] = predictions_df['Predictions'].apply(lambda x: x[0]).astype(int)

        # Calculate and print evaluation metrics
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        print("F1 score {:6.3f}".format(f1))
        print("Precision score {:6.3f}".format(precision))
        print("Recall score {:6.3f}".format(recall))

        self.next(self.end)

    @step
    def end(self):
        print("All set!")

if __name__ == '__main__':
    TitanicFlow()