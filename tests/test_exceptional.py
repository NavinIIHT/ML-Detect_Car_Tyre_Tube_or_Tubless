import os
import unittest
import numpy as np

from code.ml import Model
from code import constants


model = Model()
file_path = os.path.dirname(os.path.realpath(__file__))


class ExceptionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(file_path+"/output_exception_revised.txt", "w"):
            pass

    def test_model_exists(self):

        model_exists = os.path.exists(constants.MODEL_FOLDER + constants.MODEL_FILE)

        if model_exists:
            with open(file_path+"/output_exception_revised.txt", "a") as f:
                f.write("TestModelCreated = True\n")
                print("TestModelCreated = Passed")
        else:
            with open(file_path+"/output_exception_revised.txt", "a") as f:
                f.write("TestModelCreated = False\n")
                print("TestModelCreated = Failed")

        assert model_exists

    def test_graphs_exists(self):

        n_graphs = len(os.listdir(constants.GRAPHS_FOLDER))

        if n_graphs != 0:
            with open(file_path+"/output_exception_revised.txt", "a") as f:
                f.write("TestGraphsGenerated = True\n")
                print("TestGraphsGenerated = Passed")
        else:
            with open(file_path+"/output_exception_revised.txt", "a") as f:
                f.write("TestGraphsGenerated = False\n")
                print("TestGraphsGenerated = Failed")

        assert n_graphs != 0

    '''def test_data_leakage_numerical(self):

        X_train, X_test, y_train, y_test = model.data_transformation()

        train_data = model.data_transformation(test_data=model.train, is_train=False)

        sample_size = train_data.shape[0]/X_train.shape[0]

        train_data[config.model_config.numerical_vars] = transformer.fit_transform(
                                        train_data[config.model_config.numerical_vars])
        X_train2, X_test2, y_train2, y_test2 = split_data(train_data)

        if not X_test1[config.model_config.numerical_vars].equals(
                                        X_test2[config.model_config.numerical_vars]):
            with open(file_path+"/output_exception_revised.txt", "a") as f:
                f.write("TestNumericalDataLeakage=True\n")
                print("TestNumericalDataLeakage = Passed")
        else:
            with open(file_path+"/output_exception_revised.txt", "a") as f:
                f.write("TestNumericalDataLeakage=False\n")
                print("TestNumericalDataLeakage = Failed")

    def test_data_leakage_categorical(self):

        transformer, X_train1, X_test1, y_train1, y_test1 = \
                                            preprocess_categorical_features()

        train_data = raw_training_data.copy()

        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        train_data['BsmtQual'] = imputer.fit_transform(
                            train_data['BsmtQual'].values.reshape(-1, 1))
        train_data['BsmtQual'] = transformer.fit_transform(
                            train_data['BsmtQual'].values.reshape(-1, 1))
        X_train2, X_test2, y_train2, y_test2 = split_data(train_data)

        if not X_test1['BsmtQual'].equals(X_test2['BsmtQual']):
            with open(file_path+"/output_exception_revised.txt"", "a") as f:
                f.write("TestCategoricalDataLeakage=True\n")
                print("TestCategoricalDataLeakage = Passed")
        else:
            with open(file_path+"/output_exception_revised.txt", "a") as f:
                f.write("TestCategoricalDataLeakage=False\n")
                print("TestCategoricalDataLeakage = Failed")'''
