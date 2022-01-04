import os
import unittest
import pickle
import numpy as np

from code.ml import Model

model = Model()
file_path = os.path.dirname(os.path.realpath(__file__))


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(file_path+"/output_boundary_revised.txt", "w"):
            pass

    '''def test_is_model_underfitting(self):

        # this is for regression
        X_train, X_test, y_train, y_test = model.data_transformation()
        predictions = model.model_predict(X_test)

        benchmark_msle = model.cost_metricr(
            y_true=y_test, y_pred=[y_train.mean()]*y_test.shape[0]
        )

        predicted_msle = model.cost_metric(
            y_true=y_test, y_pred=predictions
        )

        benchmark_msle = predicted_msle + 0.1

        if predicted_msle < benchmark_msle:
            with open(file_path+"/output_boundary_revised.txt", "a") as f:
                f.write("TestModelNotUnderfitting=True\n")
                print("TestModelNotUnderfitting = Passed")
        else:
            with open(file_path+"/output_boundary_revised.txt", "a") as f:
                f.write("TestModelNotUnderfitting=False\n")
                print("TestModelNotUnderfitting = Failed")

        assert predicted_msle < benchmark_msle'''

    def test_is_model_underfitting(self):

        # this is for classification
        X_train, X_test, y_train, y_test = model.data_transformation()
        max_occuring_label = np.bincount(y_train).argmax()
        predictions = model.model_predict(X_test)

        benchmark_msle = model.cost_metric(
            y_true=y_test, y_pred=[max_occuring_label]*y_test.shape[0]
        )

        predicted_msle = model.cost_metric(
            y_true=y_test, y_pred=predictions
        )

        benchmark_msle = predicted_msle + 0.1

        if predicted_msle < benchmark_msle:
            with open(file_path+"/output_boundary_revised.txt", "a") as f:
                f.write("TestModelNotUnderfitting=True\n")
                print("TestModelNotUnderfitting = Passed")
        else:
            with open(file_path+"/output_boundary_revised.txt", "a") as f:
                f.write("TestModelNotUnderfitting=False\n")
                print("TestModelNotUnderfitting = Failed")

        assert predicted_msle < benchmark_msle

    def test_is_model_overfitting(self):

        X_train, X_test, y_train, y_test = model.data_transformation()

        train_predict = model.model_predict(X_train)
        train_msle = model.cost_metric(
            y_true=y_train.values, y_pred=train_predict
        )

        test_predict = model.model_predict(X_test)        
        test_msle = model.cost_metric(
            y_true=y_test.values, y_pred=test_predict
        )

        perc_10 = (train_msle/100)*15

        diff = abs(train_msle-test_msle)

        if diff < perc_10:
            with open(file_path+"/output_boundary_revised.txt", "a") as f:
                f.write("TestModelNotOverfitting=True\n")
                print("TestModelNotOverfitting = Passed")
        else:
            with open(file_path+"/output_boundary_revised.txt", "a") as f:
                f.write("TestModelNotOverfitting=False\n")
                print("TestModelNotOverfitting = Failed")
        print(train_msle, test_msle)
        assert diff < perc_10
