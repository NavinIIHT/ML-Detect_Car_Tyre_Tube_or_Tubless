import os
import unittest
import pickle

from code.ml import Model

model = Model()
X_train, X_test, y_train, y_test = model.data_transformation()
file_path = os.path.dirname(os.path.realpath(__file__))



class FuctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(file_path + "/output_revised.txt", "w"):
            pass

    def test_train_test_split_reproduciability(self):

        X_train2, X_test2, y_train2, y_test2 = model.data_transformation()

        if(X_test.equals(X_test2)):
            with open(file_path + "/output_revised.txt", "a") as f:
                f.write("TestTrainTestSplitReproduciability=True\n")
                print("TestTrainTestSplitReproduciability = Passed")
        else:
            with open(file_path + "/output_revised.txt", "a") as f:
                f.write("TestTrainTestSplitReproduciability=False\n")
                print("TestTrainTestSplitReproduciability = Failed")
        assert X_test.equals(X_test2)

    def test_train_vs_test_features(self):

        print(X_train.shape, X_test.shape)
        if(X_test.shape[1] == X_train.shape[1]):
            with open(file_path + "/output_revised.txt", "a") as f:
                f.write("TestTrainTestFearturesCardinality=True\n")
                print("TestTrainTestFearturesCardinality = Passed")
        else:
            with open(file_path + "/output_revised.txt", "a") as f:
                f.write("TestTrainTestFearturesCardinality=False\n")
                print("TestTrainTestFearturesCardinality = Failed")

    def test_numerical_features_imputed(self):

        na_found_train = X_train.isnull().any().any()
        na_found_test = X_test.isnull().any().any()

        if not (na_found_train or na_found_test):
            with open(file_path + "/output_revised.txt", "a") as f:
                f.write("TestNumericalFeaturesNANotFound=True\n")
                print("TestNumericalFeaturesNANotFound = Passed")
        else:
            with open(file_path + "/output_revised.txt", "a") as f:
                f.write("TestNumericalFeaturesNANotFound=False\n")
                print("TestNumericalFeaturesNANotFound = Failed")

    '''def test_numerical_features_normalized(self):

        transformer, _, _, _, _ = preprocess_numerical_features()

        if transformer is not None:
            with open("../output_functional_revised.txt", "a") as f:
                f.write("TestAreNumFeaturesNormalized=True\n")
                print("TestAreNumFeaturesNormalized = Passed")
        else:
            with open("../output_functional_revised.txt", "a") as f:
                f.write("TestAreNumFeaturesNormalized=False\n")
                print("TestAreNumFeaturesNormalized = Failed")

    def test_categorical_features_imputed(self):

        na_found_train = X_train['BsmtQual'].isnull().any().any()
        na_found_test = X_test['BsmtQual'].isnull().any().any()

        if not (na_found_train or na_found_test):
            with open("../output_functional_revised.txt", "a") as f:
                f.write("TestCategoricalFeaturesNANotFound=True\n")
                print("TestCategoricalFeaturesNANotFound = Passed")
        else:
            with open("../output_functional_revised.txt", "a") as f:
                f.write("TestCategoricalFeaturesNANotFound=False\n")
                print("TestCategoricalFeaturesNANotFound = Failed")

    def test_categorical_features_encoded(self):

        transformer, _, _, _, _ = preprocess_categorical_features()

        if transformer is not None:
            with open("../output_functional_revised.txt", "a") as f:
                f.write("TestAreNumFeaturesNormalized=True\n")
                print("TestAreNumFeaturesNormalized = Passed")
        else:
            with open("../output_functional_revised.txt", "a") as f:
                f.write("TestAreNumFeaturesNormalized=False\n")
                print("TestAreNumFeaturesNormalized = Failed")'''

    def test_pipeline_custom_model_random_state(self):
        #model.model_fit()
        loaded_model = pickle.load(open(model.model_file, 'rb'))
        model_random_state = loaded_model.random_state

        # assert model_random_state is not None
        if model_random_state is not None:
            with open(file_path + "/output_revised.txt", "a") as f:
                f.write("TestIsModelRandSeedSet=True\n")
                print("TestIsModelRandSeedSet = Passed")
        else:
            with open(file_path + "/output_revised.txt", "a") as f:
                f.write("TestIsModelRandSeedSet=False\n")
                print("TestIsModelRandSeedSet = Failed")
