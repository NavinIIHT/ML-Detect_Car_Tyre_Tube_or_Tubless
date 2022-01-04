import pickle

from code import constants

class Model():
    def __init__(self):
        self.data_file = constants.DATA_FOLDER + constants.DATA_FILE
        self.model_file = constants.MODEL_FOLDER + constants.MODEL_FILE
        self.graphs_folder = constants.GRAPHS_FOLDER

    def data_transformation(self, test_data=None, is_train=True):

        '''if is_train:
            return X_train, X_test, y_train, y_test
        else:
            return test'''
        pass

    def model_fit(self, X_train, y_train):

        # model = None
        # pickle.dump(model, open(self.model_file, 'wb'))
        pass

    def model_predict(self, X_test):

        pass

    def cost_metric(self, y_true, y_pred):

        pass

if __name__=="__main__":

    pass