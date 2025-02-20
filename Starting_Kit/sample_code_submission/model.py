'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
import sklearn

from zPreprocessor import SimpleTransform

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.mod = Pipeline([
                ('preprocessing', SimpleTransform()),
                ('classification', MLPClassifier((256, 256), activation="tanh", max_iter=8, solver="adam", alpha=1e-6, batch_size = 128, verbose=True))])

        np.random.seed(0)
        self.pairs = np.dstack((np.random.randint(256, size=768), np.random.randint(256, size=768)))[0]
        print(self.pairs)
        print(self.pairs.shape)
    
    def define_model(self, name):
        print("Model already load")
       

    def augment_data(self, X):
        new_x = np.zeros((X.shape[0], X.shape[1]+768))
        new_x[:,:256] = X
        new_x[:,256:] = X[:,self.pairs[:,0]] * X[:,self.pairs[:,1]]
        return new_x
        
    def fit(self, X, Y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        print("Shape is ", X.shape)
        X = self.augment_data(X)
        # For multi-class problems, convert target to be scikit-learn compatible
        # into one column of a categorical variable
        y=self.convert_to_num(Y, verbose=False)     
        
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1] # Does not work for sparse matrices
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        
        self.mod.fit(X, y)
        self.is_trained=True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        X = self.augment_data(X)
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        
        # Return predictions as class probabilities
        y = self.mod.predict_proba(X)
        return y

    def save(self, path="./"):
        with open(path + '_model.pickle', 'wb') as f:
            print('modele name : ', path + '_model.pickle')
            pickle.dump(self , f)

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
        
    def convert_to_num(self, Ybin, verbose=True):
        ''' Convert binary targets to numeric vector (typically classification target values)'''
        if verbose: print("Converting to numeric vector")
        Ybin = np.array(Ybin)
        if len(Ybin.shape) ==1: return Ybin
        classid=range(Ybin.shape[1])
        Ycont = np.dot(Ybin, classid)
        if verbose: print(Ycont)
        return Ycont
 
