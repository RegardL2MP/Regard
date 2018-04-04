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
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import  BaseEstimator
from sklearn.neural_network import MLPClassifier

from prepro import NormalizePreprocessor, PCAPreprocessor
from sklearn.pipeline import Pipeline


class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        Model is the class called by Codalab.
        This class must have at least a method "fit" and a method "predict".
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False

	""" Apres test de 3 methodes (normalisation, PCA, rien) le meilleur resutlat est avec normalisation, c'est donc celui ci qu'on utilise"""
        self.mod = Pipeline([
                ('preprocessing', NormalizePreprocessor()),
                ('predictor', Predictor())
                ])
        print("MODEL=" + self.mod.__str__())

    def fit(self, X, y):
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
        
        # Do not remove the "debug code" this is for "defensive programming"
        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        
        # Thi sis where training happens
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
        num_test_samples = len(X)
        if X.ndim>1: num_feat = len(X[0])
        print("PREDICT input: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")    
    
        Y = self.mod.predict(X)
        
        num_labels = 1
        if Y.ndim>1: num_labels = len(Y[0])
        print("PREDICT output: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        if (self.num_labels != num_labels):
            print("ARRGH: number of labels in X does not match training data!")
            
        return Y

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
        
class Predictor(BaseEstimator):
    '''Predictor: modify this class to create a predictor of
    your choice. This could be your own algorithm, of one for the scikit-learn
    models, for which you choose the hyper-parameters.'''
    def __init__(self):
        '''This method initializes the predictor.'''
        self.mod = MLPClassifier((256, 128), activation="tanh", max_iter=20, solver="adam", alpha=1e-6, batch_size = 128, verbose=True)
        print("PREDICTOR=" + self.mod.__str__())
	
    """
    def augment_data(self, X):
        new_x = np.zeros((X.shape[0], X.shape[1]+768))
        new_x[:,:256] = X
        new_x[:,256:] = X[:,self.pairs[:,0]] * X[:,self.pairs[:,1]]
        return new_x

    """

    def fit(self, X, y):
        ''' This is the training method: parameters are adjusted with training data.'''
        self.mod = self.mod.fit(X, y)
        return self

    def predict(self, X):
        #X = self.augment_data(X)
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.mod.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
 
from sys import argv, path       
if __name__=="__main__":
    # Modify this class to serve as test
    
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data" # A remplacer par le bon chemin
        output_dir = "../results" # A remplacer par le bon chemin
        code_dir = "../starting_kit/ingestion_program" # A remplacer par le bon chemin
        metric_dir = "../starting_kit/scoring_program" # A remplacer par le bon chemin
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        code_dir = argv[3]
        metric_dir = argv[4]
        
    path.append (code_dir)
    path.append (metric_dir)
    
    metric_name = 'bac_multiclass'
    import libscores
    scoring_function = getattr(libscores, metric_name)
    print 'Using scoring metric:', metric_name
            
    from data_manager import DataManager    
    basename = 'cifar10'
    D = DataManager(basename, input_dir) # Load data
    print D
    
    # Here we define two models and compare them; you can define more than that
    model_dict = {
            'BasicPred': Predictor(),
            'PipelinePCA': Pipeline([('prepro', PCAPreprocessor()), ('predictor', Predictor())]),
            'PipelineNormalize': Pipeline([('prepro', NormalizePreprocessor()), ('predictor', Predictor())])
	}
   
    for key in model_dict:
        mymodel = model_dict[key]
        print("\n\n *** Model {:s}:{:s}".format(key,model_dict[key].__str__()))
 
        # Train
        print("Training")
        X_train = D.data['X_train']
        Y_train = D.data['Y_train']
        mymodel.fit(X_train, Y_train)
    
        # Predictions on training data
        print("Predicting")
        Ypred_tr = mymodel.predict(X_train)
        
        # Cross-validation predictions
        print("Cross-validating")
        from sklearn.model_selection import KFold
        from numpy import zeros  
        n = 10 # 10-fold cross-validation
        kf = KFold(n_splits=n)
        kf.get_n_splits(X_train)
        Ypred_cv = zeros(Ypred_tr.shape)
        i=1
        for train_index, test_index in kf.split(X_train):
            print("Fold{:d}".format(i))
            Xtr, Xva = X_train[train_index], X_train[test_index]
            Ytr, Yva = Y_train[train_index], Y_train[test_index]
            mymodel.fit(Xtr, Ytr)
            Ypred_cv[test_index] = mymodel.predict(Xva)
            i = i+1
            

        # Compute and print performance
        training_score = scoring_function(Y_train, Ypred_tr)
        cv_score = scoring_function(Y_train, Ypred_cv)
        
        print("\nRESULTS FOR SCORE {:s}".format(metric_name))
        print("TRAINING SCORE= {:f}".format(training_score))
        print("CV SCORE= {:f}".format(cv_score))
