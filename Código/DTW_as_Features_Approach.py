"""
    Methoud described in the paper "Using dynamic time warping distances as features for improved time series classification"
    as an approach that assures the effectiveness and improved accuracy of their method against the baseline DTW-1NN approach
    because of the use of more robusts ML models.

    The method suggest the use of the DTW distances between an interest time series and the instances of training in order to
    arrange a feature vector with the distances and use it to feed a more robust ML model as SVM, for the classification
    improving the accuracy of the predictions exploiting advantages of models like SVM that are capable of down-weigh datasets
    with noisy training instances.    
    
    According to the text, and implementing a combination of DTW and SAX features composing, it's proven the upgrade of the
    method combining possibly other time series classification methods

    Cons
    ----
        - It can be computational and time expensive as the DTW-Feature vector needs to be composed of all the training instances
          by computing the DTW distance with the interest time series
        - More parameters of the classifier needs to be specified and calculated:
            * As SVC uses a polynomial kernel the degree must be computed and determined with cross validation
        
    Pros
    ----
        - The implementation of a sturdier model can improve the accuracy classification by down-weightening outliers and noisy
          instances that might influence the result

    Parameters
    ----------
    The value of the parameters used in the article:
        polinomial kernel: linear, cuadratic or cubic
"""
from logging import Filterer
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# Local Libraries
from LLibraries.DTW import DTW_1NN
from LLibraries.DTWFeatures import DTW_F_SVC, DTW_feature_vector, DTW_features_dataset
from LLibraries.SAX import *
from LLibraries.DTWWindowOptimizer import *
from LLibraries.Tools import stratified_sampling,add_label_column, to_list_time_series_dataset
from Libraries.Tools import standardize

window_size = 0

def tests():
    #============================
    # Recovering of the datasets
    #============================
    gestures_Z_dataset = []
    #----------------
    # Pebble Gesture
    #----------------
    # Z Axis
    gestures_Z_dataset = np.genfromtxt(
        './Data/GesturePebbleZ1/GesturePebbleZ1_TRAIN.tsv',
        dtype = np.float16,
        delimiter = '	',
        missing_values = {'NaN'},
        filling_values = {np.NaN}
    )
    # Class labels
    unique_labels = np.unique(gestures_Z_dataset[:,0])
    print('Number of classes >> ', unique_labels.size)
    print('Max lenght time series >> ', gestures_Z_dataset.shape[1])

    #=======================
    # Processing of Dataset
    #=======================
    # The dataset gets stratified and returns as a list of arrays of potentially variable lengths
    #    It's necesary to work with a list of arrays for the time series are of variable length thus, an structure as an
    #    array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling( 
            gestures_Z_dataset,
            max_possible_instances=False,
            instances_per_class=2
        )
    print('Stratified dataset number instances >> ', len(stratified_dataset))
    print('Stratified labels >>', [instance[0] for instance in stratified_dataset])

    #===============================
    # DTW distances features vector
    #===============================
    # Testing 1 example
    # example_vector = DTW_feature_vector(stratified_dataset[0][1:],stratified_dataset)
    # print('DTW distances feature vector >>',example_vector)
    # print('DTW distances of same label instances >>')
    # for index,distance in enumerate(example_vector):
    #     if stratified_dataset[index][0] == stratified_dataset[0][0]:
    #         print(f'Index {index} >> {distance}')


    #================================
    # DTW distances features dataset
    #================================
    # features_dataset,features_labels = DTW_features_dataset(stratified_dataset)
    # print('Features dataset >>',features_dataset)
    # print('Features labels >>', features_labels)

def training():
    global window_size
    #============================
    # Recovering of the datasets
    #============================
    print('\n<=== Training process ===>\n')
    gestures_Z_dataset = []
    instances_per_class = 5
    #----------------
    # Pebble Gesture
    #----------------
    # Z Axis
    gestures_Z_dataset = np.genfromtxt(
        './Data/GesturePebbleZ1/GesturePebbleZ1_TRAIN.tsv',
        dtype = np.float16,
        delimiter = '	',
        missing_values = {'NaN'},
        filling_values = {np.NaN}
    )

    #=======================
    # Processing of Dataset
    #=======================
    # The dataset gets stratified and returns as a list of arrays of potentially variable lengths
    #    It's necesary to work with a list of arrays for the time series which will be working with are of variable length
    #    thus, an structure as an array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling(gestures_Z_dataset,max_possible_instances=False,instances_per_class=instances_per_class)
    print('Stratified dataset number instances >> ', len(stratified_dataset))

    #======================
    # DTW-Features vectors
    #======================
    # Array vectors with the distance between each element of the dataset with each training example
    print('\n<--- Creating the training dataset with the DTW-Features instances --->')
    starting_time = time.time()
    dtw_features, dtw_features_labels = DTW_features_dataset(stratified_dataset,window_size=window_size)
    ending_time = time.time() - starting_time
    print(f'Time elapsed >> {ending_time}sec')
    print('Number DTW-Features instances >> ', dtw_features.size)

    #==============
    # SVC Training
    #==============
    # Creates the Support Vector Machine Classifier
    classifier = svm.SVC(C=1,gamma='scale',degree=3,kernel='poly')
    # Trains the model
    print('\n<--- Training the model SVC --->')
    starting_time = time.time()
    classifier.fit(dtw_features,dtw_features_labels)
    ending_time = time.time() - starting_time
    print('<--- Model training finished: {} sec --->\n'.format(ending_time))

    return classifier

def minimum_window_size():
    ###################################
    #   Value of the minimum window
    # Accuracy: 
    # window value: 
    ###################################
    #============================
    # Recovering of the datasets
    #============================
    print('\n<=== Minimum window size process ===>\n')
    gestures_Z_dataset = []
    instances_per_class = 20
    #----------------
    # Pebble Gesture
    #----------------
    # Z Axis
    gestures_Z_dataset = np.genfromtxt(
        './Data/GesturePebbleZ1/GesturePebbleZ1_TRAIN.tsv',
        dtype = np.float16,
        delimiter = '	',
        missing_values = {'NaN'},
        filling_values = {np.NaN}
    )

    #=======================
    # Processing of Dataset
    #=======================
    # The dataset gets stratified and returns as a list of arrays of potentially variable lengths
    #    It's necesary to work with a list of arrays for the time series which will be working with are of variable length
    #    thus, an structure as an array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling(gestures_Z_dataset,max_possible_instances=False,instances_per_class=instances_per_class)
    print('Stratified dataset number instances >> ', len(stratified_dataset))

    #==================
    # Cross Validation
    #==================
    accuracy_log = cross_validation(stratified_dataset,DTW_F_SVC,window_size=25,verbose=True)

    #======================
    # Minimum window width
    #======================
    #window = minimum_warping_window(stratified_dataset,DTW_F_SVC,upper_bound_window=20,verbose=True)



def main():
    #tests()
    #training()
    minimum_window_size()

if __name__ == '__main__': main()