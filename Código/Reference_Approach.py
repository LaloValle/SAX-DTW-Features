"""
    The reference or baseline approach, consist in achieving the classification of a time series with not yet identified class label
    through the simplest classification model KNN, being empirically observed with K=1 the best precision and performance.

    Different to the typical 1-NN algorithm, and due to the nature of the time series, the DTW algorithm is used as measure
    function to defined the similarity between the to-be-predicted time series and all 'training' examples.

    Pros
    ----
    - Very simple implementation that requires of the data no previous special treatment aside the basic cleaning
        of the raw series in every axis
    - Does not require a training phase
    - Very High accuracy

    Cons
    ----
    - Is computationally and time expensive due to the compute of the DTW distance with all the existing entries of the training
      set which needs to be compare with whenever a new prediction need to be made

    Upgrades
    --------
    - Previously through a technique using Cross Validation and synthetic sampling of datasets with limited entries(small dataset),
      a value of w for the window size can be found for the Sakoe-Chiba constraint of DTW. 
      This improves the cuadratic complexity, thus time, and helps in the accuracy of the similarity calculation between time series.
    - Using the lower bound LB Keogh, LB Keogh <= DTW distance, that has a linear complexity there is an improvement in the
      similarity calculations. When comparing the time series in interest with all the training series, the most expensive
      DTW distance compute is performed only on those series that had a LB Keogh smaller than the last better(smaller) DTW distance.
"""

import numpy as np
import pandas as ps
import matplotlib.pyplot as plt
# Local Libraries
from LLibraries.DTW import DTW_1NN
from LLibraries.DTWWindowOptimizer import *
from LLibraries.Tools import stratified_sampling,accuracy,cross_validation

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
    # The dataset gets stratified and returns as a list of arrays of potentially variable
    #    It's necesary to work with a list of arrays for the time series which will be working with are of variable length
    #    thus, an structure as an array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling(gestures_Z_dataset,max_possible_instances=False,instances_per_class=20)
    print('Stratified dataset number instances >> ', len(stratified_dataset))

    # Test in prediction
    dtw_distance,_ = DTW(stratified_dataset[0][1:],stratified_dataset[1][1:],window_size=3)
    print(stratified_dataset[0].size,stratified_dataset[1].size)
    print('Predictions >> ', predictions)
    print('Tests labels >> ',[instance[0] for instance in stratified_dataset[:5]])
    print('Accuracy >> ', accuracy(np.array([instance[0] for instance in stratified_dataset[:5]]),predictions))

    #=================
    # Crossvalidation
    #=================
    # Test the cross validation
    accuracy = cross_validation(stratified_dataset,DTW_1NN,window_size=7,k=20,verbose=True)

    #=============================
    # Minimum window size related
    #=============================
    # Test the warping adding
    warped_instance = np.insert(add_warping(stratified_dataset[0][1:]),0,stratified_dataset[0][0])
    print('DTW distance with warped >> ', DTW(stratified_dataset[0][1:],warped_instance[1:],window_size=3))
    # Test the creation of a new set
    new_set = create_augmented_set(stratified_dataset[:50])
    print('New set lenght >> ', len(new_set))

    #==========
    # Plotting
    #==========
    # Z Axis
    plt.plot([x for x,_ in enumerate(stratified_dataset[0][1:])],stratified_dataset[0][1:], color='m')
    plt.scatter([x for x,_ in enumerate(stratified_dataset[0][1:])],stratified_dataset[0][1:], color='m')
    plt.plot([x for x,_ in enumerate(warped_instance[1:])],warped_instance[1:], color='k')
    plt.scatter([x for x,_ in enumerate(warped_instance[1:])],warped_instance[1:], color='k')

    plt.show()


    
def minimum_window_size():
    ###################################
    #   Value of the minimum window
    # Accuracy: 86.33%
    # window value: 16
    ###################################
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
    # The dataset gets stratified and returns as a list of arrays of potentially variable
    #    It's necesary to work with a list of arrays for the time series which will be working with are of variable length
    #    thus, an structure as an array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling(gestures_Z_dataset,max_possible_instances=False,instances_per_class=20)
    print('Stratified dataset number instances >> ', len(stratified_dataset))

    #=====================
    # Minimum window size
    #=====================
    # Minimum window size
    window_size = minimum_warping_window(stratified_dataset,DTW_1NN,number_iterations=5,upper_bound_window=17,lower_bound_window=11,verbose=True)

def prediction_performance_test():
    ##############################
    #   Performance predicition
    # Window value: 16
    # LB Keogh boundary: 5
    # Testing set size: 150
    # Training set size: 120
    #
    # Accuracy: 75.2%
    # Time: 57.91 sec
    ##############################
    #============================
    # Recovering of the datasets
    #============================
    gestures_Z_test_dataset,gestures_Z_train_dataset = [],[]
    window_size = 16
    #-------------
    # Testing Set
    #-------------
    gestures_Z_test_dataset = np.genfromtxt(
        './Data/GesturePebbleZ1/GesturePebbleZ1_TEST.tsv',
        dtype = np.float16,
        delimiter = '	',
        missing_values = {'NaN'},
        filling_values = {np.NaN}
    )
    #--------------
    # Training Set
    #--------------
    gestures_Z_train_dataset = np.genfromtxt(
        './Data/GesturePebbleZ1/GesturePebbleZ1_TRAIN.tsv',
        dtype = np.float16,
        delimiter = '	',
        missing_values = {'NaN'},
        filling_values = {np.NaN}
    )
    # Class labels
    unique_labels = np.unique(gestures_Z_test_dataset[:,0])
    print('Number of classes >> ', unique_labels.size)
    print('Max lenght time series >> ', gestures_Z_test_dataset.shape[1])
    print('Number instances testing set >> ', gestures_Z_test_dataset.shape[0])
    print('Number instances training set >> ', gestures_Z_train_dataset.shape[0])

    #=======================
    # Processing of Dataset
    #=======================
    # The dataset gets stratified and returns as a list of arrays of potentially variable lengths
    #    It's necesary to work with a list of arrays for the time series which will be working with are of variable length
    #    thus, an structure as an array cannot work with arrays in a higher dimension with different sizes on them
    training_dataset = stratified_sampling(gestures_Z_train_dataset,max_possible_instances=True)
    testing_dataset = stratified_sampling(gestures_Z_test_dataset,max_possible_instances=True)
    print('Stratified testing dataset number instances >> ', len(testing_dataset))
    print('Stratified training dataset number instances >> ', len(training_dataset))

    #================
    # Classification
    #================
    print('<--- Starting the classification process --->')
    starting_time = time.time()
    predictions = DTW_1NN(testing_dataset,training_dataset,window_size)
    ending_time = time.time() - starting_time
    # Accuracy of the predictions
    results = {
        'labels' : [instance[0] for instance in testing_dataset],
        'predictions' : predictions
    }
    acc = accuracy(results['labels'],results['predictions'])
    print(ps.DataFrame(results))
    print('<--- Finished with accuracy: {}, in {:.2f} sec --->'.format(acc,ending_time))



def main():
    #prediction_performance_test()
    tests()

if __name__ == '__main__': main()