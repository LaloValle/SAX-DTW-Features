"""
    This method is proposed as a variant that combines the work of 2 papers: 
        1. "Gesture recognition using symbolic aggregate approximation and dynamic time warping on motion data"
        2. "Using dynamic time warping distances as features for improved time series classification"

    Using a SAX symbolic representation for every time series, a SAX-DTW-Feature vector is arranged by computing the DTW
    distance of the SAX representations of the input time series and each class representative already in it's SAX rep-
    resentation for the classification using either SVC(Support Vector Classifier) or Softmax(Multiclass Logistic Regre-
    ssion Classifier)

    Each class representative is calculated using the DTW Barycenter Averaging method after the training instances of
    each class has already been transformed into a SAX time series.

    Parameters
    ----------
    The value of the parameters used in the article:
        number of words per axis    = 64
        number of alphabet symbols  = 9
        polynomial kernel for SVC   = linear, cuadratic, cubic determined by 10-fold-cross-validation
        Sakoe Chiba window size     = determined by 10-fold-cross-validation
"""
import re
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from LLibraries.DTWWindowOptimizer import minimum_warping_window, create_augmented_set
# Local Libraries
from LLibraries.SAX import *
from LLibraries.Tools import accuracy, cross_validation, stratified_sampling,group_by_class
from LLibraries.DTWFeatures import SAX_DTW_F_SVC, DTW_features_dataset, representatives_by_class, DTW_representatives_features_dataset

#------------
# Parameters
#------------
number_words = 64; alphabet_size = 9
numeric_alphabet = True
window_size = 4


#------------------
# Auxiliar Methods
#------------------
TRAINING_DATASET = './Data/GesturePebbleZ1/GesturePebbleZ1_TRAIN.tsv'
TESTING_DATASET = './Data/GesturePebbleZ1/GesturePebbleZ1_TEST.tsv'
def dataset_recovering(path_dataset:str):
    return np.genfromtxt(
        path_dataset,
        dtype = np.float16,
        delimiter = '	',
        missing_values = {'NaN'},
        filling_values = {np.NaN}
    )

def dataset_processing(
        dataset:np.array,
        max_possible_instances:bool=False,
        instances_per_class:int=20,
        representatives:dict={},
        return_dtw_features:bool=False,
        verbose:bool=False
    ):
    global number_words, alphabet_size, numeric_alphabet, window_size
    #=======================
    # Processing of Dataset
    #=======================
    # The dataset gets stratified and returns as a list of arrays of potentially variable lengths
    #    It's necesary to work with a list of arrays for the time series are of variable length thus, an structure as an
    #    array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling( 
            dataset,
            max_possible_instances=max_possible_instances,
            instances_per_class=instances_per_class
        )
    if verbose: print('Stratified dataset number instances >> ', len(stratified_dataset))
    # SAX discretization of the whole dataset
    #   The class label gets added as the first element of the array
    sax_dataset = to_SAX_dataset(stratified_dataset, number_words, alphabet_size, numeric_alphabet)
    if verbose: print('SAX dataset length >> ', len(sax_dataset))

    if return_dtw_features:
        #======================
        # DTW-Features vectors
        #======================
        # Array vectors with the distance between each element of the dataset with the representative of the class
        if verbose: print('\n<--- Creating the training dataset with the DTW-Features instances --->')
        return DTW_representatives_features_dataset(sax_dataset, representatives=representatives, window_size=window_size)

    return sax_dataset

def train_save_model(classifier=svm.SVC(),representatives=dict()):
    import joblib

    if not representatives:
        # Recovering the trained model and the representatives of each class
        classifier, representatives = training()
    filename_model = './Results/Trained_SAX-DTW-F_SVM.sav'; filename_representatives = './Results/Class_representatives.csv'
    #==================
    # Saving the model
    #==================
    joblib.dump(classifier,filename_model)
    print('<--- SVM Model saved in Results --->')
    #==================================
    # Saving the class representatives
    #==================================
    representatives_aux = np.empty((len(representatives),number_words+1))
    for class_rep,representative in representatives.items():
        representatives_aux[class_rep-1,0] = class_rep
        representatives_aux[class_rep-1,1:] = representative
    np.savetxt(filename_representatives,representatives_aux,delimiter=',')
    print('<--- Representatives saved in Results --->')

#----------------------------
# SAX-DTW Features functions
#----------------------------
def representative_ploting():
    """Function that plots the representative time serie of each class set of time series
    
    Procedure
    ---------
        1.  First the time series of the Pebbble Gesture file gets recovered. This file only includes the measurements
            of each movements in the Z axis.
        2.  The recovered dataset gets processed
            2.1 First the dataset gets stratified so that each class has the same number of time series samples
            2.2 Then all the time series gets transformed into their SAX representation
        3.
    """

    ## Recovering of the datasets
    gestures_Z_dataset = dataset_recovering(TRAINING_DATASET)
    instances_per_class = 20

    ## Processing of instances in the dataset
    sax_dataset = dataset_processing(
        gestures_Z_dataset,
        max_possible_instances=True,
        instances_per_class=instances_per_class,
        verbose=True
    )
    # Grouping and splitting the dataset by class labels
    dataset_by_classes = group_by_class(sax_dataset)
    print('Classes >> ', list(dataset_by_classes.keys()))

    ## Barycenter Averaging Representatives
    # The representatives by class are computed using DTW Barycenter Averaging
    representatives = representatives_by_class(dataset_by_classes)

    #
    # Plotting
    #----------
    fig,axs = plt.subplots(3,2)
    
    ## SAX series by class
    labels = list(dataset_by_classes.keys()); labels.sort()
    for label in labels:
        for index in range(instances_per_class):
            axs[int((label+1)/2)-1][1-(label%2)].set_title('Class {}'.format(label))
            #   Black color for comparing with representative
            axs[int((label+1)/2)-1][1-(label%2)].plot([x for x,_ in enumerate(dataset_by_classes[label][index])],dataset_by_classes[label][index],linewidth=0.5,color='k',alpha=0.5)
            #   One color for each time series
            # axs[int((label+1)/2)-1][1-(label%2)].plot([x for x,_ in enumerate(dataset_by_classes[label][index])],dataset_by_classes[label][index])
        # Representative of the class
        axs[int((label+1)/2)-1][1-(label%2)].plot([x for x,_ in enumerate(representatives[label])],representatives[label],linewidth=2.5,color='r')
    
    fig.suptitle('SAX series by class')    
    fig.suptitle('Comparison SAX representation of training time series and representative by class')
    fig.supxlabel('Measurements in time units')
    fig.supylabel('Accelaration in Z axis')

    plt.show()

def training():
    global window_size
    print('\n<=== Training process ===>\n')
    gestures_Z_dataset = dataset_recovering(TRAINING_DATASET)

    ## Processing of instances in the dataset
    starting_time = time.time()
    dtw_features,dtw_features_labels,representatives = dataset_processing(
        gestures_Z_dataset,
        max_possible_instances=True,
        verbose=True, return_dtw_features=True
    )

    #==============
    # SVC Training
    #==============
    # Creates the Support Vector Machine Classifier
    classifier = svm.SVC(C=1,gamma='scale',degree=3,kernel='poly')
    # Trains the model
    print('\n<--- Training the model SVC --->')
    classifier.fit(dtw_features,dtw_features_labels)
    ending_time = time.time() - starting_time
    print('<--- Model training finished: {} sec --->\n'.format(ending_time))

    return classifier,representatives

def testing(classifier:svm.SVC, representatives:dict):
    ## Recovering the training dataset
    print('\n<=== Testing process ===>\n')
    instances_per_class = 5
    gestures_Z_dataset = dataset_recovering(TESTING_DATASET)

    ## Processing of Dataset
    dtw_features,dtw_features_labels,_ = dataset_processing(
        gestures_Z_dataset,
        instances_per_class=instances_per_class,
        representatives=representatives,
        return_dtw_features=True, verbose=True
    )
    print('Number DTW-Features instances >> ', dtw_features.shape[0])

    #=============
    # Testing SVC
    #=============
    print('<--- Starting predictions --->')
    starting_time = time.time()
    predictions = classifier.predict(dtw_features)
    ending_time = time.time() - starting_time
    #results = {
    #    'Classes' : dtw_features_labels,
    #    'Predictions' : predictions
    #}
    acc = accuracy(dtw_features_labels,predictions)
    print('Accuracy >> ', acc)
    #print(ps.DataFrame(results))
    print('<--- Predictions finished, accuracy:{:.2f}% , {:.2f} sec --->'.format(acc*100,ending_time))

def minimum_window_size():
    ###################################
    #   Value of the minimum window
    # Accuracy: 95.25%
    # window value: 4
    ###################################
    global number_words, alphabet_size, numeric_alphabet
    #============================
    # Recovering of the datasets
    #============================
    print('\n<=== Minimum window size process ===>\n')
    gestures_Z_dataset = dataset_recovering(TRAINING_DATASET)
    instances_per_class = 5

    ## Processing of the dataset
    sax_dataset = dataset_processing(gestures_Z_dataset,instances_per_class=instances_per_class,verbose=True)
    # Obtains the representatives
    representatives = representatives_by_class(sax_dataset)

    #==================
    # Cross Validation
    #==================
    # accuracy_log = cross_validation(sax_dataset,SAX_DTWF_SVC,window_size=3,alphabet_size=alphabet_size,k=5,representatives=representatives,verbose=True)

    #======================
    # Minimum window width
    #======================
    window = minimum_warping_window(
            sax_dataset,
            SAX_DTW_F_SVC,
            alphabet_size=alphabet_size,
            upper_bound_window=10,
            representatives=representatives,
            convert_int=True,verbose=True
        )
    print('Value of the optimized window >>',window)


def prediction_performance_test():
    ##############################
    #   Performance predicition
    # Window value: 4
    # Testing set size: 150
    # Training set size: 120
    #
    # Accuracy: 94.06%
    # Time: 5.181 sec
    ##############################
    #============================
    # Recovering of the datasets
    #============================
    gestures_Z_test_dataset,gestures_Z_train_dataset = [],[]
    global number_words, alphabet_size, numeric_alphabet, window_size

    gestures_Z_test_dataset = dataset_recovering(TESTING_DATASET)
    gestures_Z_train_dataset = dataset_recovering(TRAINING_DATASET)

    # Class labels
    unique_labels = np.unique(gestures_Z_test_dataset[:,0])
    print('Number of classes >> ', unique_labels.size)
    print('Max lenght time series >> ', gestures_Z_test_dataset.shape[1])
    print('Number instances testing set >> ', gestures_Z_test_dataset.shape[0])
    print('Number instances training set >> ', gestures_Z_train_dataset.shape[0])

    #
    # Processing of Dataset
    #-----------------------
    sax_training_dataset = dataset_processing(gestures_Z_test_dataset,max_possible_instances=True)
    sax_testing_dataset = dataset_processing(gestures_Z_train_dataset,max_possible_instances=True)
    print('SAX testing dataset length >> ', len(sax_testing_dataset))
    print('SAX training dataset length >> ', len(sax_training_dataset))

    #================
    # Classification
    #================
    print('<--- Starting the classification process --->')
    starting_time = time.time()
    predictions = SAX_DTW_F_SVC(sax_testing_dataset,sax_training_dataset,window_size=window_size)
    ending_time = time.time() - starting_time
    # Accuracy of the predictions
    results = {
        'labels' : [instance[0] for instance in sax_testing_dataset],
        'predictions' : predictions
    }
    acc = accuracy(results['labels'],results['predictions'])
    print(ps.DataFrame(results))
    print('<--- Finished with accuracy: {}, in {:.2f} sec --->'.format(acc,ending_time))

def main():
    #representative_ploting()

    #minimum_window_size()
    #prediction_performance_test()
    
    classifier,representatives = training()
    testing(classifier,representatives)

    train_save_model(classifier,representatives)

if __name__ == '__main__': main()