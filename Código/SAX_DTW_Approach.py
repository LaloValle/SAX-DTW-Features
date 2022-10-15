"""
    Methoud used in the paper "Gesture recognition using symbolic aggregate approximation and dynamic time warping on motion data"
    as an approach that combines the effectiveness and low complexity of SAX, with the insensitivity of DTW to speed fluctuation
    during the execution of a gesture.

    The method starts with the same arrangements in the preprocessing phase as with a original SAX method,
    but changes in the recognition phase where de 1-NN classifier uses the SAX-DTW distance to measure the similarity between instances.
    The preprocessing phase implies the follow steps:
        1. Normalize to zero mean and standard deviation of one
        2. Apply PAA to produce a new time series of fixed length
        3. Discretize using a predefined alphabet

    Cons
    ----
        - Even thought still a failrly low complexity algorithm, the combination of both mechanisms adds complexity
          in comparison to the baseline DTW-1NN
        
    Pros
    ----
        - Performs better than just using the DTW distance or the SAX distance alone as distance function in the 1-NN classifier
        - Because of the transformation of the data into words in a string, is a symbolic method that allows for humans to easily
          undertstand the new representation

    Parameters
    ----------
    The value of the parameters used in the article:
        number of words per axis    = 32
        number of alphabet symbols  = 7
"""
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt
# Local Libraries
from LLibraries.DTW import DTW_1NN
from LLibraries.SAX import *
from LLibraries.DTWWindowOptimizer import *
from LLibraries.Tools import stratified_sampling,add_label_column
from LLibraries.Tools import standardize


def plotting():
    #============================
    # Recovering of the datasets
    #============================
    gestures_Z_dataset = []
    number_words = 32; alphabet_size = 7
    numeric_alphabet = True
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
    #    It's necesary to work with a list of arrays for the time series which will be working with are of variable length
    #    thus, an structure as an array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling(gestures_Z_dataset,max_possible_instances=False,instances_per_class=20)
    print('Stratified dataset number instances >> ', len(stratified_dataset))
    print(stratified_dataset[0][:10])
    # Once the data has been stratified all the dataset needs to be standardize
    std_dataset = add_label_column(standardize([instance[1:] for instance in stratified_dataset]),[instance[0] for instance in stratified_dataset])
    print('Standardized dataset number instances >> ', len(std_dataset))

    #============================
    # PAA and SAX discretization
    #============================
    # First is test that achives it correctly with PAA
    paa_time_series_1 = PAA(std_dataset[0][1:],number_words)
    paa_time_series_2 = PAA(std_dataset[-1][1:],number_words)
    # paa_time_series_3 = PAA(std_dataset[2][1:],number_words)
    # paa_time_series_4 = PAA(std_dataset[-2][1:],number_words)
    print('PAA time series shape >> ', paa_time_series_1.shape, paa_time_series_2.shape)
    # Then the SAX representation
    #   It doesn't require to be already standardize
    sax_time_series_1 = SAX(stratified_dataset[0][1:],number_words=number_words,alphabet_size=alphabet_size,numeric_alphabet=numeric_alphabet)
    sax_time_series_2 = SAX(stratified_dataset[-1][1:],number_words=number_words,alphabet_size=alphabet_size,numeric_alphabet=numeric_alphabet)
    # sax_time_series_3 = SAX(stratified_dataset[2][1:],number_words=number_words,alphabet_size=alphabet_size,numeric_alphabet=numeric_alphabet)
    # sax_time_series_4 = SAX(stratified_dataset[-2][1:],number_words=number_words,alphabet_size=alphabet_size,numeric_alphabet=numeric_alphabet)
    print('SAX representation 1 >> ', sax_time_series_1)
    print('SAX representation 2 >> ', sax_time_series_2)
    # Distances of SAX
    sax_distance = SAX_distance(sax_time_series_1,sax_time_series_2,alphabet_size=alphabet_size)
    print('SAX distance >> ', sax_distance)
    sax_dtw_distance = SAX_DTW_distance(sax_time_series_1,sax_time_series_2,alphabet_size=alphabet_size)
    print('SAX-DTW distance >> ', sax_dtw_distance)


    #==========
    # Plotting
    #==========
    fig,axs = plt.subplots(2,2)
    #------------
    # Raw series
    #------------
    # Series 1
    axs[0][0].plot([x for x,_ in enumerate(stratified_dataset[0][1:])],stratified_dataset[0][1:])
    axs[0][0].scatter([x for x,_ in enumerate(stratified_dataset[0][1:])],stratified_dataset[0][1:])
    # Series 2
    axs[0][0].plot([x for x,_ in enumerate(stratified_dataset[-1][1:])],stratified_dataset[-1][1:])
    axs[0][0].scatter([x for x,_ in enumerate(stratified_dataset[-1][1:])],stratified_dataset[-1][1:])
    axs[0][0].set_title('Raw time series')
    #-------------------
    # Normalized series
    #-------------------
    # Series 1
    axs[0][1].plot([x for x,_ in enumerate(std_dataset[0][1:])],std_dataset[0][1:])
    axs[0][1].scatter([x for x,_ in enumerate(std_dataset[0][1:])],std_dataset[0][1:])
    # Series 2
    axs[0][1].plot([x for x,_ in enumerate(std_dataset[-1][1:])],std_dataset[-1][1:])
    axs[0][1].scatter([x for x,_ in enumerate(std_dataset[-1][1:])],std_dataset[-1][1:])
    axs[0][1].set_title('Standardized time series')
    #---------------------
    # Discrete PAA series
    #---------------------
    # Series 1
    axs[1][0].plot([x for x,_ in enumerate(paa_time_series_1)],paa_time_series_1)
    axs[1][0].scatter([x for x,_ in enumerate(paa_time_series_1)],paa_time_series_1)
    # Series 2
    axs[1][0].plot([x for x,_ in enumerate(paa_time_series_2)],paa_time_series_2)
    axs[1][0].scatter([x for x,_ in enumerate(paa_time_series_2)],paa_time_series_2)
    axs[1][0].set_title('PAA representation with number of words 32')
    #-------------------
    # SAX string series
    #-------------------
    # Series 1
    axs[1][1].plot([x for x,_ in enumerate(paa_time_series_1)],paa_time_series_1,alpha=0.25)
    axs[1][1].scatter([x for x,_ in enumerate(paa_time_series_1)],paa_time_series_1,alpha=0.25)
    # Series 2
    axs[1][1].plot([x for x,_ in enumerate(paa_time_series_2)],paa_time_series_2,alpha=0.25)
    axs[1][1].scatter([x for x,_ in enumerate(paa_time_series_2)],paa_time_series_2,alpha=0.25)
    # Series with breakpoints
    extended_breakpoints = [min([np.min(paa_time_series_1),np.min(paa_time_series_2)])] + equiprobable_breakpoints[str(alphabet_size)] + [max([np.max(paa_time_series_1),np.max(paa_time_series_2)])]
    mid_breakpoinst = []
    for i in range(len(extended_breakpoints)-1):
        mid_breakpoinst.append((extended_breakpoints[i]+extended_breakpoints[i+1])/2)
    # Series 1 breakpoints
    axs[1][1].plot([x for x,_ in enumerate(paa_time_series_1)],[mid_breakpoinst[int(value)] for value in sax_time_series_1],color='steelblue')
    axs[1][1].scatter([x for x,_ in enumerate(paa_time_series_1)],[mid_breakpoinst[int(value)] for value in sax_time_series_1],color='steelblue')
    # Series 2 breakpoints
    axs[1][1].plot([x for x,_ in enumerate(paa_time_series_2)],[mid_breakpoinst[int(value)] for value in sax_time_series_2],color='darkorange')
    axs[1][1].scatter([x for x,_ in enumerate(paa_time_series_2)],[mid_breakpoinst[int(value)] for value in sax_time_series_2],color='darkorange')
    # Breakpoints
    for bound in equiprobable_breakpoints[str(alphabet_size)]:
        plt.plot([0, len(paa_time_series_1)],[bound]*2,color='k')
    axs[1][1].set_title('SAX symbolic strings with number of words 32 and alphabet size 7')

    plt.show()



def test_minimum_window_size():
    #============================
    # Recovering of the datasets
    #============================
    gestures_Z_dataset = []
    number_words = 32; alphabet_size = 7
    numeric_alphabet = True
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
    #    It's necesary to work with a list of arrays for the time series which will be working with are of variable length
    #    thus, an structure as an array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling(gestures_Z_dataset,max_possible_instances=False,instances_per_class=20)
    print('Stratified dataset number instances >> ', len(stratified_dataset))
    synthetic_strat = add_warping(stratified_dataset[0][1:])
    # PAA for showing in plot
    paa_time_series = PAA(standardize(stratified_dataset[0][1:]),number_words)
    # SAX discretization of the whole dataset
    #   The class label gets added as the first element of the 
    sax_dataset = [np.concatenate(([int(instance[0])], SAX(instance[1:],number_words=number_words,alphabet_size=alphabet_size,numeric_alphabet=numeric_alphabet,array_style=True))) for instance in stratified_dataset]
    print('SAX dataset length >> ', len(sax_dataset))

    #================================
    # Test related to minimum window
    #================================
    # It's important to always indicate the data type as a float instead of an int
    # or the synthetic producto won't be but a 0 values array
    # Further with the ceil/floor/rint function is converted back to int
    synthetic_sax = add_warping(sax_dataset[0][1:],convert_int=True)
    sax_dtw_distance = SAX_DTW_distance(sax_dataset[0][1:],synthetic_sax,alphabet_size=alphabet_size,window_size=8)
    print('SAX DTW Distance >> ', sax_dtw_distance)
    # Generation of a new set
    new_dataset = create_augmented_set(sax_dataset,convert_int=True)
    print('New dataset length >> ', len(new_dataset))
    # Testing the classifier
    predictions = SAX_DTW_1NN(sax_dataset[:5],sax_dataset[5:],alphabet_size=alphabet_size,window_size=8)
    print('Predictions >> ', predictions)
    print('Test labesl >> ',[instance[0] for instance in sax_dataset[:5]])
    predictions = SAX_DTW_1NN(new_dataset[:5],new_dataset[5:],alphabet_size=alphabet_size,window_size=8)
    print('Predictions >> ', predictions)
    print('Test labesl >> ',[instance[0] for instance in new_dataset[:5]])
    # Cross validation
    acc = cross_validation(sax_dataset,SAX_DTW_1NN,window_size=8,alphabet_size=alphabet_size,verbose=True)

    #==========
    # Plotting
    #==========
    fig,axs = plt.subplots(2,2)
    #------------
    # Raw series
    #------------
    # Series 1
    axs[0][0].plot([x for x,_ in enumerate(stratified_dataset[0][1:])],stratified_dataset[0][1:])
    axs[0][0].scatter([x for x,_ in enumerate(stratified_dataset[0][1:])],stratified_dataset[0][1:])
    # Series 2
    axs[0][0].plot([x for x,_ in enumerate(synthetic_strat)],list(synthetic_strat))
    axs[0][0].scatter([x for x,_ in enumerate(synthetic_strat)],list(synthetic_strat))
    #-------------------
    # SAX string series
    #-------------------
    # Series 1
    axs[0][1].plot([x for x,_ in enumerate(paa_time_series)],paa_time_series)
    axs[0][1].scatter([x for x,_ in enumerate(paa_time_series)],paa_time_series)
    # Series 2
    axs[0][1].plot([x for x,_ in enumerate(sax_dataset[0][1:])],sax_dataset[0][1:])
    axs[0][1].scatter([x for x,_ in enumerate(sax_dataset[0][1:])],sax_dataset[0][1:])
    for bound in equiprobable_breakpoints[str(alphabet_size)]:
        axs[0][1].plot([0, len(paa_time_series)],[bound]*2,color='k')
    #---------------
    # Synthetic SAX
    #---------------
    # SAX
    axs[1][0].plot([x for x,_ in enumerate(sax_dataset[0][1:])],sax_dataset[0][1:])
    axs[1][0].scatter([x for x,_ in enumerate(sax_dataset[0][1:])],sax_dataset[0][1:])
    axs[1][0].plot([x for x,_ in enumerate(synthetic_sax)],synthetic_sax)
    axs[1][0].scatter([x for x,_ in enumerate(synthetic_sax)],synthetic_sax)
    #---------------
    # Synthetic SAX
    #---------------
    # SAX
    axs[1][1].plot([x for x,_ in enumerate(new_dataset[10][1:])],new_dataset[10][1:],color='k')
    axs[1][1].scatter([x for x,_ in enumerate(new_dataset[10][1:])],new_dataset[10][1:],color='k')


    plt.show()

def minimum_window_size():
    ###################################
    #   Value of the minimum window
    # Accuracy: 89.83%
    # window value: 2
    ###################################
    #============================
    # Recovering of the datasets
    #============================
    gestures_Z_dataset = []
    number_words = 32; alphabet_size = 7
    numeric_alphabet = True
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
    #    It's necesary to work with a list of arrays for the time series which will be working with are of variable length
    #    thus, an structure as an array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling(gestures_Z_dataset,max_possible_instances=False,instances_per_class=20)
    print('Stratified dataset number instances >> ', len(stratified_dataset))
    # SAX discretization of the whole dataset
    #   The class label gets added as the first element of the 
    sax_dataset = [np.concatenate(([int(instance[0])], SAX(instance[1:],number_words=number_words,alphabet_size=alphabet_size,numeric_alphabet=numeric_alphabet,array_style=True))) for instance in stratified_dataset]
    print('SAX dataset length >> ', len(sax_dataset))

    #======================
    # Minimum window width
    #======================
    acc = minimum_warping_window(sax_dataset,SAX_DTW_1NN,alphabet_size=alphabet_size,number_iterations=5,upper_bound_window=8,convert_int=True,verbose=True)

def prediction_performance_test():
    ##############################
    #   Performance predicition
    # Window value: 2
    # Testing set size: 150
    # Training set size: 120
    #
    # Accuracy: 90.33%
    # Time: 12.89 sec
    ##############################
    #============================
    # Recovering of the datasets
    #============================
    gestures_Z_test_dataset,gestures_Z_train_dataset = [],[]
    number_words = 32; alphabet_size = 7; window_size = 2
    numeric_alphabet = True
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
    stratified_training_dataset = stratified_sampling(gestures_Z_train_dataset,max_possible_instances=False,instances_per_class=20)
    stratified_testing_dataset = stratified_sampling(gestures_Z_test_dataset,max_possible_instances=False,instances_per_class=40)
    print('Stratified testing dataset number instances >> ', len(stratified_testing_dataset))
    print('Stratified training dataset number instances >> ', len(stratified_training_dataset))
    # SAX discretization of the whole dataset
    #   The class label gets added as the first element of the 
    sax_training_dataset = to_SAX_dataset(stratified_training_dataset,number_words,alphabet_size,numeric_alphabet)
    sax_testing_dataset = to_SAX_dataset(stratified_testing_dataset,number_words,alphabet_size,numeric_alphabet)
    print('SAX testing dataset length >> ', len(sax_testing_dataset))
    print('SAX training dataset length >> ', len(sax_training_dataset))

    #================
    # Classification
    #================
    print('<--- Starting the classification process --->')
    starting_time = time.time()
    predictions = SAX_DTW_1NN(sax_testing_dataset,sax_training_dataset,alphabet_size,window_size)
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
    #tests()
    #test_minimum_window_size()
    minimum_window_size()
    #prediction_performance_test()

if __name__ == '__main__': main()