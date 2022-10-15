from re import X
import numpy as np
import matplotlib.pyplot as plt
# Local Libraries
from LLibraries.SAX import *
from LLibraries.Tools import stratified_sampling,add_label_column,group_by_class,standardize
from LLibraries.DTWFeatures import representatives_by_class

#------------
# Parameters
#------------
number_words = 64; alphabet_size = 9
numeric_alphabet = True
window_size = 4

def representatives_ploting():
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
    global number_words, alphabet_size, numeric_alphabet
    #============================
    # Recovering of the datasets
    #============================
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
    #Class labels
    unique_labels = np.unique(gestures_Z_dataset[:,0])
    print('Number of classes >> ', unique_labels.size)
    print('Number instances dataset >> ', gestures_Z_dataset.shape[0])
    print('Max lenght time series >> ', gestures_Z_dataset.shape[1])

    #=======================
    # Processing of Dataset
    #=======================
    # The dataset gets stratified and returns as a list of arrays of potentially variable lengths
    #    It's necesary to work with a list of arrays for the time series are of variable length thus, an structure as an
    #    array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling( 
            gestures_Z_dataset,
            max_possible_instances=True,
            instances_per_class=instances_per_class
        )
    print('Stratified dataset number instances >> ', len(stratified_dataset))
    # SAX discretization of the whole dataset
    #   The class label gets added as the first element of the 
    sax_dataset = to_SAX_dataset(stratified_dataset, number_words, alphabet_size, numeric_alphabet)
    print('\nSAX dataset length >> ', len(sax_dataset))
    # Grouping and splitting the dataset by class labels
    dataset_by_classes = group_by_class(sax_dataset)
    print('Classes >> ', list(dataset_by_classes.keys()))

    #======================================
    # Barycenter Averaging Representatives
    #======================================
    # The representatives by class are computed using DTW Barycenter Averaging
    representatives = representatives_by_class(dataset_by_classes)

    #==========
    # Plotting
    #==========
    fig,axs = plt.subplots(3,2)
    #---------------------
    # SAX series by class
    #---------------------
    labels = list(dataset_by_classes.keys()); labels.sort()
    for label in labels:
        for index in range(instances_per_class):
            axs[int((label+1)/2)-1][1-(label%2)].set_title('Clase {}'.format(label))
            #   Black color for comparing with representative
            axs[int((label+1)/2)-1][1-(label%2)].plot([x for x,_ in enumerate(dataset_by_classes[label][index])],dataset_by_classes[label][index],'--',linewidth=0.5,color='k',alpha=0.5)
            #   One color for each time series
            # axs[int((label+1)/2)-1][1-(label%2)].plot([x for x,_ in enumerate(dataset_by_classes[label][index])],dataset_by_classes[label][index])
        #   Representative of the class
        axs[int((label+1)/2)-1][1-(label%2)].plot([x for x,_ in enumerate(representatives[label])],representatives[label],linewidth=2.5,color='r')
    
    fig.suptitle('Series en representación SAX por clase')    
    fig.suptitle('Comparación de las representaciones SAX del entrenamiento de series de tiempo y representantes por clase')
    fig.supxlabel('Mediciones en unidades de tiempo')
    fig.supylabel('Aceleración en el eje Z')

    plt.show()

def representative_ploting():
    """Function that plots the representative time serie of 1 class
    
    Procedure
    ---------
        1.  First the time series of the Pebbble Gesture file gets recovered. This file only includes the measurements
            of each movements in the Z axis.
        2.  The recovered dataset gets processed
            2.1 First the dataset gets stratified so that each class has the same number of time series samples
            2.2 Then all the time series gets transformed into their SAX representation
        3.
    """
    global number_words, alphabet_size, numeric_alphabet
    #============================
    # Recovering of the datasets
    #============================
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
    #Class labels
    unique_labels = np.unique(gestures_Z_dataset[:,0])
    print('Number of classes >> ', unique_labels.size)
    print('Number instances dataset >> ', gestures_Z_dataset.shape[0])
    print('Max lenght time series >> ', gestures_Z_dataset.shape[1])

    #=======================
    # Processing of Dataset
    #=======================
    # The dataset gets stratified and returns as a list of arrays of potentially variable lengths
    #    It's necesary to work with a list of arrays for the time series are of variable length thus, an structure as an
    #    array cannot work with arrays in a higher dimension with different sizes on them
    stratified_dataset = stratified_sampling( 
            gestures_Z_dataset,
            max_possible_instances=True,
            instances_per_class=instances_per_class
        )
    print('Stratified dataset number instances >> ', len(stratified_dataset))
    # SAX discretization of the whole dataset
    #   The class label gets added as the first element of the 
    sax_dataset = to_SAX_dataset(stratified_dataset, number_words, alphabet_size, numeric_alphabet)
    print('\nSAX dataset length >> ', len(sax_dataset))
    # Grouping and splitting the dataset by class labels
    dataset_by_classes = group_by_class(sax_dataset)
    print('Classes >> ', list(dataset_by_classes.keys()))

    #======================================
    # Barycenter Averaging Representatives
    #======================================
    # The representatives by class are computed using DTW Barycenter Averaging
    representatives = representatives_by_class(dataset_by_classes)

    #==========
    # Plotting
    #==========
    #---------------------
    # SAX series by class
    #---------------------
    labels = list(dataset_by_classes.keys()); labels.sort()
    for label in labels:
        for index in range(instances_per_class):
            plt.suptitle('Clase {}'.format(label))
            #   Black color for comparing with representative
            plt.plot([x for x,_ in enumerate(dataset_by_classes[label][index])],dataset_by_classes[label][index],'--',linewidth=0.5,color='k',alpha=0.5)
            #   One color for each time series
            # plt.plot([x for x,_ in enumerate(dataset_by_classes[label][index])],dataset_by_classes[label][index])
        #   Representative of the class
        plt.plot([x for x,_ in enumerate(representatives[label])],representatives[label],linewidth=2.5,color='r')
        break
    
    plt.title('Series en representación SAX por clase')
    plt.xlabel('Mediciones en unidades de tiempo')
    plt.ylabel('Aceleración en el eje Z')

    plt.show()

def SAX_representations():
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
    axs[0][0].plot([x for x,_ in enumerate(stratified_dataset[0][1:])],stratified_dataset[0][1:],color='darkcyan')
    #axs[0][0].scatter([x for x,_ in enumerate(stratified_dataset[0][1:])],stratified_dataset[0][1:])
    # Series 2
    axs[0][0].plot([x for x,_ in enumerate(stratified_dataset[-1][1:])],stratified_dataset[-1][1:],'--',color='darkorange')
    #axs[0][0].scatter([x for x,_ in enumerate(stratified_dataset[-1][1:])],stratified_dataset[-1][1:])
    axs[0][0].set_title('Series de tiempo originales')
    axs[0][0].set_xlabel('Unidades de tiempo')
    axs[0][0].set_ylabel('Aceleración en el eje Z')
    #-------------------
    # Normalized series
    #-------------------
    # Series 1
    axs[0][1].plot([x for x,_ in enumerate(std_dataset[0][1:])],std_dataset[0][1:],color='darkcyan')
    #axs[0][1].scatter([x for x,_ in enumerate(std_dataset[0][1:])],std_dataset[0][1:])
    # Series 2
    axs[0][1].plot([x for x,_ in enumerate(std_dataset[-1][1:])],std_dataset[-1][1:],'--',color='darkorange')
    #axs[0][1].scatter([x for x,_ in enumerate(std_dataset[-1][1:])],std_dataset[-1][1:])
    axs[0][1].set_title('Series de tiempo estandarizadas')
    axs[0][1].set_xlabel('Unidades de tiempo')
    axs[0][1].set_ylabel('Aceleración en el eje Z')
    #---------------------
    # Discrete PAA series
    #---------------------
    # Series 1
    axs[1][0].plot([x for x,_ in enumerate(paa_time_series_1)],paa_time_series_1,color='darkcyan')
    #axs[1][0].scatter([x for x,_ in enumerate(paa_time_series_1)],paa_time_series_1)
    # Series 2
    axs[1][0].plot([x for x,_ in enumerate(paa_time_series_2)],paa_time_series_2,'--',color='darkorange')
    #axs[1][0].scatter([x for x,_ in enumerate(paa_time_series_2)],paa_time_series_2)
    axs[1][0].set_title('Representaciones PAA con número de palabras igual a 32')
    axs[1][0].set_ylabel('Aceleración en el eje Z')
    axs[1][0].set_xlabel('Unidades en palabras')
    #-------------------
    # SAX string series
    #-------------------
    # Series 1
    #axs[1][1].plot([x for x,_ in enumerate(paa_time_series_1)],paa_time_series_1,alpha=0.25,color='darkcyan')
    ##axs[1][1].scatter([x for x,_ in enumerate(paa_time_series_1)],paa_time_series_1,alpha=0.25)
    # Series 2
    #axs[1][1].plot([x for x,_ in enumerate(paa_time_series_2)],paa_time_series_2,alpha=0.25,color='darkorange')
    ##axs[1][1].scatter([x for x,_ in enumerate(paa_time_series_2)],paa_time_series_2,alpha=0.25)
    # Series with breakpoints
    extended_breakpoints = [min([np.min(paa_time_series_1),np.min(paa_time_series_2)])] + equiprobable_breakpoints[str(alphabet_size)] + [max([np.max(paa_time_series_1),np.max(paa_time_series_2)])]
    mid_breakpoinst = []
    for i in range(len(extended_breakpoints)-1):
        mid_breakpoinst.append((extended_breakpoints[i]+extended_breakpoints[i+1])/2)
    # Series 1 breakpoints
    axs[1][1].plot([x for x,_ in enumerate(paa_time_series_1)],[mid_breakpoinst[int(value)] for value in sax_time_series_1],color='darkcyan')
    axs[1][1].scatter([x for x,_ in enumerate(paa_time_series_1)],[mid_breakpoinst[int(value)] for value in sax_time_series_1],color='darkcyan')
    # Series 2 breakpoints
    axs[1][1].plot([x for x,_ in enumerate(paa_time_series_2)],[mid_breakpoinst[int(value)] for value in sax_time_series_2],'--',color='darkorange')
    axs[1][1].scatter([x for x,_ in enumerate(paa_time_series_2)],[mid_breakpoinst[int(value)] for value in sax_time_series_2],color='darkorange')
    # Breakpoints
    for bound in equiprobable_breakpoints[str(alphabet_size)]:
        plt.plot([0, len(paa_time_series_1)],[bound]*2,color='k')
    axs[1][1].set_title('Cadenas simbólicas SAX con número de palabras igual a 32 y alfabeto de tamaño 7')
    axs[1][1].set_ylabel('Representación alfabética de la aceleración en Z')
    axs[1][1].set_xlabel('Unidades en palabras')
    plt.show()

def synthetic_copys():
    from LLibraries.DTWWindowOptimizer import add_warping
    #============================
    # Recovering of the datasets
    #============================
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
    stratified_dataset = stratified_sampling(gestures_Z_dataset,max_possible_instances=False,instances_per_class=1)
    print('Stratified dataset number instances >> ', len(stratified_dataset))

    #==========
    # Plotting
    #==========
    x,y = [x for x,_ in enumerate(stratified_dataset[0][1:])], stratified_dataset[0][1:]
    plt.plot(x,y,color='k')

    #x_synth,y_synth = enumerate(add_warping([stratified_dataset[0]]))
    synthetic = add_warping(stratified_dataset[0][1:])
    x_synth, y_synth = [x for x,_ in enumerate(synthetic)], synthetic

    plt.plot(x_synth,y_synth,"--",color='k')

    plt.title('Comparación entre una serie de tiempo y su copia sintética')
    plt.xlabel('Unidades de tiempo')
    plt.ylabel('Aceleración en eje Z')

    plt.show()

def paa_visualization():
    words = 9

    series_1 = [2,2.5,2.8,7,9.2,8.8,7.4,5.8,5.7,3.5,4.2,3.5,1.7,1.3,1]
    series_2 = [0.6,1,2.3,3.7,3,4.1,3.8,4,5.6,6.4,6.2,7.2,8.5,8.6,3.3,3.2,0.9,0.6,0.3]

    paa_1 = PAA(series_1,words)
    paa_2 = PAA(series_2,words)

    fig,axis = plt.subplots(2)

    axis[0].plot([x for x,_ in enumerate(series_1)],series_1,color='darkcyan')
    axis[0].scatter([x for x,_ in enumerate(series_1)],series_1,color='darkcyan')
    axis[0].plot([x for x,_ in enumerate(series_2)],series_2,'--',color='darkorange')
    axis[0].scatter([x for x,_ in enumerate(series_2)],series_2,color='darkorange')
    axis[0].set_title('Series de tiempo originales')

    axis[1].plot([x for x,_ in enumerate(paa_1)],paa_1,color='darkcyan')
    axis[1].scatter([x for x,_ in enumerate(paa_1)],paa_1,color='darkcyan')
    axis[1].plot([x for x,_ in enumerate(paa_2)],paa_2,'--',color='darkorange')
    axis[1].scatter([x for x,_ in enumerate(paa_2)],paa_2,color='darkorange')
    axis[1].set_title('Representaciones PAA con número de palabras igual a 9')
    
    fig.supxlabel('Mediciones en unidades de tiempo')
    fig.supylabel('Aceleración en el eje Z')
    plt.show()

def main():
    representatives_ploting()
    #representative_ploting()
    #SAX_representations()
    #synthetic_copys()
    #paa_visualization()


if __name__ == '__main__': main()