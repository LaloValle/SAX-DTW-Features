import numpy as np
import matplotlib.pyplot as plt
# DTW algorithm and visualization
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
# Local Libraries
from LLibraries.SAX import *
from LLibraries.Tools import stratified_sampling,add_label_column
from LLibraries.Tools import standardize

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
print('PAA time series shape >> ', paa_time_series_1.shape, paa_time_series_2.shape)
# Then the SAX representation
#   It doesn't require to be already standardize
sax_time_series_1 = SAX(stratified_dataset[0][1:],number_words=number_words,alphabet_size=alphabet_size,numeric_alphabet=numeric_alphabet)
sax_time_series_2 = SAX(stratified_dataset[-1][1:],number_words=number_words,alphabet_size=alphabet_size,numeric_alphabet=numeric_alphabet)

# DTW algorithm
path = dtw.warping_path([int(digit) for digit in sax_time_series_1],[int(digit) for digit in sax_time_series_2])
# Visualisation of the optimal correspondence between series
figure,ax = dtwvis.plot_warping([int(digit) for digit in sax_time_series_2], [int(digit) for digit in sax_time_series_2], path)
figure.suptitle('Optimal warping path between symbolic SAX strings')
figure.supxlabel('Words segments')
figure.supylabel('Numeric symbols')
plt.show()