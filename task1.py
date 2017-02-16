#!/usr/bin/python

import numpy #to use mathematical functions
import csv #to use csv reader and writer
import sys #to use the system functions
import random #to use the random generator and data shuffler

#############################################################################
######################## Main Program Definition ############################
#############################################################################

def main():
    #Test for the presence of training dataset
    if len(sys.argv) != 3:
        print "#Usage method: python ./task1_20166351 \
training_and_validation_data.csv task_or_evaluation_data_without_hands.csv"
        print "#Outputs: Guesses on hands for each row. These will be printed \
on an output file 'output_Task1'"
        print "#Firstline of all the input files will never be read!!"
        quit()

    #First, find the best weights and biases
    bestWeights,bestBiases = finding_best_weights_biases(sys.argv[1])

    #Read the file for the data on cards to guess
    dataToGuess = read_csv_file(sys.argv[2])
    
    #Main algo to load deck by deck and guess and record the hand
    open_file = open("output_Task1", "w")
    try:
            open_file.write("Hand\n")
    except:
            open_file.close()

    for (x,y) in dataToGuess:
        open_file.write(str(feed_forward_network_with_guess(\
x,bestWeights,bestBiases)) + "\n")
        
    open_file.close()

#############################################################################
############## Finding best methode to use the training data ################
#############################################################################

def finding_best_weights_biases(filename_training):
    #Initialisation
    input_data_official = read_csv_file(filename_training)
    best_accuracy = 0
                                     
    #Creating initial weights and biases randomly
    #weights and biases will be propagated through different models
    layers = [85, 50, 10] #numbers of neurons per layer
    bestWeights, bestBiases = create_weights_biases(layers)

    #
    # Methode 1:
    #------------
    # Splitting input file into diff ratio of training / testing(validation)
    # datas and test them with the whole initial input file
    #
    for ratio_split in [0.2, 0.4, 0.6, 0.8] :
        training_data,testing_data = read_split_data(filename_training\
                                                     , ratio_split)
        network_obj = neural_program(training_data,testing_data,\
                                     bestWeights, bestBiases, layers)
        accuracy = network_obj.testing_independently(input_data_official)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            bestWeights,bestBiases = network_obj.getBestParameters()

    #
    # Methode 2:
    #------------
    # Using the same data for training and testing
    #
    network_obj = neural_program(input_data_official, input_data_official,\
                                 bestWeights, bestBiases, layers)
    accuracy = network_obj.getBestAccuracy()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        bestWeights,bestBiases = network_obj. getBestParameters()

    return bestWeights,bestBiases

#############################################################################
################# Algorithm Part : Neural Network (Parameters) ##############
#############################################################################

def neural_program(training_data, testing_data, weights, biases, layers, \
                   debug=True):
    #
    # Initialisations of the main parameters
    #---------------------------------------------
    # Theses are the chosen main parametes
    # after few trials, these prove the sub-optimum
    #----------------------------------------------
    #
    learning_rate = 3.0
    size_of_partition = 10
    number_of_training = 30

    #Training and testing part
    network_obj = Network(training_data, testing_data, \
                          size_of_partition, learning_rate, weights, biases, \
                          len(layers))
    network_obj.training_testing(number_of_training, debug)
    
    return network_obj

#############################################################################
#################### Algorithm Part : Neural Network ########################
#############################################################################

#
# Model of neural network
#
class Network(object):
    #Initialisation of the neural network
    def __init__(self, training_data, testing_data, \
                 partition_size, learn_rate, weights, biases, num_layers):
        self.training_data = training_data
        self.testing_data = testing_data
        self.dataset_size = len(training_data)
        self.partition_size = partition_size
        self.learn_rate = learn_rate
        self.weights = weights
        self.biases = biases
        self.num_layers = num_layers
        self.best_weights = numpy.copy(weights)
        self.best_biases = numpy.copy(biases)
        self.best_accuracy = 0 #in percentage

    #Training and testing the neural network
    def training_testing(self, num_training, debug=False):
        for i in xrange(num_training):
            #Randomize the datas to inject randomness to our data 
            #which can increase the
            #accuracy of the algorithm (Give an ilustration of more data)
            random.shuffle(self.training_data)

            #Partitioning data for training part to increase the accuracy of \
            #algorithm as we focus on neighbouring changes 
            #(changes on small localized area)
            data_partitions = [self.training_data[j:j+self.partition_size] \
                                for j in xrange(0, self.dataset_size, \
                                                self.partition_size)]
            for data_partition in data_partitions:
                self.update_weights_biases(data_partition)
            
            #We keep track of the progress of testing with validation data
            score = self.testing()
            accuracy = ((1.0*score) / (1.0*len(self.testing_data))) * 100.0
            if debug == True:
                print "=>Training no-%i = %i/%i = %.2f%%" \
                    %(i+1,score,len(self.testing_data), accuracy)

            #We keep the best accuracy and the best weights and biases
            if accuracy > self.best_accuracy :
                self.best_accuracy = accuracy
                self.best_weights = numpy.copy(self.weights)
                self.best_biases = numpy.copy(self.biases)

            if debug == True:
                print "### Current best accuracy : %.2f%% ###\n" \
                    %(self.best_accuracy)

    #Compute outputs from inputs of a line of data through 
    #the multi-layer neural network
    def feed_forward_network(self,inputs):
        for weight,bias in zip(self.weights,self.biases):
            inputs=sigmoid_function(numpy.dot(weight,inputs)+bias,\
                                         deriv=False)
        return inputs

    #Main algorithm to calculate the gradients descent using backpropagation
    #line by line of data
    def backpropagation(self,x,y):
        #Initialisation of the lists to be returned
        layered_delta_weights = [numpy.zeros(weight.shape) \
                                 for weight in self.weights]
        layered_delta_biases = [numpy.zeros(bias.shape) \
                                for bias in self.biases]

        #Initialisation of the loop below
        activation = numpy.copy(x) #For each layer
        activations = [numpy.copy(x)] #For all the layers
        z_vectors = [] #List of z vectors of all layers

        #Feed forward to have all the activations firstly before
        #calculation of different deltas
        for weight,bias in zip(self.weights, self.biases):
            z_vector = numpy.dot(weight,activation)+bias
            z_vectors.append(z_vector)
            activation = sigmoid_function(z_vector, deriv=False)
            activations.append(activation)

        #Initialisation of the reverse loop below
        delta = self.compute_error(activations[-1], y) * \
                sigmoid_function(z_vectors[-1],deriv=True)
        layered_delta_weights[-1] = numpy.dot(delta, \
                                              numpy.transpose(activations[-2]))
        layered_delta_biases[-1] = delta
        
        #We calculate layer by layer the delta for weight and bias
        #and store them in the lists to be returned
        #Xrange is used for a faster calculation
        #in backward ordering by using negative indexes
        for layer in xrange(2, self.num_layers):
            z_vector = z_vectors[-layer]
            sigmoid_deriv =  sigmoid_function(z_vector,deriv=True)

            #Back-propagation of delta (the difference / changes) 
            #for weight and bias
            delta = numpy.dot(numpy.transpose(self.weights[-layer+1]), delta)* \
                    sigmoid_deriv
            layered_delta_weights[-layer] = numpy.dot(delta, numpy.transpose(\
activations[-layer-1]))
            layered_delta_biases[-layer] = delta
        return layered_delta_weights,layered_delta_biases

    #Update the layered weights and biases after algo backpropagation
    def update_weights_biases(self,partitioned_data):
        #Execute the backpropagation algo to a small partition of data
        #to find the delta / changes of weights and of biases 
        total_delta_weight = [numpy.zeros(weight.shape) for weight in \
                              self.weights]
        total_delta_bias = [numpy.zeros(bias.shape) for bias in self.biases]
        for (x,y) in partitioned_data:
            delta_weights, delta_biases = self.backpropagation(x,y)
            total_delta_weight = [w+dw for w, dw in \
                                  zip(total_delta_weight, delta_weights)]
            total_delta_bias = [b+db for b, db in \
                                zip(total_delta_bias, delta_biases)]

        #Update of the whole weights and biases
        #the most important part
        self.weights = [weight-(self.learn_rate/len(partitioned_data))* \
                        delta_weight \
                        for weight,delta_weight \
                        in zip(self.weights, total_delta_weight)] 
        self.biases = [bias-(self.learn_rate/len(partitioned_data))* \
                       delta_bias \
                       for bias,delta_bias \
                       in zip(self.biases, total_delta_bias)]

    #Calculate the error (the difference) between training and real outputs
    def compute_error(self, training_outputs, real_outputs):
        return training_outputs-real_outputs

    #Evaluate the model using the testing data by calculating the correct guess
    def testing(self):
        count = 0
        for (x, y) in self.testing_data : 
            test_result = self.feed_forward_network(x)
            # We look for the index of the maximum number in the array
            # as it represents the best output
            if numpy.argmax(test_result) == numpy.argmax(y):
                count += 1
        return count

    #Evaluate the model independently using the testing data in parameter 
    #and the best parameters of weight and bias
    #by calculating the correct guess
    def testing_independently(self, testing_data):
        self.weights = numpy.copy(self.best_weights)
        self.biases = numpy.copy(self.best_biases)
        count = 0
        for (x, y) in testing_data : 
            test_result = self.feed_forward_network(x)
            # We look for the index of the maximum number in the array
            # as it represents the best output (activation)
            if numpy.argmax(test_result) == numpy.argmax(y):
                count += 1
        return ((1.0*count)/(1.0*len(testing_data))) * 100.0
    
    #Return the best weights and best biases
    def getBestParameters(self):
        return self.best_weights, self.best_biases

    #Return the best accuracy
    def getBestAccuracy(self):
        return self.best_accuracy

#############################################################################
########################### Extra functions #################################
#############################################################################

#
# Create initial weights and biases for every single layer
# for time being, we use the method of generating them randomly (mean 0)
#
def create_weights_biases(neurons_sizes):
    weights = [numpy.random.randn(y,x)/numpy.sqrt(x) \
               for x,y in zip(neurons_sizes[:-1],neurons_sizes[1:])]
    biases = [numpy.random.randn(y,1) for y in neurons_sizes[1:]]
    return weights, biases

#
# Activation function to be used
#
def sigmoid_function(x, deriv=False):
    if (deriv==True):
        return (1.0/(1.0+numpy.exp(-x)))*(1.0-(1.0/(1.0+numpy.exp(-x))))
    else:
        return 1.0/(1.0+numpy.exp(-x))

#
# Feed forward network for the guessing part
#
def feed_forward_network_with_guess(inputs,weights,biases):
        for weight,bias in zip(weights,biases):
            inputs=sigmoid_function(numpy.dot(weight,inputs)+bias,\
                                         deriv=False)
        return numpy.argmax(inputs)

#############################################################################
########################### Extraction data Part ############################
#############################################################################

#
# Dictionaries of inputs and outputs
#
suit_tup = (numpy.array([[1],[0],[0],[0]]),\
            numpy.array([[0],[1],[0],[0]]),\
            numpy.array([[0],[0],[1],[0]]),\
            numpy.array([[0],[0],[0],[1]]))

rank_tup = (numpy.array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]]))

hand_tup = (numpy.array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[1],[0],[0],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[1],[0],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[1],[0],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[1],[0],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[1],[0],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[0],[1],[0],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[0],[0],[1],[0]]),\
            numpy.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]]))         

#
# To classify suits using binary
#
def classify_suit(suit):
    suit_offset = int(suit)-1
    if suit_offset > -1 and suit_offset < 4 :
        return suit_tup[suit_offset]
    else :
        print "ERROR : incorrect value in input file"
        quit()

#
# To classify ranks using binary
#
def classify_rank(rank):
    rank_offset = int(rank)-1
    if rank_offset > -1 and rank_offset < 13 :
        return rank_tup[rank_offset]
    else :
        print "ERROR : incorrect value in input file"
        quit()

#
# To classify hands using binary
#
def classify_hand(hand):
    hand_int = int(hand)
    if hand_int > -1 and hand_int < 10 :
        return hand_tup[hand_int]
    else :
        print "ERROR : incorrect value in input file"
        quit()

#
# Read a line of dataset for extraction of information on inputs and outputs
#
def read_line_dataset(column):
    if len(column) == 11 or len(column) == 10:
        cards = []
        
        #Accumulating the information on inputs
        cards = numpy.concatenate([classify_suit(column[0]),\
                                   classify_rank(column[1])])
        for counter in xrange(1,5):
            new_cards = numpy.concatenate([classify_suit(column[counter*2]),\
                                   classify_rank(column[counter*2+1])])
            cards = numpy.concatenate([cards, new_cards])
        
        #Extract the information on hand
        if len(column) == 11:
            hand = classify_hand(column[10])
        elif len(column) == 10:
            hand = 0 #absence of hand, so by default 0
        
        return cards,hand

#
# Reading csv file based on the input filename
# and extracting the inputs and outputs
#
def read_csv_file(filename):
    f = open(filename, "rt")
    try:
        #skip the first line (line where the description of column of data)
        f.readline()
        
        csv_reader = csv.reader(f)
        
        #list of paires of inputs and outputs
        line = []

        #Reading line by line the input file to find 85 card inputs (suit,rank)
        #and 10 card outputs (hand)
        #we accumulate the pair of inputs and the outputs
        for row in csv_reader:
            line_input,line_output = read_line_dataset(row)
            line.append(((line_input,) + (line_output,)))

        return line
    finally:
        f.close()

#
# Reading csv file based on the input filename
# and extracting the inputs and outputs
# training and testing data according to ratio (splitting)
#
def read_split_data(filename, ratio):
    #calculate the total number of lines and find the split point
    num_lines = sum(1 for line in open(filename))
    split_point = int(ratio*num_lines)
    
    f = open(filename, "rt")
    try:
        #skip the first line (line where the description of column of data)
        f.readline()
        
        csv_reader = csv.reader(f)
        
        #list of paires of inputs and outputs
        line = []
        counter = 0

        #Reading line by line the input file to find 85 card inputs (suit,rank)
        #and 10 card outputs (hand)
        #we accumulate the pair of inputs and the outputs for training data
        #and for testing data
        for row in csv_reader:
            if counter == split_point :
                lines_training_set = line
                line = []
            line_input,line_output = read_line_dataset(row)
            line.append(((line_input,) + (line_output,)))
            counter+=1
        lines_testing_set = line

        return lines_training_set,lines_testing_set
    finally:
        f.close()

#############################################################################
########################## Main Program Called ##############################
#############################################################################
if __name__ == '__main__':
    main()
