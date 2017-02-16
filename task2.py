#!/usr/bin/python

import task1_20166351 as task1
import numpy
import sys

#############################################################################
######################## Main Program Definition ############################
#############################################################################

def main():
    #Test for the presence of training dataset
    if len(sys.argv) != 3:
        print "#Usage methode: python ./task2_20166351 \
training_and_validation_data.csv task_or_evaluation_data.csv"
        print "#Outputs: Guesses on modified card for better hand for each row.\
These will be printed on an output file 'output_Task2' with the modified desk \
and the hand"
        print "#Firstline of all the input files will never be read!!"
        quit()

    #First, find the best weights and biases with the help of task 1
    bestWeights,bestBiases = task1.finding_best_weights_biases(sys.argv[1])

    #Read the file for the cards to modify
    dataToModify = task1.read_csv_file(sys.argv[2])
    
    #Main algo to load deck by deck and check card by card for a good
    #replacement of card
    open_file = open("output_Task2", "w")
    try:
            open_file.write("S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,Hand\n")
    except:
            open_file.close()
    
    for (x,y) in dataToModify:
        objet_dect = Deck(x,y,bestWeights,bestBiases)
        objet_dect.modify_cards()
        objet_dect.register_cards(open_file)

    open_file.close()

#############################################################################
#################### Algorithm Part : Naive approach ########################
#############################################################################

#
# Model of a deck of 5 cards
#
class Deck(object):
    #The initialisation
    def __init__(self, cards, hand, bestWeights, bestBiases):
        self.original_cards = cards
        self.current_cards = numpy.copy(cards)
        self.best_cards = numpy.copy(cards)
        self.best_hand = numpy.argmax(hand)
        self.weights = bestWeights
        self.biases = bestBiases
    
    def sigmoid_function(self, x, deriv=False):
        if (deriv==True):
            return (1.0/(1.0+numpy.exp(-x)))*(1.0-(1.0/(1.0+numpy.exp(-x))))
        else:
            return 1.0/(1.0+numpy.exp(-x))

    def feed_forward_network(self, inputs):
        for weight,bias in zip(self.weights,self.biases):
            inputs=self.sigmoid_function(numpy.dot(weight,inputs)+bias,\
                                         deriv=False)
        return inputs

    #Evaluate the whole 5 cards if it has a better score
    def evaluate_cards_by_nnetwork(self):
        results = self.feed_forward_network(self.current_cards)
        new_hand_value = numpy.argmax(results)
        if new_hand_value > self.best_hand:
            self.best_hand = new_hand_value
            self.best_cards = numpy.copy(self.current_cards)

    #Modify one by one posibility of a card and evaluate
    def modify_one_card(self,card_position):
        #Copy the reference to the current card
        current_card = self.current_cards[(17*card_position):
                                          (17*(card_position+1))]
        old_suit_pos = numpy.argmax(current_card[0:4])
        old_rank_pos = numpy.argmax(current_card[4:17])

        #Looking one by one the possibilty of a card (52 possibilities)
        current_card[old_suit_pos][0] = 0
        current_card[old_rank_pos][0] = 0
        for i in xrange(0,4):
            for j in xrange(4,17):
                if i==old_suit_pos and j==old_rank_pos:
                    continue
                else:
                    current_card[i][0] = 1
                    current_card[j][0] = 1
                    self.evaluate_cards_by_nnetwork()
                    current_card[i][0] = 0
                    current_card[j][0] = 0
        #Revert to initial state for other cards #as only one is allowed
        self.current_cards = numpy.copy(self.original_cards)

    #Modify one by one card and evaluate
    def modify_cards(self):
        for i in xrange(5):
            self.modify_one_card(i)
    
    #Convert the binary best cards to numbered version in string
    def binary_to_numbers(self):
        numbered_cards = ""
        start = 0
        end = 0
        #for every card to try to find its corresponding number by looking
        #at the indice of the biggest element of the current array
        for i in xrange(5):
            start = end
            end += 4
            current_suit = self.best_cards[start:end]
            numbered_cards = numbered_cards + str(numpy.argmax(current_suit)+1)
            start = end
            end += 13
            current_rank = self.best_cards[start:end]
            numbered_cards = numbered_cards + "," + \
                             str(numpy.argmax(current_rank)+1)
            numbered_cards = numbered_cards + ","
        return numbered_cards

    #Saving the current cards info and hand to the opened file
    #by appending a line to it
    def register_cards(self, open_file):
        try:
            open_file.write(self.binary_to_numbers())
            open_file.write(str(self.best_hand))
            open_file.write("\n")
        except:
            open_file.close()
            
#############################################################################
########################## Main Program Called ##############################
#############################################################################
if __name__ == '__main__':
    main()
