# Neural networks for hand of cards
Classification of hand of cards using neural network (machine learning &amp; big data) and identification of rule of cards

##
###Descriptions
- Task1.py is for classifying hand of cards
- Task2.py uses Task1.py to change one card in order to maximise the hand of 5-cards
- Neural network is applied in a batch manner
- Used algorithm : Multi-layered artificial neural network with backpropagation method
- The parameters which we use for the model are the following :
  - Input neurons = 85 and Output neurons = 10 (as we code the input cards and hands in binary)
  - 3 layers with one hidden layer with 50 neurons to increase the activations thus accuracy
  - Learning rate = 3.0 (in order to converge steadily with a good accuracy)
  - Size of partition of data = 10 (10 line of data inputs and outputs ; so that neighbouring changes could take place smoothly)

##
There are still bugs to be resolved, and improvements to be done.
