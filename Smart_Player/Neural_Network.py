'''
Creating a Neural Network that can play a simple runner game
'''
from Runner import *
import numpy as np


np.random.seed(0)

import time
from threading import Thread


# Neural Network Class
class Neural_Network:
    def __init__(self):
        self.runner = Runner() #Runner Game we will be using
        #Hyperparameters
        self.input = 3
        self.middle = 3
        self.output = 3
        self.learning_rate = 0 #Will be set by the user
        # In the output the player can go up or down
        #Weights for each layer based on the Hyperparameters
        #Weights are initialized to random rather than 0
        self.weights_1 = np.random.randn(self.input, self.middle)
        self.weights_2 = np.random.randn(self.middle,self.output)
        #Incase you wanna look at the weights and compare them
        print(self.weights_1)
        print(self.weights_2)

    def forward(self, x):
        # input_x = copy.deepcopy(x)
        self.input_layer = self.sigmoid(x)
        self.hidden_layer = self.sigmoid(np.matmul(x,self.weights_1))
        #return self.hidden_layer
        self.output_layer = self.sigmoid(np.matmul(self.hidden_layer,self.weights_2))
        return self.output_layer

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def sigmoidPrime(self, value):
        return value * (1 - value)

    def backward(self, given_input, desired_output, predicted_output):
        # Backward Propagation Function
        # New weight = old weight — Derivative Rate * learning rate
        error = desired_output - predicted_output
        squared_error = np.square(error)
        d_rate_2 = self.sigmoidPrime(squared_error)
        weight1_error = d_rate_2 - self.hidden_layer
        squared_error_weight_1 = np.square(weight1_error)
        d_rate = self.sigmoidPrime(squared_error_weight_1)
        self.weights_1 = self.weights_1 - self.learning_rate*d_rate*self.hidden_layer
        self.weights_2 = self.weights_2 - d_rate_2 * self.learning_rate

    def train(self, target, rate):
        # Run the game on different thread inorder to prevent freezing
        self.learning_rate = rate # Set the learning rate
        game_thread = Thread(target=self.runner.game_start)
        game_thread.setDaemon(True)
        game_thread.start()
        # This while loop is to skip the intro
        while not self.runner.intro:
            print("wait for the game to start")
            self.runner.intro = False
            break
            time.sleep(1)
            input_x = np.array(
                [self.runner.player_pos_x, self.runner.player_pos_y, self.runner.obst.y])  # ,self.runner.obst.y])
        # Wait while the game is loaded
        while self.runner.exitGame:
            time.sleep(.5)
        # Now the game is loaded so we can use the neural network
        self.runner.score = 0
        while target > self.runner.score:
            if self.runner.obst.x == None:
                time.sleep(2)
            #Create the input for NN
            input_x = np.array([self.runner.player_pos_y, self.runner.player_pos_y - self.runner.obst.y, self.runner.window_height - self.runner.player_pos_y])
            #Save the input to evaluate the choice made by NN later
            player_x = self.runner.player_pos_x
            obstacle_x = self.runner.obst.x
            player_y = self.runner.player_pos_y
            obstacle_y = self.runner.obst.y
            #Option will be use to print the choice that the NN makes
            option = ""
            # Forward Propagation := Making the decision to move up, down or stay the same
            output = self.forward(input_x)
            output_y = np.array(
                [self.runner.player_pos_x, self.runner.player_pos_y, self.runner.obst.x, self.runner.obst.y])
            print("Output: ", )
            print(output)
            maxed = max(output)
            if maxed == output[0]:
                print("Option A")
                option = "A"
                self.runner.move_up()
                output_y = np.array(
                    [self.runner.player_pos_x, self.runner.player_pos_y, self.runner.obst.y])
            elif maxed == output[1]:
                print("Option B")
                option = "B"
            elif maxed == output[2]:
                print("Option C")
                option = "C"
                self.runner.move_down()
                output_y = np.array([self.runner.player_pos_x, self.runner.player_pos_y, self.runner.obst.y])
            else:
                print("Default Option")
                continue
            # What should our Y be in order for this to work out perfectly??
            y = [0, 0, 0]
            #Reference Window Height and Width
            #self.window_height = 600
            #self.window_width = 800
            if obstacle_y == player_y:
                if player_y > 400:
                    y= [1,0,0]
                else:
                    y = [0,0,1]
            elif obstacle_y + self.runner.obst.radius > player_y:
                y = [1, 0, 0]
            elif obstacle_y - self.runner.obst.radius <= player_y:
                y = [0, 0, 1]
            else:
                y = [0, 1, 0]
            if self.runner.obst.y - self.runner.obst.radius <= player_y:
                y = [0, 0, 1]
            if self.runner.obst.y + self.runner.obst.radius >= player_y:
                y = [1, 0, 0]
            if player_y + 50 >= self.runner.window_height:
                y = [1, 0, 0]
            if player_y - 50 <= 0:
                y = [0, 0, 1]
            if self.runner.crashed:
                if option == "A":
                    y = [0, 0, 1]
                if option == "B":
                    y = [1, 0, 1]
                if option == "C":
                    y = [1, 0, 0]
                    # self.runner.game_loop()
                    if target < self.runner.score:
                        break
            if output.all() == 0.5:
                y=[1,0,0]
            time.sleep(2)
            # Backward Propogation to make the neural network learn
            self.backward(input_x, y, output)
            print(self.runner.score)
        self.saveWeights() # Save the weights after the network is trained

    def saveWeights(self):
        file_weights1 = open("weights_1.txt", "w")
        file_weights1.write(str(self.weights_1))
        file_weights2 = open("weights_2.txt", "w")
        file_weights2.write(str(self.weights_2))

    def predict(self,target):
        print("Still in Progress")
        # The weights are from previously ran neural networks that worked.
        self.weights_1 = [[ 0.98613231,0.07185029,0.21640466],
                          [ 1.46297317,1.53925107,-1.7396112 ],
                          [ 0.17216838,-0.47966412,-0.86555217]]
        self.weights_2 = [[-0.27489529, -0.77178381,  0.58154682],
                          [ 0.07554394, -0.79415236, -0.42886345],
                          [-0.35181946,  0.57825169, -1.07788495]]
        # Same code as in train function but we dont save the weights at the end, and we dont call the back prop function
        game_thread = Thread(target=self.runner.game_start)
        game_thread.setDaemon(True)
        game_thread.start()
        # This while loop is to skip the intro #Intro has been disabled for now.
        while not self.runner.intro:
            print("wait for the game to start")
            self.runner.intro = False
            break
            time.sleep(1)
            input_x = np.array(
                [self.runner.player_pos_x, self.runner.player_pos_y, self.runner.obst.y])  # ,self.runner.obst.y])
        # Wait while the game is loaded
        while self.runner.exitGame:
            time.sleep(.5)
        # Now the game is loaded so we can use the neural network
        self.runner.score = 0
        while target > self.runner.score:
            if self.runner.obst.x == None:
                time.sleep(2)
            input_x = np.array([self.runner.player_pos_y, self.runner.player_pos_y - self.runner.obst.y,
                                self.runner.window_height - self.runner.player_pos_y])
            player_x = self.runner.player_pos_x
            obstacle_x = self.runner.obst.x
            player_y = self.runner.player_pos_y
            obstacle_y = self.runner.obst.y
            option = ""
            # Forward Propagation := Making the decision to move up, down or stay the same
            output = self.forward(input_x)
            output_y = np.array(
                [self.runner.player_pos_x, self.runner.player_pos_y, self.runner.obst.x, self.runner.obst.y])
            print("Output: ", )
            print(output)
            maxed = max(output)
            if maxed == output[0]:
                print("Option A")
                option = "A"
                self.runner.move_up()
                output_y = np.array(
                    [self.runner.player_pos_x, self.runner.player_pos_y, self.runner.obst.y])  # ,self.runner.obst.y])
                # time.sleep(1)
            elif maxed == output[1]:
                print("Option B")
                option = "B"
                # time.sleep(1)
            elif maxed == output[2]:
                print("Option C")
                option = "C"
                self.runner.move_down()
                output_y = np.array([self.runner.player_pos_x, self.runner.player_pos_y, self.runner.obst.y])
            else:
                print("Default Option")
                # time.sleep(1)
                continue
            time.sleep(2)
            print(self.runner.score)