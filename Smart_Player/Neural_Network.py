'''
Creating a Neural Network that can play a simple runner game
'''
from Runner import *
#import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#import sklearn
#import sklearn.datasets
#import sklearn.linear_model
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.metrics import accuracy_score

np.random.seed(0)

import time
from threading import Thread


# Neural Network Class
class Neural_Network:
    def __init__(self):
        self.runner = Runner()
        self.input = 3
        self.middle = 3
        self.output = 3
        self.learning_rate = 0
        # In the output the player can go up or down
        self.weights_1 = np.random.randn(self.middle, self.input)
        self.weights_2 = np.random.randn(self.output,self.middle)
        #weight_fileA = open("weights_1.txt", "r")
        #weight_fileB = open("weights_2.txt", "r")
        '''
        weightA = float(weight_fileA.readline())
        weightB = float(weight_fileB.readline())
        if weightA != None:
            self.weights_1 = np.array([weightA,weightA,weightA])
        if weightB != None:
            self.weights_2 = np.array([weightB,weightB,weightB])
        '''
        print(self.weights_1)
        print(self.weights_2)

    def forward(self, x):
        # input_x = copy.deepcopy(x)
        self.input_layer = self.sigmoid(x)
        self.hidden_layer = self.sigmoid(np.matmul(self.weights_1, x))
        self.output_layer = self.sigmoid(np.matmul(self.weights_2, self.hidden_layer))
        return self.output_layer

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def sigmoidPrime(self, value):
        return value * (1 - value)

    def backward(self, given_input, desired_output, predicted_output):
        #Backward Propagation Function
        #New weight = old weight — Derivative Rate * learning rate
        d_rate = 0
        d_rate_2 = 0
        error = desired_output - predicted_output
        squared_error = np.square(error)
        d_rate_2 = np.gradient(squared_error)
        weight1_error = d_rate_2 - self.hidden_layer
        squared_error_weight_1 = np.square(weight1_error)
        d_rate = np.gradient(squared_error_weight_1)
        self.weights_1 = self.weights_1 - d_rate * self.learning_rate
        self.weights_2 = self.weights_2 - d_rate_2 * self.learning_rate
        #print(error)
        #print(squared_error)
    def train(self, target, epoch, rate):
        # Run the game on different thread so nothing freezes
        self.learning_rate = rate
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
            # if(self.runner.exitGame):
            # break
            input_x = np.array([self.runner.obst.x - self.runner.player_pos_x, self.runner.player_pos_y - self.runner.obst.y, self.runner.window_height - self.runner.player_pos_y])
            player_x = self.runner.player_pos_x
            obstacle_x = self.runner.obst.x
            player_y = self.runner.player_pos_y
            obstacle_y = obstacle_x = self.runner.obst.x
            # Check if the Array is a 0D Array or = None
            # if input_x.all() == None:
            # break
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
                time.sleep(1)
            elif maxed == output[1]:
                print("Option B")
                option = "B"
                time.sleep(1)
            elif maxed == output[2]:
                print("Option C")
                option = "C"
                self.runner.move_down()
                output_y = np.array([self.runner.player_pos_x, self.runner.player_pos_y, self.runner.obst.y])
                time.sleep(1)
            else:
                print("Default Option")
                time.sleep(1)
                continue
            # What should our Y be in order for this to work out perfectly??
            y = [0, 0, 0]
            #Reference Window Height and Width
            #self.window_height = 600  # 800
            #self.window_width = 800  # 1000
            if player_y > 400:
                if obstacle_y+50 > player_y :
                    print("Player > 400 & Player = Obstacle")
                    y = [0,0,1]
                else:
                    y = [0,1,0]
                    print("Player > 400 & Player != Obstacle")
            if player_y <=400:
                if obstacle_y == player_y:
                    print("Player < 400 & Player = Obstacle")
                    y = [1,0,0]
                else:
                    print("Player > 400 & Player != Obstacle")
                    y = [0,1,0]
            if target < self.runner.score:
                break
            time.sleep(2)
            # Backward Propogation to make the neural network learn
            self.backward(input_x, y, output)
            print(self.runner.score)
        self.saveWeights()

    def saveWeights(self):
        file_weights1 = open("weights_1.txt", "w")
        file_weights1.write(str(self.weights_1))
        file_weights2 = open("weights_2.txt", "w")
        file_weights2.write(str(self.weights_2))

    def lossFunction(self, predicted_y, actual_y):
        # We will use Mean Squared Error for our loss
        # Loss = sum of (pred_y - actual_y)^2
        error = (predicted_y - actual_y) ** 2  # **2 is squaring it
        error = np.sum(error)
        error = error / actual_y.size
        return error

    def predict(self):
        print("")
