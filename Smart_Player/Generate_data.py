#This file will be used to create/generate all the data that will be used to train the Neural Network.
import matplotlib.pyplot as plt
from matplotlib import style



def m_up(): #creates the data set for move_up label
    print("Generating Move Up data set")
    move_up = open("move_up.txt",'w')
    # self.window_height = 600 self.window_width = 800
    for i in range(450,801):
        for j in range(50):
            move_up.write(str(i) + " " + str(i-j) + "\n")
        for k in range(50):
            move_up.write(str(i) + " " + str(i + k) + "\n")
    move_up.close()

def m_down(): #creates the data set for move_down label
    move_down = open("move_down.txt",'w')
    # self.window_height = 600 self.window_width = 800
    for i in range(0,350):
        for j in range(50):
            move_down.write(str(i) + " " + str(i-j) + "\n")
        for k in range(50):
            move_down.write(str(i) + " " + str(i + k) + "\n")
    move_down.close()

def createPlot():
    print("createPlot: Creating Plot")
    dataFile = open("move_up.txt", "r")
    x_up = []
    rawData = []
    y_up = []
    for line in dataFile:
        rawData.append(line.split())
    dataFile.close()
    j = 0
    for i in rawData:
        #print(i)
        if j > 100:
            break
        x_up.append(i[0])
        y_up.append(i[1])
        j = j+1
    pltFigure = plt.figure()
    pltAxes = pltFigure.add_subplot(111)

    x_down = []
    y_down = []
    dataFile = open("move_down.txt","r")
    for line in dataFile:
        rawData.append(line.split())
    dataFile.close()
    j = 0
    for i in rawData:
        print(i)
        if j > 100:
            break
        x_down.append(i[0])
        y_down.append(i[1])
        j = j + 1

    pltAxes.scatter(x_down, x_down, s=200, c='red', marker="*")
    pltAxes.scatter(x_up, y_up, s=200, c='blue', marker=".")
    # https://stackoverflow.com/questions/4270301/matplotlib-multiple-datasets-on-the-same-scatter-plot?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # URL to plot multiple datasets
    plt.show()





def generate_data():
    print("Generating Training Data")
    m_up()
    m_down()
    createPlot()

generate_data()