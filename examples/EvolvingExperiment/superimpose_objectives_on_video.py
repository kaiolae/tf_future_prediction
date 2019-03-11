#Puts objective vectors on top of video.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import os
import neat
import sys
import pickle

#TODO: meas history -> get size. make numpy array. send to ANN -> plot.
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video_id = 0
scenario = "D3_battle_no_ammo"
cap = cv2.VideoCapture('vid'+scenario+str(video_id)+'.avi')
meas_file = "meas_history"+scenario+str(video_id)+".csv" #The meas recorded for each frame. ANN can convert to objectives.
meas_array = np.genfromtxt(meas_file, delimiter = " ")
objs_array = meas_array

winner_filename = scenario+"/winner_network.pickle"
with open(winner_filename, 'rb') as pickle_file:
    winner_genome = pickle.load(pickle_file)

config_file = "config"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)
net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

timestep_counter = 0
for timestep in meas_array:
    objectives = net.activate(timestep)
    o_count = 0
    for o in objectives:
        objs_array[timestep_counter][o_count] = o
        o_count+=1
    timestep_counter+=1

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

objective_names = ["Ammunition", "Health", "Attack"]

# Plot loc and size in pixels
plot_upper_x = 0
plot_x_width = 100
plot_upper_y = 0
plot_y_width = 50

bar_width = 15

#frames_skipped_during_eval = 4 #Skipped every 4 frames during eval -> the objectives measured last 4 "real" frames.

folder = "superimposed_vid_" + scenario
if not os.path.exists(folder):
    os.makedirs(folder)

def convert_plotted_values_y_axis(plotted_objective_values):
    # Converts the raw values we want to plot to fit inside the image frame.
    converted_values = (1 - plotted_objective_values) + 1  # Since the y-axis in images is "upside down"
    # Values now range fro 0 to 2, where 2 are those that were previously -1, and 0 are those that were previously 1.

    # Stretching out the y-range.
    converted_values = converted_values * plot_y_width
    return converted_values


def convert_plotted_values_x_axis(plotted_objective_values):
    # Stretching out the y-range.
    #First making 0 the minimum, then stretching out.
    converted_values = (1+plotted_objective_values) * plot_x_width
    return converted_values


linestyles = ["-", "--", "-."]
colors = ['blue', 'orange', 'green']

counter = 0
# Read until video is completed
#TODO Plot a line showing where obj_val = 0 goes.

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        #Converting to the colors python expects.
        b_channel, g_channel, r_channel = cv2.split(frame)

        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50  # creating a dummy alpha channel image.

        frame = cv2.merge((r_channel,g_channel,b_channel))

        #Trying to superimpose graph
        print("Img shape: ", frame.shape)
        fig, ax = plt.subplots()
        ax.imshow(frame)
        #x=np.linspace(0,frame_counter)
        objcounter = 0

        #Testing bar chart instead of line.
        y_pos = [25, 55, 85] #TODO Generalize

        for obj in objective_names:
            ax.barh(y_pos[objcounter], convert_plotted_values_x_axis(objs_array[counter][objcounter]), bar_width, align='center', color=colors[objcounter],
                    ecolor='black')
            #print("lin shape:", x.shape)
            #print("obj shape: ",objective_values[0:frame_counter,objcounter].shape)
        #    if counter < plot_x_width:
                #Before we have many frames, we just plot all obj vals.
        #        ax.plot(convert_plotted_values(objs_array[0:counter, objcounter]), linewidth = 5, label=obj, alpha=0.7, linestyle = linestyles[objcounter])
        #    else:
        #        ax.plot(convert_plotted_values(objs_array[counter-plot_x_width:counter, objcounter]), linewidth = 5, label=obj, alpha=0.7, linestyle = linestyles[objcounter])
            objcounter+=1

        ax.legend(objective_names, loc='upper right',fontsize=15)

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(tick.NullLocator())
        plt.gca().yaxis.set_major_locator(tick.NullLocator())
        plt.savefig(folder+"/"+str(counter)+".png",bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        counter += 1

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
