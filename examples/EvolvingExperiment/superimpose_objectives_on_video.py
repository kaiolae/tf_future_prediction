#Puts objective vectors on top of video.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('vid1.avi')
objectives_file = "objectives_history1.csv"
objective_values = np.genfromtxt(objectives_file, delimiter=' ')
print("Loaded objectives of shape: ", objective_values.shape)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

objective_names = ["ammo", "health", "frags"]

# In my current setup, the 3 last obj-vals of each row are the ones we want to show.
objective_values = objective_values[:, -len(objective_names):]

# Plot loc and size in pixels
plot_upper_x = 0
plot_x_width = 100
plot_upper_y = 0
plot_y_width = 100

folder = "superimposed_vid"
if not os.path.exists(folder):
    os.makedirs(folder)



def convert_plotted_values(plotted_objective_values):
    # Converts the raw values we want to plot to fit inside the image frame.
    converted_values = (1 - plotted_objective_values) + 1  # Since the y-axis in images is "upside down"
    # Values now range fro 0 to 2, where 2 are those that were previously -1, and 0 are those that were previously 1.

    # Stretching out the y-range.
    converted_values = converted_values * plot_y_width
    return converted_values


linestyles = ["-", "--", "-."]

counter = 0
# Read until video is completed
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
        for obj in objective_names:
            #print("lin shape:", x.shape)
            #print("obj shape: ",objective_values[0:frame_counter,objcounter].shape)
            if counter < plot_x_width:
                #Before we have many frames, we just plot all obj vals.
                ax.plot(convert_plotted_values(objective_values[0:counter, objcounter]), linewidth = 5, label=obj, alpha=0.7, linestyle = linestyles[objcounter])
            else:
                ax.plot(convert_plotted_values(objective_values[counter-plot_x_width:counter, objcounter]), linewidth = 5, label=obj, alpha=0.7, linestyle = linestyles[objcounter])
            objcounter+=1

        ax.legend(loc='upper right')

        plt.savefig(folder+"/"+str(counter)+".png")
        plt.close(fig)
        counter += 1

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
