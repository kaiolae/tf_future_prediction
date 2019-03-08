#Replay stored episode, generating video from it in higher resolution

import os
from random import choice
from vizdoom import *
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc, imshow

game = DoomGame()

# Use other config file if you wish.
#TODO: Make input arg.
game.load_config("../../maps/D3_battle_tiny_health_no_ammo.cfg")
game.set_episode_timeout(500) #TODO: 2100

# Record episodes while playing in 320x240 resolution without HUD
resX = 640
resY = 480
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_render_hud(True)
#Fixed Color issue!!
game.set_screen_format(ScreenFormat.BGR24)

# Episodes can be recorded in any available mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR)
game.set_mode(Mode.SPECTATOR)

game.init()

# Run and record this many episodes
episodes = 8


for i in range(0,1): #TODO in range(episodes)
    print("Episode: ", i)

    fourcc = VideoWriter_fourcc(*'DIVX')

    vw = VideoWriter('vid'+str(i)+'.avi', fourcc, 24, (resX, resY), True)
    meas_file = "meas_history" + str(i)+ ".csv"
    all_meas = []

    game.replay_episode("game_replay"+str(i)+".lmp")
    frame_counter = 0
    while not game.is_episode_finished():
        if frame_counter%100 == 0:
            print("Frame: ", frame_counter)
        s=game.get_state()
        all_meas.append(s.game_variables)
        img = s.screen_buffer
        #print("Screen buffer shape: ", img.shape)
        #img=np.transpose(img, (1,2,0))
        vw.write(img) #KOE: ImG is there! Just not being stored.
        #imshow('frame', img)
        #print(img.shape)
        # Use advance_action instead of make_action.
        game.advance_action()

        r = game.get_last_reward()
        # game.get_last_action is not supported and don't work for replay at the moment.

        #print("State #" + str(s.number))
        #print("Game variables:", s.game_variables[0])
        #print("Reward:", r)
        #print("=====================")
        #print("Episode finished.")
        #print("total reward:", game.get_total_reward())
        #print("************************")
        frame_counter+=1

    all_meas = np.array(all_meas)
    np.savetxt(meas_file, all_meas, delimiter=" ")
    print("saving meas to ", meas_file)
    vw.release()


game.close()
