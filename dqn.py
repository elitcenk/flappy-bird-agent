#!/usr/bin/env python
from __future__ import print_function

import sys

sys.path.append("game/")

import skimage as skimage
from skimage import transform, color, exposure

import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 100.  # timesteps to observe before training
N_EP = 10000   # number of episode
N_SAVE = 500  # every N_SAVE number save the model
EXPLORE = 10000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 5000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

IMG_ROWS, IMG_COLS = 80, 80
IMG_CHANNELS = 4  # Stacked 4 frames


class DQNAgent():

    def __init__(self):
        self.game_state = game.GameState()
        self.memory = deque(maxlen=REPLAY_MEMORY)
        self.scores = deque(maxlen=100)
        self.build_model()

    def build_model(self):
        print("Now we build the model")
        self.model = Sequential()
        self.model.add(Conv2D(32, (8, 8), input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), strides=(4, 4), padding="same"))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="same"))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(ACTIONS))
        self.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        print("We finish building the model")

    def train(self):
        # We go to training mode
        self.OBSERVE = OBSERVATION
        self.epsilon = INITIAL_EPSILON
        self.run_model()

    def evaluate(self):
        self.OBSERVE = 999999999  # We keep observe, never train
        self.epsilon = FINAL_EPSILON
        self.restore_model()
        self.run_model()

    def restore_model(self):
        print("Now we load weight")
        self.model.load_weights("model.h5")
        self.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        print("Weight load successfully")

    def run_model(self):
        for step in range(N_EP):
            score = 0
            current_stack = self.create_first_stack()
            while True:
                action, action_index = self.act(current_stack, step)
                # run the selected action and observed next state and reward
                frame, reward, terminal = self.game_state.frame_step(action)
                frame = self.preprocess(frame)
                frame = frame.reshape(1, frame.shape[0], frame.shape[1], 1)  # 1x[IMG_COLS]x[IMG_ROWS]x1
                self.remember(action_index, current_stack, reward, terminal, frame)
                current_stack = np.append(frame, current_stack[:, :, :, :3], axis=3)
                Q_sa, loss = self.replay(step)
                score = score + reward
                if terminal:
                    break
            self.scores.append(score)
            print("Episode {} score: {}".format(step + 1, score))
            self.mean_score = np.mean(self.scores)
            self.print(step, score)

    def print(self, step, score):
        if (step + 1) % 5 == 0:
            print("Episode {}, score: {}, exploration at {}%, mean of last 100 episodes was {}".format(step + 1, score, self.epsilon, self.mean_score))

        if (step + 1) % N_SAVE == 0 and step > 0:
            self.save_model()

    def save_model(self):
        print("Now we save model")
        self.model.save_weights("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)

    def remember(self, action_index, current_stack, reward, terminal, frame):
        # store the transition in memory
        self.memory.append((current_stack, action_index, reward, frame, terminal))

    def replay(self, step):
        Q_sa = 0
        loss = 0
        # only train if done observing
        if step > self.OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(self.memory, BATCH)
            # Now we do the experience replay
            current_stack, action_index, reward, frame, terminal = zip(*minibatch)
            current_stack = np.concatenate(current_stack)
            frame = np.concatenate(frame)
            next_stack = np.append(frame, current_stack[:, :, :, :3], axis=3)
            targets = self.model.predict(current_stack)
            Q_sa = self.model.predict(next_stack)
            targets[range(BATCH), action_index] = reward + GAMMA * np.max(Q_sa, axis=1) * np.invert(terminal)
            loss += self.model.train_on_batch(current_stack, targets)
        return Q_sa, loss

    def act(self, frame_stack, step):
        action_index = 0
        a_t = np.zeros([ACTIONS])
        # choose an action epsilon greedy
        if step % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = self.model.predict(frame_stack)  # input a stack of [IMG_CHANNELS] images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        # We reduced the epsilon gradually
        if self.epsilon > FINAL_EPSILON and step > self.OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return a_t, action_index

    def preprocess(self, frame):
        frame = skimage.color.rgb2gray(frame)
        frame = skimage.transform.resize(frame, (IMG_ROWS, IMG_COLS))
        frame = skimage.exposure.rescale_intensity(frame, out_range=(0, 255))
        return frame / 255.0

    def create_first_stack(self):
        # get the first state by doing nothing and preprocess the image
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        frame, reward, terminal = self.game_state.frame_step(do_nothing)
        frame = self.preprocess(frame)
        frame_stack = np.stack((frame, frame, frame, frame), axis=2)
        # In Keras, need to reshape
        frame_stack = frame_stack.reshape(1, frame_stack.shape[0], frame_stack.shape[1], frame_stack.shape[2])  # 1*[IMG_COLS]x[IMG_ROWS]*[IMG_CHANNELS]
        return frame_stack


def main():
    agent = DQNAgent()
    agent.train()


if __name__ == "__main__":
    main()
