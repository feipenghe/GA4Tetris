import random
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
from itertools import count
import pickle
import envs
import gym
import math
from _collections import deque
from random import sample

from multiprocessing import Pool
from numpy.random import SeedSequence, default_rng



# TODO: maybe remember states for those who run over to
# increase test time + frequency
# check the ratio
# output highest number of steps (to see if current steps limit is enough

# TOTHINK: how much steps to look ahead?

# to think, is it possible to map features into high dimensional, so that it's linearly separable
def state_encode(states):
    """
    Encode a state list or a single state
    :param states:
    :return:
    """
    # Inpute: A list of n states
    # Output: A tensor of dimension n * m, where m is the length of encoding

    if hasattr(states, "__iter__"): # for a state list
        encodes = []
        for s in states:
            # field_l = s.field.flatten().tolist()
            encode = s.self_encode()

            # # top_l = s.top.tolist()
            # state_l = field_l +  np.eye(7)[s.next_piece].tolist() + state_l

            encodes.append(encode)
        return np.array(encodes)
    else:
        state = states # rename for clearity
        encode = state.self_encode()
        return np.array(encode)


def get_next_states(env):
    # Input: current environment
    # Output: all possible next states
    state_cur = env.state.copy()
    actions = env.get_actions()
    next_states = []
    next_rewards = []
    import pdb
    pdb.set_trace()
    for act in actions:
        next_state, reward, done, _ = env.step(act)
        next_states.append(next_state)
        next_rewards.append(reward)
        env.set_state(state_cur)

    return next_rewards, next_states

# class parameter
#

class ParameterPopulation():
    # parameter data structure: need least fittest and highest fittest
    # fn: generate offspring(
        # fn: fittest selection
        # fn: crossover
            # fn: with certain chances to replace it
            # fn: add new parameter and sort it
        # fn: mutation
        # return one child

    # fn: update_iteration()
        # fn: generate offspring



    # fn: replacement
    def __init__(self, size = 1000, num_game = 128, num_move = 500, mode = "train"):
        if mode == "train":
            self.size = size
            self.num_game = num_game
            self.num_move = num_move
            self.selection_size = (int)(self.size /10) # size of sub population sampled from population
            self.replace_size = (int)(self.size /3) # number of parameters replaced

            # initialize data
            self._initialize_parameters_pop()  # (vector, score) increasing order of fit score
            self.best_score = 0
        else:
            self.num_move = 500

    def _normalize(self, v):
        """
        Normalize parameter vector, it can be either parameter vector or weight vector
        :param p:
        :return:
        """
        # if v.all() == 0:
        #     v = np.random.random((2,))
        normalized_v = v / np.linalg.norm(v, 2)
        return normalized_v
    def _sort_population(self, population):
        """
        Sort either population or subpopulation in ascending order
        :param population:
        :return:
        """
        return sorted(population, key = lambda t:t[1])

    def sort_population(self):
        """
        Sort population which is self.data
        :return:
        """
        self.data = self._sort_population(self.data)


    def comp_one_fit_score(self, i):
        np.random.seed(random.randint(0,34567829*i))
        p = np.random.random((13, )) - 0.5
        normalized_p = p / np.linalg.norm(p, 2)
        # TODO: comp fit score
        fit_score = self.comp_fit_score(p, i)
        return (normalized_p, fit_score)

    def _initialize_parameters_pop(self):
        self.data = []
        # ss = SeedSequence(12345)
        # # Spawn off 10 child SeedSequences to pass to child processes.
        # child_seeds = ss.spawn(self.size)
        # streams = [default_rng(s) for s in child_seeds]

        with Pool(64) as p:
            self.data += p.map(self.comp_one_fit_score, list(range(self.size)))
        self.sort_population()



    def replace_subpopulation(self, subpopulation):
        n = len(subpopulation)
        self.data[:n] = subpopulation
        self.sort_population()



    def train(self):
        meetStopCondition = False
        iteration = 0
        while (not meetStopCondition):
            print(f"Iteration {iteration}:  \t fit score: {self.comp_population_fit_score()}")
            offspring_population = []

            with Pool(32) as p:
                offspring_population += p.map(self.spawn_child, list(range(self.replace_size)))
            self.replace_subpopulation(offspring_population)

            # update best score
            _, self.best_score = self.data[-1]
            print("current best score is: ", self.best_score )
            for i in range(1, 10):
                print(f"best {i}th result: ", self.data[-i])
            if iteration % 1 == 0:
                print("testing......")
                p, fit_score = self.data[-1]
                self.test(p, 10)
                print("")
            iteration += 1

    def spawn_child(self, i):
        # random.seed(i**2*random.randint(0, 100))
        def cross_over(sub_population):
            "cross over the fittest two parameters from a ordered subpopultion"
            # choose two fittest (last two)
            try:
                fit_parent1 = sub_population[-1]
                fit_parent2 = sub_population[-2]
            except IndexError:
                import pdb
                pdb.set_trace()
            p1, fit_score1 = fit_parent1
            p2, fit_score2 = fit_parent2

            normalized_fit_weights = self._normalize(np.array([fit_score1, fit_score2]))
            if random.random() < 0.01:
                print("normalized_fit_weights: ", normalized_fit_weights)
                print("\t\t\t\tp1: ", p1, "p2: ", p2)
                print("\t\t\t\t\t\t\t\t fit score1: ", fit_score1, "fit score2: ", fit_parent2)

            weight1, weight2 = normalized_fit_weights

            new_p = p1 * weight1 + p2 * weight2
            return self._normalize(new_p)
        def mutation(new_p):
            new_p[random.randint(0, 3)] += random.random()*0.6 - 0.3
            return self._normalize(new_p)

        np.random.seed(random.randint(0,34567829 * i))
        sub_population = random.sample(self.data, self.selection_size)  # random sample (unordered)
        sub_population = self._sort_population(sub_population)
        p = cross_over(sub_population)
        if random.random() < 0.05:
            p = mutation(p)
        score = self.comp_fit_score(p, random.randint(0,34567829 * i))
        return (p, score)


    def comp_population_fit_score(self):
        """ compute average score"""
        score = 0
        for pair in self.data:
            _, fit_score = pair
            score += fit_score
        return score / self.size

    def comp_fit_score(self, para, random_idx):
        """
        Compute fitness score given parameter under the configuration of the number of games and the number of movements.
        Compute the fitness given the maximum number of steps
        :param p:
        :return:
        """

        score = 1e-3
        for i in range(self.num_game):
            env = gym.make('Tetris-v0')
            # we also want to make sure the these games are different
            env.seed(random.randint(0,random_idx)) # each with different random seeds
            env.reset()

            # TOTHINK: change the linear function to neural network
            # the added complexity might not give a good approximation
            # usually, nn is used for a complex feature input and return a prediction

            for step_j in range(self.num_move):
                best_action, fit_score = self._comp_best_action(env, para)
                # based on the parameters we choose the best action
                next_state, _, done, _ = env.step(best_action)
                score += env.cleared_current_turn
                # break when game ends and start the next game
                if done:
                    break
        return score * 1.0 / self.num_game

    def _comp_best_action(self, env, para):
        """
        Givn current environment, try out all actions and return the best one given the parameter vector
        best action given current knowledge p and current state
        :return: best action, reward estimation
        """
        cur_state = env.state.copy() # used to reset later
        actions = env.get_actions()
        best_action = None
        best_score = None

        # make a copy of current state
        # compute all frames for the two pieces
        # for example: rotate left left down
        # it's all moves by the two pieces

        for cur_a in actions:
            next_state, _, done, _ = env.step(cur_a)
            # do I compute reward or just clear lines

            state_encoding = state_encode(next_state)

            fit_score = np.dot(state_encoding,para) # best next state fit score
            if best_action == None and best_score == None:
                best_action = cur_a
                best_score = fit_score
            else:
                try:
                    if fit_score > best_score:
                        best_action = cur_a
                        best_score = fit_score
                except:
                    import pdb
                    pdb.set_trace()
            env.set_state(cur_state) # reset the state to the input state
        return best_action, best_score


    def test(self, p, test_time = 32):


        # TOTHINK: change the linear function to neural network
        # the added complexity might not give a good approximation
        # usually, nn is used for a complex feature input and return a prediction

        highest_clearance = 0
        score_l = []
        for i in range(test_time):
            score = 0
            env = gym.make('Tetris-v0')
            env.seed()  # each with different random seeds
            env.reset()

            for step_j in range(self.num_move):
                best_action, fit_score = self._comp_best_action(env, p)
                # based on the parameters we choose the best action
                next_state, _, done, _ = env.step(best_action)
                score += env.cleared_current_turn
                # break when game ends and start the next game
                if done:
                    score_l.append(score)
                    break

        with open("output.log", "a") as f:
            f.write(f"p vector: {str(p)} \n "
                    f"the highest is {max(score_l)} "
                    f"\n the average is {sum(score_l)/len(score_l)} "
                    f"the score is {score_l} \n")


import bisect

if __name__ == '__main__':
    # env = gym.make('Tetris-v0')
    # env.seed(0)
    # env.reset()
    # env.render()



    POPULATION_SIZE =  100  # 1000 parameters instance
    NUM_GAME = 32
    NUM_MOVE = 5000
    MODE = "train"
    # MODE = "test"
    # split data into min and max heap (with ratio of half and half)




    # parameter_population = ParameterPopulation()

    # parameter_population.train()
    # [-0.22269655, - 0.91795468 ,- 0.2598739, - 0.20057668]

    p = np.array([-0.51006, -0.760666, -0.35663, -0.184483])  # .reshape((4,))
    # # p = np.zeros((4,))
    # # # for i in range(4):
    # # #     p[i] = parameters[i]
    # #
    # # print(parameters.shape)
    # # print(p.shape)
    # # exit()
    # # population class

    parameter_population = ParameterPopulation(POPULATION_SIZE, NUM_GAME, NUM_MOVE, MODE)
    parameter_population.train()
    # parameter_population.test(p)
