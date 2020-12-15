"""
Tetris Simulator

Author - Anqi Li (anqil4@cs.washington.edu)
Adapted from the java simulator from Drew Bagnell's
course at Carnegie Mellon University

"""
import gym
from gym.utils import seeding
import numpy as np


class TetrisState:
    """
    the tetris state
    """

    def __init__(self, field, top, next_piece, lost, turn, cleared):
        # the board configuration
        self.field = field
        # the top position
        self.top = top
        # the piece ID of the next piece
        self.next_piece = next_piece
        # whether the game has lost
        self.lost = lost
        # the current turn
        self.turn = turn
        # the number of rows cleared so far
        self.cleared = cleared
        self.width = len(top)



    def copy(self):
        return TetrisState(
            self.field.copy(),
            self.top.copy(),
            self.next_piece,
            self.lost,
            self.turn,
            self.cleared
        )
    def _comp_bumpiness(self,  n_norm=4):
        # m = state.field.shape[0]
        n = self.field.shape[1]
        # # count the number of holes
        reference_heights = np.zeros(n)
        reference_heights[:-1] = self.top[1:]
        # np.linalg.norm(state.top - reference_heights, 4)



        bumpiness = np.abs(self.top - reference_heights).tolist()
        bumpiness_score1 = np.sum(bumpiness)

        bumpiness_score2 = np.linalg.norm(np.abs(self.top - reference_heights), 2) * n
            # np.linalg.norm(np.abs(state.top - reference_heights), n_norm) * n
        return bumpiness, bumpiness_score1, bumpiness_score2

    def _comp_holes(self):
        """
        Mathematically speaking, it computes the number of zeros under heights. Entries that are different than all its neighbours
        it doesn't include non-enclosed holes (as it can be filled)
        :param field:
        :return: the number of holes
        """
        m = self.field.shape[0]
        n = self.field.shape[1]
        height_cond_matrix = np.indices((m, n))[0] < self.top
        zero_cond_matrix = self.field == 0
        m = zero_cond_matrix * height_cond_matrix
        holes = np.count_nonzero(m, axis = 1)
        num_holes  = np.sum(holes)
        hole_score1 = np.linalg.norm(holes, 2)

        return num_holes, hole_score1
    def _comp_agg_height(self):
        return np.sum(self.top)

    def self_encode(self):
        bumpiness, bumpiness_score1, bumpiness_score2 = self._comp_bumpiness(n_norm=1)
        num_holes, hole_score1 = self._comp_holes()
        next_piece_encoding = np.eye(7)[self.next_piece].tolist()
        return [ self._comp_agg_height(), self.cleared, num_holes, hole_score1, bumpiness_score1, np.max(self.top)] + next_piece_encoding

class TetrisEnv(gym.Env):
    metadata = {
        'render.modes': ['ascii']
    }
    # def get_piece_state(self, piece_id):
    #     self.piece_width[piece_id]
    #     self.piece_height[piece_id]
    #     self.piece_bottom[piece_id]
    #     self.piece_top[piece_id]

    def __init__(self):
        self.n_cols = 10
        self.n_rows = 21
        self.n_pieces = 7

        # the next several lists define the piece vocabulary in detail
        # width of the pieces [piece ID][orientation]
        # pieces: O, I, L, J, T, S, Z
        self.piece_orients = [1, 2, 4, 4, 4, 2, 2]
        self.piece_width = [
            [2],
            [1, 4],
            [2, 3, 2, 3],
            [2, 3, 2, 3],
            [2, 3, 2, 3],
            [3, 2],
            [3, 2]
        ]
        # height of pieces [piece ID][orientation]
        self.piece_height = [
            [2],
            [4, 1],
            [3, 2, 3, 2],
            [3, 2, 3, 2],
            [3, 2, 3, 2],
            [2, 3],
            [2, 3]
        ]
        self.piece_bottom = [
            [[0, 0]],
            [[0], [0, 0, 0, 0]],
            [[0, 0], [0, 1, 1], [2, 0], [0, 0, 0]],
            [[0, 0], [0, 0, 0], [0, 2], [1, 1, 0]],
            [[0, 1], [1, 0, 1], [1, 0], [0, 0, 0]],
            [[0, 0, 1], [1, 0]],
            [[1, 0, 0], [0, 1]]
        ]
        self.piece_top = [
            [[2, 2]],
            [[4], [1, 1, 1, 1]],
            [[3, 1], [2, 2, 2], [3, 3], [1, 1, 2]],
            [[1, 3], [2, 1, 1], [3, 3], [2, 2, 2]],
            [[3, 2], [2, 2, 2], [2, 3], [1, 2, 1]],
            [[1, 2, 2], [3, 2]],
            [[2, 2, 1], [2, 3]]
        ]

        # initialize legal moves for all pieces
        self.legal_moves = []
        for i in range(self.n_pieces):
            piece_legal_moves = []
            for j in range(self.piece_orients[i]):
                for k in range(self.n_cols + 1 - self.piece_width[i][j]):
                    piece_legal_moves.append([j, k])
            self.legal_moves.append(piece_legal_moves)


        self.state = None
        self.cleared_current_turn = 0

    def seed(self, seed=None):
        """
        set the random seed for the environment
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        make a move based on the orientation and slot
        """
        orient, slot = action
        self.state.turn += 1

        old_state = self.state.copy()

        # height of the field
        height = max(
            self.state.top[slot + c] - self.piece_bottom[self.state.next_piece][orient][c]
            for c in range(self.piece_width[self.state.next_piece][orient])
        )


        # check if game ended
        if height + self.piece_height[self.state.next_piece][orient] >= self.n_rows:
            self.state.lost = True
            return self.state, self._get_reward(), True, {}

        # for each column in the piece - fill in the appropriate blocks
        for i in range(self.piece_width[self.state.next_piece][orient]):
            # from bottom to top of brick
            for h in range(height + self.piece_bottom[self.state.next_piece][orient][i],
                           height + self.piece_top[self.state.next_piece][orient][i]):
                self.state.field[h, i + slot] = self.state.turn

        # adjust top
        for c in range(self.piece_width[self.state.next_piece][orient]):
            self.state.top[slot + c] = height + self.piece_top[self.state.next_piece][orient][c]

        # check for full rows - starting at the top
        self.cleared_current_turn = 0
        for r in range(height + self.piece_height[self.state.next_piece][orient] - 1, height - 1, -1):
            # if the row was full - remove it and slide above stuff down
            if np.all(self.state.field[r] > 0):
                self.cleared_current_turn += 1
                self.state.cleared += 1
                # for each column
                for c in range(self.n_cols):
                    # slide down all bricks
                    self.state.field[r:self.state.top[c], c] = self.state.field[(r + 1):(self.state.top[c] + 1), c]
                    # lower the top
                    self.state.top[c] -= 1
                    while self.state.top[c] >= 1 and self.state.field[self.state.top[c] - 1, c] == 0:
                        self.state.top[c] -= 1

        # pick a new piece
        self.state.next_piece = self._get_random_piece()
        return self.state.copy(), self._get_reward(), False, {}

    def reset(self):
        lost = False
        turn = 0
        cleared = 0

        field = np.zeros((self.n_rows, self.n_cols), dtype=np.int)
        top = np.zeros(self.n_cols, dtype=np.int)
        next_piece = self._get_random_piece()

        self.state = TetrisState(field, top, next_piece, lost, turn, cleared)
        return self.state.copy()

    def render(self, mode='ascii'):
        print('\nThe wall:')
        print('-' * (2 * self.n_cols + 1))
        for r in range(self.n_rows - 1, -1, -1):
            render_string = '|'
            for c in range(self.n_cols):
                if self.state.field[r, c] > 0:
                    render_string += '*|'
                else:
                    render_string += ' |'
            render_string += ''
            print(render_string)
        print('-' * (2 * self.n_cols + 1))

        print('\nThe next piece:')
        if self.state.next_piece == 0:
            print('**\n**')
        elif self.state.next_piece == 1:
            print('****')
        elif self.state.next_piece == 2:
            print('*\n*\n**')
        elif self.state.next_piece == 3:
            print(' *\n *\n**')
        elif self.state.next_piece == 4:
            print(' * \n***')
        elif self.state.next_piece == 5:
            print(' **\n**')
        elif self.state.next_piece == 6:
            print('**\n **')

    def close(self):
        pass

    def _get_random_piece(self):
        """
        return an random integer 0-6
        """
        return np.random.randint(self.n_pieces)

    def _get_reward(self):
        """
        reward function
        """

        if self.state.lost == True:
          return 0
        return 1 + (self.cleared_current_turn**2) * self.state.width


        # # if game ends
        # if self.state.lost == True:
        #     return -2
        #
        # # TODO: change it to your own choice of rewards
        # def comp_reward(state):
        #     max_height = np.max(state.top)  # range 0 - 7
        #     num_holes = self._comp_holes(state) # approx 0-20
        #     bumpiness_score = self._comp_bumpiness(state, n_norm= 4) # usually less than 10
        #     reward = -1.0 * max_height  - 0.55 * num_holes - 0.07 * bumpiness_score
        #     return reward



        # print("score before ",  comp_reward(old_state), "score after: ",  comp_reward(self.state))
        # print("reward: ", comp_reward(self.state) - comp_reward(old_state))
        # print("top: ", self.state.top)
        # self.render()
        # exit()
        # clearance_reward = 0.75 * len(self.state.top) * (self.state.cleared - old_state.cleared)** 2
        #
        # return comp_reward(self.state) - comp_reward(old_state)  + clearance_reward

    def get_actions(self):
        """
        gives the legal moves for the next piece
        :return:
        """
        return self.legal_moves[self.state.next_piece]

    def set_state(self, state):
        """
        set the field and the next piece
        """
        self.state = state.copy()


