"""
Tetris Simulator

Author - Anqi Li (anqil4@cs.washington.edu)
Adapted from the java simulator from Drew Bagnell's
course at Carnegie Mellon University

"""
import gym
from gym.utils import seeding
import numpy as np
import torch

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

    def copy(self):
        return TetrisState(
            self.field.copy(),
            self.top.copy(),
            self.next_piece,
            self.lost,
            self.turn,
            self.cleared
        )

    def _comp_heights(self):
        field = self.field
        m = field.shape[0]
        n = field.shape[1]
        height = 0
        max_height = 0
        min_height = 0
        for i in range(n):
            col = np.copy(field[:, i])
            if np.count_nonzero(col) > 0:
                while len(col) > 0 and col[-1] == 0:
                    col = col[:-1]
                height += len(col)
                max_height = max(max_height, len(col))
                min_height = min(min_height, len(col))
        return height, max_height, max_height - min_height

    def _comp_holes(self):
        field = self.field
        m = field.shape[0]
        n = field.shape[1]
        hole = 0
        for i in range(n):
            col = np.copy(field[:, i])
            if np.count_nonzero(col) > 0:
                while len(col) > 0 and col[-1] == 0:
                    col = col[:-1]
                hole += len(col) - np.count_nonzero(col)

        return hole

    def _comp_bumpiness(self):
        field = self.field
        m = field.shape[0]
        n = field.shape[1]
        height = []
        bump = 0
        max_bump = 0
        for i in range(n):
            col = np.copy(field[:, i])
            if np.count_nonzero(col) > 0:
                while len(col) > 0 and col[-1] == 0:
                    col = col[:-1]
                height.append(len(col))
            else:
                height.append(0)
        for i in range(len(height) - 1):
            bump += np.abs(height[i] - height[i + 1])
            max_bump = max(max_bump, np.abs(height[i] - height[i + 1]))
        return bump, max_bump

    def _comp_property(self):
        field = self.field
        m = field.shape[0]
        n = field.shape[1]
        # Compute height
        height = []
        max_height = 0
        min_height = 0
        for i in range(n):
            col = np.copy(field[:, i])
            if np.count_nonzero(col) > 0:
                while len(col) > 0 and col[-1] == 0:
                    col = col[:-1]
                height.append(len(col))
            else:
                height.append(0)
        max_height = max(height)
        min_height = min(height)

        bumps = []
        bump = 0
        max_bump = 0
        # Compute bump
        for i in range(len(height) - 1):
            bump += np.abs(height[i] - height[i + 1])
            bumps.append(height[i] - height[i + 1])
            max_bump = max(max_bump, np.abs(height[i] - height[i + 1]))

        # Compute hole
        hole = 0
        holes = []
        for i in range(n):
            col = np.copy(field[:, i])
            if np.count_nonzero(col) > 0:
                while len(col) > 0 and col[-1] == 0:
                    col = col[:-1]
                holes.append(len(col) - np.count_nonzero(col))
        hole = sum(holes)

        return torch.tensor([hole, sum(height), max_height, min_height, max_bump, bump])

    def get_state_encode(self, old_state):
        return self._comp_property().float()
class TetrisEnv(gym.Env):
    metadata = {
        'render.modes': ['ascii']
    }

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
        old_reward = self._get_reward()
        old_cleared = np.copy(self.state.cleared)
        # height of the field
        height = max(
            self.state.top[slot + c] - self.piece_bottom[self.state.next_piece][orient][c]
            for c in range(self.piece_width[self.state.next_piece][orient])
        )

        # check if game ended
        if height + self.piece_height[self.state.next_piece][orient] >= self.n_rows:
            self.state.lost = True
            return self.state, self._get_reward() - old_reward, True, {}

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

    def _get_reward_old(self):
        """
        reward function
        """
        # if game ends
        if self.state.lost == True:
            return -5

        return -0.5 * self.state._comp_heights() + 0.75 * self.state.cleared \
               - 0.35 * self.state._comp_holes() - 0.15 * self.state._comp_bumpiness()

    def _get_reward(self):
        """
        reward function
        """
        # if game ends
        if self.state.lost == True:
            return -10

        return 1 + self.cleared_current_turn * 5

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

    def get_next_sa(self):
        """
        output next states encoding and actions
        """
        state_cur = self.state.copy()
        actions = self.get_actions()
        next_states = []
        next_rewards = []
        next_dones = []
        for act in actions:
            next_state, reward, done, _ = self.step(act)
            next_states.append(next_state._comp_property())
            next_rewards.append(reward)
            next_dones.append(done)
            self.set_state(state_cur)

        return next_rewards, next_states, next_dones

