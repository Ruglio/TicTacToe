import numpy as np


class Player:
    def __init__(self, name) -> None:
        self.name = name
        
    def __str__(self) -> str:
        return self.name
    
    def choose_action(self, state):
        action = int(input("Enter your move: "))
        if action not in self.feasible_actions(state):
            print("Invalid move")
            return self.choose_action(state)
        return action
    
    def update(self, state, action, reward, next_state):
        pass
    
    def feasible_actions(self, state):
        # Return a list of feasible actions
        board = self.state_to_board(state)
        return [i for i, el in enumerate(board.flatten()) if el == 0]
    
    def state_to_board(self, state):
        # Convert the base-3 number to a tuple
        board = []
        while state > 0:
            board.append(state % 3)
            state = state // 3
        board = board + [0] * (9 - len(board))
        board = np.array(board).reshape((3, 3))
        return board
    

class AI(Player):
    def __init__(self, name, epsilon=0.2, alpha=0.3, gamma=1, train=True, init=3) -> None:
        self.name = name
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.train = train
        self.init = init
        
        # Initialize the Q-table
        # Actually, not all state-action pairs are possible, but we will check feasibility later
        self.num_states = 3**9
        self.num_actions = 9
        # Optimistic initialization
        self.Q_table = np.zeros((self.num_states, self.num_actions)) + init
        
    def choose_action(self, state):
        # Choose an action based on the epsilon-greedy policy
        feasible_actions = self.feasible_actions(state)
        if self.train and np.random.random() < self.epsilon:
            # Choose a random action
            action = np.random.choice(feasible_actions)
        else:
            # Choose the best action
            Qs = self.Q_table[state, feasible_actions]
            idx = np.random.choice(np.flatnonzero(Qs == Qs.max()))
            action = feasible_actions[idx]
        return action
    
    def update(self, state, action, reward, next_state):
        feasible_actions = self.feasible_actions(state)
        new_Q = self.Q_table[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q_table[next_state, feasible_actions]) - self.Q_table[state, action])
        self.Q_table[state, action] = new_Q
        
    def save(self, filename=None):
        if filename is None:
            filename = self.name + ".npy"
        np.save(filename, self.Q_table)
        
    def load(self, filename=None):
        if filename is None:
            filename = self.name + ".npy"
        self.Q_table = np.load(filename)


class TicTacToe:
    def __init__(self, player1, player2) -> None:
        self.board = np.zeros((3, 3))
        self.p = np.random.choice([1, 2])
        self.done = False
        self.winner = 0
        
        self.players = {1: player1, 2: player2}
        self.reward = 1
    
    # We can map the board to a state and vice versa using the following functions
    # The state is a tuple of 9 elements, each element can be 0, 1, or 2
    # 0: empty, 1: player1, 2: player2
    # The integer representation of the state is obtained by converting the tuple to a base-3 number
    # Similarly for the action, we can map it to a box coordinate and vice versa
    
    def board_to_state(self, board):
        # Convert the board to a base-3 number
        state = 0
        for i, el in enumerate(board.flatten()):
            state += el * 3**i
        return int(state)
    
    def state_to_board(self, state):
        # Convert the base-3 number to a tuple
        board = []
        while state > 0:
            board.append(state % 3)
            state = state // 3
        board = board + [0] * (9 - len(board))
        board = np.array(board).reshape((3, 3))
        return board
    
    def box_to_action(self, box):
        # Convert the box coordinates to an action
        return box[0] * 3 + box[1]
    
    def action_to_box(self, action):
        # Convert the action to box coordinates
        return action // 3, action % 3
    
    def reset(self):
        self.board = np.zeros((3, 3))
        self.p = np.random.choice([1, 2])
        self.done = False
        self.winner = 0
    
    def step(self, box):
        if self.board[box] == 0:
            self.board[box] = self.p
            
            if self.check_win():
                self.done = True
                self.winner = self.p
                
            elif np.all(self.board != 0):
                self.done = True
                self.winner = 0
            
            # Change player
            self.p = 3 - self.p
            
        else:
            print("Invalid move")
            
    def player_step(self, player, show_progress=False):
        state = self.board_to_state(self.board)
        action = player.choose_action(state)
        box = self.action_to_box(action)
        self.step(box)
        next_state = self.board_to_state(self.board)
        if show_progress:
            self.show_board()
        
        return state, action, next_state
    
    def check_win(self):
        # Check rows
        for i in range(3):
            if self.board[i, 0] == self.board[i, 1] == self.board[i, 2] != 0:
                return True
        # Check columns
        for i in range(3):
            if self.board[0, i] == self.board[1, i] == self.board[2, i] != 0:
                return True
        # Check diagonals
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return True
        if self.board[2, 0] == self.board[1, 1] == self.board[0, 2] != 0:
            return True
        return False
    
    def show_board(self):
        print(self.board)
    
    def play(self, num_games=1, show_progress=False):
        
        for i in range(num_games):
            if show_progress:
                print("Game", i + 1)
                self.show_board()
            while not self.done:
                # Current player
                c_player = self.players[self.p]
                c_state, c_action, n_state_next = self.player_step(c_player, show_progress)
                reward = 0 if self.winner == 0 else self.reward
                
                # Next player
                n_player = self.players[self.p]  # Note that self.p has been changed
                if np.count_nonzero(self.board) != 1:
                    # It is not the first move of next player
                    # We can update the Q-table of next player
                    # Next player could lose or draw
                    n_player.update(n_state, n_action, -reward, n_state_next)
                    
                if self.done:
                    # Current player wins or draw
                    c_player.update(c_state, c_action, reward, n_state_next)
                else:
                    n_state, n_action, c_state_next = self.player_step(n_player, show_progress)
                    reward = 0 if self.winner == 0 else self.reward
                    
                    # Update current player Q-table
                    # Current player could lose or draw
                    c_player.update(c_state, c_action, -reward, c_state_next)
                    
                    if self.done:
                        # Next player wins or draw
                        n_player.update(n_state, n_action, reward, c_state_next)
                    
            if show_progress:
                if self.winner == 0:
                    print("Draw")
                else:
                    print("Winner:", self.players[self.winner])
            self.reset()