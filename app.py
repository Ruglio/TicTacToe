import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from matplotlib import colormaps
from time import sleep
from main import Player, AI, TicTacToe

app = Flask(__name__, static_folder='static')

# Initialize the AI and Player
player = Player("Human")
ai = AI("AI", train=False)
ai.load("Qtable.npy")
game = TicTacToe(player, ai)

class DataStore():
    colorToggle = 'off'

data = DataStore()

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if game.players[game.p] == ai:
            _, _, _ = game.player_step(ai)
    elif request.method == 'POST' and game.players[game.p] == player:
            # Get the user's move from the form data
            user_move = int(request.form.get('move'))
            # Convert the user's move to an action
            box = game.action_to_box(user_move)
            # Update the game state with the user's move
            game.step(box)
            # Reset the game it is over
            if game.done:
                return redirect(url_for('game_over'))
            if game.players[game.p] == ai:
                _, _, _ = game.player_step(ai)
        
    # Reset the game it is over
    if game.done:
        return redirect(url_for('game_over'))
    
    state = game.board_to_state(game.board)
    feasible_actions = ai.feasible_actions(state)
    Q_factors = ai.Q_table[state].copy()
    colors = Qfactors_to_color(Q_factors, feasible_actions)
    return render_template('index.html', board=game.board, colors=colors, colorToggle=data.colorToggle)
    
@app.route('/game_over', methods=['GET', 'POST'])
def game_over():
    if request.form.get('reset') == 'reset':
        game.reset()
        return redirect(url_for('index'))
    winner = 0 if game.winner == 0 else game.players[game.winner].name
    board = game.board
    return render_template('game_over.html', board=board, winner=winner)

@app.route('/toggle_colors', methods=['POST'])
def toggle_colors():
    data.colorToggle = 'on' if data.colorToggle == 'off' else 'off'
    return redirect(url_for('index'))


def Qfactors_to_color(Q_factors, feasible_actions):
    # Q_factors_f = Q_factors[feasible_actions]
    # # Normalize the Q factors
    # Q_factors_f = (Q_factors_f - Q_factors_f.min()) / (Q_factors_f.max() - Q_factors_f.min()) if Q_factors_f.max() != Q_factors_f.min() else np.ones_like(Q_factors_f)
    # Q_factors[feasible_actions] = Q_factors_f
    # # Convert the Q factors to colors using red to green color map
    # cmap = colormaps['RdYlGn']
    # colors = [cmap(x) if x != ai.init else (1, 1, 1, 1) for x in Q_factors]
    
    Q_factors_f = Q_factors[feasible_actions]
    # Normalize the Q factors, sub the avg and divide by the std
    Q_factors_f = (Q_factors_f - Q_factors_f.mean()) / (Q_factors_f.std() + 1e-16)
    # Use sigmoid to normalize the Q factors
    Q_factors_f = 1 / (1 + np.exp(-Q_factors_f))
    Q_factors[feasible_actions] = Q_factors_f
    # Convert the Q factors to colors using red to green color map
    cmap = colormaps['RdYlGn']
    colors = [cmap(x) if x != ai.init else (1, 1, 1, 1) for x in Q_factors]
    
    return colors

if __name__ == '__main__':
    app.run(debug=True)