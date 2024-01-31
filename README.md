In this project I tried to put into practice my studies into reinforcement learning. In particular, the temporal difference method: Q-learning with epsilon-greedy approach. The aim is simply to implement it to properly understand the relationship between environment (the game) and agent (the player). Therefore, the parameter of the model (epsilon, learning-rate, and decay) were not fine-tuned. As a consequence the model is not as good as it can be and we will win a few more matches :)

The computer was taught how to win by performing against itself for 5.000.000 games, quite a lot. Kinda as a trial and error, it understood the usually suggested moves. The result can be seen below and played in my [personal website](https://sites.google.com/view/andrearuglioni/projects/tictactoe). You can try to beat it.

The color of each square represent the respective Q-factor (value of that action) learnt by the model. Hence, a green color means that the respective action is revealing move, and could let you win the match. On the other hand, a red color is more likely a blunder.

The project has been interesting and educational also because I better understood how to use Flask. It is a python module that helps in the creation of a web app by linking back-end and front-end. Therefore, it can better show off the logic of the game, by making use of HTML and CSS, by creating a more appealing and user-friendly interface.

![ttt](https://github.com/Ruglio/TicTacToe/assets/67823727/6e9a2348-d0ed-451c-bb45-8dab2e9fd449)
