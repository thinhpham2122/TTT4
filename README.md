# TTT4
![output_img](images/example.png)

Rule
------
Game win condition: get 4 in a line in a 4 x 4 grid

To play
------
Edit 'first' in play.py to change first player\
Edit 'name' in play.py to change AI model\
run play.py\
enter in location to play\
location table:

| C1 | C2 | C3 | C4 |
|---|---|---|---|
|0|1|2|3|
|4|5|6|7|
|8|9|10|11|
|12|13|14|15|

Package Requirement
------
tensorflow

Purpose
------
Create an AI using deep Q-learning to master this game.

Training reward system
------
Moves that are invalid and loses the game are given -1 as reward. Moves that win the game are given 1 as reward. All other moves are given 0 as reward.

Result
------
After training overnight, it learned to plays valid moves, block other player from wining, and win when there is 3 in a line. However, there is little evident of it planning several moves ahead. This model did not master the game, and there is room for improvement.
