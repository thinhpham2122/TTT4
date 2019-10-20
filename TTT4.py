class TTT4:
    def __init__(self):
        self.board = [0] * 16
        self.player = 1
        self.played = 0

    def check_status(self):
        c_board = []
        self.played = 0
        for location in self.board:
            if location:
                c_board.append(location)
                self.played += 1
            else:
                c_board.append(-100)
        win_condition = -4 if self.player == 1 else 4

        if ((c_board[0] + c_board[1] + c_board[2] + c_board[3]) == win_condition or
                (c_board[4] + c_board[5] + c_board[6] + c_board[7]) == win_condition or
                (c_board[8] + c_board[9] + c_board[10] + c_board[11]) == win_condition or
                (c_board[12] + c_board[13] + c_board[14] + c_board[15]) == win_condition or
                (c_board[0] + c_board[4] + c_board[8] + c_board[12]) == win_condition or
                (c_board[1] + c_board[5] + c_board[9] + c_board[13]) == win_condition or
                (c_board[2] + c_board[6] + c_board[10] + c_board[14]) == win_condition or
                (c_board[3] + c_board[7] + c_board[11] + c_board[15]) == win_condition or
                (c_board[0] + c_board[5] + c_board[10] + c_board[15]) == win_condition or
                (c_board[3] + c_board[6] + c_board[9] + c_board[12]) == win_condition):
            return 'win'
        elif self.played >= 16:
            return 'draw'
        else:
            return 'played'

    def play(self, location):
        action = -1 if self.player == 1 else 1
        if self.board[location]:
            return 'invalid'
        self.board[location] = action
        self.player = 1 if self.player == 2 else 2
        self.played += 1
        return self.check_status()

    def print_board(self):
        p_board = []
        for location in self.board:
            if location:
                if location == -1:
                    location = 0
                p_board.append(location)
            else:
                p_board.append('.')

        print(f'{p_board[0]} {p_board[1]} {p_board[2]} {p_board[3]} \n'
              f'{p_board[4]} {p_board[5]} {p_board[6]} {p_board[7]} \n'
              f'{p_board[8]} {p_board[9]} {p_board[10]} {p_board[11]} \n'
              f'{p_board[12]} {p_board[13]} {p_board[14]} {p_board[15]}\n')


