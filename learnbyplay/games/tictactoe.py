import learnbyplay.games.rules
import torch
import copy
import logging

class TicTacToe(learnbyplay.games.rules.Authority):
    def __init__(self):
        super(TicTacToe, self).__init__()
        self.player_identifiers = ['X', 'O']
        self.player_to_channel = {'X': 0, 'O': 1}  # X is the agent; O is the opponent

    def LegalMoves(self, state_tsr, player):
        legal_moves = []
        for row in range(3):
            for col in range(3):
                if state_tsr[0, row, col] == 0 and \
                    state_tsr[1, row, col] == 0:
                    legal_moves.append(f"{row} {col}")
        return legal_moves

    def InitialState(self):
        initial_state_tsr = torch.zeros(2, 3, 3, dtype=torch.uint8)
        return initial_state_tsr

    def ToString(self, state_tsr):
        characters_list = []
        for row in range(3):
            for col in range(3):
                char = ' '
                if state_tsr[0, row, col] > 0:
                    char = 'X'
                elif state_tsr[1, row, col] > 0:
                    char = 'O'
                characters_list.append(char)

        repr_str = ""
        for row in range(3):
            for col in range(3):
                repr_str += ' ' + characters_list[3 * row + col] + ' '
                if col != 2:
                    repr_str += '|'
            repr_str += "\n"
            if row != 2:
                repr_str += "--- --- ---\n"
        return repr_str

    def Move(self, state_tsr, move, player_identifier):
        coords = move.split(' ')
        if len(coords) != 2:
            raise ValueError(f"TicTacToe.Move(): The move ({move}) could not be split in two, on a space")
        row = int(coords[0])
        col = int(coords[1])
        if row < 0 or row > 2 or col < 0 or col > 2:
            raise ValueError(f"TicTacToe.Move(): row ({row}) or col ({col}) is out of range [0, 2]")
        if state_tsr[0, row, col] > 0 or state_tsr[1, row, col] > 0:
            raise ValueError(f"TicTacToe.Move(): The square ({row}, {col}) is already occupied")
        new_state_tsr = copy.deepcopy(state_tsr)
        new_state_tsr[self.player_to_channel[player_identifier], row, col] = 1

        X_won = self.ThereIsALine(new_state_tsr, self.player_to_channel['X'])
        O_won = self.ThereIsALine(new_state_tsr, self.player_to_channel['O'])
        all_squares_are_taken = self.AllSquaresAreTaken(new_state_tsr)
        #logging.debug(f"TicTacToe.Move(): X_won = {X_won}; O_won = {O_won}; all_squares_are_taken = {all_squares_are_taken}")

        game_status = learnbyplay.games.rules.GameStatus.NONE
        if X_won:
            if player_identifier == 'X':
                game_status = learnbyplay.games.rules.GameStatus.WIN
            else:
                game_status = learnbyplay.games.rules.GameStatus.LOSS
        elif O_won:
            if player_identifier == 'X':
                game_status = learnbyplay.games.rules.GameStatus.LOSS
            else:
                game_status = learnbyplay.games.rules.GameStatus.WIN
        elif all_squares_are_taken:
            game_status = learnbyplay.games.rules.GameStatus.DRAW

        return (new_state_tsr, game_status)

    def MaximumNumberOfMoves(self):
        return 9

    def StateTensorShape(self):
        return (2, 3, 3)

    def SwapAgentAndOpponent(self, state_tsr):
        swapped_state_tsr = torch.zeros_like(state_tsr)
        swapped_state_tsr[0, :, :] = state_tsr[1, :, :]
        swapped_state_tsr[1, :, :] = state_tsr[0, :, :]
        return swapped_state_tsr

    def ThereIsALine(self, state_tsr, channel):
        there_is_a_line = False
        # Check rows
        for row in range(3):
            row_is_full_of_ones = True
            for col in range(3):
                if state_tsr[channel, row, col] == 0:
                    row_is_full_of_ones = False
                    break
            if row_is_full_of_ones:
                return True

        # Check columns
        for col in range(3):
            col_is_full_of_ones = True
            for row in range(3):
                if state_tsr[channel, row, col] == 0:
                    col_is_full_of_ones = False
                    break
            if col_is_full_of_ones:
                return True

        # Check diagonals
        backslash_is_full_of_ones = True
        for i in range(3):
            if state_tsr[channel, i, i] == 0:
                backslash_is_full_of_ones = False
                break
        if backslash_is_full_of_ones:
            return True
        slash_is_full_of_ones = True
        for i in range(3):
            if state_tsr[channel, i, 2 - i] == 0:
                slash_is_full_of_ones = False
                break
        if slash_is_full_of_ones:
            return True

        return False

    def AllSquaresAreTaken(self, state_tsr):
        for row in range(3):
            for col in range(3):
                if state_tsr[0, row, col] == 0 and state_tsr[1, row, col] == 0:
                    return False
        return True