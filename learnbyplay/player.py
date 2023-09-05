import abc
import copy
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import learnbyplay.games.rules
import torch
import random
import sys

class Player(abc.ABC):
    def __init__(self, identifier: str, epsilon: float = 0):
        self.identifier = identifier
        self.epsilon = epsilon

    @abc.abstractmethod
    def ChooseMove(self, authority: learnbyplay.games.rules.Authority, state_tsr: torch.Tensor) -> str:
        pass

class RandomPlayer(Player):
    def __init__(self, identifier):
        super(RandomPlayer, self).__init__(identifier)

    def ChooseMove(self, authority: learnbyplay.games.rules.Authority, state_tsr: torch.Tensor) -> str:
        legal_moves = authority.LegalMoves(state_tsr, self.identifier)
        return random.choice(legal_moves)

class ConsolePlayer(Player):
    def __init__(self, identifier):
        super(ConsolePlayer, self).__init__(identifier)

    def ChooseMove(self, authority: learnbyplay.games.rules.Authority, state_tsr: torch.Tensor) -> str:
        legal_moves = authority.LegalMoves(state_tsr, self.identifier)
        the_chosen_move_is_legal = False
        chosen_move = None
        while not the_chosen_move_is_legal:
            print(authority.ToString(state_tsr))
            chosen_move = input(f"Choose a move (legal moves: {legal_moves}): ")
            the_chosen_move_is_legal = chosen_move in legal_moves
        return chosen_move

class PositionRegressionPlayer(Player):
    def __init__(self, identifier, neural_net, temperature=1.0, look_ahead_depth=1,
                 flatten_state=True, acts_as_opponent=False, epsilon=0):
        super(PositionRegressionPlayer, self).__init__(identifier, epsilon)
        self.neural_net = neural_net
        self.neural_net.eval()
        self.temperature = temperature
        self.look_ahead_depth = look_ahead_depth
        self.flatten_state = flatten_state
        self.device = next(self.neural_net.parameters()).device
        self.acts_as_opponent = acts_as_opponent

    def ChooseMove(self, authority: learnbyplay.games.rules.Authority,
                   state_tsr: torch.Tensor) -> str:
        if self.epsilon > 0:
            return self.ChooseWithEpsilonGreedy(authority, state_tsr)
        elif self.look_ahead_depth == 1:
            return self.ChooseWithTemperatureOneLevel(authority, state_tsr)
        elif self.look_ahead_depth == 2:
            return self.ChooseWithTemperatureTwoLevels(authority, state_tsr)
        else:
            raise NotImplementedError(f"PositionRegressionPlayer.ChooseMove(): Not supported combination of self.epsilon ({self.epsilon}) and self.look_ahead_depth ({self.look_ahead_depth})")

    def ChooseWithEpsilonGreedy(self, authority, state_tsr):
        move_predicted_return_list = self.PredictReturns(state_tsr, authority)
        legal_moves, corresponding_predicted_returns, champion_move, \
            highest_predicted_return = self.ChampionMove(move_predicted_return_list)

        random_0to1 = random.random()
        if random_0to1 <= self.epsilon:
            return random.choice(legal_moves)
        else:
            return champion_move

    def ChooseWithTemperatureOneLevel(self, authority, state_tsr):
        move_predicted_return_list = self.PredictReturns(state_tsr, authority)
        legal_moves, corresponding_predicted_returns, champion_move, \
            highest_predicted_return = self.ChampionMove(move_predicted_return_list)

        if self.temperature <= 0:  # Zero temperature: return the greedy best move
            return champion_move

        # Softmax with non-zero temperature
        corresponding_predicted_temperature_returns_tsr = torch.tensor(
            corresponding_predicted_returns) / self.temperature
        corresponding_probabilities_tsr = torch.nn.functional.softmax(corresponding_predicted_temperature_returns_tsr,
                                                                      dim=0)
        random_nbr = random.random()
        running_sum = 0
        for move_ndx in range(corresponding_probabilities_tsr.shape[0]):
            running_sum += corresponding_probabilities_tsr[move_ndx].item()
            if running_sum >= random_nbr:
                return legal_moves[move_ndx]

    def ChooseWithTemperatureTwoLevels(self, authority, state_tsr):
        move_predicted_return_list = self.MaxiMinOneLevel(state_tsr, authority)
        legal_moves, corresponding_predicted_returns, champion_move, \
            highest_predicted_return = self.ChampionMove(move_predicted_return_list)

        if self.temperature <= 0:  # Zero temperature: return the greedy best move
            return champion_move

        # Softmax with non-zero temperature
        corresponding_predicted_temperature_returns_tsr = torch.tensor(
            corresponding_predicted_returns) / self.temperature
        corresponding_probabilities_tsr = torch.nn.functional.softmax(corresponding_predicted_temperature_returns_tsr,
                                                                      dim=0)
        random_nbr = random.random()
        running_sum = 0
        for move_ndx in range(corresponding_probabilities_tsr.shape[0]):
            running_sum += corresponding_probabilities_tsr[move_ndx].item()
            if running_sum >= random_nbr:
                return legal_moves[move_ndx]

    def PredictReturns(self, state_tsr, authority):
        legal_moves = authority.LegalMoves(state_tsr, self.identifier)
        move_predicted_return_list = []
        for move_ndx in range(len(legal_moves)):
            move = legal_moves[move_ndx]
            candidate_state_tsr, game_status = authority.Move(
                state_tsr, move, self.identifier
            )
            if game_status == learnbyplay.games.rules.GameStatus.WIN:
                move_predicted_return_list.append((move, 1.0))
            elif game_status == learnbyplay.games.rules.GameStatus.LOSS:
                move_predicted_return_list.append((move, -1.0))
            elif game_status == learnbyplay.games.rules.GameStatus.DRAW:
                move_predicted_return_list.append((move, 0.0))
            else:
                candidate_state_tsr = candidate_state_tsr.float().to(self.device)
                if self.acts_as_opponent:
                    candidate_state_tsr = authority.SwapAgentAndOpponent(candidate_state_tsr)
                if self.flatten_state:
                    candidate_state_tsr = candidate_state_tsr.view(-1)
                predicted_return = self.neural_net(candidate_state_tsr.unsqueeze(0)).squeeze().item()
                move_predicted_return_list.append((move, predicted_return))
        return move_predicted_return_list

    def MaxiMinOneLevel(self, state_tsr, authority):
        legal_moves = authority.LegalMoves(state_tsr, self.identifier)
        #print(f"player.MaxiMinOneLevel(): legal_moves = {legal_moves}")
        move_predicted_return_list = []
        opponent_identifier = authority.SwapIdentifier(self.identifier)
        for move_ndx in range(len(legal_moves)):
            move = legal_moves[move_ndx]
            #print(f"player.MaxiMinOneLevel(): move = {move}")
            candidate_state_tsr, game_status = authority.Move(
                state_tsr, move, self.identifier
            )
            if game_status == learnbyplay.games.rules.GameStatus.WIN:
                move_predicted_return_list.append((move, 1.0))
            elif game_status == learnbyplay.games.rules.GameStatus.LOSS:
                move_predicted_return_list.append((move, -1.0))
            elif game_status == learnbyplay.games.rules.GameStatus.DRAW:
                move_predicted_return_list.append((move, 0.0))
            else:  # The game is not over: Check the opponent candidate moves
                opponent_legal_moves = authority.LegalMoves(candidate_state_tsr, opponent_identifier)
                # Evaluate the positions resulting from the opponent legal moves
                opponent_move_to_return = {}
                for opponent_move_ndx in range(len(opponent_legal_moves)):
                    opponent_move = opponent_legal_moves[opponent_move_ndx]

                    opponent_candidate_state_tsr, game_status = authority.Move(
                        candidate_state_tsr, opponent_move, opponent_identifier
                    )
                    #print(f"player.MaxiMinOneLevel(): opponent_move: {opponent_move}; game_status = {game_status}")
                    if game_status == learnbyplay.games.rules.GameStatus.WIN:
                        opponent_move_to_return[opponent_move] = -1.0
                    elif game_status == learnbyplay.games.rules.GameStatus.LOSS:
                        opponent_move_to_return[opponent_move] = 1.0
                    elif game_status == learnbyplay.games.rules.GameStatus.DRAW:
                        opponent_move_to_return[opponent_move] = 0.0
                    else:  # The game is not over
                        # Swap the tensor to evaluate with position using the agent neural network
                        swapped_opponent_candidate_state_tsr = authority.SwapAgentAndOpponent(opponent_candidate_state_tsr)
                        if self.acts_as_opponent:
                            swapped_opponent_candidate_state_tsr = authority.SwapAgentAndOpponent(swapped_opponent_candidate_state_tsr)
                        if self.flatten_state:
                            swapped_opponent_candidate_state_tsr = swapped_opponent_candidate_state_tsr.view(-1)
                        swapped_opponent_candidate_state_tsr = swapped_opponent_candidate_state_tsr.float().to(self.device)
                        predicted_return = self.neural_net(
                            swapped_opponent_candidate_state_tsr.unsqueeze(0)).squeeze().item()
                        # Negate the return, since we want a value from the agent's point of view
                        predicted_return = -1.0 * predicted_return
                        opponent_move_to_return[opponent_move] = predicted_return
                        #print(f"player.MaxiMinOneLevel(): opponent_move: {opponent_move}; predicted_return = {predicted_return}")
                # The opponent chooses the minimum (i.e. most negative) return among the choices
                minimum_return = sys.float_info.max
                for opponent_candidate_move, opponent_candidate_return in opponent_move_to_return.items():
                    if opponent_candidate_return < minimum_return:
                        minimum_return = opponent_candidate_return
                move_predicted_return_list.append((move, minimum_return))
        return move_predicted_return_list

    def ChampionMove(self, move_predicted_return_list):
        legal_moves = []
        corresponding_predicted_returns = []
        highest_predicted_return = -2.0
        champion_move = None
        for move, predicted_return in move_predicted_return_list:
            legal_moves.append(move)
            corresponding_predicted_returns.append(predicted_return)
            if predicted_return > highest_predicted_return:
                highest_predicted_return = predicted_return
                champion_move = move
        return legal_moves, corresponding_predicted_returns, champion_move, highest_predicted_return