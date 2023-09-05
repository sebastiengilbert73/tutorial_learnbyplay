import learnbyplay.games.rules
from learnbyplay.player import Player
from typing import Dict, List, Any, Set, Tuple, Optional, Union
import torch
import copy

class Arena:
    def __init__(self, authority: learnbyplay.games.rules.Authority, agent: Player, opponent: Player,
                 device='cpu') -> None:
        self.authority = authority
        self.agent = agent
        self.opponent = opponent
        self.device = device
        #self.index_to_player = {0: self.agent, 1: self.opponent}
        #if not self.agent_starts:
        #    self.index_to_player = {0: self.opponent, 1: self.agent}

    def RunGame(self, agent_starts, epsilons=None):
        if epsilons is None:
            epsilons = [0.0]
        state_tsr = self.authority.InitialState().to(self.device)
        state_action_list = []
        number_of_moves = 0
        game_status = learnbyplay.games.rules.GameStatus.NONE
        if agent_starts:
            index_to_player = {0: self.agent, 1: self.opponent}
        else:
            index_to_player = {0: self.opponent, 1: self.agent}

        while (game_status == learnbyplay.games.rules.GameStatus.NONE) and number_of_moves < self.authority.MaximumNumberOfMoves():
            player_ndx = number_of_moves % 2
            player = index_to_player[player_ndx]
            if number_of_moves < len(epsilons):
                epsilon = epsilons[number_of_moves]
            else:
                epsilon = epsilons[-1]
            player.epsilon = epsilon
            chosen_move = player.ChooseMove(self.authority, state_tsr)
            state_action_list.append((copy.deepcopy(state_tsr), chosen_move))

            state_tsr, game_status = self.authority.Move(state_tsr, chosen_move, player.identifier)
            number_of_moves += 1
            # Make sure the win or loss is attributed to the agent
            #print(f"Arena.RunGame(): player '{player.identifier}' is self.opponent = {player is self.opponent}")
            if game_status == learnbyplay.games.rules.GameStatus.WIN and player is self.opponent:
                game_status = learnbyplay.games.rules.GameStatus.LOSS
            elif game_status == learnbyplay.games.rules.GameStatus.LOSS and player is self.opponent:
                game_status = learnbyplay.games.rules.GameStatus.WIN
            # If we reach the maximum number of moves and the status is still None, make it a DRAW
            if number_of_moves == self.authority.MaximumNumberOfMoves() and game_status == learnbyplay.games.rules.GameStatus.NONE:
                game_status = learnbyplay.games.rules.GameStatus.DRAW
        if game_status == learnbyplay.games.rules.GameStatus.NONE:  # We reached the maximum number of moves
            game_status = learnbyplay.games.rules.GameStatus.DRAW
        state_action_list.append((copy.deepcopy(state_tsr), None))
        #print(f"Arena.RunGame(): game_status = {game_status}")
        return state_action_list, game_status

    def GeneratePositionsAndExpectations(self, number_of_games: int, gamma: float, epsilons: List[float]):
        position_expectation_list = []
        for game_ndx in range(number_of_games):
            agent_starts = game_ndx %2 == 0
            state_action_list, game_status = self.RunGame(agent_starts, epsilons)
            starting_state_action_list = [state_action_list[i] for i in range(1, len(state_action_list), 2)]
            nonStarting_state_action_list = [state_action_list[i] for i in range(0, len(state_action_list), 2)]
            agent_state_action_list = None
            game_position_expectation_list = []
            if agent_starts:
                agent_state_action_list = starting_state_action_list
            else:
                agent_state_action_list = nonStarting_state_action_list
            if game_status == learnbyplay.games.rules.GameStatus.DRAW:  # All positions have a prediction of 0
                for state_action in agent_state_action_list:
                    game_position_expectation_list.append((state_action[0], 0))
            elif game_status == learnbyplay.games.rules.GameStatus.WIN:
                for state_action_ndx in range(len(agent_state_action_list)):
                    state_action = agent_state_action_list[state_action_ndx]
                    exponent = len(agent_state_action_list) - 1 - state_action_ndx  # 2, 1, 0
                    expected_return = gamma ** exponent
                    game_position_expectation_list.append((state_action[0], expected_return))
            elif game_status == learnbyplay.games.rules.GameStatus.LOSS:
                for state_action_ndx in range(len(agent_state_action_list)):
                    state_action = agent_state_action_list[state_action_ndx]
                    exponent = len(agent_state_action_list) - 1 - state_action_ndx  # 2, 1, 0
                    expected_return = - (gamma ** exponent)
                    game_position_expectation_list.append((state_action[0], expected_return))

            position_expectation_list += game_position_expectation_list

        return position_expectation_list

    def RunMultipleGames(self, number_of_games, epsilons):
        number_of_agent_wins = 0
        number_of_agent_losses = 0
        number_of_draws = 0
        for game_ndx in range(number_of_games):
            agent_starts = game_ndx %2 == 0
            state_action_list, game_status = self.RunGame(agent_starts, epsilons)
            if game_status == learnbyplay.games.rules.GameStatus.WIN:
                number_of_agent_wins += 1
            elif game_status == learnbyplay.games.rules.GameStatus.LOSS:
                number_of_agent_losses += 1
            elif game_status == learnbyplay.games.rules.GameStatus.DRAW:
                number_of_draws += 1
            else:
                raise ValueError(f"Arena.RunMultipleGames(): game_status = {game_status}")
            #print(f"RunMultipleGames(): state_action_list = \n{state_action_list}")
        return number_of_agent_wins, number_of_agent_losses, number_of_draws
