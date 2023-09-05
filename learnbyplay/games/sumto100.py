import learnbyplay.games.rules
import torch
import copy
from typing import Dict, List, Any, Set, Tuple, Optional, Union

class SumTo100(learnbyplay.games.rules.Authority):
    def __init__(self):
        super(SumTo100, self).__init__()


    def LegalMoves(self, state_tsr: torch.Tensor, player_identifier: str) -> List[str]:
        sum = self.Sum(state_tsr)
        if sum <= 90:
            return [str(n) for n in range(1, 11)]
        else:
            maximum_n = 100 - sum
            return [str(n) for n in range(1, maximum_n + 1)]

    def InitialState(self) -> torch.Tensor:
        state_tsr = torch.zeros(101)
        state_tsr[0] = 1
        return state_tsr


    def ToString(self, state_tsr: torch.Tensor) -> str:
        sum = self.Sum(state_tsr)
        return str(sum)

    def Move(self, state_tsr: torch.Tensor, move: str, player_identifier: str) -> Tuple[torch.Tensor, learnbyplay.games.rules.GameStatus]:
        initial_sum = self.Sum(state_tsr)
        move_nbr = int(move)
        if move_nbr < 1 or move_nbr > 10:
            raise ValueError(f"SumTo100.Move(): move_nbr ({move_nbr}) is out of [1, 10]")
        if initial_sum + move_nbr > 100:
            raise ValueError(f"SumTo100.Move(): initial_sum ({initial_sum}) + move_nbr ({move_nbr}) > 100")
        new_sum = initial_sum + move_nbr
        new_state_tsr = torch.zeros(101)
        new_state_tsr[new_sum] = 1
        game_status = learnbyplay.games.rules.GameStatus.NONE
        if new_sum == 100:
            game_status = learnbyplay.games.rules.GameStatus.WIN
        return new_state_tsr, game_status

    def MaximumNumberOfMoves(
            self) -> int:  # Returns the maximum number of moves in a game, after which the game is a draw
        return 100

    def StateTensorShape(self) -> Tuple[int]:
        return (101,)

    def SwapAgentAndOpponent(self, state_tsr: torch.Tensor) -> torch.Tensor:
        return state_tsr

    def Sum(self, state_tsr):
        for i in range(state_tsr.shape[0]):
            if state_tsr[i] > 0:
                return i
        return 0