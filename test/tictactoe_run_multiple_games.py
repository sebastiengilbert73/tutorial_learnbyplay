import argparse
import logging
import learnbyplay.games.tictactoe
from learnbyplay.arena import Arena
import learnbyplay
import ast
import architectures.tictactoe_arch as architectures
import torch
import random

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    consolePlayer,
    agentArchitecture,
    agentNeuralNetworkFilepath,
    agentTemperature,
    agentLookAheadDepth,
    opponentArchitecture,
    opponentNeuralNetworkFilepath,
    opponentTemperature,
    opponentLookAheadDepth,
    numberOfGames,
    useCpu,
    epsilons
):
    logging.info("tictactoe_run_multiple_games.main()")

    device = 'cpu'
    if not useCpu and torch.cuda.is_available():
        device = 'cuda'

    authority = learnbyplay.games.tictactoe.TicTacToe()
    agent_identifier = 'X'
    opponent_identifier = 'O'
    agent = learnbyplay.player.RandomPlayer(agent_identifier)
    if consolePlayer:
        agent = learnbyplay.player.ConsolePlayer(agent_identifier)
    elif agentNeuralNetworkFilepath is not None:
        neural_net = None
        if agentArchitecture.startswith('SaintAndre_'):
            chunks = ChunkArchName(agentArchitecture)
            neural_net = architectures.SaintAndre(
                latent_size=int(chunks[1]),
                dropout_ratio=0.5
            )
        elif agentArchitecture.startswith('Coptic_'):
            chunks = ChunkArchName(agentArchitecture)
            neural_net = architectures.Coptic(
                number_of_channels=int(chunks[1]),
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"tictactoe_run_multiple_games.main(): Not implemented agent architecture '{agentArchitecture}'")
        neural_net.load_state_dict(torch.load(agentNeuralNetworkFilepath))
        neural_net.to(device)
        agent = learnbyplay.player.PositionRegressionPlayer(
            identifier=agent_identifier,
            neural_net=neural_net,
            temperature=agentTemperature,
            flatten_state=True,
            acts_as_opponent=False,
            look_ahead_depth=agentLookAheadDepth,
            epsilon=0.0
        )
    opponent = learnbyplay.player.RandomPlayer(opponent_identifier)
    if opponentNeuralNetworkFilepath is not None:
        opponent_neural_net = None
        if opponentArchitecture.startswith('SaintAndre_'):
            chunks = ChunkArchName(opponentArchitecture)
            opponent_neural_net = architectures.SaintAndre(
                latent_size=int(chunks[1]),
                dropout_ratio=0.5
            )
        elif opponentArchitecture.startswith('Coptic_'):
            chunks = ChunkArchName(opponentArchitecture)
            opponent_neural_net = architectures.Coptic(
                number_of_channels=int(chunks[1]),
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(
                f"tictactoe_run_multiple_games.main(): Not implemented opponent architecture '{opponentArchitecture}'")
        opponent_neural_net.load_state_dict(torch.load(opponentNeuralNetworkFilepath))
        opponent_neural_net.to(device)
        opponent = learnbyplay.player.PositionRegressionPlayer(
            identifier=opponent_identifier,
            neural_net=opponent_neural_net,
            temperature=opponentTemperature,
            flatten_state=True,
            acts_as_opponent=True,
            look_ahead_depth=opponentLookAheadDepth,
            epsilon=0.0
        )


    arena = Arena(authority, agent, opponent)
    number_of_agent_wins, number_of_agent_losses, number_of_draws = arena.RunMultipleGames(
        numberOfGames, epsilons=epsilons)
    logging.info(f"number_of_agent_wins = {number_of_agent_wins}; number_of_agent_losses = {number_of_agent_losses}; number_of_draws = {number_of_draws}")
    return number_of_agent_wins, number_of_agent_losses, number_of_draws

def ChunkArchName(arch_name):
    chunks = arch_name.split('_')
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--consolePlayer', help="The agent plays with the console", action='store_true')
    parser.add_argument('--agentArchitecture', help="In case of a neural network, the architecture. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--agentNeuralNetworkFilepath', help="The filepath to the agent neural network. Default: 'None'", default='None')
    parser.add_argument('--agentTemperature', help="The agent temperature. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--agentLookAheadDepth', help="The agent search depth. Default: 2", type=int, default=2)
    parser.add_argument('--opponentArchitecture',
                        help="In case of a neural network, the architecture. Default: 'SaintAndre_512'",
                        default='SaintAndre_512')
    parser.add_argument('--opponentNeuralNetworkFilepath',
                        help="The filepath to the opponent neural network. Default: 'None'", default='None')
    parser.add_argument('--opponentTemperature', help="The opponent temperature. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--opponentLookAheadDepth', help="The opponent search depth. Default: 2", type=int, default=2)
    parser.add_argument('--numberOfGames', help="The number of games played. Default: 1000", type=int, default=1000)
    parser.add_argument('--useCpu', help="Force using CPU", action='store_true')
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--epsilons', help="The espsilon values. Default: '[0.0]'", default='[0.0]')
    args = parser.parse_args()

    random.seed(args.randomSeed)
    torch.manual_seed(args.randomSeed)

    if args.agentNeuralNetworkFilepath.upper() == 'NONE':
        args.agentNeuralNetworkFilepath = None
    if args.opponentNeuralNetworkFilepath.upper() == 'NONE':
        args.opponentNeuralNetworkFilepath = None
    args.epsilons = ast.literal_eval(args.epsilons)
    main(
        args.consolePlayer,
        args.agentArchitecture,
        args.agentNeuralNetworkFilepath,
        args.agentTemperature,
        args.agentLookAheadDepth,
        args.opponentArchitecture,
        args.opponentNeuralNetworkFilepath,
        args.opponentTemperature,
        args.opponentLookAheadDepth,
        args.numberOfGames,
        args.useCpu,
        args.epsilons
    )