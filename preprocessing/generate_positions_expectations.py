import argparse
import logging
import learnbyplay.games.tictactoe
import learnbyplay.games.sumto100
from learnbyplay.arena import Arena
import learnbyplay
import os
import random
import torch
import ast
import architectures.tictactoe_arch as tictactoe_arch
import architectures.sumto100_arch as sumto100_arch


logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    game,
    numberOfGames,
    gamma,
    randomSeed,
    agentArchitecture,
    agentFilepath,
    opponentArchitecture,
    opponentFilepath,
    epsilons,
    temperature,
    printPositionsAndExpectations
):
    logging.info("generate_positions_expectations.main()")

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    authority = None
    agent_identifier = 'agent'
    opponent_identifier = 'opponent'
    flatten_state = True
    if game == 'tictactoe':
        authority = learnbyplay.games.tictactoe.TicTacToe()
        agent_identifier = 'X'
        opponent_identifier = 'O'
    elif game == 'sumto100':
        authority = learnbyplay.games.sumto100.SumTo100()
    else:
        raise NotImplementedError(f"generate_positions_expectations.main(): Not implemented game '{game}'")

    agent = learnbyplay.player.RandomPlayer(agent_identifier)
    if agentFilepath is not None:
        agent_neural_net = None
        if agentArchitecture.startswith('SaintAndre_'):
            chunks = ChunkArchName(agentArchitecture)
            agent_neural_net = tictactoe_arch.SaintAndre(
                latent_size=int(chunks[1]),
                dropout_ratio=0.5
            )
        elif agentArchitecture.startswith('Coptic_'):
            chunks = ChunkArchName(agentArchitecture)
            agent_neural_net = tictactoe_arch.Coptic(
                number_of_channels=int(chunks[1]),
                dropout_ratio=0.5
            )
        elif agentArchitecture.startswith('Century21_'):
            chunks = ChunkArchName(agentArchitecture)
            agent_neural_net = sumto100_arch.Century21(
                latent_size=int(chunks[1]),
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"generate_positions_expectations.main(): Not implemented agent architecture '{agentArchitecture}'")
        agent_neural_net.load_state_dict(torch.load(agentFilepath))
        agent_neural_net.to(device)
        agent = learnbyplay.player.PositionRegressionPlayer(
            identifier=agent_identifier,
            neural_net=agent_neural_net,
            temperature=temperature,
            flatten_state=flatten_state,
            acts_as_opponent=False,
            epsilon=0
        )

    opponent = learnbyplay.player.RandomPlayer(opponent_identifier)
    if opponentFilepath is not None:
        opponent_neural_net = None
        if opponentArchitecture.startswith('SaintAndre_'):
            chunks = ChunkArchName(opponentArchitecture)
            opponent_neural_net = tictactoe_arch.SaintAndre(
                latent_size=int(chunks[1]),
                dropout_ratio=0.5
            )
        elif opponentArchitecture.startswith('Coptic_'):
            chunks = ChunkArchName(opponentArchitecture)
            opponent_neural_net = tictactoe_arch.Coptic(
                number_of_channels=int(chunks[1]),
                dropout_ratio=0.5
            )
        elif opponentArchitecture.startswith('Century21_'):
            chunks = ChunkArchName(opponentArchitecture)
            opponent_neural_net = sumto100_arch.Century21(
                latent_size=int(chunks[1]),
                dropout_ratio=0.5
            )
        else:
            raise NotImplementedError(f"generate_positions_expectations.main(): Not implemented opponent architecture '{opponentArchitecture}'")
        opponent_neural_net.load_state_dict(torch.load(opponentFilepath))
        opponent_neural_net.to(device)
        opponent = learnbyplay.player.PositionRegressionPlayer(
            identifier=opponent_identifier,
            neural_net=opponent_neural_net,
            temperature=temperature,
            flatten_state=flatten_state,
            acts_as_opponent=True,
            epsilon=0
        )

    arena = Arena(authority, agent, opponent)

    position_expectation_list = arena.GeneratePositionsAndExpectations(number_of_games=numberOfGames,
                                                                       gamma=gamma,
                                                                       epsilons=epsilons)
    number_of_cells = ProductOfElements(authority.InitialState().shape)
    with open(os.path.join(outputDirectory, "dataset.csv"), "w") as output_file:
        for feature_ndx in range(number_of_cells):
            output_file.write(f"v{feature_ndx},")
        output_file.write("return\n")
        for position, expectation in position_expectation_list:
            position_vct = position.view(-1)
            for feature_ndx in range(position_vct.shape[0]):
                output_file.write(f"{position_vct[feature_ndx].item()},")
            output_file.write(f"{expectation}\n")

    logging.info(f"Done!")

    if printPositionsAndExpectations:
        for position_expectation in position_expectation_list:
            print(f"{authority.ToString(position_expectation[0])}\n{position_expectation[1]}\n\n")

def ProductOfElements(t):
    product = 1
    for i in range(len(t)):
        product *= t[i]
    return product

def ChunkArchName(arch_name):
    chunks = arch_name.split('_')
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_generate_positions_expectations'",
                        default="./output_generate_positions_expectations")
    parser.add_argument('--game', help="The game to play. Default: 'tictactoe'", default='tictactoe')
    parser.add_argument('--numberOfGames', help="The number of games. Default: 10000", type=int, default=10000)
    parser.add_argument('--gamma', help="The discount factor. Default: 0.8", type=float, default=0.8)
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--agentArchitecture', help="The architecture for the agent. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--agentFilepath', help="The filepath to the agent neural network. Default: 'None'", default='None')
    parser.add_argument('--opponentArchitecture', help="The architecture for the opponent neural network. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--opponentFilepath', help="The filepath to the opponent neural network. Default: 'None'", default='None')
    parser.add_argument('--epsilons', help="The list of epsilon parameters. Default: '[1.0, 1.0, 0.1]'", default='[1.0, 1.0, 0.1]')
    parser.add_argument('--temperature', help="The SoftMax temperature. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--printPositionsAndExpectations', help="Print the positions and expectations to the console", action='store_true')
    args = parser.parse_args()
    if args.agentFilepath.upper() == 'NONE':
        args.agentFilepath = None
    if args.opponentFilepath.upper() == 'NONE':
        args.opponentFilepath = None
    args.epsilons = ast.literal_eval(args.epsilons)
    main(
        args.outputDirectory,
        args.game,
        args.numberOfGames,
        args.gamma,
        args.randomSeed,
        args.agentArchitecture,
        args.agentFilepath,
        args.opponentArchitecture,
        args.opponentFilepath,
        args.epsilons,
        args.temperature,
        args.printPositionsAndExpectations
    )