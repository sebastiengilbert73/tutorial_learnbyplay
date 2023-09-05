import argparse
import logging
import os
import architectures.sumto100_arch as sumto100_arch
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    neuralNetworkFilepath,
    outputDirectory,
    architecture
):
    logging.info("sumto100_state_value.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the neural network
    neural_net = None
    if architecture.startswith('Century21_'):
        chunks = ArchitectureChunks(architecture)
        neural_net = sumto100_arch.Century21(latent_size=int(chunks[1]), dropout_ratio=0)
    else:
        raise NotImplementedError(f"sumto100_state_value.main(): Not implemented architecture '{architecture}'")
    neural_net.load_state_dict(torch.load(neuralNetworkFilepath))
    neural_net.eval()

    with open(os.path.join(outputDirectory, 'state_value.csv'), 'w') as output_file:
        output_file.write("state,value\n")
        for s in range(0, 101):
            state = torch.zeros((1, 101), dtype=torch.float)
            state[0, s] = 1.0
            prediction = neural_net(state).item()
            print(prediction)
            output_file.write(f"{s},{prediction}\n")
def ArchitectureChunks(arch):
    chunks = arch.split('_')
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('neuralNetworkFilepath', help="The filepath to the agent neural network")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_sumto100_state_value'",
                        default="./output_sumto100_state_value")
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'Century21_512'",
                        default='Century21_512')
    args = parser.parse_args()
    main(
        args.neuralNetworkFilepath,
        args.outputDirectory,
        args.architecture
    )