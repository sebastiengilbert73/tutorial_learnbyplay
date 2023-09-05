import logging
import matplotlib.pyplot as plt
import argparse
import os
import torch
import architectures.sumto100_arch as sumto100_arch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    agentFilepath,
    outputDirectory,
    architecture
):
    logging.info("sumTo100_state_value.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    device = 'cpu'

    neural_net = None
    if architecture.startswith('Century21_'):
        chunks = ChunkArchName(architecture)
        neural_net = sumto100_arch.Century21(
            latent_size=int(chunks[1]),
            dropout_ratio=0.5
        )
    else:
        raise NotImplementedError(
            f"sumTo100_state_value.main(): Not implemented agent architecture '{architecture}'")
    neural_net.load_state_dict(torch.load(agentFilepath))
    neural_net.to(device)
    neural_net.eval()

    states = []
    values = []
    with open(os.path.join(outputDirectory, 'state_value.csv'), 'w') as output_file:
        output_file.write("state,value\n")
        for s in range(0, 101):
            state = torch.zeros((1, 101), dtype=torch.float)
            state[0, s] = 1.0
            prediction = neural_net(state).item()
            print(prediction)
            output_file.write(f"{s},{prediction}\n")
            states.append(s)
            values.append(prediction)

    fig, ax = plt.subplots()
    ax.bar(states, values)
    ax.set_ylabel("Value")
    ax.set_xlabel("State (i.e. the sum after the agent has played)")
    ax.set_title("Predicted State Values")

    plt.show()

def ChunkArchName(arch_name):
    chunks = arch_name.split('_')
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agentFilepath', help="The filepath to the agent neural network")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_sumTo100_state_value'",
                        default="./output_sumTo100_state_value")
    parser.add_argument('--architecture', help="The architecture for the agent. Default: 'Century21_512'",
                        default='Century21_512')

    args = parser.parse_args()
    main(
        args.agentFilepath,
        args.outputDirectory,
        args.architecture
    )