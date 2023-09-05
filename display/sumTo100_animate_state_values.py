import logging
import matplotlib.pyplot as plt
import argparse
import os
import torch
import architectures.sumto100_arch as sumto100_arch
import imageio

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    agentFilepathPrefix,
    outputDirectory,
    architecture
):
    logging.info("sumTo100_animate_state_values.main()")

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
            f"sumTo100_animate_state_values.main(): Not implemented agent architecture '{architecture}'")

    images = []  # As read by imageio
    level = 1
    neural_net_filepath = os.path.join(agentFilepathPrefix + str(level), architecture + '.pth')
    while os.path.exists(neural_net_filepath):
        #logging.debug(f"neural_net_filepath = {neural_net_filepath}")
        neural_net.load_state_dict(torch.load(neural_net_filepath))
        neural_net.to(device)
        neural_net.eval()

        states = []
        values = []
        with open(os.path.join(outputDirectory, f'state_value{level}.csv'), 'w') as output_file:
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
        image_filepath = os.path.join(outputDirectory, f"state_values{level}.png")
        plt.savefig(image_filepath)
        images.append(imageio.imread(image_filepath))

        level += 1
        neural_net_filepath = os.path.join(agentFilepathPrefix + str(level), architecture + '.pth')

    imageio.mimsave(os.path.join(outputDirectory, "state_values.gif"), images, duration=100)

def ChunkArchName(arch_name):
    chunks = arch_name.split('_')
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agentFilepathPrefix', help="The filepath prefix for the agent neural network")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_sumTo100_animate_state_values'",
                        default="./output_sumTo100_animate_state_values")
    parser.add_argument('--architecture', help="The architecture for the agent. Default: 'Century21_512'",
                        default='Century21_512')

    args = parser.parse_args()
    main(
        args.agentFilepathPrefix,
        args.outputDirectory,
        args.architecture
    )