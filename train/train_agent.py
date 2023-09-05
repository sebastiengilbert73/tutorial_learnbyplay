import torch
from torch.utils.data import Dataset, DataLoader
import logging
import pandas as pd
import argparse
import random
import os
import architectures.tictactoe_arch as tictactoe_arch
import architectures.sumto100_arch as sumto100_arch
import ast
import einops

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

class PositionExpectation(Dataset):
    def __init__(self, dataset_filepath, state_shape, unflatten_state):
        super(PositionExpectation, self).__init__()
        self.dataset_filepath = dataset_filepath
        self.df = pd.read_csv(self.dataset_filepath)
        self.number_of_entries = len(self.df.columns) - 1  # v0, v1, ..., vN, return
        self.state_shape = state_shape
        self.unflatten_state = unflatten_state

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        position_tsr = torch.tensor(self.df.iloc[idx]['v0': f'v{self.number_of_entries - 1}']).float()  # v0, v1, ..., vN
        if self.unflatten_state:
            position_tsr = position_tsr.view(self.state_shape)
        expected_return_tsr = torch.tensor(self.df.iloc[idx]['return']).float().unsqueeze(0)
        return position_tsr, expected_return_tsr

def main(
    datasetFilepath,
    outputDirectory,
    game,
    randomSeed,
    validationRatio,
    batchSize,
    architecture,
    dropoutRatio,
    useCpu,
    learningRate,
    weightDecay,
    numberOfEpochs,
    startingNeuralNetworkFilepath
):
    device = 'cpu'
    if not useCpu and torch.cuda.is_available():
        device = 'cuda'

    logging.info(f"train_agent.main(); torch.cuda.is_available() = {torch.cuda.is_available()}; device = {device}; game = {game}; architecture = {architecture}")

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    state_shape = None
    unflatten_state = None
    if game == 'tictactoe':
        state_shape = (2, 3, 3)
        unflatten_state = False
    elif game == 'sumto100':
        state_shape = (101,)
        unflatten_state = False
    else:
        raise NotImplementedError(f"train_agent.main(): Not implemented game '{game}'")

    # Load the dataset
    dataset = PositionExpectation(dataset_filepath=datasetFilepath, state_shape=state_shape, unflatten_state=unflatten_state)

    # Split the dataset into training and validation
    number_of_validation_observations = round(validationRatio * len(dataset))
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset,
                                                                      [len(dataset) - number_of_validation_observations,
                                                                       number_of_validation_observations])
    logging.info(f"len(train_dataset) = {len(train_dataset)}")
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batchSize, shuffle=False)

    # Create the neural network
    neural_net = None
    if architecture.startswith('SaintAndre_'):
        chunks = SplitArchName(architecture)  # ['SaintAndre', '512']
        neural_net = tictactoe_arch.SaintAndre(
            latent_size=int(chunks[1]),
            dropout_ratio=dropoutRatio
        )
    elif architecture.startswith('Coptic_'):
        chunks = SplitArchName(architecture)  # ['Coptic', '512']
        neural_net = tictactoe_arch.Coptic(
            number_of_channels=int(chunks[1]),
            dropout_ratio=dropoutRatio
        )
    elif architecture.startswith('Century21_'):
        chunks = SplitArchName(architecture)
        neural_net = sumto100_arch.Century21(
            latent_size=int(chunks[1]),
            dropout_ratio=dropoutRatio
        )
    else:
        raise NotImplementedError(f"train_agent.main(): Not implemented architecture '{architecture}'")
    if startingNeuralNetworkFilepath is not None:
        neural_net.load_state_dict(torch.load(startingNeuralNetworkFilepath))

    neural_net.to(device)

    # Training parameters
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learningRate, weight_decay=weightDecay)

    # Training loop
    minimum_validation_loss = float('inf')
    with open(os.path.join(outputDirectory, "epochLoss.csv"), 'w') as epoch_loss_file:
        epoch_loss_file.write("epoch,training_loss,validation_loss,is_champion\n")
        for epoch in range(1, numberOfEpochs + 1):
            # Set the neural network to training mode
            neural_net.train()
            running_loss = 0.0
            number_of_batches = 0
            for input_tsr, target_tsr in train_dataloader:
                # Move the tensors to the accelerator device
                input_tsr = input_tsr.to(device)
                target_tsr = target_tsr.to(device)
                # Set the parameter gradients to zero before every batch
                neural_net.zero_grad()
                # Pass the input tensor through the neural network
                output_tsr = neural_net(input_tsr)
                # Compute the loss, i.e., the error function we want to minimize
                loss = criterion(output_tsr, target_tsr)
                # Retropropagate the loss function, to compute the gradient of the loss function with
                # respect to every trainable parameter in the neural network
                loss.backward()
                # Perturb every trainable parameter by a small quantity, in the direction of the steepest loss descent
                optimizer.step()

                running_loss += loss.item()
                number_of_batches += 1
                if number_of_batches %100 == 1:
                    print('.', end='', flush=True)
            average_training_loss = running_loss / number_of_batches

            # Evaluate with the validation dataset
            # Set the neural network to evaluation (inference) mode
            neural_net.eval()
            validation_running_loss = 0.0
            number_of_batches = 0
            for validation_input_tsr, validation_target_output_tsr in validation_dataloader:
                # Move the tensors to the accelerator device
                validation_input_tsr = validation_input_tsr.to(device)
                validation_target_output_tsr = validation_target_output_tsr.to(device)
                # Pass the input tensor through the neural network
                validation_output_tsr = neural_net(validation_input_tsr)
                # Compute the validation loss
                validation_loss = criterion(validation_output_tsr, validation_target_output_tsr)
                validation_running_loss += validation_loss.item()
                number_of_batches += 1
            average_validation_loss = validation_running_loss / number_of_batches
            is_champion = False
            if average_validation_loss < minimum_validation_loss:
                minimum_validation_loss = average_validation_loss
                is_champion = True
                champion_filepath = os.path.join(outputDirectory, f"{architecture}.pth")
                torch.save(neural_net.state_dict(), champion_filepath)
            logging.info(
                f" **** Epoch {epoch}: average_training_loss = {average_training_loss}; average_validation_loss = {average_validation_loss}")
            if is_champion:
                logging.info(f" ++++ Champion for validation loss ({average_validation_loss}) ++++")
            epoch_loss_file.write(f"{epoch},{average_training_loss},{average_validation_loss},{is_champion}\n")


def SplitArchName(arch_name):
    chunks = arch_name.split('_')
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datasetFilepath', help="The filepath to the csv file giving the positions and the expected return")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_train_agent'",
                        default="./output_train_agent")
    parser.add_argument('--game', help="The game. Default: 'tictactoe'", default='tictactoe')
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--validationRatio', help="The proportion of examples used for validation. Default: 0.2", type=float, default=0.2)
    parser.add_argument('--batchSize', help="The batch size. Default: 16", type=int, default=16)
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'SaintAndre_512'", default='SaintAndre_512')
    parser.add_argument('--dropoutRatio', help="The dropout ratio (if applicable). Default: 0.5", type=float, default=0.5)
    parser.add_argument('--useCpu', help="Use the CPU, even if a GPU is available", action='store_true')
    parser.add_argument('--learningRate', help="The learning rate. Default: 0.001", type=float, default=0.001)
    parser.add_argument('--weightDecay', help="The weight decay. Default: 0.00001", type=float, default=0.00001)
    parser.add_argument('--numberOfEpochs', help="The number of epochs. Default: 50", type=int, default=50)
    parser.add_argument('--startingNeuralNetworkFilepath', help="The filepath to the starting neural network. Default: 'None'", default='None')
    args = parser.parse_args()
    if args.startingNeuralNetworkFilepath.upper() == 'NONE':
        args.startingNeuralNetworkFilepath = None
    main(
        args.datasetFilepath,
        args.outputDirectory,
        args.game,
        args.randomSeed,
        args.validationRatio,
        args.batchSize,
        args.architecture,
        args.dropoutRatio,
        args.useCpu,
        args.learningRate,
        args.weightDecay,
        args.numberOfEpochs,
        args.startingNeuralNetworkFilepath
    )