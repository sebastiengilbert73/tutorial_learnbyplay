import argparse
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
import random

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    inputTrainingDirectoryPrefix,
    outputDirectory
):
    logging.info("training_losses_rounds.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    epochs_list_list = []
    training_loss_list_list = []
    validation_loss_list_list = []

    level = 1
    training_directory = inputTrainingDirectoryPrefix + str(level)
    while os.path.exists(training_directory):
        #logging.debug(f"training_directory = {training_directory}")
        epoch_loss_filepath = os.path.join(training_directory, 'epochLoss.csv')
        epoch_loss_df = pd.read_csv(epoch_loss_filepath)
        epochs_list = epoch_loss_df['epoch'].tolist()
        #logging.debug(f"epochs_list = {epochs_list}")
        training_loss_list = epoch_loss_df['training_loss'].tolist()
        validation_loss_list = epoch_loss_df['validation_loss'].tolist()

        epochs_list_list.append(epochs_list)
        training_loss_list_list.append(training_loss_list)
        validation_loss_list_list.append(validation_loss_list)

        level += 1
        training_directory = inputTrainingDirectoryPrefix + str(level)

    epochs_list = epochs_list_list[0]
    for i in range(1, len(epochs_list_list)):
        if epochs_list_list[i] != epochs_list:
            raise ValueError(f"epochs_list_list[i] ({epochs_list_list[i]}) != epochs_list ({epochs_list})")

    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots()
    plt.ylim(0, 1.0)

    for i in range(len(training_loss_list_list)):
        level = i + 1
        if level %2 == 1:
            training_loss_list = training_loss_list_list[i]
            validation_loss_list = validation_loss_list_list[i]
            #color = random_color()
            #ax.plot(epochs_list, training_loss_list, '--', color=color, label="training level " + str(i + 1))
            ax.plot(epochs_list, validation_loss_list, label=f"{level} rounds")

    plt.legend(loc='best', ncol=3)
    plt.xticks(epochs_list)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss (MSE)')

    plt.show()

def random_color():  # Cf. https://www.tutorialspoint.com/how-to-generate-random-colors-in-matplotlib
    hexadecimal_alphabets = '0123456789ABCDEF'
    color = "#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)])
    return color


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputTrainingDirectoryPrefix', help="The prefix to the directories that contain the 'epochLoss.csv' files")
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './output_training_losses_rounds'",
                        default="./output_training_losses_rounds")
    args = parser.parse_args()
    main(
        args.inputTrainingDirectoryPrefix,
        args.outputDirectory
    )