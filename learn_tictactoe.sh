#!/bin/bash
declare -i NUMBER_OF_GAMES=30000
declare -i NUMBER_OF_EPOCHS=5

export PYTHONPATH='./'

python preprocessing/generate_positions_expectations.py \
	--outputDirectory=./learn_tictactoe/output_tictactoe_generate_positions_expectations_level0 \
    --game=tictactoe \
    --numberOfGames=$NUMBER_OF_GAMES \
    --gamma=0.95 \
    --randomSeed=1 \
    --agentArchitecture=None \
    --agentFilepath=None \
    --opponentArchitecture=None \
    --opponentFilepath=None \
    --epsilons="[1.0]" \
    --temperature=0
	
dataset_filepath="./learn_tictactoe/output_tictactoe_generate_positions_expectations_level0/dataset.csv"
	
python train/train_agent.py \
		$dataset_filepath \
		--outputDirectory="./learn_tictactoe/output_tictactoe_train_agent_level1" \
		--game=tictactoe \
		--randomSeed=0 \
		--validationRatio=0.2 \
		--batchSize=64 \
		--architecture=SaintAndre_1024 \
		--dropoutRatio=0.5 \
		--learningRate=0.0001 \
		--weightDecay=0.00001 \
		--numberOfEpochs=$NUMBER_OF_EPOCHS \
		--startingNeuralNetworkFilepath=None
		
	
for level in {1..16}
do
	dataset_filepath="./learn_tictactoe/output_tictactoe_generate_positions_expectations_level${level}/dataset.csv"
	python preprocessing/generate_positions_expectations.py \
		--outputDirectory="./learn_tictactoe/output_tictactoe_generate_positions_expectations_level${level}" \
		--game=tictactoe \
		--numberOfGames=$NUMBER_OF_GAMES \
		--gamma=0.95 \
		--randomSeed=0 \
		--agentArchitecture=SaintAndre_1024 \
		--agentFilepath="./learn_tictactoe/output_tictactoe_train_agent_level${level}/SaintAndre_1024.pth" \
		--opponentArchitecture=SaintAndre_1024 \
		--opponentFilepath="./learn_tictactoe/output_tictactoe_train_agent_level${level}/SaintAndre_1024.pth" \
		--epsilons="[0.5, 0.5, 0.1]" \
		--temperature=0 
		
	declare -i next_level=$((level + 1))
	python train/train_agent.py \
		"./learn_tictactoe/output_tictactoe_generate_positions_expectations_level${level}/dataset.csv" \
		--outputDirectory="./learn_tictactoe/output_tictactoe_train_agent_level${next_level}" \
		--game=tictactoe \
		--randomSeed=0 \
		--validationRatio=0.2 \
		--batchSize=64 \
		--architecture=SaintAndre_1024 \
		--dropoutRatio=0.5 \
		--learningRate=0.0001 \
		--weightDecay=0.00001 \
		--numberOfEpochs=$NUMBER_OF_EPOCHS \
		--startingNeuralNetworkFilepath="./learn_tictactoe/output_tictactoe_train_agent_level${level}/SaintAndre_1024.pth"
		
done
