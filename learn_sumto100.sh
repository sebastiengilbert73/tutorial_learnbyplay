#!/bin/bash
declare -i NUMBER_OF_GAMES=5000
declare -i NUMBER_OF_EPOCHS=5

export PYTHONPATH='./'
# Windows:
#SET PYTHONPATH="./"

python ./preprocessing/generate_positions_expectations.py \
	--outputDirectory=./learn_sumTo100/output_sumto100_generate_positions_expectations_level0 \
    --game=sumto100 \
    --numberOfGames=$NUMBER_OF_GAMES \
    --gamma=0.98 \
    --randomSeed=1 \
    --agentArchitecture=None \
    --agentFilepath=None \
    --opponentArchitecture=None \
    --opponentFilepath=None \
    --epsilons="[1.0]" \
    --temperature=0
	
dataset_filepath="./learn_sumTo100/output_sumto100_generate_positions_expectations_level0/dataset.csv"
	
python ./train/train_agent.py \
		$dataset_filepath \
		--outputDirectory="./learn_sumTo100/output_sumto100_train_agent_level1" \
		--game=sumto100 \
		--randomSeed=0 \
		--validationRatio=0.2 \
		--batchSize=64 \
		--architecture=Century21_512 \
		--dropoutRatio=0 \
		--learningRate=0.001 \
		--weightDecay=0.00001 \
		--numberOfEpochs=$NUMBER_OF_EPOCHS \
		--startingNeuralNetworkFilepath=None
		
python ./utilities/sumto100_state_value.py \
	"./learn_sumTo100/output_sumto100_train_agent_level1/Century21_512.pth" \
	--outputDirectory="./learn_sumTo100/output_sumto100_train_agent_level1" \
	--architecture=Century21_512
	
for level in {1..24}
do
	dataset_filepath="./learn_sumTo100/output_sumto100_generate_positions_expectations_level${level}/dataset.csv"
	python ./preprocessing/generate_positions_expectations.py \
		--outputDirectory="./learn_sumTo100/output_sumto100_generate_positions_expectations_level${level}" \
		--game=sumto100 \
		--numberOfGames=$NUMBER_OF_GAMES \
		--gamma=0.98 \
		--randomSeed=1 \
		--agentArchitecture=Century21_512 \
		--agentFilepath="./learn_sumTo100/output_sumto100_train_agent_level${level}/Century21_512.pth" \
		--opponentArchitecture=Century21_512 \
		--opponentFilepath="./learn_sumTo100/output_sumto100_train_agent_level${level}/Century21_512.pth" \
		--epsilons="[0.5, 0.5, 0.1]" \
		--temperature=0
		
	declare -i next_level=$((level + 1))
	python ./train/train_agent.py \
		"./learn_sumTo100/output_sumto100_generate_positions_expectations_level${level}/dataset.csv" \
		--outputDirectory="./learn_sumTo100/output_sumto100_train_agent_level${next_level}" \
		--game=sumto100 \
		--randomSeed=0 \
		--validationRatio=0.2 \
		--batchSize=64 \
		--architecture=Century21_512 \
		--dropoutRatio=0 \
		--learningRate=0.001 \
		--weightDecay=0.00001 \
		--numberOfEpochs=$NUMBER_OF_EPOCHS \
		--startingNeuralNetworkFilepath="./learn_sumTo100/output_sumto100_train_agent_level${level}/Century21_512.pth"
		
	python ./utilities/sumto100_state_value.py \
		"./learn_sumTo100/output_sumto100_train_agent_level${next_level}/Century21_512.pth" \
		--outputDirectory="./learn_sumTo100/output_sumto100_train_agent_level${next_level}" \
		--architecture=Century21_512
done