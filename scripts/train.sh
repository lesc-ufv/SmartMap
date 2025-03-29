#! /bin/bash
mkdir -p results/mappings/

path_config_modules="configs/"
config_file=$1
config_class=$2
arch=$3
arch_dim=$4

echo ""
echo "Training the model from config $config_class for the architecture $arch with dimensions $arch_dim."
echo ""
python3 train_by_config.py $path_config_modules$config_file $config_class $arch $arch_dim TRAIN
echo ""
echo "------------------------------------------------------------------------------------------------------------------------"