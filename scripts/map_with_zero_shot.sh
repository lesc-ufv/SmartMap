#! /bin/bash

mkdir -p results/mappings/zero_shot/

path_config_modules="configs/"
config_file=$1
config_class=$2
arch=$3
arch_dim=$4

echo ""
echo "Mapping (zero shot) test DFGs using the trained model from config $config_class for the architecture $arch with dimensions $arch_dim."
echo "The mapping information is being written to results/mappings/${config_class}_${arch}_${arch_dim}_zero_shot.txt" 
echo ""
python3 map_with_zero_shot_by_config.py $path_config_modules$config_file $config_class $arch $arch_dim TEST > results/mappings/zero_shot/${config_class}_${arch}_${arch_dim}_zero_shot.txt
echo ""
echo "------------------------------------------------------------------------------------------------------------------------"
