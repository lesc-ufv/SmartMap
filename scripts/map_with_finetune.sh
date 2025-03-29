#! /bin/bash
mkdir -p results/mappings/finetune/

path_config_modules="configs/"
config_file=$1
config_class=$2
arch=$3
arch_dim=$4

echo ""
echo "Mapping (finetune) test DFGs using the trained model from config $config_class for the architecture $arch with dimensions $arch_dim."
echo "The mapping information is being written to results/mappings/finetune/${config_class}_${arch}_${arch_dim}_finetune.txt" 
echo ""
python3 map_with_finetune_by_config.py $path_config_modules$config_file $config_class $arch $arch_dim TEST > results/mappings/finetune/${config_class}_${arch}_${arch_dim}_finetune.txt
echo ""
echo "------------------------------------------------------------------------------------------------------------------------"
