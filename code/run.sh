if [ $# -ne 1 ]; then
    echo "Only 1 parameters are required!"
    exit
fi

experiment_type=$1
if [ $experiment_type -eq 2 ]; then
    num_classes=3
else
    num_classes=2
fi

if [ ! -d logs/${experiment_type} ]; then
    mkdir -p logs/${experiment_type}
fi

python -u train.py \
    --train_json_path dataset/json/train-type-${experiment_type}.json \
    --val_json_path dataset/json/val-type-${experiment_type}.json \
    --test_json_path dataset/json/test-type-${experiment_type}.json \
    --clinical_data_path dataset/clinical_data/preprocessed-type-${experiment_type}.xlsx \
    --num_classes ${num_classes} \
    --backbone vgg16_bn \
    --log_name ${experiment_type} | tee logs/${experiment_type}/output.log 2>&1
