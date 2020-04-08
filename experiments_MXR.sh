echo "Activating infra-monitor environment"
source activate infra-monitor

echo "First argument is the dataset: clean, realworld"
echo "Second argument is the number of epochs"
echo ""

# $1 is the dataset name
# $2 is number of epochs


# Add improvements one by one to non-pretrained model
python3 train_mxr.py --arch mxresnet34 --loss feature_loss --epochs $2 --dataset $1 --save_prefix ExpMXR

