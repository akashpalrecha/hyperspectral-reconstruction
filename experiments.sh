echo "Activating infra-monitor environment"
source activate infra-monitor

echo "First argument is the dataset: clean, realworld"
echo "Second argument is the number of epochs"
echo ""

# $1 is the dataset name
# $2 is number of epochs

# Pretained training
python3 train.py --arch resnet34 --pretrained --loss feature_loss --epochs $2 --dataset $1

# Add improvements one by one to non-pretrained model
python3 train.py --arch resnet34 --loss feature_loss --epochs $2 --dataset $1
python3 train.py --arch mxresnet34 --loss feature_loss --epochs $2 --dataset $1
python3 train.py --arch mxresnet34 --loss feature_loss --sa_blur --epochs $2 --dataset $1

# Check MSE
python3 train.py --arch mxresnet34 --loss mse --sa_blur --epochs $2 --dataset $1

# Wild card trainings
python3 train.py --arch mxresnet18 --loss feature_loss --sa_blur --epochs $2 --dataset $1

# python3 train.py --arch mxresnet50 --loss feature_loss --sa_blur --epochs $2 --dataset $1