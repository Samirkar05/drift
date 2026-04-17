from src.finetune import train_drift
from src.args import parse_arguments
from src.eval import evaluate
data_location = "/data/139-1/datasets/merging"
models = ['ViT-B-32']
datasets = ['MNIST', 'SVHN', "Cars", "SUN397","RESISC45","GTSRB","EuroSAT","DTD"] 

epochs = {
    'Cars': 35,
    'DTD': 76,
    'EuroSAT': 12,
    'GTSRB': 11,
    'MNIST': 5,
    'RESISC45': 15,
    'SUN397': 14,
    'SVHN': 4
}

for model in models:
    for dataset in datasets:
        print('='*100)
        print(f'Finetuning {model} on {dataset}')
        print('='*100)
        args = parse_arguments()
        args.lr = 1e-5
        args.epochs = epochs[dataset]
        args.data_location = data_location
        args.train_dataset = dataset + 'Val'
        args.batch_size = 128 # 128 seems like GPUs can't handle this much batch size
        args.model = model
        args.save = f'/data/139-1/users/selkarrat/checkpoints/{model}'
        args.load = f'/data/139-1/users/selkarrat/checkpoints/{model}'
        train_drift(args)
