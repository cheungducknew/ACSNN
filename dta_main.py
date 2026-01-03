import warnings
import pandas as pd
import numpy as np
import torch
import random
from args import Args
from utils.train import train
from utils.util import *
from utils.dta_data_loader import DeepDTA_Dataset, build_vocabularies
from dta_model.deepdta import DeepDTA_MMP, DeepDTA_SNN
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def main(args):

    device = args['DEVICE']
    random_seed = args['RANDOM_SEED']

    ### Make dataset ###
    data = pd.read_csv('./data/bindingdb.csv')

    # Ensure delta is float
    data['delta'] = data['delta'].astype(float)

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)

    if args['MODEL'] in ('deepdta-mmp', 'deepdta-snn'):
        # 为DeepDTA_MMP和DeepDTA_SNN构建词汇表
        smiles_dict, fasta_dict, smiles_vocab_size, fasta_vocab_size = build_vocabularies(
            data, max_smiles_len=200, max_fasta_len=2000
        )
        
        # 初始化相应的模型
        if args['MODEL'] == 'deepdta-mmp':
            model = DeepDTA_MMP(
                pro_vocab_size=fasta_vocab_size, 
                lig_vocab_size=smiles_vocab_size, 
                channel=32, 
                protein_kernel_size=8, 
                ligand_kernel_size=6
            ).to(device)
        elif args['MODEL'] == 'deepdta-snn':
            model = DeepDTA_SNN(
                pro_vocab_size=fasta_vocab_size, 
                lig_vocab_size=smiles_vocab_size, 
                channel=32, 
                protein_kernel_size=8, 
                ligand_kernel_size=6
            ).to(device)
        
        # 创建训练和测试数据加载器，共享词汇表
        train_loader = DeepDTA_Dataset(args, train_data, True, drop_last=False, smiles_dict=smiles_dict, fasta_dict=fasta_dict)
        test_loader = DeepDTA_Dataset(args, test_data, False, drop_last=False, smiles_dict=smiles_dict, fasta_dict=fasta_dict)
    else:
        raise ValueError(f"Model {args['MODEL']} is not supported in dta_main.py")

    # Get actual target values from test loader
    y_actual = get_actual_label(test_loader)

    # Train the model and get predictions for test set
    y_pred = train(args, model, train_loader, test_loader)

    # Print regression metrics
    print_regression_metrics(y_pred, y_actual)


if __name__ == '__main__':

    args = Args().params
    
    random_seed = args['RANDOM_SEED']
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    main(args)
