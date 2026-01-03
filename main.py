import warnings
from args import Args
from utils.train import train
from utils.util import *
from utils.data_loader import *
from model.acgcn_mmp import ACGCN_MMP
from model.acgcn_sub import ACGCN_SUB
from model.acgcn_snn_base import ACGCN_SNN
from model.attentivefp_mmp import AttentiveFP_MMP
from model.attentivefp_snn import AttentiveFP_SNN
from model.weave_mmp import Weave_MMP
from model.weave_snn import Weave_SNN
from model.adapter_mmp import Adapter_MMP
from model.adapter_snn import Adapter_SNN
from model.attention_mmp import Attention_MMP
from model.attention_snn import Attention_SNN
from model.attention_diff import Attention_DIFF


from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

def main(args):

    device = args['DEVICE']
    random_seed = args['RANDOM_SEED']

    ### Make dataset ###
    data = pd.read_csv('./data/' + args['TARGET_NAME'] + '_mmps.csv')

    # Use regression target 'delta' (float)
    # ensure delta is float
    data['delta'] = data['delta'].astype(float)

    # For regression do NOT stratify by label (we use delta)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)

    if args['MODEL'] == 'acgcn-mmp':
        model = ACGCN_MMP(args).to(device)
        train_loader = ACGCN_MMP_Dataset(args, train_data, True)
        test_loader = ACGCN_MMP_Dataset(args, test_data, False)

    elif args['MODEL'] == 'acgcn-sub':
        model = ACGCN_SUB(args).to(device)
        train_loader = ACGCN_SUB_Dataset(args, train_data, True)
        test_loader = ACGCN_SUB_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'acgcn-snn':
        model = ACGCN_SNN(args).to(device)
        train_loader = ACGCN_MMP_Dataset(args, train_data, True)
        test_loader = ACGCN_MMP_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'attentivefp-mmp':
        model = AttentiveFP_MMP(args).to(device)
        train_loader = ACGCN_MMP_Dataset(args, train_data, True)
        test_loader = ACGCN_MMP_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'weave-mmp':
        model = Weave_MMP(args).to(device)
        train_loader = ACGCN_MMP_Dataset(args, train_data, True)
        test_loader = ACGCN_MMP_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'attentivefp-snn':
        model = AttentiveFP_SNN(args).to(device)
        train_loader = ACGCN_MMP_Dataset(args, train_data, True)
        test_loader = ACGCN_MMP_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'weave-snn':
        model = Weave_SNN(args).to(device)
        train_loader = ACGCN_MMP_Dataset(args, train_data, True)
        test_loader = ACGCN_MMP_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'adapter-mmp':
        model = Adapter_MMP(args).to(device)
        
        dataset = args['TARGET_NAME']
        feature_dir = f"data/{dataset}_kpgt.npz"
        model.load_kpgt_features(feature_dir)
        
        train_loader = ADAPTER_MMP_Dataset(args, train_data, True)
        test_loader = ADAPTER_MMP_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'adapter-snn':
        model = Adapter_SNN(args).to(device)
        
        dataset = args['TARGET_NAME']
        feature_dir = f"data/{dataset}_kpgt.npz"
        model.load_kpgt_features(feature_dir)
        
        train_loader = ADAPTER_MMP_Dataset(args, train_data, True)
        test_loader = ADAPTER_MMP_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'attention-mmp':
        model = Attention_MMP(args).to(device)
        
        dataset = args['TARGET_NAME']
        feature_dir = f"data/{dataset}_kpgt.npz"
        model.load_kpgt_features(feature_dir)
        
        train_loader = ADAPTER_MMP_Dataset(args, train_data, True)
        test_loader = ADAPTER_MMP_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'attention-snn':
        model = Attention_SNN(args).to(device)
        
        dataset = args['TARGET_NAME']
        feature_dir = f"data/{dataset}_kpgt.npz"
        model.load_kpgt_features(feature_dir)
        
        train_loader = ADAPTER_MMP_Dataset(args, train_data, True)
        test_loader = ADAPTER_MMP_Dataset(args, test_data, False)
    
    elif args['MODEL'] == 'attention-diff':
        model = Attention_DIFF(args).to(device)
        
        dataset = args['TARGET_NAME']
        feature_dir = f"data/{dataset}_kpgt.npz"
        model.load_kpgt_features(feature_dir)
        
        train_loader = ADAPTER_MMP_Dataset(args, train_data, True)
        test_loader = ADAPTER_MMP_Dataset(args, test_data, False)
    
    # get actual target values from test loader
    y_actual = get_actual_label(test_loader)

    # train returns predictions for test set (list of floats)
    y_pred = train(args, model, train_loader, test_loader)

    # print regression metrics
    print_regression_metrics(y_pred, y_actual)

if __name__ == '__main__':

    args = Args().params
    print(args)
    
    random_seed = args['RANDOM_SEED']
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    main(args)