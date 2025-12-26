import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
import pandas as pd


def calculate_maccs_fingerprint(smiles):
    """
    计算MACCS分子指纹
    :param smiles: 分子的SMILES字符串
    :return: MACCS指纹向量
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    except:
        return None


def calculate_ecfp_fingerprint(smiles, radius=2, n_bits=2048):
    """
    计算ECFP分子指纹
    :param smiles: 分子的SMILES字符串
    :param radius: ECFP半径
    :param n_bits: 指纹位数
    :return: ECFP指纹向量
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    except:
        return None


def concatenate_fingerprints(fp1, fp2):
    """
    拼接两个指纹
    :param fp1: 第一个指纹
    :param fp2: 第二个指纹
    :return: 拼接后的指纹向量
    """
    return np.concatenate([fp1, fp2])


def preprocess_data_with_fingerprints(data_path):
    """
    数据预处理，计算分子指纹
    :param data_path: 数据集路径
    :return: 处理后的数据集
    """
    df = pd.read_csv(data_path)
    print(f"原始数据大小: {len(df)}")
    
    # 计算molregno1和molregno2对应的MACCS和ECFP指纹
    df['maccs1'] = df['smiles1'].apply(calculate_maccs_fingerprint)
    df['maccs2'] = df['smiles2'].apply(calculate_maccs_fingerprint)
    df['ecfp1'] = df['smiles1'].apply(calculate_ecfp_fingerprint)
    df['ecfp2'] = df['smiles2'].apply(calculate_ecfp_fingerprint)
    
    # 删除指纹计算失败的样本
    df = df[df['maccs1'].notnull() & df['maccs2'].notnull() & 
            df['ecfp1'].notnull() & df['ecfp2'].notnull()]
    print(f"处理后数据大小: {len(df)}")
    
    return df


def prepare_features(df, fingerprint_type):
    """
    准备特征和标签
    :param df: 数据集
    :param fingerprint_type: 指纹类型 ('maccs' 或 'ecfp')
    :return: 特征矩阵X和标签向量y
    """
    if fingerprint_type == 'maccs':
        # 使用MACCS指纹拼接作为特征
        X = np.array(df.apply(lambda row: concatenate_fingerprints(row['maccs1'], row['maccs2']), axis=1).tolist())
    elif fingerprint_type == 'ecfp':
        # 使用ECFP指纹拼接作为特征
        X = np.array(df.apply(lambda row: concatenate_fingerprints(row['ecfp1'], row['ecfp2']), axis=1).tolist())
    else:
        raise ValueError("指纹类型必须是 'maccs' 或 'ecfp'")
    
    # 标签是delta值
    y = df['delta'].values
    
    return X, y
