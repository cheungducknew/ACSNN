import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder
from collections import Counter


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


# 氨基酸物理化学性质字典（模拟AAindex的544种性质，这里选取了一些常见性质）
# 每个氨基酸对应一个性质向量
AMINO_ACID_PROPERTIES = {
    'A': [1.8, 0.0, 89.09, 0, 6.00, 88.6, 0, 0],  # 疏水性, 极性, 分子量, 电荷, 等电点, 侧链体积, 氢键供体, 氢键受体
    'C': [2.5, 0.0, 121.15, 0, 5.07, 108.5, 0, 1],
    'D': [-3.5, 1.0, 133.10, -1, 2.77, 111.1, 0, 2],
    'E': [-3.5, 1.0, 147.13, -1, 3.22, 138.4, 0, 2],
    'F': [2.8, 0.0, 165.19, 0, 5.48, 189.9, 0, 1],
    'G': [-0.4, 0.0, 75.07, 0, 5.97, 60.1, 0, 0],
    'H': [-3.2, 1.0, 155.16, 0, 7.59, 147.9, 1, 1],
    'I': [4.5, 0.0, 131.17, 0, 6.02, 166.7, 0, 0],
    'K': [-3.9, 1.0, 146.19, 1, 9.74, 168.6, 1, 1],
    'L': [3.8, 0.0, 131.17, 0, 5.98, 166.7, 0, 0],
    'M': [1.9, 0.0, 149.21, 0, 5.74, 162.9, 0, 1],
    'N': [-3.5, 1.0, 132.12, 0, 5.41, 114.1, 1, 2],
    'P': [-1.6, 0.0, 115.13, 0, 6.30, 112.7, 0, 0],
    'Q': [-3.5, 1.0, 146.15, 0, 5.65, 143.8, 1, 2],
    'R': [-4.5, 1.0, 174.20, 1, 10.76, 173.4, 2, 1],
    'S': [-0.8, 1.0, 105.09, 0, 5.68, 89.0, 0, 1],
    'T': [-0.7, 1.0, 119.12, 0, 5.60, 116.1, 0, 1],
    'V': [4.2, 0.0, 117.15, 0, 5.96, 140.0, 0, 0],
    'W': [-0.9, 0.0, 204.23, 0, 5.89, 227.8, 1, 1],
    'Y': [-1.3, 1.0, 181.19, 0, 5.66, 193.6, 0, 2],
}

# 定义氨基酸映射
def amino_acid_encoder():
    """
    创建氨基酸编码映射
    :return: 氨基酸到整数的映射
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return {aa: i for i, aa in enumerate(amino_acids)}


# 简单的蛋白质序列特征提取：氨基酸组成
def calculate_amino_acid_composition(sequence):
    """
    计算蛋白质序列的氨基酸组成特征
    :param sequence: 蛋白质序列
    :return: 氨基酸组成特征向量
    """
    aa_dict = amino_acid_encoder()
    composition = [0] * len(aa_dict)
    
    # 统计每个氨基酸的出现频率
    counter = Counter(sequence)
    total = len(sequence)
    
    for aa, count in counter.items():
        if aa in aa_dict:
            composition[aa_dict[aa]] = count / total
    
    return np.array(composition)


# 基于AAindex的蛋白质序列特征提取：物理化学性质统计
def calculate_aaindex_properties(sequence):
    """
    计算蛋白质序列的物理化学性质统计特征（模拟AAindex）
    :param sequence: 蛋白质序列
    :return: 物理化学性质统计特征向量
    """
    # 收集所有氨基酸的性质向量
    all_properties = []
    for aa in sequence:
        if aa in AMINO_ACID_PROPERTIES:
            all_properties.append(AMINO_ACID_PROPERTIES[aa])
    
    if not all_properties:
        return np.zeros(4 * len(list(AMINO_ACID_PROPERTIES.values())[0]))
    
    all_properties = np.array(all_properties)
    num_properties = all_properties.shape[1]
    
    # 计算每个性质的统计指标：平均值、标准差、最小值、最大值
    features = []
    for i in range(num_properties):
        prop_values = all_properties[:, i]
        features.extend([
            np.mean(prop_values),
            np.std(prop_values),
            np.min(prop_values),
            np.max(prop_values)
        ])
    
    return np.array(features)


# 基于AAindex的蛋白质序列特征提取：位置特异性性质
# 这里实现一个简化版本，对序列的不同区域计算性质统计
def calculate_positional_aaindex_properties(sequence):
    """
    计算蛋白质序列不同区域的物理化学性质统计
    :param sequence: 蛋白质序列
    :return: 位置特异性物理化学性质特征向量
    """
    if len(sequence) < 3:
        # 如果序列太短，直接返回全零向量
        return np.zeros(3 * 4 * len(list(AMINO_ACID_PROPERTIES.values())[0]))
    
    # 将序列分为三个部分：N端、中间、C端
    split1 = len(sequence) // 3
    split2 = 2 * len(sequence) // 3
    
    n_term = sequence[:split1]
    middle = sequence[split1:split2]
    c_term = sequence[split2:]
    
    # 计算每个区域的AAindex性质
    n_features = calculate_aaindex_properties(n_term)
    mid_features = calculate_aaindex_properties(middle)
    c_features = calculate_aaindex_properties(c_term)
    
    # 拼接所有特征
    return np.concatenate([n_features, mid_features, c_features])


# 蛋白质序列的n-gram特征
def calculate_ngram_features(sequence, n=2):
    """
    计算蛋白质序列的n-gram特征
    :param sequence: 蛋白质序列
    :param n: n-gram的长度
    :return: n-gram特征向量
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    ngram_size = len(amino_acids) ** n
    
    # 创建n-gram到索引的映射
    ngram_dict = {}
    index = 0
    
    for aa1 in amino_acids:
        if n == 1:
            ngram_dict[aa1] = index
            index += 1
        elif n == 2:
            for aa2 in amino_acids:
                ngram_dict[aa1 + aa2] = index
                index += 1
        elif n == 3:
            for aa2 in amino_acids:
                for aa3 in amino_acids:
                    ngram_dict[aa1 + aa2 + aa3] = index
                    index += 1
    
    # 初始化特征向量
    features = [0] * ngram_size
    
    # 统计n-gram出现频率
    total = len(sequence) - n + 1
    if total <= 0:
        return np.array(features)
    
    for i in range(total):
        ngram = sequence[i:i+n]
        if ngram in ngram_dict:
            features[ngram_dict[ngram]] += 1 / total
    
    return np.array(features)


# 组合蛋白质特征
def calculate_protein_features(sequence):
    """
    组合多种蛋白质序列特征
    :param sequence: 蛋白质序列
    :return: 组合后的蛋白质特征向量
    """
    # 氨基酸组成特征
    aa_comp = calculate_amino_acid_composition(sequence)
    
    # 2-gram特征
    gram2 = calculate_ngram_features(sequence, n=2)
    
    # 基于AAindex的物理化学性质统计特征
    aaindex_props = calculate_aaindex_properties(sequence)
    
    # 基于AAindex的位置特异性物理化学性质特征
    positional_props = calculate_positional_aaindex_properties(sequence)
    
    # 3-gram特征（可选）
    # gram3 = calculate_ngram_features(sequence, n=3)
    
    # 组合所有特征
    return np.concatenate([aa_comp, gram2, aaindex_props, positional_props])


# 数据预处理，计算所有特征
def preprocess_data_with_features(data_path):
    """
    数据预处理，计算分子指纹和蛋白质特征
    :param data_path: 数据集路径
    :return: 处理后的数据集
    """
    df = pd.read_csv(data_path)
    print(f"原始数据大小: {len(df)}")
    
    # 计算分子的ECFP指纹
    df['ecfp1'] = df['smiles1'].apply(calculate_ecfp_fingerprint)
    df['ecfp2'] = df['smiles2'].apply(calculate_ecfp_fingerprint)
    
    # 计算蛋白质特征
    df['protein_features'] = df['target'].apply(calculate_protein_features)
    
    # 删除特征计算失败的样本
    df = df[df['ecfp1'].notnull() & df['ecfp2'].notnull()]
    print(f"处理后数据大小: {len(df)}")
    
    return df


# 准备模型输入特征
def prepare_features(df):
    """
    准备模型输入特征和标签
    :param df: 处理后的数据集
    :return: 特征矩阵X和标签向量y
    """
    # 组合分子1指纹、分子2指纹和蛋白质特征
    X = np.array(df.apply(lambda row: np.concatenate([row['ecfp1'], row['ecfp2'], row['protein_features']]), axis=1).tolist())
    
    # 标签是delta值
    y = df['delta'].values
    
    return X, y
