import pandas as pd
import random
import numpy as np
import torch


def build_vocabularies(data, max_smiles_len=200, max_fasta_len=2000):
    """
    构建SMILES和蛋白质序列的词汇表
    Args:
        data: 包含smiles1, smiles2和target的DataFrame
        max_smiles_len: SMILES序列的最大长度
        max_fasta_len: 蛋白质序列的最大长度
    Returns:
        smiles_dict: SMILES字符到索引的映射
        fasta_dict: 蛋白质字符到索引的映射
        smiles_vocab_size: SMILES词汇表大小
        fasta_vocab_size: 蛋白质词汇表大小
    """
    # 从数据中收集SMILES和蛋白质序列的字符
    smiles_chars = set(['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Te', 'H',
                       '(', ')', '[', ']', '=', '#', '+', '-', '\\', '/', '@', '.', ':', '1', '2', '3',
                       '4', '5', '6', '7', '8', '9', '0', 'c', 'n', 'o', 's', 'p', 'se', 'te', 'b'])
    protein_chars = set(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 
                        'P', 'S', 'T', 'W', 'Y', 'V', 'X', 'U', 'O'])
    
    # 从实际数据中收集字符
    for smiles in pd.concat([data.smiles1, data.smiles2]):
        for char in smiles[:max_smiles_len]:
            smiles_chars.add(char)
    for fasta in data.target:
        for char in fasta[:max_fasta_len]:
            protein_chars.add(char)
    
    # 添加特殊标记并创建有序词汇表
    smiles_vocab = ['<PAD>', '<UNK>'] + sorted(list(smiles_chars))
    fasta_vocab = ['<PAD>', '<UNK>'] + sorted(list(protein_chars))
    
    # 创建字符到索引的映射
    smiles_dict = {char: idx for idx, char in enumerate(smiles_vocab)}
    fasta_dict = {char: idx for idx, char in enumerate(fasta_vocab)}
    
    print(f"SMILES词汇表大小: {len(smiles_vocab)}")
    print(f"蛋白质词汇表大小: {len(fasta_vocab)}")
    
    return smiles_dict, fasta_dict, len(smiles_vocab), len(fasta_vocab)


def DeepDTA_Dataset(args, data, is_train, drop_last=False, smiles_dict=None, fasta_dict=None):
    
    device = args['DEVICE']
    
    # 设置默认参数
    max_smiles_len = 200
    max_fasta_len = 2000
    
    # 如果没有提供词汇表，则构建词汇表
    if smiles_dict is None or fasta_dict is None:
        smiles_dict, fasta_dict, _, _ = build_vocabularies(data, max_smiles_len, max_fasta_len)
    
    # 处理SMILES1、SMILES2和target序列
    smiles1_indices = []
    smiles2_indices = []
    target_indices = []
    
    for smiles1, smiles2, target in zip(data.smiles1, data.smiles2, data.target):
        # 处理SMILES1
        s1_idx = []
        for char in smiles1[:max_smiles_len]:
            s1_idx.append(smiles_dict.get(char, smiles_dict['<UNK>']))
        while len(s1_idx) < max_smiles_len:
            s1_idx.append(smiles_dict['<PAD>'])
        smiles1_indices.append(s1_idx)
        
        # 处理SMILES2
        s2_idx = []
        for char in smiles2[:max_smiles_len]:
            s2_idx.append(smiles_dict.get(char, smiles_dict['<UNK>']))
        while len(s2_idx) < max_smiles_len:
            s2_idx.append(smiles_dict['<PAD>'])
        smiles2_indices.append(s2_idx)
        
        # 处理target序列
        t_idx = []
        for char in target[:max_fasta_len]:
            t_idx.append(fasta_dict.get(char, fasta_dict['<UNK>']))
        while len(t_idx) < max_fasta_len:
            t_idx.append(fasta_dict['<PAD>'])
        target_indices.append(t_idx)
    
    # 转换为张量
    smiles1_tensors = [torch.tensor(idx, device=device) for idx in smiles1_indices]
    smiles2_tensors = [torch.tensor(idx, device=device) for idx in smiles2_indices]
    target_tensors = [torch.tensor(idx, device=device) for idx in target_indices]
    
    # 获取标签
    y = list(data.delta.astype(float))
    
    # 打乱数据（如果是训练集）
    if is_train:
        rd_idx = random.sample(range(len(data)), len(data))
        smiles1_tensors = [smiles1_tensors[i] for i in rd_idx]
        smiles2_tensors = [smiles2_tensors[i] for i in rd_idx]
        target_tensors = [target_tensors[i] for i in rd_idx]
        y = [y[i] for i in rd_idx]
    
    # 创建数据集
    data_set = []
    for i in range(len(data)):
        data_set.append({
            'smiles1': smiles1_tensors[i],
            'smiles2': smiles2_tensors[i],
            'target': target_tensors[i]
        })
    
    # 分割批次
    batch_size = args['BATCH_SIZE']
    remainder = len(data_set) % batch_size
    splited_data_set = np.array_split(data_set[remainder:], int(len(data_set)/batch_size))
    splited_data_set = [list(x) for x in splited_data_set]
    if remainder != 0:
        if not drop_last:
            splited_data_set.append(data_set[:remainder])
    
    splited_y = np.array_split(y[remainder:], int(len(data_set)/batch_size))
    splited_y = [list(y) for y in splited_y]
    if remainder != 0:
        if not drop_last:
            splited_y.append(y[:remainder])
    
    # 转换标签为张量
    splited_y = [torch.from_numpy(np.array(label)).float().to(device) for label in splited_y]
    
    # 组合数据加载器
    loader = []
    for i in range(len(splited_data_set)):
        batch = splited_data_set[i]
        batch_smiles1 = torch.stack([item['smiles1'] for item in batch], dim=0)
        batch_smiles2 = torch.stack([item['smiles2'] for item in batch], dim=0)
        batch_target = torch.stack([item['target'] for item in batch], dim=0)
        loader.append((i, (batch_smiles1, batch_smiles2, batch_target), splited_y[i]))
    
    return loader