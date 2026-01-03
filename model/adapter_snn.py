import torch
import torch.nn as nn
import numpy as np


class Adapter_SNN(nn.Module):
    """
    KPGT 特征适配器模型，用于分子对差值预测任务 (SNN架构)
    该模型使用 MLP 作为 adapter 学习分子的 KPGT 特征，并使用差值作为输入
    """

    def __init__(self, args):
        super(Adapter_SNN, self).__init__()

        # KPGT 特征维度
        kpgt_dim = 2304
        
        # Adapter 配置
        adapter_hidden_dim = 512
        target_dim = 256
        
        # 差值预测网络配置
        hidden_dim = 256 * 2  # SNN架构调整：隐藏层大小减半
        
        # MLP Adapter 用于将 KPGT 特征转换为目标维度
        self.adapter = nn.Sequential(
            nn.Linear(kpgt_dim, adapter_hidden_dim),
            nn.LayerNorm(adapter_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args['DROP_OUT']),
            nn.Linear(adapter_hidden_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(inplace=True),
        )

        # 差值预测网络 (SNN架构调整：输入维度从 target_dim*2 改为 target_dim)
        self.fc_layer = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args['DROP_OUT']),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(args['DROP_OUT']),
        )

        self.out_layer = nn.Linear(hidden_dim // 2, 1)
    
    def load_kpgt_features(self, feature_dir):
        """
        加载 KPGT 特征文件
        Args:
            feature_dir: KPGT 特征文件路径
        Returns:
            kpgt_features: 包含所有分子 KPGT 特征的字典
        """
        self.kpgt_features = np.load(feature_dir)
    
    def get_molecule_feature(self, smiles):
        """
        获取单个分子的 KPGT 特征
        Args:
            smiles: 分子的 SMILES 表示
        Returns:
            feature: 分子的 KPGT 特征张量
        """
        if hasattr(self, 'kpgt_features'):
            # 从加载的特征字典中获取特征
            feature = self.kpgt_features[smiles]
            # 转换为 PyTorch 张量并移动到模型所在设备
            return torch.tensor(feature, dtype=torch.float32).to(next(self.parameters()).device)
        else:
            raise ValueError("KPGT features not loaded. Please call load_kpgt_features first.")
    
    def get_batch_features(self, batch_smiles):
        """
        获取批量分子的 KPGT 特征
        Args:
            batch_smiles: 批量分子的 SMILES 列表
        Returns:
            batch_features: 批量分子的 KPGT 特征张量 (batch_size, kpgt_dim)
        """
        features = []
        for smiles in batch_smiles:
            feature = self.get_molecule_feature(smiles)
            features.append(feature)
        return torch.stack(features)

    def forward(self, batch_smiles1, batch_smiles2):
        """
        前向传播函数
        Args:
            batch_smiles1: 第一个分子的批量 SMILES 列表
            batch_smiles2: 第二个分子的批量 SMILES 列表
        Returns:
            delta: 分子对的性质差值预测结果
        """
        # 获取批量分子的 KPGT 特征
        features1 = self.get_batch_features(batch_smiles1)
        features2 = self.get_batch_features(batch_smiles2)
        
        # 通过 MLP Adapter 处理特征
        adapted_features1 = self.adapter(features1)
        adapted_features2 = self.adapter(features2)
        
        # SNN架构：使用差值代替拼接
        diff = adapted_features1 - adapted_features2
        
        # 通过差值预测网络
        out = self.fc_layer(diff)
        out = self.out_layer(out)
        
        # 回归任务：不应用 sigmoid
        out = out.squeeze(-1)
        
        return out