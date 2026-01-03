import torch
import torch.nn as nn

class Conv1d(nn.Module):
    """
    Three 1d convolutional layer with relu activation stacked on top of each other
    with a final global maxpooling layer
    """
    def __init__(self, vocab_size, channel, kernel_size, stride=1, padding=0):
        super(Conv1d, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=128)
        self.conv1 = nn.Conv1d(128, channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(channel, channel*2, kernel_size, stride, padding)
        self.conv3 = nn.Conv1d(channel*2, channel*3, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        return x

class DeepDTA(nn.Module):
    """DeepDTA model architecture, Y-shaped net that does 1d convolution on 
    both the ligand and the protein representation and then concatenates the
    result into a final predictor of binding affinity"""

    def __init__(self, pro_vocab_size, lig_vocab_size, channel, protein_kernel_size, ligand_kernel_size):
        super(DeepDTA, self).__init__()
        self.ligand_conv = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.protein_conv = Conv1d(pro_vocab_size, channel, protein_kernel_size)
        
        # 使用nn.Sequential实现全连接层
        self.mlp = nn.Sequential(
            nn.Linear(channel*6, 1024), # 32*6
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        # print(f'channel:{channel}') # 32

    def forward(self, protein, ligand):
        x1 = self.ligand_conv(ligand)   # 16,96
        x2 = self.protein_conv(protein)  # 16,96
        x = torch.cat((x1, x2), dim=1)  # 16,192
        x = self.mlp(x)
        return x.squeeze()


class DeepDTA_MMP(nn.Module):
    """DeepDTA_MMP model architecture, extended from DeepDTA to handle two ligands (smiles1, smiles2) 
    and one protein, predicting the delta value between the two ligands' binding affinities to the protein"""

    def __init__(self, pro_vocab_size, lig_vocab_size, channel, protein_kernel_size, ligand_kernel_size):
        super(DeepDTA_MMP, self).__init__()
        # 两个配体卷积层（共享词汇表和参数）
        self.ligand_conv1 = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.ligand_conv2 = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.protein_conv = Conv1d(pro_vocab_size, channel, protein_kernel_size)
        
        # 使用nn.Sequential实现全连接层
        self.mlp = nn.Sequential(
            nn.Linear(channel*9, 1024),  # channel*3 * 3个卷积输出
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, protein, ligand1, ligand2):
        # 处理两个配体和一个蛋白质
        x1 = self.ligand_conv1(ligand1)   # 处理smiles1
        x2 = self.ligand_conv2(ligand2)   # 处理smiles2
        x3 = self.protein_conv(protein)   # 处理target
        
        # 拼接三个卷积输出
        x = torch.cat((x1, x2, x3), dim=1)
        
        # 通过全连接层进行预测
        x = self.mlp(x)
        
        # 返回delta预测值
        return x.squeeze()


class DeepDTA_SNN(nn.Module):
    """DeepDTA_SNN model architecture, based on Siamese Neural Network, which first calculates
    the feature difference between two ligands, then concatenates it with protein feature, 
    and finally predicts the delta value through MLP"""

    def __init__(self, pro_vocab_size, lig_vocab_size, channel, protein_kernel_size, ligand_kernel_size):
        super(DeepDTA_SNN, self).__init__()
        # 两个配体卷积层（共享词汇表和参数）
        self.ligand_conv1 = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.ligand_conv2 = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.protein_conv = Conv1d(pro_vocab_size, channel, protein_kernel_size)
        
        # 使用nn.Sequential实现全连接层
        # 输入维度为：(x1-x2)的维度(channel*3) + x3的维度(channel*3) = channel*6
        self.mlp = nn.Sequential(
            nn.Linear(channel*6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, protein, ligand1, ligand2):
        # 处理两个配体和一个蛋白质
        x1 = self.ligand_conv1(ligand1)   # 处理smiles1
        x2 = self.ligand_conv2(ligand2)   # 处理smiles2
        x3 = self.protein_conv(protein)   # 处理target
        
        # 计算x1和x2的特征差值
        x_diff = x1 - x2
        
        # 将差值特征与蛋白质特征拼接
        x = torch.cat((x_diff, x3), dim=1)
        
        # 通过全连接层进行预测
        x = self.mlp(x)
        
        # 返回delta预测值
        return x.squeeze()
