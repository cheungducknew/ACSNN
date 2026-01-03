import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, output_attention=False):
        batch_size, seq_len, hidden_size = hidden_states.size()

        # Linear projections
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Reshape to [batch_size, num_heads, seq_len, head_size]
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_scores.size(-2) != attention_mask.size(-2):
                attention_mask_pad = torch.ones((attention_scores.size(0), 2), device=attention_scores.device).unsqueeze(1).unsqueeze(2)
                attention_mask = torch.cat([attention_mask_pad, attention_mask], dim=-1)
            else:
                attention_scores = attention_scores + attention_mask

        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Calculate context vector
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, hidden_size)

        if output_attention:
            return context, attention_probs
        else:
            return context, None


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.Wa = nn.Linear(d_model, hidden_dim)
        self.Wb = nn.Linear(d_model, hidden_dim)

    def forward(self, x):
        return self.Wa(x) * F.silu(self.Wb(x))


class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model, mlp_ratio=3.0, hidden_dropout_prob=0.1):
        super().__init__()
        hidden_dim = int(mlp_ratio * d_model)

        self.fc = SwiGLU(d_model, hidden_dim)
        self.proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        return self.dropout(self.proj(self.fc(x)))


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super(FeedForward, self).__init__()
        # 计算mlp_ratio
        mlp_ratio = intermediate_size / hidden_size
        # 使用SwiGLU作为前馈层
        self.swiglu = FFN_SwiGLU(hidden_size, mlp_ratio, hidden_dropout_prob)

    def forward(self, x):
        return self.swiglu(x)


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SublayerConnection, self).__init__()
        # 使用 PyTorch 内置的 LayerNorm，设置 elementwise_affine=True 以增加模型表达能力
        self.LayerNorm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 改进残差连接顺序：先残差连接，再LayerNorm
        return self.LayerNorm(hidden_states + input_tensor)


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob)
        self.sublayer = SublayerConnection(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.feed_forward(input_tensor)
        hidden_states = self.sublayer(hidden_states, input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.sublayer = SublayerConnection(hidden_size, hidden_dropout_prob)
        self.output = SelfOutput(hidden_size, intermediate_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None, output_attention=False):
        attention_output, attention_probs_0 = self.attention(input_tensor, attention_mask, output_attention)
        attention_output = self.sublayer(attention_output, input_tensor)
        layer_output = self.output(attention_output)

        return layer_output, attention_probs_0


class Attention_MMP(nn.Module):
    """
    KPGT 特征适配器模型，用于分子对差值预测任务
    该模型使用局部注意力作为 adapter 学习分子的 KPGT 特征
    将2304维特征拆分为3个768维子特征，分别应用注意力机制
    """

    def __init__(self, args):
        super(Attention_MMP, self).__init__()

        # KPGT 特征结构：3个768维子特征组成2304维总特征
        self.kpgt_subdim = 768  # 每个子特征的维度
        self.num_subfeatures = 3  # 子特征数量
        self.kpgt_dim = self.kpgt_subdim * self.num_subfeatures  # 总维度
        
        # 模型配置参数 - 局部注意力设置
        # 确保hidden_size能被3整除，因为有3个子特征
        self.hidden_size = 252  # 252 = 84 * 3，能被3整除
        self.intermediate_size = self.hidden_size * 4
        self.num_attention_heads = 6  # 6 = 2 * 3，确保能被3整除
        self.attention_probs_dropout_prob = 0.2  # 注意力Dropout率
        self.hidden_dropout_prob = args['DROP_OUT']
        self.SA_N = 3  # 三层 self-attention encoder

        # 为每个子特征创建独立的投影层和注意力模块
        self.subfeature_projections = nn.ModuleList([
            nn.Linear(self.kpgt_subdim, self.hidden_size // self.num_subfeatures)
            for _ in range(self.num_subfeatures)
        ])
        
        # 为每个子特征创建独立的layer norm
        self.subfeature_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_size // self.num_subfeatures, elementwise_affine=True, eps=1e-6),
                nn.Dropout(self.hidden_dropout_prob)
            )
            for _ in range(self.num_subfeatures)
        ])

        # 为每个子特征创建独立的注意力encoder
        self.subfeature_encoders = nn.ModuleList([
            nn.ModuleList([
                Encoder(self.hidden_size // self.num_subfeatures, 
                        self.intermediate_size // self.num_subfeatures, 
                        self.num_attention_heads // self.num_subfeatures,  # 每个子特征分配较少的注意力头
                        self.attention_probs_dropout_prob, 
                        self.hidden_dropout_prob)
                for _ in range(self.SA_N)
            ])
            for _ in range(self.num_subfeatures)
        ])
        
        # 增加额外的正则化层
        self.extra_layer_norm = nn.LayerNorm(self.hidden_size)

        # 差值预测网络（与原 adapter_mmp.py 保持一致）
        target_dim = self.hidden_size
        hidden_dim = target_dim * 4
        
        self.fc_layer = nn.Sequential(
            nn.Linear(target_dim * 2, hidden_dim),
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
        
        # 局部注意力处理函数
        def process_local_attention(features):
            # 将2304维特征拆分为3个768维子特征
            subfeatures = torch.split(features, self.kpgt_subdim, dim=1)
            processed_subfeatures = []
            
            # 对每个子特征分别应用局部注意力
            for i in range(self.num_subfeatures):
                subfeature = subfeatures[i]
                
                # 子特征投影
                projected = self.subfeature_projections[i](subfeature)
                projected = self.subfeature_layers[i](projected)
                
                # 添加序列维度以便进行注意力计算
                projected = projected.unsqueeze(1)  # (batch_size, 1, hidden_size//3)
                
                # 通过注意力encoder
                attention_mask = None
                for encoder in self.subfeature_encoders[i]:
                    projected, _ = encoder(projected, attention_mask)
                
                # 移除序列维度
                processed_subfeature = projected.squeeze(1)
                processed_subfeatures.append(processed_subfeature)
            
            # 合并处理后的子特征
            combined = torch.cat(processed_subfeatures, dim=1)
            # 添加额外的正则化
            combined = self.extra_layer_norm(combined)
            
            return combined
        
        # 处理第一个分子特征
        adapted_features1 = process_local_attention(features1)
        
        # 处理第二个分子特征
        adapted_features2 = process_local_attention(features2)
        
        # 拼接两个分子的特征
        combined_features = torch.cat((adapted_features1, adapted_features2), dim=1)
        
        # 通过差值预测网络
        out = self.fc_layer(combined_features)
        out = self.out_layer(out)
        
        # 回归任务：不应用 sigmoid
        out = out.squeeze(-1)
        
        return out