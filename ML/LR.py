import sys
import os

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# 导入当前目录下的工具函数
from fingerprint import preprocess_data_with_fingerprints, prepare_features
from evaluation import calculate_metrics, print_metrics


def train_lr_model(X_train, y_train):
    """
    训练线性回归模型
    :param X_train: 训练特征
    :param y_train: 训练标签
    :return: 训练好的模型和特征缩放器
    """
    # 特征缩放 - 处理高维度特征的关键步骤
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 设置线性回归参数
    # 使用Ridge回归（L2正则化）解决高维特征过拟合问题
    alpha = 1.0  # 正则化强度
    fit_intercept = True
    
    # 打印模型参数设置
    print(f"线性回归参数设置:")
    print(f"  使用模型: Ridge回归 (L2正则化)")
    print(f"  正则化强度alpha: {alpha}")
    print(f"  是否拟合截距: {fit_intercept}")
    print(f"  特征缩放: 已应用StandardScaler")
    
    lr = Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    
    # 返回模型和缩放器用于测试阶段
    return lr, scaler


def main(data_path, test_size=0.2, random_state=42):
    """
    主函数
    :param data_path: 数据集路径
    :param test_size: 测试集比例
    :param random_state: 随机种子
    """
    print("正在加载和预处理数据...")
    df = preprocess_data_with_fingerprints(data_path)
    
    # 分割数据集
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # 训练和评估MACCS指纹模型
    print("\n训练基于MACCS指纹的线性回归模型...")
    X_train_maccs, y_train_maccs = prepare_features(train_df, 'maccs')
    X_test_maccs, y_test_maccs = prepare_features(test_df, 'maccs')
    
    print(f"MACCS特征维度: {X_train_maccs.shape[1]}")
    
    lr_maccs, scaler_maccs = train_lr_model(X_train_maccs, y_train_maccs)
    # 对测试集应用相同的特征缩放
    X_test_maccs_scaled = scaler_maccs.transform(X_test_maccs)
    y_pred_maccs = lr_maccs.predict(X_test_maccs_scaled)
    
    metrics_maccs = calculate_metrics(y_test_maccs, y_pred_maccs)
    print_metrics(metrics_maccs, "MACCS指纹")
    
    # 训练和评估ECFP指纹模型
    print("\n训练基于ECFP指纹的线性回归模型...")
    X_train_ecfp, y_train_ecfp = prepare_features(train_df, 'ecfp')
    X_test_ecfp, y_test_ecfp = prepare_features(test_df, 'ecfp')
    
    print(f"ECFP特征维度: {X_train_ecfp.shape[1]}")
    
    lr_ecfp, scaler_ecfp = train_lr_model(X_train_ecfp, y_train_ecfp)
    # 对测试集应用相同的特征缩放
    X_test_ecfp_scaled = scaler_ecfp.transform(X_test_ecfp)
    y_pred_ecfp = lr_ecfp.predict(X_test_ecfp_scaled)
    
    metrics_ecfp = calculate_metrics(y_test_ecfp, y_pred_ecfp)
    print_metrics(metrics_ecfp, "ECFP指纹")


if __name__ == "__main__":
    # 数据集路径需要根据实际情况修改
    data_path = "../data/melanocortin_receptor_4_mmps.csv"  # 假设数据集在data目录下
    print(f"数据集：{data_path}")
    main(data_path)