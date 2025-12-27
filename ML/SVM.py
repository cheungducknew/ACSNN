import sys
import os

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 导入当前目录下的工具函数
from fingerprint import preprocess_data_with_fingerprints, prepare_features
from evaluation import calculate_metrics, print_metrics


def train_svm_model(X_train, y_train, random_state=42):
    """
    训练支持向量机模型
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param random_state: 随机种子
    :return: 训练好的模型
    """
    # 设置SVM超参数
    kernel = 'poly'
    C = 1.0
    epsilon = 0.1
    gamma = 'scale'
    
    # 打印模型参数设置
    print(f"支持向量机参数设置:")
    print(f"  核函数: {kernel}")
    print(f"  正则化参数C: {C}")
    print(f"  不敏感损失参数epsilon: {epsilon}")
    print(f"  核函数参数gamma: {gamma}")
    
    svm = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma,
#         random_state=random_state
    )
    svm.fit(X_train, y_train)
    return svm


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
    print("\n训练基于MACCS指纹的支持向量机模型...")
    X_train_maccs, y_train_maccs = prepare_features(train_df, 'maccs')
    X_test_maccs, y_test_maccs = prepare_features(test_df, 'maccs')
    
    print(f"MACCS特征维度: {X_train_maccs.shape[1]}")
    
    svm_maccs = train_svm_model(X_train_maccs, y_train_maccs, random_state=random_state)
    y_pred_maccs = svm_maccs.predict(X_test_maccs)
    
    metrics_maccs = calculate_metrics(y_test_maccs, y_pred_maccs)
    print_metrics(metrics_maccs, "MACCS指纹")
    
    # 训练和评估ECFP指纹模型
    print("\n训练基于ECFP指纹的支持向量机模型...")
    X_train_ecfp, y_train_ecfp = prepare_features(train_df, 'ecfp')
    X_test_ecfp, y_test_ecfp = prepare_features(test_df, 'ecfp')
    
    print(f"ECFP特征维度: {X_train_ecfp.shape[1]}")
    
    svm_ecfp = train_svm_model(X_train_ecfp, y_train_ecfp, random_state=random_state)
    y_pred_ecfp = svm_ecfp.predict(X_test_ecfp)
    
    metrics_ecfp = calculate_metrics(y_test_ecfp, y_pred_ecfp)
    print_metrics(metrics_ecfp, "ECFP指纹")


if __name__ == "__main__":
    # 数据集路径需要根据实际情况修改
    data_path = "../data/melanocortin_receptor_4_mmps.csv"  # 假设数据集在data目录下
    print(f"数据集：{data_path}")
    main(data_path)