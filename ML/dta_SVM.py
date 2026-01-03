import sys
import os

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# 导入当前目录下的工具函数
from dta_features import preprocess_data_with_features, prepare_features
from evaluation import calculate_metrics, print_metrics


def train_svm_model(X_train, y_train, random_state=42):
    """
    训练支持向量机模型
    :param X_train: 训练特征
    :param y_train: 训练标签
    :return: 训练好的模型和特征缩放器
    """
    # 特征缩放 - 处理高维度特征的关键步骤
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 设置SVM超参数
    kernel = 'rbf'  # 尝试使用径向基函数核，通常在回归任务中表现更好
    C = 10.0  # 增加C值以降低正则化强度
    epsilon = 0.01  # 减小epsilon以提高模型对小误差的敏感性
    gamma = 0.001  # 使用固定的gamma值而非'scale'，以获得更好的性能
    
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
        gamma=gamma
    )
    svm.fit(X_train_scaled, y_train)
    
    # 返回模型和缩放器用于测试阶段
    return svm, scaler


# 主函数
def main():
    """
    主函数
    """
    # 数据集路径
    data_path = "../data/bindingdb.csv"
    test_size = 0.2
    random_state = 42
    
    print("正在加载和预处理数据...")
    df = preprocess_data_with_features(data_path)
    
    # 分割数据集
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # 准备特征和标签
    print("\n准备模型输入特征...")
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)
    
    print(f"特征维度: {X_train.shape[1]}")
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 训练模型
    print("\n训练支持向量机模型...")
    svm_model, scaler = train_svm_model(X_train, y_train)
    
    # 对测试集应用相同的特征缩放
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    y_pred = svm_model.predict(X_test_scaled)
    
    # 评估模型
    print("\n模型评估结果:")
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "DTA SVM")


if __name__ == "__main__":
    main()
