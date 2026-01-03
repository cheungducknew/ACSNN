import sys
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# 导入当前目录下的工具函数
from dta_features import preprocess_data_with_features, prepare_features
from evaluation import calculate_metrics, print_metrics


def train_rf_model(X_train, y_train):
    """
    训练随机森林模型
    :param X_train: 训练特征
    :param y_train: 训练标签
    :return: 训练好的模型和特征缩放器
    """
    # 特征缩放 - 处理高维度特征的关键步骤
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 设置随机森林超参数
    n_estimators = 200
    # max_features = "log2"
    min_samples_leaf = 2
    criterion = "squared_error"
    max_depth = 50
    
    # 打印模型参数设置
    print(f"随机森林参数设置:")
    print(f"  决策树个数: {n_estimators}")
    # print(f"  特征选择方式: {max_features}")
    print(f"  叶节点最小样本数: {min_samples_leaf}")
    print(f"  分裂标准: {criterion}")
    print(f"  最大树深度: {max_depth}")
    
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        # max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        max_depth=max_depth,
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    
    # 返回模型和缩放器用于测试阶段
    return rf, scaler


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
    print("\n训练随机森林模型...")
    rf_model, scaler = train_rf_model(X_train, y_train)
    
    # 对测试集应用相同的特征缩放
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    y_pred = rf_model.predict(X_test_scaled)
    
    # 评估模型
    print("\n模型评估结果:")
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "DTA RF")


if __name__ == "__main__":
    main()
