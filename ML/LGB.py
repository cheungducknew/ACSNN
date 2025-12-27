import sys
import os

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 导入当前目录下的工具函数
from fingerprint import preprocess_data_with_fingerprints, prepare_features
from evaluation import calculate_metrics, print_metrics


def train_lgb_model(X_train, y_train, random_state=42):
    """
    训练LightGBM模型
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param random_state: 随机种子
    :return: 训练好的模型
    """
    # 设置LightGBM超参数
    n_estimators = 500
    max_depth = 50
    num_leaves = 255  # 通常设置为2^max_depth - 1
    learning_rate = 0.1
    subsample = 0.8
    colsample_bytree = 0.8
    min_child_samples = 2
    objective = 'regression'  # 明确设置为回归任务
    metric = 'rmse'  # 回归评估指标
    
    # 打印模型参数设置
    print(f"LightGBM参数设置:")
    print(f"  迭代次数: {n_estimators}")
    print(f"  最大树深度: {max_depth}")
    print(f"  叶子节点数: {num_leaves}")
    print(f"  学习率: {learning_rate}")
    print(f"  样本采样率: {subsample}")
    print(f"  特征采样率: {colsample_bytree}")
    print(f"  叶节点最小样本数: {min_child_samples}")
    print(f"  目标函数: {objective}")
    print(f"  评估指标: {metric}")
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        objective=objective,
        metric=metric,
        random_state=random_state
    )
    lgb_model.fit(X_train, y_train)
    return lgb_model


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
    print("\n训练基于MACCS指纹的LightGBM模型...")
    X_train_maccs, y_train_maccs = prepare_features(train_df, 'maccs')
    X_test_maccs, y_test_maccs = prepare_features(test_df, 'maccs')
    
    print(f"MACCS特征维度: {X_train_maccs.shape[1]}")
    
    lgb_maccs = train_lgb_model(X_train_maccs, y_train_maccs, random_state=random_state)
    y_pred_maccs = lgb_maccs.predict(X_test_maccs)
    
    metrics_maccs = calculate_metrics(y_test_maccs, y_pred_maccs)
    print_metrics(metrics_maccs, "MACCS指纹")
    
    # 训练和评估ECFP指纹模型
    print("\n训练基于ECFP指纹的LightGBM模型...")
    X_train_ecfp, y_train_ecfp = prepare_features(train_df, 'ecfp')
    X_test_ecfp, y_test_ecfp = prepare_features(test_df, 'ecfp')
    
    print(f"ECFP特征维度: {X_train_ecfp.shape[1]}")
    
    lgb_ecfp = train_lgb_model(X_train_ecfp, y_train_ecfp, random_state=random_state)
    y_pred_ecfp = lgb_ecfp.predict(X_test_ecfp)
    
    metrics_ecfp = calculate_metrics(y_test_ecfp, y_pred_ecfp)
    print_metrics(metrics_ecfp, "ECFP指纹")


if __name__ == "__main__":
    # 数据集路径需要根据实际情况修改
    data_path = "../data/melanocortin_receptor_4_mmps.csv"  # 假设数据集在data目录下
    print(f"数据集：{data_path}")
    main(data_path)