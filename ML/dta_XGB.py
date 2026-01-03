import sys
import os

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# 导入当前目录下的工具函数
from dta_features import preprocess_data_with_features, prepare_features
from evaluation import calculate_metrics, print_metrics


def train_xgb_model(X_train, y_train):
    """
    训练XGBoost回归模型
    :param X_train: 训练特征
    :param y_train: 训练标签
    :return: 训练好的模型和特征缩放器
    """
    # 特征缩放 - 处理高维度特征的关键步骤
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 设置XGBoost回归参数
    n_estimators = 500
    max_depth = 50
    learning_rate = 0.1
    subsample = 0.8
    colsample_bytree = 0.8
    min_child_weight = 1
    gamma = 0
    objective = 'reg:squarederror'  # 明确设置为回归任务
    eval_metric = 'rmse'  # 回归评估指标
    
    # 打印模型参数设置
    print(f"XGBoost回归参数设置:")
    print(f"  迭代次数: {n_estimators}")
    print(f"  最大树深度: {max_depth}")
    print(f"  学习率: {learning_rate}")
    print(f"  样本采样率: {subsample}")
    print(f"  特征采样率: {colsample_bytree}")
    print(f"  子节点最小权重: {min_child_weight}")
    print(f"  分裂所需最小增益: {gamma}")
    print(f"  目标函数: {objective}")
    print(f"  评估指标: {eval_metric}")
    
    # 使用XGBRegressor进行回归任务
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        objective=objective,
        eval_metric=eval_metric,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # 返回模型和缩放器用于测试阶段
    return xgb_model, scaler


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
    print("\n训练XGBoost模型...")
    xgb_model, scaler = train_xgb_model(X_train, y_train)
    
    # 对测试集应用相同的特征缩放
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    y_pred = xgb_model.predict(X_test_scaled)
    
    # 评估模型
    print("\n模型评估结果:")
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "DTA XGB")


if __name__ == "__main__":
    main()
