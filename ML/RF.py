import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from fingerprint import preprocess_data_with_fingerprints, prepare_features
from evaluation import calculate_metrics, print_metrics


def train_rf_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    训练随机森林模型
    :param X_train: 训练特征
    :param y_train: 训练标签
    :param n_estimators: 树的数量
    :param random_state: 随机种子
    :return: 训练好的模型
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


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
    print("\n训练基于MACCS指纹的随机森林模型...")
    X_train_maccs, y_train_maccs = prepare_features(train_df, 'maccs')
    X_test_maccs, y_test_maccs = prepare_features(test_df, 'maccs')
    
    rf_maccs = train_rf_model(X_train_maccs, y_train_maccs, random_state=random_state)
    y_pred_maccs = rf_maccs.predict(X_test_maccs)
    
    metrics_maccs = calculate_metrics(y_test_maccs, y_pred_maccs)
    print_metrics(metrics_maccs, "MACCS指纹")
    
    # 训练和评估ECFP指纹模型
    print("\n训练基于ECFP指纹的随机森林模型...")
    X_train_ecfp, y_train_ecfp = prepare_features(train_df, 'ecfp')
    X_test_ecfp, y_test_ecfp = prepare_features(test_df, 'ecfp')
    
    rf_ecfp = train_rf_model(X_train_ecfp, y_train_ecfp, random_state=random_state)
    y_pred_ecfp = rf_ecfp.predict(X_test_ecfp)
    
    metrics_ecfp = calculate_metrics(y_test_ecfp, y_pred_ecfp)
    print_metrics(metrics_ecfp, "ECFP指纹")


if __name__ == "__main__":
    # 数据集路径需要根据实际情况修改
    data_path = "../data/thrombin_mmps.csv"  # 假设数据集在data目录下
    print(f"数据集：{data_path}")
    main(data_path)