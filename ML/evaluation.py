import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


def calculate_metrics(y_true, y_pred):
    """
    计算回归模型的评估指标
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 评估指标字典
    """
    # 计算MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # 计算RMSE
    rmse = np.sqrt(mse)
    
    # 计算R²
    r2 = r2_score(y_true, y_pred)
    
    # 计算Pearson相关系数
    pcc, _ = pearsonr(y_true, y_pred)
    
    # 计算Spearman相关系数
    spearman, _ = spearmanr(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pcc': pcc,
        'spearman': spearman
    }


def print_metrics(metrics, model_name):
    """
    打印评估指标
    :param metrics: 评估指标字典
    :param model_name: 模型名称
    """
    print(f"{model_name} 模型评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")