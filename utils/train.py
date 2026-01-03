import os
import torch.nn as nn
from utils.util import *
from sklearn.metrics import mean_squared_error
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(args, model, train_loader, test_loader):
 
    model_save_path = './result/' + args['MODEL'] + '_' + args['TARGET_NAME'] + '.tar'
    
    if not os.path.exists('./result'):
        os.makedirs('./result')

    # regression loss
    criterion = nn.MSELoss()
    
    # 为不同模型设置不同的学习率策略
    if args['MODEL'].startswith('attention') or 'dta' in args['MODEL']:
        # 为attention模型使用较小的初始学习率和学习率调度
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=args["WEIGHT_DECAY"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    else:
        # 其他模型保持原学习率设置
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args["WEIGHT_DECAY"])
        scheduler = None
 
    num_epochs = 1000
    best_test_mse = float('inf')
    early_stopping_count = 0
    y_pred_test_best = None
    
    # history for plotting
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_mse': []
    }

    for epoch in range(num_epochs):
        model.train()
        model, loss, total_loss, train_outputs, train_y = predict(args, model, train_loader, criterion, optimizer, True)
        model.eval()
        with torch.no_grad():
            model, val_loss, test_total_loss, test_outputs, test_y = predict(args, model, test_loader, criterion, optimizer, False)
            # compute validation MSE
            test_mse = mean_squared_error(test_y, test_outputs)
            avg_train_loss = total_loss / max(1, len(train_loader))
            avg_test_loss = test_total_loss / max(1, len(test_loader))
            
            print('Epoch: {}/{} '.format(epoch + 1, num_epochs),
                  ' Training Loss: {:.6f}'.format(avg_train_loss),
                  ' Test Loss: {:.6f}'.format(avg_test_loss),
                  ' Test MSE: {:.6f}'.format(test_mse))
        
        # record history
        history['train_loss'].append(float(avg_train_loss))
        history['test_loss'].append(float(avg_test_loss))
        history['test_mse'].append(float(test_mse))
        
        model.train()
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step(test_mse)
            
        # save model when validation MSE improves (smaller)
        if test_mse < best_test_mse:
            print("Test MSE improved from {:.6f} -> {:.6f}".format(best_test_mse, test_mse))
            best_test_mse = test_mse
            torch.save(model.state_dict(), model_save_path)
            early_stopping_count = 0
            y_pred_test_best = test_outputs[:]  # save best test predictions
        else:
            early_stopping_count += 1
            print("Test MSE did not improve.. Counter {}/{}".format(early_stopping_count, args['EARLY_STOPPING_PATIENCE']))
            if early_stopping_count > args['EARLY_STOPPING_PATIENCE']:
                print("Early Stopped ..")
                break
    
    
    # plot losses
    try:
        epochs = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3)
#         plt.plot(epochs, history['test_loss'], label='Test Loss', marker='o', markersize=3)
#         plt.plot(epochs, history['test_mse'], label='Test MSE', marker='o', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss / MSE')
        plt.title(f'{args["MODEL"]}_{args["TARGET_NAME"]} Training Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join('./result', f'{args["MODEL"]}_{args["TARGET_NAME"]}_loss.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved loss curve to {plot_path}")
    except Exception as e:
        print("Warning: failed to plot/save loss curve:", e)

    # return best test predictions (list of floats)
    return y_pred_test_best