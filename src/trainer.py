import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

def train_model(model, X_train, y_train, device, epochs=50, batch_size=32, lr=1e-3, save_path=None):
    """
    增加 save_path 参数，训练结束后自动保存模型
    """
    model.to(device)
    model.train()
    
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.LongTensor(y_train)

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    dataset = TensorDataset(X_train, y_train)
    # drop_last=True 防止 Batch Norm 报错
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"   -> Training Config: Epochs={epochs}, Batch={batch_size}, LR={lr}")
    
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(bx).squeeze()
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            
    # [新增] 保存模型
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"💾 模型已保存至: {save_path}")
            
    return model