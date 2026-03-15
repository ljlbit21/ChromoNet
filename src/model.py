import torch
import torch.nn as nn

class ChromoNet(nn.Module):
    def __init__(self, input_len=None):
        super(ChromoNet, self).__init__()
        # CNN 特征提取器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(128)
        )
        
        self.flatten = nn.Flatten()
        
        # 深度特征层 (倒数第二层)
        self.feature_extractor = nn.Sequential(
            nn.Linear(64 * 128, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(0.5)
        )
        
        # 最终分类器
        self.output_layer = nn.Linear(256, 1)

    def forward(self, x):
        # 正常前向传播
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        features = self.feature_extractor(x)
        return self.output_layer(features).squeeze(1)

    def extract_features(self, x):
        """
        [新增] 专门用于提取深度特征 (用于 Silhouette Score 计算)
        """
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        return self.feature_extractor(x)
