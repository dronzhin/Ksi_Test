import torch
import pickle
import torch.nn as nn


class ConvModel(nn.Module):
    def __init__(self, input_dim=66, output_dim=39, target_length=17177, n = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_dim, 128*n, 3, stride=2, padding=1),
            nn.BatchNorm1d(128*n),
            nn.ReLU(),
            nn.Conv1d(128*n, 256*n, 3, stride=2, padding=1),
            nn.BatchNorm1d(256*n),
            nn.ReLU(),
            nn.Conv1d(256*n, 128*n, 3, stride=2, padding=1),
            nn.BatchNorm1d(128*n),
            nn.ReLU(),
            nn.Conv1d(128*n, output_dim, 1),
        )
        self.target_length = target_length

    def forward(self, x):
        x = x.permute(1, 0).unsqueeze(0)  # [1, channels, length]
        x = self.model(x)
        x = x[:, :, :self.target_length].squeeze(0)  # [channels, target_length]
        x = x.permute(1, 0)
        return x

class AttentionSelector(nn.Module):
    def __init__(self, input_dim=66, output_count=17177):
        """
        Модель с attention для выбора лучших `output_count` пар из входного тензора.
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(output_count, input_dim))  # [17177, 66]
        self.key = nn.Linear(input_dim, input_dim)  # Преобразование входа в ключи

    def forward(self, x):
        """
        x: [num_pairs, input_dim] — например, [335061, 66]
        return: [output_count, input_dim] — лучшие пары, отфильтрованные через attention
        """
        Q = self.query  # [output_count, input_dim]
        K = self.key(x)  # [num_pairs, input_dim]

        # Вычисляем attention-веса: [output_count, num_pairs]
        attn_weights = torch.matmul(Q, K.transpose(-2, -1))  # Q · K^T
        attn_weights = F.softmax(attn_weights, dim=1)  # Нормализуем по строкам

        # Применяем attention к исходным данным: [output_count, input_dim]
        selected = torch.matmul(attn_weights, x)  # Weighted sum of pairs
        return selected


if __name__ == '__main__':
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
        model = AttentionSelector()
        print(model(data).shape)


