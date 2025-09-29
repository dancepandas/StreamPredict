import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple


class StreamModel(nn.Module):
    def __init__(self,sequence_length: int = 24,
                 num_stream_features: int = 3,
                 num_rainfall_features: int = 23,
                 num_evap_features: int = 1,
                 num_non_target_features: int = 26,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 embed_dim:int = 128,
                 m:int = 16):
        '''

        :param sequence_length: 历史数据序列长度
        :param num_stream_features: 历史流量数据特征数（有几个流量输入）
        :param num_rainfall_features: 历史降雨数据特征数（有几个雨量站）
        :param num_evap_features: 历史蒸发数据特征数（有几个蒸发数据）
        :param num_non_target_features: 非目标数据特征数（总特征数-1）
        :param hidden_size: 模型隐藏层维度
        :param num_layers: LSTM层数
        :param dropout: 丢弃率
        :param embed_dim: 映射维度
        :param m: 映射维度与多头注意力的头数之间的倍数
        '''
        super(StreamModel, self).__init__()

        self.sequence_length = sequence_length
        self.num_stream_features = num_stream_features
        self.num_rainfall_features = num_rainfall_features
        self.num_evap_features = num_evap_features
        self.num_non_target_features = num_non_target_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.m = m

        #编码历史流量数据
        self.lstm1 = nn.LSTM(
            input_size=num_stream_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True)

        #对降雨序列数据进行映射
        self.embedding1 = nn.Linear(num_rainfall_features, embed_dim)

        self.embedding2 = nn.Linear(hidden_size, embed_dim)

        #处理历史降雨数据
        self.attn1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=embed_dim // m,
            dropout=dropout,
            batch_first=True
        )

        #处理经过注意力计算的历史降雨数据
        self.lstm2 = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        #处理历史蒸发数据
        self.lstm3 = nn.LSTM(
            input_size=num_evap_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        #处理当前时间步非目标序列数据
        self.lstm4 = nn.LSTM(
            input_size=num_non_target_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=embed_dim // m,
            dropout=dropout,
            batch_first=True
        )

        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 线性变化
        self.linear = nn.Linear(sequence_length*3 + 1, 1)

        # 初始化参数
        self._init_weights()


    def _init_weights(self):
        """初始化模型参数"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # LSTM forget gate bias
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1)


    def forward(self,sequences_stream:torch.Tensor,sequences_rainfall:
                        torch.Tensor,sequences_evap:torch.Tensor,current_features:torch.Tensor)->torch.Tensor:
        '''              
        :param sequences_stream: 历史流量序列数据
        :param sequences_rainfall: 历史降雨序列数据
        :param sequences_evap: 历史蒸发序列数据
        :param current_features: 当前时间步非目标特征数据
        :return: 预测值
        '''
        # 处理历史流量序列数据
        sequences_stream_lstm_out, _ = self.lstm1(sequences_stream)  # [batch_size,sequences_length,hidden_size]


        # 处理历史降雨序列数据
        sequences_rainfall_embedd = self.embedding1(sequences_rainfall)
        sequences_rainfall_attn_out, _ = self.attn1(sequences_rainfall_embedd, sequences_stream_lstm_out, sequences_rainfall_embedd)
        sequences_rainfall_attn_out = sequences_rainfall_embedd + sequences_rainfall_attn_out
        sequences_rainfall_lstm_out, _ = self.lstm2(sequences_rainfall_attn_out)  # [batch_size,sequences_length,hidden_size]


        # 处理历史蒸发序列数据
        sequences_evap_lstm_out, _ = self.lstm3(sequences_evap)  # [batch_size,sequences_length,hidden_size]


        # 处理当前时间步非目标序列数据
        current_features_lstm_out, _ = self.lstm4(current_features)  # [batch_size,hidden_size]
        current_features_lstm_out = current_features_lstm_out.unsqueeze(1) # [batch_size,1,hidden_size]


        combined_output = torch.cat([
            sequences_stream_lstm_out,sequences_rainfall_lstm_out,sequences_evap_lstm_out,current_features_lstm_out],
            dim=1)# [batch_size,sequences_length*3+1,hidden_size]
        combined_output_embedding = self.embedding2(combined_output)
        combined_output_embedding=combined_output_embedding+combined_output
        combined_output_attn_out, _ = self.attn2(combined_output_embedding,combined_output_embedding,combined_output_embedding) # [batch_size,sequences_length*3+1,embedd_dim]
        combined_output = combined_output.permute(0, 2, 1)  # [batch_size,embedd_dim,sequences_length*3+1]

        merge_output = self.linear(combined_output) # [batch_size,embedd_dim,1]
        merge_output = merge_output.permute(0, 2, 1) # [batch_size,1,embedd_dim]

        output = self.output_projection(merge_output).squeeze(-1) # [batch_size,  1]

        return output


    def predict(self,sequences_stream:torch.Tensor,sequences_rainfall:torch.Tensor,
                sequences_evap:torch.Tensor,current_features:torch.Tensor)->torch.Tensor:

        self.eval()
        with torch.no_grad():
            prediction=self.forward(sequences_stream,sequences_rainfall,sequences_evap,current_features)
        return prediction


if __name__ == '__main__':
    model = StreamModel()
    print(model)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    sequences_stream_test = torch.randn(32, 24, 3)
    sequences_rainfall_test = torch.randn(32, 24, 23)
    sequences_evap_test = torch.randn(32, 24, 1)
    current_features_test = torch.randn(32, 26)

    output=model.forward(sequences_stream_test, sequences_rainfall_test, sequences_evap_test, current_features_test)
    print(output)
