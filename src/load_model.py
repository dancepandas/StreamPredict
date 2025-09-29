import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.Model import StreamModel
from src.DataProcess import DataProcess_Predict
import config
from datetime import datetime

import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimSun']# 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class StreamPredict():
    def __init__(self, model_path:str, data_file:str,standard_scalar:str):
        self.model_path = model_path
        self.data_file = data_file
        self.standard_scalar = standard_scalar
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        # 加载检查点
        print(f"正在从 {self.model_path} 加载模型...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 从检查点中获取模型配置
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            print("正在导入保存的模型配置...")
            # 创建模型
            self.model = StreamModel(**model_config).to(self.device)
            print('模型配置导入成功！')
            print('正在加载模型权重...')
        else:
            # 如果没有保存配置，使用默认配置
            print("警告: 检查点中没有找到模型配置，使用默认配置")
            default_config = {
                'sequence_length': config.sequence_length,
                'num_stream_features': config.num_stream_features,
                'num_rainfall_features': config.num_rainfall_features,
                'num_evap_features': config.num_evap_features,
                'num_non_target_features': config.num_non_target_features,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'dropout': config.dropout,
                'embed_dim': config.embed_dim,
                'm': config.m
            }
            # 创建模型
            self.model = StreamModel(**default_config).to(self.device)
            print('模型配置导入成功！')
            print('正在加载模型权重...')
        

        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("模型权重加载成功！")
        else:
            # 如果是直接的state_dict格式
            self.model.load_state_dict(checkpoint)
            print("模型权重加载成功！")

        return self.model

    def crate_data(self):
        DPP = DataProcess_Predict(self.data_file, self.standard_scalar)
        DPP.read_predict_data()
        data_dict_p=DPP.prepare_data_for_predict()
        data_loader = DPP.create_dataloaders(data_dict_p)
        return data_loader

    def predict(self):

        self.model.eval()
        data_loader = self.crate_data()
        all_predictions = []
        all_times = []
        with torch.no_grad():
                for batch in tqdm(data_loader, desc="predict..."):
                    # 获取数据
                    sequences_stream = batch['sequence_stream'].to(self.device)
                    sequences_rainfall = batch['sequence_rainfall'].to(self.device)
                    sequences_evap = batch['sequence_evap'].to(self.device)
                    current_features = batch['current_features'].to(self.device)

                    # 预测
                    predictions = self.model.forward(sequences_stream, sequences_rainfall, sequences_evap, current_features)
                    predictions = predictions.cpu().numpy()

                    all_predictions.extend(predictions)
                    all_times.extend(batch['datatime'].tolist())
        all_predictions_np = np.array(all_predictions)

        return all_predictions_np,all_times


    def plot_predictions(self, all_predictions_np, all_times, save_path: str =config.fig_save_file+'/pre_result.png'):

        sorted_indices = np.argsort(all_times)
        sorted_predictions = np.concatenate(all_predictions_np, axis=0)[sorted_indices]
        sorted_datetimes = [datetime.fromtimestamp(ts) for ts in np.array(all_times)[sorted_indices]]

        plt.figure(figsize=(15, 5))
        x = range(len(sorted_predictions))
        plt.plot(x, sorted_predictions)
        plt.xlabel('样本索引')
        plt.ylabel('流量值')
        plt.title('预测结果')
        plt.show()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        print(f"训练结果图已保存到 {save_path}")

if __name__ == '__main__':
    SP=StreamPredict(r'D:\PycharmProjects\StreamPredict\src\best_model.pth',r'D:\PycharmProjects\StreamPredict\merged_all_data.csv',
                     r'D:\PycharmProjects\StreamPredict\standard_scalar')
    SP.load_model()
    predictions,time_list=SP.predict()
    SP.plot_predictions(predictions,time_list)