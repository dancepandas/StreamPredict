import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import joblib


class StreamDataset(Dataset):
    """PyTorch数据集类，支持训练时的teacher forcing和推理时的自回归预测"""

    def __init__(self, sequences_stream, sequences_rainfall,sequences_evap,targets, current_features, mode='train'):
        self.sequences_stream = sequences_stream  # 历史24步流量特征数据
        self.sequences_rainfall = sequences_rainfall  # 历史24步降雨特征数据
        self.sequences_evap = sequences_evap # 历史24步蒸发特征数据
        self.targets = targets  # 目标流量值
        self.current_features = current_features  # 当前时间步非目标特征

    def __len__(self):
        return len(self.sequences_stream)

    def __getitem__(self, idx):
        sequence_stream = self.sequences_stream[idx]
        sequence_rainfall = self.sequences_rainfall[idx]
        sequence_evap = self.sequences_evap[idx]
        target = self.targets[idx]
        current_feature = self.current_features[idx]
        return {
            'sequence_stream': torch.FloatTensor(sequence_stream),
            'sequence_rainfall': torch.FloatTensor(sequence_rainfall),
            'sequence_evap': torch.FloatTensor(sequence_evap),
            'target': torch.FloatTensor(target),
            'current_features': torch.FloatTensor(current_feature)
        }

class DataProcess(object):
    def __init__(self, data_file, standard_scalar_file,sequence_length=24):
        self.data_file = data_file
        self.standard_scalar_file = standard_scalar_file
        self.dataframe = None
        self.sequence_length = sequence_length  # 输入序列长度（24个时间步）
        self.scaler1 = StandardScaler()  # 流量数据标准化
        self.scaler2 = StandardScaler()   # 降雨数据标准化
        self.scaler3 = StandardScaler()  # 蒸发数据标准化
        self.scaler4 = StandardScaler()  #非目标特征数据标准化
        self.feature_stream_columns = []  # 流量特征列名
        self.feature_rainfall_columns = []  # 降雨特征列名
        self.feature_evap_columns = [] # 蒸发特征列名
        self.target_column = '雁翅5min洪水流量摘录'  # 目标列

    def read_data(self):
        """读取和预处理数据"""
        self.dataframe = pd.read_csv(self.data_file)

        self.dataframe['datetime'] = pd.to_datetime(self.dataframe['datetime'], format='%Y-%m-%d %H:%M:%S')

        self.dataframe.set_index("datetime", inplace=True)

        # 定义特征列
        self.feature_stream_columns = [
                                   '官厅水库坝下5min流量摘录',
                                   '斋堂水库5min流量摘录',
                                   '雁翅5min洪水流量摘录'
                               ]

        self.feature_rainfall_columns = ['30746200', '30746400', '30746450', '30746600', '30746700',
                                    '30746750', '30746900', '30747000', '30747100', '30747200',
                                    '30747300', '30747350', '30747370', '30747400', '30747420',
                                    '30747500', '30747530', '30747600', '30747635', '30747660',
                                    '30747700', '30747800', '30747850']

        self.feature_evap_columns = ['官厅水库5min蒸发']



        return self.dataframe

    def create_sequences_for_autoregressive_training(self):
        """
        为自回归训练创建序列数据
        返回格式：
        - sequences: [N, sequence_length, num_features] - 输入24步历史数据
        - targets: [N, 1] - 下一时间步目标流量
        - current_features: [N, num_non_target_features] - 当前时间步非流量特征
        """
        sequences_rainfall  = []
        sequences_stream = []
        sequences_evap = []
        targets = []
        current_features = []



        all_datetime_list = self.dataframe.index.tolist()

        # 非目标特征列（除去雁翅流量）
        non_target_features = [col for col in self.feature_stream_columns+self.feature_evap_columns+self.feature_rainfall_columns if col != self.target_column]

        for i in range(self.sequence_length, len(all_datetime_list)):
            current_time = all_datetime_list[i]

            # 检查数据连续性
            sequence_times = all_datetime_list[i - self.sequence_length:i]
            is_continuous = True
            for j in range(1, len(sequence_times)):
                time_diff = sequence_times[j] - sequence_times[j - 1]
                if time_diff != timedelta(minutes=5):
                    is_continuous = False
                    break

            if not is_continuous:
                continue

            # 持续检查当前时间步和前一个时间步的连续性
            if current_time - sequence_times[-1] != timedelta(minutes=5):
                continue

            # 创建序列数据（前24步所有特征）
            sequence_stream_data = []
            sequence_rainfall_data = []
            sequence_evap_data = []

            for seq_time in sequence_times:
                row_stream_data = []
                row_rainfall_data = []
                row_evap_data = []
                for stream_feature in self.feature_stream_columns:
                    row_stream_data.append(self.dataframe.loc[seq_time, stream_feature])
                sequence_stream_data.append(row_stream_data)

                for rainfall_feature in self.feature_rainfall_columns:
                    row_rainfall_data.append(self.dataframe.loc[seq_time, rainfall_feature])
                sequence_rainfall_data.append(row_rainfall_data)

                for evap_feature in self.feature_evap_columns:
                    row_evap_data.append(self.dataframe.loc[seq_time, evap_feature])
                sequence_evap_data.append(row_evap_data)




            # 当前时间步的非目标特征
            current_feature_data = []
            for feature in non_target_features:
                current_feature_data.append(self.dataframe.loc[current_time, feature])

            # 目标值（当前时间步的流量）
            target_value = self.dataframe.loc[current_time, self.target_column]

            sequences_stream.append(sequence_stream_data)
            sequences_rainfall.append(sequence_rainfall_data)
            sequences_evap.append(sequence_evap_data)
            current_features.append(current_feature_data)
            targets.append([target_value])

        sequences_stream=np.array(sequences_stream)
        sequences_rainfall=np.array(sequences_rainfall)
        sequences_evap=np.array(sequences_evap)
        current_features=np.array(current_features)
        targets=np.array(targets)

        return sequences_stream,sequences_rainfall,sequences_evap,targets,current_features

    def scaler_data_2d(self,data,data_type='rainfall'):
        data_2d = data.reshape(-1, data.shape[-1])
        if  data_type == 'stream':
            self.scaler1.fit(data_2d)
            joblib.dump(self.scaler1, self.standard_scalar_file+"/standard_stream_scaler.pkl")
            return self.scaler1
        if data_type == 'rainfall':
            self.scaler2.fit(data_2d)
            joblib.dump(self.scaler2, self.standard_scalar_file+"/standard_rainfall_scaler.pkl")
            return  self.scaler2
        if data_type == 'evap':
            self.scaler3.fit(data_2d)
            joblib.dump(self.scaler3, self.standard_scalar_file+"/standard_evap_scaler.pkl")
            return  self.scaler3
        if data_type == 'current_features':
            self.scaler4.fit(data)
            joblib.dump(self.scaler4, self.standard_scalar_file+"/standard_current_features_scaler.pkl")
            return self.scaler4


    def scaler_data(self,data,data_type='rainfall'):
        if data_type == 'stream':
            data_scaled = self.scaler1.transform(
                data.reshape(-1, data.shape[-1])
            ).reshape(data.shape)
        if data_type == 'rainfall':
            data_scaled = self.scaler2.transform(
                data.reshape(-1, data.shape[-1])
            ).reshape(data.shape)
        if data_type == 'evap':
            data_scaled = self.scaler3.transform(
                data.reshape(-1, data.shape[-1])
            ).reshape(data.shape)
        if data_type == 'current_features':
            data_scaled = self.scaler4.transform(data)
        return data_scaled



    def prepare_data_for_training(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        准备训练数据，包括数据分割和标准化
        """
        sequences_stream,sequences_rainfall,sequences_evap,targets,current_features = self.create_sequences_for_autoregressive_training()

        print(f"总数据样本数: {len(sequences_stream)}")
        print(f"流量序列数据形状: {sequences_stream.shape}")
        print(f"降雨序列数据形状: {sequences_rainfall.shape}")
        print(f"蒸发序列数据形状: {sequences_evap.shape}")
        print(f"目标数据形状: {targets.shape}")
        print(f"当前特征数据形状: {current_features.shape}")

        # 第一次分割：分出训练集和临时集
        X_temp = np.arange(len(sequences_stream))
        X_train_idx, X_temp_idx = train_test_split(
            X_temp, test_size=(test_size + val_size), random_state=random_state, shuffle=False
        )

        # 第二次分割：从临时集中分出验证集和测试集
        val_ratio = val_size / (test_size + val_size)
        X_val_idx, X_test_idx = train_test_split(
            X_temp_idx, test_size=(1 - val_ratio), random_state=random_state, shuffle=False
        )

        # 提取对应数据
        train_sequences_stream = sequences_stream[X_train_idx]
        train_sequences_rainfall = sequences_rainfall[X_train_idx]
        train_sequences_evap = sequences_evap[X_train_idx]
        train_targets = targets[X_train_idx]
        train_current_features = current_features[X_train_idx]

        val_sequences_stream = sequences_stream[X_val_idx]
        val_sequences_rainfall = sequences_rainfall[X_val_idx]
        val_sequences_evap = sequences_evap[X_val_idx]
        val_targets = targets[X_val_idx]
        val_current_features = current_features[X_val_idx]

        test_sequences_stream = sequences_stream[X_test_idx]
        test_sequences_rainfall = sequences_rainfall[X_test_idx]
        test_sequences_evap = sequences_evap[X_test_idx]
        test_targets = targets[X_test_idx]
        test_current_features = current_features[X_test_idx]

        # 数据标准化（仅基于训练集）
        # 将序列数据重塑为2D进行标准化

        self.scaler_data_2d(train_sequences_stream,data_type='stream')
        self.scaler_data_2d(train_sequences_rainfall,data_type='rainfall')
        self.scaler_data_2d(train_sequences_evap,data_type='evap')
        self.scaler_data_2d(current_features,data_type='current_features')

        # 应用标准化
        train_sequences_stream_scaled = self.scaler_data(train_sequences_stream,data_type='stream')
        train_sequences_rainfall_scaled = self.scaler_data(train_sequences_rainfall,data_type='rainfall')
        train_sequences_evap_scaled = self.scaler_data(train_sequences_evap,data_type='evap')

        #print('标准化后的流量序列训练数据形状：', train_sequences_stream_scaled.shape)

        val_sequences_stream_scaled = self.scaler_data(val_sequences_stream,data_type='stream')
        val_sequences_rainfall_scaled = self.scaler_data(val_sequences_rainfall,data_type='rainfall')
        val_sequences_evap_scaled = self.scaler_data(val_sequences_evap,data_type='evap')

        #print('标准化后的流量序列验证数据形状：', val_sequences_stream_scaled.shape)

        test_sequences_stream_scaled = self.scaler_data(test_sequences_stream,data_type='stream')
        test_sequences_rainfall_scaled = self.scaler_data(test_sequences_rainfall,data_type='rainfall')
        test_sequences_evap_scaled = self.scaler_data(test_sequences_evap,data_type='evap')

        #print('标准化后的流量序列测试数据形状：', test_sequences_stream_scaled.shape)

        train_current_features_scaled = self.scaler_data(train_current_features,data_type='current_features')
        test_current_features_scaled = self.scaler_data(test_current_features,data_type='current_features')
        val_current_features_scaled = self.scaler_data(val_current_features,data_type='current_features')

        #print('标准化后的非目标序列训练数据形状：', train_current_features_scaled.shape)


        return {
            'train': {
                'sequences_stream': train_sequences_stream_scaled,
                'sequences_rainfall': train_sequences_rainfall_scaled,
                'sequences_evap': train_sequences_evap_scaled,
                'targets': train_targets,
                'current_features': train_current_features_scaled
            },
            'val': {
                'sequences_stream': val_sequences_stream_scaled,
                'sequences_rainfall': val_sequences_rainfall_scaled,
                'sequences_evap': val_sequences_evap_scaled,
                'targets': val_targets,
                'current_features': val_current_features_scaled
            },
            'test': {
                'sequences_stream': test_sequences_stream_scaled,
                'sequences_rainfall': test_sequences_rainfall_scaled,
                'sequences_evap': test_sequences_evap_scaled,
                'targets': test_targets,
                'current_features': test_current_features_scaled
            }
        }

    def create_dataloaders(self, data_dict, batch_size=32, shuffle_train=True):
        """创建PyTorch DataLoader"""
        train_dataset = StreamDataset(
            data_dict['train']['sequences_stream'],
            data_dict['train']['sequences_rainfall'],
            data_dict['train']['sequences_evap'],
            data_dict['train']['targets'],
            data_dict['train']['current_features'],
            mode='train'
        )

        val_dataset = StreamDataset(
            data_dict['val']['sequences_stream'],
            data_dict['val']['sequences_rainfall'],
            data_dict['val']['sequences_evap'],
            data_dict['val']['targets'],
            data_dict['val']['current_features'],
            mode='train'
        )

        test_dataset = StreamDataset(
            data_dict['test']['sequences_stream'],
            data_dict['test']['sequences_rainfall'],
            data_dict['test']['sequences_evap'],
            data_dict['test']['targets'],
            data_dict['test']['current_features'],
            mode='inference'
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader



class StreamDatasetPredict(Dataset):
    """PyTorch数据集类，支持训练时的teacher forcing和推理时的自回归预测"""

    def __init__(self, sequences_stream, sequences_rainfall,sequences_evap, current_features, time_list):
        self.sequences_stream = sequences_stream  # 历史24步流量特征数据
        self.sequences_rainfall = sequences_rainfall  # 历史24步降雨特征数据
        self.sequences_evap = sequences_evap # 历史24步蒸发特征数据
        self.current_features = current_features  # 当前时间步非目标特征
        self.time_list = time_list # 时间戳

    def __len__(self):
        return len(self.sequences_stream)

    def __getitem__(self, idx):
        sequence_stream = self.sequences_stream[idx]
        sequence_rainfall = self.sequences_rainfall[idx]
        sequence_evap = self.sequences_evap[idx]
        current_feature = self.current_features[idx]
        datatime= self.time_list[idx]
        datatime_ts = datatime.timestamp()
        return {
            'sequence_stream': torch.FloatTensor(sequence_stream),
            'sequence_rainfall': torch.FloatTensor(sequence_rainfall),
            'sequence_evap': torch.FloatTensor(sequence_evap),
            'current_features': torch.FloatTensor(current_feature),
            'datatime': datatime_ts
        }

class DataProcess_Predict(object):
    def __init__(self,predict_data_file:str,standard_scaler:str,sequence_length:int=24):
        self.predict_data_file = predict_data_file
        self.standard_scaler = standard_scaler
        self.dataframe = None
        self.sequence_length = sequence_length  # 输入序列长度（24个时间步）
        self.feature_stream_columns = []  # 流量特征列名
        self.feature_rainfall_columns = []  # 降雨特征列名
        self.feature_evap_columns = []  # 蒸发特征列名
        self.target_column = '雁翅5min洪水流量摘录'  # 目标列

    def read_predict_data(self):
        """读取和预处理数据"""
        self.dataframe = pd.read_csv(self.predict_data_file)

        self.dataframe['datetime'] = pd.to_datetime(self.dataframe['datetime'], format='%Y-%m-%d %H:%M:%S')

        self.dataframe.set_index("datetime", inplace=True)


        # 定义特征列
        self.feature_stream_columns = [
            '官厅水库坝下5min流量摘录',
            '斋堂水库5min流量摘录',
            '雁翅5min洪水流量摘录'
        ]

        self.feature_rainfall_columns = ['30746200', '30746400', '30746450', '30746600', '30746700',
                                         '30746750', '30746900', '30747000', '30747100', '30747200',
                                         '30747300', '30747350', '30747370', '30747400', '30747420',
                                         '30747500', '30747530', '30747600', '30747635', '30747660',
                                         '30747700', '30747800', '30747850']

        self.feature_evap_columns = ['官厅水库5min蒸发']

        return self.dataframe

    def create_sequences_for_autoregressive_predict(self):
        """
        为自回归训练创建序列数据
        返回格式：
        - sequences: [N, sequence_length, num_features] - 输入24步历史数据
        - targets: [N, 1] - 下一时间步目标流量
        - current_features: [N, num_non_target_features] - 当前时间步非流量特征
        """
        sequences_rainfall  = []
        sequences_stream = []
        sequences_evap = []
        current_features = []
        time_list= []


        all_datetime_list = self.dataframe.index.tolist()

        # 非目标特征列（除去雁翅流量）
        non_target_features = [col for col in self.feature_stream_columns+self.feature_evap_columns+self.feature_rainfall_columns if col != self.target_column]

        for i in range(self.sequence_length, len(all_datetime_list)):
            current_time = all_datetime_list[i]
            # 检查数据连续性
            sequence_times = all_datetime_list[i - self.sequence_length:i]
            is_continuous = True
            for j in range(1, len(sequence_times)):
                time_diff = sequence_times[j] - sequence_times[j - 1]
                if time_diff != timedelta(minutes=5):
                    is_continuous = False
                    break

            if not is_continuous:
                continue

            # 持续检查当前时间步和前一个时间步的连续性
            if current_time - sequence_times[-1] != timedelta(minutes=5):
                continue

            # 创建序列数据（前24步所有特征）
            sequence_stream_data = []
            sequence_rainfall_data = []
            sequence_evap_data = []

            for seq_time in sequence_times:
                row_stream_data = []
                row_rainfall_data = []
                row_evap_data = []
                for stream_feature in self.feature_stream_columns:
                    row_stream_data.append(self.dataframe.loc[seq_time, stream_feature])
                sequence_stream_data.append(row_stream_data)

                for rainfall_feature in self.feature_rainfall_columns:
                    row_rainfall_data.append(self.dataframe.loc[seq_time, rainfall_feature])
                sequence_rainfall_data.append(row_rainfall_data)

                for evap_feature in self.feature_evap_columns:
                    row_evap_data.append(self.dataframe.loc[seq_time, evap_feature])
                sequence_evap_data.append(row_evap_data)




            # 当前时间步的非目标特征
            current_feature_data = []
            for feature in non_target_features:
                current_feature_data.append(self.dataframe.loc[current_time, feature])


            time_list.append(current_time)
            sequences_stream.append(sequence_stream_data)
            sequences_rainfall.append(sequence_rainfall_data)
            sequences_evap.append(sequence_evap_data)
            current_features.append(current_feature_data)


        sequences_stream=np.array(sequences_stream)
        sequences_rainfall=np.array(sequences_rainfall)
        sequences_evap=np.array(sequences_evap)
        current_features=np.array(current_features)

        return sequences_stream,sequences_rainfall,sequences_evap,current_features,time_list


    def scaler_data(self,data,data_type='rainfall'):
        if data_type == 'stream':
            self.scaler1 = joblib.load(self.standard_scaler+'/standard_stream_scaler.pkl')
            data_scaled = self.scaler1.transform(
                data.reshape(-1, data.shape[-1])
            ).reshape(data.shape)


        elif data_type == 'rainfall':
            self.scaler2 = joblib.load(self.standard_scaler+'/standard_rainfall_scaler.pkl')
            data_scaled = self.scaler2.transform(
                data.reshape(-1, data.shape[-1])
            ).reshape(data.shape)

        elif data_type == 'evap':
            self.scaler3 = joblib.load(self.standard_scaler+'/standard_evap_scaler.pkl')
            data_scaled = self.scaler3.transform(
                data.reshape(-1, data.shape[-1])
            ).reshape(data.shape)

        elif data_type == 'current_features':
            self.scaler4 = joblib.load(self.standard_scaler+'/standard_current_features_scaler.pkl')
            data_scaled = self.scaler4.transform(data)

        return data_scaled

    def prepare_data_for_predict(self):
        """
        准备训练数据，包括数据分割和标准化
        """
        sequences_stream,sequences_rainfall,sequences_evap,current_features,time_list = self.create_sequences_for_autoregressive_predict()


        # 数据标准化处理

        sequences_stream_scaled = self.scaler_data(sequences_stream, data_type='stream')
        sequences_rainfall_scaled = self.scaler_data(sequences_rainfall, data_type='rainfall')
        sequences_evap_scaled = self.scaler_data(sequences_evap, data_type='evap')
        current_features_scaled = self.scaler_data(current_features, data_type='current_features')

        return {
                'sequences_stream': sequences_stream_scaled,
                'sequences_rainfall': sequences_rainfall_scaled,
                'sequences_evap': sequences_evap_scaled,
                'current_features': current_features_scaled,
                'datatime': time_list
            }
    def create_dataloaders(self, data_dict, batch_size=1, shuffle_train=True):
        """创建PyTorch DataLoader"""
        predict_dataset = StreamDatasetPredict(
            data_dict['sequences_stream'],
            data_dict['sequences_rainfall'],
            data_dict['sequences_evap'],
            data_dict['current_features'],
            data_dict['datatime']
        )


        data_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=shuffle_train)

        return data_loader




if __name__ == '__main__':
    data_file='D:\PycharmProjects\StreamPredict\merged_all_data.csv'
    standard_scaler = 'D:\PycharmProjects\StreamPredict\standard_scalar'

    print('数据处理模块准备中...')
    DP=DataProcess(data_file=data_file,standard_scalar_file=standard_scaler)
    print('数据读取中...')
    DP.read_data()
    print('数据处理中...')
    data_dict=DP.prepare_data_for_training()
    print('正在创建 DataLoader...')
    train_loader, val_loader, test_loader = DP.create_dataloaders(data_dict, batch_size=32)
    print(train_loader)
    print('DataLoader创建成功！')

    standard_scaler = 'D:\PycharmProjects\StreamPredict\standard_scalar'

    DPP=DataProcess_Predict(predict_data_file=data_file, standard_scaler=standard_scaler)
    DPP.read_predict_data()
    data_dict_p=DPP.prepare_data_for_predict()
    data_loader=DPP.create_dataloaders(data_dict_p)