
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
 # 模型结构参数

sequence_length = 24 # 历史数据序列长度
num_stream_features = 3 # 历史流量数据序列特征数
num_rainfall_features = 23 # 历史降雨数据序列特征数
num_evap_features = 1 # 历史蒸发数据序列特征数
num_non_target_features = 26 # 预测时间非目标序列特征数（总特征数-1）
hidden_size = 128 # 隐藏层维度
num_layers = 2 # LSTM层数
dropout = 0.1  # 丢弃率
embed_dim = 128    # 映射维度
m = 16  # 映射维度与多头注意力头数的倍数


 # 数据处理参数
test_size=0.2 # 测试集占比
val_size=0.1  # 验证集占比
random_state=42 # 随机种子
batch_size=32 # 一个训练批次数据大小

 # 模型训练参数
num_epochs=100 # 训练周期
learning_rate=0.001 # 学习率

# 文件存储路径
# 原始训练数据路径
data_file = script_dir+'\data\merged_all_data.csv'

# 预测数据存储目录
data_predict_file = script_dir+'\data\pre_data'


# 训练数据标准化文件存储路径
standard_scalar_file=script_dir+'\standard_scalar'

# 模型文件保存路径
model_save_file=script_dir+r'\model_file\best_model.pth'

# 图片保存路径
fig_save_file=script_dir+r'\fig'