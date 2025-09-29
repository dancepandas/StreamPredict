import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
import os
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from DataProcess import  DataProcess

from Model import StreamModel
import config


mpl.rcParams['font.sans-serif'] = ['SimSun']# 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def rmse_loss(input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(input ,target))

def huber_rmse_loss(input:torch.Tensor, target:torch.Tensor,delta:float=5.0) -> torch.Tensor:
    #delta越大对峰值相应越明显
    huber_loss=F.huber_loss(input,target,delta)
    return torch.sqrt(huber_loss)

def nse_loss(input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
    loss1=torch.sum(torch.square(input - target))
    loss2=torch.sum(torch.square(target - torch.mean(target)))
    loss=loss1/(loss2+1e-8)
    return loss


class StreamModelTrainer(object):
    def __init__(self,data_file ,
                 standard_scalar_file,
                 model_config: Optional[Dict] = None,
                 device: str = 'auto'):
        self.data_file=data_file
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"正在使用 {self.device} 进行训练...")

        # 默认模型配置
        default_config = {
                 'sequence_length': config.sequence_length,
                 'num_stream_features':config.num_stream_features,
                 'num_rainfall_features': config.num_rainfall_features,
                 'num_evap_features': config.num_evap_features,
                 'num_non_target_features': config.num_non_target_features,
                 'hidden_size': config.hidden_size,
                 'num_layers':  config.num_layers,
                 'dropout': config.dropout,
                 'embed_dim': config.embed_dim,
                 'm': config.m
        }

        if model_config:
            default_config.update(model_config)

        # 模型参数
        self.model_config = default_config

        # 初始化模型
        self.model = StreamModel(**default_config).to(self.device)

        # 初始化数据处理器
        self.data_processor = DataProcess(data_file ,  standard_scalar_file)
        self.data_dict = None

        # 训练状态
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None


    def prepare_data(self,
                     test_size: float = config.test_size,
                     val_size: float = config.val_size,
                     batch_size: int = config.batch_size,
                     random_state: int = config.random_state) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备训练数据
        Args:
            test_size: 测试集比例
            val_size: 验证集比例
            batch_size: 批次大小
            random_state: 随机种子
        Returns:
            train_loader, val_loader, test_loader
        """
        print("正在读取和处理数据...")
        self.data_processor.read_data()

        print("正在准备训练数据...")
        data_dict = self.data_processor.prepare_data_for_training(
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )

        print('正在创建 DataLoader...')
        train_loader, val_loader, test_loader = self.data_processor.create_dataloaders(
            data_dict,
            batch_size=batch_size
        )
        print('DataLoader创建成功！')
        print('='*50)
        print(f"训练集样本数: {len(train_loader.dataset)}")
        print(f"验证集样本数: {len(val_loader.dataset)}")
        print(f"测试集样本数: {len(test_loader.dataset)}")

        # 保存数据字典用于后续分析
        self.data_dict = data_dict

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader, optimizer) -> float:
        """
        训练一个周期
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="train..."):
            # 获取数据
            sequences_stream = batch['sequence_stream'].to(self.device)
            sequences_rainfall = batch['sequence_rainfall'].to(self.device)
            sequences_evap = batch['sequence_evap'].to(self.device)
            targets = batch['target'].to(self.device)
            current_features = batch['current_features'].to(self.device)

            # 前向传播
            optimizer.zero_grad()
            predictions = self.model.forward(sequences_stream, sequences_rainfall, sequences_evap, current_features)

            # 计算损失
            #loss = nse_loss(predictions, targets)
            #loss = huber_rmse_loss(predictions, targets)
            loss = rmse_loss(targets, predictions)


            loss.backward()

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        验证一个周期
        Args:
            val_loader: 验证数据加载器
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val..."):
                # 获取数据
                sequences_stream = batch['sequence_stream'].to(self.device)
                sequences_rainfall = batch['sequence_rainfall'].to(self.device)
                sequences_evap = batch['sequence_evap'].to(self.device)
                targets = batch['target'].to(self.device)
                current_features = batch['current_features'].to(self.device)

                # 前向传播（验证时不使用教师强制）
                predictions = self.model.forward(sequences_stream, sequences_rainfall, sequences_evap, current_features)

                #loss = huber_rmse_loss(predictions, targets)
                loss = rmse_loss(targets, predictions)

                total_loss += loss.item()

                num_batches += 1

        return total_loss / num_batches

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-5,
              patience: int = 50,
              save_path: str =config.model_save_file) -> Dict:
        """
        训练模型
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练周期数
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            save_path: 模型保存路径
        Returns:
            训练历史记录
        """
        # 初始化优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate,weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        # 早停相关变量
        patience_counter = 0
        best_epoch = 0

        print(f"开始训练，总共 {num_epochs} 个周期...")

        for epoch in range(num_epochs):
            print(f"\n周期 {epoch + 1}/{num_epochs}")

            # 训练
            train_loss = self.train_epoch(train_loader, optimizer)

            # 验证
            val_loss,val_loss_rmse = self.validate_epoch(val_loader)


            # 学习率调度
            scheduler.step(train_loss)

            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"训练损失1-nse: {train_loss:.6f}, 验证损失rmse: {val_loss_rmse:.6f}")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.8f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0

                # 保存模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'model_config': self.model_config
                }, save_path)

                print(f"模型已保存到 {save_path}")
            else:
                patience_counter += 1

            # 早停检查
            if patience_counter >= patience:
                print(f"\n验证损失在 {patience} 个周期内没有改善，早停训练")
                print(f"最佳模型来自第 {best_epoch} 周期，验证损失: {self.best_val_loss:.6f}")
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': best_epoch
        }

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        评估模型性能
        Args:
            test_loader: 测试数据加载器
        Returns:
            评估指标字典
        """
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="test..."):
                # 获取数据
                sequences_stream = batch['sequence_stream'].to(self.device)
                sequences_rainfall = batch['sequence_rainfall'].to(self.device)
                sequences_evap = batch['sequence_evap'].to(self.device)
                targets = batch['target'].to(self.device)
                current_features = batch['current_features'].to(self.device)

                # 预测
                predictions = self.model.forward(sequences_stream, sequences_rainfall, sequences_evap, current_features)

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        # 转换为numpy数组
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)

        # 计算评估指标
        mse = mean_squared_error(targets_np, predictions_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets_np, predictions_np)
        r2 = r2_score(targets_np, predictions_np)

        # 计算平均绝对百分比误差
        mape = np.mean(np.abs((targets_np - predictions_np) / (targets_np + 1e-8))) * 100

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'predictions': predictions_np,
            'targets': targets_np
        }

        print("\n=== 测试集评估结果 ===")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        print(f"MAPE: {mape:.2f}%")

        return metrics

    def plot_training_history(self, save_path: str =config.fig_save_file+ '/training_history.png'):
        """
        绘制训练历史
        Args:
            save_path: 图片保存路径
        """
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失', color='blue')
        plt.plot(self.val_losses, label='验证损失', color='red')
        plt.xlabel('周期')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='训练损失 (log)', color='blue')
        plt.plot(self.val_losses, label='验证损失 (log)', color='red')
        plt.xlabel('周期')
        plt.ylabel('损失 (log scale)')
        plt.title('训练和验证损失 (对数尺度)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"训练历史图已保存到 {save_path}")

    def plot_predictions(self, metrics: Dict, num_samples: int = 200, save_path: str =config.fig_save_file+ '/predictions.png'):
        """
        绘制预测结果对比
        Args:
            metrics: 评估指标字典
            num_samples: 显示的样本数量
            save_path: 图片保存路径
        """
        predictions = metrics['predictions']
        targets = metrics['targets']

        plt.figure(figsize=(15, 5))

        # 时间序列对比
        plt.subplot(1, 2, 1)
        x = range(len(predictions))
        plt.plot(x, targets, label='真实值', color='blue', alpha=0.7)
        plt.plot(x, predictions, label='预测值', color='red', alpha=0.7)
        plt.xlabel('样本索引')
        plt.ylabel('流量值')
        plt.title('预测值 vs 真实值')
        plt.legend()
        plt.grid(True)

        # 散点图
        plt.subplot(1, 2, 2)
        plt.scatter(targets, predictions, alpha=0.6, color='green')

        # 完美预测线
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='完美预测')

        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'预测散点图 (R² = {metrics["R2"]:.4f})')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"预测结果图已保存到 {save_path}")




if __name__ == '__main__':
    # 使用示例
    trainer = StreamModelTrainer(
        data_file=r'D:\PycharmProjects\StreamPredict\merged_all_data.csv',
    standard_scalar_file = r'D:\PycharmProjects\StreamPredict\standard_scalar')

    # 准备数据
    train_loader, val_loader, test_loader = trainer.prepare_data(batch_size=64)

    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=0.001,
    )

    # 评估模型
    metrics = trainer.evaluate(test_loader)
    metrics_train = trainer.evaluate(train_loader)

    # 绘制结果
    trainer.plot_training_history()
    trainer.plot_predictions(metrics)

