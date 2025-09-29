#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小时降雨数据拆分脚本
将data/data_p文件夹中的CSV文件从小时数据拆分为5分钟间隔数据
拆分规则：当两组数据间的时间间隔小于等于2h时，按5分钟线性插值
"""

import pandas as pd
import os
import glob
import numpy as np
from datetime import datetime, timedelta

def process_hourly_data(df):
    """
    处理单个站点的小时降雨数据，拆分为5分钟间隔
    """
    # 确保数据按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 存储结果数据
    result_data = []
    
    for i in range(len(df)):
        current_time = df.iloc[i]['datetime']
        current_value = df.iloc[i]['rainfall']
        station_id = df.iloc[i]['station_id']
        
        # 当前时间点的数据
        result_data.append({
            'station_id': station_id,
            'datetime': current_time,
            'rainfall': current_value
        })
        
        # 检查是否有下一个数据点
        if i < len(df) - 1:
            next_time = df.iloc[i + 1]['datetime']
            next_value = df.iloc[i + 1]['rainfall']
            
            # 计算时间间隔（小时）
            time_diff = (next_time - current_time).total_seconds() / 3600
            
            # 如果时间间隔小于等于2小时，进行线性插值
            if time_diff <= 2.0:
                # 计算5分钟间隔的数据点数量
                num_intervals = int(time_diff * 12)  # 每小时12个5分钟间隔
                
                if num_intervals > 1:
                    # 线性插值
                    for j in range(1, num_intervals):
                        interp_time = current_time + timedelta(minutes=5 * j)
                        # 线性插值计算降雨量
                        ratio = j / num_intervals
                        interp_value = current_value + (next_value - current_value) * ratio
                        
                        result_data.append({
                            'station_id': station_id,
                            'datetime': interp_time,
                            'rainfall': round(interp_value, 2)
                        })
    
    return pd.DataFrame(result_data)

def load_and_process_csv(file_path):
    """
    加载并处理单个CSV文件
    """
    filename = os.path.basename(file_path)
    station_id = filename.replace('.csv', '')
    print(f"正在处理文件: {filename}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, header=None, names=['station_id', 'datetime', 'rainfall'])
        
        # 处理数据中的小数点问题（如.4 -> 0.4）
        df['rainfall'] = df['rainfall'].astype(str)
        df['rainfall'] = df['rainfall'].apply(lambda x: f"0{x}" if x.startswith('.') else x)
        df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce').fillna(0)
        
        # 转换日期时间格式
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"原数据行数: {len(df)}")
        
        # 处理小时数据，拆分为5分钟间隔
        result_df = process_hourly_data(df)
        
        print(f"拆分后数据行数: {len(result_df)}")
        print(f"时间范围: {result_df['datetime'].min()} 到 {result_df['datetime'].max()}")
        
        return result_df, station_id
        
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")
        return None, None

def process_data_p_folder(input_folder, output_folder):
    """
    处理data_p文件夹中的所有CSV文件
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    print("=" * 60)
    
    processed_count = 0
    failed_count = 0
    
    # 处理每个文件
    for csv_file in csv_files:
        result_df, station_id = load_and_process_csv(csv_file)
        
        if result_df is not None:
            # 保存处理后的数据
            output_file = os.path.join(output_folder, f"{station_id}_5min.csv")
            result_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"输出文件: {output_file}")
            processed_count += 1
        else:
            failed_count += 1
        
        print("-" * 40)
    
    print("=" * 60)
    print(f"处理完成！")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")

def main():
    """
    主函数
    """
    # 设置路径
    input_folder = "data/data_p"
    output_folder = "data/data_p_5min"
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 未找到输入文件夹 {input_folder}")
        return
    
    print("小时降雨数据拆分为5分钟间隔脚本")
    print("=" * 60)
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print("拆分规则: 当时间间隔≤2小时时，按5分钟线性插值")
    print("=" * 60)
    
    # 处理数据
    process_data_p_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()