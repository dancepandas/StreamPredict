#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并data和data_p_5min文件夹中的CSV数据
按照相同日期时间合并，使用文件名作为表头，缺少数据的日期时间将被丢弃
"""

import pandas as pd
import os
import glob
from datetime import datetime

def load_data_folder_csv(file_path):
    """
    加载data文件夹中的CSV文件（原有的5分钟数据）
    """
    filename = os.path.basename(file_path)
    file_base_name = filename.replace('.csv', '')
    print(f"正在处理data文件夹文件: {filename}")
    
    try:
        # 先读取第一行判断格式
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('DATE,'):
            # 有表头的文件（如雁翅断面5min降雨.csv）
            df = pd.read_csv(file_path, encoding='utf-8')
            df.rename(columns={'DATE': 'datetime'}, inplace=True)
            # 其他列添加文件前缀
            columns_to_rename = {col: f"{file_base_name}_{col}" for col in df.columns if col != 'datetime'}
            df.rename(columns=columns_to_rename, inplace=True)
        else:
            # 没有表头的文件（站点ID,日期时间,数值格式）
            df = pd.read_csv(file_path, header=None, encoding='utf-8')
            if df.shape[1] == 3:
                df.columns = ['station_id', 'datetime', file_base_name]
                df = df[['datetime', file_base_name]]
            else:
                print(f"警告: 文件 {filename} 格式不符合预期")
                return None
        
        # 转换日期时间格式
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.floor('min')
        
        print(f"  加载完成，数据形状: {df.shape}")
        print(f"  时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        return df
        
    except Exception as e:
        print(f"读取文件 {filename} 时出错: {str(e)}")
        return None

def load_data_p_5min_csv(file_path):
    """
    加载data_p_5min文件夹中的CSV文件（转换后的5分钟降雨数据）
    """
    filename = os.path.basename(file_path)
    file_base_name = filename.replace('_5min.csv', '')
    print(f"正在处理data_p_5min文件夹文件: {filename}")
    
    try:
        # 这些文件有表头：station_id,datetime,rainfall
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 只保留datetime和rainfall列，将rainfall列重命名为站点ID
        df = df[['datetime', 'rainfall']]
        df.rename(columns={'rainfall': file_base_name}, inplace=True)
        
        # 转换日期时间格式
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.floor('min')
        
        print(f"  加载完成，数据形状: {df.shape}")
        print(f"  时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        return df
        
    except Exception as e:
        print(f"读取文件 {filename} 时出错: {str(e)}")
        return None

def merge_all_csv_data(data_folder, data_p_5min_folder, output_file):
    """
    合并两个文件夹中的所有CSV数据
    """
    print("开始加载和合并CSV数据...")
    print("=" * 60)
    
    dataframes = []
    
    # 1. 处理data文件夹中的CSV文件
    print("1. 处理data文件夹中的CSV文件:")
    data_csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    for csv_file in data_csv_files:
        df = load_data_folder_csv(csv_file)
        if df is not None:
            dataframes.append(df)
    
    print(f"data文件夹成功加载 {len([df for df in dataframes])} 个文件")
    print()
    
    # 2. 处理data_p_5min文件夹中的CSV文件
    print("2. 处理data_p_5min文件夹中的CSV文件:")
    data_p_5min_files = glob.glob(os.path.join(data_p_5min_folder, "*.csv"))
    
    for csv_file in data_p_5min_files:
        df = load_data_p_5min_csv(csv_file)
        if df is not None:
            dataframes.append(df)

    total_data_files = len(data_csv_files)
    total_p5min_files = len(data_p_5min_files)
    successful_loads = len(dataframes)
    
    print(f"data_p_5min文件夹成功加载 {successful_loads - total_p5min_files-total_data_files} 个文件")
    print()
    print(f"总计成功加载 {successful_loads} 个文件")
    
    if not dataframes:
        print("没有成功加载任何数据文件")
        return None
    
    print("=" * 60)
    print("开始合并数据...")

    
    # 显示所有文件的时间范围
    print("\n各文件时间范围汇总:")
    all_datetime_sets = []
    time_len=[]

    for i, df in enumerate(dataframes):
        start_time = df['datetime'].min()

        end_time = df['datetime'].max()

        datetime_set = set(df['datetime'])

        time_len.append(len(datetime_set))

        all_datetime_sets.append(datetime_set)
        # 获取文件名称信息
        file_cols = [col for col in df.columns if col != 'datetime']
        file_info = file_cols[0] if file_cols else f"文件{i+1}"
        print(f"  {file_info}: {start_time} 到 {end_time} (共{len(datetime_set)}个时间点)")

    index=time_len.index(max(time_len))
    if index !=0:
        temp_time=all_datetime_sets[0]
        temp_dataframe=dataframes[0]
        all_datetime_sets[0]=all_datetime_sets[index]
        dataframes[0]=dataframes[index]
        all_datetime_sets[index]=temp_time
        dataframes[index]=temp_dataframe
    # 计算所有文件共同拥有的时间点

    if all_datetime_sets:
        common_datetimes = all_datetime_sets[0]
        for datetime_set in all_datetime_sets[1:]:
            print(f'时间长度:{len(datetime_set)}')
            common_datetimes = common_datetimes.intersection(datetime_set)
        
        print(f"\n所有文件共同拥有的时间点数量: {len(common_datetimes)}")
        if len(common_datetimes) > 0:
            common_times_list = sorted(list(common_datetimes))
            print(f"共同时间范围: {common_times_list[0]} 到 {common_times_list[-1]}")
        else:
            print("警告: 所有文件没有共同的时间点，合并结果将为空！")
    
    print(f"\n初始数据（第一个文件）行数: {len(dataframes[0])}")
    
    # 使用内连接合并所有数据框
    merged_df = dataframes[0]
    
    for i in range(1, len(dataframes)):
        before_rows = len(merged_df)
        # 使用内连接，只保留所有文件都共同拥有的日期时间
        merged_df = pd.merge(merged_df, dataframes[i], on='datetime', how='inner')
        after_rows = len(merged_df)
        print(f"与第 {i+1} 个文件合并后，剩余共同时间点: {after_rows} (减少了 {before_rows - after_rows} 个时间点)")
    
    # 按日期时间排序
    merged_df = merged_df.sort_values('datetime')
    
    # 将datetime列放在第一列
    cols = ['datetime'] + [col for col in merged_df.columns if col != 'datetime']
    merged_df = merged_df[cols]
    
    # 保存合并后的数据
    merged_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("=" * 60)
    print(f"数据合并完成！")
    print(f"合并后的文件保存在: {output_file}")
    print(f"最终数据形状: {merged_df.shape}")
    print(f"日期时间范围: {merged_df['datetime'].min()} 到 {merged_df['datetime'].max()}")
    
    # 显示列名信息
    print(f"\n列名信息:")
    print(f"总共 {len(merged_df.columns)} 列:")
    for i, col in enumerate(merged_df.columns):
        print(f"  {i+1}. {col}")
    
    # 显示前几行数据作为预览
    print("\n数据预览:")
    print(merged_df.head())
    
    # 显示数据统计信息
    print(f"\n数据统计:")
    print(f"非空值统计:")
    null_counts = merged_df.isnull().sum()
    for col in merged_df.columns:
        if col != 'datetime':
            null_count = null_counts[col]
            total_count = len(merged_df)
            if total_count > 0:
                non_null_count = total_count - null_count
                percentage = 100 * non_null_count / total_count
                print(f"  {col}: {non_null_count}/{total_count} ({percentage:.1f}% 非空)")
            else:
                print(f"  {col}: 0/0 (无数据)")
    
    return merged_df

def main():
    """
    主函数
    """
    # 设置文件路径
    data_folder = "data"
    data_p_5min_folder = "data/data_p_5min"
    output_file = "merged_all_data.csv"
    
    # 检查文件夹是否存在
    if not os.path.exists(data_folder):
        print(f"错误: 未找到 {data_folder} 文件夹")
        return
    
    if not os.path.exists(data_p_5min_folder):
        print(f"错误: 未找到 {data_p_5min_folder} 文件夹")
        return
    
    print("合并data和data_p_5min文件夹中的CSV数据")
    print("=" * 60)
    print(f"data文件夹: {data_folder}")
    print(f"data_p_5min文件夹: {data_p_5min_folder}")
    print(f"输出文件: {output_file}")
    print("合并规则: 使用内连接，缺少数据的日期时间将被丢弃")
    print("=" * 60)
    
    # 执行合并
    merged_data = merge_all_csv_data(data_folder, data_p_5min_folder, output_file)
    
    if merged_data is not None:
        print("=" * 60)
        print("合并完成！")
        print(f"合并后的数据已保存为: {output_file}")
        print("文件使用UTF-8编码，可以在Excel或其他工具中打开查看。")
    else:
        print("合并失败！")

if __name__ == "__main__":
    main()