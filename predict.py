import numpy as np
from datetime import datetime

from src.load_model import StreamPredict
import pandas as pd
import config
from utils.read_features_predict import creat_temp_data
from utils.check_time_continuous import check_time_continuous
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import plot_data

def main(model_path:str=config.model_save_file,data_file:str=config.data_predict_file+'/features.csv',
         temp_data_file:str=config.data_predict_file+'/temp.csv',sequence_length=config.sequence_length,tandard_scalar=config.standard_scalar_file):

    df=pd.read_csv(data_file)

    total_lines=df.shape[0]-(sequence_length+1)

    all_data=[]
    predict_result=0

    print(f'预计滚动预报 {total_lines} 步， 模型开始滚动预报...')
    for lines in range(total_lines):
        creat_temp_data(features_data_file=data_file,lines=lines+1,sequence_length=sequence_length,predict_result=predict_result,temp_file=temp_data_file)
        F= check_time_continuous(temp_data_file)
        if F:
            print(f'正在进行第 {lines+1} 步预测...')
            SP = StreamPredict(model_path=model_path,data_file=temp_data_file,standard_scalar=tandard_scalar)
            SP.load_model()
            prediction, time_list = SP.predict()
            time_list = pd.to_datetime(time_list, unit='s')
            predict_result=prediction[0,0]

            all_data.append([time_list[0],predict_result])
        else:
            print(f'第 {lines+1} 步特征数据有误，此步预测失败！预测结果默认为0...')
            predict_result=0
            lines+=1
    return all_data

def write_data(all_data,output_path):
    cloumns= ['datetime','pre']

    df = pd.DataFrame(all_data)
    df.to_csv(output_path,index=False,header=cloumns)


if __name__ == '__main__':
    all_data = main()
    write_data(all_data,config.data_predict_file+'/result.csv')
    target_data_df = pd.read_csv(config.data_predict_file+'/features.csv')
    target_data = target_data_df['雁翅5min洪水流量摘录'].tolist()
    plot_data.plot_data(all_data,target_data)