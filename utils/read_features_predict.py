import pandas as pd
import config

def creat_temp_data(features_data_file:str,lines:int,sequence_length:int,predict_result:float,temp_file:str,target_row:int = -2,target_col:int = 4)->None:

    df1 = pd.read_csv(features_data_file)

    cloumns = df1.columns.tolist()

    df = pd.read_csv(features_data_file, skiprows=lines-1, nrows = sequence_length+1)
    if lines !=1:
        df.iloc[target_row,target_col]=predict_result
    df.to_csv(temp_file, index=None, header=cloumns)
    pass
if __name__ == '__main__':
    creat_temp_data(config.data_predict_file+'/features.csv',2,config.sequence_length,50,config.data_predict_file+'/temp.csv')