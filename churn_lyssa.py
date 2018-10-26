import numpy as np
import pandas as pd
from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer



def impute_df(df, algorithm):
    """Returns completed dataframe given an imputation algorithm"""
    return pd.DataFrame(data=algorithm.fit_transform(df), columns=df.columns, index=df.index)

def clean_data(nan_col_list,drop_col_list,file_name,churn_threshold=60):
    df = pd.read_csv('data/'+file_name,parse_dates=[5,7])
    current_date = df['last_trip_date'].max()
    df['days_since_last_trip'] = (current_date - df['last_trip_date']).dt.days
    df['days_since_signup'] = (current_date - df['signup_date']).dt.days
    df['phone'] = df['phone'].map({'iPhone':1,'Android':0})
    df['kings_land'] = df['city']=="King's Landing"
    df['astapor'] = df['city']=="Astapor"
    df['churn'] = df['days_since_last_trip']>= churn_threshold
    df.iloc[:,nan_col_list] = impute_df(df.iloc[:,nan_col_list], IterativeImputer())
    df.dropna(inplace=True)
    df.drop(df.columns[drop_col_list],axis=1, inplace= True)
    df.to_csv('clean_'+file_name,index=False)
    return df


if __name__ == '__main__':
    df_train = clean_data([1,2],[4,5,7,12],'churn_train.csv')
    df_test = clean_data([1,2],[4,5,7,12],'churn_test.csv')
