import pandas as pd
import numpy as np
from fancyimpute import SimpleFill, KNN,  IterativeSVD, IterativeImputer

def impute_df(df, algorithm):
    """Returns completed dataframe given an imputation algorithm"""
    return pd.DataFrame(data=algorithm.fit_transform(df), columns=df.columns, index=df.index)


if __name__ == '__main__':
    train = pd.read_csv('./data/churn_train.csv', parse_dates=[5,7])

    # Need to add a Churn column as our target
    # Determine the current date to work with
    current_date  = train['last_trip_date'].max()
    train['days_away'] = (current_date - train['last_trip_date']).dt.days
    churn_threshold = 60
    train['Churn?'] = train['days_away'] >= churn_threshold

    # Imputate data
    # train = impute_df(train[['avg_rating_by_driver','avg_rating_of_driver']], KNN(k=5)) # only select desired columns then merge to original dataframe
    # train = KNN(k=5).fit_transform(train.drop(['signup_date','last_trip_date'], axis=1))
    rating_by_driver_mean = train['avg_rating_by_driver'].mean()
    rating_of_driver_mean = train['avg_rating_of_driver'].mean()
    train['avg_rating_by_driver'] = train['avg_rating_by_driver'].fillna(rating_by_driver_mean)
    train['avg_rating_of_driver'] = train['avg_rating_of_driver'].fillna(rating_of_driver_mean)

    # remove days_away column so we do not have a dependent feature
    train.drop('days_away', axis=1, inplace=True)

    # output to csv and overwrite file
    # train.to_csv('./data/churn_train.csv')
