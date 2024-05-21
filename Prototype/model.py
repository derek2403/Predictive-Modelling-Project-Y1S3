import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
import pickle

df = pd.read_csv("garments_worker_productivity.csv")

wip_mean = df['wip'].mean()
df['wip'] = df['wip'].fillna(wip_mean)

df['department'] = df['department'].str.strip()
df['department'] = df['department'].replace('sweing', 'sewing')
df['department'] = df['department'].replace('finishing ', 'finishing')

df["team"] = df["team"].astype("object")

categorical = df.select_dtypes(include='object')
numerical = df.select_dtypes(exclude='object')

def remove_outliers_z_score(df, col, threshold=3):
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    df_no_outliers = df[(np.abs(z_scores) <= threshold)]
    return df_no_outliers

for col in numerical.columns:
    df = remove_outliers_z_score(df, col)

cols_to_encode = ['quarter', 'department', 'day', 'team']
encoded_cols = pd.get_dummies(df[cols_to_encode], drop_first=True)
df.drop(cols_to_encode, axis=1, inplace=True)

cols_to_scale = df.drop(['targeted_productivity', 'actual_productivity', 'date'], axis=1).columns
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

new_df = pd.concat([encoded_cols, df], axis=1)
new_df = new_df.drop(columns=['date'])

X = new_df[['targeted_productivity', 'smv', 'over_time', 'incentive', 'no_of_workers']]
y = new_df['actual_productivity']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params = {'n_estimators': 300,
          'max_depth': 5,
          'min_samples_split': 10,
          'learning_rate': 0.01,
          'loss': 'ls',
          'subsample': 0.8,
          'max_features': 'sqrt'}

model1 = GradientBoostingRegressor(**params)
model2 = GradientBoostingRegressor(**params)
model3 = GradientBoostingRegressor(**params)

base_models = [('model1', model1), ('model2', model2), ('model3', model3)]
ensemble_model = VotingRegressor(estimators=base_models)

ensemble_model.fit(x_train, y_train)

ensemble_predictions = ensemble_model.predict(x_test)

pickle.dump(ensemble_model, open("model.pkl", "wb"))
