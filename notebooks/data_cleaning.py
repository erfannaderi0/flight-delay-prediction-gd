import pandas as pd

file_path = 'data/raw/flight_delay_2019_2023.csv'

df = pd.read_csv(file_path)

df = df.drop_duplicates()

#AIRLINE_DOT and DOT_CODE are the same as AIRLINE and AIRLINE CODE so we dont need these columns
drop_cols = ['AIRLINE_DOT', 'DOT_CODE']
df = df.drop(columns=drop_cols, errors='ignore')

#correcting dtypes
#converting datatypes to the right ones: FL_DATE: object->datetime
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
#changing the dtypes of the time variables to numeric time just in case if it was not and not getting any problem in dropna part
numeric_cols = ['DEP_DELAY', 'ARR_DELAY', 'TAXI_OUT', 'TAXI_IN',
                'CRS_ELAPSED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'DISTANCE',
                'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
                'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

#filling NAN delay causes with 0 to be able to use these columns later
delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
              'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']
df[delay_cols] = df[delay_cols].fillna(0)

#dropping row that were not cancelled but did not have time either
mask = (df['CANCELLED'] == 0) & (df['DEP_TIME'].isna() | df['ARR_TIME'].isna())
df = df[~mask]

#handelling cancellation and making it easier to appear
cancel_map = {'A': 'Carrier', 'B': 'Weather', 'C': 'NAS', 'D': 'Security'}
df['CANCELLATION_CODE'] = df['CANCELLATION_CODE'].map(cancel_map).fillna('None')

#diverted flights dont have key features so i am going to ignore them
df = df[df['DIVERTED'] == 0]

#filtering arrival delays that are not logicly right or extremely noisy and obviously flight distance should be more than 0
df = df[(df['ARR_DELAY'].between(-60, 1440)) | (df['ARR_DELAY'].isna())]
df = df[df['DISTANCE'] > 0]

# === Feature engineering ===
delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']
df['TOTAL_DELAY_CAUSE'] = df[delay_cols].sum(axis=1)
df['IS_DELAYED'] = (df['ARR_DELAY'] > 15).astype(int)
df['MONTH'] = df['FL_DATE'].dt.month
df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)

# === Text cleanup ===
df['AIRLINE'] = df['AIRLINE'].str.strip().str.title()
df['ORIGIN_CITY'] = df['ORIGIN_CITY'].str.strip().str.title()
df['DEST_CITY'] = df['DEST_CITY'].str.strip().str.title()
df['AIRLINE_CODE'] = df['AIRLINE_CODE'].str.strip().str.upper()
df['ORIGIN'] = df['ORIGIN'].str.strip().str.upper()
df['DEST'] = df['DEST'].str.strip().str.upper()

#dont need DIVERTED column anymore because we already kept only flights that did not diverted(line 39)
df = df.drop(columns=['DIVERTED'], errors='ignore')

df = df.reset_index(drop=True)
print("Cleaned shape:", df.shape)

save_path = 'data/processed/flight_delay_cleaned.csv'
df.to_csv(save_path, index=False)
print(f"Cleaned data saved to {save_path}")
