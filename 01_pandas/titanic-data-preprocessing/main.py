import seaborn as sns
import pandas as pd

data = sns.load_dataset("titanic")

data['age'] = data['age'].fillna(data['age'].mean())
data['embark_town'] = data['embark_town'].fillna(data['embark_town'].mode()[0])
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])
data.drop('deck', axis=1, inplace=True)

adults_data = data[data['age'] >= 18].reset_index(drop=True)

fare_over_mean = data[data['fare'] > data['fare'].mean()].reset_index(drop=True)

first_class_females = data[(data['sex'] == 'female') & (data['pclass'] == 1)].reset_index(drop=True)

data['family_size'] = data['sibsp'] + data['parch'] + 1

bins = [0, 12, 17, 64, 120]
labels = ['kids', 'teens', 'adults', 'seniors']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)

seniors_count = (data['age_group'] == 'seniors').sum()

surv_by_class = data.groupby('pclass')['survived'].mean()

stats_by_class = data.groupby('pclass')['survived'].agg(['mean', 'sum', 'count'])
stats_by_class = stats_by_class.rename(columns={'mean':'survival_rate','sum':'num_survived','count':'num_passengers'})

avg_fare_by_embark = data.groupby('embark_town')['fare'].mean()

fare_stats = data.groupby('embark_town')['fare'].agg(['mean','median','count'])
fare_stats = fare_stats.rename(columns={'mean':'avg_fare','median':'median_fare','count':'num_passengers'})
fare_stats['avg_fare'] = fare_stats['avg_fare'].round(2)

surv_by_sex_class = data.groupby(['sex','pclass'])['survived'].mean()
surv_table = surv_by_sex_class.unstack(level='pclass')

surv_table.plot(kind='bar')
