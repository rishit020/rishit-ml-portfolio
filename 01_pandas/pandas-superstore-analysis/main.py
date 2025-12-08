import pandas as pd
import numpy as np

data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/superstore.csv")

data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])

data['Order_Month'] = data['Order Date'].dt.month
data['Order_Quarter'] = data['Order Date'].dt.quarter
data['Order_Year'] = data['Order Date'].dt.year

first_purchase = data['Order Date'].min()
data['Days_Since_First'] = (data['Order Date'] - first_purchase).dt.days

sales_per_month = data.groupby(['Order_Year', 'Order_Month'])['Sales'].sum()
profit_per_category = data.groupby('Category')['Profit'].sum()
avg_discount_region = data.groupby('Region')['Discount'].mean()

customers = pd.DataFrame({
    'Customer ID': data['Customer ID'].unique(),
    'Customer_Segment_Custom': np.random.choice(['A','B','C'], size=data['Customer ID'].nunique())
})

merged = pd.merge(data, customers, on='Customer ID', how='left')

pivot_sales = merged.pivot_table(values='Sales', index='Category', columns='Region', aggfunc='sum')
pivot_profit = merged.pivot_table(values='Profit', index='Ship Mode', columns='Order_Month', aggfunc='sum')

top_products = data.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
top_states = data.groupby('State')['Order ID'].count().sort_values(ascending=False).head(5)
