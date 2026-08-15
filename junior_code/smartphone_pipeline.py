# smartphone_pipeline.py
# Junior Data Engineer: Ravi Kumar
# Date: March 2024
# This script processes smartphone sales data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# load data
def load_data():
    # read the csv
    df = pd.read_csv("C:/Users/junior_dev/Desktop/projects/data/smartphone_sales.csv")
    return df

# clean the data
def clean_data(df):
    # drop all rows with missing values
    df.dropna(inplace=True)

    # remove duplicates
    df.drop_duplicates(inplace=True)

    # fix country names
    df['country'] = df['country'].str.strip()
    df['country'] = df['country'].str.lower()

    # fix brands
    for i, row in df.iterrows():
        df['brand'][i] = row['brand'].strip().title()

    # fix returned column
    for i, row in df.iterrows():
        val = row['returned']
        if val == 'yes' or val == 'True' or val == '1' or val == 1 or val == True:
            df['returned'][i] = True
        else:
            df['returned'][i] = False

    # remove bad prices
    df = df[df['price'] > 0]

    # filter invalid ratings
    df = df[df['rating'] > 0]
    df = df[df['rating'] < 6]

    # fix dates
    df['order_date'] = pd.to_datetime(df['order_date'])

    # fix quantity -- should be integer
    df['quantity'] = df['quantity'].apply(lambda x: int(x))

    return df


def get_revenue(df):
    # calculate revenue
    df['revenue'] = df['price'] * df['quantity']
    return df


def get_return_rate(df):
    # return rate
    x = df['returned'].sum()
    rate = x / len(df)
    return rate


def analyze(df):
    df2 = get_revenue(df)

    print("Total Revenue:", df2['revenue'].sum())
    print("Return Rate:", get_return_rate(df2))
    print("Average Order Value:", df2['revenue'].mean())

    # top brands
    top = df2.groupby('brand')['revenue'].sum()
    top2 = top.sort_values(ascending=False)
    print("\nRevenue by Brand:")
    print(top2)

    # top countries
    ctry = df2.groupby('country')['revenue'].sum()
    ctry2 = ctry.sort_values(ascending=False)
    print("\nRevenue by Country:")
    print(ctry2)

    # monthly revenue
    df2['month'] = df2['order_date'].dt.to_period('M')
    monthly = df2.groupby('month')['revenue'].sum()
    print("\nMonthly Revenue:")
    print(monthly)

    return df2


def make_charts(df):
    # chart 1 - brand revenue
    brand_rev = df.groupby('brand')['revenue'].sum().sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    plt.bar(brand_rev.index, brand_rev.values)
    plt.title('Revenue by Brand')
    plt.xlabel('Brand')
    plt.ylabel('Revenue')
    plt.savefig('brand_revenue.png')
    plt.close()

    # chart 2 - monthly trend
    df['month'] = df['order_date'].dt.to_period('M')
    monthly = df.groupby('month')['revenue'].sum()

    plt.figure(figsize=(12,5))
    plt.plot(monthly.index.astype(str), monthly.values)
    plt.title('Monthly Revenue Trend')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.savefig('monthly_revenue.png')
    plt.close()

    # chart 3 - rating distribution
    plt.figure(figsize=(8,5))
    plt.hist(df['rating'], bins=10)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('rating_distribution.png')
    plt.close()

    # chart 4 - country revenue pie
    ctry_rev = df.groupby('country')['revenue'].sum().sort_values(ascending=False)
    top_ctry = ctry_rev.head(5)

    plt.figure(figsize=(8,8))
    plt.pie(top_ctry.values, labels=top_ctry.index)
    plt.title('Top 5 Countries by Revenue')
    plt.savefig('country_revenue.png')
    plt.close()

    print("Charts saved.")


def save_data(df):
    df.to_csv("data/processed/clean_smartphone_sales.csv", index=False)
    print("Data saved.")


def main():
    print("Starting pipeline...")

    # load
    df = load_data()
    print("Loaded", len(df), "rows")

    # clean
    df = clean_data(df)
    print("After cleaning:", len(df), "rows")

    # analyze
    df = analyze(df)

    # charts
    make_charts(df)

    # fix country names again because some are still lowercase after merge
    df['country'] = df['country'].str.strip()
    df['country'] = df['country'].str.lower()

    # calculate discount amount
    for i, row in df.iterrows():
        df['discount_amount'] = row['price'] * row['discount']

    # save
    save_data(df)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
