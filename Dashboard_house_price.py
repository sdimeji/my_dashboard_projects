# dropdown widget
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb

import plotly.express as px
from sklearn import linear_model,tree
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from IPython.display import display

st.set_page_config(layout='wide')

# read in your data
df=pd.read_csv("kaggle_london_house_price_data.csv")

#filling a null values using fillna()
for col in ['bathrooms','bedrooms','floorAreaSqM','livingRooms','rentEstimate_lowerPrice','rentEstimate_currentPrice','rentEstimate_upperPrice','saleEstimate_lowerPrice','saleEstimate_currentPrice','saleEstimate_upperPrice','saleEstimate_valueChange.numericChange','saleEstimate_valueChange.percentageChange','history_percentageChange','history_numericChange']:
    df[col] = df[col].fillna(df[col].mean())

for col in ['tenure','propertyType','currentEnergyRating','saleEstimate_confidenceLevel','saleEstimate_ingestedAt','saleEstimate_valueChange.saleDate']:
    df[col] = df[col].fillna('unknown')

# add title to your dashboard
st.title('London house price prediction dashboard')
st.header(':blue[This project is trying to predict house sales prices base on features of the house and its location using key variables in this dataset. This is a dataset of 100,000 sold house prices from 1995 -2024 in London]')

tab1, tab2 = st.tabs(["EDA", "ML"])

with tab1:
    # compute discreet values that would serve as the control for your widget
    propertyType_options = df['propertyType'].unique()
    currentEnergyRating_options = df['currentEnergyRating'].unique()

    # create a 3 column layout
    col1, col2 = st.columns([1, 1])

    # add dropdown widgets
    selected_propertyType = col1.selectbox(label='propertyType', options=propertyType_options)

    selected_currentEnergyRating = col2.selectbox(label='currentEnergyRating', options=currentEnergyRating_options)

    # filter your dataset based on widget selection
    df_plot = df[df['propertyType'] == selected_propertyType]
    df_plot = df_plot[df_plot['currentEnergyRating'] == selected_currentEnergyRating]

    fig = px.box(data_frame=df_plot, x="bedrooms", y="saleEstimate_currentPrice", boxmode='group', height=800,
                 title='Boxplot of history price and confidence level by tenure')
    fig.show()
    col1.plotly_chart(fig)

    fig = px.box(data_frame=df_plot, x="livingRooms", y="saleEstimate_currentPrice", boxmode='group', height=800,
                 title='Boxplot of history price and confidence level by tenure')
    fig.show()
    col2.plotly_chart(fig)

    fig = px.box(data_frame=df_plot, x="bathrooms", y="saleEstimate_currentPrice", boxmode='group', height=800,
                 title='Boxplot of history price and confidence level by tenure')
    fig.show()
    col1.plotly_chart(fig)

    fig = px.histogram(data_frame=df, x="propertyType", y="rentEstimate_currentPrice",
                       color="saleEstimate_confidenceLevel", barmode="group", marginal="box",
                       hover_data=df.columns, log_y=True,
                       title='Distribution of data of rent prices and property type by confidence level', width=1200,
                       height=500)
    fig.show()
    col2.plotly_chart(fig)

    plot = px.bar(data_frame=df, x='propertyType', y='saleEstimate_currentPrice', color='currentEnergyRating',
                  barmode='group', log_y=True, width=1000, height=500,
                  title='How significant is energy rating on current price of property type')
    plot.show()
    col1.plotly_chart(plot)

    fig = px.histogram(df, x="propertyType", y="saleEstimate_currentPrice", nbins=15, text_auto=True, log_y=True,
                       width=1000, height=800,
                       title='How different property types are distributed across sales current and history price')
    fig.show()
    col2.plotly_chart(fig)

    fig = px.histogram(df_plot, x="floorAreaSqM", y="saleEstimate_currentPrice", nbins=15, text_auto=True, log_y=True,
                       width=1000, height=800,
                       title='How different property types are distributed across sales current and history price')
    fig.show()
    col1.plotly_chart(fig)

    plot = px.scatter(data_frame=df, x='saleEstimate_currentPrice', y="rentEstimate_currentPrice", color="tenure",
                      title='Relationship of current rent and sales prices by confidence level', trendline="ols",
                      width=1000, height=500)
    plot.show()
    col2.plotly_chart(plot)

    fig = px.scatter_mapbox(data_frame=df_plot, lat="latitude", lon="longitude", color="propertyType", size="saleEstimate_currentPrice",color_continuous_scale=px.colors.cyclical.IceFire, size_max=15,mapbox_style='open-street-map',height=800, width=1000)
    fig.show()
    col1.plotly_chart(fig)

    # identify highly correlated features using a correlation matrix
    df3 = df.loc[:, ['saleEstimate_currentPrice', 'latitude', 'longitude', 'postcode', 'fullAddress', 'tenure',
                     'rentEstimate_upperPrice', 'history_price', 'rentEstimate_currentPrice',
                     'saleEstimate_confidenceLevel', 'saleEstimate_upperPrice', 'currentEnergyRating', 'propertyType']]
    corr = df3.corr(min_periods=5, numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True, annot=True)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right')
    st.pyplot(fig)

    # identify highly correlated features using a correlation matrix
    df4 = df.loc[:, ['bedrooms', 'bathrooms', 'floorAreaSqM', 'livingRooms', 'tenure', 'propertyType']]
    corr = df4.corr(min_periods=5, numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True, annot=True)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right')
    st.pyplot(fig)

    col1 = ['bedrooms', 'bathrooms', 'floorAreaSqM', 'livingRooms', 'tenure', 'propertyType']
    col2 = ['saleEstimate_currentPrice', 'latitude', 'longitude', 'postcode', 'fullAddress', 'tenure',
            'rentEstimate_upperPrice', 'history_price', 'rentEstimate_currentPrice', 'saleEstimate_confidenceLevel',
            'saleEstimate_upperPrice', 'currentEnergyRating', 'propertyType']

    sns.pairplot(df[col2], height=2.5)
    plt.tight_layout()
    plt.show()
    st.pyplot()

    sns.pairplot(df[col1], height=2.5)
    plt.tight_layout()
    plt.show()
    st.pyplot()
