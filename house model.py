# dropdown widget
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,RANSACRegressor
from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,mean_absolute_percentage_error,max_error
from sklearn.datasets import make_friedman1
import plotly.express as px
from sklearn import linear_model,tree
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn import svm
from sklearn.svm import SVR
from IPython.display import display
import joblib


# read in your data
df=pd.read_csv("kaggle_london_house_price_data.csv")

#filling a null values using fillna()
for col in ['bathrooms','bedrooms','floorAreaSqM','livingRooms','rentEstimate_lowerPrice','rentEstimate_currentPrice','rentEstimate_upperPrice','saleEstimate_lowerPrice','saleEstimate_currentPrice','saleEstimate_upperPrice','saleEstimate_valueChange.numericChange','saleEstimate_valueChange.percentageChange']:
    df[col] = df[col].fillna(df[col].mean())

for col in ['tenure','propertyType','currentEnergyRating','saleEstimate_confidenceLevel','saleEstimate_valueChange.saleDate']:
    df[col] = df[col].fillna('unknown')



fig = px.box(data_frame=df, x="bedrooms", y="saleEstimate_currentPrice", boxmode='group', height=800,
             title='Boxplot of bedroom influencing sale price')
fig.show()


fig = px.box(data_frame=df, x="livingRooms", y="saleEstimate_currentPrice", boxmode='group', height=800,
             title='Boxplot of living room influencing sale price')
fig.show()


fig = px.box(data_frame=df, x="bathrooms", y="saleEstimate_currentPrice", boxmode='group', height=800,
             title='Boxplot of bathroom influencing sale price')
fig.show()


fig = px.histogram(data_frame=df, x="propertyType", y="rentEstimate_currentPrice",
                   color="saleEstimate_confidenceLevel", barmode="group", marginal="box",
                   hover_data=df.columns, log_y=True,
                   title='Property type influencing sale price by confidence level', width=1200,
                   height=500)
fig.show()


plot = px.bar(data_frame=df, x='propertyType', y='saleEstimate_currentPrice', color='currentEnergyRating',
              barmode='group', log_y=True, width=1000, height=500,
              title='Property type influencing sale price by energy rating')
plot.show()


fig = px.histogram(df, x="propertyType", y="saleEstimate_currentPrice", nbins=15, text_auto=True, log_y=True,
                   width=1000, height=800,
                   title='Distribution of sale price across property type')
fig.show()


fig = px.histogram(df, x="floorAreaSqM", y="saleEstimate_currentPrice", nbins=15, text_auto=True, log_y=True,
                   width=1000, height=800,
                   title='Distribution of sale price across floor area sqmtr')
fig.show()


plot = px.scatter(data_frame=df, x='saleEstimate_currentPrice', y="rentEstimate_currentPrice", color="tenure",
                  title='How significant is the relationship between rent and sales prices by tenure', trendline="ols",
                  width=1000, height=500)
plot.show()


fig = px.scatter_mapbox(data_frame=df, lat="latitude", lon="longitude", color="propertyType", size="saleEstimate_currentPrice",color_continuous_scale=px.colors.cyclical.IceFire, size_max=15,mapbox_style='open-street-map',height=800, width=1200,title='Sales price distribution by property type across various location')
fig.show()


st.header('identify highly correlated features using a correlation matrix')
df3 = df.loc[:, ['saleEstimate_currentPrice', 'latitude', 'longitude', 'tenure',
                 'rentEstimate_upperPrice', 'history_price', 'rentEstimate_currentPrice',
                 'saleEstimate_confidenceLevel', 'saleEstimate_upperPrice', 'currentEnergyRating', 'propertyType']]
corr = df3.corr(min_periods=5, numeric_only=True)
fig, ax = plt.subplots(figsize=(6, 6))
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(10, 110, n=200),
    square=True, annot=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
st.pyplot(fig)

st.header('identify highly correlated features using a correlation matrix')
df4 = df.loc[:,['saleEstimate_currentPrice', 'saleEstimate_lowerPrice', 'rentEstimate_lowerPrice', 'bedrooms', 'bathrooms',
       'floorAreaSqM', 'livingRooms', 'tenure']]

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
col2 = ['saleEstimate_currentPrice', 'latitude', 'longitude', 'tenure',
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

# split data into test and train data
y = df['saleEstimate_currentPrice']
x = df.loc[:,
    ['bathrooms', 'floorAreaSqM', 'saleEstimate_lowerPrice', 'rentEstimate_lowerPrice', 'rentEstimate_currentPrice',
     'propertyType', 'rentEstimate_upperPrice', 'saleEstimate_upperPrice', 'livingRooms', 'history_price',
     'currentEnergyRating']]
X_enc = pd.get_dummies(x, dtype=int, dummy_na='unknown', columns=['currentEnergyRating', 'propertyType'])
x_train, x_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.33, random_state=20)

# use robust scaler to control outliers
rob_trans = RobustScaler()
X_trans = rob_trans.fit_transform(x_train)
joblib.dump(rob_trans, 'RobustScaler.pkl')


X_test = rob_trans.transform(x_test)


st.header("Plot random forest")
clf = RandomForestRegressor(n_estimators=250, max_depth=5, max_features=1.0, criterion='squared_error')
clf.fit(X_trans, y_train)
joblib.dump(clf, 'House RFmodel.pkl')

y_train_pred = clf.predict(X_trans)
y_test_pred = clf.predict(X_test)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)

tree.plot_tree(clf.estimators_[5], max_depth=10,
                   feature_names=None,
                   class_names=None,
                   filled=True, fontsize=1);fig.savefig('clf_individualtree.png')
st.pyplot(fig)

#Plot feature importance
st.header("Feature importance RF")
df_plot = pd.DataFrame({'coef': list(clf.feature_importances_), 'name': x_train.columns})
r=px.bar(data_frame=df_plot[df_plot['coef'] > 0], x='coef', y='name', height=1000)
st.plotly_chart(r)

st.header("Metrics RF")
col1, col2, col3, col4,col5,col6,col7,col8 = st.columns(8)
R2sq_train=r2_score(y_train, y_train_pred)
R2sq_test=r2_score(y_test, y_test_pred)
mean_abs_test=mean_absolute_error(y_test, y_test_pred)
mean_abs_testpercen=mean_absolute_percentage_error(y_test, y_test_pred)
mean_abs_train=mean_absolute_error(y_train, y_train_pred)
mean_abs_trainpercen=mean_absolute_percentage_error(y_train, y_train_pred)
mean_sqtest=mean_squared_error(y_test, y_test_pred)
mean_sqtrain=mean_squared_error(y_train, y_train_pred)

col1.metric("R2 value train", np.round(R2sq_train, 3))
col2.metric("R2 value test", np.round(R2sq_test, 3))
col3.metric("mean absolute error test", np.round(mean_abs_test, 2))
col4.metric("Mean absolute error test %", np.round(mean_abs_testpercen, 3))
col5.metric("mean absolute error train", np.round(mean_abs_train, 2))
col6.metric("Mean absolute error train %", np.round(mean_abs_trainpercen, 3))
col7.metric("mean sq error test", np.round(mean_sqtest, 2))
col8.metric("Mean sq error train", np.round(mean_sqtrain, 2))

# plot of actual and predicted value
st.header("RF regression chart ")
x_pred = clf.predict(X_test)
rand_plot = px.scatter(x=x_pred, y=y_test, trendline='ols',
                       title='Random forest regression prediction vs actual value', height=800)
st.plotly_chart(rand_plot)


st.header("Plot Gradient Boosting")
reg = GradientBoostingRegressor(random_state=0, n_estimators=250, max_depth=10)
reg.fit(X_trans, y_train)
joblib.dump(reg, 'HouseGdBmodel.pkl')

st.header("Feature importance Gradient Boosting")

df_plot = pd.DataFrame({'coef': list(reg.feature_importances_), 'name': x_train.columns})
b=px.bar(data_frame=df_plot[df_plot['coef'] > 0], x='coef', y='name', height=800)
st.plotly_chart(b)

st.header("Metrics Gradient Boosting")
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
R2sq_traingd = r2_score(y_train, reg.predict(X_trans))
R2sq_testgd = r2_score(y_test, reg.predict(X_test))
mean_abs_testgd = mean_absolute_error(y_test, reg.predict(X_test))
mean_abs_testpercengd = mean_absolute_percentage_error(y_test, reg.predict(X_test))
mean_abs_traingd = mean_absolute_error(y_train, reg.predict(X_trans))
mean_abs_trainpercengd = mean_absolute_percentage_error(y_train, reg.predict(X_trans))
mean_sqtestgd = mean_squared_error(y_test, reg.predict(X_test))
mean_sqtraingd = mean_squared_error(y_train, reg.predict(X_trans))

col1.metric("R2 train", np.round(R2sq_traingd, 3))
col2.metric("R2 test", np.round(R2sq_testgd, 3))
col3.metric("mae test", np.round(mean_abs_testgd, 2))
col4.metric("Mae test %", np.round(mean_abs_testpercengd, 3))
col5.metric("mae train", np.round(mean_abs_traingd, 2))
col6.metric("Mae train %", np.round(mean_abs_trainpercengd, 3))
col7.metric("m sq error test", np.round(mean_sqtestgd, 2))
col8.metric("M sq error train", np.round(mean_sqtraingd, 2))

x_pred = reg.predict(X_test)
Gd_plot = px.scatter(x=x_pred, y=y_test, trendline='ols',
                     title='Gradient boosting regression prediction vs actual value', height=800)
st.plotly_chart(Gd_plot)