# dropdown widget
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,RocCurveDisplay
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import joblib


st.set_page_config(layout='wide')

# read in your data
df = pd.read_csv('Lead Scoring.csv')
df.dropna(inplace=True)

# add title to your dashboard
st.title('Lead scoring dashboard')

tab1, tab2 = st.tabs(["EDA", "ML"])

with tab1:
    # compute discreet values that would serve as the control for your widget
    Lead_Source_options = df['Lead Source'].unique()
    Specialization_options = df['Specialization'].unique()

    # create a 2 column layout
    col1, col2 = st.columns([1, 1])

    # add dropdown widgets
    selected_Lead_Source = col1.selectbox(label='Lead Source', options=Lead_Source_options)

    selected_Specialization = col2.selectbox(label='Specialization', options=Specialization_options)

    # filter your dataset based on widget selection
    df_plot = df[df['Lead Source'] == selected_Lead_Source]
    df_plot = df_plot[df_plot['Specialization'] == selected_Specialization]

    df2 = df_plot.loc[:, ['Total Time Spent on Website', 'TotalVisits', 'Converted', 'Country']].groupby(
        ['Converted', 'Country']).sum().reset_index()
    # st.write(df2)

    plot = px.bar(data_frame=df_plot, x='Country', y='Total Time Spent on Website', color='Converted', barmode='group',
                  title='How time spent on website by country relates to converted clients', log_y=True)
    col1.plotly_chart(plot)

    # How total time spent is significantly related to lead quality by converted customers

    plot = px.box(data_frame=df_plot, x='Lead Quality', y='Total Time Spent on Website', color='Converted',
                  title='How time spent and lead quality leads to conversion')
    col2.plotly_chart(plot)

    # Customer conversion in relation to page vie and total visit of website

    plot = px.scatter(data_frame=df_plot, x='TotalVisits', y='Page Views Per Visit', color='Converted',
                      title='Customers who opted out of email service')
    col1.plotly_chart(plot)

    # Relationship between Occupation and page view
    plot = px.box(data_frame=df_plot, x='What is your current occupation', y='Page Views Per Visit', color='Converted',
                  title='How occupation influence page view in relation to convertion')
    col2.plotly_chart(plot)

    # Relationship between page view and lead source
    plot = px.box(data_frame=df_plot, x='Lead Source', y='Page Views Per Visit', color='Converted',
                  title='Lead source and page view by conversion')
    col1.plotly_chart(plot)

    # plot correlation matrix of variables
    st.title("Correlation matrix of variables :")
    corr = df_plot.corr(min_periods=5, numeric_only=True)
    # st.write(corr)
    fig,ax = plt.subplots(figsize=(6,6))
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

with tab2:
    # add dropdown widgets
    selected_ML = st.selectbox(label='Model', options=["Logistic", "Decision tree", "Random forest", "XGBoost"])
    # split data into test and train data
    y = df['Converted']
    x = df.loc[:, ['Lead Source', 'Do Not Email',
                   'Do Not Call', 'TotalVisits',
                   'Total Time Spent on Website', 'Page Views Per Visit',
                   'Country', 'Specialization']]
    X_enc = pd.get_dummies(x, dtype=int, columns=['Lead Source', 'Do Not Email',
                                                  'Do Not Call',
                                                  'Country', 'Specialization'])

    # Checking if the data is imbalanced or not
    #sum(y) / len(y)
    tr = y.value_counts(normalize=True)
    tr
    # 54% of the leads were converted. We need to make sure that we maintain the same % across both training and testing datasets
    # this kind of splitting the data maintaining the ratio is called "stratification"

    x_train, x_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.33, random_state=42, stratify=y)


    scaler=joblib.load('scaler.pkl')
    reg=joblib.load('model.pkl')

    X_transform = scaler.transform(x_train)
    x_test_trany = scaler.transform(x_test)

    predict_test = reg.predict(x_test_trany)
    predict_train = reg.predict(X_transform)

    if selected_ML == 'Logistic':
        # plot confussion matrix
        st.title("Plot confussion matrix :")
        confusion_matrix(y_test, predict_test)
        cm = confusion_matrix(y_test, predict_test, labels=reg.classes_)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=reg.classes_)
        disp.plot()
        plt.show()


        # Accuracy of logistics model test and train set
        col1, col2, col3, col4 = st.columns(4)
        accuracy_test = accuracy_score(y_test, reg.predict(x_test_trany))
        recall_test = recall_score(y_test, reg.predict(x_test_trany), average='weighted')
        precision_test = precision_score(y_test, reg.predict(x_test_trany), average='weighted')
        f1_test = f1_score(y_test, reg.predict(x_test_trany), average='weighted')



        col1.metric("accuracy test", np.round(accuracy_test, 2))
        col2.metric("recall test", np.round(recall_test, 2))
        col3.metric("precision test", np.round(precision_test, 2))
        col4.metric("F1 test", np.round(f1_test, 2))



        # Train set

        accuracy_train = accuracy_score(y_train, reg.predict(X_transform))
        recall_train = recall_score(y_train, reg.predict(X_transform), average='weighted')
        precision_train = precision_score(y_train, reg.predict(X_transform), average='weighted')
        f1_train = f1_score(y_train, reg.predict(X_transform), average='weighted')



        col1.metric("accuracy train", np.round(accuracy_train, 2))
        col2.metric("recall train", np.round(recall_train, 2))
        col3.metric("precision train", np.round(precision_train, 2))
        col4.metric("F1 train", np.round(f1_train, 2))
        st.pyplot()
        
        # Feature importance
        st.title("Feature importance :")
        #reg.coef_
        df_plot = pd.DataFrame({'coef': list(reg.coef_[0]), 'name': x_train.columns})
        y = (px.bar(data_frame=df_plot, x='coef', y='name', height=2000))
        st.plotly_chart(y)
        
    if selected_ML == "Decision tree":
        # Plot decision tree
        st.title("Plot decision tree :")
        clf = tree.DecisionTreeClassifier(max_depth=5)
        clf = clf.fit(X_transform, y_train)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
        fig.savefig('clf_individualtree.png')
        tree.plot_tree(clf)
        st.pyplot(fig)
        
        # feature importance
        st.title("feature importance DT :")
        #clf.feature_importances_
        df_plot = pd.DataFrame({'coef': list(clf.feature_importances_), 'name': x_train.columns})
        t = (px.bar(data_frame=df_plot[df_plot['coef'] > 0], x='coef', y='name', height=500))
        st.plotly_chart(t)
        
        Predict_clf = clf.predict(x_test_trany, check_input=True)
        #Predict_clf

        # decision tree confusion matrix
        st.title("Confusion matrix DT :")
        confusion_matrix(y_test, Predict_clf)
        cm = confusion_matrix(y_test, Predict_clf, labels=reg.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=reg.classes_)
        disp.plot()
        plt.show()
        st.pyplot()














