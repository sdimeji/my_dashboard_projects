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
from IPython.display import display
import h2o
import joblib


st.set_page_config(layout='wide')

# read in your data
df = pd.read_csv('Lead Scoring.csv')
df.dropna(inplace=True)

# add title to your dashboard
st.title('Lead scoring dashboard')
st.header(':blue[This project implemented a model that can determine if a lead will be converted or not, using key variables, this model will save resources and time for American business, imagine a situation where a team can use least time to achieve higher revenue]')

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
    st.text("This chart explain time spent on company website by each country translates to conversion, you can interact with the drop down by changing specialization and lead source to get different results")

    # How total time spent is significantly related to lead quality by converted customers

    plot = px.box(data_frame=df_plot, x='Lead Quality', y='Total Time Spent on Website', color='Converted',
                  title='How time spent and lead quality leads to conversion')
    col2.plotly_chart(plot)
    st.text("This boxplot explains how different category of lead quality spend time on company website and translates to conversion, you can interact with the drop down by changing specialization and lead source to get different results")

    # Customer conversion in relation to page view and total visit of website

    plot = px.scatter(data_frame=df_plot, x='TotalVisits', y='Page Views Per Visit', color='Converted',
                      title='How page views and website visits relates to conversion')
    st.text("This scatter plot explains the trend in data point and relationship between total visit and time spent on company website translates to conversion, I am trying to visualize cluster point and possible outliers in both variables, you can interact with the drop down by changing specialization and lead source to get different results")
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
    st.header('Correlation matrix of variables')
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
    st.text('This chart explains relationship between each pair of variables in our dataset, strong positive correlation are close to 1 and negative -1, while no significant correlation are near 0, with this chart we could identify relevant features for our model')


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
    st.header("How balanced is our data")
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
        st.header("Plot confussion matrix")
        confusion_matrix(y_test, predict_test)
        cm = confusion_matrix(y_test, predict_test, labels=reg.classes_)
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=reg.classes_)
        disp.plot(ax=ax)  # Pass the axis object
        disp.plot()
        plt.show()
        st.pyplot(fig)
        
        st.text("True Negative (TN) 228: These are the cases where the model correctly predicted the negative class (no conversion when the actual outcome was also no conversion)")
        st.text("True Positive (TP) 269: These are the cases where the model correctly predicted the positive class (conversion when the actual outcome was conversion)")
        st.text("False Positive (FP)  70: These are the cases where the model incorrectly predicted the positive class (conversion when the actual outcome was no conversion)")
        st.text("False Negative (FN) 75: These are the cases where the model incorrectly predicted the negative class (no conversion when the actual outcome was conversion)")


        # Accuracy of logistics model test and train set
        st.header("Accuracy Metrics")
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
        st.text("Accuracy: This is the proportion of correct predictions (both TP and TN) over the total predictions")
        st.text("Precision: This measures the accuracy of positive predictions. It tells us how many of the predicted positives were positive")
        st.text("Recall (Sensitivity): This measures how many actual positives the model correctly identified")
        st.text("F1-Score: This is the harmonic mean of precision and recall. It balances both, giving an overall measure of the model's performance.")
        st.text("The metrics above indicate this model performs well and has an accuracy of 77%, but there can be room for further improvement in minimizing false positives and false negatives. The precision and recall show that the model does a decent job of predicting positive outcomes.")
        
        
        # Feature importance
        st.header("Feature importance")
        #reg.coef_
        df_plot = pd.DataFrame({'coef': list(reg.coef_[0]), 'name': x_train.columns})
        y = (px.bar(data_frame=df_plot, x='coef', y='name', height=2000))
        st.plotly_chart(y)
        st.text('This chart explains how each feature significantly contribute to our model final prediction, the longer the bar the higher the importance, bars in negative has no significant contribution')
    if selected_ML == "Decision tree":
        # Plot decision tree
        st.header("Plot decision tree")
        clf = tree.DecisionTreeClassifier(max_depth=5)
        clf = clf.fit(X_transform, y_train)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
        fig.savefig('clf_individualtree.png')
        tree.plot_tree(clf)
        st.pyplot(fig)
        
        # feature importance
        st.header("feature importance DT")
        #clf.feature_importances_
        df_plot = pd.DataFrame({'coef': list(clf.feature_importances_), 'name': x_train.columns})
        t = (px.bar(data_frame=df_plot[df_plot['coef'] > 0], x='coef', y='name', height=500))
        st.plotly_chart(t)
        st.text('This chart explains how each feature significantly contribute to our model final prediction, the longer the bar the higher the importance')
        
        Predict_clf = clf.predict(x_test_trany, check_input=True)
        #Predict_clf

        # decision tree confusion matrix
        st.header("Confusion matrix DT")
        confusion_matrix(y_test, Predict_clf)
        cm = confusion_matrix(y_test, Predict_clf, labels=reg.classes_)
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=reg.classes_)
        disp.plot(ax=ax)  # Pass the axis object
        disp.plot()
        plt.show()
        st.pyplot(fig)
        st.text("True Negative (TN) 234: These are the cases where the model correctly predicted the negative class (no conversion when the actual outcome was also no conversion)")
        st.text("True Positive (TP) 264: These are the cases where the model correctly predicted the positive class (conversion when the actual outcome was conversion)")
        st.text("False Positive (FP) 64: These are the cases where the model incorrectly predicted the positive class (conversion when the actual outcome was no conversion)")
        st.text("False Negative (FN) 80: These are the cases where the model incorrectly predicted the negative class (no conversion when the actual outcome was conversion)")

        
        # Metrics
        st.header("Accuracy Metrics DT")
        col1, col2, col3, col4 = st.columns(4)
        accuracy_test = accuracy_score(y_test, clf.predict(x_test_trany))
        recall_test = recall_score(y_test, clf.predict(x_test_trany), average='weighted')

        precision_test = precision_score(y_test, clf.predict(x_test_trany), average='weighted')

        f1_test = f1_score(y_test, clf.predict(x_test_trany), average='weighted')


        col1.metric("accuracy test", np.round(accuracy_test, 2))
        col2.metric("recall test", np.round(recall_test, 2))
        col3.metric("precision test", np.round(precision_test, 2))
        col4.metric("F1 test", np.round(f1_test, 2))
        
        # train
        accuracy_train = accuracy_score(y_train, clf.predict(X_transform))
        recall_train = recall_score(y_train, clf.predict(X_transform), average='weighted')

        precision_train = precision_score(y_train, clf.predict(X_transform), average='weighted')

        f1_train = f1_score(y_train, clf.predict(X_transform), average='weighted')

        col1.metric("accuracy train", np.round(accuracy_train, 2))
        col2.metric("recall train", np.round(recall_train, 2))
        col3.metric("precision train", np.round(precision_train, 2))
        col4.metric("F1 train", np.round(f1_train, 2))
        st.text("Accuracy: This is the proportion of correct predictions (both TP and TN) over the total predictions")
        st.text("Precision: This measures the accuracy of positive predictions. It tells us how many of the predicted positives were positive")
        st.text("Recall (Sensitivity): This measures how many actual positives the model correctly identified")
        st.text("F1-Score: This is the harmonic mean of precision and recall. It balances both, giving an overall measure of the model's performance.")
        st.text("The metrics above indicate this model performs well and has an accuracy of 78%, but there can be room for further improvement in minimizing false positives and false negatives. The precision and recall show that the model does a decent job of predicting positive outcomes.")
        
       

    if selected_ML == "Random forest":
        # plot random forest
        st.header("Plot Random forest")
        clf1 = RandomForestClassifier(n_estimators=50, max_depth=5)
        clf1 = clf1.fit(X_transform, y_train)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
        tree.plot_tree(clf1.estimators_[5],
                       feature_names=None,
                       class_names=None,
                       filled=True);
        fig.savefig('clf1_individualtree.png')
        st.pyplot(fig)
        st.text('This ensemble machine learning method is explaining visualization of part of our multiple decision trees as output of my training data')

        # feature importance
        st.header("Feature importance RF")
        #clf1.feature_importances_
        df_plot = pd.DataFrame({'coef': list(clf1.feature_importances_), 'name': x_train.columns})
        r = (px.bar(data_frame=df_plot[df_plot['coef'] > 0], x='coef', y='name', height=500))
        st.plotly_chart(r)
        st.text('This chart explains how each feature significantly contribute to our model final prediction, the longer the bar the higher the importance')

        Predict_clf1 = clf1.predict(x_test_trany)

        # plot confusion matrix
        st.header("Confusion matrix RF :")
        confusion_matrix(y_test, Predict_clf1)
        cm = confusion_matrix(y_test, Predict_clf1, labels=reg.classes_)
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=reg.classes_)
        disp.plot(ax=ax)  # Pass the axis object
        disp.plot()
        plt.show()
        st.pyplot(fig)
        st.text("True Negative (TN) 232: These are the cases where the model correctly predicted the negative class (no conversion when the actual outcome was also no conversion)")
        st.text("True Positive (TP) 269: These are the cases where the model correctly predicted the positive class (conversion when the actual outcome was conversion)")
        st.text("False Positive (FP) 66: These are the cases where the model incorrectly predicted the positive class (conversion when the actual outcome was no conversion)")
        st.text("False Negative (FN) 75: These are the cases where the model incorrectly predicted the negative class (no conversion when the actual outcome was conversion)")



        st.header("Accuracy Metrics RF")
        col1, col2, col3, col4 = st.columns(4)
        accuracy_test = accuracy_score(y_test, clf1.predict(x_test_trany))
        recall_test = recall_score(y_test, clf1.predict(x_test_trany), average='weighted')
        precision_test = precision_score(y_test, clf1.predict(x_test_trany), average='weighted')
        f1_test = f1_score(y_test, clf1.predict(x_test_trany), average='weighted')

        col1.metric("accuracy test", np.round(accuracy_test, 2))
        col2.metric("recall test", np.round(recall_test, 2))
        col3.metric("precision test", np.round(precision_test, 2))
        col4.metric("F1 test", np.round(f1_test, 2))
        

        # train
        accuracy_train = accuracy_score(y_train, clf1.predict(X_transform))
        recall_train = recall_score(y_train, clf1.predict(X_transform), average='weighted')

        precision_train = precision_score(y_train, clf1.predict(X_transform), average='weighted')

        f1_train = f1_score(y_train, clf1.predict(X_transform), average='weighted')

        col1.metric("accuracy train", np.round(accuracy_train, 2))
        col2.metric("recall train", np.round(recall_train, 2))
        col3.metric("precision train", np.round(precision_train, 2))
        col4.metric("F1 train", np.round(f1_train, 2))
        st.text("Accuracy: This is the proportion of correct predictions (both TP and TN) over the total predictions")
        st.text("Precision: This measures the accuracy of positive predictions. It tells us how many of the predicted positives were positive")
        st.text("Recall (Sensitivity): This measures how many actual positives the model correctly identified")
        st.text("F1-Score: This is the harmonic mean of precision and recall. It balances both, giving an overall measure of the model's performance.")
        st.text("The metrics above indicate this model performs well and has an accuracy of 78%, but there can be room for further improvement in minimizing false positives and false negatives. The precision and recall show that the model does a decent job of predicting positive outcomes.")
        
        
    if selected_ML == "XGBoost":
        # Plot XGBoost model
        st.header("Plot XGBoost model")
        GBx = xgb.XGBClassifier(n_estimators=5, max_depth=5)
        GBx.fit(X_transform, y_train)
       

        # feature importance
        st.header("feature importance")
        #GBx.feature_importances_
        df_plot = pd.DataFrame({'coef': list(GBx.feature_importances_), 'name': x_train.columns})
        x = (px.bar(data_frame=df_plot[df_plot['coef'] > 0], x='coef', y='name', height=500))
        st.plotly_chart(x)
        st.text('This chart explains how each feature significantly contribute to our model final prediction, the longer the bar the higher the importance')

        # plot confusion matrix
        st.header("confusion matrix")
        Predict_GBx = GBx.predict(x_test_trany)

        confusion_matrix(y_test, Predict_GBx)
        cm = confusion_matrix(y_test, Predict_GBx, labels=reg.classes_)
        fig, ax = plt.subplots(figsize=(1.5, 1.5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=reg.classes_)
        disp.plot(ax=ax)  # Pass the axis object
        disp.plot()
        plt.show()
        st.pyplot(fig)

        st.header("Accuracy Metrics")
        col1, col2, col3, col4 = st.columns(4)
        accuracy_test = accuracy_score(y_test, GBx.predict(x_test_trany))
        recall_test = recall_score(y_test, GBx.predict(x_test_trany), average='weighted')
        precision_test = precision_score(y_test, GBx.predict(x_test_trany), average='weighted')
        f1_test = f1_score(y_test, GBx.predict(x_test_trany), average='weighted')

        col1.metric("accuracy test", np.round(accuracy_test, 2))
        col2.metric("recall test", np.round(recall_test, 2))
        col3.metric("precision test", np.round(precision_test, 2))
        col4.metric("F1 test", np.round(f1_test, 2))
        

        # Train
        accuracy_train = accuracy_score(y_train, GBx.predict(X_transform))
        recall_train = recall_score(y_train, GBx.predict(X_transform), average='weighted')
        precision_train = precision_score(y_train, GBx.predict(X_transform), average='weighted')
        f1_train = f1_score(y_train, GBx.predict(X_transform), average='weighted')

        col1.metric("accuracy train", np.round(accuracy_train, 2))
        col2.metric("recall train", np.round(recall_train, 2))
        col3.metric("precision train", np.round(precision_train, 2))
        col4.metric("F1 train", np.round(f1_train, 2))
        st.text("Accuracy: This is the proportion of correct predictions (both TP and TN) over the total predictions")
        st.text("Precision: This measures the accuracy of positive predictions. It tells us how many of the predicted positives were positive")
        st.text("Recall (Sensitivity): This measures how many actual positives the model correctly identified")
        st.text("F1-Score: This is the harmonic mean of precision and recall. It balances both, giving an overall measure of the model's performance.")
        st.text("The metrics above indicate this model performs well and has an accuracy of 77%, but there can be room for further improvement in minimizing false positives and false negatives. The precision and recall show that the model does a decent job of predicting positive outcomes.")
        
        
    

       














