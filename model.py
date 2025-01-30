# dropdown widget
import StandardScaler
import streamlit as st
import pandas as pd
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
from h2o.automl import H2OAutoML
h2o.init()
import joblib



st.set_page_config(layout='wide')

# read in your data
df = pd.read_csv('Lead Scoring.csv')
df.dropna(inplace=True)


#split data into test and train data
y=df['Converted']
x=df.loc[:,[ 'Lead Source','Do Not Email',
       'Do Not Call','TotalVisits',
       'Total Time Spent on Website', 'Page Views Per Visit',
       'Country', 'Specialization']]
X_enc = pd.get_dummies(x,dtype=int,columns=[ 'Lead Source','Do Not Email',
       'Do Not Call',
       'Country', 'Specialization'])

# Checking if the data is imbalanced or not
sum(y)/ len(y)
tr=y.value_counts(normalize=True)
tr
# 54% of the leads were converted. We need to make sure that we maintain the same % across both training and testing datasets
# this kind of splitting the data maintaining the ratio is called "stratification"


x_train,x_test,y_train,y_test=train_test_split(X_enc,y,test_size=0.33,random_state=42, stratify=y)

#perform standardization and normalization of data
scaler=StandardScaler()
scaler.fit(x_train)
joblib.dump(scaler, 'scaler.pkl')

X_transform=scaler.transform(x_train)
x_test_trany=scaler.transform(x_test)

#Train logistic model
reg = linear_model.LogisticRegression()
reg.fit(X_transform,y_train)
joblib.dump(reg, 'model.pkl')

predict_test=reg.predict(x_test_trany)
predict_train=reg.predict(X_transform)

#plot confussion matrix
confusion_matrix(y_test, predict_test)
cm = confusion_matrix(y_test, predict_test, labels=reg.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=reg.classes_)
disp.plot()
plt.show()
st.pyplot()

#Accuracy of logistics model test and train set
accuracy_test=accuracy_score(y_test, reg.predict(x_test_trany))
recall_test=recall_score(y_test, reg.predict(x_test_trany),average='weighted')
precision_test=precision_score(y_test, reg.predict(x_test_trany),average='weighted')
f1_test=f1_score(y_test, reg.predict(x_test_trany),average='weighted')

data={"Test": ["precision", "recall", "accuracy", "f1"], "Metric": [precision_test,recall_test,accuracy_test,f1_test]}
# Create DataFrame
dr = pd.DataFrame(data)

# Display the DataFrame
dr

#Train set
accuracy_train=accuracy_score(y_train, reg.predict(X_transform))
recall_train=recall_score(y_train, reg.predict(X_transform),average='weighted')
precision_train=precision_score(y_train, reg.predict(X_transform),average='weighted')
f1_train=f1_score(y_train, reg.predict(X_transform),average='weighted')

data = { "Train": ["precision", "recall", "accuracy", "f1"],
    "Metric": [precision_train, recall_train, accuracy_train,f1_train]}

# Create DataFrame
dr = pd.DataFrame(data)

# Display the DataFrame
dr


#Feature importance
reg.coef_
df_plot = pd.DataFrame({'coef':list(reg.coef_[0]),'name':x_train.columns})
y=(px.bar(data_frame=df_plot, x='coef',y='name',height=2000))
st.plotly_chart(y)




#Plot decision tree
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_transform, y_train)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
fig.savefig('clf_individualtree.png')
tree.plot_tree(clf)
st.pyplot(fig)

#feature importance
clf.feature_importances_
df_plot = pd.DataFrame({'coef':list(clf.feature_importances_),'name':x_train.columns})
t=(px.bar(data_frame=df_plot[df_plot['coef'] > 0], x='coef',y='name',height=2000))
st.plotly_chart(t)

Predict_clf=clf.predict(x_test_trany,check_input=True)
Predict_clf

#decision tree confusion matrix
confusion_matrix(y_test, Predict_clf)
cm = confusion_matrix(y_test, Predict_clf, labels=reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=reg.classes_)
disp.plot()
plt.show()
st.pyplot()

#Metrics
accuracy_test=accuracy_score(y_test, clf.predict(x_test_trany))
recall_test=recall_score(y_test, clf.predict(x_test_trany),average='weighted')

precision_test=precision_score(y_test, clf.predict(x_test_trany),average='weighted')

f1_test=f1_score(y_test, clf.predict(x_test_trany),average='weighted')
data={"Test": ["precision", "recall", "accuracy", "f1"], "Metric": [precision_test,recall_test,accuracy_test,f1_test]}
# Create DataFrame
dr = pd.DataFrame(data)

# Display the DataFrame
dr

#train
accuracy_train=accuracy_score(y_train, clf.predict(X_transform))
recall_train=recall_score(y_train, clf.predict(X_transform),average='weighted')

precision_train=precision_score(y_train, clf.predict(X_transform),average='weighted')

f1_train=f1_score(y_train, clf.predict(X_transform),average='weighted')

data = { "Train": ["precision", "recall", "accuracy", "f1"],
    "Metric": [precision_train, recall_train, accuracy_train,f1_train]}

# Create DataFrame
dr = pd.DataFrame(data)

# Display the DataFrame
dr

#plot random forest
clf1 = RandomForestClassifier(n_estimators=50,max_depth=5)
clf1 = clf1.fit(X_transform, y_train)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(clf1.estimators_[5],
               feature_names = None,
               class_names=None,
               filled = True);
fig.savefig('clf1_individualtree.png')
st.pyplot(fig)

#feature importance
clf1.feature_importances_
df_plot = pd.DataFrame({'coef':list(clf1.feature_importances_),'name':x_train.columns})
r=(px.bar(data_frame=df_plot[df_plot['coef'] > 0], x='coef',y='name',height=2000))
st.plotly_chart(r)

Predict_clf1=clf1.predict(x_test_trany)


#plot confusion matrix
confusion_matrix(y_test, Predict_clf1)
cm = confusion_matrix(y_test, Predict_clf1, labels=reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=reg.classes_)
disp.plot()
plt.show()
st.pyplot()

accuracy_test=accuracy_score(y_test, clf1.predict(x_test_trany))
recall_test=recall_score(y_test, clf1.predict(x_test_trany),average='weighted')
precision_test=precision_score(y_test, clf1.predict(x_test_trany),average='weighted')
f1_test=f1_score(y_test, clf1.predict(x_test_trany),average='weighted')

data={"Test": ["precision", "recall", "accuracy", "f1"], "Metric": [precision_test,recall_test,accuracy_test,f1_test]}
# Create DataFrame
dr = pd.DataFrame(data)

# Display the DataFrame
dr

#train
accuracy_train=accuracy_score(y_train, clf1.predict(X_transform))
recall_train=recall_score(y_train, clf1.predict(X_transform),average='weighted')

precision_train=precision_score(y_train, clf1.predict(X_transform),average='weighted')

f1_train=f1_score(y_train, clf1.predict(X_transform),average='weighted')

data = { "Train": ["precision", "recall", "accuracy", "f1"],
    "Metric": [precision_train, recall_train, accuracy_train,f1_train]}

# Create DataFrame
dr = pd.DataFrame(data)

# Display the DataFrame
dr


#Plot XGBoost model
GBx=xgb.XGBClassifier(n_estimators=5,max_depth=5)
GBx.fit(X_transform,y_train)
xgb.plot_tree(GBx,num_trees = 2)
plt.rcParams['figure.figsize'] = [5, 5]
st.pyplot()

#feature importance
GBx.feature_importances_
df_plot = pd.DataFrame({'coef':list(GBx.feature_importances_),'name':x_train.columns})
x=(px.bar(data_frame=df_plot[df_plot['coef'] > 0], x='coef',y='name',height=2000))
st.plotly_chart(x)

#plot confusion matrix
Predict_GBx=GBx.predict(x_test_trany)


confusion_matrix(y_test, Predict_GBx)
cm = confusion_matrix(y_test, Predict_GBx, labels=reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=reg.classes_)
disp.plot()
plt.show()
st.pyplot()

accuracy_test=accuracy_score(y_test, GBx.predict(x_test_trany))
recall_test=recall_score(y_test, GBx.predict(x_test_trany),average='weighted')
precision_test=precision_score(y_test, GBx.predict(x_test_trany),average='weighted')
f1_test=f1_score(y_test, GBx.predict(x_test_trany),average='weighted')

data={"Test": ["precision", "recall", "accuracy", "f1"], "Metric": [precision_test,recall_test,accuracy_test,f1_test]}
# Create DataFrame
dr = pd.DataFrame(data)

# Display the DataFrame
dr

#Train
accuracy_train=accuracy_score(y_train, GBx.predict(X_transform))
recall_train=recall_score(y_train, GBx.predict(X_transform),average='weighted')
precision_train=precision_score(y_train, GBx.predict(X_transform),average='weighted')
f1_train=f1_score(y_train, GBx.predict(X_transform),average='weighted')

data = { "Train": ["precision", "recall", "accuracy", "f1"],
    "Metric": [precision_train, recall_train, accuracy_train,f1_train]}

# Create DataFrame
dr = pd.DataFrame(data)

# Display the DataFrame
dr