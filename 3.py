#由新辅前+新辅中+时间动态特征，一起训练模型，selectKbest+LASSO结合做特征选择，最后输出训练集和验证集的各种分类指标
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV,SelectKBest,chi2,RFE,f_classif
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
from sklearn.metrics import plot_roc_curve, auc
from sklearn.model_selection import train_test_split,learning_curve,ShuffleSplit,validation_curve,StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
import sklearn.neural_network as sk_nn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve
import warnings
import seaborn as sns
#pd.set_option("display.max_columns",1000)
#pd.set_option("display.max_rows",1000)
#pd.set_option("display.width",1000)
warnings.filterwarnings("ignore")

#导入训练集-佛山
xlsx1 = r"F:\巍哥的不敏感课题\特征文件\foshan-312-p+m-Nomal.csv"
xlsx11 = r"F:\巍哥的不敏感课题\特征文件\foshan-312-delta-Nomal.csv"
data1 = pd.read_csv(xlsx1,encoding='gbk')
data11 = pd.read_csv(xlsx11,encoding='gbk')

x1 = data1
x11 = data11

x1 = pd.DataFrame(x1)
x11 = pd.DataFrame(x11)

x111 = pd.concat([x1, x11], axis=1)
data111 = x111

x1 = data111[data111.columns[4:]]
y1 = data111[data111.columns[2]]
print(x1.shape)
print(y1.shape)

#导入外部验证集-中山医
xlsx2 = r"F:\巍哥的不敏感课题\特征文件\zhongshanyi-205-p+m-Nomal.csv"
xlsx22 = r"F:\巍哥的不敏感课题\特征文件\zhongshanyi-205-delta-Nomal.csv"
data2 = pd.read_csv(xlsx2,encoding='gbk')
data22 = pd.read_csv(xlsx22,encoding='gbk')

x2 = data2
x22 = data22

x2 = pd.DataFrame(x2)
x22 = pd.DataFrame(x22)

x222 = pd.concat([x2, x22], axis=1)
data222 = x222

x2 = data222[data222.columns[4:]]
y2 = data222[data222.columns[2]]
print(x2.shape)
print(y2.shape)
print(x1.head(5))
print(x2.head(5))

#SelectKbest进行特征选择，并且不改变列名
SELECTOR = SelectKBest(f_classif,k=2000)
x5 = SELECTOR.fit_transform(x1,y1)
mask = SELECTOR._get_support_mask()
x1 = x1.loc[:, mask]
x2 = x2.loc[:, mask]
x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
print(x1.head(5))
print(x2.head(5))

#LASSO特征选择
alphas = ([0.031])    #pCR
#alphas = ([0.0584])     #不敏感
model_lassoCV = LassoCV(alphas=alphas, cv=20, max_iter=100000).fit(x1, y1)
print(model_lassoCV.alpha_)
coef = pd.Series(model_lassoCV.coef_, index=x1.columns)
print("lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)))
index = coef[coef != 0].index
print(index)
x1 = x1[index]
x2 = x2[index]
#x.to_csv(r"F:\LOOCV影像组学文章\佛山-训练集\AR-PLOT111.csv")          #导出标准化后的x值
#x = x[x.columns[1:12]]           #不加增量特征
y1 = y1
y2 = y2
print(coef[coef != 0])
print(x1.shape)
print(y1.shape)
#x2.replace(np.nan, 0, inplace=True)
print(x2.shape)
y2.replace(np.nan, 0, inplace=True)
print(y2)

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sklearn.neural_network as sk_nn
import xgboost
import lightgbm as lgb

#classifier = sk_nn.MLPClassifier(activation="logistic",solver="adam",alpha=0.0001,learning_rate="adaptive",learning_rate_init=0.001,max_iter=200)
#classifier = AdaBoostClassifier()
#classifier = RandomForestClassifier()
#classifier = LinearDiscriminantAnalysis()
#classifier = LogisticRegression()
#classifier = svm.SVC(kernel="rbf", gamma="auto",probability=True)
#classifier = xgboost.XGBClassifier()
classifier = svm.SVC(kernel="linear", gamma="auto",probability=True)
#classifier = KNeighborsClassifier()
#classifier = DecisionTreeClassifier()
#classifier = GradientBoostingClassifier()
#classifier = GaussianNB()
#classifier = lgb()

#NUM_TRIALS = 20
#nested_scores = np.zeros(NUM_TRIALS)
Y_test_pre = []
tprs = []
tprs_train = []
train_auc_ = []
test_auc_ = []
train_accuracy_ = []
test_accuracy_ = []
train_recall_ = []
test_recall_ = []
train_specificity_ = []
test_specificity_ = []
mean_fpr = np.linspace(0, 1, 100)

x_train=x1
x_test=x2
y_train=y1
y_test=y2

classifier.fit(x_train,y_train)
y_train_pred = classifier.predict(x_train)
y_test_pred = classifier.predict(x_test).tolist()
#y_tes = y_test_pred
#Y_test_pre.append(y_tes)
#df = pd.DataFrame(Y_test_pre)
#pd.set_option("display.max_rows",None)
#print(df)
#output = "F:\巍哥的不敏感课题\特征文件\csv.csv"
#df.to_csv(output,sep=",")
y_test_pred = classifier.predict(x_test)

viz_train = plot_roc_curve(classifier, x_train, y_train, alpha=0.3, lw=1)
viz_test = plot_roc_curve(classifier, x_test, y_test, alpha=0.3, lw=1)
interp_tpr_train = np.interp(mean_fpr, viz_train.fpr, viz_train.tpr)
interp_tpr_test = np.interp(mean_fpr, viz_test.fpr, viz_test.tpr)
interp_tpr_train[0] = 0.0
interp_tpr_test[0] = 0.0
tp1 = np.sum(np.logical_and(y_train_pred, y_train))
fp1 = np.sum(np.logical_and(y_train_pred, np.logical_not(y_train)))
tn1 = np.sum(np.logical_and(np.logical_not(y_train_pred), np.logical_not(y_train)))
fn1 = np.sum(np.logical_and(np.logical_not(y_train_pred), y_train))
tpr1 = 0 if (tp1 + fn1 == 0) else float(tp1) / float(tp1 + fn1)
fpr1 = 0 if (fp1 + tn1 == 0) else float(fp1) / float(fp1 + tn1)
tp2 = np.sum(np.logical_and(y_test_pred, y_test))
fp2 = np.sum(np.logical_and(y_test_pred, np.logical_not(y_test)))
tn2 = np.sum(np.logical_and(np.logical_not(y_test_pred), np.logical_not(y_test)))
fn2 = np.sum(np.logical_and(np.logical_not(y_test_pred), y_test))
tpr2 = 0 if (tp2 + fn2 == 0) else float(tp2) / float(tp2 + fn2)
fpr2 = 0 if (fp2 + tn2 == 0) else float(fp2) / float(fp2 + tn2)
train_specificity = 1 - fpr1
test_specificity = 1 - fpr2
train_specificity_.append(train_specificity)
test_specificity_.append(test_specificity)
#auc
train_auc = viz_train.roc_auc
test_auc = viz_test.roc_auc
train_auc_.append(train_auc)
test_auc_.append(test_auc)
#accuracy
train_accuracy = classifier.score(x_train, y_train)
test_accuracy = classifier.score(x_test, y_test)
train_accuracy_.append(train_accuracy)
test_accuracy_.append(test_accuracy)
#sensitivity
train_recall = metrics.recall_score(y_train, y_train_pred)
test_recall = metrics.recall_score(y_test, y_test_pred)
train_recall_.append(train_recall)
test_recall_.append(test_recall)
print(str(classifier.predict(x_test)))
print(str(y_test))
#auc
print("mean_auc_train: " + str(np.mean(train_auc_)) +"                             +2std: "+str(np.mean(train_auc_)+1.96*np.std(train_auc_))+"   -2std: "+str(np.mean(train_auc_)-1.96*np.std(train_auc_))+"  std: "+str(np.std(train_auc_)))
print("mean_auc_test: " + str(np.mean(test_auc_)) +"                               +2std: "+str(np.mean(test_auc_)+1.96*np.std(test_auc_))+"   -2std: "+str(np.mean(test_auc_)-1.96*np.std(test_auc_))+"  std: "+str(np.std(test_auc_)))
#acc
print("mean_accuracy_train: " + str(np.mean(train_accuracy_)) +"                               +2std: "+str(np.mean(train_accuracy_)+1.96*np.std(train_accuracy_))+"   -2std: "+str(np.mean(train_accuracy_)-1.96*np.std(train_accuracy_))+"  std: "+str(np.std(train_accuracy_)))
print("mean_accuracy_test: " + str(np.mean(test_accuracy_)) +"                               +2std: "+str(np.mean(test_accuracy_)+1.96*np.std(test_accuracy_))+"   -2std: "+str(np.mean(test_accuracy_)-1.96*np.std(test_accuracy_))+"  std: "+str(np.std(test_accuracy_)))
#spe
print("mean_specificity_train: " + str(np.mean(train_specificity_)) +"                             +2std: "+str(np.mean(train_specificity_)+1.96*np.std(train_specificity_))+"   -2std: "+str(np.mean(train_specificity_)-1.96*np.std(train_specificity_))+"  std: "+str(np.std(train_specificity_)))
print("mean_specificity_test: " + str(np.mean(test_specificity_)) +"                             +2std: "+str(np.mean(test_specificity_)+1.96*np.std(test_specificity_))+"   -2std: "+str(np.mean(test_specificity_)-1.96*np.std(test_specificity_))+"  std: "+str(np.std(test_specificity_)))
#sen
print("mean_recall_train: " + str(np.mean(train_recall_)) +"                            +2std: "+str(np.mean(train_recall_)+1.96*np.std(train_recall_))+"   -2std: "+str(np.mean(train_recall_)-1.96*np.std(train_recall_))+"  std: "+str(np.std(train_recall_)))
print("mean_recall_test: " + str(np.mean(test_recall_)) +"                                 +2std: "+str(np.mean(test_recall_)+1.96*np.std(test_recall_))+"   -2std: "+str(np.mean(test_recall_)-1.96*np.std(test_recall_))+"  std: "+str(np.std(test_recall_)))

#print(test_auc_)
#print(test_accuracy_)