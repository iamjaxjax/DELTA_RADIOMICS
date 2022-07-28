#由两次影像组学特征做差异化分析生成一组动态特征
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV,SelectKBest,chi2,RFE
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

#xlsx = r"F:\LOOCV影像组学文章\result3-TRA.csv"
xlsx = r"D:\省医淋巴结课题\新的数据\跑模型\预测SLN\shengyi.csv"
data1 = pd.read_csv(xlsx,encoding='gbk')
#data1 = shuffle(data1)
x1 = data1[data1.columns[4:7342]]
x2 = data1[data1.columns[7342:]]
x1.replace(np.nan, 0, inplace=True)
x1.replace(0, 0.001, inplace=True)
x2.replace(np.nan, 0, inplace=True)

x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)

x1 = x1.values
x2 = x2.values

x3 = (x1-x2)/x1

x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
x3 = pd.DataFrame(x3)
x3.to_csv(r"D:\省医淋巴结课题\新的数据\跑模型\预测SLN\shengyi-动态特征.csv")