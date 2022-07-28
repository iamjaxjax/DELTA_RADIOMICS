#由新辅前+新辅中+时间动态特征，一起训练模型，selectKbest+LASSO结合做特征选择，最后输出LASSO的那个棒状图
import pandas as pd
import time
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LassoLarsIC, LassoLarsCV
import numpy as np
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.feature_selection import RFECV,SelectKBest,chi2,RFE,f_classif
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
from sklearn.metrics import plot_roc_curve, auc, r2_score
from sklearn.model_selection import train_test_split,learning_curve,ShuffleSplit,validation_curve,StratifiedKFold, KFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
import sklearn.neural_network as sk_nn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve
from sklearn.svm import SVC
import warnings
import seaborn as sns
#pd.set_option("display.max_columns",1000)
#pd.set_option("display.max_rows",1000)
#pd.set_option("display.width",1000)
warnings.filterwarnings("ignore")

xlsx1 = r""
#xlsx1 = r""    
data1 = pd.read_csv(xlsx1,encoding='gbk')

x123 = data1
x123 = pd.DataFrame(x123)

x1 = data1[data1.columns[3151:]]                y1 = data1[data1.columns[3]]                    

colnames = x1.columns
x1 = x1.astype(np.float64)
x1 = StandardScaler().fit_transform(x1)
x1 = pd.DataFrame(x1)
x1.columns = colnames
y1 = y1.astype(np.float64)
x1 = x1.astype(np.float64)
print(x1.shape)
print(y1.shape)
print(y1)

sm = SMOTE(random_state=1)
x, y = sm.fit_resample(x1,y1)
print("Resample dataset shape %s" % Counter(y))
y1 = y.astype(np.float64)
x1 = x.astype(np.float64)

alphas = np.logspace(-2,0,200,base=10)                  
regr_cv = LassoCV(alphas=alphas,cv=10,max_iter=200000)
regr_cv.fit(x1,y1)                                  

MSEs_mean = regr_cv.mse_path_.mean(axis=1)
MSEs_std = regr_cv.mse_path_.std(axis=1)
print(MSEs_mean)                                   
print(MSEs_std)                        
print()
plt.figure()
plt.errorbar(regr_cv.alphas_,MSEs_mean,
                 yerr=MSEs_std,
                 fmt="o",
                 ms=3,
                 mfc="r",
                 mec="r",
                 ecolor="lightgray",
                 elinewidth=2,
                 capsize=4,
                 capthick=1)
plt.semilogx()
plt.axvline(regr_cv.alpha_,color="black",ls="--")
plt.axvline(regr_cv.alpha_+0.03488, color="black", ls="--")        
plt.xlabel("Lambda",fontsize=18,weight="normal")
plt.ylabel("MSE",fontsize=18,weight="normal")
plt.show()
MSEs_mean.sort()
#max = MSEs_mean[len(MSEs_mean)-1]
min = MSEs_mean[0]
print("lambda mse",min)
MSEs_std.sort()
min2 = MSEs_std[0]
print("lambda std",min2)
