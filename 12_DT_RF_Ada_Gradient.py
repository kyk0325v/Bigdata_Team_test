#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import ParameterGrid 
param_grid = {'a':[1,2],'b':[True,False]} 
list(ParameterGrid(param_grid))


# In[2]:


param_grid = [{'kernel':['linear']},{'kernel':['rbf'],'gamma':[1,10]}]
list(ParameterGrid(param_grid))


# In[ ]:


# 선형회귀 => 비선형회귀
- scikits 


# In[4]:


from sklearn.preprocessing import PolynomialFeatures # 데이터를 다차원으로 변환
from sklearn.linear_model import LinearRegression # 선형회귀 => 비선형회귀

# train / validation / test 
# 전처리를 어느 시점에서 해야 하는가
# 전처리후 나누는가
# 나누고 전처리를 하는가 : 순서적으로 적용 ===> pipe_line
from sklearn.pipeline import make_pipeline

def PolynomialRegression(degree = 2 , **kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs)) # 비선형회귀를 위해 다차원으로 보내고 비선형회귀 하도록 pipeline 으로 묶음


# In[8]:


import numpy as np
def make_data(N , err = 1.0 , rseed = 1): # 학습을 위한 데이터
    rng = np.random.RandomState(rseed) # 의사난수의 시작점을 지정 ( 실제 난수처럼 보이도록 함 )
    X = rng.rand(N,1) ** 2
    y = 10 - 1/ (X.ravel() + 0.1) # ravel() : 데이터를 1차원으로 배열
    if err > 0:
        y += err * rng.randn(N)
    return X,y
X,y = make_data(40)
print(type(X))


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
X_test = np.linspace(-0.1,1.1,500)[:,None] # -0.1 ~ 1.1 까지 지정하고 500개로 나눔 

plt.scatter(X.ravel(),y,color='black') # 점 출력
axis = plt.axis()
for degree in [1,3,5]:
    y_test = PolynomialRegression(degree).fit(X,y).predict(X_test) # 학습데이터로 바로 예측
    plt.plot(X_test.ravel(),y_test,label = 'degree={}'.format(degree))
plt.xlim(-0.1,1)
plt.ylim(-2,12)
plt.legend(loc='best')


# In[11]:


from sklearn.model_selection import GridSearchCV

param_grid = {'polynomialfeatures__degree':np.arange(21),
               'linearregression__fit_intercept' : [True,False],
               'linearregression__normalize':[True,False] }
grid = GridSearchCV(PolynomialRegression() , param_grid , cv = 7)
grid.fit(X,y)


# In[12]:


grid.best_params_ # 최적의 parameter 를 알려줌


# In[13]:


grid.get_params() # parameter를 보여줌


# In[14]:


grid.cv_results_['params'] # degree 차수에 따라 진행된 결과를 보여줌


# In[15]:


grid.best_estimator_ # 최고의 조합일때


# In[16]:


grid.best_score_ # 최고의 조합일때의 정확도


# In[17]:


model = grid.best_estimator_ # 최적의 모델

plt.scatter(X.ravel(),y)
lim = plt.axis()
y_test = model.fit(X,y).predict(X_test)
plt.plot(X_test.ravel(),y_test)
plt.axis(lim)


# In[18]:


# pip install graphviz  ==> 3d 시각화


# In[19]:


# pip install pydot


# In[20]:


# pip install pydotplus


# In[21]:


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn import tree
import matplotlib.pyplot as plt

clf = tree.DecisionTreeClassifier(random_state=0)
iris = load_iris()


# In[22]:


fig ,ax = plt.subplots(figsize=(12,12))
clf = clf.fit(iris.data,iris.target)
tree.plot_tree(clf,max_depth=4,fontsize=10)


# In[26]:


cross_val_score(clf,iris.data,iris.target,cv=10)


# In[27]:


import pandas as pd
data = pd.DataFrame(iris.data)
print(data.head())
clf.predict(data.iloc[1:150,:])

# 대학 등급 나누기 좋을듯 


# %%
import io
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz
import matplotlib as mpl

def draw_decision_tree(model, feature_names):
    dot_buf = io.StringIO()
    export_graphviz(model, out_file=dot_buf,
    feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png()
    return Image(image)

# %%
draw_decision_tree(clf, iris.feature_names)
# %%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=1000, n_features=3,
n_redundant=0)
# %%
from sklearn.tree import DecisionTreeClassifier
# hyper parameter = > GridSearschCV
dt = DecisionTreeClassifier()
dt.fit(X,y)
# %%
preds = dt.predict(X)
(y == preds).mean()
# %%
draw_decision_tree(dt, ["feat1","feat2","feat3"])
# %%
# RF (random forest)
# - ensemble
# - bagging = bootstrap + aggregation
# - voting : 연속성 - 평균, 범주형 - 다수결의 원리
# - stacking : 여러모델의 결과로 다시 모델을 생성 ( DT, RF, ADA )
# - hyper parmaeter
#   - n_estimators
#   - n_samples
#   - max_features
# %%
from sklearn.datasets import make_classification
X, y = make_classification(1000)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion="entropy")  # default = gini
rf.fit(X,y)
# %%
print("Accuracy:\t", (y == rf.predict(X)).(mean))
# %%
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(# 화일이름과 family 이름 상이
    fname="c:/Windows/Fonts/Hypost.ttf").get_name()
rc('font', family=font_name)
# %%
f, ax = plt.subplots(figsize=(7,5))
ax.bar(range(0, len(rf.feature_importances_)), # 변수중요도
rf.feature_importances_)
ax.set_title('특성중요도')
# %%
print("종속변수갯수", rf.n_classes_)
print("클래스종류", rf.classes_ )
print("특성수", rf.n_features_)
print("모델", rf.estimators_)
# %%
# Tree regression 회귀 : 외삽데이터는 불가
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
boston = load_boston()  #<class 'sklearn.utils.Bunch'>
print(type(boston))
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rf = RandomForestRegressor() # 디폴트값으로 속성
rf.fit(X,Y)
print("변수중요도 score에 의해 정렬 :")
#%%
# 입력된 변수 순서와 같음
print(list(zip(map(lambda x: round(x,2),
rf.feature_importances_), names)))
#%%
# 내림차순으로 정렬해서 출력해 보시오
ziplist = sorted(list(zip(map(lambda x: round(x,2),
rf.feature_importances_), names)))
ziplist
#%%
len(names)
#%%
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(7,5))
ax.bar(range(0, len(rf.feature_importances_)),
rf.feature_importances_)
ax.set_title('feature importance')
# %%
# 평가 - randomForestRegression ( 회귀 )
- mse, mae, 결정계수
# %%
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
# %%
pred = rf.predict(X)
mean_squared_error(Y, pred)

# %%
mean_absolute_error(Y, rf.predict(X))
# %%
r2_score(Y, rf.predict(X))
# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
forest = RandomForestRegressor(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("훈련 세트 정확도 : {:.3f}".format(forest.score(X_train,y_train)))
print("테스트 세트 정확도 : {:.3f}".format(forest.score(X_test,y_test)))
# %%
from sklearn.tree import export_graphviz
export_graphviz(forest.estimators_[0], out_file="tree.dot",
class_names=["악성","양성"],
feature_names=cancer.feature_names,
impurity=False, filled=True)
# %%
import graphviz
from IPython.display import display
with open("tree.dot", "rt", encoding='UTF-8') as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))
# %%
# 문제
# - load_breast_cancer 데이터를 사용
# - n_estimators = [12, 24, 36, 48, 60]
# - min_samples_leaf = [1,2,4,8,16]
# - 최적의 파라미터를 결정하시오
# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
n_estimators = [12, 24, 36, 48, 60]
min_samples_leaf = [1,2,4,8,16]
forest = RandomForestRegressor(n_estimators=100, random_state=0,min_samples_leaf=1)
forest.fit(X_train, y_train)
print("훈련 세트 정확도 : {:.3f}".format(forest.score(X_train,y_train)))
print("테스트 세트 정확도 : {:.3f}".format(forest.score(X_test,y_test)))

#%%
##  선생님 답

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
cancer = load_breast_cancer()
x_train , x_test , y_train , y_test = train_test_split(cancer.data, cancer.target , random_state= 0)
scaler = MinMaxScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)

learner = RandomForestClassifier(random_state=2)
n_estimators = [12,24,36,48,60] # RF의 하이퍼파라미터
min_samples_leaf = [1,2,4,8,16] # DT의 하이퍼파라미터
parameters = {'n_estimators' : n_estimators , 'min_samples_leaf' : min_samples_leaf}

# 사용자 scorer 생성해서 대체
# 모델 평가 방법이 default는 accuracy
# roc curve : 대각선이면 50% , 좌상단으로 꽉차게 나오면 100%
def auc_scorer(target_score , prediction):
    auc_value = roc_auc_score(prediction , target_score)
    return auc_value
scorer = make_scorer(auc_scorer , greater_is_better = True)
grid_obj = GridSearchCV(learner, parameters , scorer) # learner 는 모델 , parameters 는 최적의 파라미터 , scorer 는 평가 방법 
grid_obj.fit(x_train_scaled , y_train)
grid_obj.best_params_
# %%
# 예측하고 평가
# - test 데이터
# %%
scaler = MinMaxScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)
# %%
pred = grid_obj.predict(X_test_scaled)  # 자동으로 최적모델을 적용
# %%
(pred==y_test).mean()
# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)
# %%
# 문제
# - iris 데이터를 로딩
# - 데이터를 분활( train. test )
# - 사전에 모델학습이 이뤄져야함 ( RF )
# - 변수 중요도부터 SelectFromModel 함수를 사용해서 변수 선택 ( threshhold=0.15%) 이상만 선택
# - RandomForest를 이용해서 분류한 다음 분류 정확도로 평가히시오
# - 변수중요도를 시각화하시오
# - estimator_[0]을 시각화 하시오
# %%
import pandas as pd
 
from sklearn import datasets
 
if __name__ == '__main__':
    iris = datasets.load_iris()
    print('아이리스 종류 :', iris.target_names)
    print('target : [0:setosa, 1:versicolor, 2:virginica]')
    print('데어터 수 :', len(iris.data))
    print('데이터 열 이름 :', iris.feature_names)
 
    # iris data Dataframe으로
    data = pd.DataFrame(
        {
            'sepal length': iris.data[:, 0],
            'sepal width': iris.data[:, 1],
            'petal length': iris.data[:, 2],
            'petal width': iris.data[:, 3],
            'species': iris.target
        }
    )
    print(data.head())
# %%
from sklearn.model_selection import train_test_split
 
x = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = data['species']
 
# 테스트 데이터 30%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics    
 
# 학습 진행
forest = RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train)
 
# 예측
y_pred = forest.predict(x_test)
print(y_pred)
print(list(y_test))
 
# 정확도 확인
print('정확도 :', metrics.accuracy_score(y_test, y_pred))
#%%
plt.scatter(data['sepal length'], data['sepal width'], c=y)
plt.show()
# %%
# 선생님 답
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
Y = iris.target
feat_labels = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

clf.fit(X_train, y_train)

# 열순서로 변수중요도를 출력
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)

# 변수 중요도가 있는 모델에서 사용 ( for문을 이용해서 사용가능 )
sfm = SelectFromModel(clf, threshold=0.15) # 문지방 
sfm.fit(X_train, y_train) # 변수선택

for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])

# 데이터 변형 : fit - transform
# 모델을 학습 : fit - predict
X_important_train = sfm.transform(X_train) # 4 => 2
X_important_test = sfm.transform(X_test)   # 4 => 2

clf_impoartant = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

clf_important.fit(X_important_train, y_train)
# 4개의 변수로 평가했을 경우
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred)) # 0.916
# 변수를 2개로 했을 경우
y_important_pred = clf_important.predict(X_important_test)
accuracy_score(y_test, y_important_pred) #0.9 # 일반화된경우




# %%
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(7,5))
ax.bar(range(0, len(clf_important.feature_importances_)),clf_important.feature_importances_)
ax.set_title('feature importance')
# %%
# estimator 0번째 시각화하시오
names=[]
for feature_list_index in sfm.get_support(indices=True):
    names.append(iris.feature_names[feature_list_index])
# %%
print(names)
draw_decision_tree(clf_impoartant.estimators_[0], names) #특징
# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=1000, n_features=4,
n_informative=2, n_redundant=0,
random_state=0, shuffle=False)
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X,y)
clf.feature_importances_
print(clf.predict([[0,0,0,0]]))
clf.score(X,y)
# %%
# 문제 : test data에 대하여 scoring해보시오
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X,y = make_moons(n_samples=500, noise=0.30, random_state=42)
#%%
X_train, X_test, y_train, y_test =train_test_split(X,y, random_state=42)
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
ada_clf.score(X, y)
# %%
# accuracy_score, score의 차이점 : 매개변수 앞은 예측 결과가 입력
# 예측전 데이터
ada_clf.score(X_test,y_test)
# %%
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt 
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100) 
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s) # 10000
    X_new = np.c_[x1.ravel(), x2.ravel()] # column 으로 합쳐라
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.xlabel(r"$x_2$", fontsize=18, rotation=0)
# %%
plot_decision_boundary(ada_clf, X, y)
# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
rng = np.random.RandomState(1)
X = np.linspace(0,6,100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
n_estimators=300, random_state=rng)

regr_1.fit(X,y)
regr_2.fit(X,y)
y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)
plt.figure()
plt.scatter(X,y, c="k", label="학습데이터")
plt.plot(X, y_1, c="g", label="에측기1개", linewidth=2)
plt.plot(X, y_2, c="r", label="예측기300개", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("ADA Boost")
plt.legend()
plt.show()
# %%
# gradient boost
import numpy as np
np.random.seed(42)
X = np.random.rand(100,1) - 0.5
y = 3*X[:, 0] **2 + 0.05 * np.random.randn(100)
# %%
from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
# %%
y2 = y - tree_reg1.predict(X) #잔차
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X,y2)
# %%
y3 = y2 - tree_reg2.predict(X) #잔차
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X,y3)
# %%
X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
print('y_pred :', y_pred)
# %%
# 학습율 : 오차  ( 신경망 )
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2,
n_estimators=3, learning_rate=1, random_state=42)
gbrt.fit(X,y)
print('y_pred:', gbrt.predict(X_new))
# %%
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
# %%
boston = load_boston()
X = pd.DataFrame(boston.data, columns= boston.feature_names)
y = pd.Series(boston.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)
# %%
regressor = GradientBoostingRegressor( max_depth=2, n_estimators=3, learning_rate=1.0)
regressor.fit(X_train, y_train)
# %%
regressor.feature_importances_
# %%
regressor.train_score_
# %%
regressor.estimators_[0][0].predict(X_test)
# %%
# MSE 평가를 한다 : n_estimators 개수를 결정
errors = [mean_squared_error(y_test, y_pred) for y_pred in regressor.staged_predict(X_test)]
print(errors)
best_n_estimators = np.argmin(errors)+1 #에러가 가장 작은 인덱스 
best_n_estimators
# %%
best_regressor = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators, learning_rate=1.0)
best_regressor.fit(X_train, y_train)
# %%
y_pred = best_regressor.predict(X_test)
mean_absolute_error(y_test, y_pred)
# %%
# - 데이터를 로딩하시오
# - 데이터를 확인 ( 탐색적 데이터 처리 )
# - 'yes' = 1,0
# - 데이터를 8:2로 분활
# - GradientBoostingClassifiter
# - estimators_[0][0] 에 대해 시각화
# - 테스트데이터에 대하여 scoreing
# - 변수중요도를 시각화하고
# - GridSearchCV를 이용해서 최적의 파라미터를 결정하시오
#   - 중요하다고 생각하는 hyper parameter에 대하여 최적화를 3개 이상 진행한다음 
#                                                    결과를 reporting 하시오.
# %%
import pandas as pd
import numpy as np

data = pd.read_csv('D:\python\pim.csv', encoding='CP949')
# %%
data.head()
# %%
data.info()
# %%
data.describe()

# %%
data['type'] = data['type'].replace('Yes', 1)
data['type'] = data['type'].replace('No', 0)
# %%
data.head()
# %%
features = data

# %%
X = data[:, [2, 3]]
y = data.target
from sklearn.tree import DecisionTreeClassifier
tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0).fit(X, y)
pd.DataFrame(data, columns=data.feature_names).head()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)
# %%

# %%
