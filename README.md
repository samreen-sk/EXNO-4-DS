# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
 ```
![374526018-8723244a-236e-4934-94b7-3273689f1b27](https://github.com/user-attachments/assets/26fdedca-dbd9-47ef-94fa-a239560d070f)

```
data.isnull().sum()
```
![374526112-5bf94360-2c21-47ac-9638-1ab7ca6b74a3](https://github.com/user-attachments/assets/0bc6896a-b885-4945-952a-6a4f3836ff91)

```

missing=data[data.isnull().any(axis=1)]
missing
```
![374526216-09d7928d-c7b0-4777-8633-6816a03618ba](https://github.com/user-attachments/assets/d6b75096-25d4-4a58-8a14-afef97fbd2ac)
```

data2=data.dropna(axis=0)
data2
```
![374526299-b63a5ec7-389a-4dcf-9322-87cde1c46a1f](https://github.com/user-attachments/assets/2757606a-5c89-40d2-adac-77bfcf138ce4)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![374526375-0300b1c5-2526-416d-88eb-f1e7961e38f4](https://github.com/user-attachments/assets/8b7a7886-d114-4791-a34c-35b975f42a68)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![374526444-7f376d3a-3473-41b3-8201-ebd8fce0e101](https://github.com/user-attachments/assets/2cd9976f-7186-4dfd-9dbe-8006c28a46bf)

```
data
```
![374526548-b100bf4c-18ba-4db4-b7a2-3d8e8c3a2cb5](https://github.com/user-attachments/assets/f3133ca1-02a7-4c46-9f3b-0f365dcf8f4e)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![374526609-e482870e-b582-46dd-80f0-f089c74ec6fb](https://github.com/user-attachments/assets/4ca5ce23-114e-4f93-a575-650cc8e60aa8)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![322413259-7a804881-6539-434f-8707-6af6b812bafc](https://github.com/user-attachments/assets/3ed77a88-f075-4051-9ec1-e874d812d16e)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![374526750-355c1f09-bbd5-49ba-b262-0e7bc9ec1dd8](https://github.com/user-attachments/assets/36596fed-8e42-49c2-bedd-88bc3e54f024)
```
y=new_data['SalStat'].values
print(y)
```
![374526812-0d827813-f3bb-40d4-9151-0b202ed59ad8](https://github.com/user-attachments/assets/3f1ff62c-8c27-48b8-b4d7-7eea95154e42)

```
x=new_data[features].values
print(x)
```
![374526882-84730e94-62f2-4486-85e0-9543ff4b0caf](https://github.com/user-attachments/assets/51b99ce7-76c5-4946-8109-da364a9d0d5c)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![374527039-1a4a8673-dbb4-4d7f-88af-d13d500661d6](https://github.com/user-attachments/assets/36261198-4853-4b66-a28e-cddd544674f1)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![374527077-cbe025c3-74c1-430c-a6a0-a1c18c374054](https://github.com/user-attachments/assets/2515eb6a-dac4-4f11-be8a-d66c000c42e6)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

![374527132-63f15433-842d-493d-bfca-b048300c5d01](https://github.com/user-attachments/assets/d9740ef4-b6ee-4245-9fd9-a926428df182)

```
data.shape
```
![374527175-e8f30aa5-8d30-4fd4-8810-bf7c27ab6138](https://github.com/user-attachments/assets/db1937e9-ff2b-4774-b5ed-87f8e1120f90)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![374527230-98f285ef-0d8f-47f3-b547-650fd96da0c6](https://github.com/user-attachments/assets/b22276dd-bced-4590-8adf-ea84ed889f86)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![374527285-36c14ca4-2378-458a-a7f8-d4ceeda4106d](https://github.com/user-attachments/assets/737bf960-55db-4317-9876-4232f6d203e7)
```
tips.time.unique()
```
![374527458-d9472ef7-8d47-4fb8-96b4-44fe6dff8586](https://github.com/user-attachments/assets/75589966-6ef4-499f-93af-cf84523a8185)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![374527459-f0da14fb-bed1-46df-b922-bcfd4502b108](https://github.com/user-attachments/assets/d2fa16d9-12d0-4c91-80bc-6d6fe604484e)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![374527472-af5fac67-fc4f-4d3e-a762-78e851422a63](https://github.com/user-attachments/assets/39f7fdbd-8b1c-40e2-a5cb-5a13d03bcdc6)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
