## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('data.csv')
print(df)
```
<img width="960" height="415" alt="image" src="https://github.com/user-attachments/assets/7f723004-aa54-48a8-ad32-2c970578de48" />

```
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
climate = ['Cold','Warm','Hot','Very Hot']
ele = OrdinalEncoder(categories=[climate])
ele.fit_transform(df[["Ord_1"]])
```
<img width="253" height="368" alt="image" src="https://github.com/user-attachments/assets/a08a7556-2ddf-4776-bc91-3d4a3ed1e95e" />

```
df['bo2'] = ele.fit_transform(df[["Ord_1"]])
df
```
<img width="872" height="595" alt="image" src="https://github.com/user-attachments/assets/2714690b-ac37-492c-8697-f2a5c5d56827" />

```
le = LabelEncoder()
df2 = df.copy()
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2

```
<img width="811" height="597" alt="image" src="https://github.com/user-attachments/assets/c274eccf-e7ed-4612-9529-e888b9949923" />

```
df2['Ord_2'] = le.fit_transform(df2['Ord_2'])
df2
```
<img width="820" height="594" alt="image" src="https://github.com/user-attachments/assets/dcc453e2-db6a-4d69-89e8-de5659d9ff0a" />

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
df3 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["City"]]))
df2 = pd.concat([enc,df3],axis = 1)
df2
```
<img width="1012" height="617" alt="image" src="https://github.com/user-attachments/assets/12531c69-b7ff-432a-bbde-17ae27c5183d" />

```
pd.get_dummies(df,columns=['City'])
```
<img width="1458" height="602" alt="image" src="https://github.com/user-attachments/assets/55d4c098-8241-4f69-96e8-c18c7703f8cf" />

```
from category_encoders import BinaryEncoder
df = pd.read_csv('data.csv')
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
nd
```
<img width="320" height="460" alt="image" src="https://github.com/user-attachments/assets/78504eff-6376-42ba-a3e1-45b0b642131c" />

```
from category_encoders import TargetEncoder
te = TargetEncoder()
CC = df.copy()
new = te.fit_transform(CC["City"],y=CC["Target"])
CC = pd.concat([CC,new],axis = 1)
CC
```

<img width="694" height="454" alt="image" src="https://github.com/user-attachments/assets/9466fdb4-b770-491b-86e4-bc0911903bac" />

```
if 'City' in CC.columns:
    CC = CC.drop('City', axis=1)
new = te.fit_transform(X = df["City"],y=df["Target"])
CC = pd.concat([CC.reset_index(drop=True),new.reset_index(drop=True)],axis = 1)
CC
```

<img width="596" height="452" alt="image" src="https://github.com/user-attachments/assets/334fc7f9-a1ab-42db-89cf-b07693c79bf9" />

```
df = pd.read_csv('Data_to_Transform.csv')
df
```
<img width="952" height="551" alt="image" src="https://github.com/user-attachments/assets/e56876aa-68b4-4598-8387-522ba03b88c1" />

```
df.skew()
```
<img width="433" height="142" alt="image" src="https://github.com/user-attachments/assets/06bef4d5-16d8-4da5-aac2-d5e92160c045" />

```
np.log(df["Highly Positive Skew"])
```
<img width="697" height="326" alt="image" src="https://github.com/user-attachments/assets/82226fbd-5cad-425e-877c-a5f7e36d9500" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="717" height="325" alt="image" src="https://github.com/user-attachments/assets/136c1231-16f7-41e3-a889-f653b624196b" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="699" height="325" alt="image" src="https://github.com/user-attachments/assets/deceae5e-a6f6-4e46-ac20-1751c5dcd651" />

```
np.square(df["Highly Positive Skew"])
```
<img width="698" height="328" alt="image" src="https://github.com/user-attachments/assets/3c92ecf5-7bac-451c-b2fe-9fe39a9c681b" />

```
df["Highly Positive Skew_boxcox"], parameters = stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1236" height="547" alt="image" src="https://github.com/user-attachments/assets/ac165cf2-2386-481e-ae70-b102568f3175" />

```
df["Moderate Negative Skew_yeojohnson"], parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
```
<img width="1532" height="573" alt="image" src="https://github.com/user-attachments/assets/96aeeb32-e696-48a0-823f-1f25470af532" />

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1524" height="569" alt="image" src="https://github.com/user-attachments/assets/e05b443f-a8d0-4b48-a086-72770b39bd41" />

```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df["Moderate Negative Skew"],line = '45')
plt.show()
```
<img width="834" height="592" alt="image" src="https://github.com/user-attachments/assets/34df3870-0551-414b-845c-e2f8056856f6" />

```
sm.qqplot(df["Moderate Negative Skew_1"],line = '45')
plt.show()
```
<img width="801" height="606" alt="image" src="https://github.com/user-attachments/assets/c1b10897-12fa-46fe-9dbf-17076fb08a19" />

```

df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line = '45')
plt.show()
```
<img width="840" height="608" alt="image" src="https://github.com/user-attachments/assets/81456f08-33b5-41cf-8dc4-751d58b86ece" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line = '45')
plt.show()
```
<img width="854" height="610" alt="image" src="https://github.com/user-attachments/assets/f2f30e14-58ea-48bd-bf73-ad30c196538c" />

```
sm.qqplot(df["Highly Negative Skew_1"],line = '45')
plt.show()
```
<img width="806" height="613" alt="image" src="https://github.com/user-attachments/assets/14de33cf-4f8e-48d2-a358-04db5b24e6cd" />

```
sm.qqplot(np.abs(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```

<img width="811" height="612" alt="image" src="https://github.com/user-attachments/assets/3f14f637-957d-4534-8d0b-ba379e575472" />

```
sm.qqplot(np.log(df["Highly Negative Skew_1"]),line = '45')
plt.show()
```

<img width="821" height="610" alt="image" src="https://github.com/user-attachments/assets/3c1c5585-cef2-4fe1-a0c9-3c225c3ec853" />

```
sm.qqplot(np.sqrt(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```

<img width="798" height="606" alt="image" src="https://github.com/user-attachments/assets/94fdc382-cb7a-42b8-8f5f-926a35047841" />

```
pd.concat([CC,new],axis = 1)
```

<img width="622" height="414" alt="image" src="https://github.com/user-attachments/assets/de5d1e99-95f6-4a4f-9ae0-c9536c48ce1b" />

# RESULT:
Thus, we have successfully performed Feature Encoding and Transformation   process and saved the data to a file.
       
