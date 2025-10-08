## EXNO-3-DS
# NAME: DHAYALAPRABU.S
# REG NO: 212224230065
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
df=pd.read_csv("Encoding Data.csv")
df

```

<img width="609" height="452" alt="image" src="https://github.com/user-attachments/assets/1a5ab0b5-7255-433b-97e2-5b7740e06036" />


```

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```

<img width="215" height="231" alt="image" src="https://github.com/user-attachments/assets/b99b94d3-d1b2-4bb4-b2f8-59b56ab29223" />


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df

```
<img width="446" height="447" alt="image" src="https://github.com/user-attachments/assets/82704c65-cce8-4928-8217-6f75de43ae43" />

```

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

```
<img width="482" height="453" alt="image" src="https://github.com/user-attachments/assets/2ed9195e-ee1c-4b71-a82d-fc839d8b977e" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2

```
<img width="651" height="445" alt="image" src="https://github.com/user-attachments/assets/f2faec3f-6a23-4275-9942-5773ee3b2cdd" />

```
pd.get_dummies(df2,columns=["nom_0"])

```
<img width="788" height="446" alt="image" src="https://github.com/user-attachments/assets/45a67ebd-f3d9-4b65-92e6-75341de1f141" />

```

pip install --upgrade category_encoders

```
<img width="1362" height="427" alt="image" src="https://github.com/user-attachments/assets/b4cd285e-53b5-4beb-810c-79ae1164c650" />

```

from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df

```
<img width="579" height="448" alt="image" src="https://github.com/user-attachments/assets/2963633e-6979-4ffa-b591-157bc24c8faf" />

```

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb

```
<img width="839" height="460" alt="image" src="https://github.com/user-attachments/assets/8abdb44b-0512-496b-9dec-a7373c5df157" />

```

# MEAN ENCODING
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

```
<img width="668" height="451" alt="image" src="https://github.com/user-attachments/assets/a99d1705-aeec-4217-a206-8b9c71e062fa" />


```

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df

```

<img width="943" height="515" alt="image" src="https://github.com/user-attachments/assets/163c8c81-8760-4973-9e8a-6426ab4497fc" />

```

df.skew()

```

<img width="373" height="251" alt="image" src="https://github.com/user-attachments/assets/50af8abf-8b66-4df1-af93-745f471ae449" />

```

np.log(df["Highly Positive Skew"])

```
<img width="307" height="529" alt="image" src="https://github.com/user-attachments/assets/c173c17a-0cb0-4e85-bad6-cad7d48e86c7" />

```
np.reciprocal(df["Moderate Positive Skew"])

```
<img width="337" height="560" alt="image" src="https://github.com/user-attachments/assets/7d2d43f3-9e39-4ede-8114-28e1aa0b8eca" />


```
np.sqrt(df["Highly Positive Skew"])

```

<img width="305" height="532" alt="image" src="https://github.com/user-attachments/assets/0250585d-2327-4a99-b108-2a1ca66626cb" />

```

np.square(df["Highly Positive Skew"])

```
<img width="377" height="582" alt="image" src="https://github.com/user-attachments/assets/a429d5fa-5b3f-430c-b881-ec9dc3cebe17" />


```

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

```
<img width="1235" height="526" alt="image" src="https://github.com/user-attachments/assets/c0db7d7b-e6a2-4648-8414-3470910d194c" />

```
df.skew()

```
<img width="405" height="269" alt="image" src="https://github.com/user-attachments/assets/f1cfcec9-d0f6-4e30-aebd-488875560918" />

```

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

```
<img width="435" height="289" alt="image" src="https://github.com/user-attachments/assets/f01f44be-a0fe-4937-ad47-6bf933b4f7cd" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

```
<img width="1303" height="516" alt="image" src="https://github.com/user-attachments/assets/247b7a93-9c0d-473b-82c3-4f97ed10cd5e" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

<img width="761" height="556" alt="image" src="https://github.com/user-attachments/assets/0b1b3241-b893-4016-bd51-a25a908e7047" />

```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```

<img width="747" height="556" alt="image" src="https://github.com/user-attachments/assets/b38b041b-fa1e-4a1b-a49d-477ac0d66070" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

<img width="721" height="557" alt="image" src="https://github.com/user-attachments/assets/f8493e5d-447a-45dc-950c-8dbffee0f3e4" />


```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

```
<img width="732" height="563" alt="image" src="https://github.com/user-attachments/assets/58e07fd4-3c37-4183-84fe-3a9be664a24e" />




# RESULT:

Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.


       
