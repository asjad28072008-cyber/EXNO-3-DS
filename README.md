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
df=pd.read_csv('/content/Encoding Data.csv')
df
```

<img width="395" height="402" alt="image" src="https://github.com/user-attachments/assets/007b4dde-6296-47cd-b4ec-5e32288049c7" />


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder 
pm=['Hot','Warm','Cold'] 
e1=OrdinalEncoder(categories=[pm]) 
e1.fit_transform(df[["ord_2"]])
```

<img width="291" height="215" alt="image" src="https://github.com/user-attachments/assets/a4e69289-d02f-4356-a1c6-e9df329be4bf" />


```
df['bo2']=e1.fit_transform(df[["ord_2"]]) 
df
```

<img width="452" height="397" alt="image" src="https://github.com/user-attachments/assets/e9d348b7-ba8e-4b1c-92db-23bb6f9716e8" />


```
le=LabelEncoder() 
dfc=df.copy() 
dfc['ord_2']=le.fit_transform(dfc['ord_2']) 
dfc
```

<img width="417" height="396" alt="image" src="https://github.com/user-attachments/assets/faa1ff6e-4df9-4315-a77f-9098a9006e9a" />


```
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder(sparse_output=False) 
df2=df.copy() 
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) 
# Orders in Alphabetical Order Blue , Green, Red 
df2=pd.concat([df2,enc],axis=1) 
df2
```

<img width="548" height="390" alt="image" src="https://github.com/user-attachments/assets/e0eb48b6-a492-410f-91b0-13d9092b2084" />



```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="796" height="392" alt="image" src="https://github.com/user-attachments/assets/e0a9e18d-8ec0-471a-8232-bcc249fdbd12" />


```
!pip install category_encoders
import pandas as pd
from category_encoders import BinaryEncoder
df=pd.read_csv('/content/data.csv') 
df
```

<img width="1355" height="662" alt="image" src="https://github.com/user-attachments/assets/3376ef57-ab32-449a-b593-ee9057c3ca3a" />



```
be=BinaryEncoder() 
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1) 
dfb
```

<img width="790" height="391" alt="image" src="https://github.com/user-attachments/assets/5ca738da-9e41-44f0-8606-60f293d4348e" />



```
from category_encoders import TargetEncoder 
te=TargetEncoder() 
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC
```
<img width="625" height="385" alt="image" src="https://github.com/user-attachments/assets/59b933f5-2f20-478f-bf9f-e6d08bdd9834" />


```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("/content/Data_to_Transform.csv") 
df
```

<img width="915" height="448" alt="image" src="https://github.com/user-attachments/assets/3439d805-ef04-42ba-bc29-decc170a7b25" />


```
df.skew()
```

<img width="357" height="207" alt="image" src="https://github.com/user-attachments/assets/2c22462f-58a1-4db2-8764-7cb62cbd3a2a" />



```
np.log(df["Highly Positive Skew"])
```

<img width="345" height="475" alt="image" src="https://github.com/user-attachments/assets/328932ae-240b-47e0-89bf-1151bbdf0c06" />



```
# 2. RECIPROCAL TRANSFORMATION 
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="331" height="477" alt="image" src="https://github.com/user-attachments/assets/83a4fa38-8d18-4006-8c15-79d580c55b45" />



```
# 4. SQUARE ROOT TRANSFORMATION 
np.sqrt(df["Highly Positive Skew"])
```


<img width="312" height="473" alt="image" src="https://github.com/user-attachments/assets/66d7378f-cffa-470a-803f-3200d3b4a13c" />



```
# 5. SQUARE TRANSFORMATION 
np.square(df["Highly Positive Skew"])
```



<img width="406" height="483" alt="image" src="https://github.com/user-attachments/assets/7aece76c-b42d-4955-912f-eeadc5445394" />



```
# POWER TRANSFORMATIONS 
#        BOX COX 
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```

<img width="1247" height="442" alt="image" src="https://github.com/user-attachments/assets/dfd129b9-f291-476c-827b-0c7b721f8c2b" />



```
df.skew()
```

<img width="375" height="247" alt="image" src="https://github.com/user-attachments/assets/f85f2f5f-1da9-4c8a-9cd1-82462da08f63" />



```
# YEO_JOHNSON 
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```

<img width="466" height="297" alt="image" src="https://github.com/user-attachments/assets/f1ad8aa0-ac79-4543-a6b0-56c33e83355f" />


```
# QUANTILE TRANSFORMATION 
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```


<img width="1417" height="391" alt="image" src="https://github.com/user-attachments/assets/fb5dec5a-37bb-45bc-9a7b-d3cc14de27f2" />


```
import seaborn as sns 
import statsmodels.api as sm 
# STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
# QQ - QUANTILE QUANTILE PLOT 
plt.show()
```


<img width="605" height="432" alt="image" src="https://github.com/user-attachments/assets/04767eaf-07e2-4da2-892a-91919199c01c" />



```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL 
plt.show()
```

<img width="595" height="436" alt="image" src="https://github.com/user-attachments/assets/8dffc605-0071-4a77-aba2-66ac0c0c806f" />



```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```


<img width="596" height="421" alt="image" src="https://github.com/user-attachments/assets/3f06d2ff-3d70-4983-8ee6-03d3ff049ba4" />


      
# RESULT:
      

          Thus the Implementation of Feature Encoding and Feature Transformation executed successfully.
