import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# EDA

df = pd.read_csv ( 'titanic.csv')
df.drop('Cabin', axis=1, inplace=True)
df['Embarked']=df['Embarked'].fillna('S')
df['Age'] = df['Age'].fillna(df['Age'].mean()
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['isAlone'] = np.where(df['FamilySize'] == 1,1,0)
df['GenderClass'] = df.apply(lambda dff: 'child' if dff['Age']<15 else dff['Sex'], axis=1)
df['Title'] = df['Name'].apply(lambda dff : dff.split(',')[1].split('.')[0].strip())

title_mapping = {
    'Mr':'Mr',
    'Miss':'Miss',
    'Mrs':'Mrs',
    'Master':'Master',
    'Dr':'Mr',
    'Rev':'Mr',
    'Major':'Mr',
    'Mlle':'Miss',
    'Col':'Mr',
    'Don':'Mr',
    'Mme':'Mrs',
    'Ms':'Miss',
    'Lady':'Mrs',
    'Sir':'Mr',
    'Capt':'Mr',
    'the Countess':'Mr',
    'Jonkheer':'Mr'
}

df['Title'] = df['Title'].map(title_mapping)
df.drop(['PassengerId','Name','Sex','SibSp', 'Parch','Ticket'], axis= 1, inplace = True)
df = pd.get_dummies(df, columns= ['Embarked','GenderClass','Title'], drop_first=True, dtype=int)

# MODEL BUILDING
# data split

x = df.drop('Survived', axis = 1)
y = df['Survived']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0, stratify=y)
features_to_scale = ['Age','Fare']

# scaling the features  
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[features_to_scale] = sc.fit_transform(x_train[features_to_scale])
x_test[features_to_scale] =sc.transform(x_test[features_to_scale])

# modeling - Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred_train  = model.predict(x_train)
y_pred_test = model.predict(x_test)

# evaluaing model performance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy  = accuracy_score(y_test, y_pred_test)

print(f'train accuracy : {train_accuracy}')
print(f'test accuracy  : {test_accuracy}')

confusion_mat = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(confusion_mat, index= ['Actually died', 'Actually survived'],
                     columns=['Predicted Died', 'Predicted Survived'])

print(cm_df)
