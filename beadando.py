import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# adatok beolvasása, feature engineering elvégzése

# age
# nem
# cp --> chest pain
# trestbps --> blood pressure
# chol --> koleszterin
# fbs --> éhgyomor vércukorszint
# restECG --> ECG eredmény
# thalach --> maximum heart rate
# exang --> mellkasi fájdalom sportolástól
# oldpeal --> edzés által kiváltott ST abnormalitás
# slope --> ST segmens meredeksége
# ca --> number of major vessels colored (nagy erek beszínezése)
# thal --> vérbetegség
# num eredmény változó szívbeteg-e vagy nem 0-> nem 1,2,3,4-> igen



def read_data():

    data_header=["age", "sex","chest_pain_type","resting_blood_pressure","cholesterol","fasting_blood_sugar","rest_ECG","max_heart_rate","exercise_induced_angina","ST_depression","ST_slope", "major_vessels","thalassemia","heart_disease"]
    data = pd.read_csv(filepath_or_buffer="reprocessed.hungarian.csv",sep=";",header=None, names=data_header)
    data = data.replace(-9, 0)
    return data

data = read_data()
#print(data)

def feature_engineering(data):

    data = read_data()
    # Kategoriák létrehozása
    data['sex'][data['sex'] == 0] = 'female'
    data['sex'][data['sex'] == 1] = 'male'

    data['chest_pain_type'][data['chest_pain_type'] == 1] = 'typical angina'
    data['chest_pain_type'][data['chest_pain_type'] == 2] = 'atypical angina'
    data['chest_pain_type'][data['chest_pain_type'] == 3] = 'non-anginal pain'
    data['chest_pain_type'][data['chest_pain_type'] == 4] = 'asymptomatic'

    data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
    data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

    data['rest_ECG'][data['rest_ECG'] == 0] = 'normal'
    data['rest_ECG'][data['rest_ECG'] == 1] = 'ST-T wave abnormality'
    data['rest_ECG'][data['rest_ECG'] == 2] = 'left ventricular hypertrophy'

    data['exercise_induced_angina'][data['exercise_induced_angina'] == 0] = 'no'
    data['exercise_induced_angina'][data['exercise_induced_angina'] == 1] = 'yes'

    data['ST_slope'][data['ST_slope'] == 1] = 'upsloping'
    data['ST_slope'][data['ST_slope'] == 2] = 'flat'
    data['ST_slope'][data['ST_slope'] == 3] = 'downsloping'

    data['heart_disease'] = data['heart_disease'].astype('object')
    data['heart_disease'][data['heart_disease'] == 0] = 'no'
    data['heart_disease'][data['heart_disease'].isin([1, 2, 3, 4])] = 'yes'

    data['thalassemia'][data['thalassemia'] == 3] = 'normal'
    data['thalassemia'][data['thalassemia'] == 6] = 'fixed defect'
    data['thalassemia'][data['thalassemia'] == 7] = 'reverse defect'


    # átalakítás számmá
    data['resting_blood_pressure'] = data['resting_blood_pressure'].astype('int64')
    data['cholesterol'] = data['cholesterol'].astype(int)
    data['max_heart_rate'] = data['max_heart_rate'].astype(int)

    # az adatok több mint 90%-a hiányzik (feature selection)
    data = data.drop("major_vessels", axis='columns')
    #data = data.drop("thalassemia", axis='columns')

    # one-hot encoding alapján 1-es lesz ott ahol igaz
    data= pd.get_dummies(data, drop_first=True, dtype=float)
    return data

def colleration(data):
    data = feature_engineering(data)
    correlation_matrix = data.corr()
    plt.figure(figsize=(10,8))

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
    

data = feature_engineering(data)
print(data)
print(data.dtypes)
colleration(data)


#VIF mutató
X = data.drop('heart_disease_yes', axis=1)  # Jellemzők (összes oszlop kivéve a 'heart_disease' oszlopot)
y = data['heart_disease_yes']
vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
  
print(vif_data)

# Osztályeloszlás megjelenítése
def osztalyeloszlas(data):
    class_distribution = data['heart_disease_yes'].value_counts()
    print("Osztályeloszlás:")
    print(class_distribution)

    plt.figure(figsize=(8, 6))
    sns.countplot(x='heart_disease_yes', data=data)
    plt.title('Osztályeloszlás')
    plt.xlabel('Szívbetegség jelenléte (0: nincs, 1: van)')
    plt.ylabel('Gyakoriság')
    plt.show()

osztalyeloszlas(data)

# Undersampling
def undersampling(data):
    # Többségi osztály és kisebbségi osztály szétválsztása
    majority_class = data[data['heart_disease_yes'] == 0]
    minority_class = data[data['heart_disease_yes'] == 1]

    # Többségi osztály mintáinak alintavételezése (csökkentése)
    undersampled_majority = resample(majority_class, replace=False, n_samples=len(minority_class)) #randomstate = 42

    # Kisebbségi osztály és alintavételezett többségi osztály összeadása
    undersampled_data = pd.concat([undersampled_majority, minority_class])
    return undersampled_data


undersampled_data = undersampling(data)
print(undersampled_data)

osztalyeloszlas(undersampled_data)

#PCA
X = data.drop('heart_disease_yes', axis=1)
y = data['heart_disease_yes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

n_components = (cumulative_explained_variance < 0.95).sum() + 1

pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

# Calculate VIF for each principal component
X_standardized = pd.DataFrame(X_scaled, columns=X.columns)
X_standardized['intercept'] = 1

vif_data = pd.DataFrame()
vif_data['Feature'] = X_standardized.columns
vif_data['VIF'] = [variance_inflation_factor(X_standardized.values, i) for i in range(X_standardized.shape[1])]

# Drop the intercept column after calculating VIF
vif_data = vif_data[vif_data['Feature'] != 'intercept']

print("VIF for Original Features after Standardizing")
print(vif_data)
  



#split data into training and testing -> a túltanulás elkerülése érdekében
def splitdata():
    X = data.drop('heart_disease_yes', axis=1)  # Features (all columns except 'heart_disease')
    y = data['heart_disease_yes']
    #80% train 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = splitdata()
#MODEL
#1. logisztikus regresszió
model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate and display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate and display confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# def visualize(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, reg_tp: str):
#     plt.scatter(x, y_true, label='Orig', color='red')
#     plt.plot(x, y_pred, label=reg_tp, color='blue')
#     plt.xticks(())
#     plt.yticks(())
#     plt.legend()
#     plt.show()

# visualize(np.arange(len(X_test)), y_test, y_pred, reg_tp="Logostic Regression")

# x = data['ST_slope_flat']  # Example: age as feature
# y = data['heart_disease_yes']  # Example: heart_disease_yes as target variable

# # Plot logistic regression fit using seaborn regplot
# plt.figure(figsize=(8, 6))
# sns.regplot(x=x, y=y, data=data, ci=None,
#             scatter_kws={'color': 'black', 'alpha': 0.5}, line_kws={'color': 'red'})

# # Customize plot labels and title
# plt.xlabel('Age')
# plt.ylabel('Heart Disease (0: No, 1: Yes)')
# plt.title('Logistic Regression Fit: Age vs. Heart Disease')
# plt.show()

#Modell knn, swm
#AUC kiértékelés

def evaluationAuc(y_orig,y_pred):
        return roc_auc_score(y_orig,y_pred)

auc_score = evaluationAuc(y_test,y_pred)
print(auc_score)

# együtthatók lekérdezése
X = data.drop('heart_disease_yes', axis=1)  # Jellemzők (összes oszlop kivéve a 'heart_disease' oszlopot)
y = data['heart_disease_yes']

coefficients = model.coef_[0]
intercept = model.intercept_
# Együtthatók kiíratása
for feature, coef in zip(X.columns, coefficients):
    print(f'{feature}: {coef}')

# Az együtthatók abszolút értékeinek kiíratása, rangsorolva
print('rangsor')
sorted_features = sorted(zip(X.columns, abs(coefficients)), key=lambda x: -x[1])
for feature, coef in sorted_features:
    print(f'{feature}: {coef}')

#TODO másik két modell + auc kiértékelés, megnézni még milyenek vannak
#együtthatók rangsorolása
#hiperparaméter tuningolás (logregnek milyen van?)
