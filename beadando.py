from matplotlib import scale
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

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
print(data.head())

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

# az új eloszlás
osztalyeloszlas(undersampled_data)

#PCA
X = data.drop('heart_disease_yes', axis=1)
y = data['heart_disease_yes']


def apply_PCA(X_train, X_test, target_variance=0.95):

    #standardizálás
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # pca elvégzése
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # komponens szám kiválasztása
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance_ratio >= target_variance) + 1

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return X_train_pca, X_test_pca, pca, scaler

# Calculate VIF for each principal component

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data
# vif_data = pd.DataFrame()
# vif_data['Feature'] = X_standardized.columns
# vif_data['VIF'] = [variance_inflation_factor(X_standardized.values, i) for i in range(X_standardized.shape[1])]

# Drop the intercept column after calculating VIF
#vif_data = vif_data[vif_data['Feature'] != 'intercept']


#split data into training and testing -> a túltanulás elkerülése érdekében
def splitdata(data):
    X = data.drop('heart_disease_yes', axis=1)  # Features (all columns except 'heart_disease')
    y = data['heart_disease_yes']
    #80% train 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test

#MODEL
#1. logisztikus regresszió
def train_logistic_regression(X_train, y_train, X_test, y_test, class_weight):
    # Initialize and train logistic regression model
    model = LogisticRegression(class_weight=class_weight)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
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

    y_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    print("AUC Score:", auc_score)
    print("F1", f1)
    
    return model

#2.modell
#KNN
def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    # Initialize and train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    y_prob = knn.predict_proba(X_test)[:, 1]
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    print(f"KNN Classifier (n_neighbors={n_neighbors})")
    print("Accuracy:", accuracy)
    print("AUC Score:", auc_score)
    
    # Generate and display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate and display confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - KNN')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return knn

def train_and_evaluate_svm(X_train, y_train, X_test, y_test, kernel='rbf', C=1.0):
    # Initialize and train SVM classifier
    svm_classifier = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    svm_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm_classifier.predict(X_test)
    y_prob = svm_classifier.predict_proba(X_test)[:, 1]  # Probability of positive class for AUC
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    # Print evaluation metrics
    print(f"SVM Classifier (kernel={kernel}, C={C})")
    print("Accuracy:", accuracy)
    print("AUC Score:", auc_score)
    
    # Generate and display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate and display confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - SVM')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return svm_classifier

def svm_hyperparameter_tuning(X_train, y_train, kernel_options=['rbf'], C_options=[1.0]):
    # Define the parameter grid
    param_grid = {
        'kernel': kernel_options,
        'C': C_options
    }

    # Initialize SVM classifier
    svm_classifier = SVC(probability=True, random_state=42)

    # Initialize GridSearchCV with cross-validation
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')

    # Perform grid search to find the best parameters
    grid_search.fit(X_train, y_train)

    # Print best parameters and best score
    print('Best Parameters:', grid_search.best_params_)
    print('Best Accuracy:', grid_search.best_score_)

    return grid_search.best_estimator_  # Ret

# együtthatók kiírása, sorrendbe rendezése

def egyutthatok(model):

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

#split data
X_train, X_test, y_train, y_test = splitdata(data)
#PCA
X_train_pca, X_test_pca, pca, scaler = apply_PCA(X_train, X_test, target_variance=0.95)

#standardizálás
X_standardized = pd.DataFrame(scaler.transform(data.drop('heart_disease_yes', axis=1)), columns=data.drop('heart_disease_yes', axis=1).columns)
vif_data = calculate_vif(X_standardized)

#VIF a pca után
print("VIF for Original Standardized Features:")
print(vif_data)

# Train logistic regression model using PCA-transformed features
custom_class_weights = {0: 0.25, 1: 0.75}
log_model = train_logistic_regression(X_train_pca, y_train, X_test_pca, y_test,class_weight=custom_class_weights)
knn_model = train_and_evaluate_knn(X_train_pca, y_train, X_test_pca, y_test, n_neighbors=18)
trained_svm = train_and_evaluate_svm(X_train, y_train, X_test, y_test, kernel='linear', C=10.0) # itt kíírni a többi pca-t is hogy látszódjon miért ezt választottam
#best_svm = svm_hyperparameter_tuning(X_train, y_train, 
                                     #kernel_options=['linear', 'rbf', 'poly'],
                                     #_options=[0.1, 1.0, 10.0])
#Hiperparaméter tuningolás a legjobb szomszéd szám megtalálásához: (gridsearch)
n_neighbors = [3, 5, 7, 9, 11, 13, 15, 16, 18, 20,25,30]
#for i in n_neighbors:
    #train_and_evaluate_knn(X_train_pca, y_train, X_test_pca, y_test, n_neighbors=i)

# együtthatók rangsorolása
egyutthatok(log_model)
