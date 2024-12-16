import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import os

data = pd.read_csv('../heart_disease_data.csv')

X = data.drop(columns='target')
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(log_reg_model, file)

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

print("Performing SHAP Analysis...")

explainer_log_reg = shap.Explainer(log_reg_model, X_test)
shap_values_log_reg = explainer_log_reg(X_test)

shap.summary_plot(shap_values_log_reg, X_test, feature_names=data.columns[:-1])

explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test)

shap.summary_plot(shap_values_rf[1], X_test, feature_names=data.columns[:-1])  

print("Performing LIME Analysis...")

lime_explainer = LimeTabularExplainer(X_train, feature_names=data.columns[:-1], class_names=['No Disease', 'Disease'], discretize_continuous=True)

instance = X_test[0] 
lime_exp = lime_explainer.explain_instance(instance, rf_model.predict_proba)

lime_exp.show_in_notebook()



def generate_shap_plots():
    explainer_log_reg = shap.Explainer(log_reg_model, X_test)
    shap_values_log_reg = explainer_log_reg(X_test)
    
    shap.summary_plot(shap_values_log_reg, X_test, feature_names=data.columns[:-1])
    plt.savefig(os.path.join('../static', 'shap_log_reg.png'))
    plt.close()

    explainer_rf = shap.TreeExplainer(rf_model)
    shap_values_rf = explainer_rf.shap_values(X_test)
    
    shap.summary_plot(shap_values_rf[1], X_test, feature_names=data.columns[:-1])
    plt.savefig(os.path.join('../static', 'shap_rf.png'))
    plt.close()

def generate_lime_plots():
    lime_explainer = LimeTabularExplainer(X_train, feature_names=data.columns[:-1], class_names=['No Disease', 'Disease'], discretize_continuous=True)
    
    instance = X_test[0] 
    lime_exp = lime_explainer.explain_instance(instance, rf_model.predict_proba)
    
    lime_exp.save_to_file(os.path.join('../static', 'lime_exp.html'))
    
    

# generate_lime_plots()
# generate_shap_plots()