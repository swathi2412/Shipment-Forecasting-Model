from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np, joblib

def eval(model,Xt,yt):
    p=model.predict(Xt)
    return np.sqrt(mean_squared_error(yt,p)), r2_score(yt,p), p

def train_random_forest(Xtr,Xt,yt,Yt,random_state=42):
    grid={'n_estimators':[50,100,200],'max_depth':[5,10,15,None],'min_samples_split':[2,5,10]}
    rf=RandomForestRegressor(random_state=random_state)
    search=RandomizedSearchCV(rf,grid,n_iter=6,cv=3,n_jobs=-1,random_state=random_state)
    search.fit(Xtr,yt)
    best=search.best_estimator_
    rm,r2,p=eval(best,Xt,Yt)
    joblib.dump(best,'models/random_forest.joblib')
    return best,p

def train_xgboost(Xtr,Xt,yt,Yt,random_state=42):
    grid={'n_estimators':[100,200,300],'max_depth':[3,5,7],'learning_rate':[0.01,0.05,0.1]}
    xgb=XGBRegressor(objective='reg:squarederror',random_state=random_state,n_jobs=-1)
    search=RandomizedSearchCV(xgb,grid,n_iter=6,cv=3,n_jobs=-1,random_state=random_state)
    search.fit(Xtr,yt)
    best=search.best_estimator_
    rm,r2,p=eval(best,Xt,Yt)
    joblib.dump(best,'models/xgboost.joblib')
    return best,p
