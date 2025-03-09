import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# 収録している関数

# ensemble_vote
# ensemble_stakking
# baseline_rf
# baseline_gbm

def ensemble_vote(train, test):
    # xyの用意
    x_train = train.drop(['Survived'], axis=1, inplace=False)
    y_train = train['Survived']
    x_test = test.drop(['PassengerId'], axis=1, inplace=False)
    #パラメーターの用意
    lgbm_params = {
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'num_leaves': 32,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'random_state': 42
        }
    num_round = 100
    # インスタンス化
    lgbm = lgb.LGBMClassifier(**lgbm_params, n_estimators=num_round)
    rf = RandomForestClassifier(n_estimators=num_round, random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('lgbm', lgbm)], voting='soft')

    ensemble.fit(x_train, y_train)
    y_pred = ensemble.predict(x_test)
    result = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})
    return result

# ----------------------------------------------------------------------------------

def ensemble_stakking(train, test):
     # xyの用意
    x_train = train.drop(['Survived'], axis=1, inplace=False)
    y_train = train['Survived']
    x_test = test.drop(['PassengerId'], axis=1, inplace=False)
    #パラメーターの用意
    lgbm_params = {
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'num_leaves': 32,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'random_state': 42
        }
    num_round = 100
    # インスタンス化
    lgbm = lgb.LGBMClassifier(**lgbm_params, n_estimators=num_round)
    rf = RandomForestClassifier(n_estimators=num_round, random_state=42)
    # ベースモデル学習
    rf.fit(x_train, y_train)
    lgbm.fit(x_train, y_train)
    # 各Xデータから、予測を出す
    train_preds_rf = rf.predict(x_train)
    train_preds_lgbm = lgbm.predict(x_train)
    test_preds_rf = rf.predict(x_test)
    test_preds_lgbm = lgbm.predict(x_test)
    # 予測からメタデータを出す
    meta_x_train = np.column_stack((train_preds_rf, train_preds_lgbm))
    meta_x_test = np.column_stack((test_preds_rf, test_preds_lgbm))
    # メタモデルの訓練
    meta_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    meta_model.fit(meta_x_train, y_train)
    # メタモデルの予想
    final_preds = meta_model.predict(meta_x_test)
    final_preds_binary = (final_preds >= 0.5).astype(int)
    result = pd.DataFrame({'PassengerId': test['PassengerId'].values, 'Survived': final_preds_binary})
    return result

# -----------------------------------------------------------------------------------

def baseline_rf(train, test):
    x_train = train.drop('Survived', axis=1, inplace=False)
    y_train = train['Survived']
    x_test = test.drop(['PassengerId'], axis=1, inplace=False)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)

    result_rf = pd.DataFrame({'PassengerId': test['PassengerId'].values, 'Survived': y_pred})
    
    return result_rf

# ---------------------------------------------------------------------

def baseline_gbm(train, test):
    x_train = train.drop(['Survived'], axis=1, inplace=False)
    y_train = train['Survived']
    x_test = test.drop(['PassengerId'], axis=1, inplace=False)

    train_data = lgb.Dataset(x_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'boosting_type': 'gbdt',
        'num_leaves': 32,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'random_state': 42
    }
    num_round = 100
    bst = lgb.train(params, train_data, num_round)
    y_pred = bst.predict(x_test)

    y_pred_binary = (y_pred >= 0.5).astype(int)
    result_gbm = pd.DataFrame({'PassengerId': test['PassengerId'].values, 'Survived': y_pred_binary})
    return result_gbm
