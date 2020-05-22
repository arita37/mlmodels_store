
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/a463a24ea257f46bfcbd4006f805952aace8f2b1', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'a463a24ea257f46bfcbd4006f805952aace8f2b1', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/a463a24ea257f46bfcbd4006f805952aace8f2b1

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/a463a24ea257f46bfcbd4006f805952aace8f2b1

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/a463a24ea257f46bfcbd4006f805952aace8f2b1

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_gluon.gluon_automl', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae', 'model_keras.02_cnn', 'model_keras.nbeats', 'model_keras.namentity_crm_bilstm', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.keras_gan', 'model_keras.01_deepctr', 'model_keras.charcnn_zhang', 'model_keras.textcnn', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_classifier', 'model_tch.mlp', 'model_tch.torchhub', 'model_tch.pplm', 'model_tch.nbeats', 'model_tch.matchzoo_models', 'model_tch.textcnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google'] 

  Used ['model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_gluon.gluon_automl', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae', 'model_keras.02_cnn', 'model_keras.nbeats', 'model_keras.namentity_crm_bilstm', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.keras_gan', 'model_keras.01_deepctr', 'model_keras.charcnn_zhang', 'model_keras.textcnn', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_classifier', 'model_tch.mlp', 'model_tch.torchhub', 'model_tch.pplm', 'model_tch.nbeats', 'model_tch.matchzoo_models', 'model_tch.textcnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google'] 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_sklearn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 

  #### metrics   ##################################################### 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
{'roc_auc_score': 0.9642857142857143}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_sklearn/model.pkl'}
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=4, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
{'roc_auc_score': 1.0}

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f6ce9e5f550> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
None

  #### Get  metrics   ################################################ 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  ############ Model fit   ########################################## 
fit success None

  ############ Prediction############################################ 
None

  ############ Save/ Load ############################################ 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all  &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
deps.txt
error_list
log_benchmark
log_dataloader
log_import
log_json
log_jupyter
log_pullrequest
log_test_cli
log_testall
test_jupyter
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
Already up to date.
