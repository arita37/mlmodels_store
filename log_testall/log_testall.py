
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
[master 5f6021e] ml_store
 2 files changed, 144 insertions(+), 10566 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   4e9513b..5f6021e  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 0.77528533  1.47016034  1.03298378 -0.87000822  0.78655651  0.36919047
  -0.14319575  0.85328219 -0.13971173 -0.22241403]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f5d2abf3d30>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f5d44f72588> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 1.32720112e+00 -1.61198320e-01  6.02450901e-01 -2.86384915e-01
  -5.78962302e-01 -8.70887650e-01  1.37975819e+00  5.01429590e-01
  -4.78614074e-01 -8.92646674e-01]
 [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
   1.47073986e+00 -1.46140360e+00]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
  -2.28754171e-01  7.43356544e-01]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]]
None

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  ############ Model fit   ########################################## 
fit success None

  ############ Prediction############################################ 
[[ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
   0.96622086  0.4359862  -0.68748739  3.32107876]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]]
None

  ############ Save/ Load ############################################ 

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
Already up to date.
[master 3e11bf8] ml_store
 1 file changed, 273 insertions(+)
To github.com:arita37/mlmodels_store.git
   5f6021e..3e11bf8  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
    test(data_path = "model_fb/fbprophet.json", choice="json" )
TypeError: test() got an unexpected keyword argument 'choice'

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
Already up to date.
[master d2aabd1] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   3e11bf8..d2aabd1  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluonts_model.py 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU
INFO:root:Using CPU

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'deepar', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'use_feat_static_real': False, 'scaling': True, 'num_parallel_samples': 100}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepar/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 
INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:numexpr.utils:NumExpr defaulting to 2 threads.
INFO:root:Number of parameters in DeepARTrainingNetwork: 26844
100%|██████████| 10/10 [00:02<00:00,  3.94it/s, avg_epoch_loss=5.22]
INFO:root:Epoch[0] Elapsed time 2.540 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.217216
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.21721568107605 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f76659eba58>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f76659eba58>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['prediction_net-network.json', 'input_transform.json', 'parameters.json', 'prediction_net-0000.params', 'version.json', 'glutonts_model_pars.pkl', 'type.txt'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]WARNING:root:multiple 5 does not divide base seasonality 1.Falling back to seasonality 1
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 121.21it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1016.2769368489584,
    "abs_error": 360.6210632324219,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.3894570696824045,
    "sMAPE": 0.5032007754914432,
    "MSIS": 95.57828440495972,
    "QuantileLoss[0.5]": 360.62105560302734,
    "Coverage[0.5]": 1.0,
    "RMSE": 31.87909874587044,
    "NRMSE": 0.6711389209656935,
    "ND": 0.632668531986705,
    "wQuantileLoss[0.5]": 0.6326685186018024,
    "mean_wQuantileLoss": 0.6326685186018024,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'deepfactor', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'num_hidden_global': 50, 'num_layers_global': 1, 'num_factors': 10, 'num_hidden_local': 5, 'num_layers_local': 1, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'embedding_dimension': 10}, '_comment': {'distr_output': 'StudentTOutput()', 'cardinality': 'List[int] = list([1])', 'context_length': 'None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepfactor/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepFactorTrainingNetwork: 12466
100%|██████████| 10/10 [00:01<00:00,  8.31it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.204 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f763a9c5cf8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f763a9c5cf8>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['prediction_net-network.json', 'input_transform.json', 'parameters.json', 'prediction_net-0000.params', 'version.json', 'glutonts_model_pars.pkl', 'type.txt'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 168.79it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 2262.8567708333335,
    "abs_error": 552.1011962890625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 3.6581948231980244,
    "sMAPE": 1.8700508550675963,
    "MSIS": 146.3277993985751,
    "QuantileLoss[0.5]": 552.1012096405029,
    "Coverage[0.5]": 0.0,
    "RMSE": 47.5694941200065,
    "NRMSE": 1.0014630341054,
    "ND": 0.9685985899808114,
    "wQuantileLoss[0.5]": 0.9685986134043911,
    "mean_wQuantileLoss": 0.9685986134043911,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'transformer', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'dropout_rate': 0.1, 'model_dim': 32, 'inner_ff_dim_scale': 4, 'pre_seq': 'dn', 'post_seq': 'drn', 'act_type': 'softrelu', 'num_heads': 8, 'scaling': True, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False}, '_comment': {'cardinality': 'List[int] = list([1])', 'context_length': 'None', 'distr_output': 'DistributionOutput = StudentTOutput()', 'lags_seq': 'Optional[List[int]] = None', 'time_features': 'Optional[List[TimeFeature]] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_transformer/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in TransformerTrainingNetwork: 33911
100%|██████████| 10/10 [00:01<00:00,  5.52it/s, avg_epoch_loss=5.18]
INFO:root:Epoch[0] Elapsed time 1.813 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.178489
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.17848916053772 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f76380bc860>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f76380bc860>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['prediction_net-network.json', 'input_transform.json', 'parameters.json', 'prediction_net-0000.params', 'version.json', 'glutonts_model_pars.pkl', 'type.txt'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 146.64it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 413.1896158854167,
    "abs_error": 218.49172973632812,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.4477152377234432,
    "sMAPE": 0.34283908158131843,
    "MSIS": 57.908614361928315,
    "QuantileLoss[0.5]": 218.4917335510254,
    "Coverage[0.5]": 0.9166666666666666,
    "RMSE": 20.327066091431313,
    "NRMSE": 0.4279382335038171,
    "ND": 0.3833188240988213,
    "wQuantileLoss[0.5]": 0.3833188307912726,
    "mean_wQuantileLoss": 0.3833188307912726,
    "MAE_Coverage": 0.41666666666666663
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'wavenet', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'embedding_dimension': 20, 'num_parallel_samples': 100, 'num_bins': 1024, 'hybridize_prediction_net': False, 'n_residue': 24, 'n_skip': 32, 'n_stacks': 1, 'temperature': 1.0, 'act_type': 'elu'}, '_comment': {'cardinality': 'List[int] = [1]', 'context_length': 'None', 'seasonality': 'Optional[int] = None', 'dilation_depth': 'Optional[int] = None', 'train_window_length': 'Optional[int] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_wavenet/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024
INFO:root:using training windows of length = 12
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in WaveNet: 97636
 30%|███       | 3/10 [00:12<00:29,  4.17s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:23<00:16,  4.02s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:34<00:03,  3.90s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:38<00:00,  3.80s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 38.041 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.866675
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.866674566268921 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f763aa03cf8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f763aa03cf8>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['prediction_net-network.json', 'input_transform.json', 'parameters.json', 'prediction_net-0000.params', 'version.json', 'glutonts_model_pars.pkl', 'type.txt'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 165.35it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 52747.28125,
    "abs_error": 2698.045166015625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.8771118871208,
    "sMAPE": 1.4090718632897197,
    "MSIS": 715.0844625435238,
    "QuantileLoss[0.5]": 2698.0450744628906,
    "Coverage[0.5]": 1.0,
    "RMSE": 229.66776275742313,
    "NRMSE": 4.835110794893119,
    "ND": 4.733412571957237,
    "wQuantileLoss[0.5]": 4.733412411338405,
    "mean_wQuantileLoss": 4.733412411338405,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'feedforward', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'batch_normalization': False, 'mean_scaling': True, 'num_parallel_samples': 100}, '_comment': {'num_hidden_dimensions': 'Optional[List[int]] = None', 'context_length': 'Optional[int] = None', 'distr_output': 'DistributionOutput = StudentTOutput()'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_feedforward/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in SimpleFeedForwardTrainingNetwork: 20323
100%|██████████| 10/10 [00:00<00:00, 67.19it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 0.150 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.248065
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.248065185546875 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f76237e0780>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f76237e0780>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['prediction_net-network.json', 'input_transform.json', 'parameters.json', 'prediction_net-0000.params', 'version.json', 'glutonts_model_pars.pkl', 'type.txt'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 152.21it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 337.157470703125,
    "abs_error": 184.6112823486328,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2232250933905402,
    "sMAPE": 0.31918882646376456,
    "MSIS": 48.929006162116906,
    "QuantileLoss[0.5]": 184.6112823486328,
    "Coverage[0.5]": 0.5833333333333334,
    "RMSE": 18.36184823766728,
    "NRMSE": 0.3865652260561533,
    "ND": 0.32387944271689967,
    "wQuantileLoss[0.5]": 0.32387944271689967,
    "mean_wQuantileLoss": 0.32387944271689967,
    "MAE_Coverage": 0.08333333333333337
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'gp_forecaster', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': 2, 'max_iter_jitter': 10, 'jitter_method': 'iter', 'sample_noise': True, 'num_parallel_samples': 100}, '_comment': {'context_length': 'Optional[int] = None', 'kernel_output': 'KernelOutput = RBFKernelOutput()', 'dtype': 'DType = np.float64', 'time_features': 'Optional[List[TimeFeature]] = None'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_gpforecaster/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in GaussianProcessTrainingNetwork: 14
100%|██████████| 10/10 [00:01<00:00,  8.24it/s, avg_epoch_loss=123]
INFO:root:Epoch[0] Elapsed time 1.214 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=122.866774
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 122.86677375545166 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f76380d0320>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f76380d0320>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['prediction_net-network.json', 'input_transform.json', 'parameters.json', 'prediction_net-0000.params', 'version.json', 'glutonts_model_pars.pkl', 'type.txt'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 167.84it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 2213.317910911945,
    "abs_error": 549.0071371123157,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 3.6376937423470475,
    "sMAPE": 1.8441387313887985,
    "MSIS": 145.5077496938819,
    "QuantileLoss[0.5]": 549.0071371123157,
    "Coverage[0.5]": 0.0,
    "RMSE": 47.04591279709583,
    "NRMSE": 0.9904402694125438,
    "ND": 0.9631704159865188,
    "wQuantileLoss[0.5]": 0.9631704159865188,
    "mean_wQuantileLoss": 0.9631704159865188,
    "MAE_Coverage": 0.5
}

  #### Plot   ####################################################### 

  #### Loading params   ############################################## 

  model_gluon.gluonts_model 
{'model_name': 'deepstate', 'model_pars': {'freq': '5min', 'prediction_length': 12, 'cardinality': [1], 'add_trend': False, 'num_periods_to_train': 4, 'num_layers': 2, 'num_cells': 40, 'cell_type': 'lstm', 'num_parallel_samples': 100, 'dropout_rate': 0.1, 'use_feat_dynamic_real': False, 'use_feat_static_cat': False, 'scaling': True}, '_comment': {'past_length': 'Optional[int] = None', 'time_features': 'Optional[List[TimeFeature]] = None', 'noise_std_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'prior_cov_bounds': 'ParameterBounds = ParameterBounds(1e-6, 1.0)', 'innovation_bounds': 'ParameterBounds = ParameterBounds(1e-6, 0.01)', 'embedding_dimension': 'Optional[List[int]] = None', 'issm: Optional[ISSM]': 'None', 'cardinality': 'List[int]'}} {'train': True, 'dt_source': 'https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv', 'train_data_path': 'dataset/timeseries/train_deepar.csv', 'test_data_path': 'dataset/timeseries/train_deepar.csv', 'prediction_length': 12, 'freq': '5min', 'start': '2015-02-26 21:42:53', 'col_date': 'timestamp', 'col_ytarget': ['value'], 'num_series': 1, 'cols_cat': [], 'cols_num': []} {'num_samples': 100, 'compute_pars': {'batch_size': 32, 'clip_gradient': 100, 'epochs': 1, 'init': 'xavier', 'learning_rate': 0.001, 'learning_rate_decay_factor': 0.5, 'hybridize': False, 'num_batches_per_epoch': 10, 'minimum_learning_rate': 5e-05, 'patience': 10, 'weight_decay': 1e-08}} {'path': 'ztest/model_gluon/gluonts_deepstate/', 'plot_prob': True, 'quantiles': [0.5]}

  #### Loading dataset   ############################################# 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### Model init, fit   ############################################# 

INFO:root:Using CPU
INFO:root:Start model training
INFO:root:Epoch[0] Learning rate is 0.001
  0%|          | 0/10 [00:00<?, ?it/s]INFO:root:Number of parameters in DeepStateTrainingNetwork: 28054
 10%|█         | 1/10 [01:55<17:23, 115.92s/it, avg_epoch_loss=0.703] 20%|██        | 2/10 [04:57<18:05, 135.63s/it, avg_epoch_loss=0.686] 30%|███       | 3/10 [08:31<18:32, 159.00s/it, avg_epoch_loss=0.669] 40%|████      | 4/10 [11:51<17:09, 171.53s/it, avg_epoch_loss=0.652] 50%|█████     | 5/10 [15:28<15:25, 185.08s/it, avg_epoch_loss=0.634] 60%|██████    | 6/10 [18:42<12:31, 187.82s/it, avg_epoch_loss=0.617] 70%|███████   | 7/10 [22:04<09:35, 191.98s/it, avg_epoch_loss=0.599] 80%|████████  | 8/10 [25:33<06:34, 197.24s/it, avg_epoch_loss=0.581] 90%|█████████ | 9/10 [28:43<03:14, 194.80s/it, avg_epoch_loss=0.563]100%|██████████| 10/10 [31:54<00:00, 193.76s/it, avg_epoch_loss=0.546]100%|██████████| 10/10 [31:54<00:00, 191.45s/it, avg_epoch_loss=0.546]
INFO:root:Epoch[0] Elapsed time 1914.499 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.546187
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5461866021156311 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f76380d0160>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f76380d0160>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['prediction_net-network.json', 'input_transform.json', 'parameters.json', 'prediction_net-0000.params', 'version.json', 'glutonts_model_pars.pkl', 'type.txt'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 16.66it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 144.20114135742188,
    "abs_error": 105.91581726074219,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.7017928905105525,
    "sMAPE": 0.1842843903726714,
    "MSIS": 28.071716429253865,
    "QuantileLoss[0.5]": 105.91581344604492,
    "Coverage[0.5]": 0.3333333333333333,
    "RMSE": 12.00837796529664,
    "NRMSE": 0.2528079571641398,
    "ND": 0.18581722326445999,
    "wQuantileLoss[0.5]": 0.18581721657200864,
    "mean_wQuantileLoss": 0.18581721657200864,
    "MAE_Coverage": 0.16666666666666669
}

  #### Plot   ####################################################### 


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
From github.com:arita37/mlmodels_store
   d2aabd1..cb17e5e  master     -> origin/master
Updating d2aabd1..cb17e5e
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |   2 +-
 error_list/20200522/list_log_testall_20200522.md   | 379 +--------------------
 2 files changed, 3 insertions(+), 378 deletions(-)
[master f5bdc1c] ml_store
 1 file changed, 506 insertions(+)
To github.com:arita37/mlmodels_store.git
   cb17e5e..f5bdc1c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//gluon_automl.py 

  #### Loading params   ############################################## 

  #### Model params   ################################################ 

  #### Loading dataset   ############################################# 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mxnet/optimizer/optimizer.py:167: UserWarning: WARNING: New optimizer gluonnlp.optimizer.lamb.LAMB is overriding existing optimizer mxnet.optimizer.optimizer.LAMB
  Optimizer.opt_registry[name].__name__))
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv | Columns = 15 / 15 | Rows = 39073 -> 39073

  #### Model init, fit   ############################################# 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv | Columns = 15 / 15 | Rows = 39073 -> 39073
Warning: `hyperparameter_tune=True` is currently experimental and may cause the process to hang. Setting `auto_stack=True` instead is recommended to achieve maximum quality models.
Beginning AutoGluon training ... Time limit = 120s
AutoGluon will save models to dataset/
Train Data Rows:    39073
Train Data Columns: 15
Preprocessing data ...
Here are the first 10 unique label values in your data:  [' Tech-support' ' Transport-moving' ' Other-service' ' ?'
 ' Handlers-cleaners' ' Sales' ' Craft-repair' ' Adm-clerical'
 ' Exec-managerial' ' Prof-specialty']
AutoGluon infers your prediction problem is: multiclass  (because dtype of label-column == object)
If this is wrong, please specify `problem_type` argument in fit() instead (You may specify problem_type as one of: ['binary', 'multiclass', 'regression'])

Feature Generator processed 39073 data points with 14 features
Original Features:
	int features: 6
	object features: 8
Generated Features:
	int features: 0
All Features:
	int features: 6
	object features: 8
	Data preprocessing and feature engineering runtime = 0.23s ...
AutoGluon will gauge predictive performance using evaluation metric: accuracy
To change this, specify the eval_metric argument of fit()
AutoGluon will early stop models using evaluation metric: accuracy
Saving dataset/learner.pkl
Beginning hyperparameter tuning for Gradient Boosting Model...
Hyperparameter search space for Gradient Boosting Model: 
num_leaves:   Int: lower=26, upper=66
learning_rate:   Real: lower=0.005, upper=0.2
feature_fraction:   Real: lower=0.75, upper=1.0
min_data_in_leaf:   Int: lower=2, upper=30
Starting Experiments
Num of Finished Tasks is 0
Num of Pending Tasks is 5
  0%|          | 0/5 [00:00<?, ?it/s]Saving dataset/models/LightGBMClassifier/trial_0_model.pkl
Finished Task with config: {'feature_fraction': 1.0, 'learning_rate': 0.1, 'min_data_in_leaf': 20, 'num_leaves': 36} and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00learning_rateq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00learning_rateq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3908
 40%|████      | 2/5 [00:20<00:30, 10.27s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.7607865171096628, 'learning_rate': 0.06463994910008149, 'min_data_in_leaf': 26, 'num_leaves': 66} and reward: 0.3922
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8X\\\xf7G(\xeeX\r\x00\x00\x00learning_rateq\x02G?\xb0\x8c>cfcmX\x10\x00\x00\x00min_data_in_leafq\x03K\x1aX\n\x00\x00\x00num_leavesq\x04KBu.' and reward: 0.3922
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8X\\\xf7G(\xeeX\r\x00\x00\x00learning_rateq\x02G?\xb0\x8c>cfcmX\x10\x00\x00\x00min_data_in_leafq\x03K\x1aX\n\x00\x00\x00num_leavesq\x04KBu.' and reward: 0.3922
 60%|██████    | 3/5 [00:51<00:33, 16.55s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8007451084635623, 'learning_rate': 0.013289485913324508, 'min_data_in_leaf': 7, 'num_leaves': 57} and reward: 0.3904
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x9f\xb44\xa9\x0e\x1dX\r\x00\x00\x00learning_rateq\x02G?\x8b7\x84\x9b\x06\xec\xdbX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3904
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x9f\xb44\xa9\x0e\x1dX\r\x00\x00\x00learning_rateq\x02G?\x8b7\x84\x9b\x06\xec\xdbX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3904
 80%|████████  | 4/5 [01:20<00:20, 20.12s/it] 80%|████████  | 4/5 [01:20<00:20, 20.05s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.7734597403946025, 'learning_rate': 0.11384572037204098, 'min_data_in_leaf': 9, 'num_leaves': 57} and reward: 0.387
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xc0.\xa48\x8f\x19X\r\x00\x00\x00learning_rateq\x02G?\xbd$\xfe=\xc9\x98\x17X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.387
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\xc0.\xa48\x8f\x19X\r\x00\x00\x00learning_rateq\x02G?\xbd$\xfe=\xc9\x98\x17X\x10\x00\x00\x00min_data_in_leafq\x03K\tX\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.387
Time for Gradient Boosting hyperparameter optimization: 108.02683138847351
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.7607865171096628, 'learning_rate': 0.06463994910008149, 'min_data_in_leaf': 26, 'num_leaves': 66}
Saving dataset/models/trainer.pkl
Beginning hyperparameter tuning for Neural Network...
Hyperparameter search space for Neural Network: 
network_type:   Categorical['widedeep', 'feedforward']
layers:   Categorical[[100], [1000], [200, 100], [300, 200, 100]]
activation:   Categorical['relu', 'softrelu', 'tanh']
embedding_size_factor:   Real: lower=0.5, upper=1.5
use_batchnorm:   Categorical[True, False]
dropout_prob:   Real: lower=0.0, upper=0.5
learning_rate:   Real: lower=0.0001, upper=0.01
weight_decay:   Real: lower=1e-12, upper=0.1
AutoGluon Neural Network infers features are of the following types:
{
    "continuous": [
        "age",
        "education-num",
        "hours-per-week"
    ],
    "skewed": [
        "fnlwgt",
        "capital-gain",
        "capital-loss"
    ],
    "onehot": [
        "sex",
        "class"
    ],
    "embed": [
        "workclass",
        "education",
        "marital-status",
        "relationship",
        "race",
        "native-country"
    ],
    "language": []
}


Saving dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Starting Experiments
Num of Finished Tasks is 0
Num of Pending Tasks is 5
  0%|          | 0/5 [00:00<?, ?it/s]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:48<01:13, 24.48s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.44910624332309707, 'embedding_size_factor': 0.730676243893909, 'layers.choice': 0, 'learning_rate': 0.000669662039209022, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.010287747595947843} and reward: 0.3498
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdc\xbe(\x1c\xe0"\xbdX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7a\xb3%o\xa1tX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?E\xf1\x88Ge/XX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x85\x11\xbe\x1b\xca\xf46u.' and reward: 0.3498
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xdc\xbe(\x1c\xe0"\xbdX\x15\x00\x00\x00embedding_size_factorq\x03G?\xe7a\xb3%o\xa1tX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?E\xf1\x88Ge/XX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?\x85\x11\xbe\x1b\xca\xf46u.' and reward: 0.3498
 60%|██████    | 3/5 [01:38<01:03, 31.99s/it] 60%|██████    | 3/5 [01:38<01:05, 32.83s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.4197813603940682, 'embedding_size_factor': 0.981082908954347, 'layers.choice': 2, 'learning_rate': 0.0035898488558898795, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1.464063453670614e-05} and reward: 0.3754
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xda\xdd\xb2\xa3\x979\xf6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xefe\x07\xfc\x13\xf3oX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?mhum\xe0L X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xee\xb4!}'\xf4lu." and reward: 0.3754
Finished Task with config: b"\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xda\xdd\xb2\xa3\x979\xf6X\x15\x00\x00\x00embedding_size_factorq\x03G?\xefe\x07\xfc\x13\xf3oX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?mhum\xe0L X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xee\xb4!}'\xf4lu." and reward: 0.3754
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 150.20911979675293
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -143.28s of remaining time.
Ensemble size: 8
Ensemble weights: 
[0.125 0.25  0.125 0.25  0.    0.25  0.   ]
	0.3986	 = Validation accuracy score
	1.56s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 264.89s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f5c8d364c18>

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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   f5bdc1c..e4560bb  master     -> origin/master
Updating f5bdc1c..e4560bb
Fast-forward
 error_list/20200522/list_log_pullrequest_20200522.md | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
[master b979d7a] ml_store
 1 file changed, 286 insertions(+)
To github.com:arita37/mlmodels_store.git
   e4560bb..b979d7a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py", line 12, in <module>
    import autokeras as ak
ModuleNotFoundError: No module named 'autokeras'

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
Already up to date.
[master 02d1f22] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   b979d7a..02d1f22  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//armdn.py 

  #### Loading params   ############################################## 

  #### Model init   ################################################## 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_probability/python/distributions/mixture.py:154: Categorical.event_size (from tensorflow_probability.python.distributions.categorical) is deprecated and will be removed after 2019-05-19.
Instructions for updating:
The `event_size` property is deprecated.  Use `num_categories` instead.  They have the same value, but `event_size` is misnamed.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_ops.py:2509: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
LSTM_1 (LSTM)                (None, 12, 300)           362400    
_________________________________________________________________
LSTM_2 (LSTM)                (None, 12, 200)           400800    
_________________________________________________________________
LSTM_3 (LSTM)                (None, 12, 24)            21600     
_________________________________________________________________
LSTM_4 (LSTM)                (None, 12)                1776      
_________________________________________________________________
dense_1 (Dense)              (None, 10)                130       
_________________________________________________________________
mdn_1 (MDN)                  (None, 75)                825       
=================================================================
Total params: 787,531
Trainable params: 787,531
Non-trainable params: 0
_________________________________________________________________

  ### Model Fit ###################################################### 

  #### Loading dataset   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Epoch 1/10

13/13 [==============================] - 2s 129ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 10/10

13/13 [==============================] - 0s 4ms/step - loss: nan

  fitted metrics {'loss': [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]} 

  #### Predict   ##################################################### 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mdn/__init__.py:209: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.

[[nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan]]
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//armdn.py", line 380, in <module>
    test(pars_choice="json", data_path= "model_keras/armdn.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//armdn.py", line 354, in test
    y_pred, y_test = predict(model=model, model_pars=model_pars, data_pars=data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//armdn.py", line 170, in predict
    model.model_pars["n_mixes"], temp=1.0)
  File "<__array_function__ internals>", line 6, in apply_along_axis
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/shape_base.py", line 379, in apply_along_axis
    res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/mdn/__init__.py", line 237, in sample_from_output
    cov_matrix = np.identity(output_dim) * sig_vector
ValueError: operands could not be broadcast together with shapes (12,12) (0,) 

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
Already up to date.
[master 463cae7] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   02d1f22..463cae7  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textvae.py 

  #### Loading params   ############################################## 

  #### Path params   ################################################### 

  #### Model params   ################################################# 

  #### Loading dataset   ############################################# 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textvae.py", line 356, in <module>
    test(pars_choice="test01")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textvae.py", line 327, in test
    xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textvae.py", line 269, in get_dataset
    with codecs.open(data_pars["train_data_path"], encoding='utf-8') as f:
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/codecs.py", line 897, in open
    file = builtins.open(filename, mode, buffering)
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/quora/train.csv'

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
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
Already up to date.
[master 9991828] ml_store
 1 file changed, 52 insertions(+)
To github.com:arita37/mlmodels_store.git
   463cae7..9991828  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//02_cnn.py 

  ('#### Loading params   ##############################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/',) 

  ('#### Model params   ################################################',) 

  ('#### Loading dataset   #############################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz

    8192/11490434 [..............................] - ETA: 0s
  679936/11490434 [>.............................] - ETA: 0s
 2564096/11490434 [=====>........................] - ETA: 0s
 6078464/11490434 [==============>...............] - ETA: 0s
 9854976/11490434 [========================>.....] - ETA: 0s
11493376/11490434 [==============================] - 0s 0us/step

  ('#### Model init, fit   #############################################',) 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.


  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Train on 60000 samples, validate on 10000 samples
Epoch 1/1

   32/60000 [..............................] - ETA: 8:20 - loss: 2.3163 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:58 - loss: 2.2802 - categorical_accuracy: 0.1250
  128/60000 [..............................] - ETA: 3:13 - loss: 2.2708 - categorical_accuracy: 0.1328
  192/60000 [..............................] - ETA: 2:36 - loss: 2.1927 - categorical_accuracy: 0.1979
  256/60000 [..............................] - ETA: 2:18 - loss: 2.1492 - categorical_accuracy: 0.2031
  320/60000 [..............................] - ETA: 2:08 - loss: 2.0828 - categorical_accuracy: 0.2375
  384/60000 [..............................] - ETA: 2:01 - loss: 2.0454 - categorical_accuracy: 0.2578
  448/60000 [..............................] - ETA: 1:56 - loss: 2.0401 - categorical_accuracy: 0.2634
  512/60000 [..............................] - ETA: 1:52 - loss: 1.9773 - categorical_accuracy: 0.2949
  576/60000 [..............................] - ETA: 1:48 - loss: 1.9002 - categorical_accuracy: 0.3264
  640/60000 [..............................] - ETA: 1:46 - loss: 1.8653 - categorical_accuracy: 0.3406
  704/60000 [..............................] - ETA: 1:44 - loss: 1.8177 - categorical_accuracy: 0.3636
  768/60000 [..............................] - ETA: 1:41 - loss: 1.7777 - categorical_accuracy: 0.3828
  832/60000 [..............................] - ETA: 1:39 - loss: 1.7140 - categorical_accuracy: 0.4087
  896/60000 [..............................] - ETA: 1:38 - loss: 1.6681 - categorical_accuracy: 0.4241
  960/60000 [..............................] - ETA: 1:37 - loss: 1.6286 - categorical_accuracy: 0.4385
  992/60000 [..............................] - ETA: 1:37 - loss: 1.6087 - categorical_accuracy: 0.4466
 1056/60000 [..............................] - ETA: 1:37 - loss: 1.5514 - categorical_accuracy: 0.4688
 1120/60000 [..............................] - ETA: 1:35 - loss: 1.5030 - categorical_accuracy: 0.4866
 1184/60000 [..............................] - ETA: 1:34 - loss: 1.4767 - categorical_accuracy: 0.4983
 1248/60000 [..............................] - ETA: 1:34 - loss: 1.4336 - categorical_accuracy: 0.5144
 1312/60000 [..............................] - ETA: 1:33 - loss: 1.4133 - categorical_accuracy: 0.5198
 1376/60000 [..............................] - ETA: 1:32 - loss: 1.3792 - categorical_accuracy: 0.5334
 1440/60000 [..............................] - ETA: 1:31 - loss: 1.3522 - categorical_accuracy: 0.5417
 1504/60000 [..............................] - ETA: 1:31 - loss: 1.3330 - categorical_accuracy: 0.5485
 1568/60000 [..............................] - ETA: 1:30 - loss: 1.3095 - categorical_accuracy: 0.5587
 1632/60000 [..............................] - ETA: 1:30 - loss: 1.2851 - categorical_accuracy: 0.5674
 1696/60000 [..............................] - ETA: 1:30 - loss: 1.2621 - categorical_accuracy: 0.5749
 1760/60000 [..............................] - ETA: 1:29 - loss: 1.2390 - categorical_accuracy: 0.5818
 1824/60000 [..............................] - ETA: 1:29 - loss: 1.2152 - categorical_accuracy: 0.5910
 1888/60000 [..............................] - ETA: 1:28 - loss: 1.1944 - categorical_accuracy: 0.5964
 1952/60000 [..............................] - ETA: 1:28 - loss: 1.1718 - categorical_accuracy: 0.6055
 2016/60000 [>.............................] - ETA: 1:28 - loss: 1.1540 - categorical_accuracy: 0.6116
 2048/60000 [>.............................] - ETA: 1:28 - loss: 1.1406 - categorical_accuracy: 0.6167
 2112/60000 [>.............................] - ETA: 1:27 - loss: 1.1191 - categorical_accuracy: 0.6241
 2176/60000 [>.............................] - ETA: 1:27 - loss: 1.1059 - categorical_accuracy: 0.6268
 2240/60000 [>.............................] - ETA: 1:27 - loss: 1.0867 - categorical_accuracy: 0.6330
 2304/60000 [>.............................] - ETA: 1:27 - loss: 1.0756 - categorical_accuracy: 0.6380
 2368/60000 [>.............................] - ETA: 1:27 - loss: 1.0637 - categorical_accuracy: 0.6436
 2432/60000 [>.............................] - ETA: 1:26 - loss: 1.0454 - categorical_accuracy: 0.6505
 2496/60000 [>.............................] - ETA: 1:26 - loss: 1.0336 - categorical_accuracy: 0.6550
 2528/60000 [>.............................] - ETA: 1:26 - loss: 1.0251 - categorical_accuracy: 0.6574
 2592/60000 [>.............................] - ETA: 1:26 - loss: 1.0104 - categorical_accuracy: 0.6620
 2656/60000 [>.............................] - ETA: 1:26 - loss: 0.9918 - categorical_accuracy: 0.6679
 2720/60000 [>.............................] - ETA: 1:26 - loss: 0.9778 - categorical_accuracy: 0.6728
 2784/60000 [>.............................] - ETA: 1:26 - loss: 0.9638 - categorical_accuracy: 0.6782
 2848/60000 [>.............................] - ETA: 1:25 - loss: 0.9546 - categorical_accuracy: 0.6826
 2912/60000 [>.............................] - ETA: 1:25 - loss: 0.9436 - categorical_accuracy: 0.6868
 2976/60000 [>.............................] - ETA: 1:25 - loss: 0.9289 - categorical_accuracy: 0.6919
 3040/60000 [>.............................] - ETA: 1:25 - loss: 0.9185 - categorical_accuracy: 0.6961
 3104/60000 [>.............................] - ETA: 1:24 - loss: 0.9085 - categorical_accuracy: 0.6991
 3168/60000 [>.............................] - ETA: 1:24 - loss: 0.9035 - categorical_accuracy: 0.7001
 3232/60000 [>.............................] - ETA: 1:24 - loss: 0.8906 - categorical_accuracy: 0.7051
 3296/60000 [>.............................] - ETA: 1:24 - loss: 0.8788 - categorical_accuracy: 0.7087
 3360/60000 [>.............................] - ETA: 1:24 - loss: 0.8772 - categorical_accuracy: 0.7107
 3424/60000 [>.............................] - ETA: 1:23 - loss: 0.8714 - categorical_accuracy: 0.7135
 3488/60000 [>.............................] - ETA: 1:23 - loss: 0.8623 - categorical_accuracy: 0.7170
 3552/60000 [>.............................] - ETA: 1:23 - loss: 0.8544 - categorical_accuracy: 0.7193
 3616/60000 [>.............................] - ETA: 1:23 - loss: 0.8485 - categorical_accuracy: 0.7218
 3680/60000 [>.............................] - ETA: 1:23 - loss: 0.8395 - categorical_accuracy: 0.7258
 3744/60000 [>.............................] - ETA: 1:22 - loss: 0.8309 - categorical_accuracy: 0.7292
 3808/60000 [>.............................] - ETA: 1:22 - loss: 0.8245 - categorical_accuracy: 0.7311
 3872/60000 [>.............................] - ETA: 1:22 - loss: 0.8219 - categorical_accuracy: 0.7327
 3936/60000 [>.............................] - ETA: 1:22 - loss: 0.8139 - categorical_accuracy: 0.7363
 4000/60000 [=>............................] - ETA: 1:22 - loss: 0.8062 - categorical_accuracy: 0.7387
 4064/60000 [=>............................] - ETA: 1:22 - loss: 0.8003 - categorical_accuracy: 0.7411
 4128/60000 [=>............................] - ETA: 1:22 - loss: 0.7917 - categorical_accuracy: 0.7442
 4192/60000 [=>............................] - ETA: 1:21 - loss: 0.7851 - categorical_accuracy: 0.7462
 4256/60000 [=>............................] - ETA: 1:21 - loss: 0.7778 - categorical_accuracy: 0.7479
 4320/60000 [=>............................] - ETA: 1:21 - loss: 0.7716 - categorical_accuracy: 0.7500
 4384/60000 [=>............................] - ETA: 1:21 - loss: 0.7643 - categorical_accuracy: 0.7523
 4448/60000 [=>............................] - ETA: 1:21 - loss: 0.7558 - categorical_accuracy: 0.7554
 4512/60000 [=>............................] - ETA: 1:21 - loss: 0.7525 - categorical_accuracy: 0.7564
 4576/60000 [=>............................] - ETA: 1:21 - loss: 0.7475 - categorical_accuracy: 0.7581
 4640/60000 [=>............................] - ETA: 1:20 - loss: 0.7413 - categorical_accuracy: 0.7601
 4704/60000 [=>............................] - ETA: 1:20 - loss: 0.7369 - categorical_accuracy: 0.7619
 4768/60000 [=>............................] - ETA: 1:20 - loss: 0.7296 - categorical_accuracy: 0.7645
 4832/60000 [=>............................] - ETA: 1:20 - loss: 0.7277 - categorical_accuracy: 0.7653
 4896/60000 [=>............................] - ETA: 1:20 - loss: 0.7214 - categorical_accuracy: 0.7674
 4960/60000 [=>............................] - ETA: 1:20 - loss: 0.7165 - categorical_accuracy: 0.7696
 5024/60000 [=>............................] - ETA: 1:20 - loss: 0.7136 - categorical_accuracy: 0.7703
 5088/60000 [=>............................] - ETA: 1:19 - loss: 0.7081 - categorical_accuracy: 0.7722
 5152/60000 [=>............................] - ETA: 1:19 - loss: 0.7020 - categorical_accuracy: 0.7739
 5216/60000 [=>............................] - ETA: 1:19 - loss: 0.6961 - categorical_accuracy: 0.7753
 5280/60000 [=>............................] - ETA: 1:19 - loss: 0.6910 - categorical_accuracy: 0.7771
 5344/60000 [=>............................] - ETA: 1:19 - loss: 0.6845 - categorical_accuracy: 0.7792
 5408/60000 [=>............................] - ETA: 1:19 - loss: 0.6793 - categorical_accuracy: 0.7811
 5440/60000 [=>............................] - ETA: 1:19 - loss: 0.6776 - categorical_accuracy: 0.7818
 5504/60000 [=>............................] - ETA: 1:19 - loss: 0.6737 - categorical_accuracy: 0.7831
 5568/60000 [=>............................] - ETA: 1:19 - loss: 0.6717 - categorical_accuracy: 0.7841
 5632/60000 [=>............................] - ETA: 1:18 - loss: 0.6669 - categorical_accuracy: 0.7859
 5696/60000 [=>............................] - ETA: 1:18 - loss: 0.6623 - categorical_accuracy: 0.7872
 5760/60000 [=>............................] - ETA: 1:18 - loss: 0.6597 - categorical_accuracy: 0.7884
 5824/60000 [=>............................] - ETA: 1:18 - loss: 0.6551 - categorical_accuracy: 0.7897
 5888/60000 [=>............................] - ETA: 1:18 - loss: 0.6516 - categorical_accuracy: 0.7909
 5952/60000 [=>............................] - ETA: 1:18 - loss: 0.6470 - categorical_accuracy: 0.7925
 6016/60000 [==>...........................] - ETA: 1:18 - loss: 0.6442 - categorical_accuracy: 0.7936
 6080/60000 [==>...........................] - ETA: 1:17 - loss: 0.6398 - categorical_accuracy: 0.7951
 6144/60000 [==>...........................] - ETA: 1:17 - loss: 0.6358 - categorical_accuracy: 0.7967
 6208/60000 [==>...........................] - ETA: 1:17 - loss: 0.6326 - categorical_accuracy: 0.7980
 6272/60000 [==>...........................] - ETA: 1:17 - loss: 0.6292 - categorical_accuracy: 0.7991
 6336/60000 [==>...........................] - ETA: 1:17 - loss: 0.6244 - categorical_accuracy: 0.8007
 6400/60000 [==>...........................] - ETA: 1:17 - loss: 0.6198 - categorical_accuracy: 0.8020
 6464/60000 [==>...........................] - ETA: 1:17 - loss: 0.6147 - categorical_accuracy: 0.8037
 6528/60000 [==>...........................] - ETA: 1:17 - loss: 0.6119 - categorical_accuracy: 0.8045
 6592/60000 [==>...........................] - ETA: 1:17 - loss: 0.6107 - categorical_accuracy: 0.8051
 6656/60000 [==>...........................] - ETA: 1:16 - loss: 0.6063 - categorical_accuracy: 0.8065
 6720/60000 [==>...........................] - ETA: 1:16 - loss: 0.6024 - categorical_accuracy: 0.8077
 6784/60000 [==>...........................] - ETA: 1:16 - loss: 0.5995 - categorical_accuracy: 0.8088
 6848/60000 [==>...........................] - ETA: 1:16 - loss: 0.5958 - categorical_accuracy: 0.8102
 6912/60000 [==>...........................] - ETA: 1:16 - loss: 0.5915 - categorical_accuracy: 0.8115
 6976/60000 [==>...........................] - ETA: 1:16 - loss: 0.5877 - categorical_accuracy: 0.8125
 7040/60000 [==>...........................] - ETA: 1:16 - loss: 0.5851 - categorical_accuracy: 0.8134
 7104/60000 [==>...........................] - ETA: 1:16 - loss: 0.5810 - categorical_accuracy: 0.8148
 7168/60000 [==>...........................] - ETA: 1:15 - loss: 0.5795 - categorical_accuracy: 0.8147
 7232/60000 [==>...........................] - ETA: 1:15 - loss: 0.5764 - categorical_accuracy: 0.8157
 7296/60000 [==>...........................] - ETA: 1:15 - loss: 0.5739 - categorical_accuracy: 0.8162
 7360/60000 [==>...........................] - ETA: 1:15 - loss: 0.5709 - categorical_accuracy: 0.8170
 7424/60000 [==>...........................] - ETA: 1:15 - loss: 0.5693 - categorical_accuracy: 0.8178
 7488/60000 [==>...........................] - ETA: 1:15 - loss: 0.5667 - categorical_accuracy: 0.8188
 7552/60000 [==>...........................] - ETA: 1:15 - loss: 0.5642 - categorical_accuracy: 0.8194
 7616/60000 [==>...........................] - ETA: 1:15 - loss: 0.5602 - categorical_accuracy: 0.8208
 7648/60000 [==>...........................] - ETA: 1:15 - loss: 0.5589 - categorical_accuracy: 0.8211
 7712/60000 [==>...........................] - ETA: 1:15 - loss: 0.5553 - categorical_accuracy: 0.8224
 7744/60000 [==>...........................] - ETA: 1:15 - loss: 0.5542 - categorical_accuracy: 0.8228
 7808/60000 [==>...........................] - ETA: 1:15 - loss: 0.5506 - categorical_accuracy: 0.8240
 7872/60000 [==>...........................] - ETA: 1:14 - loss: 0.5475 - categorical_accuracy: 0.8249
 7936/60000 [==>...........................] - ETA: 1:14 - loss: 0.5449 - categorical_accuracy: 0.8259
 8000/60000 [===>..........................] - ETA: 1:14 - loss: 0.5424 - categorical_accuracy: 0.8265
 8064/60000 [===>..........................] - ETA: 1:14 - loss: 0.5407 - categorical_accuracy: 0.8271
 8128/60000 [===>..........................] - ETA: 1:14 - loss: 0.5379 - categorical_accuracy: 0.8280
 8192/60000 [===>..........................] - ETA: 1:14 - loss: 0.5363 - categorical_accuracy: 0.8287
 8256/60000 [===>..........................] - ETA: 1:14 - loss: 0.5340 - categorical_accuracy: 0.8296
 8320/60000 [===>..........................] - ETA: 1:14 - loss: 0.5318 - categorical_accuracy: 0.8303
 8384/60000 [===>..........................] - ETA: 1:14 - loss: 0.5297 - categorical_accuracy: 0.8311
 8448/60000 [===>..........................] - ETA: 1:13 - loss: 0.5272 - categorical_accuracy: 0.8319
 8512/60000 [===>..........................] - ETA: 1:13 - loss: 0.5252 - categorical_accuracy: 0.8324
 8576/60000 [===>..........................] - ETA: 1:13 - loss: 0.5237 - categorical_accuracy: 0.8327
 8640/60000 [===>..........................] - ETA: 1:13 - loss: 0.5209 - categorical_accuracy: 0.8334
 8704/60000 [===>..........................] - ETA: 1:13 - loss: 0.5196 - categorical_accuracy: 0.8339
 8768/60000 [===>..........................] - ETA: 1:13 - loss: 0.5175 - categorical_accuracy: 0.8341
 8832/60000 [===>..........................] - ETA: 1:13 - loss: 0.5158 - categorical_accuracy: 0.8346
 8896/60000 [===>..........................] - ETA: 1:13 - loss: 0.5137 - categorical_accuracy: 0.8351
 8960/60000 [===>..........................] - ETA: 1:13 - loss: 0.5117 - categorical_accuracy: 0.8358
 9024/60000 [===>..........................] - ETA: 1:12 - loss: 0.5095 - categorical_accuracy: 0.8367
 9088/60000 [===>..........................] - ETA: 1:12 - loss: 0.5074 - categorical_accuracy: 0.8376
 9152/60000 [===>..........................] - ETA: 1:12 - loss: 0.5056 - categorical_accuracy: 0.8384
 9216/60000 [===>..........................] - ETA: 1:12 - loss: 0.5033 - categorical_accuracy: 0.8391
 9248/60000 [===>..........................] - ETA: 1:12 - loss: 0.5021 - categorical_accuracy: 0.8394
 9312/60000 [===>..........................] - ETA: 1:12 - loss: 0.5005 - categorical_accuracy: 0.8399
 9376/60000 [===>..........................] - ETA: 1:12 - loss: 0.4981 - categorical_accuracy: 0.8408
 9440/60000 [===>..........................] - ETA: 1:12 - loss: 0.4967 - categorical_accuracy: 0.8412
 9504/60000 [===>..........................] - ETA: 1:12 - loss: 0.4944 - categorical_accuracy: 0.8419
 9568/60000 [===>..........................] - ETA: 1:11 - loss: 0.4920 - categorical_accuracy: 0.8427
 9632/60000 [===>..........................] - ETA: 1:11 - loss: 0.4899 - categorical_accuracy: 0.8434
 9696/60000 [===>..........................] - ETA: 1:11 - loss: 0.4889 - categorical_accuracy: 0.8440
 9760/60000 [===>..........................] - ETA: 1:11 - loss: 0.4873 - categorical_accuracy: 0.8446
 9824/60000 [===>..........................] - ETA: 1:11 - loss: 0.4854 - categorical_accuracy: 0.8452
 9856/60000 [===>..........................] - ETA: 1:11 - loss: 0.4845 - categorical_accuracy: 0.8455
 9920/60000 [===>..........................] - ETA: 1:11 - loss: 0.4826 - categorical_accuracy: 0.8461
 9984/60000 [===>..........................] - ETA: 1:11 - loss: 0.4804 - categorical_accuracy: 0.8468
10048/60000 [====>.........................] - ETA: 1:11 - loss: 0.4795 - categorical_accuracy: 0.8472
10112/60000 [====>.........................] - ETA: 1:11 - loss: 0.4778 - categorical_accuracy: 0.8478
10176/60000 [====>.........................] - ETA: 1:11 - loss: 0.4760 - categorical_accuracy: 0.8485
10240/60000 [====>.........................] - ETA: 1:10 - loss: 0.4735 - categorical_accuracy: 0.8493
10304/60000 [====>.........................] - ETA: 1:10 - loss: 0.4713 - categorical_accuracy: 0.8501
10368/60000 [====>.........................] - ETA: 1:10 - loss: 0.4700 - categorical_accuracy: 0.8506
10432/60000 [====>.........................] - ETA: 1:10 - loss: 0.4691 - categorical_accuracy: 0.8510
10496/60000 [====>.........................] - ETA: 1:10 - loss: 0.4670 - categorical_accuracy: 0.8517
10560/60000 [====>.........................] - ETA: 1:10 - loss: 0.4655 - categorical_accuracy: 0.8524
10624/60000 [====>.........................] - ETA: 1:10 - loss: 0.4639 - categorical_accuracy: 0.8531
10688/60000 [====>.........................] - ETA: 1:10 - loss: 0.4616 - categorical_accuracy: 0.8538
10752/60000 [====>.........................] - ETA: 1:10 - loss: 0.4595 - categorical_accuracy: 0.8544
10816/60000 [====>.........................] - ETA: 1:10 - loss: 0.4578 - categorical_accuracy: 0.8548
10880/60000 [====>.........................] - ETA: 1:09 - loss: 0.4569 - categorical_accuracy: 0.8551
10944/60000 [====>.........................] - ETA: 1:09 - loss: 0.4554 - categorical_accuracy: 0.8558
11008/60000 [====>.........................] - ETA: 1:09 - loss: 0.4533 - categorical_accuracy: 0.8565
11072/60000 [====>.........................] - ETA: 1:09 - loss: 0.4513 - categorical_accuracy: 0.8569
11136/60000 [====>.........................] - ETA: 1:09 - loss: 0.4496 - categorical_accuracy: 0.8576
11200/60000 [====>.........................] - ETA: 1:09 - loss: 0.4478 - categorical_accuracy: 0.8582
11264/60000 [====>.........................] - ETA: 1:09 - loss: 0.4467 - categorical_accuracy: 0.8587
11328/60000 [====>.........................] - ETA: 1:09 - loss: 0.4448 - categorical_accuracy: 0.8594
11392/60000 [====>.........................] - ETA: 1:09 - loss: 0.4444 - categorical_accuracy: 0.8596
11456/60000 [====>.........................] - ETA: 1:09 - loss: 0.4430 - categorical_accuracy: 0.8599
11520/60000 [====>.........................] - ETA: 1:09 - loss: 0.4425 - categorical_accuracy: 0.8602
11584/60000 [====>.........................] - ETA: 1:08 - loss: 0.4405 - categorical_accuracy: 0.8608
11648/60000 [====>.........................] - ETA: 1:08 - loss: 0.4387 - categorical_accuracy: 0.8613
11712/60000 [====>.........................] - ETA: 1:08 - loss: 0.4373 - categorical_accuracy: 0.8618
11776/60000 [====>.........................] - ETA: 1:08 - loss: 0.4359 - categorical_accuracy: 0.8623
11840/60000 [====>.........................] - ETA: 1:08 - loss: 0.4340 - categorical_accuracy: 0.8628
11904/60000 [====>.........................] - ETA: 1:08 - loss: 0.4331 - categorical_accuracy: 0.8630
11968/60000 [====>.........................] - ETA: 1:08 - loss: 0.4316 - categorical_accuracy: 0.8636
12000/60000 [=====>........................] - ETA: 1:08 - loss: 0.4310 - categorical_accuracy: 0.8637
12064/60000 [=====>........................] - ETA: 1:08 - loss: 0.4300 - categorical_accuracy: 0.8642
12128/60000 [=====>........................] - ETA: 1:08 - loss: 0.4289 - categorical_accuracy: 0.8645
12192/60000 [=====>........................] - ETA: 1:07 - loss: 0.4278 - categorical_accuracy: 0.8649
12256/60000 [=====>........................] - ETA: 1:07 - loss: 0.4272 - categorical_accuracy: 0.8654
12288/60000 [=====>........................] - ETA: 1:07 - loss: 0.4263 - categorical_accuracy: 0.8657
12352/60000 [=====>........................] - ETA: 1:07 - loss: 0.4255 - categorical_accuracy: 0.8659
12416/60000 [=====>........................] - ETA: 1:07 - loss: 0.4250 - categorical_accuracy: 0.8659
12480/60000 [=====>........................] - ETA: 1:07 - loss: 0.4237 - categorical_accuracy: 0.8664
12544/60000 [=====>........................] - ETA: 1:07 - loss: 0.4220 - categorical_accuracy: 0.8669
12608/60000 [=====>........................] - ETA: 1:07 - loss: 0.4211 - categorical_accuracy: 0.8673
12672/60000 [=====>........................] - ETA: 1:07 - loss: 0.4196 - categorical_accuracy: 0.8677
12736/60000 [=====>........................] - ETA: 1:07 - loss: 0.4183 - categorical_accuracy: 0.8681
12800/60000 [=====>........................] - ETA: 1:07 - loss: 0.4168 - categorical_accuracy: 0.8685
12864/60000 [=====>........................] - ETA: 1:07 - loss: 0.4153 - categorical_accuracy: 0.8689
12928/60000 [=====>........................] - ETA: 1:06 - loss: 0.4139 - categorical_accuracy: 0.8694
12992/60000 [=====>........................] - ETA: 1:06 - loss: 0.4124 - categorical_accuracy: 0.8697
13056/60000 [=====>........................] - ETA: 1:06 - loss: 0.4114 - categorical_accuracy: 0.8698
13120/60000 [=====>........................] - ETA: 1:06 - loss: 0.4104 - categorical_accuracy: 0.8700
13184/60000 [=====>........................] - ETA: 1:06 - loss: 0.4090 - categorical_accuracy: 0.8703
13248/60000 [=====>........................] - ETA: 1:06 - loss: 0.4072 - categorical_accuracy: 0.8709
13280/60000 [=====>........................] - ETA: 1:06 - loss: 0.4064 - categorical_accuracy: 0.8712
13312/60000 [=====>........................] - ETA: 1:06 - loss: 0.4058 - categorical_accuracy: 0.8714
13376/60000 [=====>........................] - ETA: 1:06 - loss: 0.4046 - categorical_accuracy: 0.8716
13440/60000 [=====>........................] - ETA: 1:06 - loss: 0.4028 - categorical_accuracy: 0.8722
13504/60000 [=====>........................] - ETA: 1:06 - loss: 0.4020 - categorical_accuracy: 0.8726
13568/60000 [=====>........................] - ETA: 1:06 - loss: 0.4011 - categorical_accuracy: 0.8730
13632/60000 [=====>........................] - ETA: 1:06 - loss: 0.4009 - categorical_accuracy: 0.8732
13696/60000 [=====>........................] - ETA: 1:05 - loss: 0.3998 - categorical_accuracy: 0.8734
13760/60000 [=====>........................] - ETA: 1:05 - loss: 0.3985 - categorical_accuracy: 0.8737
13824/60000 [=====>........................] - ETA: 1:05 - loss: 0.3973 - categorical_accuracy: 0.8741
13888/60000 [=====>........................] - ETA: 1:05 - loss: 0.3970 - categorical_accuracy: 0.8741
13952/60000 [=====>........................] - ETA: 1:05 - loss: 0.3959 - categorical_accuracy: 0.8744
14016/60000 [======>.......................] - ETA: 1:05 - loss: 0.3947 - categorical_accuracy: 0.8748
14080/60000 [======>.......................] - ETA: 1:05 - loss: 0.3932 - categorical_accuracy: 0.8753
14144/60000 [======>.......................] - ETA: 1:05 - loss: 0.3929 - categorical_accuracy: 0.8754
14208/60000 [======>.......................] - ETA: 1:05 - loss: 0.3918 - categorical_accuracy: 0.8758
14272/60000 [======>.......................] - ETA: 1:05 - loss: 0.3904 - categorical_accuracy: 0.8763
14336/60000 [======>.......................] - ETA: 1:04 - loss: 0.3892 - categorical_accuracy: 0.8766
14400/60000 [======>.......................] - ETA: 1:04 - loss: 0.3892 - categorical_accuracy: 0.8765
14464/60000 [======>.......................] - ETA: 1:04 - loss: 0.3880 - categorical_accuracy: 0.8769
14528/60000 [======>.......................] - ETA: 1:04 - loss: 0.3875 - categorical_accuracy: 0.8772
14592/60000 [======>.......................] - ETA: 1:04 - loss: 0.3863 - categorical_accuracy: 0.8776
14656/60000 [======>.......................] - ETA: 1:04 - loss: 0.3852 - categorical_accuracy: 0.8779
14720/60000 [======>.......................] - ETA: 1:04 - loss: 0.3850 - categorical_accuracy: 0.8781
14784/60000 [======>.......................] - ETA: 1:04 - loss: 0.3836 - categorical_accuracy: 0.8786
14848/60000 [======>.......................] - ETA: 1:04 - loss: 0.3828 - categorical_accuracy: 0.8788
14912/60000 [======>.......................] - ETA: 1:04 - loss: 0.3819 - categorical_accuracy: 0.8791
14976/60000 [======>.......................] - ETA: 1:03 - loss: 0.3811 - categorical_accuracy: 0.8793
15040/60000 [======>.......................] - ETA: 1:03 - loss: 0.3802 - categorical_accuracy: 0.8796
15104/60000 [======>.......................] - ETA: 1:03 - loss: 0.3795 - categorical_accuracy: 0.8798
15168/60000 [======>.......................] - ETA: 1:03 - loss: 0.3793 - categorical_accuracy: 0.8799
15232/60000 [======>.......................] - ETA: 1:03 - loss: 0.3786 - categorical_accuracy: 0.8801
15296/60000 [======>.......................] - ETA: 1:03 - loss: 0.3776 - categorical_accuracy: 0.8804
15360/60000 [======>.......................] - ETA: 1:03 - loss: 0.3771 - categorical_accuracy: 0.8807
15424/60000 [======>.......................] - ETA: 1:03 - loss: 0.3758 - categorical_accuracy: 0.8812
15488/60000 [======>.......................] - ETA: 1:03 - loss: 0.3747 - categorical_accuracy: 0.8815
15552/60000 [======>.......................] - ETA: 1:03 - loss: 0.3733 - categorical_accuracy: 0.8820
15616/60000 [======>.......................] - ETA: 1:02 - loss: 0.3724 - categorical_accuracy: 0.8822
15648/60000 [======>.......................] - ETA: 1:02 - loss: 0.3722 - categorical_accuracy: 0.8823
15712/60000 [======>.......................] - ETA: 1:02 - loss: 0.3711 - categorical_accuracy: 0.8826
15776/60000 [======>.......................] - ETA: 1:02 - loss: 0.3702 - categorical_accuracy: 0.8830
15840/60000 [======>.......................] - ETA: 1:02 - loss: 0.3690 - categorical_accuracy: 0.8833
15904/60000 [======>.......................] - ETA: 1:02 - loss: 0.3681 - categorical_accuracy: 0.8836
15968/60000 [======>.......................] - ETA: 1:02 - loss: 0.3668 - categorical_accuracy: 0.8840
16032/60000 [=======>......................] - ETA: 1:02 - loss: 0.3657 - categorical_accuracy: 0.8844
16096/60000 [=======>......................] - ETA: 1:02 - loss: 0.3650 - categorical_accuracy: 0.8845
16160/60000 [=======>......................] - ETA: 1:02 - loss: 0.3642 - categorical_accuracy: 0.8848
16224/60000 [=======>......................] - ETA: 1:02 - loss: 0.3631 - categorical_accuracy: 0.8851
16288/60000 [=======>......................] - ETA: 1:01 - loss: 0.3620 - categorical_accuracy: 0.8854
16320/60000 [=======>......................] - ETA: 1:01 - loss: 0.3614 - categorical_accuracy: 0.8856
16352/60000 [=======>......................] - ETA: 1:01 - loss: 0.3608 - categorical_accuracy: 0.8858
16416/60000 [=======>......................] - ETA: 1:01 - loss: 0.3596 - categorical_accuracy: 0.8862
16480/60000 [=======>......................] - ETA: 1:01 - loss: 0.3587 - categorical_accuracy: 0.8865
16544/60000 [=======>......................] - ETA: 1:01 - loss: 0.3574 - categorical_accuracy: 0.8869
16608/60000 [=======>......................] - ETA: 1:01 - loss: 0.3566 - categorical_accuracy: 0.8872
16672/60000 [=======>......................] - ETA: 1:01 - loss: 0.3559 - categorical_accuracy: 0.8875
16736/60000 [=======>......................] - ETA: 1:01 - loss: 0.3550 - categorical_accuracy: 0.8878
16800/60000 [=======>......................] - ETA: 1:01 - loss: 0.3556 - categorical_accuracy: 0.8878
16864/60000 [=======>......................] - ETA: 1:01 - loss: 0.3548 - categorical_accuracy: 0.8880
16928/60000 [=======>......................] - ETA: 1:00 - loss: 0.3537 - categorical_accuracy: 0.8884
16992/60000 [=======>......................] - ETA: 1:00 - loss: 0.3536 - categorical_accuracy: 0.8884
17056/60000 [=======>......................] - ETA: 1:00 - loss: 0.3530 - categorical_accuracy: 0.8885
17120/60000 [=======>......................] - ETA: 1:00 - loss: 0.3522 - categorical_accuracy: 0.8887
17184/60000 [=======>......................] - ETA: 1:00 - loss: 0.3522 - categorical_accuracy: 0.8889
17248/60000 [=======>......................] - ETA: 1:00 - loss: 0.3523 - categorical_accuracy: 0.8889
17312/60000 [=======>......................] - ETA: 1:00 - loss: 0.3522 - categorical_accuracy: 0.8889
17376/60000 [=======>......................] - ETA: 1:00 - loss: 0.3517 - categorical_accuracy: 0.8892
17440/60000 [=======>......................] - ETA: 1:00 - loss: 0.3512 - categorical_accuracy: 0.8893
17504/60000 [=======>......................] - ETA: 1:00 - loss: 0.3504 - categorical_accuracy: 0.8896
17568/60000 [=======>......................] - ETA: 1:00 - loss: 0.3495 - categorical_accuracy: 0.8899
17632/60000 [=======>......................] - ETA: 59s - loss: 0.3490 - categorical_accuracy: 0.8901 
17696/60000 [=======>......................] - ETA: 59s - loss: 0.3479 - categorical_accuracy: 0.8905
17760/60000 [=======>......................] - ETA: 59s - loss: 0.3473 - categorical_accuracy: 0.8908
17824/60000 [=======>......................] - ETA: 59s - loss: 0.3467 - categorical_accuracy: 0.8909
17888/60000 [=======>......................] - ETA: 59s - loss: 0.3458 - categorical_accuracy: 0.8912
17952/60000 [=======>......................] - ETA: 59s - loss: 0.3456 - categorical_accuracy: 0.8913
18016/60000 [========>.....................] - ETA: 59s - loss: 0.3450 - categorical_accuracy: 0.8915
18080/60000 [========>.....................] - ETA: 59s - loss: 0.3442 - categorical_accuracy: 0.8916
18144/60000 [========>.....................] - ETA: 59s - loss: 0.3432 - categorical_accuracy: 0.8919
18208/60000 [========>.....................] - ETA: 59s - loss: 0.3422 - categorical_accuracy: 0.8923
18272/60000 [========>.....................] - ETA: 58s - loss: 0.3424 - categorical_accuracy: 0.8923
18336/60000 [========>.....................] - ETA: 58s - loss: 0.3420 - categorical_accuracy: 0.8925
18400/60000 [========>.....................] - ETA: 58s - loss: 0.3412 - categorical_accuracy: 0.8928
18464/60000 [========>.....................] - ETA: 58s - loss: 0.3408 - categorical_accuracy: 0.8928
18528/60000 [========>.....................] - ETA: 58s - loss: 0.3400 - categorical_accuracy: 0.8930
18560/60000 [========>.....................] - ETA: 58s - loss: 0.3399 - categorical_accuracy: 0.8931
18624/60000 [========>.....................] - ETA: 58s - loss: 0.3393 - categorical_accuracy: 0.8933
18688/60000 [========>.....................] - ETA: 58s - loss: 0.3387 - categorical_accuracy: 0.8935
18720/60000 [========>.....................] - ETA: 58s - loss: 0.3382 - categorical_accuracy: 0.8937
18784/60000 [========>.....................] - ETA: 58s - loss: 0.3374 - categorical_accuracy: 0.8940
18848/60000 [========>.....................] - ETA: 58s - loss: 0.3372 - categorical_accuracy: 0.8939
18912/60000 [========>.....................] - ETA: 58s - loss: 0.3367 - categorical_accuracy: 0.8940
18976/60000 [========>.....................] - ETA: 58s - loss: 0.3358 - categorical_accuracy: 0.8943
19040/60000 [========>.....................] - ETA: 57s - loss: 0.3353 - categorical_accuracy: 0.8945
19104/60000 [========>.....................] - ETA: 57s - loss: 0.3345 - categorical_accuracy: 0.8947
19168/60000 [========>.....................] - ETA: 57s - loss: 0.3344 - categorical_accuracy: 0.8948
19232/60000 [========>.....................] - ETA: 57s - loss: 0.3335 - categorical_accuracy: 0.8950
19296/60000 [========>.....................] - ETA: 57s - loss: 0.3330 - categorical_accuracy: 0.8953
19360/60000 [========>.....................] - ETA: 57s - loss: 0.3322 - categorical_accuracy: 0.8955
19424/60000 [========>.....................] - ETA: 57s - loss: 0.3313 - categorical_accuracy: 0.8958
19456/60000 [========>.....................] - ETA: 57s - loss: 0.3308 - categorical_accuracy: 0.8960
19520/60000 [========>.....................] - ETA: 57s - loss: 0.3303 - categorical_accuracy: 0.8961
19584/60000 [========>.....................] - ETA: 57s - loss: 0.3294 - categorical_accuracy: 0.8963
19648/60000 [========>.....................] - ETA: 57s - loss: 0.3286 - categorical_accuracy: 0.8966
19712/60000 [========>.....................] - ETA: 56s - loss: 0.3283 - categorical_accuracy: 0.8969
19776/60000 [========>.....................] - ETA: 56s - loss: 0.3277 - categorical_accuracy: 0.8970
19840/60000 [========>.....................] - ETA: 56s - loss: 0.3271 - categorical_accuracy: 0.8973
19904/60000 [========>.....................] - ETA: 56s - loss: 0.3262 - categorical_accuracy: 0.8975
19968/60000 [========>.....................] - ETA: 56s - loss: 0.3258 - categorical_accuracy: 0.8975
20032/60000 [=========>....................] - ETA: 56s - loss: 0.3253 - categorical_accuracy: 0.8977
20096/60000 [=========>....................] - ETA: 56s - loss: 0.3246 - categorical_accuracy: 0.8979
20160/60000 [=========>....................] - ETA: 56s - loss: 0.3237 - categorical_accuracy: 0.8983
20224/60000 [=========>....................] - ETA: 56s - loss: 0.3242 - categorical_accuracy: 0.8984
20288/60000 [=========>....................] - ETA: 56s - loss: 0.3234 - categorical_accuracy: 0.8986
20352/60000 [=========>....................] - ETA: 56s - loss: 0.3226 - categorical_accuracy: 0.8988
20416/60000 [=========>....................] - ETA: 55s - loss: 0.3218 - categorical_accuracy: 0.8990
20480/60000 [=========>....................] - ETA: 55s - loss: 0.3212 - categorical_accuracy: 0.8991
20544/60000 [=========>....................] - ETA: 55s - loss: 0.3211 - categorical_accuracy: 0.8993
20608/60000 [=========>....................] - ETA: 55s - loss: 0.3204 - categorical_accuracy: 0.8996
20672/60000 [=========>....................] - ETA: 55s - loss: 0.3197 - categorical_accuracy: 0.8998
20736/60000 [=========>....................] - ETA: 55s - loss: 0.3191 - categorical_accuracy: 0.8999
20800/60000 [=========>....................] - ETA: 55s - loss: 0.3184 - categorical_accuracy: 0.9001
20864/60000 [=========>....................] - ETA: 55s - loss: 0.3177 - categorical_accuracy: 0.9003
20928/60000 [=========>....................] - ETA: 55s - loss: 0.3169 - categorical_accuracy: 0.9006
20992/60000 [=========>....................] - ETA: 55s - loss: 0.3160 - categorical_accuracy: 0.9009
21056/60000 [=========>....................] - ETA: 54s - loss: 0.3154 - categorical_accuracy: 0.9011
21120/60000 [=========>....................] - ETA: 54s - loss: 0.3148 - categorical_accuracy: 0.9012
21184/60000 [=========>....................] - ETA: 54s - loss: 0.3144 - categorical_accuracy: 0.9013
21248/60000 [=========>....................] - ETA: 54s - loss: 0.3137 - categorical_accuracy: 0.9015
21312/60000 [=========>....................] - ETA: 54s - loss: 0.3132 - categorical_accuracy: 0.9017
21376/60000 [=========>....................] - ETA: 54s - loss: 0.3130 - categorical_accuracy: 0.9018
21440/60000 [=========>....................] - ETA: 54s - loss: 0.3122 - categorical_accuracy: 0.9021
21504/60000 [=========>....................] - ETA: 54s - loss: 0.3114 - categorical_accuracy: 0.9023
21568/60000 [=========>....................] - ETA: 54s - loss: 0.3115 - categorical_accuracy: 0.9024
21632/60000 [=========>....................] - ETA: 54s - loss: 0.3115 - categorical_accuracy: 0.9024
21696/60000 [=========>....................] - ETA: 54s - loss: 0.3110 - categorical_accuracy: 0.9026
21760/60000 [=========>....................] - ETA: 53s - loss: 0.3104 - categorical_accuracy: 0.9028
21824/60000 [=========>....................] - ETA: 53s - loss: 0.3100 - categorical_accuracy: 0.9029
21888/60000 [=========>....................] - ETA: 53s - loss: 0.3095 - categorical_accuracy: 0.9030
21952/60000 [=========>....................] - ETA: 53s - loss: 0.3089 - categorical_accuracy: 0.9032
22016/60000 [==========>...................] - ETA: 53s - loss: 0.3082 - categorical_accuracy: 0.9034
22080/60000 [==========>...................] - ETA: 53s - loss: 0.3075 - categorical_accuracy: 0.9036
22144/60000 [==========>...................] - ETA: 53s - loss: 0.3071 - categorical_accuracy: 0.9037
22208/60000 [==========>...................] - ETA: 53s - loss: 0.3065 - categorical_accuracy: 0.9039
22272/60000 [==========>...................] - ETA: 53s - loss: 0.3058 - categorical_accuracy: 0.9040
22336/60000 [==========>...................] - ETA: 53s - loss: 0.3052 - categorical_accuracy: 0.9042
22400/60000 [==========>...................] - ETA: 53s - loss: 0.3046 - categorical_accuracy: 0.9045
22464/60000 [==========>...................] - ETA: 52s - loss: 0.3048 - categorical_accuracy: 0.9046
22528/60000 [==========>...................] - ETA: 52s - loss: 0.3045 - categorical_accuracy: 0.9047
22592/60000 [==========>...................] - ETA: 52s - loss: 0.3042 - categorical_accuracy: 0.9049
22656/60000 [==========>...................] - ETA: 52s - loss: 0.3037 - categorical_accuracy: 0.9051
22720/60000 [==========>...................] - ETA: 52s - loss: 0.3036 - categorical_accuracy: 0.9052
22784/60000 [==========>...................] - ETA: 52s - loss: 0.3032 - categorical_accuracy: 0.9054
22848/60000 [==========>...................] - ETA: 52s - loss: 0.3034 - categorical_accuracy: 0.9053
22912/60000 [==========>...................] - ETA: 52s - loss: 0.3032 - categorical_accuracy: 0.9054
22976/60000 [==========>...................] - ETA: 52s - loss: 0.3029 - categorical_accuracy: 0.9056
23040/60000 [==========>...................] - ETA: 52s - loss: 0.3026 - categorical_accuracy: 0.9056
23104/60000 [==========>...................] - ETA: 52s - loss: 0.3024 - categorical_accuracy: 0.9058
23168/60000 [==========>...................] - ETA: 51s - loss: 0.3025 - categorical_accuracy: 0.9058
23232/60000 [==========>...................] - ETA: 51s - loss: 0.3021 - categorical_accuracy: 0.9059
23296/60000 [==========>...................] - ETA: 51s - loss: 0.3017 - categorical_accuracy: 0.9059
23360/60000 [==========>...................] - ETA: 51s - loss: 0.3013 - categorical_accuracy: 0.9062
23424/60000 [==========>...................] - ETA: 51s - loss: 0.3007 - categorical_accuracy: 0.9064
23488/60000 [==========>...................] - ETA: 51s - loss: 0.3002 - categorical_accuracy: 0.9065
23552/60000 [==========>...................] - ETA: 51s - loss: 0.2996 - categorical_accuracy: 0.9066
23616/60000 [==========>...................] - ETA: 51s - loss: 0.2990 - categorical_accuracy: 0.9068
23680/60000 [==========>...................] - ETA: 51s - loss: 0.2987 - categorical_accuracy: 0.9069
23744/60000 [==========>...................] - ETA: 51s - loss: 0.2981 - categorical_accuracy: 0.9071
23808/60000 [==========>...................] - ETA: 51s - loss: 0.2977 - categorical_accuracy: 0.9073
23872/60000 [==========>...................] - ETA: 50s - loss: 0.2973 - categorical_accuracy: 0.9073
23936/60000 [==========>...................] - ETA: 50s - loss: 0.2972 - categorical_accuracy: 0.9074
24000/60000 [===========>..................] - ETA: 50s - loss: 0.2971 - categorical_accuracy: 0.9075
24064/60000 [===========>..................] - ETA: 50s - loss: 0.2968 - categorical_accuracy: 0.9077
24128/60000 [===========>..................] - ETA: 50s - loss: 0.2963 - categorical_accuracy: 0.9078
24192/60000 [===========>..................] - ETA: 50s - loss: 0.2957 - categorical_accuracy: 0.9079
24256/60000 [===========>..................] - ETA: 50s - loss: 0.2953 - categorical_accuracy: 0.9081
24320/60000 [===========>..................] - ETA: 50s - loss: 0.2949 - categorical_accuracy: 0.9082
24384/60000 [===========>..................] - ETA: 50s - loss: 0.2944 - categorical_accuracy: 0.9084
24448/60000 [===========>..................] - ETA: 50s - loss: 0.2937 - categorical_accuracy: 0.9086
24480/60000 [===========>..................] - ETA: 50s - loss: 0.2934 - categorical_accuracy: 0.9087
24544/60000 [===========>..................] - ETA: 49s - loss: 0.2928 - categorical_accuracy: 0.9089
24608/60000 [===========>..................] - ETA: 49s - loss: 0.2927 - categorical_accuracy: 0.9090
24672/60000 [===========>..................] - ETA: 49s - loss: 0.2923 - categorical_accuracy: 0.9092
24736/60000 [===========>..................] - ETA: 49s - loss: 0.2918 - categorical_accuracy: 0.9093
24800/60000 [===========>..................] - ETA: 49s - loss: 0.2913 - categorical_accuracy: 0.9095
24864/60000 [===========>..................] - ETA: 49s - loss: 0.2907 - categorical_accuracy: 0.9097
24928/60000 [===========>..................] - ETA: 49s - loss: 0.2900 - categorical_accuracy: 0.9099
24992/60000 [===========>..................] - ETA: 49s - loss: 0.2897 - categorical_accuracy: 0.9100
25056/60000 [===========>..................] - ETA: 49s - loss: 0.2893 - categorical_accuracy: 0.9101
25120/60000 [===========>..................] - ETA: 49s - loss: 0.2887 - categorical_accuracy: 0.9104
25184/60000 [===========>..................] - ETA: 49s - loss: 0.2884 - categorical_accuracy: 0.9105
25248/60000 [===========>..................] - ETA: 48s - loss: 0.2879 - categorical_accuracy: 0.9106
25312/60000 [===========>..................] - ETA: 48s - loss: 0.2873 - categorical_accuracy: 0.9108
25376/60000 [===========>..................] - ETA: 48s - loss: 0.2869 - categorical_accuracy: 0.9109
25440/60000 [===========>..................] - ETA: 48s - loss: 0.2867 - categorical_accuracy: 0.9109
25504/60000 [===========>..................] - ETA: 48s - loss: 0.2865 - categorical_accuracy: 0.9110
25568/60000 [===========>..................] - ETA: 48s - loss: 0.2865 - categorical_accuracy: 0.9111
25632/60000 [===========>..................] - ETA: 48s - loss: 0.2862 - categorical_accuracy: 0.9113
25696/60000 [===========>..................] - ETA: 48s - loss: 0.2856 - categorical_accuracy: 0.9115
25760/60000 [===========>..................] - ETA: 48s - loss: 0.2850 - categorical_accuracy: 0.9116
25824/60000 [===========>..................] - ETA: 48s - loss: 0.2847 - categorical_accuracy: 0.9118
25888/60000 [===========>..................] - ETA: 48s - loss: 0.2842 - categorical_accuracy: 0.9119
25952/60000 [===========>..................] - ETA: 47s - loss: 0.2837 - categorical_accuracy: 0.9121
26016/60000 [============>.................] - ETA: 47s - loss: 0.2837 - categorical_accuracy: 0.9122
26080/60000 [============>.................] - ETA: 47s - loss: 0.2830 - categorical_accuracy: 0.9124
26144/60000 [============>.................] - ETA: 47s - loss: 0.2826 - categorical_accuracy: 0.9125
26208/60000 [============>.................] - ETA: 47s - loss: 0.2821 - categorical_accuracy: 0.9127
26272/60000 [============>.................] - ETA: 47s - loss: 0.2816 - categorical_accuracy: 0.9128
26336/60000 [============>.................] - ETA: 47s - loss: 0.2810 - categorical_accuracy: 0.9130
26400/60000 [============>.................] - ETA: 47s - loss: 0.2807 - categorical_accuracy: 0.9131
26464/60000 [============>.................] - ETA: 47s - loss: 0.2802 - categorical_accuracy: 0.9132
26528/60000 [============>.................] - ETA: 47s - loss: 0.2799 - categorical_accuracy: 0.9133
26592/60000 [============>.................] - ETA: 47s - loss: 0.2797 - categorical_accuracy: 0.9134
26656/60000 [============>.................] - ETA: 46s - loss: 0.2798 - categorical_accuracy: 0.9134
26720/60000 [============>.................] - ETA: 46s - loss: 0.2794 - categorical_accuracy: 0.9135
26784/60000 [============>.................] - ETA: 46s - loss: 0.2788 - categorical_accuracy: 0.9137
26848/60000 [============>.................] - ETA: 46s - loss: 0.2785 - categorical_accuracy: 0.9138
26912/60000 [============>.................] - ETA: 46s - loss: 0.2781 - categorical_accuracy: 0.9139
26976/60000 [============>.................] - ETA: 46s - loss: 0.2777 - categorical_accuracy: 0.9141
27040/60000 [============>.................] - ETA: 46s - loss: 0.2775 - categorical_accuracy: 0.9142
27104/60000 [============>.................] - ETA: 46s - loss: 0.2770 - categorical_accuracy: 0.9144
27168/60000 [============>.................] - ETA: 46s - loss: 0.2768 - categorical_accuracy: 0.9145
27232/60000 [============>.................] - ETA: 46s - loss: 0.2762 - categorical_accuracy: 0.9147
27296/60000 [============>.................] - ETA: 46s - loss: 0.2758 - categorical_accuracy: 0.9148
27360/60000 [============>.................] - ETA: 45s - loss: 0.2752 - categorical_accuracy: 0.9150
27424/60000 [============>.................] - ETA: 45s - loss: 0.2747 - categorical_accuracy: 0.9152
27488/60000 [============>.................] - ETA: 45s - loss: 0.2742 - categorical_accuracy: 0.9154
27552/60000 [============>.................] - ETA: 45s - loss: 0.2742 - categorical_accuracy: 0.9154
27584/60000 [============>.................] - ETA: 45s - loss: 0.2745 - categorical_accuracy: 0.9153
27648/60000 [============>.................] - ETA: 45s - loss: 0.2741 - categorical_accuracy: 0.9154
27712/60000 [============>.................] - ETA: 45s - loss: 0.2738 - categorical_accuracy: 0.9155
27776/60000 [============>.................] - ETA: 45s - loss: 0.2737 - categorical_accuracy: 0.9155
27840/60000 [============>.................] - ETA: 45s - loss: 0.2734 - categorical_accuracy: 0.9156
27904/60000 [============>.................] - ETA: 45s - loss: 0.2728 - categorical_accuracy: 0.9157
27968/60000 [============>.................] - ETA: 45s - loss: 0.2727 - categorical_accuracy: 0.9158
28032/60000 [=============>................] - ETA: 45s - loss: 0.2722 - categorical_accuracy: 0.9160
28096/60000 [=============>................] - ETA: 44s - loss: 0.2718 - categorical_accuracy: 0.9162
28160/60000 [=============>................] - ETA: 44s - loss: 0.2716 - categorical_accuracy: 0.9162
28224/60000 [=============>................] - ETA: 44s - loss: 0.2717 - categorical_accuracy: 0.9162
28288/60000 [=============>................] - ETA: 44s - loss: 0.2716 - categorical_accuracy: 0.9162
28352/60000 [=============>................] - ETA: 44s - loss: 0.2714 - categorical_accuracy: 0.9163
28416/60000 [=============>................] - ETA: 44s - loss: 0.2709 - categorical_accuracy: 0.9164
28480/60000 [=============>................] - ETA: 44s - loss: 0.2704 - categorical_accuracy: 0.9166
28544/60000 [=============>................] - ETA: 44s - loss: 0.2701 - categorical_accuracy: 0.9167
28608/60000 [=============>................] - ETA: 44s - loss: 0.2698 - categorical_accuracy: 0.9168
28672/60000 [=============>................] - ETA: 44s - loss: 0.2693 - categorical_accuracy: 0.9170
28704/60000 [=============>................] - ETA: 44s - loss: 0.2691 - categorical_accuracy: 0.9170
28768/60000 [=============>................] - ETA: 44s - loss: 0.2689 - categorical_accuracy: 0.9170
28800/60000 [=============>................] - ETA: 43s - loss: 0.2688 - categorical_accuracy: 0.9171
28864/60000 [=============>................] - ETA: 43s - loss: 0.2684 - categorical_accuracy: 0.9172
28928/60000 [=============>................] - ETA: 43s - loss: 0.2679 - categorical_accuracy: 0.9173
28992/60000 [=============>................] - ETA: 43s - loss: 0.2675 - categorical_accuracy: 0.9175
29056/60000 [=============>................] - ETA: 43s - loss: 0.2670 - categorical_accuracy: 0.9176
29120/60000 [=============>................] - ETA: 43s - loss: 0.2667 - categorical_accuracy: 0.9177
29184/60000 [=============>................] - ETA: 43s - loss: 0.2662 - categorical_accuracy: 0.9178
29216/60000 [=============>................] - ETA: 43s - loss: 0.2659 - categorical_accuracy: 0.9179
29280/60000 [=============>................] - ETA: 43s - loss: 0.2659 - categorical_accuracy: 0.9180
29344/60000 [=============>................] - ETA: 43s - loss: 0.2654 - categorical_accuracy: 0.9181
29408/60000 [=============>................] - ETA: 43s - loss: 0.2651 - categorical_accuracy: 0.9183
29472/60000 [=============>................] - ETA: 43s - loss: 0.2647 - categorical_accuracy: 0.9184
29536/60000 [=============>................] - ETA: 42s - loss: 0.2646 - categorical_accuracy: 0.9184
29600/60000 [=============>................] - ETA: 42s - loss: 0.2646 - categorical_accuracy: 0.9185
29664/60000 [=============>................] - ETA: 42s - loss: 0.2643 - categorical_accuracy: 0.9186
29728/60000 [=============>................] - ETA: 42s - loss: 0.2639 - categorical_accuracy: 0.9188
29792/60000 [=============>................] - ETA: 42s - loss: 0.2637 - categorical_accuracy: 0.9188
29824/60000 [=============>................] - ETA: 42s - loss: 0.2636 - categorical_accuracy: 0.9188
29888/60000 [=============>................] - ETA: 42s - loss: 0.2632 - categorical_accuracy: 0.9189
29952/60000 [=============>................] - ETA: 42s - loss: 0.2628 - categorical_accuracy: 0.9191
30016/60000 [==============>...............] - ETA: 42s - loss: 0.2622 - categorical_accuracy: 0.9192
30080/60000 [==============>...............] - ETA: 42s - loss: 0.2619 - categorical_accuracy: 0.9193
30144/60000 [==============>...............] - ETA: 42s - loss: 0.2615 - categorical_accuracy: 0.9195
30208/60000 [==============>...............] - ETA: 41s - loss: 0.2614 - categorical_accuracy: 0.9195
30272/60000 [==============>...............] - ETA: 41s - loss: 0.2610 - categorical_accuracy: 0.9197
30336/60000 [==============>...............] - ETA: 41s - loss: 0.2606 - categorical_accuracy: 0.9198
30400/60000 [==============>...............] - ETA: 41s - loss: 0.2603 - categorical_accuracy: 0.9199
30464/60000 [==============>...............] - ETA: 41s - loss: 0.2602 - categorical_accuracy: 0.9200
30528/60000 [==============>...............] - ETA: 41s - loss: 0.2598 - categorical_accuracy: 0.9201
30592/60000 [==============>...............] - ETA: 41s - loss: 0.2595 - categorical_accuracy: 0.9201
30656/60000 [==============>...............] - ETA: 41s - loss: 0.2591 - categorical_accuracy: 0.9202
30720/60000 [==============>...............] - ETA: 41s - loss: 0.2588 - categorical_accuracy: 0.9203
30784/60000 [==============>...............] - ETA: 41s - loss: 0.2586 - categorical_accuracy: 0.9204
30848/60000 [==============>...............] - ETA: 41s - loss: 0.2589 - categorical_accuracy: 0.9205
30912/60000 [==============>...............] - ETA: 40s - loss: 0.2586 - categorical_accuracy: 0.9206
30976/60000 [==============>...............] - ETA: 40s - loss: 0.2583 - categorical_accuracy: 0.9207
31040/60000 [==============>...............] - ETA: 40s - loss: 0.2580 - categorical_accuracy: 0.9208
31104/60000 [==============>...............] - ETA: 40s - loss: 0.2579 - categorical_accuracy: 0.9208
31168/60000 [==============>...............] - ETA: 40s - loss: 0.2574 - categorical_accuracy: 0.9210
31232/60000 [==============>...............] - ETA: 40s - loss: 0.2572 - categorical_accuracy: 0.9211
31296/60000 [==============>...............] - ETA: 40s - loss: 0.2568 - categorical_accuracy: 0.9212
31328/60000 [==============>...............] - ETA: 40s - loss: 0.2565 - categorical_accuracy: 0.9213
31392/60000 [==============>...............] - ETA: 40s - loss: 0.2562 - categorical_accuracy: 0.9214
31456/60000 [==============>...............] - ETA: 40s - loss: 0.2560 - categorical_accuracy: 0.9215
31520/60000 [==============>...............] - ETA: 40s - loss: 0.2555 - categorical_accuracy: 0.9216
31584/60000 [==============>...............] - ETA: 40s - loss: 0.2551 - categorical_accuracy: 0.9217
31648/60000 [==============>...............] - ETA: 39s - loss: 0.2550 - categorical_accuracy: 0.9218
31712/60000 [==============>...............] - ETA: 39s - loss: 0.2545 - categorical_accuracy: 0.9219
31776/60000 [==============>...............] - ETA: 39s - loss: 0.2541 - categorical_accuracy: 0.9220
31840/60000 [==============>...............] - ETA: 39s - loss: 0.2543 - categorical_accuracy: 0.9220
31904/60000 [==============>...............] - ETA: 39s - loss: 0.2539 - categorical_accuracy: 0.9221
31968/60000 [==============>...............] - ETA: 39s - loss: 0.2536 - categorical_accuracy: 0.9222
32032/60000 [===============>..............] - ETA: 39s - loss: 0.2533 - categorical_accuracy: 0.9224
32096/60000 [===============>..............] - ETA: 39s - loss: 0.2530 - categorical_accuracy: 0.9224
32160/60000 [===============>..............] - ETA: 39s - loss: 0.2528 - categorical_accuracy: 0.9225
32224/60000 [===============>..............] - ETA: 39s - loss: 0.2524 - categorical_accuracy: 0.9226
32288/60000 [===============>..............] - ETA: 39s - loss: 0.2521 - categorical_accuracy: 0.9227
32352/60000 [===============>..............] - ETA: 38s - loss: 0.2520 - categorical_accuracy: 0.9228
32416/60000 [===============>..............] - ETA: 38s - loss: 0.2518 - categorical_accuracy: 0.9228
32480/60000 [===============>..............] - ETA: 38s - loss: 0.2517 - categorical_accuracy: 0.9228
32544/60000 [===============>..............] - ETA: 38s - loss: 0.2515 - categorical_accuracy: 0.9229
32608/60000 [===============>..............] - ETA: 38s - loss: 0.2511 - categorical_accuracy: 0.9230
32672/60000 [===============>..............] - ETA: 38s - loss: 0.2511 - categorical_accuracy: 0.9231
32736/60000 [===============>..............] - ETA: 38s - loss: 0.2507 - categorical_accuracy: 0.9232
32800/60000 [===============>..............] - ETA: 38s - loss: 0.2507 - categorical_accuracy: 0.9232
32832/60000 [===============>..............] - ETA: 38s - loss: 0.2506 - categorical_accuracy: 0.9232
32896/60000 [===============>..............] - ETA: 38s - loss: 0.2503 - categorical_accuracy: 0.9234
32960/60000 [===============>..............] - ETA: 38s - loss: 0.2498 - categorical_accuracy: 0.9235
33024/60000 [===============>..............] - ETA: 37s - loss: 0.2499 - categorical_accuracy: 0.9235
33088/60000 [===============>..............] - ETA: 37s - loss: 0.2496 - categorical_accuracy: 0.9236
33152/60000 [===============>..............] - ETA: 37s - loss: 0.2493 - categorical_accuracy: 0.9237
33216/60000 [===============>..............] - ETA: 37s - loss: 0.2489 - categorical_accuracy: 0.9238
33280/60000 [===============>..............] - ETA: 37s - loss: 0.2486 - categorical_accuracy: 0.9239
33344/60000 [===============>..............] - ETA: 37s - loss: 0.2484 - categorical_accuracy: 0.9239
33408/60000 [===============>..............] - ETA: 37s - loss: 0.2483 - categorical_accuracy: 0.9240
33472/60000 [===============>..............] - ETA: 37s - loss: 0.2482 - categorical_accuracy: 0.9240
33536/60000 [===============>..............] - ETA: 37s - loss: 0.2480 - categorical_accuracy: 0.9241
33600/60000 [===============>..............] - ETA: 37s - loss: 0.2476 - categorical_accuracy: 0.9243
33664/60000 [===============>..............] - ETA: 37s - loss: 0.2473 - categorical_accuracy: 0.9243
33728/60000 [===============>..............] - ETA: 36s - loss: 0.2472 - categorical_accuracy: 0.9244
33792/60000 [===============>..............] - ETA: 36s - loss: 0.2469 - categorical_accuracy: 0.9245
33824/60000 [===============>..............] - ETA: 36s - loss: 0.2467 - categorical_accuracy: 0.9246
33856/60000 [===============>..............] - ETA: 36s - loss: 0.2467 - categorical_accuracy: 0.9246
33920/60000 [===============>..............] - ETA: 36s - loss: 0.2466 - categorical_accuracy: 0.9246
33984/60000 [===============>..............] - ETA: 36s - loss: 0.2463 - categorical_accuracy: 0.9247
34048/60000 [================>.............] - ETA: 36s - loss: 0.2461 - categorical_accuracy: 0.9248
34112/60000 [================>.............] - ETA: 36s - loss: 0.2460 - categorical_accuracy: 0.9248
34176/60000 [================>.............] - ETA: 36s - loss: 0.2456 - categorical_accuracy: 0.9249
34240/60000 [================>.............] - ETA: 36s - loss: 0.2453 - categorical_accuracy: 0.9250
34304/60000 [================>.............] - ETA: 36s - loss: 0.2450 - categorical_accuracy: 0.9251
34368/60000 [================>.............] - ETA: 36s - loss: 0.2448 - categorical_accuracy: 0.9252
34432/60000 [================>.............] - ETA: 35s - loss: 0.2445 - categorical_accuracy: 0.9252
34496/60000 [================>.............] - ETA: 35s - loss: 0.2443 - categorical_accuracy: 0.9253
34560/60000 [================>.............] - ETA: 35s - loss: 0.2440 - categorical_accuracy: 0.9254
34624/60000 [================>.............] - ETA: 35s - loss: 0.2439 - categorical_accuracy: 0.9255
34688/60000 [================>.............] - ETA: 35s - loss: 0.2435 - categorical_accuracy: 0.9256
34752/60000 [================>.............] - ETA: 35s - loss: 0.2431 - categorical_accuracy: 0.9257
34816/60000 [================>.............] - ETA: 35s - loss: 0.2428 - categorical_accuracy: 0.9258
34880/60000 [================>.............] - ETA: 35s - loss: 0.2426 - categorical_accuracy: 0.9259
34944/60000 [================>.............] - ETA: 35s - loss: 0.2424 - categorical_accuracy: 0.9259
35008/60000 [================>.............] - ETA: 35s - loss: 0.2422 - categorical_accuracy: 0.9260
35072/60000 [================>.............] - ETA: 35s - loss: 0.2420 - categorical_accuracy: 0.9260
35136/60000 [================>.............] - ETA: 34s - loss: 0.2421 - categorical_accuracy: 0.9261
35200/60000 [================>.............] - ETA: 34s - loss: 0.2417 - categorical_accuracy: 0.9262
35264/60000 [================>.............] - ETA: 34s - loss: 0.2413 - categorical_accuracy: 0.9263
35328/60000 [================>.............] - ETA: 34s - loss: 0.2412 - categorical_accuracy: 0.9263
35392/60000 [================>.............] - ETA: 34s - loss: 0.2408 - categorical_accuracy: 0.9264
35456/60000 [================>.............] - ETA: 34s - loss: 0.2405 - categorical_accuracy: 0.9265
35520/60000 [================>.............] - ETA: 34s - loss: 0.2401 - categorical_accuracy: 0.9266
35584/60000 [================>.............] - ETA: 34s - loss: 0.2398 - categorical_accuracy: 0.9267
35648/60000 [================>.............] - ETA: 34s - loss: 0.2395 - categorical_accuracy: 0.9268
35712/60000 [================>.............] - ETA: 34s - loss: 0.2396 - categorical_accuracy: 0.9268
35776/60000 [================>.............] - ETA: 34s - loss: 0.2396 - categorical_accuracy: 0.9268
35840/60000 [================>.............] - ETA: 33s - loss: 0.2394 - categorical_accuracy: 0.9269
35904/60000 [================>.............] - ETA: 33s - loss: 0.2392 - categorical_accuracy: 0.9270
35968/60000 [================>.............] - ETA: 33s - loss: 0.2391 - categorical_accuracy: 0.9271
36000/60000 [=================>............] - ETA: 33s - loss: 0.2389 - categorical_accuracy: 0.9271
36064/60000 [=================>............] - ETA: 33s - loss: 0.2386 - categorical_accuracy: 0.9272
36096/60000 [=================>............] - ETA: 33s - loss: 0.2385 - categorical_accuracy: 0.9272
36160/60000 [=================>............] - ETA: 33s - loss: 0.2384 - categorical_accuracy: 0.9273
36224/60000 [=================>............] - ETA: 33s - loss: 0.2380 - categorical_accuracy: 0.9274
36288/60000 [=================>............] - ETA: 33s - loss: 0.2378 - categorical_accuracy: 0.9275
36352/60000 [=================>............] - ETA: 33s - loss: 0.2375 - categorical_accuracy: 0.9276
36416/60000 [=================>............] - ETA: 33s - loss: 0.2372 - categorical_accuracy: 0.9277
36480/60000 [=================>............] - ETA: 33s - loss: 0.2370 - categorical_accuracy: 0.9277
36544/60000 [=================>............] - ETA: 32s - loss: 0.2366 - categorical_accuracy: 0.9278
36608/60000 [=================>............] - ETA: 32s - loss: 0.2363 - categorical_accuracy: 0.9279
36672/60000 [=================>............] - ETA: 32s - loss: 0.2361 - categorical_accuracy: 0.9280
36736/60000 [=================>............] - ETA: 32s - loss: 0.2358 - categorical_accuracy: 0.9281
36800/60000 [=================>............] - ETA: 32s - loss: 0.2355 - categorical_accuracy: 0.9281
36864/60000 [=================>............] - ETA: 32s - loss: 0.2353 - categorical_accuracy: 0.9282
36928/60000 [=================>............] - ETA: 32s - loss: 0.2349 - categorical_accuracy: 0.9283
36992/60000 [=================>............] - ETA: 32s - loss: 0.2346 - categorical_accuracy: 0.9284
37056/60000 [=================>............] - ETA: 32s - loss: 0.2345 - categorical_accuracy: 0.9285
37120/60000 [=================>............] - ETA: 32s - loss: 0.2342 - categorical_accuracy: 0.9286
37184/60000 [=================>............] - ETA: 32s - loss: 0.2342 - categorical_accuracy: 0.9286
37248/60000 [=================>............] - ETA: 31s - loss: 0.2339 - categorical_accuracy: 0.9287
37312/60000 [=================>............] - ETA: 31s - loss: 0.2337 - categorical_accuracy: 0.9287
37376/60000 [=================>............] - ETA: 31s - loss: 0.2335 - categorical_accuracy: 0.9288
37440/60000 [=================>............] - ETA: 31s - loss: 0.2332 - categorical_accuracy: 0.9289
37504/60000 [=================>............] - ETA: 31s - loss: 0.2332 - categorical_accuracy: 0.9289
37568/60000 [=================>............] - ETA: 31s - loss: 0.2329 - categorical_accuracy: 0.9290
37632/60000 [=================>............] - ETA: 31s - loss: 0.2327 - categorical_accuracy: 0.9290
37696/60000 [=================>............] - ETA: 31s - loss: 0.2325 - categorical_accuracy: 0.9290
37760/60000 [=================>............] - ETA: 31s - loss: 0.2328 - categorical_accuracy: 0.9290
37824/60000 [=================>............] - ETA: 31s - loss: 0.2326 - categorical_accuracy: 0.9291
37888/60000 [=================>............] - ETA: 31s - loss: 0.2324 - categorical_accuracy: 0.9291
37952/60000 [=================>............] - ETA: 30s - loss: 0.2321 - categorical_accuracy: 0.9292
38016/60000 [==================>...........] - ETA: 30s - loss: 0.2318 - categorical_accuracy: 0.9293
38080/60000 [==================>...........] - ETA: 30s - loss: 0.2319 - categorical_accuracy: 0.9294
38144/60000 [==================>...........] - ETA: 30s - loss: 0.2316 - categorical_accuracy: 0.9295
38208/60000 [==================>...........] - ETA: 30s - loss: 0.2314 - categorical_accuracy: 0.9295
38272/60000 [==================>...........] - ETA: 30s - loss: 0.2312 - categorical_accuracy: 0.9296
38336/60000 [==================>...........] - ETA: 30s - loss: 0.2309 - categorical_accuracy: 0.9297
38400/60000 [==================>...........] - ETA: 30s - loss: 0.2308 - categorical_accuracy: 0.9297
38464/60000 [==================>...........] - ETA: 30s - loss: 0.2307 - categorical_accuracy: 0.9298
38528/60000 [==================>...........] - ETA: 30s - loss: 0.2308 - categorical_accuracy: 0.9298
38592/60000 [==================>...........] - ETA: 30s - loss: 0.2306 - categorical_accuracy: 0.9298
38656/60000 [==================>...........] - ETA: 29s - loss: 0.2302 - categorical_accuracy: 0.9299
38720/60000 [==================>...........] - ETA: 29s - loss: 0.2300 - categorical_accuracy: 0.9300
38784/60000 [==================>...........] - ETA: 29s - loss: 0.2297 - categorical_accuracy: 0.9300
38848/60000 [==================>...........] - ETA: 29s - loss: 0.2295 - categorical_accuracy: 0.9301
38912/60000 [==================>...........] - ETA: 29s - loss: 0.2294 - categorical_accuracy: 0.9301
38976/60000 [==================>...........] - ETA: 29s - loss: 0.2291 - categorical_accuracy: 0.9302
39040/60000 [==================>...........] - ETA: 29s - loss: 0.2289 - categorical_accuracy: 0.9302
39104/60000 [==================>...........] - ETA: 29s - loss: 0.2286 - categorical_accuracy: 0.9303
39168/60000 [==================>...........] - ETA: 29s - loss: 0.2284 - categorical_accuracy: 0.9304
39232/60000 [==================>...........] - ETA: 29s - loss: 0.2282 - categorical_accuracy: 0.9304
39296/60000 [==================>...........] - ETA: 29s - loss: 0.2281 - categorical_accuracy: 0.9305
39360/60000 [==================>...........] - ETA: 28s - loss: 0.2278 - categorical_accuracy: 0.9306
39424/60000 [==================>...........] - ETA: 28s - loss: 0.2276 - categorical_accuracy: 0.9307
39488/60000 [==================>...........] - ETA: 28s - loss: 0.2273 - categorical_accuracy: 0.9308
39552/60000 [==================>...........] - ETA: 28s - loss: 0.2270 - categorical_accuracy: 0.9309
39616/60000 [==================>...........] - ETA: 28s - loss: 0.2267 - categorical_accuracy: 0.9310
39680/60000 [==================>...........] - ETA: 28s - loss: 0.2264 - categorical_accuracy: 0.9310
39712/60000 [==================>...........] - ETA: 28s - loss: 0.2263 - categorical_accuracy: 0.9311
39776/60000 [==================>...........] - ETA: 28s - loss: 0.2262 - categorical_accuracy: 0.9311
39840/60000 [==================>...........] - ETA: 28s - loss: 0.2261 - categorical_accuracy: 0.9311
39904/60000 [==================>...........] - ETA: 28s - loss: 0.2258 - categorical_accuracy: 0.9312
39968/60000 [==================>...........] - ETA: 28s - loss: 0.2256 - categorical_accuracy: 0.9312
40032/60000 [===================>..........] - ETA: 27s - loss: 0.2254 - categorical_accuracy: 0.9313
40096/60000 [===================>..........] - ETA: 27s - loss: 0.2251 - categorical_accuracy: 0.9314
40160/60000 [===================>..........] - ETA: 27s - loss: 0.2249 - categorical_accuracy: 0.9314
40224/60000 [===================>..........] - ETA: 27s - loss: 0.2246 - categorical_accuracy: 0.9315
40288/60000 [===================>..........] - ETA: 27s - loss: 0.2247 - categorical_accuracy: 0.9316
40352/60000 [===================>..........] - ETA: 27s - loss: 0.2247 - categorical_accuracy: 0.9316
40416/60000 [===================>..........] - ETA: 27s - loss: 0.2244 - categorical_accuracy: 0.9317
40480/60000 [===================>..........] - ETA: 27s - loss: 0.2242 - categorical_accuracy: 0.9318
40544/60000 [===================>..........] - ETA: 27s - loss: 0.2239 - categorical_accuracy: 0.9319
40608/60000 [===================>..........] - ETA: 27s - loss: 0.2238 - categorical_accuracy: 0.9319
40672/60000 [===================>..........] - ETA: 27s - loss: 0.2235 - categorical_accuracy: 0.9320
40736/60000 [===================>..........] - ETA: 27s - loss: 0.2232 - categorical_accuracy: 0.9321
40800/60000 [===================>..........] - ETA: 26s - loss: 0.2233 - categorical_accuracy: 0.9322
40864/60000 [===================>..........] - ETA: 26s - loss: 0.2231 - categorical_accuracy: 0.9322
40928/60000 [===================>..........] - ETA: 26s - loss: 0.2231 - categorical_accuracy: 0.9322
40992/60000 [===================>..........] - ETA: 26s - loss: 0.2228 - categorical_accuracy: 0.9323
41056/60000 [===================>..........] - ETA: 26s - loss: 0.2226 - categorical_accuracy: 0.9324
41120/60000 [===================>..........] - ETA: 26s - loss: 0.2226 - categorical_accuracy: 0.9324
41184/60000 [===================>..........] - ETA: 26s - loss: 0.2225 - categorical_accuracy: 0.9324
41216/60000 [===================>..........] - ETA: 26s - loss: 0.2224 - categorical_accuracy: 0.9325
41280/60000 [===================>..........] - ETA: 26s - loss: 0.2222 - categorical_accuracy: 0.9325
41344/60000 [===================>..........] - ETA: 26s - loss: 0.2223 - categorical_accuracy: 0.9325
41408/60000 [===================>..........] - ETA: 26s - loss: 0.2222 - categorical_accuracy: 0.9325
41472/60000 [===================>..........] - ETA: 25s - loss: 0.2222 - categorical_accuracy: 0.9325
41536/60000 [===================>..........] - ETA: 25s - loss: 0.2219 - categorical_accuracy: 0.9326
41600/60000 [===================>..........] - ETA: 25s - loss: 0.2216 - categorical_accuracy: 0.9327
41664/60000 [===================>..........] - ETA: 25s - loss: 0.2214 - categorical_accuracy: 0.9328
41728/60000 [===================>..........] - ETA: 25s - loss: 0.2212 - categorical_accuracy: 0.9329
41792/60000 [===================>..........] - ETA: 25s - loss: 0.2210 - categorical_accuracy: 0.9330
41856/60000 [===================>..........] - ETA: 25s - loss: 0.2208 - categorical_accuracy: 0.9331
41920/60000 [===================>..........] - ETA: 25s - loss: 0.2205 - categorical_accuracy: 0.9332
41984/60000 [===================>..........] - ETA: 25s - loss: 0.2204 - categorical_accuracy: 0.9332
42048/60000 [====================>.........] - ETA: 25s - loss: 0.2201 - categorical_accuracy: 0.9333
42112/60000 [====================>.........] - ETA: 25s - loss: 0.2199 - categorical_accuracy: 0.9334
42176/60000 [====================>.........] - ETA: 24s - loss: 0.2197 - categorical_accuracy: 0.9334
42240/60000 [====================>.........] - ETA: 24s - loss: 0.2194 - categorical_accuracy: 0.9335
42304/60000 [====================>.........] - ETA: 24s - loss: 0.2193 - categorical_accuracy: 0.9336
42368/60000 [====================>.........] - ETA: 24s - loss: 0.2194 - categorical_accuracy: 0.9336
42432/60000 [====================>.........] - ETA: 24s - loss: 0.2193 - categorical_accuracy: 0.9336
42496/60000 [====================>.........] - ETA: 24s - loss: 0.2191 - categorical_accuracy: 0.9336
42528/60000 [====================>.........] - ETA: 24s - loss: 0.2190 - categorical_accuracy: 0.9337
42592/60000 [====================>.........] - ETA: 24s - loss: 0.2187 - categorical_accuracy: 0.9337
42656/60000 [====================>.........] - ETA: 24s - loss: 0.2185 - categorical_accuracy: 0.9338
42720/60000 [====================>.........] - ETA: 24s - loss: 0.2185 - categorical_accuracy: 0.9338
42784/60000 [====================>.........] - ETA: 24s - loss: 0.2185 - categorical_accuracy: 0.9339
42848/60000 [====================>.........] - ETA: 24s - loss: 0.2182 - categorical_accuracy: 0.9340
42912/60000 [====================>.........] - ETA: 23s - loss: 0.2183 - categorical_accuracy: 0.9340
42976/60000 [====================>.........] - ETA: 23s - loss: 0.2182 - categorical_accuracy: 0.9340
43040/60000 [====================>.........] - ETA: 23s - loss: 0.2180 - categorical_accuracy: 0.9341
43104/60000 [====================>.........] - ETA: 23s - loss: 0.2178 - categorical_accuracy: 0.9341
43168/60000 [====================>.........] - ETA: 23s - loss: 0.2177 - categorical_accuracy: 0.9342
43232/60000 [====================>.........] - ETA: 23s - loss: 0.2175 - categorical_accuracy: 0.9342
43296/60000 [====================>.........] - ETA: 23s - loss: 0.2172 - categorical_accuracy: 0.9343
43360/60000 [====================>.........] - ETA: 23s - loss: 0.2170 - categorical_accuracy: 0.9344
43424/60000 [====================>.........] - ETA: 23s - loss: 0.2168 - categorical_accuracy: 0.9344
43488/60000 [====================>.........] - ETA: 23s - loss: 0.2166 - categorical_accuracy: 0.9345
43552/60000 [====================>.........] - ETA: 23s - loss: 0.2163 - categorical_accuracy: 0.9345
43616/60000 [====================>.........] - ETA: 22s - loss: 0.2161 - categorical_accuracy: 0.9346
43680/60000 [====================>.........] - ETA: 22s - loss: 0.2161 - categorical_accuracy: 0.9346
43744/60000 [====================>.........] - ETA: 22s - loss: 0.2158 - categorical_accuracy: 0.9347
43808/60000 [====================>.........] - ETA: 22s - loss: 0.2158 - categorical_accuracy: 0.9347
43872/60000 [====================>.........] - ETA: 22s - loss: 0.2156 - categorical_accuracy: 0.9347
43936/60000 [====================>.........] - ETA: 22s - loss: 0.2154 - categorical_accuracy: 0.9347
44000/60000 [=====================>........] - ETA: 22s - loss: 0.2153 - categorical_accuracy: 0.9348
44064/60000 [=====================>........] - ETA: 22s - loss: 0.2151 - categorical_accuracy: 0.9348
44128/60000 [=====================>........] - ETA: 22s - loss: 0.2150 - categorical_accuracy: 0.9348
44192/60000 [=====================>........] - ETA: 22s - loss: 0.2149 - categorical_accuracy: 0.9349
44256/60000 [=====================>........] - ETA: 22s - loss: 0.2150 - categorical_accuracy: 0.9349
44320/60000 [=====================>........] - ETA: 21s - loss: 0.2149 - categorical_accuracy: 0.9349
44384/60000 [=====================>........] - ETA: 21s - loss: 0.2147 - categorical_accuracy: 0.9350
44448/60000 [=====================>........] - ETA: 21s - loss: 0.2146 - categorical_accuracy: 0.9350
44512/60000 [=====================>........] - ETA: 21s - loss: 0.2145 - categorical_accuracy: 0.9351
44576/60000 [=====================>........] - ETA: 21s - loss: 0.2142 - categorical_accuracy: 0.9351
44640/60000 [=====================>........] - ETA: 21s - loss: 0.2140 - categorical_accuracy: 0.9352
44704/60000 [=====================>........] - ETA: 21s - loss: 0.2138 - categorical_accuracy: 0.9353
44768/60000 [=====================>........] - ETA: 21s - loss: 0.2136 - categorical_accuracy: 0.9353
44832/60000 [=====================>........] - ETA: 21s - loss: 0.2137 - categorical_accuracy: 0.9353
44896/60000 [=====================>........] - ETA: 21s - loss: 0.2134 - categorical_accuracy: 0.9354
44960/60000 [=====================>........] - ETA: 21s - loss: 0.2132 - categorical_accuracy: 0.9355
45024/60000 [=====================>........] - ETA: 20s - loss: 0.2131 - categorical_accuracy: 0.9355
45088/60000 [=====================>........] - ETA: 20s - loss: 0.2129 - categorical_accuracy: 0.9356
45152/60000 [=====================>........] - ETA: 20s - loss: 0.2126 - categorical_accuracy: 0.9356
45216/60000 [=====================>........] - ETA: 20s - loss: 0.2124 - categorical_accuracy: 0.9357
45280/60000 [=====================>........] - ETA: 20s - loss: 0.2123 - categorical_accuracy: 0.9357
45344/60000 [=====================>........] - ETA: 20s - loss: 0.2122 - categorical_accuracy: 0.9357
45408/60000 [=====================>........] - ETA: 20s - loss: 0.2120 - categorical_accuracy: 0.9358
45472/60000 [=====================>........] - ETA: 20s - loss: 0.2118 - categorical_accuracy: 0.9359
45536/60000 [=====================>........] - ETA: 20s - loss: 0.2119 - categorical_accuracy: 0.9359
45600/60000 [=====================>........] - ETA: 20s - loss: 0.2118 - categorical_accuracy: 0.9359
45664/60000 [=====================>........] - ETA: 20s - loss: 0.2116 - categorical_accuracy: 0.9359
45728/60000 [=====================>........] - ETA: 19s - loss: 0.2114 - categorical_accuracy: 0.9360
45792/60000 [=====================>........] - ETA: 19s - loss: 0.2112 - categorical_accuracy: 0.9361
45856/60000 [=====================>........] - ETA: 19s - loss: 0.2112 - categorical_accuracy: 0.9361
45920/60000 [=====================>........] - ETA: 19s - loss: 0.2110 - categorical_accuracy: 0.9361
45984/60000 [=====================>........] - ETA: 19s - loss: 0.2110 - categorical_accuracy: 0.9362
46048/60000 [======================>.......] - ETA: 19s - loss: 0.2108 - categorical_accuracy: 0.9362
46112/60000 [======================>.......] - ETA: 19s - loss: 0.2108 - categorical_accuracy: 0.9362
46176/60000 [======================>.......] - ETA: 19s - loss: 0.2105 - categorical_accuracy: 0.9363
46240/60000 [======================>.......] - ETA: 19s - loss: 0.2103 - categorical_accuracy: 0.9364
46304/60000 [======================>.......] - ETA: 19s - loss: 0.2102 - categorical_accuracy: 0.9364
46368/60000 [======================>.......] - ETA: 19s - loss: 0.2099 - categorical_accuracy: 0.9365
46432/60000 [======================>.......] - ETA: 18s - loss: 0.2098 - categorical_accuracy: 0.9365
46496/60000 [======================>.......] - ETA: 18s - loss: 0.2096 - categorical_accuracy: 0.9366
46560/60000 [======================>.......] - ETA: 18s - loss: 0.2093 - categorical_accuracy: 0.9367
46624/60000 [======================>.......] - ETA: 18s - loss: 0.2091 - categorical_accuracy: 0.9367
46688/60000 [======================>.......] - ETA: 18s - loss: 0.2092 - categorical_accuracy: 0.9367
46752/60000 [======================>.......] - ETA: 18s - loss: 0.2093 - categorical_accuracy: 0.9367
46816/60000 [======================>.......] - ETA: 18s - loss: 0.2092 - categorical_accuracy: 0.9367
46880/60000 [======================>.......] - ETA: 18s - loss: 0.2090 - categorical_accuracy: 0.9368
46944/60000 [======================>.......] - ETA: 18s - loss: 0.2090 - categorical_accuracy: 0.9368
47008/60000 [======================>.......] - ETA: 18s - loss: 0.2089 - categorical_accuracy: 0.9368
47072/60000 [======================>.......] - ETA: 18s - loss: 0.2088 - categorical_accuracy: 0.9368
47136/60000 [======================>.......] - ETA: 18s - loss: 0.2086 - categorical_accuracy: 0.9369
47168/60000 [======================>.......] - ETA: 17s - loss: 0.2085 - categorical_accuracy: 0.9369
47232/60000 [======================>.......] - ETA: 17s - loss: 0.2083 - categorical_accuracy: 0.9370
47296/60000 [======================>.......] - ETA: 17s - loss: 0.2083 - categorical_accuracy: 0.9370
47360/60000 [======================>.......] - ETA: 17s - loss: 0.2082 - categorical_accuracy: 0.9370
47424/60000 [======================>.......] - ETA: 17s - loss: 0.2080 - categorical_accuracy: 0.9371
47488/60000 [======================>.......] - ETA: 17s - loss: 0.2079 - categorical_accuracy: 0.9371
47552/60000 [======================>.......] - ETA: 17s - loss: 0.2077 - categorical_accuracy: 0.9372
47616/60000 [======================>.......] - ETA: 17s - loss: 0.2075 - categorical_accuracy: 0.9372
47680/60000 [======================>.......] - ETA: 17s - loss: 0.2073 - categorical_accuracy: 0.9373
47712/60000 [======================>.......] - ETA: 17s - loss: 0.2072 - categorical_accuracy: 0.9373
47776/60000 [======================>.......] - ETA: 17s - loss: 0.2072 - categorical_accuracy: 0.9373
47840/60000 [======================>.......] - ETA: 17s - loss: 0.2073 - categorical_accuracy: 0.9373
47904/60000 [======================>.......] - ETA: 16s - loss: 0.2071 - categorical_accuracy: 0.9374
47968/60000 [======================>.......] - ETA: 16s - loss: 0.2069 - categorical_accuracy: 0.9375
48032/60000 [=======================>......] - ETA: 16s - loss: 0.2068 - categorical_accuracy: 0.9376
48096/60000 [=======================>......] - ETA: 16s - loss: 0.2065 - categorical_accuracy: 0.9376
48160/60000 [=======================>......] - ETA: 16s - loss: 0.2064 - categorical_accuracy: 0.9377
48224/60000 [=======================>......] - ETA: 16s - loss: 0.2062 - categorical_accuracy: 0.9378
48288/60000 [=======================>......] - ETA: 16s - loss: 0.2060 - categorical_accuracy: 0.9378
48352/60000 [=======================>......] - ETA: 16s - loss: 0.2059 - categorical_accuracy: 0.9379
48416/60000 [=======================>......] - ETA: 16s - loss: 0.2057 - categorical_accuracy: 0.9379
48448/60000 [=======================>......] - ETA: 16s - loss: 0.2056 - categorical_accuracy: 0.9379
48512/60000 [=======================>......] - ETA: 16s - loss: 0.2055 - categorical_accuracy: 0.9380
48576/60000 [=======================>......] - ETA: 15s - loss: 0.2053 - categorical_accuracy: 0.9381
48640/60000 [=======================>......] - ETA: 15s - loss: 0.2051 - categorical_accuracy: 0.9381
48704/60000 [=======================>......] - ETA: 15s - loss: 0.2049 - categorical_accuracy: 0.9381
48768/60000 [=======================>......] - ETA: 15s - loss: 0.2049 - categorical_accuracy: 0.9382
48832/60000 [=======================>......] - ETA: 15s - loss: 0.2046 - categorical_accuracy: 0.9382
48864/60000 [=======================>......] - ETA: 15s - loss: 0.2045 - categorical_accuracy: 0.9383
48928/60000 [=======================>......] - ETA: 15s - loss: 0.2044 - categorical_accuracy: 0.9383
48992/60000 [=======================>......] - ETA: 15s - loss: 0.2042 - categorical_accuracy: 0.9384
49056/60000 [=======================>......] - ETA: 15s - loss: 0.2039 - categorical_accuracy: 0.9384
49120/60000 [=======================>......] - ETA: 15s - loss: 0.2037 - categorical_accuracy: 0.9385
49184/60000 [=======================>......] - ETA: 15s - loss: 0.2038 - categorical_accuracy: 0.9385
49248/60000 [=======================>......] - ETA: 15s - loss: 0.2037 - categorical_accuracy: 0.9385
49312/60000 [=======================>......] - ETA: 14s - loss: 0.2035 - categorical_accuracy: 0.9386
49376/60000 [=======================>......] - ETA: 14s - loss: 0.2033 - categorical_accuracy: 0.9387
49440/60000 [=======================>......] - ETA: 14s - loss: 0.2033 - categorical_accuracy: 0.9387
49504/60000 [=======================>......] - ETA: 14s - loss: 0.2031 - categorical_accuracy: 0.9388
49568/60000 [=======================>......] - ETA: 14s - loss: 0.2029 - categorical_accuracy: 0.9388
49632/60000 [=======================>......] - ETA: 14s - loss: 0.2030 - categorical_accuracy: 0.9388
49696/60000 [=======================>......] - ETA: 14s - loss: 0.2029 - categorical_accuracy: 0.9388
49760/60000 [=======================>......] - ETA: 14s - loss: 0.2027 - categorical_accuracy: 0.9389
49824/60000 [=======================>......] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9389
49888/60000 [=======================>......] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9390
49952/60000 [=======================>......] - ETA: 14s - loss: 0.2022 - categorical_accuracy: 0.9390
50016/60000 [========================>.....] - ETA: 13s - loss: 0.2021 - categorical_accuracy: 0.9390
50080/60000 [========================>.....] - ETA: 13s - loss: 0.2018 - categorical_accuracy: 0.9391
50144/60000 [========================>.....] - ETA: 13s - loss: 0.2017 - categorical_accuracy: 0.9391
50208/60000 [========================>.....] - ETA: 13s - loss: 0.2017 - categorical_accuracy: 0.9392
50272/60000 [========================>.....] - ETA: 13s - loss: 0.2015 - categorical_accuracy: 0.9392
50336/60000 [========================>.....] - ETA: 13s - loss: 0.2015 - categorical_accuracy: 0.9392
50400/60000 [========================>.....] - ETA: 13s - loss: 0.2014 - categorical_accuracy: 0.9392
50464/60000 [========================>.....] - ETA: 13s - loss: 0.2013 - categorical_accuracy: 0.9392
50528/60000 [========================>.....] - ETA: 13s - loss: 0.2011 - categorical_accuracy: 0.9393
50592/60000 [========================>.....] - ETA: 13s - loss: 0.2010 - categorical_accuracy: 0.9393
50656/60000 [========================>.....] - ETA: 13s - loss: 0.2008 - categorical_accuracy: 0.9394
50720/60000 [========================>.....] - ETA: 12s - loss: 0.2006 - categorical_accuracy: 0.9394
50784/60000 [========================>.....] - ETA: 12s - loss: 0.2005 - categorical_accuracy: 0.9394
50848/60000 [========================>.....] - ETA: 12s - loss: 0.2005 - categorical_accuracy: 0.9394
50912/60000 [========================>.....] - ETA: 12s - loss: 0.2004 - categorical_accuracy: 0.9395
50976/60000 [========================>.....] - ETA: 12s - loss: 0.2002 - categorical_accuracy: 0.9395
51040/60000 [========================>.....] - ETA: 12s - loss: 0.2000 - categorical_accuracy: 0.9396
51104/60000 [========================>.....] - ETA: 12s - loss: 0.1998 - categorical_accuracy: 0.9397
51168/60000 [========================>.....] - ETA: 12s - loss: 0.1997 - categorical_accuracy: 0.9397
51232/60000 [========================>.....] - ETA: 12s - loss: 0.1995 - categorical_accuracy: 0.9397
51264/60000 [========================>.....] - ETA: 12s - loss: 0.1995 - categorical_accuracy: 0.9397
51328/60000 [========================>.....] - ETA: 12s - loss: 0.1993 - categorical_accuracy: 0.9398
51392/60000 [========================>.....] - ETA: 12s - loss: 0.1991 - categorical_accuracy: 0.9399
51456/60000 [========================>.....] - ETA: 11s - loss: 0.1990 - categorical_accuracy: 0.9399
51488/60000 [========================>.....] - ETA: 11s - loss: 0.1989 - categorical_accuracy: 0.9399
51552/60000 [========================>.....] - ETA: 11s - loss: 0.1987 - categorical_accuracy: 0.9400
51616/60000 [========================>.....] - ETA: 11s - loss: 0.1986 - categorical_accuracy: 0.9400
51680/60000 [========================>.....] - ETA: 11s - loss: 0.1988 - categorical_accuracy: 0.9400
51744/60000 [========================>.....] - ETA: 11s - loss: 0.1986 - categorical_accuracy: 0.9401
51808/60000 [========================>.....] - ETA: 11s - loss: 0.1984 - categorical_accuracy: 0.9401
51872/60000 [========================>.....] - ETA: 11s - loss: 0.1982 - categorical_accuracy: 0.9402
51936/60000 [========================>.....] - ETA: 11s - loss: 0.1980 - categorical_accuracy: 0.9402
51968/60000 [========================>.....] - ETA: 11s - loss: 0.1981 - categorical_accuracy: 0.9402
52032/60000 [=========================>....] - ETA: 11s - loss: 0.1979 - categorical_accuracy: 0.9402
52096/60000 [=========================>....] - ETA: 11s - loss: 0.1978 - categorical_accuracy: 0.9402
52160/60000 [=========================>....] - ETA: 10s - loss: 0.1976 - categorical_accuracy: 0.9403
52192/60000 [=========================>....] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9403
52256/60000 [=========================>....] - ETA: 10s - loss: 0.1976 - categorical_accuracy: 0.9403
52320/60000 [=========================>....] - ETA: 10s - loss: 0.1976 - categorical_accuracy: 0.9403
52384/60000 [=========================>....] - ETA: 10s - loss: 0.1974 - categorical_accuracy: 0.9404
52448/60000 [=========================>....] - ETA: 10s - loss: 0.1974 - categorical_accuracy: 0.9404
52512/60000 [=========================>....] - ETA: 10s - loss: 0.1973 - categorical_accuracy: 0.9405
52576/60000 [=========================>....] - ETA: 10s - loss: 0.1972 - categorical_accuracy: 0.9405
52640/60000 [=========================>....] - ETA: 10s - loss: 0.1971 - categorical_accuracy: 0.9405
52704/60000 [=========================>....] - ETA: 10s - loss: 0.1969 - categorical_accuracy: 0.9406
52768/60000 [=========================>....] - ETA: 10s - loss: 0.1967 - categorical_accuracy: 0.9406
52832/60000 [=========================>....] - ETA: 10s - loss: 0.1965 - categorical_accuracy: 0.9407
52896/60000 [=========================>....] - ETA: 9s - loss: 0.1964 - categorical_accuracy: 0.9407 
52960/60000 [=========================>....] - ETA: 9s - loss: 0.1962 - categorical_accuracy: 0.9408
53024/60000 [=========================>....] - ETA: 9s - loss: 0.1961 - categorical_accuracy: 0.9409
53088/60000 [=========================>....] - ETA: 9s - loss: 0.1961 - categorical_accuracy: 0.9409
53152/60000 [=========================>....] - ETA: 9s - loss: 0.1960 - categorical_accuracy: 0.9409
53216/60000 [=========================>....] - ETA: 9s - loss: 0.1958 - categorical_accuracy: 0.9410
53280/60000 [=========================>....] - ETA: 9s - loss: 0.1956 - categorical_accuracy: 0.9410
53344/60000 [=========================>....] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9411
53408/60000 [=========================>....] - ETA: 9s - loss: 0.1953 - categorical_accuracy: 0.9411
53472/60000 [=========================>....] - ETA: 9s - loss: 0.1952 - categorical_accuracy: 0.9411
53536/60000 [=========================>....] - ETA: 9s - loss: 0.1952 - categorical_accuracy: 0.9411
53600/60000 [=========================>....] - ETA: 8s - loss: 0.1951 - categorical_accuracy: 0.9412
53664/60000 [=========================>....] - ETA: 8s - loss: 0.1950 - categorical_accuracy: 0.9412
53696/60000 [=========================>....] - ETA: 8s - loss: 0.1949 - categorical_accuracy: 0.9412
53760/60000 [=========================>....] - ETA: 8s - loss: 0.1948 - categorical_accuracy: 0.9412
53824/60000 [=========================>....] - ETA: 8s - loss: 0.1947 - categorical_accuracy: 0.9413
53888/60000 [=========================>....] - ETA: 8s - loss: 0.1946 - categorical_accuracy: 0.9413
53952/60000 [=========================>....] - ETA: 8s - loss: 0.1944 - categorical_accuracy: 0.9414
54016/60000 [==========================>...] - ETA: 8s - loss: 0.1943 - categorical_accuracy: 0.9414
54080/60000 [==========================>...] - ETA: 8s - loss: 0.1942 - categorical_accuracy: 0.9415
54144/60000 [==========================>...] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9415
54208/60000 [==========================>...] - ETA: 8s - loss: 0.1939 - categorical_accuracy: 0.9415
54272/60000 [==========================>...] - ETA: 8s - loss: 0.1939 - categorical_accuracy: 0.9415
54336/60000 [==========================>...] - ETA: 7s - loss: 0.1938 - categorical_accuracy: 0.9416
54400/60000 [==========================>...] - ETA: 7s - loss: 0.1937 - categorical_accuracy: 0.9416
54464/60000 [==========================>...] - ETA: 7s - loss: 0.1937 - categorical_accuracy: 0.9416
54528/60000 [==========================>...] - ETA: 7s - loss: 0.1936 - categorical_accuracy: 0.9417
54592/60000 [==========================>...] - ETA: 7s - loss: 0.1934 - categorical_accuracy: 0.9417
54656/60000 [==========================>...] - ETA: 7s - loss: 0.1933 - categorical_accuracy: 0.9418
54720/60000 [==========================>...] - ETA: 7s - loss: 0.1931 - categorical_accuracy: 0.9418
54784/60000 [==========================>...] - ETA: 7s - loss: 0.1929 - categorical_accuracy: 0.9418
54848/60000 [==========================>...] - ETA: 7s - loss: 0.1928 - categorical_accuracy: 0.9419
54912/60000 [==========================>...] - ETA: 7s - loss: 0.1927 - categorical_accuracy: 0.9419
54976/60000 [==========================>...] - ETA: 7s - loss: 0.1927 - categorical_accuracy: 0.9419
55040/60000 [==========================>...] - ETA: 6s - loss: 0.1925 - categorical_accuracy: 0.9420
55104/60000 [==========================>...] - ETA: 6s - loss: 0.1924 - categorical_accuracy: 0.9420
55168/60000 [==========================>...] - ETA: 6s - loss: 0.1923 - categorical_accuracy: 0.9420
55232/60000 [==========================>...] - ETA: 6s - loss: 0.1921 - categorical_accuracy: 0.9421
55296/60000 [==========================>...] - ETA: 6s - loss: 0.1919 - categorical_accuracy: 0.9422
55328/60000 [==========================>...] - ETA: 6s - loss: 0.1918 - categorical_accuracy: 0.9422
55392/60000 [==========================>...] - ETA: 6s - loss: 0.1917 - categorical_accuracy: 0.9422
55456/60000 [==========================>...] - ETA: 6s - loss: 0.1916 - categorical_accuracy: 0.9423
55520/60000 [==========================>...] - ETA: 6s - loss: 0.1914 - categorical_accuracy: 0.9423
55584/60000 [==========================>...] - ETA: 6s - loss: 0.1912 - categorical_accuracy: 0.9424
55648/60000 [==========================>...] - ETA: 6s - loss: 0.1912 - categorical_accuracy: 0.9424
55712/60000 [==========================>...] - ETA: 6s - loss: 0.1911 - categorical_accuracy: 0.9424
55776/60000 [==========================>...] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9425
55840/60000 [==========================>...] - ETA: 5s - loss: 0.1908 - categorical_accuracy: 0.9425
55904/60000 [==========================>...] - ETA: 5s - loss: 0.1908 - categorical_accuracy: 0.9425
55968/60000 [==========================>...] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9425
56032/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9426
56096/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9426
56160/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9426
56224/60000 [===========================>..] - ETA: 5s - loss: 0.1905 - categorical_accuracy: 0.9427
56288/60000 [===========================>..] - ETA: 5s - loss: 0.1904 - categorical_accuracy: 0.9427
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1903 - categorical_accuracy: 0.9427
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1901 - categorical_accuracy: 0.9428
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1900 - categorical_accuracy: 0.9428
56480/60000 [===========================>..] - ETA: 4s - loss: 0.1900 - categorical_accuracy: 0.9428
56544/60000 [===========================>..] - ETA: 4s - loss: 0.1899 - categorical_accuracy: 0.9428
56608/60000 [===========================>..] - ETA: 4s - loss: 0.1897 - categorical_accuracy: 0.9429
56672/60000 [===========================>..] - ETA: 4s - loss: 0.1896 - categorical_accuracy: 0.9429
56736/60000 [===========================>..] - ETA: 4s - loss: 0.1894 - categorical_accuracy: 0.9430
56800/60000 [===========================>..] - ETA: 4s - loss: 0.1893 - categorical_accuracy: 0.9430
56864/60000 [===========================>..] - ETA: 4s - loss: 0.1891 - categorical_accuracy: 0.9430
56928/60000 [===========================>..] - ETA: 4s - loss: 0.1891 - categorical_accuracy: 0.9431
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1890 - categorical_accuracy: 0.9431
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1889 - categorical_accuracy: 0.9431
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1888 - categorical_accuracy: 0.9432
57184/60000 [===========================>..] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9432
57248/60000 [===========================>..] - ETA: 3s - loss: 0.1886 - categorical_accuracy: 0.9432
57312/60000 [===========================>..] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9432
57376/60000 [===========================>..] - ETA: 3s - loss: 0.1885 - categorical_accuracy: 0.9433
57440/60000 [===========================>..] - ETA: 3s - loss: 0.1884 - categorical_accuracy: 0.9433
57504/60000 [===========================>..] - ETA: 3s - loss: 0.1882 - categorical_accuracy: 0.9433
57568/60000 [===========================>..] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9434
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1879 - categorical_accuracy: 0.9434
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1878 - categorical_accuracy: 0.9435
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1876 - categorical_accuracy: 0.9435
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1875 - categorical_accuracy: 0.9436
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1875 - categorical_accuracy: 0.9436
57920/60000 [===========================>..] - ETA: 2s - loss: 0.1873 - categorical_accuracy: 0.9436
57984/60000 [===========================>..] - ETA: 2s - loss: 0.1872 - categorical_accuracy: 0.9436
58048/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9437
58080/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9437
58144/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9437
58208/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9438
58272/60000 [============================>.] - ETA: 2s - loss: 0.1867 - categorical_accuracy: 0.9438
58336/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9438
58400/60000 [============================>.] - ETA: 2s - loss: 0.1864 - categorical_accuracy: 0.9439
58464/60000 [============================>.] - ETA: 2s - loss: 0.1862 - categorical_accuracy: 0.9439
58528/60000 [============================>.] - ETA: 2s - loss: 0.1861 - categorical_accuracy: 0.9440
58592/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9440
58656/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9440
58720/60000 [============================>.] - ETA: 1s - loss: 0.1856 - categorical_accuracy: 0.9440
58784/60000 [============================>.] - ETA: 1s - loss: 0.1857 - categorical_accuracy: 0.9441
58848/60000 [============================>.] - ETA: 1s - loss: 0.1855 - categorical_accuracy: 0.9441
58912/60000 [============================>.] - ETA: 1s - loss: 0.1854 - categorical_accuracy: 0.9441
58976/60000 [============================>.] - ETA: 1s - loss: 0.1852 - categorical_accuracy: 0.9442
59040/60000 [============================>.] - ETA: 1s - loss: 0.1851 - categorical_accuracy: 0.9442
59104/60000 [============================>.] - ETA: 1s - loss: 0.1849 - categorical_accuracy: 0.9443
59168/60000 [============================>.] - ETA: 1s - loss: 0.1850 - categorical_accuracy: 0.9443
59232/60000 [============================>.] - ETA: 1s - loss: 0.1849 - categorical_accuracy: 0.9443
59296/60000 [============================>.] - ETA: 0s - loss: 0.1848 - categorical_accuracy: 0.9443
59360/60000 [============================>.] - ETA: 0s - loss: 0.1846 - categorical_accuracy: 0.9444
59424/60000 [============================>.] - ETA: 0s - loss: 0.1845 - categorical_accuracy: 0.9444
59488/60000 [============================>.] - ETA: 0s - loss: 0.1843 - categorical_accuracy: 0.9445
59552/60000 [============================>.] - ETA: 0s - loss: 0.1842 - categorical_accuracy: 0.9445
59616/60000 [============================>.] - ETA: 0s - loss: 0.1841 - categorical_accuracy: 0.9445
59680/60000 [============================>.] - ETA: 0s - loss: 0.1839 - categorical_accuracy: 0.9446
59744/60000 [============================>.] - ETA: 0s - loss: 0.1839 - categorical_accuracy: 0.9446
59808/60000 [============================>.] - ETA: 0s - loss: 0.1838 - categorical_accuracy: 0.9446
59872/60000 [============================>.] - ETA: 0s - loss: 0.1836 - categorical_accuracy: 0.9446
59936/60000 [============================>.] - ETA: 0s - loss: 0.1837 - categorical_accuracy: 0.9447
60000/60000 [==============================] - 87s 1ms/step - loss: 0.1836 - categorical_accuracy: 0.9447 - val_loss: 0.0434 - val_categorical_accuracy: 0.9869

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  224/10000 [..............................] - ETA: 4s 
  448/10000 [>.............................] - ETA: 3s
  640/10000 [>.............................] - ETA: 3s
  832/10000 [=>............................] - ETA: 2s
  992/10000 [=>............................] - ETA: 2s
 1184/10000 [==>...........................] - ETA: 2s
 1376/10000 [===>..........................] - ETA: 2s
 1568/10000 [===>..........................] - ETA: 2s
 1792/10000 [====>.........................] - ETA: 2s
 2016/10000 [=====>........................] - ETA: 2s
 2240/10000 [=====>........................] - ETA: 2s
 2464/10000 [======>.......................] - ETA: 2s
 2688/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 1s
 3136/10000 [========>.....................] - ETA: 1s
 3360/10000 [=========>....................] - ETA: 1s
 3584/10000 [=========>....................] - ETA: 1s
 3776/10000 [==========>...................] - ETA: 1s
 4000/10000 [===========>..................] - ETA: 1s
 4192/10000 [===========>..................] - ETA: 1s
 4416/10000 [============>.................] - ETA: 1s
 4640/10000 [============>.................] - ETA: 1s
 4864/10000 [=============>................] - ETA: 1s
 5088/10000 [==============>...............] - ETA: 1s
 5312/10000 [==============>...............] - ETA: 1s
 5536/10000 [===============>..............] - ETA: 1s
 5760/10000 [================>.............] - ETA: 1s
 5984/10000 [================>.............] - ETA: 1s
 6208/10000 [=================>............] - ETA: 1s
 6432/10000 [==================>...........] - ETA: 0s
 6656/10000 [==================>...........] - ETA: 0s
 6880/10000 [===================>..........] - ETA: 0s
 7072/10000 [====================>.........] - ETA: 0s
 7296/10000 [====================>.........] - ETA: 0s
 7488/10000 [=====================>........] - ETA: 0s
 7680/10000 [======================>.......] - ETA: 0s
 7904/10000 [======================>.......] - ETA: 0s
 8128/10000 [=======================>......] - ETA: 0s
 8352/10000 [========================>.....] - ETA: 0s
 8576/10000 [========================>.....] - ETA: 0s
 8800/10000 [=========================>....] - ETA: 0s
 9024/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9472/10000 [===========================>..] - ETA: 0s
 9696/10000 [============================>.] - ETA: 0s
 9920/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 261us/step
[[9.42266212e-08 1.41840900e-07 3.46011848e-06 ... 9.99988198e-01
  1.49755923e-08 3.85996054e-06]
 [4.18704531e-06 1.39708272e-05 9.99971509e-01 ... 2.33990196e-08
  9.18829301e-07 2.57154520e-09]
 [1.06239781e-06 9.99508858e-01 1.23378602e-04 ... 1.08480541e-04
  1.45576896e-05 2.92730738e-06]
 ...
 [2.11298605e-08 4.67221616e-06 1.11404368e-07 ... 1.77305974e-05
  7.38508925e-06 2.32347156e-04]
 [5.92110746e-06 7.46981357e-08 1.06351273e-07 ... 8.55577866e-07
  1.80876686e-03 3.55218845e-06]
 [1.13016422e-05 4.23998131e-07 1.42754325e-05 ... 9.70055503e-09
  9.72874773e-07 2.22121741e-07]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.043360872592497615, 'accuracy_test:': 0.9868999719619751}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

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
Warning: Permanently added the RSA host key for IP address '140.82.118.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   9991828..5806b66  master     -> origin/master
Updating 9991828..5806b66
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |  2 +-
 error_list/20200522/list_log_testall_20200522.md   | 36 ++++++++++++++++++++++
 2 files changed, 37 insertions(+), 1 deletion(-)
