
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
[master 9b981d3] ml_store
 1 file changed, 1118 insertions(+)
To github.com:arita37/mlmodels_store.git
   5806b66..9b981d3  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//nbeats.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Using TensorFlow backend.
Loading data...
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//nbeats.py", line 315, in <module>
    test(pars_choice="test01")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//nbeats.py", line 278, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//nbeats.py", line 172, in get_dataset
    train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
NameError: name 'Data' is not defined

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
[master db0098c] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   9b981d3..db0098c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py 

  #### Loading params   ############################################## 

  #### Loading dataset   ############################################# 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py", line 348, in <module>
    test(pars_choice="json", data_path=f"model_keras/namentity_crm_bilstm.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py", line 311, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm.py", line 193, in get_dataset
    raise Exception(f"Not support dataset yet")
Exception: Not support dataset yet

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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
Already up to date.
[master 079a207] ml_store
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   db0098c..079a207  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Using TensorFlow backend.
Loading data...
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 357, in <module>
    test(pars_choice="test01")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 320, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn.py", line 216, in get_dataset
    if data_pars['type'] == "npz":
KeyError: 'type'

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
[master c17c7a3] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   079a207..c17c7a3  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py 

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.namentity_crm_bilstm_dataloader' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py'> 

  #### Loading params   ############################################## 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//namentity_crm_bilstm_dataloader.py", line 306, in <module>
    test_module(model_uri=MODEL_URI, param_pars=param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 257, in test_module
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/namentity_crm_bilstm_dataloader.py", line 197, in get_params
    cf = json.load(open(data_path, mode="r"))
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/namentity_crm_bilstm_dataloader.json'

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
[master a67d80b] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   c17c7a3..a67d80b  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//keras_gan.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//keras_gan.py", line 31, in <module>
    'AAE' : kg.aae.aae,
AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'

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
[master 46cfa4a] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   a67d80b..46cfa4a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//01_deepctr.py 

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'AFM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'AFM', 'sparse_feature_num': 3, 'dense_feature_num': 0} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_AFM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/initializers.py:143: calling RandomNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/sequence.py:159: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/utils.py:199: calling softmax (from tensorflow.python.ops.nn_ops) with dim is deprecated and will be removed in a future version.
Instructions for updating:
dim is deprecated, use axis instead
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/utils.py:163: calling reduce_sum_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/utils.py:193: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/utils.py:180: calling reduce_max_v1 (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_4 (Seque (None, 1, 1)         0           weighted_sequence_layer_1[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_5 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_6 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_7 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
no_mask (NoMask)                (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_4[0][0]   
                                                                 sequence_pooling_layer_5[0][0]   
                                                                 sequence_pooling_layer_6[0][0]   
                                                                 sequence_pooling_layer_7[0][0]   
__________________________________________________________________________________________________
weighted_sequence_layer (Weight (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-22 12:54:15.445498: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-22 12:54:15.450554: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-22 12:54:15.450787: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5641c4481730 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 12:54:15.450802: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_1 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_2 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_3 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear (Linear)                 (None, 1, 1)         0           concatenate[0][0]                
__________________________________________________________________________________________________
afm_layer (AFMLayer)            (None, 1)            52          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer[0][0]     
                                                                 sequence_pooling_layer_1[0][0]   
                                                                 sequence_pooling_layer_2[0][0]   
                                                                 sequence_pooling_layer_3[0][0]   
__________________________________________________________________________________________________
no_mask_1 (NoMask)              (None, 1, 1)         0           linear[0][0]                     
__________________________________________________________________________________________________
add (Add)                       (None, 1)            0           afm_layer[0][0]                  
__________________________________________________________________________________________________
add_1 (Add)                     (None, 1, 1)         0           no_mask_1[0][0]                  
                                                                 add[0][0]                        
__________________________________________________________________________________________________
prediction_layer (PredictionLay (None, 1)            1           add_1[0][0]                      
==================================================================================================
Total params: 178
Trainable params: 178
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2501 - binary_crossentropy: 0.6932 - val_loss: 0.2501 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.2499350572766176}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_4 (Seque (None, 1, 1)         0           weighted_sequence_layer_1[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_5 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_6 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_7 (Seque (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
no_mask (NoMask)                (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_4[0][0]   
                                                                 sequence_pooling_layer_5[0][0]   
                                                                 sequence_pooling_layer_6[0][0]   
                                                                 sequence_pooling_layer_7[0][0]   
__________________________________________________________________________________________________
weighted_sequence_layer (Weight (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_1 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_2 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_3 (Seque (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear (Linear)                 (None, 1, 1)         0           concatenate[0][0]                
__________________________________________________________________________________________________
afm_layer (AFMLayer)            (None, 1)            52          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer[0][0]     
                                                                 sequence_pooling_layer_1[0][0]   
                                                                 sequence_pooling_layer_2[0][0]   
                                                                 sequence_pooling_layer_3[0][0]   
__________________________________________________________________________________________________
no_mask_1 (NoMask)              (None, 1, 1)         0           linear[0][0]                     
__________________________________________________________________________________________________
add (Add)                       (None, 1)            0           afm_layer[0][0]                  
__________________________________________________________________________________________________
add_1 (Add)                     (None, 1, 1)         0           no_mask_1[0][0]                  
                                                                 add[0][0]                        
__________________________________________________________________________________________________
prediction_layer (PredictionLay (None, 1)            1           add_1[0][0]                      
==================================================================================================
Total params: 178
Trainable params: 178
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'AutoInt', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'AutoInt', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_AutoInt.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/interaction.py:565: The name tf.keras.initializers.TruncatedNormal is deprecated. Please use tf.compat.v1.keras.initializers.TruncatedNormal instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/initializers.py:94: calling TruncatedNormal.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_12 (Sequ (None, 1, 4)         0           weighted_sequence_layer_3[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_13 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_14 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_15 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
weighted_sequence_layer_4 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_16 (Sequ (None, 1, 1)         0           weighted_sequence_layer_4[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_17 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_18 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_19 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 5, 4)         0           no_mask_5[0][0]                  
                                                                 no_mask_5[1][0]                  
                                                                 no_mask_5[2][0]                  
                                                                 no_mask_5[3][0]                  
                                                                 no_mask_5[4][0]                  
__________________________________________________________________________________________________
no_mask_2 (NoMask)              (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_16[0][0]  
                                                                 sequence_pooling_layer_17[0][0]  
                                                                 sequence_pooling_layer_18[0][0]  
                                                                 sequence_pooling_layer_19[0][0]  
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
interacting_layer (InteractingL (None, 5, 16)        256         concatenate_2[0][0]              
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1, 5)         0           no_mask_2[0][0]                  
                                                                 no_mask_2[1][0]                  
                                                                 no_mask_2[2][0]                  
                                                                 no_mask_2[3][0]                  
                                                                 no_mask_2[4][0]                  
__________________________________________________________________________________________________
no_mask_3 (NoMask)              (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
flatten (Flatten)               (None, 80)           0           interacting_layer[0][0]          
__________________________________________________________________________________________________
linear_1 (Linear)               (None, 1)            1           concatenate_1[0][0]              
                                                                 no_mask_3[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 1)            80          flatten[0][0]                    
__________________________________________________________________________________________________
no_mask_4 (NoMask)              (None, 1)            0           linear_1[0][0]                   
__________________________________________________________________________________________________
add_4 (Add)                     (None, 1)            0           dense[0][0]                      
                                                                 no_mask_4[0][0]                  
__________________________________________________________________________________________________
prediction_layer_1 (PredictionL (None, 1)            1           add_4[0][0]                      
==================================================================================================
Total params: 433
Trainable params: 433
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.3040 - binary_crossentropy: 0.8105500/500 [==============================] - 1s 2ms/sample - loss: 0.2879 - binary_crossentropy: 0.7757 - val_loss: 0.2691 - val_binary_crossentropy: 0.7883

  #### metrics   #################################################### 
{'MSE': 0.27752616975591526}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/init_ops.py:97: calling GlorotUniform.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/init_ops.py:97: calling Zeros.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_12 (Sequ (None, 1, 4)         0           weighted_sequence_layer_3[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_13 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_14 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_15 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
weighted_sequence_layer_4 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_16 (Sequ (None, 1, 1)         0           weighted_sequence_layer_4[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_17 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_18 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_19 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 5, 4)         0           no_mask_5[0][0]                  
                                                                 no_mask_5[1][0]                  
                                                                 no_mask_5[2][0]                  
                                                                 no_mask_5[3][0]                  
                                                                 no_mask_5[4][0]                  
__________________________________________________________________________________________________
no_mask_2 (NoMask)              (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_16[0][0]  
                                                                 sequence_pooling_layer_17[0][0]  
                                                                 sequence_pooling_layer_18[0][0]  
                                                                 sequence_pooling_layer_19[0][0]  
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
interacting_layer (InteractingL (None, 5, 16)        256         concatenate_2[0][0]              
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1, 5)         0           no_mask_2[0][0]                  
                                                                 no_mask_2[1][0]                  
                                                                 no_mask_2[2][0]                  
                                                                 no_mask_2[3][0]                  
                                                                 no_mask_2[4][0]                  
__________________________________________________________________________________________________
no_mask_3 (NoMask)              (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
flatten (Flatten)               (None, 80)           0           interacting_layer[0][0]          
__________________________________________________________________________________________________
linear_1 (Linear)               (None, 1)            1           concatenate_1[0][0]              
                                                                 no_mask_3[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 1)            80          flatten[0][0]                    
__________________________________________________________________________________________________
no_mask_4 (NoMask)              (None, 1)            0           linear_1[0][0]                   
__________________________________________________________________________________________________
add_4 (Add)                     (None, 1)            0           dense[0][0]                      
                                                                 no_mask_4[0][0]                  
__________________________________________________________________________________________________
prediction_layer_1 (PredictionL (None, 1)            1           add_4[0][0]                      
==================================================================================================
Total params: 433
Trainable params: 433
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'CCPM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'CCPM', 'sparse_feature_num': 3, 'dense_feature_num': 0} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_CCPM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_24 (Sequ (None, 1, 4)         0           weighted_sequence_layer_6[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_25 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_26 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_27 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_11 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer_24[0][0]  
                                                                 sequence_pooling_layer_25[0][0]  
                                                                 sequence_pooling_layer_26[0][0]  
                                                                 sequence_pooling_layer_27[0][0]  
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 7, 4)         0           no_mask_11[0][0]                 
                                                                 no_mask_11[1][0]                 
                                                                 no_mask_11[2][0]                 
                                                                 no_mask_11[3][0]                 
                                                                 no_mask_11[4][0]                 
                                                                 no_mask_11[5][0]                 
                                                                 no_mask_11[6][0]                 
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 7, 4, 1)      0           concatenate_6[0][0]              
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 7, 4, 2)      8           lambda_2[0][0]                   
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
k_max_pooling (KMaxPooling)     (None, 3, 4, 2)      0           conv2d[0][0]                     
__________________________________________________________________________________________________
weighted_sequence_layer_7 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_28 (Sequ (None, 1, 1)         0           weighted_sequence_layer_7[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_29 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_30 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_31 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
k_max_pooling_1 (KMaxPooling)   (None, 3, 4, 1)      0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
no_mask_9 (NoMask)              (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_28[0][0]  
                                                                 sequence_pooling_layer_29[0][0]  
                                                                 sequence_pooling_layer_30[0][0]  
                                                                 sequence_pooling_layer_31[0][0]  
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 12)           0           k_max_pooling_1[0][0]            
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 1, 7)         0           no_mask_9[0][0]                  
                                                                 no_mask_9[1][0]                  
                                                                 no_mask_9[2][0]                  
                                                                 no_mask_9[3][0]                  
                                                                 no_mask_9[4][0]                  
                                                                 no_mask_9[5][0]                  
                                                                 no_mask_9[6][0]                  
__________________________________________________________________________________________________
dnn (DNN)                       (None, 32)           416         flatten_3[0][0]                  
__________________________________________________________________________________________________
linear_2 (Linear)               (None, 1, 1)         0           concatenate_5[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            32          dnn[0][0]                        
__________________________________________________________________________________________________
no_mask_10 (NoMask)             (None, 1, 1)         0           linear_2[0][0]                   
__________________________________________________________________________________________________
add_7 (Add)                     (None, 1, 1)         0           dense_1[0][0]                    
                                                                 no_mask_10[0][0]                 
__________________________________________________________________________________________________
prediction_layer_2 (PredictionL (None, 1)            1           add_7[0][0]                      
==================================================================================================
Total params: 592
Trainable params: 592
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2536 - binary_crossentropy: 0.7794 - val_loss: 0.2506 - val_binary_crossentropy: 0.7204

  #### metrics   #################################################### 
{'MSE': 0.2517563458987908}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_24 (Sequ (None, 1, 4)         0           weighted_sequence_layer_6[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_25 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_26 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_27 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_11 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer_24[0][0]  
                                                                 sequence_pooling_layer_25[0][0]  
                                                                 sequence_pooling_layer_26[0][0]  
                                                                 sequence_pooling_layer_27[0][0]  
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 7, 4)         0           no_mask_11[0][0]                 
                                                                 no_mask_11[1][0]                 
                                                                 no_mask_11[2][0]                 
                                                                 no_mask_11[3][0]                 
                                                                 no_mask_11[4][0]                 
                                                                 no_mask_11[5][0]                 
                                                                 no_mask_11[6][0]                 
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 7, 4, 1)      0           concatenate_6[0][0]              
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 7, 4, 2)      8           lambda_2[0][0]                   
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
k_max_pooling (KMaxPooling)     (None, 3, 4, 2)      0           conv2d[0][0]                     
__________________________________________________________________________________________________
weighted_sequence_layer_7 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_28 (Sequ (None, 1, 1)         0           weighted_sequence_layer_7[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_29 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_30 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_31 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
k_max_pooling_1 (KMaxPooling)   (None, 3, 4, 1)      0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
no_mask_9 (NoMask)              (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_28[0][0]  
                                                                 sequence_pooling_layer_29[0][0]  
                                                                 sequence_pooling_layer_30[0][0]  
                                                                 sequence_pooling_layer_31[0][0]  
__________________________________________________________________________________________________
flatten_3 (Flatten)             (None, 12)           0           k_max_pooling_1[0][0]            
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 1, 7)         0           no_mask_9[0][0]                  
                                                                 no_mask_9[1][0]                  
                                                                 no_mask_9[2][0]                  
                                                                 no_mask_9[3][0]                  
                                                                 no_mask_9[4][0]                  
                                                                 no_mask_9[5][0]                  
                                                                 no_mask_9[6][0]                  
__________________________________________________________________________________________________
dnn (DNN)                       (None, 32)           416         flatten_3[0][0]                  
__________________________________________________________________________________________________
linear_2 (Linear)               (None, 1, 1)         0           concatenate_5[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            32          dnn[0][0]                        
__________________________________________________________________________________________________
no_mask_10 (NoMask)             (None, 1, 1)         0           linear_2[0][0]                   
__________________________________________________________________________________________________
add_7 (Add)                     (None, 1, 1)         0           dense_1[0][0]                    
                                                                 no_mask_10[0][0]                 
__________________________________________________________________________________________________
prediction_layer_2 (PredictionL (None, 1)            1           add_7[0][0]                      
==================================================================================================
Total params: 592
Trainable params: 592
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DCN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DCN', 'sparse_feature_num': 3, 'dense_feature_num': 3} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DCN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_36 (Sequ (None, 1, 4)         0           weighted_sequence_layer_9[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_37 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_38 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_39 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_2 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_15 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer_36[0][0]  
                                                                 sequence_pooling_layer_37[0][0]  
                                                                 sequence_pooling_layer_38[0][0]  
                                                                 sequence_pooling_layer_39[0][0]  
__________________________________________________________________________________________________
no_mask_16 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
concatenate_9 (Concatenate)     (None, 1, 28)        0           no_mask_15[0][0]                 
                                                                 no_mask_15[1][0]                 
                                                                 no_mask_15[2][0]                 
                                                                 no_mask_15[3][0]                 
                                                                 no_mask_15[4][0]                 
                                                                 no_mask_15[5][0]                 
                                                                 no_mask_15[6][0]                 
__________________________________________________________________________________________________
concatenate_10 (Concatenate)    (None, 3)            0           no_mask_16[0][0]                 
                                                                 no_mask_16[1][0]                 
                                                                 no_mask_16[2][0]                 
__________________________________________________________________________________________________
weighted_sequence_layer_10 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_40 (Sequ (None, 1, 1)         0           weighted_sequence_layer_10[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_41 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_42 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_43 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_17 (NoMask)             multiple             0           flatten_4[0][0]                  
                                                                 flatten_5[0][0]                  
__________________________________________________________________________________________________
no_mask_12 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_40[0][0]  
                                                                 sequence_pooling_layer_41[0][0]  
                                                                 sequence_pooling_layer_42[0][0]  
                                                                 sequence_pooling_layer_43[0][0]  
__________________________________________________________________________________________________
no_mask_13 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
__________________________________________________________________________________________________
concatenate_11 (Concatenate)    (None, 31)           0           no_mask_17[0][0]                 
                                                                 no_mask_17[1][0]                 
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 1, 7)         0           no_mask_12[0][0]                 
                                                                 no_mask_12[1][0]                 
                                                                 no_mask_12[2][0]                 
                                                                 no_mask_12[3][0]                 
                                                                 no_mask_12[4][0]                 
                                                                 no_mask_12[5][0]                 
                                                                 no_mask_12[6][0]                 
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 3)            0           no_mask_13[0][0]                 
                                                                 no_mask_13[1][0]                 
                                                                 no_mask_13[2][0]                 
__________________________________________________________________________________________________
dnn_1 (DNN)                     (None, 8)            256         concatenate_11[0][0]             
__________________________________________________________________________________________________
linear_3 (Linear)               (None, 1)            3           concatenate_7[0][0]              
                                                                 concatenate_8[0][0]              
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            8           dnn_1[0][0]                      
__________________________________________________________________________________________________
no_mask_14 (NoMask)             (None, 1)            0           linear_3[0][0]                   
__________________________________________________________________________________________________
add_10 (Add)                    (None, 1)            0           dense_2[0][0]                    
                                                                 no_mask_14[0][0]                 
__________________________________________________________________________________________________
prediction_layer_3 (PredictionL (None, 1)            1           add_10[0][0]                     
==================================================================================================
Total params: 418
Trainable params: 418
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2631 - binary_crossentropy: 0.7222500/500 [==============================] - 1s 3ms/sample - loss: 0.2720 - binary_crossentropy: 0.7408 - val_loss: 0.2592 - val_binary_crossentropy: 0.7125

  #### metrics   #################################################### 
{'MSE': 0.2615003401508296}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_36 (Sequ (None, 1, 4)         0           weighted_sequence_layer_9[0][0]  
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_37 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_38 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_39 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_2 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_15 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sequence_pooling_layer_36[0][0]  
                                                                 sequence_pooling_layer_37[0][0]  
                                                                 sequence_pooling_layer_38[0][0]  
                                                                 sequence_pooling_layer_39[0][0]  
__________________________________________________________________________________________________
no_mask_16 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
concatenate_9 (Concatenate)     (None, 1, 28)        0           no_mask_15[0][0]                 
                                                                 no_mask_15[1][0]                 
                                                                 no_mask_15[2][0]                 
                                                                 no_mask_15[3][0]                 
                                                                 no_mask_15[4][0]                 
                                                                 no_mask_15[5][0]                 
                                                                 no_mask_15[6][0]                 
__________________________________________________________________________________________________
concatenate_10 (Concatenate)    (None, 3)            0           no_mask_16[0][0]                 
                                                                 no_mask_16[1][0]                 
                                                                 no_mask_16[2][0]                 
__________________________________________________________________________________________________
weighted_sequence_layer_10 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_40 (Sequ (None, 1, 1)         0           weighted_sequence_layer_10[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_41 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_42 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_43 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_17 (NoMask)             multiple             0           flatten_4[0][0]                  
                                                                 flatten_5[0][0]                  
__________________________________________________________________________________________________
no_mask_12 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_40[0][0]  
                                                                 sequence_pooling_layer_41[0][0]  
                                                                 sequence_pooling_layer_42[0][0]  
                                                                 sequence_pooling_layer_43[0][0]  
__________________________________________________________________________________________________
no_mask_13 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
__________________________________________________________________________________________________
concatenate_11 (Concatenate)    (None, 31)           0           no_mask_17[0][0]                 
                                                                 no_mask_17[1][0]                 
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 1, 7)         0           no_mask_12[0][0]                 
                                                                 no_mask_12[1][0]                 
                                                                 no_mask_12[2][0]                 
                                                                 no_mask_12[3][0]                 
                                                                 no_mask_12[4][0]                 
                                                                 no_mask_12[5][0]                 
                                                                 no_mask_12[6][0]                 
__________________________________________________________________________________________________
concatenate_8 (Concatenate)     (None, 3)            0           no_mask_13[0][0]                 
                                                                 no_mask_13[1][0]                 
                                                                 no_mask_13[2][0]                 
__________________________________________________________________________________________________
dnn_1 (DNN)                     (None, 8)            256         concatenate_11[0][0]             
__________________________________________________________________________________________________
linear_3 (Linear)               (None, 1)            3           concatenate_7[0][0]              
                                                                 concatenate_8[0][0]              
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            8           dnn_1[0][0]                      
__________________________________________________________________________________________________
no_mask_14 (NoMask)             (None, 1)            0           linear_3[0][0]                   
__________________________________________________________________________________________________
add_10 (Add)                    (None, 1)            0           dense_2[0][0]                    
                                                                 no_mask_14[0][0]                 
__________________________________________________________________________________________________
prediction_layer_3 (PredictionL (None, 1)            1           add_10[0][0]                     
==================================================================================================
Total params: 418
Trainable params: 418
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DeepFM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DeepFM', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DeepFM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_4"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_48 (Sequ (None, 1, 4)         0           weighted_sequence_layer_12[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_49 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_50 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_51 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_22 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_48[0][0]  
                                                                 sequence_pooling_layer_49[0][0]  
                                                                 sequence_pooling_layer_50[0][0]  
                                                                 sequence_pooling_layer_51[0][0]  
__________________________________________________________________________________________________
weighted_sequence_layer_13 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_52 (Sequ (None, 1, 1)         0           weighted_sequence_layer_13[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_53 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_54 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_55 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_6 (Flatten)             (None, 20)           0           concatenate_14[0][0]             
__________________________________________________________________________________________________
flatten_7 (Flatten)             (None, 1)            0           no_mask_23[0][0]                 
__________________________________________________________________________________________________
no_mask_18 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_52[0][0]  
                                                                 sequence_pooling_layer_53[0][0]  
                                                                 sequence_pooling_layer_54[0][0]  
                                                                 sequence_pooling_layer_55[0][0]  
__________________________________________________________________________________________________
no_mask_21 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_48[0][0]  
                                                                 sequence_pooling_layer_49[0][0]  
                                                                 sequence_pooling_layer_50[0][0]  
                                                                 sequence_pooling_layer_51[0][0]  
__________________________________________________________________________________________________
no_mask_24 (NoMask)             multiple             0           flatten_6[0][0]                  
                                                                 flatten_7[0][0]                  
__________________________________________________________________________________________________
concatenate_12 (Concatenate)    (None, 1, 5)         0           no_mask_18[0][0]                 
                                                                 no_mask_18[1][0]                 
                                                                 no_mask_18[2][0]                 
                                                                 no_mask_18[3][0]                 
                                                                 no_mask_18[4][0]                 
__________________________________________________________________________________________________
no_mask_19 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_13 (Concatenate)    (None, 5, 4)         0           no_mask_21[0][0]                 
                                                                 no_mask_21[1][0]                 
                                                                 no_mask_21[2][0]                 
                                                                 no_mask_21[3][0]                 
                                                                 no_mask_21[4][0]                 
__________________________________________________________________________________________________
concatenate_15 (Concatenate)    (None, 21)           0           no_mask_24[0][0]                 
                                                                 no_mask_24[1][0]                 
__________________________________________________________________________________________________
linear_4 (Linear)               (None, 1)            1           concatenate_12[0][0]             
                                                                 no_mask_19[0][0]                 
__________________________________________________________________________________________________
fm (FM)                         (None, 1)            0           concatenate_13[0][0]             
__________________________________________________________________________________________________
dnn_2 (DNN)                     (None, 2)            44          concatenate_15[0][0]             
__________________________________________________________________________________________________
no_mask_20 (NoMask)             (None, 1)            0           linear_4[0][0]                   
__________________________________________________________________________________________________
add_13 (Add)                    (None, 1)            0           fm[0][0]                         
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            2           dnn_2[0][0]                      
__________________________________________________________________________________________________
add_14 (Add)                    (None, 1)            0           no_mask_20[0][0]                 
                                                                 add_13[0][0]                     
                                                                 dense_3[0][0]                    
__________________________________________________________________________________________________
prediction_layer_4 (PredictionL (None, 1)            1           add_14[0][0]                     
==================================================================================================
Total params: 158
Trainable params: 158
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2521 - binary_crossentropy: 0.7024500/500 [==============================] - 2s 3ms/sample - loss: 0.2747 - binary_crossentropy: 0.7488 - val_loss: 0.2719 - val_binary_crossentropy: 0.7423

  #### metrics   #################################################### 
{'MSE': 0.27271193107414937}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_4"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_48 (Sequ (None, 1, 4)         0           weighted_sequence_layer_12[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_49 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_50 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_51 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_22 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_48[0][0]  
                                                                 sequence_pooling_layer_49[0][0]  
                                                                 sequence_pooling_layer_50[0][0]  
                                                                 sequence_pooling_layer_51[0][0]  
__________________________________________________________________________________________________
weighted_sequence_layer_13 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_52 (Sequ (None, 1, 1)         0           weighted_sequence_layer_13[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_53 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_54 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_55 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_6 (Flatten)             (None, 20)           0           concatenate_14[0][0]             
__________________________________________________________________________________________________
flatten_7 (Flatten)             (None, 1)            0           no_mask_23[0][0]                 
__________________________________________________________________________________________________
no_mask_18 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_52[0][0]  
                                                                 sequence_pooling_layer_53[0][0]  
                                                                 sequence_pooling_layer_54[0][0]  
                                                                 sequence_pooling_layer_55[0][0]  
__________________________________________________________________________________________________
no_mask_21 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_48[0][0]  
                                                                 sequence_pooling_layer_49[0][0]  
                                                                 sequence_pooling_layer_50[0][0]  
                                                                 sequence_pooling_layer_51[0][0]  
__________________________________________________________________________________________________
no_mask_24 (NoMask)             multiple             0           flatten_6[0][0]                  
                                                                 flatten_7[0][0]                  
__________________________________________________________________________________________________
concatenate_12 (Concatenate)    (None, 1, 5)         0           no_mask_18[0][0]                 
                                                                 no_mask_18[1][0]                 
                                                                 no_mask_18[2][0]                 
                                                                 no_mask_18[3][0]                 
                                                                 no_mask_18[4][0]                 
__________________________________________________________________________________________________
no_mask_19 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_13 (Concatenate)    (None, 5, 4)         0           no_mask_21[0][0]                 
                                                                 no_mask_21[1][0]                 
                                                                 no_mask_21[2][0]                 
                                                                 no_mask_21[3][0]                 
                                                                 no_mask_21[4][0]                 
__________________________________________________________________________________________________
concatenate_15 (Concatenate)    (None, 21)           0           no_mask_24[0][0]                 
                                                                 no_mask_24[1][0]                 
__________________________________________________________________________________________________
linear_4 (Linear)               (None, 1)            1           concatenate_12[0][0]             
                                                                 no_mask_19[0][0]                 
__________________________________________________________________________________________________
fm (FM)                         (None, 1)            0           concatenate_13[0][0]             
__________________________________________________________________________________________________
dnn_2 (DNN)                     (None, 2)            44          concatenate_15[0][0]             
__________________________________________________________________________________________________
no_mask_20 (NoMask)             (None, 1)            0           linear_4[0][0]                   
__________________________________________________________________________________________________
add_13 (Add)                    (None, 1)            0           fm[0][0]                         
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            2           dnn_2[0][0]                      
__________________________________________________________________________________________________
add_14 (Add)                    (None, 1)            0           no_mask_20[0][0]                 
                                                                 add_13[0][0]                     
                                                                 dense_3[0][0]                    
__________________________________________________________________________________________________
prediction_layer_4 (PredictionL (None, 1)            1           add_14[0][0]                     
==================================================================================================
Total params: 158
Trainable params: 158
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DIEN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DIEN'} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DIEN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/layers/sequence.py:724: GRUCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
Instructions for updating:
This class is equivalent as tf.keras.layers.GRUCell, and will be replaced by that in Tensorflow 2.0.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.AUTO_REUSE is deprecated. Please use tf.compat.v1.AUTO_REUSE instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/contrib/rnn.py:798: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/rnn_cell_impl.py:559: Layer.add_variable (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.add_weight` method instead.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/rnn_cell_impl.py:565: calling Constant.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/models/dien.py:282: The name tf.keras.backend.get_session is deprecated. Please use tf.compat.v1.keras.backend.get_session instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/deepctr/models/dien.py:282: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

Model: "model_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
hist_item (InputLayer)          [(None, 4)]          0                                            
__________________________________________________________________________________________________
hist_item_gender (InputLayer)   [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_hist_item (Embed multiple             32          item[0][0]                       
                                                                 hist_item[0][0]                  
                                                                 item[0][0]                       
__________________________________________________________________________________________________
sparse_seq_emb_hist_item_gender multiple             12          item_gender[0][0]                
                                                                 hist_item_gender[0][0]           
                                                                 item_gender[0][0]                
__________________________________________________________________________________________________
no_mask_25 (NoMask)             multiple             0           sparse_seq_emb_hist_item[1][0]   
                                                                 sparse_seq_emb_hist_item_gender[1
__________________________________________________________________________________________________
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
concatenate_16 (Concatenate)    (None, 4, 12)        0           no_mask_25[0][0]                 
                                                                 no_mask_25[1][0]                 
__________________________________________________________________________________________________
seq_length (InputLayer)         [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_user (Embedding)     (None, 1, 1)         3           user[0][0]                       
__________________________________________________________________________________________________
sparse_emb_gender (Embedding)   (None, 1, 1)         2           gender[0][0]                     
__________________________________________________________________________________________________
no_mask_27 (NoMask)             multiple             0           sparse_seq_emb_hist_item[0][0]   
                                                                 sparse_seq_emb_hist_item_gender[0
__________________________________________________________________________________________________
gru1 (DynamicGRU)               (None, 4, 12)        900         concatenate_16[0][0]             
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
no_mask_26 (NoMask)             multiple             0           sparse_emb_user[0][0]            
                                                                 sparse_emb_gender[0][0]          
                                                                 sparse_seq_emb_hist_item[2][0]   
                                                                 sparse_seq_emb_hist_item_gender[2
__________________________________________________________________________________________________
concatenate_18 (Concatenate)    (None, 1, 12)        0           no_mask_27[0][0]                 
                                                                 no_mask_27[1][0]                 
__________________________________________________________________________________________________
gru2 (DynamicGRU)               (None, 4, 12)        900         gru1[0][0]                       
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
concatenate_17 (Concatenate)    (None, 1, 14)        0           no_mask_26[0][0]                 
                                                                 no_mask_26[1][0]                 
                                                                 no_mask_26[2][0]                 
                                                                 no_mask_26[3][0]                 
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 12)        4433        concatenate_18[0][0]             
                                                                 gru2[0][0]                       
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
concatenate_19 (Concatenate)    (None, 1, 26)        0           concatenate_17[0][0]             
                                                                 attention_sequence_pooling_layer[
__________________________________________________________________________________________________
flatten_8 (Flatten)             (None, 26)           0           concatenate_19[0][0]             
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_28 (NoMask)             (None, 26)           0           flatten_8[0][0]                  
__________________________________________________________________________________________________
no_mask_29 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_9 (Flatten)             (None, 26)           0           no_mask_28[0][0]                 
__________________________________________________________________________________________________
flatten_10 (Flatten)            (None, 1)            0           no_mask_29[0][0]                 
__________________________________________________________________________________________________
no_mask_30 (NoMask)             multiple             0           flatten_9[0][0]                  
                                                                 flatten_10[0][0]                 
__________________________________________________________________________________________________
concatenate_20 (Concatenate)    (None, 27)           0           no_mask_30[0][0]                 
                                                                 no_mask_30[1][0]                 
__________________________________________________________________________________________________
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-22 12:55:34.458606: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:55:34.460672: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:55:34.466237: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 12:55:34.476185: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 12:55:34.478121: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 12:55:34.479642: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:55:34.481071: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            4           dnn_4[0][0]                      
__________________________________________________________________________________________________
prediction_layer_5 (PredictionL (None, 1)            1           dense_4[0][0]                    
==================================================================================================
Total params: 6,439
Trainable params: 6,279
Non-trainable params: 160
__________________________________________________________________________________________________
Train on 1 samples, validate on 2 samples
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2533 - val_binary_crossentropy: 0.6997
2020-05-22 12:55:35.757593: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:55:35.759300: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:55:35.763525: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 12:55:35.772117: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 12:55:35.773629: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 12:55:35.775010: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:55:35.776189: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2541252546032027}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
hist_item (InputLayer)          [(None, 4)]          0                                            
__________________________________________________________________________________________________
hist_item_gender (InputLayer)   [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_hist_item (Embed multiple             32          item[0][0]                       
                                                                 hist_item[0][0]                  
                                                                 item[0][0]                       
__________________________________________________________________________________________________
sparse_seq_emb_hist_item_gender multiple             12          item_gender[0][0]                
                                                                 hist_item_gender[0][0]           
                                                                 item_gender[0][0]                
__________________________________________________________________________________________________
no_mask_25 (NoMask)             multiple             0           sparse_seq_emb_hist_item[1][0]   
                                                                 sparse_seq_emb_hist_item_gender[1
__________________________________________________________________________________________________
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
concatenate_16 (Concatenate)    (None, 4, 12)        0           no_mask_25[0][0]                 
                                                                 no_mask_25[1][0]                 
__________________________________________________________________________________________________
seq_length (InputLayer)         [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_user (Embedding)     (None, 1, 1)         3           user[0][0]                       
__________________________________________________________________________________________________
sparse_emb_gender (Embedding)   (None, 1, 1)         2           gender[0][0]                     
__________________________________________________________________________________________________
no_mask_27 (NoMask)             multiple             0           sparse_seq_emb_hist_item[0][0]   
                                                                 sparse_seq_emb_hist_item_gender[0
__________________________________________________________________________________________________
gru1 (DynamicGRU)               (None, 4, 12)        900         concatenate_16[0][0]             
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
no_mask_26 (NoMask)             multiple             0           sparse_emb_user[0][0]            
                                                                 sparse_emb_gender[0][0]          
                                                                 sparse_seq_emb_hist_item[2][0]   
                                                                 sparse_seq_emb_hist_item_gender[2
__________________________________________________________________________________________________
concatenate_18 (Concatenate)    (None, 1, 12)        0           no_mask_27[0][0]                 
                                                                 no_mask_27[1][0]                 
__________________________________________________________________________________________________
gru2 (DynamicGRU)               (None, 4, 12)        900         gru1[0][0]                       
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
concatenate_17 (Concatenate)    (None, 1, 14)        0           no_mask_26[0][0]                 
                                                                 no_mask_26[1][0]                 
                                                                 no_mask_26[2][0]                 
                                                                 no_mask_26[3][0]                 
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 12)        4433        concatenate_18[0][0]             
                                                                 gru2[0][0]                       
                                                                 seq_length[0][0]                 
__________________________________________________________________________________________________
concatenate_19 (Concatenate)    (None, 1, 26)        0           concatenate_17[0][0]             
                                                                 attention_sequence_pooling_layer[
__________________________________________________________________________________________________
flatten_8 (Flatten)             (None, 26)           0           concatenate_19[0][0]             
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_28 (NoMask)             (None, 26)           0           flatten_8[0][0]                  
__________________________________________________________________________________________________
no_mask_29 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_9 (Flatten)             (None, 26)           0           no_mask_28[0][0]                 
__________________________________________________________________________________________________
flatten_10 (Flatten)            (None, 1)            0           no_mask_29[0][0]                 
__________________________________________________________________________________________________
no_mask_30 (NoMask)             multiple             0           flatten_9[0][0]                  
                                                                 flatten_10[0][0]                 
__________________________________________________________________________________________________
concatenate_20 (Concatenate)    (None, 27)           0           no_mask_30[0][0]                 
                                                                 no_mask_30[1][0]                 
__________________________________________________________________________________________________
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 1)            4           dnn_4[0][0]                      
__________________________________________________________________________________________________
prediction_layer_5 (PredictionL (None, 1)            1           dense_4[0][0]                    
==================================================================================================
Total params: 6,439
Trainable params: 6,279
Non-trainable params: 160
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DIN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DIN'} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DIN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
2020-05-22 12:55:59.216310: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:55:59.217637: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:55:59.220949: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 12:55:59.226891: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 12:55:59.227904: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 12:55:59.228836: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:55:59.229709: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
Model: "model_6"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_user (Embedding)     (None, 1, 4)         12          user[0][0]                       
__________________________________________________________________________________________________
sparse_emb_gender (Embedding)   (None, 1, 4)         8           gender[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_hist_item (Embed multiple             32          item[0][0]                       
                                                                 hist_item[0][0]                  
                                                                 item[0][0]                       
__________________________________________________________________________________________________
sparse_seq_emb_hist_item_gender multiple             12          item_gender[0][0]                
                                                                 hist_item_gender[0][0]           
                                                                 item_gender[0][0]                
__________________________________________________________________________________________________
hist_item (InputLayer)          [(None, 4)]          0                                            
__________________________________________________________________________________________________
hist_item_gender (InputLayer)   [(None, 4)]          0                                            
__________________________________________________________________________________________________
no_mask_31 (NoMask)             multiple             0           sparse_emb_user[0][0]            
                                                                 sparse_emb_gender[0][0]          
                                                                 sparse_seq_emb_hist_item[2][0]   
                                                                 sparse_seq_emb_hist_item_gender[2
__________________________________________________________________________________________________
concatenate_22 (Concatenate)    (None, 1, 20)        0           no_mask_31[0][0]                 
                                                                 no_mask_31[1][0]                 
                                                                 no_mask_31[2][0]                 
                                                                 no_mask_31[3][0]                 
__________________________________________________________________________________________________
concatenate_23 (Concatenate)    (None, 1, 12)        0           sparse_seq_emb_hist_item[0][0]   
                                                                 sparse_seq_emb_hist_item_gender[0
__________________________________________________________________________________________________
concatenate_21 (Concatenate)    (None, 4, 12)        0           sparse_seq_emb_hist_item[1][0]   
                                                                 sparse_seq_emb_hist_item_gender[1
__________________________________________________________________________________________________
no_mask_32 (NoMask)             (None, 1, 20)        0           concatenate_22[0][0]             
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 12)        7561        concatenate_23[0][0]             
                                                                 concatenate_21[0][0]             
__________________________________________________________________________________________________
concatenate_24 (Concatenate)    (None, 1, 32)        0           no_mask_32[0][0]                 
                                                                 attention_sequence_pooling_layer_
__________________________________________________________________________________________________
flatten_11 (Flatten)            (None, 32)           0           concatenate_24[0][0]             
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_33 (NoMask)             (None, 32)           0           flatten_11[0][0]                 
__________________________________________________________________________________________________
no_mask_34 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_12 (Flatten)            (None, 32)           0           no_mask_33[0][0]                 
__________________________________________________________________________________________________
flatten_13 (Flatten)            (None, 1)            0           no_mask_34[0][0]                 
__________________________________________________________________________________________________
no_mask_35 (NoMask)             multiple             0           flatten_12[0][0]                 
                                                                 flatten_13[0][0]                 
__________________________________________________________________________________________________
concatenate_25 (Concatenate)    (None, 33)           0           no_mask_35[0][0]                 
                                                                 no_mask_35[1][0]                 
__________________________________________________________________________________________________
dnn_7 (DNN)                     (None, 4)            176         concatenate_25[0][0]             
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 1)            4           dnn_7[0][0]                      
__________________________________________________________________________________________________
prediction_layer_6 (PredictionL (None, 1)            1           dense_5[0][0]                    
==================================================================================================
Total params: 7,806
Trainable params: 7,566
Non-trainable params: 240
__________________________________________________________________________________________________
Train on 1 samples, validate on 2 samples
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2502 - val_binary_crossentropy: 0.6935
2020-05-22 12:56:00.760420: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:56:00.761664: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:56:00.764173: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 12:56:00.769086: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 12:56:00.769986: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 12:56:00.770797: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:56:00.771528: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25015204479465264}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_6"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_user (Embedding)     (None, 1, 4)         12          user[0][0]                       
__________________________________________________________________________________________________
sparse_emb_gender (Embedding)   (None, 1, 4)         8           gender[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_hist_item (Embed multiple             32          item[0][0]                       
                                                                 hist_item[0][0]                  
                                                                 item[0][0]                       
__________________________________________________________________________________________________
sparse_seq_emb_hist_item_gender multiple             12          item_gender[0][0]                
                                                                 hist_item_gender[0][0]           
                                                                 item_gender[0][0]                
__________________________________________________________________________________________________
hist_item (InputLayer)          [(None, 4)]          0                                            
__________________________________________________________________________________________________
hist_item_gender (InputLayer)   [(None, 4)]          0                                            
__________________________________________________________________________________________________
no_mask_31 (NoMask)             multiple             0           sparse_emb_user[0][0]            
                                                                 sparse_emb_gender[0][0]          
                                                                 sparse_seq_emb_hist_item[2][0]   
                                                                 sparse_seq_emb_hist_item_gender[2
__________________________________________________________________________________________________
concatenate_22 (Concatenate)    (None, 1, 20)        0           no_mask_31[0][0]                 
                                                                 no_mask_31[1][0]                 
                                                                 no_mask_31[2][0]                 
                                                                 no_mask_31[3][0]                 
__________________________________________________________________________________________________
concatenate_23 (Concatenate)    (None, 1, 12)        0           sparse_seq_emb_hist_item[0][0]   
                                                                 sparse_seq_emb_hist_item_gender[0
__________________________________________________________________________________________________
concatenate_21 (Concatenate)    (None, 4, 12)        0           sparse_seq_emb_hist_item[1][0]   
                                                                 sparse_seq_emb_hist_item_gender[1
__________________________________________________________________________________________________
no_mask_32 (NoMask)             (None, 1, 20)        0           concatenate_22[0][0]             
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 12)        7561        concatenate_23[0][0]             
                                                                 concatenate_21[0][0]             
__________________________________________________________________________________________________
concatenate_24 (Concatenate)    (None, 1, 32)        0           no_mask_32[0][0]                 
                                                                 attention_sequence_pooling_layer_
__________________________________________________________________________________________________
flatten_11 (Flatten)            (None, 32)           0           concatenate_24[0][0]             
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_33 (NoMask)             (None, 32)           0           flatten_11[0][0]                 
__________________________________________________________________________________________________
no_mask_34 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_12 (Flatten)            (None, 32)           0           no_mask_33[0][0]                 
__________________________________________________________________________________________________
flatten_13 (Flatten)            (None, 1)            0           no_mask_34[0][0]                 
__________________________________________________________________________________________________
no_mask_35 (NoMask)             multiple             0           flatten_12[0][0]                 
                                                                 flatten_13[0][0]                 
__________________________________________________________________________________________________
concatenate_25 (Concatenate)    (None, 33)           0           no_mask_35[0][0]                 
                                                                 no_mask_35[1][0]                 
__________________________________________________________________________________________________
dnn_7 (DNN)                     (None, 4)            176         concatenate_25[0][0]             
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 1)            4           dnn_7[0][0]                      
__________________________________________________________________________________________________
prediction_layer_6 (PredictionL (None, 1)            1           dense_5[0][0]                    
==================================================================================================
Total params: 7,806
Trainable params: 7,566
Non-trainable params: 240
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'DSIN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'DSIN'} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_DSIN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.string_to_hash_bucket_fast is deprecated. Please use tf.strings.to_hash_bucket_fast instead.

WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/autograph/converters/directives.py:119: The name tf.matrix_set_diag is deprecated. Please use tf.linalg.set_diag instead.

Model: "model_7"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
sess_0_item (InputLayer)        [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_0_item_gender (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_1_item (InputLayer)        [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_1_item_gender (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_4 (Hash)                   (None, 1)            0           item[0][0]                       
__________________________________________________________________________________________________
hash_5 (Hash)                   (None, 1)            0           item_gender[0][0]                
__________________________________________________________________________________________________
hash (Hash)                     (None, 1)            0           item[0][0]                       
__________________________________________________________________________________________________
hash_1 (Hash)                   (None, 1)            0           item_gender[0][0]                
__________________________________________________________________________________________________
hash_6 (Hash)                   (None, 4)            0           sess_0_item[0][0]                
__________________________________________________________________________________________________
hash_7 (Hash)                   (None, 4)            0           sess_0_item_gender[0][0]         
__________________________________________________________________________________________________
hash_8 (Hash)                   (None, 4)            0           sess_1_item[0][0]                
__________________________________________________________________________________________________
hash_9 (Hash)                   (None, 4)            0           sess_1_item_gender[0][0]         
__________________________________________________________________________________________________
sparse_emb_2-item (Embedding)   multiple             16          hash[0][0]                       
                                                                 hash_4[0][0]                     
                                                                 hash_6[0][0]                     
                                                                 hash_8[0][0]                     
__________________________________________________________________________________________________
sparse_emb_3-item_gender (Embed multiple             12          hash_1[0][0]                     
                                                                 hash_5[0][0]                     
                                                                 hash_7[0][0]                     
                                                                 hash_9[0][0]                     
__________________________________________________________________________________________________
concatenate_28 (Concatenate)    (None, 4, 8)         0           sparse_emb_2-item[2][0]          
                                                                 sparse_emb_3-item_gender[2][0]   
__________________________________________________________________________________________________
concatenate_29 (Concatenate)    (None, 4, 8)         0           sparse_emb_2-item[3][0]          
                                                                 sparse_emb_3-item_gender[3][0]   
__________________________________________________________________________________________________
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
transformer (Transformer)       (None, 1, 8)         704         concatenate_28[0][0]             
                                                                 concatenate_28[0][0]             
                                                                 concatenate_29[0][0]             
                                                                 concatenate_29[0][0]             
__________________________________________________________________________________________________
hash_2 (Hash)                   (None, 1)            0           user[0][0]                       
__________________________________________________________________________________________________
hash_3 (Hash)                   (None, 1)            0           gender[0][0]                     
__________________________________________________________________________________________________
no_mask_37 (NoMask)             (None, 1, 8)         0           transformer[0][0]                
                                                                 transformer[1][0]                
__________________________________________________________________________________________________
sparse_emb_0-user (Embedding)   (None, 1, 4)         12          hash_2[0][0]                     
__________________________________________________________________________________________________
sparse_emb_1-gender (Embedding) (None, 1, 4)         8           hash_3[0][0]                     
__________________________________________________________________________________________________
concatenate_30 (Concatenate)    (None, 2, 8)         0           no_mask_37[0][0]                 
                                                                 no_mask_37[1][0]                 
__________________________________________________________________________________________________
no_mask_36 (NoMask)             (None, 1, 4)         0           sparse_emb_0-user[0][0]          
                                                                 sparse_emb_1-gender[0][0]        
                                                                 sparse_emb_2-item[1][0]          
                                                                 sparse_emb_3-item_gender[1][0]   
__________________________________________________________________________________________________
concatenate_26 (Concatenate)    (None, 1, 8)         0           sparse_emb_2-item[0][0]          
                                                                 sparse_emb_3-item_gender[0][0]   
__________________________________________________________________________________________________
sess_length (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
bi_lstm (BiLSTM)                (None, 2, 8)         2176        concatenate_30[0][0]             
__________________________________________________________________________________________________
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-22 12:56:33.988815: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:56:33.993823: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:56:34.009342: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 12:56:34.035820: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 12:56:34.040538: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 12:56:34.044549: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:56:34.048656: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

                                                                 no_mask_36[1][0]                 
                                                                 no_mask_36[2][0]                 
                                                                 no_mask_36[3][0]                 
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 8)         3169        concatenate_26[0][0]             
                                                                 concatenate_30[0][0]             
                                                                 sess_length[0][0]                
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 8)         3169        concatenate_26[0][0]             
                                                                 bi_lstm[0][0]                    
                                                                 sess_length[0][0]                
__________________________________________________________________________________________________
flatten_14 (Flatten)            (None, 16)           0           concatenate_27[0][0]             
__________________________________________________________________________________________________
flatten_15 (Flatten)            (None, 8)            0           attention_sequence_pooling_layer_
__________________________________________________________________________________________________
flatten_16 (Flatten)            (None, 8)            0           attention_sequence_pooling_layer_
__________________________________________________________________________________________________
concatenate_31 (Concatenate)    (None, 32)           0           flatten_14[0][0]                 
                                                                 flatten_15[0][0]                 
                                                                 flatten_16[0][0]                 
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_38 (NoMask)             (None, 32)           0           concatenate_31[0][0]             
__________________________________________________________________________________________________
no_mask_39 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_17 (Flatten)            (None, 32)           0           no_mask_38[0][0]                 
__________________________________________________________________________________________________
flatten_18 (Flatten)            (None, 1)            0           no_mask_39[0][0]                 
__________________________________________________________________________________________________
no_mask_40 (NoMask)             multiple             0           flatten_17[0][0]                 
                                                                 flatten_18[0][0]                 
__________________________________________________________________________________________________
concatenate_32 (Concatenate)    (None, 33)           0           no_mask_40[0][0]                 
                                                                 no_mask_40[1][0]                 
__________________________________________________________________________________________________
dnn_11 (DNN)                    (None, 4)            176         concatenate_32[0][0]             
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 1)            4           dnn_11[0][0]                     
__________________________________________________________________________________________________
prediction_layer_7 (PredictionL (None, 1)            1           dense_6[0][0]                    
==================================================================================================
Total params: 9,447
Trainable params: 9,447
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 1 samples, validate on 2 samples
1/1 [==============================] - 5s 5s/sample - loss: 0.1625 - binary_crossentropy: 0.5161 - val_loss: 0.2618 - val_binary_crossentropy: 0.7173
2020-05-22 12:56:36.309423: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:56:36.314171: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:56:36.326445: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 12:56:36.351151: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 12:56:36.355666: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 12:56:36.359695: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 12:56:36.363657: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.22577839397287155}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_7"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
item (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_gender (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
sess_0_item (InputLayer)        [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_0_item_gender (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_1_item (InputLayer)        [(None, 4)]          0                                            
__________________________________________________________________________________________________
sess_1_item_gender (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_4 (Hash)                   (None, 1)            0           item[0][0]                       
__________________________________________________________________________________________________
hash_5 (Hash)                   (None, 1)            0           item_gender[0][0]                
__________________________________________________________________________________________________
hash (Hash)                     (None, 1)            0           item[0][0]                       
__________________________________________________________________________________________________
hash_1 (Hash)                   (None, 1)            0           item_gender[0][0]                
__________________________________________________________________________________________________
hash_6 (Hash)                   (None, 4)            0           sess_0_item[0][0]                
__________________________________________________________________________________________________
hash_7 (Hash)                   (None, 4)            0           sess_0_item_gender[0][0]         
__________________________________________________________________________________________________
hash_8 (Hash)                   (None, 4)            0           sess_1_item[0][0]                
__________________________________________________________________________________________________
hash_9 (Hash)                   (None, 4)            0           sess_1_item_gender[0][0]         
__________________________________________________________________________________________________
sparse_emb_2-item (Embedding)   multiple             16          hash[0][0]                       
                                                                 hash_4[0][0]                     
                                                                 hash_6[0][0]                     
                                                                 hash_8[0][0]                     
__________________________________________________________________________________________________
sparse_emb_3-item_gender (Embed multiple             12          hash_1[0][0]                     
                                                                 hash_5[0][0]                     
                                                                 hash_7[0][0]                     
                                                                 hash_9[0][0]                     
__________________________________________________________________________________________________
concatenate_28 (Concatenate)    (None, 4, 8)         0           sparse_emb_2-item[2][0]          
                                                                 sparse_emb_3-item_gender[2][0]   
__________________________________________________________________________________________________
concatenate_29 (Concatenate)    (None, 4, 8)         0           sparse_emb_2-item[3][0]          
                                                                 sparse_emb_3-item_gender[3][0]   
__________________________________________________________________________________________________
user (InputLayer)               [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender (InputLayer)             [(None, 1)]          0                                            
__________________________________________________________________________________________________
transformer (Transformer)       (None, 1, 8)         704         concatenate_28[0][0]             
                                                                 concatenate_28[0][0]             
                                                                 concatenate_29[0][0]             
                                                                 concatenate_29[0][0]             
__________________________________________________________________________________________________
hash_2 (Hash)                   (None, 1)            0           user[0][0]                       
__________________________________________________________________________________________________
hash_3 (Hash)                   (None, 1)            0           gender[0][0]                     
__________________________________________________________________________________________________
no_mask_37 (NoMask)             (None, 1, 8)         0           transformer[0][0]                
                                                                 transformer[1][0]                
__________________________________________________________________________________________________
sparse_emb_0-user (Embedding)   (None, 1, 4)         12          hash_2[0][0]                     
__________________________________________________________________________________________________
sparse_emb_1-gender (Embedding) (None, 1, 4)         8           hash_3[0][0]                     
__________________________________________________________________________________________________
concatenate_30 (Concatenate)    (None, 2, 8)         0           no_mask_37[0][0]                 
                                                                 no_mask_37[1][0]                 
__________________________________________________________________________________________________
no_mask_36 (NoMask)             (None, 1, 4)         0           sparse_emb_0-user[0][0]          
                                                                 sparse_emb_1-gender[0][0]        
                                                                 sparse_emb_2-item[1][0]          
                                                                 sparse_emb_3-item_gender[1][0]   
__________________________________________________________________________________________________
concatenate_26 (Concatenate)    (None, 1, 8)         0           sparse_emb_2-item[0][0]          
                                                                 sparse_emb_3-item_gender[0][0]   
__________________________________________________________________________________________________
sess_length (InputLayer)        [(None, 1)]          0                                            
__________________________________________________________________________________________________
bi_lstm (BiLSTM)                (None, 2, 8)         2176        concatenate_30[0][0]             
__________________________________________________________________________________________________
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 
                                                                 no_mask_36[1][0]                 
                                                                 no_mask_36[2][0]                 
                                                                 no_mask_36[3][0]                 
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 8)         3169        concatenate_26[0][0]             
                                                                 concatenate_30[0][0]             
                                                                 sess_length[0][0]                
__________________________________________________________________________________________________
attention_sequence_pooling_laye (None, 1, 8)         3169        concatenate_26[0][0]             
                                                                 bi_lstm[0][0]                    
                                                                 sess_length[0][0]                
__________________________________________________________________________________________________
flatten_14 (Flatten)            (None, 16)           0           concatenate_27[0][0]             
__________________________________________________________________________________________________
flatten_15 (Flatten)            (None, 8)            0           attention_sequence_pooling_layer_
__________________________________________________________________________________________________
flatten_16 (Flatten)            (None, 8)            0           attention_sequence_pooling_layer_
__________________________________________________________________________________________________
concatenate_31 (Concatenate)    (None, 32)           0           flatten_14[0][0]                 
                                                                 flatten_15[0][0]                 
                                                                 flatten_16[0][0]                 
__________________________________________________________________________________________________
score (InputLayer)              [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_38 (NoMask)             (None, 32)           0           concatenate_31[0][0]             
__________________________________________________________________________________________________
no_mask_39 (NoMask)             (None, 1)            0           score[0][0]                      
__________________________________________________________________________________________________
flatten_17 (Flatten)            (None, 32)           0           no_mask_38[0][0]                 
__________________________________________________________________________________________________
flatten_18 (Flatten)            (None, 1)            0           no_mask_39[0][0]                 
__________________________________________________________________________________________________
no_mask_40 (NoMask)             multiple             0           flatten_17[0][0]                 
                                                                 flatten_18[0][0]                 
__________________________________________________________________________________________________
concatenate_32 (Concatenate)    (None, 33)           0           no_mask_40[0][0]                 
                                                                 no_mask_40[1][0]                 
__________________________________________________________________________________________________
dnn_11 (DNN)                    (None, 4)            176         concatenate_32[0][0]             
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 1)            4           dnn_11[0][0]                     
__________________________________________________________________________________________________
prediction_layer_7 (PredictionL (None, 1)            1           dense_6[0][0]                    
==================================================================================================
Total params: 9,447
Trainable params: 9,447
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'FiBiNET', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'FiBiNET', 'sparse_feature_num': 2, 'dense_feature_num': 2} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_FiBiNET.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_8"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_15 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_60 (Sequ (None, 1, 4)         0           weighted_sequence_layer_15[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_61 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_62 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_63 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
senet_layer (SENETLayer)        [(None, 1, 4), (None 24          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_60[0][0]  
                                                                 sequence_pooling_layer_61[0][0]  
                                                                 sequence_pooling_layer_62[0][0]  
                                                                 sequence_pooling_layer_63[0][0]  
__________________________________________________________________________________________________
bilinear_interaction (BilinearI (None, 1, 60)        16          senet_layer[0][0]                
                                                                 senet_layer[0][1]                
                                                                 senet_layer[0][2]                
                                                                 senet_layer[0][3]                
                                                                 senet_layer[0][4]                
                                                                 senet_layer[0][5]                
__________________________________________________________________________________________________
bilinear_interaction_1 (Bilinea (None, 1, 60)        16          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_60[0][0]  
                                                                 sequence_pooling_layer_61[0][0]  
                                                                 sequence_pooling_layer_62[0][0]  
                                                                 sequence_pooling_layer_63[0][0]  
__________________________________________________________________________________________________
no_mask_47 (NoMask)             (None, 1, 60)        0           bilinear_interaction[0][0]       
                                                                 bilinear_interaction_1[0][0]     
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
concatenate_38 (Concatenate)    (None, 1, 120)       0           no_mask_47[0][0]                 
                                                                 no_mask_47[1][0]                 
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
flatten_19 (Flatten)            (None, 120)          0           concatenate_38[0][0]             
__________________________________________________________________________________________________
no_mask_49 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
weighted_sequence_layer_16 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_64 (Sequ (None, 1, 1)         0           weighted_sequence_layer_16[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_65 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_66 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_67 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_20 (Flatten)            (None, 120)          0           no_mask_48[0][0]                 
__________________________________________________________________________________________________
flatten_21 (Flatten)            (None, 2)            0           concatenate_39[0][0]             
__________________________________________________________________________________________________
no_mask_44 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_64[0][0]  
                                                                 sequence_pooling_layer_65[0][0]  
                                                                 sequence_pooling_layer_66[0][0]  
                                                                 sequence_pooling_layer_67[0][0]  
__________________________________________________________________________________________________
no_mask_45 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
no_mask_50 (NoMask)             multiple             0           flatten_20[0][0]                 
                                                                 flatten_21[0][0]                 
__________________________________________________________________________________________________
concatenate_36 (Concatenate)    (None, 1, 6)         0           no_mask_44[0][0]                 
                                                                 no_mask_44[1][0]                 
                                                                 no_mask_44[2][0]                 
                                                                 no_mask_44[3][0]                 
                                                                 no_mask_44[4][0]                 
                                                                 no_mask_44[5][0]                 
__________________________________________________________________________________________________
concatenate_37 (Concatenate)    (None, 2)            0           no_mask_45[0][0]                 
                                                                 no_mask_45[1][0]                 
__________________________________________________________________________________________________
concatenate_40 (Concatenate)    (None, 122)          0           no_mask_50[0][0]                 
                                                                 no_mask_50[1][0]                 
__________________________________________________________________________________________________
linear_5 (Linear)               (None, 1)            2           concatenate_36[0][0]             
                                                                 concatenate_37[0][0]             
__________________________________________________________________________________________________
dnn_14 (DNN)                    (None, 4)            492         concatenate_40[0][0]             
__________________________________________________________________________________________________
no_mask_46 (NoMask)             (None, 1)            0           linear_5[0][0]                   
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1)            4           dnn_14[0][0]                     
__________________________________________________________________________________________________
add_17 (Add)                    (None, 1)            0           no_mask_46[0][0]                 
                                                                 dense_7[0][0]                    
__________________________________________________________________________________________________
prediction_layer_8 (PredictionL (None, 1)            1           add_17[0][0]                     
==================================================================================================
Total params: 690
Trainable params: 690
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2944 - binary_crossentropy: 0.7919500/500 [==============================] - 4s 8ms/sample - loss: 0.2815 - binary_crossentropy: 0.7640 - val_loss: 0.2733 - val_binary_crossentropy: 0.7449

  #### metrics   #################################################### 
{'MSE': 0.27623707446411516}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_8"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_15 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_60 (Sequ (None, 1, 4)         0           weighted_sequence_layer_15[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_61 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_62 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_63 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
senet_layer (SENETLayer)        [(None, 1, 4), (None 24          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_60[0][0]  
                                                                 sequence_pooling_layer_61[0][0]  
                                                                 sequence_pooling_layer_62[0][0]  
                                                                 sequence_pooling_layer_63[0][0]  
__________________________________________________________________________________________________
bilinear_interaction (BilinearI (None, 1, 60)        16          senet_layer[0][0]                
                                                                 senet_layer[0][1]                
                                                                 senet_layer[0][2]                
                                                                 senet_layer[0][3]                
                                                                 senet_layer[0][4]                
                                                                 senet_layer[0][5]                
__________________________________________________________________________________________________
bilinear_interaction_1 (Bilinea (None, 1, 60)        16          sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_60[0][0]  
                                                                 sequence_pooling_layer_61[0][0]  
                                                                 sequence_pooling_layer_62[0][0]  
                                                                 sequence_pooling_layer_63[0][0]  
__________________________________________________________________________________________________
no_mask_47 (NoMask)             (None, 1, 60)        0           bilinear_interaction[0][0]       
                                                                 bilinear_interaction_1[0][0]     
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
concatenate_38 (Concatenate)    (None, 1, 120)       0           no_mask_47[0][0]                 
                                                                 no_mask_47[1][0]                 
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
flatten_19 (Flatten)            (None, 120)          0           concatenate_38[0][0]             
__________________________________________________________________________________________________
no_mask_49 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
weighted_sequence_layer_16 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_64 (Sequ (None, 1, 1)         0           weighted_sequence_layer_16[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_65 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_66 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_67 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_20 (Flatten)            (None, 120)          0           no_mask_48[0][0]                 
__________________________________________________________________________________________________
flatten_21 (Flatten)            (None, 2)            0           concatenate_39[0][0]             
__________________________________________________________________________________________________
no_mask_44 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_64[0][0]  
                                                                 sequence_pooling_layer_65[0][0]  
                                                                 sequence_pooling_layer_66[0][0]  
                                                                 sequence_pooling_layer_67[0][0]  
__________________________________________________________________________________________________
no_mask_45 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
no_mask_50 (NoMask)             multiple             0           flatten_20[0][0]                 
                                                                 flatten_21[0][0]                 
__________________________________________________________________________________________________
concatenate_36 (Concatenate)    (None, 1, 6)         0           no_mask_44[0][0]                 
                                                                 no_mask_44[1][0]                 
                                                                 no_mask_44[2][0]                 
                                                                 no_mask_44[3][0]                 
                                                                 no_mask_44[4][0]                 
                                                                 no_mask_44[5][0]                 
__________________________________________________________________________________________________
concatenate_37 (Concatenate)    (None, 2)            0           no_mask_45[0][0]                 
                                                                 no_mask_45[1][0]                 
__________________________________________________________________________________________________
concatenate_40 (Concatenate)    (None, 122)          0           no_mask_50[0][0]                 
                                                                 no_mask_50[1][0]                 
__________________________________________________________________________________________________
linear_5 (Linear)               (None, 1)            2           concatenate_36[0][0]             
                                                                 concatenate_37[0][0]             
__________________________________________________________________________________________________
dnn_14 (DNN)                    (None, 4)            492         concatenate_40[0][0]             
__________________________________________________________________________________________________
no_mask_46 (NoMask)             (None, 1)            0           linear_5[0][0]                   
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1)            4           dnn_14[0][0]                     
__________________________________________________________________________________________________
add_17 (Add)                    (None, 1)            0           no_mask_46[0][0]                 
                                                                 dense_7[0][0]                    
__________________________________________________________________________________________________
prediction_layer_8 (PredictionL (None, 1)            1           add_17[0][0]                     
==================================================================================================
Total params: 690
Trainable params: 690
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'FLEN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'FLEN', 'embedding_size': 2, 'sparse_feature_num': 6, 'dense_feature_num': 6, 'use_group': True} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_FLEN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_9"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 2)         4           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_3 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_4 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_5 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_18 (Wei (None, 3, 2)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         14          sequence_max[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_2 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_3 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_4 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_5 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         14          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         14          sparse_feature_5[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_72 (Sequ (None, 1, 2)         0           weighted_sequence_layer_18[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_73 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_74 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_75 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_61 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_3[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_4[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sparse_emb_sparse_feature_5[0][0]
                                                                 sequence_pooling_layer_72[0][0]  
                                                                 sequence_pooling_layer_73[0][0]  
                                                                 sequence_pooling_layer_74[0][0]  
                                                                 sequence_pooling_layer_75[0][0]  
__________________________________________________________________________________________________
no_mask_62 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
                                                                 dense_feature_3[0][0]            
                                                                 dense_feature_4[0][0]            
                                                                 dense_feature_5[0][0]            
__________________________________________________________________________________________________
concatenate_50 (Concatenate)    (None, 1, 20)        0           no_mask_61[0][0]                 
                                                                 no_mask_61[1][0]                 
                                                                 no_mask_61[2][0]                 
                                                                 no_mask_61[3][0]                 
                                                                 no_mask_61[4][0]                 
                                                                 no_mask_61[5][0]                 
                                                                 no_mask_61[6][0]                 
                                                                 no_mask_61[7][0]                 
                                                                 no_mask_61[8][0]                 
                                                                 no_mask_61[9][0]                 
__________________________________________________________________________________________________
concatenate_51 (Concatenate)    (None, 6)            0           no_mask_62[0][0]                 
                                                                 no_mask_62[1][0]                 
                                                                 no_mask_62[2][0]                 
                                                                 no_mask_62[3][0]                 
                                                                 no_mask_62[4][0]                 
                                                                 no_mask_62[5][0]                 
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
flatten_22 (Flatten)            (None, 20)           0           concatenate_50[0][0]             
__________________________________________________________________________________________________
flatten_23 (Flatten)            (None, 6)            0           concatenate_51[0][0]             
__________________________________________________________________________________________________
weighted_sequence_layer_19 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_57 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_3[0][0]
__________________________________________________________________________________________________
no_mask_58 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_4[0][0]
__________________________________________________________________________________________________
no_mask_59 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_2[0][0]
                                                                 sparse_emb_sparse_feature_5[0][0]
__________________________________________________________________________________________________
no_mask_60 (NoMask)             (None, 1, 2)         0           sequence_pooling_layer_72[0][0]  
                                                                 sequence_pooling_layer_73[0][0]  
                                                                 sequence_pooling_layer_74[0][0]  
                                                                 sequence_pooling_layer_75[0][0]  
__________________________________________________________________________________________________
no_mask_63 (NoMask)             multiple             0           flatten_22[0][0]                 
                                                                 flatten_23[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_5[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_76 (Sequ (None, 1, 1)         0           weighted_sequence_layer_19[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_77 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_78 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_79 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_46 (Concatenate)    (None, 2, 2)         0           no_mask_57[0][0]                 
                                                                 no_mask_57[1][0]                 
__________________________________________________________________________________________________
concatenate_47 (Concatenate)    (None, 2, 2)         0           no_mask_58[0][0]                 
                                                                 no_mask_58[1][0]                 
__________________________________________________________________________________________________
concatenate_48 (Concatenate)    (None, 2, 2)         0           no_mask_59[0][0]                 
                                                                 no_mask_59[1][0]                 
__________________________________________________________________________________________________
concatenate_49 (Concatenate)    (None, 4, 2)         0           no_mask_60[0][0]                 
                                                                 no_mask_60[1][0]                 
                                                                 no_mask_60[2][0]                 
                                                                 no_mask_60[3][0]                 
__________________________________________________________________________________________________
concatenate_52 (Concatenate)    (None, 26)           0           no_mask_63[0][0]                 
                                                                 no_mask_63[1][0]                 
__________________________________________________________________________________________________
no_mask_54 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_76[0][0]  
                                                                 sequence_pooling_layer_77[0][0]  
                                                                 sequence_pooling_layer_78[0][0]  
                                                                 sequence_pooling_layer_79[0][0]  
__________________________________________________________________________________________________
no_mask_55 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
                                                                 dense_feature_3[0][0]            
                                                                 dense_feature_4[0][0]            
                                                                 dense_feature_5[0][0]            
__________________________________________________________________________________________________
field_wise_bi_interaction (Fiel (None, 2)            14          concatenate_46[0][0]             
                                                                 concatenate_47[0][0]             
                                                                 concatenate_48[0][0]             
                                                                 concatenate_49[0][0]             
__________________________________________________________________________________________________
dnn_15 (DNN)                    (None, 3)            81          concatenate_52[0][0]             
__________________________________________________________________________________________________
concatenate_44 (Concatenate)    (None, 1, 10)        0           no_mask_54[0][0]                 
                                                                 no_mask_54[1][0]                 
                                                                 no_mask_54[2][0]                 
                                                                 no_mask_54[3][0]                 
                                                                 no_mask_54[4][0]                 
                                                                 no_mask_54[5][0]                 
                                                                 no_mask_54[6][0]                 
                                                                 no_mask_54[7][0]                 
                                                                 no_mask_54[8][0]                 
                                                                 no_mask_54[9][0]                 
__________________________________________________________________________________________________
concatenate_45 (Concatenate)    (None, 6)            0           no_mask_55[0][0]                 
                                                                 no_mask_55[1][0]                 
                                                                 no_mask_55[2][0]                 
                                                                 no_mask_55[3][0]                 
                                                                 no_mask_55[4][0]                 
                                                                 no_mask_55[5][0]                 
__________________________________________________________________________________________________
no_mask_64 (NoMask)             multiple             0           field_wise_bi_interaction[0][0]  
                                                                 dnn_15[0][0]                     
__________________________________________________________________________________________________
linear_6 (Linear)               (None, 1)            6           concatenate_44[0][0]             
                                                                 concatenate_45[0][0]             
__________________________________________________________________________________________________
concatenate_53 (Concatenate)    (None, 5)            0           no_mask_64[0][0]                 
                                                                 no_mask_64[1][0]                 
__________________________________________________________________________________________________
no_mask_56 (NoMask)             (None, 1)            0           linear_6[0][0]                   
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 1)            5           concatenate_53[0][0]             
__________________________________________________________________________________________________
add_20 (Add)                    (None, 1)            0           no_mask_56[0][0]                 
                                                                 dense_8[0][0]                    
__________________________________________________________________________________________________
prediction_layer_9 (PredictionL (None, 1)            1           add_20[0][0]                     
==================================================================================================
Total params: 269
Trainable params: 269
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3204 - binary_crossentropy: 0.8524500/500 [==============================] - 4s 9ms/sample - loss: 0.2888 - binary_crossentropy: 0.7834 - val_loss: 0.2667 - val_binary_crossentropy: 0.7322

  #### metrics   #################################################### 
{'MSE': 0.2746747961523161}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_9"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 2)         4           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_3 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_4 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_5 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_18 (Wei (None, 3, 2)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         14          sequence_max[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_2 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_3 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_4 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_5 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         14          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         14          sparse_feature_5[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_72 (Sequ (None, 1, 2)         0           weighted_sequence_layer_18[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_73 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_74 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_75 (Sequ (None, 1, 2)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_61 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_3[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_4[0][0]
                                                                 sparse_emb_sparse_feature_2[0][0]
                                                                 sparse_emb_sparse_feature_5[0][0]
                                                                 sequence_pooling_layer_72[0][0]  
                                                                 sequence_pooling_layer_73[0][0]  
                                                                 sequence_pooling_layer_74[0][0]  
                                                                 sequence_pooling_layer_75[0][0]  
__________________________________________________________________________________________________
no_mask_62 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
                                                                 dense_feature_3[0][0]            
                                                                 dense_feature_4[0][0]            
                                                                 dense_feature_5[0][0]            
__________________________________________________________________________________________________
concatenate_50 (Concatenate)    (None, 1, 20)        0           no_mask_61[0][0]                 
                                                                 no_mask_61[1][0]                 
                                                                 no_mask_61[2][0]                 
                                                                 no_mask_61[3][0]                 
                                                                 no_mask_61[4][0]                 
                                                                 no_mask_61[5][0]                 
                                                                 no_mask_61[6][0]                 
                                                                 no_mask_61[7][0]                 
                                                                 no_mask_61[8][0]                 
                                                                 no_mask_61[9][0]                 
__________________________________________________________________________________________________
concatenate_51 (Concatenate)    (None, 6)            0           no_mask_62[0][0]                 
                                                                 no_mask_62[1][0]                 
                                                                 no_mask_62[2][0]                 
                                                                 no_mask_62[3][0]                 
                                                                 no_mask_62[4][0]                 
                                                                 no_mask_62[5][0]                 
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
flatten_22 (Flatten)            (None, 20)           0           concatenate_50[0][0]             
__________________________________________________________________________________________________
flatten_23 (Flatten)            (None, 6)            0           concatenate_51[0][0]             
__________________________________________________________________________________________________
weighted_sequence_layer_19 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_57 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_3[0][0]
__________________________________________________________________________________________________
no_mask_58 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_1[0][0]
                                                                 sparse_emb_sparse_feature_4[0][0]
__________________________________________________________________________________________________
no_mask_59 (NoMask)             (None, 1, 2)         0           sparse_emb_sparse_feature_2[0][0]
                                                                 sparse_emb_sparse_feature_5[0][0]
__________________________________________________________________________________________________
no_mask_60 (NoMask)             (None, 1, 2)         0           sequence_pooling_layer_72[0][0]  
                                                                 sequence_pooling_layer_73[0][0]  
                                                                 sequence_pooling_layer_74[0][0]  
                                                                 sequence_pooling_layer_75[0][0]  
__________________________________________________________________________________________________
no_mask_63 (NoMask)             multiple             0           flatten_22[0][0]                 
                                                                 flatten_23[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_5[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_76 (Sequ (None, 1, 1)         0           weighted_sequence_layer_19[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_77 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_78 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_79 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_46 (Concatenate)    (None, 2, 2)         0           no_mask_57[0][0]                 
                                                                 no_mask_57[1][0]                 
__________________________________________________________________________________________________
concatenate_47 (Concatenate)    (None, 2, 2)         0           no_mask_58[0][0]                 
                                                                 no_mask_58[1][0]                 
__________________________________________________________________________________________________
concatenate_48 (Concatenate)    (None, 2, 2)         0           no_mask_59[0][0]                 
                                                                 no_mask_59[1][0]                 
__________________________________________________________________________________________________
concatenate_49 (Concatenate)    (None, 4, 2)         0           no_mask_60[0][0]                 
                                                                 no_mask_60[1][0]                 
                                                                 no_mask_60[2][0]                 
                                                                 no_mask_60[3][0]                 
__________________________________________________________________________________________________
concatenate_52 (Concatenate)    (None, 26)           0           no_mask_63[0][0]                 
                                                                 no_mask_63[1][0]                 
__________________________________________________________________________________________________
no_mask_54 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_76[0][0]  
                                                                 sequence_pooling_layer_77[0][0]  
                                                                 sequence_pooling_layer_78[0][0]  
                                                                 sequence_pooling_layer_79[0][0]  
__________________________________________________________________________________________________
no_mask_55 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
                                                                 dense_feature_2[0][0]            
                                                                 dense_feature_3[0][0]            
                                                                 dense_feature_4[0][0]            
                                                                 dense_feature_5[0][0]            
__________________________________________________________________________________________________
field_wise_bi_interaction (Fiel (None, 2)            14          concatenate_46[0][0]             
                                                                 concatenate_47[0][0]             
                                                                 concatenate_48[0][0]             
                                                                 concatenate_49[0][0]             
__________________________________________________________________________________________________
dnn_15 (DNN)                    (None, 3)            81          concatenate_52[0][0]             
__________________________________________________________________________________________________
concatenate_44 (Concatenate)    (None, 1, 10)        0           no_mask_54[0][0]                 
                                                                 no_mask_54[1][0]                 
                                                                 no_mask_54[2][0]                 
                                                                 no_mask_54[3][0]                 
                                                                 no_mask_54[4][0]                 
                                                                 no_mask_54[5][0]                 
                                                                 no_mask_54[6][0]                 
                                                                 no_mask_54[7][0]                 
                                                                 no_mask_54[8][0]                 
                                                                 no_mask_54[9][0]                 
__________________________________________________________________________________________________
concatenate_45 (Concatenate)    (None, 6)            0           no_mask_55[0][0]                 
                                                                 no_mask_55[1][0]                 
                                                                 no_mask_55[2][0]                 
                                                                 no_mask_55[3][0]                 
                                                                 no_mask_55[4][0]                 
                                                                 no_mask_55[5][0]                 
__________________________________________________________________________________________________
no_mask_64 (NoMask)             multiple             0           field_wise_bi_interaction[0][0]  
                                                                 dnn_15[0][0]                     
__________________________________________________________________________________________________
linear_6 (Linear)               (None, 1)            6           concatenate_44[0][0]             
                                                                 concatenate_45[0][0]             
__________________________________________________________________________________________________
concatenate_53 (Concatenate)    (None, 5)            0           no_mask_64[0][0]                 
                                                                 no_mask_64[1][0]                 
__________________________________________________________________________________________________
no_mask_56 (NoMask)             (None, 1)            0           linear_6[0][0]                   
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 1)            5           concatenate_53[0][0]             
__________________________________________________________________________________________________
add_20 (Add)                    (None, 1)            0           no_mask_56[0][0]                 
                                                                 dense_8[0][0]                    
__________________________________________________________________________________________________
prediction_layer_9 (PredictionL (None, 1)            1           add_20[0][0]                     
==================================================================================================
Total params: 269
Trainable params: 269
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'FNN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'FNN', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_FNN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_10"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_84 (Sequ (None, 1, 4)         0           weighted_sequence_layer_21[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_85 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_86 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_87 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_68 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_84[0][0]  
                                                                 sequence_pooling_layer_85[0][0]  
                                                                 sequence_pooling_layer_86[0][0]  
                                                                 sequence_pooling_layer_87[0][0]  
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
concatenate_55 (Concatenate)    (None, 1, 20)        0           no_mask_68[0][0]                 
                                                                 no_mask_68[1][0]                 
                                                                 no_mask_68[2][0]                 
                                                                 no_mask_68[3][0]                 
                                                                 no_mask_68[4][0]                 
__________________________________________________________________________________________________
no_mask_69 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
weighted_sequence_layer_22 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_88 (Sequ (None, 1, 1)         0           weighted_sequence_layer_22[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_89 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_90 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_91 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_70 (NoMask)             multiple             0           flatten_24[0][0]                 
                                                                 flatten_25[0][0]                 
__________________________________________________________________________________________________
no_mask_65 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_88[0][0]  
                                                                 sequence_pooling_layer_89[0][0]  
                                                                 sequence_pooling_layer_90[0][0]  
                                                                 sequence_pooling_layer_91[0][0]  
__________________________________________________________________________________________________
concatenate_56 (Concatenate)    (None, 21)           0           no_mask_70[0][0]                 
                                                                 no_mask_70[1][0]                 
__________________________________________________________________________________________________
concatenate_54 (Concatenate)    (None, 1, 5)         0           no_mask_65[0][0]                 
                                                                 no_mask_65[1][0]                 
                                                                 no_mask_65[2][0]                 
                                                                 no_mask_65[3][0]                 
                                                                 no_mask_65[4][0]                 
__________________________________________________________________________________________________
no_mask_66 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
dnn_16 (DNN)                    (None, 32)           1760        concatenate_56[0][0]             
__________________________________________________________________________________________________
linear_7 (Linear)               (None, 1)            1           concatenate_54[0][0]             
                                                                 no_mask_66[0][0]                 
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 1)            32          dnn_16[0][0]                     
__________________________________________________________________________________________________
no_mask_67 (NoMask)             (None, 1)            0           linear_7[0][0]                   
__________________________________________________________________________________________________
add_23 (Add)                    (None, 1)            0           dense_9[0][0]                    
                                                                 no_mask_67[0][0]                 
__________________________________________________________________________________________________
prediction_layer_10 (Prediction (None, 1)            1           add_23[0][0]                     
==================================================================================================
Total params: 1,919
Trainable params: 1,919
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2523 - binary_crossentropy: 0.6976500/500 [==============================] - 5s 9ms/sample - loss: 0.2526 - binary_crossentropy: 0.6985 - val_loss: 0.2489 - val_binary_crossentropy: 0.6909

  #### metrics   #################################################### 
{'MSE': 0.2489113086223016}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_10"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_84 (Sequ (None, 1, 4)         0           weighted_sequence_layer_21[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_85 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_86 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_87 (Sequ (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_68 (NoMask)             (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_84[0][0]  
                                                                 sequence_pooling_layer_85[0][0]  
                                                                 sequence_pooling_layer_86[0][0]  
                                                                 sequence_pooling_layer_87[0][0]  
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
concatenate_55 (Concatenate)    (None, 1, 20)        0           no_mask_68[0][0]                 
                                                                 no_mask_68[1][0]                 
                                                                 no_mask_68[2][0]                 
                                                                 no_mask_68[3][0]                 
                                                                 no_mask_68[4][0]                 
__________________________________________________________________________________________________
no_mask_69 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
weighted_sequence_layer_22 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_88 (Sequ (None, 1, 1)         0           weighted_sequence_layer_22[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_89 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_90 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_91 (Sequ (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_70 (NoMask)             multiple             0           flatten_24[0][0]                 
                                                                 flatten_25[0][0]                 
__________________________________________________________________________________________________
no_mask_65 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_88[0][0]  
                                                                 sequence_pooling_layer_89[0][0]  
                                                                 sequence_pooling_layer_90[0][0]  
                                                                 sequence_pooling_layer_91[0][0]  
__________________________________________________________________________________________________
concatenate_56 (Concatenate)    (None, 21)           0           no_mask_70[0][0]                 
                                                                 no_mask_70[1][0]                 
__________________________________________________________________________________________________
concatenate_54 (Concatenate)    (None, 1, 5)         0           no_mask_65[0][0]                 
                                                                 no_mask_65[1][0]                 
                                                                 no_mask_65[2][0]                 
                                                                 no_mask_65[3][0]                 
                                                                 no_mask_65[4][0]                 
__________________________________________________________________________________________________
no_mask_66 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
dnn_16 (DNN)                    (None, 32)           1760        concatenate_56[0][0]             
__________________________________________________________________________________________________
linear_7 (Linear)               (None, 1)            1           concatenate_54[0][0]             
                                                                 no_mask_66[0][0]                 
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 1)            32          dnn_16[0][0]                     
__________________________________________________________________________________________________
no_mask_67 (NoMask)             (None, 1)            0           linear_7[0][0]                   
__________________________________________________________________________________________________
add_23 (Add)                    (None, 1)            0           dense_9[0][0]                    
                                                                 no_mask_67[0][0]                 
__________________________________________________________________________________________________
prediction_layer_10 (Prediction (None, 1)            1           add_23[0][0]                     
==================================================================================================
Total params: 1,919
Trainable params: 1,919
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'MLR', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'MLR', 'sparse_feature_num': 0, 'dense_feature_num': 2, 'prefix': 'region'} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_MLR.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_11"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
regionweighted_seq (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
region_10sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
regionweighted_seq_seq_length ( [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionweight (InputLayer)       [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
regionsequence_sum (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 5)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
region_20sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_24 (Wei (None, 3, 1)         0           region_10sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
sequence_pooling_layer_96 (Sequ (None, 1, 1)         0           weighted_sequence_layer_24[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_97 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_98 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_99 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
regiondense_feature_0 (InputLay [(None, 1)]          0                                            
__________________________________________________________________________________________________
regiondense_feature_1 (InputLay [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_pooling_layer_104 (Seq (None, 1, 1)         0           weighted_sequence_layer_26[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_105 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_106 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_107 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_112 (Seq (None, 1, 1)         0           weighted_sequence_layer_28[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_113 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_114 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_115 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_120 (Seq (None, 1, 1)         0           weighted_sequence_layer_30[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_121 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_122 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_123 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_128 (Seq (None, 1, 1)         0           weighted_sequence_layer_32[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_129 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_130 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_131 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_136 (Seq (None, 1, 1)         0           weighted_sequence_layer_34[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_137 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_138 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_139 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_144 (Seq (None, 1, 1)         0           weighted_sequence_layer_36[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_145 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_146 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_147 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_152 (Seq (None, 1, 1)         0           weighted_sequence_layer_38[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_153 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_154 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_155 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
no_mask_71 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_96[0][0]  
                                                                 sequence_pooling_layer_97[0][0]  
                                                                 sequence_pooling_layer_98[0][0]  
                                                                 sequence_pooling_layer_99[0][0]  
__________________________________________________________________________________________________
no_mask_72 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_74 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_104[0][0] 
                                                                 sequence_pooling_layer_105[0][0] 
                                                                 sequence_pooling_layer_106[0][0] 
                                                                 sequence_pooling_layer_107[0][0] 
__________________________________________________________________________________________________
no_mask_75 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_77 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_112[0][0] 
                                                                 sequence_pooling_layer_113[0][0] 
                                                                 sequence_pooling_layer_114[0][0] 
                                                                 sequence_pooling_layer_115[0][0] 
__________________________________________________________________________________________________
no_mask_78 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_80 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_120[0][0] 
                                                                 sequence_pooling_layer_121[0][0] 
                                                                 sequence_pooling_layer_122[0][0] 
                                                                 sequence_pooling_layer_123[0][0] 
__________________________________________________________________________________________________
no_mask_81 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_84 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_128[0][0] 
                                                                 sequence_pooling_layer_129[0][0] 
                                                                 sequence_pooling_layer_130[0][0] 
                                                                 sequence_pooling_layer_131[0][0] 
__________________________________________________________________________________________________
no_mask_85 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_87 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_136[0][0] 
                                                                 sequence_pooling_layer_137[0][0] 
                                                                 sequence_pooling_layer_138[0][0] 
                                                                 sequence_pooling_layer_139[0][0] 
__________________________________________________________________________________________________
no_mask_88 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_90 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_144[0][0] 
                                                                 sequence_pooling_layer_145[0][0] 
                                                                 sequence_pooling_layer_146[0][0] 
                                                                 sequence_pooling_layer_147[0][0] 
__________________________________________________________________________________________________
no_mask_91 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_93 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_152[0][0] 
                                                                 sequence_pooling_layer_153[0][0] 
                                                                 sequence_pooling_layer_154[0][0] 
                                                                 sequence_pooling_layer_155[0][0] 
__________________________________________________________________________________________________
no_mask_94 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
concatenate_57 (Concatenate)    (None, 1, 4)         0           no_mask_71[0][0]                 
                                                                 no_mask_71[1][0]                 
                                                                 no_mask_71[2][0]                 
                                                                 no_mask_71[3][0]                 
__________________________________________________________________________________________________
concatenate_58 (Concatenate)    (None, 2)            0           no_mask_72[0][0]                 
                                                                 no_mask_72[1][0]                 
__________________________________________________________________________________________________
concatenate_59 (Concatenate)    (None, 1, 4)         0           no_mask_74[0][0]                 
                                                                 no_mask_74[1][0]                 
                                                                 no_mask_74[2][0]                 
                                                                 no_mask_74[3][0]                 
__________________________________________________________________________________________________
concatenate_60 (Concatenate)    (None, 2)            0           no_mask_75[0][0]                 
                                                                 no_mask_75[1][0]                 
__________________________________________________________________________________________________
concatenate_61 (Concatenate)    (None, 1, 4)         0           no_mask_77[0][0]                 
                                                                 no_mask_77[1][0]                 
                                                                 no_mask_77[2][0]                 
                                                                 no_mask_77[3][0]                 
__________________________________________________________________________________________________
concatenate_62 (Concatenate)    (None, 2)            0           no_mask_78[0][0]                 
                                                                 no_mask_78[1][0]                 
__________________________________________________________________________________________________
concatenate_63 (Concatenate)    (None, 1, 4)         0           no_mask_80[0][0]                 
                                                                 no_mask_80[1][0]                 
                                                                 no_mask_80[2][0]                 
                                                                 no_mask_80[3][0]                 
__________________________________________________________________________________________________
concatenate_64 (Concatenate)    (None, 2)            0           no_mask_81[0][0]                 
                                                                 no_mask_81[1][0]                 
__________________________________________________________________________________________________
concatenate_66 (Concatenate)    (None, 1, 4)         0           no_mask_84[0][0]                 
                                                                 no_mask_84[1][0]                 
                                                                 no_mask_84[2][0]                 
                                                                 no_mask_84[3][0]                 
__________________________________________________________________________________________________
concatenate_67 (Concatenate)    (None, 2)            0           no_mask_85[0][0]                 
                                                                 no_mask_85[1][0]                 
__________________________________________________________________________________________________
concatenate_68 (Concatenate)    (None, 1, 4)         0           no_mask_87[0][0]                 
                                                                 no_mask_87[1][0]                 
                                                                 no_mask_87[2][0]                 
                                                                 no_mask_87[3][0]                 
__________________________________________________________________________________________________
concatenate_69 (Concatenate)    (None, 2)            0           no_mask_88[0][0]                 
                                                                 no_mask_88[1][0]                 
__________________________________________________________________________________________________
concatenate_70 (Concatenate)    (None, 1, 4)         0           no_mask_90[0][0]                 
                                                                 no_mask_90[1][0]                 
                                                                 no_mask_90[2][0]                 
                                                                 no_mask_90[3][0]                 
__________________________________________________________________________________________________
concatenate_71 (Concatenate)    (None, 2)            0           no_mask_91[0][0]                 
                                                                 no_mask_91[1][0]                 
__________________________________________________________________________________________________
concatenate_72 (Concatenate)    (None, 1, 4)         0           no_mask_93[0][0]                 
                                                                 no_mask_93[1][0]                 
                                                                 no_mask_93[2][0]                 
                                                                 no_mask_93[3][0]                 
__________________________________________________________________________________________________
concatenate_73 (Concatenate)    (None, 2)            0           no_mask_94[0][0]                 
                                                                 no_mask_94[1][0]                 
__________________________________________________________________________________________________
linear_8 (Linear)               (None, 1)            2           concatenate_57[0][0]             
                                                                 concatenate_58[0][0]             
__________________________________________________________________________________________________
linear_9 (Linear)               (None, 1)            2           concatenate_59[0][0]             
                                                                 concatenate_60[0][0]             
__________________________________________________________________________________________________
linear_10 (Linear)              (None, 1)            2           concatenate_61[0][0]             
                                                                 concatenate_62[0][0]             
__________________________________________________________________________________________________
linear_11 (Linear)              (None, 1)            2           concatenate_63[0][0]             
                                                                 concatenate_64[0][0]             
__________________________________________________________________________________________________
linear_12 (Linear)              (None, 1)            2           concatenate_66[0][0]             
                                                                 concatenate_67[0][0]             
__________________________________________________________________________________________________
linear_13 (Linear)              (None, 1)            2           concatenate_68[0][0]             
                                                                 concatenate_69[0][0]             
__________________________________________________________________________________________________
linear_14 (Linear)              (None, 1)            2           concatenate_70[0][0]             
                                                                 concatenate_71[0][0]             
__________________________________________________________________________________________________
linear_15 (Linear)              (None, 1)            2           concatenate_72[0][0]             
                                                                 concatenate_73[0][0]             
__________________________________________________________________________________________________
no_mask_73 (NoMask)             (None, 1)            0           linear_8[0][0]                   
__________________________________________________________________________________________________
no_mask_76 (NoMask)             (None, 1)            0           linear_9[0][0]                   
__________________________________________________________________________________________________
no_mask_79 (NoMask)             (None, 1)            0           linear_10[0][0]                  
__________________________________________________________________________________________________
no_mask_82 (NoMask)             (None, 1)            0           linear_11[0][0]                  
__________________________________________________________________________________________________
no_mask_86 (NoMask)             (None, 1)            0           linear_12[0][0]                  
__________________________________________________________________________________________________
no_mask_89 (NoMask)             (None, 1)            0           linear_13[0][0]                  
__________________________________________________________________________________________________
no_mask_92 (NoMask)             (None, 1)            0           linear_14[0][0]                  
__________________________________________________________________________________________________
no_mask_95 (NoMask)             (None, 1)            0           linear_15[0][0]                  
__________________________________________________________________________________________________
no_mask_83 (NoMask)             (None, 1)            0           no_mask_73[0][0]                 
                                                                 no_mask_76[0][0]                 
                                                                 no_mask_79[0][0]                 
                                                                 no_mask_82[0][0]                 
__________________________________________________________________________________________________
prediction_layer_11 (Prediction (None, 1)            0           no_mask_86[0][0]                 
__________________________________________________________________________________________________
prediction_layer_12 (Prediction (None, 1)            0           no_mask_89[0][0]                 
__________________________________________________________________________________________________
prediction_layer_13 (Prediction (None, 1)            0           no_mask_92[0][0]                 
__________________________________________________________________________________________________
prediction_layer_14 (Prediction (None, 1)            0           no_mask_95[0][0]                 
__________________________________________________________________________________________________
concatenate_65 (Concatenate)    (None, 4)            0           no_mask_83[0][0]                 
                                                                 no_mask_83[1][0]                 
                                                                 no_mask_83[2][0]                 
                                                                 no_mask_83[3][0]                 
__________________________________________________________________________________________________
no_mask_96 (NoMask)             (None, 1)            0           prediction_layer_11[0][0]        
                                                                 prediction_layer_12[0][0]        
                                                                 prediction_layer_13[0][0]        
                                                                 prediction_layer_14[0][0]        
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 4)            0           concatenate_65[0][0]             
__________________________________________________________________________________________________
concatenate_74 (Concatenate)    (None, 4)            0           no_mask_96[0][0]                 
                                                                 no_mask_96[1][0]                 
                                                                 no_mask_96[2][0]                 
                                                                 no_mask_96[3][0]                 
__________________________________________________________________________________________________
dot (Dot)                       (None, 1)            0           activation_40[0][0]              
                                                                 concatenate_74[0][0]             
==================================================================================================
Total params: 160
Trainable params: 160
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2519 - binary_crossentropy: 0.6969500/500 [==============================] - 6s 11ms/sample - loss: 0.2517 - binary_crossentropy: 0.6964 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.2506678780840114}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_11"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
regionweighted_seq (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
region_10sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
regionweighted_seq_seq_length ( [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionweight (InputLayer)       [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
regionsequence_sum (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 5)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
region_20sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regionw (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionweighted_seq[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_24 (Wei (None, 3, 1)         0           region_10sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
sequence_pooling_layer_96 (Sequ (None, 1, 1)         0           weighted_sequence_layer_24[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_97 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_98 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_99 (Sequ (None, 1, 1)         0           region_10sparse_seq_emb_regionseq
__________________________________________________________________________________________________
regiondense_feature_0 (InputLay [(None, 1)]          0                                            
__________________________________________________________________________________________________
regiondense_feature_1 (InputLay [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_pooling_layer_104 (Seq (None, 1, 1)         0           weighted_sequence_layer_26[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_105 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_106 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_107 (Seq (None, 1, 1)         0           region_20sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_112 (Seq (None, 1, 1)         0           weighted_sequence_layer_28[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_113 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_114 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_115 (Seq (None, 1, 1)         0           region_30sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_120 (Seq (None, 1, 1)         0           weighted_sequence_layer_30[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_121 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_122 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_123 (Seq (None, 1, 1)         0           region_40sparse_seq_emb_regionseq
__________________________________________________________________________________________________
sequence_pooling_layer_128 (Seq (None, 1, 1)         0           weighted_sequence_layer_32[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_129 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_130 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_131 (Seq (None, 1, 1)         0           learner_10sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_136 (Seq (None, 1, 1)         0           weighted_sequence_layer_34[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_137 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_138 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_139 (Seq (None, 1, 1)         0           learner_20sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_144 (Seq (None, 1, 1)         0           weighted_sequence_layer_36[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_145 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_146 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_147 (Seq (None, 1, 1)         0           learner_30sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_152 (Seq (None, 1, 1)         0           weighted_sequence_layer_38[0][0] 
                                                                 regionweighted_seq_seq_length[0][
__________________________________________________________________________________________________
sequence_pooling_layer_153 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_154 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
sequence_pooling_layer_155 (Seq (None, 1, 1)         0           learner_40sparse_seq_emb_regionse
__________________________________________________________________________________________________
no_mask_71 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_96[0][0]  
                                                                 sequence_pooling_layer_97[0][0]  
                                                                 sequence_pooling_layer_98[0][0]  
                                                                 sequence_pooling_layer_99[0][0]  
__________________________________________________________________________________________________
no_mask_72 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_74 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_104[0][0] 
                                                                 sequence_pooling_layer_105[0][0] 
                                                                 sequence_pooling_layer_106[0][0] 
                                                                 sequence_pooling_layer_107[0][0] 
__________________________________________________________________________________________________
no_mask_75 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_77 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_112[0][0] 
                                                                 sequence_pooling_layer_113[0][0] 
                                                                 sequence_pooling_layer_114[0][0] 
                                                                 sequence_pooling_layer_115[0][0] 
__________________________________________________________________________________________________
no_mask_78 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_80 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_120[0][0] 
                                                                 sequence_pooling_layer_121[0][0] 
                                                                 sequence_pooling_layer_122[0][0] 
                                                                 sequence_pooling_layer_123[0][0] 
__________________________________________________________________________________________________
no_mask_81 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_84 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_128[0][0] 
                                                                 sequence_pooling_layer_129[0][0] 
                                                                 sequence_pooling_layer_130[0][0] 
                                                                 sequence_pooling_layer_131[0][0] 
__________________________________________________________________________________________________
no_mask_85 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_87 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_136[0][0] 
                                                                 sequence_pooling_layer_137[0][0] 
                                                                 sequence_pooling_layer_138[0][0] 
                                                                 sequence_pooling_layer_139[0][0] 
__________________________________________________________________________________________________
no_mask_88 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_90 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_144[0][0] 
                                                                 sequence_pooling_layer_145[0][0] 
                                                                 sequence_pooling_layer_146[0][0] 
                                                                 sequence_pooling_layer_147[0][0] 
__________________________________________________________________________________________________
no_mask_91 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
no_mask_93 (NoMask)             (None, 1, 1)         0           sequence_pooling_layer_152[0][0] 
                                                                 sequence_pooling_layer_153[0][0] 
                                                                 sequence_pooling_layer_154[0][0] 
                                                                 sequence_pooling_layer_155[0][0] 
__________________________________________________________________________________________________
no_mask_94 (NoMask)             (None, 1)            0           regiondense_feature_0[0][0]      
                                                                 regiondense_feature_1[0][0]      
__________________________________________________________________________________________________
concatenate_57 (Concatenate)    (None, 1, 4)         0           no_mask_71[0][0]                 
                                                                 no_mask_71[1][0]                 
                                                                 no_mask_71[2][0]                 
                                                                 no_mask_71[3][0]                 
__________________________________________________________________________________________________
concatenate_58 (Concatenate)    (None, 2)            0           no_mask_72[0][0]                 
                                                                 no_mask_72[1][0]                 
__________________________________________________________________________________________________
concatenate_59 (Concatenate)    (None, 1, 4)         0           no_mask_74[0][0]                 
                                                                 no_mask_74[1][0]                 
                                                                 no_mask_74[2][0]                 
                                                                 no_mask_74[3][0]                 
__________________________________________________________________________________________________
concatenate_60 (Concatenate)    (None, 2)            0           no_mask_75[0][0]                 
                                                                 no_mask_75[1][0]                 
__________________________________________________________________________________________________
concatenate_61 (Concatenate)    (None, 1, 4)         0           no_mask_77[0][0]                 
                                                                 no_mask_77[1][0]                 
                                                                 no_mask_77[2][0]                 
                                                                 no_mask_77[3][0]                 
__________________________________________________________________________________________________
concatenate_62 (Concatenate)    (None, 2)            0           no_mask_78[0][0]                 
                                                                 no_mask_78[1][0]                 
__________________________________________________________________________________________________
concatenate_63 (Concatenate)    (None, 1, 4)         0           no_mask_80[0][0]                 
                                                                 no_mask_80[1][0]                 
                                                                 no_mask_80[2][0]                 
                                                                 no_mask_80[3][0]                 
__________________________________________________________________________________________________
concatenate_64 (Concatenate)    (None, 2)            0           no_mask_81[0][0]                 
                                                                 no_mask_81[1][0]                 
__________________________________________________________________________________________________
concatenate_66 (Concatenate)    (None, 1, 4)         0           no_mask_84[0][0]                 
                                                                 no_mask_84[1][0]                 
                                                                 no_mask_84[2][0]                 
                                                                 no_mask_84[3][0]                 
__________________________________________________________________________________________________
concatenate_67 (Concatenate)    (None, 2)            0           no_mask_85[0][0]                 
                                                                 no_mask_85[1][0]                 
__________________________________________________________________________________________________
concatenate_68 (Concatenate)    (None, 1, 4)         0           no_mask_87[0][0]                 
                                                                 no_mask_87[1][0]                 
                                                                 no_mask_87[2][0]                 
                                                                 no_mask_87[3][0]                 
__________________________________________________________________________________________________
concatenate_69 (Concatenate)    (None, 2)            0           no_mask_88[0][0]                 
                                                                 no_mask_88[1][0]                 
__________________________________________________________________________________________________
concatenate_70 (Concatenate)    (None, 1, 4)         0           no_mask_90[0][0]                 
                                                                 no_mask_90[1][0]                 
                                                                 no_mask_90[2][0]                 
                                                                 no_mask_90[3][0]                 
__________________________________________________________________________________________________
concatenate_71 (Concatenate)    (None, 2)            0           no_mask_91[0][0]                 
                                                                 no_mask_91[1][0]                 
__________________________________________________________________________________________________
concatenate_72 (Concatenate)    (None, 1, 4)         0           no_mask_93[0][0]                 
                                                                 no_mask_93[1][0]                 
                                                                 no_mask_93[2][0]                 
                                                                 no_mask_93[3][0]                 
__________________________________________________________________________________________________
concatenate_73 (Concatenate)    (None, 2)            0           no_mask_94[0][0]                 
                                                                 no_mask_94[1][0]                 
__________________________________________________________________________________________________
linear_8 (Linear)               (None, 1)            2           concatenate_57[0][0]             
                                                                 concatenate_58[0][0]             
__________________________________________________________________________________________________
linear_9 (Linear)               (None, 1)            2           concatenate_59[0][0]             
                                                                 concatenate_60[0][0]             
__________________________________________________________________________________________________
linear_10 (Linear)              (None, 1)            2           concatenate_61[0][0]             
                                                                 concatenate_62[0][0]             
__________________________________________________________________________________________________
linear_11 (Linear)              (None, 1)            2           concatenate_63[0][0]             
                                                                 concatenate_64[0][0]             
__________________________________________________________________________________________________
linear_12 (Linear)              (None, 1)            2           concatenate_66[0][0]             
                                                                 concatenate_67[0][0]             
__________________________________________________________________________________________________
linear_13 (Linear)              (None, 1)            2           concatenate_68[0][0]             
                                                                 concatenate_69[0][0]             
__________________________________________________________________________________________________
linear_14 (Linear)              (None, 1)            2           concatenate_70[0][0]             
                                                                 concatenate_71[0][0]             
__________________________________________________________________________________________________
linear_15 (Linear)              (None, 1)            2           concatenate_72[0][0]             
                                                                 concatenate_73[0][0]             
__________________________________________________________________________________________________
no_mask_73 (NoMask)             (None, 1)            0           linear_8[0][0]                   
__________________________________________________________________________________________________
no_mask_76 (NoMask)             (None, 1)            0           linear_9[0][0]                   
__________________________________________________________________________________________________
no_mask_79 (NoMask)             (None, 1)            0           linear_10[0][0]                  
__________________________________________________________________________________________________
no_mask_82 (NoMask)             (None, 1)            0           linear_11[0][0]                  
__________________________________________________________________________________________________
no_mask_86 (NoMask)             (None, 1)            0           linear_12[0][0]                  
__________________________________________________________________________________________________
no_mask_89 (NoMask)             (None, 1)            0           linear_13[0][0]                  
__________________________________________________________________________________________________
no_mask_92 (NoMask)             (None, 1)            0           linear_14[0][0]                  
__________________________________________________________________________________________________
no_mask_95 (NoMask)             (None, 1)            0           linear_15[0][0]                  
__________________________________________________________________________________________________
no_mask_83 (NoMask)             (None, 1)            0           no_mask_73[0][0]                 
                                                                 no_mask_76[0][0]                 
                                                                 no_mask_79[0][0]                 
                                                                 no_mask_82[0][0]                 
__________________________________________________________________________________________________
prediction_layer_11 (Prediction (None, 1)            0           no_mask_86[0][0]                 
__________________________________________________________________________________________________
prediction_layer_12 (Prediction (None, 1)            0           no_mask_89[0][0]                 
__________________________________________________________________________________________________
prediction_layer_13 (Prediction (None, 1)            0           no_mask_92[0][0]                 
__________________________________________________________________________________________________
prediction_layer_14 (Prediction (None, 1)            0           no_mask_95[0][0]                 
__________________________________________________________________________________________________
concatenate_65 (Concatenate)    (None, 4)            0           no_mask_83[0][0]                 
                                                                 no_mask_83[1][0]                 
                                                                 no_mask_83[2][0]                 
                                                                 no_mask_83[3][0]                 
__________________________________________________________________________________________________
no_mask_96 (NoMask)             (None, 1)            0           prediction_layer_11[0][0]        
                                                                 prediction_layer_12[0][0]        
                                                                 prediction_layer_13[0][0]        
                                                                 prediction_layer_14[0][0]        
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 4)            0           concatenate_65[0][0]             
__________________________________________________________________________________________________
concatenate_74 (Concatenate)    (None, 4)            0           no_mask_96[0][0]                 
                                                                 no_mask_96[1][0]                 
                                                                 no_mask_96[2][0]                 
                                                                 no_mask_96[3][0]                 
__________________________________________________________________________________________________
dot (Dot)                       (None, 1)            0           activation_40[0][0]              
                                                                 concatenate_74[0][0]             
==================================================================================================
Total params: 160
Trainable params: 160
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'NFM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'NFM', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_NFM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_12"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_160 (Seq (None, 1, 4)         0           weighted_sequence_layer_40[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_161 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_162 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_163 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_100 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_160[0][0] 
                                                                 sequence_pooling_layer_161[0][0] 
                                                                 sequence_pooling_layer_162[0][0] 
                                                                 sequence_pooling_layer_163[0][0] 
__________________________________________________________________________________________________
concatenate_76 (Concatenate)    (None, 5, 4)         0           no_mask_100[0][0]                
                                                                 no_mask_100[1][0]                
                                                                 no_mask_100[2][0]                
                                                                 no_mask_100[3][0]                
                                                                 no_mask_100[4][0]                
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
bi_interaction_pooling (BiInter (None, 1, 4)         0           concatenate_76[0][0]             
__________________________________________________________________________________________________
weighted_sequence_layer_41 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_164 (Seq (None, 1, 1)         0           weighted_sequence_layer_41[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_165 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_166 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_167 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_26 (Flatten)            (None, 4)            0           no_mask_101[0][0]                
__________________________________________________________________________________________________
flatten_27 (Flatten)            (None, 1)            0           no_mask_102[0][0]                
__________________________________________________________________________________________________
no_mask_97 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_164[0][0] 
                                                                 sequence_pooling_layer_165[0][0] 
                                                                 sequence_pooling_layer_166[0][0] 
                                                                 sequence_pooling_layer_167[0][0] 
__________________________________________________________________________________________________
no_mask_103 (NoMask)            multiple             0           flatten_26[0][0]                 
                                                                 flatten_27[0][0]                 
__________________________________________________________________________________________________
concatenate_75 (Concatenate)    (None, 1, 5)         0           no_mask_97[0][0]                 
                                                                 no_mask_97[1][0]                 
                                                                 no_mask_97[2][0]                 
                                                                 no_mask_97[3][0]                 
                                                                 no_mask_97[4][0]                 
__________________________________________________________________________________________________
no_mask_98 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_77 (Concatenate)    (None, 5)            0           no_mask_103[0][0]                
                                                                 no_mask_103[1][0]                
__________________________________________________________________________________________________
linear_16 (Linear)              (None, 1)            1           concatenate_75[0][0]             
                                                                 no_mask_98[0][0]                 
__________________________________________________________________________________________________
dnn_17 (DNN)                    (None, 32)           1248        concatenate_77[0][0]             
__________________________________________________________________________________________________
no_mask_99 (NoMask)             (None, 1)            0           linear_16[0][0]                  
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1)            32          dnn_17[0][0]                     
__________________________________________________________________________________________________
add_26 (Add)                    (None, 1)            0           no_mask_99[0][0]                 
                                                                 dense_10[0][0]                   
__________________________________________________________________________________________________
prediction_layer_15 (Prediction (None, 1)            1           add_26[0][0]                     
==================================================================================================
Total params: 1,367
Trainable params: 1,367
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2586 - binary_crossentropy: 0.7107500/500 [==============================] - 6s 11ms/sample - loss: 0.2492 - binary_crossentropy: 0.6915 - val_loss: 0.2581 - val_binary_crossentropy: 0.7091

  #### metrics   #################################################### 
{'MSE': 0.25480229309396546}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_12"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_160 (Seq (None, 1, 4)         0           weighted_sequence_layer_40[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_161 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_162 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_163 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_100 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_160[0][0] 
                                                                 sequence_pooling_layer_161[0][0] 
                                                                 sequence_pooling_layer_162[0][0] 
                                                                 sequence_pooling_layer_163[0][0] 
__________________________________________________________________________________________________
concatenate_76 (Concatenate)    (None, 5, 4)         0           no_mask_100[0][0]                
                                                                 no_mask_100[1][0]                
                                                                 no_mask_100[2][0]                
                                                                 no_mask_100[3][0]                
                                                                 no_mask_100[4][0]                
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
bi_interaction_pooling (BiInter (None, 1, 4)         0           concatenate_76[0][0]             
__________________________________________________________________________________________________
weighted_sequence_layer_41 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_164 (Seq (None, 1, 1)         0           weighted_sequence_layer_41[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_165 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_166 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_167 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_26 (Flatten)            (None, 4)            0           no_mask_101[0][0]                
__________________________________________________________________________________________________
flatten_27 (Flatten)            (None, 1)            0           no_mask_102[0][0]                
__________________________________________________________________________________________________
no_mask_97 (NoMask)             (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_164[0][0] 
                                                                 sequence_pooling_layer_165[0][0] 
                                                                 sequence_pooling_layer_166[0][0] 
                                                                 sequence_pooling_layer_167[0][0] 
__________________________________________________________________________________________________
no_mask_103 (NoMask)            multiple             0           flatten_26[0][0]                 
                                                                 flatten_27[0][0]                 
__________________________________________________________________________________________________
concatenate_75 (Concatenate)    (None, 1, 5)         0           no_mask_97[0][0]                 
                                                                 no_mask_97[1][0]                 
                                                                 no_mask_97[2][0]                 
                                                                 no_mask_97[3][0]                 
                                                                 no_mask_97[4][0]                 
__________________________________________________________________________________________________
no_mask_98 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_77 (Concatenate)    (None, 5)            0           no_mask_103[0][0]                
                                                                 no_mask_103[1][0]                
__________________________________________________________________________________________________
linear_16 (Linear)              (None, 1)            1           concatenate_75[0][0]             
                                                                 no_mask_98[0][0]                 
__________________________________________________________________________________________________
dnn_17 (DNN)                    (None, 32)           1248        concatenate_77[0][0]             
__________________________________________________________________________________________________
no_mask_99 (NoMask)             (None, 1)            0           linear_16[0][0]                  
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1)            32          dnn_17[0][0]                     
__________________________________________________________________________________________________
add_26 (Add)                    (None, 1)            0           no_mask_99[0][0]                 
                                                                 dense_10[0][0]                   
__________________________________________________________________________________________________
prediction_layer_15 (Prediction (None, 1)            1           add_26[0][0]                     
==================================================================================================
Total params: 1,367
Trainable params: 1,367
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'ONN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'ONN', 'sparse_feature_num': 2, 'dense_feature_num': 2, 'sequence_feature': ('sum', 'mean', 'max'), 'hash_flag': True} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_ONN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_13"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_14 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
hash_15 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_16 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         32          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_107 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_spars
__________________________________________________________________________________________________
no_mask_108 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_spars
__________________________________________________________________________________________________
no_mask_109 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_178 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sparse_fe
__________________________________________________________________________________________________
no_mask_110 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_179 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sparse_f
__________________________________________________________________________________________________
no_mask_111 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_180 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sparse_fe
__________________________________________________________________________________________________
no_mask_112 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_181 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sparse_fe
__________________________________________________________________________________________________
no_mask_113 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_182 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sparse_f
__________________________________________________________________________________________________
no_mask_114 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_183 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sparse_fe
__________________________________________________________________________________________________
sequence_pooling_layer_184 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_185 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sequence
__________________________________________________________________________________________________
sequence_pooling_layer_186 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_187 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_188 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sequence
__________________________________________________________________________________________________
sequence_pooling_layer_189 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sequence_
__________________________________________________________________________________________________
multiply (Multiply)             (None, 1, 4)         0           no_mask_107[0][0]                
                                                                 no_mask_108[0][0]                
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 1, 4)         0           no_mask_109[0][0]                
                                                                 sequence_pooling_layer_178[0][0] 
__________________________________________________________________________________________________
multiply_2 (Multiply)           (None, 1, 4)         0           no_mask_110[0][0]                
                                                                 sequence_pooling_layer_179[0][0] 
__________________________________________________________________________________________________
multiply_3 (Multiply)           (None, 1, 4)         0           no_mask_111[0][0]                
                                                                 sequence_pooling_layer_180[0][0] 
__________________________________________________________________________________________________
multiply_4 (Multiply)           (None, 1, 4)         0           no_mask_112[0][0]                
                                                                 sequence_pooling_layer_181[0][0] 
__________________________________________________________________________________________________
multiply_5 (Multiply)           (None, 1, 4)         0           no_mask_113[0][0]                
                                                                 sequence_pooling_layer_182[0][0] 
__________________________________________________________________________________________________
multiply_6 (Multiply)           (None, 1, 4)         0           no_mask_114[0][0]                
                                                                 sequence_pooling_layer_183[0][0] 
__________________________________________________________________________________________________
multiply_7 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_184[0][0] 
                                                                 sequence_pooling_layer_185[0][0] 
__________________________________________________________________________________________________
multiply_8 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_186[0][0] 
                                                                 sequence_pooling_layer_187[0][0] 
__________________________________________________________________________________________________
multiply_9 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_188[0][0] 
                                                                 sequence_pooling_layer_189[0][0] 
__________________________________________________________________________________________________
no_mask_115 (NoMask)            (None, 1, 4)         0           multiply[0][0]                   
                                                                 multiply_1[0][0]                 
                                                                 multiply_2[0][0]                 
                                                                 multiply_3[0][0]                 
                                                                 multiply_4[0][0]                 
                                                                 multiply_5[0][0]                 
                                                                 multiply_6[0][0]                 
                                                                 multiply_7[0][0]                 
                                                                 multiply_8[0][0]                 
                                                                 multiply_9[0][0]                 
__________________________________________________________________________________________________
concatenate_80 (Concatenate)    (None, 10, 4)        0           no_mask_115[0][0]                
                                                                 no_mask_115[1][0]                
                                                                 no_mask_115[2][0]                
                                                                 no_mask_115[3][0]                
                                                                 no_mask_115[4][0]                
                                                                 no_mask_115[5][0]                
                                                                 no_mask_115[6][0]                
                                                                 no_mask_115[7][0]                
                                                                 no_mask_115[8][0]                
                                                                 no_mask_115[9][0]                
__________________________________________________________________________________________________
flatten_28 (Flatten)            (None, 40)           0           concatenate_80[0][0]             
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 40)           160         flatten_28[0][0]                 
__________________________________________________________________________________________________
no_mask_117 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
no_mask_116 (NoMask)            (None, 40)           0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
concatenate_81 (Concatenate)    (None, 2)            0           no_mask_117[0][0]                
                                                                 no_mask_117[1][0]                
__________________________________________________________________________________________________
hash_10 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
hash_11 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_11[0][0]                    
__________________________________________________________________________________________________
sequence_pooling_layer_172 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_173 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_174 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_118 (NoMask)            multiple             0           flatten_29[0][0]                 
                                                                 flatten_30[0][0]                 
__________________________________________________________________________________________________
no_mask_104 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_172[0][0] 
                                                                 sequence_pooling_layer_173[0][0] 
                                                                 sequence_pooling_layer_174[0][0] 
__________________________________________________________________________________________________
no_mask_105 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
concatenate_82 (Concatenate)    (None, 42)           0           no_mask_118[0][0]                
                                                                 no_mask_118[1][0]                
__________________________________________________________________________________________________
concatenate_78 (Concatenate)    (None, 1, 5)         0           no_mask_104[0][0]                
                                                                 no_mask_104[1][0]                
                                                                 no_mask_104[2][0]                
                                                                 no_mask_104[3][0]                
                                                                 no_mask_104[4][0]                
__________________________________________________________________________________________________
concatenate_79 (Concatenate)    (None, 2)            0           no_mask_105[0][0]                
                                                                 no_mask_105[1][0]                
__________________________________________________________________________________________________
dnn_18 (DNN)                    (None, 32)           2432        concatenate_82[0][0]             
__________________________________________________________________________________________________
linear_17 (Linear)              (None, 1)            2           concatenate_78[0][0]             
                                                                 concatenate_79[0][0]             
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1)            32          dnn_18[0][0]                     
__________________________________________________________________________________________________
no_mask_106 (NoMask)            (None, 1)            0           linear_17[0][0]                  
__________________________________________________________________________________________________
add_29 (Add)                    (None, 1)            0           dense_11[0][0]                   
                                                                 no_mask_106[0][0]                
__________________________________________________________________________________________________
prediction_layer_16 (Prediction (None, 1)            1           add_29[0][0]                     
==================================================================================================
Total params: 3,052
Trainable params: 2,972
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2511 - binary_crossentropy: 0.6975500/500 [==============================] - 6s 13ms/sample - loss: 0.2603 - binary_crossentropy: 0.7154 - val_loss: 0.2753 - val_binary_crossentropy: 0.8001

  #### metrics   #################################################### 
{'MSE': 0.26683485830208453}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/init_ops.py:97: calling Ones.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
Model: "model_13"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_14 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
hash_15 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_16 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         32          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         16          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_107 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_spars
__________________________________________________________________________________________________
no_mask_108 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_spars
__________________________________________________________________________________________________
no_mask_109 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_178 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sparse_fe
__________________________________________________________________________________________________
no_mask_110 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_179 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sparse_f
__________________________________________________________________________________________________
no_mask_111 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0_seque
__________________________________________________________________________________________________
sequence_pooling_layer_180 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sparse_fe
__________________________________________________________________________________________________
no_mask_112 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_181 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sparse_fe
__________________________________________________________________________________________________
no_mask_113 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_182 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sparse_f
__________________________________________________________________________________________________
no_mask_114 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_1_seque
__________________________________________________________________________________________________
sequence_pooling_layer_183 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sparse_fe
__________________________________________________________________________________________________
sequence_pooling_layer_184 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_185 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sequence
__________________________________________________________________________________________________
sequence_pooling_layer_186 (Seq (None, 1, 4)         0           sparse_emb_sequence_sum_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_187 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sequence_
__________________________________________________________________________________________________
sequence_pooling_layer_188 (Seq (None, 1, 4)         0           sparse_emb_sequence_mean_sequence
__________________________________________________________________________________________________
sequence_pooling_layer_189 (Seq (None, 1, 4)         0           sparse_emb_sequence_max_sequence_
__________________________________________________________________________________________________
multiply (Multiply)             (None, 1, 4)         0           no_mask_107[0][0]                
                                                                 no_mask_108[0][0]                
__________________________________________________________________________________________________
multiply_1 (Multiply)           (None, 1, 4)         0           no_mask_109[0][0]                
                                                                 sequence_pooling_layer_178[0][0] 
__________________________________________________________________________________________________
multiply_2 (Multiply)           (None, 1, 4)         0           no_mask_110[0][0]                
                                                                 sequence_pooling_layer_179[0][0] 
__________________________________________________________________________________________________
multiply_3 (Multiply)           (None, 1, 4)         0           no_mask_111[0][0]                
                                                                 sequence_pooling_layer_180[0][0] 
__________________________________________________________________________________________________
multiply_4 (Multiply)           (None, 1, 4)         0           no_mask_112[0][0]                
                                                                 sequence_pooling_layer_181[0][0] 
__________________________________________________________________________________________________
multiply_5 (Multiply)           (None, 1, 4)         0           no_mask_113[0][0]                
                                                                 sequence_pooling_layer_182[0][0] 
__________________________________________________________________________________________________
multiply_6 (Multiply)           (None, 1, 4)         0           no_mask_114[0][0]                
                                                                 sequence_pooling_layer_183[0][0] 
__________________________________________________________________________________________________
multiply_7 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_184[0][0] 
                                                                 sequence_pooling_layer_185[0][0] 
__________________________________________________________________________________________________
multiply_8 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_186[0][0] 
                                                                 sequence_pooling_layer_187[0][0] 
__________________________________________________________________________________________________
multiply_9 (Multiply)           (None, 1, 4)         0           sequence_pooling_layer_188[0][0] 
                                                                 sequence_pooling_layer_189[0][0] 
__________________________________________________________________________________________________
no_mask_115 (NoMask)            (None, 1, 4)         0           multiply[0][0]                   
                                                                 multiply_1[0][0]                 
                                                                 multiply_2[0][0]                 
                                                                 multiply_3[0][0]                 
                                                                 multiply_4[0][0]                 
                                                                 multiply_5[0][0]                 
                                                                 multiply_6[0][0]                 
                                                                 multiply_7[0][0]                 
                                                                 multiply_8[0][0]                 
                                                                 multiply_9[0][0]                 
__________________________________________________________________________________________________
concatenate_80 (Concatenate)    (None, 10, 4)        0           no_mask_115[0][0]                
                                                                 no_mask_115[1][0]                
                                                                 no_mask_115[2][0]                
                                                                 no_mask_115[3][0]                
                                                                 no_mask_115[4][0]                
                                                                 no_mask_115[5][0]                
                                                                 no_mask_115[6][0]                
                                                                 no_mask_115[7][0]                
                                                                 no_mask_115[8][0]                
                                                                 no_mask_115[9][0]                
__________________________________________________________________________________________________
flatten_28 (Flatten)            (None, 40)           0           concatenate_80[0][0]             
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
dense_feature_1 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 40)           160         flatten_28[0][0]                 
__________________________________________________________________________________________________
no_mask_117 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
no_mask_116 (NoMask)            (None, 40)           0           batch_normalization_8[0][0]      
__________________________________________________________________________________________________
concatenate_81 (Concatenate)    (None, 2)            0           no_mask_117[0][0]                
                                                                 no_mask_117[1][0]                
__________________________________________________________________________________________________
hash_10 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
hash_11 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_11[0][0]                    
__________________________________________________________________________________________________
sequence_pooling_layer_172 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_173 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_174 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
no_mask_118 (NoMask)            multiple             0           flatten_29[0][0]                 
                                                                 flatten_30[0][0]                 
__________________________________________________________________________________________________
no_mask_104 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_172[0][0] 
                                                                 sequence_pooling_layer_173[0][0] 
                                                                 sequence_pooling_layer_174[0][0] 
__________________________________________________________________________________________________
no_mask_105 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
                                                                 dense_feature_1[0][0]            
__________________________________________________________________________________________________
concatenate_82 (Concatenate)    (None, 42)           0           no_mask_118[0][0]                
                                                                 no_mask_118[1][0]                
__________________________________________________________________________________________________
concatenate_78 (Concatenate)    (None, 1, 5)         0           no_mask_104[0][0]                
                                                                 no_mask_104[1][0]                
                                                                 no_mask_104[2][0]                
                                                                 no_mask_104[3][0]                
                                                                 no_mask_104[4][0]                
__________________________________________________________________________________________________
concatenate_79 (Concatenate)    (None, 2)            0           no_mask_105[0][0]                
                                                                 no_mask_105[1][0]                
__________________________________________________________________________________________________
dnn_18 (DNN)                    (None, 32)           2432        concatenate_82[0][0]             
__________________________________________________________________________________________________
linear_17 (Linear)              (None, 1)            2           concatenate_78[0][0]             
                                                                 concatenate_79[0][0]             
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 1)            32          dnn_18[0][0]                     
__________________________________________________________________________________________________
no_mask_106 (NoMask)            (None, 1)            0           linear_17[0][0]                  
__________________________________________________________________________________________________
add_29 (Add)                    (None, 1)            0           dense_11[0][0]                   
                                                                 no_mask_106[0][0]                
__________________________________________________________________________________________________
prediction_layer_16 (Prediction (None, 1)            1           add_29[0][0]                     
==================================================================================================
Total params: 3,052
Trainable params: 2,972
Non-trainable params: 80
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'PNN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'PNN', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_PNN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//01_deepctr.py", line 541, in <module>
    test(pars_choice=5, **{"model_name": model_name})
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//01_deepctr.py", line 517, in test
    module, model = module_load_full("model_keras.01_deepctr", model_pars, data_pars, compute_pars, dataset=dataset)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 101, in module_load_full
    model = module.Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, **kwarg)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/01_deepctr.py", line 155, in __init__
    self.model = modeli(feature_columns, **MODEL_PARAMS[model_name])
TypeError: PNN() got an unexpected keyword argument 'embedding_size'

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
   46cfa4a..da60bc6  master     -> origin/master
Updating 46cfa4a..da60bc6
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |  2 +-
 error_list/20200522/list_log_testall_20200522.md   | 92 ++++++++++++++++++++++
 2 files changed, 93 insertions(+), 1 deletion(-)
[master b54863b] ml_store
 1 file changed, 4953 insertions(+)
To github.com:arita37/mlmodels_store.git
   da60bc6..b54863b  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py 
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset

  #### Loading params   ############################################## 

  #### Loading daaset   ############################################# 
Loading data...
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/train.csv
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/test.csv

  #### Model init, fit   ############################################# 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:4070: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

2020-05-22 13:04:56.476173: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-22 13:04:56.480784: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-22 13:04:56.480941: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5564c1985f90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 13:04:56.480953: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

CharCNNZhang model built: 
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sent_input (InputLayer)      (None, 1014)              0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 1014, 128)         8960      
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 1008, 256)         229632    
_________________________________________________________________
thresholded_re_lu_1 (Thresho (None, 1008, 256)         0         
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 336, 256)          0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 330, 256)          459008    
_________________________________________________________________
thresholded_re_lu_2 (Thresho (None, 330, 256)          0         
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 110, 256)          0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 108, 256)          196864    
_________________________________________________________________
thresholded_re_lu_3 (Thresho (None, 108, 256)          0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 106, 256)          196864    
_________________________________________________________________
thresholded_re_lu_4 (Thresho (None, 106, 256)          0         
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 104, 256)          196864    
_________________________________________________________________
thresholded_re_lu_5 (Thresho (None, 104, 256)          0         
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 102, 256)          196864    
_________________________________________________________________
thresholded_re_lu_6 (Thresho (None, 102, 256)          0         
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 34, 256)           0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8704)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              8913920   
_________________________________________________________________
thresholded_re_lu_7 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
thresholded_re_lu_8 (Thresho (None, 1024)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 4100      
=================================================================
Total params: 11,452,676
Trainable params: 11,452,676
Non-trainable params: 0
_________________________________________________________________
Loading data...
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/train.csv
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/test.csv
Train on 354 samples, validate on 236 samples
Epoch 1/1

128/354 [=========>....................] - ETA: 5s - loss: 1.3882
256/354 [====================>.........] - ETA: 2s - loss: 1.2767
354/354 [==============================] - 9s 27ms/step - loss: 1.2324 - val_loss: 1.8526

  #### Predict   ##################################################### 
Data loaded from /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/test.csv

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/callbacks/callbacks.py:846: RuntimeWarning: Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: val_loss,loss
  (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
{'path': 'ztest/ml_keras/charcnn_zhang/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
{'path': 'ztest/ml_keras/charcnn_zhang/', 'data_type': 'pandas', 'size': [0, 0, 6], 'output_size': [0, 6]}
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 284, in <module>
    test(pars_choice="json", data_path= f"{root_path}/model_keras/charcnn_zhang.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 268, in test
    model2 = load(out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 118, in load
    model = load_keras(load_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 602, in load_keras
    model.model = load_model(path_file)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/saving/save.py", line 146, in load_model
    loader_impl.parse_saved_model(filepath)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/saved_model/loader_impl.py", line 83, in parse_saved_model
    constants.SAVED_MODEL_FILENAME_PB))
OSError: SavedModel file does not exist at: ztest/ml_keras/charcnn_zhang//model.h5/{saved_model.pbtxt|saved_model.pb}

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
[master bc5e70a] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   b54863b..bc5e70a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3260416/17464789 [====>.........................] - ETA: 0s
 7487488/17464789 [===========>..................] - ETA: 0s
11927552/17464789 [===================>..........] - ETA: 0s
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)...

  #### Model init, fit   ############################################# 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-22 13:05:30.015432: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-22 13:05:30.020309: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-22 13:05:30.020462: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562a3581f2a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 13:05:30.020475: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 40, 50)       250         input_1[0][0]                    
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 38, 128)      19328       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 37, 128)      25728       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, 36, 128)      32128       embedding_1[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 128)          0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 128)          0           conv1d_3[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 384)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            385         concatenate_1[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8506 - accuracy: 0.4880
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8660 - accuracy: 0.4870 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8455 - accuracy: 0.4883
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8775 - accuracy: 0.4863
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7402 - accuracy: 0.4952
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6845 - accuracy: 0.4988
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6754 - accuracy: 0.4994
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7107 - accuracy: 0.4971
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6927 - accuracy: 0.4983
11000/25000 [============>.................] - ETA: 3s - loss: 7.6834 - accuracy: 0.4989
12000/25000 [=============>................] - ETA: 3s - loss: 7.6768 - accuracy: 0.4993
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6654 - accuracy: 0.5001
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6425 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 2s - loss: 7.6728 - accuracy: 0.4996
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6810 - accuracy: 0.4991
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6838 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6828 - accuracy: 0.4989
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6900 - accuracy: 0.4985
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6988 - accuracy: 0.4979
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7017 - accuracy: 0.4977
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6931 - accuracy: 0.4983
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6900 - accuracy: 0.4985
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6775 - accuracy: 0.4993
25000/25000 [==============================] - 7s 273us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### save the trained model  ####################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5'}

  #### Predict   ##################################################### 
Loading data...

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/keras/initializers.py:119: calling RandomUniform.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/textcnn/model.h5'}
(<mlmodels.util.Model_empty object at 0x7f13034efda0>, None)

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 40, 50)       250         input_2[0][0]                    
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, 38, 128)      19328       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 37, 128)      25728       embedding_2[0][0]                
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 36, 128)      32128       embedding_2[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_4 (GlobalM (None, 128)          0           conv1d_4[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_5 (GlobalM (None, 128)          0           conv1d_5[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_6 (GlobalM (None, 128)          0           conv1d_6[0][0]                   
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 384)          0           global_max_pooling1d_4[0][0]     
                                                                 global_max_pooling1d_5[0][0]     
                                                                 global_max_pooling1d_6[0][0]     
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            385         concatenate_2[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  <mlmodels.model_keras.textcnn.Model object at 0x7f1329cb39e8> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7816 - accuracy: 0.4925 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6615 - accuracy: 0.5003
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7088 - accuracy: 0.4972
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6789 - accuracy: 0.4992
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6922 - accuracy: 0.4983
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6228 - accuracy: 0.5029
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6570 - accuracy: 0.5006
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6912 - accuracy: 0.4984
11000/25000 [============>.................] - ETA: 3s - loss: 7.6792 - accuracy: 0.4992
12000/25000 [=============>................] - ETA: 3s - loss: 7.6602 - accuracy: 0.5004
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
15000/25000 [=================>............] - ETA: 2s - loss: 7.6523 - accuracy: 0.5009
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6542 - accuracy: 0.5008
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6423 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6402 - accuracy: 0.5017
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6368 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6615 - accuracy: 0.5003
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6422 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6646 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 7s 273us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

  #### Predict   #################################################### 
Loading data...
(array([[1.],
       [1.],
       [1.],
       ...,
       [1.],
       [1.],
       [1.]], dtype=float32), None)

  #### Get  metrics   ################################################ 

  #### Save   ######################################################## 

  #### Load   ######################################################## 

  ############ Model preparation   ################################## 

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.textcnn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 
Model: "model_3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_3 (InputLayer)            (None, 40)           0                                            
__________________________________________________________________________________________________
embedding_3 (Embedding)         (None, 40, 50)       250         input_3[0][0]                    
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, 38, 128)      19328       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, 37, 128)      25728       embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, 36, 128)      32128       embedding_3[0][0]                
__________________________________________________________________________________________________
global_max_pooling1d_7 (GlobalM (None, 128)          0           conv1d_7[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_8 (GlobalM (None, 128)          0           conv1d_8[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_9 (GlobalM (None, 128)          0           conv1d_9[0][0]                   
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 384)          0           global_max_pooling1d_7[0][0]     
                                                                 global_max_pooling1d_8[0][0]     
                                                                 global_max_pooling1d_9[0][0]     
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            385         concatenate_3[0][0]              
==================================================================================================
Total params: 77,819
Trainable params: 77,819
Non-trainable params: 0
__________________________________________________________________________________________________

  ############ Model fit   ########################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.8966 - accuracy: 0.4850
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5951 - accuracy: 0.5047
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7101 - accuracy: 0.4972
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6535 - accuracy: 0.5009
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6958 - accuracy: 0.4981
11000/25000 [============>.................] - ETA: 3s - loss: 7.7015 - accuracy: 0.4977
12000/25000 [=============>................] - ETA: 3s - loss: 7.6756 - accuracy: 0.4994
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6678 - accuracy: 0.4999
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 2s - loss: 7.6656 - accuracy: 0.5001
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6542 - accuracy: 0.5008
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6693 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6675 - accuracy: 0.4999
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6707 - accuracy: 0.4997
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6889 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6907 - accuracy: 0.4984
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6757 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
25000/25000 [==============================] - 7s 275us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
fit success None

  ############ Prediction############################################ 
Loading data...
(array([[1.],
       [1.],
       [1.],
       ...,
       [1.],
       [1.],
       [1.]], dtype=float32), None)

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
From github.com:arita37/mlmodels_store
   bc5e70a..235d56e  master     -> origin/master
Updating bc5e70a..235d56e
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |  2 +-
 error_list/20200522/list_log_testall_20200522.md   | 30 ++++++++++++++++++++++
 2 files changed, 31 insertions(+), 1 deletion(-)
[master 5691bb8] ml_store
 1 file changed, 324 insertions(+)
To github.com:arita37/mlmodels_store.git
   235d56e..5691bb8  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:12<99:27:39, 12.16s/it][A
Iteration:   0%|          | 2/29440 [01:40<286:12:44, 35.00s/it][A
Iteration:   0%|          | 3/29440 [02:53<379:36:25, 46.42s/it][A
Iteration:   0%|          | 4/29440 [03:33<364:21:50, 44.56s/it][A
Iteration:   0%|          | 5/29440 [04:37<411:04:02, 50.27s/it][A
Iteration:   0%|          | 6/29440 [05:56<482:09:21, 58.97s/it][A
Iteration:   0%|          | 7/29440 [07:16<533:42:11, 65.28s/it][A
Iteration:   0%|          | 8/29440 [09:11<655:15:07, 80.15s/it][A
Iteration:   0%|          | 9/29440 [11:11<752:46:39, 92.08s/it][A
Iteration:   0%|          | 10/29440 [15:40<1185:13:17, 144.98s/it][A
Iteration:   0%|          | 11/29440 [16:30<952:36:44, 116.53s/it] [A
Iteration:   0%|          | 12/29440 [18:38<982:14:15, 120.16s/it][A
Iteration:   0%|          | 13/29440 [19:29<810:49:05, 99.19s/it] [A
Iteration:   0%|          | 14/29440 [20:30<718:12:08, 87.87s/it][A
Iteration:   0%|          | 15/29440 [23:17<910:49:52, 111.44s/it][A
Iteration:   0%|          | 16/29440 [24:43<849:07:43, 103.89s/it][A
Iteration:   0%|          | 17/29440 [25:39<731:53:08, 89.55s/it] [A
Iteration:   0%|          | 18/29440 [28:16<896:35:32, 109.70s/it][A
Iteration:   0%|          | 19/29440 [29:30<810:42:49, 99.20s/it] [A
Iteration:   0%|          | 20/29440 [32:34<1017:02:25, 124.45s/it][A
Iteration:   0%|          | 21/29440 [34:20<971:58:09, 118.94s/it] [A
Iteration:   0%|          | 22/29440 [35:01<780:31:30, 95.52s/it] [A
Iteration:   0%|          | 23/29440 [38:20<1036:04:32, 126.79s/it][A
Iteration:   0%|          | 24/29440 [40:00<970:05:36, 118.72s/it] [A
Iteration:   0%|          | 25/29440 [41:20<873:22:15, 106.89s/it][A
Iteration:   0%|          | 26/29440 [42:18<755:27:44, 92.46s/it] [A
Iteration:   0%|          | 27/29440 [42:47<599:19:22, 73.35s/it][A
Iteration:   0%|          | 28/29440 [43:28<520:13:29, 63.67s/it][A
Iteration:   0%|          | 29/29440 [45:39<685:51:07, 83.95s/it][A
Iteration:   0%|          | 30/29440 [48:14<859:42:50, 105.24s/it][A
Iteration:   0%|          | 31/29440 [51:03<1015:11:45, 124.27s/it][A
Iteration:   0%|          | 32/29440 [52:07<866:02:23, 106.02s/it] [A
Iteration:   0%|          | 33/29440 [52:50<712:39:59, 87.24s/it] [A
Iteration:   0%|          | 34/29440 [53:40<621:48:49, 76.12s/it][A
Iteration:   0%|          | 35/29440 [55:19<677:01:00, 82.89s/it][A
Iteration:   0%|          | 36/29440 [56:07<590:55:57, 72.35s/it][A
Iteration:   0%|          | 37/29440 [58:27<757:33:18, 92.75s/it][A
Iteration:   0%|          | 38/29440 [59:12<639:53:50, 78.35s/it][A
Iteration:   0%|          | 39/29440 [1:01:30<786:49:17, 96.34s/it][A
Iteration:   0%|          | 40/29440 [1:04:19<965:27:37, 118.22s/it][A
Iteration:   0%|          | 41/29440 [1:04:43<732:48:39, 89.74s/it] [A
Iteration:   0%|          | 42/29440 [1:05:07<572:36:34, 70.12s/it][A
Iteration:   0%|          | 43/29440 [1:06:17<573:49:40, 70.27s/it][A
Iteration:   0%|          | 44/29440 [1:06:58<501:43:00, 61.44s/it][A
Iteration:   0%|          | 45/29440 [1:09:33<730:33:13, 89.47s/it][A
Iteration:   0%|          | 46/29440 [1:10:39<671:48:38, 82.28s/it][A
Iteration:   0%|          | 47/29440 [1:12:29<739:30:42, 90.57s/it][AKilled

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
   5691bb8..6d747b8  master     -> origin/master
Updating 5691bb8..6d747b8
Fast-forward
 deps.txt                                           |   6 +-
 .../20200522/list_log_pullrequest_20200522.md      |   2 +-
 ...-09_a463a24ea257f46bfcbd4006f805952aace8f2b1.py | 626 ++++++++++++++++++++
 ...-10_a463a24ea257f46bfcbd4006f805952aace8f2b1.py | 658 +++++++++++++++++++++
 4 files changed, 1290 insertions(+), 2 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-22-13-09_a463a24ea257f46bfcbd4006f805952aace8f2b1.py
 create mode 100644 log_pullrequest/log_pr_2020-05-22-14-10_a463a24ea257f46bfcbd4006f805952aace8f2b1.py
[master f06fe31] ml_store
 1 file changed, 99 insertions(+)
To github.com:arita37/mlmodels_store.git
   6d747b8..f06fe31  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
    "beta_vae": md.model.beta_vae,
AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'

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
[master 20035b3] ml_store
 2 files changed, 37 insertions(+), 5 deletions(-)
To github.com:arita37/mlmodels_store.git
   f06fe31..20035b3  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
    from dataloader import DataLoader
ModuleNotFoundError: No module named 'dataloader'

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
[master 7b0a690] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   20035b3..7b0a690  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 522, in <module>
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 418, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: 'model_tch/transformer_classifier.json'

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
[master 568cc99] ml_store
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   7b0a690..568cc99  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//mlp.py 

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
[master 88c458d] ml_store
 1 file changed, 32 insertions(+)
To github.com:arita37/mlmodels_store.git
   568cc99..88c458d  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f625d1adea0>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f625d1adea0>

  function with postional parmater data_info <function get_dataset_torch at 0x7f625d1adea0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:02<?, ?it/s] 35%|███▍      | 3440640/9912422 [00:02<00:00, 34378237.13it/s]9920512it [00:02, 3921724.97it/s]                              
0it [00:00, ?it/s]32768it [00:00, 639613.89it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 475514.66it/s]1654784it [00:00, 11985989.74it/s]                         
0it [00:00, ?it/s]8192it [00:00, 255656.62it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  #### Model init, fit   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f625c4e6bf8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f625c4e6bf8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f625c4e6bf8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.001933109720547994 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.010881881237030029 	 Accuracy: 1
model saves at 1 accuracy
Train Epoch: 2 	 Loss: 0.0013434533675511678 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.01120279848575592 	 Accuracy: 0

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f625c4e69d8>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f625c4e69d8>

  function with postional parmater data_info <function get_dataset_torch at 0x7f625c4e69d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['torch_model_pars.pkl', 'model.pb'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7f625cf627f0>

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels.preprocess.generic::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/pytorch_GAN_zoo/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/pytorch_GAN_zoo/'} 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### Predict   ##################################################### 
img_01.png

  #### metrics   ##################################################### 

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/pytorch_GAN_zoo//torch_model/ ['torch_model_pars.pkl', 'model.pb'] 
img_01.png
torch_model

Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/facebookresearch_pytorch_GAN_zoo_hub
<__main__.Model object at 0x7f62549b36d8>

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
[master cff8139] ml_store
 1 file changed, 147 insertions(+)
To github.com:arita37/mlmodels_store.git
   88c458d..cff8139  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pplm.py 
 Generating text ... 
= Prefix of sentence =
<|endoftext|>The potato

 Unperturbed generated text :

<|endoftext|>The potato-shaped, potato-eating insect of modern times (Ophiocordyceps elegans) has a unique ability to adapt quickly to a wide range of environments. It is able to survive in many different environments, including the Arctic, deserts

 Perturbed generated text :

<|endoftext|>The potato bomb is nothing new. It's been on the news a lot since 9/11. In fact, since the bombing in Paris last November, a bomb has been detonated in every major European country in the European Union.

The bomb in Brussels


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
   cff8139..27ceecd  master     -> origin/master
Updating cff8139..27ceecd
Fast-forward
 .../20200522/list_log_pullrequest_20200522.md      |  2 +-
 error_list/20200522/list_log_testall_20200522.md   | 23 ++++++++++++++++++++++
 2 files changed, 24 insertions(+), 1 deletion(-)
[master 4044870] ml_store
 1 file changed, 52 insertions(+)
To github.com:arita37/mlmodels_store.git
   27ceecd..4044870  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//nbeats.py 

  #### Loading params   ####################################### 

  #### Loading dataset  ####################################### 
   milk_production_pounds
0                     589
1                     561
2                     640
3                     656
4                     727
[[0.60784314]
 [0.57894737]
 [0.66047472]
 [0.67698658]
 [0.750258  ]
 [0.71929825]
 [0.66047472]
 [0.61816305]
 [0.58617131]
 [0.59545924]
 [0.57069143]
 [0.6006192 ]
 [0.61919505]
 [0.58410733]
 [0.67389061]
 [0.69453044]
 [0.76573787]
 [0.73890609]
 [0.68111455]
 [0.63673891]
 [0.60165119]
 [0.60577915]
 [0.58307534]
 [0.61713106]
 [0.64809082]
 [0.6377709 ]
 [0.71001032]
 [0.72755418]
 [0.79463364]
 [0.75954592]
 [0.6996904 ]
 [0.65944272]
 [0.62332301]
 [0.63054696]
 [0.6130031 ]
 [0.65428277]
 [0.67905057]
 [0.64189886]
 [0.73168215]
 [0.74509804]
 [0.80701754]
 [0.78018576]
 [0.7244582 ]
 [0.67389061]
 [0.63467492]
 [0.64086687]
 [0.62125903]
 [0.65531476]
 [0.69865841]
 [0.65531476]
 [0.75954592]
 [0.77915377]
 [0.8369453 ]
 [0.82352941]
 [0.75851393]
 [0.71929825]
 [0.68214654]
 [0.68833849]
 [0.66563467]
 [0.71001032]
 [0.73581011]
 [0.68833849]
 [0.78637771]
 [0.80908153]
 [0.86377709]
 [0.84313725]
 [0.79153767]
 [0.74509804]
 [0.70278638]
 [0.70897833]
 [0.68111455]
 [0.72033024]
 [0.73993808]
 [0.71826625]
 [0.7997936 ]
 [0.82146543]
 [0.88544892]
 [0.85242518]
 [0.80804954]
 [0.76367389]
 [0.72342621]
 [0.72858617]
 [0.69865841]
 [0.73374613]
 [0.75748194]
 [0.7120743 ]
 [0.81011352]
 [0.83075335]
 [0.89886481]
 [0.87203302]
 [0.82662539]
 [0.78844169]
 [0.74819401]
 [0.74613003]
 [0.7120743 ]
 [0.75748194]
 [0.77399381]
 [0.72961816]
 [0.83281734]
 [0.8503612 ]
 [0.91434469]
 [0.88648091]
 [0.84520124]
 [0.80804954]
 [0.76367389]
 [0.77089783]
 [0.73374613]
 [0.7750258 ]
 [0.82972136]
 [0.78018576]
 [0.8875129 ]
 [0.90608875]
 [0.97213622]
 [0.94220846]
 [0.89680083]
 [0.86068111]
 [0.81527348]
 [0.8255934 ]
 [0.7874097 ]
 [0.8255934 ]
 [0.85242518]
 [0.8245614 ]
 [0.91847265]
 [0.92879257]
 [0.99174407]
 [0.96491228]
 [0.92260062]
 [0.88235294]
 [0.83488132]
 [0.83591331]
 [0.79050568]
 [0.83075335]
 [0.84726522]
 [0.79772962]
 [0.91124871]
 [0.92672859]
 [0.9876161 ]
 [0.95356037]
 [0.90918473]
 [0.86377709]
 [0.80908153]
 [0.81630547]
 [0.78431373]
 [0.82765738]
 [0.85448916]
 [0.80288958]
 [0.91744066]
 [0.93085655]
 [1.        ]
 [0.97729618]
 [0.9370485 ]
 [0.89473684]
 [0.84107327]
 [0.8379773 ]
 [0.79772962]
 [0.83900929]
 [0.86068111]
 [0.80701754]
 [0.92053664]
 [0.93188854]
 [0.99690402]
 [0.96697626]
 [0.9246646 ]
 [0.88544892]
 [0.84313725]
 [0.85345717]
 [0.82249742]
 [0.86996904]]

  #### Model setup   ########################################## 
| N-Beats
| --  Stack Generic (#0) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139774524366800
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139774524366576
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139774524365344
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139774524364896
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139774524364392
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139774524364056

  #### Model fit   ############################################ 
   milk_production_pounds
0                     589
1                     561
2                     640
3                     656
4                     727
[[0.60784314]
 [0.57894737]
 [0.66047472]
 [0.67698658]
 [0.750258  ]
 [0.71929825]
 [0.66047472]
 [0.61816305]
 [0.58617131]
 [0.59545924]
 [0.57069143]
 [0.6006192 ]
 [0.61919505]
 [0.58410733]
 [0.67389061]
 [0.69453044]
 [0.76573787]
 [0.73890609]
 [0.68111455]
 [0.63673891]
 [0.60165119]
 [0.60577915]
 [0.58307534]
 [0.61713106]
 [0.64809082]
 [0.6377709 ]
 [0.71001032]
 [0.72755418]
 [0.79463364]
 [0.75954592]
 [0.6996904 ]
 [0.65944272]
 [0.62332301]
 [0.63054696]
 [0.6130031 ]
 [0.65428277]
 [0.67905057]
 [0.64189886]
 [0.73168215]
 [0.74509804]
 [0.80701754]
 [0.78018576]
 [0.7244582 ]
 [0.67389061]
 [0.63467492]
 [0.64086687]
 [0.62125903]
 [0.65531476]
 [0.69865841]
 [0.65531476]
 [0.75954592]
 [0.77915377]
 [0.8369453 ]
 [0.82352941]
 [0.75851393]
 [0.71929825]
 [0.68214654]
 [0.68833849]
 [0.66563467]
 [0.71001032]
 [0.73581011]
 [0.68833849]
 [0.78637771]
 [0.80908153]
 [0.86377709]
 [0.84313725]
 [0.79153767]
 [0.74509804]
 [0.70278638]
 [0.70897833]
 [0.68111455]
 [0.72033024]
 [0.73993808]
 [0.71826625]
 [0.7997936 ]
 [0.82146543]
 [0.88544892]
 [0.85242518]
 [0.80804954]
 [0.76367389]
 [0.72342621]
 [0.72858617]
 [0.69865841]
 [0.73374613]
 [0.75748194]
 [0.7120743 ]
 [0.81011352]
 [0.83075335]
 [0.89886481]
 [0.87203302]
 [0.82662539]
 [0.78844169]
 [0.74819401]
 [0.74613003]
 [0.7120743 ]
 [0.75748194]
 [0.77399381]
 [0.72961816]
 [0.83281734]
 [0.8503612 ]
 [0.91434469]
 [0.88648091]
 [0.84520124]
 [0.80804954]
 [0.76367389]
 [0.77089783]
 [0.73374613]
 [0.7750258 ]
 [0.82972136]
 [0.78018576]
 [0.8875129 ]
 [0.90608875]
 [0.97213622]
 [0.94220846]
 [0.89680083]
 [0.86068111]
 [0.81527348]
 [0.8255934 ]
 [0.7874097 ]
 [0.8255934 ]
 [0.85242518]
 [0.8245614 ]
 [0.91847265]
 [0.92879257]
 [0.99174407]
 [0.96491228]
 [0.92260062]
 [0.88235294]
 [0.83488132]
 [0.83591331]
 [0.79050568]
 [0.83075335]
 [0.84726522]
 [0.79772962]
 [0.91124871]
 [0.92672859]
 [0.9876161 ]
 [0.95356037]
 [0.90918473]
 [0.86377709]
 [0.80908153]
 [0.81630547]
 [0.78431373]
 [0.82765738]
 [0.85448916]
 [0.80288958]
 [0.91744066]
 [0.93085655]
 [1.        ]
 [0.97729618]
 [0.9370485 ]
 [0.89473684]
 [0.84107327]
 [0.8379773 ]
 [0.79772962]
 [0.83900929]
 [0.86068111]
 [0.80701754]
 [0.92053664]
 [0.93188854]
 [0.99690402]
 [0.96697626]
 [0.9246646 ]
 [0.88544892]
 [0.84313725]
 [0.85345717]
 [0.82249742]
 [0.86996904]]
--- fiting ---
grad_step = 000000, loss = 0.557448
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.402671
grad_step = 000002, loss = 0.280433
grad_step = 000003, loss = 0.163686
grad_step = 000004, loss = 0.062166
grad_step = 000005, loss = 0.036537
grad_step = 000006, loss = 0.094672
grad_step = 000007, loss = 0.085897
grad_step = 000008, loss = 0.038769
grad_step = 000009, loss = 0.009715
grad_step = 000010, loss = 0.009324
grad_step = 000011, loss = 0.021465
grad_step = 000012, loss = 0.031745
grad_step = 000013, loss = 0.034296
grad_step = 000014, loss = 0.030691
grad_step = 000015, loss = 0.024164
grad_step = 000016, loss = 0.017136
grad_step = 000017, loss = 0.011325
grad_step = 000018, loss = 0.008860
grad_step = 000019, loss = 0.009932
grad_step = 000020, loss = 0.012355
grad_step = 000021, loss = 0.013464
grad_step = 000022, loss = 0.012460
grad_step = 000023, loss = 0.010479
grad_step = 000024, loss = 0.008981
grad_step = 000025, loss = 0.008590
grad_step = 000026, loss = 0.008964
grad_step = 000027, loss = 0.009329
grad_step = 000028, loss = 0.009156
grad_step = 000029, loss = 0.008410
grad_step = 000030, loss = 0.007457
grad_step = 000031, loss = 0.006743
grad_step = 000032, loss = 0.006568
grad_step = 000033, loss = 0.006901
grad_step = 000034, loss = 0.007386
grad_step = 000035, loss = 0.007592
grad_step = 000036, loss = 0.007316
grad_step = 000037, loss = 0.006720
grad_step = 000038, loss = 0.006146
grad_step = 000039, loss = 0.005853
grad_step = 000040, loss = 0.005887
grad_step = 000041, loss = 0.006096
grad_step = 000042, loss = 0.006270
grad_step = 000043, loss = 0.006268
grad_step = 000044, loss = 0.006079
grad_step = 000045, loss = 0.005810
grad_step = 000046, loss = 0.005596
grad_step = 000047, loss = 0.005510
grad_step = 000048, loss = 0.005519
grad_step = 000049, loss = 0.005522
grad_step = 000050, loss = 0.005442
grad_step = 000051, loss = 0.005289
grad_step = 000052, loss = 0.005147
grad_step = 000053, loss = 0.005076
grad_step = 000054, loss = 0.005057
grad_step = 000055, loss = 0.005029
grad_step = 000056, loss = 0.004946
grad_step = 000057, loss = 0.004808
grad_step = 000058, loss = 0.004665
grad_step = 000059, loss = 0.004571
grad_step = 000060, loss = 0.004525
grad_step = 000061, loss = 0.004465
grad_step = 000062, loss = 0.004361
grad_step = 000063, loss = 0.004235
grad_step = 000064, loss = 0.004112
grad_step = 000065, loss = 0.004010
grad_step = 000066, loss = 0.003920
grad_step = 000067, loss = 0.003809
grad_step = 000068, loss = 0.003679
grad_step = 000069, loss = 0.003543
grad_step = 000070, loss = 0.003421
grad_step = 000071, loss = 0.003310
grad_step = 000072, loss = 0.003197
grad_step = 000073, loss = 0.003074
grad_step = 000074, loss = 0.002953
grad_step = 000075, loss = 0.002837
grad_step = 000076, loss = 0.002730
grad_step = 000077, loss = 0.002631
grad_step = 000078, loss = 0.002537
grad_step = 000079, loss = 0.002450
grad_step = 000080, loss = 0.002368
grad_step = 000081, loss = 0.002290
grad_step = 000082, loss = 0.002222
grad_step = 000083, loss = 0.002166
grad_step = 000084, loss = 0.002134
grad_step = 000085, loss = 0.002088
grad_step = 000086, loss = 0.001949
grad_step = 000087, loss = 0.001862
grad_step = 000088, loss = 0.001844
grad_step = 000089, loss = 0.001740
grad_step = 000090, loss = 0.001666
grad_step = 000091, loss = 0.001644
grad_step = 000092, loss = 0.001554
grad_step = 000093, loss = 0.001517
grad_step = 000094, loss = 0.001484
grad_step = 000095, loss = 0.001421
grad_step = 000096, loss = 0.001401
grad_step = 000097, loss = 0.001352
grad_step = 000098, loss = 0.001306
grad_step = 000099, loss = 0.001285
grad_step = 000100, loss = 0.001232
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.001201
grad_step = 000102, loss = 0.001179
grad_step = 000103, loss = 0.001137
grad_step = 000104, loss = 0.001114
grad_step = 000105, loss = 0.001096
grad_step = 000106, loss = 0.001063
grad_step = 000107, loss = 0.001038
grad_step = 000108, loss = 0.001022
grad_step = 000109, loss = 0.001003
grad_step = 000110, loss = 0.000979
grad_step = 000111, loss = 0.000959
grad_step = 000112, loss = 0.000943
grad_step = 000113, loss = 0.000928
grad_step = 000114, loss = 0.000910
grad_step = 000115, loss = 0.000891
grad_step = 000116, loss = 0.000874
grad_step = 000117, loss = 0.000862
grad_step = 000118, loss = 0.000851
grad_step = 000119, loss = 0.000836
grad_step = 000120, loss = 0.000818
grad_step = 000121, loss = 0.000802
grad_step = 000122, loss = 0.000791
grad_step = 000123, loss = 0.000782
grad_step = 000124, loss = 0.000773
grad_step = 000125, loss = 0.000761
grad_step = 000126, loss = 0.000748
grad_step = 000127, loss = 0.000738
grad_step = 000128, loss = 0.000728
grad_step = 000129, loss = 0.000719
grad_step = 000130, loss = 0.000707
grad_step = 000131, loss = 0.000695
grad_step = 000132, loss = 0.000685
grad_step = 000133, loss = 0.000677
grad_step = 000134, loss = 0.000669
grad_step = 000135, loss = 0.000661
grad_step = 000136, loss = 0.000651
grad_step = 000137, loss = 0.000642
grad_step = 000138, loss = 0.000636
grad_step = 000139, loss = 0.000632
grad_step = 000140, loss = 0.000635
grad_step = 000141, loss = 0.000652
grad_step = 000142, loss = 0.000692
grad_step = 000143, loss = 0.000730
grad_step = 000144, loss = 0.000704
grad_step = 000145, loss = 0.000609
grad_step = 000146, loss = 0.000603
grad_step = 000147, loss = 0.000657
grad_step = 000148, loss = 0.000625
grad_step = 000149, loss = 0.000573
grad_step = 000150, loss = 0.000597
grad_step = 000151, loss = 0.000610
grad_step = 000152, loss = 0.000568
grad_step = 000153, loss = 0.000560
grad_step = 000154, loss = 0.000585
grad_step = 000155, loss = 0.000572
grad_step = 000156, loss = 0.000543
grad_step = 000157, loss = 0.000552
grad_step = 000158, loss = 0.000561
grad_step = 000159, loss = 0.000539
grad_step = 000160, loss = 0.000529
grad_step = 000161, loss = 0.000540
grad_step = 000162, loss = 0.000536
grad_step = 000163, loss = 0.000520
grad_step = 000164, loss = 0.000519
grad_step = 000165, loss = 0.000524
grad_step = 000166, loss = 0.000517
grad_step = 000167, loss = 0.000506
grad_step = 000168, loss = 0.000507
grad_step = 000169, loss = 0.000510
grad_step = 000170, loss = 0.000504
grad_step = 000171, loss = 0.000497
grad_step = 000172, loss = 0.000499
grad_step = 000173, loss = 0.000504
grad_step = 000174, loss = 0.000506
grad_step = 000175, loss = 0.000504
grad_step = 000176, loss = 0.000497
grad_step = 000177, loss = 0.000488
grad_step = 000178, loss = 0.000481
grad_step = 000179, loss = 0.000482
grad_step = 000180, loss = 0.000484
grad_step = 000181, loss = 0.000481
grad_step = 000182, loss = 0.000473
grad_step = 000183, loss = 0.000468
grad_step = 000184, loss = 0.000471
grad_step = 000185, loss = 0.000472
grad_step = 000186, loss = 0.000467
grad_step = 000187, loss = 0.000459
grad_step = 000188, loss = 0.000457
grad_step = 000189, loss = 0.000459
grad_step = 000190, loss = 0.000459
grad_step = 000191, loss = 0.000455
grad_step = 000192, loss = 0.000450
grad_step = 000193, loss = 0.000448
grad_step = 000194, loss = 0.000448
grad_step = 000195, loss = 0.000446
grad_step = 000196, loss = 0.000443
grad_step = 000197, loss = 0.000439
grad_step = 000198, loss = 0.000438
grad_step = 000199, loss = 0.000437
grad_step = 000200, loss = 0.000438
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000437
grad_step = 000202, loss = 0.000438
grad_step = 000203, loss = 0.000440
grad_step = 000204, loss = 0.000448
grad_step = 000205, loss = 0.000459
grad_step = 000206, loss = 0.000474
grad_step = 000207, loss = 0.000479
grad_step = 000208, loss = 0.000469
grad_step = 000209, loss = 0.000445
grad_step = 000210, loss = 0.000428
grad_step = 000211, loss = 0.000425
grad_step = 000212, loss = 0.000428
grad_step = 000213, loss = 0.000431
grad_step = 000214, loss = 0.000434
grad_step = 000215, loss = 0.000437
grad_step = 000216, loss = 0.000434
grad_step = 000217, loss = 0.000423
grad_step = 000218, loss = 0.000409
grad_step = 000219, loss = 0.000402
grad_step = 000220, loss = 0.000406
grad_step = 000221, loss = 0.000413
grad_step = 000222, loss = 0.000415
grad_step = 000223, loss = 0.000412
grad_step = 000224, loss = 0.000408
grad_step = 000225, loss = 0.000406
grad_step = 000226, loss = 0.000405
grad_step = 000227, loss = 0.000402
grad_step = 000228, loss = 0.000396
grad_step = 000229, loss = 0.000389
grad_step = 000230, loss = 0.000386
grad_step = 000231, loss = 0.000387
grad_step = 000232, loss = 0.000391
grad_step = 000233, loss = 0.000394
grad_step = 000234, loss = 0.000397
grad_step = 000235, loss = 0.000401
grad_step = 000236, loss = 0.000407
grad_step = 000237, loss = 0.000417
grad_step = 000238, loss = 0.000424
grad_step = 000239, loss = 0.000429
grad_step = 000240, loss = 0.000421
grad_step = 000241, loss = 0.000404
grad_step = 000242, loss = 0.000384
grad_step = 000243, loss = 0.000372
grad_step = 000244, loss = 0.000374
grad_step = 000245, loss = 0.000385
grad_step = 000246, loss = 0.000399
grad_step = 000247, loss = 0.000404
grad_step = 000248, loss = 0.000399
grad_step = 000249, loss = 0.000384
grad_step = 000250, loss = 0.000368
grad_step = 000251, loss = 0.000359
grad_step = 000252, loss = 0.000360
grad_step = 000253, loss = 0.000367
grad_step = 000254, loss = 0.000374
grad_step = 000255, loss = 0.000378
grad_step = 000256, loss = 0.000375
grad_step = 000257, loss = 0.000367
grad_step = 000258, loss = 0.000358
grad_step = 000259, loss = 0.000352
grad_step = 000260, loss = 0.000351
grad_step = 000261, loss = 0.000356
grad_step = 000262, loss = 0.000369
grad_step = 000263, loss = 0.000389
grad_step = 000264, loss = 0.000412
grad_step = 000265, loss = 0.000416
grad_step = 000266, loss = 0.000385
grad_step = 000267, loss = 0.000354
grad_step = 000268, loss = 0.000363
grad_step = 000269, loss = 0.000394
grad_step = 000270, loss = 0.000397
grad_step = 000271, loss = 0.000371
grad_step = 000272, loss = 0.000354
grad_step = 000273, loss = 0.000366
grad_step = 000274, loss = 0.000370
grad_step = 000275, loss = 0.000352
grad_step = 000276, loss = 0.000333
grad_step = 000277, loss = 0.000338
grad_step = 000278, loss = 0.000351
grad_step = 000279, loss = 0.000346
grad_step = 000280, loss = 0.000334
grad_step = 000281, loss = 0.000335
grad_step = 000282, loss = 0.000346
grad_step = 000283, loss = 0.000350
grad_step = 000284, loss = 0.000344
grad_step = 000285, loss = 0.000340
grad_step = 000286, loss = 0.000347
grad_step = 000287, loss = 0.000354
grad_step = 000288, loss = 0.000355
grad_step = 000289, loss = 0.000346
grad_step = 000290, loss = 0.000343
grad_step = 000291, loss = 0.000344
grad_step = 000292, loss = 0.000344
grad_step = 000293, loss = 0.000336
grad_step = 000294, loss = 0.000327
grad_step = 000295, loss = 0.000323
grad_step = 000296, loss = 0.000324
grad_step = 000297, loss = 0.000323
grad_step = 000298, loss = 0.000319
grad_step = 000299, loss = 0.000313
grad_step = 000300, loss = 0.000311
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000313
grad_step = 000302, loss = 0.000313
grad_step = 000303, loss = 0.000311
grad_step = 000304, loss = 0.000308
grad_step = 000305, loss = 0.000306
grad_step = 000306, loss = 0.000306
grad_step = 000307, loss = 0.000307
grad_step = 000308, loss = 0.000308
grad_step = 000309, loss = 0.000309
grad_step = 000310, loss = 0.000313
grad_step = 000311, loss = 0.000323
grad_step = 000312, loss = 0.000349
grad_step = 000313, loss = 0.000396
grad_step = 000314, loss = 0.000488
grad_step = 000315, loss = 0.000561
grad_step = 000316, loss = 0.000565
grad_step = 000317, loss = 0.000408
grad_step = 000318, loss = 0.000305
grad_step = 000319, loss = 0.000367
grad_step = 000320, loss = 0.000439
grad_step = 000321, loss = 0.000389
grad_step = 000322, loss = 0.000301
grad_step = 000323, loss = 0.000330
grad_step = 000324, loss = 0.000391
grad_step = 000325, loss = 0.000351
grad_step = 000326, loss = 0.000295
grad_step = 000327, loss = 0.000315
grad_step = 000328, loss = 0.000349
grad_step = 000329, loss = 0.000327
grad_step = 000330, loss = 0.000291
grad_step = 000331, loss = 0.000304
grad_step = 000332, loss = 0.000330
grad_step = 000333, loss = 0.000314
grad_step = 000334, loss = 0.000289
grad_step = 000335, loss = 0.000296
grad_step = 000336, loss = 0.000313
grad_step = 000337, loss = 0.000307
grad_step = 000338, loss = 0.000289
grad_step = 000339, loss = 0.000291
grad_step = 000340, loss = 0.000307
grad_step = 000341, loss = 0.000308
grad_step = 000342, loss = 0.000299
grad_step = 000343, loss = 0.000297
grad_step = 000344, loss = 0.000306
grad_step = 000345, loss = 0.000314
grad_step = 000346, loss = 0.000305
grad_step = 000347, loss = 0.000294
grad_step = 000348, loss = 0.000288
grad_step = 000349, loss = 0.000287
grad_step = 000350, loss = 0.000283
grad_step = 000351, loss = 0.000275
grad_step = 000352, loss = 0.000271
grad_step = 000353, loss = 0.000274
grad_step = 000354, loss = 0.000280
grad_step = 000355, loss = 0.000284
grad_step = 000356, loss = 0.000286
grad_step = 000357, loss = 0.000288
grad_step = 000358, loss = 0.000295
grad_step = 000359, loss = 0.000302
grad_step = 000360, loss = 0.000308
grad_step = 000361, loss = 0.000305
grad_step = 000362, loss = 0.000295
grad_step = 000363, loss = 0.000280
grad_step = 000364, loss = 0.000267
grad_step = 000365, loss = 0.000263
grad_step = 000366, loss = 0.000267
grad_step = 000367, loss = 0.000274
grad_step = 000368, loss = 0.000278
grad_step = 000369, loss = 0.000280
grad_step = 000370, loss = 0.000276
grad_step = 000371, loss = 0.000271
grad_step = 000372, loss = 0.000265
grad_step = 000373, loss = 0.000261
grad_step = 000374, loss = 0.000258
grad_step = 000375, loss = 0.000256
grad_step = 000376, loss = 0.000257
grad_step = 000377, loss = 0.000258
grad_step = 000378, loss = 0.000260
grad_step = 000379, loss = 0.000263
grad_step = 000380, loss = 0.000265
grad_step = 000381, loss = 0.000268
grad_step = 000382, loss = 0.000270
grad_step = 000383, loss = 0.000271
grad_step = 000384, loss = 0.000271
grad_step = 000385, loss = 0.000268
grad_step = 000386, loss = 0.000264
grad_step = 000387, loss = 0.000258
grad_step = 000388, loss = 0.000252
grad_step = 000389, loss = 0.000247
grad_step = 000390, loss = 0.000245
grad_step = 000391, loss = 0.000246
grad_step = 000392, loss = 0.000249
grad_step = 000393, loss = 0.000253
grad_step = 000394, loss = 0.000259
grad_step = 000395, loss = 0.000270
grad_step = 000396, loss = 0.000284
grad_step = 000397, loss = 0.000308
grad_step = 000398, loss = 0.000333
grad_step = 000399, loss = 0.000359
grad_step = 000400, loss = 0.000359
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000336
grad_step = 000402, loss = 0.000298
grad_step = 000403, loss = 0.000280
grad_step = 000404, loss = 0.000287
grad_step = 000405, loss = 0.000296
grad_step = 000406, loss = 0.000282
grad_step = 000407, loss = 0.000250
grad_step = 000408, loss = 0.000239
grad_step = 000409, loss = 0.000257
grad_step = 000410, loss = 0.000280
grad_step = 000411, loss = 0.000283
grad_step = 000412, loss = 0.000268
grad_step = 000413, loss = 0.000250
grad_step = 000414, loss = 0.000250
grad_step = 000415, loss = 0.000261
grad_step = 000416, loss = 0.000269
grad_step = 000417, loss = 0.000261
grad_step = 000418, loss = 0.000244
grad_step = 000419, loss = 0.000231
grad_step = 000420, loss = 0.000230
grad_step = 000421, loss = 0.000238
grad_step = 000422, loss = 0.000244
grad_step = 000423, loss = 0.000241
grad_step = 000424, loss = 0.000233
grad_step = 000425, loss = 0.000227
grad_step = 000426, loss = 0.000229
grad_step = 000427, loss = 0.000237
grad_step = 000428, loss = 0.000244
grad_step = 000429, loss = 0.000250
grad_step = 000430, loss = 0.000257
grad_step = 000431, loss = 0.000274
grad_step = 000432, loss = 0.000294
grad_step = 000433, loss = 0.000321
grad_step = 000434, loss = 0.000333
grad_step = 000435, loss = 0.000324
grad_step = 000436, loss = 0.000287
grad_step = 000437, loss = 0.000246
grad_step = 000438, loss = 0.000225
grad_step = 000439, loss = 0.000234
grad_step = 000440, loss = 0.000259
grad_step = 000441, loss = 0.000284
grad_step = 000442, loss = 0.000290
grad_step = 000443, loss = 0.000270
grad_step = 000444, loss = 0.000240
grad_step = 000445, loss = 0.000219
grad_step = 000446, loss = 0.000220
grad_step = 000447, loss = 0.000237
grad_step = 000448, loss = 0.000251
grad_step = 000449, loss = 0.000254
grad_step = 000450, loss = 0.000246
grad_step = 000451, loss = 0.000229
grad_step = 000452, loss = 0.000219
grad_step = 000453, loss = 0.000218
grad_step = 000454, loss = 0.000225
grad_step = 000455, loss = 0.000233
grad_step = 000456, loss = 0.000239
grad_step = 000457, loss = 0.000241
grad_step = 000458, loss = 0.000241
grad_step = 000459, loss = 0.000238
grad_step = 000460, loss = 0.000236
grad_step = 000461, loss = 0.000237
grad_step = 000462, loss = 0.000240
grad_step = 000463, loss = 0.000247
grad_step = 000464, loss = 0.000250
grad_step = 000465, loss = 0.000251
grad_step = 000466, loss = 0.000246
grad_step = 000467, loss = 0.000235
grad_step = 000468, loss = 0.000225
grad_step = 000469, loss = 0.000219
grad_step = 000470, loss = 0.000217
grad_step = 000471, loss = 0.000218
grad_step = 000472, loss = 0.000220
grad_step = 000473, loss = 0.000219
grad_step = 000474, loss = 0.000217
grad_step = 000475, loss = 0.000214
grad_step = 000476, loss = 0.000213
grad_step = 000477, loss = 0.000215
grad_step = 000478, loss = 0.000219
grad_step = 000479, loss = 0.000224
grad_step = 000480, loss = 0.000229
grad_step = 000481, loss = 0.000235
grad_step = 000482, loss = 0.000240
grad_step = 000483, loss = 0.000245
grad_step = 000484, loss = 0.000250
grad_step = 000485, loss = 0.000257
grad_step = 000486, loss = 0.000259
grad_step = 000487, loss = 0.000258
grad_step = 000488, loss = 0.000244
grad_step = 000489, loss = 0.000235
grad_step = 000490, loss = 0.000238
grad_step = 000491, loss = 0.000259
grad_step = 000492, loss = 0.000275
grad_step = 000493, loss = 0.000276
grad_step = 000494, loss = 0.000245
grad_step = 000495, loss = 0.000210
grad_step = 000496, loss = 0.000194
grad_step = 000497, loss = 0.000203
grad_step = 000498, loss = 0.000222
grad_step = 000499, loss = 0.000233
grad_step = 000500, loss = 0.000230
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000216
Finished.

  #### Predict    ############################################# 
   milk_production_pounds
0                     589
1                     561
2                     640
3                     656
4                     727
[[0.60784314]
 [0.57894737]
 [0.66047472]
 [0.67698658]
 [0.750258  ]
 [0.71929825]
 [0.66047472]
 [0.61816305]
 [0.58617131]
 [0.59545924]
 [0.57069143]
 [0.6006192 ]
 [0.61919505]
 [0.58410733]
 [0.67389061]
 [0.69453044]
 [0.76573787]
 [0.73890609]
 [0.68111455]
 [0.63673891]
 [0.60165119]
 [0.60577915]
 [0.58307534]
 [0.61713106]
 [0.64809082]
 [0.6377709 ]
 [0.71001032]
 [0.72755418]
 [0.79463364]
 [0.75954592]
 [0.6996904 ]
 [0.65944272]
 [0.62332301]
 [0.63054696]
 [0.6130031 ]
 [0.65428277]
 [0.67905057]
 [0.64189886]
 [0.73168215]
 [0.74509804]
 [0.80701754]
 [0.78018576]
 [0.7244582 ]
 [0.67389061]
 [0.63467492]
 [0.64086687]
 [0.62125903]
 [0.65531476]
 [0.69865841]
 [0.65531476]
 [0.75954592]
 [0.77915377]
 [0.8369453 ]
 [0.82352941]
 [0.75851393]
 [0.71929825]
 [0.68214654]
 [0.68833849]
 [0.66563467]
 [0.71001032]
 [0.73581011]
 [0.68833849]
 [0.78637771]
 [0.80908153]
 [0.86377709]
 [0.84313725]
 [0.79153767]
 [0.74509804]
 [0.70278638]
 [0.70897833]
 [0.68111455]
 [0.72033024]
 [0.73993808]
 [0.71826625]
 [0.7997936 ]
 [0.82146543]
 [0.88544892]
 [0.85242518]
 [0.80804954]
 [0.76367389]
 [0.72342621]
 [0.72858617]
 [0.69865841]
 [0.73374613]
 [0.75748194]
 [0.7120743 ]
 [0.81011352]
 [0.83075335]
 [0.89886481]
 [0.87203302]
 [0.82662539]
 [0.78844169]
 [0.74819401]
 [0.74613003]
 [0.7120743 ]
 [0.75748194]
 [0.77399381]
 [0.72961816]
 [0.83281734]
 [0.8503612 ]
 [0.91434469]
 [0.88648091]
 [0.84520124]
 [0.80804954]
 [0.76367389]
 [0.77089783]
 [0.73374613]
 [0.7750258 ]
 [0.82972136]
 [0.78018576]
 [0.8875129 ]
 [0.90608875]
 [0.97213622]
 [0.94220846]
 [0.89680083]
 [0.86068111]
 [0.81527348]
 [0.8255934 ]
 [0.7874097 ]
 [0.8255934 ]
 [0.85242518]
 [0.8245614 ]
 [0.91847265]
 [0.92879257]
 [0.99174407]
 [0.96491228]
 [0.92260062]
 [0.88235294]
 [0.83488132]
 [0.83591331]
 [0.79050568]
 [0.83075335]
 [0.84726522]
 [0.79772962]
 [0.91124871]
 [0.92672859]
 [0.9876161 ]
 [0.95356037]
 [0.90918473]
 [0.86377709]
 [0.80908153]
 [0.81630547]
 [0.78431373]
 [0.82765738]
 [0.85448916]
 [0.80288958]
 [0.91744066]
 [0.93085655]
 [1.        ]
 [0.97729618]
 [0.9370485 ]
 [0.89473684]
 [0.84107327]
 [0.8379773 ]
 [0.79772962]
 [0.83900929]
 [0.86068111]
 [0.80701754]
 [0.92053664]
 [0.93188854]
 [0.99690402]
 [0.96697626]
 [0.9246646 ]
 [0.88544892]
 [0.84313725]
 [0.85345717]
 [0.82249742]
 [0.86996904]]
[[0.8516145  0.83946157 0.9415546  0.9356526  0.99791753]
 [0.84028494 0.9003453  0.9579866  1.006251   0.98577106]
 [0.8930714  0.90735763 1.0090588  0.9773592  0.9478042 ]
 [0.9205536  0.9823276  0.984885   0.9432784  0.91163325]
 [0.9897075  0.97636366 0.94770014 0.9079541  0.8621879 ]
 [0.98685515 0.9379855  0.9257083  0.85288936 0.86529773]
 [0.9316939  0.90633035 0.84860885 0.8618233  0.8106867 ]
 [0.8918476  0.83263534 0.8569455  0.8120022  0.8402909 ]
 [0.82169545 0.83034843 0.8117276  0.8398478  0.8449116 ]
 [0.82604253 0.79662764 0.8410375  0.8498556  0.8305466 ]
 [0.8039418  0.8099495  0.8558194  0.83108896 0.9328135 ]
 [0.8247382  0.84834087 0.8192676  0.90198576 0.9426187 ]
 [0.8434043  0.8336434  0.9379634  0.93347824 0.9967755 ]
 [0.84348625 0.911021   0.9607129  1.0038477  0.9783773 ]
 [0.9063666  0.92151326 1.0051599  0.96634585 0.9316722 ]
 [0.9313523  0.990145   0.9743624  0.9283519  0.89102805]
 [0.99329627 0.97694016 0.9304822  0.8902208  0.8366738 ]
 [0.9846313  0.92343056 0.9056078  0.82962084 0.84922874]
 [0.92566407 0.8939625  0.83234614 0.84620816 0.806736  ]
 [0.90374386 0.8366405  0.84390867 0.80373436 0.8402341 ]
 [0.8361968  0.84497494 0.8091717  0.8378833  0.8533115 ]
 [0.84605503 0.81326526 0.8440876  0.8537835  0.8379384 ]
 [0.8186637  0.8235099  0.8632819  0.8347827  0.93701184]
 [0.8335777  0.86020744 0.8232426  0.9059473  0.9457447 ]
 [0.8600327  0.8443227  0.94181645 0.9365329  1.0061936 ]
 [0.8492372  0.9067891  0.96352077 1.0105196  0.99740463]
 [0.9016398  0.9155002  1.0169711  0.9875458  0.9604687 ]
 [0.9326631  0.9939734  0.9974073  0.9528973  0.9210909 ]
 [0.99927366 0.9896117  0.9576786  0.91791606 0.86770064]
 [0.9997067  0.95114976 0.9370216  0.8592022  0.87039536]
 [0.9397038  0.91241413 0.85500306 0.86591804 0.8194556 ]]

  #### Plot     ############################################### 
Saved image to ztest/model_tch/nbeats//n_beats_test.png.

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
[master 9b0107c] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   4044870..9b0107c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 13777169.84B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 295787.24B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  0%|          | 1901568/440473133 [00:00<00:23, 19013930.16B/s]  2%|▏         | 7657472/440473133 [00:00<00:18, 23793770.65B/s]  3%|▎         | 13719552/440473133 [00:00<00:14, 29096037.65B/s]  5%|▍         | 20039680/440473133 [00:00<00:12, 34716107.53B/s]  6%|▌         | 26289152/440473133 [00:00<00:10, 40054363.13B/s]  7%|▋         | 32647168/440473133 [00:00<00:09, 45054939.98B/s]  9%|▉         | 39045120/440473133 [00:00<00:08, 49438752.42B/s] 10%|█         | 45285376/440473133 [00:00<00:07, 52724115.11B/s] 12%|█▏        | 51614720/440473133 [00:00<00:07, 55504562.80B/s] 13%|█▎        | 57882624/440473133 [00:01<00:06, 57476824.82B/s] 15%|█▍        | 64217088/440473133 [00:01<00:06, 59112436.29B/s] 16%|█▌        | 70493184/440473133 [00:01<00:06, 60161347.90B/s] 17%|█▋        | 76672000/440473133 [00:01<00:06, 60591641.42B/s] 19%|█▉        | 83046400/440473133 [00:01<00:05, 61503922.80B/s] 20%|██        | 89456640/440473133 [00:01<00:05, 62259101.47B/s] 22%|██▏       | 95873024/440473133 [00:01<00:05, 62815744.67B/s] 23%|██▎       | 102381568/440473133 [00:01<00:05, 63478545.24B/s] 25%|██▍       | 108762112/440473133 [00:01<00:05, 63004446.95B/s] 26%|██▌       | 115183616/440473133 [00:01<00:05, 63362152.03B/s] 28%|██▊       | 121692160/440473133 [00:02<00:04, 63867346.00B/s] 29%|██▉       | 128124928/440473133 [00:02<00:04, 64002493.73B/s] 31%|███       | 134565888/440473133 [00:02<00:04, 64123097.94B/s] 32%|███▏      | 140985344/440473133 [00:02<00:04, 63984347.71B/s] 33%|███▎      | 147436544/440473133 [00:02<00:04, 64139882.97B/s] 35%|███▍      | 153853952/440473133 [00:02<00:04, 64087426.54B/s] 36%|███▋      | 160288768/440473133 [00:02<00:04, 64165076.64B/s] 38%|███▊      | 166753280/440473133 [00:02<00:04, 64307076.51B/s] 39%|███▉      | 173186048/440473133 [00:02<00:04, 63338254.21B/s] 41%|████      | 179546112/440473133 [00:02<00:04, 63415629.75B/s] 42%|████▏     | 186001408/440473133 [00:03<00:03, 63749419.22B/s] 44%|████▎     | 192378880/440473133 [00:03<00:03, 63569157.63B/s] 45%|████▌     | 198758400/440473133 [00:03<00:03, 63631508.35B/s] 47%|████▋     | 205123584/440473133 [00:03<00:03, 63595285.84B/s] 48%|████▊     | 211531776/440473133 [00:03<00:03, 63740471.06B/s] 49%|████▉     | 217977856/440473133 [00:03<00:03, 63954483.36B/s] 51%|█████     | 224374784/440473133 [00:03<00:03, 63803028.69B/s] 52%|█████▏    | 230756352/440473133 [00:03<00:03, 63482098.14B/s] 54%|█████▍    | 237106176/440473133 [00:03<00:03, 62230902.46B/s] 55%|█████▌    | 243336192/440473133 [00:03<00:03, 61359566.93B/s] 57%|█████▋    | 249772032/440473133 [00:04<00:03, 62227709.61B/s] 58%|█████▊    | 256093184/440473133 [00:04<00:02, 62517870.78B/s] 60%|█████▉    | 262584320/440473133 [00:04<00:02, 63214552.10B/s] 61%|██████    | 269003776/440473133 [00:04<00:02, 63505281.38B/s] 63%|██████▎   | 275422208/440473133 [00:04<00:02, 63707000.12B/s] 64%|██████▍   | 281895936/440473133 [00:04<00:02, 64010650.43B/s] 65%|██████▌   | 288386048/440473133 [00:04<00:02, 64274641.33B/s] 67%|██████▋   | 294914048/440473133 [00:04<00:02, 64571447.34B/s] 68%|██████▊   | 301373440/440473133 [00:04<00:02, 64318895.93B/s] 70%|██████▉   | 307807232/440473133 [00:04<00:02, 64308156.92B/s] 71%|███████▏  | 314248192/440473133 [00:05<00:01, 64336827.94B/s] 73%|███████▎  | 320683008/440473133 [00:05<00:01, 64175386.51B/s] 74%|███████▍  | 327188480/440473133 [00:05<00:01, 64429212.76B/s] 76%|███████▌  | 333632512/440473133 [00:05<00:01, 64172160.52B/s] 77%|███████▋  | 340050944/440473133 [00:05<00:01, 64173118.48B/s] 79%|███████▊  | 346497024/440473133 [00:05<00:01, 64257980.04B/s] 80%|████████  | 352923648/440473133 [00:05<00:01, 64047249.01B/s] 82%|████████▏ | 359351296/440473133 [00:05<00:01, 64115498.79B/s] 83%|████████▎ | 365763584/440473133 [00:05<00:01, 64002882.46B/s] 84%|████████▍ | 372164608/440473133 [00:05<00:01, 63526550.62B/s] 86%|████████▌ | 378518528/440473133 [00:06<00:00, 63339749.86B/s] 87%|████████▋ | 384854016/440473133 [00:06<00:00, 63160471.08B/s] 89%|████████▉ | 391283712/440473133 [00:06<00:00, 63497015.29B/s] 90%|█████████ | 397665280/440473133 [00:06<00:00, 63590575.84B/s] 92%|█████████▏| 404025344/440473133 [00:06<00:00, 62651723.87B/s] 93%|█████████▎| 410363904/440473133 [00:06<00:00, 62866858.19B/s] 95%|█████████▍| 416855040/440473133 [00:06<00:00, 63464621.62B/s] 96%|█████████▌| 423205888/440473133 [00:06<00:00, 63476157.25B/s] 98%|█████████▊| 429560832/440473133 [00:06<00:00, 63496690.32B/s] 99%|█████████▉| 435912704/440473133 [00:06<00:00, 63425396.97B/s]100%|██████████| 440473133/440473133 [00:07<00:00, 62865217.14B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:  21%|██        | 449/2118 [00:00<00:00, 4485.06it/s]Processing text_left with encode:  44%|████▍     | 930/2118 [00:00<00:00, 4576.93it/s]Processing text_left with encode:  64%|██████▍   | 1352/2118 [00:00<00:00, 4463.06it/s]Processing text_left with encode:  86%|████████▌ | 1818/2118 [00:00<00:00, 4519.97it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 4529.13it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 174/18841 [00:00<00:10, 1724.58it/s]Processing text_right with encode:   2%|▏         | 356/18841 [00:00<00:10, 1749.24it/s]Processing text_right with encode:   3%|▎         | 534/18841 [00:00<00:10, 1755.55it/s]Processing text_right with encode:   4%|▍         | 711/18841 [00:00<00:10, 1759.06it/s]Processing text_right with encode:   5%|▍         | 885/18841 [00:00<00:10, 1752.19it/s]Processing text_right with encode:   6%|▌         | 1061/18841 [00:00<00:10, 1753.45it/s]Processing text_right with encode:   7%|▋         | 1225/18841 [00:00<00:10, 1712.17it/s]Processing text_right with encode:   7%|▋         | 1393/18841 [00:00<00:10, 1698.13it/s]Processing text_right with encode:   8%|▊         | 1568/18841 [00:00<00:10, 1712.11it/s]Processing text_right with encode:   9%|▉         | 1746/18841 [00:01<00:09, 1728.69it/s]Processing text_right with encode:  10%|█         | 1915/18841 [00:01<00:09, 1700.80it/s]Processing text_right with encode:  11%|█         | 2082/18841 [00:01<00:10, 1673.49it/s]Processing text_right with encode:  12%|█▏        | 2262/18841 [00:01<00:09, 1707.57it/s]Processing text_right with encode:  13%|█▎        | 2438/18841 [00:01<00:09, 1721.87it/s]Processing text_right with encode:  14%|█▍        | 2611/18841 [00:01<00:09, 1722.74it/s]Processing text_right with encode:  15%|█▍        | 2806/18841 [00:01<00:08, 1782.40it/s]Processing text_right with encode:  16%|█▌        | 2985/18841 [00:01<00:09, 1748.85it/s]Processing text_right with encode:  17%|█▋        | 3161/18841 [00:01<00:09, 1683.51it/s]Processing text_right with encode:  18%|█▊        | 3331/18841 [00:01<00:09, 1680.77it/s]Processing text_right with encode:  19%|█▊        | 3500/18841 [00:02<00:09, 1658.90it/s]Processing text_right with encode:  19%|█▉        | 3667/18841 [00:02<00:09, 1651.11it/s]Processing text_right with encode:  20%|██        | 3834/18841 [00:02<00:09, 1655.37it/s]Processing text_right with encode:  21%|██▏       | 4018/18841 [00:02<00:08, 1706.05it/s]Processing text_right with encode:  22%|██▏       | 4190/18841 [00:02<00:08, 1679.23it/s]Processing text_right with encode:  23%|██▎       | 4360/18841 [00:02<00:08, 1685.36it/s]Processing text_right with encode:  24%|██▍       | 4531/18841 [00:02<00:08, 1691.91it/s]Processing text_right with encode:  25%|██▍       | 4701/18841 [00:02<00:08, 1692.89it/s]Processing text_right with encode:  26%|██▌       | 4877/18841 [00:02<00:08, 1711.96it/s]Processing text_right with encode:  27%|██▋       | 5062/18841 [00:02<00:07, 1748.37it/s]Processing text_right with encode:  28%|██▊       | 5256/18841 [00:03<00:07, 1801.40it/s]Processing text_right with encode:  29%|██▉       | 5437/18841 [00:03<00:07, 1738.73it/s]Processing text_right with encode:  30%|██▉       | 5612/18841 [00:03<00:07, 1682.36it/s]Processing text_right with encode:  31%|███       | 5789/18841 [00:03<00:07, 1705.32it/s]Processing text_right with encode:  32%|███▏      | 5967/18841 [00:03<00:07, 1724.78it/s]Processing text_right with encode:  33%|███▎      | 6141/18841 [00:03<00:07, 1707.70it/s]Processing text_right with encode:  34%|███▎      | 6313/18841 [00:03<00:07, 1637.68it/s]Processing text_right with encode:  34%|███▍      | 6484/18841 [00:03<00:07, 1658.03it/s]Processing text_right with encode:  35%|███▌      | 6665/18841 [00:03<00:07, 1699.92it/s]Processing text_right with encode:  36%|███▋      | 6836/18841 [00:03<00:07, 1683.29it/s]Processing text_right with encode:  37%|███▋      | 7005/18841 [00:04<00:07, 1669.86it/s]Processing text_right with encode:  38%|███▊      | 7178/18841 [00:04<00:06, 1686.11it/s]Processing text_right with encode:  39%|███▉      | 7371/18841 [00:04<00:06, 1752.04it/s]Processing text_right with encode:  40%|████      | 7548/18841 [00:04<00:06, 1725.85it/s]Processing text_right with encode:  41%|████      | 7722/18841 [00:04<00:06, 1713.21it/s]Processing text_right with encode:  42%|████▏     | 7894/18841 [00:04<00:06, 1702.68it/s]Processing text_right with encode:  43%|████▎     | 8065/18841 [00:04<00:06, 1687.45it/s]Processing text_right with encode:  44%|████▎     | 8235/18841 [00:04<00:06, 1664.48it/s]Processing text_right with encode:  45%|████▍     | 8409/18841 [00:04<00:06, 1686.19it/s]Processing text_right with encode:  46%|████▌     | 8578/18841 [00:05<00:06, 1679.02it/s]Processing text_right with encode:  46%|████▋     | 8755/18841 [00:05<00:05, 1703.82it/s]Processing text_right with encode:  47%|████▋     | 8930/18841 [00:05<00:05, 1711.12it/s]Processing text_right with encode:  48%|████▊     | 9102/18841 [00:05<00:05, 1660.38it/s]Processing text_right with encode:  49%|████▉     | 9273/18841 [00:05<00:05, 1673.44it/s]Processing text_right with encode:  50%|█████     | 9442/18841 [00:05<00:05, 1676.09it/s]Processing text_right with encode:  51%|█████     | 9621/18841 [00:05<00:05, 1707.51it/s]Processing text_right with encode:  52%|█████▏    | 9793/18841 [00:05<00:05, 1674.35it/s]Processing text_right with encode:  53%|█████▎    | 9978/18841 [00:05<00:05, 1722.72it/s]Processing text_right with encode:  54%|█████▍    | 10156/18841 [00:05<00:04, 1738.30it/s]Processing text_right with encode:  55%|█████▍    | 10331/18841 [00:06<00:04, 1715.05it/s]Processing text_right with encode:  56%|█████▌    | 10522/18841 [00:06<00:04, 1767.00it/s]Processing text_right with encode:  57%|█████▋    | 10700/18841 [00:06<00:04, 1756.87it/s]Processing text_right with encode:  58%|█████▊    | 10877/18841 [00:06<00:04, 1749.21it/s]Processing text_right with encode:  59%|█████▊    | 11053/18841 [00:06<00:04, 1736.71it/s]Processing text_right with encode:  60%|█████▉    | 11227/18841 [00:06<00:04, 1711.40it/s]Processing text_right with encode:  61%|██████    | 11399/18841 [00:06<00:04, 1682.57it/s]Processing text_right with encode:  61%|██████▏   | 11573/18841 [00:06<00:04, 1697.84it/s]Processing text_right with encode:  62%|██████▏   | 11744/18841 [00:06<00:04, 1688.83it/s]Processing text_right with encode:  63%|██████▎   | 11914/18841 [00:06<00:04, 1680.35it/s]Processing text_right with encode:  64%|██████▍   | 12090/18841 [00:07<00:03, 1701.11it/s]Processing text_right with encode:  65%|██████▌   | 12261/18841 [00:07<00:03, 1701.18it/s]Processing text_right with encode:  66%|██████▌   | 12432/18841 [00:07<00:03, 1694.70it/s]Processing text_right with encode:  67%|██████▋   | 12602/18841 [00:07<00:03, 1668.19it/s]Processing text_right with encode:  68%|██████▊   | 12769/18841 [00:07<00:03, 1636.68it/s]Processing text_right with encode:  69%|██████▊   | 12944/18841 [00:07<00:03, 1667.05it/s]Processing text_right with encode:  70%|██████▉   | 13116/18841 [00:07<00:03, 1681.25it/s]Processing text_right with encode:  71%|███████   | 13296/18841 [00:07<00:03, 1712.56it/s]Processing text_right with encode:  72%|███████▏  | 13472/18841 [00:07<00:03, 1725.84it/s]Processing text_right with encode:  72%|███████▏  | 13656/18841 [00:07<00:02, 1758.50it/s]Processing text_right with encode:  73%|███████▎  | 13833/18841 [00:08<00:02, 1739.58it/s]Processing text_right with encode:  74%|███████▍  | 14010/18841 [00:08<00:02, 1746.83it/s]Processing text_right with encode:  75%|███████▌  | 14185/18841 [00:08<00:02, 1710.24it/s]Processing text_right with encode:  76%|███████▌  | 14360/18841 [00:08<00:02, 1721.23it/s]Processing text_right with encode:  77%|███████▋  | 14533/18841 [00:08<00:02, 1670.61it/s]Processing text_right with encode:  78%|███████▊  | 14723/18841 [00:08<00:02, 1731.90it/s]Processing text_right with encode:  79%|███████▉  | 14899/18841 [00:08<00:02, 1739.94it/s]Processing text_right with encode:  80%|████████  | 15077/18841 [00:08<00:02, 1749.26it/s]Processing text_right with encode:  81%|████████  | 15258/18841 [00:08<00:02, 1766.41it/s]Processing text_right with encode:  82%|████████▏ | 15436/18841 [00:09<00:01, 1759.48it/s]Processing text_right with encode:  83%|████████▎ | 15613/18841 [00:09<00:01, 1760.16it/s]Processing text_right with encode:  84%|████████▍ | 15797/18841 [00:09<00:01, 1781.21it/s]Processing text_right with encode:  85%|████████▍ | 15976/18841 [00:09<00:01, 1726.16it/s]Processing text_right with encode:  86%|████████▌ | 16150/18841 [00:09<00:01, 1707.46it/s]Processing text_right with encode:  87%|████████▋ | 16325/18841 [00:09<00:01, 1719.76it/s]Processing text_right with encode:  88%|████████▊ | 16499/18841 [00:09<00:01, 1722.85it/s]Processing text_right with encode:  88%|████████▊ | 16672/18841 [00:09<00:01, 1695.19it/s]Processing text_right with encode:  89%|████████▉ | 16846/18841 [00:09<00:01, 1707.37it/s]Processing text_right with encode:  90%|█████████ | 17017/18841 [00:09<00:01, 1700.78it/s]Processing text_right with encode:  91%|█████████ | 17188/18841 [00:10<00:00, 1696.18it/s]Processing text_right with encode:  92%|█████████▏| 17363/18841 [00:10<00:00, 1708.32it/s]Processing text_right with encode:  93%|█████████▎| 17534/18841 [00:10<00:00, 1687.18it/s]Processing text_right with encode:  94%|█████████▍| 17719/18841 [00:10<00:00, 1731.50it/s]Processing text_right with encode:  95%|█████████▍| 17893/18841 [00:10<00:00, 1715.73it/s]Processing text_right with encode:  96%|█████████▌| 18092/18841 [00:10<00:00, 1784.84it/s]Processing text_right with encode:  97%|█████████▋| 18272/18841 [00:10<00:00, 1744.66it/s]Processing text_right with encode:  98%|█████████▊| 18460/18841 [00:10<00:00, 1780.89it/s]Processing text_right with encode:  99%|█████████▉| 18643/18841 [00:10<00:00, 1792.57it/s]Processing text_right with encode: 100%|█████████▉| 18823/18841 [00:10<00:00, 1756.13it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:10<00:00, 1715.30it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 707062.71it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 805505.08it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  73%|███████▎  | 464/633 [00:00<00:00, 4636.52it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4591.95it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 177/5961 [00:00<00:03, 1762.00it/s]Processing text_right with encode:   6%|▌         | 351/5961 [00:00<00:03, 1753.85it/s]Processing text_right with encode:   9%|▊         | 514/5961 [00:00<00:03, 1714.04it/s]Processing text_right with encode:  12%|█▏        | 691/5961 [00:00<00:03, 1729.50it/s]Processing text_right with encode:  15%|█▍        | 877/5961 [00:00<00:02, 1765.07it/s]Processing text_right with encode:  18%|█▊        | 1052/5961 [00:00<00:02, 1758.41it/s]Processing text_right with encode:  21%|██        | 1232/5961 [00:00<00:02, 1768.96it/s]Processing text_right with encode:  24%|██▍       | 1416/5961 [00:00<00:02, 1787.82it/s]Processing text_right with encode:  27%|██▋       | 1589/5961 [00:00<00:02, 1767.55it/s]Processing text_right with encode:  30%|██▉       | 1759/5961 [00:01<00:02, 1711.83it/s]Processing text_right with encode:  32%|███▏      | 1932/5961 [00:01<00:02, 1715.89it/s]Processing text_right with encode:  35%|███▌      | 2101/5961 [00:01<00:02, 1686.76it/s]Processing text_right with encode:  38%|███▊      | 2278/5961 [00:01<00:02, 1708.11it/s]Processing text_right with encode:  41%|████      | 2448/5961 [00:01<00:02, 1667.57it/s]Processing text_right with encode:  44%|████▍     | 2633/5961 [00:01<00:01, 1717.94it/s]Processing text_right with encode:  48%|████▊     | 2832/5961 [00:01<00:01, 1791.20it/s]Processing text_right with encode:  51%|█████     | 3012/5961 [00:01<00:01, 1791.68it/s]Processing text_right with encode:  54%|█████▍    | 3205/5961 [00:01<00:01, 1829.25it/s]Processing text_right with encode:  57%|█████▋    | 3389/5961 [00:01<00:01, 1762.39it/s]Processing text_right with encode:  60%|█████▉    | 3568/5961 [00:02<00:01, 1770.29it/s]Processing text_right with encode:  63%|██████▎   | 3746/5961 [00:02<00:01, 1753.11it/s]Processing text_right with encode:  66%|██████▌   | 3940/5961 [00:02<00:01, 1804.37it/s]Processing text_right with encode:  69%|██████▉   | 4122/5961 [00:02<00:01, 1787.55it/s]Processing text_right with encode:  72%|███████▏  | 4302/5961 [00:02<00:00, 1761.42it/s]Processing text_right with encode:  75%|███████▌  | 4479/5961 [00:02<00:00, 1740.90it/s]Processing text_right with encode:  78%|███████▊  | 4654/5961 [00:02<00:00, 1730.99it/s]Processing text_right with encode:  81%|████████  | 4828/5961 [00:02<00:00, 1730.21it/s]Processing text_right with encode:  84%|████████▍ | 5002/5961 [00:02<00:00, 1682.99it/s]Processing text_right with encode:  87%|████████▋ | 5175/5961 [00:02<00:00, 1695.50it/s]Processing text_right with encode:  90%|████████▉ | 5347/5961 [00:03<00:00, 1700.37it/s]Processing text_right with encode:  93%|█████████▎| 5519/5961 [00:03<00:00, 1703.94it/s]Processing text_right with encode:  95%|█████████▌| 5690/5961 [00:03<00:00, 1657.27it/s]Processing text_right with encode:  98%|█████████▊| 5867/5961 [00:03<00:00, 1689.04it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1740.19it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 479154.38it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 741620.33it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:43<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:43<?, ?it/s, loss=1.098]Epoch 1/1:   1%|          | 1/102 [00:43<1:13:18, 43.55s/it, loss=1.098]Epoch 1/1:   1%|          | 1/102 [01:23<1:13:18, 43.55s/it, loss=1.098]Epoch 1/1:   1%|          | 1/102 [01:23<1:13:18, 43.55s/it, loss=1.064]Epoch 1/1:   2%|▏         | 2/102 [01:23<1:10:59, 42.59s/it, loss=1.064]Epoch 1/1:   2%|▏         | 2/102 [02:56<1:10:59, 42.59s/it, loss=1.064]Epoch 1/1:   2%|▏         | 2/102 [02:56<1:10:59, 42.59s/it, loss=1.055]Epoch 1/1:   3%|▎         | 3/102 [02:56<1:34:49, 57.47s/it, loss=1.055]Epoch 1/1:   3%|▎         | 3/102 [03:45<1:34:49, 57.47s/it, loss=1.055]Epoch 1/1:   3%|▎         | 3/102 [03:45<1:34:49, 57.47s/it, loss=0.918]Epoch 1/1:   4%|▍         | 4/102 [03:45<1:30:00, 55.11s/it, loss=0.918]Killed

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
   9b0107c..c12b18c  master     -> origin/master
Updating 9b0107c..c12b18c
Fast-forward
 error_list/20200522/list_log_pullrequest_20200522.md | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
