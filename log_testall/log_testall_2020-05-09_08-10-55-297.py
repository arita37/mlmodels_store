  ('/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json',) 
  ('test_all', 'GITHUB_REPOSITORT', 'GITHUB_SHA') 
  ('Running command', 'test_all') 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/42ac725fa1a30b94674c28fe0e2a484d0db5dba3', 'url_branch_file': 'https://github.com/{repo}/blob/{branch}/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': '42ac725fa1a30b94674c28fe0e2a484d0db5dba3', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/42ac725fa1a30b94674c28fe0e2a484d0db5dba3

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/42ac725fa1a30b94674c28fe0e2a484d0db5dba3

 ************************************************************************************************************************
  ('############Check model ################################',) 
  (['model_tf.temporal_fusion_google', 'model_tf.1_lstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_tch.pplm', 'model_tch.transformer_sentence', 'model_tch.nbeats', 'model_tch.textcnn', 'model_tch.mlp', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.torchhub', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.textcnn', 'model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae'],) 
  ('Used', ['model_tf.temporal_fusion_google', 'model_tf.1_lstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_tch.pplm', 'model_tch.transformer_sentence', 'model_tch.nbeats', 'model_tch.textcnn', 'model_tch.mlp', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.torchhub', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.textcnn', 'model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae']) 





 ************************************************************************************************************************
  ('python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py',) 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'
  (' cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;        ',) 
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   9a846b2..ea2ffdf  master     -> origin/master
Updating 9a846b2..ea2ffdf
Fast-forward
 log_dataloader/log_2020-05-09_08-09-49-941.py | 162 ++++++++++++++++++++++++++
 1 file changed, 162 insertions(+)
 create mode 100644 log_dataloader/log_2020-05-09_08-09-49-941.py
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 85be758] ml_store
 1 file changed, 58 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-09_08-10-55-297.py
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   ea2ffdf..85be758  master -> master





 ************************************************************************************************************************
  ('python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py',) 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
start
  ('#### Loading params   ##############################################',) 
  ('############# Data, Params preparation   #################',) 
  ({'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6}, {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}, {}, {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}) 
  ('#### Loading dataset   #############################################',) 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
  ('#### Model init  #############################################',) 
  ('#### Model fit   #############################################',) 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
          0         1         2         3         4         5
0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
  ('#### Predict   #####################################################',) 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas', 'train': 0}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]
          0         1         2         3         4         5
0  0.706562  0.629914  0.682052  0.599302  0.599302  0.153665
1  0.458824  0.320251  0.598101  0.478596  0.478596  0.174523
2  0.083484  0.331101  0.437246  0.476576  0.476576  0.230969
3  0.622851  0.723606  0.854891  0.853206  0.853206  0.069025
4  0.824209  1.000000  1.000000  1.000000  1.000000  0.000000
5  0.745928  0.883387  0.838176  0.904464  0.904464  0.370110
6  1.000000  0.881878  0.467996  0.486496  0.486496  1.000000
7  0.216516  0.077549  0.433808  0.329598  0.329598  0.318466
8  0.195249  0.000000  0.000000  0.000000  0.000000  0.671960
9  0.000000  0.173783  0.369041  0.411721  0.411721  0.304384
  ('#### metrics   #####################################################',) 
{'loss': 0.4738442376255989, 'loss_history': []}
  ('#### Plot   ########################################################',) 
  ('#### Save/Load   ###################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
  (' cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;        ',) 
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 12d74b9] ml_store
 1 file changed, 95 insertions(+)
To github.com:arita37/mlmodels_store.git
   85be758..12d74b9  master -> master





 ************************************************************************************************************************
  ('python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py',) 
  ('#### Loading params   ##############################################',) 
  ('#### Path params   ##########################################',) 
  ('#### Loading dataset   #############################################',) 
  ('#### Model init, fit   #############################################',) 
  ('#### save the trained model  #######################################',) 
  ('#### Predict   #####################################################',) 
[[ 0.85771953  0.09811225 -0.26046606  1.06032751 -1.39003042 -1.71116766
   0.2656424   1.65712464  1.41767401  0.44509671]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.77528533  1.47016034  1.03298378 -0.87000822  0.78655651  0.36919047
  -0.14319575  0.85328219 -0.13971173 -0.22241403]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
   0.96622086  0.4359862  -0.68748739  3.32107876]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]]
  ('#### metrics   #####################################################',) 
{}
  ('#### Plot   ########################################################',) 
  ('#### Save/Load   ###################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f19830d5e80>
  ('#### Module init   ############################################',) 
  (<module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'>,) 
  ('#### Loading params   ##############################################',) 
  ('#### Path params   ##########################################',) 
  ('#### Model init   ############################################',) 
  (<mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f19830d5ef0>,) 
  ('#### Fit   ########################################################',) 
  ('#### Predict   ####################################################',) 
[[ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 1.66752297  1.22372221 -0.4599301  -0.0593679  -0.493857    1.4489894
  -1.18110317 -0.47758085  0.02599999 -0.79079995]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 1.02242019  1.85300949  0.64435367  0.14225137  1.15080755  0.51350548
  -0.45994283  0.37245685 -0.1484898   0.37167029]
 [ 0.78344054 -0.05118845  0.82458463 -0.72559712  0.9317172  -0.86776868
   3.03085711 -0.13597733 -0.79726979  0.65458015]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]]
None
  ('#### Get  metrics   ################################################',) 
  ('#### Save   ########################################################',) 
  ('#### Load   ########################################################',) 
  ('############ Model preparation   ##################################',) 
  ('#### Module init   ############################################',) 
  (<module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'>,) 
  ('#### Loading params   ##############################################',) 
  ('#### Path params   ##########################################',) 
  ('#### Model init   ############################################',) 
  ('############ Model fit   ##########################################',) 
fit success None
  ('############ Prediction############################################',) 
[[ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]]
None
  ('############ Save/ Load ############################################',) 
  (' cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;        ',) 
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master 37938c6] ml_store
 1 file changed, 215 insertions(+)
To github.com:arita37/mlmodels_store.git
   12d74b9..37938c6  master -> master





 ************************************************************************************************************************
  ('python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_sklearn.py',) 
  ('#### Loading params   ##############################################',) 
  ('#### Path params   ##########################################',) 
  ('#### Loading dataset   #############################################',) 
  ('#### Model init, fit   #############################################',) 
  ('#### save the trained model  #######################################',) 
  ('#### Predict   #####################################################',) 
  ('#### metrics   #####################################################',) 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
{'roc_auc_score': 1.0}
  ('#### Plot   ########################################################',) 
  ('#### Save/Load   ###################################################',) 
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
{'roc_auc_score': 0.9615384615384616}
  ('#### Module init   ############################################',) 
  (<module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'>,) 
  ('#### Loading params   ##############################################',) 
  ('#### Path params   ##########################################',) 
  ('#### Model init   ############################################',) 
  (<mlmodels.model_sklearn.model_sklearn.Model object at 0x7f2907d4d6a0>,) 
  ('#### Fit   ########################################################',) 
  ('#### Predict   ####################################################',) 
None
  ('#### Get  metrics   ################################################',) 
{'mode': 'test', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/tabular/titanic_train_preprocessed.csv', 'data_type': 'pandas', 'train': True}
  ('#### Save   ########################################################',) 
  ('#### Load   ########################################################',) 
  ('############ Model preparation   ##################################',) 
  ('#### Module init   ############################################',) 
  (<module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'>,) 
  ('#### Loading params   ##############################################',) 
  ('#### Path params   ##########################################',) 
  ('#### Model init   ############################################',) 
  ('############ Model fit   ##########################################',) 
fit success None
  ('############ Prediction############################################',) 
None
  ('############ Save/ Load ############################################',) 
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
  (' cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;        ',) 
Fetching origin
Already up to date.
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
[master b3b43f4] ml_store
 1 file changed, 77 insertions(+)
To github.com:arita37/mlmodels_store.git
   37938c6..b3b43f4  master -> master





 ************************************************************************************************************************
  ('python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pplm.py',) 
 Generating text ... 
= Prefix of sentence =
<|endoftext|>The potato

 Unperturbed generated text :

<|endoftext|>The potato-shaped, potato-eating insect of modern times (Ophiocordyceps elegans) has a unique ability to adapt quickly to a wide range of environments. It is able to survive in many different environments, including the Arctic, deserts

 Perturbed generated text :

<|endoftext|>The potato bomb is nothing new. It's been on the news a lot since 9/11. In fact, since the bombing in Paris last November, a bomb has been detonated in every major European country in the European Union.

The bomb in Brussels

  (' cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;        ',) 
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   b3b43f4..2985756  master     -> origin/master
Updating b3b43f4..2985756
Fast-forward
 log_pullrequest/log_2020-05-09_08-12-14-153.py | 527 +++++++++++++++++++++++++
 1 file changed, 527 insertions(+)
 create mode 100644 log_pullrequest/log_2020-05-09_08-12-14-153.py
Logs
README.md
README_actions.md
create_error_file.py
create_github_issues.py
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
