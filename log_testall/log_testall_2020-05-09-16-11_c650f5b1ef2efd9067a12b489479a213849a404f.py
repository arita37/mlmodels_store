
  /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json 

  test_all GITHUB_REPOSITORT GITHUB_SHA 

  Running command test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/c650f5b1ef2efd9067a12b489479a213849a404f', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/refs/heads/dev/', 'repo': 'arita37/mlmodels', 'branch': 'refs/heads/dev', 'sha': 'c650f5b1ef2efd9067a12b489479a213849a404f', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/c650f5b1ef2efd9067a12b489479a213849a404f

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/c650f5b1ef2efd9067a12b489479a213849a404f

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_tf.temporal_fusion_google', 'model_tf.1_lstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_tch.pplm', 'model_tch.transformer_sentence', 'model_tch.nbeats', 'model_tch.textcnn', 'model_tch.mlp', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.torchhub', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.textcnn', 'model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae'] 

  Used ['model_tf.temporal_fusion_google', 'model_tf.1_lstm', 'model_sklearn.model_lightgbm', 'model_sklearn.model_sklearn', 'model_tch.pplm', 'model_tch.transformer_sentence', 'model_tch.nbeats', 'model_tch.textcnn', 'model_tch.mlp', 'model_tch.pytorch_vae', 'model_tch.03_nbeats_dataloader', 'model_tch.torchhub', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_gluon.gluonts_model', 'model_gluon.fb_prophet', 'model_gluon.gluon_automl', 'model_keras.charcnn', 'model_keras.charcnn_zhang', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.01_deepctr', 'model_keras.namentity_crm_bilstm', 'model_keras.nbeats', 'model_keras.textcnn', 'model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.02_cnn', 'model_keras.Autokeras', 'model_keras.armdn', 'model_keras.textvae'] 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   0bc8492..f14646f  master     -> origin/master
Updating 0bc8492..f14646f
Fast-forward
 ...-09_c650f5b1ef2efd9067a12b489479a213849a404f.py | 167 +++++++++++++++++++++
 1 file changed, 167 insertions(+)
 create mode 100644 log_dataloader/log_2020-05-09-16-09_c650f5b1ef2efd9067a12b489479a213849a404f.py
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
[master 0cc058c] ml_store
 1 file changed, 65 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-09-16-11_c650f5b1ef2efd9067a12b489479a213849a404f.py
To github.com:arita37/mlmodels_store.git
   f14646f..0cc058c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py 
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/compat/v2_compat.py:68: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
start

  #### Loading params   ############################################## 

  ############# Data, Params preparation   ################# 

  {'learning_rate': 0.001, 'num_layers': 1, 'size': 6, 'size_layer': 128, 'timestep': 4, 'epoch': 2, 'output_size': 6} {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'} {} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'} 

  #### Loading dataset   ############################################# 
{'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv', 'data_type': 'pandas'}
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/timeseries/GOOG-year.csv
         Date        Open        High  ...       Close   Adj Close   Volume
0  2016-11-02  778.200012  781.650024  ...  768.700012  768.700012  1872400
1  2016-11-03  767.250000  769.950012  ...  762.130005  762.130005  1943200
2  2016-11-04  750.659973  770.359985  ...  762.020020  762.020020  2134800
3  2016-11-07  774.500000  785.190002  ...  782.520020  782.520020  1585100
4  2016-11-08  783.400024  795.632996  ...  790.510010  790.510010  1350800

[5 rows x 7 columns]

  #### Model init  ############################################# 

  #### Model fit   ############################################# 
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

  #### Predict   ##################################################### 
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

  #### metrics   ##################################################### 
{'loss': 0.42204469442367554, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master ebc436a] ml_store
 1 file changed, 106 insertions(+)
To github.com:arita37/mlmodels_store.git
   0cc058c..ebc436a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 1.02817479 -0.50845713  1.7653351   0.77741921  0.61771419 -0.11877117
   0.45015551 -0.19899818  1.86647138  0.8709698 ]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 0.78344054 -0.05118845  0.82458463 -0.72559712  0.9317172  -0.86776868
   3.03085711 -0.13597733 -0.79726979  0.65458015]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.87874071 -0.01923163  0.31965694  0.15001628 -1.46662161  0.46353432
  -0.89868319  0.39788042 -0.99601089  0.3181542 ]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7fafb83f7e10>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7fafb83f7e80> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
  -1.84176326e-01  8.52704062e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
   1.04921829e+00 -1.27046078e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 1.02242019e+00  1.85300949e+00  6.44353666e-01  1.42251373e-01
   1.15080755e+00  5.13505480e-01 -4.59942831e-01  3.72456852e-01
  -1.48489803e-01  3.71670291e-01]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 7.22978007e-01  1.85535621e-01  9.15499268e-01  3.94428030e-01
  -8.49830738e-01  7.25522558e-01 -1.50504326e-01  1.49588477e+00
   6.75453809e-01 -4.38200267e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 1.13545112e+00  8.61623101e-01  4.90616924e-02 -2.08639057e+00
  -1.11469020e+00  3.61801641e-01 -8.06178212e-01  4.25920177e-01
   4.90803971e-02 -5.96086335e-01]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]]
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
[[ 8.88389445e-01  2.82995534e-01  1.79558917e-02  1.08030817e-01
  -8.49671873e-01  2.94176190e-02 -5.03973949e-01 -1.34793129e-01
   1.04921829e+00 -1.27046078e+00]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 1.07258847e+00 -5.86523939e-01 -1.34267579e+00 -1.23685338e+00
   1.24328724e+00  8.75838928e-01 -3.26499498e-01  6.23362177e-01
  -4.34956683e-01  1.11438298e+00]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 6.23688521e-01  1.20660790e+00  9.03999174e-01 -2.82863552e-01
  -1.18913787e+00 -2.66326884e-01  1.42361443e+00  1.06897162e+00
   4.03714310e-02  1.57546791e+00]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 7.73703613e-01  1.27852808e+00 -2.11416392e+00 -4.42229280e-01
   1.06821044e+00  3.23527354e-01 -2.50644065e+00 -1.09991490e-01
   8.54894544e-03 -4.11639163e-01]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 6.21530991e-01 -1.50957268e+00 -1.01932039e-01 -1.08071069e+00
  -1.13742855e+00  7.25474004e-01  7.98063795e-01 -3.91782562e-02
  -2.28754171e-01  7.43356544e-01]]
None

  ############ Save/ Load ############################################ 

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 2f7b017] ml_store
 1 file changed, 296 insertions(+)
To github.com:arita37/mlmodels_store.git
   ebc436a..2f7b017  master -> master





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
{'roc_auc_score': 1.0}

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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f88c281a908> 

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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 027caf0] ml_store
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   2f7b017..027caf0  master -> master





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


   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   027caf0..69b43a9  master     -> origin/master
Updating 027caf0..69b43a9
Fast-forward
 ...-09_c650f5b1ef2efd9067a12b489479a213849a404f.py | 604 +++++++++++++++++++++
 1 file changed, 604 insertions(+)
 create mode 100644 log_pullrequest/log_pr_2020-05-09-16-09_c650f5b1ef2efd9067a12b489479a213849a404f.py
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
