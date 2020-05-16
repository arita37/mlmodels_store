
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'd580c5017e28eefaf82dbb63ddf4270e71792c2b', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/d580c5017e28eefaf82dbb63ddf4270e71792c2b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/d580c5017e28eefaf82dbb63ddf4270e71792c2b

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.nbeats', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.Autokeras', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.armdn', 'model_keras.02_cnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.pplm', 'model_tch.textcnn', 'model_tch.mlp'] 

  Used ['model_keras.keras_gan', 'model_keras.textcnn_dataloader', 'model_keras.nbeats', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.Autokeras', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.armdn', 'model_keras.02_cnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.pplm', 'model_tch.textcnn', 'model_tch.mlp'] 





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//keras_gan.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//keras_gan.py", line 31, in <module>
    'AAE' : kg.aae.aae,
AttributeError: module 'mlmodels.model_keras.raw.keras_gan' has no attribute 'aae'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   ce3e033..131a67b  master     -> origin/master
Updating ce3e033..131a67b
Fast-forward
 error_list/20200516/list_log_benchmark_20200516.md |  162 +-
 error_list/20200516/list_log_import_20200516.md    |    2 +-
 error_list/20200516/list_log_jupyter_20200516.md   | 1749 ++++++++++----------
 ...-08_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py |  373 +++++
 ...-12_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py |  612 +++++++
 5 files changed, 1941 insertions(+), 957 deletions(-)
 create mode 100644 log_dataloader/log_2020-05-16-00-08_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
 create mode 100644 log_pullrequest/log_pr_2020-05-16-00-12_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
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
[master e070784] ml_store
 1 file changed, 71 insertions(+)
 create mode 100644 log_testall/log_testall_2020-05-16-00-17_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
To github.com:arita37/mlmodels_store.git
   131a67b..e070784  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn_dataloader.py 

  #### Module init   ############################################ 

  <module 'mlmodels.model_keras.textcnn_dataloader' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn_dataloader.py'> 

  #### Loading params   ############################################## 
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn_dataloader.py", line 275, in <module>
    test_module(model_uri = MODEL_URI, param_pars= param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/models.py", line 257, in test_module
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/textcnn_dataloader.py", line 182, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/json/refactor/textcnn_keras.json'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
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
[master 4b2f67b] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   e070784..4b2f67b  master -> master





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
[master 08b92c8] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   4b2f67b..08b92c8  master -> master





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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-16 00:18:29.084562: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 00:18:29.103461: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-16 00:18:29.103669: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5592eaef5ae0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 00:18:29.103686: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 268
Trainable params: 268
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2499 - val_binary_crossentropy: 0.6928

  #### metrics   #################################################### 
{'MSE': 0.2497023515010805}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
Total params: 268
Trainable params: 268
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 463
Trainable params: 463
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2792 - binary_crossentropy: 0.7539500/500 [==============================] - 1s 2ms/sample - loss: 0.2677 - binary_crossentropy: 0.7300 - val_loss: 0.2617 - val_binary_crossentropy: 0.7178

  #### metrics   #################################################### 
{'MSE': 0.2639696508233217}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 463
Trainable params: 463
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
Total params: 652
Trainable params: 652
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2498 - binary_crossentropy: 0.6923 - val_loss: 0.2502 - val_binary_crossentropy: 0.6935

  #### metrics   #################################################### 
{'MSE': 0.24978476718893275}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
Total params: 652
Trainable params: 652
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 438
Trainable params: 438
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3255 - binary_crossentropy: 0.9180500/500 [==============================] - 1s 3ms/sample - loss: 0.3242 - binary_crossentropy: 0.9043 - val_loss: 0.3164 - val_binary_crossentropy: 0.8698

  #### metrics   #################################################### 
{'MSE': 0.31462176487123916}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 438
Trainable params: 438
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 183
Trainable params: 183
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2631 - binary_crossentropy: 0.7198500/500 [==============================] - 2s 3ms/sample - loss: 0.2539 - binary_crossentropy: 0.7010 - val_loss: 0.2486 - val_binary_crossentropy: 0.6903

  #### metrics   #################################################### 
{'MSE': 0.2502350727313656}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 183
Trainable params: 183
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-16 00:19:51.972670: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:19:51.975195: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:19:51.982087: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 00:19:51.993616: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 00:19:51.995674: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 00:19:51.997588: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:19:51.999526: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2512 - val_binary_crossentropy: 0.6956
2020-05-16 00:19:53.323750: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:19:53.325552: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:19:53.330138: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 00:19:53.338564: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-16 00:19:53.340133: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 00:19:53.341388: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:19:53.342886: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2515405934465864}

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
2020-05-16 00:20:17.734586: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:17.735889: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:17.740251: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 00:20:17.746415: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 00:20:17.747533: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 00:20:17.748633: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:17.749694: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2490 - val_binary_crossentropy: 0.6911
2020-05-16 00:20:19.281030: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:19.282281: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:19.284764: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 00:20:19.290038: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-16 00:20:19.290965: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 00:20:19.291740: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:19.292453: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24855451275585239}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-16 00:20:53.032835: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:53.036944: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:53.050477: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 00:20:53.075341: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 00:20:53.080406: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 00:20:53.084799: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:53.089311: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.7319 - binary_crossentropy: 1.9346 - val_loss: 0.2798 - val_binary_crossentropy: 0.7565
2020-05-16 00:20:55.276283: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:55.281094: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:55.292654: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 00:20:55.319112: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-16 00:20:55.323239: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-16 00:20:55.327585: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-16 00:20:55.330896: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.3374459121022862}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 685
Trainable params: 685
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2834 - binary_crossentropy: 0.7625500/500 [==============================] - 4s 8ms/sample - loss: 0.2608 - binary_crossentropy: 0.7159 - val_loss: 0.2718 - val_binary_crossentropy: 0.7388

  #### metrics   #################################################### 
{'MSE': 0.26574928875454656}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 685
Trainable params: 685
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         2           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
Total params: 215
Trainable params: 215
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.4200 - binary_crossentropy: 6.4785500/500 [==============================] - 5s 9ms/sample - loss: 0.4840 - binary_crossentropy: 7.4657 - val_loss: 0.5060 - val_binary_crossentropy: 7.8050

  #### metrics   #################################################### 
{'MSE': 0.495}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         2           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
Total params: 215
Trainable params: 215
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 1,889
Trainable params: 1,889
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3256 - binary_crossentropy: 0.8637500/500 [==============================] - 5s 11ms/sample - loss: 0.3065 - binary_crossentropy: 0.8190 - val_loss: 0.2749 - val_binary_crossentropy: 0.7463

  #### metrics   #################################################### 
{'MSE': 0.28494596265871724}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 1,889
Trainable params: 1,889
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
regionsequence_sum (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         1           regionsequence_max[0][0]         
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
Total params: 120
Trainable params: 120
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.4901 - binary_crossentropy: 7.5582500/500 [==============================] - 6s 12ms/sample - loss: 0.4901 - binary_crossentropy: 7.5582 - val_loss: 0.5061 - val_binary_crossentropy: 7.8050

  #### metrics   #################################################### 
{'MSE': 0.498}

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
regionsequence_sum (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         1           regionsequence_max[0][0]         
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
Total params: 120
Trainable params: 120
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 1,372
Trainable params: 1,372
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2514 - binary_crossentropy: 0.6960500/500 [==============================] - 7s 13ms/sample - loss: 0.2542 - binary_crossentropy: 0.7016 - val_loss: 0.2505 - val_binary_crossentropy: 0.6941

  #### metrics   #################################################### 
{'MSE': 0.25030361031551507}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 1,372
Trainable params: 1,372
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_11[0][0]                    
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
Total params: 2,967
Trainable params: 2,887
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2799 - binary_crossentropy: 0.7553500/500 [==============================] - 7s 14ms/sample - loss: 0.2795 - binary_crossentropy: 0.9362 - val_loss: 0.2656 - val_binary_crossentropy: 0.9865

  #### metrics   #################################################### 
{'MSE': 0.26923797266909083}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_11[0][0]                    
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
Total params: 2,967
Trainable params: 2,887
Non-trainable params: 80
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'PNN', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'PNN', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_PNN.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_14"
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_190 (Seq (None, 1, 4)         0           weighted_sequence_layer_43[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_191 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_192 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_193 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_119 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
concatenate_83 (Concatenate)    (None, 1, 20)        0           no_mask_119[0][0]                
                                                                 no_mask_119[1][0]                
                                                                 no_mask_119[2][0]                
                                                                 no_mask_119[3][0]                
                                                                 no_mask_119[4][0]                
__________________________________________________________________________________________________
inner_product_layer (InnerProdu (None, 10, 1)        0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
reshape (Reshape)               (None, 20)           0           concatenate_83[0][0]             
__________________________________________________________________________________________________
flatten_31 (Flatten)            (None, 10)           0           inner_product_layer[0][0]        
__________________________________________________________________________________________________
outter_product_layer (OutterPro (None, 10)           160         sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
concatenate_84 (Concatenate)    (None, 40)           0           reshape[0][0]                    
                                                                 flatten_31[0][0]                 
                                                                 outter_product_layer[0][0]       
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_120 (NoMask)            (None, 40)           0           concatenate_84[0][0]             
__________________________________________________________________________________________________
no_mask_121 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
flatten_32 (Flatten)            (None, 40)           0           no_mask_120[0][0]                
__________________________________________________________________________________________________
flatten_33 (Flatten)            (None, 1)            0           no_mask_121[0][0]                
__________________________________________________________________________________________________
no_mask_122 (NoMask)            multiple             0           flatten_32[0][0]                 
                                                                 flatten_33[0][0]                 
__________________________________________________________________________________________________
concatenate_85 (Concatenate)    (None, 41)           0           no_mask_122[0][0]                
                                                                 no_mask_122[1][0]                
__________________________________________________________________________________________________
dnn_19 (DNN)                    (None, 4)            188         concatenate_85[0][0]             
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1)            4           dnn_19[0][0]                     
__________________________________________________________________________________________________
prediction_layer_17 (Prediction (None, 1)            1           dense_12[0][0]                   
==================================================================================================
Total params: 437
Trainable params: 437
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2487 - binary_crossentropy: 0.6906500/500 [==============================] - 6s 13ms/sample - loss: 0.2497 - binary_crossentropy: 0.6926 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.2500088909001035}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_14"
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_43 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_190 (Seq (None, 1, 4)         0           weighted_sequence_layer_43[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_191 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_192 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_193 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
no_mask_119 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
concatenate_83 (Concatenate)    (None, 1, 20)        0           no_mask_119[0][0]                
                                                                 no_mask_119[1][0]                
                                                                 no_mask_119[2][0]                
                                                                 no_mask_119[3][0]                
                                                                 no_mask_119[4][0]                
__________________________________________________________________________________________________
inner_product_layer (InnerProdu (None, 10, 1)        0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
reshape (Reshape)               (None, 20)           0           concatenate_83[0][0]             
__________________________________________________________________________________________________
flatten_31 (Flatten)            (None, 10)           0           inner_product_layer[0][0]        
__________________________________________________________________________________________________
outter_product_layer (OutterPro (None, 10)           160         sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_190[0][0] 
                                                                 sequence_pooling_layer_191[0][0] 
                                                                 sequence_pooling_layer_192[0][0] 
                                                                 sequence_pooling_layer_193[0][0] 
__________________________________________________________________________________________________
concatenate_84 (Concatenate)    (None, 40)           0           reshape[0][0]                    
                                                                 flatten_31[0][0]                 
                                                                 outter_product_layer[0][0]       
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_120 (NoMask)            (None, 40)           0           concatenate_84[0][0]             
__________________________________________________________________________________________________
no_mask_121 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
flatten_32 (Flatten)            (None, 40)           0           no_mask_120[0][0]                
__________________________________________________________________________________________________
flatten_33 (Flatten)            (None, 1)            0           no_mask_121[0][0]                
__________________________________________________________________________________________________
no_mask_122 (NoMask)            multiple             0           flatten_32[0][0]                 
                                                                 flatten_33[0][0]                 
__________________________________________________________________________________________________
concatenate_85 (Concatenate)    (None, 41)           0           no_mask_122[0][0]                
                                                                 no_mask_122[1][0]                
__________________________________________________________________________________________________
dnn_19 (DNN)                    (None, 4)            188         concatenate_85[0][0]             
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 1)            4           dnn_19[0][0]                     
__________________________________________________________________________________________________
prediction_layer_17 (Prediction (None, 1)            1           dense_12[0][0]                   
==================================================================================================
Total params: 437
Trainable params: 437
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'WDL', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'WDL', 'sparse_feature_num': 2, 'dense_feature_num': 0} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_WDL.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_15"
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_194 (Seq (None, 1, 4)         0           weighted_sequence_layer_44[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_195 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_196 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_197 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
weighted_sequence_layer_45 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_198 (Seq (None, 1, 1)         0           weighted_sequence_layer_45[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_199 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_200 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_201 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_87 (Concatenate)    (None, 1, 24)        0           no_mask_125[0][0]                
                                                                 no_mask_125[1][0]                
                                                                 no_mask_125[2][0]                
                                                                 no_mask_125[3][0]                
                                                                 no_mask_125[4][0]                
                                                                 no_mask_125[5][0]                
__________________________________________________________________________________________________
no_mask_123 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_198[0][0] 
                                                                 sequence_pooling_layer_199[0][0] 
                                                                 sequence_pooling_layer_200[0][0] 
                                                                 sequence_pooling_layer_201[0][0] 
__________________________________________________________________________________________________
flatten_34 (Flatten)            (None, 24)           0           concatenate_87[0][0]             
__________________________________________________________________________________________________
concatenate_86 (Concatenate)    (None, 1, 6)         0           no_mask_123[0][0]                
                                                                 no_mask_123[1][0]                
                                                                 no_mask_123[2][0]                
                                                                 no_mask_123[3][0]                
                                                                 no_mask_123[4][0]                
                                                                 no_mask_123[5][0]                
__________________________________________________________________________________________________
dnn_20 (DNN)                    (None, 32)           1856        flatten_34[0][0]                 
__________________________________________________________________________________________________
linear_18 (Linear)              (None, 1, 1)         0           concatenate_86[0][0]             
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 1)            32          dnn_20[0][0]                     
__________________________________________________________________________________________________
no_mask_124 (NoMask)            (None, 1, 1)         0           linear_18[0][0]                  
__________________________________________________________________________________________________
add_32 (Add)                    (None, 1, 1)         0           dense_13[0][0]                   
                                                                 no_mask_124[0][0]                
__________________________________________________________________________________________________
prediction_layer_18 (Prediction (None, 1)            1           add_32[0][0]                     
==================================================================================================
Total params: 2,079
Trainable params: 2,079
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 7s 14ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.24981957108894032}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_15"
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_1 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_44 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_194 (Seq (None, 1, 4)         0           weighted_sequence_layer_44[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_195 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_196 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_197 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
weighted_sequence_layer_45 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_125 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sparse_emb_sparse_feature_1[0][0]
                                                                 sequence_pooling_layer_194[0][0] 
                                                                 sequence_pooling_layer_195[0][0] 
                                                                 sequence_pooling_layer_196[0][0] 
                                                                 sequence_pooling_layer_197[0][0] 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_198 (Seq (None, 1, 1)         0           weighted_sequence_layer_45[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_199 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_200 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_201 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
concatenate_87 (Concatenate)    (None, 1, 24)        0           no_mask_125[0][0]                
                                                                 no_mask_125[1][0]                
                                                                 no_mask_125[2][0]                
                                                                 no_mask_125[3][0]                
                                                                 no_mask_125[4][0]                
                                                                 no_mask_125[5][0]                
__________________________________________________________________________________________________
no_mask_123 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_198[0][0] 
                                                                 sequence_pooling_layer_199[0][0] 
                                                                 sequence_pooling_layer_200[0][0] 
                                                                 sequence_pooling_layer_201[0][0] 
__________________________________________________________________________________________________
flatten_34 (Flatten)            (None, 24)           0           concatenate_87[0][0]             
__________________________________________________________________________________________________
concatenate_86 (Concatenate)    (None, 1, 6)         0           no_mask_123[0][0]                
                                                                 no_mask_123[1][0]                
                                                                 no_mask_123[2][0]                
                                                                 no_mask_123[3][0]                
                                                                 no_mask_123[4][0]                
                                                                 no_mask_123[5][0]                
__________________________________________________________________________________________________
dnn_20 (DNN)                    (None, 32)           1856        flatten_34[0][0]                 
__________________________________________________________________________________________________
linear_18 (Linear)              (None, 1, 1)         0           concatenate_86[0][0]             
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 1)            32          dnn_20[0][0]                     
__________________________________________________________________________________________________
no_mask_124 (NoMask)            (None, 1, 1)         0           linear_18[0][0]                  
__________________________________________________________________________________________________
add_32 (Add)                    (None, 1, 1)         0           dense_13[0][0]                   
                                                                 no_mask_124[0][0]                
__________________________________________________________________________________________________
prediction_layer_18 (Prediction (None, 1)            1           add_32[0][0]                     
==================================================================================================
Total params: 2,079
Trainable params: 2,079
Non-trainable params: 0
__________________________________________________________________________________________________

  #### Loading params   ############################################## 

  #### Path params   ################################################# 

  #### Model params   ################################################ 
{'model_name': 'xDeepFM', 'optimization': 'adam', 'cost': 'mse'} {'dataset_type': 'synthesis', 'sample_size': 8, 'test_size': 0.2, 'dataset_name': 'xDeepFM', 'sparse_feature_num': 1, 'dense_feature_num': 1} {'batch_size': 100, 'epochs': 1, 'validation_split': 0.5} {'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/deepctr/model_xDeepFM.h5'}

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Model: "model_16"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_206 (Seq (None, 1, 4)         0           weighted_sequence_layer_47[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_207 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_208 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_209 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_130 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_206[0][0] 
                                                                 sequence_pooling_layer_207[0][0] 
                                                                 sequence_pooling_layer_208[0][0] 
                                                                 sequence_pooling_layer_209[0][0] 
__________________________________________________________________________________________________
weighted_sequence_layer_48 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_210 (Seq (None, 1, 1)         0           weighted_sequence_layer_48[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_211 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_212 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_213 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_35 (Flatten)            (None, 20)           0           concatenate_90[0][0]             
__________________________________________________________________________________________________
flatten_36 (Flatten)            (None, 1)            0           no_mask_131[0][0]                
__________________________________________________________________________________________________
no_mask_126 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_210[0][0] 
                                                                 sequence_pooling_layer_211[0][0] 
                                                                 sequence_pooling_layer_212[0][0] 
                                                                 sequence_pooling_layer_213[0][0] 
__________________________________________________________________________________________________
no_mask_132 (NoMask)            multiple             0           flatten_35[0][0]                 
                                                                 flatten_36[0][0]                 
__________________________________________________________________________________________________
concatenate_88 (Concatenate)    (None, 1, 5)         0           no_mask_126[0][0]                
                                                                 no_mask_126[1][0]                
                                                                 no_mask_126[2][0]                
                                                                 no_mask_126[3][0]                
                                                                 no_mask_126[4][0]                
__________________________________________________________________________________________________
no_mask_127 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_91 (Concatenate)    (None, 21)           0           no_mask_132[0][0]                
                                                                 no_mask_132[1][0]                
__________________________________________________________________________________________________
linear_19 (Linear)              (None, 1)            1           concatenate_88[0][0]             
                                                                 no_mask_127[0][0]                
__________________________________________________________________________________________________
dnn_21 (DNN)                    (None, 8)            176         concatenate_91[0][0]             
__________________________________________________________________________________________________
no_mask_128 (NoMask)            (None, 1)            0           linear_19[0][0]                  
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 1)            8           dnn_21[0][0]                     
__________________________________________________________________________________________________
add_35 (Add)                    (None, 1)            0           no_mask_128[0][0]                
                                                                 dense_14[0][0]                   
__________________________________________________________________________________________________
prediction_layer_19 (Prediction (None, 1)            1           add_35[0][0]                     
==================================================================================================
Total params: 281
Trainable params: 281
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.5300 - binary_crossentropy: 8.1752500/500 [==============================] - 7s 14ms/sample - loss: 0.5100 - binary_crossentropy: 7.8667 - val_loss: 0.5180 - val_binary_crossentropy: 7.9901

  #### metrics   #################################################### 
{'MSE': 0.514}

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Model: "model_16"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
weighted_seq (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
weighted_seq_seq_length (InputL [(None, 1)]          0                                            
__________________________________________________________________________________________________
weight (InputLayer)             [(None, 3, 1)]       0                                            
__________________________________________________________________________________________________
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_47 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_206 (Seq (None, 1, 4)         0           weighted_sequence_layer_47[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_207 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_sum[0][0]
__________________________________________________________________________________________________
sequence_pooling_layer_208 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_mean[0][0
__________________________________________________________________________________________________
sequence_pooling_layer_209 (Seq (None, 1, 4)         0           sparse_seq_emb_sequence_max[0][0]
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
dense_feature_0 (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
no_mask_130 (NoMask)            (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_206[0][0] 
                                                                 sequence_pooling_layer_207[0][0] 
                                                                 sequence_pooling_layer_208[0][0] 
                                                                 sequence_pooling_layer_209[0][0] 
__________________________________________________________________________________________________
weighted_sequence_layer_48 (Wei (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_90 (Concatenate)    (None, 1, 20)        0           no_mask_130[0][0]                
                                                                 no_mask_130[1][0]                
                                                                 no_mask_130[2][0]                
                                                                 no_mask_130[3][0]                
                                                                 no_mask_130[4][0]                
__________________________________________________________________________________________________
no_mask_131 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer_210 (Seq (None, 1, 1)         0           weighted_sequence_layer_48[0][0] 
                                                                 weighted_seq_seq_length[0][0]    
__________________________________________________________________________________________________
sequence_pooling_layer_211 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_su
__________________________________________________________________________________________________
sequence_pooling_layer_212 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_me
__________________________________________________________________________________________________
sequence_pooling_layer_213 (Seq (None, 1, 1)         0           linear0sparse_seq_emb_sequence_ma
__________________________________________________________________________________________________
flatten_35 (Flatten)            (None, 20)           0           concatenate_90[0][0]             
__________________________________________________________________________________________________
flatten_36 (Flatten)            (None, 1)            0           no_mask_131[0][0]                
__________________________________________________________________________________________________
no_mask_126 (NoMask)            (None, 1, 1)         0           linear0sparse_emb_sparse_feature_
                                                                 sequence_pooling_layer_210[0][0] 
                                                                 sequence_pooling_layer_211[0][0] 
                                                                 sequence_pooling_layer_212[0][0] 
                                                                 sequence_pooling_layer_213[0][0] 
__________________________________________________________________________________________________
no_mask_132 (NoMask)            multiple             0           flatten_35[0][0]                 
                                                                 flatten_36[0][0]                 
__________________________________________________________________________________________________
concatenate_88 (Concatenate)    (None, 1, 5)         0           no_mask_126[0][0]                
                                                                 no_mask_126[1][0]                
                                                                 no_mask_126[2][0]                
                                                                 no_mask_126[3][0]                
                                                                 no_mask_126[4][0]                
__________________________________________________________________________________________________
no_mask_127 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
concatenate_91 (Concatenate)    (None, 21)           0           no_mask_132[0][0]                
                                                                 no_mask_132[1][0]                
__________________________________________________________________________________________________
linear_19 (Linear)              (None, 1)            1           concatenate_88[0][0]             
                                                                 no_mask_127[0][0]                
__________________________________________________________________________________________________
dnn_21 (DNN)                    (None, 8)            176         concatenate_91[0][0]             
__________________________________________________________________________________________________
no_mask_128 (NoMask)            (None, 1)            0           linear_19[0][0]                  
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 1)            8           dnn_21[0][0]                     
__________________________________________________________________________________________________
add_35 (Add)                    (None, 1)            0           no_mask_128[0][0]                
                                                                 dense_14[0][0]                   
__________________________________________________________________________________________________
prediction_layer_19 (Prediction (None, 1)            1           add_35[0][0]                     
==================================================================================================
Total params: 281
Trainable params: 281
Non-trainable params: 0
__________________________________________________________________________________________________

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   08b92c8..4455072  master     -> origin/master
Updating 08b92c8..4455072
Fast-forward
 error_list/20200516/list_log_import_20200516.md    |    2 +-
 .../20200516/list_log_pullrequest_20200516.md      |    2 +-
 error_list/20200516/list_log_testall_20200516.md   |  756 +----
 ...-19_18ac2a1774ee5f44eee229f2e2cad9101f46304e.py |  504 +++
 ...-23_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py | 2057 ++++++++++++
 ...-22_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py | 3513 ++++++++++++++++++++
 6 files changed, 6088 insertions(+), 746 deletions(-)
 create mode 100644 log_dataloader/log_2020-05-16-00-19_18ac2a1774ee5f44eee229f2e2cad9101f46304e.py
 create mode 100644 log_jupyter/log_jupyter_2020-05-16-00-23_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
 create mode 100644 log_test_cli/log_cli_2020-05-16-00-22_d580c5017e28eefaf82dbb63ddf4270e71792c2b.py
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
[master a23adc5] ml_store
 1 file changed, 5674 insertions(+)
To github.com:arita37/mlmodels_store.git
   4455072..a23adc5  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
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
[master 0e07c77] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   a23adc5..0e07c77  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
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
[master 500b36e] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   0e07c77..500b36e  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py", line 12, in <module>
    import autokeras as ak
ModuleNotFoundError: No module named 'autokeras'

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
[master d86cff0] ml_store
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   500b36e..d86cff0  master -> master





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

2020-05-16 00:34:36.371244: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 00:34:36.376073: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-16 00:34:36.376213: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561725c57630 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 00:34:36.376228: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3851
256/354 [====================>.........] - ETA: 3s - loss: 1.2444
354/354 [==============================] - 15s 43ms/step - loss: 1.3132 - val_loss: 2.0631

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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
Warning: Permanently added the RSA host key for IP address '140.82.118.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   d86cff0..faaf2f7  master     -> origin/master
Updating d86cff0..faaf2f7
Fast-forward
 ...-34_c7f60da4b2d69cec32e6753f33fd0c5dcf6078fc.py | 504 +++++++++++++++++++++
 1 file changed, 504 insertions(+)
 create mode 100644 log_dataloader/log_2020-05-16-00-34_c7f60da4b2d69cec32e6753f33fd0c5dcf6078fc.py
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
[master 1bbbc13] ml_store
 1 file changed, 156 insertions(+)
To github.com:arita37/mlmodels_store.git
   faaf2f7..1bbbc13  master -> master





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
[master 4e06714] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   1bbbc13..4e06714  master -> master





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
[master 1126772] ml_store
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   4e06714..1126772  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1753088/17464789 [==>...........................] - ETA: 0s
 7438336/17464789 [===========>..................] - ETA: 0s
14098432/17464789 [=======================>......] - ETA: 0s
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
2020-05-16 00:35:45.525738: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-16 00:35:45.529940: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-16 00:35:45.530085: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5558c88b9770 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-16 00:35:45.530100: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 8.0346 - accuracy: 0.4760
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8276 - accuracy: 0.4895 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7586 - accuracy: 0.4940
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7088 - accuracy: 0.4972
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7157 - accuracy: 0.4968
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6947 - accuracy: 0.4982
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6666 - accuracy: 0.5000
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7143 - accuracy: 0.4969
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6697 - accuracy: 0.4998
11000/25000 [============>.................] - ETA: 4s - loss: 7.6471 - accuracy: 0.5013
12000/25000 [=============>................] - ETA: 4s - loss: 7.6372 - accuracy: 0.5019
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6289 - accuracy: 0.5025
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6414 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 3s - loss: 7.6370 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6580 - accuracy: 0.5006
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6432 - accuracy: 0.5015
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6317 - accuracy: 0.5023
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6343 - accuracy: 0.5021
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6398 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6381 - accuracy: 0.5019
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6478 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
25000/25000 [==============================] - 10s 392us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fca46f56d68>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fca50547ba8> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6053 - accuracy: 0.5040
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6283 - accuracy: 0.5025
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6257 - accuracy: 0.5027 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6781 - accuracy: 0.4992
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7893 - accuracy: 0.4920
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7637 - accuracy: 0.4937
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7521 - accuracy: 0.4944
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7126 - accuracy: 0.4970
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6835 - accuracy: 0.4989
11000/25000 [============>.................] - ETA: 4s - loss: 7.6834 - accuracy: 0.4989
12000/25000 [=============>................] - ETA: 4s - loss: 7.6641 - accuracy: 0.5002
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6643 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6414 - accuracy: 0.5016
15000/25000 [=================>............] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6477 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6453 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6505 - accuracy: 0.5011
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6452 - accuracy: 0.5014
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6798 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6764 - accuracy: 0.4994
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6711 - accuracy: 0.4997
25000/25000 [==============================] - 10s 391us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7280 - accuracy: 0.4960
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6513 - accuracy: 0.5010
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7382 - accuracy: 0.4953 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5976 - accuracy: 0.5045
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6482 - accuracy: 0.5012
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6820 - accuracy: 0.4990
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6907 - accuracy: 0.4984
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6130 - accuracy: 0.5035
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6036 - accuracy: 0.5041
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6176 - accuracy: 0.5032
11000/25000 [============>.................] - ETA: 4s - loss: 7.6137 - accuracy: 0.5035
12000/25000 [=============>................] - ETA: 4s - loss: 7.6142 - accuracy: 0.5034
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5935 - accuracy: 0.5048
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5932 - accuracy: 0.5048
15000/25000 [=================>............] - ETA: 3s - loss: 7.6298 - accuracy: 0.5024
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6398 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6423 - accuracy: 0.5016
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6618 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6712 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6703 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6792 - accuracy: 0.4992
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6720 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6756 - accuracy: 0.4994
25000/25000 [==============================] - 10s 393us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   1126772..5c2e517  master     -> origin/master
Updating 1126772..5c2e517
Fast-forward
 ...-35_be4e81fe281eae9822d779771f5b85f7e37f3171.py | 136 +++++++++++++++++++++
 1 file changed, 136 insertions(+)
 create mode 100644 log_import/log_import_2020-05-16-00-35_be4e81fe281eae9822d779771f5b85f7e37f3171.py
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
[master ae441f7] ml_store
 1 file changed, 323 insertions(+)
To github.com:arita37/mlmodels_store.git
   5c2e517..ae441f7  master -> master





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

13/13 [==============================] - 2s 123ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 10/10

13/13 [==============================] - 0s 5ms/step - loss: nan

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
[master 7aac53e] ml_store
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   ae441f7..7aac53e  master -> master





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
 1638400/11490434 [===>..........................] - ETA: 0s
 6799360/11490434 [================>.............] - ETA: 0s
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

   32/60000 [..............................] - ETA: 8:00 - loss: 2.2699 - categorical_accuracy: 0.2188
   64/60000 [..............................] - ETA: 4:59 - loss: 2.3033 - categorical_accuracy: 0.1250
   96/60000 [..............................] - ETA: 3:57 - loss: 2.3044 - categorical_accuracy: 0.1250
  128/60000 [..............................] - ETA: 3:27 - loss: 2.2829 - categorical_accuracy: 0.1641
  160/60000 [..............................] - ETA: 3:09 - loss: 2.2653 - categorical_accuracy: 0.1750
  192/60000 [..............................] - ETA: 2:56 - loss: 2.2306 - categorical_accuracy: 0.2031
  224/60000 [..............................] - ETA: 2:46 - loss: 2.1948 - categorical_accuracy: 0.2232
  256/60000 [..............................] - ETA: 2:39 - loss: 2.1740 - categorical_accuracy: 0.2305
  288/60000 [..............................] - ETA: 2:33 - loss: 2.1481 - categorical_accuracy: 0.2535
  320/60000 [..............................] - ETA: 2:28 - loss: 2.1410 - categorical_accuracy: 0.2469
  352/60000 [..............................] - ETA: 2:24 - loss: 2.1220 - categorical_accuracy: 0.2585
  384/60000 [..............................] - ETA: 2:22 - loss: 2.0850 - categorical_accuracy: 0.2786
  416/60000 [..............................] - ETA: 2:19 - loss: 2.0315 - categorical_accuracy: 0.3053
  448/60000 [..............................] - ETA: 2:18 - loss: 2.0316 - categorical_accuracy: 0.3058
  480/60000 [..............................] - ETA: 2:16 - loss: 2.0121 - categorical_accuracy: 0.3104
  512/60000 [..............................] - ETA: 2:14 - loss: 1.9637 - categorical_accuracy: 0.3320
  544/60000 [..............................] - ETA: 2:13 - loss: 1.9233 - categorical_accuracy: 0.3456
  576/60000 [..............................] - ETA: 2:11 - loss: 1.8944 - categorical_accuracy: 0.3542
  608/60000 [..............................] - ETA: 2:10 - loss: 1.8646 - categorical_accuracy: 0.3684
  640/60000 [..............................] - ETA: 2:09 - loss: 1.8379 - categorical_accuracy: 0.3750
  672/60000 [..............................] - ETA: 2:08 - loss: 1.8203 - categorical_accuracy: 0.3854
  704/60000 [..............................] - ETA: 2:07 - loss: 1.8017 - categorical_accuracy: 0.3878
  736/60000 [..............................] - ETA: 2:07 - loss: 1.7665 - categorical_accuracy: 0.4022
  768/60000 [..............................] - ETA: 2:06 - loss: 1.7395 - categorical_accuracy: 0.4089
  800/60000 [..............................] - ETA: 2:05 - loss: 1.7120 - categorical_accuracy: 0.4200
  832/60000 [..............................] - ETA: 2:04 - loss: 1.6806 - categorical_accuracy: 0.4315
  864/60000 [..............................] - ETA: 2:04 - loss: 1.6453 - categorical_accuracy: 0.4421
  896/60000 [..............................] - ETA: 2:03 - loss: 1.6266 - categorical_accuracy: 0.4509
  928/60000 [..............................] - ETA: 2:03 - loss: 1.6018 - categorical_accuracy: 0.4591
  960/60000 [..............................] - ETA: 2:02 - loss: 1.5777 - categorical_accuracy: 0.4708
  992/60000 [..............................] - ETA: 2:02 - loss: 1.5498 - categorical_accuracy: 0.4808
 1024/60000 [..............................] - ETA: 2:01 - loss: 1.5308 - categorical_accuracy: 0.4873
 1056/60000 [..............................] - ETA: 2:01 - loss: 1.5162 - categorical_accuracy: 0.4934
 1088/60000 [..............................] - ETA: 2:00 - loss: 1.4947 - categorical_accuracy: 0.4991
 1120/60000 [..............................] - ETA: 2:00 - loss: 1.4657 - categorical_accuracy: 0.5089
 1152/60000 [..............................] - ETA: 1:59 - loss: 1.4557 - categorical_accuracy: 0.5130
 1184/60000 [..............................] - ETA: 1:59 - loss: 1.4292 - categorical_accuracy: 0.5228
 1216/60000 [..............................] - ETA: 1:58 - loss: 1.4093 - categorical_accuracy: 0.5296
 1248/60000 [..............................] - ETA: 1:58 - loss: 1.3970 - categorical_accuracy: 0.5353
 1280/60000 [..............................] - ETA: 1:58 - loss: 1.3756 - categorical_accuracy: 0.5422
 1312/60000 [..............................] - ETA: 1:57 - loss: 1.3606 - categorical_accuracy: 0.5465
 1344/60000 [..............................] - ETA: 1:57 - loss: 1.3471 - categorical_accuracy: 0.5521
 1376/60000 [..............................] - ETA: 1:57 - loss: 1.3328 - categorical_accuracy: 0.5567
 1408/60000 [..............................] - ETA: 1:57 - loss: 1.3157 - categorical_accuracy: 0.5639
 1440/60000 [..............................] - ETA: 1:57 - loss: 1.2962 - categorical_accuracy: 0.5701
 1472/60000 [..............................] - ETA: 1:56 - loss: 1.2766 - categorical_accuracy: 0.5761
 1504/60000 [..............................] - ETA: 1:56 - loss: 1.2640 - categorical_accuracy: 0.5798
 1536/60000 [..............................] - ETA: 1:56 - loss: 1.2551 - categorical_accuracy: 0.5807
 1568/60000 [..............................] - ETA: 1:56 - loss: 1.2426 - categorical_accuracy: 0.5855
 1600/60000 [..............................] - ETA: 1:56 - loss: 1.2330 - categorical_accuracy: 0.5900
 1632/60000 [..............................] - ETA: 1:56 - loss: 1.2192 - categorical_accuracy: 0.5938
 1664/60000 [..............................] - ETA: 1:56 - loss: 1.2020 - categorical_accuracy: 0.6004
 1696/60000 [..............................] - ETA: 1:56 - loss: 1.1873 - categorical_accuracy: 0.6055
 1728/60000 [..............................] - ETA: 1:56 - loss: 1.1737 - categorical_accuracy: 0.6100
 1760/60000 [..............................] - ETA: 1:55 - loss: 1.1621 - categorical_accuracy: 0.6148
 1792/60000 [..............................] - ETA: 1:55 - loss: 1.1548 - categorical_accuracy: 0.6166
 1824/60000 [..............................] - ETA: 1:55 - loss: 1.1472 - categorical_accuracy: 0.6206
 1856/60000 [..............................] - ETA: 1:55 - loss: 1.1393 - categorical_accuracy: 0.6234
 1888/60000 [..............................] - ETA: 1:55 - loss: 1.1315 - categorical_accuracy: 0.6261
 1920/60000 [..............................] - ETA: 1:54 - loss: 1.1217 - categorical_accuracy: 0.6292
 1952/60000 [..............................] - ETA: 1:54 - loss: 1.1121 - categorical_accuracy: 0.6327
 1984/60000 [..............................] - ETA: 1:54 - loss: 1.1024 - categorical_accuracy: 0.6366
 2016/60000 [>.............................] - ETA: 1:54 - loss: 1.0926 - categorical_accuracy: 0.6399
 2048/60000 [>.............................] - ETA: 1:54 - loss: 1.0836 - categorical_accuracy: 0.6431
 2080/60000 [>.............................] - ETA: 1:54 - loss: 1.0713 - categorical_accuracy: 0.6471
 2112/60000 [>.............................] - ETA: 1:54 - loss: 1.0651 - categorical_accuracy: 0.6491
 2144/60000 [>.............................] - ETA: 1:54 - loss: 1.0545 - categorical_accuracy: 0.6525
 2176/60000 [>.............................] - ETA: 1:53 - loss: 1.0428 - categorical_accuracy: 0.6558
 2208/60000 [>.............................] - ETA: 1:53 - loss: 1.0339 - categorical_accuracy: 0.6585
 2240/60000 [>.............................] - ETA: 1:53 - loss: 1.0284 - categorical_accuracy: 0.6603
 2272/60000 [>.............................] - ETA: 1:53 - loss: 1.0172 - categorical_accuracy: 0.6642
 2304/60000 [>.............................] - ETA: 1:53 - loss: 1.0082 - categorical_accuracy: 0.6671
 2336/60000 [>.............................] - ETA: 1:52 - loss: 1.0033 - categorical_accuracy: 0.6691
 2368/60000 [>.............................] - ETA: 1:52 - loss: 0.9958 - categorical_accuracy: 0.6723
 2400/60000 [>.............................] - ETA: 1:52 - loss: 0.9860 - categorical_accuracy: 0.6758
 2432/60000 [>.............................] - ETA: 1:52 - loss: 0.9770 - categorical_accuracy: 0.6789
 2464/60000 [>.............................] - ETA: 1:52 - loss: 0.9689 - categorical_accuracy: 0.6814
 2496/60000 [>.............................] - ETA: 1:52 - loss: 0.9624 - categorical_accuracy: 0.6847
 2528/60000 [>.............................] - ETA: 1:52 - loss: 0.9541 - categorical_accuracy: 0.6875
 2560/60000 [>.............................] - ETA: 1:52 - loss: 0.9464 - categorical_accuracy: 0.6898
 2592/60000 [>.............................] - ETA: 1:51 - loss: 0.9429 - categorical_accuracy: 0.6910
 2624/60000 [>.............................] - ETA: 1:51 - loss: 0.9345 - categorical_accuracy: 0.6936
 2656/60000 [>.............................] - ETA: 1:51 - loss: 0.9262 - categorical_accuracy: 0.6965
 2688/60000 [>.............................] - ETA: 1:51 - loss: 0.9200 - categorical_accuracy: 0.6983
 2720/60000 [>.............................] - ETA: 1:51 - loss: 0.9120 - categorical_accuracy: 0.7011
 2752/60000 [>.............................] - ETA: 1:51 - loss: 0.9088 - categorical_accuracy: 0.7009
 2784/60000 [>.............................] - ETA: 1:51 - loss: 0.9037 - categorical_accuracy: 0.7022
 2816/60000 [>.............................] - ETA: 1:50 - loss: 0.8989 - categorical_accuracy: 0.7038
 2848/60000 [>.............................] - ETA: 1:50 - loss: 0.8958 - categorical_accuracy: 0.7051
 2880/60000 [>.............................] - ETA: 1:50 - loss: 0.8902 - categorical_accuracy: 0.7066
 2912/60000 [>.............................] - ETA: 1:50 - loss: 0.8855 - categorical_accuracy: 0.7078
 2944/60000 [>.............................] - ETA: 1:50 - loss: 0.8790 - categorical_accuracy: 0.7099
 2976/60000 [>.............................] - ETA: 1:50 - loss: 0.8733 - categorical_accuracy: 0.7124
 3008/60000 [>.............................] - ETA: 1:50 - loss: 0.8674 - categorical_accuracy: 0.7144
 3040/60000 [>.............................] - ETA: 1:50 - loss: 0.8621 - categorical_accuracy: 0.7164
 3072/60000 [>.............................] - ETA: 1:50 - loss: 0.8599 - categorical_accuracy: 0.7171
 3104/60000 [>.............................] - ETA: 1:49 - loss: 0.8548 - categorical_accuracy: 0.7184
 3136/60000 [>.............................] - ETA: 1:49 - loss: 0.8518 - categorical_accuracy: 0.7200
 3168/60000 [>.............................] - ETA: 1:49 - loss: 0.8482 - categorical_accuracy: 0.7213
 3200/60000 [>.............................] - ETA: 1:49 - loss: 0.8444 - categorical_accuracy: 0.7225
 3232/60000 [>.............................] - ETA: 1:49 - loss: 0.8408 - categorical_accuracy: 0.7237
 3264/60000 [>.............................] - ETA: 1:49 - loss: 0.8349 - categorical_accuracy: 0.7258
 3296/60000 [>.............................] - ETA: 1:49 - loss: 0.8303 - categorical_accuracy: 0.7275
 3328/60000 [>.............................] - ETA: 1:49 - loss: 0.8252 - categorical_accuracy: 0.7290
 3360/60000 [>.............................] - ETA: 1:49 - loss: 0.8197 - categorical_accuracy: 0.7307
 3392/60000 [>.............................] - ETA: 1:49 - loss: 0.8152 - categorical_accuracy: 0.7323
 3424/60000 [>.............................] - ETA: 1:49 - loss: 0.8101 - categorical_accuracy: 0.7342
 3456/60000 [>.............................] - ETA: 1:49 - loss: 0.8066 - categorical_accuracy: 0.7358
 3488/60000 [>.............................] - ETA: 1:48 - loss: 0.8042 - categorical_accuracy: 0.7371
 3520/60000 [>.............................] - ETA: 1:48 - loss: 0.8010 - categorical_accuracy: 0.7381
 3552/60000 [>.............................] - ETA: 1:48 - loss: 0.7968 - categorical_accuracy: 0.7390
 3584/60000 [>.............................] - ETA: 1:48 - loss: 0.7922 - categorical_accuracy: 0.7411
 3616/60000 [>.............................] - ETA: 1:48 - loss: 0.7882 - categorical_accuracy: 0.7428
 3648/60000 [>.............................] - ETA: 1:48 - loss: 0.7820 - categorical_accuracy: 0.7448
 3680/60000 [>.............................] - ETA: 1:48 - loss: 0.7764 - categorical_accuracy: 0.7465
 3712/60000 [>.............................] - ETA: 1:48 - loss: 0.7719 - categorical_accuracy: 0.7478
 3744/60000 [>.............................] - ETA: 1:48 - loss: 0.7728 - categorical_accuracy: 0.7484
 3776/60000 [>.............................] - ETA: 1:48 - loss: 0.7688 - categorical_accuracy: 0.7495
 3808/60000 [>.............................] - ETA: 1:48 - loss: 0.7647 - categorical_accuracy: 0.7513
 3840/60000 [>.............................] - ETA: 1:47 - loss: 0.7611 - categorical_accuracy: 0.7526
 3872/60000 [>.............................] - ETA: 1:47 - loss: 0.7615 - categorical_accuracy: 0.7531
 3904/60000 [>.............................] - ETA: 1:47 - loss: 0.7580 - categorical_accuracy: 0.7544
 3936/60000 [>.............................] - ETA: 1:47 - loss: 0.7542 - categorical_accuracy: 0.7558
 3968/60000 [>.............................] - ETA: 1:47 - loss: 0.7501 - categorical_accuracy: 0.7573
 4000/60000 [=>............................] - ETA: 1:47 - loss: 0.7464 - categorical_accuracy: 0.7585
 4032/60000 [=>............................] - ETA: 1:47 - loss: 0.7436 - categorical_accuracy: 0.7594
 4064/60000 [=>............................] - ETA: 1:47 - loss: 0.7404 - categorical_accuracy: 0.7603
 4096/60000 [=>............................] - ETA: 1:47 - loss: 0.7386 - categorical_accuracy: 0.7612
 4128/60000 [=>............................] - ETA: 1:47 - loss: 0.7352 - categorical_accuracy: 0.7624
 4160/60000 [=>............................] - ETA: 1:47 - loss: 0.7311 - categorical_accuracy: 0.7637
 4192/60000 [=>............................] - ETA: 1:47 - loss: 0.7276 - categorical_accuracy: 0.7650
 4224/60000 [=>............................] - ETA: 1:47 - loss: 0.7247 - categorical_accuracy: 0.7666
 4256/60000 [=>............................] - ETA: 1:46 - loss: 0.7238 - categorical_accuracy: 0.7672
 4288/60000 [=>............................] - ETA: 1:46 - loss: 0.7225 - categorical_accuracy: 0.7677
 4320/60000 [=>............................] - ETA: 1:46 - loss: 0.7197 - categorical_accuracy: 0.7683
 4352/60000 [=>............................] - ETA: 1:46 - loss: 0.7162 - categorical_accuracy: 0.7695
 4384/60000 [=>............................] - ETA: 1:46 - loss: 0.7135 - categorical_accuracy: 0.7703
 4416/60000 [=>............................] - ETA: 1:46 - loss: 0.7112 - categorical_accuracy: 0.7711
 4448/60000 [=>............................] - ETA: 1:46 - loss: 0.7085 - categorical_accuracy: 0.7720
 4480/60000 [=>............................] - ETA: 1:46 - loss: 0.7052 - categorical_accuracy: 0.7732
 4512/60000 [=>............................] - ETA: 1:46 - loss: 0.7014 - categorical_accuracy: 0.7742
 4544/60000 [=>............................] - ETA: 1:46 - loss: 0.6981 - categorical_accuracy: 0.7751
 4576/60000 [=>............................] - ETA: 1:46 - loss: 0.6951 - categorical_accuracy: 0.7758
 4608/60000 [=>............................] - ETA: 1:46 - loss: 0.6915 - categorical_accuracy: 0.7767
 4640/60000 [=>............................] - ETA: 1:45 - loss: 0.6878 - categorical_accuracy: 0.7778
 4672/60000 [=>............................] - ETA: 1:45 - loss: 0.6865 - categorical_accuracy: 0.7785
 4704/60000 [=>............................] - ETA: 1:45 - loss: 0.6833 - categorical_accuracy: 0.7795
 4736/60000 [=>............................] - ETA: 1:45 - loss: 0.6811 - categorical_accuracy: 0.7800
 4768/60000 [=>............................] - ETA: 1:45 - loss: 0.6784 - categorical_accuracy: 0.7808
 4800/60000 [=>............................] - ETA: 1:45 - loss: 0.6751 - categorical_accuracy: 0.7819
 4832/60000 [=>............................] - ETA: 1:45 - loss: 0.6745 - categorical_accuracy: 0.7825
 4864/60000 [=>............................] - ETA: 1:45 - loss: 0.6723 - categorical_accuracy: 0.7833
 4896/60000 [=>............................] - ETA: 1:45 - loss: 0.6693 - categorical_accuracy: 0.7839
 4928/60000 [=>............................] - ETA: 1:45 - loss: 0.6672 - categorical_accuracy: 0.7847
 4960/60000 [=>............................] - ETA: 1:45 - loss: 0.6643 - categorical_accuracy: 0.7855
 4992/60000 [=>............................] - ETA: 1:45 - loss: 0.6620 - categorical_accuracy: 0.7861
 5024/60000 [=>............................] - ETA: 1:45 - loss: 0.6589 - categorical_accuracy: 0.7870
 5056/60000 [=>............................] - ETA: 1:45 - loss: 0.6565 - categorical_accuracy: 0.7876
 5088/60000 [=>............................] - ETA: 1:45 - loss: 0.6536 - categorical_accuracy: 0.7887
 5120/60000 [=>............................] - ETA: 1:45 - loss: 0.6501 - categorical_accuracy: 0.7898
 5152/60000 [=>............................] - ETA: 1:45 - loss: 0.6488 - categorical_accuracy: 0.7900
 5184/60000 [=>............................] - ETA: 1:45 - loss: 0.6474 - categorical_accuracy: 0.7905
 5216/60000 [=>............................] - ETA: 1:45 - loss: 0.6457 - categorical_accuracy: 0.7908
 5248/60000 [=>............................] - ETA: 1:45 - loss: 0.6432 - categorical_accuracy: 0.7913
 5280/60000 [=>............................] - ETA: 1:44 - loss: 0.6405 - categorical_accuracy: 0.7920
 5312/60000 [=>............................] - ETA: 1:44 - loss: 0.6402 - categorical_accuracy: 0.7924
 5344/60000 [=>............................] - ETA: 1:44 - loss: 0.6380 - categorical_accuracy: 0.7930
 5376/60000 [=>............................] - ETA: 1:44 - loss: 0.6350 - categorical_accuracy: 0.7941
 5408/60000 [=>............................] - ETA: 1:44 - loss: 0.6328 - categorical_accuracy: 0.7946
 5440/60000 [=>............................] - ETA: 1:44 - loss: 0.6304 - categorical_accuracy: 0.7954
 5472/60000 [=>............................] - ETA: 1:44 - loss: 0.6281 - categorical_accuracy: 0.7964
 5504/60000 [=>............................] - ETA: 1:44 - loss: 0.6259 - categorical_accuracy: 0.7972
 5536/60000 [=>............................] - ETA: 1:44 - loss: 0.6236 - categorical_accuracy: 0.7979
 5568/60000 [=>............................] - ETA: 1:44 - loss: 0.6236 - categorical_accuracy: 0.7981
 5600/60000 [=>............................] - ETA: 1:44 - loss: 0.6207 - categorical_accuracy: 0.7991
 5632/60000 [=>............................] - ETA: 1:44 - loss: 0.6184 - categorical_accuracy: 0.7997
 5664/60000 [=>............................] - ETA: 1:44 - loss: 0.6165 - categorical_accuracy: 0.8005
 5696/60000 [=>............................] - ETA: 1:43 - loss: 0.6140 - categorical_accuracy: 0.8014
 5728/60000 [=>............................] - ETA: 1:43 - loss: 0.6110 - categorical_accuracy: 0.8024
 5760/60000 [=>............................] - ETA: 1:43 - loss: 0.6103 - categorical_accuracy: 0.8026
 5792/60000 [=>............................] - ETA: 1:43 - loss: 0.6090 - categorical_accuracy: 0.8032
 5824/60000 [=>............................] - ETA: 1:43 - loss: 0.6064 - categorical_accuracy: 0.8041
 5856/60000 [=>............................] - ETA: 1:43 - loss: 0.6043 - categorical_accuracy: 0.8048
 5888/60000 [=>............................] - ETA: 1:43 - loss: 0.6017 - categorical_accuracy: 0.8057
 5920/60000 [=>............................] - ETA: 1:43 - loss: 0.5990 - categorical_accuracy: 0.8066
 5952/60000 [=>............................] - ETA: 1:43 - loss: 0.5972 - categorical_accuracy: 0.8073
 5984/60000 [=>............................] - ETA: 1:43 - loss: 0.5969 - categorical_accuracy: 0.8075
 6016/60000 [==>...........................] - ETA: 1:43 - loss: 0.5958 - categorical_accuracy: 0.8078
 6048/60000 [==>...........................] - ETA: 1:43 - loss: 0.5946 - categorical_accuracy: 0.8084
 6080/60000 [==>...........................] - ETA: 1:43 - loss: 0.5924 - categorical_accuracy: 0.8092
 6112/60000 [==>...........................] - ETA: 1:43 - loss: 0.5909 - categorical_accuracy: 0.8097
 6144/60000 [==>...........................] - ETA: 1:42 - loss: 0.5889 - categorical_accuracy: 0.8105
 6176/60000 [==>...........................] - ETA: 1:42 - loss: 0.5864 - categorical_accuracy: 0.8114
 6208/60000 [==>...........................] - ETA: 1:42 - loss: 0.5855 - categorical_accuracy: 0.8115
 6240/60000 [==>...........................] - ETA: 1:42 - loss: 0.5840 - categorical_accuracy: 0.8120
 6272/60000 [==>...........................] - ETA: 1:42 - loss: 0.5823 - categorical_accuracy: 0.8125
 6304/60000 [==>...........................] - ETA: 1:42 - loss: 0.5808 - categorical_accuracy: 0.8130
 6336/60000 [==>...........................] - ETA: 1:42 - loss: 0.5789 - categorical_accuracy: 0.8136
 6368/60000 [==>...........................] - ETA: 1:42 - loss: 0.5781 - categorical_accuracy: 0.8141
 6400/60000 [==>...........................] - ETA: 1:42 - loss: 0.5764 - categorical_accuracy: 0.8145
 6432/60000 [==>...........................] - ETA: 1:42 - loss: 0.5756 - categorical_accuracy: 0.8150
 6464/60000 [==>...........................] - ETA: 1:42 - loss: 0.5732 - categorical_accuracy: 0.8159
 6496/60000 [==>...........................] - ETA: 1:42 - loss: 0.5722 - categorical_accuracy: 0.8165
 6528/60000 [==>...........................] - ETA: 1:42 - loss: 0.5698 - categorical_accuracy: 0.8172
 6560/60000 [==>...........................] - ETA: 1:42 - loss: 0.5699 - categorical_accuracy: 0.8177
 6592/60000 [==>...........................] - ETA: 1:41 - loss: 0.5681 - categorical_accuracy: 0.8184
 6624/60000 [==>...........................] - ETA: 1:41 - loss: 0.5673 - categorical_accuracy: 0.8187
 6656/60000 [==>...........................] - ETA: 1:41 - loss: 0.5663 - categorical_accuracy: 0.8193
 6688/60000 [==>...........................] - ETA: 1:41 - loss: 0.5641 - categorical_accuracy: 0.8201
 6720/60000 [==>...........................] - ETA: 1:41 - loss: 0.5622 - categorical_accuracy: 0.8208
 6752/60000 [==>...........................] - ETA: 1:41 - loss: 0.5601 - categorical_accuracy: 0.8214
 6784/60000 [==>...........................] - ETA: 1:41 - loss: 0.5581 - categorical_accuracy: 0.8221
 6816/60000 [==>...........................] - ETA: 1:41 - loss: 0.5562 - categorical_accuracy: 0.8226
 6848/60000 [==>...........................] - ETA: 1:41 - loss: 0.5544 - categorical_accuracy: 0.8233
 6880/60000 [==>...........................] - ETA: 1:41 - loss: 0.5532 - categorical_accuracy: 0.8238
 6912/60000 [==>...........................] - ETA: 1:41 - loss: 0.5518 - categorical_accuracy: 0.8245
 6944/60000 [==>...........................] - ETA: 1:41 - loss: 0.5499 - categorical_accuracy: 0.8252
 6976/60000 [==>...........................] - ETA: 1:40 - loss: 0.5479 - categorical_accuracy: 0.8257
 7008/60000 [==>...........................] - ETA: 1:40 - loss: 0.5486 - categorical_accuracy: 0.8259
 7040/60000 [==>...........................] - ETA: 1:40 - loss: 0.5470 - categorical_accuracy: 0.8264
 7072/60000 [==>...........................] - ETA: 1:40 - loss: 0.5457 - categorical_accuracy: 0.8271
 7104/60000 [==>...........................] - ETA: 1:40 - loss: 0.5456 - categorical_accuracy: 0.8270
 7136/60000 [==>...........................] - ETA: 1:40 - loss: 0.5436 - categorical_accuracy: 0.8278
 7168/60000 [==>...........................] - ETA: 1:40 - loss: 0.5433 - categorical_accuracy: 0.8277
 7200/60000 [==>...........................] - ETA: 1:40 - loss: 0.5423 - categorical_accuracy: 0.8281
 7232/60000 [==>...........................] - ETA: 1:40 - loss: 0.5404 - categorical_accuracy: 0.8287
 7264/60000 [==>...........................] - ETA: 1:40 - loss: 0.5387 - categorical_accuracy: 0.8292
 7296/60000 [==>...........................] - ETA: 1:40 - loss: 0.5370 - categorical_accuracy: 0.8298
 7328/60000 [==>...........................] - ETA: 1:40 - loss: 0.5358 - categorical_accuracy: 0.8300
 7360/60000 [==>...........................] - ETA: 1:40 - loss: 0.5342 - categorical_accuracy: 0.8306
 7392/60000 [==>...........................] - ETA: 1:40 - loss: 0.5326 - categorical_accuracy: 0.8310
 7424/60000 [==>...........................] - ETA: 1:39 - loss: 0.5320 - categorical_accuracy: 0.8312
 7456/60000 [==>...........................] - ETA: 1:39 - loss: 0.5311 - categorical_accuracy: 0.8315
 7488/60000 [==>...........................] - ETA: 1:39 - loss: 0.5306 - categorical_accuracy: 0.8317
 7520/60000 [==>...........................] - ETA: 1:39 - loss: 0.5294 - categorical_accuracy: 0.8322
 7552/60000 [==>...........................] - ETA: 1:39 - loss: 0.5285 - categorical_accuracy: 0.8325
 7584/60000 [==>...........................] - ETA: 1:39 - loss: 0.5274 - categorical_accuracy: 0.8329
 7616/60000 [==>...........................] - ETA: 1:39 - loss: 0.5265 - categorical_accuracy: 0.8335
 7648/60000 [==>...........................] - ETA: 1:39 - loss: 0.5245 - categorical_accuracy: 0.8342
 7680/60000 [==>...........................] - ETA: 1:39 - loss: 0.5228 - categorical_accuracy: 0.8346
 7712/60000 [==>...........................] - ETA: 1:39 - loss: 0.5221 - categorical_accuracy: 0.8349
 7744/60000 [==>...........................] - ETA: 1:39 - loss: 0.5209 - categorical_accuracy: 0.8355
 7776/60000 [==>...........................] - ETA: 1:39 - loss: 0.5201 - categorical_accuracy: 0.8358
 7808/60000 [==>...........................] - ETA: 1:39 - loss: 0.5188 - categorical_accuracy: 0.8362
 7840/60000 [==>...........................] - ETA: 1:39 - loss: 0.5174 - categorical_accuracy: 0.8366
 7872/60000 [==>...........................] - ETA: 1:39 - loss: 0.5158 - categorical_accuracy: 0.8371
 7904/60000 [==>...........................] - ETA: 1:38 - loss: 0.5148 - categorical_accuracy: 0.8376
 7936/60000 [==>...........................] - ETA: 1:38 - loss: 0.5136 - categorical_accuracy: 0.8377
 7968/60000 [==>...........................] - ETA: 1:38 - loss: 0.5132 - categorical_accuracy: 0.8379
 8000/60000 [===>..........................] - ETA: 1:38 - loss: 0.5117 - categorical_accuracy: 0.8384
 8032/60000 [===>..........................] - ETA: 1:38 - loss: 0.5109 - categorical_accuracy: 0.8386
 8064/60000 [===>..........................] - ETA: 1:38 - loss: 0.5122 - categorical_accuracy: 0.8387
 8096/60000 [===>..........................] - ETA: 1:38 - loss: 0.5108 - categorical_accuracy: 0.8391
 8128/60000 [===>..........................] - ETA: 1:38 - loss: 0.5101 - categorical_accuracy: 0.8392
 8160/60000 [===>..........................] - ETA: 1:38 - loss: 0.5084 - categorical_accuracy: 0.8398
 8192/60000 [===>..........................] - ETA: 1:38 - loss: 0.5076 - categorical_accuracy: 0.8401
 8224/60000 [===>..........................] - ETA: 1:38 - loss: 0.5068 - categorical_accuracy: 0.8405
 8256/60000 [===>..........................] - ETA: 1:38 - loss: 0.5057 - categorical_accuracy: 0.8408
 8288/60000 [===>..........................] - ETA: 1:38 - loss: 0.5041 - categorical_accuracy: 0.8413
 8320/60000 [===>..........................] - ETA: 1:38 - loss: 0.5033 - categorical_accuracy: 0.8415
 8352/60000 [===>..........................] - ETA: 1:38 - loss: 0.5022 - categorical_accuracy: 0.8418
 8384/60000 [===>..........................] - ETA: 1:37 - loss: 0.5018 - categorical_accuracy: 0.8420
 8416/60000 [===>..........................] - ETA: 1:37 - loss: 0.5007 - categorical_accuracy: 0.8422
 8448/60000 [===>..........................] - ETA: 1:37 - loss: 0.5006 - categorical_accuracy: 0.8423
 8480/60000 [===>..........................] - ETA: 1:37 - loss: 0.5003 - categorical_accuracy: 0.8426
 8512/60000 [===>..........................] - ETA: 1:37 - loss: 0.4991 - categorical_accuracy: 0.8430
 8544/60000 [===>..........................] - ETA: 1:37 - loss: 0.4976 - categorical_accuracy: 0.8436
 8576/60000 [===>..........................] - ETA: 1:37 - loss: 0.4964 - categorical_accuracy: 0.8441
 8608/60000 [===>..........................] - ETA: 1:37 - loss: 0.4950 - categorical_accuracy: 0.8446
 8640/60000 [===>..........................] - ETA: 1:37 - loss: 0.4935 - categorical_accuracy: 0.8450
 8672/60000 [===>..........................] - ETA: 1:37 - loss: 0.4920 - categorical_accuracy: 0.8456
 8704/60000 [===>..........................] - ETA: 1:37 - loss: 0.4906 - categorical_accuracy: 0.8460
 8736/60000 [===>..........................] - ETA: 1:37 - loss: 0.4908 - categorical_accuracy: 0.8463
 8768/60000 [===>..........................] - ETA: 1:37 - loss: 0.4898 - categorical_accuracy: 0.8466
 8800/60000 [===>..........................] - ETA: 1:37 - loss: 0.4882 - categorical_accuracy: 0.8470
 8832/60000 [===>..........................] - ETA: 1:37 - loss: 0.4869 - categorical_accuracy: 0.8475
 8864/60000 [===>..........................] - ETA: 1:37 - loss: 0.4867 - categorical_accuracy: 0.8478
 8896/60000 [===>..........................] - ETA: 1:37 - loss: 0.4859 - categorical_accuracy: 0.8480
 8928/60000 [===>..........................] - ETA: 1:36 - loss: 0.4847 - categorical_accuracy: 0.8483
 8960/60000 [===>..........................] - ETA: 1:36 - loss: 0.4834 - categorical_accuracy: 0.8487
 8992/60000 [===>..........................] - ETA: 1:36 - loss: 0.4821 - categorical_accuracy: 0.8490
 9024/60000 [===>..........................] - ETA: 1:36 - loss: 0.4819 - categorical_accuracy: 0.8490
 9056/60000 [===>..........................] - ETA: 1:36 - loss: 0.4811 - categorical_accuracy: 0.8492
 9088/60000 [===>..........................] - ETA: 1:36 - loss: 0.4806 - categorical_accuracy: 0.8494
 9120/60000 [===>..........................] - ETA: 1:36 - loss: 0.4803 - categorical_accuracy: 0.8497
 9152/60000 [===>..........................] - ETA: 1:36 - loss: 0.4788 - categorical_accuracy: 0.8502
 9184/60000 [===>..........................] - ETA: 1:36 - loss: 0.4775 - categorical_accuracy: 0.8506
 9216/60000 [===>..........................] - ETA: 1:36 - loss: 0.4777 - categorical_accuracy: 0.8509
 9248/60000 [===>..........................] - ETA: 1:36 - loss: 0.4770 - categorical_accuracy: 0.8510
 9280/60000 [===>..........................] - ETA: 1:36 - loss: 0.4761 - categorical_accuracy: 0.8513
 9312/60000 [===>..........................] - ETA: 1:36 - loss: 0.4758 - categorical_accuracy: 0.8515
 9344/60000 [===>..........................] - ETA: 1:36 - loss: 0.4747 - categorical_accuracy: 0.8518
 9376/60000 [===>..........................] - ETA: 1:36 - loss: 0.4740 - categorical_accuracy: 0.8519
 9408/60000 [===>..........................] - ETA: 1:35 - loss: 0.4730 - categorical_accuracy: 0.8521
 9440/60000 [===>..........................] - ETA: 1:35 - loss: 0.4722 - categorical_accuracy: 0.8524
 9472/60000 [===>..........................] - ETA: 1:35 - loss: 0.4715 - categorical_accuracy: 0.8526
 9504/60000 [===>..........................] - ETA: 1:35 - loss: 0.4712 - categorical_accuracy: 0.8527
 9536/60000 [===>..........................] - ETA: 1:35 - loss: 0.4701 - categorical_accuracy: 0.8530
 9568/60000 [===>..........................] - ETA: 1:35 - loss: 0.4689 - categorical_accuracy: 0.8534
 9600/60000 [===>..........................] - ETA: 1:35 - loss: 0.4675 - categorical_accuracy: 0.8537
 9632/60000 [===>..........................] - ETA: 1:35 - loss: 0.4664 - categorical_accuracy: 0.8540
 9664/60000 [===>..........................] - ETA: 1:35 - loss: 0.4655 - categorical_accuracy: 0.8543
 9696/60000 [===>..........................] - ETA: 1:35 - loss: 0.4641 - categorical_accuracy: 0.8548
 9728/60000 [===>..........................] - ETA: 1:35 - loss: 0.4639 - categorical_accuracy: 0.8551
 9760/60000 [===>..........................] - ETA: 1:35 - loss: 0.4635 - categorical_accuracy: 0.8553
 9792/60000 [===>..........................] - ETA: 1:35 - loss: 0.4627 - categorical_accuracy: 0.8556
 9824/60000 [===>..........................] - ETA: 1:35 - loss: 0.4613 - categorical_accuracy: 0.8561
 9856/60000 [===>..........................] - ETA: 1:35 - loss: 0.4604 - categorical_accuracy: 0.8563
 9888/60000 [===>..........................] - ETA: 1:35 - loss: 0.4592 - categorical_accuracy: 0.8567
 9920/60000 [===>..........................] - ETA: 1:34 - loss: 0.4584 - categorical_accuracy: 0.8570
 9952/60000 [===>..........................] - ETA: 1:34 - loss: 0.4572 - categorical_accuracy: 0.8573
 9984/60000 [===>..........................] - ETA: 1:34 - loss: 0.4566 - categorical_accuracy: 0.8575
10016/60000 [====>.........................] - ETA: 1:34 - loss: 0.4554 - categorical_accuracy: 0.8577
10048/60000 [====>.........................] - ETA: 1:34 - loss: 0.4557 - categorical_accuracy: 0.8577
10080/60000 [====>.........................] - ETA: 1:34 - loss: 0.4549 - categorical_accuracy: 0.8577
10112/60000 [====>.........................] - ETA: 1:34 - loss: 0.4544 - categorical_accuracy: 0.8580
10144/60000 [====>.........................] - ETA: 1:34 - loss: 0.4533 - categorical_accuracy: 0.8582
10176/60000 [====>.........................] - ETA: 1:34 - loss: 0.4519 - categorical_accuracy: 0.8587
10208/60000 [====>.........................] - ETA: 1:34 - loss: 0.4506 - categorical_accuracy: 0.8591
10240/60000 [====>.........................] - ETA: 1:34 - loss: 0.4504 - categorical_accuracy: 0.8592
10272/60000 [====>.........................] - ETA: 1:34 - loss: 0.4505 - categorical_accuracy: 0.8591
10304/60000 [====>.........................] - ETA: 1:34 - loss: 0.4497 - categorical_accuracy: 0.8592
10336/60000 [====>.........................] - ETA: 1:34 - loss: 0.4489 - categorical_accuracy: 0.8593
10368/60000 [====>.........................] - ETA: 1:34 - loss: 0.4490 - categorical_accuracy: 0.8594
10400/60000 [====>.........................] - ETA: 1:34 - loss: 0.4485 - categorical_accuracy: 0.8596
10432/60000 [====>.........................] - ETA: 1:33 - loss: 0.4475 - categorical_accuracy: 0.8600
10464/60000 [====>.........................] - ETA: 1:33 - loss: 0.4467 - categorical_accuracy: 0.8603
10496/60000 [====>.........................] - ETA: 1:33 - loss: 0.4460 - categorical_accuracy: 0.8604
10528/60000 [====>.........................] - ETA: 1:33 - loss: 0.4450 - categorical_accuracy: 0.8608
10560/60000 [====>.........................] - ETA: 1:33 - loss: 0.4438 - categorical_accuracy: 0.8612
10592/60000 [====>.........................] - ETA: 1:33 - loss: 0.4426 - categorical_accuracy: 0.8616
10624/60000 [====>.........................] - ETA: 1:33 - loss: 0.4421 - categorical_accuracy: 0.8617
10656/60000 [====>.........................] - ETA: 1:33 - loss: 0.4413 - categorical_accuracy: 0.8620
10688/60000 [====>.........................] - ETA: 1:33 - loss: 0.4406 - categorical_accuracy: 0.8623
10720/60000 [====>.........................] - ETA: 1:33 - loss: 0.4397 - categorical_accuracy: 0.8625
10752/60000 [====>.........................] - ETA: 1:33 - loss: 0.4394 - categorical_accuracy: 0.8628
10784/60000 [====>.........................] - ETA: 1:33 - loss: 0.4391 - categorical_accuracy: 0.8629
10816/60000 [====>.........................] - ETA: 1:33 - loss: 0.4388 - categorical_accuracy: 0.8631
10848/60000 [====>.........................] - ETA: 1:33 - loss: 0.4383 - categorical_accuracy: 0.8632
10880/60000 [====>.........................] - ETA: 1:33 - loss: 0.4381 - categorical_accuracy: 0.8630
10912/60000 [====>.........................] - ETA: 1:32 - loss: 0.4387 - categorical_accuracy: 0.8630
10944/60000 [====>.........................] - ETA: 1:32 - loss: 0.4383 - categorical_accuracy: 0.8631
10976/60000 [====>.........................] - ETA: 1:32 - loss: 0.4375 - categorical_accuracy: 0.8633
11008/60000 [====>.........................] - ETA: 1:32 - loss: 0.4370 - categorical_accuracy: 0.8636
11040/60000 [====>.........................] - ETA: 1:32 - loss: 0.4365 - categorical_accuracy: 0.8639
11072/60000 [====>.........................] - ETA: 1:32 - loss: 0.4357 - categorical_accuracy: 0.8640
11104/60000 [====>.........................] - ETA: 1:32 - loss: 0.4351 - categorical_accuracy: 0.8642
11136/60000 [====>.........................] - ETA: 1:32 - loss: 0.4343 - categorical_accuracy: 0.8644
11168/60000 [====>.........................] - ETA: 1:32 - loss: 0.4336 - categorical_accuracy: 0.8646
11200/60000 [====>.........................] - ETA: 1:32 - loss: 0.4326 - categorical_accuracy: 0.8649
11232/60000 [====>.........................] - ETA: 1:32 - loss: 0.4323 - categorical_accuracy: 0.8649
11264/60000 [====>.........................] - ETA: 1:32 - loss: 0.4311 - categorical_accuracy: 0.8653
11296/60000 [====>.........................] - ETA: 1:32 - loss: 0.4305 - categorical_accuracy: 0.8655
11328/60000 [====>.........................] - ETA: 1:32 - loss: 0.4305 - categorical_accuracy: 0.8655
11360/60000 [====>.........................] - ETA: 1:32 - loss: 0.4295 - categorical_accuracy: 0.8658
11392/60000 [====>.........................] - ETA: 1:31 - loss: 0.4287 - categorical_accuracy: 0.8661
11424/60000 [====>.........................] - ETA: 1:31 - loss: 0.4283 - categorical_accuracy: 0.8662
11456/60000 [====>.........................] - ETA: 1:31 - loss: 0.4274 - categorical_accuracy: 0.8665
11488/60000 [====>.........................] - ETA: 1:31 - loss: 0.4265 - categorical_accuracy: 0.8668
11520/60000 [====>.........................] - ETA: 1:31 - loss: 0.4258 - categorical_accuracy: 0.8670
11552/60000 [====>.........................] - ETA: 1:31 - loss: 0.4252 - categorical_accuracy: 0.8672
11584/60000 [====>.........................] - ETA: 1:31 - loss: 0.4243 - categorical_accuracy: 0.8675
11616/60000 [====>.........................] - ETA: 1:31 - loss: 0.4240 - categorical_accuracy: 0.8676
11648/60000 [====>.........................] - ETA: 1:31 - loss: 0.4231 - categorical_accuracy: 0.8678
11680/60000 [====>.........................] - ETA: 1:31 - loss: 0.4221 - categorical_accuracy: 0.8682
11712/60000 [====>.........................] - ETA: 1:31 - loss: 0.4211 - categorical_accuracy: 0.8685
11744/60000 [====>.........................] - ETA: 1:31 - loss: 0.4206 - categorical_accuracy: 0.8686
11776/60000 [====>.........................] - ETA: 1:31 - loss: 0.4197 - categorical_accuracy: 0.8689
11808/60000 [====>.........................] - ETA: 1:31 - loss: 0.4187 - categorical_accuracy: 0.8692
11840/60000 [====>.........................] - ETA: 1:31 - loss: 0.4181 - categorical_accuracy: 0.8693
11872/60000 [====>.........................] - ETA: 1:31 - loss: 0.4172 - categorical_accuracy: 0.8695
11904/60000 [====>.........................] - ETA: 1:30 - loss: 0.4168 - categorical_accuracy: 0.8697
11936/60000 [====>.........................] - ETA: 1:30 - loss: 0.4160 - categorical_accuracy: 0.8700
11968/60000 [====>.........................] - ETA: 1:30 - loss: 0.4152 - categorical_accuracy: 0.8702
12000/60000 [=====>........................] - ETA: 1:30 - loss: 0.4146 - categorical_accuracy: 0.8703
12032/60000 [=====>........................] - ETA: 1:30 - loss: 0.4139 - categorical_accuracy: 0.8705
12064/60000 [=====>........................] - ETA: 1:30 - loss: 0.4136 - categorical_accuracy: 0.8705
12096/60000 [=====>........................] - ETA: 1:30 - loss: 0.4130 - categorical_accuracy: 0.8708
12128/60000 [=====>........................] - ETA: 1:30 - loss: 0.4126 - categorical_accuracy: 0.8709
12160/60000 [=====>........................] - ETA: 1:30 - loss: 0.4123 - categorical_accuracy: 0.8711
12192/60000 [=====>........................] - ETA: 1:30 - loss: 0.4117 - categorical_accuracy: 0.8713
12224/60000 [=====>........................] - ETA: 1:30 - loss: 0.4113 - categorical_accuracy: 0.8715
12256/60000 [=====>........................] - ETA: 1:30 - loss: 0.4105 - categorical_accuracy: 0.8718
12288/60000 [=====>........................] - ETA: 1:30 - loss: 0.4111 - categorical_accuracy: 0.8719
12320/60000 [=====>........................] - ETA: 1:30 - loss: 0.4107 - categorical_accuracy: 0.8722
12352/60000 [=====>........................] - ETA: 1:30 - loss: 0.4108 - categorical_accuracy: 0.8721
12384/60000 [=====>........................] - ETA: 1:29 - loss: 0.4098 - categorical_accuracy: 0.8724
12416/60000 [=====>........................] - ETA: 1:29 - loss: 0.4093 - categorical_accuracy: 0.8726
12448/60000 [=====>........................] - ETA: 1:29 - loss: 0.4083 - categorical_accuracy: 0.8729
12480/60000 [=====>........................] - ETA: 1:29 - loss: 0.4080 - categorical_accuracy: 0.8731
12512/60000 [=====>........................] - ETA: 1:29 - loss: 0.4072 - categorical_accuracy: 0.8733
12544/60000 [=====>........................] - ETA: 1:29 - loss: 0.4067 - categorical_accuracy: 0.8735
12576/60000 [=====>........................] - ETA: 1:29 - loss: 0.4061 - categorical_accuracy: 0.8735
12608/60000 [=====>........................] - ETA: 1:29 - loss: 0.4054 - categorical_accuracy: 0.8737
12640/60000 [=====>........................] - ETA: 1:29 - loss: 0.4058 - categorical_accuracy: 0.8738
12672/60000 [=====>........................] - ETA: 1:29 - loss: 0.4056 - categorical_accuracy: 0.8739
12704/60000 [=====>........................] - ETA: 1:29 - loss: 0.4046 - categorical_accuracy: 0.8742
12736/60000 [=====>........................] - ETA: 1:29 - loss: 0.4039 - categorical_accuracy: 0.8745
12768/60000 [=====>........................] - ETA: 1:29 - loss: 0.4030 - categorical_accuracy: 0.8748
12800/60000 [=====>........................] - ETA: 1:29 - loss: 0.4022 - categorical_accuracy: 0.8750
12832/60000 [=====>........................] - ETA: 1:29 - loss: 0.4013 - categorical_accuracy: 0.8753
12864/60000 [=====>........................] - ETA: 1:29 - loss: 0.4010 - categorical_accuracy: 0.8754
12896/60000 [=====>........................] - ETA: 1:29 - loss: 0.4005 - categorical_accuracy: 0.8755
12928/60000 [=====>........................] - ETA: 1:29 - loss: 0.3998 - categorical_accuracy: 0.8758
12960/60000 [=====>........................] - ETA: 1:29 - loss: 0.3995 - categorical_accuracy: 0.8759
12992/60000 [=====>........................] - ETA: 1:28 - loss: 0.3986 - categorical_accuracy: 0.8762
13024/60000 [=====>........................] - ETA: 1:28 - loss: 0.3979 - categorical_accuracy: 0.8765
13056/60000 [=====>........................] - ETA: 1:28 - loss: 0.3976 - categorical_accuracy: 0.8765
13088/60000 [=====>........................] - ETA: 1:28 - loss: 0.3967 - categorical_accuracy: 0.8768
13120/60000 [=====>........................] - ETA: 1:28 - loss: 0.3966 - categorical_accuracy: 0.8769
13152/60000 [=====>........................] - ETA: 1:28 - loss: 0.3958 - categorical_accuracy: 0.8771
13184/60000 [=====>........................] - ETA: 1:28 - loss: 0.3957 - categorical_accuracy: 0.8773
13216/60000 [=====>........................] - ETA: 1:28 - loss: 0.3950 - categorical_accuracy: 0.8775
13248/60000 [=====>........................] - ETA: 1:28 - loss: 0.3946 - categorical_accuracy: 0.8776
13280/60000 [=====>........................] - ETA: 1:28 - loss: 0.3942 - categorical_accuracy: 0.8777
13312/60000 [=====>........................] - ETA: 1:28 - loss: 0.3943 - categorical_accuracy: 0.8779
13344/60000 [=====>........................] - ETA: 1:28 - loss: 0.3938 - categorical_accuracy: 0.8779
13376/60000 [=====>........................] - ETA: 1:28 - loss: 0.3932 - categorical_accuracy: 0.8781
13408/60000 [=====>........................] - ETA: 1:28 - loss: 0.3924 - categorical_accuracy: 0.8784
13440/60000 [=====>........................] - ETA: 1:28 - loss: 0.3921 - categorical_accuracy: 0.8783
13472/60000 [=====>........................] - ETA: 1:28 - loss: 0.3913 - categorical_accuracy: 0.8786
13504/60000 [=====>........................] - ETA: 1:27 - loss: 0.3907 - categorical_accuracy: 0.8789
13536/60000 [=====>........................] - ETA: 1:27 - loss: 0.3900 - categorical_accuracy: 0.8791
13568/60000 [=====>........................] - ETA: 1:27 - loss: 0.3892 - categorical_accuracy: 0.8793
13600/60000 [=====>........................] - ETA: 1:27 - loss: 0.3888 - categorical_accuracy: 0.8794
13632/60000 [=====>........................] - ETA: 1:27 - loss: 0.3886 - categorical_accuracy: 0.8795
13664/60000 [=====>........................] - ETA: 1:27 - loss: 0.3878 - categorical_accuracy: 0.8798
13696/60000 [=====>........................] - ETA: 1:27 - loss: 0.3877 - categorical_accuracy: 0.8798
13728/60000 [=====>........................] - ETA: 1:27 - loss: 0.3870 - categorical_accuracy: 0.8800
13760/60000 [=====>........................] - ETA: 1:27 - loss: 0.3867 - categorical_accuracy: 0.8800
13792/60000 [=====>........................] - ETA: 1:27 - loss: 0.3864 - categorical_accuracy: 0.8800
13824/60000 [=====>........................] - ETA: 1:27 - loss: 0.3867 - categorical_accuracy: 0.8799
13856/60000 [=====>........................] - ETA: 1:27 - loss: 0.3860 - categorical_accuracy: 0.8802
13888/60000 [=====>........................] - ETA: 1:27 - loss: 0.3855 - categorical_accuracy: 0.8803
13920/60000 [=====>........................] - ETA: 1:27 - loss: 0.3854 - categorical_accuracy: 0.8803
13952/60000 [=====>........................] - ETA: 1:26 - loss: 0.3847 - categorical_accuracy: 0.8804
13984/60000 [=====>........................] - ETA: 1:26 - loss: 0.3842 - categorical_accuracy: 0.8805
14016/60000 [======>.......................] - ETA: 1:26 - loss: 0.3834 - categorical_accuracy: 0.8808
14048/60000 [======>.......................] - ETA: 1:26 - loss: 0.3828 - categorical_accuracy: 0.8811
14080/60000 [======>.......................] - ETA: 1:26 - loss: 0.3823 - categorical_accuracy: 0.8812
14112/60000 [======>.......................] - ETA: 1:26 - loss: 0.3820 - categorical_accuracy: 0.8813
14144/60000 [======>.......................] - ETA: 1:26 - loss: 0.3812 - categorical_accuracy: 0.8815
14176/60000 [======>.......................] - ETA: 1:26 - loss: 0.3807 - categorical_accuracy: 0.8816
14208/60000 [======>.......................] - ETA: 1:26 - loss: 0.3803 - categorical_accuracy: 0.8817
14240/60000 [======>.......................] - ETA: 1:26 - loss: 0.3799 - categorical_accuracy: 0.8818
14272/60000 [======>.......................] - ETA: 1:26 - loss: 0.3795 - categorical_accuracy: 0.8819
14304/60000 [======>.......................] - ETA: 1:26 - loss: 0.3792 - categorical_accuracy: 0.8820
14336/60000 [======>.......................] - ETA: 1:26 - loss: 0.3787 - categorical_accuracy: 0.8822
14368/60000 [======>.......................] - ETA: 1:26 - loss: 0.3780 - categorical_accuracy: 0.8824
14400/60000 [======>.......................] - ETA: 1:26 - loss: 0.3775 - categorical_accuracy: 0.8826
14432/60000 [======>.......................] - ETA: 1:26 - loss: 0.3772 - categorical_accuracy: 0.8827
14464/60000 [======>.......................] - ETA: 1:26 - loss: 0.3765 - categorical_accuracy: 0.8829
14496/60000 [======>.......................] - ETA: 1:25 - loss: 0.3765 - categorical_accuracy: 0.8829
14528/60000 [======>.......................] - ETA: 1:25 - loss: 0.3765 - categorical_accuracy: 0.8830
14560/60000 [======>.......................] - ETA: 1:25 - loss: 0.3759 - categorical_accuracy: 0.8832
14592/60000 [======>.......................] - ETA: 1:25 - loss: 0.3754 - categorical_accuracy: 0.8834
14624/60000 [======>.......................] - ETA: 1:25 - loss: 0.3750 - categorical_accuracy: 0.8834
14656/60000 [======>.......................] - ETA: 1:25 - loss: 0.3746 - categorical_accuracy: 0.8836
14688/60000 [======>.......................] - ETA: 1:25 - loss: 0.3744 - categorical_accuracy: 0.8834
14720/60000 [======>.......................] - ETA: 1:25 - loss: 0.3737 - categorical_accuracy: 0.8837
14752/60000 [======>.......................] - ETA: 1:25 - loss: 0.3733 - categorical_accuracy: 0.8839
14784/60000 [======>.......................] - ETA: 1:25 - loss: 0.3730 - categorical_accuracy: 0.8840
14816/60000 [======>.......................] - ETA: 1:25 - loss: 0.3724 - categorical_accuracy: 0.8842
14848/60000 [======>.......................] - ETA: 1:25 - loss: 0.3720 - categorical_accuracy: 0.8843
14880/60000 [======>.......................] - ETA: 1:25 - loss: 0.3715 - categorical_accuracy: 0.8843
14912/60000 [======>.......................] - ETA: 1:25 - loss: 0.3711 - categorical_accuracy: 0.8845
14944/60000 [======>.......................] - ETA: 1:25 - loss: 0.3707 - categorical_accuracy: 0.8847
14976/60000 [======>.......................] - ETA: 1:25 - loss: 0.3703 - categorical_accuracy: 0.8847
15008/60000 [======>.......................] - ETA: 1:24 - loss: 0.3701 - categorical_accuracy: 0.8847
15040/60000 [======>.......................] - ETA: 1:24 - loss: 0.3696 - categorical_accuracy: 0.8848
15072/60000 [======>.......................] - ETA: 1:24 - loss: 0.3691 - categorical_accuracy: 0.8850
15104/60000 [======>.......................] - ETA: 1:24 - loss: 0.3690 - categorical_accuracy: 0.8850
15136/60000 [======>.......................] - ETA: 1:24 - loss: 0.3688 - categorical_accuracy: 0.8851
15168/60000 [======>.......................] - ETA: 1:24 - loss: 0.3686 - categorical_accuracy: 0.8852
15200/60000 [======>.......................] - ETA: 1:24 - loss: 0.3681 - categorical_accuracy: 0.8853
15232/60000 [======>.......................] - ETA: 1:24 - loss: 0.3678 - categorical_accuracy: 0.8854
15264/60000 [======>.......................] - ETA: 1:24 - loss: 0.3672 - categorical_accuracy: 0.8856
15296/60000 [======>.......................] - ETA: 1:24 - loss: 0.3666 - categorical_accuracy: 0.8859
15328/60000 [======>.......................] - ETA: 1:24 - loss: 0.3659 - categorical_accuracy: 0.8861
15360/60000 [======>.......................] - ETA: 1:24 - loss: 0.3653 - categorical_accuracy: 0.8863
15392/60000 [======>.......................] - ETA: 1:24 - loss: 0.3648 - categorical_accuracy: 0.8864
15424/60000 [======>.......................] - ETA: 1:24 - loss: 0.3646 - categorical_accuracy: 0.8865
15456/60000 [======>.......................] - ETA: 1:24 - loss: 0.3645 - categorical_accuracy: 0.8865
15488/60000 [======>.......................] - ETA: 1:24 - loss: 0.3646 - categorical_accuracy: 0.8866
15520/60000 [======>.......................] - ETA: 1:24 - loss: 0.3641 - categorical_accuracy: 0.8869
15552/60000 [======>.......................] - ETA: 1:23 - loss: 0.3635 - categorical_accuracy: 0.8871
15584/60000 [======>.......................] - ETA: 1:23 - loss: 0.3629 - categorical_accuracy: 0.8873
15616/60000 [======>.......................] - ETA: 1:23 - loss: 0.3628 - categorical_accuracy: 0.8874
15648/60000 [======>.......................] - ETA: 1:23 - loss: 0.3622 - categorical_accuracy: 0.8876
15680/60000 [======>.......................] - ETA: 1:23 - loss: 0.3621 - categorical_accuracy: 0.8877
15712/60000 [======>.......................] - ETA: 1:23 - loss: 0.3615 - categorical_accuracy: 0.8879
15744/60000 [======>.......................] - ETA: 1:23 - loss: 0.3615 - categorical_accuracy: 0.8880
15776/60000 [======>.......................] - ETA: 1:23 - loss: 0.3609 - categorical_accuracy: 0.8882
15808/60000 [======>.......................] - ETA: 1:23 - loss: 0.3603 - categorical_accuracy: 0.8883
15840/60000 [======>.......................] - ETA: 1:23 - loss: 0.3600 - categorical_accuracy: 0.8885
15872/60000 [======>.......................] - ETA: 1:23 - loss: 0.3594 - categorical_accuracy: 0.8887
15904/60000 [======>.......................] - ETA: 1:23 - loss: 0.3587 - categorical_accuracy: 0.8889
15936/60000 [======>.......................] - ETA: 1:23 - loss: 0.3584 - categorical_accuracy: 0.8890
15968/60000 [======>.......................] - ETA: 1:23 - loss: 0.3578 - categorical_accuracy: 0.8892
16000/60000 [=======>......................] - ETA: 1:23 - loss: 0.3572 - categorical_accuracy: 0.8894
16032/60000 [=======>......................] - ETA: 1:23 - loss: 0.3566 - categorical_accuracy: 0.8896
16064/60000 [=======>......................] - ETA: 1:22 - loss: 0.3561 - categorical_accuracy: 0.8898
16096/60000 [=======>......................] - ETA: 1:22 - loss: 0.3557 - categorical_accuracy: 0.8899
16128/60000 [=======>......................] - ETA: 1:22 - loss: 0.3554 - categorical_accuracy: 0.8899
16160/60000 [=======>......................] - ETA: 1:22 - loss: 0.3550 - categorical_accuracy: 0.8900
16192/60000 [=======>......................] - ETA: 1:22 - loss: 0.3544 - categorical_accuracy: 0.8903
16224/60000 [=======>......................] - ETA: 1:22 - loss: 0.3544 - categorical_accuracy: 0.8903
16256/60000 [=======>......................] - ETA: 1:22 - loss: 0.3542 - categorical_accuracy: 0.8904
16288/60000 [=======>......................] - ETA: 1:22 - loss: 0.3538 - categorical_accuracy: 0.8904
16320/60000 [=======>......................] - ETA: 1:22 - loss: 0.3533 - categorical_accuracy: 0.8906
16352/60000 [=======>......................] - ETA: 1:22 - loss: 0.3526 - categorical_accuracy: 0.8908
16384/60000 [=======>......................] - ETA: 1:22 - loss: 0.3522 - categorical_accuracy: 0.8909
16416/60000 [=======>......................] - ETA: 1:22 - loss: 0.3518 - categorical_accuracy: 0.8910
16448/60000 [=======>......................] - ETA: 1:22 - loss: 0.3515 - categorical_accuracy: 0.8911
16480/60000 [=======>......................] - ETA: 1:22 - loss: 0.3509 - categorical_accuracy: 0.8913
16512/60000 [=======>......................] - ETA: 1:22 - loss: 0.3504 - categorical_accuracy: 0.8914
16544/60000 [=======>......................] - ETA: 1:22 - loss: 0.3500 - categorical_accuracy: 0.8916
16576/60000 [=======>......................] - ETA: 1:22 - loss: 0.3497 - categorical_accuracy: 0.8917
16608/60000 [=======>......................] - ETA: 1:21 - loss: 0.3497 - categorical_accuracy: 0.8917
16640/60000 [=======>......................] - ETA: 1:21 - loss: 0.3491 - categorical_accuracy: 0.8919
16672/60000 [=======>......................] - ETA: 1:21 - loss: 0.3486 - categorical_accuracy: 0.8921
16704/60000 [=======>......................] - ETA: 1:21 - loss: 0.3481 - categorical_accuracy: 0.8922
16736/60000 [=======>......................] - ETA: 1:21 - loss: 0.3477 - categorical_accuracy: 0.8923
16768/60000 [=======>......................] - ETA: 1:21 - loss: 0.3473 - categorical_accuracy: 0.8925
16800/60000 [=======>......................] - ETA: 1:21 - loss: 0.3467 - categorical_accuracy: 0.8927
16832/60000 [=======>......................] - ETA: 1:21 - loss: 0.3462 - categorical_accuracy: 0.8928
16864/60000 [=======>......................] - ETA: 1:21 - loss: 0.3456 - categorical_accuracy: 0.8930
16896/60000 [=======>......................] - ETA: 1:21 - loss: 0.3454 - categorical_accuracy: 0.8930
16928/60000 [=======>......................] - ETA: 1:21 - loss: 0.3453 - categorical_accuracy: 0.8930
16960/60000 [=======>......................] - ETA: 1:21 - loss: 0.3450 - categorical_accuracy: 0.8930
16992/60000 [=======>......................] - ETA: 1:21 - loss: 0.3450 - categorical_accuracy: 0.8931
17024/60000 [=======>......................] - ETA: 1:21 - loss: 0.3445 - categorical_accuracy: 0.8932
17056/60000 [=======>......................] - ETA: 1:21 - loss: 0.3440 - categorical_accuracy: 0.8933
17088/60000 [=======>......................] - ETA: 1:20 - loss: 0.3437 - categorical_accuracy: 0.8934
17120/60000 [=======>......................] - ETA: 1:20 - loss: 0.3432 - categorical_accuracy: 0.8936
17152/60000 [=======>......................] - ETA: 1:20 - loss: 0.3428 - categorical_accuracy: 0.8937
17184/60000 [=======>......................] - ETA: 1:20 - loss: 0.3426 - categorical_accuracy: 0.8939
17216/60000 [=======>......................] - ETA: 1:20 - loss: 0.3421 - categorical_accuracy: 0.8940
17248/60000 [=======>......................] - ETA: 1:20 - loss: 0.3423 - categorical_accuracy: 0.8941
17280/60000 [=======>......................] - ETA: 1:20 - loss: 0.3418 - categorical_accuracy: 0.8943
17312/60000 [=======>......................] - ETA: 1:20 - loss: 0.3414 - categorical_accuracy: 0.8944
17344/60000 [=======>......................] - ETA: 1:20 - loss: 0.3410 - categorical_accuracy: 0.8945
17376/60000 [=======>......................] - ETA: 1:20 - loss: 0.3404 - categorical_accuracy: 0.8947
17408/60000 [=======>......................] - ETA: 1:20 - loss: 0.3398 - categorical_accuracy: 0.8949
17440/60000 [=======>......................] - ETA: 1:20 - loss: 0.3397 - categorical_accuracy: 0.8950
17472/60000 [=======>......................] - ETA: 1:20 - loss: 0.3393 - categorical_accuracy: 0.8951
17504/60000 [=======>......................] - ETA: 1:20 - loss: 0.3388 - categorical_accuracy: 0.8953
17536/60000 [=======>......................] - ETA: 1:20 - loss: 0.3384 - categorical_accuracy: 0.8954
17568/60000 [=======>......................] - ETA: 1:20 - loss: 0.3378 - categorical_accuracy: 0.8956
17600/60000 [=======>......................] - ETA: 1:19 - loss: 0.3378 - categorical_accuracy: 0.8956
17632/60000 [=======>......................] - ETA: 1:19 - loss: 0.3372 - categorical_accuracy: 0.8958
17664/60000 [=======>......................] - ETA: 1:19 - loss: 0.3367 - categorical_accuracy: 0.8960
17696/60000 [=======>......................] - ETA: 1:19 - loss: 0.3364 - categorical_accuracy: 0.8960
17728/60000 [=======>......................] - ETA: 1:19 - loss: 0.3362 - categorical_accuracy: 0.8961
17760/60000 [=======>......................] - ETA: 1:19 - loss: 0.3358 - categorical_accuracy: 0.8962
17792/60000 [=======>......................] - ETA: 1:19 - loss: 0.3354 - categorical_accuracy: 0.8963
17824/60000 [=======>......................] - ETA: 1:19 - loss: 0.3349 - categorical_accuracy: 0.8965
17856/60000 [=======>......................] - ETA: 1:19 - loss: 0.3344 - categorical_accuracy: 0.8966
17888/60000 [=======>......................] - ETA: 1:19 - loss: 0.3339 - categorical_accuracy: 0.8968
17920/60000 [=======>......................] - ETA: 1:19 - loss: 0.3337 - categorical_accuracy: 0.8968
17952/60000 [=======>......................] - ETA: 1:19 - loss: 0.3336 - categorical_accuracy: 0.8968
17984/60000 [=======>......................] - ETA: 1:19 - loss: 0.3333 - categorical_accuracy: 0.8969
18016/60000 [========>.....................] - ETA: 1:19 - loss: 0.3329 - categorical_accuracy: 0.8971
18048/60000 [========>.....................] - ETA: 1:19 - loss: 0.3323 - categorical_accuracy: 0.8973
18080/60000 [========>.....................] - ETA: 1:19 - loss: 0.3322 - categorical_accuracy: 0.8973
18112/60000 [========>.....................] - ETA: 1:18 - loss: 0.3317 - categorical_accuracy: 0.8975
18144/60000 [========>.....................] - ETA: 1:18 - loss: 0.3313 - categorical_accuracy: 0.8976
18176/60000 [========>.....................] - ETA: 1:18 - loss: 0.3308 - categorical_accuracy: 0.8978
18208/60000 [========>.....................] - ETA: 1:18 - loss: 0.3311 - categorical_accuracy: 0.8977
18240/60000 [========>.....................] - ETA: 1:18 - loss: 0.3307 - categorical_accuracy: 0.8978
18272/60000 [========>.....................] - ETA: 1:18 - loss: 0.3304 - categorical_accuracy: 0.8979
18304/60000 [========>.....................] - ETA: 1:18 - loss: 0.3303 - categorical_accuracy: 0.8980
18336/60000 [========>.....................] - ETA: 1:18 - loss: 0.3298 - categorical_accuracy: 0.8982
18368/60000 [========>.....................] - ETA: 1:18 - loss: 0.3294 - categorical_accuracy: 0.8984
18400/60000 [========>.....................] - ETA: 1:18 - loss: 0.3292 - categorical_accuracy: 0.8984
18432/60000 [========>.....................] - ETA: 1:18 - loss: 0.3290 - categorical_accuracy: 0.8985
18464/60000 [========>.....................] - ETA: 1:18 - loss: 0.3289 - categorical_accuracy: 0.8985
18496/60000 [========>.....................] - ETA: 1:18 - loss: 0.3287 - categorical_accuracy: 0.8986
18528/60000 [========>.....................] - ETA: 1:18 - loss: 0.3283 - categorical_accuracy: 0.8987
18560/60000 [========>.....................] - ETA: 1:18 - loss: 0.3281 - categorical_accuracy: 0.8988
18592/60000 [========>.....................] - ETA: 1:18 - loss: 0.3276 - categorical_accuracy: 0.8990
18624/60000 [========>.....................] - ETA: 1:17 - loss: 0.3274 - categorical_accuracy: 0.8991
18656/60000 [========>.....................] - ETA: 1:17 - loss: 0.3269 - categorical_accuracy: 0.8992
18688/60000 [========>.....................] - ETA: 1:17 - loss: 0.3265 - categorical_accuracy: 0.8993
18720/60000 [========>.....................] - ETA: 1:17 - loss: 0.3261 - categorical_accuracy: 0.8994
18752/60000 [========>.....................] - ETA: 1:17 - loss: 0.3256 - categorical_accuracy: 0.8996
18784/60000 [========>.....................] - ETA: 1:17 - loss: 0.3253 - categorical_accuracy: 0.8996
18816/60000 [========>.....................] - ETA: 1:17 - loss: 0.3248 - categorical_accuracy: 0.8998
18848/60000 [========>.....................] - ETA: 1:17 - loss: 0.3247 - categorical_accuracy: 0.8998
18880/60000 [========>.....................] - ETA: 1:17 - loss: 0.3248 - categorical_accuracy: 0.8998
18912/60000 [========>.....................] - ETA: 1:17 - loss: 0.3247 - categorical_accuracy: 0.8997
18944/60000 [========>.....................] - ETA: 1:17 - loss: 0.3243 - categorical_accuracy: 0.8999
18976/60000 [========>.....................] - ETA: 1:17 - loss: 0.3246 - categorical_accuracy: 0.8999
19008/60000 [========>.....................] - ETA: 1:17 - loss: 0.3242 - categorical_accuracy: 0.9000
19040/60000 [========>.....................] - ETA: 1:17 - loss: 0.3236 - categorical_accuracy: 0.9002
19072/60000 [========>.....................] - ETA: 1:17 - loss: 0.3232 - categorical_accuracy: 0.9003
19104/60000 [========>.....................] - ETA: 1:17 - loss: 0.3228 - categorical_accuracy: 0.9004
19136/60000 [========>.....................] - ETA: 1:16 - loss: 0.3223 - categorical_accuracy: 0.9006
19168/60000 [========>.....................] - ETA: 1:16 - loss: 0.3218 - categorical_accuracy: 0.9008
19200/60000 [========>.....................] - ETA: 1:16 - loss: 0.3215 - categorical_accuracy: 0.9008
19232/60000 [========>.....................] - ETA: 1:16 - loss: 0.3211 - categorical_accuracy: 0.9008
19264/60000 [========>.....................] - ETA: 1:16 - loss: 0.3207 - categorical_accuracy: 0.9010
19296/60000 [========>.....................] - ETA: 1:16 - loss: 0.3203 - categorical_accuracy: 0.9011
19328/60000 [========>.....................] - ETA: 1:16 - loss: 0.3198 - categorical_accuracy: 0.9013
19360/60000 [========>.....................] - ETA: 1:16 - loss: 0.3196 - categorical_accuracy: 0.9013
19392/60000 [========>.....................] - ETA: 1:16 - loss: 0.3191 - categorical_accuracy: 0.9015
19424/60000 [========>.....................] - ETA: 1:16 - loss: 0.3195 - categorical_accuracy: 0.9015
19456/60000 [========>.....................] - ETA: 1:16 - loss: 0.3195 - categorical_accuracy: 0.9014
19488/60000 [========>.....................] - ETA: 1:16 - loss: 0.3198 - categorical_accuracy: 0.9014
19520/60000 [========>.....................] - ETA: 1:16 - loss: 0.3196 - categorical_accuracy: 0.9014
19552/60000 [========>.....................] - ETA: 1:16 - loss: 0.3198 - categorical_accuracy: 0.9014
19584/60000 [========>.....................] - ETA: 1:16 - loss: 0.3194 - categorical_accuracy: 0.9015
19616/60000 [========>.....................] - ETA: 1:16 - loss: 0.3190 - categorical_accuracy: 0.9016
19648/60000 [========>.....................] - ETA: 1:16 - loss: 0.3193 - categorical_accuracy: 0.9015
19680/60000 [========>.....................] - ETA: 1:15 - loss: 0.3191 - categorical_accuracy: 0.9016
19712/60000 [========>.....................] - ETA: 1:15 - loss: 0.3188 - categorical_accuracy: 0.9017
19744/60000 [========>.....................] - ETA: 1:15 - loss: 0.3188 - categorical_accuracy: 0.9016
19776/60000 [========>.....................] - ETA: 1:15 - loss: 0.3187 - categorical_accuracy: 0.9016
19808/60000 [========>.....................] - ETA: 1:15 - loss: 0.3184 - categorical_accuracy: 0.9018
19840/60000 [========>.....................] - ETA: 1:15 - loss: 0.3183 - categorical_accuracy: 0.9018
19872/60000 [========>.....................] - ETA: 1:15 - loss: 0.3183 - categorical_accuracy: 0.9018
19904/60000 [========>.....................] - ETA: 1:15 - loss: 0.3179 - categorical_accuracy: 0.9019
19936/60000 [========>.....................] - ETA: 1:15 - loss: 0.3176 - categorical_accuracy: 0.9020
19968/60000 [========>.....................] - ETA: 1:15 - loss: 0.3171 - categorical_accuracy: 0.9022
20000/60000 [=========>....................] - ETA: 1:15 - loss: 0.3168 - categorical_accuracy: 0.9023
20032/60000 [=========>....................] - ETA: 1:15 - loss: 0.3164 - categorical_accuracy: 0.9024
20064/60000 [=========>....................] - ETA: 1:15 - loss: 0.3162 - categorical_accuracy: 0.9024
20096/60000 [=========>....................] - ETA: 1:15 - loss: 0.3158 - categorical_accuracy: 0.9026
20128/60000 [=========>....................] - ETA: 1:15 - loss: 0.3160 - categorical_accuracy: 0.9026
20160/60000 [=========>....................] - ETA: 1:14 - loss: 0.3157 - categorical_accuracy: 0.9027
20192/60000 [=========>....................] - ETA: 1:14 - loss: 0.3155 - categorical_accuracy: 0.9027
20224/60000 [=========>....................] - ETA: 1:14 - loss: 0.3155 - categorical_accuracy: 0.9028
20256/60000 [=========>....................] - ETA: 1:14 - loss: 0.3151 - categorical_accuracy: 0.9029
20288/60000 [=========>....................] - ETA: 1:14 - loss: 0.3147 - categorical_accuracy: 0.9030
20320/60000 [=========>....................] - ETA: 1:14 - loss: 0.3149 - categorical_accuracy: 0.9031
20352/60000 [=========>....................] - ETA: 1:14 - loss: 0.3148 - categorical_accuracy: 0.9032
20384/60000 [=========>....................] - ETA: 1:14 - loss: 0.3145 - categorical_accuracy: 0.9033
20416/60000 [=========>....................] - ETA: 1:14 - loss: 0.3143 - categorical_accuracy: 0.9033
20448/60000 [=========>....................] - ETA: 1:14 - loss: 0.3142 - categorical_accuracy: 0.9032
20480/60000 [=========>....................] - ETA: 1:14 - loss: 0.3141 - categorical_accuracy: 0.9033
20512/60000 [=========>....................] - ETA: 1:14 - loss: 0.3138 - categorical_accuracy: 0.9033
20544/60000 [=========>....................] - ETA: 1:14 - loss: 0.3135 - categorical_accuracy: 0.9034
20576/60000 [=========>....................] - ETA: 1:14 - loss: 0.3131 - categorical_accuracy: 0.9035
20608/60000 [=========>....................] - ETA: 1:14 - loss: 0.3129 - categorical_accuracy: 0.9036
20640/60000 [=========>....................] - ETA: 1:14 - loss: 0.3126 - categorical_accuracy: 0.9037
20672/60000 [=========>....................] - ETA: 1:13 - loss: 0.3123 - categorical_accuracy: 0.9038
20704/60000 [=========>....................] - ETA: 1:13 - loss: 0.3119 - categorical_accuracy: 0.9039
20736/60000 [=========>....................] - ETA: 1:13 - loss: 0.3117 - categorical_accuracy: 0.9039
20768/60000 [=========>....................] - ETA: 1:13 - loss: 0.3113 - categorical_accuracy: 0.9040
20800/60000 [=========>....................] - ETA: 1:13 - loss: 0.3111 - categorical_accuracy: 0.9041
20832/60000 [=========>....................] - ETA: 1:13 - loss: 0.3106 - categorical_accuracy: 0.9042
20864/60000 [=========>....................] - ETA: 1:13 - loss: 0.3104 - categorical_accuracy: 0.9043
20896/60000 [=========>....................] - ETA: 1:13 - loss: 0.3101 - categorical_accuracy: 0.9044
20928/60000 [=========>....................] - ETA: 1:13 - loss: 0.3097 - categorical_accuracy: 0.9045
20960/60000 [=========>....................] - ETA: 1:13 - loss: 0.3093 - categorical_accuracy: 0.9046
20992/60000 [=========>....................] - ETA: 1:13 - loss: 0.3100 - categorical_accuracy: 0.9045
21024/60000 [=========>....................] - ETA: 1:13 - loss: 0.3095 - categorical_accuracy: 0.9047
21056/60000 [=========>....................] - ETA: 1:13 - loss: 0.3092 - categorical_accuracy: 0.9048
21088/60000 [=========>....................] - ETA: 1:13 - loss: 0.3088 - categorical_accuracy: 0.9050
21120/60000 [=========>....................] - ETA: 1:13 - loss: 0.3087 - categorical_accuracy: 0.9050
21152/60000 [=========>....................] - ETA: 1:13 - loss: 0.3084 - categorical_accuracy: 0.9051
21184/60000 [=========>....................] - ETA: 1:13 - loss: 0.3081 - categorical_accuracy: 0.9052
21216/60000 [=========>....................] - ETA: 1:12 - loss: 0.3076 - categorical_accuracy: 0.9054
21248/60000 [=========>....................] - ETA: 1:12 - loss: 0.3073 - categorical_accuracy: 0.9054
21280/60000 [=========>....................] - ETA: 1:12 - loss: 0.3069 - categorical_accuracy: 0.9056
21312/60000 [=========>....................] - ETA: 1:12 - loss: 0.3065 - categorical_accuracy: 0.9057
21344/60000 [=========>....................] - ETA: 1:12 - loss: 0.3061 - categorical_accuracy: 0.9059
21376/60000 [=========>....................] - ETA: 1:12 - loss: 0.3061 - categorical_accuracy: 0.9059
21408/60000 [=========>....................] - ETA: 1:12 - loss: 0.3056 - categorical_accuracy: 0.9061
21440/60000 [=========>....................] - ETA: 1:12 - loss: 0.3053 - categorical_accuracy: 0.9062
21472/60000 [=========>....................] - ETA: 1:12 - loss: 0.3051 - categorical_accuracy: 0.9062
21504/60000 [=========>....................] - ETA: 1:12 - loss: 0.3049 - categorical_accuracy: 0.9062
21536/60000 [=========>....................] - ETA: 1:12 - loss: 0.3048 - categorical_accuracy: 0.9062
21568/60000 [=========>....................] - ETA: 1:12 - loss: 0.3045 - categorical_accuracy: 0.9063
21600/60000 [=========>....................] - ETA: 1:12 - loss: 0.3042 - categorical_accuracy: 0.9064
21632/60000 [=========>....................] - ETA: 1:12 - loss: 0.3038 - categorical_accuracy: 0.9065
21664/60000 [=========>....................] - ETA: 1:12 - loss: 0.3035 - categorical_accuracy: 0.9066
21696/60000 [=========>....................] - ETA: 1:12 - loss: 0.3032 - categorical_accuracy: 0.9067
21728/60000 [=========>....................] - ETA: 1:11 - loss: 0.3029 - categorical_accuracy: 0.9068
21760/60000 [=========>....................] - ETA: 1:11 - loss: 0.3026 - categorical_accuracy: 0.9069
21792/60000 [=========>....................] - ETA: 1:11 - loss: 0.3022 - categorical_accuracy: 0.9070
21824/60000 [=========>....................] - ETA: 1:11 - loss: 0.3019 - categorical_accuracy: 0.9071
21856/60000 [=========>....................] - ETA: 1:11 - loss: 0.3016 - categorical_accuracy: 0.9072
21888/60000 [=========>....................] - ETA: 1:11 - loss: 0.3012 - categorical_accuracy: 0.9073
21920/60000 [=========>....................] - ETA: 1:11 - loss: 0.3010 - categorical_accuracy: 0.9074
21952/60000 [=========>....................] - ETA: 1:11 - loss: 0.3009 - categorical_accuracy: 0.9075
21984/60000 [=========>....................] - ETA: 1:11 - loss: 0.3007 - categorical_accuracy: 0.9076
22016/60000 [==========>...................] - ETA: 1:11 - loss: 0.3002 - categorical_accuracy: 0.9077
22048/60000 [==========>...................] - ETA: 1:11 - loss: 0.3000 - categorical_accuracy: 0.9078
22080/60000 [==========>...................] - ETA: 1:11 - loss: 0.2996 - categorical_accuracy: 0.9079
22112/60000 [==========>...................] - ETA: 1:11 - loss: 0.2991 - categorical_accuracy: 0.9081
22144/60000 [==========>...................] - ETA: 1:11 - loss: 0.2991 - categorical_accuracy: 0.9081
22176/60000 [==========>...................] - ETA: 1:11 - loss: 0.2990 - categorical_accuracy: 0.9081
22208/60000 [==========>...................] - ETA: 1:11 - loss: 0.2986 - categorical_accuracy: 0.9083
22240/60000 [==========>...................] - ETA: 1:10 - loss: 0.2983 - categorical_accuracy: 0.9084
22272/60000 [==========>...................] - ETA: 1:10 - loss: 0.2983 - categorical_accuracy: 0.9084
22304/60000 [==========>...................] - ETA: 1:10 - loss: 0.2981 - categorical_accuracy: 0.9084
22336/60000 [==========>...................] - ETA: 1:10 - loss: 0.2977 - categorical_accuracy: 0.9086
22368/60000 [==========>...................] - ETA: 1:10 - loss: 0.2976 - categorical_accuracy: 0.9086
22400/60000 [==========>...................] - ETA: 1:10 - loss: 0.2972 - categorical_accuracy: 0.9087
22432/60000 [==========>...................] - ETA: 1:10 - loss: 0.2969 - categorical_accuracy: 0.9088
22464/60000 [==========>...................] - ETA: 1:10 - loss: 0.2972 - categorical_accuracy: 0.9088
22496/60000 [==========>...................] - ETA: 1:10 - loss: 0.2970 - categorical_accuracy: 0.9089
22528/60000 [==========>...................] - ETA: 1:10 - loss: 0.2971 - categorical_accuracy: 0.9090
22560/60000 [==========>...................] - ETA: 1:10 - loss: 0.2967 - categorical_accuracy: 0.9091
22592/60000 [==========>...................] - ETA: 1:10 - loss: 0.2967 - categorical_accuracy: 0.9091
22624/60000 [==========>...................] - ETA: 1:10 - loss: 0.2964 - categorical_accuracy: 0.9092
22656/60000 [==========>...................] - ETA: 1:10 - loss: 0.2961 - categorical_accuracy: 0.9093
22688/60000 [==========>...................] - ETA: 1:10 - loss: 0.2958 - categorical_accuracy: 0.9094
22720/60000 [==========>...................] - ETA: 1:10 - loss: 0.2956 - categorical_accuracy: 0.9094
22752/60000 [==========>...................] - ETA: 1:10 - loss: 0.2953 - categorical_accuracy: 0.9095
22784/60000 [==========>...................] - ETA: 1:09 - loss: 0.2950 - categorical_accuracy: 0.9095
22816/60000 [==========>...................] - ETA: 1:09 - loss: 0.2948 - categorical_accuracy: 0.9097
22848/60000 [==========>...................] - ETA: 1:09 - loss: 0.2945 - categorical_accuracy: 0.9098
22880/60000 [==========>...................] - ETA: 1:09 - loss: 0.2943 - categorical_accuracy: 0.9097
22912/60000 [==========>...................] - ETA: 1:09 - loss: 0.2944 - categorical_accuracy: 0.9098
22944/60000 [==========>...................] - ETA: 1:09 - loss: 0.2942 - categorical_accuracy: 0.9099
22976/60000 [==========>...................] - ETA: 1:09 - loss: 0.2938 - categorical_accuracy: 0.9100
23008/60000 [==========>...................] - ETA: 1:09 - loss: 0.2935 - categorical_accuracy: 0.9101
23040/60000 [==========>...................] - ETA: 1:09 - loss: 0.2933 - categorical_accuracy: 0.9101
23072/60000 [==========>...................] - ETA: 1:09 - loss: 0.2930 - categorical_accuracy: 0.9102
23104/60000 [==========>...................] - ETA: 1:09 - loss: 0.2927 - categorical_accuracy: 0.9103
23136/60000 [==========>...................] - ETA: 1:09 - loss: 0.2928 - categorical_accuracy: 0.9102
23168/60000 [==========>...................] - ETA: 1:09 - loss: 0.2926 - categorical_accuracy: 0.9103
23200/60000 [==========>...................] - ETA: 1:09 - loss: 0.2924 - categorical_accuracy: 0.9103
23232/60000 [==========>...................] - ETA: 1:09 - loss: 0.2925 - categorical_accuracy: 0.9103
23264/60000 [==========>...................] - ETA: 1:09 - loss: 0.2923 - categorical_accuracy: 0.9104
23296/60000 [==========>...................] - ETA: 1:09 - loss: 0.2924 - categorical_accuracy: 0.9104
23328/60000 [==========>...................] - ETA: 1:08 - loss: 0.2921 - categorical_accuracy: 0.9105
23360/60000 [==========>...................] - ETA: 1:08 - loss: 0.2919 - categorical_accuracy: 0.9105
23392/60000 [==========>...................] - ETA: 1:08 - loss: 0.2916 - categorical_accuracy: 0.9106
23424/60000 [==========>...................] - ETA: 1:08 - loss: 0.2913 - categorical_accuracy: 0.9107
23456/60000 [==========>...................] - ETA: 1:08 - loss: 0.2912 - categorical_accuracy: 0.9108
23488/60000 [==========>...................] - ETA: 1:08 - loss: 0.2909 - categorical_accuracy: 0.9108
23520/60000 [==========>...................] - ETA: 1:08 - loss: 0.2906 - categorical_accuracy: 0.9109
23552/60000 [==========>...................] - ETA: 1:08 - loss: 0.2904 - categorical_accuracy: 0.9110
23584/60000 [==========>...................] - ETA: 1:08 - loss: 0.2901 - categorical_accuracy: 0.9111
23616/60000 [==========>...................] - ETA: 1:08 - loss: 0.2900 - categorical_accuracy: 0.9112
23648/60000 [==========>...................] - ETA: 1:08 - loss: 0.2898 - categorical_accuracy: 0.9113
23680/60000 [==========>...................] - ETA: 1:08 - loss: 0.2894 - categorical_accuracy: 0.9114
23712/60000 [==========>...................] - ETA: 1:08 - loss: 0.2891 - categorical_accuracy: 0.9115
23744/60000 [==========>...................] - ETA: 1:08 - loss: 0.2887 - categorical_accuracy: 0.9116
23776/60000 [==========>...................] - ETA: 1:08 - loss: 0.2884 - categorical_accuracy: 0.9118
23808/60000 [==========>...................] - ETA: 1:08 - loss: 0.2881 - categorical_accuracy: 0.9118
23840/60000 [==========>...................] - ETA: 1:07 - loss: 0.2879 - categorical_accuracy: 0.9119
23872/60000 [==========>...................] - ETA: 1:07 - loss: 0.2875 - categorical_accuracy: 0.9120
23904/60000 [==========>...................] - ETA: 1:07 - loss: 0.2875 - categorical_accuracy: 0.9121
23936/60000 [==========>...................] - ETA: 1:07 - loss: 0.2873 - categorical_accuracy: 0.9121
23968/60000 [==========>...................] - ETA: 1:07 - loss: 0.2871 - categorical_accuracy: 0.9121
24000/60000 [===========>..................] - ETA: 1:07 - loss: 0.2867 - categorical_accuracy: 0.9122
24032/60000 [===========>..................] - ETA: 1:07 - loss: 0.2864 - categorical_accuracy: 0.9124
24064/60000 [===========>..................] - ETA: 1:07 - loss: 0.2861 - categorical_accuracy: 0.9124
24096/60000 [===========>..................] - ETA: 1:07 - loss: 0.2858 - categorical_accuracy: 0.9126
24128/60000 [===========>..................] - ETA: 1:07 - loss: 0.2856 - categorical_accuracy: 0.9126
24160/60000 [===========>..................] - ETA: 1:07 - loss: 0.2855 - categorical_accuracy: 0.9126
24192/60000 [===========>..................] - ETA: 1:07 - loss: 0.2851 - categorical_accuracy: 0.9127
24224/60000 [===========>..................] - ETA: 1:07 - loss: 0.2849 - categorical_accuracy: 0.9127
24256/60000 [===========>..................] - ETA: 1:07 - loss: 0.2848 - categorical_accuracy: 0.9128
24288/60000 [===========>..................] - ETA: 1:07 - loss: 0.2845 - categorical_accuracy: 0.9129
24320/60000 [===========>..................] - ETA: 1:07 - loss: 0.2843 - categorical_accuracy: 0.9130
24352/60000 [===========>..................] - ETA: 1:06 - loss: 0.2841 - categorical_accuracy: 0.9131
24384/60000 [===========>..................] - ETA: 1:06 - loss: 0.2838 - categorical_accuracy: 0.9132
24416/60000 [===========>..................] - ETA: 1:06 - loss: 0.2835 - categorical_accuracy: 0.9133
24448/60000 [===========>..................] - ETA: 1:06 - loss: 0.2836 - categorical_accuracy: 0.9133
24480/60000 [===========>..................] - ETA: 1:06 - loss: 0.2833 - categorical_accuracy: 0.9134
24512/60000 [===========>..................] - ETA: 1:06 - loss: 0.2831 - categorical_accuracy: 0.9135
24544/60000 [===========>..................] - ETA: 1:06 - loss: 0.2830 - categorical_accuracy: 0.9135
24576/60000 [===========>..................] - ETA: 1:06 - loss: 0.2827 - categorical_accuracy: 0.9137
24608/60000 [===========>..................] - ETA: 1:06 - loss: 0.2825 - categorical_accuracy: 0.9137
24640/60000 [===========>..................] - ETA: 1:06 - loss: 0.2822 - categorical_accuracy: 0.9138
24672/60000 [===========>..................] - ETA: 1:06 - loss: 0.2819 - categorical_accuracy: 0.9139
24704/60000 [===========>..................] - ETA: 1:06 - loss: 0.2818 - categorical_accuracy: 0.9139
24736/60000 [===========>..................] - ETA: 1:06 - loss: 0.2814 - categorical_accuracy: 0.9141
24768/60000 [===========>..................] - ETA: 1:06 - loss: 0.2811 - categorical_accuracy: 0.9142
24800/60000 [===========>..................] - ETA: 1:06 - loss: 0.2813 - categorical_accuracy: 0.9142
24832/60000 [===========>..................] - ETA: 1:06 - loss: 0.2812 - categorical_accuracy: 0.9142
24864/60000 [===========>..................] - ETA: 1:05 - loss: 0.2813 - categorical_accuracy: 0.9143
24896/60000 [===========>..................] - ETA: 1:05 - loss: 0.2814 - categorical_accuracy: 0.9143
24928/60000 [===========>..................] - ETA: 1:05 - loss: 0.2813 - categorical_accuracy: 0.9143
24960/60000 [===========>..................] - ETA: 1:05 - loss: 0.2811 - categorical_accuracy: 0.9143
24992/60000 [===========>..................] - ETA: 1:05 - loss: 0.2808 - categorical_accuracy: 0.9144
25024/60000 [===========>..................] - ETA: 1:05 - loss: 0.2806 - categorical_accuracy: 0.9145
25056/60000 [===========>..................] - ETA: 1:05 - loss: 0.2803 - categorical_accuracy: 0.9146
25088/60000 [===========>..................] - ETA: 1:05 - loss: 0.2801 - categorical_accuracy: 0.9147
25120/60000 [===========>..................] - ETA: 1:05 - loss: 0.2798 - categorical_accuracy: 0.9147
25152/60000 [===========>..................] - ETA: 1:05 - loss: 0.2795 - categorical_accuracy: 0.9148
25184/60000 [===========>..................] - ETA: 1:05 - loss: 0.2794 - categorical_accuracy: 0.9149
25216/60000 [===========>..................] - ETA: 1:05 - loss: 0.2791 - categorical_accuracy: 0.9149
25248/60000 [===========>..................] - ETA: 1:05 - loss: 0.2789 - categorical_accuracy: 0.9150
25280/60000 [===========>..................] - ETA: 1:05 - loss: 0.2787 - categorical_accuracy: 0.9150
25312/60000 [===========>..................] - ETA: 1:05 - loss: 0.2784 - categorical_accuracy: 0.9151
25344/60000 [===========>..................] - ETA: 1:05 - loss: 0.2783 - categorical_accuracy: 0.9152
25376/60000 [===========>..................] - ETA: 1:04 - loss: 0.2780 - categorical_accuracy: 0.9152
25408/60000 [===========>..................] - ETA: 1:04 - loss: 0.2777 - categorical_accuracy: 0.9153
25440/60000 [===========>..................] - ETA: 1:04 - loss: 0.2777 - categorical_accuracy: 0.9154
25472/60000 [===========>..................] - ETA: 1:04 - loss: 0.2775 - categorical_accuracy: 0.9154
25504/60000 [===========>..................] - ETA: 1:04 - loss: 0.2774 - categorical_accuracy: 0.9154
25536/60000 [===========>..................] - ETA: 1:04 - loss: 0.2772 - categorical_accuracy: 0.9155
25568/60000 [===========>..................] - ETA: 1:04 - loss: 0.2771 - categorical_accuracy: 0.9154
25600/60000 [===========>..................] - ETA: 1:04 - loss: 0.2770 - categorical_accuracy: 0.9155
25632/60000 [===========>..................] - ETA: 1:04 - loss: 0.2767 - categorical_accuracy: 0.9156
25664/60000 [===========>..................] - ETA: 1:04 - loss: 0.2769 - categorical_accuracy: 0.9156
25696/60000 [===========>..................] - ETA: 1:04 - loss: 0.2766 - categorical_accuracy: 0.9157
25728/60000 [===========>..................] - ETA: 1:04 - loss: 0.2764 - categorical_accuracy: 0.9157
25760/60000 [===========>..................] - ETA: 1:04 - loss: 0.2763 - categorical_accuracy: 0.9158
25792/60000 [===========>..................] - ETA: 1:04 - loss: 0.2763 - categorical_accuracy: 0.9157
25824/60000 [===========>..................] - ETA: 1:04 - loss: 0.2761 - categorical_accuracy: 0.9158
25856/60000 [===========>..................] - ETA: 1:04 - loss: 0.2761 - categorical_accuracy: 0.9158
25888/60000 [===========>..................] - ETA: 1:03 - loss: 0.2758 - categorical_accuracy: 0.9159
25920/60000 [===========>..................] - ETA: 1:03 - loss: 0.2755 - categorical_accuracy: 0.9160
25952/60000 [===========>..................] - ETA: 1:03 - loss: 0.2757 - categorical_accuracy: 0.9160
25984/60000 [===========>..................] - ETA: 1:03 - loss: 0.2754 - categorical_accuracy: 0.9160
26016/60000 [============>.................] - ETA: 1:03 - loss: 0.2751 - categorical_accuracy: 0.9161
26048/60000 [============>.................] - ETA: 1:03 - loss: 0.2750 - categorical_accuracy: 0.9162
26080/60000 [============>.................] - ETA: 1:03 - loss: 0.2747 - categorical_accuracy: 0.9162
26112/60000 [============>.................] - ETA: 1:03 - loss: 0.2746 - categorical_accuracy: 0.9163
26144/60000 [============>.................] - ETA: 1:03 - loss: 0.2743 - categorical_accuracy: 0.9164
26176/60000 [============>.................] - ETA: 1:03 - loss: 0.2740 - categorical_accuracy: 0.9165
26208/60000 [============>.................] - ETA: 1:03 - loss: 0.2739 - categorical_accuracy: 0.9165
26240/60000 [============>.................] - ETA: 1:03 - loss: 0.2737 - categorical_accuracy: 0.9165
26272/60000 [============>.................] - ETA: 1:03 - loss: 0.2735 - categorical_accuracy: 0.9166
26304/60000 [============>.................] - ETA: 1:03 - loss: 0.2734 - categorical_accuracy: 0.9167
26336/60000 [============>.................] - ETA: 1:03 - loss: 0.2731 - categorical_accuracy: 0.9167
26368/60000 [============>.................] - ETA: 1:03 - loss: 0.2732 - categorical_accuracy: 0.9167
26400/60000 [============>.................] - ETA: 1:03 - loss: 0.2729 - categorical_accuracy: 0.9168
26432/60000 [============>.................] - ETA: 1:02 - loss: 0.2728 - categorical_accuracy: 0.9168
26464/60000 [============>.................] - ETA: 1:02 - loss: 0.2729 - categorical_accuracy: 0.9168
26496/60000 [============>.................] - ETA: 1:02 - loss: 0.2727 - categorical_accuracy: 0.9168
26528/60000 [============>.................] - ETA: 1:02 - loss: 0.2726 - categorical_accuracy: 0.9168
26560/60000 [============>.................] - ETA: 1:02 - loss: 0.2723 - categorical_accuracy: 0.9168
26592/60000 [============>.................] - ETA: 1:02 - loss: 0.2720 - categorical_accuracy: 0.9169
26624/60000 [============>.................] - ETA: 1:02 - loss: 0.2717 - categorical_accuracy: 0.9170
26656/60000 [============>.................] - ETA: 1:02 - loss: 0.2715 - categorical_accuracy: 0.9171
26688/60000 [============>.................] - ETA: 1:02 - loss: 0.2713 - categorical_accuracy: 0.9172
26720/60000 [============>.................] - ETA: 1:02 - loss: 0.2710 - categorical_accuracy: 0.9173
26752/60000 [============>.................] - ETA: 1:02 - loss: 0.2708 - categorical_accuracy: 0.9173
26784/60000 [============>.................] - ETA: 1:02 - loss: 0.2706 - categorical_accuracy: 0.9174
26816/60000 [============>.................] - ETA: 1:02 - loss: 0.2704 - categorical_accuracy: 0.9174
26848/60000 [============>.................] - ETA: 1:02 - loss: 0.2703 - categorical_accuracy: 0.9174
26880/60000 [============>.................] - ETA: 1:02 - loss: 0.2702 - categorical_accuracy: 0.9174
26912/60000 [============>.................] - ETA: 1:02 - loss: 0.2699 - categorical_accuracy: 0.9175
26944/60000 [============>.................] - ETA: 1:02 - loss: 0.2697 - categorical_accuracy: 0.9176
26976/60000 [============>.................] - ETA: 1:01 - loss: 0.2696 - categorical_accuracy: 0.9176
27008/60000 [============>.................] - ETA: 1:01 - loss: 0.2694 - categorical_accuracy: 0.9177
27040/60000 [============>.................] - ETA: 1:01 - loss: 0.2691 - categorical_accuracy: 0.9178
27072/60000 [============>.................] - ETA: 1:01 - loss: 0.2689 - categorical_accuracy: 0.9178
27104/60000 [============>.................] - ETA: 1:01 - loss: 0.2686 - categorical_accuracy: 0.9179
27136/60000 [============>.................] - ETA: 1:01 - loss: 0.2685 - categorical_accuracy: 0.9180
27168/60000 [============>.................] - ETA: 1:01 - loss: 0.2683 - categorical_accuracy: 0.9180
27200/60000 [============>.................] - ETA: 1:01 - loss: 0.2681 - categorical_accuracy: 0.9180
27232/60000 [============>.................] - ETA: 1:01 - loss: 0.2678 - categorical_accuracy: 0.9181
27264/60000 [============>.................] - ETA: 1:01 - loss: 0.2675 - categorical_accuracy: 0.9182
27296/60000 [============>.................] - ETA: 1:01 - loss: 0.2672 - categorical_accuracy: 0.9183
27328/60000 [============>.................] - ETA: 1:01 - loss: 0.2670 - categorical_accuracy: 0.9183
27360/60000 [============>.................] - ETA: 1:01 - loss: 0.2667 - categorical_accuracy: 0.9184
27392/60000 [============>.................] - ETA: 1:01 - loss: 0.2665 - categorical_accuracy: 0.9184
27424/60000 [============>.................] - ETA: 1:01 - loss: 0.2664 - categorical_accuracy: 0.9185
27456/60000 [============>.................] - ETA: 1:01 - loss: 0.2661 - categorical_accuracy: 0.9186
27488/60000 [============>.................] - ETA: 1:00 - loss: 0.2659 - categorical_accuracy: 0.9187
27520/60000 [============>.................] - ETA: 1:00 - loss: 0.2657 - categorical_accuracy: 0.9187
27552/60000 [============>.................] - ETA: 1:00 - loss: 0.2659 - categorical_accuracy: 0.9187
27584/60000 [============>.................] - ETA: 1:00 - loss: 0.2656 - categorical_accuracy: 0.9188
27616/60000 [============>.................] - ETA: 1:00 - loss: 0.2654 - categorical_accuracy: 0.9189
27648/60000 [============>.................] - ETA: 1:00 - loss: 0.2655 - categorical_accuracy: 0.9188
27680/60000 [============>.................] - ETA: 1:00 - loss: 0.2654 - categorical_accuracy: 0.9188
27712/60000 [============>.................] - ETA: 1:00 - loss: 0.2651 - categorical_accuracy: 0.9189
27744/60000 [============>.................] - ETA: 1:00 - loss: 0.2650 - categorical_accuracy: 0.9189
27776/60000 [============>.................] - ETA: 1:00 - loss: 0.2647 - categorical_accuracy: 0.9190
27808/60000 [============>.................] - ETA: 1:00 - loss: 0.2645 - categorical_accuracy: 0.9190
27840/60000 [============>.................] - ETA: 1:00 - loss: 0.2643 - categorical_accuracy: 0.9190
27872/60000 [============>.................] - ETA: 1:00 - loss: 0.2641 - categorical_accuracy: 0.9191
27904/60000 [============>.................] - ETA: 1:00 - loss: 0.2638 - categorical_accuracy: 0.9192
27936/60000 [============>.................] - ETA: 1:00 - loss: 0.2636 - categorical_accuracy: 0.9192
27968/60000 [============>.................] - ETA: 1:00 - loss: 0.2635 - categorical_accuracy: 0.9193
28000/60000 [=============>................] - ETA: 1:00 - loss: 0.2633 - categorical_accuracy: 0.9193
28032/60000 [=============>................] - ETA: 59s - loss: 0.2631 - categorical_accuracy: 0.9193 
28064/60000 [=============>................] - ETA: 59s - loss: 0.2632 - categorical_accuracy: 0.9193
28096/60000 [=============>................] - ETA: 59s - loss: 0.2629 - categorical_accuracy: 0.9194
28128/60000 [=============>................] - ETA: 59s - loss: 0.2627 - categorical_accuracy: 0.9195
28160/60000 [=============>................] - ETA: 59s - loss: 0.2624 - categorical_accuracy: 0.9196
28192/60000 [=============>................] - ETA: 59s - loss: 0.2622 - categorical_accuracy: 0.9196
28224/60000 [=============>................] - ETA: 59s - loss: 0.2622 - categorical_accuracy: 0.9196
28256/60000 [=============>................] - ETA: 59s - loss: 0.2619 - categorical_accuracy: 0.9197
28288/60000 [=============>................] - ETA: 59s - loss: 0.2619 - categorical_accuracy: 0.9197
28320/60000 [=============>................] - ETA: 59s - loss: 0.2616 - categorical_accuracy: 0.9198
28352/60000 [=============>................] - ETA: 59s - loss: 0.2615 - categorical_accuracy: 0.9198
28384/60000 [=============>................] - ETA: 59s - loss: 0.2617 - categorical_accuracy: 0.9198
28416/60000 [=============>................] - ETA: 59s - loss: 0.2616 - categorical_accuracy: 0.9198
28448/60000 [=============>................] - ETA: 59s - loss: 0.2615 - categorical_accuracy: 0.9199
28480/60000 [=============>................] - ETA: 59s - loss: 0.2612 - categorical_accuracy: 0.9199
28512/60000 [=============>................] - ETA: 59s - loss: 0.2610 - categorical_accuracy: 0.9200
28544/60000 [=============>................] - ETA: 58s - loss: 0.2608 - categorical_accuracy: 0.9201
28576/60000 [=============>................] - ETA: 58s - loss: 0.2606 - categorical_accuracy: 0.9201
28608/60000 [=============>................] - ETA: 58s - loss: 0.2607 - categorical_accuracy: 0.9201
28640/60000 [=============>................] - ETA: 58s - loss: 0.2605 - categorical_accuracy: 0.9202
28672/60000 [=============>................] - ETA: 58s - loss: 0.2602 - categorical_accuracy: 0.9202
28704/60000 [=============>................] - ETA: 58s - loss: 0.2602 - categorical_accuracy: 0.9203
28736/60000 [=============>................] - ETA: 58s - loss: 0.2600 - categorical_accuracy: 0.9203
28768/60000 [=============>................] - ETA: 58s - loss: 0.2598 - categorical_accuracy: 0.9203
28800/60000 [=============>................] - ETA: 58s - loss: 0.2597 - categorical_accuracy: 0.9204
28832/60000 [=============>................] - ETA: 58s - loss: 0.2596 - categorical_accuracy: 0.9204
28864/60000 [=============>................] - ETA: 58s - loss: 0.2595 - categorical_accuracy: 0.9205
28896/60000 [=============>................] - ETA: 58s - loss: 0.2592 - categorical_accuracy: 0.9206
28928/60000 [=============>................] - ETA: 58s - loss: 0.2589 - categorical_accuracy: 0.9207
28960/60000 [=============>................] - ETA: 58s - loss: 0.2589 - categorical_accuracy: 0.9207
28992/60000 [=============>................] - ETA: 58s - loss: 0.2586 - categorical_accuracy: 0.9208
29024/60000 [=============>................] - ETA: 58s - loss: 0.2584 - categorical_accuracy: 0.9209
29056/60000 [=============>................] - ETA: 58s - loss: 0.2583 - categorical_accuracy: 0.9209
29088/60000 [=============>................] - ETA: 57s - loss: 0.2580 - categorical_accuracy: 0.9210
29120/60000 [=============>................] - ETA: 57s - loss: 0.2577 - categorical_accuracy: 0.9211
29152/60000 [=============>................] - ETA: 57s - loss: 0.2576 - categorical_accuracy: 0.9211
29184/60000 [=============>................] - ETA: 57s - loss: 0.2573 - categorical_accuracy: 0.9212
29216/60000 [=============>................] - ETA: 57s - loss: 0.2574 - categorical_accuracy: 0.9212
29248/60000 [=============>................] - ETA: 57s - loss: 0.2572 - categorical_accuracy: 0.9213
29280/60000 [=============>................] - ETA: 57s - loss: 0.2571 - categorical_accuracy: 0.9213
29312/60000 [=============>................] - ETA: 57s - loss: 0.2568 - categorical_accuracy: 0.9214
29344/60000 [=============>................] - ETA: 57s - loss: 0.2565 - categorical_accuracy: 0.9215
29376/60000 [=============>................] - ETA: 57s - loss: 0.2563 - categorical_accuracy: 0.9216
29408/60000 [=============>................] - ETA: 57s - loss: 0.2561 - categorical_accuracy: 0.9216
29440/60000 [=============>................] - ETA: 57s - loss: 0.2560 - categorical_accuracy: 0.9216
29472/60000 [=============>................] - ETA: 57s - loss: 0.2558 - categorical_accuracy: 0.9217
29504/60000 [=============>................] - ETA: 57s - loss: 0.2557 - categorical_accuracy: 0.9217
29536/60000 [=============>................] - ETA: 57s - loss: 0.2556 - categorical_accuracy: 0.9217
29568/60000 [=============>................] - ETA: 57s - loss: 0.2556 - categorical_accuracy: 0.9217
29600/60000 [=============>................] - ETA: 57s - loss: 0.2555 - categorical_accuracy: 0.9217
29632/60000 [=============>................] - ETA: 56s - loss: 0.2554 - categorical_accuracy: 0.9217
29664/60000 [=============>................] - ETA: 56s - loss: 0.2552 - categorical_accuracy: 0.9218
29696/60000 [=============>................] - ETA: 56s - loss: 0.2550 - categorical_accuracy: 0.9218
29728/60000 [=============>................] - ETA: 56s - loss: 0.2548 - categorical_accuracy: 0.9219
29760/60000 [=============>................] - ETA: 56s - loss: 0.2546 - categorical_accuracy: 0.9220
29792/60000 [=============>................] - ETA: 56s - loss: 0.2543 - categorical_accuracy: 0.9221
29824/60000 [=============>................] - ETA: 56s - loss: 0.2541 - categorical_accuracy: 0.9221
29856/60000 [=============>................] - ETA: 56s - loss: 0.2539 - categorical_accuracy: 0.9222
29888/60000 [=============>................] - ETA: 56s - loss: 0.2537 - categorical_accuracy: 0.9223
29920/60000 [=============>................] - ETA: 56s - loss: 0.2535 - categorical_accuracy: 0.9223
29952/60000 [=============>................] - ETA: 56s - loss: 0.2537 - categorical_accuracy: 0.9223
29984/60000 [=============>................] - ETA: 56s - loss: 0.2536 - categorical_accuracy: 0.9223
30016/60000 [==============>...............] - ETA: 56s - loss: 0.2534 - categorical_accuracy: 0.9224
30048/60000 [==============>...............] - ETA: 56s - loss: 0.2531 - categorical_accuracy: 0.9225
30080/60000 [==============>...............] - ETA: 56s - loss: 0.2529 - categorical_accuracy: 0.9225
30112/60000 [==============>...............] - ETA: 56s - loss: 0.2531 - categorical_accuracy: 0.9225
30144/60000 [==============>...............] - ETA: 55s - loss: 0.2528 - categorical_accuracy: 0.9225
30176/60000 [==============>...............] - ETA: 55s - loss: 0.2526 - categorical_accuracy: 0.9226
30208/60000 [==============>...............] - ETA: 55s - loss: 0.2524 - categorical_accuracy: 0.9227
30240/60000 [==============>...............] - ETA: 55s - loss: 0.2521 - categorical_accuracy: 0.9228
30272/60000 [==============>...............] - ETA: 55s - loss: 0.2519 - categorical_accuracy: 0.9229
30304/60000 [==============>...............] - ETA: 55s - loss: 0.2518 - categorical_accuracy: 0.9229
30336/60000 [==============>...............] - ETA: 55s - loss: 0.2520 - categorical_accuracy: 0.9229
30368/60000 [==============>...............] - ETA: 55s - loss: 0.2518 - categorical_accuracy: 0.9230
30400/60000 [==============>...............] - ETA: 55s - loss: 0.2517 - categorical_accuracy: 0.9230
30432/60000 [==============>...............] - ETA: 55s - loss: 0.2518 - categorical_accuracy: 0.9230
30464/60000 [==============>...............] - ETA: 55s - loss: 0.2516 - categorical_accuracy: 0.9231
30496/60000 [==============>...............] - ETA: 55s - loss: 0.2514 - categorical_accuracy: 0.9232
30528/60000 [==============>...............] - ETA: 55s - loss: 0.2512 - categorical_accuracy: 0.9232
30560/60000 [==============>...............] - ETA: 55s - loss: 0.2514 - categorical_accuracy: 0.9232
30592/60000 [==============>...............] - ETA: 55s - loss: 0.2512 - categorical_accuracy: 0.9232
30624/60000 [==============>...............] - ETA: 55s - loss: 0.2510 - categorical_accuracy: 0.9233
30656/60000 [==============>...............] - ETA: 54s - loss: 0.2509 - categorical_accuracy: 0.9233
30688/60000 [==============>...............] - ETA: 54s - loss: 0.2506 - categorical_accuracy: 0.9234
30720/60000 [==============>...............] - ETA: 54s - loss: 0.2504 - categorical_accuracy: 0.9234
30752/60000 [==============>...............] - ETA: 54s - loss: 0.2502 - categorical_accuracy: 0.9235
30784/60000 [==============>...............] - ETA: 54s - loss: 0.2501 - categorical_accuracy: 0.9235
30816/60000 [==============>...............] - ETA: 54s - loss: 0.2500 - categorical_accuracy: 0.9235
30848/60000 [==============>...............] - ETA: 54s - loss: 0.2498 - categorical_accuracy: 0.9236
30880/60000 [==============>...............] - ETA: 54s - loss: 0.2496 - categorical_accuracy: 0.9236
30912/60000 [==============>...............] - ETA: 54s - loss: 0.2496 - categorical_accuracy: 0.9236
30944/60000 [==============>...............] - ETA: 54s - loss: 0.2495 - categorical_accuracy: 0.9237
30976/60000 [==============>...............] - ETA: 54s - loss: 0.2493 - categorical_accuracy: 0.9237
31008/60000 [==============>...............] - ETA: 54s - loss: 0.2491 - categorical_accuracy: 0.9237
31040/60000 [==============>...............] - ETA: 54s - loss: 0.2490 - categorical_accuracy: 0.9238
31072/60000 [==============>...............] - ETA: 54s - loss: 0.2489 - categorical_accuracy: 0.9238
31104/60000 [==============>...............] - ETA: 54s - loss: 0.2487 - categorical_accuracy: 0.9239
31136/60000 [==============>...............] - ETA: 54s - loss: 0.2485 - categorical_accuracy: 0.9239
31168/60000 [==============>...............] - ETA: 54s - loss: 0.2482 - categorical_accuracy: 0.9240
31200/60000 [==============>...............] - ETA: 53s - loss: 0.2481 - categorical_accuracy: 0.9241
31232/60000 [==============>...............] - ETA: 53s - loss: 0.2478 - categorical_accuracy: 0.9241
31264/60000 [==============>...............] - ETA: 53s - loss: 0.2479 - categorical_accuracy: 0.9241
31296/60000 [==============>...............] - ETA: 53s - loss: 0.2477 - categorical_accuracy: 0.9242
31328/60000 [==============>...............] - ETA: 53s - loss: 0.2475 - categorical_accuracy: 0.9243
31360/60000 [==============>...............] - ETA: 53s - loss: 0.2473 - categorical_accuracy: 0.9244
31392/60000 [==============>...............] - ETA: 53s - loss: 0.2471 - categorical_accuracy: 0.9244
31424/60000 [==============>...............] - ETA: 53s - loss: 0.2471 - categorical_accuracy: 0.9245
31456/60000 [==============>...............] - ETA: 53s - loss: 0.2470 - categorical_accuracy: 0.9245
31488/60000 [==============>...............] - ETA: 53s - loss: 0.2469 - categorical_accuracy: 0.9245
31520/60000 [==============>...............] - ETA: 53s - loss: 0.2467 - categorical_accuracy: 0.9246
31552/60000 [==============>...............] - ETA: 53s - loss: 0.2467 - categorical_accuracy: 0.9245
31584/60000 [==============>...............] - ETA: 53s - loss: 0.2465 - categorical_accuracy: 0.9246
31616/60000 [==============>...............] - ETA: 53s - loss: 0.2463 - categorical_accuracy: 0.9246
31648/60000 [==============>...............] - ETA: 53s - loss: 0.2462 - categorical_accuracy: 0.9247
31680/60000 [==============>...............] - ETA: 53s - loss: 0.2461 - categorical_accuracy: 0.9247
31712/60000 [==============>...............] - ETA: 53s - loss: 0.2460 - categorical_accuracy: 0.9247
31744/60000 [==============>...............] - ETA: 52s - loss: 0.2459 - categorical_accuracy: 0.9247
31776/60000 [==============>...............] - ETA: 52s - loss: 0.2458 - categorical_accuracy: 0.9248
31808/60000 [==============>...............] - ETA: 52s - loss: 0.2457 - categorical_accuracy: 0.9248
31840/60000 [==============>...............] - ETA: 52s - loss: 0.2456 - categorical_accuracy: 0.9248
31872/60000 [==============>...............] - ETA: 52s - loss: 0.2455 - categorical_accuracy: 0.9248
31904/60000 [==============>...............] - ETA: 52s - loss: 0.2454 - categorical_accuracy: 0.9248
31936/60000 [==============>...............] - ETA: 52s - loss: 0.2454 - categorical_accuracy: 0.9248
31968/60000 [==============>...............] - ETA: 52s - loss: 0.2454 - categorical_accuracy: 0.9249
32000/60000 [===============>..............] - ETA: 52s - loss: 0.2453 - categorical_accuracy: 0.9249
32032/60000 [===============>..............] - ETA: 52s - loss: 0.2452 - categorical_accuracy: 0.9250
32064/60000 [===============>..............] - ETA: 52s - loss: 0.2449 - categorical_accuracy: 0.9250
32096/60000 [===============>..............] - ETA: 52s - loss: 0.2448 - categorical_accuracy: 0.9251
32128/60000 [===============>..............] - ETA: 52s - loss: 0.2446 - categorical_accuracy: 0.9251
32160/60000 [===============>..............] - ETA: 52s - loss: 0.2445 - categorical_accuracy: 0.9252
32192/60000 [===============>..............] - ETA: 52s - loss: 0.2444 - categorical_accuracy: 0.9252
32224/60000 [===============>..............] - ETA: 52s - loss: 0.2443 - categorical_accuracy: 0.9252
32256/60000 [===============>..............] - ETA: 51s - loss: 0.2441 - categorical_accuracy: 0.9253
32288/60000 [===============>..............] - ETA: 51s - loss: 0.2440 - categorical_accuracy: 0.9253
32320/60000 [===============>..............] - ETA: 51s - loss: 0.2438 - categorical_accuracy: 0.9254
32352/60000 [===============>..............] - ETA: 51s - loss: 0.2438 - categorical_accuracy: 0.9254
32384/60000 [===============>..............] - ETA: 51s - loss: 0.2436 - categorical_accuracy: 0.9254
32416/60000 [===============>..............] - ETA: 51s - loss: 0.2435 - categorical_accuracy: 0.9255
32448/60000 [===============>..............] - ETA: 51s - loss: 0.2432 - categorical_accuracy: 0.9255
32480/60000 [===============>..............] - ETA: 51s - loss: 0.2431 - categorical_accuracy: 0.9256
32512/60000 [===============>..............] - ETA: 51s - loss: 0.2429 - categorical_accuracy: 0.9256
32544/60000 [===============>..............] - ETA: 51s - loss: 0.2428 - categorical_accuracy: 0.9256
32576/60000 [===============>..............] - ETA: 51s - loss: 0.2427 - categorical_accuracy: 0.9257
32608/60000 [===============>..............] - ETA: 51s - loss: 0.2426 - categorical_accuracy: 0.9257
32640/60000 [===============>..............] - ETA: 51s - loss: 0.2424 - categorical_accuracy: 0.9258
32672/60000 [===============>..............] - ETA: 51s - loss: 0.2422 - categorical_accuracy: 0.9257
32704/60000 [===============>..............] - ETA: 51s - loss: 0.2424 - categorical_accuracy: 0.9258
32736/60000 [===============>..............] - ETA: 51s - loss: 0.2423 - categorical_accuracy: 0.9258
32768/60000 [===============>..............] - ETA: 51s - loss: 0.2422 - categorical_accuracy: 0.9258
32800/60000 [===============>..............] - ETA: 50s - loss: 0.2420 - categorical_accuracy: 0.9259
32832/60000 [===============>..............] - ETA: 50s - loss: 0.2418 - categorical_accuracy: 0.9260
32864/60000 [===============>..............] - ETA: 50s - loss: 0.2417 - categorical_accuracy: 0.9260
32896/60000 [===============>..............] - ETA: 50s - loss: 0.2416 - categorical_accuracy: 0.9260
32928/60000 [===============>..............] - ETA: 50s - loss: 0.2416 - categorical_accuracy: 0.9260
32960/60000 [===============>..............] - ETA: 50s - loss: 0.2416 - categorical_accuracy: 0.9260
32992/60000 [===============>..............] - ETA: 50s - loss: 0.2413 - categorical_accuracy: 0.9261
33024/60000 [===============>..............] - ETA: 50s - loss: 0.2413 - categorical_accuracy: 0.9261
33056/60000 [===============>..............] - ETA: 50s - loss: 0.2412 - categorical_accuracy: 0.9261
33088/60000 [===============>..............] - ETA: 50s - loss: 0.2410 - categorical_accuracy: 0.9262
33120/60000 [===============>..............] - ETA: 50s - loss: 0.2409 - categorical_accuracy: 0.9262
33152/60000 [===============>..............] - ETA: 50s - loss: 0.2408 - categorical_accuracy: 0.9263
33184/60000 [===============>..............] - ETA: 50s - loss: 0.2406 - categorical_accuracy: 0.9264
33216/60000 [===============>..............] - ETA: 50s - loss: 0.2405 - categorical_accuracy: 0.9264
33248/60000 [===============>..............] - ETA: 50s - loss: 0.2403 - categorical_accuracy: 0.9264
33280/60000 [===============>..............] - ETA: 50s - loss: 0.2401 - categorical_accuracy: 0.9265
33312/60000 [===============>..............] - ETA: 50s - loss: 0.2399 - categorical_accuracy: 0.9266
33344/60000 [===============>..............] - ETA: 49s - loss: 0.2397 - categorical_accuracy: 0.9266
33376/60000 [===============>..............] - ETA: 49s - loss: 0.2395 - categorical_accuracy: 0.9267
33408/60000 [===============>..............] - ETA: 49s - loss: 0.2394 - categorical_accuracy: 0.9267
33440/60000 [===============>..............] - ETA: 49s - loss: 0.2394 - categorical_accuracy: 0.9267
33472/60000 [===============>..............] - ETA: 49s - loss: 0.2393 - categorical_accuracy: 0.9267
33504/60000 [===============>..............] - ETA: 49s - loss: 0.2391 - categorical_accuracy: 0.9268
33536/60000 [===============>..............] - ETA: 49s - loss: 0.2392 - categorical_accuracy: 0.9268
33568/60000 [===============>..............] - ETA: 49s - loss: 0.2391 - categorical_accuracy: 0.9268
33600/60000 [===============>..............] - ETA: 49s - loss: 0.2389 - categorical_accuracy: 0.9268
33632/60000 [===============>..............] - ETA: 49s - loss: 0.2390 - categorical_accuracy: 0.9268
33664/60000 [===============>..............] - ETA: 49s - loss: 0.2388 - categorical_accuracy: 0.9269
33696/60000 [===============>..............] - ETA: 49s - loss: 0.2390 - categorical_accuracy: 0.9269
33728/60000 [===============>..............] - ETA: 49s - loss: 0.2389 - categorical_accuracy: 0.9269
33760/60000 [===============>..............] - ETA: 49s - loss: 0.2389 - categorical_accuracy: 0.9269
33792/60000 [===============>..............] - ETA: 49s - loss: 0.2388 - categorical_accuracy: 0.9269
33824/60000 [===============>..............] - ETA: 49s - loss: 0.2387 - categorical_accuracy: 0.9270
33856/60000 [===============>..............] - ETA: 48s - loss: 0.2387 - categorical_accuracy: 0.9270
33888/60000 [===============>..............] - ETA: 48s - loss: 0.2385 - categorical_accuracy: 0.9271
33920/60000 [===============>..............] - ETA: 48s - loss: 0.2384 - categorical_accuracy: 0.9271
33952/60000 [===============>..............] - ETA: 48s - loss: 0.2383 - categorical_accuracy: 0.9271
33984/60000 [===============>..............] - ETA: 48s - loss: 0.2383 - categorical_accuracy: 0.9271
34016/60000 [================>.............] - ETA: 48s - loss: 0.2382 - categorical_accuracy: 0.9271
34048/60000 [================>.............] - ETA: 48s - loss: 0.2381 - categorical_accuracy: 0.9271
34080/60000 [================>.............] - ETA: 48s - loss: 0.2380 - categorical_accuracy: 0.9271
34112/60000 [================>.............] - ETA: 48s - loss: 0.2379 - categorical_accuracy: 0.9272
34144/60000 [================>.............] - ETA: 48s - loss: 0.2377 - categorical_accuracy: 0.9272
34176/60000 [================>.............] - ETA: 48s - loss: 0.2377 - categorical_accuracy: 0.9272
34208/60000 [================>.............] - ETA: 48s - loss: 0.2376 - categorical_accuracy: 0.9272
34240/60000 [================>.............] - ETA: 48s - loss: 0.2374 - categorical_accuracy: 0.9273
34272/60000 [================>.............] - ETA: 48s - loss: 0.2373 - categorical_accuracy: 0.9273
34304/60000 [================>.............] - ETA: 48s - loss: 0.2372 - categorical_accuracy: 0.9273
34336/60000 [================>.............] - ETA: 48s - loss: 0.2372 - categorical_accuracy: 0.9273
34368/60000 [================>.............] - ETA: 48s - loss: 0.2371 - categorical_accuracy: 0.9274
34400/60000 [================>.............] - ETA: 47s - loss: 0.2370 - categorical_accuracy: 0.9274
34432/60000 [================>.............] - ETA: 47s - loss: 0.2368 - categorical_accuracy: 0.9275
34464/60000 [================>.............] - ETA: 47s - loss: 0.2366 - categorical_accuracy: 0.9275
34496/60000 [================>.............] - ETA: 47s - loss: 0.2364 - categorical_accuracy: 0.9276
34528/60000 [================>.............] - ETA: 47s - loss: 0.2363 - categorical_accuracy: 0.9276
34560/60000 [================>.............] - ETA: 47s - loss: 0.2361 - categorical_accuracy: 0.9277
34592/60000 [================>.............] - ETA: 47s - loss: 0.2359 - categorical_accuracy: 0.9277
34624/60000 [================>.............] - ETA: 47s - loss: 0.2359 - categorical_accuracy: 0.9277
34656/60000 [================>.............] - ETA: 47s - loss: 0.2357 - categorical_accuracy: 0.9278
34688/60000 [================>.............] - ETA: 47s - loss: 0.2356 - categorical_accuracy: 0.9278
34720/60000 [================>.............] - ETA: 47s - loss: 0.2355 - categorical_accuracy: 0.9278
34752/60000 [================>.............] - ETA: 47s - loss: 0.2356 - categorical_accuracy: 0.9278
34784/60000 [================>.............] - ETA: 47s - loss: 0.2354 - categorical_accuracy: 0.9279
34816/60000 [================>.............] - ETA: 47s - loss: 0.2352 - categorical_accuracy: 0.9279
34848/60000 [================>.............] - ETA: 47s - loss: 0.2350 - categorical_accuracy: 0.9280
34880/60000 [================>.............] - ETA: 47s - loss: 0.2351 - categorical_accuracy: 0.9280
34912/60000 [================>.............] - ETA: 46s - loss: 0.2352 - categorical_accuracy: 0.9280
34944/60000 [================>.............] - ETA: 46s - loss: 0.2350 - categorical_accuracy: 0.9280
34976/60000 [================>.............] - ETA: 46s - loss: 0.2350 - categorical_accuracy: 0.9281
35008/60000 [================>.............] - ETA: 46s - loss: 0.2348 - categorical_accuracy: 0.9281
35040/60000 [================>.............] - ETA: 46s - loss: 0.2348 - categorical_accuracy: 0.9282
35072/60000 [================>.............] - ETA: 46s - loss: 0.2346 - categorical_accuracy: 0.9282
35104/60000 [================>.............] - ETA: 46s - loss: 0.2346 - categorical_accuracy: 0.9282
35136/60000 [================>.............] - ETA: 46s - loss: 0.2344 - categorical_accuracy: 0.9283
35168/60000 [================>.............] - ETA: 46s - loss: 0.2343 - categorical_accuracy: 0.9283
35200/60000 [================>.............] - ETA: 46s - loss: 0.2342 - categorical_accuracy: 0.9283
35232/60000 [================>.............] - ETA: 46s - loss: 0.2341 - categorical_accuracy: 0.9284
35264/60000 [================>.............] - ETA: 46s - loss: 0.2342 - categorical_accuracy: 0.9283
35296/60000 [================>.............] - ETA: 46s - loss: 0.2340 - categorical_accuracy: 0.9284
35328/60000 [================>.............] - ETA: 46s - loss: 0.2338 - categorical_accuracy: 0.9285
35360/60000 [================>.............] - ETA: 46s - loss: 0.2336 - categorical_accuracy: 0.9285
35392/60000 [================>.............] - ETA: 46s - loss: 0.2337 - categorical_accuracy: 0.9285
35424/60000 [================>.............] - ETA: 46s - loss: 0.2336 - categorical_accuracy: 0.9286
35456/60000 [================>.............] - ETA: 45s - loss: 0.2334 - categorical_accuracy: 0.9286
35488/60000 [================>.............] - ETA: 45s - loss: 0.2336 - categorical_accuracy: 0.9286
35520/60000 [================>.............] - ETA: 45s - loss: 0.2336 - categorical_accuracy: 0.9287
35552/60000 [================>.............] - ETA: 45s - loss: 0.2334 - categorical_accuracy: 0.9287
35584/60000 [================>.............] - ETA: 45s - loss: 0.2333 - categorical_accuracy: 0.9287
35616/60000 [================>.............] - ETA: 45s - loss: 0.2331 - categorical_accuracy: 0.9288
35648/60000 [================>.............] - ETA: 45s - loss: 0.2333 - categorical_accuracy: 0.9287
35680/60000 [================>.............] - ETA: 45s - loss: 0.2331 - categorical_accuracy: 0.9288
35712/60000 [================>.............] - ETA: 45s - loss: 0.2330 - categorical_accuracy: 0.9288
35744/60000 [================>.............] - ETA: 45s - loss: 0.2328 - categorical_accuracy: 0.9289
35776/60000 [================>.............] - ETA: 45s - loss: 0.2326 - categorical_accuracy: 0.9289
35808/60000 [================>.............] - ETA: 45s - loss: 0.2325 - categorical_accuracy: 0.9290
35840/60000 [================>.............] - ETA: 45s - loss: 0.2324 - categorical_accuracy: 0.9290
35872/60000 [================>.............] - ETA: 45s - loss: 0.2322 - categorical_accuracy: 0.9291
35904/60000 [================>.............] - ETA: 45s - loss: 0.2321 - categorical_accuracy: 0.9291
35936/60000 [================>.............] - ETA: 45s - loss: 0.2319 - categorical_accuracy: 0.9292
35968/60000 [================>.............] - ETA: 44s - loss: 0.2319 - categorical_accuracy: 0.9292
36000/60000 [=================>............] - ETA: 44s - loss: 0.2318 - categorical_accuracy: 0.9292
36032/60000 [=================>............] - ETA: 44s - loss: 0.2317 - categorical_accuracy: 0.9292
36064/60000 [=================>............] - ETA: 44s - loss: 0.2317 - categorical_accuracy: 0.9293
36096/60000 [=================>............] - ETA: 44s - loss: 0.2316 - categorical_accuracy: 0.9293
36128/60000 [=================>............] - ETA: 44s - loss: 0.2315 - categorical_accuracy: 0.9293
36160/60000 [=================>............] - ETA: 44s - loss: 0.2314 - categorical_accuracy: 0.9293
36192/60000 [=================>............] - ETA: 44s - loss: 0.2313 - categorical_accuracy: 0.9293
36224/60000 [=================>............] - ETA: 44s - loss: 0.2311 - categorical_accuracy: 0.9294
36256/60000 [=================>............] - ETA: 44s - loss: 0.2310 - categorical_accuracy: 0.9294
36288/60000 [=================>............] - ETA: 44s - loss: 0.2309 - categorical_accuracy: 0.9294
36320/60000 [=================>............] - ETA: 44s - loss: 0.2308 - categorical_accuracy: 0.9294
36352/60000 [=================>............] - ETA: 44s - loss: 0.2306 - categorical_accuracy: 0.9295
36384/60000 [=================>............] - ETA: 44s - loss: 0.2305 - categorical_accuracy: 0.9295
36416/60000 [=================>............] - ETA: 44s - loss: 0.2305 - categorical_accuracy: 0.9295
36448/60000 [=================>............] - ETA: 44s - loss: 0.2304 - categorical_accuracy: 0.9295
36480/60000 [=================>............] - ETA: 44s - loss: 0.2302 - categorical_accuracy: 0.9296
36512/60000 [=================>............] - ETA: 43s - loss: 0.2300 - categorical_accuracy: 0.9296
36544/60000 [=================>............] - ETA: 43s - loss: 0.2299 - categorical_accuracy: 0.9297
36576/60000 [=================>............] - ETA: 43s - loss: 0.2297 - categorical_accuracy: 0.9297
36608/60000 [=================>............] - ETA: 43s - loss: 0.2297 - categorical_accuracy: 0.9297
36640/60000 [=================>............] - ETA: 43s - loss: 0.2297 - categorical_accuracy: 0.9298
36672/60000 [=================>............] - ETA: 43s - loss: 0.2295 - categorical_accuracy: 0.9298
36704/60000 [=================>............] - ETA: 43s - loss: 0.2294 - categorical_accuracy: 0.9298
36736/60000 [=================>............] - ETA: 43s - loss: 0.2292 - categorical_accuracy: 0.9299
36768/60000 [=================>............] - ETA: 43s - loss: 0.2291 - categorical_accuracy: 0.9299
36800/60000 [=================>............] - ETA: 43s - loss: 0.2290 - categorical_accuracy: 0.9300
36832/60000 [=================>............] - ETA: 43s - loss: 0.2288 - categorical_accuracy: 0.9300
36864/60000 [=================>............] - ETA: 43s - loss: 0.2288 - categorical_accuracy: 0.9300
36896/60000 [=================>............] - ETA: 43s - loss: 0.2286 - categorical_accuracy: 0.9301
36928/60000 [=================>............] - ETA: 43s - loss: 0.2285 - categorical_accuracy: 0.9301
36960/60000 [=================>............] - ETA: 43s - loss: 0.2284 - categorical_accuracy: 0.9302
36992/60000 [=================>............] - ETA: 43s - loss: 0.2283 - categorical_accuracy: 0.9302
37024/60000 [=================>............] - ETA: 42s - loss: 0.2284 - categorical_accuracy: 0.9302
37056/60000 [=================>............] - ETA: 42s - loss: 0.2282 - categorical_accuracy: 0.9302
37088/60000 [=================>............] - ETA: 42s - loss: 0.2281 - categorical_accuracy: 0.9302
37120/60000 [=================>............] - ETA: 42s - loss: 0.2281 - categorical_accuracy: 0.9302
37152/60000 [=================>............] - ETA: 42s - loss: 0.2280 - categorical_accuracy: 0.9303
37184/60000 [=================>............] - ETA: 42s - loss: 0.2279 - categorical_accuracy: 0.9303
37216/60000 [=================>............] - ETA: 42s - loss: 0.2283 - categorical_accuracy: 0.9302
37248/60000 [=================>............] - ETA: 42s - loss: 0.2282 - categorical_accuracy: 0.9303
37280/60000 [=================>............] - ETA: 42s - loss: 0.2282 - categorical_accuracy: 0.9303
37312/60000 [=================>............] - ETA: 42s - loss: 0.2282 - categorical_accuracy: 0.9303
37344/60000 [=================>............] - ETA: 42s - loss: 0.2281 - categorical_accuracy: 0.9303
37376/60000 [=================>............] - ETA: 42s - loss: 0.2279 - categorical_accuracy: 0.9304
37408/60000 [=================>............] - ETA: 42s - loss: 0.2278 - categorical_accuracy: 0.9304
37440/60000 [=================>............] - ETA: 42s - loss: 0.2276 - categorical_accuracy: 0.9304
37472/60000 [=================>............] - ETA: 42s - loss: 0.2275 - categorical_accuracy: 0.9305
37504/60000 [=================>............] - ETA: 42s - loss: 0.2273 - categorical_accuracy: 0.9305
37536/60000 [=================>............] - ETA: 42s - loss: 0.2271 - categorical_accuracy: 0.9306
37568/60000 [=================>............] - ETA: 41s - loss: 0.2270 - categorical_accuracy: 0.9307
37600/60000 [=================>............] - ETA: 41s - loss: 0.2271 - categorical_accuracy: 0.9306
37632/60000 [=================>............] - ETA: 41s - loss: 0.2273 - categorical_accuracy: 0.9306
37664/60000 [=================>............] - ETA: 41s - loss: 0.2272 - categorical_accuracy: 0.9306
37696/60000 [=================>............] - ETA: 41s - loss: 0.2270 - categorical_accuracy: 0.9307
37728/60000 [=================>............] - ETA: 41s - loss: 0.2271 - categorical_accuracy: 0.9306
37760/60000 [=================>............] - ETA: 41s - loss: 0.2269 - categorical_accuracy: 0.9307
37792/60000 [=================>............] - ETA: 41s - loss: 0.2267 - categorical_accuracy: 0.9307
37824/60000 [=================>............] - ETA: 41s - loss: 0.2266 - categorical_accuracy: 0.9307
37856/60000 [=================>............] - ETA: 41s - loss: 0.2265 - categorical_accuracy: 0.9308
37888/60000 [=================>............] - ETA: 41s - loss: 0.2264 - categorical_accuracy: 0.9308
37920/60000 [=================>............] - ETA: 41s - loss: 0.2262 - categorical_accuracy: 0.9309
37952/60000 [=================>............] - ETA: 41s - loss: 0.2261 - categorical_accuracy: 0.9309
37984/60000 [=================>............] - ETA: 41s - loss: 0.2262 - categorical_accuracy: 0.9309
38016/60000 [==================>...........] - ETA: 41s - loss: 0.2260 - categorical_accuracy: 0.9309
38048/60000 [==================>...........] - ETA: 41s - loss: 0.2259 - categorical_accuracy: 0.9309
38080/60000 [==================>...........] - ETA: 41s - loss: 0.2258 - categorical_accuracy: 0.9310
38112/60000 [==================>...........] - ETA: 40s - loss: 0.2256 - categorical_accuracy: 0.9310
38144/60000 [==================>...........] - ETA: 40s - loss: 0.2255 - categorical_accuracy: 0.9311
38176/60000 [==================>...........] - ETA: 40s - loss: 0.2254 - categorical_accuracy: 0.9311
38208/60000 [==================>...........] - ETA: 40s - loss: 0.2253 - categorical_accuracy: 0.9311
38240/60000 [==================>...........] - ETA: 40s - loss: 0.2252 - categorical_accuracy: 0.9311
38272/60000 [==================>...........] - ETA: 40s - loss: 0.2251 - categorical_accuracy: 0.9312
38304/60000 [==================>...........] - ETA: 40s - loss: 0.2250 - categorical_accuracy: 0.9312
38336/60000 [==================>...........] - ETA: 40s - loss: 0.2249 - categorical_accuracy: 0.9312
38368/60000 [==================>...........] - ETA: 40s - loss: 0.2248 - categorical_accuracy: 0.9313
38400/60000 [==================>...........] - ETA: 40s - loss: 0.2247 - categorical_accuracy: 0.9313
38432/60000 [==================>...........] - ETA: 40s - loss: 0.2247 - categorical_accuracy: 0.9313
38464/60000 [==================>...........] - ETA: 40s - loss: 0.2246 - categorical_accuracy: 0.9313
38496/60000 [==================>...........] - ETA: 40s - loss: 0.2245 - categorical_accuracy: 0.9313
38528/60000 [==================>...........] - ETA: 40s - loss: 0.2244 - categorical_accuracy: 0.9313
38560/60000 [==================>...........] - ETA: 40s - loss: 0.2242 - categorical_accuracy: 0.9314
38592/60000 [==================>...........] - ETA: 40s - loss: 0.2241 - categorical_accuracy: 0.9314
38624/60000 [==================>...........] - ETA: 40s - loss: 0.2239 - categorical_accuracy: 0.9315
38656/60000 [==================>...........] - ETA: 39s - loss: 0.2238 - categorical_accuracy: 0.9315
38688/60000 [==================>...........] - ETA: 39s - loss: 0.2237 - categorical_accuracy: 0.9315
38720/60000 [==================>...........] - ETA: 39s - loss: 0.2238 - categorical_accuracy: 0.9315
38752/60000 [==================>...........] - ETA: 39s - loss: 0.2237 - categorical_accuracy: 0.9316
38784/60000 [==================>...........] - ETA: 39s - loss: 0.2236 - categorical_accuracy: 0.9316
38816/60000 [==================>...........] - ETA: 39s - loss: 0.2239 - categorical_accuracy: 0.9316
38848/60000 [==================>...........] - ETA: 39s - loss: 0.2237 - categorical_accuracy: 0.9316
38880/60000 [==================>...........] - ETA: 39s - loss: 0.2236 - categorical_accuracy: 0.9317
38912/60000 [==================>...........] - ETA: 39s - loss: 0.2234 - categorical_accuracy: 0.9317
38944/60000 [==================>...........] - ETA: 39s - loss: 0.2233 - categorical_accuracy: 0.9318
38976/60000 [==================>...........] - ETA: 39s - loss: 0.2233 - categorical_accuracy: 0.9318
39008/60000 [==================>...........] - ETA: 39s - loss: 0.2232 - categorical_accuracy: 0.9319
39040/60000 [==================>...........] - ETA: 39s - loss: 0.2231 - categorical_accuracy: 0.9319
39072/60000 [==================>...........] - ETA: 39s - loss: 0.2231 - categorical_accuracy: 0.9319
39104/60000 [==================>...........] - ETA: 39s - loss: 0.2230 - categorical_accuracy: 0.9320
39136/60000 [==================>...........] - ETA: 39s - loss: 0.2229 - categorical_accuracy: 0.9320
39168/60000 [==================>...........] - ETA: 38s - loss: 0.2227 - categorical_accuracy: 0.9320
39200/60000 [==================>...........] - ETA: 38s - loss: 0.2228 - categorical_accuracy: 0.9320
39232/60000 [==================>...........] - ETA: 38s - loss: 0.2228 - categorical_accuracy: 0.9320
39264/60000 [==================>...........] - ETA: 38s - loss: 0.2227 - categorical_accuracy: 0.9320
39296/60000 [==================>...........] - ETA: 38s - loss: 0.2225 - categorical_accuracy: 0.9321
39328/60000 [==================>...........] - ETA: 38s - loss: 0.2224 - categorical_accuracy: 0.9321
39360/60000 [==================>...........] - ETA: 38s - loss: 0.2224 - categorical_accuracy: 0.9321
39392/60000 [==================>...........] - ETA: 38s - loss: 0.2223 - categorical_accuracy: 0.9321
39424/60000 [==================>...........] - ETA: 38s - loss: 0.2222 - categorical_accuracy: 0.9322
39456/60000 [==================>...........] - ETA: 38s - loss: 0.2221 - categorical_accuracy: 0.9322
39488/60000 [==================>...........] - ETA: 38s - loss: 0.2221 - categorical_accuracy: 0.9323
39520/60000 [==================>...........] - ETA: 38s - loss: 0.2220 - categorical_accuracy: 0.9323
39552/60000 [==================>...........] - ETA: 38s - loss: 0.2219 - categorical_accuracy: 0.9323
39584/60000 [==================>...........] - ETA: 38s - loss: 0.2218 - categorical_accuracy: 0.9323
39616/60000 [==================>...........] - ETA: 38s - loss: 0.2217 - categorical_accuracy: 0.9323
39648/60000 [==================>...........] - ETA: 38s - loss: 0.2216 - categorical_accuracy: 0.9324
39680/60000 [==================>...........] - ETA: 38s - loss: 0.2215 - categorical_accuracy: 0.9324
39712/60000 [==================>...........] - ETA: 37s - loss: 0.2214 - categorical_accuracy: 0.9324
39744/60000 [==================>...........] - ETA: 37s - loss: 0.2216 - categorical_accuracy: 0.9323
39776/60000 [==================>...........] - ETA: 37s - loss: 0.2215 - categorical_accuracy: 0.9324
39808/60000 [==================>...........] - ETA: 37s - loss: 0.2214 - categorical_accuracy: 0.9324
39840/60000 [==================>...........] - ETA: 37s - loss: 0.2214 - categorical_accuracy: 0.9324
39872/60000 [==================>...........] - ETA: 37s - loss: 0.2213 - categorical_accuracy: 0.9324
39904/60000 [==================>...........] - ETA: 37s - loss: 0.2212 - categorical_accuracy: 0.9325
39936/60000 [==================>...........] - ETA: 37s - loss: 0.2210 - categorical_accuracy: 0.9325
39968/60000 [==================>...........] - ETA: 37s - loss: 0.2210 - categorical_accuracy: 0.9325
40000/60000 [===================>..........] - ETA: 37s - loss: 0.2211 - categorical_accuracy: 0.9326
40032/60000 [===================>..........] - ETA: 37s - loss: 0.2209 - categorical_accuracy: 0.9326
40064/60000 [===================>..........] - ETA: 37s - loss: 0.2208 - categorical_accuracy: 0.9326
40096/60000 [===================>..........] - ETA: 37s - loss: 0.2207 - categorical_accuracy: 0.9327
40128/60000 [===================>..........] - ETA: 37s - loss: 0.2206 - categorical_accuracy: 0.9327
40160/60000 [===================>..........] - ETA: 37s - loss: 0.2206 - categorical_accuracy: 0.9327
40192/60000 [===================>..........] - ETA: 37s - loss: 0.2205 - categorical_accuracy: 0.9327
40224/60000 [===================>..........] - ETA: 36s - loss: 0.2204 - categorical_accuracy: 0.9328
40256/60000 [===================>..........] - ETA: 36s - loss: 0.2203 - categorical_accuracy: 0.9328
40288/60000 [===================>..........] - ETA: 36s - loss: 0.2202 - categorical_accuracy: 0.9328
40320/60000 [===================>..........] - ETA: 36s - loss: 0.2201 - categorical_accuracy: 0.9329
40352/60000 [===================>..........] - ETA: 36s - loss: 0.2200 - categorical_accuracy: 0.9328
40384/60000 [===================>..........] - ETA: 36s - loss: 0.2199 - categorical_accuracy: 0.9328
40416/60000 [===================>..........] - ETA: 36s - loss: 0.2198 - categorical_accuracy: 0.9328
40448/60000 [===================>..........] - ETA: 36s - loss: 0.2198 - categorical_accuracy: 0.9329
40480/60000 [===================>..........] - ETA: 36s - loss: 0.2199 - categorical_accuracy: 0.9329
40512/60000 [===================>..........] - ETA: 36s - loss: 0.2197 - categorical_accuracy: 0.9330
40544/60000 [===================>..........] - ETA: 36s - loss: 0.2196 - categorical_accuracy: 0.9330
40576/60000 [===================>..........] - ETA: 36s - loss: 0.2196 - categorical_accuracy: 0.9330
40608/60000 [===================>..........] - ETA: 36s - loss: 0.2195 - categorical_accuracy: 0.9330
40640/60000 [===================>..........] - ETA: 36s - loss: 0.2197 - categorical_accuracy: 0.9330
40672/60000 [===================>..........] - ETA: 36s - loss: 0.2196 - categorical_accuracy: 0.9330
40704/60000 [===================>..........] - ETA: 36s - loss: 0.2196 - categorical_accuracy: 0.9330
40736/60000 [===================>..........] - ETA: 36s - loss: 0.2195 - categorical_accuracy: 0.9330
40768/60000 [===================>..........] - ETA: 35s - loss: 0.2196 - categorical_accuracy: 0.9330
40800/60000 [===================>..........] - ETA: 35s - loss: 0.2195 - categorical_accuracy: 0.9330
40832/60000 [===================>..........] - ETA: 35s - loss: 0.2196 - categorical_accuracy: 0.9330
40864/60000 [===================>..........] - ETA: 35s - loss: 0.2195 - categorical_accuracy: 0.9330
40896/60000 [===================>..........] - ETA: 35s - loss: 0.2194 - categorical_accuracy: 0.9330
40928/60000 [===================>..........] - ETA: 35s - loss: 0.2194 - categorical_accuracy: 0.9331
40960/60000 [===================>..........] - ETA: 35s - loss: 0.2192 - categorical_accuracy: 0.9331
40992/60000 [===================>..........] - ETA: 35s - loss: 0.2191 - categorical_accuracy: 0.9331
41024/60000 [===================>..........] - ETA: 35s - loss: 0.2190 - categorical_accuracy: 0.9332
41056/60000 [===================>..........] - ETA: 35s - loss: 0.2188 - categorical_accuracy: 0.9332
41088/60000 [===================>..........] - ETA: 35s - loss: 0.2187 - categorical_accuracy: 0.9332
41120/60000 [===================>..........] - ETA: 35s - loss: 0.2187 - categorical_accuracy: 0.9332
41152/60000 [===================>..........] - ETA: 35s - loss: 0.2185 - categorical_accuracy: 0.9333
41184/60000 [===================>..........] - ETA: 35s - loss: 0.2185 - categorical_accuracy: 0.9333
41216/60000 [===================>..........] - ETA: 35s - loss: 0.2184 - categorical_accuracy: 0.9333
41248/60000 [===================>..........] - ETA: 35s - loss: 0.2183 - categorical_accuracy: 0.9333
41280/60000 [===================>..........] - ETA: 35s - loss: 0.2182 - categorical_accuracy: 0.9334
41312/60000 [===================>..........] - ETA: 34s - loss: 0.2181 - categorical_accuracy: 0.9334
41344/60000 [===================>..........] - ETA: 34s - loss: 0.2179 - categorical_accuracy: 0.9334
41376/60000 [===================>..........] - ETA: 34s - loss: 0.2178 - categorical_accuracy: 0.9335
41408/60000 [===================>..........] - ETA: 34s - loss: 0.2177 - categorical_accuracy: 0.9335
41440/60000 [===================>..........] - ETA: 34s - loss: 0.2177 - categorical_accuracy: 0.9335
41472/60000 [===================>..........] - ETA: 34s - loss: 0.2175 - categorical_accuracy: 0.9335
41504/60000 [===================>..........] - ETA: 34s - loss: 0.2174 - categorical_accuracy: 0.9336
41536/60000 [===================>..........] - ETA: 34s - loss: 0.2172 - categorical_accuracy: 0.9336
41568/60000 [===================>..........] - ETA: 34s - loss: 0.2172 - categorical_accuracy: 0.9336
41600/60000 [===================>..........] - ETA: 34s - loss: 0.2170 - categorical_accuracy: 0.9337
41632/60000 [===================>..........] - ETA: 34s - loss: 0.2171 - categorical_accuracy: 0.9337
41664/60000 [===================>..........] - ETA: 34s - loss: 0.2171 - categorical_accuracy: 0.9337
41696/60000 [===================>..........] - ETA: 34s - loss: 0.2170 - categorical_accuracy: 0.9337
41728/60000 [===================>..........] - ETA: 34s - loss: 0.2171 - categorical_accuracy: 0.9337
41760/60000 [===================>..........] - ETA: 34s - loss: 0.2170 - categorical_accuracy: 0.9337
41792/60000 [===================>..........] - ETA: 34s - loss: 0.2169 - categorical_accuracy: 0.9337
41824/60000 [===================>..........] - ETA: 33s - loss: 0.2168 - categorical_accuracy: 0.9338
41856/60000 [===================>..........] - ETA: 33s - loss: 0.2167 - categorical_accuracy: 0.9338
41888/60000 [===================>..........] - ETA: 33s - loss: 0.2166 - categorical_accuracy: 0.9338
41920/60000 [===================>..........] - ETA: 33s - loss: 0.2164 - categorical_accuracy: 0.9339
41952/60000 [===================>..........] - ETA: 33s - loss: 0.2163 - categorical_accuracy: 0.9339
41984/60000 [===================>..........] - ETA: 33s - loss: 0.2163 - categorical_accuracy: 0.9339
42016/60000 [====================>.........] - ETA: 33s - loss: 0.2162 - categorical_accuracy: 0.9339
42048/60000 [====================>.........] - ETA: 33s - loss: 0.2160 - categorical_accuracy: 0.9340
42080/60000 [====================>.........] - ETA: 33s - loss: 0.2159 - categorical_accuracy: 0.9340
42112/60000 [====================>.........] - ETA: 33s - loss: 0.2157 - categorical_accuracy: 0.9341
42144/60000 [====================>.........] - ETA: 33s - loss: 0.2156 - categorical_accuracy: 0.9341
42176/60000 [====================>.........] - ETA: 33s - loss: 0.2155 - categorical_accuracy: 0.9341
42208/60000 [====================>.........] - ETA: 33s - loss: 0.2154 - categorical_accuracy: 0.9342
42240/60000 [====================>.........] - ETA: 33s - loss: 0.2153 - categorical_accuracy: 0.9342
42272/60000 [====================>.........] - ETA: 33s - loss: 0.2152 - categorical_accuracy: 0.9342
42304/60000 [====================>.........] - ETA: 33s - loss: 0.2152 - categorical_accuracy: 0.9342
42336/60000 [====================>.........] - ETA: 33s - loss: 0.2150 - categorical_accuracy: 0.9342
42368/60000 [====================>.........] - ETA: 32s - loss: 0.2153 - categorical_accuracy: 0.9342
42400/60000 [====================>.........] - ETA: 32s - loss: 0.2152 - categorical_accuracy: 0.9342
42432/60000 [====================>.........] - ETA: 32s - loss: 0.2151 - categorical_accuracy: 0.9343
42464/60000 [====================>.........] - ETA: 32s - loss: 0.2150 - categorical_accuracy: 0.9343
42496/60000 [====================>.........] - ETA: 32s - loss: 0.2148 - categorical_accuracy: 0.9344
42528/60000 [====================>.........] - ETA: 32s - loss: 0.2149 - categorical_accuracy: 0.9344
42560/60000 [====================>.........] - ETA: 32s - loss: 0.2148 - categorical_accuracy: 0.9345
42592/60000 [====================>.........] - ETA: 32s - loss: 0.2147 - categorical_accuracy: 0.9345
42624/60000 [====================>.........] - ETA: 32s - loss: 0.2145 - categorical_accuracy: 0.9345
42656/60000 [====================>.........] - ETA: 32s - loss: 0.2144 - categorical_accuracy: 0.9345
42688/60000 [====================>.........] - ETA: 32s - loss: 0.2144 - categorical_accuracy: 0.9346
42720/60000 [====================>.........] - ETA: 32s - loss: 0.2143 - categorical_accuracy: 0.9346
42752/60000 [====================>.........] - ETA: 32s - loss: 0.2142 - categorical_accuracy: 0.9346
42784/60000 [====================>.........] - ETA: 32s - loss: 0.2141 - categorical_accuracy: 0.9346
42816/60000 [====================>.........] - ETA: 32s - loss: 0.2140 - categorical_accuracy: 0.9347
42848/60000 [====================>.........] - ETA: 32s - loss: 0.2139 - categorical_accuracy: 0.9347
42880/60000 [====================>.........] - ETA: 31s - loss: 0.2138 - categorical_accuracy: 0.9347
42912/60000 [====================>.........] - ETA: 31s - loss: 0.2137 - categorical_accuracy: 0.9347
42944/60000 [====================>.........] - ETA: 31s - loss: 0.2137 - categorical_accuracy: 0.9347
42976/60000 [====================>.........] - ETA: 31s - loss: 0.2136 - categorical_accuracy: 0.9347
43008/60000 [====================>.........] - ETA: 31s - loss: 0.2135 - categorical_accuracy: 0.9348
43040/60000 [====================>.........] - ETA: 31s - loss: 0.2133 - categorical_accuracy: 0.9348
43072/60000 [====================>.........] - ETA: 31s - loss: 0.2134 - categorical_accuracy: 0.9348
43104/60000 [====================>.........] - ETA: 31s - loss: 0.2134 - categorical_accuracy: 0.9348
43136/60000 [====================>.........] - ETA: 31s - loss: 0.2132 - categorical_accuracy: 0.9349
43168/60000 [====================>.........] - ETA: 31s - loss: 0.2131 - categorical_accuracy: 0.9349
43200/60000 [====================>.........] - ETA: 31s - loss: 0.2129 - categorical_accuracy: 0.9350
43232/60000 [====================>.........] - ETA: 31s - loss: 0.2128 - categorical_accuracy: 0.9350
43264/60000 [====================>.........] - ETA: 31s - loss: 0.2127 - categorical_accuracy: 0.9350
43296/60000 [====================>.........] - ETA: 31s - loss: 0.2126 - categorical_accuracy: 0.9351
43328/60000 [====================>.........] - ETA: 31s - loss: 0.2125 - categorical_accuracy: 0.9351
43360/60000 [====================>.........] - ETA: 31s - loss: 0.2123 - categorical_accuracy: 0.9352
43392/60000 [====================>.........] - ETA: 31s - loss: 0.2122 - categorical_accuracy: 0.9352
43424/60000 [====================>.........] - ETA: 30s - loss: 0.2122 - categorical_accuracy: 0.9352
43456/60000 [====================>.........] - ETA: 30s - loss: 0.2121 - categorical_accuracy: 0.9353
43488/60000 [====================>.........] - ETA: 30s - loss: 0.2122 - categorical_accuracy: 0.9352
43520/60000 [====================>.........] - ETA: 30s - loss: 0.2121 - categorical_accuracy: 0.9352
43552/60000 [====================>.........] - ETA: 30s - loss: 0.2120 - categorical_accuracy: 0.9353
43584/60000 [====================>.........] - ETA: 30s - loss: 0.2119 - categorical_accuracy: 0.9353
43616/60000 [====================>.........] - ETA: 30s - loss: 0.2118 - categorical_accuracy: 0.9353
43648/60000 [====================>.........] - ETA: 30s - loss: 0.2118 - categorical_accuracy: 0.9353
43680/60000 [====================>.........] - ETA: 30s - loss: 0.2117 - categorical_accuracy: 0.9353
43712/60000 [====================>.........] - ETA: 30s - loss: 0.2117 - categorical_accuracy: 0.9353
43744/60000 [====================>.........] - ETA: 30s - loss: 0.2116 - categorical_accuracy: 0.9354
43776/60000 [====================>.........] - ETA: 30s - loss: 0.2115 - categorical_accuracy: 0.9354
43808/60000 [====================>.........] - ETA: 30s - loss: 0.2115 - categorical_accuracy: 0.9354
43840/60000 [====================>.........] - ETA: 30s - loss: 0.2114 - categorical_accuracy: 0.9354
43872/60000 [====================>.........] - ETA: 30s - loss: 0.2114 - categorical_accuracy: 0.9354
43904/60000 [====================>.........] - ETA: 30s - loss: 0.2114 - categorical_accuracy: 0.9355
43936/60000 [====================>.........] - ETA: 30s - loss: 0.2113 - categorical_accuracy: 0.9355
43968/60000 [====================>.........] - ETA: 29s - loss: 0.2112 - categorical_accuracy: 0.9355
44000/60000 [=====================>........] - ETA: 29s - loss: 0.2110 - categorical_accuracy: 0.9356
44032/60000 [=====================>........] - ETA: 29s - loss: 0.2109 - categorical_accuracy: 0.9356
44064/60000 [=====================>........] - ETA: 29s - loss: 0.2108 - categorical_accuracy: 0.9356
44096/60000 [=====================>........] - ETA: 29s - loss: 0.2107 - categorical_accuracy: 0.9357
44128/60000 [=====================>........] - ETA: 29s - loss: 0.2110 - categorical_accuracy: 0.9356
44160/60000 [=====================>........] - ETA: 29s - loss: 0.2109 - categorical_accuracy: 0.9357
44192/60000 [=====================>........] - ETA: 29s - loss: 0.2108 - categorical_accuracy: 0.9357
44224/60000 [=====================>........] - ETA: 29s - loss: 0.2108 - categorical_accuracy: 0.9357
44256/60000 [=====================>........] - ETA: 29s - loss: 0.2107 - categorical_accuracy: 0.9358
44288/60000 [=====================>........] - ETA: 29s - loss: 0.2105 - categorical_accuracy: 0.9358
44320/60000 [=====================>........] - ETA: 29s - loss: 0.2105 - categorical_accuracy: 0.9358
44352/60000 [=====================>........] - ETA: 29s - loss: 0.2104 - categorical_accuracy: 0.9359
44384/60000 [=====================>........] - ETA: 29s - loss: 0.2103 - categorical_accuracy: 0.9359
44416/60000 [=====================>........] - ETA: 29s - loss: 0.2102 - categorical_accuracy: 0.9359
44448/60000 [=====================>........] - ETA: 29s - loss: 0.2102 - categorical_accuracy: 0.9359
44480/60000 [=====================>........] - ETA: 28s - loss: 0.2100 - categorical_accuracy: 0.9359
44512/60000 [=====================>........] - ETA: 28s - loss: 0.2099 - categorical_accuracy: 0.9360
44544/60000 [=====================>........] - ETA: 28s - loss: 0.2097 - categorical_accuracy: 0.9360
44576/60000 [=====================>........] - ETA: 28s - loss: 0.2096 - categorical_accuracy: 0.9361
44608/60000 [=====================>........] - ETA: 28s - loss: 0.2096 - categorical_accuracy: 0.9360
44640/60000 [=====================>........] - ETA: 28s - loss: 0.2095 - categorical_accuracy: 0.9361
44672/60000 [=====================>........] - ETA: 28s - loss: 0.2094 - categorical_accuracy: 0.9361
44704/60000 [=====================>........] - ETA: 28s - loss: 0.2093 - categorical_accuracy: 0.9362
44736/60000 [=====================>........] - ETA: 28s - loss: 0.2092 - categorical_accuracy: 0.9362
44768/60000 [=====================>........] - ETA: 28s - loss: 0.2090 - categorical_accuracy: 0.9362
44800/60000 [=====================>........] - ETA: 28s - loss: 0.2090 - categorical_accuracy: 0.9362
44832/60000 [=====================>........] - ETA: 28s - loss: 0.2090 - categorical_accuracy: 0.9362
44864/60000 [=====================>........] - ETA: 28s - loss: 0.2089 - categorical_accuracy: 0.9363
44896/60000 [=====================>........] - ETA: 28s - loss: 0.2089 - categorical_accuracy: 0.9363
44928/60000 [=====================>........] - ETA: 28s - loss: 0.2089 - categorical_accuracy: 0.9363
44960/60000 [=====================>........] - ETA: 28s - loss: 0.2088 - categorical_accuracy: 0.9363
44992/60000 [=====================>........] - ETA: 28s - loss: 0.2087 - categorical_accuracy: 0.9363
45024/60000 [=====================>........] - ETA: 27s - loss: 0.2086 - categorical_accuracy: 0.9363
45056/60000 [=====================>........] - ETA: 27s - loss: 0.2086 - categorical_accuracy: 0.9363
45088/60000 [=====================>........] - ETA: 27s - loss: 0.2085 - categorical_accuracy: 0.9363
45120/60000 [=====================>........] - ETA: 27s - loss: 0.2085 - categorical_accuracy: 0.9364
45152/60000 [=====================>........] - ETA: 27s - loss: 0.2084 - categorical_accuracy: 0.9364
45184/60000 [=====================>........] - ETA: 27s - loss: 0.2083 - categorical_accuracy: 0.9364
45216/60000 [=====================>........] - ETA: 27s - loss: 0.2082 - categorical_accuracy: 0.9365
45248/60000 [=====================>........] - ETA: 27s - loss: 0.2081 - categorical_accuracy: 0.9364
45280/60000 [=====================>........] - ETA: 27s - loss: 0.2081 - categorical_accuracy: 0.9365
45312/60000 [=====================>........] - ETA: 27s - loss: 0.2080 - categorical_accuracy: 0.9365
45344/60000 [=====================>........] - ETA: 27s - loss: 0.2079 - categorical_accuracy: 0.9365
45376/60000 [=====================>........] - ETA: 27s - loss: 0.2080 - categorical_accuracy: 0.9365
45408/60000 [=====================>........] - ETA: 27s - loss: 0.2080 - categorical_accuracy: 0.9365
45440/60000 [=====================>........] - ETA: 27s - loss: 0.2079 - categorical_accuracy: 0.9366
45472/60000 [=====================>........] - ETA: 27s - loss: 0.2079 - categorical_accuracy: 0.9366
45504/60000 [=====================>........] - ETA: 27s - loss: 0.2077 - categorical_accuracy: 0.9366
45536/60000 [=====================>........] - ETA: 27s - loss: 0.2076 - categorical_accuracy: 0.9366
45568/60000 [=====================>........] - ETA: 26s - loss: 0.2075 - categorical_accuracy: 0.9367
45600/60000 [=====================>........] - ETA: 26s - loss: 0.2074 - categorical_accuracy: 0.9367
45632/60000 [=====================>........] - ETA: 26s - loss: 0.2072 - categorical_accuracy: 0.9368
45664/60000 [=====================>........] - ETA: 26s - loss: 0.2072 - categorical_accuracy: 0.9368
45696/60000 [=====================>........] - ETA: 26s - loss: 0.2072 - categorical_accuracy: 0.9368
45728/60000 [=====================>........] - ETA: 26s - loss: 0.2071 - categorical_accuracy: 0.9368
45760/60000 [=====================>........] - ETA: 26s - loss: 0.2071 - categorical_accuracy: 0.9368
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2070 - categorical_accuracy: 0.9368
45824/60000 [=====================>........] - ETA: 26s - loss: 0.2070 - categorical_accuracy: 0.9368
45856/60000 [=====================>........] - ETA: 26s - loss: 0.2069 - categorical_accuracy: 0.9369
45888/60000 [=====================>........] - ETA: 26s - loss: 0.2069 - categorical_accuracy: 0.9369
45920/60000 [=====================>........] - ETA: 26s - loss: 0.2072 - categorical_accuracy: 0.9368
45952/60000 [=====================>........] - ETA: 26s - loss: 0.2072 - categorical_accuracy: 0.9369
45984/60000 [=====================>........] - ETA: 26s - loss: 0.2072 - categorical_accuracy: 0.9369
46016/60000 [======================>.......] - ETA: 26s - loss: 0.2071 - categorical_accuracy: 0.9369
46048/60000 [======================>.......] - ETA: 26s - loss: 0.2069 - categorical_accuracy: 0.9370
46080/60000 [======================>.......] - ETA: 26s - loss: 0.2068 - categorical_accuracy: 0.9370
46112/60000 [======================>.......] - ETA: 25s - loss: 0.2068 - categorical_accuracy: 0.9370
46144/60000 [======================>.......] - ETA: 25s - loss: 0.2067 - categorical_accuracy: 0.9370
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2069 - categorical_accuracy: 0.9370
46208/60000 [======================>.......] - ETA: 25s - loss: 0.2068 - categorical_accuracy: 0.9371
46240/60000 [======================>.......] - ETA: 25s - loss: 0.2067 - categorical_accuracy: 0.9371
46272/60000 [======================>.......] - ETA: 25s - loss: 0.2067 - categorical_accuracy: 0.9371
46304/60000 [======================>.......] - ETA: 25s - loss: 0.2066 - categorical_accuracy: 0.9372
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2065 - categorical_accuracy: 0.9372
46368/60000 [======================>.......] - ETA: 25s - loss: 0.2064 - categorical_accuracy: 0.9372
46400/60000 [======================>.......] - ETA: 25s - loss: 0.2064 - categorical_accuracy: 0.9372
46432/60000 [======================>.......] - ETA: 25s - loss: 0.2064 - categorical_accuracy: 0.9372
46464/60000 [======================>.......] - ETA: 25s - loss: 0.2062 - categorical_accuracy: 0.9373
46496/60000 [======================>.......] - ETA: 25s - loss: 0.2061 - categorical_accuracy: 0.9373
46528/60000 [======================>.......] - ETA: 25s - loss: 0.2060 - categorical_accuracy: 0.9374
46560/60000 [======================>.......] - ETA: 25s - loss: 0.2059 - categorical_accuracy: 0.9374
46592/60000 [======================>.......] - ETA: 25s - loss: 0.2058 - categorical_accuracy: 0.9374
46624/60000 [======================>.......] - ETA: 24s - loss: 0.2060 - categorical_accuracy: 0.9374
46656/60000 [======================>.......] - ETA: 24s - loss: 0.2060 - categorical_accuracy: 0.9374
46688/60000 [======================>.......] - ETA: 24s - loss: 0.2059 - categorical_accuracy: 0.9374
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2058 - categorical_accuracy: 0.9375
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9375
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9375
46816/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9376
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9376
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9376
46912/60000 [======================>.......] - ETA: 24s - loss: 0.2053 - categorical_accuracy: 0.9376
46944/60000 [======================>.......] - ETA: 24s - loss: 0.2052 - categorical_accuracy: 0.9376
46976/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9376
47008/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9376
47040/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9376
47072/60000 [======================>.......] - ETA: 24s - loss: 0.2053 - categorical_accuracy: 0.9377
47104/60000 [======================>.......] - ETA: 24s - loss: 0.2052 - categorical_accuracy: 0.9377
47136/60000 [======================>.......] - ETA: 24s - loss: 0.2051 - categorical_accuracy: 0.9378
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2051 - categorical_accuracy: 0.9378
47200/60000 [======================>.......] - ETA: 23s - loss: 0.2051 - categorical_accuracy: 0.9378
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2050 - categorical_accuracy: 0.9378
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2050 - categorical_accuracy: 0.9378
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2049 - categorical_accuracy: 0.9378
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9379
47360/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9379
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9379
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9379
47456/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9379
47488/60000 [======================>.......] - ETA: 23s - loss: 0.2045 - categorical_accuracy: 0.9379
47520/60000 [======================>.......] - ETA: 23s - loss: 0.2044 - categorical_accuracy: 0.9379
47552/60000 [======================>.......] - ETA: 23s - loss: 0.2043 - categorical_accuracy: 0.9380
47584/60000 [======================>.......] - ETA: 23s - loss: 0.2042 - categorical_accuracy: 0.9380
47616/60000 [======================>.......] - ETA: 23s - loss: 0.2041 - categorical_accuracy: 0.9381
47648/60000 [======================>.......] - ETA: 23s - loss: 0.2041 - categorical_accuracy: 0.9381
47680/60000 [======================>.......] - ETA: 23s - loss: 0.2040 - categorical_accuracy: 0.9381
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2039 - categorical_accuracy: 0.9381
47744/60000 [======================>.......] - ETA: 22s - loss: 0.2039 - categorical_accuracy: 0.9381
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2038 - categorical_accuracy: 0.9381
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2037 - categorical_accuracy: 0.9381
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2038 - categorical_accuracy: 0.9381
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2037 - categorical_accuracy: 0.9381
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2036 - categorical_accuracy: 0.9381
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2035 - categorical_accuracy: 0.9382
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2033 - categorical_accuracy: 0.9382
48000/60000 [=======================>......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9383
48032/60000 [=======================>......] - ETA: 22s - loss: 0.2031 - categorical_accuracy: 0.9383
48064/60000 [=======================>......] - ETA: 22s - loss: 0.2030 - categorical_accuracy: 0.9384
48096/60000 [=======================>......] - ETA: 22s - loss: 0.2029 - categorical_accuracy: 0.9384
48128/60000 [=======================>......] - ETA: 22s - loss: 0.2028 - categorical_accuracy: 0.9384
48160/60000 [=======================>......] - ETA: 22s - loss: 0.2028 - categorical_accuracy: 0.9384
48192/60000 [=======================>......] - ETA: 22s - loss: 0.2027 - categorical_accuracy: 0.9384
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2026 - categorical_accuracy: 0.9385
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2025 - categorical_accuracy: 0.9385
48288/60000 [=======================>......] - ETA: 21s - loss: 0.2024 - categorical_accuracy: 0.9385
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2023 - categorical_accuracy: 0.9385
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2022 - categorical_accuracy: 0.9385
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2021 - categorical_accuracy: 0.9386
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2020 - categorical_accuracy: 0.9386
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2020 - categorical_accuracy: 0.9386
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2019 - categorical_accuracy: 0.9386
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2018 - categorical_accuracy: 0.9386
48544/60000 [=======================>......] - ETA: 21s - loss: 0.2017 - categorical_accuracy: 0.9387
48576/60000 [=======================>......] - ETA: 21s - loss: 0.2016 - categorical_accuracy: 0.9386
48608/60000 [=======================>......] - ETA: 21s - loss: 0.2015 - categorical_accuracy: 0.9387
48640/60000 [=======================>......] - ETA: 21s - loss: 0.2014 - categorical_accuracy: 0.9387
48672/60000 [=======================>......] - ETA: 21s - loss: 0.2014 - categorical_accuracy: 0.9387
48704/60000 [=======================>......] - ETA: 21s - loss: 0.2013 - categorical_accuracy: 0.9387
48736/60000 [=======================>......] - ETA: 21s - loss: 0.2012 - categorical_accuracy: 0.9387
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2011 - categorical_accuracy: 0.9388
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2010 - categorical_accuracy: 0.9388
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2009 - categorical_accuracy: 0.9388
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2009 - categorical_accuracy: 0.9388
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2008 - categorical_accuracy: 0.9388
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2008 - categorical_accuracy: 0.9388
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2008 - categorical_accuracy: 0.9389
48992/60000 [=======================>......] - ETA: 20s - loss: 0.2007 - categorical_accuracy: 0.9389
49024/60000 [=======================>......] - ETA: 20s - loss: 0.2006 - categorical_accuracy: 0.9389
49056/60000 [=======================>......] - ETA: 20s - loss: 0.2006 - categorical_accuracy: 0.9389
49088/60000 [=======================>......] - ETA: 20s - loss: 0.2005 - categorical_accuracy: 0.9389
49120/60000 [=======================>......] - ETA: 20s - loss: 0.2004 - categorical_accuracy: 0.9390
49152/60000 [=======================>......] - ETA: 20s - loss: 0.2003 - categorical_accuracy: 0.9390
49184/60000 [=======================>......] - ETA: 20s - loss: 0.2002 - categorical_accuracy: 0.9390
49216/60000 [=======================>......] - ETA: 20s - loss: 0.2001 - categorical_accuracy: 0.9391
49248/60000 [=======================>......] - ETA: 20s - loss: 0.2000 - categorical_accuracy: 0.9391
49280/60000 [=======================>......] - ETA: 20s - loss: 0.1999 - categorical_accuracy: 0.9391
49312/60000 [=======================>......] - ETA: 19s - loss: 0.1998 - categorical_accuracy: 0.9392
49344/60000 [=======================>......] - ETA: 19s - loss: 0.1998 - categorical_accuracy: 0.9392
49376/60000 [=======================>......] - ETA: 19s - loss: 0.1997 - categorical_accuracy: 0.9392
49408/60000 [=======================>......] - ETA: 19s - loss: 0.1996 - categorical_accuracy: 0.9393
49440/60000 [=======================>......] - ETA: 19s - loss: 0.1996 - categorical_accuracy: 0.9393
49472/60000 [=======================>......] - ETA: 19s - loss: 0.1995 - categorical_accuracy: 0.9393
49504/60000 [=======================>......] - ETA: 19s - loss: 0.1994 - categorical_accuracy: 0.9393
49536/60000 [=======================>......] - ETA: 19s - loss: 0.1993 - categorical_accuracy: 0.9394
49568/60000 [=======================>......] - ETA: 19s - loss: 0.1992 - categorical_accuracy: 0.9394
49600/60000 [=======================>......] - ETA: 19s - loss: 0.1991 - categorical_accuracy: 0.9394
49632/60000 [=======================>......] - ETA: 19s - loss: 0.1990 - categorical_accuracy: 0.9395
49664/60000 [=======================>......] - ETA: 19s - loss: 0.1989 - categorical_accuracy: 0.9395
49696/60000 [=======================>......] - ETA: 19s - loss: 0.1988 - categorical_accuracy: 0.9395
49728/60000 [=======================>......] - ETA: 19s - loss: 0.1987 - categorical_accuracy: 0.9396
49760/60000 [=======================>......] - ETA: 19s - loss: 0.1986 - categorical_accuracy: 0.9395
49792/60000 [=======================>......] - ETA: 19s - loss: 0.1985 - categorical_accuracy: 0.9396
49824/60000 [=======================>......] - ETA: 19s - loss: 0.1985 - categorical_accuracy: 0.9396
49856/60000 [=======================>......] - ETA: 18s - loss: 0.1984 - categorical_accuracy: 0.9396
49888/60000 [=======================>......] - ETA: 18s - loss: 0.1983 - categorical_accuracy: 0.9396
49920/60000 [=======================>......] - ETA: 18s - loss: 0.1982 - categorical_accuracy: 0.9397
49952/60000 [=======================>......] - ETA: 18s - loss: 0.1982 - categorical_accuracy: 0.9397
49984/60000 [=======================>......] - ETA: 18s - loss: 0.1981 - categorical_accuracy: 0.9397
50016/60000 [========================>.....] - ETA: 18s - loss: 0.1980 - categorical_accuracy: 0.9398
50048/60000 [========================>.....] - ETA: 18s - loss: 0.1980 - categorical_accuracy: 0.9397
50080/60000 [========================>.....] - ETA: 18s - loss: 0.1980 - categorical_accuracy: 0.9397
50112/60000 [========================>.....] - ETA: 18s - loss: 0.1979 - categorical_accuracy: 0.9397
50144/60000 [========================>.....] - ETA: 18s - loss: 0.1979 - categorical_accuracy: 0.9397
50176/60000 [========================>.....] - ETA: 18s - loss: 0.1978 - categorical_accuracy: 0.9398
50208/60000 [========================>.....] - ETA: 18s - loss: 0.1977 - categorical_accuracy: 0.9398
50240/60000 [========================>.....] - ETA: 18s - loss: 0.1975 - categorical_accuracy: 0.9398
50272/60000 [========================>.....] - ETA: 18s - loss: 0.1975 - categorical_accuracy: 0.9398
50304/60000 [========================>.....] - ETA: 18s - loss: 0.1974 - categorical_accuracy: 0.9398
50336/60000 [========================>.....] - ETA: 18s - loss: 0.1974 - categorical_accuracy: 0.9399
50368/60000 [========================>.....] - ETA: 17s - loss: 0.1975 - categorical_accuracy: 0.9398
50400/60000 [========================>.....] - ETA: 17s - loss: 0.1973 - categorical_accuracy: 0.9399
50432/60000 [========================>.....] - ETA: 17s - loss: 0.1973 - categorical_accuracy: 0.9399
50464/60000 [========================>.....] - ETA: 17s - loss: 0.1972 - categorical_accuracy: 0.9399
50496/60000 [========================>.....] - ETA: 17s - loss: 0.1971 - categorical_accuracy: 0.9399
50528/60000 [========================>.....] - ETA: 17s - loss: 0.1971 - categorical_accuracy: 0.9399
50560/60000 [========================>.....] - ETA: 17s - loss: 0.1969 - categorical_accuracy: 0.9399
50592/60000 [========================>.....] - ETA: 17s - loss: 0.1969 - categorical_accuracy: 0.9400
50624/60000 [========================>.....] - ETA: 17s - loss: 0.1968 - categorical_accuracy: 0.9400
50656/60000 [========================>.....] - ETA: 17s - loss: 0.1967 - categorical_accuracy: 0.9400
50688/60000 [========================>.....] - ETA: 17s - loss: 0.1966 - categorical_accuracy: 0.9400
50720/60000 [========================>.....] - ETA: 17s - loss: 0.1965 - categorical_accuracy: 0.9400
50752/60000 [========================>.....] - ETA: 17s - loss: 0.1964 - categorical_accuracy: 0.9401
50784/60000 [========================>.....] - ETA: 17s - loss: 0.1965 - categorical_accuracy: 0.9401
50816/60000 [========================>.....] - ETA: 17s - loss: 0.1964 - categorical_accuracy: 0.9401
50848/60000 [========================>.....] - ETA: 17s - loss: 0.1964 - categorical_accuracy: 0.9401
50880/60000 [========================>.....] - ETA: 17s - loss: 0.1963 - categorical_accuracy: 0.9401
50912/60000 [========================>.....] - ETA: 16s - loss: 0.1963 - categorical_accuracy: 0.9401
50944/60000 [========================>.....] - ETA: 16s - loss: 0.1962 - categorical_accuracy: 0.9401
50976/60000 [========================>.....] - ETA: 16s - loss: 0.1961 - categorical_accuracy: 0.9401
51008/60000 [========================>.....] - ETA: 16s - loss: 0.1961 - categorical_accuracy: 0.9401
51040/60000 [========================>.....] - ETA: 16s - loss: 0.1960 - categorical_accuracy: 0.9402
51072/60000 [========================>.....] - ETA: 16s - loss: 0.1959 - categorical_accuracy: 0.9402
51104/60000 [========================>.....] - ETA: 16s - loss: 0.1958 - categorical_accuracy: 0.9402
51136/60000 [========================>.....] - ETA: 16s - loss: 0.1959 - categorical_accuracy: 0.9402
51168/60000 [========================>.....] - ETA: 16s - loss: 0.1958 - categorical_accuracy: 0.9403
51200/60000 [========================>.....] - ETA: 16s - loss: 0.1957 - categorical_accuracy: 0.9403
51232/60000 [========================>.....] - ETA: 16s - loss: 0.1957 - categorical_accuracy: 0.9403
51264/60000 [========================>.....] - ETA: 16s - loss: 0.1956 - categorical_accuracy: 0.9403
51296/60000 [========================>.....] - ETA: 16s - loss: 0.1955 - categorical_accuracy: 0.9403
51328/60000 [========================>.....] - ETA: 16s - loss: 0.1954 - categorical_accuracy: 0.9404
51360/60000 [========================>.....] - ETA: 16s - loss: 0.1953 - categorical_accuracy: 0.9404
51392/60000 [========================>.....] - ETA: 16s - loss: 0.1953 - categorical_accuracy: 0.9404
51424/60000 [========================>.....] - ETA: 16s - loss: 0.1953 - categorical_accuracy: 0.9404
51456/60000 [========================>.....] - ETA: 15s - loss: 0.1952 - categorical_accuracy: 0.9405
51488/60000 [========================>.....] - ETA: 15s - loss: 0.1952 - categorical_accuracy: 0.9404
51520/60000 [========================>.....] - ETA: 15s - loss: 0.1951 - categorical_accuracy: 0.9405
51552/60000 [========================>.....] - ETA: 15s - loss: 0.1950 - categorical_accuracy: 0.9405
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1950 - categorical_accuracy: 0.9405
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1949 - categorical_accuracy: 0.9405
51648/60000 [========================>.....] - ETA: 15s - loss: 0.1948 - categorical_accuracy: 0.9406
51680/60000 [========================>.....] - ETA: 15s - loss: 0.1947 - categorical_accuracy: 0.9406
51712/60000 [========================>.....] - ETA: 15s - loss: 0.1946 - categorical_accuracy: 0.9406
51744/60000 [========================>.....] - ETA: 15s - loss: 0.1945 - categorical_accuracy: 0.9407
51776/60000 [========================>.....] - ETA: 15s - loss: 0.1945 - categorical_accuracy: 0.9406
51808/60000 [========================>.....] - ETA: 15s - loss: 0.1945 - categorical_accuracy: 0.9407
51840/60000 [========================>.....] - ETA: 15s - loss: 0.1944 - categorical_accuracy: 0.9407
51872/60000 [========================>.....] - ETA: 15s - loss: 0.1942 - categorical_accuracy: 0.9407
51904/60000 [========================>.....] - ETA: 15s - loss: 0.1942 - categorical_accuracy: 0.9408
51936/60000 [========================>.....] - ETA: 15s - loss: 0.1941 - categorical_accuracy: 0.9408
51968/60000 [========================>.....] - ETA: 15s - loss: 0.1940 - categorical_accuracy: 0.9408
52000/60000 [=========================>....] - ETA: 14s - loss: 0.1940 - categorical_accuracy: 0.9408
52032/60000 [=========================>....] - ETA: 14s - loss: 0.1939 - categorical_accuracy: 0.9408
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9409
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1940 - categorical_accuracy: 0.9409
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1939 - categorical_accuracy: 0.9409
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9409
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9409
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1937 - categorical_accuracy: 0.9409
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1936 - categorical_accuracy: 0.9409
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1935 - categorical_accuracy: 0.9410
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1934 - categorical_accuracy: 0.9410
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1934 - categorical_accuracy: 0.9410
52384/60000 [=========================>....] - ETA: 14s - loss: 0.1933 - categorical_accuracy: 0.9411
52416/60000 [=========================>....] - ETA: 14s - loss: 0.1932 - categorical_accuracy: 0.9411
52448/60000 [=========================>....] - ETA: 14s - loss: 0.1931 - categorical_accuracy: 0.9411
52480/60000 [=========================>....] - ETA: 14s - loss: 0.1931 - categorical_accuracy: 0.9411
52512/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9412
52544/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9412
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9412
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9412
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9412
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9412
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9412
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9412
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1927 - categorical_accuracy: 0.9413
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9413
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9412
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1927 - categorical_accuracy: 0.9413
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1927 - categorical_accuracy: 0.9413
52928/60000 [=========================>....] - ETA: 13s - loss: 0.1927 - categorical_accuracy: 0.9413
52960/60000 [=========================>....] - ETA: 13s - loss: 0.1926 - categorical_accuracy: 0.9413
52992/60000 [=========================>....] - ETA: 13s - loss: 0.1925 - categorical_accuracy: 0.9413
53024/60000 [=========================>....] - ETA: 13s - loss: 0.1925 - categorical_accuracy: 0.9413
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1924 - categorical_accuracy: 0.9413
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1924 - categorical_accuracy: 0.9413
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1924 - categorical_accuracy: 0.9413
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1923 - categorical_accuracy: 0.9414
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1923 - categorical_accuracy: 0.9414
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9414
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9414
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1920 - categorical_accuracy: 0.9414
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9414
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9414
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9414
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1920 - categorical_accuracy: 0.9415
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9415
53472/60000 [=========================>....] - ETA: 12s - loss: 0.1918 - categorical_accuracy: 0.9415
53504/60000 [=========================>....] - ETA: 12s - loss: 0.1917 - categorical_accuracy: 0.9415
53536/60000 [=========================>....] - ETA: 12s - loss: 0.1917 - categorical_accuracy: 0.9416
53568/60000 [=========================>....] - ETA: 12s - loss: 0.1916 - categorical_accuracy: 0.9416
53600/60000 [=========================>....] - ETA: 12s - loss: 0.1915 - categorical_accuracy: 0.9416
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1914 - categorical_accuracy: 0.9417
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1913 - categorical_accuracy: 0.9416
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1913 - categorical_accuracy: 0.9417
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1912 - categorical_accuracy: 0.9417
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1911 - categorical_accuracy: 0.9417
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1910 - categorical_accuracy: 0.9417
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1909 - categorical_accuracy: 0.9418
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1909 - categorical_accuracy: 0.9418
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1910 - categorical_accuracy: 0.9417
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1909 - categorical_accuracy: 0.9418
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1909 - categorical_accuracy: 0.9418
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1908 - categorical_accuracy: 0.9418
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1907 - categorical_accuracy: 0.9418
54048/60000 [==========================>...] - ETA: 11s - loss: 0.1907 - categorical_accuracy: 0.9418
54080/60000 [==========================>...] - ETA: 11s - loss: 0.1907 - categorical_accuracy: 0.9418
54112/60000 [==========================>...] - ETA: 11s - loss: 0.1906 - categorical_accuracy: 0.9418
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1905 - categorical_accuracy: 0.9419
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9419
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9419
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9419
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9419
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1902 - categorical_accuracy: 0.9419
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9419
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9420
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1902 - categorical_accuracy: 0.9420
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9420
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1902 - categorical_accuracy: 0.9420
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1901 - categorical_accuracy: 0.9421
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1901 - categorical_accuracy: 0.9420
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1902 - categorical_accuracy: 0.9420
54592/60000 [==========================>...] - ETA: 10s - loss: 0.1901 - categorical_accuracy: 0.9420
54624/60000 [==========================>...] - ETA: 10s - loss: 0.1900 - categorical_accuracy: 0.9420
54656/60000 [==========================>...] - ETA: 10s - loss: 0.1900 - categorical_accuracy: 0.9421
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9421 
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1898 - categorical_accuracy: 0.9421
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1898 - categorical_accuracy: 0.9421
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9421
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9421
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9421
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9421
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9422
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1895 - categorical_accuracy: 0.9422
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1895 - categorical_accuracy: 0.9422
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1894 - categorical_accuracy: 0.9422
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1894 - categorical_accuracy: 0.9422
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1893 - categorical_accuracy: 0.9423
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1892 - categorical_accuracy: 0.9423
55136/60000 [==========================>...] - ETA: 9s - loss: 0.1892 - categorical_accuracy: 0.9423
55168/60000 [==========================>...] - ETA: 9s - loss: 0.1891 - categorical_accuracy: 0.9423
55200/60000 [==========================>...] - ETA: 9s - loss: 0.1890 - categorical_accuracy: 0.9424
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9424
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9424
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1888 - categorical_accuracy: 0.9424
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9425
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9425
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1885 - categorical_accuracy: 0.9425
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1884 - categorical_accuracy: 0.9426
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1883 - categorical_accuracy: 0.9426
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1882 - categorical_accuracy: 0.9426
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1881 - categorical_accuracy: 0.9427
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1882 - categorical_accuracy: 0.9426
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1883 - categorical_accuracy: 0.9426
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1882 - categorical_accuracy: 0.9426
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1883 - categorical_accuracy: 0.9427
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1882 - categorical_accuracy: 0.9427
55712/60000 [==========================>...] - ETA: 8s - loss: 0.1881 - categorical_accuracy: 0.9427
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9427
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9427
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9427
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9427
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1879 - categorical_accuracy: 0.9428
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9428
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1879 - categorical_accuracy: 0.9428
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1878 - categorical_accuracy: 0.9428
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9429
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9429
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9429
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9429
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9429
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1874 - categorical_accuracy: 0.9429
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1873 - categorical_accuracy: 0.9430
56224/60000 [===========================>..] - ETA: 7s - loss: 0.1873 - categorical_accuracy: 0.9430
56256/60000 [===========================>..] - ETA: 7s - loss: 0.1872 - categorical_accuracy: 0.9430
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9430
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9430
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9431
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9431
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9430
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9431
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9431
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9431
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9431
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9431
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9432
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9432
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9432
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9432
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9432
56768/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9432
56800/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9432
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9433
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9433
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9433
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9433
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1861 - categorical_accuracy: 0.9434
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9434
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9434
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9434
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9434
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9435
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9434
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9435
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9435
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1857 - categorical_accuracy: 0.9435
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1856 - categorical_accuracy: 0.9435
57312/60000 [===========================>..] - ETA: 5s - loss: 0.1855 - categorical_accuracy: 0.9435
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1856 - categorical_accuracy: 0.9435
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1855 - categorical_accuracy: 0.9435
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9435
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9436
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9436
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9436
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9436
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9436
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9437
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9437
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9437
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9437
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1849 - categorical_accuracy: 0.9437
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9437
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1849 - categorical_accuracy: 0.9437
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1848 - categorical_accuracy: 0.9437
57856/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9438
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9438
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9438
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9438
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9438
58016/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9438
58048/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9438
58080/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9438
58112/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9438
58144/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9439
58176/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9439
58208/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9439
58240/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9439
58272/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9439
58304/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9439
58336/60000 [============================>.] - ETA: 3s - loss: 0.1842 - categorical_accuracy: 0.9440
58368/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9440
58400/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9440
58432/60000 [============================>.] - ETA: 2s - loss: 0.1841 - categorical_accuracy: 0.9440
58464/60000 [============================>.] - ETA: 2s - loss: 0.1840 - categorical_accuracy: 0.9440
58496/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9440
58528/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9440
58560/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9440
58592/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9441
58624/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9441
58656/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9441
58688/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9441
58720/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9441
58752/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9441
58784/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9442
58816/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9442
58848/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9442
58880/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9442
58912/60000 [============================>.] - ETA: 2s - loss: 0.1832 - categorical_accuracy: 0.9443
58944/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9443
58976/60000 [============================>.] - ETA: 1s - loss: 0.1830 - categorical_accuracy: 0.9443
59008/60000 [============================>.] - ETA: 1s - loss: 0.1830 - categorical_accuracy: 0.9443
59040/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9444
59072/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9444
59104/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9444
59136/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9444
59168/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9444
59200/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9444
59232/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9444
59264/60000 [============================>.] - ETA: 1s - loss: 0.1830 - categorical_accuracy: 0.9444
59296/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9444
59328/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9444
59360/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9444
59392/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9444
59424/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9445
59456/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9445
59488/60000 [============================>.] - ETA: 0s - loss: 0.1827 - categorical_accuracy: 0.9445
59520/60000 [============================>.] - ETA: 0s - loss: 0.1828 - categorical_accuracy: 0.9445
59552/60000 [============================>.] - ETA: 0s - loss: 0.1827 - categorical_accuracy: 0.9445
59584/60000 [============================>.] - ETA: 0s - loss: 0.1828 - categorical_accuracy: 0.9445
59616/60000 [============================>.] - ETA: 0s - loss: 0.1827 - categorical_accuracy: 0.9445
59648/60000 [============================>.] - ETA: 0s - loss: 0.1827 - categorical_accuracy: 0.9445
59680/60000 [============================>.] - ETA: 0s - loss: 0.1826 - categorical_accuracy: 0.9445
59712/60000 [============================>.] - ETA: 0s - loss: 0.1826 - categorical_accuracy: 0.9445
59744/60000 [============================>.] - ETA: 0s - loss: 0.1826 - categorical_accuracy: 0.9445
59776/60000 [============================>.] - ETA: 0s - loss: 0.1825 - categorical_accuracy: 0.9446
59808/60000 [============================>.] - ETA: 0s - loss: 0.1826 - categorical_accuracy: 0.9446
59840/60000 [============================>.] - ETA: 0s - loss: 0.1825 - categorical_accuracy: 0.9446
59872/60000 [============================>.] - ETA: 0s - loss: 0.1824 - categorical_accuracy: 0.9446
59904/60000 [============================>.] - ETA: 0s - loss: 0.1823 - categorical_accuracy: 0.9446
59936/60000 [============================>.] - ETA: 0s - loss: 0.1823 - categorical_accuracy: 0.9446
59968/60000 [============================>.] - ETA: 0s - loss: 0.1822 - categorical_accuracy: 0.9447
60000/60000 [==============================] - 116s 2ms/step - loss: 0.1821 - categorical_accuracy: 0.9447 - val_loss: 0.0482 - val_categorical_accuracy: 0.9850

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  192/10000 [..............................] - ETA: 6s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  672/10000 [=>............................] - ETA: 4s
  832/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 3s
 1312/10000 [==>...........................] - ETA: 3s
 1472/10000 [===>..........................] - ETA: 3s
 1632/10000 [===>..........................] - ETA: 3s
 1792/10000 [====>.........................] - ETA: 3s
 1952/10000 [====>.........................] - ETA: 3s
 2112/10000 [=====>........................] - ETA: 3s
 2272/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2592/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
 3200/10000 [========>.....................] - ETA: 2s
 3360/10000 [=========>....................] - ETA: 2s
 3520/10000 [=========>....................] - ETA: 2s
 3680/10000 [==========>...................] - ETA: 2s
 3840/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 2s
 4320/10000 [===========>..................] - ETA: 2s
 4480/10000 [============>.................] - ETA: 2s
 4608/10000 [============>.................] - ETA: 2s
 4768/10000 [=============>................] - ETA: 1s
 4928/10000 [=============>................] - ETA: 1s
 5088/10000 [==============>...............] - ETA: 1s
 5248/10000 [==============>...............] - ETA: 1s
 5408/10000 [===============>..............] - ETA: 1s
 5568/10000 [===============>..............] - ETA: 1s
 5728/10000 [================>.............] - ETA: 1s
 5888/10000 [================>.............] - ETA: 1s
 6016/10000 [=================>............] - ETA: 1s
 6176/10000 [=================>............] - ETA: 1s
 6336/10000 [==================>...........] - ETA: 1s
 6496/10000 [==================>...........] - ETA: 1s
 6624/10000 [==================>...........] - ETA: 1s
 6752/10000 [===================>..........] - ETA: 1s
 6880/10000 [===================>..........] - ETA: 1s
 7040/10000 [====================>.........] - ETA: 1s
 7200/10000 [====================>.........] - ETA: 1s
 7360/10000 [=====================>........] - ETA: 0s
 7520/10000 [=====================>........] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
 7808/10000 [======================>.......] - ETA: 0s
 7968/10000 [======================>.......] - ETA: 0s
 8128/10000 [=======================>......] - ETA: 0s
 8288/10000 [=======================>......] - ETA: 0s
 8448/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8928/10000 [=========================>....] - ETA: 0s
 9088/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 375us/step
[[7.63723733e-08 7.84361820e-09 1.76697836e-06 ... 9.99997139e-01
  4.85496576e-09 7.69058147e-07]
 [1.23702728e-06 1.24911257e-05 9.99984741e-01 ... 6.65484334e-09
  2.43994066e-07 2.35351516e-10]
 [6.47230763e-07 9.99634504e-01 5.52613092e-05 ... 1.46640392e-04
  1.63413824e-05 3.12786824e-06]
 ...
 [1.13429756e-07 5.99201348e-06 3.72634048e-07 ... 1.72998421e-04
  1.12868856e-05 1.36274311e-05]
 [9.13718122e-06 7.71804082e-07 2.14927297e-07 ... 3.35590840e-07
  2.48534168e-04 9.83403311e-07]
 [1.05193658e-05 2.42704658e-07 2.88727897e-05 ... 6.78236800e-10
  3.42356600e-07 3.04557908e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.0481684825374512, 'accuracy_test:': 0.9850000143051147}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git pull --all    ;            ls &&  git add --all &&  git commit -m "ml_store"  ;            git push --all ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
Fetching origin
From github.com:arita37/mlmodels_store
   7aac53e..94fa8ed  master     -> origin/master
Updating 7aac53e..94fa8ed
Fast-forward
 ...-34_be4e81fe281eae9822d779771f5b85f7e37f3171.py | 3994 ++++++++++++++++++++
 ...-35_be4e81fe281eae9822d779771f5b85f7e37f3171.py | 3495 +++++++++++++++++
 2 files changed, 7489 insertions(+)
 create mode 100644 log_json/log_json_2020-05-16-00-34_be4e81fe281eae9822d779771f5b85f7e37f3171.py
 create mode 100644 log_test_cli/log_cli_2020-05-16-00-35_be4e81fe281eae9822d779771f5b85f7e37f3171.py
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
[master 886fd7a] ml_store
 1 file changed, 2046 insertions(+)
To github.com:arita37/mlmodels_store.git
   94fa8ed..886fd7a  master -> master





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
{'loss': 0.4183550179004669, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-16 00:39:39.615482: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1365, in _do_call
    return fn(*args)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1350, in _run_fn
    target_list, run_metadata)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1443, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
	 [[{{node save_1/RestoreV2}}]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1290, in restore
    {self.saver_def.filename_tensor_name: save_path})
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 956, in run
    run_metadata_ptr)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1180, in _run
    feed_dict_tensor, options, run_metadata)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1359, in _do_run
    run_metadata)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/client/session.py", line 1384, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.NotFoundError: Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
    test(data_path="", pars_choice="test01", config_mode="test")
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
    self.build()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
    build_restore=build_restore)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
    restore_sequentially, reshape)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
    restore_sequentially)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
    name=name)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1300, in restore
    names_to_keys = object_graph_key_mapping(save_path)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1618, in object_graph_key_mapping
    object_graph_string = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/pywrap_tensorflow_internal.py", line 915, in get_tensor
    return CheckpointReader_GetTensor(self, compat.as_bytes(tensor_str))
tensorflow.python.framework.errors_impl.NotFoundError: Key _CHECKPOINTABLE_OBJECT_GRAPH not found in checkpoint

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
    test(data_path="", pars_choice="test01", config_mode="test")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
    session = load(out_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 477, in load_tf
    saver.restore(sess,  full_name)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 1306, in restore
    err, "a Variable name or other graph key that is missing")
tensorflow.python.framework.errors_impl.NotFoundError: Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:

Key Variable not found in checkpoint
	 [[node save_1/RestoreV2 (defined at opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py:1748) ]]

Original stack trace for 'save_1/RestoreV2':
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 332, in <module>
    test(data_path="", pars_choice="test01", config_mode="test")
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 320, in test
    session = load(out_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//1_lstm.py", line 199, in load
    return load_tf(load_pars)
  File "home/runner/work/mlmodels/mlmodels/mlmodels/util.py", line 474, in load_tf
    saver      = tf.compat.v1.train.Saver()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 828, in __init__
    self.build()
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 840, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 878, in _build
    build_restore=build_restore)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 508, in _build_internal
    restore_sequentially, reshape)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 328, in _AddRestoreOps
    restore_sequentially)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/training/saver.py", line 575, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_io_ops.py", line 1696, in restore_v2
    name=name)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/op_def_library.py", line 794, in _apply_op_helper
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/util/deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3357, in create_op
    attrs, op_def, compute_device)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 3426, in _create_op_internal
    op_def=op_def)
  File "opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/framework/ops.py", line 1748, in __init__
    self._traceback = tf_stack.extract_stack()


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
