
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '51b64e342c7b2661e79b8abaa33db92672ae95c7', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ************************************************************************************************************************

  ############Check model ################################ 

  ['model_keras.keras_gan', 'model_keras.nbeats', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.Autokeras', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.armdn', 'model_keras.02_cnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.pplm', 'model_tch.textcnn', 'model_tch.mlp'] 

  Used ['model_keras.keras_gan', 'model_keras.nbeats', 'model_keras.01_deepctr', 'model_keras.textvae', 'model_keras.namentity_crm_bilstm_dataloader', 'model_keras.Autokeras', 'model_keras.charcnn_zhang', 'model_keras.charcnn', 'model_keras.namentity_crm_bilstm', 'model_keras.textcnn', 'model_keras.armdn', 'model_keras.02_cnn', 'model_tf.1_lstm', 'model_tf.temporal_fusion_google', 'model_gluon.gluon_automl', 'model_gluon.fb_prophet', 'model_gluon.gluonts_model', 'model_sklearn.model_sklearn', 'model_sklearn.model_lightgbm', 'model_tch.nbeats', 'model_tch.transformer_classifier', 'model_tch.matchzoo_models', 'model_tch.torchhub', 'model_tch.03_nbeats_dataloader', 'model_tch.transformer_sentence', 'model_tch.pytorch_vae', 'model_tch.pplm', 'model_tch.textcnn', 'model_tch.mlp'] 





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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
Already up to date.
[master e487500] ml_store
 2 files changed, 64 insertions(+), 10298 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   44c7e2d..e487500  master -> master





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
[master de7ca2e] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   e487500..de7ca2e  master -> master





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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-21 12:12:09.756893: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 12:12:09.761654: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-21 12:12:09.761824: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55dd9129b840 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 12:12:09.761840: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 263
Trainable params: 263
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2504 - binary_crossentropy: 0.6938 - val_loss: 0.2501 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.24997821482170382}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
Total params: 263
Trainable params: 263
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 473
Trainable params: 473
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2815 - binary_crossentropy: 0.7634500/500 [==============================] - 1s 2ms/sample - loss: 0.2712 - binary_crossentropy: 0.7408 - val_loss: 0.2776 - val_binary_crossentropy: 0.7539

  #### metrics   #################################################### 
{'MSE': 0.27385427466600754}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 473
Trainable params: 473
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
Total params: 602
Trainable params: 602
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2502 - binary_crossentropy: 0.6936 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934

  #### metrics   #################################################### 
{'MSE': 0.24996954842972516}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
Total params: 602
Trainable params: 602
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 443
Trainable params: 443
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2670 - binary_crossentropy: 0.7338500/500 [==============================] - 1s 2ms/sample - loss: 0.2633 - binary_crossentropy: 0.7241 - val_loss: 0.2552 - val_binary_crossentropy: 0.7038

  #### metrics   #################################################### 
{'MSE': 0.2538729855191101}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 443
Trainable params: 443
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 123
Trainable params: 123
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2526 - binary_crossentropy: 0.6984500/500 [==============================] - 1s 3ms/sample - loss: 0.2502 - binary_crossentropy: 0.7197 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.25024928687647846}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 123
Trainable params: 123
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-21 12:13:21.425594: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:21.427537: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:21.432999: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 12:13:21.441652: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 12:13:21.443302: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 12:13:21.444719: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:21.446498: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2552 - val_binary_crossentropy: 0.7036
2020-05-21 12:13:22.478512: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:22.479968: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:22.483538: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 12:13:22.490843: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 12:13:22.492402: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 12:13:22.493748: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:22.494913: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.256574058476897}

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
2020-05-21 12:13:41.962855: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:41.964120: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:41.967349: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 12:13:41.972936: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 12:13:41.974007: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 12:13:41.974858: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:41.975704: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2518 - val_binary_crossentropy: 0.6968
2020-05-21 12:13:43.205615: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:43.206653: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:43.209026: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 12:13:43.215719: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 12:13:43.216658: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 12:13:43.217429: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:13:43.218136: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2523130775546232}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-21 12:14:13.072490: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:14:13.076803: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:14:13.089376: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 12:14:13.110547: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 12:14:13.114360: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 12:14:13.117942: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:14:13.121637: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 4s 4s/sample - loss: 0.5233 - binary_crossentropy: 1.2850 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934
2020-05-21 12:14:15.167829: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:14:15.171937: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:14:15.182572: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 12:14:15.205393: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 12:14:15.209000: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 12:14:15.212668: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 12:14:15.216372: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24831471674082906}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 700
Trainable params: 700
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3337 - binary_crossentropy: 0.9349500/500 [==============================] - 4s 8ms/sample - loss: 0.3340 - binary_crossentropy: 0.9738 - val_loss: 0.3218 - val_binary_crossentropy: 0.9168

  #### metrics   #################################################### 
{'MSE': 0.3271233521533729}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 700
Trainable params: 700
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         18          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         6           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_5[0][0]           
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
Total params: 233
Trainable params: 233
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.3163 - binary_crossentropy: 0.8590500/500 [==============================] - 4s 8ms/sample - loss: 0.2918 - binary_crossentropy: 0.7993 - val_loss: 0.2886 - val_binary_crossentropy: 0.7889

  #### metrics   #################################################### 
{'MSE': 0.28901563370265165}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         18          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         10          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         6           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_5[0][0]           
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
Total params: 233
Trainable params: 233
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 1,909
Trainable params: 1,909
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2688 - binary_crossentropy: 0.7322500/500 [==============================] - 4s 8ms/sample - loss: 0.2641 - binary_crossentropy: 0.7226 - val_loss: 0.2608 - val_binary_crossentropy: 0.7152

  #### metrics   #################################################### 
{'MSE': 0.25991892238571634}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 1,909
Trainable params: 1,909
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
regionsequence_mean (InputLayer [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         1           regionsequence_max[0][0]         
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
Total params: 72
Trainable params: 72
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.4601 - binary_crossentropy: 7.0955500/500 [==============================] - 5s 11ms/sample - loss: 0.4801 - binary_crossentropy: 7.4040 - val_loss: 0.5241 - val_binary_crossentropy: 8.0827

  #### metrics   #################################################### 
{'MSE': 0.502}

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
regionsequence_mean (InputLayer [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         2           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         1           regionsequence_max[0][0]         
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
Total params: 72
Trainable params: 72
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
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
Total params: 1,417
Trainable params: 1,417
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2896 - binary_crossentropy: 0.7855500/500 [==============================] - 5s 11ms/sample - loss: 0.2785 - binary_crossentropy: 0.7592 - val_loss: 0.2634 - val_binary_crossentropy: 0.7236

  #### metrics   #################################################### 
{'MSE': 0.2666235276150391}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
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
Total params: 1,417
Trainable params: 1,417
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         20          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_11[0][0]                    
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
Total params: 3,120
Trainable params: 3,040
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3183 - binary_crossentropy: 1.6236500/500 [==============================] - 6s 11ms/sample - loss: 0.2816 - binary_crossentropy: 1.3299 - val_loss: 0.2865 - val_binary_crossentropy: 1.5526

  #### metrics   #################################################### 
{'MSE': 0.2813623703220477}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         20          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 1, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_11[0][0]                    
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
Total params: 3,120
Trainable params: 3,040
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
   de7ca2e..1915659  master     -> origin/master
Updating de7ca2e..1915659
Fast-forward
 error_list/20200521/list_log_testall_20200521.md   | 769 ---------------------
 ...-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py | 625 +++++++++++++++++
 2 files changed, 625 insertions(+), 769 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-21-12-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py
[master b6649e6] ml_store
 1 file changed, 4954 insertions(+)
To github.com:arita37/mlmodels_store.git
   1915659..b6649e6  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
Already up to date.
[master 1e5c536] ml_store
 1 file changed, 52 insertions(+)
To github.com:arita37/mlmodels_store.git
   b6649e6..1e5c536  master -> master





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
[master 6859a7c] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   1e5c536..6859a7c  master -> master





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
[master d7d2a75] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   6859a7c..d7d2a75  master -> master





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

2020-05-21 12:22:40.792258: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 12:22:40.796705: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-21 12:22:40.796846: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56412e7bf170 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 12:22:40.796861: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 7s - loss: 1.3871
256/354 [====================>.........] - ETA: 3s - loss: 1.2794
354/354 [==============================] - 14s 40ms/step - loss: 1.3800 - val_loss: 1.6881

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
[master c4038b7] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   d7d2a75..c4038b7  master -> master





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
[master 4535872] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   c4038b7..4535872  master -> master





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
Already up to date.
[master 3c7c483] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   4535872..3c7c483  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3424256/17464789 [====>.........................] - ETA: 0s
10887168/17464789 [=================>............] - ETA: 0s
16195584/17464789 [==========================>...] - ETA: 0s
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
2020-05-21 12:23:42.163469: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 12:23:42.167406: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-21 12:23:42.167547: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x560c2b9a4150 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 12:23:42.167562: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5593 - accuracy: 0.5070
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5516 - accuracy: 0.5075 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5031 - accuracy: 0.5107
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5401 - accuracy: 0.5082
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6574 - accuracy: 0.5006
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6845 - accuracy: 0.4988
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6841 - accuracy: 0.4989
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6858 - accuracy: 0.4988
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6717 - accuracy: 0.4997
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6268 - accuracy: 0.5026
11000/25000 [============>.................] - ETA: 4s - loss: 7.6457 - accuracy: 0.5014
12000/25000 [=============>................] - ETA: 4s - loss: 7.6551 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6336 - accuracy: 0.5022
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6371 - accuracy: 0.5019
15000/25000 [=================>............] - ETA: 3s - loss: 7.6308 - accuracy: 0.5023
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6321 - accuracy: 0.5023
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6278 - accuracy: 0.5025
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6343 - accuracy: 0.5021
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6126 - accuracy: 0.5035
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6375 - accuracy: 0.5019
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6484 - accuracy: 0.5012
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
25000/25000 [==============================] - 9s 377us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f96411d3d30>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f9667995978> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7740 - accuracy: 0.4930 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7637 - accuracy: 0.4937
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8238 - accuracy: 0.4897
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8261 - accuracy: 0.4896
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8123 - accuracy: 0.4905
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7959 - accuracy: 0.4916
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.8372 - accuracy: 0.4889
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7893 - accuracy: 0.4920
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7939 - accuracy: 0.4917
11000/25000 [============>.................] - ETA: 4s - loss: 7.7600 - accuracy: 0.4939
12000/25000 [=============>................] - ETA: 4s - loss: 7.7177 - accuracy: 0.4967
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6831 - accuracy: 0.4989
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6798 - accuracy: 0.4991
15000/25000 [=================>............] - ETA: 3s - loss: 7.6881 - accuracy: 0.4986
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6992 - accuracy: 0.4979
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6738 - accuracy: 0.4995
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6751 - accuracy: 0.4994
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6626 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6505 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6520 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6620 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6679 - accuracy: 0.4999
25000/25000 [==============================] - 9s 378us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8353 - accuracy: 0.4890
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6206 - accuracy: 0.5030 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6053 - accuracy: 0.5040
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6475 - accuracy: 0.5013
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6896 - accuracy: 0.4985
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6820 - accuracy: 0.4990
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6743 - accuracy: 0.4995
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6871 - accuracy: 0.4987
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7096 - accuracy: 0.4972
11000/25000 [============>.................] - ETA: 4s - loss: 7.7321 - accuracy: 0.4957
12000/25000 [=============>................] - ETA: 4s - loss: 7.7062 - accuracy: 0.4974
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7091 - accuracy: 0.4972
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7345 - accuracy: 0.4956
15000/25000 [=================>............] - ETA: 3s - loss: 7.7116 - accuracy: 0.4971
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7069 - accuracy: 0.4974
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6964 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6854 - accuracy: 0.4988
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6699 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6681 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6708 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6773 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6743 - accuracy: 0.4995
25000/25000 [==============================] - 9s 376us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   3c7c483..d42d7d3  master     -> origin/master
Updating 3c7c483..d42d7d3
Fast-forward
 error_list/20200521/list_log_testall_20200521.md | 103 +++++++++++++++++++++++
 1 file changed, 103 insertions(+)
[master 3586586] ml_store
 1 file changed, 324 insertions(+)
To github.com:arita37/mlmodels_store.git
   d42d7d3..3586586  master -> master





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

13/13 [==============================] - 2s 125ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 6318d37] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   3586586..6318d37  master -> master





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
 3072000/11490434 [=======>......................] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:26 - loss: 2.3196 - categorical_accuracy: 0.1250
   64/60000 [..............................] - ETA: 4:43 - loss: 2.3068 - categorical_accuracy: 0.0938
   96/60000 [..............................] - ETA: 3:46 - loss: 2.2793 - categorical_accuracy: 0.1458
  128/60000 [..............................] - ETA: 3:16 - loss: 2.2344 - categorical_accuracy: 0.2031
  160/60000 [..............................] - ETA: 2:57 - loss: 2.2441 - categorical_accuracy: 0.2062
  192/60000 [..............................] - ETA: 2:45 - loss: 2.2308 - categorical_accuracy: 0.2135
  224/60000 [..............................] - ETA: 2:37 - loss: 2.2194 - categorical_accuracy: 0.2188
  256/60000 [..............................] - ETA: 2:31 - loss: 2.1843 - categorical_accuracy: 0.2461
  288/60000 [..............................] - ETA: 2:26 - loss: 2.1301 - categorical_accuracy: 0.2847
  320/60000 [..............................] - ETA: 2:22 - loss: 2.0956 - categorical_accuracy: 0.2906
  352/60000 [..............................] - ETA: 2:18 - loss: 2.0587 - categorical_accuracy: 0.3097
  384/60000 [..............................] - ETA: 2:15 - loss: 2.0155 - categorical_accuracy: 0.3281
  416/60000 [..............................] - ETA: 2:13 - loss: 1.9710 - categorical_accuracy: 0.3389
  448/60000 [..............................] - ETA: 2:11 - loss: 1.9506 - categorical_accuracy: 0.3460
  480/60000 [..............................] - ETA: 2:09 - loss: 1.9284 - categorical_accuracy: 0.3562
  512/60000 [..............................] - ETA: 2:08 - loss: 1.9172 - categorical_accuracy: 0.3594
  544/60000 [..............................] - ETA: 2:07 - loss: 1.8850 - categorical_accuracy: 0.3732
  576/60000 [..............................] - ETA: 2:05 - loss: 1.8627 - categorical_accuracy: 0.3802
  608/60000 [..............................] - ETA: 2:05 - loss: 1.8272 - categorical_accuracy: 0.3947
  640/60000 [..............................] - ETA: 2:04 - loss: 1.7955 - categorical_accuracy: 0.4047
  672/60000 [..............................] - ETA: 2:03 - loss: 1.7516 - categorical_accuracy: 0.4196
  704/60000 [..............................] - ETA: 2:02 - loss: 1.7268 - categorical_accuracy: 0.4247
  736/60000 [..............................] - ETA: 2:01 - loss: 1.7000 - categorical_accuracy: 0.4280
  768/60000 [..............................] - ETA: 2:00 - loss: 1.6763 - categorical_accuracy: 0.4375
  800/60000 [..............................] - ETA: 2:00 - loss: 1.6649 - categorical_accuracy: 0.4412
  832/60000 [..............................] - ETA: 1:59 - loss: 1.6451 - categorical_accuracy: 0.4483
  864/60000 [..............................] - ETA: 1:59 - loss: 1.6176 - categorical_accuracy: 0.4537
  896/60000 [..............................] - ETA: 1:59 - loss: 1.5970 - categorical_accuracy: 0.4632
  928/60000 [..............................] - ETA: 1:58 - loss: 1.5750 - categorical_accuracy: 0.4709
  960/60000 [..............................] - ETA: 1:58 - loss: 1.5470 - categorical_accuracy: 0.4781
  992/60000 [..............................] - ETA: 1:58 - loss: 1.5233 - categorical_accuracy: 0.4879
 1024/60000 [..............................] - ETA: 1:57 - loss: 1.5083 - categorical_accuracy: 0.4932
 1056/60000 [..............................] - ETA: 1:57 - loss: 1.4932 - categorical_accuracy: 0.4981
 1088/60000 [..............................] - ETA: 1:57 - loss: 1.4834 - categorical_accuracy: 0.5028
 1120/60000 [..............................] - ETA: 1:57 - loss: 1.4786 - categorical_accuracy: 0.5054
 1152/60000 [..............................] - ETA: 1:57 - loss: 1.4719 - categorical_accuracy: 0.5061
 1184/60000 [..............................] - ETA: 1:57 - loss: 1.4477 - categorical_accuracy: 0.5160
 1216/60000 [..............................] - ETA: 1:56 - loss: 1.4371 - categorical_accuracy: 0.5173
 1248/60000 [..............................] - ETA: 1:56 - loss: 1.4254 - categorical_accuracy: 0.5208
 1280/60000 [..............................] - ETA: 1:55 - loss: 1.4136 - categorical_accuracy: 0.5227
 1312/60000 [..............................] - ETA: 1:55 - loss: 1.3925 - categorical_accuracy: 0.5305
 1344/60000 [..............................] - ETA: 1:55 - loss: 1.3742 - categorical_accuracy: 0.5379
 1376/60000 [..............................] - ETA: 1:54 - loss: 1.3524 - categorical_accuracy: 0.5458
 1408/60000 [..............................] - ETA: 1:54 - loss: 1.3326 - categorical_accuracy: 0.5526
 1440/60000 [..............................] - ETA: 1:54 - loss: 1.3203 - categorical_accuracy: 0.5583
 1472/60000 [..............................] - ETA: 1:54 - loss: 1.3090 - categorical_accuracy: 0.5611
 1504/60000 [..............................] - ETA: 1:53 - loss: 1.2914 - categorical_accuracy: 0.5672
 1536/60000 [..............................] - ETA: 1:53 - loss: 1.2812 - categorical_accuracy: 0.5710
 1568/60000 [..............................] - ETA: 1:53 - loss: 1.2716 - categorical_accuracy: 0.5740
 1600/60000 [..............................] - ETA: 1:53 - loss: 1.2564 - categorical_accuracy: 0.5794
 1632/60000 [..............................] - ETA: 1:52 - loss: 1.2381 - categorical_accuracy: 0.5864
 1664/60000 [..............................] - ETA: 1:52 - loss: 1.2324 - categorical_accuracy: 0.5883
 1696/60000 [..............................] - ETA: 1:52 - loss: 1.2167 - categorical_accuracy: 0.5932
 1728/60000 [..............................] - ETA: 1:52 - loss: 1.2018 - categorical_accuracy: 0.5995
 1760/60000 [..............................] - ETA: 1:52 - loss: 1.1977 - categorical_accuracy: 0.6006
 1792/60000 [..............................] - ETA: 1:52 - loss: 1.1879 - categorical_accuracy: 0.6038
 1824/60000 [..............................] - ETA: 1:51 - loss: 1.1757 - categorical_accuracy: 0.6080
 1856/60000 [..............................] - ETA: 1:51 - loss: 1.1621 - categorical_accuracy: 0.6126
 1888/60000 [..............................] - ETA: 1:51 - loss: 1.1573 - categorical_accuracy: 0.6144
 1920/60000 [..............................] - ETA: 1:51 - loss: 1.1499 - categorical_accuracy: 0.6172
 1952/60000 [..............................] - ETA: 1:51 - loss: 1.1348 - categorical_accuracy: 0.6230
 1984/60000 [..............................] - ETA: 1:51 - loss: 1.1298 - categorical_accuracy: 0.6245
 2016/60000 [>.............................] - ETA: 1:51 - loss: 1.1189 - categorical_accuracy: 0.6285
 2048/60000 [>.............................] - ETA: 1:50 - loss: 1.1065 - categorical_accuracy: 0.6333
 2080/60000 [>.............................] - ETA: 1:50 - loss: 1.0971 - categorical_accuracy: 0.6361
 2112/60000 [>.............................] - ETA: 1:50 - loss: 1.0903 - categorical_accuracy: 0.6378
 2144/60000 [>.............................] - ETA: 1:50 - loss: 1.0805 - categorical_accuracy: 0.6409
 2176/60000 [>.............................] - ETA: 1:50 - loss: 1.0745 - categorical_accuracy: 0.6415
 2208/60000 [>.............................] - ETA: 1:50 - loss: 1.0641 - categorical_accuracy: 0.6454
 2240/60000 [>.............................] - ETA: 1:49 - loss: 1.0575 - categorical_accuracy: 0.6487
 2272/60000 [>.............................] - ETA: 1:49 - loss: 1.0475 - categorical_accuracy: 0.6523
 2304/60000 [>.............................] - ETA: 1:49 - loss: 1.0391 - categorical_accuracy: 0.6558
 2336/60000 [>.............................] - ETA: 1:49 - loss: 1.0297 - categorical_accuracy: 0.6588
 2368/60000 [>.............................] - ETA: 1:49 - loss: 1.0197 - categorical_accuracy: 0.6622
 2400/60000 [>.............................] - ETA: 1:49 - loss: 1.0119 - categorical_accuracy: 0.6642
 2432/60000 [>.............................] - ETA: 1:49 - loss: 1.0012 - categorical_accuracy: 0.6678
 2464/60000 [>.............................] - ETA: 1:49 - loss: 0.9933 - categorical_accuracy: 0.6705
 2496/60000 [>.............................] - ETA: 1:48 - loss: 0.9848 - categorical_accuracy: 0.6731
 2528/60000 [>.............................] - ETA: 1:48 - loss: 0.9758 - categorical_accuracy: 0.6760
 2560/60000 [>.............................] - ETA: 1:49 - loss: 0.9719 - categorical_accuracy: 0.6773
 2592/60000 [>.............................] - ETA: 1:48 - loss: 0.9686 - categorical_accuracy: 0.6798
 2624/60000 [>.............................] - ETA: 1:48 - loss: 0.9620 - categorical_accuracy: 0.6822
 2656/60000 [>.............................] - ETA: 1:48 - loss: 0.9543 - categorical_accuracy: 0.6856
 2688/60000 [>.............................] - ETA: 1:48 - loss: 0.9492 - categorical_accuracy: 0.6875
 2720/60000 [>.............................] - ETA: 1:48 - loss: 0.9416 - categorical_accuracy: 0.6901
 2752/60000 [>.............................] - ETA: 1:48 - loss: 0.9400 - categorical_accuracy: 0.6915
 2784/60000 [>.............................] - ETA: 1:47 - loss: 0.9398 - categorical_accuracy: 0.6911
 2816/60000 [>.............................] - ETA: 1:47 - loss: 0.9352 - categorical_accuracy: 0.6918
 2848/60000 [>.............................] - ETA: 1:47 - loss: 0.9281 - categorical_accuracy: 0.6938
 2880/60000 [>.............................] - ETA: 1:47 - loss: 0.9264 - categorical_accuracy: 0.6955
 2912/60000 [>.............................] - ETA: 1:47 - loss: 0.9211 - categorical_accuracy: 0.6975
 2944/60000 [>.............................] - ETA: 1:47 - loss: 0.9149 - categorical_accuracy: 0.6997
 2976/60000 [>.............................] - ETA: 1:47 - loss: 0.9090 - categorical_accuracy: 0.7016
 3008/60000 [>.............................] - ETA: 1:47 - loss: 0.9043 - categorical_accuracy: 0.7031
 3040/60000 [>.............................] - ETA: 1:47 - loss: 0.8991 - categorical_accuracy: 0.7049
 3072/60000 [>.............................] - ETA: 1:46 - loss: 0.8929 - categorical_accuracy: 0.7070
 3104/60000 [>.............................] - ETA: 1:46 - loss: 0.8905 - categorical_accuracy: 0.7078
 3136/60000 [>.............................] - ETA: 1:46 - loss: 0.8865 - categorical_accuracy: 0.7089
 3168/60000 [>.............................] - ETA: 1:46 - loss: 0.8816 - categorical_accuracy: 0.7105
 3200/60000 [>.............................] - ETA: 1:46 - loss: 0.8767 - categorical_accuracy: 0.7122
 3232/60000 [>.............................] - ETA: 1:46 - loss: 0.8719 - categorical_accuracy: 0.7141
 3264/60000 [>.............................] - ETA: 1:46 - loss: 0.8670 - categorical_accuracy: 0.7160
 3296/60000 [>.............................] - ETA: 1:46 - loss: 0.8614 - categorical_accuracy: 0.7178
 3328/60000 [>.............................] - ETA: 1:46 - loss: 0.8581 - categorical_accuracy: 0.7184
 3360/60000 [>.............................] - ETA: 1:45 - loss: 0.8517 - categorical_accuracy: 0.7205
 3392/60000 [>.............................] - ETA: 1:45 - loss: 0.8489 - categorical_accuracy: 0.7220
 3424/60000 [>.............................] - ETA: 1:45 - loss: 0.8425 - categorical_accuracy: 0.7240
 3456/60000 [>.............................] - ETA: 1:45 - loss: 0.8371 - categorical_accuracy: 0.7254
 3488/60000 [>.............................] - ETA: 1:45 - loss: 0.8325 - categorical_accuracy: 0.7271
 3520/60000 [>.............................] - ETA: 1:45 - loss: 0.8270 - categorical_accuracy: 0.7293
 3552/60000 [>.............................] - ETA: 1:45 - loss: 0.8227 - categorical_accuracy: 0.7303
 3584/60000 [>.............................] - ETA: 1:45 - loss: 0.8174 - categorical_accuracy: 0.7321
 3616/60000 [>.............................] - ETA: 1:45 - loss: 0.8118 - categorical_accuracy: 0.7340
 3648/60000 [>.............................] - ETA: 1:44 - loss: 0.8089 - categorical_accuracy: 0.7349
 3680/60000 [>.............................] - ETA: 1:44 - loss: 0.8066 - categorical_accuracy: 0.7359
 3712/60000 [>.............................] - ETA: 1:44 - loss: 0.8029 - categorical_accuracy: 0.7371
 3744/60000 [>.............................] - ETA: 1:44 - loss: 0.7967 - categorical_accuracy: 0.7393
 3776/60000 [>.............................] - ETA: 1:44 - loss: 0.7943 - categorical_accuracy: 0.7399
 3808/60000 [>.............................] - ETA: 1:44 - loss: 0.7921 - categorical_accuracy: 0.7408
 3840/60000 [>.............................] - ETA: 1:44 - loss: 0.7911 - categorical_accuracy: 0.7419
 3872/60000 [>.............................] - ETA: 1:44 - loss: 0.7877 - categorical_accuracy: 0.7435
 3904/60000 [>.............................] - ETA: 1:44 - loss: 0.7833 - categorical_accuracy: 0.7449
 3936/60000 [>.............................] - ETA: 1:44 - loss: 0.7778 - categorical_accuracy: 0.7470
 3968/60000 [>.............................] - ETA: 1:44 - loss: 0.7740 - categorical_accuracy: 0.7482
 4000/60000 [=>............................] - ETA: 1:44 - loss: 0.7703 - categorical_accuracy: 0.7495
 4032/60000 [=>............................] - ETA: 1:44 - loss: 0.7702 - categorical_accuracy: 0.7498
 4064/60000 [=>............................] - ETA: 1:43 - loss: 0.7670 - categorical_accuracy: 0.7502
 4096/60000 [=>............................] - ETA: 1:43 - loss: 0.7642 - categorical_accuracy: 0.7507
 4128/60000 [=>............................] - ETA: 1:43 - loss: 0.7614 - categorical_accuracy: 0.7517
 4160/60000 [=>............................] - ETA: 1:43 - loss: 0.7575 - categorical_accuracy: 0.7529
 4192/60000 [=>............................] - ETA: 1:43 - loss: 0.7551 - categorical_accuracy: 0.7536
 4224/60000 [=>............................] - ETA: 1:43 - loss: 0.7507 - categorical_accuracy: 0.7552
 4256/60000 [=>............................] - ETA: 1:43 - loss: 0.7473 - categorical_accuracy: 0.7566
 4288/60000 [=>............................] - ETA: 1:43 - loss: 0.7444 - categorical_accuracy: 0.7572
 4320/60000 [=>............................] - ETA: 1:43 - loss: 0.7418 - categorical_accuracy: 0.7579
 4352/60000 [=>............................] - ETA: 1:43 - loss: 0.7393 - categorical_accuracy: 0.7585
 4384/60000 [=>............................] - ETA: 1:43 - loss: 0.7346 - categorical_accuracy: 0.7603
 4416/60000 [=>............................] - ETA: 1:43 - loss: 0.7316 - categorical_accuracy: 0.7613
 4448/60000 [=>............................] - ETA: 1:43 - loss: 0.7284 - categorical_accuracy: 0.7624
 4480/60000 [=>............................] - ETA: 1:42 - loss: 0.7248 - categorical_accuracy: 0.7636
 4512/60000 [=>............................] - ETA: 1:42 - loss: 0.7220 - categorical_accuracy: 0.7640
 4544/60000 [=>............................] - ETA: 1:42 - loss: 0.7181 - categorical_accuracy: 0.7652
 4576/60000 [=>............................] - ETA: 1:42 - loss: 0.7151 - categorical_accuracy: 0.7660
 4608/60000 [=>............................] - ETA: 1:42 - loss: 0.7118 - categorical_accuracy: 0.7671
 4640/60000 [=>............................] - ETA: 1:42 - loss: 0.7115 - categorical_accuracy: 0.7670
 4672/60000 [=>............................] - ETA: 1:42 - loss: 0.7084 - categorical_accuracy: 0.7682
 4704/60000 [=>............................] - ETA: 1:42 - loss: 0.7070 - categorical_accuracy: 0.7685
 4736/60000 [=>............................] - ETA: 1:42 - loss: 0.7040 - categorical_accuracy: 0.7694
 4768/60000 [=>............................] - ETA: 1:42 - loss: 0.7005 - categorical_accuracy: 0.7706
 4800/60000 [=>............................] - ETA: 1:42 - loss: 0.6999 - categorical_accuracy: 0.7708
 4832/60000 [=>............................] - ETA: 1:41 - loss: 0.6956 - categorical_accuracy: 0.7724
 4864/60000 [=>............................] - ETA: 1:41 - loss: 0.6926 - categorical_accuracy: 0.7734
 4896/60000 [=>............................] - ETA: 1:41 - loss: 0.6894 - categorical_accuracy: 0.7743
 4928/60000 [=>............................] - ETA: 1:41 - loss: 0.6875 - categorical_accuracy: 0.7752
 4960/60000 [=>............................] - ETA: 1:41 - loss: 0.6843 - categorical_accuracy: 0.7762
 4992/60000 [=>............................] - ETA: 1:41 - loss: 0.6813 - categorical_accuracy: 0.7772
 5024/60000 [=>............................] - ETA: 1:41 - loss: 0.6791 - categorical_accuracy: 0.7781
 5056/60000 [=>............................] - ETA: 1:41 - loss: 0.6765 - categorical_accuracy: 0.7791
 5088/60000 [=>............................] - ETA: 1:41 - loss: 0.6734 - categorical_accuracy: 0.7803
 5120/60000 [=>............................] - ETA: 1:41 - loss: 0.6702 - categorical_accuracy: 0.7814
 5152/60000 [=>............................] - ETA: 1:41 - loss: 0.6681 - categorical_accuracy: 0.7822
 5184/60000 [=>............................] - ETA: 1:40 - loss: 0.6665 - categorical_accuracy: 0.7828
 5216/60000 [=>............................] - ETA: 1:40 - loss: 0.6638 - categorical_accuracy: 0.7839
 5248/60000 [=>............................] - ETA: 1:40 - loss: 0.6614 - categorical_accuracy: 0.7847
 5280/60000 [=>............................] - ETA: 1:40 - loss: 0.6579 - categorical_accuracy: 0.7860
 5312/60000 [=>............................] - ETA: 1:40 - loss: 0.6566 - categorical_accuracy: 0.7861
 5344/60000 [=>............................] - ETA: 1:40 - loss: 0.6553 - categorical_accuracy: 0.7865
 5376/60000 [=>............................] - ETA: 1:40 - loss: 0.6537 - categorical_accuracy: 0.7870
 5408/60000 [=>............................] - ETA: 1:40 - loss: 0.6510 - categorical_accuracy: 0.7879
 5440/60000 [=>............................] - ETA: 1:40 - loss: 0.6489 - categorical_accuracy: 0.7886
 5472/60000 [=>............................] - ETA: 1:40 - loss: 0.6459 - categorical_accuracy: 0.7897
 5504/60000 [=>............................] - ETA: 1:40 - loss: 0.6437 - categorical_accuracy: 0.7903
 5536/60000 [=>............................] - ETA: 1:40 - loss: 0.6404 - categorical_accuracy: 0.7915
 5568/60000 [=>............................] - ETA: 1:39 - loss: 0.6370 - categorical_accuracy: 0.7927
 5600/60000 [=>............................] - ETA: 1:39 - loss: 0.6344 - categorical_accuracy: 0.7936
 5632/60000 [=>............................] - ETA: 1:39 - loss: 0.6316 - categorical_accuracy: 0.7946
 5664/60000 [=>............................] - ETA: 1:39 - loss: 0.6306 - categorical_accuracy: 0.7947
 5696/60000 [=>............................] - ETA: 1:39 - loss: 0.6280 - categorical_accuracy: 0.7955
 5728/60000 [=>............................] - ETA: 1:39 - loss: 0.6253 - categorical_accuracy: 0.7963
 5760/60000 [=>............................] - ETA: 1:39 - loss: 0.6234 - categorical_accuracy: 0.7969
 5792/60000 [=>............................] - ETA: 1:39 - loss: 0.6205 - categorical_accuracy: 0.7978
 5824/60000 [=>............................] - ETA: 1:39 - loss: 0.6177 - categorical_accuracy: 0.7988
 5856/60000 [=>............................] - ETA: 1:39 - loss: 0.6163 - categorical_accuracy: 0.7997
 5888/60000 [=>............................] - ETA: 1:39 - loss: 0.6151 - categorical_accuracy: 0.8001
 5920/60000 [=>............................] - ETA: 1:39 - loss: 0.6127 - categorical_accuracy: 0.8008
 5952/60000 [=>............................] - ETA: 1:39 - loss: 0.6108 - categorical_accuracy: 0.8014
 5984/60000 [=>............................] - ETA: 1:39 - loss: 0.6085 - categorical_accuracy: 0.8021
 6016/60000 [==>...........................] - ETA: 1:39 - loss: 0.6080 - categorical_accuracy: 0.8029
 6048/60000 [==>...........................] - ETA: 1:39 - loss: 0.6062 - categorical_accuracy: 0.8036
 6080/60000 [==>...........................] - ETA: 1:39 - loss: 0.6043 - categorical_accuracy: 0.8041
 6112/60000 [==>...........................] - ETA: 1:39 - loss: 0.6037 - categorical_accuracy: 0.8045
 6144/60000 [==>...........................] - ETA: 1:39 - loss: 0.6012 - categorical_accuracy: 0.8053
 6176/60000 [==>...........................] - ETA: 1:39 - loss: 0.5998 - categorical_accuracy: 0.8060
 6208/60000 [==>...........................] - ETA: 1:39 - loss: 0.5982 - categorical_accuracy: 0.8062
 6240/60000 [==>...........................] - ETA: 1:38 - loss: 0.5959 - categorical_accuracy: 0.8072
 6272/60000 [==>...........................] - ETA: 1:38 - loss: 0.5941 - categorical_accuracy: 0.8076
 6304/60000 [==>...........................] - ETA: 1:38 - loss: 0.5933 - categorical_accuracy: 0.8077
 6336/60000 [==>...........................] - ETA: 1:38 - loss: 0.5914 - categorical_accuracy: 0.8084
 6368/60000 [==>...........................] - ETA: 1:38 - loss: 0.5899 - categorical_accuracy: 0.8090
 6400/60000 [==>...........................] - ETA: 1:38 - loss: 0.5884 - categorical_accuracy: 0.8095
 6432/60000 [==>...........................] - ETA: 1:38 - loss: 0.5861 - categorical_accuracy: 0.8103
 6464/60000 [==>...........................] - ETA: 1:38 - loss: 0.5838 - categorical_accuracy: 0.8111
 6496/60000 [==>...........................] - ETA: 1:38 - loss: 0.5818 - categorical_accuracy: 0.8116
 6528/60000 [==>...........................] - ETA: 1:38 - loss: 0.5798 - categorical_accuracy: 0.8120
 6560/60000 [==>...........................] - ETA: 1:38 - loss: 0.5784 - categorical_accuracy: 0.8125
 6592/60000 [==>...........................] - ETA: 1:38 - loss: 0.5771 - categorical_accuracy: 0.8130
 6624/60000 [==>...........................] - ETA: 1:38 - loss: 0.5749 - categorical_accuracy: 0.8136
 6656/60000 [==>...........................] - ETA: 1:38 - loss: 0.5735 - categorical_accuracy: 0.8137
 6688/60000 [==>...........................] - ETA: 1:38 - loss: 0.5717 - categorical_accuracy: 0.8143
 6720/60000 [==>...........................] - ETA: 1:38 - loss: 0.5694 - categorical_accuracy: 0.8152
 6752/60000 [==>...........................] - ETA: 1:38 - loss: 0.5680 - categorical_accuracy: 0.8158
 6784/60000 [==>...........................] - ETA: 1:37 - loss: 0.5659 - categorical_accuracy: 0.8166
 6816/60000 [==>...........................] - ETA: 1:37 - loss: 0.5645 - categorical_accuracy: 0.8170
 6848/60000 [==>...........................] - ETA: 1:37 - loss: 0.5630 - categorical_accuracy: 0.8176
 6880/60000 [==>...........................] - ETA: 1:37 - loss: 0.5614 - categorical_accuracy: 0.8180
 6912/60000 [==>...........................] - ETA: 1:37 - loss: 0.5606 - categorical_accuracy: 0.8181
 6944/60000 [==>...........................] - ETA: 1:37 - loss: 0.5597 - categorical_accuracy: 0.8181
 6976/60000 [==>...........................] - ETA: 1:37 - loss: 0.5580 - categorical_accuracy: 0.8187
 7008/60000 [==>...........................] - ETA: 1:37 - loss: 0.5558 - categorical_accuracy: 0.8193
 7040/60000 [==>...........................] - ETA: 1:37 - loss: 0.5540 - categorical_accuracy: 0.8197
 7072/60000 [==>...........................] - ETA: 1:37 - loss: 0.5536 - categorical_accuracy: 0.8201
 7104/60000 [==>...........................] - ETA: 1:37 - loss: 0.5514 - categorical_accuracy: 0.8208
 7136/60000 [==>...........................] - ETA: 1:37 - loss: 0.5494 - categorical_accuracy: 0.8215
 7168/60000 [==>...........................] - ETA: 1:37 - loss: 0.5494 - categorical_accuracy: 0.8214
 7200/60000 [==>...........................] - ETA: 1:37 - loss: 0.5487 - categorical_accuracy: 0.8215
 7232/60000 [==>...........................] - ETA: 1:37 - loss: 0.5467 - categorical_accuracy: 0.8223
 7264/60000 [==>...........................] - ETA: 1:37 - loss: 0.5459 - categorical_accuracy: 0.8224
 7296/60000 [==>...........................] - ETA: 1:36 - loss: 0.5442 - categorical_accuracy: 0.8231
 7328/60000 [==>...........................] - ETA: 1:36 - loss: 0.5437 - categorical_accuracy: 0.8233
 7360/60000 [==>...........................] - ETA: 1:36 - loss: 0.5419 - categorical_accuracy: 0.8239
 7392/60000 [==>...........................] - ETA: 1:36 - loss: 0.5402 - categorical_accuracy: 0.8245
 7424/60000 [==>...........................] - ETA: 1:36 - loss: 0.5385 - categorical_accuracy: 0.8250
 7456/60000 [==>...........................] - ETA: 1:36 - loss: 0.5368 - categorical_accuracy: 0.8255
 7488/60000 [==>...........................] - ETA: 1:36 - loss: 0.5350 - categorical_accuracy: 0.8261
 7520/60000 [==>...........................] - ETA: 1:36 - loss: 0.5335 - categorical_accuracy: 0.8265
 7552/60000 [==>...........................] - ETA: 1:36 - loss: 0.5327 - categorical_accuracy: 0.8269
 7584/60000 [==>...........................] - ETA: 1:36 - loss: 0.5312 - categorical_accuracy: 0.8274
 7616/60000 [==>...........................] - ETA: 1:36 - loss: 0.5292 - categorical_accuracy: 0.8281
 7648/60000 [==>...........................] - ETA: 1:36 - loss: 0.5274 - categorical_accuracy: 0.8288
 7680/60000 [==>...........................] - ETA: 1:36 - loss: 0.5264 - categorical_accuracy: 0.8292
 7712/60000 [==>...........................] - ETA: 1:36 - loss: 0.5248 - categorical_accuracy: 0.8296
 7744/60000 [==>...........................] - ETA: 1:36 - loss: 0.5239 - categorical_accuracy: 0.8301
 7776/60000 [==>...........................] - ETA: 1:36 - loss: 0.5228 - categorical_accuracy: 0.8306
 7808/60000 [==>...........................] - ETA: 1:35 - loss: 0.5217 - categorical_accuracy: 0.8311
 7840/60000 [==>...........................] - ETA: 1:35 - loss: 0.5200 - categorical_accuracy: 0.8315
 7872/60000 [==>...........................] - ETA: 1:35 - loss: 0.5195 - categorical_accuracy: 0.8317
 7904/60000 [==>...........................] - ETA: 1:35 - loss: 0.5178 - categorical_accuracy: 0.8321
 7936/60000 [==>...........................] - ETA: 1:35 - loss: 0.5169 - categorical_accuracy: 0.8323
 7968/60000 [==>...........................] - ETA: 1:35 - loss: 0.5163 - categorical_accuracy: 0.8326
 8000/60000 [===>..........................] - ETA: 1:35 - loss: 0.5146 - categorical_accuracy: 0.8330
 8032/60000 [===>..........................] - ETA: 1:35 - loss: 0.5137 - categorical_accuracy: 0.8332
 8064/60000 [===>..........................] - ETA: 1:35 - loss: 0.5119 - categorical_accuracy: 0.8337
 8096/60000 [===>..........................] - ETA: 1:35 - loss: 0.5108 - categorical_accuracy: 0.8340
 8128/60000 [===>..........................] - ETA: 1:35 - loss: 0.5093 - categorical_accuracy: 0.8345
 8160/60000 [===>..........................] - ETA: 1:35 - loss: 0.5093 - categorical_accuracy: 0.8348
 8192/60000 [===>..........................] - ETA: 1:35 - loss: 0.5082 - categorical_accuracy: 0.8352
 8224/60000 [===>..........................] - ETA: 1:35 - loss: 0.5065 - categorical_accuracy: 0.8357
 8256/60000 [===>..........................] - ETA: 1:34 - loss: 0.5048 - categorical_accuracy: 0.8362
 8288/60000 [===>..........................] - ETA: 1:34 - loss: 0.5041 - categorical_accuracy: 0.8366
 8320/60000 [===>..........................] - ETA: 1:34 - loss: 0.5029 - categorical_accuracy: 0.8370
 8352/60000 [===>..........................] - ETA: 1:34 - loss: 0.5013 - categorical_accuracy: 0.8375
 8384/60000 [===>..........................] - ETA: 1:34 - loss: 0.5006 - categorical_accuracy: 0.8379
 8416/60000 [===>..........................] - ETA: 1:34 - loss: 0.4998 - categorical_accuracy: 0.8382
 8448/60000 [===>..........................] - ETA: 1:34 - loss: 0.4983 - categorical_accuracy: 0.8388
 8480/60000 [===>..........................] - ETA: 1:34 - loss: 0.4973 - categorical_accuracy: 0.8390
 8512/60000 [===>..........................] - ETA: 1:34 - loss: 0.4965 - categorical_accuracy: 0.8393
 8544/60000 [===>..........................] - ETA: 1:34 - loss: 0.4955 - categorical_accuracy: 0.8397
 8576/60000 [===>..........................] - ETA: 1:34 - loss: 0.4942 - categorical_accuracy: 0.8400
 8608/60000 [===>..........................] - ETA: 1:34 - loss: 0.4932 - categorical_accuracy: 0.8401
 8640/60000 [===>..........................] - ETA: 1:34 - loss: 0.4916 - categorical_accuracy: 0.8406
 8672/60000 [===>..........................] - ETA: 1:34 - loss: 0.4916 - categorical_accuracy: 0.8408
 8704/60000 [===>..........................] - ETA: 1:34 - loss: 0.4911 - categorical_accuracy: 0.8410
 8736/60000 [===>..........................] - ETA: 1:34 - loss: 0.4895 - categorical_accuracy: 0.8415
 8768/60000 [===>..........................] - ETA: 1:33 - loss: 0.4879 - categorical_accuracy: 0.8420
 8800/60000 [===>..........................] - ETA: 1:33 - loss: 0.4863 - categorical_accuracy: 0.8426
 8832/60000 [===>..........................] - ETA: 1:33 - loss: 0.4848 - categorical_accuracy: 0.8431
 8864/60000 [===>..........................] - ETA: 1:33 - loss: 0.4833 - categorical_accuracy: 0.8436
 8896/60000 [===>..........................] - ETA: 1:33 - loss: 0.4820 - categorical_accuracy: 0.8441
 8928/60000 [===>..........................] - ETA: 1:33 - loss: 0.4815 - categorical_accuracy: 0.8440
 8960/60000 [===>..........................] - ETA: 1:33 - loss: 0.4803 - categorical_accuracy: 0.8444
 8992/60000 [===>..........................] - ETA: 1:33 - loss: 0.4803 - categorical_accuracy: 0.8446
 9024/60000 [===>..........................] - ETA: 1:33 - loss: 0.4787 - categorical_accuracy: 0.8452
 9056/60000 [===>..........................] - ETA: 1:33 - loss: 0.4778 - categorical_accuracy: 0.8455
 9088/60000 [===>..........................] - ETA: 1:33 - loss: 0.4768 - categorical_accuracy: 0.8457
 9120/60000 [===>..........................] - ETA: 1:33 - loss: 0.4755 - categorical_accuracy: 0.8462
 9152/60000 [===>..........................] - ETA: 1:33 - loss: 0.4746 - categorical_accuracy: 0.8465
 9184/60000 [===>..........................] - ETA: 1:33 - loss: 0.4735 - categorical_accuracy: 0.8469
 9216/60000 [===>..........................] - ETA: 1:33 - loss: 0.4725 - categorical_accuracy: 0.8473
 9248/60000 [===>..........................] - ETA: 1:33 - loss: 0.4712 - categorical_accuracy: 0.8478
 9280/60000 [===>..........................] - ETA: 1:32 - loss: 0.4699 - categorical_accuracy: 0.8482
 9312/60000 [===>..........................] - ETA: 1:32 - loss: 0.4700 - categorical_accuracy: 0.8484
 9344/60000 [===>..........................] - ETA: 1:32 - loss: 0.4710 - categorical_accuracy: 0.8486
 9376/60000 [===>..........................] - ETA: 1:32 - loss: 0.4703 - categorical_accuracy: 0.8490
 9408/60000 [===>..........................] - ETA: 1:32 - loss: 0.4694 - categorical_accuracy: 0.8493
 9440/60000 [===>..........................] - ETA: 1:32 - loss: 0.4684 - categorical_accuracy: 0.8497
 9472/60000 [===>..........................] - ETA: 1:32 - loss: 0.4672 - categorical_accuracy: 0.8500
 9504/60000 [===>..........................] - ETA: 1:32 - loss: 0.4664 - categorical_accuracy: 0.8504
 9536/60000 [===>..........................] - ETA: 1:32 - loss: 0.4656 - categorical_accuracy: 0.8506
 9568/60000 [===>..........................] - ETA: 1:32 - loss: 0.4651 - categorical_accuracy: 0.8509
 9600/60000 [===>..........................] - ETA: 1:32 - loss: 0.4639 - categorical_accuracy: 0.8514
 9632/60000 [===>..........................] - ETA: 1:32 - loss: 0.4634 - categorical_accuracy: 0.8514
 9664/60000 [===>..........................] - ETA: 1:32 - loss: 0.4623 - categorical_accuracy: 0.8518
 9696/60000 [===>..........................] - ETA: 1:32 - loss: 0.4626 - categorical_accuracy: 0.8519
 9728/60000 [===>..........................] - ETA: 1:32 - loss: 0.4617 - categorical_accuracy: 0.8520
 9760/60000 [===>..........................] - ETA: 1:32 - loss: 0.4613 - categorical_accuracy: 0.8523
 9792/60000 [===>..........................] - ETA: 1:31 - loss: 0.4605 - categorical_accuracy: 0.8524
 9824/60000 [===>..........................] - ETA: 1:31 - loss: 0.4605 - categorical_accuracy: 0.8525
 9856/60000 [===>..........................] - ETA: 1:31 - loss: 0.4594 - categorical_accuracy: 0.8530
 9888/60000 [===>..........................] - ETA: 1:31 - loss: 0.4584 - categorical_accuracy: 0.8534
 9920/60000 [===>..........................] - ETA: 1:31 - loss: 0.4572 - categorical_accuracy: 0.8537
 9952/60000 [===>..........................] - ETA: 1:31 - loss: 0.4562 - categorical_accuracy: 0.8541
 9984/60000 [===>..........................] - ETA: 1:31 - loss: 0.4552 - categorical_accuracy: 0.8545
10016/60000 [====>.........................] - ETA: 1:31 - loss: 0.4557 - categorical_accuracy: 0.8546
10048/60000 [====>.........................] - ETA: 1:31 - loss: 0.4550 - categorical_accuracy: 0.8550
10080/60000 [====>.........................] - ETA: 1:31 - loss: 0.4559 - categorical_accuracy: 0.8551
10112/60000 [====>.........................] - ETA: 1:31 - loss: 0.4552 - categorical_accuracy: 0.8552
10144/60000 [====>.........................] - ETA: 1:31 - loss: 0.4545 - categorical_accuracy: 0.8554
10176/60000 [====>.........................] - ETA: 1:31 - loss: 0.4536 - categorical_accuracy: 0.8556
10208/60000 [====>.........................] - ETA: 1:31 - loss: 0.4531 - categorical_accuracy: 0.8558
10240/60000 [====>.........................] - ETA: 1:31 - loss: 0.4520 - categorical_accuracy: 0.8561
10272/60000 [====>.........................] - ETA: 1:31 - loss: 0.4520 - categorical_accuracy: 0.8561
10304/60000 [====>.........................] - ETA: 1:31 - loss: 0.4512 - categorical_accuracy: 0.8564
10336/60000 [====>.........................] - ETA: 1:30 - loss: 0.4501 - categorical_accuracy: 0.8567
10368/60000 [====>.........................] - ETA: 1:30 - loss: 0.4496 - categorical_accuracy: 0.8568
10400/60000 [====>.........................] - ETA: 1:30 - loss: 0.4488 - categorical_accuracy: 0.8570
10432/60000 [====>.........................] - ETA: 1:30 - loss: 0.4479 - categorical_accuracy: 0.8573
10464/60000 [====>.........................] - ETA: 1:30 - loss: 0.4473 - categorical_accuracy: 0.8575
10496/60000 [====>.........................] - ETA: 1:30 - loss: 0.4469 - categorical_accuracy: 0.8578
10528/60000 [====>.........................] - ETA: 1:30 - loss: 0.4459 - categorical_accuracy: 0.8581
10560/60000 [====>.........................] - ETA: 1:30 - loss: 0.4450 - categorical_accuracy: 0.8584
10592/60000 [====>.........................] - ETA: 1:30 - loss: 0.4449 - categorical_accuracy: 0.8587
10624/60000 [====>.........................] - ETA: 1:30 - loss: 0.4439 - categorical_accuracy: 0.8589
10656/60000 [====>.........................] - ETA: 1:30 - loss: 0.4431 - categorical_accuracy: 0.8591
10688/60000 [====>.........................] - ETA: 1:30 - loss: 0.4428 - categorical_accuracy: 0.8594
10720/60000 [====>.........................] - ETA: 1:30 - loss: 0.4420 - categorical_accuracy: 0.8596
10752/60000 [====>.........................] - ETA: 1:30 - loss: 0.4409 - categorical_accuracy: 0.8599
10784/60000 [====>.........................] - ETA: 1:30 - loss: 0.4399 - categorical_accuracy: 0.8603
10816/60000 [====>.........................] - ETA: 1:29 - loss: 0.4390 - categorical_accuracy: 0.8607
10848/60000 [====>.........................] - ETA: 1:29 - loss: 0.4379 - categorical_accuracy: 0.8611
10880/60000 [====>.........................] - ETA: 1:29 - loss: 0.4372 - categorical_accuracy: 0.8613
10912/60000 [====>.........................] - ETA: 1:29 - loss: 0.4365 - categorical_accuracy: 0.8614
10944/60000 [====>.........................] - ETA: 1:29 - loss: 0.4357 - categorical_accuracy: 0.8617
10976/60000 [====>.........................] - ETA: 1:29 - loss: 0.4350 - categorical_accuracy: 0.8619
11008/60000 [====>.........................] - ETA: 1:29 - loss: 0.4344 - categorical_accuracy: 0.8621
11040/60000 [====>.........................] - ETA: 1:29 - loss: 0.4335 - categorical_accuracy: 0.8624
11072/60000 [====>.........................] - ETA: 1:29 - loss: 0.4328 - categorical_accuracy: 0.8626
11104/60000 [====>.........................] - ETA: 1:29 - loss: 0.4322 - categorical_accuracy: 0.8629
11136/60000 [====>.........................] - ETA: 1:29 - loss: 0.4311 - categorical_accuracy: 0.8632
11168/60000 [====>.........................] - ETA: 1:29 - loss: 0.4303 - categorical_accuracy: 0.8634
11200/60000 [====>.........................] - ETA: 1:29 - loss: 0.4299 - categorical_accuracy: 0.8635
11232/60000 [====>.........................] - ETA: 1:29 - loss: 0.4289 - categorical_accuracy: 0.8638
11264/60000 [====>.........................] - ETA: 1:29 - loss: 0.4278 - categorical_accuracy: 0.8642
11296/60000 [====>.........................] - ETA: 1:29 - loss: 0.4269 - categorical_accuracy: 0.8645
11328/60000 [====>.........................] - ETA: 1:29 - loss: 0.4259 - categorical_accuracy: 0.8648
11360/60000 [====>.........................] - ETA: 1:28 - loss: 0.4251 - categorical_accuracy: 0.8650
11392/60000 [====>.........................] - ETA: 1:28 - loss: 0.4242 - categorical_accuracy: 0.8652
11424/60000 [====>.........................] - ETA: 1:28 - loss: 0.4245 - categorical_accuracy: 0.8651
11456/60000 [====>.........................] - ETA: 1:28 - loss: 0.4249 - categorical_accuracy: 0.8651
11488/60000 [====>.........................] - ETA: 1:28 - loss: 0.4250 - categorical_accuracy: 0.8653
11520/60000 [====>.........................] - ETA: 1:28 - loss: 0.4247 - categorical_accuracy: 0.8655
11552/60000 [====>.........................] - ETA: 1:28 - loss: 0.4238 - categorical_accuracy: 0.8657
11584/60000 [====>.........................] - ETA: 1:28 - loss: 0.4231 - categorical_accuracy: 0.8658
11616/60000 [====>.........................] - ETA: 1:28 - loss: 0.4223 - categorical_accuracy: 0.8660
11648/60000 [====>.........................] - ETA: 1:28 - loss: 0.4219 - categorical_accuracy: 0.8661
11680/60000 [====>.........................] - ETA: 1:28 - loss: 0.4213 - categorical_accuracy: 0.8663
11712/60000 [====>.........................] - ETA: 1:28 - loss: 0.4205 - categorical_accuracy: 0.8665
11744/60000 [====>.........................] - ETA: 1:28 - loss: 0.4195 - categorical_accuracy: 0.8667
11776/60000 [====>.........................] - ETA: 1:28 - loss: 0.4186 - categorical_accuracy: 0.8669
11808/60000 [====>.........................] - ETA: 1:28 - loss: 0.4184 - categorical_accuracy: 0.8670
11840/60000 [====>.........................] - ETA: 1:27 - loss: 0.4179 - categorical_accuracy: 0.8672
11872/60000 [====>.........................] - ETA: 1:27 - loss: 0.4174 - categorical_accuracy: 0.8675
11904/60000 [====>.........................] - ETA: 1:27 - loss: 0.4166 - categorical_accuracy: 0.8677
11936/60000 [====>.........................] - ETA: 1:27 - loss: 0.4158 - categorical_accuracy: 0.8680
11968/60000 [====>.........................] - ETA: 1:27 - loss: 0.4149 - categorical_accuracy: 0.8682
12000/60000 [=====>........................] - ETA: 1:27 - loss: 0.4141 - categorical_accuracy: 0.8685
12032/60000 [=====>........................] - ETA: 1:27 - loss: 0.4133 - categorical_accuracy: 0.8687
12064/60000 [=====>........................] - ETA: 1:27 - loss: 0.4123 - categorical_accuracy: 0.8690
12096/60000 [=====>........................] - ETA: 1:27 - loss: 0.4116 - categorical_accuracy: 0.8693
12128/60000 [=====>........................] - ETA: 1:27 - loss: 0.4107 - categorical_accuracy: 0.8696
12160/60000 [=====>........................] - ETA: 1:27 - loss: 0.4107 - categorical_accuracy: 0.8695
12192/60000 [=====>........................] - ETA: 1:27 - loss: 0.4107 - categorical_accuracy: 0.8695
12224/60000 [=====>........................] - ETA: 1:27 - loss: 0.4098 - categorical_accuracy: 0.8698
12256/60000 [=====>........................] - ETA: 1:27 - loss: 0.4094 - categorical_accuracy: 0.8699
12288/60000 [=====>........................] - ETA: 1:27 - loss: 0.4094 - categorical_accuracy: 0.8700
12320/60000 [=====>........................] - ETA: 1:27 - loss: 0.4092 - categorical_accuracy: 0.8701
12352/60000 [=====>........................] - ETA: 1:26 - loss: 0.4096 - categorical_accuracy: 0.8700
12384/60000 [=====>........................] - ETA: 1:26 - loss: 0.4092 - categorical_accuracy: 0.8702
12416/60000 [=====>........................] - ETA: 1:26 - loss: 0.4085 - categorical_accuracy: 0.8704
12448/60000 [=====>........................] - ETA: 1:26 - loss: 0.4079 - categorical_accuracy: 0.8706
12480/60000 [=====>........................] - ETA: 1:26 - loss: 0.4075 - categorical_accuracy: 0.8707
12512/60000 [=====>........................] - ETA: 1:26 - loss: 0.4072 - categorical_accuracy: 0.8708
12544/60000 [=====>........................] - ETA: 1:26 - loss: 0.4062 - categorical_accuracy: 0.8711
12576/60000 [=====>........................] - ETA: 1:26 - loss: 0.4058 - categorical_accuracy: 0.8713
12608/60000 [=====>........................] - ETA: 1:26 - loss: 0.4052 - categorical_accuracy: 0.8714
12640/60000 [=====>........................] - ETA: 1:26 - loss: 0.4045 - categorical_accuracy: 0.8717
12672/60000 [=====>........................] - ETA: 1:26 - loss: 0.4041 - categorical_accuracy: 0.8719
12704/60000 [=====>........................] - ETA: 1:26 - loss: 0.4032 - categorical_accuracy: 0.8722
12736/60000 [=====>........................] - ETA: 1:26 - loss: 0.4028 - categorical_accuracy: 0.8724
12768/60000 [=====>........................] - ETA: 1:26 - loss: 0.4022 - categorical_accuracy: 0.8726
12800/60000 [=====>........................] - ETA: 1:26 - loss: 0.4020 - categorical_accuracy: 0.8727
12832/60000 [=====>........................] - ETA: 1:26 - loss: 0.4011 - categorical_accuracy: 0.8731
12864/60000 [=====>........................] - ETA: 1:26 - loss: 0.4005 - categorical_accuracy: 0.8731
12896/60000 [=====>........................] - ETA: 1:25 - loss: 0.4006 - categorical_accuracy: 0.8732
12928/60000 [=====>........................] - ETA: 1:25 - loss: 0.4002 - categorical_accuracy: 0.8734
12960/60000 [=====>........................] - ETA: 1:25 - loss: 0.4000 - categorical_accuracy: 0.8735
12992/60000 [=====>........................] - ETA: 1:25 - loss: 0.3996 - categorical_accuracy: 0.8737
13024/60000 [=====>........................] - ETA: 1:25 - loss: 0.3995 - categorical_accuracy: 0.8738
13056/60000 [=====>........................] - ETA: 1:25 - loss: 0.3987 - categorical_accuracy: 0.8741
13088/60000 [=====>........................] - ETA: 1:25 - loss: 0.3979 - categorical_accuracy: 0.8744
13120/60000 [=====>........................] - ETA: 1:25 - loss: 0.3972 - categorical_accuracy: 0.8746
13152/60000 [=====>........................] - ETA: 1:25 - loss: 0.3964 - categorical_accuracy: 0.8749
13184/60000 [=====>........................] - ETA: 1:25 - loss: 0.3959 - categorical_accuracy: 0.8751
13216/60000 [=====>........................] - ETA: 1:25 - loss: 0.3952 - categorical_accuracy: 0.8754
13248/60000 [=====>........................] - ETA: 1:25 - loss: 0.3946 - categorical_accuracy: 0.8755
13280/60000 [=====>........................] - ETA: 1:25 - loss: 0.3942 - categorical_accuracy: 0.8756
13312/60000 [=====>........................] - ETA: 1:25 - loss: 0.3938 - categorical_accuracy: 0.8757
13344/60000 [=====>........................] - ETA: 1:25 - loss: 0.3930 - categorical_accuracy: 0.8760
13376/60000 [=====>........................] - ETA: 1:25 - loss: 0.3922 - categorical_accuracy: 0.8763
13408/60000 [=====>........................] - ETA: 1:24 - loss: 0.3915 - categorical_accuracy: 0.8765
13440/60000 [=====>........................] - ETA: 1:24 - loss: 0.3906 - categorical_accuracy: 0.8768
13472/60000 [=====>........................] - ETA: 1:24 - loss: 0.3900 - categorical_accuracy: 0.8769
13504/60000 [=====>........................] - ETA: 1:24 - loss: 0.3895 - categorical_accuracy: 0.8770
13536/60000 [=====>........................] - ETA: 1:24 - loss: 0.3889 - categorical_accuracy: 0.8771
13568/60000 [=====>........................] - ETA: 1:24 - loss: 0.3885 - categorical_accuracy: 0.8773
13600/60000 [=====>........................] - ETA: 1:24 - loss: 0.3878 - categorical_accuracy: 0.8775
13632/60000 [=====>........................] - ETA: 1:24 - loss: 0.3870 - categorical_accuracy: 0.8778
13664/60000 [=====>........................] - ETA: 1:24 - loss: 0.3866 - categorical_accuracy: 0.8779
13696/60000 [=====>........................] - ETA: 1:24 - loss: 0.3858 - categorical_accuracy: 0.8781
13728/60000 [=====>........................] - ETA: 1:24 - loss: 0.3851 - categorical_accuracy: 0.8783
13760/60000 [=====>........................] - ETA: 1:24 - loss: 0.3847 - categorical_accuracy: 0.8785
13792/60000 [=====>........................] - ETA: 1:24 - loss: 0.3843 - categorical_accuracy: 0.8787
13824/60000 [=====>........................] - ETA: 1:24 - loss: 0.3837 - categorical_accuracy: 0.8788
13856/60000 [=====>........................] - ETA: 1:24 - loss: 0.3835 - categorical_accuracy: 0.8790
13888/60000 [=====>........................] - ETA: 1:24 - loss: 0.3833 - categorical_accuracy: 0.8791
13920/60000 [=====>........................] - ETA: 1:23 - loss: 0.3828 - categorical_accuracy: 0.8792
13952/60000 [=====>........................] - ETA: 1:23 - loss: 0.3820 - categorical_accuracy: 0.8794
13984/60000 [=====>........................] - ETA: 1:23 - loss: 0.3815 - categorical_accuracy: 0.8796
14016/60000 [======>.......................] - ETA: 1:23 - loss: 0.3812 - categorical_accuracy: 0.8796
14048/60000 [======>.......................] - ETA: 1:23 - loss: 0.3810 - categorical_accuracy: 0.8796
14080/60000 [======>.......................] - ETA: 1:23 - loss: 0.3803 - categorical_accuracy: 0.8798
14112/60000 [======>.......................] - ETA: 1:23 - loss: 0.3799 - categorical_accuracy: 0.8798
14144/60000 [======>.......................] - ETA: 1:23 - loss: 0.3795 - categorical_accuracy: 0.8799
14176/60000 [======>.......................] - ETA: 1:23 - loss: 0.3788 - categorical_accuracy: 0.8801
14208/60000 [======>.......................] - ETA: 1:23 - loss: 0.3787 - categorical_accuracy: 0.8801
14240/60000 [======>.......................] - ETA: 1:23 - loss: 0.3784 - categorical_accuracy: 0.8802
14272/60000 [======>.......................] - ETA: 1:23 - loss: 0.3782 - categorical_accuracy: 0.8803
14304/60000 [======>.......................] - ETA: 1:23 - loss: 0.3776 - categorical_accuracy: 0.8805
14336/60000 [======>.......................] - ETA: 1:23 - loss: 0.3769 - categorical_accuracy: 0.8807
14368/60000 [======>.......................] - ETA: 1:23 - loss: 0.3761 - categorical_accuracy: 0.8809
14400/60000 [======>.......................] - ETA: 1:23 - loss: 0.3758 - categorical_accuracy: 0.8810
14432/60000 [======>.......................] - ETA: 1:23 - loss: 0.3755 - categorical_accuracy: 0.8812
14464/60000 [======>.......................] - ETA: 1:22 - loss: 0.3751 - categorical_accuracy: 0.8812
14496/60000 [======>.......................] - ETA: 1:22 - loss: 0.3751 - categorical_accuracy: 0.8813
14528/60000 [======>.......................] - ETA: 1:22 - loss: 0.3747 - categorical_accuracy: 0.8815
14560/60000 [======>.......................] - ETA: 1:22 - loss: 0.3741 - categorical_accuracy: 0.8816
14592/60000 [======>.......................] - ETA: 1:22 - loss: 0.3736 - categorical_accuracy: 0.8817
14624/60000 [======>.......................] - ETA: 1:22 - loss: 0.3738 - categorical_accuracy: 0.8816
14656/60000 [======>.......................] - ETA: 1:22 - loss: 0.3734 - categorical_accuracy: 0.8818
14688/60000 [======>.......................] - ETA: 1:22 - loss: 0.3727 - categorical_accuracy: 0.8819
14720/60000 [======>.......................] - ETA: 1:22 - loss: 0.3722 - categorical_accuracy: 0.8820
14752/60000 [======>.......................] - ETA: 1:22 - loss: 0.3717 - categorical_accuracy: 0.8821
14784/60000 [======>.......................] - ETA: 1:22 - loss: 0.3715 - categorical_accuracy: 0.8822
14816/60000 [======>.......................] - ETA: 1:22 - loss: 0.3711 - categorical_accuracy: 0.8824
14848/60000 [======>.......................] - ETA: 1:22 - loss: 0.3703 - categorical_accuracy: 0.8826
14880/60000 [======>.......................] - ETA: 1:22 - loss: 0.3698 - categorical_accuracy: 0.8828
14912/60000 [======>.......................] - ETA: 1:22 - loss: 0.3692 - categorical_accuracy: 0.8830
14944/60000 [======>.......................] - ETA: 1:22 - loss: 0.3688 - categorical_accuracy: 0.8832
14976/60000 [======>.......................] - ETA: 1:22 - loss: 0.3684 - categorical_accuracy: 0.8832
15008/60000 [======>.......................] - ETA: 1:22 - loss: 0.3679 - categorical_accuracy: 0.8834
15040/60000 [======>.......................] - ETA: 1:22 - loss: 0.3675 - categorical_accuracy: 0.8836
15072/60000 [======>.......................] - ETA: 1:21 - loss: 0.3675 - categorical_accuracy: 0.8836
15104/60000 [======>.......................] - ETA: 1:21 - loss: 0.3668 - categorical_accuracy: 0.8838
15136/60000 [======>.......................] - ETA: 1:21 - loss: 0.3671 - categorical_accuracy: 0.8837
15168/60000 [======>.......................] - ETA: 1:21 - loss: 0.3667 - categorical_accuracy: 0.8838
15200/60000 [======>.......................] - ETA: 1:21 - loss: 0.3661 - categorical_accuracy: 0.8840
15232/60000 [======>.......................] - ETA: 1:21 - loss: 0.3656 - categorical_accuracy: 0.8841
15264/60000 [======>.......................] - ETA: 1:21 - loss: 0.3649 - categorical_accuracy: 0.8843
15296/60000 [======>.......................] - ETA: 1:21 - loss: 0.3644 - categorical_accuracy: 0.8845
15328/60000 [======>.......................] - ETA: 1:21 - loss: 0.3637 - categorical_accuracy: 0.8847
15360/60000 [======>.......................] - ETA: 1:21 - loss: 0.3634 - categorical_accuracy: 0.8848
15392/60000 [======>.......................] - ETA: 1:21 - loss: 0.3632 - categorical_accuracy: 0.8849
15424/60000 [======>.......................] - ETA: 1:21 - loss: 0.3628 - categorical_accuracy: 0.8850
15456/60000 [======>.......................] - ETA: 1:21 - loss: 0.3621 - categorical_accuracy: 0.8853
15488/60000 [======>.......................] - ETA: 1:21 - loss: 0.3618 - categorical_accuracy: 0.8854
15520/60000 [======>.......................] - ETA: 1:21 - loss: 0.3612 - categorical_accuracy: 0.8856
15552/60000 [======>.......................] - ETA: 1:21 - loss: 0.3613 - categorical_accuracy: 0.8857
15584/60000 [======>.......................] - ETA: 1:20 - loss: 0.3609 - categorical_accuracy: 0.8858
15616/60000 [======>.......................] - ETA: 1:20 - loss: 0.3604 - categorical_accuracy: 0.8860
15648/60000 [======>.......................] - ETA: 1:20 - loss: 0.3601 - categorical_accuracy: 0.8861
15680/60000 [======>.......................] - ETA: 1:20 - loss: 0.3602 - categorical_accuracy: 0.8860
15712/60000 [======>.......................] - ETA: 1:20 - loss: 0.3596 - categorical_accuracy: 0.8862
15744/60000 [======>.......................] - ETA: 1:20 - loss: 0.3599 - categorical_accuracy: 0.8864
15776/60000 [======>.......................] - ETA: 1:20 - loss: 0.3601 - categorical_accuracy: 0.8865
15808/60000 [======>.......................] - ETA: 1:20 - loss: 0.3596 - categorical_accuracy: 0.8866
15840/60000 [======>.......................] - ETA: 1:20 - loss: 0.3592 - categorical_accuracy: 0.8867
15872/60000 [======>.......................] - ETA: 1:20 - loss: 0.3587 - categorical_accuracy: 0.8869
15904/60000 [======>.......................] - ETA: 1:20 - loss: 0.3586 - categorical_accuracy: 0.8869
15936/60000 [======>.......................] - ETA: 1:20 - loss: 0.3581 - categorical_accuracy: 0.8871
15968/60000 [======>.......................] - ETA: 1:20 - loss: 0.3575 - categorical_accuracy: 0.8873
16000/60000 [=======>......................] - ETA: 1:20 - loss: 0.3569 - categorical_accuracy: 0.8876
16032/60000 [=======>......................] - ETA: 1:20 - loss: 0.3569 - categorical_accuracy: 0.8876
16064/60000 [=======>......................] - ETA: 1:20 - loss: 0.3564 - categorical_accuracy: 0.8878
16096/60000 [=======>......................] - ETA: 1:20 - loss: 0.3562 - categorical_accuracy: 0.8878
16128/60000 [=======>......................] - ETA: 1:20 - loss: 0.3559 - categorical_accuracy: 0.8879
16160/60000 [=======>......................] - ETA: 1:19 - loss: 0.3557 - categorical_accuracy: 0.8880
16192/60000 [=======>......................] - ETA: 1:19 - loss: 0.3551 - categorical_accuracy: 0.8882
16224/60000 [=======>......................] - ETA: 1:19 - loss: 0.3548 - categorical_accuracy: 0.8883
16256/60000 [=======>......................] - ETA: 1:19 - loss: 0.3545 - categorical_accuracy: 0.8883
16288/60000 [=======>......................] - ETA: 1:19 - loss: 0.3543 - categorical_accuracy: 0.8884
16320/60000 [=======>......................] - ETA: 1:19 - loss: 0.3539 - categorical_accuracy: 0.8885
16352/60000 [=======>......................] - ETA: 1:19 - loss: 0.3536 - categorical_accuracy: 0.8886
16384/60000 [=======>......................] - ETA: 1:19 - loss: 0.3533 - categorical_accuracy: 0.8886
16416/60000 [=======>......................] - ETA: 1:19 - loss: 0.3526 - categorical_accuracy: 0.8888
16448/60000 [=======>......................] - ETA: 1:19 - loss: 0.3521 - categorical_accuracy: 0.8890
16480/60000 [=======>......................] - ETA: 1:19 - loss: 0.3516 - categorical_accuracy: 0.8891
16512/60000 [=======>......................] - ETA: 1:19 - loss: 0.3512 - categorical_accuracy: 0.8893
16544/60000 [=======>......................] - ETA: 1:19 - loss: 0.3509 - categorical_accuracy: 0.8894
16576/60000 [=======>......................] - ETA: 1:19 - loss: 0.3503 - categorical_accuracy: 0.8895
16608/60000 [=======>......................] - ETA: 1:19 - loss: 0.3500 - categorical_accuracy: 0.8896
16640/60000 [=======>......................] - ETA: 1:19 - loss: 0.3496 - categorical_accuracy: 0.8897
16672/60000 [=======>......................] - ETA: 1:19 - loss: 0.3492 - categorical_accuracy: 0.8898
16704/60000 [=======>......................] - ETA: 1:18 - loss: 0.3491 - categorical_accuracy: 0.8899
16736/60000 [=======>......................] - ETA: 1:18 - loss: 0.3490 - categorical_accuracy: 0.8899
16768/60000 [=======>......................] - ETA: 1:18 - loss: 0.3491 - categorical_accuracy: 0.8900
16800/60000 [=======>......................] - ETA: 1:18 - loss: 0.3485 - categorical_accuracy: 0.8902
16832/60000 [=======>......................] - ETA: 1:18 - loss: 0.3482 - categorical_accuracy: 0.8903
16864/60000 [=======>......................] - ETA: 1:18 - loss: 0.3476 - categorical_accuracy: 0.8905
16896/60000 [=======>......................] - ETA: 1:18 - loss: 0.3473 - categorical_accuracy: 0.8906
16928/60000 [=======>......................] - ETA: 1:18 - loss: 0.3472 - categorical_accuracy: 0.8905
16960/60000 [=======>......................] - ETA: 1:18 - loss: 0.3468 - categorical_accuracy: 0.8907
16992/60000 [=======>......................] - ETA: 1:18 - loss: 0.3463 - categorical_accuracy: 0.8908
17024/60000 [=======>......................] - ETA: 1:18 - loss: 0.3459 - categorical_accuracy: 0.8910
17056/60000 [=======>......................] - ETA: 1:18 - loss: 0.3453 - categorical_accuracy: 0.8912
17088/60000 [=======>......................] - ETA: 1:18 - loss: 0.3450 - categorical_accuracy: 0.8913
17120/60000 [=======>......................] - ETA: 1:18 - loss: 0.3447 - categorical_accuracy: 0.8914
17152/60000 [=======>......................] - ETA: 1:18 - loss: 0.3448 - categorical_accuracy: 0.8913
17184/60000 [=======>......................] - ETA: 1:18 - loss: 0.3443 - categorical_accuracy: 0.8915
17216/60000 [=======>......................] - ETA: 1:18 - loss: 0.3440 - categorical_accuracy: 0.8916
17248/60000 [=======>......................] - ETA: 1:17 - loss: 0.3435 - categorical_accuracy: 0.8918
17280/60000 [=======>......................] - ETA: 1:17 - loss: 0.3431 - categorical_accuracy: 0.8919
17312/60000 [=======>......................] - ETA: 1:17 - loss: 0.3429 - categorical_accuracy: 0.8920
17344/60000 [=======>......................] - ETA: 1:17 - loss: 0.3430 - categorical_accuracy: 0.8921
17376/60000 [=======>......................] - ETA: 1:17 - loss: 0.3425 - categorical_accuracy: 0.8923
17408/60000 [=======>......................] - ETA: 1:17 - loss: 0.3420 - categorical_accuracy: 0.8924
17440/60000 [=======>......................] - ETA: 1:17 - loss: 0.3421 - categorical_accuracy: 0.8923
17472/60000 [=======>......................] - ETA: 1:17 - loss: 0.3417 - categorical_accuracy: 0.8924
17504/60000 [=======>......................] - ETA: 1:17 - loss: 0.3413 - categorical_accuracy: 0.8925
17536/60000 [=======>......................] - ETA: 1:17 - loss: 0.3409 - categorical_accuracy: 0.8926
17568/60000 [=======>......................] - ETA: 1:17 - loss: 0.3403 - categorical_accuracy: 0.8928
17600/60000 [=======>......................] - ETA: 1:17 - loss: 0.3400 - categorical_accuracy: 0.8929
17632/60000 [=======>......................] - ETA: 1:17 - loss: 0.3400 - categorical_accuracy: 0.8930
17664/60000 [=======>......................] - ETA: 1:17 - loss: 0.3396 - categorical_accuracy: 0.8931
17696/60000 [=======>......................] - ETA: 1:17 - loss: 0.3394 - categorical_accuracy: 0.8931
17728/60000 [=======>......................] - ETA: 1:17 - loss: 0.3391 - categorical_accuracy: 0.8933
17760/60000 [=======>......................] - ETA: 1:17 - loss: 0.3388 - categorical_accuracy: 0.8934
17792/60000 [=======>......................] - ETA: 1:16 - loss: 0.3383 - categorical_accuracy: 0.8935
17824/60000 [=======>......................] - ETA: 1:16 - loss: 0.3382 - categorical_accuracy: 0.8936
17856/60000 [=======>......................] - ETA: 1:16 - loss: 0.3382 - categorical_accuracy: 0.8936
17888/60000 [=======>......................] - ETA: 1:16 - loss: 0.3379 - categorical_accuracy: 0.8936
17920/60000 [=======>......................] - ETA: 1:16 - loss: 0.3374 - categorical_accuracy: 0.8938
17952/60000 [=======>......................] - ETA: 1:16 - loss: 0.3373 - categorical_accuracy: 0.8939
17984/60000 [=======>......................] - ETA: 1:16 - loss: 0.3369 - categorical_accuracy: 0.8940
18016/60000 [========>.....................] - ETA: 1:16 - loss: 0.3367 - categorical_accuracy: 0.8940
18048/60000 [========>.....................] - ETA: 1:16 - loss: 0.3362 - categorical_accuracy: 0.8942
18080/60000 [========>.....................] - ETA: 1:16 - loss: 0.3358 - categorical_accuracy: 0.8943
18112/60000 [========>.....................] - ETA: 1:16 - loss: 0.3352 - categorical_accuracy: 0.8945
18144/60000 [========>.....................] - ETA: 1:16 - loss: 0.3347 - categorical_accuracy: 0.8946
18176/60000 [========>.....................] - ETA: 1:16 - loss: 0.3342 - categorical_accuracy: 0.8948
18208/60000 [========>.....................] - ETA: 1:16 - loss: 0.3337 - categorical_accuracy: 0.8950
18240/60000 [========>.....................] - ETA: 1:16 - loss: 0.3332 - categorical_accuracy: 0.8952
18272/60000 [========>.....................] - ETA: 1:16 - loss: 0.3327 - categorical_accuracy: 0.8954
18304/60000 [========>.....................] - ETA: 1:15 - loss: 0.3330 - categorical_accuracy: 0.8954
18336/60000 [========>.....................] - ETA: 1:15 - loss: 0.3329 - categorical_accuracy: 0.8955
18368/60000 [========>.....................] - ETA: 1:15 - loss: 0.3329 - categorical_accuracy: 0.8955
18400/60000 [========>.....................] - ETA: 1:15 - loss: 0.3326 - categorical_accuracy: 0.8956
18432/60000 [========>.....................] - ETA: 1:15 - loss: 0.3323 - categorical_accuracy: 0.8957
18464/60000 [========>.....................] - ETA: 1:15 - loss: 0.3320 - categorical_accuracy: 0.8958
18496/60000 [========>.....................] - ETA: 1:15 - loss: 0.3317 - categorical_accuracy: 0.8959
18528/60000 [========>.....................] - ETA: 1:15 - loss: 0.3317 - categorical_accuracy: 0.8958
18560/60000 [========>.....................] - ETA: 1:15 - loss: 0.3314 - categorical_accuracy: 0.8959
18592/60000 [========>.....................] - ETA: 1:15 - loss: 0.3312 - categorical_accuracy: 0.8960
18624/60000 [========>.....................] - ETA: 1:15 - loss: 0.3309 - categorical_accuracy: 0.8961
18656/60000 [========>.....................] - ETA: 1:15 - loss: 0.3304 - categorical_accuracy: 0.8962
18688/60000 [========>.....................] - ETA: 1:15 - loss: 0.3301 - categorical_accuracy: 0.8964
18720/60000 [========>.....................] - ETA: 1:15 - loss: 0.3300 - categorical_accuracy: 0.8964
18752/60000 [========>.....................] - ETA: 1:15 - loss: 0.3296 - categorical_accuracy: 0.8965
18784/60000 [========>.....................] - ETA: 1:15 - loss: 0.3291 - categorical_accuracy: 0.8967
18816/60000 [========>.....................] - ETA: 1:15 - loss: 0.3287 - categorical_accuracy: 0.8968
18848/60000 [========>.....................] - ETA: 1:14 - loss: 0.3283 - categorical_accuracy: 0.8970
18880/60000 [========>.....................] - ETA: 1:14 - loss: 0.3278 - categorical_accuracy: 0.8971
18912/60000 [========>.....................] - ETA: 1:14 - loss: 0.3274 - categorical_accuracy: 0.8972
18944/60000 [========>.....................] - ETA: 1:14 - loss: 0.3269 - categorical_accuracy: 0.8974
18976/60000 [========>.....................] - ETA: 1:14 - loss: 0.3269 - categorical_accuracy: 0.8974
19008/60000 [========>.....................] - ETA: 1:14 - loss: 0.3267 - categorical_accuracy: 0.8974
19040/60000 [========>.....................] - ETA: 1:14 - loss: 0.3268 - categorical_accuracy: 0.8975
19072/60000 [========>.....................] - ETA: 1:14 - loss: 0.3264 - categorical_accuracy: 0.8977
19104/60000 [========>.....................] - ETA: 1:14 - loss: 0.3260 - categorical_accuracy: 0.8978
19136/60000 [========>.....................] - ETA: 1:14 - loss: 0.3257 - categorical_accuracy: 0.8978
19168/60000 [========>.....................] - ETA: 1:14 - loss: 0.3256 - categorical_accuracy: 0.8978
19200/60000 [========>.....................] - ETA: 1:14 - loss: 0.3251 - categorical_accuracy: 0.8980
19232/60000 [========>.....................] - ETA: 1:14 - loss: 0.3246 - categorical_accuracy: 0.8981
19264/60000 [========>.....................] - ETA: 1:14 - loss: 0.3243 - categorical_accuracy: 0.8982
19296/60000 [========>.....................] - ETA: 1:14 - loss: 0.3239 - categorical_accuracy: 0.8983
19328/60000 [========>.....................] - ETA: 1:14 - loss: 0.3235 - categorical_accuracy: 0.8984
19360/60000 [========>.....................] - ETA: 1:14 - loss: 0.3232 - categorical_accuracy: 0.8986
19392/60000 [========>.....................] - ETA: 1:14 - loss: 0.3228 - categorical_accuracy: 0.8987
19424/60000 [========>.....................] - ETA: 1:13 - loss: 0.3224 - categorical_accuracy: 0.8988
19456/60000 [========>.....................] - ETA: 1:13 - loss: 0.3219 - categorical_accuracy: 0.8990
19488/60000 [========>.....................] - ETA: 1:13 - loss: 0.3216 - categorical_accuracy: 0.8991
19520/60000 [========>.....................] - ETA: 1:13 - loss: 0.3213 - categorical_accuracy: 0.8992
19552/60000 [========>.....................] - ETA: 1:13 - loss: 0.3209 - categorical_accuracy: 0.8993
19584/60000 [========>.....................] - ETA: 1:13 - loss: 0.3206 - categorical_accuracy: 0.8994
19616/60000 [========>.....................] - ETA: 1:13 - loss: 0.3203 - categorical_accuracy: 0.8994
19648/60000 [========>.....................] - ETA: 1:13 - loss: 0.3199 - categorical_accuracy: 0.8996
19680/60000 [========>.....................] - ETA: 1:13 - loss: 0.3200 - categorical_accuracy: 0.8996
19712/60000 [========>.....................] - ETA: 1:13 - loss: 0.3196 - categorical_accuracy: 0.8997
19744/60000 [========>.....................] - ETA: 1:13 - loss: 0.3192 - categorical_accuracy: 0.8998
19776/60000 [========>.....................] - ETA: 1:13 - loss: 0.3188 - categorical_accuracy: 0.9000
19808/60000 [========>.....................] - ETA: 1:13 - loss: 0.3183 - categorical_accuracy: 0.9001
19840/60000 [========>.....................] - ETA: 1:13 - loss: 0.3180 - categorical_accuracy: 0.9002
19872/60000 [========>.....................] - ETA: 1:13 - loss: 0.3183 - categorical_accuracy: 0.9002
19904/60000 [========>.....................] - ETA: 1:13 - loss: 0.3182 - categorical_accuracy: 0.9003
19936/60000 [========>.....................] - ETA: 1:13 - loss: 0.3178 - categorical_accuracy: 0.9004
19968/60000 [========>.....................] - ETA: 1:12 - loss: 0.3175 - categorical_accuracy: 0.9004
20000/60000 [=========>....................] - ETA: 1:12 - loss: 0.3172 - categorical_accuracy: 0.9006
20032/60000 [=========>....................] - ETA: 1:12 - loss: 0.3168 - categorical_accuracy: 0.9007
20064/60000 [=========>....................] - ETA: 1:12 - loss: 0.3163 - categorical_accuracy: 0.9009
20096/60000 [=========>....................] - ETA: 1:12 - loss: 0.3158 - categorical_accuracy: 0.9010
20128/60000 [=========>....................] - ETA: 1:12 - loss: 0.3155 - categorical_accuracy: 0.9011
20160/60000 [=========>....................] - ETA: 1:12 - loss: 0.3151 - categorical_accuracy: 0.9013
20192/60000 [=========>....................] - ETA: 1:12 - loss: 0.3147 - categorical_accuracy: 0.9014
20224/60000 [=========>....................] - ETA: 1:12 - loss: 0.3143 - categorical_accuracy: 0.9015
20256/60000 [=========>....................] - ETA: 1:12 - loss: 0.3146 - categorical_accuracy: 0.9016
20288/60000 [=========>....................] - ETA: 1:12 - loss: 0.3145 - categorical_accuracy: 0.9016
20320/60000 [=========>....................] - ETA: 1:12 - loss: 0.3144 - categorical_accuracy: 0.9017
20352/60000 [=========>....................] - ETA: 1:12 - loss: 0.3141 - categorical_accuracy: 0.9017
20384/60000 [=========>....................] - ETA: 1:12 - loss: 0.3137 - categorical_accuracy: 0.9018
20416/60000 [=========>....................] - ETA: 1:12 - loss: 0.3132 - categorical_accuracy: 0.9020
20448/60000 [=========>....................] - ETA: 1:12 - loss: 0.3128 - categorical_accuracy: 0.9021
20480/60000 [=========>....................] - ETA: 1:11 - loss: 0.3126 - categorical_accuracy: 0.9021
20512/60000 [=========>....................] - ETA: 1:11 - loss: 0.3125 - categorical_accuracy: 0.9022
20544/60000 [=========>....................] - ETA: 1:11 - loss: 0.3121 - categorical_accuracy: 0.9023
20576/60000 [=========>....................] - ETA: 1:11 - loss: 0.3121 - categorical_accuracy: 0.9024
20608/60000 [=========>....................] - ETA: 1:11 - loss: 0.3117 - categorical_accuracy: 0.9025
20640/60000 [=========>....................] - ETA: 1:11 - loss: 0.3114 - categorical_accuracy: 0.9026
20672/60000 [=========>....................] - ETA: 1:11 - loss: 0.3110 - categorical_accuracy: 0.9027
20704/60000 [=========>....................] - ETA: 1:11 - loss: 0.3107 - categorical_accuracy: 0.9028
20736/60000 [=========>....................] - ETA: 1:11 - loss: 0.3102 - categorical_accuracy: 0.9030
20768/60000 [=========>....................] - ETA: 1:11 - loss: 0.3101 - categorical_accuracy: 0.9030
20800/60000 [=========>....................] - ETA: 1:11 - loss: 0.3097 - categorical_accuracy: 0.9031
20832/60000 [=========>....................] - ETA: 1:11 - loss: 0.3094 - categorical_accuracy: 0.9032
20864/60000 [=========>....................] - ETA: 1:11 - loss: 0.3094 - categorical_accuracy: 0.9032
20896/60000 [=========>....................] - ETA: 1:11 - loss: 0.3094 - categorical_accuracy: 0.9033
20928/60000 [=========>....................] - ETA: 1:11 - loss: 0.3090 - categorical_accuracy: 0.9034
20960/60000 [=========>....................] - ETA: 1:11 - loss: 0.3086 - categorical_accuracy: 0.9035
20992/60000 [=========>....................] - ETA: 1:11 - loss: 0.3082 - categorical_accuracy: 0.9037
21024/60000 [=========>....................] - ETA: 1:10 - loss: 0.3078 - categorical_accuracy: 0.9038
21056/60000 [=========>....................] - ETA: 1:10 - loss: 0.3074 - categorical_accuracy: 0.9039
21088/60000 [=========>....................] - ETA: 1:10 - loss: 0.3070 - categorical_accuracy: 0.9041
21120/60000 [=========>....................] - ETA: 1:10 - loss: 0.3071 - categorical_accuracy: 0.9040
21152/60000 [=========>....................] - ETA: 1:10 - loss: 0.3068 - categorical_accuracy: 0.9041
21184/60000 [=========>....................] - ETA: 1:10 - loss: 0.3064 - categorical_accuracy: 0.9043
21216/60000 [=========>....................] - ETA: 1:10 - loss: 0.3062 - categorical_accuracy: 0.9044
21248/60000 [=========>....................] - ETA: 1:10 - loss: 0.3059 - categorical_accuracy: 0.9044
21280/60000 [=========>....................] - ETA: 1:10 - loss: 0.3056 - categorical_accuracy: 0.9045
21312/60000 [=========>....................] - ETA: 1:10 - loss: 0.3053 - categorical_accuracy: 0.9046
21344/60000 [=========>....................] - ETA: 1:10 - loss: 0.3051 - categorical_accuracy: 0.9047
21376/60000 [=========>....................] - ETA: 1:10 - loss: 0.3048 - categorical_accuracy: 0.9048
21408/60000 [=========>....................] - ETA: 1:10 - loss: 0.3047 - categorical_accuracy: 0.9048
21440/60000 [=========>....................] - ETA: 1:10 - loss: 0.3050 - categorical_accuracy: 0.9048
21472/60000 [=========>....................] - ETA: 1:10 - loss: 0.3051 - categorical_accuracy: 0.9049
21504/60000 [=========>....................] - ETA: 1:10 - loss: 0.3049 - categorical_accuracy: 0.9049
21536/60000 [=========>....................] - ETA: 1:10 - loss: 0.3047 - categorical_accuracy: 0.9049
21568/60000 [=========>....................] - ETA: 1:10 - loss: 0.3047 - categorical_accuracy: 0.9050
21600/60000 [=========>....................] - ETA: 1:09 - loss: 0.3043 - categorical_accuracy: 0.9050
21632/60000 [=========>....................] - ETA: 1:09 - loss: 0.3044 - categorical_accuracy: 0.9050
21664/60000 [=========>....................] - ETA: 1:09 - loss: 0.3040 - categorical_accuracy: 0.9052
21696/60000 [=========>....................] - ETA: 1:09 - loss: 0.3040 - categorical_accuracy: 0.9052
21728/60000 [=========>....................] - ETA: 1:09 - loss: 0.3041 - categorical_accuracy: 0.9052
21760/60000 [=========>....................] - ETA: 1:09 - loss: 0.3037 - categorical_accuracy: 0.9053
21792/60000 [=========>....................] - ETA: 1:09 - loss: 0.3041 - categorical_accuracy: 0.9052
21824/60000 [=========>....................] - ETA: 1:09 - loss: 0.3039 - categorical_accuracy: 0.9052
21856/60000 [=========>....................] - ETA: 1:09 - loss: 0.3036 - categorical_accuracy: 0.9052
21888/60000 [=========>....................] - ETA: 1:09 - loss: 0.3032 - categorical_accuracy: 0.9054
21920/60000 [=========>....................] - ETA: 1:09 - loss: 0.3029 - categorical_accuracy: 0.9055
21952/60000 [=========>....................] - ETA: 1:09 - loss: 0.3025 - categorical_accuracy: 0.9056
21984/60000 [=========>....................] - ETA: 1:09 - loss: 0.3022 - categorical_accuracy: 0.9057
22016/60000 [==========>...................] - ETA: 1:09 - loss: 0.3019 - categorical_accuracy: 0.9058
22048/60000 [==========>...................] - ETA: 1:09 - loss: 0.3015 - categorical_accuracy: 0.9059
22080/60000 [==========>...................] - ETA: 1:09 - loss: 0.3012 - categorical_accuracy: 0.9060
22112/60000 [==========>...................] - ETA: 1:08 - loss: 0.3009 - categorical_accuracy: 0.9061
22144/60000 [==========>...................] - ETA: 1:08 - loss: 0.3008 - categorical_accuracy: 0.9061
22176/60000 [==========>...................] - ETA: 1:08 - loss: 0.3004 - categorical_accuracy: 0.9062
22208/60000 [==========>...................] - ETA: 1:08 - loss: 0.3001 - categorical_accuracy: 0.9064
22240/60000 [==========>...................] - ETA: 1:08 - loss: 0.2997 - categorical_accuracy: 0.9065
22272/60000 [==========>...................] - ETA: 1:08 - loss: 0.2994 - categorical_accuracy: 0.9067
22304/60000 [==========>...................] - ETA: 1:08 - loss: 0.2990 - categorical_accuracy: 0.9067
22336/60000 [==========>...................] - ETA: 1:08 - loss: 0.2987 - categorical_accuracy: 0.9069
22368/60000 [==========>...................] - ETA: 1:08 - loss: 0.2987 - categorical_accuracy: 0.9069
22400/60000 [==========>...................] - ETA: 1:08 - loss: 0.2984 - categorical_accuracy: 0.9070
22432/60000 [==========>...................] - ETA: 1:08 - loss: 0.2983 - categorical_accuracy: 0.9070
22464/60000 [==========>...................] - ETA: 1:08 - loss: 0.2979 - categorical_accuracy: 0.9071
22496/60000 [==========>...................] - ETA: 1:08 - loss: 0.2977 - categorical_accuracy: 0.9071
22528/60000 [==========>...................] - ETA: 1:08 - loss: 0.2975 - categorical_accuracy: 0.9072
22560/60000 [==========>...................] - ETA: 1:08 - loss: 0.2972 - categorical_accuracy: 0.9073
22592/60000 [==========>...................] - ETA: 1:08 - loss: 0.2974 - categorical_accuracy: 0.9073
22624/60000 [==========>...................] - ETA: 1:08 - loss: 0.2971 - categorical_accuracy: 0.9074
22656/60000 [==========>...................] - ETA: 1:07 - loss: 0.2968 - categorical_accuracy: 0.9074
22688/60000 [==========>...................] - ETA: 1:07 - loss: 0.2965 - categorical_accuracy: 0.9075
22720/60000 [==========>...................] - ETA: 1:07 - loss: 0.2963 - categorical_accuracy: 0.9076
22752/60000 [==========>...................] - ETA: 1:07 - loss: 0.2960 - categorical_accuracy: 0.9077
22784/60000 [==========>...................] - ETA: 1:07 - loss: 0.2957 - categorical_accuracy: 0.9078
22816/60000 [==========>...................] - ETA: 1:07 - loss: 0.2954 - categorical_accuracy: 0.9079
22848/60000 [==========>...................] - ETA: 1:07 - loss: 0.2953 - categorical_accuracy: 0.9080
22880/60000 [==========>...................] - ETA: 1:07 - loss: 0.2949 - categorical_accuracy: 0.9081
22912/60000 [==========>...................] - ETA: 1:07 - loss: 0.2947 - categorical_accuracy: 0.9082
22944/60000 [==========>...................] - ETA: 1:07 - loss: 0.2947 - categorical_accuracy: 0.9082
22976/60000 [==========>...................] - ETA: 1:07 - loss: 0.2945 - categorical_accuracy: 0.9082
23008/60000 [==========>...................] - ETA: 1:07 - loss: 0.2943 - categorical_accuracy: 0.9083
23040/60000 [==========>...................] - ETA: 1:07 - loss: 0.2943 - categorical_accuracy: 0.9083
23072/60000 [==========>...................] - ETA: 1:07 - loss: 0.2940 - categorical_accuracy: 0.9084
23104/60000 [==========>...................] - ETA: 1:07 - loss: 0.2940 - categorical_accuracy: 0.9084
23136/60000 [==========>...................] - ETA: 1:07 - loss: 0.2939 - categorical_accuracy: 0.9084
23168/60000 [==========>...................] - ETA: 1:06 - loss: 0.2937 - categorical_accuracy: 0.9085
23200/60000 [==========>...................] - ETA: 1:06 - loss: 0.2935 - categorical_accuracy: 0.9085
23232/60000 [==========>...................] - ETA: 1:06 - loss: 0.2931 - categorical_accuracy: 0.9087
23264/60000 [==========>...................] - ETA: 1:06 - loss: 0.2928 - categorical_accuracy: 0.9087
23296/60000 [==========>...................] - ETA: 1:06 - loss: 0.2925 - categorical_accuracy: 0.9088
23328/60000 [==========>...................] - ETA: 1:06 - loss: 0.2922 - categorical_accuracy: 0.9089
23360/60000 [==========>...................] - ETA: 1:06 - loss: 0.2920 - categorical_accuracy: 0.9089
23392/60000 [==========>...................] - ETA: 1:06 - loss: 0.2918 - categorical_accuracy: 0.9090
23424/60000 [==========>...................] - ETA: 1:06 - loss: 0.2915 - categorical_accuracy: 0.9091
23456/60000 [==========>...................] - ETA: 1:06 - loss: 0.2911 - categorical_accuracy: 0.9092
23488/60000 [==========>...................] - ETA: 1:06 - loss: 0.2908 - categorical_accuracy: 0.9093
23520/60000 [==========>...................] - ETA: 1:06 - loss: 0.2906 - categorical_accuracy: 0.9094
23552/60000 [==========>...................] - ETA: 1:06 - loss: 0.2906 - categorical_accuracy: 0.9095
23584/60000 [==========>...................] - ETA: 1:06 - loss: 0.2902 - categorical_accuracy: 0.9096
23616/60000 [==========>...................] - ETA: 1:06 - loss: 0.2900 - categorical_accuracy: 0.9096
23648/60000 [==========>...................] - ETA: 1:06 - loss: 0.2898 - categorical_accuracy: 0.9097
23680/60000 [==========>...................] - ETA: 1:06 - loss: 0.2895 - categorical_accuracy: 0.9098
23712/60000 [==========>...................] - ETA: 1:05 - loss: 0.2892 - categorical_accuracy: 0.9098
23744/60000 [==========>...................] - ETA: 1:05 - loss: 0.2889 - categorical_accuracy: 0.9100
23776/60000 [==========>...................] - ETA: 1:05 - loss: 0.2886 - categorical_accuracy: 0.9100
23808/60000 [==========>...................] - ETA: 1:05 - loss: 0.2886 - categorical_accuracy: 0.9100
23840/60000 [==========>...................] - ETA: 1:05 - loss: 0.2885 - categorical_accuracy: 0.9101
23872/60000 [==========>...................] - ETA: 1:05 - loss: 0.2882 - categorical_accuracy: 0.9102
23904/60000 [==========>...................] - ETA: 1:05 - loss: 0.2879 - categorical_accuracy: 0.9103
23936/60000 [==========>...................] - ETA: 1:05 - loss: 0.2876 - categorical_accuracy: 0.9104
23968/60000 [==========>...................] - ETA: 1:05 - loss: 0.2876 - categorical_accuracy: 0.9105
24000/60000 [===========>..................] - ETA: 1:05 - loss: 0.2873 - categorical_accuracy: 0.9105
24032/60000 [===========>..................] - ETA: 1:05 - loss: 0.2871 - categorical_accuracy: 0.9106
24064/60000 [===========>..................] - ETA: 1:05 - loss: 0.2868 - categorical_accuracy: 0.9107
24096/60000 [===========>..................] - ETA: 1:05 - loss: 0.2867 - categorical_accuracy: 0.9106
24128/60000 [===========>..................] - ETA: 1:05 - loss: 0.2871 - categorical_accuracy: 0.9106
24160/60000 [===========>..................] - ETA: 1:05 - loss: 0.2868 - categorical_accuracy: 0.9107
24192/60000 [===========>..................] - ETA: 1:05 - loss: 0.2866 - categorical_accuracy: 0.9108
24224/60000 [===========>..................] - ETA: 1:05 - loss: 0.2864 - categorical_accuracy: 0.9108
24256/60000 [===========>..................] - ETA: 1:04 - loss: 0.2862 - categorical_accuracy: 0.9109
24288/60000 [===========>..................] - ETA: 1:04 - loss: 0.2860 - categorical_accuracy: 0.9109
24320/60000 [===========>..................] - ETA: 1:04 - loss: 0.2857 - categorical_accuracy: 0.9110
24352/60000 [===========>..................] - ETA: 1:04 - loss: 0.2854 - categorical_accuracy: 0.9111
24384/60000 [===========>..................] - ETA: 1:04 - loss: 0.2851 - categorical_accuracy: 0.9112
24416/60000 [===========>..................] - ETA: 1:04 - loss: 0.2848 - categorical_accuracy: 0.9113
24448/60000 [===========>..................] - ETA: 1:04 - loss: 0.2845 - categorical_accuracy: 0.9114
24480/60000 [===========>..................] - ETA: 1:04 - loss: 0.2843 - categorical_accuracy: 0.9115
24512/60000 [===========>..................] - ETA: 1:04 - loss: 0.2840 - categorical_accuracy: 0.9115
24544/60000 [===========>..................] - ETA: 1:04 - loss: 0.2837 - categorical_accuracy: 0.9116
24576/60000 [===========>..................] - ETA: 1:04 - loss: 0.2836 - categorical_accuracy: 0.9117
24608/60000 [===========>..................] - ETA: 1:04 - loss: 0.2835 - categorical_accuracy: 0.9117
24640/60000 [===========>..................] - ETA: 1:04 - loss: 0.2834 - categorical_accuracy: 0.9116
24672/60000 [===========>..................] - ETA: 1:04 - loss: 0.2831 - categorical_accuracy: 0.9118
24704/60000 [===========>..................] - ETA: 1:04 - loss: 0.2830 - categorical_accuracy: 0.9118
24736/60000 [===========>..................] - ETA: 1:04 - loss: 0.2828 - categorical_accuracy: 0.9118
24768/60000 [===========>..................] - ETA: 1:03 - loss: 0.2826 - categorical_accuracy: 0.9119
24800/60000 [===========>..................] - ETA: 1:03 - loss: 0.2825 - categorical_accuracy: 0.9119
24832/60000 [===========>..................] - ETA: 1:03 - loss: 0.2824 - categorical_accuracy: 0.9120
24864/60000 [===========>..................] - ETA: 1:03 - loss: 0.2823 - categorical_accuracy: 0.9120
24896/60000 [===========>..................] - ETA: 1:03 - loss: 0.2821 - categorical_accuracy: 0.9121
24928/60000 [===========>..................] - ETA: 1:03 - loss: 0.2819 - categorical_accuracy: 0.9122
24960/60000 [===========>..................] - ETA: 1:03 - loss: 0.2815 - categorical_accuracy: 0.9123
24992/60000 [===========>..................] - ETA: 1:03 - loss: 0.2814 - categorical_accuracy: 0.9124
25024/60000 [===========>..................] - ETA: 1:03 - loss: 0.2813 - categorical_accuracy: 0.9124
25056/60000 [===========>..................] - ETA: 1:03 - loss: 0.2811 - categorical_accuracy: 0.9124
25088/60000 [===========>..................] - ETA: 1:03 - loss: 0.2811 - categorical_accuracy: 0.9124
25120/60000 [===========>..................] - ETA: 1:03 - loss: 0.2811 - categorical_accuracy: 0.9125
25152/60000 [===========>..................] - ETA: 1:03 - loss: 0.2813 - categorical_accuracy: 0.9124
25184/60000 [===========>..................] - ETA: 1:03 - loss: 0.2810 - categorical_accuracy: 0.9125
25216/60000 [===========>..................] - ETA: 1:03 - loss: 0.2808 - categorical_accuracy: 0.9126
25248/60000 [===========>..................] - ETA: 1:03 - loss: 0.2806 - categorical_accuracy: 0.9127
25280/60000 [===========>..................] - ETA: 1:03 - loss: 0.2803 - categorical_accuracy: 0.9128
25312/60000 [===========>..................] - ETA: 1:02 - loss: 0.2804 - categorical_accuracy: 0.9128
25344/60000 [===========>..................] - ETA: 1:02 - loss: 0.2801 - categorical_accuracy: 0.9129
25376/60000 [===========>..................] - ETA: 1:02 - loss: 0.2800 - categorical_accuracy: 0.9129
25408/60000 [===========>..................] - ETA: 1:02 - loss: 0.2797 - categorical_accuracy: 0.9130
25440/60000 [===========>..................] - ETA: 1:02 - loss: 0.2794 - categorical_accuracy: 0.9131
25472/60000 [===========>..................] - ETA: 1:02 - loss: 0.2792 - categorical_accuracy: 0.9132
25504/60000 [===========>..................] - ETA: 1:02 - loss: 0.2792 - categorical_accuracy: 0.9132
25536/60000 [===========>..................] - ETA: 1:02 - loss: 0.2789 - categorical_accuracy: 0.9133
25568/60000 [===========>..................] - ETA: 1:02 - loss: 0.2789 - categorical_accuracy: 0.9133
25600/60000 [===========>..................] - ETA: 1:02 - loss: 0.2786 - categorical_accuracy: 0.9134
25632/60000 [===========>..................] - ETA: 1:02 - loss: 0.2784 - categorical_accuracy: 0.9134
25664/60000 [===========>..................] - ETA: 1:02 - loss: 0.2785 - categorical_accuracy: 0.9134
25696/60000 [===========>..................] - ETA: 1:02 - loss: 0.2784 - categorical_accuracy: 0.9134
25728/60000 [===========>..................] - ETA: 1:02 - loss: 0.2784 - categorical_accuracy: 0.9134
25760/60000 [===========>..................] - ETA: 1:02 - loss: 0.2782 - categorical_accuracy: 0.9135
25792/60000 [===========>..................] - ETA: 1:02 - loss: 0.2780 - categorical_accuracy: 0.9135
25824/60000 [===========>..................] - ETA: 1:02 - loss: 0.2778 - categorical_accuracy: 0.9135
25856/60000 [===========>..................] - ETA: 1:01 - loss: 0.2775 - categorical_accuracy: 0.9136
25888/60000 [===========>..................] - ETA: 1:01 - loss: 0.2773 - categorical_accuracy: 0.9137
25920/60000 [===========>..................] - ETA: 1:01 - loss: 0.2774 - categorical_accuracy: 0.9136
25952/60000 [===========>..................] - ETA: 1:01 - loss: 0.2772 - categorical_accuracy: 0.9136
25984/60000 [===========>..................] - ETA: 1:01 - loss: 0.2770 - categorical_accuracy: 0.9137
26016/60000 [============>.................] - ETA: 1:01 - loss: 0.2768 - categorical_accuracy: 0.9138
26048/60000 [============>.................] - ETA: 1:01 - loss: 0.2766 - categorical_accuracy: 0.9139
26080/60000 [============>.................] - ETA: 1:01 - loss: 0.2763 - categorical_accuracy: 0.9140
26112/60000 [============>.................] - ETA: 1:01 - loss: 0.2760 - categorical_accuracy: 0.9141
26144/60000 [============>.................] - ETA: 1:01 - loss: 0.2760 - categorical_accuracy: 0.9141
26176/60000 [============>.................] - ETA: 1:01 - loss: 0.2759 - categorical_accuracy: 0.9142
26208/60000 [============>.................] - ETA: 1:01 - loss: 0.2757 - categorical_accuracy: 0.9142
26240/60000 [============>.................] - ETA: 1:01 - loss: 0.2755 - categorical_accuracy: 0.9143
26272/60000 [============>.................] - ETA: 1:01 - loss: 0.2754 - categorical_accuracy: 0.9143
26304/60000 [============>.................] - ETA: 1:01 - loss: 0.2753 - categorical_accuracy: 0.9143
26336/60000 [============>.................] - ETA: 1:01 - loss: 0.2751 - categorical_accuracy: 0.9144
26368/60000 [============>.................] - ETA: 1:01 - loss: 0.2749 - categorical_accuracy: 0.9144
26400/60000 [============>.................] - ETA: 1:00 - loss: 0.2747 - categorical_accuracy: 0.9145
26432/60000 [============>.................] - ETA: 1:00 - loss: 0.2745 - categorical_accuracy: 0.9145
26464/60000 [============>.................] - ETA: 1:00 - loss: 0.2746 - categorical_accuracy: 0.9146
26496/60000 [============>.................] - ETA: 1:00 - loss: 0.2743 - categorical_accuracy: 0.9147
26528/60000 [============>.................] - ETA: 1:00 - loss: 0.2741 - categorical_accuracy: 0.9147
26560/60000 [============>.................] - ETA: 1:00 - loss: 0.2740 - categorical_accuracy: 0.9147
26592/60000 [============>.................] - ETA: 1:00 - loss: 0.2737 - categorical_accuracy: 0.9148
26624/60000 [============>.................] - ETA: 1:00 - loss: 0.2735 - categorical_accuracy: 0.9149
26656/60000 [============>.................] - ETA: 1:00 - loss: 0.2732 - categorical_accuracy: 0.9150
26688/60000 [============>.................] - ETA: 1:00 - loss: 0.2730 - categorical_accuracy: 0.9150
26720/60000 [============>.................] - ETA: 1:00 - loss: 0.2728 - categorical_accuracy: 0.9151
26752/60000 [============>.................] - ETA: 1:00 - loss: 0.2729 - categorical_accuracy: 0.9151
26784/60000 [============>.................] - ETA: 1:00 - loss: 0.2730 - categorical_accuracy: 0.9151
26816/60000 [============>.................] - ETA: 1:00 - loss: 0.2729 - categorical_accuracy: 0.9151
26848/60000 [============>.................] - ETA: 1:00 - loss: 0.2728 - categorical_accuracy: 0.9151
26880/60000 [============>.................] - ETA: 1:00 - loss: 0.2725 - categorical_accuracy: 0.9152
26912/60000 [============>.................] - ETA: 1:00 - loss: 0.2723 - categorical_accuracy: 0.9152
26944/60000 [============>.................] - ETA: 59s - loss: 0.2720 - categorical_accuracy: 0.9153 
26976/60000 [============>.................] - ETA: 59s - loss: 0.2718 - categorical_accuracy: 0.9153
27008/60000 [============>.................] - ETA: 59s - loss: 0.2716 - categorical_accuracy: 0.9154
27040/60000 [============>.................] - ETA: 59s - loss: 0.2717 - categorical_accuracy: 0.9154
27072/60000 [============>.................] - ETA: 59s - loss: 0.2715 - categorical_accuracy: 0.9155
27104/60000 [============>.................] - ETA: 59s - loss: 0.2714 - categorical_accuracy: 0.9155
27136/60000 [============>.................] - ETA: 59s - loss: 0.2712 - categorical_accuracy: 0.9155
27168/60000 [============>.................] - ETA: 59s - loss: 0.2710 - categorical_accuracy: 0.9156
27200/60000 [============>.................] - ETA: 59s - loss: 0.2708 - categorical_accuracy: 0.9157
27232/60000 [============>.................] - ETA: 59s - loss: 0.2705 - categorical_accuracy: 0.9158
27264/60000 [============>.................] - ETA: 59s - loss: 0.2707 - categorical_accuracy: 0.9157
27296/60000 [============>.................] - ETA: 59s - loss: 0.2705 - categorical_accuracy: 0.9158
27328/60000 [============>.................] - ETA: 59s - loss: 0.2704 - categorical_accuracy: 0.9158
27360/60000 [============>.................] - ETA: 59s - loss: 0.2702 - categorical_accuracy: 0.9159
27392/60000 [============>.................] - ETA: 59s - loss: 0.2701 - categorical_accuracy: 0.9159
27424/60000 [============>.................] - ETA: 59s - loss: 0.2699 - categorical_accuracy: 0.9159
27456/60000 [============>.................] - ETA: 59s - loss: 0.2699 - categorical_accuracy: 0.9160
27488/60000 [============>.................] - ETA: 58s - loss: 0.2696 - categorical_accuracy: 0.9161
27520/60000 [============>.................] - ETA: 58s - loss: 0.2695 - categorical_accuracy: 0.9161
27552/60000 [============>.................] - ETA: 58s - loss: 0.2694 - categorical_accuracy: 0.9161
27584/60000 [============>.................] - ETA: 58s - loss: 0.2692 - categorical_accuracy: 0.9162
27616/60000 [============>.................] - ETA: 58s - loss: 0.2690 - categorical_accuracy: 0.9162
27648/60000 [============>.................] - ETA: 58s - loss: 0.2687 - categorical_accuracy: 0.9163
27680/60000 [============>.................] - ETA: 58s - loss: 0.2685 - categorical_accuracy: 0.9164
27712/60000 [============>.................] - ETA: 58s - loss: 0.2683 - categorical_accuracy: 0.9164
27744/60000 [============>.................] - ETA: 58s - loss: 0.2680 - categorical_accuracy: 0.9165
27776/60000 [============>.................] - ETA: 58s - loss: 0.2677 - categorical_accuracy: 0.9166
27808/60000 [============>.................] - ETA: 58s - loss: 0.2675 - categorical_accuracy: 0.9167
27840/60000 [============>.................] - ETA: 58s - loss: 0.2672 - categorical_accuracy: 0.9168
27872/60000 [============>.................] - ETA: 58s - loss: 0.2670 - categorical_accuracy: 0.9169
27904/60000 [============>.................] - ETA: 58s - loss: 0.2667 - categorical_accuracy: 0.9170
27936/60000 [============>.................] - ETA: 58s - loss: 0.2665 - categorical_accuracy: 0.9171
27968/60000 [============>.................] - ETA: 58s - loss: 0.2662 - categorical_accuracy: 0.9172
28000/60000 [=============>................] - ETA: 58s - loss: 0.2660 - categorical_accuracy: 0.9172
28032/60000 [=============>................] - ETA: 57s - loss: 0.2660 - categorical_accuracy: 0.9172
28064/60000 [=============>................] - ETA: 57s - loss: 0.2661 - categorical_accuracy: 0.9173
28096/60000 [=============>................] - ETA: 57s - loss: 0.2659 - categorical_accuracy: 0.9173
28128/60000 [=============>................] - ETA: 57s - loss: 0.2658 - categorical_accuracy: 0.9174
28160/60000 [=============>................] - ETA: 57s - loss: 0.2655 - categorical_accuracy: 0.9175
28192/60000 [=============>................] - ETA: 57s - loss: 0.2652 - categorical_accuracy: 0.9176
28224/60000 [=============>................] - ETA: 57s - loss: 0.2650 - categorical_accuracy: 0.9176
28256/60000 [=============>................] - ETA: 57s - loss: 0.2648 - categorical_accuracy: 0.9177
28288/60000 [=============>................] - ETA: 57s - loss: 0.2645 - categorical_accuracy: 0.9178
28320/60000 [=============>................] - ETA: 57s - loss: 0.2642 - categorical_accuracy: 0.9179
28352/60000 [=============>................] - ETA: 57s - loss: 0.2641 - categorical_accuracy: 0.9179
28384/60000 [=============>................] - ETA: 57s - loss: 0.2639 - categorical_accuracy: 0.9179
28416/60000 [=============>................] - ETA: 57s - loss: 0.2636 - categorical_accuracy: 0.9180
28448/60000 [=============>................] - ETA: 57s - loss: 0.2635 - categorical_accuracy: 0.9180
28480/60000 [=============>................] - ETA: 57s - loss: 0.2632 - categorical_accuracy: 0.9181
28512/60000 [=============>................] - ETA: 57s - loss: 0.2631 - categorical_accuracy: 0.9181
28544/60000 [=============>................] - ETA: 57s - loss: 0.2629 - categorical_accuracy: 0.9182
28576/60000 [=============>................] - ETA: 56s - loss: 0.2627 - categorical_accuracy: 0.9182
28608/60000 [=============>................] - ETA: 56s - loss: 0.2625 - categorical_accuracy: 0.9183
28640/60000 [=============>................] - ETA: 56s - loss: 0.2625 - categorical_accuracy: 0.9183
28672/60000 [=============>................] - ETA: 56s - loss: 0.2623 - categorical_accuracy: 0.9184
28704/60000 [=============>................] - ETA: 56s - loss: 0.2623 - categorical_accuracy: 0.9184
28736/60000 [=============>................] - ETA: 56s - loss: 0.2626 - categorical_accuracy: 0.9183
28768/60000 [=============>................] - ETA: 56s - loss: 0.2624 - categorical_accuracy: 0.9184
28800/60000 [=============>................] - ETA: 56s - loss: 0.2623 - categorical_accuracy: 0.9183
28832/60000 [=============>................] - ETA: 56s - loss: 0.2621 - categorical_accuracy: 0.9184
28864/60000 [=============>................] - ETA: 56s - loss: 0.2620 - categorical_accuracy: 0.9184
28896/60000 [=============>................] - ETA: 56s - loss: 0.2618 - categorical_accuracy: 0.9185
28928/60000 [=============>................] - ETA: 56s - loss: 0.2617 - categorical_accuracy: 0.9185
28960/60000 [=============>................] - ETA: 56s - loss: 0.2615 - categorical_accuracy: 0.9185
28992/60000 [=============>................] - ETA: 56s - loss: 0.2612 - categorical_accuracy: 0.9186
29024/60000 [=============>................] - ETA: 56s - loss: 0.2611 - categorical_accuracy: 0.9187
29056/60000 [=============>................] - ETA: 56s - loss: 0.2609 - categorical_accuracy: 0.9187
29088/60000 [=============>................] - ETA: 56s - loss: 0.2608 - categorical_accuracy: 0.9187
29120/60000 [=============>................] - ETA: 55s - loss: 0.2605 - categorical_accuracy: 0.9188
29152/60000 [=============>................] - ETA: 55s - loss: 0.2603 - categorical_accuracy: 0.9189
29184/60000 [=============>................] - ETA: 55s - loss: 0.2603 - categorical_accuracy: 0.9189
29216/60000 [=============>................] - ETA: 55s - loss: 0.2600 - categorical_accuracy: 0.9190
29248/60000 [=============>................] - ETA: 55s - loss: 0.2599 - categorical_accuracy: 0.9191
29280/60000 [=============>................] - ETA: 55s - loss: 0.2596 - categorical_accuracy: 0.9192
29312/60000 [=============>................] - ETA: 55s - loss: 0.2597 - categorical_accuracy: 0.9192
29344/60000 [=============>................] - ETA: 55s - loss: 0.2594 - categorical_accuracy: 0.9193
29376/60000 [=============>................] - ETA: 55s - loss: 0.2591 - categorical_accuracy: 0.9194
29408/60000 [=============>................] - ETA: 55s - loss: 0.2588 - categorical_accuracy: 0.9195
29440/60000 [=============>................] - ETA: 55s - loss: 0.2586 - categorical_accuracy: 0.9196
29472/60000 [=============>................] - ETA: 55s - loss: 0.2585 - categorical_accuracy: 0.9196
29504/60000 [=============>................] - ETA: 55s - loss: 0.2582 - categorical_accuracy: 0.9197
29536/60000 [=============>................] - ETA: 55s - loss: 0.2582 - categorical_accuracy: 0.9198
29568/60000 [=============>................] - ETA: 55s - loss: 0.2580 - categorical_accuracy: 0.9198
29600/60000 [=============>................] - ETA: 55s - loss: 0.2578 - categorical_accuracy: 0.9199
29632/60000 [=============>................] - ETA: 55s - loss: 0.2578 - categorical_accuracy: 0.9199
29664/60000 [=============>................] - ETA: 55s - loss: 0.2578 - categorical_accuracy: 0.9199
29696/60000 [=============>................] - ETA: 54s - loss: 0.2577 - categorical_accuracy: 0.9199
29728/60000 [=============>................] - ETA: 54s - loss: 0.2574 - categorical_accuracy: 0.9200
29760/60000 [=============>................] - ETA: 54s - loss: 0.2572 - categorical_accuracy: 0.9201
29792/60000 [=============>................] - ETA: 54s - loss: 0.2571 - categorical_accuracy: 0.9201
29824/60000 [=============>................] - ETA: 54s - loss: 0.2568 - categorical_accuracy: 0.9202
29856/60000 [=============>................] - ETA: 54s - loss: 0.2566 - categorical_accuracy: 0.9203
29888/60000 [=============>................] - ETA: 54s - loss: 0.2564 - categorical_accuracy: 0.9203
29920/60000 [=============>................] - ETA: 54s - loss: 0.2563 - categorical_accuracy: 0.9203
29952/60000 [=============>................] - ETA: 54s - loss: 0.2561 - categorical_accuracy: 0.9203
29984/60000 [=============>................] - ETA: 54s - loss: 0.2560 - categorical_accuracy: 0.9204
30016/60000 [==============>...............] - ETA: 54s - loss: 0.2559 - categorical_accuracy: 0.9204
30048/60000 [==============>...............] - ETA: 54s - loss: 0.2559 - categorical_accuracy: 0.9204
30080/60000 [==============>...............] - ETA: 54s - loss: 0.2558 - categorical_accuracy: 0.9204
30112/60000 [==============>...............] - ETA: 54s - loss: 0.2556 - categorical_accuracy: 0.9205
30144/60000 [==============>...............] - ETA: 54s - loss: 0.2558 - categorical_accuracy: 0.9204
30176/60000 [==============>...............] - ETA: 54s - loss: 0.2557 - categorical_accuracy: 0.9205
30208/60000 [==============>...............] - ETA: 54s - loss: 0.2556 - categorical_accuracy: 0.9205
30240/60000 [==============>...............] - ETA: 53s - loss: 0.2554 - categorical_accuracy: 0.9206
30272/60000 [==============>...............] - ETA: 53s - loss: 0.2552 - categorical_accuracy: 0.9207
30304/60000 [==============>...............] - ETA: 53s - loss: 0.2550 - categorical_accuracy: 0.9207
30336/60000 [==============>...............] - ETA: 53s - loss: 0.2548 - categorical_accuracy: 0.9208
30368/60000 [==============>...............] - ETA: 53s - loss: 0.2546 - categorical_accuracy: 0.9208
30400/60000 [==============>...............] - ETA: 53s - loss: 0.2544 - categorical_accuracy: 0.9208
30432/60000 [==============>...............] - ETA: 53s - loss: 0.2542 - categorical_accuracy: 0.9209
30464/60000 [==============>...............] - ETA: 53s - loss: 0.2545 - categorical_accuracy: 0.9209
30496/60000 [==============>...............] - ETA: 53s - loss: 0.2546 - categorical_accuracy: 0.9208
30528/60000 [==============>...............] - ETA: 53s - loss: 0.2545 - categorical_accuracy: 0.9209
30560/60000 [==============>...............] - ETA: 53s - loss: 0.2543 - categorical_accuracy: 0.9209
30592/60000 [==============>...............] - ETA: 53s - loss: 0.2540 - categorical_accuracy: 0.9210
30624/60000 [==============>...............] - ETA: 53s - loss: 0.2539 - categorical_accuracy: 0.9211
30656/60000 [==============>...............] - ETA: 53s - loss: 0.2538 - categorical_accuracy: 0.9211
30688/60000 [==============>...............] - ETA: 53s - loss: 0.2539 - categorical_accuracy: 0.9211
30720/60000 [==============>...............] - ETA: 53s - loss: 0.2539 - categorical_accuracy: 0.9211
30752/60000 [==============>...............] - ETA: 52s - loss: 0.2538 - categorical_accuracy: 0.9211
30784/60000 [==============>...............] - ETA: 52s - loss: 0.2537 - categorical_accuracy: 0.9212
30816/60000 [==============>...............] - ETA: 52s - loss: 0.2536 - categorical_accuracy: 0.9212
30848/60000 [==============>...............] - ETA: 52s - loss: 0.2536 - categorical_accuracy: 0.9212
30880/60000 [==============>...............] - ETA: 52s - loss: 0.2533 - categorical_accuracy: 0.9213
30912/60000 [==============>...............] - ETA: 52s - loss: 0.2531 - categorical_accuracy: 0.9213
30944/60000 [==============>...............] - ETA: 52s - loss: 0.2531 - categorical_accuracy: 0.9213
30976/60000 [==============>...............] - ETA: 52s - loss: 0.2531 - categorical_accuracy: 0.9214
31008/60000 [==============>...............] - ETA: 52s - loss: 0.2532 - categorical_accuracy: 0.9214
31040/60000 [==============>...............] - ETA: 52s - loss: 0.2532 - categorical_accuracy: 0.9214
31072/60000 [==============>...............] - ETA: 52s - loss: 0.2530 - categorical_accuracy: 0.9215
31104/60000 [==============>...............] - ETA: 52s - loss: 0.2528 - categorical_accuracy: 0.9216
31136/60000 [==============>...............] - ETA: 52s - loss: 0.2527 - categorical_accuracy: 0.9216
31168/60000 [==============>...............] - ETA: 52s - loss: 0.2526 - categorical_accuracy: 0.9216
31200/60000 [==============>...............] - ETA: 52s - loss: 0.2525 - categorical_accuracy: 0.9216
31232/60000 [==============>...............] - ETA: 52s - loss: 0.2526 - categorical_accuracy: 0.9216
31264/60000 [==============>...............] - ETA: 52s - loss: 0.2526 - categorical_accuracy: 0.9216
31296/60000 [==============>...............] - ETA: 52s - loss: 0.2525 - categorical_accuracy: 0.9217
31328/60000 [==============>...............] - ETA: 51s - loss: 0.2523 - categorical_accuracy: 0.9217
31360/60000 [==============>...............] - ETA: 51s - loss: 0.2521 - categorical_accuracy: 0.9218
31392/60000 [==============>...............] - ETA: 51s - loss: 0.2519 - categorical_accuracy: 0.9219
31424/60000 [==============>...............] - ETA: 51s - loss: 0.2519 - categorical_accuracy: 0.9218
31456/60000 [==============>...............] - ETA: 51s - loss: 0.2519 - categorical_accuracy: 0.9218
31488/60000 [==============>...............] - ETA: 51s - loss: 0.2517 - categorical_accuracy: 0.9219
31520/60000 [==============>...............] - ETA: 51s - loss: 0.2515 - categorical_accuracy: 0.9220
31552/60000 [==============>...............] - ETA: 51s - loss: 0.2513 - categorical_accuracy: 0.9220
31584/60000 [==============>...............] - ETA: 51s - loss: 0.2512 - categorical_accuracy: 0.9220
31616/60000 [==============>...............] - ETA: 51s - loss: 0.2509 - categorical_accuracy: 0.9221
31648/60000 [==============>...............] - ETA: 51s - loss: 0.2508 - categorical_accuracy: 0.9222
31680/60000 [==============>...............] - ETA: 51s - loss: 0.2506 - categorical_accuracy: 0.9223
31712/60000 [==============>...............] - ETA: 51s - loss: 0.2504 - categorical_accuracy: 0.9223
31744/60000 [==============>...............] - ETA: 51s - loss: 0.2503 - categorical_accuracy: 0.9223
31776/60000 [==============>...............] - ETA: 51s - loss: 0.2500 - categorical_accuracy: 0.9224
31808/60000 [==============>...............] - ETA: 51s - loss: 0.2500 - categorical_accuracy: 0.9224
31840/60000 [==============>...............] - ETA: 51s - loss: 0.2497 - categorical_accuracy: 0.9225
31872/60000 [==============>...............] - ETA: 50s - loss: 0.2495 - categorical_accuracy: 0.9226
31904/60000 [==============>...............] - ETA: 50s - loss: 0.2494 - categorical_accuracy: 0.9226
31936/60000 [==============>...............] - ETA: 50s - loss: 0.2492 - categorical_accuracy: 0.9227
31968/60000 [==============>...............] - ETA: 50s - loss: 0.2490 - categorical_accuracy: 0.9228
32000/60000 [===============>..............] - ETA: 50s - loss: 0.2490 - categorical_accuracy: 0.9228
32032/60000 [===============>..............] - ETA: 50s - loss: 0.2488 - categorical_accuracy: 0.9229
32064/60000 [===============>..............] - ETA: 50s - loss: 0.2487 - categorical_accuracy: 0.9229
32096/60000 [===============>..............] - ETA: 50s - loss: 0.2485 - categorical_accuracy: 0.9229
32128/60000 [===============>..............] - ETA: 50s - loss: 0.2486 - categorical_accuracy: 0.9229
32160/60000 [===============>..............] - ETA: 50s - loss: 0.2485 - categorical_accuracy: 0.9229
32192/60000 [===============>..............] - ETA: 50s - loss: 0.2483 - categorical_accuracy: 0.9230
32224/60000 [===============>..............] - ETA: 50s - loss: 0.2485 - categorical_accuracy: 0.9230
32256/60000 [===============>..............] - ETA: 50s - loss: 0.2485 - categorical_accuracy: 0.9230
32288/60000 [===============>..............] - ETA: 50s - loss: 0.2483 - categorical_accuracy: 0.9230
32320/60000 [===============>..............] - ETA: 50s - loss: 0.2482 - categorical_accuracy: 0.9231
32352/60000 [===============>..............] - ETA: 50s - loss: 0.2481 - categorical_accuracy: 0.9231
32384/60000 [===============>..............] - ETA: 50s - loss: 0.2478 - categorical_accuracy: 0.9232
32416/60000 [===============>..............] - ETA: 49s - loss: 0.2477 - categorical_accuracy: 0.9232
32448/60000 [===============>..............] - ETA: 49s - loss: 0.2475 - categorical_accuracy: 0.9233
32480/60000 [===============>..............] - ETA: 49s - loss: 0.2474 - categorical_accuracy: 0.9233
32512/60000 [===============>..............] - ETA: 49s - loss: 0.2472 - categorical_accuracy: 0.9234
32544/60000 [===============>..............] - ETA: 49s - loss: 0.2471 - categorical_accuracy: 0.9234
32576/60000 [===============>..............] - ETA: 49s - loss: 0.2470 - categorical_accuracy: 0.9234
32608/60000 [===============>..............] - ETA: 49s - loss: 0.2468 - categorical_accuracy: 0.9235
32640/60000 [===============>..............] - ETA: 49s - loss: 0.2466 - categorical_accuracy: 0.9236
32672/60000 [===============>..............] - ETA: 49s - loss: 0.2463 - categorical_accuracy: 0.9236
32704/60000 [===============>..............] - ETA: 49s - loss: 0.2463 - categorical_accuracy: 0.9236
32736/60000 [===============>..............] - ETA: 49s - loss: 0.2461 - categorical_accuracy: 0.9237
32768/60000 [===============>..............] - ETA: 49s - loss: 0.2459 - categorical_accuracy: 0.9238
32800/60000 [===============>..............] - ETA: 49s - loss: 0.2461 - categorical_accuracy: 0.9238
32832/60000 [===============>..............] - ETA: 49s - loss: 0.2459 - categorical_accuracy: 0.9239
32864/60000 [===============>..............] - ETA: 49s - loss: 0.2458 - categorical_accuracy: 0.9239
32896/60000 [===============>..............] - ETA: 49s - loss: 0.2456 - categorical_accuracy: 0.9239
32928/60000 [===============>..............] - ETA: 49s - loss: 0.2455 - categorical_accuracy: 0.9240
32960/60000 [===============>..............] - ETA: 48s - loss: 0.2454 - categorical_accuracy: 0.9240
32992/60000 [===============>..............] - ETA: 48s - loss: 0.2454 - categorical_accuracy: 0.9240
33024/60000 [===============>..............] - ETA: 48s - loss: 0.2452 - categorical_accuracy: 0.9241
33056/60000 [===============>..............] - ETA: 48s - loss: 0.2451 - categorical_accuracy: 0.9241
33088/60000 [===============>..............] - ETA: 48s - loss: 0.2449 - categorical_accuracy: 0.9241
33120/60000 [===============>..............] - ETA: 48s - loss: 0.2448 - categorical_accuracy: 0.9241
33152/60000 [===============>..............] - ETA: 48s - loss: 0.2446 - categorical_accuracy: 0.9242
33184/60000 [===============>..............] - ETA: 48s - loss: 0.2444 - categorical_accuracy: 0.9243
33216/60000 [===============>..............] - ETA: 48s - loss: 0.2441 - categorical_accuracy: 0.9243
33248/60000 [===============>..............] - ETA: 48s - loss: 0.2439 - categorical_accuracy: 0.9244
33280/60000 [===============>..............] - ETA: 48s - loss: 0.2437 - categorical_accuracy: 0.9245
33312/60000 [===============>..............] - ETA: 48s - loss: 0.2436 - categorical_accuracy: 0.9245
33344/60000 [===============>..............] - ETA: 48s - loss: 0.2434 - categorical_accuracy: 0.9246
33376/60000 [===============>..............] - ETA: 48s - loss: 0.2434 - categorical_accuracy: 0.9245
33408/60000 [===============>..............] - ETA: 48s - loss: 0.2433 - categorical_accuracy: 0.9246
33440/60000 [===============>..............] - ETA: 48s - loss: 0.2431 - categorical_accuracy: 0.9246
33472/60000 [===============>..............] - ETA: 48s - loss: 0.2430 - categorical_accuracy: 0.9247
33504/60000 [===============>..............] - ETA: 48s - loss: 0.2430 - categorical_accuracy: 0.9247
33536/60000 [===============>..............] - ETA: 47s - loss: 0.2429 - categorical_accuracy: 0.9248
33568/60000 [===============>..............] - ETA: 47s - loss: 0.2428 - categorical_accuracy: 0.9247
33600/60000 [===============>..............] - ETA: 47s - loss: 0.2426 - categorical_accuracy: 0.9248
33632/60000 [===============>..............] - ETA: 47s - loss: 0.2425 - categorical_accuracy: 0.9248
33664/60000 [===============>..............] - ETA: 47s - loss: 0.2423 - categorical_accuracy: 0.9249
33696/60000 [===============>..............] - ETA: 47s - loss: 0.2422 - categorical_accuracy: 0.9250
33728/60000 [===============>..............] - ETA: 47s - loss: 0.2420 - categorical_accuracy: 0.9250
33760/60000 [===============>..............] - ETA: 47s - loss: 0.2419 - categorical_accuracy: 0.9251
33792/60000 [===============>..............] - ETA: 47s - loss: 0.2417 - categorical_accuracy: 0.9251
33824/60000 [===============>..............] - ETA: 47s - loss: 0.2416 - categorical_accuracy: 0.9252
33856/60000 [===============>..............] - ETA: 47s - loss: 0.2414 - categorical_accuracy: 0.9252
33888/60000 [===============>..............] - ETA: 47s - loss: 0.2413 - categorical_accuracy: 0.9253
33920/60000 [===============>..............] - ETA: 47s - loss: 0.2413 - categorical_accuracy: 0.9253
33952/60000 [===============>..............] - ETA: 47s - loss: 0.2413 - categorical_accuracy: 0.9252
33984/60000 [===============>..............] - ETA: 47s - loss: 0.2412 - categorical_accuracy: 0.9253
34016/60000 [================>.............] - ETA: 47s - loss: 0.2411 - categorical_accuracy: 0.9253
34048/60000 [================>.............] - ETA: 47s - loss: 0.2411 - categorical_accuracy: 0.9253
34080/60000 [================>.............] - ETA: 46s - loss: 0.2410 - categorical_accuracy: 0.9253
34112/60000 [================>.............] - ETA: 46s - loss: 0.2409 - categorical_accuracy: 0.9253
34144/60000 [================>.............] - ETA: 46s - loss: 0.2407 - categorical_accuracy: 0.9254
34176/60000 [================>.............] - ETA: 46s - loss: 0.2405 - categorical_accuracy: 0.9254
34208/60000 [================>.............] - ETA: 46s - loss: 0.2405 - categorical_accuracy: 0.9255
34240/60000 [================>.............] - ETA: 46s - loss: 0.2403 - categorical_accuracy: 0.9255
34272/60000 [================>.............] - ETA: 46s - loss: 0.2401 - categorical_accuracy: 0.9255
34304/60000 [================>.............] - ETA: 46s - loss: 0.2400 - categorical_accuracy: 0.9256
34336/60000 [================>.............] - ETA: 46s - loss: 0.2398 - categorical_accuracy: 0.9256
34368/60000 [================>.............] - ETA: 46s - loss: 0.2397 - categorical_accuracy: 0.9257
34400/60000 [================>.............] - ETA: 46s - loss: 0.2395 - categorical_accuracy: 0.9258
34432/60000 [================>.............] - ETA: 46s - loss: 0.2393 - categorical_accuracy: 0.9258
34464/60000 [================>.............] - ETA: 46s - loss: 0.2391 - categorical_accuracy: 0.9258
34496/60000 [================>.............] - ETA: 46s - loss: 0.2389 - categorical_accuracy: 0.9259
34528/60000 [================>.............] - ETA: 46s - loss: 0.2389 - categorical_accuracy: 0.9259
34560/60000 [================>.............] - ETA: 46s - loss: 0.2387 - categorical_accuracy: 0.9260
34592/60000 [================>.............] - ETA: 46s - loss: 0.2389 - categorical_accuracy: 0.9259
34624/60000 [================>.............] - ETA: 45s - loss: 0.2389 - categorical_accuracy: 0.9259
34656/60000 [================>.............] - ETA: 45s - loss: 0.2387 - categorical_accuracy: 0.9260
34688/60000 [================>.............] - ETA: 45s - loss: 0.2388 - categorical_accuracy: 0.9260
34720/60000 [================>.............] - ETA: 45s - loss: 0.2388 - categorical_accuracy: 0.9260
34752/60000 [================>.............] - ETA: 45s - loss: 0.2386 - categorical_accuracy: 0.9260
34784/60000 [================>.............] - ETA: 45s - loss: 0.2385 - categorical_accuracy: 0.9261
34816/60000 [================>.............] - ETA: 45s - loss: 0.2385 - categorical_accuracy: 0.9261
34848/60000 [================>.............] - ETA: 45s - loss: 0.2383 - categorical_accuracy: 0.9261
34880/60000 [================>.............] - ETA: 45s - loss: 0.2381 - categorical_accuracy: 0.9262
34912/60000 [================>.............] - ETA: 45s - loss: 0.2379 - categorical_accuracy: 0.9263
34944/60000 [================>.............] - ETA: 45s - loss: 0.2378 - categorical_accuracy: 0.9263
34976/60000 [================>.............] - ETA: 45s - loss: 0.2377 - categorical_accuracy: 0.9263
35008/60000 [================>.............] - ETA: 45s - loss: 0.2377 - categorical_accuracy: 0.9263
35040/60000 [================>.............] - ETA: 45s - loss: 0.2377 - categorical_accuracy: 0.9263
35072/60000 [================>.............] - ETA: 45s - loss: 0.2375 - categorical_accuracy: 0.9264
35104/60000 [================>.............] - ETA: 45s - loss: 0.2373 - categorical_accuracy: 0.9264
35136/60000 [================>.............] - ETA: 45s - loss: 0.2372 - categorical_accuracy: 0.9265
35168/60000 [================>.............] - ETA: 44s - loss: 0.2370 - categorical_accuracy: 0.9266
35200/60000 [================>.............] - ETA: 44s - loss: 0.2370 - categorical_accuracy: 0.9266
35232/60000 [================>.............] - ETA: 44s - loss: 0.2368 - categorical_accuracy: 0.9266
35264/60000 [================>.............] - ETA: 44s - loss: 0.2367 - categorical_accuracy: 0.9267
35296/60000 [================>.............] - ETA: 44s - loss: 0.2365 - categorical_accuracy: 0.9267
35328/60000 [================>.............] - ETA: 44s - loss: 0.2364 - categorical_accuracy: 0.9267
35360/60000 [================>.............] - ETA: 44s - loss: 0.2362 - categorical_accuracy: 0.9268
35392/60000 [================>.............] - ETA: 44s - loss: 0.2360 - categorical_accuracy: 0.9269
35424/60000 [================>.............] - ETA: 44s - loss: 0.2358 - categorical_accuracy: 0.9269
35456/60000 [================>.............] - ETA: 44s - loss: 0.2356 - categorical_accuracy: 0.9270
35488/60000 [================>.............] - ETA: 44s - loss: 0.2354 - categorical_accuracy: 0.9271
35520/60000 [================>.............] - ETA: 44s - loss: 0.2353 - categorical_accuracy: 0.9271
35552/60000 [================>.............] - ETA: 44s - loss: 0.2354 - categorical_accuracy: 0.9271
35584/60000 [================>.............] - ETA: 44s - loss: 0.2355 - categorical_accuracy: 0.9271
35616/60000 [================>.............] - ETA: 44s - loss: 0.2353 - categorical_accuracy: 0.9271
35648/60000 [================>.............] - ETA: 44s - loss: 0.2352 - categorical_accuracy: 0.9271
35680/60000 [================>.............] - ETA: 44s - loss: 0.2351 - categorical_accuracy: 0.9272
35712/60000 [================>.............] - ETA: 43s - loss: 0.2350 - categorical_accuracy: 0.9272
35744/60000 [================>.............] - ETA: 43s - loss: 0.2348 - categorical_accuracy: 0.9273
35776/60000 [================>.............] - ETA: 43s - loss: 0.2346 - categorical_accuracy: 0.9273
35808/60000 [================>.............] - ETA: 43s - loss: 0.2345 - categorical_accuracy: 0.9273
35840/60000 [================>.............] - ETA: 43s - loss: 0.2343 - categorical_accuracy: 0.9274
35872/60000 [================>.............] - ETA: 43s - loss: 0.2341 - categorical_accuracy: 0.9274
35904/60000 [================>.............] - ETA: 43s - loss: 0.2341 - categorical_accuracy: 0.9274
35936/60000 [================>.............] - ETA: 43s - loss: 0.2340 - categorical_accuracy: 0.9274
35968/60000 [================>.............] - ETA: 43s - loss: 0.2338 - categorical_accuracy: 0.9275
36000/60000 [=================>............] - ETA: 43s - loss: 0.2336 - categorical_accuracy: 0.9275
36032/60000 [=================>............] - ETA: 43s - loss: 0.2334 - categorical_accuracy: 0.9276
36064/60000 [=================>............] - ETA: 43s - loss: 0.2333 - categorical_accuracy: 0.9276
36096/60000 [=================>............] - ETA: 43s - loss: 0.2331 - categorical_accuracy: 0.9277
36128/60000 [=================>............] - ETA: 43s - loss: 0.2331 - categorical_accuracy: 0.9277
36160/60000 [=================>............] - ETA: 43s - loss: 0.2330 - categorical_accuracy: 0.9277
36192/60000 [=================>............] - ETA: 43s - loss: 0.2329 - categorical_accuracy: 0.9278
36224/60000 [=================>............] - ETA: 43s - loss: 0.2328 - categorical_accuracy: 0.9278
36256/60000 [=================>............] - ETA: 43s - loss: 0.2326 - categorical_accuracy: 0.9278
36288/60000 [=================>............] - ETA: 42s - loss: 0.2326 - categorical_accuracy: 0.9279
36320/60000 [=================>............] - ETA: 42s - loss: 0.2324 - categorical_accuracy: 0.9279
36352/60000 [=================>............] - ETA: 42s - loss: 0.2324 - categorical_accuracy: 0.9279
36384/60000 [=================>............] - ETA: 42s - loss: 0.2322 - categorical_accuracy: 0.9280
36416/60000 [=================>............] - ETA: 42s - loss: 0.2321 - categorical_accuracy: 0.9280
36448/60000 [=================>............] - ETA: 42s - loss: 0.2319 - categorical_accuracy: 0.9281
36480/60000 [=================>............] - ETA: 42s - loss: 0.2317 - categorical_accuracy: 0.9281
36512/60000 [=================>............] - ETA: 42s - loss: 0.2316 - categorical_accuracy: 0.9282
36544/60000 [=================>............] - ETA: 42s - loss: 0.2314 - categorical_accuracy: 0.9282
36576/60000 [=================>............] - ETA: 42s - loss: 0.2312 - categorical_accuracy: 0.9283
36608/60000 [=================>............] - ETA: 42s - loss: 0.2312 - categorical_accuracy: 0.9283
36640/60000 [=================>............] - ETA: 42s - loss: 0.2312 - categorical_accuracy: 0.9283
36672/60000 [=================>............] - ETA: 42s - loss: 0.2310 - categorical_accuracy: 0.9283
36704/60000 [=================>............] - ETA: 42s - loss: 0.2308 - categorical_accuracy: 0.9284
36736/60000 [=================>............] - ETA: 42s - loss: 0.2306 - categorical_accuracy: 0.9285
36768/60000 [=================>............] - ETA: 42s - loss: 0.2304 - categorical_accuracy: 0.9285
36800/60000 [=================>............] - ETA: 42s - loss: 0.2302 - categorical_accuracy: 0.9286
36832/60000 [=================>............] - ETA: 41s - loss: 0.2301 - categorical_accuracy: 0.9286
36864/60000 [=================>............] - ETA: 41s - loss: 0.2299 - categorical_accuracy: 0.9287
36896/60000 [=================>............] - ETA: 41s - loss: 0.2299 - categorical_accuracy: 0.9287
36928/60000 [=================>............] - ETA: 41s - loss: 0.2299 - categorical_accuracy: 0.9287
36960/60000 [=================>............] - ETA: 41s - loss: 0.2298 - categorical_accuracy: 0.9287
36992/60000 [=================>............] - ETA: 41s - loss: 0.2297 - categorical_accuracy: 0.9288
37024/60000 [=================>............] - ETA: 41s - loss: 0.2297 - categorical_accuracy: 0.9288
37056/60000 [=================>............] - ETA: 41s - loss: 0.2295 - categorical_accuracy: 0.9288
37088/60000 [=================>............] - ETA: 41s - loss: 0.2294 - categorical_accuracy: 0.9288
37120/60000 [=================>............] - ETA: 41s - loss: 0.2294 - categorical_accuracy: 0.9289
37152/60000 [=================>............] - ETA: 41s - loss: 0.2292 - categorical_accuracy: 0.9289
37184/60000 [=================>............] - ETA: 41s - loss: 0.2291 - categorical_accuracy: 0.9289
37216/60000 [=================>............] - ETA: 41s - loss: 0.2291 - categorical_accuracy: 0.9290
37248/60000 [=================>............] - ETA: 41s - loss: 0.2291 - categorical_accuracy: 0.9290
37280/60000 [=================>............] - ETA: 41s - loss: 0.2290 - categorical_accuracy: 0.9290
37312/60000 [=================>............] - ETA: 41s - loss: 0.2289 - categorical_accuracy: 0.9290
37344/60000 [=================>............] - ETA: 41s - loss: 0.2289 - categorical_accuracy: 0.9290
37376/60000 [=================>............] - ETA: 40s - loss: 0.2287 - categorical_accuracy: 0.9291
37408/60000 [=================>............] - ETA: 40s - loss: 0.2286 - categorical_accuracy: 0.9291
37440/60000 [=================>............] - ETA: 40s - loss: 0.2285 - categorical_accuracy: 0.9291
37472/60000 [=================>............] - ETA: 40s - loss: 0.2283 - categorical_accuracy: 0.9292
37504/60000 [=================>............] - ETA: 40s - loss: 0.2281 - categorical_accuracy: 0.9292
37536/60000 [=================>............] - ETA: 40s - loss: 0.2281 - categorical_accuracy: 0.9292
37568/60000 [=================>............] - ETA: 40s - loss: 0.2280 - categorical_accuracy: 0.9292
37600/60000 [=================>............] - ETA: 40s - loss: 0.2278 - categorical_accuracy: 0.9293
37632/60000 [=================>............] - ETA: 40s - loss: 0.2276 - categorical_accuracy: 0.9294
37664/60000 [=================>............] - ETA: 40s - loss: 0.2275 - categorical_accuracy: 0.9294
37696/60000 [=================>............] - ETA: 40s - loss: 0.2273 - categorical_accuracy: 0.9295
37728/60000 [=================>............] - ETA: 40s - loss: 0.2271 - categorical_accuracy: 0.9295
37760/60000 [=================>............] - ETA: 40s - loss: 0.2271 - categorical_accuracy: 0.9295
37792/60000 [=================>............] - ETA: 40s - loss: 0.2270 - categorical_accuracy: 0.9296
37824/60000 [=================>............] - ETA: 40s - loss: 0.2269 - categorical_accuracy: 0.9296
37856/60000 [=================>............] - ETA: 40s - loss: 0.2268 - categorical_accuracy: 0.9297
37888/60000 [=================>............] - ETA: 40s - loss: 0.2269 - categorical_accuracy: 0.9296
37920/60000 [=================>............] - ETA: 39s - loss: 0.2267 - categorical_accuracy: 0.9297
37952/60000 [=================>............] - ETA: 39s - loss: 0.2265 - categorical_accuracy: 0.9297
37984/60000 [=================>............] - ETA: 39s - loss: 0.2267 - categorical_accuracy: 0.9297
38016/60000 [==================>...........] - ETA: 39s - loss: 0.2265 - categorical_accuracy: 0.9298
38048/60000 [==================>...........] - ETA: 39s - loss: 0.2263 - categorical_accuracy: 0.9298
38080/60000 [==================>...........] - ETA: 39s - loss: 0.2262 - categorical_accuracy: 0.9299
38112/60000 [==================>...........] - ETA: 39s - loss: 0.2260 - categorical_accuracy: 0.9299
38144/60000 [==================>...........] - ETA: 39s - loss: 0.2261 - categorical_accuracy: 0.9299
38176/60000 [==================>...........] - ETA: 39s - loss: 0.2260 - categorical_accuracy: 0.9299
38208/60000 [==================>...........] - ETA: 39s - loss: 0.2260 - categorical_accuracy: 0.9299
38240/60000 [==================>...........] - ETA: 39s - loss: 0.2259 - categorical_accuracy: 0.9299
38272/60000 [==================>...........] - ETA: 39s - loss: 0.2260 - categorical_accuracy: 0.9299
38304/60000 [==================>...........] - ETA: 39s - loss: 0.2259 - categorical_accuracy: 0.9299
38336/60000 [==================>...........] - ETA: 39s - loss: 0.2258 - categorical_accuracy: 0.9300
38368/60000 [==================>...........] - ETA: 39s - loss: 0.2257 - categorical_accuracy: 0.9300
38400/60000 [==================>...........] - ETA: 39s - loss: 0.2256 - categorical_accuracy: 0.9300
38432/60000 [==================>...........] - ETA: 39s - loss: 0.2255 - categorical_accuracy: 0.9300
38464/60000 [==================>...........] - ETA: 38s - loss: 0.2254 - categorical_accuracy: 0.9300
38496/60000 [==================>...........] - ETA: 38s - loss: 0.2253 - categorical_accuracy: 0.9300
38528/60000 [==================>...........] - ETA: 38s - loss: 0.2252 - categorical_accuracy: 0.9301
38560/60000 [==================>...........] - ETA: 38s - loss: 0.2251 - categorical_accuracy: 0.9301
38592/60000 [==================>...........] - ETA: 38s - loss: 0.2249 - categorical_accuracy: 0.9302
38624/60000 [==================>...........] - ETA: 38s - loss: 0.2249 - categorical_accuracy: 0.9302
38656/60000 [==================>...........] - ETA: 38s - loss: 0.2247 - categorical_accuracy: 0.9302
38688/60000 [==================>...........] - ETA: 38s - loss: 0.2246 - categorical_accuracy: 0.9303
38720/60000 [==================>...........] - ETA: 38s - loss: 0.2245 - categorical_accuracy: 0.9303
38752/60000 [==================>...........] - ETA: 38s - loss: 0.2243 - categorical_accuracy: 0.9304
38784/60000 [==================>...........] - ETA: 38s - loss: 0.2244 - categorical_accuracy: 0.9303
38816/60000 [==================>...........] - ETA: 38s - loss: 0.2243 - categorical_accuracy: 0.9303
38848/60000 [==================>...........] - ETA: 38s - loss: 0.2243 - categorical_accuracy: 0.9304
38880/60000 [==================>...........] - ETA: 38s - loss: 0.2241 - categorical_accuracy: 0.9304
38912/60000 [==================>...........] - ETA: 38s - loss: 0.2241 - categorical_accuracy: 0.9304
38944/60000 [==================>...........] - ETA: 38s - loss: 0.2240 - categorical_accuracy: 0.9304
38976/60000 [==================>...........] - ETA: 38s - loss: 0.2240 - categorical_accuracy: 0.9305
39008/60000 [==================>...........] - ETA: 37s - loss: 0.2239 - categorical_accuracy: 0.9305
39040/60000 [==================>...........] - ETA: 37s - loss: 0.2238 - categorical_accuracy: 0.9305
39072/60000 [==================>...........] - ETA: 37s - loss: 0.2237 - categorical_accuracy: 0.9305
39104/60000 [==================>...........] - ETA: 37s - loss: 0.2236 - categorical_accuracy: 0.9306
39136/60000 [==================>...........] - ETA: 37s - loss: 0.2237 - categorical_accuracy: 0.9305
39168/60000 [==================>...........] - ETA: 37s - loss: 0.2235 - categorical_accuracy: 0.9306
39200/60000 [==================>...........] - ETA: 37s - loss: 0.2234 - categorical_accuracy: 0.9306
39232/60000 [==================>...........] - ETA: 37s - loss: 0.2233 - categorical_accuracy: 0.9306
39264/60000 [==================>...........] - ETA: 37s - loss: 0.2232 - categorical_accuracy: 0.9307
39296/60000 [==================>...........] - ETA: 37s - loss: 0.2231 - categorical_accuracy: 0.9307
39328/60000 [==================>...........] - ETA: 37s - loss: 0.2229 - categorical_accuracy: 0.9307
39360/60000 [==================>...........] - ETA: 37s - loss: 0.2230 - categorical_accuracy: 0.9307
39392/60000 [==================>...........] - ETA: 37s - loss: 0.2231 - categorical_accuracy: 0.9307
39424/60000 [==================>...........] - ETA: 37s - loss: 0.2230 - categorical_accuracy: 0.9307
39456/60000 [==================>...........] - ETA: 37s - loss: 0.2228 - categorical_accuracy: 0.9307
39488/60000 [==================>...........] - ETA: 37s - loss: 0.2228 - categorical_accuracy: 0.9308
39520/60000 [==================>...........] - ETA: 37s - loss: 0.2228 - categorical_accuracy: 0.9308
39552/60000 [==================>...........] - ETA: 37s - loss: 0.2227 - categorical_accuracy: 0.9308
39584/60000 [==================>...........] - ETA: 36s - loss: 0.2226 - categorical_accuracy: 0.9309
39616/60000 [==================>...........] - ETA: 36s - loss: 0.2226 - categorical_accuracy: 0.9309
39648/60000 [==================>...........] - ETA: 36s - loss: 0.2225 - categorical_accuracy: 0.9309
39680/60000 [==================>...........] - ETA: 36s - loss: 0.2225 - categorical_accuracy: 0.9309
39712/60000 [==================>...........] - ETA: 36s - loss: 0.2223 - categorical_accuracy: 0.9310
39744/60000 [==================>...........] - ETA: 36s - loss: 0.2222 - categorical_accuracy: 0.9310
39776/60000 [==================>...........] - ETA: 36s - loss: 0.2220 - categorical_accuracy: 0.9311
39808/60000 [==================>...........] - ETA: 36s - loss: 0.2219 - categorical_accuracy: 0.9311
39840/60000 [==================>...........] - ETA: 36s - loss: 0.2218 - categorical_accuracy: 0.9311
39872/60000 [==================>...........] - ETA: 36s - loss: 0.2218 - categorical_accuracy: 0.9312
39904/60000 [==================>...........] - ETA: 36s - loss: 0.2218 - categorical_accuracy: 0.9312
39936/60000 [==================>...........] - ETA: 36s - loss: 0.2218 - categorical_accuracy: 0.9312
39968/60000 [==================>...........] - ETA: 36s - loss: 0.2217 - categorical_accuracy: 0.9312
40000/60000 [===================>..........] - ETA: 36s - loss: 0.2215 - categorical_accuracy: 0.9313
40032/60000 [===================>..........] - ETA: 36s - loss: 0.2213 - categorical_accuracy: 0.9313
40064/60000 [===================>..........] - ETA: 36s - loss: 0.2214 - categorical_accuracy: 0.9313
40096/60000 [===================>..........] - ETA: 36s - loss: 0.2212 - categorical_accuracy: 0.9314
40128/60000 [===================>..........] - ETA: 35s - loss: 0.2213 - categorical_accuracy: 0.9314
40160/60000 [===================>..........] - ETA: 35s - loss: 0.2212 - categorical_accuracy: 0.9314
40192/60000 [===================>..........] - ETA: 35s - loss: 0.2213 - categorical_accuracy: 0.9314
40224/60000 [===================>..........] - ETA: 35s - loss: 0.2212 - categorical_accuracy: 0.9315
40256/60000 [===================>..........] - ETA: 35s - loss: 0.2211 - categorical_accuracy: 0.9315
40288/60000 [===================>..........] - ETA: 35s - loss: 0.2210 - categorical_accuracy: 0.9315
40320/60000 [===================>..........] - ETA: 35s - loss: 0.2210 - categorical_accuracy: 0.9314
40352/60000 [===================>..........] - ETA: 35s - loss: 0.2209 - categorical_accuracy: 0.9315
40384/60000 [===================>..........] - ETA: 35s - loss: 0.2208 - categorical_accuracy: 0.9315
40416/60000 [===================>..........] - ETA: 35s - loss: 0.2209 - categorical_accuracy: 0.9315
40448/60000 [===================>..........] - ETA: 35s - loss: 0.2208 - categorical_accuracy: 0.9315
40480/60000 [===================>..........] - ETA: 35s - loss: 0.2206 - categorical_accuracy: 0.9316
40512/60000 [===================>..........] - ETA: 35s - loss: 0.2205 - categorical_accuracy: 0.9316
40544/60000 [===================>..........] - ETA: 35s - loss: 0.2205 - categorical_accuracy: 0.9316
40576/60000 [===================>..........] - ETA: 35s - loss: 0.2204 - categorical_accuracy: 0.9316
40608/60000 [===================>..........] - ETA: 35s - loss: 0.2203 - categorical_accuracy: 0.9317
40640/60000 [===================>..........] - ETA: 35s - loss: 0.2202 - categorical_accuracy: 0.9317
40672/60000 [===================>..........] - ETA: 34s - loss: 0.2201 - categorical_accuracy: 0.9316
40704/60000 [===================>..........] - ETA: 34s - loss: 0.2200 - categorical_accuracy: 0.9317
40736/60000 [===================>..........] - ETA: 34s - loss: 0.2199 - categorical_accuracy: 0.9317
40768/60000 [===================>..........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9317
40800/60000 [===================>..........] - ETA: 34s - loss: 0.2197 - categorical_accuracy: 0.9317
40832/60000 [===================>..........] - ETA: 34s - loss: 0.2196 - categorical_accuracy: 0.9317
40864/60000 [===================>..........] - ETA: 34s - loss: 0.2197 - categorical_accuracy: 0.9317
40896/60000 [===================>..........] - ETA: 34s - loss: 0.2199 - categorical_accuracy: 0.9317
40928/60000 [===================>..........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9318
40960/60000 [===================>..........] - ETA: 34s - loss: 0.2197 - categorical_accuracy: 0.9318
40992/60000 [===================>..........] - ETA: 34s - loss: 0.2196 - categorical_accuracy: 0.9318
41024/60000 [===================>..........] - ETA: 34s - loss: 0.2195 - categorical_accuracy: 0.9318
41056/60000 [===================>..........] - ETA: 34s - loss: 0.2194 - categorical_accuracy: 0.9319
41088/60000 [===================>..........] - ETA: 34s - loss: 0.2192 - categorical_accuracy: 0.9319
41120/60000 [===================>..........] - ETA: 34s - loss: 0.2191 - categorical_accuracy: 0.9320
41152/60000 [===================>..........] - ETA: 34s - loss: 0.2189 - categorical_accuracy: 0.9320
41184/60000 [===================>..........] - ETA: 34s - loss: 0.2189 - categorical_accuracy: 0.9321
41216/60000 [===================>..........] - ETA: 33s - loss: 0.2187 - categorical_accuracy: 0.9321
41248/60000 [===================>..........] - ETA: 33s - loss: 0.2186 - categorical_accuracy: 0.9321
41280/60000 [===================>..........] - ETA: 33s - loss: 0.2186 - categorical_accuracy: 0.9321
41312/60000 [===================>..........] - ETA: 33s - loss: 0.2186 - categorical_accuracy: 0.9322
41344/60000 [===================>..........] - ETA: 33s - loss: 0.2185 - categorical_accuracy: 0.9322
41376/60000 [===================>..........] - ETA: 33s - loss: 0.2184 - categorical_accuracy: 0.9322
41408/60000 [===================>..........] - ETA: 33s - loss: 0.2183 - categorical_accuracy: 0.9322
41440/60000 [===================>..........] - ETA: 33s - loss: 0.2181 - categorical_accuracy: 0.9323
41472/60000 [===================>..........] - ETA: 33s - loss: 0.2180 - categorical_accuracy: 0.9323
41504/60000 [===================>..........] - ETA: 33s - loss: 0.2178 - categorical_accuracy: 0.9324
41536/60000 [===================>..........] - ETA: 33s - loss: 0.2180 - categorical_accuracy: 0.9323
41568/60000 [===================>..........] - ETA: 33s - loss: 0.2179 - categorical_accuracy: 0.9324
41600/60000 [===================>..........] - ETA: 33s - loss: 0.2181 - categorical_accuracy: 0.9323
41632/60000 [===================>..........] - ETA: 33s - loss: 0.2180 - categorical_accuracy: 0.9323
41664/60000 [===================>..........] - ETA: 33s - loss: 0.2179 - categorical_accuracy: 0.9324
41696/60000 [===================>..........] - ETA: 33s - loss: 0.2178 - categorical_accuracy: 0.9324
41728/60000 [===================>..........] - ETA: 33s - loss: 0.2177 - categorical_accuracy: 0.9324
41760/60000 [===================>..........] - ETA: 33s - loss: 0.2176 - categorical_accuracy: 0.9325
41792/60000 [===================>..........] - ETA: 32s - loss: 0.2175 - categorical_accuracy: 0.9325
41824/60000 [===================>..........] - ETA: 32s - loss: 0.2176 - categorical_accuracy: 0.9325
41856/60000 [===================>..........] - ETA: 32s - loss: 0.2176 - categorical_accuracy: 0.9325
41888/60000 [===================>..........] - ETA: 32s - loss: 0.2174 - categorical_accuracy: 0.9325
41920/60000 [===================>..........] - ETA: 32s - loss: 0.2173 - categorical_accuracy: 0.9326
41952/60000 [===================>..........] - ETA: 32s - loss: 0.2173 - categorical_accuracy: 0.9326
41984/60000 [===================>..........] - ETA: 32s - loss: 0.2172 - categorical_accuracy: 0.9326
42016/60000 [====================>.........] - ETA: 32s - loss: 0.2171 - categorical_accuracy: 0.9326
42048/60000 [====================>.........] - ETA: 32s - loss: 0.2170 - categorical_accuracy: 0.9327
42080/60000 [====================>.........] - ETA: 32s - loss: 0.2169 - categorical_accuracy: 0.9327
42112/60000 [====================>.........] - ETA: 32s - loss: 0.2167 - categorical_accuracy: 0.9328
42144/60000 [====================>.........] - ETA: 32s - loss: 0.2165 - categorical_accuracy: 0.9328
42176/60000 [====================>.........] - ETA: 32s - loss: 0.2166 - categorical_accuracy: 0.9328
42208/60000 [====================>.........] - ETA: 32s - loss: 0.2165 - categorical_accuracy: 0.9329
42240/60000 [====================>.........] - ETA: 32s - loss: 0.2164 - categorical_accuracy: 0.9329
42272/60000 [====================>.........] - ETA: 32s - loss: 0.2162 - categorical_accuracy: 0.9329
42304/60000 [====================>.........] - ETA: 32s - loss: 0.2161 - categorical_accuracy: 0.9330
42336/60000 [====================>.........] - ETA: 31s - loss: 0.2161 - categorical_accuracy: 0.9330
42368/60000 [====================>.........] - ETA: 31s - loss: 0.2159 - categorical_accuracy: 0.9330
42400/60000 [====================>.........] - ETA: 31s - loss: 0.2158 - categorical_accuracy: 0.9331
42432/60000 [====================>.........] - ETA: 31s - loss: 0.2157 - categorical_accuracy: 0.9331
42464/60000 [====================>.........] - ETA: 31s - loss: 0.2157 - categorical_accuracy: 0.9331
42496/60000 [====================>.........] - ETA: 31s - loss: 0.2156 - categorical_accuracy: 0.9332
42528/60000 [====================>.........] - ETA: 31s - loss: 0.2154 - categorical_accuracy: 0.9332
42560/60000 [====================>.........] - ETA: 31s - loss: 0.2153 - categorical_accuracy: 0.9332
42592/60000 [====================>.........] - ETA: 31s - loss: 0.2151 - categorical_accuracy: 0.9333
42624/60000 [====================>.........] - ETA: 31s - loss: 0.2150 - categorical_accuracy: 0.9333
42656/60000 [====================>.........] - ETA: 31s - loss: 0.2149 - categorical_accuracy: 0.9334
42688/60000 [====================>.........] - ETA: 31s - loss: 0.2148 - categorical_accuracy: 0.9334
42720/60000 [====================>.........] - ETA: 31s - loss: 0.2146 - categorical_accuracy: 0.9335
42752/60000 [====================>.........] - ETA: 31s - loss: 0.2146 - categorical_accuracy: 0.9335
42784/60000 [====================>.........] - ETA: 31s - loss: 0.2145 - categorical_accuracy: 0.9335
42816/60000 [====================>.........] - ETA: 31s - loss: 0.2144 - categorical_accuracy: 0.9335
42848/60000 [====================>.........] - ETA: 31s - loss: 0.2143 - categorical_accuracy: 0.9335
42880/60000 [====================>.........] - ETA: 30s - loss: 0.2143 - categorical_accuracy: 0.9335
42912/60000 [====================>.........] - ETA: 30s - loss: 0.2142 - categorical_accuracy: 0.9336
42944/60000 [====================>.........] - ETA: 30s - loss: 0.2140 - categorical_accuracy: 0.9336
42976/60000 [====================>.........] - ETA: 30s - loss: 0.2140 - categorical_accuracy: 0.9336
43008/60000 [====================>.........] - ETA: 30s - loss: 0.2139 - categorical_accuracy: 0.9337
43040/60000 [====================>.........] - ETA: 30s - loss: 0.2138 - categorical_accuracy: 0.9337
43072/60000 [====================>.........] - ETA: 30s - loss: 0.2138 - categorical_accuracy: 0.9337
43104/60000 [====================>.........] - ETA: 30s - loss: 0.2137 - categorical_accuracy: 0.9337
43136/60000 [====================>.........] - ETA: 30s - loss: 0.2137 - categorical_accuracy: 0.9337
43168/60000 [====================>.........] - ETA: 30s - loss: 0.2136 - categorical_accuracy: 0.9338
43200/60000 [====================>.........] - ETA: 30s - loss: 0.2134 - categorical_accuracy: 0.9338
43232/60000 [====================>.........] - ETA: 30s - loss: 0.2133 - categorical_accuracy: 0.9339
43264/60000 [====================>.........] - ETA: 30s - loss: 0.2134 - categorical_accuracy: 0.9339
43296/60000 [====================>.........] - ETA: 30s - loss: 0.2133 - categorical_accuracy: 0.9339
43328/60000 [====================>.........] - ETA: 30s - loss: 0.2131 - categorical_accuracy: 0.9339
43360/60000 [====================>.........] - ETA: 30s - loss: 0.2130 - categorical_accuracy: 0.9340
43392/60000 [====================>.........] - ETA: 30s - loss: 0.2129 - categorical_accuracy: 0.9340
43424/60000 [====================>.........] - ETA: 29s - loss: 0.2128 - categorical_accuracy: 0.9340
43456/60000 [====================>.........] - ETA: 29s - loss: 0.2126 - categorical_accuracy: 0.9341
43488/60000 [====================>.........] - ETA: 29s - loss: 0.2125 - categorical_accuracy: 0.9341
43520/60000 [====================>.........] - ETA: 29s - loss: 0.2124 - categorical_accuracy: 0.9342
43552/60000 [====================>.........] - ETA: 29s - loss: 0.2124 - categorical_accuracy: 0.9341
43584/60000 [====================>.........] - ETA: 29s - loss: 0.2124 - categorical_accuracy: 0.9342
43616/60000 [====================>.........] - ETA: 29s - loss: 0.2123 - categorical_accuracy: 0.9342
43648/60000 [====================>.........] - ETA: 29s - loss: 0.2122 - categorical_accuracy: 0.9343
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2121 - categorical_accuracy: 0.9343
43712/60000 [====================>.........] - ETA: 29s - loss: 0.2121 - categorical_accuracy: 0.9343
43744/60000 [====================>.........] - ETA: 29s - loss: 0.2121 - categorical_accuracy: 0.9343
43776/60000 [====================>.........] - ETA: 29s - loss: 0.2121 - categorical_accuracy: 0.9343
43808/60000 [====================>.........] - ETA: 29s - loss: 0.2122 - categorical_accuracy: 0.9343
43840/60000 [====================>.........] - ETA: 29s - loss: 0.2120 - categorical_accuracy: 0.9344
43872/60000 [====================>.........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9344
43904/60000 [====================>.........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9344
43936/60000 [====================>.........] - ETA: 29s - loss: 0.2118 - categorical_accuracy: 0.9344
43968/60000 [====================>.........] - ETA: 28s - loss: 0.2118 - categorical_accuracy: 0.9344
44000/60000 [=====================>........] - ETA: 28s - loss: 0.2117 - categorical_accuracy: 0.9345
44032/60000 [=====================>........] - ETA: 28s - loss: 0.2117 - categorical_accuracy: 0.9345
44064/60000 [=====================>........] - ETA: 28s - loss: 0.2117 - categorical_accuracy: 0.9345
44096/60000 [=====================>........] - ETA: 28s - loss: 0.2116 - categorical_accuracy: 0.9345
44128/60000 [=====================>........] - ETA: 28s - loss: 0.2116 - categorical_accuracy: 0.9345
44160/60000 [=====================>........] - ETA: 28s - loss: 0.2114 - categorical_accuracy: 0.9346
44192/60000 [=====================>........] - ETA: 28s - loss: 0.2113 - categorical_accuracy: 0.9346
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2112 - categorical_accuracy: 0.9346
44256/60000 [=====================>........] - ETA: 28s - loss: 0.2113 - categorical_accuracy: 0.9346
44288/60000 [=====================>........] - ETA: 28s - loss: 0.2112 - categorical_accuracy: 0.9347
44320/60000 [=====================>........] - ETA: 28s - loss: 0.2111 - categorical_accuracy: 0.9347
44352/60000 [=====================>........] - ETA: 28s - loss: 0.2109 - categorical_accuracy: 0.9348
44384/60000 [=====================>........] - ETA: 28s - loss: 0.2108 - categorical_accuracy: 0.9348
44416/60000 [=====================>........] - ETA: 28s - loss: 0.2107 - categorical_accuracy: 0.9349
44448/60000 [=====================>........] - ETA: 28s - loss: 0.2108 - categorical_accuracy: 0.9349
44480/60000 [=====================>........] - ETA: 28s - loss: 0.2108 - categorical_accuracy: 0.9348
44512/60000 [=====================>........] - ETA: 28s - loss: 0.2107 - categorical_accuracy: 0.9349
44544/60000 [=====================>........] - ETA: 27s - loss: 0.2106 - categorical_accuracy: 0.9349
44576/60000 [=====================>........] - ETA: 27s - loss: 0.2105 - categorical_accuracy: 0.9349
44608/60000 [=====================>........] - ETA: 27s - loss: 0.2104 - categorical_accuracy: 0.9350
44640/60000 [=====================>........] - ETA: 27s - loss: 0.2103 - categorical_accuracy: 0.9350
44672/60000 [=====================>........] - ETA: 27s - loss: 0.2102 - categorical_accuracy: 0.9350
44704/60000 [=====================>........] - ETA: 27s - loss: 0.2101 - categorical_accuracy: 0.9351
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2100 - categorical_accuracy: 0.9351
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2099 - categorical_accuracy: 0.9351
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2098 - categorical_accuracy: 0.9352
44832/60000 [=====================>........] - ETA: 27s - loss: 0.2097 - categorical_accuracy: 0.9352
44864/60000 [=====================>........] - ETA: 27s - loss: 0.2096 - categorical_accuracy: 0.9352
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2095 - categorical_accuracy: 0.9353
44928/60000 [=====================>........] - ETA: 27s - loss: 0.2094 - categorical_accuracy: 0.9353
44960/60000 [=====================>........] - ETA: 27s - loss: 0.2094 - categorical_accuracy: 0.9353
44992/60000 [=====================>........] - ETA: 27s - loss: 0.2093 - categorical_accuracy: 0.9353
45024/60000 [=====================>........] - ETA: 27s - loss: 0.2092 - categorical_accuracy: 0.9353
45056/60000 [=====================>........] - ETA: 27s - loss: 0.2092 - categorical_accuracy: 0.9353
45088/60000 [=====================>........] - ETA: 26s - loss: 0.2091 - categorical_accuracy: 0.9354
45120/60000 [=====================>........] - ETA: 26s - loss: 0.2090 - categorical_accuracy: 0.9354
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2089 - categorical_accuracy: 0.9354
45184/60000 [=====================>........] - ETA: 26s - loss: 0.2087 - categorical_accuracy: 0.9354
45216/60000 [=====================>........] - ETA: 26s - loss: 0.2086 - categorical_accuracy: 0.9355
45248/60000 [=====================>........] - ETA: 26s - loss: 0.2085 - categorical_accuracy: 0.9355
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2084 - categorical_accuracy: 0.9355
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2084 - categorical_accuracy: 0.9355
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2083 - categorical_accuracy: 0.9356
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9356
45408/60000 [=====================>........] - ETA: 26s - loss: 0.2083 - categorical_accuracy: 0.9356
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9356
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9356
45504/60000 [=====================>........] - ETA: 26s - loss: 0.2083 - categorical_accuracy: 0.9356
45536/60000 [=====================>........] - ETA: 26s - loss: 0.2083 - categorical_accuracy: 0.9356
45568/60000 [=====================>........] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9356
45600/60000 [=====================>........] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9356
45632/60000 [=====================>........] - ETA: 25s - loss: 0.2081 - categorical_accuracy: 0.9356
45664/60000 [=====================>........] - ETA: 25s - loss: 0.2080 - categorical_accuracy: 0.9357
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2079 - categorical_accuracy: 0.9357
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2079 - categorical_accuracy: 0.9357
45760/60000 [=====================>........] - ETA: 25s - loss: 0.2079 - categorical_accuracy: 0.9357
45792/60000 [=====================>........] - ETA: 25s - loss: 0.2079 - categorical_accuracy: 0.9357
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2078 - categorical_accuracy: 0.9357
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2077 - categorical_accuracy: 0.9357
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2076 - categorical_accuracy: 0.9358
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2076 - categorical_accuracy: 0.9358
45952/60000 [=====================>........] - ETA: 25s - loss: 0.2074 - categorical_accuracy: 0.9358
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2073 - categorical_accuracy: 0.9359
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2072 - categorical_accuracy: 0.9359
46048/60000 [======================>.......] - ETA: 25s - loss: 0.2071 - categorical_accuracy: 0.9359
46080/60000 [======================>.......] - ETA: 25s - loss: 0.2069 - categorical_accuracy: 0.9360
46112/60000 [======================>.......] - ETA: 25s - loss: 0.2069 - categorical_accuracy: 0.9360
46144/60000 [======================>.......] - ETA: 25s - loss: 0.2068 - categorical_accuracy: 0.9360
46176/60000 [======================>.......] - ETA: 24s - loss: 0.2067 - categorical_accuracy: 0.9360
46208/60000 [======================>.......] - ETA: 24s - loss: 0.2066 - categorical_accuracy: 0.9361
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2065 - categorical_accuracy: 0.9361
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2064 - categorical_accuracy: 0.9362
46304/60000 [======================>.......] - ETA: 24s - loss: 0.2063 - categorical_accuracy: 0.9362
46336/60000 [======================>.......] - ETA: 24s - loss: 0.2062 - categorical_accuracy: 0.9362
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2062 - categorical_accuracy: 0.9362
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2060 - categorical_accuracy: 0.9363
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2059 - categorical_accuracy: 0.9363
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2058 - categorical_accuracy: 0.9363
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2057 - categorical_accuracy: 0.9364
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9364
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2057 - categorical_accuracy: 0.9364
46592/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9364
46624/60000 [======================>.......] - ETA: 24s - loss: 0.2055 - categorical_accuracy: 0.9364
46656/60000 [======================>.......] - ETA: 24s - loss: 0.2055 - categorical_accuracy: 0.9365
46688/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9365
46720/60000 [======================>.......] - ETA: 23s - loss: 0.2054 - categorical_accuracy: 0.9365
46752/60000 [======================>.......] - ETA: 23s - loss: 0.2052 - categorical_accuracy: 0.9366
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2051 - categorical_accuracy: 0.9366
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2050 - categorical_accuracy: 0.9366
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2052 - categorical_accuracy: 0.9366
46880/60000 [======================>.......] - ETA: 23s - loss: 0.2052 - categorical_accuracy: 0.9366
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2050 - categorical_accuracy: 0.9366
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2049 - categorical_accuracy: 0.9367
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2048 - categorical_accuracy: 0.9367
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9367
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9368
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2045 - categorical_accuracy: 0.9368
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2045 - categorical_accuracy: 0.9368
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2044 - categorical_accuracy: 0.9368
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2043 - categorical_accuracy: 0.9369
47200/60000 [======================>.......] - ETA: 23s - loss: 0.2042 - categorical_accuracy: 0.9369
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2041 - categorical_accuracy: 0.9369
47264/60000 [======================>.......] - ETA: 22s - loss: 0.2040 - categorical_accuracy: 0.9369
47296/60000 [======================>.......] - ETA: 22s - loss: 0.2041 - categorical_accuracy: 0.9370
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2040 - categorical_accuracy: 0.9370
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2039 - categorical_accuracy: 0.9370
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2038 - categorical_accuracy: 0.9371
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2037 - categorical_accuracy: 0.9371
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2036 - categorical_accuracy: 0.9371
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2035 - categorical_accuracy: 0.9371
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2034 - categorical_accuracy: 0.9371
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9372
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2031 - categorical_accuracy: 0.9372
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9372
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9373
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9373
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9373
47744/60000 [======================>.......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9373
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2031 - categorical_accuracy: 0.9373
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2030 - categorical_accuracy: 0.9374
47840/60000 [======================>.......] - ETA: 21s - loss: 0.2029 - categorical_accuracy: 0.9374
47872/60000 [======================>.......] - ETA: 21s - loss: 0.2028 - categorical_accuracy: 0.9374
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2029 - categorical_accuracy: 0.9374
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2028 - categorical_accuracy: 0.9374
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2028 - categorical_accuracy: 0.9374
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2030 - categorical_accuracy: 0.9374
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2029 - categorical_accuracy: 0.9374
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2028 - categorical_accuracy: 0.9374
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2027 - categorical_accuracy: 0.9374
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2027 - categorical_accuracy: 0.9375
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2026 - categorical_accuracy: 0.9375
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2025 - categorical_accuracy: 0.9375
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2024 - categorical_accuracy: 0.9375
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2024 - categorical_accuracy: 0.9376
48288/60000 [=======================>......] - ETA: 21s - loss: 0.2023 - categorical_accuracy: 0.9376
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2023 - categorical_accuracy: 0.9376
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2022 - categorical_accuracy: 0.9376
48384/60000 [=======================>......] - ETA: 20s - loss: 0.2021 - categorical_accuracy: 0.9376
48416/60000 [=======================>......] - ETA: 20s - loss: 0.2020 - categorical_accuracy: 0.9377
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2019 - categorical_accuracy: 0.9377
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2018 - categorical_accuracy: 0.9377
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2017 - categorical_accuracy: 0.9378
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2016 - categorical_accuracy: 0.9378
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9379
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9379
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9379
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2012 - categorical_accuracy: 0.9379
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9379
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2012 - categorical_accuracy: 0.9380
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2011 - categorical_accuracy: 0.9380
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2011 - categorical_accuracy: 0.9380
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2010 - categorical_accuracy: 0.9380
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2010 - categorical_accuracy: 0.9380
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2009 - categorical_accuracy: 0.9381
48928/60000 [=======================>......] - ETA: 19s - loss: 0.2008 - categorical_accuracy: 0.9381
48960/60000 [=======================>......] - ETA: 19s - loss: 0.2010 - categorical_accuracy: 0.9380
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2009 - categorical_accuracy: 0.9381
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2008 - categorical_accuracy: 0.9381
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2007 - categorical_accuracy: 0.9381
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2007 - categorical_accuracy: 0.9381
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2006 - categorical_accuracy: 0.9381
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2005 - categorical_accuracy: 0.9382
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9382
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9382
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9382
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2005 - categorical_accuracy: 0.9382
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9382
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9382
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9382
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9382
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9383
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2001 - categorical_accuracy: 0.9383
49504/60000 [=======================>......] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9383
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9383
49568/60000 [=======================>......] - ETA: 18s - loss: 0.1999 - categorical_accuracy: 0.9383
49600/60000 [=======================>......] - ETA: 18s - loss: 0.1999 - categorical_accuracy: 0.9383
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9383
49664/60000 [=======================>......] - ETA: 18s - loss: 0.1999 - categorical_accuracy: 0.9384
49696/60000 [=======================>......] - ETA: 18s - loss: 0.1999 - categorical_accuracy: 0.9384
49728/60000 [=======================>......] - ETA: 18s - loss: 0.1997 - categorical_accuracy: 0.9384
49760/60000 [=======================>......] - ETA: 18s - loss: 0.1996 - categorical_accuracy: 0.9385
49792/60000 [=======================>......] - ETA: 18s - loss: 0.1995 - categorical_accuracy: 0.9385
49824/60000 [=======================>......] - ETA: 18s - loss: 0.1994 - categorical_accuracy: 0.9385
49856/60000 [=======================>......] - ETA: 18s - loss: 0.1993 - categorical_accuracy: 0.9386
49888/60000 [=======================>......] - ETA: 18s - loss: 0.1992 - categorical_accuracy: 0.9386
49920/60000 [=======================>......] - ETA: 18s - loss: 0.1992 - categorical_accuracy: 0.9386
49952/60000 [=======================>......] - ETA: 18s - loss: 0.1991 - categorical_accuracy: 0.9386
49984/60000 [=======================>......] - ETA: 18s - loss: 0.1990 - categorical_accuracy: 0.9386
50016/60000 [========================>.....] - ETA: 18s - loss: 0.1989 - categorical_accuracy: 0.9387
50048/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9387
50080/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9387
50112/60000 [========================>.....] - ETA: 17s - loss: 0.1987 - categorical_accuracy: 0.9387
50144/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9387
50176/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9387
50208/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9388
50240/60000 [========================>.....] - ETA: 17s - loss: 0.1986 - categorical_accuracy: 0.9388
50272/60000 [========================>.....] - ETA: 17s - loss: 0.1986 - categorical_accuracy: 0.9388
50304/60000 [========================>.....] - ETA: 17s - loss: 0.1985 - categorical_accuracy: 0.9388
50336/60000 [========================>.....] - ETA: 17s - loss: 0.1984 - categorical_accuracy: 0.9389
50368/60000 [========================>.....] - ETA: 17s - loss: 0.1983 - categorical_accuracy: 0.9389
50400/60000 [========================>.....] - ETA: 17s - loss: 0.1983 - categorical_accuracy: 0.9389
50432/60000 [========================>.....] - ETA: 17s - loss: 0.1982 - categorical_accuracy: 0.9389
50464/60000 [========================>.....] - ETA: 17s - loss: 0.1981 - categorical_accuracy: 0.9389
50496/60000 [========================>.....] - ETA: 17s - loss: 0.1980 - categorical_accuracy: 0.9390
50528/60000 [========================>.....] - ETA: 17s - loss: 0.1980 - categorical_accuracy: 0.9390
50560/60000 [========================>.....] - ETA: 17s - loss: 0.1979 - categorical_accuracy: 0.9390
50592/60000 [========================>.....] - ETA: 16s - loss: 0.1980 - categorical_accuracy: 0.9390
50624/60000 [========================>.....] - ETA: 16s - loss: 0.1979 - categorical_accuracy: 0.9390
50656/60000 [========================>.....] - ETA: 16s - loss: 0.1978 - categorical_accuracy: 0.9390
50688/60000 [========================>.....] - ETA: 16s - loss: 0.1977 - categorical_accuracy: 0.9391
50720/60000 [========================>.....] - ETA: 16s - loss: 0.1977 - categorical_accuracy: 0.9391
50752/60000 [========================>.....] - ETA: 16s - loss: 0.1976 - categorical_accuracy: 0.9391
50784/60000 [========================>.....] - ETA: 16s - loss: 0.1975 - categorical_accuracy: 0.9391
50816/60000 [========================>.....] - ETA: 16s - loss: 0.1975 - categorical_accuracy: 0.9391
50848/60000 [========================>.....] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9391
50880/60000 [========================>.....] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9392
50912/60000 [========================>.....] - ETA: 16s - loss: 0.1973 - categorical_accuracy: 0.9392
50944/60000 [========================>.....] - ETA: 16s - loss: 0.1972 - categorical_accuracy: 0.9392
50976/60000 [========================>.....] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9392
51008/60000 [========================>.....] - ETA: 16s - loss: 0.1970 - categorical_accuracy: 0.9392
51040/60000 [========================>.....] - ETA: 16s - loss: 0.1970 - categorical_accuracy: 0.9392
51072/60000 [========================>.....] - ETA: 16s - loss: 0.1969 - categorical_accuracy: 0.9393
51104/60000 [========================>.....] - ETA: 16s - loss: 0.1968 - categorical_accuracy: 0.9393
51136/60000 [========================>.....] - ETA: 16s - loss: 0.1966 - categorical_accuracy: 0.9394
51168/60000 [========================>.....] - ETA: 15s - loss: 0.1966 - categorical_accuracy: 0.9394
51200/60000 [========================>.....] - ETA: 15s - loss: 0.1966 - categorical_accuracy: 0.9393
51232/60000 [========================>.....] - ETA: 15s - loss: 0.1966 - categorical_accuracy: 0.9393
51264/60000 [========================>.....] - ETA: 15s - loss: 0.1966 - categorical_accuracy: 0.9393
51296/60000 [========================>.....] - ETA: 15s - loss: 0.1966 - categorical_accuracy: 0.9393
51328/60000 [========================>.....] - ETA: 15s - loss: 0.1965 - categorical_accuracy: 0.9393
51360/60000 [========================>.....] - ETA: 15s - loss: 0.1964 - categorical_accuracy: 0.9394
51392/60000 [========================>.....] - ETA: 15s - loss: 0.1963 - categorical_accuracy: 0.9394
51424/60000 [========================>.....] - ETA: 15s - loss: 0.1962 - categorical_accuracy: 0.9394
51456/60000 [========================>.....] - ETA: 15s - loss: 0.1961 - categorical_accuracy: 0.9395
51488/60000 [========================>.....] - ETA: 15s - loss: 0.1960 - categorical_accuracy: 0.9395
51520/60000 [========================>.....] - ETA: 15s - loss: 0.1959 - categorical_accuracy: 0.9395
51552/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9396
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1957 - categorical_accuracy: 0.9396
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1956 - categorical_accuracy: 0.9396
51648/60000 [========================>.....] - ETA: 15s - loss: 0.1955 - categorical_accuracy: 0.9396
51680/60000 [========================>.....] - ETA: 15s - loss: 0.1956 - categorical_accuracy: 0.9396
51712/60000 [========================>.....] - ETA: 14s - loss: 0.1956 - categorical_accuracy: 0.9397
51744/60000 [========================>.....] - ETA: 14s - loss: 0.1955 - categorical_accuracy: 0.9397
51776/60000 [========================>.....] - ETA: 14s - loss: 0.1955 - categorical_accuracy: 0.9397
51808/60000 [========================>.....] - ETA: 14s - loss: 0.1954 - categorical_accuracy: 0.9397
51840/60000 [========================>.....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9398
51872/60000 [========================>.....] - ETA: 14s - loss: 0.1952 - categorical_accuracy: 0.9398
51904/60000 [========================>.....] - ETA: 14s - loss: 0.1951 - categorical_accuracy: 0.9398
51936/60000 [========================>.....] - ETA: 14s - loss: 0.1950 - categorical_accuracy: 0.9398
51968/60000 [========================>.....] - ETA: 14s - loss: 0.1950 - categorical_accuracy: 0.9398
52000/60000 [=========================>....] - ETA: 14s - loss: 0.1951 - categorical_accuracy: 0.9398
52032/60000 [=========================>....] - ETA: 14s - loss: 0.1950 - categorical_accuracy: 0.9398
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1949 - categorical_accuracy: 0.9399
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1948 - categorical_accuracy: 0.9399
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1947 - categorical_accuracy: 0.9399
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1948 - categorical_accuracy: 0.9399
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1949 - categorical_accuracy: 0.9399
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1948 - categorical_accuracy: 0.9399
52256/60000 [=========================>....] - ETA: 13s - loss: 0.1948 - categorical_accuracy: 0.9399
52288/60000 [=========================>....] - ETA: 13s - loss: 0.1947 - categorical_accuracy: 0.9400
52320/60000 [=========================>....] - ETA: 13s - loss: 0.1946 - categorical_accuracy: 0.9400
52352/60000 [=========================>....] - ETA: 13s - loss: 0.1945 - categorical_accuracy: 0.9400
52384/60000 [=========================>....] - ETA: 13s - loss: 0.1945 - categorical_accuracy: 0.9400
52416/60000 [=========================>....] - ETA: 13s - loss: 0.1944 - categorical_accuracy: 0.9401
52448/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9401
52480/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9401
52512/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9401
52544/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9401
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9401
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1940 - categorical_accuracy: 0.9401
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1940 - categorical_accuracy: 0.9401
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1939 - categorical_accuracy: 0.9402
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9402
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9402
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9402
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9402
52832/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9402
52864/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9402
52896/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9402
52928/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9401
52960/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9402
52992/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9402
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9402
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9403
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9403
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9402
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9402
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9402
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9402
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1935 - categorical_accuracy: 0.9403
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9403
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1933 - categorical_accuracy: 0.9404
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1932 - categorical_accuracy: 0.9404
53376/60000 [=========================>....] - ETA: 11s - loss: 0.1931 - categorical_accuracy: 0.9404
53408/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9404
53440/60000 [=========================>....] - ETA: 11s - loss: 0.1929 - categorical_accuracy: 0.9405
53472/60000 [=========================>....] - ETA: 11s - loss: 0.1928 - categorical_accuracy: 0.9405
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9405
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9405
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9406
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9406
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9406
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9406
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9406
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1924 - categorical_accuracy: 0.9406
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1923 - categorical_accuracy: 0.9407
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1923 - categorical_accuracy: 0.9407
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1922 - categorical_accuracy: 0.9407
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1923 - categorical_accuracy: 0.9407
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1922 - categorical_accuracy: 0.9407
53920/60000 [=========================>....] - ETA: 10s - loss: 0.1921 - categorical_accuracy: 0.9407
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1920 - categorical_accuracy: 0.9407
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1920 - categorical_accuracy: 0.9407
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1920 - categorical_accuracy: 0.9407
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1919 - categorical_accuracy: 0.9408
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1918 - categorical_accuracy: 0.9408
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1917 - categorical_accuracy: 0.9408
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1916 - categorical_accuracy: 0.9409
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9409
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1914 - categorical_accuracy: 0.9409
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1913 - categorical_accuracy: 0.9410
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1916 - categorical_accuracy: 0.9410
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1916 - categorical_accuracy: 0.9410
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1916 - categorical_accuracy: 0.9410
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9410
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9410
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9410
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1915 - categorical_accuracy: 0.9410 
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1914 - categorical_accuracy: 0.9411
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1913 - categorical_accuracy: 0.9411
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1913 - categorical_accuracy: 0.9411
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1913 - categorical_accuracy: 0.9411
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1912 - categorical_accuracy: 0.9411
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1912 - categorical_accuracy: 0.9411
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1911 - categorical_accuracy: 0.9412
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1910 - categorical_accuracy: 0.9412
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1909 - categorical_accuracy: 0.9412
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1909 - categorical_accuracy: 0.9412
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9413
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9413
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9413
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1906 - categorical_accuracy: 0.9413
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9413
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9414
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1904 - categorical_accuracy: 0.9414
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1903 - categorical_accuracy: 0.9414
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1903 - categorical_accuracy: 0.9414
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1902 - categorical_accuracy: 0.9414
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1902 - categorical_accuracy: 0.9414
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1900 - categorical_accuracy: 0.9415
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1900 - categorical_accuracy: 0.9415
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1899 - categorical_accuracy: 0.9415
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1898 - categorical_accuracy: 0.9415
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1897 - categorical_accuracy: 0.9416
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1897 - categorical_accuracy: 0.9416
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1896 - categorical_accuracy: 0.9416
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1896 - categorical_accuracy: 0.9416
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1895 - categorical_accuracy: 0.9416
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1894 - categorical_accuracy: 0.9416
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1893 - categorical_accuracy: 0.9417
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1894 - categorical_accuracy: 0.9417
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1893 - categorical_accuracy: 0.9417
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1893 - categorical_accuracy: 0.9417
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1892 - categorical_accuracy: 0.9417
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1891 - categorical_accuracy: 0.9418
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1890 - categorical_accuracy: 0.9418
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1890 - categorical_accuracy: 0.9418
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1889 - categorical_accuracy: 0.9418
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1889 - categorical_accuracy: 0.9418
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1888 - categorical_accuracy: 0.9419
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1887 - categorical_accuracy: 0.9419
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1887 - categorical_accuracy: 0.9419
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1886 - categorical_accuracy: 0.9419
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1885 - categorical_accuracy: 0.9419
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1885 - categorical_accuracy: 0.9419
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1884 - categorical_accuracy: 0.9420
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1883 - categorical_accuracy: 0.9420
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1882 - categorical_accuracy: 0.9420
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9420
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1880 - categorical_accuracy: 0.9421
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1879 - categorical_accuracy: 0.9421
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1879 - categorical_accuracy: 0.9421
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1879 - categorical_accuracy: 0.9421
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1879 - categorical_accuracy: 0.9421
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1878 - categorical_accuracy: 0.9422
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1877 - categorical_accuracy: 0.9422
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1876 - categorical_accuracy: 0.9422
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1876 - categorical_accuracy: 0.9422
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1875 - categorical_accuracy: 0.9423
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1874 - categorical_accuracy: 0.9422
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1874 - categorical_accuracy: 0.9422
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1874 - categorical_accuracy: 0.9422
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1873 - categorical_accuracy: 0.9422
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1873 - categorical_accuracy: 0.9422
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1873 - categorical_accuracy: 0.9422
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1872 - categorical_accuracy: 0.9423
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9423
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1870 - categorical_accuracy: 0.9423
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1869 - categorical_accuracy: 0.9424
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1869 - categorical_accuracy: 0.9424
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1868 - categorical_accuracy: 0.9424
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1868 - categorical_accuracy: 0.9424
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1867 - categorical_accuracy: 0.9424
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1866 - categorical_accuracy: 0.9424
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1865 - categorical_accuracy: 0.9425
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1865 - categorical_accuracy: 0.9425
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9425
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9425
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9425
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9425
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9426
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9426
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9426
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9426
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1862 - categorical_accuracy: 0.9426
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1862 - categorical_accuracy: 0.9426
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1861 - categorical_accuracy: 0.9427
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1860 - categorical_accuracy: 0.9427
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1859 - categorical_accuracy: 0.9427
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1858 - categorical_accuracy: 0.9428
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1857 - categorical_accuracy: 0.9428
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1857 - categorical_accuracy: 0.9428
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1856 - categorical_accuracy: 0.9428
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1856 - categorical_accuracy: 0.9428
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1856 - categorical_accuracy: 0.9429
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1855 - categorical_accuracy: 0.9429
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1855 - categorical_accuracy: 0.9429
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9429
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9429
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9430
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9430
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1851 - categorical_accuracy: 0.9430
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1851 - categorical_accuracy: 0.9430
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1852 - categorical_accuracy: 0.9430
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1851 - categorical_accuracy: 0.9431
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1850 - categorical_accuracy: 0.9431
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1850 - categorical_accuracy: 0.9431
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1849 - categorical_accuracy: 0.9431
58016/60000 [============================>.] - ETA: 3s - loss: 0.1849 - categorical_accuracy: 0.9431
58048/60000 [============================>.] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9432
58080/60000 [============================>.] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9432
58112/60000 [============================>.] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9432
58144/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9432
58176/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9432
58208/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9432
58240/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9432
58272/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9433
58304/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9433
58336/60000 [============================>.] - ETA: 3s - loss: 0.1842 - categorical_accuracy: 0.9433
58368/60000 [============================>.] - ETA: 2s - loss: 0.1841 - categorical_accuracy: 0.9433
58400/60000 [============================>.] - ETA: 2s - loss: 0.1841 - categorical_accuracy: 0.9433
58432/60000 [============================>.] - ETA: 2s - loss: 0.1840 - categorical_accuracy: 0.9434
58464/60000 [============================>.] - ETA: 2s - loss: 0.1840 - categorical_accuracy: 0.9434
58496/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9434
58528/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9434
58560/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9434
58592/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9435
58624/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9434
58656/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9434
58688/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9434
58720/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9434
58752/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9435
58784/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9435
58816/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9435
58848/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9435
58880/60000 [============================>.] - ETA: 2s - loss: 0.1832 - categorical_accuracy: 0.9435
58912/60000 [============================>.] - ETA: 1s - loss: 0.1832 - categorical_accuracy: 0.9435
58944/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9436
58976/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9436
59008/60000 [============================>.] - ETA: 1s - loss: 0.1830 - categorical_accuracy: 0.9436
59040/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9436
59072/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9437
59104/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9437
59136/60000 [============================>.] - ETA: 1s - loss: 0.1827 - categorical_accuracy: 0.9437
59168/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9438
59200/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9438
59232/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9438
59264/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9438
59296/60000 [============================>.] - ETA: 1s - loss: 0.1824 - categorical_accuracy: 0.9438
59328/60000 [============================>.] - ETA: 1s - loss: 0.1823 - categorical_accuracy: 0.9439
59360/60000 [============================>.] - ETA: 1s - loss: 0.1822 - categorical_accuracy: 0.9439
59392/60000 [============================>.] - ETA: 1s - loss: 0.1822 - categorical_accuracy: 0.9439
59424/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9439
59456/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9439
59488/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9440
59520/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9440
59552/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9440
59584/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9440
59616/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9440
59648/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9440
59680/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9440
59712/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9440
59744/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9441
59776/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9441
59808/60000 [============================>.] - ETA: 0s - loss: 0.1815 - categorical_accuracy: 0.9441
59840/60000 [============================>.] - ETA: 0s - loss: 0.1815 - categorical_accuracy: 0.9441
59872/60000 [============================>.] - ETA: 0s - loss: 0.1814 - categorical_accuracy: 0.9441
59904/60000 [============================>.] - ETA: 0s - loss: 0.1815 - categorical_accuracy: 0.9441
59936/60000 [============================>.] - ETA: 0s - loss: 0.1814 - categorical_accuracy: 0.9441
59968/60000 [============================>.] - ETA: 0s - loss: 0.1815 - categorical_accuracy: 0.9442
60000/60000 [==============================] - 112s 2ms/step - loss: 0.1814 - categorical_accuracy: 0.9442 - val_loss: 0.0427 - val_categorical_accuracy: 0.9862

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 15s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  672/10000 [=>............................] - ETA: 3s
  832/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 3s
 1312/10000 [==>...........................] - ETA: 3s
 1472/10000 [===>..........................] - ETA: 3s
 1632/10000 [===>..........................] - ETA: 3s
 1792/10000 [====>.........................] - ETA: 3s
 1952/10000 [====>.........................] - ETA: 2s
 2112/10000 [=====>........................] - ETA: 2s
 2272/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2592/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
 3232/10000 [========>.....................] - ETA: 2s
 3392/10000 [=========>....................] - ETA: 2s
 3552/10000 [=========>....................] - ETA: 2s
 3712/10000 [==========>...................] - ETA: 2s
 3872/10000 [==========>...................] - ETA: 2s
 4032/10000 [===========>..................] - ETA: 2s
 4192/10000 [===========>..................] - ETA: 2s
 4352/10000 [============>.................] - ETA: 1s
 4512/10000 [============>.................] - ETA: 1s
 4672/10000 [=============>................] - ETA: 1s
 4832/10000 [=============>................] - ETA: 1s
 4992/10000 [=============>................] - ETA: 1s
 5152/10000 [==============>...............] - ETA: 1s
 5312/10000 [==============>...............] - ETA: 1s
 5472/10000 [===============>..............] - ETA: 1s
 5632/10000 [===============>..............] - ETA: 1s
 5792/10000 [================>.............] - ETA: 1s
 5920/10000 [================>.............] - ETA: 1s
 6048/10000 [=================>............] - ETA: 1s
 6176/10000 [=================>............] - ETA: 1s
 6336/10000 [==================>...........] - ETA: 1s
 6496/10000 [==================>...........] - ETA: 1s
 6656/10000 [==================>...........] - ETA: 1s
 6816/10000 [===================>..........] - ETA: 1s
 6976/10000 [===================>..........] - ETA: 1s
 7136/10000 [====================>.........] - ETA: 1s
 7296/10000 [====================>.........] - ETA: 0s
 7456/10000 [=====================>........] - ETA: 0s
 7616/10000 [=====================>........] - ETA: 0s
 7776/10000 [======================>.......] - ETA: 0s
 7936/10000 [======================>.......] - ETA: 0s
 8096/10000 [=======================>......] - ETA: 0s
 8224/10000 [=======================>......] - ETA: 0s
 8384/10000 [========================>.....] - ETA: 0s
 8544/10000 [========================>.....] - ETA: 0s
 8704/10000 [=========================>....] - ETA: 0s
 8864/10000 [=========================>....] - ETA: 0s
 9024/10000 [==========================>...] - ETA: 0s
 9184/10000 [==========================>...] - ETA: 0s
 9344/10000 [===========================>..] - ETA: 0s
 9504/10000 [===========================>..] - ETA: 0s
 9664/10000 [===========================>..] - ETA: 0s
 9824/10000 [============================>.] - ETA: 0s
 9984/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 355us/step
[[1.0465826e-07 1.5616136e-07 5.8198322e-07 ... 9.9999774e-01
  2.7940333e-08 1.1831776e-06]
 [3.9218307e-06 1.4956120e-05 9.9996924e-01 ... 1.2402359e-07
  6.0498423e-06 8.3504812e-09]
 [1.9926708e-06 9.9985313e-01 1.2148197e-05 ... 1.6872615e-05
  1.6143442e-05 2.3412285e-07]
 ...
 [4.2537365e-09 4.3947333e-07 2.4099103e-08 ... 3.8268789e-07
  3.9159331e-06 1.5254813e-05]
 [1.1147683e-07 2.9895062e-07 2.3135495e-08 ... 1.4285189e-07
  5.3309195e-04 3.9027364e-06]
 [2.0544599e-06 3.1117256e-07 7.8024141e-06 ... 4.9838169e-09
  9.5212619e-07 1.0315734e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04274454051963985, 'accuracy_test:': 0.9861999750137329}

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
From github.com:arita37/mlmodels_store
   6318d37..c7cdf06  master     -> origin/master
Updating 6318d37..c7cdf06
Fast-forward
 error_list/20200521/list_log_testall_20200521.md | 433 +++++++++++++++++++++++
 1 file changed, 433 insertions(+)
[master 7525436] ml_store
 1 file changed, 2043 insertions(+)
To github.com:arita37/mlmodels_store.git
   c7cdf06..7525436  master -> master





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
{'loss': 0.40887312963604927, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-21 12:27:29.615668: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 2baf335] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   7525436..2baf335  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'

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
[master 41d73ea] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   2baf335..41d73ea  master -> master





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
	Data preprocessing and feature engineering runtime = 0.22s ...
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
 40%|████      | 2/5 [00:19<00:29,  9.79s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.7853227164086268, 'learning_rate': 0.18510191400544823, 'min_data_in_leaf': 17, 'num_leaves': 49} and reward: 0.3922
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9!]\x1a\xf8\xfdkX\r\x00\x00\x00learning_rateq\x02G?\xc7\xb1ke\x8aJ\xb2X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K1u.' and reward: 0.3922
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9!]\x1a\xf8\xfdkX\r\x00\x00\x00learning_rateq\x02G?\xc7\xb1ke\x8aJ\xb2X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K1u.' and reward: 0.3922
 60%|██████    | 3/5 [00:42<00:27, 13.69s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.7841249229176862, 'learning_rate': 0.012352760902951137, 'min_data_in_leaf': 11, 'num_leaves': 35} and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x17\x8d&}\x1e\x95X\r\x00\x00\x00learning_rateq\x02G?\x89Lg\x80\xbfs\xc9X\x10\x00\x00\x00min_data_in_leafq\x03K\x0bX\n\x00\x00\x00num_leavesq\x04K#u.' and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\x17\x8d&}\x1e\x95X\r\x00\x00\x00learning_rateq\x02G?\x89Lg\x80\xbfs\xc9X\x10\x00\x00\x00min_data_in_leafq\x03K\x0bX\n\x00\x00\x00num_leavesq\x04K#u.' and reward: 0.39
 80%|████████  | 4/5 [01:01<00:15, 15.36s/it] 80%|████████  | 4/5 [01:01<00:15, 15.41s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9225775311002837, 'learning_rate': 0.15149071638777856, 'min_data_in_leaf': 26, 'num_leaves': 49} and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x85\xc1P\x834\\X\r\x00\x00\x00learning_rateq\x02G?\xc3d\x0c<D=HX\x10\x00\x00\x00min_data_in_leafq\x03K\x1aX\n\x00\x00\x00num_leavesq\x04K1u.' and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x85\xc1P\x834\\X\r\x00\x00\x00learning_rateq\x02G?\xc3d\x0c<D=HX\x10\x00\x00\x00min_data_in_leafq\x03K\x1aX\n\x00\x00\x00num_leavesq\x04K1u.' and reward: 0.3896
Time for Gradient Boosting hyperparameter optimization: 85.27952694892883
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.7853227164086268, 'learning_rate': 0.18510191400544823, 'min_data_in_leaf': 17, 'num_leaves': 49}
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:46<01:10, 23.36s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.2996783322812887, 'embedding_size_factor': 1.0736689560487636, 'layers.choice': 2, 'learning_rate': 0.0006365904650368666, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.00013862031329606362} and reward: 0.3764
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd3-\xee\x07\x1d\xf2\xfeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1-\xbf\x7f\xcf[\xc1X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?D\xdc\x1b\x9d8=%X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?"+Sl\xa1DFu.' and reward: 0.3764
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd3-\xee\x07\x1d\xf2\xfeX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1-\xbf\x7f\xcf[\xc1X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?D\xdc\x1b\x9d8=%X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?"+Sl\xa1DFu.' and reward: 0.3764
 60%|██████    | 3/5 [01:33<01:00, 30.48s/it] 60%|██████    | 3/5 [01:33<01:02, 31.27s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4572424533874028, 'embedding_size_factor': 0.9412358390445762, 'layers.choice': 2, 'learning_rate': 0.000552096800292631, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 9.853366932840157e-09} and reward: 0.3676
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xddCu\xd9\xe9\x11\x9bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\x1e\x9a\x9fP\xa0[X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?B\x17R\xd9\xca2\xb0X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>E(\xf2\x1d-:8u.' and reward: 0.3676
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xddCu\xd9\xe9\x11\x9bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xee\x1e\x9a\x9fP\xa0[X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?B\x17R\xd9\xca2\xb0X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>E(\xf2\x1d-:8u.' and reward: 0.3676
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 142.77706217765808
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -111.3s of remaining time.
Ensemble size: 88
Ensemble weights: 
[0.15909091 0.17045455 0.14772727 0.18181818 0.14772727 0.06818182
 0.125     ]
	0.3986	 = Validation accuracy score
	1.49s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 232.84s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f2865328cc0>

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
   41d73ea..941bfe4  master     -> origin/master
Updating 41d73ea..941bfe4
Fast-forward
 error_list/20200521/list_log_testall_20200521.md | 175 +++++++++++++++++++++++
 1 file changed, 175 insertions(+)
[master a0a50b7] ml_store
 1 file changed, 214 insertions(+)
To github.com:arita37/mlmodels_store.git
   941bfe4..a0a50b7  master -> master





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
[master eee2ae8] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   a0a50b7..eee2ae8  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.72it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 2.690 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.225292
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.22529182434082 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2f291ac400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2f291ac400>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]WARNING:root:multiple 5 does not divide base seasonality 1.Falling back to seasonality 1
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 93.91it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1154.4959309895833,
    "abs_error": 387.4217529296875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.567037094807621,
    "sMAPE": 0.5294876927155538,
    "MSIS": 102.68149026295896,
    "QuantileLoss[0.5]": 387.4217834472656,
    "Coverage[0.5]": 1.0,
    "RMSE": 33.977874138762466,
    "NRMSE": 0.7153236660792098,
    "ND": 0.679687285841557,
    "wQuantileLoss[0.5]": 0.6796873393811678,
    "mean_wQuantileLoss": 0.6796873393811678,
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
100%|██████████| 10/10 [00:01<00:00,  7.49it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.336 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2efcc59ef0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2efcc59ef0>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 160.69it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.66it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 1.768 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.253315
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.253314685821533 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2f2184c9b0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2f2184c9b0>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 163.04it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 231.0383504231771,
    "abs_error": 153.94161987304688,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.0200094487740463,
    "sMAPE": 0.2643360319453782,
    "MSIS": 40.800376333298324,
    "QuantileLoss[0.5]": 153.94160842895508,
    "Coverage[0.5]": 0.5,
    "RMSE": 15.199945737507655,
    "NRMSE": 0.3199988576317401,
    "ND": 0.27007301732113487,
    "wQuantileLoss[0.5]": 0.27007299724378087,
    "mean_wQuantileLoss": 0.27007299724378087,
    "MAE_Coverage": 0.0
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
 30%|███       | 3/10 [00:12<00:28,  4.09s/it, avg_epoch_loss=6.92] 60%|██████    | 6/10 [00:23<00:15,  3.96s/it, avg_epoch_loss=6.89] 90%|█████████ | 9/10 [00:34<00:03,  3.86s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:37<00:00,  3.75s/it, avg_epoch_loss=6.85]
INFO:root:Epoch[0] Elapsed time 37.512 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.854353
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.854352712631226 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2efcbb45c0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2efcbb45c0>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU
INFO:gluonts.model.wavenet._estimator:Using dilation depth 10 and receptive field length 1024

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 177.90it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 52303.296875,
    "abs_error": 2678.7744140625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.749424851649962,
    "sMAPE": 1.405925290350076,
    "MSIS": 709.9771493616975,
    "QuantileLoss[0.5]": 2678.774642944336,
    "Coverage[0.5]": 1.0,
    "RMSE": 228.69914052090357,
    "NRMSE": 4.814718747808496,
    "ND": 4.699604235197368,
    "wQuantileLoss[0.5]": 4.699604636744449,
    "mean_wQuantileLoss": 4.699604636744449,
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
100%|██████████| 10/10 [00:00<00:00, 55.79it/s, avg_epoch_loss=5.18]
INFO:root:Epoch[0] Elapsed time 0.180 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.182149
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.182148790359497 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2efc030630>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2efc030630>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 158.95it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 517.9512532552084,
    "abs_error": 190.4029083251953,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2616001164982014,
    "sMAPE": 0.31977413481832845,
    "MSIS": 50.464003851096294,
    "QuantileLoss[0.5]": 190.40290451049805,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 22.758542423784707,
    "NRMSE": 0.4791272089217833,
    "ND": 0.3340401900442023,
    "wQuantileLoss[0.5]": 0.33404018335175095,
    "mean_wQuantileLoss": 0.33404018335175095,
    "MAE_Coverage": 0.16666666666666663
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
100%|██████████| 10/10 [00:01<00:00,  9.71it/s, avg_epoch_loss=160]
INFO:root:Epoch[0] Elapsed time 1.031 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=160.137221
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 160.13722096982082 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2efc11f828>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2efc11f828>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 149.60it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 721.6386473701223,
    "abs_error": 263.04314901123786,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.7429107063314655,
    "sMAPE": 0.5744170818242088,
    "MSIS": 69.71642825325863,
    "QuantileLoss[0.5]": 263.04314901123786,
    "Coverage[0.5]": 0.08333333333333333,
    "RMSE": 26.86333276736381,
    "NRMSE": 0.5655438477339749,
    "ND": 0.4614792087916454,
    "wQuantileLoss[0.5]": 0.4614792087916454,
    "mean_wQuantileLoss": 0.4614792087916454,
    "MAE_Coverage": 0.4166666666666667
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
 10%|█         | 1/10 [01:53<17:05, 113.98s/it, avg_epoch_loss=0.653] 20%|██        | 2/10 [04:37<17:10, 128.85s/it, avg_epoch_loss=0.635] 30%|███       | 3/10 [07:59<17:34, 150.70s/it, avg_epoch_loss=0.618] 40%|████      | 4/10 [10:50<15:42, 157.01s/it, avg_epoch_loss=0.6]   50%|█████     | 5/10 [14:02<13:57, 167.41s/it, avg_epoch_loss=0.582] 60%|██████    | 6/10 [17:11<11:35, 173.79s/it, avg_epoch_loss=0.565] 70%|███████   | 7/10 [20:09<08:45, 175.09s/it, avg_epoch_loss=0.548] 80%|████████  | 8/10 [23:22<06:00, 180.39s/it, avg_epoch_loss=0.531] 90%|█████████ | 9/10 [26:12<02:57, 177.50s/it, avg_epoch_loss=0.516]100%|██████████| 10/10 [29:27<00:00, 182.74s/it, avg_epoch_loss=0.502]100%|██████████| 10/10 [29:27<00:00, 176.79s/it, avg_epoch_loss=0.502]
INFO:root:Epoch[0] Elapsed time 1767.946 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.501657
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5016570329666138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2eedaf6128>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2eedaf6128>

  #### Save the trained model  ###################################### 
WARNING:root:Serializing RepresentableBlockPredictor instances does not save the prediction network structure in a backwards-compatible manner. Be careful not to use this method in production.

  ['version.json', 'glutonts_model_pars.pkl', 'prediction_net-network.json', 'prediction_net-0000.params', 'parameters.json', 'type.txt', 'input_transform.json'] 

  #### Load the trained model  ###################################### 
INFO:root:Using CPU
INFO:root:Using CPU

  #### Predict   #################################################### 
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}

  #### metrics   #################################################### 
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 18.59it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 143.64437866210938,
    "abs_error": 113.09156799316406,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.7493390547977266,
    "sMAPE": 0.20097564209453803,
    "MSIS": 29.97356785373142,
    "QuantileLoss[0.5]": 113.0915756225586,
    "Coverage[0.5]": 0.4166666666666667,
    "RMSE": 11.985173284609171,
    "NRMSE": 0.2523194375707194,
    "ND": 0.19840625963712993,
    "wQuantileLoss[0.5]": 0.19840627302203262,
    "mean_wQuantileLoss": 0.19840627302203262,
    "MAE_Coverage": 0.08333333333333331
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
   eee2ae8..4d4375e  master     -> origin/master
Updating eee2ae8..4d4375e
Fast-forward
 error_list/20200521/list_log_testall_20200521.md | 7 +++++++
 1 file changed, 7 insertions(+)
