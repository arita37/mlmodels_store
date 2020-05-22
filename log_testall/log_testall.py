
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
Already up to date.
[master 6f871b1] ml_store
 2 files changed, 63 insertions(+), 8284 deletions(-)
 rewrite log_testall/log_testall.py (99%)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   f8a53f1..6f871b1  master -> master





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
[master 5cd651e] ml_store
 1 file changed, 50 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   6f871b1..5cd651e  master -> master





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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-22 20:13:03.955661: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 20:13:03.960980: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-22 20:13:03.961158: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558680a31b80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 20:13:03.961174: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 228
Trainable params: 228
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2501 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.24983123516735145}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
Total params: 228
Trainable params: 228
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 498
Trainable params: 498
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2504 - binary_crossentropy: 0.6939500/500 [==============================] - 1s 2ms/sample - loss: 0.2576 - binary_crossentropy: 0.7086 - val_loss: 0.2502 - val_binary_crossentropy: 0.6937

  #### metrics   #################################################### 
{'MSE': 0.25351560043634475}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 498
Trainable params: 498
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 662
Trainable params: 662
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2501 - binary_crossentropy: 0.6934 - val_loss: 0.2498 - val_binary_crossentropy: 0.6928

  #### metrics   #################################################### 
{'MSE': 0.24980249610697003}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 662
Trainable params: 662
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
Total params: 433
Trainable params: 433
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3211 - binary_crossentropy: 0.8604500/500 [==============================] - 1s 3ms/sample - loss: 0.2864 - binary_crossentropy: 0.7816 - val_loss: 0.2868 - val_binary_crossentropy: 0.7796

  #### metrics   #################################################### 
{'MSE': 0.2836242619059101}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
Total params: 433
Trainable params: 433
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2630 - binary_crossentropy: 0.9808500/500 [==============================] - 2s 3ms/sample - loss: 0.2846 - binary_crossentropy: 1.0531 - val_loss: 0.2840 - val_binary_crossentropy: 1.2831

  #### metrics   #################################################### 
{'MSE': 0.2806461244704381}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-22 20:14:25.033380: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:25.035356: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:25.041680: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 20:14:25.051234: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 20:14:25.052884: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 20:14:25.054530: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:25.056531: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2463 - val_binary_crossentropy: 0.6856
2020-05-22 20:14:26.294082: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:26.295749: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:26.299574: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 20:14:26.307959: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-22 20:14:26.309537: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 20:14:26.310765: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:26.311985: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2446039779250017}

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
2020-05-22 20:14:49.799214: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:49.800534: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:49.803987: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 20:14:49.810071: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 20:14:49.811106: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 20:14:49.812046: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:49.812955: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2511 - val_binary_crossentropy: 0.6954
2020-05-22 20:14:51.387039: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:51.388427: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:51.391753: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 20:14:51.397200: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-22 20:14:51.398194: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 20:14:51.399364: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:14:51.400339: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2513936081852757}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-22 20:15:25.854116: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:15:25.858786: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:15:25.873395: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 20:15:25.898443: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 20:15:25.902747: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 20:15:25.906704: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:15:25.910549: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.5413 - binary_crossentropy: 1.3308 - val_loss: 0.2504 - val_binary_crossentropy: 0.6940
2020-05-22 20:15:28.252253: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:15:28.256530: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:15:28.268625: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 20:15:28.294417: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-22 20:15:28.298587: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-22 20:15:28.302058: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-22 20:15:28.305719: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25738705797529676}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 660
Trainable params: 660
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2796 - binary_crossentropy: 0.8886500/500 [==============================] - 4s 9ms/sample - loss: 0.2682 - binary_crossentropy: 0.7581 - val_loss: 0.2613 - val_binary_crossentropy: 0.7179

  #### metrics   #################################################### 
{'MSE': 0.26425570921194275}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 660
Trainable params: 660
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         14          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         18          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
Total params: 302
Trainable params: 302
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2595 - binary_crossentropy: 0.7134500/500 [==============================] - 5s 9ms/sample - loss: 0.2705 - binary_crossentropy: 0.7363 - val_loss: 0.2549 - val_binary_crossentropy: 0.7032

  #### metrics   #################################################### 
{'MSE': 0.2545706263957645}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         14          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         18          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
Total params: 302
Trainable params: 302
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 6s - loss: 0.2983 - binary_crossentropy: 0.8131500/500 [==============================] - 5s 9ms/sample - loss: 0.3093 - binary_crossentropy: 0.8426 - val_loss: 0.2772 - val_binary_crossentropy: 0.7606

  #### metrics   #################################################### 
{'MSE': 0.2887788497975062}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
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
regionsequence_mean (InputLayer [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 1)]          0                                            
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
region_10sparse_seq_emb_regions (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
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
Total params: 112
Trainable params: 112
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2962 - binary_crossentropy: 1.9570500/500 [==============================] - 6s 12ms/sample - loss: 0.3121 - binary_crossentropy: 2.2048 - val_loss: 0.2834 - val_binary_crossentropy: 1.7201

  #### metrics   #################################################### 
{'MSE': 0.29753832840618727}

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
regionsequence_mean (InputLayer [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 1)]          0                                            
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
region_10sparse_seq_emb_regions (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         6           regionsequence_max[0][0]         
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
Total params: 112
Trainable params: 112
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 1,377
Trainable params: 1,377
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.5000 - binary_crossentropy: 7.7125500/500 [==============================] - 6s 12ms/sample - loss: 0.5060 - binary_crossentropy: 7.8050 - val_loss: 0.5080 - val_binary_crossentropy: 7.8359

  #### metrics   #################################################### 
{'MSE': 0.507}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 1,377
Trainable params: 1,377
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
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
Total params: 3,137
Trainable params: 3,057
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.5800 - binary_crossentropy: 8.9465500/500 [==============================] - 7s 13ms/sample - loss: 0.5020 - binary_crossentropy: 7.7433 - val_loss: 0.5040 - val_binary_crossentropy: 7.7742

  #### metrics   #################################################### 
{'MSE': 0.503}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         16          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
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
Total params: 3,137
Trainable params: 3,057
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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
Already up to date.
[master df2c761] ml_store
 1 file changed, 4949 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   5cd651e..df2c761  master -> master





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
Already up to date.
[master 48c2d60] ml_store
 1 file changed, 52 insertions(+)
To github.com:arita37/mlmodels_store.git
   df2c761..48c2d60  master -> master





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
[master 5e3f1ef] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   48c2d60..5e3f1ef  master -> master





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
[master 0d3aef9] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   5e3f1ef..0d3aef9  master -> master





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

2020-05-22 20:24:52.367039: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 20:24:52.374487: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-22 20:24:52.374699: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5648ffe20430 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 20:24:52.374715: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3931
256/354 [====================>.........] - ETA: 3s - loss: 1.3317
354/354 [==============================] - 15s 43ms/step - loss: 1.1431 - val_loss: 2.5816

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
[master 03c2887] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   0d3aef9..03c2887  master -> master





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
[master 26fbc58] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   03c2887..26fbc58  master -> master





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
[master e9f090c] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   26fbc58..e9f090c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2818048/17464789 [===>..........................] - ETA: 0s
 8962048/17464789 [==============>...............] - ETA: 0s
15286272/17464789 [=========================>....] - ETA: 0s
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
2020-05-22 20:25:55.002713: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-22 20:25:55.006582: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-22 20:25:55.006725: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cb2e1773d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-22 20:25:55.006739: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8583 - accuracy: 0.4875 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8506 - accuracy: 0.4880
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7931 - accuracy: 0.4918
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7617 - accuracy: 0.4938
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7101 - accuracy: 0.4972
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7323 - accuracy: 0.4957
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7184 - accuracy: 0.4966
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7058 - accuracy: 0.4974
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6789 - accuracy: 0.4992
11000/25000 [============>.................] - ETA: 4s - loss: 7.6931 - accuracy: 0.4983
12000/25000 [=============>................] - ETA: 4s - loss: 7.6947 - accuracy: 0.4982
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6643 - accuracy: 0.5002
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6458 - accuracy: 0.5014
15000/25000 [=================>............] - ETA: 3s - loss: 7.6339 - accuracy: 0.5021
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6459 - accuracy: 0.5014
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6538 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6473 - accuracy: 0.5013
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6484 - accuracy: 0.5012
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6659 - accuracy: 0.5000
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
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
(<mlmodels.util.Model_empty object at 0x7f338928e1d0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f338c8c1470> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8430 - accuracy: 0.4885 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6615 - accuracy: 0.5003
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5976 - accuracy: 0.5045
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6329 - accuracy: 0.5022
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6334 - accuracy: 0.5022
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6009 - accuracy: 0.5043
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5650 - accuracy: 0.5066
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5474 - accuracy: 0.5078
10000/25000 [===========>..................] - ETA: 4s - loss: 7.5578 - accuracy: 0.5071
11000/25000 [============>.................] - ETA: 4s - loss: 7.5593 - accuracy: 0.5070
12000/25000 [=============>................] - ETA: 4s - loss: 7.5299 - accuracy: 0.5089
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5640 - accuracy: 0.5067
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5889 - accuracy: 0.5051
15000/25000 [=================>............] - ETA: 3s - loss: 7.6216 - accuracy: 0.5029
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6407 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6487 - accuracy: 0.5012
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6618 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6482 - accuracy: 0.5012
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6806 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6760 - accuracy: 0.4994
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6647 - accuracy: 0.5001
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5593 - accuracy: 0.5070
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6513 - accuracy: 0.5010 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5644 - accuracy: 0.5067
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6130 - accuracy: 0.5035
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6421 - accuracy: 0.5016
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6206 - accuracy: 0.5030
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6338 - accuracy: 0.5021
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6187 - accuracy: 0.5031
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6479 - accuracy: 0.5012
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6651 - accuracy: 0.5001
11000/25000 [============>.................] - ETA: 4s - loss: 7.6931 - accuracy: 0.4983
12000/25000 [=============>................] - ETA: 4s - loss: 7.6832 - accuracy: 0.4989
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6536 - accuracy: 0.5008
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6765 - accuracy: 0.4994
15000/25000 [=================>............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6877 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7207 - accuracy: 0.4965
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7254 - accuracy: 0.4962
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7134 - accuracy: 0.4969
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6914 - accuracy: 0.4984
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6910 - accuracy: 0.4984
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6786 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6788 - accuracy: 0.4992
25000/25000 [==============================] - 9s 380us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Already up to date.
[master e4fc1c6] ml_store
 1 file changed, 318 insertions(+)
To github.com:arita37/mlmodels_store.git
   e9f090c..e4fc1c6  master -> master





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

13/13 [==============================] - 2s 122ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 3849fef] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   e4fc1c6..3849fef  master -> master





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
 2547712/11490434 [=====>........................] - ETA: 0s
 9674752/11490434 [========================>.....] - ETA: 0s
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

   32/60000 [..............................] - ETA: 6:49 - loss: 2.3101 - categorical_accuracy: 0.0312
   64/60000 [..............................] - ETA: 4:24 - loss: 2.2872 - categorical_accuracy: 0.1406
   96/60000 [..............................] - ETA: 3:32 - loss: 2.2630 - categorical_accuracy: 0.1667
  128/60000 [..............................] - ETA: 3:06 - loss: 2.2533 - categorical_accuracy: 0.1719
  160/60000 [..............................] - ETA: 2:53 - loss: 2.2203 - categorical_accuracy: 0.1937
  192/60000 [..............................] - ETA: 2:42 - loss: 2.1851 - categorical_accuracy: 0.1979
  224/60000 [..............................] - ETA: 2:34 - loss: 2.1512 - categorical_accuracy: 0.2143
  256/60000 [..............................] - ETA: 2:27 - loss: 2.1329 - categorical_accuracy: 0.2266
  288/60000 [..............................] - ETA: 2:22 - loss: 2.0883 - categorical_accuracy: 0.2465
  320/60000 [..............................] - ETA: 2:19 - loss: 2.0491 - categorical_accuracy: 0.2594
  352/60000 [..............................] - ETA: 2:16 - loss: 2.0095 - categorical_accuracy: 0.2898
  384/60000 [..............................] - ETA: 2:13 - loss: 1.9813 - categorical_accuracy: 0.2969
  416/60000 [..............................] - ETA: 2:12 - loss: 1.9523 - categorical_accuracy: 0.3173
  448/60000 [..............................] - ETA: 2:11 - loss: 1.9081 - categorical_accuracy: 0.3326
  480/60000 [..............................] - ETA: 2:09 - loss: 1.8874 - categorical_accuracy: 0.3500
  512/60000 [..............................] - ETA: 2:08 - loss: 1.8963 - categorical_accuracy: 0.3535
  544/60000 [..............................] - ETA: 2:07 - loss: 1.8777 - categorical_accuracy: 0.3621
  576/60000 [..............................] - ETA: 2:06 - loss: 1.8440 - categorical_accuracy: 0.3733
  608/60000 [..............................] - ETA: 2:05 - loss: 1.8084 - categorical_accuracy: 0.3882
  640/60000 [..............................] - ETA: 2:04 - loss: 1.7669 - categorical_accuracy: 0.4031
  672/60000 [..............................] - ETA: 2:04 - loss: 1.7345 - categorical_accuracy: 0.4092
  704/60000 [..............................] - ETA: 2:03 - loss: 1.7437 - categorical_accuracy: 0.4077
  736/60000 [..............................] - ETA: 2:02 - loss: 1.7248 - categorical_accuracy: 0.4103
  768/60000 [..............................] - ETA: 2:02 - loss: 1.7172 - categorical_accuracy: 0.4128
  800/60000 [..............................] - ETA: 2:02 - loss: 1.6941 - categorical_accuracy: 0.4187
  832/60000 [..............................] - ETA: 2:01 - loss: 1.6717 - categorical_accuracy: 0.4267
  864/60000 [..............................] - ETA: 2:01 - loss: 1.6414 - categorical_accuracy: 0.4387
  896/60000 [..............................] - ETA: 2:00 - loss: 1.6173 - categorical_accuracy: 0.4442
  928/60000 [..............................] - ETA: 2:00 - loss: 1.5983 - categorical_accuracy: 0.4537
  960/60000 [..............................] - ETA: 2:00 - loss: 1.5813 - categorical_accuracy: 0.4625
  992/60000 [..............................] - ETA: 1:59 - loss: 1.5582 - categorical_accuracy: 0.4708
 1024/60000 [..............................] - ETA: 1:59 - loss: 1.5389 - categorical_accuracy: 0.4785
 1056/60000 [..............................] - ETA: 1:59 - loss: 1.5133 - categorical_accuracy: 0.4877
 1088/60000 [..............................] - ETA: 1:58 - loss: 1.4870 - categorical_accuracy: 0.4982
 1120/60000 [..............................] - ETA: 1:58 - loss: 1.4630 - categorical_accuracy: 0.5063
 1152/60000 [..............................] - ETA: 1:58 - loss: 1.4442 - categorical_accuracy: 0.5130
 1184/60000 [..............................] - ETA: 1:57 - loss: 1.4235 - categorical_accuracy: 0.5211
 1216/60000 [..............................] - ETA: 1:57 - loss: 1.3967 - categorical_accuracy: 0.5296
 1248/60000 [..............................] - ETA: 1:57 - loss: 1.3852 - categorical_accuracy: 0.5353
 1280/60000 [..............................] - ETA: 1:56 - loss: 1.3726 - categorical_accuracy: 0.5391
 1312/60000 [..............................] - ETA: 1:56 - loss: 1.3557 - categorical_accuracy: 0.5457
 1344/60000 [..............................] - ETA: 1:56 - loss: 1.3380 - categorical_accuracy: 0.5506
 1376/60000 [..............................] - ETA: 1:56 - loss: 1.3196 - categorical_accuracy: 0.5574
 1408/60000 [..............................] - ETA: 1:55 - loss: 1.3031 - categorical_accuracy: 0.5653
 1440/60000 [..............................] - ETA: 1:55 - loss: 1.2853 - categorical_accuracy: 0.5715
 1472/60000 [..............................] - ETA: 1:55 - loss: 1.2925 - categorical_accuracy: 0.5747
 1504/60000 [..............................] - ETA: 1:55 - loss: 1.2832 - categorical_accuracy: 0.5785
 1536/60000 [..............................] - ETA: 1:54 - loss: 1.2726 - categorical_accuracy: 0.5833
 1568/60000 [..............................] - ETA: 1:54 - loss: 1.2643 - categorical_accuracy: 0.5867
 1600/60000 [..............................] - ETA: 1:54 - loss: 1.2534 - categorical_accuracy: 0.5919
 1632/60000 [..............................] - ETA: 1:54 - loss: 1.2408 - categorical_accuracy: 0.5968
 1664/60000 [..............................] - ETA: 1:54 - loss: 1.2223 - categorical_accuracy: 0.6034
 1696/60000 [..............................] - ETA: 1:54 - loss: 1.2181 - categorical_accuracy: 0.6061
 1728/60000 [..............................] - ETA: 1:54 - loss: 1.2055 - categorical_accuracy: 0.6105
 1760/60000 [..............................] - ETA: 1:54 - loss: 1.1974 - categorical_accuracy: 0.6142
 1792/60000 [..............................] - ETA: 1:53 - loss: 1.1856 - categorical_accuracy: 0.6194
 1824/60000 [..............................] - ETA: 1:53 - loss: 1.1754 - categorical_accuracy: 0.6234
 1856/60000 [..............................] - ETA: 1:53 - loss: 1.1660 - categorical_accuracy: 0.6266
 1888/60000 [..............................] - ETA: 1:53 - loss: 1.1535 - categorical_accuracy: 0.6314
 1920/60000 [..............................] - ETA: 1:53 - loss: 1.1433 - categorical_accuracy: 0.6333
 1952/60000 [..............................] - ETA: 1:53 - loss: 1.1336 - categorical_accuracy: 0.6373
 1984/60000 [..............................] - ETA: 1:53 - loss: 1.1237 - categorical_accuracy: 0.6416
 2016/60000 [>.............................] - ETA: 1:52 - loss: 1.1159 - categorical_accuracy: 0.6434
 2048/60000 [>.............................] - ETA: 1:52 - loss: 1.1051 - categorical_accuracy: 0.6465
 2080/60000 [>.............................] - ETA: 1:52 - loss: 1.0924 - categorical_accuracy: 0.6510
 2112/60000 [>.............................] - ETA: 1:52 - loss: 1.0813 - categorical_accuracy: 0.6548
 2144/60000 [>.............................] - ETA: 1:51 - loss: 1.0710 - categorical_accuracy: 0.6576
 2176/60000 [>.............................] - ETA: 1:51 - loss: 1.0639 - categorical_accuracy: 0.6604
 2208/60000 [>.............................] - ETA: 1:51 - loss: 1.0569 - categorical_accuracy: 0.6639
 2240/60000 [>.............................] - ETA: 1:51 - loss: 1.0487 - categorical_accuracy: 0.6656
 2272/60000 [>.............................] - ETA: 1:51 - loss: 1.0385 - categorical_accuracy: 0.6686
 2304/60000 [>.............................] - ETA: 1:51 - loss: 1.0274 - categorical_accuracy: 0.6719
 2336/60000 [>.............................] - ETA: 1:50 - loss: 1.0172 - categorical_accuracy: 0.6751
 2368/60000 [>.............................] - ETA: 1:50 - loss: 1.0092 - categorical_accuracy: 0.6774
 2400/60000 [>.............................] - ETA: 1:50 - loss: 1.0028 - categorical_accuracy: 0.6792
 2432/60000 [>.............................] - ETA: 1:50 - loss: 0.9976 - categorical_accuracy: 0.6822
 2464/60000 [>.............................] - ETA: 1:50 - loss: 0.9890 - categorical_accuracy: 0.6851
 2496/60000 [>.............................] - ETA: 1:50 - loss: 0.9824 - categorical_accuracy: 0.6875
 2528/60000 [>.............................] - ETA: 1:50 - loss: 0.9746 - categorical_accuracy: 0.6899
 2560/60000 [>.............................] - ETA: 1:50 - loss: 0.9685 - categorical_accuracy: 0.6918
 2592/60000 [>.............................] - ETA: 1:49 - loss: 0.9611 - categorical_accuracy: 0.6941
 2624/60000 [>.............................] - ETA: 1:49 - loss: 0.9578 - categorical_accuracy: 0.6947
 2656/60000 [>.............................] - ETA: 1:49 - loss: 0.9529 - categorical_accuracy: 0.6958
 2688/60000 [>.............................] - ETA: 1:49 - loss: 0.9575 - categorical_accuracy: 0.6961
 2720/60000 [>.............................] - ETA: 1:49 - loss: 0.9494 - categorical_accuracy: 0.6989
 2752/60000 [>.............................] - ETA: 1:49 - loss: 0.9411 - categorical_accuracy: 0.7017
 2784/60000 [>.............................] - ETA: 1:49 - loss: 0.9401 - categorical_accuracy: 0.7026
 2816/60000 [>.............................] - ETA: 1:48 - loss: 0.9382 - categorical_accuracy: 0.7024
 2848/60000 [>.............................] - ETA: 1:48 - loss: 0.9304 - categorical_accuracy: 0.7051
 2880/60000 [>.............................] - ETA: 1:48 - loss: 0.9272 - categorical_accuracy: 0.7069
 2912/60000 [>.............................] - ETA: 1:48 - loss: 0.9214 - categorical_accuracy: 0.7088
 2944/60000 [>.............................] - ETA: 1:48 - loss: 0.9160 - categorical_accuracy: 0.7106
 2976/60000 [>.............................] - ETA: 1:48 - loss: 0.9128 - categorical_accuracy: 0.7110
 3008/60000 [>.............................] - ETA: 1:47 - loss: 0.9079 - categorical_accuracy: 0.7124
 3040/60000 [>.............................] - ETA: 1:47 - loss: 0.9028 - categorical_accuracy: 0.7148
 3072/60000 [>.............................] - ETA: 1:47 - loss: 0.8997 - categorical_accuracy: 0.7155
 3104/60000 [>.............................] - ETA: 1:47 - loss: 0.8939 - categorical_accuracy: 0.7178
 3136/60000 [>.............................] - ETA: 1:47 - loss: 0.8901 - categorical_accuracy: 0.7188
 3168/60000 [>.............................] - ETA: 1:47 - loss: 0.8821 - categorical_accuracy: 0.7216
 3200/60000 [>.............................] - ETA: 1:47 - loss: 0.8747 - categorical_accuracy: 0.7241
 3232/60000 [>.............................] - ETA: 1:46 - loss: 0.8688 - categorical_accuracy: 0.7259
 3264/60000 [>.............................] - ETA: 1:46 - loss: 0.8633 - categorical_accuracy: 0.7273
 3296/60000 [>.............................] - ETA: 1:46 - loss: 0.8590 - categorical_accuracy: 0.7288
 3328/60000 [>.............................] - ETA: 1:46 - loss: 0.8522 - categorical_accuracy: 0.7311
 3360/60000 [>.............................] - ETA: 1:46 - loss: 0.8454 - categorical_accuracy: 0.7330
 3392/60000 [>.............................] - ETA: 1:46 - loss: 0.8439 - categorical_accuracy: 0.7341
 3424/60000 [>.............................] - ETA: 1:46 - loss: 0.8395 - categorical_accuracy: 0.7354
 3456/60000 [>.............................] - ETA: 1:46 - loss: 0.8345 - categorical_accuracy: 0.7370
 3488/60000 [>.............................] - ETA: 1:46 - loss: 0.8314 - categorical_accuracy: 0.7377
 3520/60000 [>.............................] - ETA: 1:45 - loss: 0.8275 - categorical_accuracy: 0.7386
 3552/60000 [>.............................] - ETA: 1:45 - loss: 0.8216 - categorical_accuracy: 0.7407
 3584/60000 [>.............................] - ETA: 1:45 - loss: 0.8167 - categorical_accuracy: 0.7422
 3616/60000 [>.............................] - ETA: 1:45 - loss: 0.8120 - categorical_accuracy: 0.7436
 3648/60000 [>.............................] - ETA: 1:45 - loss: 0.8061 - categorical_accuracy: 0.7453
 3680/60000 [>.............................] - ETA: 1:45 - loss: 0.8025 - categorical_accuracy: 0.7467
 3712/60000 [>.............................] - ETA: 1:45 - loss: 0.8000 - categorical_accuracy: 0.7478
 3744/60000 [>.............................] - ETA: 1:45 - loss: 0.7969 - categorical_accuracy: 0.7484
 3776/60000 [>.............................] - ETA: 1:44 - loss: 0.8002 - categorical_accuracy: 0.7481
 3808/60000 [>.............................] - ETA: 1:44 - loss: 0.7995 - categorical_accuracy: 0.7484
 3840/60000 [>.............................] - ETA: 1:44 - loss: 0.7985 - categorical_accuracy: 0.7492
 3872/60000 [>.............................] - ETA: 1:44 - loss: 0.7943 - categorical_accuracy: 0.7505
 3904/60000 [>.............................] - ETA: 1:44 - loss: 0.7907 - categorical_accuracy: 0.7520
 3936/60000 [>.............................] - ETA: 1:44 - loss: 0.7855 - categorical_accuracy: 0.7541
 3968/60000 [>.............................] - ETA: 1:44 - loss: 0.7809 - categorical_accuracy: 0.7555
 4000/60000 [=>............................] - ETA: 1:44 - loss: 0.7764 - categorical_accuracy: 0.7570
 4032/60000 [=>............................] - ETA: 1:44 - loss: 0.7724 - categorical_accuracy: 0.7579
 4064/60000 [=>............................] - ETA: 1:43 - loss: 0.7692 - categorical_accuracy: 0.7589
 4096/60000 [=>............................] - ETA: 1:43 - loss: 0.7640 - categorical_accuracy: 0.7607
 4128/60000 [=>............................] - ETA: 1:43 - loss: 0.7595 - categorical_accuracy: 0.7624
 4160/60000 [=>............................] - ETA: 1:43 - loss: 0.7553 - categorical_accuracy: 0.7637
 4192/60000 [=>............................] - ETA: 1:43 - loss: 0.7524 - categorical_accuracy: 0.7648
 4224/60000 [=>............................] - ETA: 1:43 - loss: 0.7487 - categorical_accuracy: 0.7659
 4256/60000 [=>............................] - ETA: 1:43 - loss: 0.7461 - categorical_accuracy: 0.7669
 4288/60000 [=>............................] - ETA: 1:43 - loss: 0.7434 - categorical_accuracy: 0.7680
 4320/60000 [=>............................] - ETA: 1:43 - loss: 0.7394 - categorical_accuracy: 0.7690
 4352/60000 [=>............................] - ETA: 1:43 - loss: 0.7426 - categorical_accuracy: 0.7688
 4384/60000 [=>............................] - ETA: 1:43 - loss: 0.7388 - categorical_accuracy: 0.7698
 4416/60000 [=>............................] - ETA: 1:43 - loss: 0.7355 - categorical_accuracy: 0.7706
 4448/60000 [=>............................] - ETA: 1:43 - loss: 0.7341 - categorical_accuracy: 0.7709
 4480/60000 [=>............................] - ETA: 1:43 - loss: 0.7308 - categorical_accuracy: 0.7721
 4512/60000 [=>............................] - ETA: 1:42 - loss: 0.7266 - categorical_accuracy: 0.7737
 4544/60000 [=>............................] - ETA: 1:42 - loss: 0.7232 - categorical_accuracy: 0.7746
 4576/60000 [=>............................] - ETA: 1:42 - loss: 0.7204 - categorical_accuracy: 0.7753
 4608/60000 [=>............................] - ETA: 1:42 - loss: 0.7180 - categorical_accuracy: 0.7758
 4640/60000 [=>............................] - ETA: 1:42 - loss: 0.7167 - categorical_accuracy: 0.7765
 4672/60000 [=>............................] - ETA: 1:42 - loss: 0.7128 - categorical_accuracy: 0.7780
 4704/60000 [=>............................] - ETA: 1:42 - loss: 0.7091 - categorical_accuracy: 0.7791
 4736/60000 [=>............................] - ETA: 1:42 - loss: 0.7063 - categorical_accuracy: 0.7798
 4768/60000 [=>............................] - ETA: 1:42 - loss: 0.7024 - categorical_accuracy: 0.7812
 4800/60000 [=>............................] - ETA: 1:42 - loss: 0.6992 - categorical_accuracy: 0.7823
 4832/60000 [=>............................] - ETA: 1:41 - loss: 0.6959 - categorical_accuracy: 0.7833
 4864/60000 [=>............................] - ETA: 1:41 - loss: 0.6925 - categorical_accuracy: 0.7843
 4896/60000 [=>............................] - ETA: 1:41 - loss: 0.6917 - categorical_accuracy: 0.7843
 4928/60000 [=>............................] - ETA: 1:41 - loss: 0.6900 - categorical_accuracy: 0.7849
 4960/60000 [=>............................] - ETA: 1:41 - loss: 0.6880 - categorical_accuracy: 0.7855
 4992/60000 [=>............................] - ETA: 1:41 - loss: 0.6856 - categorical_accuracy: 0.7859
 5024/60000 [=>............................] - ETA: 1:41 - loss: 0.6829 - categorical_accuracy: 0.7868
 5056/60000 [=>............................] - ETA: 1:41 - loss: 0.6799 - categorical_accuracy: 0.7880
 5088/60000 [=>............................] - ETA: 1:41 - loss: 0.6763 - categorical_accuracy: 0.7893
 5120/60000 [=>............................] - ETA: 1:41 - loss: 0.6767 - categorical_accuracy: 0.7898
 5152/60000 [=>............................] - ETA: 1:40 - loss: 0.6736 - categorical_accuracy: 0.7906
 5184/60000 [=>............................] - ETA: 1:40 - loss: 0.6713 - categorical_accuracy: 0.7913
 5216/60000 [=>............................] - ETA: 1:40 - loss: 0.6697 - categorical_accuracy: 0.7914
 5248/60000 [=>............................] - ETA: 1:40 - loss: 0.6688 - categorical_accuracy: 0.7917
 5280/60000 [=>............................] - ETA: 1:40 - loss: 0.6667 - categorical_accuracy: 0.7926
 5312/60000 [=>............................] - ETA: 1:40 - loss: 0.6636 - categorical_accuracy: 0.7937
 5344/60000 [=>............................] - ETA: 1:40 - loss: 0.6604 - categorical_accuracy: 0.7945
 5376/60000 [=>............................] - ETA: 1:40 - loss: 0.6601 - categorical_accuracy: 0.7950
 5408/60000 [=>............................] - ETA: 1:40 - loss: 0.6583 - categorical_accuracy: 0.7959
 5440/60000 [=>............................] - ETA: 1:40 - loss: 0.6564 - categorical_accuracy: 0.7965
 5472/60000 [=>............................] - ETA: 1:40 - loss: 0.6544 - categorical_accuracy: 0.7971
 5504/60000 [=>............................] - ETA: 1:40 - loss: 0.6513 - categorical_accuracy: 0.7980
 5536/60000 [=>............................] - ETA: 1:40 - loss: 0.6500 - categorical_accuracy: 0.7982
 5568/60000 [=>............................] - ETA: 1:40 - loss: 0.6469 - categorical_accuracy: 0.7994
 5600/60000 [=>............................] - ETA: 1:39 - loss: 0.6461 - categorical_accuracy: 0.8000
 5632/60000 [=>............................] - ETA: 1:39 - loss: 0.6455 - categorical_accuracy: 0.8001
 5664/60000 [=>............................] - ETA: 1:39 - loss: 0.6442 - categorical_accuracy: 0.8005
 5696/60000 [=>............................] - ETA: 1:39 - loss: 0.6423 - categorical_accuracy: 0.8009
 5728/60000 [=>............................] - ETA: 1:39 - loss: 0.6406 - categorical_accuracy: 0.8015
 5760/60000 [=>............................] - ETA: 1:39 - loss: 0.6380 - categorical_accuracy: 0.8023
 5792/60000 [=>............................] - ETA: 1:39 - loss: 0.6358 - categorical_accuracy: 0.8028
 5824/60000 [=>............................] - ETA: 1:39 - loss: 0.6330 - categorical_accuracy: 0.8036
 5856/60000 [=>............................] - ETA: 1:39 - loss: 0.6309 - categorical_accuracy: 0.8043
 5888/60000 [=>............................] - ETA: 1:39 - loss: 0.6307 - categorical_accuracy: 0.8045
 5920/60000 [=>............................] - ETA: 1:39 - loss: 0.6288 - categorical_accuracy: 0.8051
 5952/60000 [=>............................] - ETA: 1:39 - loss: 0.6258 - categorical_accuracy: 0.8061
 5984/60000 [=>............................] - ETA: 1:39 - loss: 0.6231 - categorical_accuracy: 0.8072
 6016/60000 [==>...........................] - ETA: 1:39 - loss: 0.6209 - categorical_accuracy: 0.8078
 6048/60000 [==>...........................] - ETA: 1:38 - loss: 0.6197 - categorical_accuracy: 0.8085
 6080/60000 [==>...........................] - ETA: 1:38 - loss: 0.6172 - categorical_accuracy: 0.8094
 6112/60000 [==>...........................] - ETA: 1:38 - loss: 0.6158 - categorical_accuracy: 0.8097
 6144/60000 [==>...........................] - ETA: 1:38 - loss: 0.6142 - categorical_accuracy: 0.8102
 6176/60000 [==>...........................] - ETA: 1:38 - loss: 0.6118 - categorical_accuracy: 0.8109
 6208/60000 [==>...........................] - ETA: 1:38 - loss: 0.6108 - categorical_accuracy: 0.8112
 6240/60000 [==>...........................] - ETA: 1:38 - loss: 0.6105 - categorical_accuracy: 0.8112
 6272/60000 [==>...........................] - ETA: 1:38 - loss: 0.6083 - categorical_accuracy: 0.8119
 6304/60000 [==>...........................] - ETA: 1:38 - loss: 0.6070 - categorical_accuracy: 0.8123
 6336/60000 [==>...........................] - ETA: 1:38 - loss: 0.6047 - categorical_accuracy: 0.8131
 6368/60000 [==>...........................] - ETA: 1:38 - loss: 0.6048 - categorical_accuracy: 0.8130
 6400/60000 [==>...........................] - ETA: 1:37 - loss: 0.6032 - categorical_accuracy: 0.8134
 6432/60000 [==>...........................] - ETA: 1:37 - loss: 0.6021 - categorical_accuracy: 0.8139
 6464/60000 [==>...........................] - ETA: 1:37 - loss: 0.6006 - categorical_accuracy: 0.8142
 6496/60000 [==>...........................] - ETA: 1:37 - loss: 0.6005 - categorical_accuracy: 0.8145
 6528/60000 [==>...........................] - ETA: 1:37 - loss: 0.5994 - categorical_accuracy: 0.8151
 6560/60000 [==>...........................] - ETA: 1:37 - loss: 0.5972 - categorical_accuracy: 0.8157
 6592/60000 [==>...........................] - ETA: 1:37 - loss: 0.5968 - categorical_accuracy: 0.8160
 6624/60000 [==>...........................] - ETA: 1:37 - loss: 0.5964 - categorical_accuracy: 0.8166
 6656/60000 [==>...........................] - ETA: 1:37 - loss: 0.5947 - categorical_accuracy: 0.8170
 6688/60000 [==>...........................] - ETA: 1:37 - loss: 0.5923 - categorical_accuracy: 0.8179
 6720/60000 [==>...........................] - ETA: 1:37 - loss: 0.5908 - categorical_accuracy: 0.8180
 6752/60000 [==>...........................] - ETA: 1:37 - loss: 0.5898 - categorical_accuracy: 0.8183
 6784/60000 [==>...........................] - ETA: 1:37 - loss: 0.5890 - categorical_accuracy: 0.8185
 6816/60000 [==>...........................] - ETA: 1:36 - loss: 0.5867 - categorical_accuracy: 0.8192
 6848/60000 [==>...........................] - ETA: 1:36 - loss: 0.5841 - categorical_accuracy: 0.8201
 6880/60000 [==>...........................] - ETA: 1:36 - loss: 0.5825 - categorical_accuracy: 0.8205
 6912/60000 [==>...........................] - ETA: 1:36 - loss: 0.5813 - categorical_accuracy: 0.8209
 6944/60000 [==>...........................] - ETA: 1:36 - loss: 0.5793 - categorical_accuracy: 0.8216
 6976/60000 [==>...........................] - ETA: 1:36 - loss: 0.5774 - categorical_accuracy: 0.8220
 7008/60000 [==>...........................] - ETA: 1:36 - loss: 0.5751 - categorical_accuracy: 0.8226
 7040/60000 [==>...........................] - ETA: 1:36 - loss: 0.5734 - categorical_accuracy: 0.8233
 7072/60000 [==>...........................] - ETA: 1:36 - loss: 0.5723 - categorical_accuracy: 0.8235
 7104/60000 [==>...........................] - ETA: 1:36 - loss: 0.5707 - categorical_accuracy: 0.8240
 7136/60000 [==>...........................] - ETA: 1:36 - loss: 0.5687 - categorical_accuracy: 0.8247
 7168/60000 [==>...........................] - ETA: 1:36 - loss: 0.5673 - categorical_accuracy: 0.8252
 7200/60000 [==>...........................] - ETA: 1:36 - loss: 0.5659 - categorical_accuracy: 0.8256
 7232/60000 [==>...........................] - ETA: 1:36 - loss: 0.5642 - categorical_accuracy: 0.8259
 7264/60000 [==>...........................] - ETA: 1:36 - loss: 0.5627 - categorical_accuracy: 0.8264
 7296/60000 [==>...........................] - ETA: 1:35 - loss: 0.5621 - categorical_accuracy: 0.8268
 7328/60000 [==>...........................] - ETA: 1:35 - loss: 0.5607 - categorical_accuracy: 0.8274
 7360/60000 [==>...........................] - ETA: 1:35 - loss: 0.5593 - categorical_accuracy: 0.8279
 7392/60000 [==>...........................] - ETA: 1:35 - loss: 0.5577 - categorical_accuracy: 0.8283
 7424/60000 [==>...........................] - ETA: 1:35 - loss: 0.5566 - categorical_accuracy: 0.8287
 7456/60000 [==>...........................] - ETA: 1:35 - loss: 0.5547 - categorical_accuracy: 0.8293
 7488/60000 [==>...........................] - ETA: 1:35 - loss: 0.5527 - categorical_accuracy: 0.8300
 7520/60000 [==>...........................] - ETA: 1:35 - loss: 0.5506 - categorical_accuracy: 0.8307
 7552/60000 [==>...........................] - ETA: 1:35 - loss: 0.5493 - categorical_accuracy: 0.8312
 7584/60000 [==>...........................] - ETA: 1:35 - loss: 0.5485 - categorical_accuracy: 0.8314
 7616/60000 [==>...........................] - ETA: 1:35 - loss: 0.5473 - categorical_accuracy: 0.8317
 7648/60000 [==>...........................] - ETA: 1:35 - loss: 0.5460 - categorical_accuracy: 0.8322
 7680/60000 [==>...........................] - ETA: 1:35 - loss: 0.5455 - categorical_accuracy: 0.8323
 7712/60000 [==>...........................] - ETA: 1:35 - loss: 0.5443 - categorical_accuracy: 0.8326
 7744/60000 [==>...........................] - ETA: 1:35 - loss: 0.5426 - categorical_accuracy: 0.8332
 7776/60000 [==>...........................] - ETA: 1:35 - loss: 0.5408 - categorical_accuracy: 0.8337
 7808/60000 [==>...........................] - ETA: 1:35 - loss: 0.5397 - categorical_accuracy: 0.8341
 7840/60000 [==>...........................] - ETA: 1:34 - loss: 0.5380 - categorical_accuracy: 0.8347
 7872/60000 [==>...........................] - ETA: 1:34 - loss: 0.5363 - categorical_accuracy: 0.8352
 7904/60000 [==>...........................] - ETA: 1:34 - loss: 0.5342 - categorical_accuracy: 0.8359
 7936/60000 [==>...........................] - ETA: 1:34 - loss: 0.5338 - categorical_accuracy: 0.8361
 7968/60000 [==>...........................] - ETA: 1:34 - loss: 0.5323 - categorical_accuracy: 0.8365
 8000/60000 [===>..........................] - ETA: 1:34 - loss: 0.5316 - categorical_accuracy: 0.8367
 8032/60000 [===>..........................] - ETA: 1:34 - loss: 0.5299 - categorical_accuracy: 0.8374
 8064/60000 [===>..........................] - ETA: 1:34 - loss: 0.5286 - categorical_accuracy: 0.8378
 8096/60000 [===>..........................] - ETA: 1:34 - loss: 0.5272 - categorical_accuracy: 0.8383
 8128/60000 [===>..........................] - ETA: 1:34 - loss: 0.5259 - categorical_accuracy: 0.8386
 8160/60000 [===>..........................] - ETA: 1:34 - loss: 0.5243 - categorical_accuracy: 0.8392
 8192/60000 [===>..........................] - ETA: 1:34 - loss: 0.5234 - categorical_accuracy: 0.8394
 8224/60000 [===>..........................] - ETA: 1:34 - loss: 0.5237 - categorical_accuracy: 0.8391
 8256/60000 [===>..........................] - ETA: 1:34 - loss: 0.5224 - categorical_accuracy: 0.8394
 8288/60000 [===>..........................] - ETA: 1:34 - loss: 0.5209 - categorical_accuracy: 0.8399
 8320/60000 [===>..........................] - ETA: 1:34 - loss: 0.5203 - categorical_accuracy: 0.8400
 8352/60000 [===>..........................] - ETA: 1:34 - loss: 0.5195 - categorical_accuracy: 0.8404
 8384/60000 [===>..........................] - ETA: 1:33 - loss: 0.5186 - categorical_accuracy: 0.8408
 8416/60000 [===>..........................] - ETA: 1:33 - loss: 0.5177 - categorical_accuracy: 0.8410
 8448/60000 [===>..........................] - ETA: 1:33 - loss: 0.5162 - categorical_accuracy: 0.8415
 8480/60000 [===>..........................] - ETA: 1:33 - loss: 0.5152 - categorical_accuracy: 0.8419
 8512/60000 [===>..........................] - ETA: 1:33 - loss: 0.5144 - categorical_accuracy: 0.8421
 8544/60000 [===>..........................] - ETA: 1:33 - loss: 0.5128 - categorical_accuracy: 0.8427
 8576/60000 [===>..........................] - ETA: 1:33 - loss: 0.5121 - categorical_accuracy: 0.8429
 8608/60000 [===>..........................] - ETA: 1:33 - loss: 0.5123 - categorical_accuracy: 0.8427
 8640/60000 [===>..........................] - ETA: 1:33 - loss: 0.5117 - categorical_accuracy: 0.8429
 8672/60000 [===>..........................] - ETA: 1:33 - loss: 0.5109 - categorical_accuracy: 0.8431
 8704/60000 [===>..........................] - ETA: 1:33 - loss: 0.5099 - categorical_accuracy: 0.8434
 8736/60000 [===>..........................] - ETA: 1:33 - loss: 0.5086 - categorical_accuracy: 0.8439
 8768/60000 [===>..........................] - ETA: 1:33 - loss: 0.5075 - categorical_accuracy: 0.8442
 8800/60000 [===>..........................] - ETA: 1:33 - loss: 0.5071 - categorical_accuracy: 0.8443
 8832/60000 [===>..........................] - ETA: 1:33 - loss: 0.5066 - categorical_accuracy: 0.8445
 8864/60000 [===>..........................] - ETA: 1:33 - loss: 0.5058 - categorical_accuracy: 0.8450
 8896/60000 [===>..........................] - ETA: 1:32 - loss: 0.5045 - categorical_accuracy: 0.8454
 8928/60000 [===>..........................] - ETA: 1:32 - loss: 0.5030 - categorical_accuracy: 0.8459
 8960/60000 [===>..........................] - ETA: 1:32 - loss: 0.5019 - categorical_accuracy: 0.8461
 8992/60000 [===>..........................] - ETA: 1:32 - loss: 0.5010 - categorical_accuracy: 0.8463
 9024/60000 [===>..........................] - ETA: 1:32 - loss: 0.4995 - categorical_accuracy: 0.8469
 9056/60000 [===>..........................] - ETA: 1:32 - loss: 0.4987 - categorical_accuracy: 0.8470
 9088/60000 [===>..........................] - ETA: 1:32 - loss: 0.4974 - categorical_accuracy: 0.8474
 9120/60000 [===>..........................] - ETA: 1:32 - loss: 0.4968 - categorical_accuracy: 0.8476
 9152/60000 [===>..........................] - ETA: 1:32 - loss: 0.4958 - categorical_accuracy: 0.8478
 9184/60000 [===>..........................] - ETA: 1:32 - loss: 0.4956 - categorical_accuracy: 0.8478
 9216/60000 [===>..........................] - ETA: 1:32 - loss: 0.4941 - categorical_accuracy: 0.8482
 9248/60000 [===>..........................] - ETA: 1:32 - loss: 0.4931 - categorical_accuracy: 0.8485
 9280/60000 [===>..........................] - ETA: 1:32 - loss: 0.4926 - categorical_accuracy: 0.8486
 9312/60000 [===>..........................] - ETA: 1:32 - loss: 0.4914 - categorical_accuracy: 0.8490
 9344/60000 [===>..........................] - ETA: 1:32 - loss: 0.4900 - categorical_accuracy: 0.8494
 9376/60000 [===>..........................] - ETA: 1:32 - loss: 0.4885 - categorical_accuracy: 0.8499
 9408/60000 [===>..........................] - ETA: 1:32 - loss: 0.4880 - categorical_accuracy: 0.8502
 9440/60000 [===>..........................] - ETA: 1:32 - loss: 0.4873 - categorical_accuracy: 0.8503
 9472/60000 [===>..........................] - ETA: 1:31 - loss: 0.4866 - categorical_accuracy: 0.8504
 9504/60000 [===>..........................] - ETA: 1:31 - loss: 0.4858 - categorical_accuracy: 0.8505
 9536/60000 [===>..........................] - ETA: 1:31 - loss: 0.4853 - categorical_accuracy: 0.8508
 9568/60000 [===>..........................] - ETA: 1:31 - loss: 0.4844 - categorical_accuracy: 0.8511
 9600/60000 [===>..........................] - ETA: 1:31 - loss: 0.4848 - categorical_accuracy: 0.8509
 9632/60000 [===>..........................] - ETA: 1:31 - loss: 0.4840 - categorical_accuracy: 0.8513
 9664/60000 [===>..........................] - ETA: 1:31 - loss: 0.4836 - categorical_accuracy: 0.8516
 9696/60000 [===>..........................] - ETA: 1:31 - loss: 0.4828 - categorical_accuracy: 0.8518
 9728/60000 [===>..........................] - ETA: 1:31 - loss: 0.4817 - categorical_accuracy: 0.8521
 9760/60000 [===>..........................] - ETA: 1:31 - loss: 0.4813 - categorical_accuracy: 0.8520
 9792/60000 [===>..........................] - ETA: 1:31 - loss: 0.4814 - categorical_accuracy: 0.8520
 9824/60000 [===>..........................] - ETA: 1:31 - loss: 0.4800 - categorical_accuracy: 0.8524
 9856/60000 [===>..........................] - ETA: 1:31 - loss: 0.4787 - categorical_accuracy: 0.8528
 9888/60000 [===>..........................] - ETA: 1:31 - loss: 0.4779 - categorical_accuracy: 0.8530
 9920/60000 [===>..........................] - ETA: 1:30 - loss: 0.4767 - categorical_accuracy: 0.8533
 9952/60000 [===>..........................] - ETA: 1:30 - loss: 0.4760 - categorical_accuracy: 0.8537
 9984/60000 [===>..........................] - ETA: 1:30 - loss: 0.4752 - categorical_accuracy: 0.8539
10016/60000 [====>.........................] - ETA: 1:30 - loss: 0.4743 - categorical_accuracy: 0.8541
10048/60000 [====>.........................] - ETA: 1:30 - loss: 0.4731 - categorical_accuracy: 0.8544
10080/60000 [====>.........................] - ETA: 1:30 - loss: 0.4720 - categorical_accuracy: 0.8548
10112/60000 [====>.........................] - ETA: 1:30 - loss: 0.4708 - categorical_accuracy: 0.8551
10144/60000 [====>.........................] - ETA: 1:30 - loss: 0.4695 - categorical_accuracy: 0.8556
10176/60000 [====>.........................] - ETA: 1:30 - loss: 0.4684 - categorical_accuracy: 0.8559
10208/60000 [====>.........................] - ETA: 1:30 - loss: 0.4673 - categorical_accuracy: 0.8563
10240/60000 [====>.........................] - ETA: 1:30 - loss: 0.4664 - categorical_accuracy: 0.8565
10272/60000 [====>.........................] - ETA: 1:30 - loss: 0.4664 - categorical_accuracy: 0.8566
10304/60000 [====>.........................] - ETA: 1:30 - loss: 0.4653 - categorical_accuracy: 0.8570
10336/60000 [====>.........................] - ETA: 1:30 - loss: 0.4657 - categorical_accuracy: 0.8572
10368/60000 [====>.........................] - ETA: 1:30 - loss: 0.4653 - categorical_accuracy: 0.8573
10400/60000 [====>.........................] - ETA: 1:30 - loss: 0.4650 - categorical_accuracy: 0.8574
10432/60000 [====>.........................] - ETA: 1:30 - loss: 0.4643 - categorical_accuracy: 0.8576
10464/60000 [====>.........................] - ETA: 1:30 - loss: 0.4639 - categorical_accuracy: 0.8576
10496/60000 [====>.........................] - ETA: 1:29 - loss: 0.4637 - categorical_accuracy: 0.8577
10528/60000 [====>.........................] - ETA: 1:29 - loss: 0.4627 - categorical_accuracy: 0.8580
10560/60000 [====>.........................] - ETA: 1:29 - loss: 0.4614 - categorical_accuracy: 0.8584
10592/60000 [====>.........................] - ETA: 1:29 - loss: 0.4614 - categorical_accuracy: 0.8583
10624/60000 [====>.........................] - ETA: 1:29 - loss: 0.4606 - categorical_accuracy: 0.8585
10656/60000 [====>.........................] - ETA: 1:29 - loss: 0.4597 - categorical_accuracy: 0.8587
10688/60000 [====>.........................] - ETA: 1:29 - loss: 0.4589 - categorical_accuracy: 0.8589
10720/60000 [====>.........................] - ETA: 1:29 - loss: 0.4579 - categorical_accuracy: 0.8591
10752/60000 [====>.........................] - ETA: 1:29 - loss: 0.4573 - categorical_accuracy: 0.8594
10784/60000 [====>.........................] - ETA: 1:29 - loss: 0.4572 - categorical_accuracy: 0.8594
10816/60000 [====>.........................] - ETA: 1:29 - loss: 0.4564 - categorical_accuracy: 0.8596
10848/60000 [====>.........................] - ETA: 1:29 - loss: 0.4556 - categorical_accuracy: 0.8598
10880/60000 [====>.........................] - ETA: 1:29 - loss: 0.4547 - categorical_accuracy: 0.8601
10912/60000 [====>.........................] - ETA: 1:29 - loss: 0.4534 - categorical_accuracy: 0.8605
10944/60000 [====>.........................] - ETA: 1:29 - loss: 0.4527 - categorical_accuracy: 0.8607
10976/60000 [====>.........................] - ETA: 1:29 - loss: 0.4519 - categorical_accuracy: 0.8608
11008/60000 [====>.........................] - ETA: 1:29 - loss: 0.4509 - categorical_accuracy: 0.8611
11040/60000 [====>.........................] - ETA: 1:28 - loss: 0.4499 - categorical_accuracy: 0.8614
11072/60000 [====>.........................] - ETA: 1:28 - loss: 0.4492 - categorical_accuracy: 0.8616
11104/60000 [====>.........................] - ETA: 1:28 - loss: 0.4491 - categorical_accuracy: 0.8618
11136/60000 [====>.........................] - ETA: 1:28 - loss: 0.4486 - categorical_accuracy: 0.8620
11168/60000 [====>.........................] - ETA: 1:28 - loss: 0.4473 - categorical_accuracy: 0.8624
11200/60000 [====>.........................] - ETA: 1:28 - loss: 0.4465 - categorical_accuracy: 0.8627
11232/60000 [====>.........................] - ETA: 1:28 - loss: 0.4456 - categorical_accuracy: 0.8629
11264/60000 [====>.........................] - ETA: 1:28 - loss: 0.4448 - categorical_accuracy: 0.8631
11296/60000 [====>.........................] - ETA: 1:28 - loss: 0.4438 - categorical_accuracy: 0.8634
11328/60000 [====>.........................] - ETA: 1:28 - loss: 0.4433 - categorical_accuracy: 0.8635
11360/60000 [====>.........................] - ETA: 1:28 - loss: 0.4422 - categorical_accuracy: 0.8639
11392/60000 [====>.........................] - ETA: 1:28 - loss: 0.4413 - categorical_accuracy: 0.8642
11424/60000 [====>.........................] - ETA: 1:28 - loss: 0.4403 - categorical_accuracy: 0.8645
11456/60000 [====>.........................] - ETA: 1:28 - loss: 0.4393 - categorical_accuracy: 0.8648
11488/60000 [====>.........................] - ETA: 1:28 - loss: 0.4383 - categorical_accuracy: 0.8650
11520/60000 [====>.........................] - ETA: 1:28 - loss: 0.4376 - categorical_accuracy: 0.8653
11552/60000 [====>.........................] - ETA: 1:27 - loss: 0.4368 - categorical_accuracy: 0.8656
11584/60000 [====>.........................] - ETA: 1:27 - loss: 0.4366 - categorical_accuracy: 0.8657
11616/60000 [====>.........................] - ETA: 1:27 - loss: 0.4356 - categorical_accuracy: 0.8659
11648/60000 [====>.........................] - ETA: 1:27 - loss: 0.4351 - categorical_accuracy: 0.8661
11680/60000 [====>.........................] - ETA: 1:27 - loss: 0.4343 - categorical_accuracy: 0.8664
11712/60000 [====>.........................] - ETA: 1:27 - loss: 0.4335 - categorical_accuracy: 0.8666
11744/60000 [====>.........................] - ETA: 1:27 - loss: 0.4330 - categorical_accuracy: 0.8668
11776/60000 [====>.........................] - ETA: 1:27 - loss: 0.4330 - categorical_accuracy: 0.8668
11808/60000 [====>.........................] - ETA: 1:27 - loss: 0.4322 - categorical_accuracy: 0.8670
11840/60000 [====>.........................] - ETA: 1:27 - loss: 0.4314 - categorical_accuracy: 0.8672
11872/60000 [====>.........................] - ETA: 1:27 - loss: 0.4303 - categorical_accuracy: 0.8676
11904/60000 [====>.........................] - ETA: 1:27 - loss: 0.4296 - categorical_accuracy: 0.8679
11936/60000 [====>.........................] - ETA: 1:27 - loss: 0.4293 - categorical_accuracy: 0.8680
11968/60000 [====>.........................] - ETA: 1:27 - loss: 0.4290 - categorical_accuracy: 0.8682
12000/60000 [=====>........................] - ETA: 1:27 - loss: 0.4290 - categorical_accuracy: 0.8683
12032/60000 [=====>........................] - ETA: 1:27 - loss: 0.4292 - categorical_accuracy: 0.8685
12064/60000 [=====>........................] - ETA: 1:27 - loss: 0.4286 - categorical_accuracy: 0.8687
12096/60000 [=====>........................] - ETA: 1:26 - loss: 0.4276 - categorical_accuracy: 0.8690
12128/60000 [=====>........................] - ETA: 1:26 - loss: 0.4272 - categorical_accuracy: 0.8691
12160/60000 [=====>........................] - ETA: 1:26 - loss: 0.4265 - categorical_accuracy: 0.8693
12192/60000 [=====>........................] - ETA: 1:26 - loss: 0.4255 - categorical_accuracy: 0.8697
12224/60000 [=====>........................] - ETA: 1:26 - loss: 0.4245 - categorical_accuracy: 0.8700
12256/60000 [=====>........................] - ETA: 1:26 - loss: 0.4235 - categorical_accuracy: 0.8703
12288/60000 [=====>........................] - ETA: 1:26 - loss: 0.4226 - categorical_accuracy: 0.8706
12320/60000 [=====>........................] - ETA: 1:26 - loss: 0.4217 - categorical_accuracy: 0.8709
12352/60000 [=====>........................] - ETA: 1:26 - loss: 0.4210 - categorical_accuracy: 0.8712
12384/60000 [=====>........................] - ETA: 1:26 - loss: 0.4204 - categorical_accuracy: 0.8714
12416/60000 [=====>........................] - ETA: 1:26 - loss: 0.4205 - categorical_accuracy: 0.8714
12448/60000 [=====>........................] - ETA: 1:26 - loss: 0.4199 - categorical_accuracy: 0.8715
12480/60000 [=====>........................] - ETA: 1:26 - loss: 0.4195 - categorical_accuracy: 0.8717
12512/60000 [=====>........................] - ETA: 1:26 - loss: 0.4192 - categorical_accuracy: 0.8718
12544/60000 [=====>........................] - ETA: 1:26 - loss: 0.4186 - categorical_accuracy: 0.8720
12576/60000 [=====>........................] - ETA: 1:26 - loss: 0.4182 - categorical_accuracy: 0.8721
12608/60000 [=====>........................] - ETA: 1:26 - loss: 0.4176 - categorical_accuracy: 0.8721
12640/60000 [=====>........................] - ETA: 1:25 - loss: 0.4171 - categorical_accuracy: 0.8723
12672/60000 [=====>........................] - ETA: 1:25 - loss: 0.4165 - categorical_accuracy: 0.8725
12704/60000 [=====>........................] - ETA: 1:25 - loss: 0.4160 - categorical_accuracy: 0.8726
12736/60000 [=====>........................] - ETA: 1:25 - loss: 0.4150 - categorical_accuracy: 0.8730
12768/60000 [=====>........................] - ETA: 1:25 - loss: 0.4147 - categorical_accuracy: 0.8730
12800/60000 [=====>........................] - ETA: 1:25 - loss: 0.4142 - categorical_accuracy: 0.8732
12832/60000 [=====>........................] - ETA: 1:25 - loss: 0.4142 - categorical_accuracy: 0.8734
12864/60000 [=====>........................] - ETA: 1:25 - loss: 0.4133 - categorical_accuracy: 0.8737
12896/60000 [=====>........................] - ETA: 1:25 - loss: 0.4130 - categorical_accuracy: 0.8738
12928/60000 [=====>........................] - ETA: 1:25 - loss: 0.4123 - categorical_accuracy: 0.8739
12960/60000 [=====>........................] - ETA: 1:25 - loss: 0.4124 - categorical_accuracy: 0.8739
12992/60000 [=====>........................] - ETA: 1:25 - loss: 0.4116 - categorical_accuracy: 0.8742
13024/60000 [=====>........................] - ETA: 1:25 - loss: 0.4110 - categorical_accuracy: 0.8744
13056/60000 [=====>........................] - ETA: 1:25 - loss: 0.4104 - categorical_accuracy: 0.8745
13088/60000 [=====>........................] - ETA: 1:25 - loss: 0.4098 - categorical_accuracy: 0.8747
13120/60000 [=====>........................] - ETA: 1:25 - loss: 0.4097 - categorical_accuracy: 0.8748
13152/60000 [=====>........................] - ETA: 1:25 - loss: 0.4091 - categorical_accuracy: 0.8749
13184/60000 [=====>........................] - ETA: 1:25 - loss: 0.4082 - categorical_accuracy: 0.8752
13216/60000 [=====>........................] - ETA: 1:25 - loss: 0.4075 - categorical_accuracy: 0.8755
13248/60000 [=====>........................] - ETA: 1:24 - loss: 0.4077 - categorical_accuracy: 0.8755
13280/60000 [=====>........................] - ETA: 1:24 - loss: 0.4080 - categorical_accuracy: 0.8756
13312/60000 [=====>........................] - ETA: 1:24 - loss: 0.4073 - categorical_accuracy: 0.8758
13344/60000 [=====>........................] - ETA: 1:24 - loss: 0.4069 - categorical_accuracy: 0.8759
13376/60000 [=====>........................] - ETA: 1:24 - loss: 0.4062 - categorical_accuracy: 0.8761
13408/60000 [=====>........................] - ETA: 1:24 - loss: 0.4058 - categorical_accuracy: 0.8762
13440/60000 [=====>........................] - ETA: 1:24 - loss: 0.4054 - categorical_accuracy: 0.8763
13472/60000 [=====>........................] - ETA: 1:24 - loss: 0.4046 - categorical_accuracy: 0.8766
13504/60000 [=====>........................] - ETA: 1:24 - loss: 0.4040 - categorical_accuracy: 0.8768
13536/60000 [=====>........................] - ETA: 1:24 - loss: 0.4040 - categorical_accuracy: 0.8769
13568/60000 [=====>........................] - ETA: 1:24 - loss: 0.4031 - categorical_accuracy: 0.8772
13600/60000 [=====>........................] - ETA: 1:24 - loss: 0.4028 - categorical_accuracy: 0.8774
13632/60000 [=====>........................] - ETA: 1:24 - loss: 0.4026 - categorical_accuracy: 0.8775
13664/60000 [=====>........................] - ETA: 1:24 - loss: 0.4019 - categorical_accuracy: 0.8777
13696/60000 [=====>........................] - ETA: 1:24 - loss: 0.4017 - categorical_accuracy: 0.8778
13728/60000 [=====>........................] - ETA: 1:23 - loss: 0.4009 - categorical_accuracy: 0.8780
13760/60000 [=====>........................] - ETA: 1:23 - loss: 0.4007 - categorical_accuracy: 0.8781
13792/60000 [=====>........................] - ETA: 1:23 - loss: 0.4007 - categorical_accuracy: 0.8780
13824/60000 [=====>........................] - ETA: 1:23 - loss: 0.4002 - categorical_accuracy: 0.8782
13856/60000 [=====>........................] - ETA: 1:23 - loss: 0.3995 - categorical_accuracy: 0.8783
13888/60000 [=====>........................] - ETA: 1:23 - loss: 0.3992 - categorical_accuracy: 0.8783
13920/60000 [=====>........................] - ETA: 1:23 - loss: 0.3988 - categorical_accuracy: 0.8783
13952/60000 [=====>........................] - ETA: 1:23 - loss: 0.3981 - categorical_accuracy: 0.8785
13984/60000 [=====>........................] - ETA: 1:23 - loss: 0.3974 - categorical_accuracy: 0.8787
14016/60000 [======>.......................] - ETA: 1:23 - loss: 0.3966 - categorical_accuracy: 0.8790
14048/60000 [======>.......................] - ETA: 1:23 - loss: 0.3960 - categorical_accuracy: 0.8792
14080/60000 [======>.......................] - ETA: 1:23 - loss: 0.3953 - categorical_accuracy: 0.8794
14112/60000 [======>.......................] - ETA: 1:23 - loss: 0.3949 - categorical_accuracy: 0.8795
14144/60000 [======>.......................] - ETA: 1:23 - loss: 0.3941 - categorical_accuracy: 0.8798
14176/60000 [======>.......................] - ETA: 1:23 - loss: 0.3938 - categorical_accuracy: 0.8799
14208/60000 [======>.......................] - ETA: 1:23 - loss: 0.3931 - categorical_accuracy: 0.8801
14240/60000 [======>.......................] - ETA: 1:22 - loss: 0.3922 - categorical_accuracy: 0.8804
14272/60000 [======>.......................] - ETA: 1:22 - loss: 0.3916 - categorical_accuracy: 0.8806
14304/60000 [======>.......................] - ETA: 1:22 - loss: 0.3915 - categorical_accuracy: 0.8807
14336/60000 [======>.......................] - ETA: 1:22 - loss: 0.3914 - categorical_accuracy: 0.8807
14368/60000 [======>.......................] - ETA: 1:22 - loss: 0.3908 - categorical_accuracy: 0.8809
14400/60000 [======>.......................] - ETA: 1:22 - loss: 0.3907 - categorical_accuracy: 0.8809
14432/60000 [======>.......................] - ETA: 1:22 - loss: 0.3904 - categorical_accuracy: 0.8810
14464/60000 [======>.......................] - ETA: 1:22 - loss: 0.3901 - categorical_accuracy: 0.8809
14496/60000 [======>.......................] - ETA: 1:22 - loss: 0.3893 - categorical_accuracy: 0.8812
14528/60000 [======>.......................] - ETA: 1:22 - loss: 0.3896 - categorical_accuracy: 0.8812
14560/60000 [======>.......................] - ETA: 1:22 - loss: 0.3889 - categorical_accuracy: 0.8815
14592/60000 [======>.......................] - ETA: 1:22 - loss: 0.3883 - categorical_accuracy: 0.8817
14624/60000 [======>.......................] - ETA: 1:22 - loss: 0.3887 - categorical_accuracy: 0.8817
14656/60000 [======>.......................] - ETA: 1:22 - loss: 0.3885 - categorical_accuracy: 0.8818
14688/60000 [======>.......................] - ETA: 1:22 - loss: 0.3877 - categorical_accuracy: 0.8820
14720/60000 [======>.......................] - ETA: 1:22 - loss: 0.3871 - categorical_accuracy: 0.8822
14752/60000 [======>.......................] - ETA: 1:22 - loss: 0.3865 - categorical_accuracy: 0.8824
14784/60000 [======>.......................] - ETA: 1:21 - loss: 0.3860 - categorical_accuracy: 0.8825
14816/60000 [======>.......................] - ETA: 1:21 - loss: 0.3855 - categorical_accuracy: 0.8827
14848/60000 [======>.......................] - ETA: 1:21 - loss: 0.3851 - categorical_accuracy: 0.8829
14880/60000 [======>.......................] - ETA: 1:21 - loss: 0.3845 - categorical_accuracy: 0.8830
14912/60000 [======>.......................] - ETA: 1:21 - loss: 0.3849 - categorical_accuracy: 0.8830
14944/60000 [======>.......................] - ETA: 1:21 - loss: 0.3845 - categorical_accuracy: 0.8831
14976/60000 [======>.......................] - ETA: 1:21 - loss: 0.3841 - categorical_accuracy: 0.8832
15008/60000 [======>.......................] - ETA: 1:21 - loss: 0.3834 - categorical_accuracy: 0.8835
15040/60000 [======>.......................] - ETA: 1:21 - loss: 0.3827 - categorical_accuracy: 0.8837
15072/60000 [======>.......................] - ETA: 1:21 - loss: 0.3826 - categorical_accuracy: 0.8838
15104/60000 [======>.......................] - ETA: 1:21 - loss: 0.3820 - categorical_accuracy: 0.8840
15136/60000 [======>.......................] - ETA: 1:21 - loss: 0.3813 - categorical_accuracy: 0.8842
15168/60000 [======>.......................] - ETA: 1:21 - loss: 0.3808 - categorical_accuracy: 0.8844
15200/60000 [======>.......................] - ETA: 1:21 - loss: 0.3802 - categorical_accuracy: 0.8845
15232/60000 [======>.......................] - ETA: 1:21 - loss: 0.3795 - categorical_accuracy: 0.8848
15264/60000 [======>.......................] - ETA: 1:21 - loss: 0.3792 - categorical_accuracy: 0.8847
15296/60000 [======>.......................] - ETA: 1:20 - loss: 0.3789 - categorical_accuracy: 0.8849
15328/60000 [======>.......................] - ETA: 1:20 - loss: 0.3782 - categorical_accuracy: 0.8850
15360/60000 [======>.......................] - ETA: 1:20 - loss: 0.3776 - categorical_accuracy: 0.8852
15392/60000 [======>.......................] - ETA: 1:20 - loss: 0.3775 - categorical_accuracy: 0.8852
15424/60000 [======>.......................] - ETA: 1:20 - loss: 0.3773 - categorical_accuracy: 0.8852
15456/60000 [======>.......................] - ETA: 1:20 - loss: 0.3773 - categorical_accuracy: 0.8852
15488/60000 [======>.......................] - ETA: 1:20 - loss: 0.3773 - categorical_accuracy: 0.8853
15520/60000 [======>.......................] - ETA: 1:20 - loss: 0.3771 - categorical_accuracy: 0.8854
15552/60000 [======>.......................] - ETA: 1:20 - loss: 0.3768 - categorical_accuracy: 0.8854
15584/60000 [======>.......................] - ETA: 1:20 - loss: 0.3764 - categorical_accuracy: 0.8856
15616/60000 [======>.......................] - ETA: 1:20 - loss: 0.3759 - categorical_accuracy: 0.8857
15648/60000 [======>.......................] - ETA: 1:20 - loss: 0.3753 - categorical_accuracy: 0.8859
15680/60000 [======>.......................] - ETA: 1:20 - loss: 0.3751 - categorical_accuracy: 0.8858
15712/60000 [======>.......................] - ETA: 1:20 - loss: 0.3751 - categorical_accuracy: 0.8858
15744/60000 [======>.......................] - ETA: 1:20 - loss: 0.3748 - categorical_accuracy: 0.8859
15776/60000 [======>.......................] - ETA: 1:20 - loss: 0.3744 - categorical_accuracy: 0.8860
15808/60000 [======>.......................] - ETA: 1:20 - loss: 0.3741 - categorical_accuracy: 0.8862
15840/60000 [======>.......................] - ETA: 1:19 - loss: 0.3740 - categorical_accuracy: 0.8862
15872/60000 [======>.......................] - ETA: 1:19 - loss: 0.3736 - categorical_accuracy: 0.8863
15904/60000 [======>.......................] - ETA: 1:19 - loss: 0.3730 - categorical_accuracy: 0.8865
15936/60000 [======>.......................] - ETA: 1:19 - loss: 0.3729 - categorical_accuracy: 0.8866
15968/60000 [======>.......................] - ETA: 1:19 - loss: 0.3727 - categorical_accuracy: 0.8866
16000/60000 [=======>......................] - ETA: 1:19 - loss: 0.3725 - categorical_accuracy: 0.8866
16032/60000 [=======>......................] - ETA: 1:19 - loss: 0.3719 - categorical_accuracy: 0.8867
16064/60000 [=======>......................] - ETA: 1:19 - loss: 0.3714 - categorical_accuracy: 0.8870
16096/60000 [=======>......................] - ETA: 1:19 - loss: 0.3710 - categorical_accuracy: 0.8870
16128/60000 [=======>......................] - ETA: 1:19 - loss: 0.3705 - categorical_accuracy: 0.8872
16160/60000 [=======>......................] - ETA: 1:19 - loss: 0.3701 - categorical_accuracy: 0.8874
16192/60000 [=======>......................] - ETA: 1:19 - loss: 0.3700 - categorical_accuracy: 0.8874
16224/60000 [=======>......................] - ETA: 1:19 - loss: 0.3695 - categorical_accuracy: 0.8875
16256/60000 [=======>......................] - ETA: 1:19 - loss: 0.3692 - categorical_accuracy: 0.8875
16288/60000 [=======>......................] - ETA: 1:19 - loss: 0.3686 - categorical_accuracy: 0.8878
16320/60000 [=======>......................] - ETA: 1:19 - loss: 0.3681 - categorical_accuracy: 0.8879
16352/60000 [=======>......................] - ETA: 1:19 - loss: 0.3679 - categorical_accuracy: 0.8879
16384/60000 [=======>......................] - ETA: 1:18 - loss: 0.3673 - categorical_accuracy: 0.8881
16416/60000 [=======>......................] - ETA: 1:18 - loss: 0.3669 - categorical_accuracy: 0.8882
16448/60000 [=======>......................] - ETA: 1:18 - loss: 0.3664 - categorical_accuracy: 0.8883
16480/60000 [=======>......................] - ETA: 1:18 - loss: 0.3658 - categorical_accuracy: 0.8885
16512/60000 [=======>......................] - ETA: 1:18 - loss: 0.3656 - categorical_accuracy: 0.8886
16544/60000 [=======>......................] - ETA: 1:18 - loss: 0.3652 - categorical_accuracy: 0.8887
16576/60000 [=======>......................] - ETA: 1:18 - loss: 0.3647 - categorical_accuracy: 0.8888
16608/60000 [=======>......................] - ETA: 1:18 - loss: 0.3643 - categorical_accuracy: 0.8889
16640/60000 [=======>......................] - ETA: 1:18 - loss: 0.3636 - categorical_accuracy: 0.8891
16672/60000 [=======>......................] - ETA: 1:18 - loss: 0.3639 - categorical_accuracy: 0.8892
16704/60000 [=======>......................] - ETA: 1:18 - loss: 0.3633 - categorical_accuracy: 0.8894
16736/60000 [=======>......................] - ETA: 1:18 - loss: 0.3632 - categorical_accuracy: 0.8893
16768/60000 [=======>......................] - ETA: 1:18 - loss: 0.3629 - categorical_accuracy: 0.8894
16800/60000 [=======>......................] - ETA: 1:18 - loss: 0.3627 - categorical_accuracy: 0.8895
16832/60000 [=======>......................] - ETA: 1:18 - loss: 0.3624 - categorical_accuracy: 0.8896
16864/60000 [=======>......................] - ETA: 1:18 - loss: 0.3623 - categorical_accuracy: 0.8896
16896/60000 [=======>......................] - ETA: 1:17 - loss: 0.3617 - categorical_accuracy: 0.8899
16928/60000 [=======>......................] - ETA: 1:17 - loss: 0.3616 - categorical_accuracy: 0.8899
16960/60000 [=======>......................] - ETA: 1:17 - loss: 0.3613 - categorical_accuracy: 0.8900
16992/60000 [=======>......................] - ETA: 1:17 - loss: 0.3607 - categorical_accuracy: 0.8902
17024/60000 [=======>......................] - ETA: 1:17 - loss: 0.3602 - categorical_accuracy: 0.8903
17056/60000 [=======>......................] - ETA: 1:17 - loss: 0.3598 - categorical_accuracy: 0.8904
17088/60000 [=======>......................] - ETA: 1:17 - loss: 0.3594 - categorical_accuracy: 0.8905
17120/60000 [=======>......................] - ETA: 1:17 - loss: 0.3591 - categorical_accuracy: 0.8907
17152/60000 [=======>......................] - ETA: 1:17 - loss: 0.3588 - categorical_accuracy: 0.8907
17184/60000 [=======>......................] - ETA: 1:17 - loss: 0.3583 - categorical_accuracy: 0.8909
17216/60000 [=======>......................] - ETA: 1:17 - loss: 0.3578 - categorical_accuracy: 0.8910
17248/60000 [=======>......................] - ETA: 1:17 - loss: 0.3572 - categorical_accuracy: 0.8912
17280/60000 [=======>......................] - ETA: 1:17 - loss: 0.3568 - categorical_accuracy: 0.8913
17312/60000 [=======>......................] - ETA: 1:17 - loss: 0.3571 - categorical_accuracy: 0.8912
17344/60000 [=======>......................] - ETA: 1:17 - loss: 0.3566 - categorical_accuracy: 0.8914
17376/60000 [=======>......................] - ETA: 1:17 - loss: 0.3564 - categorical_accuracy: 0.8915
17408/60000 [=======>......................] - ETA: 1:17 - loss: 0.3561 - categorical_accuracy: 0.8916
17440/60000 [=======>......................] - ETA: 1:16 - loss: 0.3556 - categorical_accuracy: 0.8917
17472/60000 [=======>......................] - ETA: 1:16 - loss: 0.3550 - categorical_accuracy: 0.8919
17504/60000 [=======>......................] - ETA: 1:16 - loss: 0.3547 - categorical_accuracy: 0.8920
17536/60000 [=======>......................] - ETA: 1:16 - loss: 0.3542 - categorical_accuracy: 0.8921
17568/60000 [=======>......................] - ETA: 1:16 - loss: 0.3538 - categorical_accuracy: 0.8922
17600/60000 [=======>......................] - ETA: 1:16 - loss: 0.3537 - categorical_accuracy: 0.8923
17632/60000 [=======>......................] - ETA: 1:16 - loss: 0.3533 - categorical_accuracy: 0.8924
17664/60000 [=======>......................] - ETA: 1:16 - loss: 0.3531 - categorical_accuracy: 0.8924
17696/60000 [=======>......................] - ETA: 1:16 - loss: 0.3526 - categorical_accuracy: 0.8926
17728/60000 [=======>......................] - ETA: 1:16 - loss: 0.3523 - categorical_accuracy: 0.8927
17760/60000 [=======>......................] - ETA: 1:16 - loss: 0.3517 - categorical_accuracy: 0.8928
17792/60000 [=======>......................] - ETA: 1:16 - loss: 0.3512 - categorical_accuracy: 0.8930
17824/60000 [=======>......................] - ETA: 1:16 - loss: 0.3506 - categorical_accuracy: 0.8932
17856/60000 [=======>......................] - ETA: 1:16 - loss: 0.3504 - categorical_accuracy: 0.8933
17888/60000 [=======>......................] - ETA: 1:16 - loss: 0.3502 - categorical_accuracy: 0.8933
17920/60000 [=======>......................] - ETA: 1:16 - loss: 0.3499 - categorical_accuracy: 0.8934
17952/60000 [=======>......................] - ETA: 1:16 - loss: 0.3494 - categorical_accuracy: 0.8936
17984/60000 [=======>......................] - ETA: 1:15 - loss: 0.3488 - categorical_accuracy: 0.8938
18016/60000 [========>.....................] - ETA: 1:15 - loss: 0.3483 - categorical_accuracy: 0.8939
18048/60000 [========>.....................] - ETA: 1:15 - loss: 0.3477 - categorical_accuracy: 0.8941
18080/60000 [========>.....................] - ETA: 1:15 - loss: 0.3471 - categorical_accuracy: 0.8943
18112/60000 [========>.....................] - ETA: 1:15 - loss: 0.3465 - categorical_accuracy: 0.8945
18144/60000 [========>.....................] - ETA: 1:15 - loss: 0.3462 - categorical_accuracy: 0.8946
18176/60000 [========>.....................] - ETA: 1:15 - loss: 0.3460 - categorical_accuracy: 0.8946
18208/60000 [========>.....................] - ETA: 1:15 - loss: 0.3456 - categorical_accuracy: 0.8947
18240/60000 [========>.....................] - ETA: 1:15 - loss: 0.3461 - categorical_accuracy: 0.8946
18272/60000 [========>.....................] - ETA: 1:15 - loss: 0.3456 - categorical_accuracy: 0.8947
18304/60000 [========>.....................] - ETA: 1:15 - loss: 0.3452 - categorical_accuracy: 0.8948
18336/60000 [========>.....................] - ETA: 1:15 - loss: 0.3451 - categorical_accuracy: 0.8949
18368/60000 [========>.....................] - ETA: 1:15 - loss: 0.3446 - categorical_accuracy: 0.8950
18400/60000 [========>.....................] - ETA: 1:15 - loss: 0.3442 - categorical_accuracy: 0.8951
18432/60000 [========>.....................] - ETA: 1:15 - loss: 0.3441 - categorical_accuracy: 0.8952
18464/60000 [========>.....................] - ETA: 1:15 - loss: 0.3436 - categorical_accuracy: 0.8954
18496/60000 [========>.....................] - ETA: 1:15 - loss: 0.3432 - categorical_accuracy: 0.8955
18528/60000 [========>.....................] - ETA: 1:14 - loss: 0.3428 - categorical_accuracy: 0.8956
18560/60000 [========>.....................] - ETA: 1:14 - loss: 0.3424 - categorical_accuracy: 0.8957
18592/60000 [========>.....................] - ETA: 1:14 - loss: 0.3419 - categorical_accuracy: 0.8958
18624/60000 [========>.....................] - ETA: 1:14 - loss: 0.3415 - categorical_accuracy: 0.8959
18656/60000 [========>.....................] - ETA: 1:14 - loss: 0.3416 - categorical_accuracy: 0.8960
18688/60000 [========>.....................] - ETA: 1:14 - loss: 0.3412 - categorical_accuracy: 0.8961
18720/60000 [========>.....................] - ETA: 1:14 - loss: 0.3407 - categorical_accuracy: 0.8963
18752/60000 [========>.....................] - ETA: 1:14 - loss: 0.3402 - categorical_accuracy: 0.8964
18784/60000 [========>.....................] - ETA: 1:14 - loss: 0.3400 - categorical_accuracy: 0.8965
18816/60000 [========>.....................] - ETA: 1:14 - loss: 0.3395 - categorical_accuracy: 0.8967
18848/60000 [========>.....................] - ETA: 1:14 - loss: 0.3391 - categorical_accuracy: 0.8967
18880/60000 [========>.....................] - ETA: 1:14 - loss: 0.3389 - categorical_accuracy: 0.8968
18912/60000 [========>.....................] - ETA: 1:14 - loss: 0.3386 - categorical_accuracy: 0.8968
18944/60000 [========>.....................] - ETA: 1:14 - loss: 0.3382 - categorical_accuracy: 0.8970
18976/60000 [========>.....................] - ETA: 1:14 - loss: 0.3378 - categorical_accuracy: 0.8970
19008/60000 [========>.....................] - ETA: 1:14 - loss: 0.3377 - categorical_accuracy: 0.8970
19040/60000 [========>.....................] - ETA: 1:14 - loss: 0.3378 - categorical_accuracy: 0.8970
19072/60000 [========>.....................] - ETA: 1:14 - loss: 0.3375 - categorical_accuracy: 0.8970
19104/60000 [========>.....................] - ETA: 1:13 - loss: 0.3370 - categorical_accuracy: 0.8972
19136/60000 [========>.....................] - ETA: 1:13 - loss: 0.3367 - categorical_accuracy: 0.8973
19168/60000 [========>.....................] - ETA: 1:13 - loss: 0.3364 - categorical_accuracy: 0.8974
19200/60000 [========>.....................] - ETA: 1:13 - loss: 0.3359 - categorical_accuracy: 0.8976
19232/60000 [========>.....................] - ETA: 1:13 - loss: 0.3354 - categorical_accuracy: 0.8977
19264/60000 [========>.....................] - ETA: 1:13 - loss: 0.3353 - categorical_accuracy: 0.8978
19296/60000 [========>.....................] - ETA: 1:13 - loss: 0.3348 - categorical_accuracy: 0.8980
19328/60000 [========>.....................] - ETA: 1:13 - loss: 0.3346 - categorical_accuracy: 0.8980
19360/60000 [========>.....................] - ETA: 1:13 - loss: 0.3340 - categorical_accuracy: 0.8982
19392/60000 [========>.....................] - ETA: 1:13 - loss: 0.3339 - categorical_accuracy: 0.8982
19424/60000 [========>.....................] - ETA: 1:13 - loss: 0.3335 - categorical_accuracy: 0.8983
19456/60000 [========>.....................] - ETA: 1:13 - loss: 0.3331 - categorical_accuracy: 0.8984
19488/60000 [========>.....................] - ETA: 1:13 - loss: 0.3329 - categorical_accuracy: 0.8986
19520/60000 [========>.....................] - ETA: 1:13 - loss: 0.3326 - categorical_accuracy: 0.8986
19552/60000 [========>.....................] - ETA: 1:13 - loss: 0.3323 - categorical_accuracy: 0.8987
19584/60000 [========>.....................] - ETA: 1:13 - loss: 0.3322 - categorical_accuracy: 0.8987
19616/60000 [========>.....................] - ETA: 1:13 - loss: 0.3318 - categorical_accuracy: 0.8989
19648/60000 [========>.....................] - ETA: 1:13 - loss: 0.3315 - categorical_accuracy: 0.8990
19680/60000 [========>.....................] - ETA: 1:12 - loss: 0.3312 - categorical_accuracy: 0.8990
19712/60000 [========>.....................] - ETA: 1:12 - loss: 0.3310 - categorical_accuracy: 0.8990
19744/60000 [========>.....................] - ETA: 1:12 - loss: 0.3307 - categorical_accuracy: 0.8991
19776/60000 [========>.....................] - ETA: 1:12 - loss: 0.3302 - categorical_accuracy: 0.8993
19808/60000 [========>.....................] - ETA: 1:12 - loss: 0.3297 - categorical_accuracy: 0.8994
19840/60000 [========>.....................] - ETA: 1:12 - loss: 0.3295 - categorical_accuracy: 0.8995
19872/60000 [========>.....................] - ETA: 1:12 - loss: 0.3291 - categorical_accuracy: 0.8996
19904/60000 [========>.....................] - ETA: 1:12 - loss: 0.3288 - categorical_accuracy: 0.8997
19936/60000 [========>.....................] - ETA: 1:12 - loss: 0.3288 - categorical_accuracy: 0.8997
19968/60000 [========>.....................] - ETA: 1:12 - loss: 0.3287 - categorical_accuracy: 0.8998
20000/60000 [=========>....................] - ETA: 1:12 - loss: 0.3283 - categorical_accuracy: 0.8999
20032/60000 [=========>....................] - ETA: 1:12 - loss: 0.3280 - categorical_accuracy: 0.9001
20064/60000 [=========>....................] - ETA: 1:12 - loss: 0.3279 - categorical_accuracy: 0.9001
20096/60000 [=========>....................] - ETA: 1:12 - loss: 0.3277 - categorical_accuracy: 0.9002
20128/60000 [=========>....................] - ETA: 1:12 - loss: 0.3278 - categorical_accuracy: 0.9001
20160/60000 [=========>....................] - ETA: 1:12 - loss: 0.3277 - categorical_accuracy: 0.9002
20192/60000 [=========>....................] - ETA: 1:12 - loss: 0.3275 - categorical_accuracy: 0.9003
20224/60000 [=========>....................] - ETA: 1:12 - loss: 0.3272 - categorical_accuracy: 0.9003
20256/60000 [=========>....................] - ETA: 1:11 - loss: 0.3271 - categorical_accuracy: 0.9003
20288/60000 [=========>....................] - ETA: 1:11 - loss: 0.3267 - categorical_accuracy: 0.9004
20320/60000 [=========>....................] - ETA: 1:11 - loss: 0.3264 - categorical_accuracy: 0.9005
20352/60000 [=========>....................] - ETA: 1:11 - loss: 0.3261 - categorical_accuracy: 0.9006
20384/60000 [=========>....................] - ETA: 1:11 - loss: 0.3261 - categorical_accuracy: 0.9006
20416/60000 [=========>....................] - ETA: 1:11 - loss: 0.3257 - categorical_accuracy: 0.9007
20448/60000 [=========>....................] - ETA: 1:11 - loss: 0.3257 - categorical_accuracy: 0.9007
20480/60000 [=========>....................] - ETA: 1:11 - loss: 0.3253 - categorical_accuracy: 0.9008
20512/60000 [=========>....................] - ETA: 1:11 - loss: 0.3252 - categorical_accuracy: 0.9008
20544/60000 [=========>....................] - ETA: 1:11 - loss: 0.3249 - categorical_accuracy: 0.9009
20576/60000 [=========>....................] - ETA: 1:11 - loss: 0.3247 - categorical_accuracy: 0.9009
20608/60000 [=========>....................] - ETA: 1:11 - loss: 0.3247 - categorical_accuracy: 0.9009
20640/60000 [=========>....................] - ETA: 1:11 - loss: 0.3244 - categorical_accuracy: 0.9010
20672/60000 [=========>....................] - ETA: 1:11 - loss: 0.3242 - categorical_accuracy: 0.9010
20704/60000 [=========>....................] - ETA: 1:11 - loss: 0.3238 - categorical_accuracy: 0.9012
20736/60000 [=========>....................] - ETA: 1:11 - loss: 0.3239 - categorical_accuracy: 0.9010
20768/60000 [=========>....................] - ETA: 1:11 - loss: 0.3241 - categorical_accuracy: 0.9010
20800/60000 [=========>....................] - ETA: 1:10 - loss: 0.3237 - categorical_accuracy: 0.9012
20832/60000 [=========>....................] - ETA: 1:10 - loss: 0.3235 - categorical_accuracy: 0.9013
20864/60000 [=========>....................] - ETA: 1:10 - loss: 0.3233 - categorical_accuracy: 0.9013
20896/60000 [=========>....................] - ETA: 1:10 - loss: 0.3231 - categorical_accuracy: 0.9014
20928/60000 [=========>....................] - ETA: 1:10 - loss: 0.3227 - categorical_accuracy: 0.9015
20960/60000 [=========>....................] - ETA: 1:10 - loss: 0.3223 - categorical_accuracy: 0.9016
20992/60000 [=========>....................] - ETA: 1:10 - loss: 0.3222 - categorical_accuracy: 0.9016
21024/60000 [=========>....................] - ETA: 1:10 - loss: 0.3220 - categorical_accuracy: 0.9017
21056/60000 [=========>....................] - ETA: 1:10 - loss: 0.3221 - categorical_accuracy: 0.9016
21088/60000 [=========>....................] - ETA: 1:10 - loss: 0.3218 - categorical_accuracy: 0.9017
21120/60000 [=========>....................] - ETA: 1:10 - loss: 0.3214 - categorical_accuracy: 0.9018
21152/60000 [=========>....................] - ETA: 1:10 - loss: 0.3210 - categorical_accuracy: 0.9019
21184/60000 [=========>....................] - ETA: 1:10 - loss: 0.3207 - categorical_accuracy: 0.9020
21216/60000 [=========>....................] - ETA: 1:10 - loss: 0.3203 - categorical_accuracy: 0.9021
21248/60000 [=========>....................] - ETA: 1:10 - loss: 0.3202 - categorical_accuracy: 0.9022
21280/60000 [=========>....................] - ETA: 1:10 - loss: 0.3199 - categorical_accuracy: 0.9023
21312/60000 [=========>....................] - ETA: 1:09 - loss: 0.3197 - categorical_accuracy: 0.9024
21344/60000 [=========>....................] - ETA: 1:09 - loss: 0.3194 - categorical_accuracy: 0.9025
21376/60000 [=========>....................] - ETA: 1:09 - loss: 0.3191 - categorical_accuracy: 0.9026
21408/60000 [=========>....................] - ETA: 1:09 - loss: 0.3190 - categorical_accuracy: 0.9026
21440/60000 [=========>....................] - ETA: 1:09 - loss: 0.3188 - categorical_accuracy: 0.9027
21472/60000 [=========>....................] - ETA: 1:09 - loss: 0.3184 - categorical_accuracy: 0.9028
21504/60000 [=========>....................] - ETA: 1:09 - loss: 0.3181 - categorical_accuracy: 0.9029
21536/60000 [=========>....................] - ETA: 1:09 - loss: 0.3178 - categorical_accuracy: 0.9030
21568/60000 [=========>....................] - ETA: 1:09 - loss: 0.3176 - categorical_accuracy: 0.9031
21600/60000 [=========>....................] - ETA: 1:09 - loss: 0.3172 - categorical_accuracy: 0.9032
21632/60000 [=========>....................] - ETA: 1:09 - loss: 0.3170 - categorical_accuracy: 0.9033
21664/60000 [=========>....................] - ETA: 1:09 - loss: 0.3166 - categorical_accuracy: 0.9034
21696/60000 [=========>....................] - ETA: 1:09 - loss: 0.3165 - categorical_accuracy: 0.9034
21728/60000 [=========>....................] - ETA: 1:09 - loss: 0.3161 - categorical_accuracy: 0.9036
21760/60000 [=========>....................] - ETA: 1:09 - loss: 0.3160 - categorical_accuracy: 0.9035
21792/60000 [=========>....................] - ETA: 1:09 - loss: 0.3157 - categorical_accuracy: 0.9036
21824/60000 [=========>....................] - ETA: 1:09 - loss: 0.3152 - categorical_accuracy: 0.9037
21856/60000 [=========>....................] - ETA: 1:08 - loss: 0.3150 - categorical_accuracy: 0.9038
21888/60000 [=========>....................] - ETA: 1:08 - loss: 0.3147 - categorical_accuracy: 0.9039
21920/60000 [=========>....................] - ETA: 1:08 - loss: 0.3143 - categorical_accuracy: 0.9040
21952/60000 [=========>....................] - ETA: 1:08 - loss: 0.3142 - categorical_accuracy: 0.9041
21984/60000 [=========>....................] - ETA: 1:08 - loss: 0.3141 - categorical_accuracy: 0.9042
22016/60000 [==========>...................] - ETA: 1:08 - loss: 0.3137 - categorical_accuracy: 0.9043
22048/60000 [==========>...................] - ETA: 1:08 - loss: 0.3134 - categorical_accuracy: 0.9044
22080/60000 [==========>...................] - ETA: 1:08 - loss: 0.3131 - categorical_accuracy: 0.9045
22112/60000 [==========>...................] - ETA: 1:08 - loss: 0.3128 - categorical_accuracy: 0.9046
22144/60000 [==========>...................] - ETA: 1:08 - loss: 0.3125 - categorical_accuracy: 0.9047
22176/60000 [==========>...................] - ETA: 1:08 - loss: 0.3121 - categorical_accuracy: 0.9048
22208/60000 [==========>...................] - ETA: 1:08 - loss: 0.3119 - categorical_accuracy: 0.9049
22240/60000 [==========>...................] - ETA: 1:08 - loss: 0.3115 - categorical_accuracy: 0.9050
22272/60000 [==========>...................] - ETA: 1:08 - loss: 0.3113 - categorical_accuracy: 0.9051
22304/60000 [==========>...................] - ETA: 1:08 - loss: 0.3110 - categorical_accuracy: 0.9051
22336/60000 [==========>...................] - ETA: 1:08 - loss: 0.3110 - categorical_accuracy: 0.9052
22368/60000 [==========>...................] - ETA: 1:08 - loss: 0.3110 - categorical_accuracy: 0.9052
22400/60000 [==========>...................] - ETA: 1:08 - loss: 0.3108 - categorical_accuracy: 0.9053
22432/60000 [==========>...................] - ETA: 1:07 - loss: 0.3105 - categorical_accuracy: 0.9054
22464/60000 [==========>...................] - ETA: 1:07 - loss: 0.3105 - categorical_accuracy: 0.9054
22496/60000 [==========>...................] - ETA: 1:07 - loss: 0.3103 - categorical_accuracy: 0.9054
22528/60000 [==========>...................] - ETA: 1:07 - loss: 0.3105 - categorical_accuracy: 0.9054
22560/60000 [==========>...................] - ETA: 1:07 - loss: 0.3104 - categorical_accuracy: 0.9054
22592/60000 [==========>...................] - ETA: 1:07 - loss: 0.3102 - categorical_accuracy: 0.9054
22624/60000 [==========>...................] - ETA: 1:07 - loss: 0.3098 - categorical_accuracy: 0.9055
22656/60000 [==========>...................] - ETA: 1:07 - loss: 0.3098 - categorical_accuracy: 0.9056
22688/60000 [==========>...................] - ETA: 1:07 - loss: 0.3094 - categorical_accuracy: 0.9057
22720/60000 [==========>...................] - ETA: 1:07 - loss: 0.3091 - categorical_accuracy: 0.9059
22752/60000 [==========>...................] - ETA: 1:07 - loss: 0.3087 - categorical_accuracy: 0.9060
22784/60000 [==========>...................] - ETA: 1:07 - loss: 0.3083 - categorical_accuracy: 0.9061
22816/60000 [==========>...................] - ETA: 1:07 - loss: 0.3081 - categorical_accuracy: 0.9062
22848/60000 [==========>...................] - ETA: 1:07 - loss: 0.3078 - categorical_accuracy: 0.9063
22880/60000 [==========>...................] - ETA: 1:07 - loss: 0.3076 - categorical_accuracy: 0.9063
22912/60000 [==========>...................] - ETA: 1:07 - loss: 0.3073 - categorical_accuracy: 0.9064
22944/60000 [==========>...................] - ETA: 1:07 - loss: 0.3073 - categorical_accuracy: 0.9065
22976/60000 [==========>...................] - ETA: 1:06 - loss: 0.3070 - categorical_accuracy: 0.9066
23008/60000 [==========>...................] - ETA: 1:06 - loss: 0.3070 - categorical_accuracy: 0.9066
23040/60000 [==========>...................] - ETA: 1:06 - loss: 0.3066 - categorical_accuracy: 0.9067
23072/60000 [==========>...................] - ETA: 1:06 - loss: 0.3065 - categorical_accuracy: 0.9067
23104/60000 [==========>...................] - ETA: 1:06 - loss: 0.3061 - categorical_accuracy: 0.9068
23136/60000 [==========>...................] - ETA: 1:06 - loss: 0.3058 - categorical_accuracy: 0.9069
23168/60000 [==========>...................] - ETA: 1:06 - loss: 0.3056 - categorical_accuracy: 0.9070
23200/60000 [==========>...................] - ETA: 1:06 - loss: 0.3055 - categorical_accuracy: 0.9070
23232/60000 [==========>...................] - ETA: 1:06 - loss: 0.3052 - categorical_accuracy: 0.9071
23264/60000 [==========>...................] - ETA: 1:06 - loss: 0.3052 - categorical_accuracy: 0.9072
23296/60000 [==========>...................] - ETA: 1:06 - loss: 0.3048 - categorical_accuracy: 0.9073
23328/60000 [==========>...................] - ETA: 1:06 - loss: 0.3045 - categorical_accuracy: 0.9074
23360/60000 [==========>...................] - ETA: 1:06 - loss: 0.3042 - categorical_accuracy: 0.9074
23392/60000 [==========>...................] - ETA: 1:06 - loss: 0.3039 - categorical_accuracy: 0.9075
23424/60000 [==========>...................] - ETA: 1:06 - loss: 0.3036 - categorical_accuracy: 0.9076
23456/60000 [==========>...................] - ETA: 1:06 - loss: 0.3036 - categorical_accuracy: 0.9076
23488/60000 [==========>...................] - ETA: 1:06 - loss: 0.3034 - categorical_accuracy: 0.9076
23520/60000 [==========>...................] - ETA: 1:06 - loss: 0.3036 - categorical_accuracy: 0.9075
23552/60000 [==========>...................] - ETA: 1:05 - loss: 0.3034 - categorical_accuracy: 0.9076
23584/60000 [==========>...................] - ETA: 1:05 - loss: 0.3031 - categorical_accuracy: 0.9076
23616/60000 [==========>...................] - ETA: 1:05 - loss: 0.3027 - categorical_accuracy: 0.9078
23648/60000 [==========>...................] - ETA: 1:05 - loss: 0.3024 - categorical_accuracy: 0.9079
23680/60000 [==========>...................] - ETA: 1:05 - loss: 0.3023 - categorical_accuracy: 0.9079
23712/60000 [==========>...................] - ETA: 1:05 - loss: 0.3020 - categorical_accuracy: 0.9080
23744/60000 [==========>...................] - ETA: 1:05 - loss: 0.3018 - categorical_accuracy: 0.9080
23776/60000 [==========>...................] - ETA: 1:05 - loss: 0.3019 - categorical_accuracy: 0.9080
23808/60000 [==========>...................] - ETA: 1:05 - loss: 0.3015 - categorical_accuracy: 0.9081
23840/60000 [==========>...................] - ETA: 1:05 - loss: 0.3012 - categorical_accuracy: 0.9082
23872/60000 [==========>...................] - ETA: 1:05 - loss: 0.3008 - categorical_accuracy: 0.9083
23904/60000 [==========>...................] - ETA: 1:05 - loss: 0.3005 - categorical_accuracy: 0.9085
23936/60000 [==========>...................] - ETA: 1:05 - loss: 0.3002 - categorical_accuracy: 0.9085
23968/60000 [==========>...................] - ETA: 1:05 - loss: 0.2999 - categorical_accuracy: 0.9086
24000/60000 [===========>..................] - ETA: 1:05 - loss: 0.2997 - categorical_accuracy: 0.9087
24032/60000 [===========>..................] - ETA: 1:05 - loss: 0.2995 - categorical_accuracy: 0.9088
24064/60000 [===========>..................] - ETA: 1:05 - loss: 0.2991 - categorical_accuracy: 0.9089
24096/60000 [===========>..................] - ETA: 1:04 - loss: 0.2988 - categorical_accuracy: 0.9089
24128/60000 [===========>..................] - ETA: 1:04 - loss: 0.2988 - categorical_accuracy: 0.9089
24160/60000 [===========>..................] - ETA: 1:04 - loss: 0.2985 - categorical_accuracy: 0.9090
24192/60000 [===========>..................] - ETA: 1:04 - loss: 0.2983 - categorical_accuracy: 0.9090
24224/60000 [===========>..................] - ETA: 1:04 - loss: 0.2984 - categorical_accuracy: 0.9090
24256/60000 [===========>..................] - ETA: 1:04 - loss: 0.2982 - categorical_accuracy: 0.9091
24288/60000 [===========>..................] - ETA: 1:04 - loss: 0.2979 - categorical_accuracy: 0.9092
24320/60000 [===========>..................] - ETA: 1:04 - loss: 0.2977 - categorical_accuracy: 0.9092
24352/60000 [===========>..................] - ETA: 1:04 - loss: 0.2976 - categorical_accuracy: 0.9092
24384/60000 [===========>..................] - ETA: 1:04 - loss: 0.2973 - categorical_accuracy: 0.9093
24416/60000 [===========>..................] - ETA: 1:04 - loss: 0.2969 - categorical_accuracy: 0.9094
24448/60000 [===========>..................] - ETA: 1:04 - loss: 0.2967 - categorical_accuracy: 0.9095
24480/60000 [===========>..................] - ETA: 1:04 - loss: 0.2964 - categorical_accuracy: 0.9096
24512/60000 [===========>..................] - ETA: 1:04 - loss: 0.2965 - categorical_accuracy: 0.9095
24544/60000 [===========>..................] - ETA: 1:04 - loss: 0.2966 - categorical_accuracy: 0.9095
24576/60000 [===========>..................] - ETA: 1:04 - loss: 0.2963 - categorical_accuracy: 0.9095
24608/60000 [===========>..................] - ETA: 1:04 - loss: 0.2960 - categorical_accuracy: 0.9097
24640/60000 [===========>..................] - ETA: 1:03 - loss: 0.2957 - categorical_accuracy: 0.9097
24672/60000 [===========>..................] - ETA: 1:03 - loss: 0.2953 - categorical_accuracy: 0.9098
24704/60000 [===========>..................] - ETA: 1:03 - loss: 0.2953 - categorical_accuracy: 0.9098
24736/60000 [===========>..................] - ETA: 1:03 - loss: 0.2951 - categorical_accuracy: 0.9098
24768/60000 [===========>..................] - ETA: 1:03 - loss: 0.2951 - categorical_accuracy: 0.9099
24800/60000 [===========>..................] - ETA: 1:03 - loss: 0.2948 - categorical_accuracy: 0.9100
24832/60000 [===========>..................] - ETA: 1:03 - loss: 0.2946 - categorical_accuracy: 0.9100
24864/60000 [===========>..................] - ETA: 1:03 - loss: 0.2943 - categorical_accuracy: 0.9101
24896/60000 [===========>..................] - ETA: 1:03 - loss: 0.2942 - categorical_accuracy: 0.9102
24928/60000 [===========>..................] - ETA: 1:03 - loss: 0.2939 - categorical_accuracy: 0.9103
24960/60000 [===========>..................] - ETA: 1:03 - loss: 0.2936 - categorical_accuracy: 0.9104
24992/60000 [===========>..................] - ETA: 1:03 - loss: 0.2933 - categorical_accuracy: 0.9105
25024/60000 [===========>..................] - ETA: 1:03 - loss: 0.2931 - categorical_accuracy: 0.9105
25056/60000 [===========>..................] - ETA: 1:03 - loss: 0.2928 - categorical_accuracy: 0.9106
25088/60000 [===========>..................] - ETA: 1:03 - loss: 0.2925 - categorical_accuracy: 0.9108
25120/60000 [===========>..................] - ETA: 1:03 - loss: 0.2921 - categorical_accuracy: 0.9109
25152/60000 [===========>..................] - ETA: 1:03 - loss: 0.2918 - categorical_accuracy: 0.9110
25184/60000 [===========>..................] - ETA: 1:03 - loss: 0.2914 - categorical_accuracy: 0.9111
25216/60000 [===========>..................] - ETA: 1:02 - loss: 0.2912 - categorical_accuracy: 0.9111
25248/60000 [===========>..................] - ETA: 1:02 - loss: 0.2909 - categorical_accuracy: 0.9112
25280/60000 [===========>..................] - ETA: 1:02 - loss: 0.2911 - categorical_accuracy: 0.9112
25312/60000 [===========>..................] - ETA: 1:02 - loss: 0.2911 - categorical_accuracy: 0.9112
25344/60000 [===========>..................] - ETA: 1:02 - loss: 0.2909 - categorical_accuracy: 0.9113
25376/60000 [===========>..................] - ETA: 1:02 - loss: 0.2907 - categorical_accuracy: 0.9114
25408/60000 [===========>..................] - ETA: 1:02 - loss: 0.2904 - categorical_accuracy: 0.9114
25440/60000 [===========>..................] - ETA: 1:02 - loss: 0.2902 - categorical_accuracy: 0.9115
25472/60000 [===========>..................] - ETA: 1:02 - loss: 0.2900 - categorical_accuracy: 0.9115
25504/60000 [===========>..................] - ETA: 1:02 - loss: 0.2896 - categorical_accuracy: 0.9116
25536/60000 [===========>..................] - ETA: 1:02 - loss: 0.2893 - categorical_accuracy: 0.9117
25568/60000 [===========>..................] - ETA: 1:02 - loss: 0.2890 - categorical_accuracy: 0.9118
25600/60000 [===========>..................] - ETA: 1:02 - loss: 0.2888 - categorical_accuracy: 0.9119
25632/60000 [===========>..................] - ETA: 1:02 - loss: 0.2889 - categorical_accuracy: 0.9119
25664/60000 [===========>..................] - ETA: 1:02 - loss: 0.2886 - categorical_accuracy: 0.9119
25696/60000 [===========>..................] - ETA: 1:02 - loss: 0.2883 - categorical_accuracy: 0.9120
25728/60000 [===========>..................] - ETA: 1:02 - loss: 0.2882 - categorical_accuracy: 0.9121
25760/60000 [===========>..................] - ETA: 1:02 - loss: 0.2884 - categorical_accuracy: 0.9120
25792/60000 [===========>..................] - ETA: 1:01 - loss: 0.2883 - categorical_accuracy: 0.9119
25824/60000 [===========>..................] - ETA: 1:01 - loss: 0.2882 - categorical_accuracy: 0.9119
25856/60000 [===========>..................] - ETA: 1:01 - loss: 0.2879 - categorical_accuracy: 0.9120
25888/60000 [===========>..................] - ETA: 1:01 - loss: 0.2876 - categorical_accuracy: 0.9121
25920/60000 [===========>..................] - ETA: 1:01 - loss: 0.2874 - categorical_accuracy: 0.9122
25952/60000 [===========>..................] - ETA: 1:01 - loss: 0.2873 - categorical_accuracy: 0.9122
25984/60000 [===========>..................] - ETA: 1:01 - loss: 0.2870 - categorical_accuracy: 0.9123
26016/60000 [============>.................] - ETA: 1:01 - loss: 0.2872 - categorical_accuracy: 0.9122
26048/60000 [============>.................] - ETA: 1:01 - loss: 0.2872 - categorical_accuracy: 0.9122
26080/60000 [============>.................] - ETA: 1:01 - loss: 0.2870 - categorical_accuracy: 0.9122
26112/60000 [============>.................] - ETA: 1:01 - loss: 0.2869 - categorical_accuracy: 0.9123
26144/60000 [============>.................] - ETA: 1:01 - loss: 0.2869 - categorical_accuracy: 0.9123
26176/60000 [============>.................] - ETA: 1:01 - loss: 0.2870 - categorical_accuracy: 0.9123
26208/60000 [============>.................] - ETA: 1:01 - loss: 0.2869 - categorical_accuracy: 0.9123
26240/60000 [============>.................] - ETA: 1:01 - loss: 0.2866 - categorical_accuracy: 0.9124
26272/60000 [============>.................] - ETA: 1:01 - loss: 0.2863 - categorical_accuracy: 0.9125
26304/60000 [============>.................] - ETA: 1:01 - loss: 0.2861 - categorical_accuracy: 0.9126
26336/60000 [============>.................] - ETA: 1:01 - loss: 0.2858 - categorical_accuracy: 0.9127
26368/60000 [============>.................] - ETA: 1:00 - loss: 0.2855 - categorical_accuracy: 0.9128
26400/60000 [============>.................] - ETA: 1:00 - loss: 0.2852 - categorical_accuracy: 0.9129
26432/60000 [============>.................] - ETA: 1:00 - loss: 0.2849 - categorical_accuracy: 0.9129
26464/60000 [============>.................] - ETA: 1:00 - loss: 0.2848 - categorical_accuracy: 0.9130
26496/60000 [============>.................] - ETA: 1:00 - loss: 0.2848 - categorical_accuracy: 0.9130
26528/60000 [============>.................] - ETA: 1:00 - loss: 0.2846 - categorical_accuracy: 0.9130
26560/60000 [============>.................] - ETA: 1:00 - loss: 0.2843 - categorical_accuracy: 0.9131
26592/60000 [============>.................] - ETA: 1:00 - loss: 0.2840 - categorical_accuracy: 0.9132
26624/60000 [============>.................] - ETA: 1:00 - loss: 0.2839 - categorical_accuracy: 0.9133
26656/60000 [============>.................] - ETA: 1:00 - loss: 0.2837 - categorical_accuracy: 0.9133
26688/60000 [============>.................] - ETA: 1:00 - loss: 0.2836 - categorical_accuracy: 0.9134
26720/60000 [============>.................] - ETA: 1:00 - loss: 0.2833 - categorical_accuracy: 0.9134
26752/60000 [============>.................] - ETA: 1:00 - loss: 0.2833 - categorical_accuracy: 0.9135
26784/60000 [============>.................] - ETA: 1:00 - loss: 0.2830 - categorical_accuracy: 0.9136
26816/60000 [============>.................] - ETA: 1:00 - loss: 0.2827 - categorical_accuracy: 0.9137
26848/60000 [============>.................] - ETA: 1:00 - loss: 0.2824 - categorical_accuracy: 0.9138
26880/60000 [============>.................] - ETA: 1:00 - loss: 0.2821 - categorical_accuracy: 0.9139
26912/60000 [============>.................] - ETA: 1:00 - loss: 0.2818 - categorical_accuracy: 0.9140
26944/60000 [============>.................] - ETA: 1:00 - loss: 0.2816 - categorical_accuracy: 0.9141
26976/60000 [============>.................] - ETA: 59s - loss: 0.2813 - categorical_accuracy: 0.9141 
27008/60000 [============>.................] - ETA: 59s - loss: 0.2811 - categorical_accuracy: 0.9141
27040/60000 [============>.................] - ETA: 59s - loss: 0.2809 - categorical_accuracy: 0.9142
27072/60000 [============>.................] - ETA: 59s - loss: 0.2806 - categorical_accuracy: 0.9143
27104/60000 [============>.................] - ETA: 59s - loss: 0.2804 - categorical_accuracy: 0.9143
27136/60000 [============>.................] - ETA: 59s - loss: 0.2803 - categorical_accuracy: 0.9144
27168/60000 [============>.................] - ETA: 59s - loss: 0.2803 - categorical_accuracy: 0.9144
27200/60000 [============>.................] - ETA: 59s - loss: 0.2801 - categorical_accuracy: 0.9144
27232/60000 [============>.................] - ETA: 59s - loss: 0.2798 - categorical_accuracy: 0.9145
27264/60000 [============>.................] - ETA: 59s - loss: 0.2795 - categorical_accuracy: 0.9146
27296/60000 [============>.................] - ETA: 59s - loss: 0.2795 - categorical_accuracy: 0.9146
27328/60000 [============>.................] - ETA: 59s - loss: 0.2792 - categorical_accuracy: 0.9147
27360/60000 [============>.................] - ETA: 59s - loss: 0.2789 - categorical_accuracy: 0.9148
27392/60000 [============>.................] - ETA: 59s - loss: 0.2788 - categorical_accuracy: 0.9149
27424/60000 [============>.................] - ETA: 59s - loss: 0.2787 - categorical_accuracy: 0.9150
27456/60000 [============>.................] - ETA: 59s - loss: 0.2786 - categorical_accuracy: 0.9150
27488/60000 [============>.................] - ETA: 59s - loss: 0.2784 - categorical_accuracy: 0.9150
27520/60000 [============>.................] - ETA: 58s - loss: 0.2783 - categorical_accuracy: 0.9150
27552/60000 [============>.................] - ETA: 58s - loss: 0.2783 - categorical_accuracy: 0.9150
27584/60000 [============>.................] - ETA: 58s - loss: 0.2782 - categorical_accuracy: 0.9150
27616/60000 [============>.................] - ETA: 58s - loss: 0.2781 - categorical_accuracy: 0.9151
27648/60000 [============>.................] - ETA: 58s - loss: 0.2779 - categorical_accuracy: 0.9151
27680/60000 [============>.................] - ETA: 58s - loss: 0.2777 - categorical_accuracy: 0.9152
27712/60000 [============>.................] - ETA: 58s - loss: 0.2777 - categorical_accuracy: 0.9151
27744/60000 [============>.................] - ETA: 58s - loss: 0.2776 - categorical_accuracy: 0.9151
27776/60000 [============>.................] - ETA: 58s - loss: 0.2774 - categorical_accuracy: 0.9152
27808/60000 [============>.................] - ETA: 58s - loss: 0.2771 - categorical_accuracy: 0.9153
27840/60000 [============>.................] - ETA: 58s - loss: 0.2770 - categorical_accuracy: 0.9153
27872/60000 [============>.................] - ETA: 58s - loss: 0.2767 - categorical_accuracy: 0.9154
27904/60000 [============>.................] - ETA: 58s - loss: 0.2764 - categorical_accuracy: 0.9155
27936/60000 [============>.................] - ETA: 58s - loss: 0.2762 - categorical_accuracy: 0.9155
27968/60000 [============>.................] - ETA: 58s - loss: 0.2760 - categorical_accuracy: 0.9155
28000/60000 [=============>................] - ETA: 58s - loss: 0.2760 - categorical_accuracy: 0.9155
28032/60000 [=============>................] - ETA: 58s - loss: 0.2762 - categorical_accuracy: 0.9155
28064/60000 [=============>................] - ETA: 57s - loss: 0.2760 - categorical_accuracy: 0.9156
28096/60000 [=============>................] - ETA: 57s - loss: 0.2760 - categorical_accuracy: 0.9156
28128/60000 [=============>................] - ETA: 57s - loss: 0.2758 - categorical_accuracy: 0.9157
28160/60000 [=============>................] - ETA: 57s - loss: 0.2755 - categorical_accuracy: 0.9158
28192/60000 [=============>................] - ETA: 57s - loss: 0.2753 - categorical_accuracy: 0.9159
28224/60000 [=============>................] - ETA: 57s - loss: 0.2751 - categorical_accuracy: 0.9159
28256/60000 [=============>................] - ETA: 57s - loss: 0.2749 - categorical_accuracy: 0.9159
28288/60000 [=============>................] - ETA: 57s - loss: 0.2748 - categorical_accuracy: 0.9160
28320/60000 [=============>................] - ETA: 57s - loss: 0.2745 - categorical_accuracy: 0.9161
28352/60000 [=============>................] - ETA: 57s - loss: 0.2743 - categorical_accuracy: 0.9162
28384/60000 [=============>................] - ETA: 57s - loss: 0.2740 - categorical_accuracy: 0.9163
28416/60000 [=============>................] - ETA: 57s - loss: 0.2738 - categorical_accuracy: 0.9163
28448/60000 [=============>................] - ETA: 57s - loss: 0.2737 - categorical_accuracy: 0.9164
28480/60000 [=============>................] - ETA: 57s - loss: 0.2735 - categorical_accuracy: 0.9164
28512/60000 [=============>................] - ETA: 57s - loss: 0.2735 - categorical_accuracy: 0.9164
28544/60000 [=============>................] - ETA: 57s - loss: 0.2733 - categorical_accuracy: 0.9165
28576/60000 [=============>................] - ETA: 57s - loss: 0.2732 - categorical_accuracy: 0.9165
28608/60000 [=============>................] - ETA: 57s - loss: 0.2730 - categorical_accuracy: 0.9166
28640/60000 [=============>................] - ETA: 56s - loss: 0.2730 - categorical_accuracy: 0.9166
28672/60000 [=============>................] - ETA: 56s - loss: 0.2729 - categorical_accuracy: 0.9166
28704/60000 [=============>................] - ETA: 56s - loss: 0.2726 - categorical_accuracy: 0.9167
28736/60000 [=============>................] - ETA: 56s - loss: 0.2724 - categorical_accuracy: 0.9168
28768/60000 [=============>................] - ETA: 56s - loss: 0.2723 - categorical_accuracy: 0.9168
28800/60000 [=============>................] - ETA: 56s - loss: 0.2722 - categorical_accuracy: 0.9168
28832/60000 [=============>................] - ETA: 56s - loss: 0.2719 - categorical_accuracy: 0.9169
28864/60000 [=============>................] - ETA: 56s - loss: 0.2717 - categorical_accuracy: 0.9170
28896/60000 [=============>................] - ETA: 56s - loss: 0.2714 - categorical_accuracy: 0.9171
28928/60000 [=============>................] - ETA: 56s - loss: 0.2712 - categorical_accuracy: 0.9171
28960/60000 [=============>................] - ETA: 56s - loss: 0.2711 - categorical_accuracy: 0.9172
28992/60000 [=============>................] - ETA: 56s - loss: 0.2708 - categorical_accuracy: 0.9173
29024/60000 [=============>................] - ETA: 56s - loss: 0.2707 - categorical_accuracy: 0.9173
29056/60000 [=============>................] - ETA: 56s - loss: 0.2705 - categorical_accuracy: 0.9174
29088/60000 [=============>................] - ETA: 56s - loss: 0.2703 - categorical_accuracy: 0.9175
29120/60000 [=============>................] - ETA: 56s - loss: 0.2700 - categorical_accuracy: 0.9175
29152/60000 [=============>................] - ETA: 55s - loss: 0.2699 - categorical_accuracy: 0.9176
29184/60000 [=============>................] - ETA: 55s - loss: 0.2696 - categorical_accuracy: 0.9177
29216/60000 [=============>................] - ETA: 55s - loss: 0.2694 - categorical_accuracy: 0.9178
29248/60000 [=============>................] - ETA: 55s - loss: 0.2692 - categorical_accuracy: 0.9178
29280/60000 [=============>................] - ETA: 55s - loss: 0.2692 - categorical_accuracy: 0.9178
29312/60000 [=============>................] - ETA: 55s - loss: 0.2690 - categorical_accuracy: 0.9179
29344/60000 [=============>................] - ETA: 55s - loss: 0.2691 - categorical_accuracy: 0.9179
29376/60000 [=============>................] - ETA: 55s - loss: 0.2690 - categorical_accuracy: 0.9179
29408/60000 [=============>................] - ETA: 55s - loss: 0.2688 - categorical_accuracy: 0.9180
29440/60000 [=============>................] - ETA: 55s - loss: 0.2689 - categorical_accuracy: 0.9180
29472/60000 [=============>................] - ETA: 55s - loss: 0.2687 - categorical_accuracy: 0.9180
29504/60000 [=============>................] - ETA: 55s - loss: 0.2685 - categorical_accuracy: 0.9181
29536/60000 [=============>................] - ETA: 55s - loss: 0.2683 - categorical_accuracy: 0.9181
29568/60000 [=============>................] - ETA: 55s - loss: 0.2682 - categorical_accuracy: 0.9182
29600/60000 [=============>................] - ETA: 55s - loss: 0.2682 - categorical_accuracy: 0.9182
29632/60000 [=============>................] - ETA: 55s - loss: 0.2681 - categorical_accuracy: 0.9183
29664/60000 [=============>................] - ETA: 55s - loss: 0.2680 - categorical_accuracy: 0.9183
29696/60000 [=============>................] - ETA: 55s - loss: 0.2678 - categorical_accuracy: 0.9183
29728/60000 [=============>................] - ETA: 54s - loss: 0.2676 - categorical_accuracy: 0.9184
29760/60000 [=============>................] - ETA: 54s - loss: 0.2674 - categorical_accuracy: 0.9185
29792/60000 [=============>................] - ETA: 54s - loss: 0.2674 - categorical_accuracy: 0.9185
29824/60000 [=============>................] - ETA: 54s - loss: 0.2672 - categorical_accuracy: 0.9186
29856/60000 [=============>................] - ETA: 54s - loss: 0.2671 - categorical_accuracy: 0.9186
29888/60000 [=============>................] - ETA: 54s - loss: 0.2669 - categorical_accuracy: 0.9187
29920/60000 [=============>................] - ETA: 54s - loss: 0.2666 - categorical_accuracy: 0.9188
29952/60000 [=============>................] - ETA: 54s - loss: 0.2664 - categorical_accuracy: 0.9189
29984/60000 [=============>................] - ETA: 54s - loss: 0.2661 - categorical_accuracy: 0.9190
30016/60000 [==============>...............] - ETA: 54s - loss: 0.2662 - categorical_accuracy: 0.9189
30048/60000 [==============>...............] - ETA: 54s - loss: 0.2662 - categorical_accuracy: 0.9189
30080/60000 [==============>...............] - ETA: 54s - loss: 0.2660 - categorical_accuracy: 0.9190
30112/60000 [==============>...............] - ETA: 54s - loss: 0.2658 - categorical_accuracy: 0.9190
30144/60000 [==============>...............] - ETA: 54s - loss: 0.2656 - categorical_accuracy: 0.9191
30176/60000 [==============>...............] - ETA: 54s - loss: 0.2654 - categorical_accuracy: 0.9192
30208/60000 [==============>...............] - ETA: 54s - loss: 0.2651 - categorical_accuracy: 0.9193
30240/60000 [==============>...............] - ETA: 54s - loss: 0.2650 - categorical_accuracy: 0.9193
30272/60000 [==============>...............] - ETA: 54s - loss: 0.2648 - categorical_accuracy: 0.9194
30304/60000 [==============>...............] - ETA: 53s - loss: 0.2646 - categorical_accuracy: 0.9194
30336/60000 [==============>...............] - ETA: 53s - loss: 0.2644 - categorical_accuracy: 0.9195
30368/60000 [==============>...............] - ETA: 53s - loss: 0.2643 - categorical_accuracy: 0.9195
30400/60000 [==============>...............] - ETA: 53s - loss: 0.2641 - categorical_accuracy: 0.9196
30432/60000 [==============>...............] - ETA: 53s - loss: 0.2639 - categorical_accuracy: 0.9197
30464/60000 [==============>...............] - ETA: 53s - loss: 0.2637 - categorical_accuracy: 0.9197
30496/60000 [==============>...............] - ETA: 53s - loss: 0.2639 - categorical_accuracy: 0.9197
30528/60000 [==============>...............] - ETA: 53s - loss: 0.2637 - categorical_accuracy: 0.9197
30560/60000 [==============>...............] - ETA: 53s - loss: 0.2636 - categorical_accuracy: 0.9198
30592/60000 [==============>...............] - ETA: 53s - loss: 0.2635 - categorical_accuracy: 0.9198
30624/60000 [==============>...............] - ETA: 53s - loss: 0.2633 - categorical_accuracy: 0.9199
30656/60000 [==============>...............] - ETA: 53s - loss: 0.2631 - categorical_accuracy: 0.9200
30688/60000 [==============>...............] - ETA: 53s - loss: 0.2629 - categorical_accuracy: 0.9200
30720/60000 [==============>...............] - ETA: 53s - loss: 0.2627 - categorical_accuracy: 0.9201
30752/60000 [==============>...............] - ETA: 53s - loss: 0.2625 - categorical_accuracy: 0.9201
30784/60000 [==============>...............] - ETA: 53s - loss: 0.2622 - categorical_accuracy: 0.9202
30816/60000 [==============>...............] - ETA: 53s - loss: 0.2621 - categorical_accuracy: 0.9202
30848/60000 [==============>...............] - ETA: 53s - loss: 0.2618 - categorical_accuracy: 0.9203
30880/60000 [==============>...............] - ETA: 52s - loss: 0.2616 - categorical_accuracy: 0.9204
30912/60000 [==============>...............] - ETA: 52s - loss: 0.2613 - categorical_accuracy: 0.9205
30944/60000 [==============>...............] - ETA: 52s - loss: 0.2612 - categorical_accuracy: 0.9205
30976/60000 [==============>...............] - ETA: 52s - loss: 0.2612 - categorical_accuracy: 0.9206
31008/60000 [==============>...............] - ETA: 52s - loss: 0.2610 - categorical_accuracy: 0.9206
31040/60000 [==============>...............] - ETA: 52s - loss: 0.2608 - categorical_accuracy: 0.9207
31072/60000 [==============>...............] - ETA: 52s - loss: 0.2607 - categorical_accuracy: 0.9207
31104/60000 [==============>...............] - ETA: 52s - loss: 0.2606 - categorical_accuracy: 0.9207
31136/60000 [==============>...............] - ETA: 52s - loss: 0.2606 - categorical_accuracy: 0.9207
31168/60000 [==============>...............] - ETA: 52s - loss: 0.2604 - categorical_accuracy: 0.9208
31200/60000 [==============>...............] - ETA: 52s - loss: 0.2604 - categorical_accuracy: 0.9208
31232/60000 [==============>...............] - ETA: 52s - loss: 0.2601 - categorical_accuracy: 0.9209
31264/60000 [==============>...............] - ETA: 52s - loss: 0.2601 - categorical_accuracy: 0.9209
31296/60000 [==============>...............] - ETA: 52s - loss: 0.2600 - categorical_accuracy: 0.9209
31328/60000 [==============>...............] - ETA: 52s - loss: 0.2598 - categorical_accuracy: 0.9210
31360/60000 [==============>...............] - ETA: 52s - loss: 0.2597 - categorical_accuracy: 0.9210
31392/60000 [==============>...............] - ETA: 52s - loss: 0.2595 - categorical_accuracy: 0.9211
31424/60000 [==============>...............] - ETA: 51s - loss: 0.2594 - categorical_accuracy: 0.9211
31456/60000 [==============>...............] - ETA: 51s - loss: 0.2592 - categorical_accuracy: 0.9212
31488/60000 [==============>...............] - ETA: 51s - loss: 0.2591 - categorical_accuracy: 0.9212
31520/60000 [==============>...............] - ETA: 51s - loss: 0.2590 - categorical_accuracy: 0.9212
31552/60000 [==============>...............] - ETA: 51s - loss: 0.2588 - categorical_accuracy: 0.9213
31584/60000 [==============>...............] - ETA: 51s - loss: 0.2585 - categorical_accuracy: 0.9214
31616/60000 [==============>...............] - ETA: 51s - loss: 0.2583 - categorical_accuracy: 0.9214
31648/60000 [==============>...............] - ETA: 51s - loss: 0.2583 - categorical_accuracy: 0.9214
31680/60000 [==============>...............] - ETA: 51s - loss: 0.2582 - categorical_accuracy: 0.9215
31712/60000 [==============>...............] - ETA: 51s - loss: 0.2581 - categorical_accuracy: 0.9215
31744/60000 [==============>...............] - ETA: 51s - loss: 0.2578 - categorical_accuracy: 0.9216
31776/60000 [==============>...............] - ETA: 51s - loss: 0.2577 - categorical_accuracy: 0.9216
31808/60000 [==============>...............] - ETA: 51s - loss: 0.2575 - categorical_accuracy: 0.9217
31840/60000 [==============>...............] - ETA: 51s - loss: 0.2573 - categorical_accuracy: 0.9217
31872/60000 [==============>...............] - ETA: 51s - loss: 0.2572 - categorical_accuracy: 0.9217
31904/60000 [==============>...............] - ETA: 51s - loss: 0.2572 - categorical_accuracy: 0.9217
31936/60000 [==============>...............] - ETA: 51s - loss: 0.2570 - categorical_accuracy: 0.9217
31968/60000 [==============>...............] - ETA: 51s - loss: 0.2569 - categorical_accuracy: 0.9218
32000/60000 [===============>..............] - ETA: 50s - loss: 0.2568 - categorical_accuracy: 0.9218
32032/60000 [===============>..............] - ETA: 50s - loss: 0.2566 - categorical_accuracy: 0.9219
32064/60000 [===============>..............] - ETA: 50s - loss: 0.2563 - categorical_accuracy: 0.9220
32096/60000 [===============>..............] - ETA: 50s - loss: 0.2561 - categorical_accuracy: 0.9220
32128/60000 [===============>..............] - ETA: 50s - loss: 0.2561 - categorical_accuracy: 0.9221
32160/60000 [===============>..............] - ETA: 50s - loss: 0.2560 - categorical_accuracy: 0.9221
32192/60000 [===============>..............] - ETA: 50s - loss: 0.2561 - categorical_accuracy: 0.9221
32224/60000 [===============>..............] - ETA: 50s - loss: 0.2560 - categorical_accuracy: 0.9221
32256/60000 [===============>..............] - ETA: 50s - loss: 0.2558 - categorical_accuracy: 0.9222
32288/60000 [===============>..............] - ETA: 50s - loss: 0.2557 - categorical_accuracy: 0.9222
32320/60000 [===============>..............] - ETA: 50s - loss: 0.2556 - categorical_accuracy: 0.9222
32352/60000 [===============>..............] - ETA: 50s - loss: 0.2554 - categorical_accuracy: 0.9223
32384/60000 [===============>..............] - ETA: 50s - loss: 0.2555 - categorical_accuracy: 0.9222
32416/60000 [===============>..............] - ETA: 50s - loss: 0.2553 - categorical_accuracy: 0.9223
32448/60000 [===============>..............] - ETA: 50s - loss: 0.2553 - categorical_accuracy: 0.9223
32480/60000 [===============>..............] - ETA: 50s - loss: 0.2551 - categorical_accuracy: 0.9223
32512/60000 [===============>..............] - ETA: 50s - loss: 0.2549 - categorical_accuracy: 0.9224
32544/60000 [===============>..............] - ETA: 49s - loss: 0.2547 - categorical_accuracy: 0.9224
32576/60000 [===============>..............] - ETA: 49s - loss: 0.2545 - categorical_accuracy: 0.9225
32608/60000 [===============>..............] - ETA: 49s - loss: 0.2544 - categorical_accuracy: 0.9225
32640/60000 [===============>..............] - ETA: 49s - loss: 0.2541 - categorical_accuracy: 0.9226
32672/60000 [===============>..............] - ETA: 49s - loss: 0.2539 - categorical_accuracy: 0.9226
32704/60000 [===============>..............] - ETA: 49s - loss: 0.2538 - categorical_accuracy: 0.9226
32736/60000 [===============>..............] - ETA: 49s - loss: 0.2538 - categorical_accuracy: 0.9226
32768/60000 [===============>..............] - ETA: 49s - loss: 0.2537 - categorical_accuracy: 0.9226
32800/60000 [===============>..............] - ETA: 49s - loss: 0.2535 - categorical_accuracy: 0.9227
32832/60000 [===============>..............] - ETA: 49s - loss: 0.2534 - categorical_accuracy: 0.9227
32864/60000 [===============>..............] - ETA: 49s - loss: 0.2533 - categorical_accuracy: 0.9228
32896/60000 [===============>..............] - ETA: 49s - loss: 0.2531 - categorical_accuracy: 0.9228
32928/60000 [===============>..............] - ETA: 49s - loss: 0.2530 - categorical_accuracy: 0.9228
32960/60000 [===============>..............] - ETA: 49s - loss: 0.2529 - categorical_accuracy: 0.9228
32992/60000 [===============>..............] - ETA: 49s - loss: 0.2528 - categorical_accuracy: 0.9229
33024/60000 [===============>..............] - ETA: 49s - loss: 0.2525 - categorical_accuracy: 0.9229
33056/60000 [===============>..............] - ETA: 49s - loss: 0.2524 - categorical_accuracy: 0.9230
33088/60000 [===============>..............] - ETA: 49s - loss: 0.2525 - categorical_accuracy: 0.9230
33120/60000 [===============>..............] - ETA: 48s - loss: 0.2524 - categorical_accuracy: 0.9230
33152/60000 [===============>..............] - ETA: 48s - loss: 0.2523 - categorical_accuracy: 0.9231
33184/60000 [===============>..............] - ETA: 48s - loss: 0.2521 - categorical_accuracy: 0.9232
33216/60000 [===============>..............] - ETA: 48s - loss: 0.2519 - categorical_accuracy: 0.9232
33248/60000 [===============>..............] - ETA: 48s - loss: 0.2518 - categorical_accuracy: 0.9233
33280/60000 [===============>..............] - ETA: 48s - loss: 0.2516 - categorical_accuracy: 0.9233
33312/60000 [===============>..............] - ETA: 48s - loss: 0.2515 - categorical_accuracy: 0.9234
33344/60000 [===============>..............] - ETA: 48s - loss: 0.2512 - categorical_accuracy: 0.9235
33376/60000 [===============>..............] - ETA: 48s - loss: 0.2511 - categorical_accuracy: 0.9235
33408/60000 [===============>..............] - ETA: 48s - loss: 0.2509 - categorical_accuracy: 0.9236
33440/60000 [===============>..............] - ETA: 48s - loss: 0.2508 - categorical_accuracy: 0.9236
33472/60000 [===============>..............] - ETA: 48s - loss: 0.2506 - categorical_accuracy: 0.9237
33504/60000 [===============>..............] - ETA: 48s - loss: 0.2504 - categorical_accuracy: 0.9237
33536/60000 [===============>..............] - ETA: 48s - loss: 0.2503 - categorical_accuracy: 0.9238
33568/60000 [===============>..............] - ETA: 48s - loss: 0.2501 - categorical_accuracy: 0.9238
33600/60000 [===============>..............] - ETA: 48s - loss: 0.2500 - categorical_accuracy: 0.9238
33632/60000 [===============>..............] - ETA: 48s - loss: 0.2500 - categorical_accuracy: 0.9238
33664/60000 [===============>..............] - ETA: 48s - loss: 0.2499 - categorical_accuracy: 0.9238
33696/60000 [===============>..............] - ETA: 47s - loss: 0.2498 - categorical_accuracy: 0.9238
33728/60000 [===============>..............] - ETA: 47s - loss: 0.2498 - categorical_accuracy: 0.9239
33760/60000 [===============>..............] - ETA: 47s - loss: 0.2496 - categorical_accuracy: 0.9239
33792/60000 [===============>..............] - ETA: 47s - loss: 0.2494 - categorical_accuracy: 0.9239
33824/60000 [===============>..............] - ETA: 47s - loss: 0.2493 - categorical_accuracy: 0.9240
33856/60000 [===============>..............] - ETA: 47s - loss: 0.2491 - categorical_accuracy: 0.9241
33888/60000 [===============>..............] - ETA: 47s - loss: 0.2491 - categorical_accuracy: 0.9241
33920/60000 [===============>..............] - ETA: 47s - loss: 0.2493 - categorical_accuracy: 0.9241
33952/60000 [===============>..............] - ETA: 47s - loss: 0.2492 - categorical_accuracy: 0.9242
33984/60000 [===============>..............] - ETA: 47s - loss: 0.2490 - categorical_accuracy: 0.9242
34016/60000 [================>.............] - ETA: 47s - loss: 0.2488 - categorical_accuracy: 0.9243
34048/60000 [================>.............] - ETA: 47s - loss: 0.2486 - categorical_accuracy: 0.9244
34080/60000 [================>.............] - ETA: 47s - loss: 0.2485 - categorical_accuracy: 0.9244
34112/60000 [================>.............] - ETA: 47s - loss: 0.2483 - categorical_accuracy: 0.9245
34144/60000 [================>.............] - ETA: 47s - loss: 0.2484 - categorical_accuracy: 0.9245
34176/60000 [================>.............] - ETA: 47s - loss: 0.2482 - categorical_accuracy: 0.9245
34208/60000 [================>.............] - ETA: 47s - loss: 0.2480 - categorical_accuracy: 0.9246
34240/60000 [================>.............] - ETA: 46s - loss: 0.2478 - categorical_accuracy: 0.9247
34272/60000 [================>.............] - ETA: 46s - loss: 0.2476 - categorical_accuracy: 0.9247
34304/60000 [================>.............] - ETA: 46s - loss: 0.2475 - categorical_accuracy: 0.9247
34336/60000 [================>.............] - ETA: 46s - loss: 0.2474 - categorical_accuracy: 0.9248
34368/60000 [================>.............] - ETA: 46s - loss: 0.2472 - categorical_accuracy: 0.9248
34400/60000 [================>.............] - ETA: 46s - loss: 0.2470 - categorical_accuracy: 0.9249
34432/60000 [================>.............] - ETA: 46s - loss: 0.2470 - categorical_accuracy: 0.9249
34464/60000 [================>.............] - ETA: 46s - loss: 0.2468 - categorical_accuracy: 0.9249
34496/60000 [================>.............] - ETA: 46s - loss: 0.2467 - categorical_accuracy: 0.9249
34528/60000 [================>.............] - ETA: 46s - loss: 0.2466 - categorical_accuracy: 0.9249
34560/60000 [================>.............] - ETA: 46s - loss: 0.2464 - categorical_accuracy: 0.9250
34592/60000 [================>.............] - ETA: 46s - loss: 0.2462 - categorical_accuracy: 0.9250
34624/60000 [================>.............] - ETA: 46s - loss: 0.2460 - categorical_accuracy: 0.9251
34656/60000 [================>.............] - ETA: 46s - loss: 0.2458 - categorical_accuracy: 0.9252
34688/60000 [================>.............] - ETA: 46s - loss: 0.2456 - categorical_accuracy: 0.9252
34720/60000 [================>.............] - ETA: 46s - loss: 0.2456 - categorical_accuracy: 0.9253
34752/60000 [================>.............] - ETA: 46s - loss: 0.2454 - categorical_accuracy: 0.9253
34784/60000 [================>.............] - ETA: 46s - loss: 0.2453 - categorical_accuracy: 0.9253
34816/60000 [================>.............] - ETA: 45s - loss: 0.2453 - categorical_accuracy: 0.9253
34848/60000 [================>.............] - ETA: 45s - loss: 0.2453 - categorical_accuracy: 0.9253
34880/60000 [================>.............] - ETA: 45s - loss: 0.2451 - categorical_accuracy: 0.9254
34912/60000 [================>.............] - ETA: 45s - loss: 0.2449 - categorical_accuracy: 0.9254
34944/60000 [================>.............] - ETA: 45s - loss: 0.2448 - categorical_accuracy: 0.9255
34976/60000 [================>.............] - ETA: 45s - loss: 0.2446 - categorical_accuracy: 0.9255
35008/60000 [================>.............] - ETA: 45s - loss: 0.2444 - categorical_accuracy: 0.9256
35040/60000 [================>.............] - ETA: 45s - loss: 0.2443 - categorical_accuracy: 0.9256
35072/60000 [================>.............] - ETA: 45s - loss: 0.2441 - categorical_accuracy: 0.9257
35104/60000 [================>.............] - ETA: 45s - loss: 0.2439 - categorical_accuracy: 0.9257
35136/60000 [================>.............] - ETA: 45s - loss: 0.2438 - categorical_accuracy: 0.9257
35168/60000 [================>.............] - ETA: 45s - loss: 0.2437 - categorical_accuracy: 0.9258
35200/60000 [================>.............] - ETA: 45s - loss: 0.2436 - categorical_accuracy: 0.9258
35232/60000 [================>.............] - ETA: 45s - loss: 0.2434 - categorical_accuracy: 0.9259
35264/60000 [================>.............] - ETA: 45s - loss: 0.2434 - categorical_accuracy: 0.9258
35296/60000 [================>.............] - ETA: 45s - loss: 0.2435 - categorical_accuracy: 0.9258
35328/60000 [================>.............] - ETA: 45s - loss: 0.2433 - categorical_accuracy: 0.9259
35360/60000 [================>.............] - ETA: 44s - loss: 0.2431 - categorical_accuracy: 0.9260
35392/60000 [================>.............] - ETA: 44s - loss: 0.2430 - categorical_accuracy: 0.9260
35424/60000 [================>.............] - ETA: 44s - loss: 0.2429 - categorical_accuracy: 0.9260
35456/60000 [================>.............] - ETA: 44s - loss: 0.2427 - categorical_accuracy: 0.9261
35488/60000 [================>.............] - ETA: 44s - loss: 0.2425 - categorical_accuracy: 0.9261
35520/60000 [================>.............] - ETA: 44s - loss: 0.2423 - categorical_accuracy: 0.9262
35552/60000 [================>.............] - ETA: 44s - loss: 0.2422 - categorical_accuracy: 0.9262
35584/60000 [================>.............] - ETA: 44s - loss: 0.2421 - categorical_accuracy: 0.9263
35616/60000 [================>.............] - ETA: 44s - loss: 0.2421 - categorical_accuracy: 0.9263
35648/60000 [================>.............] - ETA: 44s - loss: 0.2422 - categorical_accuracy: 0.9263
35680/60000 [================>.............] - ETA: 44s - loss: 0.2423 - categorical_accuracy: 0.9263
35712/60000 [================>.............] - ETA: 44s - loss: 0.2421 - categorical_accuracy: 0.9264
35744/60000 [================>.............] - ETA: 44s - loss: 0.2420 - categorical_accuracy: 0.9264
35776/60000 [================>.............] - ETA: 44s - loss: 0.2420 - categorical_accuracy: 0.9264
35808/60000 [================>.............] - ETA: 44s - loss: 0.2418 - categorical_accuracy: 0.9265
35840/60000 [================>.............] - ETA: 44s - loss: 0.2416 - categorical_accuracy: 0.9265
35872/60000 [================>.............] - ETA: 44s - loss: 0.2414 - categorical_accuracy: 0.9266
35904/60000 [================>.............] - ETA: 43s - loss: 0.2413 - categorical_accuracy: 0.9266
35936/60000 [================>.............] - ETA: 43s - loss: 0.2411 - categorical_accuracy: 0.9267
35968/60000 [================>.............] - ETA: 43s - loss: 0.2409 - categorical_accuracy: 0.9267
36000/60000 [=================>............] - ETA: 43s - loss: 0.2409 - categorical_accuracy: 0.9267
36032/60000 [=================>............] - ETA: 43s - loss: 0.2408 - categorical_accuracy: 0.9267
36064/60000 [=================>............] - ETA: 43s - loss: 0.2408 - categorical_accuracy: 0.9267
36096/60000 [=================>............] - ETA: 43s - loss: 0.2407 - categorical_accuracy: 0.9267
36128/60000 [=================>............] - ETA: 43s - loss: 0.2406 - categorical_accuracy: 0.9267
36160/60000 [=================>............] - ETA: 43s - loss: 0.2405 - categorical_accuracy: 0.9267
36192/60000 [=================>............] - ETA: 43s - loss: 0.2403 - categorical_accuracy: 0.9268
36224/60000 [=================>............] - ETA: 43s - loss: 0.2402 - categorical_accuracy: 0.9268
36256/60000 [=================>............] - ETA: 43s - loss: 0.2402 - categorical_accuracy: 0.9268
36288/60000 [=================>............] - ETA: 43s - loss: 0.2402 - categorical_accuracy: 0.9268
36320/60000 [=================>............] - ETA: 43s - loss: 0.2400 - categorical_accuracy: 0.9269
36352/60000 [=================>............] - ETA: 43s - loss: 0.2398 - categorical_accuracy: 0.9269
36384/60000 [=================>............] - ETA: 43s - loss: 0.2397 - categorical_accuracy: 0.9270
36416/60000 [=================>............] - ETA: 43s - loss: 0.2397 - categorical_accuracy: 0.9270
36448/60000 [=================>............] - ETA: 43s - loss: 0.2395 - categorical_accuracy: 0.9271
36480/60000 [=================>............] - ETA: 42s - loss: 0.2393 - categorical_accuracy: 0.9271
36512/60000 [=================>............] - ETA: 42s - loss: 0.2391 - categorical_accuracy: 0.9272
36544/60000 [=================>............] - ETA: 42s - loss: 0.2392 - categorical_accuracy: 0.9272
36576/60000 [=================>............] - ETA: 42s - loss: 0.2390 - categorical_accuracy: 0.9272
36608/60000 [=================>............] - ETA: 42s - loss: 0.2389 - categorical_accuracy: 0.9273
36640/60000 [=================>............] - ETA: 42s - loss: 0.2388 - categorical_accuracy: 0.9273
36672/60000 [=================>............] - ETA: 42s - loss: 0.2387 - categorical_accuracy: 0.9273
36704/60000 [=================>............] - ETA: 42s - loss: 0.2387 - categorical_accuracy: 0.9274
36736/60000 [=================>............] - ETA: 42s - loss: 0.2387 - categorical_accuracy: 0.9273
36768/60000 [=================>............] - ETA: 42s - loss: 0.2385 - categorical_accuracy: 0.9274
36800/60000 [=================>............] - ETA: 42s - loss: 0.2386 - categorical_accuracy: 0.9274
36832/60000 [=================>............] - ETA: 42s - loss: 0.2384 - categorical_accuracy: 0.9274
36864/60000 [=================>............] - ETA: 42s - loss: 0.2383 - categorical_accuracy: 0.9275
36896/60000 [=================>............] - ETA: 42s - loss: 0.2382 - categorical_accuracy: 0.9275
36928/60000 [=================>............] - ETA: 42s - loss: 0.2380 - categorical_accuracy: 0.9276
36960/60000 [=================>............] - ETA: 42s - loss: 0.2380 - categorical_accuracy: 0.9276
36992/60000 [=================>............] - ETA: 42s - loss: 0.2378 - categorical_accuracy: 0.9276
37024/60000 [=================>............] - ETA: 41s - loss: 0.2376 - categorical_accuracy: 0.9277
37056/60000 [=================>............] - ETA: 41s - loss: 0.2374 - categorical_accuracy: 0.9277
37088/60000 [=================>............] - ETA: 41s - loss: 0.2373 - categorical_accuracy: 0.9277
37120/60000 [=================>............] - ETA: 41s - loss: 0.2371 - categorical_accuracy: 0.9278
37152/60000 [=================>............] - ETA: 41s - loss: 0.2370 - categorical_accuracy: 0.9279
37184/60000 [=================>............] - ETA: 41s - loss: 0.2369 - categorical_accuracy: 0.9279
37216/60000 [=================>............] - ETA: 41s - loss: 0.2367 - categorical_accuracy: 0.9279
37248/60000 [=================>............] - ETA: 41s - loss: 0.2366 - categorical_accuracy: 0.9280
37280/60000 [=================>............] - ETA: 41s - loss: 0.2365 - categorical_accuracy: 0.9280
37312/60000 [=================>............] - ETA: 41s - loss: 0.2363 - categorical_accuracy: 0.9280
37344/60000 [=================>............] - ETA: 41s - loss: 0.2363 - categorical_accuracy: 0.9280
37376/60000 [=================>............] - ETA: 41s - loss: 0.2362 - categorical_accuracy: 0.9281
37408/60000 [=================>............] - ETA: 41s - loss: 0.2360 - categorical_accuracy: 0.9281
37440/60000 [=================>............] - ETA: 41s - loss: 0.2360 - categorical_accuracy: 0.9281
37472/60000 [=================>............] - ETA: 41s - loss: 0.2358 - categorical_accuracy: 0.9282
37504/60000 [=================>............] - ETA: 41s - loss: 0.2357 - categorical_accuracy: 0.9282
37536/60000 [=================>............] - ETA: 41s - loss: 0.2356 - categorical_accuracy: 0.9282
37568/60000 [=================>............] - ETA: 40s - loss: 0.2356 - categorical_accuracy: 0.9282
37600/60000 [=================>............] - ETA: 40s - loss: 0.2356 - categorical_accuracy: 0.9282
37632/60000 [=================>............] - ETA: 40s - loss: 0.2357 - categorical_accuracy: 0.9283
37664/60000 [=================>............] - ETA: 40s - loss: 0.2358 - categorical_accuracy: 0.9282
37696/60000 [=================>............] - ETA: 40s - loss: 0.2358 - categorical_accuracy: 0.9282
37728/60000 [=================>............] - ETA: 40s - loss: 0.2356 - categorical_accuracy: 0.9283
37760/60000 [=================>............] - ETA: 40s - loss: 0.2354 - categorical_accuracy: 0.9283
37792/60000 [=================>............] - ETA: 40s - loss: 0.2354 - categorical_accuracy: 0.9283
37824/60000 [=================>............] - ETA: 40s - loss: 0.2353 - categorical_accuracy: 0.9284
37856/60000 [=================>............] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9284
37888/60000 [=================>............] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9284
37920/60000 [=================>............] - ETA: 40s - loss: 0.2349 - categorical_accuracy: 0.9285
37952/60000 [=================>............] - ETA: 40s - loss: 0.2348 - categorical_accuracy: 0.9285
37984/60000 [=================>............] - ETA: 40s - loss: 0.2350 - categorical_accuracy: 0.9285
38016/60000 [==================>...........] - ETA: 40s - loss: 0.2348 - categorical_accuracy: 0.9285
38048/60000 [==================>...........] - ETA: 40s - loss: 0.2347 - categorical_accuracy: 0.9286
38080/60000 [==================>...........] - ETA: 40s - loss: 0.2345 - categorical_accuracy: 0.9287
38112/60000 [==================>...........] - ETA: 39s - loss: 0.2344 - categorical_accuracy: 0.9287
38144/60000 [==================>...........] - ETA: 39s - loss: 0.2342 - categorical_accuracy: 0.9287
38176/60000 [==================>...........] - ETA: 39s - loss: 0.2341 - categorical_accuracy: 0.9288
38208/60000 [==================>...........] - ETA: 39s - loss: 0.2340 - categorical_accuracy: 0.9288
38240/60000 [==================>...........] - ETA: 39s - loss: 0.2338 - categorical_accuracy: 0.9289
38272/60000 [==================>...........] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9289
38304/60000 [==================>...........] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9289
38336/60000 [==================>...........] - ETA: 39s - loss: 0.2336 - categorical_accuracy: 0.9289
38368/60000 [==================>...........] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9290
38400/60000 [==================>...........] - ETA: 39s - loss: 0.2336 - categorical_accuracy: 0.9290
38432/60000 [==================>...........] - ETA: 39s - loss: 0.2338 - categorical_accuracy: 0.9289
38464/60000 [==================>...........] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9289
38496/60000 [==================>...........] - ETA: 39s - loss: 0.2336 - categorical_accuracy: 0.9290
38528/60000 [==================>...........] - ETA: 39s - loss: 0.2335 - categorical_accuracy: 0.9290
38560/60000 [==================>...........] - ETA: 39s - loss: 0.2334 - categorical_accuracy: 0.9290
38592/60000 [==================>...........] - ETA: 39s - loss: 0.2335 - categorical_accuracy: 0.9291
38624/60000 [==================>...........] - ETA: 39s - loss: 0.2334 - categorical_accuracy: 0.9291
38656/60000 [==================>...........] - ETA: 38s - loss: 0.2334 - categorical_accuracy: 0.9291
38688/60000 [==================>...........] - ETA: 38s - loss: 0.2333 - categorical_accuracy: 0.9291
38720/60000 [==================>...........] - ETA: 38s - loss: 0.2332 - categorical_accuracy: 0.9291
38752/60000 [==================>...........] - ETA: 38s - loss: 0.2330 - categorical_accuracy: 0.9292
38784/60000 [==================>...........] - ETA: 38s - loss: 0.2330 - categorical_accuracy: 0.9292
38816/60000 [==================>...........] - ETA: 38s - loss: 0.2329 - categorical_accuracy: 0.9293
38848/60000 [==================>...........] - ETA: 38s - loss: 0.2327 - categorical_accuracy: 0.9293
38880/60000 [==================>...........] - ETA: 38s - loss: 0.2328 - categorical_accuracy: 0.9293
38912/60000 [==================>...........] - ETA: 38s - loss: 0.2326 - categorical_accuracy: 0.9293
38944/60000 [==================>...........] - ETA: 38s - loss: 0.2324 - categorical_accuracy: 0.9294
38976/60000 [==================>...........] - ETA: 38s - loss: 0.2324 - categorical_accuracy: 0.9294
39008/60000 [==================>...........] - ETA: 38s - loss: 0.2323 - categorical_accuracy: 0.9295
39040/60000 [==================>...........] - ETA: 38s - loss: 0.2322 - categorical_accuracy: 0.9295
39072/60000 [==================>...........] - ETA: 38s - loss: 0.2320 - categorical_accuracy: 0.9296
39104/60000 [==================>...........] - ETA: 38s - loss: 0.2318 - categorical_accuracy: 0.9296
39136/60000 [==================>...........] - ETA: 38s - loss: 0.2317 - categorical_accuracy: 0.9297
39168/60000 [==================>...........] - ETA: 38s - loss: 0.2316 - categorical_accuracy: 0.9297
39200/60000 [==================>...........] - ETA: 38s - loss: 0.2314 - categorical_accuracy: 0.9297
39232/60000 [==================>...........] - ETA: 37s - loss: 0.2313 - categorical_accuracy: 0.9298
39264/60000 [==================>...........] - ETA: 37s - loss: 0.2311 - categorical_accuracy: 0.9298
39296/60000 [==================>...........] - ETA: 37s - loss: 0.2310 - categorical_accuracy: 0.9299
39328/60000 [==================>...........] - ETA: 37s - loss: 0.2308 - categorical_accuracy: 0.9299
39360/60000 [==================>...........] - ETA: 37s - loss: 0.2307 - categorical_accuracy: 0.9300
39392/60000 [==================>...........] - ETA: 37s - loss: 0.2307 - categorical_accuracy: 0.9299
39424/60000 [==================>...........] - ETA: 37s - loss: 0.2305 - categorical_accuracy: 0.9300
39456/60000 [==================>...........] - ETA: 37s - loss: 0.2304 - categorical_accuracy: 0.9300
39488/60000 [==================>...........] - ETA: 37s - loss: 0.2303 - categorical_accuracy: 0.9300
39520/60000 [==================>...........] - ETA: 37s - loss: 0.2302 - categorical_accuracy: 0.9301
39552/60000 [==================>...........] - ETA: 37s - loss: 0.2301 - categorical_accuracy: 0.9300
39584/60000 [==================>...........] - ETA: 37s - loss: 0.2300 - categorical_accuracy: 0.9301
39616/60000 [==================>...........] - ETA: 37s - loss: 0.2298 - categorical_accuracy: 0.9301
39648/60000 [==================>...........] - ETA: 37s - loss: 0.2298 - categorical_accuracy: 0.9301
39680/60000 [==================>...........] - ETA: 37s - loss: 0.2296 - categorical_accuracy: 0.9302
39712/60000 [==================>...........] - ETA: 37s - loss: 0.2295 - categorical_accuracy: 0.9302
39744/60000 [==================>...........] - ETA: 37s - loss: 0.2294 - categorical_accuracy: 0.9303
39776/60000 [==================>...........] - ETA: 36s - loss: 0.2293 - categorical_accuracy: 0.9303
39808/60000 [==================>...........] - ETA: 36s - loss: 0.2292 - categorical_accuracy: 0.9303
39840/60000 [==================>...........] - ETA: 36s - loss: 0.2293 - categorical_accuracy: 0.9303
39872/60000 [==================>...........] - ETA: 36s - loss: 0.2293 - categorical_accuracy: 0.9303
39904/60000 [==================>...........] - ETA: 36s - loss: 0.2291 - categorical_accuracy: 0.9304
39936/60000 [==================>...........] - ETA: 36s - loss: 0.2290 - categorical_accuracy: 0.9304
39968/60000 [==================>...........] - ETA: 36s - loss: 0.2288 - categorical_accuracy: 0.9305
40000/60000 [===================>..........] - ETA: 36s - loss: 0.2288 - categorical_accuracy: 0.9305
40032/60000 [===================>..........] - ETA: 36s - loss: 0.2289 - categorical_accuracy: 0.9305
40064/60000 [===================>..........] - ETA: 36s - loss: 0.2288 - categorical_accuracy: 0.9305
40096/60000 [===================>..........] - ETA: 36s - loss: 0.2288 - categorical_accuracy: 0.9305
40128/60000 [===================>..........] - ETA: 36s - loss: 0.2290 - categorical_accuracy: 0.9305
40160/60000 [===================>..........] - ETA: 36s - loss: 0.2289 - categorical_accuracy: 0.9305
40192/60000 [===================>..........] - ETA: 36s - loss: 0.2289 - categorical_accuracy: 0.9306
40224/60000 [===================>..........] - ETA: 36s - loss: 0.2290 - categorical_accuracy: 0.9305
40256/60000 [===================>..........] - ETA: 36s - loss: 0.2289 - categorical_accuracy: 0.9306
40288/60000 [===================>..........] - ETA: 36s - loss: 0.2288 - categorical_accuracy: 0.9306
40320/60000 [===================>..........] - ETA: 35s - loss: 0.2287 - categorical_accuracy: 0.9306
40352/60000 [===================>..........] - ETA: 35s - loss: 0.2286 - categorical_accuracy: 0.9306
40384/60000 [===================>..........] - ETA: 35s - loss: 0.2285 - categorical_accuracy: 0.9307
40416/60000 [===================>..........] - ETA: 35s - loss: 0.2283 - categorical_accuracy: 0.9307
40448/60000 [===================>..........] - ETA: 35s - loss: 0.2282 - categorical_accuracy: 0.9308
40480/60000 [===================>..........] - ETA: 35s - loss: 0.2280 - categorical_accuracy: 0.9308
40512/60000 [===================>..........] - ETA: 35s - loss: 0.2279 - categorical_accuracy: 0.9309
40544/60000 [===================>..........] - ETA: 35s - loss: 0.2277 - categorical_accuracy: 0.9309
40576/60000 [===================>..........] - ETA: 35s - loss: 0.2276 - categorical_accuracy: 0.9309
40608/60000 [===================>..........] - ETA: 35s - loss: 0.2276 - categorical_accuracy: 0.9309
40640/60000 [===================>..........] - ETA: 35s - loss: 0.2276 - categorical_accuracy: 0.9310
40672/60000 [===================>..........] - ETA: 35s - loss: 0.2275 - categorical_accuracy: 0.9310
40704/60000 [===================>..........] - ETA: 35s - loss: 0.2274 - categorical_accuracy: 0.9310
40736/60000 [===================>..........] - ETA: 35s - loss: 0.2274 - categorical_accuracy: 0.9310
40768/60000 [===================>..........] - ETA: 35s - loss: 0.2273 - categorical_accuracy: 0.9310
40800/60000 [===================>..........] - ETA: 35s - loss: 0.2272 - categorical_accuracy: 0.9310
40832/60000 [===================>..........] - ETA: 35s - loss: 0.2271 - categorical_accuracy: 0.9311
40864/60000 [===================>..........] - ETA: 34s - loss: 0.2269 - categorical_accuracy: 0.9311
40896/60000 [===================>..........] - ETA: 34s - loss: 0.2269 - categorical_accuracy: 0.9311
40928/60000 [===================>..........] - ETA: 34s - loss: 0.2268 - categorical_accuracy: 0.9312
40960/60000 [===================>..........] - ETA: 34s - loss: 0.2271 - categorical_accuracy: 0.9312
40992/60000 [===================>..........] - ETA: 34s - loss: 0.2269 - categorical_accuracy: 0.9312
41024/60000 [===================>..........] - ETA: 34s - loss: 0.2268 - categorical_accuracy: 0.9313
41056/60000 [===================>..........] - ETA: 34s - loss: 0.2268 - categorical_accuracy: 0.9313
41088/60000 [===================>..........] - ETA: 34s - loss: 0.2267 - categorical_accuracy: 0.9313
41120/60000 [===================>..........] - ETA: 34s - loss: 0.2267 - categorical_accuracy: 0.9313
41152/60000 [===================>..........] - ETA: 34s - loss: 0.2266 - categorical_accuracy: 0.9313
41184/60000 [===================>..........] - ETA: 34s - loss: 0.2264 - categorical_accuracy: 0.9314
41216/60000 [===================>..........] - ETA: 34s - loss: 0.2263 - categorical_accuracy: 0.9314
41248/60000 [===================>..........] - ETA: 34s - loss: 0.2262 - categorical_accuracy: 0.9315
41280/60000 [===================>..........] - ETA: 34s - loss: 0.2263 - categorical_accuracy: 0.9315
41312/60000 [===================>..........] - ETA: 34s - loss: 0.2262 - categorical_accuracy: 0.9315
41344/60000 [===================>..........] - ETA: 34s - loss: 0.2262 - categorical_accuracy: 0.9315
41376/60000 [===================>..........] - ETA: 34s - loss: 0.2262 - categorical_accuracy: 0.9315
41408/60000 [===================>..........] - ETA: 33s - loss: 0.2261 - categorical_accuracy: 0.9315
41440/60000 [===================>..........] - ETA: 33s - loss: 0.2259 - categorical_accuracy: 0.9315
41472/60000 [===================>..........] - ETA: 33s - loss: 0.2258 - categorical_accuracy: 0.9316
41504/60000 [===================>..........] - ETA: 33s - loss: 0.2256 - categorical_accuracy: 0.9316
41536/60000 [===================>..........] - ETA: 33s - loss: 0.2255 - categorical_accuracy: 0.9317
41568/60000 [===================>..........] - ETA: 33s - loss: 0.2253 - categorical_accuracy: 0.9318
41600/60000 [===================>..........] - ETA: 33s - loss: 0.2253 - categorical_accuracy: 0.9318
41632/60000 [===================>..........] - ETA: 33s - loss: 0.2251 - categorical_accuracy: 0.9318
41664/60000 [===================>..........] - ETA: 33s - loss: 0.2250 - categorical_accuracy: 0.9318
41696/60000 [===================>..........] - ETA: 33s - loss: 0.2249 - categorical_accuracy: 0.9319
41728/60000 [===================>..........] - ETA: 33s - loss: 0.2248 - categorical_accuracy: 0.9319
41760/60000 [===================>..........] - ETA: 33s - loss: 0.2247 - categorical_accuracy: 0.9319
41792/60000 [===================>..........] - ETA: 33s - loss: 0.2246 - categorical_accuracy: 0.9320
41824/60000 [===================>..........] - ETA: 33s - loss: 0.2245 - categorical_accuracy: 0.9320
41856/60000 [===================>..........] - ETA: 33s - loss: 0.2243 - categorical_accuracy: 0.9321
41888/60000 [===================>..........] - ETA: 33s - loss: 0.2243 - categorical_accuracy: 0.9321
41920/60000 [===================>..........] - ETA: 33s - loss: 0.2242 - categorical_accuracy: 0.9321
41952/60000 [===================>..........] - ETA: 32s - loss: 0.2240 - categorical_accuracy: 0.9321
41984/60000 [===================>..........] - ETA: 32s - loss: 0.2240 - categorical_accuracy: 0.9322
42016/60000 [====================>.........] - ETA: 32s - loss: 0.2239 - categorical_accuracy: 0.9322
42048/60000 [====================>.........] - ETA: 32s - loss: 0.2238 - categorical_accuracy: 0.9322
42080/60000 [====================>.........] - ETA: 32s - loss: 0.2236 - categorical_accuracy: 0.9323
42112/60000 [====================>.........] - ETA: 32s - loss: 0.2236 - categorical_accuracy: 0.9323
42144/60000 [====================>.........] - ETA: 32s - loss: 0.2234 - categorical_accuracy: 0.9324
42176/60000 [====================>.........] - ETA: 32s - loss: 0.2234 - categorical_accuracy: 0.9324
42208/60000 [====================>.........] - ETA: 32s - loss: 0.2233 - categorical_accuracy: 0.9324
42240/60000 [====================>.........] - ETA: 32s - loss: 0.2232 - categorical_accuracy: 0.9324
42272/60000 [====================>.........] - ETA: 32s - loss: 0.2231 - categorical_accuracy: 0.9324
42304/60000 [====================>.........] - ETA: 32s - loss: 0.2231 - categorical_accuracy: 0.9325
42336/60000 [====================>.........] - ETA: 32s - loss: 0.2230 - categorical_accuracy: 0.9325
42368/60000 [====================>.........] - ETA: 32s - loss: 0.2228 - categorical_accuracy: 0.9326
42400/60000 [====================>.........] - ETA: 32s - loss: 0.2228 - categorical_accuracy: 0.9326
42432/60000 [====================>.........] - ETA: 32s - loss: 0.2228 - categorical_accuracy: 0.9326
42464/60000 [====================>.........] - ETA: 32s - loss: 0.2227 - categorical_accuracy: 0.9326
42496/60000 [====================>.........] - ETA: 31s - loss: 0.2226 - categorical_accuracy: 0.9327
42528/60000 [====================>.........] - ETA: 31s - loss: 0.2225 - categorical_accuracy: 0.9327
42560/60000 [====================>.........] - ETA: 31s - loss: 0.2223 - categorical_accuracy: 0.9327
42592/60000 [====================>.........] - ETA: 31s - loss: 0.2223 - categorical_accuracy: 0.9328
42624/60000 [====================>.........] - ETA: 31s - loss: 0.2223 - categorical_accuracy: 0.9328
42656/60000 [====================>.........] - ETA: 31s - loss: 0.2222 - categorical_accuracy: 0.9328
42688/60000 [====================>.........] - ETA: 31s - loss: 0.2222 - categorical_accuracy: 0.9328
42720/60000 [====================>.........] - ETA: 31s - loss: 0.2220 - categorical_accuracy: 0.9328
42752/60000 [====================>.........] - ETA: 31s - loss: 0.2219 - categorical_accuracy: 0.9329
42784/60000 [====================>.........] - ETA: 31s - loss: 0.2218 - categorical_accuracy: 0.9329
42816/60000 [====================>.........] - ETA: 31s - loss: 0.2217 - categorical_accuracy: 0.9329
42848/60000 [====================>.........] - ETA: 31s - loss: 0.2215 - categorical_accuracy: 0.9330
42880/60000 [====================>.........] - ETA: 31s - loss: 0.2214 - categorical_accuracy: 0.9330
42912/60000 [====================>.........] - ETA: 31s - loss: 0.2213 - categorical_accuracy: 0.9330
42944/60000 [====================>.........] - ETA: 31s - loss: 0.2212 - categorical_accuracy: 0.9331
42976/60000 [====================>.........] - ETA: 31s - loss: 0.2211 - categorical_accuracy: 0.9331
43008/60000 [====================>.........] - ETA: 31s - loss: 0.2210 - categorical_accuracy: 0.9331
43040/60000 [====================>.........] - ETA: 31s - loss: 0.2209 - categorical_accuracy: 0.9332
43072/60000 [====================>.........] - ETA: 30s - loss: 0.2207 - categorical_accuracy: 0.9332
43104/60000 [====================>.........] - ETA: 30s - loss: 0.2206 - categorical_accuracy: 0.9332
43136/60000 [====================>.........] - ETA: 30s - loss: 0.2205 - categorical_accuracy: 0.9333
43168/60000 [====================>.........] - ETA: 30s - loss: 0.2205 - categorical_accuracy: 0.9333
43200/60000 [====================>.........] - ETA: 30s - loss: 0.2204 - categorical_accuracy: 0.9333
43232/60000 [====================>.........] - ETA: 30s - loss: 0.2203 - categorical_accuracy: 0.9333
43264/60000 [====================>.........] - ETA: 30s - loss: 0.2203 - categorical_accuracy: 0.9333
43296/60000 [====================>.........] - ETA: 30s - loss: 0.2203 - categorical_accuracy: 0.9333
43328/60000 [====================>.........] - ETA: 30s - loss: 0.2201 - categorical_accuracy: 0.9333
43360/60000 [====================>.........] - ETA: 30s - loss: 0.2200 - categorical_accuracy: 0.9334
43392/60000 [====================>.........] - ETA: 30s - loss: 0.2200 - categorical_accuracy: 0.9334
43424/60000 [====================>.........] - ETA: 30s - loss: 0.2200 - categorical_accuracy: 0.9334
43456/60000 [====================>.........] - ETA: 30s - loss: 0.2198 - categorical_accuracy: 0.9334
43488/60000 [====================>.........] - ETA: 30s - loss: 0.2197 - categorical_accuracy: 0.9335
43520/60000 [====================>.........] - ETA: 30s - loss: 0.2196 - categorical_accuracy: 0.9335
43552/60000 [====================>.........] - ETA: 30s - loss: 0.2194 - categorical_accuracy: 0.9336
43584/60000 [====================>.........] - ETA: 30s - loss: 0.2193 - categorical_accuracy: 0.9336
43616/60000 [====================>.........] - ETA: 29s - loss: 0.2194 - categorical_accuracy: 0.9336
43648/60000 [====================>.........] - ETA: 29s - loss: 0.2194 - categorical_accuracy: 0.9336
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2192 - categorical_accuracy: 0.9336
43712/60000 [====================>.........] - ETA: 29s - loss: 0.2191 - categorical_accuracy: 0.9337
43744/60000 [====================>.........] - ETA: 29s - loss: 0.2190 - categorical_accuracy: 0.9337
43776/60000 [====================>.........] - ETA: 29s - loss: 0.2189 - categorical_accuracy: 0.9338
43808/60000 [====================>.........] - ETA: 29s - loss: 0.2187 - categorical_accuracy: 0.9338
43840/60000 [====================>.........] - ETA: 29s - loss: 0.2186 - categorical_accuracy: 0.9339
43872/60000 [====================>.........] - ETA: 29s - loss: 0.2186 - categorical_accuracy: 0.9339
43904/60000 [====================>.........] - ETA: 29s - loss: 0.2184 - categorical_accuracy: 0.9339
43936/60000 [====================>.........] - ETA: 29s - loss: 0.2184 - categorical_accuracy: 0.9339
43968/60000 [====================>.........] - ETA: 29s - loss: 0.2185 - categorical_accuracy: 0.9339
44000/60000 [=====================>........] - ETA: 29s - loss: 0.2185 - categorical_accuracy: 0.9338
44032/60000 [=====================>........] - ETA: 29s - loss: 0.2183 - categorical_accuracy: 0.9339
44064/60000 [=====================>........] - ETA: 29s - loss: 0.2182 - categorical_accuracy: 0.9339
44096/60000 [=====================>........] - ETA: 29s - loss: 0.2181 - categorical_accuracy: 0.9339
44128/60000 [=====================>........] - ETA: 29s - loss: 0.2180 - categorical_accuracy: 0.9339
44160/60000 [=====================>........] - ETA: 28s - loss: 0.2179 - categorical_accuracy: 0.9340
44192/60000 [=====================>........] - ETA: 28s - loss: 0.2177 - categorical_accuracy: 0.9340
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2178 - categorical_accuracy: 0.9340
44256/60000 [=====================>........] - ETA: 28s - loss: 0.2177 - categorical_accuracy: 0.9341
44288/60000 [=====================>........] - ETA: 28s - loss: 0.2175 - categorical_accuracy: 0.9341
44320/60000 [=====================>........] - ETA: 28s - loss: 0.2174 - categorical_accuracy: 0.9342
44352/60000 [=====================>........] - ETA: 28s - loss: 0.2173 - categorical_accuracy: 0.9342
44384/60000 [=====================>........] - ETA: 28s - loss: 0.2171 - categorical_accuracy: 0.9342
44416/60000 [=====================>........] - ETA: 28s - loss: 0.2170 - categorical_accuracy: 0.9343
44448/60000 [=====================>........] - ETA: 28s - loss: 0.2169 - categorical_accuracy: 0.9343
44480/60000 [=====================>........] - ETA: 28s - loss: 0.2168 - categorical_accuracy: 0.9344
44512/60000 [=====================>........] - ETA: 28s - loss: 0.2169 - categorical_accuracy: 0.9344
44544/60000 [=====================>........] - ETA: 28s - loss: 0.2167 - categorical_accuracy: 0.9344
44576/60000 [=====================>........] - ETA: 28s - loss: 0.2167 - categorical_accuracy: 0.9344
44608/60000 [=====================>........] - ETA: 28s - loss: 0.2167 - categorical_accuracy: 0.9344
44640/60000 [=====================>........] - ETA: 28s - loss: 0.2167 - categorical_accuracy: 0.9345
44672/60000 [=====================>........] - ETA: 28s - loss: 0.2166 - categorical_accuracy: 0.9345
44704/60000 [=====================>........] - ETA: 27s - loss: 0.2165 - categorical_accuracy: 0.9345
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2165 - categorical_accuracy: 0.9345
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2163 - categorical_accuracy: 0.9345
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2163 - categorical_accuracy: 0.9345
44832/60000 [=====================>........] - ETA: 27s - loss: 0.2163 - categorical_accuracy: 0.9345
44864/60000 [=====================>........] - ETA: 27s - loss: 0.2163 - categorical_accuracy: 0.9345
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2162 - categorical_accuracy: 0.9345
44928/60000 [=====================>........] - ETA: 27s - loss: 0.2161 - categorical_accuracy: 0.9346
44960/60000 [=====================>........] - ETA: 27s - loss: 0.2160 - categorical_accuracy: 0.9346
44992/60000 [=====================>........] - ETA: 27s - loss: 0.2159 - categorical_accuracy: 0.9346
45024/60000 [=====================>........] - ETA: 27s - loss: 0.2157 - categorical_accuracy: 0.9347
45056/60000 [=====================>........] - ETA: 27s - loss: 0.2157 - categorical_accuracy: 0.9347
45088/60000 [=====================>........] - ETA: 27s - loss: 0.2155 - categorical_accuracy: 0.9347
45120/60000 [=====================>........] - ETA: 27s - loss: 0.2154 - categorical_accuracy: 0.9348
45152/60000 [=====================>........] - ETA: 27s - loss: 0.2153 - categorical_accuracy: 0.9348
45184/60000 [=====================>........] - ETA: 27s - loss: 0.2152 - categorical_accuracy: 0.9348
45216/60000 [=====================>........] - ETA: 27s - loss: 0.2150 - categorical_accuracy: 0.9349
45248/60000 [=====================>........] - ETA: 27s - loss: 0.2150 - categorical_accuracy: 0.9349
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2150 - categorical_accuracy: 0.9349
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2149 - categorical_accuracy: 0.9349
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2148 - categorical_accuracy: 0.9349
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2147 - categorical_accuracy: 0.9350
45408/60000 [=====================>........] - ETA: 26s - loss: 0.2145 - categorical_accuracy: 0.9350
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2144 - categorical_accuracy: 0.9351
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2144 - categorical_accuracy: 0.9351
45504/60000 [=====================>........] - ETA: 26s - loss: 0.2143 - categorical_accuracy: 0.9351
45536/60000 [=====================>........] - ETA: 26s - loss: 0.2142 - categorical_accuracy: 0.9351
45568/60000 [=====================>........] - ETA: 26s - loss: 0.2142 - categorical_accuracy: 0.9351
45600/60000 [=====================>........] - ETA: 26s - loss: 0.2141 - categorical_accuracy: 0.9352
45632/60000 [=====================>........] - ETA: 26s - loss: 0.2141 - categorical_accuracy: 0.9352
45664/60000 [=====================>........] - ETA: 26s - loss: 0.2141 - categorical_accuracy: 0.9352
45696/60000 [=====================>........] - ETA: 26s - loss: 0.2139 - categorical_accuracy: 0.9352
45728/60000 [=====================>........] - ETA: 26s - loss: 0.2138 - categorical_accuracy: 0.9352
45760/60000 [=====================>........] - ETA: 26s - loss: 0.2137 - categorical_accuracy: 0.9353
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2137 - categorical_accuracy: 0.9353
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2136 - categorical_accuracy: 0.9353
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2135 - categorical_accuracy: 0.9354
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2133 - categorical_accuracy: 0.9354
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2133 - categorical_accuracy: 0.9354
45952/60000 [=====================>........] - ETA: 25s - loss: 0.2133 - categorical_accuracy: 0.9354
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2132 - categorical_accuracy: 0.9355
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2130 - categorical_accuracy: 0.9355
46048/60000 [======================>.......] - ETA: 25s - loss: 0.2130 - categorical_accuracy: 0.9355
46080/60000 [======================>.......] - ETA: 25s - loss: 0.2128 - categorical_accuracy: 0.9355
46112/60000 [======================>.......] - ETA: 25s - loss: 0.2128 - categorical_accuracy: 0.9356
46144/60000 [======================>.......] - ETA: 25s - loss: 0.2126 - categorical_accuracy: 0.9356
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2125 - categorical_accuracy: 0.9357
46208/60000 [======================>.......] - ETA: 25s - loss: 0.2124 - categorical_accuracy: 0.9357
46240/60000 [======================>.......] - ETA: 25s - loss: 0.2123 - categorical_accuracy: 0.9357
46272/60000 [======================>.......] - ETA: 25s - loss: 0.2123 - categorical_accuracy: 0.9357
46304/60000 [======================>.......] - ETA: 25s - loss: 0.2121 - categorical_accuracy: 0.9358
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2120 - categorical_accuracy: 0.9358
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2119 - categorical_accuracy: 0.9358
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2119 - categorical_accuracy: 0.9358
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2118 - categorical_accuracy: 0.9358
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2118 - categorical_accuracy: 0.9358
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2117 - categorical_accuracy: 0.9359
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2116 - categorical_accuracy: 0.9359
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2116 - categorical_accuracy: 0.9359
46592/60000 [======================>.......] - ETA: 24s - loss: 0.2115 - categorical_accuracy: 0.9360
46624/60000 [======================>.......] - ETA: 24s - loss: 0.2114 - categorical_accuracy: 0.9360
46656/60000 [======================>.......] - ETA: 24s - loss: 0.2113 - categorical_accuracy: 0.9360
46688/60000 [======================>.......] - ETA: 24s - loss: 0.2113 - categorical_accuracy: 0.9360
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2112 - categorical_accuracy: 0.9360
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2111 - categorical_accuracy: 0.9360
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2111 - categorical_accuracy: 0.9360
46816/60000 [======================>.......] - ETA: 24s - loss: 0.2110 - categorical_accuracy: 0.9360
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2111 - categorical_accuracy: 0.9360
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2109 - categorical_accuracy: 0.9360
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2108 - categorical_accuracy: 0.9361
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2107 - categorical_accuracy: 0.9361
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2109 - categorical_accuracy: 0.9361
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2108 - categorical_accuracy: 0.9361
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2107 - categorical_accuracy: 0.9362
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2107 - categorical_accuracy: 0.9362
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2107 - categorical_accuracy: 0.9362
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2106 - categorical_accuracy: 0.9362
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2106 - categorical_accuracy: 0.9362
47200/60000 [======================>.......] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9362
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2104 - categorical_accuracy: 0.9363
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2104 - categorical_accuracy: 0.9363
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2103 - categorical_accuracy: 0.9363
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9363
47360/60000 [======================>.......] - ETA: 23s - loss: 0.2105 - categorical_accuracy: 0.9363
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2104 - categorical_accuracy: 0.9363
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2103 - categorical_accuracy: 0.9363
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2103 - categorical_accuracy: 0.9363
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2101 - categorical_accuracy: 0.9364
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2101 - categorical_accuracy: 0.9364
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2100 - categorical_accuracy: 0.9364
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2099 - categorical_accuracy: 0.9364
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2098 - categorical_accuracy: 0.9365
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2097 - categorical_accuracy: 0.9365
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2096 - categorical_accuracy: 0.9366
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2095 - categorical_accuracy: 0.9366
47744/60000 [======================>.......] - ETA: 22s - loss: 0.2094 - categorical_accuracy: 0.9366
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2092 - categorical_accuracy: 0.9367
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2091 - categorical_accuracy: 0.9367
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2090 - categorical_accuracy: 0.9367
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2089 - categorical_accuracy: 0.9368
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2088 - categorical_accuracy: 0.9368
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2088 - categorical_accuracy: 0.9368
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2086 - categorical_accuracy: 0.9369
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2085 - categorical_accuracy: 0.9369
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2086 - categorical_accuracy: 0.9369
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2087 - categorical_accuracy: 0.9369
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2086 - categorical_accuracy: 0.9370
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2085 - categorical_accuracy: 0.9370
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2084 - categorical_accuracy: 0.9370
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2083 - categorical_accuracy: 0.9371
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2082 - categorical_accuracy: 0.9371
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2081 - categorical_accuracy: 0.9371
48288/60000 [=======================>......] - ETA: 21s - loss: 0.2080 - categorical_accuracy: 0.9371
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9372
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2079 - categorical_accuracy: 0.9372
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2078 - categorical_accuracy: 0.9372
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2077 - categorical_accuracy: 0.9372
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2077 - categorical_accuracy: 0.9372
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2076 - categorical_accuracy: 0.9373
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2075 - categorical_accuracy: 0.9373
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2073 - categorical_accuracy: 0.9373
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2073 - categorical_accuracy: 0.9373
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2075 - categorical_accuracy: 0.9374
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2075 - categorical_accuracy: 0.9374
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2076 - categorical_accuracy: 0.9373
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2075 - categorical_accuracy: 0.9374
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2074 - categorical_accuracy: 0.9374
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2074 - categorical_accuracy: 0.9374
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2073 - categorical_accuracy: 0.9374
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2073 - categorical_accuracy: 0.9374
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2072 - categorical_accuracy: 0.9374
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2071 - categorical_accuracy: 0.9375
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2070 - categorical_accuracy: 0.9375
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2069 - categorical_accuracy: 0.9375
48992/60000 [=======================>......] - ETA: 20s - loss: 0.2068 - categorical_accuracy: 0.9376
49024/60000 [=======================>......] - ETA: 20s - loss: 0.2067 - categorical_accuracy: 0.9376
49056/60000 [=======================>......] - ETA: 20s - loss: 0.2066 - categorical_accuracy: 0.9376
49088/60000 [=======================>......] - ETA: 20s - loss: 0.2067 - categorical_accuracy: 0.9376
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2067 - categorical_accuracy: 0.9376
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2066 - categorical_accuracy: 0.9377
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2065 - categorical_accuracy: 0.9377
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2063 - categorical_accuracy: 0.9378
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2063 - categorical_accuracy: 0.9378
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2063 - categorical_accuracy: 0.9378
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2062 - categorical_accuracy: 0.9378
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2061 - categorical_accuracy: 0.9378
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2060 - categorical_accuracy: 0.9379
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2060 - categorical_accuracy: 0.9379
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2058 - categorical_accuracy: 0.9379
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2058 - categorical_accuracy: 0.9379
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2057 - categorical_accuracy: 0.9380
49536/60000 [=======================>......] - ETA: 19s - loss: 0.2057 - categorical_accuracy: 0.9380
49568/60000 [=======================>......] - ETA: 19s - loss: 0.2055 - categorical_accuracy: 0.9380
49600/60000 [=======================>......] - ETA: 19s - loss: 0.2055 - categorical_accuracy: 0.9380
49632/60000 [=======================>......] - ETA: 19s - loss: 0.2054 - categorical_accuracy: 0.9381
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2053 - categorical_accuracy: 0.9381
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2052 - categorical_accuracy: 0.9381
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2051 - categorical_accuracy: 0.9381
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2051 - categorical_accuracy: 0.9382
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2049 - categorical_accuracy: 0.9382
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2048 - categorical_accuracy: 0.9382
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2047 - categorical_accuracy: 0.9383
49888/60000 [=======================>......] - ETA: 18s - loss: 0.2046 - categorical_accuracy: 0.9383
49920/60000 [=======================>......] - ETA: 18s - loss: 0.2045 - categorical_accuracy: 0.9383
49952/60000 [=======================>......] - ETA: 18s - loss: 0.2045 - categorical_accuracy: 0.9383
49984/60000 [=======================>......] - ETA: 18s - loss: 0.2045 - categorical_accuracy: 0.9383
50016/60000 [========================>.....] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9383
50048/60000 [========================>.....] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9383
50080/60000 [========================>.....] - ETA: 18s - loss: 0.2044 - categorical_accuracy: 0.9384
50112/60000 [========================>.....] - ETA: 18s - loss: 0.2042 - categorical_accuracy: 0.9384
50144/60000 [========================>.....] - ETA: 18s - loss: 0.2041 - categorical_accuracy: 0.9384
50176/60000 [========================>.....] - ETA: 18s - loss: 0.2040 - categorical_accuracy: 0.9385
50208/60000 [========================>.....] - ETA: 17s - loss: 0.2039 - categorical_accuracy: 0.9385
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2038 - categorical_accuracy: 0.9385
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2039 - categorical_accuracy: 0.9385
50304/60000 [========================>.....] - ETA: 17s - loss: 0.2038 - categorical_accuracy: 0.9385
50336/60000 [========================>.....] - ETA: 17s - loss: 0.2036 - categorical_accuracy: 0.9386
50368/60000 [========================>.....] - ETA: 17s - loss: 0.2036 - categorical_accuracy: 0.9386
50400/60000 [========================>.....] - ETA: 17s - loss: 0.2034 - categorical_accuracy: 0.9386
50432/60000 [========================>.....] - ETA: 17s - loss: 0.2033 - categorical_accuracy: 0.9387
50464/60000 [========================>.....] - ETA: 17s - loss: 0.2034 - categorical_accuracy: 0.9387
50496/60000 [========================>.....] - ETA: 17s - loss: 0.2034 - categorical_accuracy: 0.9387
50528/60000 [========================>.....] - ETA: 17s - loss: 0.2034 - categorical_accuracy: 0.9387
50560/60000 [========================>.....] - ETA: 17s - loss: 0.2033 - categorical_accuracy: 0.9387
50592/60000 [========================>.....] - ETA: 17s - loss: 0.2032 - categorical_accuracy: 0.9387
50624/60000 [========================>.....] - ETA: 17s - loss: 0.2032 - categorical_accuracy: 0.9387
50656/60000 [========================>.....] - ETA: 17s - loss: 0.2031 - categorical_accuracy: 0.9388
50688/60000 [========================>.....] - ETA: 17s - loss: 0.2030 - categorical_accuracy: 0.9388
50720/60000 [========================>.....] - ETA: 17s - loss: 0.2029 - categorical_accuracy: 0.9388
50752/60000 [========================>.....] - ETA: 16s - loss: 0.2029 - categorical_accuracy: 0.9388
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2028 - categorical_accuracy: 0.9389
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2026 - categorical_accuracy: 0.9389
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2026 - categorical_accuracy: 0.9389
50880/60000 [========================>.....] - ETA: 16s - loss: 0.2025 - categorical_accuracy: 0.9390
50912/60000 [========================>.....] - ETA: 16s - loss: 0.2026 - categorical_accuracy: 0.9390
50944/60000 [========================>.....] - ETA: 16s - loss: 0.2025 - categorical_accuracy: 0.9390
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2024 - categorical_accuracy: 0.9390
51008/60000 [========================>.....] - ETA: 16s - loss: 0.2023 - categorical_accuracy: 0.9390
51040/60000 [========================>.....] - ETA: 16s - loss: 0.2022 - categorical_accuracy: 0.9391
51072/60000 [========================>.....] - ETA: 16s - loss: 0.2022 - categorical_accuracy: 0.9391
51104/60000 [========================>.....] - ETA: 16s - loss: 0.2021 - categorical_accuracy: 0.9391
51136/60000 [========================>.....] - ETA: 16s - loss: 0.2021 - categorical_accuracy: 0.9391
51168/60000 [========================>.....] - ETA: 16s - loss: 0.2021 - categorical_accuracy: 0.9391
51200/60000 [========================>.....] - ETA: 16s - loss: 0.2021 - categorical_accuracy: 0.9391
51232/60000 [========================>.....] - ETA: 16s - loss: 0.2021 - categorical_accuracy: 0.9391
51264/60000 [========================>.....] - ETA: 16s - loss: 0.2021 - categorical_accuracy: 0.9391
51296/60000 [========================>.....] - ETA: 15s - loss: 0.2019 - categorical_accuracy: 0.9392
51328/60000 [========================>.....] - ETA: 15s - loss: 0.2018 - categorical_accuracy: 0.9392
51360/60000 [========================>.....] - ETA: 15s - loss: 0.2017 - categorical_accuracy: 0.9393
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2016 - categorical_accuracy: 0.9393
51424/60000 [========================>.....] - ETA: 15s - loss: 0.2017 - categorical_accuracy: 0.9393
51456/60000 [========================>.....] - ETA: 15s - loss: 0.2016 - categorical_accuracy: 0.9393
51488/60000 [========================>.....] - ETA: 15s - loss: 0.2014 - categorical_accuracy: 0.9394
51520/60000 [========================>.....] - ETA: 15s - loss: 0.2014 - categorical_accuracy: 0.9394
51552/60000 [========================>.....] - ETA: 15s - loss: 0.2013 - categorical_accuracy: 0.9394
51584/60000 [========================>.....] - ETA: 15s - loss: 0.2012 - categorical_accuracy: 0.9394
51616/60000 [========================>.....] - ETA: 15s - loss: 0.2011 - categorical_accuracy: 0.9395
51648/60000 [========================>.....] - ETA: 15s - loss: 0.2011 - categorical_accuracy: 0.9395
51680/60000 [========================>.....] - ETA: 15s - loss: 0.2010 - categorical_accuracy: 0.9395
51712/60000 [========================>.....] - ETA: 15s - loss: 0.2009 - categorical_accuracy: 0.9395
51744/60000 [========================>.....] - ETA: 15s - loss: 0.2008 - categorical_accuracy: 0.9395
51776/60000 [========================>.....] - ETA: 15s - loss: 0.2008 - categorical_accuracy: 0.9395
51808/60000 [========================>.....] - ETA: 15s - loss: 0.2007 - categorical_accuracy: 0.9396
51840/60000 [========================>.....] - ETA: 14s - loss: 0.2007 - categorical_accuracy: 0.9396
51872/60000 [========================>.....] - ETA: 14s - loss: 0.2006 - categorical_accuracy: 0.9396
51904/60000 [========================>.....] - ETA: 14s - loss: 0.2006 - categorical_accuracy: 0.9396
51936/60000 [========================>.....] - ETA: 14s - loss: 0.2006 - categorical_accuracy: 0.9396
51968/60000 [========================>.....] - ETA: 14s - loss: 0.2006 - categorical_accuracy: 0.9396
52000/60000 [=========================>....] - ETA: 14s - loss: 0.2005 - categorical_accuracy: 0.9396
52032/60000 [=========================>....] - ETA: 14s - loss: 0.2004 - categorical_accuracy: 0.9396
52064/60000 [=========================>....] - ETA: 14s - loss: 0.2004 - categorical_accuracy: 0.9397
52096/60000 [=========================>....] - ETA: 14s - loss: 0.2003 - categorical_accuracy: 0.9397
52128/60000 [=========================>....] - ETA: 14s - loss: 0.2002 - categorical_accuracy: 0.9397
52160/60000 [=========================>....] - ETA: 14s - loss: 0.2001 - categorical_accuracy: 0.9397
52192/60000 [=========================>....] - ETA: 14s - loss: 0.2000 - categorical_accuracy: 0.9398
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9398
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1998 - categorical_accuracy: 0.9399
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1997 - categorical_accuracy: 0.9399
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1997 - categorical_accuracy: 0.9399
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1997 - categorical_accuracy: 0.9399
52384/60000 [=========================>....] - ETA: 13s - loss: 0.1996 - categorical_accuracy: 0.9399
52416/60000 [=========================>....] - ETA: 13s - loss: 0.1995 - categorical_accuracy: 0.9399
52448/60000 [=========================>....] - ETA: 13s - loss: 0.1996 - categorical_accuracy: 0.9399
52480/60000 [=========================>....] - ETA: 13s - loss: 0.1995 - categorical_accuracy: 0.9400
52512/60000 [=========================>....] - ETA: 13s - loss: 0.1994 - categorical_accuracy: 0.9400
52544/60000 [=========================>....] - ETA: 13s - loss: 0.1993 - categorical_accuracy: 0.9400
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1992 - categorical_accuracy: 0.9400
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1992 - categorical_accuracy: 0.9400
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1992 - categorical_accuracy: 0.9400
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1991 - categorical_accuracy: 0.9401
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1990 - categorical_accuracy: 0.9401
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1990 - categorical_accuracy: 0.9401
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1991 - categorical_accuracy: 0.9401
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1990 - categorical_accuracy: 0.9401
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1989 - categorical_accuracy: 0.9401
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9402
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9402
52928/60000 [=========================>....] - ETA: 12s - loss: 0.1988 - categorical_accuracy: 0.9402
52960/60000 [=========================>....] - ETA: 12s - loss: 0.1988 - categorical_accuracy: 0.9402
52992/60000 [=========================>....] - ETA: 12s - loss: 0.1987 - categorical_accuracy: 0.9402
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1987 - categorical_accuracy: 0.9402
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1987 - categorical_accuracy: 0.9402
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1986 - categorical_accuracy: 0.9403
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1987 - categorical_accuracy: 0.9403
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1987 - categorical_accuracy: 0.9403
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1986 - categorical_accuracy: 0.9403
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1986 - categorical_accuracy: 0.9403
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1985 - categorical_accuracy: 0.9403
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1984 - categorical_accuracy: 0.9404
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1983 - categorical_accuracy: 0.9404
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1983 - categorical_accuracy: 0.9404
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1982 - categorical_accuracy: 0.9404
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1981 - categorical_accuracy: 0.9404
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1981 - categorical_accuracy: 0.9405
53472/60000 [=========================>....] - ETA: 11s - loss: 0.1980 - categorical_accuracy: 0.9405
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1980 - categorical_accuracy: 0.9405
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1979 - categorical_accuracy: 0.9405
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1978 - categorical_accuracy: 0.9405
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1978 - categorical_accuracy: 0.9405
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1977 - categorical_accuracy: 0.9406
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1977 - categorical_accuracy: 0.9406
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1976 - categorical_accuracy: 0.9406
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1976 - categorical_accuracy: 0.9406
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1975 - categorical_accuracy: 0.9406
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1975 - categorical_accuracy: 0.9407
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1974 - categorical_accuracy: 0.9407
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1973 - categorical_accuracy: 0.9407
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1974 - categorical_accuracy: 0.9407
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1973 - categorical_accuracy: 0.9407
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1974 - categorical_accuracy: 0.9407
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1973 - categorical_accuracy: 0.9407
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1973 - categorical_accuracy: 0.9408
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1972 - categorical_accuracy: 0.9408
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1971 - categorical_accuracy: 0.9408
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1970 - categorical_accuracy: 0.9409
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1971 - categorical_accuracy: 0.9409
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1970 - categorical_accuracy: 0.9409
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1969 - categorical_accuracy: 0.9409
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1969 - categorical_accuracy: 0.9409
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1968 - categorical_accuracy: 0.9410
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1967 - categorical_accuracy: 0.9410
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1967 - categorical_accuracy: 0.9410
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1967 - categorical_accuracy: 0.9410
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1966 - categorical_accuracy: 0.9410
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1965 - categorical_accuracy: 0.9411
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1964 - categorical_accuracy: 0.9411
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1963 - categorical_accuracy: 0.9411
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1962 - categorical_accuracy: 0.9412
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1961 - categorical_accuracy: 0.9412 
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1960 - categorical_accuracy: 0.9412
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1959 - categorical_accuracy: 0.9412
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1960 - categorical_accuracy: 0.9412
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1960 - categorical_accuracy: 0.9412
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1959 - categorical_accuracy: 0.9413
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1958 - categorical_accuracy: 0.9413
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1958 - categorical_accuracy: 0.9413
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1957 - categorical_accuracy: 0.9413
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1956 - categorical_accuracy: 0.9414
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1955 - categorical_accuracy: 0.9414
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9414
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9415
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9414
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1953 - categorical_accuracy: 0.9414
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1952 - categorical_accuracy: 0.9415
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1952 - categorical_accuracy: 0.9415
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1951 - categorical_accuracy: 0.9415
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1950 - categorical_accuracy: 0.9416
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1949 - categorical_accuracy: 0.9416
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1948 - categorical_accuracy: 0.9416
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1947 - categorical_accuracy: 0.9416
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1946 - categorical_accuracy: 0.9417
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1945 - categorical_accuracy: 0.9417
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1946 - categorical_accuracy: 0.9417
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1945 - categorical_accuracy: 0.9417
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1944 - categorical_accuracy: 0.9417
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1943 - categorical_accuracy: 0.9418
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1942 - categorical_accuracy: 0.9418
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1941 - categorical_accuracy: 0.9418
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1941 - categorical_accuracy: 0.9419
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9419
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9419
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1939 - categorical_accuracy: 0.9419
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1938 - categorical_accuracy: 0.9419
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1939 - categorical_accuracy: 0.9419
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1939 - categorical_accuracy: 0.9420
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1938 - categorical_accuracy: 0.9420
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1938 - categorical_accuracy: 0.9420
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1937 - categorical_accuracy: 0.9420
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1936 - categorical_accuracy: 0.9420
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1935 - categorical_accuracy: 0.9420
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1934 - categorical_accuracy: 0.9420
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1933 - categorical_accuracy: 0.9421
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1933 - categorical_accuracy: 0.9421
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1932 - categorical_accuracy: 0.9421
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1931 - categorical_accuracy: 0.9421
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1930 - categorical_accuracy: 0.9422
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1929 - categorical_accuracy: 0.9422
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1928 - categorical_accuracy: 0.9422
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1928 - categorical_accuracy: 0.9422
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1928 - categorical_accuracy: 0.9422
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1927 - categorical_accuracy: 0.9422
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1926 - categorical_accuracy: 0.9422
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1926 - categorical_accuracy: 0.9423
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1925 - categorical_accuracy: 0.9423
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1924 - categorical_accuracy: 0.9423
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1924 - categorical_accuracy: 0.9423
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1924 - categorical_accuracy: 0.9423
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1923 - categorical_accuracy: 0.9423
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1922 - categorical_accuracy: 0.9424
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1921 - categorical_accuracy: 0.9424
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1921 - categorical_accuracy: 0.9424
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1920 - categorical_accuracy: 0.9424
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1919 - categorical_accuracy: 0.9425
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1918 - categorical_accuracy: 0.9425
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1918 - categorical_accuracy: 0.9425
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1917 - categorical_accuracy: 0.9425
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1916 - categorical_accuracy: 0.9426
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1915 - categorical_accuracy: 0.9426
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1914 - categorical_accuracy: 0.9426
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1914 - categorical_accuracy: 0.9426
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1914 - categorical_accuracy: 0.9426
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1914 - categorical_accuracy: 0.9426
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1914 - categorical_accuracy: 0.9426
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1913 - categorical_accuracy: 0.9427
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1912 - categorical_accuracy: 0.9427
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1911 - categorical_accuracy: 0.9427
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1912 - categorical_accuracy: 0.9427
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1911 - categorical_accuracy: 0.9427
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1910 - categorical_accuracy: 0.9427
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1910 - categorical_accuracy: 0.9427
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9427
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1908 - categorical_accuracy: 0.9428
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1908 - categorical_accuracy: 0.9428
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9428
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9429
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9429
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9429
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1907 - categorical_accuracy: 0.9429
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9429
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9429
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1905 - categorical_accuracy: 0.9429
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1904 - categorical_accuracy: 0.9430
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1903 - categorical_accuracy: 0.9430
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1904 - categorical_accuracy: 0.9430
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1903 - categorical_accuracy: 0.9430
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1903 - categorical_accuracy: 0.9430
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1903 - categorical_accuracy: 0.9430
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1902 - categorical_accuracy: 0.9430
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1901 - categorical_accuracy: 0.9430
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1901 - categorical_accuracy: 0.9431
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1900 - categorical_accuracy: 0.9431
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1900 - categorical_accuracy: 0.9431
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1899 - categorical_accuracy: 0.9431
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9431
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1897 - categorical_accuracy: 0.9431
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1897 - categorical_accuracy: 0.9432
58016/60000 [============================>.] - ETA: 3s - loss: 0.1896 - categorical_accuracy: 0.9432
58048/60000 [============================>.] - ETA: 3s - loss: 0.1895 - categorical_accuracy: 0.9432
58080/60000 [============================>.] - ETA: 3s - loss: 0.1894 - categorical_accuracy: 0.9432
58112/60000 [============================>.] - ETA: 3s - loss: 0.1894 - categorical_accuracy: 0.9432
58144/60000 [============================>.] - ETA: 3s - loss: 0.1893 - categorical_accuracy: 0.9432
58176/60000 [============================>.] - ETA: 3s - loss: 0.1893 - categorical_accuracy: 0.9432
58208/60000 [============================>.] - ETA: 3s - loss: 0.1893 - categorical_accuracy: 0.9432
58240/60000 [============================>.] - ETA: 3s - loss: 0.1892 - categorical_accuracy: 0.9433
58272/60000 [============================>.] - ETA: 3s - loss: 0.1892 - categorical_accuracy: 0.9432
58304/60000 [============================>.] - ETA: 3s - loss: 0.1891 - categorical_accuracy: 0.9433
58336/60000 [============================>.] - ETA: 3s - loss: 0.1891 - categorical_accuracy: 0.9433
58368/60000 [============================>.] - ETA: 3s - loss: 0.1890 - categorical_accuracy: 0.9433
58400/60000 [============================>.] - ETA: 2s - loss: 0.1890 - categorical_accuracy: 0.9433
58432/60000 [============================>.] - ETA: 2s - loss: 0.1889 - categorical_accuracy: 0.9434
58464/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9434
58496/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9434
58528/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9433
58560/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9434
58592/60000 [============================>.] - ETA: 2s - loss: 0.1887 - categorical_accuracy: 0.9434
58624/60000 [============================>.] - ETA: 2s - loss: 0.1887 - categorical_accuracy: 0.9434
58656/60000 [============================>.] - ETA: 2s - loss: 0.1886 - categorical_accuracy: 0.9434
58688/60000 [============================>.] - ETA: 2s - loss: 0.1886 - categorical_accuracy: 0.9434
58720/60000 [============================>.] - ETA: 2s - loss: 0.1885 - categorical_accuracy: 0.9435
58752/60000 [============================>.] - ETA: 2s - loss: 0.1886 - categorical_accuracy: 0.9435
58784/60000 [============================>.] - ETA: 2s - loss: 0.1885 - categorical_accuracy: 0.9435
58816/60000 [============================>.] - ETA: 2s - loss: 0.1884 - categorical_accuracy: 0.9435
58848/60000 [============================>.] - ETA: 2s - loss: 0.1883 - categorical_accuracy: 0.9435
58880/60000 [============================>.] - ETA: 2s - loss: 0.1884 - categorical_accuracy: 0.9435
58912/60000 [============================>.] - ETA: 2s - loss: 0.1883 - categorical_accuracy: 0.9436
58944/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9436
58976/60000 [============================>.] - ETA: 1s - loss: 0.1884 - categorical_accuracy: 0.9436
59008/60000 [============================>.] - ETA: 1s - loss: 0.1884 - categorical_accuracy: 0.9436
59040/60000 [============================>.] - ETA: 1s - loss: 0.1883 - categorical_accuracy: 0.9436
59072/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9436
59104/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9437
59136/60000 [============================>.] - ETA: 1s - loss: 0.1881 - categorical_accuracy: 0.9437
59168/60000 [============================>.] - ETA: 1s - loss: 0.1880 - categorical_accuracy: 0.9437
59200/60000 [============================>.] - ETA: 1s - loss: 0.1879 - categorical_accuracy: 0.9437
59232/60000 [============================>.] - ETA: 1s - loss: 0.1879 - categorical_accuracy: 0.9437
59264/60000 [============================>.] - ETA: 1s - loss: 0.1879 - categorical_accuracy: 0.9438
59296/60000 [============================>.] - ETA: 1s - loss: 0.1878 - categorical_accuracy: 0.9438
59328/60000 [============================>.] - ETA: 1s - loss: 0.1877 - categorical_accuracy: 0.9438
59360/60000 [============================>.] - ETA: 1s - loss: 0.1876 - categorical_accuracy: 0.9438
59392/60000 [============================>.] - ETA: 1s - loss: 0.1875 - categorical_accuracy: 0.9439
59424/60000 [============================>.] - ETA: 1s - loss: 0.1875 - categorical_accuracy: 0.9439
59456/60000 [============================>.] - ETA: 1s - loss: 0.1874 - categorical_accuracy: 0.9439
59488/60000 [============================>.] - ETA: 0s - loss: 0.1873 - categorical_accuracy: 0.9439
59520/60000 [============================>.] - ETA: 0s - loss: 0.1874 - categorical_accuracy: 0.9439
59552/60000 [============================>.] - ETA: 0s - loss: 0.1873 - categorical_accuracy: 0.9439
59584/60000 [============================>.] - ETA: 0s - loss: 0.1872 - categorical_accuracy: 0.9439
59616/60000 [============================>.] - ETA: 0s - loss: 0.1872 - categorical_accuracy: 0.9440
59648/60000 [============================>.] - ETA: 0s - loss: 0.1871 - categorical_accuracy: 0.9440
59680/60000 [============================>.] - ETA: 0s - loss: 0.1870 - categorical_accuracy: 0.9440
59712/60000 [============================>.] - ETA: 0s - loss: 0.1869 - categorical_accuracy: 0.9440
59744/60000 [============================>.] - ETA: 0s - loss: 0.1868 - categorical_accuracy: 0.9441
59776/60000 [============================>.] - ETA: 0s - loss: 0.1868 - categorical_accuracy: 0.9441
59808/60000 [============================>.] - ETA: 0s - loss: 0.1868 - categorical_accuracy: 0.9441
59840/60000 [============================>.] - ETA: 0s - loss: 0.1868 - categorical_accuracy: 0.9441
59872/60000 [============================>.] - ETA: 0s - loss: 0.1867 - categorical_accuracy: 0.9441
59904/60000 [============================>.] - ETA: 0s - loss: 0.1866 - categorical_accuracy: 0.9441
59936/60000 [============================>.] - ETA: 0s - loss: 0.1866 - categorical_accuracy: 0.9442
59968/60000 [============================>.] - ETA: 0s - loss: 0.1865 - categorical_accuracy: 0.9442
60000/60000 [==============================] - 114s 2ms/step - loss: 0.1864 - categorical_accuracy: 0.9442 - val_loss: 0.0512 - val_categorical_accuracy: 0.9834

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  640/10000 [>.............................] - ETA: 4s
  768/10000 [=>............................] - ETA: 4s
  896/10000 [=>............................] - ETA: 3s
 1056/10000 [==>...........................] - ETA: 3s
 1216/10000 [==>...........................] - ETA: 3s
 1376/10000 [===>..........................] - ETA: 3s
 1536/10000 [===>..........................] - ETA: 3s
 1696/10000 [====>.........................] - ETA: 3s
 1856/10000 [====>.........................] - ETA: 3s
 2016/10000 [=====>........................] - ETA: 3s
 2176/10000 [=====>........................] - ETA: 3s
 2336/10000 [======>.......................] - ETA: 3s
 2496/10000 [======>.......................] - ETA: 2s
 2656/10000 [======>.......................] - ETA: 2s
 2816/10000 [=======>......................] - ETA: 2s
 2976/10000 [=======>......................] - ETA: 2s
 3136/10000 [========>.....................] - ETA: 2s
 3296/10000 [========>.....................] - ETA: 2s
 3456/10000 [=========>....................] - ETA: 2s
 3616/10000 [=========>....................] - ETA: 2s
 3776/10000 [==========>...................] - ETA: 2s
 3936/10000 [==========>...................] - ETA: 2s
 4096/10000 [===========>..................] - ETA: 2s
 4256/10000 [===========>..................] - ETA: 2s
 4416/10000 [============>.................] - ETA: 2s
 4576/10000 [============>.................] - ETA: 2s
 4736/10000 [=============>................] - ETA: 1s
 4864/10000 [=============>................] - ETA: 1s
 5024/10000 [==============>...............] - ETA: 1s
 5184/10000 [==============>...............] - ETA: 1s
 5344/10000 [===============>..............] - ETA: 1s
 5504/10000 [===============>..............] - ETA: 1s
 5664/10000 [===============>..............] - ETA: 1s
 5824/10000 [================>.............] - ETA: 1s
 5984/10000 [================>.............] - ETA: 1s
 6144/10000 [=================>............] - ETA: 1s
 6304/10000 [=================>............] - ETA: 1s
 6432/10000 [==================>...........] - ETA: 1s
 6560/10000 [==================>...........] - ETA: 1s
 6720/10000 [===================>..........] - ETA: 1s
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
 8896/10000 [=========================>....] - ETA: 0s
 9024/10000 [==========================>...] - ETA: 0s
 9184/10000 [==========================>...] - ETA: 0s
 9344/10000 [===========================>..] - ETA: 0s
 9504/10000 [===========================>..] - ETA: 0s
 9664/10000 [===========================>..] - ETA: 0s
 9824/10000 [============================>.] - ETA: 0s
 9984/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 375us/step
[[1.8753429e-08 7.1602010e-08 1.7238608e-07 ... 9.9999809e-01
  4.7865822e-08 5.6864707e-07]
 [3.6139279e-06 6.5294307e-07 9.9999487e-01 ... 1.8674502e-10
  2.1489866e-07 2.0610033e-11]
 [2.5695051e-06 9.9976760e-01 3.3878365e-05 ... 2.2900222e-05
  3.5496505e-05 7.6642181e-07]
 ...
 [2.5992053e-08 4.7040334e-07 1.2274147e-08 ... 1.5858872e-05
  1.8478294e-06 1.2305267e-04]
 [6.5216982e-06 1.2292561e-07 3.1866374e-08 ... 2.7245903e-07
  1.5992546e-03 7.8410642e-07]
 [2.0355983e-06 1.0600466e-07 1.2594178e-05 ... 2.1051869e-09
  8.1151705e-07 2.7032785e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05124406540420605, 'accuracy_test:': 0.9833999872207642}

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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
Already up to date.
[master 0d18a24] ml_store
 1 file changed, 2041 insertions(+)
To github.com:arita37/mlmodels_store.git
   3849fef..0d18a24  master -> master





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
{'loss': 0.5165653675794601, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-22 20:29:45.786702: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 2c44e1d] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   0d18a24..2c44e1d  master -> master





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
[master 4361f36] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   2c44e1d..4361f36  master -> master





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
 40%|████      | 2/5 [00:21<00:31, 10.55s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9353227328496303, 'learning_rate': 0.051087088246482906, 'min_data_in_leaf': 6, 'num_leaves': 52} and reward: 0.393
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xee)\xf0\x99l\x9aX\r\x00\x00\x00learning_rateq\x02G?\xaa(\x16:\x88t\x02X\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.393
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xee)\xf0\x99l\x9aX\r\x00\x00\x00learning_rateq\x02G?\xaa(\x16:\x88t\x02X\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K4u.' and reward: 0.393
 60%|██████    | 3/5 [00:48<00:31, 15.61s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9736845677222974, 'learning_rate': 0.04689901796733487, 'min_data_in_leaf': 10, 'num_leaves': 46} and reward: 0.3936
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef(l\x89\xdf\x96\xdaX\r\x00\x00\x00learning_rateq\x02G?\xa8\x03%\xe8\xc4\xb8qX\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K.u.' and reward: 0.3936
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef(l\x89\xdf\x96\xdaX\r\x00\x00\x00learning_rateq\x02G?\xa8\x03%\xe8\xc4\xb8qX\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K.u.' and reward: 0.3936
 80%|████████  | 4/5 [01:14<00:18, 18.70s/it] 80%|████████  | 4/5 [01:14<00:18, 18.61s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9999330617469364, 'learning_rate': 0.015673941035673426, 'min_data_in_leaf': 22, 'num_leaves': 29} and reward: 0.3882
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\xffs\x9e\xcc\x8d<X\r\x00\x00\x00learning_rateq\x02G?\x90\x0c\xd4`\x971~X\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K\x1du.' and reward: 0.3882
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\xffs\x9e\xcc\x8d<X\r\x00\x00\x00learning_rateq\x02G?\x90\x0c\xd4`\x971~X\x10\x00\x00\x00min_data_in_leafq\x03K\x16X\n\x00\x00\x00num_leavesq\x04K\x1du.' and reward: 0.3882
Time for Gradient Boosting hyperparameter optimization: 93.49993705749512
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9736845677222974, 'learning_rate': 0.04689901796733487, 'min_data_in_leaf': 10, 'num_leaves': 46}
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
 40%|████      | 2/5 [00:51<01:16, 25.56s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.20924598657881266, 'embedding_size_factor': 0.7005089791359167, 'layers.choice': 2, 'learning_rate': 0.00010182051344533066, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 3.602125918695252e-11} and reward: 0.3738
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xca\xc8\x92\x8e\x96n\xc4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6j\x91\xce~-\xf2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x1a\xb1\x0f\x19\xea\xc4\x14X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xc3\xcd\x8a\xa2\xb7\x9c\xc8u.' and reward: 0.3738
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xca\xc8\x92\x8e\x96n\xc4X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe6j\x91\xce~-\xf2X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x1a\xb1\x0f\x19\xea\xc4\x14X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xc3\xcd\x8a\xa2\xb7\x9c\xc8u.' and reward: 0.3738
 60%|██████    | 3/5 [01:42<01:06, 33.27s/it] 60%|██████    | 3/5 [01:42<01:08, 34.13s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.3616632099766422, 'embedding_size_factor': 1.0570312814236114, 'layers.choice': 2, 'learning_rate': 0.000980583072739864, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.528768753021817e-07} and reward: 0.3786
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd7%}r\xc1\x07\x1dX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\xe9\x99\xa2\t\x03\x9eX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?P\x10\xdd\x0e\x9e\x94EX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\x84\x84\xcf7R~\tu.' and reward: 0.3786
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd7%}r\xc1\x07\x1dX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\xe9\x99\xa2\t\x03\x9eX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?P\x10\xdd\x0e\x9e\x94EX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\x84\x84\xcf7R~\tu.' and reward: 0.3786
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 156.71485424041748
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.77s of the -134.42s of remaining time.
Ensemble size: 78
Ensemble weights: 
[0.19230769 0.06410256 0.19230769 0.24358974 0.06410256 0.20512821
 0.03846154]
	0.399	 = Validation accuracy score
	1.6s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 256.07s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fef82a43ac8>

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
[master 68dc337] ml_store
 1 file changed, 209 insertions(+)
To github.com:arita37/mlmodels_store.git
   4361f36..68dc337  master -> master





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
[master 775fe95] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   68dc337..775fe95  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.62it/s, avg_epoch_loss=5.24]
INFO:root:Epoch[0] Elapsed time 2.761 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.242686
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.242685604095459 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe93c0c5438>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe93c0c5438>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 92.59it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1061.0990397135417,
    "abs_error": 369.5638427734375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.448711478189095,
    "sMAPE": 0.5121981161847634,
    "MSIS": 97.94845912756378,
    "QuantileLoss[0.5]": 369.5638656616211,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.5745151876976,
    "NRMSE": 0.6857792671094232,
    "ND": 0.6483576189007676,
    "wQuantileLoss[0.5]": 0.6483576590554756,
    "mean_wQuantileLoss": 0.6483576590554756,
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
100%|██████████| 10/10 [00:01<00:00,  7.51it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.333 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe910373a90>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe910373a90>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 139.46it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.18it/s, avg_epoch_loss=5.27]
INFO:root:Epoch[0] Elapsed time 1.933 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.271683
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.271683168411255 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe91005a860>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe91005a860>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 163.35it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 233.45357259114584,
    "abs_error": 163.2577362060547,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.0817375681311296,
    "sMAPE": 0.274889346643925,
    "MSIS": 43.26950353407696,
    "QuantileLoss[0.5]": 163.2577362060547,
    "Coverage[0.5]": 0.75,
    "RMSE": 15.279187563190192,
    "NRMSE": 0.3216671065934777,
    "ND": 0.28641708106325386,
    "wQuantileLoss[0.5]": 0.28641708106325386,
    "mean_wQuantileLoss": 0.28641708106325386,
    "MAE_Coverage": 0.25
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
 30%|███       | 3/10 [00:12<00:29,  4.18s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:23<00:16,  4.05s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:34<00:03,  3.96s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:38<00:00,  3.87s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 38.684 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.862795
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.862794828414917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe910075b70>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe910075b70>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 162.29it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 53119.770833333336,
    "abs_error": 2704.8466796875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.922178375423314,
    "sMAPE": 1.410033658106263,
    "MSIS": 716.8870314864666,
    "QuantileLoss[0.5]": 2704.846450805664,
    "Coverage[0.5]": 1.0,
    "RMSE": 230.47726749797545,
    "NRMSE": 4.852152999957378,
    "ND": 4.745345052083334,
    "wQuantileLoss[0.5]": 4.745344650536253,
    "mean_wQuantileLoss": 4.745344650536253,
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
100%|██████████| 10/10 [00:00<00:00, 46.41it/s, avg_epoch_loss=5.17]
INFO:root:Epoch[0] Elapsed time 0.216 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.167233
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.167233467102051 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe9088abf60>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe9088abf60>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 144.77it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 539.26123046875,
    "abs_error": 194.99441528320312,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.292023211208664,
    "sMAPE": 0.3250103909280549,
    "MSIS": 51.68092683068303,
    "QuantileLoss[0.5]": 194.99440383911133,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 23.22199884740222,
    "NRMSE": 0.48888418626109936,
    "ND": 0.3420954654091283,
    "wQuantileLoss[0.5]": 0.3420954453317743,
    "mean_wQuantileLoss": 0.3420954453317743,
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
100%|██████████| 10/10 [00:01<00:00,  8.40it/s, avg_epoch_loss=161]
INFO:root:Epoch[0] Elapsed time 1.191 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=161.108291
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 161.1082910709855 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe908ad4cf8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe908ad4cf8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 148.38it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 439.8828247741723,
    "abs_error": 223.62751574072416,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.4817446980843618,
    "sMAPE": 0.42600252840216285,
    "MSIS": 59.26978792337448,
    "QuantileLoss[0.5]": 223.62751574072416,
    "Coverage[0.5]": 0.3333333333333333,
    "RMSE": 20.973383722570194,
    "NRMSE": 0.441544920475162,
    "ND": 0.3923289749837266,
    "wQuantileLoss[0.5]": 0.3923289749837266,
    "mean_wQuantileLoss": 0.3923289749837266,
    "MAE_Coverage": 0.16666666666666669
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
 10%|█         | 1/10 [01:53<17:05, 113.97s/it, avg_epoch_loss=0.582] 20%|██        | 2/10 [04:51<17:45, 133.13s/it, avg_epoch_loss=0.565] 30%|███       | 3/10 [08:03<17:34, 150.70s/it, avg_epoch_loss=0.548] 40%|████      | 4/10 [11:37<16:57, 169.54s/it, avg_epoch_loss=0.531] 50%|█████     | 5/10 [15:04<15:04, 180.90s/it, avg_epoch_loss=0.515] 60%|██████    | 6/10 [18:19<12:20, 185.10s/it, avg_epoch_loss=0.499] 70%|███████   | 7/10 [21:16<09:08, 182.86s/it, avg_epoch_loss=0.484] 80%|████████  | 8/10 [24:25<06:08, 184.48s/it, avg_epoch_loss=0.47]  90%|█████████ | 9/10 [27:31<03:04, 184.98s/it, avg_epoch_loss=0.457]100%|██████████| 10/10 [30:58<00:00, 191.61s/it, avg_epoch_loss=0.447]100%|██████████| 10/10 [30:58<00:00, 185.85s/it, avg_epoch_loss=0.447]
INFO:root:Epoch[0] Elapsed time 1858.481 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.446552
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.4465524971485138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fe9089a06d8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fe9089a06d8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 20.80it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 164.05573527018228,
    "abs_error": 114.22518920898438,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.7568503720907435,
    "sMAPE": 0.20283677998139085,
    "MSIS": 30.274016501293268,
    "QuantileLoss[0.5]": 114.22519302368164,
    "Coverage[0.5]": 0.4166666666666667,
    "RMSE": 12.808424386714483,
    "NRMSE": 0.2696510397203049,
    "ND": 0.2003950687876919,
    "wQuantileLoss[0.5]": 0.20039507548014324,
    "mean_wQuantileLoss": 0.20039507548014324,
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
Already up to date.
[master feb5003] ml_store
 1 file changed, 500 insertions(+)
To github.com:arita37/mlmodels_store.git
   775fe95..feb5003  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f86c7bad4e0> 

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
Already up to date.
[master f66a19f] ml_store
 1 file changed, 109 insertions(+)
To github.com:arita37/mlmodels_store.git
   feb5003..f66a19f  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
   9.10423603e-01  1.21987438e+00]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
  -1.02887218e-01  1.71647264e-02]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 1.27991386e+00 -8.71422066e-01 -3.24032329e-01 -8.64829941e-01
  -9.68539694e-01  6.08749082e-01  5.07984337e-01  5.61638097e-01
   1.51475038e+00 -1.51107661e+00]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 6.67591795e-01 -4.52524973e-01 -6.05981321e-01  1.16128569e+00
  -1.44620987e+00  1.06996554e+00  1.92381543e+00 -1.04553425e+00
   3.55284507e-01  1.80358898e+00]
 [ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.06523311e+00 -6.64867767e-01  1.00806543e+00 -1.94504696e+00
  -1.23017555e+00 -9.15424368e-01  3.37220938e-01  1.22515585e+00
  -1.05354607e+00  7.85226920e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f2edc72bda0>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f2efd799fd0> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
   7.21711292e-01  6.74396105e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
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
[[ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]
 [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
   0.42952614  0.1340482   1.20205486  0.10622272]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]]
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
[master 29c8998] ml_store
 1 file changed, 297 insertions(+)
To github.com:arita37/mlmodels_store.git
   f66a19f..29c8998  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140092106158032
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140092106157808
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140092106156576
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140092106156128
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140092106155624
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140092106155288

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
grad_step = 000000, loss = 1.083826
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.925779
grad_step = 000002, loss = 0.795971
grad_step = 000003, loss = 0.655361
grad_step = 000004, loss = 0.510385
grad_step = 000005, loss = 0.366350
grad_step = 000006, loss = 0.277632
grad_step = 000007, loss = 0.255866
grad_step = 000008, loss = 0.213651
grad_step = 000009, loss = 0.141989
grad_step = 000010, loss = 0.084176
grad_step = 000011, loss = 0.048326
grad_step = 000012, loss = 0.035526
grad_step = 000013, loss = 0.039450
grad_step = 000014, loss = 0.048370
grad_step = 000015, loss = 0.051228
grad_step = 000016, loss = 0.043261
grad_step = 000017, loss = 0.029220
grad_step = 000018, loss = 0.017221
grad_step = 000019, loss = 0.013924
grad_step = 000020, loss = 0.020649
grad_step = 000021, loss = 0.030406
grad_step = 000022, loss = 0.033232
grad_step = 000023, loss = 0.026749
grad_step = 000024, loss = 0.016674
grad_step = 000025, loss = 0.009544
grad_step = 000026, loss = 0.007871
grad_step = 000027, loss = 0.010239
grad_step = 000028, loss = 0.013637
grad_step = 000029, loss = 0.015576
grad_step = 000030, loss = 0.015089
grad_step = 000031, loss = 0.012659
grad_step = 000032, loss = 0.009562
grad_step = 000033, loss = 0.007143
grad_step = 000034, loss = 0.006270
grad_step = 000035, loss = 0.006990
grad_step = 000036, loss = 0.008426
grad_step = 000037, loss = 0.009335
grad_step = 000038, loss = 0.009090
grad_step = 000039, loss = 0.007995
grad_step = 000040, loss = 0.006797
grad_step = 000041, loss = 0.006062
grad_step = 000042, loss = 0.005933
grad_step = 000043, loss = 0.006227
grad_step = 000044, loss = 0.006631
grad_step = 000045, loss = 0.006810
grad_step = 000046, loss = 0.006588
grad_step = 000047, loss = 0.006076
grad_step = 000048, loss = 0.005539
grad_step = 000049, loss = 0.005174
grad_step = 000050, loss = 0.005040
grad_step = 000051, loss = 0.005086
grad_step = 000052, loss = 0.005197
grad_step = 000053, loss = 0.005249
grad_step = 000054, loss = 0.005169
grad_step = 000055, loss = 0.004977
grad_step = 000056, loss = 0.004771
grad_step = 000057, loss = 0.004638
grad_step = 000058, loss = 0.004605
grad_step = 000059, loss = 0.004651
grad_step = 000060, loss = 0.004711
grad_step = 000061, loss = 0.004695
grad_step = 000062, loss = 0.004563
grad_step = 000063, loss = 0.004383
grad_step = 000064, loss = 0.004262
grad_step = 000065, loss = 0.004238
grad_step = 000066, loss = 0.004265
grad_step = 000067, loss = 0.004273
grad_step = 000068, loss = 0.004220
grad_step = 000069, loss = 0.004113
grad_step = 000070, loss = 0.004003
grad_step = 000071, loss = 0.003941
grad_step = 000072, loss = 0.003931
grad_step = 000073, loss = 0.003930
grad_step = 000074, loss = 0.003898
grad_step = 000075, loss = 0.003837
grad_step = 000076, loss = 0.003771
grad_step = 000077, loss = 0.003715
grad_step = 000078, loss = 0.003670
grad_step = 000079, loss = 0.003632
grad_step = 000080, loss = 0.003591
grad_step = 000081, loss = 0.003543
grad_step = 000082, loss = 0.003485
grad_step = 000083, loss = 0.003428
grad_step = 000084, loss = 0.003380
grad_step = 000085, loss = 0.003337
grad_step = 000086, loss = 0.003294
grad_step = 000087, loss = 0.003244
grad_step = 000088, loss = 0.003186
grad_step = 000089, loss = 0.003130
grad_step = 000090, loss = 0.003082
grad_step = 000091, loss = 0.003038
grad_step = 000092, loss = 0.002985
grad_step = 000093, loss = 0.002923
grad_step = 000094, loss = 0.002866
grad_step = 000095, loss = 0.002814
grad_step = 000096, loss = 0.002760
grad_step = 000097, loss = 0.002703
grad_step = 000098, loss = 0.002647
grad_step = 000099, loss = 0.002590
grad_step = 000100, loss = 0.002533
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002480
grad_step = 000102, loss = 0.002424
grad_step = 000103, loss = 0.002368
grad_step = 000104, loss = 0.002315
grad_step = 000105, loss = 0.002259
grad_step = 000106, loss = 0.002206
grad_step = 000107, loss = 0.002156
grad_step = 000108, loss = 0.002128
grad_step = 000109, loss = 0.002060
grad_step = 000110, loss = 0.002013
grad_step = 000111, loss = 0.001967
grad_step = 000112, loss = 0.001918
grad_step = 000113, loss = 0.001878
grad_step = 000114, loss = 0.001828
grad_step = 000115, loss = 0.001794
grad_step = 000116, loss = 0.001746
grad_step = 000117, loss = 0.001717
grad_step = 000118, loss = 0.001673
grad_step = 000119, loss = 0.001642
grad_step = 000120, loss = 0.001604
grad_step = 000121, loss = 0.001576
grad_step = 000122, loss = 0.001539
grad_step = 000123, loss = 0.001513
grad_step = 000124, loss = 0.001480
grad_step = 000125, loss = 0.001453
grad_step = 000126, loss = 0.001433
grad_step = 000127, loss = 0.001409
grad_step = 000128, loss = 0.001390
grad_step = 000129, loss = 0.001369
grad_step = 000130, loss = 0.001351
grad_step = 000131, loss = 0.001330
grad_step = 000132, loss = 0.001313
grad_step = 000133, loss = 0.001297
grad_step = 000134, loss = 0.001279
grad_step = 000135, loss = 0.001261
grad_step = 000136, loss = 0.001239
grad_step = 000137, loss = 0.001221
grad_step = 000138, loss = 0.001201
grad_step = 000139, loss = 0.001179
grad_step = 000140, loss = 0.001157
grad_step = 000141, loss = 0.001134
grad_step = 000142, loss = 0.001111
grad_step = 000143, loss = 0.001093
grad_step = 000144, loss = 0.001074
grad_step = 000145, loss = 0.001053
grad_step = 000146, loss = 0.001032
grad_step = 000147, loss = 0.001009
grad_step = 000148, loss = 0.000983
grad_step = 000149, loss = 0.000960
grad_step = 000150, loss = 0.000939
grad_step = 000151, loss = 0.000917
grad_step = 000152, loss = 0.000895
grad_step = 000153, loss = 0.000872
grad_step = 000154, loss = 0.000850
grad_step = 000155, loss = 0.000828
grad_step = 000156, loss = 0.000807
grad_step = 000157, loss = 0.000787
grad_step = 000158, loss = 0.000770
grad_step = 000159, loss = 0.000756
grad_step = 000160, loss = 0.000747
grad_step = 000161, loss = 0.000735
grad_step = 000162, loss = 0.000719
grad_step = 000163, loss = 0.000710
grad_step = 000164, loss = 0.000719
grad_step = 000165, loss = 0.000730
grad_step = 000166, loss = 0.000678
grad_step = 000167, loss = 0.000653
grad_step = 000168, loss = 0.000675
grad_step = 000169, loss = 0.000655
grad_step = 000170, loss = 0.000635
grad_step = 000171, loss = 0.000644
grad_step = 000172, loss = 0.000623
grad_step = 000173, loss = 0.000614
grad_step = 000174, loss = 0.000624
grad_step = 000175, loss = 0.000604
grad_step = 000176, loss = 0.000596
grad_step = 000177, loss = 0.000603
grad_step = 000178, loss = 0.000590
grad_step = 000179, loss = 0.000585
grad_step = 000180, loss = 0.000586
grad_step = 000181, loss = 0.000576
grad_step = 000182, loss = 0.000571
grad_step = 000183, loss = 0.000574
grad_step = 000184, loss = 0.000567
grad_step = 000185, loss = 0.000560
grad_step = 000186, loss = 0.000560
grad_step = 000187, loss = 0.000554
grad_step = 000188, loss = 0.000547
grad_step = 000189, loss = 0.000549
grad_step = 000190, loss = 0.000544
grad_step = 000191, loss = 0.000538
grad_step = 000192, loss = 0.000537
grad_step = 000193, loss = 0.000534
grad_step = 000194, loss = 0.000529
grad_step = 000195, loss = 0.000526
grad_step = 000196, loss = 0.000524
grad_step = 000197, loss = 0.000522
grad_step = 000198, loss = 0.000519
grad_step = 000199, loss = 0.000522
grad_step = 000200, loss = 0.000527
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000533
grad_step = 000202, loss = 0.000534
grad_step = 000203, loss = 0.000525
grad_step = 000204, loss = 0.000507
grad_step = 000205, loss = 0.000494
grad_step = 000206, loss = 0.000494
grad_step = 000207, loss = 0.000501
grad_step = 000208, loss = 0.000506
grad_step = 000209, loss = 0.000505
grad_step = 000210, loss = 0.000498
grad_step = 000211, loss = 0.000487
grad_step = 000212, loss = 0.000476
grad_step = 000213, loss = 0.000473
grad_step = 000214, loss = 0.000476
grad_step = 000215, loss = 0.000480
grad_step = 000216, loss = 0.000479
grad_step = 000217, loss = 0.000474
grad_step = 000218, loss = 0.000466
grad_step = 000219, loss = 0.000460
grad_step = 000220, loss = 0.000456
grad_step = 000221, loss = 0.000456
grad_step = 000222, loss = 0.000458
grad_step = 000223, loss = 0.000460
grad_step = 000224, loss = 0.000460
grad_step = 000225, loss = 0.000459
grad_step = 000226, loss = 0.000456
grad_step = 000227, loss = 0.000452
grad_step = 000228, loss = 0.000446
grad_step = 000229, loss = 0.000439
grad_step = 000230, loss = 0.000435
grad_step = 000231, loss = 0.000432
grad_step = 000232, loss = 0.000432
grad_step = 000233, loss = 0.000433
grad_step = 000234, loss = 0.000436
grad_step = 000235, loss = 0.000441
grad_step = 000236, loss = 0.000452
grad_step = 000237, loss = 0.000469
grad_step = 000238, loss = 0.000483
grad_step = 000239, loss = 0.000478
grad_step = 000240, loss = 0.000444
grad_step = 000241, loss = 0.000416
grad_step = 000242, loss = 0.000419
grad_step = 000243, loss = 0.000438
grad_step = 000244, loss = 0.000443
grad_step = 000245, loss = 0.000425
grad_step = 000246, loss = 0.000407
grad_step = 000247, loss = 0.000408
grad_step = 000248, loss = 0.000419
grad_step = 000249, loss = 0.000422
grad_step = 000250, loss = 0.000412
grad_step = 000251, loss = 0.000399
grad_step = 000252, loss = 0.000397
grad_step = 000253, loss = 0.000404
grad_step = 000254, loss = 0.000410
grad_step = 000255, loss = 0.000407
grad_step = 000256, loss = 0.000399
grad_step = 000257, loss = 0.000392
grad_step = 000258, loss = 0.000390
grad_step = 000259, loss = 0.000392
grad_step = 000260, loss = 0.000393
grad_step = 000261, loss = 0.000392
grad_step = 000262, loss = 0.000387
grad_step = 000263, loss = 0.000381
grad_step = 000264, loss = 0.000377
grad_step = 000265, loss = 0.000375
grad_step = 000266, loss = 0.000375
grad_step = 000267, loss = 0.000377
grad_step = 000268, loss = 0.000379
grad_step = 000269, loss = 0.000382
grad_step = 000270, loss = 0.000385
grad_step = 000271, loss = 0.000389
grad_step = 000272, loss = 0.000392
grad_step = 000273, loss = 0.000394
grad_step = 000274, loss = 0.000391
grad_step = 000275, loss = 0.000383
grad_step = 000276, loss = 0.000371
grad_step = 000277, loss = 0.000365
grad_step = 000278, loss = 0.000365
grad_step = 000279, loss = 0.000373
grad_step = 000280, loss = 0.000385
grad_step = 000281, loss = 0.000396
grad_step = 000282, loss = 0.000399
grad_step = 000283, loss = 0.000395
grad_step = 000284, loss = 0.000379
grad_step = 000285, loss = 0.000364
grad_step = 000286, loss = 0.000360
grad_step = 000287, loss = 0.000368
grad_step = 000288, loss = 0.000373
grad_step = 000289, loss = 0.000368
grad_step = 000290, loss = 0.000353
grad_step = 000291, loss = 0.000345
grad_step = 000292, loss = 0.000348
grad_step = 000293, loss = 0.000355
grad_step = 000294, loss = 0.000357
grad_step = 000295, loss = 0.000347
grad_step = 000296, loss = 0.000337
grad_step = 000297, loss = 0.000335
grad_step = 000298, loss = 0.000340
grad_step = 000299, loss = 0.000343
grad_step = 000300, loss = 0.000339
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000332
grad_step = 000302, loss = 0.000329
grad_step = 000303, loss = 0.000331
grad_step = 000304, loss = 0.000334
grad_step = 000305, loss = 0.000335
grad_step = 000306, loss = 0.000337
grad_step = 000307, loss = 0.000347
grad_step = 000308, loss = 0.000373
grad_step = 000309, loss = 0.000432
grad_step = 000310, loss = 0.000499
grad_step = 000311, loss = 0.000540
grad_step = 000312, loss = 0.000442
grad_step = 000313, loss = 0.000335
grad_step = 000314, loss = 0.000346
grad_step = 000315, loss = 0.000417
grad_step = 000316, loss = 0.000406
grad_step = 000317, loss = 0.000333
grad_step = 000318, loss = 0.000331
grad_step = 000319, loss = 0.000379
grad_step = 000320, loss = 0.000371
grad_step = 000321, loss = 0.000326
grad_step = 000322, loss = 0.000320
grad_step = 000323, loss = 0.000347
grad_step = 000324, loss = 0.000350
grad_step = 000325, loss = 0.000321
grad_step = 000326, loss = 0.000314
grad_step = 000327, loss = 0.000330
grad_step = 000328, loss = 0.000331
grad_step = 000329, loss = 0.000314
grad_step = 000330, loss = 0.000307
grad_step = 000331, loss = 0.000316
grad_step = 000332, loss = 0.000321
grad_step = 000333, loss = 0.000308
grad_step = 000334, loss = 0.000301
grad_step = 000335, loss = 0.000307
grad_step = 000336, loss = 0.000310
grad_step = 000337, loss = 0.000302
grad_step = 000338, loss = 0.000296
grad_step = 000339, loss = 0.000299
grad_step = 000340, loss = 0.000304
grad_step = 000341, loss = 0.000300
grad_step = 000342, loss = 0.000293
grad_step = 000343, loss = 0.000291
grad_step = 000344, loss = 0.000294
grad_step = 000345, loss = 0.000295
grad_step = 000346, loss = 0.000293
grad_step = 000347, loss = 0.000288
grad_step = 000348, loss = 0.000286
grad_step = 000349, loss = 0.000287
grad_step = 000350, loss = 0.000288
grad_step = 000351, loss = 0.000287
grad_step = 000352, loss = 0.000284
grad_step = 000353, loss = 0.000282
grad_step = 000354, loss = 0.000281
grad_step = 000355, loss = 0.000282
grad_step = 000356, loss = 0.000281
grad_step = 000357, loss = 0.000280
grad_step = 000358, loss = 0.000278
grad_step = 000359, loss = 0.000276
grad_step = 000360, loss = 0.000275
grad_step = 000361, loss = 0.000274
grad_step = 000362, loss = 0.000274
grad_step = 000363, loss = 0.000273
grad_step = 000364, loss = 0.000273
grad_step = 000365, loss = 0.000272
grad_step = 000366, loss = 0.000271
grad_step = 000367, loss = 0.000270
grad_step = 000368, loss = 0.000270
grad_step = 000369, loss = 0.000271
grad_step = 000370, loss = 0.000275
grad_step = 000371, loss = 0.000283
grad_step = 000372, loss = 0.000293
grad_step = 000373, loss = 0.000309
grad_step = 000374, loss = 0.000316
grad_step = 000375, loss = 0.000312
grad_step = 000376, loss = 0.000287
grad_step = 000377, loss = 0.000265
grad_step = 000378, loss = 0.000261
grad_step = 000379, loss = 0.000275
grad_step = 000380, loss = 0.000286
grad_step = 000381, loss = 0.000279
grad_step = 000382, loss = 0.000263
grad_step = 000383, loss = 0.000258
grad_step = 000384, loss = 0.000266
grad_step = 000385, loss = 0.000275
grad_step = 000386, loss = 0.000271
grad_step = 000387, loss = 0.000264
grad_step = 000388, loss = 0.000266
grad_step = 000389, loss = 0.000282
grad_step = 000390, loss = 0.000299
grad_step = 000391, loss = 0.000314
grad_step = 000392, loss = 0.000315
grad_step = 000393, loss = 0.000314
grad_step = 000394, loss = 0.000295
grad_step = 000395, loss = 0.000271
grad_step = 000396, loss = 0.000252
grad_step = 000397, loss = 0.000254
grad_step = 000398, loss = 0.000274
grad_step = 000399, loss = 0.000285
grad_step = 000400, loss = 0.000278
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000258
grad_step = 000402, loss = 0.000244
grad_step = 000403, loss = 0.000244
grad_step = 000404, loss = 0.000250
grad_step = 000405, loss = 0.000255
grad_step = 000406, loss = 0.000257
grad_step = 000407, loss = 0.000261
grad_step = 000408, loss = 0.000258
grad_step = 000409, loss = 0.000249
grad_step = 000410, loss = 0.000238
grad_step = 000411, loss = 0.000234
grad_step = 000412, loss = 0.000238
grad_step = 000413, loss = 0.000244
grad_step = 000414, loss = 0.000245
grad_step = 000415, loss = 0.000242
grad_step = 000416, loss = 0.000239
grad_step = 000417, loss = 0.000236
grad_step = 000418, loss = 0.000233
grad_step = 000419, loss = 0.000229
grad_step = 000420, loss = 0.000227
grad_step = 000421, loss = 0.000228
grad_step = 000422, loss = 0.000231
grad_step = 000423, loss = 0.000234
grad_step = 000424, loss = 0.000236
grad_step = 000425, loss = 0.000240
grad_step = 000426, loss = 0.000247
grad_step = 000427, loss = 0.000256
grad_step = 000428, loss = 0.000258
grad_step = 000429, loss = 0.000256
grad_step = 000430, loss = 0.000246
grad_step = 000431, loss = 0.000233
grad_step = 000432, loss = 0.000223
grad_step = 000433, loss = 0.000219
grad_step = 000434, loss = 0.000222
grad_step = 000435, loss = 0.000227
grad_step = 000436, loss = 0.000233
grad_step = 000437, loss = 0.000237
grad_step = 000438, loss = 0.000241
grad_step = 000439, loss = 0.000239
grad_step = 000440, loss = 0.000235
grad_step = 000441, loss = 0.000228
grad_step = 000442, loss = 0.000221
grad_step = 000443, loss = 0.000215
grad_step = 000444, loss = 0.000212
grad_step = 000445, loss = 0.000212
grad_step = 000446, loss = 0.000215
grad_step = 000447, loss = 0.000218
grad_step = 000448, loss = 0.000220
grad_step = 000449, loss = 0.000222
grad_step = 000450, loss = 0.000223
grad_step = 000451, loss = 0.000225
grad_step = 000452, loss = 0.000226
grad_step = 000453, loss = 0.000228
grad_step = 000454, loss = 0.000229
grad_step = 000455, loss = 0.000229
grad_step = 000456, loss = 0.000226
grad_step = 000457, loss = 0.000220
grad_step = 000458, loss = 0.000213
grad_step = 000459, loss = 0.000207
grad_step = 000460, loss = 0.000204
grad_step = 000461, loss = 0.000204
grad_step = 000462, loss = 0.000205
grad_step = 000463, loss = 0.000206
grad_step = 000464, loss = 0.000208
grad_step = 000465, loss = 0.000210
grad_step = 000466, loss = 0.000213
grad_step = 000467, loss = 0.000218
grad_step = 000468, loss = 0.000224
grad_step = 000469, loss = 0.000231
grad_step = 000470, loss = 0.000241
grad_step = 000471, loss = 0.000247
grad_step = 000472, loss = 0.000252
grad_step = 000473, loss = 0.000246
grad_step = 000474, loss = 0.000235
grad_step = 000475, loss = 0.000216
grad_step = 000476, loss = 0.000201
grad_step = 000477, loss = 0.000197
grad_step = 000478, loss = 0.000202
grad_step = 000479, loss = 0.000211
grad_step = 000480, loss = 0.000217
grad_step = 000481, loss = 0.000219
grad_step = 000482, loss = 0.000214
grad_step = 000483, loss = 0.000206
grad_step = 000484, loss = 0.000198
grad_step = 000485, loss = 0.000193
grad_step = 000486, loss = 0.000193
grad_step = 000487, loss = 0.000196
grad_step = 000488, loss = 0.000200
grad_step = 000489, loss = 0.000201
grad_step = 000490, loss = 0.000202
grad_step = 000491, loss = 0.000201
grad_step = 000492, loss = 0.000199
grad_step = 000493, loss = 0.000197
grad_step = 000494, loss = 0.000196
grad_step = 000495, loss = 0.000194
grad_step = 000496, loss = 0.000192
grad_step = 000497, loss = 0.000190
grad_step = 000498, loss = 0.000189
grad_step = 000499, loss = 0.000187
grad_step = 000500, loss = 0.000186
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000187
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
[[0.86479867 0.85639936 0.9286983  0.9409574  0.9923568 ]
 [0.84741926 0.92561793 0.9443172  1.0156721  0.9922182 ]
 [0.9008298  0.9273345  1.0010777  0.984519   0.94729143]
 [0.904547   1.0004873  0.99790573 0.955492   0.91218626]
 [0.9691901  0.9970138  0.9620675  0.9083855  0.87106377]
 [0.9868469  0.95428175 0.9238314  0.8660096  0.8658279 ]
 [0.95114183 0.8975079  0.84880155 0.8447225  0.80713016]
 [0.89592266 0.8263509  0.8610336  0.8198771  0.8272053 ]
 [0.83144224 0.83836865 0.8167499  0.8408533  0.85225415]
 [0.8326882  0.8132102  0.85032725 0.8400923  0.81741846]
 [0.7958647  0.8272669  0.8587589  0.8303807  0.9049143 ]
 [0.8334819  0.8431128  0.80446345 0.9112038  0.9408685 ]
 [0.85434246 0.8529919  0.9277237  0.948425   1.0003376 ]
 [0.833529   0.94231343 0.9540551  1.0233153  0.9939997 ]
 [0.9050276  0.9391016  1.0061705  0.9818272  0.93432236]
 [0.917789   1.0042965  0.9892179  0.94210744 0.88927084]
 [0.9802501  0.9875518  0.9444553  0.88158315 0.8442707 ]
 [0.9798856  0.9371248  0.90475595 0.8402609  0.8459734 ]
 [0.9371449  0.8939102  0.8362344  0.8382851  0.8107841 ]
 [0.89392364 0.83024865 0.8535761  0.82421136 0.83711624]
 [0.8395879  0.8482753  0.82219577 0.84544003 0.8635723 ]
 [0.84629154 0.82669055 0.8546928  0.85074836 0.8245946 ]
 [0.8092965  0.8404756  0.8703524  0.8390043  0.9101014 ]
 [0.8376914  0.8533763  0.8140302  0.9120985  0.94096   ]
 [0.86856365 0.8631869  0.9266865  0.9439024  0.9988579 ]
 [0.8521992  0.93234676 0.947615   1.025099   1.0070006 ]
 [0.9051112  0.9387361  1.0111574  0.99841064 0.96088475]
 [0.9156853  1.012227   1.0119932  0.96879554 0.92421883]
 [0.97382    1.0101334  0.9780274  0.91418713 0.87731683]
 [0.99369717 0.9659794  0.9302951  0.8736732  0.87214077]
 [0.9572793  0.90695095 0.856717   0.8506398  0.81488156]]

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
[master d1bf361] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   29c8998..d1bf361  master -> master





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
[master c5936c5] ml_store
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   d1bf361..c5936c5  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 10969193.31B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 158572.74B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 3884032/440473133 [00:00<00:11, 38829803.20B/s]  2%|▏         | 9110528/440473133 [00:00<00:10, 42074446.79B/s]  3%|▎         | 14468096/440473133 [00:00<00:09, 44970323.31B/s]  4%|▍         | 19814400/440473133 [00:00<00:08, 47218070.19B/s]  6%|▌         | 25321472/440473133 [00:00<00:08, 49325098.46B/s]  7%|▋         | 30570496/440473133 [00:00<00:08, 50232220.28B/s]  8%|▊         | 36162560/440473133 [00:00<00:07, 51810913.24B/s]  9%|▉         | 41583616/440473133 [00:00<00:07, 52507887.79B/s] 11%|█         | 46968832/440473133 [00:00<00:07, 52903378.08B/s] 12%|█▏        | 52136960/440473133 [00:01<00:07, 52421815.88B/s] 13%|█▎        | 57294848/440473133 [00:01<00:07, 51801793.10B/s] 14%|█▍        | 62837760/440473133 [00:01<00:07, 52837141.07B/s] 16%|█▌        | 68366336/440473133 [00:01<00:06, 53547695.91B/s] 17%|█▋        | 73699328/440473133 [00:01<00:06, 52435933.67B/s] 18%|█▊        | 79147008/440473133 [00:01<00:06, 53031009.81B/s] 19%|█▉        | 84460544/440473133 [00:01<00:06, 53058307.67B/s] 20%|██        | 89872384/440473133 [00:01<00:06, 53370533.31B/s] 22%|██▏       | 95302656/440473133 [00:01<00:06, 53646106.35B/s] 23%|██▎       | 100877312/440473133 [00:01<00:06, 54258692.21B/s] 24%|██▍       | 106304512/440473133 [00:02<00:06, 54138381.82B/s] 25%|██▌       | 111719424/440473133 [00:02<00:06, 53790362.12B/s] 27%|██▋       | 117099520/440473133 [00:02<00:06, 53578368.38B/s] 28%|██▊       | 122459136/440473133 [00:02<00:06, 52989943.47B/s] 29%|██▉       | 127784960/440473133 [00:02<00:05, 53065889.45B/s] 30%|███       | 133093376/440473133 [00:02<00:05, 52868784.26B/s] 31%|███▏      | 138382336/440473133 [00:02<00:05, 52599124.64B/s] 33%|███▎      | 143643648/440473133 [00:02<00:05, 51014454.52B/s] 34%|███▍      | 148756480/440473133 [00:02<00:05, 50688255.41B/s] 35%|███▍      | 154063872/440473133 [00:02<00:05, 51380857.16B/s] 36%|███▌      | 159312896/440473133 [00:03<00:05, 51708002.65B/s] 37%|███▋      | 164490240/440473133 [00:03<00:05, 51409367.29B/s] 39%|███▊      | 169766912/440473133 [00:03<00:05, 51807997.60B/s] 40%|███▉      | 174952448/440473133 [00:03<00:05, 51801439.57B/s] 41%|████      | 180249600/440473133 [00:03<00:04, 52144224.16B/s] 42%|████▏     | 185466880/440473133 [00:03<00:04, 51999261.26B/s] 43%|████▎     | 190668800/440473133 [00:03<00:04, 51777077.70B/s] 45%|████▍     | 196109312/440473133 [00:03<00:04, 52535260.63B/s] 46%|████▌     | 201384960/440473133 [00:03<00:04, 52600883.71B/s] 47%|████▋     | 206727168/440473133 [00:03<00:04, 52839547.37B/s] 48%|████▊     | 212145152/440473133 [00:04<00:04, 53233812.26B/s] 49%|████▉     | 217470976/440473133 [00:04<00:04, 52456306.16B/s] 51%|█████     | 222853120/440473133 [00:04<00:04, 52857865.72B/s] 52%|█████▏    | 228361216/440473133 [00:04<00:03, 53500531.54B/s] 53%|█████▎    | 234086400/440473133 [00:04<00:03, 54569975.39B/s] 54%|█████▍    | 239552512/440473133 [00:04<00:03, 54535626.58B/s] 56%|█████▌    | 245012480/440473133 [00:04<00:03, 54264642.52B/s] 57%|█████▋    | 250443776/440473133 [00:04<00:03, 52283403.57B/s] 58%|█████▊    | 255877120/440473133 [00:04<00:03, 52880232.92B/s] 59%|█████▉    | 261199872/440473133 [00:04<00:03, 52982750.23B/s] 61%|██████    | 266509312/440473133 [00:05<00:03, 52827299.94B/s] 62%|██████▏   | 271800320/440473133 [00:05<00:03, 52703759.68B/s] 63%|██████▎   | 277394432/440473133 [00:05<00:03, 53634907.34B/s] 64%|██████▍   | 282766336/440473133 [00:05<00:02, 53610800.03B/s] 65%|██████▌   | 288133120/440473133 [00:05<00:02, 53512562.59B/s] 67%|██████▋   | 293488640/440473133 [00:05<00:02, 53211694.34B/s] 68%|██████▊   | 298813440/440473133 [00:05<00:02, 53045004.32B/s] 69%|██████▉   | 304208896/440473133 [00:05<00:02, 53311635.44B/s] 70%|███████   | 309542912/440473133 [00:05<00:02, 52967209.94B/s] 71%|███████▏  | 314889216/440473133 [00:05<00:02, 53113345.02B/s] 73%|███████▎  | 320276480/440473133 [00:06<00:02, 53335454.61B/s] 74%|███████▍  | 325611520/440473133 [00:06<00:02, 53022245.73B/s] 75%|███████▌  | 331129856/440473133 [00:06<00:02, 53652162.68B/s] 76%|███████▋  | 336577536/440473133 [00:06<00:01, 53896325.54B/s] 78%|███████▊  | 342083584/440473133 [00:06<00:01, 54237366.19B/s] 79%|███████▉  | 347509760/440473133 [00:06<00:01, 53031585.90B/s] 80%|████████  | 352835584/440473133 [00:06<00:01, 53098637.14B/s] 81%|████████▏ | 358151168/440473133 [00:06<00:01, 52395895.37B/s] 83%|████████▎ | 363587584/440473133 [00:06<00:01, 52970521.97B/s] 84%|████████▎ | 368890880/440473133 [00:06<00:01, 52732759.96B/s] 85%|████████▍ | 374168576/440473133 [00:07<00:01, 52655164.72B/s] 86%|████████▌ | 379480064/440473133 [00:07<00:01, 52791993.86B/s] 87%|████████▋ | 384923648/440473133 [00:07<00:01, 53274530.90B/s] 89%|████████▊ | 390303744/440473133 [00:07<00:00, 53428353.73B/s] 90%|████████▉ | 395753472/440473133 [00:07<00:00, 53744011.25B/s] 91%|█████████ | 401199104/440473133 [00:07<00:00, 53953804.80B/s] 92%|█████████▏| 406596608/440473133 [00:07<00:00, 53697324.87B/s] 94%|█████████▎| 411968512/440473133 [00:07<00:00, 52029781.01B/s] 95%|█████████▍| 417329152/440473133 [00:07<00:00, 52492221.73B/s] 96%|█████████▌| 422596608/440473133 [00:07<00:00, 52545305.83B/s] 97%|█████████▋| 427943936/440473133 [00:08<00:00, 52813410.41B/s] 98%|█████████▊| 433230848/440473133 [00:08<00:00, 52304735.96B/s]100%|█████████▉| 438548480/440473133 [00:08<00:00, 52560832.92B/s]100%|██████████| 440473133/440473133 [00:08<00:00, 52867621.48B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
4530176/7094233 [==================>...........] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   7%|▋         | 153/2118 [00:00<00:01, 1526.23it/s]Processing text_left with encode:  29%|██▉       | 615/2118 [00:00<00:00, 1909.63it/s]Processing text_left with encode:  48%|████▊     | 1009/2118 [00:00<00:00, 2258.72it/s]Processing text_left with encode:  72%|███████▏  | 1523/2118 [00:00<00:00, 2715.34it/s]Processing text_left with encode:  93%|█████████▎| 1980/2118 [00:00<00:00, 3090.61it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 4008.40it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 169/18841 [00:00<00:11, 1685.58it/s]Processing text_right with encode:   2%|▏         | 341/18841 [00:00<00:10, 1694.16it/s]Processing text_right with encode:   3%|▎         | 525/18841 [00:00<00:10, 1731.35it/s]Processing text_right with encode:   4%|▍         | 713/18841 [00:00<00:10, 1770.96it/s]Processing text_right with encode:   5%|▍         | 886/18841 [00:00<00:10, 1757.17it/s]Processing text_right with encode:   6%|▌         | 1046/18841 [00:00<00:10, 1706.62it/s]Processing text_right with encode:   6%|▋         | 1199/18841 [00:00<00:10, 1649.47it/s]Processing text_right with encode:   7%|▋         | 1376/18841 [00:00<00:10, 1681.98it/s]Processing text_right with encode:   8%|▊         | 1554/18841 [00:00<00:10, 1708.79it/s]Processing text_right with encode:   9%|▉         | 1745/18841 [00:01<00:09, 1762.03it/s]Processing text_right with encode:  10%|█         | 1924/18841 [00:01<00:09, 1766.42it/s]Processing text_right with encode:  11%|█         | 2098/18841 [00:01<00:09, 1700.56it/s]Processing text_right with encode:  12%|█▏        | 2277/18841 [00:01<00:09, 1722.85it/s]Processing text_right with encode:  13%|█▎        | 2464/18841 [00:01<00:09, 1763.19it/s]Processing text_right with encode:  14%|█▍        | 2643/18841 [00:01<00:09, 1771.11it/s]Processing text_right with encode:  15%|█▌        | 2853/18841 [00:01<00:08, 1853.51it/s]Processing text_right with encode:  16%|█▌        | 3040/18841 [00:01<00:09, 1739.63it/s]Processing text_right with encode:  17%|█▋        | 3229/18841 [00:01<00:08, 1780.94it/s]Processing text_right with encode:  18%|█▊        | 3409/18841 [00:01<00:08, 1778.42it/s]Processing text_right with encode:  19%|█▉        | 3588/18841 [00:02<00:08, 1765.27it/s]Processing text_right with encode:  20%|██        | 3771/18841 [00:02<00:08, 1782.10it/s]Processing text_right with encode:  21%|██        | 3953/18841 [00:02<00:08, 1789.49it/s]Processing text_right with encode:  22%|██▏       | 4133/18841 [00:02<00:08, 1755.16it/s]Processing text_right with encode:  23%|██▎       | 4312/18841 [00:02<00:08, 1762.96it/s]Processing text_right with encode:  24%|██▍       | 4495/18841 [00:02<00:08, 1781.27it/s]Processing text_right with encode:  25%|██▍       | 4674/18841 [00:02<00:07, 1775.74it/s]Processing text_right with encode:  26%|██▌       | 4853/18841 [00:02<00:07, 1777.52it/s]Processing text_right with encode:  27%|██▋       | 5031/18841 [00:02<00:07, 1777.12it/s]Processing text_right with encode:  28%|██▊       | 5220/18841 [00:02<00:07, 1808.41it/s]Processing text_right with encode:  29%|██▊       | 5409/18841 [00:03<00:07, 1831.05it/s]Processing text_right with encode:  30%|██▉       | 5593/18841 [00:03<00:07, 1761.22it/s]Processing text_right with encode:  31%|███       | 5770/18841 [00:03<00:07, 1706.27it/s]Processing text_right with encode:  32%|███▏      | 5942/18841 [00:03<00:07, 1666.54it/s]Processing text_right with encode:  32%|███▏      | 6110/18841 [00:03<00:07, 1645.64it/s]Processing text_right with encode:  33%|███▎      | 6276/18841 [00:03<00:07, 1596.79it/s]Processing text_right with encode:  34%|███▍      | 6450/18841 [00:03<00:07, 1629.44it/s]Processing text_right with encode:  35%|███▌      | 6614/18841 [00:03<00:07, 1613.12it/s]Processing text_right with encode:  36%|███▌      | 6776/18841 [00:03<00:07, 1602.04it/s]Processing text_right with encode:  37%|███▋      | 6937/18841 [00:04<00:07, 1582.96it/s]Processing text_right with encode:  38%|███▊      | 7098/18841 [00:04<00:07, 1589.48it/s]Processing text_right with encode:  39%|███▊      | 7272/18841 [00:04<00:07, 1626.73it/s]Processing text_right with encode:  40%|███▉      | 7455/18841 [00:04<00:06, 1680.26it/s]Processing text_right with encode:  40%|████      | 7624/18841 [00:04<00:06, 1653.74it/s]Processing text_right with encode:  41%|████▏     | 7790/18841 [00:04<00:06, 1647.10it/s]Processing text_right with encode:  42%|████▏     | 7956/18841 [00:04<00:06, 1643.78it/s]Processing text_right with encode:  43%|████▎     | 8131/18841 [00:04<00:06, 1668.69it/s]Processing text_right with encode:  44%|████▍     | 8316/18841 [00:04<00:06, 1718.58it/s]Processing text_right with encode:  45%|████▌     | 8500/18841 [00:04<00:05, 1749.97it/s]Processing text_right with encode:  46%|████▌     | 8676/18841 [00:05<00:05, 1741.19it/s]Processing text_right with encode:  47%|████▋     | 8868/18841 [00:05<00:05, 1789.93it/s]Processing text_right with encode:  48%|████▊     | 9048/18841 [00:05<00:05, 1730.96it/s]Processing text_right with encode:  49%|████▉     | 9238/18841 [00:05<00:05, 1776.69it/s]Processing text_right with encode:  50%|████▉     | 9417/18841 [00:05<00:05, 1725.41it/s]Processing text_right with encode:  51%|█████     | 9595/18841 [00:05<00:05, 1740.59it/s]Processing text_right with encode:  52%|█████▏    | 9770/18841 [00:05<00:05, 1685.22it/s]Processing text_right with encode:  53%|█████▎    | 9948/18841 [00:05<00:05, 1711.26it/s]Processing text_right with encode:  54%|█████▎    | 10121/18841 [00:05<00:05, 1715.28it/s]Processing text_right with encode:  55%|█████▍    | 10294/18841 [00:05<00:05, 1705.15it/s]Processing text_right with encode:  56%|█████▌    | 10492/18841 [00:06<00:04, 1771.14it/s]Processing text_right with encode:  57%|█████▋    | 10671/18841 [00:06<00:04, 1677.25it/s]Processing text_right with encode:  58%|█████▊    | 10841/18841 [00:06<00:04, 1657.24it/s]Processing text_right with encode:  58%|█████▊    | 11008/18841 [00:06<00:04, 1644.58it/s]Processing text_right with encode:  59%|█████▉    | 11174/18841 [00:06<00:04, 1621.81it/s]Processing text_right with encode:  60%|██████    | 11337/18841 [00:06<00:04, 1566.46it/s]Processing text_right with encode:  61%|██████    | 11499/18841 [00:06<00:04, 1579.72it/s]Processing text_right with encode:  62%|██████▏   | 11669/18841 [00:06<00:04, 1612.69it/s]Processing text_right with encode:  63%|██████▎   | 11857/18841 [00:06<00:04, 1684.14it/s]Processing text_right with encode:  64%|██████▍   | 12027/18841 [00:07<00:04, 1688.39it/s]Processing text_right with encode:  65%|██████▍   | 12207/18841 [00:07<00:03, 1719.38it/s]Processing text_right with encode:  66%|██████▌   | 12391/18841 [00:07<00:03, 1752.12it/s]Processing text_right with encode:  67%|██████▋   | 12567/18841 [00:07<00:03, 1690.22it/s]Processing text_right with encode:  68%|██████▊   | 12737/18841 [00:07<00:03, 1674.48it/s]Processing text_right with encode:  69%|██████▊   | 12916/18841 [00:07<00:03, 1706.79it/s]Processing text_right with encode:  70%|██████▉   | 13097/18841 [00:07<00:03, 1735.57it/s]Processing text_right with encode:  70%|███████   | 13272/18841 [00:07<00:03, 1725.27it/s]Processing text_right with encode:  71%|███████▏  | 13460/18841 [00:07<00:03, 1768.24it/s]Processing text_right with encode:  72%|███████▏  | 13650/18841 [00:07<00:02, 1804.76it/s]Processing text_right with encode:  73%|███████▎  | 13842/18841 [00:08<00:02, 1837.79it/s]Processing text_right with encode:  74%|███████▍  | 14036/18841 [00:08<00:02, 1865.46it/s]Processing text_right with encode:  75%|███████▌  | 14224/18841 [00:08<00:02, 1814.55it/s]Processing text_right with encode:  76%|███████▋  | 14407/18841 [00:08<00:02, 1763.99it/s]Processing text_right with encode:  77%|███████▋  | 14585/18841 [00:08<00:02, 1736.71it/s]Processing text_right with encode:  78%|███████▊  | 14770/18841 [00:08<00:02, 1768.63it/s]Processing text_right with encode:  79%|███████▉  | 14948/18841 [00:08<00:02, 1771.48it/s]Processing text_right with encode:  80%|████████  | 15126/18841 [00:08<00:02, 1742.70it/s]Processing text_right with encode:  81%|████████  | 15301/18841 [00:08<00:02, 1734.20it/s]Processing text_right with encode:  82%|████████▏ | 15475/18841 [00:08<00:01, 1703.61it/s]Processing text_right with encode:  83%|████████▎ | 15649/18841 [00:09<00:01, 1711.66it/s]Processing text_right with encode:  84%|████████▍ | 15831/18841 [00:09<00:01, 1742.47it/s]Processing text_right with encode:  85%|████████▍ | 16006/18841 [00:09<00:01, 1713.48it/s]Processing text_right with encode:  86%|████████▌ | 16178/18841 [00:09<00:01, 1694.37it/s]Processing text_right with encode:  87%|████████▋ | 16348/18841 [00:09<00:01, 1692.22it/s]Processing text_right with encode:  88%|████████▊ | 16519/18841 [00:09<00:01, 1696.64it/s]Processing text_right with encode:  89%|████████▊ | 16689/18841 [00:09<00:01, 1689.79it/s]Processing text_right with encode:  90%|████████▉ | 16871/18841 [00:09<00:01, 1721.77it/s]Processing text_right with encode:  91%|█████████ | 17053/18841 [00:09<00:01, 1749.18it/s]Processing text_right with encode:  91%|█████████▏| 17229/18841 [00:10<00:00, 1721.46it/s]Processing text_right with encode:  92%|█████████▏| 17409/18841 [00:10<00:00, 1741.99it/s]Processing text_right with encode:  93%|█████████▎| 17598/18841 [00:10<00:00, 1782.70it/s]Processing text_right with encode:  94%|█████████▍| 17777/18841 [00:10<00:00, 1760.84it/s]Processing text_right with encode:  95%|█████████▌| 17961/18841 [00:10<00:00, 1780.95it/s]Processing text_right with encode:  96%|█████████▋| 18160/18841 [00:10<00:00, 1838.37it/s]Processing text_right with encode:  97%|█████████▋| 18345/18841 [00:10<00:00, 1772.57it/s]Processing text_right with encode:  98%|█████████▊| 18536/18841 [00:10<00:00, 1811.51it/s]Processing text_right with encode:  99%|█████████▉| 18719/18841 [00:10<00:00, 1768.87it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:10<00:00, 1728.12it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 682928.65it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 896034.67it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  78%|███████▊  | 493/633 [00:00<00:00, 4918.00it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4900.56it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 188/5961 [00:00<00:03, 1876.35it/s]Processing text_right with encode:   6%|▋         | 382/5961 [00:00<00:02, 1894.96it/s]Processing text_right with encode:   9%|▉         | 556/5961 [00:00<00:02, 1844.19it/s]Processing text_right with encode:  12%|█▏        | 730/5961 [00:00<00:02, 1811.28it/s]Processing text_right with encode:  15%|█▌        | 921/5961 [00:00<00:02, 1838.59it/s]Processing text_right with encode:  19%|█▊        | 1109/5961 [00:00<00:02, 1849.81it/s]Processing text_right with encode:  22%|██▏       | 1308/5961 [00:00<00:02, 1886.34it/s]Processing text_right with encode:  25%|██▌       | 1496/5961 [00:00<00:02, 1882.05it/s]Processing text_right with encode:  28%|██▊       | 1674/5961 [00:00<00:02, 1821.48it/s]Processing text_right with encode:  31%|███       | 1853/5961 [00:01<00:02, 1807.30it/s]Processing text_right with encode:  34%|███▍      | 2041/5961 [00:01<00:02, 1828.30it/s]Processing text_right with encode:  38%|███▊      | 2244/5961 [00:01<00:01, 1883.83it/s]Processing text_right with encode:  41%|████      | 2431/5961 [00:01<00:01, 1785.96it/s]Processing text_right with encode:  44%|████▍     | 2613/5961 [00:01<00:01, 1795.87it/s]Processing text_right with encode:  47%|████▋     | 2808/5961 [00:01<00:01, 1837.40it/s]Processing text_right with encode:  50%|█████     | 2992/5961 [00:01<00:01, 1823.56it/s]Processing text_right with encode:  54%|█████▎    | 3191/5961 [00:01<00:01, 1869.57it/s]Processing text_right with encode:  57%|█████▋    | 3379/5961 [00:01<00:01, 1833.40it/s]Processing text_right with encode:  60%|█████▉    | 3563/5961 [00:01<00:01, 1797.24it/s]Processing text_right with encode:  63%|██████▎   | 3744/5961 [00:02<00:01, 1798.48it/s]Processing text_right with encode:  66%|██████▌   | 3936/5961 [00:02<00:01, 1831.98it/s]Processing text_right with encode:  69%|██████▉   | 4127/5961 [00:02<00:00, 1853.75it/s]Processing text_right with encode:  72%|███████▏  | 4318/5961 [00:02<00:00, 1867.68it/s]Processing text_right with encode:  76%|███████▌  | 4506/5961 [00:02<00:00, 1811.92it/s]Processing text_right with encode:  79%|███████▉  | 4699/5961 [00:02<00:00, 1845.06it/s]Processing text_right with encode:  82%|████████▏ | 4885/5961 [00:02<00:00, 1770.03it/s]Processing text_right with encode:  85%|████████▍ | 5064/5961 [00:02<00:00, 1774.12it/s]Processing text_right with encode:  88%|████████▊ | 5254/5961 [00:02<00:00, 1807.72it/s]Processing text_right with encode:  91%|█████████ | 5436/5961 [00:02<00:00, 1792.80it/s]Processing text_right with encode:  94%|█████████▍| 5616/5961 [00:03<00:00, 1740.35it/s]Processing text_right with encode:  97%|█████████▋| 5791/5961 [00:03<00:00, 1689.67it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1818.22it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 524184.49it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 830198.11it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:26<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:26<?, ?it/s, loss=1.046]Epoch 1/1:   1%|          | 1/102 [00:26<44:33, 26.47s/it, loss=1.046]Epoch 1/1:   1%|          | 1/102 [00:42<44:33, 26.47s/it, loss=1.046]Epoch 1/1:   1%|          | 1/102 [00:42<44:33, 26.47s/it, loss=0.918]Epoch 1/1:   2%|▏         | 2/102 [00:42<39:07, 23.47s/it, loss=0.918]Epoch 1/1:   2%|▏         | 2/102 [02:37<39:07, 23.47s/it, loss=0.918]Epoch 1/1:   2%|▏         | 2/102 [02:37<39:07, 23.47s/it, loss=0.672]Epoch 1/1:   3%|▎         | 3/102 [02:37<1:23:49, 50.81s/it, loss=0.672]Killed

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
   c5936c5..0e759a5  master     -> origin/master
Updating c5936c5..0e759a5
Fast-forward
 deps.txt                                           |   6 +-
 .../20200522/list_log_dataloader_20200522.md       |  14 +-
 error_list/20200522/list_log_testall_20200522.md   | 644 ++++++++++++++++-----
 ...-10_a463a24ea257f46bfcbd4006f805952aace8f2b1.py | 626 ++++++++++++++++++++
 4 files changed, 1134 insertions(+), 156 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-22-21-10_a463a24ea257f46bfcbd4006f805952aace8f2b1.py
