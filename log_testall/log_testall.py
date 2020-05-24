
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'dbbd1e3505a2b3043e7688c1260e13ddacd09d91', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/dbbd1e3505a2b3043e7688c1260e13ddacd09d91

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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 2fb84f7] ml_store  && git pull --all
 2 files changed, 61 insertions(+), 9983 deletions(-)
 rewrite log_testall/log_testall.py (99%)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + 14c9bd5...2fb84f7 master -> master (forced update)





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 33ea8fd] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   2fb84f7..33ea8fd  master -> master





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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-24 12:15:18.495193: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-24 12:15:18.501061: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095125000 Hz
2020-05-24 12:15:18.501392: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55de93ff4690 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-24 12:15:18.501412: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 163
Trainable params: 163
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2570 - binary_crossentropy: 0.8390 - val_loss: 0.2490 - val_binary_crossentropy: 0.7156

  #### metrics   #################################################### 
{'MSE': 0.25289784456137165}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
Total params: 163
Trainable params: 163
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
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
Total params: 418
Trainable params: 418
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2412 - binary_crossentropy: 0.6749500/500 [==============================] - 1s 2ms/sample - loss: 0.2568 - binary_crossentropy: 0.7072 - val_loss: 0.2575 - val_binary_crossentropy: 0.7087

  #### metrics   #################################################### 
{'MSE': 0.25688404976144524}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
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
Total params: 418
Trainable params: 418
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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2515 - binary_crossentropy: 0.7225 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934

  #### metrics   #################################################### 
{'MSE': 0.25059369702545226}

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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 473
Trainable params: 473
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2932 - binary_crossentropy: 0.7987500/500 [==============================] - 1s 3ms/sample - loss: 0.2917 - binary_crossentropy: 0.7974 - val_loss: 0.2845 - val_binary_crossentropy: 0.7724

  #### metrics   #################################################### 
{'MSE': 0.28124120060056373}

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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 473
Trainable params: 473
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2482 - binary_crossentropy: 0.6896500/500 [==============================] - 2s 3ms/sample - loss: 0.2480 - binary_crossentropy: 0.6892 - val_loss: 0.2490 - val_binary_crossentropy: 0.6911

  #### metrics   #################################################### 
{'MSE': 0.24881913126177937}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-24 12:16:38.179049: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:16:38.181086: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:16:38.186631: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-24 12:16:38.196791: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-24 12:16:38.198542: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 12:16:38.200133: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:16:38.201568: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2517 - val_binary_crossentropy: 0.6965
2020-05-24 12:16:39.476042: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:16:39.477732: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:16:39.481840: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-24 12:16:39.490423: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-24 12:16:39.491917: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 12:16:39.493260: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:16:39.494476: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2521210078049784}

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
2020-05-24 12:17:03.214558: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:03.215909: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:03.219359: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-24 12:17:03.225698: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-24 12:17:03.226722: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 12:17:03.227785: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:03.228665: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931
2020-05-24 12:17:04.833656: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:04.834959: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:04.837610: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-24 12:17:04.843002: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-24 12:17:04.843949: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 12:17:04.844811: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:04.845597: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24989898868453272}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-24 12:17:39.146869: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:39.152195: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:39.168034: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-24 12:17:39.195608: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-24 12:17:39.200318: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 12:17:39.204769: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:39.209085: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.5560 - binary_crossentropy: 1.3690 - val_loss: 0.2778 - val_binary_crossentropy: 0.7520
2020-05-24 12:17:41.512863: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:41.517607: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:41.529836: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-24 12:17:41.554804: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-24 12:17:41.559062: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 12:17:41.563027: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 12:17:41.566889: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.33363756071026235}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
Total params: 695
Trainable params: 695
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2667 - binary_crossentropy: 1.1178500/500 [==============================] - 4s 9ms/sample - loss: 0.2882 - binary_crossentropy: 1.4229 - val_loss: 0.2900 - val_binary_crossentropy: 1.6364

  #### metrics   #################################################### 
{'MSE': 0.28855322452301646}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
Total params: 695
Trainable params: 695
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 7, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         16          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_5[0][0]           
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
100/500 [=====>........................] - ETA: 6s - loss: 0.3027 - binary_crossentropy: 1.8421500/500 [==============================] - 5s 9ms/sample - loss: 0.3028 - binary_crossentropy: 1.7882 - val_loss: 0.3093 - val_binary_crossentropy: 1.9645

  #### metrics   #################################################### 
{'MSE': 0.29849914979844505}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 7, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         16          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_5[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 1,904
Trainable params: 1,904
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2751 - binary_crossentropy: 1.6516500/500 [==============================] - 5s 9ms/sample - loss: 0.2923 - binary_crossentropy: 1.8436 - val_loss: 0.3014 - val_binary_crossentropy: 1.9986

  #### metrics   #################################################### 
{'MSE': 0.29623729681307737}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 1,904
Trainable params: 1,904
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
regionsequence_sum (InputLayer) [(None, 9)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         3           regionsequence_max[0][0]         
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
Total params: 136
Trainable params: 136
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.3153 - binary_crossentropy: 1.6139500/500 [==============================] - 6s 12ms/sample - loss: 0.2938 - binary_crossentropy: 1.4608 - val_loss: 0.2824 - val_binary_crossentropy: 1.3555

  #### metrics   #################################################### 
{'MSE': 0.2876215871277569}

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
regionsequence_sum (InputLayer) [(None, 9)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         3           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         3           regionsequence_max[0][0]         
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
Total params: 136
Trainable params: 136
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,432
Trainable params: 1,432
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2548 - binary_crossentropy: 0.7032500/500 [==============================] - 6s 12ms/sample - loss: 0.2523 - binary_crossentropy: 0.6979 - val_loss: 0.2503 - val_binary_crossentropy: 0.6938

  #### metrics   #################################################### 
{'MSE': 0.25064267237927634}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,432
Trainable params: 1,432
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         32          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
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
Total params: 3,086
Trainable params: 3,006
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3090 - binary_crossentropy: 1.3478500/500 [==============================] - 7s 13ms/sample - loss: 0.2915 - binary_crossentropy: 1.3556 - val_loss: 0.2853 - val_binary_crossentropy: 1.4437

  #### metrics   #################################################### 
{'MSE': 0.2890733169630902}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         32          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
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
Total params: 3,086
Trainable params: 3,006
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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 5c417c1] ml_store  && git pull --all
 1 file changed, 4946 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 62ad065...5c417c1 master -> master (forced update)





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master bc30a76] ml_store  && git pull --all
 1 file changed, 49 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   5c417c1..bc30a76  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master faca93c] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   bc30a76..faca93c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//Autokeras.py", line 12, in <module>
    import autokeras as ak
ModuleNotFoundError: No module named 'autokeras'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 1a4087a] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   faca93c..1a4087a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py 
/home/runner/work/mlmodels/mlmodels/mlmodels/dataset

  #### Loading params   ############################################## 

  #### Loading daaset   ############################################# 
Using TensorFlow backend.
Loading data...
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 284, in <module>
    test(pars_choice="json", data_path= f"{root_path}/model_keras/charcnn_zhang.json")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 248, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//charcnn_zhang.py", line 139, in get_dataset
    train_data.load_data()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/raw/char_cnn/data_utils.py", line 41, in load_data
    with open(self.data_source, 'r', encoding='utf-8') as f:
FileNotFoundError: [Errno 2] No such file or directory: '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/text/ag_news_csv/train.csv'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 1d87f1b] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   1a4087a..1d87f1b  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master d732d65] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   1d87f1b..d732d65  master -> master





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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 5e464ac] ml_store  && git pull --all
 1 file changed, 43 insertions(+)
To github.com:arita37/mlmodels_store.git
   d732d65..5e464ac  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 2s
 1728512/17464789 [=>............................] - ETA: 0s
 8118272/17464789 [============>.................] - ETA: 0s
16400384/17464789 [===========================>..] - ETA: 0s
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
2020-05-24 12:27:37.945128: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-24 12:27:37.949980: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095125000 Hz
2020-05-24 12:27:37.950126: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a1b5acc150 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-24 12:27:37.950140: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9886 - accuracy: 0.4790
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7126 - accuracy: 0.4970 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6551 - accuracy: 0.5008
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6820 - accuracy: 0.4990
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6462 - accuracy: 0.5013
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6436 - accuracy: 0.5015
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6257 - accuracy: 0.5027
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6222 - accuracy: 0.5029
11000/25000 [============>.................] - ETA: 3s - loss: 7.6708 - accuracy: 0.4997
12000/25000 [=============>................] - ETA: 3s - loss: 7.6845 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6784 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6732 - accuracy: 0.4996
15000/25000 [=================>............] - ETA: 2s - loss: 7.6881 - accuracy: 0.4986
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6752 - accuracy: 0.4994
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6522 - accuracy: 0.5009
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6547 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6497 - accuracy: 0.5011
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6536 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6601 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6583 - accuracy: 0.5005
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 7s 291us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f46511191d0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f46546f7470> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9580 - accuracy: 0.4810
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8276 - accuracy: 0.4895 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8353 - accuracy: 0.4890
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8046 - accuracy: 0.4910
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7188 - accuracy: 0.4966
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7663 - accuracy: 0.4935
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7499 - accuracy: 0.4946
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6877 - accuracy: 0.4986
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6871 - accuracy: 0.4987
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6896 - accuracy: 0.4985
11000/25000 [============>.................] - ETA: 3s - loss: 7.7098 - accuracy: 0.4972
12000/25000 [=============>................] - ETA: 3s - loss: 7.6475 - accuracy: 0.5013
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6501 - accuracy: 0.5011
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6644 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.6728 - accuracy: 0.4996
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6887 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6829 - accuracy: 0.4989
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6726 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6789 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6805 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6736 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6553 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6596 - accuracy: 0.5005
25000/25000 [==============================] - 7s 293us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8506 - accuracy: 0.4880 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.8302 - accuracy: 0.4893
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7356 - accuracy: 0.4955
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7525 - accuracy: 0.4944
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7484 - accuracy: 0.4947
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7389 - accuracy: 0.4953
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7433 - accuracy: 0.4950
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7518 - accuracy: 0.4944
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7602 - accuracy: 0.4939
11000/25000 [============>.................] - ETA: 3s - loss: 7.7321 - accuracy: 0.4957
12000/25000 [=============>................] - ETA: 3s - loss: 7.7305 - accuracy: 0.4958
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7291 - accuracy: 0.4959
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7082 - accuracy: 0.4973
15000/25000 [=================>............] - ETA: 2s - loss: 7.7239 - accuracy: 0.4963
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7184 - accuracy: 0.4966
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7298 - accuracy: 0.4959
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7305 - accuracy: 0.4958
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7280 - accuracy: 0.4960
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7111 - accuracy: 0.4971
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6885 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6861 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6700 - accuracy: 0.4998
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6730 - accuracy: 0.4996
25000/25000 [==============================] - 7s 291us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master d35055a] ml_store  && git pull --all
 1 file changed, 316 insertions(+)
To github.com:arita37/mlmodels_store.git
 + a3441e7...d35055a master -> master (forced update)





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

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 6ms/step - loss: nan
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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master e8d9519] ml_store  && git pull --all
 1 file changed, 124 insertions(+)
To github.com:arita37/mlmodels_store.git
   d35055a..e8d9519  master -> master





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
 3063808/11490434 [======>.......................] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:52 - loss: 2.2920 - categorical_accuracy: 0.1562
   96/60000 [..............................] - ETA: 3:37 - loss: 2.2291 - categorical_accuracy: 0.2292
  160/60000 [..............................] - ETA: 2:46 - loss: 2.1578 - categorical_accuracy: 0.2937
  224/60000 [..............................] - ETA: 2:23 - loss: 2.0533 - categorical_accuracy: 0.3259
  288/60000 [..............................] - ETA: 2:11 - loss: 1.9981 - categorical_accuracy: 0.3368
  352/60000 [..............................] - ETA: 2:02 - loss: 1.9442 - categorical_accuracy: 0.3523
  416/60000 [..............................] - ETA: 1:57 - loss: 1.8841 - categorical_accuracy: 0.3726
  480/60000 [..............................] - ETA: 1:53 - loss: 1.8660 - categorical_accuracy: 0.3750
  544/60000 [..............................] - ETA: 1:50 - loss: 1.8218 - categorical_accuracy: 0.3842
  608/60000 [..............................] - ETA: 1:47 - loss: 1.7614 - categorical_accuracy: 0.4095
  672/60000 [..............................] - ETA: 1:45 - loss: 1.6917 - categorical_accuracy: 0.4286
  736/60000 [..............................] - ETA: 1:43 - loss: 1.6310 - categorical_accuracy: 0.4457
  800/60000 [..............................] - ETA: 1:42 - loss: 1.5956 - categorical_accuracy: 0.4575
  832/60000 [..............................] - ETA: 1:41 - loss: 1.5738 - categorical_accuracy: 0.4627
  896/60000 [..............................] - ETA: 1:40 - loss: 1.5338 - categorical_accuracy: 0.4777
  928/60000 [..............................] - ETA: 1:40 - loss: 1.5148 - categorical_accuracy: 0.4860
  992/60000 [..............................] - ETA: 1:39 - loss: 1.4819 - categorical_accuracy: 0.5020
 1056/60000 [..............................] - ETA: 1:38 - loss: 1.4523 - categorical_accuracy: 0.5170
 1120/60000 [..............................] - ETA: 1:38 - loss: 1.4149 - categorical_accuracy: 0.5286
 1184/60000 [..............................] - ETA: 1:37 - loss: 1.3870 - categorical_accuracy: 0.5414
 1248/60000 [..............................] - ETA: 1:37 - loss: 1.3470 - categorical_accuracy: 0.5553
 1280/60000 [..............................] - ETA: 1:37 - loss: 1.3287 - categorical_accuracy: 0.5641
 1344/60000 [..............................] - ETA: 1:36 - loss: 1.2969 - categorical_accuracy: 0.5729
 1408/60000 [..............................] - ETA: 1:36 - loss: 1.2744 - categorical_accuracy: 0.5824
 1440/60000 [..............................] - ETA: 1:36 - loss: 1.2625 - categorical_accuracy: 0.5868
 1504/60000 [..............................] - ETA: 1:35 - loss: 1.2344 - categorical_accuracy: 0.5951
 1536/60000 [..............................] - ETA: 1:35 - loss: 1.2250 - categorical_accuracy: 0.5983
 1600/60000 [..............................] - ETA: 1:35 - loss: 1.1974 - categorical_accuracy: 0.6069
 1664/60000 [..............................] - ETA: 1:34 - loss: 1.1743 - categorical_accuracy: 0.6148
 1728/60000 [..............................] - ETA: 1:34 - loss: 1.1553 - categorical_accuracy: 0.6215
 1792/60000 [..............................] - ETA: 1:33 - loss: 1.1334 - categorical_accuracy: 0.6289
 1856/60000 [..............................] - ETA: 1:33 - loss: 1.1175 - categorical_accuracy: 0.6342
 1920/60000 [..............................] - ETA: 1:33 - loss: 1.0995 - categorical_accuracy: 0.6406
 1984/60000 [..............................] - ETA: 1:32 - loss: 1.0846 - categorical_accuracy: 0.6457
 2048/60000 [>.............................] - ETA: 1:32 - loss: 1.0679 - categorical_accuracy: 0.6499
 2080/60000 [>.............................] - ETA: 1:32 - loss: 1.0578 - categorical_accuracy: 0.6529
 2144/60000 [>.............................] - ETA: 1:32 - loss: 1.0453 - categorical_accuracy: 0.6558
 2208/60000 [>.............................] - ETA: 1:32 - loss: 1.0323 - categorical_accuracy: 0.6590
 2272/60000 [>.............................] - ETA: 1:32 - loss: 1.0170 - categorical_accuracy: 0.6651
 2336/60000 [>.............................] - ETA: 1:31 - loss: 1.0002 - categorical_accuracy: 0.6708
 2400/60000 [>.............................] - ETA: 1:31 - loss: 0.9841 - categorical_accuracy: 0.6762
 2464/60000 [>.............................] - ETA: 1:31 - loss: 0.9735 - categorical_accuracy: 0.6794
 2496/60000 [>.............................] - ETA: 1:31 - loss: 0.9640 - categorical_accuracy: 0.6831
 2528/60000 [>.............................] - ETA: 1:31 - loss: 0.9554 - categorical_accuracy: 0.6855
 2560/60000 [>.............................] - ETA: 1:31 - loss: 0.9459 - categorical_accuracy: 0.6891
 2592/60000 [>.............................] - ETA: 1:31 - loss: 0.9431 - categorical_accuracy: 0.6898
 2656/60000 [>.............................] - ETA: 1:31 - loss: 0.9327 - categorical_accuracy: 0.6943
 2720/60000 [>.............................] - ETA: 1:30 - loss: 0.9193 - categorical_accuracy: 0.6996
 2784/60000 [>.............................] - ETA: 1:30 - loss: 0.9065 - categorical_accuracy: 0.7044
 2848/60000 [>.............................] - ETA: 1:30 - loss: 0.8986 - categorical_accuracy: 0.7075
 2912/60000 [>.............................] - ETA: 1:29 - loss: 0.8912 - categorical_accuracy: 0.7095
 2976/60000 [>.............................] - ETA: 1:29 - loss: 0.8801 - categorical_accuracy: 0.7140
 3040/60000 [>.............................] - ETA: 1:29 - loss: 0.8686 - categorical_accuracy: 0.7174
 3104/60000 [>.............................] - ETA: 1:29 - loss: 0.8585 - categorical_accuracy: 0.7204
 3168/60000 [>.............................] - ETA: 1:29 - loss: 0.8486 - categorical_accuracy: 0.7241
 3200/60000 [>.............................] - ETA: 1:29 - loss: 0.8447 - categorical_accuracy: 0.7250
 3264/60000 [>.............................] - ETA: 1:28 - loss: 0.8354 - categorical_accuracy: 0.7279
 3328/60000 [>.............................] - ETA: 1:28 - loss: 0.8252 - categorical_accuracy: 0.7314
 3392/60000 [>.............................] - ETA: 1:28 - loss: 0.8153 - categorical_accuracy: 0.7347
 3456/60000 [>.............................] - ETA: 1:28 - loss: 0.8097 - categorical_accuracy: 0.7367
 3520/60000 [>.............................] - ETA: 1:28 - loss: 0.7993 - categorical_accuracy: 0.7398
 3584/60000 [>.............................] - ETA: 1:27 - loss: 0.7933 - categorical_accuracy: 0.7422
 3648/60000 [>.............................] - ETA: 1:27 - loss: 0.7889 - categorical_accuracy: 0.7429
 3712/60000 [>.............................] - ETA: 1:27 - loss: 0.7849 - categorical_accuracy: 0.7449
 3776/60000 [>.............................] - ETA: 1:27 - loss: 0.7773 - categorical_accuracy: 0.7479
 3840/60000 [>.............................] - ETA: 1:26 - loss: 0.7728 - categorical_accuracy: 0.7487
 3904/60000 [>.............................] - ETA: 1:26 - loss: 0.7675 - categorical_accuracy: 0.7503
 3968/60000 [>.............................] - ETA: 1:26 - loss: 0.7614 - categorical_accuracy: 0.7528
 4032/60000 [=>............................] - ETA: 1:26 - loss: 0.7534 - categorical_accuracy: 0.7552
 4096/60000 [=>............................] - ETA: 1:26 - loss: 0.7451 - categorical_accuracy: 0.7585
 4160/60000 [=>............................] - ETA: 1:26 - loss: 0.7360 - categorical_accuracy: 0.7618
 4224/60000 [=>............................] - ETA: 1:26 - loss: 0.7315 - categorical_accuracy: 0.7630
 4256/60000 [=>............................] - ETA: 1:26 - loss: 0.7283 - categorical_accuracy: 0.7639
 4288/60000 [=>............................] - ETA: 1:26 - loss: 0.7245 - categorical_accuracy: 0.7649
 4352/60000 [=>............................] - ETA: 1:26 - loss: 0.7182 - categorical_accuracy: 0.7672
 4384/60000 [=>............................] - ETA: 1:26 - loss: 0.7147 - categorical_accuracy: 0.7682
 4416/60000 [=>............................] - ETA: 1:26 - loss: 0.7120 - categorical_accuracy: 0.7692
 4480/60000 [=>............................] - ETA: 1:25 - loss: 0.7070 - categorical_accuracy: 0.7710
 4544/60000 [=>............................] - ETA: 1:25 - loss: 0.7016 - categorical_accuracy: 0.7731
 4608/60000 [=>............................] - ETA: 1:25 - loss: 0.6950 - categorical_accuracy: 0.7752
 4672/60000 [=>............................] - ETA: 1:25 - loss: 0.6935 - categorical_accuracy: 0.7757
 4736/60000 [=>............................] - ETA: 1:25 - loss: 0.6885 - categorical_accuracy: 0.7768
 4800/60000 [=>............................] - ETA: 1:24 - loss: 0.6831 - categorical_accuracy: 0.7783
 4864/60000 [=>............................] - ETA: 1:24 - loss: 0.6774 - categorical_accuracy: 0.7804
 4928/60000 [=>............................] - ETA: 1:24 - loss: 0.6775 - categorical_accuracy: 0.7810
 4992/60000 [=>............................] - ETA: 1:24 - loss: 0.6727 - categorical_accuracy: 0.7825
 5056/60000 [=>............................] - ETA: 1:24 - loss: 0.6676 - categorical_accuracy: 0.7842
 5120/60000 [=>............................] - ETA: 1:24 - loss: 0.6630 - categorical_accuracy: 0.7855
 5152/60000 [=>............................] - ETA: 1:24 - loss: 0.6622 - categorical_accuracy: 0.7859
 5184/60000 [=>............................] - ETA: 1:24 - loss: 0.6609 - categorical_accuracy: 0.7861
 5248/60000 [=>............................] - ETA: 1:23 - loss: 0.6578 - categorical_accuracy: 0.7870
 5312/60000 [=>............................] - ETA: 1:23 - loss: 0.6524 - categorical_accuracy: 0.7890
 5376/60000 [=>............................] - ETA: 1:23 - loss: 0.6480 - categorical_accuracy: 0.7904
 5440/60000 [=>............................] - ETA: 1:23 - loss: 0.6432 - categorical_accuracy: 0.7919
 5504/60000 [=>............................] - ETA: 1:23 - loss: 0.6395 - categorical_accuracy: 0.7934
 5568/60000 [=>............................] - ETA: 1:23 - loss: 0.6362 - categorical_accuracy: 0.7944
 5600/60000 [=>............................] - ETA: 1:23 - loss: 0.6347 - categorical_accuracy: 0.7950
 5664/60000 [=>............................] - ETA: 1:23 - loss: 0.6305 - categorical_accuracy: 0.7963
 5728/60000 [=>............................] - ETA: 1:22 - loss: 0.6283 - categorical_accuracy: 0.7968
 5792/60000 [=>............................] - ETA: 1:22 - loss: 0.6234 - categorical_accuracy: 0.7983
 5856/60000 [=>............................] - ETA: 1:22 - loss: 0.6213 - categorical_accuracy: 0.7988
 5920/60000 [=>............................] - ETA: 1:22 - loss: 0.6175 - categorical_accuracy: 0.7997
 5984/60000 [=>............................] - ETA: 1:22 - loss: 0.6143 - categorical_accuracy: 0.8011
 6048/60000 [==>...........................] - ETA: 1:22 - loss: 0.6093 - categorical_accuracy: 0.8029
 6080/60000 [==>...........................] - ETA: 1:22 - loss: 0.6074 - categorical_accuracy: 0.8036
 6144/60000 [==>...........................] - ETA: 1:22 - loss: 0.6035 - categorical_accuracy: 0.8047
 6208/60000 [==>...........................] - ETA: 1:22 - loss: 0.5994 - categorical_accuracy: 0.8057
 6272/60000 [==>...........................] - ETA: 1:21 - loss: 0.5984 - categorical_accuracy: 0.8060
 6336/60000 [==>...........................] - ETA: 1:21 - loss: 0.5954 - categorical_accuracy: 0.8068
 6400/60000 [==>...........................] - ETA: 1:21 - loss: 0.5911 - categorical_accuracy: 0.8083
 6464/60000 [==>...........................] - ETA: 1:21 - loss: 0.5901 - categorical_accuracy: 0.8091
 6528/60000 [==>...........................] - ETA: 1:21 - loss: 0.5878 - categorical_accuracy: 0.8096
 6592/60000 [==>...........................] - ETA: 1:21 - loss: 0.5853 - categorical_accuracy: 0.8107
 6656/60000 [==>...........................] - ETA: 1:21 - loss: 0.5814 - categorical_accuracy: 0.8119
 6720/60000 [==>...........................] - ETA: 1:21 - loss: 0.5789 - categorical_accuracy: 0.8128
 6784/60000 [==>...........................] - ETA: 1:20 - loss: 0.5783 - categorical_accuracy: 0.8135
 6848/60000 [==>...........................] - ETA: 1:20 - loss: 0.5756 - categorical_accuracy: 0.8145
 6912/60000 [==>...........................] - ETA: 1:20 - loss: 0.5727 - categorical_accuracy: 0.8158
 6976/60000 [==>...........................] - ETA: 1:20 - loss: 0.5706 - categorical_accuracy: 0.8171
 7040/60000 [==>...........................] - ETA: 1:20 - loss: 0.5679 - categorical_accuracy: 0.8182
 7104/60000 [==>...........................] - ETA: 1:20 - loss: 0.5648 - categorical_accuracy: 0.8193
 7168/60000 [==>...........................] - ETA: 1:20 - loss: 0.5614 - categorical_accuracy: 0.8205
 7232/60000 [==>...........................] - ETA: 1:20 - loss: 0.5588 - categorical_accuracy: 0.8212
 7296/60000 [==>...........................] - ETA: 1:20 - loss: 0.5553 - categorical_accuracy: 0.8224
 7360/60000 [==>...........................] - ETA: 1:19 - loss: 0.5523 - categorical_accuracy: 0.8232
 7424/60000 [==>...........................] - ETA: 1:19 - loss: 0.5500 - categorical_accuracy: 0.8239
 7488/60000 [==>...........................] - ETA: 1:19 - loss: 0.5471 - categorical_accuracy: 0.8248
 7520/60000 [==>...........................] - ETA: 1:19 - loss: 0.5458 - categorical_accuracy: 0.8253
 7584/60000 [==>...........................] - ETA: 1:19 - loss: 0.5432 - categorical_accuracy: 0.8261
 7648/60000 [==>...........................] - ETA: 1:19 - loss: 0.5396 - categorical_accuracy: 0.8273
 7712/60000 [==>...........................] - ETA: 1:19 - loss: 0.5366 - categorical_accuracy: 0.8282
 7776/60000 [==>...........................] - ETA: 1:19 - loss: 0.5336 - categorical_accuracy: 0.8292
 7840/60000 [==>...........................] - ETA: 1:19 - loss: 0.5309 - categorical_accuracy: 0.8302
 7904/60000 [==>...........................] - ETA: 1:19 - loss: 0.5286 - categorical_accuracy: 0.8308
 7936/60000 [==>...........................] - ETA: 1:19 - loss: 0.5274 - categorical_accuracy: 0.8313
 8000/60000 [===>..........................] - ETA: 1:18 - loss: 0.5264 - categorical_accuracy: 0.8317
 8064/60000 [===>..........................] - ETA: 1:18 - loss: 0.5263 - categorical_accuracy: 0.8320
 8128/60000 [===>..........................] - ETA: 1:18 - loss: 0.5247 - categorical_accuracy: 0.8329
 8160/60000 [===>..........................] - ETA: 1:18 - loss: 0.5232 - categorical_accuracy: 0.8332
 8192/60000 [===>..........................] - ETA: 1:18 - loss: 0.5215 - categorical_accuracy: 0.8337
 8256/60000 [===>..........................] - ETA: 1:18 - loss: 0.5197 - categorical_accuracy: 0.8344
 8320/60000 [===>..........................] - ETA: 1:18 - loss: 0.5161 - categorical_accuracy: 0.8357
 8384/60000 [===>..........................] - ETA: 1:18 - loss: 0.5138 - categorical_accuracy: 0.8366
 8448/60000 [===>..........................] - ETA: 1:18 - loss: 0.5108 - categorical_accuracy: 0.8376
 8512/60000 [===>..........................] - ETA: 1:18 - loss: 0.5091 - categorical_accuracy: 0.8382
 8576/60000 [===>..........................] - ETA: 1:18 - loss: 0.5072 - categorical_accuracy: 0.8387
 8640/60000 [===>..........................] - ETA: 1:17 - loss: 0.5058 - categorical_accuracy: 0.8392
 8704/60000 [===>..........................] - ETA: 1:17 - loss: 0.5039 - categorical_accuracy: 0.8400
 8768/60000 [===>..........................] - ETA: 1:17 - loss: 0.5021 - categorical_accuracy: 0.8404
 8832/60000 [===>..........................] - ETA: 1:17 - loss: 0.4995 - categorical_accuracy: 0.8414
 8896/60000 [===>..........................] - ETA: 1:17 - loss: 0.4966 - categorical_accuracy: 0.8423
 8928/60000 [===>..........................] - ETA: 1:17 - loss: 0.4958 - categorical_accuracy: 0.8425
 8960/60000 [===>..........................] - ETA: 1:17 - loss: 0.4946 - categorical_accuracy: 0.8430
 8992/60000 [===>..........................] - ETA: 1:17 - loss: 0.4934 - categorical_accuracy: 0.8434
 9056/60000 [===>..........................] - ETA: 1:17 - loss: 0.4912 - categorical_accuracy: 0.8442
 9120/60000 [===>..........................] - ETA: 1:17 - loss: 0.4899 - categorical_accuracy: 0.8444
 9184/60000 [===>..........................] - ETA: 1:17 - loss: 0.4880 - categorical_accuracy: 0.8449
 9248/60000 [===>..........................] - ETA: 1:16 - loss: 0.4870 - categorical_accuracy: 0.8454
 9312/60000 [===>..........................] - ETA: 1:16 - loss: 0.4852 - categorical_accuracy: 0.8460
 9376/60000 [===>..........................] - ETA: 1:16 - loss: 0.4838 - categorical_accuracy: 0.8467
 9408/60000 [===>..........................] - ETA: 1:16 - loss: 0.4825 - categorical_accuracy: 0.8472
 9472/60000 [===>..........................] - ETA: 1:16 - loss: 0.4819 - categorical_accuracy: 0.8476
 9536/60000 [===>..........................] - ETA: 1:16 - loss: 0.4813 - categorical_accuracy: 0.8476
 9600/60000 [===>..........................] - ETA: 1:16 - loss: 0.4803 - categorical_accuracy: 0.8480
 9664/60000 [===>..........................] - ETA: 1:16 - loss: 0.4794 - categorical_accuracy: 0.8482
 9728/60000 [===>..........................] - ETA: 1:16 - loss: 0.4777 - categorical_accuracy: 0.8487
 9792/60000 [===>..........................] - ETA: 1:15 - loss: 0.4767 - categorical_accuracy: 0.8488
 9856/60000 [===>..........................] - ETA: 1:15 - loss: 0.4757 - categorical_accuracy: 0.8493
 9920/60000 [===>..........................] - ETA: 1:15 - loss: 0.4759 - categorical_accuracy: 0.8496
 9952/60000 [===>..........................] - ETA: 1:15 - loss: 0.4759 - categorical_accuracy: 0.8496
10016/60000 [====>.........................] - ETA: 1:15 - loss: 0.4739 - categorical_accuracy: 0.8503
10080/60000 [====>.........................] - ETA: 1:15 - loss: 0.4720 - categorical_accuracy: 0.8509
10144/60000 [====>.........................] - ETA: 1:15 - loss: 0.4697 - categorical_accuracy: 0.8517
10208/60000 [====>.........................] - ETA: 1:15 - loss: 0.4672 - categorical_accuracy: 0.8527
10272/60000 [====>.........................] - ETA: 1:15 - loss: 0.4648 - categorical_accuracy: 0.8535
10336/60000 [====>.........................] - ETA: 1:15 - loss: 0.4631 - categorical_accuracy: 0.8541
10400/60000 [====>.........................] - ETA: 1:14 - loss: 0.4615 - categorical_accuracy: 0.8545
10464/60000 [====>.........................] - ETA: 1:14 - loss: 0.4596 - categorical_accuracy: 0.8552
10528/60000 [====>.........................] - ETA: 1:14 - loss: 0.4586 - categorical_accuracy: 0.8558
10592/60000 [====>.........................] - ETA: 1:14 - loss: 0.4563 - categorical_accuracy: 0.8565
10624/60000 [====>.........................] - ETA: 1:14 - loss: 0.4553 - categorical_accuracy: 0.8568
10688/60000 [====>.........................] - ETA: 1:14 - loss: 0.4536 - categorical_accuracy: 0.8572
10752/60000 [====>.........................] - ETA: 1:14 - loss: 0.4514 - categorical_accuracy: 0.8579
10816/60000 [====>.........................] - ETA: 1:14 - loss: 0.4493 - categorical_accuracy: 0.8585
10880/60000 [====>.........................] - ETA: 1:14 - loss: 0.4476 - categorical_accuracy: 0.8591
10944/60000 [====>.........................] - ETA: 1:13 - loss: 0.4464 - categorical_accuracy: 0.8595
11008/60000 [====>.........................] - ETA: 1:13 - loss: 0.4444 - categorical_accuracy: 0.8600
11072/60000 [====>.........................] - ETA: 1:13 - loss: 0.4443 - categorical_accuracy: 0.8605
11136/60000 [====>.........................] - ETA: 1:13 - loss: 0.4427 - categorical_accuracy: 0.8610
11200/60000 [====>.........................] - ETA: 1:13 - loss: 0.4409 - categorical_accuracy: 0.8617
11232/60000 [====>.........................] - ETA: 1:13 - loss: 0.4401 - categorical_accuracy: 0.8618
11296/60000 [====>.........................] - ETA: 1:13 - loss: 0.4386 - categorical_accuracy: 0.8624
11360/60000 [====>.........................] - ETA: 1:13 - loss: 0.4368 - categorical_accuracy: 0.8629
11424/60000 [====>.........................] - ETA: 1:13 - loss: 0.4349 - categorical_accuracy: 0.8634
11488/60000 [====>.........................] - ETA: 1:13 - loss: 0.4330 - categorical_accuracy: 0.8639
11552/60000 [====>.........................] - ETA: 1:12 - loss: 0.4322 - categorical_accuracy: 0.8643
11616/60000 [====>.........................] - ETA: 1:12 - loss: 0.4315 - categorical_accuracy: 0.8647
11648/60000 [====>.........................] - ETA: 1:12 - loss: 0.4315 - categorical_accuracy: 0.8645
11712/60000 [====>.........................] - ETA: 1:12 - loss: 0.4303 - categorical_accuracy: 0.8651
11776/60000 [====>.........................] - ETA: 1:12 - loss: 0.4285 - categorical_accuracy: 0.8657
11840/60000 [====>.........................] - ETA: 1:12 - loss: 0.4274 - categorical_accuracy: 0.8660
11904/60000 [====>.........................] - ETA: 1:12 - loss: 0.4265 - categorical_accuracy: 0.8663
11968/60000 [====>.........................] - ETA: 1:12 - loss: 0.4250 - categorical_accuracy: 0.8669
12032/60000 [=====>........................] - ETA: 1:12 - loss: 0.4237 - categorical_accuracy: 0.8673
12096/60000 [=====>........................] - ETA: 1:12 - loss: 0.4223 - categorical_accuracy: 0.8677
12160/60000 [=====>........................] - ETA: 1:11 - loss: 0.4210 - categorical_accuracy: 0.8682
12224/60000 [=====>........................] - ETA: 1:11 - loss: 0.4201 - categorical_accuracy: 0.8686
12288/60000 [=====>........................] - ETA: 1:11 - loss: 0.4187 - categorical_accuracy: 0.8691
12352/60000 [=====>........................] - ETA: 1:11 - loss: 0.4178 - categorical_accuracy: 0.8695
12416/60000 [=====>........................] - ETA: 1:11 - loss: 0.4164 - categorical_accuracy: 0.8700
12480/60000 [=====>........................] - ETA: 1:11 - loss: 0.4153 - categorical_accuracy: 0.8704
12544/60000 [=====>........................] - ETA: 1:11 - loss: 0.4143 - categorical_accuracy: 0.8707
12608/60000 [=====>........................] - ETA: 1:11 - loss: 0.4131 - categorical_accuracy: 0.8710
12672/60000 [=====>........................] - ETA: 1:11 - loss: 0.4119 - categorical_accuracy: 0.8714
12736/60000 [=====>........................] - ETA: 1:10 - loss: 0.4110 - categorical_accuracy: 0.8717
12800/60000 [=====>........................] - ETA: 1:10 - loss: 0.4099 - categorical_accuracy: 0.8720
12864/60000 [=====>........................] - ETA: 1:10 - loss: 0.4091 - categorical_accuracy: 0.8723
12928/60000 [=====>........................] - ETA: 1:10 - loss: 0.4079 - categorical_accuracy: 0.8727
12992/60000 [=====>........................] - ETA: 1:10 - loss: 0.4065 - categorical_accuracy: 0.8731
13056/60000 [=====>........................] - ETA: 1:10 - loss: 0.4054 - categorical_accuracy: 0.8735
13120/60000 [=====>........................] - ETA: 1:10 - loss: 0.4045 - categorical_accuracy: 0.8737
13184/60000 [=====>........................] - ETA: 1:10 - loss: 0.4033 - categorical_accuracy: 0.8740
13248/60000 [=====>........................] - ETA: 1:10 - loss: 0.4032 - categorical_accuracy: 0.8742
13312/60000 [=====>........................] - ETA: 1:09 - loss: 0.4021 - categorical_accuracy: 0.8745
13376/60000 [=====>........................] - ETA: 1:09 - loss: 0.4009 - categorical_accuracy: 0.8750
13440/60000 [=====>........................] - ETA: 1:09 - loss: 0.4000 - categorical_accuracy: 0.8754
13504/60000 [=====>........................] - ETA: 1:09 - loss: 0.3986 - categorical_accuracy: 0.8759
13568/60000 [=====>........................] - ETA: 1:09 - loss: 0.3975 - categorical_accuracy: 0.8763
13600/60000 [=====>........................] - ETA: 1:09 - loss: 0.3969 - categorical_accuracy: 0.8764
13664/60000 [=====>........................] - ETA: 1:09 - loss: 0.3962 - categorical_accuracy: 0.8765
13728/60000 [=====>........................] - ETA: 1:09 - loss: 0.3956 - categorical_accuracy: 0.8768
13792/60000 [=====>........................] - ETA: 1:09 - loss: 0.3953 - categorical_accuracy: 0.8770
13856/60000 [=====>........................] - ETA: 1:09 - loss: 0.3946 - categorical_accuracy: 0.8774
13920/60000 [=====>........................] - ETA: 1:09 - loss: 0.3935 - categorical_accuracy: 0.8778
13984/60000 [=====>........................] - ETA: 1:08 - loss: 0.3926 - categorical_accuracy: 0.8779
14048/60000 [======>.......................] - ETA: 1:08 - loss: 0.3921 - categorical_accuracy: 0.8782
14112/60000 [======>.......................] - ETA: 1:08 - loss: 0.3909 - categorical_accuracy: 0.8786
14176/60000 [======>.......................] - ETA: 1:08 - loss: 0.3898 - categorical_accuracy: 0.8790
14240/60000 [======>.......................] - ETA: 1:08 - loss: 0.3902 - categorical_accuracy: 0.8789
14304/60000 [======>.......................] - ETA: 1:08 - loss: 0.3890 - categorical_accuracy: 0.8794
14368/60000 [======>.......................] - ETA: 1:08 - loss: 0.3882 - categorical_accuracy: 0.8797
14432/60000 [======>.......................] - ETA: 1:08 - loss: 0.3875 - categorical_accuracy: 0.8800
14496/60000 [======>.......................] - ETA: 1:08 - loss: 0.3868 - categorical_accuracy: 0.8802
14560/60000 [======>.......................] - ETA: 1:07 - loss: 0.3855 - categorical_accuracy: 0.8806
14624/60000 [======>.......................] - ETA: 1:07 - loss: 0.3844 - categorical_accuracy: 0.8809
14688/60000 [======>.......................] - ETA: 1:07 - loss: 0.3834 - categorical_accuracy: 0.8813
14752/60000 [======>.......................] - ETA: 1:07 - loss: 0.3829 - categorical_accuracy: 0.8815
14816/60000 [======>.......................] - ETA: 1:07 - loss: 0.3815 - categorical_accuracy: 0.8820
14880/60000 [======>.......................] - ETA: 1:07 - loss: 0.3805 - categorical_accuracy: 0.8823
14944/60000 [======>.......................] - ETA: 1:07 - loss: 0.3797 - categorical_accuracy: 0.8826
15008/60000 [======>.......................] - ETA: 1:07 - loss: 0.3783 - categorical_accuracy: 0.8831
15072/60000 [======>.......................] - ETA: 1:07 - loss: 0.3778 - categorical_accuracy: 0.8834
15136/60000 [======>.......................] - ETA: 1:07 - loss: 0.3768 - categorical_accuracy: 0.8837
15200/60000 [======>.......................] - ETA: 1:06 - loss: 0.3761 - categorical_accuracy: 0.8838
15264/60000 [======>.......................] - ETA: 1:06 - loss: 0.3753 - categorical_accuracy: 0.8839
15328/60000 [======>.......................] - ETA: 1:06 - loss: 0.3743 - categorical_accuracy: 0.8843
15392/60000 [======>.......................] - ETA: 1:06 - loss: 0.3737 - categorical_accuracy: 0.8845
15424/60000 [======>.......................] - ETA: 1:06 - loss: 0.3731 - categorical_accuracy: 0.8847
15488/60000 [======>.......................] - ETA: 1:06 - loss: 0.3719 - categorical_accuracy: 0.8849
15552/60000 [======>.......................] - ETA: 1:06 - loss: 0.3711 - categorical_accuracy: 0.8852
15616/60000 [======>.......................] - ETA: 1:06 - loss: 0.3706 - categorical_accuracy: 0.8854
15648/60000 [======>.......................] - ETA: 1:06 - loss: 0.3702 - categorical_accuracy: 0.8855
15712/60000 [======>.......................] - ETA: 1:06 - loss: 0.3697 - categorical_accuracy: 0.8856
15776/60000 [======>.......................] - ETA: 1:06 - loss: 0.3685 - categorical_accuracy: 0.8860
15808/60000 [======>.......................] - ETA: 1:06 - loss: 0.3679 - categorical_accuracy: 0.8861
15872/60000 [======>.......................] - ETA: 1:05 - loss: 0.3671 - categorical_accuracy: 0.8863
15936/60000 [======>.......................] - ETA: 1:05 - loss: 0.3665 - categorical_accuracy: 0.8865
15968/60000 [======>.......................] - ETA: 1:05 - loss: 0.3660 - categorical_accuracy: 0.8867
16032/60000 [=======>......................] - ETA: 1:05 - loss: 0.3647 - categorical_accuracy: 0.8871
16096/60000 [=======>......................] - ETA: 1:05 - loss: 0.3641 - categorical_accuracy: 0.8872
16160/60000 [=======>......................] - ETA: 1:05 - loss: 0.3636 - categorical_accuracy: 0.8874
16224/60000 [=======>......................] - ETA: 1:05 - loss: 0.3629 - categorical_accuracy: 0.8876
16256/60000 [=======>......................] - ETA: 1:05 - loss: 0.3629 - categorical_accuracy: 0.8877
16320/60000 [=======>......................] - ETA: 1:05 - loss: 0.3622 - categorical_accuracy: 0.8879
16384/60000 [=======>......................] - ETA: 1:05 - loss: 0.3615 - categorical_accuracy: 0.8881
16448/60000 [=======>......................] - ETA: 1:05 - loss: 0.3621 - categorical_accuracy: 0.8879
16512/60000 [=======>......................] - ETA: 1:04 - loss: 0.3611 - categorical_accuracy: 0.8883
16576/60000 [=======>......................] - ETA: 1:04 - loss: 0.3604 - categorical_accuracy: 0.8886
16640/60000 [=======>......................] - ETA: 1:04 - loss: 0.3596 - categorical_accuracy: 0.8886
16704/60000 [=======>......................] - ETA: 1:04 - loss: 0.3585 - categorical_accuracy: 0.8890
16768/60000 [=======>......................] - ETA: 1:04 - loss: 0.3577 - categorical_accuracy: 0.8893
16832/60000 [=======>......................] - ETA: 1:04 - loss: 0.3571 - categorical_accuracy: 0.8895
16896/60000 [=======>......................] - ETA: 1:04 - loss: 0.3561 - categorical_accuracy: 0.8899
16960/60000 [=======>......................] - ETA: 1:04 - loss: 0.3558 - categorical_accuracy: 0.8901
16992/60000 [=======>......................] - ETA: 1:04 - loss: 0.3560 - categorical_accuracy: 0.8901
17056/60000 [=======>......................] - ETA: 1:04 - loss: 0.3552 - categorical_accuracy: 0.8903
17120/60000 [=======>......................] - ETA: 1:04 - loss: 0.3541 - categorical_accuracy: 0.8907
17152/60000 [=======>......................] - ETA: 1:03 - loss: 0.3539 - categorical_accuracy: 0.8907
17216/60000 [=======>......................] - ETA: 1:03 - loss: 0.3531 - categorical_accuracy: 0.8910
17280/60000 [=======>......................] - ETA: 1:03 - loss: 0.3520 - categorical_accuracy: 0.8913
17344/60000 [=======>......................] - ETA: 1:03 - loss: 0.3514 - categorical_accuracy: 0.8915
17408/60000 [=======>......................] - ETA: 1:03 - loss: 0.3510 - categorical_accuracy: 0.8917
17472/60000 [=======>......................] - ETA: 1:03 - loss: 0.3503 - categorical_accuracy: 0.8919
17536/60000 [=======>......................] - ETA: 1:03 - loss: 0.3494 - categorical_accuracy: 0.8922
17600/60000 [=======>......................] - ETA: 1:03 - loss: 0.3486 - categorical_accuracy: 0.8924
17664/60000 [=======>......................] - ETA: 1:03 - loss: 0.3483 - categorical_accuracy: 0.8925
17696/60000 [=======>......................] - ETA: 1:03 - loss: 0.3480 - categorical_accuracy: 0.8926
17760/60000 [=======>......................] - ETA: 1:02 - loss: 0.3474 - categorical_accuracy: 0.8927
17792/60000 [=======>......................] - ETA: 1:02 - loss: 0.3474 - categorical_accuracy: 0.8926
17856/60000 [=======>......................] - ETA: 1:02 - loss: 0.3468 - categorical_accuracy: 0.8928
17920/60000 [=======>......................] - ETA: 1:02 - loss: 0.3461 - categorical_accuracy: 0.8930
17984/60000 [=======>......................] - ETA: 1:02 - loss: 0.3452 - categorical_accuracy: 0.8933
18048/60000 [========>.....................] - ETA: 1:02 - loss: 0.3445 - categorical_accuracy: 0.8935
18112/60000 [========>.....................] - ETA: 1:02 - loss: 0.3437 - categorical_accuracy: 0.8937
18176/60000 [========>.....................] - ETA: 1:02 - loss: 0.3432 - categorical_accuracy: 0.8940
18240/60000 [========>.....................] - ETA: 1:02 - loss: 0.3422 - categorical_accuracy: 0.8942
18304/60000 [========>.....................] - ETA: 1:02 - loss: 0.3417 - categorical_accuracy: 0.8944
18368/60000 [========>.....................] - ETA: 1:02 - loss: 0.3409 - categorical_accuracy: 0.8947
18432/60000 [========>.....................] - ETA: 1:01 - loss: 0.3401 - categorical_accuracy: 0.8949
18496/60000 [========>.....................] - ETA: 1:01 - loss: 0.3393 - categorical_accuracy: 0.8952
18528/60000 [========>.....................] - ETA: 1:01 - loss: 0.3389 - categorical_accuracy: 0.8953
18592/60000 [========>.....................] - ETA: 1:01 - loss: 0.3382 - categorical_accuracy: 0.8955
18656/60000 [========>.....................] - ETA: 1:01 - loss: 0.3387 - categorical_accuracy: 0.8955
18720/60000 [========>.....................] - ETA: 1:01 - loss: 0.3380 - categorical_accuracy: 0.8958
18784/60000 [========>.....................] - ETA: 1:01 - loss: 0.3373 - categorical_accuracy: 0.8960
18816/60000 [========>.....................] - ETA: 1:01 - loss: 0.3370 - categorical_accuracy: 0.8962
18880/60000 [========>.....................] - ETA: 1:01 - loss: 0.3362 - categorical_accuracy: 0.8965
18944/60000 [========>.....................] - ETA: 1:01 - loss: 0.3354 - categorical_accuracy: 0.8967
19008/60000 [========>.....................] - ETA: 1:01 - loss: 0.3350 - categorical_accuracy: 0.8968
19072/60000 [========>.....................] - ETA: 1:01 - loss: 0.3342 - categorical_accuracy: 0.8971
19104/60000 [========>.....................] - ETA: 1:01 - loss: 0.3337 - categorical_accuracy: 0.8972
19136/60000 [========>.....................] - ETA: 1:00 - loss: 0.3335 - categorical_accuracy: 0.8973
19200/60000 [========>.....................] - ETA: 1:00 - loss: 0.3330 - categorical_accuracy: 0.8976
19264/60000 [========>.....................] - ETA: 1:00 - loss: 0.3328 - categorical_accuracy: 0.8976
19328/60000 [========>.....................] - ETA: 1:00 - loss: 0.3320 - categorical_accuracy: 0.8979
19392/60000 [========>.....................] - ETA: 1:00 - loss: 0.3314 - categorical_accuracy: 0.8981
19456/60000 [========>.....................] - ETA: 1:00 - loss: 0.3306 - categorical_accuracy: 0.8983
19520/60000 [========>.....................] - ETA: 1:00 - loss: 0.3301 - categorical_accuracy: 0.8985
19584/60000 [========>.....................] - ETA: 1:00 - loss: 0.3293 - categorical_accuracy: 0.8987
19648/60000 [========>.....................] - ETA: 1:00 - loss: 0.3288 - categorical_accuracy: 0.8989
19712/60000 [========>.....................] - ETA: 1:00 - loss: 0.3281 - categorical_accuracy: 0.8990
19776/60000 [========>.....................] - ETA: 59s - loss: 0.3273 - categorical_accuracy: 0.8992 
19840/60000 [========>.....................] - ETA: 59s - loss: 0.3270 - categorical_accuracy: 0.8993
19872/60000 [========>.....................] - ETA: 59s - loss: 0.3267 - categorical_accuracy: 0.8994
19904/60000 [========>.....................] - ETA: 59s - loss: 0.3264 - categorical_accuracy: 0.8995
19968/60000 [========>.....................] - ETA: 59s - loss: 0.3260 - categorical_accuracy: 0.8996
20032/60000 [=========>....................] - ETA: 59s - loss: 0.3253 - categorical_accuracy: 0.8998
20096/60000 [=========>....................] - ETA: 59s - loss: 0.3245 - categorical_accuracy: 0.9000
20128/60000 [=========>....................] - ETA: 59s - loss: 0.3242 - categorical_accuracy: 0.9001
20192/60000 [=========>....................] - ETA: 59s - loss: 0.3234 - categorical_accuracy: 0.9004
20256/60000 [=========>....................] - ETA: 59s - loss: 0.3228 - categorical_accuracy: 0.9005
20320/60000 [=========>....................] - ETA: 59s - loss: 0.3222 - categorical_accuracy: 0.9007
20384/60000 [=========>....................] - ETA: 59s - loss: 0.3216 - categorical_accuracy: 0.9008
20448/60000 [=========>....................] - ETA: 58s - loss: 0.3208 - categorical_accuracy: 0.9010
20512/60000 [=========>....................] - ETA: 58s - loss: 0.3205 - categorical_accuracy: 0.9011
20576/60000 [=========>....................] - ETA: 58s - loss: 0.3201 - categorical_accuracy: 0.9013
20640/60000 [=========>....................] - ETA: 58s - loss: 0.3197 - categorical_accuracy: 0.9015
20704/60000 [=========>....................] - ETA: 58s - loss: 0.3189 - categorical_accuracy: 0.9017
20768/60000 [=========>....................] - ETA: 58s - loss: 0.3183 - categorical_accuracy: 0.9019
20832/60000 [=========>....................] - ETA: 58s - loss: 0.3178 - categorical_accuracy: 0.9020
20896/60000 [=========>....................] - ETA: 58s - loss: 0.3170 - categorical_accuracy: 0.9022
20960/60000 [=========>....................] - ETA: 58s - loss: 0.3162 - categorical_accuracy: 0.9025
21024/60000 [=========>....................] - ETA: 58s - loss: 0.3156 - categorical_accuracy: 0.9027
21088/60000 [=========>....................] - ETA: 57s - loss: 0.3151 - categorical_accuracy: 0.9029
21152/60000 [=========>....................] - ETA: 57s - loss: 0.3153 - categorical_accuracy: 0.9028
21216/60000 [=========>....................] - ETA: 57s - loss: 0.3147 - categorical_accuracy: 0.9029
21280/60000 [=========>....................] - ETA: 57s - loss: 0.3143 - categorical_accuracy: 0.9031
21344/60000 [=========>....................] - ETA: 57s - loss: 0.3142 - categorical_accuracy: 0.9033
21376/60000 [=========>....................] - ETA: 57s - loss: 0.3139 - categorical_accuracy: 0.9033
21440/60000 [=========>....................] - ETA: 57s - loss: 0.3132 - categorical_accuracy: 0.9035
21504/60000 [=========>....................] - ETA: 57s - loss: 0.3128 - categorical_accuracy: 0.9036
21568/60000 [=========>....................] - ETA: 57s - loss: 0.3120 - categorical_accuracy: 0.9038
21632/60000 [=========>....................] - ETA: 57s - loss: 0.3123 - categorical_accuracy: 0.9040
21664/60000 [=========>....................] - ETA: 57s - loss: 0.3120 - categorical_accuracy: 0.9041
21728/60000 [=========>....................] - ETA: 57s - loss: 0.3122 - categorical_accuracy: 0.9040
21792/60000 [=========>....................] - ETA: 56s - loss: 0.3116 - categorical_accuracy: 0.9042
21856/60000 [=========>....................] - ETA: 56s - loss: 0.3118 - categorical_accuracy: 0.9042
21920/60000 [=========>....................] - ETA: 56s - loss: 0.3111 - categorical_accuracy: 0.9044
21984/60000 [=========>....................] - ETA: 56s - loss: 0.3109 - categorical_accuracy: 0.9045
22048/60000 [==========>...................] - ETA: 56s - loss: 0.3103 - categorical_accuracy: 0.9047
22112/60000 [==========>...................] - ETA: 56s - loss: 0.3097 - categorical_accuracy: 0.9049
22176/60000 [==========>...................] - ETA: 56s - loss: 0.3095 - categorical_accuracy: 0.9050
22240/60000 [==========>...................] - ETA: 56s - loss: 0.3091 - categorical_accuracy: 0.9051
22304/60000 [==========>...................] - ETA: 56s - loss: 0.3083 - categorical_accuracy: 0.9054
22368/60000 [==========>...................] - ETA: 56s - loss: 0.3082 - categorical_accuracy: 0.9053
22432/60000 [==========>...................] - ETA: 55s - loss: 0.3079 - categorical_accuracy: 0.9055
22464/60000 [==========>...................] - ETA: 55s - loss: 0.3077 - categorical_accuracy: 0.9055
22528/60000 [==========>...................] - ETA: 55s - loss: 0.3074 - categorical_accuracy: 0.9057
22592/60000 [==========>...................] - ETA: 55s - loss: 0.3067 - categorical_accuracy: 0.9059
22656/60000 [==========>...................] - ETA: 55s - loss: 0.3064 - categorical_accuracy: 0.9060
22720/60000 [==========>...................] - ETA: 55s - loss: 0.3057 - categorical_accuracy: 0.9062
22784/60000 [==========>...................] - ETA: 55s - loss: 0.3050 - categorical_accuracy: 0.9065
22848/60000 [==========>...................] - ETA: 55s - loss: 0.3045 - categorical_accuracy: 0.9066
22912/60000 [==========>...................] - ETA: 55s - loss: 0.3038 - categorical_accuracy: 0.9068
22976/60000 [==========>...................] - ETA: 55s - loss: 0.3033 - categorical_accuracy: 0.9070
23040/60000 [==========>...................] - ETA: 54s - loss: 0.3030 - categorical_accuracy: 0.9072
23104/60000 [==========>...................] - ETA: 54s - loss: 0.3025 - categorical_accuracy: 0.9073
23168/60000 [==========>...................] - ETA: 54s - loss: 0.3019 - categorical_accuracy: 0.9075
23232/60000 [==========>...................] - ETA: 54s - loss: 0.3013 - categorical_accuracy: 0.9076
23296/60000 [==========>...................] - ETA: 54s - loss: 0.3010 - categorical_accuracy: 0.9078
23360/60000 [==========>...................] - ETA: 54s - loss: 0.3007 - categorical_accuracy: 0.9078
23424/60000 [==========>...................] - ETA: 54s - loss: 0.3000 - categorical_accuracy: 0.9080
23488/60000 [==========>...................] - ETA: 54s - loss: 0.2999 - categorical_accuracy: 0.9082
23520/60000 [==========>...................] - ETA: 54s - loss: 0.2997 - categorical_accuracy: 0.9082
23584/60000 [==========>...................] - ETA: 54s - loss: 0.2991 - categorical_accuracy: 0.9085
23648/60000 [==========>...................] - ETA: 54s - loss: 0.2986 - categorical_accuracy: 0.9085
23712/60000 [==========>...................] - ETA: 53s - loss: 0.2980 - categorical_accuracy: 0.9087
23776/60000 [==========>...................] - ETA: 53s - loss: 0.2974 - categorical_accuracy: 0.9089
23840/60000 [==========>...................] - ETA: 53s - loss: 0.2967 - categorical_accuracy: 0.9091
23904/60000 [==========>...................] - ETA: 53s - loss: 0.2963 - categorical_accuracy: 0.9093
23968/60000 [==========>...................] - ETA: 53s - loss: 0.2959 - categorical_accuracy: 0.9094
24032/60000 [===========>..................] - ETA: 53s - loss: 0.2954 - categorical_accuracy: 0.9095
24064/60000 [===========>..................] - ETA: 53s - loss: 0.2951 - categorical_accuracy: 0.9096
24128/60000 [===========>..................] - ETA: 53s - loss: 0.2945 - categorical_accuracy: 0.9098
24192/60000 [===========>..................] - ETA: 53s - loss: 0.2943 - categorical_accuracy: 0.9098
24256/60000 [===========>..................] - ETA: 53s - loss: 0.2937 - categorical_accuracy: 0.9100
24320/60000 [===========>..................] - ETA: 53s - loss: 0.2930 - categorical_accuracy: 0.9102
24384/60000 [===========>..................] - ETA: 52s - loss: 0.2924 - categorical_accuracy: 0.9104
24448/60000 [===========>..................] - ETA: 52s - loss: 0.2926 - categorical_accuracy: 0.9104
24512/60000 [===========>..................] - ETA: 52s - loss: 0.2920 - categorical_accuracy: 0.9106
24576/60000 [===========>..................] - ETA: 52s - loss: 0.2917 - categorical_accuracy: 0.9107
24640/60000 [===========>..................] - ETA: 52s - loss: 0.2911 - categorical_accuracy: 0.9109
24672/60000 [===========>..................] - ETA: 52s - loss: 0.2908 - categorical_accuracy: 0.9110
24736/60000 [===========>..................] - ETA: 52s - loss: 0.2905 - categorical_accuracy: 0.9111
24800/60000 [===========>..................] - ETA: 52s - loss: 0.2899 - categorical_accuracy: 0.9112
24864/60000 [===========>..................] - ETA: 52s - loss: 0.2896 - categorical_accuracy: 0.9113
24928/60000 [===========>..................] - ETA: 52s - loss: 0.2893 - categorical_accuracy: 0.9113
24992/60000 [===========>..................] - ETA: 52s - loss: 0.2889 - categorical_accuracy: 0.9115
25056/60000 [===========>..................] - ETA: 51s - loss: 0.2886 - categorical_accuracy: 0.9116
25120/60000 [===========>..................] - ETA: 51s - loss: 0.2881 - categorical_accuracy: 0.9118
25184/60000 [===========>..................] - ETA: 51s - loss: 0.2877 - categorical_accuracy: 0.9118
25248/60000 [===========>..................] - ETA: 51s - loss: 0.2873 - categorical_accuracy: 0.9120
25280/60000 [===========>..................] - ETA: 51s - loss: 0.2870 - categorical_accuracy: 0.9121
25312/60000 [===========>..................] - ETA: 51s - loss: 0.2868 - categorical_accuracy: 0.9122
25376/60000 [===========>..................] - ETA: 51s - loss: 0.2863 - categorical_accuracy: 0.9123
25440/60000 [===========>..................] - ETA: 51s - loss: 0.2866 - categorical_accuracy: 0.9123
25504/60000 [===========>..................] - ETA: 51s - loss: 0.2861 - categorical_accuracy: 0.9124
25568/60000 [===========>..................] - ETA: 51s - loss: 0.2859 - categorical_accuracy: 0.9126
25600/60000 [===========>..................] - ETA: 51s - loss: 0.2858 - categorical_accuracy: 0.9126
25664/60000 [===========>..................] - ETA: 51s - loss: 0.2854 - categorical_accuracy: 0.9127
25728/60000 [===========>..................] - ETA: 50s - loss: 0.2851 - categorical_accuracy: 0.9129
25792/60000 [===========>..................] - ETA: 50s - loss: 0.2845 - categorical_accuracy: 0.9130
25856/60000 [===========>..................] - ETA: 50s - loss: 0.2843 - categorical_accuracy: 0.9132
25920/60000 [===========>..................] - ETA: 50s - loss: 0.2838 - categorical_accuracy: 0.9133
25952/60000 [===========>..................] - ETA: 50s - loss: 0.2835 - categorical_accuracy: 0.9134
26016/60000 [============>.................] - ETA: 50s - loss: 0.2832 - categorical_accuracy: 0.9136
26080/60000 [============>.................] - ETA: 50s - loss: 0.2829 - categorical_accuracy: 0.9136
26112/60000 [============>.................] - ETA: 50s - loss: 0.2825 - categorical_accuracy: 0.9137
26176/60000 [============>.................] - ETA: 50s - loss: 0.2823 - categorical_accuracy: 0.9138
26240/60000 [============>.................] - ETA: 50s - loss: 0.2817 - categorical_accuracy: 0.9140
26304/60000 [============>.................] - ETA: 50s - loss: 0.2813 - categorical_accuracy: 0.9142
26368/60000 [============>.................] - ETA: 49s - loss: 0.2807 - categorical_accuracy: 0.9144
26432/60000 [============>.................] - ETA: 49s - loss: 0.2806 - categorical_accuracy: 0.9144
26496/60000 [============>.................] - ETA: 49s - loss: 0.2806 - categorical_accuracy: 0.9144
26560/60000 [============>.................] - ETA: 49s - loss: 0.2805 - categorical_accuracy: 0.9143
26624/60000 [============>.................] - ETA: 49s - loss: 0.2803 - categorical_accuracy: 0.9145
26688/60000 [============>.................] - ETA: 49s - loss: 0.2802 - categorical_accuracy: 0.9146
26752/60000 [============>.................] - ETA: 49s - loss: 0.2803 - categorical_accuracy: 0.9146
26816/60000 [============>.................] - ETA: 49s - loss: 0.2797 - categorical_accuracy: 0.9148
26880/60000 [============>.................] - ETA: 49s - loss: 0.2795 - categorical_accuracy: 0.9149
26912/60000 [============>.................] - ETA: 49s - loss: 0.2794 - categorical_accuracy: 0.9149
26976/60000 [============>.................] - ETA: 49s - loss: 0.2794 - categorical_accuracy: 0.9149
27040/60000 [============>.................] - ETA: 48s - loss: 0.2792 - categorical_accuracy: 0.9149
27104/60000 [============>.................] - ETA: 48s - loss: 0.2791 - categorical_accuracy: 0.9149
27168/60000 [============>.................] - ETA: 48s - loss: 0.2787 - categorical_accuracy: 0.9150
27232/60000 [============>.................] - ETA: 48s - loss: 0.2782 - categorical_accuracy: 0.9152
27296/60000 [============>.................] - ETA: 48s - loss: 0.2779 - categorical_accuracy: 0.9153
27328/60000 [============>.................] - ETA: 48s - loss: 0.2778 - categorical_accuracy: 0.9153
27392/60000 [============>.................] - ETA: 48s - loss: 0.2774 - categorical_accuracy: 0.9154
27456/60000 [============>.................] - ETA: 48s - loss: 0.2772 - categorical_accuracy: 0.9155
27520/60000 [============>.................] - ETA: 48s - loss: 0.2766 - categorical_accuracy: 0.9157
27584/60000 [============>.................] - ETA: 48s - loss: 0.2761 - categorical_accuracy: 0.9159
27616/60000 [============>.................] - ETA: 48s - loss: 0.2761 - categorical_accuracy: 0.9159
27680/60000 [============>.................] - ETA: 48s - loss: 0.2761 - categorical_accuracy: 0.9158
27744/60000 [============>.................] - ETA: 47s - loss: 0.2755 - categorical_accuracy: 0.9160
27808/60000 [============>.................] - ETA: 47s - loss: 0.2755 - categorical_accuracy: 0.9160
27872/60000 [============>.................] - ETA: 47s - loss: 0.2753 - categorical_accuracy: 0.9161
27936/60000 [============>.................] - ETA: 47s - loss: 0.2749 - categorical_accuracy: 0.9162
28000/60000 [=============>................] - ETA: 47s - loss: 0.2747 - categorical_accuracy: 0.9163
28064/60000 [=============>................] - ETA: 47s - loss: 0.2742 - categorical_accuracy: 0.9164
28128/60000 [=============>................] - ETA: 47s - loss: 0.2737 - categorical_accuracy: 0.9166
28192/60000 [=============>................] - ETA: 47s - loss: 0.2733 - categorical_accuracy: 0.9167
28256/60000 [=============>................] - ETA: 47s - loss: 0.2730 - categorical_accuracy: 0.9169
28320/60000 [=============>................] - ETA: 47s - loss: 0.2724 - categorical_accuracy: 0.9171
28384/60000 [=============>................] - ETA: 46s - loss: 0.2721 - categorical_accuracy: 0.9171
28416/60000 [=============>................] - ETA: 46s - loss: 0.2719 - categorical_accuracy: 0.9172
28480/60000 [=============>................] - ETA: 46s - loss: 0.2714 - categorical_accuracy: 0.9173
28544/60000 [=============>................] - ETA: 46s - loss: 0.2710 - categorical_accuracy: 0.9174
28576/60000 [=============>................] - ETA: 46s - loss: 0.2708 - categorical_accuracy: 0.9174
28640/60000 [=============>................] - ETA: 46s - loss: 0.2705 - categorical_accuracy: 0.9175
28704/60000 [=============>................] - ETA: 46s - loss: 0.2702 - categorical_accuracy: 0.9175
28768/60000 [=============>................] - ETA: 46s - loss: 0.2699 - categorical_accuracy: 0.9177
28832/60000 [=============>................] - ETA: 46s - loss: 0.2695 - categorical_accuracy: 0.9178
28896/60000 [=============>................] - ETA: 46s - loss: 0.2693 - categorical_accuracy: 0.9178
28960/60000 [=============>................] - ETA: 46s - loss: 0.2688 - categorical_accuracy: 0.9180
29024/60000 [=============>................] - ETA: 46s - loss: 0.2683 - categorical_accuracy: 0.9182
29088/60000 [=============>................] - ETA: 45s - loss: 0.2680 - categorical_accuracy: 0.9182
29152/60000 [=============>................] - ETA: 45s - loss: 0.2676 - categorical_accuracy: 0.9184
29216/60000 [=============>................] - ETA: 45s - loss: 0.2675 - categorical_accuracy: 0.9184
29280/60000 [=============>................] - ETA: 45s - loss: 0.2671 - categorical_accuracy: 0.9185
29344/60000 [=============>................] - ETA: 45s - loss: 0.2669 - categorical_accuracy: 0.9186
29408/60000 [=============>................] - ETA: 45s - loss: 0.2664 - categorical_accuracy: 0.9187
29472/60000 [=============>................] - ETA: 45s - loss: 0.2661 - categorical_accuracy: 0.9187
29536/60000 [=============>................] - ETA: 45s - loss: 0.2658 - categorical_accuracy: 0.9188
29600/60000 [=============>................] - ETA: 45s - loss: 0.2655 - categorical_accuracy: 0.9189
29664/60000 [=============>................] - ETA: 45s - loss: 0.2650 - categorical_accuracy: 0.9191
29728/60000 [=============>................] - ETA: 44s - loss: 0.2647 - categorical_accuracy: 0.9192
29792/60000 [=============>................] - ETA: 44s - loss: 0.2644 - categorical_accuracy: 0.9193
29856/60000 [=============>................] - ETA: 44s - loss: 0.2644 - categorical_accuracy: 0.9193
29920/60000 [=============>................] - ETA: 44s - loss: 0.2643 - categorical_accuracy: 0.9193
29984/60000 [=============>................] - ETA: 44s - loss: 0.2638 - categorical_accuracy: 0.9195
30048/60000 [==============>...............] - ETA: 44s - loss: 0.2639 - categorical_accuracy: 0.9194
30112/60000 [==============>...............] - ETA: 44s - loss: 0.2636 - categorical_accuracy: 0.9194
30176/60000 [==============>...............] - ETA: 44s - loss: 0.2633 - categorical_accuracy: 0.9195
30240/60000 [==============>...............] - ETA: 44s - loss: 0.2630 - categorical_accuracy: 0.9196
30304/60000 [==============>...............] - ETA: 44s - loss: 0.2627 - categorical_accuracy: 0.9197
30368/60000 [==============>...............] - ETA: 43s - loss: 0.2622 - categorical_accuracy: 0.9199
30432/60000 [==============>...............] - ETA: 43s - loss: 0.2617 - categorical_accuracy: 0.9201
30496/60000 [==============>...............] - ETA: 43s - loss: 0.2612 - categorical_accuracy: 0.9202
30560/60000 [==============>...............] - ETA: 43s - loss: 0.2612 - categorical_accuracy: 0.9203
30592/60000 [==============>...............] - ETA: 43s - loss: 0.2611 - categorical_accuracy: 0.9203
30656/60000 [==============>...............] - ETA: 43s - loss: 0.2607 - categorical_accuracy: 0.9204
30688/60000 [==============>...............] - ETA: 43s - loss: 0.2608 - categorical_accuracy: 0.9204
30752/60000 [==============>...............] - ETA: 43s - loss: 0.2609 - categorical_accuracy: 0.9204
30816/60000 [==============>...............] - ETA: 43s - loss: 0.2605 - categorical_accuracy: 0.9205
30880/60000 [==============>...............] - ETA: 43s - loss: 0.2601 - categorical_accuracy: 0.9206
30944/60000 [==============>...............] - ETA: 43s - loss: 0.2600 - categorical_accuracy: 0.9207
31008/60000 [==============>...............] - ETA: 43s - loss: 0.2596 - categorical_accuracy: 0.9208
31072/60000 [==============>...............] - ETA: 42s - loss: 0.2592 - categorical_accuracy: 0.9209
31136/60000 [==============>...............] - ETA: 42s - loss: 0.2588 - categorical_accuracy: 0.9210
31200/60000 [==============>...............] - ETA: 42s - loss: 0.2586 - categorical_accuracy: 0.9211
31264/60000 [==============>...............] - ETA: 42s - loss: 0.2581 - categorical_accuracy: 0.9212
31328/60000 [==============>...............] - ETA: 42s - loss: 0.2578 - categorical_accuracy: 0.9213
31392/60000 [==============>...............] - ETA: 42s - loss: 0.2576 - categorical_accuracy: 0.9214
31424/60000 [==============>...............] - ETA: 42s - loss: 0.2575 - categorical_accuracy: 0.9215
31488/60000 [==============>...............] - ETA: 42s - loss: 0.2574 - categorical_accuracy: 0.9215
31520/60000 [==============>...............] - ETA: 42s - loss: 0.2572 - categorical_accuracy: 0.9215
31584/60000 [==============>...............] - ETA: 42s - loss: 0.2569 - categorical_accuracy: 0.9216
31648/60000 [==============>...............] - ETA: 42s - loss: 0.2564 - categorical_accuracy: 0.9218
31712/60000 [==============>...............] - ETA: 41s - loss: 0.2562 - categorical_accuracy: 0.9218
31776/60000 [==============>...............] - ETA: 41s - loss: 0.2560 - categorical_accuracy: 0.9219
31840/60000 [==============>...............] - ETA: 41s - loss: 0.2557 - categorical_accuracy: 0.9220
31872/60000 [==============>...............] - ETA: 41s - loss: 0.2555 - categorical_accuracy: 0.9221
31904/60000 [==============>...............] - ETA: 41s - loss: 0.2553 - categorical_accuracy: 0.9222
31968/60000 [==============>...............] - ETA: 41s - loss: 0.2549 - categorical_accuracy: 0.9223
32032/60000 [===============>..............] - ETA: 41s - loss: 0.2546 - categorical_accuracy: 0.9224
32096/60000 [===============>..............] - ETA: 41s - loss: 0.2542 - categorical_accuracy: 0.9225
32160/60000 [===============>..............] - ETA: 41s - loss: 0.2539 - categorical_accuracy: 0.9227
32224/60000 [===============>..............] - ETA: 41s - loss: 0.2535 - categorical_accuracy: 0.9228
32288/60000 [===============>..............] - ETA: 41s - loss: 0.2532 - categorical_accuracy: 0.9228
32352/60000 [===============>..............] - ETA: 41s - loss: 0.2528 - categorical_accuracy: 0.9229
32416/60000 [===============>..............] - ETA: 40s - loss: 0.2528 - categorical_accuracy: 0.9230
32480/60000 [===============>..............] - ETA: 40s - loss: 0.2524 - categorical_accuracy: 0.9232
32544/60000 [===============>..............] - ETA: 40s - loss: 0.2520 - categorical_accuracy: 0.9233
32608/60000 [===============>..............] - ETA: 40s - loss: 0.2518 - categorical_accuracy: 0.9234
32672/60000 [===============>..............] - ETA: 40s - loss: 0.2514 - categorical_accuracy: 0.9235
32736/60000 [===============>..............] - ETA: 40s - loss: 0.2512 - categorical_accuracy: 0.9235
32800/60000 [===============>..............] - ETA: 40s - loss: 0.2508 - categorical_accuracy: 0.9235
32864/60000 [===============>..............] - ETA: 40s - loss: 0.2507 - categorical_accuracy: 0.9236
32928/60000 [===============>..............] - ETA: 40s - loss: 0.2505 - categorical_accuracy: 0.9236
32992/60000 [===============>..............] - ETA: 40s - loss: 0.2502 - categorical_accuracy: 0.9237
33056/60000 [===============>..............] - ETA: 39s - loss: 0.2500 - categorical_accuracy: 0.9238
33120/60000 [===============>..............] - ETA: 39s - loss: 0.2498 - categorical_accuracy: 0.9238
33184/60000 [===============>..............] - ETA: 39s - loss: 0.2497 - categorical_accuracy: 0.9237
33248/60000 [===============>..............] - ETA: 39s - loss: 0.2494 - categorical_accuracy: 0.9238
33312/60000 [===============>..............] - ETA: 39s - loss: 0.2491 - categorical_accuracy: 0.9239
33376/60000 [===============>..............] - ETA: 39s - loss: 0.2487 - categorical_accuracy: 0.9240
33440/60000 [===============>..............] - ETA: 39s - loss: 0.2488 - categorical_accuracy: 0.9241
33504/60000 [===============>..............] - ETA: 39s - loss: 0.2487 - categorical_accuracy: 0.9241
33568/60000 [===============>..............] - ETA: 39s - loss: 0.2483 - categorical_accuracy: 0.9242
33632/60000 [===============>..............] - ETA: 39s - loss: 0.2485 - categorical_accuracy: 0.9243
33696/60000 [===============>..............] - ETA: 39s - loss: 0.2482 - categorical_accuracy: 0.9244
33760/60000 [===============>..............] - ETA: 38s - loss: 0.2478 - categorical_accuracy: 0.9245
33824/60000 [===============>..............] - ETA: 38s - loss: 0.2476 - categorical_accuracy: 0.9246
33888/60000 [===============>..............] - ETA: 38s - loss: 0.2477 - categorical_accuracy: 0.9245
33952/60000 [===============>..............] - ETA: 38s - loss: 0.2475 - categorical_accuracy: 0.9246
34016/60000 [================>.............] - ETA: 38s - loss: 0.2475 - categorical_accuracy: 0.9247
34080/60000 [================>.............] - ETA: 38s - loss: 0.2472 - categorical_accuracy: 0.9247
34144/60000 [================>.............] - ETA: 38s - loss: 0.2470 - categorical_accuracy: 0.9248
34208/60000 [================>.............] - ETA: 38s - loss: 0.2469 - categorical_accuracy: 0.9248
34272/60000 [================>.............] - ETA: 38s - loss: 0.2466 - categorical_accuracy: 0.9249
34336/60000 [================>.............] - ETA: 38s - loss: 0.2464 - categorical_accuracy: 0.9249
34400/60000 [================>.............] - ETA: 37s - loss: 0.2463 - categorical_accuracy: 0.9250
34464/60000 [================>.............] - ETA: 37s - loss: 0.2459 - categorical_accuracy: 0.9252
34528/60000 [================>.............] - ETA: 37s - loss: 0.2455 - categorical_accuracy: 0.9252
34592/60000 [================>.............] - ETA: 37s - loss: 0.2451 - categorical_accuracy: 0.9254
34656/60000 [================>.............] - ETA: 37s - loss: 0.2447 - categorical_accuracy: 0.9255
34720/60000 [================>.............] - ETA: 37s - loss: 0.2446 - categorical_accuracy: 0.9256
34784/60000 [================>.............] - ETA: 37s - loss: 0.2445 - categorical_accuracy: 0.9256
34848/60000 [================>.............] - ETA: 37s - loss: 0.2443 - categorical_accuracy: 0.9256
34880/60000 [================>.............] - ETA: 37s - loss: 0.2443 - categorical_accuracy: 0.9256
34912/60000 [================>.............] - ETA: 37s - loss: 0.2442 - categorical_accuracy: 0.9256
34976/60000 [================>.............] - ETA: 37s - loss: 0.2438 - categorical_accuracy: 0.9257
35040/60000 [================>.............] - ETA: 37s - loss: 0.2436 - categorical_accuracy: 0.9258
35104/60000 [================>.............] - ETA: 36s - loss: 0.2434 - categorical_accuracy: 0.9259
35168/60000 [================>.............] - ETA: 36s - loss: 0.2433 - categorical_accuracy: 0.9259
35232/60000 [================>.............] - ETA: 36s - loss: 0.2431 - categorical_accuracy: 0.9260
35264/60000 [================>.............] - ETA: 36s - loss: 0.2429 - categorical_accuracy: 0.9260
35328/60000 [================>.............] - ETA: 36s - loss: 0.2426 - categorical_accuracy: 0.9261
35360/60000 [================>.............] - ETA: 36s - loss: 0.2424 - categorical_accuracy: 0.9262
35392/60000 [================>.............] - ETA: 36s - loss: 0.2426 - categorical_accuracy: 0.9262
35456/60000 [================>.............] - ETA: 36s - loss: 0.2425 - categorical_accuracy: 0.9262
35520/60000 [================>.............] - ETA: 36s - loss: 0.2423 - categorical_accuracy: 0.9263
35584/60000 [================>.............] - ETA: 36s - loss: 0.2423 - categorical_accuracy: 0.9263
35616/60000 [================>.............] - ETA: 36s - loss: 0.2424 - categorical_accuracy: 0.9263
35680/60000 [================>.............] - ETA: 36s - loss: 0.2422 - categorical_accuracy: 0.9264
35744/60000 [================>.............] - ETA: 36s - loss: 0.2419 - categorical_accuracy: 0.9264
35776/60000 [================>.............] - ETA: 35s - loss: 0.2421 - categorical_accuracy: 0.9264
35840/60000 [================>.............] - ETA: 35s - loss: 0.2418 - categorical_accuracy: 0.9265
35904/60000 [================>.............] - ETA: 35s - loss: 0.2415 - categorical_accuracy: 0.9266
35936/60000 [================>.............] - ETA: 35s - loss: 0.2413 - categorical_accuracy: 0.9267
36000/60000 [=================>............] - ETA: 35s - loss: 0.2411 - categorical_accuracy: 0.9268
36064/60000 [=================>............] - ETA: 35s - loss: 0.2410 - categorical_accuracy: 0.9268
36128/60000 [=================>............] - ETA: 35s - loss: 0.2406 - categorical_accuracy: 0.9269
36192/60000 [=================>............] - ETA: 35s - loss: 0.2405 - categorical_accuracy: 0.9270
36256/60000 [=================>............] - ETA: 35s - loss: 0.2405 - categorical_accuracy: 0.9270
36288/60000 [=================>............] - ETA: 35s - loss: 0.2404 - categorical_accuracy: 0.9270
36352/60000 [=================>............] - ETA: 35s - loss: 0.2401 - categorical_accuracy: 0.9271
36416/60000 [=================>............] - ETA: 35s - loss: 0.2400 - categorical_accuracy: 0.9271
36448/60000 [=================>............] - ETA: 34s - loss: 0.2398 - categorical_accuracy: 0.9272
36512/60000 [=================>............] - ETA: 34s - loss: 0.2395 - categorical_accuracy: 0.9273
36544/60000 [=================>............] - ETA: 34s - loss: 0.2394 - categorical_accuracy: 0.9273
36576/60000 [=================>............] - ETA: 34s - loss: 0.2395 - categorical_accuracy: 0.9273
36640/60000 [=================>............] - ETA: 34s - loss: 0.2393 - categorical_accuracy: 0.9273
36704/60000 [=================>............] - ETA: 34s - loss: 0.2393 - categorical_accuracy: 0.9274
36768/60000 [=================>............] - ETA: 34s - loss: 0.2391 - categorical_accuracy: 0.9275
36832/60000 [=================>............] - ETA: 34s - loss: 0.2388 - categorical_accuracy: 0.9276
36896/60000 [=================>............] - ETA: 34s - loss: 0.2385 - categorical_accuracy: 0.9277
36960/60000 [=================>............] - ETA: 34s - loss: 0.2382 - categorical_accuracy: 0.9278
37024/60000 [=================>............] - ETA: 34s - loss: 0.2379 - categorical_accuracy: 0.9279
37056/60000 [=================>............] - ETA: 34s - loss: 0.2379 - categorical_accuracy: 0.9278
37120/60000 [=================>............] - ETA: 33s - loss: 0.2377 - categorical_accuracy: 0.9279
37184/60000 [=================>............] - ETA: 33s - loss: 0.2374 - categorical_accuracy: 0.9279
37248/60000 [=================>............] - ETA: 33s - loss: 0.2371 - categorical_accuracy: 0.9280
37280/60000 [=================>............] - ETA: 33s - loss: 0.2370 - categorical_accuracy: 0.9280
37344/60000 [=================>............] - ETA: 33s - loss: 0.2368 - categorical_accuracy: 0.9281
37408/60000 [=================>............] - ETA: 33s - loss: 0.2367 - categorical_accuracy: 0.9282
37472/60000 [=================>............] - ETA: 33s - loss: 0.2368 - categorical_accuracy: 0.9282
37536/60000 [=================>............] - ETA: 33s - loss: 0.2367 - categorical_accuracy: 0.9283
37600/60000 [=================>............] - ETA: 33s - loss: 0.2366 - categorical_accuracy: 0.9284
37664/60000 [=================>............] - ETA: 33s - loss: 0.2362 - categorical_accuracy: 0.9284
37728/60000 [=================>............] - ETA: 33s - loss: 0.2359 - categorical_accuracy: 0.9285
37792/60000 [=================>............] - ETA: 32s - loss: 0.2357 - categorical_accuracy: 0.9286
37856/60000 [=================>............] - ETA: 32s - loss: 0.2357 - categorical_accuracy: 0.9287
37920/60000 [=================>............] - ETA: 32s - loss: 0.2357 - categorical_accuracy: 0.9287
37984/60000 [=================>............] - ETA: 32s - loss: 0.2356 - categorical_accuracy: 0.9287
38048/60000 [==================>...........] - ETA: 32s - loss: 0.2354 - categorical_accuracy: 0.9288
38112/60000 [==================>...........] - ETA: 32s - loss: 0.2353 - categorical_accuracy: 0.9288
38176/60000 [==================>...........] - ETA: 32s - loss: 0.2351 - categorical_accuracy: 0.9289
38240/60000 [==================>...........] - ETA: 32s - loss: 0.2347 - categorical_accuracy: 0.9290
38304/60000 [==================>...........] - ETA: 32s - loss: 0.2346 - categorical_accuracy: 0.9290
38368/60000 [==================>...........] - ETA: 32s - loss: 0.2343 - categorical_accuracy: 0.9291
38432/60000 [==================>...........] - ETA: 31s - loss: 0.2342 - categorical_accuracy: 0.9292
38496/60000 [==================>...........] - ETA: 31s - loss: 0.2343 - categorical_accuracy: 0.9292
38560/60000 [==================>...........] - ETA: 31s - loss: 0.2344 - categorical_accuracy: 0.9291
38624/60000 [==================>...........] - ETA: 31s - loss: 0.2344 - categorical_accuracy: 0.9292
38688/60000 [==================>...........] - ETA: 31s - loss: 0.2340 - categorical_accuracy: 0.9293
38720/60000 [==================>...........] - ETA: 31s - loss: 0.2341 - categorical_accuracy: 0.9292
38752/60000 [==================>...........] - ETA: 31s - loss: 0.2340 - categorical_accuracy: 0.9293
38816/60000 [==================>...........] - ETA: 31s - loss: 0.2337 - categorical_accuracy: 0.9293
38880/60000 [==================>...........] - ETA: 31s - loss: 0.2335 - categorical_accuracy: 0.9294
38944/60000 [==================>...........] - ETA: 31s - loss: 0.2332 - categorical_accuracy: 0.9295
39008/60000 [==================>...........] - ETA: 31s - loss: 0.2329 - categorical_accuracy: 0.9296
39040/60000 [==================>...........] - ETA: 31s - loss: 0.2327 - categorical_accuracy: 0.9296
39104/60000 [==================>...........] - ETA: 30s - loss: 0.2324 - categorical_accuracy: 0.9297
39168/60000 [==================>...........] - ETA: 30s - loss: 0.2324 - categorical_accuracy: 0.9297
39232/60000 [==================>...........] - ETA: 30s - loss: 0.2321 - categorical_accuracy: 0.9298
39296/60000 [==================>...........] - ETA: 30s - loss: 0.2323 - categorical_accuracy: 0.9298
39328/60000 [==================>...........] - ETA: 30s - loss: 0.2322 - categorical_accuracy: 0.9298
39392/60000 [==================>...........] - ETA: 30s - loss: 0.2320 - categorical_accuracy: 0.9299
39456/60000 [==================>...........] - ETA: 30s - loss: 0.2319 - categorical_accuracy: 0.9299
39520/60000 [==================>...........] - ETA: 30s - loss: 0.2316 - categorical_accuracy: 0.9300
39584/60000 [==================>...........] - ETA: 30s - loss: 0.2314 - categorical_accuracy: 0.9300
39648/60000 [==================>...........] - ETA: 30s - loss: 0.2311 - categorical_accuracy: 0.9301
39712/60000 [==================>...........] - ETA: 30s - loss: 0.2308 - categorical_accuracy: 0.9302
39776/60000 [==================>...........] - ETA: 29s - loss: 0.2307 - categorical_accuracy: 0.9303
39840/60000 [==================>...........] - ETA: 29s - loss: 0.2304 - categorical_accuracy: 0.9304
39904/60000 [==================>...........] - ETA: 29s - loss: 0.2302 - categorical_accuracy: 0.9304
39968/60000 [==================>...........] - ETA: 29s - loss: 0.2299 - categorical_accuracy: 0.9305
40032/60000 [===================>..........] - ETA: 29s - loss: 0.2298 - categorical_accuracy: 0.9305
40096/60000 [===================>..........] - ETA: 29s - loss: 0.2295 - categorical_accuracy: 0.9306
40160/60000 [===================>..........] - ETA: 29s - loss: 0.2294 - categorical_accuracy: 0.9307
40192/60000 [===================>..........] - ETA: 29s - loss: 0.2293 - categorical_accuracy: 0.9307
40256/60000 [===================>..........] - ETA: 29s - loss: 0.2290 - categorical_accuracy: 0.9308
40320/60000 [===================>..........] - ETA: 29s - loss: 0.2288 - categorical_accuracy: 0.9309
40384/60000 [===================>..........] - ETA: 29s - loss: 0.2287 - categorical_accuracy: 0.9309
40416/60000 [===================>..........] - ETA: 29s - loss: 0.2286 - categorical_accuracy: 0.9310
40480/60000 [===================>..........] - ETA: 28s - loss: 0.2283 - categorical_accuracy: 0.9310
40544/60000 [===================>..........] - ETA: 28s - loss: 0.2281 - categorical_accuracy: 0.9311
40608/60000 [===================>..........] - ETA: 28s - loss: 0.2283 - categorical_accuracy: 0.9311
40672/60000 [===================>..........] - ETA: 28s - loss: 0.2280 - categorical_accuracy: 0.9312
40736/60000 [===================>..........] - ETA: 28s - loss: 0.2278 - categorical_accuracy: 0.9313
40800/60000 [===================>..........] - ETA: 28s - loss: 0.2276 - categorical_accuracy: 0.9313
40832/60000 [===================>..........] - ETA: 28s - loss: 0.2275 - categorical_accuracy: 0.9313
40896/60000 [===================>..........] - ETA: 28s - loss: 0.2276 - categorical_accuracy: 0.9313
40960/60000 [===================>..........] - ETA: 28s - loss: 0.2275 - categorical_accuracy: 0.9313
41024/60000 [===================>..........] - ETA: 28s - loss: 0.2274 - categorical_accuracy: 0.9314
41088/60000 [===================>..........] - ETA: 28s - loss: 0.2272 - categorical_accuracy: 0.9314
41152/60000 [===================>..........] - ETA: 27s - loss: 0.2270 - categorical_accuracy: 0.9314
41216/60000 [===================>..........] - ETA: 27s - loss: 0.2267 - categorical_accuracy: 0.9316
41280/60000 [===================>..........] - ETA: 27s - loss: 0.2269 - categorical_accuracy: 0.9315
41344/60000 [===================>..........] - ETA: 27s - loss: 0.2267 - categorical_accuracy: 0.9315
41408/60000 [===================>..........] - ETA: 27s - loss: 0.2266 - categorical_accuracy: 0.9316
41440/60000 [===================>..........] - ETA: 27s - loss: 0.2266 - categorical_accuracy: 0.9316
41504/60000 [===================>..........] - ETA: 27s - loss: 0.2264 - categorical_accuracy: 0.9316
41568/60000 [===================>..........] - ETA: 27s - loss: 0.2261 - categorical_accuracy: 0.9317
41632/60000 [===================>..........] - ETA: 27s - loss: 0.2259 - categorical_accuracy: 0.9318
41696/60000 [===================>..........] - ETA: 27s - loss: 0.2257 - categorical_accuracy: 0.9318
41760/60000 [===================>..........] - ETA: 27s - loss: 0.2254 - categorical_accuracy: 0.9319
41792/60000 [===================>..........] - ETA: 26s - loss: 0.2254 - categorical_accuracy: 0.9319
41856/60000 [===================>..........] - ETA: 26s - loss: 0.2251 - categorical_accuracy: 0.9320
41920/60000 [===================>..........] - ETA: 26s - loss: 0.2250 - categorical_accuracy: 0.9320
41984/60000 [===================>..........] - ETA: 26s - loss: 0.2248 - categorical_accuracy: 0.9321
42048/60000 [====================>.........] - ETA: 26s - loss: 0.2246 - categorical_accuracy: 0.9321
42112/60000 [====================>.........] - ETA: 26s - loss: 0.2243 - categorical_accuracy: 0.9322
42176/60000 [====================>.........] - ETA: 26s - loss: 0.2242 - categorical_accuracy: 0.9323
42240/60000 [====================>.........] - ETA: 26s - loss: 0.2240 - categorical_accuracy: 0.9323
42304/60000 [====================>.........] - ETA: 26s - loss: 0.2238 - categorical_accuracy: 0.9324
42368/60000 [====================>.........] - ETA: 26s - loss: 0.2236 - categorical_accuracy: 0.9325
42432/60000 [====================>.........] - ETA: 26s - loss: 0.2234 - categorical_accuracy: 0.9325
42496/60000 [====================>.........] - ETA: 25s - loss: 0.2232 - categorical_accuracy: 0.9326
42560/60000 [====================>.........] - ETA: 25s - loss: 0.2229 - categorical_accuracy: 0.9327
42624/60000 [====================>.........] - ETA: 25s - loss: 0.2228 - categorical_accuracy: 0.9327
42688/60000 [====================>.........] - ETA: 25s - loss: 0.2227 - categorical_accuracy: 0.9327
42752/60000 [====================>.........] - ETA: 25s - loss: 0.2225 - categorical_accuracy: 0.9328
42816/60000 [====================>.........] - ETA: 25s - loss: 0.2222 - categorical_accuracy: 0.9328
42880/60000 [====================>.........] - ETA: 25s - loss: 0.2222 - categorical_accuracy: 0.9329
42912/60000 [====================>.........] - ETA: 25s - loss: 0.2221 - categorical_accuracy: 0.9329
42976/60000 [====================>.........] - ETA: 25s - loss: 0.2218 - categorical_accuracy: 0.9330
43040/60000 [====================>.........] - ETA: 25s - loss: 0.2216 - categorical_accuracy: 0.9331
43104/60000 [====================>.........] - ETA: 25s - loss: 0.2213 - categorical_accuracy: 0.9332
43168/60000 [====================>.........] - ETA: 24s - loss: 0.2211 - categorical_accuracy: 0.9332
43232/60000 [====================>.........] - ETA: 24s - loss: 0.2210 - categorical_accuracy: 0.9332
43296/60000 [====================>.........] - ETA: 24s - loss: 0.2209 - categorical_accuracy: 0.9333
43360/60000 [====================>.........] - ETA: 24s - loss: 0.2207 - categorical_accuracy: 0.9333
43392/60000 [====================>.........] - ETA: 24s - loss: 0.2209 - categorical_accuracy: 0.9333
43456/60000 [====================>.........] - ETA: 24s - loss: 0.2206 - categorical_accuracy: 0.9334
43520/60000 [====================>.........] - ETA: 24s - loss: 0.2204 - categorical_accuracy: 0.9334
43552/60000 [====================>.........] - ETA: 24s - loss: 0.2203 - categorical_accuracy: 0.9334
43616/60000 [====================>.........] - ETA: 24s - loss: 0.2201 - categorical_accuracy: 0.9335
43680/60000 [====================>.........] - ETA: 24s - loss: 0.2198 - categorical_accuracy: 0.9336
43744/60000 [====================>.........] - ETA: 24s - loss: 0.2195 - categorical_accuracy: 0.9337
43808/60000 [====================>.........] - ETA: 24s - loss: 0.2193 - categorical_accuracy: 0.9337
43872/60000 [====================>.........] - ETA: 23s - loss: 0.2191 - categorical_accuracy: 0.9338
43936/60000 [====================>.........] - ETA: 23s - loss: 0.2189 - categorical_accuracy: 0.9338
44000/60000 [=====================>........] - ETA: 23s - loss: 0.2187 - categorical_accuracy: 0.9338
44064/60000 [=====================>........] - ETA: 23s - loss: 0.2185 - categorical_accuracy: 0.9339
44128/60000 [=====================>........] - ETA: 23s - loss: 0.2184 - categorical_accuracy: 0.9339
44192/60000 [=====================>........] - ETA: 23s - loss: 0.2183 - categorical_accuracy: 0.9339
44256/60000 [=====================>........] - ETA: 23s - loss: 0.2180 - categorical_accuracy: 0.9340
44320/60000 [=====================>........] - ETA: 23s - loss: 0.2179 - categorical_accuracy: 0.9340
44352/60000 [=====================>........] - ETA: 23s - loss: 0.2178 - categorical_accuracy: 0.9341
44384/60000 [=====================>........] - ETA: 23s - loss: 0.2177 - categorical_accuracy: 0.9341
44448/60000 [=====================>........] - ETA: 23s - loss: 0.2176 - categorical_accuracy: 0.9341
44480/60000 [=====================>........] - ETA: 23s - loss: 0.2175 - categorical_accuracy: 0.9342
44544/60000 [=====================>........] - ETA: 22s - loss: 0.2173 - categorical_accuracy: 0.9342
44608/60000 [=====================>........] - ETA: 22s - loss: 0.2171 - categorical_accuracy: 0.9343
44672/60000 [=====================>........] - ETA: 22s - loss: 0.2168 - categorical_accuracy: 0.9343
44736/60000 [=====================>........] - ETA: 22s - loss: 0.2168 - categorical_accuracy: 0.9343
44800/60000 [=====================>........] - ETA: 22s - loss: 0.2167 - categorical_accuracy: 0.9344
44832/60000 [=====================>........] - ETA: 22s - loss: 0.2169 - categorical_accuracy: 0.9344
44896/60000 [=====================>........] - ETA: 22s - loss: 0.2167 - categorical_accuracy: 0.9344
44960/60000 [=====================>........] - ETA: 22s - loss: 0.2167 - categorical_accuracy: 0.9345
45024/60000 [=====================>........] - ETA: 22s - loss: 0.2164 - categorical_accuracy: 0.9345
45088/60000 [=====================>........] - ETA: 22s - loss: 0.2162 - categorical_accuracy: 0.9346
45152/60000 [=====================>........] - ETA: 22s - loss: 0.2163 - categorical_accuracy: 0.9346
45216/60000 [=====================>........] - ETA: 21s - loss: 0.2162 - categorical_accuracy: 0.9347
45280/60000 [=====================>........] - ETA: 21s - loss: 0.2160 - categorical_accuracy: 0.9347
45344/60000 [=====================>........] - ETA: 21s - loss: 0.2158 - categorical_accuracy: 0.9348
45408/60000 [=====================>........] - ETA: 21s - loss: 0.2156 - categorical_accuracy: 0.9348
45472/60000 [=====================>........] - ETA: 21s - loss: 0.2153 - categorical_accuracy: 0.9349
45536/60000 [=====================>........] - ETA: 21s - loss: 0.2151 - categorical_accuracy: 0.9350
45600/60000 [=====================>........] - ETA: 21s - loss: 0.2149 - categorical_accuracy: 0.9350
45664/60000 [=====================>........] - ETA: 21s - loss: 0.2147 - categorical_accuracy: 0.9351
45728/60000 [=====================>........] - ETA: 21s - loss: 0.2146 - categorical_accuracy: 0.9351
45792/60000 [=====================>........] - ETA: 21s - loss: 0.2147 - categorical_accuracy: 0.9351
45856/60000 [=====================>........] - ETA: 20s - loss: 0.2145 - categorical_accuracy: 0.9352
45920/60000 [=====================>........] - ETA: 20s - loss: 0.2142 - categorical_accuracy: 0.9353
45984/60000 [=====================>........] - ETA: 20s - loss: 0.2140 - categorical_accuracy: 0.9353
46048/60000 [======================>.......] - ETA: 20s - loss: 0.2137 - categorical_accuracy: 0.9354
46080/60000 [======================>.......] - ETA: 20s - loss: 0.2136 - categorical_accuracy: 0.9354
46112/60000 [======================>.......] - ETA: 20s - loss: 0.2135 - categorical_accuracy: 0.9355
46176/60000 [======================>.......] - ETA: 20s - loss: 0.2135 - categorical_accuracy: 0.9355
46240/60000 [======================>.......] - ETA: 20s - loss: 0.2135 - categorical_accuracy: 0.9354
46304/60000 [======================>.......] - ETA: 20s - loss: 0.2133 - categorical_accuracy: 0.9355
46368/60000 [======================>.......] - ETA: 20s - loss: 0.2132 - categorical_accuracy: 0.9355
46432/60000 [======================>.......] - ETA: 20s - loss: 0.2130 - categorical_accuracy: 0.9356
46496/60000 [======================>.......] - ETA: 20s - loss: 0.2128 - categorical_accuracy: 0.9356
46560/60000 [======================>.......] - ETA: 19s - loss: 0.2126 - categorical_accuracy: 0.9356
46592/60000 [======================>.......] - ETA: 19s - loss: 0.2125 - categorical_accuracy: 0.9357
46656/60000 [======================>.......] - ETA: 19s - loss: 0.2123 - categorical_accuracy: 0.9357
46720/60000 [======================>.......] - ETA: 19s - loss: 0.2122 - categorical_accuracy: 0.9358
46784/60000 [======================>.......] - ETA: 19s - loss: 0.2120 - categorical_accuracy: 0.9359
46848/60000 [======================>.......] - ETA: 19s - loss: 0.2117 - categorical_accuracy: 0.9359
46912/60000 [======================>.......] - ETA: 19s - loss: 0.2117 - categorical_accuracy: 0.9360
46976/60000 [======================>.......] - ETA: 19s - loss: 0.2115 - categorical_accuracy: 0.9361
47008/60000 [======================>.......] - ETA: 19s - loss: 0.2114 - categorical_accuracy: 0.9361
47040/60000 [======================>.......] - ETA: 19s - loss: 0.2113 - categorical_accuracy: 0.9361
47104/60000 [======================>.......] - ETA: 19s - loss: 0.2113 - categorical_accuracy: 0.9361
47168/60000 [======================>.......] - ETA: 19s - loss: 0.2111 - categorical_accuracy: 0.9362
47232/60000 [======================>.......] - ETA: 18s - loss: 0.2109 - categorical_accuracy: 0.9363
47296/60000 [======================>.......] - ETA: 18s - loss: 0.2107 - categorical_accuracy: 0.9363
47360/60000 [======================>.......] - ETA: 18s - loss: 0.2107 - categorical_accuracy: 0.9363
47424/60000 [======================>.......] - ETA: 18s - loss: 0.2105 - categorical_accuracy: 0.9363
47488/60000 [======================>.......] - ETA: 18s - loss: 0.2103 - categorical_accuracy: 0.9364
47552/60000 [======================>.......] - ETA: 18s - loss: 0.2101 - categorical_accuracy: 0.9364
47616/60000 [======================>.......] - ETA: 18s - loss: 0.2098 - categorical_accuracy: 0.9365
47680/60000 [======================>.......] - ETA: 18s - loss: 0.2096 - categorical_accuracy: 0.9366
47744/60000 [======================>.......] - ETA: 18s - loss: 0.2095 - categorical_accuracy: 0.9366
47808/60000 [======================>.......] - ETA: 18s - loss: 0.2093 - categorical_accuracy: 0.9367
47872/60000 [======================>.......] - ETA: 17s - loss: 0.2093 - categorical_accuracy: 0.9367
47936/60000 [======================>.......] - ETA: 17s - loss: 0.2090 - categorical_accuracy: 0.9368
47968/60000 [======================>.......] - ETA: 17s - loss: 0.2090 - categorical_accuracy: 0.9368
48032/60000 [=======================>......] - ETA: 17s - loss: 0.2088 - categorical_accuracy: 0.9369
48096/60000 [=======================>......] - ETA: 17s - loss: 0.2087 - categorical_accuracy: 0.9369
48160/60000 [=======================>......] - ETA: 17s - loss: 0.2085 - categorical_accuracy: 0.9370
48224/60000 [=======================>......] - ETA: 17s - loss: 0.2084 - categorical_accuracy: 0.9370
48288/60000 [=======================>......] - ETA: 17s - loss: 0.2082 - categorical_accuracy: 0.9371
48320/60000 [=======================>......] - ETA: 17s - loss: 0.2082 - categorical_accuracy: 0.9371
48384/60000 [=======================>......] - ETA: 17s - loss: 0.2081 - categorical_accuracy: 0.9371
48448/60000 [=======================>......] - ETA: 17s - loss: 0.2081 - categorical_accuracy: 0.9371
48512/60000 [=======================>......] - ETA: 17s - loss: 0.2083 - categorical_accuracy: 0.9371
48576/60000 [=======================>......] - ETA: 16s - loss: 0.2081 - categorical_accuracy: 0.9372
48640/60000 [=======================>......] - ETA: 16s - loss: 0.2080 - categorical_accuracy: 0.9372
48704/60000 [=======================>......] - ETA: 16s - loss: 0.2078 - categorical_accuracy: 0.9373
48768/60000 [=======================>......] - ETA: 16s - loss: 0.2078 - categorical_accuracy: 0.9373
48832/60000 [=======================>......] - ETA: 16s - loss: 0.2076 - categorical_accuracy: 0.9373
48896/60000 [=======================>......] - ETA: 16s - loss: 0.2075 - categorical_accuracy: 0.9374
48928/60000 [=======================>......] - ETA: 16s - loss: 0.2076 - categorical_accuracy: 0.9373
48992/60000 [=======================>......] - ETA: 16s - loss: 0.2074 - categorical_accuracy: 0.9374
49024/60000 [=======================>......] - ETA: 16s - loss: 0.2074 - categorical_accuracy: 0.9373
49088/60000 [=======================>......] - ETA: 16s - loss: 0.2072 - categorical_accuracy: 0.9374
49152/60000 [=======================>......] - ETA: 16s - loss: 0.2072 - categorical_accuracy: 0.9374
49184/60000 [=======================>......] - ETA: 16s - loss: 0.2072 - categorical_accuracy: 0.9374
49248/60000 [=======================>......] - ETA: 15s - loss: 0.2071 - categorical_accuracy: 0.9374
49312/60000 [=======================>......] - ETA: 15s - loss: 0.2069 - categorical_accuracy: 0.9375
49376/60000 [=======================>......] - ETA: 15s - loss: 0.2068 - categorical_accuracy: 0.9375
49440/60000 [=======================>......] - ETA: 15s - loss: 0.2067 - categorical_accuracy: 0.9375
49504/60000 [=======================>......] - ETA: 15s - loss: 0.2066 - categorical_accuracy: 0.9376
49568/60000 [=======================>......] - ETA: 15s - loss: 0.2065 - categorical_accuracy: 0.9376
49632/60000 [=======================>......] - ETA: 15s - loss: 0.2063 - categorical_accuracy: 0.9377
49696/60000 [=======================>......] - ETA: 15s - loss: 0.2061 - categorical_accuracy: 0.9377
49760/60000 [=======================>......] - ETA: 15s - loss: 0.2059 - categorical_accuracy: 0.9378
49824/60000 [=======================>......] - ETA: 15s - loss: 0.2059 - categorical_accuracy: 0.9379
49888/60000 [=======================>......] - ETA: 14s - loss: 0.2057 - categorical_accuracy: 0.9379
49952/60000 [=======================>......] - ETA: 14s - loss: 0.2055 - categorical_accuracy: 0.9380
50016/60000 [========================>.....] - ETA: 14s - loss: 0.2053 - categorical_accuracy: 0.9380
50048/60000 [========================>.....] - ETA: 14s - loss: 0.2052 - categorical_accuracy: 0.9380
50112/60000 [========================>.....] - ETA: 14s - loss: 0.2050 - categorical_accuracy: 0.9381
50176/60000 [========================>.....] - ETA: 14s - loss: 0.2048 - categorical_accuracy: 0.9382
50240/60000 [========================>.....] - ETA: 14s - loss: 0.2047 - categorical_accuracy: 0.9382
50304/60000 [========================>.....] - ETA: 14s - loss: 0.2044 - categorical_accuracy: 0.9383
50336/60000 [========================>.....] - ETA: 14s - loss: 0.2043 - categorical_accuracy: 0.9383
50400/60000 [========================>.....] - ETA: 14s - loss: 0.2041 - categorical_accuracy: 0.9384
50464/60000 [========================>.....] - ETA: 14s - loss: 0.2041 - categorical_accuracy: 0.9384
50528/60000 [========================>.....] - ETA: 14s - loss: 0.2039 - categorical_accuracy: 0.9384
50592/60000 [========================>.....] - ETA: 13s - loss: 0.2036 - categorical_accuracy: 0.9385
50656/60000 [========================>.....] - ETA: 13s - loss: 0.2037 - categorical_accuracy: 0.9386
50720/60000 [========================>.....] - ETA: 13s - loss: 0.2034 - categorical_accuracy: 0.9386
50784/60000 [========================>.....] - ETA: 13s - loss: 0.2034 - categorical_accuracy: 0.9386
50848/60000 [========================>.....] - ETA: 13s - loss: 0.2032 - categorical_accuracy: 0.9387
50880/60000 [========================>.....] - ETA: 13s - loss: 0.2031 - categorical_accuracy: 0.9387
50944/60000 [========================>.....] - ETA: 13s - loss: 0.2030 - categorical_accuracy: 0.9387
51008/60000 [========================>.....] - ETA: 13s - loss: 0.2028 - categorical_accuracy: 0.9388
51072/60000 [========================>.....] - ETA: 13s - loss: 0.2027 - categorical_accuracy: 0.9388
51136/60000 [========================>.....] - ETA: 13s - loss: 0.2026 - categorical_accuracy: 0.9388
51200/60000 [========================>.....] - ETA: 13s - loss: 0.2025 - categorical_accuracy: 0.9388
51232/60000 [========================>.....] - ETA: 12s - loss: 0.2023 - categorical_accuracy: 0.9389
51296/60000 [========================>.....] - ETA: 12s - loss: 0.2025 - categorical_accuracy: 0.9389
51360/60000 [========================>.....] - ETA: 12s - loss: 0.2023 - categorical_accuracy: 0.9389
51424/60000 [========================>.....] - ETA: 12s - loss: 0.2022 - categorical_accuracy: 0.9390
51488/60000 [========================>.....] - ETA: 12s - loss: 0.2020 - categorical_accuracy: 0.9390
51520/60000 [========================>.....] - ETA: 12s - loss: 0.2019 - categorical_accuracy: 0.9391
51584/60000 [========================>.....] - ETA: 12s - loss: 0.2017 - categorical_accuracy: 0.9391
51616/60000 [========================>.....] - ETA: 12s - loss: 0.2016 - categorical_accuracy: 0.9391
51680/60000 [========================>.....] - ETA: 12s - loss: 0.2014 - categorical_accuracy: 0.9392
51744/60000 [========================>.....] - ETA: 12s - loss: 0.2012 - categorical_accuracy: 0.9393
51808/60000 [========================>.....] - ETA: 12s - loss: 0.2010 - categorical_accuracy: 0.9393
51872/60000 [========================>.....] - ETA: 12s - loss: 0.2009 - categorical_accuracy: 0.9393
51936/60000 [========================>.....] - ETA: 11s - loss: 0.2007 - categorical_accuracy: 0.9394
52000/60000 [=========================>....] - ETA: 11s - loss: 0.2006 - categorical_accuracy: 0.9395
52064/60000 [=========================>....] - ETA: 11s - loss: 0.2004 - categorical_accuracy: 0.9395
52128/60000 [=========================>....] - ETA: 11s - loss: 0.2003 - categorical_accuracy: 0.9395
52192/60000 [=========================>....] - ETA: 11s - loss: 0.2001 - categorical_accuracy: 0.9396
52256/60000 [=========================>....] - ETA: 11s - loss: 0.1999 - categorical_accuracy: 0.9396
52320/60000 [=========================>....] - ETA: 11s - loss: 0.1997 - categorical_accuracy: 0.9396
52384/60000 [=========================>....] - ETA: 11s - loss: 0.1996 - categorical_accuracy: 0.9397
52448/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9397
52512/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9397
52576/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9398
52640/60000 [=========================>....] - ETA: 10s - loss: 0.1992 - categorical_accuracy: 0.9398
52704/60000 [=========================>....] - ETA: 10s - loss: 0.1991 - categorical_accuracy: 0.9398
52768/60000 [=========================>....] - ETA: 10s - loss: 0.1989 - categorical_accuracy: 0.9398
52832/60000 [=========================>....] - ETA: 10s - loss: 0.1987 - categorical_accuracy: 0.9399
52896/60000 [=========================>....] - ETA: 10s - loss: 0.1985 - categorical_accuracy: 0.9400
52960/60000 [=========================>....] - ETA: 10s - loss: 0.1983 - categorical_accuracy: 0.9400
53024/60000 [=========================>....] - ETA: 10s - loss: 0.1982 - categorical_accuracy: 0.9401
53088/60000 [=========================>....] - ETA: 10s - loss: 0.1981 - categorical_accuracy: 0.9401
53120/60000 [=========================>....] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9401
53184/60000 [=========================>....] - ETA: 10s - loss: 0.1981 - categorical_accuracy: 0.9401
53248/60000 [=========================>....] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9402
53280/60000 [=========================>....] - ETA: 9s - loss: 0.1981 - categorical_accuracy: 0.9401 
53312/60000 [=========================>....] - ETA: 9s - loss: 0.1981 - categorical_accuracy: 0.9401
53376/60000 [=========================>....] - ETA: 9s - loss: 0.1982 - categorical_accuracy: 0.9401
53408/60000 [=========================>....] - ETA: 9s - loss: 0.1981 - categorical_accuracy: 0.9401
53472/60000 [=========================>....] - ETA: 9s - loss: 0.1981 - categorical_accuracy: 0.9401
53536/60000 [=========================>....] - ETA: 9s - loss: 0.1980 - categorical_accuracy: 0.9402
53600/60000 [=========================>....] - ETA: 9s - loss: 0.1979 - categorical_accuracy: 0.9402
53664/60000 [=========================>....] - ETA: 9s - loss: 0.1977 - categorical_accuracy: 0.9403
53728/60000 [=========================>....] - ETA: 9s - loss: 0.1975 - categorical_accuracy: 0.9404
53792/60000 [=========================>....] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9404
53856/60000 [=========================>....] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9404
53920/60000 [=========================>....] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9404
53984/60000 [=========================>....] - ETA: 8s - loss: 0.1973 - categorical_accuracy: 0.9404
54048/60000 [==========================>...] - ETA: 8s - loss: 0.1973 - categorical_accuracy: 0.9404
54080/60000 [==========================>...] - ETA: 8s - loss: 0.1972 - categorical_accuracy: 0.9404
54144/60000 [==========================>...] - ETA: 8s - loss: 0.1971 - categorical_accuracy: 0.9405
54208/60000 [==========================>...] - ETA: 8s - loss: 0.1970 - categorical_accuracy: 0.9405
54272/60000 [==========================>...] - ETA: 8s - loss: 0.1969 - categorical_accuracy: 0.9406
54336/60000 [==========================>...] - ETA: 8s - loss: 0.1968 - categorical_accuracy: 0.9406
54400/60000 [==========================>...] - ETA: 8s - loss: 0.1966 - categorical_accuracy: 0.9406
54464/60000 [==========================>...] - ETA: 8s - loss: 0.1965 - categorical_accuracy: 0.9407
54528/60000 [==========================>...] - ETA: 8s - loss: 0.1964 - categorical_accuracy: 0.9407
54592/60000 [==========================>...] - ETA: 8s - loss: 0.1963 - categorical_accuracy: 0.9407
54656/60000 [==========================>...] - ETA: 7s - loss: 0.1962 - categorical_accuracy: 0.9408
54720/60000 [==========================>...] - ETA: 7s - loss: 0.1961 - categorical_accuracy: 0.9408
54784/60000 [==========================>...] - ETA: 7s - loss: 0.1959 - categorical_accuracy: 0.9408
54848/60000 [==========================>...] - ETA: 7s - loss: 0.1959 - categorical_accuracy: 0.9408
54912/60000 [==========================>...] - ETA: 7s - loss: 0.1957 - categorical_accuracy: 0.9409
54976/60000 [==========================>...] - ETA: 7s - loss: 0.1956 - categorical_accuracy: 0.9409
55040/60000 [==========================>...] - ETA: 7s - loss: 0.1955 - categorical_accuracy: 0.9409
55072/60000 [==========================>...] - ETA: 7s - loss: 0.1955 - categorical_accuracy: 0.9409
55136/60000 [==========================>...] - ETA: 7s - loss: 0.1957 - categorical_accuracy: 0.9409
55200/60000 [==========================>...] - ETA: 7s - loss: 0.1956 - categorical_accuracy: 0.9409
55264/60000 [==========================>...] - ETA: 7s - loss: 0.1954 - categorical_accuracy: 0.9410
55328/60000 [==========================>...] - ETA: 6s - loss: 0.1954 - categorical_accuracy: 0.9409
55392/60000 [==========================>...] - ETA: 6s - loss: 0.1953 - categorical_accuracy: 0.9410
55424/60000 [==========================>...] - ETA: 6s - loss: 0.1952 - categorical_accuracy: 0.9410
55488/60000 [==========================>...] - ETA: 6s - loss: 0.1951 - categorical_accuracy: 0.9410
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1951 - categorical_accuracy: 0.9410
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1950 - categorical_accuracy: 0.9411
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1948 - categorical_accuracy: 0.9411
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1946 - categorical_accuracy: 0.9411
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1945 - categorical_accuracy: 0.9412
55872/60000 [==========================>...] - ETA: 6s - loss: 0.1943 - categorical_accuracy: 0.9412
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1943 - categorical_accuracy: 0.9413
56000/60000 [===========================>..] - ETA: 5s - loss: 0.1943 - categorical_accuracy: 0.9413
56064/60000 [===========================>..] - ETA: 5s - loss: 0.1941 - categorical_accuracy: 0.9414
56128/60000 [===========================>..] - ETA: 5s - loss: 0.1939 - categorical_accuracy: 0.9414
56192/60000 [===========================>..] - ETA: 5s - loss: 0.1937 - categorical_accuracy: 0.9415
56256/60000 [===========================>..] - ETA: 5s - loss: 0.1935 - categorical_accuracy: 0.9416
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1934 - categorical_accuracy: 0.9416
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1933 - categorical_accuracy: 0.9416
56448/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9416
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9417
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1929 - categorical_accuracy: 0.9417
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1928 - categorical_accuracy: 0.9417
56672/60000 [===========================>..] - ETA: 4s - loss: 0.1927 - categorical_accuracy: 0.9418
56704/60000 [===========================>..] - ETA: 4s - loss: 0.1927 - categorical_accuracy: 0.9418
56768/60000 [===========================>..] - ETA: 4s - loss: 0.1926 - categorical_accuracy: 0.9418
56800/60000 [===========================>..] - ETA: 4s - loss: 0.1926 - categorical_accuracy: 0.9418
56864/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9419
56928/60000 [===========================>..] - ETA: 4s - loss: 0.1923 - categorical_accuracy: 0.9419
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1922 - categorical_accuracy: 0.9420
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1920 - categorical_accuracy: 0.9420
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1919 - categorical_accuracy: 0.9421
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1919 - categorical_accuracy: 0.9421
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1918 - categorical_accuracy: 0.9421
57312/60000 [===========================>..] - ETA: 3s - loss: 0.1916 - categorical_accuracy: 0.9422
57376/60000 [===========================>..] - ETA: 3s - loss: 0.1915 - categorical_accuracy: 0.9422
57440/60000 [===========================>..] - ETA: 3s - loss: 0.1914 - categorical_accuracy: 0.9423
57472/60000 [===========================>..] - ETA: 3s - loss: 0.1913 - categorical_accuracy: 0.9423
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1913 - categorical_accuracy: 0.9423
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1913 - categorical_accuracy: 0.9423
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1912 - categorical_accuracy: 0.9423
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1912 - categorical_accuracy: 0.9423
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1914 - categorical_accuracy: 0.9423
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1912 - categorical_accuracy: 0.9423
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1911 - categorical_accuracy: 0.9424
57984/60000 [===========================>..] - ETA: 2s - loss: 0.1909 - categorical_accuracy: 0.9424
58048/60000 [============================>.] - ETA: 2s - loss: 0.1907 - categorical_accuracy: 0.9425
58080/60000 [============================>.] - ETA: 2s - loss: 0.1906 - categorical_accuracy: 0.9425
58144/60000 [============================>.] - ETA: 2s - loss: 0.1905 - categorical_accuracy: 0.9425
58208/60000 [============================>.] - ETA: 2s - loss: 0.1905 - categorical_accuracy: 0.9426
58272/60000 [============================>.] - ETA: 2s - loss: 0.1904 - categorical_accuracy: 0.9426
58336/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9426
58400/60000 [============================>.] - ETA: 2s - loss: 0.1901 - categorical_accuracy: 0.9427
58464/60000 [============================>.] - ETA: 2s - loss: 0.1900 - categorical_accuracy: 0.9427
58528/60000 [============================>.] - ETA: 2s - loss: 0.1899 - categorical_accuracy: 0.9427
58592/60000 [============================>.] - ETA: 2s - loss: 0.1898 - categorical_accuracy: 0.9428
58656/60000 [============================>.] - ETA: 1s - loss: 0.1898 - categorical_accuracy: 0.9428
58720/60000 [============================>.] - ETA: 1s - loss: 0.1897 - categorical_accuracy: 0.9428
58784/60000 [============================>.] - ETA: 1s - loss: 0.1896 - categorical_accuracy: 0.9428
58816/60000 [============================>.] - ETA: 1s - loss: 0.1895 - categorical_accuracy: 0.9428
58880/60000 [============================>.] - ETA: 1s - loss: 0.1894 - categorical_accuracy: 0.9429
58944/60000 [============================>.] - ETA: 1s - loss: 0.1892 - categorical_accuracy: 0.9429
59008/60000 [============================>.] - ETA: 1s - loss: 0.1894 - categorical_accuracy: 0.9429
59072/60000 [============================>.] - ETA: 1s - loss: 0.1893 - categorical_accuracy: 0.9429
59136/60000 [============================>.] - ETA: 1s - loss: 0.1893 - categorical_accuracy: 0.9430
59200/60000 [============================>.] - ETA: 1s - loss: 0.1891 - categorical_accuracy: 0.9430
59264/60000 [============================>.] - ETA: 1s - loss: 0.1890 - categorical_accuracy: 0.9431
59328/60000 [============================>.] - ETA: 0s - loss: 0.1890 - categorical_accuracy: 0.9430
59392/60000 [============================>.] - ETA: 0s - loss: 0.1890 - categorical_accuracy: 0.9431
59456/60000 [============================>.] - ETA: 0s - loss: 0.1888 - categorical_accuracy: 0.9431
59520/60000 [============================>.] - ETA: 0s - loss: 0.1887 - categorical_accuracy: 0.9431
59552/60000 [============================>.] - ETA: 0s - loss: 0.1886 - categorical_accuracy: 0.9431
59584/60000 [============================>.] - ETA: 0s - loss: 0.1887 - categorical_accuracy: 0.9431
59616/60000 [============================>.] - ETA: 0s - loss: 0.1886 - categorical_accuracy: 0.9432
59680/60000 [============================>.] - ETA: 0s - loss: 0.1885 - categorical_accuracy: 0.9432
59744/60000 [============================>.] - ETA: 0s - loss: 0.1884 - categorical_accuracy: 0.9432
59808/60000 [============================>.] - ETA: 0s - loss: 0.1882 - categorical_accuracy: 0.9433
59872/60000 [============================>.] - ETA: 0s - loss: 0.1881 - categorical_accuracy: 0.9433
59904/60000 [============================>.] - ETA: 0s - loss: 0.1880 - categorical_accuracy: 0.9433
59936/60000 [============================>.] - ETA: 0s - loss: 0.1880 - categorical_accuracy: 0.9433
60000/60000 [==============================] - 92s 2ms/step - loss: 0.1879 - categorical_accuracy: 0.9434 - val_loss: 0.0475 - val_categorical_accuracy: 0.9843

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  224/10000 [..............................] - ETA: 4s 
  416/10000 [>.............................] - ETA: 3s
  608/10000 [>.............................] - ETA: 3s
  800/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 2s
 1184/10000 [==>...........................] - ETA: 2s
 1376/10000 [===>..........................] - ETA: 2s
 1568/10000 [===>..........................] - ETA: 2s
 1760/10000 [====>.........................] - ETA: 2s
 1952/10000 [====>.........................] - ETA: 2s
 2144/10000 [=====>........................] - ETA: 2s
 2336/10000 [======>.......................] - ETA: 2s
 2528/10000 [======>.......................] - ETA: 2s
 2720/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3104/10000 [========>.....................] - ETA: 2s
 3264/10000 [========>.....................] - ETA: 1s
 3456/10000 [=========>....................] - ETA: 1s
 3648/10000 [=========>....................] - ETA: 1s
 3840/10000 [==========>...................] - ETA: 1s
 4032/10000 [===========>..................] - ETA: 1s
 4224/10000 [===========>..................] - ETA: 1s
 4416/10000 [============>.................] - ETA: 1s
 4608/10000 [============>.................] - ETA: 1s
 4800/10000 [=============>................] - ETA: 1s
 4992/10000 [=============>................] - ETA: 1s
 5184/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5568/10000 [===============>..............] - ETA: 1s
 5760/10000 [================>.............] - ETA: 1s
 5952/10000 [================>.............] - ETA: 1s
 6144/10000 [=================>............] - ETA: 1s
 6336/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 0s
 6688/10000 [===================>..........] - ETA: 0s
 6880/10000 [===================>..........] - ETA: 0s
 7040/10000 [====================>.........] - ETA: 0s
 7232/10000 [====================>.........] - ETA: 0s
 7424/10000 [=====================>........] - ETA: 0s
 7616/10000 [=====================>........] - ETA: 0s
 7808/10000 [======================>.......] - ETA: 0s
 8000/10000 [=======================>......] - ETA: 0s
 8224/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8640/10000 [========================>.....] - ETA: 0s
 8832/10000 [=========================>....] - ETA: 0s
 9024/10000 [==========================>...] - ETA: 0s
 9184/10000 [==========================>...] - ETA: 0s
 9376/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9760/10000 [============================>.] - ETA: 0s
 9952/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 285us/step
[[6.16708249e-08 3.96933224e-08 6.67075710e-06 ... 9.99979854e-01
  1.13580036e-07 6.56760585e-06]
 [1.54219833e-07 9.52165556e-06 9.99983072e-01 ... 4.60769201e-08
  9.56533768e-07 1.80976636e-10]
 [1.28401484e-06 9.99636292e-01 5.29552635e-05 ... 4.83183685e-05
  5.13107807e-05 1.11118402e-06]
 ...
 [1.76308657e-08 7.07742061e-07 7.31361993e-08 ... 5.14163412e-06
  1.00034476e-05 1.05477106e-04]
 [5.37386313e-07 2.10952066e-07 3.00467242e-08 ... 3.47405162e-08
  3.56629316e-04 1.08302811e-06]
 [2.99500448e-06 3.55548707e-07 1.16139825e-06 ... 1.28715583e-09
  4.56101020e-07 3.93127042e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.047516555091179906, 'accuracy_test:': 0.9843000173568726}

  ('#### Save   #######################################################',) 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_keras/charcnn/result'}

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 41b74c5] ml_store  && git pull --all
 1 file changed, 1155 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 2b6f562...41b74c5 master -> master (forced update)





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
{'loss': 0.5156298503279686, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  ['model_pars.pkl', 'model.ckpt.index', 'model.ckpt.data-00000-of-00001', 'checkpoint', 'model.ckpt.meta'] 

  #### Load   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/', 'model_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model'}
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model
Loaded saved model from /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/model

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 97feae2] ml_store  && git pull --all
 1 file changed, 111 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   41b74c5..97feae2  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tf//temporal_fusion_google.py", line 17, in <module>
    from mlmodels.mode_tf.raw  import temporal_fusion_google
ModuleNotFoundError: No module named 'mlmodels.mode_tf'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master e1faab1] ml_store  && git pull --all
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   97feae2..e1faab1  master -> master





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
	Data preprocessing and feature engineering runtime = 0.24s ...
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
 40%|████      | 2/5 [00:21<00:31, 10.53s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9142751336494146, 'learning_rate': 0.048530602168068884, 'min_data_in_leaf': 2, 'num_leaves': 63} and reward: 0.3936
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xedA\xbd\xec\xd2?\x9cX\r\x00\x00\x00learning_rateq\x02G?\xa8\xd9\x00\xcaU\x86\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x02X\n\x00\x00\x00num_leavesq\x04K?u.' and reward: 0.3936
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xedA\xbd\xec\xd2?\x9cX\r\x00\x00\x00learning_rateq\x02G?\xa8\xd9\x00\xcaU\x86\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x02X\n\x00\x00\x00num_leavesq\x04K?u.' and reward: 0.3936
 60%|██████    | 3/5 [00:52<00:33, 16.77s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9875616238326328, 'learning_rate': 0.10524707492635525, 'min_data_in_leaf': 27, 'num_leaves': 51} and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x9a\x1a\xd5\xa4\xaa~X\r\x00\x00\x00learning_rateq\x02G?\xba\xf1x\xe8\xce\xf0\x83X\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K3u.' and reward: 0.392
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x9a\x1a\xd5\xa4\xaa~X\r\x00\x00\x00learning_rateq\x02G?\xba\xf1x\xe8\xce\xf0\x83X\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K3u.' and reward: 0.392
 80%|████████  | 4/5 [01:18<00:19, 19.51s/it] 80%|████████  | 4/5 [01:18<00:19, 19.57s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9528603266695382, 'learning_rate': 0.005240332214786216, 'min_data_in_leaf': 24, 'num_leaves': 65} and reward: 0.382
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee}\xd4\xf0\x96s\x0bX\r\x00\x00\x00learning_rateq\x02G?uv\xe2\xf7\xb9\x10\xcfX\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04KAu.' and reward: 0.382
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee}\xd4\xf0\x96s\x0bX\r\x00\x00\x00learning_rateq\x02G?uv\xe2\xf7\xb9\x10\xcfX\x10\x00\x00\x00min_data_in_leafq\x03K\x18X\n\x00\x00\x00num_leavesq\x04KAu.' and reward: 0.382
Time for Gradient Boosting hyperparameter optimization: 110.21336388587952
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9142751336494146, 'learning_rate': 0.048530602168068884, 'min_data_in_leaf': 2, 'num_leaves': 63}
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:50<01:15, 25.29s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.04198110976310466, 'embedding_size_factor': 1.3773067559658871, 'layers.choice': 3, 'learning_rate': 0.00016540557709210005, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.45227236213548e-11} and reward: 0.3658
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xa5~\x8cJ\xf6\xf2\xc1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\tr\xcf\x16\xee\xf4X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?%\xae\x17\x16\x9f\x9fzX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xaf\xef\x91\x0b\xca\xee-u.' and reward: 0.3658
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xa5~\x8cJ\xf6\xf2\xc1X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf6\tr\xcf\x16\xee\xf4X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?%\xae\x17\x16\x9f\x9fzX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xaf\xef\x91\x0b\xca\xee-u.' and reward: 0.3658
 60%|██████    | 3/5 [01:43<01:06, 33.49s/it] 60%|██████    | 3/5 [01:43<01:08, 34.41s/it]
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
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.3276101881648259, 'embedding_size_factor': 0.8580814822645979, 'layers.choice': 1, 'learning_rate': 0.003466786925791166, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 6.9216641018534706e-09} and reward: 0.3844
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\xf7\x90\xb9\x00G\x00X\x15\x00\x00\x00embedding_size_factorq\x03G?\xebugK\xf4&\x1cX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?lfa\x0e\xfd\xbc\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>=\xbas=\xeb\x12\xdbu.' and reward: 0.3844
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd4\xf7\x90\xb9\x00G\x00X\x15\x00\x00\x00embedding_size_factorq\x03G?\xebugK\xf4&\x1cX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?lfa\x0e\xfd\xbc\x01X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>=\xbas=\xeb\x12\xdbu.' and reward: 0.3844
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 171.80647468566895
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -166.12s of remaining time.
Ensemble size: 20
Ensemble weights: 
[0.8  0.05 0.   0.05 0.   0.1  0.  ]
	0.395	 = Validation accuracy score
	1.56s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 287.72s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fb76d92f0f0>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 014e545] ml_store  && git pull --all
 1 file changed, 238 insertions(+)
To github.com:arita37/mlmodels_store.git
 + e6f1869...014e545 master -> master (forced update)





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon//fb_prophet.py", line 160, in <module>
    test(data_path = "model_fb/fbprophet.json", choice="json" )
TypeError: test() got an unexpected keyword argument 'choice'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 1d68a98] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   014e545..1d68a98  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.91it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 2.561 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.249592
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.249592399597168 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff2bdf1c3c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff2bdf1c3c8>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 112.08it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1032.9454752604167,
    "abs_error": 364.15802001953125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.412892767875864,
    "sMAPE": 0.506772095691526,
    "MSIS": 96.51570100905339,
    "QuantileLoss[0.5]": 364.15799713134766,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.139469119144096,
    "NRMSE": 0.6766204025082968,
    "ND": 0.6388737193325109,
    "wQuantileLoss[0.5]": 0.6388736791778029,
    "mean_wQuantileLoss": 0.6388736791778029,
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
100%|██████████| 10/10 [00:01<00:00,  8.12it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.233 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff2b65bc9b0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff2b65bc9b0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 167.39it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:02<00:00,  4.81it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 2.081 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.253882
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.253881645202637 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff291925198>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff291925198>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 163.30it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 271.7430419921875,
    "abs_error": 174.53558349609375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1564640183734838,
    "sMAPE": 0.28823035862033874,
    "MSIS": 46.258557499612294,
    "QuantileLoss[0.5]": 174.53557968139648,
    "Coverage[0.5]": 0.75,
    "RMSE": 16.4846304778781,
    "NRMSE": 0.34704485216585473,
    "ND": 0.30620277806332236,
    "wQuantileLoss[0.5]": 0.30620277137087104,
    "mean_wQuantileLoss": 0.30620277137087104,
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
 30%|███       | 3/10 [00:12<00:29,  4.28s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:24<00:16,  4.14s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:35<00:04,  4.04s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:39<00:00,  3.95s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 39.511 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.869623
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.869623041152954 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff29347f8d0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff29347f8d0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 121.38it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 54572.182291666664,
    "abs_error": 2760.95751953125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.293965984690203,
    "sMAPE": 1.4186309220146134,
    "MSIS": 731.7586135049916,
    "QuantileLoss[0.5]": 2760.957473754883,
    "Coverage[0.5]": 1.0,
    "RMSE": 233.6068969265819,
    "NRMSE": 4.918039935296461,
    "ND": 4.8437851219846495,
    "wQuantileLoss[0.5]": 4.843785041675233,
    "mean_wQuantileLoss": 4.843785041675233,
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
100%|██████████| 10/10 [00:00<00:00, 55.92it/s, avg_epoch_loss=5.06]
INFO:root:Epoch[0] Elapsed time 0.179 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.057436
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.057436132431031 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff27a8140b8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff27a8140b8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 152.75it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 354.3365478515625,
    "abs_error": 205.75613403320312,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.3633298196430914,
    "sMAPE": 0.3341621369864098,
    "MSIS": 54.53319763871425,
    "QuantileLoss[0.5]": 205.75614547729492,
    "Coverage[0.5]": 0.75,
    "RMSE": 18.823829255801343,
    "NRMSE": 0.3962911422273967,
    "ND": 0.36097567374246164,
    "wQuantileLoss[0.5]": 0.36097569381981565,
    "mean_wQuantileLoss": 0.36097569381981565,
    "MAE_Coverage": 0.25
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
100%|██████████| 10/10 [00:01<00:00,  8.09it/s, avg_epoch_loss=144]
INFO:root:Epoch[0] Elapsed time 1.236 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=144.412806
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 144.41280616614463 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff27a8490b8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff27a8490b8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 169.02it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 2434.5670123443265,
    "abs_error": 573.7377910258112,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 3.8015578142395565,
    "sMAPE": 1.9586154749802402,
    "MSIS": 152.0623125695823,
    "QuantileLoss[0.5]": 573.7377910258112,
    "Coverage[0.5]": 0.0,
    "RMSE": 49.34133168393742,
    "NRMSE": 1.0387648775565772,
    "ND": 1.0065575281154584,
    "wQuantileLoss[0.5]": 1.0065575281154584,
    "mean_wQuantileLoss": 1.0065575281154584,
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
 10%|█         | 1/10 [01:54<17:06, 114.00s/it, avg_epoch_loss=0.488] 20%|██        | 2/10 [04:48<17:36, 132.10s/it, avg_epoch_loss=0.47]  30%|███       | 3/10 [07:37<16:42, 143.26s/it, avg_epoch_loss=0.453] 40%|████      | 4/10 [10:26<15:06, 151.02s/it, avg_epoch_loss=0.437] 50%|█████     | 5/10 [13:41<13:40, 164.12s/it, avg_epoch_loss=0.424] 60%|██████    | 6/10 [16:53<11:29, 172.40s/it, avg_epoch_loss=0.414] 70%|███████   | 7/10 [20:03<08:53, 177.74s/it, avg_epoch_loss=0.407] 80%|████████  | 8/10 [23:31<06:14, 187.00s/it, avg_epoch_loss=0.404] 90%|█████████ | 9/10 [26:35<03:06, 186.03s/it, avg_epoch_loss=0.401]100%|██████████| 10/10 [29:46<00:00, 187.42s/it, avg_epoch_loss=0.399]100%|██████████| 10/10 [29:46<00:00, 178.64s/it, avg_epoch_loss=0.399]
INFO:root:Epoch[0] Elapsed time 1786.452 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.398938
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3989381194114685 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7ff27a80f278>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7ff27a80f278>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 21.07it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 137.49435424804688,
    "abs_error": 102.26634979248047,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.6776117021893543,
    "sMAPE": 0.177204681090722,
    "MSIS": 27.104465256662994,
    "QuantileLoss[0.5]": 102.26634216308594,
    "Coverage[0.5]": 0.4166666666666667,
    "RMSE": 11.725798661415217,
    "NRMSE": 0.2468589191876888,
    "ND": 0.17941464875873767,
    "wQuantileLoss[0.5]": 0.17941463537383498,
    "mean_wQuantileLoss": 0.17941463537383498,
    "MAE_Coverage": 0.08333333333333331
}

  #### Plot   ####################################################### 


   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 36fe03b] ml_store  && git pull --all
 1 file changed, 498 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.118.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + ee5f013...36fe03b master -> master (forced update)





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7ff8fd921710> 

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

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master fb16d56] ml_store  && git pull --all
 1 file changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   36fe03b..fb16d56  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 1.58463774e+00  5.71209961e-02 -1.77183179e-02 -7.99547491e-01
   1.32970299e+00 -2.91594596e-01 -1.10771250e+00 -2.58982853e-01
   1.89293198e-01 -1.71939447e+00]
 [ 1.37661405e+00 -6.00225330e-01  7.25916853e-01 -3.79517516e-01
  -6.27546260e-01 -1.01480369e+00  9.66220863e-01  4.35986196e-01
  -6.87487393e-01  3.32107876e+00]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 1.22867367e+00  1.34373116e-01 -1.82420406e-01 -2.68371304e-01
  -1.73963799e+00 -1.31675626e-01 -9.26871939e-01  1.01855247e+00
   1.23055820e+00 -4.91125138e-01]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 8.78643802e-01  1.03703898e+00 -4.77124206e-01  6.72619748e-01
  -1.04948638e+00  2.42887697e+00  5.24750492e-01  1.00568668e+00
   3.53567216e-01 -3.59901817e-02]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f48a5140d68>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f48c61aefd0> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.77528533  1.47016034  1.03298378 -0.87000822  0.78655651  0.36919047
  -0.14319575  0.85328219 -0.13971173 -0.22241403]
 [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
   0.59921324 -0.21465384  1.02124813  0.60640394]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]]
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
[[ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 1.02242019  1.85300949  0.64435367  0.14225137  1.15080755  0.51350548
  -0.45994283  0.37245685 -0.1484898   0.37167029]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 1.02817479 -0.50845713  1.7653351   0.77741921  0.61771419 -0.11877117
   0.45015551 -0.19899818  1.86647138  0.8709698 ]
 [ 1.01195228 -1.88141087  1.70018815  0.4972691  -0.91766462  0.2373327
  -1.09033833 -2.14444405 -0.36956243  0.60878366]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]
 [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
   0.42952614  0.1340482   1.20205486  0.10622272]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]]
None

  ############ Save/ Load ############################################ 

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master acfaae1] ml_store  && git pull --all
 1 file changed, 270 insertions(+)
To github.com:arita37/mlmodels_store.git
   fb16d56..acfaae1  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140347183767504
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140347183767280
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140347183766048
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140347183765600
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140347183765096
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140347183764760

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
grad_step = 000000, loss = 0.809968
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.645265
grad_step = 000002, loss = 0.523239
grad_step = 000003, loss = 0.391148
grad_step = 000004, loss = 0.245964
grad_step = 000005, loss = 0.103957
grad_step = 000006, loss = 0.027190
grad_step = 000007, loss = 0.114983
grad_step = 000008, loss = 0.146354
grad_step = 000009, loss = 0.082156
grad_step = 000010, loss = 0.027237
grad_step = 000011, loss = 0.008978
grad_step = 000012, loss = 0.014956
grad_step = 000013, loss = 0.028283
grad_step = 000014, loss = 0.038355
grad_step = 000015, loss = 0.041259
grad_step = 000016, loss = 0.037295
grad_step = 000017, loss = 0.028915
grad_step = 000018, loss = 0.019827
grad_step = 000019, loss = 0.013548
grad_step = 000020, loss = 0.011895
grad_step = 000021, loss = 0.014103
grad_step = 000022, loss = 0.017493
grad_step = 000023, loss = 0.019193
grad_step = 000024, loss = 0.017684
grad_step = 000025, loss = 0.013758
grad_step = 000026, loss = 0.009693
grad_step = 000027, loss = 0.007309
grad_step = 000028, loss = 0.006962
grad_step = 000029, loss = 0.007893
grad_step = 000030, loss = 0.009025
grad_step = 000031, loss = 0.009582
grad_step = 000032, loss = 0.009330
grad_step = 000033, loss = 0.008517
grad_step = 000034, loss = 0.007571
grad_step = 000035, loss = 0.006872
grad_step = 000036, loss = 0.006577
grad_step = 000037, loss = 0.006609
grad_step = 000038, loss = 0.006766
grad_step = 000039, loss = 0.006883
grad_step = 000040, loss = 0.006863
grad_step = 000041, loss = 0.006689
grad_step = 000042, loss = 0.006391
grad_step = 000043, loss = 0.006046
grad_step = 000044, loss = 0.005711
grad_step = 000045, loss = 0.005436
grad_step = 000046, loss = 0.005251
grad_step = 000047, loss = 0.005162
grad_step = 000048, loss = 0.005157
grad_step = 000049, loss = 0.005208
grad_step = 000050, loss = 0.005264
grad_step = 000051, loss = 0.005256
grad_step = 000052, loss = 0.005155
grad_step = 000053, loss = 0.004995
grad_step = 000054, loss = 0.004853
grad_step = 000055, loss = 0.004796
grad_step = 000056, loss = 0.004819
grad_step = 000057, loss = 0.004868
grad_step = 000058, loss = 0.004878
grad_step = 000059, loss = 0.004817
grad_step = 000060, loss = 0.004704
grad_step = 000061, loss = 0.004592
grad_step = 000062, loss = 0.004526
grad_step = 000063, loss = 0.004512
grad_step = 000064, loss = 0.004520
grad_step = 000065, loss = 0.004507
grad_step = 000066, loss = 0.004454
grad_step = 000067, loss = 0.004380
grad_step = 000068, loss = 0.004316
grad_step = 000069, loss = 0.004286
grad_step = 000070, loss = 0.004285
grad_step = 000071, loss = 0.004287
grad_step = 000072, loss = 0.004271
grad_step = 000073, loss = 0.004231
grad_step = 000074, loss = 0.004181
grad_step = 000075, loss = 0.004137
grad_step = 000076, loss = 0.004106
grad_step = 000077, loss = 0.004083
grad_step = 000078, loss = 0.004060
grad_step = 000079, loss = 0.004033
grad_step = 000080, loss = 0.004004
grad_step = 000081, loss = 0.003975
grad_step = 000082, loss = 0.003944
grad_step = 000083, loss = 0.003911
grad_step = 000084, loss = 0.003880
grad_step = 000085, loss = 0.003854
grad_step = 000086, loss = 0.003831
grad_step = 000087, loss = 0.003806
grad_step = 000088, loss = 0.003775
grad_step = 000089, loss = 0.003741
grad_step = 000090, loss = 0.003710
grad_step = 000091, loss = 0.003682
grad_step = 000092, loss = 0.003652
grad_step = 000093, loss = 0.003616
grad_step = 000094, loss = 0.003580
grad_step = 000095, loss = 0.003546
grad_step = 000096, loss = 0.003513
grad_step = 000097, loss = 0.003477
grad_step = 000098, loss = 0.003441
grad_step = 000099, loss = 0.003407
grad_step = 000100, loss = 0.003369
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003329
grad_step = 000102, loss = 0.003290
grad_step = 000103, loss = 0.003253
grad_step = 000104, loss = 0.003213
grad_step = 000105, loss = 0.003174
grad_step = 000106, loss = 0.003132
grad_step = 000107, loss = 0.003089
grad_step = 000108, loss = 0.003045
grad_step = 000109, loss = 0.003003
grad_step = 000110, loss = 0.002961
grad_step = 000111, loss = 0.002919
grad_step = 000112, loss = 0.002876
grad_step = 000113, loss = 0.002834
grad_step = 000114, loss = 0.002790
grad_step = 000115, loss = 0.002748
grad_step = 000116, loss = 0.002706
grad_step = 000117, loss = 0.002663
grad_step = 000118, loss = 0.002622
grad_step = 000119, loss = 0.002580
grad_step = 000120, loss = 0.002539
grad_step = 000121, loss = 0.002497
grad_step = 000122, loss = 0.002457
grad_step = 000123, loss = 0.002418
grad_step = 000124, loss = 0.002381
grad_step = 000125, loss = 0.002345
grad_step = 000126, loss = 0.002310
grad_step = 000127, loss = 0.002276
grad_step = 000128, loss = 0.002244
grad_step = 000129, loss = 0.002210
grad_step = 000130, loss = 0.002178
grad_step = 000131, loss = 0.002147
grad_step = 000132, loss = 0.002117
grad_step = 000133, loss = 0.002089
grad_step = 000134, loss = 0.002063
grad_step = 000135, loss = 0.002036
grad_step = 000136, loss = 0.002006
grad_step = 000137, loss = 0.001979
grad_step = 000138, loss = 0.001962
grad_step = 000139, loss = 0.001943
grad_step = 000140, loss = 0.001914
grad_step = 000141, loss = 0.001887
grad_step = 000142, loss = 0.001869
grad_step = 000143, loss = 0.001845
grad_step = 000144, loss = 0.001814
grad_step = 000145, loss = 0.001791
grad_step = 000146, loss = 0.001766
grad_step = 000147, loss = 0.001734
grad_step = 000148, loss = 0.001706
grad_step = 000149, loss = 0.001678
grad_step = 000150, loss = 0.001645
grad_step = 000151, loss = 0.001614
grad_step = 000152, loss = 0.001586
grad_step = 000153, loss = 0.001554
grad_step = 000154, loss = 0.001517
grad_step = 000155, loss = 0.001482
grad_step = 000156, loss = 0.001448
grad_step = 000157, loss = 0.001409
grad_step = 000158, loss = 0.001371
grad_step = 000159, loss = 0.001334
grad_step = 000160, loss = 0.001297
grad_step = 000161, loss = 0.001259
grad_step = 000162, loss = 0.001224
grad_step = 000163, loss = 0.001191
grad_step = 000164, loss = 0.001154
grad_step = 000165, loss = 0.001116
grad_step = 000166, loss = 0.001080
grad_step = 000167, loss = 0.001047
grad_step = 000168, loss = 0.001020
grad_step = 000169, loss = 0.001001
grad_step = 000170, loss = 0.000989
grad_step = 000171, loss = 0.000977
grad_step = 000172, loss = 0.000963
grad_step = 000173, loss = 0.000948
grad_step = 000174, loss = 0.000933
grad_step = 000175, loss = 0.000920
grad_step = 000176, loss = 0.000907
grad_step = 000177, loss = 0.000894
grad_step = 000178, loss = 0.000882
grad_step = 000179, loss = 0.000870
grad_step = 000180, loss = 0.000859
grad_step = 000181, loss = 0.000849
grad_step = 000182, loss = 0.000840
grad_step = 000183, loss = 0.000830
grad_step = 000184, loss = 0.000818
grad_step = 000185, loss = 0.000807
grad_step = 000186, loss = 0.000799
grad_step = 000187, loss = 0.000791
grad_step = 000188, loss = 0.000782
grad_step = 000189, loss = 0.000773
grad_step = 000190, loss = 0.000764
grad_step = 000191, loss = 0.000755
grad_step = 000192, loss = 0.000746
grad_step = 000193, loss = 0.000737
grad_step = 000194, loss = 0.000729
grad_step = 000195, loss = 0.000721
grad_step = 000196, loss = 0.000714
grad_step = 000197, loss = 0.000705
grad_step = 000198, loss = 0.000696
grad_step = 000199, loss = 0.000687
grad_step = 000200, loss = 0.000678
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000673
grad_step = 000202, loss = 0.000670
grad_step = 000203, loss = 0.000668
grad_step = 000204, loss = 0.000665
grad_step = 000205, loss = 0.000664
grad_step = 000206, loss = 0.000651
grad_step = 000207, loss = 0.000628
grad_step = 000208, loss = 0.000625
grad_step = 000209, loss = 0.000629
grad_step = 000210, loss = 0.000618
grad_step = 000211, loss = 0.000606
grad_step = 000212, loss = 0.000602
grad_step = 000213, loss = 0.000598
grad_step = 000214, loss = 0.000596
grad_step = 000215, loss = 0.000584
grad_step = 000216, loss = 0.000576
grad_step = 000217, loss = 0.000576
grad_step = 000218, loss = 0.000568
grad_step = 000219, loss = 0.000559
grad_step = 000220, loss = 0.000556
grad_step = 000221, loss = 0.000551
grad_step = 000222, loss = 0.000546
grad_step = 000223, loss = 0.000541
grad_step = 000224, loss = 0.000536
grad_step = 000225, loss = 0.000532
grad_step = 000226, loss = 0.000529
grad_step = 000227, loss = 0.000524
grad_step = 000228, loss = 0.000520
grad_step = 000229, loss = 0.000517
grad_step = 000230, loss = 0.000513
grad_step = 000231, loss = 0.000509
grad_step = 000232, loss = 0.000507
grad_step = 000233, loss = 0.000508
grad_step = 000234, loss = 0.000514
grad_step = 000235, loss = 0.000526
grad_step = 000236, loss = 0.000528
grad_step = 000237, loss = 0.000523
grad_step = 000238, loss = 0.000497
grad_step = 000239, loss = 0.000486
grad_step = 000240, loss = 0.000490
grad_step = 000241, loss = 0.000492
grad_step = 000242, loss = 0.000486
grad_step = 000243, loss = 0.000476
grad_step = 000244, loss = 0.000476
grad_step = 000245, loss = 0.000481
grad_step = 000246, loss = 0.000476
grad_step = 000247, loss = 0.000469
grad_step = 000248, loss = 0.000463
grad_step = 000249, loss = 0.000461
grad_step = 000250, loss = 0.000462
grad_step = 000251, loss = 0.000462
grad_step = 000252, loss = 0.000461
grad_step = 000253, loss = 0.000454
grad_step = 000254, loss = 0.000449
grad_step = 000255, loss = 0.000446
grad_step = 000256, loss = 0.000445
grad_step = 000257, loss = 0.000447
grad_step = 000258, loss = 0.000446
grad_step = 000259, loss = 0.000447
grad_step = 000260, loss = 0.000444
grad_step = 000261, loss = 0.000443
grad_step = 000262, loss = 0.000439
grad_step = 000263, loss = 0.000438
grad_step = 000264, loss = 0.000433
grad_step = 000265, loss = 0.000429
grad_step = 000266, loss = 0.000424
grad_step = 000267, loss = 0.000421
grad_step = 000268, loss = 0.000419
grad_step = 000269, loss = 0.000417
grad_step = 000270, loss = 0.000415
grad_step = 000271, loss = 0.000415
grad_step = 000272, loss = 0.000416
grad_step = 000273, loss = 0.000421
grad_step = 000274, loss = 0.000443
grad_step = 000275, loss = 0.000461
grad_step = 000276, loss = 0.000484
grad_step = 000277, loss = 0.000441
grad_step = 000278, loss = 0.000409
grad_step = 000279, loss = 0.000407
grad_step = 000280, loss = 0.000423
grad_step = 000281, loss = 0.000425
grad_step = 000282, loss = 0.000401
grad_step = 000283, loss = 0.000400
grad_step = 000284, loss = 0.000415
grad_step = 000285, loss = 0.000408
grad_step = 000286, loss = 0.000396
grad_step = 000287, loss = 0.000389
grad_step = 000288, loss = 0.000394
grad_step = 000289, loss = 0.000404
grad_step = 000290, loss = 0.000399
grad_step = 000291, loss = 0.000390
grad_step = 000292, loss = 0.000382
grad_step = 000293, loss = 0.000383
grad_step = 000294, loss = 0.000389
grad_step = 000295, loss = 0.000387
grad_step = 000296, loss = 0.000383
grad_step = 000297, loss = 0.000376
grad_step = 000298, loss = 0.000375
grad_step = 000299, loss = 0.000380
grad_step = 000300, loss = 0.000384
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000386
grad_step = 000302, loss = 0.000377
grad_step = 000303, loss = 0.000370
grad_step = 000304, loss = 0.000366
grad_step = 000305, loss = 0.000367
grad_step = 000306, loss = 0.000373
grad_step = 000307, loss = 0.000372
grad_step = 000308, loss = 0.000371
grad_step = 000309, loss = 0.000363
grad_step = 000310, loss = 0.000359
grad_step = 000311, loss = 0.000357
grad_step = 000312, loss = 0.000358
grad_step = 000313, loss = 0.000361
grad_step = 000314, loss = 0.000362
grad_step = 000315, loss = 0.000365
grad_step = 000316, loss = 0.000358
grad_step = 000317, loss = 0.000354
grad_step = 000318, loss = 0.000350
grad_step = 000319, loss = 0.000348
grad_step = 000320, loss = 0.000349
grad_step = 000321, loss = 0.000351
grad_step = 000322, loss = 0.000358
grad_step = 000323, loss = 0.000357
grad_step = 000324, loss = 0.000356
grad_step = 000325, loss = 0.000348
grad_step = 000326, loss = 0.000343
grad_step = 000327, loss = 0.000340
grad_step = 000328, loss = 0.000338
grad_step = 000329, loss = 0.000338
grad_step = 000330, loss = 0.000339
grad_step = 000331, loss = 0.000344
grad_step = 000332, loss = 0.000347
grad_step = 000333, loss = 0.000350
grad_step = 000334, loss = 0.000344
grad_step = 000335, loss = 0.000340
grad_step = 000336, loss = 0.000332
grad_step = 000337, loss = 0.000329
grad_step = 000338, loss = 0.000330
grad_step = 000339, loss = 0.000334
grad_step = 000340, loss = 0.000342
grad_step = 000341, loss = 0.000345
grad_step = 000342, loss = 0.000345
grad_step = 000343, loss = 0.000335
grad_step = 000344, loss = 0.000326
grad_step = 000345, loss = 0.000322
grad_step = 000346, loss = 0.000324
grad_step = 000347, loss = 0.000327
grad_step = 000348, loss = 0.000325
grad_step = 000349, loss = 0.000322
grad_step = 000350, loss = 0.000317
grad_step = 000351, loss = 0.000315
grad_step = 000352, loss = 0.000314
grad_step = 000353, loss = 0.000313
grad_step = 000354, loss = 0.000312
grad_step = 000355, loss = 0.000315
grad_step = 000356, loss = 0.000322
grad_step = 000357, loss = 0.000331
grad_step = 000358, loss = 0.000346
grad_step = 000359, loss = 0.000346
grad_step = 000360, loss = 0.000337
grad_step = 000361, loss = 0.000313
grad_step = 000362, loss = 0.000306
grad_step = 000363, loss = 0.000319
grad_step = 000364, loss = 0.000327
grad_step = 000365, loss = 0.000330
grad_step = 000366, loss = 0.000312
grad_step = 000367, loss = 0.000300
grad_step = 000368, loss = 0.000301
grad_step = 000369, loss = 0.000309
grad_step = 000370, loss = 0.000314
grad_step = 000371, loss = 0.000305
grad_step = 000372, loss = 0.000296
grad_step = 000373, loss = 0.000293
grad_step = 000374, loss = 0.000296
grad_step = 000375, loss = 0.000298
grad_step = 000376, loss = 0.000296
grad_step = 000377, loss = 0.000296
grad_step = 000378, loss = 0.000294
grad_step = 000379, loss = 0.000293
grad_step = 000380, loss = 0.000289
grad_step = 000381, loss = 0.000288
grad_step = 000382, loss = 0.000285
grad_step = 000383, loss = 0.000282
grad_step = 000384, loss = 0.000281
grad_step = 000385, loss = 0.000282
grad_step = 000386, loss = 0.000286
grad_step = 000387, loss = 0.000294
grad_step = 000388, loss = 0.000311
grad_step = 000389, loss = 0.000314
grad_step = 000390, loss = 0.000312
grad_step = 000391, loss = 0.000286
grad_step = 000392, loss = 0.000275
grad_step = 000393, loss = 0.000284
grad_step = 000394, loss = 0.000289
grad_step = 000395, loss = 0.000288
grad_step = 000396, loss = 0.000276
grad_step = 000397, loss = 0.000269
grad_step = 000398, loss = 0.000272
grad_step = 000399, loss = 0.000278
grad_step = 000400, loss = 0.000284
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000275
grad_step = 000402, loss = 0.000268
grad_step = 000403, loss = 0.000263
grad_step = 000404, loss = 0.000264
grad_step = 000405, loss = 0.000267
grad_step = 000406, loss = 0.000267
grad_step = 000407, loss = 0.000268
grad_step = 000408, loss = 0.000267
grad_step = 000409, loss = 0.000266
grad_step = 000410, loss = 0.000263
grad_step = 000411, loss = 0.000259
grad_step = 000412, loss = 0.000254
grad_step = 000413, loss = 0.000252
grad_step = 000414, loss = 0.000254
grad_step = 000415, loss = 0.000257
grad_step = 000416, loss = 0.000263
grad_step = 000417, loss = 0.000269
grad_step = 000418, loss = 0.000279
grad_step = 000419, loss = 0.000275
grad_step = 000420, loss = 0.000264
grad_step = 000421, loss = 0.000248
grad_step = 000422, loss = 0.000246
grad_step = 000423, loss = 0.000254
grad_step = 000424, loss = 0.000253
grad_step = 000425, loss = 0.000247
grad_step = 000426, loss = 0.000242
grad_step = 000427, loss = 0.000242
grad_step = 000428, loss = 0.000244
grad_step = 000429, loss = 0.000246
grad_step = 000430, loss = 0.000249
grad_step = 000431, loss = 0.000245
grad_step = 000432, loss = 0.000240
grad_step = 000433, loss = 0.000236
grad_step = 000434, loss = 0.000235
grad_step = 000435, loss = 0.000236
grad_step = 000436, loss = 0.000239
grad_step = 000437, loss = 0.000249
grad_step = 000438, loss = 0.000264
grad_step = 000439, loss = 0.000278
grad_step = 000440, loss = 0.000267
grad_step = 000441, loss = 0.000248
grad_step = 000442, loss = 0.000234
grad_step = 000443, loss = 0.000235
grad_step = 000444, loss = 0.000239
grad_step = 000445, loss = 0.000238
grad_step = 000446, loss = 0.000234
grad_step = 000447, loss = 0.000229
grad_step = 000448, loss = 0.000227
grad_step = 000449, loss = 0.000231
grad_step = 000450, loss = 0.000235
grad_step = 000451, loss = 0.000230
grad_step = 000452, loss = 0.000224
grad_step = 000453, loss = 0.000223
grad_step = 000454, loss = 0.000224
grad_step = 000455, loss = 0.000221
grad_step = 000456, loss = 0.000218
grad_step = 000457, loss = 0.000218
grad_step = 000458, loss = 0.000219
grad_step = 000459, loss = 0.000218
grad_step = 000460, loss = 0.000215
grad_step = 000461, loss = 0.000216
grad_step = 000462, loss = 0.000219
grad_step = 000463, loss = 0.000224
grad_step = 000464, loss = 0.000230
grad_step = 000465, loss = 0.000237
grad_step = 000466, loss = 0.000246
grad_step = 000467, loss = 0.000249
grad_step = 000468, loss = 0.000244
grad_step = 000469, loss = 0.000226
grad_step = 000470, loss = 0.000214
grad_step = 000471, loss = 0.000216
grad_step = 000472, loss = 0.000223
grad_step = 000473, loss = 0.000230
grad_step = 000474, loss = 0.000225
grad_step = 000475, loss = 0.000216
grad_step = 000476, loss = 0.000207
grad_step = 000477, loss = 0.000208
grad_step = 000478, loss = 0.000213
grad_step = 000479, loss = 0.000216
grad_step = 000480, loss = 0.000219
grad_step = 000481, loss = 0.000220
grad_step = 000482, loss = 0.000219
grad_step = 000483, loss = 0.000213
grad_step = 000484, loss = 0.000206
grad_step = 000485, loss = 0.000203
grad_step = 000486, loss = 0.000203
grad_step = 000487, loss = 0.000203
grad_step = 000488, loss = 0.000203
grad_step = 000489, loss = 0.000205
grad_step = 000490, loss = 0.000208
grad_step = 000491, loss = 0.000210
grad_step = 000492, loss = 0.000210
grad_step = 000493, loss = 0.000210
grad_step = 000494, loss = 0.000209
grad_step = 000495, loss = 0.000206
grad_step = 000496, loss = 0.000202
grad_step = 000497, loss = 0.000199
grad_step = 000498, loss = 0.000197
grad_step = 000499, loss = 0.000196
grad_step = 000500, loss = 0.000194
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000194
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
[[0.85332626 0.8271703  0.93550503 0.942009   1.0161345 ]
 [0.84712446 0.9025303  0.93097675 1.0115128  0.9900996 ]
 [0.8823037  0.89380395 1.0086468  0.9806017  0.9568375 ]
 [0.9221758  0.9763727  1.0065277  0.9399937  0.91839474]
 [0.99031705 0.99587226 0.9486424  0.9085857  0.8503476 ]
 [0.98261094 0.96009046 0.9194281  0.8572098  0.86445844]
 [0.93199176 0.9006581  0.8536247  0.8539782  0.8200536 ]
 [0.90348494 0.8387072  0.8550192  0.82188165 0.8387761 ]
 [0.8181293  0.8314078  0.8122394  0.8386062  0.8453915 ]
 [0.8292981  0.80644745 0.83857614 0.84515566 0.84383714]
 [0.80180097 0.809098   0.8525707  0.821515   0.91790193]
 [0.817418   0.84844863 0.80865365 0.93082833 0.9432301 ]
 [0.84562373 0.8244934  0.92793703 0.94405866 1.0139366 ]
 [0.8470322  0.90848225 0.9381386  1.0125167  0.98169184]
 [0.8909048  0.91465735 1.012104   0.98057777 0.9365642 ]
 [0.9390764  0.9929769  0.99208814 0.92746466 0.8967142 ]
 [1.0016108  0.97888803 0.9324134  0.8972064  0.8376419 ]
 [0.9828359  0.9357977  0.8956764  0.8341501  0.8497636 ]
 [0.9276098  0.88875407 0.8383027  0.84197074 0.81715596]
 [0.9082252  0.83540636 0.84808326 0.8180194  0.84160155]
 [0.8322001  0.83632946 0.8118073  0.8386901  0.8520452 ]
 [0.84584796 0.81275713 0.8441859  0.8491998  0.84999955]
 [0.81387925 0.8183117  0.8604902  0.82807624 0.92246366]
 [0.8294331  0.85490173 0.8141086  0.9314058  0.9454683 ]
 [0.85859275 0.83224046 0.9366397  0.94625217 1.0209298 ]
 [0.85711396 0.9095887  0.93602866 1.0212848  1.0038073 ]
 [0.8912022  0.9020204  1.0157447  0.9941408  0.9700048 ]
 [0.9322609  0.9829759  1.0210665  0.9533749  0.9316087 ]
 [0.9957434  1.0065362  0.9604852  0.9186051  0.85654426]
 [0.9930508  0.9677556  0.93019307 0.8642398  0.87359786]
 [0.93781954 0.9075228  0.85973376 0.8572729  0.82640064]]

  #### Plot     ############################################### 
Saved image to ztest/model_tch/nbeats//n_beats_test.png.

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 54aa332] ml_store  && git pull --all
 1 file changed, 1121 insertions(+)
To github.com:arita37/mlmodels_store.git
   acfaae1..54aa332  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 522, in <module>
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_classifier.py", line 418, in get_params
    cf = json.load(open(data_path, mode='r'))
FileNotFoundError: [Errno 2] No such file or directory: 'model_tch/transformer_classifier.json'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 4f3e2a9] ml_store  && git pull --all
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   54aa332..4f3e2a9  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 31930777.06B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 332138.56B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4021248/440473133 [00:00<00:10, 40210792.69B/s]  2%|▏         | 9601024/440473133 [00:00<00:09, 43888135.99B/s]  3%|▎         | 15357952/440473133 [00:00<00:08, 47256059.40B/s]  5%|▍         | 21132288/440473133 [00:00<00:08, 49977927.16B/s]  6%|▌         | 26381312/440473133 [00:00<00:08, 50549961.12B/s]  7%|▋         | 32019456/440473133 [00:00<00:07, 52168115.39B/s]  9%|▊         | 37922816/440473133 [00:00<00:07, 54050819.41B/s] 10%|▉         | 43842560/440473133 [00:00<00:07, 55497521.54B/s] 11%|█▏        | 49766400/440473133 [00:00<00:06, 56567301.38B/s] 13%|█▎        | 55672832/440473133 [00:01<00:06, 57293413.01B/s] 14%|█▍        | 61550592/440473133 [00:01<00:06, 57726294.17B/s] 15%|█▌        | 67436544/440473133 [00:01<00:06, 58059662.63B/s] 17%|█▋        | 73393152/440473133 [00:01<00:06, 58502761.87B/s] 18%|█▊        | 79333376/440473133 [00:01<00:06, 58763469.59B/s] 19%|█▉        | 85192704/440473133 [00:01<00:06, 58395543.43B/s] 21%|██        | 91021312/440473133 [00:01<00:06, 57754387.17B/s] 22%|██▏       | 96790528/440473133 [00:01<00:05, 57613734.52B/s] 23%|██▎       | 102548480/440473133 [00:01<00:05, 56962092.17B/s] 25%|██▍       | 108243968/440473133 [00:01<00:05, 55423483.67B/s] 26%|██▌       | 113795072/440473133 [00:02<00:05, 54878775.77B/s] 27%|██▋       | 119290880/440473133 [00:02<00:06, 53402125.96B/s] 28%|██▊       | 125241344/440473133 [00:02<00:05, 55094923.67B/s] 30%|██▉       | 131209216/440473133 [00:02<00:05, 56393293.58B/s] 31%|███       | 137209856/440473133 [00:02<00:05, 57429535.38B/s] 32%|███▏      | 142974976/440473133 [00:02<00:05, 57142876.09B/s] 34%|███▍      | 149006336/440473133 [00:02<00:05, 58057944.48B/s] 35%|███▌      | 155099136/440473133 [00:02<00:04, 58889894.32B/s] 37%|███▋      | 161001472/440473133 [00:02<00:04, 57797237.07B/s] 38%|███▊      | 166795264/440473133 [00:02<00:04, 57029039.25B/s] 39%|███▉      | 172510208/440473133 [00:03<00:04, 56314499.57B/s] 40%|████      | 178374656/440473133 [00:03<00:04, 56992448.44B/s] 42%|████▏     | 184180736/440473133 [00:03<00:04, 57222043.37B/s] 43%|████▎     | 190033920/440473133 [00:03<00:04, 57608436.89B/s] 44%|████▍     | 195999744/440473133 [00:03<00:04, 58207515.57B/s] 46%|████▌     | 201826304/440473133 [00:03<00:04, 57485360.57B/s] 47%|████▋     | 207838208/440473133 [00:03<00:03, 58248547.31B/s] 49%|████▊     | 213842944/440473133 [00:03<00:03, 58775924.69B/s] 50%|████▉     | 219905024/440473133 [00:03<00:03, 59315927.23B/s] 51%|█████▏    | 225916928/440473133 [00:03<00:03, 59551071.56B/s] 53%|█████▎    | 231881728/440473133 [00:04<00:03, 59578601.67B/s] 54%|█████▍    | 237868032/440473133 [00:04<00:03, 59658732.58B/s] 55%|█████▌    | 243862528/440473133 [00:04<00:03, 59742541.72B/s] 57%|█████▋    | 249871360/440473133 [00:04<00:03, 59844594.74B/s] 58%|█████▊    | 255905792/440473133 [00:04<00:03, 59992207.71B/s] 59%|█████▉    | 261931008/440473133 [00:04<00:02, 60068561.17B/s] 61%|██████    | 267938816/440473133 [00:04<00:02, 58910502.58B/s] 62%|██████▏   | 273913856/440473133 [00:04<00:02, 59154008.18B/s] 64%|██████▎   | 279928832/440473133 [00:04<00:02, 59448515.11B/s] 65%|██████▍   | 285999104/440473133 [00:04<00:02, 59819022.49B/s] 66%|██████▋   | 292044800/440473133 [00:05<00:02, 60008363.64B/s] 68%|██████▊   | 298048512/440473133 [00:05<00:02, 59754882.63B/s] 69%|██████▉   | 304030720/440473133 [00:05<00:02, 59775000.35B/s] 70%|███████   | 310009856/440473133 [00:05<00:02, 59761163.15B/s] 72%|███████▏  | 316187648/440473133 [00:05<00:02, 60352160.10B/s] 73%|███████▎  | 322225152/440473133 [00:05<00:01, 60217846.61B/s] 75%|███████▍  | 328249344/440473133 [00:05<00:01, 60191469.39B/s] 76%|███████▌  | 334300160/440473133 [00:05<00:01, 60286016.24B/s] 77%|███████▋  | 340330496/440473133 [00:05<00:01, 60112066.35B/s] 79%|███████▊  | 346402816/440473133 [00:05<00:01, 60292929.07B/s] 80%|████████  | 352549888/440473133 [00:06<00:01, 60638650.33B/s] 81%|████████▏ | 358615040/440473133 [00:06<00:01, 60416900.38B/s] 83%|████████▎ | 364657664/440473133 [00:06<00:01, 60413195.11B/s] 84%|████████▍ | 370700288/440473133 [00:06<00:01, 60403043.01B/s] 86%|████████▌ | 376776704/440473133 [00:06<00:01, 60510771.46B/s] 87%|████████▋ | 382828544/440473133 [00:06<00:00, 59225157.78B/s] 88%|████████▊ | 388757504/440473133 [00:06<00:00, 58235461.24B/s] 90%|████████▉ | 394725376/440473133 [00:06<00:00, 58658350.20B/s] 91%|█████████ | 400788480/440473133 [00:06<00:00, 59232050.64B/s] 92%|█████████▏| 406812672/440473133 [00:06<00:00, 59529730.07B/s] 94%|█████████▎| 412863488/440473133 [00:07<00:00, 59818583.71B/s] 95%|█████████▌| 418849792/440473133 [00:07<00:00, 59727789.27B/s] 96%|█████████▋| 424934400/440473133 [00:07<00:00, 60056409.37B/s] 98%|█████████▊| 431013888/440473133 [00:07<00:00, 60275692.34B/s] 99%|█████████▉| 437068800/440473133 [00:07<00:00, 60353949.59B/s]100%|██████████| 440473133/440473133 [00:07<00:00, 58453433.44B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
  65536/7094233 [..............................] - ETA: 5s
 344064/7094233 [>.............................] - ETA: 2s
1343488/7094233 [====>.........................] - ETA: 0s
2105344/7094233 [=======>......................] - ETA: 0s
3145728/7094233 [============>.................] - ETA: 0s
4186112/7094233 [================>.............] - ETA: 0s
5226496/7094233 [=====================>........] - ETA: 0s
6012928/7094233 [========================>.....] - ETA: 0s
7053312/7094233 [============================>.] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:  12%|█▏        | 252/2118 [00:00<00:00, 2514.72it/s]Processing text_left with encode:  32%|███▏      | 676/2118 [00:00<00:00, 2864.14it/s]Processing text_left with encode:  47%|████▋     | 1001/2118 [00:00<00:00, 2968.47it/s]Processing text_left with encode:  69%|██████▉   | 1459/2118 [00:00<00:00, 3317.70it/s]Processing text_left with encode:  90%|█████████ | 1916/2118 [00:00<00:00, 3614.42it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3875.93it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:00<00:13, 1402.39it/s]Processing text_right with encode:   2%|▏         | 322/18841 [00:00<00:12, 1484.37it/s]Processing text_right with encode:   3%|▎         | 492/18841 [00:00<00:11, 1541.67it/s]Processing text_right with encode:   4%|▎         | 668/18841 [00:00<00:11, 1596.79it/s]Processing text_right with encode:   4%|▍         | 832/18841 [00:00<00:11, 1607.04it/s]Processing text_right with encode:   5%|▌         | 995/18841 [00:00<00:11, 1611.85it/s]Processing text_right with encode:   6%|▌         | 1155/18841 [00:00<00:11, 1607.70it/s]Processing text_right with encode:   7%|▋         | 1314/18841 [00:00<00:10, 1601.79it/s]Processing text_right with encode:   8%|▊         | 1469/18841 [00:00<00:10, 1585.53it/s]Processing text_right with encode:   9%|▊         | 1632/18841 [00:01<00:10, 1591.80it/s]Processing text_right with encode:  10%|▉         | 1798/18841 [00:01<00:10, 1608.03it/s]Processing text_right with encode:  10%|█         | 1958/18841 [00:01<00:10, 1602.60it/s]Processing text_right with encode:  11%|█▏        | 2124/18841 [00:01<00:10, 1616.54it/s]Processing text_right with encode:  12%|█▏        | 2298/18841 [00:01<00:10, 1650.94it/s]Processing text_right with encode:  13%|█▎        | 2469/18841 [00:01<00:09, 1667.45it/s]Processing text_right with encode:  14%|█▍        | 2638/18841 [00:01<00:09, 1673.09it/s]Processing text_right with encode:  15%|█▍        | 2824/18841 [00:01<00:09, 1724.15it/s]Processing text_right with encode:  16%|█▌        | 2997/18841 [00:01<00:09, 1667.91it/s]Processing text_right with encode:  17%|█▋        | 3165/18841 [00:01<00:09, 1648.03it/s]Processing text_right with encode:  18%|█▊        | 3331/18841 [00:02<00:09, 1627.99it/s]Processing text_right with encode:  19%|█▊        | 3495/18841 [00:02<00:09, 1625.19it/s]Processing text_right with encode:  19%|█▉        | 3659/18841 [00:02<00:09, 1629.59it/s]Processing text_right with encode:  20%|██        | 3825/18841 [00:02<00:09, 1636.52it/s]Processing text_right with encode:  21%|██        | 4002/18841 [00:02<00:08, 1672.69it/s]Processing text_right with encode:  22%|██▏       | 4170/18841 [00:02<00:08, 1660.96it/s]Processing text_right with encode:  23%|██▎       | 4337/18841 [00:02<00:08, 1658.77it/s]Processing text_right with encode:  24%|██▍       | 4504/18841 [00:02<00:08, 1639.59it/s]Processing text_right with encode:  25%|██▍       | 4672/18841 [00:02<00:08, 1650.03it/s]Processing text_right with encode:  26%|██▌       | 4850/18841 [00:02<00:08, 1685.84it/s]Processing text_right with encode:  27%|██▋       | 5022/18841 [00:03<00:08, 1693.78it/s]Processing text_right with encode:  28%|██▊       | 5202/18841 [00:03<00:07, 1722.81it/s]Processing text_right with encode:  29%|██▊       | 5382/18841 [00:03<00:07, 1743.66it/s]Processing text_right with encode:  29%|██▉       | 5557/18841 [00:03<00:07, 1723.45it/s]Processing text_right with encode:  30%|███       | 5730/18841 [00:03<00:07, 1714.82it/s]Processing text_right with encode:  31%|███▏      | 5902/18841 [00:03<00:07, 1703.03it/s]Processing text_right with encode:  32%|███▏      | 6073/18841 [00:03<00:07, 1680.40it/s]Processing text_right with encode:  33%|███▎      | 6242/18841 [00:03<00:07, 1660.01it/s]Processing text_right with encode:  34%|███▍      | 6415/18841 [00:03<00:07, 1677.10it/s]Processing text_right with encode:  35%|███▌      | 6595/18841 [00:03<00:07, 1710.70it/s]Processing text_right with encode:  36%|███▌      | 6767/18841 [00:04<00:07, 1697.00it/s]Processing text_right with encode:  37%|███▋      | 6937/18841 [00:04<00:07, 1684.48it/s]Processing text_right with encode:  38%|███▊      | 7106/18841 [00:04<00:07, 1654.00it/s]Processing text_right with encode:  39%|███▊      | 7286/18841 [00:04<00:06, 1694.24it/s]Processing text_right with encode:  40%|███▉      | 7464/18841 [00:04<00:06, 1718.70it/s]Processing text_right with encode:  41%|████      | 7637/18841 [00:04<00:06, 1699.73it/s]Processing text_right with encode:  41%|████▏     | 7808/18841 [00:04<00:06, 1698.16it/s]Processing text_right with encode:  42%|████▏     | 7979/18841 [00:04<00:06, 1665.19it/s]Processing text_right with encode:  43%|████▎     | 8146/18841 [00:04<00:06, 1664.78it/s]Processing text_right with encode:  44%|████▍     | 8316/18841 [00:04<00:06, 1673.24it/s]Processing text_right with encode:  45%|████▌     | 8484/18841 [00:05<00:06, 1666.44it/s]Processing text_right with encode:  46%|████▌     | 8651/18841 [00:05<00:06, 1657.23it/s]Processing text_right with encode:  47%|████▋     | 8817/18841 [00:05<00:06, 1654.44it/s]Processing text_right with encode:  48%|████▊     | 8983/18841 [00:05<00:06, 1618.78it/s]Processing text_right with encode:  49%|████▊     | 9157/18841 [00:05<00:05, 1651.96it/s]Processing text_right with encode:  49%|████▉     | 9324/18841 [00:05<00:05, 1654.74it/s]Processing text_right with encode:  50%|█████     | 9491/18841 [00:05<00:05, 1657.93it/s]Processing text_right with encode:  51%|█████▏    | 9662/18841 [00:05<00:05, 1671.58it/s]Processing text_right with encode:  52%|█████▏    | 9830/18841 [00:05<00:05, 1653.90it/s]Processing text_right with encode:  53%|█████▎    | 10002/18841 [00:06<00:05, 1672.29it/s]Processing text_right with encode:  54%|█████▍    | 10170/18841 [00:06<00:05, 1659.31it/s]Processing text_right with encode:  55%|█████▍    | 10337/18841 [00:06<00:05, 1644.72it/s]Processing text_right with encode:  56%|█████▌    | 10526/18841 [00:06<00:04, 1709.28it/s]Processing text_right with encode:  57%|█████▋    | 10698/18841 [00:06<00:04, 1695.97it/s]Processing text_right with encode:  58%|█████▊    | 10869/18841 [00:06<00:04, 1690.27it/s]Processing text_right with encode:  59%|█████▊    | 11039/18841 [00:06<00:04, 1683.08it/s]Processing text_right with encode:  59%|█████▉    | 11208/18841 [00:06<00:04, 1666.55it/s]Processing text_right with encode:  60%|██████    | 11375/18841 [00:06<00:04, 1635.25it/s]Processing text_right with encode:  61%|██████    | 11539/18841 [00:06<00:04, 1615.25it/s]Processing text_right with encode:  62%|██████▏   | 11705/18841 [00:07<00:04, 1626.35it/s]Processing text_right with encode:  63%|██████▎   | 11875/18841 [00:07<00:04, 1645.20it/s]Processing text_right with encode:  64%|██████▍   | 12040/18841 [00:07<00:04, 1641.82it/s]Processing text_right with encode:  65%|██████▍   | 12205/18841 [00:07<00:04, 1553.41it/s]Processing text_right with encode:  66%|██████▌   | 12380/18841 [00:07<00:04, 1606.04it/s]Processing text_right with encode:  67%|██████▋   | 12546/18841 [00:07<00:03, 1620.81it/s]Processing text_right with encode:  67%|██████▋   | 12709/18841 [00:07<00:03, 1621.35it/s]Processing text_right with encode:  68%|██████▊   | 12875/18841 [00:07<00:03, 1631.45it/s]Processing text_right with encode:  69%|██████▉   | 13043/18841 [00:07<00:03, 1644.94it/s]Processing text_right with encode:  70%|███████   | 13208/18841 [00:07<00:03, 1646.28it/s]Processing text_right with encode:  71%|███████   | 13381/18841 [00:08<00:03, 1669.10it/s]Processing text_right with encode:  72%|███████▏  | 13554/18841 [00:08<00:03, 1684.46it/s]Processing text_right with encode:  73%|███████▎  | 13740/18841 [00:08<00:02, 1732.80it/s]Processing text_right with encode:  74%|███████▍  | 13914/18841 [00:08<00:02, 1701.97it/s]Processing text_right with encode:  75%|███████▍  | 14087/18841 [00:08<00:02, 1708.04it/s]Processing text_right with encode:  76%|███████▌  | 14259/18841 [00:08<00:02, 1679.88it/s]Processing text_right with encode:  77%|███████▋  | 14430/18841 [00:08<00:02, 1686.84it/s]Processing text_right with encode:  77%|███████▋  | 14601/18841 [00:08<00:02, 1693.59it/s]Processing text_right with encode:  78%|███████▊  | 14783/18841 [00:08<00:02, 1727.02it/s]Processing text_right with encode:  79%|███████▉  | 14956/18841 [00:08<00:02, 1699.08it/s]Processing text_right with encode:  80%|████████  | 15127/18841 [00:09<00:02, 1698.18it/s]Processing text_right with encode:  81%|████████  | 15298/18841 [00:09<00:02, 1690.30it/s]Processing text_right with encode:  82%|████████▏ | 15468/18841 [00:09<00:02, 1676.30it/s]Processing text_right with encode:  83%|████████▎ | 15644/18841 [00:09<00:01, 1699.33it/s]Processing text_right with encode:  84%|████████▍ | 15815/18841 [00:09<00:01, 1654.77it/s]Processing text_right with encode:  85%|████████▍ | 15981/18841 [00:09<00:01, 1631.85it/s]Processing text_right with encode:  86%|████████▌ | 16145/18841 [00:09<00:01, 1615.53it/s]Processing text_right with encode:  87%|████████▋ | 16307/18841 [00:09<00:01, 1609.13it/s]Processing text_right with encode:  87%|████████▋ | 16475/18841 [00:09<00:01, 1629.02it/s]Processing text_right with encode:  88%|████████▊ | 16639/18841 [00:10<00:01, 1615.62it/s]Processing text_right with encode:  89%|████████▉ | 16809/18841 [00:10<00:01, 1638.12it/s]Processing text_right with encode:  90%|█████████ | 16974/18841 [00:10<00:01, 1622.48it/s]Processing text_right with encode:  91%|█████████ | 17140/18841 [00:10<00:01, 1631.59it/s]Processing text_right with encode:  92%|█████████▏| 17304/18841 [00:10<00:00, 1629.10it/s]Processing text_right with encode:  93%|█████████▎| 17471/18841 [00:10<00:00, 1641.10it/s]Processing text_right with encode:  94%|█████████▎| 17642/18841 [00:10<00:00, 1660.52it/s]Processing text_right with encode:  95%|█████████▍| 17816/18841 [00:10<00:00, 1683.11it/s]Processing text_right with encode:  95%|█████████▌| 17992/18841 [00:10<00:00, 1701.27it/s]Processing text_right with encode:  96%|█████████▋| 18177/18841 [00:10<00:00, 1741.02it/s]Processing text_right with encode:  97%|█████████▋| 18352/18841 [00:11<00:00, 1671.37it/s]Processing text_right with encode:  98%|█████████▊| 18531/18841 [00:11<00:00, 1703.63it/s]Processing text_right with encode:  99%|█████████▉| 18703/18841 [00:11<00:00, 1576.77it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1659.99it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 556055.07it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 753005.18it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  71%|███████   | 451/633 [00:00<00:00, 4509.40it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4447.88it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 166/5961 [00:00<00:03, 1655.21it/s]Processing text_right with encode:   6%|▌         | 330/5961 [00:00<00:03, 1649.68it/s]Processing text_right with encode:   8%|▊         | 483/5961 [00:00<00:03, 1611.44it/s]Processing text_right with encode:  11%|█         | 642/5961 [00:00<00:03, 1604.21it/s]Processing text_right with encode:  13%|█▎        | 803/5961 [00:00<00:03, 1605.50it/s]Processing text_right with encode:  16%|█▋        | 974/5961 [00:00<00:03, 1633.30it/s]Processing text_right with encode:  19%|█▉        | 1139/5961 [00:00<00:02, 1636.34it/s]Processing text_right with encode:  22%|██▏       | 1314/5961 [00:00<00:02, 1667.36it/s]Processing text_right with encode:  25%|██▍       | 1483/5961 [00:00<00:02, 1671.77it/s]Processing text_right with encode:  28%|██▊       | 1644/5961 [00:01<00:02, 1651.08it/s]Processing text_right with encode:  30%|███       | 1807/5961 [00:01<00:02, 1643.84it/s]Processing text_right with encode:  33%|███▎      | 1973/5961 [00:01<00:02, 1648.55it/s]Processing text_right with encode:  36%|███▌      | 2150/5961 [00:01<00:02, 1683.10it/s]Processing text_right with encode:  39%|███▉      | 2317/5961 [00:01<00:02, 1655.05it/s]Processing text_right with encode:  42%|████▏     | 2482/5961 [00:01<00:02, 1629.39it/s]Processing text_right with encode:  45%|████▍     | 2654/5961 [00:01<00:01, 1655.55it/s]Processing text_right with encode:  48%|████▊     | 2845/5961 [00:01<00:01, 1722.53it/s]Processing text_right with encode:  51%|█████     | 3018/5961 [00:01<00:01, 1707.27it/s]Processing text_right with encode:  54%|█████▎    | 3197/5961 [00:01<00:01, 1730.08it/s]Processing text_right with encode:  57%|█████▋    | 3371/5961 [00:02<00:01, 1659.38it/s]Processing text_right with encode:  59%|█████▉    | 3540/5961 [00:02<00:01, 1667.22it/s]Processing text_right with encode:  62%|██████▏   | 3708/5961 [00:02<00:01, 1669.75it/s]Processing text_right with encode:  65%|██████▌   | 3876/5961 [00:02<00:01, 1665.05it/s]Processing text_right with encode:  68%|██████▊   | 4054/5961 [00:02<00:01, 1695.81it/s]Processing text_right with encode:  71%|███████   | 4224/5961 [00:02<00:01, 1679.82it/s]Processing text_right with encode:  74%|███████▎  | 4396/5961 [00:02<00:00, 1690.62it/s]Processing text_right with encode:  77%|███████▋  | 4569/5961 [00:02<00:00, 1701.55it/s]Processing text_right with encode:  80%|███████▉  | 4740/5961 [00:02<00:00, 1672.31it/s]Processing text_right with encode:  82%|████████▏ | 4908/5961 [00:02<00:00, 1626.18it/s]Processing text_right with encode:  85%|████████▌ | 5072/5961 [00:03<00:00, 1625.38it/s]Processing text_right with encode:  88%|████████▊ | 5246/5961 [00:03<00:00, 1658.09it/s]Processing text_right with encode:  91%|█████████ | 5413/5961 [00:03<00:00, 1623.69it/s]Processing text_right with encode:  94%|█████████▎| 5576/5961 [00:03<00:00, 1574.87it/s]Processing text_right with encode:  96%|█████████▌| 5735/5961 [00:03<00:00, 1531.87it/s]Processing text_right with encode:  99%|█████████▉| 5915/5961 [00:03<00:00, 1602.86it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1652.98it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 484576.46it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 725438.74it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:17<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:17<?, ?it/s, loss=1.132]Epoch 1/1:   1%|          | 1/102 [00:17<28:52, 17.15s/it, loss=1.132]Epoch 1/1:   1%|          | 1/102 [02:16<28:52, 17.15s/it, loss=1.132]Epoch 1/1:   1%|          | 1/102 [02:16<28:52, 17.15s/it, loss=1.144]Epoch 1/1:   2%|▏         | 2/102 [02:16<1:19:49, 47.90s/it, loss=1.144]Killed

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master a450bf7] ml_store  && git pull --all
 1 file changed, 65 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 1483ba3...a450bf7 master -> master (forced update)





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'dataset/vision/MNIST/', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/MNIST/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/MNIST/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d81adce18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d81adce18>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d81adce18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:03, 154869.17it/s] 92%|█████████▏| 9109504/9912422 [00:00<00:03, 221080.29it/s]9920512it [00:00, 46303266.37it/s]                           
0it [00:00, ?it/s]32768it [00:00, 568061.01it/s]
0it [00:00, ?it/s]1654784it [00:00, 34782640.17it/s]
0it [00:00, ?it/s]8192it [00:00, 88577.94it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  #### Model init, fit   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d80e15b70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d80e15b70>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d80e15b70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0030018726785977683 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.02250635242462158 	 Accuracy: 2
model saves at 2 accuracy
Train Epoch: 2 	 Loss: 0.0024697184065977733 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.02894674253463745 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 3 	 Loss: 0.001892967700958252 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.009029149547219277 	 Accuracy: 7
model saves at 7 accuracy
Train Epoch: 4 	 Loss: 0.0019765630265076955 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.007644044563174248 	 Accuracy: 7
Train Epoch: 5 	 Loss: 0.0016128014226754507 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.01077912414073944 	 Accuracy: 7

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f7d80e15950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f7d80e15950>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f7d80e15950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/MNIST/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7f7d818917b8>

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels.preprocess.generic::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/pytorch_GAN_zoo/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/pytorch_GAN_zoo/'} 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### Predict   ##################################################### 
img_01.png

  #### metrics   ##################################################### 

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/pytorch_GAN_zoo//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 
img_01.png
torch_model

Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/facebookresearch_pytorch_GAN_zoo_hub
<__main__.Model object at 0x7f7d805b26d8>

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 0a85536] ml_store  && git pull --all
 1 file changed, 156 insertions(+)
To github.com:arita37/mlmodels_store.git
   a450bf7..0a85536  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
    from dataloader import DataLoader
ModuleNotFoundError: No module named 'dataloader'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master b9425e2] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   0a85536..b9425e2  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py", line 487, in <module>
    test(pars_choice="test01", data_path= "model_tch/transformer_sentence.json", config_mode="test")
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py", line 438, in test
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
TypeError: 'NoneType' object is not iterable

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
[master 1c3a9bc] ml_store  && git pull --all
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   b9425e2..1c3a9bc  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
    "beta_vae": md.model.beta_vae,
AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'

   cd /home/runner/work/mlmodels/mlmodels_store/ ;            pip3 freeze > deps.txt ;            ls ;            git config --local user.email "noelkev0@gmail.com" && git config --local user.name "arita37"         ;            git add --all &&  git commit -m "ml_store  && git pull --all"  ;            git push --all -f ;            cd /home/runner/work/mlmodels/mlmodels/ ;         
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
