
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '3aee4395159545a95b0d7c8ed6830ec48eff1164', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/3aee4395159545a95b0d7c8ed6830ec48eff1164

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/3aee4395159545a95b0d7c8ed6830ec48eff1164

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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
Already up to date.
[master 1f6abbe] ml_store
 2 files changed, 78 insertions(+), 10501 deletions(-)
 rewrite log_testall/log_testall.py (99%)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   2a49ecc..1f6abbe  master -> master





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
[master fda39cd] ml_store
 1 file changed, 50 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   1f6abbe..fda39cd  master -> master





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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         12          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-23 04:15:13.758879: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 04:15:13.776074: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-23 04:15:13.776286: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cd668868c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 04:15:13.776326: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 168
Trainable params: 168
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2516 - binary_crossentropy: 0.7227 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.2506405212103269}

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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         12          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
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
Total params: 168
Trainable params: 168
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
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
Total params: 413
Trainable params: 413
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.3109 - binary_crossentropy: 2.7665500/500 [==============================] - 1s 2ms/sample - loss: 0.3043 - binary_crossentropy: 2.3355 - val_loss: 0.3352 - val_binary_crossentropy: 2.6912

  #### metrics   #################################################### 
{'MSE': 0.31965993263927633}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
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
Total params: 413
Trainable params: 413
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 587
Trainable params: 587
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2501 - binary_crossentropy: 0.6932 - val_loss: 0.2498 - val_binary_crossentropy: 0.6928

  #### metrics   #################################################### 
{'MSE': 0.24982362230508792}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 587
Trainable params: 587
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
Total params: 413
Trainable params: 413
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2567 - binary_crossentropy: 0.7068500/500 [==============================] - 1s 3ms/sample - loss: 0.2520 - binary_crossentropy: 0.6971 - val_loss: 0.2508 - val_binary_crossentropy: 0.6947

  #### metrics   #################################################### 
{'MSE': 0.24992522327630157}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
Total params: 413
Trainable params: 413
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 163
Trainable params: 163
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2577 - binary_crossentropy: 0.7087500/500 [==============================] - 2s 3ms/sample - loss: 0.2552 - binary_crossentropy: 0.7036 - val_loss: 0.2502 - val_binary_crossentropy: 0.6935

  #### metrics   #################################################### 
{'MSE': 0.25224794334704986}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 163
Trainable params: 163
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-23 04:16:41.513929: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:16:41.516005: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:16:41.522237: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-23 04:16:41.533222: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-23 04:16:41.535103: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 04:16:41.536856: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:16:41.538561: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2514 - val_binary_crossentropy: 0.6959
2020-05-23 04:16:42.919222: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:16:42.921300: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:16:42.926570: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-23 04:16:42.936254: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-23 04:16:42.937780: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 04:16:42.939165: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:16:42.940534: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2517038849884088}

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
2020-05-23 04:17:08.136863: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:08.138434: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:08.142884: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-23 04:17:08.150372: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-23 04:17:08.151715: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 04:17:08.153102: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:08.154315: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2506 - val_binary_crossentropy: 0.6944
2020-05-23 04:17:09.783908: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:09.785313: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:09.787932: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-23 04:17:09.792827: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-23 04:17:09.793730: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 04:17:09.794787: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:09.795558: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25073242217765035}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-23 04:17:46.215917: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:46.221206: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:46.237449: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-23 04:17:46.265500: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-23 04:17:46.269805: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 04:17:46.274978: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:46.279359: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.2776 - binary_crossentropy: 0.7484 - val_loss: 0.2697 - val_binary_crossentropy: 0.7342
2020-05-23 04:17:48.709437: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:48.714905: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:48.728006: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-23 04:17:48.756844: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-23 04:17:48.761039: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 04:17:48.766270: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 04:17:48.771904: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.31649948159791713}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 655
Trainable params: 655
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2687 - binary_crossentropy: 0.7354500/500 [==============================] - 5s 9ms/sample - loss: 0.2786 - binary_crossentropy: 0.7563 - val_loss: 0.2818 - val_binary_crossentropy: 0.7631

  #### metrics   #################################################### 
{'MSE': 0.27946968844655146}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 655
Trainable params: 655
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         2           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         10          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         18          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_5[0][0]           
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
Total params: 206
Trainable params: 206
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.4800 - binary_crossentropy: 7.3688500/500 [==============================] - 5s 10ms/sample - loss: 0.4960 - binary_crossentropy: 7.6104 - val_loss: 0.4880 - val_binary_crossentropy: 7.5274

  #### metrics   #################################################### 
{'MSE': 0.5}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         2           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         10          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         18          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_5[0][0]           
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
Total params: 206
Trainable params: 206
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,924
Trainable params: 1,924
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.5400 - binary_crossentropy: 8.3260500/500 [==============================] - 5s 10ms/sample - loss: 0.5020 - binary_crossentropy: 7.7388 - val_loss: 0.5060 - val_binary_crossentropy: 7.8050

  #### metrics   #################################################### 
{'MSE': 0.499}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 1,924
Trainable params: 1,924
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
regionsequence_mean (InputLayer [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 9)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         5           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 9s - loss: 0.2812 - binary_crossentropy: 0.7601500/500 [==============================] - 6s 12ms/sample - loss: 0.2777 - binary_crossentropy: 0.7529 - val_loss: 0.2811 - val_binary_crossentropy: 0.7601

  #### metrics   #################################################### 
{'MSE': 0.27813267300164624}

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
regionsequence_mean (InputLayer [(None, 1)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 9)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         6           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 1, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         5           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 7s - loss: 0.3010 - binary_crossentropy: 1.0682500/500 [==============================] - 6s 12ms/sample - loss: 0.2882 - binary_crossentropy: 0.9846 - val_loss: 0.2660 - val_binary_crossentropy: 0.9615

  #### metrics   #################################################### 
{'MSE': 0.2740805539439767}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         28          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_10[0][0]                    
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
100/500 [=====>........................] - ETA: 9s - loss: 0.2817 - binary_crossentropy: 0.7584500/500 [==============================] - 7s 13ms/sample - loss: 0.2604 - binary_crossentropy: 0.7149 - val_loss: 0.2630 - val_binary_crossentropy: 0.7198

  #### metrics   #################################################### 
{'MSE': 0.2606525159725297}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         28          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_10[0][0]                    
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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   fda39cd..c2b1e51  master     -> origin/master
Updating fda39cd..c2b1e51
Fast-forward
 .../20200523/list_log_pullrequest_20200523.md      |   2 +-
 error_list/20200523/list_log_testall_20200523.md   | 736 +--------------------
 2 files changed, 6 insertions(+), 732 deletions(-)
[master 546a5c8] ml_store
 1 file changed, 4955 insertions(+)
To github.com:arita37/mlmodels_store.git
   c2b1e51..546a5c8  master -> master





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
[master 589d41c] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   546a5c8..589d41c  master -> master





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
[master 04e2fac] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   589d41c..04e2fac  master -> master





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
[master 599c07f] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   04e2fac..599c07f  master -> master





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
[master 81c821b] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   599c07f..81c821b  master -> master





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
[master 91c48ea] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   81c821b..91c48ea  master -> master





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
[master 7d94d54] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   91c48ea..7d94d54  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3825664/17464789 [=====>........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
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
2020-05-23 04:27:58.958155: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 04:27:58.962832: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-23 04:27:58.962995: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5569b5eff400 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 04:27:58.963013: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 14s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6130 - accuracy: 0.5035
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.5797 - accuracy: 0.5057 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.6398 - accuracy: 0.5017
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6114 - accuracy: 0.5036
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6973 - accuracy: 0.4980
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6951 - accuracy: 0.4981
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6915 - accuracy: 0.4984
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6615 - accuracy: 0.5003
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6390 - accuracy: 0.5018
11000/25000 [============>.................] - ETA: 4s - loss: 7.6624 - accuracy: 0.5003
12000/25000 [=============>................] - ETA: 4s - loss: 7.7177 - accuracy: 0.4967
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7209 - accuracy: 0.4965
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6907 - accuracy: 0.4984
15000/25000 [=================>............] - ETA: 3s - loss: 7.6860 - accuracy: 0.4987
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6762 - accuracy: 0.4994
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6729 - accuracy: 0.4996
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6697 - accuracy: 0.4998
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6593 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6708 - accuracy: 0.4997
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6726 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6596 - accuracy: 0.5005
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fe8a75ce1d0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fe8aabde518> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6130 - accuracy: 0.5035
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5286 - accuracy: 0.5090 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6942 - accuracy: 0.4982
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6590 - accuracy: 0.5005
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6557 - accuracy: 0.5007
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6628 - accuracy: 0.5002
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6257 - accuracy: 0.5027
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5961 - accuracy: 0.5046
11000/25000 [============>.................] - ETA: 4s - loss: 7.5677 - accuracy: 0.5065
12000/25000 [=============>................] - ETA: 4s - loss: 7.6142 - accuracy: 0.5034
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6206 - accuracy: 0.5030
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6360 - accuracy: 0.5020
15000/25000 [=================>............] - ETA: 3s - loss: 7.6441 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6570 - accuracy: 0.5006
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6369 - accuracy: 0.5019
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6308 - accuracy: 0.5023
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6319 - accuracy: 0.5023
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6306 - accuracy: 0.5023
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6528 - accuracy: 0.5009
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6652 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6740 - accuracy: 0.4995
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 10s 398us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.3906 - accuracy: 0.5180
 2000/25000 [=>............................] - ETA: 9s - loss: 7.3983 - accuracy: 0.5175 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6257 - accuracy: 0.5027
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6168 - accuracy: 0.5033
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6605 - accuracy: 0.5004
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6998 - accuracy: 0.4978
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7017 - accuracy: 0.4977
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7165 - accuracy: 0.4967
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6905 - accuracy: 0.4984
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6605 - accuracy: 0.5004
11000/25000 [============>.................] - ETA: 4s - loss: 7.7001 - accuracy: 0.4978
12000/25000 [=============>................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6666 - accuracy: 0.5000
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6644 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 3s - loss: 7.6697 - accuracy: 0.4998
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6637 - accuracy: 0.5002
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6802 - accuracy: 0.4991
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6734 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6884 - accuracy: 0.4986
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6774 - accuracy: 0.4993
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6601 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6569 - accuracy: 0.5006
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6473 - accuracy: 0.5013
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6705 - accuracy: 0.4997
25000/25000 [==============================] - 10s 404us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   7d94d54..f508179  master     -> origin/master
Updating 7d94d54..f508179
Fast-forward
 .../20200523/list_log_pullrequest_20200523.md      |  2 +-
 error_list/20200523/list_log_testall_20200523.md   | 99 ++++++++++++++++++++++
 2 files changed, 100 insertions(+), 1 deletion(-)
[master e54be46] ml_store
 1 file changed, 323 insertions(+)
To github.com:arita37/mlmodels_store.git
   f508179..e54be46  master -> master





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

13/13 [==============================] - 2s 135ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 92a3cf0] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   e54be46..92a3cf0  master -> master





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
 1310720/11490434 [==>...........................] - ETA: 0s
 4177920/11490434 [=========>....................] - ETA: 0s
 8904704/11490434 [======================>.......] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:45 - loss: 2.3298 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:54 - loss: 2.2803 - categorical_accuracy: 0.1875
   96/60000 [..............................] - ETA: 3:56 - loss: 2.2465 - categorical_accuracy: 0.1771
  128/60000 [..............................] - ETA: 3:26 - loss: 2.2309 - categorical_accuracy: 0.1875
  160/60000 [..............................] - ETA: 3:08 - loss: 2.2280 - categorical_accuracy: 0.1937
  192/60000 [..............................] - ETA: 2:56 - loss: 2.2131 - categorical_accuracy: 0.2083
  224/60000 [..............................] - ETA: 2:47 - loss: 2.1993 - categorical_accuracy: 0.2098
  256/60000 [..............................] - ETA: 2:40 - loss: 2.1707 - categorical_accuracy: 0.2344
  288/60000 [..............................] - ETA: 2:35 - loss: 2.1448 - categorical_accuracy: 0.2500
  320/60000 [..............................] - ETA: 2:31 - loss: 2.1050 - categorical_accuracy: 0.2625
  352/60000 [..............................] - ETA: 2:28 - loss: 2.0779 - categorical_accuracy: 0.2727
  384/60000 [..............................] - ETA: 2:26 - loss: 2.0358 - categorical_accuracy: 0.2943
  416/60000 [..............................] - ETA: 2:24 - loss: 1.9930 - categorical_accuracy: 0.3125
  448/60000 [..............................] - ETA: 2:22 - loss: 1.9516 - categorical_accuracy: 0.3281
  480/60000 [..............................] - ETA: 2:20 - loss: 1.9237 - categorical_accuracy: 0.3417
  512/60000 [..............................] - ETA: 2:18 - loss: 1.9101 - categorical_accuracy: 0.3457
  544/60000 [..............................] - ETA: 2:16 - loss: 1.8763 - categorical_accuracy: 0.3621
  576/60000 [..............................] - ETA: 2:15 - loss: 1.8505 - categorical_accuracy: 0.3733
  608/60000 [..............................] - ETA: 2:13 - loss: 1.8157 - categorical_accuracy: 0.3882
  640/60000 [..............................] - ETA: 2:12 - loss: 1.7713 - categorical_accuracy: 0.4016
  672/60000 [..............................] - ETA: 2:11 - loss: 1.7440 - categorical_accuracy: 0.4122
  704/60000 [..............................] - ETA: 2:11 - loss: 1.7386 - categorical_accuracy: 0.4176
  736/60000 [..............................] - ETA: 2:09 - loss: 1.7102 - categorical_accuracy: 0.4321
  768/60000 [..............................] - ETA: 2:09 - loss: 1.6731 - categorical_accuracy: 0.4453
  800/60000 [..............................] - ETA: 2:08 - loss: 1.6474 - categorical_accuracy: 0.4525
  832/60000 [..............................] - ETA: 2:08 - loss: 1.6320 - categorical_accuracy: 0.4615
  864/60000 [..............................] - ETA: 2:07 - loss: 1.6193 - categorical_accuracy: 0.4641
  896/60000 [..............................] - ETA: 2:07 - loss: 1.6024 - categorical_accuracy: 0.4699
  928/60000 [..............................] - ETA: 2:06 - loss: 1.5724 - categorical_accuracy: 0.4806
  960/60000 [..............................] - ETA: 2:06 - loss: 1.5484 - categorical_accuracy: 0.4865
  992/60000 [..............................] - ETA: 2:05 - loss: 1.5185 - categorical_accuracy: 0.4950
 1024/60000 [..............................] - ETA: 2:05 - loss: 1.4984 - categorical_accuracy: 0.5010
 1056/60000 [..............................] - ETA: 2:06 - loss: 1.4735 - categorical_accuracy: 0.5104
 1088/60000 [..............................] - ETA: 2:06 - loss: 1.4484 - categorical_accuracy: 0.5175
 1120/60000 [..............................] - ETA: 2:06 - loss: 1.4306 - categorical_accuracy: 0.5250
 1152/60000 [..............................] - ETA: 2:06 - loss: 1.4167 - categorical_accuracy: 0.5304
 1184/60000 [..............................] - ETA: 2:06 - loss: 1.3968 - categorical_accuracy: 0.5397
 1216/60000 [..............................] - ETA: 2:05 - loss: 1.3839 - categorical_accuracy: 0.5436
 1248/60000 [..............................] - ETA: 2:04 - loss: 1.3699 - categorical_accuracy: 0.5505
 1280/60000 [..............................] - ETA: 2:04 - loss: 1.3545 - categorical_accuracy: 0.5547
 1312/60000 [..............................] - ETA: 2:05 - loss: 1.3386 - categorical_accuracy: 0.5587
 1344/60000 [..............................] - ETA: 2:05 - loss: 1.3197 - categorical_accuracy: 0.5655
 1376/60000 [..............................] - ETA: 2:05 - loss: 1.3051 - categorical_accuracy: 0.5712
 1408/60000 [..............................] - ETA: 2:05 - loss: 1.2943 - categorical_accuracy: 0.5767
 1440/60000 [..............................] - ETA: 2:04 - loss: 1.2815 - categorical_accuracy: 0.5806
 1472/60000 [..............................] - ETA: 2:04 - loss: 1.2690 - categorical_accuracy: 0.5849
 1504/60000 [..............................] - ETA: 2:04 - loss: 1.2582 - categorical_accuracy: 0.5884
 1536/60000 [..............................] - ETA: 2:03 - loss: 1.2446 - categorical_accuracy: 0.5944
 1568/60000 [..............................] - ETA: 2:03 - loss: 1.2316 - categorical_accuracy: 0.5995
 1600/60000 [..............................] - ETA: 2:03 - loss: 1.2244 - categorical_accuracy: 0.6025
 1632/60000 [..............................] - ETA: 2:03 - loss: 1.2135 - categorical_accuracy: 0.6060
 1664/60000 [..............................] - ETA: 2:03 - loss: 1.1998 - categorical_accuracy: 0.6118
 1696/60000 [..............................] - ETA: 2:03 - loss: 1.1859 - categorical_accuracy: 0.6156
 1728/60000 [..............................] - ETA: 2:03 - loss: 1.1811 - categorical_accuracy: 0.6186
 1760/60000 [..............................] - ETA: 2:02 - loss: 1.1733 - categorical_accuracy: 0.6222
 1792/60000 [..............................] - ETA: 2:02 - loss: 1.1666 - categorical_accuracy: 0.6250
 1824/60000 [..............................] - ETA: 2:02 - loss: 1.1570 - categorical_accuracy: 0.6272
 1856/60000 [..............................] - ETA: 2:02 - loss: 1.1473 - categorical_accuracy: 0.6298
 1888/60000 [..............................] - ETA: 2:02 - loss: 1.1387 - categorical_accuracy: 0.6319
 1920/60000 [..............................] - ETA: 2:01 - loss: 1.1354 - categorical_accuracy: 0.6328
 1952/60000 [..............................] - ETA: 2:01 - loss: 1.1264 - categorical_accuracy: 0.6358
 1984/60000 [..............................] - ETA: 2:01 - loss: 1.1149 - categorical_accuracy: 0.6401
 2016/60000 [>.............................] - ETA: 2:01 - loss: 1.1168 - categorical_accuracy: 0.6399
 2048/60000 [>.............................] - ETA: 2:00 - loss: 1.1120 - categorical_accuracy: 0.6411
 2080/60000 [>.............................] - ETA: 2:00 - loss: 1.1027 - categorical_accuracy: 0.6442
 2112/60000 [>.............................] - ETA: 2:00 - loss: 1.0941 - categorical_accuracy: 0.6473
 2144/60000 [>.............................] - ETA: 2:00 - loss: 1.0828 - categorical_accuracy: 0.6511
 2176/60000 [>.............................] - ETA: 2:00 - loss: 1.0721 - categorical_accuracy: 0.6544
 2208/60000 [>.............................] - ETA: 2:00 - loss: 1.0608 - categorical_accuracy: 0.6581
 2240/60000 [>.............................] - ETA: 1:59 - loss: 1.0501 - categorical_accuracy: 0.6612
 2272/60000 [>.............................] - ETA: 1:59 - loss: 1.0409 - categorical_accuracy: 0.6642
 2304/60000 [>.............................] - ETA: 1:59 - loss: 1.0295 - categorical_accuracy: 0.6675
 2336/60000 [>.............................] - ETA: 1:59 - loss: 1.0199 - categorical_accuracy: 0.6708
 2368/60000 [>.............................] - ETA: 1:59 - loss: 1.0109 - categorical_accuracy: 0.6727
 2400/60000 [>.............................] - ETA: 1:58 - loss: 1.0015 - categorical_accuracy: 0.6758
 2432/60000 [>.............................] - ETA: 1:58 - loss: 0.9930 - categorical_accuracy: 0.6785
 2464/60000 [>.............................] - ETA: 1:58 - loss: 0.9871 - categorical_accuracy: 0.6814
 2496/60000 [>.............................] - ETA: 1:58 - loss: 0.9786 - categorical_accuracy: 0.6843
 2528/60000 [>.............................] - ETA: 1:58 - loss: 0.9708 - categorical_accuracy: 0.6871
 2560/60000 [>.............................] - ETA: 1:58 - loss: 0.9640 - categorical_accuracy: 0.6891
 2592/60000 [>.............................] - ETA: 1:58 - loss: 0.9556 - categorical_accuracy: 0.6917
 2624/60000 [>.............................] - ETA: 1:58 - loss: 0.9476 - categorical_accuracy: 0.6947
 2656/60000 [>.............................] - ETA: 1:58 - loss: 0.9448 - categorical_accuracy: 0.6954
 2688/60000 [>.............................] - ETA: 1:57 - loss: 0.9379 - categorical_accuracy: 0.6968
 2720/60000 [>.............................] - ETA: 1:57 - loss: 0.9312 - categorical_accuracy: 0.6989
 2752/60000 [>.............................] - ETA: 1:57 - loss: 0.9250 - categorical_accuracy: 0.7009
 2784/60000 [>.............................] - ETA: 1:57 - loss: 0.9180 - categorical_accuracy: 0.7037
 2816/60000 [>.............................] - ETA: 1:57 - loss: 0.9161 - categorical_accuracy: 0.7045
 2848/60000 [>.............................] - ETA: 1:57 - loss: 0.9084 - categorical_accuracy: 0.7068
 2880/60000 [>.............................] - ETA: 1:57 - loss: 0.9008 - categorical_accuracy: 0.7094
 2912/60000 [>.............................] - ETA: 1:57 - loss: 0.8958 - categorical_accuracy: 0.7112
 2944/60000 [>.............................] - ETA: 1:57 - loss: 0.8927 - categorical_accuracy: 0.7126
 2976/60000 [>.............................] - ETA: 1:56 - loss: 0.8890 - categorical_accuracy: 0.7140
 3008/60000 [>.............................] - ETA: 1:56 - loss: 0.8852 - categorical_accuracy: 0.7158
 3040/60000 [>.............................] - ETA: 1:56 - loss: 0.8822 - categorical_accuracy: 0.7168
 3072/60000 [>.............................] - ETA: 1:56 - loss: 0.8778 - categorical_accuracy: 0.7184
 3104/60000 [>.............................] - ETA: 1:56 - loss: 0.8732 - categorical_accuracy: 0.7200
 3136/60000 [>.............................] - ETA: 1:56 - loss: 0.8662 - categorical_accuracy: 0.7229
 3168/60000 [>.............................] - ETA: 1:55 - loss: 0.8606 - categorical_accuracy: 0.7244
 3200/60000 [>.............................] - ETA: 1:55 - loss: 0.8536 - categorical_accuracy: 0.7269
 3232/60000 [>.............................] - ETA: 1:55 - loss: 0.8483 - categorical_accuracy: 0.7287
 3264/60000 [>.............................] - ETA: 1:55 - loss: 0.8441 - categorical_accuracy: 0.7295
 3296/60000 [>.............................] - ETA: 1:55 - loss: 0.8415 - categorical_accuracy: 0.7300
 3328/60000 [>.............................] - ETA: 1:55 - loss: 0.8410 - categorical_accuracy: 0.7311
 3360/60000 [>.............................] - ETA: 1:55 - loss: 0.8384 - categorical_accuracy: 0.7324
 3392/60000 [>.............................] - ETA: 1:55 - loss: 0.8345 - categorical_accuracy: 0.7338
 3424/60000 [>.............................] - ETA: 1:55 - loss: 0.8332 - categorical_accuracy: 0.7342
 3456/60000 [>.............................] - ETA: 1:55 - loss: 0.8289 - categorical_accuracy: 0.7358
 3488/60000 [>.............................] - ETA: 1:54 - loss: 0.8231 - categorical_accuracy: 0.7377
 3520/60000 [>.............................] - ETA: 1:54 - loss: 0.8171 - categorical_accuracy: 0.7401
 3552/60000 [>.............................] - ETA: 1:54 - loss: 0.8125 - categorical_accuracy: 0.7416
 3584/60000 [>.............................] - ETA: 1:54 - loss: 0.8093 - categorical_accuracy: 0.7430
 3616/60000 [>.............................] - ETA: 1:54 - loss: 0.8041 - categorical_accuracy: 0.7450
 3648/60000 [>.............................] - ETA: 1:54 - loss: 0.7996 - categorical_accuracy: 0.7467
 3680/60000 [>.............................] - ETA: 1:54 - loss: 0.7957 - categorical_accuracy: 0.7478
 3712/60000 [>.............................] - ETA: 1:54 - loss: 0.7926 - categorical_accuracy: 0.7484
 3744/60000 [>.............................] - ETA: 1:54 - loss: 0.7940 - categorical_accuracy: 0.7489
 3776/60000 [>.............................] - ETA: 1:54 - loss: 0.7890 - categorical_accuracy: 0.7508
 3808/60000 [>.............................] - ETA: 1:54 - loss: 0.7858 - categorical_accuracy: 0.7518
 3840/60000 [>.............................] - ETA: 1:53 - loss: 0.7810 - categorical_accuracy: 0.7531
 3872/60000 [>.............................] - ETA: 1:53 - loss: 0.7769 - categorical_accuracy: 0.7546
 3904/60000 [>.............................] - ETA: 1:53 - loss: 0.7751 - categorical_accuracy: 0.7546
 3936/60000 [>.............................] - ETA: 1:53 - loss: 0.7727 - categorical_accuracy: 0.7556
 3968/60000 [>.............................] - ETA: 1:53 - loss: 0.7674 - categorical_accuracy: 0.7576
 4000/60000 [=>............................] - ETA: 1:53 - loss: 0.7639 - categorical_accuracy: 0.7588
 4032/60000 [=>............................] - ETA: 1:53 - loss: 0.7595 - categorical_accuracy: 0.7604
 4064/60000 [=>............................] - ETA: 1:53 - loss: 0.7568 - categorical_accuracy: 0.7613
 4096/60000 [=>............................] - ETA: 1:52 - loss: 0.7525 - categorical_accuracy: 0.7627
 4128/60000 [=>............................] - ETA: 1:52 - loss: 0.7506 - categorical_accuracy: 0.7631
 4160/60000 [=>............................] - ETA: 1:52 - loss: 0.7479 - categorical_accuracy: 0.7639
 4192/60000 [=>............................] - ETA: 1:52 - loss: 0.7443 - categorical_accuracy: 0.7650
 4224/60000 [=>............................] - ETA: 1:52 - loss: 0.7419 - categorical_accuracy: 0.7656
 4256/60000 [=>............................] - ETA: 1:52 - loss: 0.7376 - categorical_accuracy: 0.7669
 4288/60000 [=>............................] - ETA: 1:52 - loss: 0.7358 - categorical_accuracy: 0.7675
 4320/60000 [=>............................] - ETA: 1:52 - loss: 0.7324 - categorical_accuracy: 0.7690
 4352/60000 [=>............................] - ETA: 1:52 - loss: 0.7284 - categorical_accuracy: 0.7700
 4384/60000 [=>............................] - ETA: 1:52 - loss: 0.7260 - categorical_accuracy: 0.7705
 4416/60000 [=>............................] - ETA: 1:52 - loss: 0.7234 - categorical_accuracy: 0.7713
 4448/60000 [=>............................] - ETA: 1:52 - loss: 0.7198 - categorical_accuracy: 0.7725
 4480/60000 [=>............................] - ETA: 1:52 - loss: 0.7168 - categorical_accuracy: 0.7732
 4512/60000 [=>............................] - ETA: 1:52 - loss: 0.7131 - categorical_accuracy: 0.7742
 4544/60000 [=>............................] - ETA: 1:51 - loss: 0.7090 - categorical_accuracy: 0.7755
 4576/60000 [=>............................] - ETA: 1:51 - loss: 0.7057 - categorical_accuracy: 0.7767
 4608/60000 [=>............................] - ETA: 1:51 - loss: 0.7023 - categorical_accuracy: 0.7778
 4640/60000 [=>............................] - ETA: 1:51 - loss: 0.7017 - categorical_accuracy: 0.7782
 4672/60000 [=>............................] - ETA: 1:51 - loss: 0.6991 - categorical_accuracy: 0.7789
 4704/60000 [=>............................] - ETA: 1:51 - loss: 0.6959 - categorical_accuracy: 0.7800
 4736/60000 [=>............................] - ETA: 1:51 - loss: 0.6954 - categorical_accuracy: 0.7802
 4768/60000 [=>............................] - ETA: 1:51 - loss: 0.6918 - categorical_accuracy: 0.7810
 4800/60000 [=>............................] - ETA: 1:50 - loss: 0.6886 - categorical_accuracy: 0.7821
 4832/60000 [=>............................] - ETA: 1:50 - loss: 0.6847 - categorical_accuracy: 0.7835
 4864/60000 [=>............................] - ETA: 1:50 - loss: 0.6813 - categorical_accuracy: 0.7845
 4896/60000 [=>............................] - ETA: 1:50 - loss: 0.6790 - categorical_accuracy: 0.7851
 4928/60000 [=>............................] - ETA: 1:50 - loss: 0.6758 - categorical_accuracy: 0.7859
 4960/60000 [=>............................] - ETA: 1:50 - loss: 0.6723 - categorical_accuracy: 0.7873
 4992/60000 [=>............................] - ETA: 1:50 - loss: 0.6691 - categorical_accuracy: 0.7885
 5024/60000 [=>............................] - ETA: 1:50 - loss: 0.6655 - categorical_accuracy: 0.7896
 5056/60000 [=>............................] - ETA: 1:50 - loss: 0.6635 - categorical_accuracy: 0.7903
 5088/60000 [=>............................] - ETA: 1:50 - loss: 0.6600 - categorical_accuracy: 0.7915
 5120/60000 [=>............................] - ETA: 1:50 - loss: 0.6581 - categorical_accuracy: 0.7918
 5152/60000 [=>............................] - ETA: 1:49 - loss: 0.6548 - categorical_accuracy: 0.7929
 5184/60000 [=>............................] - ETA: 1:49 - loss: 0.6529 - categorical_accuracy: 0.7934
 5216/60000 [=>............................] - ETA: 1:49 - loss: 0.6524 - categorical_accuracy: 0.7939
 5248/60000 [=>............................] - ETA: 1:49 - loss: 0.6493 - categorical_accuracy: 0.7950
 5280/60000 [=>............................] - ETA: 1:49 - loss: 0.6465 - categorical_accuracy: 0.7956
 5312/60000 [=>............................] - ETA: 1:49 - loss: 0.6442 - categorical_accuracy: 0.7963
 5344/60000 [=>............................] - ETA: 1:49 - loss: 0.6413 - categorical_accuracy: 0.7972
 5376/60000 [=>............................] - ETA: 1:49 - loss: 0.6392 - categorical_accuracy: 0.7978
 5408/60000 [=>............................] - ETA: 1:49 - loss: 0.6361 - categorical_accuracy: 0.7988
 5440/60000 [=>............................] - ETA: 1:49 - loss: 0.6339 - categorical_accuracy: 0.7994
 5472/60000 [=>............................] - ETA: 1:49 - loss: 0.6315 - categorical_accuracy: 0.8001
 5504/60000 [=>............................] - ETA: 1:49 - loss: 0.6296 - categorical_accuracy: 0.8007
 5536/60000 [=>............................] - ETA: 1:49 - loss: 0.6268 - categorical_accuracy: 0.8015
 5568/60000 [=>............................] - ETA: 1:48 - loss: 0.6245 - categorical_accuracy: 0.8021
 5600/60000 [=>............................] - ETA: 1:48 - loss: 0.6238 - categorical_accuracy: 0.8021
 5632/60000 [=>............................] - ETA: 1:48 - loss: 0.6214 - categorical_accuracy: 0.8029
 5664/60000 [=>............................] - ETA: 1:48 - loss: 0.6190 - categorical_accuracy: 0.8037
 5696/60000 [=>............................] - ETA: 1:48 - loss: 0.6165 - categorical_accuracy: 0.8046
 5728/60000 [=>............................] - ETA: 1:48 - loss: 0.6142 - categorical_accuracy: 0.8053
 5760/60000 [=>............................] - ETA: 1:48 - loss: 0.6138 - categorical_accuracy: 0.8050
 5792/60000 [=>............................] - ETA: 1:48 - loss: 0.6130 - categorical_accuracy: 0.8051
 5824/60000 [=>............................] - ETA: 1:48 - loss: 0.6111 - categorical_accuracy: 0.8056
 5856/60000 [=>............................] - ETA: 1:48 - loss: 0.6083 - categorical_accuracy: 0.8065
 5888/60000 [=>............................] - ETA: 1:48 - loss: 0.6064 - categorical_accuracy: 0.8071
 5920/60000 [=>............................] - ETA: 1:48 - loss: 0.6042 - categorical_accuracy: 0.8076
 5952/60000 [=>............................] - ETA: 1:48 - loss: 0.6029 - categorical_accuracy: 0.8078
 5984/60000 [=>............................] - ETA: 1:47 - loss: 0.6012 - categorical_accuracy: 0.8085
 6016/60000 [==>...........................] - ETA: 1:47 - loss: 0.5989 - categorical_accuracy: 0.8092
 6048/60000 [==>...........................] - ETA: 1:47 - loss: 0.5976 - categorical_accuracy: 0.8097
 6080/60000 [==>...........................] - ETA: 1:47 - loss: 0.5970 - categorical_accuracy: 0.8099
 6112/60000 [==>...........................] - ETA: 1:47 - loss: 0.5943 - categorical_accuracy: 0.8109
 6144/60000 [==>...........................] - ETA: 1:47 - loss: 0.5920 - categorical_accuracy: 0.8115
 6176/60000 [==>...........................] - ETA: 1:47 - loss: 0.5904 - categorical_accuracy: 0.8120
 6208/60000 [==>...........................] - ETA: 1:47 - loss: 0.5878 - categorical_accuracy: 0.8128
 6240/60000 [==>...........................] - ETA: 1:47 - loss: 0.5857 - categorical_accuracy: 0.8135
 6272/60000 [==>...........................] - ETA: 1:47 - loss: 0.5837 - categorical_accuracy: 0.8143
 6304/60000 [==>...........................] - ETA: 1:47 - loss: 0.5820 - categorical_accuracy: 0.8149
 6336/60000 [==>...........................] - ETA: 1:47 - loss: 0.5823 - categorical_accuracy: 0.8152
 6368/60000 [==>...........................] - ETA: 1:47 - loss: 0.5817 - categorical_accuracy: 0.8153
 6400/60000 [==>...........................] - ETA: 1:46 - loss: 0.5811 - categorical_accuracy: 0.8155
 6432/60000 [==>...........................] - ETA: 1:46 - loss: 0.5788 - categorical_accuracy: 0.8164
 6464/60000 [==>...........................] - ETA: 1:46 - loss: 0.5764 - categorical_accuracy: 0.8173
 6496/60000 [==>...........................] - ETA: 1:46 - loss: 0.5766 - categorical_accuracy: 0.8174
 6528/60000 [==>...........................] - ETA: 1:46 - loss: 0.5749 - categorical_accuracy: 0.8182
 6560/60000 [==>...........................] - ETA: 1:46 - loss: 0.5724 - categorical_accuracy: 0.8191
 6592/60000 [==>...........................] - ETA: 1:46 - loss: 0.5711 - categorical_accuracy: 0.8195
 6624/60000 [==>...........................] - ETA: 1:46 - loss: 0.5700 - categorical_accuracy: 0.8199
 6656/60000 [==>...........................] - ETA: 1:46 - loss: 0.5681 - categorical_accuracy: 0.8203
 6688/60000 [==>...........................] - ETA: 1:46 - loss: 0.5669 - categorical_accuracy: 0.8207
 6720/60000 [==>...........................] - ETA: 1:46 - loss: 0.5651 - categorical_accuracy: 0.8213
 6752/60000 [==>...........................] - ETA: 1:45 - loss: 0.5638 - categorical_accuracy: 0.8217
 6784/60000 [==>...........................] - ETA: 1:45 - loss: 0.5626 - categorical_accuracy: 0.8219
 6816/60000 [==>...........................] - ETA: 1:45 - loss: 0.5611 - categorical_accuracy: 0.8222
 6848/60000 [==>...........................] - ETA: 1:45 - loss: 0.5591 - categorical_accuracy: 0.8230
 6880/60000 [==>...........................] - ETA: 1:45 - loss: 0.5573 - categorical_accuracy: 0.8234
 6912/60000 [==>...........................] - ETA: 1:45 - loss: 0.5559 - categorical_accuracy: 0.8238
 6944/60000 [==>...........................] - ETA: 1:45 - loss: 0.5541 - categorical_accuracy: 0.8243
 6976/60000 [==>...........................] - ETA: 1:45 - loss: 0.5531 - categorical_accuracy: 0.8245
 7008/60000 [==>...........................] - ETA: 1:45 - loss: 0.5515 - categorical_accuracy: 0.8252
 7040/60000 [==>...........................] - ETA: 1:45 - loss: 0.5507 - categorical_accuracy: 0.8254
 7072/60000 [==>...........................] - ETA: 1:45 - loss: 0.5485 - categorical_accuracy: 0.8262
 7104/60000 [==>...........................] - ETA: 1:45 - loss: 0.5472 - categorical_accuracy: 0.8267
 7136/60000 [==>...........................] - ETA: 1:45 - loss: 0.5467 - categorical_accuracy: 0.8268
 7168/60000 [==>...........................] - ETA: 1:45 - loss: 0.5448 - categorical_accuracy: 0.8274
 7200/60000 [==>...........................] - ETA: 1:44 - loss: 0.5433 - categorical_accuracy: 0.8279
 7232/60000 [==>...........................] - ETA: 1:44 - loss: 0.5423 - categorical_accuracy: 0.8283
 7264/60000 [==>...........................] - ETA: 1:44 - loss: 0.5405 - categorical_accuracy: 0.8289
 7296/60000 [==>...........................] - ETA: 1:44 - loss: 0.5387 - categorical_accuracy: 0.8295
 7328/60000 [==>...........................] - ETA: 1:44 - loss: 0.5373 - categorical_accuracy: 0.8300
 7360/60000 [==>...........................] - ETA: 1:44 - loss: 0.5371 - categorical_accuracy: 0.8299
 7392/60000 [==>...........................] - ETA: 1:44 - loss: 0.5353 - categorical_accuracy: 0.8305
 7424/60000 [==>...........................] - ETA: 1:44 - loss: 0.5343 - categorical_accuracy: 0.8308
 7456/60000 [==>...........................] - ETA: 1:44 - loss: 0.5339 - categorical_accuracy: 0.8311
 7488/60000 [==>...........................] - ETA: 1:44 - loss: 0.5328 - categorical_accuracy: 0.8316
 7520/60000 [==>...........................] - ETA: 1:44 - loss: 0.5312 - categorical_accuracy: 0.8320
 7552/60000 [==>...........................] - ETA: 1:44 - loss: 0.5301 - categorical_accuracy: 0.8324
 7584/60000 [==>...........................] - ETA: 1:44 - loss: 0.5284 - categorical_accuracy: 0.8329
 7616/60000 [==>...........................] - ETA: 1:44 - loss: 0.5269 - categorical_accuracy: 0.8336
 7648/60000 [==>...........................] - ETA: 1:44 - loss: 0.5256 - categorical_accuracy: 0.8341
 7680/60000 [==>...........................] - ETA: 1:43 - loss: 0.5237 - categorical_accuracy: 0.8348
 7712/60000 [==>...........................] - ETA: 1:43 - loss: 0.5220 - categorical_accuracy: 0.8353
 7744/60000 [==>...........................] - ETA: 1:43 - loss: 0.5202 - categorical_accuracy: 0.8359
 7776/60000 [==>...........................] - ETA: 1:43 - loss: 0.5190 - categorical_accuracy: 0.8364
 7808/60000 [==>...........................] - ETA: 1:43 - loss: 0.5172 - categorical_accuracy: 0.8371
 7840/60000 [==>...........................] - ETA: 1:43 - loss: 0.5165 - categorical_accuracy: 0.8376
 7872/60000 [==>...........................] - ETA: 1:43 - loss: 0.5148 - categorical_accuracy: 0.8380
 7904/60000 [==>...........................] - ETA: 1:43 - loss: 0.5136 - categorical_accuracy: 0.8382
 7936/60000 [==>...........................] - ETA: 1:43 - loss: 0.5125 - categorical_accuracy: 0.8385
 7968/60000 [==>...........................] - ETA: 1:43 - loss: 0.5109 - categorical_accuracy: 0.8390
 8000/60000 [===>..........................] - ETA: 1:43 - loss: 0.5098 - categorical_accuracy: 0.8391
 8032/60000 [===>..........................] - ETA: 1:43 - loss: 0.5095 - categorical_accuracy: 0.8395
 8064/60000 [===>..........................] - ETA: 1:43 - loss: 0.5081 - categorical_accuracy: 0.8399
 8096/60000 [===>..........................] - ETA: 1:43 - loss: 0.5072 - categorical_accuracy: 0.8402
 8128/60000 [===>..........................] - ETA: 1:42 - loss: 0.5062 - categorical_accuracy: 0.8406
 8160/60000 [===>..........................] - ETA: 1:42 - loss: 0.5049 - categorical_accuracy: 0.8409
 8192/60000 [===>..........................] - ETA: 1:42 - loss: 0.5036 - categorical_accuracy: 0.8411
 8224/60000 [===>..........................] - ETA: 1:42 - loss: 0.5023 - categorical_accuracy: 0.8416
 8256/60000 [===>..........................] - ETA: 1:42 - loss: 0.5009 - categorical_accuracy: 0.8421
 8288/60000 [===>..........................] - ETA: 1:42 - loss: 0.4996 - categorical_accuracy: 0.8424
 8320/60000 [===>..........................] - ETA: 1:42 - loss: 0.4982 - categorical_accuracy: 0.8428
 8352/60000 [===>..........................] - ETA: 1:42 - loss: 0.4965 - categorical_accuracy: 0.8434
 8384/60000 [===>..........................] - ETA: 1:42 - loss: 0.4962 - categorical_accuracy: 0.8438
 8416/60000 [===>..........................] - ETA: 1:42 - loss: 0.4948 - categorical_accuracy: 0.8442
 8448/60000 [===>..........................] - ETA: 1:42 - loss: 0.4932 - categorical_accuracy: 0.8447
 8480/60000 [===>..........................] - ETA: 1:42 - loss: 0.4921 - categorical_accuracy: 0.8449
 8512/60000 [===>..........................] - ETA: 1:42 - loss: 0.4905 - categorical_accuracy: 0.8455
 8544/60000 [===>..........................] - ETA: 1:41 - loss: 0.4892 - categorical_accuracy: 0.8459
 8576/60000 [===>..........................] - ETA: 1:41 - loss: 0.4886 - categorical_accuracy: 0.8462
 8608/60000 [===>..........................] - ETA: 1:41 - loss: 0.4876 - categorical_accuracy: 0.8465
 8640/60000 [===>..........................] - ETA: 1:41 - loss: 0.4875 - categorical_accuracy: 0.8466
 8672/60000 [===>..........................] - ETA: 1:41 - loss: 0.4865 - categorical_accuracy: 0.8469
 8704/60000 [===>..........................] - ETA: 1:41 - loss: 0.4855 - categorical_accuracy: 0.8471
 8736/60000 [===>..........................] - ETA: 1:41 - loss: 0.4841 - categorical_accuracy: 0.8475
 8768/60000 [===>..........................] - ETA: 1:41 - loss: 0.4830 - categorical_accuracy: 0.8479
 8800/60000 [===>..........................] - ETA: 1:41 - loss: 0.4825 - categorical_accuracy: 0.8482
 8832/60000 [===>..........................] - ETA: 1:41 - loss: 0.4814 - categorical_accuracy: 0.8485
 8864/60000 [===>..........................] - ETA: 1:41 - loss: 0.4801 - categorical_accuracy: 0.8489
 8896/60000 [===>..........................] - ETA: 1:41 - loss: 0.4786 - categorical_accuracy: 0.8495
 8928/60000 [===>..........................] - ETA: 1:41 - loss: 0.4784 - categorical_accuracy: 0.8496
 8960/60000 [===>..........................] - ETA: 1:40 - loss: 0.4778 - categorical_accuracy: 0.8497
 8992/60000 [===>..........................] - ETA: 1:40 - loss: 0.4766 - categorical_accuracy: 0.8501
 9024/60000 [===>..........................] - ETA: 1:40 - loss: 0.4759 - categorical_accuracy: 0.8504
 9056/60000 [===>..........................] - ETA: 1:40 - loss: 0.4750 - categorical_accuracy: 0.8507
 9088/60000 [===>..........................] - ETA: 1:40 - loss: 0.4735 - categorical_accuracy: 0.8512
 9120/60000 [===>..........................] - ETA: 1:40 - loss: 0.4726 - categorical_accuracy: 0.8516
 9152/60000 [===>..........................] - ETA: 1:40 - loss: 0.4718 - categorical_accuracy: 0.8519
 9184/60000 [===>..........................] - ETA: 1:40 - loss: 0.4713 - categorical_accuracy: 0.8520
 9216/60000 [===>..........................] - ETA: 1:40 - loss: 0.4700 - categorical_accuracy: 0.8524
 9248/60000 [===>..........................] - ETA: 1:40 - loss: 0.4693 - categorical_accuracy: 0.8527
 9280/60000 [===>..........................] - ETA: 1:40 - loss: 0.4689 - categorical_accuracy: 0.8530
 9312/60000 [===>..........................] - ETA: 1:40 - loss: 0.4682 - categorical_accuracy: 0.8534
 9344/60000 [===>..........................] - ETA: 1:40 - loss: 0.4670 - categorical_accuracy: 0.8538
 9376/60000 [===>..........................] - ETA: 1:40 - loss: 0.4664 - categorical_accuracy: 0.8540
 9408/60000 [===>..........................] - ETA: 1:40 - loss: 0.4654 - categorical_accuracy: 0.8544
 9440/60000 [===>..........................] - ETA: 1:39 - loss: 0.4644 - categorical_accuracy: 0.8548
 9472/60000 [===>..........................] - ETA: 1:39 - loss: 0.4631 - categorical_accuracy: 0.8552
 9504/60000 [===>..........................] - ETA: 1:39 - loss: 0.4618 - categorical_accuracy: 0.8555
 9536/60000 [===>..........................] - ETA: 1:39 - loss: 0.4608 - categorical_accuracy: 0.8558
 9568/60000 [===>..........................] - ETA: 1:39 - loss: 0.4596 - categorical_accuracy: 0.8562
 9600/60000 [===>..........................] - ETA: 1:39 - loss: 0.4584 - categorical_accuracy: 0.8566
 9632/60000 [===>..........................] - ETA: 1:39 - loss: 0.4579 - categorical_accuracy: 0.8567
 9664/60000 [===>..........................] - ETA: 1:39 - loss: 0.4572 - categorical_accuracy: 0.8570
 9696/60000 [===>..........................] - ETA: 1:39 - loss: 0.4564 - categorical_accuracy: 0.8574
 9728/60000 [===>..........................] - ETA: 1:39 - loss: 0.4552 - categorical_accuracy: 0.8576
 9760/60000 [===>..........................] - ETA: 1:39 - loss: 0.4550 - categorical_accuracy: 0.8580
 9792/60000 [===>..........................] - ETA: 1:39 - loss: 0.4547 - categorical_accuracy: 0.8578
 9824/60000 [===>..........................] - ETA: 1:39 - loss: 0.4545 - categorical_accuracy: 0.8580
 9856/60000 [===>..........................] - ETA: 1:39 - loss: 0.4533 - categorical_accuracy: 0.8585
 9888/60000 [===>..........................] - ETA: 1:38 - loss: 0.4528 - categorical_accuracy: 0.8587
 9920/60000 [===>..........................] - ETA: 1:38 - loss: 0.4516 - categorical_accuracy: 0.8591
 9952/60000 [===>..........................] - ETA: 1:38 - loss: 0.4512 - categorical_accuracy: 0.8591
 9984/60000 [===>..........................] - ETA: 1:38 - loss: 0.4503 - categorical_accuracy: 0.8594
10016/60000 [====>.........................] - ETA: 1:38 - loss: 0.4492 - categorical_accuracy: 0.8596
10048/60000 [====>.........................] - ETA: 1:38 - loss: 0.4482 - categorical_accuracy: 0.8600
10080/60000 [====>.........................] - ETA: 1:38 - loss: 0.4483 - categorical_accuracy: 0.8600
10112/60000 [====>.........................] - ETA: 1:38 - loss: 0.4475 - categorical_accuracy: 0.8603
10144/60000 [====>.........................] - ETA: 1:38 - loss: 0.4473 - categorical_accuracy: 0.8601
10176/60000 [====>.........................] - ETA: 1:38 - loss: 0.4468 - categorical_accuracy: 0.8604
10208/60000 [====>.........................] - ETA: 1:38 - loss: 0.4466 - categorical_accuracy: 0.8605
10240/60000 [====>.........................] - ETA: 1:38 - loss: 0.4461 - categorical_accuracy: 0.8604
10272/60000 [====>.........................] - ETA: 1:38 - loss: 0.4453 - categorical_accuracy: 0.8606
10304/60000 [====>.........................] - ETA: 1:38 - loss: 0.4444 - categorical_accuracy: 0.8609
10336/60000 [====>.........................] - ETA: 1:37 - loss: 0.4445 - categorical_accuracy: 0.8611
10368/60000 [====>.........................] - ETA: 1:37 - loss: 0.4434 - categorical_accuracy: 0.8613
10400/60000 [====>.........................] - ETA: 1:37 - loss: 0.4423 - categorical_accuracy: 0.8616
10432/60000 [====>.........................] - ETA: 1:37 - loss: 0.4414 - categorical_accuracy: 0.8619
10464/60000 [====>.........................] - ETA: 1:37 - loss: 0.4414 - categorical_accuracy: 0.8620
10496/60000 [====>.........................] - ETA: 1:37 - loss: 0.4404 - categorical_accuracy: 0.8624
10528/60000 [====>.........................] - ETA: 1:37 - loss: 0.4398 - categorical_accuracy: 0.8627
10560/60000 [====>.........................] - ETA: 1:37 - loss: 0.4387 - categorical_accuracy: 0.8630
10592/60000 [====>.........................] - ETA: 1:37 - loss: 0.4377 - categorical_accuracy: 0.8634
10624/60000 [====>.........................] - ETA: 1:37 - loss: 0.4369 - categorical_accuracy: 0.8636
10656/60000 [====>.........................] - ETA: 1:37 - loss: 0.4360 - categorical_accuracy: 0.8639
10688/60000 [====>.........................] - ETA: 1:37 - loss: 0.4354 - categorical_accuracy: 0.8641
10720/60000 [====>.........................] - ETA: 1:37 - loss: 0.4342 - categorical_accuracy: 0.8646
10752/60000 [====>.........................] - ETA: 1:37 - loss: 0.4341 - categorical_accuracy: 0.8646
10784/60000 [====>.........................] - ETA: 1:37 - loss: 0.4335 - categorical_accuracy: 0.8648
10816/60000 [====>.........................] - ETA: 1:37 - loss: 0.4327 - categorical_accuracy: 0.8651
10848/60000 [====>.........................] - ETA: 1:37 - loss: 0.4321 - categorical_accuracy: 0.8653
10880/60000 [====>.........................] - ETA: 1:36 - loss: 0.4323 - categorical_accuracy: 0.8653
10912/60000 [====>.........................] - ETA: 1:36 - loss: 0.4319 - categorical_accuracy: 0.8655
10944/60000 [====>.........................] - ETA: 1:36 - loss: 0.4308 - categorical_accuracy: 0.8659
10976/60000 [====>.........................] - ETA: 1:36 - loss: 0.4305 - categorical_accuracy: 0.8661
11008/60000 [====>.........................] - ETA: 1:36 - loss: 0.4296 - categorical_accuracy: 0.8663
11040/60000 [====>.........................] - ETA: 1:36 - loss: 0.4290 - categorical_accuracy: 0.8665
11072/60000 [====>.........................] - ETA: 1:36 - loss: 0.4280 - categorical_accuracy: 0.8668
11104/60000 [====>.........................] - ETA: 1:36 - loss: 0.4280 - categorical_accuracy: 0.8669
11136/60000 [====>.........................] - ETA: 1:36 - loss: 0.4279 - categorical_accuracy: 0.8668
11168/60000 [====>.........................] - ETA: 1:36 - loss: 0.4271 - categorical_accuracy: 0.8671
11200/60000 [====>.........................] - ETA: 1:36 - loss: 0.4264 - categorical_accuracy: 0.8673
11232/60000 [====>.........................] - ETA: 1:36 - loss: 0.4255 - categorical_accuracy: 0.8676
11264/60000 [====>.........................] - ETA: 1:36 - loss: 0.4250 - categorical_accuracy: 0.8678
11296/60000 [====>.........................] - ETA: 1:36 - loss: 0.4241 - categorical_accuracy: 0.8680
11328/60000 [====>.........................] - ETA: 1:35 - loss: 0.4231 - categorical_accuracy: 0.8683
11360/60000 [====>.........................] - ETA: 1:35 - loss: 0.4226 - categorical_accuracy: 0.8685
11392/60000 [====>.........................] - ETA: 1:35 - loss: 0.4217 - categorical_accuracy: 0.8688
11424/60000 [====>.........................] - ETA: 1:35 - loss: 0.4207 - categorical_accuracy: 0.8690
11456/60000 [====>.........................] - ETA: 1:35 - loss: 0.4201 - categorical_accuracy: 0.8693
11488/60000 [====>.........................] - ETA: 1:35 - loss: 0.4190 - categorical_accuracy: 0.8697
11520/60000 [====>.........................] - ETA: 1:35 - loss: 0.4182 - categorical_accuracy: 0.8700
11552/60000 [====>.........................] - ETA: 1:35 - loss: 0.4180 - categorical_accuracy: 0.8701
11584/60000 [====>.........................] - ETA: 1:35 - loss: 0.4177 - categorical_accuracy: 0.8703
11616/60000 [====>.........................] - ETA: 1:35 - loss: 0.4174 - categorical_accuracy: 0.8704
11648/60000 [====>.........................] - ETA: 1:35 - loss: 0.4166 - categorical_accuracy: 0.8707
11680/60000 [====>.........................] - ETA: 1:35 - loss: 0.4161 - categorical_accuracy: 0.8708
11712/60000 [====>.........................] - ETA: 1:35 - loss: 0.4167 - categorical_accuracy: 0.8710
11744/60000 [====>.........................] - ETA: 1:35 - loss: 0.4159 - categorical_accuracy: 0.8713
11776/60000 [====>.........................] - ETA: 1:34 - loss: 0.4150 - categorical_accuracy: 0.8716
11808/60000 [====>.........................] - ETA: 1:34 - loss: 0.4141 - categorical_accuracy: 0.8719
11840/60000 [====>.........................] - ETA: 1:34 - loss: 0.4137 - categorical_accuracy: 0.8720
11872/60000 [====>.........................] - ETA: 1:34 - loss: 0.4134 - categorical_accuracy: 0.8722
11904/60000 [====>.........................] - ETA: 1:34 - loss: 0.4125 - categorical_accuracy: 0.8726
11936/60000 [====>.........................] - ETA: 1:34 - loss: 0.4116 - categorical_accuracy: 0.8728
11968/60000 [====>.........................] - ETA: 1:34 - loss: 0.4121 - categorical_accuracy: 0.8727
12000/60000 [=====>........................] - ETA: 1:34 - loss: 0.4113 - categorical_accuracy: 0.8729
12032/60000 [=====>........................] - ETA: 1:34 - loss: 0.4108 - categorical_accuracy: 0.8730
12064/60000 [=====>........................] - ETA: 1:34 - loss: 0.4099 - categorical_accuracy: 0.8733
12096/60000 [=====>........................] - ETA: 1:34 - loss: 0.4096 - categorical_accuracy: 0.8733
12128/60000 [=====>........................] - ETA: 1:34 - loss: 0.4097 - categorical_accuracy: 0.8734
12160/60000 [=====>........................] - ETA: 1:34 - loss: 0.4090 - categorical_accuracy: 0.8735
12192/60000 [=====>........................] - ETA: 1:34 - loss: 0.4084 - categorical_accuracy: 0.8738
12224/60000 [=====>........................] - ETA: 1:34 - loss: 0.4082 - categorical_accuracy: 0.8739
12256/60000 [=====>........................] - ETA: 1:33 - loss: 0.4079 - categorical_accuracy: 0.8740
12288/60000 [=====>........................] - ETA: 1:33 - loss: 0.4072 - categorical_accuracy: 0.8743
12320/60000 [=====>........................] - ETA: 1:33 - loss: 0.4068 - categorical_accuracy: 0.8744
12352/60000 [=====>........................] - ETA: 1:33 - loss: 0.4061 - categorical_accuracy: 0.8748
12384/60000 [=====>........................] - ETA: 1:33 - loss: 0.4057 - categorical_accuracy: 0.8750
12416/60000 [=====>........................] - ETA: 1:33 - loss: 0.4052 - categorical_accuracy: 0.8752
12448/60000 [=====>........................] - ETA: 1:33 - loss: 0.4045 - categorical_accuracy: 0.8755
12480/60000 [=====>........................] - ETA: 1:33 - loss: 0.4040 - categorical_accuracy: 0.8756
12512/60000 [=====>........................] - ETA: 1:33 - loss: 0.4036 - categorical_accuracy: 0.8758
12544/60000 [=====>........................] - ETA: 1:33 - loss: 0.4029 - categorical_accuracy: 0.8760
12576/60000 [=====>........................] - ETA: 1:33 - loss: 0.4024 - categorical_accuracy: 0.8761
12608/60000 [=====>........................] - ETA: 1:33 - loss: 0.4017 - categorical_accuracy: 0.8763
12640/60000 [=====>........................] - ETA: 1:33 - loss: 0.4010 - categorical_accuracy: 0.8763
12672/60000 [=====>........................] - ETA: 1:33 - loss: 0.4005 - categorical_accuracy: 0.8765
12704/60000 [=====>........................] - ETA: 1:33 - loss: 0.3998 - categorical_accuracy: 0.8767
12736/60000 [=====>........................] - ETA: 1:32 - loss: 0.3991 - categorical_accuracy: 0.8770
12768/60000 [=====>........................] - ETA: 1:32 - loss: 0.3985 - categorical_accuracy: 0.8772
12800/60000 [=====>........................] - ETA: 1:32 - loss: 0.3987 - categorical_accuracy: 0.8773
12832/60000 [=====>........................] - ETA: 1:32 - loss: 0.3978 - categorical_accuracy: 0.8776
12864/60000 [=====>........................] - ETA: 1:32 - loss: 0.3974 - categorical_accuracy: 0.8777
12896/60000 [=====>........................] - ETA: 1:32 - loss: 0.3975 - categorical_accuracy: 0.8776
12928/60000 [=====>........................] - ETA: 1:32 - loss: 0.3968 - categorical_accuracy: 0.8779
12960/60000 [=====>........................] - ETA: 1:32 - loss: 0.3962 - categorical_accuracy: 0.8780
12992/60000 [=====>........................] - ETA: 1:32 - loss: 0.3954 - categorical_accuracy: 0.8783
13024/60000 [=====>........................] - ETA: 1:32 - loss: 0.3948 - categorical_accuracy: 0.8785
13056/60000 [=====>........................] - ETA: 1:32 - loss: 0.3940 - categorical_accuracy: 0.8787
13088/60000 [=====>........................] - ETA: 1:32 - loss: 0.3935 - categorical_accuracy: 0.8787
13120/60000 [=====>........................] - ETA: 1:32 - loss: 0.3933 - categorical_accuracy: 0.8787
13152/60000 [=====>........................] - ETA: 1:32 - loss: 0.3927 - categorical_accuracy: 0.8790
13184/60000 [=====>........................] - ETA: 1:32 - loss: 0.3921 - categorical_accuracy: 0.8791
13216/60000 [=====>........................] - ETA: 1:31 - loss: 0.3914 - categorical_accuracy: 0.8792
13248/60000 [=====>........................] - ETA: 1:31 - loss: 0.3906 - categorical_accuracy: 0.8795
13280/60000 [=====>........................] - ETA: 1:31 - loss: 0.3898 - categorical_accuracy: 0.8798
13312/60000 [=====>........................] - ETA: 1:31 - loss: 0.3890 - categorical_accuracy: 0.8800
13344/60000 [=====>........................] - ETA: 1:31 - loss: 0.3886 - categorical_accuracy: 0.8799
13376/60000 [=====>........................] - ETA: 1:31 - loss: 0.3885 - categorical_accuracy: 0.8800
13408/60000 [=====>........................] - ETA: 1:31 - loss: 0.3885 - categorical_accuracy: 0.8801
13440/60000 [=====>........................] - ETA: 1:31 - loss: 0.3883 - categorical_accuracy: 0.8802
13472/60000 [=====>........................] - ETA: 1:31 - loss: 0.3876 - categorical_accuracy: 0.8803
13504/60000 [=====>........................] - ETA: 1:31 - loss: 0.3867 - categorical_accuracy: 0.8806
13536/60000 [=====>........................] - ETA: 1:31 - loss: 0.3860 - categorical_accuracy: 0.8808
13568/60000 [=====>........................] - ETA: 1:31 - loss: 0.3862 - categorical_accuracy: 0.8809
13600/60000 [=====>........................] - ETA: 1:31 - loss: 0.3861 - categorical_accuracy: 0.8810
13632/60000 [=====>........................] - ETA: 1:31 - loss: 0.3858 - categorical_accuracy: 0.8811
13664/60000 [=====>........................] - ETA: 1:31 - loss: 0.3853 - categorical_accuracy: 0.8812
13696/60000 [=====>........................] - ETA: 1:30 - loss: 0.3849 - categorical_accuracy: 0.8814
13728/60000 [=====>........................] - ETA: 1:30 - loss: 0.3844 - categorical_accuracy: 0.8816
13760/60000 [=====>........................] - ETA: 1:30 - loss: 0.3841 - categorical_accuracy: 0.8817
13792/60000 [=====>........................] - ETA: 1:30 - loss: 0.3835 - categorical_accuracy: 0.8820
13824/60000 [=====>........................] - ETA: 1:30 - loss: 0.3829 - categorical_accuracy: 0.8822
13856/60000 [=====>........................] - ETA: 1:30 - loss: 0.3822 - categorical_accuracy: 0.8823
13888/60000 [=====>........................] - ETA: 1:30 - loss: 0.3816 - categorical_accuracy: 0.8825
13920/60000 [=====>........................] - ETA: 1:30 - loss: 0.3814 - categorical_accuracy: 0.8826
13952/60000 [=====>........................] - ETA: 1:30 - loss: 0.3809 - categorical_accuracy: 0.8828
13984/60000 [=====>........................] - ETA: 1:30 - loss: 0.3802 - categorical_accuracy: 0.8830
14016/60000 [======>.......................] - ETA: 1:30 - loss: 0.3800 - categorical_accuracy: 0.8831
14048/60000 [======>.......................] - ETA: 1:30 - loss: 0.3796 - categorical_accuracy: 0.8832
14080/60000 [======>.......................] - ETA: 1:30 - loss: 0.3796 - categorical_accuracy: 0.8832
14112/60000 [======>.......................] - ETA: 1:30 - loss: 0.3793 - categorical_accuracy: 0.8833
14144/60000 [======>.......................] - ETA: 1:30 - loss: 0.3788 - categorical_accuracy: 0.8835
14176/60000 [======>.......................] - ETA: 1:29 - loss: 0.3781 - categorical_accuracy: 0.8837
14208/60000 [======>.......................] - ETA: 1:29 - loss: 0.3774 - categorical_accuracy: 0.8839
14240/60000 [======>.......................] - ETA: 1:29 - loss: 0.3767 - categorical_accuracy: 0.8842
14272/60000 [======>.......................] - ETA: 1:29 - loss: 0.3761 - categorical_accuracy: 0.8844
14304/60000 [======>.......................] - ETA: 1:29 - loss: 0.3764 - categorical_accuracy: 0.8845
14336/60000 [======>.......................] - ETA: 1:29 - loss: 0.3758 - categorical_accuracy: 0.8846
14368/60000 [======>.......................] - ETA: 1:29 - loss: 0.3752 - categorical_accuracy: 0.8848
14400/60000 [======>.......................] - ETA: 1:29 - loss: 0.3745 - categorical_accuracy: 0.8850
14432/60000 [======>.......................] - ETA: 1:29 - loss: 0.3744 - categorical_accuracy: 0.8850
14464/60000 [======>.......................] - ETA: 1:29 - loss: 0.3741 - categorical_accuracy: 0.8851
14496/60000 [======>.......................] - ETA: 1:29 - loss: 0.3736 - categorical_accuracy: 0.8853
14528/60000 [======>.......................] - ETA: 1:29 - loss: 0.3733 - categorical_accuracy: 0.8854
14560/60000 [======>.......................] - ETA: 1:29 - loss: 0.3726 - categorical_accuracy: 0.8856
14592/60000 [======>.......................] - ETA: 1:29 - loss: 0.3723 - categorical_accuracy: 0.8857
14624/60000 [======>.......................] - ETA: 1:28 - loss: 0.3720 - categorical_accuracy: 0.8857
14656/60000 [======>.......................] - ETA: 1:28 - loss: 0.3714 - categorical_accuracy: 0.8858
14688/60000 [======>.......................] - ETA: 1:28 - loss: 0.3709 - categorical_accuracy: 0.8860
14720/60000 [======>.......................] - ETA: 1:28 - loss: 0.3703 - categorical_accuracy: 0.8861
14752/60000 [======>.......................] - ETA: 1:28 - loss: 0.3697 - categorical_accuracy: 0.8864
14784/60000 [======>.......................] - ETA: 1:28 - loss: 0.3689 - categorical_accuracy: 0.8866
14816/60000 [======>.......................] - ETA: 1:28 - loss: 0.3682 - categorical_accuracy: 0.8869
14848/60000 [======>.......................] - ETA: 1:28 - loss: 0.3677 - categorical_accuracy: 0.8871
14880/60000 [======>.......................] - ETA: 1:28 - loss: 0.3672 - categorical_accuracy: 0.8872
14912/60000 [======>.......................] - ETA: 1:28 - loss: 0.3665 - categorical_accuracy: 0.8875
14944/60000 [======>.......................] - ETA: 1:28 - loss: 0.3659 - categorical_accuracy: 0.8876
14976/60000 [======>.......................] - ETA: 1:28 - loss: 0.3654 - categorical_accuracy: 0.8878
15008/60000 [======>.......................] - ETA: 1:28 - loss: 0.3649 - categorical_accuracy: 0.8879
15040/60000 [======>.......................] - ETA: 1:28 - loss: 0.3644 - categorical_accuracy: 0.8880
15072/60000 [======>.......................] - ETA: 1:28 - loss: 0.3642 - categorical_accuracy: 0.8880
15104/60000 [======>.......................] - ETA: 1:27 - loss: 0.3637 - categorical_accuracy: 0.8882
15136/60000 [======>.......................] - ETA: 1:27 - loss: 0.3636 - categorical_accuracy: 0.8883
15168/60000 [======>.......................] - ETA: 1:27 - loss: 0.3630 - categorical_accuracy: 0.8885
15200/60000 [======>.......................] - ETA: 1:27 - loss: 0.3624 - categorical_accuracy: 0.8887
15232/60000 [======>.......................] - ETA: 1:27 - loss: 0.3619 - categorical_accuracy: 0.8889
15264/60000 [======>.......................] - ETA: 1:27 - loss: 0.3613 - categorical_accuracy: 0.8890
15296/60000 [======>.......................] - ETA: 1:27 - loss: 0.3611 - categorical_accuracy: 0.8892
15328/60000 [======>.......................] - ETA: 1:27 - loss: 0.3605 - categorical_accuracy: 0.8894
15360/60000 [======>.......................] - ETA: 1:27 - loss: 0.3599 - categorical_accuracy: 0.8896
15392/60000 [======>.......................] - ETA: 1:27 - loss: 0.3596 - categorical_accuracy: 0.8897
15424/60000 [======>.......................] - ETA: 1:27 - loss: 0.3590 - categorical_accuracy: 0.8898
15456/60000 [======>.......................] - ETA: 1:27 - loss: 0.3584 - categorical_accuracy: 0.8901
15488/60000 [======>.......................] - ETA: 1:27 - loss: 0.3580 - categorical_accuracy: 0.8901
15520/60000 [======>.......................] - ETA: 1:27 - loss: 0.3580 - categorical_accuracy: 0.8901
15552/60000 [======>.......................] - ETA: 1:27 - loss: 0.3582 - categorical_accuracy: 0.8900
15584/60000 [======>.......................] - ETA: 1:26 - loss: 0.3585 - categorical_accuracy: 0.8901
15616/60000 [======>.......................] - ETA: 1:26 - loss: 0.3581 - categorical_accuracy: 0.8902
15648/60000 [======>.......................] - ETA: 1:26 - loss: 0.3579 - categorical_accuracy: 0.8903
15680/60000 [======>.......................] - ETA: 1:26 - loss: 0.3578 - categorical_accuracy: 0.8903
15712/60000 [======>.......................] - ETA: 1:26 - loss: 0.3574 - categorical_accuracy: 0.8905
15744/60000 [======>.......................] - ETA: 1:26 - loss: 0.3568 - categorical_accuracy: 0.8906
15776/60000 [======>.......................] - ETA: 1:26 - loss: 0.3566 - categorical_accuracy: 0.8907
15808/60000 [======>.......................] - ETA: 1:26 - loss: 0.3562 - categorical_accuracy: 0.8908
15840/60000 [======>.......................] - ETA: 1:26 - loss: 0.3559 - categorical_accuracy: 0.8910
15872/60000 [======>.......................] - ETA: 1:26 - loss: 0.3555 - categorical_accuracy: 0.8911
15904/60000 [======>.......................] - ETA: 1:26 - loss: 0.3552 - categorical_accuracy: 0.8912
15936/60000 [======>.......................] - ETA: 1:26 - loss: 0.3547 - categorical_accuracy: 0.8913
15968/60000 [======>.......................] - ETA: 1:26 - loss: 0.3547 - categorical_accuracy: 0.8913
16000/60000 [=======>......................] - ETA: 1:26 - loss: 0.3541 - categorical_accuracy: 0.8914
16032/60000 [=======>......................] - ETA: 1:26 - loss: 0.3535 - categorical_accuracy: 0.8917
16064/60000 [=======>......................] - ETA: 1:26 - loss: 0.3530 - categorical_accuracy: 0.8918
16096/60000 [=======>......................] - ETA: 1:25 - loss: 0.3529 - categorical_accuracy: 0.8918
16128/60000 [=======>......................] - ETA: 1:25 - loss: 0.3525 - categorical_accuracy: 0.8919
16160/60000 [=======>......................] - ETA: 1:25 - loss: 0.3518 - categorical_accuracy: 0.8921
16192/60000 [=======>......................] - ETA: 1:25 - loss: 0.3518 - categorical_accuracy: 0.8923
16224/60000 [=======>......................] - ETA: 1:25 - loss: 0.3515 - categorical_accuracy: 0.8923
16256/60000 [=======>......................] - ETA: 1:25 - loss: 0.3510 - categorical_accuracy: 0.8925
16288/60000 [=======>......................] - ETA: 1:25 - loss: 0.3507 - categorical_accuracy: 0.8926
16320/60000 [=======>......................] - ETA: 1:25 - loss: 0.3505 - categorical_accuracy: 0.8927
16352/60000 [=======>......................] - ETA: 1:25 - loss: 0.3505 - categorical_accuracy: 0.8927
16384/60000 [=======>......................] - ETA: 1:25 - loss: 0.3505 - categorical_accuracy: 0.8929
16416/60000 [=======>......................] - ETA: 1:25 - loss: 0.3502 - categorical_accuracy: 0.8930
16448/60000 [=======>......................] - ETA: 1:25 - loss: 0.3497 - categorical_accuracy: 0.8931
16480/60000 [=======>......................] - ETA: 1:25 - loss: 0.3499 - categorical_accuracy: 0.8932
16512/60000 [=======>......................] - ETA: 1:25 - loss: 0.3494 - categorical_accuracy: 0.8934
16544/60000 [=======>......................] - ETA: 1:25 - loss: 0.3489 - categorical_accuracy: 0.8936
16576/60000 [=======>......................] - ETA: 1:24 - loss: 0.3486 - categorical_accuracy: 0.8936
16608/60000 [=======>......................] - ETA: 1:24 - loss: 0.3484 - categorical_accuracy: 0.8937
16640/60000 [=======>......................] - ETA: 1:24 - loss: 0.3480 - categorical_accuracy: 0.8938
16672/60000 [=======>......................] - ETA: 1:24 - loss: 0.3476 - categorical_accuracy: 0.8939
16704/60000 [=======>......................] - ETA: 1:24 - loss: 0.3470 - categorical_accuracy: 0.8940
16736/60000 [=======>......................] - ETA: 1:24 - loss: 0.3467 - categorical_accuracy: 0.8942
16768/60000 [=======>......................] - ETA: 1:24 - loss: 0.3472 - categorical_accuracy: 0.8943
16800/60000 [=======>......................] - ETA: 1:24 - loss: 0.3467 - categorical_accuracy: 0.8944
16832/60000 [=======>......................] - ETA: 1:24 - loss: 0.3464 - categorical_accuracy: 0.8944
16864/60000 [=======>......................] - ETA: 1:24 - loss: 0.3461 - categorical_accuracy: 0.8946
16896/60000 [=======>......................] - ETA: 1:24 - loss: 0.3456 - categorical_accuracy: 0.8947
16928/60000 [=======>......................] - ETA: 1:24 - loss: 0.3451 - categorical_accuracy: 0.8949
16960/60000 [=======>......................] - ETA: 1:24 - loss: 0.3446 - categorical_accuracy: 0.8950
16992/60000 [=======>......................] - ETA: 1:24 - loss: 0.3442 - categorical_accuracy: 0.8952
17024/60000 [=======>......................] - ETA: 1:24 - loss: 0.3436 - categorical_accuracy: 0.8954
17056/60000 [=======>......................] - ETA: 1:23 - loss: 0.3432 - categorical_accuracy: 0.8955
17088/60000 [=======>......................] - ETA: 1:23 - loss: 0.3428 - categorical_accuracy: 0.8956
17120/60000 [=======>......................] - ETA: 1:23 - loss: 0.3426 - categorical_accuracy: 0.8957
17152/60000 [=======>......................] - ETA: 1:23 - loss: 0.3427 - categorical_accuracy: 0.8957
17184/60000 [=======>......................] - ETA: 1:23 - loss: 0.3422 - categorical_accuracy: 0.8959
17216/60000 [=======>......................] - ETA: 1:23 - loss: 0.3420 - categorical_accuracy: 0.8959
17248/60000 [=======>......................] - ETA: 1:23 - loss: 0.3417 - categorical_accuracy: 0.8960
17280/60000 [=======>......................] - ETA: 1:23 - loss: 0.3413 - categorical_accuracy: 0.8961
17312/60000 [=======>......................] - ETA: 1:23 - loss: 0.3409 - categorical_accuracy: 0.8963
17344/60000 [=======>......................] - ETA: 1:23 - loss: 0.3404 - categorical_accuracy: 0.8964
17376/60000 [=======>......................] - ETA: 1:23 - loss: 0.3398 - categorical_accuracy: 0.8966
17408/60000 [=======>......................] - ETA: 1:23 - loss: 0.3396 - categorical_accuracy: 0.8967
17440/60000 [=======>......................] - ETA: 1:23 - loss: 0.3392 - categorical_accuracy: 0.8968
17472/60000 [=======>......................] - ETA: 1:23 - loss: 0.3389 - categorical_accuracy: 0.8969
17504/60000 [=======>......................] - ETA: 1:23 - loss: 0.3385 - categorical_accuracy: 0.8970
17536/60000 [=======>......................] - ETA: 1:23 - loss: 0.3381 - categorical_accuracy: 0.8971
17568/60000 [=======>......................] - ETA: 1:22 - loss: 0.3379 - categorical_accuracy: 0.8971
17600/60000 [=======>......................] - ETA: 1:22 - loss: 0.3378 - categorical_accuracy: 0.8972
17632/60000 [=======>......................] - ETA: 1:22 - loss: 0.3373 - categorical_accuracy: 0.8973
17664/60000 [=======>......................] - ETA: 1:22 - loss: 0.3370 - categorical_accuracy: 0.8974
17696/60000 [=======>......................] - ETA: 1:22 - loss: 0.3365 - categorical_accuracy: 0.8975
17728/60000 [=======>......................] - ETA: 1:22 - loss: 0.3365 - categorical_accuracy: 0.8975
17760/60000 [=======>......................] - ETA: 1:22 - loss: 0.3359 - categorical_accuracy: 0.8977
17792/60000 [=======>......................] - ETA: 1:22 - loss: 0.3355 - categorical_accuracy: 0.8978
17824/60000 [=======>......................] - ETA: 1:22 - loss: 0.3358 - categorical_accuracy: 0.8978
17856/60000 [=======>......................] - ETA: 1:22 - loss: 0.3358 - categorical_accuracy: 0.8977
17888/60000 [=======>......................] - ETA: 1:22 - loss: 0.3355 - categorical_accuracy: 0.8978
17920/60000 [=======>......................] - ETA: 1:22 - loss: 0.3353 - categorical_accuracy: 0.8979
17952/60000 [=======>......................] - ETA: 1:22 - loss: 0.3352 - categorical_accuracy: 0.8980
17984/60000 [=======>......................] - ETA: 1:22 - loss: 0.3347 - categorical_accuracy: 0.8981
18016/60000 [========>.....................] - ETA: 1:22 - loss: 0.3344 - categorical_accuracy: 0.8983
18048/60000 [========>.....................] - ETA: 1:21 - loss: 0.3339 - categorical_accuracy: 0.8984
18080/60000 [========>.....................] - ETA: 1:21 - loss: 0.3335 - categorical_accuracy: 0.8985
18112/60000 [========>.....................] - ETA: 1:21 - loss: 0.3337 - categorical_accuracy: 0.8985
18144/60000 [========>.....................] - ETA: 1:21 - loss: 0.3333 - categorical_accuracy: 0.8987
18176/60000 [========>.....................] - ETA: 1:21 - loss: 0.3328 - categorical_accuracy: 0.8989
18208/60000 [========>.....................] - ETA: 1:21 - loss: 0.3326 - categorical_accuracy: 0.8989
18240/60000 [========>.....................] - ETA: 1:21 - loss: 0.3322 - categorical_accuracy: 0.8990
18272/60000 [========>.....................] - ETA: 1:21 - loss: 0.3321 - categorical_accuracy: 0.8991
18304/60000 [========>.....................] - ETA: 1:21 - loss: 0.3317 - categorical_accuracy: 0.8992
18336/60000 [========>.....................] - ETA: 1:21 - loss: 0.3312 - categorical_accuracy: 0.8994
18368/60000 [========>.....................] - ETA: 1:21 - loss: 0.3308 - categorical_accuracy: 0.8994
18400/60000 [========>.....................] - ETA: 1:21 - loss: 0.3304 - categorical_accuracy: 0.8995
18432/60000 [========>.....................] - ETA: 1:21 - loss: 0.3300 - categorical_accuracy: 0.8996
18464/60000 [========>.....................] - ETA: 1:21 - loss: 0.3296 - categorical_accuracy: 0.8998
18496/60000 [========>.....................] - ETA: 1:21 - loss: 0.3293 - categorical_accuracy: 0.8998
18528/60000 [========>.....................] - ETA: 1:20 - loss: 0.3287 - categorical_accuracy: 0.9000
18560/60000 [========>.....................] - ETA: 1:20 - loss: 0.3284 - categorical_accuracy: 0.9001
18592/60000 [========>.....................] - ETA: 1:20 - loss: 0.3278 - categorical_accuracy: 0.9002
18624/60000 [========>.....................] - ETA: 1:20 - loss: 0.3274 - categorical_accuracy: 0.9003
18656/60000 [========>.....................] - ETA: 1:20 - loss: 0.3270 - categorical_accuracy: 0.9005
18688/60000 [========>.....................] - ETA: 1:20 - loss: 0.3267 - categorical_accuracy: 0.9005
18720/60000 [========>.....................] - ETA: 1:20 - loss: 0.3264 - categorical_accuracy: 0.9006
18752/60000 [========>.....................] - ETA: 1:20 - loss: 0.3259 - categorical_accuracy: 0.9008
18784/60000 [========>.....................] - ETA: 1:20 - loss: 0.3253 - categorical_accuracy: 0.9010
18816/60000 [========>.....................] - ETA: 1:20 - loss: 0.3251 - categorical_accuracy: 0.9010
18848/60000 [========>.....................] - ETA: 1:20 - loss: 0.3246 - categorical_accuracy: 0.9012
18880/60000 [========>.....................] - ETA: 1:20 - loss: 0.3246 - categorical_accuracy: 0.9012
18912/60000 [========>.....................] - ETA: 1:20 - loss: 0.3243 - categorical_accuracy: 0.9012
18944/60000 [========>.....................] - ETA: 1:20 - loss: 0.3239 - categorical_accuracy: 0.9013
18976/60000 [========>.....................] - ETA: 1:20 - loss: 0.3236 - categorical_accuracy: 0.9015
19008/60000 [========>.....................] - ETA: 1:20 - loss: 0.3234 - categorical_accuracy: 0.9015
19040/60000 [========>.....................] - ETA: 1:19 - loss: 0.3233 - categorical_accuracy: 0.9015
19072/60000 [========>.....................] - ETA: 1:19 - loss: 0.3228 - categorical_accuracy: 0.9016
19104/60000 [========>.....................] - ETA: 1:19 - loss: 0.3224 - categorical_accuracy: 0.9018
19136/60000 [========>.....................] - ETA: 1:19 - loss: 0.3219 - categorical_accuracy: 0.9019
19168/60000 [========>.....................] - ETA: 1:19 - loss: 0.3214 - categorical_accuracy: 0.9021
19200/60000 [========>.....................] - ETA: 1:19 - loss: 0.3209 - categorical_accuracy: 0.9022
19232/60000 [========>.....................] - ETA: 1:19 - loss: 0.3207 - categorical_accuracy: 0.9023
19264/60000 [========>.....................] - ETA: 1:19 - loss: 0.3205 - categorical_accuracy: 0.9024
19296/60000 [========>.....................] - ETA: 1:19 - loss: 0.3203 - categorical_accuracy: 0.9024
19328/60000 [========>.....................] - ETA: 1:19 - loss: 0.3199 - categorical_accuracy: 0.9025
19360/60000 [========>.....................] - ETA: 1:19 - loss: 0.3194 - categorical_accuracy: 0.9027
19392/60000 [========>.....................] - ETA: 1:19 - loss: 0.3190 - categorical_accuracy: 0.9028
19424/60000 [========>.....................] - ETA: 1:19 - loss: 0.3186 - categorical_accuracy: 0.9029
19456/60000 [========>.....................] - ETA: 1:19 - loss: 0.3182 - categorical_accuracy: 0.9031
19488/60000 [========>.....................] - ETA: 1:19 - loss: 0.3177 - categorical_accuracy: 0.9032
19520/60000 [========>.....................] - ETA: 1:18 - loss: 0.3175 - categorical_accuracy: 0.9033
19552/60000 [========>.....................] - ETA: 1:18 - loss: 0.3170 - categorical_accuracy: 0.9034
19584/60000 [========>.....................] - ETA: 1:18 - loss: 0.3169 - categorical_accuracy: 0.9034
19616/60000 [========>.....................] - ETA: 1:18 - loss: 0.3168 - categorical_accuracy: 0.9035
19648/60000 [========>.....................] - ETA: 1:18 - loss: 0.3165 - categorical_accuracy: 0.9036
19680/60000 [========>.....................] - ETA: 1:18 - loss: 0.3160 - categorical_accuracy: 0.9038
19712/60000 [========>.....................] - ETA: 1:18 - loss: 0.3157 - categorical_accuracy: 0.9039
19744/60000 [========>.....................] - ETA: 1:18 - loss: 0.3152 - categorical_accuracy: 0.9040
19776/60000 [========>.....................] - ETA: 1:18 - loss: 0.3153 - categorical_accuracy: 0.9039
19808/60000 [========>.....................] - ETA: 1:18 - loss: 0.3153 - categorical_accuracy: 0.9039
19840/60000 [========>.....................] - ETA: 1:18 - loss: 0.3152 - categorical_accuracy: 0.9039
19872/60000 [========>.....................] - ETA: 1:18 - loss: 0.3148 - categorical_accuracy: 0.9040
19904/60000 [========>.....................] - ETA: 1:18 - loss: 0.3146 - categorical_accuracy: 0.9041
19936/60000 [========>.....................] - ETA: 1:18 - loss: 0.3142 - categorical_accuracy: 0.9043
19968/60000 [========>.....................] - ETA: 1:18 - loss: 0.3137 - categorical_accuracy: 0.9044
20000/60000 [=========>....................] - ETA: 1:18 - loss: 0.3134 - categorical_accuracy: 0.9046
20032/60000 [=========>....................] - ETA: 1:17 - loss: 0.3130 - categorical_accuracy: 0.9047
20064/60000 [=========>....................] - ETA: 1:17 - loss: 0.3128 - categorical_accuracy: 0.9048
20096/60000 [=========>....................] - ETA: 1:17 - loss: 0.3125 - categorical_accuracy: 0.9049
20128/60000 [=========>....................] - ETA: 1:17 - loss: 0.3124 - categorical_accuracy: 0.9049
20160/60000 [=========>....................] - ETA: 1:17 - loss: 0.3121 - categorical_accuracy: 0.9049
20192/60000 [=========>....................] - ETA: 1:17 - loss: 0.3117 - categorical_accuracy: 0.9051
20224/60000 [=========>....................] - ETA: 1:17 - loss: 0.3115 - categorical_accuracy: 0.9052
20256/60000 [=========>....................] - ETA: 1:17 - loss: 0.3111 - categorical_accuracy: 0.9053
20288/60000 [=========>....................] - ETA: 1:17 - loss: 0.3108 - categorical_accuracy: 0.9054
20320/60000 [=========>....................] - ETA: 1:17 - loss: 0.3104 - categorical_accuracy: 0.9055
20352/60000 [=========>....................] - ETA: 1:17 - loss: 0.3103 - categorical_accuracy: 0.9055
20384/60000 [=========>....................] - ETA: 1:17 - loss: 0.3104 - categorical_accuracy: 0.9056
20416/60000 [=========>....................] - ETA: 1:17 - loss: 0.3103 - categorical_accuracy: 0.9056
20448/60000 [=========>....................] - ETA: 1:17 - loss: 0.3102 - categorical_accuracy: 0.9056
20480/60000 [=========>....................] - ETA: 1:17 - loss: 0.3098 - categorical_accuracy: 0.9058
20512/60000 [=========>....................] - ETA: 1:17 - loss: 0.3096 - categorical_accuracy: 0.9058
20544/60000 [=========>....................] - ETA: 1:16 - loss: 0.3095 - categorical_accuracy: 0.9059
20576/60000 [=========>....................] - ETA: 1:16 - loss: 0.3092 - categorical_accuracy: 0.9059
20608/60000 [=========>....................] - ETA: 1:16 - loss: 0.3088 - categorical_accuracy: 0.9060
20640/60000 [=========>....................] - ETA: 1:16 - loss: 0.3086 - categorical_accuracy: 0.9061
20672/60000 [=========>....................] - ETA: 1:16 - loss: 0.3082 - categorical_accuracy: 0.9062
20704/60000 [=========>....................] - ETA: 1:16 - loss: 0.3082 - categorical_accuracy: 0.9062
20736/60000 [=========>....................] - ETA: 1:16 - loss: 0.3083 - categorical_accuracy: 0.9062
20768/60000 [=========>....................] - ETA: 1:16 - loss: 0.3080 - categorical_accuracy: 0.9062
20800/60000 [=========>....................] - ETA: 1:16 - loss: 0.3077 - categorical_accuracy: 0.9062
20832/60000 [=========>....................] - ETA: 1:16 - loss: 0.3074 - categorical_accuracy: 0.9063
20864/60000 [=========>....................] - ETA: 1:16 - loss: 0.3071 - categorical_accuracy: 0.9064
20896/60000 [=========>....................] - ETA: 1:16 - loss: 0.3069 - categorical_accuracy: 0.9064
20928/60000 [=========>....................] - ETA: 1:16 - loss: 0.3066 - categorical_accuracy: 0.9065
20960/60000 [=========>....................] - ETA: 1:16 - loss: 0.3066 - categorical_accuracy: 0.9065
20992/60000 [=========>....................] - ETA: 1:16 - loss: 0.3063 - categorical_accuracy: 0.9066
21024/60000 [=========>....................] - ETA: 1:15 - loss: 0.3059 - categorical_accuracy: 0.9067
21056/60000 [=========>....................] - ETA: 1:15 - loss: 0.3057 - categorical_accuracy: 0.9068
21088/60000 [=========>....................] - ETA: 1:15 - loss: 0.3053 - categorical_accuracy: 0.9069
21120/60000 [=========>....................] - ETA: 1:15 - loss: 0.3054 - categorical_accuracy: 0.9069
21152/60000 [=========>....................] - ETA: 1:15 - loss: 0.3051 - categorical_accuracy: 0.9070
21184/60000 [=========>....................] - ETA: 1:15 - loss: 0.3049 - categorical_accuracy: 0.9070
21216/60000 [=========>....................] - ETA: 1:15 - loss: 0.3046 - categorical_accuracy: 0.9071
21248/60000 [=========>....................] - ETA: 1:15 - loss: 0.3046 - categorical_accuracy: 0.9071
21280/60000 [=========>....................] - ETA: 1:15 - loss: 0.3044 - categorical_accuracy: 0.9072
21312/60000 [=========>....................] - ETA: 1:15 - loss: 0.3043 - categorical_accuracy: 0.9072
21344/60000 [=========>....................] - ETA: 1:15 - loss: 0.3043 - categorical_accuracy: 0.9071
21376/60000 [=========>....................] - ETA: 1:15 - loss: 0.3041 - categorical_accuracy: 0.9072
21408/60000 [=========>....................] - ETA: 1:15 - loss: 0.3037 - categorical_accuracy: 0.9073
21440/60000 [=========>....................] - ETA: 1:15 - loss: 0.3037 - categorical_accuracy: 0.9073
21472/60000 [=========>....................] - ETA: 1:15 - loss: 0.3037 - categorical_accuracy: 0.9074
21504/60000 [=========>....................] - ETA: 1:14 - loss: 0.3033 - categorical_accuracy: 0.9075
21536/60000 [=========>....................] - ETA: 1:14 - loss: 0.3030 - categorical_accuracy: 0.9076
21568/60000 [=========>....................] - ETA: 1:14 - loss: 0.3026 - categorical_accuracy: 0.9077
21600/60000 [=========>....................] - ETA: 1:14 - loss: 0.3023 - categorical_accuracy: 0.9078
21632/60000 [=========>....................] - ETA: 1:14 - loss: 0.3019 - categorical_accuracy: 0.9079
21664/60000 [=========>....................] - ETA: 1:14 - loss: 0.3016 - categorical_accuracy: 0.9081
21696/60000 [=========>....................] - ETA: 1:14 - loss: 0.3013 - categorical_accuracy: 0.9081
21728/60000 [=========>....................] - ETA: 1:14 - loss: 0.3009 - categorical_accuracy: 0.9083
21760/60000 [=========>....................] - ETA: 1:14 - loss: 0.3006 - categorical_accuracy: 0.9084
21792/60000 [=========>....................] - ETA: 1:14 - loss: 0.3004 - categorical_accuracy: 0.9084
21824/60000 [=========>....................] - ETA: 1:14 - loss: 0.3001 - categorical_accuracy: 0.9084
21856/60000 [=========>....................] - ETA: 1:14 - loss: 0.2997 - categorical_accuracy: 0.9086
21888/60000 [=========>....................] - ETA: 1:14 - loss: 0.2994 - categorical_accuracy: 0.9086
21920/60000 [=========>....................] - ETA: 1:14 - loss: 0.2995 - categorical_accuracy: 0.9086
21952/60000 [=========>....................] - ETA: 1:14 - loss: 0.2992 - categorical_accuracy: 0.9087
21984/60000 [=========>....................] - ETA: 1:14 - loss: 0.2993 - categorical_accuracy: 0.9088
22016/60000 [==========>...................] - ETA: 1:13 - loss: 0.2990 - categorical_accuracy: 0.9088
22048/60000 [==========>...................] - ETA: 1:13 - loss: 0.2986 - categorical_accuracy: 0.9090
22080/60000 [==========>...................] - ETA: 1:13 - loss: 0.2983 - categorical_accuracy: 0.9091
22112/60000 [==========>...................] - ETA: 1:13 - loss: 0.2980 - categorical_accuracy: 0.9091
22144/60000 [==========>...................] - ETA: 1:13 - loss: 0.2982 - categorical_accuracy: 0.9090
22176/60000 [==========>...................] - ETA: 1:13 - loss: 0.2980 - categorical_accuracy: 0.9091
22208/60000 [==========>...................] - ETA: 1:13 - loss: 0.2977 - categorical_accuracy: 0.9091
22240/60000 [==========>...................] - ETA: 1:13 - loss: 0.2975 - categorical_accuracy: 0.9092
22272/60000 [==========>...................] - ETA: 1:13 - loss: 0.2971 - categorical_accuracy: 0.9093
22304/60000 [==========>...................] - ETA: 1:13 - loss: 0.2970 - categorical_accuracy: 0.9094
22336/60000 [==========>...................] - ETA: 1:13 - loss: 0.2966 - categorical_accuracy: 0.9096
22368/60000 [==========>...................] - ETA: 1:13 - loss: 0.2963 - categorical_accuracy: 0.9096
22400/60000 [==========>...................] - ETA: 1:13 - loss: 0.2960 - categorical_accuracy: 0.9098
22432/60000 [==========>...................] - ETA: 1:13 - loss: 0.2957 - categorical_accuracy: 0.9099
22464/60000 [==========>...................] - ETA: 1:13 - loss: 0.2956 - categorical_accuracy: 0.9099
22496/60000 [==========>...................] - ETA: 1:13 - loss: 0.2954 - categorical_accuracy: 0.9100
22528/60000 [==========>...................] - ETA: 1:12 - loss: 0.2953 - categorical_accuracy: 0.9100
22560/60000 [==========>...................] - ETA: 1:12 - loss: 0.2953 - categorical_accuracy: 0.9101
22592/60000 [==========>...................] - ETA: 1:12 - loss: 0.2949 - categorical_accuracy: 0.9102
22624/60000 [==========>...................] - ETA: 1:12 - loss: 0.2947 - categorical_accuracy: 0.9102
22656/60000 [==========>...................] - ETA: 1:12 - loss: 0.2944 - categorical_accuracy: 0.9103
22688/60000 [==========>...................] - ETA: 1:12 - loss: 0.2941 - categorical_accuracy: 0.9103
22720/60000 [==========>...................] - ETA: 1:12 - loss: 0.2937 - categorical_accuracy: 0.9104
22752/60000 [==========>...................] - ETA: 1:12 - loss: 0.2936 - categorical_accuracy: 0.9105
22784/60000 [==========>...................] - ETA: 1:12 - loss: 0.2933 - categorical_accuracy: 0.9106
22816/60000 [==========>...................] - ETA: 1:12 - loss: 0.2933 - categorical_accuracy: 0.9106
22848/60000 [==========>...................] - ETA: 1:12 - loss: 0.2933 - categorical_accuracy: 0.9105
22880/60000 [==========>...................] - ETA: 1:12 - loss: 0.2934 - categorical_accuracy: 0.9106
22912/60000 [==========>...................] - ETA: 1:12 - loss: 0.2931 - categorical_accuracy: 0.9107
22944/60000 [==========>...................] - ETA: 1:12 - loss: 0.2930 - categorical_accuracy: 0.9107
22976/60000 [==========>...................] - ETA: 1:12 - loss: 0.2929 - categorical_accuracy: 0.9108
23008/60000 [==========>...................] - ETA: 1:12 - loss: 0.2929 - categorical_accuracy: 0.9107
23040/60000 [==========>...................] - ETA: 1:11 - loss: 0.2927 - categorical_accuracy: 0.9107
23072/60000 [==========>...................] - ETA: 1:11 - loss: 0.2923 - categorical_accuracy: 0.9108
23104/60000 [==========>...................] - ETA: 1:11 - loss: 0.2922 - categorical_accuracy: 0.9108
23136/60000 [==========>...................] - ETA: 1:11 - loss: 0.2923 - categorical_accuracy: 0.9109
23168/60000 [==========>...................] - ETA: 1:11 - loss: 0.2920 - categorical_accuracy: 0.9110
23200/60000 [==========>...................] - ETA: 1:11 - loss: 0.2917 - categorical_accuracy: 0.9111
23232/60000 [==========>...................] - ETA: 1:11 - loss: 0.2913 - categorical_accuracy: 0.9112
23264/60000 [==========>...................] - ETA: 1:11 - loss: 0.2912 - categorical_accuracy: 0.9112
23296/60000 [==========>...................] - ETA: 1:11 - loss: 0.2910 - categorical_accuracy: 0.9112
23328/60000 [==========>...................] - ETA: 1:11 - loss: 0.2907 - categorical_accuracy: 0.9113
23360/60000 [==========>...................] - ETA: 1:11 - loss: 0.2907 - categorical_accuracy: 0.9113
23392/60000 [==========>...................] - ETA: 1:11 - loss: 0.2905 - categorical_accuracy: 0.9113
23424/60000 [==========>...................] - ETA: 1:11 - loss: 0.2901 - categorical_accuracy: 0.9115
23456/60000 [==========>...................] - ETA: 1:11 - loss: 0.2900 - categorical_accuracy: 0.9115
23488/60000 [==========>...................] - ETA: 1:11 - loss: 0.2898 - categorical_accuracy: 0.9115
23520/60000 [==========>...................] - ETA: 1:11 - loss: 0.2894 - categorical_accuracy: 0.9116
23552/60000 [==========>...................] - ETA: 1:10 - loss: 0.2891 - categorical_accuracy: 0.9118
23584/60000 [==========>...................] - ETA: 1:10 - loss: 0.2891 - categorical_accuracy: 0.9118
23616/60000 [==========>...................] - ETA: 1:10 - loss: 0.2891 - categorical_accuracy: 0.9118
23648/60000 [==========>...................] - ETA: 1:10 - loss: 0.2888 - categorical_accuracy: 0.9118
23680/60000 [==========>...................] - ETA: 1:10 - loss: 0.2885 - categorical_accuracy: 0.9119
23712/60000 [==========>...................] - ETA: 1:10 - loss: 0.2885 - categorical_accuracy: 0.9119
23744/60000 [==========>...................] - ETA: 1:10 - loss: 0.2883 - categorical_accuracy: 0.9120
23776/60000 [==========>...................] - ETA: 1:10 - loss: 0.2881 - categorical_accuracy: 0.9121
23808/60000 [==========>...................] - ETA: 1:10 - loss: 0.2881 - categorical_accuracy: 0.9121
23840/60000 [==========>...................] - ETA: 1:10 - loss: 0.2878 - categorical_accuracy: 0.9122
23872/60000 [==========>...................] - ETA: 1:10 - loss: 0.2874 - categorical_accuracy: 0.9123
23904/60000 [==========>...................] - ETA: 1:10 - loss: 0.2875 - categorical_accuracy: 0.9123
23936/60000 [==========>...................] - ETA: 1:10 - loss: 0.2874 - categorical_accuracy: 0.9123
23968/60000 [==========>...................] - ETA: 1:10 - loss: 0.2876 - categorical_accuracy: 0.9123
24000/60000 [===========>..................] - ETA: 1:10 - loss: 0.2873 - categorical_accuracy: 0.9123
24032/60000 [===========>..................] - ETA: 1:10 - loss: 0.2875 - categorical_accuracy: 0.9124
24064/60000 [===========>..................] - ETA: 1:10 - loss: 0.2874 - categorical_accuracy: 0.9124
24096/60000 [===========>..................] - ETA: 1:09 - loss: 0.2871 - categorical_accuracy: 0.9125
24128/60000 [===========>..................] - ETA: 1:09 - loss: 0.2868 - categorical_accuracy: 0.9126
24160/60000 [===========>..................] - ETA: 1:09 - loss: 0.2866 - categorical_accuracy: 0.9126
24192/60000 [===========>..................] - ETA: 1:09 - loss: 0.2863 - categorical_accuracy: 0.9127
24224/60000 [===========>..................] - ETA: 1:09 - loss: 0.2860 - categorical_accuracy: 0.9128
24256/60000 [===========>..................] - ETA: 1:09 - loss: 0.2860 - categorical_accuracy: 0.9128
24288/60000 [===========>..................] - ETA: 1:09 - loss: 0.2862 - categorical_accuracy: 0.9128
24320/60000 [===========>..................] - ETA: 1:09 - loss: 0.2859 - categorical_accuracy: 0.9129
24352/60000 [===========>..................] - ETA: 1:09 - loss: 0.2856 - categorical_accuracy: 0.9130
24384/60000 [===========>..................] - ETA: 1:09 - loss: 0.2852 - categorical_accuracy: 0.9131
24416/60000 [===========>..................] - ETA: 1:09 - loss: 0.2850 - categorical_accuracy: 0.9132
24448/60000 [===========>..................] - ETA: 1:09 - loss: 0.2850 - categorical_accuracy: 0.9132
24480/60000 [===========>..................] - ETA: 1:09 - loss: 0.2849 - categorical_accuracy: 0.9132
24512/60000 [===========>..................] - ETA: 1:09 - loss: 0.2849 - categorical_accuracy: 0.9132
24544/60000 [===========>..................] - ETA: 1:09 - loss: 0.2849 - categorical_accuracy: 0.9132
24576/60000 [===========>..................] - ETA: 1:08 - loss: 0.2849 - categorical_accuracy: 0.9132
24608/60000 [===========>..................] - ETA: 1:08 - loss: 0.2847 - categorical_accuracy: 0.9133
24640/60000 [===========>..................] - ETA: 1:08 - loss: 0.2846 - categorical_accuracy: 0.9133
24672/60000 [===========>..................] - ETA: 1:08 - loss: 0.2844 - categorical_accuracy: 0.9134
24704/60000 [===========>..................] - ETA: 1:08 - loss: 0.2841 - categorical_accuracy: 0.9135
24736/60000 [===========>..................] - ETA: 1:08 - loss: 0.2839 - categorical_accuracy: 0.9136
24768/60000 [===========>..................] - ETA: 1:08 - loss: 0.2836 - categorical_accuracy: 0.9136
24800/60000 [===========>..................] - ETA: 1:08 - loss: 0.2834 - categorical_accuracy: 0.9137
24832/60000 [===========>..................] - ETA: 1:08 - loss: 0.2831 - categorical_accuracy: 0.9138
24864/60000 [===========>..................] - ETA: 1:08 - loss: 0.2827 - categorical_accuracy: 0.9139
24896/60000 [===========>..................] - ETA: 1:08 - loss: 0.2825 - categorical_accuracy: 0.9140
24928/60000 [===========>..................] - ETA: 1:08 - loss: 0.2822 - categorical_accuracy: 0.9141
24960/60000 [===========>..................] - ETA: 1:08 - loss: 0.2820 - categorical_accuracy: 0.9141
24992/60000 [===========>..................] - ETA: 1:08 - loss: 0.2816 - categorical_accuracy: 0.9142
25024/60000 [===========>..................] - ETA: 1:08 - loss: 0.2814 - categorical_accuracy: 0.9143
25056/60000 [===========>..................] - ETA: 1:08 - loss: 0.2812 - categorical_accuracy: 0.9143
25088/60000 [===========>..................] - ETA: 1:07 - loss: 0.2810 - categorical_accuracy: 0.9143
25120/60000 [===========>..................] - ETA: 1:07 - loss: 0.2807 - categorical_accuracy: 0.9144
25152/60000 [===========>..................] - ETA: 1:07 - loss: 0.2807 - categorical_accuracy: 0.9144
25184/60000 [===========>..................] - ETA: 1:07 - loss: 0.2804 - categorical_accuracy: 0.9145
25216/60000 [===========>..................] - ETA: 1:07 - loss: 0.2804 - categorical_accuracy: 0.9145
25248/60000 [===========>..................] - ETA: 1:07 - loss: 0.2800 - categorical_accuracy: 0.9146
25280/60000 [===========>..................] - ETA: 1:07 - loss: 0.2803 - categorical_accuracy: 0.9146
25312/60000 [===========>..................] - ETA: 1:07 - loss: 0.2802 - categorical_accuracy: 0.9146
25344/60000 [===========>..................] - ETA: 1:07 - loss: 0.2799 - categorical_accuracy: 0.9147
25376/60000 [===========>..................] - ETA: 1:07 - loss: 0.2796 - categorical_accuracy: 0.9148
25408/60000 [===========>..................] - ETA: 1:07 - loss: 0.2794 - categorical_accuracy: 0.9149
25440/60000 [===========>..................] - ETA: 1:07 - loss: 0.2791 - categorical_accuracy: 0.9149
25472/60000 [===========>..................] - ETA: 1:07 - loss: 0.2790 - categorical_accuracy: 0.9150
25504/60000 [===========>..................] - ETA: 1:07 - loss: 0.2787 - categorical_accuracy: 0.9151
25536/60000 [===========>..................] - ETA: 1:07 - loss: 0.2787 - categorical_accuracy: 0.9150
25568/60000 [===========>..................] - ETA: 1:07 - loss: 0.2790 - categorical_accuracy: 0.9150
25600/60000 [===========>..................] - ETA: 1:06 - loss: 0.2792 - categorical_accuracy: 0.9150
25632/60000 [===========>..................] - ETA: 1:06 - loss: 0.2792 - categorical_accuracy: 0.9150
25664/60000 [===========>..................] - ETA: 1:06 - loss: 0.2791 - categorical_accuracy: 0.9151
25696/60000 [===========>..................] - ETA: 1:06 - loss: 0.2788 - categorical_accuracy: 0.9152
25728/60000 [===========>..................] - ETA: 1:06 - loss: 0.2785 - categorical_accuracy: 0.9153
25760/60000 [===========>..................] - ETA: 1:06 - loss: 0.2783 - categorical_accuracy: 0.9153
25792/60000 [===========>..................] - ETA: 1:06 - loss: 0.2782 - categorical_accuracy: 0.9153
25824/60000 [===========>..................] - ETA: 1:06 - loss: 0.2780 - categorical_accuracy: 0.9154
25856/60000 [===========>..................] - ETA: 1:06 - loss: 0.2777 - categorical_accuracy: 0.9154
25888/60000 [===========>..................] - ETA: 1:06 - loss: 0.2776 - categorical_accuracy: 0.9155
25920/60000 [===========>..................] - ETA: 1:06 - loss: 0.2775 - categorical_accuracy: 0.9155
25952/60000 [===========>..................] - ETA: 1:06 - loss: 0.2773 - categorical_accuracy: 0.9155
25984/60000 [===========>..................] - ETA: 1:06 - loss: 0.2770 - categorical_accuracy: 0.9156
26016/60000 [============>.................] - ETA: 1:06 - loss: 0.2769 - categorical_accuracy: 0.9156
26048/60000 [============>.................] - ETA: 1:06 - loss: 0.2767 - categorical_accuracy: 0.9157
26080/60000 [============>.................] - ETA: 1:06 - loss: 0.2765 - categorical_accuracy: 0.9158
26112/60000 [============>.................] - ETA: 1:05 - loss: 0.2765 - categorical_accuracy: 0.9158
26144/60000 [============>.................] - ETA: 1:05 - loss: 0.2763 - categorical_accuracy: 0.9159
26176/60000 [============>.................] - ETA: 1:05 - loss: 0.2760 - categorical_accuracy: 0.9160
26208/60000 [============>.................] - ETA: 1:05 - loss: 0.2757 - categorical_accuracy: 0.9161
26240/60000 [============>.................] - ETA: 1:05 - loss: 0.2756 - categorical_accuracy: 0.9162
26272/60000 [============>.................] - ETA: 1:05 - loss: 0.2754 - categorical_accuracy: 0.9162
26304/60000 [============>.................] - ETA: 1:05 - loss: 0.2752 - categorical_accuracy: 0.9162
26336/60000 [============>.................] - ETA: 1:05 - loss: 0.2753 - categorical_accuracy: 0.9162
26368/60000 [============>.................] - ETA: 1:05 - loss: 0.2750 - categorical_accuracy: 0.9163
26400/60000 [============>.................] - ETA: 1:05 - loss: 0.2747 - categorical_accuracy: 0.9164
26432/60000 [============>.................] - ETA: 1:05 - loss: 0.2745 - categorical_accuracy: 0.9165
26464/60000 [============>.................] - ETA: 1:05 - loss: 0.2742 - categorical_accuracy: 0.9166
26496/60000 [============>.................] - ETA: 1:05 - loss: 0.2740 - categorical_accuracy: 0.9166
26528/60000 [============>.................] - ETA: 1:05 - loss: 0.2741 - categorical_accuracy: 0.9167
26560/60000 [============>.................] - ETA: 1:05 - loss: 0.2738 - categorical_accuracy: 0.9167
26592/60000 [============>.................] - ETA: 1:04 - loss: 0.2735 - categorical_accuracy: 0.9168
26624/60000 [============>.................] - ETA: 1:04 - loss: 0.2734 - categorical_accuracy: 0.9168
26656/60000 [============>.................] - ETA: 1:04 - loss: 0.2735 - categorical_accuracy: 0.9169
26688/60000 [============>.................] - ETA: 1:04 - loss: 0.2732 - categorical_accuracy: 0.9169
26720/60000 [============>.................] - ETA: 1:04 - loss: 0.2731 - categorical_accuracy: 0.9169
26752/60000 [============>.................] - ETA: 1:04 - loss: 0.2730 - categorical_accuracy: 0.9169
26784/60000 [============>.................] - ETA: 1:04 - loss: 0.2728 - categorical_accuracy: 0.9170
26816/60000 [============>.................] - ETA: 1:04 - loss: 0.2728 - categorical_accuracy: 0.9171
26848/60000 [============>.................] - ETA: 1:04 - loss: 0.2726 - categorical_accuracy: 0.9171
26880/60000 [============>.................] - ETA: 1:04 - loss: 0.2726 - categorical_accuracy: 0.9171
26912/60000 [============>.................] - ETA: 1:04 - loss: 0.2725 - categorical_accuracy: 0.9170
26944/60000 [============>.................] - ETA: 1:04 - loss: 0.2724 - categorical_accuracy: 0.9171
26976/60000 [============>.................] - ETA: 1:04 - loss: 0.2723 - categorical_accuracy: 0.9171
27008/60000 [============>.................] - ETA: 1:04 - loss: 0.2720 - categorical_accuracy: 0.9172
27040/60000 [============>.................] - ETA: 1:04 - loss: 0.2719 - categorical_accuracy: 0.9173
27072/60000 [============>.................] - ETA: 1:04 - loss: 0.2718 - categorical_accuracy: 0.9173
27104/60000 [============>.................] - ETA: 1:03 - loss: 0.2717 - categorical_accuracy: 0.9173
27136/60000 [============>.................] - ETA: 1:03 - loss: 0.2714 - categorical_accuracy: 0.9174
27168/60000 [============>.................] - ETA: 1:03 - loss: 0.2712 - categorical_accuracy: 0.9174
27200/60000 [============>.................] - ETA: 1:03 - loss: 0.2710 - categorical_accuracy: 0.9175
27232/60000 [============>.................] - ETA: 1:03 - loss: 0.2709 - categorical_accuracy: 0.9175
27264/60000 [============>.................] - ETA: 1:03 - loss: 0.2706 - categorical_accuracy: 0.9176
27296/60000 [============>.................] - ETA: 1:03 - loss: 0.2705 - categorical_accuracy: 0.9176
27328/60000 [============>.................] - ETA: 1:03 - loss: 0.2703 - categorical_accuracy: 0.9177
27360/60000 [============>.................] - ETA: 1:03 - loss: 0.2700 - categorical_accuracy: 0.9178
27392/60000 [============>.................] - ETA: 1:03 - loss: 0.2699 - categorical_accuracy: 0.9178
27424/60000 [============>.................] - ETA: 1:03 - loss: 0.2697 - categorical_accuracy: 0.9178
27456/60000 [============>.................] - ETA: 1:03 - loss: 0.2695 - categorical_accuracy: 0.9179
27488/60000 [============>.................] - ETA: 1:03 - loss: 0.2695 - categorical_accuracy: 0.9179
27520/60000 [============>.................] - ETA: 1:03 - loss: 0.2692 - categorical_accuracy: 0.9180
27552/60000 [============>.................] - ETA: 1:03 - loss: 0.2690 - categorical_accuracy: 0.9181
27584/60000 [============>.................] - ETA: 1:02 - loss: 0.2688 - categorical_accuracy: 0.9181
27616/60000 [============>.................] - ETA: 1:02 - loss: 0.2686 - categorical_accuracy: 0.9182
27648/60000 [============>.................] - ETA: 1:02 - loss: 0.2685 - categorical_accuracy: 0.9183
27680/60000 [============>.................] - ETA: 1:02 - loss: 0.2683 - categorical_accuracy: 0.9183
27712/60000 [============>.................] - ETA: 1:02 - loss: 0.2681 - categorical_accuracy: 0.9184
27744/60000 [============>.................] - ETA: 1:02 - loss: 0.2679 - categorical_accuracy: 0.9185
27776/60000 [============>.................] - ETA: 1:02 - loss: 0.2677 - categorical_accuracy: 0.9185
27808/60000 [============>.................] - ETA: 1:02 - loss: 0.2675 - categorical_accuracy: 0.9186
27840/60000 [============>.................] - ETA: 1:02 - loss: 0.2673 - categorical_accuracy: 0.9186
27872/60000 [============>.................] - ETA: 1:02 - loss: 0.2671 - categorical_accuracy: 0.9187
27904/60000 [============>.................] - ETA: 1:02 - loss: 0.2669 - categorical_accuracy: 0.9187
27936/60000 [============>.................] - ETA: 1:02 - loss: 0.2666 - categorical_accuracy: 0.9188
27968/60000 [============>.................] - ETA: 1:02 - loss: 0.2664 - categorical_accuracy: 0.9189
28000/60000 [=============>................] - ETA: 1:02 - loss: 0.2662 - categorical_accuracy: 0.9189
28032/60000 [=============>................] - ETA: 1:02 - loss: 0.2660 - categorical_accuracy: 0.9190
28064/60000 [=============>................] - ETA: 1:02 - loss: 0.2658 - categorical_accuracy: 0.9191
28096/60000 [=============>................] - ETA: 1:02 - loss: 0.2656 - categorical_accuracy: 0.9191
28128/60000 [=============>................] - ETA: 1:01 - loss: 0.2656 - categorical_accuracy: 0.9191
28160/60000 [=============>................] - ETA: 1:01 - loss: 0.2655 - categorical_accuracy: 0.9191
28192/60000 [=============>................] - ETA: 1:01 - loss: 0.2653 - categorical_accuracy: 0.9192
28224/60000 [=============>................] - ETA: 1:01 - loss: 0.2653 - categorical_accuracy: 0.9193
28256/60000 [=============>................] - ETA: 1:01 - loss: 0.2651 - categorical_accuracy: 0.9193
28288/60000 [=============>................] - ETA: 1:01 - loss: 0.2649 - categorical_accuracy: 0.9194
28320/60000 [=============>................] - ETA: 1:01 - loss: 0.2648 - categorical_accuracy: 0.9195
28352/60000 [=============>................] - ETA: 1:01 - loss: 0.2647 - categorical_accuracy: 0.9195
28384/60000 [=============>................] - ETA: 1:01 - loss: 0.2647 - categorical_accuracy: 0.9195
28416/60000 [=============>................] - ETA: 1:01 - loss: 0.2644 - categorical_accuracy: 0.9196
28448/60000 [=============>................] - ETA: 1:01 - loss: 0.2644 - categorical_accuracy: 0.9196
28480/60000 [=============>................] - ETA: 1:01 - loss: 0.2642 - categorical_accuracy: 0.9197
28512/60000 [=============>................] - ETA: 1:01 - loss: 0.2639 - categorical_accuracy: 0.9198
28544/60000 [=============>................] - ETA: 1:01 - loss: 0.2637 - categorical_accuracy: 0.9199
28576/60000 [=============>................] - ETA: 1:01 - loss: 0.2634 - categorical_accuracy: 0.9200
28608/60000 [=============>................] - ETA: 1:00 - loss: 0.2633 - categorical_accuracy: 0.9200
28640/60000 [=============>................] - ETA: 1:00 - loss: 0.2631 - categorical_accuracy: 0.9200
28672/60000 [=============>................] - ETA: 1:00 - loss: 0.2631 - categorical_accuracy: 0.9201
28704/60000 [=============>................] - ETA: 1:00 - loss: 0.2630 - categorical_accuracy: 0.9201
28736/60000 [=============>................] - ETA: 1:00 - loss: 0.2629 - categorical_accuracy: 0.9202
28768/60000 [=============>................] - ETA: 1:00 - loss: 0.2626 - categorical_accuracy: 0.9203
28800/60000 [=============>................] - ETA: 1:00 - loss: 0.2624 - categorical_accuracy: 0.9203
28832/60000 [=============>................] - ETA: 1:00 - loss: 0.2621 - categorical_accuracy: 0.9204
28864/60000 [=============>................] - ETA: 1:00 - loss: 0.2619 - categorical_accuracy: 0.9205
28896/60000 [=============>................] - ETA: 1:00 - loss: 0.2619 - categorical_accuracy: 0.9205
28928/60000 [=============>................] - ETA: 1:00 - loss: 0.2616 - categorical_accuracy: 0.9206
28960/60000 [=============>................] - ETA: 1:00 - loss: 0.2614 - categorical_accuracy: 0.9206
28992/60000 [=============>................] - ETA: 1:00 - loss: 0.2612 - categorical_accuracy: 0.9207
29024/60000 [=============>................] - ETA: 1:00 - loss: 0.2609 - categorical_accuracy: 0.9208
29056/60000 [=============>................] - ETA: 1:00 - loss: 0.2607 - categorical_accuracy: 0.9209
29088/60000 [=============>................] - ETA: 1:00 - loss: 0.2605 - categorical_accuracy: 0.9209
29120/60000 [=============>................] - ETA: 59s - loss: 0.2603 - categorical_accuracy: 0.9210 
29152/60000 [=============>................] - ETA: 59s - loss: 0.2602 - categorical_accuracy: 0.9210
29184/60000 [=============>................] - ETA: 59s - loss: 0.2599 - categorical_accuracy: 0.9211
29216/60000 [=============>................] - ETA: 59s - loss: 0.2597 - categorical_accuracy: 0.9211
29248/60000 [=============>................] - ETA: 59s - loss: 0.2594 - categorical_accuracy: 0.9212
29280/60000 [=============>................] - ETA: 59s - loss: 0.2592 - categorical_accuracy: 0.9213
29312/60000 [=============>................] - ETA: 59s - loss: 0.2590 - categorical_accuracy: 0.9213
29344/60000 [=============>................] - ETA: 59s - loss: 0.2589 - categorical_accuracy: 0.9213
29376/60000 [=============>................] - ETA: 59s - loss: 0.2587 - categorical_accuracy: 0.9214
29408/60000 [=============>................] - ETA: 59s - loss: 0.2584 - categorical_accuracy: 0.9215
29440/60000 [=============>................] - ETA: 59s - loss: 0.2582 - categorical_accuracy: 0.9215
29472/60000 [=============>................] - ETA: 59s - loss: 0.2583 - categorical_accuracy: 0.9216
29504/60000 [=============>................] - ETA: 59s - loss: 0.2581 - categorical_accuracy: 0.9216
29536/60000 [=============>................] - ETA: 59s - loss: 0.2578 - categorical_accuracy: 0.9217
29568/60000 [=============>................] - ETA: 59s - loss: 0.2577 - categorical_accuracy: 0.9218
29600/60000 [=============>................] - ETA: 59s - loss: 0.2576 - categorical_accuracy: 0.9218
29632/60000 [=============>................] - ETA: 58s - loss: 0.2574 - categorical_accuracy: 0.9219
29664/60000 [=============>................] - ETA: 58s - loss: 0.2571 - categorical_accuracy: 0.9219
29696/60000 [=============>................] - ETA: 58s - loss: 0.2573 - categorical_accuracy: 0.9219
29728/60000 [=============>................] - ETA: 58s - loss: 0.2571 - categorical_accuracy: 0.9220
29760/60000 [=============>................] - ETA: 58s - loss: 0.2570 - categorical_accuracy: 0.9220
29792/60000 [=============>................] - ETA: 58s - loss: 0.2568 - categorical_accuracy: 0.9221
29824/60000 [=============>................] - ETA: 58s - loss: 0.2566 - categorical_accuracy: 0.9221
29856/60000 [=============>................] - ETA: 58s - loss: 0.2563 - categorical_accuracy: 0.9222
29888/60000 [=============>................] - ETA: 58s - loss: 0.2563 - categorical_accuracy: 0.9222
29920/60000 [=============>................] - ETA: 58s - loss: 0.2567 - categorical_accuracy: 0.9221
29952/60000 [=============>................] - ETA: 58s - loss: 0.2566 - categorical_accuracy: 0.9221
29984/60000 [=============>................] - ETA: 58s - loss: 0.2565 - categorical_accuracy: 0.9222
30016/60000 [==============>...............] - ETA: 58s - loss: 0.2563 - categorical_accuracy: 0.9222
30048/60000 [==============>...............] - ETA: 58s - loss: 0.2561 - categorical_accuracy: 0.9223
30080/60000 [==============>...............] - ETA: 58s - loss: 0.2561 - categorical_accuracy: 0.9223
30112/60000 [==============>...............] - ETA: 58s - loss: 0.2560 - categorical_accuracy: 0.9223
30144/60000 [==============>...............] - ETA: 57s - loss: 0.2559 - categorical_accuracy: 0.9224
30176/60000 [==============>...............] - ETA: 57s - loss: 0.2558 - categorical_accuracy: 0.9224
30208/60000 [==============>...............] - ETA: 57s - loss: 0.2556 - categorical_accuracy: 0.9224
30240/60000 [==============>...............] - ETA: 57s - loss: 0.2554 - categorical_accuracy: 0.9225
30272/60000 [==============>...............] - ETA: 57s - loss: 0.2553 - categorical_accuracy: 0.9226
30304/60000 [==============>...............] - ETA: 57s - loss: 0.2552 - categorical_accuracy: 0.9226
30336/60000 [==============>...............] - ETA: 57s - loss: 0.2551 - categorical_accuracy: 0.9226
30368/60000 [==============>...............] - ETA: 57s - loss: 0.2549 - categorical_accuracy: 0.9227
30400/60000 [==============>...............] - ETA: 57s - loss: 0.2546 - categorical_accuracy: 0.9228
30432/60000 [==============>...............] - ETA: 57s - loss: 0.2546 - categorical_accuracy: 0.9227
30464/60000 [==============>...............] - ETA: 57s - loss: 0.2545 - categorical_accuracy: 0.9228
30496/60000 [==============>...............] - ETA: 57s - loss: 0.2544 - categorical_accuracy: 0.9228
30528/60000 [==============>...............] - ETA: 57s - loss: 0.2541 - categorical_accuracy: 0.9229
30560/60000 [==============>...............] - ETA: 57s - loss: 0.2540 - categorical_accuracy: 0.9229
30592/60000 [==============>...............] - ETA: 57s - loss: 0.2539 - categorical_accuracy: 0.9229
30624/60000 [==============>...............] - ETA: 57s - loss: 0.2537 - categorical_accuracy: 0.9230
30656/60000 [==============>...............] - ETA: 56s - loss: 0.2537 - categorical_accuracy: 0.9230
30688/60000 [==============>...............] - ETA: 56s - loss: 0.2535 - categorical_accuracy: 0.9230
30720/60000 [==============>...............] - ETA: 56s - loss: 0.2533 - categorical_accuracy: 0.9230
30752/60000 [==============>...............] - ETA: 56s - loss: 0.2531 - categorical_accuracy: 0.9231
30784/60000 [==============>...............] - ETA: 56s - loss: 0.2529 - categorical_accuracy: 0.9231
30816/60000 [==============>...............] - ETA: 56s - loss: 0.2528 - categorical_accuracy: 0.9232
30848/60000 [==============>...............] - ETA: 56s - loss: 0.2526 - categorical_accuracy: 0.9233
30880/60000 [==============>...............] - ETA: 56s - loss: 0.2523 - categorical_accuracy: 0.9233
30912/60000 [==============>...............] - ETA: 56s - loss: 0.2521 - categorical_accuracy: 0.9234
30944/60000 [==============>...............] - ETA: 56s - loss: 0.2520 - categorical_accuracy: 0.9234
30976/60000 [==============>...............] - ETA: 56s - loss: 0.2520 - categorical_accuracy: 0.9235
31008/60000 [==============>...............] - ETA: 56s - loss: 0.2521 - categorical_accuracy: 0.9235
31040/60000 [==============>...............] - ETA: 56s - loss: 0.2522 - categorical_accuracy: 0.9235
31072/60000 [==============>...............] - ETA: 56s - loss: 0.2521 - categorical_accuracy: 0.9235
31104/60000 [==============>...............] - ETA: 56s - loss: 0.2522 - categorical_accuracy: 0.9235
31136/60000 [==============>...............] - ETA: 56s - loss: 0.2522 - categorical_accuracy: 0.9234
31168/60000 [==============>...............] - ETA: 55s - loss: 0.2520 - categorical_accuracy: 0.9235
31200/60000 [==============>...............] - ETA: 55s - loss: 0.2520 - categorical_accuracy: 0.9235
31232/60000 [==============>...............] - ETA: 55s - loss: 0.2518 - categorical_accuracy: 0.9235
31264/60000 [==============>...............] - ETA: 55s - loss: 0.2517 - categorical_accuracy: 0.9236
31296/60000 [==============>...............] - ETA: 55s - loss: 0.2516 - categorical_accuracy: 0.9236
31328/60000 [==============>...............] - ETA: 55s - loss: 0.2517 - categorical_accuracy: 0.9236
31360/60000 [==============>...............] - ETA: 55s - loss: 0.2515 - categorical_accuracy: 0.9236
31392/60000 [==============>...............] - ETA: 55s - loss: 0.2513 - categorical_accuracy: 0.9237
31424/60000 [==============>...............] - ETA: 55s - loss: 0.2515 - categorical_accuracy: 0.9237
31456/60000 [==============>...............] - ETA: 55s - loss: 0.2512 - categorical_accuracy: 0.9238
31488/60000 [==============>...............] - ETA: 55s - loss: 0.2511 - categorical_accuracy: 0.9238
31520/60000 [==============>...............] - ETA: 55s - loss: 0.2509 - categorical_accuracy: 0.9239
31552/60000 [==============>...............] - ETA: 55s - loss: 0.2508 - categorical_accuracy: 0.9239
31584/60000 [==============>...............] - ETA: 55s - loss: 0.2508 - categorical_accuracy: 0.9239
31616/60000 [==============>...............] - ETA: 55s - loss: 0.2507 - categorical_accuracy: 0.9239
31648/60000 [==============>...............] - ETA: 55s - loss: 0.2505 - categorical_accuracy: 0.9240
31680/60000 [==============>...............] - ETA: 54s - loss: 0.2505 - categorical_accuracy: 0.9240
31712/60000 [==============>...............] - ETA: 54s - loss: 0.2503 - categorical_accuracy: 0.9241
31744/60000 [==============>...............] - ETA: 54s - loss: 0.2501 - categorical_accuracy: 0.9241
31776/60000 [==============>...............] - ETA: 54s - loss: 0.2499 - categorical_accuracy: 0.9242
31808/60000 [==============>...............] - ETA: 54s - loss: 0.2496 - categorical_accuracy: 0.9242
31840/60000 [==============>...............] - ETA: 54s - loss: 0.2497 - categorical_accuracy: 0.9243
31872/60000 [==============>...............] - ETA: 54s - loss: 0.2496 - categorical_accuracy: 0.9243
31904/60000 [==============>...............] - ETA: 54s - loss: 0.2493 - categorical_accuracy: 0.9244
31936/60000 [==============>...............] - ETA: 54s - loss: 0.2492 - categorical_accuracy: 0.9245
31968/60000 [==============>...............] - ETA: 54s - loss: 0.2490 - categorical_accuracy: 0.9245
32000/60000 [===============>..............] - ETA: 54s - loss: 0.2489 - categorical_accuracy: 0.9245
32032/60000 [===============>..............] - ETA: 54s - loss: 0.2488 - categorical_accuracy: 0.9246
32064/60000 [===============>..............] - ETA: 54s - loss: 0.2488 - categorical_accuracy: 0.9246
32096/60000 [===============>..............] - ETA: 54s - loss: 0.2489 - categorical_accuracy: 0.9246
32128/60000 [===============>..............] - ETA: 54s - loss: 0.2487 - categorical_accuracy: 0.9246
32160/60000 [===============>..............] - ETA: 54s - loss: 0.2486 - categorical_accuracy: 0.9247
32192/60000 [===============>..............] - ETA: 53s - loss: 0.2483 - categorical_accuracy: 0.9247
32224/60000 [===============>..............] - ETA: 53s - loss: 0.2485 - categorical_accuracy: 0.9247
32256/60000 [===============>..............] - ETA: 53s - loss: 0.2483 - categorical_accuracy: 0.9248
32288/60000 [===============>..............] - ETA: 53s - loss: 0.2481 - categorical_accuracy: 0.9248
32320/60000 [===============>..............] - ETA: 53s - loss: 0.2480 - categorical_accuracy: 0.9249
32352/60000 [===============>..............] - ETA: 53s - loss: 0.2478 - categorical_accuracy: 0.9250
32384/60000 [===============>..............] - ETA: 53s - loss: 0.2477 - categorical_accuracy: 0.9250
32416/60000 [===============>..............] - ETA: 53s - loss: 0.2475 - categorical_accuracy: 0.9250
32448/60000 [===============>..............] - ETA: 53s - loss: 0.2473 - categorical_accuracy: 0.9251
32480/60000 [===============>..............] - ETA: 53s - loss: 0.2471 - categorical_accuracy: 0.9252
32512/60000 [===============>..............] - ETA: 53s - loss: 0.2469 - categorical_accuracy: 0.9252
32544/60000 [===============>..............] - ETA: 53s - loss: 0.2467 - categorical_accuracy: 0.9253
32576/60000 [===============>..............] - ETA: 53s - loss: 0.2466 - categorical_accuracy: 0.9253
32608/60000 [===============>..............] - ETA: 53s - loss: 0.2465 - categorical_accuracy: 0.9254
32640/60000 [===============>..............] - ETA: 53s - loss: 0.2463 - categorical_accuracy: 0.9254
32672/60000 [===============>..............] - ETA: 53s - loss: 0.2461 - categorical_accuracy: 0.9254
32704/60000 [===============>..............] - ETA: 52s - loss: 0.2458 - categorical_accuracy: 0.9255
32736/60000 [===============>..............] - ETA: 52s - loss: 0.2462 - categorical_accuracy: 0.9255
32768/60000 [===============>..............] - ETA: 52s - loss: 0.2459 - categorical_accuracy: 0.9256
32800/60000 [===============>..............] - ETA: 52s - loss: 0.2459 - categorical_accuracy: 0.9256
32832/60000 [===============>..............] - ETA: 52s - loss: 0.2459 - categorical_accuracy: 0.9255
32864/60000 [===============>..............] - ETA: 52s - loss: 0.2457 - categorical_accuracy: 0.9256
32896/60000 [===============>..............] - ETA: 52s - loss: 0.2456 - categorical_accuracy: 0.9256
32928/60000 [===============>..............] - ETA: 52s - loss: 0.2454 - categorical_accuracy: 0.9256
32960/60000 [===============>..............] - ETA: 52s - loss: 0.2452 - categorical_accuracy: 0.9257
32992/60000 [===============>..............] - ETA: 52s - loss: 0.2451 - categorical_accuracy: 0.9257
33024/60000 [===============>..............] - ETA: 52s - loss: 0.2450 - categorical_accuracy: 0.9257
33056/60000 [===============>..............] - ETA: 52s - loss: 0.2449 - categorical_accuracy: 0.9258
33088/60000 [===============>..............] - ETA: 52s - loss: 0.2449 - categorical_accuracy: 0.9258
33120/60000 [===============>..............] - ETA: 52s - loss: 0.2451 - categorical_accuracy: 0.9258
33152/60000 [===============>..............] - ETA: 52s - loss: 0.2449 - categorical_accuracy: 0.9259
33184/60000 [===============>..............] - ETA: 51s - loss: 0.2448 - categorical_accuracy: 0.9259
33216/60000 [===============>..............] - ETA: 51s - loss: 0.2445 - categorical_accuracy: 0.9260
33248/60000 [===============>..............] - ETA: 51s - loss: 0.2445 - categorical_accuracy: 0.9260
33280/60000 [===============>..............] - ETA: 51s - loss: 0.2443 - categorical_accuracy: 0.9261
33312/60000 [===============>..............] - ETA: 51s - loss: 0.2441 - categorical_accuracy: 0.9262
33344/60000 [===============>..............] - ETA: 51s - loss: 0.2439 - categorical_accuracy: 0.9263
33376/60000 [===============>..............] - ETA: 51s - loss: 0.2438 - categorical_accuracy: 0.9263
33408/60000 [===============>..............] - ETA: 51s - loss: 0.2436 - categorical_accuracy: 0.9263
33440/60000 [===============>..............] - ETA: 51s - loss: 0.2436 - categorical_accuracy: 0.9264
33472/60000 [===============>..............] - ETA: 51s - loss: 0.2436 - categorical_accuracy: 0.9264
33504/60000 [===============>..............] - ETA: 51s - loss: 0.2436 - categorical_accuracy: 0.9264
33536/60000 [===============>..............] - ETA: 51s - loss: 0.2435 - categorical_accuracy: 0.9265
33568/60000 [===============>..............] - ETA: 51s - loss: 0.2434 - categorical_accuracy: 0.9265
33600/60000 [===============>..............] - ETA: 51s - loss: 0.2436 - categorical_accuracy: 0.9264
33632/60000 [===============>..............] - ETA: 51s - loss: 0.2435 - categorical_accuracy: 0.9265
33664/60000 [===============>..............] - ETA: 51s - loss: 0.2433 - categorical_accuracy: 0.9265
33696/60000 [===============>..............] - ETA: 51s - loss: 0.2431 - categorical_accuracy: 0.9266
33728/60000 [===============>..............] - ETA: 50s - loss: 0.2430 - categorical_accuracy: 0.9266
33760/60000 [===============>..............] - ETA: 50s - loss: 0.2428 - categorical_accuracy: 0.9267
33792/60000 [===============>..............] - ETA: 50s - loss: 0.2426 - categorical_accuracy: 0.9268
33824/60000 [===============>..............] - ETA: 50s - loss: 0.2425 - categorical_accuracy: 0.9268
33856/60000 [===============>..............] - ETA: 50s - loss: 0.2424 - categorical_accuracy: 0.9269
33888/60000 [===============>..............] - ETA: 50s - loss: 0.2422 - categorical_accuracy: 0.9269
33920/60000 [===============>..............] - ETA: 50s - loss: 0.2421 - categorical_accuracy: 0.9269
33952/60000 [===============>..............] - ETA: 50s - loss: 0.2422 - categorical_accuracy: 0.9270
33984/60000 [===============>..............] - ETA: 50s - loss: 0.2423 - categorical_accuracy: 0.9270
34016/60000 [================>.............] - ETA: 50s - loss: 0.2422 - categorical_accuracy: 0.9270
34048/60000 [================>.............] - ETA: 50s - loss: 0.2420 - categorical_accuracy: 0.9270
34080/60000 [================>.............] - ETA: 50s - loss: 0.2419 - categorical_accuracy: 0.9270
34112/60000 [================>.............] - ETA: 50s - loss: 0.2419 - categorical_accuracy: 0.9270
34144/60000 [================>.............] - ETA: 50s - loss: 0.2418 - categorical_accuracy: 0.9271
34176/60000 [================>.............] - ETA: 50s - loss: 0.2417 - categorical_accuracy: 0.9271
34208/60000 [================>.............] - ETA: 49s - loss: 0.2416 - categorical_accuracy: 0.9271
34240/60000 [================>.............] - ETA: 49s - loss: 0.2415 - categorical_accuracy: 0.9272
34272/60000 [================>.............] - ETA: 49s - loss: 0.2413 - categorical_accuracy: 0.9272
34304/60000 [================>.............] - ETA: 49s - loss: 0.2412 - categorical_accuracy: 0.9272
34336/60000 [================>.............] - ETA: 49s - loss: 0.2412 - categorical_accuracy: 0.9272
34368/60000 [================>.............] - ETA: 49s - loss: 0.2413 - categorical_accuracy: 0.9272
34400/60000 [================>.............] - ETA: 49s - loss: 0.2412 - categorical_accuracy: 0.9272
34432/60000 [================>.............] - ETA: 49s - loss: 0.2411 - categorical_accuracy: 0.9272
34464/60000 [================>.............] - ETA: 49s - loss: 0.2409 - categorical_accuracy: 0.9273
34496/60000 [================>.............] - ETA: 49s - loss: 0.2409 - categorical_accuracy: 0.9273
34528/60000 [================>.............] - ETA: 49s - loss: 0.2408 - categorical_accuracy: 0.9273
34560/60000 [================>.............] - ETA: 49s - loss: 0.2409 - categorical_accuracy: 0.9273
34592/60000 [================>.............] - ETA: 49s - loss: 0.2407 - categorical_accuracy: 0.9274
34624/60000 [================>.............] - ETA: 49s - loss: 0.2406 - categorical_accuracy: 0.9274
34656/60000 [================>.............] - ETA: 49s - loss: 0.2404 - categorical_accuracy: 0.9275
34688/60000 [================>.............] - ETA: 49s - loss: 0.2403 - categorical_accuracy: 0.9275
34720/60000 [================>.............] - ETA: 48s - loss: 0.2402 - categorical_accuracy: 0.9275
34752/60000 [================>.............] - ETA: 48s - loss: 0.2401 - categorical_accuracy: 0.9275
34784/60000 [================>.............] - ETA: 48s - loss: 0.2402 - categorical_accuracy: 0.9276
34816/60000 [================>.............] - ETA: 48s - loss: 0.2400 - categorical_accuracy: 0.9276
34848/60000 [================>.............] - ETA: 48s - loss: 0.2398 - categorical_accuracy: 0.9277
34880/60000 [================>.............] - ETA: 48s - loss: 0.2397 - categorical_accuracy: 0.9277
34912/60000 [================>.............] - ETA: 48s - loss: 0.2395 - categorical_accuracy: 0.9278
34944/60000 [================>.............] - ETA: 48s - loss: 0.2394 - categorical_accuracy: 0.9278
34976/60000 [================>.............] - ETA: 48s - loss: 0.2395 - categorical_accuracy: 0.9278
35008/60000 [================>.............] - ETA: 48s - loss: 0.2393 - categorical_accuracy: 0.9278
35040/60000 [================>.............] - ETA: 48s - loss: 0.2392 - categorical_accuracy: 0.9279
35072/60000 [================>.............] - ETA: 48s - loss: 0.2391 - categorical_accuracy: 0.9279
35104/60000 [================>.............] - ETA: 48s - loss: 0.2390 - categorical_accuracy: 0.9279
35136/60000 [================>.............] - ETA: 48s - loss: 0.2388 - categorical_accuracy: 0.9279
35168/60000 [================>.............] - ETA: 48s - loss: 0.2387 - categorical_accuracy: 0.9280
35200/60000 [================>.............] - ETA: 48s - loss: 0.2385 - categorical_accuracy: 0.9280
35232/60000 [================>.............] - ETA: 47s - loss: 0.2383 - categorical_accuracy: 0.9281
35264/60000 [================>.............] - ETA: 47s - loss: 0.2383 - categorical_accuracy: 0.9281
35296/60000 [================>.............] - ETA: 47s - loss: 0.2382 - categorical_accuracy: 0.9282
35328/60000 [================>.............] - ETA: 47s - loss: 0.2382 - categorical_accuracy: 0.9282
35360/60000 [================>.............] - ETA: 47s - loss: 0.2381 - categorical_accuracy: 0.9282
35392/60000 [================>.............] - ETA: 47s - loss: 0.2381 - categorical_accuracy: 0.9282
35424/60000 [================>.............] - ETA: 47s - loss: 0.2382 - categorical_accuracy: 0.9282
35456/60000 [================>.............] - ETA: 47s - loss: 0.2380 - categorical_accuracy: 0.9283
35488/60000 [================>.............] - ETA: 47s - loss: 0.2378 - categorical_accuracy: 0.9283
35520/60000 [================>.............] - ETA: 47s - loss: 0.2378 - categorical_accuracy: 0.9283
35552/60000 [================>.............] - ETA: 47s - loss: 0.2377 - categorical_accuracy: 0.9283
35584/60000 [================>.............] - ETA: 47s - loss: 0.2375 - categorical_accuracy: 0.9284
35616/60000 [================>.............] - ETA: 47s - loss: 0.2374 - categorical_accuracy: 0.9284
35648/60000 [================>.............] - ETA: 47s - loss: 0.2372 - categorical_accuracy: 0.9285
35680/60000 [================>.............] - ETA: 47s - loss: 0.2371 - categorical_accuracy: 0.9285
35712/60000 [================>.............] - ETA: 47s - loss: 0.2371 - categorical_accuracy: 0.9285
35744/60000 [================>.............] - ETA: 47s - loss: 0.2370 - categorical_accuracy: 0.9286
35776/60000 [================>.............] - ETA: 46s - loss: 0.2368 - categorical_accuracy: 0.9286
35808/60000 [================>.............] - ETA: 46s - loss: 0.2367 - categorical_accuracy: 0.9287
35840/60000 [================>.............] - ETA: 46s - loss: 0.2369 - categorical_accuracy: 0.9287
35872/60000 [================>.............] - ETA: 46s - loss: 0.2367 - categorical_accuracy: 0.9287
35904/60000 [================>.............] - ETA: 46s - loss: 0.2365 - categorical_accuracy: 0.9288
35936/60000 [================>.............] - ETA: 46s - loss: 0.2364 - categorical_accuracy: 0.9288
35968/60000 [================>.............] - ETA: 46s - loss: 0.2362 - categorical_accuracy: 0.9289
36000/60000 [=================>............] - ETA: 46s - loss: 0.2364 - categorical_accuracy: 0.9289
36032/60000 [=================>............] - ETA: 46s - loss: 0.2363 - categorical_accuracy: 0.9289
36064/60000 [=================>............] - ETA: 46s - loss: 0.2362 - categorical_accuracy: 0.9289
36096/60000 [=================>............] - ETA: 46s - loss: 0.2361 - categorical_accuracy: 0.9289
36128/60000 [=================>............] - ETA: 46s - loss: 0.2359 - categorical_accuracy: 0.9290
36160/60000 [=================>............] - ETA: 46s - loss: 0.2358 - categorical_accuracy: 0.9290
36192/60000 [=================>............] - ETA: 46s - loss: 0.2357 - categorical_accuracy: 0.9291
36224/60000 [=================>............] - ETA: 46s - loss: 0.2355 - categorical_accuracy: 0.9291
36256/60000 [=================>............] - ETA: 46s - loss: 0.2354 - categorical_accuracy: 0.9292
36288/60000 [=================>............] - ETA: 45s - loss: 0.2352 - categorical_accuracy: 0.9292
36320/60000 [=================>............] - ETA: 45s - loss: 0.2350 - categorical_accuracy: 0.9293
36352/60000 [=================>............] - ETA: 45s - loss: 0.2349 - categorical_accuracy: 0.9293
36384/60000 [=================>............] - ETA: 45s - loss: 0.2349 - categorical_accuracy: 0.9293
36416/60000 [=================>............] - ETA: 45s - loss: 0.2348 - categorical_accuracy: 0.9293
36448/60000 [=================>............] - ETA: 45s - loss: 0.2347 - categorical_accuracy: 0.9294
36480/60000 [=================>............] - ETA: 45s - loss: 0.2345 - categorical_accuracy: 0.9294
36512/60000 [=================>............] - ETA: 45s - loss: 0.2344 - categorical_accuracy: 0.9295
36544/60000 [=================>............] - ETA: 45s - loss: 0.2343 - categorical_accuracy: 0.9295
36576/60000 [=================>............] - ETA: 45s - loss: 0.2341 - categorical_accuracy: 0.9295
36608/60000 [=================>............] - ETA: 45s - loss: 0.2340 - categorical_accuracy: 0.9296
36640/60000 [=================>............] - ETA: 45s - loss: 0.2338 - categorical_accuracy: 0.9297
36672/60000 [=================>............] - ETA: 45s - loss: 0.2336 - categorical_accuracy: 0.9297
36704/60000 [=================>............] - ETA: 45s - loss: 0.2335 - categorical_accuracy: 0.9298
36736/60000 [=================>............] - ETA: 45s - loss: 0.2333 - categorical_accuracy: 0.9298
36768/60000 [=================>............] - ETA: 45s - loss: 0.2331 - categorical_accuracy: 0.9299
36800/60000 [=================>............] - ETA: 44s - loss: 0.2330 - categorical_accuracy: 0.9299
36832/60000 [=================>............] - ETA: 44s - loss: 0.2328 - categorical_accuracy: 0.9300
36864/60000 [=================>............] - ETA: 44s - loss: 0.2330 - categorical_accuracy: 0.9299
36896/60000 [=================>............] - ETA: 44s - loss: 0.2329 - categorical_accuracy: 0.9299
36928/60000 [=================>............] - ETA: 44s - loss: 0.2327 - categorical_accuracy: 0.9300
36960/60000 [=================>............] - ETA: 44s - loss: 0.2327 - categorical_accuracy: 0.9300
36992/60000 [=================>............] - ETA: 44s - loss: 0.2327 - categorical_accuracy: 0.9300
37024/60000 [=================>............] - ETA: 44s - loss: 0.2327 - categorical_accuracy: 0.9300
37056/60000 [=================>............] - ETA: 44s - loss: 0.2325 - categorical_accuracy: 0.9301
37088/60000 [=================>............] - ETA: 44s - loss: 0.2323 - categorical_accuracy: 0.9301
37120/60000 [=================>............] - ETA: 44s - loss: 0.2322 - categorical_accuracy: 0.9301
37152/60000 [=================>............] - ETA: 44s - loss: 0.2321 - categorical_accuracy: 0.9302
37184/60000 [=================>............] - ETA: 44s - loss: 0.2320 - categorical_accuracy: 0.9302
37216/60000 [=================>............] - ETA: 44s - loss: 0.2319 - categorical_accuracy: 0.9302
37248/60000 [=================>............] - ETA: 44s - loss: 0.2319 - categorical_accuracy: 0.9302
37280/60000 [=================>............] - ETA: 44s - loss: 0.2318 - categorical_accuracy: 0.9303
37312/60000 [=================>............] - ETA: 43s - loss: 0.2318 - categorical_accuracy: 0.9303
37344/60000 [=================>............] - ETA: 43s - loss: 0.2316 - categorical_accuracy: 0.9304
37376/60000 [=================>............] - ETA: 43s - loss: 0.2315 - categorical_accuracy: 0.9304
37408/60000 [=================>............] - ETA: 43s - loss: 0.2313 - categorical_accuracy: 0.9304
37440/60000 [=================>............] - ETA: 43s - loss: 0.2313 - categorical_accuracy: 0.9304
37472/60000 [=================>............] - ETA: 43s - loss: 0.2313 - categorical_accuracy: 0.9305
37504/60000 [=================>............] - ETA: 43s - loss: 0.2311 - categorical_accuracy: 0.9305
37536/60000 [=================>............] - ETA: 43s - loss: 0.2310 - categorical_accuracy: 0.9305
37568/60000 [=================>............] - ETA: 43s - loss: 0.2310 - categorical_accuracy: 0.9306
37600/60000 [=================>............] - ETA: 43s - loss: 0.2308 - categorical_accuracy: 0.9306
37632/60000 [=================>............] - ETA: 43s - loss: 0.2309 - categorical_accuracy: 0.9306
37664/60000 [=================>............] - ETA: 43s - loss: 0.2308 - categorical_accuracy: 0.9306
37696/60000 [=================>............] - ETA: 43s - loss: 0.2306 - categorical_accuracy: 0.9307
37728/60000 [=================>............] - ETA: 43s - loss: 0.2305 - categorical_accuracy: 0.9307
37760/60000 [=================>............] - ETA: 43s - loss: 0.2306 - categorical_accuracy: 0.9307
37792/60000 [=================>............] - ETA: 43s - loss: 0.2305 - categorical_accuracy: 0.9307
37824/60000 [=================>............] - ETA: 42s - loss: 0.2303 - categorical_accuracy: 0.9308
37856/60000 [=================>............] - ETA: 42s - loss: 0.2302 - categorical_accuracy: 0.9308
37888/60000 [=================>............] - ETA: 42s - loss: 0.2301 - categorical_accuracy: 0.9308
37920/60000 [=================>............] - ETA: 42s - loss: 0.2300 - categorical_accuracy: 0.9309
37952/60000 [=================>............] - ETA: 42s - loss: 0.2298 - categorical_accuracy: 0.9310
37984/60000 [=================>............] - ETA: 42s - loss: 0.2296 - categorical_accuracy: 0.9310
38016/60000 [==================>...........] - ETA: 42s - loss: 0.2296 - categorical_accuracy: 0.9310
38048/60000 [==================>...........] - ETA: 42s - loss: 0.2295 - categorical_accuracy: 0.9311
38080/60000 [==================>...........] - ETA: 42s - loss: 0.2295 - categorical_accuracy: 0.9311
38112/60000 [==================>...........] - ETA: 42s - loss: 0.2294 - categorical_accuracy: 0.9311
38144/60000 [==================>...........] - ETA: 42s - loss: 0.2292 - categorical_accuracy: 0.9312
38176/60000 [==================>...........] - ETA: 42s - loss: 0.2292 - categorical_accuracy: 0.9312
38208/60000 [==================>...........] - ETA: 42s - loss: 0.2290 - categorical_accuracy: 0.9312
38240/60000 [==================>...........] - ETA: 42s - loss: 0.2290 - categorical_accuracy: 0.9312
38272/60000 [==================>...........] - ETA: 42s - loss: 0.2288 - categorical_accuracy: 0.9313
38304/60000 [==================>...........] - ETA: 42s - loss: 0.2287 - categorical_accuracy: 0.9313
38336/60000 [==================>...........] - ETA: 41s - loss: 0.2285 - categorical_accuracy: 0.9314
38368/60000 [==================>...........] - ETA: 41s - loss: 0.2284 - categorical_accuracy: 0.9314
38400/60000 [==================>...........] - ETA: 41s - loss: 0.2283 - categorical_accuracy: 0.9315
38432/60000 [==================>...........] - ETA: 41s - loss: 0.2281 - categorical_accuracy: 0.9315
38464/60000 [==================>...........] - ETA: 41s - loss: 0.2280 - categorical_accuracy: 0.9315
38496/60000 [==================>...........] - ETA: 41s - loss: 0.2278 - categorical_accuracy: 0.9316
38528/60000 [==================>...........] - ETA: 41s - loss: 0.2277 - categorical_accuracy: 0.9316
38560/60000 [==================>...........] - ETA: 41s - loss: 0.2276 - categorical_accuracy: 0.9316
38592/60000 [==================>...........] - ETA: 41s - loss: 0.2275 - categorical_accuracy: 0.9317
38624/60000 [==================>...........] - ETA: 41s - loss: 0.2274 - categorical_accuracy: 0.9317
38656/60000 [==================>...........] - ETA: 41s - loss: 0.2273 - categorical_accuracy: 0.9318
38688/60000 [==================>...........] - ETA: 41s - loss: 0.2273 - categorical_accuracy: 0.9318
38720/60000 [==================>...........] - ETA: 41s - loss: 0.2273 - categorical_accuracy: 0.9317
38752/60000 [==================>...........] - ETA: 41s - loss: 0.2272 - categorical_accuracy: 0.9318
38784/60000 [==================>...........] - ETA: 41s - loss: 0.2273 - categorical_accuracy: 0.9317
38816/60000 [==================>...........] - ETA: 41s - loss: 0.2271 - categorical_accuracy: 0.9317
38848/60000 [==================>...........] - ETA: 40s - loss: 0.2271 - categorical_accuracy: 0.9317
38880/60000 [==================>...........] - ETA: 40s - loss: 0.2270 - categorical_accuracy: 0.9318
38912/60000 [==================>...........] - ETA: 40s - loss: 0.2271 - categorical_accuracy: 0.9318
38944/60000 [==================>...........] - ETA: 40s - loss: 0.2269 - categorical_accuracy: 0.9318
38976/60000 [==================>...........] - ETA: 40s - loss: 0.2267 - categorical_accuracy: 0.9319
39008/60000 [==================>...........] - ETA: 40s - loss: 0.2266 - categorical_accuracy: 0.9319
39040/60000 [==================>...........] - ETA: 40s - loss: 0.2265 - categorical_accuracy: 0.9319
39072/60000 [==================>...........] - ETA: 40s - loss: 0.2265 - categorical_accuracy: 0.9319
39104/60000 [==================>...........] - ETA: 40s - loss: 0.2263 - categorical_accuracy: 0.9320
39136/60000 [==================>...........] - ETA: 40s - loss: 0.2262 - categorical_accuracy: 0.9320
39168/60000 [==================>...........] - ETA: 40s - loss: 0.2261 - categorical_accuracy: 0.9320
39200/60000 [==================>...........] - ETA: 40s - loss: 0.2260 - categorical_accuracy: 0.9320
39232/60000 [==================>...........] - ETA: 40s - loss: 0.2259 - categorical_accuracy: 0.9320
39264/60000 [==================>...........] - ETA: 40s - loss: 0.2258 - categorical_accuracy: 0.9321
39296/60000 [==================>...........] - ETA: 40s - loss: 0.2256 - categorical_accuracy: 0.9321
39328/60000 [==================>...........] - ETA: 40s - loss: 0.2255 - categorical_accuracy: 0.9321
39360/60000 [==================>...........] - ETA: 39s - loss: 0.2253 - categorical_accuracy: 0.9322
39392/60000 [==================>...........] - ETA: 39s - loss: 0.2251 - categorical_accuracy: 0.9322
39424/60000 [==================>...........] - ETA: 39s - loss: 0.2253 - categorical_accuracy: 0.9323
39456/60000 [==================>...........] - ETA: 39s - loss: 0.2252 - categorical_accuracy: 0.9323
39488/60000 [==================>...........] - ETA: 39s - loss: 0.2252 - categorical_accuracy: 0.9323
39520/60000 [==================>...........] - ETA: 39s - loss: 0.2252 - categorical_accuracy: 0.9323
39552/60000 [==================>...........] - ETA: 39s - loss: 0.2252 - categorical_accuracy: 0.9323
39584/60000 [==================>...........] - ETA: 39s - loss: 0.2251 - categorical_accuracy: 0.9323
39616/60000 [==================>...........] - ETA: 39s - loss: 0.2251 - categorical_accuracy: 0.9323
39648/60000 [==================>...........] - ETA: 39s - loss: 0.2250 - categorical_accuracy: 0.9323
39680/60000 [==================>...........] - ETA: 39s - loss: 0.2249 - categorical_accuracy: 0.9323
39712/60000 [==================>...........] - ETA: 39s - loss: 0.2249 - categorical_accuracy: 0.9323
39744/60000 [==================>...........] - ETA: 39s - loss: 0.2247 - categorical_accuracy: 0.9324
39776/60000 [==================>...........] - ETA: 39s - loss: 0.2246 - categorical_accuracy: 0.9324
39808/60000 [==================>...........] - ETA: 39s - loss: 0.2244 - categorical_accuracy: 0.9325
39840/60000 [==================>...........] - ETA: 39s - loss: 0.2243 - categorical_accuracy: 0.9325
39872/60000 [==================>...........] - ETA: 38s - loss: 0.2242 - categorical_accuracy: 0.9326
39904/60000 [==================>...........] - ETA: 38s - loss: 0.2243 - categorical_accuracy: 0.9326
39936/60000 [==================>...........] - ETA: 38s - loss: 0.2242 - categorical_accuracy: 0.9326
39968/60000 [==================>...........] - ETA: 38s - loss: 0.2242 - categorical_accuracy: 0.9326
40000/60000 [===================>..........] - ETA: 38s - loss: 0.2241 - categorical_accuracy: 0.9327
40032/60000 [===================>..........] - ETA: 38s - loss: 0.2240 - categorical_accuracy: 0.9327
40064/60000 [===================>..........] - ETA: 38s - loss: 0.2240 - categorical_accuracy: 0.9327
40096/60000 [===================>..........] - ETA: 38s - loss: 0.2239 - categorical_accuracy: 0.9327
40128/60000 [===================>..........] - ETA: 38s - loss: 0.2237 - categorical_accuracy: 0.9327
40160/60000 [===================>..........] - ETA: 38s - loss: 0.2236 - categorical_accuracy: 0.9328
40192/60000 [===================>..........] - ETA: 38s - loss: 0.2235 - categorical_accuracy: 0.9328
40224/60000 [===================>..........] - ETA: 38s - loss: 0.2234 - categorical_accuracy: 0.9328
40256/60000 [===================>..........] - ETA: 38s - loss: 0.2234 - categorical_accuracy: 0.9328
40288/60000 [===================>..........] - ETA: 38s - loss: 0.2233 - categorical_accuracy: 0.9328
40320/60000 [===================>..........] - ETA: 38s - loss: 0.2232 - categorical_accuracy: 0.9329
40352/60000 [===================>..........] - ETA: 38s - loss: 0.2230 - categorical_accuracy: 0.9329
40384/60000 [===================>..........] - ETA: 37s - loss: 0.2229 - categorical_accuracy: 0.9329
40416/60000 [===================>..........] - ETA: 37s - loss: 0.2228 - categorical_accuracy: 0.9330
40448/60000 [===================>..........] - ETA: 37s - loss: 0.2226 - categorical_accuracy: 0.9330
40480/60000 [===================>..........] - ETA: 37s - loss: 0.2224 - categorical_accuracy: 0.9331
40512/60000 [===================>..........] - ETA: 37s - loss: 0.2225 - categorical_accuracy: 0.9331
40544/60000 [===================>..........] - ETA: 37s - loss: 0.2224 - categorical_accuracy: 0.9331
40576/60000 [===================>..........] - ETA: 37s - loss: 0.2222 - categorical_accuracy: 0.9332
40608/60000 [===================>..........] - ETA: 37s - loss: 0.2221 - categorical_accuracy: 0.9332
40640/60000 [===================>..........] - ETA: 37s - loss: 0.2219 - categorical_accuracy: 0.9333
40672/60000 [===================>..........] - ETA: 37s - loss: 0.2219 - categorical_accuracy: 0.9333
40704/60000 [===================>..........] - ETA: 37s - loss: 0.2218 - categorical_accuracy: 0.9333
40736/60000 [===================>..........] - ETA: 37s - loss: 0.2216 - categorical_accuracy: 0.9334
40768/60000 [===================>..........] - ETA: 37s - loss: 0.2217 - categorical_accuracy: 0.9334
40800/60000 [===================>..........] - ETA: 37s - loss: 0.2215 - categorical_accuracy: 0.9335
40832/60000 [===================>..........] - ETA: 37s - loss: 0.2216 - categorical_accuracy: 0.9335
40864/60000 [===================>..........] - ETA: 37s - loss: 0.2216 - categorical_accuracy: 0.9335
40896/60000 [===================>..........] - ETA: 37s - loss: 0.2215 - categorical_accuracy: 0.9336
40928/60000 [===================>..........] - ETA: 36s - loss: 0.2213 - categorical_accuracy: 0.9336
40960/60000 [===================>..........] - ETA: 36s - loss: 0.2212 - categorical_accuracy: 0.9337
40992/60000 [===================>..........] - ETA: 36s - loss: 0.2212 - categorical_accuracy: 0.9336
41024/60000 [===================>..........] - ETA: 36s - loss: 0.2212 - categorical_accuracy: 0.9336
41056/60000 [===================>..........] - ETA: 36s - loss: 0.2210 - categorical_accuracy: 0.9337
41088/60000 [===================>..........] - ETA: 36s - loss: 0.2210 - categorical_accuracy: 0.9337
41120/60000 [===================>..........] - ETA: 36s - loss: 0.2209 - categorical_accuracy: 0.9337
41152/60000 [===================>..........] - ETA: 36s - loss: 0.2207 - categorical_accuracy: 0.9338
41184/60000 [===================>..........] - ETA: 36s - loss: 0.2206 - categorical_accuracy: 0.9338
41216/60000 [===================>..........] - ETA: 36s - loss: 0.2206 - categorical_accuracy: 0.9338
41248/60000 [===================>..........] - ETA: 36s - loss: 0.2205 - categorical_accuracy: 0.9339
41280/60000 [===================>..........] - ETA: 36s - loss: 0.2208 - categorical_accuracy: 0.9339
41312/60000 [===================>..........] - ETA: 36s - loss: 0.2207 - categorical_accuracy: 0.9339
41344/60000 [===================>..........] - ETA: 36s - loss: 0.2206 - categorical_accuracy: 0.9339
41376/60000 [===================>..........] - ETA: 36s - loss: 0.2205 - categorical_accuracy: 0.9339
41408/60000 [===================>..........] - ETA: 36s - loss: 0.2204 - categorical_accuracy: 0.9340
41440/60000 [===================>..........] - ETA: 35s - loss: 0.2203 - categorical_accuracy: 0.9340
41472/60000 [===================>..........] - ETA: 35s - loss: 0.2202 - categorical_accuracy: 0.9341
41504/60000 [===================>..........] - ETA: 35s - loss: 0.2204 - categorical_accuracy: 0.9340
41536/60000 [===================>..........] - ETA: 35s - loss: 0.2202 - categorical_accuracy: 0.9341
41568/60000 [===================>..........] - ETA: 35s - loss: 0.2201 - categorical_accuracy: 0.9341
41600/60000 [===================>..........] - ETA: 35s - loss: 0.2202 - categorical_accuracy: 0.9341
41632/60000 [===================>..........] - ETA: 35s - loss: 0.2200 - categorical_accuracy: 0.9342
41664/60000 [===================>..........] - ETA: 35s - loss: 0.2199 - categorical_accuracy: 0.9342
41696/60000 [===================>..........] - ETA: 35s - loss: 0.2198 - categorical_accuracy: 0.9342
41728/60000 [===================>..........] - ETA: 35s - loss: 0.2196 - categorical_accuracy: 0.9343
41760/60000 [===================>..........] - ETA: 35s - loss: 0.2197 - categorical_accuracy: 0.9343
41792/60000 [===================>..........] - ETA: 35s - loss: 0.2196 - categorical_accuracy: 0.9343
41824/60000 [===================>..........] - ETA: 35s - loss: 0.2195 - categorical_accuracy: 0.9343
41856/60000 [===================>..........] - ETA: 35s - loss: 0.2194 - categorical_accuracy: 0.9343
41888/60000 [===================>..........] - ETA: 35s - loss: 0.2193 - categorical_accuracy: 0.9344
41920/60000 [===================>..........] - ETA: 35s - loss: 0.2192 - categorical_accuracy: 0.9344
41952/60000 [===================>..........] - ETA: 34s - loss: 0.2192 - categorical_accuracy: 0.9344
41984/60000 [===================>..........] - ETA: 34s - loss: 0.2191 - categorical_accuracy: 0.9345
42016/60000 [====================>.........] - ETA: 34s - loss: 0.2189 - categorical_accuracy: 0.9345
42048/60000 [====================>.........] - ETA: 34s - loss: 0.2188 - categorical_accuracy: 0.9346
42080/60000 [====================>.........] - ETA: 34s - loss: 0.2186 - categorical_accuracy: 0.9346
42112/60000 [====================>.........] - ETA: 34s - loss: 0.2186 - categorical_accuracy: 0.9346
42144/60000 [====================>.........] - ETA: 34s - loss: 0.2185 - categorical_accuracy: 0.9346
42176/60000 [====================>.........] - ETA: 34s - loss: 0.2184 - categorical_accuracy: 0.9346
42208/60000 [====================>.........] - ETA: 34s - loss: 0.2183 - categorical_accuracy: 0.9346
42240/60000 [====================>.........] - ETA: 34s - loss: 0.2182 - categorical_accuracy: 0.9347
42272/60000 [====================>.........] - ETA: 34s - loss: 0.2181 - categorical_accuracy: 0.9347
42304/60000 [====================>.........] - ETA: 34s - loss: 0.2180 - categorical_accuracy: 0.9347
42336/60000 [====================>.........] - ETA: 34s - loss: 0.2179 - categorical_accuracy: 0.9347
42368/60000 [====================>.........] - ETA: 34s - loss: 0.2180 - categorical_accuracy: 0.9347
42400/60000 [====================>.........] - ETA: 34s - loss: 0.2179 - categorical_accuracy: 0.9347
42432/60000 [====================>.........] - ETA: 34s - loss: 0.2178 - categorical_accuracy: 0.9348
42464/60000 [====================>.........] - ETA: 33s - loss: 0.2176 - categorical_accuracy: 0.9348
42496/60000 [====================>.........] - ETA: 33s - loss: 0.2176 - categorical_accuracy: 0.9348
42528/60000 [====================>.........] - ETA: 33s - loss: 0.2176 - categorical_accuracy: 0.9348
42560/60000 [====================>.........] - ETA: 33s - loss: 0.2175 - categorical_accuracy: 0.9349
42592/60000 [====================>.........] - ETA: 33s - loss: 0.2175 - categorical_accuracy: 0.9349
42624/60000 [====================>.........] - ETA: 33s - loss: 0.2174 - categorical_accuracy: 0.9349
42656/60000 [====================>.........] - ETA: 33s - loss: 0.2173 - categorical_accuracy: 0.9350
42688/60000 [====================>.........] - ETA: 33s - loss: 0.2171 - categorical_accuracy: 0.9350
42720/60000 [====================>.........] - ETA: 33s - loss: 0.2170 - categorical_accuracy: 0.9350
42752/60000 [====================>.........] - ETA: 33s - loss: 0.2169 - categorical_accuracy: 0.9350
42784/60000 [====================>.........] - ETA: 33s - loss: 0.2167 - categorical_accuracy: 0.9351
42816/60000 [====================>.........] - ETA: 33s - loss: 0.2166 - categorical_accuracy: 0.9351
42848/60000 [====================>.........] - ETA: 33s - loss: 0.2165 - categorical_accuracy: 0.9351
42880/60000 [====================>.........] - ETA: 33s - loss: 0.2164 - categorical_accuracy: 0.9352
42912/60000 [====================>.........] - ETA: 33s - loss: 0.2163 - categorical_accuracy: 0.9352
42944/60000 [====================>.........] - ETA: 33s - loss: 0.2162 - categorical_accuracy: 0.9352
42976/60000 [====================>.........] - ETA: 32s - loss: 0.2162 - categorical_accuracy: 0.9353
43008/60000 [====================>.........] - ETA: 32s - loss: 0.2161 - categorical_accuracy: 0.9353
43040/60000 [====================>.........] - ETA: 32s - loss: 0.2161 - categorical_accuracy: 0.9353
43072/60000 [====================>.........] - ETA: 32s - loss: 0.2160 - categorical_accuracy: 0.9353
43104/60000 [====================>.........] - ETA: 32s - loss: 0.2159 - categorical_accuracy: 0.9354
43136/60000 [====================>.........] - ETA: 32s - loss: 0.2158 - categorical_accuracy: 0.9354
43168/60000 [====================>.........] - ETA: 32s - loss: 0.2156 - categorical_accuracy: 0.9354
43200/60000 [====================>.........] - ETA: 32s - loss: 0.2156 - categorical_accuracy: 0.9355
43232/60000 [====================>.........] - ETA: 32s - loss: 0.2154 - categorical_accuracy: 0.9355
43264/60000 [====================>.........] - ETA: 32s - loss: 0.2154 - categorical_accuracy: 0.9355
43296/60000 [====================>.........] - ETA: 32s - loss: 0.2153 - categorical_accuracy: 0.9355
43328/60000 [====================>.........] - ETA: 32s - loss: 0.2152 - categorical_accuracy: 0.9355
43360/60000 [====================>.........] - ETA: 32s - loss: 0.2152 - categorical_accuracy: 0.9355
43392/60000 [====================>.........] - ETA: 32s - loss: 0.2151 - categorical_accuracy: 0.9356
43424/60000 [====================>.........] - ETA: 32s - loss: 0.2151 - categorical_accuracy: 0.9356
43456/60000 [====================>.........] - ETA: 32s - loss: 0.2150 - categorical_accuracy: 0.9356
43488/60000 [====================>.........] - ETA: 31s - loss: 0.2149 - categorical_accuracy: 0.9356
43520/60000 [====================>.........] - ETA: 31s - loss: 0.2148 - categorical_accuracy: 0.9357
43552/60000 [====================>.........] - ETA: 31s - loss: 0.2147 - categorical_accuracy: 0.9357
43584/60000 [====================>.........] - ETA: 31s - loss: 0.2146 - categorical_accuracy: 0.9357
43616/60000 [====================>.........] - ETA: 31s - loss: 0.2145 - categorical_accuracy: 0.9357
43648/60000 [====================>.........] - ETA: 31s - loss: 0.2145 - categorical_accuracy: 0.9357
43680/60000 [====================>.........] - ETA: 31s - loss: 0.2143 - categorical_accuracy: 0.9358
43712/60000 [====================>.........] - ETA: 31s - loss: 0.2143 - categorical_accuracy: 0.9358
43744/60000 [====================>.........] - ETA: 31s - loss: 0.2142 - categorical_accuracy: 0.9358
43776/60000 [====================>.........] - ETA: 31s - loss: 0.2140 - categorical_accuracy: 0.9358
43808/60000 [====================>.........] - ETA: 31s - loss: 0.2140 - categorical_accuracy: 0.9359
43840/60000 [====================>.........] - ETA: 31s - loss: 0.2139 - categorical_accuracy: 0.9359
43872/60000 [====================>.........] - ETA: 31s - loss: 0.2138 - categorical_accuracy: 0.9359
43904/60000 [====================>.........] - ETA: 31s - loss: 0.2137 - categorical_accuracy: 0.9359
43936/60000 [====================>.........] - ETA: 31s - loss: 0.2135 - categorical_accuracy: 0.9360
43968/60000 [====================>.........] - ETA: 31s - loss: 0.2134 - categorical_accuracy: 0.9360
44000/60000 [=====================>........] - ETA: 30s - loss: 0.2133 - categorical_accuracy: 0.9360
44032/60000 [=====================>........] - ETA: 30s - loss: 0.2133 - categorical_accuracy: 0.9360
44064/60000 [=====================>........] - ETA: 30s - loss: 0.2132 - categorical_accuracy: 0.9361
44096/60000 [=====================>........] - ETA: 30s - loss: 0.2131 - categorical_accuracy: 0.9361
44128/60000 [=====================>........] - ETA: 30s - loss: 0.2131 - categorical_accuracy: 0.9361
44160/60000 [=====================>........] - ETA: 30s - loss: 0.2129 - categorical_accuracy: 0.9361
44192/60000 [=====================>........] - ETA: 30s - loss: 0.2129 - categorical_accuracy: 0.9362
44224/60000 [=====================>........] - ETA: 30s - loss: 0.2128 - categorical_accuracy: 0.9362
44256/60000 [=====================>........] - ETA: 30s - loss: 0.2127 - categorical_accuracy: 0.9362
44288/60000 [=====================>........] - ETA: 30s - loss: 0.2125 - categorical_accuracy: 0.9363
44320/60000 [=====================>........] - ETA: 30s - loss: 0.2124 - categorical_accuracy: 0.9363
44352/60000 [=====================>........] - ETA: 30s - loss: 0.2124 - categorical_accuracy: 0.9363
44384/60000 [=====================>........] - ETA: 30s - loss: 0.2123 - categorical_accuracy: 0.9363
44416/60000 [=====================>........] - ETA: 30s - loss: 0.2123 - categorical_accuracy: 0.9363
44448/60000 [=====================>........] - ETA: 30s - loss: 0.2122 - categorical_accuracy: 0.9364
44480/60000 [=====================>........] - ETA: 30s - loss: 0.2120 - categorical_accuracy: 0.9364
44512/60000 [=====================>........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9365
44544/60000 [=====================>........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9365
44576/60000 [=====================>........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9364
44608/60000 [=====================>........] - ETA: 29s - loss: 0.2118 - categorical_accuracy: 0.9365
44640/60000 [=====================>........] - ETA: 29s - loss: 0.2117 - categorical_accuracy: 0.9365
44672/60000 [=====================>........] - ETA: 29s - loss: 0.2116 - categorical_accuracy: 0.9365
44704/60000 [=====================>........] - ETA: 29s - loss: 0.2115 - categorical_accuracy: 0.9366
44736/60000 [=====================>........] - ETA: 29s - loss: 0.2115 - categorical_accuracy: 0.9366
44768/60000 [=====================>........] - ETA: 29s - loss: 0.2114 - categorical_accuracy: 0.9366
44800/60000 [=====================>........] - ETA: 29s - loss: 0.2113 - categorical_accuracy: 0.9367
44832/60000 [=====================>........] - ETA: 29s - loss: 0.2114 - categorical_accuracy: 0.9367
44864/60000 [=====================>........] - ETA: 29s - loss: 0.2112 - categorical_accuracy: 0.9367
44896/60000 [=====================>........] - ETA: 29s - loss: 0.2111 - categorical_accuracy: 0.9367
44928/60000 [=====================>........] - ETA: 29s - loss: 0.2110 - categorical_accuracy: 0.9368
44960/60000 [=====================>........] - ETA: 29s - loss: 0.2110 - categorical_accuracy: 0.9367
44992/60000 [=====================>........] - ETA: 29s - loss: 0.2110 - categorical_accuracy: 0.9367
45024/60000 [=====================>........] - ETA: 29s - loss: 0.2109 - categorical_accuracy: 0.9368
45056/60000 [=====================>........] - ETA: 28s - loss: 0.2108 - categorical_accuracy: 0.9368
45088/60000 [=====================>........] - ETA: 28s - loss: 0.2109 - categorical_accuracy: 0.9368
45120/60000 [=====================>........] - ETA: 28s - loss: 0.2109 - categorical_accuracy: 0.9368
45152/60000 [=====================>........] - ETA: 28s - loss: 0.2108 - categorical_accuracy: 0.9368
45184/60000 [=====================>........] - ETA: 28s - loss: 0.2107 - categorical_accuracy: 0.9368
45216/60000 [=====================>........] - ETA: 28s - loss: 0.2105 - categorical_accuracy: 0.9369
45248/60000 [=====================>........] - ETA: 28s - loss: 0.2104 - categorical_accuracy: 0.9369
45280/60000 [=====================>........] - ETA: 28s - loss: 0.2103 - categorical_accuracy: 0.9369
45312/60000 [=====================>........] - ETA: 28s - loss: 0.2102 - categorical_accuracy: 0.9370
45344/60000 [=====================>........] - ETA: 28s - loss: 0.2101 - categorical_accuracy: 0.9370
45376/60000 [=====================>........] - ETA: 28s - loss: 0.2100 - categorical_accuracy: 0.9370
45408/60000 [=====================>........] - ETA: 28s - loss: 0.2099 - categorical_accuracy: 0.9370
45440/60000 [=====================>........] - ETA: 28s - loss: 0.2097 - categorical_accuracy: 0.9371
45472/60000 [=====================>........] - ETA: 28s - loss: 0.2096 - categorical_accuracy: 0.9371
45504/60000 [=====================>........] - ETA: 28s - loss: 0.2096 - categorical_accuracy: 0.9371
45536/60000 [=====================>........] - ETA: 28s - loss: 0.2095 - categorical_accuracy: 0.9371
45568/60000 [=====================>........] - ETA: 27s - loss: 0.2096 - categorical_accuracy: 0.9371
45600/60000 [=====================>........] - ETA: 27s - loss: 0.2095 - categorical_accuracy: 0.9371
45632/60000 [=====================>........] - ETA: 27s - loss: 0.2094 - categorical_accuracy: 0.9372
45664/60000 [=====================>........] - ETA: 27s - loss: 0.2093 - categorical_accuracy: 0.9372
45696/60000 [=====================>........] - ETA: 27s - loss: 0.2091 - categorical_accuracy: 0.9373
45728/60000 [=====================>........] - ETA: 27s - loss: 0.2091 - categorical_accuracy: 0.9373
45760/60000 [=====================>........] - ETA: 27s - loss: 0.2090 - categorical_accuracy: 0.9373
45792/60000 [=====================>........] - ETA: 27s - loss: 0.2089 - categorical_accuracy: 0.9374
45824/60000 [=====================>........] - ETA: 27s - loss: 0.2091 - categorical_accuracy: 0.9373
45856/60000 [=====================>........] - ETA: 27s - loss: 0.2090 - categorical_accuracy: 0.9374
45888/60000 [=====================>........] - ETA: 27s - loss: 0.2089 - categorical_accuracy: 0.9374
45920/60000 [=====================>........] - ETA: 27s - loss: 0.2088 - categorical_accuracy: 0.9374
45952/60000 [=====================>........] - ETA: 27s - loss: 0.2087 - categorical_accuracy: 0.9375
45984/60000 [=====================>........] - ETA: 27s - loss: 0.2087 - categorical_accuracy: 0.9375
46016/60000 [======================>.......] - ETA: 27s - loss: 0.2086 - categorical_accuracy: 0.9375
46048/60000 [======================>.......] - ETA: 27s - loss: 0.2085 - categorical_accuracy: 0.9375
46080/60000 [======================>.......] - ETA: 26s - loss: 0.2084 - categorical_accuracy: 0.9375
46112/60000 [======================>.......] - ETA: 26s - loss: 0.2083 - categorical_accuracy: 0.9376
46144/60000 [======================>.......] - ETA: 26s - loss: 0.2083 - categorical_accuracy: 0.9376
46176/60000 [======================>.......] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9376
46208/60000 [======================>.......] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9376
46240/60000 [======================>.......] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9376
46272/60000 [======================>.......] - ETA: 26s - loss: 0.2081 - categorical_accuracy: 0.9377
46304/60000 [======================>.......] - ETA: 26s - loss: 0.2079 - categorical_accuracy: 0.9377
46336/60000 [======================>.......] - ETA: 26s - loss: 0.2078 - categorical_accuracy: 0.9377
46368/60000 [======================>.......] - ETA: 26s - loss: 0.2077 - categorical_accuracy: 0.9378
46400/60000 [======================>.......] - ETA: 26s - loss: 0.2076 - categorical_accuracy: 0.9378
46432/60000 [======================>.......] - ETA: 26s - loss: 0.2074 - categorical_accuracy: 0.9379
46464/60000 [======================>.......] - ETA: 26s - loss: 0.2073 - categorical_accuracy: 0.9379
46496/60000 [======================>.......] - ETA: 26s - loss: 0.2072 - categorical_accuracy: 0.9380
46528/60000 [======================>.......] - ETA: 26s - loss: 0.2071 - categorical_accuracy: 0.9380
46560/60000 [======================>.......] - ETA: 26s - loss: 0.2070 - categorical_accuracy: 0.9380
46592/60000 [======================>.......] - ETA: 25s - loss: 0.2069 - categorical_accuracy: 0.9380
46624/60000 [======================>.......] - ETA: 25s - loss: 0.2068 - categorical_accuracy: 0.9381
46656/60000 [======================>.......] - ETA: 25s - loss: 0.2067 - categorical_accuracy: 0.9381
46688/60000 [======================>.......] - ETA: 25s - loss: 0.2066 - categorical_accuracy: 0.9381
46720/60000 [======================>.......] - ETA: 25s - loss: 0.2065 - categorical_accuracy: 0.9381
46752/60000 [======================>.......] - ETA: 25s - loss: 0.2064 - categorical_accuracy: 0.9381
46784/60000 [======================>.......] - ETA: 25s - loss: 0.2063 - categorical_accuracy: 0.9382
46816/60000 [======================>.......] - ETA: 25s - loss: 0.2063 - categorical_accuracy: 0.9382
46848/60000 [======================>.......] - ETA: 25s - loss: 0.2062 - categorical_accuracy: 0.9382
46880/60000 [======================>.......] - ETA: 25s - loss: 0.2061 - categorical_accuracy: 0.9382
46912/60000 [======================>.......] - ETA: 25s - loss: 0.2060 - categorical_accuracy: 0.9383
46944/60000 [======================>.......] - ETA: 25s - loss: 0.2059 - categorical_accuracy: 0.9383
46976/60000 [======================>.......] - ETA: 25s - loss: 0.2059 - categorical_accuracy: 0.9383
47008/60000 [======================>.......] - ETA: 25s - loss: 0.2058 - categorical_accuracy: 0.9384
47040/60000 [======================>.......] - ETA: 25s - loss: 0.2057 - categorical_accuracy: 0.9384
47072/60000 [======================>.......] - ETA: 25s - loss: 0.2056 - categorical_accuracy: 0.9384
47104/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9384
47136/60000 [======================>.......] - ETA: 24s - loss: 0.2057 - categorical_accuracy: 0.9384
47168/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9384
47200/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9385
47232/60000 [======================>.......] - ETA: 24s - loss: 0.2055 - categorical_accuracy: 0.9385
47264/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9385
47296/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9385
47328/60000 [======================>.......] - ETA: 24s - loss: 0.2058 - categorical_accuracy: 0.9385
47360/60000 [======================>.......] - ETA: 24s - loss: 0.2058 - categorical_accuracy: 0.9385
47392/60000 [======================>.......] - ETA: 24s - loss: 0.2057 - categorical_accuracy: 0.9385
47424/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9385
47456/60000 [======================>.......] - ETA: 24s - loss: 0.2055 - categorical_accuracy: 0.9386
47488/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9386
47520/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9385
47552/60000 [======================>.......] - ETA: 24s - loss: 0.2055 - categorical_accuracy: 0.9385
47584/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9386
47616/60000 [======================>.......] - ETA: 23s - loss: 0.2054 - categorical_accuracy: 0.9386
47648/60000 [======================>.......] - ETA: 23s - loss: 0.2053 - categorical_accuracy: 0.9386
47680/60000 [======================>.......] - ETA: 23s - loss: 0.2052 - categorical_accuracy: 0.9386
47712/60000 [======================>.......] - ETA: 23s - loss: 0.2051 - categorical_accuracy: 0.9386
47744/60000 [======================>.......] - ETA: 23s - loss: 0.2050 - categorical_accuracy: 0.9387
47776/60000 [======================>.......] - ETA: 23s - loss: 0.2049 - categorical_accuracy: 0.9387
47808/60000 [======================>.......] - ETA: 23s - loss: 0.2048 - categorical_accuracy: 0.9387
47840/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9388
47872/60000 [======================>.......] - ETA: 23s - loss: 0.2048 - categorical_accuracy: 0.9387
47904/60000 [======================>.......] - ETA: 23s - loss: 0.2048 - categorical_accuracy: 0.9387
47936/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9387
47968/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9387
48000/60000 [=======================>......] - ETA: 23s - loss: 0.2045 - categorical_accuracy: 0.9388
48032/60000 [=======================>......] - ETA: 23s - loss: 0.2044 - categorical_accuracy: 0.9388
48064/60000 [=======================>......] - ETA: 23s - loss: 0.2044 - categorical_accuracy: 0.9388
48096/60000 [=======================>......] - ETA: 23s - loss: 0.2045 - categorical_accuracy: 0.9388
48128/60000 [=======================>......] - ETA: 22s - loss: 0.2043 - categorical_accuracy: 0.9388
48160/60000 [=======================>......] - ETA: 22s - loss: 0.2042 - categorical_accuracy: 0.9389
48192/60000 [=======================>......] - ETA: 22s - loss: 0.2041 - categorical_accuracy: 0.9389
48224/60000 [=======================>......] - ETA: 22s - loss: 0.2042 - categorical_accuracy: 0.9389
48256/60000 [=======================>......] - ETA: 22s - loss: 0.2041 - categorical_accuracy: 0.9389
48288/60000 [=======================>......] - ETA: 22s - loss: 0.2040 - categorical_accuracy: 0.9389
48320/60000 [=======================>......] - ETA: 22s - loss: 0.2039 - categorical_accuracy: 0.9390
48352/60000 [=======================>......] - ETA: 22s - loss: 0.2038 - categorical_accuracy: 0.9390
48384/60000 [=======================>......] - ETA: 22s - loss: 0.2036 - categorical_accuracy: 0.9391
48416/60000 [=======================>......] - ETA: 22s - loss: 0.2035 - categorical_accuracy: 0.9391
48448/60000 [=======================>......] - ETA: 22s - loss: 0.2034 - categorical_accuracy: 0.9392
48480/60000 [=======================>......] - ETA: 22s - loss: 0.2034 - categorical_accuracy: 0.9392
48512/60000 [=======================>......] - ETA: 22s - loss: 0.2033 - categorical_accuracy: 0.9392
48544/60000 [=======================>......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9393
48576/60000 [=======================>......] - ETA: 22s - loss: 0.2031 - categorical_accuracy: 0.9393
48608/60000 [=======================>......] - ETA: 22s - loss: 0.2031 - categorical_accuracy: 0.9393
48640/60000 [=======================>......] - ETA: 21s - loss: 0.2030 - categorical_accuracy: 0.9393
48672/60000 [=======================>......] - ETA: 21s - loss: 0.2029 - categorical_accuracy: 0.9393
48704/60000 [=======================>......] - ETA: 21s - loss: 0.2031 - categorical_accuracy: 0.9393
48736/60000 [=======================>......] - ETA: 21s - loss: 0.2030 - categorical_accuracy: 0.9393
48768/60000 [=======================>......] - ETA: 21s - loss: 0.2029 - categorical_accuracy: 0.9393
48800/60000 [=======================>......] - ETA: 21s - loss: 0.2028 - categorical_accuracy: 0.9394
48832/60000 [=======================>......] - ETA: 21s - loss: 0.2027 - categorical_accuracy: 0.9394
48864/60000 [=======================>......] - ETA: 21s - loss: 0.2026 - categorical_accuracy: 0.9395
48896/60000 [=======================>......] - ETA: 21s - loss: 0.2025 - categorical_accuracy: 0.9395
48928/60000 [=======================>......] - ETA: 21s - loss: 0.2026 - categorical_accuracy: 0.9395
48960/60000 [=======================>......] - ETA: 21s - loss: 0.2025 - categorical_accuracy: 0.9395
48992/60000 [=======================>......] - ETA: 21s - loss: 0.2024 - categorical_accuracy: 0.9395
49024/60000 [=======================>......] - ETA: 21s - loss: 0.2023 - categorical_accuracy: 0.9396
49056/60000 [=======================>......] - ETA: 21s - loss: 0.2022 - categorical_accuracy: 0.9396
49088/60000 [=======================>......] - ETA: 21s - loss: 0.2022 - categorical_accuracy: 0.9396
49120/60000 [=======================>......] - ETA: 21s - loss: 0.2022 - categorical_accuracy: 0.9396
49152/60000 [=======================>......] - ETA: 21s - loss: 0.2022 - categorical_accuracy: 0.9396
49184/60000 [=======================>......] - ETA: 20s - loss: 0.2021 - categorical_accuracy: 0.9397
49216/60000 [=======================>......] - ETA: 20s - loss: 0.2020 - categorical_accuracy: 0.9397
49248/60000 [=======================>......] - ETA: 20s - loss: 0.2019 - categorical_accuracy: 0.9397
49280/60000 [=======================>......] - ETA: 20s - loss: 0.2018 - categorical_accuracy: 0.9398
49312/60000 [=======================>......] - ETA: 20s - loss: 0.2018 - categorical_accuracy: 0.9398
49344/60000 [=======================>......] - ETA: 20s - loss: 0.2017 - categorical_accuracy: 0.9398
49376/60000 [=======================>......] - ETA: 20s - loss: 0.2017 - categorical_accuracy: 0.9398
49408/60000 [=======================>......] - ETA: 20s - loss: 0.2016 - categorical_accuracy: 0.9398
49440/60000 [=======================>......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9399
49472/60000 [=======================>......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9399
49504/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9399
49536/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9399
49568/60000 [=======================>......] - ETA: 20s - loss: 0.2012 - categorical_accuracy: 0.9400
49600/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9399
49632/60000 [=======================>......] - ETA: 20s - loss: 0.2012 - categorical_accuracy: 0.9399
49664/60000 [=======================>......] - ETA: 20s - loss: 0.2012 - categorical_accuracy: 0.9400
49696/60000 [=======================>......] - ETA: 19s - loss: 0.2011 - categorical_accuracy: 0.9400
49728/60000 [=======================>......] - ETA: 19s - loss: 0.2012 - categorical_accuracy: 0.9400
49760/60000 [=======================>......] - ETA: 19s - loss: 0.2012 - categorical_accuracy: 0.9400
49792/60000 [=======================>......] - ETA: 19s - loss: 0.2011 - categorical_accuracy: 0.9400
49824/60000 [=======================>......] - ETA: 19s - loss: 0.2011 - categorical_accuracy: 0.9400
49856/60000 [=======================>......] - ETA: 19s - loss: 0.2010 - categorical_accuracy: 0.9400
49888/60000 [=======================>......] - ETA: 19s - loss: 0.2009 - categorical_accuracy: 0.9400
49920/60000 [=======================>......] - ETA: 19s - loss: 0.2008 - categorical_accuracy: 0.9401
49952/60000 [=======================>......] - ETA: 19s - loss: 0.2008 - categorical_accuracy: 0.9401
49984/60000 [=======================>......] - ETA: 19s - loss: 0.2007 - categorical_accuracy: 0.9401
50016/60000 [========================>.....] - ETA: 19s - loss: 0.2006 - categorical_accuracy: 0.9401
50048/60000 [========================>.....] - ETA: 19s - loss: 0.2005 - categorical_accuracy: 0.9401
50080/60000 [========================>.....] - ETA: 19s - loss: 0.2006 - categorical_accuracy: 0.9401
50112/60000 [========================>.....] - ETA: 19s - loss: 0.2005 - categorical_accuracy: 0.9401
50144/60000 [========================>.....] - ETA: 19s - loss: 0.2004 - categorical_accuracy: 0.9401
50176/60000 [========================>.....] - ETA: 19s - loss: 0.2003 - categorical_accuracy: 0.9402
50208/60000 [========================>.....] - ETA: 18s - loss: 0.2002 - categorical_accuracy: 0.9402
50240/60000 [========================>.....] - ETA: 18s - loss: 0.2002 - categorical_accuracy: 0.9402
50272/60000 [========================>.....] - ETA: 18s - loss: 0.2002 - categorical_accuracy: 0.9402
50304/60000 [========================>.....] - ETA: 18s - loss: 0.2001 - categorical_accuracy: 0.9402
50336/60000 [========================>.....] - ETA: 18s - loss: 0.2001 - categorical_accuracy: 0.9402
50368/60000 [========================>.....] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9403
50400/60000 [========================>.....] - ETA: 18s - loss: 0.2001 - categorical_accuracy: 0.9402
50432/60000 [========================>.....] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9403
50464/60000 [========================>.....] - ETA: 18s - loss: 0.1999 - categorical_accuracy: 0.9403
50496/60000 [========================>.....] - ETA: 18s - loss: 0.1999 - categorical_accuracy: 0.9403
50528/60000 [========================>.....] - ETA: 18s - loss: 0.1998 - categorical_accuracy: 0.9403
50560/60000 [========================>.....] - ETA: 18s - loss: 0.1997 - categorical_accuracy: 0.9404
50592/60000 [========================>.....] - ETA: 18s - loss: 0.1996 - categorical_accuracy: 0.9404
50624/60000 [========================>.....] - ETA: 18s - loss: 0.1995 - categorical_accuracy: 0.9404
50656/60000 [========================>.....] - ETA: 18s - loss: 0.1994 - categorical_accuracy: 0.9405
50688/60000 [========================>.....] - ETA: 18s - loss: 0.1994 - categorical_accuracy: 0.9405
50720/60000 [========================>.....] - ETA: 17s - loss: 0.1993 - categorical_accuracy: 0.9405
50752/60000 [========================>.....] - ETA: 17s - loss: 0.1991 - categorical_accuracy: 0.9406
50784/60000 [========================>.....] - ETA: 17s - loss: 0.1990 - categorical_accuracy: 0.9406
50816/60000 [========================>.....] - ETA: 17s - loss: 0.1990 - categorical_accuracy: 0.9406
50848/60000 [========================>.....] - ETA: 17s - loss: 0.1990 - categorical_accuracy: 0.9406
50880/60000 [========================>.....] - ETA: 17s - loss: 0.1989 - categorical_accuracy: 0.9407
50912/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9407
50944/60000 [========================>.....] - ETA: 17s - loss: 0.1989 - categorical_accuracy: 0.9407
50976/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9407
51008/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9408
51040/60000 [========================>.....] - ETA: 17s - loss: 0.1987 - categorical_accuracy: 0.9408
51072/60000 [========================>.....] - ETA: 17s - loss: 0.1985 - categorical_accuracy: 0.9408
51104/60000 [========================>.....] - ETA: 17s - loss: 0.1984 - categorical_accuracy: 0.9409
51136/60000 [========================>.....] - ETA: 17s - loss: 0.1983 - categorical_accuracy: 0.9409
51168/60000 [========================>.....] - ETA: 17s - loss: 0.1982 - categorical_accuracy: 0.9409
51200/60000 [========================>.....] - ETA: 17s - loss: 0.1981 - categorical_accuracy: 0.9410
51232/60000 [========================>.....] - ETA: 16s - loss: 0.1982 - categorical_accuracy: 0.9409
51264/60000 [========================>.....] - ETA: 16s - loss: 0.1981 - categorical_accuracy: 0.9410
51296/60000 [========================>.....] - ETA: 16s - loss: 0.1981 - categorical_accuracy: 0.9409
51328/60000 [========================>.....] - ETA: 16s - loss: 0.1980 - categorical_accuracy: 0.9410
51360/60000 [========================>.....] - ETA: 16s - loss: 0.1979 - categorical_accuracy: 0.9410
51392/60000 [========================>.....] - ETA: 16s - loss: 0.1981 - categorical_accuracy: 0.9410
51424/60000 [========================>.....] - ETA: 16s - loss: 0.1980 - categorical_accuracy: 0.9410
51456/60000 [========================>.....] - ETA: 16s - loss: 0.1980 - categorical_accuracy: 0.9410
51488/60000 [========================>.....] - ETA: 16s - loss: 0.1979 - categorical_accuracy: 0.9410
51520/60000 [========================>.....] - ETA: 16s - loss: 0.1978 - categorical_accuracy: 0.9411
51552/60000 [========================>.....] - ETA: 16s - loss: 0.1977 - categorical_accuracy: 0.9411
51584/60000 [========================>.....] - ETA: 16s - loss: 0.1976 - categorical_accuracy: 0.9411
51616/60000 [========================>.....] - ETA: 16s - loss: 0.1975 - categorical_accuracy: 0.9412
51648/60000 [========================>.....] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9412
51680/60000 [========================>.....] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9412
51712/60000 [========================>.....] - ETA: 16s - loss: 0.1973 - categorical_accuracy: 0.9412
51744/60000 [========================>.....] - ETA: 15s - loss: 0.1972 - categorical_accuracy: 0.9413
51776/60000 [========================>.....] - ETA: 15s - loss: 0.1972 - categorical_accuracy: 0.9413
51808/60000 [========================>.....] - ETA: 15s - loss: 0.1971 - categorical_accuracy: 0.9413
51840/60000 [========================>.....] - ETA: 15s - loss: 0.1970 - categorical_accuracy: 0.9414
51872/60000 [========================>.....] - ETA: 15s - loss: 0.1970 - categorical_accuracy: 0.9413
51904/60000 [========================>.....] - ETA: 15s - loss: 0.1969 - categorical_accuracy: 0.9414
51936/60000 [========================>.....] - ETA: 15s - loss: 0.1968 - categorical_accuracy: 0.9414
51968/60000 [========================>.....] - ETA: 15s - loss: 0.1969 - categorical_accuracy: 0.9414
52000/60000 [=========================>....] - ETA: 15s - loss: 0.1968 - categorical_accuracy: 0.9414
52032/60000 [=========================>....] - ETA: 15s - loss: 0.1968 - categorical_accuracy: 0.9414
52064/60000 [=========================>....] - ETA: 15s - loss: 0.1968 - categorical_accuracy: 0.9414
52096/60000 [=========================>....] - ETA: 15s - loss: 0.1967 - categorical_accuracy: 0.9415
52128/60000 [=========================>....] - ETA: 15s - loss: 0.1967 - categorical_accuracy: 0.9415
52160/60000 [=========================>....] - ETA: 15s - loss: 0.1966 - categorical_accuracy: 0.9415
52192/60000 [=========================>....] - ETA: 15s - loss: 0.1965 - categorical_accuracy: 0.9415
52224/60000 [=========================>....] - ETA: 15s - loss: 0.1964 - categorical_accuracy: 0.9416
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1963 - categorical_accuracy: 0.9416
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1962 - categorical_accuracy: 0.9416
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1962 - categorical_accuracy: 0.9416
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1961 - categorical_accuracy: 0.9416
52384/60000 [=========================>....] - ETA: 14s - loss: 0.1961 - categorical_accuracy: 0.9417
52416/60000 [=========================>....] - ETA: 14s - loss: 0.1960 - categorical_accuracy: 0.9417
52448/60000 [=========================>....] - ETA: 14s - loss: 0.1959 - categorical_accuracy: 0.9417
52480/60000 [=========================>....] - ETA: 14s - loss: 0.1958 - categorical_accuracy: 0.9417
52512/60000 [=========================>....] - ETA: 14s - loss: 0.1957 - categorical_accuracy: 0.9418
52544/60000 [=========================>....] - ETA: 14s - loss: 0.1956 - categorical_accuracy: 0.9418
52576/60000 [=========================>....] - ETA: 14s - loss: 0.1956 - categorical_accuracy: 0.9418
52608/60000 [=========================>....] - ETA: 14s - loss: 0.1954 - categorical_accuracy: 0.9419
52640/60000 [=========================>....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9419
52672/60000 [=========================>....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9419
52704/60000 [=========================>....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9419
52736/60000 [=========================>....] - ETA: 14s - loss: 0.1952 - categorical_accuracy: 0.9420
52768/60000 [=========================>....] - ETA: 14s - loss: 0.1951 - categorical_accuracy: 0.9420
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1950 - categorical_accuracy: 0.9420
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1949 - categorical_accuracy: 0.9420
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1948 - categorical_accuracy: 0.9421
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1947 - categorical_accuracy: 0.9421
52928/60000 [=========================>....] - ETA: 13s - loss: 0.1946 - categorical_accuracy: 0.9421
52960/60000 [=========================>....] - ETA: 13s - loss: 0.1946 - categorical_accuracy: 0.9421
52992/60000 [=========================>....] - ETA: 13s - loss: 0.1944 - categorical_accuracy: 0.9422
53024/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9422
53056/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9422
53088/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9423
53120/60000 [=========================>....] - ETA: 13s - loss: 0.1940 - categorical_accuracy: 0.9423
53152/60000 [=========================>....] - ETA: 13s - loss: 0.1940 - categorical_accuracy: 0.9423
53184/60000 [=========================>....] - ETA: 13s - loss: 0.1939 - categorical_accuracy: 0.9423
53216/60000 [=========================>....] - ETA: 13s - loss: 0.1938 - categorical_accuracy: 0.9423
53248/60000 [=========================>....] - ETA: 13s - loss: 0.1939 - categorical_accuracy: 0.9423
53280/60000 [=========================>....] - ETA: 13s - loss: 0.1939 - categorical_accuracy: 0.9423
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9424
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9423
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9423
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9423
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9423
53472/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9423
53504/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9424
53536/60000 [=========================>....] - ETA: 12s - loss: 0.1935 - categorical_accuracy: 0.9424
53568/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9424
53600/60000 [=========================>....] - ETA: 12s - loss: 0.1933 - categorical_accuracy: 0.9425
53632/60000 [=========================>....] - ETA: 12s - loss: 0.1932 - categorical_accuracy: 0.9425
53664/60000 [=========================>....] - ETA: 12s - loss: 0.1931 - categorical_accuracy: 0.9425
53696/60000 [=========================>....] - ETA: 12s - loss: 0.1930 - categorical_accuracy: 0.9425
53728/60000 [=========================>....] - ETA: 12s - loss: 0.1929 - categorical_accuracy: 0.9426
53760/60000 [=========================>....] - ETA: 12s - loss: 0.1928 - categorical_accuracy: 0.9426
53792/60000 [=========================>....] - ETA: 12s - loss: 0.1928 - categorical_accuracy: 0.9426
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9426
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9427
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9427
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9427
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9427
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9427
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9427
54048/60000 [==========================>...] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9428
54080/60000 [==========================>...] - ETA: 11s - loss: 0.1924 - categorical_accuracy: 0.9428
54112/60000 [==========================>...] - ETA: 11s - loss: 0.1923 - categorical_accuracy: 0.9428
54144/60000 [==========================>...] - ETA: 11s - loss: 0.1922 - categorical_accuracy: 0.9428
54176/60000 [==========================>...] - ETA: 11s - loss: 0.1922 - categorical_accuracy: 0.9428
54208/60000 [==========================>...] - ETA: 11s - loss: 0.1921 - categorical_accuracy: 0.9428
54240/60000 [==========================>...] - ETA: 11s - loss: 0.1920 - categorical_accuracy: 0.9429
54272/60000 [==========================>...] - ETA: 11s - loss: 0.1919 - categorical_accuracy: 0.9429
54304/60000 [==========================>...] - ETA: 11s - loss: 0.1920 - categorical_accuracy: 0.9429
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1919 - categorical_accuracy: 0.9429
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1918 - categorical_accuracy: 0.9429
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1917 - categorical_accuracy: 0.9430
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1916 - categorical_accuracy: 0.9430
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1916 - categorical_accuracy: 0.9430
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9430
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9430
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1914 - categorical_accuracy: 0.9430
54592/60000 [==========================>...] - ETA: 10s - loss: 0.1913 - categorical_accuracy: 0.9430
54624/60000 [==========================>...] - ETA: 10s - loss: 0.1914 - categorical_accuracy: 0.9430
54656/60000 [==========================>...] - ETA: 10s - loss: 0.1913 - categorical_accuracy: 0.9431
54688/60000 [==========================>...] - ETA: 10s - loss: 0.1912 - categorical_accuracy: 0.9431
54720/60000 [==========================>...] - ETA: 10s - loss: 0.1911 - categorical_accuracy: 0.9431
54752/60000 [==========================>...] - ETA: 10s - loss: 0.1910 - categorical_accuracy: 0.9431
54784/60000 [==========================>...] - ETA: 10s - loss: 0.1910 - categorical_accuracy: 0.9431
54816/60000 [==========================>...] - ETA: 10s - loss: 0.1910 - categorical_accuracy: 0.9432
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1909 - categorical_accuracy: 0.9432 
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1909 - categorical_accuracy: 0.9432
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1908 - categorical_accuracy: 0.9432
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9432
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9433
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9433
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1904 - categorical_accuracy: 0.9433
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1903 - categorical_accuracy: 0.9434
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1903 - categorical_accuracy: 0.9434
55136/60000 [==========================>...] - ETA: 9s - loss: 0.1902 - categorical_accuracy: 0.9434
55168/60000 [==========================>...] - ETA: 9s - loss: 0.1901 - categorical_accuracy: 0.9434
55200/60000 [==========================>...] - ETA: 9s - loss: 0.1900 - categorical_accuracy: 0.9435
55232/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9435
55264/60000 [==========================>...] - ETA: 9s - loss: 0.1898 - categorical_accuracy: 0.9435
55296/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9436
55328/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9435
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1897 - categorical_accuracy: 0.9436
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1896 - categorical_accuracy: 0.9436
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1895 - categorical_accuracy: 0.9436
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1894 - categorical_accuracy: 0.9436
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1893 - categorical_accuracy: 0.9437
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1892 - categorical_accuracy: 0.9437
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1891 - categorical_accuracy: 0.9437
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1891 - categorical_accuracy: 0.9437
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1890 - categorical_accuracy: 0.9438
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9438
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9438
55712/60000 [==========================>...] - ETA: 8s - loss: 0.1890 - categorical_accuracy: 0.9438
55744/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9438
55776/60000 [==========================>...] - ETA: 8s - loss: 0.1888 - categorical_accuracy: 0.9438
55808/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9438
55840/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9439
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1885 - categorical_accuracy: 0.9439
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1884 - categorical_accuracy: 0.9439
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1883 - categorical_accuracy: 0.9439
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1884 - categorical_accuracy: 0.9439
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1884 - categorical_accuracy: 0.9439
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1884 - categorical_accuracy: 0.9440
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1883 - categorical_accuracy: 0.9440
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1882 - categorical_accuracy: 0.9440
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9440
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9440
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9440
56224/60000 [===========================>..] - ETA: 7s - loss: 0.1879 - categorical_accuracy: 0.9440
56256/60000 [===========================>..] - ETA: 7s - loss: 0.1878 - categorical_accuracy: 0.9441
56288/60000 [===========================>..] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9441
56320/60000 [===========================>..] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9441
56352/60000 [===========================>..] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9442
56384/60000 [===========================>..] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9442
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1874 - categorical_accuracy: 0.9442
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1874 - categorical_accuracy: 0.9442
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1873 - categorical_accuracy: 0.9442
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1872 - categorical_accuracy: 0.9442
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9443
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9443
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9443
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9443
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9443
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9444
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9444
56768/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9444
56800/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9444
56832/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9445
56864/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9444
56896/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9445
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9445
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9445
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9446
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9445
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9445
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1861 - categorical_accuracy: 0.9446
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1861 - categorical_accuracy: 0.9446
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9446
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9446
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9447
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1857 - categorical_accuracy: 0.9447
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1857 - categorical_accuracy: 0.9447
57312/60000 [===========================>..] - ETA: 5s - loss: 0.1856 - categorical_accuracy: 0.9447
57344/60000 [===========================>..] - ETA: 5s - loss: 0.1855 - categorical_accuracy: 0.9448
57376/60000 [===========================>..] - ETA: 5s - loss: 0.1854 - categorical_accuracy: 0.9448
57408/60000 [===========================>..] - ETA: 5s - loss: 0.1853 - categorical_accuracy: 0.9448
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9448
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9448
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9448
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9448
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9448
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9449
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9449
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1849 - categorical_accuracy: 0.9449
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1848 - categorical_accuracy: 0.9449
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1848 - categorical_accuracy: 0.9450
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9450
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1848 - categorical_accuracy: 0.9450
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1848 - categorical_accuracy: 0.9450
57856/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9450
57888/60000 [===========================>..] - ETA: 4s - loss: 0.1846 - categorical_accuracy: 0.9450
57920/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9450
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9450
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9450
58016/60000 [============================>.] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9450
58048/60000 [============================>.] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9450
58080/60000 [============================>.] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9450
58112/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9451
58144/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9451
58176/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9451
58208/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9451
58240/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9451
58272/60000 [============================>.] - ETA: 3s - loss: 0.1842 - categorical_accuracy: 0.9452
58304/60000 [============================>.] - ETA: 3s - loss: 0.1842 - categorical_accuracy: 0.9452
58336/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9452
58368/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9452
58400/60000 [============================>.] - ETA: 3s - loss: 0.1842 - categorical_accuracy: 0.9452
58432/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9452
58464/60000 [============================>.] - ETA: 2s - loss: 0.1842 - categorical_accuracy: 0.9452
58496/60000 [============================>.] - ETA: 2s - loss: 0.1841 - categorical_accuracy: 0.9452
58528/60000 [============================>.] - ETA: 2s - loss: 0.1841 - categorical_accuracy: 0.9452
58560/60000 [============================>.] - ETA: 2s - loss: 0.1841 - categorical_accuracy: 0.9452
58592/60000 [============================>.] - ETA: 2s - loss: 0.1840 - categorical_accuracy: 0.9452
58624/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9453
58656/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9453
58688/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9453
58720/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9453
58752/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9454
58784/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9454
58816/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9454
58848/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9454
58880/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9454
58912/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9454
58944/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9454
58976/60000 [============================>.] - ETA: 1s - loss: 0.1834 - categorical_accuracy: 0.9454
59008/60000 [============================>.] - ETA: 1s - loss: 0.1834 - categorical_accuracy: 0.9454
59040/60000 [============================>.] - ETA: 1s - loss: 0.1833 - categorical_accuracy: 0.9454
59072/60000 [============================>.] - ETA: 1s - loss: 0.1833 - categorical_accuracy: 0.9455
59104/60000 [============================>.] - ETA: 1s - loss: 0.1832 - categorical_accuracy: 0.9455
59136/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9455
59168/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9455
59200/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9455
59232/60000 [============================>.] - ETA: 1s - loss: 0.1830 - categorical_accuracy: 0.9456
59264/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9456
59296/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9456
59328/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9456
59360/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9456
59392/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9456
59424/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9456
59456/60000 [============================>.] - ETA: 1s - loss: 0.1827 - categorical_accuracy: 0.9456
59488/60000 [============================>.] - ETA: 0s - loss: 0.1827 - categorical_accuracy: 0.9456
59520/60000 [============================>.] - ETA: 0s - loss: 0.1827 - categorical_accuracy: 0.9456
59552/60000 [============================>.] - ETA: 0s - loss: 0.1826 - categorical_accuracy: 0.9457
59584/60000 [============================>.] - ETA: 0s - loss: 0.1826 - categorical_accuracy: 0.9457
59616/60000 [============================>.] - ETA: 0s - loss: 0.1825 - categorical_accuracy: 0.9457
59648/60000 [============================>.] - ETA: 0s - loss: 0.1824 - categorical_accuracy: 0.9457
59680/60000 [============================>.] - ETA: 0s - loss: 0.1824 - categorical_accuracy: 0.9457
59712/60000 [============================>.] - ETA: 0s - loss: 0.1823 - categorical_accuracy: 0.9458
59744/60000 [============================>.] - ETA: 0s - loss: 0.1822 - categorical_accuracy: 0.9458
59776/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9458
59808/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9458
59840/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9459
59872/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9459
59904/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9459
59936/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9459
59968/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9459
60000/60000 [==============================] - 120s 2ms/step - loss: 0.1818 - categorical_accuracy: 0.9459 - val_loss: 0.0511 - val_categorical_accuracy: 0.9841

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  672/10000 [=>............................] - ETA: 4s
  800/10000 [=>............................] - ETA: 3s
  928/10000 [=>............................] - ETA: 3s
 1056/10000 [==>...........................] - ETA: 3s
 1216/10000 [==>...........................] - ETA: 3s
 1376/10000 [===>..........................] - ETA: 3s
 1536/10000 [===>..........................] - ETA: 3s
 1664/10000 [===>..........................] - ETA: 3s
 1824/10000 [====>.........................] - ETA: 3s
 1984/10000 [====>.........................] - ETA: 3s
 2144/10000 [=====>........................] - ETA: 3s
 2304/10000 [=====>........................] - ETA: 3s
 2464/10000 [======>.......................] - ETA: 3s
 2624/10000 [======>.......................] - ETA: 2s
 2784/10000 [=======>......................] - ETA: 2s
 2944/10000 [=======>......................] - ETA: 2s
 3104/10000 [========>.....................] - ETA: 2s
 3264/10000 [========>.....................] - ETA: 2s
 3424/10000 [=========>....................] - ETA: 2s
 3552/10000 [=========>....................] - ETA: 2s
 3712/10000 [==========>...................] - ETA: 2s
 3840/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 2s
 4320/10000 [===========>..................] - ETA: 2s
 4480/10000 [============>.................] - ETA: 2s
 4640/10000 [============>.................] - ETA: 2s
 4768/10000 [=============>................] - ETA: 2s
 4928/10000 [=============>................] - ETA: 1s
 5088/10000 [==============>...............] - ETA: 1s
 5248/10000 [==============>...............] - ETA: 1s
 5408/10000 [===============>..............] - ETA: 1s
 5568/10000 [===============>..............] - ETA: 1s
 5728/10000 [================>.............] - ETA: 1s
 5888/10000 [================>.............] - ETA: 1s
 6048/10000 [=================>............] - ETA: 1s
 6208/10000 [=================>............] - ETA: 1s
 6368/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 1s
 6688/10000 [===================>..........] - ETA: 1s
 6848/10000 [===================>..........] - ETA: 1s
 7008/10000 [====================>.........] - ETA: 1s
 7168/10000 [====================>.........] - ETA: 1s
 7296/10000 [====================>.........] - ETA: 1s
 7456/10000 [=====================>........] - ETA: 0s
 7616/10000 [=====================>........] - ETA: 0s
 7776/10000 [======================>.......] - ETA: 0s
 7936/10000 [======================>.......] - ETA: 0s
 8096/10000 [=======================>......] - ETA: 0s
 8256/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8576/10000 [========================>.....] - ETA: 0s
 8736/10000 [=========================>....] - ETA: 0s
 8864/10000 [=========================>....] - ETA: 0s
 8992/10000 [=========================>....] - ETA: 0s
 9120/10000 [==========================>...] - ETA: 0s
 9280/10000 [==========================>...] - ETA: 0s
 9440/10000 [===========================>..] - ETA: 0s
 9600/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 385us/step
[[8.6585059e-09 2.0540323e-08 1.6702048e-06 ... 9.9999654e-01
  4.9618163e-09 1.0344066e-06]
 [1.6195478e-05 2.3945911e-05 9.9992883e-01 ... 2.7530351e-08
  4.2974516e-06 8.0570534e-10]
 [1.0508988e-06 9.9988043e-01 5.7498710e-06 ... 2.9092736e-05
  9.6124950e-06 1.0207461e-06]
 ...
 [1.0119346e-09 4.5157751e-08 8.1180751e-10 ... 4.3167378e-07
  6.7945507e-07 5.3592212e-06]
 [1.6573771e-05 1.3419418e-07 6.7428310e-08 ... 3.5229641e-07
  3.2655324e-03 2.8016427e-06]
 [6.0341966e-07 6.3686400e-08 6.2035588e-07 ... 3.3296566e-10
  6.4415858e-08 7.7096340e-10]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.0511123079446028, 'accuracy_test:': 0.9840999841690063}

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
   92a3cf0..7dfc470  master     -> origin/master
Updating 92a3cf0..7dfc470
Fast-forward
 .../20200523/list_log_pullrequest_20200523.md      |   2 +-
 error_list/20200523/list_log_testall_20200523.md   | 432 +++++++++++++++++++++
 2 files changed, 433 insertions(+), 1 deletion(-)
