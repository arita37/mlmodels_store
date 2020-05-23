
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '7423a9c1aea8d708841a3941e104542978e088ce', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/7423a9c1aea8d708841a3941e104542978e088ce

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/7423a9c1aea8d708841a3941e104542978e088ce

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
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   07d27d6..f7e5309  master     -> origin/master
Updating 07d27d6..f7e5309
Fast-forward
 .../20200523/list_log_pullrequest_20200523.md      |   2 +-
 ...-17_7423a9c1aea8d708841a3941e104542978e088ce.py | 627 +++++++++++++++++++++
 2 files changed, 628 insertions(+), 1 deletion(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-23-00-17_7423a9c1aea8d708841a3941e104542978e088ce.py
[master a5c82e3] ml_store
 1 file changed, 70 insertions(+), 11026 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   f7e5309..a5c82e3  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
Already up to date.
[master f821e8c] ml_store
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   a5c82e3..f821e8c  master -> master





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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-23 00:22:11.453454: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 00:22:11.458553: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-23 00:22:11.458757: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c7b2c4f4f0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 00:22:11.458773: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 238
Trainable params: 238
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2501 - binary_crossentropy: 0.6934 - val_loss: 0.2501 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.24986638216092866}

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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
Total params: 238
Trainable params: 238
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 443
Trainable params: 443
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2467 - binary_crossentropy: 0.6865500/500 [==============================] - 1s 1ms/sample - loss: 0.2578 - binary_crossentropy: 0.7090 - val_loss: 0.2541 - val_binary_crossentropy: 0.7015

  #### metrics   #################################################### 
{'MSE': 0.25572915132726454}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 443
Trainable params: 443
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 707
Trainable params: 707
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2950 - binary_crossentropy: 1.8301500/500 [==============================] - 1s 2ms/sample - loss: 0.2715 - binary_crossentropy: 1.4107 - val_loss: 0.2774 - val_binary_crossentropy: 1.5030

  #### metrics   #################################################### 
{'MSE': 0.2743004435418703}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
Total params: 707
Trainable params: 707
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 433
Trainable params: 433
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2837 - binary_crossentropy: 1.0256500/500 [==============================] - 1s 2ms/sample - loss: 0.2713 - binary_crossentropy: 0.8422 - val_loss: 0.2617 - val_binary_crossentropy: 0.8483

  #### metrics   #################################################### 
{'MSE': 0.26446122111033504}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_max (Em (None, 6, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
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
Total params: 148
Trainable params: 148
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3089 - binary_crossentropy: 0.8318500/500 [==============================] - 1s 3ms/sample - loss: 0.2898 - binary_crossentropy: 0.7848 - val_loss: 0.2852 - val_binary_crossentropy: 0.7735

  #### metrics   #################################################### 
{'MSE': 0.28660993126349965}

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
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_max (Em (None, 6, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
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
Total params: 148
Trainable params: 148
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-23 00:23:22.293956: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:22.296285: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:22.302657: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-23 00:23:22.313809: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-23 00:23:22.316137: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 00:23:22.317870: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:22.319914: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2552 - val_binary_crossentropy: 0.7035
2020-05-23 00:23:23.469668: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:23.471642: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:23.475816: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-23 00:23:23.483764: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-23 00:23:23.485462: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 00:23:23.487007: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:23.488304: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2564847183431483}

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
2020-05-23 00:23:44.652382: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:44.653631: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:44.657010: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-23 00:23:44.662893: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-23 00:23:44.663898: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 00:23:44.664827: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:44.665740: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934
2020-05-23 00:23:46.122012: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:46.123098: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:46.125579: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-23 00:23:46.130688: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-23 00:23:46.131569: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 00:23:46.132395: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:23:46.133173: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25007487424707797}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-23 00:24:17.108715: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:24:17.113529: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:24:17.126930: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-23 00:24:17.150116: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-23 00:24:17.154040: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 00:24:17.157759: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:24:17.163437: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 4s 4s/sample - loss: 0.2198 - binary_crossentropy: 0.6326 - val_loss: 0.2685 - val_binary_crossentropy: 0.7315
2020-05-23 00:24:19.260520: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:24:19.264885: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:24:19.278061: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-23 00:24:19.303404: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-23 00:24:19.307503: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-23 00:24:19.311120: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-23 00:24:19.314541: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.22321392819589292}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 665
Trainable params: 665
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3116 - binary_crossentropy: 1.9870500/500 [==============================] - 4s 8ms/sample - loss: 0.3375 - binary_crossentropy: 2.2364 - val_loss: 0.3237 - val_binary_crossentropy: 2.1776

  #### metrics   #################################################### 
{'MSE': 0.32935599257475373}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 665
Trainable params: 665
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 2)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         14          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         2           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 6s - loss: 0.2627 - binary_crossentropy: 0.7153500/500 [==============================] - 4s 8ms/sample - loss: 0.2665 - binary_crossentropy: 0.8334 - val_loss: 0.2521 - val_binary_crossentropy: 0.8011

  #### metrics   #################################################### 
{'MSE': 0.25666526299341413}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 2)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         14          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         2           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 1,954
Trainable params: 1,954
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.2914 - binary_crossentropy: 0.7856500/500 [==============================] - 4s 8ms/sample - loss: 0.2706 - binary_crossentropy: 0.7420 - val_loss: 0.2677 - val_binary_crossentropy: 0.7366

  #### metrics   #################################################### 
{'MSE': 0.268137870899321}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 1,954
Trainable params: 1,954
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
regionsequence_sum (InputLayer) [(None, 2)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 2)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 3)]          0                                            
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
region_10sparse_seq_emb_regions (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2459 - binary_crossentropy: 0.6828500/500 [==============================] - 5s 11ms/sample - loss: 0.2531 - binary_crossentropy: 0.8023 - val_loss: 0.2546 - val_binary_crossentropy: 0.7815

  #### metrics   #################################################### 
{'MSE': 0.2536465563444088}

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
regionsequence_sum (InputLayer) [(None, 2)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 2)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 3)]          0                                            
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
region_10sparse_seq_emb_regions (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         4           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 6s - loss: 0.3367 - binary_crossentropy: 2.0527500/500 [==============================] - 5s 10ms/sample - loss: 0.3354 - binary_crossentropy: 2.1808 - val_loss: 0.3190 - val_binary_crossentropy: 1.9334

  #### metrics   #################################################### 
{'MSE': 0.3254280787287262}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
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
Total params: 2,814
Trainable params: 2,734
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.4500 - binary_crossentropy: 6.9412500/500 [==============================] - 6s 12ms/sample - loss: 0.4900 - binary_crossentropy: 7.5582 - val_loss: 0.5360 - val_binary_crossentropy: 8.2678

  #### metrics   #################################################### 
{'MSE': 0.513}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 1, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 4, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
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
Total params: 2,814
Trainable params: 2,734
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
   f821e8c..6c6b316  master     -> origin/master
Updating f821e8c..6c6b316
Fast-forward
 .../20200523/list_log_pullrequest_20200523.md      |   2 +-
 error_list/20200523/list_log_testall_20200523.md   | 773 +--------------------
 2 files changed, 6 insertions(+), 769 deletions(-)
[master 2d4c453] ml_store
 1 file changed, 4953 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   6c6b316..2d4c453  master -> master





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
[master 333f132] ml_store
 1 file changed, 52 insertions(+)
To github.com:arita37/mlmodels_store.git
   2d4c453..333f132  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
Already up to date.
[master 978f545] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   333f132..978f545  master -> master





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
[master 352d927] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   978f545..352d927  master -> master





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

2020-05-23 00:32:37.187740: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 00:32:37.192258: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-23 00:32:37.192416: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fa0f9f7e60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 00:32:37.192431: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3876
256/354 [====================>.........] - ETA: 3s - loss: 1.2554
354/354 [==============================] - 15s 42ms/step - loss: 1.3306 - val_loss: 2.4176

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
[master 2a086fe] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   352d927..2a086fe  master -> master





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
[master 90204e5] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   2a086fe..90204e5  master -> master





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
[master a877a11] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   90204e5..a877a11  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3252224/17464789 [====>.........................] - ETA: 0s
12009472/17464789 [===================>..........] - ETA: 0s
17080320/17464789 [============================>.] - ETA: 0s
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
2020-05-23 00:33:38.471834: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-23 00:33:38.476016: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-23 00:33:38.476157: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fb75ffce00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-23 00:33:38.476171: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8353 - accuracy: 0.4890
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7356 - accuracy: 0.4955 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7177 - accuracy: 0.4967
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7625 - accuracy: 0.4938
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7648 - accuracy: 0.4936
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7842 - accuracy: 0.4923
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7915 - accuracy: 0.4919
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7970 - accuracy: 0.4915
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7586 - accuracy: 0.4940
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7479 - accuracy: 0.4947
11000/25000 [============>.................] - ETA: 4s - loss: 7.7112 - accuracy: 0.4971
12000/25000 [=============>................] - ETA: 3s - loss: 7.6922 - accuracy: 0.4983
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6867 - accuracy: 0.4987
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6907 - accuracy: 0.4984
15000/25000 [=================>............] - ETA: 3s - loss: 7.6728 - accuracy: 0.4996
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6877 - accuracy: 0.4986
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7162 - accuracy: 0.4968
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7152 - accuracy: 0.4968
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7110 - accuracy: 0.4971
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7142 - accuracy: 0.4969
21000/25000 [========================>.....] - ETA: 1s - loss: 7.7214 - accuracy: 0.4964
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7328 - accuracy: 0.4957
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6946 - accuracy: 0.4982
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 9s 364us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f3c34671860>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f3c3eb378d0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9120 - accuracy: 0.4840
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7816 - accuracy: 0.4925 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7688 - accuracy: 0.4933
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7280 - accuracy: 0.4960
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7034 - accuracy: 0.4976
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7061 - accuracy: 0.4974
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7030 - accuracy: 0.4976
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7228 - accuracy: 0.4963
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7203 - accuracy: 0.4965
11000/25000 [============>.................] - ETA: 4s - loss: 7.7544 - accuracy: 0.4943
12000/25000 [=============>................] - ETA: 3s - loss: 7.7228 - accuracy: 0.4963
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6985 - accuracy: 0.4979
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6863 - accuracy: 0.4987
15000/25000 [=================>............] - ETA: 3s - loss: 7.6431 - accuracy: 0.5015
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6187 - accuracy: 0.5031
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6080 - accuracy: 0.5038
18000/25000 [====================>.........] - ETA: 2s - loss: 7.5951 - accuracy: 0.5047
19000/25000 [=====================>........] - ETA: 1s - loss: 7.5972 - accuracy: 0.5045
20000/25000 [=======================>......] - ETA: 1s - loss: 7.5992 - accuracy: 0.5044
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6082 - accuracy: 0.5038
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6269 - accuracy: 0.5026
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6300 - accuracy: 0.5024
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6321 - accuracy: 0.5023
25000/25000 [==============================] - 9s 363us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9733 - accuracy: 0.4800
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7050 - accuracy: 0.4975 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6820 - accuracy: 0.4990
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7586 - accuracy: 0.4940
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7770 - accuracy: 0.4928
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8353 - accuracy: 0.4890
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7586 - accuracy: 0.4940
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7356 - accuracy: 0.4955
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7450 - accuracy: 0.4949
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7448 - accuracy: 0.4949
11000/25000 [============>.................] - ETA: 4s - loss: 7.7140 - accuracy: 0.4969
12000/25000 [=============>................] - ETA: 3s - loss: 7.6998 - accuracy: 0.4978
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6949 - accuracy: 0.4982
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6940 - accuracy: 0.4982
15000/25000 [=================>............] - ETA: 2s - loss: 7.6932 - accuracy: 0.4983
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7011 - accuracy: 0.4978
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7000 - accuracy: 0.4978
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6728 - accuracy: 0.4996
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6695 - accuracy: 0.4998
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6806 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6773 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6788 - accuracy: 0.4992
25000/25000 [==============================] - 9s 363us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   a877a11..f149ab9  master     -> origin/master
Updating a877a11..f149ab9
Fast-forward
 error_list/20200523/list_log_testall_20200523.md | 103 +++++++++++++++++++++++
 1 file changed, 103 insertions(+)
[master 72edabd] ml_store
 1 file changed, 323 insertions(+)
To github.com:arita37/mlmodels_store.git
   f149ab9..72edabd  master -> master





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

13/13 [==============================] - 1s 107ms/step - loss: nan
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
[master 54f13b0] ml_store
 1 file changed, 126 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   72edabd..54f13b0  master -> master





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

    8192/11490434 [..............................] - ETA: 7s
 3571712/11490434 [========>.....................] - ETA: 0s
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

   32/60000 [..............................] - ETA: 6:43 - loss: 2.3029 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:15 - loss: 2.2934 - categorical_accuracy: 0.0781
   96/60000 [..............................] - ETA: 3:23 - loss: 2.2775 - categorical_accuracy: 0.1146
  128/60000 [..............................] - ETA: 2:58 - loss: 2.2638 - categorical_accuracy: 0.1484
  160/60000 [..............................] - ETA: 2:43 - loss: 2.2430 - categorical_accuracy: 0.1562
  192/60000 [..............................] - ETA: 2:34 - loss: 2.1977 - categorical_accuracy: 0.1927
  224/60000 [..............................] - ETA: 2:30 - loss: 2.1777 - categorical_accuracy: 0.2009
  256/60000 [..............................] - ETA: 2:24 - loss: 2.1361 - categorical_accuracy: 0.2266
  288/60000 [..............................] - ETA: 2:19 - loss: 2.1102 - categorical_accuracy: 0.2465
  320/60000 [..............................] - ETA: 2:15 - loss: 2.0825 - categorical_accuracy: 0.2656
  352/60000 [..............................] - ETA: 2:12 - loss: 2.0326 - categorical_accuracy: 0.2926
  384/60000 [..............................] - ETA: 2:10 - loss: 2.0450 - categorical_accuracy: 0.2943
  416/60000 [..............................] - ETA: 2:07 - loss: 2.0219 - categorical_accuracy: 0.3077
  448/60000 [..............................] - ETA: 2:05 - loss: 1.9927 - categorical_accuracy: 0.3214
  480/60000 [..............................] - ETA: 2:05 - loss: 1.9591 - categorical_accuracy: 0.3354
  512/60000 [..............................] - ETA: 2:03 - loss: 1.9171 - categorical_accuracy: 0.3535
  544/60000 [..............................] - ETA: 2:02 - loss: 1.8732 - categorical_accuracy: 0.3713
  576/60000 [..............................] - ETA: 2:03 - loss: 1.8631 - categorical_accuracy: 0.3767
  608/60000 [..............................] - ETA: 2:02 - loss: 1.8544 - categorical_accuracy: 0.3766
  640/60000 [..............................] - ETA: 2:01 - loss: 1.8260 - categorical_accuracy: 0.3922
  672/60000 [..............................] - ETA: 2:00 - loss: 1.8046 - categorical_accuracy: 0.3973
  704/60000 [..............................] - ETA: 1:59 - loss: 1.7722 - categorical_accuracy: 0.4062
  736/60000 [..............................] - ETA: 1:58 - loss: 1.7462 - categorical_accuracy: 0.4130
  768/60000 [..............................] - ETA: 1:58 - loss: 1.7206 - categorical_accuracy: 0.4232
  800/60000 [..............................] - ETA: 1:57 - loss: 1.7134 - categorical_accuracy: 0.4263
  832/60000 [..............................] - ETA: 1:57 - loss: 1.6904 - categorical_accuracy: 0.4339
  864/60000 [..............................] - ETA: 1:56 - loss: 1.6659 - categorical_accuracy: 0.4456
  896/60000 [..............................] - ETA: 1:55 - loss: 1.6399 - categorical_accuracy: 0.4542
  928/60000 [..............................] - ETA: 1:55 - loss: 1.6134 - categorical_accuracy: 0.4644
  960/60000 [..............................] - ETA: 1:55 - loss: 1.5944 - categorical_accuracy: 0.4729
  992/60000 [..............................] - ETA: 1:54 - loss: 1.5652 - categorical_accuracy: 0.4839
 1024/60000 [..............................] - ETA: 1:54 - loss: 1.5361 - categorical_accuracy: 0.4932
 1056/60000 [..............................] - ETA: 1:54 - loss: 1.5068 - categorical_accuracy: 0.5038
 1088/60000 [..............................] - ETA: 1:54 - loss: 1.4937 - categorical_accuracy: 0.5101
 1120/60000 [..............................] - ETA: 1:53 - loss: 1.4742 - categorical_accuracy: 0.5170
 1152/60000 [..............................] - ETA: 1:53 - loss: 1.4516 - categorical_accuracy: 0.5243
 1184/60000 [..............................] - ETA: 1:52 - loss: 1.4347 - categorical_accuracy: 0.5279
 1216/60000 [..............................] - ETA: 1:52 - loss: 1.4146 - categorical_accuracy: 0.5362
 1248/60000 [..............................] - ETA: 1:51 - loss: 1.4005 - categorical_accuracy: 0.5425
 1280/60000 [..............................] - ETA: 1:51 - loss: 1.3847 - categorical_accuracy: 0.5477
 1312/60000 [..............................] - ETA: 1:51 - loss: 1.3652 - categorical_accuracy: 0.5526
 1344/60000 [..............................] - ETA: 1:50 - loss: 1.3521 - categorical_accuracy: 0.5565
 1376/60000 [..............................] - ETA: 1:51 - loss: 1.3363 - categorical_accuracy: 0.5596
 1408/60000 [..............................] - ETA: 1:50 - loss: 1.3317 - categorical_accuracy: 0.5632
 1440/60000 [..............................] - ETA: 1:50 - loss: 1.3174 - categorical_accuracy: 0.5681
 1472/60000 [..............................] - ETA: 1:50 - loss: 1.3038 - categorical_accuracy: 0.5734
 1504/60000 [..............................] - ETA: 1:49 - loss: 1.2943 - categorical_accuracy: 0.5758
 1536/60000 [..............................] - ETA: 1:49 - loss: 1.2752 - categorical_accuracy: 0.5820
 1568/60000 [..............................] - ETA: 1:49 - loss: 1.2661 - categorical_accuracy: 0.5848
 1600/60000 [..............................] - ETA: 1:49 - loss: 1.2529 - categorical_accuracy: 0.5900
 1632/60000 [..............................] - ETA: 1:48 - loss: 1.2354 - categorical_accuracy: 0.5962
 1664/60000 [..............................] - ETA: 1:49 - loss: 1.2181 - categorical_accuracy: 0.6022
 1696/60000 [..............................] - ETA: 1:48 - loss: 1.2072 - categorical_accuracy: 0.6067
 1728/60000 [..............................] - ETA: 1:48 - loss: 1.1982 - categorical_accuracy: 0.6094
 1760/60000 [..............................] - ETA: 1:48 - loss: 1.1857 - categorical_accuracy: 0.6142
 1792/60000 [..............................] - ETA: 1:48 - loss: 1.1729 - categorical_accuracy: 0.6183
 1824/60000 [..............................] - ETA: 1:48 - loss: 1.1715 - categorical_accuracy: 0.6201
 1856/60000 [..............................] - ETA: 1:47 - loss: 1.1629 - categorical_accuracy: 0.6239
 1888/60000 [..............................] - ETA: 1:47 - loss: 1.1556 - categorical_accuracy: 0.6250
 1920/60000 [..............................] - ETA: 1:47 - loss: 1.1423 - categorical_accuracy: 0.6297
 1952/60000 [..............................] - ETA: 1:47 - loss: 1.1348 - categorical_accuracy: 0.6327
 1984/60000 [..............................] - ETA: 1:47 - loss: 1.1298 - categorical_accuracy: 0.6331
 2016/60000 [>.............................] - ETA: 1:46 - loss: 1.1162 - categorical_accuracy: 0.6379
 2048/60000 [>.............................] - ETA: 1:46 - loss: 1.1082 - categorical_accuracy: 0.6411
 2080/60000 [>.............................] - ETA: 1:46 - loss: 1.0994 - categorical_accuracy: 0.6438
 2112/60000 [>.............................] - ETA: 1:46 - loss: 1.0925 - categorical_accuracy: 0.6449
 2144/60000 [>.............................] - ETA: 1:46 - loss: 1.0837 - categorical_accuracy: 0.6479
 2176/60000 [>.............................] - ETA: 1:46 - loss: 1.0746 - categorical_accuracy: 0.6512
 2208/60000 [>.............................] - ETA: 1:45 - loss: 1.0724 - categorical_accuracy: 0.6526
 2240/60000 [>.............................] - ETA: 1:45 - loss: 1.0645 - categorical_accuracy: 0.6549
 2272/60000 [>.............................] - ETA: 1:45 - loss: 1.0525 - categorical_accuracy: 0.6589
 2304/60000 [>.............................] - ETA: 1:45 - loss: 1.0491 - categorical_accuracy: 0.6606
 2336/60000 [>.............................] - ETA: 1:45 - loss: 1.0460 - categorical_accuracy: 0.6622
 2368/60000 [>.............................] - ETA: 1:45 - loss: 1.0396 - categorical_accuracy: 0.6643
 2400/60000 [>.............................] - ETA: 1:45 - loss: 1.0301 - categorical_accuracy: 0.6675
 2432/60000 [>.............................] - ETA: 1:45 - loss: 1.0269 - categorical_accuracy: 0.6690
 2464/60000 [>.............................] - ETA: 1:45 - loss: 1.0216 - categorical_accuracy: 0.6709
 2496/60000 [>.............................] - ETA: 1:45 - loss: 1.0159 - categorical_accuracy: 0.6727
 2528/60000 [>.............................] - ETA: 1:44 - loss: 1.0098 - categorical_accuracy: 0.6744
 2560/60000 [>.............................] - ETA: 1:44 - loss: 1.0058 - categorical_accuracy: 0.6766
 2592/60000 [>.............................] - ETA: 1:44 - loss: 1.0015 - categorical_accuracy: 0.6794
 2624/60000 [>.............................] - ETA: 1:44 - loss: 0.9945 - categorical_accuracy: 0.6825
 2656/60000 [>.............................] - ETA: 1:44 - loss: 0.9936 - categorical_accuracy: 0.6826
 2688/60000 [>.............................] - ETA: 1:44 - loss: 0.9853 - categorical_accuracy: 0.6853
 2720/60000 [>.............................] - ETA: 1:44 - loss: 0.9792 - categorical_accuracy: 0.6868
 2752/60000 [>.............................] - ETA: 1:44 - loss: 0.9761 - categorical_accuracy: 0.6886
 2784/60000 [>.............................] - ETA: 1:43 - loss: 0.9701 - categorical_accuracy: 0.6907
 2816/60000 [>.............................] - ETA: 1:43 - loss: 0.9656 - categorical_accuracy: 0.6925
 2848/60000 [>.............................] - ETA: 1:43 - loss: 0.9619 - categorical_accuracy: 0.6931
 2880/60000 [>.............................] - ETA: 1:43 - loss: 0.9556 - categorical_accuracy: 0.6955
 2912/60000 [>.............................] - ETA: 1:43 - loss: 0.9515 - categorical_accuracy: 0.6975
 2944/60000 [>.............................] - ETA: 1:43 - loss: 0.9479 - categorical_accuracy: 0.6990
 2976/60000 [>.............................] - ETA: 1:43 - loss: 0.9420 - categorical_accuracy: 0.7006
 3008/60000 [>.............................] - ETA: 1:43 - loss: 0.9376 - categorical_accuracy: 0.7011
 3040/60000 [>.............................] - ETA: 1:42 - loss: 0.9339 - categorical_accuracy: 0.7020
 3072/60000 [>.............................] - ETA: 1:42 - loss: 0.9278 - categorical_accuracy: 0.7038
 3104/60000 [>.............................] - ETA: 1:42 - loss: 0.9234 - categorical_accuracy: 0.7052
 3136/60000 [>.............................] - ETA: 1:42 - loss: 0.9186 - categorical_accuracy: 0.7070
 3168/60000 [>.............................] - ETA: 1:42 - loss: 0.9130 - categorical_accuracy: 0.7083
 3200/60000 [>.............................] - ETA: 1:42 - loss: 0.9124 - categorical_accuracy: 0.7078
 3232/60000 [>.............................] - ETA: 1:42 - loss: 0.9085 - categorical_accuracy: 0.7092
 3264/60000 [>.............................] - ETA: 1:42 - loss: 0.9052 - categorical_accuracy: 0.7099
 3296/60000 [>.............................] - ETA: 1:42 - loss: 0.9004 - categorical_accuracy: 0.7115
 3328/60000 [>.............................] - ETA: 1:41 - loss: 0.8981 - categorical_accuracy: 0.7121
 3360/60000 [>.............................] - ETA: 1:41 - loss: 0.8918 - categorical_accuracy: 0.7143
 3392/60000 [>.............................] - ETA: 1:41 - loss: 0.8878 - categorical_accuracy: 0.7152
 3424/60000 [>.............................] - ETA: 1:41 - loss: 0.8816 - categorical_accuracy: 0.7170
 3456/60000 [>.............................] - ETA: 1:41 - loss: 0.8756 - categorical_accuracy: 0.7190
 3488/60000 [>.............................] - ETA: 1:41 - loss: 0.8718 - categorical_accuracy: 0.7199
 3520/60000 [>.............................] - ETA: 1:41 - loss: 0.8702 - categorical_accuracy: 0.7216
 3552/60000 [>.............................] - ETA: 1:41 - loss: 0.8661 - categorical_accuracy: 0.7230
 3584/60000 [>.............................] - ETA: 1:41 - loss: 0.8605 - categorical_accuracy: 0.7246
 3616/60000 [>.............................] - ETA: 1:41 - loss: 0.8630 - categorical_accuracy: 0.7243
 3648/60000 [>.............................] - ETA: 1:41 - loss: 0.8609 - categorical_accuracy: 0.7240
 3680/60000 [>.............................] - ETA: 1:40 - loss: 0.8585 - categorical_accuracy: 0.7245
 3712/60000 [>.............................] - ETA: 1:40 - loss: 0.8540 - categorical_accuracy: 0.7263
 3744/60000 [>.............................] - ETA: 1:40 - loss: 0.8487 - categorical_accuracy: 0.7281
 3776/60000 [>.............................] - ETA: 1:40 - loss: 0.8438 - categorical_accuracy: 0.7296
 3808/60000 [>.............................] - ETA: 1:40 - loss: 0.8398 - categorical_accuracy: 0.7308
 3840/60000 [>.............................] - ETA: 1:40 - loss: 0.8347 - categorical_accuracy: 0.7323
 3872/60000 [>.............................] - ETA: 1:40 - loss: 0.8297 - categorical_accuracy: 0.7340
 3904/60000 [>.............................] - ETA: 1:40 - loss: 0.8276 - categorical_accuracy: 0.7349
 3936/60000 [>.............................] - ETA: 1:40 - loss: 0.8225 - categorical_accuracy: 0.7365
 3968/60000 [>.............................] - ETA: 1:40 - loss: 0.8213 - categorical_accuracy: 0.7374
 4000/60000 [=>............................] - ETA: 1:40 - loss: 0.8170 - categorical_accuracy: 0.7390
 4032/60000 [=>............................] - ETA: 1:40 - loss: 0.8152 - categorical_accuracy: 0.7393
 4064/60000 [=>............................] - ETA: 1:40 - loss: 0.8128 - categorical_accuracy: 0.7402
 4096/60000 [=>............................] - ETA: 1:40 - loss: 0.8093 - categorical_accuracy: 0.7412
 4128/60000 [=>............................] - ETA: 1:40 - loss: 0.8064 - categorical_accuracy: 0.7427
 4160/60000 [=>............................] - ETA: 1:40 - loss: 0.8059 - categorical_accuracy: 0.7435
 4192/60000 [=>............................] - ETA: 1:39 - loss: 0.8033 - categorical_accuracy: 0.7450
 4224/60000 [=>............................] - ETA: 1:39 - loss: 0.8012 - categorical_accuracy: 0.7457
 4256/60000 [=>............................] - ETA: 1:39 - loss: 0.7990 - categorical_accuracy: 0.7462
 4288/60000 [=>............................] - ETA: 1:39 - loss: 0.7958 - categorical_accuracy: 0.7472
 4320/60000 [=>............................] - ETA: 1:39 - loss: 0.7953 - categorical_accuracy: 0.7477
 4352/60000 [=>............................] - ETA: 1:39 - loss: 0.7926 - categorical_accuracy: 0.7484
 4384/60000 [=>............................] - ETA: 1:39 - loss: 0.7896 - categorical_accuracy: 0.7493
 4416/60000 [=>............................] - ETA: 1:39 - loss: 0.7859 - categorical_accuracy: 0.7505
 4448/60000 [=>............................] - ETA: 1:39 - loss: 0.7829 - categorical_accuracy: 0.7511
 4480/60000 [=>............................] - ETA: 1:39 - loss: 0.7823 - categorical_accuracy: 0.7520
 4512/60000 [=>............................] - ETA: 1:39 - loss: 0.7778 - categorical_accuracy: 0.7535
 4544/60000 [=>............................] - ETA: 1:39 - loss: 0.7743 - categorical_accuracy: 0.7548
 4576/60000 [=>............................] - ETA: 1:39 - loss: 0.7709 - categorical_accuracy: 0.7559
 4608/60000 [=>............................] - ETA: 1:39 - loss: 0.7679 - categorical_accuracy: 0.7569
 4640/60000 [=>............................] - ETA: 1:38 - loss: 0.7654 - categorical_accuracy: 0.7578
 4672/60000 [=>............................] - ETA: 1:38 - loss: 0.7634 - categorical_accuracy: 0.7586
 4704/60000 [=>............................] - ETA: 1:38 - loss: 0.7613 - categorical_accuracy: 0.7594
 4736/60000 [=>............................] - ETA: 1:38 - loss: 0.7587 - categorical_accuracy: 0.7603
 4768/60000 [=>............................] - ETA: 1:38 - loss: 0.7614 - categorical_accuracy: 0.7603
 4800/60000 [=>............................] - ETA: 1:38 - loss: 0.7589 - categorical_accuracy: 0.7610
 4832/60000 [=>............................] - ETA: 1:38 - loss: 0.7559 - categorical_accuracy: 0.7622
 4864/60000 [=>............................] - ETA: 1:38 - loss: 0.7529 - categorical_accuracy: 0.7634
 4896/60000 [=>............................] - ETA: 1:38 - loss: 0.7513 - categorical_accuracy: 0.7637
 4928/60000 [=>............................] - ETA: 1:38 - loss: 0.7487 - categorical_accuracy: 0.7644
 4960/60000 [=>............................] - ETA: 1:38 - loss: 0.7460 - categorical_accuracy: 0.7651
 4992/60000 [=>............................] - ETA: 1:38 - loss: 0.7436 - categorical_accuracy: 0.7656
 5024/60000 [=>............................] - ETA: 1:37 - loss: 0.7407 - categorical_accuracy: 0.7665
 5056/60000 [=>............................] - ETA: 1:37 - loss: 0.7384 - categorical_accuracy: 0.7676
 5088/60000 [=>............................] - ETA: 1:37 - loss: 0.7360 - categorical_accuracy: 0.7681
 5120/60000 [=>............................] - ETA: 1:37 - loss: 0.7323 - categorical_accuracy: 0.7693
 5152/60000 [=>............................] - ETA: 1:37 - loss: 0.7283 - categorical_accuracy: 0.7708
 5184/60000 [=>............................] - ETA: 1:37 - loss: 0.7261 - categorical_accuracy: 0.7712
 5216/60000 [=>............................] - ETA: 1:37 - loss: 0.7230 - categorical_accuracy: 0.7720
 5248/60000 [=>............................] - ETA: 1:37 - loss: 0.7234 - categorical_accuracy: 0.7727
 5280/60000 [=>............................] - ETA: 1:37 - loss: 0.7205 - categorical_accuracy: 0.7735
 5312/60000 [=>............................] - ETA: 1:37 - loss: 0.7194 - categorical_accuracy: 0.7739
 5344/60000 [=>............................] - ETA: 1:37 - loss: 0.7164 - categorical_accuracy: 0.7751
 5376/60000 [=>............................] - ETA: 1:37 - loss: 0.7143 - categorical_accuracy: 0.7751
 5408/60000 [=>............................] - ETA: 1:36 - loss: 0.7123 - categorical_accuracy: 0.7757
 5440/60000 [=>............................] - ETA: 1:36 - loss: 0.7105 - categorical_accuracy: 0.7763
 5472/60000 [=>............................] - ETA: 1:36 - loss: 0.7067 - categorical_accuracy: 0.7776
 5504/60000 [=>............................] - ETA: 1:36 - loss: 0.7038 - categorical_accuracy: 0.7785
 5536/60000 [=>............................] - ETA: 1:36 - loss: 0.7014 - categorical_accuracy: 0.7794
 5568/60000 [=>............................] - ETA: 1:36 - loss: 0.6985 - categorical_accuracy: 0.7802
 5600/60000 [=>............................] - ETA: 1:36 - loss: 0.6977 - categorical_accuracy: 0.7804
 5632/60000 [=>............................] - ETA: 1:36 - loss: 0.6959 - categorical_accuracy: 0.7805
 5664/60000 [=>............................] - ETA: 1:36 - loss: 0.6931 - categorical_accuracy: 0.7814
 5696/60000 [=>............................] - ETA: 1:36 - loss: 0.6903 - categorical_accuracy: 0.7825
 5728/60000 [=>............................] - ETA: 1:36 - loss: 0.6872 - categorical_accuracy: 0.7835
 5760/60000 [=>............................] - ETA: 1:36 - loss: 0.6848 - categorical_accuracy: 0.7845
 5792/60000 [=>............................] - ETA: 1:36 - loss: 0.6825 - categorical_accuracy: 0.7852
 5824/60000 [=>............................] - ETA: 1:36 - loss: 0.6799 - categorical_accuracy: 0.7862
 5856/60000 [=>............................] - ETA: 1:36 - loss: 0.6772 - categorical_accuracy: 0.7872
 5888/60000 [=>............................] - ETA: 1:36 - loss: 0.6743 - categorical_accuracy: 0.7882
 5920/60000 [=>............................] - ETA: 1:36 - loss: 0.6731 - categorical_accuracy: 0.7890
 5952/60000 [=>............................] - ETA: 1:35 - loss: 0.6725 - categorical_accuracy: 0.7900
 5984/60000 [=>............................] - ETA: 1:35 - loss: 0.6707 - categorical_accuracy: 0.7904
 6016/60000 [==>...........................] - ETA: 1:35 - loss: 0.6686 - categorical_accuracy: 0.7914
 6048/60000 [==>...........................] - ETA: 1:35 - loss: 0.6676 - categorical_accuracy: 0.7918
 6080/60000 [==>...........................] - ETA: 1:35 - loss: 0.6676 - categorical_accuracy: 0.7921
 6112/60000 [==>...........................] - ETA: 1:35 - loss: 0.6659 - categorical_accuracy: 0.7929
 6144/60000 [==>...........................] - ETA: 1:35 - loss: 0.6640 - categorical_accuracy: 0.7935
 6176/60000 [==>...........................] - ETA: 1:35 - loss: 0.6618 - categorical_accuracy: 0.7942
 6208/60000 [==>...........................] - ETA: 1:35 - loss: 0.6596 - categorical_accuracy: 0.7949
 6240/60000 [==>...........................] - ETA: 1:35 - loss: 0.6583 - categorical_accuracy: 0.7954
 6272/60000 [==>...........................] - ETA: 1:35 - loss: 0.6566 - categorical_accuracy: 0.7959
 6304/60000 [==>...........................] - ETA: 1:35 - loss: 0.6544 - categorical_accuracy: 0.7968
 6336/60000 [==>...........................] - ETA: 1:35 - loss: 0.6517 - categorical_accuracy: 0.7978
 6368/60000 [==>...........................] - ETA: 1:35 - loss: 0.6508 - categorical_accuracy: 0.7982
 6400/60000 [==>...........................] - ETA: 1:35 - loss: 0.6506 - categorical_accuracy: 0.7984
 6432/60000 [==>...........................] - ETA: 1:35 - loss: 0.6486 - categorical_accuracy: 0.7990
 6464/60000 [==>...........................] - ETA: 1:35 - loss: 0.6460 - categorical_accuracy: 0.7998
 6496/60000 [==>...........................] - ETA: 1:34 - loss: 0.6444 - categorical_accuracy: 0.8003
 6528/60000 [==>...........................] - ETA: 1:34 - loss: 0.6439 - categorical_accuracy: 0.8007
 6560/60000 [==>...........................] - ETA: 1:34 - loss: 0.6419 - categorical_accuracy: 0.8014
 6592/60000 [==>...........................] - ETA: 1:34 - loss: 0.6392 - categorical_accuracy: 0.8023
 6624/60000 [==>...........................] - ETA: 1:34 - loss: 0.6377 - categorical_accuracy: 0.8025
 6656/60000 [==>...........................] - ETA: 1:34 - loss: 0.6356 - categorical_accuracy: 0.8033
 6688/60000 [==>...........................] - ETA: 1:34 - loss: 0.6345 - categorical_accuracy: 0.8037
 6720/60000 [==>...........................] - ETA: 1:34 - loss: 0.6340 - categorical_accuracy: 0.8039
 6752/60000 [==>...........................] - ETA: 1:34 - loss: 0.6333 - categorical_accuracy: 0.8041
 6784/60000 [==>...........................] - ETA: 1:34 - loss: 0.6318 - categorical_accuracy: 0.8044
 6816/60000 [==>...........................] - ETA: 1:34 - loss: 0.6297 - categorical_accuracy: 0.8053
 6848/60000 [==>...........................] - ETA: 1:34 - loss: 0.6271 - categorical_accuracy: 0.8061
 6880/60000 [==>...........................] - ETA: 1:34 - loss: 0.6255 - categorical_accuracy: 0.8067
 6912/60000 [==>...........................] - ETA: 1:33 - loss: 0.6232 - categorical_accuracy: 0.8074
 6944/60000 [==>...........................] - ETA: 1:33 - loss: 0.6228 - categorical_accuracy: 0.8076
 6976/60000 [==>...........................] - ETA: 1:33 - loss: 0.6212 - categorical_accuracy: 0.8081
 7008/60000 [==>...........................] - ETA: 1:33 - loss: 0.6192 - categorical_accuracy: 0.8086
 7040/60000 [==>...........................] - ETA: 1:33 - loss: 0.6183 - categorical_accuracy: 0.8089
 7072/60000 [==>...........................] - ETA: 1:33 - loss: 0.6160 - categorical_accuracy: 0.8097
 7104/60000 [==>...........................] - ETA: 1:33 - loss: 0.6150 - categorical_accuracy: 0.8100
 7136/60000 [==>...........................] - ETA: 1:33 - loss: 0.6140 - categorical_accuracy: 0.8101
 7168/60000 [==>...........................] - ETA: 1:33 - loss: 0.6118 - categorical_accuracy: 0.8110
 7200/60000 [==>...........................] - ETA: 1:33 - loss: 0.6106 - categorical_accuracy: 0.8112
 7232/60000 [==>...........................] - ETA: 1:33 - loss: 0.6087 - categorical_accuracy: 0.8119
 7264/60000 [==>...........................] - ETA: 1:33 - loss: 0.6066 - categorical_accuracy: 0.8125
 7296/60000 [==>...........................] - ETA: 1:33 - loss: 0.6062 - categorical_accuracy: 0.8129
 7328/60000 [==>...........................] - ETA: 1:33 - loss: 0.6045 - categorical_accuracy: 0.8133
 7360/60000 [==>...........................] - ETA: 1:33 - loss: 0.6023 - categorical_accuracy: 0.8141
 7392/60000 [==>...........................] - ETA: 1:33 - loss: 0.6003 - categorical_accuracy: 0.8149
 7424/60000 [==>...........................] - ETA: 1:32 - loss: 0.5988 - categorical_accuracy: 0.8155
 7456/60000 [==>...........................] - ETA: 1:32 - loss: 0.5984 - categorical_accuracy: 0.8157
 7488/60000 [==>...........................] - ETA: 1:32 - loss: 0.5973 - categorical_accuracy: 0.8164
 7520/60000 [==>...........................] - ETA: 1:32 - loss: 0.5951 - categorical_accuracy: 0.8170
 7552/60000 [==>...........................] - ETA: 1:32 - loss: 0.5930 - categorical_accuracy: 0.8175
 7584/60000 [==>...........................] - ETA: 1:32 - loss: 0.5917 - categorical_accuracy: 0.8179
 7616/60000 [==>...........................] - ETA: 1:32 - loss: 0.5905 - categorical_accuracy: 0.8184
 7648/60000 [==>...........................] - ETA: 1:32 - loss: 0.5888 - categorical_accuracy: 0.8189
 7680/60000 [==>...........................] - ETA: 1:32 - loss: 0.5874 - categorical_accuracy: 0.8193
 7712/60000 [==>...........................] - ETA: 1:32 - loss: 0.5855 - categorical_accuracy: 0.8198
 7744/60000 [==>...........................] - ETA: 1:32 - loss: 0.5835 - categorical_accuracy: 0.8205
 7776/60000 [==>...........................] - ETA: 1:32 - loss: 0.5830 - categorical_accuracy: 0.8209
 7808/60000 [==>...........................] - ETA: 1:32 - loss: 0.5823 - categorical_accuracy: 0.8212
 7840/60000 [==>...........................] - ETA: 1:32 - loss: 0.5803 - categorical_accuracy: 0.8219
 7872/60000 [==>...........................] - ETA: 1:32 - loss: 0.5785 - categorical_accuracy: 0.8224
 7904/60000 [==>...........................] - ETA: 1:32 - loss: 0.5782 - categorical_accuracy: 0.8225
 7936/60000 [==>...........................] - ETA: 1:32 - loss: 0.5776 - categorical_accuracy: 0.8225
 7968/60000 [==>...........................] - ETA: 1:32 - loss: 0.5756 - categorical_accuracy: 0.8232
 8000/60000 [===>..........................] - ETA: 1:31 - loss: 0.5741 - categorical_accuracy: 0.8238
 8032/60000 [===>..........................] - ETA: 1:31 - loss: 0.5726 - categorical_accuracy: 0.8242
 8064/60000 [===>..........................] - ETA: 1:31 - loss: 0.5705 - categorical_accuracy: 0.8249
 8096/60000 [===>..........................] - ETA: 1:31 - loss: 0.5692 - categorical_accuracy: 0.8255
 8128/60000 [===>..........................] - ETA: 1:31 - loss: 0.5695 - categorical_accuracy: 0.8254
 8160/60000 [===>..........................] - ETA: 1:31 - loss: 0.5677 - categorical_accuracy: 0.8260
 8192/60000 [===>..........................] - ETA: 1:31 - loss: 0.5667 - categorical_accuracy: 0.8262
 8224/60000 [===>..........................] - ETA: 1:31 - loss: 0.5659 - categorical_accuracy: 0.8265
 8256/60000 [===>..........................] - ETA: 1:31 - loss: 0.5658 - categorical_accuracy: 0.8268
 8288/60000 [===>..........................] - ETA: 1:31 - loss: 0.5649 - categorical_accuracy: 0.8270
 8320/60000 [===>..........................] - ETA: 1:31 - loss: 0.5634 - categorical_accuracy: 0.8274
 8352/60000 [===>..........................] - ETA: 1:31 - loss: 0.5620 - categorical_accuracy: 0.8279
 8384/60000 [===>..........................] - ETA: 1:31 - loss: 0.5602 - categorical_accuracy: 0.8286
 8416/60000 [===>..........................] - ETA: 1:31 - loss: 0.5599 - categorical_accuracy: 0.8289
 8448/60000 [===>..........................] - ETA: 1:31 - loss: 0.5586 - categorical_accuracy: 0.8294
 8480/60000 [===>..........................] - ETA: 1:31 - loss: 0.5576 - categorical_accuracy: 0.8298
 8512/60000 [===>..........................] - ETA: 1:31 - loss: 0.5560 - categorical_accuracy: 0.8304
 8544/60000 [===>..........................] - ETA: 1:30 - loss: 0.5542 - categorical_accuracy: 0.8309
 8576/60000 [===>..........................] - ETA: 1:30 - loss: 0.5531 - categorical_accuracy: 0.8313
 8608/60000 [===>..........................] - ETA: 1:30 - loss: 0.5527 - categorical_accuracy: 0.8314
 8640/60000 [===>..........................] - ETA: 1:30 - loss: 0.5513 - categorical_accuracy: 0.8317
 8672/60000 [===>..........................] - ETA: 1:30 - loss: 0.5500 - categorical_accuracy: 0.8321
 8704/60000 [===>..........................] - ETA: 1:30 - loss: 0.5483 - categorical_accuracy: 0.8327
 8736/60000 [===>..........................] - ETA: 1:30 - loss: 0.5471 - categorical_accuracy: 0.8330
 8768/60000 [===>..........................] - ETA: 1:30 - loss: 0.5461 - categorical_accuracy: 0.8333
 8800/60000 [===>..........................] - ETA: 1:30 - loss: 0.5455 - categorical_accuracy: 0.8332
 8832/60000 [===>..........................] - ETA: 1:30 - loss: 0.5447 - categorical_accuracy: 0.8334
 8864/60000 [===>..........................] - ETA: 1:30 - loss: 0.5444 - categorical_accuracy: 0.8336
 8896/60000 [===>..........................] - ETA: 1:30 - loss: 0.5427 - categorical_accuracy: 0.8342
 8928/60000 [===>..........................] - ETA: 1:30 - loss: 0.5417 - categorical_accuracy: 0.8343
 8960/60000 [===>..........................] - ETA: 1:30 - loss: 0.5400 - categorical_accuracy: 0.8349
 8992/60000 [===>..........................] - ETA: 1:29 - loss: 0.5389 - categorical_accuracy: 0.8353
 9024/60000 [===>..........................] - ETA: 1:29 - loss: 0.5374 - categorical_accuracy: 0.8358
 9056/60000 [===>..........................] - ETA: 1:29 - loss: 0.5363 - categorical_accuracy: 0.8359
 9088/60000 [===>..........................] - ETA: 1:29 - loss: 0.5350 - categorical_accuracy: 0.8363
 9120/60000 [===>..........................] - ETA: 1:29 - loss: 0.5341 - categorical_accuracy: 0.8365
 9152/60000 [===>..........................] - ETA: 1:29 - loss: 0.5332 - categorical_accuracy: 0.8370
 9184/60000 [===>..........................] - ETA: 1:29 - loss: 0.5323 - categorical_accuracy: 0.8373
 9216/60000 [===>..........................] - ETA: 1:29 - loss: 0.5313 - categorical_accuracy: 0.8377
 9248/60000 [===>..........................] - ETA: 1:29 - loss: 0.5302 - categorical_accuracy: 0.8381
 9280/60000 [===>..........................] - ETA: 1:29 - loss: 0.5291 - categorical_accuracy: 0.8386
 9312/60000 [===>..........................] - ETA: 1:29 - loss: 0.5287 - categorical_accuracy: 0.8386
 9344/60000 [===>..........................] - ETA: 1:29 - loss: 0.5272 - categorical_accuracy: 0.8389
 9376/60000 [===>..........................] - ETA: 1:29 - loss: 0.5261 - categorical_accuracy: 0.8392
 9408/60000 [===>..........................] - ETA: 1:29 - loss: 0.5248 - categorical_accuracy: 0.8395
 9440/60000 [===>..........................] - ETA: 1:29 - loss: 0.5244 - categorical_accuracy: 0.8394
 9472/60000 [===>..........................] - ETA: 1:29 - loss: 0.5231 - categorical_accuracy: 0.8398
 9504/60000 [===>..........................] - ETA: 1:29 - loss: 0.5216 - categorical_accuracy: 0.8403
 9536/60000 [===>..........................] - ETA: 1:28 - loss: 0.5206 - categorical_accuracy: 0.8406
 9568/60000 [===>..........................] - ETA: 1:28 - loss: 0.5191 - categorical_accuracy: 0.8411
 9600/60000 [===>..........................] - ETA: 1:28 - loss: 0.5183 - categorical_accuracy: 0.8415
 9632/60000 [===>..........................] - ETA: 1:28 - loss: 0.5170 - categorical_accuracy: 0.8419
 9664/60000 [===>..........................] - ETA: 1:28 - loss: 0.5160 - categorical_accuracy: 0.8422
 9696/60000 [===>..........................] - ETA: 1:28 - loss: 0.5164 - categorical_accuracy: 0.8421
 9728/60000 [===>..........................] - ETA: 1:28 - loss: 0.5154 - categorical_accuracy: 0.8424
 9760/60000 [===>..........................] - ETA: 1:28 - loss: 0.5151 - categorical_accuracy: 0.8424
 9792/60000 [===>..........................] - ETA: 1:28 - loss: 0.5142 - categorical_accuracy: 0.8427
 9824/60000 [===>..........................] - ETA: 1:28 - loss: 0.5126 - categorical_accuracy: 0.8432
 9856/60000 [===>..........................] - ETA: 1:28 - loss: 0.5117 - categorical_accuracy: 0.8435
 9888/60000 [===>..........................] - ETA: 1:28 - loss: 0.5111 - categorical_accuracy: 0.8438
 9920/60000 [===>..........................] - ETA: 1:28 - loss: 0.5105 - categorical_accuracy: 0.8440
 9952/60000 [===>..........................] - ETA: 1:28 - loss: 0.5097 - categorical_accuracy: 0.8442
 9984/60000 [===>..........................] - ETA: 1:28 - loss: 0.5085 - categorical_accuracy: 0.8446
10016/60000 [====>.........................] - ETA: 1:28 - loss: 0.5079 - categorical_accuracy: 0.8447
10048/60000 [====>.........................] - ETA: 1:27 - loss: 0.5071 - categorical_accuracy: 0.8450
10080/60000 [====>.........................] - ETA: 1:27 - loss: 0.5061 - categorical_accuracy: 0.8454
10112/60000 [====>.........................] - ETA: 1:27 - loss: 0.5052 - categorical_accuracy: 0.8457
10144/60000 [====>.........................] - ETA: 1:27 - loss: 0.5059 - categorical_accuracy: 0.8455
10176/60000 [====>.........................] - ETA: 1:27 - loss: 0.5050 - categorical_accuracy: 0.8459
10208/60000 [====>.........................] - ETA: 1:27 - loss: 0.5040 - categorical_accuracy: 0.8464
10240/60000 [====>.........................] - ETA: 1:27 - loss: 0.5038 - categorical_accuracy: 0.8465
10272/60000 [====>.........................] - ETA: 1:27 - loss: 0.5036 - categorical_accuracy: 0.8467
10304/60000 [====>.........................] - ETA: 1:27 - loss: 0.5027 - categorical_accuracy: 0.8469
10336/60000 [====>.........................] - ETA: 1:27 - loss: 0.5014 - categorical_accuracy: 0.8473
10368/60000 [====>.........................] - ETA: 1:27 - loss: 0.5007 - categorical_accuracy: 0.8475
10400/60000 [====>.........................] - ETA: 1:27 - loss: 0.4999 - categorical_accuracy: 0.8477
10432/60000 [====>.........................] - ETA: 1:27 - loss: 0.4991 - categorical_accuracy: 0.8480
10464/60000 [====>.........................] - ETA: 1:27 - loss: 0.4990 - categorical_accuracy: 0.8481
10496/60000 [====>.........................] - ETA: 1:27 - loss: 0.4981 - categorical_accuracy: 0.8483
10528/60000 [====>.........................] - ETA: 1:27 - loss: 0.4970 - categorical_accuracy: 0.8487
10560/60000 [====>.........................] - ETA: 1:27 - loss: 0.4965 - categorical_accuracy: 0.8488
10592/60000 [====>.........................] - ETA: 1:26 - loss: 0.4956 - categorical_accuracy: 0.8490
10624/60000 [====>.........................] - ETA: 1:26 - loss: 0.4948 - categorical_accuracy: 0.8493
10656/60000 [====>.........................] - ETA: 1:26 - loss: 0.4935 - categorical_accuracy: 0.8498
10688/60000 [====>.........................] - ETA: 1:26 - loss: 0.4922 - categorical_accuracy: 0.8502
10720/60000 [====>.........................] - ETA: 1:26 - loss: 0.4914 - categorical_accuracy: 0.8505
10752/60000 [====>.........................] - ETA: 1:26 - loss: 0.4901 - categorical_accuracy: 0.8508
10784/60000 [====>.........................] - ETA: 1:26 - loss: 0.4891 - categorical_accuracy: 0.8511
10816/60000 [====>.........................] - ETA: 1:26 - loss: 0.4884 - categorical_accuracy: 0.8513
10848/60000 [====>.........................] - ETA: 1:26 - loss: 0.4874 - categorical_accuracy: 0.8516
10880/60000 [====>.........................] - ETA: 1:26 - loss: 0.4863 - categorical_accuracy: 0.8520
10912/60000 [====>.........................] - ETA: 1:26 - loss: 0.4858 - categorical_accuracy: 0.8523
10944/60000 [====>.........................] - ETA: 1:26 - loss: 0.4846 - categorical_accuracy: 0.8526
10976/60000 [====>.........................] - ETA: 1:26 - loss: 0.4837 - categorical_accuracy: 0.8529
11008/60000 [====>.........................] - ETA: 1:26 - loss: 0.4829 - categorical_accuracy: 0.8531
11040/60000 [====>.........................] - ETA: 1:26 - loss: 0.4817 - categorical_accuracy: 0.8534
11072/60000 [====>.........................] - ETA: 1:26 - loss: 0.4811 - categorical_accuracy: 0.8537
11104/60000 [====>.........................] - ETA: 1:26 - loss: 0.4798 - categorical_accuracy: 0.8541
11136/60000 [====>.........................] - ETA: 1:26 - loss: 0.4804 - categorical_accuracy: 0.8541
11168/60000 [====>.........................] - ETA: 1:25 - loss: 0.4803 - categorical_accuracy: 0.8543
11200/60000 [====>.........................] - ETA: 1:25 - loss: 0.4807 - categorical_accuracy: 0.8545
11232/60000 [====>.........................] - ETA: 1:25 - loss: 0.4799 - categorical_accuracy: 0.8546
11264/60000 [====>.........................] - ETA: 1:25 - loss: 0.4794 - categorical_accuracy: 0.8549
11296/60000 [====>.........................] - ETA: 1:25 - loss: 0.4784 - categorical_accuracy: 0.8552
11328/60000 [====>.........................] - ETA: 1:25 - loss: 0.4777 - categorical_accuracy: 0.8553
11360/60000 [====>.........................] - ETA: 1:25 - loss: 0.4774 - categorical_accuracy: 0.8555
11392/60000 [====>.........................] - ETA: 1:25 - loss: 0.4764 - categorical_accuracy: 0.8558
11424/60000 [====>.........................] - ETA: 1:25 - loss: 0.4760 - categorical_accuracy: 0.8560
11456/60000 [====>.........................] - ETA: 1:25 - loss: 0.4750 - categorical_accuracy: 0.8564
11488/60000 [====>.........................] - ETA: 1:25 - loss: 0.4743 - categorical_accuracy: 0.8566
11520/60000 [====>.........................] - ETA: 1:25 - loss: 0.4735 - categorical_accuracy: 0.8569
11552/60000 [====>.........................] - ETA: 1:25 - loss: 0.4732 - categorical_accuracy: 0.8570
11584/60000 [====>.........................] - ETA: 1:25 - loss: 0.4724 - categorical_accuracy: 0.8570
11616/60000 [====>.........................] - ETA: 1:25 - loss: 0.4722 - categorical_accuracy: 0.8572
11648/60000 [====>.........................] - ETA: 1:25 - loss: 0.4713 - categorical_accuracy: 0.8574
11680/60000 [====>.........................] - ETA: 1:24 - loss: 0.4708 - categorical_accuracy: 0.8574
11712/60000 [====>.........................] - ETA: 1:24 - loss: 0.4700 - categorical_accuracy: 0.8577
11744/60000 [====>.........................] - ETA: 1:24 - loss: 0.4695 - categorical_accuracy: 0.8578
11776/60000 [====>.........................] - ETA: 1:24 - loss: 0.4686 - categorical_accuracy: 0.8581
11808/60000 [====>.........................] - ETA: 1:24 - loss: 0.4679 - categorical_accuracy: 0.8583
11840/60000 [====>.........................] - ETA: 1:24 - loss: 0.4669 - categorical_accuracy: 0.8586
11872/60000 [====>.........................] - ETA: 1:24 - loss: 0.4667 - categorical_accuracy: 0.8587
11904/60000 [====>.........................] - ETA: 1:24 - loss: 0.4660 - categorical_accuracy: 0.8590
11936/60000 [====>.........................] - ETA: 1:24 - loss: 0.4652 - categorical_accuracy: 0.8592
11968/60000 [====>.........................] - ETA: 1:24 - loss: 0.4644 - categorical_accuracy: 0.8594
12000/60000 [=====>........................] - ETA: 1:24 - loss: 0.4638 - categorical_accuracy: 0.8595
12032/60000 [=====>........................] - ETA: 1:24 - loss: 0.4628 - categorical_accuracy: 0.8598
12064/60000 [=====>........................] - ETA: 1:24 - loss: 0.4621 - categorical_accuracy: 0.8600
12096/60000 [=====>........................] - ETA: 1:24 - loss: 0.4620 - categorical_accuracy: 0.8601
12128/60000 [=====>........................] - ETA: 1:24 - loss: 0.4611 - categorical_accuracy: 0.8604
12160/60000 [=====>........................] - ETA: 1:24 - loss: 0.4608 - categorical_accuracy: 0.8606
12192/60000 [=====>........................] - ETA: 1:24 - loss: 0.4599 - categorical_accuracy: 0.8609
12224/60000 [=====>........................] - ETA: 1:23 - loss: 0.4590 - categorical_accuracy: 0.8611
12256/60000 [=====>........................] - ETA: 1:23 - loss: 0.4584 - categorical_accuracy: 0.8613
12288/60000 [=====>........................] - ETA: 1:23 - loss: 0.4575 - categorical_accuracy: 0.8616
12320/60000 [=====>........................] - ETA: 1:23 - loss: 0.4566 - categorical_accuracy: 0.8619
12352/60000 [=====>........................] - ETA: 1:23 - loss: 0.4561 - categorical_accuracy: 0.8619
12384/60000 [=====>........................] - ETA: 1:23 - loss: 0.4551 - categorical_accuracy: 0.8622
12416/60000 [=====>........................] - ETA: 1:23 - loss: 0.4547 - categorical_accuracy: 0.8623
12448/60000 [=====>........................] - ETA: 1:23 - loss: 0.4538 - categorical_accuracy: 0.8625
12480/60000 [=====>........................] - ETA: 1:23 - loss: 0.4531 - categorical_accuracy: 0.8627
12512/60000 [=====>........................] - ETA: 1:23 - loss: 0.4521 - categorical_accuracy: 0.8631
12544/60000 [=====>........................] - ETA: 1:23 - loss: 0.4512 - categorical_accuracy: 0.8634
12576/60000 [=====>........................] - ETA: 1:23 - loss: 0.4504 - categorical_accuracy: 0.8636
12608/60000 [=====>........................] - ETA: 1:23 - loss: 0.4498 - categorical_accuracy: 0.8638
12640/60000 [=====>........................] - ETA: 1:23 - loss: 0.4491 - categorical_accuracy: 0.8641
12672/60000 [=====>........................] - ETA: 1:23 - loss: 0.4481 - categorical_accuracy: 0.8644
12704/60000 [=====>........................] - ETA: 1:23 - loss: 0.4474 - categorical_accuracy: 0.8647
12736/60000 [=====>........................] - ETA: 1:23 - loss: 0.4476 - categorical_accuracy: 0.8648
12768/60000 [=====>........................] - ETA: 1:22 - loss: 0.4477 - categorical_accuracy: 0.8649
12800/60000 [=====>........................] - ETA: 1:22 - loss: 0.4469 - categorical_accuracy: 0.8651
12832/60000 [=====>........................] - ETA: 1:22 - loss: 0.4461 - categorical_accuracy: 0.8653
12864/60000 [=====>........................] - ETA: 1:22 - loss: 0.4453 - categorical_accuracy: 0.8654
12896/60000 [=====>........................] - ETA: 1:22 - loss: 0.4446 - categorical_accuracy: 0.8657
12928/60000 [=====>........................] - ETA: 1:22 - loss: 0.4438 - categorical_accuracy: 0.8659
12960/60000 [=====>........................] - ETA: 1:22 - loss: 0.4436 - categorical_accuracy: 0.8660
12992/60000 [=====>........................] - ETA: 1:22 - loss: 0.4433 - categorical_accuracy: 0.8660
13024/60000 [=====>........................] - ETA: 1:22 - loss: 0.4425 - categorical_accuracy: 0.8662
13056/60000 [=====>........................] - ETA: 1:22 - loss: 0.4417 - categorical_accuracy: 0.8665
13088/60000 [=====>........................] - ETA: 1:22 - loss: 0.4407 - categorical_accuracy: 0.8668
13120/60000 [=====>........................] - ETA: 1:22 - loss: 0.4398 - categorical_accuracy: 0.8671
13152/60000 [=====>........................] - ETA: 1:22 - loss: 0.4391 - categorical_accuracy: 0.8672
13184/60000 [=====>........................] - ETA: 1:22 - loss: 0.4386 - categorical_accuracy: 0.8674
13216/60000 [=====>........................] - ETA: 1:22 - loss: 0.4382 - categorical_accuracy: 0.8676
13248/60000 [=====>........................] - ETA: 1:22 - loss: 0.4375 - categorical_accuracy: 0.8678
13280/60000 [=====>........................] - ETA: 1:22 - loss: 0.4370 - categorical_accuracy: 0.8679
13312/60000 [=====>........................] - ETA: 1:21 - loss: 0.4364 - categorical_accuracy: 0.8682
13344/60000 [=====>........................] - ETA: 1:21 - loss: 0.4356 - categorical_accuracy: 0.8684
13376/60000 [=====>........................] - ETA: 1:21 - loss: 0.4353 - categorical_accuracy: 0.8685
13408/60000 [=====>........................] - ETA: 1:21 - loss: 0.4346 - categorical_accuracy: 0.8687
13440/60000 [=====>........................] - ETA: 1:21 - loss: 0.4342 - categorical_accuracy: 0.8687
13472/60000 [=====>........................] - ETA: 1:21 - loss: 0.4335 - categorical_accuracy: 0.8688
13504/60000 [=====>........................] - ETA: 1:21 - loss: 0.4333 - categorical_accuracy: 0.8689
13536/60000 [=====>........................] - ETA: 1:21 - loss: 0.4328 - categorical_accuracy: 0.8690
13568/60000 [=====>........................] - ETA: 1:21 - loss: 0.4321 - categorical_accuracy: 0.8692
13600/60000 [=====>........................] - ETA: 1:21 - loss: 0.4317 - categorical_accuracy: 0.8694
13632/60000 [=====>........................] - ETA: 1:21 - loss: 0.4309 - categorical_accuracy: 0.8696
13664/60000 [=====>........................] - ETA: 1:21 - loss: 0.4302 - categorical_accuracy: 0.8699
13696/60000 [=====>........................] - ETA: 1:21 - loss: 0.4301 - categorical_accuracy: 0.8700
13728/60000 [=====>........................] - ETA: 1:21 - loss: 0.4293 - categorical_accuracy: 0.8701
13760/60000 [=====>........................] - ETA: 1:21 - loss: 0.4287 - categorical_accuracy: 0.8703
13792/60000 [=====>........................] - ETA: 1:21 - loss: 0.4280 - categorical_accuracy: 0.8706
13824/60000 [=====>........................] - ETA: 1:21 - loss: 0.4271 - categorical_accuracy: 0.8709
13856/60000 [=====>........................] - ETA: 1:20 - loss: 0.4264 - categorical_accuracy: 0.8711
13888/60000 [=====>........................] - ETA: 1:20 - loss: 0.4259 - categorical_accuracy: 0.8712
13920/60000 [=====>........................] - ETA: 1:20 - loss: 0.4250 - categorical_accuracy: 0.8715
13952/60000 [=====>........................] - ETA: 1:20 - loss: 0.4241 - categorical_accuracy: 0.8718
13984/60000 [=====>........................] - ETA: 1:20 - loss: 0.4236 - categorical_accuracy: 0.8719
14016/60000 [======>.......................] - ETA: 1:20 - loss: 0.4237 - categorical_accuracy: 0.8719
14048/60000 [======>.......................] - ETA: 1:20 - loss: 0.4236 - categorical_accuracy: 0.8720
14080/60000 [======>.......................] - ETA: 1:20 - loss: 0.4227 - categorical_accuracy: 0.8723
14112/60000 [======>.......................] - ETA: 1:20 - loss: 0.4219 - categorical_accuracy: 0.8726
14144/60000 [======>.......................] - ETA: 1:20 - loss: 0.4217 - categorical_accuracy: 0.8727
14176/60000 [======>.......................] - ETA: 1:20 - loss: 0.4211 - categorical_accuracy: 0.8729
14208/60000 [======>.......................] - ETA: 1:20 - loss: 0.4202 - categorical_accuracy: 0.8732
14240/60000 [======>.......................] - ETA: 1:20 - loss: 0.4199 - categorical_accuracy: 0.8732
14272/60000 [======>.......................] - ETA: 1:20 - loss: 0.4196 - categorical_accuracy: 0.8732
14304/60000 [======>.......................] - ETA: 1:20 - loss: 0.4190 - categorical_accuracy: 0.8733
14336/60000 [======>.......................] - ETA: 1:20 - loss: 0.4192 - categorical_accuracy: 0.8732
14368/60000 [======>.......................] - ETA: 1:20 - loss: 0.4187 - categorical_accuracy: 0.8733
14400/60000 [======>.......................] - ETA: 1:20 - loss: 0.4180 - categorical_accuracy: 0.8735
14432/60000 [======>.......................] - ETA: 1:19 - loss: 0.4175 - categorical_accuracy: 0.8735
14464/60000 [======>.......................] - ETA: 1:19 - loss: 0.4171 - categorical_accuracy: 0.8735
14496/60000 [======>.......................] - ETA: 1:19 - loss: 0.4164 - categorical_accuracy: 0.8737
14528/60000 [======>.......................] - ETA: 1:19 - loss: 0.4157 - categorical_accuracy: 0.8740
14560/60000 [======>.......................] - ETA: 1:19 - loss: 0.4149 - categorical_accuracy: 0.8742
14592/60000 [======>.......................] - ETA: 1:19 - loss: 0.4142 - categorical_accuracy: 0.8744
14624/60000 [======>.......................] - ETA: 1:19 - loss: 0.4137 - categorical_accuracy: 0.8745
14656/60000 [======>.......................] - ETA: 1:19 - loss: 0.4131 - categorical_accuracy: 0.8747
14688/60000 [======>.......................] - ETA: 1:19 - loss: 0.4124 - categorical_accuracy: 0.8749
14720/60000 [======>.......................] - ETA: 1:19 - loss: 0.4117 - categorical_accuracy: 0.8751
14752/60000 [======>.......................] - ETA: 1:19 - loss: 0.4118 - categorical_accuracy: 0.8751
14784/60000 [======>.......................] - ETA: 1:19 - loss: 0.4111 - categorical_accuracy: 0.8753
14816/60000 [======>.......................] - ETA: 1:19 - loss: 0.4104 - categorical_accuracy: 0.8755
14848/60000 [======>.......................] - ETA: 1:19 - loss: 0.4102 - categorical_accuracy: 0.8755
14880/60000 [======>.......................] - ETA: 1:19 - loss: 0.4095 - categorical_accuracy: 0.8757
14912/60000 [======>.......................] - ETA: 1:19 - loss: 0.4091 - categorical_accuracy: 0.8758
14944/60000 [======>.......................] - ETA: 1:19 - loss: 0.4083 - categorical_accuracy: 0.8761
14976/60000 [======>.......................] - ETA: 1:19 - loss: 0.4078 - categorical_accuracy: 0.8762
15008/60000 [======>.......................] - ETA: 1:18 - loss: 0.4071 - categorical_accuracy: 0.8764
15040/60000 [======>.......................] - ETA: 1:18 - loss: 0.4064 - categorical_accuracy: 0.8766
15072/60000 [======>.......................] - ETA: 1:18 - loss: 0.4062 - categorical_accuracy: 0.8767
15104/60000 [======>.......................] - ETA: 1:18 - loss: 0.4055 - categorical_accuracy: 0.8769
15136/60000 [======>.......................] - ETA: 1:18 - loss: 0.4057 - categorical_accuracy: 0.8768
15168/60000 [======>.......................] - ETA: 1:18 - loss: 0.4058 - categorical_accuracy: 0.8768
15200/60000 [======>.......................] - ETA: 1:18 - loss: 0.4057 - categorical_accuracy: 0.8770
15232/60000 [======>.......................] - ETA: 1:18 - loss: 0.4051 - categorical_accuracy: 0.8772
15264/60000 [======>.......................] - ETA: 1:18 - loss: 0.4049 - categorical_accuracy: 0.8772
15296/60000 [======>.......................] - ETA: 1:18 - loss: 0.4045 - categorical_accuracy: 0.8774
15328/60000 [======>.......................] - ETA: 1:18 - loss: 0.4039 - categorical_accuracy: 0.8775
15360/60000 [======>.......................] - ETA: 1:18 - loss: 0.4032 - categorical_accuracy: 0.8777
15392/60000 [======>.......................] - ETA: 1:18 - loss: 0.4025 - categorical_accuracy: 0.8780
15424/60000 [======>.......................] - ETA: 1:18 - loss: 0.4022 - categorical_accuracy: 0.8780
15456/60000 [======>.......................] - ETA: 1:18 - loss: 0.4015 - categorical_accuracy: 0.8782
15488/60000 [======>.......................] - ETA: 1:18 - loss: 0.4008 - categorical_accuracy: 0.8784
15520/60000 [======>.......................] - ETA: 1:17 - loss: 0.4000 - categorical_accuracy: 0.8787
15552/60000 [======>.......................] - ETA: 1:17 - loss: 0.3997 - categorical_accuracy: 0.8787
15584/60000 [======>.......................] - ETA: 1:17 - loss: 0.3992 - categorical_accuracy: 0.8788
15616/60000 [======>.......................] - ETA: 1:17 - loss: 0.3985 - categorical_accuracy: 0.8790
15648/60000 [======>.......................] - ETA: 1:17 - loss: 0.3978 - categorical_accuracy: 0.8792
15680/60000 [======>.......................] - ETA: 1:17 - loss: 0.3971 - categorical_accuracy: 0.8794
15712/60000 [======>.......................] - ETA: 1:17 - loss: 0.3967 - categorical_accuracy: 0.8795
15744/60000 [======>.......................] - ETA: 1:17 - loss: 0.3963 - categorical_accuracy: 0.8797
15776/60000 [======>.......................] - ETA: 1:17 - loss: 0.3957 - categorical_accuracy: 0.8798
15808/60000 [======>.......................] - ETA: 1:17 - loss: 0.3955 - categorical_accuracy: 0.8799
15840/60000 [======>.......................] - ETA: 1:17 - loss: 0.3952 - categorical_accuracy: 0.8801
15872/60000 [======>.......................] - ETA: 1:17 - loss: 0.3947 - categorical_accuracy: 0.8802
15904/60000 [======>.......................] - ETA: 1:17 - loss: 0.3946 - categorical_accuracy: 0.8802
15936/60000 [======>.......................] - ETA: 1:17 - loss: 0.3946 - categorical_accuracy: 0.8802
15968/60000 [======>.......................] - ETA: 1:17 - loss: 0.3939 - categorical_accuracy: 0.8804
16000/60000 [=======>......................] - ETA: 1:17 - loss: 0.3939 - categorical_accuracy: 0.8804
16032/60000 [=======>......................] - ETA: 1:17 - loss: 0.3932 - categorical_accuracy: 0.8806
16064/60000 [=======>......................] - ETA: 1:17 - loss: 0.3933 - categorical_accuracy: 0.8807
16096/60000 [=======>......................] - ETA: 1:16 - loss: 0.3926 - categorical_accuracy: 0.8809
16128/60000 [=======>......................] - ETA: 1:16 - loss: 0.3920 - categorical_accuracy: 0.8811
16160/60000 [=======>......................] - ETA: 1:16 - loss: 0.3918 - categorical_accuracy: 0.8811
16192/60000 [=======>......................] - ETA: 1:16 - loss: 0.3914 - categorical_accuracy: 0.8812
16224/60000 [=======>......................] - ETA: 1:16 - loss: 0.3908 - categorical_accuracy: 0.8814
16256/60000 [=======>......................] - ETA: 1:16 - loss: 0.3903 - categorical_accuracy: 0.8816
16288/60000 [=======>......................] - ETA: 1:16 - loss: 0.3899 - categorical_accuracy: 0.8817
16320/60000 [=======>......................] - ETA: 1:16 - loss: 0.3901 - categorical_accuracy: 0.8817
16352/60000 [=======>......................] - ETA: 1:16 - loss: 0.3900 - categorical_accuracy: 0.8817
16384/60000 [=======>......................] - ETA: 1:16 - loss: 0.3895 - categorical_accuracy: 0.8817
16416/60000 [=======>......................] - ETA: 1:16 - loss: 0.3892 - categorical_accuracy: 0.8818
16448/60000 [=======>......................] - ETA: 1:16 - loss: 0.3888 - categorical_accuracy: 0.8819
16480/60000 [=======>......................] - ETA: 1:16 - loss: 0.3883 - categorical_accuracy: 0.8820
16512/60000 [=======>......................] - ETA: 1:16 - loss: 0.3879 - categorical_accuracy: 0.8821
16544/60000 [=======>......................] - ETA: 1:16 - loss: 0.3879 - categorical_accuracy: 0.8823
16576/60000 [=======>......................] - ETA: 1:16 - loss: 0.3873 - categorical_accuracy: 0.8824
16608/60000 [=======>......................] - ETA: 1:16 - loss: 0.3868 - categorical_accuracy: 0.8825
16640/60000 [=======>......................] - ETA: 1:15 - loss: 0.3862 - categorical_accuracy: 0.8827
16672/60000 [=======>......................] - ETA: 1:15 - loss: 0.3858 - categorical_accuracy: 0.8829
16704/60000 [=======>......................] - ETA: 1:15 - loss: 0.3853 - categorical_accuracy: 0.8830
16736/60000 [=======>......................] - ETA: 1:15 - loss: 0.3848 - categorical_accuracy: 0.8831
16768/60000 [=======>......................] - ETA: 1:15 - loss: 0.3845 - categorical_accuracy: 0.8832
16800/60000 [=======>......................] - ETA: 1:15 - loss: 0.3842 - categorical_accuracy: 0.8833
16832/60000 [=======>......................] - ETA: 1:15 - loss: 0.3839 - categorical_accuracy: 0.8834
16864/60000 [=======>......................] - ETA: 1:15 - loss: 0.3833 - categorical_accuracy: 0.8837
16896/60000 [=======>......................] - ETA: 1:15 - loss: 0.3828 - categorical_accuracy: 0.8838
16928/60000 [=======>......................] - ETA: 1:15 - loss: 0.3824 - categorical_accuracy: 0.8840
16960/60000 [=======>......................] - ETA: 1:15 - loss: 0.3822 - categorical_accuracy: 0.8841
16992/60000 [=======>......................] - ETA: 1:15 - loss: 0.3817 - categorical_accuracy: 0.8842
17024/60000 [=======>......................] - ETA: 1:15 - loss: 0.3812 - categorical_accuracy: 0.8844
17056/60000 [=======>......................] - ETA: 1:15 - loss: 0.3808 - categorical_accuracy: 0.8845
17088/60000 [=======>......................] - ETA: 1:15 - loss: 0.3801 - categorical_accuracy: 0.8847
17120/60000 [=======>......................] - ETA: 1:15 - loss: 0.3799 - categorical_accuracy: 0.8848
17152/60000 [=======>......................] - ETA: 1:15 - loss: 0.3795 - categorical_accuracy: 0.8849
17184/60000 [=======>......................] - ETA: 1:14 - loss: 0.3791 - categorical_accuracy: 0.8850
17216/60000 [=======>......................] - ETA: 1:14 - loss: 0.3785 - categorical_accuracy: 0.8852
17248/60000 [=======>......................] - ETA: 1:14 - loss: 0.3785 - categorical_accuracy: 0.8853
17280/60000 [=======>......................] - ETA: 1:14 - loss: 0.3783 - categorical_accuracy: 0.8854
17312/60000 [=======>......................] - ETA: 1:14 - loss: 0.3778 - categorical_accuracy: 0.8855
17344/60000 [=======>......................] - ETA: 1:14 - loss: 0.3775 - categorical_accuracy: 0.8856
17376/60000 [=======>......................] - ETA: 1:14 - loss: 0.3770 - categorical_accuracy: 0.8857
17408/60000 [=======>......................] - ETA: 1:14 - loss: 0.3766 - categorical_accuracy: 0.8859
17440/60000 [=======>......................] - ETA: 1:14 - loss: 0.3763 - categorical_accuracy: 0.8860
17472/60000 [=======>......................] - ETA: 1:14 - loss: 0.3764 - categorical_accuracy: 0.8860
17504/60000 [=======>......................] - ETA: 1:14 - loss: 0.3757 - categorical_accuracy: 0.8862
17536/60000 [=======>......................] - ETA: 1:14 - loss: 0.3753 - categorical_accuracy: 0.8863
17568/60000 [=======>......................] - ETA: 1:14 - loss: 0.3748 - categorical_accuracy: 0.8864
17600/60000 [=======>......................] - ETA: 1:14 - loss: 0.3744 - categorical_accuracy: 0.8865
17632/60000 [=======>......................] - ETA: 1:14 - loss: 0.3739 - categorical_accuracy: 0.8867
17664/60000 [=======>......................] - ETA: 1:14 - loss: 0.3735 - categorical_accuracy: 0.8868
17696/60000 [=======>......................] - ETA: 1:14 - loss: 0.3730 - categorical_accuracy: 0.8869
17728/60000 [=======>......................] - ETA: 1:13 - loss: 0.3724 - categorical_accuracy: 0.8871
17760/60000 [=======>......................] - ETA: 1:13 - loss: 0.3723 - categorical_accuracy: 0.8870
17792/60000 [=======>......................] - ETA: 1:13 - loss: 0.3718 - categorical_accuracy: 0.8871
17824/60000 [=======>......................] - ETA: 1:13 - loss: 0.3715 - categorical_accuracy: 0.8872
17856/60000 [=======>......................] - ETA: 1:13 - loss: 0.3711 - categorical_accuracy: 0.8873
17888/60000 [=======>......................] - ETA: 1:13 - loss: 0.3716 - categorical_accuracy: 0.8872
17920/60000 [=======>......................] - ETA: 1:13 - loss: 0.3715 - categorical_accuracy: 0.8873
17952/60000 [=======>......................] - ETA: 1:13 - loss: 0.3712 - categorical_accuracy: 0.8874
17984/60000 [=======>......................] - ETA: 1:13 - loss: 0.3711 - categorical_accuracy: 0.8874
18016/60000 [========>.....................] - ETA: 1:13 - loss: 0.3709 - categorical_accuracy: 0.8875
18048/60000 [========>.....................] - ETA: 1:13 - loss: 0.3707 - categorical_accuracy: 0.8875
18080/60000 [========>.....................] - ETA: 1:13 - loss: 0.3703 - categorical_accuracy: 0.8877
18112/60000 [========>.....................] - ETA: 1:13 - loss: 0.3698 - categorical_accuracy: 0.8879
18144/60000 [========>.....................] - ETA: 1:13 - loss: 0.3696 - categorical_accuracy: 0.8880
18176/60000 [========>.....................] - ETA: 1:13 - loss: 0.3692 - categorical_accuracy: 0.8881
18208/60000 [========>.....................] - ETA: 1:13 - loss: 0.3687 - categorical_accuracy: 0.8882
18240/60000 [========>.....................] - ETA: 1:13 - loss: 0.3682 - categorical_accuracy: 0.8884
18272/60000 [========>.....................] - ETA: 1:13 - loss: 0.3679 - categorical_accuracy: 0.8884
18304/60000 [========>.....................] - ETA: 1:12 - loss: 0.3677 - categorical_accuracy: 0.8884
18336/60000 [========>.....................] - ETA: 1:12 - loss: 0.3671 - categorical_accuracy: 0.8885
18368/60000 [========>.....................] - ETA: 1:12 - loss: 0.3666 - categorical_accuracy: 0.8887
18400/60000 [========>.....................] - ETA: 1:12 - loss: 0.3664 - categorical_accuracy: 0.8888
18432/60000 [========>.....................] - ETA: 1:12 - loss: 0.3660 - categorical_accuracy: 0.8889
18464/60000 [========>.....................] - ETA: 1:12 - loss: 0.3662 - categorical_accuracy: 0.8890
18496/60000 [========>.....................] - ETA: 1:12 - loss: 0.3658 - categorical_accuracy: 0.8891
18528/60000 [========>.....................] - ETA: 1:12 - loss: 0.3654 - categorical_accuracy: 0.8892
18560/60000 [========>.....................] - ETA: 1:12 - loss: 0.3649 - categorical_accuracy: 0.8893
18592/60000 [========>.....................] - ETA: 1:12 - loss: 0.3644 - categorical_accuracy: 0.8895
18624/60000 [========>.....................] - ETA: 1:12 - loss: 0.3642 - categorical_accuracy: 0.8896
18656/60000 [========>.....................] - ETA: 1:12 - loss: 0.3640 - categorical_accuracy: 0.8896
18688/60000 [========>.....................] - ETA: 1:12 - loss: 0.3640 - categorical_accuracy: 0.8897
18720/60000 [========>.....................] - ETA: 1:12 - loss: 0.3644 - categorical_accuracy: 0.8896
18752/60000 [========>.....................] - ETA: 1:12 - loss: 0.3639 - categorical_accuracy: 0.8898
18784/60000 [========>.....................] - ETA: 1:12 - loss: 0.3639 - categorical_accuracy: 0.8897
18816/60000 [========>.....................] - ETA: 1:12 - loss: 0.3635 - categorical_accuracy: 0.8898
18848/60000 [========>.....................] - ETA: 1:11 - loss: 0.3633 - categorical_accuracy: 0.8898
18880/60000 [========>.....................] - ETA: 1:11 - loss: 0.3629 - categorical_accuracy: 0.8899
18912/60000 [========>.....................] - ETA: 1:11 - loss: 0.3628 - categorical_accuracy: 0.8900
18944/60000 [========>.....................] - ETA: 1:11 - loss: 0.3623 - categorical_accuracy: 0.8902
18976/60000 [========>.....................] - ETA: 1:11 - loss: 0.3620 - categorical_accuracy: 0.8902
19008/60000 [========>.....................] - ETA: 1:11 - loss: 0.3617 - categorical_accuracy: 0.8902
19040/60000 [========>.....................] - ETA: 1:11 - loss: 0.3616 - categorical_accuracy: 0.8903
19072/60000 [========>.....................] - ETA: 1:11 - loss: 0.3611 - categorical_accuracy: 0.8905
19104/60000 [========>.....................] - ETA: 1:11 - loss: 0.3607 - categorical_accuracy: 0.8905
19136/60000 [========>.....................] - ETA: 1:11 - loss: 0.3603 - categorical_accuracy: 0.8907
19168/60000 [========>.....................] - ETA: 1:11 - loss: 0.3598 - categorical_accuracy: 0.8909
19200/60000 [========>.....................] - ETA: 1:11 - loss: 0.3594 - categorical_accuracy: 0.8910
19232/60000 [========>.....................] - ETA: 1:11 - loss: 0.3590 - categorical_accuracy: 0.8911
19264/60000 [========>.....................] - ETA: 1:11 - loss: 0.3586 - categorical_accuracy: 0.8912
19296/60000 [========>.....................] - ETA: 1:11 - loss: 0.3589 - categorical_accuracy: 0.8913
19328/60000 [========>.....................] - ETA: 1:11 - loss: 0.3584 - categorical_accuracy: 0.8915
19360/60000 [========>.....................] - ETA: 1:11 - loss: 0.3583 - categorical_accuracy: 0.8916
19392/60000 [========>.....................] - ETA: 1:11 - loss: 0.3578 - categorical_accuracy: 0.8917
19424/60000 [========>.....................] - ETA: 1:10 - loss: 0.3576 - categorical_accuracy: 0.8918
19456/60000 [========>.....................] - ETA: 1:10 - loss: 0.3573 - categorical_accuracy: 0.8919
19488/60000 [========>.....................] - ETA: 1:10 - loss: 0.3569 - categorical_accuracy: 0.8920
19520/60000 [========>.....................] - ETA: 1:10 - loss: 0.3563 - categorical_accuracy: 0.8922
19552/60000 [========>.....................] - ETA: 1:10 - loss: 0.3562 - categorical_accuracy: 0.8922
19584/60000 [========>.....................] - ETA: 1:10 - loss: 0.3558 - categorical_accuracy: 0.8923
19616/60000 [========>.....................] - ETA: 1:10 - loss: 0.3560 - categorical_accuracy: 0.8923
19648/60000 [========>.....................] - ETA: 1:10 - loss: 0.3557 - categorical_accuracy: 0.8924
19680/60000 [========>.....................] - ETA: 1:10 - loss: 0.3555 - categorical_accuracy: 0.8924
19712/60000 [========>.....................] - ETA: 1:10 - loss: 0.3555 - categorical_accuracy: 0.8924
19744/60000 [========>.....................] - ETA: 1:10 - loss: 0.3552 - categorical_accuracy: 0.8925
19776/60000 [========>.....................] - ETA: 1:10 - loss: 0.3550 - categorical_accuracy: 0.8925
19808/60000 [========>.....................] - ETA: 1:10 - loss: 0.3549 - categorical_accuracy: 0.8925
19840/60000 [========>.....................] - ETA: 1:10 - loss: 0.3545 - categorical_accuracy: 0.8926
19872/60000 [========>.....................] - ETA: 1:10 - loss: 0.3544 - categorical_accuracy: 0.8927
19904/60000 [========>.....................] - ETA: 1:10 - loss: 0.3540 - categorical_accuracy: 0.8928
19936/60000 [========>.....................] - ETA: 1:10 - loss: 0.3539 - categorical_accuracy: 0.8929
19968/60000 [========>.....................] - ETA: 1:10 - loss: 0.3536 - categorical_accuracy: 0.8930
20000/60000 [=========>....................] - ETA: 1:09 - loss: 0.3531 - categorical_accuracy: 0.8931
20032/60000 [=========>....................] - ETA: 1:09 - loss: 0.3530 - categorical_accuracy: 0.8932
20064/60000 [=========>....................] - ETA: 1:09 - loss: 0.3527 - categorical_accuracy: 0.8933
20096/60000 [=========>....................] - ETA: 1:09 - loss: 0.3526 - categorical_accuracy: 0.8934
20128/60000 [=========>....................] - ETA: 1:09 - loss: 0.3532 - categorical_accuracy: 0.8933
20160/60000 [=========>....................] - ETA: 1:09 - loss: 0.3527 - categorical_accuracy: 0.8935
20192/60000 [=========>....................] - ETA: 1:09 - loss: 0.3525 - categorical_accuracy: 0.8936
20224/60000 [=========>....................] - ETA: 1:09 - loss: 0.3520 - categorical_accuracy: 0.8937
20256/60000 [=========>....................] - ETA: 1:09 - loss: 0.3518 - categorical_accuracy: 0.8938
20288/60000 [=========>....................] - ETA: 1:09 - loss: 0.3517 - categorical_accuracy: 0.8939
20320/60000 [=========>....................] - ETA: 1:09 - loss: 0.3515 - categorical_accuracy: 0.8938
20352/60000 [=========>....................] - ETA: 1:09 - loss: 0.3511 - categorical_accuracy: 0.8940
20384/60000 [=========>....................] - ETA: 1:09 - loss: 0.3507 - categorical_accuracy: 0.8941
20416/60000 [=========>....................] - ETA: 1:09 - loss: 0.3505 - categorical_accuracy: 0.8942
20448/60000 [=========>....................] - ETA: 1:09 - loss: 0.3502 - categorical_accuracy: 0.8943
20480/60000 [=========>....................] - ETA: 1:09 - loss: 0.3499 - categorical_accuracy: 0.8944
20512/60000 [=========>....................] - ETA: 1:09 - loss: 0.3498 - categorical_accuracy: 0.8945
20544/60000 [=========>....................] - ETA: 1:09 - loss: 0.3502 - categorical_accuracy: 0.8944
20576/60000 [=========>....................] - ETA: 1:08 - loss: 0.3500 - categorical_accuracy: 0.8944
20608/60000 [=========>....................] - ETA: 1:08 - loss: 0.3497 - categorical_accuracy: 0.8945
20640/60000 [=========>....................] - ETA: 1:08 - loss: 0.3494 - categorical_accuracy: 0.8945
20672/60000 [=========>....................] - ETA: 1:08 - loss: 0.3491 - categorical_accuracy: 0.8946
20704/60000 [=========>....................] - ETA: 1:08 - loss: 0.3487 - categorical_accuracy: 0.8947
20736/60000 [=========>....................] - ETA: 1:08 - loss: 0.3482 - categorical_accuracy: 0.8949
20768/60000 [=========>....................] - ETA: 1:08 - loss: 0.3478 - categorical_accuracy: 0.8950
20800/60000 [=========>....................] - ETA: 1:08 - loss: 0.3475 - categorical_accuracy: 0.8950
20832/60000 [=========>....................] - ETA: 1:08 - loss: 0.3477 - categorical_accuracy: 0.8950
20864/60000 [=========>....................] - ETA: 1:08 - loss: 0.3476 - categorical_accuracy: 0.8951
20896/60000 [=========>....................] - ETA: 1:08 - loss: 0.3472 - categorical_accuracy: 0.8953
20928/60000 [=========>....................] - ETA: 1:08 - loss: 0.3468 - categorical_accuracy: 0.8954
20960/60000 [=========>....................] - ETA: 1:08 - loss: 0.3466 - categorical_accuracy: 0.8954
20992/60000 [=========>....................] - ETA: 1:08 - loss: 0.3466 - categorical_accuracy: 0.8954
21024/60000 [=========>....................] - ETA: 1:08 - loss: 0.3464 - categorical_accuracy: 0.8955
21056/60000 [=========>....................] - ETA: 1:08 - loss: 0.3462 - categorical_accuracy: 0.8955
21088/60000 [=========>....................] - ETA: 1:08 - loss: 0.3458 - categorical_accuracy: 0.8956
21120/60000 [=========>....................] - ETA: 1:07 - loss: 0.3457 - categorical_accuracy: 0.8957
21152/60000 [=========>....................] - ETA: 1:07 - loss: 0.3455 - categorical_accuracy: 0.8958
21184/60000 [=========>....................] - ETA: 1:07 - loss: 0.3451 - categorical_accuracy: 0.8959
21216/60000 [=========>....................] - ETA: 1:07 - loss: 0.3451 - categorical_accuracy: 0.8959
21248/60000 [=========>....................] - ETA: 1:07 - loss: 0.3447 - categorical_accuracy: 0.8960
21280/60000 [=========>....................] - ETA: 1:07 - loss: 0.3446 - categorical_accuracy: 0.8961
21312/60000 [=========>....................] - ETA: 1:07 - loss: 0.3445 - categorical_accuracy: 0.8960
21344/60000 [=========>....................] - ETA: 1:07 - loss: 0.3441 - categorical_accuracy: 0.8962
21376/60000 [=========>....................] - ETA: 1:07 - loss: 0.3439 - categorical_accuracy: 0.8962
21408/60000 [=========>....................] - ETA: 1:07 - loss: 0.3434 - categorical_accuracy: 0.8963
21440/60000 [=========>....................] - ETA: 1:07 - loss: 0.3432 - categorical_accuracy: 0.8964
21472/60000 [=========>....................] - ETA: 1:07 - loss: 0.3428 - categorical_accuracy: 0.8965
21504/60000 [=========>....................] - ETA: 1:07 - loss: 0.3426 - categorical_accuracy: 0.8966
21536/60000 [=========>....................] - ETA: 1:07 - loss: 0.3425 - categorical_accuracy: 0.8967
21568/60000 [=========>....................] - ETA: 1:07 - loss: 0.3421 - categorical_accuracy: 0.8967
21600/60000 [=========>....................] - ETA: 1:07 - loss: 0.3419 - categorical_accuracy: 0.8968
21632/60000 [=========>....................] - ETA: 1:07 - loss: 0.3418 - categorical_accuracy: 0.8969
21664/60000 [=========>....................] - ETA: 1:07 - loss: 0.3418 - categorical_accuracy: 0.8968
21696/60000 [=========>....................] - ETA: 1:07 - loss: 0.3415 - categorical_accuracy: 0.8969
21728/60000 [=========>....................] - ETA: 1:07 - loss: 0.3411 - categorical_accuracy: 0.8970
21760/60000 [=========>....................] - ETA: 1:06 - loss: 0.3406 - categorical_accuracy: 0.8972
21792/60000 [=========>....................] - ETA: 1:06 - loss: 0.3403 - categorical_accuracy: 0.8973
21824/60000 [=========>....................] - ETA: 1:06 - loss: 0.3399 - categorical_accuracy: 0.8974
21856/60000 [=========>....................] - ETA: 1:06 - loss: 0.3397 - categorical_accuracy: 0.8975
21888/60000 [=========>....................] - ETA: 1:06 - loss: 0.3395 - categorical_accuracy: 0.8976
21920/60000 [=========>....................] - ETA: 1:06 - loss: 0.3392 - categorical_accuracy: 0.8977
21952/60000 [=========>....................] - ETA: 1:06 - loss: 0.3388 - categorical_accuracy: 0.8978
21984/60000 [=========>....................] - ETA: 1:06 - loss: 0.3384 - categorical_accuracy: 0.8979
22016/60000 [==========>...................] - ETA: 1:06 - loss: 0.3382 - categorical_accuracy: 0.8980
22048/60000 [==========>...................] - ETA: 1:06 - loss: 0.3378 - categorical_accuracy: 0.8981
22080/60000 [==========>...................] - ETA: 1:06 - loss: 0.3375 - categorical_accuracy: 0.8982
22112/60000 [==========>...................] - ETA: 1:06 - loss: 0.3371 - categorical_accuracy: 0.8983
22144/60000 [==========>...................] - ETA: 1:06 - loss: 0.3367 - categorical_accuracy: 0.8984
22176/60000 [==========>...................] - ETA: 1:06 - loss: 0.3363 - categorical_accuracy: 0.8985
22208/60000 [==========>...................] - ETA: 1:06 - loss: 0.3359 - categorical_accuracy: 0.8986
22240/60000 [==========>...................] - ETA: 1:06 - loss: 0.3356 - categorical_accuracy: 0.8987
22272/60000 [==========>...................] - ETA: 1:06 - loss: 0.3355 - categorical_accuracy: 0.8988
22304/60000 [==========>...................] - ETA: 1:06 - loss: 0.3353 - categorical_accuracy: 0.8989
22336/60000 [==========>...................] - ETA: 1:05 - loss: 0.3353 - categorical_accuracy: 0.8988
22368/60000 [==========>...................] - ETA: 1:05 - loss: 0.3356 - categorical_accuracy: 0.8987
22400/60000 [==========>...................] - ETA: 1:05 - loss: 0.3351 - categorical_accuracy: 0.8989
22432/60000 [==========>...................] - ETA: 1:05 - loss: 0.3347 - categorical_accuracy: 0.8990
22464/60000 [==========>...................] - ETA: 1:05 - loss: 0.3343 - categorical_accuracy: 0.8992
22496/60000 [==========>...................] - ETA: 1:05 - loss: 0.3340 - categorical_accuracy: 0.8993
22528/60000 [==========>...................] - ETA: 1:05 - loss: 0.3336 - categorical_accuracy: 0.8994
22560/60000 [==========>...................] - ETA: 1:05 - loss: 0.3333 - categorical_accuracy: 0.8994
22592/60000 [==========>...................] - ETA: 1:05 - loss: 0.3328 - categorical_accuracy: 0.8996
22624/60000 [==========>...................] - ETA: 1:05 - loss: 0.3324 - categorical_accuracy: 0.8997
22656/60000 [==========>...................] - ETA: 1:05 - loss: 0.3324 - categorical_accuracy: 0.8998
22688/60000 [==========>...................] - ETA: 1:05 - loss: 0.3320 - categorical_accuracy: 0.8998
22720/60000 [==========>...................] - ETA: 1:05 - loss: 0.3317 - categorical_accuracy: 0.9000
22752/60000 [==========>...................] - ETA: 1:05 - loss: 0.3312 - categorical_accuracy: 0.9001
22784/60000 [==========>...................] - ETA: 1:05 - loss: 0.3309 - categorical_accuracy: 0.9002
22816/60000 [==========>...................] - ETA: 1:05 - loss: 0.3307 - categorical_accuracy: 0.9002
22848/60000 [==========>...................] - ETA: 1:05 - loss: 0.3304 - categorical_accuracy: 0.9003
22880/60000 [==========>...................] - ETA: 1:05 - loss: 0.3300 - categorical_accuracy: 0.9004
22912/60000 [==========>...................] - ETA: 1:04 - loss: 0.3298 - categorical_accuracy: 0.9005
22944/60000 [==========>...................] - ETA: 1:04 - loss: 0.3296 - categorical_accuracy: 0.9006
22976/60000 [==========>...................] - ETA: 1:04 - loss: 0.3293 - categorical_accuracy: 0.9007
23008/60000 [==========>...................] - ETA: 1:04 - loss: 0.3288 - categorical_accuracy: 0.9009
23040/60000 [==========>...................] - ETA: 1:04 - loss: 0.3285 - categorical_accuracy: 0.9010
23072/60000 [==========>...................] - ETA: 1:04 - loss: 0.3281 - categorical_accuracy: 0.9010
23104/60000 [==========>...................] - ETA: 1:04 - loss: 0.3280 - categorical_accuracy: 0.9011
23136/60000 [==========>...................] - ETA: 1:04 - loss: 0.3278 - categorical_accuracy: 0.9011
23168/60000 [==========>...................] - ETA: 1:04 - loss: 0.3275 - categorical_accuracy: 0.9012
23200/60000 [==========>...................] - ETA: 1:04 - loss: 0.3271 - categorical_accuracy: 0.9013
23232/60000 [==========>...................] - ETA: 1:04 - loss: 0.3267 - categorical_accuracy: 0.9015
23264/60000 [==========>...................] - ETA: 1:04 - loss: 0.3263 - categorical_accuracy: 0.9016
23296/60000 [==========>...................] - ETA: 1:04 - loss: 0.3261 - categorical_accuracy: 0.9017
23328/60000 [==========>...................] - ETA: 1:04 - loss: 0.3259 - categorical_accuracy: 0.9017
23360/60000 [==========>...................] - ETA: 1:04 - loss: 0.3258 - categorical_accuracy: 0.9018
23392/60000 [==========>...................] - ETA: 1:04 - loss: 0.3258 - categorical_accuracy: 0.9018
23424/60000 [==========>...................] - ETA: 1:04 - loss: 0.3258 - categorical_accuracy: 0.9018
23456/60000 [==========>...................] - ETA: 1:04 - loss: 0.3258 - categorical_accuracy: 0.9018
23488/60000 [==========>...................] - ETA: 1:03 - loss: 0.3256 - categorical_accuracy: 0.9019
23520/60000 [==========>...................] - ETA: 1:03 - loss: 0.3255 - categorical_accuracy: 0.9019
23552/60000 [==========>...................] - ETA: 1:03 - loss: 0.3251 - categorical_accuracy: 0.9020
23584/60000 [==========>...................] - ETA: 1:03 - loss: 0.3250 - categorical_accuracy: 0.9019
23616/60000 [==========>...................] - ETA: 1:03 - loss: 0.3247 - categorical_accuracy: 0.9020
23648/60000 [==========>...................] - ETA: 1:03 - loss: 0.3245 - categorical_accuracy: 0.9021
23680/60000 [==========>...................] - ETA: 1:03 - loss: 0.3241 - categorical_accuracy: 0.9022
23712/60000 [==========>...................] - ETA: 1:03 - loss: 0.3239 - categorical_accuracy: 0.9022
23744/60000 [==========>...................] - ETA: 1:03 - loss: 0.3235 - categorical_accuracy: 0.9023
23776/60000 [==========>...................] - ETA: 1:03 - loss: 0.3234 - categorical_accuracy: 0.9024
23808/60000 [==========>...................] - ETA: 1:03 - loss: 0.3231 - categorical_accuracy: 0.9025
23840/60000 [==========>...................] - ETA: 1:03 - loss: 0.3228 - categorical_accuracy: 0.9026
23872/60000 [==========>...................] - ETA: 1:03 - loss: 0.3225 - categorical_accuracy: 0.9026
23904/60000 [==========>...................] - ETA: 1:03 - loss: 0.3225 - categorical_accuracy: 0.9027
23936/60000 [==========>...................] - ETA: 1:03 - loss: 0.3223 - categorical_accuracy: 0.9027
23968/60000 [==========>...................] - ETA: 1:03 - loss: 0.3220 - categorical_accuracy: 0.9028
24000/60000 [===========>..................] - ETA: 1:03 - loss: 0.3217 - categorical_accuracy: 0.9029
24032/60000 [===========>..................] - ETA: 1:03 - loss: 0.3214 - categorical_accuracy: 0.9030
24064/60000 [===========>..................] - ETA: 1:02 - loss: 0.3213 - categorical_accuracy: 0.9030
24096/60000 [===========>..................] - ETA: 1:02 - loss: 0.3210 - categorical_accuracy: 0.9031
24128/60000 [===========>..................] - ETA: 1:02 - loss: 0.3208 - categorical_accuracy: 0.9032
24160/60000 [===========>..................] - ETA: 1:02 - loss: 0.3210 - categorical_accuracy: 0.9032
24192/60000 [===========>..................] - ETA: 1:02 - loss: 0.3210 - categorical_accuracy: 0.9031
24224/60000 [===========>..................] - ETA: 1:02 - loss: 0.3207 - categorical_accuracy: 0.9032
24256/60000 [===========>..................] - ETA: 1:02 - loss: 0.3203 - categorical_accuracy: 0.9034
24288/60000 [===========>..................] - ETA: 1:02 - loss: 0.3200 - categorical_accuracy: 0.9035
24320/60000 [===========>..................] - ETA: 1:02 - loss: 0.3197 - categorical_accuracy: 0.9036
24352/60000 [===========>..................] - ETA: 1:02 - loss: 0.3195 - categorical_accuracy: 0.9036
24384/60000 [===========>..................] - ETA: 1:02 - loss: 0.3192 - categorical_accuracy: 0.9037
24416/60000 [===========>..................] - ETA: 1:02 - loss: 0.3190 - categorical_accuracy: 0.9037
24448/60000 [===========>..................] - ETA: 1:02 - loss: 0.3187 - categorical_accuracy: 0.9038
24480/60000 [===========>..................] - ETA: 1:02 - loss: 0.3183 - categorical_accuracy: 0.9040
24512/60000 [===========>..................] - ETA: 1:02 - loss: 0.3182 - categorical_accuracy: 0.9040
24544/60000 [===========>..................] - ETA: 1:02 - loss: 0.3178 - categorical_accuracy: 0.9041
24576/60000 [===========>..................] - ETA: 1:02 - loss: 0.3175 - categorical_accuracy: 0.9042
24608/60000 [===========>..................] - ETA: 1:02 - loss: 0.3173 - categorical_accuracy: 0.9042
24640/60000 [===========>..................] - ETA: 1:01 - loss: 0.3169 - categorical_accuracy: 0.9043
24672/60000 [===========>..................] - ETA: 1:01 - loss: 0.3165 - categorical_accuracy: 0.9045
24704/60000 [===========>..................] - ETA: 1:01 - loss: 0.3162 - categorical_accuracy: 0.9045
24736/60000 [===========>..................] - ETA: 1:01 - loss: 0.3160 - categorical_accuracy: 0.9046
24768/60000 [===========>..................] - ETA: 1:01 - loss: 0.3157 - categorical_accuracy: 0.9047
24800/60000 [===========>..................] - ETA: 1:01 - loss: 0.3158 - categorical_accuracy: 0.9048
24832/60000 [===========>..................] - ETA: 1:01 - loss: 0.3156 - categorical_accuracy: 0.9048
24864/60000 [===========>..................] - ETA: 1:01 - loss: 0.3152 - categorical_accuracy: 0.9049
24896/60000 [===========>..................] - ETA: 1:01 - loss: 0.3149 - categorical_accuracy: 0.9050
24928/60000 [===========>..................] - ETA: 1:01 - loss: 0.3146 - categorical_accuracy: 0.9051
24960/60000 [===========>..................] - ETA: 1:01 - loss: 0.3143 - categorical_accuracy: 0.9052
24992/60000 [===========>..................] - ETA: 1:01 - loss: 0.3142 - categorical_accuracy: 0.9052
25024/60000 [===========>..................] - ETA: 1:01 - loss: 0.3138 - categorical_accuracy: 0.9053
25056/60000 [===========>..................] - ETA: 1:01 - loss: 0.3135 - categorical_accuracy: 0.9055
25088/60000 [===========>..................] - ETA: 1:01 - loss: 0.3131 - categorical_accuracy: 0.9056
25120/60000 [===========>..................] - ETA: 1:01 - loss: 0.3127 - categorical_accuracy: 0.9057
25152/60000 [===========>..................] - ETA: 1:01 - loss: 0.3125 - categorical_accuracy: 0.9058
25184/60000 [===========>..................] - ETA: 1:01 - loss: 0.3122 - categorical_accuracy: 0.9059
25216/60000 [===========>..................] - ETA: 1:00 - loss: 0.3118 - categorical_accuracy: 0.9060
25248/60000 [===========>..................] - ETA: 1:00 - loss: 0.3115 - categorical_accuracy: 0.9061
25280/60000 [===========>..................] - ETA: 1:00 - loss: 0.3111 - categorical_accuracy: 0.9061
25312/60000 [===========>..................] - ETA: 1:00 - loss: 0.3108 - categorical_accuracy: 0.9062
25344/60000 [===========>..................] - ETA: 1:00 - loss: 0.3104 - categorical_accuracy: 0.9064
25376/60000 [===========>..................] - ETA: 1:00 - loss: 0.3102 - categorical_accuracy: 0.9064
25408/60000 [===========>..................] - ETA: 1:00 - loss: 0.3100 - categorical_accuracy: 0.9064
25440/60000 [===========>..................] - ETA: 1:00 - loss: 0.3106 - categorical_accuracy: 0.9064
25472/60000 [===========>..................] - ETA: 1:00 - loss: 0.3106 - categorical_accuracy: 0.9064
25504/60000 [===========>..................] - ETA: 1:00 - loss: 0.3103 - categorical_accuracy: 0.9064
25536/60000 [===========>..................] - ETA: 1:00 - loss: 0.3101 - categorical_accuracy: 0.9065
25568/60000 [===========>..................] - ETA: 1:00 - loss: 0.3101 - categorical_accuracy: 0.9065
25600/60000 [===========>..................] - ETA: 1:00 - loss: 0.3098 - categorical_accuracy: 0.9066
25632/60000 [===========>..................] - ETA: 1:00 - loss: 0.3095 - categorical_accuracy: 0.9067
25664/60000 [===========>..................] - ETA: 1:00 - loss: 0.3092 - categorical_accuracy: 0.9068
25696/60000 [===========>..................] - ETA: 1:00 - loss: 0.3090 - categorical_accuracy: 0.9069
25728/60000 [===========>..................] - ETA: 1:00 - loss: 0.3088 - categorical_accuracy: 0.9069
25760/60000 [===========>..................] - ETA: 1:00 - loss: 0.3086 - categorical_accuracy: 0.9069
25792/60000 [===========>..................] - ETA: 59s - loss: 0.3085 - categorical_accuracy: 0.9069 
25824/60000 [===========>..................] - ETA: 59s - loss: 0.3084 - categorical_accuracy: 0.9070
25856/60000 [===========>..................] - ETA: 59s - loss: 0.3082 - categorical_accuracy: 0.9071
25888/60000 [===========>..................] - ETA: 59s - loss: 0.3080 - categorical_accuracy: 0.9071
25920/60000 [===========>..................] - ETA: 59s - loss: 0.3078 - categorical_accuracy: 0.9072
25952/60000 [===========>..................] - ETA: 59s - loss: 0.3076 - categorical_accuracy: 0.9073
25984/60000 [===========>..................] - ETA: 59s - loss: 0.3074 - categorical_accuracy: 0.9073
26016/60000 [============>.................] - ETA: 59s - loss: 0.3071 - categorical_accuracy: 0.9074
26048/60000 [============>.................] - ETA: 59s - loss: 0.3072 - categorical_accuracy: 0.9074
26080/60000 [============>.................] - ETA: 59s - loss: 0.3069 - categorical_accuracy: 0.9074
26112/60000 [============>.................] - ETA: 59s - loss: 0.3066 - categorical_accuracy: 0.9075
26144/60000 [============>.................] - ETA: 59s - loss: 0.3063 - categorical_accuracy: 0.9076
26176/60000 [============>.................] - ETA: 59s - loss: 0.3063 - categorical_accuracy: 0.9075
26208/60000 [============>.................] - ETA: 59s - loss: 0.3060 - categorical_accuracy: 0.9077
26240/60000 [============>.................] - ETA: 59s - loss: 0.3056 - categorical_accuracy: 0.9078
26272/60000 [============>.................] - ETA: 59s - loss: 0.3053 - categorical_accuracy: 0.9079
26304/60000 [============>.................] - ETA: 59s - loss: 0.3053 - categorical_accuracy: 0.9079
26336/60000 [============>.................] - ETA: 58s - loss: 0.3050 - categorical_accuracy: 0.9080
26368/60000 [============>.................] - ETA: 58s - loss: 0.3048 - categorical_accuracy: 0.9081
26400/60000 [============>.................] - ETA: 58s - loss: 0.3045 - categorical_accuracy: 0.9082
26432/60000 [============>.................] - ETA: 58s - loss: 0.3041 - categorical_accuracy: 0.9083
26464/60000 [============>.................] - ETA: 58s - loss: 0.3039 - categorical_accuracy: 0.9083
26496/60000 [============>.................] - ETA: 58s - loss: 0.3036 - categorical_accuracy: 0.9084
26528/60000 [============>.................] - ETA: 58s - loss: 0.3033 - categorical_accuracy: 0.9085
26560/60000 [============>.................] - ETA: 58s - loss: 0.3033 - categorical_accuracy: 0.9085
26592/60000 [============>.................] - ETA: 58s - loss: 0.3030 - categorical_accuracy: 0.9086
26624/60000 [============>.................] - ETA: 58s - loss: 0.3027 - categorical_accuracy: 0.9087
26656/60000 [============>.................] - ETA: 58s - loss: 0.3025 - categorical_accuracy: 0.9088
26688/60000 [============>.................] - ETA: 58s - loss: 0.3022 - categorical_accuracy: 0.9089
26720/60000 [============>.................] - ETA: 58s - loss: 0.3019 - categorical_accuracy: 0.9090
26752/60000 [============>.................] - ETA: 58s - loss: 0.3016 - categorical_accuracy: 0.9091
26784/60000 [============>.................] - ETA: 58s - loss: 0.3014 - categorical_accuracy: 0.9091
26816/60000 [============>.................] - ETA: 58s - loss: 0.3012 - categorical_accuracy: 0.9092
26848/60000 [============>.................] - ETA: 58s - loss: 0.3011 - categorical_accuracy: 0.9092
26880/60000 [============>.................] - ETA: 58s - loss: 0.3008 - categorical_accuracy: 0.9093
26912/60000 [============>.................] - ETA: 57s - loss: 0.3005 - categorical_accuracy: 0.9094
26944/60000 [============>.................] - ETA: 57s - loss: 0.3002 - categorical_accuracy: 0.9095
26976/60000 [============>.................] - ETA: 57s - loss: 0.3000 - categorical_accuracy: 0.9096
27008/60000 [============>.................] - ETA: 57s - loss: 0.3001 - categorical_accuracy: 0.9096
27040/60000 [============>.................] - ETA: 57s - loss: 0.2998 - categorical_accuracy: 0.9097
27072/60000 [============>.................] - ETA: 57s - loss: 0.2995 - categorical_accuracy: 0.9098
27104/60000 [============>.................] - ETA: 57s - loss: 0.2992 - categorical_accuracy: 0.9099
27136/60000 [============>.................] - ETA: 57s - loss: 0.2990 - categorical_accuracy: 0.9099
27168/60000 [============>.................] - ETA: 57s - loss: 0.2988 - categorical_accuracy: 0.9100
27200/60000 [============>.................] - ETA: 57s - loss: 0.2986 - categorical_accuracy: 0.9100
27232/60000 [============>.................] - ETA: 57s - loss: 0.2984 - categorical_accuracy: 0.9101
27264/60000 [============>.................] - ETA: 57s - loss: 0.2981 - categorical_accuracy: 0.9102
27296/60000 [============>.................] - ETA: 57s - loss: 0.2978 - categorical_accuracy: 0.9102
27328/60000 [============>.................] - ETA: 57s - loss: 0.2979 - categorical_accuracy: 0.9102
27360/60000 [============>.................] - ETA: 57s - loss: 0.2977 - categorical_accuracy: 0.9102
27392/60000 [============>.................] - ETA: 57s - loss: 0.2974 - categorical_accuracy: 0.9103
27424/60000 [============>.................] - ETA: 57s - loss: 0.2972 - categorical_accuracy: 0.9104
27456/60000 [============>.................] - ETA: 57s - loss: 0.2971 - categorical_accuracy: 0.9104
27488/60000 [============>.................] - ETA: 56s - loss: 0.2968 - categorical_accuracy: 0.9105
27520/60000 [============>.................] - ETA: 56s - loss: 0.2965 - categorical_accuracy: 0.9106
27552/60000 [============>.................] - ETA: 56s - loss: 0.2963 - categorical_accuracy: 0.9107
27584/60000 [============>.................] - ETA: 56s - loss: 0.2961 - categorical_accuracy: 0.9107
27616/60000 [============>.................] - ETA: 56s - loss: 0.2960 - categorical_accuracy: 0.9108
27648/60000 [============>.................] - ETA: 56s - loss: 0.2958 - categorical_accuracy: 0.9108
27680/60000 [============>.................] - ETA: 56s - loss: 0.2956 - categorical_accuracy: 0.9109
27712/60000 [============>.................] - ETA: 56s - loss: 0.2955 - categorical_accuracy: 0.9109
27744/60000 [============>.................] - ETA: 56s - loss: 0.2953 - categorical_accuracy: 0.9109
27776/60000 [============>.................] - ETA: 56s - loss: 0.2956 - categorical_accuracy: 0.9109
27808/60000 [============>.................] - ETA: 56s - loss: 0.2958 - categorical_accuracy: 0.9109
27840/60000 [============>.................] - ETA: 56s - loss: 0.2957 - categorical_accuracy: 0.9109
27872/60000 [============>.................] - ETA: 56s - loss: 0.2955 - categorical_accuracy: 0.9110
27904/60000 [============>.................] - ETA: 56s - loss: 0.2952 - categorical_accuracy: 0.9111
27936/60000 [============>.................] - ETA: 56s - loss: 0.2950 - categorical_accuracy: 0.9111
27968/60000 [============>.................] - ETA: 56s - loss: 0.2953 - categorical_accuracy: 0.9111
28000/60000 [=============>................] - ETA: 56s - loss: 0.2951 - categorical_accuracy: 0.9112
28032/60000 [=============>................] - ETA: 56s - loss: 0.2948 - categorical_accuracy: 0.9113
28064/60000 [=============>................] - ETA: 55s - loss: 0.2947 - categorical_accuracy: 0.9113
28096/60000 [=============>................] - ETA: 55s - loss: 0.2944 - categorical_accuracy: 0.9114
28128/60000 [=============>................] - ETA: 55s - loss: 0.2942 - categorical_accuracy: 0.9115
28160/60000 [=============>................] - ETA: 55s - loss: 0.2939 - categorical_accuracy: 0.9116
28192/60000 [=============>................] - ETA: 55s - loss: 0.2936 - categorical_accuracy: 0.9117
28224/60000 [=============>................] - ETA: 55s - loss: 0.2933 - categorical_accuracy: 0.9118
28256/60000 [=============>................] - ETA: 55s - loss: 0.2930 - categorical_accuracy: 0.9119
28288/60000 [=============>................] - ETA: 55s - loss: 0.2928 - categorical_accuracy: 0.9119
28320/60000 [=============>................] - ETA: 55s - loss: 0.2925 - categorical_accuracy: 0.9120
28352/60000 [=============>................] - ETA: 55s - loss: 0.2924 - categorical_accuracy: 0.9121
28384/60000 [=============>................] - ETA: 55s - loss: 0.2921 - categorical_accuracy: 0.9121
28416/60000 [=============>................] - ETA: 55s - loss: 0.2919 - categorical_accuracy: 0.9122
28448/60000 [=============>................] - ETA: 55s - loss: 0.2917 - categorical_accuracy: 0.9122
28480/60000 [=============>................] - ETA: 55s - loss: 0.2916 - categorical_accuracy: 0.9122
28512/60000 [=============>................] - ETA: 55s - loss: 0.2914 - categorical_accuracy: 0.9123
28544/60000 [=============>................] - ETA: 55s - loss: 0.2911 - categorical_accuracy: 0.9124
28576/60000 [=============>................] - ETA: 55s - loss: 0.2909 - categorical_accuracy: 0.9124
28608/60000 [=============>................] - ETA: 54s - loss: 0.2906 - categorical_accuracy: 0.9125
28640/60000 [=============>................] - ETA: 54s - loss: 0.2904 - categorical_accuracy: 0.9126
28672/60000 [=============>................] - ETA: 54s - loss: 0.2902 - categorical_accuracy: 0.9126
28704/60000 [=============>................] - ETA: 54s - loss: 0.2900 - categorical_accuracy: 0.9127
28736/60000 [=============>................] - ETA: 54s - loss: 0.2897 - categorical_accuracy: 0.9128
28768/60000 [=============>................] - ETA: 54s - loss: 0.2894 - categorical_accuracy: 0.9129
28800/60000 [=============>................] - ETA: 54s - loss: 0.2893 - categorical_accuracy: 0.9129
28832/60000 [=============>................] - ETA: 54s - loss: 0.2890 - categorical_accuracy: 0.9130
28864/60000 [=============>................] - ETA: 54s - loss: 0.2888 - categorical_accuracy: 0.9131
28896/60000 [=============>................] - ETA: 54s - loss: 0.2886 - categorical_accuracy: 0.9131
28928/60000 [=============>................] - ETA: 54s - loss: 0.2886 - categorical_accuracy: 0.9131
28960/60000 [=============>................] - ETA: 54s - loss: 0.2886 - categorical_accuracy: 0.9131
28992/60000 [=============>................] - ETA: 54s - loss: 0.2884 - categorical_accuracy: 0.9131
29024/60000 [=============>................] - ETA: 54s - loss: 0.2881 - categorical_accuracy: 0.9132
29056/60000 [=============>................] - ETA: 54s - loss: 0.2882 - categorical_accuracy: 0.9132
29088/60000 [=============>................] - ETA: 54s - loss: 0.2879 - categorical_accuracy: 0.9133
29120/60000 [=============>................] - ETA: 54s - loss: 0.2877 - categorical_accuracy: 0.9134
29152/60000 [=============>................] - ETA: 54s - loss: 0.2874 - categorical_accuracy: 0.9135
29184/60000 [=============>................] - ETA: 53s - loss: 0.2873 - categorical_accuracy: 0.9135
29216/60000 [=============>................] - ETA: 53s - loss: 0.2871 - categorical_accuracy: 0.9136
29248/60000 [=============>................] - ETA: 53s - loss: 0.2869 - categorical_accuracy: 0.9136
29280/60000 [=============>................] - ETA: 53s - loss: 0.2870 - categorical_accuracy: 0.9136
29312/60000 [=============>................] - ETA: 53s - loss: 0.2868 - categorical_accuracy: 0.9137
29344/60000 [=============>................] - ETA: 53s - loss: 0.2866 - categorical_accuracy: 0.9137
29376/60000 [=============>................] - ETA: 53s - loss: 0.2864 - categorical_accuracy: 0.9137
29408/60000 [=============>................] - ETA: 53s - loss: 0.2861 - categorical_accuracy: 0.9138
29440/60000 [=============>................] - ETA: 53s - loss: 0.2858 - categorical_accuracy: 0.9139
29472/60000 [=============>................] - ETA: 53s - loss: 0.2856 - categorical_accuracy: 0.9140
29504/60000 [=============>................] - ETA: 53s - loss: 0.2853 - categorical_accuracy: 0.9141
29536/60000 [=============>................] - ETA: 53s - loss: 0.2851 - categorical_accuracy: 0.9142
29568/60000 [=============>................] - ETA: 53s - loss: 0.2849 - categorical_accuracy: 0.9142
29600/60000 [=============>................] - ETA: 53s - loss: 0.2847 - categorical_accuracy: 0.9142
29632/60000 [=============>................] - ETA: 53s - loss: 0.2845 - categorical_accuracy: 0.9143
29664/60000 [=============>................] - ETA: 53s - loss: 0.2842 - categorical_accuracy: 0.9144
29696/60000 [=============>................] - ETA: 53s - loss: 0.2840 - categorical_accuracy: 0.9145
29728/60000 [=============>................] - ETA: 53s - loss: 0.2838 - categorical_accuracy: 0.9145
29760/60000 [=============>................] - ETA: 52s - loss: 0.2835 - categorical_accuracy: 0.9146
29792/60000 [=============>................] - ETA: 52s - loss: 0.2833 - categorical_accuracy: 0.9147
29824/60000 [=============>................] - ETA: 52s - loss: 0.2831 - categorical_accuracy: 0.9147
29856/60000 [=============>................] - ETA: 52s - loss: 0.2829 - categorical_accuracy: 0.9148
29888/60000 [=============>................] - ETA: 52s - loss: 0.2826 - categorical_accuracy: 0.9149
29920/60000 [=============>................] - ETA: 52s - loss: 0.2824 - categorical_accuracy: 0.9149
29952/60000 [=============>................] - ETA: 52s - loss: 0.2823 - categorical_accuracy: 0.9150
29984/60000 [=============>................] - ETA: 52s - loss: 0.2820 - categorical_accuracy: 0.9151
30016/60000 [==============>...............] - ETA: 52s - loss: 0.2818 - categorical_accuracy: 0.9151
30048/60000 [==============>...............] - ETA: 52s - loss: 0.2818 - categorical_accuracy: 0.9151
30080/60000 [==============>...............] - ETA: 52s - loss: 0.2819 - categorical_accuracy: 0.9152
30112/60000 [==============>...............] - ETA: 52s - loss: 0.2818 - categorical_accuracy: 0.9152
30144/60000 [==============>...............] - ETA: 52s - loss: 0.2816 - categorical_accuracy: 0.9153
30176/60000 [==============>...............] - ETA: 52s - loss: 0.2813 - categorical_accuracy: 0.9154
30208/60000 [==============>...............] - ETA: 52s - loss: 0.2811 - categorical_accuracy: 0.9155
30240/60000 [==============>...............] - ETA: 52s - loss: 0.2808 - categorical_accuracy: 0.9155
30272/60000 [==============>...............] - ETA: 52s - loss: 0.2806 - categorical_accuracy: 0.9156
30304/60000 [==============>...............] - ETA: 51s - loss: 0.2803 - categorical_accuracy: 0.9156
30336/60000 [==============>...............] - ETA: 51s - loss: 0.2803 - categorical_accuracy: 0.9156
30368/60000 [==============>...............] - ETA: 51s - loss: 0.2801 - categorical_accuracy: 0.9157
30400/60000 [==============>...............] - ETA: 51s - loss: 0.2800 - categorical_accuracy: 0.9158
30432/60000 [==============>...............] - ETA: 51s - loss: 0.2799 - categorical_accuracy: 0.9158
30464/60000 [==============>...............] - ETA: 51s - loss: 0.2797 - categorical_accuracy: 0.9159
30496/60000 [==============>...............] - ETA: 51s - loss: 0.2796 - categorical_accuracy: 0.9159
30528/60000 [==============>...............] - ETA: 51s - loss: 0.2793 - categorical_accuracy: 0.9160
30560/60000 [==============>...............] - ETA: 51s - loss: 0.2792 - categorical_accuracy: 0.9161
30592/60000 [==============>...............] - ETA: 51s - loss: 0.2789 - categorical_accuracy: 0.9162
30624/60000 [==============>...............] - ETA: 51s - loss: 0.2786 - categorical_accuracy: 0.9162
30656/60000 [==============>...............] - ETA: 51s - loss: 0.2788 - categorical_accuracy: 0.9162
30688/60000 [==============>...............] - ETA: 51s - loss: 0.2786 - categorical_accuracy: 0.9163
30720/60000 [==============>...............] - ETA: 51s - loss: 0.2784 - categorical_accuracy: 0.9163
30752/60000 [==============>...............] - ETA: 51s - loss: 0.2781 - categorical_accuracy: 0.9164
30784/60000 [==============>...............] - ETA: 51s - loss: 0.2779 - categorical_accuracy: 0.9165
30816/60000 [==============>...............] - ETA: 51s - loss: 0.2776 - categorical_accuracy: 0.9166
30848/60000 [==============>...............] - ETA: 51s - loss: 0.2779 - categorical_accuracy: 0.9166
30880/60000 [==============>...............] - ETA: 50s - loss: 0.2778 - categorical_accuracy: 0.9166
30912/60000 [==============>...............] - ETA: 50s - loss: 0.2776 - categorical_accuracy: 0.9167
30944/60000 [==============>...............] - ETA: 50s - loss: 0.2777 - categorical_accuracy: 0.9167
30976/60000 [==============>...............] - ETA: 50s - loss: 0.2776 - categorical_accuracy: 0.9167
31008/60000 [==============>...............] - ETA: 50s - loss: 0.2773 - categorical_accuracy: 0.9168
31040/60000 [==============>...............] - ETA: 50s - loss: 0.2771 - categorical_accuracy: 0.9168
31072/60000 [==============>...............] - ETA: 50s - loss: 0.2770 - categorical_accuracy: 0.9169
31104/60000 [==============>...............] - ETA: 50s - loss: 0.2768 - categorical_accuracy: 0.9169
31136/60000 [==============>...............] - ETA: 50s - loss: 0.2768 - categorical_accuracy: 0.9169
31168/60000 [==============>...............] - ETA: 50s - loss: 0.2768 - categorical_accuracy: 0.9169
31200/60000 [==============>...............] - ETA: 50s - loss: 0.2766 - categorical_accuracy: 0.9170
31232/60000 [==============>...............] - ETA: 50s - loss: 0.2764 - categorical_accuracy: 0.9170
31264/60000 [==============>...............] - ETA: 50s - loss: 0.2764 - categorical_accuracy: 0.9170
31296/60000 [==============>...............] - ETA: 50s - loss: 0.2762 - categorical_accuracy: 0.9171
31328/60000 [==============>...............] - ETA: 50s - loss: 0.2760 - categorical_accuracy: 0.9171
31360/60000 [==============>...............] - ETA: 50s - loss: 0.2758 - categorical_accuracy: 0.9172
31392/60000 [==============>...............] - ETA: 50s - loss: 0.2756 - categorical_accuracy: 0.9172
31424/60000 [==============>...............] - ETA: 50s - loss: 0.2754 - categorical_accuracy: 0.9172
31456/60000 [==============>...............] - ETA: 49s - loss: 0.2751 - categorical_accuracy: 0.9173
31488/60000 [==============>...............] - ETA: 49s - loss: 0.2749 - categorical_accuracy: 0.9174
31520/60000 [==============>...............] - ETA: 49s - loss: 0.2747 - categorical_accuracy: 0.9174
31552/60000 [==============>...............] - ETA: 49s - loss: 0.2745 - categorical_accuracy: 0.9175
31584/60000 [==============>...............] - ETA: 49s - loss: 0.2744 - categorical_accuracy: 0.9175
31616/60000 [==============>...............] - ETA: 49s - loss: 0.2742 - categorical_accuracy: 0.9176
31648/60000 [==============>...............] - ETA: 49s - loss: 0.2741 - categorical_accuracy: 0.9176
31680/60000 [==============>...............] - ETA: 49s - loss: 0.2740 - categorical_accuracy: 0.9177
31712/60000 [==============>...............] - ETA: 49s - loss: 0.2737 - categorical_accuracy: 0.9178
31744/60000 [==============>...............] - ETA: 49s - loss: 0.2735 - categorical_accuracy: 0.9178
31776/60000 [==============>...............] - ETA: 49s - loss: 0.2732 - categorical_accuracy: 0.9179
31808/60000 [==============>...............] - ETA: 49s - loss: 0.2731 - categorical_accuracy: 0.9180
31840/60000 [==============>...............] - ETA: 49s - loss: 0.2730 - categorical_accuracy: 0.9180
31872/60000 [==============>...............] - ETA: 49s - loss: 0.2732 - categorical_accuracy: 0.9180
31904/60000 [==============>...............] - ETA: 49s - loss: 0.2730 - categorical_accuracy: 0.9180
31936/60000 [==============>...............] - ETA: 49s - loss: 0.2728 - categorical_accuracy: 0.9181
31968/60000 [==============>...............] - ETA: 49s - loss: 0.2725 - categorical_accuracy: 0.9182
32000/60000 [===============>..............] - ETA: 49s - loss: 0.2722 - categorical_accuracy: 0.9183
32032/60000 [===============>..............] - ETA: 48s - loss: 0.2720 - categorical_accuracy: 0.9183
32064/60000 [===============>..............] - ETA: 48s - loss: 0.2722 - categorical_accuracy: 0.9184
32096/60000 [===============>..............] - ETA: 48s - loss: 0.2719 - categorical_accuracy: 0.9184
32128/60000 [===============>..............] - ETA: 48s - loss: 0.2717 - categorical_accuracy: 0.9185
32160/60000 [===============>..............] - ETA: 48s - loss: 0.2715 - categorical_accuracy: 0.9186
32192/60000 [===============>..............] - ETA: 48s - loss: 0.2713 - categorical_accuracy: 0.9186
32224/60000 [===============>..............] - ETA: 48s - loss: 0.2710 - categorical_accuracy: 0.9187
32256/60000 [===============>..............] - ETA: 48s - loss: 0.2709 - categorical_accuracy: 0.9187
32288/60000 [===============>..............] - ETA: 48s - loss: 0.2708 - categorical_accuracy: 0.9188
32320/60000 [===============>..............] - ETA: 48s - loss: 0.2706 - categorical_accuracy: 0.9188
32352/60000 [===============>..............] - ETA: 48s - loss: 0.2704 - categorical_accuracy: 0.9189
32384/60000 [===============>..............] - ETA: 48s - loss: 0.2701 - categorical_accuracy: 0.9190
32416/60000 [===============>..............] - ETA: 48s - loss: 0.2699 - categorical_accuracy: 0.9191
32448/60000 [===============>..............] - ETA: 48s - loss: 0.2697 - categorical_accuracy: 0.9191
32480/60000 [===============>..............] - ETA: 48s - loss: 0.2697 - categorical_accuracy: 0.9191
32512/60000 [===============>..............] - ETA: 48s - loss: 0.2695 - categorical_accuracy: 0.9192
32544/60000 [===============>..............] - ETA: 48s - loss: 0.2693 - categorical_accuracy: 0.9192
32576/60000 [===============>..............] - ETA: 48s - loss: 0.2693 - categorical_accuracy: 0.9193
32608/60000 [===============>..............] - ETA: 47s - loss: 0.2693 - categorical_accuracy: 0.9193
32640/60000 [===============>..............] - ETA: 47s - loss: 0.2692 - categorical_accuracy: 0.9193
32672/60000 [===============>..............] - ETA: 47s - loss: 0.2690 - categorical_accuracy: 0.9193
32704/60000 [===============>..............] - ETA: 47s - loss: 0.2689 - categorical_accuracy: 0.9193
32736/60000 [===============>..............] - ETA: 47s - loss: 0.2689 - categorical_accuracy: 0.9194
32768/60000 [===============>..............] - ETA: 47s - loss: 0.2687 - categorical_accuracy: 0.9194
32800/60000 [===============>..............] - ETA: 47s - loss: 0.2686 - categorical_accuracy: 0.9195
32832/60000 [===============>..............] - ETA: 47s - loss: 0.2685 - categorical_accuracy: 0.9194
32864/60000 [===============>..............] - ETA: 47s - loss: 0.2683 - categorical_accuracy: 0.9195
32896/60000 [===============>..............] - ETA: 47s - loss: 0.2682 - categorical_accuracy: 0.9195
32928/60000 [===============>..............] - ETA: 47s - loss: 0.2680 - categorical_accuracy: 0.9196
32960/60000 [===============>..............] - ETA: 47s - loss: 0.2678 - categorical_accuracy: 0.9196
32992/60000 [===============>..............] - ETA: 47s - loss: 0.2678 - categorical_accuracy: 0.9196
33024/60000 [===============>..............] - ETA: 47s - loss: 0.2676 - categorical_accuracy: 0.9197
33056/60000 [===============>..............] - ETA: 47s - loss: 0.2679 - categorical_accuracy: 0.9197
33088/60000 [===============>..............] - ETA: 47s - loss: 0.2677 - categorical_accuracy: 0.9197
33120/60000 [===============>..............] - ETA: 47s - loss: 0.2675 - categorical_accuracy: 0.9198
33152/60000 [===============>..............] - ETA: 46s - loss: 0.2675 - categorical_accuracy: 0.9199
33184/60000 [===============>..............] - ETA: 46s - loss: 0.2673 - categorical_accuracy: 0.9199
33216/60000 [===============>..............] - ETA: 46s - loss: 0.2674 - categorical_accuracy: 0.9199
33248/60000 [===============>..............] - ETA: 46s - loss: 0.2672 - categorical_accuracy: 0.9200
33280/60000 [===============>..............] - ETA: 46s - loss: 0.2670 - categorical_accuracy: 0.9201
33312/60000 [===============>..............] - ETA: 46s - loss: 0.2667 - categorical_accuracy: 0.9201
33344/60000 [===============>..............] - ETA: 46s - loss: 0.2665 - categorical_accuracy: 0.9202
33376/60000 [===============>..............] - ETA: 46s - loss: 0.2663 - categorical_accuracy: 0.9202
33408/60000 [===============>..............] - ETA: 46s - loss: 0.2662 - categorical_accuracy: 0.9203
33440/60000 [===============>..............] - ETA: 46s - loss: 0.2664 - categorical_accuracy: 0.9203
33472/60000 [===============>..............] - ETA: 46s - loss: 0.2663 - categorical_accuracy: 0.9203
33504/60000 [===============>..............] - ETA: 46s - loss: 0.2661 - categorical_accuracy: 0.9203
33536/60000 [===============>..............] - ETA: 46s - loss: 0.2660 - categorical_accuracy: 0.9204
33568/60000 [===============>..............] - ETA: 46s - loss: 0.2659 - categorical_accuracy: 0.9204
33600/60000 [===============>..............] - ETA: 46s - loss: 0.2656 - categorical_accuracy: 0.9205
33632/60000 [===============>..............] - ETA: 46s - loss: 0.2655 - categorical_accuracy: 0.9206
33664/60000 [===============>..............] - ETA: 46s - loss: 0.2652 - categorical_accuracy: 0.9206
33696/60000 [===============>..............] - ETA: 46s - loss: 0.2650 - categorical_accuracy: 0.9207
33728/60000 [===============>..............] - ETA: 45s - loss: 0.2649 - categorical_accuracy: 0.9207
33760/60000 [===============>..............] - ETA: 45s - loss: 0.2648 - categorical_accuracy: 0.9207
33792/60000 [===============>..............] - ETA: 45s - loss: 0.2647 - categorical_accuracy: 0.9208
33824/60000 [===============>..............] - ETA: 45s - loss: 0.2645 - categorical_accuracy: 0.9209
33856/60000 [===============>..............] - ETA: 45s - loss: 0.2645 - categorical_accuracy: 0.9208
33888/60000 [===============>..............] - ETA: 45s - loss: 0.2643 - categorical_accuracy: 0.9209
33920/60000 [===============>..............] - ETA: 45s - loss: 0.2641 - categorical_accuracy: 0.9209
33952/60000 [===============>..............] - ETA: 45s - loss: 0.2640 - categorical_accuracy: 0.9210
33984/60000 [===============>..............] - ETA: 45s - loss: 0.2637 - categorical_accuracy: 0.9211
34016/60000 [================>.............] - ETA: 45s - loss: 0.2635 - categorical_accuracy: 0.9211
34048/60000 [================>.............] - ETA: 45s - loss: 0.2634 - categorical_accuracy: 0.9211
34080/60000 [================>.............] - ETA: 45s - loss: 0.2633 - categorical_accuracy: 0.9212
34112/60000 [================>.............] - ETA: 45s - loss: 0.2630 - categorical_accuracy: 0.9213
34144/60000 [================>.............] - ETA: 45s - loss: 0.2629 - categorical_accuracy: 0.9213
34176/60000 [================>.............] - ETA: 45s - loss: 0.2627 - categorical_accuracy: 0.9213
34208/60000 [================>.............] - ETA: 45s - loss: 0.2625 - categorical_accuracy: 0.9214
34240/60000 [================>.............] - ETA: 45s - loss: 0.2623 - categorical_accuracy: 0.9214
34272/60000 [================>.............] - ETA: 45s - loss: 0.2623 - categorical_accuracy: 0.9215
34304/60000 [================>.............] - ETA: 44s - loss: 0.2621 - categorical_accuracy: 0.9215
34336/60000 [================>.............] - ETA: 44s - loss: 0.2621 - categorical_accuracy: 0.9215
34368/60000 [================>.............] - ETA: 44s - loss: 0.2620 - categorical_accuracy: 0.9215
34400/60000 [================>.............] - ETA: 44s - loss: 0.2620 - categorical_accuracy: 0.9215
34432/60000 [================>.............] - ETA: 44s - loss: 0.2620 - categorical_accuracy: 0.9215
34464/60000 [================>.............] - ETA: 44s - loss: 0.2618 - categorical_accuracy: 0.9216
34496/60000 [================>.............] - ETA: 44s - loss: 0.2617 - categorical_accuracy: 0.9216
34528/60000 [================>.............] - ETA: 44s - loss: 0.2615 - categorical_accuracy: 0.9217
34560/60000 [================>.............] - ETA: 44s - loss: 0.2612 - categorical_accuracy: 0.9217
34592/60000 [================>.............] - ETA: 44s - loss: 0.2615 - categorical_accuracy: 0.9217
34624/60000 [================>.............] - ETA: 44s - loss: 0.2616 - categorical_accuracy: 0.9218
34656/60000 [================>.............] - ETA: 44s - loss: 0.2615 - categorical_accuracy: 0.9218
34688/60000 [================>.............] - ETA: 44s - loss: 0.2612 - categorical_accuracy: 0.9219
34720/60000 [================>.............] - ETA: 44s - loss: 0.2612 - categorical_accuracy: 0.9219
34752/60000 [================>.............] - ETA: 44s - loss: 0.2611 - categorical_accuracy: 0.9219
34784/60000 [================>.............] - ETA: 44s - loss: 0.2611 - categorical_accuracy: 0.9219
34816/60000 [================>.............] - ETA: 44s - loss: 0.2608 - categorical_accuracy: 0.9220
34848/60000 [================>.............] - ETA: 44s - loss: 0.2606 - categorical_accuracy: 0.9221
34880/60000 [================>.............] - ETA: 43s - loss: 0.2606 - categorical_accuracy: 0.9221
34912/60000 [================>.............] - ETA: 43s - loss: 0.2605 - categorical_accuracy: 0.9221
34944/60000 [================>.............] - ETA: 43s - loss: 0.2603 - categorical_accuracy: 0.9222
34976/60000 [================>.............] - ETA: 43s - loss: 0.2602 - categorical_accuracy: 0.9222
35008/60000 [================>.............] - ETA: 43s - loss: 0.2600 - categorical_accuracy: 0.9222
35040/60000 [================>.............] - ETA: 43s - loss: 0.2599 - categorical_accuracy: 0.9223
35072/60000 [================>.............] - ETA: 43s - loss: 0.2598 - categorical_accuracy: 0.9223
35104/60000 [================>.............] - ETA: 43s - loss: 0.2597 - categorical_accuracy: 0.9223
35136/60000 [================>.............] - ETA: 43s - loss: 0.2597 - categorical_accuracy: 0.9223
35168/60000 [================>.............] - ETA: 43s - loss: 0.2597 - categorical_accuracy: 0.9223
35200/60000 [================>.............] - ETA: 43s - loss: 0.2596 - categorical_accuracy: 0.9223
35232/60000 [================>.............] - ETA: 43s - loss: 0.2595 - categorical_accuracy: 0.9223
35264/60000 [================>.............] - ETA: 43s - loss: 0.2593 - categorical_accuracy: 0.9224
35296/60000 [================>.............] - ETA: 43s - loss: 0.2592 - categorical_accuracy: 0.9224
35328/60000 [================>.............] - ETA: 43s - loss: 0.2589 - categorical_accuracy: 0.9225
35360/60000 [================>.............] - ETA: 43s - loss: 0.2588 - categorical_accuracy: 0.9225
35392/60000 [================>.............] - ETA: 43s - loss: 0.2587 - categorical_accuracy: 0.9225
35424/60000 [================>.............] - ETA: 43s - loss: 0.2586 - categorical_accuracy: 0.9226
35456/60000 [================>.............] - ETA: 42s - loss: 0.2583 - categorical_accuracy: 0.9226
35488/60000 [================>.............] - ETA: 42s - loss: 0.2586 - categorical_accuracy: 0.9226
35520/60000 [================>.............] - ETA: 42s - loss: 0.2585 - categorical_accuracy: 0.9226
35552/60000 [================>.............] - ETA: 42s - loss: 0.2583 - categorical_accuracy: 0.9227
35584/60000 [================>.............] - ETA: 42s - loss: 0.2581 - categorical_accuracy: 0.9228
35616/60000 [================>.............] - ETA: 42s - loss: 0.2579 - categorical_accuracy: 0.9228
35648/60000 [================>.............] - ETA: 42s - loss: 0.2577 - categorical_accuracy: 0.9229
35680/60000 [================>.............] - ETA: 42s - loss: 0.2577 - categorical_accuracy: 0.9229
35712/60000 [================>.............] - ETA: 42s - loss: 0.2577 - categorical_accuracy: 0.9229
35744/60000 [================>.............] - ETA: 42s - loss: 0.2576 - categorical_accuracy: 0.9230
35776/60000 [================>.............] - ETA: 42s - loss: 0.2575 - categorical_accuracy: 0.9230
35808/60000 [================>.............] - ETA: 42s - loss: 0.2574 - categorical_accuracy: 0.9230
35840/60000 [================>.............] - ETA: 42s - loss: 0.2572 - categorical_accuracy: 0.9230
35872/60000 [================>.............] - ETA: 42s - loss: 0.2571 - categorical_accuracy: 0.9231
35904/60000 [================>.............] - ETA: 42s - loss: 0.2569 - categorical_accuracy: 0.9231
35936/60000 [================>.............] - ETA: 42s - loss: 0.2568 - categorical_accuracy: 0.9231
35968/60000 [================>.............] - ETA: 42s - loss: 0.2566 - categorical_accuracy: 0.9232
36000/60000 [=================>............] - ETA: 42s - loss: 0.2564 - categorical_accuracy: 0.9232
36032/60000 [=================>............] - ETA: 41s - loss: 0.2564 - categorical_accuracy: 0.9233
36064/60000 [=================>............] - ETA: 41s - loss: 0.2562 - categorical_accuracy: 0.9233
36096/60000 [=================>............] - ETA: 41s - loss: 0.2560 - categorical_accuracy: 0.9234
36128/60000 [=================>............] - ETA: 41s - loss: 0.2560 - categorical_accuracy: 0.9234
36160/60000 [=================>............] - ETA: 41s - loss: 0.2558 - categorical_accuracy: 0.9234
36192/60000 [=================>............] - ETA: 41s - loss: 0.2556 - categorical_accuracy: 0.9235
36224/60000 [=================>............] - ETA: 41s - loss: 0.2556 - categorical_accuracy: 0.9235
36256/60000 [=================>............] - ETA: 41s - loss: 0.2555 - categorical_accuracy: 0.9235
36288/60000 [=================>............] - ETA: 41s - loss: 0.2555 - categorical_accuracy: 0.9235
36320/60000 [=================>............] - ETA: 41s - loss: 0.2554 - categorical_accuracy: 0.9236
36352/60000 [=================>............] - ETA: 41s - loss: 0.2554 - categorical_accuracy: 0.9236
36384/60000 [=================>............] - ETA: 41s - loss: 0.2552 - categorical_accuracy: 0.9236
36416/60000 [=================>............] - ETA: 41s - loss: 0.2552 - categorical_accuracy: 0.9236
36448/60000 [=================>............] - ETA: 41s - loss: 0.2552 - categorical_accuracy: 0.9236
36480/60000 [=================>............] - ETA: 41s - loss: 0.2551 - categorical_accuracy: 0.9237
36512/60000 [=================>............] - ETA: 41s - loss: 0.2551 - categorical_accuracy: 0.9237
36544/60000 [=================>............] - ETA: 41s - loss: 0.2549 - categorical_accuracy: 0.9237
36576/60000 [=================>............] - ETA: 40s - loss: 0.2547 - categorical_accuracy: 0.9238
36608/60000 [=================>............] - ETA: 40s - loss: 0.2545 - categorical_accuracy: 0.9238
36640/60000 [=================>............] - ETA: 40s - loss: 0.2543 - categorical_accuracy: 0.9239
36672/60000 [=================>............] - ETA: 40s - loss: 0.2542 - categorical_accuracy: 0.9239
36704/60000 [=================>............] - ETA: 40s - loss: 0.2543 - categorical_accuracy: 0.9239
36736/60000 [=================>............] - ETA: 40s - loss: 0.2541 - categorical_accuracy: 0.9240
36768/60000 [=================>............] - ETA: 40s - loss: 0.2541 - categorical_accuracy: 0.9240
36800/60000 [=================>............] - ETA: 40s - loss: 0.2540 - categorical_accuracy: 0.9240
36832/60000 [=================>............] - ETA: 40s - loss: 0.2542 - categorical_accuracy: 0.9240
36864/60000 [=================>............] - ETA: 40s - loss: 0.2540 - categorical_accuracy: 0.9240
36896/60000 [=================>............] - ETA: 40s - loss: 0.2538 - categorical_accuracy: 0.9241
36928/60000 [=================>............] - ETA: 40s - loss: 0.2539 - categorical_accuracy: 0.9241
36960/60000 [=================>............] - ETA: 40s - loss: 0.2537 - categorical_accuracy: 0.9242
36992/60000 [=================>............] - ETA: 40s - loss: 0.2535 - categorical_accuracy: 0.9242
37024/60000 [=================>............] - ETA: 40s - loss: 0.2535 - categorical_accuracy: 0.9242
37056/60000 [=================>............] - ETA: 40s - loss: 0.2533 - categorical_accuracy: 0.9242
37088/60000 [=================>............] - ETA: 40s - loss: 0.2532 - categorical_accuracy: 0.9243
37120/60000 [=================>............] - ETA: 40s - loss: 0.2532 - categorical_accuracy: 0.9243
37152/60000 [=================>............] - ETA: 39s - loss: 0.2530 - categorical_accuracy: 0.9243
37184/60000 [=================>............] - ETA: 39s - loss: 0.2529 - categorical_accuracy: 0.9244
37216/60000 [=================>............] - ETA: 39s - loss: 0.2527 - categorical_accuracy: 0.9244
37248/60000 [=================>............] - ETA: 39s - loss: 0.2525 - categorical_accuracy: 0.9245
37280/60000 [=================>............] - ETA: 39s - loss: 0.2524 - categorical_accuracy: 0.9245
37312/60000 [=================>............] - ETA: 39s - loss: 0.2523 - categorical_accuracy: 0.9246
37344/60000 [=================>............] - ETA: 39s - loss: 0.2521 - categorical_accuracy: 0.9246
37376/60000 [=================>............] - ETA: 39s - loss: 0.2520 - categorical_accuracy: 0.9246
37408/60000 [=================>............] - ETA: 39s - loss: 0.2518 - categorical_accuracy: 0.9247
37440/60000 [=================>............] - ETA: 39s - loss: 0.2518 - categorical_accuracy: 0.9247
37472/60000 [=================>............] - ETA: 39s - loss: 0.2516 - categorical_accuracy: 0.9248
37504/60000 [=================>............] - ETA: 39s - loss: 0.2515 - categorical_accuracy: 0.9248
37536/60000 [=================>............] - ETA: 39s - loss: 0.2514 - categorical_accuracy: 0.9248
37568/60000 [=================>............] - ETA: 39s - loss: 0.2513 - categorical_accuracy: 0.9249
37600/60000 [=================>............] - ETA: 39s - loss: 0.2511 - categorical_accuracy: 0.9249
37632/60000 [=================>............] - ETA: 39s - loss: 0.2511 - categorical_accuracy: 0.9249
37664/60000 [=================>............] - ETA: 39s - loss: 0.2509 - categorical_accuracy: 0.9249
37696/60000 [=================>............] - ETA: 39s - loss: 0.2507 - categorical_accuracy: 0.9250
37728/60000 [=================>............] - ETA: 38s - loss: 0.2506 - categorical_accuracy: 0.9250
37760/60000 [=================>............] - ETA: 38s - loss: 0.2504 - categorical_accuracy: 0.9251
37792/60000 [=================>............] - ETA: 38s - loss: 0.2502 - categorical_accuracy: 0.9251
37824/60000 [=================>............] - ETA: 38s - loss: 0.2500 - categorical_accuracy: 0.9252
37856/60000 [=================>............] - ETA: 38s - loss: 0.2498 - categorical_accuracy: 0.9252
37888/60000 [=================>............] - ETA: 38s - loss: 0.2497 - categorical_accuracy: 0.9253
37920/60000 [=================>............] - ETA: 38s - loss: 0.2496 - categorical_accuracy: 0.9253
37952/60000 [=================>............] - ETA: 38s - loss: 0.2495 - categorical_accuracy: 0.9253
37984/60000 [=================>............] - ETA: 38s - loss: 0.2494 - categorical_accuracy: 0.9253
38016/60000 [==================>...........] - ETA: 38s - loss: 0.2492 - categorical_accuracy: 0.9253
38048/60000 [==================>...........] - ETA: 38s - loss: 0.2492 - categorical_accuracy: 0.9254
38080/60000 [==================>...........] - ETA: 38s - loss: 0.2491 - categorical_accuracy: 0.9254
38112/60000 [==================>...........] - ETA: 38s - loss: 0.2491 - categorical_accuracy: 0.9254
38144/60000 [==================>...........] - ETA: 38s - loss: 0.2490 - categorical_accuracy: 0.9254
38176/60000 [==================>...........] - ETA: 38s - loss: 0.2488 - categorical_accuracy: 0.9255
38208/60000 [==================>...........] - ETA: 38s - loss: 0.2487 - categorical_accuracy: 0.9255
38240/60000 [==================>...........] - ETA: 38s - loss: 0.2485 - categorical_accuracy: 0.9255
38272/60000 [==================>...........] - ETA: 38s - loss: 0.2485 - categorical_accuracy: 0.9255
38304/60000 [==================>...........] - ETA: 37s - loss: 0.2485 - categorical_accuracy: 0.9255
38336/60000 [==================>...........] - ETA: 37s - loss: 0.2483 - categorical_accuracy: 0.9256
38368/60000 [==================>...........] - ETA: 37s - loss: 0.2482 - categorical_accuracy: 0.9256
38400/60000 [==================>...........] - ETA: 37s - loss: 0.2483 - categorical_accuracy: 0.9256
38432/60000 [==================>...........] - ETA: 37s - loss: 0.2482 - categorical_accuracy: 0.9256
38464/60000 [==================>...........] - ETA: 37s - loss: 0.2481 - categorical_accuracy: 0.9256
38496/60000 [==================>...........] - ETA: 37s - loss: 0.2480 - categorical_accuracy: 0.9257
38528/60000 [==================>...........] - ETA: 37s - loss: 0.2478 - categorical_accuracy: 0.9257
38560/60000 [==================>...........] - ETA: 37s - loss: 0.2477 - categorical_accuracy: 0.9258
38592/60000 [==================>...........] - ETA: 37s - loss: 0.2476 - categorical_accuracy: 0.9258
38624/60000 [==================>...........] - ETA: 37s - loss: 0.2475 - categorical_accuracy: 0.9258
38656/60000 [==================>...........] - ETA: 37s - loss: 0.2474 - categorical_accuracy: 0.9258
38688/60000 [==================>...........] - ETA: 37s - loss: 0.2474 - categorical_accuracy: 0.9258
38720/60000 [==================>...........] - ETA: 37s - loss: 0.2473 - categorical_accuracy: 0.9259
38752/60000 [==================>...........] - ETA: 37s - loss: 0.2471 - categorical_accuracy: 0.9259
38784/60000 [==================>...........] - ETA: 37s - loss: 0.2471 - categorical_accuracy: 0.9259
38816/60000 [==================>...........] - ETA: 37s - loss: 0.2469 - categorical_accuracy: 0.9259
38848/60000 [==================>...........] - ETA: 37s - loss: 0.2468 - categorical_accuracy: 0.9260
38880/60000 [==================>...........] - ETA: 36s - loss: 0.2466 - categorical_accuracy: 0.9261
38912/60000 [==================>...........] - ETA: 36s - loss: 0.2464 - categorical_accuracy: 0.9261
38944/60000 [==================>...........] - ETA: 36s - loss: 0.2463 - categorical_accuracy: 0.9262
38976/60000 [==================>...........] - ETA: 36s - loss: 0.2462 - categorical_accuracy: 0.9262
39008/60000 [==================>...........] - ETA: 36s - loss: 0.2461 - categorical_accuracy: 0.9262
39040/60000 [==================>...........] - ETA: 36s - loss: 0.2460 - categorical_accuracy: 0.9262
39072/60000 [==================>...........] - ETA: 36s - loss: 0.2459 - categorical_accuracy: 0.9263
39104/60000 [==================>...........] - ETA: 36s - loss: 0.2457 - categorical_accuracy: 0.9263
39136/60000 [==================>...........] - ETA: 36s - loss: 0.2457 - categorical_accuracy: 0.9264
39168/60000 [==================>...........] - ETA: 36s - loss: 0.2455 - categorical_accuracy: 0.9264
39200/60000 [==================>...........] - ETA: 36s - loss: 0.2454 - categorical_accuracy: 0.9264
39232/60000 [==================>...........] - ETA: 36s - loss: 0.2454 - categorical_accuracy: 0.9264
39264/60000 [==================>...........] - ETA: 36s - loss: 0.2453 - categorical_accuracy: 0.9264
39296/60000 [==================>...........] - ETA: 36s - loss: 0.2451 - categorical_accuracy: 0.9265
39328/60000 [==================>...........] - ETA: 36s - loss: 0.2449 - categorical_accuracy: 0.9265
39360/60000 [==================>...........] - ETA: 36s - loss: 0.2448 - categorical_accuracy: 0.9266
39392/60000 [==================>...........] - ETA: 36s - loss: 0.2447 - categorical_accuracy: 0.9266
39424/60000 [==================>...........] - ETA: 36s - loss: 0.2446 - categorical_accuracy: 0.9266
39456/60000 [==================>...........] - ETA: 35s - loss: 0.2445 - categorical_accuracy: 0.9267
39488/60000 [==================>...........] - ETA: 35s - loss: 0.2443 - categorical_accuracy: 0.9268
39520/60000 [==================>...........] - ETA: 35s - loss: 0.2441 - categorical_accuracy: 0.9268
39552/60000 [==================>...........] - ETA: 35s - loss: 0.2441 - categorical_accuracy: 0.9268
39584/60000 [==================>...........] - ETA: 35s - loss: 0.2440 - categorical_accuracy: 0.9268
39616/60000 [==================>...........] - ETA: 35s - loss: 0.2441 - categorical_accuracy: 0.9268
39648/60000 [==================>...........] - ETA: 35s - loss: 0.2440 - categorical_accuracy: 0.9268
39680/60000 [==================>...........] - ETA: 35s - loss: 0.2439 - categorical_accuracy: 0.9269
39712/60000 [==================>...........] - ETA: 35s - loss: 0.2437 - categorical_accuracy: 0.9269
39744/60000 [==================>...........] - ETA: 35s - loss: 0.2435 - categorical_accuracy: 0.9270
39776/60000 [==================>...........] - ETA: 35s - loss: 0.2434 - categorical_accuracy: 0.9270
39808/60000 [==================>...........] - ETA: 35s - loss: 0.2434 - categorical_accuracy: 0.9270
39840/60000 [==================>...........] - ETA: 35s - loss: 0.2435 - categorical_accuracy: 0.9270
39872/60000 [==================>...........] - ETA: 35s - loss: 0.2434 - categorical_accuracy: 0.9270
39904/60000 [==================>...........] - ETA: 35s - loss: 0.2433 - categorical_accuracy: 0.9271
39936/60000 [==================>...........] - ETA: 35s - loss: 0.2431 - categorical_accuracy: 0.9271
39968/60000 [==================>...........] - ETA: 35s - loss: 0.2430 - categorical_accuracy: 0.9271
40000/60000 [===================>..........] - ETA: 35s - loss: 0.2429 - categorical_accuracy: 0.9272
40032/60000 [===================>..........] - ETA: 34s - loss: 0.2430 - categorical_accuracy: 0.9272
40064/60000 [===================>..........] - ETA: 34s - loss: 0.2429 - categorical_accuracy: 0.9272
40096/60000 [===================>..........] - ETA: 34s - loss: 0.2428 - categorical_accuracy: 0.9272
40128/60000 [===================>..........] - ETA: 34s - loss: 0.2427 - categorical_accuracy: 0.9272
40160/60000 [===================>..........] - ETA: 34s - loss: 0.2426 - categorical_accuracy: 0.9273
40192/60000 [===================>..........] - ETA: 34s - loss: 0.2425 - categorical_accuracy: 0.9273
40224/60000 [===================>..........] - ETA: 34s - loss: 0.2423 - categorical_accuracy: 0.9273
40256/60000 [===================>..........] - ETA: 34s - loss: 0.2423 - categorical_accuracy: 0.9273
40288/60000 [===================>..........] - ETA: 34s - loss: 0.2423 - categorical_accuracy: 0.9273
40320/60000 [===================>..........] - ETA: 34s - loss: 0.2422 - categorical_accuracy: 0.9274
40352/60000 [===================>..........] - ETA: 34s - loss: 0.2422 - categorical_accuracy: 0.9274
40384/60000 [===================>..........] - ETA: 34s - loss: 0.2420 - categorical_accuracy: 0.9274
40416/60000 [===================>..........] - ETA: 34s - loss: 0.2419 - categorical_accuracy: 0.9274
40448/60000 [===================>..........] - ETA: 34s - loss: 0.2419 - categorical_accuracy: 0.9274
40480/60000 [===================>..........] - ETA: 34s - loss: 0.2419 - categorical_accuracy: 0.9274
40512/60000 [===================>..........] - ETA: 34s - loss: 0.2417 - categorical_accuracy: 0.9275
40544/60000 [===================>..........] - ETA: 34s - loss: 0.2416 - categorical_accuracy: 0.9275
40576/60000 [===================>..........] - ETA: 34s - loss: 0.2415 - categorical_accuracy: 0.9275
40608/60000 [===================>..........] - ETA: 33s - loss: 0.2414 - categorical_accuracy: 0.9276
40640/60000 [===================>..........] - ETA: 33s - loss: 0.2413 - categorical_accuracy: 0.9276
40672/60000 [===================>..........] - ETA: 33s - loss: 0.2412 - categorical_accuracy: 0.9277
40704/60000 [===================>..........] - ETA: 33s - loss: 0.2411 - categorical_accuracy: 0.9277
40736/60000 [===================>..........] - ETA: 33s - loss: 0.2410 - categorical_accuracy: 0.9277
40768/60000 [===================>..........] - ETA: 33s - loss: 0.2411 - categorical_accuracy: 0.9277
40800/60000 [===================>..........] - ETA: 33s - loss: 0.2411 - categorical_accuracy: 0.9277
40832/60000 [===================>..........] - ETA: 33s - loss: 0.2409 - categorical_accuracy: 0.9277
40864/60000 [===================>..........] - ETA: 33s - loss: 0.2408 - categorical_accuracy: 0.9278
40896/60000 [===================>..........] - ETA: 33s - loss: 0.2407 - categorical_accuracy: 0.9278
40928/60000 [===================>..........] - ETA: 33s - loss: 0.2405 - categorical_accuracy: 0.9278
40960/60000 [===================>..........] - ETA: 33s - loss: 0.2404 - categorical_accuracy: 0.9279
40992/60000 [===================>..........] - ETA: 33s - loss: 0.2406 - categorical_accuracy: 0.9278
41024/60000 [===================>..........] - ETA: 33s - loss: 0.2405 - categorical_accuracy: 0.9278
41056/60000 [===================>..........] - ETA: 33s - loss: 0.2407 - categorical_accuracy: 0.9278
41088/60000 [===================>..........] - ETA: 33s - loss: 0.2406 - categorical_accuracy: 0.9278
41120/60000 [===================>..........] - ETA: 33s - loss: 0.2404 - categorical_accuracy: 0.9278
41152/60000 [===================>..........] - ETA: 33s - loss: 0.2403 - categorical_accuracy: 0.9279
41184/60000 [===================>..........] - ETA: 32s - loss: 0.2402 - categorical_accuracy: 0.9279
41216/60000 [===================>..........] - ETA: 32s - loss: 0.2400 - categorical_accuracy: 0.9280
41248/60000 [===================>..........] - ETA: 32s - loss: 0.2400 - categorical_accuracy: 0.9280
41280/60000 [===================>..........] - ETA: 32s - loss: 0.2399 - categorical_accuracy: 0.9280
41312/60000 [===================>..........] - ETA: 32s - loss: 0.2397 - categorical_accuracy: 0.9281
41344/60000 [===================>..........] - ETA: 32s - loss: 0.2398 - categorical_accuracy: 0.9280
41376/60000 [===================>..........] - ETA: 32s - loss: 0.2397 - categorical_accuracy: 0.9281
41408/60000 [===================>..........] - ETA: 32s - loss: 0.2396 - categorical_accuracy: 0.9281
41440/60000 [===================>..........] - ETA: 32s - loss: 0.2395 - categorical_accuracy: 0.9281
41472/60000 [===================>..........] - ETA: 32s - loss: 0.2394 - categorical_accuracy: 0.9282
41504/60000 [===================>..........] - ETA: 32s - loss: 0.2393 - categorical_accuracy: 0.9282
41536/60000 [===================>..........] - ETA: 32s - loss: 0.2392 - categorical_accuracy: 0.9282
41568/60000 [===================>..........] - ETA: 32s - loss: 0.2391 - categorical_accuracy: 0.9283
41600/60000 [===================>..........] - ETA: 32s - loss: 0.2390 - categorical_accuracy: 0.9283
41632/60000 [===================>..........] - ETA: 32s - loss: 0.2389 - categorical_accuracy: 0.9283
41664/60000 [===================>..........] - ETA: 32s - loss: 0.2388 - categorical_accuracy: 0.9284
41696/60000 [===================>..........] - ETA: 32s - loss: 0.2386 - categorical_accuracy: 0.9284
41728/60000 [===================>..........] - ETA: 31s - loss: 0.2386 - categorical_accuracy: 0.9284
41760/60000 [===================>..........] - ETA: 31s - loss: 0.2385 - categorical_accuracy: 0.9285
41792/60000 [===================>..........] - ETA: 31s - loss: 0.2384 - categorical_accuracy: 0.9285
41824/60000 [===================>..........] - ETA: 31s - loss: 0.2383 - categorical_accuracy: 0.9285
41856/60000 [===================>..........] - ETA: 31s - loss: 0.2382 - categorical_accuracy: 0.9285
41888/60000 [===================>..........] - ETA: 31s - loss: 0.2381 - categorical_accuracy: 0.9286
41920/60000 [===================>..........] - ETA: 31s - loss: 0.2380 - categorical_accuracy: 0.9286
41952/60000 [===================>..........] - ETA: 31s - loss: 0.2378 - categorical_accuracy: 0.9287
41984/60000 [===================>..........] - ETA: 31s - loss: 0.2377 - categorical_accuracy: 0.9287
42016/60000 [====================>.........] - ETA: 31s - loss: 0.2375 - categorical_accuracy: 0.9287
42048/60000 [====================>.........] - ETA: 31s - loss: 0.2375 - categorical_accuracy: 0.9287
42080/60000 [====================>.........] - ETA: 31s - loss: 0.2373 - categorical_accuracy: 0.9288
42112/60000 [====================>.........] - ETA: 31s - loss: 0.2372 - categorical_accuracy: 0.9288
42144/60000 [====================>.........] - ETA: 31s - loss: 0.2370 - categorical_accuracy: 0.9289
42176/60000 [====================>.........] - ETA: 31s - loss: 0.2370 - categorical_accuracy: 0.9289
42208/60000 [====================>.........] - ETA: 31s - loss: 0.2369 - categorical_accuracy: 0.9289
42240/60000 [====================>.........] - ETA: 31s - loss: 0.2367 - categorical_accuracy: 0.9290
42272/60000 [====================>.........] - ETA: 31s - loss: 0.2366 - categorical_accuracy: 0.9290
42304/60000 [====================>.........] - ETA: 30s - loss: 0.2365 - categorical_accuracy: 0.9290
42336/60000 [====================>.........] - ETA: 30s - loss: 0.2364 - categorical_accuracy: 0.9291
42368/60000 [====================>.........] - ETA: 30s - loss: 0.2363 - categorical_accuracy: 0.9291
42400/60000 [====================>.........] - ETA: 30s - loss: 0.2363 - categorical_accuracy: 0.9291
42432/60000 [====================>.........] - ETA: 30s - loss: 0.2362 - categorical_accuracy: 0.9292
42464/60000 [====================>.........] - ETA: 30s - loss: 0.2361 - categorical_accuracy: 0.9292
42496/60000 [====================>.........] - ETA: 30s - loss: 0.2360 - categorical_accuracy: 0.9292
42528/60000 [====================>.........] - ETA: 30s - loss: 0.2358 - categorical_accuracy: 0.9293
42560/60000 [====================>.........] - ETA: 30s - loss: 0.2357 - categorical_accuracy: 0.9293
42592/60000 [====================>.........] - ETA: 30s - loss: 0.2357 - categorical_accuracy: 0.9293
42624/60000 [====================>.........] - ETA: 30s - loss: 0.2356 - categorical_accuracy: 0.9293
42656/60000 [====================>.........] - ETA: 30s - loss: 0.2355 - categorical_accuracy: 0.9294
42688/60000 [====================>.........] - ETA: 30s - loss: 0.2353 - categorical_accuracy: 0.9294
42720/60000 [====================>.........] - ETA: 30s - loss: 0.2352 - categorical_accuracy: 0.9295
42752/60000 [====================>.........] - ETA: 30s - loss: 0.2352 - categorical_accuracy: 0.9295
42784/60000 [====================>.........] - ETA: 30s - loss: 0.2350 - categorical_accuracy: 0.9296
42816/60000 [====================>.........] - ETA: 30s - loss: 0.2349 - categorical_accuracy: 0.9296
42848/60000 [====================>.........] - ETA: 30s - loss: 0.2347 - categorical_accuracy: 0.9297
42880/60000 [====================>.........] - ETA: 29s - loss: 0.2346 - categorical_accuracy: 0.9297
42912/60000 [====================>.........] - ETA: 29s - loss: 0.2346 - categorical_accuracy: 0.9296
42944/60000 [====================>.........] - ETA: 29s - loss: 0.2346 - categorical_accuracy: 0.9297
42976/60000 [====================>.........] - ETA: 29s - loss: 0.2345 - categorical_accuracy: 0.9297
43008/60000 [====================>.........] - ETA: 29s - loss: 0.2344 - categorical_accuracy: 0.9297
43040/60000 [====================>.........] - ETA: 29s - loss: 0.2343 - categorical_accuracy: 0.9297
43072/60000 [====================>.........] - ETA: 29s - loss: 0.2341 - categorical_accuracy: 0.9298
43104/60000 [====================>.........] - ETA: 29s - loss: 0.2340 - categorical_accuracy: 0.9298
43136/60000 [====================>.........] - ETA: 29s - loss: 0.2340 - categorical_accuracy: 0.9298
43168/60000 [====================>.........] - ETA: 29s - loss: 0.2340 - categorical_accuracy: 0.9298
43200/60000 [====================>.........] - ETA: 29s - loss: 0.2338 - categorical_accuracy: 0.9299
43232/60000 [====================>.........] - ETA: 29s - loss: 0.2337 - categorical_accuracy: 0.9299
43264/60000 [====================>.........] - ETA: 29s - loss: 0.2336 - categorical_accuracy: 0.9299
43296/60000 [====================>.........] - ETA: 29s - loss: 0.2336 - categorical_accuracy: 0.9299
43328/60000 [====================>.........] - ETA: 29s - loss: 0.2336 - categorical_accuracy: 0.9300
43360/60000 [====================>.........] - ETA: 29s - loss: 0.2334 - categorical_accuracy: 0.9300
43392/60000 [====================>.........] - ETA: 29s - loss: 0.2333 - categorical_accuracy: 0.9301
43424/60000 [====================>.........] - ETA: 29s - loss: 0.2333 - categorical_accuracy: 0.9301
43456/60000 [====================>.........] - ETA: 28s - loss: 0.2333 - categorical_accuracy: 0.9301
43488/60000 [====================>.........] - ETA: 28s - loss: 0.2333 - categorical_accuracy: 0.9300
43520/60000 [====================>.........] - ETA: 28s - loss: 0.2332 - categorical_accuracy: 0.9301
43552/60000 [====================>.........] - ETA: 28s - loss: 0.2331 - categorical_accuracy: 0.9301
43584/60000 [====================>.........] - ETA: 28s - loss: 0.2330 - categorical_accuracy: 0.9301
43616/60000 [====================>.........] - ETA: 28s - loss: 0.2329 - categorical_accuracy: 0.9302
43648/60000 [====================>.........] - ETA: 28s - loss: 0.2327 - categorical_accuracy: 0.9302
43680/60000 [====================>.........] - ETA: 28s - loss: 0.2326 - categorical_accuracy: 0.9303
43712/60000 [====================>.........] - ETA: 28s - loss: 0.2324 - categorical_accuracy: 0.9303
43744/60000 [====================>.........] - ETA: 28s - loss: 0.2324 - categorical_accuracy: 0.9303
43776/60000 [====================>.........] - ETA: 28s - loss: 0.2323 - categorical_accuracy: 0.9304
43808/60000 [====================>.........] - ETA: 28s - loss: 0.2321 - categorical_accuracy: 0.9304
43840/60000 [====================>.........] - ETA: 28s - loss: 0.2320 - categorical_accuracy: 0.9304
43872/60000 [====================>.........] - ETA: 28s - loss: 0.2319 - categorical_accuracy: 0.9304
43904/60000 [====================>.........] - ETA: 28s - loss: 0.2318 - categorical_accuracy: 0.9305
43936/60000 [====================>.........] - ETA: 28s - loss: 0.2317 - categorical_accuracy: 0.9305
43968/60000 [====================>.........] - ETA: 28s - loss: 0.2316 - categorical_accuracy: 0.9305
44000/60000 [=====================>........] - ETA: 28s - loss: 0.2314 - categorical_accuracy: 0.9306
44032/60000 [=====================>........] - ETA: 27s - loss: 0.2313 - categorical_accuracy: 0.9306
44064/60000 [=====================>........] - ETA: 27s - loss: 0.2312 - categorical_accuracy: 0.9306
44096/60000 [=====================>........] - ETA: 27s - loss: 0.2311 - categorical_accuracy: 0.9307
44128/60000 [=====================>........] - ETA: 27s - loss: 0.2310 - categorical_accuracy: 0.9307
44160/60000 [=====================>........] - ETA: 27s - loss: 0.2310 - categorical_accuracy: 0.9307
44192/60000 [=====================>........] - ETA: 27s - loss: 0.2308 - categorical_accuracy: 0.9308
44224/60000 [=====================>........] - ETA: 27s - loss: 0.2308 - categorical_accuracy: 0.9308
44256/60000 [=====================>........] - ETA: 27s - loss: 0.2306 - categorical_accuracy: 0.9308
44288/60000 [=====================>........] - ETA: 27s - loss: 0.2306 - categorical_accuracy: 0.9308
44320/60000 [=====================>........] - ETA: 27s - loss: 0.2305 - categorical_accuracy: 0.9309
44352/60000 [=====================>........] - ETA: 27s - loss: 0.2304 - categorical_accuracy: 0.9309
44384/60000 [=====================>........] - ETA: 27s - loss: 0.2305 - categorical_accuracy: 0.9309
44416/60000 [=====================>........] - ETA: 27s - loss: 0.2304 - categorical_accuracy: 0.9309
44448/60000 [=====================>........] - ETA: 27s - loss: 0.2303 - categorical_accuracy: 0.9309
44480/60000 [=====================>........] - ETA: 27s - loss: 0.2302 - categorical_accuracy: 0.9310
44512/60000 [=====================>........] - ETA: 27s - loss: 0.2301 - categorical_accuracy: 0.9310
44544/60000 [=====================>........] - ETA: 27s - loss: 0.2300 - categorical_accuracy: 0.9310
44576/60000 [=====================>........] - ETA: 27s - loss: 0.2299 - categorical_accuracy: 0.9311
44608/60000 [=====================>........] - ETA: 26s - loss: 0.2298 - categorical_accuracy: 0.9311
44640/60000 [=====================>........] - ETA: 26s - loss: 0.2297 - categorical_accuracy: 0.9311
44672/60000 [=====================>........] - ETA: 26s - loss: 0.2296 - categorical_accuracy: 0.9311
44704/60000 [=====================>........] - ETA: 26s - loss: 0.2295 - categorical_accuracy: 0.9311
44736/60000 [=====================>........] - ETA: 26s - loss: 0.2294 - categorical_accuracy: 0.9312
44768/60000 [=====================>........] - ETA: 26s - loss: 0.2293 - categorical_accuracy: 0.9312
44800/60000 [=====================>........] - ETA: 26s - loss: 0.2291 - categorical_accuracy: 0.9313
44832/60000 [=====================>........] - ETA: 26s - loss: 0.2292 - categorical_accuracy: 0.9313
44864/60000 [=====================>........] - ETA: 26s - loss: 0.2291 - categorical_accuracy: 0.9313
44896/60000 [=====================>........] - ETA: 26s - loss: 0.2290 - categorical_accuracy: 0.9314
44928/60000 [=====================>........] - ETA: 26s - loss: 0.2288 - categorical_accuracy: 0.9314
44960/60000 [=====================>........] - ETA: 26s - loss: 0.2287 - categorical_accuracy: 0.9315
44992/60000 [=====================>........] - ETA: 26s - loss: 0.2286 - categorical_accuracy: 0.9315
45024/60000 [=====================>........] - ETA: 26s - loss: 0.2285 - categorical_accuracy: 0.9315
45056/60000 [=====================>........] - ETA: 26s - loss: 0.2283 - categorical_accuracy: 0.9316
45088/60000 [=====================>........] - ETA: 26s - loss: 0.2282 - categorical_accuracy: 0.9316
45120/60000 [=====================>........] - ETA: 26s - loss: 0.2281 - categorical_accuracy: 0.9317
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2280 - categorical_accuracy: 0.9317
45184/60000 [=====================>........] - ETA: 25s - loss: 0.2279 - categorical_accuracy: 0.9317
45216/60000 [=====================>........] - ETA: 25s - loss: 0.2279 - categorical_accuracy: 0.9318
45248/60000 [=====================>........] - ETA: 25s - loss: 0.2278 - categorical_accuracy: 0.9318
45280/60000 [=====================>........] - ETA: 25s - loss: 0.2276 - categorical_accuracy: 0.9319
45312/60000 [=====================>........] - ETA: 25s - loss: 0.2275 - categorical_accuracy: 0.9319
45344/60000 [=====================>........] - ETA: 25s - loss: 0.2274 - categorical_accuracy: 0.9319
45376/60000 [=====================>........] - ETA: 25s - loss: 0.2272 - categorical_accuracy: 0.9320
45408/60000 [=====================>........] - ETA: 25s - loss: 0.2272 - categorical_accuracy: 0.9320
45440/60000 [=====================>........] - ETA: 25s - loss: 0.2272 - categorical_accuracy: 0.9320
45472/60000 [=====================>........] - ETA: 25s - loss: 0.2270 - categorical_accuracy: 0.9321
45504/60000 [=====================>........] - ETA: 25s - loss: 0.2271 - categorical_accuracy: 0.9321
45536/60000 [=====================>........] - ETA: 25s - loss: 0.2270 - categorical_accuracy: 0.9321
45568/60000 [=====================>........] - ETA: 25s - loss: 0.2269 - categorical_accuracy: 0.9321
45600/60000 [=====================>........] - ETA: 25s - loss: 0.2268 - categorical_accuracy: 0.9322
45632/60000 [=====================>........] - ETA: 25s - loss: 0.2266 - categorical_accuracy: 0.9322
45664/60000 [=====================>........] - ETA: 25s - loss: 0.2267 - categorical_accuracy: 0.9322
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2266 - categorical_accuracy: 0.9322
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2264 - categorical_accuracy: 0.9323
45760/60000 [=====================>........] - ETA: 24s - loss: 0.2263 - categorical_accuracy: 0.9323
45792/60000 [=====================>........] - ETA: 24s - loss: 0.2262 - categorical_accuracy: 0.9324
45824/60000 [=====================>........] - ETA: 24s - loss: 0.2261 - categorical_accuracy: 0.9324
45856/60000 [=====================>........] - ETA: 24s - loss: 0.2260 - categorical_accuracy: 0.9324
45888/60000 [=====================>........] - ETA: 24s - loss: 0.2260 - categorical_accuracy: 0.9325
45920/60000 [=====================>........] - ETA: 24s - loss: 0.2261 - categorical_accuracy: 0.9324
45952/60000 [=====================>........] - ETA: 24s - loss: 0.2260 - categorical_accuracy: 0.9325
45984/60000 [=====================>........] - ETA: 24s - loss: 0.2259 - categorical_accuracy: 0.9325
46016/60000 [======================>.......] - ETA: 24s - loss: 0.2258 - categorical_accuracy: 0.9325
46048/60000 [======================>.......] - ETA: 24s - loss: 0.2257 - categorical_accuracy: 0.9325
46080/60000 [======================>.......] - ETA: 24s - loss: 0.2256 - categorical_accuracy: 0.9326
46112/60000 [======================>.......] - ETA: 24s - loss: 0.2255 - categorical_accuracy: 0.9326
46144/60000 [======================>.......] - ETA: 24s - loss: 0.2254 - categorical_accuracy: 0.9326
46176/60000 [======================>.......] - ETA: 24s - loss: 0.2254 - categorical_accuracy: 0.9326
46208/60000 [======================>.......] - ETA: 24s - loss: 0.2253 - categorical_accuracy: 0.9327
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2253 - categorical_accuracy: 0.9327
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2251 - categorical_accuracy: 0.9327
46304/60000 [======================>.......] - ETA: 23s - loss: 0.2250 - categorical_accuracy: 0.9327
46336/60000 [======================>.......] - ETA: 23s - loss: 0.2249 - categorical_accuracy: 0.9328
46368/60000 [======================>.......] - ETA: 23s - loss: 0.2248 - categorical_accuracy: 0.9328
46400/60000 [======================>.......] - ETA: 23s - loss: 0.2246 - categorical_accuracy: 0.9328
46432/60000 [======================>.......] - ETA: 23s - loss: 0.2246 - categorical_accuracy: 0.9328
46464/60000 [======================>.......] - ETA: 23s - loss: 0.2245 - categorical_accuracy: 0.9329
46496/60000 [======================>.......] - ETA: 23s - loss: 0.2244 - categorical_accuracy: 0.9329
46528/60000 [======================>.......] - ETA: 23s - loss: 0.2243 - categorical_accuracy: 0.9329
46560/60000 [======================>.......] - ETA: 23s - loss: 0.2242 - categorical_accuracy: 0.9330
46592/60000 [======================>.......] - ETA: 23s - loss: 0.2241 - categorical_accuracy: 0.9330
46624/60000 [======================>.......] - ETA: 23s - loss: 0.2241 - categorical_accuracy: 0.9330
46656/60000 [======================>.......] - ETA: 23s - loss: 0.2240 - categorical_accuracy: 0.9330
46688/60000 [======================>.......] - ETA: 23s - loss: 0.2242 - categorical_accuracy: 0.9330
46720/60000 [======================>.......] - ETA: 23s - loss: 0.2242 - categorical_accuracy: 0.9329
46752/60000 [======================>.......] - ETA: 23s - loss: 0.2242 - categorical_accuracy: 0.9329
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2241 - categorical_accuracy: 0.9329
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2241 - categorical_accuracy: 0.9329
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2240 - categorical_accuracy: 0.9329
46880/60000 [======================>.......] - ETA: 22s - loss: 0.2239 - categorical_accuracy: 0.9330
46912/60000 [======================>.......] - ETA: 22s - loss: 0.2239 - categorical_accuracy: 0.9330
46944/60000 [======================>.......] - ETA: 22s - loss: 0.2237 - categorical_accuracy: 0.9330
46976/60000 [======================>.......] - ETA: 22s - loss: 0.2236 - categorical_accuracy: 0.9330
47008/60000 [======================>.......] - ETA: 22s - loss: 0.2235 - categorical_accuracy: 0.9331
47040/60000 [======================>.......] - ETA: 22s - loss: 0.2235 - categorical_accuracy: 0.9331
47072/60000 [======================>.......] - ETA: 22s - loss: 0.2233 - categorical_accuracy: 0.9331
47104/60000 [======================>.......] - ETA: 22s - loss: 0.2233 - categorical_accuracy: 0.9331
47136/60000 [======================>.......] - ETA: 22s - loss: 0.2232 - categorical_accuracy: 0.9331
47168/60000 [======================>.......] - ETA: 22s - loss: 0.2230 - categorical_accuracy: 0.9332
47200/60000 [======================>.......] - ETA: 22s - loss: 0.2230 - categorical_accuracy: 0.9332
47232/60000 [======================>.......] - ETA: 22s - loss: 0.2229 - categorical_accuracy: 0.9332
47264/60000 [======================>.......] - ETA: 22s - loss: 0.2229 - categorical_accuracy: 0.9332
47296/60000 [======================>.......] - ETA: 22s - loss: 0.2229 - categorical_accuracy: 0.9332
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2228 - categorical_accuracy: 0.9332
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2227 - categorical_accuracy: 0.9332
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2226 - categorical_accuracy: 0.9333
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2225 - categorical_accuracy: 0.9333
47456/60000 [======================>.......] - ETA: 21s - loss: 0.2224 - categorical_accuracy: 0.9333
47488/60000 [======================>.......] - ETA: 21s - loss: 0.2222 - categorical_accuracy: 0.9334
47520/60000 [======================>.......] - ETA: 21s - loss: 0.2221 - categorical_accuracy: 0.9334
47552/60000 [======================>.......] - ETA: 21s - loss: 0.2220 - categorical_accuracy: 0.9335
47584/60000 [======================>.......] - ETA: 21s - loss: 0.2219 - categorical_accuracy: 0.9335
47616/60000 [======================>.......] - ETA: 21s - loss: 0.2220 - categorical_accuracy: 0.9335
47648/60000 [======================>.......] - ETA: 21s - loss: 0.2219 - categorical_accuracy: 0.9336
47680/60000 [======================>.......] - ETA: 21s - loss: 0.2218 - categorical_accuracy: 0.9336
47712/60000 [======================>.......] - ETA: 21s - loss: 0.2217 - categorical_accuracy: 0.9336
47744/60000 [======================>.......] - ETA: 21s - loss: 0.2216 - categorical_accuracy: 0.9336
47776/60000 [======================>.......] - ETA: 21s - loss: 0.2216 - categorical_accuracy: 0.9336
47808/60000 [======================>.......] - ETA: 21s - loss: 0.2216 - categorical_accuracy: 0.9336
47840/60000 [======================>.......] - ETA: 21s - loss: 0.2215 - categorical_accuracy: 0.9336
47872/60000 [======================>.......] - ETA: 21s - loss: 0.2215 - categorical_accuracy: 0.9336
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2213 - categorical_accuracy: 0.9337
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2212 - categorical_accuracy: 0.9337
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2211 - categorical_accuracy: 0.9337
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2210 - categorical_accuracy: 0.9338
48032/60000 [=======================>......] - ETA: 20s - loss: 0.2211 - categorical_accuracy: 0.9338
48064/60000 [=======================>......] - ETA: 20s - loss: 0.2209 - categorical_accuracy: 0.9338
48096/60000 [=======================>......] - ETA: 20s - loss: 0.2208 - categorical_accuracy: 0.9339
48128/60000 [=======================>......] - ETA: 20s - loss: 0.2209 - categorical_accuracy: 0.9339
48160/60000 [=======================>......] - ETA: 20s - loss: 0.2208 - categorical_accuracy: 0.9339
48192/60000 [=======================>......] - ETA: 20s - loss: 0.2206 - categorical_accuracy: 0.9339
48224/60000 [=======================>......] - ETA: 20s - loss: 0.2205 - categorical_accuracy: 0.9340
48256/60000 [=======================>......] - ETA: 20s - loss: 0.2205 - categorical_accuracy: 0.9340
48288/60000 [=======================>......] - ETA: 20s - loss: 0.2204 - categorical_accuracy: 0.9340
48320/60000 [=======================>......] - ETA: 20s - loss: 0.2203 - categorical_accuracy: 0.9341
48352/60000 [=======================>......] - ETA: 20s - loss: 0.2202 - categorical_accuracy: 0.9341
48384/60000 [=======================>......] - ETA: 20s - loss: 0.2201 - categorical_accuracy: 0.9341
48416/60000 [=======================>......] - ETA: 20s - loss: 0.2200 - categorical_accuracy: 0.9342
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2198 - categorical_accuracy: 0.9342
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2197 - categorical_accuracy: 0.9342
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2196 - categorical_accuracy: 0.9343
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2194 - categorical_accuracy: 0.9343
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2193 - categorical_accuracy: 0.9344
48608/60000 [=======================>......] - ETA: 19s - loss: 0.2192 - categorical_accuracy: 0.9344
48640/60000 [=======================>......] - ETA: 19s - loss: 0.2191 - categorical_accuracy: 0.9345
48672/60000 [=======================>......] - ETA: 19s - loss: 0.2190 - categorical_accuracy: 0.9345
48704/60000 [=======================>......] - ETA: 19s - loss: 0.2190 - categorical_accuracy: 0.9345
48736/60000 [=======================>......] - ETA: 19s - loss: 0.2188 - categorical_accuracy: 0.9345
48768/60000 [=======================>......] - ETA: 19s - loss: 0.2188 - categorical_accuracy: 0.9345
48800/60000 [=======================>......] - ETA: 19s - loss: 0.2186 - categorical_accuracy: 0.9346
48832/60000 [=======================>......] - ETA: 19s - loss: 0.2185 - categorical_accuracy: 0.9346
48864/60000 [=======================>......] - ETA: 19s - loss: 0.2184 - categorical_accuracy: 0.9346
48896/60000 [=======================>......] - ETA: 19s - loss: 0.2183 - categorical_accuracy: 0.9347
48928/60000 [=======================>......] - ETA: 19s - loss: 0.2182 - categorical_accuracy: 0.9347
48960/60000 [=======================>......] - ETA: 19s - loss: 0.2181 - categorical_accuracy: 0.9347
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2180 - categorical_accuracy: 0.9348
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2179 - categorical_accuracy: 0.9348
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2178 - categorical_accuracy: 0.9348
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2177 - categorical_accuracy: 0.9349
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2175 - categorical_accuracy: 0.9349
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2174 - categorical_accuracy: 0.9349
49184/60000 [=======================>......] - ETA: 18s - loss: 0.2174 - categorical_accuracy: 0.9349
49216/60000 [=======================>......] - ETA: 18s - loss: 0.2173 - categorical_accuracy: 0.9350
49248/60000 [=======================>......] - ETA: 18s - loss: 0.2172 - categorical_accuracy: 0.9350
49280/60000 [=======================>......] - ETA: 18s - loss: 0.2171 - categorical_accuracy: 0.9350
49312/60000 [=======================>......] - ETA: 18s - loss: 0.2169 - categorical_accuracy: 0.9351
49344/60000 [=======================>......] - ETA: 18s - loss: 0.2168 - categorical_accuracy: 0.9351
49376/60000 [=======================>......] - ETA: 18s - loss: 0.2168 - categorical_accuracy: 0.9351
49408/60000 [=======================>......] - ETA: 18s - loss: 0.2167 - categorical_accuracy: 0.9352
49440/60000 [=======================>......] - ETA: 18s - loss: 0.2166 - categorical_accuracy: 0.9352
49472/60000 [=======================>......] - ETA: 18s - loss: 0.2165 - categorical_accuracy: 0.9352
49504/60000 [=======================>......] - ETA: 18s - loss: 0.2164 - categorical_accuracy: 0.9353
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2163 - categorical_accuracy: 0.9353
49568/60000 [=======================>......] - ETA: 18s - loss: 0.2164 - categorical_accuracy: 0.9353
49600/60000 [=======================>......] - ETA: 18s - loss: 0.2164 - categorical_accuracy: 0.9353
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2163 - categorical_accuracy: 0.9353
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2162 - categorical_accuracy: 0.9354
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2161 - categorical_accuracy: 0.9354
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2161 - categorical_accuracy: 0.9354
49760/60000 [=======================>......] - ETA: 17s - loss: 0.2160 - categorical_accuracy: 0.9354
49792/60000 [=======================>......] - ETA: 17s - loss: 0.2160 - categorical_accuracy: 0.9354
49824/60000 [=======================>......] - ETA: 17s - loss: 0.2159 - categorical_accuracy: 0.9354
49856/60000 [=======================>......] - ETA: 17s - loss: 0.2158 - categorical_accuracy: 0.9355
49888/60000 [=======================>......] - ETA: 17s - loss: 0.2157 - categorical_accuracy: 0.9355
49920/60000 [=======================>......] - ETA: 17s - loss: 0.2156 - categorical_accuracy: 0.9355
49952/60000 [=======================>......] - ETA: 17s - loss: 0.2155 - categorical_accuracy: 0.9356
49984/60000 [=======================>......] - ETA: 17s - loss: 0.2154 - categorical_accuracy: 0.9356
50016/60000 [========================>.....] - ETA: 17s - loss: 0.2154 - categorical_accuracy: 0.9356
50048/60000 [========================>.....] - ETA: 17s - loss: 0.2155 - categorical_accuracy: 0.9356
50080/60000 [========================>.....] - ETA: 17s - loss: 0.2153 - categorical_accuracy: 0.9356
50112/60000 [========================>.....] - ETA: 17s - loss: 0.2152 - categorical_accuracy: 0.9356
50144/60000 [========================>.....] - ETA: 17s - loss: 0.2151 - categorical_accuracy: 0.9357
50176/60000 [========================>.....] - ETA: 17s - loss: 0.2150 - categorical_accuracy: 0.9357
50208/60000 [========================>.....] - ETA: 17s - loss: 0.2150 - categorical_accuracy: 0.9357
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2149 - categorical_accuracy: 0.9357
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2149 - categorical_accuracy: 0.9357
50304/60000 [========================>.....] - ETA: 16s - loss: 0.2149 - categorical_accuracy: 0.9358
50336/60000 [========================>.....] - ETA: 16s - loss: 0.2148 - categorical_accuracy: 0.9358
50368/60000 [========================>.....] - ETA: 16s - loss: 0.2147 - categorical_accuracy: 0.9358
50400/60000 [========================>.....] - ETA: 16s - loss: 0.2147 - categorical_accuracy: 0.9358
50432/60000 [========================>.....] - ETA: 16s - loss: 0.2148 - categorical_accuracy: 0.9358
50464/60000 [========================>.....] - ETA: 16s - loss: 0.2147 - categorical_accuracy: 0.9358
50496/60000 [========================>.....] - ETA: 16s - loss: 0.2146 - categorical_accuracy: 0.9359
50528/60000 [========================>.....] - ETA: 16s - loss: 0.2145 - categorical_accuracy: 0.9359
50560/60000 [========================>.....] - ETA: 16s - loss: 0.2145 - categorical_accuracy: 0.9359
50592/60000 [========================>.....] - ETA: 16s - loss: 0.2145 - categorical_accuracy: 0.9359
50624/60000 [========================>.....] - ETA: 16s - loss: 0.2145 - categorical_accuracy: 0.9359
50656/60000 [========================>.....] - ETA: 16s - loss: 0.2143 - categorical_accuracy: 0.9359
50688/60000 [========================>.....] - ETA: 16s - loss: 0.2142 - categorical_accuracy: 0.9359
50720/60000 [========================>.....] - ETA: 16s - loss: 0.2142 - categorical_accuracy: 0.9359
50752/60000 [========================>.....] - ETA: 16s - loss: 0.2141 - categorical_accuracy: 0.9359
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2142 - categorical_accuracy: 0.9359
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2141 - categorical_accuracy: 0.9359
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2139 - categorical_accuracy: 0.9360
50880/60000 [========================>.....] - ETA: 15s - loss: 0.2139 - categorical_accuracy: 0.9360
50912/60000 [========================>.....] - ETA: 15s - loss: 0.2137 - categorical_accuracy: 0.9360
50944/60000 [========================>.....] - ETA: 15s - loss: 0.2136 - categorical_accuracy: 0.9361
50976/60000 [========================>.....] - ETA: 15s - loss: 0.2135 - categorical_accuracy: 0.9361
51008/60000 [========================>.....] - ETA: 15s - loss: 0.2135 - categorical_accuracy: 0.9361
51040/60000 [========================>.....] - ETA: 15s - loss: 0.2134 - categorical_accuracy: 0.9361
51072/60000 [========================>.....] - ETA: 15s - loss: 0.2133 - categorical_accuracy: 0.9361
51104/60000 [========================>.....] - ETA: 15s - loss: 0.2133 - categorical_accuracy: 0.9361
51136/60000 [========================>.....] - ETA: 15s - loss: 0.2132 - categorical_accuracy: 0.9362
51168/60000 [========================>.....] - ETA: 15s - loss: 0.2131 - categorical_accuracy: 0.9362
51200/60000 [========================>.....] - ETA: 15s - loss: 0.2131 - categorical_accuracy: 0.9362
51232/60000 [========================>.....] - ETA: 15s - loss: 0.2130 - categorical_accuracy: 0.9362
51264/60000 [========================>.....] - ETA: 15s - loss: 0.2129 - categorical_accuracy: 0.9363
51296/60000 [========================>.....] - ETA: 15s - loss: 0.2127 - categorical_accuracy: 0.9363
51328/60000 [========================>.....] - ETA: 15s - loss: 0.2127 - categorical_accuracy: 0.9363
51360/60000 [========================>.....] - ETA: 15s - loss: 0.2126 - categorical_accuracy: 0.9363
51392/60000 [========================>.....] - ETA: 15s - loss: 0.2125 - categorical_accuracy: 0.9364
51424/60000 [========================>.....] - ETA: 15s - loss: 0.2124 - categorical_accuracy: 0.9364
51456/60000 [========================>.....] - ETA: 14s - loss: 0.2123 - categorical_accuracy: 0.9364
51488/60000 [========================>.....] - ETA: 14s - loss: 0.2122 - categorical_accuracy: 0.9365
51520/60000 [========================>.....] - ETA: 14s - loss: 0.2120 - categorical_accuracy: 0.9365
51552/60000 [========================>.....] - ETA: 14s - loss: 0.2120 - categorical_accuracy: 0.9365
51584/60000 [========================>.....] - ETA: 14s - loss: 0.2119 - categorical_accuracy: 0.9366
51616/60000 [========================>.....] - ETA: 14s - loss: 0.2118 - categorical_accuracy: 0.9366
51648/60000 [========================>.....] - ETA: 14s - loss: 0.2117 - categorical_accuracy: 0.9366
51680/60000 [========================>.....] - ETA: 14s - loss: 0.2118 - categorical_accuracy: 0.9366
51712/60000 [========================>.....] - ETA: 14s - loss: 0.2118 - categorical_accuracy: 0.9367
51744/60000 [========================>.....] - ETA: 14s - loss: 0.2117 - categorical_accuracy: 0.9367
51776/60000 [========================>.....] - ETA: 14s - loss: 0.2116 - categorical_accuracy: 0.9367
51808/60000 [========================>.....] - ETA: 14s - loss: 0.2117 - categorical_accuracy: 0.9367
51840/60000 [========================>.....] - ETA: 14s - loss: 0.2116 - categorical_accuracy: 0.9367
51872/60000 [========================>.....] - ETA: 14s - loss: 0.2116 - categorical_accuracy: 0.9367
51904/60000 [========================>.....] - ETA: 14s - loss: 0.2114 - categorical_accuracy: 0.9368
51936/60000 [========================>.....] - ETA: 14s - loss: 0.2113 - categorical_accuracy: 0.9368
51968/60000 [========================>.....] - ETA: 14s - loss: 0.2112 - categorical_accuracy: 0.9368
52000/60000 [=========================>....] - ETA: 14s - loss: 0.2112 - categorical_accuracy: 0.9368
52032/60000 [=========================>....] - ETA: 13s - loss: 0.2112 - categorical_accuracy: 0.9368
52064/60000 [=========================>....] - ETA: 13s - loss: 0.2111 - categorical_accuracy: 0.9368
52096/60000 [=========================>....] - ETA: 13s - loss: 0.2110 - categorical_accuracy: 0.9369
52128/60000 [=========================>....] - ETA: 13s - loss: 0.2109 - categorical_accuracy: 0.9369
52160/60000 [=========================>....] - ETA: 13s - loss: 0.2109 - categorical_accuracy: 0.9369
52192/60000 [=========================>....] - ETA: 13s - loss: 0.2107 - categorical_accuracy: 0.9369
52224/60000 [=========================>....] - ETA: 13s - loss: 0.2106 - categorical_accuracy: 0.9370
52256/60000 [=========================>....] - ETA: 13s - loss: 0.2106 - categorical_accuracy: 0.9370
52288/60000 [=========================>....] - ETA: 13s - loss: 0.2105 - categorical_accuracy: 0.9370
52320/60000 [=========================>....] - ETA: 13s - loss: 0.2105 - categorical_accuracy: 0.9370
52352/60000 [=========================>....] - ETA: 13s - loss: 0.2104 - categorical_accuracy: 0.9370
52384/60000 [=========================>....] - ETA: 13s - loss: 0.2103 - categorical_accuracy: 0.9371
52416/60000 [=========================>....] - ETA: 13s - loss: 0.2102 - categorical_accuracy: 0.9371
52448/60000 [=========================>....] - ETA: 13s - loss: 0.2101 - categorical_accuracy: 0.9371
52480/60000 [=========================>....] - ETA: 13s - loss: 0.2101 - categorical_accuracy: 0.9371
52512/60000 [=========================>....] - ETA: 13s - loss: 0.2101 - categorical_accuracy: 0.9371
52544/60000 [=========================>....] - ETA: 13s - loss: 0.2100 - categorical_accuracy: 0.9372
52576/60000 [=========================>....] - ETA: 13s - loss: 0.2099 - categorical_accuracy: 0.9372
52608/60000 [=========================>....] - ETA: 12s - loss: 0.2098 - categorical_accuracy: 0.9372
52640/60000 [=========================>....] - ETA: 12s - loss: 0.2098 - categorical_accuracy: 0.9373
52672/60000 [=========================>....] - ETA: 12s - loss: 0.2098 - categorical_accuracy: 0.9373
52704/60000 [=========================>....] - ETA: 12s - loss: 0.2097 - categorical_accuracy: 0.9373
52736/60000 [=========================>....] - ETA: 12s - loss: 0.2096 - categorical_accuracy: 0.9373
52768/60000 [=========================>....] - ETA: 12s - loss: 0.2096 - categorical_accuracy: 0.9373
52800/60000 [=========================>....] - ETA: 12s - loss: 0.2094 - categorical_accuracy: 0.9373
52832/60000 [=========================>....] - ETA: 12s - loss: 0.2095 - categorical_accuracy: 0.9373
52864/60000 [=========================>....] - ETA: 12s - loss: 0.2094 - categorical_accuracy: 0.9374
52896/60000 [=========================>....] - ETA: 12s - loss: 0.2093 - categorical_accuracy: 0.9374
52928/60000 [=========================>....] - ETA: 12s - loss: 0.2091 - categorical_accuracy: 0.9374
52960/60000 [=========================>....] - ETA: 12s - loss: 0.2093 - categorical_accuracy: 0.9374
52992/60000 [=========================>....] - ETA: 12s - loss: 0.2092 - categorical_accuracy: 0.9374
53024/60000 [=========================>....] - ETA: 12s - loss: 0.2094 - categorical_accuracy: 0.9375
53056/60000 [=========================>....] - ETA: 12s - loss: 0.2093 - categorical_accuracy: 0.9375
53088/60000 [=========================>....] - ETA: 12s - loss: 0.2092 - categorical_accuracy: 0.9375
53120/60000 [=========================>....] - ETA: 12s - loss: 0.2092 - categorical_accuracy: 0.9375
53152/60000 [=========================>....] - ETA: 12s - loss: 0.2091 - categorical_accuracy: 0.9376
53184/60000 [=========================>....] - ETA: 11s - loss: 0.2089 - categorical_accuracy: 0.9376
53216/60000 [=========================>....] - ETA: 11s - loss: 0.2088 - categorical_accuracy: 0.9376
53248/60000 [=========================>....] - ETA: 11s - loss: 0.2087 - categorical_accuracy: 0.9377
53280/60000 [=========================>....] - ETA: 11s - loss: 0.2086 - categorical_accuracy: 0.9377
53312/60000 [=========================>....] - ETA: 11s - loss: 0.2085 - categorical_accuracy: 0.9377
53344/60000 [=========================>....] - ETA: 11s - loss: 0.2084 - categorical_accuracy: 0.9377
53376/60000 [=========================>....] - ETA: 11s - loss: 0.2084 - categorical_accuracy: 0.9378
53408/60000 [=========================>....] - ETA: 11s - loss: 0.2082 - categorical_accuracy: 0.9378
53440/60000 [=========================>....] - ETA: 11s - loss: 0.2082 - categorical_accuracy: 0.9378
53472/60000 [=========================>....] - ETA: 11s - loss: 0.2080 - categorical_accuracy: 0.9379
53504/60000 [=========================>....] - ETA: 11s - loss: 0.2082 - categorical_accuracy: 0.9379
53536/60000 [=========================>....] - ETA: 11s - loss: 0.2080 - categorical_accuracy: 0.9379
53568/60000 [=========================>....] - ETA: 11s - loss: 0.2079 - categorical_accuracy: 0.9379
53600/60000 [=========================>....] - ETA: 11s - loss: 0.2078 - categorical_accuracy: 0.9380
53632/60000 [=========================>....] - ETA: 11s - loss: 0.2077 - categorical_accuracy: 0.9380
53664/60000 [=========================>....] - ETA: 11s - loss: 0.2076 - categorical_accuracy: 0.9380
53696/60000 [=========================>....] - ETA: 11s - loss: 0.2075 - categorical_accuracy: 0.9380
53728/60000 [=========================>....] - ETA: 10s - loss: 0.2075 - categorical_accuracy: 0.9380
53760/60000 [=========================>....] - ETA: 10s - loss: 0.2075 - categorical_accuracy: 0.9380
53792/60000 [=========================>....] - ETA: 10s - loss: 0.2074 - categorical_accuracy: 0.9381
53824/60000 [=========================>....] - ETA: 10s - loss: 0.2073 - categorical_accuracy: 0.9381
53856/60000 [=========================>....] - ETA: 10s - loss: 0.2072 - categorical_accuracy: 0.9381
53888/60000 [=========================>....] - ETA: 10s - loss: 0.2073 - categorical_accuracy: 0.9381
53920/60000 [=========================>....] - ETA: 10s - loss: 0.2071 - categorical_accuracy: 0.9382
53952/60000 [=========================>....] - ETA: 10s - loss: 0.2070 - categorical_accuracy: 0.9382
53984/60000 [=========================>....] - ETA: 10s - loss: 0.2069 - categorical_accuracy: 0.9382
54016/60000 [==========================>...] - ETA: 10s - loss: 0.2069 - categorical_accuracy: 0.9382
54048/60000 [==========================>...] - ETA: 10s - loss: 0.2068 - categorical_accuracy: 0.9383
54080/60000 [==========================>...] - ETA: 10s - loss: 0.2067 - categorical_accuracy: 0.9383
54112/60000 [==========================>...] - ETA: 10s - loss: 0.2067 - categorical_accuracy: 0.9383
54144/60000 [==========================>...] - ETA: 10s - loss: 0.2066 - categorical_accuracy: 0.9383
54176/60000 [==========================>...] - ETA: 10s - loss: 0.2065 - categorical_accuracy: 0.9384
54208/60000 [==========================>...] - ETA: 10s - loss: 0.2064 - categorical_accuracy: 0.9384
54240/60000 [==========================>...] - ETA: 10s - loss: 0.2065 - categorical_accuracy: 0.9384
54272/60000 [==========================>...] - ETA: 10s - loss: 0.2064 - categorical_accuracy: 0.9384
54304/60000 [==========================>...] - ETA: 9s - loss: 0.2064 - categorical_accuracy: 0.9384 
54336/60000 [==========================>...] - ETA: 9s - loss: 0.2063 - categorical_accuracy: 0.9384
54368/60000 [==========================>...] - ETA: 9s - loss: 0.2063 - categorical_accuracy: 0.9384
54400/60000 [==========================>...] - ETA: 9s - loss: 0.2062 - categorical_accuracy: 0.9384
54432/60000 [==========================>...] - ETA: 9s - loss: 0.2061 - categorical_accuracy: 0.9385
54464/60000 [==========================>...] - ETA: 9s - loss: 0.2060 - categorical_accuracy: 0.9385
54496/60000 [==========================>...] - ETA: 9s - loss: 0.2059 - categorical_accuracy: 0.9385
54528/60000 [==========================>...] - ETA: 9s - loss: 0.2058 - categorical_accuracy: 0.9386
54560/60000 [==========================>...] - ETA: 9s - loss: 0.2057 - categorical_accuracy: 0.9386
54592/60000 [==========================>...] - ETA: 9s - loss: 0.2057 - categorical_accuracy: 0.9386
54624/60000 [==========================>...] - ETA: 9s - loss: 0.2057 - categorical_accuracy: 0.9386
54656/60000 [==========================>...] - ETA: 9s - loss: 0.2057 - categorical_accuracy: 0.9386
54688/60000 [==========================>...] - ETA: 9s - loss: 0.2056 - categorical_accuracy: 0.9386
54720/60000 [==========================>...] - ETA: 9s - loss: 0.2056 - categorical_accuracy: 0.9386
54752/60000 [==========================>...] - ETA: 9s - loss: 0.2055 - categorical_accuracy: 0.9387
54784/60000 [==========================>...] - ETA: 9s - loss: 0.2054 - categorical_accuracy: 0.9387
54816/60000 [==========================>...] - ETA: 9s - loss: 0.2054 - categorical_accuracy: 0.9387
54848/60000 [==========================>...] - ETA: 9s - loss: 0.2053 - categorical_accuracy: 0.9387
54880/60000 [==========================>...] - ETA: 8s - loss: 0.2052 - categorical_accuracy: 0.9387
54912/60000 [==========================>...] - ETA: 8s - loss: 0.2051 - categorical_accuracy: 0.9388
54944/60000 [==========================>...] - ETA: 8s - loss: 0.2050 - categorical_accuracy: 0.9388
54976/60000 [==========================>...] - ETA: 8s - loss: 0.2050 - categorical_accuracy: 0.9388
55008/60000 [==========================>...] - ETA: 8s - loss: 0.2049 - categorical_accuracy: 0.9388
55040/60000 [==========================>...] - ETA: 8s - loss: 0.2048 - categorical_accuracy: 0.9389
55072/60000 [==========================>...] - ETA: 8s - loss: 0.2048 - categorical_accuracy: 0.9388
55104/60000 [==========================>...] - ETA: 8s - loss: 0.2047 - categorical_accuracy: 0.9389
55136/60000 [==========================>...] - ETA: 8s - loss: 0.2046 - categorical_accuracy: 0.9389
55168/60000 [==========================>...] - ETA: 8s - loss: 0.2045 - categorical_accuracy: 0.9389
55200/60000 [==========================>...] - ETA: 8s - loss: 0.2045 - categorical_accuracy: 0.9389
55232/60000 [==========================>...] - ETA: 8s - loss: 0.2044 - categorical_accuracy: 0.9390
55264/60000 [==========================>...] - ETA: 8s - loss: 0.2044 - categorical_accuracy: 0.9390
55296/60000 [==========================>...] - ETA: 8s - loss: 0.2044 - categorical_accuracy: 0.9390
55328/60000 [==========================>...] - ETA: 8s - loss: 0.2044 - categorical_accuracy: 0.9390
55360/60000 [==========================>...] - ETA: 8s - loss: 0.2044 - categorical_accuracy: 0.9390
55392/60000 [==========================>...] - ETA: 8s - loss: 0.2043 - categorical_accuracy: 0.9390
55424/60000 [==========================>...] - ETA: 8s - loss: 0.2043 - categorical_accuracy: 0.9390
55456/60000 [==========================>...] - ETA: 7s - loss: 0.2042 - categorical_accuracy: 0.9391
55488/60000 [==========================>...] - ETA: 7s - loss: 0.2041 - categorical_accuracy: 0.9391
55520/60000 [==========================>...] - ETA: 7s - loss: 0.2041 - categorical_accuracy: 0.9391
55552/60000 [==========================>...] - ETA: 7s - loss: 0.2041 - categorical_accuracy: 0.9391
55584/60000 [==========================>...] - ETA: 7s - loss: 0.2040 - categorical_accuracy: 0.9391
55616/60000 [==========================>...] - ETA: 7s - loss: 0.2040 - categorical_accuracy: 0.9391
55648/60000 [==========================>...] - ETA: 7s - loss: 0.2039 - categorical_accuracy: 0.9392
55680/60000 [==========================>...] - ETA: 7s - loss: 0.2038 - categorical_accuracy: 0.9392
55712/60000 [==========================>...] - ETA: 7s - loss: 0.2037 - categorical_accuracy: 0.9392
55744/60000 [==========================>...] - ETA: 7s - loss: 0.2036 - categorical_accuracy: 0.9393
55776/60000 [==========================>...] - ETA: 7s - loss: 0.2036 - categorical_accuracy: 0.9393
55808/60000 [==========================>...] - ETA: 7s - loss: 0.2035 - categorical_accuracy: 0.9393
55840/60000 [==========================>...] - ETA: 7s - loss: 0.2034 - categorical_accuracy: 0.9393
55872/60000 [==========================>...] - ETA: 7s - loss: 0.2034 - categorical_accuracy: 0.9393
55904/60000 [==========================>...] - ETA: 7s - loss: 0.2033 - categorical_accuracy: 0.9393
55936/60000 [==========================>...] - ETA: 7s - loss: 0.2033 - categorical_accuracy: 0.9393
55968/60000 [==========================>...] - ETA: 7s - loss: 0.2033 - categorical_accuracy: 0.9394
56000/60000 [===========================>..] - ETA: 7s - loss: 0.2032 - categorical_accuracy: 0.9394
56032/60000 [===========================>..] - ETA: 6s - loss: 0.2031 - categorical_accuracy: 0.9394
56064/60000 [===========================>..] - ETA: 6s - loss: 0.2030 - categorical_accuracy: 0.9394
56096/60000 [===========================>..] - ETA: 6s - loss: 0.2029 - categorical_accuracy: 0.9394
56128/60000 [===========================>..] - ETA: 6s - loss: 0.2028 - categorical_accuracy: 0.9395
56160/60000 [===========================>..] - ETA: 6s - loss: 0.2028 - categorical_accuracy: 0.9395
56192/60000 [===========================>..] - ETA: 6s - loss: 0.2027 - categorical_accuracy: 0.9395
56224/60000 [===========================>..] - ETA: 6s - loss: 0.2028 - categorical_accuracy: 0.9395
56256/60000 [===========================>..] - ETA: 6s - loss: 0.2027 - categorical_accuracy: 0.9395
56288/60000 [===========================>..] - ETA: 6s - loss: 0.2026 - categorical_accuracy: 0.9395
56320/60000 [===========================>..] - ETA: 6s - loss: 0.2025 - categorical_accuracy: 0.9395
56352/60000 [===========================>..] - ETA: 6s - loss: 0.2024 - categorical_accuracy: 0.9396
56384/60000 [===========================>..] - ETA: 6s - loss: 0.2023 - categorical_accuracy: 0.9396
56416/60000 [===========================>..] - ETA: 6s - loss: 0.2023 - categorical_accuracy: 0.9396
56448/60000 [===========================>..] - ETA: 6s - loss: 0.2023 - categorical_accuracy: 0.9396
56480/60000 [===========================>..] - ETA: 6s - loss: 0.2023 - categorical_accuracy: 0.9396
56512/60000 [===========================>..] - ETA: 6s - loss: 0.2023 - categorical_accuracy: 0.9396
56544/60000 [===========================>..] - ETA: 6s - loss: 0.2022 - categorical_accuracy: 0.9396
56576/60000 [===========================>..] - ETA: 6s - loss: 0.2022 - categorical_accuracy: 0.9396
56608/60000 [===========================>..] - ETA: 5s - loss: 0.2021 - categorical_accuracy: 0.9396
56640/60000 [===========================>..] - ETA: 5s - loss: 0.2021 - categorical_accuracy: 0.9396
56672/60000 [===========================>..] - ETA: 5s - loss: 0.2021 - categorical_accuracy: 0.9396
56704/60000 [===========================>..] - ETA: 5s - loss: 0.2020 - categorical_accuracy: 0.9397
56736/60000 [===========================>..] - ETA: 5s - loss: 0.2020 - categorical_accuracy: 0.9397
56768/60000 [===========================>..] - ETA: 5s - loss: 0.2019 - categorical_accuracy: 0.9397
56800/60000 [===========================>..] - ETA: 5s - loss: 0.2018 - categorical_accuracy: 0.9397
56832/60000 [===========================>..] - ETA: 5s - loss: 0.2017 - categorical_accuracy: 0.9398
56864/60000 [===========================>..] - ETA: 5s - loss: 0.2016 - categorical_accuracy: 0.9398
56896/60000 [===========================>..] - ETA: 5s - loss: 0.2016 - categorical_accuracy: 0.9397
56928/60000 [===========================>..] - ETA: 5s - loss: 0.2015 - categorical_accuracy: 0.9398
56960/60000 [===========================>..] - ETA: 5s - loss: 0.2014 - categorical_accuracy: 0.9398
56992/60000 [===========================>..] - ETA: 5s - loss: 0.2014 - categorical_accuracy: 0.9398
57024/60000 [===========================>..] - ETA: 5s - loss: 0.2013 - categorical_accuracy: 0.9398
57056/60000 [===========================>..] - ETA: 5s - loss: 0.2012 - categorical_accuracy: 0.9399
57088/60000 [===========================>..] - ETA: 5s - loss: 0.2011 - categorical_accuracy: 0.9399
57120/60000 [===========================>..] - ETA: 5s - loss: 0.2010 - categorical_accuracy: 0.9399
57152/60000 [===========================>..] - ETA: 4s - loss: 0.2009 - categorical_accuracy: 0.9399
57184/60000 [===========================>..] - ETA: 4s - loss: 0.2009 - categorical_accuracy: 0.9399
57216/60000 [===========================>..] - ETA: 4s - loss: 0.2008 - categorical_accuracy: 0.9399
57248/60000 [===========================>..] - ETA: 4s - loss: 0.2008 - categorical_accuracy: 0.9399
57280/60000 [===========================>..] - ETA: 4s - loss: 0.2008 - categorical_accuracy: 0.9400
57312/60000 [===========================>..] - ETA: 4s - loss: 0.2007 - categorical_accuracy: 0.9400
57344/60000 [===========================>..] - ETA: 4s - loss: 0.2006 - categorical_accuracy: 0.9400
57376/60000 [===========================>..] - ETA: 4s - loss: 0.2005 - categorical_accuracy: 0.9401
57408/60000 [===========================>..] - ETA: 4s - loss: 0.2005 - categorical_accuracy: 0.9400
57440/60000 [===========================>..] - ETA: 4s - loss: 0.2004 - categorical_accuracy: 0.9401
57472/60000 [===========================>..] - ETA: 4s - loss: 0.2003 - categorical_accuracy: 0.9401
57504/60000 [===========================>..] - ETA: 4s - loss: 0.2002 - categorical_accuracy: 0.9401
57536/60000 [===========================>..] - ETA: 4s - loss: 0.2002 - categorical_accuracy: 0.9401
57568/60000 [===========================>..] - ETA: 4s - loss: 0.2001 - categorical_accuracy: 0.9401
57600/60000 [===========================>..] - ETA: 4s - loss: 0.2000 - categorical_accuracy: 0.9402
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1999 - categorical_accuracy: 0.9402
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1999 - categorical_accuracy: 0.9402
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1998 - categorical_accuracy: 0.9402
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1997 - categorical_accuracy: 0.9402
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1997 - categorical_accuracy: 0.9403
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1996 - categorical_accuracy: 0.9403
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1995 - categorical_accuracy: 0.9403
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1997 - categorical_accuracy: 0.9403
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1997 - categorical_accuracy: 0.9403
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1996 - categorical_accuracy: 0.9403
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1995 - categorical_accuracy: 0.9404
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1994 - categorical_accuracy: 0.9404
58016/60000 [============================>.] - ETA: 3s - loss: 0.1993 - categorical_accuracy: 0.9404
58048/60000 [============================>.] - ETA: 3s - loss: 0.1992 - categorical_accuracy: 0.9404
58080/60000 [============================>.] - ETA: 3s - loss: 0.1991 - categorical_accuracy: 0.9405
58112/60000 [============================>.] - ETA: 3s - loss: 0.1990 - categorical_accuracy: 0.9405
58144/60000 [============================>.] - ETA: 3s - loss: 0.1991 - categorical_accuracy: 0.9405
58176/60000 [============================>.] - ETA: 3s - loss: 0.1990 - categorical_accuracy: 0.9405
58208/60000 [============================>.] - ETA: 3s - loss: 0.1990 - categorical_accuracy: 0.9405
58240/60000 [============================>.] - ETA: 3s - loss: 0.1989 - categorical_accuracy: 0.9405
58272/60000 [============================>.] - ETA: 3s - loss: 0.1990 - categorical_accuracy: 0.9405
58304/60000 [============================>.] - ETA: 2s - loss: 0.1989 - categorical_accuracy: 0.9406
58336/60000 [============================>.] - ETA: 2s - loss: 0.1988 - categorical_accuracy: 0.9406
58368/60000 [============================>.] - ETA: 2s - loss: 0.1987 - categorical_accuracy: 0.9406
58400/60000 [============================>.] - ETA: 2s - loss: 0.1986 - categorical_accuracy: 0.9407
58432/60000 [============================>.] - ETA: 2s - loss: 0.1985 - categorical_accuracy: 0.9407
58464/60000 [============================>.] - ETA: 2s - loss: 0.1985 - categorical_accuracy: 0.9407
58496/60000 [============================>.] - ETA: 2s - loss: 0.1983 - categorical_accuracy: 0.9407
58528/60000 [============================>.] - ETA: 2s - loss: 0.1983 - categorical_accuracy: 0.9408
58560/60000 [============================>.] - ETA: 2s - loss: 0.1983 - categorical_accuracy: 0.9408
58592/60000 [============================>.] - ETA: 2s - loss: 0.1982 - categorical_accuracy: 0.9408
58624/60000 [============================>.] - ETA: 2s - loss: 0.1981 - categorical_accuracy: 0.9408
58656/60000 [============================>.] - ETA: 2s - loss: 0.1980 - categorical_accuracy: 0.9408
58688/60000 [============================>.] - ETA: 2s - loss: 0.1979 - categorical_accuracy: 0.9409
58720/60000 [============================>.] - ETA: 2s - loss: 0.1979 - categorical_accuracy: 0.9409
58752/60000 [============================>.] - ETA: 2s - loss: 0.1978 - categorical_accuracy: 0.9409
58784/60000 [============================>.] - ETA: 2s - loss: 0.1978 - categorical_accuracy: 0.9409
58816/60000 [============================>.] - ETA: 2s - loss: 0.1979 - categorical_accuracy: 0.9409
58848/60000 [============================>.] - ETA: 2s - loss: 0.1978 - categorical_accuracy: 0.9410
58880/60000 [============================>.] - ETA: 1s - loss: 0.1977 - categorical_accuracy: 0.9410
58912/60000 [============================>.] - ETA: 1s - loss: 0.1977 - categorical_accuracy: 0.9410
58944/60000 [============================>.] - ETA: 1s - loss: 0.1976 - categorical_accuracy: 0.9410
58976/60000 [============================>.] - ETA: 1s - loss: 0.1975 - categorical_accuracy: 0.9410
59008/60000 [============================>.] - ETA: 1s - loss: 0.1975 - categorical_accuracy: 0.9411
59040/60000 [============================>.] - ETA: 1s - loss: 0.1975 - categorical_accuracy: 0.9411
59072/60000 [============================>.] - ETA: 1s - loss: 0.1974 - categorical_accuracy: 0.9411
59104/60000 [============================>.] - ETA: 1s - loss: 0.1973 - categorical_accuracy: 0.9411
59136/60000 [============================>.] - ETA: 1s - loss: 0.1974 - categorical_accuracy: 0.9411
59168/60000 [============================>.] - ETA: 1s - loss: 0.1975 - categorical_accuracy: 0.9411
59200/60000 [============================>.] - ETA: 1s - loss: 0.1974 - categorical_accuracy: 0.9411
59232/60000 [============================>.] - ETA: 1s - loss: 0.1973 - categorical_accuracy: 0.9412
59264/60000 [============================>.] - ETA: 1s - loss: 0.1973 - categorical_accuracy: 0.9412
59296/60000 [============================>.] - ETA: 1s - loss: 0.1973 - categorical_accuracy: 0.9412
59328/60000 [============================>.] - ETA: 1s - loss: 0.1972 - categorical_accuracy: 0.9412
59360/60000 [============================>.] - ETA: 1s - loss: 0.1971 - categorical_accuracy: 0.9412
59392/60000 [============================>.] - ETA: 1s - loss: 0.1970 - categorical_accuracy: 0.9412
59424/60000 [============================>.] - ETA: 1s - loss: 0.1969 - categorical_accuracy: 0.9413
59456/60000 [============================>.] - ETA: 0s - loss: 0.1969 - categorical_accuracy: 0.9413
59488/60000 [============================>.] - ETA: 0s - loss: 0.1968 - categorical_accuracy: 0.9413
59520/60000 [============================>.] - ETA: 0s - loss: 0.1967 - categorical_accuracy: 0.9413
59552/60000 [============================>.] - ETA: 0s - loss: 0.1967 - categorical_accuracy: 0.9413
59584/60000 [============================>.] - ETA: 0s - loss: 0.1968 - categorical_accuracy: 0.9413
59616/60000 [============================>.] - ETA: 0s - loss: 0.1968 - categorical_accuracy: 0.9413
59648/60000 [============================>.] - ETA: 0s - loss: 0.1967 - categorical_accuracy: 0.9414
59680/60000 [============================>.] - ETA: 0s - loss: 0.1966 - categorical_accuracy: 0.9414
59712/60000 [============================>.] - ETA: 0s - loss: 0.1966 - categorical_accuracy: 0.9414
59744/60000 [============================>.] - ETA: 0s - loss: 0.1965 - categorical_accuracy: 0.9414
59776/60000 [============================>.] - ETA: 0s - loss: 0.1965 - categorical_accuracy: 0.9414
59808/60000 [============================>.] - ETA: 0s - loss: 0.1964 - categorical_accuracy: 0.9414
59840/60000 [============================>.] - ETA: 0s - loss: 0.1964 - categorical_accuracy: 0.9414
59872/60000 [============================>.] - ETA: 0s - loss: 0.1963 - categorical_accuracy: 0.9415
59904/60000 [============================>.] - ETA: 0s - loss: 0.1962 - categorical_accuracy: 0.9415
59936/60000 [============================>.] - ETA: 0s - loss: 0.1961 - categorical_accuracy: 0.9415
59968/60000 [============================>.] - ETA: 0s - loss: 0.1960 - categorical_accuracy: 0.9415
60000/60000 [==============================] - 109s 2ms/step - loss: 0.1960 - categorical_accuracy: 0.9415 - val_loss: 0.0521 - val_categorical_accuracy: 0.9829

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  672/10000 [=>............................] - ETA: 3s
  800/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 3s
 1120/10000 [==>...........................] - ETA: 3s
 1280/10000 [==>...........................] - ETA: 3s
 1440/10000 [===>..........................] - ETA: 3s
 1600/10000 [===>..........................] - ETA: 3s
 1760/10000 [====>.........................] - ETA: 3s
 1920/10000 [====>.........................] - ETA: 2s
 2080/10000 [=====>........................] - ETA: 2s
 2240/10000 [=====>........................] - ETA: 2s
 2400/10000 [======>.......................] - ETA: 2s
 2560/10000 [======>.......................] - ETA: 2s
 2720/10000 [=======>......................] - ETA: 2s
 2880/10000 [=======>......................] - ETA: 2s
 3040/10000 [========>.....................] - ETA: 2s
 3200/10000 [========>.....................] - ETA: 2s
 3360/10000 [=========>....................] - ETA: 2s
 3520/10000 [=========>....................] - ETA: 2s
 3680/10000 [==========>...................] - ETA: 2s
 3840/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 2s
 4320/10000 [===========>..................] - ETA: 1s
 4480/10000 [============>.................] - ETA: 1s
 4640/10000 [============>.................] - ETA: 1s
 4800/10000 [=============>................] - ETA: 1s
 4960/10000 [=============>................] - ETA: 1s
 5120/10000 [==============>...............] - ETA: 1s
 5280/10000 [==============>...............] - ETA: 1s
 5440/10000 [===============>..............] - ETA: 1s
 5600/10000 [===============>..............] - ETA: 1s
 5760/10000 [================>.............] - ETA: 1s
 5920/10000 [================>.............] - ETA: 1s
 6080/10000 [=================>............] - ETA: 1s
 6208/10000 [=================>............] - ETA: 1s
 6368/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 1s
 6688/10000 [===================>..........] - ETA: 1s
 6848/10000 [===================>..........] - ETA: 1s
 7008/10000 [====================>.........] - ETA: 1s
 7168/10000 [====================>.........] - ETA: 0s
 7328/10000 [====================>.........] - ETA: 0s
 7488/10000 [=====================>........] - ETA: 0s
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
 9856/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 348us/step
[[6.2273733e-09 2.8728360e-09 2.0198244e-07 ... 9.9999881e-01
  4.6998139e-10 9.8388318e-07]
 [8.6416112e-06 1.5409820e-05 9.9996448e-01 ... 7.9136573e-09
  6.1460219e-06 6.4532255e-09]
 [2.0301127e-07 9.9998689e-01 2.7395495e-06 ... 2.5982345e-06
  2.1740036e-06 1.2065622e-07]
 ...
 [1.9176545e-10 5.4868813e-08 9.8301345e-10 ... 5.1379480e-08
  4.1545979e-07 3.1054205e-06]
 [7.6418667e-07 2.1773856e-07 3.5410181e-08 ... 6.9212152e-09
  1.4157620e-03 5.3683476e-07]
 [1.3992378e-06 3.8874941e-08 6.5146787e-07 ... 2.8264743e-11
  4.9930441e-07 8.9462437e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05205049658230273, 'accuracy_test:': 0.9829000234603882}

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
   54f13b0..1ba3d5f  master     -> origin/master
Updating 54f13b0..1ba3d5f
Fast-forward
 .../20200523/list_log_pullrequest_20200523.md      |   2 +-
 error_list/20200523/list_log_testall_20200523.md   | 432 +++++++++++++++++++++
 2 files changed, 433 insertions(+), 1 deletion(-)
[master f6e0b00] ml_store
 1 file changed, 2044 insertions(+)
To github.com:arita37/mlmodels_store.git
   1ba3d5f..f6e0b00  master -> master





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
{'loss': 0.4575369507074356, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-23 00:37:17.322879: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master d36708b] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   f6e0b00..d36708b  master -> master





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
[master 95b3c6b] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   d36708b..95b3c6b  master -> master





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
 40%|████      | 2/5 [00:19<00:28,  9.62s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8693008020486012, 'learning_rate': 0.009810574698305086, 'min_data_in_leaf': 18, 'num_leaves': 30} and reward: 0.3874
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\xd1O\xeae\xee\x13X\r\x00\x00\x00learning_rateq\x02G?\x84\x17\x91\x0b\xdf\x9c\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.3874
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\xd1O\xeae\xee\x13X\r\x00\x00\x00learning_rateq\x02G?\x84\x17\x91\x0b\xdf\x9c\x9aX\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K\x1eu.' and reward: 0.3874
 60%|██████    | 3/5 [00:36<00:24, 12.06s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8206127360691404, 'learning_rate': 0.005307845138055458, 'min_data_in_leaf': 12, 'num_leaves': 65} and reward: 0.384
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaBu\xa4\x03# X\r\x00\x00\x00learning_rateq\x02G?u\xbd\xad\xd4{\x9e\xdcX\x10\x00\x00\x00min_data_in_leafq\x03K\x0cX\n\x00\x00\x00num_leavesq\x04KAu.' and reward: 0.384
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaBu\xa4\x03# X\r\x00\x00\x00learning_rateq\x02G?u\xbd\xad\xd4{\x9e\xdcX\x10\x00\x00\x00min_data_in_leafq\x03K\x0cX\n\x00\x00\x00num_leavesq\x04KAu.' and reward: 0.384
 80%|████████  | 4/5 [01:06<00:17, 17.34s/it] 80%|████████  | 4/5 [01:06<00:16, 16.67s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.836577756256386, 'learning_rate': 0.024367423435848617, 'min_data_in_leaf': 20, 'num_leaves': 45} and reward: 0.3926
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xc5>\xb6\xf5\xd4\xe3X\r\x00\x00\x00learning_rateq\x02G?\x98\xf3\xc6\x1a\xfa\xa0\xbbX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K-u.' and reward: 0.3926
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xc5>\xb6\xf5\xd4\xe3X\r\x00\x00\x00learning_rateq\x02G?\x98\xf3\xc6\x1a\xfa\xa0\xbbX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K-u.' and reward: 0.3926
Time for Gradient Boosting hyperparameter optimization: 90.31813168525696
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.836577756256386, 'learning_rate': 0.024367423435848617, 'min_data_in_leaf': 20, 'num_leaves': 45}
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
 40%|████      | 2/5 [00:54<01:21, 27.18s/it] 40%|████      | 2/5 [00:54<01:21, 27.18s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.24940363153840922, 'embedding_size_factor': 1.0930118842303727, 'layers.choice': 1, 'learning_rate': 0.004668875117385519, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 3.995355398844257e-08} and reward: 0.383
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xcf\xecuL{\x05bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1|\xfa\x07\x8e\x8afX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?s\x1f\xab\x9f\x030\xceX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>es,\xb5\xb5~\x01u.' and reward: 0.383
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xcf\xecuL{\x05bX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1|\xfa\x07\x8e\x8afX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?s\x1f\xab\x9f\x030\xceX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>es,\xb5\xb5~\x01u.' and reward: 0.383
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 136.31588912010193
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -110.35s of remaining time.
Ensemble size: 22
Ensemble weights: 
[0.22727273 0.22727273 0.04545455 0.04545455 0.22727273 0.22727273]
	0.3974	 = Validation accuracy score
	1.38s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 231.78s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f46b32db898>

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
   95b3c6b..dd77401  master     -> origin/master
Updating 95b3c6b..dd77401
Fast-forward
 deps.txt                                           |   33 +-
 error_list/20200523/list_log_jupyter_20200523.md   | 1727 ++++++++++---------
 .../20200523/list_log_pullrequest_20200523.md      |    2 +-
 error_list/20200523/list_log_testall_20200523.md   |  175 ++
 log_jupyter/log_jupyter.py                         | 1766 ++++++++++----------
 5 files changed, 1949 insertions(+), 1754 deletions(-)
[master 8edc5cc] ml_store
 1 file changed, 209 insertions(+)
To github.com:arita37/mlmodels_store.git
   dd77401..8edc5cc  master -> master





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
[master 4ff34ce] ml_store
 2 files changed, 38 insertions(+), 31 deletions(-)
To github.com:arita37/mlmodels_store.git
   8edc5cc..4ff34ce  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.91it/s, avg_epoch_loss=5.22]
INFO:root:Epoch[0] Elapsed time 2.561 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.222427
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.222426509857177 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f694c61f400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f694c61f400>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 110.74it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1040.3418782552083,
    "abs_error": 365.45660400390625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.4214971201957987,
    "sMAPE": 0.5080158696314547,
    "MSIS": 96.85986863119665,
    "QuantileLoss[0.5]": 365.45655822753906,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.25433115498147,
    "NRMSE": 0.6790385506311889,
    "ND": 0.6411519368489583,
    "wQuantileLoss[0.5]": 0.6411518565395422,
    "mean_wQuantileLoss": 0.6411518565395422,
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
100%|██████████| 10/10 [00:01<00:00,  8.01it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.249 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f6921b83940>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f6921b83940>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 158.21it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.42it/s, avg_epoch_loss=5.18]
INFO:root:Epoch[0] Elapsed time 1.846 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.178048
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.178048419952392 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f6944cc0a20>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f6944cc0a20>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 150.48it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 382.1051025390625,
    "abs_error": 208.52398681640625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.38166947329888,
    "sMAPE": 0.33021140896954876,
    "MSIS": 55.26677569662814,
    "QuantileLoss[0.5]": 208.52399444580078,
    "Coverage[0.5]": 0.8333333333333334,
    "RMSE": 19.547508857628443,
    "NRMSE": 0.41152650226586196,
    "ND": 0.3658315558182566,
    "wQuantileLoss[0.5]": 0.3658315692031593,
    "mean_wQuantileLoss": 0.3658315692031593,
    "MAE_Coverage": 0.33333333333333337
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
 30%|███       | 3/10 [00:12<00:28,  4.13s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:23<00:15,  3.98s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:34<00:03,  3.89s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:38<00:00,  3.80s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 38.013 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.861812
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.861812114715576 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f6910f84128>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f6910f84128>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 145.63it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 51828.010416666664,
    "abs_error": 2659.50439453125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.621742669169713,
    "sMAPE": 1.4028503169024111,
    "MSIS": 704.869680884172,
    "QuantileLoss[0.5]": 2659.5042114257812,
    "Coverage[0.5]": 1.0,
    "RMSE": 227.65766057101322,
    "NRMSE": 4.792792854126594,
    "ND": 4.6657971833881575,
    "wQuantileLoss[0.5]": 4.6657968621504935,
    "mean_wQuantileLoss": 4.6657968621504935,
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
100%|██████████| 10/10 [00:00<00:00, 57.99it/s, avg_epoch_loss=5.3]
INFO:root:Epoch[0] Elapsed time 0.173 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.296765
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.296765279769898 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f6910e24f28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f6910e24f28>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 160.31it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 332.49232991536456,
    "abs_error": 181.42330932617188,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2021017441102833,
    "sMAPE": 0.3152905056561849,
    "MSIS": 48.0840681467478,
    "QuantileLoss[0.5]": 181.42330932617188,
    "Coverage[0.5]": 0.5833333333333334,
    "RMSE": 18.23437221061818,
    "NRMSE": 0.3838815202235406,
    "ND": 0.3182865075897752,
    "wQuantileLoss[0.5]": 0.3182865075897752,
    "mean_wQuantileLoss": 0.3182865075897752,
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
100%|██████████| 10/10 [00:01<00:00,  9.07it/s, avg_epoch_loss=123]
INFO:root:Epoch[0] Elapsed time 1.102 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=122.866774
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 122.86677375545166 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f6920128ef0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f6920128ef0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 154.34it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [01:58<17:47, 118.61s/it, avg_epoch_loss=0.703] 20%|██        | 2/10 [04:56<18:11, 136.42s/it, avg_epoch_loss=0.686] 30%|███       | 3/10 [07:59<17:32, 150.37s/it, avg_epoch_loss=0.669] 40%|████      | 4/10 [11:05<16:06, 161.06s/it, avg_epoch_loss=0.652] 50%|█████     | 5/10 [14:27<14:26, 173.33s/it, avg_epoch_loss=0.634] 60%|██████    | 6/10 [17:48<12:06, 181.57s/it, avg_epoch_loss=0.617] 70%|███████   | 7/10 [21:10<09:22, 187.62s/it, avg_epoch_loss=0.599] 80%|████████  | 8/10 [24:39<06:28, 194.05s/it, avg_epoch_loss=0.581] 90%|█████████ | 9/10 [28:04<03:17, 197.39s/it, avg_epoch_loss=0.563]100%|██████████| 10/10 [31:29<00:00, 199.78s/it, avg_epoch_loss=0.546]100%|██████████| 10/10 [31:29<00:00, 188.97s/it, avg_epoch_loss=0.546]
INFO:root:Epoch[0] Elapsed time 1889.709 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.546187
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5461866021156311 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f6910da9ef0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f6910da9ef0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 19.43it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
   4ff34ce..a8fae09  master     -> origin/master
Updating 4ff34ce..a8fae09
Fast-forward
 deps.txt                                           |    4 +-
 .../20200523/list_log_pullrequest_20200523.md      |    2 +-
 error_list/20200523/list_log_test_cli_20200523.md  | 1798 ++++++------
 error_list/20200523/list_log_testall_20200523.md   |    7 +
 log_test_cli/log_cli.py                            | 2936 ++++++++++----------
 5 files changed, 2437 insertions(+), 2310 deletions(-)
[master 69477a4] ml_store
 1 file changed, 509 insertions(+)
To github.com:arita37/mlmodels_store.git
   a8fae09..69477a4  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f5546d494e0> 

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
[master 5ed85fc] ml_store
 2 files changed, 111 insertions(+), 2 deletions(-)
To github.com:arita37/mlmodels_store.git
   69477a4..5ed85fc  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.61363671  0.3166589   1.34710546 -1.89526695 -0.76045809  0.08972912
  -0.32905155  0.41026575  0.85987097 -1.04906775]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
   0.42952614  0.1340482   1.20205486  0.10622272]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 0.69174373  1.00978733 -1.21333813 -1.55694156 -1.20257258 -0.61244213
  -2.69836174 -0.13935181 -0.72853749  0.0722519 ]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f6cf394edd8>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f6d0dcc05f8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
   0.42952614  0.1340482   1.20205486  0.10622272]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]
 [ 0.69174373  1.00978733 -1.21333813 -1.55694156 -1.20257258 -0.61244213
  -2.69836174 -0.13935181 -0.72853749  0.0722519 ]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]
 [ 1.34728643 -0.36453805  0.08075099 -0.45971768 -0.8894876   1.70548352
   0.09499611  0.24050555 -0.9994265  -0.76780375]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 1.02817479 -0.50845713  1.7653351   0.77741921  0.61771419 -0.11877117
   0.45015551 -0.19899818  1.86647138  0.8709698 ]
 [ 0.61363671  0.3166589   1.34710546 -1.89526695 -0.76045809  0.08972912
  -0.32905155  0.41026575  0.85987097 -1.04906775]
 [ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 0.98042741  1.93752881 -0.23083974  0.36633201  1.10018476 -1.04458938
  -0.34498721  2.05117344  0.585662   -2.793085  ]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]]
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
[[ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 1.1437713   0.7278135   0.35249436  0.51507361  1.17718111 -2.78253447
  -1.94332341  0.58464661  0.32427424 -0.23643695]
 [ 1.66752297  1.22372221 -0.4599301  -0.0593679  -0.493857    1.4489894
  -1.18110317 -0.47758085  0.02599999 -0.79079995]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 1.01195228 -1.88141087  1.70018815  0.4972691  -0.91766462  0.2373327
  -1.09033833 -2.14444405 -0.36956243  0.60878366]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]]
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
[master f506695] ml_store
 1 file changed, 247 insertions(+)
To github.com:arita37/mlmodels_store.git
   5ed85fc..f506695  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140511647690704
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140511647690480
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140511647689248
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140511647688800
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140511647688296
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140511647687960

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
grad_step = 000000, loss = 0.638348
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.503735
grad_step = 000002, loss = 0.391491
grad_step = 000003, loss = 0.273683
grad_step = 000004, loss = 0.160142
grad_step = 000005, loss = 0.086190
grad_step = 000006, loss = 0.073789
grad_step = 000007, loss = 0.070168
grad_step = 000008, loss = 0.053109
grad_step = 000009, loss = 0.044375
grad_step = 000010, loss = 0.039559
grad_step = 000011, loss = 0.035269
grad_step = 000012, loss = 0.031154
grad_step = 000013, loss = 0.026953
grad_step = 000014, loss = 0.023351
grad_step = 000015, loss = 0.021094
grad_step = 000016, loss = 0.019058
grad_step = 000017, loss = 0.017286
grad_step = 000018, loss = 0.015820
grad_step = 000019, loss = 0.014391
grad_step = 000020, loss = 0.013122
grad_step = 000021, loss = 0.012383
grad_step = 000022, loss = 0.012086
grad_step = 000023, loss = 0.011730
grad_step = 000024, loss = 0.011071
grad_step = 000025, loss = 0.010225
grad_step = 000026, loss = 0.009391
grad_step = 000027, loss = 0.008564
grad_step = 000028, loss = 0.007751
grad_step = 000029, loss = 0.007195
grad_step = 000030, loss = 0.007181
grad_step = 000031, loss = 0.007608
grad_step = 000032, loss = 0.007928
grad_step = 000033, loss = 0.007754
grad_step = 000034, loss = 0.007179
grad_step = 000035, loss = 0.006547
grad_step = 000036, loss = 0.006079
grad_step = 000037, loss = 0.005811
grad_step = 000038, loss = 0.005719
grad_step = 000039, loss = 0.005762
grad_step = 000040, loss = 0.005847
grad_step = 000041, loss = 0.005853
grad_step = 000042, loss = 0.005730
grad_step = 000043, loss = 0.005534
grad_step = 000044, loss = 0.005365
grad_step = 000045, loss = 0.005235
grad_step = 000046, loss = 0.005082
grad_step = 000047, loss = 0.004916
grad_step = 000048, loss = 0.004803
grad_step = 000049, loss = 0.004742
grad_step = 000050, loss = 0.004675
grad_step = 000051, loss = 0.004594
grad_step = 000052, loss = 0.004534
grad_step = 000053, loss = 0.004487
grad_step = 000054, loss = 0.004421
grad_step = 000055, loss = 0.004338
grad_step = 000056, loss = 0.004251
grad_step = 000057, loss = 0.004140
grad_step = 000058, loss = 0.004003
grad_step = 000059, loss = 0.003891
grad_step = 000060, loss = 0.003826
grad_step = 000061, loss = 0.003779
grad_step = 000062, loss = 0.003730
grad_step = 000063, loss = 0.003679
grad_step = 000064, loss = 0.003605
grad_step = 000065, loss = 0.003508
grad_step = 000066, loss = 0.003411
grad_step = 000067, loss = 0.003315
grad_step = 000068, loss = 0.003222
grad_step = 000069, loss = 0.003156
grad_step = 000070, loss = 0.003102
grad_step = 000071, loss = 0.003042
grad_step = 000072, loss = 0.002980
grad_step = 000073, loss = 0.002909
grad_step = 000074, loss = 0.002838
grad_step = 000075, loss = 0.002767
grad_step = 000076, loss = 0.002696
grad_step = 000077, loss = 0.002633
grad_step = 000078, loss = 0.002574
grad_step = 000079, loss = 0.002525
grad_step = 000080, loss = 0.002477
grad_step = 000081, loss = 0.002431
grad_step = 000082, loss = 0.002382
grad_step = 000083, loss = 0.002337
grad_step = 000084, loss = 0.002295
grad_step = 000085, loss = 0.002257
grad_step = 000086, loss = 0.002224
grad_step = 000087, loss = 0.002192
grad_step = 000088, loss = 0.002158
grad_step = 000089, loss = 0.002125
grad_step = 000090, loss = 0.002093
grad_step = 000091, loss = 0.002064
grad_step = 000092, loss = 0.002032
grad_step = 000093, loss = 0.002000
grad_step = 000094, loss = 0.001964
grad_step = 000095, loss = 0.001953
grad_step = 000096, loss = 0.002035
grad_step = 000097, loss = 0.001988
grad_step = 000098, loss = 0.001812
grad_step = 000099, loss = 0.001898
grad_step = 000100, loss = 0.001793
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.001742
grad_step = 000102, loss = 0.001769
grad_step = 000103, loss = 0.001654
grad_step = 000104, loss = 0.001708
grad_step = 000105, loss = 0.001606
grad_step = 000106, loss = 0.001596
grad_step = 000107, loss = 0.001570
grad_step = 000108, loss = 0.001503
grad_step = 000109, loss = 0.001515
grad_step = 000110, loss = 0.001436
grad_step = 000111, loss = 0.001438
grad_step = 000112, loss = 0.001386
grad_step = 000113, loss = 0.001355
grad_step = 000114, loss = 0.001336
grad_step = 000115, loss = 0.001283
grad_step = 000116, loss = 0.001273
grad_step = 000117, loss = 0.001227
grad_step = 000118, loss = 0.001200
grad_step = 000119, loss = 0.001179
grad_step = 000120, loss = 0.001136
grad_step = 000121, loss = 0.001117
grad_step = 000122, loss = 0.001090
grad_step = 000123, loss = 0.001054
grad_step = 000124, loss = 0.001040
grad_step = 000125, loss = 0.001009
grad_step = 000126, loss = 0.000981
grad_step = 000127, loss = 0.000966
grad_step = 000128, loss = 0.000940
grad_step = 000129, loss = 0.000918
grad_step = 000130, loss = 0.000904
grad_step = 000131, loss = 0.000884
grad_step = 000132, loss = 0.000866
grad_step = 000133, loss = 0.000850
grad_step = 000134, loss = 0.000838
grad_step = 000135, loss = 0.000828
grad_step = 000136, loss = 0.000815
grad_step = 000137, loss = 0.000802
grad_step = 000138, loss = 0.000794
grad_step = 000139, loss = 0.000787
grad_step = 000140, loss = 0.000780
grad_step = 000141, loss = 0.000774
grad_step = 000142, loss = 0.000766
grad_step = 000143, loss = 0.000758
grad_step = 000144, loss = 0.000752
grad_step = 000145, loss = 0.000745
grad_step = 000146, loss = 0.000738
grad_step = 000147, loss = 0.000733
grad_step = 000148, loss = 0.000727
grad_step = 000149, loss = 0.000721
grad_step = 000150, loss = 0.000716
grad_step = 000151, loss = 0.000711
grad_step = 000152, loss = 0.000707
grad_step = 000153, loss = 0.000704
grad_step = 000154, loss = 0.000710
grad_step = 000155, loss = 0.000735
grad_step = 000156, loss = 0.000776
grad_step = 000157, loss = 0.000811
grad_step = 000158, loss = 0.000791
grad_step = 000159, loss = 0.000715
grad_step = 000160, loss = 0.000673
grad_step = 000161, loss = 0.000710
grad_step = 000162, loss = 0.000743
grad_step = 000163, loss = 0.000712
grad_step = 000164, loss = 0.000669
grad_step = 000165, loss = 0.000666
grad_step = 000166, loss = 0.000694
grad_step = 000167, loss = 0.000694
grad_step = 000168, loss = 0.000658
grad_step = 000169, loss = 0.000646
grad_step = 000170, loss = 0.000663
grad_step = 000171, loss = 0.000667
grad_step = 000172, loss = 0.000650
grad_step = 000173, loss = 0.000632
grad_step = 000174, loss = 0.000635
grad_step = 000175, loss = 0.000645
grad_step = 000176, loss = 0.000639
grad_step = 000177, loss = 0.000622
grad_step = 000178, loss = 0.000617
grad_step = 000179, loss = 0.000620
grad_step = 000180, loss = 0.000623
grad_step = 000181, loss = 0.000617
grad_step = 000182, loss = 0.000606
grad_step = 000183, loss = 0.000600
grad_step = 000184, loss = 0.000601
grad_step = 000185, loss = 0.000602
grad_step = 000186, loss = 0.000599
grad_step = 000187, loss = 0.000592
grad_step = 000188, loss = 0.000586
grad_step = 000189, loss = 0.000583
grad_step = 000190, loss = 0.000581
grad_step = 000191, loss = 0.000581
grad_step = 000192, loss = 0.000579
grad_step = 000193, loss = 0.000575
grad_step = 000194, loss = 0.000569
grad_step = 000195, loss = 0.000565
grad_step = 000196, loss = 0.000561
grad_step = 000197, loss = 0.000559
grad_step = 000198, loss = 0.000557
grad_step = 000199, loss = 0.000555
grad_step = 000200, loss = 0.000553
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000551
grad_step = 000202, loss = 0.000550
grad_step = 000203, loss = 0.000550
grad_step = 000204, loss = 0.000549
grad_step = 000205, loss = 0.000548
grad_step = 000206, loss = 0.000547
grad_step = 000207, loss = 0.000544
grad_step = 000208, loss = 0.000543
grad_step = 000209, loss = 0.000543
grad_step = 000210, loss = 0.000545
grad_step = 000211, loss = 0.000547
grad_step = 000212, loss = 0.000549
grad_step = 000213, loss = 0.000549
grad_step = 000214, loss = 0.000549
grad_step = 000215, loss = 0.000548
grad_step = 000216, loss = 0.000548
grad_step = 000217, loss = 0.000546
grad_step = 000218, loss = 0.000542
grad_step = 000219, loss = 0.000527
grad_step = 000220, loss = 0.000508
grad_step = 000221, loss = 0.000493
grad_step = 000222, loss = 0.000490
grad_step = 000223, loss = 0.000493
grad_step = 000224, loss = 0.000497
grad_step = 000225, loss = 0.000496
grad_step = 000226, loss = 0.000495
grad_step = 000227, loss = 0.000499
grad_step = 000228, loss = 0.000509
grad_step = 000229, loss = 0.000522
grad_step = 000230, loss = 0.000533
grad_step = 000231, loss = 0.000540
grad_step = 000232, loss = 0.000542
grad_step = 000233, loss = 0.000542
grad_step = 000234, loss = 0.000534
grad_step = 000235, loss = 0.000515
grad_step = 000236, loss = 0.000485
grad_step = 000237, loss = 0.000457
grad_step = 000238, loss = 0.000447
grad_step = 000239, loss = 0.000457
grad_step = 000240, loss = 0.000473
grad_step = 000241, loss = 0.000482
grad_step = 000242, loss = 0.000478
grad_step = 000243, loss = 0.000467
grad_step = 000244, loss = 0.000457
grad_step = 000245, loss = 0.000448
grad_step = 000246, loss = 0.000440
grad_step = 000247, loss = 0.000433
grad_step = 000248, loss = 0.000428
grad_step = 000249, loss = 0.000429
grad_step = 000250, loss = 0.000434
grad_step = 000251, loss = 0.000442
grad_step = 000252, loss = 0.000450
grad_step = 000253, loss = 0.000455
grad_step = 000254, loss = 0.000456
grad_step = 000255, loss = 0.000457
grad_step = 000256, loss = 0.000459
grad_step = 000257, loss = 0.000465
grad_step = 000258, loss = 0.000474
grad_step = 000259, loss = 0.000478
grad_step = 000260, loss = 0.000471
grad_step = 000261, loss = 0.000444
grad_step = 000262, loss = 0.000415
grad_step = 000263, loss = 0.000403
grad_step = 000264, loss = 0.000406
grad_step = 000265, loss = 0.000411
grad_step = 000266, loss = 0.000406
grad_step = 000267, loss = 0.000400
grad_step = 000268, loss = 0.000399
grad_step = 000269, loss = 0.000407
grad_step = 000270, loss = 0.000416
grad_step = 000271, loss = 0.000416
grad_step = 000272, loss = 0.000408
grad_step = 000273, loss = 0.000400
grad_step = 000274, loss = 0.000399
grad_step = 000275, loss = 0.000404
grad_step = 000276, loss = 0.000410
grad_step = 000277, loss = 0.000410
grad_step = 000278, loss = 0.000404
grad_step = 000279, loss = 0.000399
grad_step = 000280, loss = 0.000401
grad_step = 000281, loss = 0.000407
grad_step = 000282, loss = 0.000413
grad_step = 000283, loss = 0.000412
grad_step = 000284, loss = 0.000405
grad_step = 000285, loss = 0.000395
grad_step = 000286, loss = 0.000387
grad_step = 000287, loss = 0.000386
grad_step = 000288, loss = 0.000390
grad_step = 000289, loss = 0.000396
grad_step = 000290, loss = 0.000392
grad_step = 000291, loss = 0.000380
grad_step = 000292, loss = 0.000364
grad_step = 000293, loss = 0.000358
grad_step = 000294, loss = 0.000361
grad_step = 000295, loss = 0.000363
grad_step = 000296, loss = 0.000359
grad_step = 000297, loss = 0.000348
grad_step = 000298, loss = 0.000339
grad_step = 000299, loss = 0.000336
grad_step = 000300, loss = 0.000339
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000343
grad_step = 000302, loss = 0.000344
grad_step = 000303, loss = 0.000342
grad_step = 000304, loss = 0.000337
grad_step = 000305, loss = 0.000331
grad_step = 000306, loss = 0.000328
grad_step = 000307, loss = 0.000329
grad_step = 000308, loss = 0.000333
grad_step = 000309, loss = 0.000342
grad_step = 000310, loss = 0.000357
grad_step = 000311, loss = 0.000388
grad_step = 000312, loss = 0.000450
grad_step = 000313, loss = 0.000570
grad_step = 000314, loss = 0.000743
grad_step = 000315, loss = 0.000916
grad_step = 000316, loss = 0.000848
grad_step = 000317, loss = 0.000564
grad_step = 000318, loss = 0.000358
grad_step = 000319, loss = 0.000457
grad_step = 000320, loss = 0.000587
grad_step = 000321, loss = 0.000477
grad_step = 000322, loss = 0.000356
grad_step = 000323, loss = 0.000420
grad_step = 000324, loss = 0.000473
grad_step = 000325, loss = 0.000385
grad_step = 000326, loss = 0.000344
grad_step = 000327, loss = 0.000407
grad_step = 000328, loss = 0.000394
grad_step = 000329, loss = 0.000334
grad_step = 000330, loss = 0.000357
grad_step = 000331, loss = 0.000380
grad_step = 000332, loss = 0.000340
grad_step = 000333, loss = 0.000325
grad_step = 000334, loss = 0.000350
grad_step = 000335, loss = 0.000343
grad_step = 000336, loss = 0.000313
grad_step = 000337, loss = 0.000320
grad_step = 000338, loss = 0.000338
grad_step = 000339, loss = 0.000321
grad_step = 000340, loss = 0.000301
grad_step = 000341, loss = 0.000314
grad_step = 000342, loss = 0.000324
grad_step = 000343, loss = 0.000306
grad_step = 000344, loss = 0.000294
grad_step = 000345, loss = 0.000306
grad_step = 000346, loss = 0.000312
grad_step = 000347, loss = 0.000298
grad_step = 000348, loss = 0.000289
grad_step = 000349, loss = 0.000297
grad_step = 000350, loss = 0.000302
grad_step = 000351, loss = 0.000294
grad_step = 000352, loss = 0.000287
grad_step = 000353, loss = 0.000289
grad_step = 000354, loss = 0.000292
grad_step = 000355, loss = 0.000288
grad_step = 000356, loss = 0.000284
grad_step = 000357, loss = 0.000285
grad_step = 000358, loss = 0.000286
grad_step = 000359, loss = 0.000283
grad_step = 000360, loss = 0.000279
grad_step = 000361, loss = 0.000280
grad_step = 000362, loss = 0.000282
grad_step = 000363, loss = 0.000282
grad_step = 000364, loss = 0.000279
grad_step = 000365, loss = 0.000276
grad_step = 000366, loss = 0.000276
grad_step = 000367, loss = 0.000276
grad_step = 000368, loss = 0.000274
grad_step = 000369, loss = 0.000272
grad_step = 000370, loss = 0.000271
grad_step = 000371, loss = 0.000271
grad_step = 000372, loss = 0.000271
grad_step = 000373, loss = 0.000271
grad_step = 000374, loss = 0.000270
grad_step = 000375, loss = 0.000269
grad_step = 000376, loss = 0.000270
grad_step = 000377, loss = 0.000272
grad_step = 000378, loss = 0.000274
grad_step = 000379, loss = 0.000277
grad_step = 000380, loss = 0.000282
grad_step = 000381, loss = 0.000287
grad_step = 000382, loss = 0.000293
grad_step = 000383, loss = 0.000297
grad_step = 000384, loss = 0.000299
grad_step = 000385, loss = 0.000290
grad_step = 000386, loss = 0.000279
grad_step = 000387, loss = 0.000267
grad_step = 000388, loss = 0.000262
grad_step = 000389, loss = 0.000263
grad_step = 000390, loss = 0.000267
grad_step = 000391, loss = 0.000271
grad_step = 000392, loss = 0.000272
grad_step = 000393, loss = 0.000272
grad_step = 000394, loss = 0.000270
grad_step = 000395, loss = 0.000268
grad_step = 000396, loss = 0.000266
grad_step = 000397, loss = 0.000263
grad_step = 000398, loss = 0.000259
grad_step = 000399, loss = 0.000256
grad_step = 000400, loss = 0.000253
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000251
grad_step = 000402, loss = 0.000252
grad_step = 000403, loss = 0.000253
grad_step = 000404, loss = 0.000256
grad_step = 000405, loss = 0.000260
grad_step = 000406, loss = 0.000265
grad_step = 000407, loss = 0.000273
grad_step = 000408, loss = 0.000288
grad_step = 000409, loss = 0.000308
grad_step = 000410, loss = 0.000342
grad_step = 000411, loss = 0.000365
grad_step = 000412, loss = 0.000376
grad_step = 000413, loss = 0.000325
grad_step = 000414, loss = 0.000268
grad_step = 000415, loss = 0.000249
grad_step = 000416, loss = 0.000275
grad_step = 000417, loss = 0.000299
grad_step = 000418, loss = 0.000277
grad_step = 000419, loss = 0.000250
grad_step = 000420, loss = 0.000257
grad_step = 000421, loss = 0.000277
grad_step = 000422, loss = 0.000277
grad_step = 000423, loss = 0.000258
grad_step = 000424, loss = 0.000252
grad_step = 000425, loss = 0.000263
grad_step = 000426, loss = 0.000268
grad_step = 000427, loss = 0.000258
grad_step = 000428, loss = 0.000247
grad_step = 000429, loss = 0.000249
grad_step = 000430, loss = 0.000256
grad_step = 000431, loss = 0.000253
grad_step = 000432, loss = 0.000245
grad_step = 000433, loss = 0.000240
grad_step = 000434, loss = 0.000243
grad_step = 000435, loss = 0.000246
grad_step = 000436, loss = 0.000245
grad_step = 000437, loss = 0.000241
grad_step = 000438, loss = 0.000239
grad_step = 000439, loss = 0.000242
grad_step = 000440, loss = 0.000247
grad_step = 000441, loss = 0.000252
grad_step = 000442, loss = 0.000257
grad_step = 000443, loss = 0.000266
grad_step = 000444, loss = 0.000282
grad_step = 000445, loss = 0.000313
grad_step = 000446, loss = 0.000358
grad_step = 000447, loss = 0.000423
grad_step = 000448, loss = 0.000484
grad_step = 000449, loss = 0.000536
grad_step = 000450, loss = 0.000508
grad_step = 000451, loss = 0.000417
grad_step = 000452, loss = 0.000296
grad_step = 000453, loss = 0.000238
grad_step = 000454, loss = 0.000267
grad_step = 000455, loss = 0.000326
grad_step = 000456, loss = 0.000343
grad_step = 000457, loss = 0.000299
grad_step = 000458, loss = 0.000248
grad_step = 000459, loss = 0.000240
grad_step = 000460, loss = 0.000267
grad_step = 000461, loss = 0.000287
grad_step = 000462, loss = 0.000273
grad_step = 000463, loss = 0.000247
grad_step = 000464, loss = 0.000235
grad_step = 000465, loss = 0.000244
grad_step = 000466, loss = 0.000257
grad_step = 000467, loss = 0.000256
grad_step = 000468, loss = 0.000244
grad_step = 000469, loss = 0.000232
grad_step = 000470, loss = 0.000232
grad_step = 000471, loss = 0.000238
grad_step = 000472, loss = 0.000241
grad_step = 000473, loss = 0.000235
grad_step = 000474, loss = 0.000227
grad_step = 000475, loss = 0.000225
grad_step = 000476, loss = 0.000228
grad_step = 000477, loss = 0.000232
grad_step = 000478, loss = 0.000231
grad_step = 000479, loss = 0.000225
grad_step = 000480, loss = 0.000219
grad_step = 000481, loss = 0.000217
grad_step = 000482, loss = 0.000219
grad_step = 000483, loss = 0.000223
grad_step = 000484, loss = 0.000225
grad_step = 000485, loss = 0.000225
grad_step = 000486, loss = 0.000222
grad_step = 000487, loss = 0.000219
grad_step = 000488, loss = 0.000216
grad_step = 000489, loss = 0.000214
grad_step = 000490, loss = 0.000214
grad_step = 000491, loss = 0.000214
grad_step = 000492, loss = 0.000214
grad_step = 000493, loss = 0.000214
grad_step = 000494, loss = 0.000212
grad_step = 000495, loss = 0.000211
grad_step = 000496, loss = 0.000210
grad_step = 000497, loss = 0.000210
grad_step = 000498, loss = 0.000210
grad_step = 000499, loss = 0.000212
grad_step = 000500, loss = 0.000213
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000217
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
[[0.85026944 0.85567236 0.9206177  0.9472797  1.0001922 ]
 [0.85330945 0.9079582  0.9479213  1.0191392  0.9919573 ]
 [0.8901996  0.9108099  0.98933786 0.99364734 0.9426537 ]
 [0.91064954 0.9779262  0.987291   0.9663657  0.9188093 ]
 [0.99085975 0.9931003  0.9509028  0.9156958  0.85238445]
 [0.982259   0.94957566 0.90774137 0.86911476 0.84898746]
 [0.94563085 0.92083365 0.8616665  0.87178755 0.8177147 ]
 [0.89681846 0.8462674  0.8568913  0.81511277 0.8447461 ]
 [0.82126147 0.8398618  0.8112659  0.8492936  0.84831035]
 [0.8254653  0.80795205 0.8327215  0.8680159  0.82151   ]
 [0.79940224 0.81754214 0.855504   0.8146263  0.92848456]
 [0.82630146 0.8334982  0.8140114  0.9334261  0.9384544 ]
 [0.83995414 0.8501426  0.91793823 0.9480416  1.0001403 ]
 [0.85062504 0.91334057 0.9548253  1.0201181  0.98481536]
 [0.89871407 0.92334974 0.9944104  0.98605704 0.9289155 ]
 [0.9215821  0.98950386 0.97875595 0.94876486 0.8974931 ]
 [0.9899362  0.98253536 0.93295133 0.8944576  0.8316864 ]
 [0.9765891  0.9310295  0.89047843 0.84731925 0.8363147 ]
 [0.9409576  0.90596473 0.84056306 0.8554764  0.8126266 ]
 [0.9034006  0.84244126 0.8439397  0.8134579  0.8409289 ]
 [0.83360237 0.846643   0.80963004 0.8516863  0.85251486]
 [0.8413987  0.8173807  0.8333863  0.87844175 0.8228136 ]
 [0.8089288  0.821835   0.86138415 0.8237381  0.9324994 ]
 [0.83214855 0.8395358  0.8181144  0.93459564 0.9405379 ]
 [0.85696286 0.8596599  0.9201809  0.9495494  1.0065864 ]
 [0.8609884  0.9137281  0.9519719  1.027426   1.0032593 ]
 [0.89744806 0.9175189  0.99684477 1.0066594  0.9551805 ]
 [0.9167261  0.984977   0.9973002  0.97771806 0.9304426 ]
 [0.9953091  1.0019803  0.9595414  0.92802453 0.8592907 ]
 [0.9891851  0.9547345  0.91505355 0.87688065 0.85428876]
 [0.95377505 0.9262019  0.8658987  0.87579036 0.8253281 ]]

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
[master a57f6b2] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   f506695..a57f6b2  master -> master





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
