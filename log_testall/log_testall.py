
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
[master 7916ca4] ml_store  && git pull --all
 2 files changed, 61 insertions(+), 9758 deletions(-)
 rewrite log_testall/log_testall.py (99%)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + b453e25...7916ca4 master -> master (forced update)





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
[master 65871dc] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   7916ca4..65871dc  master -> master





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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-26 08:15:42.988156: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-26 08:15:43.003120: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-26 08:15:43.003321: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b545c40050 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 08:15:43.003366: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 208
Trainable params: 208
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2925 - binary_crossentropy: 1.9497500/500 [==============================] - 1s 2ms/sample - loss: 0.2814 - binary_crossentropy: 1.7672 - val_loss: 0.2811 - val_binary_crossentropy: 1.6395

  #### metrics   #################################################### 
{'MSE': 0.2810241774103407}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         24          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
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
Total params: 208
Trainable params: 208
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
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
Total params: 458
Trainable params: 458
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2560 - binary_crossentropy: 0.7072500/500 [==============================] - 1s 2ms/sample - loss: 0.2672 - binary_crossentropy: 0.7314 - val_loss: 0.2837 - val_binary_crossentropy: 0.7674

  #### metrics   #################################################### 
{'MSE': 0.27500844727648266}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
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
Total params: 458
Trainable params: 458
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 622
Trainable params: 622
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 3ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.2498078546537938}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 622
Trainable params: 622
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
Total params: 403
Trainable params: 403
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.4900 - binary_crossentropy: 7.5582500/500 [==============================] - 2s 3ms/sample - loss: 0.4800 - binary_crossentropy: 7.4040 - val_loss: 0.5120 - val_binary_crossentropy: 7.8976

  #### metrics   #################################################### 
{'MSE': 0.496}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
Total params: 403
Trainable params: 403
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 153
Trainable params: 153
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 3s - loss: 0.2656 - binary_crossentropy: 0.7323500/500 [==============================] - 2s 5ms/sample - loss: 0.2859 - binary_crossentropy: 0.8053 - val_loss: 0.2908 - val_binary_crossentropy: 0.7883

  #### metrics   #################################################### 
{'MSE': 0.28772333308350745}

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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 153
Trainable params: 153
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-26 08:17:26.525283: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:26.527839: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:26.535566: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 08:17:26.548070: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 08:17:26.550416: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 08:17:26.552518: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:26.554330: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2443 - val_binary_crossentropy: 0.6817
2020-05-26 08:17:28.248302: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:28.250338: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:28.255287: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 08:17:28.265343: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 08:17:28.267283: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 08:17:28.269075: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:28.270632: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24150024823980468}

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
2020-05-26 08:17:57.166126: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:57.167608: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:57.171733: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 08:17:57.178688: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 08:17:57.179874: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 08:17:57.180994: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:57.182009: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2508 - val_binary_crossentropy: 0.6948
2020-05-26 08:17:59.074959: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:59.076300: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:59.079229: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 08:17:59.085223: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 08:17:59.086210: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 08:17:59.087069: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:17:59.088161: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2510086459333711}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-26 08:18:40.103527: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:18:40.110321: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:18:40.128816: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 08:18:40.160908: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 08:18:40.166409: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 08:18:40.171581: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:18:40.176791: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 6s 6s/sample - loss: 0.0383 - binary_crossentropy: 0.2177 - val_loss: 0.3558 - val_binary_crossentropy: 0.9682
2020-05-26 08:18:43.173472: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:18:43.180028: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:18:43.197408: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 08:18:43.228585: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 08:18:43.233841: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 08:18:43.238976: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 08:18:43.244010: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2473600085067614}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 725
Trainable params: 725
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2504 - binary_crossentropy: 0.6887500/500 [==============================] - 6s 11ms/sample - loss: 0.2589 - binary_crossentropy: 0.8406 - val_loss: 0.2635 - val_binary_crossentropy: 1.0585

  #### metrics   #################################################### 
{'MSE': 0.26073129899294223}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 725
Trainable params: 725
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         14          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
Total params: 266
Trainable params: 266
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3077 - binary_crossentropy: 0.8339500/500 [==============================] - 6s 11ms/sample - loss: 0.3123 - binary_crossentropy: 0.8520 - val_loss: 0.2884 - val_binary_crossentropy: 0.8141

  #### metrics   #################################################### 
{'MSE': 0.2991451523126142}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 2)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         14          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         6           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
Total params: 266
Trainable params: 266
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
Total params: 1,929
Trainable params: 1,929
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2933 - binary_crossentropy: 0.7913500/500 [==============================] - 6s 12ms/sample - loss: 0.2989 - binary_crossentropy: 0.8033 - val_loss: 0.2905 - val_binary_crossentropy: 0.7837

  #### metrics   #################################################### 
{'MSE': 0.2923157236539568}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
Total params: 1,929
Trainable params: 1,929
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
regionsequence_sum (InputLayer) [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 9)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
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
Total params: 192
Trainable params: 192
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 10s - loss: 0.2602 - binary_crossentropy: 0.7140500/500 [==============================] - 7s 15ms/sample - loss: 0.2687 - binary_crossentropy: 0.7313 - val_loss: 0.2596 - val_binary_crossentropy: 0.7386

  #### metrics   #################################################### 
{'MSE': 0.2629319863802216}

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
regionsequence_sum (InputLayer) [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 9)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         8           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         9           regionsequence_max[0][0]         
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
Total params: 192
Trainable params: 192
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 1,397
Trainable params: 1,397
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2559 - binary_crossentropy: 0.7063500/500 [==============================] - 7s 15ms/sample - loss: 0.2608 - binary_crossentropy: 0.7162 - val_loss: 0.2635 - val_binary_crossentropy: 0.7211

  #### metrics   #################################################### 
{'MSE': 0.2607006399655863}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 1,397
Trainable params: 1,397
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
Total params: 3,052
Trainable params: 2,972
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 10s - loss: 0.2636 - binary_crossentropy: 0.7191500/500 [==============================] - 8s 16ms/sample - loss: 0.2592 - binary_crossentropy: 0.7117 - val_loss: 0.2519 - val_binary_crossentropy: 0.6971

  #### metrics   #################################################### 
{'MSE': 0.2549501275204424}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
[master 96b8992] ml_store  && git pull --all
 1 file changed, 4946 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + d1efd79...96b8992 master -> master (forced update)





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
[master af59274] ml_store  && git pull --all
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   96b8992..af59274  master -> master





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
[master 2971844] ml_store  && git pull --all
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   af59274..2971844  master -> master





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
[master f2a7363] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   2971844..f2a7363  master -> master





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
[master 1480448] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   f2a7363..1480448  master -> master





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
[master d792c59] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   1480448..d792c59  master -> master





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
[master 97ee645] ml_store  && git pull --all
 1 file changed, 43 insertions(+)
To github.com:arita37/mlmodels_store.git
   d792c59..97ee645  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 12s
 2301952/17464789 [==>...........................] - ETA: 0s 
 6692864/17464789 [==========>...................] - ETA: 0s
11444224/17464789 [==================>...........] - ETA: 0s
16392192/17464789 [===========================>..] - ETA: 0s
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
2020-05-26 08:30:53.281187: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-26 08:30:53.287648: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-26 08:30:53.287891: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f8fcf1d070 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 08:30:53.287910: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 16s - loss: 7.4213 - accuracy: 0.5160
 2000/25000 [=>............................] - ETA: 12s - loss: 7.6973 - accuracy: 0.4980
 3000/25000 [==>...........................] - ETA: 10s - loss: 7.6768 - accuracy: 0.4993
 4000/25000 [===>..........................] - ETA: 9s - loss: 7.6896 - accuracy: 0.4985 
 5000/25000 [=====>........................] - ETA: 8s - loss: 7.6482 - accuracy: 0.5012
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.5772 - accuracy: 0.5058
 7000/25000 [=======>......................] - ETA: 7s - loss: 7.6272 - accuracy: 0.5026
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6091 - accuracy: 0.5038
 9000/25000 [=========>....................] - ETA: 6s - loss: 7.6070 - accuracy: 0.5039
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6160 - accuracy: 0.5033
11000/25000 [============>.................] - ETA: 5s - loss: 7.6248 - accuracy: 0.5027
12000/25000 [=============>................] - ETA: 5s - loss: 7.6027 - accuracy: 0.5042
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6430 - accuracy: 0.5015
14000/25000 [===============>..............] - ETA: 4s - loss: 7.6261 - accuracy: 0.5026
15000/25000 [=================>............] - ETA: 3s - loss: 7.6124 - accuracy: 0.5035
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6379 - accuracy: 0.5019
17000/25000 [===================>..........] - ETA: 3s - loss: 7.6477 - accuracy: 0.5012
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6470 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6456 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6406 - accuracy: 0.5017
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6469 - accuracy: 0.5013
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6394 - accuracy: 0.5018
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6486 - accuracy: 0.5012
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
25000/25000 [==============================] - 11s 454us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f9251a4c1d0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f9255057470> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 15s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 11s - loss: 7.5900 - accuracy: 0.5050
 3000/25000 [==>...........................] - ETA: 10s - loss: 7.5593 - accuracy: 0.5070
 4000/25000 [===>..........................] - ETA: 9s - loss: 7.5555 - accuracy: 0.5073 
 5000/25000 [=====>........................] - ETA: 8s - loss: 7.6298 - accuracy: 0.5024
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.6513 - accuracy: 0.5010
 7000/25000 [=======>......................] - ETA: 7s - loss: 7.6447 - accuracy: 0.5014
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6724 - accuracy: 0.4996
 9000/25000 [=========>....................] - ETA: 6s - loss: 7.6087 - accuracy: 0.5038
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6068 - accuracy: 0.5039
11000/25000 [============>.................] - ETA: 5s - loss: 7.6178 - accuracy: 0.5032
12000/25000 [=============>................] - ETA: 5s - loss: 7.6066 - accuracy: 0.5039
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6183 - accuracy: 0.5032
14000/25000 [===============>..............] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
15000/25000 [=================>............] - ETA: 3s - loss: 7.6155 - accuracy: 0.5033
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6283 - accuracy: 0.5025
17000/25000 [===================>..........] - ETA: 3s - loss: 7.6675 - accuracy: 0.4999
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6658 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6699 - accuracy: 0.4998
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6597 - accuracy: 0.5005
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6717 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6645 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 11s 460us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 14s - loss: 7.9886 - accuracy: 0.4790
 2000/25000 [=>............................] - ETA: 11s - loss: 7.7740 - accuracy: 0.4930
 3000/25000 [==>...........................] - ETA: 9s - loss: 7.7331 - accuracy: 0.4957 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.7165 - accuracy: 0.4967
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6452 - accuracy: 0.5014
 6000/25000 [======>.......................] - ETA: 7s - loss: 7.5976 - accuracy: 0.5045
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.5987 - accuracy: 0.5044
 8000/25000 [========>.....................] - ETA: 6s - loss: 7.6455 - accuracy: 0.5014
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6939 - accuracy: 0.4982
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6789 - accuracy: 0.4992
11000/25000 [============>.................] - ETA: 5s - loss: 7.6778 - accuracy: 0.4993
12000/25000 [=============>................] - ETA: 4s - loss: 7.6986 - accuracy: 0.4979
13000/25000 [==============>...............] - ETA: 4s - loss: 7.7091 - accuracy: 0.4972
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7236 - accuracy: 0.4963
15000/25000 [=================>............] - ETA: 3s - loss: 7.7218 - accuracy: 0.4964
16000/25000 [==================>...........] - ETA: 3s - loss: 7.7040 - accuracy: 0.4976
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7108 - accuracy: 0.4971
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7075 - accuracy: 0.4973
19000/25000 [=====================>........] - ETA: 2s - loss: 7.6860 - accuracy: 0.4987
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6965 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6885 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 1s - loss: 7.6994 - accuracy: 0.4979
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
25000/25000 [==============================] - 10s 419us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master c762186] ml_store  && git pull --all
 1 file changed, 317 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 4861841...c762186 master -> master (forced update)





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

13/13 [==============================] - 2s 138ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 6ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 8/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 9/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 67c0ad8] ml_store  && git pull --all
 1 file changed, 124 insertions(+)
To github.com:arita37/mlmodels_store.git
   c762186..67c0ad8  master -> master





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
 2146304/11490434 [====>.........................] - ETA: 0s
 6799360/11490434 [================>.............] - ETA: 0s
10911744/11490434 [===========================>..] - ETA: 0s
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

   32/60000 [..............................] - ETA: 8:49 - loss: 2.3154 - categorical_accuracy: 0.0312
   64/60000 [..............................] - ETA: 5:31 - loss: 2.2600 - categorical_accuracy: 0.1406
   96/60000 [..............................] - ETA: 4:22 - loss: 2.2532 - categorical_accuracy: 0.1875
  128/60000 [..............................] - ETA: 3:49 - loss: 2.2156 - categorical_accuracy: 0.2188
  160/60000 [..............................] - ETA: 3:27 - loss: 2.1654 - categorical_accuracy: 0.2313
  192/60000 [..............................] - ETA: 3:14 - loss: 2.1574 - categorical_accuracy: 0.2448
  224/60000 [..............................] - ETA: 3:04 - loss: 2.1233 - categorical_accuracy: 0.2634
  256/60000 [..............................] - ETA: 2:58 - loss: 2.1305 - categorical_accuracy: 0.2539
  288/60000 [..............................] - ETA: 2:53 - loss: 2.0927 - categorical_accuracy: 0.2917
  320/60000 [..............................] - ETA: 2:49 - loss: 2.0371 - categorical_accuracy: 0.3125
  352/60000 [..............................] - ETA: 2:46 - loss: 2.0029 - categorical_accuracy: 0.3267
  384/60000 [..............................] - ETA: 2:42 - loss: 1.9802 - categorical_accuracy: 0.3359
  416/60000 [..............................] - ETA: 2:38 - loss: 1.9541 - categorical_accuracy: 0.3413
  448/60000 [..............................] - ETA: 2:36 - loss: 1.9124 - categorical_accuracy: 0.3571
  480/60000 [..............................] - ETA: 2:35 - loss: 1.8737 - categorical_accuracy: 0.3708
  512/60000 [..............................] - ETA: 2:32 - loss: 1.8341 - categorical_accuracy: 0.3887
  544/60000 [..............................] - ETA: 2:31 - loss: 1.7955 - categorical_accuracy: 0.4062
  576/60000 [..............................] - ETA: 2:29 - loss: 1.7820 - categorical_accuracy: 0.4097
  608/60000 [..............................] - ETA: 2:28 - loss: 1.7602 - categorical_accuracy: 0.4211
  640/60000 [..............................] - ETA: 2:26 - loss: 1.7397 - categorical_accuracy: 0.4297
  672/60000 [..............................] - ETA: 2:25 - loss: 1.7069 - categorical_accuracy: 0.4435
  704/60000 [..............................] - ETA: 2:24 - loss: 1.6892 - categorical_accuracy: 0.4489
  736/60000 [..............................] - ETA: 2:23 - loss: 1.6730 - categorical_accuracy: 0.4524
  768/60000 [..............................] - ETA: 2:22 - loss: 1.6636 - categorical_accuracy: 0.4544
  800/60000 [..............................] - ETA: 2:21 - loss: 1.6385 - categorical_accuracy: 0.4638
  832/60000 [..............................] - ETA: 2:20 - loss: 1.6223 - categorical_accuracy: 0.4688
  864/60000 [..............................] - ETA: 2:19 - loss: 1.5841 - categorical_accuracy: 0.4850
  896/60000 [..............................] - ETA: 2:18 - loss: 1.5678 - categorical_accuracy: 0.4866
  928/60000 [..............................] - ETA: 2:18 - loss: 1.5436 - categorical_accuracy: 0.4935
  960/60000 [..............................] - ETA: 2:18 - loss: 1.5143 - categorical_accuracy: 0.5052
  992/60000 [..............................] - ETA: 2:17 - loss: 1.4978 - categorical_accuracy: 0.5121
 1024/60000 [..............................] - ETA: 2:17 - loss: 1.4767 - categorical_accuracy: 0.5166
 1056/60000 [..............................] - ETA: 2:17 - loss: 1.4610 - categorical_accuracy: 0.5218
 1088/60000 [..............................] - ETA: 2:17 - loss: 1.4392 - categorical_accuracy: 0.5285
 1120/60000 [..............................] - ETA: 2:16 - loss: 1.4185 - categorical_accuracy: 0.5339
 1152/60000 [..............................] - ETA: 2:16 - loss: 1.4098 - categorical_accuracy: 0.5391
 1184/60000 [..............................] - ETA: 2:16 - loss: 1.3948 - categorical_accuracy: 0.5456
 1216/60000 [..............................] - ETA: 2:16 - loss: 1.3859 - categorical_accuracy: 0.5510
 1248/60000 [..............................] - ETA: 2:15 - loss: 1.3719 - categorical_accuracy: 0.5561
 1280/60000 [..............................] - ETA: 2:15 - loss: 1.3578 - categorical_accuracy: 0.5602
 1312/60000 [..............................] - ETA: 2:15 - loss: 1.3382 - categorical_accuracy: 0.5678
 1344/60000 [..............................] - ETA: 2:14 - loss: 1.3224 - categorical_accuracy: 0.5714
 1376/60000 [..............................] - ETA: 2:14 - loss: 1.3106 - categorical_accuracy: 0.5741
 1408/60000 [..............................] - ETA: 2:14 - loss: 1.3035 - categorical_accuracy: 0.5753
 1440/60000 [..............................] - ETA: 2:13 - loss: 1.2975 - categorical_accuracy: 0.5792
 1472/60000 [..............................] - ETA: 2:13 - loss: 1.2854 - categorical_accuracy: 0.5822
 1504/60000 [..............................] - ETA: 2:13 - loss: 1.2734 - categorical_accuracy: 0.5871
 1536/60000 [..............................] - ETA: 2:12 - loss: 1.2652 - categorical_accuracy: 0.5905
 1568/60000 [..............................] - ETA: 2:12 - loss: 1.2522 - categorical_accuracy: 0.5938
 1600/60000 [..............................] - ETA: 2:12 - loss: 1.2356 - categorical_accuracy: 0.5987
 1632/60000 [..............................] - ETA: 2:12 - loss: 1.2276 - categorical_accuracy: 0.6011
 1664/60000 [..............................] - ETA: 2:12 - loss: 1.2204 - categorical_accuracy: 0.6028
 1696/60000 [..............................] - ETA: 2:11 - loss: 1.2127 - categorical_accuracy: 0.6061
 1728/60000 [..............................] - ETA: 2:11 - loss: 1.2029 - categorical_accuracy: 0.6088
 1760/60000 [..............................] - ETA: 2:11 - loss: 1.1952 - categorical_accuracy: 0.6102
 1792/60000 [..............................] - ETA: 2:11 - loss: 1.1896 - categorical_accuracy: 0.6122
 1824/60000 [..............................] - ETA: 2:11 - loss: 1.1835 - categorical_accuracy: 0.6135
 1856/60000 [..............................] - ETA: 2:11 - loss: 1.1796 - categorical_accuracy: 0.6148
 1888/60000 [..............................] - ETA: 2:11 - loss: 1.1683 - categorical_accuracy: 0.6176
 1920/60000 [..............................] - ETA: 2:10 - loss: 1.1542 - categorical_accuracy: 0.6224
 1952/60000 [..............................] - ETA: 2:10 - loss: 1.1431 - categorical_accuracy: 0.6265
 1984/60000 [..............................] - ETA: 2:10 - loss: 1.1308 - categorical_accuracy: 0.6310
 2016/60000 [>.............................] - ETA: 2:10 - loss: 1.1199 - categorical_accuracy: 0.6344
 2048/60000 [>.............................] - ETA: 2:10 - loss: 1.1231 - categorical_accuracy: 0.6362
 2080/60000 [>.............................] - ETA: 2:10 - loss: 1.1204 - categorical_accuracy: 0.6361
 2112/60000 [>.............................] - ETA: 2:10 - loss: 1.1156 - categorical_accuracy: 0.6378
 2144/60000 [>.............................] - ETA: 2:10 - loss: 1.1057 - categorical_accuracy: 0.6413
 2176/60000 [>.............................] - ETA: 2:09 - loss: 1.0999 - categorical_accuracy: 0.6434
 2208/60000 [>.............................] - ETA: 2:09 - loss: 1.0932 - categorical_accuracy: 0.6454
 2240/60000 [>.............................] - ETA: 2:09 - loss: 1.0918 - categorical_accuracy: 0.6473
 2272/60000 [>.............................] - ETA: 2:09 - loss: 1.0857 - categorical_accuracy: 0.6492
 2304/60000 [>.............................] - ETA: 2:09 - loss: 1.0820 - categorical_accuracy: 0.6515
 2336/60000 [>.............................] - ETA: 2:08 - loss: 1.0759 - categorical_accuracy: 0.6533
 2368/60000 [>.............................] - ETA: 2:08 - loss: 1.0659 - categorical_accuracy: 0.6562
 2400/60000 [>.............................] - ETA: 2:08 - loss: 1.0552 - categorical_accuracy: 0.6600
 2432/60000 [>.............................] - ETA: 2:08 - loss: 1.0483 - categorical_accuracy: 0.6624
 2464/60000 [>.............................] - ETA: 2:08 - loss: 1.0430 - categorical_accuracy: 0.6631
 2496/60000 [>.............................] - ETA: 2:08 - loss: 1.0342 - categorical_accuracy: 0.6659
 2528/60000 [>.............................] - ETA: 2:08 - loss: 1.0269 - categorical_accuracy: 0.6673
 2560/60000 [>.............................] - ETA: 2:07 - loss: 1.0206 - categorical_accuracy: 0.6695
 2592/60000 [>.............................] - ETA: 2:07 - loss: 1.0128 - categorical_accuracy: 0.6713
 2624/60000 [>.............................] - ETA: 2:07 - loss: 1.0063 - categorical_accuracy: 0.6738
 2656/60000 [>.............................] - ETA: 2:07 - loss: 0.9990 - categorical_accuracy: 0.6762
 2688/60000 [>.............................] - ETA: 2:07 - loss: 0.9916 - categorical_accuracy: 0.6782
 2720/60000 [>.............................] - ETA: 2:06 - loss: 0.9820 - categorical_accuracy: 0.6809
 2752/60000 [>.............................] - ETA: 2:06 - loss: 0.9765 - categorical_accuracy: 0.6820
 2784/60000 [>.............................] - ETA: 2:06 - loss: 0.9720 - categorical_accuracy: 0.6835
 2816/60000 [>.............................] - ETA: 2:06 - loss: 0.9688 - categorical_accuracy: 0.6839
 2848/60000 [>.............................] - ETA: 2:06 - loss: 0.9623 - categorical_accuracy: 0.6861
 2880/60000 [>.............................] - ETA: 2:06 - loss: 0.9568 - categorical_accuracy: 0.6875
 2912/60000 [>.............................] - ETA: 2:06 - loss: 0.9506 - categorical_accuracy: 0.6892
 2944/60000 [>.............................] - ETA: 2:06 - loss: 0.9455 - categorical_accuracy: 0.6909
 2976/60000 [>.............................] - ETA: 2:06 - loss: 0.9383 - categorical_accuracy: 0.6932
 3008/60000 [>.............................] - ETA: 2:05 - loss: 0.9334 - categorical_accuracy: 0.6941
 3040/60000 [>.............................] - ETA: 2:05 - loss: 0.9281 - categorical_accuracy: 0.6961
 3072/60000 [>.............................] - ETA: 2:05 - loss: 0.9239 - categorical_accuracy: 0.6969
 3104/60000 [>.............................] - ETA: 2:05 - loss: 0.9181 - categorical_accuracy: 0.6994
 3136/60000 [>.............................] - ETA: 2:05 - loss: 0.9158 - categorical_accuracy: 0.7009
 3168/60000 [>.............................] - ETA: 2:05 - loss: 0.9100 - categorical_accuracy: 0.7027
 3200/60000 [>.............................] - ETA: 2:04 - loss: 0.9031 - categorical_accuracy: 0.7050
 3232/60000 [>.............................] - ETA: 2:04 - loss: 0.8970 - categorical_accuracy: 0.7070
 3264/60000 [>.............................] - ETA: 2:04 - loss: 0.8922 - categorical_accuracy: 0.7089
 3296/60000 [>.............................] - ETA: 2:04 - loss: 0.8915 - categorical_accuracy: 0.7100
 3328/60000 [>.............................] - ETA: 2:04 - loss: 0.8885 - categorical_accuracy: 0.7109
 3360/60000 [>.............................] - ETA: 2:03 - loss: 0.8862 - categorical_accuracy: 0.7119
 3392/60000 [>.............................] - ETA: 2:03 - loss: 0.8810 - categorical_accuracy: 0.7134
 3424/60000 [>.............................] - ETA: 2:03 - loss: 0.8753 - categorical_accuracy: 0.7155
 3456/60000 [>.............................] - ETA: 2:03 - loss: 0.8716 - categorical_accuracy: 0.7164
 3488/60000 [>.............................] - ETA: 2:03 - loss: 0.8665 - categorical_accuracy: 0.7182
 3520/60000 [>.............................] - ETA: 2:03 - loss: 0.8631 - categorical_accuracy: 0.7193
 3552/60000 [>.............................] - ETA: 2:03 - loss: 0.8583 - categorical_accuracy: 0.7213
 3584/60000 [>.............................] - ETA: 2:03 - loss: 0.8531 - categorical_accuracy: 0.7229
 3616/60000 [>.............................] - ETA: 2:02 - loss: 0.8493 - categorical_accuracy: 0.7243
 3648/60000 [>.............................] - ETA: 2:02 - loss: 0.8445 - categorical_accuracy: 0.7256
 3680/60000 [>.............................] - ETA: 2:02 - loss: 0.8404 - categorical_accuracy: 0.7266
 3712/60000 [>.............................] - ETA: 2:02 - loss: 0.8355 - categorical_accuracy: 0.7284
 3744/60000 [>.............................] - ETA: 2:02 - loss: 0.8328 - categorical_accuracy: 0.7297
 3776/60000 [>.............................] - ETA: 2:01 - loss: 0.8270 - categorical_accuracy: 0.7317
 3808/60000 [>.............................] - ETA: 2:01 - loss: 0.8247 - categorical_accuracy: 0.7321
 3840/60000 [>.............................] - ETA: 2:01 - loss: 0.8206 - categorical_accuracy: 0.7336
 3872/60000 [>.............................] - ETA: 2:01 - loss: 0.8147 - categorical_accuracy: 0.7355
 3904/60000 [>.............................] - ETA: 2:01 - loss: 0.8150 - categorical_accuracy: 0.7367
 3936/60000 [>.............................] - ETA: 2:01 - loss: 0.8103 - categorical_accuracy: 0.7381
 3968/60000 [>.............................] - ETA: 2:01 - loss: 0.8069 - categorical_accuracy: 0.7394
 4000/60000 [=>............................] - ETA: 2:01 - loss: 0.8031 - categorical_accuracy: 0.7402
 4032/60000 [=>............................] - ETA: 2:01 - loss: 0.8013 - categorical_accuracy: 0.7406
 4064/60000 [=>............................] - ETA: 2:01 - loss: 0.7982 - categorical_accuracy: 0.7419
 4096/60000 [=>............................] - ETA: 2:01 - loss: 0.7941 - categorical_accuracy: 0.7429
 4128/60000 [=>............................] - ETA: 2:01 - loss: 0.7896 - categorical_accuracy: 0.7447
 4160/60000 [=>............................] - ETA: 2:00 - loss: 0.7877 - categorical_accuracy: 0.7452
 4192/60000 [=>............................] - ETA: 2:00 - loss: 0.7840 - categorical_accuracy: 0.7467
 4224/60000 [=>............................] - ETA: 2:00 - loss: 0.7800 - categorical_accuracy: 0.7479
 4256/60000 [=>............................] - ETA: 2:00 - loss: 0.7761 - categorical_accuracy: 0.7493
 4288/60000 [=>............................] - ETA: 2:00 - loss: 0.7719 - categorical_accuracy: 0.7507
 4320/60000 [=>............................] - ETA: 2:00 - loss: 0.7673 - categorical_accuracy: 0.7523
 4352/60000 [=>............................] - ETA: 2:00 - loss: 0.7641 - categorical_accuracy: 0.7532
 4384/60000 [=>............................] - ETA: 2:00 - loss: 0.7602 - categorical_accuracy: 0.7546
 4416/60000 [=>............................] - ETA: 1:59 - loss: 0.7562 - categorical_accuracy: 0.7559
 4448/60000 [=>............................] - ETA: 1:59 - loss: 0.7518 - categorical_accuracy: 0.7572
 4480/60000 [=>............................] - ETA: 1:59 - loss: 0.7492 - categorical_accuracy: 0.7587
 4512/60000 [=>............................] - ETA: 1:59 - loss: 0.7472 - categorical_accuracy: 0.7600
 4544/60000 [=>............................] - ETA: 1:59 - loss: 0.7447 - categorical_accuracy: 0.7610
 4576/60000 [=>............................] - ETA: 1:59 - loss: 0.7408 - categorical_accuracy: 0.7625
 4608/60000 [=>............................] - ETA: 1:59 - loss: 0.7377 - categorical_accuracy: 0.7635
 4640/60000 [=>............................] - ETA: 1:59 - loss: 0.7345 - categorical_accuracy: 0.7644
 4672/60000 [=>............................] - ETA: 1:58 - loss: 0.7330 - categorical_accuracy: 0.7650
 4704/60000 [=>............................] - ETA: 1:58 - loss: 0.7315 - categorical_accuracy: 0.7657
 4736/60000 [=>............................] - ETA: 1:58 - loss: 0.7287 - categorical_accuracy: 0.7667
 4768/60000 [=>............................] - ETA: 1:58 - loss: 0.7274 - categorical_accuracy: 0.7670
 4800/60000 [=>............................] - ETA: 1:58 - loss: 0.7264 - categorical_accuracy: 0.7675
 4832/60000 [=>............................] - ETA: 1:58 - loss: 0.7241 - categorical_accuracy: 0.7682
 4864/60000 [=>............................] - ETA: 1:58 - loss: 0.7207 - categorical_accuracy: 0.7695
 4896/60000 [=>............................] - ETA: 1:58 - loss: 0.7192 - categorical_accuracy: 0.7698
 4928/60000 [=>............................] - ETA: 1:58 - loss: 0.7158 - categorical_accuracy: 0.7707
 4960/60000 [=>............................] - ETA: 1:58 - loss: 0.7125 - categorical_accuracy: 0.7718
 4992/60000 [=>............................] - ETA: 1:58 - loss: 0.7081 - categorical_accuracy: 0.7732
 5024/60000 [=>............................] - ETA: 1:57 - loss: 0.7057 - categorical_accuracy: 0.7739
 5056/60000 [=>............................] - ETA: 1:57 - loss: 0.7031 - categorical_accuracy: 0.7747
 5088/60000 [=>............................] - ETA: 1:57 - loss: 0.6992 - categorical_accuracy: 0.7761
 5120/60000 [=>............................] - ETA: 1:57 - loss: 0.6969 - categorical_accuracy: 0.7768
 5152/60000 [=>............................] - ETA: 1:57 - loss: 0.6937 - categorical_accuracy: 0.7780
 5184/60000 [=>............................] - ETA: 1:57 - loss: 0.6911 - categorical_accuracy: 0.7791
 5216/60000 [=>............................] - ETA: 1:57 - loss: 0.6907 - categorical_accuracy: 0.7797
 5248/60000 [=>............................] - ETA: 1:57 - loss: 0.6890 - categorical_accuracy: 0.7803
 5280/60000 [=>............................] - ETA: 1:57 - loss: 0.6874 - categorical_accuracy: 0.7811
 5312/60000 [=>............................] - ETA: 1:57 - loss: 0.6850 - categorical_accuracy: 0.7820
 5344/60000 [=>............................] - ETA: 1:57 - loss: 0.6818 - categorical_accuracy: 0.7831
 5376/60000 [=>............................] - ETA: 1:56 - loss: 0.6788 - categorical_accuracy: 0.7840
 5408/60000 [=>............................] - ETA: 1:56 - loss: 0.6757 - categorical_accuracy: 0.7851
 5440/60000 [=>............................] - ETA: 1:56 - loss: 0.6745 - categorical_accuracy: 0.7855
 5472/60000 [=>............................] - ETA: 1:56 - loss: 0.6729 - categorical_accuracy: 0.7858
 5504/60000 [=>............................] - ETA: 1:56 - loss: 0.6718 - categorical_accuracy: 0.7863
 5536/60000 [=>............................] - ETA: 1:56 - loss: 0.6705 - categorical_accuracy: 0.7867
 5568/60000 [=>............................] - ETA: 1:56 - loss: 0.6671 - categorical_accuracy: 0.7877
 5600/60000 [=>............................] - ETA: 1:56 - loss: 0.6642 - categorical_accuracy: 0.7887
 5632/60000 [=>............................] - ETA: 1:56 - loss: 0.6626 - categorical_accuracy: 0.7896
 5664/60000 [=>............................] - ETA: 1:56 - loss: 0.6607 - categorical_accuracy: 0.7901
 5696/60000 [=>............................] - ETA: 1:56 - loss: 0.6582 - categorical_accuracy: 0.7911
 5728/60000 [=>............................] - ETA: 1:55 - loss: 0.6556 - categorical_accuracy: 0.7917
 5760/60000 [=>............................] - ETA: 1:55 - loss: 0.6538 - categorical_accuracy: 0.7922
 5792/60000 [=>............................] - ETA: 1:55 - loss: 0.6519 - categorical_accuracy: 0.7928
 5824/60000 [=>............................] - ETA: 1:55 - loss: 0.6496 - categorical_accuracy: 0.7934
 5856/60000 [=>............................] - ETA: 1:55 - loss: 0.6466 - categorical_accuracy: 0.7944
 5888/60000 [=>............................] - ETA: 1:55 - loss: 0.6440 - categorical_accuracy: 0.7953
 5920/60000 [=>............................] - ETA: 1:55 - loss: 0.6450 - categorical_accuracy: 0.7958
 5952/60000 [=>............................] - ETA: 1:55 - loss: 0.6427 - categorical_accuracy: 0.7967
 5984/60000 [=>............................] - ETA: 1:55 - loss: 0.6403 - categorical_accuracy: 0.7976
 6016/60000 [==>...........................] - ETA: 1:55 - loss: 0.6390 - categorical_accuracy: 0.7980
 6048/60000 [==>...........................] - ETA: 1:55 - loss: 0.6366 - categorical_accuracy: 0.7988
 6080/60000 [==>...........................] - ETA: 1:55 - loss: 0.6348 - categorical_accuracy: 0.7990
 6112/60000 [==>...........................] - ETA: 1:54 - loss: 0.6325 - categorical_accuracy: 0.7999
 6144/60000 [==>...........................] - ETA: 1:54 - loss: 0.6302 - categorical_accuracy: 0.8005
 6176/60000 [==>...........................] - ETA: 1:54 - loss: 0.6280 - categorical_accuracy: 0.8013
 6208/60000 [==>...........................] - ETA: 1:54 - loss: 0.6261 - categorical_accuracy: 0.8019
 6240/60000 [==>...........................] - ETA: 1:54 - loss: 0.6244 - categorical_accuracy: 0.8024
 6272/60000 [==>...........................] - ETA: 1:54 - loss: 0.6245 - categorical_accuracy: 0.8025
 6304/60000 [==>...........................] - ETA: 1:54 - loss: 0.6221 - categorical_accuracy: 0.8033
 6336/60000 [==>...........................] - ETA: 1:54 - loss: 0.6194 - categorical_accuracy: 0.8043
 6368/60000 [==>...........................] - ETA: 1:54 - loss: 0.6175 - categorical_accuracy: 0.8048
 6400/60000 [==>...........................] - ETA: 1:54 - loss: 0.6155 - categorical_accuracy: 0.8053
 6432/60000 [==>...........................] - ETA: 1:54 - loss: 0.6155 - categorical_accuracy: 0.8055
 6464/60000 [==>...........................] - ETA: 1:54 - loss: 0.6142 - categorical_accuracy: 0.8060
 6496/60000 [==>...........................] - ETA: 1:54 - loss: 0.6141 - categorical_accuracy: 0.8062
 6528/60000 [==>...........................] - ETA: 1:54 - loss: 0.6120 - categorical_accuracy: 0.8067
 6560/60000 [==>...........................] - ETA: 1:54 - loss: 0.6095 - categorical_accuracy: 0.8075
 6592/60000 [==>...........................] - ETA: 1:54 - loss: 0.6080 - categorical_accuracy: 0.8078
 6624/60000 [==>...........................] - ETA: 1:53 - loss: 0.6058 - categorical_accuracy: 0.8087
 6656/60000 [==>...........................] - ETA: 1:53 - loss: 0.6044 - categorical_accuracy: 0.8089
 6688/60000 [==>...........................] - ETA: 1:53 - loss: 0.6034 - categorical_accuracy: 0.8092
 6720/60000 [==>...........................] - ETA: 1:53 - loss: 0.6018 - categorical_accuracy: 0.8095
 6752/60000 [==>...........................] - ETA: 1:53 - loss: 0.6010 - categorical_accuracy: 0.8098
 6784/60000 [==>...........................] - ETA: 1:53 - loss: 0.5991 - categorical_accuracy: 0.8106
 6816/60000 [==>...........................] - ETA: 1:53 - loss: 0.5966 - categorical_accuracy: 0.8115
 6848/60000 [==>...........................] - ETA: 1:53 - loss: 0.5948 - categorical_accuracy: 0.8121
 6880/60000 [==>...........................] - ETA: 1:53 - loss: 0.5940 - categorical_accuracy: 0.8122
 6912/60000 [==>...........................] - ETA: 1:53 - loss: 0.5924 - categorical_accuracy: 0.8129
 6944/60000 [==>...........................] - ETA: 1:53 - loss: 0.5909 - categorical_accuracy: 0.8132
 6976/60000 [==>...........................] - ETA: 1:53 - loss: 0.5888 - categorical_accuracy: 0.8139
 7008/60000 [==>...........................] - ETA: 1:52 - loss: 0.5878 - categorical_accuracy: 0.8145
 7040/60000 [==>...........................] - ETA: 1:52 - loss: 0.5865 - categorical_accuracy: 0.8148
 7072/60000 [==>...........................] - ETA: 1:52 - loss: 0.5852 - categorical_accuracy: 0.8152
 7104/60000 [==>...........................] - ETA: 1:52 - loss: 0.5831 - categorical_accuracy: 0.8159
 7136/60000 [==>...........................] - ETA: 1:52 - loss: 0.5814 - categorical_accuracy: 0.8166
 7168/60000 [==>...........................] - ETA: 1:52 - loss: 0.5793 - categorical_accuracy: 0.8171
 7200/60000 [==>...........................] - ETA: 1:52 - loss: 0.5772 - categorical_accuracy: 0.8179
 7232/60000 [==>...........................] - ETA: 1:52 - loss: 0.5759 - categorical_accuracy: 0.8182
 7264/60000 [==>...........................] - ETA: 1:52 - loss: 0.5754 - categorical_accuracy: 0.8184
 7296/60000 [==>...........................] - ETA: 1:52 - loss: 0.5740 - categorical_accuracy: 0.8189
 7328/60000 [==>...........................] - ETA: 1:52 - loss: 0.5723 - categorical_accuracy: 0.8196
 7360/60000 [==>...........................] - ETA: 1:51 - loss: 0.5704 - categorical_accuracy: 0.8201
 7392/60000 [==>...........................] - ETA: 1:51 - loss: 0.5683 - categorical_accuracy: 0.8208
 7424/60000 [==>...........................] - ETA: 1:51 - loss: 0.5667 - categorical_accuracy: 0.8210
 7456/60000 [==>...........................] - ETA: 1:51 - loss: 0.5662 - categorical_accuracy: 0.8209
 7488/60000 [==>...........................] - ETA: 1:51 - loss: 0.5642 - categorical_accuracy: 0.8216
 7520/60000 [==>...........................] - ETA: 1:51 - loss: 0.5624 - categorical_accuracy: 0.8222
 7552/60000 [==>...........................] - ETA: 1:51 - loss: 0.5616 - categorical_accuracy: 0.8226
 7584/60000 [==>...........................] - ETA: 1:51 - loss: 0.5604 - categorical_accuracy: 0.8230
 7616/60000 [==>...........................] - ETA: 1:51 - loss: 0.5585 - categorical_accuracy: 0.8235
 7648/60000 [==>...........................] - ETA: 1:51 - loss: 0.5581 - categorical_accuracy: 0.8235
 7680/60000 [==>...........................] - ETA: 1:51 - loss: 0.5564 - categorical_accuracy: 0.8241
 7712/60000 [==>...........................] - ETA: 1:51 - loss: 0.5549 - categorical_accuracy: 0.8246
 7744/60000 [==>...........................] - ETA: 1:51 - loss: 0.5532 - categorical_accuracy: 0.8249
 7776/60000 [==>...........................] - ETA: 1:51 - loss: 0.5521 - categorical_accuracy: 0.8254
 7808/60000 [==>...........................] - ETA: 1:51 - loss: 0.5508 - categorical_accuracy: 0.8258
 7840/60000 [==>...........................] - ETA: 1:51 - loss: 0.5492 - categorical_accuracy: 0.8261
 7872/60000 [==>...........................] - ETA: 1:51 - loss: 0.5478 - categorical_accuracy: 0.8265
 7904/60000 [==>...........................] - ETA: 1:51 - loss: 0.5468 - categorical_accuracy: 0.8268
 7936/60000 [==>...........................] - ETA: 1:51 - loss: 0.5457 - categorical_accuracy: 0.8271
 7968/60000 [==>...........................] - ETA: 1:51 - loss: 0.5448 - categorical_accuracy: 0.8273
 8000/60000 [===>..........................] - ETA: 1:51 - loss: 0.5432 - categorical_accuracy: 0.8278
 8032/60000 [===>..........................] - ETA: 1:51 - loss: 0.5423 - categorical_accuracy: 0.8279
 8064/60000 [===>..........................] - ETA: 1:51 - loss: 0.5405 - categorical_accuracy: 0.8286
 8096/60000 [===>..........................] - ETA: 1:50 - loss: 0.5385 - categorical_accuracy: 0.8293
 8128/60000 [===>..........................] - ETA: 1:50 - loss: 0.5368 - categorical_accuracy: 0.8298
 8160/60000 [===>..........................] - ETA: 1:50 - loss: 0.5361 - categorical_accuracy: 0.8303
 8192/60000 [===>..........................] - ETA: 1:50 - loss: 0.5352 - categorical_accuracy: 0.8306
 8224/60000 [===>..........................] - ETA: 1:50 - loss: 0.5335 - categorical_accuracy: 0.8310
 8256/60000 [===>..........................] - ETA: 1:50 - loss: 0.5319 - categorical_accuracy: 0.8316
 8288/60000 [===>..........................] - ETA: 1:50 - loss: 0.5304 - categorical_accuracy: 0.8322
 8320/60000 [===>..........................] - ETA: 1:50 - loss: 0.5287 - categorical_accuracy: 0.8328
 8352/60000 [===>..........................] - ETA: 1:50 - loss: 0.5275 - categorical_accuracy: 0.8330
 8384/60000 [===>..........................] - ETA: 1:50 - loss: 0.5267 - categorical_accuracy: 0.8330
 8416/60000 [===>..........................] - ETA: 1:50 - loss: 0.5267 - categorical_accuracy: 0.8333
 8448/60000 [===>..........................] - ETA: 1:50 - loss: 0.5256 - categorical_accuracy: 0.8336
 8480/60000 [===>..........................] - ETA: 1:50 - loss: 0.5248 - categorical_accuracy: 0.8337
 8512/60000 [===>..........................] - ETA: 1:49 - loss: 0.5237 - categorical_accuracy: 0.8341
 8544/60000 [===>..........................] - ETA: 1:49 - loss: 0.5221 - categorical_accuracy: 0.8346
 8576/60000 [===>..........................] - ETA: 1:49 - loss: 0.5212 - categorical_accuracy: 0.8348
 8608/60000 [===>..........................] - ETA: 1:49 - loss: 0.5196 - categorical_accuracy: 0.8354
 8640/60000 [===>..........................] - ETA: 1:49 - loss: 0.5193 - categorical_accuracy: 0.8356
 8672/60000 [===>..........................] - ETA: 1:49 - loss: 0.5186 - categorical_accuracy: 0.8359
 8704/60000 [===>..........................] - ETA: 1:49 - loss: 0.5172 - categorical_accuracy: 0.8364
 8736/60000 [===>..........................] - ETA: 1:49 - loss: 0.5163 - categorical_accuracy: 0.8367
 8768/60000 [===>..........................] - ETA: 1:49 - loss: 0.5155 - categorical_accuracy: 0.8368
 8800/60000 [===>..........................] - ETA: 1:49 - loss: 0.5144 - categorical_accuracy: 0.8373
 8832/60000 [===>..........................] - ETA: 1:49 - loss: 0.5137 - categorical_accuracy: 0.8376
 8864/60000 [===>..........................] - ETA: 1:49 - loss: 0.5126 - categorical_accuracy: 0.8379
 8896/60000 [===>..........................] - ETA: 1:49 - loss: 0.5113 - categorical_accuracy: 0.8382
 8928/60000 [===>..........................] - ETA: 1:49 - loss: 0.5108 - categorical_accuracy: 0.8386
 8960/60000 [===>..........................] - ETA: 1:48 - loss: 0.5092 - categorical_accuracy: 0.8391
 8992/60000 [===>..........................] - ETA: 1:48 - loss: 0.5078 - categorical_accuracy: 0.8395
 9024/60000 [===>..........................] - ETA: 1:48 - loss: 0.5069 - categorical_accuracy: 0.8399
 9056/60000 [===>..........................] - ETA: 1:48 - loss: 0.5056 - categorical_accuracy: 0.8403
 9088/60000 [===>..........................] - ETA: 1:48 - loss: 0.5041 - categorical_accuracy: 0.8409
 9120/60000 [===>..........................] - ETA: 1:48 - loss: 0.5029 - categorical_accuracy: 0.8412
 9152/60000 [===>..........................] - ETA: 1:48 - loss: 0.5015 - categorical_accuracy: 0.8418
 9184/60000 [===>..........................] - ETA: 1:48 - loss: 0.5007 - categorical_accuracy: 0.8419
 9216/60000 [===>..........................] - ETA: 1:48 - loss: 0.5000 - categorical_accuracy: 0.8420
 9248/60000 [===>..........................] - ETA: 1:48 - loss: 0.4985 - categorical_accuracy: 0.8426
 9280/60000 [===>..........................] - ETA: 1:48 - loss: 0.4978 - categorical_accuracy: 0.8429
 9312/60000 [===>..........................] - ETA: 1:48 - loss: 0.4964 - categorical_accuracy: 0.8432
 9344/60000 [===>..........................] - ETA: 1:48 - loss: 0.4953 - categorical_accuracy: 0.8436
 9376/60000 [===>..........................] - ETA: 1:48 - loss: 0.4944 - categorical_accuracy: 0.8439
 9408/60000 [===>..........................] - ETA: 1:48 - loss: 0.4929 - categorical_accuracy: 0.8444
 9440/60000 [===>..........................] - ETA: 1:48 - loss: 0.4915 - categorical_accuracy: 0.8449
 9472/60000 [===>..........................] - ETA: 1:48 - loss: 0.4900 - categorical_accuracy: 0.8454
 9504/60000 [===>..........................] - ETA: 1:47 - loss: 0.4887 - categorical_accuracy: 0.8457
 9536/60000 [===>..........................] - ETA: 1:47 - loss: 0.4887 - categorical_accuracy: 0.8457
 9568/60000 [===>..........................] - ETA: 1:47 - loss: 0.4873 - categorical_accuracy: 0.8463
 9600/60000 [===>..........................] - ETA: 1:47 - loss: 0.4869 - categorical_accuracy: 0.8466
 9632/60000 [===>..........................] - ETA: 1:47 - loss: 0.4861 - categorical_accuracy: 0.8468
 9664/60000 [===>..........................] - ETA: 1:47 - loss: 0.4850 - categorical_accuracy: 0.8471
 9696/60000 [===>..........................] - ETA: 1:47 - loss: 0.4839 - categorical_accuracy: 0.8474
 9728/60000 [===>..........................] - ETA: 1:47 - loss: 0.4827 - categorical_accuracy: 0.8478
 9760/60000 [===>..........................] - ETA: 1:47 - loss: 0.4815 - categorical_accuracy: 0.8482
 9792/60000 [===>..........................] - ETA: 1:47 - loss: 0.4806 - categorical_accuracy: 0.8483
 9824/60000 [===>..........................] - ETA: 1:47 - loss: 0.4804 - categorical_accuracy: 0.8482
 9856/60000 [===>..........................] - ETA: 1:47 - loss: 0.4803 - categorical_accuracy: 0.8482
 9888/60000 [===>..........................] - ETA: 1:47 - loss: 0.4791 - categorical_accuracy: 0.8485
 9920/60000 [===>..........................] - ETA: 1:47 - loss: 0.4787 - categorical_accuracy: 0.8486
 9952/60000 [===>..........................] - ETA: 1:47 - loss: 0.4777 - categorical_accuracy: 0.8489
 9984/60000 [===>..........................] - ETA: 1:47 - loss: 0.4768 - categorical_accuracy: 0.8491
10016/60000 [====>.........................] - ETA: 1:47 - loss: 0.4762 - categorical_accuracy: 0.8492
10048/60000 [====>.........................] - ETA: 1:47 - loss: 0.4754 - categorical_accuracy: 0.8496
10080/60000 [====>.........................] - ETA: 1:46 - loss: 0.4751 - categorical_accuracy: 0.8498
10112/60000 [====>.........................] - ETA: 1:46 - loss: 0.4743 - categorical_accuracy: 0.8501
10144/60000 [====>.........................] - ETA: 1:46 - loss: 0.4742 - categorical_accuracy: 0.8503
10176/60000 [====>.........................] - ETA: 1:46 - loss: 0.4730 - categorical_accuracy: 0.8506
10208/60000 [====>.........................] - ETA: 1:46 - loss: 0.4736 - categorical_accuracy: 0.8506
10240/60000 [====>.........................] - ETA: 1:46 - loss: 0.4732 - categorical_accuracy: 0.8507
10272/60000 [====>.........................] - ETA: 1:46 - loss: 0.4723 - categorical_accuracy: 0.8509
10304/60000 [====>.........................] - ETA: 1:46 - loss: 0.4716 - categorical_accuracy: 0.8511
10336/60000 [====>.........................] - ETA: 1:46 - loss: 0.4708 - categorical_accuracy: 0.8514
10368/60000 [====>.........................] - ETA: 1:46 - loss: 0.4702 - categorical_accuracy: 0.8518
10400/60000 [====>.........................] - ETA: 1:46 - loss: 0.4690 - categorical_accuracy: 0.8521
10432/60000 [====>.........................] - ETA: 1:46 - loss: 0.4682 - categorical_accuracy: 0.8523
10464/60000 [====>.........................] - ETA: 1:45 - loss: 0.4678 - categorical_accuracy: 0.8525
10496/60000 [====>.........................] - ETA: 1:45 - loss: 0.4671 - categorical_accuracy: 0.8528
10528/60000 [====>.........................] - ETA: 1:45 - loss: 0.4663 - categorical_accuracy: 0.8531
10560/60000 [====>.........................] - ETA: 1:45 - loss: 0.4666 - categorical_accuracy: 0.8529
10592/60000 [====>.........................] - ETA: 1:45 - loss: 0.4655 - categorical_accuracy: 0.8534
10624/60000 [====>.........................] - ETA: 1:45 - loss: 0.4648 - categorical_accuracy: 0.8536
10656/60000 [====>.........................] - ETA: 1:45 - loss: 0.4637 - categorical_accuracy: 0.8540
10688/60000 [====>.........................] - ETA: 1:45 - loss: 0.4630 - categorical_accuracy: 0.8542
10720/60000 [====>.........................] - ETA: 1:45 - loss: 0.4616 - categorical_accuracy: 0.8547
10752/60000 [====>.........................] - ETA: 1:45 - loss: 0.4606 - categorical_accuracy: 0.8550
10784/60000 [====>.........................] - ETA: 1:45 - loss: 0.4601 - categorical_accuracy: 0.8551
10816/60000 [====>.........................] - ETA: 1:45 - loss: 0.4590 - categorical_accuracy: 0.8555
10848/60000 [====>.........................] - ETA: 1:45 - loss: 0.4593 - categorical_accuracy: 0.8556
10880/60000 [====>.........................] - ETA: 1:45 - loss: 0.4585 - categorical_accuracy: 0.8559
10912/60000 [====>.........................] - ETA: 1:45 - loss: 0.4577 - categorical_accuracy: 0.8560
10944/60000 [====>.........................] - ETA: 1:45 - loss: 0.4566 - categorical_accuracy: 0.8563
10976/60000 [====>.........................] - ETA: 1:44 - loss: 0.4556 - categorical_accuracy: 0.8566
11008/60000 [====>.........................] - ETA: 1:44 - loss: 0.4550 - categorical_accuracy: 0.8569
11040/60000 [====>.........................] - ETA: 1:44 - loss: 0.4544 - categorical_accuracy: 0.8571
11072/60000 [====>.........................] - ETA: 1:44 - loss: 0.4537 - categorical_accuracy: 0.8573
11104/60000 [====>.........................] - ETA: 1:44 - loss: 0.4525 - categorical_accuracy: 0.8577
11136/60000 [====>.........................] - ETA: 1:44 - loss: 0.4521 - categorical_accuracy: 0.8578
11168/60000 [====>.........................] - ETA: 1:44 - loss: 0.4509 - categorical_accuracy: 0.8582
11200/60000 [====>.........................] - ETA: 1:44 - loss: 0.4501 - categorical_accuracy: 0.8583
11232/60000 [====>.........................] - ETA: 1:44 - loss: 0.4498 - categorical_accuracy: 0.8584
11264/60000 [====>.........................] - ETA: 1:44 - loss: 0.4490 - categorical_accuracy: 0.8587
11296/60000 [====>.........................] - ETA: 1:44 - loss: 0.4486 - categorical_accuracy: 0.8587
11328/60000 [====>.........................] - ETA: 1:44 - loss: 0.4479 - categorical_accuracy: 0.8589
11360/60000 [====>.........................] - ETA: 1:44 - loss: 0.4473 - categorical_accuracy: 0.8591
11392/60000 [====>.........................] - ETA: 1:44 - loss: 0.4473 - categorical_accuracy: 0.8591
11424/60000 [====>.........................] - ETA: 1:43 - loss: 0.4466 - categorical_accuracy: 0.8593
11456/60000 [====>.........................] - ETA: 1:43 - loss: 0.4458 - categorical_accuracy: 0.8595
11488/60000 [====>.........................] - ETA: 1:43 - loss: 0.4448 - categorical_accuracy: 0.8598
11520/60000 [====>.........................] - ETA: 1:43 - loss: 0.4441 - categorical_accuracy: 0.8600
11552/60000 [====>.........................] - ETA: 1:43 - loss: 0.4442 - categorical_accuracy: 0.8601
11584/60000 [====>.........................] - ETA: 1:43 - loss: 0.4436 - categorical_accuracy: 0.8602
11616/60000 [====>.........................] - ETA: 1:43 - loss: 0.4427 - categorical_accuracy: 0.8605
11648/60000 [====>.........................] - ETA: 1:43 - loss: 0.4429 - categorical_accuracy: 0.8607
11680/60000 [====>.........................] - ETA: 1:43 - loss: 0.4418 - categorical_accuracy: 0.8610
11712/60000 [====>.........................] - ETA: 1:43 - loss: 0.4409 - categorical_accuracy: 0.8613
11744/60000 [====>.........................] - ETA: 1:43 - loss: 0.4399 - categorical_accuracy: 0.8617
11776/60000 [====>.........................] - ETA: 1:43 - loss: 0.4390 - categorical_accuracy: 0.8620
11808/60000 [====>.........................] - ETA: 1:43 - loss: 0.4379 - categorical_accuracy: 0.8624
11840/60000 [====>.........................] - ETA: 1:43 - loss: 0.4371 - categorical_accuracy: 0.8627
11872/60000 [====>.........................] - ETA: 1:43 - loss: 0.4365 - categorical_accuracy: 0.8628
11904/60000 [====>.........................] - ETA: 1:42 - loss: 0.4368 - categorical_accuracy: 0.8629
11936/60000 [====>.........................] - ETA: 1:42 - loss: 0.4371 - categorical_accuracy: 0.8629
11968/60000 [====>.........................] - ETA: 1:42 - loss: 0.4366 - categorical_accuracy: 0.8631
12000/60000 [=====>........................] - ETA: 1:42 - loss: 0.4361 - categorical_accuracy: 0.8632
12032/60000 [=====>........................] - ETA: 1:42 - loss: 0.4365 - categorical_accuracy: 0.8631
12064/60000 [=====>........................] - ETA: 1:42 - loss: 0.4361 - categorical_accuracy: 0.8633
12096/60000 [=====>........................] - ETA: 1:42 - loss: 0.4357 - categorical_accuracy: 0.8635
12128/60000 [=====>........................] - ETA: 1:42 - loss: 0.4351 - categorical_accuracy: 0.8636
12160/60000 [=====>........................] - ETA: 1:42 - loss: 0.4342 - categorical_accuracy: 0.8639
12192/60000 [=====>........................] - ETA: 1:42 - loss: 0.4333 - categorical_accuracy: 0.8643
12224/60000 [=====>........................] - ETA: 1:42 - loss: 0.4332 - categorical_accuracy: 0.8642
12256/60000 [=====>........................] - ETA: 1:42 - loss: 0.4328 - categorical_accuracy: 0.8643
12288/60000 [=====>........................] - ETA: 1:42 - loss: 0.4323 - categorical_accuracy: 0.8645
12320/60000 [=====>........................] - ETA: 1:42 - loss: 0.4319 - categorical_accuracy: 0.8647
12352/60000 [=====>........................] - ETA: 1:42 - loss: 0.4317 - categorical_accuracy: 0.8649
12384/60000 [=====>........................] - ETA: 1:42 - loss: 0.4315 - categorical_accuracy: 0.8650
12416/60000 [=====>........................] - ETA: 1:42 - loss: 0.4305 - categorical_accuracy: 0.8653
12448/60000 [=====>........................] - ETA: 1:42 - loss: 0.4296 - categorical_accuracy: 0.8657
12480/60000 [=====>........................] - ETA: 1:42 - loss: 0.4292 - categorical_accuracy: 0.8659
12512/60000 [=====>........................] - ETA: 1:42 - loss: 0.4284 - categorical_accuracy: 0.8662
12544/60000 [=====>........................] - ETA: 1:42 - loss: 0.4278 - categorical_accuracy: 0.8664
12576/60000 [=====>........................] - ETA: 1:42 - loss: 0.4270 - categorical_accuracy: 0.8667
12608/60000 [=====>........................] - ETA: 1:42 - loss: 0.4264 - categorical_accuracy: 0.8668
12640/60000 [=====>........................] - ETA: 1:42 - loss: 0.4257 - categorical_accuracy: 0.8671
12672/60000 [=====>........................] - ETA: 1:42 - loss: 0.4254 - categorical_accuracy: 0.8673
12704/60000 [=====>........................] - ETA: 1:41 - loss: 0.4248 - categorical_accuracy: 0.8675
12736/60000 [=====>........................] - ETA: 1:41 - loss: 0.4238 - categorical_accuracy: 0.8679
12768/60000 [=====>........................] - ETA: 1:41 - loss: 0.4233 - categorical_accuracy: 0.8680
12800/60000 [=====>........................] - ETA: 1:41 - loss: 0.4224 - categorical_accuracy: 0.8683
12832/60000 [=====>........................] - ETA: 1:41 - loss: 0.4215 - categorical_accuracy: 0.8686
12864/60000 [=====>........................] - ETA: 1:41 - loss: 0.4209 - categorical_accuracy: 0.8688
12896/60000 [=====>........................] - ETA: 1:41 - loss: 0.4199 - categorical_accuracy: 0.8691
12928/60000 [=====>........................] - ETA: 1:41 - loss: 0.4192 - categorical_accuracy: 0.8694
12960/60000 [=====>........................] - ETA: 1:41 - loss: 0.4185 - categorical_accuracy: 0.8695
12992/60000 [=====>........................] - ETA: 1:41 - loss: 0.4176 - categorical_accuracy: 0.8697
13024/60000 [=====>........................] - ETA: 1:41 - loss: 0.4169 - categorical_accuracy: 0.8699
13056/60000 [=====>........................] - ETA: 1:41 - loss: 0.4163 - categorical_accuracy: 0.8701
13088/60000 [=====>........................] - ETA: 1:41 - loss: 0.4164 - categorical_accuracy: 0.8700
13120/60000 [=====>........................] - ETA: 1:41 - loss: 0.4155 - categorical_accuracy: 0.8703
13152/60000 [=====>........................] - ETA: 1:41 - loss: 0.4148 - categorical_accuracy: 0.8705
13184/60000 [=====>........................] - ETA: 1:41 - loss: 0.4150 - categorical_accuracy: 0.8704
13216/60000 [=====>........................] - ETA: 1:41 - loss: 0.4149 - categorical_accuracy: 0.8705
13248/60000 [=====>........................] - ETA: 1:41 - loss: 0.4144 - categorical_accuracy: 0.8706
13280/60000 [=====>........................] - ETA: 1:41 - loss: 0.4138 - categorical_accuracy: 0.8708
13312/60000 [=====>........................] - ETA: 1:41 - loss: 0.4132 - categorical_accuracy: 0.8710
13344/60000 [=====>........................] - ETA: 1:40 - loss: 0.4124 - categorical_accuracy: 0.8713
13376/60000 [=====>........................] - ETA: 1:40 - loss: 0.4117 - categorical_accuracy: 0.8714
13408/60000 [=====>........................] - ETA: 1:40 - loss: 0.4112 - categorical_accuracy: 0.8716
13440/60000 [=====>........................] - ETA: 1:40 - loss: 0.4105 - categorical_accuracy: 0.8718
13472/60000 [=====>........................] - ETA: 1:40 - loss: 0.4098 - categorical_accuracy: 0.8721
13504/60000 [=====>........................] - ETA: 1:40 - loss: 0.4095 - categorical_accuracy: 0.8723
13536/60000 [=====>........................] - ETA: 1:40 - loss: 0.4088 - categorical_accuracy: 0.8724
13568/60000 [=====>........................] - ETA: 1:40 - loss: 0.4086 - categorical_accuracy: 0.8725
13600/60000 [=====>........................] - ETA: 1:40 - loss: 0.4079 - categorical_accuracy: 0.8726
13632/60000 [=====>........................] - ETA: 1:40 - loss: 0.4070 - categorical_accuracy: 0.8729
13664/60000 [=====>........................] - ETA: 1:40 - loss: 0.4063 - categorical_accuracy: 0.8732
13696/60000 [=====>........................] - ETA: 1:40 - loss: 0.4054 - categorical_accuracy: 0.8735
13728/60000 [=====>........................] - ETA: 1:40 - loss: 0.4046 - categorical_accuracy: 0.8737
13760/60000 [=====>........................] - ETA: 1:40 - loss: 0.4049 - categorical_accuracy: 0.8736
13792/60000 [=====>........................] - ETA: 1:40 - loss: 0.4045 - categorical_accuracy: 0.8738
13824/60000 [=====>........................] - ETA: 1:40 - loss: 0.4037 - categorical_accuracy: 0.8741
13856/60000 [=====>........................] - ETA: 1:40 - loss: 0.4030 - categorical_accuracy: 0.8744
13888/60000 [=====>........................] - ETA: 1:40 - loss: 0.4026 - categorical_accuracy: 0.8745
13920/60000 [=====>........................] - ETA: 1:40 - loss: 0.4023 - categorical_accuracy: 0.8746
13952/60000 [=====>........................] - ETA: 1:40 - loss: 0.4015 - categorical_accuracy: 0.8748
13984/60000 [=====>........................] - ETA: 1:40 - loss: 0.4010 - categorical_accuracy: 0.8749
14016/60000 [======>.......................] - ETA: 1:40 - loss: 0.4003 - categorical_accuracy: 0.8751
14048/60000 [======>.......................] - ETA: 1:39 - loss: 0.3997 - categorical_accuracy: 0.8753
14080/60000 [======>.......................] - ETA: 1:39 - loss: 0.3992 - categorical_accuracy: 0.8754
14112/60000 [======>.......................] - ETA: 1:39 - loss: 0.3991 - categorical_accuracy: 0.8756
14144/60000 [======>.......................] - ETA: 1:39 - loss: 0.3988 - categorical_accuracy: 0.8757
14176/60000 [======>.......................] - ETA: 1:39 - loss: 0.3985 - categorical_accuracy: 0.8758
14208/60000 [======>.......................] - ETA: 1:39 - loss: 0.3978 - categorical_accuracy: 0.8761
14240/60000 [======>.......................] - ETA: 1:39 - loss: 0.3975 - categorical_accuracy: 0.8761
14272/60000 [======>.......................] - ETA: 1:39 - loss: 0.3975 - categorical_accuracy: 0.8761
14304/60000 [======>.......................] - ETA: 1:39 - loss: 0.3968 - categorical_accuracy: 0.8763
14336/60000 [======>.......................] - ETA: 1:39 - loss: 0.3970 - categorical_accuracy: 0.8763
14368/60000 [======>.......................] - ETA: 1:39 - loss: 0.3964 - categorical_accuracy: 0.8764
14400/60000 [======>.......................] - ETA: 1:39 - loss: 0.3957 - categorical_accuracy: 0.8766
14432/60000 [======>.......................] - ETA: 1:39 - loss: 0.3950 - categorical_accuracy: 0.8769
14464/60000 [======>.......................] - ETA: 1:39 - loss: 0.3947 - categorical_accuracy: 0.8770
14496/60000 [======>.......................] - ETA: 1:39 - loss: 0.3940 - categorical_accuracy: 0.8772
14528/60000 [======>.......................] - ETA: 1:39 - loss: 0.3936 - categorical_accuracy: 0.8773
14560/60000 [======>.......................] - ETA: 1:39 - loss: 0.3931 - categorical_accuracy: 0.8774
14592/60000 [======>.......................] - ETA: 1:38 - loss: 0.3926 - categorical_accuracy: 0.8775
14624/60000 [======>.......................] - ETA: 1:38 - loss: 0.3926 - categorical_accuracy: 0.8776
14656/60000 [======>.......................] - ETA: 1:38 - loss: 0.3920 - categorical_accuracy: 0.8779
14688/60000 [======>.......................] - ETA: 1:38 - loss: 0.3912 - categorical_accuracy: 0.8781
14720/60000 [======>.......................] - ETA: 1:38 - loss: 0.3905 - categorical_accuracy: 0.8784
14752/60000 [======>.......................] - ETA: 1:38 - loss: 0.3898 - categorical_accuracy: 0.8787
14784/60000 [======>.......................] - ETA: 1:38 - loss: 0.3895 - categorical_accuracy: 0.8788
14816/60000 [======>.......................] - ETA: 1:38 - loss: 0.3901 - categorical_accuracy: 0.8787
14848/60000 [======>.......................] - ETA: 1:38 - loss: 0.3894 - categorical_accuracy: 0.8790
14880/60000 [======>.......................] - ETA: 1:38 - loss: 0.3891 - categorical_accuracy: 0.8791
14912/60000 [======>.......................] - ETA: 1:38 - loss: 0.3891 - categorical_accuracy: 0.8792
14944/60000 [======>.......................] - ETA: 1:38 - loss: 0.3886 - categorical_accuracy: 0.8793
14976/60000 [======>.......................] - ETA: 1:38 - loss: 0.3879 - categorical_accuracy: 0.8796
15008/60000 [======>.......................] - ETA: 1:38 - loss: 0.3874 - categorical_accuracy: 0.8798
15040/60000 [======>.......................] - ETA: 1:38 - loss: 0.3868 - categorical_accuracy: 0.8800
15072/60000 [======>.......................] - ETA: 1:38 - loss: 0.3861 - categorical_accuracy: 0.8802
15104/60000 [======>.......................] - ETA: 1:37 - loss: 0.3857 - categorical_accuracy: 0.8803
15136/60000 [======>.......................] - ETA: 1:37 - loss: 0.3858 - categorical_accuracy: 0.8804
15168/60000 [======>.......................] - ETA: 1:37 - loss: 0.3850 - categorical_accuracy: 0.8806
15200/60000 [======>.......................] - ETA: 1:37 - loss: 0.3846 - categorical_accuracy: 0.8808
15232/60000 [======>.......................] - ETA: 1:37 - loss: 0.3840 - categorical_accuracy: 0.8810
15264/60000 [======>.......................] - ETA: 1:37 - loss: 0.3835 - categorical_accuracy: 0.8812
15296/60000 [======>.......................] - ETA: 1:37 - loss: 0.3828 - categorical_accuracy: 0.8814
15328/60000 [======>.......................] - ETA: 1:37 - loss: 0.3822 - categorical_accuracy: 0.8815
15360/60000 [======>.......................] - ETA: 1:37 - loss: 0.3815 - categorical_accuracy: 0.8817
15392/60000 [======>.......................] - ETA: 1:37 - loss: 0.3808 - categorical_accuracy: 0.8820
15424/60000 [======>.......................] - ETA: 1:37 - loss: 0.3802 - categorical_accuracy: 0.8821
15456/60000 [======>.......................] - ETA: 1:37 - loss: 0.3799 - categorical_accuracy: 0.8822
15488/60000 [======>.......................] - ETA: 1:37 - loss: 0.3797 - categorical_accuracy: 0.8823
15520/60000 [======>.......................] - ETA: 1:37 - loss: 0.3796 - categorical_accuracy: 0.8824
15552/60000 [======>.......................] - ETA: 1:36 - loss: 0.3793 - categorical_accuracy: 0.8825
15584/60000 [======>.......................] - ETA: 1:36 - loss: 0.3786 - categorical_accuracy: 0.8827
15616/60000 [======>.......................] - ETA: 1:36 - loss: 0.3781 - categorical_accuracy: 0.8828
15648/60000 [======>.......................] - ETA: 1:36 - loss: 0.3777 - categorical_accuracy: 0.8830
15680/60000 [======>.......................] - ETA: 1:36 - loss: 0.3776 - categorical_accuracy: 0.8830
15712/60000 [======>.......................] - ETA: 1:36 - loss: 0.3770 - categorical_accuracy: 0.8831
15744/60000 [======>.......................] - ETA: 1:36 - loss: 0.3768 - categorical_accuracy: 0.8833
15776/60000 [======>.......................] - ETA: 1:36 - loss: 0.3764 - categorical_accuracy: 0.8834
15808/60000 [======>.......................] - ETA: 1:36 - loss: 0.3760 - categorical_accuracy: 0.8835
15840/60000 [======>.......................] - ETA: 1:36 - loss: 0.3754 - categorical_accuracy: 0.8837
15872/60000 [======>.......................] - ETA: 1:36 - loss: 0.3750 - categorical_accuracy: 0.8838
15904/60000 [======>.......................] - ETA: 1:36 - loss: 0.3744 - categorical_accuracy: 0.8840
15936/60000 [======>.......................] - ETA: 1:35 - loss: 0.3737 - categorical_accuracy: 0.8842
15968/60000 [======>.......................] - ETA: 1:35 - loss: 0.3737 - categorical_accuracy: 0.8843
16000/60000 [=======>......................] - ETA: 1:35 - loss: 0.3733 - categorical_accuracy: 0.8844
16032/60000 [=======>......................] - ETA: 1:35 - loss: 0.3728 - categorical_accuracy: 0.8845
16064/60000 [=======>......................] - ETA: 1:35 - loss: 0.3723 - categorical_accuracy: 0.8846
16096/60000 [=======>......................] - ETA: 1:35 - loss: 0.3718 - categorical_accuracy: 0.8848
16128/60000 [=======>......................] - ETA: 1:35 - loss: 0.3714 - categorical_accuracy: 0.8849
16160/60000 [=======>......................] - ETA: 1:35 - loss: 0.3709 - categorical_accuracy: 0.8850
16192/60000 [=======>......................] - ETA: 1:35 - loss: 0.3703 - categorical_accuracy: 0.8852
16224/60000 [=======>......................] - ETA: 1:35 - loss: 0.3696 - categorical_accuracy: 0.8854
16256/60000 [=======>......................] - ETA: 1:35 - loss: 0.3693 - categorical_accuracy: 0.8855
16288/60000 [=======>......................] - ETA: 1:35 - loss: 0.3687 - categorical_accuracy: 0.8857
16320/60000 [=======>......................] - ETA: 1:35 - loss: 0.3683 - categorical_accuracy: 0.8858
16352/60000 [=======>......................] - ETA: 1:34 - loss: 0.3676 - categorical_accuracy: 0.8860
16384/60000 [=======>......................] - ETA: 1:34 - loss: 0.3675 - categorical_accuracy: 0.8860
16416/60000 [=======>......................] - ETA: 1:34 - loss: 0.3669 - categorical_accuracy: 0.8863
16448/60000 [=======>......................] - ETA: 1:34 - loss: 0.3663 - categorical_accuracy: 0.8864
16480/60000 [=======>......................] - ETA: 1:34 - loss: 0.3660 - categorical_accuracy: 0.8864
16512/60000 [=======>......................] - ETA: 1:34 - loss: 0.3659 - categorical_accuracy: 0.8865
16544/60000 [=======>......................] - ETA: 1:34 - loss: 0.3657 - categorical_accuracy: 0.8865
16576/60000 [=======>......................] - ETA: 1:34 - loss: 0.3651 - categorical_accuracy: 0.8867
16608/60000 [=======>......................] - ETA: 1:34 - loss: 0.3648 - categorical_accuracy: 0.8868
16640/60000 [=======>......................] - ETA: 1:34 - loss: 0.3643 - categorical_accuracy: 0.8870
16672/60000 [=======>......................] - ETA: 1:34 - loss: 0.3641 - categorical_accuracy: 0.8870
16704/60000 [=======>......................] - ETA: 1:33 - loss: 0.3636 - categorical_accuracy: 0.8871
16736/60000 [=======>......................] - ETA: 1:33 - loss: 0.3634 - categorical_accuracy: 0.8872
16768/60000 [=======>......................] - ETA: 1:33 - loss: 0.3631 - categorical_accuracy: 0.8873
16800/60000 [=======>......................] - ETA: 1:33 - loss: 0.3625 - categorical_accuracy: 0.8876
16832/60000 [=======>......................] - ETA: 1:33 - loss: 0.3622 - categorical_accuracy: 0.8877
16864/60000 [=======>......................] - ETA: 1:33 - loss: 0.3618 - categorical_accuracy: 0.8877
16896/60000 [=======>......................] - ETA: 1:33 - loss: 0.3612 - categorical_accuracy: 0.8880
16928/60000 [=======>......................] - ETA: 1:33 - loss: 0.3608 - categorical_accuracy: 0.8881
16960/60000 [=======>......................] - ETA: 1:33 - loss: 0.3604 - categorical_accuracy: 0.8883
16992/60000 [=======>......................] - ETA: 1:33 - loss: 0.3601 - categorical_accuracy: 0.8883
17024/60000 [=======>......................] - ETA: 1:33 - loss: 0.3600 - categorical_accuracy: 0.8884
17056/60000 [=======>......................] - ETA: 1:33 - loss: 0.3597 - categorical_accuracy: 0.8885
17088/60000 [=======>......................] - ETA: 1:32 - loss: 0.3601 - categorical_accuracy: 0.8885
17120/60000 [=======>......................] - ETA: 1:32 - loss: 0.3603 - categorical_accuracy: 0.8885
17152/60000 [=======>......................] - ETA: 1:32 - loss: 0.3598 - categorical_accuracy: 0.8887
17184/60000 [=======>......................] - ETA: 1:32 - loss: 0.3593 - categorical_accuracy: 0.8887
17216/60000 [=======>......................] - ETA: 1:32 - loss: 0.3588 - categorical_accuracy: 0.8889
17248/60000 [=======>......................] - ETA: 1:32 - loss: 0.3583 - categorical_accuracy: 0.8890
17280/60000 [=======>......................] - ETA: 1:32 - loss: 0.3579 - categorical_accuracy: 0.8891
17312/60000 [=======>......................] - ETA: 1:32 - loss: 0.3576 - categorical_accuracy: 0.8891
17344/60000 [=======>......................] - ETA: 1:32 - loss: 0.3571 - categorical_accuracy: 0.8892
17376/60000 [=======>......................] - ETA: 1:32 - loss: 0.3566 - categorical_accuracy: 0.8894
17408/60000 [=======>......................] - ETA: 1:32 - loss: 0.3560 - categorical_accuracy: 0.8896
17440/60000 [=======>......................] - ETA: 1:32 - loss: 0.3557 - categorical_accuracy: 0.8897
17472/60000 [=======>......................] - ETA: 1:32 - loss: 0.3553 - categorical_accuracy: 0.8898
17504/60000 [=======>......................] - ETA: 1:31 - loss: 0.3548 - categorical_accuracy: 0.8899
17536/60000 [=======>......................] - ETA: 1:31 - loss: 0.3544 - categorical_accuracy: 0.8901
17568/60000 [=======>......................] - ETA: 1:31 - loss: 0.3539 - categorical_accuracy: 0.8903
17600/60000 [=======>......................] - ETA: 1:31 - loss: 0.3538 - categorical_accuracy: 0.8902
17632/60000 [=======>......................] - ETA: 1:31 - loss: 0.3536 - categorical_accuracy: 0.8903
17664/60000 [=======>......................] - ETA: 1:31 - loss: 0.3531 - categorical_accuracy: 0.8905
17696/60000 [=======>......................] - ETA: 1:31 - loss: 0.3530 - categorical_accuracy: 0.8905
17728/60000 [=======>......................] - ETA: 1:31 - loss: 0.3526 - categorical_accuracy: 0.8906
17760/60000 [=======>......................] - ETA: 1:31 - loss: 0.3520 - categorical_accuracy: 0.8908
17792/60000 [=======>......................] - ETA: 1:31 - loss: 0.3518 - categorical_accuracy: 0.8908
17824/60000 [=======>......................] - ETA: 1:31 - loss: 0.3512 - categorical_accuracy: 0.8910
17856/60000 [=======>......................] - ETA: 1:31 - loss: 0.3511 - categorical_accuracy: 0.8911
17888/60000 [=======>......................] - ETA: 1:31 - loss: 0.3509 - categorical_accuracy: 0.8910
17920/60000 [=======>......................] - ETA: 1:30 - loss: 0.3505 - categorical_accuracy: 0.8912
17952/60000 [=======>......................] - ETA: 1:30 - loss: 0.3501 - categorical_accuracy: 0.8913
17984/60000 [=======>......................] - ETA: 1:30 - loss: 0.3496 - categorical_accuracy: 0.8915
18016/60000 [========>.....................] - ETA: 1:30 - loss: 0.3493 - categorical_accuracy: 0.8917
18048/60000 [========>.....................] - ETA: 1:30 - loss: 0.3488 - categorical_accuracy: 0.8918
18080/60000 [========>.....................] - ETA: 1:30 - loss: 0.3490 - categorical_accuracy: 0.8919
18112/60000 [========>.....................] - ETA: 1:30 - loss: 0.3485 - categorical_accuracy: 0.8920
18144/60000 [========>.....................] - ETA: 1:30 - loss: 0.3482 - categorical_accuracy: 0.8921
18176/60000 [========>.....................] - ETA: 1:30 - loss: 0.3476 - categorical_accuracy: 0.8923
18208/60000 [========>.....................] - ETA: 1:30 - loss: 0.3472 - categorical_accuracy: 0.8925
18240/60000 [========>.....................] - ETA: 1:30 - loss: 0.3470 - categorical_accuracy: 0.8925
18272/60000 [========>.....................] - ETA: 1:30 - loss: 0.3469 - categorical_accuracy: 0.8926
18304/60000 [========>.....................] - ETA: 1:30 - loss: 0.3467 - categorical_accuracy: 0.8925
18336/60000 [========>.....................] - ETA: 1:30 - loss: 0.3463 - categorical_accuracy: 0.8926
18368/60000 [========>.....................] - ETA: 1:29 - loss: 0.3460 - categorical_accuracy: 0.8927
18400/60000 [========>.....................] - ETA: 1:29 - loss: 0.3456 - categorical_accuracy: 0.8929
18432/60000 [========>.....................] - ETA: 1:29 - loss: 0.3451 - categorical_accuracy: 0.8931
18464/60000 [========>.....................] - ETA: 1:29 - loss: 0.3447 - categorical_accuracy: 0.8933
18496/60000 [========>.....................] - ETA: 1:29 - loss: 0.3442 - categorical_accuracy: 0.8934
18528/60000 [========>.....................] - ETA: 1:29 - loss: 0.3438 - categorical_accuracy: 0.8935
18560/60000 [========>.....................] - ETA: 1:29 - loss: 0.3434 - categorical_accuracy: 0.8936
18592/60000 [========>.....................] - ETA: 1:29 - loss: 0.3429 - categorical_accuracy: 0.8938
18624/60000 [========>.....................] - ETA: 1:29 - loss: 0.3426 - categorical_accuracy: 0.8938
18656/60000 [========>.....................] - ETA: 1:29 - loss: 0.3421 - categorical_accuracy: 0.8940
18688/60000 [========>.....................] - ETA: 1:29 - loss: 0.3416 - categorical_accuracy: 0.8942
18720/60000 [========>.....................] - ETA: 1:29 - loss: 0.3416 - categorical_accuracy: 0.8941
18752/60000 [========>.....................] - ETA: 1:28 - loss: 0.3411 - categorical_accuracy: 0.8943
18784/60000 [========>.....................] - ETA: 1:28 - loss: 0.3410 - categorical_accuracy: 0.8944
18816/60000 [========>.....................] - ETA: 1:28 - loss: 0.3409 - categorical_accuracy: 0.8946
18848/60000 [========>.....................] - ETA: 1:28 - loss: 0.3407 - categorical_accuracy: 0.8947
18880/60000 [========>.....................] - ETA: 1:28 - loss: 0.3404 - categorical_accuracy: 0.8948
18912/60000 [========>.....................] - ETA: 1:28 - loss: 0.3399 - categorical_accuracy: 0.8950
18944/60000 [========>.....................] - ETA: 1:28 - loss: 0.3396 - categorical_accuracy: 0.8951
18976/60000 [========>.....................] - ETA: 1:28 - loss: 0.3394 - categorical_accuracy: 0.8950
19008/60000 [========>.....................] - ETA: 1:28 - loss: 0.3388 - categorical_accuracy: 0.8952
19040/60000 [========>.....................] - ETA: 1:28 - loss: 0.3384 - categorical_accuracy: 0.8954
19072/60000 [========>.....................] - ETA: 1:28 - loss: 0.3384 - categorical_accuracy: 0.8954
19104/60000 [========>.....................] - ETA: 1:28 - loss: 0.3380 - categorical_accuracy: 0.8956
19136/60000 [========>.....................] - ETA: 1:28 - loss: 0.3377 - categorical_accuracy: 0.8956
19168/60000 [========>.....................] - ETA: 1:27 - loss: 0.3374 - categorical_accuracy: 0.8957
19200/60000 [========>.....................] - ETA: 1:27 - loss: 0.3370 - categorical_accuracy: 0.8958
19232/60000 [========>.....................] - ETA: 1:27 - loss: 0.3366 - categorical_accuracy: 0.8960
19264/60000 [========>.....................] - ETA: 1:27 - loss: 0.3361 - categorical_accuracy: 0.8961
19296/60000 [========>.....................] - ETA: 1:27 - loss: 0.3360 - categorical_accuracy: 0.8962
19328/60000 [========>.....................] - ETA: 1:27 - loss: 0.3357 - categorical_accuracy: 0.8964
19360/60000 [========>.....................] - ETA: 1:27 - loss: 0.3355 - categorical_accuracy: 0.8964
19392/60000 [========>.....................] - ETA: 1:27 - loss: 0.3352 - categorical_accuracy: 0.8965
19424/60000 [========>.....................] - ETA: 1:27 - loss: 0.3353 - categorical_accuracy: 0.8965
19456/60000 [========>.....................] - ETA: 1:27 - loss: 0.3353 - categorical_accuracy: 0.8964
19488/60000 [========>.....................] - ETA: 1:27 - loss: 0.3348 - categorical_accuracy: 0.8966
19520/60000 [========>.....................] - ETA: 1:27 - loss: 0.3349 - categorical_accuracy: 0.8966
19552/60000 [========>.....................] - ETA: 1:27 - loss: 0.3348 - categorical_accuracy: 0.8965
19584/60000 [========>.....................] - ETA: 1:26 - loss: 0.3348 - categorical_accuracy: 0.8965
19616/60000 [========>.....................] - ETA: 1:26 - loss: 0.3346 - categorical_accuracy: 0.8967
19648/60000 [========>.....................] - ETA: 1:26 - loss: 0.3343 - categorical_accuracy: 0.8967
19680/60000 [========>.....................] - ETA: 1:26 - loss: 0.3341 - categorical_accuracy: 0.8968
19712/60000 [========>.....................] - ETA: 1:26 - loss: 0.3336 - categorical_accuracy: 0.8970
19744/60000 [========>.....................] - ETA: 1:26 - loss: 0.3334 - categorical_accuracy: 0.8971
19776/60000 [========>.....................] - ETA: 1:26 - loss: 0.3329 - categorical_accuracy: 0.8972
19808/60000 [========>.....................] - ETA: 1:26 - loss: 0.3324 - categorical_accuracy: 0.8974
19840/60000 [========>.....................] - ETA: 1:26 - loss: 0.3321 - categorical_accuracy: 0.8975
19872/60000 [========>.....................] - ETA: 1:26 - loss: 0.3319 - categorical_accuracy: 0.8976
19904/60000 [========>.....................] - ETA: 1:26 - loss: 0.3316 - categorical_accuracy: 0.8977
19936/60000 [========>.....................] - ETA: 1:26 - loss: 0.3315 - categorical_accuracy: 0.8977
19968/60000 [========>.....................] - ETA: 1:25 - loss: 0.3312 - categorical_accuracy: 0.8977
20000/60000 [=========>....................] - ETA: 1:25 - loss: 0.3308 - categorical_accuracy: 0.8978
20032/60000 [=========>....................] - ETA: 1:25 - loss: 0.3305 - categorical_accuracy: 0.8979
20064/60000 [=========>....................] - ETA: 1:25 - loss: 0.3302 - categorical_accuracy: 0.8979
20096/60000 [=========>....................] - ETA: 1:25 - loss: 0.3302 - categorical_accuracy: 0.8979
20128/60000 [=========>....................] - ETA: 1:25 - loss: 0.3303 - categorical_accuracy: 0.8980
20160/60000 [=========>....................] - ETA: 1:25 - loss: 0.3300 - categorical_accuracy: 0.8980
20192/60000 [=========>....................] - ETA: 1:25 - loss: 0.3297 - categorical_accuracy: 0.8981
20224/60000 [=========>....................] - ETA: 1:25 - loss: 0.3293 - categorical_accuracy: 0.8981
20256/60000 [=========>....................] - ETA: 1:25 - loss: 0.3289 - categorical_accuracy: 0.8983
20288/60000 [=========>....................] - ETA: 1:25 - loss: 0.3286 - categorical_accuracy: 0.8984
20320/60000 [=========>....................] - ETA: 1:25 - loss: 0.3286 - categorical_accuracy: 0.8984
20352/60000 [=========>....................] - ETA: 1:25 - loss: 0.3281 - categorical_accuracy: 0.8985
20384/60000 [=========>....................] - ETA: 1:24 - loss: 0.3277 - categorical_accuracy: 0.8987
20416/60000 [=========>....................] - ETA: 1:24 - loss: 0.3275 - categorical_accuracy: 0.8987
20448/60000 [=========>....................] - ETA: 1:24 - loss: 0.3275 - categorical_accuracy: 0.8988
20480/60000 [=========>....................] - ETA: 1:24 - loss: 0.3272 - categorical_accuracy: 0.8988
20512/60000 [=========>....................] - ETA: 1:24 - loss: 0.3269 - categorical_accuracy: 0.8988
20544/60000 [=========>....................] - ETA: 1:24 - loss: 0.3272 - categorical_accuracy: 0.8989
20576/60000 [=========>....................] - ETA: 1:24 - loss: 0.3268 - categorical_accuracy: 0.8990
20608/60000 [=========>....................] - ETA: 1:24 - loss: 0.3265 - categorical_accuracy: 0.8991
20640/60000 [=========>....................] - ETA: 1:24 - loss: 0.3266 - categorical_accuracy: 0.8990
20672/60000 [=========>....................] - ETA: 1:24 - loss: 0.3261 - categorical_accuracy: 0.8992
20704/60000 [=========>....................] - ETA: 1:24 - loss: 0.3257 - categorical_accuracy: 0.8993
20736/60000 [=========>....................] - ETA: 1:24 - loss: 0.3254 - categorical_accuracy: 0.8994
20768/60000 [=========>....................] - ETA: 1:24 - loss: 0.3249 - categorical_accuracy: 0.8996
20800/60000 [=========>....................] - ETA: 1:23 - loss: 0.3245 - categorical_accuracy: 0.8997
20832/60000 [=========>....................] - ETA: 1:23 - loss: 0.3243 - categorical_accuracy: 0.8997
20864/60000 [=========>....................] - ETA: 1:23 - loss: 0.3241 - categorical_accuracy: 0.8998
20896/60000 [=========>....................] - ETA: 1:23 - loss: 0.3238 - categorical_accuracy: 0.8999
20928/60000 [=========>....................] - ETA: 1:23 - loss: 0.3236 - categorical_accuracy: 0.9000
20960/60000 [=========>....................] - ETA: 1:23 - loss: 0.3232 - categorical_accuracy: 0.9001
20992/60000 [=========>....................] - ETA: 1:23 - loss: 0.3230 - categorical_accuracy: 0.9002
21024/60000 [=========>....................] - ETA: 1:23 - loss: 0.3226 - categorical_accuracy: 0.9004
21056/60000 [=========>....................] - ETA: 1:23 - loss: 0.3228 - categorical_accuracy: 0.9004
21088/60000 [=========>....................] - ETA: 1:23 - loss: 0.3226 - categorical_accuracy: 0.9005
21120/60000 [=========>....................] - ETA: 1:23 - loss: 0.3221 - categorical_accuracy: 0.9006
21152/60000 [=========>....................] - ETA: 1:23 - loss: 0.3217 - categorical_accuracy: 0.9008
21184/60000 [=========>....................] - ETA: 1:23 - loss: 0.3214 - categorical_accuracy: 0.9009
21216/60000 [=========>....................] - ETA: 1:23 - loss: 0.3211 - categorical_accuracy: 0.9010
21248/60000 [=========>....................] - ETA: 1:23 - loss: 0.3209 - categorical_accuracy: 0.9011
21280/60000 [=========>....................] - ETA: 1:22 - loss: 0.3205 - categorical_accuracy: 0.9012
21312/60000 [=========>....................] - ETA: 1:22 - loss: 0.3203 - categorical_accuracy: 0.9012
21344/60000 [=========>....................] - ETA: 1:22 - loss: 0.3199 - categorical_accuracy: 0.9013
21376/60000 [=========>....................] - ETA: 1:22 - loss: 0.3195 - categorical_accuracy: 0.9015
21408/60000 [=========>....................] - ETA: 1:22 - loss: 0.3191 - categorical_accuracy: 0.9016
21440/60000 [=========>....................] - ETA: 1:22 - loss: 0.3187 - categorical_accuracy: 0.9017
21472/60000 [=========>....................] - ETA: 1:22 - loss: 0.3184 - categorical_accuracy: 0.9018
21504/60000 [=========>....................] - ETA: 1:22 - loss: 0.3180 - categorical_accuracy: 0.9020
21536/60000 [=========>....................] - ETA: 1:22 - loss: 0.3176 - categorical_accuracy: 0.9021
21568/60000 [=========>....................] - ETA: 1:22 - loss: 0.3175 - categorical_accuracy: 0.9021
21600/60000 [=========>....................] - ETA: 1:22 - loss: 0.3177 - categorical_accuracy: 0.9021
21632/60000 [=========>....................] - ETA: 1:22 - loss: 0.3176 - categorical_accuracy: 0.9021
21664/60000 [=========>....................] - ETA: 1:22 - loss: 0.3175 - categorical_accuracy: 0.9022
21696/60000 [=========>....................] - ETA: 1:21 - loss: 0.3171 - categorical_accuracy: 0.9024
21728/60000 [=========>....................] - ETA: 1:21 - loss: 0.3167 - categorical_accuracy: 0.9025
21760/60000 [=========>....................] - ETA: 1:21 - loss: 0.3164 - categorical_accuracy: 0.9026
21792/60000 [=========>....................] - ETA: 1:21 - loss: 0.3161 - categorical_accuracy: 0.9026
21824/60000 [=========>....................] - ETA: 1:21 - loss: 0.3158 - categorical_accuracy: 0.9027
21856/60000 [=========>....................] - ETA: 1:21 - loss: 0.3154 - categorical_accuracy: 0.9029
21888/60000 [=========>....................] - ETA: 1:21 - loss: 0.3150 - categorical_accuracy: 0.9030
21920/60000 [=========>....................] - ETA: 1:21 - loss: 0.3146 - categorical_accuracy: 0.9031
21952/60000 [=========>....................] - ETA: 1:21 - loss: 0.3142 - categorical_accuracy: 0.9032
21984/60000 [=========>....................] - ETA: 1:21 - loss: 0.3139 - categorical_accuracy: 0.9033
22016/60000 [==========>...................] - ETA: 1:21 - loss: 0.3136 - categorical_accuracy: 0.9033
22048/60000 [==========>...................] - ETA: 1:21 - loss: 0.3133 - categorical_accuracy: 0.9034
22080/60000 [==========>...................] - ETA: 1:21 - loss: 0.3132 - categorical_accuracy: 0.9035
22112/60000 [==========>...................] - ETA: 1:20 - loss: 0.3128 - categorical_accuracy: 0.9037
22144/60000 [==========>...................] - ETA: 1:20 - loss: 0.3124 - categorical_accuracy: 0.9038
22176/60000 [==========>...................] - ETA: 1:20 - loss: 0.3121 - categorical_accuracy: 0.9039
22208/60000 [==========>...................] - ETA: 1:20 - loss: 0.3117 - categorical_accuracy: 0.9040
22240/60000 [==========>...................] - ETA: 1:20 - loss: 0.3114 - categorical_accuracy: 0.9041
22272/60000 [==========>...................] - ETA: 1:20 - loss: 0.3113 - categorical_accuracy: 0.9042
22304/60000 [==========>...................] - ETA: 1:20 - loss: 0.3109 - categorical_accuracy: 0.9044
22336/60000 [==========>...................] - ETA: 1:20 - loss: 0.3105 - categorical_accuracy: 0.9045
22368/60000 [==========>...................] - ETA: 1:20 - loss: 0.3101 - categorical_accuracy: 0.9046
22400/60000 [==========>...................] - ETA: 1:20 - loss: 0.3098 - categorical_accuracy: 0.9047
22432/60000 [==========>...................] - ETA: 1:20 - loss: 0.3095 - categorical_accuracy: 0.9048
22464/60000 [==========>...................] - ETA: 1:20 - loss: 0.3091 - categorical_accuracy: 0.9049
22496/60000 [==========>...................] - ETA: 1:20 - loss: 0.3091 - categorical_accuracy: 0.9049
22528/60000 [==========>...................] - ETA: 1:20 - loss: 0.3087 - categorical_accuracy: 0.9051
22560/60000 [==========>...................] - ETA: 1:19 - loss: 0.3085 - categorical_accuracy: 0.9051
22592/60000 [==========>...................] - ETA: 1:19 - loss: 0.3080 - categorical_accuracy: 0.9053
22624/60000 [==========>...................] - ETA: 1:19 - loss: 0.3078 - categorical_accuracy: 0.9054
22656/60000 [==========>...................] - ETA: 1:19 - loss: 0.3074 - categorical_accuracy: 0.9055
22688/60000 [==========>...................] - ETA: 1:19 - loss: 0.3071 - categorical_accuracy: 0.9055
22720/60000 [==========>...................] - ETA: 1:19 - loss: 0.3070 - categorical_accuracy: 0.9056
22752/60000 [==========>...................] - ETA: 1:19 - loss: 0.3069 - categorical_accuracy: 0.9057
22784/60000 [==========>...................] - ETA: 1:19 - loss: 0.3068 - categorical_accuracy: 0.9057
22816/60000 [==========>...................] - ETA: 1:19 - loss: 0.3064 - categorical_accuracy: 0.9058
22848/60000 [==========>...................] - ETA: 1:19 - loss: 0.3064 - categorical_accuracy: 0.9058
22880/60000 [==========>...................] - ETA: 1:19 - loss: 0.3064 - categorical_accuracy: 0.9058
22912/60000 [==========>...................] - ETA: 1:19 - loss: 0.3064 - categorical_accuracy: 0.9058
22944/60000 [==========>...................] - ETA: 1:19 - loss: 0.3061 - categorical_accuracy: 0.9059
22976/60000 [==========>...................] - ETA: 1:18 - loss: 0.3059 - categorical_accuracy: 0.9059
23008/60000 [==========>...................] - ETA: 1:18 - loss: 0.3056 - categorical_accuracy: 0.9059
23040/60000 [==========>...................] - ETA: 1:18 - loss: 0.3053 - categorical_accuracy: 0.9060
23072/60000 [==========>...................] - ETA: 1:18 - loss: 0.3052 - categorical_accuracy: 0.9060
23104/60000 [==========>...................] - ETA: 1:18 - loss: 0.3049 - categorical_accuracy: 0.9061
23136/60000 [==========>...................] - ETA: 1:18 - loss: 0.3047 - categorical_accuracy: 0.9061
23168/60000 [==========>...................] - ETA: 1:18 - loss: 0.3045 - categorical_accuracy: 0.9062
23200/60000 [==========>...................] - ETA: 1:18 - loss: 0.3043 - categorical_accuracy: 0.9062
23232/60000 [==========>...................] - ETA: 1:18 - loss: 0.3039 - categorical_accuracy: 0.9064
23264/60000 [==========>...................] - ETA: 1:18 - loss: 0.3037 - categorical_accuracy: 0.9064
23296/60000 [==========>...................] - ETA: 1:18 - loss: 0.3034 - categorical_accuracy: 0.9065
23328/60000 [==========>...................] - ETA: 1:18 - loss: 0.3035 - categorical_accuracy: 0.9066
23360/60000 [==========>...................] - ETA: 1:18 - loss: 0.3031 - categorical_accuracy: 0.9066
23392/60000 [==========>...................] - ETA: 1:17 - loss: 0.3031 - categorical_accuracy: 0.9067
23424/60000 [==========>...................] - ETA: 1:17 - loss: 0.3029 - categorical_accuracy: 0.9068
23456/60000 [==========>...................] - ETA: 1:17 - loss: 0.3026 - categorical_accuracy: 0.9068
23488/60000 [==========>...................] - ETA: 1:17 - loss: 0.3022 - categorical_accuracy: 0.9069
23520/60000 [==========>...................] - ETA: 1:17 - loss: 0.3020 - categorical_accuracy: 0.9070
23552/60000 [==========>...................] - ETA: 1:17 - loss: 0.3018 - categorical_accuracy: 0.9070
23584/60000 [==========>...................] - ETA: 1:17 - loss: 0.3017 - categorical_accuracy: 0.9071
23616/60000 [==========>...................] - ETA: 1:17 - loss: 0.3016 - categorical_accuracy: 0.9071
23648/60000 [==========>...................] - ETA: 1:17 - loss: 0.3013 - categorical_accuracy: 0.9072
23680/60000 [==========>...................] - ETA: 1:17 - loss: 0.3010 - categorical_accuracy: 0.9073
23712/60000 [==========>...................] - ETA: 1:17 - loss: 0.3008 - categorical_accuracy: 0.9074
23744/60000 [==========>...................] - ETA: 1:17 - loss: 0.3004 - categorical_accuracy: 0.9075
23776/60000 [==========>...................] - ETA: 1:17 - loss: 0.3002 - categorical_accuracy: 0.9076
23808/60000 [==========>...................] - ETA: 1:16 - loss: 0.2999 - categorical_accuracy: 0.9076
23840/60000 [==========>...................] - ETA: 1:16 - loss: 0.2996 - categorical_accuracy: 0.9078
23872/60000 [==========>...................] - ETA: 1:16 - loss: 0.2994 - categorical_accuracy: 0.9078
23904/60000 [==========>...................] - ETA: 1:16 - loss: 0.2992 - categorical_accuracy: 0.9078
23936/60000 [==========>...................] - ETA: 1:16 - loss: 0.2989 - categorical_accuracy: 0.9080
23968/60000 [==========>...................] - ETA: 1:16 - loss: 0.2987 - categorical_accuracy: 0.9080
24000/60000 [===========>..................] - ETA: 1:16 - loss: 0.2992 - categorical_accuracy: 0.9080
24032/60000 [===========>..................] - ETA: 1:16 - loss: 0.2991 - categorical_accuracy: 0.9081
24064/60000 [===========>..................] - ETA: 1:16 - loss: 0.2988 - categorical_accuracy: 0.9082
24096/60000 [===========>..................] - ETA: 1:16 - loss: 0.2985 - categorical_accuracy: 0.9083
24128/60000 [===========>..................] - ETA: 1:16 - loss: 0.2986 - categorical_accuracy: 0.9083
24160/60000 [===========>..................] - ETA: 1:16 - loss: 0.2986 - categorical_accuracy: 0.9083
24192/60000 [===========>..................] - ETA: 1:16 - loss: 0.2983 - categorical_accuracy: 0.9084
24224/60000 [===========>..................] - ETA: 1:15 - loss: 0.2983 - categorical_accuracy: 0.9084
24256/60000 [===========>..................] - ETA: 1:15 - loss: 0.2982 - categorical_accuracy: 0.9084
24288/60000 [===========>..................] - ETA: 1:15 - loss: 0.2978 - categorical_accuracy: 0.9086
24320/60000 [===========>..................] - ETA: 1:15 - loss: 0.2977 - categorical_accuracy: 0.9086
24352/60000 [===========>..................] - ETA: 1:15 - loss: 0.2974 - categorical_accuracy: 0.9087
24384/60000 [===========>..................] - ETA: 1:15 - loss: 0.2973 - categorical_accuracy: 0.9088
24416/60000 [===========>..................] - ETA: 1:15 - loss: 0.2970 - categorical_accuracy: 0.9089
24448/60000 [===========>..................] - ETA: 1:15 - loss: 0.2966 - categorical_accuracy: 0.9090
24480/60000 [===========>..................] - ETA: 1:15 - loss: 0.2967 - categorical_accuracy: 0.9090
24512/60000 [===========>..................] - ETA: 1:15 - loss: 0.2964 - categorical_accuracy: 0.9091
24544/60000 [===========>..................] - ETA: 1:15 - loss: 0.2962 - categorical_accuracy: 0.9092
24576/60000 [===========>..................] - ETA: 1:15 - loss: 0.2960 - categorical_accuracy: 0.9093
24608/60000 [===========>..................] - ETA: 1:15 - loss: 0.2956 - categorical_accuracy: 0.9094
24640/60000 [===========>..................] - ETA: 1:15 - loss: 0.2953 - categorical_accuracy: 0.9095
24672/60000 [===========>..................] - ETA: 1:15 - loss: 0.2950 - categorical_accuracy: 0.9096
24704/60000 [===========>..................] - ETA: 1:14 - loss: 0.2948 - categorical_accuracy: 0.9097
24736/60000 [===========>..................] - ETA: 1:14 - loss: 0.2947 - categorical_accuracy: 0.9097
24768/60000 [===========>..................] - ETA: 1:14 - loss: 0.2945 - categorical_accuracy: 0.9097
24800/60000 [===========>..................] - ETA: 1:14 - loss: 0.2943 - categorical_accuracy: 0.9098
24832/60000 [===========>..................] - ETA: 1:14 - loss: 0.2940 - categorical_accuracy: 0.9099
24864/60000 [===========>..................] - ETA: 1:14 - loss: 0.2940 - categorical_accuracy: 0.9099
24896/60000 [===========>..................] - ETA: 1:14 - loss: 0.2938 - categorical_accuracy: 0.9100
24928/60000 [===========>..................] - ETA: 1:14 - loss: 0.2936 - categorical_accuracy: 0.9100
24960/60000 [===========>..................] - ETA: 1:14 - loss: 0.2934 - categorical_accuracy: 0.9101
24992/60000 [===========>..................] - ETA: 1:14 - loss: 0.2931 - categorical_accuracy: 0.9102
25024/60000 [===========>..................] - ETA: 1:14 - loss: 0.2928 - categorical_accuracy: 0.9102
25056/60000 [===========>..................] - ETA: 1:14 - loss: 0.2925 - categorical_accuracy: 0.9103
25088/60000 [===========>..................] - ETA: 1:14 - loss: 0.2923 - categorical_accuracy: 0.9104
25120/60000 [===========>..................] - ETA: 1:13 - loss: 0.2920 - categorical_accuracy: 0.9105
25152/60000 [===========>..................] - ETA: 1:13 - loss: 0.2916 - categorical_accuracy: 0.9106
25184/60000 [===========>..................] - ETA: 1:13 - loss: 0.2919 - categorical_accuracy: 0.9106
25216/60000 [===========>..................] - ETA: 1:13 - loss: 0.2917 - categorical_accuracy: 0.9106
25248/60000 [===========>..................] - ETA: 1:13 - loss: 0.2915 - categorical_accuracy: 0.9107
25280/60000 [===========>..................] - ETA: 1:13 - loss: 0.2914 - categorical_accuracy: 0.9106
25312/60000 [===========>..................] - ETA: 1:13 - loss: 0.2912 - categorical_accuracy: 0.9107
25344/60000 [===========>..................] - ETA: 1:13 - loss: 0.2913 - categorical_accuracy: 0.9106
25376/60000 [===========>..................] - ETA: 1:13 - loss: 0.2910 - categorical_accuracy: 0.9107
25408/60000 [===========>..................] - ETA: 1:13 - loss: 0.2908 - categorical_accuracy: 0.9108
25440/60000 [===========>..................] - ETA: 1:13 - loss: 0.2905 - categorical_accuracy: 0.9109
25472/60000 [===========>..................] - ETA: 1:13 - loss: 0.2904 - categorical_accuracy: 0.9109
25504/60000 [===========>..................] - ETA: 1:13 - loss: 0.2903 - categorical_accuracy: 0.9109
25536/60000 [===========>..................] - ETA: 1:13 - loss: 0.2900 - categorical_accuracy: 0.9110
25568/60000 [===========>..................] - ETA: 1:13 - loss: 0.2898 - categorical_accuracy: 0.9110
25600/60000 [===========>..................] - ETA: 1:12 - loss: 0.2896 - categorical_accuracy: 0.9111
25632/60000 [===========>..................] - ETA: 1:12 - loss: 0.2893 - categorical_accuracy: 0.9112
25664/60000 [===========>..................] - ETA: 1:12 - loss: 0.2890 - categorical_accuracy: 0.9113
25696/60000 [===========>..................] - ETA: 1:12 - loss: 0.2889 - categorical_accuracy: 0.9113
25728/60000 [===========>..................] - ETA: 1:12 - loss: 0.2886 - categorical_accuracy: 0.9114
25760/60000 [===========>..................] - ETA: 1:12 - loss: 0.2884 - categorical_accuracy: 0.9114
25792/60000 [===========>..................] - ETA: 1:12 - loss: 0.2881 - categorical_accuracy: 0.9115
25824/60000 [===========>..................] - ETA: 1:12 - loss: 0.2882 - categorical_accuracy: 0.9115
25856/60000 [===========>..................] - ETA: 1:12 - loss: 0.2879 - categorical_accuracy: 0.9115
25888/60000 [===========>..................] - ETA: 1:12 - loss: 0.2880 - categorical_accuracy: 0.9115
25920/60000 [===========>..................] - ETA: 1:12 - loss: 0.2879 - categorical_accuracy: 0.9116
25952/60000 [===========>..................] - ETA: 1:12 - loss: 0.2877 - categorical_accuracy: 0.9116
25984/60000 [===========>..................] - ETA: 1:12 - loss: 0.2874 - categorical_accuracy: 0.9117
26016/60000 [============>.................] - ETA: 1:12 - loss: 0.2870 - categorical_accuracy: 0.9118
26048/60000 [============>.................] - ETA: 1:12 - loss: 0.2868 - categorical_accuracy: 0.9119
26080/60000 [============>.................] - ETA: 1:11 - loss: 0.2866 - categorical_accuracy: 0.9119
26112/60000 [============>.................] - ETA: 1:11 - loss: 0.2864 - categorical_accuracy: 0.9120
26144/60000 [============>.................] - ETA: 1:11 - loss: 0.2862 - categorical_accuracy: 0.9121
26176/60000 [============>.................] - ETA: 1:11 - loss: 0.2859 - categorical_accuracy: 0.9122
26208/60000 [============>.................] - ETA: 1:11 - loss: 0.2856 - categorical_accuracy: 0.9123
26240/60000 [============>.................] - ETA: 1:11 - loss: 0.2853 - categorical_accuracy: 0.9123
26272/60000 [============>.................] - ETA: 1:11 - loss: 0.2853 - categorical_accuracy: 0.9123
26304/60000 [============>.................] - ETA: 1:11 - loss: 0.2849 - categorical_accuracy: 0.9124
26336/60000 [============>.................] - ETA: 1:11 - loss: 0.2847 - categorical_accuracy: 0.9125
26368/60000 [============>.................] - ETA: 1:11 - loss: 0.2847 - categorical_accuracy: 0.9126
26400/60000 [============>.................] - ETA: 1:11 - loss: 0.2846 - categorical_accuracy: 0.9126
26432/60000 [============>.................] - ETA: 1:11 - loss: 0.2843 - categorical_accuracy: 0.9127
26464/60000 [============>.................] - ETA: 1:11 - loss: 0.2846 - categorical_accuracy: 0.9127
26496/60000 [============>.................] - ETA: 1:11 - loss: 0.2848 - categorical_accuracy: 0.9127
26528/60000 [============>.................] - ETA: 1:11 - loss: 0.2846 - categorical_accuracy: 0.9128
26560/60000 [============>.................] - ETA: 1:11 - loss: 0.2843 - categorical_accuracy: 0.9129
26592/60000 [============>.................] - ETA: 1:10 - loss: 0.2841 - categorical_accuracy: 0.9129
26624/60000 [============>.................] - ETA: 1:10 - loss: 0.2839 - categorical_accuracy: 0.9130
26656/60000 [============>.................] - ETA: 1:10 - loss: 0.2836 - categorical_accuracy: 0.9131
26688/60000 [============>.................] - ETA: 1:10 - loss: 0.2834 - categorical_accuracy: 0.9132
26720/60000 [============>.................] - ETA: 1:10 - loss: 0.2832 - categorical_accuracy: 0.9132
26752/60000 [============>.................] - ETA: 1:10 - loss: 0.2831 - categorical_accuracy: 0.9133
26784/60000 [============>.................] - ETA: 1:10 - loss: 0.2828 - categorical_accuracy: 0.9133
26816/60000 [============>.................] - ETA: 1:10 - loss: 0.2828 - categorical_accuracy: 0.9133
26848/60000 [============>.................] - ETA: 1:10 - loss: 0.2826 - categorical_accuracy: 0.9134
26880/60000 [============>.................] - ETA: 1:10 - loss: 0.2825 - categorical_accuracy: 0.9134
26912/60000 [============>.................] - ETA: 1:10 - loss: 0.2824 - categorical_accuracy: 0.9134
26944/60000 [============>.................] - ETA: 1:10 - loss: 0.2822 - categorical_accuracy: 0.9135
26976/60000 [============>.................] - ETA: 1:10 - loss: 0.2819 - categorical_accuracy: 0.9136
27008/60000 [============>.................] - ETA: 1:10 - loss: 0.2817 - categorical_accuracy: 0.9136
27040/60000 [============>.................] - ETA: 1:10 - loss: 0.2814 - categorical_accuracy: 0.9137
27072/60000 [============>.................] - ETA: 1:09 - loss: 0.2812 - categorical_accuracy: 0.9138
27104/60000 [============>.................] - ETA: 1:09 - loss: 0.2812 - categorical_accuracy: 0.9138
27136/60000 [============>.................] - ETA: 1:09 - loss: 0.2810 - categorical_accuracy: 0.9139
27168/60000 [============>.................] - ETA: 1:09 - loss: 0.2808 - categorical_accuracy: 0.9140
27200/60000 [============>.................] - ETA: 1:09 - loss: 0.2807 - categorical_accuracy: 0.9140
27232/60000 [============>.................] - ETA: 1:09 - loss: 0.2805 - categorical_accuracy: 0.9141
27264/60000 [============>.................] - ETA: 1:09 - loss: 0.2803 - categorical_accuracy: 0.9141
27296/60000 [============>.................] - ETA: 1:09 - loss: 0.2801 - categorical_accuracy: 0.9142
27328/60000 [============>.................] - ETA: 1:09 - loss: 0.2798 - categorical_accuracy: 0.9143
27360/60000 [============>.................] - ETA: 1:09 - loss: 0.2795 - categorical_accuracy: 0.9144
27392/60000 [============>.................] - ETA: 1:09 - loss: 0.2796 - categorical_accuracy: 0.9144
27424/60000 [============>.................] - ETA: 1:09 - loss: 0.2795 - categorical_accuracy: 0.9144
27456/60000 [============>.................] - ETA: 1:09 - loss: 0.2793 - categorical_accuracy: 0.9144
27488/60000 [============>.................] - ETA: 1:09 - loss: 0.2792 - categorical_accuracy: 0.9145
27520/60000 [============>.................] - ETA: 1:08 - loss: 0.2790 - categorical_accuracy: 0.9146
27552/60000 [============>.................] - ETA: 1:08 - loss: 0.2789 - categorical_accuracy: 0.9146
27584/60000 [============>.................] - ETA: 1:08 - loss: 0.2789 - categorical_accuracy: 0.9146
27616/60000 [============>.................] - ETA: 1:08 - loss: 0.2789 - categorical_accuracy: 0.9147
27648/60000 [============>.................] - ETA: 1:08 - loss: 0.2787 - categorical_accuracy: 0.9147
27680/60000 [============>.................] - ETA: 1:08 - loss: 0.2784 - categorical_accuracy: 0.9148
27712/60000 [============>.................] - ETA: 1:08 - loss: 0.2781 - categorical_accuracy: 0.9149
27744/60000 [============>.................] - ETA: 1:08 - loss: 0.2779 - categorical_accuracy: 0.9150
27776/60000 [============>.................] - ETA: 1:08 - loss: 0.2778 - categorical_accuracy: 0.9150
27808/60000 [============>.................] - ETA: 1:08 - loss: 0.2775 - categorical_accuracy: 0.9151
27840/60000 [============>.................] - ETA: 1:08 - loss: 0.2773 - categorical_accuracy: 0.9152
27872/60000 [============>.................] - ETA: 1:08 - loss: 0.2770 - categorical_accuracy: 0.9153
27904/60000 [============>.................] - ETA: 1:08 - loss: 0.2767 - categorical_accuracy: 0.9154
27936/60000 [============>.................] - ETA: 1:08 - loss: 0.2765 - categorical_accuracy: 0.9154
27968/60000 [============>.................] - ETA: 1:08 - loss: 0.2762 - categorical_accuracy: 0.9155
28000/60000 [=============>................] - ETA: 1:07 - loss: 0.2760 - categorical_accuracy: 0.9155
28032/60000 [=============>................] - ETA: 1:07 - loss: 0.2760 - categorical_accuracy: 0.9156
28064/60000 [=============>................] - ETA: 1:07 - loss: 0.2758 - categorical_accuracy: 0.9156
28096/60000 [=============>................] - ETA: 1:07 - loss: 0.2755 - categorical_accuracy: 0.9157
28128/60000 [=============>................] - ETA: 1:07 - loss: 0.2752 - categorical_accuracy: 0.9158
28160/60000 [=============>................] - ETA: 1:07 - loss: 0.2750 - categorical_accuracy: 0.9159
28192/60000 [=============>................] - ETA: 1:07 - loss: 0.2748 - categorical_accuracy: 0.9160
28224/60000 [=============>................] - ETA: 1:07 - loss: 0.2745 - categorical_accuracy: 0.9161
28256/60000 [=============>................] - ETA: 1:07 - loss: 0.2749 - categorical_accuracy: 0.9161
28288/60000 [=============>................] - ETA: 1:07 - loss: 0.2746 - categorical_accuracy: 0.9161
28320/60000 [=============>................] - ETA: 1:07 - loss: 0.2744 - categorical_accuracy: 0.9162
28352/60000 [=============>................] - ETA: 1:07 - loss: 0.2746 - categorical_accuracy: 0.9162
28384/60000 [=============>................] - ETA: 1:07 - loss: 0.2745 - categorical_accuracy: 0.9162
28416/60000 [=============>................] - ETA: 1:07 - loss: 0.2745 - categorical_accuracy: 0.9161
28448/60000 [=============>................] - ETA: 1:07 - loss: 0.2743 - categorical_accuracy: 0.9162
28480/60000 [=============>................] - ETA: 1:06 - loss: 0.2743 - categorical_accuracy: 0.9162
28512/60000 [=============>................] - ETA: 1:06 - loss: 0.2742 - categorical_accuracy: 0.9162
28544/60000 [=============>................] - ETA: 1:06 - loss: 0.2739 - categorical_accuracy: 0.9163
28576/60000 [=============>................] - ETA: 1:06 - loss: 0.2737 - categorical_accuracy: 0.9164
28608/60000 [=============>................] - ETA: 1:06 - loss: 0.2736 - categorical_accuracy: 0.9165
28640/60000 [=============>................] - ETA: 1:06 - loss: 0.2735 - categorical_accuracy: 0.9165
28672/60000 [=============>................] - ETA: 1:06 - loss: 0.2734 - categorical_accuracy: 0.9165
28704/60000 [=============>................] - ETA: 1:06 - loss: 0.2731 - categorical_accuracy: 0.9166
28736/60000 [=============>................] - ETA: 1:06 - loss: 0.2730 - categorical_accuracy: 0.9166
28768/60000 [=============>................] - ETA: 1:06 - loss: 0.2728 - categorical_accuracy: 0.9167
28800/60000 [=============>................] - ETA: 1:06 - loss: 0.2728 - categorical_accuracy: 0.9168
28832/60000 [=============>................] - ETA: 1:06 - loss: 0.2727 - categorical_accuracy: 0.9168
28864/60000 [=============>................] - ETA: 1:06 - loss: 0.2728 - categorical_accuracy: 0.9167
28896/60000 [=============>................] - ETA: 1:06 - loss: 0.2726 - categorical_accuracy: 0.9167
28928/60000 [=============>................] - ETA: 1:06 - loss: 0.2725 - categorical_accuracy: 0.9167
28960/60000 [=============>................] - ETA: 1:05 - loss: 0.2722 - categorical_accuracy: 0.9168
28992/60000 [=============>................] - ETA: 1:05 - loss: 0.2721 - categorical_accuracy: 0.9168
29024/60000 [=============>................] - ETA: 1:05 - loss: 0.2720 - categorical_accuracy: 0.9169
29056/60000 [=============>................] - ETA: 1:05 - loss: 0.2717 - categorical_accuracy: 0.9170
29088/60000 [=============>................] - ETA: 1:05 - loss: 0.2715 - categorical_accuracy: 0.9170
29120/60000 [=============>................] - ETA: 1:05 - loss: 0.2713 - categorical_accuracy: 0.9171
29152/60000 [=============>................] - ETA: 1:05 - loss: 0.2712 - categorical_accuracy: 0.9170
29184/60000 [=============>................] - ETA: 1:05 - loss: 0.2710 - categorical_accuracy: 0.9171
29216/60000 [=============>................] - ETA: 1:05 - loss: 0.2708 - categorical_accuracy: 0.9171
29248/60000 [=============>................] - ETA: 1:05 - loss: 0.2706 - categorical_accuracy: 0.9172
29280/60000 [=============>................] - ETA: 1:05 - loss: 0.2703 - categorical_accuracy: 0.9173
29312/60000 [=============>................] - ETA: 1:05 - loss: 0.2701 - categorical_accuracy: 0.9174
29344/60000 [=============>................] - ETA: 1:05 - loss: 0.2698 - categorical_accuracy: 0.9174
29376/60000 [=============>................] - ETA: 1:05 - loss: 0.2696 - categorical_accuracy: 0.9175
29408/60000 [=============>................] - ETA: 1:04 - loss: 0.2695 - categorical_accuracy: 0.9175
29440/60000 [=============>................] - ETA: 1:04 - loss: 0.2695 - categorical_accuracy: 0.9176
29472/60000 [=============>................] - ETA: 1:04 - loss: 0.2693 - categorical_accuracy: 0.9176
29504/60000 [=============>................] - ETA: 1:04 - loss: 0.2693 - categorical_accuracy: 0.9176
29536/60000 [=============>................] - ETA: 1:04 - loss: 0.2692 - categorical_accuracy: 0.9177
29568/60000 [=============>................] - ETA: 1:04 - loss: 0.2691 - categorical_accuracy: 0.9177
29600/60000 [=============>................] - ETA: 1:04 - loss: 0.2690 - categorical_accuracy: 0.9177
29632/60000 [=============>................] - ETA: 1:04 - loss: 0.2691 - categorical_accuracy: 0.9177
29664/60000 [=============>................] - ETA: 1:04 - loss: 0.2692 - categorical_accuracy: 0.9177
29696/60000 [=============>................] - ETA: 1:04 - loss: 0.2689 - categorical_accuracy: 0.9178
29728/60000 [=============>................] - ETA: 1:04 - loss: 0.2687 - categorical_accuracy: 0.9179
29760/60000 [=============>................] - ETA: 1:04 - loss: 0.2685 - categorical_accuracy: 0.9179
29792/60000 [=============>................] - ETA: 1:04 - loss: 0.2684 - categorical_accuracy: 0.9180
29824/60000 [=============>................] - ETA: 1:04 - loss: 0.2681 - categorical_accuracy: 0.9181
29856/60000 [=============>................] - ETA: 1:03 - loss: 0.2679 - categorical_accuracy: 0.9182
29888/60000 [=============>................] - ETA: 1:03 - loss: 0.2681 - categorical_accuracy: 0.9182
29920/60000 [=============>................] - ETA: 1:03 - loss: 0.2679 - categorical_accuracy: 0.9182
29952/60000 [=============>................] - ETA: 1:03 - loss: 0.2677 - categorical_accuracy: 0.9183
29984/60000 [=============>................] - ETA: 1:03 - loss: 0.2675 - categorical_accuracy: 0.9184
30016/60000 [==============>...............] - ETA: 1:03 - loss: 0.2673 - categorical_accuracy: 0.9185
30048/60000 [==============>...............] - ETA: 1:03 - loss: 0.2671 - categorical_accuracy: 0.9186
30080/60000 [==============>...............] - ETA: 1:03 - loss: 0.2670 - categorical_accuracy: 0.9186
30112/60000 [==============>...............] - ETA: 1:03 - loss: 0.2668 - categorical_accuracy: 0.9186
30144/60000 [==============>...............] - ETA: 1:03 - loss: 0.2665 - categorical_accuracy: 0.9187
30176/60000 [==============>...............] - ETA: 1:03 - loss: 0.2665 - categorical_accuracy: 0.9187
30208/60000 [==============>...............] - ETA: 1:03 - loss: 0.2662 - categorical_accuracy: 0.9188
30240/60000 [==============>...............] - ETA: 1:03 - loss: 0.2660 - categorical_accuracy: 0.9188
30272/60000 [==============>...............] - ETA: 1:03 - loss: 0.2658 - categorical_accuracy: 0.9189
30304/60000 [==============>...............] - ETA: 1:02 - loss: 0.2657 - categorical_accuracy: 0.9190
30336/60000 [==============>...............] - ETA: 1:02 - loss: 0.2656 - categorical_accuracy: 0.9190
30368/60000 [==============>...............] - ETA: 1:02 - loss: 0.2654 - categorical_accuracy: 0.9190
30400/60000 [==============>...............] - ETA: 1:02 - loss: 0.2652 - categorical_accuracy: 0.9191
30432/60000 [==============>...............] - ETA: 1:02 - loss: 0.2651 - categorical_accuracy: 0.9191
30464/60000 [==============>...............] - ETA: 1:02 - loss: 0.2651 - categorical_accuracy: 0.9191
30496/60000 [==============>...............] - ETA: 1:02 - loss: 0.2654 - categorical_accuracy: 0.9191
30528/60000 [==============>...............] - ETA: 1:02 - loss: 0.2652 - categorical_accuracy: 0.9192
30560/60000 [==============>...............] - ETA: 1:02 - loss: 0.2650 - categorical_accuracy: 0.9192
30592/60000 [==============>...............] - ETA: 1:02 - loss: 0.2649 - categorical_accuracy: 0.9192
30624/60000 [==============>...............] - ETA: 1:02 - loss: 0.2647 - categorical_accuracy: 0.9192
30656/60000 [==============>...............] - ETA: 1:02 - loss: 0.2645 - categorical_accuracy: 0.9193
30688/60000 [==============>...............] - ETA: 1:02 - loss: 0.2643 - categorical_accuracy: 0.9194
30720/60000 [==============>...............] - ETA: 1:02 - loss: 0.2641 - categorical_accuracy: 0.9195
30752/60000 [==============>...............] - ETA: 1:02 - loss: 0.2638 - categorical_accuracy: 0.9195
30784/60000 [==============>...............] - ETA: 1:02 - loss: 0.2636 - categorical_accuracy: 0.9196
30816/60000 [==============>...............] - ETA: 1:01 - loss: 0.2635 - categorical_accuracy: 0.9197
30848/60000 [==============>...............] - ETA: 1:01 - loss: 0.2632 - categorical_accuracy: 0.9198
30880/60000 [==============>...............] - ETA: 1:01 - loss: 0.2630 - categorical_accuracy: 0.9199
30912/60000 [==============>...............] - ETA: 1:01 - loss: 0.2631 - categorical_accuracy: 0.9199
30944/60000 [==============>...............] - ETA: 1:01 - loss: 0.2629 - categorical_accuracy: 0.9200
30976/60000 [==============>...............] - ETA: 1:01 - loss: 0.2627 - categorical_accuracy: 0.9200
31008/60000 [==============>...............] - ETA: 1:01 - loss: 0.2625 - categorical_accuracy: 0.9201
31040/60000 [==============>...............] - ETA: 1:01 - loss: 0.2623 - categorical_accuracy: 0.9201
31072/60000 [==============>...............] - ETA: 1:01 - loss: 0.2621 - categorical_accuracy: 0.9202
31104/60000 [==============>...............] - ETA: 1:01 - loss: 0.2622 - categorical_accuracy: 0.9202
31136/60000 [==============>...............] - ETA: 1:01 - loss: 0.2624 - categorical_accuracy: 0.9202
31168/60000 [==============>...............] - ETA: 1:01 - loss: 0.2621 - categorical_accuracy: 0.9203
31200/60000 [==============>...............] - ETA: 1:01 - loss: 0.2621 - categorical_accuracy: 0.9203
31232/60000 [==============>...............] - ETA: 1:01 - loss: 0.2620 - categorical_accuracy: 0.9203
31264/60000 [==============>...............] - ETA: 1:01 - loss: 0.2617 - categorical_accuracy: 0.9204
31296/60000 [==============>...............] - ETA: 1:00 - loss: 0.2616 - categorical_accuracy: 0.9205
31328/60000 [==============>...............] - ETA: 1:00 - loss: 0.2614 - categorical_accuracy: 0.9205
31360/60000 [==============>...............] - ETA: 1:00 - loss: 0.2613 - categorical_accuracy: 0.9206
31392/60000 [==============>...............] - ETA: 1:00 - loss: 0.2611 - categorical_accuracy: 0.9206
31424/60000 [==============>...............] - ETA: 1:00 - loss: 0.2608 - categorical_accuracy: 0.9207
31456/60000 [==============>...............] - ETA: 1:00 - loss: 0.2607 - categorical_accuracy: 0.9207
31488/60000 [==============>...............] - ETA: 1:00 - loss: 0.2605 - categorical_accuracy: 0.9208
31520/60000 [==============>...............] - ETA: 1:00 - loss: 0.2604 - categorical_accuracy: 0.9208
31552/60000 [==============>...............] - ETA: 1:00 - loss: 0.2601 - categorical_accuracy: 0.9209
31584/60000 [==============>...............] - ETA: 1:00 - loss: 0.2600 - categorical_accuracy: 0.9210
31616/60000 [==============>...............] - ETA: 1:00 - loss: 0.2597 - categorical_accuracy: 0.9211
31648/60000 [==============>...............] - ETA: 1:00 - loss: 0.2596 - categorical_accuracy: 0.9211
31680/60000 [==============>...............] - ETA: 1:00 - loss: 0.2595 - categorical_accuracy: 0.9211
31712/60000 [==============>...............] - ETA: 1:00 - loss: 0.2593 - categorical_accuracy: 0.9212
31744/60000 [==============>...............] - ETA: 1:00 - loss: 0.2591 - categorical_accuracy: 0.9212
31776/60000 [==============>...............] - ETA: 1:00 - loss: 0.2590 - categorical_accuracy: 0.9213
31808/60000 [==============>...............] - ETA: 59s - loss: 0.2591 - categorical_accuracy: 0.9212 
31840/60000 [==============>...............] - ETA: 59s - loss: 0.2589 - categorical_accuracy: 0.9213
31872/60000 [==============>...............] - ETA: 59s - loss: 0.2587 - categorical_accuracy: 0.9214
31904/60000 [==============>...............] - ETA: 59s - loss: 0.2585 - categorical_accuracy: 0.9214
31936/60000 [==============>...............] - ETA: 59s - loss: 0.2583 - categorical_accuracy: 0.9215
31968/60000 [==============>...............] - ETA: 59s - loss: 0.2582 - categorical_accuracy: 0.9215
32000/60000 [===============>..............] - ETA: 59s - loss: 0.2581 - categorical_accuracy: 0.9216
32032/60000 [===============>..............] - ETA: 59s - loss: 0.2579 - categorical_accuracy: 0.9216
32064/60000 [===============>..............] - ETA: 59s - loss: 0.2577 - categorical_accuracy: 0.9217
32096/60000 [===============>..............] - ETA: 59s - loss: 0.2575 - categorical_accuracy: 0.9218
32128/60000 [===============>..............] - ETA: 59s - loss: 0.2573 - categorical_accuracy: 0.9218
32160/60000 [===============>..............] - ETA: 59s - loss: 0.2571 - categorical_accuracy: 0.9219
32192/60000 [===============>..............] - ETA: 59s - loss: 0.2569 - categorical_accuracy: 0.9219
32224/60000 [===============>..............] - ETA: 59s - loss: 0.2567 - categorical_accuracy: 0.9220
32256/60000 [===============>..............] - ETA: 59s - loss: 0.2568 - categorical_accuracy: 0.9220
32288/60000 [===============>..............] - ETA: 59s - loss: 0.2568 - categorical_accuracy: 0.9220
32320/60000 [===============>..............] - ETA: 58s - loss: 0.2565 - categorical_accuracy: 0.9221
32352/60000 [===============>..............] - ETA: 58s - loss: 0.2566 - categorical_accuracy: 0.9221
32384/60000 [===============>..............] - ETA: 58s - loss: 0.2563 - categorical_accuracy: 0.9222
32416/60000 [===============>..............] - ETA: 58s - loss: 0.2562 - categorical_accuracy: 0.9222
32448/60000 [===============>..............] - ETA: 58s - loss: 0.2560 - categorical_accuracy: 0.9223
32480/60000 [===============>..............] - ETA: 58s - loss: 0.2558 - categorical_accuracy: 0.9223
32512/60000 [===============>..............] - ETA: 58s - loss: 0.2557 - categorical_accuracy: 0.9224
32544/60000 [===============>..............] - ETA: 58s - loss: 0.2556 - categorical_accuracy: 0.9224
32576/60000 [===============>..............] - ETA: 58s - loss: 0.2554 - categorical_accuracy: 0.9224
32608/60000 [===============>..............] - ETA: 58s - loss: 0.2553 - categorical_accuracy: 0.9224
32640/60000 [===============>..............] - ETA: 58s - loss: 0.2553 - categorical_accuracy: 0.9224
32672/60000 [===============>..............] - ETA: 58s - loss: 0.2551 - categorical_accuracy: 0.9225
32704/60000 [===============>..............] - ETA: 58s - loss: 0.2549 - categorical_accuracy: 0.9225
32736/60000 [===============>..............] - ETA: 58s - loss: 0.2547 - categorical_accuracy: 0.9226
32768/60000 [===============>..............] - ETA: 58s - loss: 0.2546 - categorical_accuracy: 0.9227
32800/60000 [===============>..............] - ETA: 58s - loss: 0.2544 - categorical_accuracy: 0.9227
32832/60000 [===============>..............] - ETA: 57s - loss: 0.2542 - categorical_accuracy: 0.9228
32864/60000 [===============>..............] - ETA: 57s - loss: 0.2540 - categorical_accuracy: 0.9228
32896/60000 [===============>..............] - ETA: 57s - loss: 0.2539 - categorical_accuracy: 0.9229
32928/60000 [===============>..............] - ETA: 57s - loss: 0.2537 - categorical_accuracy: 0.9230
32960/60000 [===============>..............] - ETA: 57s - loss: 0.2536 - categorical_accuracy: 0.9230
32992/60000 [===============>..............] - ETA: 57s - loss: 0.2534 - categorical_accuracy: 0.9230
33024/60000 [===============>..............] - ETA: 57s - loss: 0.2533 - categorical_accuracy: 0.9231
33056/60000 [===============>..............] - ETA: 57s - loss: 0.2535 - categorical_accuracy: 0.9230
33088/60000 [===============>..............] - ETA: 57s - loss: 0.2533 - categorical_accuracy: 0.9231
33120/60000 [===============>..............] - ETA: 57s - loss: 0.2531 - categorical_accuracy: 0.9232
33152/60000 [===============>..............] - ETA: 57s - loss: 0.2529 - categorical_accuracy: 0.9232
33184/60000 [===============>..............] - ETA: 57s - loss: 0.2527 - categorical_accuracy: 0.9233
33216/60000 [===============>..............] - ETA: 57s - loss: 0.2525 - categorical_accuracy: 0.9233
33248/60000 [===============>..............] - ETA: 57s - loss: 0.2523 - categorical_accuracy: 0.9234
33280/60000 [===============>..............] - ETA: 57s - loss: 0.2521 - categorical_accuracy: 0.9234
33312/60000 [===============>..............] - ETA: 56s - loss: 0.2520 - categorical_accuracy: 0.9234
33344/60000 [===============>..............] - ETA: 56s - loss: 0.2518 - categorical_accuracy: 0.9235
33376/60000 [===============>..............] - ETA: 56s - loss: 0.2517 - categorical_accuracy: 0.9235
33408/60000 [===============>..............] - ETA: 56s - loss: 0.2516 - categorical_accuracy: 0.9236
33440/60000 [===============>..............] - ETA: 56s - loss: 0.2515 - categorical_accuracy: 0.9236
33472/60000 [===============>..............] - ETA: 56s - loss: 0.2514 - categorical_accuracy: 0.9236
33504/60000 [===============>..............] - ETA: 56s - loss: 0.2512 - categorical_accuracy: 0.9237
33536/60000 [===============>..............] - ETA: 56s - loss: 0.2511 - categorical_accuracy: 0.9237
33568/60000 [===============>..............] - ETA: 56s - loss: 0.2509 - categorical_accuracy: 0.9238
33600/60000 [===============>..............] - ETA: 56s - loss: 0.2507 - categorical_accuracy: 0.9238
33632/60000 [===============>..............] - ETA: 56s - loss: 0.2505 - categorical_accuracy: 0.9239
33664/60000 [===============>..............] - ETA: 56s - loss: 0.2503 - categorical_accuracy: 0.9240
33696/60000 [===============>..............] - ETA: 56s - loss: 0.2501 - categorical_accuracy: 0.9241
33728/60000 [===============>..............] - ETA: 56s - loss: 0.2499 - categorical_accuracy: 0.9241
33760/60000 [===============>..............] - ETA: 56s - loss: 0.2496 - categorical_accuracy: 0.9242
33792/60000 [===============>..............] - ETA: 56s - loss: 0.2496 - categorical_accuracy: 0.9242
33824/60000 [===============>..............] - ETA: 55s - loss: 0.2494 - categorical_accuracy: 0.9243
33856/60000 [===============>..............] - ETA: 55s - loss: 0.2492 - categorical_accuracy: 0.9243
33888/60000 [===============>..............] - ETA: 55s - loss: 0.2491 - categorical_accuracy: 0.9244
33920/60000 [===============>..............] - ETA: 55s - loss: 0.2489 - categorical_accuracy: 0.9244
33952/60000 [===============>..............] - ETA: 55s - loss: 0.2487 - categorical_accuracy: 0.9245
33984/60000 [===============>..............] - ETA: 55s - loss: 0.2485 - categorical_accuracy: 0.9246
34016/60000 [================>.............] - ETA: 55s - loss: 0.2485 - categorical_accuracy: 0.9246
34048/60000 [================>.............] - ETA: 55s - loss: 0.2484 - categorical_accuracy: 0.9246
34080/60000 [================>.............] - ETA: 55s - loss: 0.2483 - categorical_accuracy: 0.9246
34112/60000 [================>.............] - ETA: 55s - loss: 0.2482 - categorical_accuracy: 0.9247
34144/60000 [================>.............] - ETA: 55s - loss: 0.2482 - categorical_accuracy: 0.9247
34176/60000 [================>.............] - ETA: 55s - loss: 0.2480 - categorical_accuracy: 0.9247
34208/60000 [================>.............] - ETA: 55s - loss: 0.2482 - categorical_accuracy: 0.9247
34240/60000 [================>.............] - ETA: 55s - loss: 0.2482 - categorical_accuracy: 0.9246
34272/60000 [================>.............] - ETA: 54s - loss: 0.2481 - categorical_accuracy: 0.9246
34304/60000 [================>.............] - ETA: 54s - loss: 0.2479 - categorical_accuracy: 0.9247
34336/60000 [================>.............] - ETA: 54s - loss: 0.2479 - categorical_accuracy: 0.9247
34368/60000 [================>.............] - ETA: 54s - loss: 0.2477 - categorical_accuracy: 0.9248
34400/60000 [================>.............] - ETA: 54s - loss: 0.2476 - categorical_accuracy: 0.9248
34432/60000 [================>.............] - ETA: 54s - loss: 0.2474 - categorical_accuracy: 0.9249
34464/60000 [================>.............] - ETA: 54s - loss: 0.2472 - categorical_accuracy: 0.9250
34496/60000 [================>.............] - ETA: 54s - loss: 0.2469 - categorical_accuracy: 0.9250
34528/60000 [================>.............] - ETA: 54s - loss: 0.2468 - categorical_accuracy: 0.9251
34560/60000 [================>.............] - ETA: 54s - loss: 0.2467 - categorical_accuracy: 0.9251
34592/60000 [================>.............] - ETA: 54s - loss: 0.2464 - categorical_accuracy: 0.9252
34624/60000 [================>.............] - ETA: 54s - loss: 0.2464 - categorical_accuracy: 0.9252
34656/60000 [================>.............] - ETA: 54s - loss: 0.2462 - categorical_accuracy: 0.9252
34688/60000 [================>.............] - ETA: 54s - loss: 0.2464 - categorical_accuracy: 0.9252
34720/60000 [================>.............] - ETA: 53s - loss: 0.2462 - categorical_accuracy: 0.9253
34752/60000 [================>.............] - ETA: 53s - loss: 0.2462 - categorical_accuracy: 0.9253
34784/60000 [================>.............] - ETA: 53s - loss: 0.2460 - categorical_accuracy: 0.9253
34816/60000 [================>.............] - ETA: 53s - loss: 0.2458 - categorical_accuracy: 0.9254
34848/60000 [================>.............] - ETA: 53s - loss: 0.2458 - categorical_accuracy: 0.9254
34880/60000 [================>.............] - ETA: 53s - loss: 0.2457 - categorical_accuracy: 0.9254
34912/60000 [================>.............] - ETA: 53s - loss: 0.2455 - categorical_accuracy: 0.9255
34944/60000 [================>.............] - ETA: 53s - loss: 0.2453 - categorical_accuracy: 0.9255
34976/60000 [================>.............] - ETA: 53s - loss: 0.2452 - categorical_accuracy: 0.9255
35008/60000 [================>.............] - ETA: 53s - loss: 0.2450 - categorical_accuracy: 0.9256
35040/60000 [================>.............] - ETA: 53s - loss: 0.2448 - categorical_accuracy: 0.9257
35072/60000 [================>.............] - ETA: 53s - loss: 0.2447 - categorical_accuracy: 0.9257
35104/60000 [================>.............] - ETA: 53s - loss: 0.2445 - categorical_accuracy: 0.9258
35136/60000 [================>.............] - ETA: 53s - loss: 0.2444 - categorical_accuracy: 0.9258
35168/60000 [================>.............] - ETA: 52s - loss: 0.2443 - categorical_accuracy: 0.9259
35200/60000 [================>.............] - ETA: 52s - loss: 0.2441 - categorical_accuracy: 0.9259
35232/60000 [================>.............] - ETA: 52s - loss: 0.2439 - categorical_accuracy: 0.9260
35264/60000 [================>.............] - ETA: 52s - loss: 0.2437 - categorical_accuracy: 0.9261
35296/60000 [================>.............] - ETA: 52s - loss: 0.2437 - categorical_accuracy: 0.9261
35328/60000 [================>.............] - ETA: 52s - loss: 0.2435 - categorical_accuracy: 0.9261
35360/60000 [================>.............] - ETA: 52s - loss: 0.2433 - categorical_accuracy: 0.9262
35392/60000 [================>.............] - ETA: 52s - loss: 0.2431 - categorical_accuracy: 0.9263
35424/60000 [================>.............] - ETA: 52s - loss: 0.2430 - categorical_accuracy: 0.9263
35456/60000 [================>.............] - ETA: 52s - loss: 0.2429 - categorical_accuracy: 0.9263
35488/60000 [================>.............] - ETA: 52s - loss: 0.2428 - categorical_accuracy: 0.9263
35520/60000 [================>.............] - ETA: 52s - loss: 0.2428 - categorical_accuracy: 0.9263
35552/60000 [================>.............] - ETA: 52s - loss: 0.2428 - categorical_accuracy: 0.9263
35584/60000 [================>.............] - ETA: 52s - loss: 0.2429 - categorical_accuracy: 0.9263
35616/60000 [================>.............] - ETA: 51s - loss: 0.2429 - categorical_accuracy: 0.9263
35648/60000 [================>.............] - ETA: 51s - loss: 0.2427 - categorical_accuracy: 0.9264
35680/60000 [================>.............] - ETA: 51s - loss: 0.2426 - categorical_accuracy: 0.9264
35712/60000 [================>.............] - ETA: 51s - loss: 0.2424 - categorical_accuracy: 0.9265
35744/60000 [================>.............] - ETA: 51s - loss: 0.2423 - categorical_accuracy: 0.9265
35776/60000 [================>.............] - ETA: 51s - loss: 0.2421 - categorical_accuracy: 0.9266
35808/60000 [================>.............] - ETA: 51s - loss: 0.2420 - categorical_accuracy: 0.9266
35840/60000 [================>.............] - ETA: 51s - loss: 0.2419 - categorical_accuracy: 0.9266
35872/60000 [================>.............] - ETA: 51s - loss: 0.2417 - categorical_accuracy: 0.9266
35904/60000 [================>.............] - ETA: 51s - loss: 0.2417 - categorical_accuracy: 0.9266
35936/60000 [================>.............] - ETA: 51s - loss: 0.2415 - categorical_accuracy: 0.9267
35968/60000 [================>.............] - ETA: 51s - loss: 0.2416 - categorical_accuracy: 0.9267
36000/60000 [=================>............] - ETA: 51s - loss: 0.2415 - categorical_accuracy: 0.9266
36032/60000 [=================>............] - ETA: 51s - loss: 0.2416 - categorical_accuracy: 0.9266
36064/60000 [=================>............] - ETA: 50s - loss: 0.2415 - categorical_accuracy: 0.9266
36096/60000 [=================>............] - ETA: 50s - loss: 0.2414 - categorical_accuracy: 0.9266
36128/60000 [=================>............] - ETA: 50s - loss: 0.2414 - categorical_accuracy: 0.9267
36160/60000 [=================>............] - ETA: 50s - loss: 0.2413 - categorical_accuracy: 0.9267
36192/60000 [=================>............] - ETA: 50s - loss: 0.2412 - categorical_accuracy: 0.9268
36224/60000 [=================>............] - ETA: 50s - loss: 0.2410 - categorical_accuracy: 0.9268
36256/60000 [=================>............] - ETA: 50s - loss: 0.2409 - categorical_accuracy: 0.9269
36288/60000 [=================>............] - ETA: 50s - loss: 0.2408 - categorical_accuracy: 0.9269
36320/60000 [=================>............] - ETA: 50s - loss: 0.2408 - categorical_accuracy: 0.9269
36352/60000 [=================>............] - ETA: 50s - loss: 0.2406 - categorical_accuracy: 0.9269
36384/60000 [=================>............] - ETA: 50s - loss: 0.2405 - categorical_accuracy: 0.9269
36416/60000 [=================>............] - ETA: 50s - loss: 0.2403 - categorical_accuracy: 0.9270
36448/60000 [=================>............] - ETA: 50s - loss: 0.2401 - categorical_accuracy: 0.9270
36480/60000 [=================>............] - ETA: 50s - loss: 0.2401 - categorical_accuracy: 0.9271
36512/60000 [=================>............] - ETA: 49s - loss: 0.2399 - categorical_accuracy: 0.9271
36544/60000 [=================>............] - ETA: 49s - loss: 0.2397 - categorical_accuracy: 0.9272
36576/60000 [=================>............] - ETA: 49s - loss: 0.2396 - categorical_accuracy: 0.9272
36608/60000 [=================>............] - ETA: 49s - loss: 0.2394 - categorical_accuracy: 0.9273
36640/60000 [=================>............] - ETA: 49s - loss: 0.2393 - categorical_accuracy: 0.9273
36672/60000 [=================>............] - ETA: 49s - loss: 0.2392 - categorical_accuracy: 0.9273
36704/60000 [=================>............] - ETA: 49s - loss: 0.2391 - categorical_accuracy: 0.9273
36736/60000 [=================>............] - ETA: 49s - loss: 0.2389 - categorical_accuracy: 0.9274
36768/60000 [=================>............] - ETA: 49s - loss: 0.2388 - categorical_accuracy: 0.9274
36800/60000 [=================>............] - ETA: 49s - loss: 0.2386 - categorical_accuracy: 0.9274
36832/60000 [=================>............] - ETA: 49s - loss: 0.2385 - categorical_accuracy: 0.9275
36864/60000 [=================>............] - ETA: 49s - loss: 0.2384 - categorical_accuracy: 0.9275
36896/60000 [=================>............] - ETA: 49s - loss: 0.2382 - categorical_accuracy: 0.9276
36928/60000 [=================>............] - ETA: 49s - loss: 0.2382 - categorical_accuracy: 0.9276
36960/60000 [=================>............] - ETA: 48s - loss: 0.2383 - categorical_accuracy: 0.9275
36992/60000 [=================>............] - ETA: 48s - loss: 0.2383 - categorical_accuracy: 0.9275
37024/60000 [=================>............] - ETA: 48s - loss: 0.2381 - categorical_accuracy: 0.9276
37056/60000 [=================>............] - ETA: 48s - loss: 0.2383 - categorical_accuracy: 0.9276
37088/60000 [=================>............] - ETA: 48s - loss: 0.2382 - categorical_accuracy: 0.9276
37120/60000 [=================>............] - ETA: 48s - loss: 0.2383 - categorical_accuracy: 0.9276
37152/60000 [=================>............] - ETA: 48s - loss: 0.2382 - categorical_accuracy: 0.9276
37184/60000 [=================>............] - ETA: 48s - loss: 0.2380 - categorical_accuracy: 0.9276
37216/60000 [=================>............] - ETA: 48s - loss: 0.2379 - categorical_accuracy: 0.9277
37248/60000 [=================>............] - ETA: 48s - loss: 0.2378 - categorical_accuracy: 0.9277
37280/60000 [=================>............] - ETA: 48s - loss: 0.2376 - categorical_accuracy: 0.9277
37312/60000 [=================>............] - ETA: 48s - loss: 0.2375 - categorical_accuracy: 0.9278
37344/60000 [=================>............] - ETA: 48s - loss: 0.2373 - categorical_accuracy: 0.9278
37376/60000 [=================>............] - ETA: 48s - loss: 0.2373 - categorical_accuracy: 0.9279
37408/60000 [=================>............] - ETA: 47s - loss: 0.2371 - categorical_accuracy: 0.9279
37440/60000 [=================>............] - ETA: 47s - loss: 0.2369 - categorical_accuracy: 0.9280
37472/60000 [=================>............] - ETA: 47s - loss: 0.2369 - categorical_accuracy: 0.9280
37504/60000 [=================>............] - ETA: 47s - loss: 0.2369 - categorical_accuracy: 0.9280
37536/60000 [=================>............] - ETA: 47s - loss: 0.2368 - categorical_accuracy: 0.9280
37568/60000 [=================>............] - ETA: 47s - loss: 0.2368 - categorical_accuracy: 0.9280
37600/60000 [=================>............] - ETA: 47s - loss: 0.2368 - categorical_accuracy: 0.9280
37632/60000 [=================>............] - ETA: 47s - loss: 0.2367 - categorical_accuracy: 0.9280
37664/60000 [=================>............] - ETA: 47s - loss: 0.2365 - categorical_accuracy: 0.9281
37696/60000 [=================>............] - ETA: 47s - loss: 0.2363 - categorical_accuracy: 0.9282
37728/60000 [=================>............] - ETA: 47s - loss: 0.2366 - categorical_accuracy: 0.9281
37760/60000 [=================>............] - ETA: 47s - loss: 0.2365 - categorical_accuracy: 0.9282
37792/60000 [=================>............] - ETA: 47s - loss: 0.2363 - categorical_accuracy: 0.9282
37824/60000 [=================>............] - ETA: 47s - loss: 0.2362 - categorical_accuracy: 0.9282
37856/60000 [=================>............] - ETA: 46s - loss: 0.2360 - categorical_accuracy: 0.9283
37888/60000 [=================>............] - ETA: 46s - loss: 0.2360 - categorical_accuracy: 0.9283
37920/60000 [=================>............] - ETA: 46s - loss: 0.2358 - categorical_accuracy: 0.9283
37952/60000 [=================>............] - ETA: 46s - loss: 0.2358 - categorical_accuracy: 0.9283
37984/60000 [=================>............] - ETA: 46s - loss: 0.2358 - categorical_accuracy: 0.9283
38016/60000 [==================>...........] - ETA: 46s - loss: 0.2361 - categorical_accuracy: 0.9283
38048/60000 [==================>...........] - ETA: 46s - loss: 0.2363 - categorical_accuracy: 0.9282
38080/60000 [==================>...........] - ETA: 46s - loss: 0.2362 - categorical_accuracy: 0.9282
38112/60000 [==================>...........] - ETA: 46s - loss: 0.2361 - categorical_accuracy: 0.9283
38144/60000 [==================>...........] - ETA: 46s - loss: 0.2361 - categorical_accuracy: 0.9283
38176/60000 [==================>...........] - ETA: 46s - loss: 0.2359 - categorical_accuracy: 0.9283
38208/60000 [==================>...........] - ETA: 46s - loss: 0.2359 - categorical_accuracy: 0.9284
38240/60000 [==================>...........] - ETA: 46s - loss: 0.2358 - categorical_accuracy: 0.9284
38272/60000 [==================>...........] - ETA: 46s - loss: 0.2357 - categorical_accuracy: 0.9285
38304/60000 [==================>...........] - ETA: 46s - loss: 0.2355 - categorical_accuracy: 0.9285
38336/60000 [==================>...........] - ETA: 45s - loss: 0.2354 - categorical_accuracy: 0.9285
38368/60000 [==================>...........] - ETA: 45s - loss: 0.2352 - categorical_accuracy: 0.9286
38400/60000 [==================>...........] - ETA: 45s - loss: 0.2351 - categorical_accuracy: 0.9286
38432/60000 [==================>...........] - ETA: 45s - loss: 0.2352 - categorical_accuracy: 0.9286
38464/60000 [==================>...........] - ETA: 45s - loss: 0.2351 - categorical_accuracy: 0.9287
38496/60000 [==================>...........] - ETA: 45s - loss: 0.2350 - categorical_accuracy: 0.9287
38528/60000 [==================>...........] - ETA: 45s - loss: 0.2348 - categorical_accuracy: 0.9287
38560/60000 [==================>...........] - ETA: 45s - loss: 0.2347 - categorical_accuracy: 0.9288
38592/60000 [==================>...........] - ETA: 45s - loss: 0.2346 - categorical_accuracy: 0.9288
38624/60000 [==================>...........] - ETA: 45s - loss: 0.2344 - categorical_accuracy: 0.9289
38656/60000 [==================>...........] - ETA: 45s - loss: 0.2343 - categorical_accuracy: 0.9289
38688/60000 [==================>...........] - ETA: 45s - loss: 0.2341 - categorical_accuracy: 0.9289
38720/60000 [==================>...........] - ETA: 45s - loss: 0.2340 - categorical_accuracy: 0.9290
38752/60000 [==================>...........] - ETA: 45s - loss: 0.2338 - categorical_accuracy: 0.9290
38784/60000 [==================>...........] - ETA: 44s - loss: 0.2337 - categorical_accuracy: 0.9291
38816/60000 [==================>...........] - ETA: 44s - loss: 0.2335 - categorical_accuracy: 0.9291
38848/60000 [==================>...........] - ETA: 44s - loss: 0.2335 - categorical_accuracy: 0.9292
38880/60000 [==================>...........] - ETA: 44s - loss: 0.2334 - categorical_accuracy: 0.9292
38912/60000 [==================>...........] - ETA: 44s - loss: 0.2333 - categorical_accuracy: 0.9292
38944/60000 [==================>...........] - ETA: 44s - loss: 0.2331 - categorical_accuracy: 0.9293
38976/60000 [==================>...........] - ETA: 44s - loss: 0.2330 - categorical_accuracy: 0.9293
39008/60000 [==================>...........] - ETA: 44s - loss: 0.2328 - categorical_accuracy: 0.9294
39040/60000 [==================>...........] - ETA: 44s - loss: 0.2327 - categorical_accuracy: 0.9294
39072/60000 [==================>...........] - ETA: 44s - loss: 0.2326 - categorical_accuracy: 0.9295
39104/60000 [==================>...........] - ETA: 44s - loss: 0.2324 - categorical_accuracy: 0.9295
39136/60000 [==================>...........] - ETA: 44s - loss: 0.2322 - categorical_accuracy: 0.9296
39168/60000 [==================>...........] - ETA: 44s - loss: 0.2321 - categorical_accuracy: 0.9296
39200/60000 [==================>...........] - ETA: 44s - loss: 0.2319 - categorical_accuracy: 0.9297
39232/60000 [==================>...........] - ETA: 43s - loss: 0.2320 - categorical_accuracy: 0.9297
39264/60000 [==================>...........] - ETA: 43s - loss: 0.2319 - categorical_accuracy: 0.9297
39296/60000 [==================>...........] - ETA: 43s - loss: 0.2318 - categorical_accuracy: 0.9297
39328/60000 [==================>...........] - ETA: 43s - loss: 0.2316 - categorical_accuracy: 0.9298
39360/60000 [==================>...........] - ETA: 43s - loss: 0.2315 - categorical_accuracy: 0.9298
39392/60000 [==================>...........] - ETA: 43s - loss: 0.2318 - categorical_accuracy: 0.9298
39424/60000 [==================>...........] - ETA: 43s - loss: 0.2317 - categorical_accuracy: 0.9298
39456/60000 [==================>...........] - ETA: 43s - loss: 0.2315 - categorical_accuracy: 0.9299
39488/60000 [==================>...........] - ETA: 43s - loss: 0.2314 - categorical_accuracy: 0.9299
39520/60000 [==================>...........] - ETA: 43s - loss: 0.2313 - categorical_accuracy: 0.9299
39552/60000 [==================>...........] - ETA: 43s - loss: 0.2311 - categorical_accuracy: 0.9300
39584/60000 [==================>...........] - ETA: 43s - loss: 0.2309 - categorical_accuracy: 0.9300
39616/60000 [==================>...........] - ETA: 43s - loss: 0.2308 - categorical_accuracy: 0.9300
39648/60000 [==================>...........] - ETA: 43s - loss: 0.2307 - categorical_accuracy: 0.9300
39680/60000 [==================>...........] - ETA: 43s - loss: 0.2306 - categorical_accuracy: 0.9301
39712/60000 [==================>...........] - ETA: 42s - loss: 0.2304 - categorical_accuracy: 0.9301
39744/60000 [==================>...........] - ETA: 42s - loss: 0.2306 - categorical_accuracy: 0.9301
39776/60000 [==================>...........] - ETA: 42s - loss: 0.2305 - categorical_accuracy: 0.9302
39808/60000 [==================>...........] - ETA: 42s - loss: 0.2303 - categorical_accuracy: 0.9302
39840/60000 [==================>...........] - ETA: 42s - loss: 0.2302 - categorical_accuracy: 0.9302
39872/60000 [==================>...........] - ETA: 42s - loss: 0.2300 - categorical_accuracy: 0.9303
39904/60000 [==================>...........] - ETA: 42s - loss: 0.2300 - categorical_accuracy: 0.9303
39936/60000 [==================>...........] - ETA: 42s - loss: 0.2300 - categorical_accuracy: 0.9302
39968/60000 [==================>...........] - ETA: 42s - loss: 0.2299 - categorical_accuracy: 0.9303
40000/60000 [===================>..........] - ETA: 42s - loss: 0.2297 - categorical_accuracy: 0.9304
40032/60000 [===================>..........] - ETA: 42s - loss: 0.2297 - categorical_accuracy: 0.9304
40064/60000 [===================>..........] - ETA: 42s - loss: 0.2296 - categorical_accuracy: 0.9304
40096/60000 [===================>..........] - ETA: 42s - loss: 0.2294 - categorical_accuracy: 0.9304
40128/60000 [===================>..........] - ETA: 42s - loss: 0.2293 - categorical_accuracy: 0.9305
40160/60000 [===================>..........] - ETA: 41s - loss: 0.2291 - categorical_accuracy: 0.9306
40192/60000 [===================>..........] - ETA: 41s - loss: 0.2289 - categorical_accuracy: 0.9306
40224/60000 [===================>..........] - ETA: 41s - loss: 0.2289 - categorical_accuracy: 0.9306
40256/60000 [===================>..........] - ETA: 41s - loss: 0.2289 - categorical_accuracy: 0.9306
40288/60000 [===================>..........] - ETA: 41s - loss: 0.2288 - categorical_accuracy: 0.9306
40320/60000 [===================>..........] - ETA: 41s - loss: 0.2287 - categorical_accuracy: 0.9307
40352/60000 [===================>..........] - ETA: 41s - loss: 0.2286 - categorical_accuracy: 0.9307
40384/60000 [===================>..........] - ETA: 41s - loss: 0.2285 - categorical_accuracy: 0.9308
40416/60000 [===================>..........] - ETA: 41s - loss: 0.2285 - categorical_accuracy: 0.9308
40448/60000 [===================>..........] - ETA: 41s - loss: 0.2285 - categorical_accuracy: 0.9308
40480/60000 [===================>..........] - ETA: 41s - loss: 0.2284 - categorical_accuracy: 0.9309
40512/60000 [===================>..........] - ETA: 41s - loss: 0.2282 - categorical_accuracy: 0.9309
40544/60000 [===================>..........] - ETA: 41s - loss: 0.2281 - categorical_accuracy: 0.9309
40576/60000 [===================>..........] - ETA: 41s - loss: 0.2280 - categorical_accuracy: 0.9309
40608/60000 [===================>..........] - ETA: 40s - loss: 0.2278 - categorical_accuracy: 0.9310
40640/60000 [===================>..........] - ETA: 40s - loss: 0.2278 - categorical_accuracy: 0.9310
40672/60000 [===================>..........] - ETA: 40s - loss: 0.2278 - categorical_accuracy: 0.9310
40704/60000 [===================>..........] - ETA: 40s - loss: 0.2276 - categorical_accuracy: 0.9311
40736/60000 [===================>..........] - ETA: 40s - loss: 0.2277 - categorical_accuracy: 0.9311
40768/60000 [===================>..........] - ETA: 40s - loss: 0.2276 - categorical_accuracy: 0.9311
40800/60000 [===================>..........] - ETA: 40s - loss: 0.2275 - categorical_accuracy: 0.9312
40832/60000 [===================>..........] - ETA: 40s - loss: 0.2273 - categorical_accuracy: 0.9312
40864/60000 [===================>..........] - ETA: 40s - loss: 0.2272 - categorical_accuracy: 0.9313
40896/60000 [===================>..........] - ETA: 40s - loss: 0.2270 - categorical_accuracy: 0.9313
40928/60000 [===================>..........] - ETA: 40s - loss: 0.2268 - categorical_accuracy: 0.9313
40960/60000 [===================>..........] - ETA: 40s - loss: 0.2267 - categorical_accuracy: 0.9314
40992/60000 [===================>..........] - ETA: 40s - loss: 0.2267 - categorical_accuracy: 0.9314
41024/60000 [===================>..........] - ETA: 40s - loss: 0.2265 - categorical_accuracy: 0.9314
41056/60000 [===================>..........] - ETA: 40s - loss: 0.2264 - categorical_accuracy: 0.9315
41088/60000 [===================>..........] - ETA: 39s - loss: 0.2264 - categorical_accuracy: 0.9314
41120/60000 [===================>..........] - ETA: 39s - loss: 0.2262 - categorical_accuracy: 0.9315
41152/60000 [===================>..........] - ETA: 39s - loss: 0.2262 - categorical_accuracy: 0.9315
41184/60000 [===================>..........] - ETA: 39s - loss: 0.2261 - categorical_accuracy: 0.9315
41216/60000 [===================>..........] - ETA: 39s - loss: 0.2261 - categorical_accuracy: 0.9315
41248/60000 [===================>..........] - ETA: 39s - loss: 0.2259 - categorical_accuracy: 0.9316
41280/60000 [===================>..........] - ETA: 39s - loss: 0.2258 - categorical_accuracy: 0.9316
41312/60000 [===================>..........] - ETA: 39s - loss: 0.2257 - categorical_accuracy: 0.9317
41344/60000 [===================>..........] - ETA: 39s - loss: 0.2257 - categorical_accuracy: 0.9316
41376/60000 [===================>..........] - ETA: 39s - loss: 0.2256 - categorical_accuracy: 0.9317
41408/60000 [===================>..........] - ETA: 39s - loss: 0.2257 - categorical_accuracy: 0.9317
41440/60000 [===================>..........] - ETA: 39s - loss: 0.2256 - categorical_accuracy: 0.9317
41472/60000 [===================>..........] - ETA: 39s - loss: 0.2255 - categorical_accuracy: 0.9317
41504/60000 [===================>..........] - ETA: 39s - loss: 0.2254 - categorical_accuracy: 0.9317
41536/60000 [===================>..........] - ETA: 38s - loss: 0.2252 - categorical_accuracy: 0.9318
41568/60000 [===================>..........] - ETA: 38s - loss: 0.2252 - categorical_accuracy: 0.9318
41600/60000 [===================>..........] - ETA: 38s - loss: 0.2250 - categorical_accuracy: 0.9318
41632/60000 [===================>..........] - ETA: 38s - loss: 0.2249 - categorical_accuracy: 0.9319
41664/60000 [===================>..........] - ETA: 38s - loss: 0.2248 - categorical_accuracy: 0.9319
41696/60000 [===================>..........] - ETA: 38s - loss: 0.2248 - categorical_accuracy: 0.9319
41728/60000 [===================>..........] - ETA: 38s - loss: 0.2247 - categorical_accuracy: 0.9319
41760/60000 [===================>..........] - ETA: 38s - loss: 0.2245 - categorical_accuracy: 0.9320
41792/60000 [===================>..........] - ETA: 38s - loss: 0.2244 - categorical_accuracy: 0.9320
41824/60000 [===================>..........] - ETA: 38s - loss: 0.2243 - categorical_accuracy: 0.9320
41856/60000 [===================>..........] - ETA: 38s - loss: 0.2242 - categorical_accuracy: 0.9321
41888/60000 [===================>..........] - ETA: 38s - loss: 0.2240 - categorical_accuracy: 0.9321
41920/60000 [===================>..........] - ETA: 38s - loss: 0.2239 - categorical_accuracy: 0.9322
41952/60000 [===================>..........] - ETA: 38s - loss: 0.2237 - categorical_accuracy: 0.9322
41984/60000 [===================>..........] - ETA: 38s - loss: 0.2235 - categorical_accuracy: 0.9323
42016/60000 [====================>.........] - ETA: 37s - loss: 0.2236 - categorical_accuracy: 0.9323
42048/60000 [====================>.........] - ETA: 37s - loss: 0.2234 - categorical_accuracy: 0.9323
42080/60000 [====================>.........] - ETA: 37s - loss: 0.2234 - categorical_accuracy: 0.9323
42112/60000 [====================>.........] - ETA: 37s - loss: 0.2233 - categorical_accuracy: 0.9323
42144/60000 [====================>.........] - ETA: 37s - loss: 0.2231 - categorical_accuracy: 0.9324
42176/60000 [====================>.........] - ETA: 37s - loss: 0.2230 - categorical_accuracy: 0.9324
42208/60000 [====================>.........] - ETA: 37s - loss: 0.2229 - categorical_accuracy: 0.9324
42240/60000 [====================>.........] - ETA: 37s - loss: 0.2228 - categorical_accuracy: 0.9325
42272/60000 [====================>.........] - ETA: 37s - loss: 0.2227 - categorical_accuracy: 0.9325
42304/60000 [====================>.........] - ETA: 37s - loss: 0.2227 - categorical_accuracy: 0.9325
42336/60000 [====================>.........] - ETA: 37s - loss: 0.2225 - categorical_accuracy: 0.9325
42368/60000 [====================>.........] - ETA: 37s - loss: 0.2225 - categorical_accuracy: 0.9325
42400/60000 [====================>.........] - ETA: 37s - loss: 0.2223 - categorical_accuracy: 0.9326
42432/60000 [====================>.........] - ETA: 37s - loss: 0.2223 - categorical_accuracy: 0.9326
42464/60000 [====================>.........] - ETA: 36s - loss: 0.2223 - categorical_accuracy: 0.9326
42496/60000 [====================>.........] - ETA: 36s - loss: 0.2221 - categorical_accuracy: 0.9327
42528/60000 [====================>.........] - ETA: 36s - loss: 0.2222 - categorical_accuracy: 0.9327
42560/60000 [====================>.........] - ETA: 36s - loss: 0.2220 - categorical_accuracy: 0.9328
42592/60000 [====================>.........] - ETA: 36s - loss: 0.2221 - categorical_accuracy: 0.9327
42624/60000 [====================>.........] - ETA: 36s - loss: 0.2220 - categorical_accuracy: 0.9327
42656/60000 [====================>.........] - ETA: 36s - loss: 0.2220 - categorical_accuracy: 0.9327
42688/60000 [====================>.........] - ETA: 36s - loss: 0.2218 - categorical_accuracy: 0.9328
42720/60000 [====================>.........] - ETA: 36s - loss: 0.2217 - categorical_accuracy: 0.9328
42752/60000 [====================>.........] - ETA: 36s - loss: 0.2216 - categorical_accuracy: 0.9329
42784/60000 [====================>.........] - ETA: 36s - loss: 0.2216 - categorical_accuracy: 0.9328
42816/60000 [====================>.........] - ETA: 36s - loss: 0.2215 - categorical_accuracy: 0.9329
42848/60000 [====================>.........] - ETA: 36s - loss: 0.2213 - categorical_accuracy: 0.9329
42880/60000 [====================>.........] - ETA: 36s - loss: 0.2212 - categorical_accuracy: 0.9330
42912/60000 [====================>.........] - ETA: 36s - loss: 0.2211 - categorical_accuracy: 0.9330
42944/60000 [====================>.........] - ETA: 35s - loss: 0.2209 - categorical_accuracy: 0.9331
42976/60000 [====================>.........] - ETA: 35s - loss: 0.2208 - categorical_accuracy: 0.9331
43008/60000 [====================>.........] - ETA: 35s - loss: 0.2208 - categorical_accuracy: 0.9332
43040/60000 [====================>.........] - ETA: 35s - loss: 0.2208 - categorical_accuracy: 0.9332
43072/60000 [====================>.........] - ETA: 35s - loss: 0.2207 - categorical_accuracy: 0.9332
43104/60000 [====================>.........] - ETA: 35s - loss: 0.2206 - categorical_accuracy: 0.9332
43136/60000 [====================>.........] - ETA: 35s - loss: 0.2206 - categorical_accuracy: 0.9332
43168/60000 [====================>.........] - ETA: 35s - loss: 0.2205 - categorical_accuracy: 0.9332
43200/60000 [====================>.........] - ETA: 35s - loss: 0.2204 - categorical_accuracy: 0.9332
43232/60000 [====================>.........] - ETA: 35s - loss: 0.2204 - categorical_accuracy: 0.9332
43264/60000 [====================>.........] - ETA: 35s - loss: 0.2203 - categorical_accuracy: 0.9333
43296/60000 [====================>.........] - ETA: 35s - loss: 0.2202 - categorical_accuracy: 0.9333
43328/60000 [====================>.........] - ETA: 35s - loss: 0.2203 - categorical_accuracy: 0.9333
43360/60000 [====================>.........] - ETA: 35s - loss: 0.2202 - categorical_accuracy: 0.9333
43392/60000 [====================>.........] - ETA: 35s - loss: 0.2201 - categorical_accuracy: 0.9333
43424/60000 [====================>.........] - ETA: 34s - loss: 0.2201 - categorical_accuracy: 0.9334
43456/60000 [====================>.........] - ETA: 34s - loss: 0.2199 - categorical_accuracy: 0.9334
43488/60000 [====================>.........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9335
43520/60000 [====================>.........] - ETA: 34s - loss: 0.2197 - categorical_accuracy: 0.9335
43552/60000 [====================>.........] - ETA: 34s - loss: 0.2195 - categorical_accuracy: 0.9335
43584/60000 [====================>.........] - ETA: 34s - loss: 0.2195 - categorical_accuracy: 0.9336
43616/60000 [====================>.........] - ETA: 34s - loss: 0.2193 - categorical_accuracy: 0.9336
43648/60000 [====================>.........] - ETA: 34s - loss: 0.2194 - categorical_accuracy: 0.9336
43680/60000 [====================>.........] - ETA: 34s - loss: 0.2193 - categorical_accuracy: 0.9336
43712/60000 [====================>.........] - ETA: 34s - loss: 0.2192 - categorical_accuracy: 0.9337
43744/60000 [====================>.........] - ETA: 34s - loss: 0.2190 - categorical_accuracy: 0.9337
43776/60000 [====================>.........] - ETA: 34s - loss: 0.2189 - categorical_accuracy: 0.9337
43808/60000 [====================>.........] - ETA: 34s - loss: 0.2188 - categorical_accuracy: 0.9338
43840/60000 [====================>.........] - ETA: 34s - loss: 0.2187 - categorical_accuracy: 0.9338
43872/60000 [====================>.........] - ETA: 33s - loss: 0.2187 - categorical_accuracy: 0.9338
43904/60000 [====================>.........] - ETA: 33s - loss: 0.2185 - categorical_accuracy: 0.9338
43936/60000 [====================>.........] - ETA: 33s - loss: 0.2185 - categorical_accuracy: 0.9338
43968/60000 [====================>.........] - ETA: 33s - loss: 0.2184 - categorical_accuracy: 0.9339
44000/60000 [=====================>........] - ETA: 33s - loss: 0.2182 - categorical_accuracy: 0.9339
44032/60000 [=====================>........] - ETA: 33s - loss: 0.2181 - categorical_accuracy: 0.9340
44064/60000 [=====================>........] - ETA: 33s - loss: 0.2180 - categorical_accuracy: 0.9340
44096/60000 [=====================>........] - ETA: 33s - loss: 0.2179 - categorical_accuracy: 0.9341
44128/60000 [=====================>........] - ETA: 33s - loss: 0.2180 - categorical_accuracy: 0.9341
44160/60000 [=====================>........] - ETA: 33s - loss: 0.2178 - categorical_accuracy: 0.9341
44192/60000 [=====================>........] - ETA: 33s - loss: 0.2178 - categorical_accuracy: 0.9342
44224/60000 [=====================>........] - ETA: 33s - loss: 0.2176 - categorical_accuracy: 0.9342
44256/60000 [=====================>........] - ETA: 33s - loss: 0.2175 - categorical_accuracy: 0.9342
44288/60000 [=====================>........] - ETA: 33s - loss: 0.2175 - categorical_accuracy: 0.9342
44320/60000 [=====================>........] - ETA: 33s - loss: 0.2174 - categorical_accuracy: 0.9343
44352/60000 [=====================>........] - ETA: 32s - loss: 0.2172 - categorical_accuracy: 0.9343
44384/60000 [=====================>........] - ETA: 32s - loss: 0.2171 - categorical_accuracy: 0.9344
44416/60000 [=====================>........] - ETA: 32s - loss: 0.2170 - categorical_accuracy: 0.9344
44448/60000 [=====================>........] - ETA: 32s - loss: 0.2168 - categorical_accuracy: 0.9344
44480/60000 [=====================>........] - ETA: 32s - loss: 0.2167 - categorical_accuracy: 0.9345
44512/60000 [=====================>........] - ETA: 32s - loss: 0.2166 - categorical_accuracy: 0.9345
44544/60000 [=====================>........] - ETA: 32s - loss: 0.2166 - categorical_accuracy: 0.9345
44576/60000 [=====================>........] - ETA: 32s - loss: 0.2166 - categorical_accuracy: 0.9345
44608/60000 [=====================>........] - ETA: 32s - loss: 0.2164 - categorical_accuracy: 0.9346
44640/60000 [=====================>........] - ETA: 32s - loss: 0.2163 - categorical_accuracy: 0.9346
44672/60000 [=====================>........] - ETA: 32s - loss: 0.2162 - categorical_accuracy: 0.9346
44704/60000 [=====================>........] - ETA: 32s - loss: 0.2162 - categorical_accuracy: 0.9346
44736/60000 [=====================>........] - ETA: 32s - loss: 0.2160 - categorical_accuracy: 0.9347
44768/60000 [=====================>........] - ETA: 32s - loss: 0.2160 - categorical_accuracy: 0.9347
44800/60000 [=====================>........] - ETA: 31s - loss: 0.2158 - categorical_accuracy: 0.9347
44832/60000 [=====================>........] - ETA: 31s - loss: 0.2157 - categorical_accuracy: 0.9348
44864/60000 [=====================>........] - ETA: 31s - loss: 0.2156 - categorical_accuracy: 0.9348
44896/60000 [=====================>........] - ETA: 31s - loss: 0.2155 - categorical_accuracy: 0.9348
44928/60000 [=====================>........] - ETA: 31s - loss: 0.2153 - categorical_accuracy: 0.9349
44960/60000 [=====================>........] - ETA: 31s - loss: 0.2152 - categorical_accuracy: 0.9349
44992/60000 [=====================>........] - ETA: 31s - loss: 0.2151 - categorical_accuracy: 0.9349
45024/60000 [=====================>........] - ETA: 31s - loss: 0.2152 - categorical_accuracy: 0.9349
45056/60000 [=====================>........] - ETA: 31s - loss: 0.2152 - categorical_accuracy: 0.9349
45088/60000 [=====================>........] - ETA: 31s - loss: 0.2152 - categorical_accuracy: 0.9349
45120/60000 [=====================>........] - ETA: 31s - loss: 0.2151 - categorical_accuracy: 0.9350
45152/60000 [=====================>........] - ETA: 31s - loss: 0.2151 - categorical_accuracy: 0.9350
45184/60000 [=====================>........] - ETA: 31s - loss: 0.2152 - categorical_accuracy: 0.9349
45216/60000 [=====================>........] - ETA: 31s - loss: 0.2152 - categorical_accuracy: 0.9349
45248/60000 [=====================>........] - ETA: 31s - loss: 0.2151 - categorical_accuracy: 0.9349
45280/60000 [=====================>........] - ETA: 30s - loss: 0.2150 - categorical_accuracy: 0.9350
45312/60000 [=====================>........] - ETA: 30s - loss: 0.2149 - categorical_accuracy: 0.9350
45344/60000 [=====================>........] - ETA: 30s - loss: 0.2148 - categorical_accuracy: 0.9351
45376/60000 [=====================>........] - ETA: 30s - loss: 0.2146 - categorical_accuracy: 0.9351
45408/60000 [=====================>........] - ETA: 30s - loss: 0.2145 - categorical_accuracy: 0.9352
45440/60000 [=====================>........] - ETA: 30s - loss: 0.2145 - categorical_accuracy: 0.9352
45472/60000 [=====================>........] - ETA: 30s - loss: 0.2143 - categorical_accuracy: 0.9352
45504/60000 [=====================>........] - ETA: 30s - loss: 0.2142 - categorical_accuracy: 0.9353
45536/60000 [=====================>........] - ETA: 30s - loss: 0.2142 - categorical_accuracy: 0.9353
45568/60000 [=====================>........] - ETA: 30s - loss: 0.2142 - categorical_accuracy: 0.9353
45600/60000 [=====================>........] - ETA: 30s - loss: 0.2141 - categorical_accuracy: 0.9353
45632/60000 [=====================>........] - ETA: 30s - loss: 0.2140 - categorical_accuracy: 0.9353
45664/60000 [=====================>........] - ETA: 30s - loss: 0.2141 - categorical_accuracy: 0.9354
45696/60000 [=====================>........] - ETA: 30s - loss: 0.2140 - categorical_accuracy: 0.9354
45728/60000 [=====================>........] - ETA: 29s - loss: 0.2138 - categorical_accuracy: 0.9354
45760/60000 [=====================>........] - ETA: 29s - loss: 0.2137 - categorical_accuracy: 0.9355
45792/60000 [=====================>........] - ETA: 29s - loss: 0.2136 - categorical_accuracy: 0.9355
45824/60000 [=====================>........] - ETA: 29s - loss: 0.2135 - categorical_accuracy: 0.9356
45856/60000 [=====================>........] - ETA: 29s - loss: 0.2134 - categorical_accuracy: 0.9356
45888/60000 [=====================>........] - ETA: 29s - loss: 0.2133 - categorical_accuracy: 0.9356
45920/60000 [=====================>........] - ETA: 29s - loss: 0.2132 - categorical_accuracy: 0.9356
45952/60000 [=====================>........] - ETA: 29s - loss: 0.2131 - categorical_accuracy: 0.9357
45984/60000 [=====================>........] - ETA: 29s - loss: 0.2130 - categorical_accuracy: 0.9357
46016/60000 [======================>.......] - ETA: 29s - loss: 0.2129 - categorical_accuracy: 0.9357
46048/60000 [======================>.......] - ETA: 29s - loss: 0.2127 - categorical_accuracy: 0.9358
46080/60000 [======================>.......] - ETA: 29s - loss: 0.2128 - categorical_accuracy: 0.9358
46112/60000 [======================>.......] - ETA: 29s - loss: 0.2128 - categorical_accuracy: 0.9358
46144/60000 [======================>.......] - ETA: 29s - loss: 0.2127 - categorical_accuracy: 0.9358
46176/60000 [======================>.......] - ETA: 29s - loss: 0.2126 - categorical_accuracy: 0.9358
46208/60000 [======================>.......] - ETA: 28s - loss: 0.2125 - categorical_accuracy: 0.9359
46240/60000 [======================>.......] - ETA: 28s - loss: 0.2124 - categorical_accuracy: 0.9359
46272/60000 [======================>.......] - ETA: 28s - loss: 0.2123 - categorical_accuracy: 0.9359
46304/60000 [======================>.......] - ETA: 28s - loss: 0.2122 - categorical_accuracy: 0.9359
46336/60000 [======================>.......] - ETA: 28s - loss: 0.2121 - categorical_accuracy: 0.9360
46368/60000 [======================>.......] - ETA: 28s - loss: 0.2120 - categorical_accuracy: 0.9360
46400/60000 [======================>.......] - ETA: 28s - loss: 0.2120 - categorical_accuracy: 0.9360
46432/60000 [======================>.......] - ETA: 28s - loss: 0.2118 - categorical_accuracy: 0.9360
46464/60000 [======================>.......] - ETA: 28s - loss: 0.2118 - categorical_accuracy: 0.9361
46496/60000 [======================>.......] - ETA: 28s - loss: 0.2117 - categorical_accuracy: 0.9361
46528/60000 [======================>.......] - ETA: 28s - loss: 0.2116 - categorical_accuracy: 0.9361
46560/60000 [======================>.......] - ETA: 28s - loss: 0.2115 - categorical_accuracy: 0.9362
46592/60000 [======================>.......] - ETA: 28s - loss: 0.2114 - categorical_accuracy: 0.9362
46624/60000 [======================>.......] - ETA: 28s - loss: 0.2114 - categorical_accuracy: 0.9362
46656/60000 [======================>.......] - ETA: 28s - loss: 0.2113 - categorical_accuracy: 0.9362
46688/60000 [======================>.......] - ETA: 27s - loss: 0.2113 - categorical_accuracy: 0.9362
46720/60000 [======================>.......] - ETA: 27s - loss: 0.2112 - categorical_accuracy: 0.9363
46752/60000 [======================>.......] - ETA: 27s - loss: 0.2113 - categorical_accuracy: 0.9362
46784/60000 [======================>.......] - ETA: 27s - loss: 0.2112 - categorical_accuracy: 0.9363
46816/60000 [======================>.......] - ETA: 27s - loss: 0.2111 - categorical_accuracy: 0.9363
46848/60000 [======================>.......] - ETA: 27s - loss: 0.2110 - categorical_accuracy: 0.9363
46880/60000 [======================>.......] - ETA: 27s - loss: 0.2109 - categorical_accuracy: 0.9364
46912/60000 [======================>.......] - ETA: 27s - loss: 0.2108 - categorical_accuracy: 0.9364
46944/60000 [======================>.......] - ETA: 27s - loss: 0.2107 - categorical_accuracy: 0.9364
46976/60000 [======================>.......] - ETA: 27s - loss: 0.2106 - categorical_accuracy: 0.9365
47008/60000 [======================>.......] - ETA: 27s - loss: 0.2105 - categorical_accuracy: 0.9365
47040/60000 [======================>.......] - ETA: 27s - loss: 0.2103 - categorical_accuracy: 0.9365
47072/60000 [======================>.......] - ETA: 27s - loss: 0.2102 - categorical_accuracy: 0.9366
47104/60000 [======================>.......] - ETA: 27s - loss: 0.2102 - categorical_accuracy: 0.9366
47136/60000 [======================>.......] - ETA: 27s - loss: 0.2103 - categorical_accuracy: 0.9365
47168/60000 [======================>.......] - ETA: 26s - loss: 0.2101 - categorical_accuracy: 0.9366
47200/60000 [======================>.......] - ETA: 26s - loss: 0.2101 - categorical_accuracy: 0.9366
47232/60000 [======================>.......] - ETA: 26s - loss: 0.2101 - categorical_accuracy: 0.9366
47264/60000 [======================>.......] - ETA: 26s - loss: 0.2100 - categorical_accuracy: 0.9367
47296/60000 [======================>.......] - ETA: 26s - loss: 0.2099 - categorical_accuracy: 0.9367
47328/60000 [======================>.......] - ETA: 26s - loss: 0.2098 - categorical_accuracy: 0.9367
47360/60000 [======================>.......] - ETA: 26s - loss: 0.2097 - categorical_accuracy: 0.9368
47392/60000 [======================>.......] - ETA: 26s - loss: 0.2096 - categorical_accuracy: 0.9368
47424/60000 [======================>.......] - ETA: 26s - loss: 0.2095 - categorical_accuracy: 0.9368
47456/60000 [======================>.......] - ETA: 26s - loss: 0.2095 - categorical_accuracy: 0.9368
47488/60000 [======================>.......] - ETA: 26s - loss: 0.2093 - categorical_accuracy: 0.9368
47520/60000 [======================>.......] - ETA: 26s - loss: 0.2094 - categorical_accuracy: 0.9368
47552/60000 [======================>.......] - ETA: 26s - loss: 0.2093 - categorical_accuracy: 0.9368
47584/60000 [======================>.......] - ETA: 26s - loss: 0.2093 - categorical_accuracy: 0.9369
47616/60000 [======================>.......] - ETA: 25s - loss: 0.2092 - categorical_accuracy: 0.9369
47648/60000 [======================>.......] - ETA: 25s - loss: 0.2091 - categorical_accuracy: 0.9369
47680/60000 [======================>.......] - ETA: 25s - loss: 0.2091 - categorical_accuracy: 0.9369
47712/60000 [======================>.......] - ETA: 25s - loss: 0.2092 - categorical_accuracy: 0.9369
47744/60000 [======================>.......] - ETA: 25s - loss: 0.2091 - categorical_accuracy: 0.9369
47776/60000 [======================>.......] - ETA: 25s - loss: 0.2090 - categorical_accuracy: 0.9369
47808/60000 [======================>.......] - ETA: 25s - loss: 0.2089 - categorical_accuracy: 0.9370
47840/60000 [======================>.......] - ETA: 25s - loss: 0.2088 - categorical_accuracy: 0.9370
47872/60000 [======================>.......] - ETA: 25s - loss: 0.2087 - categorical_accuracy: 0.9370
47904/60000 [======================>.......] - ETA: 25s - loss: 0.2086 - categorical_accuracy: 0.9370
47936/60000 [======================>.......] - ETA: 25s - loss: 0.2085 - categorical_accuracy: 0.9371
47968/60000 [======================>.......] - ETA: 25s - loss: 0.2084 - categorical_accuracy: 0.9371
48000/60000 [=======================>......] - ETA: 25s - loss: 0.2083 - categorical_accuracy: 0.9371
48032/60000 [=======================>......] - ETA: 25s - loss: 0.2082 - categorical_accuracy: 0.9372
48064/60000 [=======================>......] - ETA: 25s - loss: 0.2081 - categorical_accuracy: 0.9372
48096/60000 [=======================>......] - ETA: 24s - loss: 0.2081 - categorical_accuracy: 0.9372
48128/60000 [=======================>......] - ETA: 24s - loss: 0.2080 - categorical_accuracy: 0.9372
48160/60000 [=======================>......] - ETA: 24s - loss: 0.2079 - categorical_accuracy: 0.9372
48192/60000 [=======================>......] - ETA: 24s - loss: 0.2078 - categorical_accuracy: 0.9373
48224/60000 [=======================>......] - ETA: 24s - loss: 0.2078 - categorical_accuracy: 0.9373
48256/60000 [=======================>......] - ETA: 24s - loss: 0.2077 - categorical_accuracy: 0.9373
48288/60000 [=======================>......] - ETA: 24s - loss: 0.2077 - categorical_accuracy: 0.9373
48320/60000 [=======================>......] - ETA: 24s - loss: 0.2077 - categorical_accuracy: 0.9374
48352/60000 [=======================>......] - ETA: 24s - loss: 0.2076 - categorical_accuracy: 0.9374
48384/60000 [=======================>......] - ETA: 24s - loss: 0.2075 - categorical_accuracy: 0.9374
48416/60000 [=======================>......] - ETA: 24s - loss: 0.2075 - categorical_accuracy: 0.9374
48448/60000 [=======================>......] - ETA: 24s - loss: 0.2073 - categorical_accuracy: 0.9374
48480/60000 [=======================>......] - ETA: 24s - loss: 0.2073 - categorical_accuracy: 0.9375
48512/60000 [=======================>......] - ETA: 24s - loss: 0.2072 - categorical_accuracy: 0.9375
48544/60000 [=======================>......] - ETA: 24s - loss: 0.2071 - categorical_accuracy: 0.9375
48576/60000 [=======================>......] - ETA: 23s - loss: 0.2070 - categorical_accuracy: 0.9375
48608/60000 [=======================>......] - ETA: 23s - loss: 0.2069 - categorical_accuracy: 0.9376
48640/60000 [=======================>......] - ETA: 23s - loss: 0.2067 - categorical_accuracy: 0.9376
48672/60000 [=======================>......] - ETA: 23s - loss: 0.2066 - categorical_accuracy: 0.9377
48704/60000 [=======================>......] - ETA: 23s - loss: 0.2066 - categorical_accuracy: 0.9377
48736/60000 [=======================>......] - ETA: 23s - loss: 0.2065 - categorical_accuracy: 0.9377
48768/60000 [=======================>......] - ETA: 23s - loss: 0.2066 - categorical_accuracy: 0.9377
48800/60000 [=======================>......] - ETA: 23s - loss: 0.2069 - categorical_accuracy: 0.9377
48832/60000 [=======================>......] - ETA: 23s - loss: 0.2067 - categorical_accuracy: 0.9377
48864/60000 [=======================>......] - ETA: 23s - loss: 0.2068 - categorical_accuracy: 0.9377
48896/60000 [=======================>......] - ETA: 23s - loss: 0.2067 - categorical_accuracy: 0.9378
48928/60000 [=======================>......] - ETA: 23s - loss: 0.2069 - categorical_accuracy: 0.9377
48960/60000 [=======================>......] - ETA: 23s - loss: 0.2068 - categorical_accuracy: 0.9378
48992/60000 [=======================>......] - ETA: 23s - loss: 0.2070 - categorical_accuracy: 0.9378
49024/60000 [=======================>......] - ETA: 22s - loss: 0.2069 - categorical_accuracy: 0.9378
49056/60000 [=======================>......] - ETA: 22s - loss: 0.2070 - categorical_accuracy: 0.9378
49088/60000 [=======================>......] - ETA: 22s - loss: 0.2069 - categorical_accuracy: 0.9378
49120/60000 [=======================>......] - ETA: 22s - loss: 0.2069 - categorical_accuracy: 0.9378
49152/60000 [=======================>......] - ETA: 22s - loss: 0.2068 - categorical_accuracy: 0.9378
49184/60000 [=======================>......] - ETA: 22s - loss: 0.2068 - categorical_accuracy: 0.9379
49216/60000 [=======================>......] - ETA: 22s - loss: 0.2067 - categorical_accuracy: 0.9379
49248/60000 [=======================>......] - ETA: 22s - loss: 0.2066 - categorical_accuracy: 0.9379
49280/60000 [=======================>......] - ETA: 22s - loss: 0.2067 - categorical_accuracy: 0.9379
49312/60000 [=======================>......] - ETA: 22s - loss: 0.2066 - categorical_accuracy: 0.9379
49344/60000 [=======================>......] - ETA: 22s - loss: 0.2066 - categorical_accuracy: 0.9379
49376/60000 [=======================>......] - ETA: 22s - loss: 0.2065 - categorical_accuracy: 0.9379
49408/60000 [=======================>......] - ETA: 22s - loss: 0.2065 - categorical_accuracy: 0.9379
49440/60000 [=======================>......] - ETA: 22s - loss: 0.2064 - categorical_accuracy: 0.9380
49472/60000 [=======================>......] - ETA: 22s - loss: 0.2063 - categorical_accuracy: 0.9380
49504/60000 [=======================>......] - ETA: 21s - loss: 0.2062 - categorical_accuracy: 0.9380
49536/60000 [=======================>......] - ETA: 21s - loss: 0.2061 - categorical_accuracy: 0.9381
49568/60000 [=======================>......] - ETA: 21s - loss: 0.2061 - categorical_accuracy: 0.9381
49600/60000 [=======================>......] - ETA: 21s - loss: 0.2060 - categorical_accuracy: 0.9381
49632/60000 [=======================>......] - ETA: 21s - loss: 0.2061 - categorical_accuracy: 0.9381
49664/60000 [=======================>......] - ETA: 21s - loss: 0.2060 - categorical_accuracy: 0.9381
49696/60000 [=======================>......] - ETA: 21s - loss: 0.2059 - categorical_accuracy: 0.9381
49728/60000 [=======================>......] - ETA: 21s - loss: 0.2057 - categorical_accuracy: 0.9382
49760/60000 [=======================>......] - ETA: 21s - loss: 0.2057 - categorical_accuracy: 0.9382
49792/60000 [=======================>......] - ETA: 21s - loss: 0.2057 - categorical_accuracy: 0.9382
49824/60000 [=======================>......] - ETA: 21s - loss: 0.2056 - categorical_accuracy: 0.9382
49856/60000 [=======================>......] - ETA: 21s - loss: 0.2054 - categorical_accuracy: 0.9383
49888/60000 [=======================>......] - ETA: 21s - loss: 0.2053 - categorical_accuracy: 0.9383
49920/60000 [=======================>......] - ETA: 21s - loss: 0.2053 - categorical_accuracy: 0.9383
49952/60000 [=======================>......] - ETA: 21s - loss: 0.2052 - categorical_accuracy: 0.9383
49984/60000 [=======================>......] - ETA: 20s - loss: 0.2051 - categorical_accuracy: 0.9383
50016/60000 [========================>.....] - ETA: 20s - loss: 0.2052 - categorical_accuracy: 0.9383
50048/60000 [========================>.....] - ETA: 20s - loss: 0.2050 - categorical_accuracy: 0.9383
50080/60000 [========================>.....] - ETA: 20s - loss: 0.2050 - categorical_accuracy: 0.9384
50112/60000 [========================>.....] - ETA: 20s - loss: 0.2049 - categorical_accuracy: 0.9384
50144/60000 [========================>.....] - ETA: 20s - loss: 0.2048 - categorical_accuracy: 0.9384
50176/60000 [========================>.....] - ETA: 20s - loss: 0.2047 - categorical_accuracy: 0.9384
50208/60000 [========================>.....] - ETA: 20s - loss: 0.2046 - categorical_accuracy: 0.9385
50240/60000 [========================>.....] - ETA: 20s - loss: 0.2047 - categorical_accuracy: 0.9385
50272/60000 [========================>.....] - ETA: 20s - loss: 0.2046 - categorical_accuracy: 0.9385
50304/60000 [========================>.....] - ETA: 20s - loss: 0.2045 - categorical_accuracy: 0.9385
50336/60000 [========================>.....] - ETA: 20s - loss: 0.2044 - categorical_accuracy: 0.9386
50368/60000 [========================>.....] - ETA: 20s - loss: 0.2044 - categorical_accuracy: 0.9386
50400/60000 [========================>.....] - ETA: 20s - loss: 0.2043 - categorical_accuracy: 0.9386
50432/60000 [========================>.....] - ETA: 20s - loss: 0.2042 - categorical_accuracy: 0.9386
50464/60000 [========================>.....] - ETA: 19s - loss: 0.2041 - categorical_accuracy: 0.9386
50496/60000 [========================>.....] - ETA: 19s - loss: 0.2040 - categorical_accuracy: 0.9386
50528/60000 [========================>.....] - ETA: 19s - loss: 0.2039 - categorical_accuracy: 0.9387
50560/60000 [========================>.....] - ETA: 19s - loss: 0.2040 - categorical_accuracy: 0.9387
50592/60000 [========================>.....] - ETA: 19s - loss: 0.2039 - categorical_accuracy: 0.9387
50624/60000 [========================>.....] - ETA: 19s - loss: 0.2038 - categorical_accuracy: 0.9387
50656/60000 [========================>.....] - ETA: 19s - loss: 0.2039 - categorical_accuracy: 0.9387
50688/60000 [========================>.....] - ETA: 19s - loss: 0.2038 - categorical_accuracy: 0.9387
50720/60000 [========================>.....] - ETA: 19s - loss: 0.2037 - categorical_accuracy: 0.9387
50752/60000 [========================>.....] - ETA: 19s - loss: 0.2036 - categorical_accuracy: 0.9387
50784/60000 [========================>.....] - ETA: 19s - loss: 0.2035 - categorical_accuracy: 0.9387
50816/60000 [========================>.....] - ETA: 19s - loss: 0.2034 - categorical_accuracy: 0.9388
50848/60000 [========================>.....] - ETA: 19s - loss: 0.2036 - categorical_accuracy: 0.9388
50880/60000 [========================>.....] - ETA: 19s - loss: 0.2035 - categorical_accuracy: 0.9388
50912/60000 [========================>.....] - ETA: 19s - loss: 0.2034 - categorical_accuracy: 0.9389
50944/60000 [========================>.....] - ETA: 18s - loss: 0.2032 - categorical_accuracy: 0.9389
50976/60000 [========================>.....] - ETA: 18s - loss: 0.2032 - categorical_accuracy: 0.9389
51008/60000 [========================>.....] - ETA: 18s - loss: 0.2031 - categorical_accuracy: 0.9389
51040/60000 [========================>.....] - ETA: 18s - loss: 0.2031 - categorical_accuracy: 0.9389
51072/60000 [========================>.....] - ETA: 18s - loss: 0.2031 - categorical_accuracy: 0.9389
51104/60000 [========================>.....] - ETA: 18s - loss: 0.2030 - categorical_accuracy: 0.9389
51136/60000 [========================>.....] - ETA: 18s - loss: 0.2029 - categorical_accuracy: 0.9390
51168/60000 [========================>.....] - ETA: 18s - loss: 0.2028 - categorical_accuracy: 0.9390
51200/60000 [========================>.....] - ETA: 18s - loss: 0.2029 - categorical_accuracy: 0.9390
51232/60000 [========================>.....] - ETA: 18s - loss: 0.2030 - categorical_accuracy: 0.9390
51264/60000 [========================>.....] - ETA: 18s - loss: 0.2030 - categorical_accuracy: 0.9390
51296/60000 [========================>.....] - ETA: 18s - loss: 0.2029 - categorical_accuracy: 0.9390
51328/60000 [========================>.....] - ETA: 18s - loss: 0.2029 - categorical_accuracy: 0.9390
51360/60000 [========================>.....] - ETA: 18s - loss: 0.2028 - categorical_accuracy: 0.9390
51392/60000 [========================>.....] - ETA: 17s - loss: 0.2027 - categorical_accuracy: 0.9391
51424/60000 [========================>.....] - ETA: 17s - loss: 0.2026 - categorical_accuracy: 0.9391
51456/60000 [========================>.....] - ETA: 17s - loss: 0.2025 - categorical_accuracy: 0.9391
51488/60000 [========================>.....] - ETA: 17s - loss: 0.2024 - categorical_accuracy: 0.9392
51520/60000 [========================>.....] - ETA: 17s - loss: 0.2023 - categorical_accuracy: 0.9392
51552/60000 [========================>.....] - ETA: 17s - loss: 0.2022 - categorical_accuracy: 0.9392
51584/60000 [========================>.....] - ETA: 17s - loss: 0.2021 - categorical_accuracy: 0.9392
51616/60000 [========================>.....] - ETA: 17s - loss: 0.2021 - categorical_accuracy: 0.9393
51648/60000 [========================>.....] - ETA: 17s - loss: 0.2020 - categorical_accuracy: 0.9393
51680/60000 [========================>.....] - ETA: 17s - loss: 0.2019 - categorical_accuracy: 0.9393
51712/60000 [========================>.....] - ETA: 17s - loss: 0.2019 - categorical_accuracy: 0.9393
51744/60000 [========================>.....] - ETA: 17s - loss: 0.2018 - categorical_accuracy: 0.9393
51776/60000 [========================>.....] - ETA: 17s - loss: 0.2017 - categorical_accuracy: 0.9394
51808/60000 [========================>.....] - ETA: 17s - loss: 0.2017 - categorical_accuracy: 0.9394
51840/60000 [========================>.....] - ETA: 17s - loss: 0.2016 - categorical_accuracy: 0.9394
51872/60000 [========================>.....] - ETA: 16s - loss: 0.2015 - categorical_accuracy: 0.9394
51904/60000 [========================>.....] - ETA: 16s - loss: 0.2014 - categorical_accuracy: 0.9394
51936/60000 [========================>.....] - ETA: 16s - loss: 0.2013 - categorical_accuracy: 0.9395
51968/60000 [========================>.....] - ETA: 16s - loss: 0.2011 - categorical_accuracy: 0.9395
52000/60000 [=========================>....] - ETA: 16s - loss: 0.2010 - categorical_accuracy: 0.9396
52032/60000 [=========================>....] - ETA: 16s - loss: 0.2009 - categorical_accuracy: 0.9396
52064/60000 [=========================>....] - ETA: 16s - loss: 0.2008 - categorical_accuracy: 0.9396
52096/60000 [=========================>....] - ETA: 16s - loss: 0.2009 - categorical_accuracy: 0.9396
52128/60000 [=========================>....] - ETA: 16s - loss: 0.2008 - categorical_accuracy: 0.9396
52160/60000 [=========================>....] - ETA: 16s - loss: 0.2007 - categorical_accuracy: 0.9397
52192/60000 [=========================>....] - ETA: 16s - loss: 0.2008 - categorical_accuracy: 0.9397
52224/60000 [=========================>....] - ETA: 16s - loss: 0.2007 - categorical_accuracy: 0.9397
52256/60000 [=========================>....] - ETA: 16s - loss: 0.2006 - categorical_accuracy: 0.9397
52288/60000 [=========================>....] - ETA: 16s - loss: 0.2006 - categorical_accuracy: 0.9397
52320/60000 [=========================>....] - ETA: 16s - loss: 0.2005 - categorical_accuracy: 0.9398
52352/60000 [=========================>....] - ETA: 15s - loss: 0.2004 - categorical_accuracy: 0.9398
52384/60000 [=========================>....] - ETA: 15s - loss: 0.2004 - categorical_accuracy: 0.9398
52416/60000 [=========================>....] - ETA: 15s - loss: 0.2003 - categorical_accuracy: 0.9398
52448/60000 [=========================>....] - ETA: 15s - loss: 0.2002 - categorical_accuracy: 0.9398
52480/60000 [=========================>....] - ETA: 15s - loss: 0.2002 - categorical_accuracy: 0.9398
52512/60000 [=========================>....] - ETA: 15s - loss: 0.2001 - categorical_accuracy: 0.9398
52544/60000 [=========================>....] - ETA: 15s - loss: 0.2001 - categorical_accuracy: 0.9399
52576/60000 [=========================>....] - ETA: 15s - loss: 0.2000 - categorical_accuracy: 0.9399
52608/60000 [=========================>....] - ETA: 15s - loss: 0.2000 - categorical_accuracy: 0.9399
52640/60000 [=========================>....] - ETA: 15s - loss: 0.1999 - categorical_accuracy: 0.9399
52672/60000 [=========================>....] - ETA: 15s - loss: 0.1998 - categorical_accuracy: 0.9399
52704/60000 [=========================>....] - ETA: 15s - loss: 0.1997 - categorical_accuracy: 0.9400
52736/60000 [=========================>....] - ETA: 15s - loss: 0.1998 - categorical_accuracy: 0.9400
52768/60000 [=========================>....] - ETA: 15s - loss: 0.1998 - categorical_accuracy: 0.9400
52800/60000 [=========================>....] - ETA: 15s - loss: 0.1997 - categorical_accuracy: 0.9400
52832/60000 [=========================>....] - ETA: 14s - loss: 0.1996 - categorical_accuracy: 0.9400
52864/60000 [=========================>....] - ETA: 14s - loss: 0.1995 - categorical_accuracy: 0.9401
52896/60000 [=========================>....] - ETA: 14s - loss: 0.1994 - categorical_accuracy: 0.9401
52928/60000 [=========================>....] - ETA: 14s - loss: 0.1995 - categorical_accuracy: 0.9401
52960/60000 [=========================>....] - ETA: 14s - loss: 0.1995 - categorical_accuracy: 0.9401
52992/60000 [=========================>....] - ETA: 14s - loss: 0.1994 - categorical_accuracy: 0.9401
53024/60000 [=========================>....] - ETA: 14s - loss: 0.1993 - categorical_accuracy: 0.9401
53056/60000 [=========================>....] - ETA: 14s - loss: 0.1992 - categorical_accuracy: 0.9402
53088/60000 [=========================>....] - ETA: 14s - loss: 0.1992 - categorical_accuracy: 0.9402
53120/60000 [=========================>....] - ETA: 14s - loss: 0.1991 - categorical_accuracy: 0.9402
53152/60000 [=========================>....] - ETA: 14s - loss: 0.1990 - categorical_accuracy: 0.9402
53184/60000 [=========================>....] - ETA: 14s - loss: 0.1989 - categorical_accuracy: 0.9403
53216/60000 [=========================>....] - ETA: 14s - loss: 0.1990 - categorical_accuracy: 0.9403
53248/60000 [=========================>....] - ETA: 14s - loss: 0.1989 - categorical_accuracy: 0.9403
53280/60000 [=========================>....] - ETA: 14s - loss: 0.1988 - categorical_accuracy: 0.9403
53312/60000 [=========================>....] - ETA: 13s - loss: 0.1987 - categorical_accuracy: 0.9403
53344/60000 [=========================>....] - ETA: 13s - loss: 0.1987 - categorical_accuracy: 0.9403
53376/60000 [=========================>....] - ETA: 13s - loss: 0.1986 - categorical_accuracy: 0.9404
53408/60000 [=========================>....] - ETA: 13s - loss: 0.1986 - categorical_accuracy: 0.9404
53440/60000 [=========================>....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9404
53472/60000 [=========================>....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9404
53504/60000 [=========================>....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9404
53536/60000 [=========================>....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9404
53568/60000 [=========================>....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9404
53600/60000 [=========================>....] - ETA: 13s - loss: 0.1984 - categorical_accuracy: 0.9404
53632/60000 [=========================>....] - ETA: 13s - loss: 0.1983 - categorical_accuracy: 0.9404
53664/60000 [=========================>....] - ETA: 13s - loss: 0.1982 - categorical_accuracy: 0.9405
53696/60000 [=========================>....] - ETA: 13s - loss: 0.1981 - categorical_accuracy: 0.9405
53728/60000 [=========================>....] - ETA: 13s - loss: 0.1981 - categorical_accuracy: 0.9405
53760/60000 [=========================>....] - ETA: 13s - loss: 0.1981 - categorical_accuracy: 0.9405
53792/60000 [=========================>....] - ETA: 12s - loss: 0.1980 - categorical_accuracy: 0.9406
53824/60000 [=========================>....] - ETA: 12s - loss: 0.1981 - categorical_accuracy: 0.9405
53856/60000 [=========================>....] - ETA: 12s - loss: 0.1980 - categorical_accuracy: 0.9406
53888/60000 [=========================>....] - ETA: 12s - loss: 0.1979 - categorical_accuracy: 0.9406
53920/60000 [=========================>....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9406
53952/60000 [=========================>....] - ETA: 12s - loss: 0.1977 - categorical_accuracy: 0.9407
53984/60000 [=========================>....] - ETA: 12s - loss: 0.1976 - categorical_accuracy: 0.9407
54016/60000 [==========================>...] - ETA: 12s - loss: 0.1975 - categorical_accuracy: 0.9407
54048/60000 [==========================>...] - ETA: 12s - loss: 0.1975 - categorical_accuracy: 0.9407
54080/60000 [==========================>...] - ETA: 12s - loss: 0.1974 - categorical_accuracy: 0.9408
54112/60000 [==========================>...] - ETA: 12s - loss: 0.1973 - categorical_accuracy: 0.9408
54144/60000 [==========================>...] - ETA: 12s - loss: 0.1973 - categorical_accuracy: 0.9408
54176/60000 [==========================>...] - ETA: 12s - loss: 0.1972 - categorical_accuracy: 0.9408
54208/60000 [==========================>...] - ETA: 12s - loss: 0.1972 - categorical_accuracy: 0.9409
54240/60000 [==========================>...] - ETA: 11s - loss: 0.1972 - categorical_accuracy: 0.9409
54272/60000 [==========================>...] - ETA: 11s - loss: 0.1970 - categorical_accuracy: 0.9409
54304/60000 [==========================>...] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9409
54336/60000 [==========================>...] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9409
54368/60000 [==========================>...] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9409
54400/60000 [==========================>...] - ETA: 11s - loss: 0.1971 - categorical_accuracy: 0.9409
54432/60000 [==========================>...] - ETA: 11s - loss: 0.1970 - categorical_accuracy: 0.9410
54464/60000 [==========================>...] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9410
54496/60000 [==========================>...] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9410
54528/60000 [==========================>...] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9410
54560/60000 [==========================>...] - ETA: 11s - loss: 0.1966 - categorical_accuracy: 0.9411
54592/60000 [==========================>...] - ETA: 11s - loss: 0.1966 - categorical_accuracy: 0.9411
54624/60000 [==========================>...] - ETA: 11s - loss: 0.1965 - categorical_accuracy: 0.9411
54656/60000 [==========================>...] - ETA: 11s - loss: 0.1963 - categorical_accuracy: 0.9411
54688/60000 [==========================>...] - ETA: 11s - loss: 0.1964 - categorical_accuracy: 0.9411
54720/60000 [==========================>...] - ETA: 10s - loss: 0.1963 - categorical_accuracy: 0.9411
54752/60000 [==========================>...] - ETA: 10s - loss: 0.1962 - categorical_accuracy: 0.9412
54784/60000 [==========================>...] - ETA: 10s - loss: 0.1961 - categorical_accuracy: 0.9412
54816/60000 [==========================>...] - ETA: 10s - loss: 0.1961 - categorical_accuracy: 0.9412
54848/60000 [==========================>...] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9412
54880/60000 [==========================>...] - ETA: 10s - loss: 0.1960 - categorical_accuracy: 0.9412
54912/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9413
54944/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9413
54976/60000 [==========================>...] - ETA: 10s - loss: 0.1959 - categorical_accuracy: 0.9413
55008/60000 [==========================>...] - ETA: 10s - loss: 0.1958 - categorical_accuracy: 0.9413
55040/60000 [==========================>...] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9413
55072/60000 [==========================>...] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9413
55104/60000 [==========================>...] - ETA: 10s - loss: 0.1957 - categorical_accuracy: 0.9413
55136/60000 [==========================>...] - ETA: 10s - loss: 0.1956 - categorical_accuracy: 0.9414
55168/60000 [==========================>...] - ETA: 10s - loss: 0.1955 - categorical_accuracy: 0.9414
55200/60000 [==========================>...] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9414 
55232/60000 [==========================>...] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9414
55264/60000 [==========================>...] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9414
55296/60000 [==========================>...] - ETA: 9s - loss: 0.1953 - categorical_accuracy: 0.9415
55328/60000 [==========================>...] - ETA: 9s - loss: 0.1954 - categorical_accuracy: 0.9414
55360/60000 [==========================>...] - ETA: 9s - loss: 0.1952 - categorical_accuracy: 0.9415
55392/60000 [==========================>...] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9415
55424/60000 [==========================>...] - ETA: 9s - loss: 0.1951 - categorical_accuracy: 0.9415
55456/60000 [==========================>...] - ETA: 9s - loss: 0.1950 - categorical_accuracy: 0.9416
55488/60000 [==========================>...] - ETA: 9s - loss: 0.1950 - categorical_accuracy: 0.9416
55520/60000 [==========================>...] - ETA: 9s - loss: 0.1949 - categorical_accuracy: 0.9416
55552/60000 [==========================>...] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9416
55584/60000 [==========================>...] - ETA: 9s - loss: 0.1947 - categorical_accuracy: 0.9416
55616/60000 [==========================>...] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9416
55648/60000 [==========================>...] - ETA: 9s - loss: 0.1948 - categorical_accuracy: 0.9416
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1947 - categorical_accuracy: 0.9416
55712/60000 [==========================>...] - ETA: 8s - loss: 0.1946 - categorical_accuracy: 0.9417
55744/60000 [==========================>...] - ETA: 8s - loss: 0.1946 - categorical_accuracy: 0.9417
55776/60000 [==========================>...] - ETA: 8s - loss: 0.1945 - categorical_accuracy: 0.9417
55808/60000 [==========================>...] - ETA: 8s - loss: 0.1945 - categorical_accuracy: 0.9417
55840/60000 [==========================>...] - ETA: 8s - loss: 0.1944 - categorical_accuracy: 0.9417
55872/60000 [==========================>...] - ETA: 8s - loss: 0.1945 - categorical_accuracy: 0.9417
55904/60000 [==========================>...] - ETA: 8s - loss: 0.1944 - categorical_accuracy: 0.9417
55936/60000 [==========================>...] - ETA: 8s - loss: 0.1943 - categorical_accuracy: 0.9418
55968/60000 [==========================>...] - ETA: 8s - loss: 0.1943 - categorical_accuracy: 0.9418
56000/60000 [===========================>..] - ETA: 8s - loss: 0.1942 - categorical_accuracy: 0.9418
56032/60000 [===========================>..] - ETA: 8s - loss: 0.1941 - categorical_accuracy: 0.9418
56064/60000 [===========================>..] - ETA: 8s - loss: 0.1940 - categorical_accuracy: 0.9419
56096/60000 [===========================>..] - ETA: 8s - loss: 0.1939 - categorical_accuracy: 0.9419
56128/60000 [===========================>..] - ETA: 8s - loss: 0.1939 - categorical_accuracy: 0.9419
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1939 - categorical_accuracy: 0.9419
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1938 - categorical_accuracy: 0.9419
56224/60000 [===========================>..] - ETA: 7s - loss: 0.1937 - categorical_accuracy: 0.9419
56256/60000 [===========================>..] - ETA: 7s - loss: 0.1937 - categorical_accuracy: 0.9419
56288/60000 [===========================>..] - ETA: 7s - loss: 0.1936 - categorical_accuracy: 0.9420
56320/60000 [===========================>..] - ETA: 7s - loss: 0.1935 - categorical_accuracy: 0.9420
56352/60000 [===========================>..] - ETA: 7s - loss: 0.1936 - categorical_accuracy: 0.9420
56384/60000 [===========================>..] - ETA: 7s - loss: 0.1934 - categorical_accuracy: 0.9420
56416/60000 [===========================>..] - ETA: 7s - loss: 0.1934 - categorical_accuracy: 0.9421
56448/60000 [===========================>..] - ETA: 7s - loss: 0.1935 - categorical_accuracy: 0.9420
56480/60000 [===========================>..] - ETA: 7s - loss: 0.1935 - categorical_accuracy: 0.9421
56512/60000 [===========================>..] - ETA: 7s - loss: 0.1936 - categorical_accuracy: 0.9420
56544/60000 [===========================>..] - ETA: 7s - loss: 0.1935 - categorical_accuracy: 0.9421
56576/60000 [===========================>..] - ETA: 7s - loss: 0.1935 - categorical_accuracy: 0.9421
56608/60000 [===========================>..] - ETA: 7s - loss: 0.1934 - categorical_accuracy: 0.9421
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1934 - categorical_accuracy: 0.9421
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1933 - categorical_accuracy: 0.9421
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1932 - categorical_accuracy: 0.9422
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1931 - categorical_accuracy: 0.9422
56768/60000 [===========================>..] - ETA: 6s - loss: 0.1930 - categorical_accuracy: 0.9422
56800/60000 [===========================>..] - ETA: 6s - loss: 0.1929 - categorical_accuracy: 0.9423
56832/60000 [===========================>..] - ETA: 6s - loss: 0.1932 - categorical_accuracy: 0.9422
56864/60000 [===========================>..] - ETA: 6s - loss: 0.1932 - categorical_accuracy: 0.9422
56896/60000 [===========================>..] - ETA: 6s - loss: 0.1931 - categorical_accuracy: 0.9422
56928/60000 [===========================>..] - ETA: 6s - loss: 0.1931 - categorical_accuracy: 0.9422
56960/60000 [===========================>..] - ETA: 6s - loss: 0.1930 - categorical_accuracy: 0.9423
56992/60000 [===========================>..] - ETA: 6s - loss: 0.1929 - categorical_accuracy: 0.9423
57024/60000 [===========================>..] - ETA: 6s - loss: 0.1928 - categorical_accuracy: 0.9423
57056/60000 [===========================>..] - ETA: 6s - loss: 0.1927 - categorical_accuracy: 0.9424
57088/60000 [===========================>..] - ETA: 6s - loss: 0.1927 - categorical_accuracy: 0.9424
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1926 - categorical_accuracy: 0.9424
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1925 - categorical_accuracy: 0.9424
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1924 - categorical_accuracy: 0.9424
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1925 - categorical_accuracy: 0.9424
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1924 - categorical_accuracy: 0.9424
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1923 - categorical_accuracy: 0.9425
57312/60000 [===========================>..] - ETA: 5s - loss: 0.1922 - categorical_accuracy: 0.9425
57344/60000 [===========================>..] - ETA: 5s - loss: 0.1921 - categorical_accuracy: 0.9425
57376/60000 [===========================>..] - ETA: 5s - loss: 0.1921 - categorical_accuracy: 0.9425
57408/60000 [===========================>..] - ETA: 5s - loss: 0.1921 - categorical_accuracy: 0.9426
57440/60000 [===========================>..] - ETA: 5s - loss: 0.1920 - categorical_accuracy: 0.9425
57472/60000 [===========================>..] - ETA: 5s - loss: 0.1919 - categorical_accuracy: 0.9426
57504/60000 [===========================>..] - ETA: 5s - loss: 0.1919 - categorical_accuracy: 0.9426
57536/60000 [===========================>..] - ETA: 5s - loss: 0.1918 - categorical_accuracy: 0.9426
57568/60000 [===========================>..] - ETA: 5s - loss: 0.1918 - categorical_accuracy: 0.9426
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1918 - categorical_accuracy: 0.9426
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1917 - categorical_accuracy: 0.9426
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1917 - categorical_accuracy: 0.9426
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1916 - categorical_accuracy: 0.9426
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1916 - categorical_accuracy: 0.9427
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1915 - categorical_accuracy: 0.9427
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1915 - categorical_accuracy: 0.9427
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1914 - categorical_accuracy: 0.9427
57856/60000 [===========================>..] - ETA: 4s - loss: 0.1913 - categorical_accuracy: 0.9427
57888/60000 [===========================>..] - ETA: 4s - loss: 0.1913 - categorical_accuracy: 0.9428
57920/60000 [===========================>..] - ETA: 4s - loss: 0.1914 - categorical_accuracy: 0.9427
57952/60000 [===========================>..] - ETA: 4s - loss: 0.1913 - categorical_accuracy: 0.9427
57984/60000 [===========================>..] - ETA: 4s - loss: 0.1913 - categorical_accuracy: 0.9427
58016/60000 [============================>.] - ETA: 4s - loss: 0.1913 - categorical_accuracy: 0.9428
58048/60000 [============================>.] - ETA: 4s - loss: 0.1912 - categorical_accuracy: 0.9428
58080/60000 [============================>.] - ETA: 3s - loss: 0.1911 - categorical_accuracy: 0.9428
58112/60000 [============================>.] - ETA: 3s - loss: 0.1909 - categorical_accuracy: 0.9429
58144/60000 [============================>.] - ETA: 3s - loss: 0.1909 - categorical_accuracy: 0.9429
58176/60000 [============================>.] - ETA: 3s - loss: 0.1908 - categorical_accuracy: 0.9429
58208/60000 [============================>.] - ETA: 3s - loss: 0.1907 - categorical_accuracy: 0.9429
58240/60000 [============================>.] - ETA: 3s - loss: 0.1907 - categorical_accuracy: 0.9429
58272/60000 [============================>.] - ETA: 3s - loss: 0.1906 - categorical_accuracy: 0.9430
58304/60000 [============================>.] - ETA: 3s - loss: 0.1905 - categorical_accuracy: 0.9430
58336/60000 [============================>.] - ETA: 3s - loss: 0.1904 - categorical_accuracy: 0.9430
58368/60000 [============================>.] - ETA: 3s - loss: 0.1904 - categorical_accuracy: 0.9430
58400/60000 [============================>.] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9430
58432/60000 [============================>.] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9430
58464/60000 [============================>.] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9430
58496/60000 [============================>.] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9430
58528/60000 [============================>.] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9430
58560/60000 [============================>.] - ETA: 2s - loss: 0.1904 - categorical_accuracy: 0.9430
58592/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9431
58624/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9431
58656/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9431
58688/60000 [============================>.] - ETA: 2s - loss: 0.1902 - categorical_accuracy: 0.9431
58720/60000 [============================>.] - ETA: 2s - loss: 0.1901 - categorical_accuracy: 0.9431
58752/60000 [============================>.] - ETA: 2s - loss: 0.1901 - categorical_accuracy: 0.9432
58784/60000 [============================>.] - ETA: 2s - loss: 0.1900 - categorical_accuracy: 0.9432
58816/60000 [============================>.] - ETA: 2s - loss: 0.1899 - categorical_accuracy: 0.9432
58848/60000 [============================>.] - ETA: 2s - loss: 0.1898 - categorical_accuracy: 0.9432
58880/60000 [============================>.] - ETA: 2s - loss: 0.1897 - categorical_accuracy: 0.9433
58912/60000 [============================>.] - ETA: 2s - loss: 0.1897 - categorical_accuracy: 0.9433
58944/60000 [============================>.] - ETA: 2s - loss: 0.1897 - categorical_accuracy: 0.9433
58976/60000 [============================>.] - ETA: 2s - loss: 0.1896 - categorical_accuracy: 0.9433
59008/60000 [============================>.] - ETA: 2s - loss: 0.1895 - categorical_accuracy: 0.9433
59040/60000 [============================>.] - ETA: 1s - loss: 0.1895 - categorical_accuracy: 0.9433
59072/60000 [============================>.] - ETA: 1s - loss: 0.1894 - categorical_accuracy: 0.9434
59104/60000 [============================>.] - ETA: 1s - loss: 0.1893 - categorical_accuracy: 0.9434
59136/60000 [============================>.] - ETA: 1s - loss: 0.1892 - categorical_accuracy: 0.9434
59168/60000 [============================>.] - ETA: 1s - loss: 0.1891 - categorical_accuracy: 0.9434
59200/60000 [============================>.] - ETA: 1s - loss: 0.1890 - categorical_accuracy: 0.9434
59232/60000 [============================>.] - ETA: 1s - loss: 0.1890 - categorical_accuracy: 0.9435
59264/60000 [============================>.] - ETA: 1s - loss: 0.1889 - categorical_accuracy: 0.9435
59296/60000 [============================>.] - ETA: 1s - loss: 0.1889 - categorical_accuracy: 0.9435
59328/60000 [============================>.] - ETA: 1s - loss: 0.1889 - categorical_accuracy: 0.9435
59360/60000 [============================>.] - ETA: 1s - loss: 0.1888 - categorical_accuracy: 0.9435
59392/60000 [============================>.] - ETA: 1s - loss: 0.1887 - categorical_accuracy: 0.9435
59424/60000 [============================>.] - ETA: 1s - loss: 0.1887 - categorical_accuracy: 0.9435
59456/60000 [============================>.] - ETA: 1s - loss: 0.1887 - categorical_accuracy: 0.9435
59488/60000 [============================>.] - ETA: 1s - loss: 0.1887 - categorical_accuracy: 0.9435
59520/60000 [============================>.] - ETA: 0s - loss: 0.1886 - categorical_accuracy: 0.9435
59552/60000 [============================>.] - ETA: 0s - loss: 0.1885 - categorical_accuracy: 0.9436
59584/60000 [============================>.] - ETA: 0s - loss: 0.1885 - categorical_accuracy: 0.9436
59616/60000 [============================>.] - ETA: 0s - loss: 0.1884 - categorical_accuracy: 0.9436
59648/60000 [============================>.] - ETA: 0s - loss: 0.1883 - categorical_accuracy: 0.9437
59680/60000 [============================>.] - ETA: 0s - loss: 0.1882 - categorical_accuracy: 0.9437
59712/60000 [============================>.] - ETA: 0s - loss: 0.1881 - categorical_accuracy: 0.9437
59744/60000 [============================>.] - ETA: 0s - loss: 0.1881 - categorical_accuracy: 0.9437
59776/60000 [============================>.] - ETA: 0s - loss: 0.1880 - categorical_accuracy: 0.9437
59808/60000 [============================>.] - ETA: 0s - loss: 0.1880 - categorical_accuracy: 0.9438
59840/60000 [============================>.] - ETA: 0s - loss: 0.1879 - categorical_accuracy: 0.9438
59872/60000 [============================>.] - ETA: 0s - loss: 0.1878 - categorical_accuracy: 0.9438
59904/60000 [============================>.] - ETA: 0s - loss: 0.1877 - categorical_accuracy: 0.9438
59936/60000 [============================>.] - ETA: 0s - loss: 0.1876 - categorical_accuracy: 0.9439
59968/60000 [============================>.] - ETA: 0s - loss: 0.1876 - categorical_accuracy: 0.9439
60000/60000 [==============================] - 129s 2ms/step - loss: 0.1875 - categorical_accuracy: 0.9439 - val_loss: 0.0553 - val_categorical_accuracy: 0.9823

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 20s
  160/10000 [..............................] - ETA: 7s 
  288/10000 [..............................] - ETA: 5s
  448/10000 [>.............................] - ETA: 4s
  608/10000 [>.............................] - ETA: 4s
  768/10000 [=>............................] - ETA: 4s
  896/10000 [=>............................] - ETA: 4s
 1056/10000 [==>...........................] - ETA: 3s
 1184/10000 [==>...........................] - ETA: 3s
 1312/10000 [==>...........................] - ETA: 3s
 1440/10000 [===>..........................] - ETA: 3s
 1568/10000 [===>..........................] - ETA: 3s
 1696/10000 [====>.........................] - ETA: 3s
 1856/10000 [====>.........................] - ETA: 3s
 2016/10000 [=====>........................] - ETA: 3s
 2176/10000 [=====>........................] - ETA: 3s
 2336/10000 [======>.......................] - ETA: 3s
 2464/10000 [======>.......................] - ETA: 3s
 2624/10000 [======>.......................] - ETA: 3s
 2784/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3040/10000 [========>.....................] - ETA: 2s
 3200/10000 [========>.....................] - ETA: 2s
 3360/10000 [=========>....................] - ETA: 2s
 3520/10000 [=========>....................] - ETA: 2s
 3648/10000 [=========>....................] - ETA: 2s
 3808/10000 [==========>...................] - ETA: 2s
 3968/10000 [==========>...................] - ETA: 2s
 4128/10000 [===========>..................] - ETA: 2s
 4288/10000 [===========>..................] - ETA: 2s
 4448/10000 [============>.................] - ETA: 2s
 4576/10000 [============>.................] - ETA: 2s
 4736/10000 [=============>................] - ETA: 2s
 4896/10000 [=============>................] - ETA: 2s
 5024/10000 [==============>...............] - ETA: 2s
 5152/10000 [==============>...............] - ETA: 1s
 5280/10000 [==============>...............] - ETA: 1s
 5408/10000 [===============>..............] - ETA: 1s
 5536/10000 [===============>..............] - ETA: 1s
 5664/10000 [===============>..............] - ETA: 1s
 5792/10000 [================>.............] - ETA: 1s
 5920/10000 [================>.............] - ETA: 1s
 6080/10000 [=================>............] - ETA: 1s
 6240/10000 [=================>............] - ETA: 1s
 6400/10000 [==================>...........] - ETA: 1s
 6560/10000 [==================>...........] - ETA: 1s
 6720/10000 [===================>..........] - ETA: 1s
 6880/10000 [===================>..........] - ETA: 1s
 7040/10000 [====================>.........] - ETA: 1s
 7200/10000 [====================>.........] - ETA: 1s
 7328/10000 [====================>.........] - ETA: 1s
 7488/10000 [=====================>........] - ETA: 1s
 7648/10000 [=====================>........] - ETA: 0s
 7776/10000 [======================>.......] - ETA: 0s
 7936/10000 [======================>.......] - ETA: 0s
 8064/10000 [=======================>......] - ETA: 0s
 8192/10000 [=======================>......] - ETA: 0s
 8352/10000 [========================>.....] - ETA: 0s
 8512/10000 [========================>.....] - ETA: 0s
 8672/10000 [=========================>....] - ETA: 0s
 8832/10000 [=========================>....] - ETA: 0s
 8992/10000 [=========================>....] - ETA: 0s
 9152/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9376/10000 [===========================>..] - ETA: 0s
 9504/10000 [===========================>..] - ETA: 0s
 9632/10000 [===========================>..] - ETA: 0s
 9760/10000 [============================>.] - ETA: 0s
 9856/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 404us/step
[[5.32790789e-09 3.51208946e-08 4.46142622e-07 ... 9.99999166e-01
  4.93009500e-09 1.48875614e-07]
 [3.00984834e-06 5.28218879e-05 9.99924898e-01 ... 4.78792117e-08
  1.15467001e-05 3.72043063e-09]
 [5.03500885e-08 9.99933720e-01 2.71982822e-06 ... 7.86300916e-06
  3.84266787e-06 4.65336356e-08]
 ...
 [9.24323187e-11 1.06856595e-08 1.16531368e-10 ... 3.33419159e-08
  2.85608934e-07 2.56987136e-07]
 [3.60543254e-07 4.61434162e-08 1.24759969e-09 ... 2.00335339e-08
  5.91395050e-03 3.23226033e-08]
 [1.00532738e-07 9.46504102e-08 4.01810098e-07 ... 8.58269233e-10
  1.27543643e-07 5.92708937e-10]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05527686522648437, 'accuracy_test:': 0.9822999835014343}

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
[master c4b68cc] ml_store  && git pull --all
 1 file changed, 2043 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 48cbed0...c4b68cc master -> master (forced update)





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
{'loss': 0.4519955441355705, 'loss_history': []}

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
[master 09481c1] ml_store  && git pull --all
 1 file changed, 111 insertions(+)
To github.com:arita37/mlmodels_store.git
   c4b68cc..09481c1  master -> master





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
