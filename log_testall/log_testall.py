
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
[master b2d9696] ml_store  && git pull --all
 2 files changed, 1 insertion(+), 53 deletions(-)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + 486468f...b2d9696 master -> master (forced update)





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
[master 53555cc] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   b2d9696..53555cc  master -> master





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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-26 04:14:27.581244: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-26 04:14:27.585905: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-26 04:14:27.586153: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5651562ba1e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 04:14:27.586210: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 158
Trainable params: 158
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.4400 - binary_crossentropy: 6.7870500/500 [==============================] - 1s 1ms/sample - loss: 0.4760 - binary_crossentropy: 7.3423 - val_loss: 0.4820 - val_binary_crossentropy: 7.4348

  #### metrics   #################################################### 
{'MSE': 0.479}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
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
Total params: 158
Trainable params: 158
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
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
Total params: 493
Trainable params: 493
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2769 - binary_crossentropy: 0.7511500/500 [==============================] - 1s 2ms/sample - loss: 0.2690 - binary_crossentropy: 0.7599 - val_loss: 0.2618 - val_binary_crossentropy: 0.7441

  #### metrics   #################################################### 
{'MSE': 0.26473843682106374}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
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
Total params: 493
Trainable params: 493
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
Total params: 627
Trainable params: 627
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2499 - val_binary_crossentropy: 0.6930

  #### metrics   #################################################### 
{'MSE': 0.24983761949414152}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
Total params: 627
Trainable params: 627
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 453
Trainable params: 453
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2608 - binary_crossentropy: 0.9743500/500 [==============================] - 1s 3ms/sample - loss: 0.2718 - binary_crossentropy: 1.0506 - val_loss: 0.2630 - val_binary_crossentropy: 1.0582

  #### metrics   #################################################### 
{'MSE': 0.264515271235115}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 453
Trainable params: 453
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_max[0][0]               
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
Total params: 163
Trainable params: 163
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2487 - binary_crossentropy: 0.6905500/500 [==============================] - 2s 3ms/sample - loss: 0.2509 - binary_crossentropy: 0.6950 - val_loss: 0.2495 - val_binary_crossentropy: 0.6922

  #### metrics   #################################################### 
{'MSE': 0.25012966387934815}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-26 04:15:48.606817: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:15:48.608981: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:15:48.614950: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 04:15:48.625258: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 04:15:48.627129: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 04:15:48.628657: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:15:48.630229: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2514 - val_binary_crossentropy: 0.6960
2020-05-26 04:15:49.901331: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:15:49.903335: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:15:49.907848: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 04:15:49.918441: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 04:15:49.920244: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 04:15:49.921828: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:15:49.923277: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2518076973676931}

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
2020-05-26 04:16:13.624272: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:13.625703: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:13.629509: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 04:16:13.636097: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 04:16:13.637225: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 04:16:13.638295: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:13.639296: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933
2020-05-26 04:16:15.199161: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:15.200479: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:15.204590: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 04:16:15.211100: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 04:16:15.212046: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 04:16:15.212910: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:15.213711: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2500029395658793}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-26 04:16:49.240332: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:49.245276: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:49.259902: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 04:16:49.284742: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 04:16:49.289110: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 04:16:49.293048: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:49.297118: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.4177 - binary_crossentropy: 1.0393 - val_loss: 0.2504 - val_binary_crossentropy: 0.6939
2020-05-26 04:16:51.593550: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:51.598476: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:51.610781: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 04:16:51.636052: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 04:16:51.640485: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 04:16:51.644524: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 04:16:51.648481: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24394782665857306}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 675
Trainable params: 675
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2708 - binary_crossentropy: 0.7367500/500 [==============================] - 4s 8ms/sample - loss: 0.2671 - binary_crossentropy: 0.7288 - val_loss: 0.2632 - val_binary_crossentropy: 0.7211

  #### metrics   #################################################### 
{'MSE': 0.2641761275660748}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 675
Trainable params: 675
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         14          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
Total params: 239
Trainable params: 239
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3660 - binary_crossentropy: 1.5633500/500 [==============================] - 4s 9ms/sample - loss: 0.3779 - binary_crossentropy: 1.4234 - val_loss: 0.3903 - val_binary_crossentropy: 1.4268

  #### metrics   #################################################### 
{'MSE': 0.3788723267242509}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         14          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         16          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
Total params: 239
Trainable params: 239
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
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
Total params: 1,934
Trainable params: 1,934
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2668 - binary_crossentropy: 0.7294500/500 [==============================] - 5s 9ms/sample - loss: 0.2613 - binary_crossentropy: 0.7175 - val_loss: 0.2741 - val_binary_crossentropy: 0.7446

  #### metrics   #################################################### 
{'MSE': 0.2677549764789149}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
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
Total params: 1,934
Trainable params: 1,934
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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         4           regionsequence_max[0][0]         
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
Total params: 128
Trainable params: 128
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2580 - binary_crossentropy: 0.7073500/500 [==============================] - 6s 11ms/sample - loss: 0.2511 - binary_crossentropy: 0.6949 - val_loss: 0.2550 - val_binary_crossentropy: 0.7023

  #### metrics   #################################################### 
{'MSE': 0.252674444490356}

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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         4           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         5           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         4           regionsequence_max[0][0]         
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
Total params: 128
Trainable params: 128
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 1,387
Trainable params: 1,387
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2570 - binary_crossentropy: 0.7075500/500 [==============================] - 6s 11ms/sample - loss: 0.2588 - binary_crossentropy: 0.7377 - val_loss: 0.2534 - val_binary_crossentropy: 0.7781

  #### metrics   #################################################### 
{'MSE': 0.2527094380013732}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 1,387
Trainable params: 1,387
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         28          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         28          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_11[0][0]                    
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
Total params: 3,103
Trainable params: 3,023
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3413 - binary_crossentropy: 0.9260500/500 [==============================] - 6s 12ms/sample - loss: 0.3298 - binary_crossentropy: 0.8917 - val_loss: 0.2959 - val_binary_crossentropy: 0.8072

  #### metrics   #################################################### 
{'MSE': 0.3106366878668254}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         28          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         28          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         28          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 5, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_11[0][0]                    
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
Total params: 3,103
Trainable params: 3,023
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
[master adbfa3e] ml_store  && git pull --all
 1 file changed, 4946 insertions(+)
To github.com:arita37/mlmodels_store.git
 + f9e6d11...adbfa3e master -> master (forced update)





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
[master ee3392c] ml_store  && git pull --all
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   adbfa3e..ee3392c  master -> master





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
[master 6203a31] ml_store  && git pull --all
 1 file changed, 45 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   ee3392c..6203a31  master -> master





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
[master 712b6eb] ml_store  && git pull --all
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   6203a31..712b6eb  master -> master





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
[master fea0628] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   712b6eb..fea0628  master -> master





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
[master 6f11c91] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   fea0628..6f11c91  master -> master





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
[master d9aef85] ml_store  && git pull --all
 1 file changed, 43 insertions(+)
To github.com:arita37/mlmodels_store.git
   6f11c91..d9aef85  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3301376/17464789 [====>.........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
16900096/17464789 [============================>.] - ETA: 0s
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
2020-05-26 04:26:30.876597: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-26 04:26:30.881033: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394450000 Hz
2020-05-26 04:26:30.881197: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a31f6daaf0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 04:26:30.881214: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 8.0193 - accuracy: 0.4770
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8583 - accuracy: 0.4875 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7177 - accuracy: 0.4967
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6935 - accuracy: 0.4983
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7280 - accuracy: 0.4960
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7389 - accuracy: 0.4953
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7395 - accuracy: 0.4952
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7535 - accuracy: 0.4943
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7295 - accuracy: 0.4959
11000/25000 [============>.................] - ETA: 4s - loss: 7.7084 - accuracy: 0.4973
12000/25000 [=============>................] - ETA: 3s - loss: 7.7152 - accuracy: 0.4968
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7303 - accuracy: 0.4958
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7071 - accuracy: 0.4974
15000/25000 [=================>............] - ETA: 2s - loss: 7.7147 - accuracy: 0.4969
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6944 - accuracy: 0.4982
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6856 - accuracy: 0.4988
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6888 - accuracy: 0.4986
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6739 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6613 - accuracy: 0.5003
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6893 - accuracy: 0.4985
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6987 - accuracy: 0.4979
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6946 - accuracy: 0.4982
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6749 - accuracy: 0.4995
25000/25000 [==============================] - 9s 343us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f80ee56dbe0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f8113f902b0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9350 - accuracy: 0.4825 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.9222 - accuracy: 0.4833
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8621 - accuracy: 0.4873
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8782 - accuracy: 0.4862
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8711 - accuracy: 0.4867
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8660 - accuracy: 0.4870
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.8238 - accuracy: 0.4897
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.8131 - accuracy: 0.4904
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7893 - accuracy: 0.4920
11000/25000 [============>.................] - ETA: 4s - loss: 7.7391 - accuracy: 0.4953
12000/25000 [=============>................] - ETA: 3s - loss: 7.7075 - accuracy: 0.4973
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6902 - accuracy: 0.4985
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6984 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 2s - loss: 7.6799 - accuracy: 0.4991
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6752 - accuracy: 0.4994
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6756 - accuracy: 0.4994
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6530 - accuracy: 0.5009
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6303 - accuracy: 0.5024
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6314 - accuracy: 0.5023
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6411 - accuracy: 0.5017
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6471 - accuracy: 0.5013
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6593 - accuracy: 0.5005
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6487 - accuracy: 0.5012
25000/25000 [==============================] - 9s 346us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4060 - accuracy: 0.5170
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5823 - accuracy: 0.5055 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.5593 - accuracy: 0.5070
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7203 - accuracy: 0.4965
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7494 - accuracy: 0.4946
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7203 - accuracy: 0.4965
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6951 - accuracy: 0.4981
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6398 - accuracy: 0.5017
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6479 - accuracy: 0.5012
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6452 - accuracy: 0.5014
11000/25000 [============>.................] - ETA: 4s - loss: 7.6360 - accuracy: 0.5020
12000/25000 [=============>................] - ETA: 3s - loss: 7.6257 - accuracy: 0.5027
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6301 - accuracy: 0.5024
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6644 - accuracy: 0.5001
15000/25000 [=================>............] - ETA: 2s - loss: 7.7004 - accuracy: 0.4978
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6848 - accuracy: 0.4988
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7108 - accuracy: 0.4971
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6990 - accuracy: 0.4979
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7062 - accuracy: 0.4974
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6965 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6878 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6889 - accuracy: 0.4985
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6926 - accuracy: 0.4983
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6858 - accuracy: 0.4988
25000/25000 [==============================] - 9s 349us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master 6b8482e] ml_store  && git pull --all
 1 file changed, 316 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   d9aef85..6b8482e  master -> master





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

13/13 [==============================] - 1s 115ms/step - loss: nan
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
[master 53f6a4a] ml_store  && git pull --all
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   6b8482e..53f6a4a  master -> master





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
 3497984/11490434 [========>.....................] - ETA: 0s
10297344/11490434 [=========================>....] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:05 - loss: 2.3033 - categorical_accuracy: 0.1250
   64/60000 [..............................] - ETA: 4:22 - loss: 2.2678 - categorical_accuracy: 0.1562
   96/60000 [..............................] - ETA: 3:27 - loss: 2.3033 - categorical_accuracy: 0.1458
  128/60000 [..............................] - ETA: 2:59 - loss: 2.2940 - categorical_accuracy: 0.1250
  160/60000 [..............................] - ETA: 2:44 - loss: 2.2639 - categorical_accuracy: 0.1562
  192/60000 [..............................] - ETA: 2:34 - loss: 2.2480 - categorical_accuracy: 0.1823
  224/60000 [..............................] - ETA: 2:25 - loss: 2.2390 - categorical_accuracy: 0.1741
  256/60000 [..............................] - ETA: 2:20 - loss: 2.2079 - categorical_accuracy: 0.2070
  288/60000 [..............................] - ETA: 2:17 - loss: 2.1785 - categorical_accuracy: 0.2326
  320/60000 [..............................] - ETA: 2:13 - loss: 2.1371 - categorical_accuracy: 0.2562
  352/60000 [..............................] - ETA: 2:09 - loss: 2.1340 - categorical_accuracy: 0.2614
  416/60000 [..............................] - ETA: 2:03 - loss: 2.0807 - categorical_accuracy: 0.2933
  448/60000 [..............................] - ETA: 2:01 - loss: 2.0480 - categorical_accuracy: 0.3058
  480/60000 [..............................] - ETA: 1:59 - loss: 2.0291 - categorical_accuracy: 0.3063
  512/60000 [..............................] - ETA: 1:58 - loss: 2.0170 - categorical_accuracy: 0.3164
  544/60000 [..............................] - ETA: 1:57 - loss: 1.9794 - categorical_accuracy: 0.3309
  576/60000 [..............................] - ETA: 1:56 - loss: 1.9500 - categorical_accuracy: 0.3385
  640/60000 [..............................] - ETA: 1:53 - loss: 1.9030 - categorical_accuracy: 0.3500
  672/60000 [..............................] - ETA: 1:52 - loss: 1.8665 - categorical_accuracy: 0.3646
  736/60000 [..............................] - ETA: 1:51 - loss: 1.8156 - categorical_accuracy: 0.3832
  768/60000 [..............................] - ETA: 1:50 - loss: 1.7846 - categorical_accuracy: 0.3919
  800/60000 [..............................] - ETA: 1:50 - loss: 1.7643 - categorical_accuracy: 0.4000
  832/60000 [..............................] - ETA: 1:49 - loss: 1.7522 - categorical_accuracy: 0.4026
  864/60000 [..............................] - ETA: 1:48 - loss: 1.7352 - categorical_accuracy: 0.4120
  896/60000 [..............................] - ETA: 1:48 - loss: 1.7075 - categorical_accuracy: 0.4241
  928/60000 [..............................] - ETA: 1:48 - loss: 1.6851 - categorical_accuracy: 0.4332
  992/60000 [..............................] - ETA: 1:47 - loss: 1.6837 - categorical_accuracy: 0.4365
 1056/60000 [..............................] - ETA: 1:46 - loss: 1.6470 - categorical_accuracy: 0.4517
 1088/60000 [..............................] - ETA: 1:45 - loss: 1.6283 - categorical_accuracy: 0.4577
 1120/60000 [..............................] - ETA: 1:45 - loss: 1.6038 - categorical_accuracy: 0.4643
 1152/60000 [..............................] - ETA: 1:45 - loss: 1.5821 - categorical_accuracy: 0.4714
 1184/60000 [..............................] - ETA: 1:44 - loss: 1.5761 - categorical_accuracy: 0.4747
 1216/60000 [..............................] - ETA: 1:44 - loss: 1.5611 - categorical_accuracy: 0.4794
 1248/60000 [..............................] - ETA: 1:44 - loss: 1.5553 - categorical_accuracy: 0.4840
 1280/60000 [..............................] - ETA: 1:43 - loss: 1.5375 - categorical_accuracy: 0.4898
 1344/60000 [..............................] - ETA: 1:42 - loss: 1.5014 - categorical_accuracy: 0.5007
 1376/60000 [..............................] - ETA: 1:42 - loss: 1.4828 - categorical_accuracy: 0.5051
 1440/60000 [..............................] - ETA: 1:42 - loss: 1.4549 - categorical_accuracy: 0.5167
 1472/60000 [..............................] - ETA: 1:41 - loss: 1.4420 - categorical_accuracy: 0.5211
 1504/60000 [..............................] - ETA: 1:41 - loss: 1.4308 - categorical_accuracy: 0.5239
 1536/60000 [..............................] - ETA: 1:41 - loss: 1.4138 - categorical_accuracy: 0.5293
 1568/60000 [..............................] - ETA: 1:41 - loss: 1.4026 - categorical_accuracy: 0.5319
 1600/60000 [..............................] - ETA: 1:40 - loss: 1.3868 - categorical_accuracy: 0.5369
 1664/60000 [..............................] - ETA: 1:40 - loss: 1.3511 - categorical_accuracy: 0.5517
 1696/60000 [..............................] - ETA: 1:40 - loss: 1.3397 - categorical_accuracy: 0.5548
 1728/60000 [..............................] - ETA: 1:40 - loss: 1.3309 - categorical_accuracy: 0.5579
 1760/60000 [..............................] - ETA: 1:40 - loss: 1.3185 - categorical_accuracy: 0.5619
 1824/60000 [..............................] - ETA: 1:39 - loss: 1.3001 - categorical_accuracy: 0.5674
 1888/60000 [..............................] - ETA: 1:39 - loss: 1.2755 - categorical_accuracy: 0.5747
 1920/60000 [..............................] - ETA: 1:39 - loss: 1.2662 - categorical_accuracy: 0.5781
 1952/60000 [..............................] - ETA: 1:39 - loss: 1.2553 - categorical_accuracy: 0.5809
 2016/60000 [>.............................] - ETA: 1:38 - loss: 1.2419 - categorical_accuracy: 0.5868
 2048/60000 [>.............................] - ETA: 1:38 - loss: 1.2323 - categorical_accuracy: 0.5894
 2080/60000 [>.............................] - ETA: 1:38 - loss: 1.2243 - categorical_accuracy: 0.5928
 2112/60000 [>.............................] - ETA: 1:38 - loss: 1.2210 - categorical_accuracy: 0.5938
 2144/60000 [>.............................] - ETA: 1:38 - loss: 1.2103 - categorical_accuracy: 0.5984
 2176/60000 [>.............................] - ETA: 1:38 - loss: 1.2025 - categorical_accuracy: 0.6006
 2208/60000 [>.............................] - ETA: 1:38 - loss: 1.1919 - categorical_accuracy: 0.6024
 2272/60000 [>.............................] - ETA: 1:38 - loss: 1.1752 - categorical_accuracy: 0.6074
 2336/60000 [>.............................] - ETA: 1:37 - loss: 1.1571 - categorical_accuracy: 0.6147
 2368/60000 [>.............................] - ETA: 1:37 - loss: 1.1529 - categorical_accuracy: 0.6170
 2400/60000 [>.............................] - ETA: 1:37 - loss: 1.1440 - categorical_accuracy: 0.6200
 2432/60000 [>.............................] - ETA: 1:37 - loss: 1.1344 - categorical_accuracy: 0.6238
 2464/60000 [>.............................] - ETA: 1:37 - loss: 1.1283 - categorical_accuracy: 0.6254
 2496/60000 [>.............................] - ETA: 1:37 - loss: 1.1176 - categorical_accuracy: 0.6290
 2528/60000 [>.............................] - ETA: 1:37 - loss: 1.1081 - categorical_accuracy: 0.6321
 2560/60000 [>.............................] - ETA: 1:37 - loss: 1.0985 - categorical_accuracy: 0.6348
 2592/60000 [>.............................] - ETA: 1:36 - loss: 1.0929 - categorical_accuracy: 0.6366
 2624/60000 [>.............................] - ETA: 1:36 - loss: 1.0856 - categorical_accuracy: 0.6387
 2656/60000 [>.............................] - ETA: 1:36 - loss: 1.0789 - categorical_accuracy: 0.6412
 2688/60000 [>.............................] - ETA: 1:36 - loss: 1.0692 - categorical_accuracy: 0.6440
 2720/60000 [>.............................] - ETA: 1:36 - loss: 1.0618 - categorical_accuracy: 0.6463
 2752/60000 [>.............................] - ETA: 1:36 - loss: 1.0552 - categorical_accuracy: 0.6486
 2784/60000 [>.............................] - ETA: 1:36 - loss: 1.0500 - categorical_accuracy: 0.6505
 2816/60000 [>.............................] - ETA: 1:36 - loss: 1.0471 - categorical_accuracy: 0.6516
 2880/60000 [>.............................] - ETA: 1:36 - loss: 1.0415 - categorical_accuracy: 0.6542
 2944/60000 [>.............................] - ETA: 1:35 - loss: 1.0294 - categorical_accuracy: 0.6583
 3008/60000 [>.............................] - ETA: 1:35 - loss: 1.0200 - categorical_accuracy: 0.6616
 3040/60000 [>.............................] - ETA: 1:35 - loss: 1.0135 - categorical_accuracy: 0.6641
 3072/60000 [>.............................] - ETA: 1:35 - loss: 1.0087 - categorical_accuracy: 0.6660
 3104/60000 [>.............................] - ETA: 1:35 - loss: 1.0035 - categorical_accuracy: 0.6688
 3136/60000 [>.............................] - ETA: 1:35 - loss: 0.9973 - categorical_accuracy: 0.6709
 3168/60000 [>.............................] - ETA: 1:35 - loss: 0.9897 - categorical_accuracy: 0.6739
 3200/60000 [>.............................] - ETA: 1:35 - loss: 0.9871 - categorical_accuracy: 0.6737
 3264/60000 [>.............................] - ETA: 1:34 - loss: 0.9739 - categorical_accuracy: 0.6780
 3328/60000 [>.............................] - ETA: 1:34 - loss: 0.9629 - categorical_accuracy: 0.6827
 3392/60000 [>.............................] - ETA: 1:34 - loss: 0.9531 - categorical_accuracy: 0.6860
 3456/60000 [>.............................] - ETA: 1:34 - loss: 0.9481 - categorical_accuracy: 0.6884
 3520/60000 [>.............................] - ETA: 1:33 - loss: 0.9369 - categorical_accuracy: 0.6920
 3584/60000 [>.............................] - ETA: 1:33 - loss: 0.9273 - categorical_accuracy: 0.6953
 3616/60000 [>.............................] - ETA: 1:33 - loss: 0.9219 - categorical_accuracy: 0.6969
 3648/60000 [>.............................] - ETA: 1:33 - loss: 0.9162 - categorical_accuracy: 0.6990
 3680/60000 [>.............................] - ETA: 1:33 - loss: 0.9108 - categorical_accuracy: 0.7011
 3712/60000 [>.............................] - ETA: 1:33 - loss: 0.9047 - categorical_accuracy: 0.7029
 3744/60000 [>.............................] - ETA: 1:33 - loss: 0.9017 - categorical_accuracy: 0.7038
 3808/60000 [>.............................] - ETA: 1:33 - loss: 0.8958 - categorical_accuracy: 0.7054
 3840/60000 [>.............................] - ETA: 1:33 - loss: 0.8906 - categorical_accuracy: 0.7070
 3872/60000 [>.............................] - ETA: 1:33 - loss: 0.8858 - categorical_accuracy: 0.7089
 3904/60000 [>.............................] - ETA: 1:33 - loss: 0.8814 - categorical_accuracy: 0.7106
 3936/60000 [>.............................] - ETA: 1:32 - loss: 0.8765 - categorical_accuracy: 0.7121
 3968/60000 [>.............................] - ETA: 1:32 - loss: 0.8717 - categorical_accuracy: 0.7137
 4000/60000 [=>............................] - ETA: 1:32 - loss: 0.8669 - categorical_accuracy: 0.7153
 4032/60000 [=>............................] - ETA: 1:32 - loss: 0.8636 - categorical_accuracy: 0.7168
 4064/60000 [=>............................] - ETA: 1:32 - loss: 0.8608 - categorical_accuracy: 0.7178
 4096/60000 [=>............................] - ETA: 1:32 - loss: 0.8565 - categorical_accuracy: 0.7192
 4128/60000 [=>............................] - ETA: 1:32 - loss: 0.8534 - categorical_accuracy: 0.7202
 4160/60000 [=>............................] - ETA: 1:32 - loss: 0.8501 - categorical_accuracy: 0.7212
 4192/60000 [=>............................] - ETA: 1:32 - loss: 0.8507 - categorical_accuracy: 0.7211
 4224/60000 [=>............................] - ETA: 1:32 - loss: 0.8465 - categorical_accuracy: 0.7225
 4256/60000 [=>............................] - ETA: 1:32 - loss: 0.8419 - categorical_accuracy: 0.7239
 4288/60000 [=>............................] - ETA: 1:32 - loss: 0.8376 - categorical_accuracy: 0.7257
 4320/60000 [=>............................] - ETA: 1:32 - loss: 0.8350 - categorical_accuracy: 0.7271
 4352/60000 [=>............................] - ETA: 1:32 - loss: 0.8322 - categorical_accuracy: 0.7284
 4384/60000 [=>............................] - ETA: 1:32 - loss: 0.8287 - categorical_accuracy: 0.7295
 4416/60000 [=>............................] - ETA: 1:32 - loss: 0.8260 - categorical_accuracy: 0.7301
 4448/60000 [=>............................] - ETA: 1:32 - loss: 0.8217 - categorical_accuracy: 0.7316
 4480/60000 [=>............................] - ETA: 1:31 - loss: 0.8186 - categorical_accuracy: 0.7328
 4544/60000 [=>............................] - ETA: 1:31 - loss: 0.8116 - categorical_accuracy: 0.7355
 4576/60000 [=>............................] - ETA: 1:31 - loss: 0.8090 - categorical_accuracy: 0.7369
 4608/60000 [=>............................] - ETA: 1:31 - loss: 0.8067 - categorical_accuracy: 0.7381
 4640/60000 [=>............................] - ETA: 1:31 - loss: 0.8022 - categorical_accuracy: 0.7397
 4704/60000 [=>............................] - ETA: 1:31 - loss: 0.7958 - categorical_accuracy: 0.7423
 4768/60000 [=>............................] - ETA: 1:31 - loss: 0.7883 - categorical_accuracy: 0.7450
 4832/60000 [=>............................] - ETA: 1:31 - loss: 0.7805 - categorical_accuracy: 0.7473
 4896/60000 [=>............................] - ETA: 1:30 - loss: 0.7742 - categorical_accuracy: 0.7496
 4928/60000 [=>............................] - ETA: 1:30 - loss: 0.7700 - categorical_accuracy: 0.7510
 4960/60000 [=>............................] - ETA: 1:30 - loss: 0.7664 - categorical_accuracy: 0.7524
 4992/60000 [=>............................] - ETA: 1:30 - loss: 0.7629 - categorical_accuracy: 0.7534
 5024/60000 [=>............................] - ETA: 1:30 - loss: 0.7608 - categorical_accuracy: 0.7542
 5088/60000 [=>............................] - ETA: 1:30 - loss: 0.7544 - categorical_accuracy: 0.7563
 5120/60000 [=>............................] - ETA: 1:30 - loss: 0.7510 - categorical_accuracy: 0.7574
 5184/60000 [=>............................] - ETA: 1:30 - loss: 0.7456 - categorical_accuracy: 0.7595
 5216/60000 [=>............................] - ETA: 1:30 - loss: 0.7432 - categorical_accuracy: 0.7600
 5248/60000 [=>............................] - ETA: 1:30 - loss: 0.7407 - categorical_accuracy: 0.7611
 5280/60000 [=>............................] - ETA: 1:30 - loss: 0.7380 - categorical_accuracy: 0.7617
 5312/60000 [=>............................] - ETA: 1:30 - loss: 0.7354 - categorical_accuracy: 0.7624
 5376/60000 [=>............................] - ETA: 1:29 - loss: 0.7282 - categorical_accuracy: 0.7647
 5440/60000 [=>............................] - ETA: 1:29 - loss: 0.7244 - categorical_accuracy: 0.7660
 5472/60000 [=>............................] - ETA: 1:29 - loss: 0.7208 - categorical_accuracy: 0.7674
 5504/60000 [=>............................] - ETA: 1:29 - loss: 0.7208 - categorical_accuracy: 0.7674
 5536/60000 [=>............................] - ETA: 1:29 - loss: 0.7178 - categorical_accuracy: 0.7682
 5568/60000 [=>............................] - ETA: 1:29 - loss: 0.7155 - categorical_accuracy: 0.7692
 5600/60000 [=>............................] - ETA: 1:29 - loss: 0.7138 - categorical_accuracy: 0.7698
 5632/60000 [=>............................] - ETA: 1:29 - loss: 0.7112 - categorical_accuracy: 0.7704
 5664/60000 [=>............................] - ETA: 1:29 - loss: 0.7089 - categorical_accuracy: 0.7712
 5696/60000 [=>............................] - ETA: 1:29 - loss: 0.7071 - categorical_accuracy: 0.7719
 5728/60000 [=>............................] - ETA: 1:29 - loss: 0.7043 - categorical_accuracy: 0.7727
 5760/60000 [=>............................] - ETA: 1:29 - loss: 0.7024 - categorical_accuracy: 0.7736
 5792/60000 [=>............................] - ETA: 1:29 - loss: 0.7000 - categorical_accuracy: 0.7745
 5856/60000 [=>............................] - ETA: 1:29 - loss: 0.6945 - categorical_accuracy: 0.7763
 5920/60000 [=>............................] - ETA: 1:28 - loss: 0.6900 - categorical_accuracy: 0.7780
 5952/60000 [=>............................] - ETA: 1:28 - loss: 0.6892 - categorical_accuracy: 0.7786
 5984/60000 [=>............................] - ETA: 1:28 - loss: 0.6863 - categorical_accuracy: 0.7796
 6048/60000 [==>...........................] - ETA: 1:28 - loss: 0.6821 - categorical_accuracy: 0.7809
 6080/60000 [==>...........................] - ETA: 1:28 - loss: 0.6800 - categorical_accuracy: 0.7816
 6144/60000 [==>...........................] - ETA: 1:28 - loss: 0.6772 - categorical_accuracy: 0.7822
 6176/60000 [==>...........................] - ETA: 1:28 - loss: 0.6743 - categorical_accuracy: 0.7832
 6208/60000 [==>...........................] - ETA: 1:28 - loss: 0.6714 - categorical_accuracy: 0.7841
 6272/60000 [==>...........................] - ETA: 1:28 - loss: 0.6693 - categorical_accuracy: 0.7849
 6304/60000 [==>...........................] - ETA: 1:28 - loss: 0.6681 - categorical_accuracy: 0.7854
 6368/60000 [==>...........................] - ETA: 1:28 - loss: 0.6627 - categorical_accuracy: 0.7875
 6400/60000 [==>...........................] - ETA: 1:28 - loss: 0.6613 - categorical_accuracy: 0.7878
 6432/60000 [==>...........................] - ETA: 1:27 - loss: 0.6594 - categorical_accuracy: 0.7881
 6464/60000 [==>...........................] - ETA: 1:27 - loss: 0.6579 - categorical_accuracy: 0.7885
 6496/60000 [==>...........................] - ETA: 1:27 - loss: 0.6562 - categorical_accuracy: 0.7891
 6528/60000 [==>...........................] - ETA: 1:27 - loss: 0.6536 - categorical_accuracy: 0.7898
 6592/60000 [==>...........................] - ETA: 1:27 - loss: 0.6506 - categorical_accuracy: 0.7907
 6624/60000 [==>...........................] - ETA: 1:27 - loss: 0.6479 - categorical_accuracy: 0.7915
 6656/60000 [==>...........................] - ETA: 1:27 - loss: 0.6453 - categorical_accuracy: 0.7922
 6688/60000 [==>...........................] - ETA: 1:27 - loss: 0.6450 - categorical_accuracy: 0.7923
 6752/60000 [==>...........................] - ETA: 1:27 - loss: 0.6408 - categorical_accuracy: 0.7937
 6784/60000 [==>...........................] - ETA: 1:27 - loss: 0.6388 - categorical_accuracy: 0.7944
 6816/60000 [==>...........................] - ETA: 1:27 - loss: 0.6371 - categorical_accuracy: 0.7947
 6848/60000 [==>...........................] - ETA: 1:27 - loss: 0.6358 - categorical_accuracy: 0.7948
 6912/60000 [==>...........................] - ETA: 1:27 - loss: 0.6333 - categorical_accuracy: 0.7960
 6944/60000 [==>...........................] - ETA: 1:26 - loss: 0.6310 - categorical_accuracy: 0.7967
 7008/60000 [==>...........................] - ETA: 1:26 - loss: 0.6271 - categorical_accuracy: 0.7982
 7040/60000 [==>...........................] - ETA: 1:26 - loss: 0.6253 - categorical_accuracy: 0.7987
 7104/60000 [==>...........................] - ETA: 1:26 - loss: 0.6218 - categorical_accuracy: 0.8003
 7136/60000 [==>...........................] - ETA: 1:26 - loss: 0.6201 - categorical_accuracy: 0.8007
 7168/60000 [==>...........................] - ETA: 1:26 - loss: 0.6184 - categorical_accuracy: 0.8015
 7200/60000 [==>...........................] - ETA: 1:26 - loss: 0.6177 - categorical_accuracy: 0.8019
 7232/60000 [==>...........................] - ETA: 1:26 - loss: 0.6151 - categorical_accuracy: 0.8028
 7264/60000 [==>...........................] - ETA: 1:26 - loss: 0.6149 - categorical_accuracy: 0.8031
 7328/60000 [==>...........................] - ETA: 1:26 - loss: 0.6103 - categorical_accuracy: 0.8047
 7392/60000 [==>...........................] - ETA: 1:26 - loss: 0.6085 - categorical_accuracy: 0.8053
 7456/60000 [==>...........................] - ETA: 1:26 - loss: 0.6057 - categorical_accuracy: 0.8065
 7488/60000 [==>...........................] - ETA: 1:25 - loss: 0.6034 - categorical_accuracy: 0.8073
 7520/60000 [==>...........................] - ETA: 1:25 - loss: 0.6019 - categorical_accuracy: 0.8078
 7584/60000 [==>...........................] - ETA: 1:25 - loss: 0.6007 - categorical_accuracy: 0.8085
 7648/60000 [==>...........................] - ETA: 1:25 - loss: 0.5986 - categorical_accuracy: 0.8095
 7712/60000 [==>...........................] - ETA: 1:25 - loss: 0.5951 - categorical_accuracy: 0.8106
 7744/60000 [==>...........................] - ETA: 1:25 - loss: 0.5933 - categorical_accuracy: 0.8111
 7776/60000 [==>...........................] - ETA: 1:25 - loss: 0.5917 - categorical_accuracy: 0.8116
 7808/60000 [==>...........................] - ETA: 1:25 - loss: 0.5915 - categorical_accuracy: 0.8117
 7872/60000 [==>...........................] - ETA: 1:25 - loss: 0.5886 - categorical_accuracy: 0.8128
 7904/60000 [==>...........................] - ETA: 1:25 - loss: 0.5873 - categorical_accuracy: 0.8131
 7936/60000 [==>...........................] - ETA: 1:25 - loss: 0.5854 - categorical_accuracy: 0.8138
 7968/60000 [==>...........................] - ETA: 1:25 - loss: 0.5841 - categorical_accuracy: 0.8141
 8032/60000 [===>..........................] - ETA: 1:24 - loss: 0.5810 - categorical_accuracy: 0.8151
 8064/60000 [===>..........................] - ETA: 1:24 - loss: 0.5800 - categorical_accuracy: 0.8155
 8096/60000 [===>..........................] - ETA: 1:24 - loss: 0.5786 - categorical_accuracy: 0.8161
 8128/60000 [===>..........................] - ETA: 1:24 - loss: 0.5770 - categorical_accuracy: 0.8166
 8160/60000 [===>..........................] - ETA: 1:24 - loss: 0.5760 - categorical_accuracy: 0.8168
 8192/60000 [===>..........................] - ETA: 1:24 - loss: 0.5745 - categorical_accuracy: 0.8171
 8256/60000 [===>..........................] - ETA: 1:24 - loss: 0.5718 - categorical_accuracy: 0.8180
 8320/60000 [===>..........................] - ETA: 1:24 - loss: 0.5696 - categorical_accuracy: 0.8188
 8384/60000 [===>..........................] - ETA: 1:24 - loss: 0.5669 - categorical_accuracy: 0.8194
 8416/60000 [===>..........................] - ETA: 1:24 - loss: 0.5660 - categorical_accuracy: 0.8195
 8480/60000 [===>..........................] - ETA: 1:24 - loss: 0.5634 - categorical_accuracy: 0.8203
 8544/60000 [===>..........................] - ETA: 1:24 - loss: 0.5608 - categorical_accuracy: 0.8212
 8576/60000 [===>..........................] - ETA: 1:23 - loss: 0.5603 - categorical_accuracy: 0.8212
 8640/60000 [===>..........................] - ETA: 1:23 - loss: 0.5579 - categorical_accuracy: 0.8216
 8704/60000 [===>..........................] - ETA: 1:23 - loss: 0.5558 - categorical_accuracy: 0.8225
 8768/60000 [===>..........................] - ETA: 1:23 - loss: 0.5537 - categorical_accuracy: 0.8232
 8800/60000 [===>..........................] - ETA: 1:23 - loss: 0.5534 - categorical_accuracy: 0.8234
 8832/60000 [===>..........................] - ETA: 1:23 - loss: 0.5521 - categorical_accuracy: 0.8239
 8864/60000 [===>..........................] - ETA: 1:23 - loss: 0.5506 - categorical_accuracy: 0.8245
 8896/60000 [===>..........................] - ETA: 1:23 - loss: 0.5493 - categorical_accuracy: 0.8249
 8960/60000 [===>..........................] - ETA: 1:23 - loss: 0.5463 - categorical_accuracy: 0.8259
 9024/60000 [===>..........................] - ETA: 1:23 - loss: 0.5435 - categorical_accuracy: 0.8267
 9056/60000 [===>..........................] - ETA: 1:23 - loss: 0.5419 - categorical_accuracy: 0.8272
 9088/60000 [===>..........................] - ETA: 1:23 - loss: 0.5417 - categorical_accuracy: 0.8272
 9152/60000 [===>..........................] - ETA: 1:22 - loss: 0.5398 - categorical_accuracy: 0.8281
 9216/60000 [===>..........................] - ETA: 1:22 - loss: 0.5378 - categorical_accuracy: 0.8289
 9248/60000 [===>..........................] - ETA: 1:22 - loss: 0.5362 - categorical_accuracy: 0.8295
 9280/60000 [===>..........................] - ETA: 1:22 - loss: 0.5353 - categorical_accuracy: 0.8297
 9312/60000 [===>..........................] - ETA: 1:22 - loss: 0.5342 - categorical_accuracy: 0.8302
 9344/60000 [===>..........................] - ETA: 1:22 - loss: 0.5328 - categorical_accuracy: 0.8306
 9408/60000 [===>..........................] - ETA: 1:22 - loss: 0.5304 - categorical_accuracy: 0.8315
 9440/60000 [===>..........................] - ETA: 1:22 - loss: 0.5292 - categorical_accuracy: 0.8318
 9504/60000 [===>..........................] - ETA: 1:22 - loss: 0.5270 - categorical_accuracy: 0.8325
 9568/60000 [===>..........................] - ETA: 1:22 - loss: 0.5257 - categorical_accuracy: 0.8330
 9632/60000 [===>..........................] - ETA: 1:21 - loss: 0.5236 - categorical_accuracy: 0.8334
 9664/60000 [===>..........................] - ETA: 1:21 - loss: 0.5231 - categorical_accuracy: 0.8336
 9696/60000 [===>..........................] - ETA: 1:21 - loss: 0.5221 - categorical_accuracy: 0.8340
 9728/60000 [===>..........................] - ETA: 1:21 - loss: 0.5211 - categorical_accuracy: 0.8343
 9760/60000 [===>..........................] - ETA: 1:21 - loss: 0.5201 - categorical_accuracy: 0.8347
 9792/60000 [===>..........................] - ETA: 1:21 - loss: 0.5195 - categorical_accuracy: 0.8349
 9856/60000 [===>..........................] - ETA: 1:21 - loss: 0.5174 - categorical_accuracy: 0.8355
 9888/60000 [===>..........................] - ETA: 1:21 - loss: 0.5162 - categorical_accuracy: 0.8359
 9920/60000 [===>..........................] - ETA: 1:21 - loss: 0.5148 - categorical_accuracy: 0.8363
 9984/60000 [===>..........................] - ETA: 1:21 - loss: 0.5127 - categorical_accuracy: 0.8369
10016/60000 [====>.........................] - ETA: 1:21 - loss: 0.5117 - categorical_accuracy: 0.8372
10048/60000 [====>.........................] - ETA: 1:21 - loss: 0.5108 - categorical_accuracy: 0.8374
10080/60000 [====>.........................] - ETA: 1:21 - loss: 0.5101 - categorical_accuracy: 0.8377
10112/60000 [====>.........................] - ETA: 1:21 - loss: 0.5086 - categorical_accuracy: 0.8382
10176/60000 [====>.........................] - ETA: 1:21 - loss: 0.5060 - categorical_accuracy: 0.8390
10240/60000 [====>.........................] - ETA: 1:21 - loss: 0.5046 - categorical_accuracy: 0.8396
10272/60000 [====>.........................] - ETA: 1:20 - loss: 0.5034 - categorical_accuracy: 0.8401
10304/60000 [====>.........................] - ETA: 1:20 - loss: 0.5027 - categorical_accuracy: 0.8405
10336/60000 [====>.........................] - ETA: 1:20 - loss: 0.5015 - categorical_accuracy: 0.8408
10400/60000 [====>.........................] - ETA: 1:20 - loss: 0.4992 - categorical_accuracy: 0.8416
10432/60000 [====>.........................] - ETA: 1:20 - loss: 0.4979 - categorical_accuracy: 0.8420
10464/60000 [====>.........................] - ETA: 1:20 - loss: 0.4972 - categorical_accuracy: 0.8423
10496/60000 [====>.........................] - ETA: 1:20 - loss: 0.4962 - categorical_accuracy: 0.8426
10528/60000 [====>.........................] - ETA: 1:20 - loss: 0.4949 - categorical_accuracy: 0.8431
10560/60000 [====>.........................] - ETA: 1:20 - loss: 0.4939 - categorical_accuracy: 0.8435
10592/60000 [====>.........................] - ETA: 1:20 - loss: 0.4925 - categorical_accuracy: 0.8439
10624/60000 [====>.........................] - ETA: 1:20 - loss: 0.4913 - categorical_accuracy: 0.8443
10656/60000 [====>.........................] - ETA: 1:20 - loss: 0.4902 - categorical_accuracy: 0.8447
10688/60000 [====>.........................] - ETA: 1:20 - loss: 0.4894 - categorical_accuracy: 0.8449
10720/60000 [====>.........................] - ETA: 1:20 - loss: 0.4885 - categorical_accuracy: 0.8451
10784/60000 [====>.........................] - ETA: 1:20 - loss: 0.4869 - categorical_accuracy: 0.8456
10848/60000 [====>.........................] - ETA: 1:19 - loss: 0.4866 - categorical_accuracy: 0.8460
10880/60000 [====>.........................] - ETA: 1:19 - loss: 0.4859 - categorical_accuracy: 0.8462
10912/60000 [====>.........................] - ETA: 1:19 - loss: 0.4860 - categorical_accuracy: 0.8463
10944/60000 [====>.........................] - ETA: 1:19 - loss: 0.4853 - categorical_accuracy: 0.8466
10976/60000 [====>.........................] - ETA: 1:19 - loss: 0.4841 - categorical_accuracy: 0.8470
11008/60000 [====>.........................] - ETA: 1:19 - loss: 0.4835 - categorical_accuracy: 0.8473
11040/60000 [====>.........................] - ETA: 1:19 - loss: 0.4833 - categorical_accuracy: 0.8474
11104/60000 [====>.........................] - ETA: 1:19 - loss: 0.4811 - categorical_accuracy: 0.8482
11168/60000 [====>.........................] - ETA: 1:19 - loss: 0.4794 - categorical_accuracy: 0.8488
11200/60000 [====>.........................] - ETA: 1:19 - loss: 0.4782 - categorical_accuracy: 0.8492
11232/60000 [====>.........................] - ETA: 1:19 - loss: 0.4771 - categorical_accuracy: 0.8495
11264/60000 [====>.........................] - ETA: 1:19 - loss: 0.4761 - categorical_accuracy: 0.8499
11296/60000 [====>.........................] - ETA: 1:19 - loss: 0.4751 - categorical_accuracy: 0.8501
11328/60000 [====>.........................] - ETA: 1:19 - loss: 0.4741 - categorical_accuracy: 0.8504
11360/60000 [====>.........................] - ETA: 1:19 - loss: 0.4736 - categorical_accuracy: 0.8505
11392/60000 [====>.........................] - ETA: 1:19 - loss: 0.4726 - categorical_accuracy: 0.8509
11456/60000 [====>.........................] - ETA: 1:18 - loss: 0.4705 - categorical_accuracy: 0.8513
11520/60000 [====>.........................] - ETA: 1:18 - loss: 0.4697 - categorical_accuracy: 0.8516
11552/60000 [====>.........................] - ETA: 1:18 - loss: 0.4685 - categorical_accuracy: 0.8521
11584/60000 [====>.........................] - ETA: 1:18 - loss: 0.4677 - categorical_accuracy: 0.8524
11616/60000 [====>.........................] - ETA: 1:18 - loss: 0.4668 - categorical_accuracy: 0.8527
11680/60000 [====>.........................] - ETA: 1:18 - loss: 0.4650 - categorical_accuracy: 0.8532
11712/60000 [====>.........................] - ETA: 1:18 - loss: 0.4644 - categorical_accuracy: 0.8534
11744/60000 [====>.........................] - ETA: 1:18 - loss: 0.4646 - categorical_accuracy: 0.8535
11776/60000 [====>.........................] - ETA: 1:18 - loss: 0.4636 - categorical_accuracy: 0.8539
11808/60000 [====>.........................] - ETA: 1:18 - loss: 0.4627 - categorical_accuracy: 0.8542
11840/60000 [====>.........................] - ETA: 1:18 - loss: 0.4619 - categorical_accuracy: 0.8545
11872/60000 [====>.........................] - ETA: 1:18 - loss: 0.4610 - categorical_accuracy: 0.8547
11904/60000 [====>.........................] - ETA: 1:18 - loss: 0.4603 - categorical_accuracy: 0.8549
11936/60000 [====>.........................] - ETA: 1:18 - loss: 0.4595 - categorical_accuracy: 0.8551
11968/60000 [====>.........................] - ETA: 1:18 - loss: 0.4595 - categorical_accuracy: 0.8550
12000/60000 [=====>........................] - ETA: 1:18 - loss: 0.4589 - categorical_accuracy: 0.8553
12032/60000 [=====>........................] - ETA: 1:18 - loss: 0.4582 - categorical_accuracy: 0.8555
12064/60000 [=====>........................] - ETA: 1:17 - loss: 0.4573 - categorical_accuracy: 0.8556
12096/60000 [=====>........................] - ETA: 1:17 - loss: 0.4567 - categorical_accuracy: 0.8559
12160/60000 [=====>........................] - ETA: 1:17 - loss: 0.4551 - categorical_accuracy: 0.8562
12224/60000 [=====>........................] - ETA: 1:17 - loss: 0.4555 - categorical_accuracy: 0.8563
12288/60000 [=====>........................] - ETA: 1:17 - loss: 0.4541 - categorical_accuracy: 0.8568
12320/60000 [=====>........................] - ETA: 1:17 - loss: 0.4532 - categorical_accuracy: 0.8570
12352/60000 [=====>........................] - ETA: 1:17 - loss: 0.4525 - categorical_accuracy: 0.8572
12384/60000 [=====>........................] - ETA: 1:17 - loss: 0.4516 - categorical_accuracy: 0.8576
12416/60000 [=====>........................] - ETA: 1:17 - loss: 0.4512 - categorical_accuracy: 0.8577
12448/60000 [=====>........................] - ETA: 1:17 - loss: 0.4507 - categorical_accuracy: 0.8578
12480/60000 [=====>........................] - ETA: 1:17 - loss: 0.4497 - categorical_accuracy: 0.8582
12544/60000 [=====>........................] - ETA: 1:17 - loss: 0.4486 - categorical_accuracy: 0.8585
12576/60000 [=====>........................] - ETA: 1:17 - loss: 0.4477 - categorical_accuracy: 0.8588
12608/60000 [=====>........................] - ETA: 1:17 - loss: 0.4468 - categorical_accuracy: 0.8591
12640/60000 [=====>........................] - ETA: 1:17 - loss: 0.4462 - categorical_accuracy: 0.8593
12672/60000 [=====>........................] - ETA: 1:16 - loss: 0.4452 - categorical_accuracy: 0.8596
12736/60000 [=====>........................] - ETA: 1:16 - loss: 0.4440 - categorical_accuracy: 0.8598
12768/60000 [=====>........................] - ETA: 1:16 - loss: 0.4442 - categorical_accuracy: 0.8598
12800/60000 [=====>........................] - ETA: 1:16 - loss: 0.4436 - categorical_accuracy: 0.8600
12832/60000 [=====>........................] - ETA: 1:16 - loss: 0.4430 - categorical_accuracy: 0.8603
12864/60000 [=====>........................] - ETA: 1:16 - loss: 0.4425 - categorical_accuracy: 0.8605
12896/60000 [=====>........................] - ETA: 1:16 - loss: 0.4418 - categorical_accuracy: 0.8606
12928/60000 [=====>........................] - ETA: 1:16 - loss: 0.4411 - categorical_accuracy: 0.8608
12960/60000 [=====>........................] - ETA: 1:16 - loss: 0.4402 - categorical_accuracy: 0.8611
12992/60000 [=====>........................] - ETA: 1:16 - loss: 0.4394 - categorical_accuracy: 0.8613
13024/60000 [=====>........................] - ETA: 1:16 - loss: 0.4385 - categorical_accuracy: 0.8615
13056/60000 [=====>........................] - ETA: 1:16 - loss: 0.4381 - categorical_accuracy: 0.8616
13088/60000 [=====>........................] - ETA: 1:16 - loss: 0.4379 - categorical_accuracy: 0.8617
13120/60000 [=====>........................] - ETA: 1:16 - loss: 0.4374 - categorical_accuracy: 0.8619
13152/60000 [=====>........................] - ETA: 1:16 - loss: 0.4368 - categorical_accuracy: 0.8622
13184/60000 [=====>........................] - ETA: 1:16 - loss: 0.4360 - categorical_accuracy: 0.8624
13248/60000 [=====>........................] - ETA: 1:16 - loss: 0.4346 - categorical_accuracy: 0.8628
13280/60000 [=====>........................] - ETA: 1:16 - loss: 0.4338 - categorical_accuracy: 0.8630
13312/60000 [=====>........................] - ETA: 1:16 - loss: 0.4340 - categorical_accuracy: 0.8631
13344/60000 [=====>........................] - ETA: 1:15 - loss: 0.4333 - categorical_accuracy: 0.8632
13408/60000 [=====>........................] - ETA: 1:15 - loss: 0.4318 - categorical_accuracy: 0.8637
13440/60000 [=====>........................] - ETA: 1:15 - loss: 0.4314 - categorical_accuracy: 0.8639
13472/60000 [=====>........................] - ETA: 1:15 - loss: 0.4304 - categorical_accuracy: 0.8642
13504/60000 [=====>........................] - ETA: 1:15 - loss: 0.4296 - categorical_accuracy: 0.8645
13536/60000 [=====>........................] - ETA: 1:15 - loss: 0.4290 - categorical_accuracy: 0.8647
13568/60000 [=====>........................] - ETA: 1:15 - loss: 0.4284 - categorical_accuracy: 0.8649
13600/60000 [=====>........................] - ETA: 1:15 - loss: 0.4278 - categorical_accuracy: 0.8651
13664/60000 [=====>........................] - ETA: 1:15 - loss: 0.4264 - categorical_accuracy: 0.8657
13696/60000 [=====>........................] - ETA: 1:15 - loss: 0.4257 - categorical_accuracy: 0.8659
13728/60000 [=====>........................] - ETA: 1:15 - loss: 0.4249 - categorical_accuracy: 0.8663
13760/60000 [=====>........................] - ETA: 1:15 - loss: 0.4244 - categorical_accuracy: 0.8664
13792/60000 [=====>........................] - ETA: 1:15 - loss: 0.4237 - categorical_accuracy: 0.8666
13824/60000 [=====>........................] - ETA: 1:15 - loss: 0.4229 - categorical_accuracy: 0.8668
13888/60000 [=====>........................] - ETA: 1:15 - loss: 0.4219 - categorical_accuracy: 0.8671
13952/60000 [=====>........................] - ETA: 1:14 - loss: 0.4214 - categorical_accuracy: 0.8674
13984/60000 [=====>........................] - ETA: 1:14 - loss: 0.4209 - categorical_accuracy: 0.8676
14016/60000 [======>.......................] - ETA: 1:14 - loss: 0.4208 - categorical_accuracy: 0.8677
14048/60000 [======>.......................] - ETA: 1:14 - loss: 0.4200 - categorical_accuracy: 0.8679
14080/60000 [======>.......................] - ETA: 1:14 - loss: 0.4196 - categorical_accuracy: 0.8680
14112/60000 [======>.......................] - ETA: 1:14 - loss: 0.4188 - categorical_accuracy: 0.8683
14176/60000 [======>.......................] - ETA: 1:14 - loss: 0.4173 - categorical_accuracy: 0.8687
14208/60000 [======>.......................] - ETA: 1:14 - loss: 0.4170 - categorical_accuracy: 0.8687
14240/60000 [======>.......................] - ETA: 1:14 - loss: 0.4162 - categorical_accuracy: 0.8690
14304/60000 [======>.......................] - ETA: 1:14 - loss: 0.4155 - categorical_accuracy: 0.8693
14368/60000 [======>.......................] - ETA: 1:14 - loss: 0.4143 - categorical_accuracy: 0.8698
14400/60000 [======>.......................] - ETA: 1:14 - loss: 0.4139 - categorical_accuracy: 0.8699
14432/60000 [======>.......................] - ETA: 1:14 - loss: 0.4132 - categorical_accuracy: 0.8700
14496/60000 [======>.......................] - ETA: 1:13 - loss: 0.4120 - categorical_accuracy: 0.8703
14528/60000 [======>.......................] - ETA: 1:13 - loss: 0.4113 - categorical_accuracy: 0.8705
14560/60000 [======>.......................] - ETA: 1:13 - loss: 0.4105 - categorical_accuracy: 0.8707
14592/60000 [======>.......................] - ETA: 1:13 - loss: 0.4097 - categorical_accuracy: 0.8710
14624/60000 [======>.......................] - ETA: 1:13 - loss: 0.4090 - categorical_accuracy: 0.8712
14688/60000 [======>.......................] - ETA: 1:13 - loss: 0.4079 - categorical_accuracy: 0.8715
14720/60000 [======>.......................] - ETA: 1:13 - loss: 0.4072 - categorical_accuracy: 0.8718
14752/60000 [======>.......................] - ETA: 1:13 - loss: 0.4066 - categorical_accuracy: 0.8719
14784/60000 [======>.......................] - ETA: 1:13 - loss: 0.4060 - categorical_accuracy: 0.8721
14816/60000 [======>.......................] - ETA: 1:13 - loss: 0.4062 - categorical_accuracy: 0.8720
14848/60000 [======>.......................] - ETA: 1:13 - loss: 0.4060 - categorical_accuracy: 0.8722
14912/60000 [======>.......................] - ETA: 1:13 - loss: 0.4047 - categorical_accuracy: 0.8726
14944/60000 [======>.......................] - ETA: 1:13 - loss: 0.4039 - categorical_accuracy: 0.8729
14976/60000 [======>.......................] - ETA: 1:13 - loss: 0.4033 - categorical_accuracy: 0.8730
15008/60000 [======>.......................] - ETA: 1:13 - loss: 0.4035 - categorical_accuracy: 0.8730
15040/60000 [======>.......................] - ETA: 1:13 - loss: 0.4031 - categorical_accuracy: 0.8731
15072/60000 [======>.......................] - ETA: 1:13 - loss: 0.4026 - categorical_accuracy: 0.8733
15104/60000 [======>.......................] - ETA: 1:12 - loss: 0.4019 - categorical_accuracy: 0.8735
15168/60000 [======>.......................] - ETA: 1:12 - loss: 0.4009 - categorical_accuracy: 0.8738
15232/60000 [======>.......................] - ETA: 1:12 - loss: 0.3997 - categorical_accuracy: 0.8743
15264/60000 [======>.......................] - ETA: 1:12 - loss: 0.3995 - categorical_accuracy: 0.8745
15296/60000 [======>.......................] - ETA: 1:12 - loss: 0.3998 - categorical_accuracy: 0.8744
15328/60000 [======>.......................] - ETA: 1:12 - loss: 0.3991 - categorical_accuracy: 0.8747
15392/60000 [======>.......................] - ETA: 1:12 - loss: 0.3982 - categorical_accuracy: 0.8749
15456/60000 [======>.......................] - ETA: 1:12 - loss: 0.3968 - categorical_accuracy: 0.8754
15520/60000 [======>.......................] - ETA: 1:12 - loss: 0.3960 - categorical_accuracy: 0.8756
15584/60000 [======>.......................] - ETA: 1:12 - loss: 0.3947 - categorical_accuracy: 0.8760
15616/60000 [======>.......................] - ETA: 1:12 - loss: 0.3940 - categorical_accuracy: 0.8763
15680/60000 [======>.......................] - ETA: 1:12 - loss: 0.3928 - categorical_accuracy: 0.8767
15712/60000 [======>.......................] - ETA: 1:11 - loss: 0.3921 - categorical_accuracy: 0.8769
15744/60000 [======>.......................] - ETA: 1:11 - loss: 0.3913 - categorical_accuracy: 0.8771
15776/60000 [======>.......................] - ETA: 1:11 - loss: 0.3908 - categorical_accuracy: 0.8773
15808/60000 [======>.......................] - ETA: 1:11 - loss: 0.3906 - categorical_accuracy: 0.8773
15840/60000 [======>.......................] - ETA: 1:11 - loss: 0.3901 - categorical_accuracy: 0.8775
15872/60000 [======>.......................] - ETA: 1:11 - loss: 0.3898 - categorical_accuracy: 0.8776
15904/60000 [======>.......................] - ETA: 1:11 - loss: 0.3892 - categorical_accuracy: 0.8778
15936/60000 [======>.......................] - ETA: 1:11 - loss: 0.3886 - categorical_accuracy: 0.8780
15968/60000 [======>.......................] - ETA: 1:11 - loss: 0.3880 - categorical_accuracy: 0.8782
16000/60000 [=======>......................] - ETA: 1:11 - loss: 0.3876 - categorical_accuracy: 0.8784
16064/60000 [=======>......................] - ETA: 1:11 - loss: 0.3866 - categorical_accuracy: 0.8787
16096/60000 [=======>......................] - ETA: 1:11 - loss: 0.3863 - categorical_accuracy: 0.8789
16128/60000 [=======>......................] - ETA: 1:11 - loss: 0.3858 - categorical_accuracy: 0.8790
16160/60000 [=======>......................] - ETA: 1:11 - loss: 0.3853 - categorical_accuracy: 0.8791
16192/60000 [=======>......................] - ETA: 1:11 - loss: 0.3846 - categorical_accuracy: 0.8794
16224/60000 [=======>......................] - ETA: 1:11 - loss: 0.3840 - categorical_accuracy: 0.8796
16256/60000 [=======>......................] - ETA: 1:11 - loss: 0.3835 - categorical_accuracy: 0.8797
16288/60000 [=======>......................] - ETA: 1:11 - loss: 0.3830 - categorical_accuracy: 0.8799
16320/60000 [=======>......................] - ETA: 1:10 - loss: 0.3823 - categorical_accuracy: 0.8801
16384/60000 [=======>......................] - ETA: 1:10 - loss: 0.3818 - categorical_accuracy: 0.8804
16448/60000 [=======>......................] - ETA: 1:10 - loss: 0.3813 - categorical_accuracy: 0.8806
16512/60000 [=======>......................] - ETA: 1:10 - loss: 0.3805 - categorical_accuracy: 0.8808
16576/60000 [=======>......................] - ETA: 1:10 - loss: 0.3797 - categorical_accuracy: 0.8810
16640/60000 [=======>......................] - ETA: 1:10 - loss: 0.3787 - categorical_accuracy: 0.8813
16672/60000 [=======>......................] - ETA: 1:10 - loss: 0.3781 - categorical_accuracy: 0.8815
16704/60000 [=======>......................] - ETA: 1:10 - loss: 0.3780 - categorical_accuracy: 0.8815
16768/60000 [=======>......................] - ETA: 1:10 - loss: 0.3776 - categorical_accuracy: 0.8817
16800/60000 [=======>......................] - ETA: 1:10 - loss: 0.3773 - categorical_accuracy: 0.8818
16832/60000 [=======>......................] - ETA: 1:10 - loss: 0.3771 - categorical_accuracy: 0.8819
16864/60000 [=======>......................] - ETA: 1:10 - loss: 0.3767 - categorical_accuracy: 0.8819
16896/60000 [=======>......................] - ETA: 1:09 - loss: 0.3769 - categorical_accuracy: 0.8819
16928/60000 [=======>......................] - ETA: 1:09 - loss: 0.3767 - categorical_accuracy: 0.8819
16960/60000 [=======>......................] - ETA: 1:09 - loss: 0.3762 - categorical_accuracy: 0.8821
17024/60000 [=======>......................] - ETA: 1:09 - loss: 0.3751 - categorical_accuracy: 0.8825
17088/60000 [=======>......................] - ETA: 1:09 - loss: 0.3743 - categorical_accuracy: 0.8825
17120/60000 [=======>......................] - ETA: 1:09 - loss: 0.3742 - categorical_accuracy: 0.8826
17152/60000 [=======>......................] - ETA: 1:09 - loss: 0.3738 - categorical_accuracy: 0.8827
17216/60000 [=======>......................] - ETA: 1:09 - loss: 0.3728 - categorical_accuracy: 0.8830
17280/60000 [=======>......................] - ETA: 1:09 - loss: 0.3719 - categorical_accuracy: 0.8833
17344/60000 [=======>......................] - ETA: 1:09 - loss: 0.3716 - categorical_accuracy: 0.8835
17376/60000 [=======>......................] - ETA: 1:09 - loss: 0.3711 - categorical_accuracy: 0.8837
17408/60000 [=======>......................] - ETA: 1:09 - loss: 0.3708 - categorical_accuracy: 0.8837
17472/60000 [=======>......................] - ETA: 1:09 - loss: 0.3697 - categorical_accuracy: 0.8841
17536/60000 [=======>......................] - ETA: 1:08 - loss: 0.3693 - categorical_accuracy: 0.8842
17568/60000 [=======>......................] - ETA: 1:08 - loss: 0.3689 - categorical_accuracy: 0.8843
17600/60000 [=======>......................] - ETA: 1:08 - loss: 0.3685 - categorical_accuracy: 0.8845
17632/60000 [=======>......................] - ETA: 1:08 - loss: 0.3681 - categorical_accuracy: 0.8846
17696/60000 [=======>......................] - ETA: 1:08 - loss: 0.3670 - categorical_accuracy: 0.8850
17728/60000 [=======>......................] - ETA: 1:08 - loss: 0.3668 - categorical_accuracy: 0.8851
17760/60000 [=======>......................] - ETA: 1:08 - loss: 0.3666 - categorical_accuracy: 0.8852
17792/60000 [=======>......................] - ETA: 1:08 - loss: 0.3662 - categorical_accuracy: 0.8852
17824/60000 [=======>......................] - ETA: 1:08 - loss: 0.3656 - categorical_accuracy: 0.8854
17888/60000 [=======>......................] - ETA: 1:08 - loss: 0.3645 - categorical_accuracy: 0.8857
17920/60000 [=======>......................] - ETA: 1:08 - loss: 0.3640 - categorical_accuracy: 0.8859
17952/60000 [=======>......................] - ETA: 1:08 - loss: 0.3634 - categorical_accuracy: 0.8861
18016/60000 [========>.....................] - ETA: 1:08 - loss: 0.3626 - categorical_accuracy: 0.8863
18080/60000 [========>.....................] - ETA: 1:08 - loss: 0.3615 - categorical_accuracy: 0.8866
18112/60000 [========>.....................] - ETA: 1:08 - loss: 0.3624 - categorical_accuracy: 0.8865
18144/60000 [========>.....................] - ETA: 1:07 - loss: 0.3620 - categorical_accuracy: 0.8867
18176/60000 [========>.....................] - ETA: 1:07 - loss: 0.3614 - categorical_accuracy: 0.8869
18240/60000 [========>.....................] - ETA: 1:07 - loss: 0.3606 - categorical_accuracy: 0.8872
18272/60000 [========>.....................] - ETA: 1:07 - loss: 0.3602 - categorical_accuracy: 0.8873
18336/60000 [========>.....................] - ETA: 1:07 - loss: 0.3600 - categorical_accuracy: 0.8873
18368/60000 [========>.....................] - ETA: 1:07 - loss: 0.3595 - categorical_accuracy: 0.8875
18432/60000 [========>.....................] - ETA: 1:07 - loss: 0.3587 - categorical_accuracy: 0.8877
18464/60000 [========>.....................] - ETA: 1:07 - loss: 0.3582 - categorical_accuracy: 0.8879
18496/60000 [========>.....................] - ETA: 1:07 - loss: 0.3579 - categorical_accuracy: 0.8880
18528/60000 [========>.....................] - ETA: 1:07 - loss: 0.3574 - categorical_accuracy: 0.8881
18560/60000 [========>.....................] - ETA: 1:07 - loss: 0.3574 - categorical_accuracy: 0.8881
18624/60000 [========>.....................] - ETA: 1:07 - loss: 0.3569 - categorical_accuracy: 0.8884
18688/60000 [========>.....................] - ETA: 1:07 - loss: 0.3562 - categorical_accuracy: 0.8886
18752/60000 [========>.....................] - ETA: 1:06 - loss: 0.3552 - categorical_accuracy: 0.8889
18784/60000 [========>.....................] - ETA: 1:06 - loss: 0.3553 - categorical_accuracy: 0.8890
18816/60000 [========>.....................] - ETA: 1:06 - loss: 0.3549 - categorical_accuracy: 0.8891
18848/60000 [========>.....................] - ETA: 1:06 - loss: 0.3545 - categorical_accuracy: 0.8893
18880/60000 [========>.....................] - ETA: 1:06 - loss: 0.3541 - categorical_accuracy: 0.8894
18912/60000 [========>.....................] - ETA: 1:06 - loss: 0.3537 - categorical_accuracy: 0.8895
18944/60000 [========>.....................] - ETA: 1:06 - loss: 0.3531 - categorical_accuracy: 0.8897
18976/60000 [========>.....................] - ETA: 1:06 - loss: 0.3530 - categorical_accuracy: 0.8898
19040/60000 [========>.....................] - ETA: 1:06 - loss: 0.3523 - categorical_accuracy: 0.8900
19072/60000 [========>.....................] - ETA: 1:06 - loss: 0.3519 - categorical_accuracy: 0.8901
19136/60000 [========>.....................] - ETA: 1:06 - loss: 0.3511 - categorical_accuracy: 0.8904
19168/60000 [========>.....................] - ETA: 1:06 - loss: 0.3507 - categorical_accuracy: 0.8905
19200/60000 [========>.....................] - ETA: 1:06 - loss: 0.3503 - categorical_accuracy: 0.8906
19232/60000 [========>.....................] - ETA: 1:06 - loss: 0.3500 - categorical_accuracy: 0.8907
19264/60000 [========>.....................] - ETA: 1:06 - loss: 0.3497 - categorical_accuracy: 0.8908
19296/60000 [========>.....................] - ETA: 1:06 - loss: 0.3495 - categorical_accuracy: 0.8909
19328/60000 [========>.....................] - ETA: 1:05 - loss: 0.3492 - categorical_accuracy: 0.8910
19360/60000 [========>.....................] - ETA: 1:05 - loss: 0.3487 - categorical_accuracy: 0.8912
19424/60000 [========>.....................] - ETA: 1:05 - loss: 0.3477 - categorical_accuracy: 0.8915
19488/60000 [========>.....................] - ETA: 1:05 - loss: 0.3475 - categorical_accuracy: 0.8916
19520/60000 [========>.....................] - ETA: 1:05 - loss: 0.3471 - categorical_accuracy: 0.8916
19584/60000 [========>.....................] - ETA: 1:05 - loss: 0.3465 - categorical_accuracy: 0.8919
19616/60000 [========>.....................] - ETA: 1:05 - loss: 0.3466 - categorical_accuracy: 0.8919
19648/60000 [========>.....................] - ETA: 1:05 - loss: 0.3462 - categorical_accuracy: 0.8921
19680/60000 [========>.....................] - ETA: 1:05 - loss: 0.3464 - categorical_accuracy: 0.8920
19744/60000 [========>.....................] - ETA: 1:05 - loss: 0.3455 - categorical_accuracy: 0.8923
19808/60000 [========>.....................] - ETA: 1:05 - loss: 0.3446 - categorical_accuracy: 0.8926
19872/60000 [========>.....................] - ETA: 1:05 - loss: 0.3438 - categorical_accuracy: 0.8929
19936/60000 [========>.....................] - ETA: 1:04 - loss: 0.3428 - categorical_accuracy: 0.8932
19968/60000 [========>.....................] - ETA: 1:04 - loss: 0.3425 - categorical_accuracy: 0.8933
20032/60000 [=========>....................] - ETA: 1:04 - loss: 0.3418 - categorical_accuracy: 0.8936
20064/60000 [=========>....................] - ETA: 1:04 - loss: 0.3413 - categorical_accuracy: 0.8937
20096/60000 [=========>....................] - ETA: 1:04 - loss: 0.3409 - categorical_accuracy: 0.8939
20160/60000 [=========>....................] - ETA: 1:04 - loss: 0.3401 - categorical_accuracy: 0.8941
20224/60000 [=========>....................] - ETA: 1:04 - loss: 0.3396 - categorical_accuracy: 0.8943
20288/60000 [=========>....................] - ETA: 1:04 - loss: 0.3389 - categorical_accuracy: 0.8946
20352/60000 [=========>....................] - ETA: 1:04 - loss: 0.3381 - categorical_accuracy: 0.8949
20384/60000 [=========>....................] - ETA: 1:04 - loss: 0.3381 - categorical_accuracy: 0.8949
20416/60000 [=========>....................] - ETA: 1:04 - loss: 0.3377 - categorical_accuracy: 0.8950
20448/60000 [=========>....................] - ETA: 1:04 - loss: 0.3372 - categorical_accuracy: 0.8952
20480/60000 [=========>....................] - ETA: 1:03 - loss: 0.3367 - categorical_accuracy: 0.8954
20512/60000 [=========>....................] - ETA: 1:03 - loss: 0.3364 - categorical_accuracy: 0.8954
20544/60000 [=========>....................] - ETA: 1:03 - loss: 0.3359 - categorical_accuracy: 0.8956
20576/60000 [=========>....................] - ETA: 1:03 - loss: 0.3356 - categorical_accuracy: 0.8957
20608/60000 [=========>....................] - ETA: 1:03 - loss: 0.3351 - categorical_accuracy: 0.8959
20672/60000 [=========>....................] - ETA: 1:03 - loss: 0.3348 - categorical_accuracy: 0.8959
20736/60000 [=========>....................] - ETA: 1:03 - loss: 0.3344 - categorical_accuracy: 0.8961
20768/60000 [=========>....................] - ETA: 1:03 - loss: 0.3340 - categorical_accuracy: 0.8961
20832/60000 [=========>....................] - ETA: 1:03 - loss: 0.3342 - categorical_accuracy: 0.8960
20896/60000 [=========>....................] - ETA: 1:03 - loss: 0.3337 - categorical_accuracy: 0.8962
20960/60000 [=========>....................] - ETA: 1:03 - loss: 0.3332 - categorical_accuracy: 0.8963
20992/60000 [=========>....................] - ETA: 1:03 - loss: 0.3328 - categorical_accuracy: 0.8965
21024/60000 [=========>....................] - ETA: 1:03 - loss: 0.3327 - categorical_accuracy: 0.8965
21056/60000 [=========>....................] - ETA: 1:03 - loss: 0.3323 - categorical_accuracy: 0.8966
21088/60000 [=========>....................] - ETA: 1:03 - loss: 0.3321 - categorical_accuracy: 0.8967
21120/60000 [=========>....................] - ETA: 1:02 - loss: 0.3317 - categorical_accuracy: 0.8968
21152/60000 [=========>....................] - ETA: 1:02 - loss: 0.3316 - categorical_accuracy: 0.8968
21184/60000 [=========>....................] - ETA: 1:02 - loss: 0.3313 - categorical_accuracy: 0.8970
21248/60000 [=========>....................] - ETA: 1:02 - loss: 0.3306 - categorical_accuracy: 0.8972
21312/60000 [=========>....................] - ETA: 1:02 - loss: 0.3304 - categorical_accuracy: 0.8974
21344/60000 [=========>....................] - ETA: 1:02 - loss: 0.3306 - categorical_accuracy: 0.8974
21376/60000 [=========>....................] - ETA: 1:02 - loss: 0.3301 - categorical_accuracy: 0.8976
21408/60000 [=========>....................] - ETA: 1:02 - loss: 0.3302 - categorical_accuracy: 0.8976
21440/60000 [=========>....................] - ETA: 1:02 - loss: 0.3299 - categorical_accuracy: 0.8977
21472/60000 [=========>....................] - ETA: 1:02 - loss: 0.3295 - categorical_accuracy: 0.8978
21504/60000 [=========>....................] - ETA: 1:02 - loss: 0.3291 - categorical_accuracy: 0.8980
21568/60000 [=========>....................] - ETA: 1:02 - loss: 0.3286 - categorical_accuracy: 0.8982
21632/60000 [=========>....................] - ETA: 1:02 - loss: 0.3280 - categorical_accuracy: 0.8983
21664/60000 [=========>....................] - ETA: 1:02 - loss: 0.3279 - categorical_accuracy: 0.8984
21696/60000 [=========>....................] - ETA: 1:02 - loss: 0.3279 - categorical_accuracy: 0.8984
21728/60000 [=========>....................] - ETA: 1:01 - loss: 0.3278 - categorical_accuracy: 0.8984
21760/60000 [=========>....................] - ETA: 1:01 - loss: 0.3274 - categorical_accuracy: 0.8985
21824/60000 [=========>....................] - ETA: 1:01 - loss: 0.3266 - categorical_accuracy: 0.8988
21888/60000 [=========>....................] - ETA: 1:01 - loss: 0.3261 - categorical_accuracy: 0.8991
21920/60000 [=========>....................] - ETA: 1:01 - loss: 0.3260 - categorical_accuracy: 0.8991
21952/60000 [=========>....................] - ETA: 1:01 - loss: 0.3257 - categorical_accuracy: 0.8992
21984/60000 [=========>....................] - ETA: 1:01 - loss: 0.3254 - categorical_accuracy: 0.8993
22048/60000 [==========>...................] - ETA: 1:01 - loss: 0.3248 - categorical_accuracy: 0.8995
22112/60000 [==========>...................] - ETA: 1:01 - loss: 0.3246 - categorical_accuracy: 0.8996
22176/60000 [==========>...................] - ETA: 1:01 - loss: 0.3239 - categorical_accuracy: 0.8998
22240/60000 [==========>...................] - ETA: 1:01 - loss: 0.3233 - categorical_accuracy: 0.8999
22304/60000 [==========>...................] - ETA: 1:01 - loss: 0.3231 - categorical_accuracy: 0.9000
22336/60000 [==========>...................] - ETA: 1:00 - loss: 0.3228 - categorical_accuracy: 0.9001
22368/60000 [==========>...................] - ETA: 1:00 - loss: 0.3229 - categorical_accuracy: 0.9001
22432/60000 [==========>...................] - ETA: 1:00 - loss: 0.3223 - categorical_accuracy: 0.9003
22464/60000 [==========>...................] - ETA: 1:00 - loss: 0.3219 - categorical_accuracy: 0.9005
22496/60000 [==========>...................] - ETA: 1:00 - loss: 0.3220 - categorical_accuracy: 0.9005
22560/60000 [==========>...................] - ETA: 1:00 - loss: 0.3214 - categorical_accuracy: 0.9006
22592/60000 [==========>...................] - ETA: 1:00 - loss: 0.3213 - categorical_accuracy: 0.9007
22624/60000 [==========>...................] - ETA: 1:00 - loss: 0.3210 - categorical_accuracy: 0.9008
22656/60000 [==========>...................] - ETA: 1:00 - loss: 0.3208 - categorical_accuracy: 0.9008
22688/60000 [==========>...................] - ETA: 1:00 - loss: 0.3206 - categorical_accuracy: 0.9009
22720/60000 [==========>...................] - ETA: 1:00 - loss: 0.3204 - categorical_accuracy: 0.9010
22784/60000 [==========>...................] - ETA: 1:00 - loss: 0.3203 - categorical_accuracy: 0.9010
22848/60000 [==========>...................] - ETA: 1:00 - loss: 0.3198 - categorical_accuracy: 0.9011
22912/60000 [==========>...................] - ETA: 59s - loss: 0.3194 - categorical_accuracy: 0.9012 
22944/60000 [==========>...................] - ETA: 59s - loss: 0.3191 - categorical_accuracy: 0.9013
22976/60000 [==========>...................] - ETA: 59s - loss: 0.3189 - categorical_accuracy: 0.9013
23040/60000 [==========>...................] - ETA: 59s - loss: 0.3184 - categorical_accuracy: 0.9015
23072/60000 [==========>...................] - ETA: 59s - loss: 0.3181 - categorical_accuracy: 0.9016
23104/60000 [==========>...................] - ETA: 59s - loss: 0.3179 - categorical_accuracy: 0.9017
23136/60000 [==========>...................] - ETA: 59s - loss: 0.3176 - categorical_accuracy: 0.9018
23200/60000 [==========>...................] - ETA: 59s - loss: 0.3168 - categorical_accuracy: 0.9020
23264/60000 [==========>...................] - ETA: 59s - loss: 0.3162 - categorical_accuracy: 0.9023
23328/60000 [==========>...................] - ETA: 59s - loss: 0.3158 - categorical_accuracy: 0.9023
23360/60000 [==========>...................] - ETA: 59s - loss: 0.3155 - categorical_accuracy: 0.9024
23392/60000 [==========>...................] - ETA: 59s - loss: 0.3156 - categorical_accuracy: 0.9024
23424/60000 [==========>...................] - ETA: 59s - loss: 0.3153 - categorical_accuracy: 0.9025
23456/60000 [==========>...................] - ETA: 59s - loss: 0.3150 - categorical_accuracy: 0.9027
23488/60000 [==========>...................] - ETA: 59s - loss: 0.3147 - categorical_accuracy: 0.9027
23520/60000 [==========>...................] - ETA: 58s - loss: 0.3148 - categorical_accuracy: 0.9027
23552/60000 [==========>...................] - ETA: 58s - loss: 0.3145 - categorical_accuracy: 0.9028
23584/60000 [==========>...................] - ETA: 58s - loss: 0.3143 - categorical_accuracy: 0.9029
23648/60000 [==========>...................] - ETA: 58s - loss: 0.3140 - categorical_accuracy: 0.9030
23680/60000 [==========>...................] - ETA: 58s - loss: 0.3139 - categorical_accuracy: 0.9031
23712/60000 [==========>...................] - ETA: 58s - loss: 0.3136 - categorical_accuracy: 0.9032
23744/60000 [==========>...................] - ETA: 58s - loss: 0.3133 - categorical_accuracy: 0.9033
23776/60000 [==========>...................] - ETA: 58s - loss: 0.3131 - categorical_accuracy: 0.9034
23840/60000 [==========>...................] - ETA: 58s - loss: 0.3126 - categorical_accuracy: 0.9035
23872/60000 [==========>...................] - ETA: 58s - loss: 0.3123 - categorical_accuracy: 0.9036
23904/60000 [==========>...................] - ETA: 58s - loss: 0.3122 - categorical_accuracy: 0.9036
23968/60000 [==========>...................] - ETA: 58s - loss: 0.3118 - categorical_accuracy: 0.9036
24032/60000 [===========>..................] - ETA: 58s - loss: 0.3120 - categorical_accuracy: 0.9035
24096/60000 [===========>..................] - ETA: 58s - loss: 0.3119 - categorical_accuracy: 0.9036
24128/60000 [===========>..................] - ETA: 57s - loss: 0.3118 - categorical_accuracy: 0.9036
24192/60000 [===========>..................] - ETA: 57s - loss: 0.3113 - categorical_accuracy: 0.9037
24224/60000 [===========>..................] - ETA: 57s - loss: 0.3110 - categorical_accuracy: 0.9038
24288/60000 [===========>..................] - ETA: 57s - loss: 0.3109 - categorical_accuracy: 0.9039
24320/60000 [===========>..................] - ETA: 57s - loss: 0.3105 - categorical_accuracy: 0.9040
24384/60000 [===========>..................] - ETA: 57s - loss: 0.3099 - categorical_accuracy: 0.9042
24416/60000 [===========>..................] - ETA: 57s - loss: 0.3096 - categorical_accuracy: 0.9043
24480/60000 [===========>..................] - ETA: 57s - loss: 0.3091 - categorical_accuracy: 0.9045
24512/60000 [===========>..................] - ETA: 57s - loss: 0.3088 - categorical_accuracy: 0.9046
24544/60000 [===========>..................] - ETA: 57s - loss: 0.3084 - categorical_accuracy: 0.9047
24608/60000 [===========>..................] - ETA: 57s - loss: 0.3080 - categorical_accuracy: 0.9048
24640/60000 [===========>..................] - ETA: 57s - loss: 0.3079 - categorical_accuracy: 0.9048
24672/60000 [===========>..................] - ETA: 57s - loss: 0.3075 - categorical_accuracy: 0.9049
24736/60000 [===========>..................] - ETA: 56s - loss: 0.3071 - categorical_accuracy: 0.9050
24800/60000 [===========>..................] - ETA: 56s - loss: 0.3070 - categorical_accuracy: 0.9052
24864/60000 [===========>..................] - ETA: 56s - loss: 0.3063 - categorical_accuracy: 0.9054
24928/60000 [===========>..................] - ETA: 56s - loss: 0.3058 - categorical_accuracy: 0.9056
24992/60000 [===========>..................] - ETA: 56s - loss: 0.3051 - categorical_accuracy: 0.9058
25024/60000 [===========>..................] - ETA: 56s - loss: 0.3049 - categorical_accuracy: 0.9058
25088/60000 [===========>..................] - ETA: 56s - loss: 0.3046 - categorical_accuracy: 0.9058
25120/60000 [===========>..................] - ETA: 56s - loss: 0.3045 - categorical_accuracy: 0.9059
25152/60000 [===========>..................] - ETA: 56s - loss: 0.3045 - categorical_accuracy: 0.9059
25216/60000 [===========>..................] - ETA: 56s - loss: 0.3039 - categorical_accuracy: 0.9060
25280/60000 [===========>..................] - ETA: 56s - loss: 0.3034 - categorical_accuracy: 0.9061
25344/60000 [===========>..................] - ETA: 56s - loss: 0.3028 - categorical_accuracy: 0.9064
25376/60000 [===========>..................] - ETA: 55s - loss: 0.3027 - categorical_accuracy: 0.9064
25440/60000 [===========>..................] - ETA: 55s - loss: 0.3021 - categorical_accuracy: 0.9066
25472/60000 [===========>..................] - ETA: 55s - loss: 0.3019 - categorical_accuracy: 0.9066
25504/60000 [===========>..................] - ETA: 55s - loss: 0.3017 - categorical_accuracy: 0.9067
25536/60000 [===========>..................] - ETA: 55s - loss: 0.3013 - categorical_accuracy: 0.9068
25568/60000 [===========>..................] - ETA: 55s - loss: 0.3010 - categorical_accuracy: 0.9069
25600/60000 [===========>..................] - ETA: 55s - loss: 0.3007 - categorical_accuracy: 0.9070
25632/60000 [===========>..................] - ETA: 55s - loss: 0.3005 - categorical_accuracy: 0.9071
25696/60000 [===========>..................] - ETA: 55s - loss: 0.3006 - categorical_accuracy: 0.9071
25760/60000 [===========>..................] - ETA: 55s - loss: 0.3000 - categorical_accuracy: 0.9073
25792/60000 [===========>..................] - ETA: 55s - loss: 0.2997 - categorical_accuracy: 0.9074
25856/60000 [===========>..................] - ETA: 55s - loss: 0.2992 - categorical_accuracy: 0.9075
25920/60000 [===========>..................] - ETA: 55s - loss: 0.2989 - categorical_accuracy: 0.9076
25984/60000 [===========>..................] - ETA: 54s - loss: 0.2984 - categorical_accuracy: 0.9078
26048/60000 [============>.................] - ETA: 54s - loss: 0.2982 - categorical_accuracy: 0.9077
26112/60000 [============>.................] - ETA: 54s - loss: 0.2981 - categorical_accuracy: 0.9077
26176/60000 [============>.................] - ETA: 54s - loss: 0.2979 - categorical_accuracy: 0.9078
26208/60000 [============>.................] - ETA: 54s - loss: 0.2977 - categorical_accuracy: 0.9079
26240/60000 [============>.................] - ETA: 54s - loss: 0.2975 - categorical_accuracy: 0.9079
26272/60000 [============>.................] - ETA: 54s - loss: 0.2973 - categorical_accuracy: 0.9080
26304/60000 [============>.................] - ETA: 54s - loss: 0.2973 - categorical_accuracy: 0.9081
26336/60000 [============>.................] - ETA: 54s - loss: 0.2971 - categorical_accuracy: 0.9081
26400/60000 [============>.................] - ETA: 54s - loss: 0.2965 - categorical_accuracy: 0.9083
26432/60000 [============>.................] - ETA: 54s - loss: 0.2961 - categorical_accuracy: 0.9084
26496/60000 [============>.................] - ETA: 54s - loss: 0.2958 - categorical_accuracy: 0.9085
26528/60000 [============>.................] - ETA: 54s - loss: 0.2955 - categorical_accuracy: 0.9085
26560/60000 [============>.................] - ETA: 53s - loss: 0.2953 - categorical_accuracy: 0.9086
26592/60000 [============>.................] - ETA: 53s - loss: 0.2951 - categorical_accuracy: 0.9087
26624/60000 [============>.................] - ETA: 53s - loss: 0.2952 - categorical_accuracy: 0.9087
26656/60000 [============>.................] - ETA: 53s - loss: 0.2950 - categorical_accuracy: 0.9087
26720/60000 [============>.................] - ETA: 53s - loss: 0.2947 - categorical_accuracy: 0.9089
26752/60000 [============>.................] - ETA: 53s - loss: 0.2945 - categorical_accuracy: 0.9089
26784/60000 [============>.................] - ETA: 53s - loss: 0.2942 - categorical_accuracy: 0.9090
26848/60000 [============>.................] - ETA: 53s - loss: 0.2936 - categorical_accuracy: 0.9092
26880/60000 [============>.................] - ETA: 53s - loss: 0.2935 - categorical_accuracy: 0.9093
26944/60000 [============>.................] - ETA: 53s - loss: 0.2932 - categorical_accuracy: 0.9093
27008/60000 [============>.................] - ETA: 53s - loss: 0.2926 - categorical_accuracy: 0.9095
27072/60000 [============>.................] - ETA: 53s - loss: 0.2922 - categorical_accuracy: 0.9096
27104/60000 [============>.................] - ETA: 53s - loss: 0.2921 - categorical_accuracy: 0.9096
27168/60000 [============>.................] - ETA: 52s - loss: 0.2918 - categorical_accuracy: 0.9096
27232/60000 [============>.................] - ETA: 52s - loss: 0.2914 - categorical_accuracy: 0.9098
27296/60000 [============>.................] - ETA: 52s - loss: 0.2914 - categorical_accuracy: 0.9098
27328/60000 [============>.................] - ETA: 52s - loss: 0.2911 - categorical_accuracy: 0.9099
27392/60000 [============>.................] - ETA: 52s - loss: 0.2906 - categorical_accuracy: 0.9101
27424/60000 [============>.................] - ETA: 52s - loss: 0.2905 - categorical_accuracy: 0.9101
27456/60000 [============>.................] - ETA: 52s - loss: 0.2902 - categorical_accuracy: 0.9102
27488/60000 [============>.................] - ETA: 52s - loss: 0.2899 - categorical_accuracy: 0.9103
27520/60000 [============>.................] - ETA: 52s - loss: 0.2898 - categorical_accuracy: 0.9103
27552/60000 [============>.................] - ETA: 52s - loss: 0.2896 - categorical_accuracy: 0.9104
27584/60000 [============>.................] - ETA: 52s - loss: 0.2895 - categorical_accuracy: 0.9104
27648/60000 [============>.................] - ETA: 52s - loss: 0.2894 - categorical_accuracy: 0.9104
27680/60000 [============>.................] - ETA: 52s - loss: 0.2893 - categorical_accuracy: 0.9105
27712/60000 [============>.................] - ETA: 52s - loss: 0.2891 - categorical_accuracy: 0.9105
27744/60000 [============>.................] - ETA: 52s - loss: 0.2888 - categorical_accuracy: 0.9106
27776/60000 [============>.................] - ETA: 52s - loss: 0.2886 - categorical_accuracy: 0.9106
27808/60000 [============>.................] - ETA: 51s - loss: 0.2884 - categorical_accuracy: 0.9107
27872/60000 [============>.................] - ETA: 51s - loss: 0.2880 - categorical_accuracy: 0.9108
27936/60000 [============>.................] - ETA: 51s - loss: 0.2875 - categorical_accuracy: 0.9109
27968/60000 [============>.................] - ETA: 51s - loss: 0.2873 - categorical_accuracy: 0.9110
28000/60000 [=============>................] - ETA: 51s - loss: 0.2871 - categorical_accuracy: 0.9110
28064/60000 [=============>................] - ETA: 51s - loss: 0.2869 - categorical_accuracy: 0.9111
28128/60000 [=============>................] - ETA: 51s - loss: 0.2865 - categorical_accuracy: 0.9113
28160/60000 [=============>................] - ETA: 51s - loss: 0.2862 - categorical_accuracy: 0.9114
28224/60000 [=============>................] - ETA: 51s - loss: 0.2857 - categorical_accuracy: 0.9115
28256/60000 [=============>................] - ETA: 51s - loss: 0.2854 - categorical_accuracy: 0.9116
28288/60000 [=============>................] - ETA: 51s - loss: 0.2851 - categorical_accuracy: 0.9117
28320/60000 [=============>................] - ETA: 51s - loss: 0.2849 - categorical_accuracy: 0.9118
28352/60000 [=============>................] - ETA: 51s - loss: 0.2846 - categorical_accuracy: 0.9119
28384/60000 [=============>................] - ETA: 50s - loss: 0.2845 - categorical_accuracy: 0.9120
28448/60000 [=============>................] - ETA: 50s - loss: 0.2841 - categorical_accuracy: 0.9121
28512/60000 [=============>................] - ETA: 50s - loss: 0.2835 - categorical_accuracy: 0.9122
28576/60000 [=============>................] - ETA: 50s - loss: 0.2832 - categorical_accuracy: 0.9123
28640/60000 [=============>................] - ETA: 50s - loss: 0.2831 - categorical_accuracy: 0.9124
28672/60000 [=============>................] - ETA: 50s - loss: 0.2830 - categorical_accuracy: 0.9125
28704/60000 [=============>................] - ETA: 50s - loss: 0.2827 - categorical_accuracy: 0.9125
28736/60000 [=============>................] - ETA: 50s - loss: 0.2826 - categorical_accuracy: 0.9125
28768/60000 [=============>................] - ETA: 50s - loss: 0.2823 - categorical_accuracy: 0.9126
28800/60000 [=============>................] - ETA: 50s - loss: 0.2820 - categorical_accuracy: 0.9127
28832/60000 [=============>................] - ETA: 50s - loss: 0.2818 - categorical_accuracy: 0.9128
28864/60000 [=============>................] - ETA: 50s - loss: 0.2817 - categorical_accuracy: 0.9128
28896/60000 [=============>................] - ETA: 50s - loss: 0.2815 - categorical_accuracy: 0.9129
28928/60000 [=============>................] - ETA: 50s - loss: 0.2813 - categorical_accuracy: 0.9129
28960/60000 [=============>................] - ETA: 50s - loss: 0.2811 - categorical_accuracy: 0.9129
28992/60000 [=============>................] - ETA: 50s - loss: 0.2809 - categorical_accuracy: 0.9130
29024/60000 [=============>................] - ETA: 49s - loss: 0.2808 - categorical_accuracy: 0.9131
29088/60000 [=============>................] - ETA: 49s - loss: 0.2808 - categorical_accuracy: 0.9130
29120/60000 [=============>................] - ETA: 49s - loss: 0.2805 - categorical_accuracy: 0.9131
29152/60000 [=============>................] - ETA: 49s - loss: 0.2804 - categorical_accuracy: 0.9131
29216/60000 [=============>................] - ETA: 49s - loss: 0.2799 - categorical_accuracy: 0.9133
29280/60000 [=============>................] - ETA: 49s - loss: 0.2798 - categorical_accuracy: 0.9134
29344/60000 [=============>................] - ETA: 49s - loss: 0.2794 - categorical_accuracy: 0.9135
29376/60000 [=============>................] - ETA: 49s - loss: 0.2793 - categorical_accuracy: 0.9135
29440/60000 [=============>................] - ETA: 49s - loss: 0.2790 - categorical_accuracy: 0.9137
29504/60000 [=============>................] - ETA: 49s - loss: 0.2787 - categorical_accuracy: 0.9137
29568/60000 [=============>................] - ETA: 49s - loss: 0.2785 - categorical_accuracy: 0.9138
29632/60000 [=============>................] - ETA: 48s - loss: 0.2783 - categorical_accuracy: 0.9138
29696/60000 [=============>................] - ETA: 48s - loss: 0.2779 - categorical_accuracy: 0.9140
29728/60000 [=============>................] - ETA: 48s - loss: 0.2776 - categorical_accuracy: 0.9141
29792/60000 [=============>................] - ETA: 48s - loss: 0.2774 - categorical_accuracy: 0.9142
29824/60000 [=============>................] - ETA: 48s - loss: 0.2771 - categorical_accuracy: 0.9143
29856/60000 [=============>................] - ETA: 48s - loss: 0.2770 - categorical_accuracy: 0.9143
29920/60000 [=============>................] - ETA: 48s - loss: 0.2767 - categorical_accuracy: 0.9144
29952/60000 [=============>................] - ETA: 48s - loss: 0.2766 - categorical_accuracy: 0.9144
29984/60000 [=============>................] - ETA: 48s - loss: 0.2764 - categorical_accuracy: 0.9145
30016/60000 [==============>...............] - ETA: 48s - loss: 0.2762 - categorical_accuracy: 0.9145
30048/60000 [==============>...............] - ETA: 48s - loss: 0.2761 - categorical_accuracy: 0.9146
30080/60000 [==============>...............] - ETA: 48s - loss: 0.2758 - categorical_accuracy: 0.9147
30112/60000 [==============>...............] - ETA: 48s - loss: 0.2756 - categorical_accuracy: 0.9147
30176/60000 [==============>...............] - ETA: 48s - loss: 0.2752 - categorical_accuracy: 0.9148
30208/60000 [==============>...............] - ETA: 48s - loss: 0.2751 - categorical_accuracy: 0.9148
30240/60000 [==============>...............] - ETA: 47s - loss: 0.2751 - categorical_accuracy: 0.9148
30304/60000 [==============>...............] - ETA: 47s - loss: 0.2749 - categorical_accuracy: 0.9149
30368/60000 [==============>...............] - ETA: 47s - loss: 0.2746 - categorical_accuracy: 0.9149
30432/60000 [==============>...............] - ETA: 47s - loss: 0.2742 - categorical_accuracy: 0.9150
30496/60000 [==============>...............] - ETA: 47s - loss: 0.2738 - categorical_accuracy: 0.9152
30560/60000 [==============>...............] - ETA: 47s - loss: 0.2734 - categorical_accuracy: 0.9152
30624/60000 [==============>...............] - ETA: 47s - loss: 0.2731 - categorical_accuracy: 0.9153
30656/60000 [==============>...............] - ETA: 47s - loss: 0.2730 - categorical_accuracy: 0.9154
30720/60000 [==============>...............] - ETA: 47s - loss: 0.2727 - categorical_accuracy: 0.9154
30752/60000 [==============>...............] - ETA: 47s - loss: 0.2725 - categorical_accuracy: 0.9155
30784/60000 [==============>...............] - ETA: 47s - loss: 0.2722 - categorical_accuracy: 0.9156
30816/60000 [==============>...............] - ETA: 47s - loss: 0.2720 - categorical_accuracy: 0.9157
30848/60000 [==============>...............] - ETA: 47s - loss: 0.2717 - categorical_accuracy: 0.9158
30880/60000 [==============>...............] - ETA: 46s - loss: 0.2715 - categorical_accuracy: 0.9159
30912/60000 [==============>...............] - ETA: 46s - loss: 0.2714 - categorical_accuracy: 0.9159
30944/60000 [==============>...............] - ETA: 46s - loss: 0.2712 - categorical_accuracy: 0.9159
30976/60000 [==============>...............] - ETA: 46s - loss: 0.2709 - categorical_accuracy: 0.9160
31040/60000 [==============>...............] - ETA: 46s - loss: 0.2709 - categorical_accuracy: 0.9161
31104/60000 [==============>...............] - ETA: 46s - loss: 0.2704 - categorical_accuracy: 0.9162
31168/60000 [==============>...............] - ETA: 46s - loss: 0.2702 - categorical_accuracy: 0.9163
31200/60000 [==============>...............] - ETA: 46s - loss: 0.2701 - categorical_accuracy: 0.9163
31232/60000 [==============>...............] - ETA: 46s - loss: 0.2699 - categorical_accuracy: 0.9164
31264/60000 [==============>...............] - ETA: 46s - loss: 0.2699 - categorical_accuracy: 0.9164
31296/60000 [==============>...............] - ETA: 46s - loss: 0.2697 - categorical_accuracy: 0.9164
31360/60000 [==============>...............] - ETA: 46s - loss: 0.2693 - categorical_accuracy: 0.9165
31392/60000 [==============>...............] - ETA: 46s - loss: 0.2692 - categorical_accuracy: 0.9165
31424/60000 [==============>...............] - ETA: 46s - loss: 0.2690 - categorical_accuracy: 0.9166
31456/60000 [==============>...............] - ETA: 46s - loss: 0.2688 - categorical_accuracy: 0.9167
31488/60000 [==============>...............] - ETA: 45s - loss: 0.2686 - categorical_accuracy: 0.9167
31552/60000 [==============>...............] - ETA: 45s - loss: 0.2684 - categorical_accuracy: 0.9168
31584/60000 [==============>...............] - ETA: 45s - loss: 0.2682 - categorical_accuracy: 0.9168
31648/60000 [==============>...............] - ETA: 45s - loss: 0.2677 - categorical_accuracy: 0.9170
31680/60000 [==============>...............] - ETA: 45s - loss: 0.2675 - categorical_accuracy: 0.9171
31712/60000 [==============>...............] - ETA: 45s - loss: 0.2674 - categorical_accuracy: 0.9171
31744/60000 [==============>...............] - ETA: 45s - loss: 0.2672 - categorical_accuracy: 0.9171
31808/60000 [==============>...............] - ETA: 45s - loss: 0.2669 - categorical_accuracy: 0.9172
31840/60000 [==============>...............] - ETA: 45s - loss: 0.2667 - categorical_accuracy: 0.9172
31904/60000 [==============>...............] - ETA: 45s - loss: 0.2663 - categorical_accuracy: 0.9173
31968/60000 [==============>...............] - ETA: 45s - loss: 0.2660 - categorical_accuracy: 0.9175
32032/60000 [===============>..............] - ETA: 45s - loss: 0.2659 - categorical_accuracy: 0.9175
32096/60000 [===============>..............] - ETA: 44s - loss: 0.2655 - categorical_accuracy: 0.9177
32160/60000 [===============>..............] - ETA: 44s - loss: 0.2652 - categorical_accuracy: 0.9178
32224/60000 [===============>..............] - ETA: 44s - loss: 0.2648 - categorical_accuracy: 0.9179
32256/60000 [===============>..............] - ETA: 44s - loss: 0.2646 - categorical_accuracy: 0.9180
32288/60000 [===============>..............] - ETA: 44s - loss: 0.2645 - categorical_accuracy: 0.9180
32320/60000 [===============>..............] - ETA: 44s - loss: 0.2643 - categorical_accuracy: 0.9181
32384/60000 [===============>..............] - ETA: 44s - loss: 0.2644 - categorical_accuracy: 0.9181
32448/60000 [===============>..............] - ETA: 44s - loss: 0.2644 - categorical_accuracy: 0.9182
32512/60000 [===============>..............] - ETA: 44s - loss: 0.2640 - categorical_accuracy: 0.9183
32544/60000 [===============>..............] - ETA: 44s - loss: 0.2638 - categorical_accuracy: 0.9184
32608/60000 [===============>..............] - ETA: 44s - loss: 0.2636 - categorical_accuracy: 0.9185
32640/60000 [===============>..............] - ETA: 44s - loss: 0.2634 - categorical_accuracy: 0.9186
32672/60000 [===============>..............] - ETA: 44s - loss: 0.2632 - categorical_accuracy: 0.9186
32704/60000 [===============>..............] - ETA: 44s - loss: 0.2631 - categorical_accuracy: 0.9187
32768/60000 [===============>..............] - ETA: 43s - loss: 0.2627 - categorical_accuracy: 0.9188
32832/60000 [===============>..............] - ETA: 43s - loss: 0.2625 - categorical_accuracy: 0.9188
32864/60000 [===============>..............] - ETA: 43s - loss: 0.2623 - categorical_accuracy: 0.9189
32896/60000 [===============>..............] - ETA: 43s - loss: 0.2622 - categorical_accuracy: 0.9190
32960/60000 [===============>..............] - ETA: 43s - loss: 0.2619 - categorical_accuracy: 0.9190
33024/60000 [===============>..............] - ETA: 43s - loss: 0.2615 - categorical_accuracy: 0.9192
33056/60000 [===============>..............] - ETA: 43s - loss: 0.2613 - categorical_accuracy: 0.9192
33120/60000 [===============>..............] - ETA: 43s - loss: 0.2609 - categorical_accuracy: 0.9193
33152/60000 [===============>..............] - ETA: 43s - loss: 0.2607 - categorical_accuracy: 0.9194
33184/60000 [===============>..............] - ETA: 43s - loss: 0.2604 - categorical_accuracy: 0.9194
33248/60000 [===============>..............] - ETA: 43s - loss: 0.2603 - categorical_accuracy: 0.9195
33312/60000 [===============>..............] - ETA: 43s - loss: 0.2601 - categorical_accuracy: 0.9196
33376/60000 [===============>..............] - ETA: 42s - loss: 0.2598 - categorical_accuracy: 0.9197
33408/60000 [===============>..............] - ETA: 42s - loss: 0.2596 - categorical_accuracy: 0.9198
33472/60000 [===============>..............] - ETA: 42s - loss: 0.2595 - categorical_accuracy: 0.9198
33504/60000 [===============>..............] - ETA: 42s - loss: 0.2593 - categorical_accuracy: 0.9199
33536/60000 [===============>..............] - ETA: 42s - loss: 0.2591 - categorical_accuracy: 0.9199
33568/60000 [===============>..............] - ETA: 42s - loss: 0.2589 - categorical_accuracy: 0.9200
33600/60000 [===============>..............] - ETA: 42s - loss: 0.2589 - categorical_accuracy: 0.9200
33632/60000 [===============>..............] - ETA: 42s - loss: 0.2588 - categorical_accuracy: 0.9200
33696/60000 [===============>..............] - ETA: 42s - loss: 0.2586 - categorical_accuracy: 0.9201
33760/60000 [===============>..............] - ETA: 42s - loss: 0.2583 - categorical_accuracy: 0.9202
33824/60000 [===============>..............] - ETA: 42s - loss: 0.2579 - categorical_accuracy: 0.9203
33856/60000 [===============>..............] - ETA: 42s - loss: 0.2578 - categorical_accuracy: 0.9203
33920/60000 [===============>..............] - ETA: 42s - loss: 0.2574 - categorical_accuracy: 0.9205
33984/60000 [===============>..............] - ETA: 41s - loss: 0.2570 - categorical_accuracy: 0.9206
34016/60000 [================>.............] - ETA: 41s - loss: 0.2569 - categorical_accuracy: 0.9206
34048/60000 [================>.............] - ETA: 41s - loss: 0.2567 - categorical_accuracy: 0.9207
34080/60000 [================>.............] - ETA: 41s - loss: 0.2567 - categorical_accuracy: 0.9206
34144/60000 [================>.............] - ETA: 41s - loss: 0.2563 - categorical_accuracy: 0.9207
34208/60000 [================>.............] - ETA: 41s - loss: 0.2559 - categorical_accuracy: 0.9209
34240/60000 [================>.............] - ETA: 41s - loss: 0.2558 - categorical_accuracy: 0.9209
34272/60000 [================>.............] - ETA: 41s - loss: 0.2556 - categorical_accuracy: 0.9210
34336/60000 [================>.............] - ETA: 41s - loss: 0.2552 - categorical_accuracy: 0.9211
34368/60000 [================>.............] - ETA: 41s - loss: 0.2550 - categorical_accuracy: 0.9212
34400/60000 [================>.............] - ETA: 41s - loss: 0.2548 - categorical_accuracy: 0.9212
34464/60000 [================>.............] - ETA: 41s - loss: 0.2544 - categorical_accuracy: 0.9214
34496/60000 [================>.............] - ETA: 41s - loss: 0.2543 - categorical_accuracy: 0.9214
34528/60000 [================>.............] - ETA: 41s - loss: 0.2541 - categorical_accuracy: 0.9215
34560/60000 [================>.............] - ETA: 40s - loss: 0.2539 - categorical_accuracy: 0.9215
34592/60000 [================>.............] - ETA: 40s - loss: 0.2539 - categorical_accuracy: 0.9216
34624/60000 [================>.............] - ETA: 40s - loss: 0.2537 - categorical_accuracy: 0.9216
34656/60000 [================>.............] - ETA: 40s - loss: 0.2537 - categorical_accuracy: 0.9217
34688/60000 [================>.............] - ETA: 40s - loss: 0.2537 - categorical_accuracy: 0.9217
34720/60000 [================>.............] - ETA: 40s - loss: 0.2535 - categorical_accuracy: 0.9217
34784/60000 [================>.............] - ETA: 40s - loss: 0.2533 - categorical_accuracy: 0.9218
34816/60000 [================>.............] - ETA: 40s - loss: 0.2531 - categorical_accuracy: 0.9218
34848/60000 [================>.............] - ETA: 40s - loss: 0.2530 - categorical_accuracy: 0.9218
34912/60000 [================>.............] - ETA: 40s - loss: 0.2530 - categorical_accuracy: 0.9218
34944/60000 [================>.............] - ETA: 40s - loss: 0.2529 - categorical_accuracy: 0.9219
35008/60000 [================>.............] - ETA: 40s - loss: 0.2528 - categorical_accuracy: 0.9219
35072/60000 [================>.............] - ETA: 40s - loss: 0.2524 - categorical_accuracy: 0.9220
35104/60000 [================>.............] - ETA: 40s - loss: 0.2523 - categorical_accuracy: 0.9221
35168/60000 [================>.............] - ETA: 39s - loss: 0.2519 - categorical_accuracy: 0.9222
35232/60000 [================>.............] - ETA: 39s - loss: 0.2517 - categorical_accuracy: 0.9223
35264/60000 [================>.............] - ETA: 39s - loss: 0.2516 - categorical_accuracy: 0.9223
35296/60000 [================>.............] - ETA: 39s - loss: 0.2514 - categorical_accuracy: 0.9224
35328/60000 [================>.............] - ETA: 39s - loss: 0.2513 - categorical_accuracy: 0.9224
35360/60000 [================>.............] - ETA: 39s - loss: 0.2511 - categorical_accuracy: 0.9224
35392/60000 [================>.............] - ETA: 39s - loss: 0.2509 - categorical_accuracy: 0.9225
35424/60000 [================>.............] - ETA: 39s - loss: 0.2507 - categorical_accuracy: 0.9226
35488/60000 [================>.............] - ETA: 39s - loss: 0.2503 - categorical_accuracy: 0.9227
35552/60000 [================>.............] - ETA: 39s - loss: 0.2498 - categorical_accuracy: 0.9228
35584/60000 [================>.............] - ETA: 39s - loss: 0.2498 - categorical_accuracy: 0.9229
35648/60000 [================>.............] - ETA: 39s - loss: 0.2496 - categorical_accuracy: 0.9229
35680/60000 [================>.............] - ETA: 39s - loss: 0.2499 - categorical_accuracy: 0.9229
35712/60000 [================>.............] - ETA: 39s - loss: 0.2498 - categorical_accuracy: 0.9229
35744/60000 [================>.............] - ETA: 39s - loss: 0.2498 - categorical_accuracy: 0.9230
35808/60000 [================>.............] - ETA: 38s - loss: 0.2494 - categorical_accuracy: 0.9231
35872/60000 [================>.............] - ETA: 38s - loss: 0.2492 - categorical_accuracy: 0.9231
35936/60000 [================>.............] - ETA: 38s - loss: 0.2488 - categorical_accuracy: 0.9233
36000/60000 [=================>............] - ETA: 38s - loss: 0.2485 - categorical_accuracy: 0.9233
36032/60000 [=================>............] - ETA: 38s - loss: 0.2483 - categorical_accuracy: 0.9234
36064/60000 [=================>............] - ETA: 38s - loss: 0.2482 - categorical_accuracy: 0.9234
36096/60000 [=================>............] - ETA: 38s - loss: 0.2480 - categorical_accuracy: 0.9235
36128/60000 [=================>............] - ETA: 38s - loss: 0.2479 - categorical_accuracy: 0.9235
36192/60000 [=================>............] - ETA: 38s - loss: 0.2480 - categorical_accuracy: 0.9235
36224/60000 [=================>............] - ETA: 38s - loss: 0.2478 - categorical_accuracy: 0.9235
36288/60000 [=================>............] - ETA: 38s - loss: 0.2474 - categorical_accuracy: 0.9237
36320/60000 [=================>............] - ETA: 38s - loss: 0.2472 - categorical_accuracy: 0.9237
36352/60000 [=================>............] - ETA: 38s - loss: 0.2471 - categorical_accuracy: 0.9238
36416/60000 [=================>............] - ETA: 37s - loss: 0.2469 - categorical_accuracy: 0.9239
36448/60000 [=================>............] - ETA: 37s - loss: 0.2468 - categorical_accuracy: 0.9239
36512/60000 [=================>............] - ETA: 37s - loss: 0.2464 - categorical_accuracy: 0.9240
36544/60000 [=================>............] - ETA: 37s - loss: 0.2462 - categorical_accuracy: 0.9240
36576/60000 [=================>............] - ETA: 37s - loss: 0.2461 - categorical_accuracy: 0.9241
36640/60000 [=================>............] - ETA: 37s - loss: 0.2458 - categorical_accuracy: 0.9241
36672/60000 [=================>............] - ETA: 37s - loss: 0.2456 - categorical_accuracy: 0.9242
36704/60000 [=================>............] - ETA: 37s - loss: 0.2454 - categorical_accuracy: 0.9243
36768/60000 [=================>............] - ETA: 37s - loss: 0.2454 - categorical_accuracy: 0.9243
36832/60000 [=================>............] - ETA: 37s - loss: 0.2452 - categorical_accuracy: 0.9244
36896/60000 [=================>............] - ETA: 37s - loss: 0.2451 - categorical_accuracy: 0.9244
36960/60000 [=================>............] - ETA: 37s - loss: 0.2451 - categorical_accuracy: 0.9245
36992/60000 [=================>............] - ETA: 37s - loss: 0.2450 - categorical_accuracy: 0.9245
37024/60000 [=================>............] - ETA: 36s - loss: 0.2449 - categorical_accuracy: 0.9245
37056/60000 [=================>............] - ETA: 36s - loss: 0.2448 - categorical_accuracy: 0.9246
37088/60000 [=================>............] - ETA: 36s - loss: 0.2447 - categorical_accuracy: 0.9246
37152/60000 [=================>............] - ETA: 36s - loss: 0.2443 - categorical_accuracy: 0.9247
37216/60000 [=================>............] - ETA: 36s - loss: 0.2439 - categorical_accuracy: 0.9248
37280/60000 [=================>............] - ETA: 36s - loss: 0.2440 - categorical_accuracy: 0.9248
37312/60000 [=================>............] - ETA: 36s - loss: 0.2439 - categorical_accuracy: 0.9249
37344/60000 [=================>............] - ETA: 36s - loss: 0.2437 - categorical_accuracy: 0.9249
37408/60000 [=================>............] - ETA: 36s - loss: 0.2436 - categorical_accuracy: 0.9249
37472/60000 [=================>............] - ETA: 36s - loss: 0.2435 - categorical_accuracy: 0.9250
37504/60000 [=================>............] - ETA: 36s - loss: 0.2434 - categorical_accuracy: 0.9250
37568/60000 [=================>............] - ETA: 36s - loss: 0.2432 - categorical_accuracy: 0.9251
37632/60000 [=================>............] - ETA: 35s - loss: 0.2431 - categorical_accuracy: 0.9251
37664/60000 [=================>............] - ETA: 35s - loss: 0.2429 - categorical_accuracy: 0.9252
37696/60000 [=================>............] - ETA: 35s - loss: 0.2427 - categorical_accuracy: 0.9252
37760/60000 [=================>............] - ETA: 35s - loss: 0.2427 - categorical_accuracy: 0.9252
37824/60000 [=================>............] - ETA: 35s - loss: 0.2428 - categorical_accuracy: 0.9253
37888/60000 [=================>............] - ETA: 35s - loss: 0.2425 - categorical_accuracy: 0.9253
37920/60000 [=================>............] - ETA: 35s - loss: 0.2424 - categorical_accuracy: 0.9254
37952/60000 [=================>............] - ETA: 35s - loss: 0.2424 - categorical_accuracy: 0.9254
37984/60000 [=================>............] - ETA: 35s - loss: 0.2423 - categorical_accuracy: 0.9254
38016/60000 [==================>...........] - ETA: 35s - loss: 0.2421 - categorical_accuracy: 0.9255
38048/60000 [==================>...........] - ETA: 35s - loss: 0.2421 - categorical_accuracy: 0.9255
38080/60000 [==================>...........] - ETA: 35s - loss: 0.2419 - categorical_accuracy: 0.9255
38112/60000 [==================>...........] - ETA: 35s - loss: 0.2417 - categorical_accuracy: 0.9256
38144/60000 [==================>...........] - ETA: 35s - loss: 0.2416 - categorical_accuracy: 0.9256
38176/60000 [==================>...........] - ETA: 35s - loss: 0.2416 - categorical_accuracy: 0.9256
38240/60000 [==================>...........] - ETA: 35s - loss: 0.2412 - categorical_accuracy: 0.9257
38304/60000 [==================>...........] - ETA: 34s - loss: 0.2410 - categorical_accuracy: 0.9258
38336/60000 [==================>...........] - ETA: 34s - loss: 0.2408 - categorical_accuracy: 0.9258
38400/60000 [==================>...........] - ETA: 34s - loss: 0.2407 - categorical_accuracy: 0.9259
38432/60000 [==================>...........] - ETA: 34s - loss: 0.2405 - categorical_accuracy: 0.9259
38464/60000 [==================>...........] - ETA: 34s - loss: 0.2404 - categorical_accuracy: 0.9259
38528/60000 [==================>...........] - ETA: 34s - loss: 0.2401 - categorical_accuracy: 0.9260
38592/60000 [==================>...........] - ETA: 34s - loss: 0.2398 - categorical_accuracy: 0.9261
38624/60000 [==================>...........] - ETA: 34s - loss: 0.2397 - categorical_accuracy: 0.9261
38688/60000 [==================>...........] - ETA: 34s - loss: 0.2393 - categorical_accuracy: 0.9262
38720/60000 [==================>...........] - ETA: 34s - loss: 0.2392 - categorical_accuracy: 0.9263
38784/60000 [==================>...........] - ETA: 34s - loss: 0.2392 - categorical_accuracy: 0.9263
38816/60000 [==================>...........] - ETA: 34s - loss: 0.2393 - categorical_accuracy: 0.9263
38880/60000 [==================>...........] - ETA: 33s - loss: 0.2390 - categorical_accuracy: 0.9264
38944/60000 [==================>...........] - ETA: 33s - loss: 0.2390 - categorical_accuracy: 0.9264
38976/60000 [==================>...........] - ETA: 33s - loss: 0.2389 - categorical_accuracy: 0.9264
39040/60000 [==================>...........] - ETA: 33s - loss: 0.2387 - categorical_accuracy: 0.9265
39104/60000 [==================>...........] - ETA: 33s - loss: 0.2385 - categorical_accuracy: 0.9266
39136/60000 [==================>...........] - ETA: 33s - loss: 0.2384 - categorical_accuracy: 0.9266
39168/60000 [==================>...........] - ETA: 33s - loss: 0.2383 - categorical_accuracy: 0.9266
39200/60000 [==================>...........] - ETA: 33s - loss: 0.2382 - categorical_accuracy: 0.9267
39264/60000 [==================>...........] - ETA: 33s - loss: 0.2382 - categorical_accuracy: 0.9267
39296/60000 [==================>...........] - ETA: 33s - loss: 0.2380 - categorical_accuracy: 0.9267
39360/60000 [==================>...........] - ETA: 33s - loss: 0.2377 - categorical_accuracy: 0.9268
39392/60000 [==================>...........] - ETA: 33s - loss: 0.2376 - categorical_accuracy: 0.9269
39424/60000 [==================>...........] - ETA: 33s - loss: 0.2375 - categorical_accuracy: 0.9269
39456/60000 [==================>...........] - ETA: 33s - loss: 0.2373 - categorical_accuracy: 0.9270
39520/60000 [==================>...........] - ETA: 32s - loss: 0.2371 - categorical_accuracy: 0.9270
39552/60000 [==================>...........] - ETA: 32s - loss: 0.2369 - categorical_accuracy: 0.9271
39584/60000 [==================>...........] - ETA: 32s - loss: 0.2369 - categorical_accuracy: 0.9271
39648/60000 [==================>...........] - ETA: 32s - loss: 0.2366 - categorical_accuracy: 0.9272
39680/60000 [==================>...........] - ETA: 32s - loss: 0.2366 - categorical_accuracy: 0.9272
39712/60000 [==================>...........] - ETA: 32s - loss: 0.2365 - categorical_accuracy: 0.9273
39776/60000 [==================>...........] - ETA: 32s - loss: 0.2363 - categorical_accuracy: 0.9273
39840/60000 [==================>...........] - ETA: 32s - loss: 0.2360 - categorical_accuracy: 0.9274
39872/60000 [==================>...........] - ETA: 32s - loss: 0.2360 - categorical_accuracy: 0.9274
39904/60000 [==================>...........] - ETA: 32s - loss: 0.2358 - categorical_accuracy: 0.9275
39936/60000 [==================>...........] - ETA: 32s - loss: 0.2357 - categorical_accuracy: 0.9275
39968/60000 [==================>...........] - ETA: 32s - loss: 0.2356 - categorical_accuracy: 0.9275
40000/60000 [===================>..........] - ETA: 32s - loss: 0.2357 - categorical_accuracy: 0.9276
40032/60000 [===================>..........] - ETA: 32s - loss: 0.2355 - categorical_accuracy: 0.9276
40096/60000 [===================>..........] - ETA: 32s - loss: 0.2354 - categorical_accuracy: 0.9276
40160/60000 [===================>..........] - ETA: 31s - loss: 0.2352 - categorical_accuracy: 0.9277
40192/60000 [===================>..........] - ETA: 31s - loss: 0.2352 - categorical_accuracy: 0.9277
40224/60000 [===================>..........] - ETA: 31s - loss: 0.2351 - categorical_accuracy: 0.9278
40256/60000 [===================>..........] - ETA: 31s - loss: 0.2350 - categorical_accuracy: 0.9278
40288/60000 [===================>..........] - ETA: 31s - loss: 0.2348 - categorical_accuracy: 0.9278
40352/60000 [===================>..........] - ETA: 31s - loss: 0.2346 - categorical_accuracy: 0.9279
40384/60000 [===================>..........] - ETA: 31s - loss: 0.2345 - categorical_accuracy: 0.9279
40416/60000 [===================>..........] - ETA: 31s - loss: 0.2345 - categorical_accuracy: 0.9279
40480/60000 [===================>..........] - ETA: 31s - loss: 0.2343 - categorical_accuracy: 0.9280
40512/60000 [===================>..........] - ETA: 31s - loss: 0.2342 - categorical_accuracy: 0.9280
40544/60000 [===================>..........] - ETA: 31s - loss: 0.2341 - categorical_accuracy: 0.9281
40576/60000 [===================>..........] - ETA: 31s - loss: 0.2340 - categorical_accuracy: 0.9281
40608/60000 [===================>..........] - ETA: 31s - loss: 0.2338 - categorical_accuracy: 0.9281
40672/60000 [===================>..........] - ETA: 31s - loss: 0.2335 - categorical_accuracy: 0.9282
40704/60000 [===================>..........] - ETA: 31s - loss: 0.2334 - categorical_accuracy: 0.9283
40768/60000 [===================>..........] - ETA: 30s - loss: 0.2331 - categorical_accuracy: 0.9284
40800/60000 [===================>..........] - ETA: 30s - loss: 0.2329 - categorical_accuracy: 0.9284
40832/60000 [===================>..........] - ETA: 30s - loss: 0.2328 - categorical_accuracy: 0.9285
40896/60000 [===================>..........] - ETA: 30s - loss: 0.2325 - categorical_accuracy: 0.9286
40960/60000 [===================>..........] - ETA: 30s - loss: 0.2323 - categorical_accuracy: 0.9286
40992/60000 [===================>..........] - ETA: 30s - loss: 0.2322 - categorical_accuracy: 0.9286
41056/60000 [===================>..........] - ETA: 30s - loss: 0.2319 - categorical_accuracy: 0.9287
41120/60000 [===================>..........] - ETA: 30s - loss: 0.2319 - categorical_accuracy: 0.9287
41152/60000 [===================>..........] - ETA: 30s - loss: 0.2318 - categorical_accuracy: 0.9288
41216/60000 [===================>..........] - ETA: 30s - loss: 0.2315 - categorical_accuracy: 0.9289
41280/60000 [===================>..........] - ETA: 30s - loss: 0.2312 - categorical_accuracy: 0.9290
41312/60000 [===================>..........] - ETA: 30s - loss: 0.2310 - categorical_accuracy: 0.9290
41376/60000 [===================>..........] - ETA: 29s - loss: 0.2308 - categorical_accuracy: 0.9291
41440/60000 [===================>..........] - ETA: 29s - loss: 0.2304 - categorical_accuracy: 0.9292
41472/60000 [===================>..........] - ETA: 29s - loss: 0.2304 - categorical_accuracy: 0.9293
41536/60000 [===================>..........] - ETA: 29s - loss: 0.2302 - categorical_accuracy: 0.9293
41600/60000 [===================>..........] - ETA: 29s - loss: 0.2301 - categorical_accuracy: 0.9293
41664/60000 [===================>..........] - ETA: 29s - loss: 0.2298 - categorical_accuracy: 0.9293
41728/60000 [===================>..........] - ETA: 29s - loss: 0.2297 - categorical_accuracy: 0.9294
41792/60000 [===================>..........] - ETA: 29s - loss: 0.2294 - categorical_accuracy: 0.9295
41824/60000 [===================>..........] - ETA: 29s - loss: 0.2293 - categorical_accuracy: 0.9295
41856/60000 [===================>..........] - ETA: 29s - loss: 0.2291 - categorical_accuracy: 0.9295
41888/60000 [===================>..........] - ETA: 29s - loss: 0.2290 - categorical_accuracy: 0.9296
41952/60000 [===================>..........] - ETA: 28s - loss: 0.2287 - categorical_accuracy: 0.9296
41984/60000 [===================>..........] - ETA: 28s - loss: 0.2285 - categorical_accuracy: 0.9297
42048/60000 [====================>.........] - ETA: 28s - loss: 0.2284 - categorical_accuracy: 0.9297
42112/60000 [====================>.........] - ETA: 28s - loss: 0.2282 - categorical_accuracy: 0.9298
42144/60000 [====================>.........] - ETA: 28s - loss: 0.2281 - categorical_accuracy: 0.9298
42176/60000 [====================>.........] - ETA: 28s - loss: 0.2280 - categorical_accuracy: 0.9299
42240/60000 [====================>.........] - ETA: 28s - loss: 0.2278 - categorical_accuracy: 0.9299
42304/60000 [====================>.........] - ETA: 28s - loss: 0.2276 - categorical_accuracy: 0.9300
42336/60000 [====================>.........] - ETA: 28s - loss: 0.2275 - categorical_accuracy: 0.9300
42368/60000 [====================>.........] - ETA: 28s - loss: 0.2273 - categorical_accuracy: 0.9301
42432/60000 [====================>.........] - ETA: 28s - loss: 0.2272 - categorical_accuracy: 0.9301
42464/60000 [====================>.........] - ETA: 28s - loss: 0.2271 - categorical_accuracy: 0.9301
42528/60000 [====================>.........] - ETA: 28s - loss: 0.2271 - categorical_accuracy: 0.9301
42592/60000 [====================>.........] - ETA: 27s - loss: 0.2268 - categorical_accuracy: 0.9302
42624/60000 [====================>.........] - ETA: 27s - loss: 0.2267 - categorical_accuracy: 0.9303
42688/60000 [====================>.........] - ETA: 27s - loss: 0.2264 - categorical_accuracy: 0.9304
42752/60000 [====================>.........] - ETA: 27s - loss: 0.2261 - categorical_accuracy: 0.9304
42784/60000 [====================>.........] - ETA: 27s - loss: 0.2260 - categorical_accuracy: 0.9304
42848/60000 [====================>.........] - ETA: 27s - loss: 0.2260 - categorical_accuracy: 0.9304
42912/60000 [====================>.........] - ETA: 27s - loss: 0.2257 - categorical_accuracy: 0.9305
42944/60000 [====================>.........] - ETA: 27s - loss: 0.2257 - categorical_accuracy: 0.9305
42976/60000 [====================>.........] - ETA: 27s - loss: 0.2258 - categorical_accuracy: 0.9305
43008/60000 [====================>.........] - ETA: 27s - loss: 0.2258 - categorical_accuracy: 0.9305
43072/60000 [====================>.........] - ETA: 27s - loss: 0.2256 - categorical_accuracy: 0.9306
43104/60000 [====================>.........] - ETA: 27s - loss: 0.2255 - categorical_accuracy: 0.9306
43136/60000 [====================>.........] - ETA: 27s - loss: 0.2255 - categorical_accuracy: 0.9306
43200/60000 [====================>.........] - ETA: 26s - loss: 0.2252 - categorical_accuracy: 0.9307
43232/60000 [====================>.........] - ETA: 26s - loss: 0.2251 - categorical_accuracy: 0.9307
43264/60000 [====================>.........] - ETA: 26s - loss: 0.2250 - categorical_accuracy: 0.9308
43296/60000 [====================>.........] - ETA: 26s - loss: 0.2248 - categorical_accuracy: 0.9308
43328/60000 [====================>.........] - ETA: 26s - loss: 0.2247 - categorical_accuracy: 0.9309
43360/60000 [====================>.........] - ETA: 26s - loss: 0.2248 - categorical_accuracy: 0.9309
43392/60000 [====================>.........] - ETA: 26s - loss: 0.2247 - categorical_accuracy: 0.9309
43424/60000 [====================>.........] - ETA: 26s - loss: 0.2247 - categorical_accuracy: 0.9309
43488/60000 [====================>.........] - ETA: 26s - loss: 0.2245 - categorical_accuracy: 0.9309
43520/60000 [====================>.........] - ETA: 26s - loss: 0.2245 - categorical_accuracy: 0.9310
43552/60000 [====================>.........] - ETA: 26s - loss: 0.2244 - categorical_accuracy: 0.9310
43616/60000 [====================>.........] - ETA: 26s - loss: 0.2244 - categorical_accuracy: 0.9310
43680/60000 [====================>.........] - ETA: 26s - loss: 0.2242 - categorical_accuracy: 0.9311
43712/60000 [====================>.........] - ETA: 26s - loss: 0.2241 - categorical_accuracy: 0.9311
43744/60000 [====================>.........] - ETA: 26s - loss: 0.2240 - categorical_accuracy: 0.9311
43776/60000 [====================>.........] - ETA: 26s - loss: 0.2239 - categorical_accuracy: 0.9312
43808/60000 [====================>.........] - ETA: 26s - loss: 0.2238 - categorical_accuracy: 0.9312
43840/60000 [====================>.........] - ETA: 25s - loss: 0.2238 - categorical_accuracy: 0.9312
43872/60000 [====================>.........] - ETA: 25s - loss: 0.2236 - categorical_accuracy: 0.9313
43904/60000 [====================>.........] - ETA: 25s - loss: 0.2237 - categorical_accuracy: 0.9313
43936/60000 [====================>.........] - ETA: 25s - loss: 0.2236 - categorical_accuracy: 0.9313
43968/60000 [====================>.........] - ETA: 25s - loss: 0.2235 - categorical_accuracy: 0.9313
44000/60000 [=====================>........] - ETA: 25s - loss: 0.2234 - categorical_accuracy: 0.9313
44032/60000 [=====================>........] - ETA: 25s - loss: 0.2233 - categorical_accuracy: 0.9313
44064/60000 [=====================>........] - ETA: 25s - loss: 0.2233 - categorical_accuracy: 0.9314
44096/60000 [=====================>........] - ETA: 25s - loss: 0.2231 - categorical_accuracy: 0.9314
44160/60000 [=====================>........] - ETA: 25s - loss: 0.2230 - categorical_accuracy: 0.9314
44192/60000 [=====================>........] - ETA: 25s - loss: 0.2229 - categorical_accuracy: 0.9314
44224/60000 [=====================>........] - ETA: 25s - loss: 0.2228 - categorical_accuracy: 0.9315
44256/60000 [=====================>........] - ETA: 25s - loss: 0.2227 - categorical_accuracy: 0.9315
44320/60000 [=====================>........] - ETA: 25s - loss: 0.2224 - categorical_accuracy: 0.9316
44352/60000 [=====================>........] - ETA: 25s - loss: 0.2225 - categorical_accuracy: 0.9316
44384/60000 [=====================>........] - ETA: 25s - loss: 0.2225 - categorical_accuracy: 0.9316
44416/60000 [=====================>........] - ETA: 25s - loss: 0.2225 - categorical_accuracy: 0.9316
44448/60000 [=====================>........] - ETA: 24s - loss: 0.2224 - categorical_accuracy: 0.9316
44480/60000 [=====================>........] - ETA: 24s - loss: 0.2223 - categorical_accuracy: 0.9316
44544/60000 [=====================>........] - ETA: 24s - loss: 0.2222 - categorical_accuracy: 0.9317
44576/60000 [=====================>........] - ETA: 24s - loss: 0.2220 - categorical_accuracy: 0.9317
44640/60000 [=====================>........] - ETA: 24s - loss: 0.2218 - categorical_accuracy: 0.9318
44704/60000 [=====================>........] - ETA: 24s - loss: 0.2215 - categorical_accuracy: 0.9319
44736/60000 [=====================>........] - ETA: 24s - loss: 0.2214 - categorical_accuracy: 0.9319
44800/60000 [=====================>........] - ETA: 24s - loss: 0.2212 - categorical_accuracy: 0.9320
44864/60000 [=====================>........] - ETA: 24s - loss: 0.2211 - categorical_accuracy: 0.9321
44896/60000 [=====================>........] - ETA: 24s - loss: 0.2210 - categorical_accuracy: 0.9321
44928/60000 [=====================>........] - ETA: 24s - loss: 0.2209 - categorical_accuracy: 0.9322
44992/60000 [=====================>........] - ETA: 24s - loss: 0.2208 - categorical_accuracy: 0.9322
45024/60000 [=====================>........] - ETA: 24s - loss: 0.2207 - categorical_accuracy: 0.9322
45056/60000 [=====================>........] - ETA: 24s - loss: 0.2207 - categorical_accuracy: 0.9322
45088/60000 [=====================>........] - ETA: 23s - loss: 0.2206 - categorical_accuracy: 0.9323
45152/60000 [=====================>........] - ETA: 23s - loss: 0.2205 - categorical_accuracy: 0.9323
45216/60000 [=====================>........] - ETA: 23s - loss: 0.2205 - categorical_accuracy: 0.9323
45280/60000 [=====================>........] - ETA: 23s - loss: 0.2204 - categorical_accuracy: 0.9324
45312/60000 [=====================>........] - ETA: 23s - loss: 0.2204 - categorical_accuracy: 0.9324
45344/60000 [=====================>........] - ETA: 23s - loss: 0.2205 - categorical_accuracy: 0.9324
45408/60000 [=====================>........] - ETA: 23s - loss: 0.2204 - categorical_accuracy: 0.9324
45440/60000 [=====================>........] - ETA: 23s - loss: 0.2203 - categorical_accuracy: 0.9324
45472/60000 [=====================>........] - ETA: 23s - loss: 0.2203 - categorical_accuracy: 0.9324
45536/60000 [=====================>........] - ETA: 23s - loss: 0.2201 - categorical_accuracy: 0.9324
45568/60000 [=====================>........] - ETA: 23s - loss: 0.2200 - categorical_accuracy: 0.9325
45600/60000 [=====================>........] - ETA: 23s - loss: 0.2199 - categorical_accuracy: 0.9325
45632/60000 [=====================>........] - ETA: 23s - loss: 0.2198 - categorical_accuracy: 0.9325
45664/60000 [=====================>........] - ETA: 23s - loss: 0.2197 - categorical_accuracy: 0.9326
45728/60000 [=====================>........] - ETA: 22s - loss: 0.2195 - categorical_accuracy: 0.9326
45792/60000 [=====================>........] - ETA: 22s - loss: 0.2193 - categorical_accuracy: 0.9327
45856/60000 [=====================>........] - ETA: 22s - loss: 0.2191 - categorical_accuracy: 0.9327
45920/60000 [=====================>........] - ETA: 22s - loss: 0.2190 - categorical_accuracy: 0.9328
45952/60000 [=====================>........] - ETA: 22s - loss: 0.2189 - categorical_accuracy: 0.9328
45984/60000 [=====================>........] - ETA: 22s - loss: 0.2187 - categorical_accuracy: 0.9329
46048/60000 [======================>.......] - ETA: 22s - loss: 0.2185 - categorical_accuracy: 0.9329
46112/60000 [======================>.......] - ETA: 22s - loss: 0.2183 - categorical_accuracy: 0.9330
46144/60000 [======================>.......] - ETA: 22s - loss: 0.2182 - categorical_accuracy: 0.9331
46176/60000 [======================>.......] - ETA: 22s - loss: 0.2181 - categorical_accuracy: 0.9331
46240/60000 [======================>.......] - ETA: 22s - loss: 0.2179 - categorical_accuracy: 0.9331
46272/60000 [======================>.......] - ETA: 22s - loss: 0.2178 - categorical_accuracy: 0.9332
46304/60000 [======================>.......] - ETA: 22s - loss: 0.2176 - categorical_accuracy: 0.9332
46368/60000 [======================>.......] - ETA: 21s - loss: 0.2175 - categorical_accuracy: 0.9333
46432/60000 [======================>.......] - ETA: 21s - loss: 0.2173 - categorical_accuracy: 0.9333
46496/60000 [======================>.......] - ETA: 21s - loss: 0.2171 - categorical_accuracy: 0.9334
46528/60000 [======================>.......] - ETA: 21s - loss: 0.2170 - categorical_accuracy: 0.9335
46560/60000 [======================>.......] - ETA: 21s - loss: 0.2169 - categorical_accuracy: 0.9335
46592/60000 [======================>.......] - ETA: 21s - loss: 0.2169 - categorical_accuracy: 0.9334
46624/60000 [======================>.......] - ETA: 21s - loss: 0.2168 - categorical_accuracy: 0.9334
46656/60000 [======================>.......] - ETA: 21s - loss: 0.2167 - categorical_accuracy: 0.9335
46688/60000 [======================>.......] - ETA: 21s - loss: 0.2166 - categorical_accuracy: 0.9335
46720/60000 [======================>.......] - ETA: 21s - loss: 0.2165 - categorical_accuracy: 0.9336
46752/60000 [======================>.......] - ETA: 21s - loss: 0.2164 - categorical_accuracy: 0.9336
46784/60000 [======================>.......] - ETA: 21s - loss: 0.2164 - categorical_accuracy: 0.9336
46816/60000 [======================>.......] - ETA: 21s - loss: 0.2165 - categorical_accuracy: 0.9335
46848/60000 [======================>.......] - ETA: 21s - loss: 0.2165 - categorical_accuracy: 0.9335
46880/60000 [======================>.......] - ETA: 21s - loss: 0.2166 - categorical_accuracy: 0.9335
46912/60000 [======================>.......] - ETA: 21s - loss: 0.2165 - categorical_accuracy: 0.9335
46944/60000 [======================>.......] - ETA: 20s - loss: 0.2164 - categorical_accuracy: 0.9335
46976/60000 [======================>.......] - ETA: 20s - loss: 0.2164 - categorical_accuracy: 0.9335
47040/60000 [======================>.......] - ETA: 20s - loss: 0.2162 - categorical_accuracy: 0.9336
47072/60000 [======================>.......] - ETA: 20s - loss: 0.2162 - categorical_accuracy: 0.9336
47104/60000 [======================>.......] - ETA: 20s - loss: 0.2161 - categorical_accuracy: 0.9336
47136/60000 [======================>.......] - ETA: 20s - loss: 0.2160 - categorical_accuracy: 0.9336
47168/60000 [======================>.......] - ETA: 20s - loss: 0.2159 - categorical_accuracy: 0.9337
47200/60000 [======================>.......] - ETA: 20s - loss: 0.2158 - categorical_accuracy: 0.9337
47264/60000 [======================>.......] - ETA: 20s - loss: 0.2156 - categorical_accuracy: 0.9338
47296/60000 [======================>.......] - ETA: 20s - loss: 0.2155 - categorical_accuracy: 0.9338
47360/60000 [======================>.......] - ETA: 20s - loss: 0.2154 - categorical_accuracy: 0.9339
47392/60000 [======================>.......] - ETA: 20s - loss: 0.2153 - categorical_accuracy: 0.9339
47424/60000 [======================>.......] - ETA: 20s - loss: 0.2152 - categorical_accuracy: 0.9339
47456/60000 [======================>.......] - ETA: 20s - loss: 0.2152 - categorical_accuracy: 0.9339
47520/60000 [======================>.......] - ETA: 20s - loss: 0.2150 - categorical_accuracy: 0.9340
47552/60000 [======================>.......] - ETA: 19s - loss: 0.2150 - categorical_accuracy: 0.9339
47584/60000 [======================>.......] - ETA: 19s - loss: 0.2149 - categorical_accuracy: 0.9339
47648/60000 [======================>.......] - ETA: 19s - loss: 0.2147 - categorical_accuracy: 0.9340
47712/60000 [======================>.......] - ETA: 19s - loss: 0.2146 - categorical_accuracy: 0.9340
47744/60000 [======================>.......] - ETA: 19s - loss: 0.2146 - categorical_accuracy: 0.9340
47776/60000 [======================>.......] - ETA: 19s - loss: 0.2145 - categorical_accuracy: 0.9341
47808/60000 [======================>.......] - ETA: 19s - loss: 0.2145 - categorical_accuracy: 0.9341
47840/60000 [======================>.......] - ETA: 19s - loss: 0.2144 - categorical_accuracy: 0.9341
47872/60000 [======================>.......] - ETA: 19s - loss: 0.2143 - categorical_accuracy: 0.9341
47936/60000 [======================>.......] - ETA: 19s - loss: 0.2141 - categorical_accuracy: 0.9342
48000/60000 [=======================>......] - ETA: 19s - loss: 0.2138 - categorical_accuracy: 0.9343
48032/60000 [=======================>......] - ETA: 19s - loss: 0.2137 - categorical_accuracy: 0.9343
48064/60000 [=======================>......] - ETA: 19s - loss: 0.2138 - categorical_accuracy: 0.9343
48128/60000 [=======================>......] - ETA: 19s - loss: 0.2136 - categorical_accuracy: 0.9344
48160/60000 [=======================>......] - ETA: 19s - loss: 0.2135 - categorical_accuracy: 0.9344
48224/60000 [=======================>......] - ETA: 18s - loss: 0.2134 - categorical_accuracy: 0.9344
48288/60000 [=======================>......] - ETA: 18s - loss: 0.2135 - categorical_accuracy: 0.9345
48320/60000 [=======================>......] - ETA: 18s - loss: 0.2133 - categorical_accuracy: 0.9345
48352/60000 [=======================>......] - ETA: 18s - loss: 0.2133 - categorical_accuracy: 0.9345
48416/60000 [=======================>......] - ETA: 18s - loss: 0.2130 - categorical_accuracy: 0.9346
48448/60000 [=======================>......] - ETA: 18s - loss: 0.2129 - categorical_accuracy: 0.9346
48480/60000 [=======================>......] - ETA: 18s - loss: 0.2129 - categorical_accuracy: 0.9346
48544/60000 [=======================>......] - ETA: 18s - loss: 0.2129 - categorical_accuracy: 0.9346
48576/60000 [=======================>......] - ETA: 18s - loss: 0.2127 - categorical_accuracy: 0.9346
48608/60000 [=======================>......] - ETA: 18s - loss: 0.2126 - categorical_accuracy: 0.9347
48640/60000 [=======================>......] - ETA: 18s - loss: 0.2125 - categorical_accuracy: 0.9347
48672/60000 [=======================>......] - ETA: 18s - loss: 0.2125 - categorical_accuracy: 0.9347
48704/60000 [=======================>......] - ETA: 18s - loss: 0.2127 - categorical_accuracy: 0.9347
48736/60000 [=======================>......] - ETA: 18s - loss: 0.2126 - categorical_accuracy: 0.9347
48768/60000 [=======================>......] - ETA: 18s - loss: 0.2126 - categorical_accuracy: 0.9347
48800/60000 [=======================>......] - ETA: 17s - loss: 0.2126 - categorical_accuracy: 0.9347
48832/60000 [=======================>......] - ETA: 17s - loss: 0.2125 - categorical_accuracy: 0.9347
48864/60000 [=======================>......] - ETA: 17s - loss: 0.2125 - categorical_accuracy: 0.9347
48896/60000 [=======================>......] - ETA: 17s - loss: 0.2125 - categorical_accuracy: 0.9347
48928/60000 [=======================>......] - ETA: 17s - loss: 0.2124 - categorical_accuracy: 0.9348
48960/60000 [=======================>......] - ETA: 17s - loss: 0.2123 - categorical_accuracy: 0.9348
48992/60000 [=======================>......] - ETA: 17s - loss: 0.2121 - categorical_accuracy: 0.9349
49024/60000 [=======================>......] - ETA: 17s - loss: 0.2122 - categorical_accuracy: 0.9349
49056/60000 [=======================>......] - ETA: 17s - loss: 0.2121 - categorical_accuracy: 0.9349
49088/60000 [=======================>......] - ETA: 17s - loss: 0.2120 - categorical_accuracy: 0.9350
49120/60000 [=======================>......] - ETA: 17s - loss: 0.2119 - categorical_accuracy: 0.9350
49152/60000 [=======================>......] - ETA: 17s - loss: 0.2117 - categorical_accuracy: 0.9350
49216/60000 [=======================>......] - ETA: 17s - loss: 0.2117 - categorical_accuracy: 0.9351
49248/60000 [=======================>......] - ETA: 17s - loss: 0.2116 - categorical_accuracy: 0.9351
49312/60000 [=======================>......] - ETA: 17s - loss: 0.2114 - categorical_accuracy: 0.9351
49376/60000 [=======================>......] - ETA: 17s - loss: 0.2112 - categorical_accuracy: 0.9352
49408/60000 [=======================>......] - ETA: 17s - loss: 0.2111 - categorical_accuracy: 0.9352
49440/60000 [=======================>......] - ETA: 16s - loss: 0.2110 - categorical_accuracy: 0.9353
49472/60000 [=======================>......] - ETA: 16s - loss: 0.2109 - categorical_accuracy: 0.9353
49504/60000 [=======================>......] - ETA: 16s - loss: 0.2108 - categorical_accuracy: 0.9353
49536/60000 [=======================>......] - ETA: 16s - loss: 0.2108 - categorical_accuracy: 0.9353
49568/60000 [=======================>......] - ETA: 16s - loss: 0.2107 - categorical_accuracy: 0.9354
49600/60000 [=======================>......] - ETA: 16s - loss: 0.2106 - categorical_accuracy: 0.9354
49632/60000 [=======================>......] - ETA: 16s - loss: 0.2105 - categorical_accuracy: 0.9354
49664/60000 [=======================>......] - ETA: 16s - loss: 0.2104 - categorical_accuracy: 0.9354
49696/60000 [=======================>......] - ETA: 16s - loss: 0.2103 - categorical_accuracy: 0.9355
49728/60000 [=======================>......] - ETA: 16s - loss: 0.2102 - categorical_accuracy: 0.9355
49760/60000 [=======================>......] - ETA: 16s - loss: 0.2102 - categorical_accuracy: 0.9355
49824/60000 [=======================>......] - ETA: 16s - loss: 0.2102 - categorical_accuracy: 0.9356
49888/60000 [=======================>......] - ETA: 16s - loss: 0.2100 - categorical_accuracy: 0.9356
49920/60000 [=======================>......] - ETA: 16s - loss: 0.2100 - categorical_accuracy: 0.9356
49952/60000 [=======================>......] - ETA: 16s - loss: 0.2099 - categorical_accuracy: 0.9356
50016/60000 [========================>.....] - ETA: 16s - loss: 0.2098 - categorical_accuracy: 0.9357
50048/60000 [========================>.....] - ETA: 15s - loss: 0.2097 - categorical_accuracy: 0.9357
50112/60000 [========================>.....] - ETA: 15s - loss: 0.2097 - categorical_accuracy: 0.9357
50144/60000 [========================>.....] - ETA: 15s - loss: 0.2096 - categorical_accuracy: 0.9357
50176/60000 [========================>.....] - ETA: 15s - loss: 0.2095 - categorical_accuracy: 0.9357
50208/60000 [========================>.....] - ETA: 15s - loss: 0.2094 - categorical_accuracy: 0.9358
50240/60000 [========================>.....] - ETA: 15s - loss: 0.2094 - categorical_accuracy: 0.9358
50304/60000 [========================>.....] - ETA: 15s - loss: 0.2095 - categorical_accuracy: 0.9358
50336/60000 [========================>.....] - ETA: 15s - loss: 0.2094 - categorical_accuracy: 0.9358
50368/60000 [========================>.....] - ETA: 15s - loss: 0.2094 - categorical_accuracy: 0.9358
50432/60000 [========================>.....] - ETA: 15s - loss: 0.2092 - categorical_accuracy: 0.9359
50496/60000 [========================>.....] - ETA: 15s - loss: 0.2092 - categorical_accuracy: 0.9359
50560/60000 [========================>.....] - ETA: 15s - loss: 0.2090 - categorical_accuracy: 0.9359
50592/60000 [========================>.....] - ETA: 15s - loss: 0.2089 - categorical_accuracy: 0.9359
50656/60000 [========================>.....] - ETA: 15s - loss: 0.2091 - categorical_accuracy: 0.9359
50688/60000 [========================>.....] - ETA: 14s - loss: 0.2090 - categorical_accuracy: 0.9359
50720/60000 [========================>.....] - ETA: 14s - loss: 0.2089 - categorical_accuracy: 0.9360
50752/60000 [========================>.....] - ETA: 14s - loss: 0.2090 - categorical_accuracy: 0.9359
50784/60000 [========================>.....] - ETA: 14s - loss: 0.2089 - categorical_accuracy: 0.9359
50848/60000 [========================>.....] - ETA: 14s - loss: 0.2088 - categorical_accuracy: 0.9360
50880/60000 [========================>.....] - ETA: 14s - loss: 0.2087 - categorical_accuracy: 0.9360
50912/60000 [========================>.....] - ETA: 14s - loss: 0.2086 - categorical_accuracy: 0.9360
50976/60000 [========================>.....] - ETA: 14s - loss: 0.2086 - categorical_accuracy: 0.9361
51008/60000 [========================>.....] - ETA: 14s - loss: 0.2084 - categorical_accuracy: 0.9361
51072/60000 [========================>.....] - ETA: 14s - loss: 0.2082 - categorical_accuracy: 0.9362
51136/60000 [========================>.....] - ETA: 14s - loss: 0.2080 - categorical_accuracy: 0.9363
51168/60000 [========================>.....] - ETA: 14s - loss: 0.2079 - categorical_accuracy: 0.9363
51232/60000 [========================>.....] - ETA: 14s - loss: 0.2078 - categorical_accuracy: 0.9364
51264/60000 [========================>.....] - ETA: 14s - loss: 0.2077 - categorical_accuracy: 0.9364
51328/60000 [========================>.....] - ETA: 13s - loss: 0.2074 - categorical_accuracy: 0.9365
51360/60000 [========================>.....] - ETA: 13s - loss: 0.2074 - categorical_accuracy: 0.9365
51392/60000 [========================>.....] - ETA: 13s - loss: 0.2074 - categorical_accuracy: 0.9365
51456/60000 [========================>.....] - ETA: 13s - loss: 0.2074 - categorical_accuracy: 0.9365
51520/60000 [========================>.....] - ETA: 13s - loss: 0.2072 - categorical_accuracy: 0.9366
51552/60000 [========================>.....] - ETA: 13s - loss: 0.2071 - categorical_accuracy: 0.9366
51584/60000 [========================>.....] - ETA: 13s - loss: 0.2070 - categorical_accuracy: 0.9366
51616/60000 [========================>.....] - ETA: 13s - loss: 0.2069 - categorical_accuracy: 0.9367
51648/60000 [========================>.....] - ETA: 13s - loss: 0.2070 - categorical_accuracy: 0.9367
51680/60000 [========================>.....] - ETA: 13s - loss: 0.2069 - categorical_accuracy: 0.9367
51712/60000 [========================>.....] - ETA: 13s - loss: 0.2068 - categorical_accuracy: 0.9367
51776/60000 [========================>.....] - ETA: 13s - loss: 0.2067 - categorical_accuracy: 0.9367
51840/60000 [========================>.....] - ETA: 13s - loss: 0.2066 - categorical_accuracy: 0.9368
51872/60000 [========================>.....] - ETA: 13s - loss: 0.2065 - categorical_accuracy: 0.9368
51904/60000 [========================>.....] - ETA: 13s - loss: 0.2063 - categorical_accuracy: 0.9368
51936/60000 [========================>.....] - ETA: 12s - loss: 0.2062 - categorical_accuracy: 0.9369
52000/60000 [=========================>....] - ETA: 12s - loss: 0.2063 - categorical_accuracy: 0.9369
52032/60000 [=========================>....] - ETA: 12s - loss: 0.2062 - categorical_accuracy: 0.9369
52064/60000 [=========================>....] - ETA: 12s - loss: 0.2061 - categorical_accuracy: 0.9370
52128/60000 [=========================>....] - ETA: 12s - loss: 0.2059 - categorical_accuracy: 0.9370
52192/60000 [=========================>....] - ETA: 12s - loss: 0.2057 - categorical_accuracy: 0.9371
52224/60000 [=========================>....] - ETA: 12s - loss: 0.2058 - categorical_accuracy: 0.9371
52256/60000 [=========================>....] - ETA: 12s - loss: 0.2057 - categorical_accuracy: 0.9371
52288/60000 [=========================>....] - ETA: 12s - loss: 0.2057 - categorical_accuracy: 0.9371
52352/60000 [=========================>....] - ETA: 12s - loss: 0.2054 - categorical_accuracy: 0.9372
52416/60000 [=========================>....] - ETA: 12s - loss: 0.2053 - categorical_accuracy: 0.9372
52480/60000 [=========================>....] - ETA: 12s - loss: 0.2052 - categorical_accuracy: 0.9372
52544/60000 [=========================>....] - ETA: 11s - loss: 0.2051 - categorical_accuracy: 0.9372
52608/60000 [=========================>....] - ETA: 11s - loss: 0.2050 - categorical_accuracy: 0.9372
52672/60000 [=========================>....] - ETA: 11s - loss: 0.2049 - categorical_accuracy: 0.9373
52704/60000 [=========================>....] - ETA: 11s - loss: 0.2049 - categorical_accuracy: 0.9373
52736/60000 [=========================>....] - ETA: 11s - loss: 0.2051 - categorical_accuracy: 0.9373
52768/60000 [=========================>....] - ETA: 11s - loss: 0.2051 - categorical_accuracy: 0.9373
52832/60000 [=========================>....] - ETA: 11s - loss: 0.2049 - categorical_accuracy: 0.9373
52864/60000 [=========================>....] - ETA: 11s - loss: 0.2048 - categorical_accuracy: 0.9373
52896/60000 [=========================>....] - ETA: 11s - loss: 0.2047 - categorical_accuracy: 0.9374
52928/60000 [=========================>....] - ETA: 11s - loss: 0.2046 - categorical_accuracy: 0.9374
52960/60000 [=========================>....] - ETA: 11s - loss: 0.2046 - categorical_accuracy: 0.9374
52992/60000 [=========================>....] - ETA: 11s - loss: 0.2045 - categorical_accuracy: 0.9375
53024/60000 [=========================>....] - ETA: 11s - loss: 0.2046 - categorical_accuracy: 0.9374
53056/60000 [=========================>....] - ETA: 11s - loss: 0.2046 - categorical_accuracy: 0.9374
53088/60000 [=========================>....] - ETA: 11s - loss: 0.2045 - categorical_accuracy: 0.9374
53152/60000 [=========================>....] - ETA: 10s - loss: 0.2043 - categorical_accuracy: 0.9375
53184/60000 [=========================>....] - ETA: 10s - loss: 0.2042 - categorical_accuracy: 0.9375
53216/60000 [=========================>....] - ETA: 10s - loss: 0.2041 - categorical_accuracy: 0.9375
53248/60000 [=========================>....] - ETA: 10s - loss: 0.2040 - categorical_accuracy: 0.9376
53280/60000 [=========================>....] - ETA: 10s - loss: 0.2039 - categorical_accuracy: 0.9376
53312/60000 [=========================>....] - ETA: 10s - loss: 0.2038 - categorical_accuracy: 0.9377
53344/60000 [=========================>....] - ETA: 10s - loss: 0.2038 - categorical_accuracy: 0.9377
53376/60000 [=========================>....] - ETA: 10s - loss: 0.2037 - categorical_accuracy: 0.9377
53408/60000 [=========================>....] - ETA: 10s - loss: 0.2036 - categorical_accuracy: 0.9377
53472/60000 [=========================>....] - ETA: 10s - loss: 0.2035 - categorical_accuracy: 0.9377
53504/60000 [=========================>....] - ETA: 10s - loss: 0.2034 - categorical_accuracy: 0.9378
53568/60000 [=========================>....] - ETA: 10s - loss: 0.2032 - categorical_accuracy: 0.9378
53600/60000 [=========================>....] - ETA: 10s - loss: 0.2032 - categorical_accuracy: 0.9379
53632/60000 [=========================>....] - ETA: 10s - loss: 0.2032 - categorical_accuracy: 0.9379
53664/60000 [=========================>....] - ETA: 10s - loss: 0.2031 - categorical_accuracy: 0.9379
53696/60000 [=========================>....] - ETA: 10s - loss: 0.2030 - categorical_accuracy: 0.9379
53728/60000 [=========================>....] - ETA: 10s - loss: 0.2030 - categorical_accuracy: 0.9379
53760/60000 [=========================>....] - ETA: 10s - loss: 0.2029 - categorical_accuracy: 0.9379
53792/60000 [=========================>....] - ETA: 9s - loss: 0.2028 - categorical_accuracy: 0.9379 
53824/60000 [=========================>....] - ETA: 9s - loss: 0.2027 - categorical_accuracy: 0.9380
53856/60000 [=========================>....] - ETA: 9s - loss: 0.2026 - categorical_accuracy: 0.9380
53888/60000 [=========================>....] - ETA: 9s - loss: 0.2025 - categorical_accuracy: 0.9380
53920/60000 [=========================>....] - ETA: 9s - loss: 0.2025 - categorical_accuracy: 0.9381
53984/60000 [=========================>....] - ETA: 9s - loss: 0.2022 - categorical_accuracy: 0.9381
54016/60000 [==========================>...] - ETA: 9s - loss: 0.2021 - categorical_accuracy: 0.9381
54048/60000 [==========================>...] - ETA: 9s - loss: 0.2021 - categorical_accuracy: 0.9381
54080/60000 [==========================>...] - ETA: 9s - loss: 0.2021 - categorical_accuracy: 0.9381
54112/60000 [==========================>...] - ETA: 9s - loss: 0.2022 - categorical_accuracy: 0.9381
54176/60000 [==========================>...] - ETA: 9s - loss: 0.2022 - categorical_accuracy: 0.9381
54208/60000 [==========================>...] - ETA: 9s - loss: 0.2022 - categorical_accuracy: 0.9381
54272/60000 [==========================>...] - ETA: 9s - loss: 0.2021 - categorical_accuracy: 0.9382
54336/60000 [==========================>...] - ETA: 9s - loss: 0.2020 - categorical_accuracy: 0.9382
54400/60000 [==========================>...] - ETA: 8s - loss: 0.2018 - categorical_accuracy: 0.9382
54464/60000 [==========================>...] - ETA: 8s - loss: 0.2017 - categorical_accuracy: 0.9382
54496/60000 [==========================>...] - ETA: 8s - loss: 0.2017 - categorical_accuracy: 0.9382
54528/60000 [==========================>...] - ETA: 8s - loss: 0.2016 - categorical_accuracy: 0.9383
54560/60000 [==========================>...] - ETA: 8s - loss: 0.2016 - categorical_accuracy: 0.9383
54624/60000 [==========================>...] - ETA: 8s - loss: 0.2016 - categorical_accuracy: 0.9383
54656/60000 [==========================>...] - ETA: 8s - loss: 0.2016 - categorical_accuracy: 0.9383
54688/60000 [==========================>...] - ETA: 8s - loss: 0.2015 - categorical_accuracy: 0.9383
54752/60000 [==========================>...] - ETA: 8s - loss: 0.2013 - categorical_accuracy: 0.9384
54816/60000 [==========================>...] - ETA: 8s - loss: 0.2012 - categorical_accuracy: 0.9384
54848/60000 [==========================>...] - ETA: 8s - loss: 0.2011 - categorical_accuracy: 0.9384
54880/60000 [==========================>...] - ETA: 8s - loss: 0.2011 - categorical_accuracy: 0.9384
54912/60000 [==========================>...] - ETA: 8s - loss: 0.2010 - categorical_accuracy: 0.9384
54976/60000 [==========================>...] - ETA: 8s - loss: 0.2008 - categorical_accuracy: 0.9385
55040/60000 [==========================>...] - ETA: 7s - loss: 0.2007 - categorical_accuracy: 0.9385
55104/60000 [==========================>...] - ETA: 7s - loss: 0.2006 - categorical_accuracy: 0.9385
55168/60000 [==========================>...] - ETA: 7s - loss: 0.2005 - categorical_accuracy: 0.9385
55200/60000 [==========================>...] - ETA: 7s - loss: 0.2004 - categorical_accuracy: 0.9386
55232/60000 [==========================>...] - ETA: 7s - loss: 0.2004 - categorical_accuracy: 0.9386
55264/60000 [==========================>...] - ETA: 7s - loss: 0.2003 - categorical_accuracy: 0.9386
55296/60000 [==========================>...] - ETA: 7s - loss: 0.2002 - categorical_accuracy: 0.9386
55360/60000 [==========================>...] - ETA: 7s - loss: 0.2003 - categorical_accuracy: 0.9386
55424/60000 [==========================>...] - ETA: 7s - loss: 0.2003 - categorical_accuracy: 0.9387
55456/60000 [==========================>...] - ETA: 7s - loss: 0.2002 - categorical_accuracy: 0.9387
55488/60000 [==========================>...] - ETA: 7s - loss: 0.2001 - categorical_accuracy: 0.9387
55520/60000 [==========================>...] - ETA: 7s - loss: 0.2000 - categorical_accuracy: 0.9388
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1999 - categorical_accuracy: 0.9388
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1997 - categorical_accuracy: 0.9388
55648/60000 [==========================>...] - ETA: 6s - loss: 0.1997 - categorical_accuracy: 0.9388
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1996 - categorical_accuracy: 0.9389
55712/60000 [==========================>...] - ETA: 6s - loss: 0.1995 - categorical_accuracy: 0.9389
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1994 - categorical_accuracy: 0.9389
55776/60000 [==========================>...] - ETA: 6s - loss: 0.1994 - categorical_accuracy: 0.9389
55808/60000 [==========================>...] - ETA: 6s - loss: 0.1993 - categorical_accuracy: 0.9389
55840/60000 [==========================>...] - ETA: 6s - loss: 0.1992 - categorical_accuracy: 0.9390
55904/60000 [==========================>...] - ETA: 6s - loss: 0.1992 - categorical_accuracy: 0.9389
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1992 - categorical_accuracy: 0.9389
55968/60000 [==========================>...] - ETA: 6s - loss: 0.1991 - categorical_accuracy: 0.9390
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1990 - categorical_accuracy: 0.9390
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1990 - categorical_accuracy: 0.9390
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1988 - categorical_accuracy: 0.9391
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1987 - categorical_accuracy: 0.9391
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1987 - categorical_accuracy: 0.9391
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1986 - categorical_accuracy: 0.9391
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1985 - categorical_accuracy: 0.9392
56288/60000 [===========================>..] - ETA: 5s - loss: 0.1985 - categorical_accuracy: 0.9392
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1984 - categorical_accuracy: 0.9392
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1982 - categorical_accuracy: 0.9393
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1981 - categorical_accuracy: 0.9393
56480/60000 [===========================>..] - ETA: 5s - loss: 0.1979 - categorical_accuracy: 0.9394
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1979 - categorical_accuracy: 0.9394
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1979 - categorical_accuracy: 0.9394
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1978 - categorical_accuracy: 0.9394
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1978 - categorical_accuracy: 0.9394
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1977 - categorical_accuracy: 0.9395
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1977 - categorical_accuracy: 0.9395
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1975 - categorical_accuracy: 0.9395
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1975 - categorical_accuracy: 0.9395
56896/60000 [===========================>..] - ETA: 4s - loss: 0.1974 - categorical_accuracy: 0.9396
56928/60000 [===========================>..] - ETA: 4s - loss: 0.1973 - categorical_accuracy: 0.9396
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1972 - categorical_accuracy: 0.9396
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1971 - categorical_accuracy: 0.9396
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1971 - categorical_accuracy: 0.9397
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1970 - categorical_accuracy: 0.9397
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1969 - categorical_accuracy: 0.9397
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1967 - categorical_accuracy: 0.9398
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1967 - categorical_accuracy: 0.9398
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1967 - categorical_accuracy: 0.9398
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1965 - categorical_accuracy: 0.9398
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1964 - categorical_accuracy: 0.9399
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1963 - categorical_accuracy: 0.9399
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1963 - categorical_accuracy: 0.9399
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1963 - categorical_accuracy: 0.9399
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1962 - categorical_accuracy: 0.9400
57568/60000 [===========================>..] - ETA: 3s - loss: 0.1961 - categorical_accuracy: 0.9400
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1960 - categorical_accuracy: 0.9400
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1959 - categorical_accuracy: 0.9401
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1957 - categorical_accuracy: 0.9401
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1957 - categorical_accuracy: 0.9401
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1956 - categorical_accuracy: 0.9401
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1956 - categorical_accuracy: 0.9401
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1955 - categorical_accuracy: 0.9401
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1954 - categorical_accuracy: 0.9402
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1953 - categorical_accuracy: 0.9402
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1952 - categorical_accuracy: 0.9402
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1951 - categorical_accuracy: 0.9403
58016/60000 [============================>.] - ETA: 3s - loss: 0.1950 - categorical_accuracy: 0.9403
58048/60000 [============================>.] - ETA: 3s - loss: 0.1949 - categorical_accuracy: 0.9403
58080/60000 [============================>.] - ETA: 3s - loss: 0.1949 - categorical_accuracy: 0.9403
58112/60000 [============================>.] - ETA: 3s - loss: 0.1948 - categorical_accuracy: 0.9404
58144/60000 [============================>.] - ETA: 2s - loss: 0.1947 - categorical_accuracy: 0.9404
58208/60000 [============================>.] - ETA: 2s - loss: 0.1948 - categorical_accuracy: 0.9404
58272/60000 [============================>.] - ETA: 2s - loss: 0.1947 - categorical_accuracy: 0.9404
58304/60000 [============================>.] - ETA: 2s - loss: 0.1946 - categorical_accuracy: 0.9405
58336/60000 [============================>.] - ETA: 2s - loss: 0.1946 - categorical_accuracy: 0.9405
58368/60000 [============================>.] - ETA: 2s - loss: 0.1946 - categorical_accuracy: 0.9405
58400/60000 [============================>.] - ETA: 2s - loss: 0.1946 - categorical_accuracy: 0.9405
58464/60000 [============================>.] - ETA: 2s - loss: 0.1944 - categorical_accuracy: 0.9405
58496/60000 [============================>.] - ETA: 2s - loss: 0.1944 - categorical_accuracy: 0.9405
58560/60000 [============================>.] - ETA: 2s - loss: 0.1942 - categorical_accuracy: 0.9406
58624/60000 [============================>.] - ETA: 2s - loss: 0.1941 - categorical_accuracy: 0.9406
58688/60000 [============================>.] - ETA: 2s - loss: 0.1940 - categorical_accuracy: 0.9406
58752/60000 [============================>.] - ETA: 2s - loss: 0.1939 - categorical_accuracy: 0.9407
58816/60000 [============================>.] - ETA: 1s - loss: 0.1940 - categorical_accuracy: 0.9407
58880/60000 [============================>.] - ETA: 1s - loss: 0.1938 - categorical_accuracy: 0.9408
58912/60000 [============================>.] - ETA: 1s - loss: 0.1940 - categorical_accuracy: 0.9407
58944/60000 [============================>.] - ETA: 1s - loss: 0.1940 - categorical_accuracy: 0.9407
58976/60000 [============================>.] - ETA: 1s - loss: 0.1939 - categorical_accuracy: 0.9408
59040/60000 [============================>.] - ETA: 1s - loss: 0.1939 - categorical_accuracy: 0.9408
59104/60000 [============================>.] - ETA: 1s - loss: 0.1938 - categorical_accuracy: 0.9408
59136/60000 [============================>.] - ETA: 1s - loss: 0.1938 - categorical_accuracy: 0.9408
59168/60000 [============================>.] - ETA: 1s - loss: 0.1938 - categorical_accuracy: 0.9407
59232/60000 [============================>.] - ETA: 1s - loss: 0.1936 - categorical_accuracy: 0.9408
59264/60000 [============================>.] - ETA: 1s - loss: 0.1936 - categorical_accuracy: 0.9408
59296/60000 [============================>.] - ETA: 1s - loss: 0.1935 - categorical_accuracy: 0.9408
59328/60000 [============================>.] - ETA: 1s - loss: 0.1935 - categorical_accuracy: 0.9409
59360/60000 [============================>.] - ETA: 1s - loss: 0.1935 - categorical_accuracy: 0.9409
59392/60000 [============================>.] - ETA: 0s - loss: 0.1934 - categorical_accuracy: 0.9409
59424/60000 [============================>.] - ETA: 0s - loss: 0.1933 - categorical_accuracy: 0.9409
59488/60000 [============================>.] - ETA: 0s - loss: 0.1932 - categorical_accuracy: 0.9409
59520/60000 [============================>.] - ETA: 0s - loss: 0.1931 - categorical_accuracy: 0.9409
59584/60000 [============================>.] - ETA: 0s - loss: 0.1931 - categorical_accuracy: 0.9410
59648/60000 [============================>.] - ETA: 0s - loss: 0.1930 - categorical_accuracy: 0.9410
59712/60000 [============================>.] - ETA: 0s - loss: 0.1930 - categorical_accuracy: 0.9410
59744/60000 [============================>.] - ETA: 0s - loss: 0.1929 - categorical_accuracy: 0.9410
59776/60000 [============================>.] - ETA: 0s - loss: 0.1929 - categorical_accuracy: 0.9410
59840/60000 [============================>.] - ETA: 0s - loss: 0.1929 - categorical_accuracy: 0.9410
59904/60000 [============================>.] - ETA: 0s - loss: 0.1927 - categorical_accuracy: 0.9410
59968/60000 [============================>.] - ETA: 0s - loss: 0.1925 - categorical_accuracy: 0.9411
60000/60000 [==============================] - 100s 2ms/step - loss: 0.1925 - categorical_accuracy: 0.9411 - val_loss: 0.0464 - val_categorical_accuracy: 0.9844

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 15s
  224/10000 [..............................] - ETA: 4s 
  384/10000 [>.............................] - ETA: 4s
  576/10000 [>.............................] - ETA: 3s
  768/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 3s
 1344/10000 [===>..........................] - ETA: 2s
 1536/10000 [===>..........................] - ETA: 2s
 1728/10000 [====>.........................] - ETA: 2s
 1920/10000 [====>.........................] - ETA: 2s
 2080/10000 [=====>........................] - ETA: 2s
 2240/10000 [=====>........................] - ETA: 2s
 2400/10000 [======>.......................] - ETA: 2s
 2592/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
 3264/10000 [========>.....................] - ETA: 2s
 3456/10000 [=========>....................] - ETA: 2s
 3648/10000 [=========>....................] - ETA: 2s
 3808/10000 [==========>...................] - ETA: 1s
 4000/10000 [===========>..................] - ETA: 1s
 4160/10000 [===========>..................] - ETA: 1s
 4352/10000 [============>.................] - ETA: 1s
 4544/10000 [============>.................] - ETA: 1s
 4704/10000 [=============>................] - ETA: 1s
 4896/10000 [=============>................] - ETA: 1s
 5056/10000 [==============>...............] - ETA: 1s
 5216/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5568/10000 [===============>..............] - ETA: 1s
 5760/10000 [================>.............] - ETA: 1s
 5952/10000 [================>.............] - ETA: 1s
 6144/10000 [=================>............] - ETA: 1s
 6336/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 1s
 6720/10000 [===================>..........] - ETA: 1s
 6912/10000 [===================>..........] - ETA: 0s
 7072/10000 [====================>.........] - ETA: 0s
 7264/10000 [====================>.........] - ETA: 0s
 7456/10000 [=====================>........] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
 7840/10000 [======================>.......] - ETA: 0s
 8032/10000 [=======================>......] - ETA: 0s
 8224/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9152/10000 [==========================>...] - ETA: 0s
 9344/10000 [===========================>..] - ETA: 0s
 9536/10000 [===========================>..] - ETA: 0s
 9696/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 314us/step
[[3.19320236e-07 1.09672103e-08 1.91400932e-06 ... 9.99996066e-01
  4.65409720e-08 1.31248942e-06]
 [1.90175542e-05 2.04097869e-05 9.99875665e-01 ... 4.76808060e-09
  9.44065232e-06 1.84129778e-09]
 [5.64145921e-06 9.99412060e-01 4.31497720e-05 ... 1.48438092e-04
  5.96657192e-05 6.06731783e-06]
 ...
 [2.65356306e-07 2.43573982e-06 3.51304720e-07 ... 4.65282028e-05
  1.22824385e-05 1.13492795e-04]
 [6.65049229e-05 3.62377705e-07 3.37900104e-07 ... 1.42374859e-06
  6.95338054e-03 9.13977510e-06]
 [3.60986155e-06 1.27533553e-07 4.64704681e-06 ... 4.67096861e-10
  1.95092539e-07 5.52357182e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.046417724947468375, 'accuracy_test:': 0.9843999743461609}

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
[master 57ac7fb] ml_store  && git pull --all
 1 file changed, 1553 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + 29b5ff0...57ac7fb master -> master (forced update)





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
{'loss': 0.5992859676480293, 'loss_history': []}

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
[master d260ba9] ml_store  && git pull --all
 1 file changed, 112 insertions(+)
To github.com:arita37/mlmodels_store.git
   57ac7fb..d260ba9  master -> master





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
[master ec4b5b6] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   d260ba9..ec4b5b6  master -> master





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
 40%|████      | 2/5 [00:16<00:24,  8.19s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.836410624931228, 'learning_rate': 0.05630740844181091, 'min_data_in_leaf': 4, 'num_leaves': 41} and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xc3\xe07\x03h\xd4X\r\x00\x00\x00learning_rateq\x02G?\xac\xd4S\x1b\x8fh>X\x10\x00\x00\x00min_data_in_leafq\x03K\x04X\n\x00\x00\x00num_leavesq\x04K)u.' and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xc3\xe07\x03h\xd4X\r\x00\x00\x00learning_rateq\x02G?\xac\xd4S\x1b\x8fh>X\x10\x00\x00\x00min_data_in_leafq\x03K\x04X\n\x00\x00\x00num_leavesq\x04K)u.' and reward: 0.39
 60%|██████    | 3/5 [00:33<00:21, 10.94s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8383740059165958, 'learning_rate': 0.1870983263161547, 'min_data_in_leaf': 6, 'num_leaves': 49} and reward: 0.3842
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xd3\xf5\xb9'N&X\r\x00\x00\x00learning_rateq\x02G?\xc7\xf2\xd6\x84U\x05.X\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K1u." and reward: 0.3842
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xd3\xf5\xb9'N&X\r\x00\x00\x00learning_rateq\x02G?\xc7\xf2\xd6\x84U\x05.X\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K1u." and reward: 0.3842
 80%|████████  | 4/5 [00:51<00:13, 13.05s/it]Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.7573883042933204, 'learning_rate': 0.0578798384253015, 'min_data_in_leaf': 11, 'num_leaves': 37} and reward: 0.3934
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8<\x86e\xaa\x01\xb3X\r\x00\x00\x00learning_rateq\x02G?\xad\xa2m\x1aD\xd4\x8bX\x10\x00\x00\x00min_data_in_leafq\x03K\x0bX\n\x00\x00\x00num_leavesq\x04K%u.' and reward: 0.3934
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8<\x86e\xaa\x01\xb3X\r\x00\x00\x00learning_rateq\x02G?\xad\xa2m\x1aD\xd4\x8bX\x10\x00\x00\x00min_data_in_leafq\x03K\x0bX\n\x00\x00\x00num_leavesq\x04K%u.' and reward: 0.3934
100%|██████████| 5/5 [01:08<00:00, 14.17s/it]100%|██████████| 5/5 [01:08<00:00, 13.70s/it]
Saving dataset/models/LightGBMClassifier/trial_4_model.pkl
Finished Task with config: {'feature_fraction': 0.882769422403265, 'learning_rate': 0.03134612719929234, 'min_data_in_leaf': 18, 'num_leaves': 31} and reward: 0.3928
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec?\xa5\xa8\xe4/\xc8X\r\x00\x00\x00learning_rateq\x02G?\xa0\x0c\x99~Z\xbc%X\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K\x1fu.' and reward: 0.3928
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec?\xa5\xa8\xe4/\xc8X\r\x00\x00\x00learning_rateq\x02G?\xa0\x0c\x99~Z\xbc%X\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K\x1fu.' and reward: 0.3928
Time for Gradient Boosting hyperparameter optimization: 84.0520396232605
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.7573883042933204, 'learning_rate': 0.0578798384253015, 'min_data_in_leaf': 11, 'num_leaves': 37}
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
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3868
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3868
 40%|████      | 2/5 [00:54<01:22, 27.36s/it] 40%|████      | 2/5 [00:54<01:22, 27.36s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.41987053767945476, 'embedding_size_factor': 1.1297824276777326, 'layers.choice': 1, 'learning_rate': 0.000151382609152199, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.00013824588954361938} and reward: 0.375
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xda\xdf(\xac\xf8\xc7\xc9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\x13\x96\xbd\'\x8a\xa3X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?#\xd7\x8e\xb6\x03\x06\xf0X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?"\x1e\xc3&\x17D\xb4u.' and reward: 0.375
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xda\xdf(\xac\xf8\xc7\xc9X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf2\x13\x96\xbd\'\x8a\xa3X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?#\xd7\x8e\xb6\x03\x06\xf0X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?"\x1e\xc3&\x17D\xb4u.' and reward: 0.375
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 152.96902346611023
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_4_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -121.22s of remaining time.
Ensemble size: 96
Ensemble weights: 
[0.1875     0.125      0.13541667 0.11458333 0.41666667 0.
 0.02083333]
	0.3966	 = Validation accuracy score
	1.69s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 242.97s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_4_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f940d083940>

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
[master f9e8a43] ml_store  && git pull --all
 1 file changed, 204 insertions(+)
To github.com:arita37/mlmodels_store.git
 + e1139bc...f9e8a43 master -> master (forced update)





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
