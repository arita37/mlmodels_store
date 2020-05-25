
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
[master 9b88643] ml_store  && git pull --all
 2 files changed, 75 insertions(+), 9900 deletions(-)
 rewrite log_testall/log_testall.py (99%)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + 4f1b819...9b88643 master -> master (forced update)





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
[master b4de3ba] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   9b88643..b4de3ba  master -> master





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
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         32          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-25 12:15:49.662636: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-25 12:15:49.677578: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-25 12:15:49.677784: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561345d63870 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 12:15:49.677804: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 178
Trainable params: 178
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2525 - val_binary_crossentropy: 0.7506

  #### metrics   #################################################### 
{'MSE': 0.25109322545638485}

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
sparse_feature_2 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
Total params: 178
Trainable params: 178
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 453
Trainable params: 453
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2625 - binary_crossentropy: 0.7198500/500 [==============================] - 1s 1ms/sample - loss: 0.2523 - binary_crossentropy: 0.7244 - val_loss: 0.2693 - val_binary_crossentropy: 0.7598

  #### metrics   #################################################### 
{'MSE': 0.2605806705979778}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 453
Trainable params: 453
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 647
Trainable params: 647
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2625 - binary_crossentropy: 0.9808500/500 [==============================] - 1s 2ms/sample - loss: 0.2617 - binary_crossentropy: 0.9786 - val_loss: 0.2646 - val_binary_crossentropy: 1.0621

  #### metrics   #################################################### 
{'MSE': 0.26282096104204605}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 647
Trainable params: 647
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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 403
Trainable params: 403
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.4900 - binary_crossentropy: 7.5582500/500 [==============================] - 1s 3ms/sample - loss: 0.4920 - binary_crossentropy: 7.5891 - val_loss: 0.5340 - val_binary_crossentropy: 8.2369

  #### metrics   #################################################### 
{'MSE': 0.513}

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
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
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
Total params: 148
Trainable params: 148
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2877 - binary_crossentropy: 0.7749500/500 [==============================] - 2s 3ms/sample - loss: 0.2825 - binary_crossentropy: 0.7627 - val_loss: 0.2657 - val_binary_crossentropy: 0.7269

  #### metrics   #################################################### 
{'MSE': 0.27327626730884674}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-25 12:17:07.946687: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:07.949073: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:07.955924: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 12:17:07.967686: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 12:17:07.969847: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 12:17:07.971405: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:07.973612: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2489 - val_binary_crossentropy: 0.6909
2020-05-25 12:17:09.249085: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:09.250818: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:09.255455: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 12:17:09.264269: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 12:17:09.265849: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 12:17:09.267322: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:09.268810: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24839563120466104}

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
2020-05-25 12:17:33.440607: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:33.442065: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:33.446354: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 12:17:33.452408: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 12:17:33.453469: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 12:17:33.454385: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:33.455247: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2502 - val_binary_crossentropy: 0.6936
2020-05-25 12:17:34.974095: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:34.975291: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:34.978074: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 12:17:34.983161: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 12:17:34.984160: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 12:17:34.985078: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:17:34.986185: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2502318732151381}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-25 12:18:09.822535: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:18:09.827959: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:18:09.846063: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 12:18:09.873716: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 12:18:09.878443: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 12:18:09.882545: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:18:09.886510: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.3364 - binary_crossentropy: 0.8675 - val_loss: 0.2576 - val_binary_crossentropy: 0.7086
2020-05-25 12:18:12.168090: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:18:12.172758: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:18:12.185095: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 12:18:12.210390: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 12:18:12.215919: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 12:18:12.219702: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 12:18:12.224019: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.28730306704916764}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 720
Trainable params: 720
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2355 - binary_crossentropy: 0.6667500/500 [==============================] - 4s 9ms/sample - loss: 0.2712 - binary_crossentropy: 0.7427 - val_loss: 0.2831 - val_binary_crossentropy: 0.7671

  #### metrics   #################################################### 
{'MSE': 0.276034574659431}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         4           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
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
Total params: 720
Trainable params: 720
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         12          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         4           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_5[0][0]           
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
Total params: 254
Trainable params: 254
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2685 - binary_crossentropy: 0.7363500/500 [==============================] - 5s 9ms/sample - loss: 0.2673 - binary_crossentropy: 0.7326 - val_loss: 0.2778 - val_binary_crossentropy: 0.7538

  #### metrics   #################################################### 
{'MSE': 0.2700651135261797}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         12          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         4           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_5[0][0]           
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
Total params: 254
Trainable params: 254
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 1,874
Trainable params: 1,874
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3093 - binary_crossentropy: 0.8450500/500 [==============================] - 5s 9ms/sample - loss: 0.3022 - binary_crossentropy: 0.8285 - val_loss: 0.2928 - val_binary_crossentropy: 0.8532

  #### metrics   #################################################### 
{'MSE': 0.296708713358641}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 1,874
Trainable params: 1,874
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
regionsequence_mean (InputLayer [(None, 4)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2581 - binary_crossentropy: 0.7095500/500 [==============================] - 6s 12ms/sample - loss: 0.2584 - binary_crossentropy: 0.7100 - val_loss: 0.2572 - val_binary_crossentropy: 0.7075

  #### metrics   #################################################### 
{'MSE': 0.2575140449191917}

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
regionsequence_mean (InputLayer [(None, 4)]          0                                            
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
region_10sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,437
Trainable params: 1,437
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2694 - binary_crossentropy: 0.8647500/500 [==============================] - 6s 11ms/sample - loss: 0.2576 - binary_crossentropy: 0.7347 - val_loss: 0.2505 - val_binary_crossentropy: 0.6941

  #### metrics   #################################################### 
{'MSE': 0.25125978481500844}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,437
Trainable params: 1,437
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
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
Total params: 3,086
Trainable params: 3,006
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2573 - binary_crossentropy: 0.8400500/500 [==============================] - 7s 13ms/sample - loss: 0.2606 - binary_crossentropy: 0.8717 - val_loss: 0.2588 - val_binary_crossentropy: 0.9194

  #### metrics   #################################################### 
{'MSE': 0.25798682831929104}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         32          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_10[0][0]                    
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
[master 8c772d9] ml_store  && git pull --all
 1 file changed, 4946 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + fdbd843...8c772d9 master -> master (forced update)





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
[master cde44fb] ml_store  && git pull --all
 1 file changed, 50 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   8c772d9..cde44fb  master -> master





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
[master 0ace189] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   cde44fb..0ace189  master -> master





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
[master 0587cd5] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   0ace189..0587cd5  master -> master





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
[master fcae46c] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   0587cd5..fcae46c  master -> master





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
[master aebde57] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   fcae46c..aebde57  master -> master





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
[master 3eda2a7] ml_store  && git pull --all
 1 file changed, 43 insertions(+)
To github.com:arita37/mlmodels_store.git
   aebde57..3eda2a7  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2400256/17464789 [===>..........................] - ETA: 0s
 7389184/17464789 [===========>..................] - ETA: 0s
13434880/17464789 [======================>.......] - ETA: 0s
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
2020-05-25 12:28:06.221201: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-25 12:28:06.225810: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-25 12:28:06.225962: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a7ad5dd060 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 12:28:06.225978: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.2526 - accuracy: 0.5270
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6206 - accuracy: 0.5030
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7024 - accuracy: 0.4977 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7126 - accuracy: 0.4970
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7561 - accuracy: 0.4942
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7039 - accuracy: 0.4976
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6973 - accuracy: 0.4980
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7092 - accuracy: 0.4972
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7157 - accuracy: 0.4968
11000/25000 [============>.................] - ETA: 4s - loss: 7.7433 - accuracy: 0.4950
12000/25000 [=============>................] - ETA: 4s - loss: 7.7535 - accuracy: 0.4943
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7303 - accuracy: 0.4958
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7301 - accuracy: 0.4959
15000/25000 [=================>............] - ETA: 3s - loss: 7.7228 - accuracy: 0.4963
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7002 - accuracy: 0.4978
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6892 - accuracy: 0.4985
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6844 - accuracy: 0.4988
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7027 - accuracy: 0.4976
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6885 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6861 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6840 - accuracy: 0.4989
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6839 - accuracy: 0.4989
25000/25000 [==============================] - 10s 391us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f01e04ca9b0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f01b9d8d828> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7280 - accuracy: 0.4960 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7280 - accuracy: 0.4960
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7510 - accuracy: 0.4945
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7586 - accuracy: 0.4940
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7177 - accuracy: 0.4967
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7477 - accuracy: 0.4947
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7510 - accuracy: 0.4945
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7876 - accuracy: 0.4921
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7632 - accuracy: 0.4937
11000/25000 [============>.................] - ETA: 4s - loss: 7.7057 - accuracy: 0.4975
12000/25000 [=============>................] - ETA: 4s - loss: 7.7343 - accuracy: 0.4956
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7032 - accuracy: 0.4976
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7126 - accuracy: 0.4970
15000/25000 [=================>............] - ETA: 3s - loss: 7.7433 - accuracy: 0.4950
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7270 - accuracy: 0.4961
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7307 - accuracy: 0.4958
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7169 - accuracy: 0.4967
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6981 - accuracy: 0.4979
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6804 - accuracy: 0.4991
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6790 - accuracy: 0.4992
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6834 - accuracy: 0.4989
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6773 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6653 - accuracy: 0.5001
25000/25000 [==============================] - 9s 374us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.9886 - accuracy: 0.4790
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8430 - accuracy: 0.4885 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8506 - accuracy: 0.4880
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8430 - accuracy: 0.4885
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8108 - accuracy: 0.4906
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7663 - accuracy: 0.4935
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7148 - accuracy: 0.4969
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7337 - accuracy: 0.4956
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7620 - accuracy: 0.4938
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7663 - accuracy: 0.4935
11000/25000 [============>.................] - ETA: 4s - loss: 7.7544 - accuracy: 0.4943
12000/25000 [=============>................] - ETA: 4s - loss: 7.7369 - accuracy: 0.4954
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7209 - accuracy: 0.4965
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7214 - accuracy: 0.4964
15000/25000 [=================>............] - ETA: 3s - loss: 7.7024 - accuracy: 0.4977
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6724 - accuracy: 0.4996
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6441 - accuracy: 0.5015
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6283 - accuracy: 0.5025
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6352 - accuracy: 0.5021
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6429 - accuracy: 0.5016
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6454 - accuracy: 0.5014
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6729 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6846 - accuracy: 0.4988
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6826 - accuracy: 0.4990
25000/25000 [==============================] - 10s 381us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master 894138e] ml_store  && git pull --all
 1 file changed, 316 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 86b20eb...894138e master -> master (forced update)





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

13/13 [==============================] - 2s 116ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 6ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
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
[master 56e61a4] ml_store  && git pull --all
 1 file changed, 124 insertions(+)
To github.com:arita37/mlmodels_store.git
   894138e..56e61a4  master -> master





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
 3850240/11490434 [=========>....................] - ETA: 0s
11091968/11490434 [===========================>..] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:29 - loss: 2.3473 - categorical_accuracy: 0.0000e+00
   64/60000 [..............................] - ETA: 4:44 - loss: 2.3312 - categorical_accuracy: 0.0625    
   96/60000 [..............................] - ETA: 3:48 - loss: 2.2879 - categorical_accuracy: 0.1562
  128/60000 [..............................] - ETA: 3:20 - loss: 2.2610 - categorical_accuracy: 0.1719
  160/60000 [..............................] - ETA: 3:02 - loss: 2.2447 - categorical_accuracy: 0.2000
  192/60000 [..............................] - ETA: 2:49 - loss: 2.2123 - categorical_accuracy: 0.2240
  224/60000 [..............................] - ETA: 2:40 - loss: 2.1729 - categorical_accuracy: 0.2500
  256/60000 [..............................] - ETA: 2:34 - loss: 2.1492 - categorical_accuracy: 0.2578
  288/60000 [..............................] - ETA: 2:29 - loss: 2.1019 - categorical_accuracy: 0.2812
  320/60000 [..............................] - ETA: 2:25 - loss: 2.0631 - categorical_accuracy: 0.3000
  352/60000 [..............................] - ETA: 2:21 - loss: 2.0334 - categorical_accuracy: 0.3182
  384/60000 [..............................] - ETA: 2:19 - loss: 2.0217 - categorical_accuracy: 0.3281
  416/60000 [..............................] - ETA: 2:17 - loss: 1.9963 - categorical_accuracy: 0.3269
  448/60000 [..............................] - ETA: 2:15 - loss: 1.9621 - categorical_accuracy: 0.3371
  480/60000 [..............................] - ETA: 2:13 - loss: 1.9407 - categorical_accuracy: 0.3396
  512/60000 [..............................] - ETA: 2:11 - loss: 1.9160 - categorical_accuracy: 0.3535
  544/60000 [..............................] - ETA: 2:09 - loss: 1.8819 - categorical_accuracy: 0.3621
  576/60000 [..............................] - ETA: 2:08 - loss: 1.8850 - categorical_accuracy: 0.3715
  608/60000 [..............................] - ETA: 2:07 - loss: 1.8732 - categorical_accuracy: 0.3783
  640/60000 [..............................] - ETA: 2:06 - loss: 1.8487 - categorical_accuracy: 0.3891
  672/60000 [..............................] - ETA: 2:06 - loss: 1.8083 - categorical_accuracy: 0.3988
  704/60000 [..............................] - ETA: 2:05 - loss: 1.7692 - categorical_accuracy: 0.4148
  736/60000 [..............................] - ETA: 2:04 - loss: 1.7413 - categorical_accuracy: 0.4239
  768/60000 [..............................] - ETA: 2:04 - loss: 1.7174 - categorical_accuracy: 0.4323
  800/60000 [..............................] - ETA: 2:03 - loss: 1.6939 - categorical_accuracy: 0.4375
  832/60000 [..............................] - ETA: 2:02 - loss: 1.6785 - categorical_accuracy: 0.4435
  864/60000 [..............................] - ETA: 2:02 - loss: 1.6453 - categorical_accuracy: 0.4583
  896/60000 [..............................] - ETA: 2:02 - loss: 1.6188 - categorical_accuracy: 0.4654
  928/60000 [..............................] - ETA: 2:01 - loss: 1.6167 - categorical_accuracy: 0.4677
  960/60000 [..............................] - ETA: 2:01 - loss: 1.5970 - categorical_accuracy: 0.4750
  992/60000 [..............................] - ETA: 2:00 - loss: 1.5770 - categorical_accuracy: 0.4819
 1024/60000 [..............................] - ETA: 2:00 - loss: 1.5562 - categorical_accuracy: 0.4863
 1056/60000 [..............................] - ETA: 1:59 - loss: 1.5367 - categorical_accuracy: 0.4943
 1088/60000 [..............................] - ETA: 1:59 - loss: 1.5123 - categorical_accuracy: 0.5000
 1120/60000 [..............................] - ETA: 1:59 - loss: 1.4886 - categorical_accuracy: 0.5080
 1152/60000 [..............................] - ETA: 1:58 - loss: 1.4794 - categorical_accuracy: 0.5087
 1184/60000 [..............................] - ETA: 1:58 - loss: 1.4579 - categorical_accuracy: 0.5144
 1216/60000 [..............................] - ETA: 1:58 - loss: 1.4355 - categorical_accuracy: 0.5214
 1248/60000 [..............................] - ETA: 1:57 - loss: 1.4210 - categorical_accuracy: 0.5264
 1280/60000 [..............................] - ETA: 1:57 - loss: 1.4033 - categorical_accuracy: 0.5312
 1312/60000 [..............................] - ETA: 1:57 - loss: 1.3882 - categorical_accuracy: 0.5351
 1344/60000 [..............................] - ETA: 1:57 - loss: 1.3703 - categorical_accuracy: 0.5409
 1376/60000 [..............................] - ETA: 1:57 - loss: 1.3628 - categorical_accuracy: 0.5458
 1408/60000 [..............................] - ETA: 1:56 - loss: 1.3573 - categorical_accuracy: 0.5476
 1440/60000 [..............................] - ETA: 1:56 - loss: 1.3396 - categorical_accuracy: 0.5535
 1472/60000 [..............................] - ETA: 1:56 - loss: 1.3285 - categorical_accuracy: 0.5591
 1504/60000 [..............................] - ETA: 1:56 - loss: 1.3144 - categorical_accuracy: 0.5625
 1536/60000 [..............................] - ETA: 1:56 - loss: 1.2957 - categorical_accuracy: 0.5697
 1568/60000 [..............................] - ETA: 1:55 - loss: 1.2760 - categorical_accuracy: 0.5765
 1600/60000 [..............................] - ETA: 1:55 - loss: 1.2643 - categorical_accuracy: 0.5794
 1632/60000 [..............................] - ETA: 1:55 - loss: 1.2485 - categorical_accuracy: 0.5839
 1664/60000 [..............................] - ETA: 1:55 - loss: 1.2483 - categorical_accuracy: 0.5853
 1696/60000 [..............................] - ETA: 1:55 - loss: 1.2387 - categorical_accuracy: 0.5884
 1728/60000 [..............................] - ETA: 1:54 - loss: 1.2262 - categorical_accuracy: 0.5914
 1760/60000 [..............................] - ETA: 1:54 - loss: 1.2129 - categorical_accuracy: 0.5966
 1792/60000 [..............................] - ETA: 1:54 - loss: 1.2076 - categorical_accuracy: 0.5988
 1824/60000 [..............................] - ETA: 1:54 - loss: 1.1971 - categorical_accuracy: 0.6036
 1856/60000 [..............................] - ETA: 1:54 - loss: 1.1883 - categorical_accuracy: 0.6067
 1888/60000 [..............................] - ETA: 1:54 - loss: 1.1767 - categorical_accuracy: 0.6112
 1920/60000 [..............................] - ETA: 1:54 - loss: 1.1644 - categorical_accuracy: 0.6151
 1952/60000 [..............................] - ETA: 1:53 - loss: 1.1499 - categorical_accuracy: 0.6204
 1984/60000 [..............................] - ETA: 1:53 - loss: 1.1476 - categorical_accuracy: 0.6225
 2016/60000 [>.............................] - ETA: 1:53 - loss: 1.1401 - categorical_accuracy: 0.6250
 2048/60000 [>.............................] - ETA: 1:53 - loss: 1.1399 - categorical_accuracy: 0.6255
 2080/60000 [>.............................] - ETA: 1:53 - loss: 1.1325 - categorical_accuracy: 0.6274
 2112/60000 [>.............................] - ETA: 1:53 - loss: 1.1202 - categorical_accuracy: 0.6321
 2144/60000 [>.............................] - ETA: 1:53 - loss: 1.1103 - categorical_accuracy: 0.6348
 2176/60000 [>.............................] - ETA: 1:53 - loss: 1.1062 - categorical_accuracy: 0.6356
 2208/60000 [>.............................] - ETA: 1:52 - loss: 1.0977 - categorical_accuracy: 0.6381
 2240/60000 [>.............................] - ETA: 1:52 - loss: 1.0921 - categorical_accuracy: 0.6402
 2272/60000 [>.............................] - ETA: 1:52 - loss: 1.0823 - categorical_accuracy: 0.6439
 2304/60000 [>.............................] - ETA: 1:52 - loss: 1.0702 - categorical_accuracy: 0.6484
 2336/60000 [>.............................] - ETA: 1:52 - loss: 1.0671 - categorical_accuracy: 0.6498
 2368/60000 [>.............................] - ETA: 1:52 - loss: 1.0590 - categorical_accuracy: 0.6529
 2400/60000 [>.............................] - ETA: 1:52 - loss: 1.0491 - categorical_accuracy: 0.6558
 2432/60000 [>.............................] - ETA: 1:52 - loss: 1.0435 - categorical_accuracy: 0.6575
 2464/60000 [>.............................] - ETA: 1:52 - loss: 1.0353 - categorical_accuracy: 0.6599
 2496/60000 [>.............................] - ETA: 1:52 - loss: 1.0288 - categorical_accuracy: 0.6623
 2528/60000 [>.............................] - ETA: 1:51 - loss: 1.0226 - categorical_accuracy: 0.6642
 2560/60000 [>.............................] - ETA: 1:51 - loss: 1.0147 - categorical_accuracy: 0.6668
 2592/60000 [>.............................] - ETA: 1:51 - loss: 1.0071 - categorical_accuracy: 0.6694
 2624/60000 [>.............................] - ETA: 1:51 - loss: 1.0019 - categorical_accuracy: 0.6711
 2656/60000 [>.............................] - ETA: 1:51 - loss: 0.9953 - categorical_accuracy: 0.6736
 2688/60000 [>.............................] - ETA: 1:51 - loss: 0.9858 - categorical_accuracy: 0.6771
 2720/60000 [>.............................] - ETA: 1:50 - loss: 0.9803 - categorical_accuracy: 0.6783
 2752/60000 [>.............................] - ETA: 1:50 - loss: 0.9808 - categorical_accuracy: 0.6788
 2784/60000 [>.............................] - ETA: 1:50 - loss: 0.9762 - categorical_accuracy: 0.6800
 2816/60000 [>.............................] - ETA: 1:50 - loss: 0.9707 - categorical_accuracy: 0.6818
 2848/60000 [>.............................] - ETA: 1:50 - loss: 0.9649 - categorical_accuracy: 0.6847
 2880/60000 [>.............................] - ETA: 1:50 - loss: 0.9592 - categorical_accuracy: 0.6868
 2912/60000 [>.............................] - ETA: 1:50 - loss: 0.9511 - categorical_accuracy: 0.6896
 2944/60000 [>.............................] - ETA: 1:50 - loss: 0.9434 - categorical_accuracy: 0.6923
 2976/60000 [>.............................] - ETA: 1:49 - loss: 0.9417 - categorical_accuracy: 0.6935
 3008/60000 [>.............................] - ETA: 1:49 - loss: 0.9405 - categorical_accuracy: 0.6945
 3040/60000 [>.............................] - ETA: 1:49 - loss: 0.9369 - categorical_accuracy: 0.6961
 3072/60000 [>.............................] - ETA: 1:49 - loss: 0.9341 - categorical_accuracy: 0.6963
 3104/60000 [>.............................] - ETA: 1:49 - loss: 0.9327 - categorical_accuracy: 0.6978
 3136/60000 [>.............................] - ETA: 1:49 - loss: 0.9253 - categorical_accuracy: 0.7006
 3168/60000 [>.............................] - ETA: 1:49 - loss: 0.9206 - categorical_accuracy: 0.7020
 3200/60000 [>.............................] - ETA: 1:49 - loss: 0.9151 - categorical_accuracy: 0.7047
 3232/60000 [>.............................] - ETA: 1:49 - loss: 0.9096 - categorical_accuracy: 0.7058
 3264/60000 [>.............................] - ETA: 1:49 - loss: 0.9033 - categorical_accuracy: 0.7080
 3296/60000 [>.............................] - ETA: 1:49 - loss: 0.8974 - categorical_accuracy: 0.7096
 3328/60000 [>.............................] - ETA: 1:49 - loss: 0.8911 - categorical_accuracy: 0.7118
 3360/60000 [>.............................] - ETA: 1:49 - loss: 0.8844 - categorical_accuracy: 0.7140
 3392/60000 [>.............................] - ETA: 1:49 - loss: 0.8787 - categorical_accuracy: 0.7158
 3424/60000 [>.............................] - ETA: 1:48 - loss: 0.8742 - categorical_accuracy: 0.7176
 3456/60000 [>.............................] - ETA: 1:48 - loss: 0.8694 - categorical_accuracy: 0.7190
 3488/60000 [>.............................] - ETA: 1:48 - loss: 0.8648 - categorical_accuracy: 0.7205
 3520/60000 [>.............................] - ETA: 1:48 - loss: 0.8594 - categorical_accuracy: 0.7224
 3552/60000 [>.............................] - ETA: 1:48 - loss: 0.8562 - categorical_accuracy: 0.7235
 3584/60000 [>.............................] - ETA: 1:48 - loss: 0.8521 - categorical_accuracy: 0.7252
 3616/60000 [>.............................] - ETA: 1:48 - loss: 0.8466 - categorical_accuracy: 0.7268
 3648/60000 [>.............................] - ETA: 1:48 - loss: 0.8408 - categorical_accuracy: 0.7289
 3680/60000 [>.............................] - ETA: 1:48 - loss: 0.8391 - categorical_accuracy: 0.7291
 3712/60000 [>.............................] - ETA: 1:48 - loss: 0.8349 - categorical_accuracy: 0.7306
 3744/60000 [>.............................] - ETA: 1:47 - loss: 0.8313 - categorical_accuracy: 0.7313
 3776/60000 [>.............................] - ETA: 1:47 - loss: 0.8277 - categorical_accuracy: 0.7325
 3808/60000 [>.............................] - ETA: 1:47 - loss: 0.8252 - categorical_accuracy: 0.7340
 3840/60000 [>.............................] - ETA: 1:47 - loss: 0.8226 - categorical_accuracy: 0.7349
 3872/60000 [>.............................] - ETA: 1:47 - loss: 0.8178 - categorical_accuracy: 0.7368
 3904/60000 [>.............................] - ETA: 1:47 - loss: 0.8151 - categorical_accuracy: 0.7380
 3936/60000 [>.............................] - ETA: 1:47 - loss: 0.8114 - categorical_accuracy: 0.7393
 3968/60000 [>.............................] - ETA: 1:47 - loss: 0.8071 - categorical_accuracy: 0.7409
 4000/60000 [=>............................] - ETA: 1:47 - loss: 0.8048 - categorical_accuracy: 0.7418
 4032/60000 [=>............................] - ETA: 1:47 - loss: 0.8004 - categorical_accuracy: 0.7431
 4064/60000 [=>............................] - ETA: 1:46 - loss: 0.7964 - categorical_accuracy: 0.7443
 4096/60000 [=>............................] - ETA: 1:46 - loss: 0.7934 - categorical_accuracy: 0.7456
 4128/60000 [=>............................] - ETA: 1:46 - loss: 0.7910 - categorical_accuracy: 0.7459
 4160/60000 [=>............................] - ETA: 1:46 - loss: 0.7884 - categorical_accuracy: 0.7474
 4192/60000 [=>............................] - ETA: 1:46 - loss: 0.7848 - categorical_accuracy: 0.7483
 4224/60000 [=>............................] - ETA: 1:46 - loss: 0.7799 - categorical_accuracy: 0.7500
 4256/60000 [=>............................] - ETA: 1:46 - loss: 0.7772 - categorical_accuracy: 0.7507
 4288/60000 [=>............................] - ETA: 1:46 - loss: 0.7743 - categorical_accuracy: 0.7519
 4320/60000 [=>............................] - ETA: 1:46 - loss: 0.7702 - categorical_accuracy: 0.7532
 4352/60000 [=>............................] - ETA: 1:46 - loss: 0.7684 - categorical_accuracy: 0.7541
 4384/60000 [=>............................] - ETA: 1:45 - loss: 0.7662 - categorical_accuracy: 0.7552
 4416/60000 [=>............................] - ETA: 1:45 - loss: 0.7636 - categorical_accuracy: 0.7557
 4448/60000 [=>............................] - ETA: 1:45 - loss: 0.7598 - categorical_accuracy: 0.7567
 4480/60000 [=>............................] - ETA: 1:45 - loss: 0.7573 - categorical_accuracy: 0.7576
 4512/60000 [=>............................] - ETA: 1:45 - loss: 0.7531 - categorical_accuracy: 0.7591
 4544/60000 [=>............................] - ETA: 1:45 - loss: 0.7497 - categorical_accuracy: 0.7603
 4576/60000 [=>............................] - ETA: 1:45 - loss: 0.7468 - categorical_accuracy: 0.7614
 4608/60000 [=>............................] - ETA: 1:45 - loss: 0.7431 - categorical_accuracy: 0.7628
 4640/60000 [=>............................] - ETA: 1:45 - loss: 0.7407 - categorical_accuracy: 0.7636
 4672/60000 [=>............................] - ETA: 1:45 - loss: 0.7366 - categorical_accuracy: 0.7650
 4704/60000 [=>............................] - ETA: 1:45 - loss: 0.7346 - categorical_accuracy: 0.7657
 4736/60000 [=>............................] - ETA: 1:45 - loss: 0.7326 - categorical_accuracy: 0.7665
 4768/60000 [=>............................] - ETA: 1:45 - loss: 0.7304 - categorical_accuracy: 0.7672
 4800/60000 [=>............................] - ETA: 1:45 - loss: 0.7276 - categorical_accuracy: 0.7681
 4832/60000 [=>............................] - ETA: 1:44 - loss: 0.7258 - categorical_accuracy: 0.7686
 4864/60000 [=>............................] - ETA: 1:44 - loss: 0.7225 - categorical_accuracy: 0.7693
 4896/60000 [=>............................] - ETA: 1:44 - loss: 0.7185 - categorical_accuracy: 0.7706
 4928/60000 [=>............................] - ETA: 1:44 - loss: 0.7149 - categorical_accuracy: 0.7719
 4960/60000 [=>............................] - ETA: 1:44 - loss: 0.7114 - categorical_accuracy: 0.7730
 4992/60000 [=>............................] - ETA: 1:44 - loss: 0.7110 - categorical_accuracy: 0.7736
 5024/60000 [=>............................] - ETA: 1:44 - loss: 0.7095 - categorical_accuracy: 0.7745
 5056/60000 [=>............................] - ETA: 1:44 - loss: 0.7071 - categorical_accuracy: 0.7755
 5088/60000 [=>............................] - ETA: 1:43 - loss: 0.7037 - categorical_accuracy: 0.7767
 5120/60000 [=>............................] - ETA: 1:43 - loss: 0.7007 - categorical_accuracy: 0.7779
 5152/60000 [=>............................] - ETA: 1:43 - loss: 0.6976 - categorical_accuracy: 0.7789
 5184/60000 [=>............................] - ETA: 1:43 - loss: 0.6941 - categorical_accuracy: 0.7799
 5216/60000 [=>............................] - ETA: 1:43 - loss: 0.6932 - categorical_accuracy: 0.7807
 5248/60000 [=>............................] - ETA: 1:43 - loss: 0.6917 - categorical_accuracy: 0.7814
 5280/60000 [=>............................] - ETA: 1:43 - loss: 0.6907 - categorical_accuracy: 0.7818
 5312/60000 [=>............................] - ETA: 1:43 - loss: 0.6890 - categorical_accuracy: 0.7822
 5344/60000 [=>............................] - ETA: 1:43 - loss: 0.6870 - categorical_accuracy: 0.7829
 5376/60000 [=>............................] - ETA: 1:43 - loss: 0.6849 - categorical_accuracy: 0.7835
 5408/60000 [=>............................] - ETA: 1:43 - loss: 0.6829 - categorical_accuracy: 0.7838
 5440/60000 [=>............................] - ETA: 1:43 - loss: 0.6809 - categorical_accuracy: 0.7846
 5472/60000 [=>............................] - ETA: 1:42 - loss: 0.6783 - categorical_accuracy: 0.7855
 5504/60000 [=>............................] - ETA: 1:42 - loss: 0.6766 - categorical_accuracy: 0.7863
 5536/60000 [=>............................] - ETA: 1:42 - loss: 0.6738 - categorical_accuracy: 0.7872
 5568/60000 [=>............................] - ETA: 1:42 - loss: 0.6706 - categorical_accuracy: 0.7884
 5600/60000 [=>............................] - ETA: 1:42 - loss: 0.6692 - categorical_accuracy: 0.7884
 5632/60000 [=>............................] - ETA: 1:42 - loss: 0.6674 - categorical_accuracy: 0.7889
 5664/60000 [=>............................] - ETA: 1:42 - loss: 0.6658 - categorical_accuracy: 0.7892
 5696/60000 [=>............................] - ETA: 1:42 - loss: 0.6645 - categorical_accuracy: 0.7899
 5728/60000 [=>............................] - ETA: 1:42 - loss: 0.6618 - categorical_accuracy: 0.7909
 5760/60000 [=>............................] - ETA: 1:42 - loss: 0.6588 - categorical_accuracy: 0.7918
 5792/60000 [=>............................] - ETA: 1:42 - loss: 0.6562 - categorical_accuracy: 0.7925
 5824/60000 [=>............................] - ETA: 1:41 - loss: 0.6544 - categorical_accuracy: 0.7931
 5856/60000 [=>............................] - ETA: 1:41 - loss: 0.6514 - categorical_accuracy: 0.7941
 5888/60000 [=>............................] - ETA: 1:41 - loss: 0.6495 - categorical_accuracy: 0.7948
 5920/60000 [=>............................] - ETA: 1:41 - loss: 0.6469 - categorical_accuracy: 0.7956
 5952/60000 [=>............................] - ETA: 1:41 - loss: 0.6461 - categorical_accuracy: 0.7960
 5984/60000 [=>............................] - ETA: 1:41 - loss: 0.6438 - categorical_accuracy: 0.7970
 6016/60000 [==>...........................] - ETA: 1:41 - loss: 0.6417 - categorical_accuracy: 0.7975
 6048/60000 [==>...........................] - ETA: 1:41 - loss: 0.6407 - categorical_accuracy: 0.7976
 6080/60000 [==>...........................] - ETA: 1:41 - loss: 0.6394 - categorical_accuracy: 0.7980
 6112/60000 [==>...........................] - ETA: 1:41 - loss: 0.6384 - categorical_accuracy: 0.7986
 6144/60000 [==>...........................] - ETA: 1:41 - loss: 0.6363 - categorical_accuracy: 0.7990
 6176/60000 [==>...........................] - ETA: 1:40 - loss: 0.6337 - categorical_accuracy: 0.7999
 6208/60000 [==>...........................] - ETA: 1:40 - loss: 0.6330 - categorical_accuracy: 0.8001
 6240/60000 [==>...........................] - ETA: 1:40 - loss: 0.6309 - categorical_accuracy: 0.8006
 6272/60000 [==>...........................] - ETA: 1:40 - loss: 0.6293 - categorical_accuracy: 0.8013
 6304/60000 [==>...........................] - ETA: 1:40 - loss: 0.6267 - categorical_accuracy: 0.8022
 6336/60000 [==>...........................] - ETA: 1:40 - loss: 0.6256 - categorical_accuracy: 0.8027
 6368/60000 [==>...........................] - ETA: 1:40 - loss: 0.6239 - categorical_accuracy: 0.8031
 6400/60000 [==>...........................] - ETA: 1:40 - loss: 0.6241 - categorical_accuracy: 0.8033
 6432/60000 [==>...........................] - ETA: 1:40 - loss: 0.6220 - categorical_accuracy: 0.8039
 6464/60000 [==>...........................] - ETA: 1:40 - loss: 0.6197 - categorical_accuracy: 0.8046
 6496/60000 [==>...........................] - ETA: 1:40 - loss: 0.6172 - categorical_accuracy: 0.8056
 6528/60000 [==>...........................] - ETA: 1:40 - loss: 0.6159 - categorical_accuracy: 0.8059
 6560/60000 [==>...........................] - ETA: 1:40 - loss: 0.6144 - categorical_accuracy: 0.8064
 6592/60000 [==>...........................] - ETA: 1:39 - loss: 0.6126 - categorical_accuracy: 0.8070
 6624/60000 [==>...........................] - ETA: 1:39 - loss: 0.6116 - categorical_accuracy: 0.8075
 6656/60000 [==>...........................] - ETA: 1:39 - loss: 0.6097 - categorical_accuracy: 0.8081
 6688/60000 [==>...........................] - ETA: 1:39 - loss: 0.6074 - categorical_accuracy: 0.8091
 6720/60000 [==>...........................] - ETA: 1:39 - loss: 0.6063 - categorical_accuracy: 0.8092
 6752/60000 [==>...........................] - ETA: 1:39 - loss: 0.6040 - categorical_accuracy: 0.8101
 6784/60000 [==>...........................] - ETA: 1:39 - loss: 0.6014 - categorical_accuracy: 0.8110
 6816/60000 [==>...........................] - ETA: 1:39 - loss: 0.6000 - categorical_accuracy: 0.8115
 6848/60000 [==>...........................] - ETA: 1:39 - loss: 0.5985 - categorical_accuracy: 0.8119
 6880/60000 [==>...........................] - ETA: 1:39 - loss: 0.5970 - categorical_accuracy: 0.8122
 6912/60000 [==>...........................] - ETA: 1:39 - loss: 0.5949 - categorical_accuracy: 0.8128
 6944/60000 [==>...........................] - ETA: 1:38 - loss: 0.5937 - categorical_accuracy: 0.8131
 6976/60000 [==>...........................] - ETA: 1:38 - loss: 0.5924 - categorical_accuracy: 0.8135
 7008/60000 [==>...........................] - ETA: 1:38 - loss: 0.5911 - categorical_accuracy: 0.8141
 7040/60000 [==>...........................] - ETA: 1:38 - loss: 0.5893 - categorical_accuracy: 0.8148
 7072/60000 [==>...........................] - ETA: 1:38 - loss: 0.5876 - categorical_accuracy: 0.8153
 7104/60000 [==>...........................] - ETA: 1:38 - loss: 0.5858 - categorical_accuracy: 0.8159
 7136/60000 [==>...........................] - ETA: 1:38 - loss: 0.5841 - categorical_accuracy: 0.8166
 7168/60000 [==>...........................] - ETA: 1:38 - loss: 0.5833 - categorical_accuracy: 0.8171
 7200/60000 [==>...........................] - ETA: 1:38 - loss: 0.5820 - categorical_accuracy: 0.8175
 7232/60000 [==>...........................] - ETA: 1:38 - loss: 0.5807 - categorical_accuracy: 0.8180
 7264/60000 [==>...........................] - ETA: 1:38 - loss: 0.5792 - categorical_accuracy: 0.8186
 7296/60000 [==>...........................] - ETA: 1:38 - loss: 0.5781 - categorical_accuracy: 0.8189
 7328/60000 [==>...........................] - ETA: 1:38 - loss: 0.5769 - categorical_accuracy: 0.8195
 7360/60000 [==>...........................] - ETA: 1:38 - loss: 0.5752 - categorical_accuracy: 0.8200
 7392/60000 [==>...........................] - ETA: 1:37 - loss: 0.5732 - categorical_accuracy: 0.8206
 7424/60000 [==>...........................] - ETA: 1:37 - loss: 0.5730 - categorical_accuracy: 0.8206
 7456/60000 [==>...........................] - ETA: 1:37 - loss: 0.5718 - categorical_accuracy: 0.8208
 7488/60000 [==>...........................] - ETA: 1:37 - loss: 0.5701 - categorical_accuracy: 0.8213
 7520/60000 [==>...........................] - ETA: 1:37 - loss: 0.5685 - categorical_accuracy: 0.8218
 7552/60000 [==>...........................] - ETA: 1:37 - loss: 0.5668 - categorical_accuracy: 0.8223
 7584/60000 [==>...........................] - ETA: 1:37 - loss: 0.5652 - categorical_accuracy: 0.8228
 7616/60000 [==>...........................] - ETA: 1:37 - loss: 0.5639 - categorical_accuracy: 0.8231
 7648/60000 [==>...........................] - ETA: 1:37 - loss: 0.5639 - categorical_accuracy: 0.8230
 7680/60000 [==>...........................] - ETA: 1:37 - loss: 0.5624 - categorical_accuracy: 0.8234
 7712/60000 [==>...........................] - ETA: 1:37 - loss: 0.5612 - categorical_accuracy: 0.8240
 7744/60000 [==>...........................] - ETA: 1:37 - loss: 0.5598 - categorical_accuracy: 0.8244
 7776/60000 [==>...........................] - ETA: 1:36 - loss: 0.5594 - categorical_accuracy: 0.8247
 7808/60000 [==>...........................] - ETA: 1:36 - loss: 0.5577 - categorical_accuracy: 0.8253
 7840/60000 [==>...........................] - ETA: 1:36 - loss: 0.5561 - categorical_accuracy: 0.8259
 7872/60000 [==>...........................] - ETA: 1:36 - loss: 0.5547 - categorical_accuracy: 0.8263
 7904/60000 [==>...........................] - ETA: 1:36 - loss: 0.5540 - categorical_accuracy: 0.8267
 7936/60000 [==>...........................] - ETA: 1:36 - loss: 0.5525 - categorical_accuracy: 0.8270
 7968/60000 [==>...........................] - ETA: 1:36 - loss: 0.5507 - categorical_accuracy: 0.8277
 8000/60000 [===>..........................] - ETA: 1:36 - loss: 0.5490 - categorical_accuracy: 0.8281
 8032/60000 [===>..........................] - ETA: 1:36 - loss: 0.5478 - categorical_accuracy: 0.8286
 8064/60000 [===>..........................] - ETA: 1:36 - loss: 0.5466 - categorical_accuracy: 0.8290
 8096/60000 [===>..........................] - ETA: 1:36 - loss: 0.5453 - categorical_accuracy: 0.8294
 8128/60000 [===>..........................] - ETA: 1:36 - loss: 0.5439 - categorical_accuracy: 0.8300
 8160/60000 [===>..........................] - ETA: 1:36 - loss: 0.5427 - categorical_accuracy: 0.8304
 8192/60000 [===>..........................] - ETA: 1:36 - loss: 0.5412 - categorical_accuracy: 0.8308
 8224/60000 [===>..........................] - ETA: 1:36 - loss: 0.5401 - categorical_accuracy: 0.8312
 8256/60000 [===>..........................] - ETA: 1:36 - loss: 0.5383 - categorical_accuracy: 0.8319
 8288/60000 [===>..........................] - ETA: 1:36 - loss: 0.5369 - categorical_accuracy: 0.8322
 8320/60000 [===>..........................] - ETA: 1:35 - loss: 0.5351 - categorical_accuracy: 0.8327
 8352/60000 [===>..........................] - ETA: 1:35 - loss: 0.5340 - categorical_accuracy: 0.8332
 8384/60000 [===>..........................] - ETA: 1:35 - loss: 0.5337 - categorical_accuracy: 0.8333
 8416/60000 [===>..........................] - ETA: 1:35 - loss: 0.5336 - categorical_accuracy: 0.8332
 8448/60000 [===>..........................] - ETA: 1:35 - loss: 0.5319 - categorical_accuracy: 0.8338
 8480/60000 [===>..........................] - ETA: 1:35 - loss: 0.5303 - categorical_accuracy: 0.8343
 8512/60000 [===>..........................] - ETA: 1:35 - loss: 0.5293 - categorical_accuracy: 0.8346
 8544/60000 [===>..........................] - ETA: 1:35 - loss: 0.5274 - categorical_accuracy: 0.8352
 8576/60000 [===>..........................] - ETA: 1:35 - loss: 0.5256 - categorical_accuracy: 0.8358
 8608/60000 [===>..........................] - ETA: 1:35 - loss: 0.5250 - categorical_accuracy: 0.8359
 8640/60000 [===>..........................] - ETA: 1:35 - loss: 0.5237 - categorical_accuracy: 0.8363
 8672/60000 [===>..........................] - ETA: 1:35 - loss: 0.5220 - categorical_accuracy: 0.8368
 8704/60000 [===>..........................] - ETA: 1:35 - loss: 0.5216 - categorical_accuracy: 0.8367
 8736/60000 [===>..........................] - ETA: 1:35 - loss: 0.5205 - categorical_accuracy: 0.8370
 8768/60000 [===>..........................] - ETA: 1:35 - loss: 0.5190 - categorical_accuracy: 0.8375
 8800/60000 [===>..........................] - ETA: 1:35 - loss: 0.5178 - categorical_accuracy: 0.8380
 8832/60000 [===>..........................] - ETA: 1:34 - loss: 0.5163 - categorical_accuracy: 0.8384
 8864/60000 [===>..........................] - ETA: 1:34 - loss: 0.5148 - categorical_accuracy: 0.8390
 8896/60000 [===>..........................] - ETA: 1:34 - loss: 0.5135 - categorical_accuracy: 0.8395
 8928/60000 [===>..........................] - ETA: 1:34 - loss: 0.5135 - categorical_accuracy: 0.8395
 8960/60000 [===>..........................] - ETA: 1:34 - loss: 0.5123 - categorical_accuracy: 0.8398
 8992/60000 [===>..........................] - ETA: 1:34 - loss: 0.5109 - categorical_accuracy: 0.8403
 9024/60000 [===>..........................] - ETA: 1:34 - loss: 0.5094 - categorical_accuracy: 0.8408
 9056/60000 [===>..........................] - ETA: 1:34 - loss: 0.5097 - categorical_accuracy: 0.8410
 9088/60000 [===>..........................] - ETA: 1:34 - loss: 0.5084 - categorical_accuracy: 0.8414
 9120/60000 [===>..........................] - ETA: 1:34 - loss: 0.5076 - categorical_accuracy: 0.8417
 9152/60000 [===>..........................] - ETA: 1:34 - loss: 0.5078 - categorical_accuracy: 0.8419
 9184/60000 [===>..........................] - ETA: 1:34 - loss: 0.5070 - categorical_accuracy: 0.8421
 9216/60000 [===>..........................] - ETA: 1:34 - loss: 0.5066 - categorical_accuracy: 0.8423
 9248/60000 [===>..........................] - ETA: 1:34 - loss: 0.5055 - categorical_accuracy: 0.8428
 9280/60000 [===>..........................] - ETA: 1:34 - loss: 0.5042 - categorical_accuracy: 0.8432
 9312/60000 [===>..........................] - ETA: 1:34 - loss: 0.5032 - categorical_accuracy: 0.8434
 9344/60000 [===>..........................] - ETA: 1:34 - loss: 0.5019 - categorical_accuracy: 0.8439
 9376/60000 [===>..........................] - ETA: 1:33 - loss: 0.5012 - categorical_accuracy: 0.8441
 9408/60000 [===>..........................] - ETA: 1:33 - loss: 0.5001 - categorical_accuracy: 0.8444
 9440/60000 [===>..........................] - ETA: 1:33 - loss: 0.4992 - categorical_accuracy: 0.8446
 9472/60000 [===>..........................] - ETA: 1:33 - loss: 0.4985 - categorical_accuracy: 0.8449
 9504/60000 [===>..........................] - ETA: 1:33 - loss: 0.4969 - categorical_accuracy: 0.8454
 9536/60000 [===>..........................] - ETA: 1:33 - loss: 0.4958 - categorical_accuracy: 0.8457
 9568/60000 [===>..........................] - ETA: 1:33 - loss: 0.4946 - categorical_accuracy: 0.8462
 9600/60000 [===>..........................] - ETA: 1:33 - loss: 0.4933 - categorical_accuracy: 0.8466
 9632/60000 [===>..........................] - ETA: 1:33 - loss: 0.4927 - categorical_accuracy: 0.8467
 9664/60000 [===>..........................] - ETA: 1:33 - loss: 0.4917 - categorical_accuracy: 0.8471
 9696/60000 [===>..........................] - ETA: 1:33 - loss: 0.4921 - categorical_accuracy: 0.8468
 9728/60000 [===>..........................] - ETA: 1:33 - loss: 0.4915 - categorical_accuracy: 0.8470
 9760/60000 [===>..........................] - ETA: 1:33 - loss: 0.4902 - categorical_accuracy: 0.8474
 9792/60000 [===>..........................] - ETA: 1:33 - loss: 0.4891 - categorical_accuracy: 0.8478
 9824/60000 [===>..........................] - ETA: 1:33 - loss: 0.4887 - categorical_accuracy: 0.8479
 9856/60000 [===>..........................] - ETA: 1:33 - loss: 0.4877 - categorical_accuracy: 0.8482
 9888/60000 [===>..........................] - ETA: 1:33 - loss: 0.4866 - categorical_accuracy: 0.8485
 9920/60000 [===>..........................] - ETA: 1:32 - loss: 0.4861 - categorical_accuracy: 0.8488
 9952/60000 [===>..........................] - ETA: 1:32 - loss: 0.4855 - categorical_accuracy: 0.8489
 9984/60000 [===>..........................] - ETA: 1:32 - loss: 0.4843 - categorical_accuracy: 0.8493
10016/60000 [====>.........................] - ETA: 1:32 - loss: 0.4835 - categorical_accuracy: 0.8494
10048/60000 [====>.........................] - ETA: 1:32 - loss: 0.4828 - categorical_accuracy: 0.8498
10080/60000 [====>.........................] - ETA: 1:32 - loss: 0.4817 - categorical_accuracy: 0.8502
10112/60000 [====>.........................] - ETA: 1:32 - loss: 0.4814 - categorical_accuracy: 0.8504
10144/60000 [====>.........................] - ETA: 1:32 - loss: 0.4804 - categorical_accuracy: 0.8507
10176/60000 [====>.........................] - ETA: 1:32 - loss: 0.4793 - categorical_accuracy: 0.8510
10208/60000 [====>.........................] - ETA: 1:32 - loss: 0.4782 - categorical_accuracy: 0.8512
10240/60000 [====>.........................] - ETA: 1:32 - loss: 0.4771 - categorical_accuracy: 0.8517
10272/60000 [====>.........................] - ETA: 1:32 - loss: 0.4759 - categorical_accuracy: 0.8519
10304/60000 [====>.........................] - ETA: 1:32 - loss: 0.4746 - categorical_accuracy: 0.8524
10336/60000 [====>.........................] - ETA: 1:32 - loss: 0.4733 - categorical_accuracy: 0.8527
10368/60000 [====>.........................] - ETA: 1:32 - loss: 0.4729 - categorical_accuracy: 0.8528
10400/60000 [====>.........................] - ETA: 1:32 - loss: 0.4721 - categorical_accuracy: 0.8529
10432/60000 [====>.........................] - ETA: 1:32 - loss: 0.4711 - categorical_accuracy: 0.8532
10464/60000 [====>.........................] - ETA: 1:31 - loss: 0.4707 - categorical_accuracy: 0.8535
10496/60000 [====>.........................] - ETA: 1:31 - loss: 0.4704 - categorical_accuracy: 0.8536
10528/60000 [====>.........................] - ETA: 1:31 - loss: 0.4696 - categorical_accuracy: 0.8539
10560/60000 [====>.........................] - ETA: 1:31 - loss: 0.4686 - categorical_accuracy: 0.8543
10592/60000 [====>.........................] - ETA: 1:31 - loss: 0.4682 - categorical_accuracy: 0.8545
10624/60000 [====>.........................] - ETA: 1:31 - loss: 0.4671 - categorical_accuracy: 0.8549
10656/60000 [====>.........................] - ETA: 1:31 - loss: 0.4667 - categorical_accuracy: 0.8550
10688/60000 [====>.........................] - ETA: 1:31 - loss: 0.4655 - categorical_accuracy: 0.8554
10720/60000 [====>.........................] - ETA: 1:31 - loss: 0.4647 - categorical_accuracy: 0.8557
10752/60000 [====>.........................] - ETA: 1:31 - loss: 0.4642 - categorical_accuracy: 0.8557
10784/60000 [====>.........................] - ETA: 1:31 - loss: 0.4634 - categorical_accuracy: 0.8559
10816/60000 [====>.........................] - ETA: 1:31 - loss: 0.4625 - categorical_accuracy: 0.8561
10848/60000 [====>.........................] - ETA: 1:31 - loss: 0.4618 - categorical_accuracy: 0.8565
10880/60000 [====>.........................] - ETA: 1:31 - loss: 0.4609 - categorical_accuracy: 0.8568
10912/60000 [====>.........................] - ETA: 1:31 - loss: 0.4602 - categorical_accuracy: 0.8570
10944/60000 [====>.........................] - ETA: 1:31 - loss: 0.4605 - categorical_accuracy: 0.8570
10976/60000 [====>.........................] - ETA: 1:30 - loss: 0.4598 - categorical_accuracy: 0.8571
11008/60000 [====>.........................] - ETA: 1:30 - loss: 0.4590 - categorical_accuracy: 0.8573
11040/60000 [====>.........................] - ETA: 1:30 - loss: 0.4577 - categorical_accuracy: 0.8577
11072/60000 [====>.........................] - ETA: 1:30 - loss: 0.4568 - categorical_accuracy: 0.8580
11104/60000 [====>.........................] - ETA: 1:30 - loss: 0.4558 - categorical_accuracy: 0.8583
11136/60000 [====>.........................] - ETA: 1:30 - loss: 0.4548 - categorical_accuracy: 0.8586
11168/60000 [====>.........................] - ETA: 1:30 - loss: 0.4538 - categorical_accuracy: 0.8589
11200/60000 [====>.........................] - ETA: 1:30 - loss: 0.4531 - categorical_accuracy: 0.8590
11232/60000 [====>.........................] - ETA: 1:30 - loss: 0.4519 - categorical_accuracy: 0.8594
11264/60000 [====>.........................] - ETA: 1:30 - loss: 0.4509 - categorical_accuracy: 0.8597
11296/60000 [====>.........................] - ETA: 1:30 - loss: 0.4501 - categorical_accuracy: 0.8600
11328/60000 [====>.........................] - ETA: 1:30 - loss: 0.4489 - categorical_accuracy: 0.8603
11360/60000 [====>.........................] - ETA: 1:30 - loss: 0.4485 - categorical_accuracy: 0.8607
11392/60000 [====>.........................] - ETA: 1:30 - loss: 0.4482 - categorical_accuracy: 0.8607
11424/60000 [====>.........................] - ETA: 1:30 - loss: 0.4473 - categorical_accuracy: 0.8610
11456/60000 [====>.........................] - ETA: 1:30 - loss: 0.4466 - categorical_accuracy: 0.8611
11488/60000 [====>.........................] - ETA: 1:30 - loss: 0.4462 - categorical_accuracy: 0.8612
11520/60000 [====>.........................] - ETA: 1:30 - loss: 0.4451 - categorical_accuracy: 0.8615
11552/60000 [====>.........................] - ETA: 1:30 - loss: 0.4448 - categorical_accuracy: 0.8617
11584/60000 [====>.........................] - ETA: 1:30 - loss: 0.4442 - categorical_accuracy: 0.8618
11616/60000 [====>.........................] - ETA: 1:30 - loss: 0.4433 - categorical_accuracy: 0.8620
11648/60000 [====>.........................] - ETA: 1:30 - loss: 0.4429 - categorical_accuracy: 0.8621
11680/60000 [====>.........................] - ETA: 1:29 - loss: 0.4419 - categorical_accuracy: 0.8625
11712/60000 [====>.........................] - ETA: 1:29 - loss: 0.4410 - categorical_accuracy: 0.8628
11744/60000 [====>.........................] - ETA: 1:29 - loss: 0.4404 - categorical_accuracy: 0.8629
11776/60000 [====>.........................] - ETA: 1:29 - loss: 0.4402 - categorical_accuracy: 0.8628
11808/60000 [====>.........................] - ETA: 1:29 - loss: 0.4391 - categorical_accuracy: 0.8631
11840/60000 [====>.........................] - ETA: 1:29 - loss: 0.4385 - categorical_accuracy: 0.8632
11872/60000 [====>.........................] - ETA: 1:29 - loss: 0.4379 - categorical_accuracy: 0.8633
11904/60000 [====>.........................] - ETA: 1:29 - loss: 0.4369 - categorical_accuracy: 0.8636
11936/60000 [====>.........................] - ETA: 1:29 - loss: 0.4358 - categorical_accuracy: 0.8639
11968/60000 [====>.........................] - ETA: 1:29 - loss: 0.4348 - categorical_accuracy: 0.8642
12000/60000 [=====>........................] - ETA: 1:29 - loss: 0.4338 - categorical_accuracy: 0.8646
12032/60000 [=====>........................] - ETA: 1:29 - loss: 0.4328 - categorical_accuracy: 0.8649
12064/60000 [=====>........................] - ETA: 1:29 - loss: 0.4320 - categorical_accuracy: 0.8651
12096/60000 [=====>........................] - ETA: 1:29 - loss: 0.4313 - categorical_accuracy: 0.8654
12128/60000 [=====>........................] - ETA: 1:28 - loss: 0.4306 - categorical_accuracy: 0.8655
12160/60000 [=====>........................] - ETA: 1:28 - loss: 0.4305 - categorical_accuracy: 0.8655
12192/60000 [=====>........................] - ETA: 1:28 - loss: 0.4302 - categorical_accuracy: 0.8656
12224/60000 [=====>........................] - ETA: 1:28 - loss: 0.4297 - categorical_accuracy: 0.8658
12256/60000 [=====>........................] - ETA: 1:28 - loss: 0.4292 - categorical_accuracy: 0.8660
12288/60000 [=====>........................] - ETA: 1:28 - loss: 0.4283 - categorical_accuracy: 0.8664
12320/60000 [=====>........................] - ETA: 1:28 - loss: 0.4275 - categorical_accuracy: 0.8666
12352/60000 [=====>........................] - ETA: 1:28 - loss: 0.4267 - categorical_accuracy: 0.8669
12384/60000 [=====>........................] - ETA: 1:28 - loss: 0.4261 - categorical_accuracy: 0.8671
12416/60000 [=====>........................] - ETA: 1:28 - loss: 0.4254 - categorical_accuracy: 0.8673
12448/60000 [=====>........................] - ETA: 1:28 - loss: 0.4246 - categorical_accuracy: 0.8674
12480/60000 [=====>........................] - ETA: 1:28 - loss: 0.4238 - categorical_accuracy: 0.8676
12512/60000 [=====>........................] - ETA: 1:28 - loss: 0.4229 - categorical_accuracy: 0.8680
12544/60000 [=====>........................] - ETA: 1:28 - loss: 0.4219 - categorical_accuracy: 0.8683
12576/60000 [=====>........................] - ETA: 1:28 - loss: 0.4214 - categorical_accuracy: 0.8686
12608/60000 [=====>........................] - ETA: 1:27 - loss: 0.4209 - categorical_accuracy: 0.8687
12640/60000 [=====>........................] - ETA: 1:27 - loss: 0.4202 - categorical_accuracy: 0.8689
12672/60000 [=====>........................] - ETA: 1:27 - loss: 0.4193 - categorical_accuracy: 0.8692
12704/60000 [=====>........................] - ETA: 1:27 - loss: 0.4188 - categorical_accuracy: 0.8693
12736/60000 [=====>........................] - ETA: 1:27 - loss: 0.4187 - categorical_accuracy: 0.8694
12768/60000 [=====>........................] - ETA: 1:27 - loss: 0.4179 - categorical_accuracy: 0.8696
12800/60000 [=====>........................] - ETA: 1:27 - loss: 0.4176 - categorical_accuracy: 0.8697
12832/60000 [=====>........................] - ETA: 1:27 - loss: 0.4169 - categorical_accuracy: 0.8699
12864/60000 [=====>........................] - ETA: 1:27 - loss: 0.4161 - categorical_accuracy: 0.8701
12896/60000 [=====>........................] - ETA: 1:27 - loss: 0.4153 - categorical_accuracy: 0.8704
12928/60000 [=====>........................] - ETA: 1:27 - loss: 0.4145 - categorical_accuracy: 0.8707
12960/60000 [=====>........................] - ETA: 1:27 - loss: 0.4137 - categorical_accuracy: 0.8709
12992/60000 [=====>........................] - ETA: 1:27 - loss: 0.4132 - categorical_accuracy: 0.8710
13024/60000 [=====>........................] - ETA: 1:27 - loss: 0.4127 - categorical_accuracy: 0.8712
13056/60000 [=====>........................] - ETA: 1:27 - loss: 0.4117 - categorical_accuracy: 0.8715
13088/60000 [=====>........................] - ETA: 1:27 - loss: 0.4111 - categorical_accuracy: 0.8717
13120/60000 [=====>........................] - ETA: 1:26 - loss: 0.4109 - categorical_accuracy: 0.8718
13152/60000 [=====>........................] - ETA: 1:26 - loss: 0.4108 - categorical_accuracy: 0.8717
13184/60000 [=====>........................] - ETA: 1:26 - loss: 0.4099 - categorical_accuracy: 0.8720
13216/60000 [=====>........................] - ETA: 1:26 - loss: 0.4092 - categorical_accuracy: 0.8723
13248/60000 [=====>........................] - ETA: 1:26 - loss: 0.4084 - categorical_accuracy: 0.8726
13280/60000 [=====>........................] - ETA: 1:26 - loss: 0.4079 - categorical_accuracy: 0.8727
13312/60000 [=====>........................] - ETA: 1:26 - loss: 0.4080 - categorical_accuracy: 0.8727
13344/60000 [=====>........................] - ETA: 1:26 - loss: 0.4080 - categorical_accuracy: 0.8728
13376/60000 [=====>........................] - ETA: 1:26 - loss: 0.4076 - categorical_accuracy: 0.8729
13408/60000 [=====>........................] - ETA: 1:26 - loss: 0.4071 - categorical_accuracy: 0.8731
13440/60000 [=====>........................] - ETA: 1:26 - loss: 0.4072 - categorical_accuracy: 0.8731
13472/60000 [=====>........................] - ETA: 1:26 - loss: 0.4069 - categorical_accuracy: 0.8731
13504/60000 [=====>........................] - ETA: 1:26 - loss: 0.4062 - categorical_accuracy: 0.8732
13536/60000 [=====>........................] - ETA: 1:26 - loss: 0.4060 - categorical_accuracy: 0.8734
13568/60000 [=====>........................] - ETA: 1:26 - loss: 0.4057 - categorical_accuracy: 0.8734
13600/60000 [=====>........................] - ETA: 1:26 - loss: 0.4048 - categorical_accuracy: 0.8737
13632/60000 [=====>........................] - ETA: 1:25 - loss: 0.4041 - categorical_accuracy: 0.8739
13664/60000 [=====>........................] - ETA: 1:25 - loss: 0.4035 - categorical_accuracy: 0.8741
13696/60000 [=====>........................] - ETA: 1:25 - loss: 0.4029 - categorical_accuracy: 0.8743
13728/60000 [=====>........................] - ETA: 1:25 - loss: 0.4026 - categorical_accuracy: 0.8743
13760/60000 [=====>........................] - ETA: 1:25 - loss: 0.4023 - categorical_accuracy: 0.8745
13792/60000 [=====>........................] - ETA: 1:25 - loss: 0.4021 - categorical_accuracy: 0.8746
13824/60000 [=====>........................] - ETA: 1:25 - loss: 0.4014 - categorical_accuracy: 0.8749
13856/60000 [=====>........................] - ETA: 1:25 - loss: 0.4008 - categorical_accuracy: 0.8751
13888/60000 [=====>........................] - ETA: 1:25 - loss: 0.4007 - categorical_accuracy: 0.8752
13920/60000 [=====>........................] - ETA: 1:25 - loss: 0.4000 - categorical_accuracy: 0.8754
13952/60000 [=====>........................] - ETA: 1:25 - loss: 0.3994 - categorical_accuracy: 0.8756
13984/60000 [=====>........................] - ETA: 1:25 - loss: 0.3987 - categorical_accuracy: 0.8757
14016/60000 [======>.......................] - ETA: 1:25 - loss: 0.3980 - categorical_accuracy: 0.8759
14048/60000 [======>.......................] - ETA: 1:25 - loss: 0.3975 - categorical_accuracy: 0.8761
14080/60000 [======>.......................] - ETA: 1:25 - loss: 0.3971 - categorical_accuracy: 0.8761
14112/60000 [======>.......................] - ETA: 1:25 - loss: 0.3972 - categorical_accuracy: 0.8761
14144/60000 [======>.......................] - ETA: 1:24 - loss: 0.3964 - categorical_accuracy: 0.8763
14176/60000 [======>.......................] - ETA: 1:24 - loss: 0.3963 - categorical_accuracy: 0.8764
14208/60000 [======>.......................] - ETA: 1:24 - loss: 0.3956 - categorical_accuracy: 0.8767
14240/60000 [======>.......................] - ETA: 1:24 - loss: 0.3949 - categorical_accuracy: 0.8770
14272/60000 [======>.......................] - ETA: 1:24 - loss: 0.3940 - categorical_accuracy: 0.8772
14304/60000 [======>.......................] - ETA: 1:24 - loss: 0.3938 - categorical_accuracy: 0.8774
14336/60000 [======>.......................] - ETA: 1:24 - loss: 0.3932 - categorical_accuracy: 0.8776
14368/60000 [======>.......................] - ETA: 1:24 - loss: 0.3928 - categorical_accuracy: 0.8777
14400/60000 [======>.......................] - ETA: 1:24 - loss: 0.3925 - categorical_accuracy: 0.8778
14432/60000 [======>.......................] - ETA: 1:24 - loss: 0.3923 - categorical_accuracy: 0.8778
14464/60000 [======>.......................] - ETA: 1:24 - loss: 0.3919 - categorical_accuracy: 0.8780
14496/60000 [======>.......................] - ETA: 1:24 - loss: 0.3913 - categorical_accuracy: 0.8782
14528/60000 [======>.......................] - ETA: 1:24 - loss: 0.3906 - categorical_accuracy: 0.8784
14560/60000 [======>.......................] - ETA: 1:24 - loss: 0.3901 - categorical_accuracy: 0.8785
14592/60000 [======>.......................] - ETA: 1:24 - loss: 0.3896 - categorical_accuracy: 0.8787
14624/60000 [======>.......................] - ETA: 1:24 - loss: 0.3888 - categorical_accuracy: 0.8790
14656/60000 [======>.......................] - ETA: 1:24 - loss: 0.3881 - categorical_accuracy: 0.8792
14688/60000 [======>.......................] - ETA: 1:23 - loss: 0.3879 - categorical_accuracy: 0.8792
14720/60000 [======>.......................] - ETA: 1:23 - loss: 0.3873 - categorical_accuracy: 0.8793
14752/60000 [======>.......................] - ETA: 1:23 - loss: 0.3866 - categorical_accuracy: 0.8796
14784/60000 [======>.......................] - ETA: 1:23 - loss: 0.3859 - categorical_accuracy: 0.8799
14816/60000 [======>.......................] - ETA: 1:23 - loss: 0.3853 - categorical_accuracy: 0.8800
14848/60000 [======>.......................] - ETA: 1:23 - loss: 0.3849 - categorical_accuracy: 0.8801
14880/60000 [======>.......................] - ETA: 1:23 - loss: 0.3848 - categorical_accuracy: 0.8802
14912/60000 [======>.......................] - ETA: 1:23 - loss: 0.3846 - categorical_accuracy: 0.8803
14944/60000 [======>.......................] - ETA: 1:23 - loss: 0.3841 - categorical_accuracy: 0.8805
14976/60000 [======>.......................] - ETA: 1:23 - loss: 0.3835 - categorical_accuracy: 0.8807
15008/60000 [======>.......................] - ETA: 1:23 - loss: 0.3831 - categorical_accuracy: 0.8808
15040/60000 [======>.......................] - ETA: 1:23 - loss: 0.3825 - categorical_accuracy: 0.8810
15072/60000 [======>.......................] - ETA: 1:23 - loss: 0.3823 - categorical_accuracy: 0.8811
15104/60000 [======>.......................] - ETA: 1:23 - loss: 0.3816 - categorical_accuracy: 0.8813
15136/60000 [======>.......................] - ETA: 1:23 - loss: 0.3817 - categorical_accuracy: 0.8813
15168/60000 [======>.......................] - ETA: 1:23 - loss: 0.3812 - categorical_accuracy: 0.8815
15200/60000 [======>.......................] - ETA: 1:22 - loss: 0.3812 - categorical_accuracy: 0.8814
15232/60000 [======>.......................] - ETA: 1:22 - loss: 0.3805 - categorical_accuracy: 0.8816
15264/60000 [======>.......................] - ETA: 1:22 - loss: 0.3799 - categorical_accuracy: 0.8818
15296/60000 [======>.......................] - ETA: 1:22 - loss: 0.3795 - categorical_accuracy: 0.8819
15328/60000 [======>.......................] - ETA: 1:22 - loss: 0.3790 - categorical_accuracy: 0.8820
15360/60000 [======>.......................] - ETA: 1:22 - loss: 0.3784 - categorical_accuracy: 0.8822
15392/60000 [======>.......................] - ETA: 1:22 - loss: 0.3789 - categorical_accuracy: 0.8821
15424/60000 [======>.......................] - ETA: 1:22 - loss: 0.3784 - categorical_accuracy: 0.8823
15456/60000 [======>.......................] - ETA: 1:22 - loss: 0.3778 - categorical_accuracy: 0.8826
15488/60000 [======>.......................] - ETA: 1:22 - loss: 0.3771 - categorical_accuracy: 0.8827
15520/60000 [======>.......................] - ETA: 1:22 - loss: 0.3768 - categorical_accuracy: 0.8829
15552/60000 [======>.......................] - ETA: 1:22 - loss: 0.3762 - categorical_accuracy: 0.8830
15584/60000 [======>.......................] - ETA: 1:22 - loss: 0.3755 - categorical_accuracy: 0.8832
15616/60000 [======>.......................] - ETA: 1:22 - loss: 0.3749 - categorical_accuracy: 0.8835
15648/60000 [======>.......................] - ETA: 1:22 - loss: 0.3744 - categorical_accuracy: 0.8836
15680/60000 [======>.......................] - ETA: 1:22 - loss: 0.3741 - categorical_accuracy: 0.8837
15712/60000 [======>.......................] - ETA: 1:22 - loss: 0.3738 - categorical_accuracy: 0.8838
15744/60000 [======>.......................] - ETA: 1:21 - loss: 0.3731 - categorical_accuracy: 0.8841
15776/60000 [======>.......................] - ETA: 1:21 - loss: 0.3730 - categorical_accuracy: 0.8841
15808/60000 [======>.......................] - ETA: 1:21 - loss: 0.3725 - categorical_accuracy: 0.8842
15840/60000 [======>.......................] - ETA: 1:21 - loss: 0.3722 - categorical_accuracy: 0.8844
15872/60000 [======>.......................] - ETA: 1:21 - loss: 0.3715 - categorical_accuracy: 0.8846
15904/60000 [======>.......................] - ETA: 1:21 - loss: 0.3710 - categorical_accuracy: 0.8848
15936/60000 [======>.......................] - ETA: 1:21 - loss: 0.3712 - categorical_accuracy: 0.8849
15968/60000 [======>.......................] - ETA: 1:21 - loss: 0.3709 - categorical_accuracy: 0.8850
16000/60000 [=======>......................] - ETA: 1:21 - loss: 0.3704 - categorical_accuracy: 0.8851
16032/60000 [=======>......................] - ETA: 1:21 - loss: 0.3703 - categorical_accuracy: 0.8850
16064/60000 [=======>......................] - ETA: 1:21 - loss: 0.3699 - categorical_accuracy: 0.8852
16096/60000 [=======>......................] - ETA: 1:21 - loss: 0.3693 - categorical_accuracy: 0.8854
16128/60000 [=======>......................] - ETA: 1:21 - loss: 0.3689 - categorical_accuracy: 0.8855
16160/60000 [=======>......................] - ETA: 1:21 - loss: 0.3684 - categorical_accuracy: 0.8856
16192/60000 [=======>......................] - ETA: 1:21 - loss: 0.3678 - categorical_accuracy: 0.8859
16224/60000 [=======>......................] - ETA: 1:21 - loss: 0.3672 - categorical_accuracy: 0.8860
16256/60000 [=======>......................] - ETA: 1:21 - loss: 0.3667 - categorical_accuracy: 0.8862
16288/60000 [=======>......................] - ETA: 1:21 - loss: 0.3663 - categorical_accuracy: 0.8863
16320/60000 [=======>......................] - ETA: 1:20 - loss: 0.3661 - categorical_accuracy: 0.8863
16352/60000 [=======>......................] - ETA: 1:20 - loss: 0.3655 - categorical_accuracy: 0.8865
16384/60000 [=======>......................] - ETA: 1:20 - loss: 0.3653 - categorical_accuracy: 0.8866
16416/60000 [=======>......................] - ETA: 1:20 - loss: 0.3650 - categorical_accuracy: 0.8867
16448/60000 [=======>......................] - ETA: 1:20 - loss: 0.3644 - categorical_accuracy: 0.8869
16480/60000 [=======>......................] - ETA: 1:20 - loss: 0.3638 - categorical_accuracy: 0.8871
16512/60000 [=======>......................] - ETA: 1:20 - loss: 0.3632 - categorical_accuracy: 0.8873
16544/60000 [=======>......................] - ETA: 1:20 - loss: 0.3634 - categorical_accuracy: 0.8873
16576/60000 [=======>......................] - ETA: 1:20 - loss: 0.3632 - categorical_accuracy: 0.8875
16608/60000 [=======>......................] - ETA: 1:20 - loss: 0.3628 - categorical_accuracy: 0.8876
16640/60000 [=======>......................] - ETA: 1:20 - loss: 0.3624 - categorical_accuracy: 0.8877
16672/60000 [=======>......................] - ETA: 1:20 - loss: 0.3622 - categorical_accuracy: 0.8879
16704/60000 [=======>......................] - ETA: 1:20 - loss: 0.3616 - categorical_accuracy: 0.8881
16736/60000 [=======>......................] - ETA: 1:20 - loss: 0.3612 - categorical_accuracy: 0.8882
16768/60000 [=======>......................] - ETA: 1:20 - loss: 0.3606 - categorical_accuracy: 0.8884
16800/60000 [=======>......................] - ETA: 1:20 - loss: 0.3602 - categorical_accuracy: 0.8886
16832/60000 [=======>......................] - ETA: 1:20 - loss: 0.3596 - categorical_accuracy: 0.8887
16864/60000 [=======>......................] - ETA: 1:19 - loss: 0.3592 - categorical_accuracy: 0.8888
16896/60000 [=======>......................] - ETA: 1:19 - loss: 0.3592 - categorical_accuracy: 0.8889
16928/60000 [=======>......................] - ETA: 1:19 - loss: 0.3593 - categorical_accuracy: 0.8889
16960/60000 [=======>......................] - ETA: 1:19 - loss: 0.3588 - categorical_accuracy: 0.8891
16992/60000 [=======>......................] - ETA: 1:19 - loss: 0.3588 - categorical_accuracy: 0.8891
17024/60000 [=======>......................] - ETA: 1:19 - loss: 0.3583 - categorical_accuracy: 0.8893
17056/60000 [=======>......................] - ETA: 1:19 - loss: 0.3580 - categorical_accuracy: 0.8893
17088/60000 [=======>......................] - ETA: 1:19 - loss: 0.3575 - categorical_accuracy: 0.8895
17120/60000 [=======>......................] - ETA: 1:19 - loss: 0.3573 - categorical_accuracy: 0.8895
17152/60000 [=======>......................] - ETA: 1:19 - loss: 0.3569 - categorical_accuracy: 0.8896
17184/60000 [=======>......................] - ETA: 1:19 - loss: 0.3564 - categorical_accuracy: 0.8898
17216/60000 [=======>......................] - ETA: 1:19 - loss: 0.3557 - categorical_accuracy: 0.8900
17248/60000 [=======>......................] - ETA: 1:19 - loss: 0.3552 - categorical_accuracy: 0.8901
17280/60000 [=======>......................] - ETA: 1:19 - loss: 0.3548 - categorical_accuracy: 0.8902
17312/60000 [=======>......................] - ETA: 1:19 - loss: 0.3543 - categorical_accuracy: 0.8904
17344/60000 [=======>......................] - ETA: 1:18 - loss: 0.3541 - categorical_accuracy: 0.8905
17376/60000 [=======>......................] - ETA: 1:18 - loss: 0.3535 - categorical_accuracy: 0.8907
17408/60000 [=======>......................] - ETA: 1:18 - loss: 0.3530 - categorical_accuracy: 0.8909
17440/60000 [=======>......................] - ETA: 1:18 - loss: 0.3524 - categorical_accuracy: 0.8911
17472/60000 [=======>......................] - ETA: 1:18 - loss: 0.3519 - categorical_accuracy: 0.8911
17504/60000 [=======>......................] - ETA: 1:18 - loss: 0.3517 - categorical_accuracy: 0.8912
17536/60000 [=======>......................] - ETA: 1:18 - loss: 0.3511 - categorical_accuracy: 0.8914
17568/60000 [=======>......................] - ETA: 1:18 - loss: 0.3510 - categorical_accuracy: 0.8915
17600/60000 [=======>......................] - ETA: 1:18 - loss: 0.3504 - categorical_accuracy: 0.8916
17632/60000 [=======>......................] - ETA: 1:18 - loss: 0.3499 - categorical_accuracy: 0.8918
17664/60000 [=======>......................] - ETA: 1:18 - loss: 0.3493 - categorical_accuracy: 0.8920
17696/60000 [=======>......................] - ETA: 1:18 - loss: 0.3490 - categorical_accuracy: 0.8921
17728/60000 [=======>......................] - ETA: 1:18 - loss: 0.3487 - categorical_accuracy: 0.8921
17760/60000 [=======>......................] - ETA: 1:18 - loss: 0.3482 - categorical_accuracy: 0.8923
17792/60000 [=======>......................] - ETA: 1:18 - loss: 0.3480 - categorical_accuracy: 0.8924
17824/60000 [=======>......................] - ETA: 1:18 - loss: 0.3475 - categorical_accuracy: 0.8926
17856/60000 [=======>......................] - ETA: 1:18 - loss: 0.3478 - categorical_accuracy: 0.8925
17888/60000 [=======>......................] - ETA: 1:17 - loss: 0.3473 - categorical_accuracy: 0.8927
17920/60000 [=======>......................] - ETA: 1:17 - loss: 0.3468 - categorical_accuracy: 0.8928
17952/60000 [=======>......................] - ETA: 1:17 - loss: 0.3463 - categorical_accuracy: 0.8930
17984/60000 [=======>......................] - ETA: 1:17 - loss: 0.3459 - categorical_accuracy: 0.8931
18016/60000 [========>.....................] - ETA: 1:17 - loss: 0.3457 - categorical_accuracy: 0.8932
18048/60000 [========>.....................] - ETA: 1:17 - loss: 0.3453 - categorical_accuracy: 0.8933
18080/60000 [========>.....................] - ETA: 1:17 - loss: 0.3450 - categorical_accuracy: 0.8933
18112/60000 [========>.....................] - ETA: 1:17 - loss: 0.3452 - categorical_accuracy: 0.8933
18144/60000 [========>.....................] - ETA: 1:17 - loss: 0.3449 - categorical_accuracy: 0.8934
18176/60000 [========>.....................] - ETA: 1:17 - loss: 0.3445 - categorical_accuracy: 0.8935
18208/60000 [========>.....................] - ETA: 1:17 - loss: 0.3449 - categorical_accuracy: 0.8935
18240/60000 [========>.....................] - ETA: 1:17 - loss: 0.3445 - categorical_accuracy: 0.8936
18272/60000 [========>.....................] - ETA: 1:17 - loss: 0.3440 - categorical_accuracy: 0.8937
18304/60000 [========>.....................] - ETA: 1:17 - loss: 0.3436 - categorical_accuracy: 0.8938
18336/60000 [========>.....................] - ETA: 1:17 - loss: 0.3431 - categorical_accuracy: 0.8940
18368/60000 [========>.....................] - ETA: 1:17 - loss: 0.3426 - categorical_accuracy: 0.8942
18400/60000 [========>.....................] - ETA: 1:17 - loss: 0.3422 - categorical_accuracy: 0.8942
18432/60000 [========>.....................] - ETA: 1:16 - loss: 0.3417 - categorical_accuracy: 0.8944
18464/60000 [========>.....................] - ETA: 1:16 - loss: 0.3412 - categorical_accuracy: 0.8946
18496/60000 [========>.....................] - ETA: 1:16 - loss: 0.3408 - categorical_accuracy: 0.8947
18528/60000 [========>.....................] - ETA: 1:16 - loss: 0.3404 - categorical_accuracy: 0.8948
18560/60000 [========>.....................] - ETA: 1:16 - loss: 0.3399 - categorical_accuracy: 0.8950
18592/60000 [========>.....................] - ETA: 1:16 - loss: 0.3395 - categorical_accuracy: 0.8951
18624/60000 [========>.....................] - ETA: 1:16 - loss: 0.3391 - categorical_accuracy: 0.8952
18656/60000 [========>.....................] - ETA: 1:16 - loss: 0.3391 - categorical_accuracy: 0.8953
18688/60000 [========>.....................] - ETA: 1:16 - loss: 0.3389 - categorical_accuracy: 0.8954
18720/60000 [========>.....................] - ETA: 1:16 - loss: 0.3384 - categorical_accuracy: 0.8956
18752/60000 [========>.....................] - ETA: 1:16 - loss: 0.3379 - categorical_accuracy: 0.8957
18784/60000 [========>.....................] - ETA: 1:16 - loss: 0.3375 - categorical_accuracy: 0.8958
18816/60000 [========>.....................] - ETA: 1:16 - loss: 0.3371 - categorical_accuracy: 0.8959
18848/60000 [========>.....................] - ETA: 1:16 - loss: 0.3370 - categorical_accuracy: 0.8960
18880/60000 [========>.....................] - ETA: 1:16 - loss: 0.3365 - categorical_accuracy: 0.8961
18912/60000 [========>.....................] - ETA: 1:16 - loss: 0.3360 - categorical_accuracy: 0.8963
18944/60000 [========>.....................] - ETA: 1:15 - loss: 0.3357 - categorical_accuracy: 0.8964
18976/60000 [========>.....................] - ETA: 1:15 - loss: 0.3355 - categorical_accuracy: 0.8966
19008/60000 [========>.....................] - ETA: 1:15 - loss: 0.3351 - categorical_accuracy: 0.8967
19040/60000 [========>.....................] - ETA: 1:15 - loss: 0.3347 - categorical_accuracy: 0.8968
19072/60000 [========>.....................] - ETA: 1:15 - loss: 0.3342 - categorical_accuracy: 0.8970
19104/60000 [========>.....................] - ETA: 1:15 - loss: 0.3337 - categorical_accuracy: 0.8971
19136/60000 [========>.....................] - ETA: 1:15 - loss: 0.3332 - categorical_accuracy: 0.8973
19168/60000 [========>.....................] - ETA: 1:15 - loss: 0.3329 - categorical_accuracy: 0.8974
19200/60000 [========>.....................] - ETA: 1:15 - loss: 0.3329 - categorical_accuracy: 0.8974
19232/60000 [========>.....................] - ETA: 1:15 - loss: 0.3324 - categorical_accuracy: 0.8975
19264/60000 [========>.....................] - ETA: 1:15 - loss: 0.3319 - categorical_accuracy: 0.8977
19296/60000 [========>.....................] - ETA: 1:15 - loss: 0.3317 - categorical_accuracy: 0.8978
19328/60000 [========>.....................] - ETA: 1:15 - loss: 0.3312 - categorical_accuracy: 0.8979
19360/60000 [========>.....................] - ETA: 1:15 - loss: 0.3310 - categorical_accuracy: 0.8980
19392/60000 [========>.....................] - ETA: 1:15 - loss: 0.3308 - categorical_accuracy: 0.8981
19424/60000 [========>.....................] - ETA: 1:15 - loss: 0.3304 - categorical_accuracy: 0.8982
19456/60000 [========>.....................] - ETA: 1:15 - loss: 0.3304 - categorical_accuracy: 0.8982
19488/60000 [========>.....................] - ETA: 1:14 - loss: 0.3300 - categorical_accuracy: 0.8983
19520/60000 [========>.....................] - ETA: 1:14 - loss: 0.3295 - categorical_accuracy: 0.8985
19552/60000 [========>.....................] - ETA: 1:14 - loss: 0.3291 - categorical_accuracy: 0.8986
19584/60000 [========>.....................] - ETA: 1:14 - loss: 0.3287 - categorical_accuracy: 0.8987
19616/60000 [========>.....................] - ETA: 1:14 - loss: 0.3284 - categorical_accuracy: 0.8988
19648/60000 [========>.....................] - ETA: 1:14 - loss: 0.3281 - categorical_accuracy: 0.8989
19680/60000 [========>.....................] - ETA: 1:14 - loss: 0.3279 - categorical_accuracy: 0.8989
19712/60000 [========>.....................] - ETA: 1:14 - loss: 0.3276 - categorical_accuracy: 0.8990
19744/60000 [========>.....................] - ETA: 1:14 - loss: 0.3275 - categorical_accuracy: 0.8990
19776/60000 [========>.....................] - ETA: 1:14 - loss: 0.3276 - categorical_accuracy: 0.8991
19808/60000 [========>.....................] - ETA: 1:14 - loss: 0.3276 - categorical_accuracy: 0.8991
19840/60000 [========>.....................] - ETA: 1:14 - loss: 0.3272 - categorical_accuracy: 0.8992
19872/60000 [========>.....................] - ETA: 1:14 - loss: 0.3268 - categorical_accuracy: 0.8993
19904/60000 [========>.....................] - ETA: 1:14 - loss: 0.3264 - categorical_accuracy: 0.8994
19936/60000 [========>.....................] - ETA: 1:14 - loss: 0.3270 - categorical_accuracy: 0.8993
19968/60000 [========>.....................] - ETA: 1:14 - loss: 0.3267 - categorical_accuracy: 0.8994
20000/60000 [=========>....................] - ETA: 1:14 - loss: 0.3265 - categorical_accuracy: 0.8995
20032/60000 [=========>....................] - ETA: 1:13 - loss: 0.3261 - categorical_accuracy: 0.8997
20064/60000 [=========>....................] - ETA: 1:13 - loss: 0.3257 - categorical_accuracy: 0.8998
20096/60000 [=========>....................] - ETA: 1:13 - loss: 0.3252 - categorical_accuracy: 0.8999
20128/60000 [=========>....................] - ETA: 1:13 - loss: 0.3248 - categorical_accuracy: 0.9001
20160/60000 [=========>....................] - ETA: 1:13 - loss: 0.3250 - categorical_accuracy: 0.9000
20192/60000 [=========>....................] - ETA: 1:13 - loss: 0.3247 - categorical_accuracy: 0.9001
20224/60000 [=========>....................] - ETA: 1:13 - loss: 0.3242 - categorical_accuracy: 0.9002
20256/60000 [=========>....................] - ETA: 1:13 - loss: 0.3238 - categorical_accuracy: 0.9003
20288/60000 [=========>....................] - ETA: 1:13 - loss: 0.3236 - categorical_accuracy: 0.9003
20320/60000 [=========>....................] - ETA: 1:13 - loss: 0.3236 - categorical_accuracy: 0.9004
20352/60000 [=========>....................] - ETA: 1:13 - loss: 0.3232 - categorical_accuracy: 0.9005
20384/60000 [=========>....................] - ETA: 1:13 - loss: 0.3229 - categorical_accuracy: 0.9006
20416/60000 [=========>....................] - ETA: 1:13 - loss: 0.3227 - categorical_accuracy: 0.9007
20448/60000 [=========>....................] - ETA: 1:13 - loss: 0.3222 - categorical_accuracy: 0.9008
20480/60000 [=========>....................] - ETA: 1:13 - loss: 0.3220 - categorical_accuracy: 0.9009
20512/60000 [=========>....................] - ETA: 1:13 - loss: 0.3215 - categorical_accuracy: 0.9010
20544/60000 [=========>....................] - ETA: 1:12 - loss: 0.3213 - categorical_accuracy: 0.9011
20576/60000 [=========>....................] - ETA: 1:12 - loss: 0.3215 - categorical_accuracy: 0.9011
20608/60000 [=========>....................] - ETA: 1:12 - loss: 0.3211 - categorical_accuracy: 0.9012
20640/60000 [=========>....................] - ETA: 1:12 - loss: 0.3208 - categorical_accuracy: 0.9013
20672/60000 [=========>....................] - ETA: 1:12 - loss: 0.3208 - categorical_accuracy: 0.9014
20704/60000 [=========>....................] - ETA: 1:12 - loss: 0.3205 - categorical_accuracy: 0.9015
20736/60000 [=========>....................] - ETA: 1:12 - loss: 0.3203 - categorical_accuracy: 0.9015
20768/60000 [=========>....................] - ETA: 1:12 - loss: 0.3202 - categorical_accuracy: 0.9016
20800/60000 [=========>....................] - ETA: 1:12 - loss: 0.3202 - categorical_accuracy: 0.9016
20832/60000 [=========>....................] - ETA: 1:12 - loss: 0.3199 - categorical_accuracy: 0.9017
20864/60000 [=========>....................] - ETA: 1:12 - loss: 0.3195 - categorical_accuracy: 0.9017
20896/60000 [=========>....................] - ETA: 1:12 - loss: 0.3194 - categorical_accuracy: 0.9018
20928/60000 [=========>....................] - ETA: 1:12 - loss: 0.3191 - categorical_accuracy: 0.9019
20960/60000 [=========>....................] - ETA: 1:12 - loss: 0.3187 - categorical_accuracy: 0.9020
20992/60000 [=========>....................] - ETA: 1:12 - loss: 0.3183 - categorical_accuracy: 0.9021
21024/60000 [=========>....................] - ETA: 1:12 - loss: 0.3182 - categorical_accuracy: 0.9022
21056/60000 [=========>....................] - ETA: 1:12 - loss: 0.3178 - categorical_accuracy: 0.9023
21088/60000 [=========>....................] - ETA: 1:12 - loss: 0.3175 - categorical_accuracy: 0.9024
21120/60000 [=========>....................] - ETA: 1:11 - loss: 0.3174 - categorical_accuracy: 0.9025
21152/60000 [=========>....................] - ETA: 1:11 - loss: 0.3172 - categorical_accuracy: 0.9025
21184/60000 [=========>....................] - ETA: 1:11 - loss: 0.3170 - categorical_accuracy: 0.9025
21216/60000 [=========>....................] - ETA: 1:11 - loss: 0.3170 - categorical_accuracy: 0.9025
21248/60000 [=========>....................] - ETA: 1:11 - loss: 0.3168 - categorical_accuracy: 0.9025
21280/60000 [=========>....................] - ETA: 1:11 - loss: 0.3166 - categorical_accuracy: 0.9025
21312/60000 [=========>....................] - ETA: 1:11 - loss: 0.3162 - categorical_accuracy: 0.9027
21344/60000 [=========>....................] - ETA: 1:11 - loss: 0.3159 - categorical_accuracy: 0.9028
21376/60000 [=========>....................] - ETA: 1:11 - loss: 0.3155 - categorical_accuracy: 0.9029
21408/60000 [=========>....................] - ETA: 1:11 - loss: 0.3152 - categorical_accuracy: 0.9030
21440/60000 [=========>....................] - ETA: 1:11 - loss: 0.3150 - categorical_accuracy: 0.9030
21472/60000 [=========>....................] - ETA: 1:11 - loss: 0.3147 - categorical_accuracy: 0.9031
21504/60000 [=========>....................] - ETA: 1:11 - loss: 0.3145 - categorical_accuracy: 0.9032
21536/60000 [=========>....................] - ETA: 1:11 - loss: 0.3142 - categorical_accuracy: 0.9033
21568/60000 [=========>....................] - ETA: 1:11 - loss: 0.3140 - categorical_accuracy: 0.9033
21600/60000 [=========>....................] - ETA: 1:11 - loss: 0.3137 - categorical_accuracy: 0.9033
21632/60000 [=========>....................] - ETA: 1:11 - loss: 0.3134 - categorical_accuracy: 0.9035
21664/60000 [=========>....................] - ETA: 1:10 - loss: 0.3135 - categorical_accuracy: 0.9035
21696/60000 [=========>....................] - ETA: 1:10 - loss: 0.3131 - categorical_accuracy: 0.9036
21728/60000 [=========>....................] - ETA: 1:10 - loss: 0.3127 - categorical_accuracy: 0.9037
21760/60000 [=========>....................] - ETA: 1:10 - loss: 0.3124 - categorical_accuracy: 0.9038
21792/60000 [=========>....................] - ETA: 1:10 - loss: 0.3121 - categorical_accuracy: 0.9039
21824/60000 [=========>....................] - ETA: 1:10 - loss: 0.3117 - categorical_accuracy: 0.9041
21856/60000 [=========>....................] - ETA: 1:10 - loss: 0.3117 - categorical_accuracy: 0.9041
21888/60000 [=========>....................] - ETA: 1:10 - loss: 0.3117 - categorical_accuracy: 0.9041
21920/60000 [=========>....................] - ETA: 1:10 - loss: 0.3118 - categorical_accuracy: 0.9042
21952/60000 [=========>....................] - ETA: 1:10 - loss: 0.3114 - categorical_accuracy: 0.9043
21984/60000 [=========>....................] - ETA: 1:10 - loss: 0.3110 - categorical_accuracy: 0.9044
22016/60000 [==========>...................] - ETA: 1:10 - loss: 0.3107 - categorical_accuracy: 0.9045
22048/60000 [==========>...................] - ETA: 1:10 - loss: 0.3107 - categorical_accuracy: 0.9045
22080/60000 [==========>...................] - ETA: 1:10 - loss: 0.3104 - categorical_accuracy: 0.9046
22112/60000 [==========>...................] - ETA: 1:10 - loss: 0.3102 - categorical_accuracy: 0.9047
22144/60000 [==========>...................] - ETA: 1:10 - loss: 0.3101 - categorical_accuracy: 0.9048
22176/60000 [==========>...................] - ETA: 1:10 - loss: 0.3098 - categorical_accuracy: 0.9049
22208/60000 [==========>...................] - ETA: 1:10 - loss: 0.3095 - categorical_accuracy: 0.9049
22240/60000 [==========>...................] - ETA: 1:09 - loss: 0.3093 - categorical_accuracy: 0.9050
22272/60000 [==========>...................] - ETA: 1:09 - loss: 0.3090 - categorical_accuracy: 0.9050
22304/60000 [==========>...................] - ETA: 1:09 - loss: 0.3087 - categorical_accuracy: 0.9052
22336/60000 [==========>...................] - ETA: 1:09 - loss: 0.3085 - categorical_accuracy: 0.9052
22368/60000 [==========>...................] - ETA: 1:09 - loss: 0.3081 - categorical_accuracy: 0.9053
22400/60000 [==========>...................] - ETA: 1:09 - loss: 0.3079 - categorical_accuracy: 0.9054
22432/60000 [==========>...................] - ETA: 1:09 - loss: 0.3078 - categorical_accuracy: 0.9055
22464/60000 [==========>...................] - ETA: 1:09 - loss: 0.3074 - categorical_accuracy: 0.9056
22496/60000 [==========>...................] - ETA: 1:09 - loss: 0.3071 - categorical_accuracy: 0.9057
22528/60000 [==========>...................] - ETA: 1:09 - loss: 0.3069 - categorical_accuracy: 0.9057
22560/60000 [==========>...................] - ETA: 1:09 - loss: 0.3069 - categorical_accuracy: 0.9057
22592/60000 [==========>...................] - ETA: 1:09 - loss: 0.3067 - categorical_accuracy: 0.9058
22624/60000 [==========>...................] - ETA: 1:09 - loss: 0.3063 - categorical_accuracy: 0.9059
22656/60000 [==========>...................] - ETA: 1:09 - loss: 0.3060 - categorical_accuracy: 0.9060
22688/60000 [==========>...................] - ETA: 1:09 - loss: 0.3060 - categorical_accuracy: 0.9060
22720/60000 [==========>...................] - ETA: 1:09 - loss: 0.3057 - categorical_accuracy: 0.9061
22752/60000 [==========>...................] - ETA: 1:08 - loss: 0.3054 - categorical_accuracy: 0.9062
22784/60000 [==========>...................] - ETA: 1:08 - loss: 0.3051 - categorical_accuracy: 0.9062
22816/60000 [==========>...................] - ETA: 1:08 - loss: 0.3048 - categorical_accuracy: 0.9063
22848/60000 [==========>...................] - ETA: 1:08 - loss: 0.3044 - categorical_accuracy: 0.9065
22880/60000 [==========>...................] - ETA: 1:08 - loss: 0.3044 - categorical_accuracy: 0.9065
22912/60000 [==========>...................] - ETA: 1:08 - loss: 0.3046 - categorical_accuracy: 0.9065
22944/60000 [==========>...................] - ETA: 1:08 - loss: 0.3044 - categorical_accuracy: 0.9065
22976/60000 [==========>...................] - ETA: 1:08 - loss: 0.3041 - categorical_accuracy: 0.9065
23008/60000 [==========>...................] - ETA: 1:08 - loss: 0.3040 - categorical_accuracy: 0.9065
23040/60000 [==========>...................] - ETA: 1:08 - loss: 0.3036 - categorical_accuracy: 0.9066
23072/60000 [==========>...................] - ETA: 1:08 - loss: 0.3034 - categorical_accuracy: 0.9066
23104/60000 [==========>...................] - ETA: 1:08 - loss: 0.3031 - categorical_accuracy: 0.9067
23136/60000 [==========>...................] - ETA: 1:08 - loss: 0.3031 - categorical_accuracy: 0.9067
23168/60000 [==========>...................] - ETA: 1:08 - loss: 0.3028 - categorical_accuracy: 0.9068
23200/60000 [==========>...................] - ETA: 1:08 - loss: 0.3027 - categorical_accuracy: 0.9068
23232/60000 [==========>...................] - ETA: 1:08 - loss: 0.3023 - categorical_accuracy: 0.9069
23264/60000 [==========>...................] - ETA: 1:08 - loss: 0.3021 - categorical_accuracy: 0.9070
23296/60000 [==========>...................] - ETA: 1:07 - loss: 0.3019 - categorical_accuracy: 0.9070
23328/60000 [==========>...................] - ETA: 1:07 - loss: 0.3017 - categorical_accuracy: 0.9070
23360/60000 [==========>...................] - ETA: 1:07 - loss: 0.3013 - categorical_accuracy: 0.9071
23392/60000 [==========>...................] - ETA: 1:07 - loss: 0.3009 - categorical_accuracy: 0.9073
23424/60000 [==========>...................] - ETA: 1:07 - loss: 0.3009 - categorical_accuracy: 0.9073
23456/60000 [==========>...................] - ETA: 1:07 - loss: 0.3006 - categorical_accuracy: 0.9073
23488/60000 [==========>...................] - ETA: 1:07 - loss: 0.3005 - categorical_accuracy: 0.9074
23520/60000 [==========>...................] - ETA: 1:07 - loss: 0.3001 - categorical_accuracy: 0.9075
23552/60000 [==========>...................] - ETA: 1:07 - loss: 0.3000 - categorical_accuracy: 0.9075
23584/60000 [==========>...................] - ETA: 1:07 - loss: 0.2997 - categorical_accuracy: 0.9076
23616/60000 [==========>...................] - ETA: 1:07 - loss: 0.2995 - categorical_accuracy: 0.9077
23648/60000 [==========>...................] - ETA: 1:07 - loss: 0.2992 - categorical_accuracy: 0.9078
23680/60000 [==========>...................] - ETA: 1:07 - loss: 0.2991 - categorical_accuracy: 0.9079
23712/60000 [==========>...................] - ETA: 1:07 - loss: 0.2990 - categorical_accuracy: 0.9080
23744/60000 [==========>...................] - ETA: 1:07 - loss: 0.2988 - categorical_accuracy: 0.9080
23776/60000 [==========>...................] - ETA: 1:07 - loss: 0.2985 - categorical_accuracy: 0.9081
23808/60000 [==========>...................] - ETA: 1:06 - loss: 0.2982 - categorical_accuracy: 0.9082
23840/60000 [==========>...................] - ETA: 1:06 - loss: 0.2980 - categorical_accuracy: 0.9082
23872/60000 [==========>...................] - ETA: 1:06 - loss: 0.2978 - categorical_accuracy: 0.9083
23904/60000 [==========>...................] - ETA: 1:06 - loss: 0.2975 - categorical_accuracy: 0.9084
23936/60000 [==========>...................] - ETA: 1:06 - loss: 0.2971 - categorical_accuracy: 0.9085
23968/60000 [==========>...................] - ETA: 1:06 - loss: 0.2969 - categorical_accuracy: 0.9085
24000/60000 [===========>..................] - ETA: 1:06 - loss: 0.2967 - categorical_accuracy: 0.9086
24032/60000 [===========>..................] - ETA: 1:06 - loss: 0.2967 - categorical_accuracy: 0.9086
24064/60000 [===========>..................] - ETA: 1:06 - loss: 0.2966 - categorical_accuracy: 0.9086
24096/60000 [===========>..................] - ETA: 1:06 - loss: 0.2962 - categorical_accuracy: 0.9087
24128/60000 [===========>..................] - ETA: 1:06 - loss: 0.2961 - categorical_accuracy: 0.9087
24160/60000 [===========>..................] - ETA: 1:06 - loss: 0.2960 - categorical_accuracy: 0.9087
24192/60000 [===========>..................] - ETA: 1:06 - loss: 0.2958 - categorical_accuracy: 0.9088
24224/60000 [===========>..................] - ETA: 1:06 - loss: 0.2955 - categorical_accuracy: 0.9089
24256/60000 [===========>..................] - ETA: 1:06 - loss: 0.2954 - categorical_accuracy: 0.9089
24288/60000 [===========>..................] - ETA: 1:06 - loss: 0.2951 - categorical_accuracy: 0.9090
24320/60000 [===========>..................] - ETA: 1:05 - loss: 0.2948 - categorical_accuracy: 0.9091
24352/60000 [===========>..................] - ETA: 1:05 - loss: 0.2945 - categorical_accuracy: 0.9092
24384/60000 [===========>..................] - ETA: 1:05 - loss: 0.2943 - categorical_accuracy: 0.9092
24416/60000 [===========>..................] - ETA: 1:05 - loss: 0.2940 - categorical_accuracy: 0.9093
24448/60000 [===========>..................] - ETA: 1:05 - loss: 0.2939 - categorical_accuracy: 0.9094
24480/60000 [===========>..................] - ETA: 1:05 - loss: 0.2937 - categorical_accuracy: 0.9094
24512/60000 [===========>..................] - ETA: 1:05 - loss: 0.2934 - categorical_accuracy: 0.9095
24544/60000 [===========>..................] - ETA: 1:05 - loss: 0.2933 - categorical_accuracy: 0.9096
24576/60000 [===========>..................] - ETA: 1:05 - loss: 0.2930 - categorical_accuracy: 0.9097
24608/60000 [===========>..................] - ETA: 1:05 - loss: 0.2927 - categorical_accuracy: 0.9098
24640/60000 [===========>..................] - ETA: 1:05 - loss: 0.2926 - categorical_accuracy: 0.9097
24672/60000 [===========>..................] - ETA: 1:05 - loss: 0.2923 - categorical_accuracy: 0.9098
24704/60000 [===========>..................] - ETA: 1:05 - loss: 0.2920 - categorical_accuracy: 0.9099
24736/60000 [===========>..................] - ETA: 1:05 - loss: 0.2918 - categorical_accuracy: 0.9099
24768/60000 [===========>..................] - ETA: 1:05 - loss: 0.2916 - categorical_accuracy: 0.9100
24800/60000 [===========>..................] - ETA: 1:05 - loss: 0.2913 - categorical_accuracy: 0.9101
24832/60000 [===========>..................] - ETA: 1:04 - loss: 0.2912 - categorical_accuracy: 0.9101
24864/60000 [===========>..................] - ETA: 1:04 - loss: 0.2911 - categorical_accuracy: 0.9100
24896/60000 [===========>..................] - ETA: 1:04 - loss: 0.2908 - categorical_accuracy: 0.9101
24928/60000 [===========>..................] - ETA: 1:04 - loss: 0.2906 - categorical_accuracy: 0.9102
24960/60000 [===========>..................] - ETA: 1:04 - loss: 0.2907 - categorical_accuracy: 0.9103
24992/60000 [===========>..................] - ETA: 1:04 - loss: 0.2904 - categorical_accuracy: 0.9103
25024/60000 [===========>..................] - ETA: 1:04 - loss: 0.2901 - categorical_accuracy: 0.9104
25056/60000 [===========>..................] - ETA: 1:04 - loss: 0.2898 - categorical_accuracy: 0.9105
25088/60000 [===========>..................] - ETA: 1:04 - loss: 0.2901 - categorical_accuracy: 0.9105
25120/60000 [===========>..................] - ETA: 1:04 - loss: 0.2899 - categorical_accuracy: 0.9105
25152/60000 [===========>..................] - ETA: 1:04 - loss: 0.2897 - categorical_accuracy: 0.9106
25184/60000 [===========>..................] - ETA: 1:04 - loss: 0.2898 - categorical_accuracy: 0.9106
25216/60000 [===========>..................] - ETA: 1:04 - loss: 0.2895 - categorical_accuracy: 0.9107
25248/60000 [===========>..................] - ETA: 1:04 - loss: 0.2895 - categorical_accuracy: 0.9107
25280/60000 [===========>..................] - ETA: 1:04 - loss: 0.2893 - categorical_accuracy: 0.9108
25312/60000 [===========>..................] - ETA: 1:04 - loss: 0.2890 - categorical_accuracy: 0.9108
25344/60000 [===========>..................] - ETA: 1:04 - loss: 0.2887 - categorical_accuracy: 0.9109
25376/60000 [===========>..................] - ETA: 1:03 - loss: 0.2884 - categorical_accuracy: 0.9110
25408/60000 [===========>..................] - ETA: 1:03 - loss: 0.2881 - categorical_accuracy: 0.9111
25440/60000 [===========>..................] - ETA: 1:03 - loss: 0.2878 - categorical_accuracy: 0.9112
25472/60000 [===========>..................] - ETA: 1:03 - loss: 0.2875 - categorical_accuracy: 0.9113
25504/60000 [===========>..................] - ETA: 1:03 - loss: 0.2877 - categorical_accuracy: 0.9112
25536/60000 [===========>..................] - ETA: 1:03 - loss: 0.2875 - categorical_accuracy: 0.9113
25568/60000 [===========>..................] - ETA: 1:03 - loss: 0.2873 - categorical_accuracy: 0.9114
25600/60000 [===========>..................] - ETA: 1:03 - loss: 0.2873 - categorical_accuracy: 0.9113
25632/60000 [===========>..................] - ETA: 1:03 - loss: 0.2873 - categorical_accuracy: 0.9112
25664/60000 [===========>..................] - ETA: 1:03 - loss: 0.2870 - categorical_accuracy: 0.9112
25696/60000 [===========>..................] - ETA: 1:03 - loss: 0.2868 - categorical_accuracy: 0.9113
25728/60000 [===========>..................] - ETA: 1:03 - loss: 0.2870 - categorical_accuracy: 0.9112
25760/60000 [===========>..................] - ETA: 1:03 - loss: 0.2867 - categorical_accuracy: 0.9113
25792/60000 [===========>..................] - ETA: 1:03 - loss: 0.2866 - categorical_accuracy: 0.9113
25824/60000 [===========>..................] - ETA: 1:03 - loss: 0.2864 - categorical_accuracy: 0.9114
25856/60000 [===========>..................] - ETA: 1:03 - loss: 0.2862 - categorical_accuracy: 0.9114
25888/60000 [===========>..................] - ETA: 1:03 - loss: 0.2861 - categorical_accuracy: 0.9115
25920/60000 [===========>..................] - ETA: 1:02 - loss: 0.2860 - categorical_accuracy: 0.9115
25952/60000 [===========>..................] - ETA: 1:02 - loss: 0.2857 - categorical_accuracy: 0.9116
25984/60000 [===========>..................] - ETA: 1:02 - loss: 0.2854 - categorical_accuracy: 0.9117
26016/60000 [============>.................] - ETA: 1:02 - loss: 0.2851 - categorical_accuracy: 0.9117
26048/60000 [============>.................] - ETA: 1:02 - loss: 0.2851 - categorical_accuracy: 0.9118
26080/60000 [============>.................] - ETA: 1:02 - loss: 0.2848 - categorical_accuracy: 0.9119
26112/60000 [============>.................] - ETA: 1:02 - loss: 0.2849 - categorical_accuracy: 0.9119
26144/60000 [============>.................] - ETA: 1:02 - loss: 0.2847 - categorical_accuracy: 0.9120
26176/60000 [============>.................] - ETA: 1:02 - loss: 0.2843 - categorical_accuracy: 0.9121
26208/60000 [============>.................] - ETA: 1:02 - loss: 0.2841 - categorical_accuracy: 0.9122
26240/60000 [============>.................] - ETA: 1:02 - loss: 0.2838 - categorical_accuracy: 0.9123
26272/60000 [============>.................] - ETA: 1:02 - loss: 0.2835 - categorical_accuracy: 0.9124
26304/60000 [============>.................] - ETA: 1:02 - loss: 0.2832 - categorical_accuracy: 0.9125
26336/60000 [============>.................] - ETA: 1:02 - loss: 0.2829 - categorical_accuracy: 0.9126
26368/60000 [============>.................] - ETA: 1:02 - loss: 0.2827 - categorical_accuracy: 0.9127
26400/60000 [============>.................] - ETA: 1:02 - loss: 0.2827 - categorical_accuracy: 0.9127
26432/60000 [============>.................] - ETA: 1:01 - loss: 0.2825 - categorical_accuracy: 0.9127
26464/60000 [============>.................] - ETA: 1:01 - loss: 0.2824 - categorical_accuracy: 0.9128
26496/60000 [============>.................] - ETA: 1:01 - loss: 0.2823 - categorical_accuracy: 0.9128
26528/60000 [============>.................] - ETA: 1:01 - loss: 0.2820 - categorical_accuracy: 0.9129
26560/60000 [============>.................] - ETA: 1:01 - loss: 0.2819 - categorical_accuracy: 0.9130
26592/60000 [============>.................] - ETA: 1:01 - loss: 0.2816 - categorical_accuracy: 0.9131
26624/60000 [============>.................] - ETA: 1:01 - loss: 0.2814 - categorical_accuracy: 0.9131
26656/60000 [============>.................] - ETA: 1:01 - loss: 0.2812 - categorical_accuracy: 0.9132
26688/60000 [============>.................] - ETA: 1:01 - loss: 0.2809 - categorical_accuracy: 0.9132
26720/60000 [============>.................] - ETA: 1:01 - loss: 0.2806 - categorical_accuracy: 0.9133
26752/60000 [============>.................] - ETA: 1:01 - loss: 0.2803 - categorical_accuracy: 0.9134
26784/60000 [============>.................] - ETA: 1:01 - loss: 0.2802 - categorical_accuracy: 0.9135
26816/60000 [============>.................] - ETA: 1:01 - loss: 0.2803 - categorical_accuracy: 0.9135
26848/60000 [============>.................] - ETA: 1:01 - loss: 0.2800 - categorical_accuracy: 0.9136
26880/60000 [============>.................] - ETA: 1:01 - loss: 0.2798 - categorical_accuracy: 0.9137
26912/60000 [============>.................] - ETA: 1:01 - loss: 0.2795 - categorical_accuracy: 0.9138
26944/60000 [============>.................] - ETA: 1:01 - loss: 0.2799 - categorical_accuracy: 0.9137
26976/60000 [============>.................] - ETA: 1:00 - loss: 0.2797 - categorical_accuracy: 0.9138
27008/60000 [============>.................] - ETA: 1:00 - loss: 0.2794 - categorical_accuracy: 0.9138
27040/60000 [============>.................] - ETA: 1:00 - loss: 0.2791 - categorical_accuracy: 0.9139
27072/60000 [============>.................] - ETA: 1:00 - loss: 0.2789 - categorical_accuracy: 0.9140
27104/60000 [============>.................] - ETA: 1:00 - loss: 0.2789 - categorical_accuracy: 0.9141
27136/60000 [============>.................] - ETA: 1:00 - loss: 0.2785 - categorical_accuracy: 0.9142
27168/60000 [============>.................] - ETA: 1:00 - loss: 0.2783 - categorical_accuracy: 0.9142
27200/60000 [============>.................] - ETA: 1:00 - loss: 0.2784 - categorical_accuracy: 0.9142
27232/60000 [============>.................] - ETA: 1:00 - loss: 0.2781 - categorical_accuracy: 0.9143
27264/60000 [============>.................] - ETA: 1:00 - loss: 0.2778 - categorical_accuracy: 0.9144
27296/60000 [============>.................] - ETA: 1:00 - loss: 0.2776 - categorical_accuracy: 0.9144
27328/60000 [============>.................] - ETA: 1:00 - loss: 0.2773 - categorical_accuracy: 0.9145
27360/60000 [============>.................] - ETA: 1:00 - loss: 0.2771 - categorical_accuracy: 0.9145
27392/60000 [============>.................] - ETA: 1:00 - loss: 0.2768 - categorical_accuracy: 0.9146
27424/60000 [============>.................] - ETA: 1:00 - loss: 0.2766 - categorical_accuracy: 0.9147
27456/60000 [============>.................] - ETA: 1:00 - loss: 0.2763 - categorical_accuracy: 0.9148
27488/60000 [============>.................] - ETA: 1:00 - loss: 0.2761 - categorical_accuracy: 0.9149
27520/60000 [============>.................] - ETA: 59s - loss: 0.2759 - categorical_accuracy: 0.9149 
27552/60000 [============>.................] - ETA: 59s - loss: 0.2757 - categorical_accuracy: 0.9150
27584/60000 [============>.................] - ETA: 59s - loss: 0.2755 - categorical_accuracy: 0.9150
27616/60000 [============>.................] - ETA: 59s - loss: 0.2755 - categorical_accuracy: 0.9150
27648/60000 [============>.................] - ETA: 59s - loss: 0.2755 - categorical_accuracy: 0.9150
27680/60000 [============>.................] - ETA: 59s - loss: 0.2759 - categorical_accuracy: 0.9150
27712/60000 [============>.................] - ETA: 59s - loss: 0.2757 - categorical_accuracy: 0.9150
27744/60000 [============>.................] - ETA: 59s - loss: 0.2755 - categorical_accuracy: 0.9150
27776/60000 [============>.................] - ETA: 59s - loss: 0.2752 - categorical_accuracy: 0.9151
27808/60000 [============>.................] - ETA: 59s - loss: 0.2749 - categorical_accuracy: 0.9152
27840/60000 [============>.................] - ETA: 59s - loss: 0.2747 - categorical_accuracy: 0.9153
27872/60000 [============>.................] - ETA: 59s - loss: 0.2744 - categorical_accuracy: 0.9154
27904/60000 [============>.................] - ETA: 59s - loss: 0.2744 - categorical_accuracy: 0.9154
27936/60000 [============>.................] - ETA: 59s - loss: 0.2744 - categorical_accuracy: 0.9154
27968/60000 [============>.................] - ETA: 59s - loss: 0.2742 - categorical_accuracy: 0.9155
28000/60000 [=============>................] - ETA: 59s - loss: 0.2739 - categorical_accuracy: 0.9156
28032/60000 [=============>................] - ETA: 59s - loss: 0.2736 - categorical_accuracy: 0.9157
28064/60000 [=============>................] - ETA: 58s - loss: 0.2733 - categorical_accuracy: 0.9158
28096/60000 [=============>................] - ETA: 58s - loss: 0.2731 - categorical_accuracy: 0.9159
28128/60000 [=============>................] - ETA: 58s - loss: 0.2728 - categorical_accuracy: 0.9160
28160/60000 [=============>................] - ETA: 58s - loss: 0.2728 - categorical_accuracy: 0.9160
28192/60000 [=============>................] - ETA: 58s - loss: 0.2728 - categorical_accuracy: 0.9160
28224/60000 [=============>................] - ETA: 58s - loss: 0.2726 - categorical_accuracy: 0.9161
28256/60000 [=============>................] - ETA: 58s - loss: 0.2725 - categorical_accuracy: 0.9161
28288/60000 [=============>................] - ETA: 58s - loss: 0.2726 - categorical_accuracy: 0.9161
28320/60000 [=============>................] - ETA: 58s - loss: 0.2726 - categorical_accuracy: 0.9161
28352/60000 [=============>................] - ETA: 58s - loss: 0.2723 - categorical_accuracy: 0.9162
28384/60000 [=============>................] - ETA: 58s - loss: 0.2721 - categorical_accuracy: 0.9162
28416/60000 [=============>................] - ETA: 58s - loss: 0.2719 - categorical_accuracy: 0.9162
28448/60000 [=============>................] - ETA: 58s - loss: 0.2717 - categorical_accuracy: 0.9163
28480/60000 [=============>................] - ETA: 58s - loss: 0.2714 - categorical_accuracy: 0.9164
28512/60000 [=============>................] - ETA: 58s - loss: 0.2713 - categorical_accuracy: 0.9165
28544/60000 [=============>................] - ETA: 58s - loss: 0.2713 - categorical_accuracy: 0.9165
28576/60000 [=============>................] - ETA: 57s - loss: 0.2711 - categorical_accuracy: 0.9166
28608/60000 [=============>................] - ETA: 57s - loss: 0.2709 - categorical_accuracy: 0.9166
28640/60000 [=============>................] - ETA: 57s - loss: 0.2707 - categorical_accuracy: 0.9167
28672/60000 [=============>................] - ETA: 57s - loss: 0.2704 - categorical_accuracy: 0.9168
28704/60000 [=============>................] - ETA: 57s - loss: 0.2702 - categorical_accuracy: 0.9169
28736/60000 [=============>................] - ETA: 57s - loss: 0.2700 - categorical_accuracy: 0.9169
28768/60000 [=============>................] - ETA: 57s - loss: 0.2698 - categorical_accuracy: 0.9170
28800/60000 [=============>................] - ETA: 57s - loss: 0.2697 - categorical_accuracy: 0.9170
28832/60000 [=============>................] - ETA: 57s - loss: 0.2696 - categorical_accuracy: 0.9170
28864/60000 [=============>................] - ETA: 57s - loss: 0.2693 - categorical_accuracy: 0.9171
28896/60000 [=============>................] - ETA: 57s - loss: 0.2695 - categorical_accuracy: 0.9170
28928/60000 [=============>................] - ETA: 57s - loss: 0.2693 - categorical_accuracy: 0.9171
28960/60000 [=============>................] - ETA: 57s - loss: 0.2691 - categorical_accuracy: 0.9172
28992/60000 [=============>................] - ETA: 57s - loss: 0.2689 - categorical_accuracy: 0.9172
29024/60000 [=============>................] - ETA: 57s - loss: 0.2687 - categorical_accuracy: 0.9173
29056/60000 [=============>................] - ETA: 57s - loss: 0.2685 - categorical_accuracy: 0.9174
29088/60000 [=============>................] - ETA: 57s - loss: 0.2683 - categorical_accuracy: 0.9175
29120/60000 [=============>................] - ETA: 56s - loss: 0.2681 - categorical_accuracy: 0.9175
29152/60000 [=============>................] - ETA: 56s - loss: 0.2680 - categorical_accuracy: 0.9175
29184/60000 [=============>................] - ETA: 56s - loss: 0.2678 - categorical_accuracy: 0.9176
29216/60000 [=============>................] - ETA: 56s - loss: 0.2676 - categorical_accuracy: 0.9176
29248/60000 [=============>................] - ETA: 56s - loss: 0.2674 - categorical_accuracy: 0.9177
29280/60000 [=============>................] - ETA: 56s - loss: 0.2676 - categorical_accuracy: 0.9176
29312/60000 [=============>................] - ETA: 56s - loss: 0.2674 - categorical_accuracy: 0.9176
29344/60000 [=============>................] - ETA: 56s - loss: 0.2672 - categorical_accuracy: 0.9177
29376/60000 [=============>................] - ETA: 56s - loss: 0.2669 - categorical_accuracy: 0.9178
29408/60000 [=============>................] - ETA: 56s - loss: 0.2670 - categorical_accuracy: 0.9178
29440/60000 [=============>................] - ETA: 56s - loss: 0.2668 - categorical_accuracy: 0.9178
29472/60000 [=============>................] - ETA: 56s - loss: 0.2666 - categorical_accuracy: 0.9179
29504/60000 [=============>................] - ETA: 56s - loss: 0.2665 - categorical_accuracy: 0.9180
29536/60000 [=============>................] - ETA: 56s - loss: 0.2662 - categorical_accuracy: 0.9181
29568/60000 [=============>................] - ETA: 56s - loss: 0.2660 - categorical_accuracy: 0.9182
29600/60000 [=============>................] - ETA: 56s - loss: 0.2659 - categorical_accuracy: 0.9182
29632/60000 [=============>................] - ETA: 56s - loss: 0.2656 - categorical_accuracy: 0.9183
29664/60000 [=============>................] - ETA: 55s - loss: 0.2654 - categorical_accuracy: 0.9184
29696/60000 [=============>................] - ETA: 55s - loss: 0.2652 - categorical_accuracy: 0.9184
29728/60000 [=============>................] - ETA: 55s - loss: 0.2653 - categorical_accuracy: 0.9185
29760/60000 [=============>................] - ETA: 55s - loss: 0.2652 - categorical_accuracy: 0.9185
29792/60000 [=============>................] - ETA: 55s - loss: 0.2652 - categorical_accuracy: 0.9185
29824/60000 [=============>................] - ETA: 55s - loss: 0.2650 - categorical_accuracy: 0.9186
29856/60000 [=============>................] - ETA: 55s - loss: 0.2647 - categorical_accuracy: 0.9186
29888/60000 [=============>................] - ETA: 55s - loss: 0.2645 - categorical_accuracy: 0.9187
29920/60000 [=============>................] - ETA: 55s - loss: 0.2645 - categorical_accuracy: 0.9187
29952/60000 [=============>................] - ETA: 55s - loss: 0.2643 - categorical_accuracy: 0.9188
29984/60000 [=============>................] - ETA: 55s - loss: 0.2640 - categorical_accuracy: 0.9189
30016/60000 [==============>...............] - ETA: 55s - loss: 0.2638 - categorical_accuracy: 0.9189
30048/60000 [==============>...............] - ETA: 55s - loss: 0.2636 - categorical_accuracy: 0.9189
30080/60000 [==============>...............] - ETA: 55s - loss: 0.2634 - categorical_accuracy: 0.9190
30112/60000 [==============>...............] - ETA: 55s - loss: 0.2635 - categorical_accuracy: 0.9190
30144/60000 [==============>...............] - ETA: 55s - loss: 0.2633 - categorical_accuracy: 0.9191
30176/60000 [==============>...............] - ETA: 54s - loss: 0.2631 - categorical_accuracy: 0.9191
30208/60000 [==============>...............] - ETA: 54s - loss: 0.2629 - categorical_accuracy: 0.9192
30240/60000 [==============>...............] - ETA: 54s - loss: 0.2627 - categorical_accuracy: 0.9193
30272/60000 [==============>...............] - ETA: 54s - loss: 0.2625 - categorical_accuracy: 0.9193
30304/60000 [==============>...............] - ETA: 54s - loss: 0.2624 - categorical_accuracy: 0.9193
30336/60000 [==============>...............] - ETA: 54s - loss: 0.2622 - categorical_accuracy: 0.9194
30368/60000 [==============>...............] - ETA: 54s - loss: 0.2621 - categorical_accuracy: 0.9194
30400/60000 [==============>...............] - ETA: 54s - loss: 0.2621 - categorical_accuracy: 0.9194
30432/60000 [==============>...............] - ETA: 54s - loss: 0.2619 - categorical_accuracy: 0.9195
30464/60000 [==============>...............] - ETA: 54s - loss: 0.2616 - categorical_accuracy: 0.9196
30496/60000 [==============>...............] - ETA: 54s - loss: 0.2615 - categorical_accuracy: 0.9196
30528/60000 [==============>...............] - ETA: 54s - loss: 0.2612 - categorical_accuracy: 0.9197
30560/60000 [==============>...............] - ETA: 54s - loss: 0.2610 - categorical_accuracy: 0.9198
30592/60000 [==============>...............] - ETA: 54s - loss: 0.2608 - categorical_accuracy: 0.9198
30624/60000 [==============>...............] - ETA: 54s - loss: 0.2605 - categorical_accuracy: 0.9199
30656/60000 [==============>...............] - ETA: 54s - loss: 0.2604 - categorical_accuracy: 0.9199
30688/60000 [==============>...............] - ETA: 54s - loss: 0.2602 - categorical_accuracy: 0.9200
30720/60000 [==============>...............] - ETA: 53s - loss: 0.2601 - categorical_accuracy: 0.9201
30752/60000 [==============>...............] - ETA: 53s - loss: 0.2601 - categorical_accuracy: 0.9200
30784/60000 [==============>...............] - ETA: 53s - loss: 0.2600 - categorical_accuracy: 0.9201
30816/60000 [==============>...............] - ETA: 53s - loss: 0.2598 - categorical_accuracy: 0.9201
30848/60000 [==============>...............] - ETA: 53s - loss: 0.2596 - categorical_accuracy: 0.9202
30880/60000 [==============>...............] - ETA: 53s - loss: 0.2593 - categorical_accuracy: 0.9202
30912/60000 [==============>...............] - ETA: 53s - loss: 0.2591 - categorical_accuracy: 0.9203
30944/60000 [==============>...............] - ETA: 53s - loss: 0.2589 - categorical_accuracy: 0.9203
30976/60000 [==============>...............] - ETA: 53s - loss: 0.2588 - categorical_accuracy: 0.9204
31008/60000 [==============>...............] - ETA: 53s - loss: 0.2586 - categorical_accuracy: 0.9204
31040/60000 [==============>...............] - ETA: 53s - loss: 0.2586 - categorical_accuracy: 0.9205
31072/60000 [==============>...............] - ETA: 53s - loss: 0.2587 - categorical_accuracy: 0.9204
31104/60000 [==============>...............] - ETA: 53s - loss: 0.2586 - categorical_accuracy: 0.9205
31136/60000 [==============>...............] - ETA: 53s - loss: 0.2584 - categorical_accuracy: 0.9205
31168/60000 [==============>...............] - ETA: 53s - loss: 0.2581 - categorical_accuracy: 0.9206
31200/60000 [==============>...............] - ETA: 53s - loss: 0.2580 - categorical_accuracy: 0.9206
31232/60000 [==============>...............] - ETA: 53s - loss: 0.2579 - categorical_accuracy: 0.9206
31264/60000 [==============>...............] - ETA: 52s - loss: 0.2577 - categorical_accuracy: 0.9207
31296/60000 [==============>...............] - ETA: 52s - loss: 0.2576 - categorical_accuracy: 0.9207
31328/60000 [==============>...............] - ETA: 52s - loss: 0.2575 - categorical_accuracy: 0.9208
31360/60000 [==============>...............] - ETA: 52s - loss: 0.2575 - categorical_accuracy: 0.9207
31392/60000 [==============>...............] - ETA: 52s - loss: 0.2573 - categorical_accuracy: 0.9208
31424/60000 [==============>...............] - ETA: 52s - loss: 0.2573 - categorical_accuracy: 0.9208
31456/60000 [==============>...............] - ETA: 52s - loss: 0.2572 - categorical_accuracy: 0.9208
31488/60000 [==============>...............] - ETA: 52s - loss: 0.2570 - categorical_accuracy: 0.9208
31520/60000 [==============>...............] - ETA: 52s - loss: 0.2568 - categorical_accuracy: 0.9209
31552/60000 [==============>...............] - ETA: 52s - loss: 0.2566 - categorical_accuracy: 0.9210
31584/60000 [==============>...............] - ETA: 52s - loss: 0.2567 - categorical_accuracy: 0.9210
31616/60000 [==============>...............] - ETA: 52s - loss: 0.2565 - categorical_accuracy: 0.9211
31648/60000 [==============>...............] - ETA: 52s - loss: 0.2562 - categorical_accuracy: 0.9212
31680/60000 [==============>...............] - ETA: 52s - loss: 0.2560 - categorical_accuracy: 0.9212
31712/60000 [==============>...............] - ETA: 52s - loss: 0.2559 - categorical_accuracy: 0.9213
31744/60000 [==============>...............] - ETA: 52s - loss: 0.2558 - categorical_accuracy: 0.9213
31776/60000 [==============>...............] - ETA: 52s - loss: 0.2558 - categorical_accuracy: 0.9214
31808/60000 [==============>...............] - ETA: 51s - loss: 0.2556 - categorical_accuracy: 0.9214
31840/60000 [==============>...............] - ETA: 51s - loss: 0.2555 - categorical_accuracy: 0.9215
31872/60000 [==============>...............] - ETA: 51s - loss: 0.2554 - categorical_accuracy: 0.9215
31904/60000 [==============>...............] - ETA: 51s - loss: 0.2554 - categorical_accuracy: 0.9215
31936/60000 [==============>...............] - ETA: 51s - loss: 0.2552 - categorical_accuracy: 0.9215
31968/60000 [==============>...............] - ETA: 51s - loss: 0.2549 - categorical_accuracy: 0.9216
32000/60000 [===============>..............] - ETA: 51s - loss: 0.2547 - categorical_accuracy: 0.9217
32032/60000 [===============>..............] - ETA: 51s - loss: 0.2547 - categorical_accuracy: 0.9217
32064/60000 [===============>..............] - ETA: 51s - loss: 0.2545 - categorical_accuracy: 0.9218
32096/60000 [===============>..............] - ETA: 51s - loss: 0.2543 - categorical_accuracy: 0.9218
32128/60000 [===============>..............] - ETA: 51s - loss: 0.2541 - categorical_accuracy: 0.9219
32160/60000 [===============>..............] - ETA: 51s - loss: 0.2540 - categorical_accuracy: 0.9219
32192/60000 [===============>..............] - ETA: 51s - loss: 0.2538 - categorical_accuracy: 0.9219
32224/60000 [===============>..............] - ETA: 51s - loss: 0.2537 - categorical_accuracy: 0.9219
32256/60000 [===============>..............] - ETA: 51s - loss: 0.2534 - categorical_accuracy: 0.9220
32288/60000 [===============>..............] - ETA: 51s - loss: 0.2534 - categorical_accuracy: 0.9220
32320/60000 [===============>..............] - ETA: 51s - loss: 0.2532 - categorical_accuracy: 0.9221
32352/60000 [===============>..............] - ETA: 50s - loss: 0.2535 - categorical_accuracy: 0.9221
32384/60000 [===============>..............] - ETA: 50s - loss: 0.2534 - categorical_accuracy: 0.9222
32416/60000 [===============>..............] - ETA: 50s - loss: 0.2534 - categorical_accuracy: 0.9221
32448/60000 [===============>..............] - ETA: 50s - loss: 0.2532 - categorical_accuracy: 0.9222
32480/60000 [===============>..............] - ETA: 50s - loss: 0.2530 - categorical_accuracy: 0.9222
32512/60000 [===============>..............] - ETA: 50s - loss: 0.2531 - categorical_accuracy: 0.9222
32544/60000 [===============>..............] - ETA: 50s - loss: 0.2531 - categorical_accuracy: 0.9222
32576/60000 [===============>..............] - ETA: 50s - loss: 0.2529 - categorical_accuracy: 0.9222
32608/60000 [===============>..............] - ETA: 50s - loss: 0.2527 - categorical_accuracy: 0.9223
32640/60000 [===============>..............] - ETA: 50s - loss: 0.2528 - categorical_accuracy: 0.9223
32672/60000 [===============>..............] - ETA: 50s - loss: 0.2527 - categorical_accuracy: 0.9223
32704/60000 [===============>..............] - ETA: 50s - loss: 0.2529 - categorical_accuracy: 0.9223
32736/60000 [===============>..............] - ETA: 50s - loss: 0.2529 - categorical_accuracy: 0.9223
32768/60000 [===============>..............] - ETA: 50s - loss: 0.2527 - categorical_accuracy: 0.9224
32800/60000 [===============>..............] - ETA: 50s - loss: 0.2526 - categorical_accuracy: 0.9224
32832/60000 [===============>..............] - ETA: 50s - loss: 0.2524 - categorical_accuracy: 0.9225
32864/60000 [===============>..............] - ETA: 50s - loss: 0.2523 - categorical_accuracy: 0.9226
32896/60000 [===============>..............] - ETA: 49s - loss: 0.2521 - categorical_accuracy: 0.9226
32928/60000 [===============>..............] - ETA: 49s - loss: 0.2519 - categorical_accuracy: 0.9226
32960/60000 [===============>..............] - ETA: 49s - loss: 0.2517 - categorical_accuracy: 0.9227
32992/60000 [===============>..............] - ETA: 49s - loss: 0.2515 - categorical_accuracy: 0.9228
33024/60000 [===============>..............] - ETA: 49s - loss: 0.2514 - categorical_accuracy: 0.9228
33056/60000 [===============>..............] - ETA: 49s - loss: 0.2515 - categorical_accuracy: 0.9228
33088/60000 [===============>..............] - ETA: 49s - loss: 0.2517 - categorical_accuracy: 0.9228
33120/60000 [===============>..............] - ETA: 49s - loss: 0.2515 - categorical_accuracy: 0.9229
33152/60000 [===============>..............] - ETA: 49s - loss: 0.2513 - categorical_accuracy: 0.9229
33184/60000 [===============>..............] - ETA: 49s - loss: 0.2513 - categorical_accuracy: 0.9230
33216/60000 [===============>..............] - ETA: 49s - loss: 0.2510 - categorical_accuracy: 0.9230
33248/60000 [===============>..............] - ETA: 49s - loss: 0.2508 - categorical_accuracy: 0.9231
33280/60000 [===============>..............] - ETA: 49s - loss: 0.2507 - categorical_accuracy: 0.9232
33312/60000 [===============>..............] - ETA: 49s - loss: 0.2506 - categorical_accuracy: 0.9232
33344/60000 [===============>..............] - ETA: 49s - loss: 0.2504 - categorical_accuracy: 0.9233
33376/60000 [===============>..............] - ETA: 49s - loss: 0.2503 - categorical_accuracy: 0.9232
33408/60000 [===============>..............] - ETA: 48s - loss: 0.2503 - categorical_accuracy: 0.9233
33440/60000 [===============>..............] - ETA: 48s - loss: 0.2501 - categorical_accuracy: 0.9233
33472/60000 [===============>..............] - ETA: 48s - loss: 0.2500 - categorical_accuracy: 0.9233
33504/60000 [===============>..............] - ETA: 48s - loss: 0.2499 - categorical_accuracy: 0.9233
33536/60000 [===============>..............] - ETA: 48s - loss: 0.2498 - categorical_accuracy: 0.9234
33568/60000 [===============>..............] - ETA: 48s - loss: 0.2497 - categorical_accuracy: 0.9233
33600/60000 [===============>..............] - ETA: 48s - loss: 0.2496 - categorical_accuracy: 0.9234
33632/60000 [===============>..............] - ETA: 48s - loss: 0.2494 - categorical_accuracy: 0.9234
33664/60000 [===============>..............] - ETA: 48s - loss: 0.2493 - categorical_accuracy: 0.9235
33696/60000 [===============>..............] - ETA: 48s - loss: 0.2490 - categorical_accuracy: 0.9236
33728/60000 [===============>..............] - ETA: 48s - loss: 0.2488 - categorical_accuracy: 0.9236
33760/60000 [===============>..............] - ETA: 48s - loss: 0.2486 - categorical_accuracy: 0.9237
33792/60000 [===============>..............] - ETA: 48s - loss: 0.2484 - categorical_accuracy: 0.9238
33824/60000 [===============>..............] - ETA: 48s - loss: 0.2483 - categorical_accuracy: 0.9238
33856/60000 [===============>..............] - ETA: 48s - loss: 0.2483 - categorical_accuracy: 0.9238
33888/60000 [===============>..............] - ETA: 48s - loss: 0.2482 - categorical_accuracy: 0.9238
33920/60000 [===============>..............] - ETA: 48s - loss: 0.2485 - categorical_accuracy: 0.9238
33952/60000 [===============>..............] - ETA: 47s - loss: 0.2483 - categorical_accuracy: 0.9239
33984/60000 [===============>..............] - ETA: 47s - loss: 0.2482 - categorical_accuracy: 0.9239
34016/60000 [================>.............] - ETA: 47s - loss: 0.2482 - categorical_accuracy: 0.9239
34048/60000 [================>.............] - ETA: 47s - loss: 0.2480 - categorical_accuracy: 0.9240
34080/60000 [================>.............] - ETA: 47s - loss: 0.2478 - categorical_accuracy: 0.9241
34112/60000 [================>.............] - ETA: 47s - loss: 0.2477 - categorical_accuracy: 0.9241
34144/60000 [================>.............] - ETA: 47s - loss: 0.2475 - categorical_accuracy: 0.9242
34176/60000 [================>.............] - ETA: 47s - loss: 0.2473 - categorical_accuracy: 0.9242
34208/60000 [================>.............] - ETA: 47s - loss: 0.2472 - categorical_accuracy: 0.9242
34240/60000 [================>.............] - ETA: 47s - loss: 0.2470 - categorical_accuracy: 0.9243
34272/60000 [================>.............] - ETA: 47s - loss: 0.2468 - categorical_accuracy: 0.9244
34304/60000 [================>.............] - ETA: 47s - loss: 0.2466 - categorical_accuracy: 0.9244
34336/60000 [================>.............] - ETA: 47s - loss: 0.2464 - categorical_accuracy: 0.9245
34368/60000 [================>.............] - ETA: 47s - loss: 0.2462 - categorical_accuracy: 0.9245
34400/60000 [================>.............] - ETA: 47s - loss: 0.2460 - categorical_accuracy: 0.9246
34432/60000 [================>.............] - ETA: 47s - loss: 0.2459 - categorical_accuracy: 0.9246
34464/60000 [================>.............] - ETA: 47s - loss: 0.2457 - categorical_accuracy: 0.9247
34496/60000 [================>.............] - ETA: 46s - loss: 0.2455 - categorical_accuracy: 0.9247
34528/60000 [================>.............] - ETA: 46s - loss: 0.2454 - categorical_accuracy: 0.9247
34560/60000 [================>.............] - ETA: 46s - loss: 0.2452 - categorical_accuracy: 0.9248
34592/60000 [================>.............] - ETA: 46s - loss: 0.2450 - categorical_accuracy: 0.9248
34624/60000 [================>.............] - ETA: 46s - loss: 0.2449 - categorical_accuracy: 0.9248
34656/60000 [================>.............] - ETA: 46s - loss: 0.2447 - categorical_accuracy: 0.9249
34688/60000 [================>.............] - ETA: 46s - loss: 0.2449 - categorical_accuracy: 0.9250
34720/60000 [================>.............] - ETA: 46s - loss: 0.2447 - categorical_accuracy: 0.9250
34752/60000 [================>.............] - ETA: 46s - loss: 0.2446 - categorical_accuracy: 0.9250
34784/60000 [================>.............] - ETA: 46s - loss: 0.2445 - categorical_accuracy: 0.9251
34816/60000 [================>.............] - ETA: 46s - loss: 0.2444 - categorical_accuracy: 0.9251
34848/60000 [================>.............] - ETA: 46s - loss: 0.2443 - categorical_accuracy: 0.9251
34880/60000 [================>.............] - ETA: 46s - loss: 0.2441 - categorical_accuracy: 0.9252
34912/60000 [================>.............] - ETA: 46s - loss: 0.2438 - categorical_accuracy: 0.9253
34944/60000 [================>.............] - ETA: 46s - loss: 0.2437 - categorical_accuracy: 0.9253
34976/60000 [================>.............] - ETA: 46s - loss: 0.2436 - categorical_accuracy: 0.9253
35008/60000 [================>.............] - ETA: 46s - loss: 0.2434 - categorical_accuracy: 0.9254
35040/60000 [================>.............] - ETA: 45s - loss: 0.2432 - categorical_accuracy: 0.9255
35072/60000 [================>.............] - ETA: 45s - loss: 0.2432 - categorical_accuracy: 0.9255
35104/60000 [================>.............] - ETA: 45s - loss: 0.2430 - categorical_accuracy: 0.9255
35136/60000 [================>.............] - ETA: 45s - loss: 0.2428 - categorical_accuracy: 0.9256
35168/60000 [================>.............] - ETA: 45s - loss: 0.2429 - categorical_accuracy: 0.9256
35200/60000 [================>.............] - ETA: 45s - loss: 0.2427 - categorical_accuracy: 0.9257
35232/60000 [================>.............] - ETA: 45s - loss: 0.2427 - categorical_accuracy: 0.9257
35264/60000 [================>.............] - ETA: 45s - loss: 0.2428 - categorical_accuracy: 0.9256
35296/60000 [================>.............] - ETA: 45s - loss: 0.2426 - categorical_accuracy: 0.9257
35328/60000 [================>.............] - ETA: 45s - loss: 0.2424 - categorical_accuracy: 0.9258
35360/60000 [================>.............] - ETA: 45s - loss: 0.2422 - categorical_accuracy: 0.9258
35392/60000 [================>.............] - ETA: 45s - loss: 0.2421 - categorical_accuracy: 0.9259
35424/60000 [================>.............] - ETA: 45s - loss: 0.2419 - categorical_accuracy: 0.9259
35456/60000 [================>.............] - ETA: 45s - loss: 0.2417 - categorical_accuracy: 0.9260
35488/60000 [================>.............] - ETA: 45s - loss: 0.2416 - categorical_accuracy: 0.9260
35520/60000 [================>.............] - ETA: 45s - loss: 0.2414 - categorical_accuracy: 0.9260
35552/60000 [================>.............] - ETA: 45s - loss: 0.2414 - categorical_accuracy: 0.9261
35584/60000 [================>.............] - ETA: 44s - loss: 0.2413 - categorical_accuracy: 0.9261
35616/60000 [================>.............] - ETA: 44s - loss: 0.2411 - categorical_accuracy: 0.9262
35648/60000 [================>.............] - ETA: 44s - loss: 0.2410 - categorical_accuracy: 0.9262
35680/60000 [================>.............] - ETA: 44s - loss: 0.2408 - categorical_accuracy: 0.9262
35712/60000 [================>.............] - ETA: 44s - loss: 0.2406 - categorical_accuracy: 0.9263
35744/60000 [================>.............] - ETA: 44s - loss: 0.2405 - categorical_accuracy: 0.9263
35776/60000 [================>.............] - ETA: 44s - loss: 0.2407 - categorical_accuracy: 0.9263
35808/60000 [================>.............] - ETA: 44s - loss: 0.2406 - categorical_accuracy: 0.9263
35840/60000 [================>.............] - ETA: 44s - loss: 0.2404 - categorical_accuracy: 0.9264
35872/60000 [================>.............] - ETA: 44s - loss: 0.2403 - categorical_accuracy: 0.9264
35904/60000 [================>.............] - ETA: 44s - loss: 0.2404 - categorical_accuracy: 0.9264
35936/60000 [================>.............] - ETA: 44s - loss: 0.2402 - categorical_accuracy: 0.9264
35968/60000 [================>.............] - ETA: 44s - loss: 0.2401 - categorical_accuracy: 0.9265
36000/60000 [=================>............] - ETA: 44s - loss: 0.2402 - categorical_accuracy: 0.9264
36032/60000 [=================>............] - ETA: 44s - loss: 0.2403 - categorical_accuracy: 0.9265
36064/60000 [=================>............] - ETA: 44s - loss: 0.2401 - categorical_accuracy: 0.9265
36096/60000 [=================>............] - ETA: 44s - loss: 0.2400 - categorical_accuracy: 0.9265
36128/60000 [=================>............] - ETA: 43s - loss: 0.2398 - categorical_accuracy: 0.9266
36160/60000 [=================>............] - ETA: 43s - loss: 0.2397 - categorical_accuracy: 0.9266
36192/60000 [=================>............] - ETA: 43s - loss: 0.2396 - categorical_accuracy: 0.9267
36224/60000 [=================>............] - ETA: 43s - loss: 0.2394 - categorical_accuracy: 0.9267
36256/60000 [=================>............] - ETA: 43s - loss: 0.2392 - categorical_accuracy: 0.9268
36288/60000 [=================>............] - ETA: 43s - loss: 0.2390 - categorical_accuracy: 0.9268
36320/60000 [=================>............] - ETA: 43s - loss: 0.2388 - categorical_accuracy: 0.9269
36352/60000 [=================>............] - ETA: 43s - loss: 0.2388 - categorical_accuracy: 0.9269
36384/60000 [=================>............] - ETA: 43s - loss: 0.2386 - categorical_accuracy: 0.9269
36416/60000 [=================>............] - ETA: 43s - loss: 0.2384 - categorical_accuracy: 0.9270
36448/60000 [=================>............] - ETA: 43s - loss: 0.2383 - categorical_accuracy: 0.9270
36480/60000 [=================>............] - ETA: 43s - loss: 0.2383 - categorical_accuracy: 0.9270
36512/60000 [=================>............] - ETA: 43s - loss: 0.2382 - categorical_accuracy: 0.9270
36544/60000 [=================>............] - ETA: 43s - loss: 0.2381 - categorical_accuracy: 0.9270
36576/60000 [=================>............] - ETA: 43s - loss: 0.2382 - categorical_accuracy: 0.9269
36608/60000 [=================>............] - ETA: 43s - loss: 0.2381 - categorical_accuracy: 0.9270
36640/60000 [=================>............] - ETA: 43s - loss: 0.2380 - categorical_accuracy: 0.9270
36672/60000 [=================>............] - ETA: 42s - loss: 0.2378 - categorical_accuracy: 0.9271
36704/60000 [=================>............] - ETA: 42s - loss: 0.2377 - categorical_accuracy: 0.9271
36736/60000 [=================>............] - ETA: 42s - loss: 0.2378 - categorical_accuracy: 0.9270
36768/60000 [=================>............] - ETA: 42s - loss: 0.2376 - categorical_accuracy: 0.9271
36800/60000 [=================>............] - ETA: 42s - loss: 0.2375 - categorical_accuracy: 0.9271
36832/60000 [=================>............] - ETA: 42s - loss: 0.2375 - categorical_accuracy: 0.9271
36864/60000 [=================>............] - ETA: 42s - loss: 0.2380 - categorical_accuracy: 0.9270
36896/60000 [=================>............] - ETA: 42s - loss: 0.2379 - categorical_accuracy: 0.9270
36928/60000 [=================>............] - ETA: 42s - loss: 0.2378 - categorical_accuracy: 0.9270
36960/60000 [=================>............] - ETA: 42s - loss: 0.2376 - categorical_accuracy: 0.9271
36992/60000 [=================>............] - ETA: 42s - loss: 0.2375 - categorical_accuracy: 0.9271
37024/60000 [=================>............] - ETA: 42s - loss: 0.2373 - categorical_accuracy: 0.9272
37056/60000 [=================>............] - ETA: 42s - loss: 0.2371 - categorical_accuracy: 0.9272
37088/60000 [=================>............] - ETA: 42s - loss: 0.2370 - categorical_accuracy: 0.9273
37120/60000 [=================>............] - ETA: 42s - loss: 0.2369 - categorical_accuracy: 0.9273
37152/60000 [=================>............] - ETA: 42s - loss: 0.2368 - categorical_accuracy: 0.9273
37184/60000 [=================>............] - ETA: 42s - loss: 0.2369 - categorical_accuracy: 0.9273
37216/60000 [=================>............] - ETA: 41s - loss: 0.2368 - categorical_accuracy: 0.9273
37248/60000 [=================>............] - ETA: 41s - loss: 0.2366 - categorical_accuracy: 0.9274
37280/60000 [=================>............] - ETA: 41s - loss: 0.2365 - categorical_accuracy: 0.9274
37312/60000 [=================>............] - ETA: 41s - loss: 0.2365 - categorical_accuracy: 0.9274
37344/60000 [=================>............] - ETA: 41s - loss: 0.2363 - categorical_accuracy: 0.9275
37376/60000 [=================>............] - ETA: 41s - loss: 0.2363 - categorical_accuracy: 0.9275
37408/60000 [=================>............] - ETA: 41s - loss: 0.2362 - categorical_accuracy: 0.9275
37440/60000 [=================>............] - ETA: 41s - loss: 0.2362 - categorical_accuracy: 0.9275
37472/60000 [=================>............] - ETA: 41s - loss: 0.2363 - categorical_accuracy: 0.9274
37504/60000 [=================>............] - ETA: 41s - loss: 0.2362 - categorical_accuracy: 0.9274
37536/60000 [=================>............] - ETA: 41s - loss: 0.2362 - categorical_accuracy: 0.9275
37568/60000 [=================>............] - ETA: 41s - loss: 0.2360 - categorical_accuracy: 0.9275
37600/60000 [=================>............] - ETA: 41s - loss: 0.2358 - categorical_accuracy: 0.9276
37632/60000 [=================>............] - ETA: 41s - loss: 0.2357 - categorical_accuracy: 0.9276
37664/60000 [=================>............] - ETA: 41s - loss: 0.2355 - categorical_accuracy: 0.9277
37696/60000 [=================>............] - ETA: 41s - loss: 0.2353 - categorical_accuracy: 0.9278
37728/60000 [=================>............] - ETA: 41s - loss: 0.2352 - categorical_accuracy: 0.9278
37760/60000 [=================>............] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9278
37792/60000 [=================>............] - ETA: 40s - loss: 0.2349 - categorical_accuracy: 0.9279
37824/60000 [=================>............] - ETA: 40s - loss: 0.2347 - categorical_accuracy: 0.9279
37856/60000 [=================>............] - ETA: 40s - loss: 0.2347 - categorical_accuracy: 0.9279
37888/60000 [=================>............] - ETA: 40s - loss: 0.2347 - categorical_accuracy: 0.9279
37920/60000 [=================>............] - ETA: 40s - loss: 0.2347 - categorical_accuracy: 0.9279
37952/60000 [=================>............] - ETA: 40s - loss: 0.2346 - categorical_accuracy: 0.9279
37984/60000 [=================>............] - ETA: 40s - loss: 0.2345 - categorical_accuracy: 0.9279
38016/60000 [==================>...........] - ETA: 40s - loss: 0.2344 - categorical_accuracy: 0.9280
38048/60000 [==================>...........] - ETA: 40s - loss: 0.2343 - categorical_accuracy: 0.9280
38080/60000 [==================>...........] - ETA: 40s - loss: 0.2341 - categorical_accuracy: 0.9280
38112/60000 [==================>...........] - ETA: 40s - loss: 0.2340 - categorical_accuracy: 0.9281
38144/60000 [==================>...........] - ETA: 40s - loss: 0.2338 - categorical_accuracy: 0.9281
38176/60000 [==================>...........] - ETA: 40s - loss: 0.2337 - categorical_accuracy: 0.9282
38208/60000 [==================>...........] - ETA: 40s - loss: 0.2337 - categorical_accuracy: 0.9282
38240/60000 [==================>...........] - ETA: 40s - loss: 0.2335 - categorical_accuracy: 0.9282
38272/60000 [==================>...........] - ETA: 40s - loss: 0.2334 - categorical_accuracy: 0.9283
38304/60000 [==================>...........] - ETA: 39s - loss: 0.2333 - categorical_accuracy: 0.9283
38336/60000 [==================>...........] - ETA: 39s - loss: 0.2332 - categorical_accuracy: 0.9284
38368/60000 [==================>...........] - ETA: 39s - loss: 0.2331 - categorical_accuracy: 0.9284
38400/60000 [==================>...........] - ETA: 39s - loss: 0.2330 - categorical_accuracy: 0.9284
38432/60000 [==================>...........] - ETA: 39s - loss: 0.2329 - categorical_accuracy: 0.9284
38464/60000 [==================>...........] - ETA: 39s - loss: 0.2328 - categorical_accuracy: 0.9285
38496/60000 [==================>...........] - ETA: 39s - loss: 0.2326 - categorical_accuracy: 0.9285
38528/60000 [==================>...........] - ETA: 39s - loss: 0.2325 - categorical_accuracy: 0.9285
38560/60000 [==================>...........] - ETA: 39s - loss: 0.2323 - categorical_accuracy: 0.9286
38592/60000 [==================>...........] - ETA: 39s - loss: 0.2322 - categorical_accuracy: 0.9286
38624/60000 [==================>...........] - ETA: 39s - loss: 0.2320 - categorical_accuracy: 0.9287
38656/60000 [==================>...........] - ETA: 39s - loss: 0.2319 - categorical_accuracy: 0.9287
38688/60000 [==================>...........] - ETA: 39s - loss: 0.2319 - categorical_accuracy: 0.9287
38720/60000 [==================>...........] - ETA: 39s - loss: 0.2318 - categorical_accuracy: 0.9288
38752/60000 [==================>...........] - ETA: 39s - loss: 0.2317 - categorical_accuracy: 0.9288
38784/60000 [==================>...........] - ETA: 39s - loss: 0.2316 - categorical_accuracy: 0.9288
38816/60000 [==================>...........] - ETA: 39s - loss: 0.2315 - categorical_accuracy: 0.9289
38848/60000 [==================>...........] - ETA: 38s - loss: 0.2313 - categorical_accuracy: 0.9289
38880/60000 [==================>...........] - ETA: 38s - loss: 0.2312 - categorical_accuracy: 0.9289
38912/60000 [==================>...........] - ETA: 38s - loss: 0.2311 - categorical_accuracy: 0.9290
38944/60000 [==================>...........] - ETA: 38s - loss: 0.2309 - categorical_accuracy: 0.9291
38976/60000 [==================>...........] - ETA: 38s - loss: 0.2307 - categorical_accuracy: 0.9291
39008/60000 [==================>...........] - ETA: 38s - loss: 0.2306 - categorical_accuracy: 0.9292
39040/60000 [==================>...........] - ETA: 38s - loss: 0.2304 - categorical_accuracy: 0.9292
39072/60000 [==================>...........] - ETA: 38s - loss: 0.2303 - categorical_accuracy: 0.9293
39104/60000 [==================>...........] - ETA: 38s - loss: 0.2301 - categorical_accuracy: 0.9293
39136/60000 [==================>...........] - ETA: 38s - loss: 0.2299 - categorical_accuracy: 0.9294
39168/60000 [==================>...........] - ETA: 38s - loss: 0.2298 - categorical_accuracy: 0.9294
39200/60000 [==================>...........] - ETA: 38s - loss: 0.2297 - categorical_accuracy: 0.9294
39232/60000 [==================>...........] - ETA: 38s - loss: 0.2296 - categorical_accuracy: 0.9295
39264/60000 [==================>...........] - ETA: 38s - loss: 0.2294 - categorical_accuracy: 0.9295
39296/60000 [==================>...........] - ETA: 38s - loss: 0.2293 - categorical_accuracy: 0.9296
39328/60000 [==================>...........] - ETA: 38s - loss: 0.2291 - categorical_accuracy: 0.9296
39360/60000 [==================>...........] - ETA: 38s - loss: 0.2290 - categorical_accuracy: 0.9297
39392/60000 [==================>...........] - ETA: 37s - loss: 0.2289 - categorical_accuracy: 0.9297
39424/60000 [==================>...........] - ETA: 37s - loss: 0.2288 - categorical_accuracy: 0.9297
39456/60000 [==================>...........] - ETA: 37s - loss: 0.2288 - categorical_accuracy: 0.9297
39488/60000 [==================>...........] - ETA: 37s - loss: 0.2288 - categorical_accuracy: 0.9298
39520/60000 [==================>...........] - ETA: 37s - loss: 0.2286 - categorical_accuracy: 0.9298
39552/60000 [==================>...........] - ETA: 37s - loss: 0.2286 - categorical_accuracy: 0.9298
39584/60000 [==================>...........] - ETA: 37s - loss: 0.2286 - categorical_accuracy: 0.9298
39616/60000 [==================>...........] - ETA: 37s - loss: 0.2284 - categorical_accuracy: 0.9299
39648/60000 [==================>...........] - ETA: 37s - loss: 0.2283 - categorical_accuracy: 0.9299
39680/60000 [==================>...........] - ETA: 37s - loss: 0.2282 - categorical_accuracy: 0.9299
39712/60000 [==================>...........] - ETA: 37s - loss: 0.2281 - categorical_accuracy: 0.9299
39744/60000 [==================>...........] - ETA: 37s - loss: 0.2281 - categorical_accuracy: 0.9299
39776/60000 [==================>...........] - ETA: 37s - loss: 0.2280 - categorical_accuracy: 0.9299
39808/60000 [==================>...........] - ETA: 37s - loss: 0.2278 - categorical_accuracy: 0.9300
39840/60000 [==================>...........] - ETA: 37s - loss: 0.2276 - categorical_accuracy: 0.9300
39872/60000 [==================>...........] - ETA: 37s - loss: 0.2277 - categorical_accuracy: 0.9301
39904/60000 [==================>...........] - ETA: 37s - loss: 0.2276 - categorical_accuracy: 0.9301
39936/60000 [==================>...........] - ETA: 36s - loss: 0.2275 - categorical_accuracy: 0.9301
39968/60000 [==================>...........] - ETA: 36s - loss: 0.2274 - categorical_accuracy: 0.9301
40000/60000 [===================>..........] - ETA: 36s - loss: 0.2272 - categorical_accuracy: 0.9302
40032/60000 [===================>..........] - ETA: 36s - loss: 0.2271 - categorical_accuracy: 0.9302
40064/60000 [===================>..........] - ETA: 36s - loss: 0.2271 - categorical_accuracy: 0.9302
40096/60000 [===================>..........] - ETA: 36s - loss: 0.2269 - categorical_accuracy: 0.9303
40128/60000 [===================>..........] - ETA: 36s - loss: 0.2268 - categorical_accuracy: 0.9303
40160/60000 [===================>..........] - ETA: 36s - loss: 0.2269 - categorical_accuracy: 0.9303
40192/60000 [===================>..........] - ETA: 36s - loss: 0.2267 - categorical_accuracy: 0.9304
40224/60000 [===================>..........] - ETA: 36s - loss: 0.2266 - categorical_accuracy: 0.9304
40256/60000 [===================>..........] - ETA: 36s - loss: 0.2265 - categorical_accuracy: 0.9304
40288/60000 [===================>..........] - ETA: 36s - loss: 0.2264 - categorical_accuracy: 0.9304
40320/60000 [===================>..........] - ETA: 36s - loss: 0.2263 - categorical_accuracy: 0.9305
40352/60000 [===================>..........] - ETA: 36s - loss: 0.2261 - categorical_accuracy: 0.9305
40384/60000 [===================>..........] - ETA: 36s - loss: 0.2262 - categorical_accuracy: 0.9305
40416/60000 [===================>..........] - ETA: 36s - loss: 0.2260 - categorical_accuracy: 0.9306
40448/60000 [===================>..........] - ETA: 36s - loss: 0.2259 - categorical_accuracy: 0.9306
40480/60000 [===================>..........] - ETA: 35s - loss: 0.2258 - categorical_accuracy: 0.9307
40512/60000 [===================>..........] - ETA: 35s - loss: 0.2256 - categorical_accuracy: 0.9307
40544/60000 [===================>..........] - ETA: 35s - loss: 0.2254 - categorical_accuracy: 0.9308
40576/60000 [===================>..........] - ETA: 35s - loss: 0.2253 - categorical_accuracy: 0.9308
40608/60000 [===================>..........] - ETA: 35s - loss: 0.2252 - categorical_accuracy: 0.9309
40640/60000 [===================>..........] - ETA: 35s - loss: 0.2251 - categorical_accuracy: 0.9309
40672/60000 [===================>..........] - ETA: 35s - loss: 0.2252 - categorical_accuracy: 0.9308
40704/60000 [===================>..........] - ETA: 35s - loss: 0.2251 - categorical_accuracy: 0.9309
40736/60000 [===================>..........] - ETA: 35s - loss: 0.2249 - categorical_accuracy: 0.9309
40768/60000 [===================>..........] - ETA: 35s - loss: 0.2248 - categorical_accuracy: 0.9310
40800/60000 [===================>..........] - ETA: 35s - loss: 0.2246 - categorical_accuracy: 0.9310
40832/60000 [===================>..........] - ETA: 35s - loss: 0.2245 - categorical_accuracy: 0.9311
40864/60000 [===================>..........] - ETA: 35s - loss: 0.2243 - categorical_accuracy: 0.9311
40896/60000 [===================>..........] - ETA: 35s - loss: 0.2242 - categorical_accuracy: 0.9311
40928/60000 [===================>..........] - ETA: 35s - loss: 0.2242 - categorical_accuracy: 0.9311
40960/60000 [===================>..........] - ETA: 35s - loss: 0.2244 - categorical_accuracy: 0.9311
40992/60000 [===================>..........] - ETA: 35s - loss: 0.2244 - categorical_accuracy: 0.9311
41024/60000 [===================>..........] - ETA: 34s - loss: 0.2244 - categorical_accuracy: 0.9311
41056/60000 [===================>..........] - ETA: 34s - loss: 0.2243 - categorical_accuracy: 0.9312
41088/60000 [===================>..........] - ETA: 34s - loss: 0.2242 - categorical_accuracy: 0.9312
41120/60000 [===================>..........] - ETA: 34s - loss: 0.2242 - categorical_accuracy: 0.9312
41152/60000 [===================>..........] - ETA: 34s - loss: 0.2241 - categorical_accuracy: 0.9313
41184/60000 [===================>..........] - ETA: 34s - loss: 0.2241 - categorical_accuracy: 0.9313
41216/60000 [===================>..........] - ETA: 34s - loss: 0.2239 - categorical_accuracy: 0.9313
41248/60000 [===================>..........] - ETA: 34s - loss: 0.2239 - categorical_accuracy: 0.9314
41280/60000 [===================>..........] - ETA: 34s - loss: 0.2237 - categorical_accuracy: 0.9314
41312/60000 [===================>..........] - ETA: 34s - loss: 0.2236 - categorical_accuracy: 0.9315
41344/60000 [===================>..........] - ETA: 34s - loss: 0.2234 - categorical_accuracy: 0.9315
41376/60000 [===================>..........] - ETA: 34s - loss: 0.2234 - categorical_accuracy: 0.9316
41408/60000 [===================>..........] - ETA: 34s - loss: 0.2233 - categorical_accuracy: 0.9316
41440/60000 [===================>..........] - ETA: 34s - loss: 0.2233 - categorical_accuracy: 0.9316
41472/60000 [===================>..........] - ETA: 34s - loss: 0.2232 - categorical_accuracy: 0.9316
41504/60000 [===================>..........] - ETA: 34s - loss: 0.2230 - categorical_accuracy: 0.9317
41536/60000 [===================>..........] - ETA: 34s - loss: 0.2231 - categorical_accuracy: 0.9317
41568/60000 [===================>..........] - ETA: 33s - loss: 0.2230 - categorical_accuracy: 0.9317
41600/60000 [===================>..........] - ETA: 33s - loss: 0.2231 - categorical_accuracy: 0.9317
41632/60000 [===================>..........] - ETA: 33s - loss: 0.2229 - categorical_accuracy: 0.9317
41664/60000 [===================>..........] - ETA: 33s - loss: 0.2229 - categorical_accuracy: 0.9317
41696/60000 [===================>..........] - ETA: 33s - loss: 0.2228 - categorical_accuracy: 0.9317
41728/60000 [===================>..........] - ETA: 33s - loss: 0.2227 - categorical_accuracy: 0.9318
41760/60000 [===================>..........] - ETA: 33s - loss: 0.2226 - categorical_accuracy: 0.9318
41792/60000 [===================>..........] - ETA: 33s - loss: 0.2225 - categorical_accuracy: 0.9319
41824/60000 [===================>..........] - ETA: 33s - loss: 0.2225 - categorical_accuracy: 0.9318
41856/60000 [===================>..........] - ETA: 33s - loss: 0.2223 - categorical_accuracy: 0.9319
41888/60000 [===================>..........] - ETA: 33s - loss: 0.2222 - categorical_accuracy: 0.9319
41920/60000 [===================>..........] - ETA: 33s - loss: 0.2221 - categorical_accuracy: 0.9320
41952/60000 [===================>..........] - ETA: 33s - loss: 0.2220 - categorical_accuracy: 0.9320
41984/60000 [===================>..........] - ETA: 33s - loss: 0.2218 - categorical_accuracy: 0.9320
42016/60000 [====================>.........] - ETA: 33s - loss: 0.2217 - categorical_accuracy: 0.9321
42048/60000 [====================>.........] - ETA: 33s - loss: 0.2216 - categorical_accuracy: 0.9321
42080/60000 [====================>.........] - ETA: 33s - loss: 0.2214 - categorical_accuracy: 0.9322
42112/60000 [====================>.........] - ETA: 32s - loss: 0.2213 - categorical_accuracy: 0.9322
42144/60000 [====================>.........] - ETA: 32s - loss: 0.2211 - categorical_accuracy: 0.9323
42176/60000 [====================>.........] - ETA: 32s - loss: 0.2212 - categorical_accuracy: 0.9322
42208/60000 [====================>.........] - ETA: 32s - loss: 0.2212 - categorical_accuracy: 0.9323
42240/60000 [====================>.........] - ETA: 32s - loss: 0.2211 - categorical_accuracy: 0.9323
42272/60000 [====================>.........] - ETA: 32s - loss: 0.2210 - categorical_accuracy: 0.9323
42304/60000 [====================>.........] - ETA: 32s - loss: 0.2208 - categorical_accuracy: 0.9324
42336/60000 [====================>.........] - ETA: 32s - loss: 0.2209 - categorical_accuracy: 0.9324
42368/60000 [====================>.........] - ETA: 32s - loss: 0.2207 - categorical_accuracy: 0.9324
42400/60000 [====================>.........] - ETA: 32s - loss: 0.2207 - categorical_accuracy: 0.9324
42432/60000 [====================>.........] - ETA: 32s - loss: 0.2207 - categorical_accuracy: 0.9324
42464/60000 [====================>.........] - ETA: 32s - loss: 0.2206 - categorical_accuracy: 0.9324
42496/60000 [====================>.........] - ETA: 32s - loss: 0.2206 - categorical_accuracy: 0.9324
42528/60000 [====================>.........] - ETA: 32s - loss: 0.2204 - categorical_accuracy: 0.9324
42560/60000 [====================>.........] - ETA: 32s - loss: 0.2203 - categorical_accuracy: 0.9324
42592/60000 [====================>.........] - ETA: 32s - loss: 0.2203 - categorical_accuracy: 0.9325
42624/60000 [====================>.........] - ETA: 32s - loss: 0.2202 - categorical_accuracy: 0.9325
42656/60000 [====================>.........] - ETA: 31s - loss: 0.2203 - categorical_accuracy: 0.9325
42688/60000 [====================>.........] - ETA: 31s - loss: 0.2202 - categorical_accuracy: 0.9325
42720/60000 [====================>.........] - ETA: 31s - loss: 0.2204 - categorical_accuracy: 0.9324
42752/60000 [====================>.........] - ETA: 31s - loss: 0.2204 - categorical_accuracy: 0.9324
42784/60000 [====================>.........] - ETA: 31s - loss: 0.2203 - categorical_accuracy: 0.9325
42816/60000 [====================>.........] - ETA: 31s - loss: 0.2204 - categorical_accuracy: 0.9325
42848/60000 [====================>.........] - ETA: 31s - loss: 0.2202 - categorical_accuracy: 0.9325
42880/60000 [====================>.........] - ETA: 31s - loss: 0.2201 - categorical_accuracy: 0.9326
42912/60000 [====================>.........] - ETA: 31s - loss: 0.2200 - categorical_accuracy: 0.9326
42944/60000 [====================>.........] - ETA: 31s - loss: 0.2200 - categorical_accuracy: 0.9326
42976/60000 [====================>.........] - ETA: 31s - loss: 0.2199 - categorical_accuracy: 0.9326
43008/60000 [====================>.........] - ETA: 31s - loss: 0.2198 - categorical_accuracy: 0.9327
43040/60000 [====================>.........] - ETA: 31s - loss: 0.2197 - categorical_accuracy: 0.9327
43072/60000 [====================>.........] - ETA: 31s - loss: 0.2196 - categorical_accuracy: 0.9327
43104/60000 [====================>.........] - ETA: 31s - loss: 0.2195 - categorical_accuracy: 0.9327
43136/60000 [====================>.........] - ETA: 31s - loss: 0.2194 - categorical_accuracy: 0.9328
43168/60000 [====================>.........] - ETA: 30s - loss: 0.2192 - categorical_accuracy: 0.9328
43200/60000 [====================>.........] - ETA: 30s - loss: 0.2191 - categorical_accuracy: 0.9329
43232/60000 [====================>.........] - ETA: 30s - loss: 0.2190 - categorical_accuracy: 0.9329
43264/60000 [====================>.........] - ETA: 30s - loss: 0.2189 - categorical_accuracy: 0.9330
43296/60000 [====================>.........] - ETA: 30s - loss: 0.2188 - categorical_accuracy: 0.9330
43328/60000 [====================>.........] - ETA: 30s - loss: 0.2187 - categorical_accuracy: 0.9331
43360/60000 [====================>.........] - ETA: 30s - loss: 0.2187 - categorical_accuracy: 0.9330
43392/60000 [====================>.........] - ETA: 30s - loss: 0.2186 - categorical_accuracy: 0.9331
43424/60000 [====================>.........] - ETA: 30s - loss: 0.2185 - categorical_accuracy: 0.9331
43456/60000 [====================>.........] - ETA: 30s - loss: 0.2183 - categorical_accuracy: 0.9332
43488/60000 [====================>.........] - ETA: 30s - loss: 0.2182 - categorical_accuracy: 0.9332
43520/60000 [====================>.........] - ETA: 30s - loss: 0.2182 - categorical_accuracy: 0.9332
43552/60000 [====================>.........] - ETA: 30s - loss: 0.2181 - categorical_accuracy: 0.9332
43584/60000 [====================>.........] - ETA: 30s - loss: 0.2180 - categorical_accuracy: 0.9333
43616/60000 [====================>.........] - ETA: 30s - loss: 0.2178 - categorical_accuracy: 0.9333
43648/60000 [====================>.........] - ETA: 30s - loss: 0.2178 - categorical_accuracy: 0.9333
43680/60000 [====================>.........] - ETA: 30s - loss: 0.2177 - categorical_accuracy: 0.9334
43712/60000 [====================>.........] - ETA: 29s - loss: 0.2177 - categorical_accuracy: 0.9334
43744/60000 [====================>.........] - ETA: 29s - loss: 0.2178 - categorical_accuracy: 0.9334
43776/60000 [====================>.........] - ETA: 29s - loss: 0.2177 - categorical_accuracy: 0.9334
43808/60000 [====================>.........] - ETA: 29s - loss: 0.2175 - categorical_accuracy: 0.9335
43840/60000 [====================>.........] - ETA: 29s - loss: 0.2174 - categorical_accuracy: 0.9335
43872/60000 [====================>.........] - ETA: 29s - loss: 0.2174 - categorical_accuracy: 0.9335
43904/60000 [====================>.........] - ETA: 29s - loss: 0.2174 - categorical_accuracy: 0.9335
43936/60000 [====================>.........] - ETA: 29s - loss: 0.2173 - categorical_accuracy: 0.9335
43968/60000 [====================>.........] - ETA: 29s - loss: 0.2171 - categorical_accuracy: 0.9336
44000/60000 [=====================>........] - ETA: 29s - loss: 0.2170 - categorical_accuracy: 0.9336
44032/60000 [=====================>........] - ETA: 29s - loss: 0.2169 - categorical_accuracy: 0.9336
44064/60000 [=====================>........] - ETA: 29s - loss: 0.2168 - categorical_accuracy: 0.9336
44096/60000 [=====================>........] - ETA: 29s - loss: 0.2168 - categorical_accuracy: 0.9336
44128/60000 [=====================>........] - ETA: 29s - loss: 0.2168 - categorical_accuracy: 0.9336
44160/60000 [=====================>........] - ETA: 29s - loss: 0.2167 - categorical_accuracy: 0.9337
44192/60000 [=====================>........] - ETA: 29s - loss: 0.2166 - categorical_accuracy: 0.9337
44224/60000 [=====================>........] - ETA: 29s - loss: 0.2164 - categorical_accuracy: 0.9338
44256/60000 [=====================>........] - ETA: 29s - loss: 0.2164 - categorical_accuracy: 0.9338
44288/60000 [=====================>........] - ETA: 28s - loss: 0.2163 - categorical_accuracy: 0.9338
44320/60000 [=====================>........] - ETA: 28s - loss: 0.2161 - categorical_accuracy: 0.9338
44352/60000 [=====================>........] - ETA: 28s - loss: 0.2161 - categorical_accuracy: 0.9339
44384/60000 [=====================>........] - ETA: 28s - loss: 0.2159 - categorical_accuracy: 0.9339
44416/60000 [=====================>........] - ETA: 28s - loss: 0.2159 - categorical_accuracy: 0.9339
44448/60000 [=====================>........] - ETA: 28s - loss: 0.2159 - categorical_accuracy: 0.9340
44480/60000 [=====================>........] - ETA: 28s - loss: 0.2157 - categorical_accuracy: 0.9340
44512/60000 [=====================>........] - ETA: 28s - loss: 0.2159 - categorical_accuracy: 0.9340
44544/60000 [=====================>........] - ETA: 28s - loss: 0.2158 - categorical_accuracy: 0.9340
44576/60000 [=====================>........] - ETA: 28s - loss: 0.2158 - categorical_accuracy: 0.9340
44608/60000 [=====================>........] - ETA: 28s - loss: 0.2157 - categorical_accuracy: 0.9340
44640/60000 [=====================>........] - ETA: 28s - loss: 0.2155 - categorical_accuracy: 0.9341
44672/60000 [=====================>........] - ETA: 28s - loss: 0.2154 - categorical_accuracy: 0.9341
44704/60000 [=====================>........] - ETA: 28s - loss: 0.2153 - categorical_accuracy: 0.9342
44736/60000 [=====================>........] - ETA: 28s - loss: 0.2151 - categorical_accuracy: 0.9342
44768/60000 [=====================>........] - ETA: 28s - loss: 0.2151 - categorical_accuracy: 0.9342
44800/60000 [=====================>........] - ETA: 28s - loss: 0.2151 - categorical_accuracy: 0.9342
44832/60000 [=====================>........] - ETA: 27s - loss: 0.2151 - categorical_accuracy: 0.9343
44864/60000 [=====================>........] - ETA: 27s - loss: 0.2151 - categorical_accuracy: 0.9343
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2149 - categorical_accuracy: 0.9343
44928/60000 [=====================>........] - ETA: 27s - loss: 0.2148 - categorical_accuracy: 0.9344
44960/60000 [=====================>........] - ETA: 27s - loss: 0.2147 - categorical_accuracy: 0.9344
44992/60000 [=====================>........] - ETA: 27s - loss: 0.2146 - categorical_accuracy: 0.9344
45024/60000 [=====================>........] - ETA: 27s - loss: 0.2145 - categorical_accuracy: 0.9345
45056/60000 [=====================>........] - ETA: 27s - loss: 0.2144 - categorical_accuracy: 0.9345
45088/60000 [=====================>........] - ETA: 27s - loss: 0.2142 - categorical_accuracy: 0.9346
45120/60000 [=====================>........] - ETA: 27s - loss: 0.2144 - categorical_accuracy: 0.9346
45152/60000 [=====================>........] - ETA: 27s - loss: 0.2143 - categorical_accuracy: 0.9345
45184/60000 [=====================>........] - ETA: 27s - loss: 0.2142 - categorical_accuracy: 0.9346
45216/60000 [=====================>........] - ETA: 27s - loss: 0.2141 - categorical_accuracy: 0.9346
45248/60000 [=====================>........] - ETA: 27s - loss: 0.2140 - categorical_accuracy: 0.9346
45280/60000 [=====================>........] - ETA: 27s - loss: 0.2139 - categorical_accuracy: 0.9347
45312/60000 [=====================>........] - ETA: 27s - loss: 0.2138 - categorical_accuracy: 0.9347
45344/60000 [=====================>........] - ETA: 27s - loss: 0.2137 - categorical_accuracy: 0.9347
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2137 - categorical_accuracy: 0.9347
45408/60000 [=====================>........] - ETA: 26s - loss: 0.2135 - categorical_accuracy: 0.9348
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2134 - categorical_accuracy: 0.9348
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2134 - categorical_accuracy: 0.9348
45504/60000 [=====================>........] - ETA: 26s - loss: 0.2134 - categorical_accuracy: 0.9348
45536/60000 [=====================>........] - ETA: 26s - loss: 0.2135 - categorical_accuracy: 0.9348
45568/60000 [=====================>........] - ETA: 26s - loss: 0.2134 - categorical_accuracy: 0.9348
45600/60000 [=====================>........] - ETA: 26s - loss: 0.2133 - categorical_accuracy: 0.9348
45632/60000 [=====================>........] - ETA: 26s - loss: 0.2131 - categorical_accuracy: 0.9349
45664/60000 [=====================>........] - ETA: 26s - loss: 0.2130 - categorical_accuracy: 0.9349
45696/60000 [=====================>........] - ETA: 26s - loss: 0.2129 - categorical_accuracy: 0.9349
45728/60000 [=====================>........] - ETA: 26s - loss: 0.2128 - categorical_accuracy: 0.9350
45760/60000 [=====================>........] - ETA: 26s - loss: 0.2127 - categorical_accuracy: 0.9350
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2126 - categorical_accuracy: 0.9350
45824/60000 [=====================>........] - ETA: 26s - loss: 0.2125 - categorical_accuracy: 0.9351
45856/60000 [=====================>........] - ETA: 26s - loss: 0.2124 - categorical_accuracy: 0.9351
45888/60000 [=====================>........] - ETA: 26s - loss: 0.2123 - categorical_accuracy: 0.9351
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2122 - categorical_accuracy: 0.9352
45952/60000 [=====================>........] - ETA: 25s - loss: 0.2122 - categorical_accuracy: 0.9352
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2120 - categorical_accuracy: 0.9353
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2119 - categorical_accuracy: 0.9353
46048/60000 [======================>.......] - ETA: 25s - loss: 0.2118 - categorical_accuracy: 0.9353
46080/60000 [======================>.......] - ETA: 25s - loss: 0.2117 - categorical_accuracy: 0.9354
46112/60000 [======================>.......] - ETA: 25s - loss: 0.2116 - categorical_accuracy: 0.9354
46144/60000 [======================>.......] - ETA: 25s - loss: 0.2115 - categorical_accuracy: 0.9354
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2114 - categorical_accuracy: 0.9354
46208/60000 [======================>.......] - ETA: 25s - loss: 0.2114 - categorical_accuracy: 0.9354
46240/60000 [======================>.......] - ETA: 25s - loss: 0.2114 - categorical_accuracy: 0.9354
46272/60000 [======================>.......] - ETA: 25s - loss: 0.2113 - categorical_accuracy: 0.9354
46304/60000 [======================>.......] - ETA: 25s - loss: 0.2112 - categorical_accuracy: 0.9354
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2113 - categorical_accuracy: 0.9354
46368/60000 [======================>.......] - ETA: 25s - loss: 0.2112 - categorical_accuracy: 0.9355
46400/60000 [======================>.......] - ETA: 25s - loss: 0.2110 - categorical_accuracy: 0.9355
46432/60000 [======================>.......] - ETA: 25s - loss: 0.2111 - categorical_accuracy: 0.9355
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2110 - categorical_accuracy: 0.9355
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2110 - categorical_accuracy: 0.9356
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2109 - categorical_accuracy: 0.9356
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2109 - categorical_accuracy: 0.9356
46592/60000 [======================>.......] - ETA: 24s - loss: 0.2107 - categorical_accuracy: 0.9357
46624/60000 [======================>.......] - ETA: 24s - loss: 0.2107 - categorical_accuracy: 0.9357
46656/60000 [======================>.......] - ETA: 24s - loss: 0.2106 - categorical_accuracy: 0.9357
46688/60000 [======================>.......] - ETA: 24s - loss: 0.2106 - categorical_accuracy: 0.9357
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2105 - categorical_accuracy: 0.9357
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2104 - categorical_accuracy: 0.9358
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2103 - categorical_accuracy: 0.9358
46816/60000 [======================>.......] - ETA: 24s - loss: 0.2103 - categorical_accuracy: 0.9358
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2103 - categorical_accuracy: 0.9358
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2102 - categorical_accuracy: 0.9359
46912/60000 [======================>.......] - ETA: 24s - loss: 0.2102 - categorical_accuracy: 0.9359
46944/60000 [======================>.......] - ETA: 24s - loss: 0.2103 - categorical_accuracy: 0.9359
46976/60000 [======================>.......] - ETA: 24s - loss: 0.2102 - categorical_accuracy: 0.9359
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2102 - categorical_accuracy: 0.9359
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2101 - categorical_accuracy: 0.9359
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2100 - categorical_accuracy: 0.9359
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2099 - categorical_accuracy: 0.9360
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2099 - categorical_accuracy: 0.9359
47168/60000 [======================>.......] - ETA: 23s - loss: 0.2099 - categorical_accuracy: 0.9359
47200/60000 [======================>.......] - ETA: 23s - loss: 0.2098 - categorical_accuracy: 0.9360
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2097 - categorical_accuracy: 0.9360
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2096 - categorical_accuracy: 0.9360
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2095 - categorical_accuracy: 0.9361
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2095 - categorical_accuracy: 0.9361
47360/60000 [======================>.......] - ETA: 23s - loss: 0.2094 - categorical_accuracy: 0.9361
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2093 - categorical_accuracy: 0.9361
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2092 - categorical_accuracy: 0.9362
47456/60000 [======================>.......] - ETA: 23s - loss: 0.2091 - categorical_accuracy: 0.9362
47488/60000 [======================>.......] - ETA: 23s - loss: 0.2090 - categorical_accuracy: 0.9362
47520/60000 [======================>.......] - ETA: 23s - loss: 0.2089 - categorical_accuracy: 0.9362
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2089 - categorical_accuracy: 0.9362
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2088 - categorical_accuracy: 0.9363
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2087 - categorical_accuracy: 0.9363
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2087 - categorical_accuracy: 0.9363
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2086 - categorical_accuracy: 0.9363
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2085 - categorical_accuracy: 0.9363
47744/60000 [======================>.......] - ETA: 22s - loss: 0.2084 - categorical_accuracy: 0.9364
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2083 - categorical_accuracy: 0.9364
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2082 - categorical_accuracy: 0.9364
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2081 - categorical_accuracy: 0.9365
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2080 - categorical_accuracy: 0.9365
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2080 - categorical_accuracy: 0.9365
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2079 - categorical_accuracy: 0.9366
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2077 - categorical_accuracy: 0.9366
48000/60000 [=======================>......] - ETA: 22s - loss: 0.2078 - categorical_accuracy: 0.9366
48032/60000 [=======================>......] - ETA: 22s - loss: 0.2077 - categorical_accuracy: 0.9366
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2076 - categorical_accuracy: 0.9367
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2075 - categorical_accuracy: 0.9367
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2074 - categorical_accuracy: 0.9367
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2073 - categorical_accuracy: 0.9368
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2072 - categorical_accuracy: 0.9368
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2071 - categorical_accuracy: 0.9369
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2070 - categorical_accuracy: 0.9369
48288/60000 [=======================>......] - ETA: 21s - loss: 0.2069 - categorical_accuracy: 0.9369
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2068 - categorical_accuracy: 0.9370
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2066 - categorical_accuracy: 0.9370
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2065 - categorical_accuracy: 0.9370
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2064 - categorical_accuracy: 0.9371
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2064 - categorical_accuracy: 0.9371
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2063 - categorical_accuracy: 0.9371
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2063 - categorical_accuracy: 0.9371
48544/60000 [=======================>......] - ETA: 21s - loss: 0.2062 - categorical_accuracy: 0.9372
48576/60000 [=======================>......] - ETA: 21s - loss: 0.2062 - categorical_accuracy: 0.9372
48608/60000 [=======================>......] - ETA: 21s - loss: 0.2061 - categorical_accuracy: 0.9372
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2060 - categorical_accuracy: 0.9372
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2059 - categorical_accuracy: 0.9373
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2058 - categorical_accuracy: 0.9373
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2057 - categorical_accuracy: 0.9373
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2056 - categorical_accuracy: 0.9373
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2055 - categorical_accuracy: 0.9374
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2055 - categorical_accuracy: 0.9374
48864/60000 [=======================>......] - ETA: 20s - loss: 0.2054 - categorical_accuracy: 0.9374
48896/60000 [=======================>......] - ETA: 20s - loss: 0.2052 - categorical_accuracy: 0.9374
48928/60000 [=======================>......] - ETA: 20s - loss: 0.2051 - categorical_accuracy: 0.9375
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2051 - categorical_accuracy: 0.9375
48992/60000 [=======================>......] - ETA: 20s - loss: 0.2053 - categorical_accuracy: 0.9375
49024/60000 [=======================>......] - ETA: 20s - loss: 0.2053 - categorical_accuracy: 0.9375
49056/60000 [=======================>......] - ETA: 20s - loss: 0.2052 - categorical_accuracy: 0.9375
49088/60000 [=======================>......] - ETA: 20s - loss: 0.2051 - categorical_accuracy: 0.9375
49120/60000 [=======================>......] - ETA: 20s - loss: 0.2049 - categorical_accuracy: 0.9376
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2048 - categorical_accuracy: 0.9376
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2047 - categorical_accuracy: 0.9376
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2048 - categorical_accuracy: 0.9376
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2047 - categorical_accuracy: 0.9376
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2048 - categorical_accuracy: 0.9376
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2047 - categorical_accuracy: 0.9377
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2046 - categorical_accuracy: 0.9377
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2044 - categorical_accuracy: 0.9377
49408/60000 [=======================>......] - ETA: 19s - loss: 0.2043 - categorical_accuracy: 0.9378
49440/60000 [=======================>......] - ETA: 19s - loss: 0.2042 - categorical_accuracy: 0.9378
49472/60000 [=======================>......] - ETA: 19s - loss: 0.2042 - categorical_accuracy: 0.9378
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2041 - categorical_accuracy: 0.9379
49536/60000 [=======================>......] - ETA: 19s - loss: 0.2040 - categorical_accuracy: 0.9379
49568/60000 [=======================>......] - ETA: 19s - loss: 0.2040 - categorical_accuracy: 0.9379
49600/60000 [=======================>......] - ETA: 19s - loss: 0.2039 - categorical_accuracy: 0.9379
49632/60000 [=======================>......] - ETA: 19s - loss: 0.2038 - categorical_accuracy: 0.9380
49664/60000 [=======================>......] - ETA: 19s - loss: 0.2038 - categorical_accuracy: 0.9380
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2038 - categorical_accuracy: 0.9380
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2037 - categorical_accuracy: 0.9380
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2037 - categorical_accuracy: 0.9380
49792/60000 [=======================>......] - ETA: 18s - loss: 0.2036 - categorical_accuracy: 0.9381
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2035 - categorical_accuracy: 0.9381
49856/60000 [=======================>......] - ETA: 18s - loss: 0.2034 - categorical_accuracy: 0.9382
49888/60000 [=======================>......] - ETA: 18s - loss: 0.2033 - categorical_accuracy: 0.9382
49920/60000 [=======================>......] - ETA: 18s - loss: 0.2033 - categorical_accuracy: 0.9382
49952/60000 [=======================>......] - ETA: 18s - loss: 0.2031 - categorical_accuracy: 0.9382
49984/60000 [=======================>......] - ETA: 18s - loss: 0.2030 - categorical_accuracy: 0.9383
50016/60000 [========================>.....] - ETA: 18s - loss: 0.2030 - categorical_accuracy: 0.9383
50048/60000 [========================>.....] - ETA: 18s - loss: 0.2029 - categorical_accuracy: 0.9383
50080/60000 [========================>.....] - ETA: 18s - loss: 0.2029 - categorical_accuracy: 0.9383
50112/60000 [========================>.....] - ETA: 18s - loss: 0.2029 - categorical_accuracy: 0.9383
50144/60000 [========================>.....] - ETA: 18s - loss: 0.2028 - categorical_accuracy: 0.9384
50176/60000 [========================>.....] - ETA: 18s - loss: 0.2027 - categorical_accuracy: 0.9384
50208/60000 [========================>.....] - ETA: 18s - loss: 0.2027 - categorical_accuracy: 0.9384
50240/60000 [========================>.....] - ETA: 17s - loss: 0.2026 - categorical_accuracy: 0.9384
50272/60000 [========================>.....] - ETA: 17s - loss: 0.2025 - categorical_accuracy: 0.9384
50304/60000 [========================>.....] - ETA: 17s - loss: 0.2025 - categorical_accuracy: 0.9384
50336/60000 [========================>.....] - ETA: 17s - loss: 0.2025 - categorical_accuracy: 0.9384
50368/60000 [========================>.....] - ETA: 17s - loss: 0.2024 - categorical_accuracy: 0.9384
50400/60000 [========================>.....] - ETA: 17s - loss: 0.2023 - categorical_accuracy: 0.9385
50432/60000 [========================>.....] - ETA: 17s - loss: 0.2022 - categorical_accuracy: 0.9385
50464/60000 [========================>.....] - ETA: 17s - loss: 0.2021 - categorical_accuracy: 0.9385
50496/60000 [========================>.....] - ETA: 17s - loss: 0.2020 - categorical_accuracy: 0.9386
50528/60000 [========================>.....] - ETA: 17s - loss: 0.2021 - categorical_accuracy: 0.9386
50560/60000 [========================>.....] - ETA: 17s - loss: 0.2020 - categorical_accuracy: 0.9386
50592/60000 [========================>.....] - ETA: 17s - loss: 0.2019 - categorical_accuracy: 0.9386
50624/60000 [========================>.....] - ETA: 17s - loss: 0.2018 - categorical_accuracy: 0.9386
50656/60000 [========================>.....] - ETA: 17s - loss: 0.2017 - categorical_accuracy: 0.9387
50688/60000 [========================>.....] - ETA: 17s - loss: 0.2017 - categorical_accuracy: 0.9387
50720/60000 [========================>.....] - ETA: 17s - loss: 0.2017 - categorical_accuracy: 0.9387
50752/60000 [========================>.....] - ETA: 17s - loss: 0.2016 - categorical_accuracy: 0.9387
50784/60000 [========================>.....] - ETA: 16s - loss: 0.2015 - categorical_accuracy: 0.9387
50816/60000 [========================>.....] - ETA: 16s - loss: 0.2014 - categorical_accuracy: 0.9388
50848/60000 [========================>.....] - ETA: 16s - loss: 0.2013 - categorical_accuracy: 0.9388
50880/60000 [========================>.....] - ETA: 16s - loss: 0.2011 - categorical_accuracy: 0.9388
50912/60000 [========================>.....] - ETA: 16s - loss: 0.2011 - categorical_accuracy: 0.9389
50944/60000 [========================>.....] - ETA: 16s - loss: 0.2010 - categorical_accuracy: 0.9389
50976/60000 [========================>.....] - ETA: 16s - loss: 0.2008 - categorical_accuracy: 0.9389
51008/60000 [========================>.....] - ETA: 16s - loss: 0.2008 - categorical_accuracy: 0.9390
51040/60000 [========================>.....] - ETA: 16s - loss: 0.2007 - categorical_accuracy: 0.9390
51072/60000 [========================>.....] - ETA: 16s - loss: 0.2007 - categorical_accuracy: 0.9390
51104/60000 [========================>.....] - ETA: 16s - loss: 0.2006 - categorical_accuracy: 0.9390
51136/60000 [========================>.....] - ETA: 16s - loss: 0.2005 - categorical_accuracy: 0.9391
51168/60000 [========================>.....] - ETA: 16s - loss: 0.2003 - categorical_accuracy: 0.9391
51200/60000 [========================>.....] - ETA: 16s - loss: 0.2003 - categorical_accuracy: 0.9391
51232/60000 [========================>.....] - ETA: 16s - loss: 0.2003 - categorical_accuracy: 0.9391
51264/60000 [========================>.....] - ETA: 16s - loss: 0.2001 - categorical_accuracy: 0.9391
51296/60000 [========================>.....] - ETA: 16s - loss: 0.2000 - categorical_accuracy: 0.9392
51328/60000 [========================>.....] - ETA: 15s - loss: 0.1999 - categorical_accuracy: 0.9392
51360/60000 [========================>.....] - ETA: 15s - loss: 0.1999 - categorical_accuracy: 0.9392
51392/60000 [========================>.....] - ETA: 15s - loss: 0.1998 - categorical_accuracy: 0.9392
51424/60000 [========================>.....] - ETA: 15s - loss: 0.1997 - categorical_accuracy: 0.9392
51456/60000 [========================>.....] - ETA: 15s - loss: 0.1996 - categorical_accuracy: 0.9393
51488/60000 [========================>.....] - ETA: 15s - loss: 0.1996 - categorical_accuracy: 0.9393
51520/60000 [========================>.....] - ETA: 15s - loss: 0.1996 - categorical_accuracy: 0.9393
51552/60000 [========================>.....] - ETA: 15s - loss: 0.1995 - categorical_accuracy: 0.9393
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1995 - categorical_accuracy: 0.9393
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1996 - categorical_accuracy: 0.9393
51648/60000 [========================>.....] - ETA: 15s - loss: 0.1996 - categorical_accuracy: 0.9393
51680/60000 [========================>.....] - ETA: 15s - loss: 0.1995 - categorical_accuracy: 0.9393
51712/60000 [========================>.....] - ETA: 15s - loss: 0.1994 - categorical_accuracy: 0.9394
51744/60000 [========================>.....] - ETA: 15s - loss: 0.1993 - categorical_accuracy: 0.9394
51776/60000 [========================>.....] - ETA: 15s - loss: 0.1992 - categorical_accuracy: 0.9394
51808/60000 [========================>.....] - ETA: 15s - loss: 0.1992 - categorical_accuracy: 0.9394
51840/60000 [========================>.....] - ETA: 15s - loss: 0.1993 - categorical_accuracy: 0.9394
51872/60000 [========================>.....] - ETA: 14s - loss: 0.1993 - categorical_accuracy: 0.9394
51904/60000 [========================>.....] - ETA: 14s - loss: 0.1992 - categorical_accuracy: 0.9394
51936/60000 [========================>.....] - ETA: 14s - loss: 0.1991 - categorical_accuracy: 0.9395
51968/60000 [========================>.....] - ETA: 14s - loss: 0.1991 - categorical_accuracy: 0.9395
52000/60000 [=========================>....] - ETA: 14s - loss: 0.1991 - categorical_accuracy: 0.9395
52032/60000 [=========================>....] - ETA: 14s - loss: 0.1990 - categorical_accuracy: 0.9395
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1990 - categorical_accuracy: 0.9395
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1989 - categorical_accuracy: 0.9396
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1988 - categorical_accuracy: 0.9396
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1988 - categorical_accuracy: 0.9396
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1987 - categorical_accuracy: 0.9396
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1987 - categorical_accuracy: 0.9396
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1986 - categorical_accuracy: 0.9396
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1986 - categorical_accuracy: 0.9396
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1985 - categorical_accuracy: 0.9397
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1984 - categorical_accuracy: 0.9397
52384/60000 [=========================>....] - ETA: 14s - loss: 0.1983 - categorical_accuracy: 0.9397
52416/60000 [=========================>....] - ETA: 13s - loss: 0.1983 - categorical_accuracy: 0.9397
52448/60000 [=========================>....] - ETA: 13s - loss: 0.1982 - categorical_accuracy: 0.9397
52480/60000 [=========================>....] - ETA: 13s - loss: 0.1982 - categorical_accuracy: 0.9397
52512/60000 [=========================>....] - ETA: 13s - loss: 0.1980 - categorical_accuracy: 0.9398
52544/60000 [=========================>....] - ETA: 13s - loss: 0.1980 - categorical_accuracy: 0.9398
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1979 - categorical_accuracy: 0.9398
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1980 - categorical_accuracy: 0.9398
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1979 - categorical_accuracy: 0.9399
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1978 - categorical_accuracy: 0.9399
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1977 - categorical_accuracy: 0.9399
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1976 - categorical_accuracy: 0.9400
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1975 - categorical_accuracy: 0.9400
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1974 - categorical_accuracy: 0.9400
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1973 - categorical_accuracy: 0.9401
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1973 - categorical_accuracy: 0.9401
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1972 - categorical_accuracy: 0.9401
52928/60000 [=========================>....] - ETA: 13s - loss: 0.1971 - categorical_accuracy: 0.9401
52960/60000 [=========================>....] - ETA: 12s - loss: 0.1970 - categorical_accuracy: 0.9402
52992/60000 [=========================>....] - ETA: 12s - loss: 0.1970 - categorical_accuracy: 0.9402
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1969 - categorical_accuracy: 0.9402
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1969 - categorical_accuracy: 0.9402
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1969 - categorical_accuracy: 0.9402
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1969 - categorical_accuracy: 0.9402
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1968 - categorical_accuracy: 0.9402
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1967 - categorical_accuracy: 0.9403
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1966 - categorical_accuracy: 0.9403
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1965 - categorical_accuracy: 0.9404
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1964 - categorical_accuracy: 0.9404
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1963 - categorical_accuracy: 0.9404
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1963 - categorical_accuracy: 0.9404
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1962 - categorical_accuracy: 0.9405
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1961 - categorical_accuracy: 0.9405
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1960 - categorical_accuracy: 0.9405
53472/60000 [=========================>....] - ETA: 12s - loss: 0.1959 - categorical_accuracy: 0.9405
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1958 - categorical_accuracy: 0.9406
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1958 - categorical_accuracy: 0.9406
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1958 - categorical_accuracy: 0.9406
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1959 - categorical_accuracy: 0.9406
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1960 - categorical_accuracy: 0.9406
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1959 - categorical_accuracy: 0.9406
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1958 - categorical_accuracy: 0.9406
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1958 - categorical_accuracy: 0.9406
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1957 - categorical_accuracy: 0.9407
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1956 - categorical_accuracy: 0.9407
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1956 - categorical_accuracy: 0.9407
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1956 - categorical_accuracy: 0.9407
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1955 - categorical_accuracy: 0.9407
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1954 - categorical_accuracy: 0.9407
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1954 - categorical_accuracy: 0.9407
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1953 - categorical_accuracy: 0.9408
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1952 - categorical_accuracy: 0.9408
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1951 - categorical_accuracy: 0.9408
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1950 - categorical_accuracy: 0.9408
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1949 - categorical_accuracy: 0.9409
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1948 - categorical_accuracy: 0.9409
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1947 - categorical_accuracy: 0.9410
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1946 - categorical_accuracy: 0.9410
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1944 - categorical_accuracy: 0.9410
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1944 - categorical_accuracy: 0.9411
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1944 - categorical_accuracy: 0.9411
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1943 - categorical_accuracy: 0.9411
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1942 - categorical_accuracy: 0.9411
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1942 - categorical_accuracy: 0.9411
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1941 - categorical_accuracy: 0.9411
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1940 - categorical_accuracy: 0.9412
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1939 - categorical_accuracy: 0.9412
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1938 - categorical_accuracy: 0.9412
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1937 - categorical_accuracy: 0.9413
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1937 - categorical_accuracy: 0.9412 
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1936 - categorical_accuracy: 0.9413
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1935 - categorical_accuracy: 0.9413
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1934 - categorical_accuracy: 0.9413
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1933 - categorical_accuracy: 0.9414
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1933 - categorical_accuracy: 0.9414
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1932 - categorical_accuracy: 0.9414
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1931 - categorical_accuracy: 0.9414
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1930 - categorical_accuracy: 0.9414
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1929 - categorical_accuracy: 0.9415
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1929 - categorical_accuracy: 0.9415
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1929 - categorical_accuracy: 0.9414
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1930 - categorical_accuracy: 0.9414
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1929 - categorical_accuracy: 0.9415
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1928 - categorical_accuracy: 0.9415
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1928 - categorical_accuracy: 0.9415
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1926 - categorical_accuracy: 0.9416
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1927 - categorical_accuracy: 0.9416
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1926 - categorical_accuracy: 0.9416
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1925 - categorical_accuracy: 0.9416
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1924 - categorical_accuracy: 0.9417
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1925 - categorical_accuracy: 0.9417
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1924 - categorical_accuracy: 0.9417
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1924 - categorical_accuracy: 0.9417
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1923 - categorical_accuracy: 0.9418
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1922 - categorical_accuracy: 0.9418
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1921 - categorical_accuracy: 0.9418
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1922 - categorical_accuracy: 0.9418
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1921 - categorical_accuracy: 0.9418
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1920 - categorical_accuracy: 0.9419
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1919 - categorical_accuracy: 0.9419
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1918 - categorical_accuracy: 0.9419
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1917 - categorical_accuracy: 0.9419
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1917 - categorical_accuracy: 0.9420
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1916 - categorical_accuracy: 0.9420
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1915 - categorical_accuracy: 0.9420
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1915 - categorical_accuracy: 0.9420
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1915 - categorical_accuracy: 0.9420
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1914 - categorical_accuracy: 0.9421
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1913 - categorical_accuracy: 0.9421
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1912 - categorical_accuracy: 0.9421
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1911 - categorical_accuracy: 0.9422
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1910 - categorical_accuracy: 0.9422
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1909 - categorical_accuracy: 0.9422
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1910 - categorical_accuracy: 0.9422
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1909 - categorical_accuracy: 0.9422
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1909 - categorical_accuracy: 0.9423
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1908 - categorical_accuracy: 0.9423
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1909 - categorical_accuracy: 0.9423
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1908 - categorical_accuracy: 0.9423
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1907 - categorical_accuracy: 0.9423
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1907 - categorical_accuracy: 0.9423
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1906 - categorical_accuracy: 0.9423
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1905 - categorical_accuracy: 0.9424
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1906 - categorical_accuracy: 0.9424
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1906 - categorical_accuracy: 0.9424
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1906 - categorical_accuracy: 0.9424
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1905 - categorical_accuracy: 0.9424
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1904 - categorical_accuracy: 0.9424
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1903 - categorical_accuracy: 0.9425
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1903 - categorical_accuracy: 0.9425
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1904 - categorical_accuracy: 0.9424
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1904 - categorical_accuracy: 0.9424
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1903 - categorical_accuracy: 0.9425
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1902 - categorical_accuracy: 0.9425
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1901 - categorical_accuracy: 0.9425
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1900 - categorical_accuracy: 0.9426
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1900 - categorical_accuracy: 0.9426
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1899 - categorical_accuracy: 0.9426
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1899 - categorical_accuracy: 0.9426
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1898 - categorical_accuracy: 0.9426
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1897 - categorical_accuracy: 0.9427
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1896 - categorical_accuracy: 0.9427
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1896 - categorical_accuracy: 0.9427
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1895 - categorical_accuracy: 0.9427
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1895 - categorical_accuracy: 0.9427
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1894 - categorical_accuracy: 0.9427
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1894 - categorical_accuracy: 0.9427
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1893 - categorical_accuracy: 0.9428
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1893 - categorical_accuracy: 0.9428
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1892 - categorical_accuracy: 0.9428
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1892 - categorical_accuracy: 0.9428
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1891 - categorical_accuracy: 0.9428
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1891 - categorical_accuracy: 0.9428
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1890 - categorical_accuracy: 0.9429
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1890 - categorical_accuracy: 0.9428
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1889 - categorical_accuracy: 0.9429
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1889 - categorical_accuracy: 0.9429
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1888 - categorical_accuracy: 0.9429
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1888 - categorical_accuracy: 0.9429
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1887 - categorical_accuracy: 0.9429
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1887 - categorical_accuracy: 0.9429
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1886 - categorical_accuracy: 0.9430
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1886 - categorical_accuracy: 0.9430
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1885 - categorical_accuracy: 0.9430
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1884 - categorical_accuracy: 0.9430
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1883 - categorical_accuracy: 0.9431
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1882 - categorical_accuracy: 0.9431
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1883 - categorical_accuracy: 0.9431
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1883 - categorical_accuracy: 0.9431
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1883 - categorical_accuracy: 0.9431
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1882 - categorical_accuracy: 0.9431
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1882 - categorical_accuracy: 0.9431
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1881 - categorical_accuracy: 0.9431
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1880 - categorical_accuracy: 0.9431
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1879 - categorical_accuracy: 0.9431
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1880 - categorical_accuracy: 0.9432
58016/60000 [============================>.] - ETA: 3s - loss: 0.1880 - categorical_accuracy: 0.9432
58048/60000 [============================>.] - ETA: 3s - loss: 0.1879 - categorical_accuracy: 0.9432
58080/60000 [============================>.] - ETA: 3s - loss: 0.1878 - categorical_accuracy: 0.9432
58112/60000 [============================>.] - ETA: 3s - loss: 0.1877 - categorical_accuracy: 0.9433
58144/60000 [============================>.] - ETA: 3s - loss: 0.1877 - categorical_accuracy: 0.9433
58176/60000 [============================>.] - ETA: 3s - loss: 0.1876 - categorical_accuracy: 0.9433
58208/60000 [============================>.] - ETA: 3s - loss: 0.1875 - categorical_accuracy: 0.9433
58240/60000 [============================>.] - ETA: 3s - loss: 0.1874 - categorical_accuracy: 0.9434
58272/60000 [============================>.] - ETA: 3s - loss: 0.1873 - categorical_accuracy: 0.9434
58304/60000 [============================>.] - ETA: 3s - loss: 0.1873 - categorical_accuracy: 0.9434
58336/60000 [============================>.] - ETA: 3s - loss: 0.1872 - categorical_accuracy: 0.9434
58368/60000 [============================>.] - ETA: 3s - loss: 0.1871 - categorical_accuracy: 0.9435
58400/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9435
58432/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9435
58464/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9435
58496/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9436
58528/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9436
58560/60000 [============================>.] - ETA: 2s - loss: 0.1867 - categorical_accuracy: 0.9436
58592/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9436
58624/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9437
58656/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9437
58688/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9437
58720/60000 [============================>.] - ETA: 2s - loss: 0.1865 - categorical_accuracy: 0.9437
58752/60000 [============================>.] - ETA: 2s - loss: 0.1865 - categorical_accuracy: 0.9437
58784/60000 [============================>.] - ETA: 2s - loss: 0.1864 - categorical_accuracy: 0.9437
58816/60000 [============================>.] - ETA: 2s - loss: 0.1865 - categorical_accuracy: 0.9437
58848/60000 [============================>.] - ETA: 2s - loss: 0.1864 - categorical_accuracy: 0.9437
58880/60000 [============================>.] - ETA: 2s - loss: 0.1864 - categorical_accuracy: 0.9437
58912/60000 [============================>.] - ETA: 2s - loss: 0.1863 - categorical_accuracy: 0.9437
58944/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9438
58976/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9438
59008/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9438
59040/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9438
59072/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9438
59104/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9438
59136/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9438
59168/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9439
59200/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9439
59232/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9438
59264/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9439
59296/60000 [============================>.] - ETA: 1s - loss: 0.1857 - categorical_accuracy: 0.9439
59328/60000 [============================>.] - ETA: 1s - loss: 0.1856 - categorical_accuracy: 0.9439
59360/60000 [============================>.] - ETA: 1s - loss: 0.1855 - categorical_accuracy: 0.9440
59392/60000 [============================>.] - ETA: 1s - loss: 0.1855 - categorical_accuracy: 0.9440
59424/60000 [============================>.] - ETA: 1s - loss: 0.1854 - categorical_accuracy: 0.9440
59456/60000 [============================>.] - ETA: 1s - loss: 0.1853 - categorical_accuracy: 0.9441
59488/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9441
59520/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9441
59552/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9441
59584/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9441
59616/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9441
59648/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9441
59680/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9441
59712/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9441
59744/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9441
59776/60000 [============================>.] - ETA: 0s - loss: 0.1851 - categorical_accuracy: 0.9441
59808/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9441
59840/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9442
59872/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9442
59904/60000 [============================>.] - ETA: 0s - loss: 0.1848 - categorical_accuracy: 0.9442
59936/60000 [============================>.] - ETA: 0s - loss: 0.1847 - categorical_accuracy: 0.9442
59968/60000 [============================>.] - ETA: 0s - loss: 0.1847 - categorical_accuracy: 0.9442
60000/60000 [==============================] - 114s 2ms/step - loss: 0.1846 - categorical_accuracy: 0.9442 - val_loss: 0.0455 - val_categorical_accuracy: 0.9856

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 16s
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
 1952/10000 [====>.........................] - ETA: 3s
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
 4352/10000 [============>.................] - ETA: 2s
 4512/10000 [============>.................] - ETA: 1s
 4672/10000 [=============>................] - ETA: 1s
 4832/10000 [=============>................] - ETA: 1s
 4960/10000 [=============>................] - ETA: 1s
 5120/10000 [==============>...............] - ETA: 1s
 5280/10000 [==============>...............] - ETA: 1s
 5440/10000 [===============>..............] - ETA: 1s
 5600/10000 [===============>..............] - ETA: 1s
 5760/10000 [================>.............] - ETA: 1s
 5920/10000 [================>.............] - ETA: 1s
 6080/10000 [=================>............] - ETA: 1s
 6240/10000 [=================>............] - ETA: 1s
 6400/10000 [==================>...........] - ETA: 1s
 6560/10000 [==================>...........] - ETA: 1s
 6720/10000 [===================>..........] - ETA: 1s
 6880/10000 [===================>..........] - ETA: 1s
 7040/10000 [====================>.........] - ETA: 1s
 7200/10000 [====================>.........] - ETA: 1s
 7360/10000 [=====================>........] - ETA: 0s
 7520/10000 [=====================>........] - ETA: 0s
 7680/10000 [======================>.......] - ETA: 0s
 7840/10000 [======================>.......] - ETA: 0s
 8000/10000 [=======================>......] - ETA: 0s
 8160/10000 [=======================>......] - ETA: 0s
 8320/10000 [=======================>......] - ETA: 0s
 8480/10000 [========================>.....] - ETA: 0s
 8640/10000 [========================>.....] - ETA: 0s
 8800/10000 [=========================>....] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9120/10000 [==========================>...] - ETA: 0s
 9280/10000 [==========================>...] - ETA: 0s
 9440/10000 [===========================>..] - ETA: 0s
 9600/10000 [===========================>..] - ETA: 0s
 9760/10000 [============================>.] - ETA: 0s
 9920/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 360us/step
[[3.5482049e-08 3.4834060e-09 3.3666568e-06 ... 9.9999499e-01
  4.2167734e-08 7.6333004e-07]
 [6.4091269e-06 8.0625932e-06 9.9997520e-01 ... 6.9778380e-08
  2.6471935e-06 4.2229356e-08]
 [3.1199145e-06 9.9978906e-01 2.2757282e-05 ... 7.8692115e-05
  2.7958771e-05 3.3382503e-06]
 ...
 [7.0212209e-09 7.5472173e-07 5.9461652e-08 ... 8.4231560e-06
  8.9666173e-06 3.4236818e-05]
 [3.0343191e-08 5.0236248e-10 1.3388331e-10 ... 9.3514387e-09
  1.9868823e-05 1.6093210e-07]
 [2.2148949e-06 7.8312439e-08 2.2633860e-06 ... 2.2298823e-09
  4.6809191e-07 2.4052316e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.045507689163682516, 'accuracy_test:': 0.9855999946594238}

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
[master 1ec7ef1] ml_store  && git pull --all
 1 file changed, 2036 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 51f0f11...1ec7ef1 master -> master (forced update)





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
{'loss': 0.4683315008878708, 'loss_history': []}

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
[master af31b45] ml_store  && git pull --all
 1 file changed, 111 insertions(+)
To github.com:arita37/mlmodels_store.git
   1ec7ef1..af31b45  master -> master





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
[master d5f3711] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   af31b45..d5f3711  master -> master





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
	Data preprocessing and feature engineering runtime = 0.25s ...
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
 40%|████      | 2/5 [00:21<00:31, 10.60s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9843372045753411, 'learning_rate': 0.08472541124697783, 'min_data_in_leaf': 23, 'num_leaves': 51} and reward: 0.3912
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x7f\xb0\xbc\xbcc\x8cX\r\x00\x00\x00learning_rateq\x02G?\xb5\xb0\x90\x86r'\xd6X\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K3u." and reward: 0.3912
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\x7f\xb0\xbc\xbcc\x8cX\r\x00\x00\x00learning_rateq\x02G?\xb5\xb0\x90\x86r'\xd6X\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K3u." and reward: 0.3912
 60%|██████    | 3/5 [00:48<00:30, 15.50s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8080556872864336, 'learning_rate': 0.13125730887493334, 'min_data_in_leaf': 23, 'num_leaves': 57} and reward: 0.393
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\xdb\x97\x99\xc7\xbe\xc1X\r\x00\x00\x00learning_rateq\x02G?\xc0\xcd\n\x1c}I\x9fX\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.393
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\xdb\x97\x99\xc7\xbe\xc1X\r\x00\x00\x00learning_rateq\x02G?\xc0\xcd\n\x1c}I\x9fX\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.393
 80%|████████  | 4/5 [01:16<00:19, 19.21s/it] 80%|████████  | 4/5 [01:16<00:19, 19.00s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8471704171312153, 'learning_rate': 0.006535944131512876, 'min_data_in_leaf': 6, 'num_leaves': 28} and reward: 0.3852
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\x1c\x05"v\xf3\xb2X\r\x00\x00\x00learning_rateq\x02G?z\xc5o$\xb1{\xc5X\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K\x1cu.' and reward: 0.3852
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\x1c\x05"v\xf3\xb2X\r\x00\x00\x00learning_rateq\x02G?z\xc5o$\xb1{\xc5X\x10\x00\x00\x00min_data_in_leafq\x03K\x06X\n\x00\x00\x00num_leavesq\x04K\x1cu.' and reward: 0.3852
Time for Gradient Boosting hyperparameter optimization: 94.59060144424438
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8080556872864336, 'learning_rate': 0.13125730887493334, 'min_data_in_leaf': 23, 'num_leaves': 57}
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
 40%|████      | 2/5 [00:51<01:17, 25.81s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.35507218126273304, 'embedding_size_factor': 1.29960201532915, 'layers.choice': 1, 'learning_rate': 0.009734842299546645, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 8.115987838498307e-09} and reward: 0.3764
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd6\xb9\x80\xab\x8f\x8agX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xcb+{\x9axbX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x83\xef\xdcjwVhX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>Am\xcf\xbeq\x0f\x08u.' and reward: 0.3764
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd6\xb9\x80\xab\x8f\x8agX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xcb+{\x9axbX\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?\x83\xef\xdcjwVhX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>Am\xcf\xbeq\x0f\x08u.' and reward: 0.3764
 60%|██████    | 3/5 [02:24<01:32, 46.00s/it] 60%|██████    | 3/5 [02:24<01:36, 48.25s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.36015797574734487, 'embedding_size_factor': 0.8938571574040378, 'layers.choice': 3, 'learning_rate': 0.00040396236393605664, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 2.6508898369434073e-10} and reward: 0.3684
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd7\x0c\xd4\t\xce\x9e:X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\x9azSK\x11WX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?:y]$T\xe2\x12X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xf27~\xa5\xef\x82\x1fu.' and reward: 0.3684
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd7\x0c\xd4\t\xce\x9e:X\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\x9azSK\x11WX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?:y]$T\xe2\x12X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xf27~\xa5\xef\x82\x1fu.' and reward: 0.3684
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 199.8617479801178
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -178.45s of remaining time.
Ensemble size: 77
Ensemble weights: 
[0.20779221 0.06493506 0.14285714 0.09090909 0.18181818 0.01298701
 0.2987013 ]
	0.4042	 = Validation accuracy score
	1.66s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 300.17s ...
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
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f3b704e7748>

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
[master f4c0688] ml_store  && git pull --all
 1 file changed, 207 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 0bcd4e8...f4c0688 master -> master (forced update)





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
[master 5a42649] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   f4c0688..5a42649  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.81it/s, avg_epoch_loss=5.24]
INFO:root:Epoch[0] Elapsed time 2.625 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.242700
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.242699766159058 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f73d114a3c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f73d114a3c8>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 99.58it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1111.126953125,
    "abs_error": 379.58270263671875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.515095940915404,
    "sMAPE": 0.5218481693535606,
    "MSIS": 100.60382793063498,
    "QuantileLoss[0.5]": 379.58267974853516,
    "Coverage[0.5]": 1.0,
    "RMSE": 33.33357096269465,
    "NRMSE": 0.7017593886883084,
    "ND": 0.6659345660293311,
    "wQuantileLoss[0.5]": 0.6659345258746231,
    "mean_wQuantileLoss": 0.6659345258746231,
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
100%|██████████| 10/10 [00:01<00:00,  7.60it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.316 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f73d114a3c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f73d114a3c8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 164.10it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.49it/s, avg_epoch_loss=5.28]
INFO:root:Epoch[0] Elapsed time 1.822 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.281720
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.281720399856567 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f73a4b1ae48>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f73a4b1ae48>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 135.43it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 229.8017781575521,
    "abs_error": 157.14442443847656,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.041231071113795,
    "sMAPE": 0.2669867239137485,
    "MSIS": 41.649235565065915,
    "QuantileLoss[0.5]": 157.14440536499023,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 15.15921429882011,
    "NRMSE": 0.31914135365937074,
    "ND": 0.2756919726990817,
    "wQuantileLoss[0.5]": 0.27569193923682495,
    "mean_wQuantileLoss": 0.27569193923682495,
    "MAE_Coverage": 0.16666666666666663
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
 30%|███       | 3/10 [00:12<00:29,  4.27s/it, avg_epoch_loss=6.94] 60%|██████    | 6/10 [00:24<00:16,  4.12s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:35<00:04,  4.03s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:39<00:00,  3.93s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 39.300 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.872793
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.872792673110962 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f73a40f4400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f73a40f4400>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 164.75it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 53108.411458333336,
    "abs_error": 2709.380859375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.952221622515967,
    "sMAPE": 1.4107672441467576,
    "MSIS": 718.0887613701727,
    "QuantileLoss[0.5]": 2709.3806915283203,
    "Coverage[0.5]": 1.0,
    "RMSE": 230.45262302333063,
    "NRMSE": 4.8516341689122235,
    "ND": 4.753299753289474,
    "wQuantileLoss[0.5]": 4.753299458821615,
    "mean_wQuantileLoss": 4.753299458821615,
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
100%|██████████| 10/10 [00:00<00:00, 53.51it/s, avg_epoch_loss=5.2]
INFO:root:Epoch[0] Elapsed time 0.187 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.196630
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.196629858016967 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f738daa8fd0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f738daa8fd0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 156.92it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 523.8976236979166,
    "abs_error": 191.10845947265625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2662750629980304,
    "sMAPE": 0.31974027452426035,
    "MSIS": 50.65099604926709,
    "QuantileLoss[0.5]": 191.10844802856445,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 22.888810010525155,
    "NRMSE": 0.48186968443210854,
    "ND": 0.3352779990748355,
    "wQuantileLoss[0.5]": 0.3352779789974815,
    "mean_wQuantileLoss": 0.3352779789974815,
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
100%|██████████| 10/10 [00:01<00:00,  8.44it/s, avg_epoch_loss=161]
INFO:root:Epoch[0] Elapsed time 1.186 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=161.108291
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 161.1082910709855 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f738da73160>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f738da73160>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 130.62it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [01:56<17:25, 116.11s/it, avg_epoch_loss=0.582] 20%|██        | 2/10 [04:50<17:48, 133.62s/it, avg_epoch_loss=0.565] 30%|███       | 3/10 [07:55<17:22, 148.98s/it, avg_epoch_loss=0.548] 40%|████      | 4/10 [10:57<15:53, 158.91s/it, avg_epoch_loss=0.531] 50%|█████     | 5/10 [14:15<14:13, 170.67s/it, avg_epoch_loss=0.515] 60%|██████    | 6/10 [17:33<11:55, 178.82s/it, avg_epoch_loss=0.499] 70%|███████   | 7/10 [20:56<09:18, 186.12s/it, avg_epoch_loss=0.484] 80%|████████  | 8/10 [24:23<06:24, 192.28s/it, avg_epoch_loss=0.47]  90%|█████████ | 9/10 [27:27<03:09, 189.77s/it, avg_epoch_loss=0.457]100%|██████████| 10/10 [30:57<00:00, 195.94s/it, avg_epoch_loss=0.447]100%|██████████| 10/10 [30:57<00:00, 185.77s/it, avg_epoch_loss=0.447]
INFO:root:Epoch[0] Elapsed time 1857.685 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.446552
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.4465524971485138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f738da73f98>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f738da73f98>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 17.56it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
