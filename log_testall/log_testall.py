
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
[master 8454998] ml_store  && git pull --all
 2 files changed, 13 insertions(+), 90 deletions(-)
To github.com:arita37/mlmodels_store.git
 + 95241b7...8454998 master -> master (forced update)





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
[master ca57ce2] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   8454998..ca57ce2  master -> master





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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-24 04:15:07.793949: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-24 04:15:07.799720: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-24 04:15:07.800501: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ba6533f1e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-24 04:15:07.800526: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 243
Trainable params: 243
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2499 - binary_crossentropy: 0.6929 - val_loss: 0.2500 - val_binary_crossentropy: 0.6932

  #### metrics   #################################################### 
{'MSE': 0.24975834292867388}

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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 2, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         28          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         16          sparse_feature_2[0][0]           
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
Total params: 243
Trainable params: 243
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
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
Total params: 458
Trainable params: 458
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2921 - binary_crossentropy: 0.7864500/500 [==============================] - 1s 2ms/sample - loss: 0.2790 - binary_crossentropy: 0.7574 - val_loss: 0.2876 - val_binary_crossentropy: 0.8023

  #### metrics   #################################################### 
{'MSE': 0.28236973155867234}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
Total params: 637
Trainable params: 637
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2498 - binary_crossentropy: 0.6927 - val_loss: 0.2503 - val_binary_crossentropy: 0.6937

  #### metrics   #################################################### 
{'MSE': 0.24974877031266043}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
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
Total params: 637
Trainable params: 637
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 408
Trainable params: 408
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.5400 - binary_crossentropy: 8.3295500/500 [==============================] - 2s 3ms/sample - loss: 0.5140 - binary_crossentropy: 7.9284 - val_loss: 0.5160 - val_binary_crossentropy: 7.9593

  #### metrics   #################################################### 
{'MSE': 0.515}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 408
Trainable params: 408
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
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
Total params: 158
Trainable params: 158
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 3s - loss: 0.5400 - binary_crossentropy: 8.3295500/500 [==============================] - 2s 4ms/sample - loss: 0.5340 - binary_crossentropy: 8.2369 - val_loss: 0.4640 - val_binary_crossentropy: 7.1572

  #### metrics   #################################################### 
{'MSE': 0.499}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-24 04:16:37.195334: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:16:37.197646: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:16:37.205609: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-24 04:16:37.218198: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-24 04:16:37.220154: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 04:16:37.222381: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:16:37.224174: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2495 - val_binary_crossentropy: 0.6921
2020-05-24 04:16:38.742695: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:16:38.744904: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:16:38.749868: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-24 04:16:38.760001: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-24 04:16:38.762499: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 04:16:38.764215: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:16:38.765685: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24919553375677594}

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
2020-05-24 04:17:05.831190: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:05.832900: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:05.837147: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-24 04:17:05.844382: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-24 04:17:05.845541: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 04:17:05.846733: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:05.847878: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2506 - val_binary_crossentropy: 0.6943
2020-05-24 04:17:07.601601: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:07.602712: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:07.606424: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-24 04:17:07.612760: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-24 04:17:07.614249: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 04:17:07.615335: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:07.616357: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2506463795893958}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-24 04:17:46.390055: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:46.395263: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:46.411805: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-24 04:17:46.439158: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-24 04:17:46.444853: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 04:17:46.449637: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:46.454328: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.7010 - binary_crossentropy: 1.8157 - val_loss: 0.2504 - val_binary_crossentropy: 0.6939
2020-05-24 04:17:48.990283: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:48.995181: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:49.008896: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-24 04:17:49.036677: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-24 04:17:49.040929: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-24 04:17:49.045659: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-24 04:17:49.049784: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25072634042965863}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
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
Total params: 690
Trainable params: 690
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3056 - binary_crossentropy: 0.8205500/500 [==============================] - 5s 10ms/sample - loss: 0.2796 - binary_crossentropy: 0.7625 - val_loss: 0.2981 - val_binary_crossentropy: 0.8054

  #### metrics   #################################################### 
{'MSE': 0.2876758009737044}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
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
Total params: 690
Trainable params: 690
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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         12          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_5[0][0]           
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
100/500 [=====>........................] - ETA: 7s - loss: 0.2635 - binary_crossentropy: 0.7248500/500 [==============================] - 5s 10ms/sample - loss: 0.2709 - binary_crossentropy: 0.7934 - val_loss: 0.2647 - val_binary_crossentropy: 0.8555

  #### metrics   #################################################### 
{'MSE': 0.26318165606787847}

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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         6           sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         10          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         12          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_5[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
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
Total params: 1,869
Trainable params: 1,869
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.4700 - binary_crossentropy: 7.2497500/500 [==============================] - 5s 10ms/sample - loss: 0.4900 - binary_crossentropy: 7.5582 - val_loss: 0.4440 - val_binary_crossentropy: 6.8487

  #### metrics   #################################################### 
{'MSE': 0.467}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_max[0][0]               
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
Total params: 1,869
Trainable params: 1,869
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
regionsequence_sum (InputLayer) [(None, 7)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         3           regionsequence_mean[0][0]        
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
Total params: 184
Trainable params: 184
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2518 - binary_crossentropy: 0.6965500/500 [==============================] - 6s 13ms/sample - loss: 0.2490 - binary_crossentropy: 0.6910 - val_loss: 0.2545 - val_binary_crossentropy: 0.7020

  #### metrics   #################################################### 
{'MSE': 0.25146868370117326}

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
regionsequence_sum (InputLayer) [(None, 7)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 2)]          0                                            
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
region_10sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         3           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         3           regionsequence_mean[0][0]        
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
Total params: 184
Trainable params: 184
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
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
Total params: 1,367
Trainable params: 1,367
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3588 - binary_crossentropy: 3.2724500/500 [==============================] - 6s 13ms/sample - loss: 0.3459 - binary_crossentropy: 2.7736 - val_loss: 0.3168 - val_binary_crossentropy: 2.4668

  #### metrics   #################################################### 
{'MSE': 0.3296591857291}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
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
Total params: 1,367
Trainable params: 1,367
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         20          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         20          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           hash_11[0][0]                    
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
Total params: 2,984
Trainable params: 2,904
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 10s - loss: 0.3102 - binary_crossentropy: 1.0891500/500 [==============================] - 7s 14ms/sample - loss: 0.2869 - binary_crossentropy: 0.8808 - val_loss: 0.2857 - val_binary_crossentropy: 0.9006

  #### metrics   #################################################### 
{'MSE': 0.28477750435501653}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         20          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         20          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         20          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           hash_11[0][0]                    
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
Total params: 2,984
Trainable params: 2,904
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
[master c27bff2] ml_store  && git pull --all
 1 file changed, 4946 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + 8fd3ef2...c27bff2 master -> master (forced update)





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
[master 34d2b15] ml_store  && git pull --all
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   c27bff2..34d2b15  master -> master





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
[master bb4ce22] ml_store  && git pull --all
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   34d2b15..bb4ce22  master -> master





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
[master ac6e2b0] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   bb4ce22..ac6e2b0  master -> master





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
[master 82bf52d] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   ac6e2b0..82bf52d  master -> master





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
[master 2b0bd1d] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   82bf52d..2b0bd1d  master -> master





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
[master f529004] ml_store  && git pull --all
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   2b0bd1d..f529004  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3620864/17464789 [=====>........................] - ETA: 0s
 8503296/17464789 [=============>................] - ETA: 0s
12607488/17464789 [====================>.........] - ETA: 0s
16818176/17464789 [===========================>..] - ETA: 0s
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
2020-05-24 04:28:39.828464: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-24 04:28:39.832888: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-24 04:28:39.833084: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555ef731b960 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-24 04:28:39.833100: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8660 - accuracy: 0.4870
 2000/25000 [=>............................] - ETA: 10s - loss: 7.5823 - accuracy: 0.5055
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7075 - accuracy: 0.4973 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8621 - accuracy: 0.4873
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8506 - accuracy: 0.4880
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8353 - accuracy: 0.4890
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.8725 - accuracy: 0.4866
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7797 - accuracy: 0.4926
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7944 - accuracy: 0.4917
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7694 - accuracy: 0.4933
11000/25000 [============>.................] - ETA: 4s - loss: 7.7405 - accuracy: 0.4952
12000/25000 [=============>................] - ETA: 4s - loss: 7.7382 - accuracy: 0.4953
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7539 - accuracy: 0.4943
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7203 - accuracy: 0.4965
15000/25000 [=================>............] - ETA: 3s - loss: 7.7259 - accuracy: 0.4961
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7050 - accuracy: 0.4975
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6955 - accuracy: 0.4981
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7007 - accuracy: 0.4978
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6949 - accuracy: 0.4982
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6904 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6722 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6766 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
25000/25000 [==============================] - 10s 395us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f3a49a0c9e8>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f3a42e90cc0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6283 - accuracy: 0.5025
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7024 - accuracy: 0.4977 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.5670 - accuracy: 0.5065
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6758 - accuracy: 0.4994
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6922 - accuracy: 0.4983
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7017 - accuracy: 0.4977
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6724 - accuracy: 0.4996
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
11000/25000 [============>.................] - ETA: 4s - loss: 7.6429 - accuracy: 0.5015
12000/25000 [=============>................] - ETA: 4s - loss: 7.6372 - accuracy: 0.5019
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6395 - accuracy: 0.5018
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6316 - accuracy: 0.5023
15000/25000 [=================>............] - ETA: 3s - loss: 7.6380 - accuracy: 0.5019
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6292 - accuracy: 0.5024
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6323 - accuracy: 0.5022
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6257 - accuracy: 0.5027
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6497 - accuracy: 0.5011
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6720 - accuracy: 0.4997
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6871 - accuracy: 0.4987
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6868 - accuracy: 0.4987
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6545 - accuracy: 0.5008
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7203 - accuracy: 0.4965
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7893 - accuracy: 0.4920 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7778 - accuracy: 0.4927
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.7648 - accuracy: 0.4936
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7458 - accuracy: 0.4948
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6622 - accuracy: 0.5003
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6762 - accuracy: 0.4994
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6666 - accuracy: 0.5000
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6421 - accuracy: 0.5016
11000/25000 [============>.................] - ETA: 4s - loss: 7.6778 - accuracy: 0.4993
12000/25000 [=============>................] - ETA: 4s - loss: 7.6538 - accuracy: 0.5008
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6607 - accuracy: 0.5004
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6513 - accuracy: 0.5010
15000/25000 [=================>............] - ETA: 3s - loss: 7.6574 - accuracy: 0.5006
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6609 - accuracy: 0.5004
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6603 - accuracy: 0.5004
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6487 - accuracy: 0.5012
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6247 - accuracy: 0.5027
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6268 - accuracy: 0.5026
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6338 - accuracy: 0.5021
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6283 - accuracy: 0.5025
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6440 - accuracy: 0.5015
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6526 - accuracy: 0.5009
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master 6a5c679] ml_store  && git pull --all
 1 file changed, 317 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + 73898eb...6a5c679 master -> master (forced update)





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

13/13 [==============================] - 2s 126ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 6ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master b923674] ml_store  && git pull --all
 1 file changed, 125 insertions(+)
To github.com:arita37/mlmodels_store.git
   6a5c679..b923674  master -> master





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
 4177920/11490434 [=========>....................] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:56 - loss: 2.3009 - categorical_accuracy: 0.0938
   64/60000 [..............................] - ETA: 4:59 - loss: 2.2821 - categorical_accuracy: 0.1562
   96/60000 [..............................] - ETA: 3:59 - loss: 2.2184 - categorical_accuracy: 0.2083
  128/60000 [..............................] - ETA: 3:27 - loss: 2.1800 - categorical_accuracy: 0.2188
  160/60000 [..............................] - ETA: 3:08 - loss: 2.2004 - categorical_accuracy: 0.2125
  192/60000 [..............................] - ETA: 2:56 - loss: 2.1870 - categorical_accuracy: 0.2396
  224/60000 [..............................] - ETA: 2:47 - loss: 2.1384 - categorical_accuracy: 0.2723
  256/60000 [..............................] - ETA: 2:40 - loss: 2.0999 - categorical_accuracy: 0.2930
  288/60000 [..............................] - ETA: 2:35 - loss: 2.0627 - categorical_accuracy: 0.3056
  320/60000 [..............................] - ETA: 2:31 - loss: 2.0240 - categorical_accuracy: 0.3156
  352/60000 [..............................] - ETA: 2:27 - loss: 1.9760 - categorical_accuracy: 0.3324
  384/60000 [..............................] - ETA: 2:24 - loss: 1.9572 - categorical_accuracy: 0.3438
  416/60000 [..............................] - ETA: 2:23 - loss: 1.9318 - categorical_accuracy: 0.3510
  448/60000 [..............................] - ETA: 2:20 - loss: 1.8881 - categorical_accuracy: 0.3705
  480/60000 [..............................] - ETA: 2:18 - loss: 1.8484 - categorical_accuracy: 0.3875
  512/60000 [..............................] - ETA: 2:17 - loss: 1.8338 - categorical_accuracy: 0.3906
  544/60000 [..............................] - ETA: 2:16 - loss: 1.8210 - categorical_accuracy: 0.3989
  576/60000 [..............................] - ETA: 2:14 - loss: 1.7924 - categorical_accuracy: 0.4080
  608/60000 [..............................] - ETA: 2:13 - loss: 1.7640 - categorical_accuracy: 0.4194
  640/60000 [..............................] - ETA: 2:12 - loss: 1.7225 - categorical_accuracy: 0.4313
  672/60000 [..............................] - ETA: 2:11 - loss: 1.6819 - categorical_accuracy: 0.4464
  704/60000 [..............................] - ETA: 2:10 - loss: 1.6606 - categorical_accuracy: 0.4545
  736/60000 [..............................] - ETA: 2:09 - loss: 1.6503 - categorical_accuracy: 0.4565
  768/60000 [..............................] - ETA: 2:08 - loss: 1.6301 - categorical_accuracy: 0.4635
  800/60000 [..............................] - ETA: 2:07 - loss: 1.5986 - categorical_accuracy: 0.4712
  832/60000 [..............................] - ETA: 2:07 - loss: 1.5816 - categorical_accuracy: 0.4772
  864/60000 [..............................] - ETA: 2:06 - loss: 1.5668 - categorical_accuracy: 0.4815
  896/60000 [..............................] - ETA: 2:05 - loss: 1.5530 - categorical_accuracy: 0.4877
  928/60000 [..............................] - ETA: 2:05 - loss: 1.5435 - categorical_accuracy: 0.4935
  960/60000 [..............................] - ETA: 2:04 - loss: 1.5268 - categorical_accuracy: 0.5010
  992/60000 [..............................] - ETA: 2:04 - loss: 1.5099 - categorical_accuracy: 0.5060
 1024/60000 [..............................] - ETA: 2:04 - loss: 1.4875 - categorical_accuracy: 0.5127
 1056/60000 [..............................] - ETA: 2:04 - loss: 1.4587 - categorical_accuracy: 0.5237
 1088/60000 [..............................] - ETA: 2:03 - loss: 1.4505 - categorical_accuracy: 0.5248
 1120/60000 [..............................] - ETA: 2:03 - loss: 1.4299 - categorical_accuracy: 0.5312
 1152/60000 [..............................] - ETA: 2:02 - loss: 1.4099 - categorical_accuracy: 0.5391
 1184/60000 [..............................] - ETA: 2:02 - loss: 1.3959 - categorical_accuracy: 0.5456
 1216/60000 [..............................] - ETA: 2:02 - loss: 1.3749 - categorical_accuracy: 0.5526
 1248/60000 [..............................] - ETA: 2:01 - loss: 1.3593 - categorical_accuracy: 0.5577
 1280/60000 [..............................] - ETA: 2:01 - loss: 1.3415 - categorical_accuracy: 0.5633
 1312/60000 [..............................] - ETA: 2:01 - loss: 1.3278 - categorical_accuracy: 0.5678
 1344/60000 [..............................] - ETA: 2:01 - loss: 1.3153 - categorical_accuracy: 0.5714
 1376/60000 [..............................] - ETA: 2:00 - loss: 1.2983 - categorical_accuracy: 0.5756
 1408/60000 [..............................] - ETA: 2:00 - loss: 1.2829 - categorical_accuracy: 0.5788
 1440/60000 [..............................] - ETA: 2:00 - loss: 1.2692 - categorical_accuracy: 0.5826
 1472/60000 [..............................] - ETA: 2:00 - loss: 1.2520 - categorical_accuracy: 0.5876
 1504/60000 [..............................] - ETA: 1:59 - loss: 1.2391 - categorical_accuracy: 0.5931
 1536/60000 [..............................] - ETA: 1:59 - loss: 1.2241 - categorical_accuracy: 0.5977
 1568/60000 [..............................] - ETA: 1:59 - loss: 1.2203 - categorical_accuracy: 0.5969
 1600/60000 [..............................] - ETA: 1:59 - loss: 1.2119 - categorical_accuracy: 0.5994
 1632/60000 [..............................] - ETA: 1:59 - loss: 1.2012 - categorical_accuracy: 0.6029
 1664/60000 [..............................] - ETA: 1:59 - loss: 1.1885 - categorical_accuracy: 0.6064
 1696/60000 [..............................] - ETA: 1:58 - loss: 1.1809 - categorical_accuracy: 0.6085
 1728/60000 [..............................] - ETA: 1:58 - loss: 1.1698 - categorical_accuracy: 0.6111
 1760/60000 [..............................] - ETA: 1:58 - loss: 1.1562 - categorical_accuracy: 0.6165
 1792/60000 [..............................] - ETA: 1:58 - loss: 1.1451 - categorical_accuracy: 0.6217
 1824/60000 [..............................] - ETA: 1:58 - loss: 1.1319 - categorical_accuracy: 0.6261
 1856/60000 [..............................] - ETA: 1:58 - loss: 1.1207 - categorical_accuracy: 0.6298
 1888/60000 [..............................] - ETA: 1:58 - loss: 1.1118 - categorical_accuracy: 0.6329
 1920/60000 [..............................] - ETA: 1:58 - loss: 1.1015 - categorical_accuracy: 0.6365
 1952/60000 [..............................] - ETA: 1:57 - loss: 1.0964 - categorical_accuracy: 0.6393
 1984/60000 [..............................] - ETA: 1:58 - loss: 1.0923 - categorical_accuracy: 0.6396
 2016/60000 [>.............................] - ETA: 1:57 - loss: 1.0829 - categorical_accuracy: 0.6429
 2048/60000 [>.............................] - ETA: 1:57 - loss: 1.0733 - categorical_accuracy: 0.6450
 2080/60000 [>.............................] - ETA: 1:57 - loss: 1.0662 - categorical_accuracy: 0.6481
 2112/60000 [>.............................] - ETA: 1:57 - loss: 1.0579 - categorical_accuracy: 0.6501
 2144/60000 [>.............................] - ETA: 1:57 - loss: 1.0520 - categorical_accuracy: 0.6525
 2176/60000 [>.............................] - ETA: 1:56 - loss: 1.0434 - categorical_accuracy: 0.6562
 2208/60000 [>.............................] - ETA: 1:56 - loss: 1.0426 - categorical_accuracy: 0.6558
 2240/60000 [>.............................] - ETA: 1:56 - loss: 1.0341 - categorical_accuracy: 0.6594
 2272/60000 [>.............................] - ETA: 1:56 - loss: 1.0253 - categorical_accuracy: 0.6629
 2304/60000 [>.............................] - ETA: 1:56 - loss: 1.0202 - categorical_accuracy: 0.6654
 2336/60000 [>.............................] - ETA: 1:55 - loss: 1.0175 - categorical_accuracy: 0.6652
 2368/60000 [>.............................] - ETA: 1:55 - loss: 1.0114 - categorical_accuracy: 0.6677
 2400/60000 [>.............................] - ETA: 1:55 - loss: 1.0030 - categorical_accuracy: 0.6708
 2432/60000 [>.............................] - ETA: 1:55 - loss: 0.9945 - categorical_accuracy: 0.6731
 2464/60000 [>.............................] - ETA: 1:55 - loss: 0.9868 - categorical_accuracy: 0.6757
 2496/60000 [>.............................] - ETA: 1:55 - loss: 0.9803 - categorical_accuracy: 0.6771
 2528/60000 [>.............................] - ETA: 1:55 - loss: 0.9745 - categorical_accuracy: 0.6788
 2560/60000 [>.............................] - ETA: 1:54 - loss: 0.9664 - categorical_accuracy: 0.6816
 2592/60000 [>.............................] - ETA: 1:54 - loss: 0.9603 - categorical_accuracy: 0.6840
 2624/60000 [>.............................] - ETA: 1:54 - loss: 0.9553 - categorical_accuracy: 0.6852
 2656/60000 [>.............................] - ETA: 1:54 - loss: 0.9546 - categorical_accuracy: 0.6849
 2688/60000 [>.............................] - ETA: 1:54 - loss: 0.9516 - categorical_accuracy: 0.6864
 2720/60000 [>.............................] - ETA: 1:54 - loss: 0.9487 - categorical_accuracy: 0.6868
 2752/60000 [>.............................] - ETA: 1:53 - loss: 0.9414 - categorical_accuracy: 0.6893
 2784/60000 [>.............................] - ETA: 1:53 - loss: 0.9360 - categorical_accuracy: 0.6915
 2816/60000 [>.............................] - ETA: 1:53 - loss: 0.9284 - categorical_accuracy: 0.6935
 2848/60000 [>.............................] - ETA: 1:53 - loss: 0.9224 - categorical_accuracy: 0.6956
 2880/60000 [>.............................] - ETA: 1:53 - loss: 0.9174 - categorical_accuracy: 0.6972
 2912/60000 [>.............................] - ETA: 1:53 - loss: 0.9101 - categorical_accuracy: 0.7002
 2944/60000 [>.............................] - ETA: 1:53 - loss: 0.9072 - categorical_accuracy: 0.7021
 2976/60000 [>.............................] - ETA: 1:53 - loss: 0.9036 - categorical_accuracy: 0.7033
 3008/60000 [>.............................] - ETA: 1:53 - loss: 0.8980 - categorical_accuracy: 0.7051
 3040/60000 [>.............................] - ETA: 1:52 - loss: 0.8945 - categorical_accuracy: 0.7063
 3072/60000 [>.............................] - ETA: 1:52 - loss: 0.8905 - categorical_accuracy: 0.7074
 3104/60000 [>.............................] - ETA: 1:52 - loss: 0.8833 - categorical_accuracy: 0.7094
 3136/60000 [>.............................] - ETA: 1:52 - loss: 0.8794 - categorical_accuracy: 0.7114
 3168/60000 [>.............................] - ETA: 1:52 - loss: 0.8767 - categorical_accuracy: 0.7121
 3200/60000 [>.............................] - ETA: 1:52 - loss: 0.8694 - categorical_accuracy: 0.7147
 3232/60000 [>.............................] - ETA: 1:52 - loss: 0.8639 - categorical_accuracy: 0.7160
 3264/60000 [>.............................] - ETA: 1:52 - loss: 0.8609 - categorical_accuracy: 0.7178
 3296/60000 [>.............................] - ETA: 1:52 - loss: 0.8580 - categorical_accuracy: 0.7191
 3328/60000 [>.............................] - ETA: 1:51 - loss: 0.8541 - categorical_accuracy: 0.7203
 3360/60000 [>.............................] - ETA: 1:51 - loss: 0.8492 - categorical_accuracy: 0.7223
 3392/60000 [>.............................] - ETA: 1:51 - loss: 0.8454 - categorical_accuracy: 0.7238
 3424/60000 [>.............................] - ETA: 1:51 - loss: 0.8398 - categorical_accuracy: 0.7261
 3456/60000 [>.............................] - ETA: 1:51 - loss: 0.8364 - categorical_accuracy: 0.7274
 3488/60000 [>.............................] - ETA: 1:51 - loss: 0.8310 - categorical_accuracy: 0.7291
 3520/60000 [>.............................] - ETA: 1:51 - loss: 0.8268 - categorical_accuracy: 0.7307
 3552/60000 [>.............................] - ETA: 1:51 - loss: 0.8214 - categorical_accuracy: 0.7325
 3584/60000 [>.............................] - ETA: 1:51 - loss: 0.8162 - categorical_accuracy: 0.7344
 3616/60000 [>.............................] - ETA: 1:51 - loss: 0.8158 - categorical_accuracy: 0.7351
 3648/60000 [>.............................] - ETA: 1:51 - loss: 0.8152 - categorical_accuracy: 0.7360
 3680/60000 [>.............................] - ETA: 1:51 - loss: 0.8122 - categorical_accuracy: 0.7375
 3712/60000 [>.............................] - ETA: 1:51 - loss: 0.8080 - categorical_accuracy: 0.7390
 3744/60000 [>.............................] - ETA: 1:51 - loss: 0.8038 - categorical_accuracy: 0.7401
 3776/60000 [>.............................] - ETA: 1:50 - loss: 0.7999 - categorical_accuracy: 0.7413
 3808/60000 [>.............................] - ETA: 1:50 - loss: 0.7965 - categorical_accuracy: 0.7421
 3840/60000 [>.............................] - ETA: 1:50 - loss: 0.7947 - categorical_accuracy: 0.7430
 3872/60000 [>.............................] - ETA: 1:50 - loss: 0.7919 - categorical_accuracy: 0.7435
 3904/60000 [>.............................] - ETA: 1:50 - loss: 0.7884 - categorical_accuracy: 0.7446
 3936/60000 [>.............................] - ETA: 1:50 - loss: 0.7853 - categorical_accuracy: 0.7449
 3968/60000 [>.............................] - ETA: 1:50 - loss: 0.7810 - categorical_accuracy: 0.7462
 4000/60000 [=>............................] - ETA: 1:50 - loss: 0.7764 - categorical_accuracy: 0.7475
 4032/60000 [=>............................] - ETA: 1:50 - loss: 0.7730 - categorical_accuracy: 0.7490
 4064/60000 [=>............................] - ETA: 1:50 - loss: 0.7706 - categorical_accuracy: 0.7500
 4096/60000 [=>............................] - ETA: 1:50 - loss: 0.7697 - categorical_accuracy: 0.7505
 4128/60000 [=>............................] - ETA: 1:49 - loss: 0.7654 - categorical_accuracy: 0.7524
 4160/60000 [=>............................] - ETA: 1:49 - loss: 0.7633 - categorical_accuracy: 0.7531
 4192/60000 [=>............................] - ETA: 1:49 - loss: 0.7601 - categorical_accuracy: 0.7541
 4224/60000 [=>............................] - ETA: 1:49 - loss: 0.7580 - categorical_accuracy: 0.7547
 4256/60000 [=>............................] - ETA: 1:49 - loss: 0.7543 - categorical_accuracy: 0.7556
 4288/60000 [=>............................] - ETA: 1:49 - loss: 0.7501 - categorical_accuracy: 0.7570
 4320/60000 [=>............................] - ETA: 1:49 - loss: 0.7459 - categorical_accuracy: 0.7583
 4352/60000 [=>............................] - ETA: 1:49 - loss: 0.7424 - categorical_accuracy: 0.7594
 4384/60000 [=>............................] - ETA: 1:49 - loss: 0.7426 - categorical_accuracy: 0.7594
 4416/60000 [=>............................] - ETA: 1:49 - loss: 0.7407 - categorical_accuracy: 0.7595
 4448/60000 [=>............................] - ETA: 1:49 - loss: 0.7398 - categorical_accuracy: 0.7601
 4480/60000 [=>............................] - ETA: 1:48 - loss: 0.7375 - categorical_accuracy: 0.7603
 4512/60000 [=>............................] - ETA: 1:48 - loss: 0.7335 - categorical_accuracy: 0.7617
 4544/60000 [=>............................] - ETA: 1:48 - loss: 0.7309 - categorical_accuracy: 0.7623
 4576/60000 [=>............................] - ETA: 1:48 - loss: 0.7275 - categorical_accuracy: 0.7635
 4608/60000 [=>............................] - ETA: 1:48 - loss: 0.7255 - categorical_accuracy: 0.7645
 4640/60000 [=>............................] - ETA: 1:48 - loss: 0.7211 - categorical_accuracy: 0.7662
 4672/60000 [=>............................] - ETA: 1:48 - loss: 0.7216 - categorical_accuracy: 0.7665
 4704/60000 [=>............................] - ETA: 1:48 - loss: 0.7178 - categorical_accuracy: 0.7681
 4736/60000 [=>............................] - ETA: 1:48 - loss: 0.7144 - categorical_accuracy: 0.7692
 4768/60000 [=>............................] - ETA: 1:48 - loss: 0.7118 - categorical_accuracy: 0.7703
 4800/60000 [=>............................] - ETA: 1:48 - loss: 0.7087 - categorical_accuracy: 0.7715
 4832/60000 [=>............................] - ETA: 1:48 - loss: 0.7093 - categorical_accuracy: 0.7711
 4864/60000 [=>............................] - ETA: 1:48 - loss: 0.7078 - categorical_accuracy: 0.7720
 4896/60000 [=>............................] - ETA: 1:48 - loss: 0.7041 - categorical_accuracy: 0.7731
 4928/60000 [=>............................] - ETA: 1:47 - loss: 0.7022 - categorical_accuracy: 0.7733
 4960/60000 [=>............................] - ETA: 1:47 - loss: 0.6990 - categorical_accuracy: 0.7742
 4992/60000 [=>............................] - ETA: 1:47 - loss: 0.6967 - categorical_accuracy: 0.7750
 5024/60000 [=>............................] - ETA: 1:47 - loss: 0.6951 - categorical_accuracy: 0.7759
 5056/60000 [=>............................] - ETA: 1:47 - loss: 0.6928 - categorical_accuracy: 0.7765
 5088/60000 [=>............................] - ETA: 1:47 - loss: 0.6898 - categorical_accuracy: 0.7777
 5120/60000 [=>............................] - ETA: 1:47 - loss: 0.6866 - categorical_accuracy: 0.7789
 5152/60000 [=>............................] - ETA: 1:47 - loss: 0.6840 - categorical_accuracy: 0.7797
 5184/60000 [=>............................] - ETA: 1:47 - loss: 0.6830 - categorical_accuracy: 0.7803
 5216/60000 [=>............................] - ETA: 1:47 - loss: 0.6812 - categorical_accuracy: 0.7811
 5248/60000 [=>............................] - ETA: 1:47 - loss: 0.6805 - categorical_accuracy: 0.7814
 5280/60000 [=>............................] - ETA: 1:47 - loss: 0.6786 - categorical_accuracy: 0.7820
 5312/60000 [=>............................] - ETA: 1:47 - loss: 0.6759 - categorical_accuracy: 0.7828
 5344/60000 [=>............................] - ETA: 1:46 - loss: 0.6734 - categorical_accuracy: 0.7837
 5376/60000 [=>............................] - ETA: 1:46 - loss: 0.6723 - categorical_accuracy: 0.7846
 5408/60000 [=>............................] - ETA: 1:46 - loss: 0.6699 - categorical_accuracy: 0.7853
 5440/60000 [=>............................] - ETA: 1:46 - loss: 0.6674 - categorical_accuracy: 0.7860
 5472/60000 [=>............................] - ETA: 1:46 - loss: 0.6667 - categorical_accuracy: 0.7862
 5504/60000 [=>............................] - ETA: 1:46 - loss: 0.6645 - categorical_accuracy: 0.7871
 5536/60000 [=>............................] - ETA: 1:46 - loss: 0.6624 - categorical_accuracy: 0.7878
 5568/60000 [=>............................] - ETA: 1:46 - loss: 0.6596 - categorical_accuracy: 0.7888
 5600/60000 [=>............................] - ETA: 1:46 - loss: 0.6577 - categorical_accuracy: 0.7893
 5632/60000 [=>............................] - ETA: 1:46 - loss: 0.6548 - categorical_accuracy: 0.7900
 5664/60000 [=>............................] - ETA: 1:46 - loss: 0.6531 - categorical_accuracy: 0.7908
 5696/60000 [=>............................] - ETA: 1:46 - loss: 0.6504 - categorical_accuracy: 0.7918
 5728/60000 [=>............................] - ETA: 1:46 - loss: 0.6482 - categorical_accuracy: 0.7924
 5760/60000 [=>............................] - ETA: 1:45 - loss: 0.6461 - categorical_accuracy: 0.7931
 5792/60000 [=>............................] - ETA: 1:45 - loss: 0.6446 - categorical_accuracy: 0.7935
 5824/60000 [=>............................] - ETA: 1:45 - loss: 0.6423 - categorical_accuracy: 0.7943
 5856/60000 [=>............................] - ETA: 1:45 - loss: 0.6395 - categorical_accuracy: 0.7953
 5888/60000 [=>............................] - ETA: 1:45 - loss: 0.6371 - categorical_accuracy: 0.7960
 5920/60000 [=>............................] - ETA: 1:45 - loss: 0.6365 - categorical_accuracy: 0.7966
 5952/60000 [=>............................] - ETA: 1:45 - loss: 0.6338 - categorical_accuracy: 0.7975
 5984/60000 [=>............................] - ETA: 1:45 - loss: 0.6312 - categorical_accuracy: 0.7985
 6016/60000 [==>...........................] - ETA: 1:45 - loss: 0.6303 - categorical_accuracy: 0.7989
 6048/60000 [==>...........................] - ETA: 1:45 - loss: 0.6290 - categorical_accuracy: 0.7994
 6080/60000 [==>...........................] - ETA: 1:45 - loss: 0.6270 - categorical_accuracy: 0.8003
 6112/60000 [==>...........................] - ETA: 1:45 - loss: 0.6245 - categorical_accuracy: 0.8010
 6144/60000 [==>...........................] - ETA: 1:45 - loss: 0.6222 - categorical_accuracy: 0.8014
 6176/60000 [==>...........................] - ETA: 1:45 - loss: 0.6206 - categorical_accuracy: 0.8018
 6208/60000 [==>...........................] - ETA: 1:44 - loss: 0.6190 - categorical_accuracy: 0.8024
 6240/60000 [==>...........................] - ETA: 1:44 - loss: 0.6182 - categorical_accuracy: 0.8027
 6272/60000 [==>...........................] - ETA: 1:44 - loss: 0.6168 - categorical_accuracy: 0.8034
 6304/60000 [==>...........................] - ETA: 1:44 - loss: 0.6157 - categorical_accuracy: 0.8039
 6336/60000 [==>...........................] - ETA: 1:44 - loss: 0.6133 - categorical_accuracy: 0.8046
 6368/60000 [==>...........................] - ETA: 1:44 - loss: 0.6109 - categorical_accuracy: 0.8056
 6400/60000 [==>...........................] - ETA: 1:44 - loss: 0.6101 - categorical_accuracy: 0.8062
 6432/60000 [==>...........................] - ETA: 1:44 - loss: 0.6081 - categorical_accuracy: 0.8067
 6464/60000 [==>...........................] - ETA: 1:44 - loss: 0.6061 - categorical_accuracy: 0.8074
 6496/60000 [==>...........................] - ETA: 1:44 - loss: 0.6055 - categorical_accuracy: 0.8076
 6528/60000 [==>...........................] - ETA: 1:44 - loss: 0.6047 - categorical_accuracy: 0.8078
 6560/60000 [==>...........................] - ETA: 1:44 - loss: 0.6055 - categorical_accuracy: 0.8076
 6592/60000 [==>...........................] - ETA: 1:44 - loss: 0.6029 - categorical_accuracy: 0.8086
 6624/60000 [==>...........................] - ETA: 1:44 - loss: 0.6016 - categorical_accuracy: 0.8090
 6656/60000 [==>...........................] - ETA: 1:44 - loss: 0.6000 - categorical_accuracy: 0.8098
 6688/60000 [==>...........................] - ETA: 1:43 - loss: 0.5987 - categorical_accuracy: 0.8103
 6720/60000 [==>...........................] - ETA: 1:43 - loss: 0.5974 - categorical_accuracy: 0.8106
 6752/60000 [==>...........................] - ETA: 1:43 - loss: 0.5955 - categorical_accuracy: 0.8113
 6784/60000 [==>...........................] - ETA: 1:43 - loss: 0.5951 - categorical_accuracy: 0.8115
 6816/60000 [==>...........................] - ETA: 1:43 - loss: 0.5938 - categorical_accuracy: 0.8118
 6848/60000 [==>...........................] - ETA: 1:43 - loss: 0.5919 - categorical_accuracy: 0.8124
 6880/60000 [==>...........................] - ETA: 1:43 - loss: 0.5901 - categorical_accuracy: 0.8129
 6912/60000 [==>...........................] - ETA: 1:43 - loss: 0.5890 - categorical_accuracy: 0.8135
 6944/60000 [==>...........................] - ETA: 1:43 - loss: 0.5875 - categorical_accuracy: 0.8137
 6976/60000 [==>...........................] - ETA: 1:43 - loss: 0.5852 - categorical_accuracy: 0.8145
 7008/60000 [==>...........................] - ETA: 1:43 - loss: 0.5839 - categorical_accuracy: 0.8151
 7040/60000 [==>...........................] - ETA: 1:43 - loss: 0.5817 - categorical_accuracy: 0.8159
 7072/60000 [==>...........................] - ETA: 1:43 - loss: 0.5808 - categorical_accuracy: 0.8163
 7104/60000 [==>...........................] - ETA: 1:43 - loss: 0.5793 - categorical_accuracy: 0.8169
 7136/60000 [==>...........................] - ETA: 1:42 - loss: 0.5773 - categorical_accuracy: 0.8175
 7168/60000 [==>...........................] - ETA: 1:42 - loss: 0.5769 - categorical_accuracy: 0.8178
 7200/60000 [==>...........................] - ETA: 1:42 - loss: 0.5755 - categorical_accuracy: 0.8182
 7232/60000 [==>...........................] - ETA: 1:42 - loss: 0.5740 - categorical_accuracy: 0.8187
 7264/60000 [==>...........................] - ETA: 1:42 - loss: 0.5718 - categorical_accuracy: 0.8195
 7296/60000 [==>...........................] - ETA: 1:42 - loss: 0.5697 - categorical_accuracy: 0.8202
 7328/60000 [==>...........................] - ETA: 1:42 - loss: 0.5681 - categorical_accuracy: 0.8207
 7360/60000 [==>...........................] - ETA: 1:42 - loss: 0.5664 - categorical_accuracy: 0.8212
 7392/60000 [==>...........................] - ETA: 1:42 - loss: 0.5646 - categorical_accuracy: 0.8217
 7424/60000 [==>...........................] - ETA: 1:42 - loss: 0.5631 - categorical_accuracy: 0.8219
 7456/60000 [==>...........................] - ETA: 1:42 - loss: 0.5611 - categorical_accuracy: 0.8224
 7488/60000 [==>...........................] - ETA: 1:42 - loss: 0.5596 - categorical_accuracy: 0.8228
 7520/60000 [==>...........................] - ETA: 1:42 - loss: 0.5580 - categorical_accuracy: 0.8231
 7552/60000 [==>...........................] - ETA: 1:42 - loss: 0.5566 - categorical_accuracy: 0.8236
 7584/60000 [==>...........................] - ETA: 1:42 - loss: 0.5567 - categorical_accuracy: 0.8237
 7616/60000 [==>...........................] - ETA: 1:41 - loss: 0.5552 - categorical_accuracy: 0.8242
 7648/60000 [==>...........................] - ETA: 1:41 - loss: 0.5538 - categorical_accuracy: 0.8247
 7680/60000 [==>...........................] - ETA: 1:41 - loss: 0.5533 - categorical_accuracy: 0.8249
 7712/60000 [==>...........................] - ETA: 1:41 - loss: 0.5518 - categorical_accuracy: 0.8253
 7744/60000 [==>...........................] - ETA: 1:41 - loss: 0.5500 - categorical_accuracy: 0.8258
 7776/60000 [==>...........................] - ETA: 1:41 - loss: 0.5507 - categorical_accuracy: 0.8255
 7808/60000 [==>...........................] - ETA: 1:41 - loss: 0.5500 - categorical_accuracy: 0.8258
 7840/60000 [==>...........................] - ETA: 1:41 - loss: 0.5485 - categorical_accuracy: 0.8263
 7872/60000 [==>...........................] - ETA: 1:41 - loss: 0.5469 - categorical_accuracy: 0.8269
 7904/60000 [==>...........................] - ETA: 1:41 - loss: 0.5457 - categorical_accuracy: 0.8273
 7936/60000 [==>...........................] - ETA: 1:41 - loss: 0.5446 - categorical_accuracy: 0.8276
 7968/60000 [==>...........................] - ETA: 1:41 - loss: 0.5434 - categorical_accuracy: 0.8282
 8000/60000 [===>..........................] - ETA: 1:41 - loss: 0.5419 - categorical_accuracy: 0.8288
 8032/60000 [===>..........................] - ETA: 1:40 - loss: 0.5410 - categorical_accuracy: 0.8291
 8064/60000 [===>..........................] - ETA: 1:40 - loss: 0.5391 - categorical_accuracy: 0.8297
 8096/60000 [===>..........................] - ETA: 1:40 - loss: 0.5390 - categorical_accuracy: 0.8299
 8128/60000 [===>..........................] - ETA: 1:40 - loss: 0.5382 - categorical_accuracy: 0.8300
 8160/60000 [===>..........................] - ETA: 1:40 - loss: 0.5378 - categorical_accuracy: 0.8300
 8192/60000 [===>..........................] - ETA: 1:40 - loss: 0.5372 - categorical_accuracy: 0.8304
 8224/60000 [===>..........................] - ETA: 1:40 - loss: 0.5365 - categorical_accuracy: 0.8310
 8256/60000 [===>..........................] - ETA: 1:40 - loss: 0.5350 - categorical_accuracy: 0.8314
 8288/60000 [===>..........................] - ETA: 1:40 - loss: 0.5342 - categorical_accuracy: 0.8317
 8320/60000 [===>..........................] - ETA: 1:40 - loss: 0.5327 - categorical_accuracy: 0.8322
 8352/60000 [===>..........................] - ETA: 1:40 - loss: 0.5313 - categorical_accuracy: 0.8326
 8384/60000 [===>..........................] - ETA: 1:40 - loss: 0.5302 - categorical_accuracy: 0.8328
 8416/60000 [===>..........................] - ETA: 1:40 - loss: 0.5285 - categorical_accuracy: 0.8333
 8448/60000 [===>..........................] - ETA: 1:40 - loss: 0.5272 - categorical_accuracy: 0.8337
 8480/60000 [===>..........................] - ETA: 1:39 - loss: 0.5264 - categorical_accuracy: 0.8338
 8512/60000 [===>..........................] - ETA: 1:39 - loss: 0.5257 - categorical_accuracy: 0.8340
 8544/60000 [===>..........................] - ETA: 1:39 - loss: 0.5247 - categorical_accuracy: 0.8340
 8576/60000 [===>..........................] - ETA: 1:39 - loss: 0.5231 - categorical_accuracy: 0.8345
 8608/60000 [===>..........................] - ETA: 1:39 - loss: 0.5214 - categorical_accuracy: 0.8352
 8640/60000 [===>..........................] - ETA: 1:39 - loss: 0.5200 - categorical_accuracy: 0.8356
 8672/60000 [===>..........................] - ETA: 1:39 - loss: 0.5186 - categorical_accuracy: 0.8361
 8704/60000 [===>..........................] - ETA: 1:39 - loss: 0.5176 - categorical_accuracy: 0.8365
 8736/60000 [===>..........................] - ETA: 1:39 - loss: 0.5161 - categorical_accuracy: 0.8370
 8768/60000 [===>..........................] - ETA: 1:39 - loss: 0.5151 - categorical_accuracy: 0.8371
 8800/60000 [===>..........................] - ETA: 1:39 - loss: 0.5139 - categorical_accuracy: 0.8375
 8832/60000 [===>..........................] - ETA: 1:39 - loss: 0.5132 - categorical_accuracy: 0.8380
 8864/60000 [===>..........................] - ETA: 1:39 - loss: 0.5124 - categorical_accuracy: 0.8384
 8896/60000 [===>..........................] - ETA: 1:39 - loss: 0.5110 - categorical_accuracy: 0.8389
 8928/60000 [===>..........................] - ETA: 1:39 - loss: 0.5099 - categorical_accuracy: 0.8392
 8960/60000 [===>..........................] - ETA: 1:39 - loss: 0.5085 - categorical_accuracy: 0.8397
 8992/60000 [===>..........................] - ETA: 1:38 - loss: 0.5079 - categorical_accuracy: 0.8400
 9024/60000 [===>..........................] - ETA: 1:38 - loss: 0.5073 - categorical_accuracy: 0.8402
 9056/60000 [===>..........................] - ETA: 1:38 - loss: 0.5058 - categorical_accuracy: 0.8407
 9088/60000 [===>..........................] - ETA: 1:38 - loss: 0.5044 - categorical_accuracy: 0.8412
 9120/60000 [===>..........................] - ETA: 1:38 - loss: 0.5041 - categorical_accuracy: 0.8413
 9152/60000 [===>..........................] - ETA: 1:38 - loss: 0.5025 - categorical_accuracy: 0.8419
 9184/60000 [===>..........................] - ETA: 1:38 - loss: 0.5014 - categorical_accuracy: 0.8422
 9216/60000 [===>..........................] - ETA: 1:38 - loss: 0.4999 - categorical_accuracy: 0.8428
 9248/60000 [===>..........................] - ETA: 1:38 - loss: 0.4987 - categorical_accuracy: 0.8431
 9280/60000 [===>..........................] - ETA: 1:38 - loss: 0.4995 - categorical_accuracy: 0.8432
 9312/60000 [===>..........................] - ETA: 1:38 - loss: 0.4980 - categorical_accuracy: 0.8438
 9344/60000 [===>..........................] - ETA: 1:38 - loss: 0.4966 - categorical_accuracy: 0.8443
 9376/60000 [===>..........................] - ETA: 1:38 - loss: 0.4959 - categorical_accuracy: 0.8445
 9408/60000 [===>..........................] - ETA: 1:38 - loss: 0.4952 - categorical_accuracy: 0.8447
 9440/60000 [===>..........................] - ETA: 1:38 - loss: 0.4956 - categorical_accuracy: 0.8447
 9472/60000 [===>..........................] - ETA: 1:37 - loss: 0.4943 - categorical_accuracy: 0.8451
 9504/60000 [===>..........................] - ETA: 1:37 - loss: 0.4931 - categorical_accuracy: 0.8454
 9536/60000 [===>..........................] - ETA: 1:37 - loss: 0.4920 - categorical_accuracy: 0.8457
 9568/60000 [===>..........................] - ETA: 1:37 - loss: 0.4913 - categorical_accuracy: 0.8458
 9600/60000 [===>..........................] - ETA: 1:37 - loss: 0.4899 - categorical_accuracy: 0.8464
 9632/60000 [===>..........................] - ETA: 1:37 - loss: 0.4892 - categorical_accuracy: 0.8467
 9664/60000 [===>..........................] - ETA: 1:37 - loss: 0.4880 - categorical_accuracy: 0.8471
 9696/60000 [===>..........................] - ETA: 1:37 - loss: 0.4871 - categorical_accuracy: 0.8472
 9728/60000 [===>..........................] - ETA: 1:37 - loss: 0.4864 - categorical_accuracy: 0.8475
 9760/60000 [===>..........................] - ETA: 1:37 - loss: 0.4852 - categorical_accuracy: 0.8478
 9792/60000 [===>..........................] - ETA: 1:37 - loss: 0.4841 - categorical_accuracy: 0.8481
 9824/60000 [===>..........................] - ETA: 1:37 - loss: 0.4841 - categorical_accuracy: 0.8482
 9856/60000 [===>..........................] - ETA: 1:37 - loss: 0.4832 - categorical_accuracy: 0.8484
 9888/60000 [===>..........................] - ETA: 1:37 - loss: 0.4819 - categorical_accuracy: 0.8489
 9920/60000 [===>..........................] - ETA: 1:36 - loss: 0.4805 - categorical_accuracy: 0.8494
 9952/60000 [===>..........................] - ETA: 1:36 - loss: 0.4794 - categorical_accuracy: 0.8497
 9984/60000 [===>..........................] - ETA: 1:36 - loss: 0.4785 - categorical_accuracy: 0.8500
10016/60000 [====>.........................] - ETA: 1:36 - loss: 0.4771 - categorical_accuracy: 0.8504
10048/60000 [====>.........................] - ETA: 1:36 - loss: 0.4764 - categorical_accuracy: 0.8507
10080/60000 [====>.........................] - ETA: 1:36 - loss: 0.4755 - categorical_accuracy: 0.8509
10112/60000 [====>.........................] - ETA: 1:36 - loss: 0.4744 - categorical_accuracy: 0.8513
10144/60000 [====>.........................] - ETA: 1:36 - loss: 0.4733 - categorical_accuracy: 0.8514
10176/60000 [====>.........................] - ETA: 1:36 - loss: 0.4721 - categorical_accuracy: 0.8518
10208/60000 [====>.........................] - ETA: 1:36 - loss: 0.4712 - categorical_accuracy: 0.8522
10240/60000 [====>.........................] - ETA: 1:36 - loss: 0.4701 - categorical_accuracy: 0.8525
10272/60000 [====>.........................] - ETA: 1:36 - loss: 0.4695 - categorical_accuracy: 0.8527
10304/60000 [====>.........................] - ETA: 1:36 - loss: 0.4689 - categorical_accuracy: 0.8530
10336/60000 [====>.........................] - ETA: 1:35 - loss: 0.4682 - categorical_accuracy: 0.8532
10368/60000 [====>.........................] - ETA: 1:35 - loss: 0.4672 - categorical_accuracy: 0.8534
10400/60000 [====>.........................] - ETA: 1:35 - loss: 0.4664 - categorical_accuracy: 0.8536
10432/60000 [====>.........................] - ETA: 1:35 - loss: 0.4656 - categorical_accuracy: 0.8538
10464/60000 [====>.........................] - ETA: 1:35 - loss: 0.4649 - categorical_accuracy: 0.8542
10496/60000 [====>.........................] - ETA: 1:35 - loss: 0.4638 - categorical_accuracy: 0.8544
10528/60000 [====>.........................] - ETA: 1:35 - loss: 0.4631 - categorical_accuracy: 0.8546
10560/60000 [====>.........................] - ETA: 1:35 - loss: 0.4639 - categorical_accuracy: 0.8545
10592/60000 [====>.........................] - ETA: 1:35 - loss: 0.4633 - categorical_accuracy: 0.8547
10624/60000 [====>.........................] - ETA: 1:35 - loss: 0.4625 - categorical_accuracy: 0.8549
10656/60000 [====>.........................] - ETA: 1:35 - loss: 0.4616 - categorical_accuracy: 0.8550
10688/60000 [====>.........................] - ETA: 1:35 - loss: 0.4609 - categorical_accuracy: 0.8552
10720/60000 [====>.........................] - ETA: 1:35 - loss: 0.4599 - categorical_accuracy: 0.8556
10752/60000 [====>.........................] - ETA: 1:35 - loss: 0.4587 - categorical_accuracy: 0.8560
10784/60000 [====>.........................] - ETA: 1:35 - loss: 0.4581 - categorical_accuracy: 0.8562
10816/60000 [====>.........................] - ETA: 1:34 - loss: 0.4576 - categorical_accuracy: 0.8563
10848/60000 [====>.........................] - ETA: 1:34 - loss: 0.4564 - categorical_accuracy: 0.8567
10880/60000 [====>.........................] - ETA: 1:34 - loss: 0.4558 - categorical_accuracy: 0.8569
10912/60000 [====>.........................] - ETA: 1:34 - loss: 0.4546 - categorical_accuracy: 0.8573
10944/60000 [====>.........................] - ETA: 1:34 - loss: 0.4540 - categorical_accuracy: 0.8575
10976/60000 [====>.........................] - ETA: 1:34 - loss: 0.4532 - categorical_accuracy: 0.8576
11008/60000 [====>.........................] - ETA: 1:34 - loss: 0.4525 - categorical_accuracy: 0.8576
11040/60000 [====>.........................] - ETA: 1:34 - loss: 0.4522 - categorical_accuracy: 0.8577
11072/60000 [====>.........................] - ETA: 1:34 - loss: 0.4513 - categorical_accuracy: 0.8580
11104/60000 [====>.........................] - ETA: 1:34 - loss: 0.4506 - categorical_accuracy: 0.8582
11136/60000 [====>.........................] - ETA: 1:34 - loss: 0.4498 - categorical_accuracy: 0.8583
11168/60000 [====>.........................] - ETA: 1:34 - loss: 0.4491 - categorical_accuracy: 0.8585
11200/60000 [====>.........................] - ETA: 1:34 - loss: 0.4488 - categorical_accuracy: 0.8586
11232/60000 [====>.........................] - ETA: 1:34 - loss: 0.4480 - categorical_accuracy: 0.8588
11264/60000 [====>.........................] - ETA: 1:34 - loss: 0.4469 - categorical_accuracy: 0.8591
11296/60000 [====>.........................] - ETA: 1:34 - loss: 0.4459 - categorical_accuracy: 0.8594
11328/60000 [====>.........................] - ETA: 1:33 - loss: 0.4453 - categorical_accuracy: 0.8596
11360/60000 [====>.........................] - ETA: 1:33 - loss: 0.4447 - categorical_accuracy: 0.8597
11392/60000 [====>.........................] - ETA: 1:33 - loss: 0.4439 - categorical_accuracy: 0.8600
11424/60000 [====>.........................] - ETA: 1:33 - loss: 0.4433 - categorical_accuracy: 0.8602
11456/60000 [====>.........................] - ETA: 1:33 - loss: 0.4432 - categorical_accuracy: 0.8602
11488/60000 [====>.........................] - ETA: 1:33 - loss: 0.4422 - categorical_accuracy: 0.8606
11520/60000 [====>.........................] - ETA: 1:33 - loss: 0.4415 - categorical_accuracy: 0.8608
11552/60000 [====>.........................] - ETA: 1:33 - loss: 0.4406 - categorical_accuracy: 0.8611
11584/60000 [====>.........................] - ETA: 1:33 - loss: 0.4399 - categorical_accuracy: 0.8613
11616/60000 [====>.........................] - ETA: 1:33 - loss: 0.4392 - categorical_accuracy: 0.8616
11648/60000 [====>.........................] - ETA: 1:33 - loss: 0.4392 - categorical_accuracy: 0.8617
11680/60000 [====>.........................] - ETA: 1:33 - loss: 0.4394 - categorical_accuracy: 0.8616
11712/60000 [====>.........................] - ETA: 1:33 - loss: 0.4385 - categorical_accuracy: 0.8619
11744/60000 [====>.........................] - ETA: 1:33 - loss: 0.4380 - categorical_accuracy: 0.8621
11776/60000 [====>.........................] - ETA: 1:33 - loss: 0.4375 - categorical_accuracy: 0.8622
11808/60000 [====>.........................] - ETA: 1:33 - loss: 0.4369 - categorical_accuracy: 0.8622
11840/60000 [====>.........................] - ETA: 1:32 - loss: 0.4368 - categorical_accuracy: 0.8623
11872/60000 [====>.........................] - ETA: 1:32 - loss: 0.4367 - categorical_accuracy: 0.8623
11904/60000 [====>.........................] - ETA: 1:32 - loss: 0.4363 - categorical_accuracy: 0.8625
11936/60000 [====>.........................] - ETA: 1:32 - loss: 0.4357 - categorical_accuracy: 0.8627
11968/60000 [====>.........................] - ETA: 1:32 - loss: 0.4348 - categorical_accuracy: 0.8630
12000/60000 [=====>........................] - ETA: 1:32 - loss: 0.4340 - categorical_accuracy: 0.8632
12032/60000 [=====>........................] - ETA: 1:32 - loss: 0.4338 - categorical_accuracy: 0.8634
12064/60000 [=====>........................] - ETA: 1:32 - loss: 0.4330 - categorical_accuracy: 0.8636
12096/60000 [=====>........................] - ETA: 1:32 - loss: 0.4322 - categorical_accuracy: 0.8638
12128/60000 [=====>........................] - ETA: 1:32 - loss: 0.4322 - categorical_accuracy: 0.8638
12160/60000 [=====>........................] - ETA: 1:32 - loss: 0.4313 - categorical_accuracy: 0.8641
12192/60000 [=====>........................] - ETA: 1:32 - loss: 0.4315 - categorical_accuracy: 0.8641
12224/60000 [=====>........................] - ETA: 1:32 - loss: 0.4310 - categorical_accuracy: 0.8643
12256/60000 [=====>........................] - ETA: 1:32 - loss: 0.4310 - categorical_accuracy: 0.8644
12288/60000 [=====>........................] - ETA: 1:32 - loss: 0.4301 - categorical_accuracy: 0.8647
12320/60000 [=====>........................] - ETA: 1:31 - loss: 0.4299 - categorical_accuracy: 0.8648
12352/60000 [=====>........................] - ETA: 1:31 - loss: 0.4295 - categorical_accuracy: 0.8650
12384/60000 [=====>........................] - ETA: 1:31 - loss: 0.4288 - categorical_accuracy: 0.8651
12416/60000 [=====>........................] - ETA: 1:31 - loss: 0.4291 - categorical_accuracy: 0.8651
12448/60000 [=====>........................] - ETA: 1:31 - loss: 0.4284 - categorical_accuracy: 0.8654
12480/60000 [=====>........................] - ETA: 1:31 - loss: 0.4279 - categorical_accuracy: 0.8655
12512/60000 [=====>........................] - ETA: 1:31 - loss: 0.4271 - categorical_accuracy: 0.8659
12544/60000 [=====>........................] - ETA: 1:31 - loss: 0.4263 - categorical_accuracy: 0.8662
12576/60000 [=====>........................] - ETA: 1:31 - loss: 0.4260 - categorical_accuracy: 0.8663
12608/60000 [=====>........................] - ETA: 1:31 - loss: 0.4250 - categorical_accuracy: 0.8666
12640/60000 [=====>........................] - ETA: 1:31 - loss: 0.4246 - categorical_accuracy: 0.8667
12672/60000 [=====>........................] - ETA: 1:31 - loss: 0.4238 - categorical_accuracy: 0.8670
12704/60000 [=====>........................] - ETA: 1:31 - loss: 0.4230 - categorical_accuracy: 0.8672
12736/60000 [=====>........................] - ETA: 1:31 - loss: 0.4221 - categorical_accuracy: 0.8675
12768/60000 [=====>........................] - ETA: 1:31 - loss: 0.4220 - categorical_accuracy: 0.8677
12800/60000 [=====>........................] - ETA: 1:30 - loss: 0.4212 - categorical_accuracy: 0.8680
12832/60000 [=====>........................] - ETA: 1:30 - loss: 0.4203 - categorical_accuracy: 0.8682
12864/60000 [=====>........................] - ETA: 1:30 - loss: 0.4197 - categorical_accuracy: 0.8684
12896/60000 [=====>........................] - ETA: 1:30 - loss: 0.4191 - categorical_accuracy: 0.8686
12928/60000 [=====>........................] - ETA: 1:30 - loss: 0.4186 - categorical_accuracy: 0.8687
12960/60000 [=====>........................] - ETA: 1:30 - loss: 0.4186 - categorical_accuracy: 0.8687
12992/60000 [=====>........................] - ETA: 1:30 - loss: 0.4182 - categorical_accuracy: 0.8688
13024/60000 [=====>........................] - ETA: 1:30 - loss: 0.4180 - categorical_accuracy: 0.8689
13056/60000 [=====>........................] - ETA: 1:30 - loss: 0.4176 - categorical_accuracy: 0.8689
13088/60000 [=====>........................] - ETA: 1:30 - loss: 0.4169 - categorical_accuracy: 0.8692
13120/60000 [=====>........................] - ETA: 1:30 - loss: 0.4162 - categorical_accuracy: 0.8694
13152/60000 [=====>........................] - ETA: 1:30 - loss: 0.4153 - categorical_accuracy: 0.8697
13184/60000 [=====>........................] - ETA: 1:30 - loss: 0.4153 - categorical_accuracy: 0.8698
13216/60000 [=====>........................] - ETA: 1:30 - loss: 0.4161 - categorical_accuracy: 0.8697
13248/60000 [=====>........................] - ETA: 1:30 - loss: 0.4153 - categorical_accuracy: 0.8700
13280/60000 [=====>........................] - ETA: 1:29 - loss: 0.4159 - categorical_accuracy: 0.8700
13312/60000 [=====>........................] - ETA: 1:29 - loss: 0.4153 - categorical_accuracy: 0.8701
13344/60000 [=====>........................] - ETA: 1:29 - loss: 0.4150 - categorical_accuracy: 0.8702
13376/60000 [=====>........................] - ETA: 1:29 - loss: 0.4145 - categorical_accuracy: 0.8703
13408/60000 [=====>........................] - ETA: 1:29 - loss: 0.4139 - categorical_accuracy: 0.8706
13440/60000 [=====>........................] - ETA: 1:29 - loss: 0.4133 - categorical_accuracy: 0.8707
13472/60000 [=====>........................] - ETA: 1:29 - loss: 0.4130 - categorical_accuracy: 0.8708
13504/60000 [=====>........................] - ETA: 1:29 - loss: 0.4123 - categorical_accuracy: 0.8711
13536/60000 [=====>........................] - ETA: 1:29 - loss: 0.4116 - categorical_accuracy: 0.8714
13568/60000 [=====>........................] - ETA: 1:29 - loss: 0.4110 - categorical_accuracy: 0.8715
13600/60000 [=====>........................] - ETA: 1:29 - loss: 0.4102 - categorical_accuracy: 0.8718
13632/60000 [=====>........................] - ETA: 1:29 - loss: 0.4096 - categorical_accuracy: 0.8719
13664/60000 [=====>........................] - ETA: 1:29 - loss: 0.4090 - categorical_accuracy: 0.8721
13696/60000 [=====>........................] - ETA: 1:29 - loss: 0.4085 - categorical_accuracy: 0.8722
13728/60000 [=====>........................] - ETA: 1:29 - loss: 0.4079 - categorical_accuracy: 0.8725
13760/60000 [=====>........................] - ETA: 1:29 - loss: 0.4072 - categorical_accuracy: 0.8726
13792/60000 [=====>........................] - ETA: 1:28 - loss: 0.4065 - categorical_accuracy: 0.8728
13824/60000 [=====>........................] - ETA: 1:28 - loss: 0.4060 - categorical_accuracy: 0.8728
13856/60000 [=====>........................] - ETA: 1:28 - loss: 0.4051 - categorical_accuracy: 0.8731
13888/60000 [=====>........................] - ETA: 1:28 - loss: 0.4043 - categorical_accuracy: 0.8734
13920/60000 [=====>........................] - ETA: 1:28 - loss: 0.4038 - categorical_accuracy: 0.8736
13952/60000 [=====>........................] - ETA: 1:28 - loss: 0.4029 - categorical_accuracy: 0.8739
13984/60000 [=====>........................] - ETA: 1:28 - loss: 0.4023 - categorical_accuracy: 0.8740
14016/60000 [======>.......................] - ETA: 1:28 - loss: 0.4021 - categorical_accuracy: 0.8741
14048/60000 [======>.......................] - ETA: 1:28 - loss: 0.4019 - categorical_accuracy: 0.8741
14080/60000 [======>.......................] - ETA: 1:28 - loss: 0.4034 - categorical_accuracy: 0.8741
14112/60000 [======>.......................] - ETA: 1:28 - loss: 0.4030 - categorical_accuracy: 0.8742
14144/60000 [======>.......................] - ETA: 1:28 - loss: 0.4028 - categorical_accuracy: 0.8743
14176/60000 [======>.......................] - ETA: 1:28 - loss: 0.4022 - categorical_accuracy: 0.8745
14208/60000 [======>.......................] - ETA: 1:28 - loss: 0.4017 - categorical_accuracy: 0.8746
14240/60000 [======>.......................] - ETA: 1:28 - loss: 0.4010 - categorical_accuracy: 0.8748
14272/60000 [======>.......................] - ETA: 1:28 - loss: 0.4006 - categorical_accuracy: 0.8749
14304/60000 [======>.......................] - ETA: 1:27 - loss: 0.3998 - categorical_accuracy: 0.8752
14336/60000 [======>.......................] - ETA: 1:27 - loss: 0.3992 - categorical_accuracy: 0.8754
14368/60000 [======>.......................] - ETA: 1:27 - loss: 0.3984 - categorical_accuracy: 0.8757
14400/60000 [======>.......................] - ETA: 1:27 - loss: 0.3976 - categorical_accuracy: 0.8760
14432/60000 [======>.......................] - ETA: 1:27 - loss: 0.3969 - categorical_accuracy: 0.8762
14464/60000 [======>.......................] - ETA: 1:27 - loss: 0.3963 - categorical_accuracy: 0.8765
14496/60000 [======>.......................] - ETA: 1:27 - loss: 0.3964 - categorical_accuracy: 0.8764
14528/60000 [======>.......................] - ETA: 1:27 - loss: 0.3957 - categorical_accuracy: 0.8767
14560/60000 [======>.......................] - ETA: 1:27 - loss: 0.3954 - categorical_accuracy: 0.8766
14592/60000 [======>.......................] - ETA: 1:27 - loss: 0.3949 - categorical_accuracy: 0.8768
14624/60000 [======>.......................] - ETA: 1:27 - loss: 0.3944 - categorical_accuracy: 0.8769
14656/60000 [======>.......................] - ETA: 1:27 - loss: 0.3940 - categorical_accuracy: 0.8770
14688/60000 [======>.......................] - ETA: 1:27 - loss: 0.3932 - categorical_accuracy: 0.8773
14720/60000 [======>.......................] - ETA: 1:27 - loss: 0.3930 - categorical_accuracy: 0.8774
14752/60000 [======>.......................] - ETA: 1:27 - loss: 0.3928 - categorical_accuracy: 0.8775
14784/60000 [======>.......................] - ETA: 1:27 - loss: 0.3923 - categorical_accuracy: 0.8776
14816/60000 [======>.......................] - ETA: 1:26 - loss: 0.3916 - categorical_accuracy: 0.8779
14848/60000 [======>.......................] - ETA: 1:26 - loss: 0.3916 - categorical_accuracy: 0.8780
14880/60000 [======>.......................] - ETA: 1:26 - loss: 0.3913 - categorical_accuracy: 0.8781
14912/60000 [======>.......................] - ETA: 1:26 - loss: 0.3910 - categorical_accuracy: 0.8781
14944/60000 [======>.......................] - ETA: 1:26 - loss: 0.3907 - categorical_accuracy: 0.8782
14976/60000 [======>.......................] - ETA: 1:26 - loss: 0.3902 - categorical_accuracy: 0.8783
15008/60000 [======>.......................] - ETA: 1:26 - loss: 0.3897 - categorical_accuracy: 0.8784
15040/60000 [======>.......................] - ETA: 1:26 - loss: 0.3893 - categorical_accuracy: 0.8785
15072/60000 [======>.......................] - ETA: 1:26 - loss: 0.3888 - categorical_accuracy: 0.8786
15104/60000 [======>.......................] - ETA: 1:26 - loss: 0.3883 - categorical_accuracy: 0.8788
15136/60000 [======>.......................] - ETA: 1:26 - loss: 0.3879 - categorical_accuracy: 0.8788
15168/60000 [======>.......................] - ETA: 1:26 - loss: 0.3876 - categorical_accuracy: 0.8789
15200/60000 [======>.......................] - ETA: 1:26 - loss: 0.3870 - categorical_accuracy: 0.8790
15232/60000 [======>.......................] - ETA: 1:26 - loss: 0.3869 - categorical_accuracy: 0.8790
15264/60000 [======>.......................] - ETA: 1:26 - loss: 0.3869 - categorical_accuracy: 0.8790
15296/60000 [======>.......................] - ETA: 1:26 - loss: 0.3864 - categorical_accuracy: 0.8791
15328/60000 [======>.......................] - ETA: 1:25 - loss: 0.3858 - categorical_accuracy: 0.8792
15360/60000 [======>.......................] - ETA: 1:25 - loss: 0.3853 - categorical_accuracy: 0.8794
15392/60000 [======>.......................] - ETA: 1:25 - loss: 0.3850 - categorical_accuracy: 0.8795
15424/60000 [======>.......................] - ETA: 1:25 - loss: 0.3845 - categorical_accuracy: 0.8797
15456/60000 [======>.......................] - ETA: 1:25 - loss: 0.3841 - categorical_accuracy: 0.8799
15488/60000 [======>.......................] - ETA: 1:25 - loss: 0.3835 - categorical_accuracy: 0.8800
15520/60000 [======>.......................] - ETA: 1:25 - loss: 0.3836 - categorical_accuracy: 0.8800
15552/60000 [======>.......................] - ETA: 1:25 - loss: 0.3832 - categorical_accuracy: 0.8800
15584/60000 [======>.......................] - ETA: 1:25 - loss: 0.3827 - categorical_accuracy: 0.8801
15616/60000 [======>.......................] - ETA: 1:25 - loss: 0.3821 - categorical_accuracy: 0.8803
15648/60000 [======>.......................] - ETA: 1:25 - loss: 0.3815 - categorical_accuracy: 0.8804
15680/60000 [======>.......................] - ETA: 1:25 - loss: 0.3808 - categorical_accuracy: 0.8807
15712/60000 [======>.......................] - ETA: 1:25 - loss: 0.3803 - categorical_accuracy: 0.8808
15744/60000 [======>.......................] - ETA: 1:25 - loss: 0.3804 - categorical_accuracy: 0.8808
15776/60000 [======>.......................] - ETA: 1:25 - loss: 0.3799 - categorical_accuracy: 0.8810
15808/60000 [======>.......................] - ETA: 1:24 - loss: 0.3793 - categorical_accuracy: 0.8812
15840/60000 [======>.......................] - ETA: 1:24 - loss: 0.3792 - categorical_accuracy: 0.8813
15872/60000 [======>.......................] - ETA: 1:24 - loss: 0.3789 - categorical_accuracy: 0.8814
15904/60000 [======>.......................] - ETA: 1:24 - loss: 0.3784 - categorical_accuracy: 0.8815
15936/60000 [======>.......................] - ETA: 1:24 - loss: 0.3779 - categorical_accuracy: 0.8817
15968/60000 [======>.......................] - ETA: 1:24 - loss: 0.3777 - categorical_accuracy: 0.8818
16000/60000 [=======>......................] - ETA: 1:24 - loss: 0.3776 - categorical_accuracy: 0.8819
16032/60000 [=======>......................] - ETA: 1:24 - loss: 0.3770 - categorical_accuracy: 0.8820
16064/60000 [=======>......................] - ETA: 1:24 - loss: 0.3765 - categorical_accuracy: 0.8822
16096/60000 [=======>......................] - ETA: 1:24 - loss: 0.3762 - categorical_accuracy: 0.8822
16128/60000 [=======>......................] - ETA: 1:24 - loss: 0.3757 - categorical_accuracy: 0.8824
16160/60000 [=======>......................] - ETA: 1:24 - loss: 0.3756 - categorical_accuracy: 0.8825
16192/60000 [=======>......................] - ETA: 1:24 - loss: 0.3751 - categorical_accuracy: 0.8827
16224/60000 [=======>......................] - ETA: 1:24 - loss: 0.3744 - categorical_accuracy: 0.8829
16256/60000 [=======>......................] - ETA: 1:24 - loss: 0.3740 - categorical_accuracy: 0.8829
16288/60000 [=======>......................] - ETA: 1:24 - loss: 0.3734 - categorical_accuracy: 0.8832
16320/60000 [=======>......................] - ETA: 1:23 - loss: 0.3729 - categorical_accuracy: 0.8833
16352/60000 [=======>......................] - ETA: 1:23 - loss: 0.3725 - categorical_accuracy: 0.8834
16384/60000 [=======>......................] - ETA: 1:23 - loss: 0.3720 - categorical_accuracy: 0.8835
16416/60000 [=======>......................] - ETA: 1:23 - loss: 0.3726 - categorical_accuracy: 0.8835
16448/60000 [=======>......................] - ETA: 1:23 - loss: 0.3720 - categorical_accuracy: 0.8837
16480/60000 [=======>......................] - ETA: 1:23 - loss: 0.3716 - categorical_accuracy: 0.8838
16512/60000 [=======>......................] - ETA: 1:23 - loss: 0.3716 - categorical_accuracy: 0.8839
16544/60000 [=======>......................] - ETA: 1:23 - loss: 0.3720 - categorical_accuracy: 0.8838
16576/60000 [=======>......................] - ETA: 1:23 - loss: 0.3714 - categorical_accuracy: 0.8839
16608/60000 [=======>......................] - ETA: 1:23 - loss: 0.3710 - categorical_accuracy: 0.8840
16640/60000 [=======>......................] - ETA: 1:23 - loss: 0.3706 - categorical_accuracy: 0.8841
16672/60000 [=======>......................] - ETA: 1:23 - loss: 0.3703 - categorical_accuracy: 0.8842
16704/60000 [=======>......................] - ETA: 1:23 - loss: 0.3700 - categorical_accuracy: 0.8843
16736/60000 [=======>......................] - ETA: 1:23 - loss: 0.3697 - categorical_accuracy: 0.8844
16768/60000 [=======>......................] - ETA: 1:23 - loss: 0.3691 - categorical_accuracy: 0.8847
16800/60000 [=======>......................] - ETA: 1:22 - loss: 0.3690 - categorical_accuracy: 0.8846
16832/60000 [=======>......................] - ETA: 1:22 - loss: 0.3697 - categorical_accuracy: 0.8845
16864/60000 [=======>......................] - ETA: 1:22 - loss: 0.3696 - categorical_accuracy: 0.8845
16896/60000 [=======>......................] - ETA: 1:22 - loss: 0.3693 - categorical_accuracy: 0.8847
16928/60000 [=======>......................] - ETA: 1:22 - loss: 0.3688 - categorical_accuracy: 0.8849
16960/60000 [=======>......................] - ETA: 1:22 - loss: 0.3681 - categorical_accuracy: 0.8851
16992/60000 [=======>......................] - ETA: 1:22 - loss: 0.3681 - categorical_accuracy: 0.8852
17024/60000 [=======>......................] - ETA: 1:22 - loss: 0.3678 - categorical_accuracy: 0.8852
17056/60000 [=======>......................] - ETA: 1:22 - loss: 0.3675 - categorical_accuracy: 0.8853
17088/60000 [=======>......................] - ETA: 1:22 - loss: 0.3674 - categorical_accuracy: 0.8853
17120/60000 [=======>......................] - ETA: 1:22 - loss: 0.3671 - categorical_accuracy: 0.8853
17152/60000 [=======>......................] - ETA: 1:22 - loss: 0.3665 - categorical_accuracy: 0.8855
17184/60000 [=======>......................] - ETA: 1:22 - loss: 0.3662 - categorical_accuracy: 0.8856
17216/60000 [=======>......................] - ETA: 1:22 - loss: 0.3661 - categorical_accuracy: 0.8857
17248/60000 [=======>......................] - ETA: 1:22 - loss: 0.3655 - categorical_accuracy: 0.8859
17280/60000 [=======>......................] - ETA: 1:21 - loss: 0.3652 - categorical_accuracy: 0.8860
17312/60000 [=======>......................] - ETA: 1:21 - loss: 0.3651 - categorical_accuracy: 0.8860
17344/60000 [=======>......................] - ETA: 1:21 - loss: 0.3647 - categorical_accuracy: 0.8862
17376/60000 [=======>......................] - ETA: 1:21 - loss: 0.3643 - categorical_accuracy: 0.8863
17408/60000 [=======>......................] - ETA: 1:21 - loss: 0.3641 - categorical_accuracy: 0.8864
17440/60000 [=======>......................] - ETA: 1:21 - loss: 0.3637 - categorical_accuracy: 0.8865
17472/60000 [=======>......................] - ETA: 1:21 - loss: 0.3635 - categorical_accuracy: 0.8865
17504/60000 [=======>......................] - ETA: 1:21 - loss: 0.3634 - categorical_accuracy: 0.8865
17536/60000 [=======>......................] - ETA: 1:21 - loss: 0.3630 - categorical_accuracy: 0.8866
17568/60000 [=======>......................] - ETA: 1:21 - loss: 0.3627 - categorical_accuracy: 0.8867
17600/60000 [=======>......................] - ETA: 1:21 - loss: 0.3621 - categorical_accuracy: 0.8869
17632/60000 [=======>......................] - ETA: 1:21 - loss: 0.3619 - categorical_accuracy: 0.8870
17664/60000 [=======>......................] - ETA: 1:21 - loss: 0.3614 - categorical_accuracy: 0.8871
17696/60000 [=======>......................] - ETA: 1:21 - loss: 0.3609 - categorical_accuracy: 0.8873
17728/60000 [=======>......................] - ETA: 1:21 - loss: 0.3607 - categorical_accuracy: 0.8874
17760/60000 [=======>......................] - ETA: 1:21 - loss: 0.3602 - categorical_accuracy: 0.8876
17792/60000 [=======>......................] - ETA: 1:20 - loss: 0.3597 - categorical_accuracy: 0.8876
17824/60000 [=======>......................] - ETA: 1:20 - loss: 0.3594 - categorical_accuracy: 0.8877
17856/60000 [=======>......................] - ETA: 1:20 - loss: 0.3590 - categorical_accuracy: 0.8879
17888/60000 [=======>......................] - ETA: 1:20 - loss: 0.3587 - categorical_accuracy: 0.8880
17920/60000 [=======>......................] - ETA: 1:20 - loss: 0.3582 - categorical_accuracy: 0.8881
17952/60000 [=======>......................] - ETA: 1:20 - loss: 0.3580 - categorical_accuracy: 0.8881
17984/60000 [=======>......................] - ETA: 1:20 - loss: 0.3582 - categorical_accuracy: 0.8882
18016/60000 [========>.....................] - ETA: 1:20 - loss: 0.3580 - categorical_accuracy: 0.8883
18048/60000 [========>.....................] - ETA: 1:20 - loss: 0.3575 - categorical_accuracy: 0.8885
18080/60000 [========>.....................] - ETA: 1:20 - loss: 0.3573 - categorical_accuracy: 0.8885
18112/60000 [========>.....................] - ETA: 1:20 - loss: 0.3569 - categorical_accuracy: 0.8886
18144/60000 [========>.....................] - ETA: 1:20 - loss: 0.3564 - categorical_accuracy: 0.8888
18176/60000 [========>.....................] - ETA: 1:20 - loss: 0.3565 - categorical_accuracy: 0.8889
18208/60000 [========>.....................] - ETA: 1:20 - loss: 0.3563 - categorical_accuracy: 0.8890
18240/60000 [========>.....................] - ETA: 1:20 - loss: 0.3561 - categorical_accuracy: 0.8891
18272/60000 [========>.....................] - ETA: 1:19 - loss: 0.3555 - categorical_accuracy: 0.8892
18304/60000 [========>.....................] - ETA: 1:19 - loss: 0.3550 - categorical_accuracy: 0.8894
18336/60000 [========>.....................] - ETA: 1:19 - loss: 0.3547 - categorical_accuracy: 0.8895
18368/60000 [========>.....................] - ETA: 1:19 - loss: 0.3543 - categorical_accuracy: 0.8896
18400/60000 [========>.....................] - ETA: 1:19 - loss: 0.3539 - categorical_accuracy: 0.8897
18432/60000 [========>.....................] - ETA: 1:19 - loss: 0.3535 - categorical_accuracy: 0.8898
18464/60000 [========>.....................] - ETA: 1:19 - loss: 0.3531 - categorical_accuracy: 0.8899
18496/60000 [========>.....................] - ETA: 1:19 - loss: 0.3528 - categorical_accuracy: 0.8900
18528/60000 [========>.....................] - ETA: 1:19 - loss: 0.3525 - categorical_accuracy: 0.8901
18560/60000 [========>.....................] - ETA: 1:19 - loss: 0.3521 - categorical_accuracy: 0.8902
18592/60000 [========>.....................] - ETA: 1:19 - loss: 0.3519 - categorical_accuracy: 0.8903
18624/60000 [========>.....................] - ETA: 1:19 - loss: 0.3515 - categorical_accuracy: 0.8904
18656/60000 [========>.....................] - ETA: 1:19 - loss: 0.3511 - categorical_accuracy: 0.8905
18688/60000 [========>.....................] - ETA: 1:19 - loss: 0.3505 - categorical_accuracy: 0.8907
18720/60000 [========>.....................] - ETA: 1:19 - loss: 0.3506 - categorical_accuracy: 0.8909
18752/60000 [========>.....................] - ETA: 1:19 - loss: 0.3504 - categorical_accuracy: 0.8909
18784/60000 [========>.....................] - ETA: 1:18 - loss: 0.3499 - categorical_accuracy: 0.8911
18816/60000 [========>.....................] - ETA: 1:18 - loss: 0.3498 - categorical_accuracy: 0.8912
18848/60000 [========>.....................] - ETA: 1:18 - loss: 0.3494 - categorical_accuracy: 0.8913
18880/60000 [========>.....................] - ETA: 1:18 - loss: 0.3493 - categorical_accuracy: 0.8913
18912/60000 [========>.....................] - ETA: 1:18 - loss: 0.3491 - categorical_accuracy: 0.8914
18944/60000 [========>.....................] - ETA: 1:18 - loss: 0.3490 - categorical_accuracy: 0.8914
18976/60000 [========>.....................] - ETA: 1:18 - loss: 0.3491 - categorical_accuracy: 0.8913
19008/60000 [========>.....................] - ETA: 1:18 - loss: 0.3488 - categorical_accuracy: 0.8914
19040/60000 [========>.....................] - ETA: 1:18 - loss: 0.3483 - categorical_accuracy: 0.8916
19072/60000 [========>.....................] - ETA: 1:18 - loss: 0.3479 - categorical_accuracy: 0.8917
19104/60000 [========>.....................] - ETA: 1:18 - loss: 0.3475 - categorical_accuracy: 0.8919
19136/60000 [========>.....................] - ETA: 1:18 - loss: 0.3473 - categorical_accuracy: 0.8919
19168/60000 [========>.....................] - ETA: 1:18 - loss: 0.3468 - categorical_accuracy: 0.8920
19200/60000 [========>.....................] - ETA: 1:18 - loss: 0.3464 - categorical_accuracy: 0.8922
19232/60000 [========>.....................] - ETA: 1:18 - loss: 0.3461 - categorical_accuracy: 0.8923
19264/60000 [========>.....................] - ETA: 1:18 - loss: 0.3457 - categorical_accuracy: 0.8924
19296/60000 [========>.....................] - ETA: 1:17 - loss: 0.3452 - categorical_accuracy: 0.8925
19328/60000 [========>.....................] - ETA: 1:17 - loss: 0.3451 - categorical_accuracy: 0.8926
19360/60000 [========>.....................] - ETA: 1:17 - loss: 0.3447 - categorical_accuracy: 0.8927
19392/60000 [========>.....................] - ETA: 1:17 - loss: 0.3442 - categorical_accuracy: 0.8928
19424/60000 [========>.....................] - ETA: 1:17 - loss: 0.3436 - categorical_accuracy: 0.8930
19456/60000 [========>.....................] - ETA: 1:17 - loss: 0.3431 - categorical_accuracy: 0.8932
19488/60000 [========>.....................] - ETA: 1:17 - loss: 0.3428 - categorical_accuracy: 0.8933
19520/60000 [========>.....................] - ETA: 1:17 - loss: 0.3424 - categorical_accuracy: 0.8934
19552/60000 [========>.....................] - ETA: 1:17 - loss: 0.3420 - categorical_accuracy: 0.8936
19584/60000 [========>.....................] - ETA: 1:17 - loss: 0.3417 - categorical_accuracy: 0.8936
19616/60000 [========>.....................] - ETA: 1:17 - loss: 0.3417 - categorical_accuracy: 0.8937
19648/60000 [========>.....................] - ETA: 1:17 - loss: 0.3413 - categorical_accuracy: 0.8937
19680/60000 [========>.....................] - ETA: 1:17 - loss: 0.3410 - categorical_accuracy: 0.8939
19712/60000 [========>.....................] - ETA: 1:17 - loss: 0.3406 - categorical_accuracy: 0.8940
19744/60000 [========>.....................] - ETA: 1:17 - loss: 0.3402 - categorical_accuracy: 0.8940
19776/60000 [========>.....................] - ETA: 1:17 - loss: 0.3401 - categorical_accuracy: 0.8940
19808/60000 [========>.....................] - ETA: 1:17 - loss: 0.3397 - categorical_accuracy: 0.8941
19840/60000 [========>.....................] - ETA: 1:16 - loss: 0.3393 - categorical_accuracy: 0.8943
19872/60000 [========>.....................] - ETA: 1:16 - loss: 0.3390 - categorical_accuracy: 0.8943
19904/60000 [========>.....................] - ETA: 1:16 - loss: 0.3385 - categorical_accuracy: 0.8944
19936/60000 [========>.....................] - ETA: 1:16 - loss: 0.3386 - categorical_accuracy: 0.8945
19968/60000 [========>.....................] - ETA: 1:16 - loss: 0.3389 - categorical_accuracy: 0.8944
20000/60000 [=========>....................] - ETA: 1:16 - loss: 0.3385 - categorical_accuracy: 0.8946
20032/60000 [=========>....................] - ETA: 1:16 - loss: 0.3382 - categorical_accuracy: 0.8946
20064/60000 [=========>....................] - ETA: 1:16 - loss: 0.3378 - categorical_accuracy: 0.8948
20096/60000 [=========>....................] - ETA: 1:16 - loss: 0.3376 - categorical_accuracy: 0.8949
20128/60000 [=========>....................] - ETA: 1:16 - loss: 0.3372 - categorical_accuracy: 0.8951
20160/60000 [=========>....................] - ETA: 1:16 - loss: 0.3372 - categorical_accuracy: 0.8950
20192/60000 [=========>....................] - ETA: 1:16 - loss: 0.3368 - categorical_accuracy: 0.8952
20224/60000 [=========>....................] - ETA: 1:16 - loss: 0.3363 - categorical_accuracy: 0.8954
20256/60000 [=========>....................] - ETA: 1:16 - loss: 0.3359 - categorical_accuracy: 0.8955
20288/60000 [=========>....................] - ETA: 1:16 - loss: 0.3354 - categorical_accuracy: 0.8957
20320/60000 [=========>....................] - ETA: 1:15 - loss: 0.3351 - categorical_accuracy: 0.8957
20352/60000 [=========>....................] - ETA: 1:15 - loss: 0.3347 - categorical_accuracy: 0.8959
20384/60000 [=========>....................] - ETA: 1:15 - loss: 0.3342 - categorical_accuracy: 0.8960
20416/60000 [=========>....................] - ETA: 1:15 - loss: 0.3338 - categorical_accuracy: 0.8962
20448/60000 [=========>....................] - ETA: 1:15 - loss: 0.3334 - categorical_accuracy: 0.8963
20480/60000 [=========>....................] - ETA: 1:15 - loss: 0.3334 - categorical_accuracy: 0.8964
20512/60000 [=========>....................] - ETA: 1:15 - loss: 0.3330 - categorical_accuracy: 0.8965
20544/60000 [=========>....................] - ETA: 1:15 - loss: 0.3325 - categorical_accuracy: 0.8967
20576/60000 [=========>....................] - ETA: 1:15 - loss: 0.3322 - categorical_accuracy: 0.8967
20608/60000 [=========>....................] - ETA: 1:15 - loss: 0.3319 - categorical_accuracy: 0.8968
20640/60000 [=========>....................] - ETA: 1:15 - loss: 0.3317 - categorical_accuracy: 0.8968
20672/60000 [=========>....................] - ETA: 1:15 - loss: 0.3312 - categorical_accuracy: 0.8969
20704/60000 [=========>....................] - ETA: 1:15 - loss: 0.3311 - categorical_accuracy: 0.8969
20736/60000 [=========>....................] - ETA: 1:15 - loss: 0.3308 - categorical_accuracy: 0.8969
20768/60000 [=========>....................] - ETA: 1:15 - loss: 0.3307 - categorical_accuracy: 0.8970
20800/60000 [=========>....................] - ETA: 1:15 - loss: 0.3308 - categorical_accuracy: 0.8970
20832/60000 [=========>....................] - ETA: 1:14 - loss: 0.3304 - categorical_accuracy: 0.8971
20864/60000 [=========>....................] - ETA: 1:14 - loss: 0.3299 - categorical_accuracy: 0.8973
20896/60000 [=========>....................] - ETA: 1:14 - loss: 0.3296 - categorical_accuracy: 0.8974
20928/60000 [=========>....................] - ETA: 1:14 - loss: 0.3292 - categorical_accuracy: 0.8976
20960/60000 [=========>....................] - ETA: 1:14 - loss: 0.3289 - categorical_accuracy: 0.8976
20992/60000 [=========>....................] - ETA: 1:14 - loss: 0.3285 - categorical_accuracy: 0.8977
21024/60000 [=========>....................] - ETA: 1:14 - loss: 0.3282 - categorical_accuracy: 0.8978
21056/60000 [=========>....................] - ETA: 1:14 - loss: 0.3277 - categorical_accuracy: 0.8979
21088/60000 [=========>....................] - ETA: 1:14 - loss: 0.3273 - categorical_accuracy: 0.8981
21120/60000 [=========>....................] - ETA: 1:14 - loss: 0.3271 - categorical_accuracy: 0.8982
21152/60000 [=========>....................] - ETA: 1:14 - loss: 0.3266 - categorical_accuracy: 0.8983
21184/60000 [=========>....................] - ETA: 1:14 - loss: 0.3265 - categorical_accuracy: 0.8984
21216/60000 [=========>....................] - ETA: 1:14 - loss: 0.3266 - categorical_accuracy: 0.8983
21248/60000 [=========>....................] - ETA: 1:14 - loss: 0.3262 - categorical_accuracy: 0.8984
21280/60000 [=========>....................] - ETA: 1:14 - loss: 0.3259 - categorical_accuracy: 0.8985
21312/60000 [=========>....................] - ETA: 1:14 - loss: 0.3257 - categorical_accuracy: 0.8986
21344/60000 [=========>....................] - ETA: 1:13 - loss: 0.3254 - categorical_accuracy: 0.8987
21376/60000 [=========>....................] - ETA: 1:13 - loss: 0.3252 - categorical_accuracy: 0.8988
21408/60000 [=========>....................] - ETA: 1:13 - loss: 0.3249 - categorical_accuracy: 0.8989
21440/60000 [=========>....................] - ETA: 1:13 - loss: 0.3246 - categorical_accuracy: 0.8989
21472/60000 [=========>....................] - ETA: 1:13 - loss: 0.3244 - categorical_accuracy: 0.8990
21504/60000 [=========>....................] - ETA: 1:13 - loss: 0.3246 - categorical_accuracy: 0.8989
21536/60000 [=========>....................] - ETA: 1:13 - loss: 0.3242 - categorical_accuracy: 0.8990
21568/60000 [=========>....................] - ETA: 1:13 - loss: 0.3238 - categorical_accuracy: 0.8992
21600/60000 [=========>....................] - ETA: 1:13 - loss: 0.3237 - categorical_accuracy: 0.8992
21632/60000 [=========>....................] - ETA: 1:13 - loss: 0.3233 - categorical_accuracy: 0.8993
21664/60000 [=========>....................] - ETA: 1:13 - loss: 0.3231 - categorical_accuracy: 0.8993
21696/60000 [=========>....................] - ETA: 1:13 - loss: 0.3227 - categorical_accuracy: 0.8994
21728/60000 [=========>....................] - ETA: 1:13 - loss: 0.3224 - categorical_accuracy: 0.8995
21760/60000 [=========>....................] - ETA: 1:13 - loss: 0.3223 - categorical_accuracy: 0.8996
21792/60000 [=========>....................] - ETA: 1:13 - loss: 0.3219 - categorical_accuracy: 0.8996
21824/60000 [=========>....................] - ETA: 1:13 - loss: 0.3216 - categorical_accuracy: 0.8997
21856/60000 [=========>....................] - ETA: 1:12 - loss: 0.3212 - categorical_accuracy: 0.8998
21888/60000 [=========>....................] - ETA: 1:12 - loss: 0.3208 - categorical_accuracy: 0.9000
21920/60000 [=========>....................] - ETA: 1:12 - loss: 0.3207 - categorical_accuracy: 0.9000
21952/60000 [=========>....................] - ETA: 1:12 - loss: 0.3204 - categorical_accuracy: 0.9001
21984/60000 [=========>....................] - ETA: 1:12 - loss: 0.3201 - categorical_accuracy: 0.9002
22016/60000 [==========>...................] - ETA: 1:12 - loss: 0.3197 - categorical_accuracy: 0.9004
22048/60000 [==========>...................] - ETA: 1:12 - loss: 0.3199 - categorical_accuracy: 0.9004
22080/60000 [==========>...................] - ETA: 1:12 - loss: 0.3197 - categorical_accuracy: 0.9005
22112/60000 [==========>...................] - ETA: 1:12 - loss: 0.3198 - categorical_accuracy: 0.9005
22144/60000 [==========>...................] - ETA: 1:12 - loss: 0.3194 - categorical_accuracy: 0.9006
22176/60000 [==========>...................] - ETA: 1:12 - loss: 0.3190 - categorical_accuracy: 0.9007
22208/60000 [==========>...................] - ETA: 1:12 - loss: 0.3191 - categorical_accuracy: 0.9006
22240/60000 [==========>...................] - ETA: 1:12 - loss: 0.3190 - categorical_accuracy: 0.9007
22272/60000 [==========>...................] - ETA: 1:12 - loss: 0.3186 - categorical_accuracy: 0.9008
22304/60000 [==========>...................] - ETA: 1:12 - loss: 0.3183 - categorical_accuracy: 0.9009
22336/60000 [==========>...................] - ETA: 1:12 - loss: 0.3179 - categorical_accuracy: 0.9011
22368/60000 [==========>...................] - ETA: 1:12 - loss: 0.3176 - categorical_accuracy: 0.9012
22400/60000 [==========>...................] - ETA: 1:11 - loss: 0.3173 - categorical_accuracy: 0.9012
22432/60000 [==========>...................] - ETA: 1:11 - loss: 0.3169 - categorical_accuracy: 0.9013
22464/60000 [==========>...................] - ETA: 1:11 - loss: 0.3168 - categorical_accuracy: 0.9013
22496/60000 [==========>...................] - ETA: 1:11 - loss: 0.3165 - categorical_accuracy: 0.9014
22528/60000 [==========>...................] - ETA: 1:11 - loss: 0.3162 - categorical_accuracy: 0.9015
22560/60000 [==========>...................] - ETA: 1:11 - loss: 0.3158 - categorical_accuracy: 0.9016
22592/60000 [==========>...................] - ETA: 1:11 - loss: 0.3155 - categorical_accuracy: 0.9018
22624/60000 [==========>...................] - ETA: 1:11 - loss: 0.3151 - categorical_accuracy: 0.9019
22656/60000 [==========>...................] - ETA: 1:11 - loss: 0.3147 - categorical_accuracy: 0.9020
22688/60000 [==========>...................] - ETA: 1:11 - loss: 0.3145 - categorical_accuracy: 0.9021
22720/60000 [==========>...................] - ETA: 1:11 - loss: 0.3144 - categorical_accuracy: 0.9022
22752/60000 [==========>...................] - ETA: 1:11 - loss: 0.3143 - categorical_accuracy: 0.9022
22784/60000 [==========>...................] - ETA: 1:11 - loss: 0.3140 - categorical_accuracy: 0.9023
22816/60000 [==========>...................] - ETA: 1:11 - loss: 0.3139 - categorical_accuracy: 0.9023
22848/60000 [==========>...................] - ETA: 1:11 - loss: 0.3135 - categorical_accuracy: 0.9024
22880/60000 [==========>...................] - ETA: 1:11 - loss: 0.3132 - categorical_accuracy: 0.9025
22912/60000 [==========>...................] - ETA: 1:10 - loss: 0.3131 - categorical_accuracy: 0.9025
22944/60000 [==========>...................] - ETA: 1:10 - loss: 0.3130 - categorical_accuracy: 0.9025
22976/60000 [==========>...................] - ETA: 1:10 - loss: 0.3126 - categorical_accuracy: 0.9027
23008/60000 [==========>...................] - ETA: 1:10 - loss: 0.3124 - categorical_accuracy: 0.9028
23040/60000 [==========>...................] - ETA: 1:10 - loss: 0.3123 - categorical_accuracy: 0.9028
23072/60000 [==========>...................] - ETA: 1:10 - loss: 0.3119 - categorical_accuracy: 0.9030
23104/60000 [==========>...................] - ETA: 1:10 - loss: 0.3116 - categorical_accuracy: 0.9030
23136/60000 [==========>...................] - ETA: 1:10 - loss: 0.3112 - categorical_accuracy: 0.9031
23168/60000 [==========>...................] - ETA: 1:10 - loss: 0.3109 - categorical_accuracy: 0.9032
23200/60000 [==========>...................] - ETA: 1:10 - loss: 0.3109 - categorical_accuracy: 0.9032
23232/60000 [==========>...................] - ETA: 1:10 - loss: 0.3106 - categorical_accuracy: 0.9033
23264/60000 [==========>...................] - ETA: 1:10 - loss: 0.3104 - categorical_accuracy: 0.9034
23296/60000 [==========>...................] - ETA: 1:10 - loss: 0.3100 - categorical_accuracy: 0.9035
23328/60000 [==========>...................] - ETA: 1:10 - loss: 0.3096 - categorical_accuracy: 0.9036
23360/60000 [==========>...................] - ETA: 1:10 - loss: 0.3095 - categorical_accuracy: 0.9036
23392/60000 [==========>...................] - ETA: 1:10 - loss: 0.3093 - categorical_accuracy: 0.9037
23424/60000 [==========>...................] - ETA: 1:09 - loss: 0.3091 - categorical_accuracy: 0.9037
23456/60000 [==========>...................] - ETA: 1:09 - loss: 0.3092 - categorical_accuracy: 0.9036
23488/60000 [==========>...................] - ETA: 1:09 - loss: 0.3090 - categorical_accuracy: 0.9037
23520/60000 [==========>...................] - ETA: 1:09 - loss: 0.3087 - categorical_accuracy: 0.9038
23552/60000 [==========>...................] - ETA: 1:09 - loss: 0.3085 - categorical_accuracy: 0.9039
23584/60000 [==========>...................] - ETA: 1:09 - loss: 0.3083 - categorical_accuracy: 0.9039
23616/60000 [==========>...................] - ETA: 1:09 - loss: 0.3080 - categorical_accuracy: 0.9040
23648/60000 [==========>...................] - ETA: 1:09 - loss: 0.3076 - categorical_accuracy: 0.9041
23680/60000 [==========>...................] - ETA: 1:09 - loss: 0.3076 - categorical_accuracy: 0.9042
23712/60000 [==========>...................] - ETA: 1:09 - loss: 0.3074 - categorical_accuracy: 0.9042
23744/60000 [==========>...................] - ETA: 1:09 - loss: 0.3073 - categorical_accuracy: 0.9042
23776/60000 [==========>...................] - ETA: 1:09 - loss: 0.3069 - categorical_accuracy: 0.9044
23808/60000 [==========>...................] - ETA: 1:09 - loss: 0.3067 - categorical_accuracy: 0.9044
23840/60000 [==========>...................] - ETA: 1:09 - loss: 0.3064 - categorical_accuracy: 0.9045
23872/60000 [==========>...................] - ETA: 1:09 - loss: 0.3060 - categorical_accuracy: 0.9046
23904/60000 [==========>...................] - ETA: 1:09 - loss: 0.3060 - categorical_accuracy: 0.9046
23936/60000 [==========>...................] - ETA: 1:08 - loss: 0.3056 - categorical_accuracy: 0.9047
23968/60000 [==========>...................] - ETA: 1:08 - loss: 0.3055 - categorical_accuracy: 0.9047
24000/60000 [===========>..................] - ETA: 1:08 - loss: 0.3052 - categorical_accuracy: 0.9048
24032/60000 [===========>..................] - ETA: 1:08 - loss: 0.3049 - categorical_accuracy: 0.9050
24064/60000 [===========>..................] - ETA: 1:08 - loss: 0.3046 - categorical_accuracy: 0.9050
24096/60000 [===========>..................] - ETA: 1:08 - loss: 0.3052 - categorical_accuracy: 0.9049
24128/60000 [===========>..................] - ETA: 1:08 - loss: 0.3050 - categorical_accuracy: 0.9049
24160/60000 [===========>..................] - ETA: 1:08 - loss: 0.3047 - categorical_accuracy: 0.9050
24192/60000 [===========>..................] - ETA: 1:08 - loss: 0.3044 - categorical_accuracy: 0.9051
24224/60000 [===========>..................] - ETA: 1:08 - loss: 0.3040 - categorical_accuracy: 0.9052
24256/60000 [===========>..................] - ETA: 1:08 - loss: 0.3037 - categorical_accuracy: 0.9053
24288/60000 [===========>..................] - ETA: 1:08 - loss: 0.3033 - categorical_accuracy: 0.9055
24320/60000 [===========>..................] - ETA: 1:08 - loss: 0.3031 - categorical_accuracy: 0.9056
24352/60000 [===========>..................] - ETA: 1:08 - loss: 0.3028 - categorical_accuracy: 0.9057
24384/60000 [===========>..................] - ETA: 1:08 - loss: 0.3025 - categorical_accuracy: 0.9057
24416/60000 [===========>..................] - ETA: 1:08 - loss: 0.3024 - categorical_accuracy: 0.9058
24448/60000 [===========>..................] - ETA: 1:07 - loss: 0.3022 - categorical_accuracy: 0.9058
24480/60000 [===========>..................] - ETA: 1:07 - loss: 0.3021 - categorical_accuracy: 0.9058
24512/60000 [===========>..................] - ETA: 1:07 - loss: 0.3018 - categorical_accuracy: 0.9059
24544/60000 [===========>..................] - ETA: 1:07 - loss: 0.3018 - categorical_accuracy: 0.9059
24576/60000 [===========>..................] - ETA: 1:07 - loss: 0.3017 - categorical_accuracy: 0.9059
24608/60000 [===========>..................] - ETA: 1:07 - loss: 0.3014 - categorical_accuracy: 0.9060
24640/60000 [===========>..................] - ETA: 1:07 - loss: 0.3011 - categorical_accuracy: 0.9061
24672/60000 [===========>..................] - ETA: 1:07 - loss: 0.3008 - categorical_accuracy: 0.9062
24704/60000 [===========>..................] - ETA: 1:07 - loss: 0.3011 - categorical_accuracy: 0.9062
24736/60000 [===========>..................] - ETA: 1:07 - loss: 0.3008 - categorical_accuracy: 0.9063
24768/60000 [===========>..................] - ETA: 1:07 - loss: 0.3004 - categorical_accuracy: 0.9064
24800/60000 [===========>..................] - ETA: 1:07 - loss: 0.3002 - categorical_accuracy: 0.9065
24832/60000 [===========>..................] - ETA: 1:07 - loss: 0.2999 - categorical_accuracy: 0.9065
24864/60000 [===========>..................] - ETA: 1:07 - loss: 0.2998 - categorical_accuracy: 0.9066
24896/60000 [===========>..................] - ETA: 1:07 - loss: 0.2997 - categorical_accuracy: 0.9065
24928/60000 [===========>..................] - ETA: 1:07 - loss: 0.2995 - categorical_accuracy: 0.9066
24960/60000 [===========>..................] - ETA: 1:06 - loss: 0.2992 - categorical_accuracy: 0.9067
24992/60000 [===========>..................] - ETA: 1:06 - loss: 0.2992 - categorical_accuracy: 0.9067
25024/60000 [===========>..................] - ETA: 1:06 - loss: 0.2989 - categorical_accuracy: 0.9068
25056/60000 [===========>..................] - ETA: 1:06 - loss: 0.2987 - categorical_accuracy: 0.9069
25088/60000 [===========>..................] - ETA: 1:06 - loss: 0.2986 - categorical_accuracy: 0.9069
25120/60000 [===========>..................] - ETA: 1:06 - loss: 0.2983 - categorical_accuracy: 0.9070
25152/60000 [===========>..................] - ETA: 1:06 - loss: 0.2981 - categorical_accuracy: 0.9070
25184/60000 [===========>..................] - ETA: 1:06 - loss: 0.2981 - categorical_accuracy: 0.9071
25216/60000 [===========>..................] - ETA: 1:06 - loss: 0.2979 - categorical_accuracy: 0.9072
25248/60000 [===========>..................] - ETA: 1:06 - loss: 0.2976 - categorical_accuracy: 0.9072
25280/60000 [===========>..................] - ETA: 1:06 - loss: 0.2979 - categorical_accuracy: 0.9072
25312/60000 [===========>..................] - ETA: 1:06 - loss: 0.2977 - categorical_accuracy: 0.9073
25344/60000 [===========>..................] - ETA: 1:06 - loss: 0.2974 - categorical_accuracy: 0.9074
25376/60000 [===========>..................] - ETA: 1:06 - loss: 0.2972 - categorical_accuracy: 0.9074
25408/60000 [===========>..................] - ETA: 1:06 - loss: 0.2970 - categorical_accuracy: 0.9075
25440/60000 [===========>..................] - ETA: 1:06 - loss: 0.2970 - categorical_accuracy: 0.9075
25472/60000 [===========>..................] - ETA: 1:05 - loss: 0.2969 - categorical_accuracy: 0.9075
25504/60000 [===========>..................] - ETA: 1:05 - loss: 0.2966 - categorical_accuracy: 0.9076
25536/60000 [===========>..................] - ETA: 1:05 - loss: 0.2968 - categorical_accuracy: 0.9075
25568/60000 [===========>..................] - ETA: 1:05 - loss: 0.2965 - categorical_accuracy: 0.9076
25600/60000 [===========>..................] - ETA: 1:05 - loss: 0.2963 - categorical_accuracy: 0.9077
25632/60000 [===========>..................] - ETA: 1:05 - loss: 0.2964 - categorical_accuracy: 0.9077
25664/60000 [===========>..................] - ETA: 1:05 - loss: 0.2962 - categorical_accuracy: 0.9078
25696/60000 [===========>..................] - ETA: 1:05 - loss: 0.2960 - categorical_accuracy: 0.9078
25728/60000 [===========>..................] - ETA: 1:05 - loss: 0.2957 - categorical_accuracy: 0.9079
25760/60000 [===========>..................] - ETA: 1:05 - loss: 0.2956 - categorical_accuracy: 0.9080
25792/60000 [===========>..................] - ETA: 1:05 - loss: 0.2953 - categorical_accuracy: 0.9081
25824/60000 [===========>..................] - ETA: 1:05 - loss: 0.2950 - categorical_accuracy: 0.9081
25856/60000 [===========>..................] - ETA: 1:05 - loss: 0.2947 - categorical_accuracy: 0.9083
25888/60000 [===========>..................] - ETA: 1:05 - loss: 0.2945 - categorical_accuracy: 0.9083
25920/60000 [===========>..................] - ETA: 1:05 - loss: 0.2947 - categorical_accuracy: 0.9083
25952/60000 [===========>..................] - ETA: 1:05 - loss: 0.2943 - categorical_accuracy: 0.9084
25984/60000 [===========>..................] - ETA: 1:04 - loss: 0.2940 - categorical_accuracy: 0.9086
26016/60000 [============>.................] - ETA: 1:04 - loss: 0.2939 - categorical_accuracy: 0.9086
26048/60000 [============>.................] - ETA: 1:04 - loss: 0.2937 - categorical_accuracy: 0.9087
26080/60000 [============>.................] - ETA: 1:04 - loss: 0.2934 - categorical_accuracy: 0.9088
26112/60000 [============>.................] - ETA: 1:04 - loss: 0.2932 - categorical_accuracy: 0.9089
26144/60000 [============>.................] - ETA: 1:04 - loss: 0.2932 - categorical_accuracy: 0.9089
26176/60000 [============>.................] - ETA: 1:04 - loss: 0.2931 - categorical_accuracy: 0.9089
26208/60000 [============>.................] - ETA: 1:04 - loss: 0.2929 - categorical_accuracy: 0.9089
26240/60000 [============>.................] - ETA: 1:04 - loss: 0.2927 - categorical_accuracy: 0.9090
26272/60000 [============>.................] - ETA: 1:04 - loss: 0.2924 - categorical_accuracy: 0.9091
26304/60000 [============>.................] - ETA: 1:04 - loss: 0.2921 - categorical_accuracy: 0.9092
26336/60000 [============>.................] - ETA: 1:04 - loss: 0.2919 - categorical_accuracy: 0.9092
26368/60000 [============>.................] - ETA: 1:04 - loss: 0.2917 - categorical_accuracy: 0.9093
26400/60000 [============>.................] - ETA: 1:04 - loss: 0.2916 - categorical_accuracy: 0.9093
26432/60000 [============>.................] - ETA: 1:04 - loss: 0.2913 - categorical_accuracy: 0.9094
26464/60000 [============>.................] - ETA: 1:04 - loss: 0.2912 - categorical_accuracy: 0.9094
26496/60000 [============>.................] - ETA: 1:03 - loss: 0.2910 - categorical_accuracy: 0.9095
26528/60000 [============>.................] - ETA: 1:03 - loss: 0.2907 - categorical_accuracy: 0.9095
26560/60000 [============>.................] - ETA: 1:03 - loss: 0.2905 - categorical_accuracy: 0.9096
26592/60000 [============>.................] - ETA: 1:03 - loss: 0.2901 - categorical_accuracy: 0.9097
26624/60000 [============>.................] - ETA: 1:03 - loss: 0.2899 - categorical_accuracy: 0.9098
26656/60000 [============>.................] - ETA: 1:03 - loss: 0.2896 - categorical_accuracy: 0.9099
26688/60000 [============>.................] - ETA: 1:03 - loss: 0.2893 - categorical_accuracy: 0.9100
26720/60000 [============>.................] - ETA: 1:03 - loss: 0.2890 - categorical_accuracy: 0.9101
26752/60000 [============>.................] - ETA: 1:03 - loss: 0.2888 - categorical_accuracy: 0.9101
26784/60000 [============>.................] - ETA: 1:03 - loss: 0.2887 - categorical_accuracy: 0.9102
26816/60000 [============>.................] - ETA: 1:03 - loss: 0.2884 - categorical_accuracy: 0.9102
26848/60000 [============>.................] - ETA: 1:03 - loss: 0.2881 - categorical_accuracy: 0.9103
26880/60000 [============>.................] - ETA: 1:03 - loss: 0.2880 - categorical_accuracy: 0.9103
26912/60000 [============>.................] - ETA: 1:03 - loss: 0.2877 - categorical_accuracy: 0.9104
26944/60000 [============>.................] - ETA: 1:03 - loss: 0.2879 - categorical_accuracy: 0.9104
26976/60000 [============>.................] - ETA: 1:02 - loss: 0.2879 - categorical_accuracy: 0.9104
27008/60000 [============>.................] - ETA: 1:02 - loss: 0.2877 - categorical_accuracy: 0.9105
27040/60000 [============>.................] - ETA: 1:02 - loss: 0.2875 - categorical_accuracy: 0.9105
27072/60000 [============>.................] - ETA: 1:02 - loss: 0.2874 - categorical_accuracy: 0.9105
27104/60000 [============>.................] - ETA: 1:02 - loss: 0.2873 - categorical_accuracy: 0.9105
27136/60000 [============>.................] - ETA: 1:02 - loss: 0.2870 - categorical_accuracy: 0.9106
27168/60000 [============>.................] - ETA: 1:02 - loss: 0.2866 - categorical_accuracy: 0.9107
27200/60000 [============>.................] - ETA: 1:02 - loss: 0.2864 - categorical_accuracy: 0.9107
27232/60000 [============>.................] - ETA: 1:02 - loss: 0.2862 - categorical_accuracy: 0.9108
27264/60000 [============>.................] - ETA: 1:02 - loss: 0.2859 - categorical_accuracy: 0.9109
27296/60000 [============>.................] - ETA: 1:02 - loss: 0.2857 - categorical_accuracy: 0.9109
27328/60000 [============>.................] - ETA: 1:02 - loss: 0.2854 - categorical_accuracy: 0.9110
27360/60000 [============>.................] - ETA: 1:02 - loss: 0.2852 - categorical_accuracy: 0.9111
27392/60000 [============>.................] - ETA: 1:02 - loss: 0.2849 - categorical_accuracy: 0.9112
27424/60000 [============>.................] - ETA: 1:02 - loss: 0.2846 - categorical_accuracy: 0.9113
27456/60000 [============>.................] - ETA: 1:02 - loss: 0.2843 - categorical_accuracy: 0.9113
27488/60000 [============>.................] - ETA: 1:02 - loss: 0.2841 - categorical_accuracy: 0.9114
27520/60000 [============>.................] - ETA: 1:01 - loss: 0.2840 - categorical_accuracy: 0.9114
27552/60000 [============>.................] - ETA: 1:01 - loss: 0.2838 - categorical_accuracy: 0.9114
27584/60000 [============>.................] - ETA: 1:01 - loss: 0.2835 - categorical_accuracy: 0.9115
27616/60000 [============>.................] - ETA: 1:01 - loss: 0.2833 - categorical_accuracy: 0.9116
27648/60000 [============>.................] - ETA: 1:01 - loss: 0.2830 - categorical_accuracy: 0.9117
27680/60000 [============>.................] - ETA: 1:01 - loss: 0.2827 - categorical_accuracy: 0.9118
27712/60000 [============>.................] - ETA: 1:01 - loss: 0.2826 - categorical_accuracy: 0.9118
27744/60000 [============>.................] - ETA: 1:01 - loss: 0.2823 - categorical_accuracy: 0.9119
27776/60000 [============>.................] - ETA: 1:01 - loss: 0.2822 - categorical_accuracy: 0.9120
27808/60000 [============>.................] - ETA: 1:01 - loss: 0.2819 - categorical_accuracy: 0.9121
27840/60000 [============>.................] - ETA: 1:01 - loss: 0.2817 - categorical_accuracy: 0.9121
27872/60000 [============>.................] - ETA: 1:01 - loss: 0.2815 - categorical_accuracy: 0.9122
27904/60000 [============>.................] - ETA: 1:01 - loss: 0.2812 - categorical_accuracy: 0.9123
27936/60000 [============>.................] - ETA: 1:01 - loss: 0.2809 - categorical_accuracy: 0.9124
27968/60000 [============>.................] - ETA: 1:01 - loss: 0.2807 - categorical_accuracy: 0.9125
28000/60000 [=============>................] - ETA: 1:01 - loss: 0.2804 - categorical_accuracy: 0.9126
28032/60000 [=============>................] - ETA: 1:00 - loss: 0.2801 - categorical_accuracy: 0.9127
28064/60000 [=============>................] - ETA: 1:00 - loss: 0.2798 - categorical_accuracy: 0.9128
28096/60000 [=============>................] - ETA: 1:00 - loss: 0.2798 - categorical_accuracy: 0.9128
28128/60000 [=============>................] - ETA: 1:00 - loss: 0.2795 - categorical_accuracy: 0.9129
28160/60000 [=============>................] - ETA: 1:00 - loss: 0.2793 - categorical_accuracy: 0.9130
28192/60000 [=============>................] - ETA: 1:00 - loss: 0.2791 - categorical_accuracy: 0.9130
28224/60000 [=============>................] - ETA: 1:00 - loss: 0.2791 - categorical_accuracy: 0.9131
28256/60000 [=============>................] - ETA: 1:00 - loss: 0.2788 - categorical_accuracy: 0.9132
28288/60000 [=============>................] - ETA: 1:00 - loss: 0.2788 - categorical_accuracy: 0.9131
28320/60000 [=============>................] - ETA: 1:00 - loss: 0.2786 - categorical_accuracy: 0.9132
28352/60000 [=============>................] - ETA: 1:00 - loss: 0.2785 - categorical_accuracy: 0.9132
28384/60000 [=============>................] - ETA: 1:00 - loss: 0.2782 - categorical_accuracy: 0.9133
28416/60000 [=============>................] - ETA: 1:00 - loss: 0.2781 - categorical_accuracy: 0.9132
28448/60000 [=============>................] - ETA: 1:00 - loss: 0.2781 - categorical_accuracy: 0.9132
28480/60000 [=============>................] - ETA: 1:00 - loss: 0.2780 - categorical_accuracy: 0.9132
28512/60000 [=============>................] - ETA: 1:00 - loss: 0.2779 - categorical_accuracy: 0.9133
28544/60000 [=============>................] - ETA: 59s - loss: 0.2778 - categorical_accuracy: 0.9133 
28576/60000 [=============>................] - ETA: 59s - loss: 0.2776 - categorical_accuracy: 0.9134
28608/60000 [=============>................] - ETA: 59s - loss: 0.2776 - categorical_accuracy: 0.9134
28640/60000 [=============>................] - ETA: 59s - loss: 0.2774 - categorical_accuracy: 0.9135
28672/60000 [=============>................] - ETA: 59s - loss: 0.2771 - categorical_accuracy: 0.9135
28704/60000 [=============>................] - ETA: 59s - loss: 0.2770 - categorical_accuracy: 0.9136
28736/60000 [=============>................] - ETA: 59s - loss: 0.2767 - categorical_accuracy: 0.9137
28768/60000 [=============>................] - ETA: 59s - loss: 0.2765 - categorical_accuracy: 0.9138
28800/60000 [=============>................] - ETA: 59s - loss: 0.2763 - categorical_accuracy: 0.9138
28832/60000 [=============>................] - ETA: 59s - loss: 0.2760 - categorical_accuracy: 0.9139
28864/60000 [=============>................] - ETA: 59s - loss: 0.2759 - categorical_accuracy: 0.9140
28896/60000 [=============>................] - ETA: 59s - loss: 0.2756 - categorical_accuracy: 0.9141
28928/60000 [=============>................] - ETA: 59s - loss: 0.2754 - categorical_accuracy: 0.9142
28960/60000 [=============>................] - ETA: 59s - loss: 0.2752 - categorical_accuracy: 0.9142
28992/60000 [=============>................] - ETA: 59s - loss: 0.2752 - categorical_accuracy: 0.9143
29024/60000 [=============>................] - ETA: 59s - loss: 0.2750 - categorical_accuracy: 0.9144
29056/60000 [=============>................] - ETA: 58s - loss: 0.2747 - categorical_accuracy: 0.9145
29088/60000 [=============>................] - ETA: 58s - loss: 0.2747 - categorical_accuracy: 0.9145
29120/60000 [=============>................] - ETA: 58s - loss: 0.2745 - categorical_accuracy: 0.9146
29152/60000 [=============>................] - ETA: 58s - loss: 0.2743 - categorical_accuracy: 0.9146
29184/60000 [=============>................] - ETA: 58s - loss: 0.2745 - categorical_accuracy: 0.9146
29216/60000 [=============>................] - ETA: 58s - loss: 0.2742 - categorical_accuracy: 0.9147
29248/60000 [=============>................] - ETA: 58s - loss: 0.2739 - categorical_accuracy: 0.9148
29280/60000 [=============>................] - ETA: 58s - loss: 0.2737 - categorical_accuracy: 0.9149
29312/60000 [=============>................] - ETA: 58s - loss: 0.2734 - categorical_accuracy: 0.9149
29344/60000 [=============>................] - ETA: 58s - loss: 0.2731 - categorical_accuracy: 0.9150
29376/60000 [=============>................] - ETA: 58s - loss: 0.2730 - categorical_accuracy: 0.9151
29408/60000 [=============>................] - ETA: 58s - loss: 0.2733 - categorical_accuracy: 0.9150
29440/60000 [=============>................] - ETA: 58s - loss: 0.2730 - categorical_accuracy: 0.9151
29472/60000 [=============>................] - ETA: 58s - loss: 0.2728 - categorical_accuracy: 0.9152
29504/60000 [=============>................] - ETA: 58s - loss: 0.2728 - categorical_accuracy: 0.9153
29536/60000 [=============>................] - ETA: 58s - loss: 0.2726 - categorical_accuracy: 0.9153
29568/60000 [=============>................] - ETA: 57s - loss: 0.2724 - categorical_accuracy: 0.9154
29600/60000 [=============>................] - ETA: 57s - loss: 0.2721 - categorical_accuracy: 0.9154
29632/60000 [=============>................] - ETA: 57s - loss: 0.2721 - categorical_accuracy: 0.9154
29664/60000 [=============>................] - ETA: 57s - loss: 0.2718 - categorical_accuracy: 0.9155
29696/60000 [=============>................] - ETA: 57s - loss: 0.2716 - categorical_accuracy: 0.9156
29728/60000 [=============>................] - ETA: 57s - loss: 0.2715 - categorical_accuracy: 0.9156
29760/60000 [=============>................] - ETA: 57s - loss: 0.2712 - categorical_accuracy: 0.9157
29792/60000 [=============>................] - ETA: 57s - loss: 0.2711 - categorical_accuracy: 0.9157
29824/60000 [=============>................] - ETA: 57s - loss: 0.2710 - categorical_accuracy: 0.9158
29856/60000 [=============>................] - ETA: 57s - loss: 0.2707 - categorical_accuracy: 0.9159
29888/60000 [=============>................] - ETA: 57s - loss: 0.2706 - categorical_accuracy: 0.9159
29920/60000 [=============>................] - ETA: 57s - loss: 0.2704 - categorical_accuracy: 0.9160
29952/60000 [=============>................] - ETA: 57s - loss: 0.2704 - categorical_accuracy: 0.9160
29984/60000 [=============>................] - ETA: 57s - loss: 0.2701 - categorical_accuracy: 0.9161
30016/60000 [==============>...............] - ETA: 57s - loss: 0.2701 - categorical_accuracy: 0.9161
30048/60000 [==============>...............] - ETA: 57s - loss: 0.2699 - categorical_accuracy: 0.9161
30080/60000 [==============>...............] - ETA: 56s - loss: 0.2700 - categorical_accuracy: 0.9162
30112/60000 [==============>...............] - ETA: 56s - loss: 0.2698 - categorical_accuracy: 0.9162
30144/60000 [==============>...............] - ETA: 56s - loss: 0.2695 - categorical_accuracy: 0.9163
30176/60000 [==============>...............] - ETA: 56s - loss: 0.2693 - categorical_accuracy: 0.9164
30208/60000 [==============>...............] - ETA: 56s - loss: 0.2691 - categorical_accuracy: 0.9164
30240/60000 [==============>...............] - ETA: 56s - loss: 0.2689 - categorical_accuracy: 0.9165
30272/60000 [==============>...............] - ETA: 56s - loss: 0.2686 - categorical_accuracy: 0.9166
30304/60000 [==============>...............] - ETA: 56s - loss: 0.2685 - categorical_accuracy: 0.9166
30336/60000 [==============>...............] - ETA: 56s - loss: 0.2682 - categorical_accuracy: 0.9167
30368/60000 [==============>...............] - ETA: 56s - loss: 0.2680 - categorical_accuracy: 0.9168
30400/60000 [==============>...............] - ETA: 56s - loss: 0.2681 - categorical_accuracy: 0.9168
30432/60000 [==============>...............] - ETA: 56s - loss: 0.2681 - categorical_accuracy: 0.9168
30464/60000 [==============>...............] - ETA: 56s - loss: 0.2680 - categorical_accuracy: 0.9169
30496/60000 [==============>...............] - ETA: 56s - loss: 0.2678 - categorical_accuracy: 0.9169
30528/60000 [==============>...............] - ETA: 56s - loss: 0.2675 - categorical_accuracy: 0.9170
30560/60000 [==============>...............] - ETA: 56s - loss: 0.2674 - categorical_accuracy: 0.9170
30592/60000 [==============>...............] - ETA: 56s - loss: 0.2673 - categorical_accuracy: 0.9170
30624/60000 [==============>...............] - ETA: 55s - loss: 0.2673 - categorical_accuracy: 0.9170
30656/60000 [==============>...............] - ETA: 55s - loss: 0.2671 - categorical_accuracy: 0.9171
30688/60000 [==============>...............] - ETA: 55s - loss: 0.2669 - categorical_accuracy: 0.9172
30720/60000 [==============>...............] - ETA: 55s - loss: 0.2670 - categorical_accuracy: 0.9172
30752/60000 [==============>...............] - ETA: 55s - loss: 0.2668 - categorical_accuracy: 0.9172
30784/60000 [==============>...............] - ETA: 55s - loss: 0.2666 - categorical_accuracy: 0.9173
30816/60000 [==============>...............] - ETA: 55s - loss: 0.2666 - categorical_accuracy: 0.9173
30848/60000 [==============>...............] - ETA: 55s - loss: 0.2664 - categorical_accuracy: 0.9174
30880/60000 [==============>...............] - ETA: 55s - loss: 0.2662 - categorical_accuracy: 0.9174
30912/60000 [==============>...............] - ETA: 55s - loss: 0.2661 - categorical_accuracy: 0.9174
30944/60000 [==============>...............] - ETA: 55s - loss: 0.2659 - categorical_accuracy: 0.9175
30976/60000 [==============>...............] - ETA: 55s - loss: 0.2657 - categorical_accuracy: 0.9175
31008/60000 [==============>...............] - ETA: 55s - loss: 0.2657 - categorical_accuracy: 0.9175
31040/60000 [==============>...............] - ETA: 55s - loss: 0.2657 - categorical_accuracy: 0.9176
31072/60000 [==============>...............] - ETA: 55s - loss: 0.2655 - categorical_accuracy: 0.9177
31104/60000 [==============>...............] - ETA: 55s - loss: 0.2652 - categorical_accuracy: 0.9178
31136/60000 [==============>...............] - ETA: 54s - loss: 0.2651 - categorical_accuracy: 0.9178
31168/60000 [==============>...............] - ETA: 54s - loss: 0.2650 - categorical_accuracy: 0.9178
31200/60000 [==============>...............] - ETA: 54s - loss: 0.2648 - categorical_accuracy: 0.9179
31232/60000 [==============>...............] - ETA: 54s - loss: 0.2646 - categorical_accuracy: 0.9180
31264/60000 [==============>...............] - ETA: 54s - loss: 0.2643 - categorical_accuracy: 0.9181
31296/60000 [==============>...............] - ETA: 54s - loss: 0.2642 - categorical_accuracy: 0.9181
31328/60000 [==============>...............] - ETA: 54s - loss: 0.2639 - categorical_accuracy: 0.9182
31360/60000 [==============>...............] - ETA: 54s - loss: 0.2638 - categorical_accuracy: 0.9183
31392/60000 [==============>...............] - ETA: 54s - loss: 0.2638 - categorical_accuracy: 0.9183
31424/60000 [==============>...............] - ETA: 54s - loss: 0.2636 - categorical_accuracy: 0.9183
31456/60000 [==============>...............] - ETA: 54s - loss: 0.2635 - categorical_accuracy: 0.9183
31488/60000 [==============>...............] - ETA: 54s - loss: 0.2634 - categorical_accuracy: 0.9184
31520/60000 [==============>...............] - ETA: 54s - loss: 0.2632 - categorical_accuracy: 0.9184
31552/60000 [==============>...............] - ETA: 54s - loss: 0.2630 - categorical_accuracy: 0.9185
31584/60000 [==============>...............] - ETA: 54s - loss: 0.2630 - categorical_accuracy: 0.9185
31616/60000 [==============>...............] - ETA: 54s - loss: 0.2632 - categorical_accuracy: 0.9184
31648/60000 [==============>...............] - ETA: 53s - loss: 0.2630 - categorical_accuracy: 0.9184
31680/60000 [==============>...............] - ETA: 53s - loss: 0.2629 - categorical_accuracy: 0.9185
31712/60000 [==============>...............] - ETA: 53s - loss: 0.2626 - categorical_accuracy: 0.9185
31744/60000 [==============>...............] - ETA: 53s - loss: 0.2624 - categorical_accuracy: 0.9186
31776/60000 [==============>...............] - ETA: 53s - loss: 0.2624 - categorical_accuracy: 0.9186
31808/60000 [==============>...............] - ETA: 53s - loss: 0.2624 - categorical_accuracy: 0.9186
31840/60000 [==============>...............] - ETA: 53s - loss: 0.2622 - categorical_accuracy: 0.9187
31872/60000 [==============>...............] - ETA: 53s - loss: 0.2620 - categorical_accuracy: 0.9187
31904/60000 [==============>...............] - ETA: 53s - loss: 0.2618 - categorical_accuracy: 0.9188
31936/60000 [==============>...............] - ETA: 53s - loss: 0.2617 - categorical_accuracy: 0.9189
31968/60000 [==============>...............] - ETA: 53s - loss: 0.2614 - categorical_accuracy: 0.9190
32000/60000 [===============>..............] - ETA: 53s - loss: 0.2614 - categorical_accuracy: 0.9190
32032/60000 [===============>..............] - ETA: 53s - loss: 0.2612 - categorical_accuracy: 0.9190
32064/60000 [===============>..............] - ETA: 53s - loss: 0.2609 - categorical_accuracy: 0.9191
32096/60000 [===============>..............] - ETA: 53s - loss: 0.2609 - categorical_accuracy: 0.9191
32128/60000 [===============>..............] - ETA: 53s - loss: 0.2607 - categorical_accuracy: 0.9192
32160/60000 [===============>..............] - ETA: 53s - loss: 0.2609 - categorical_accuracy: 0.9192
32192/60000 [===============>..............] - ETA: 52s - loss: 0.2607 - categorical_accuracy: 0.9192
32224/60000 [===============>..............] - ETA: 52s - loss: 0.2605 - categorical_accuracy: 0.9193
32256/60000 [===============>..............] - ETA: 52s - loss: 0.2604 - categorical_accuracy: 0.9193
32288/60000 [===============>..............] - ETA: 52s - loss: 0.2602 - categorical_accuracy: 0.9194
32320/60000 [===============>..............] - ETA: 52s - loss: 0.2600 - categorical_accuracy: 0.9194
32352/60000 [===============>..............] - ETA: 52s - loss: 0.2598 - categorical_accuracy: 0.9195
32384/60000 [===============>..............] - ETA: 52s - loss: 0.2597 - categorical_accuracy: 0.9195
32416/60000 [===============>..............] - ETA: 52s - loss: 0.2595 - categorical_accuracy: 0.9196
32448/60000 [===============>..............] - ETA: 52s - loss: 0.2593 - categorical_accuracy: 0.9197
32480/60000 [===============>..............] - ETA: 52s - loss: 0.2592 - categorical_accuracy: 0.9197
32512/60000 [===============>..............] - ETA: 52s - loss: 0.2593 - categorical_accuracy: 0.9197
32544/60000 [===============>..............] - ETA: 52s - loss: 0.2593 - categorical_accuracy: 0.9197
32576/60000 [===============>..............] - ETA: 52s - loss: 0.2592 - categorical_accuracy: 0.9198
32608/60000 [===============>..............] - ETA: 52s - loss: 0.2592 - categorical_accuracy: 0.9198
32640/60000 [===============>..............] - ETA: 52s - loss: 0.2591 - categorical_accuracy: 0.9198
32672/60000 [===============>..............] - ETA: 52s - loss: 0.2589 - categorical_accuracy: 0.9199
32704/60000 [===============>..............] - ETA: 51s - loss: 0.2587 - categorical_accuracy: 0.9199
32736/60000 [===============>..............] - ETA: 51s - loss: 0.2586 - categorical_accuracy: 0.9199
32768/60000 [===============>..............] - ETA: 51s - loss: 0.2585 - categorical_accuracy: 0.9200
32800/60000 [===============>..............] - ETA: 51s - loss: 0.2583 - categorical_accuracy: 0.9200
32832/60000 [===============>..............] - ETA: 51s - loss: 0.2583 - categorical_accuracy: 0.9200
32864/60000 [===============>..............] - ETA: 51s - loss: 0.2581 - categorical_accuracy: 0.9201
32896/60000 [===============>..............] - ETA: 51s - loss: 0.2580 - categorical_accuracy: 0.9201
32928/60000 [===============>..............] - ETA: 51s - loss: 0.2581 - categorical_accuracy: 0.9201
32960/60000 [===============>..............] - ETA: 51s - loss: 0.2579 - categorical_accuracy: 0.9201
32992/60000 [===============>..............] - ETA: 51s - loss: 0.2578 - categorical_accuracy: 0.9202
33024/60000 [===============>..............] - ETA: 51s - loss: 0.2578 - categorical_accuracy: 0.9202
33056/60000 [===============>..............] - ETA: 51s - loss: 0.2576 - categorical_accuracy: 0.9203
33088/60000 [===============>..............] - ETA: 51s - loss: 0.2574 - categorical_accuracy: 0.9204
33120/60000 [===============>..............] - ETA: 51s - loss: 0.2572 - categorical_accuracy: 0.9204
33152/60000 [===============>..............] - ETA: 51s - loss: 0.2570 - categorical_accuracy: 0.9205
33184/60000 [===============>..............] - ETA: 51s - loss: 0.2568 - categorical_accuracy: 0.9206
33216/60000 [===============>..............] - ETA: 51s - loss: 0.2566 - categorical_accuracy: 0.9206
33248/60000 [===============>..............] - ETA: 50s - loss: 0.2564 - categorical_accuracy: 0.9207
33280/60000 [===============>..............] - ETA: 50s - loss: 0.2562 - categorical_accuracy: 0.9208
33312/60000 [===============>..............] - ETA: 50s - loss: 0.2560 - categorical_accuracy: 0.9208
33344/60000 [===============>..............] - ETA: 50s - loss: 0.2559 - categorical_accuracy: 0.9209
33376/60000 [===============>..............] - ETA: 50s - loss: 0.2557 - categorical_accuracy: 0.9210
33408/60000 [===============>..............] - ETA: 50s - loss: 0.2555 - categorical_accuracy: 0.9210
33440/60000 [===============>..............] - ETA: 50s - loss: 0.2555 - categorical_accuracy: 0.9210
33472/60000 [===============>..............] - ETA: 50s - loss: 0.2554 - categorical_accuracy: 0.9211
33504/60000 [===============>..............] - ETA: 50s - loss: 0.2553 - categorical_accuracy: 0.9211
33536/60000 [===============>..............] - ETA: 50s - loss: 0.2551 - categorical_accuracy: 0.9211
33568/60000 [===============>..............] - ETA: 50s - loss: 0.2550 - categorical_accuracy: 0.9211
33600/60000 [===============>..............] - ETA: 50s - loss: 0.2550 - categorical_accuracy: 0.9211
33632/60000 [===============>..............] - ETA: 50s - loss: 0.2547 - categorical_accuracy: 0.9212
33664/60000 [===============>..............] - ETA: 50s - loss: 0.2545 - categorical_accuracy: 0.9213
33696/60000 [===============>..............] - ETA: 50s - loss: 0.2544 - categorical_accuracy: 0.9213
33728/60000 [===============>..............] - ETA: 50s - loss: 0.2542 - categorical_accuracy: 0.9213
33760/60000 [===============>..............] - ETA: 49s - loss: 0.2540 - categorical_accuracy: 0.9214
33792/60000 [===============>..............] - ETA: 49s - loss: 0.2539 - categorical_accuracy: 0.9214
33824/60000 [===============>..............] - ETA: 49s - loss: 0.2538 - categorical_accuracy: 0.9214
33856/60000 [===============>..............] - ETA: 49s - loss: 0.2536 - categorical_accuracy: 0.9215
33888/60000 [===============>..............] - ETA: 49s - loss: 0.2535 - categorical_accuracy: 0.9215
33920/60000 [===============>..............] - ETA: 49s - loss: 0.2536 - categorical_accuracy: 0.9216
33952/60000 [===============>..............] - ETA: 49s - loss: 0.2535 - categorical_accuracy: 0.9216
33984/60000 [===============>..............] - ETA: 49s - loss: 0.2532 - categorical_accuracy: 0.9217
34016/60000 [================>.............] - ETA: 49s - loss: 0.2531 - categorical_accuracy: 0.9217
34048/60000 [================>.............] - ETA: 49s - loss: 0.2530 - categorical_accuracy: 0.9218
34080/60000 [================>.............] - ETA: 49s - loss: 0.2530 - categorical_accuracy: 0.9218
34112/60000 [================>.............] - ETA: 49s - loss: 0.2528 - categorical_accuracy: 0.9218
34144/60000 [================>.............] - ETA: 49s - loss: 0.2526 - categorical_accuracy: 0.9219
34176/60000 [================>.............] - ETA: 49s - loss: 0.2524 - categorical_accuracy: 0.9220
34208/60000 [================>.............] - ETA: 49s - loss: 0.2522 - categorical_accuracy: 0.9220
34240/60000 [================>.............] - ETA: 49s - loss: 0.2521 - categorical_accuracy: 0.9220
34272/60000 [================>.............] - ETA: 49s - loss: 0.2520 - categorical_accuracy: 0.9221
34304/60000 [================>.............] - ETA: 48s - loss: 0.2518 - categorical_accuracy: 0.9221
34336/60000 [================>.............] - ETA: 48s - loss: 0.2516 - categorical_accuracy: 0.9222
34368/60000 [================>.............] - ETA: 48s - loss: 0.2517 - categorical_accuracy: 0.9222
34400/60000 [================>.............] - ETA: 48s - loss: 0.2515 - categorical_accuracy: 0.9222
34432/60000 [================>.............] - ETA: 48s - loss: 0.2513 - categorical_accuracy: 0.9223
34464/60000 [================>.............] - ETA: 48s - loss: 0.2511 - categorical_accuracy: 0.9224
34496/60000 [================>.............] - ETA: 48s - loss: 0.2509 - categorical_accuracy: 0.9224
34528/60000 [================>.............] - ETA: 48s - loss: 0.2507 - categorical_accuracy: 0.9224
34560/60000 [================>.............] - ETA: 48s - loss: 0.2505 - categorical_accuracy: 0.9225
34592/60000 [================>.............] - ETA: 48s - loss: 0.2503 - categorical_accuracy: 0.9226
34624/60000 [================>.............] - ETA: 48s - loss: 0.2501 - categorical_accuracy: 0.9227
34656/60000 [================>.............] - ETA: 48s - loss: 0.2499 - categorical_accuracy: 0.9227
34688/60000 [================>.............] - ETA: 48s - loss: 0.2497 - categorical_accuracy: 0.9228
34720/60000 [================>.............] - ETA: 48s - loss: 0.2496 - categorical_accuracy: 0.9228
34752/60000 [================>.............] - ETA: 48s - loss: 0.2494 - categorical_accuracy: 0.9229
34784/60000 [================>.............] - ETA: 48s - loss: 0.2492 - categorical_accuracy: 0.9230
34816/60000 [================>.............] - ETA: 47s - loss: 0.2491 - categorical_accuracy: 0.9230
34848/60000 [================>.............] - ETA: 47s - loss: 0.2489 - categorical_accuracy: 0.9230
34880/60000 [================>.............] - ETA: 47s - loss: 0.2487 - categorical_accuracy: 0.9231
34912/60000 [================>.............] - ETA: 47s - loss: 0.2486 - categorical_accuracy: 0.9231
34944/60000 [================>.............] - ETA: 47s - loss: 0.2487 - categorical_accuracy: 0.9232
34976/60000 [================>.............] - ETA: 47s - loss: 0.2486 - categorical_accuracy: 0.9232
35008/60000 [================>.............] - ETA: 47s - loss: 0.2485 - categorical_accuracy: 0.9233
35040/60000 [================>.............] - ETA: 47s - loss: 0.2483 - categorical_accuracy: 0.9233
35072/60000 [================>.............] - ETA: 47s - loss: 0.2481 - categorical_accuracy: 0.9234
35104/60000 [================>.............] - ETA: 47s - loss: 0.2482 - categorical_accuracy: 0.9234
35136/60000 [================>.............] - ETA: 47s - loss: 0.2483 - categorical_accuracy: 0.9234
35168/60000 [================>.............] - ETA: 47s - loss: 0.2482 - categorical_accuracy: 0.9235
35200/60000 [================>.............] - ETA: 47s - loss: 0.2480 - categorical_accuracy: 0.9235
35232/60000 [================>.............] - ETA: 47s - loss: 0.2480 - categorical_accuracy: 0.9235
35264/60000 [================>.............] - ETA: 47s - loss: 0.2479 - categorical_accuracy: 0.9235
35296/60000 [================>.............] - ETA: 47s - loss: 0.2478 - categorical_accuracy: 0.9236
35328/60000 [================>.............] - ETA: 46s - loss: 0.2476 - categorical_accuracy: 0.9236
35360/60000 [================>.............] - ETA: 46s - loss: 0.2474 - categorical_accuracy: 0.9237
35392/60000 [================>.............] - ETA: 46s - loss: 0.2474 - categorical_accuracy: 0.9237
35424/60000 [================>.............] - ETA: 46s - loss: 0.2472 - categorical_accuracy: 0.9238
35456/60000 [================>.............] - ETA: 46s - loss: 0.2470 - categorical_accuracy: 0.9238
35488/60000 [================>.............] - ETA: 46s - loss: 0.2469 - categorical_accuracy: 0.9238
35520/60000 [================>.............] - ETA: 46s - loss: 0.2467 - categorical_accuracy: 0.9239
35552/60000 [================>.............] - ETA: 46s - loss: 0.2465 - categorical_accuracy: 0.9239
35584/60000 [================>.............] - ETA: 46s - loss: 0.2468 - categorical_accuracy: 0.9239
35616/60000 [================>.............] - ETA: 46s - loss: 0.2466 - categorical_accuracy: 0.9240
35648/60000 [================>.............] - ETA: 46s - loss: 0.2465 - categorical_accuracy: 0.9240
35680/60000 [================>.............] - ETA: 46s - loss: 0.2466 - categorical_accuracy: 0.9240
35712/60000 [================>.............] - ETA: 46s - loss: 0.2466 - categorical_accuracy: 0.9240
35744/60000 [================>.............] - ETA: 46s - loss: 0.2465 - categorical_accuracy: 0.9240
35776/60000 [================>.............] - ETA: 46s - loss: 0.2464 - categorical_accuracy: 0.9240
35808/60000 [================>.............] - ETA: 46s - loss: 0.2463 - categorical_accuracy: 0.9241
35840/60000 [================>.............] - ETA: 45s - loss: 0.2461 - categorical_accuracy: 0.9242
35872/60000 [================>.............] - ETA: 45s - loss: 0.2459 - categorical_accuracy: 0.9242
35904/60000 [================>.............] - ETA: 45s - loss: 0.2461 - categorical_accuracy: 0.9242
35936/60000 [================>.............] - ETA: 45s - loss: 0.2460 - categorical_accuracy: 0.9242
35968/60000 [================>.............] - ETA: 45s - loss: 0.2459 - categorical_accuracy: 0.9242
36000/60000 [=================>............] - ETA: 45s - loss: 0.2459 - categorical_accuracy: 0.9242
36032/60000 [=================>............] - ETA: 45s - loss: 0.2457 - categorical_accuracy: 0.9243
36064/60000 [=================>............] - ETA: 45s - loss: 0.2455 - categorical_accuracy: 0.9244
36096/60000 [=================>............] - ETA: 45s - loss: 0.2455 - categorical_accuracy: 0.9243
36128/60000 [=================>............] - ETA: 45s - loss: 0.2453 - categorical_accuracy: 0.9244
36160/60000 [=================>............] - ETA: 45s - loss: 0.2455 - categorical_accuracy: 0.9244
36192/60000 [=================>............] - ETA: 45s - loss: 0.2453 - categorical_accuracy: 0.9245
36224/60000 [=================>............] - ETA: 45s - loss: 0.2451 - categorical_accuracy: 0.9245
36256/60000 [=================>............] - ETA: 45s - loss: 0.2450 - categorical_accuracy: 0.9246
36288/60000 [=================>............] - ETA: 45s - loss: 0.2448 - categorical_accuracy: 0.9246
36320/60000 [=================>............] - ETA: 45s - loss: 0.2446 - categorical_accuracy: 0.9246
36352/60000 [=================>............] - ETA: 45s - loss: 0.2446 - categorical_accuracy: 0.9247
36384/60000 [=================>............] - ETA: 44s - loss: 0.2450 - categorical_accuracy: 0.9247
36416/60000 [=================>............] - ETA: 44s - loss: 0.2451 - categorical_accuracy: 0.9247
36448/60000 [=================>............] - ETA: 44s - loss: 0.2450 - categorical_accuracy: 0.9248
36480/60000 [=================>............] - ETA: 44s - loss: 0.2448 - categorical_accuracy: 0.9248
36512/60000 [=================>............] - ETA: 44s - loss: 0.2446 - categorical_accuracy: 0.9248
36544/60000 [=================>............] - ETA: 44s - loss: 0.2446 - categorical_accuracy: 0.9249
36576/60000 [=================>............] - ETA: 44s - loss: 0.2444 - categorical_accuracy: 0.9249
36608/60000 [=================>............] - ETA: 44s - loss: 0.2443 - categorical_accuracy: 0.9250
36640/60000 [=================>............] - ETA: 44s - loss: 0.2442 - categorical_accuracy: 0.9250
36672/60000 [=================>............] - ETA: 44s - loss: 0.2441 - categorical_accuracy: 0.9250
36704/60000 [=================>............] - ETA: 44s - loss: 0.2439 - categorical_accuracy: 0.9251
36736/60000 [=================>............] - ETA: 44s - loss: 0.2438 - categorical_accuracy: 0.9251
36768/60000 [=================>............] - ETA: 44s - loss: 0.2436 - categorical_accuracy: 0.9252
36800/60000 [=================>............] - ETA: 44s - loss: 0.2434 - categorical_accuracy: 0.9253
36832/60000 [=================>............] - ETA: 44s - loss: 0.2432 - categorical_accuracy: 0.9253
36864/60000 [=================>............] - ETA: 44s - loss: 0.2431 - categorical_accuracy: 0.9254
36896/60000 [=================>............] - ETA: 43s - loss: 0.2429 - categorical_accuracy: 0.9254
36928/60000 [=================>............] - ETA: 43s - loss: 0.2427 - categorical_accuracy: 0.9255
36960/60000 [=================>............] - ETA: 43s - loss: 0.2425 - categorical_accuracy: 0.9256
36992/60000 [=================>............] - ETA: 43s - loss: 0.2425 - categorical_accuracy: 0.9256
37024/60000 [=================>............] - ETA: 43s - loss: 0.2423 - categorical_accuracy: 0.9256
37056/60000 [=================>............] - ETA: 43s - loss: 0.2422 - categorical_accuracy: 0.9257
37088/60000 [=================>............] - ETA: 43s - loss: 0.2422 - categorical_accuracy: 0.9257
37120/60000 [=================>............] - ETA: 43s - loss: 0.2422 - categorical_accuracy: 0.9257
37152/60000 [=================>............] - ETA: 43s - loss: 0.2420 - categorical_accuracy: 0.9257
37184/60000 [=================>............] - ETA: 43s - loss: 0.2418 - categorical_accuracy: 0.9258
37216/60000 [=================>............] - ETA: 43s - loss: 0.2416 - categorical_accuracy: 0.9258
37248/60000 [=================>............] - ETA: 43s - loss: 0.2415 - categorical_accuracy: 0.9259
37280/60000 [=================>............] - ETA: 43s - loss: 0.2413 - categorical_accuracy: 0.9259
37312/60000 [=================>............] - ETA: 43s - loss: 0.2412 - categorical_accuracy: 0.9259
37344/60000 [=================>............] - ETA: 43s - loss: 0.2410 - categorical_accuracy: 0.9260
37376/60000 [=================>............] - ETA: 43s - loss: 0.2410 - categorical_accuracy: 0.9260
37408/60000 [=================>............] - ETA: 42s - loss: 0.2408 - categorical_accuracy: 0.9261
37440/60000 [=================>............] - ETA: 42s - loss: 0.2406 - categorical_accuracy: 0.9261
37472/60000 [=================>............] - ETA: 42s - loss: 0.2404 - categorical_accuracy: 0.9262
37504/60000 [=================>............] - ETA: 42s - loss: 0.2404 - categorical_accuracy: 0.9262
37536/60000 [=================>............] - ETA: 42s - loss: 0.2405 - categorical_accuracy: 0.9263
37568/60000 [=================>............] - ETA: 42s - loss: 0.2403 - categorical_accuracy: 0.9263
37600/60000 [=================>............] - ETA: 42s - loss: 0.2402 - categorical_accuracy: 0.9263
37632/60000 [=================>............] - ETA: 42s - loss: 0.2400 - categorical_accuracy: 0.9264
37664/60000 [=================>............] - ETA: 42s - loss: 0.2398 - categorical_accuracy: 0.9265
37696/60000 [=================>............] - ETA: 42s - loss: 0.2397 - categorical_accuracy: 0.9265
37728/60000 [=================>............] - ETA: 42s - loss: 0.2395 - categorical_accuracy: 0.9265
37760/60000 [=================>............] - ETA: 42s - loss: 0.2393 - categorical_accuracy: 0.9266
37792/60000 [=================>............] - ETA: 42s - loss: 0.2395 - categorical_accuracy: 0.9266
37824/60000 [=================>............] - ETA: 42s - loss: 0.2393 - categorical_accuracy: 0.9266
37856/60000 [=================>............] - ETA: 42s - loss: 0.2392 - categorical_accuracy: 0.9266
37888/60000 [=================>............] - ETA: 42s - loss: 0.2391 - categorical_accuracy: 0.9267
37920/60000 [=================>............] - ETA: 42s - loss: 0.2390 - categorical_accuracy: 0.9267
37952/60000 [=================>............] - ETA: 41s - loss: 0.2388 - categorical_accuracy: 0.9267
37984/60000 [=================>............] - ETA: 41s - loss: 0.2387 - categorical_accuracy: 0.9268
38016/60000 [==================>...........] - ETA: 41s - loss: 0.2385 - categorical_accuracy: 0.9268
38048/60000 [==================>...........] - ETA: 41s - loss: 0.2383 - categorical_accuracy: 0.9269
38080/60000 [==================>...........] - ETA: 41s - loss: 0.2381 - categorical_accuracy: 0.9269
38112/60000 [==================>...........] - ETA: 41s - loss: 0.2380 - categorical_accuracy: 0.9270
38144/60000 [==================>...........] - ETA: 41s - loss: 0.2378 - categorical_accuracy: 0.9270
38176/60000 [==================>...........] - ETA: 41s - loss: 0.2377 - categorical_accuracy: 0.9271
38208/60000 [==================>...........] - ETA: 41s - loss: 0.2376 - categorical_accuracy: 0.9271
38240/60000 [==================>...........] - ETA: 41s - loss: 0.2375 - categorical_accuracy: 0.9271
38272/60000 [==================>...........] - ETA: 41s - loss: 0.2374 - categorical_accuracy: 0.9271
38304/60000 [==================>...........] - ETA: 41s - loss: 0.2373 - categorical_accuracy: 0.9271
38336/60000 [==================>...........] - ETA: 41s - loss: 0.2371 - categorical_accuracy: 0.9271
38368/60000 [==================>...........] - ETA: 41s - loss: 0.2369 - categorical_accuracy: 0.9272
38400/60000 [==================>...........] - ETA: 41s - loss: 0.2371 - categorical_accuracy: 0.9272
38432/60000 [==================>...........] - ETA: 41s - loss: 0.2370 - categorical_accuracy: 0.9272
38464/60000 [==================>...........] - ETA: 40s - loss: 0.2368 - categorical_accuracy: 0.9273
38496/60000 [==================>...........] - ETA: 40s - loss: 0.2367 - categorical_accuracy: 0.9273
38528/60000 [==================>...........] - ETA: 40s - loss: 0.2366 - categorical_accuracy: 0.9274
38560/60000 [==================>...........] - ETA: 40s - loss: 0.2365 - categorical_accuracy: 0.9274
38592/60000 [==================>...........] - ETA: 40s - loss: 0.2364 - categorical_accuracy: 0.9274
38624/60000 [==================>...........] - ETA: 40s - loss: 0.2362 - categorical_accuracy: 0.9275
38656/60000 [==================>...........] - ETA: 40s - loss: 0.2360 - categorical_accuracy: 0.9275
38688/60000 [==================>...........] - ETA: 40s - loss: 0.2359 - categorical_accuracy: 0.9276
38720/60000 [==================>...........] - ETA: 40s - loss: 0.2359 - categorical_accuracy: 0.9276
38752/60000 [==================>...........] - ETA: 40s - loss: 0.2358 - categorical_accuracy: 0.9276
38784/60000 [==================>...........] - ETA: 40s - loss: 0.2356 - categorical_accuracy: 0.9277
38816/60000 [==================>...........] - ETA: 40s - loss: 0.2356 - categorical_accuracy: 0.9277
38848/60000 [==================>...........] - ETA: 40s - loss: 0.2356 - categorical_accuracy: 0.9277
38880/60000 [==================>...........] - ETA: 40s - loss: 0.2354 - categorical_accuracy: 0.9277
38912/60000 [==================>...........] - ETA: 40s - loss: 0.2352 - categorical_accuracy: 0.9278
38944/60000 [==================>...........] - ETA: 40s - loss: 0.2353 - categorical_accuracy: 0.9278
38976/60000 [==================>...........] - ETA: 40s - loss: 0.2351 - categorical_accuracy: 0.9278
39008/60000 [==================>...........] - ETA: 39s - loss: 0.2350 - categorical_accuracy: 0.9278
39040/60000 [==================>...........] - ETA: 39s - loss: 0.2349 - categorical_accuracy: 0.9279
39072/60000 [==================>...........] - ETA: 39s - loss: 0.2348 - categorical_accuracy: 0.9279
39104/60000 [==================>...........] - ETA: 39s - loss: 0.2346 - categorical_accuracy: 0.9280
39136/60000 [==================>...........] - ETA: 39s - loss: 0.2345 - categorical_accuracy: 0.9280
39168/60000 [==================>...........] - ETA: 39s - loss: 0.2343 - categorical_accuracy: 0.9281
39200/60000 [==================>...........] - ETA: 39s - loss: 0.2341 - categorical_accuracy: 0.9281
39232/60000 [==================>...........] - ETA: 39s - loss: 0.2340 - categorical_accuracy: 0.9282
39264/60000 [==================>...........] - ETA: 39s - loss: 0.2339 - categorical_accuracy: 0.9282
39296/60000 [==================>...........] - ETA: 39s - loss: 0.2338 - categorical_accuracy: 0.9283
39328/60000 [==================>...........] - ETA: 39s - loss: 0.2336 - categorical_accuracy: 0.9283
39360/60000 [==================>...........] - ETA: 39s - loss: 0.2337 - categorical_accuracy: 0.9282
39392/60000 [==================>...........] - ETA: 39s - loss: 0.2335 - categorical_accuracy: 0.9283
39424/60000 [==================>...........] - ETA: 39s - loss: 0.2334 - categorical_accuracy: 0.9283
39456/60000 [==================>...........] - ETA: 39s - loss: 0.2334 - categorical_accuracy: 0.9283
39488/60000 [==================>...........] - ETA: 39s - loss: 0.2333 - categorical_accuracy: 0.9283
39520/60000 [==================>...........] - ETA: 38s - loss: 0.2332 - categorical_accuracy: 0.9283
39552/60000 [==================>...........] - ETA: 38s - loss: 0.2331 - categorical_accuracy: 0.9283
39584/60000 [==================>...........] - ETA: 38s - loss: 0.2332 - categorical_accuracy: 0.9284
39616/60000 [==================>...........] - ETA: 38s - loss: 0.2331 - categorical_accuracy: 0.9284
39648/60000 [==================>...........] - ETA: 38s - loss: 0.2330 - categorical_accuracy: 0.9285
39680/60000 [==================>...........] - ETA: 38s - loss: 0.2328 - categorical_accuracy: 0.9285
39712/60000 [==================>...........] - ETA: 38s - loss: 0.2326 - categorical_accuracy: 0.9286
39744/60000 [==================>...........] - ETA: 38s - loss: 0.2325 - categorical_accuracy: 0.9286
39776/60000 [==================>...........] - ETA: 38s - loss: 0.2324 - categorical_accuracy: 0.9287
39808/60000 [==================>...........] - ETA: 38s - loss: 0.2323 - categorical_accuracy: 0.9287
39840/60000 [==================>...........] - ETA: 38s - loss: 0.2322 - categorical_accuracy: 0.9287
39872/60000 [==================>...........] - ETA: 38s - loss: 0.2321 - categorical_accuracy: 0.9287
39904/60000 [==================>...........] - ETA: 38s - loss: 0.2320 - categorical_accuracy: 0.9288
39936/60000 [==================>...........] - ETA: 38s - loss: 0.2319 - categorical_accuracy: 0.9288
39968/60000 [==================>...........] - ETA: 38s - loss: 0.2317 - categorical_accuracy: 0.9288
40000/60000 [===================>..........] - ETA: 38s - loss: 0.2316 - categorical_accuracy: 0.9288
40032/60000 [===================>..........] - ETA: 37s - loss: 0.2315 - categorical_accuracy: 0.9289
40064/60000 [===================>..........] - ETA: 37s - loss: 0.2313 - categorical_accuracy: 0.9289
40096/60000 [===================>..........] - ETA: 37s - loss: 0.2312 - categorical_accuracy: 0.9290
40128/60000 [===================>..........] - ETA: 37s - loss: 0.2312 - categorical_accuracy: 0.9290
40160/60000 [===================>..........] - ETA: 37s - loss: 0.2312 - categorical_accuracy: 0.9290
40192/60000 [===================>..........] - ETA: 37s - loss: 0.2310 - categorical_accuracy: 0.9290
40224/60000 [===================>..........] - ETA: 37s - loss: 0.2309 - categorical_accuracy: 0.9291
40256/60000 [===================>..........] - ETA: 37s - loss: 0.2308 - categorical_accuracy: 0.9291
40288/60000 [===================>..........] - ETA: 37s - loss: 0.2307 - categorical_accuracy: 0.9291
40320/60000 [===================>..........] - ETA: 37s - loss: 0.2306 - categorical_accuracy: 0.9292
40352/60000 [===================>..........] - ETA: 37s - loss: 0.2304 - categorical_accuracy: 0.9292
40384/60000 [===================>..........] - ETA: 37s - loss: 0.2303 - categorical_accuracy: 0.9293
40416/60000 [===================>..........] - ETA: 37s - loss: 0.2301 - categorical_accuracy: 0.9293
40448/60000 [===================>..........] - ETA: 37s - loss: 0.2301 - categorical_accuracy: 0.9293
40480/60000 [===================>..........] - ETA: 37s - loss: 0.2301 - categorical_accuracy: 0.9293
40512/60000 [===================>..........] - ETA: 37s - loss: 0.2300 - categorical_accuracy: 0.9294
40544/60000 [===================>..........] - ETA: 37s - loss: 0.2298 - categorical_accuracy: 0.9294
40576/60000 [===================>..........] - ETA: 36s - loss: 0.2298 - categorical_accuracy: 0.9294
40608/60000 [===================>..........] - ETA: 36s - loss: 0.2296 - categorical_accuracy: 0.9295
40640/60000 [===================>..........] - ETA: 36s - loss: 0.2296 - categorical_accuracy: 0.9295
40672/60000 [===================>..........] - ETA: 36s - loss: 0.2294 - categorical_accuracy: 0.9296
40704/60000 [===================>..........] - ETA: 36s - loss: 0.2292 - categorical_accuracy: 0.9296
40736/60000 [===================>..........] - ETA: 36s - loss: 0.2294 - categorical_accuracy: 0.9296
40768/60000 [===================>..........] - ETA: 36s - loss: 0.2292 - categorical_accuracy: 0.9296
40800/60000 [===================>..........] - ETA: 36s - loss: 0.2291 - categorical_accuracy: 0.9297
40832/60000 [===================>..........] - ETA: 36s - loss: 0.2290 - categorical_accuracy: 0.9296
40864/60000 [===================>..........] - ETA: 36s - loss: 0.2289 - categorical_accuracy: 0.9297
40896/60000 [===================>..........] - ETA: 36s - loss: 0.2288 - categorical_accuracy: 0.9297
40928/60000 [===================>..........] - ETA: 36s - loss: 0.2287 - categorical_accuracy: 0.9297
40960/60000 [===================>..........] - ETA: 36s - loss: 0.2285 - categorical_accuracy: 0.9298
40992/60000 [===================>..........] - ETA: 36s - loss: 0.2284 - categorical_accuracy: 0.9298
41024/60000 [===================>..........] - ETA: 36s - loss: 0.2284 - categorical_accuracy: 0.9298
41056/60000 [===================>..........] - ETA: 36s - loss: 0.2283 - categorical_accuracy: 0.9299
41088/60000 [===================>..........] - ETA: 35s - loss: 0.2282 - categorical_accuracy: 0.9299
41120/60000 [===================>..........] - ETA: 35s - loss: 0.2281 - categorical_accuracy: 0.9299
41152/60000 [===================>..........] - ETA: 35s - loss: 0.2280 - categorical_accuracy: 0.9300
41184/60000 [===================>..........] - ETA: 35s - loss: 0.2279 - categorical_accuracy: 0.9300
41216/60000 [===================>..........] - ETA: 35s - loss: 0.2278 - categorical_accuracy: 0.9300
41248/60000 [===================>..........] - ETA: 35s - loss: 0.2277 - categorical_accuracy: 0.9300
41280/60000 [===================>..........] - ETA: 35s - loss: 0.2277 - categorical_accuracy: 0.9300
41312/60000 [===================>..........] - ETA: 35s - loss: 0.2277 - categorical_accuracy: 0.9300
41344/60000 [===================>..........] - ETA: 35s - loss: 0.2277 - categorical_accuracy: 0.9300
41376/60000 [===================>..........] - ETA: 35s - loss: 0.2276 - categorical_accuracy: 0.9300
41408/60000 [===================>..........] - ETA: 35s - loss: 0.2274 - categorical_accuracy: 0.9301
41440/60000 [===================>..........] - ETA: 35s - loss: 0.2273 - categorical_accuracy: 0.9301
41472/60000 [===================>..........] - ETA: 35s - loss: 0.2272 - categorical_accuracy: 0.9301
41504/60000 [===================>..........] - ETA: 35s - loss: 0.2271 - categorical_accuracy: 0.9302
41536/60000 [===================>..........] - ETA: 35s - loss: 0.2270 - categorical_accuracy: 0.9302
41568/60000 [===================>..........] - ETA: 35s - loss: 0.2270 - categorical_accuracy: 0.9302
41600/60000 [===================>..........] - ETA: 34s - loss: 0.2269 - categorical_accuracy: 0.9302
41632/60000 [===================>..........] - ETA: 34s - loss: 0.2267 - categorical_accuracy: 0.9302
41664/60000 [===================>..........] - ETA: 34s - loss: 0.2267 - categorical_accuracy: 0.9302
41696/60000 [===================>..........] - ETA: 34s - loss: 0.2265 - categorical_accuracy: 0.9303
41728/60000 [===================>..........] - ETA: 34s - loss: 0.2264 - categorical_accuracy: 0.9303
41760/60000 [===================>..........] - ETA: 34s - loss: 0.2263 - categorical_accuracy: 0.9304
41792/60000 [===================>..........] - ETA: 34s - loss: 0.2261 - categorical_accuracy: 0.9304
41824/60000 [===================>..........] - ETA: 34s - loss: 0.2261 - categorical_accuracy: 0.9304
41856/60000 [===================>..........] - ETA: 34s - loss: 0.2259 - categorical_accuracy: 0.9305
41888/60000 [===================>..........] - ETA: 34s - loss: 0.2259 - categorical_accuracy: 0.9305
41920/60000 [===================>..........] - ETA: 34s - loss: 0.2259 - categorical_accuracy: 0.9305
41952/60000 [===================>..........] - ETA: 34s - loss: 0.2258 - categorical_accuracy: 0.9306
41984/60000 [===================>..........] - ETA: 34s - loss: 0.2257 - categorical_accuracy: 0.9306
42016/60000 [====================>.........] - ETA: 34s - loss: 0.2257 - categorical_accuracy: 0.9306
42048/60000 [====================>.........] - ETA: 34s - loss: 0.2257 - categorical_accuracy: 0.9306
42080/60000 [====================>.........] - ETA: 34s - loss: 0.2256 - categorical_accuracy: 0.9306
42112/60000 [====================>.........] - ETA: 34s - loss: 0.2256 - categorical_accuracy: 0.9307
42144/60000 [====================>.........] - ETA: 33s - loss: 0.2254 - categorical_accuracy: 0.9307
42176/60000 [====================>.........] - ETA: 33s - loss: 0.2253 - categorical_accuracy: 0.9307
42208/60000 [====================>.........] - ETA: 33s - loss: 0.2252 - categorical_accuracy: 0.9308
42240/60000 [====================>.........] - ETA: 33s - loss: 0.2251 - categorical_accuracy: 0.9308
42272/60000 [====================>.........] - ETA: 33s - loss: 0.2250 - categorical_accuracy: 0.9308
42304/60000 [====================>.........] - ETA: 33s - loss: 0.2249 - categorical_accuracy: 0.9309
42336/60000 [====================>.........] - ETA: 33s - loss: 0.2247 - categorical_accuracy: 0.9309
42368/60000 [====================>.........] - ETA: 33s - loss: 0.2246 - categorical_accuracy: 0.9310
42400/60000 [====================>.........] - ETA: 33s - loss: 0.2244 - categorical_accuracy: 0.9310
42432/60000 [====================>.........] - ETA: 33s - loss: 0.2244 - categorical_accuracy: 0.9310
42464/60000 [====================>.........] - ETA: 33s - loss: 0.2243 - categorical_accuracy: 0.9310
42496/60000 [====================>.........] - ETA: 33s - loss: 0.2242 - categorical_accuracy: 0.9311
42528/60000 [====================>.........] - ETA: 33s - loss: 0.2241 - categorical_accuracy: 0.9311
42560/60000 [====================>.........] - ETA: 33s - loss: 0.2241 - categorical_accuracy: 0.9311
42592/60000 [====================>.........] - ETA: 33s - loss: 0.2240 - categorical_accuracy: 0.9312
42624/60000 [====================>.........] - ETA: 33s - loss: 0.2240 - categorical_accuracy: 0.9311
42656/60000 [====================>.........] - ETA: 32s - loss: 0.2238 - categorical_accuracy: 0.9312
42688/60000 [====================>.........] - ETA: 32s - loss: 0.2237 - categorical_accuracy: 0.9312
42720/60000 [====================>.........] - ETA: 32s - loss: 0.2237 - categorical_accuracy: 0.9313
42752/60000 [====================>.........] - ETA: 32s - loss: 0.2235 - categorical_accuracy: 0.9313
42784/60000 [====================>.........] - ETA: 32s - loss: 0.2234 - categorical_accuracy: 0.9314
42816/60000 [====================>.........] - ETA: 32s - loss: 0.2232 - categorical_accuracy: 0.9314
42848/60000 [====================>.........] - ETA: 32s - loss: 0.2232 - categorical_accuracy: 0.9314
42880/60000 [====================>.........] - ETA: 32s - loss: 0.2231 - categorical_accuracy: 0.9315
42912/60000 [====================>.........] - ETA: 32s - loss: 0.2229 - categorical_accuracy: 0.9315
42944/60000 [====================>.........] - ETA: 32s - loss: 0.2228 - categorical_accuracy: 0.9316
42976/60000 [====================>.........] - ETA: 32s - loss: 0.2226 - categorical_accuracy: 0.9316
43008/60000 [====================>.........] - ETA: 32s - loss: 0.2225 - categorical_accuracy: 0.9317
43040/60000 [====================>.........] - ETA: 32s - loss: 0.2225 - categorical_accuracy: 0.9317
43072/60000 [====================>.........] - ETA: 32s - loss: 0.2223 - categorical_accuracy: 0.9317
43104/60000 [====================>.........] - ETA: 32s - loss: 0.2223 - categorical_accuracy: 0.9317
43136/60000 [====================>.........] - ETA: 32s - loss: 0.2221 - categorical_accuracy: 0.9318
43168/60000 [====================>.........] - ETA: 32s - loss: 0.2221 - categorical_accuracy: 0.9318
43200/60000 [====================>.........] - ETA: 31s - loss: 0.2220 - categorical_accuracy: 0.9318
43232/60000 [====================>.........] - ETA: 31s - loss: 0.2218 - categorical_accuracy: 0.9319
43264/60000 [====================>.........] - ETA: 31s - loss: 0.2217 - categorical_accuracy: 0.9319
43296/60000 [====================>.........] - ETA: 31s - loss: 0.2217 - categorical_accuracy: 0.9319
43328/60000 [====================>.........] - ETA: 31s - loss: 0.2215 - categorical_accuracy: 0.9320
43360/60000 [====================>.........] - ETA: 31s - loss: 0.2218 - categorical_accuracy: 0.9320
43392/60000 [====================>.........] - ETA: 31s - loss: 0.2216 - categorical_accuracy: 0.9320
43424/60000 [====================>.........] - ETA: 31s - loss: 0.2215 - categorical_accuracy: 0.9320
43456/60000 [====================>.........] - ETA: 31s - loss: 0.2214 - categorical_accuracy: 0.9321
43488/60000 [====================>.........] - ETA: 31s - loss: 0.2213 - categorical_accuracy: 0.9321
43520/60000 [====================>.........] - ETA: 31s - loss: 0.2211 - categorical_accuracy: 0.9322
43552/60000 [====================>.........] - ETA: 31s - loss: 0.2211 - categorical_accuracy: 0.9322
43584/60000 [====================>.........] - ETA: 31s - loss: 0.2211 - categorical_accuracy: 0.9322
43616/60000 [====================>.........] - ETA: 31s - loss: 0.2209 - categorical_accuracy: 0.9322
43648/60000 [====================>.........] - ETA: 31s - loss: 0.2209 - categorical_accuracy: 0.9322
43680/60000 [====================>.........] - ETA: 31s - loss: 0.2208 - categorical_accuracy: 0.9323
43712/60000 [====================>.........] - ETA: 30s - loss: 0.2206 - categorical_accuracy: 0.9323
43744/60000 [====================>.........] - ETA: 30s - loss: 0.2206 - categorical_accuracy: 0.9323
43776/60000 [====================>.........] - ETA: 30s - loss: 0.2207 - categorical_accuracy: 0.9323
43808/60000 [====================>.........] - ETA: 30s - loss: 0.2206 - categorical_accuracy: 0.9323
43840/60000 [====================>.........] - ETA: 30s - loss: 0.2205 - categorical_accuracy: 0.9323
43872/60000 [====================>.........] - ETA: 30s - loss: 0.2203 - categorical_accuracy: 0.9324
43904/60000 [====================>.........] - ETA: 30s - loss: 0.2203 - categorical_accuracy: 0.9324
43936/60000 [====================>.........] - ETA: 30s - loss: 0.2202 - categorical_accuracy: 0.9324
43968/60000 [====================>.........] - ETA: 30s - loss: 0.2201 - categorical_accuracy: 0.9324
44000/60000 [=====================>........] - ETA: 30s - loss: 0.2199 - categorical_accuracy: 0.9325
44032/60000 [=====================>........] - ETA: 30s - loss: 0.2198 - categorical_accuracy: 0.9325
44064/60000 [=====================>........] - ETA: 30s - loss: 0.2196 - categorical_accuracy: 0.9326
44096/60000 [=====================>........] - ETA: 30s - loss: 0.2196 - categorical_accuracy: 0.9326
44128/60000 [=====================>........] - ETA: 30s - loss: 0.2195 - categorical_accuracy: 0.9326
44160/60000 [=====================>........] - ETA: 30s - loss: 0.2194 - categorical_accuracy: 0.9326
44192/60000 [=====================>........] - ETA: 30s - loss: 0.2194 - categorical_accuracy: 0.9326
44224/60000 [=====================>........] - ETA: 29s - loss: 0.2195 - categorical_accuracy: 0.9326
44256/60000 [=====================>........] - ETA: 29s - loss: 0.2193 - categorical_accuracy: 0.9327
44288/60000 [=====================>........] - ETA: 29s - loss: 0.2194 - categorical_accuracy: 0.9327
44320/60000 [=====================>........] - ETA: 29s - loss: 0.2193 - categorical_accuracy: 0.9327
44352/60000 [=====================>........] - ETA: 29s - loss: 0.2192 - categorical_accuracy: 0.9328
44384/60000 [=====================>........] - ETA: 29s - loss: 0.2191 - categorical_accuracy: 0.9328
44416/60000 [=====================>........] - ETA: 29s - loss: 0.2192 - categorical_accuracy: 0.9328
44448/60000 [=====================>........] - ETA: 29s - loss: 0.2193 - categorical_accuracy: 0.9328
44480/60000 [=====================>........] - ETA: 29s - loss: 0.2192 - categorical_accuracy: 0.9328
44512/60000 [=====================>........] - ETA: 29s - loss: 0.2191 - categorical_accuracy: 0.9328
44544/60000 [=====================>........] - ETA: 29s - loss: 0.2190 - categorical_accuracy: 0.9329
44576/60000 [=====================>........] - ETA: 29s - loss: 0.2188 - categorical_accuracy: 0.9329
44608/60000 [=====================>........] - ETA: 29s - loss: 0.2188 - categorical_accuracy: 0.9329
44640/60000 [=====================>........] - ETA: 29s - loss: 0.2188 - categorical_accuracy: 0.9329
44672/60000 [=====================>........] - ETA: 29s - loss: 0.2187 - categorical_accuracy: 0.9330
44704/60000 [=====================>........] - ETA: 29s - loss: 0.2186 - categorical_accuracy: 0.9330
44736/60000 [=====================>........] - ETA: 29s - loss: 0.2188 - categorical_accuracy: 0.9330
44768/60000 [=====================>........] - ETA: 28s - loss: 0.2187 - categorical_accuracy: 0.9330
44800/60000 [=====================>........] - ETA: 28s - loss: 0.2187 - categorical_accuracy: 0.9330
44832/60000 [=====================>........] - ETA: 28s - loss: 0.2186 - categorical_accuracy: 0.9331
44864/60000 [=====================>........] - ETA: 28s - loss: 0.2185 - categorical_accuracy: 0.9331
44896/60000 [=====================>........] - ETA: 28s - loss: 0.2184 - categorical_accuracy: 0.9331
44928/60000 [=====================>........] - ETA: 28s - loss: 0.2184 - categorical_accuracy: 0.9331
44960/60000 [=====================>........] - ETA: 28s - loss: 0.2183 - categorical_accuracy: 0.9332
44992/60000 [=====================>........] - ETA: 28s - loss: 0.2181 - categorical_accuracy: 0.9332
45024/60000 [=====================>........] - ETA: 28s - loss: 0.2181 - categorical_accuracy: 0.9332
45056/60000 [=====================>........] - ETA: 28s - loss: 0.2180 - categorical_accuracy: 0.9332
45088/60000 [=====================>........] - ETA: 28s - loss: 0.2181 - categorical_accuracy: 0.9332
45120/60000 [=====================>........] - ETA: 28s - loss: 0.2180 - categorical_accuracy: 0.9333
45152/60000 [=====================>........] - ETA: 28s - loss: 0.2179 - categorical_accuracy: 0.9333
45184/60000 [=====================>........] - ETA: 28s - loss: 0.2177 - categorical_accuracy: 0.9333
45216/60000 [=====================>........] - ETA: 28s - loss: 0.2176 - categorical_accuracy: 0.9334
45248/60000 [=====================>........] - ETA: 28s - loss: 0.2175 - categorical_accuracy: 0.9334
45280/60000 [=====================>........] - ETA: 27s - loss: 0.2174 - categorical_accuracy: 0.9335
45312/60000 [=====================>........] - ETA: 27s - loss: 0.2172 - categorical_accuracy: 0.9335
45344/60000 [=====================>........] - ETA: 27s - loss: 0.2172 - categorical_accuracy: 0.9336
45376/60000 [=====================>........] - ETA: 27s - loss: 0.2171 - categorical_accuracy: 0.9336
45408/60000 [=====================>........] - ETA: 27s - loss: 0.2170 - categorical_accuracy: 0.9336
45440/60000 [=====================>........] - ETA: 27s - loss: 0.2169 - categorical_accuracy: 0.9336
45472/60000 [=====================>........] - ETA: 27s - loss: 0.2168 - categorical_accuracy: 0.9337
45504/60000 [=====================>........] - ETA: 27s - loss: 0.2166 - categorical_accuracy: 0.9337
45536/60000 [=====================>........] - ETA: 27s - loss: 0.2165 - categorical_accuracy: 0.9337
45568/60000 [=====================>........] - ETA: 27s - loss: 0.2164 - categorical_accuracy: 0.9338
45600/60000 [=====================>........] - ETA: 27s - loss: 0.2164 - categorical_accuracy: 0.9338
45632/60000 [=====================>........] - ETA: 27s - loss: 0.2163 - categorical_accuracy: 0.9339
45664/60000 [=====================>........] - ETA: 27s - loss: 0.2162 - categorical_accuracy: 0.9339
45696/60000 [=====================>........] - ETA: 27s - loss: 0.2161 - categorical_accuracy: 0.9339
45728/60000 [=====================>........] - ETA: 27s - loss: 0.2164 - categorical_accuracy: 0.9339
45760/60000 [=====================>........] - ETA: 27s - loss: 0.2163 - categorical_accuracy: 0.9339
45792/60000 [=====================>........] - ETA: 27s - loss: 0.2162 - categorical_accuracy: 0.9340
45824/60000 [=====================>........] - ETA: 26s - loss: 0.2161 - categorical_accuracy: 0.9340
45856/60000 [=====================>........] - ETA: 26s - loss: 0.2160 - categorical_accuracy: 0.9340
45888/60000 [=====================>........] - ETA: 26s - loss: 0.2159 - categorical_accuracy: 0.9340
45920/60000 [=====================>........] - ETA: 26s - loss: 0.2158 - categorical_accuracy: 0.9341
45952/60000 [=====================>........] - ETA: 26s - loss: 0.2157 - categorical_accuracy: 0.9341
45984/60000 [=====================>........] - ETA: 26s - loss: 0.2156 - categorical_accuracy: 0.9342
46016/60000 [======================>.......] - ETA: 26s - loss: 0.2156 - categorical_accuracy: 0.9342
46048/60000 [======================>.......] - ETA: 26s - loss: 0.2155 - categorical_accuracy: 0.9342
46080/60000 [======================>.......] - ETA: 26s - loss: 0.2155 - categorical_accuracy: 0.9342
46112/60000 [======================>.......] - ETA: 26s - loss: 0.2153 - categorical_accuracy: 0.9342
46144/60000 [======================>.......] - ETA: 26s - loss: 0.2152 - categorical_accuracy: 0.9343
46176/60000 [======================>.......] - ETA: 26s - loss: 0.2151 - categorical_accuracy: 0.9343
46208/60000 [======================>.......] - ETA: 26s - loss: 0.2150 - categorical_accuracy: 0.9343
46240/60000 [======================>.......] - ETA: 26s - loss: 0.2149 - categorical_accuracy: 0.9344
46272/60000 [======================>.......] - ETA: 26s - loss: 0.2148 - categorical_accuracy: 0.9344
46304/60000 [======================>.......] - ETA: 26s - loss: 0.2147 - categorical_accuracy: 0.9344
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2147 - categorical_accuracy: 0.9345
46368/60000 [======================>.......] - ETA: 25s - loss: 0.2146 - categorical_accuracy: 0.9345
46400/60000 [======================>.......] - ETA: 25s - loss: 0.2145 - categorical_accuracy: 0.9345
46432/60000 [======================>.......] - ETA: 25s - loss: 0.2143 - categorical_accuracy: 0.9345
46464/60000 [======================>.......] - ETA: 25s - loss: 0.2142 - categorical_accuracy: 0.9346
46496/60000 [======================>.......] - ETA: 25s - loss: 0.2141 - categorical_accuracy: 0.9346
46528/60000 [======================>.......] - ETA: 25s - loss: 0.2140 - categorical_accuracy: 0.9347
46560/60000 [======================>.......] - ETA: 25s - loss: 0.2139 - categorical_accuracy: 0.9347
46592/60000 [======================>.......] - ETA: 25s - loss: 0.2138 - categorical_accuracy: 0.9347
46624/60000 [======================>.......] - ETA: 25s - loss: 0.2137 - categorical_accuracy: 0.9348
46656/60000 [======================>.......] - ETA: 25s - loss: 0.2136 - categorical_accuracy: 0.9348
46688/60000 [======================>.......] - ETA: 25s - loss: 0.2134 - categorical_accuracy: 0.9348
46720/60000 [======================>.......] - ETA: 25s - loss: 0.2133 - categorical_accuracy: 0.9349
46752/60000 [======================>.......] - ETA: 25s - loss: 0.2133 - categorical_accuracy: 0.9349
46784/60000 [======================>.......] - ETA: 25s - loss: 0.2133 - categorical_accuracy: 0.9349
46816/60000 [======================>.......] - ETA: 25s - loss: 0.2132 - categorical_accuracy: 0.9349
46848/60000 [======================>.......] - ETA: 25s - loss: 0.2131 - categorical_accuracy: 0.9349
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2131 - categorical_accuracy: 0.9350
46912/60000 [======================>.......] - ETA: 24s - loss: 0.2129 - categorical_accuracy: 0.9350
46944/60000 [======================>.......] - ETA: 24s - loss: 0.2128 - categorical_accuracy: 0.9351
46976/60000 [======================>.......] - ETA: 24s - loss: 0.2128 - categorical_accuracy: 0.9351
47008/60000 [======================>.......] - ETA: 24s - loss: 0.2127 - categorical_accuracy: 0.9351
47040/60000 [======================>.......] - ETA: 24s - loss: 0.2126 - categorical_accuracy: 0.9351
47072/60000 [======================>.......] - ETA: 24s - loss: 0.2127 - categorical_accuracy: 0.9351
47104/60000 [======================>.......] - ETA: 24s - loss: 0.2127 - categorical_accuracy: 0.9351
47136/60000 [======================>.......] - ETA: 24s - loss: 0.2125 - categorical_accuracy: 0.9351
47168/60000 [======================>.......] - ETA: 24s - loss: 0.2125 - categorical_accuracy: 0.9351
47200/60000 [======================>.......] - ETA: 24s - loss: 0.2124 - categorical_accuracy: 0.9352
47232/60000 [======================>.......] - ETA: 24s - loss: 0.2124 - categorical_accuracy: 0.9352
47264/60000 [======================>.......] - ETA: 24s - loss: 0.2123 - categorical_accuracy: 0.9352
47296/60000 [======================>.......] - ETA: 24s - loss: 0.2123 - categorical_accuracy: 0.9352
47328/60000 [======================>.......] - ETA: 24s - loss: 0.2125 - categorical_accuracy: 0.9352
47360/60000 [======================>.......] - ETA: 24s - loss: 0.2124 - categorical_accuracy: 0.9352
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2124 - categorical_accuracy: 0.9352
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2126 - categorical_accuracy: 0.9352
47456/60000 [======================>.......] - ETA: 23s - loss: 0.2126 - categorical_accuracy: 0.9352
47488/60000 [======================>.......] - ETA: 23s - loss: 0.2125 - categorical_accuracy: 0.9352
47520/60000 [======================>.......] - ETA: 23s - loss: 0.2123 - categorical_accuracy: 0.9353
47552/60000 [======================>.......] - ETA: 23s - loss: 0.2123 - categorical_accuracy: 0.9352
47584/60000 [======================>.......] - ETA: 23s - loss: 0.2123 - categorical_accuracy: 0.9353
47616/60000 [======================>.......] - ETA: 23s - loss: 0.2122 - categorical_accuracy: 0.9353
47648/60000 [======================>.......] - ETA: 23s - loss: 0.2121 - categorical_accuracy: 0.9353
47680/60000 [======================>.......] - ETA: 23s - loss: 0.2120 - categorical_accuracy: 0.9353
47712/60000 [======================>.......] - ETA: 23s - loss: 0.2119 - categorical_accuracy: 0.9353
47744/60000 [======================>.......] - ETA: 23s - loss: 0.2118 - categorical_accuracy: 0.9354
47776/60000 [======================>.......] - ETA: 23s - loss: 0.2118 - categorical_accuracy: 0.9354
47808/60000 [======================>.......] - ETA: 23s - loss: 0.2117 - categorical_accuracy: 0.9354
47840/60000 [======================>.......] - ETA: 23s - loss: 0.2116 - categorical_accuracy: 0.9355
47872/60000 [======================>.......] - ETA: 23s - loss: 0.2115 - categorical_accuracy: 0.9355
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2115 - categorical_accuracy: 0.9355
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9354
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9354
48000/60000 [=======================>......] - ETA: 22s - loss: 0.2113 - categorical_accuracy: 0.9354
48032/60000 [=======================>......] - ETA: 22s - loss: 0.2112 - categorical_accuracy: 0.9355
48064/60000 [=======================>......] - ETA: 22s - loss: 0.2111 - categorical_accuracy: 0.9355
48096/60000 [=======================>......] - ETA: 22s - loss: 0.2109 - categorical_accuracy: 0.9355
48128/60000 [=======================>......] - ETA: 22s - loss: 0.2110 - categorical_accuracy: 0.9355
48160/60000 [=======================>......] - ETA: 22s - loss: 0.2109 - categorical_accuracy: 0.9355
48192/60000 [=======================>......] - ETA: 22s - loss: 0.2108 - categorical_accuracy: 0.9355
48224/60000 [=======================>......] - ETA: 22s - loss: 0.2108 - categorical_accuracy: 0.9356
48256/60000 [=======================>......] - ETA: 22s - loss: 0.2109 - categorical_accuracy: 0.9356
48288/60000 [=======================>......] - ETA: 22s - loss: 0.2108 - categorical_accuracy: 0.9356
48320/60000 [=======================>......] - ETA: 22s - loss: 0.2107 - categorical_accuracy: 0.9356
48352/60000 [=======================>......] - ETA: 22s - loss: 0.2105 - categorical_accuracy: 0.9356
48384/60000 [=======================>......] - ETA: 22s - loss: 0.2105 - categorical_accuracy: 0.9357
48416/60000 [=======================>......] - ETA: 22s - loss: 0.2104 - categorical_accuracy: 0.9357
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2104 - categorical_accuracy: 0.9357
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2103 - categorical_accuracy: 0.9357
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2102 - categorical_accuracy: 0.9357
48544/60000 [=======================>......] - ETA: 21s - loss: 0.2102 - categorical_accuracy: 0.9358
48576/60000 [=======================>......] - ETA: 21s - loss: 0.2101 - categorical_accuracy: 0.9358
48608/60000 [=======================>......] - ETA: 21s - loss: 0.2100 - categorical_accuracy: 0.9359
48640/60000 [=======================>......] - ETA: 21s - loss: 0.2099 - categorical_accuracy: 0.9359
48672/60000 [=======================>......] - ETA: 21s - loss: 0.2098 - categorical_accuracy: 0.9359
48704/60000 [=======================>......] - ETA: 21s - loss: 0.2097 - categorical_accuracy: 0.9359
48736/60000 [=======================>......] - ETA: 21s - loss: 0.2097 - categorical_accuracy: 0.9359
48768/60000 [=======================>......] - ETA: 21s - loss: 0.2096 - categorical_accuracy: 0.9359
48800/60000 [=======================>......] - ETA: 21s - loss: 0.2095 - categorical_accuracy: 0.9359
48832/60000 [=======================>......] - ETA: 21s - loss: 0.2095 - categorical_accuracy: 0.9360
48864/60000 [=======================>......] - ETA: 21s - loss: 0.2094 - categorical_accuracy: 0.9360
48896/60000 [=======================>......] - ETA: 21s - loss: 0.2093 - categorical_accuracy: 0.9360
48928/60000 [=======================>......] - ETA: 21s - loss: 0.2092 - categorical_accuracy: 0.9361
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2092 - categorical_accuracy: 0.9361
48992/60000 [=======================>......] - ETA: 20s - loss: 0.2092 - categorical_accuracy: 0.9361
49024/60000 [=======================>......] - ETA: 20s - loss: 0.2092 - categorical_accuracy: 0.9361
49056/60000 [=======================>......] - ETA: 20s - loss: 0.2090 - categorical_accuracy: 0.9361
49088/60000 [=======================>......] - ETA: 20s - loss: 0.2089 - categorical_accuracy: 0.9362
49120/60000 [=======================>......] - ETA: 20s - loss: 0.2088 - categorical_accuracy: 0.9362
49152/60000 [=======================>......] - ETA: 20s - loss: 0.2087 - categorical_accuracy: 0.9362
49184/60000 [=======================>......] - ETA: 20s - loss: 0.2086 - categorical_accuracy: 0.9362
49216/60000 [=======================>......] - ETA: 20s - loss: 0.2085 - categorical_accuracy: 0.9363
49248/60000 [=======================>......] - ETA: 20s - loss: 0.2084 - categorical_accuracy: 0.9363
49280/60000 [=======================>......] - ETA: 20s - loss: 0.2083 - categorical_accuracy: 0.9363
49312/60000 [=======================>......] - ETA: 20s - loss: 0.2081 - categorical_accuracy: 0.9364
49344/60000 [=======================>......] - ETA: 20s - loss: 0.2082 - categorical_accuracy: 0.9364
49376/60000 [=======================>......] - ETA: 20s - loss: 0.2081 - categorical_accuracy: 0.9364
49408/60000 [=======================>......] - ETA: 20s - loss: 0.2080 - categorical_accuracy: 0.9364
49440/60000 [=======================>......] - ETA: 20s - loss: 0.2080 - categorical_accuracy: 0.9364
49472/60000 [=======================>......] - ETA: 20s - loss: 0.2079 - categorical_accuracy: 0.9365
49504/60000 [=======================>......] - ETA: 19s - loss: 0.2079 - categorical_accuracy: 0.9364
49536/60000 [=======================>......] - ETA: 19s - loss: 0.2078 - categorical_accuracy: 0.9365
49568/60000 [=======================>......] - ETA: 19s - loss: 0.2078 - categorical_accuracy: 0.9365
49600/60000 [=======================>......] - ETA: 19s - loss: 0.2077 - categorical_accuracy: 0.9365
49632/60000 [=======================>......] - ETA: 19s - loss: 0.2076 - categorical_accuracy: 0.9365
49664/60000 [=======================>......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9365
49696/60000 [=======================>......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9366
49728/60000 [=======================>......] - ETA: 19s - loss: 0.2074 - categorical_accuracy: 0.9366
49760/60000 [=======================>......] - ETA: 19s - loss: 0.2074 - categorical_accuracy: 0.9366
49792/60000 [=======================>......] - ETA: 19s - loss: 0.2072 - categorical_accuracy: 0.9366
49824/60000 [=======================>......] - ETA: 19s - loss: 0.2071 - categorical_accuracy: 0.9366
49856/60000 [=======================>......] - ETA: 19s - loss: 0.2070 - categorical_accuracy: 0.9367
49888/60000 [=======================>......] - ETA: 19s - loss: 0.2070 - categorical_accuracy: 0.9367
49920/60000 [=======================>......] - ETA: 19s - loss: 0.2069 - categorical_accuracy: 0.9367
49952/60000 [=======================>......] - ETA: 19s - loss: 0.2068 - categorical_accuracy: 0.9367
49984/60000 [=======================>......] - ETA: 19s - loss: 0.2068 - categorical_accuracy: 0.9367
50016/60000 [========================>.....] - ETA: 18s - loss: 0.2068 - categorical_accuracy: 0.9367
50048/60000 [========================>.....] - ETA: 18s - loss: 0.2068 - categorical_accuracy: 0.9368
50080/60000 [========================>.....] - ETA: 18s - loss: 0.2067 - categorical_accuracy: 0.9368
50112/60000 [========================>.....] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9368
50144/60000 [========================>.....] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9368
50176/60000 [========================>.....] - ETA: 18s - loss: 0.2065 - categorical_accuracy: 0.9368
50208/60000 [========================>.....] - ETA: 18s - loss: 0.2064 - categorical_accuracy: 0.9369
50240/60000 [========================>.....] - ETA: 18s - loss: 0.2063 - categorical_accuracy: 0.9369
50272/60000 [========================>.....] - ETA: 18s - loss: 0.2063 - categorical_accuracy: 0.9369
50304/60000 [========================>.....] - ETA: 18s - loss: 0.2062 - categorical_accuracy: 0.9369
50336/60000 [========================>.....] - ETA: 18s - loss: 0.2062 - categorical_accuracy: 0.9369
50368/60000 [========================>.....] - ETA: 18s - loss: 0.2061 - categorical_accuracy: 0.9369
50400/60000 [========================>.....] - ETA: 18s - loss: 0.2061 - categorical_accuracy: 0.9369
50432/60000 [========================>.....] - ETA: 18s - loss: 0.2060 - categorical_accuracy: 0.9369
50464/60000 [========================>.....] - ETA: 18s - loss: 0.2060 - categorical_accuracy: 0.9369
50496/60000 [========================>.....] - ETA: 18s - loss: 0.2059 - categorical_accuracy: 0.9370
50528/60000 [========================>.....] - ETA: 18s - loss: 0.2057 - categorical_accuracy: 0.9370
50560/60000 [========================>.....] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9370
50592/60000 [========================>.....] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9370
50624/60000 [========================>.....] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9370
50656/60000 [========================>.....] - ETA: 17s - loss: 0.2055 - categorical_accuracy: 0.9371
50688/60000 [========================>.....] - ETA: 17s - loss: 0.2054 - categorical_accuracy: 0.9371
50720/60000 [========================>.....] - ETA: 17s - loss: 0.2053 - categorical_accuracy: 0.9371
50752/60000 [========================>.....] - ETA: 17s - loss: 0.2052 - categorical_accuracy: 0.9371
50784/60000 [========================>.....] - ETA: 17s - loss: 0.2051 - categorical_accuracy: 0.9372
50816/60000 [========================>.....] - ETA: 17s - loss: 0.2051 - categorical_accuracy: 0.9372
50848/60000 [========================>.....] - ETA: 17s - loss: 0.2050 - categorical_accuracy: 0.9372
50880/60000 [========================>.....] - ETA: 17s - loss: 0.2049 - categorical_accuracy: 0.9372
50912/60000 [========================>.....] - ETA: 17s - loss: 0.2049 - categorical_accuracy: 0.9372
50944/60000 [========================>.....] - ETA: 17s - loss: 0.2049 - categorical_accuracy: 0.9373
50976/60000 [========================>.....] - ETA: 17s - loss: 0.2048 - categorical_accuracy: 0.9373
51008/60000 [========================>.....] - ETA: 17s - loss: 0.2048 - categorical_accuracy: 0.9373
51040/60000 [========================>.....] - ETA: 17s - loss: 0.2047 - categorical_accuracy: 0.9373
51072/60000 [========================>.....] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9373
51104/60000 [========================>.....] - ETA: 16s - loss: 0.2046 - categorical_accuracy: 0.9374
51136/60000 [========================>.....] - ETA: 16s - loss: 0.2045 - categorical_accuracy: 0.9374
51168/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9374
51200/60000 [========================>.....] - ETA: 16s - loss: 0.2043 - categorical_accuracy: 0.9374
51232/60000 [========================>.....] - ETA: 16s - loss: 0.2042 - categorical_accuracy: 0.9375
51264/60000 [========================>.....] - ETA: 16s - loss: 0.2042 - categorical_accuracy: 0.9375
51296/60000 [========================>.....] - ETA: 16s - loss: 0.2041 - categorical_accuracy: 0.9375
51328/60000 [========================>.....] - ETA: 16s - loss: 0.2040 - categorical_accuracy: 0.9375
51360/60000 [========================>.....] - ETA: 16s - loss: 0.2039 - categorical_accuracy: 0.9376
51392/60000 [========================>.....] - ETA: 16s - loss: 0.2038 - categorical_accuracy: 0.9376
51424/60000 [========================>.....] - ETA: 16s - loss: 0.2037 - categorical_accuracy: 0.9376
51456/60000 [========================>.....] - ETA: 16s - loss: 0.2037 - categorical_accuracy: 0.9376
51488/60000 [========================>.....] - ETA: 16s - loss: 0.2036 - categorical_accuracy: 0.9376
51520/60000 [========================>.....] - ETA: 16s - loss: 0.2036 - categorical_accuracy: 0.9376
51552/60000 [========================>.....] - ETA: 16s - loss: 0.2036 - categorical_accuracy: 0.9376
51584/60000 [========================>.....] - ETA: 16s - loss: 0.2036 - categorical_accuracy: 0.9376
51616/60000 [========================>.....] - ETA: 15s - loss: 0.2035 - categorical_accuracy: 0.9377
51648/60000 [========================>.....] - ETA: 15s - loss: 0.2035 - categorical_accuracy: 0.9377
51680/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9377
51712/60000 [========================>.....] - ETA: 15s - loss: 0.2036 - categorical_accuracy: 0.9377
51744/60000 [========================>.....] - ETA: 15s - loss: 0.2035 - categorical_accuracy: 0.9377
51776/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9377
51808/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9378
51840/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9378
51872/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9378
51904/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9378
51936/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9378
51968/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9378
52000/60000 [=========================>....] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9378
52032/60000 [=========================>....] - ETA: 15s - loss: 0.2030 - categorical_accuracy: 0.9379
52064/60000 [=========================>....] - ETA: 15s - loss: 0.2029 - categorical_accuracy: 0.9379
52096/60000 [=========================>....] - ETA: 15s - loss: 0.2028 - categorical_accuracy: 0.9380
52128/60000 [=========================>....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9380
52160/60000 [=========================>....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9380
52192/60000 [=========================>....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9381
52224/60000 [=========================>....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9381
52256/60000 [=========================>....] - ETA: 14s - loss: 0.2023 - categorical_accuracy: 0.9381
52288/60000 [=========================>....] - ETA: 14s - loss: 0.2022 - categorical_accuracy: 0.9381
52320/60000 [=========================>....] - ETA: 14s - loss: 0.2022 - categorical_accuracy: 0.9381
52352/60000 [=========================>....] - ETA: 14s - loss: 0.2021 - categorical_accuracy: 0.9381
52384/60000 [=========================>....] - ETA: 14s - loss: 0.2020 - categorical_accuracy: 0.9382
52416/60000 [=========================>....] - ETA: 14s - loss: 0.2019 - categorical_accuracy: 0.9382
52448/60000 [=========================>....] - ETA: 14s - loss: 0.2018 - categorical_accuracy: 0.9382
52480/60000 [=========================>....] - ETA: 14s - loss: 0.2018 - categorical_accuracy: 0.9382
52512/60000 [=========================>....] - ETA: 14s - loss: 0.2017 - categorical_accuracy: 0.9383
52544/60000 [=========================>....] - ETA: 14s - loss: 0.2016 - categorical_accuracy: 0.9383
52576/60000 [=========================>....] - ETA: 14s - loss: 0.2015 - categorical_accuracy: 0.9383
52608/60000 [=========================>....] - ETA: 14s - loss: 0.2014 - categorical_accuracy: 0.9384
52640/60000 [=========================>....] - ETA: 13s - loss: 0.2013 - categorical_accuracy: 0.9384
52672/60000 [=========================>....] - ETA: 13s - loss: 0.2012 - categorical_accuracy: 0.9384
52704/60000 [=========================>....] - ETA: 13s - loss: 0.2011 - categorical_accuracy: 0.9385
52736/60000 [=========================>....] - ETA: 13s - loss: 0.2010 - categorical_accuracy: 0.9385
52768/60000 [=========================>....] - ETA: 13s - loss: 0.2009 - categorical_accuracy: 0.9385
52800/60000 [=========================>....] - ETA: 13s - loss: 0.2010 - categorical_accuracy: 0.9385
52832/60000 [=========================>....] - ETA: 13s - loss: 0.2008 - categorical_accuracy: 0.9386
52864/60000 [=========================>....] - ETA: 13s - loss: 0.2008 - categorical_accuracy: 0.9386
52896/60000 [=========================>....] - ETA: 13s - loss: 0.2008 - categorical_accuracy: 0.9386
52928/60000 [=========================>....] - ETA: 13s - loss: 0.2007 - categorical_accuracy: 0.9386
52960/60000 [=========================>....] - ETA: 13s - loss: 0.2006 - categorical_accuracy: 0.9387
52992/60000 [=========================>....] - ETA: 13s - loss: 0.2005 - categorical_accuracy: 0.9387
53024/60000 [=========================>....] - ETA: 13s - loss: 0.2007 - categorical_accuracy: 0.9387
53056/60000 [=========================>....] - ETA: 13s - loss: 0.2006 - categorical_accuracy: 0.9387
53088/60000 [=========================>....] - ETA: 13s - loss: 0.2005 - categorical_accuracy: 0.9387
53120/60000 [=========================>....] - ETA: 13s - loss: 0.2004 - categorical_accuracy: 0.9388
53152/60000 [=========================>....] - ETA: 13s - loss: 0.2004 - categorical_accuracy: 0.9388
53184/60000 [=========================>....] - ETA: 12s - loss: 0.2003 - categorical_accuracy: 0.9388
53216/60000 [=========================>....] - ETA: 12s - loss: 0.2002 - categorical_accuracy: 0.9388
53248/60000 [=========================>....] - ETA: 12s - loss: 0.2001 - categorical_accuracy: 0.9389
53280/60000 [=========================>....] - ETA: 12s - loss: 0.2000 - categorical_accuracy: 0.9389
53312/60000 [=========================>....] - ETA: 12s - loss: 0.2001 - categorical_accuracy: 0.9389
53344/60000 [=========================>....] - ETA: 12s - loss: 0.2001 - categorical_accuracy: 0.9389
53376/60000 [=========================>....] - ETA: 12s - loss: 0.2001 - categorical_accuracy: 0.9388
53408/60000 [=========================>....] - ETA: 12s - loss: 0.2002 - categorical_accuracy: 0.9388
53440/60000 [=========================>....] - ETA: 12s - loss: 0.2001 - categorical_accuracy: 0.9389
53472/60000 [=========================>....] - ETA: 12s - loss: 0.2001 - categorical_accuracy: 0.9389
53504/60000 [=========================>....] - ETA: 12s - loss: 0.2000 - categorical_accuracy: 0.9389
53536/60000 [=========================>....] - ETA: 12s - loss: 0.1999 - categorical_accuracy: 0.9389
53568/60000 [=========================>....] - ETA: 12s - loss: 0.1998 - categorical_accuracy: 0.9390
53600/60000 [=========================>....] - ETA: 12s - loss: 0.1997 - categorical_accuracy: 0.9390
53632/60000 [=========================>....] - ETA: 12s - loss: 0.1997 - categorical_accuracy: 0.9390
53664/60000 [=========================>....] - ETA: 12s - loss: 0.1996 - categorical_accuracy: 0.9390
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1995 - categorical_accuracy: 0.9390
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1994 - categorical_accuracy: 0.9391
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9391
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9391
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9391
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9392
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9391
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1990 - categorical_accuracy: 0.9392
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1989 - categorical_accuracy: 0.9392
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1988 - categorical_accuracy: 0.9392
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1987 - categorical_accuracy: 0.9393
54048/60000 [==========================>...] - ETA: 11s - loss: 0.1986 - categorical_accuracy: 0.9393
54080/60000 [==========================>...] - ETA: 11s - loss: 0.1985 - categorical_accuracy: 0.9393
54112/60000 [==========================>...] - ETA: 11s - loss: 0.1985 - categorical_accuracy: 0.9393
54144/60000 [==========================>...] - ETA: 11s - loss: 0.1984 - categorical_accuracy: 0.9393
54176/60000 [==========================>...] - ETA: 11s - loss: 0.1984 - categorical_accuracy: 0.9394
54208/60000 [==========================>...] - ETA: 11s - loss: 0.1982 - categorical_accuracy: 0.9394
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1981 - categorical_accuracy: 0.9394
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9395
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9395
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1978 - categorical_accuracy: 0.9395
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1978 - categorical_accuracy: 0.9395
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9396
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9396
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1978 - categorical_accuracy: 0.9396
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9396
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1976 - categorical_accuracy: 0.9396
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1976 - categorical_accuracy: 0.9396
54592/60000 [==========================>...] - ETA: 10s - loss: 0.1975 - categorical_accuracy: 0.9396
54624/60000 [==========================>...] - ETA: 10s - loss: 0.1974 - categorical_accuracy: 0.9396
54656/60000 [==========================>...] - ETA: 10s - loss: 0.1974 - categorical_accuracy: 0.9396
54688/60000 [==========================>...] - ETA: 10s - loss: 0.1973 - categorical_accuracy: 0.9397
54720/60000 [==========================>...] - ETA: 10s - loss: 0.1972 - categorical_accuracy: 0.9397
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1971 - categorical_accuracy: 0.9397 
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1970 - categorical_accuracy: 0.9397
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1970 - categorical_accuracy: 0.9397
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1969 - categorical_accuracy: 0.9397
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1969 - categorical_accuracy: 0.9398
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1968 - categorical_accuracy: 0.9398
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1968 - categorical_accuracy: 0.9398
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1967 - categorical_accuracy: 0.9398
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1966 - categorical_accuracy: 0.9399
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1965 - categorical_accuracy: 0.9399
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1964 - categorical_accuracy: 0.9399
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1964 - categorical_accuracy: 0.9399
55136/60000 [==========================>...] - ETA: 9s - loss: 0.1963 - categorical_accuracy: 0.9399
55168/60000 [==========================>...] - ETA: 9s - loss: 0.1963 - categorical_accuracy: 0.9399
55200/60000 [==========================>...] - ETA: 9s - loss: 0.1963 - categorical_accuracy: 0.9399
55232/60000 [==========================>...] - ETA: 9s - loss: 0.1962 - categorical_accuracy: 0.9399
55264/60000 [==========================>...] - ETA: 9s - loss: 0.1962 - categorical_accuracy: 0.9400
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9400
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9400
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9400
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1959 - categorical_accuracy: 0.9400
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1958 - categorical_accuracy: 0.9401
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1958 - categorical_accuracy: 0.9401
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1957 - categorical_accuracy: 0.9401
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1956 - categorical_accuracy: 0.9401
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1956 - categorical_accuracy: 0.9401
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1955 - categorical_accuracy: 0.9401
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9401
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9401
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1953 - categorical_accuracy: 0.9402
55712/60000 [==========================>...] - ETA: 8s - loss: 0.1952 - categorical_accuracy: 0.9402
55744/60000 [==========================>...] - ETA: 8s - loss: 0.1951 - categorical_accuracy: 0.9402
55776/60000 [==========================>...] - ETA: 8s - loss: 0.1950 - categorical_accuracy: 0.9403
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9403
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9403
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9403
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1948 - categorical_accuracy: 0.9403
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9404
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1946 - categorical_accuracy: 0.9404
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1946 - categorical_accuracy: 0.9404
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1945 - categorical_accuracy: 0.9404
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1944 - categorical_accuracy: 0.9405
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1943 - categorical_accuracy: 0.9405
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1943 - categorical_accuracy: 0.9405
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1942 - categorical_accuracy: 0.9405
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1941 - categorical_accuracy: 0.9405
56224/60000 [===========================>..] - ETA: 7s - loss: 0.1940 - categorical_accuracy: 0.9406
56256/60000 [===========================>..] - ETA: 7s - loss: 0.1941 - categorical_accuracy: 0.9406
56288/60000 [===========================>..] - ETA: 7s - loss: 0.1940 - categorical_accuracy: 0.9406
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9406
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9406
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9407
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9407
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9407
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1936 - categorical_accuracy: 0.9407
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9408
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9407
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1934 - categorical_accuracy: 0.9408
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1933 - categorical_accuracy: 0.9408
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1933 - categorical_accuracy: 0.9408
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1932 - categorical_accuracy: 0.9408
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1932 - categorical_accuracy: 0.9409
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1931 - categorical_accuracy: 0.9409
56768/60000 [===========================>..] - ETA: 6s - loss: 0.1930 - categorical_accuracy: 0.9409
56800/60000 [===========================>..] - ETA: 6s - loss: 0.1930 - categorical_accuracy: 0.9409
56832/60000 [===========================>..] - ETA: 6s - loss: 0.1929 - categorical_accuracy: 0.9409
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1928 - categorical_accuracy: 0.9410
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1928 - categorical_accuracy: 0.9410
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1927 - categorical_accuracy: 0.9410
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1926 - categorical_accuracy: 0.9410
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1925 - categorical_accuracy: 0.9411
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1925 - categorical_accuracy: 0.9411
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1924 - categorical_accuracy: 0.9411
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1924 - categorical_accuracy: 0.9411
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1923 - categorical_accuracy: 0.9412
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1923 - categorical_accuracy: 0.9412
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1922 - categorical_accuracy: 0.9412
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1922 - categorical_accuracy: 0.9412
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1922 - categorical_accuracy: 0.9412
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1921 - categorical_accuracy: 0.9413
57312/60000 [===========================>..] - ETA: 5s - loss: 0.1920 - categorical_accuracy: 0.9413
57344/60000 [===========================>..] - ETA: 5s - loss: 0.1919 - categorical_accuracy: 0.9413
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1918 - categorical_accuracy: 0.9414
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1918 - categorical_accuracy: 0.9414
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1917 - categorical_accuracy: 0.9414
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1916 - categorical_accuracy: 0.9414
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1915 - categorical_accuracy: 0.9414
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1914 - categorical_accuracy: 0.9415
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1914 - categorical_accuracy: 0.9415
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1913 - categorical_accuracy: 0.9415
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1912 - categorical_accuracy: 0.9415
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1911 - categorical_accuracy: 0.9416
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1910 - categorical_accuracy: 0.9416
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1910 - categorical_accuracy: 0.9416
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1909 - categorical_accuracy: 0.9416
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1909 - categorical_accuracy: 0.9416
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1908 - categorical_accuracy: 0.9417
57856/60000 [===========================>..] - ETA: 4s - loss: 0.1907 - categorical_accuracy: 0.9417
57888/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9417
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1906 - categorical_accuracy: 0.9417
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1905 - categorical_accuracy: 0.9417
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1904 - categorical_accuracy: 0.9418
58016/60000 [============================>.] - ETA: 3s - loss: 0.1903 - categorical_accuracy: 0.9418
58048/60000 [============================>.] - ETA: 3s - loss: 0.1902 - categorical_accuracy: 0.9418
58080/60000 [============================>.] - ETA: 3s - loss: 0.1902 - categorical_accuracy: 0.9418
58112/60000 [============================>.] - ETA: 3s - loss: 0.1902 - categorical_accuracy: 0.9419
58144/60000 [============================>.] - ETA: 3s - loss: 0.1901 - categorical_accuracy: 0.9419
58176/60000 [============================>.] - ETA: 3s - loss: 0.1900 - categorical_accuracy: 0.9419
58208/60000 [============================>.] - ETA: 3s - loss: 0.1900 - categorical_accuracy: 0.9419
58240/60000 [============================>.] - ETA: 3s - loss: 0.1899 - categorical_accuracy: 0.9420
58272/60000 [============================>.] - ETA: 3s - loss: 0.1899 - categorical_accuracy: 0.9420
58304/60000 [============================>.] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9420
58336/60000 [============================>.] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9420
58368/60000 [============================>.] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9420
58400/60000 [============================>.] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9420
58432/60000 [============================>.] - ETA: 2s - loss: 0.1897 - categorical_accuracy: 0.9420
58464/60000 [============================>.] - ETA: 2s - loss: 0.1896 - categorical_accuracy: 0.9420
58496/60000 [============================>.] - ETA: 2s - loss: 0.1895 - categorical_accuracy: 0.9421
58528/60000 [============================>.] - ETA: 2s - loss: 0.1895 - categorical_accuracy: 0.9421
58560/60000 [============================>.] - ETA: 2s - loss: 0.1895 - categorical_accuracy: 0.9421
58592/60000 [============================>.] - ETA: 2s - loss: 0.1894 - categorical_accuracy: 0.9421
58624/60000 [============================>.] - ETA: 2s - loss: 0.1894 - categorical_accuracy: 0.9422
58656/60000 [============================>.] - ETA: 2s - loss: 0.1894 - categorical_accuracy: 0.9422
58688/60000 [============================>.] - ETA: 2s - loss: 0.1893 - categorical_accuracy: 0.9422
58720/60000 [============================>.] - ETA: 2s - loss: 0.1892 - categorical_accuracy: 0.9422
58752/60000 [============================>.] - ETA: 2s - loss: 0.1891 - categorical_accuracy: 0.9422
58784/60000 [============================>.] - ETA: 2s - loss: 0.1891 - categorical_accuracy: 0.9422
58816/60000 [============================>.] - ETA: 2s - loss: 0.1890 - categorical_accuracy: 0.9422
58848/60000 [============================>.] - ETA: 2s - loss: 0.1889 - categorical_accuracy: 0.9423
58880/60000 [============================>.] - ETA: 2s - loss: 0.1889 - categorical_accuracy: 0.9423
58912/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9423
58944/60000 [============================>.] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9423
58976/60000 [============================>.] - ETA: 1s - loss: 0.1888 - categorical_accuracy: 0.9423
59008/60000 [============================>.] - ETA: 1s - loss: 0.1887 - categorical_accuracy: 0.9423
59040/60000 [============================>.] - ETA: 1s - loss: 0.1886 - categorical_accuracy: 0.9423
59072/60000 [============================>.] - ETA: 1s - loss: 0.1885 - categorical_accuracy: 0.9424
59104/60000 [============================>.] - ETA: 1s - loss: 0.1885 - categorical_accuracy: 0.9424
59136/60000 [============================>.] - ETA: 1s - loss: 0.1885 - categorical_accuracy: 0.9424
59168/60000 [============================>.] - ETA: 1s - loss: 0.1884 - categorical_accuracy: 0.9424
59200/60000 [============================>.] - ETA: 1s - loss: 0.1883 - categorical_accuracy: 0.9424
59232/60000 [============================>.] - ETA: 1s - loss: 0.1883 - categorical_accuracy: 0.9424
59264/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9424
59296/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9424
59328/60000 [============================>.] - ETA: 1s - loss: 0.1882 - categorical_accuracy: 0.9424
59360/60000 [============================>.] - ETA: 1s - loss: 0.1881 - categorical_accuracy: 0.9424
59392/60000 [============================>.] - ETA: 1s - loss: 0.1881 - categorical_accuracy: 0.9424
59424/60000 [============================>.] - ETA: 1s - loss: 0.1881 - categorical_accuracy: 0.9424
59456/60000 [============================>.] - ETA: 1s - loss: 0.1880 - categorical_accuracy: 0.9424
59488/60000 [============================>.] - ETA: 0s - loss: 0.1879 - categorical_accuracy: 0.9425
59520/60000 [============================>.] - ETA: 0s - loss: 0.1879 - categorical_accuracy: 0.9425
59552/60000 [============================>.] - ETA: 0s - loss: 0.1879 - categorical_accuracy: 0.9425
59584/60000 [============================>.] - ETA: 0s - loss: 0.1878 - categorical_accuracy: 0.9425
59616/60000 [============================>.] - ETA: 0s - loss: 0.1877 - categorical_accuracy: 0.9425
59648/60000 [============================>.] - ETA: 0s - loss: 0.1876 - categorical_accuracy: 0.9426
59680/60000 [============================>.] - ETA: 0s - loss: 0.1877 - categorical_accuracy: 0.9426
59712/60000 [============================>.] - ETA: 0s - loss: 0.1876 - categorical_accuracy: 0.9426
59744/60000 [============================>.] - ETA: 0s - loss: 0.1876 - categorical_accuracy: 0.9426
59776/60000 [============================>.] - ETA: 0s - loss: 0.1875 - categorical_accuracy: 0.9426
59808/60000 [============================>.] - ETA: 0s - loss: 0.1874 - categorical_accuracy: 0.9426
59840/60000 [============================>.] - ETA: 0s - loss: 0.1874 - categorical_accuracy: 0.9426
59872/60000 [============================>.] - ETA: 0s - loss: 0.1873 - categorical_accuracy: 0.9427
59904/60000 [============================>.] - ETA: 0s - loss: 0.1873 - categorical_accuracy: 0.9427
59936/60000 [============================>.] - ETA: 0s - loss: 0.1872 - categorical_accuracy: 0.9427
59968/60000 [============================>.] - ETA: 0s - loss: 0.1871 - categorical_accuracy: 0.9428
60000/60000 [==============================] - 118s 2ms/step - loss: 0.1870 - categorical_accuracy: 0.9428 - val_loss: 0.0477 - val_categorical_accuracy: 0.9845

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 18s
  160/10000 [..............................] - ETA: 6s 
  288/10000 [..............................] - ETA: 5s
  448/10000 [>.............................] - ETA: 4s
  608/10000 [>.............................] - ETA: 4s
  768/10000 [=>............................] - ETA: 4s
  928/10000 [=>............................] - ETA: 3s
 1088/10000 [==>...........................] - ETA: 3s
 1216/10000 [==>...........................] - ETA: 3s
 1376/10000 [===>..........................] - ETA: 3s
 1536/10000 [===>..........................] - ETA: 3s
 1696/10000 [====>.........................] - ETA: 3s
 1856/10000 [====>.........................] - ETA: 3s
 2016/10000 [=====>........................] - ETA: 3s
 2176/10000 [=====>........................] - ETA: 3s
 2336/10000 [======>.......................] - ETA: 3s
 2496/10000 [======>.......................] - ETA: 3s
 2656/10000 [======>.......................] - ETA: 2s
 2784/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
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
 5056/10000 [==============>...............] - ETA: 1s
 5216/10000 [==============>...............] - ETA: 1s
 5344/10000 [===============>..............] - ETA: 1s
 5504/10000 [===============>..............] - ETA: 1s
 5664/10000 [===============>..............] - ETA: 1s
 5824/10000 [================>.............] - ETA: 1s
 5952/10000 [================>.............] - ETA: 1s
 6112/10000 [=================>............] - ETA: 1s
 6272/10000 [=================>............] - ETA: 1s
 6400/10000 [==================>...........] - ETA: 1s
 6528/10000 [==================>...........] - ETA: 1s
 6656/10000 [==================>...........] - ETA: 1s
 6784/10000 [===================>..........] - ETA: 1s
 6944/10000 [===================>..........] - ETA: 1s
 7104/10000 [====================>.........] - ETA: 1s
 7232/10000 [====================>.........] - ETA: 1s
 7392/10000 [=====================>........] - ETA: 1s
 7552/10000 [=====================>........] - ETA: 0s
 7712/10000 [======================>.......] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8032/10000 [=======================>......] - ETA: 0s
 8160/10000 [=======================>......] - ETA: 0s
 8288/10000 [=======================>......] - ETA: 0s
 8448/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8896/10000 [=========================>....] - ETA: 0s
 9056/10000 [==========================>...] - ETA: 0s
 9184/10000 [==========================>...] - ETA: 0s
 9344/10000 [===========================>..] - ETA: 0s
 9504/10000 [===========================>..] - ETA: 0s
 9664/10000 [===========================>..] - ETA: 0s
 9824/10000 [============================>.] - ETA: 0s
 9984/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 390us/step
[[1.6761273e-08 6.3997119e-09 5.9408237e-08 ... 9.9999917e-01
  3.7824579e-09 4.0661942e-07]
 [5.6616454e-07 7.6670867e-06 9.9998689e-01 ... 6.7744224e-09
  1.5419037e-06 1.7265815e-10]
 [7.4624872e-08 9.9997461e-01 1.0041559e-05 ... 4.5801357e-06
  2.8479858e-06 4.8942052e-08]
 ...
 [1.5616328e-09 2.3368267e-07 1.9686330e-09 ... 2.6198175e-06
  1.0360448e-06 3.1701955e-05]
 [8.7358485e-06 1.0024041e-07 2.1761667e-07 ... 2.9657867e-08
  2.5205174e-03 1.6482220e-06]
 [3.0431509e-06 9.9773551e-08 1.6299409e-06 ... 2.2226614e-10
  7.9141921e-07 2.3889253e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.047662324998306575, 'accuracy_test:': 0.984499990940094}

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
[master 0984e65] ml_store  && git pull --all
 1 file changed, 2039 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 153f369...0984e65 master -> master (forced update)





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
{'loss': 0.4525422677397728, 'loss_history': []}

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
[master 5d4c29b] ml_store  && git pull --all
 1 file changed, 111 insertions(+)
To github.com:arita37/mlmodels_store.git
   0984e65..5d4c29b  master -> master





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
[master c239b25] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   5d4c29b..c239b25  master -> master





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
	Data preprocessing and feature engineering runtime = 0.26s ...
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
 40%|████      | 2/5 [00:22<00:34, 11.43s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9548765553068805, 'learning_rate': 0.14200008152063784, 'min_data_in_leaf': 16, 'num_leaves': 32} and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee\x8eYG\x18StX\r\x00\x00\x00learning_rateq\x02G?\xc2-\x0f\x05\x14\x8aeX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04K u.' and reward: 0.3908
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee\x8eYG\x18StX\r\x00\x00\x00learning_rateq\x02G?\xc2-\x0f\x05\x14\x8aeX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04K u.' and reward: 0.3908
 60%|██████    | 3/5 [00:43<00:28, 14.23s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9295283243034723, 'learning_rate': 0.005794340851071489, 'min_data_in_leaf': 3, 'num_leaves': 38} and reward: 0.385
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xbe\xb2/2\xd9\xdfX\r\x00\x00\x00learning_rateq\x02G?w\xbb\xce\x87P\xcf3X\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K&u.' and reward: 0.385
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xbe\xb2/2\xd9\xdfX\r\x00\x00\x00learning_rateq\x02G?w\xbb\xce\x87P\xcf3X\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K&u.' and reward: 0.385
 80%|████████  | 4/5 [01:07<00:17, 17.27s/it] 80%|████████  | 4/5 [01:08<00:17, 17.00s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9039629129955767, 'learning_rate': 0.009833622364190673, 'min_data_in_leaf': 23, 'num_leaves': 57} and reward: 0.3876
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xedC\xa1\x83\x9c\xd7X\r\x00\x00\x00learning_rateq\x02G?\x84#\xa6s\xa4\xaf\xa5X\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3876
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xedC\xa1\x83\x9c\xd7X\r\x00\x00\x00learning_rateq\x02G?\x84#\xa6s\xa4\xaf\xa5X\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3876
Time for Gradient Boosting hyperparameter optimization: 100.1417293548584
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 1.0, 'learning_rate': 0.1, 'min_data_in_leaf': 20, 'num_leaves': 36}
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
 40%|████      | 2/5 [00:55<01:23, 27.84s/it] 40%|████      | 2/5 [00:55<01:23, 27.85s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.4305919790377471, 'embedding_size_factor': 0.917445944972391, 'layers.choice': 1, 'learning_rate': 0.00048473140101954756, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 0.0012890685911458467} and reward: 0.3514
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\x8e\xd1\xa8\xf8\xc5IX\x15\x00\x00\x00embedding_size_factorq\x03G?\xed[\xb7\x990"\xb1X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G??\xc4q\x83\xc4\xe6\x9bX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?U\x1e\xbe\xdc=\xcd\xc8u.' and reward: 0.3514
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xdb\x8e\xd1\xa8\xf8\xc5IX\x15\x00\x00\x00embedding_size_factorq\x03G?\xed[\xb7\x990"\xb1X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G??\xc4q\x83\xc4\xe6\x9bX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?U\x1e\xbe\xdc=\xcd\xc8u.' and reward: 0.3514
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 155.55453181266785
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -139.8s of remaining time.
Ensemble size: 13
Ensemble weights: 
[0.38461538 0.07692308 0.23076923 0.         0.07692308 0.23076923]
	0.4	 = Validation accuracy score
	1.59s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 261.44s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f1e23bcc7f0>

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
[master 97516b7] ml_store  && git pull --all
 1 file changed, 197 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 6ef78f3...97516b7 master -> master (forced update)





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
[master e3a8ffc] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   97516b7..e3a8ffc  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.41it/s, avg_epoch_loss=5.27]
INFO:root:Epoch[0] Elapsed time 2.935 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.271965
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.271964883804321 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7eff735823c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7eff735823c8>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 94.21it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1054.6168619791667,
    "abs_error": 368.46429443359375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.441425926062585,
    "sMAPE": 0.5109808692462534,
    "MSIS": 97.65704674848458,
    "QuantileLoss[0.5]": 368.46431732177734,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.47486508023038,
    "NRMSE": 0.6836813701101133,
    "ND": 0.6464285867256031,
    "wQuantileLoss[0.5]": 0.6464286268803111,
    "mean_wQuantileLoss": 0.6464286268803111,
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
100%|██████████| 10/10 [00:01<00:00,  7.09it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.412 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7eff6bc229e8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7eff6bc229e8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 139.09it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.14it/s, avg_epoch_loss=5.26]
INFO:root:Epoch[0] Elapsed time 1.948 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.263111
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.2631114482879635 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7eff480abd30>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7eff480abd30>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 142.08it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 283.06024169921875,
    "abs_error": 187.05648803710938,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2394269035865029,
    "sMAPE": 0.3052227156271287,
    "MSIS": 49.577071290469526,
    "QuantileLoss[0.5]": 187.05647659301758,
    "Coverage[0.5]": 0.75,
    "RMSE": 16.824394244644257,
    "NRMSE": 0.35419777357145804,
    "ND": 0.32816927725808664,
    "wQuantileLoss[0.5]": 0.3281692571807326,
    "mean_wQuantileLoss": 0.3281692571807326,
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
 30%|███       | 3/10 [00:13<00:31,  4.53s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:25<00:17,  4.36s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:37<00:04,  4.22s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:41<00:00,  4.12s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 41.184 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.855105
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.855105018615722 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7eff407bf8d0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7eff407bf8d0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 108.20it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 52922.223958333336,
    "abs_error": 2702.5791015625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.907153516549926,
    "sMAPE": 1.4096994170755426,
    "MSIS": 716.28619242723,
    "QuantileLoss[0.5]": 2702.579345703125,
    "Coverage[0.5]": 1.0,
    "RMSE": 230.04830787974367,
    "NRMSE": 4.843122271152499,
    "ND": 4.741366844846492,
    "wQuantileLoss[0.5]": 4.741367273163378,
    "mean_wQuantileLoss": 4.741367273163378,
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
100%|██████████| 10/10 [00:00<00:00, 45.64it/s, avg_epoch_loss=5.18]
INFO:root:Epoch[0] Elapsed time 0.220 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.177787
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.177786922454834 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7eff40564f28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7eff40564f28>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 92.55it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 521.1596272786459,
    "abs_error": 189.61862182617188,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.256403473509713,
    "sMAPE": 0.31833825939142657,
    "MSIS": 50.256134087397925,
    "QuantileLoss[0.5]": 189.61861419677734,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 22.82892085225769,
    "NRMSE": 0.4806088600475303,
    "ND": 0.3326642488178454,
    "wQuantileLoss[0.5]": 0.3326642354329427,
    "mean_wQuantileLoss": 0.3326642354329427,
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
100%|██████████| 10/10 [00:01<00:00,  7.49it/s, avg_epoch_loss=161]
INFO:root:Epoch[0] Elapsed time 1.335 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=161.108291
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 161.1082910709855 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7eff407f6eb8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7eff407f6eb8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 125.65it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [02:05<18:47, 125.29s/it, avg_epoch_loss=0.582] 20%|██        | 2/10 [05:14<19:15, 144.38s/it, avg_epoch_loss=0.565] 30%|███       | 3/10 [08:19<18:16, 156.58s/it, avg_epoch_loss=0.548] 40%|████      | 4/10 [11:34<16:48, 168.10s/it, avg_epoch_loss=0.531] 50%|█████     | 5/10 [15:24<15:34, 186.89s/it, avg_epoch_loss=0.515] 60%|██████    | 6/10 [18:48<12:47, 191.84s/it, avg_epoch_loss=0.499] 70%|███████   | 7/10 [21:58<09:33, 191.29s/it, avg_epoch_loss=0.484] 80%|████████  | 8/10 [25:41<06:41, 200.83s/it, avg_epoch_loss=0.47]  90%|█████████ | 9/10 [29:04<03:21, 201.61s/it, avg_epoch_loss=0.457]100%|██████████| 10/10 [32:56<00:00, 210.51s/it, avg_epoch_loss=0.447]100%|██████████| 10/10 [32:56<00:00, 197.62s/it, avg_epoch_loss=0.447]
INFO:root:Epoch[0] Elapsed time 1976.235 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.446552
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.4465524971485138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7eff404aecc0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7eff404aecc0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 15.91it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
[master e9cb669] ml_store  && git pull --all
 1 file changed, 499 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 46eba0f...e9cb669 master -> master (forced update)





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f30c7809470> 

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
[master 2fd3bfe] ml_store  && git pull --all
 1 file changed, 107 insertions(+)
To github.com:arita37/mlmodels_store.git
   e9cb669..2fd3bfe  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
   0.96622086  0.4359862  -0.68748739  3.32107876]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 0.78344054 -0.05118845  0.82458463 -0.72559712  0.9317172  -0.86776868
   3.03085711 -0.13597733 -0.79726979  0.65458015]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 0.77528533  1.47016034  1.03298378 -0.87000822  0.78655651  0.36919047
  -0.14319575  0.85328219 -0.13971173 -0.22241403]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f059dd63d68>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f05beddffd0> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 1.16777676e+00 -6.65754518e-01 -1.23312074e+00 -1.67419581e+00
   1.01313574e+00  8.25029824e-01 -1.20464572e-01 -4.98213564e-01
  -3.10984978e-01 -1.18231813e+00]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 1.02817479e+00 -5.08457134e-01  1.76533510e+00  7.77419205e-01
   6.17714185e-01 -1.18771172e-01  4.50155513e-01 -1.98998184e-01
   1.86647138e+00  8.70969803e-01]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 1.13545112e+00  8.61623101e-01  4.90616924e-02 -2.08639057e+00
  -1.11469020e+00  3.61801641e-01 -8.06178212e-01  4.25920177e-01
   4.90803971e-02 -5.96086335e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
  -1.02887218e-01  1.71647264e-02]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 1.32720112e+00 -1.61198320e-01  6.02450901e-01 -2.86384915e-01
  -5.78962302e-01 -8.70887650e-01  1.37975819e+00  5.01429590e-01
  -4.78614074e-01 -8.92646674e-01]
 [ 5.63077902e-01 -1.17598267e+00 -1.74180344e-01  1.01012718e+00
   1.06796368e+00  9.20017933e-01 -1.68198840e-01 -1.95057341e-01
   8.05393424e-01  4.61164100e-01]
 [ 3.54133613e-01  2.11124755e-01  9.21450069e-01  1.65275673e-02
   9.03945451e-01  1.77187720e-01  9.54250872e-02 -1.11647002e+00
   8.09271010e-02  6.07501958e-02]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 1.58463774e+00  5.71209961e-02 -1.77183179e-02 -7.99547491e-01
   1.32970299e+00 -2.91594596e-01 -1.10771250e+00 -2.58982853e-01
   1.89293198e-01 -1.71939447e+00]
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
[[ 7.00175710e-01  5.56073510e-01  8.96864073e-02  1.69380911e+00
   8.82393314e-01  1.96869779e-01 -5.63788735e-01  1.69869255e-01
  -1.16400797e+00 -6.01156801e-01]
 [ 5.58538729e-01 -5.16347909e-01 -5.18145552e-01  3.51116897e-01
   8.25506954e-01 -6.87704631e-02 -9.52062101e-01 -1.34776494e+00
   1.47073986e+00 -1.46140360e+00]
 [ 1.13545112e+00  8.61623101e-01  4.90616924e-02 -2.08639057e+00
  -1.11469020e+00  3.61801641e-01 -8.06178212e-01  4.25920177e-01
   4.90803971e-02 -5.96086335e-01]
 [ 1.34728643e+00 -3.64538050e-01  8.07509886e-02 -4.59717681e-01
  -8.89487596e-01  1.70548352e+00  9.49961101e-02  2.40505552e-01
  -9.99426501e-01 -7.67803746e-01]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 8.78740711e-01 -1.92316341e-02  3.19656942e-01  1.50016279e-01
  -1.46662161e+00  4.63534322e-01 -8.98683193e-01  3.97880425e-01
  -9.96010889e-01  3.18154200e-01]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 1.02817479e+00 -5.08457134e-01  1.76533510e+00  7.77419205e-01
   6.17714185e-01 -1.18771172e-01  4.50155513e-01 -1.98998184e-01
   1.86647138e+00  8.70969803e-01]
 [ 1.17867274e+00 -5.99804531e-01 -6.94693595e-01  1.12341216e+00
   1.17899425e+00  3.05267040e-01  1.33526763e-02  1.38877940e+00
  -6.61344243e-01  6.21803504e-01]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 1.01195228e+00 -1.88141087e+00  1.70018815e+00  4.97269099e-01
  -9.17664624e-01  2.37332699e-01 -1.09033833e+00 -2.14444405e+00
  -3.69562425e-01  6.08783659e-01]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 1.83829400e+00  5.02740882e-01  1.29101580e-01  1.55880554e+00
   1.32551412e+00  1.09402696e-01  1.40754000e+00 -1.21974440e+00
   2.44936865e+00  1.61694960e+00]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 6.10942600e-01 -2.79099641e+00 -1.33520272e+00 -4.56117555e-01
  -9.44959948e-01 -9.79890252e-01 -1.56993672e-01  6.92574348e-01
  -4.78672356e-01 -1.06460122e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]]
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
[master e55962f] ml_store  && git pull --all
 1 file changed, 295 insertions(+)
To github.com:arita37/mlmodels_store.git
   2fd3bfe..e55962f  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139770779680720
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139770779680496
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @139770779679264
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139770779678816
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139770779678312
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @139770779677976

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
grad_step = 000000, loss = 0.916247
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.777371
grad_step = 000002, loss = 0.678086
grad_step = 000003, loss = 0.565008
grad_step = 000004, loss = 0.438476
grad_step = 000005, loss = 0.302521
grad_step = 000006, loss = 0.205028
grad_step = 000007, loss = 0.170908
grad_step = 000008, loss = 0.141226
grad_step = 000009, loss = 0.095019
grad_step = 000010, loss = 0.049789
grad_step = 000011, loss = 0.026046
grad_step = 000012, loss = 0.025908
grad_step = 000013, loss = 0.036788
grad_step = 000014, loss = 0.046714
grad_step = 000015, loss = 0.050650
grad_step = 000016, loss = 0.047741
grad_step = 000017, loss = 0.039181
grad_step = 000018, loss = 0.028343
grad_step = 000019, loss = 0.018943
grad_step = 000020, loss = 0.013691
grad_step = 000021, loss = 0.013035
grad_step = 000022, loss = 0.013825
grad_step = 000023, loss = 0.013633
grad_step = 000024, loss = 0.011866
grad_step = 000025, loss = 0.009783
grad_step = 000026, loss = 0.009060
grad_step = 000027, loss = 0.009211
grad_step = 000028, loss = 0.010215
grad_step = 000029, loss = 0.010755
grad_step = 000030, loss = 0.010289
grad_step = 000031, loss = 0.009043
grad_step = 000032, loss = 0.007787
grad_step = 000033, loss = 0.007221
grad_step = 000034, loss = 0.007423
grad_step = 000035, loss = 0.007749
grad_step = 000036, loss = 0.007536
grad_step = 000037, loss = 0.006782
grad_step = 000038, loss = 0.006016
grad_step = 000039, loss = 0.005656
grad_step = 000040, loss = 0.005656
grad_step = 000041, loss = 0.005672
grad_step = 000042, loss = 0.005463
grad_step = 000043, loss = 0.005089
grad_step = 000044, loss = 0.004811
grad_step = 000045, loss = 0.004809
grad_step = 000046, loss = 0.005012
grad_step = 000047, loss = 0.005165
grad_step = 000048, loss = 0.005091
grad_step = 000049, loss = 0.004848
grad_step = 000050, loss = 0.004623
grad_step = 000051, loss = 0.004517
grad_step = 000052, loss = 0.004469
grad_step = 000053, loss = 0.004369
grad_step = 000054, loss = 0.004200
grad_step = 000055, loss = 0.004052
grad_step = 000056, loss = 0.004021
grad_step = 000057, loss = 0.004089
grad_step = 000058, loss = 0.004145
grad_step = 000059, loss = 0.004104
grad_step = 000060, loss = 0.003992
grad_step = 000061, loss = 0.003893
grad_step = 000062, loss = 0.003843
grad_step = 000063, loss = 0.003807
grad_step = 000064, loss = 0.003744
grad_step = 000065, loss = 0.003665
grad_step = 000066, loss = 0.003611
grad_step = 000067, loss = 0.003600
grad_step = 000068, loss = 0.003596
grad_step = 000069, loss = 0.003559
grad_step = 000070, loss = 0.003488
grad_step = 000071, loss = 0.003423
grad_step = 000072, loss = 0.003387
grad_step = 000073, loss = 0.003364
grad_step = 000074, loss = 0.003331
grad_step = 000075, loss = 0.003286
grad_step = 000076, loss = 0.003248
grad_step = 000077, loss = 0.003219
grad_step = 000078, loss = 0.003184
grad_step = 000079, loss = 0.003133
grad_step = 000080, loss = 0.003080
grad_step = 000081, loss = 0.003041
grad_step = 000082, loss = 0.003016
grad_step = 000083, loss = 0.002990
grad_step = 000084, loss = 0.002957
grad_step = 000085, loss = 0.002922
grad_step = 000086, loss = 0.002888
grad_step = 000087, loss = 0.002853
grad_step = 000088, loss = 0.002813
grad_step = 000089, loss = 0.002776
grad_step = 000090, loss = 0.002746
grad_step = 000091, loss = 0.002718
grad_step = 000092, loss = 0.002687
grad_step = 000093, loss = 0.002651
grad_step = 000094, loss = 0.002618
grad_step = 000095, loss = 0.002587
grad_step = 000096, loss = 0.002556
grad_step = 000097, loss = 0.002524
grad_step = 000098, loss = 0.002493
grad_step = 000099, loss = 0.002463
grad_step = 000100, loss = 0.002433
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002403
grad_step = 000102, loss = 0.002375
grad_step = 000103, loss = 0.002346
grad_step = 000104, loss = 0.002317
grad_step = 000105, loss = 0.002289
grad_step = 000106, loss = 0.002262
grad_step = 000107, loss = 0.002237
grad_step = 000108, loss = 0.002211
grad_step = 000109, loss = 0.002185
grad_step = 000110, loss = 0.002158
grad_step = 000111, loss = 0.002133
grad_step = 000112, loss = 0.002108
grad_step = 000113, loss = 0.002083
grad_step = 000114, loss = 0.002060
grad_step = 000115, loss = 0.002037
grad_step = 000116, loss = 0.002014
grad_step = 000117, loss = 0.001991
grad_step = 000118, loss = 0.001970
grad_step = 000119, loss = 0.001947
grad_step = 000120, loss = 0.001926
grad_step = 000121, loss = 0.001905
grad_step = 000122, loss = 0.001885
grad_step = 000123, loss = 0.001865
grad_step = 000124, loss = 0.001844
grad_step = 000125, loss = 0.001825
grad_step = 000126, loss = 0.001805
grad_step = 000127, loss = 0.001786
grad_step = 000128, loss = 0.001766
grad_step = 000129, loss = 0.001745
grad_step = 000130, loss = 0.001725
grad_step = 000131, loss = 0.001704
grad_step = 000132, loss = 0.001683
grad_step = 000133, loss = 0.001661
grad_step = 000134, loss = 0.001638
grad_step = 000135, loss = 0.001615
grad_step = 000136, loss = 0.001590
grad_step = 000137, loss = 0.001566
grad_step = 000138, loss = 0.001539
grad_step = 000139, loss = 0.001512
grad_step = 000140, loss = 0.001483
grad_step = 000141, loss = 0.001452
grad_step = 000142, loss = 0.001421
grad_step = 000143, loss = 0.001389
grad_step = 000144, loss = 0.001355
grad_step = 000145, loss = 0.001321
grad_step = 000146, loss = 0.001286
grad_step = 000147, loss = 0.001251
grad_step = 000148, loss = 0.001215
grad_step = 000149, loss = 0.001180
grad_step = 000150, loss = 0.001145
grad_step = 000151, loss = 0.001111
grad_step = 000152, loss = 0.001079
grad_step = 000153, loss = 0.001047
grad_step = 000154, loss = 0.001020
grad_step = 000155, loss = 0.000996
grad_step = 000156, loss = 0.000977
grad_step = 000157, loss = 0.000957
grad_step = 000158, loss = 0.000933
grad_step = 000159, loss = 0.000913
grad_step = 000160, loss = 0.000897
grad_step = 000161, loss = 0.000886
grad_step = 000162, loss = 0.000873
grad_step = 000163, loss = 0.000856
grad_step = 000164, loss = 0.000837
grad_step = 000165, loss = 0.000825
grad_step = 000166, loss = 0.000815
grad_step = 000167, loss = 0.000806
grad_step = 000168, loss = 0.000794
grad_step = 000169, loss = 0.000781
grad_step = 000170, loss = 0.000771
grad_step = 000171, loss = 0.000764
grad_step = 000172, loss = 0.000758
grad_step = 000173, loss = 0.000751
grad_step = 000174, loss = 0.000744
grad_step = 000175, loss = 0.000735
grad_step = 000176, loss = 0.000727
grad_step = 000177, loss = 0.000720
grad_step = 000178, loss = 0.000716
grad_step = 000179, loss = 0.000711
grad_step = 000180, loss = 0.000707
grad_step = 000181, loss = 0.000703
grad_step = 000182, loss = 0.000698
grad_step = 000183, loss = 0.000692
grad_step = 000184, loss = 0.000683
grad_step = 000185, loss = 0.000676
grad_step = 000186, loss = 0.000669
grad_step = 000187, loss = 0.000665
grad_step = 000188, loss = 0.000662
grad_step = 000189, loss = 0.000660
grad_step = 000190, loss = 0.000659
grad_step = 000191, loss = 0.000662
grad_step = 000192, loss = 0.000664
grad_step = 000193, loss = 0.000657
grad_step = 000194, loss = 0.000639
grad_step = 000195, loss = 0.000626
grad_step = 000196, loss = 0.000628
grad_step = 000197, loss = 0.000632
grad_step = 000198, loss = 0.000624
grad_step = 000199, loss = 0.000611
grad_step = 000200, loss = 0.000605
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000607
grad_step = 000202, loss = 0.000608
grad_step = 000203, loss = 0.000601
grad_step = 000204, loss = 0.000589
grad_step = 000205, loss = 0.000582
grad_step = 000206, loss = 0.000580
grad_step = 000207, loss = 0.000579
grad_step = 000208, loss = 0.000576
grad_step = 000209, loss = 0.000569
grad_step = 000210, loss = 0.000562
grad_step = 000211, loss = 0.000557
grad_step = 000212, loss = 0.000553
grad_step = 000213, loss = 0.000551
grad_step = 000214, loss = 0.000549
grad_step = 000215, loss = 0.000545
grad_step = 000216, loss = 0.000542
grad_step = 000217, loss = 0.000537
grad_step = 000218, loss = 0.000532
grad_step = 000219, loss = 0.000527
grad_step = 000220, loss = 0.000522
grad_step = 000221, loss = 0.000516
grad_step = 000222, loss = 0.000512
grad_step = 000223, loss = 0.000509
grad_step = 000224, loss = 0.000507
grad_step = 000225, loss = 0.000508
grad_step = 000226, loss = 0.000519
grad_step = 000227, loss = 0.000546
grad_step = 000228, loss = 0.000584
grad_step = 000229, loss = 0.000575
grad_step = 000230, loss = 0.000507
grad_step = 000231, loss = 0.000481
grad_step = 000232, loss = 0.000521
grad_step = 000233, loss = 0.000522
grad_step = 000234, loss = 0.000475
grad_step = 000235, loss = 0.000476
grad_step = 000236, loss = 0.000501
grad_step = 000237, loss = 0.000477
grad_step = 000238, loss = 0.000455
grad_step = 000239, loss = 0.000473
grad_step = 000240, loss = 0.000473
grad_step = 000241, loss = 0.000448
grad_step = 000242, loss = 0.000446
grad_step = 000243, loss = 0.000458
grad_step = 000244, loss = 0.000449
grad_step = 000245, loss = 0.000431
grad_step = 000246, loss = 0.000434
grad_step = 000247, loss = 0.000440
grad_step = 000248, loss = 0.000428
grad_step = 000249, loss = 0.000417
grad_step = 000250, loss = 0.000418
grad_step = 000251, loss = 0.000420
grad_step = 000252, loss = 0.000414
grad_step = 000253, loss = 0.000404
grad_step = 000254, loss = 0.000401
grad_step = 000255, loss = 0.000403
grad_step = 000256, loss = 0.000400
grad_step = 000257, loss = 0.000393
grad_step = 000258, loss = 0.000386
grad_step = 000259, loss = 0.000384
grad_step = 000260, loss = 0.000385
grad_step = 000261, loss = 0.000383
grad_step = 000262, loss = 0.000378
grad_step = 000263, loss = 0.000372
grad_step = 000264, loss = 0.000367
grad_step = 000265, loss = 0.000364
grad_step = 000266, loss = 0.000361
grad_step = 000267, loss = 0.000360
grad_step = 000268, loss = 0.000357
grad_step = 000269, loss = 0.000355
grad_step = 000270, loss = 0.000353
grad_step = 000271, loss = 0.000351
grad_step = 000272, loss = 0.000348
grad_step = 000273, loss = 0.000347
grad_step = 000274, loss = 0.000346
grad_step = 000275, loss = 0.000346
grad_step = 000276, loss = 0.000349
grad_step = 000277, loss = 0.000354
grad_step = 000278, loss = 0.000363
grad_step = 000279, loss = 0.000377
grad_step = 000280, loss = 0.000383
grad_step = 000281, loss = 0.000377
grad_step = 000282, loss = 0.000351
grad_step = 000283, loss = 0.000325
grad_step = 000284, loss = 0.000319
grad_step = 000285, loss = 0.000331
grad_step = 000286, loss = 0.000342
grad_step = 000287, loss = 0.000338
grad_step = 000288, loss = 0.000321
grad_step = 000289, loss = 0.000310
grad_step = 000290, loss = 0.000313
grad_step = 000291, loss = 0.000322
grad_step = 000292, loss = 0.000324
grad_step = 000293, loss = 0.000315
grad_step = 000294, loss = 0.000305
grad_step = 000295, loss = 0.000301
grad_step = 000296, loss = 0.000304
grad_step = 000297, loss = 0.000308
grad_step = 000298, loss = 0.000309
grad_step = 000299, loss = 0.000305
grad_step = 000300, loss = 0.000298
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000293
grad_step = 000302, loss = 0.000291
grad_step = 000303, loss = 0.000292
grad_step = 000304, loss = 0.000294
grad_step = 000305, loss = 0.000295
grad_step = 000306, loss = 0.000295
grad_step = 000307, loss = 0.000293
grad_step = 000308, loss = 0.000290
grad_step = 000309, loss = 0.000287
grad_step = 000310, loss = 0.000283
grad_step = 000311, loss = 0.000281
grad_step = 000312, loss = 0.000279
grad_step = 000313, loss = 0.000278
grad_step = 000314, loss = 0.000278
grad_step = 000315, loss = 0.000279
grad_step = 000316, loss = 0.000281
grad_step = 000317, loss = 0.000284
grad_step = 000318, loss = 0.000289
grad_step = 000319, loss = 0.000295
grad_step = 000320, loss = 0.000302
grad_step = 000321, loss = 0.000305
grad_step = 000322, loss = 0.000307
grad_step = 000323, loss = 0.000298
grad_step = 000324, loss = 0.000285
grad_step = 000325, loss = 0.000272
grad_step = 000326, loss = 0.000265
grad_step = 000327, loss = 0.000266
grad_step = 000328, loss = 0.000271
grad_step = 000329, loss = 0.000276
grad_step = 000330, loss = 0.000277
grad_step = 000331, loss = 0.000274
grad_step = 000332, loss = 0.000267
grad_step = 000333, loss = 0.000261
grad_step = 000334, loss = 0.000257
grad_step = 000335, loss = 0.000257
grad_step = 000336, loss = 0.000260
grad_step = 000337, loss = 0.000262
grad_step = 000338, loss = 0.000264
grad_step = 000339, loss = 0.000264
grad_step = 000340, loss = 0.000262
grad_step = 000341, loss = 0.000259
grad_step = 000342, loss = 0.000255
grad_step = 000343, loss = 0.000252
grad_step = 000344, loss = 0.000249
grad_step = 000345, loss = 0.000248
grad_step = 000346, loss = 0.000247
grad_step = 000347, loss = 0.000247
grad_step = 000348, loss = 0.000248
grad_step = 000349, loss = 0.000248
grad_step = 000350, loss = 0.000249
grad_step = 000351, loss = 0.000250
grad_step = 000352, loss = 0.000252
grad_step = 000353, loss = 0.000254
grad_step = 000354, loss = 0.000256
grad_step = 000355, loss = 0.000259
grad_step = 000356, loss = 0.000261
grad_step = 000357, loss = 0.000262
grad_step = 000358, loss = 0.000261
grad_step = 000359, loss = 0.000258
grad_step = 000360, loss = 0.000253
grad_step = 000361, loss = 0.000246
grad_step = 000362, loss = 0.000240
grad_step = 000363, loss = 0.000235
grad_step = 000364, loss = 0.000233
grad_step = 000365, loss = 0.000233
grad_step = 000366, loss = 0.000235
grad_step = 000367, loss = 0.000237
grad_step = 000368, loss = 0.000240
grad_step = 000369, loss = 0.000241
grad_step = 000370, loss = 0.000242
grad_step = 000371, loss = 0.000242
grad_step = 000372, loss = 0.000242
grad_step = 000373, loss = 0.000240
grad_step = 000374, loss = 0.000238
grad_step = 000375, loss = 0.000236
grad_step = 000376, loss = 0.000233
grad_step = 000377, loss = 0.000230
grad_step = 000378, loss = 0.000227
grad_step = 000379, loss = 0.000225
grad_step = 000380, loss = 0.000223
grad_step = 000381, loss = 0.000222
grad_step = 000382, loss = 0.000221
grad_step = 000383, loss = 0.000220
grad_step = 000384, loss = 0.000219
grad_step = 000385, loss = 0.000218
grad_step = 000386, loss = 0.000218
grad_step = 000387, loss = 0.000218
grad_step = 000388, loss = 0.000218
grad_step = 000389, loss = 0.000219
grad_step = 000390, loss = 0.000222
grad_step = 000391, loss = 0.000228
grad_step = 000392, loss = 0.000240
grad_step = 000393, loss = 0.000261
grad_step = 000394, loss = 0.000294
grad_step = 000395, loss = 0.000338
grad_step = 000396, loss = 0.000359
grad_step = 000397, loss = 0.000341
grad_step = 000398, loss = 0.000268
grad_step = 000399, loss = 0.000215
grad_step = 000400, loss = 0.000227
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000270
grad_step = 000402, loss = 0.000273
grad_step = 000403, loss = 0.000231
grad_step = 000404, loss = 0.000209
grad_step = 000405, loss = 0.000231
grad_step = 000406, loss = 0.000250
grad_step = 000407, loss = 0.000232
grad_step = 000408, loss = 0.000208
grad_step = 000409, loss = 0.000214
grad_step = 000410, loss = 0.000231
grad_step = 000411, loss = 0.000227
grad_step = 000412, loss = 0.000209
grad_step = 000413, loss = 0.000205
grad_step = 000414, loss = 0.000216
grad_step = 000415, loss = 0.000221
grad_step = 000416, loss = 0.000211
grad_step = 000417, loss = 0.000202
grad_step = 000418, loss = 0.000204
grad_step = 000419, loss = 0.000211
grad_step = 000420, loss = 0.000210
grad_step = 000421, loss = 0.000203
grad_step = 000422, loss = 0.000199
grad_step = 000423, loss = 0.000201
grad_step = 000424, loss = 0.000205
grad_step = 000425, loss = 0.000204
grad_step = 000426, loss = 0.000199
grad_step = 000427, loss = 0.000196
grad_step = 000428, loss = 0.000196
grad_step = 000429, loss = 0.000199
grad_step = 000430, loss = 0.000199
grad_step = 000431, loss = 0.000197
grad_step = 000432, loss = 0.000194
grad_step = 000433, loss = 0.000193
grad_step = 000434, loss = 0.000193
grad_step = 000435, loss = 0.000194
grad_step = 000436, loss = 0.000194
grad_step = 000437, loss = 0.000193
grad_step = 000438, loss = 0.000191
grad_step = 000439, loss = 0.000190
grad_step = 000440, loss = 0.000189
grad_step = 000441, loss = 0.000189
grad_step = 000442, loss = 0.000189
grad_step = 000443, loss = 0.000189
grad_step = 000444, loss = 0.000189
grad_step = 000445, loss = 0.000188
grad_step = 000446, loss = 0.000187
grad_step = 000447, loss = 0.000186
grad_step = 000448, loss = 0.000185
grad_step = 000449, loss = 0.000184
grad_step = 000450, loss = 0.000184
grad_step = 000451, loss = 0.000184
grad_step = 000452, loss = 0.000184
grad_step = 000453, loss = 0.000184
grad_step = 000454, loss = 0.000184
grad_step = 000455, loss = 0.000184
grad_step = 000456, loss = 0.000183
grad_step = 000457, loss = 0.000183
grad_step = 000458, loss = 0.000183
grad_step = 000459, loss = 0.000182
grad_step = 000460, loss = 0.000181
grad_step = 000461, loss = 0.000181
grad_step = 000462, loss = 0.000181
grad_step = 000463, loss = 0.000181
grad_step = 000464, loss = 0.000181
grad_step = 000465, loss = 0.000182
grad_step = 000466, loss = 0.000183
grad_step = 000467, loss = 0.000185
grad_step = 000468, loss = 0.000188
grad_step = 000469, loss = 0.000192
grad_step = 000470, loss = 0.000197
grad_step = 000471, loss = 0.000204
grad_step = 000472, loss = 0.000210
grad_step = 000473, loss = 0.000215
grad_step = 000474, loss = 0.000214
grad_step = 000475, loss = 0.000208
grad_step = 000476, loss = 0.000195
grad_step = 000477, loss = 0.000181
grad_step = 000478, loss = 0.000172
grad_step = 000479, loss = 0.000170
grad_step = 000480, loss = 0.000175
grad_step = 000481, loss = 0.000182
grad_step = 000482, loss = 0.000188
grad_step = 000483, loss = 0.000190
grad_step = 000484, loss = 0.000187
grad_step = 000485, loss = 0.000182
grad_step = 000486, loss = 0.000176
grad_step = 000487, loss = 0.000171
grad_step = 000488, loss = 0.000167
grad_step = 000489, loss = 0.000165
grad_step = 000490, loss = 0.000165
grad_step = 000491, loss = 0.000167
grad_step = 000492, loss = 0.000169
grad_step = 000493, loss = 0.000173
grad_step = 000494, loss = 0.000177
grad_step = 000495, loss = 0.000181
grad_step = 000496, loss = 0.000185
grad_step = 000497, loss = 0.000189
grad_step = 000498, loss = 0.000191
grad_step = 000499, loss = 0.000190
grad_step = 000500, loss = 0.000185
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000178
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
[[0.84558153 0.8388268  0.92725074 0.9350555  1.0240383 ]
 [0.84006673 0.91220945 0.94230545 1.0116376  1.0000796 ]
 [0.8899828  0.9203848  1.0022128  0.97817373 0.9754827 ]
 [0.9263692  0.98981524 0.9807037  0.95322835 0.9122143 ]
 [0.99215543 0.9953245  0.9533976  0.90774226 0.8556386 ]
 [0.9758576  0.9468968  0.9310067  0.8407006  0.86439824]
 [0.9401772  0.89121413 0.84780794 0.8443587  0.82091534]
 [0.89227957 0.8370233  0.84706426 0.80547595 0.84313995]
 [0.8184953  0.8300039  0.80331004 0.82752526 0.8517567 ]
 [0.81549346 0.8006336  0.8326819  0.8376978  0.8433913 ]
 [0.7846521  0.8138771  0.84917253 0.80195045 0.92811155]
 [0.8172798  0.85528505 0.8243097  0.92298746 0.94744337]
 [0.8358911  0.8379823  0.9313638  0.9410247  1.0216259 ]
 [0.8413261  0.92311186 0.9496491  1.0176036  0.986697  ]
 [0.906935   0.93379056 1.0055993  0.9750782  0.9554702 ]
 [0.93359977 1.0026224  0.9791703  0.9293264  0.8890374 ]
 [0.99535096 0.9855094  0.9391945  0.881606   0.8400607 ]
 [0.9699937  0.92199296 0.9030617  0.82460296 0.8493763 ]
 [0.9292066  0.8820344  0.8353789  0.8405845  0.81983906]
 [0.8994149  0.8475597  0.8416101  0.80775166 0.8484351 ]
 [0.83569133 0.8508005  0.80750984 0.8377793  0.8626917 ]
 [0.84552693 0.8175625  0.84135735 0.85377693 0.848989  ]
 [0.7992027  0.8236429  0.8619052  0.81731045 0.9318069 ]
 [0.8340743  0.8659417  0.8324856  0.92453015 0.94989944]
 [0.8501469  0.8394467  0.92679805 0.9355545  1.0311502 ]
 [0.84651303 0.9203822  0.94941604 1.0195978  1.0112683 ]
 [0.89898205 0.93332297 1.0120617  0.9906786  0.9888713 ]
 [0.93966347 1.00461    0.9944432  0.9660255  0.92576313]
 [1.0021605  1.0077388  0.96164644 0.9151057  0.8657434 ]
 [0.9857505  0.95376045 0.9385059  0.84634805 0.8712837 ]
 [0.94505745 0.8961392  0.8539115  0.84942114 0.83005756]]

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
[master 39a6555] ml_store  && git pull --all
 1 file changed, 1121 insertions(+)
To github.com:arita37/mlmodels_store.git
   e55962f..39a6555  master -> master





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
[master 5f7d4c2] ml_store  && git pull --all
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   39a6555..5f7d4c2  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 9661263.31B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 288320.94B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 3076096/440473133 [00:00<00:14, 30760035.95B/s]  2%|▏         | 7140352/440473133 [00:00<00:13, 33175696.26B/s]  3%|▎         | 11310080/440473133 [00:00<00:12, 35341541.45B/s]  4%|▎         | 15810560/440473133 [00:00<00:11, 37774445.83B/s]  5%|▍         | 20379648/440473133 [00:00<00:10, 39844711.00B/s]  6%|▌         | 24967168/440473133 [00:00<00:10, 41478360.10B/s]  7%|▋         | 29463552/440473133 [00:00<00:09, 42465466.75B/s]  8%|▊         | 34102272/440473133 [00:00<00:09, 43568364.33B/s]  9%|▉         | 38564864/440473133 [00:00<00:09, 43878768.76B/s] 10%|▉         | 42961920/440473133 [00:01<00:09, 43905243.61B/s] 11%|█         | 47597568/440473133 [00:01<00:08, 44611525.31B/s] 12%|█▏        | 52236288/440473133 [00:01<00:08, 45128732.66B/s] 13%|█▎        | 56997888/440473133 [00:01<00:08, 45844175.98B/s] 14%|█▍        | 61656064/440473133 [00:01<00:08, 46061924.80B/s] 15%|█▌        | 66251776/440473133 [00:01<00:08, 45804618.45B/s] 16%|█▌        | 70825984/440473133 [00:01<00:08, 44423040.40B/s] 17%|█▋        | 75273216/440473133 [00:01<00:08, 44182271.95B/s] 18%|█▊        | 79694848/440473133 [00:01<00:08, 43606442.02B/s] 19%|█▉        | 84060160/440473133 [00:01<00:08, 43505988.11B/s] 20%|██        | 88414208/440473133 [00:02<00:08, 42086886.89B/s] 21%|██        | 92635136/440473133 [00:02<00:08, 41803217.83B/s] 22%|██▏       | 96824320/440473133 [00:02<00:08, 41076700.90B/s] 23%|██▎       | 101226496/440473133 [00:02<00:08, 41917436.48B/s] 24%|██▍       | 105510912/440473133 [00:02<00:07, 42189891.20B/s] 25%|██▍       | 109925376/440473133 [00:02<00:07, 42755654.77B/s] 26%|██▌       | 114696192/440473133 [00:02<00:07, 44128466.12B/s] 27%|██▋       | 119383040/440473133 [00:02<00:07, 44914611.29B/s] 28%|██▊       | 124090368/440473133 [00:02<00:06, 45537459.57B/s] 29%|██▉       | 128785408/440473133 [00:02<00:06, 45951989.84B/s] 30%|███       | 133390336/440473133 [00:03<00:06, 44663481.72B/s] 31%|███▏      | 137871360/440473133 [00:03<00:06, 43875232.33B/s] 32%|███▏      | 142333952/440473133 [00:03<00:06, 44097224.77B/s] 33%|███▎      | 146753536/440473133 [00:03<00:06, 43404945.73B/s] 34%|███▍      | 151183360/440473133 [00:03<00:06, 43668370.66B/s] 35%|███▌      | 155841536/440473133 [00:03<00:06, 44499906.05B/s] 36%|███▋      | 160541696/440473133 [00:03<00:06, 45220485.64B/s] 38%|███▊      | 165409792/440473133 [00:03<00:05, 46203630.54B/s] 39%|███▊      | 170042368/440473133 [00:03<00:05, 46172704.02B/s] 40%|███▉      | 174667776/440473133 [00:03<00:05, 46104385.65B/s] 41%|████      | 179283968/440473133 [00:04<00:05, 45364627.99B/s] 42%|████▏     | 183918592/440473133 [00:04<00:05, 45653057.88B/s] 43%|████▎     | 188489728/440473133 [00:04<00:05, 44639189.36B/s] 44%|████▍     | 192962560/440473133 [00:04<00:05, 43925166.55B/s] 45%|████▍     | 197433344/440473133 [00:04<00:05, 44157026.11B/s] 46%|████▌     | 202215424/440473133 [00:04<00:05, 45194278.92B/s] 47%|████▋     | 206765056/440473133 [00:04<00:05, 45280903.95B/s] 48%|████▊     | 211429376/440473133 [00:04<00:05, 45680792.18B/s] 49%|████▉     | 216003584/440473133 [00:04<00:04, 45257054.14B/s] 50%|█████     | 220534784/440473133 [00:04<00:04, 44779622.69B/s] 51%|█████     | 225017856/440473133 [00:05<00:04, 44520963.62B/s] 52%|█████▏    | 229765120/440473133 [00:05<00:04, 45367020.18B/s] 53%|█████▎    | 234435584/440473133 [00:05<00:04, 45755971.62B/s] 54%|█████▍    | 239093760/440473133 [00:05<00:04, 45998421.10B/s] 55%|█████▌    | 243697664/440473133 [00:05<00:04, 44459972.64B/s] 56%|█████▋    | 248158208/440473133 [00:05<00:04, 43441319.20B/s] 57%|█████▋    | 252952576/440473133 [00:05<00:04, 44697991.73B/s] 58%|█████▊    | 257623040/440473133 [00:05<00:04, 45280652.41B/s] 60%|█████▉    | 262301696/440473133 [00:05<00:03, 45720433.74B/s] 61%|██████    | 266886144/440473133 [00:06<00:03, 45410709.94B/s] 62%|██████▏   | 271550464/440473133 [00:06<00:03, 45770813.12B/s] 63%|██████▎   | 276134912/440473133 [00:06<00:03, 45647578.06B/s] 64%|██████▎   | 280705024/440473133 [00:06<00:03, 45493598.20B/s] 65%|██████▍   | 285258752/440473133 [00:06<00:03, 45389910.00B/s] 66%|██████▌   | 289801216/440473133 [00:06<00:03, 44148149.06B/s] 67%|██████▋   | 294324224/440473133 [00:06<00:03, 44466359.89B/s] 68%|██████▊   | 298778624/440473133 [00:06<00:03, 44368999.51B/s] 69%|██████▉   | 303334400/440473133 [00:06<00:03, 44718698.10B/s] 70%|██████▉   | 307811328/440473133 [00:06<00:03, 44049802.03B/s] 71%|███████   | 312225792/440473133 [00:07<00:02, 44077058.48B/s] 72%|███████▏  | 316638208/440473133 [00:07<00:02, 44090655.27B/s] 73%|███████▎  | 321262592/440473133 [00:07<00:02, 44715023.72B/s] 74%|███████▍  | 325738496/440473133 [00:07<00:02, 43877628.77B/s] 75%|███████▍  | 330133504/440473133 [00:07<00:02, 43040067.36B/s] 76%|███████▌  | 334445568/440473133 [00:07<00:02, 42972836.78B/s] 77%|███████▋  | 339281920/440473133 [00:07<00:02, 44456731.70B/s] 78%|███████▊  | 344084480/440473133 [00:07<00:02, 45468777.95B/s] 79%|███████▉  | 348845056/440473133 [00:07<00:01, 46085731.77B/s] 80%|████████  | 353495040/440473133 [00:07<00:01, 46208970.93B/s] 81%|████████▏ | 358126592/440473133 [00:08<00:01, 45652199.55B/s] 82%|████████▏ | 362727424/440473133 [00:08<00:01, 45755571.61B/s] 83%|████████▎ | 367309824/440473133 [00:08<00:01, 45453756.82B/s] 84%|████████▍ | 371987456/440473133 [00:08<00:01, 45841269.00B/s] 85%|████████▌ | 376576000/440473133 [00:08<00:01, 45575703.75B/s] 87%|████████▋ | 381136896/440473133 [00:08<00:01, 44648051.29B/s] 88%|████████▊ | 385646592/440473133 [00:08<00:01, 44779982.51B/s] 89%|████████▊ | 390209536/440473133 [00:08<00:01, 45029744.69B/s] 90%|████████▉ | 394919936/440473133 [00:08<00:00, 45631646.93B/s] 91%|█████████ | 399537152/440473133 [00:08<00:00, 45789524.53B/s] 92%|█████████▏| 404119552/440473133 [00:09<00:00, 45133314.94B/s] 93%|█████████▎| 408717312/440473133 [00:09<00:00, 45379558.95B/s] 94%|█████████▍| 413279232/440473133 [00:09<00:00, 45447474.95B/s] 95%|█████████▍| 418046976/440473133 [00:09<00:00, 46092987.63B/s] 96%|█████████▌| 422881280/440473133 [00:09<00:00, 46742673.69B/s] 97%|█████████▋| 427560960/440473133 [00:09<00:00, 45934784.69B/s] 98%|█████████▊| 432161792/440473133 [00:09<00:00, 45712117.84B/s] 99%|█████████▉| 436738048/440473133 [00:09<00:00, 45183792.45B/s]100%|██████████| 440473133/440473133 [00:09<00:00, 44677475.70B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
2736128/7094233 [==========>...................] - ETA: 0s
5931008/7094233 [========================>.....] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:  15%|█▍        | 315/2118 [00:00<00:00, 3147.76it/s]Processing text_left with encode:  35%|███▌      | 743/2118 [00:00<00:00, 3417.91it/s]Processing text_left with encode:  52%|█████▏    | 1108/2118 [00:00<00:00, 3479.12it/s]Processing text_left with encode:  72%|███████▏  | 1522/2118 [00:00<00:00, 3652.99it/s]Processing text_left with encode:  93%|█████████▎| 1962/2118 [00:00<00:00, 3847.50it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3933.27it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:00<00:12, 1489.21it/s]Processing text_right with encode:   2%|▏         | 317/18841 [00:00<00:12, 1537.47it/s]Processing text_right with encode:   3%|▎         | 480/18841 [00:00<00:11, 1564.08it/s]Processing text_right with encode:   3%|▎         | 637/18841 [00:00<00:11, 1560.40it/s]Processing text_right with encode:   4%|▍         | 795/18841 [00:00<00:11, 1565.75it/s]Processing text_right with encode:   5%|▌         | 954/18841 [00:00<00:11, 1569.40it/s]Processing text_right with encode:   6%|▌         | 1102/18841 [00:00<00:11, 1539.33it/s]Processing text_right with encode:   7%|▋         | 1259/18841 [00:00<00:11, 1546.46it/s]Processing text_right with encode:   7%|▋         | 1406/18841 [00:00<00:11, 1516.11it/s]Processing text_right with encode:   8%|▊         | 1566/18841 [00:01<00:11, 1538.20it/s]Processing text_right with encode:   9%|▉         | 1733/18841 [00:01<00:10, 1572.87it/s]Processing text_right with encode:  10%|█         | 1891/18841 [00:01<00:10, 1574.32it/s]Processing text_right with encode:  11%|█         | 2047/18841 [00:01<00:10, 1561.63it/s]Processing text_right with encode:  12%|█▏        | 2206/18841 [00:01<00:10, 1569.71it/s]Processing text_right with encode:  13%|█▎        | 2363/18841 [00:01<00:10, 1564.49it/s]Processing text_right with encode:  13%|█▎        | 2519/18841 [00:01<00:10, 1554.13it/s]Processing text_right with encode:  14%|█▍        | 2677/18841 [00:01<00:10, 1541.86it/s]Processing text_right with encode:  15%|█▌        | 2856/18841 [00:01<00:09, 1608.05it/s]Processing text_right with encode:  16%|█▌        | 3018/18841 [00:01<00:10, 1525.28it/s]Processing text_right with encode:  17%|█▋        | 3172/18841 [00:02<00:10, 1483.21it/s]Processing text_right with encode:  18%|█▊        | 3322/18841 [00:02<00:10, 1459.77it/s]Processing text_right with encode:  18%|█▊        | 3473/18841 [00:02<00:10, 1473.85it/s]Processing text_right with encode:  19%|█▉        | 3631/18841 [00:02<00:10, 1500.49it/s]Processing text_right with encode:  20%|██        | 3782/18841 [00:02<00:10, 1462.17it/s]Processing text_right with encode:  21%|██        | 3950/18841 [00:02<00:09, 1516.98it/s]Processing text_right with encode:  22%|██▏       | 4112/18841 [00:02<00:09, 1544.46it/s]Processing text_right with encode:  23%|██▎       | 4268/18841 [00:02<00:09, 1540.84it/s]Processing text_right with encode:  23%|██▎       | 4423/18841 [00:02<00:09, 1514.76it/s]Processing text_right with encode:  24%|██▍       | 4575/18841 [00:02<00:09, 1508.08it/s]Processing text_right with encode:  25%|██▌       | 4731/18841 [00:03<00:09, 1521.92it/s]Processing text_right with encode:  26%|██▌       | 4893/18841 [00:03<00:09, 1549.45it/s]Processing text_right with encode:  27%|██▋       | 5059/18841 [00:03<00:08, 1579.50it/s]Processing text_right with encode:  28%|██▊       | 5228/18841 [00:03<00:08, 1609.61it/s]Processing text_right with encode:  29%|██▊       | 5390/18841 [00:03<00:08, 1596.59it/s]Processing text_right with encode:  29%|██▉       | 5550/18841 [00:03<00:08, 1585.25it/s]Processing text_right with encode:  30%|███       | 5709/18841 [00:03<00:08, 1569.51it/s]Processing text_right with encode:  31%|███       | 5867/18841 [00:03<00:08, 1551.61it/s]Processing text_right with encode:  32%|███▏      | 6023/18841 [00:03<00:08, 1534.06it/s]Processing text_right with encode:  33%|███▎      | 6177/18841 [00:04<00:08, 1502.95it/s]Processing text_right with encode:  34%|███▎      | 6328/18841 [00:04<00:08, 1497.20it/s]Processing text_right with encode:  34%|███▍      | 6488/18841 [00:04<00:08, 1526.35it/s]Processing text_right with encode:  35%|███▌      | 6657/18841 [00:04<00:07, 1568.39it/s]Processing text_right with encode:  36%|███▌      | 6815/18841 [00:04<00:07, 1560.25it/s]Processing text_right with encode:  37%|███▋      | 6972/18841 [00:04<00:07, 1534.23it/s]Processing text_right with encode:  38%|███▊      | 7126/18841 [00:04<00:07, 1526.39it/s]Processing text_right with encode:  39%|███▊      | 7291/18841 [00:04<00:07, 1558.34it/s]Processing text_right with encode:  40%|███▉      | 7462/18841 [00:04<00:07, 1600.48it/s]Processing text_right with encode:  40%|████      | 7623/18841 [00:04<00:07, 1565.44it/s]Processing text_right with encode:  41%|████▏     | 7782/18841 [00:05<00:07, 1565.29it/s]Processing text_right with encode:  42%|████▏     | 7940/18841 [00:05<00:06, 1567.91it/s]Processing text_right with encode:  43%|████▎     | 8101/18841 [00:05<00:06, 1577.96it/s]Processing text_right with encode:  44%|████▍     | 8259/18841 [00:05<00:06, 1558.07it/s]Processing text_right with encode:  45%|████▍     | 8416/18841 [00:05<00:06, 1550.14it/s]Processing text_right with encode:  45%|████▌     | 8572/18841 [00:05<00:06, 1520.21it/s]Processing text_right with encode:  46%|████▋     | 8735/18841 [00:05<00:06, 1551.12it/s]Processing text_right with encode:  47%|████▋     | 8901/18841 [00:05<00:06, 1580.71it/s]Processing text_right with encode:  48%|████▊     | 9060/18841 [00:05<00:06, 1532.12it/s]Processing text_right with encode:  49%|████▉     | 9216/18841 [00:05<00:06, 1536.26it/s]Processing text_right with encode:  50%|████▉     | 9371/18841 [00:06<00:06, 1515.89it/s]Processing text_right with encode:  51%|█████     | 9534/18841 [00:06<00:06, 1545.60it/s]Processing text_right with encode:  51%|█████▏    | 9691/18841 [00:06<00:05, 1552.72it/s]Processing text_right with encode:  52%|█████▏    | 9847/18841 [00:06<00:05, 1533.14it/s]Processing text_right with encode:  53%|█████▎    | 10004/18841 [00:06<00:05, 1539.45it/s]Processing text_right with encode:  54%|█████▍    | 10161/18841 [00:06<00:05, 1547.85it/s]Processing text_right with encode:  55%|█████▍    | 10316/18841 [00:06<00:05, 1530.37it/s]Processing text_right with encode:  56%|█████▌    | 10501/18841 [00:06<00:05, 1612.87it/s]Processing text_right with encode:  57%|█████▋    | 10664/18841 [00:06<00:05, 1534.83it/s]Processing text_right with encode:  57%|█████▋    | 10820/18841 [00:06<00:05, 1514.11it/s]Processing text_right with encode:  58%|█████▊    | 10973/18841 [00:07<00:05, 1517.97it/s]Processing text_right with encode:  59%|█████▉    | 11126/18841 [00:07<00:05, 1471.98it/s]Processing text_right with encode:  60%|█████▉    | 11279/18841 [00:07<00:05, 1488.59it/s]Processing text_right with encode:  61%|██████    | 11435/18841 [00:07<00:04, 1506.77it/s]Processing text_right with encode:  61%|██████▏   | 11587/18841 [00:07<00:04, 1490.30it/s]Processing text_right with encode:  62%|██████▏   | 11741/18841 [00:07<00:04, 1503.74it/s]Processing text_right with encode:  63%|██████▎   | 11894/18841 [00:07<00:04, 1508.15it/s]Processing text_right with encode:  64%|██████▍   | 12059/18841 [00:07<00:04, 1547.91it/s]Processing text_right with encode:  65%|██████▍   | 12215/18841 [00:07<00:04, 1548.22it/s]Processing text_right with encode:  66%|██████▌   | 12371/18841 [00:08<00:04, 1535.35it/s]Processing text_right with encode:  66%|██████▋   | 12525/18841 [00:08<00:04, 1530.64it/s]Processing text_right with encode:  67%|██████▋   | 12679/18841 [00:08<00:04, 1521.52it/s]Processing text_right with encode:  68%|██████▊   | 12835/18841 [00:08<00:03, 1530.46it/s]Processing text_right with encode:  69%|██████▉   | 12997/18841 [00:08<00:03, 1556.04it/s]Processing text_right with encode:  70%|██████▉   | 13153/18841 [00:08<00:03, 1546.59it/s]Processing text_right with encode:  71%|███████   | 13308/18841 [00:08<00:03, 1539.98it/s]Processing text_right with encode:  71%|███████▏  | 13463/18841 [00:08<00:03, 1540.41it/s]Processing text_right with encode:  72%|███████▏  | 13631/18841 [00:08<00:03, 1579.71it/s]Processing text_right with encode:  73%|███████▎  | 13796/18841 [00:08<00:03, 1599.40it/s]Processing text_right with encode:  74%|███████▍  | 13957/18841 [00:09<00:03, 1593.59it/s]Processing text_right with encode:  75%|███████▍  | 14117/18841 [00:09<00:02, 1575.85it/s]Processing text_right with encode:  76%|███████▌  | 14275/18841 [00:09<00:02, 1552.55it/s]Processing text_right with encode:  77%|███████▋  | 14433/18841 [00:09<00:02, 1552.72it/s]Processing text_right with encode:  77%|███████▋  | 14589/18841 [00:09<00:02, 1537.43it/s]Processing text_right with encode:  78%|███████▊  | 14757/18841 [00:09<00:02, 1575.33it/s]Processing text_right with encode:  79%|███████▉  | 14915/18841 [00:09<00:02, 1555.13it/s]Processing text_right with encode:  80%|███████▉  | 15071/18841 [00:09<00:02, 1555.10it/s]Processing text_right with encode:  81%|████████  | 15227/18841 [00:09<00:02, 1556.12it/s]Processing text_right with encode:  82%|████████▏ | 15383/18841 [00:09<00:02, 1548.14it/s]Processing text_right with encode:  82%|████████▏ | 15541/18841 [00:10<00:02, 1551.83it/s]Processing text_right with encode:  83%|████████▎ | 15697/18841 [00:10<00:02, 1550.37it/s]Processing text_right with encode:  84%|████████▍ | 15859/18841 [00:10<00:01, 1570.25it/s]Processing text_right with encode:  85%|████████▌ | 16017/18841 [00:10<00:01, 1526.86it/s]Processing text_right with encode:  86%|████████▌ | 16171/18841 [00:10<00:01, 1475.30it/s]Processing text_right with encode:  87%|████████▋ | 16322/18841 [00:10<00:01, 1485.44it/s]Processing text_right with encode:  87%|████████▋ | 16485/18841 [00:10<00:01, 1523.02it/s]Processing text_right with encode:  88%|████████▊ | 16638/18841 [00:10<00:01, 1509.80it/s]Processing text_right with encode:  89%|████████▉ | 16791/18841 [00:10<00:01, 1513.44it/s]Processing text_right with encode:  90%|████████▉ | 16943/18841 [00:10<00:01, 1503.30it/s]Processing text_right with encode:  91%|█████████ | 17094/18841 [00:11<00:01, 1476.09it/s]Processing text_right with encode:  92%|█████████▏| 17242/18841 [00:11<00:01, 1429.33it/s]Processing text_right with encode:  92%|█████████▏| 17401/18841 [00:11<00:00, 1473.87it/s]Processing text_right with encode:  93%|█████████▎| 17558/18841 [00:11<00:00, 1500.56it/s]Processing text_right with encode:  94%|█████████▍| 17724/18841 [00:11<00:00, 1542.97it/s]Processing text_right with encode:  95%|█████████▍| 17880/18841 [00:11<00:00, 1507.89it/s]Processing text_right with encode:  96%|█████████▌| 18054/18841 [00:11<00:00, 1569.78it/s]Processing text_right with encode:  97%|█████████▋| 18221/18841 [00:11<00:00, 1590.18it/s]Processing text_right with encode:  98%|█████████▊| 18381/18841 [00:11<00:00, 1572.61it/s]Processing text_right with encode:  98%|█████████▊| 18558/18841 [00:12<00:00, 1625.17it/s]Processing text_right with encode:  99%|█████████▉| 18722/18841 [00:12<00:00, 1545.07it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:12<00:00, 1543.10it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 592038.38it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 718983.20it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  64%|██████▎   | 402/633 [00:00<00:00, 4013.97it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 3945.88it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 156/5961 [00:00<00:03, 1550.06it/s]Processing text_right with encode:   5%|▌         | 321/5961 [00:00<00:03, 1575.01it/s]Processing text_right with encode:   7%|▋         | 445/5961 [00:00<00:03, 1456.32it/s]Processing text_right with encode:  10%|█         | 601/5961 [00:00<00:03, 1477.62it/s]Processing text_right with encode:  13%|█▎        | 757/5961 [00:00<00:03, 1498.70it/s]Processing text_right with encode:  15%|█▌        | 897/5961 [00:00<00:03, 1463.96it/s]Processing text_right with encode:  18%|█▊        | 1046/5961 [00:00<00:03, 1471.66it/s]Processing text_right with encode:  20%|██        | 1203/5961 [00:00<00:03, 1499.68it/s]Processing text_right with encode:  23%|██▎       | 1369/5961 [00:00<00:02, 1543.58it/s]Processing text_right with encode:  26%|██▌       | 1527/5961 [00:01<00:02, 1552.16it/s]Processing text_right with encode:  28%|██▊       | 1679/5961 [00:01<00:02, 1542.19it/s]Processing text_right with encode:  31%|███       | 1831/5961 [00:01<00:02, 1515.45it/s]Processing text_right with encode:  33%|███▎      | 1981/5961 [00:01<00:02, 1498.54it/s]Processing text_right with encode:  36%|███▌      | 2140/5961 [00:01<00:02, 1522.68it/s]Processing text_right with encode:  39%|███▊      | 2301/5961 [00:01<00:02, 1546.71it/s]Processing text_right with encode:  41%|████      | 2456/5961 [00:01<00:02, 1496.44it/s]Processing text_right with encode:  44%|████▍     | 2617/5961 [00:01<00:02, 1528.45it/s]Processing text_right with encode:  47%|████▋     | 2781/5961 [00:01<00:02, 1560.25it/s]Processing text_right with encode:  49%|████▉     | 2938/5961 [00:01<00:01, 1537.13it/s]Processing text_right with encode:  52%|█████▏    | 3106/5961 [00:02<00:01, 1572.75it/s]Processing text_right with encode:  55%|█████▍    | 3264/5961 [00:02<00:01, 1573.04it/s]Processing text_right with encode:  57%|█████▋    | 3422/5961 [00:02<00:01, 1528.16it/s]Processing text_right with encode:  60%|█████▉    | 3576/5961 [00:02<00:01, 1512.45it/s]Processing text_right with encode:  63%|██████▎   | 3728/5961 [00:02<00:01, 1501.29it/s]Processing text_right with encode:  65%|██████▌   | 3895/5961 [00:02<00:01, 1546.66it/s]Processing text_right with encode:  68%|██████▊   | 4058/5961 [00:02<00:01, 1566.64it/s]Processing text_right with encode:  71%|███████   | 4216/5961 [00:02<00:01, 1562.91it/s]Processing text_right with encode:  73%|███████▎  | 4375/5961 [00:02<00:01, 1570.72it/s]Processing text_right with encode:  76%|███████▌  | 4533/5961 [00:02<00:00, 1551.79it/s]Processing text_right with encode:  79%|███████▉  | 4696/5961 [00:03<00:00, 1570.01it/s]Processing text_right with encode:  81%|████████▏ | 4854/5961 [00:03<00:00, 1545.46it/s]Processing text_right with encode:  84%|████████▍ | 5009/5961 [00:03<00:00, 1529.47it/s]Processing text_right with encode:  87%|████████▋ | 5165/5961 [00:03<00:00, 1522.20it/s]Processing text_right with encode:  89%|████████▉ | 5318/5961 [00:03<00:00, 1516.22it/s]Processing text_right with encode:  92%|█████████▏| 5470/5961 [00:03<00:00, 1502.35it/s]Processing text_right with encode:  94%|█████████▍| 5621/5961 [00:03<00:00, 1475.72it/s]Processing text_right with encode:  97%|█████████▋| 5769/5961 [00:03<00:00, 1476.00it/s]Processing text_right with encode: 100%|█████████▉| 5937/5961 [00:03<00:00, 1531.61it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1531.27it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 439568.61it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 678873.88it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:48<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:48<?, ?it/s, loss=0.980]Epoch 1/1:   1%|          | 1/102 [00:48<1:21:43, 48.55s/it, loss=0.980]Epoch 1/1:   1%|          | 1/102 [01:23<1:21:43, 48.55s/it, loss=0.980]Epoch 1/1:   1%|          | 1/102 [01:23<1:21:43, 48.55s/it, loss=0.898]Epoch 1/1:   2%|▏         | 2/102 [01:23<1:13:59, 44.40s/it, loss=0.898]Epoch 1/1:   2%|▏         | 2/102 [01:48<1:13:59, 44.40s/it, loss=0.898]Epoch 1/1:   2%|▏         | 2/102 [01:48<1:13:59, 44.40s/it, loss=0.992]Epoch 1/1:   3%|▎         | 3/102 [01:48<1:03:36, 38.56s/it, loss=0.992]Epoch 1/1:   3%|▎         | 3/102 [03:24<1:03:36, 38.56s/it, loss=0.992]Epoch 1/1:   3%|▎         | 3/102 [03:24<1:03:36, 38.56s/it, loss=1.031]Epoch 1/1:   4%|▍         | 4/102 [03:24<1:31:03, 55.75s/it, loss=1.031]Epoch 1/1:   4%|▍         | 4/102 [04:15<1:31:03, 55.75s/it, loss=1.031]Epoch 1/1:   4%|▍         | 4/102 [04:15<1:31:03, 55.75s/it, loss=0.848]Epoch 1/1:   5%|▍         | 5/102 [04:15<1:28:15, 54.59s/it, loss=0.848]Epoch 1/1:   5%|▍         | 5/102 [06:03<1:28:15, 54.59s/it, loss=0.848]Epoch 1/1:   5%|▍         | 5/102 [06:03<1:28:15, 54.59s/it, loss=1.026]Epoch 1/1:   6%|▌         | 6/102 [06:03<1:52:45, 70.47s/it, loss=1.026]Epoch 1/1:   6%|▌         | 6/102 [07:02<1:52:45, 70.47s/it, loss=1.026]Epoch 1/1:   6%|▌         | 6/102 [07:02<1:52:45, 70.47s/it, loss=1.040]Epoch 1/1:   7%|▋         | 7/102 [07:02<1:46:19, 67.15s/it, loss=1.040]Epoch 1/1:   7%|▋         | 7/102 [09:27<1:46:19, 67.15s/it, loss=1.040]Epoch 1/1:   7%|▋         | 7/102 [09:27<1:46:19, 67.15s/it, loss=0.862]Epoch 1/1:   8%|▊         | 8/102 [09:27<2:21:30, 90.32s/it, loss=0.862]Epoch 1/1:   8%|▊         | 8/102 [10:35<2:21:30, 90.32s/it, loss=0.862]Epoch 1/1:   8%|▊         | 8/102 [10:35<2:21:30, 90.32s/it, loss=0.906]Epoch 1/1:   9%|▉         | 9/102 [10:35<2:09:51, 83.78s/it, loss=0.906]Epoch 1/1:   9%|▉         | 9/102 [13:24<2:09:51, 83.78s/it, loss=0.906]Epoch 1/1:   9%|▉         | 9/102 [13:24<2:09:51, 83.78s/it, loss=0.608]Epoch 1/1:  10%|▉         | 10/102 [13:24<2:47:39, 109.34s/it, loss=0.608]Epoch 1/1:  10%|▉         | 10/102 [14:19<2:47:39, 109.34s/it, loss=0.608]Epoch 1/1:  10%|▉         | 10/102 [14:19<2:47:39, 109.34s/it, loss=0.869]Epoch 1/1:  11%|█         | 11/102 [14:19<2:20:57, 92.94s/it, loss=0.869] Epoch 1/1:  11%|█         | 11/102 [16:44<2:20:57, 92.94s/it, loss=0.869]Epoch 1/1:  11%|█         | 11/102 [16:44<2:20:57, 92.94s/it, loss=0.565]Epoch 1/1:  12%|█▏        | 12/102 [16:44<2:42:47, 108.53s/it, loss=0.565]Killed

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
[master 288da6d] ml_store  && git pull --all
 1 file changed, 58 insertions(+)
To github.com:arita37/mlmodels_store.git
 + a1975e6...288da6d master -> master (forced update)





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'dataset/vision/MNIST/', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/MNIST/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/MNIST/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f439fa19e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f439fa19e18>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f439fa19e18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:21, 121752.38it/s] 48%|████▊     | 4710400/9912422 [00:00<00:29, 173738.07it/s]9920512it [00:00, 33450901.41it/s]                           
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 256011.40it/s]           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:13, 122813.58it/s]1654784it [00:00, 9130018.10it/s]                          
0it [00:00, ?it/s]8192it [00:00, 188784.58it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f439c076b70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f439c076b70>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f439c076b70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.003322386513153712 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.01802121353149414 	 Accuracy: 4
model saves at 4 accuracy
Train Epoch: 2 	 Loss: 0.002014933447043101 	 Accuracy: 1
Train Epoch: 2 	 Loss: 0.017052953004837036 	 Accuracy: 4
Train Epoch: 3 	 Loss: 0.0017688941061496735 	 Accuracy: 1
Train Epoch: 3 	 Loss: 0.022381241917610168 	 Accuracy: 4
Train Epoch: 4 	 Loss: 0.001778024325768153 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.014828021824359893 	 Accuracy: 6
model saves at 6 accuracy
Train Epoch: 5 	 Loss: 0.0017910289565722147 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.010188598692417145 	 Accuracy: 7
model saves at 7 accuracy

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f439c076950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f439c076950>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f439c076950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/MNIST/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7f439f7ce7b8>

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
<__main__.Model object at 0x7f439c176b00>

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
[master 458086d] ml_store  && git pull --all
 1 file changed, 156 insertions(+)
To github.com:arita37/mlmodels_store.git
   288da6d..458086d  master -> master





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
[master 070fd98] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   458086d..070fd98  master -> master





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
