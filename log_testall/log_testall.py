
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '458b0439a169873cbce08726558e091efacd7d2f', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/458b0439a169873cbce08726558e091efacd7d2f

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/458b0439a169873cbce08726558e091efacd7d2f

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
[master 8f279dd] ml_store  && git pull --all
 2 files changed, 1 insertion(+), 53 deletions(-)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   ee7cdc2..8f279dd  master -> master





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
[master e5cdd98] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   8f279dd..e5cdd98  master -> master





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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-27 04:13:38.829127: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-27 04:13:38.833405: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-27 04:13:38.833555: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ebb3a978a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 04:13:38.833569: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 153
Trainable params: 153
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2546 - binary_crossentropy: 0.8070 - val_loss: 0.2526 - val_binary_crossentropy: 0.7507

  #### metrics   #################################################### 
{'MSE': 0.25337312346936325}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         8           sequence_max[0][0]               
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
Total params: 153
Trainable params: 153
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
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
Total params: 448
Trainable params: 448
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2473 - binary_crossentropy: 0.8180500/500 [==============================] - 1s 1ms/sample - loss: 0.2538 - binary_crossentropy: 0.7270 - val_loss: 0.2563 - val_binary_crossentropy: 0.7059

  #### metrics   #################################################### 
{'MSE': 0.2547759768907606}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_max[0][0]               
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
Total params: 448
Trainable params: 448
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2500 - binary_crossentropy: 0.6932 - val_loss: 0.2499 - val_binary_crossentropy: 0.6930

  #### metrics   #################################################### 
{'MSE': 0.24984578968842602}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 328
Trainable params: 328
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.6200 - binary_crossentropy: 9.5582500/500 [==============================] - 1s 2ms/sample - loss: 0.5160 - binary_crossentropy: 7.9537 - val_loss: 0.5000 - val_binary_crossentropy: 7.7125

  #### metrics   #################################################### 
{'MSE': 0.508}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 328
Trainable params: 328
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.4400 - binary_crossentropy: 6.7870500/500 [==============================] - 2s 3ms/sample - loss: 0.5000 - binary_crossentropy: 7.7125 - val_loss: 0.4900 - val_binary_crossentropy: 7.5582

  #### metrics   #################################################### 
{'MSE': 0.495}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-27 04:14:50.030760: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:14:50.032505: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:14:50.039307: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 04:14:50.049812: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 04:14:50.051647: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 04:14:50.053106: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:14:50.054573: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2494 - val_binary_crossentropy: 0.6920
2020-05-27 04:14:51.221436: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:14:51.223174: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:14:51.227404: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 04:14:51.234856: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 04:14:51.236157: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 04:14:51.237339: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:14:51.238828: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24912335831243126}

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
2020-05-27 04:15:12.784974: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:12.786594: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:12.790532: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 04:15:12.799236: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 04:15:12.800314: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 04:15:12.801466: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:12.803389: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2509 - val_binary_crossentropy: 0.6950
2020-05-27 04:15:14.261500: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:14.262576: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:14.264962: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 04:15:14.269821: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 04:15:14.270820: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 04:15:14.271529: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:14.272185: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25116511551919807}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-27 04:15:45.311458: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:45.315601: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:45.329497: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 04:15:45.356415: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 04:15:45.360184: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 04:15:45.364044: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:45.367610: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 4s 4s/sample - loss: 0.2454 - binary_crossentropy: 0.6839 - val_loss: 0.2503 - val_binary_crossentropy: 0.6937
2020-05-27 04:15:47.513575: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:47.518199: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:47.529729: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 04:15:47.554726: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 04:15:47.558939: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 04:15:47.563105: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 04:15:47.567551: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24384994570709514}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
Total params: 695
Trainable params: 695
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3551 - binary_crossentropy: 1.7380500/500 [==============================] - 4s 8ms/sample - loss: 0.3294 - binary_crossentropy: 1.4356 - val_loss: 0.3237 - val_binary_crossentropy: 1.2925

  #### metrics   #################################################### 
{'MSE': 0.3253158325561005}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 212
Trainable params: 212
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3753 - binary_crossentropy: 1.1871500/500 [==============================] - 4s 8ms/sample - loss: 0.3707 - binary_crossentropy: 1.0991 - val_loss: 0.3447 - val_binary_crossentropy: 1.0068

  #### metrics   #################################################### 
{'MSE': 0.35269592032567165}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         4           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 212
Trainable params: 212
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 1,849
Trainable params: 1,849
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 5s - loss: 0.4700 - binary_crossentropy: 7.2427500/500 [==============================] - 4s 8ms/sample - loss: 0.4480 - binary_crossentropy: 6.9026 - val_loss: 0.5240 - val_binary_crossentropy: 8.0827

  #### metrics   #################################################### 
{'MSE': 0.487}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 1,849
Trainable params: 1,849
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
regionsequence_sum (InputLayer) [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         2           regionsequence_max[0][0]         
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
Total params: 96
Trainable params: 96
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2491 - binary_crossentropy: 0.8140500/500 [==============================] - 6s 11ms/sample - loss: 0.2528 - binary_crossentropy: 0.7984 - val_loss: 0.2574 - val_binary_crossentropy: 0.9144

  #### metrics   #################################################### 
{'MSE': 0.25484961008127727}

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
regionsequence_sum (InputLayer) [(None, 5)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         2           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         5           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         2           regionsequence_max[0][0]         
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
Total params: 96
Trainable params: 96
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
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
Total params: 1,397
Trainable params: 1,397
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.3340 - binary_crossentropy: 0.9083500/500 [==============================] - 5s 10ms/sample - loss: 0.2985 - binary_crossentropy: 0.8236 - val_loss: 0.3006 - val_binary_crossentropy: 0.8470

  #### metrics   #################################################### 
{'MSE': 0.29600507822177424}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_11[0][0]                    
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
Total params: 2,933
Trainable params: 2,853
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2390 - binary_crossentropy: 0.6714500/500 [==============================] - 6s 11ms/sample - loss: 0.2663 - binary_crossentropy: 0.8063 - val_loss: 0.2634 - val_binary_crossentropy: 0.8522

  #### metrics   #################################################### 
{'MSE': 0.26142543914111843}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_11[0][0]                    
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
Total params: 2,933
Trainable params: 2,853
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
[master 253721e] ml_store  && git pull --all
 1 file changed, 4945 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 224f417...253721e master -> master (forced update)





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
[master fe08d2d] ml_store  && git pull --all
 1 file changed, 49 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   253721e..fe08d2d  master -> master





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
[master 89412e8] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   fe08d2d..89412e8  master -> master





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
[master c426d5a] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   89412e8..c426d5a  master -> master





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
[master 29ad168] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   c426d5a..29ad168  master -> master





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
[master fe39431] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   29ad168..fe39431  master -> master





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
[master ffff33a] ml_store  && git pull --all
 1 file changed, 43 insertions(+)
To github.com:arita37/mlmodels_store.git
   fe39431..ffff33a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 3268608/17464789 [====>.........................] - ETA: 0s
 9388032/17464789 [===============>..............] - ETA: 0s
16474112/17464789 [===========================>..] - ETA: 0s
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
2020-05-27 04:24:50.885217: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-27 04:24:50.888794: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-27 04:24:50.888926: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556881a9a3b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 04:24:50.888940: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7893 - accuracy: 0.4920 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7382 - accuracy: 0.4953
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7471 - accuracy: 0.4947
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7801 - accuracy: 0.4926
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7586 - accuracy: 0.4940
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6863 - accuracy: 0.4987
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6609 - accuracy: 0.5004
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6905 - accuracy: 0.4984
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
11000/25000 [============>.................] - ETA: 4s - loss: 7.7126 - accuracy: 0.4970
12000/25000 [=============>................] - ETA: 3s - loss: 7.7420 - accuracy: 0.4951
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7197 - accuracy: 0.4965
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6995 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 2s - loss: 7.7147 - accuracy: 0.4969
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7241 - accuracy: 0.4963
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7189 - accuracy: 0.4966
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7135 - accuracy: 0.4969
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7086 - accuracy: 0.4973
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7042 - accuracy: 0.4976
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6856 - accuracy: 0.4988
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6917 - accuracy: 0.4984
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 9s 351us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f057b41f1d0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f057ea17470> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 7.6053 - accuracy: 0.5040
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7816 - accuracy: 0.4925 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6206 - accuracy: 0.5030
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.4857 - accuracy: 0.5118
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5184 - accuracy: 0.5097
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5812 - accuracy: 0.5056
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6110 - accuracy: 0.5036
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6189 - accuracy: 0.5031
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6145 - accuracy: 0.5034
11000/25000 [============>.................] - ETA: 4s - loss: 7.6624 - accuracy: 0.5003
12000/25000 [=============>................] - ETA: 3s - loss: 7.6832 - accuracy: 0.4989
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6699 - accuracy: 0.4998
15000/25000 [=================>............] - ETA: 2s - loss: 7.6738 - accuracy: 0.4995
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6561 - accuracy: 0.5007
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6793 - accuracy: 0.4992
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6785 - accuracy: 0.4992
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6779 - accuracy: 0.4993
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6804 - accuracy: 0.4991
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6732 - accuracy: 0.4996
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6813 - accuracy: 0.4990
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6646 - accuracy: 0.5001
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 9s 349us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9120 - accuracy: 0.4840 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.7995 - accuracy: 0.4913
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7165 - accuracy: 0.4967
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6544 - accuracy: 0.5008
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6104 - accuracy: 0.5037
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6097 - accuracy: 0.5037
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6149 - accuracy: 0.5034
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6308 - accuracy: 0.5023
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6114 - accuracy: 0.5036
11000/25000 [============>.................] - ETA: 4s - loss: 7.5844 - accuracy: 0.5054
12000/25000 [=============>................] - ETA: 3s - loss: 7.5797 - accuracy: 0.5057
13000/25000 [==============>...............] - ETA: 3s - loss: 7.5841 - accuracy: 0.5054
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6119 - accuracy: 0.5036
15000/25000 [=================>............] - ETA: 2s - loss: 7.6247 - accuracy: 0.5027
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6455 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6693 - accuracy: 0.4998
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6734 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6747 - accuracy: 0.4995
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6850 - accuracy: 0.4988
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6710 - accuracy: 0.4997
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6631 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6720 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6762 - accuracy: 0.4994
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
[master de9f8ae] ml_store  && git pull --all
 1 file changed, 316 insertions(+)
To github.com:arita37/mlmodels_store.git
   ffff33a..de9f8ae  master -> master





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

13/13 [==============================] - 1s 99ms/step - loss: nan
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
[master 681841f] ml_store  && git pull --all
 1 file changed, 124 insertions(+)
To github.com:arita37/mlmodels_store.git
   de9f8ae..681841f  master -> master





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
 2637824/11490434 [=====>........................] - ETA: 0s
 6348800/11490434 [===============>..............] - ETA: 0s
 8945664/11490434 [======================>.......] - ETA: 0s
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

   32/60000 [..............................] - ETA: 6:56 - loss: 2.3188 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:24 - loss: 2.2982 - categorical_accuracy: 0.1094
   96/60000 [..............................] - ETA: 3:31 - loss: 2.2939 - categorical_accuracy: 0.0938
  128/60000 [..............................] - ETA: 3:03 - loss: 2.2579 - categorical_accuracy: 0.1172
  160/60000 [..............................] - ETA: 2:47 - loss: 2.2104 - categorical_accuracy: 0.1500
  224/60000 [..............................] - ETA: 2:26 - loss: 2.0819 - categorical_accuracy: 0.2098
  256/60000 [..............................] - ETA: 2:22 - loss: 2.0692 - categorical_accuracy: 0.2305
  288/60000 [..............................] - ETA: 2:18 - loss: 2.0464 - categorical_accuracy: 0.2431
  320/60000 [..............................] - ETA: 2:13 - loss: 2.0347 - categorical_accuracy: 0.2656
  352/60000 [..............................] - ETA: 2:11 - loss: 2.0120 - categorical_accuracy: 0.2812
  384/60000 [..............................] - ETA: 2:08 - loss: 1.9698 - categorical_accuracy: 0.3073
  416/60000 [..............................] - ETA: 2:06 - loss: 1.9529 - categorical_accuracy: 0.3101
  448/60000 [..............................] - ETA: 2:04 - loss: 1.9289 - categorical_accuracy: 0.3170
  480/60000 [..............................] - ETA: 2:03 - loss: 1.8991 - categorical_accuracy: 0.3333
  512/60000 [..............................] - ETA: 2:02 - loss: 1.8574 - categorical_accuracy: 0.3516
  544/60000 [..............................] - ETA: 2:00 - loss: 1.8181 - categorical_accuracy: 0.3640
  576/60000 [..............................] - ETA: 1:59 - loss: 1.7828 - categorical_accuracy: 0.3733
  608/60000 [..............................] - ETA: 1:58 - loss: 1.7349 - categorical_accuracy: 0.3914
  640/60000 [..............................] - ETA: 1:56 - loss: 1.6941 - categorical_accuracy: 0.4000
  672/60000 [..............................] - ETA: 1:55 - loss: 1.6697 - categorical_accuracy: 0.4107
  704/60000 [..............................] - ETA: 1:54 - loss: 1.6499 - categorical_accuracy: 0.4205
  736/60000 [..............................] - ETA: 1:53 - loss: 1.6269 - categorical_accuracy: 0.4293
  768/60000 [..............................] - ETA: 1:53 - loss: 1.6012 - categorical_accuracy: 0.4401
  800/60000 [..............................] - ETA: 1:52 - loss: 1.5784 - categorical_accuracy: 0.4512
  832/60000 [..............................] - ETA: 1:51 - loss: 1.5566 - categorical_accuracy: 0.4579
  864/60000 [..............................] - ETA: 1:51 - loss: 1.5330 - categorical_accuracy: 0.4653
  896/60000 [..............................] - ETA: 1:51 - loss: 1.5139 - categorical_accuracy: 0.4743
  928/60000 [..............................] - ETA: 1:50 - loss: 1.4915 - categorical_accuracy: 0.4828
  960/60000 [..............................] - ETA: 1:50 - loss: 1.4600 - categorical_accuracy: 0.4948
  992/60000 [..............................] - ETA: 1:49 - loss: 1.4363 - categorical_accuracy: 0.5000
 1024/60000 [..............................] - ETA: 1:49 - loss: 1.4111 - categorical_accuracy: 0.5059
 1056/60000 [..............................] - ETA: 1:48 - loss: 1.4016 - categorical_accuracy: 0.5114
 1088/60000 [..............................] - ETA: 1:48 - loss: 1.3939 - categorical_accuracy: 0.5165
 1120/60000 [..............................] - ETA: 1:48 - loss: 1.3782 - categorical_accuracy: 0.5232
 1152/60000 [..............................] - ETA: 1:47 - loss: 1.3645 - categorical_accuracy: 0.5269
 1184/60000 [..............................] - ETA: 1:47 - loss: 1.3482 - categorical_accuracy: 0.5338
 1216/60000 [..............................] - ETA: 1:47 - loss: 1.3254 - categorical_accuracy: 0.5411
 1248/60000 [..............................] - ETA: 1:46 - loss: 1.3124 - categorical_accuracy: 0.5473
 1280/60000 [..............................] - ETA: 1:46 - loss: 1.3001 - categorical_accuracy: 0.5531
 1312/60000 [..............................] - ETA: 1:46 - loss: 1.2936 - categorical_accuracy: 0.5556
 1344/60000 [..............................] - ETA: 1:45 - loss: 1.2760 - categorical_accuracy: 0.5647
 1376/60000 [..............................] - ETA: 1:45 - loss: 1.2650 - categorical_accuracy: 0.5690
 1408/60000 [..............................] - ETA: 1:45 - loss: 1.2588 - categorical_accuracy: 0.5696
 1440/60000 [..............................] - ETA: 1:45 - loss: 1.2437 - categorical_accuracy: 0.5750
 1472/60000 [..............................] - ETA: 1:44 - loss: 1.2328 - categorical_accuracy: 0.5781
 1504/60000 [..............................] - ETA: 1:44 - loss: 1.2197 - categorical_accuracy: 0.5831
 1536/60000 [..............................] - ETA: 1:44 - loss: 1.2148 - categorical_accuracy: 0.5853
 1568/60000 [..............................] - ETA: 1:44 - loss: 1.2059 - categorical_accuracy: 0.5899
 1600/60000 [..............................] - ETA: 1:43 - loss: 1.1955 - categorical_accuracy: 0.5938
 1632/60000 [..............................] - ETA: 1:43 - loss: 1.1848 - categorical_accuracy: 0.5980
 1664/60000 [..............................] - ETA: 1:43 - loss: 1.1743 - categorical_accuracy: 0.6022
 1696/60000 [..............................] - ETA: 1:43 - loss: 1.1654 - categorical_accuracy: 0.6044
 1728/60000 [..............................] - ETA: 1:43 - loss: 1.1524 - categorical_accuracy: 0.6076
 1760/60000 [..............................] - ETA: 1:43 - loss: 1.1398 - categorical_accuracy: 0.6136
 1792/60000 [..............................] - ETA: 1:43 - loss: 1.1257 - categorical_accuracy: 0.6189
 1824/60000 [..............................] - ETA: 1:43 - loss: 1.1156 - categorical_accuracy: 0.6217
 1856/60000 [..............................] - ETA: 1:43 - loss: 1.1058 - categorical_accuracy: 0.6245
 1888/60000 [..............................] - ETA: 1:42 - loss: 1.0956 - categorical_accuracy: 0.6282
 1920/60000 [..............................] - ETA: 1:42 - loss: 1.0874 - categorical_accuracy: 0.6307
 1952/60000 [..............................] - ETA: 1:42 - loss: 1.0868 - categorical_accuracy: 0.6332
 1984/60000 [..............................] - ETA: 1:42 - loss: 1.0792 - categorical_accuracy: 0.6366
 2048/60000 [>.............................] - ETA: 1:42 - loss: 1.0654 - categorical_accuracy: 0.6406
 2112/60000 [>.............................] - ETA: 1:41 - loss: 1.0497 - categorical_accuracy: 0.6473
 2144/60000 [>.............................] - ETA: 1:41 - loss: 1.0428 - categorical_accuracy: 0.6497
 2176/60000 [>.............................] - ETA: 1:41 - loss: 1.0379 - categorical_accuracy: 0.6507
 2208/60000 [>.............................] - ETA: 1:41 - loss: 1.0325 - categorical_accuracy: 0.6526
 2240/60000 [>.............................] - ETA: 1:41 - loss: 1.0220 - categorical_accuracy: 0.6562
 2272/60000 [>.............................] - ETA: 1:41 - loss: 1.0124 - categorical_accuracy: 0.6593
 2304/60000 [>.............................] - ETA: 1:41 - loss: 1.0024 - categorical_accuracy: 0.6632
 2336/60000 [>.............................] - ETA: 1:40 - loss: 0.9979 - categorical_accuracy: 0.6648
 2368/60000 [>.............................] - ETA: 1:40 - loss: 0.9907 - categorical_accuracy: 0.6677
 2400/60000 [>.............................] - ETA: 1:40 - loss: 0.9809 - categorical_accuracy: 0.6712
 2432/60000 [>.............................] - ETA: 1:40 - loss: 0.9777 - categorical_accuracy: 0.6719
 2464/60000 [>.............................] - ETA: 1:40 - loss: 0.9698 - categorical_accuracy: 0.6745
 2496/60000 [>.............................] - ETA: 1:40 - loss: 0.9702 - categorical_accuracy: 0.6747
 2528/60000 [>.............................] - ETA: 1:40 - loss: 0.9652 - categorical_accuracy: 0.6772
 2560/60000 [>.............................] - ETA: 1:40 - loss: 0.9552 - categorical_accuracy: 0.6812
 2592/60000 [>.............................] - ETA: 1:40 - loss: 0.9490 - categorical_accuracy: 0.6833
 2624/60000 [>.............................] - ETA: 1:39 - loss: 0.9446 - categorical_accuracy: 0.6856
 2688/60000 [>.............................] - ETA: 1:39 - loss: 0.9308 - categorical_accuracy: 0.6905
 2720/60000 [>.............................] - ETA: 1:39 - loss: 0.9272 - categorical_accuracy: 0.6904
 2752/60000 [>.............................] - ETA: 1:39 - loss: 0.9201 - categorical_accuracy: 0.6926
 2784/60000 [>.............................] - ETA: 1:39 - loss: 0.9151 - categorical_accuracy: 0.6943
 2816/60000 [>.............................] - ETA: 1:39 - loss: 0.9106 - categorical_accuracy: 0.6960
 2848/60000 [>.............................] - ETA: 1:39 - loss: 0.9095 - categorical_accuracy: 0.6966
 2880/60000 [>.............................] - ETA: 1:39 - loss: 0.9078 - categorical_accuracy: 0.6976
 2912/60000 [>.............................] - ETA: 1:39 - loss: 0.9020 - categorical_accuracy: 0.6999
 2944/60000 [>.............................] - ETA: 1:38 - loss: 0.8976 - categorical_accuracy: 0.7014
 2976/60000 [>.............................] - ETA: 1:38 - loss: 0.8923 - categorical_accuracy: 0.7040
 3008/60000 [>.............................] - ETA: 1:38 - loss: 0.8849 - categorical_accuracy: 0.7064
 3040/60000 [>.............................] - ETA: 1:38 - loss: 0.8814 - categorical_accuracy: 0.7076
 3072/60000 [>.............................] - ETA: 1:38 - loss: 0.8741 - categorical_accuracy: 0.7100
 3104/60000 [>.............................] - ETA: 1:38 - loss: 0.8695 - categorical_accuracy: 0.7110
 3136/60000 [>.............................] - ETA: 1:38 - loss: 0.8647 - categorical_accuracy: 0.7124
 3168/60000 [>.............................] - ETA: 1:38 - loss: 0.8615 - categorical_accuracy: 0.7140
 3200/60000 [>.............................] - ETA: 1:38 - loss: 0.8588 - categorical_accuracy: 0.7153
 3232/60000 [>.............................] - ETA: 1:38 - loss: 0.8545 - categorical_accuracy: 0.7169
 3264/60000 [>.............................] - ETA: 1:37 - loss: 0.8485 - categorical_accuracy: 0.7194
 3296/60000 [>.............................] - ETA: 1:37 - loss: 0.8442 - categorical_accuracy: 0.7203
 3328/60000 [>.............................] - ETA: 1:37 - loss: 0.8381 - categorical_accuracy: 0.7218
 3360/60000 [>.............................] - ETA: 1:37 - loss: 0.8340 - categorical_accuracy: 0.7232
 3392/60000 [>.............................] - ETA: 1:37 - loss: 0.8355 - categorical_accuracy: 0.7235
 3424/60000 [>.............................] - ETA: 1:37 - loss: 0.8322 - categorical_accuracy: 0.7249
 3456/60000 [>.............................] - ETA: 1:37 - loss: 0.8274 - categorical_accuracy: 0.7260
 3488/60000 [>.............................] - ETA: 1:37 - loss: 0.8229 - categorical_accuracy: 0.7276
 3520/60000 [>.............................] - ETA: 1:37 - loss: 0.8189 - categorical_accuracy: 0.7290
 3552/60000 [>.............................] - ETA: 1:37 - loss: 0.8147 - categorical_accuracy: 0.7309
 3584/60000 [>.............................] - ETA: 1:37 - loss: 0.8094 - categorical_accuracy: 0.7327
 3616/60000 [>.............................] - ETA: 1:37 - loss: 0.8062 - categorical_accuracy: 0.7337
 3648/60000 [>.............................] - ETA: 1:37 - loss: 0.8016 - categorical_accuracy: 0.7352
 3680/60000 [>.............................] - ETA: 1:37 - loss: 0.7969 - categorical_accuracy: 0.7367
 3712/60000 [>.............................] - ETA: 1:37 - loss: 0.7942 - categorical_accuracy: 0.7373
 3744/60000 [>.............................] - ETA: 1:36 - loss: 0.7912 - categorical_accuracy: 0.7382
 3776/60000 [>.............................] - ETA: 1:36 - loss: 0.7889 - categorical_accuracy: 0.7397
 3808/60000 [>.............................] - ETA: 1:36 - loss: 0.7852 - categorical_accuracy: 0.7408
 3840/60000 [>.............................] - ETA: 1:36 - loss: 0.7826 - categorical_accuracy: 0.7419
 3872/60000 [>.............................] - ETA: 1:36 - loss: 0.7785 - categorical_accuracy: 0.7430
 3904/60000 [>.............................] - ETA: 1:36 - loss: 0.7746 - categorical_accuracy: 0.7444
 3936/60000 [>.............................] - ETA: 1:36 - loss: 0.7714 - categorical_accuracy: 0.7454
 3968/60000 [>.............................] - ETA: 1:36 - loss: 0.7684 - categorical_accuracy: 0.7462
 4000/60000 [=>............................] - ETA: 1:36 - loss: 0.7653 - categorical_accuracy: 0.7475
 4032/60000 [=>............................] - ETA: 1:36 - loss: 0.7619 - categorical_accuracy: 0.7488
 4064/60000 [=>............................] - ETA: 1:36 - loss: 0.7586 - categorical_accuracy: 0.7498
 4096/60000 [=>............................] - ETA: 1:36 - loss: 0.7576 - categorical_accuracy: 0.7505
 4128/60000 [=>............................] - ETA: 1:36 - loss: 0.7534 - categorical_accuracy: 0.7519
 4160/60000 [=>............................] - ETA: 1:36 - loss: 0.7494 - categorical_accuracy: 0.7534
 4192/60000 [=>............................] - ETA: 1:36 - loss: 0.7467 - categorical_accuracy: 0.7543
 4224/60000 [=>............................] - ETA: 1:35 - loss: 0.7426 - categorical_accuracy: 0.7557
 4256/60000 [=>............................] - ETA: 1:35 - loss: 0.7390 - categorical_accuracy: 0.7568
 4288/60000 [=>............................] - ETA: 1:35 - loss: 0.7356 - categorical_accuracy: 0.7577
 4320/60000 [=>............................] - ETA: 1:35 - loss: 0.7318 - categorical_accuracy: 0.7588
 4352/60000 [=>............................] - ETA: 1:35 - loss: 0.7275 - categorical_accuracy: 0.7603
 4384/60000 [=>............................] - ETA: 1:35 - loss: 0.7261 - categorical_accuracy: 0.7614
 4416/60000 [=>............................] - ETA: 1:35 - loss: 0.7227 - categorical_accuracy: 0.7627
 4448/60000 [=>............................] - ETA: 1:35 - loss: 0.7208 - categorical_accuracy: 0.7637
 4480/60000 [=>............................] - ETA: 1:35 - loss: 0.7200 - categorical_accuracy: 0.7641
 4512/60000 [=>............................] - ETA: 1:35 - loss: 0.7163 - categorical_accuracy: 0.7651
 4544/60000 [=>............................] - ETA: 1:35 - loss: 0.7129 - categorical_accuracy: 0.7663
 4576/60000 [=>............................] - ETA: 1:35 - loss: 0.7106 - categorical_accuracy: 0.7673
 4608/60000 [=>............................] - ETA: 1:35 - loss: 0.7071 - categorical_accuracy: 0.7684
 4640/60000 [=>............................] - ETA: 1:34 - loss: 0.7046 - categorical_accuracy: 0.7694
 4672/60000 [=>............................] - ETA: 1:34 - loss: 0.7037 - categorical_accuracy: 0.7705
 4704/60000 [=>............................] - ETA: 1:34 - loss: 0.7004 - categorical_accuracy: 0.7717
 4736/60000 [=>............................] - ETA: 1:34 - loss: 0.6975 - categorical_accuracy: 0.7726
 4768/60000 [=>............................] - ETA: 1:34 - loss: 0.6940 - categorical_accuracy: 0.7737
 4800/60000 [=>............................] - ETA: 1:34 - loss: 0.6915 - categorical_accuracy: 0.7744
 4832/60000 [=>............................] - ETA: 1:34 - loss: 0.6900 - categorical_accuracy: 0.7748
 4864/60000 [=>............................] - ETA: 1:34 - loss: 0.6876 - categorical_accuracy: 0.7753
 4896/60000 [=>............................] - ETA: 1:34 - loss: 0.6865 - categorical_accuracy: 0.7755
 4928/60000 [=>............................] - ETA: 1:34 - loss: 0.6830 - categorical_accuracy: 0.7766
 4960/60000 [=>............................] - ETA: 1:34 - loss: 0.6817 - categorical_accuracy: 0.7770
 4992/60000 [=>............................] - ETA: 1:34 - loss: 0.6790 - categorical_accuracy: 0.7778
 5024/60000 [=>............................] - ETA: 1:34 - loss: 0.6760 - categorical_accuracy: 0.7791
 5056/60000 [=>............................] - ETA: 1:34 - loss: 0.6731 - categorical_accuracy: 0.7797
 5088/60000 [=>............................] - ETA: 1:34 - loss: 0.6707 - categorical_accuracy: 0.7805
 5152/60000 [=>............................] - ETA: 1:34 - loss: 0.6656 - categorical_accuracy: 0.7824
 5184/60000 [=>............................] - ETA: 1:34 - loss: 0.6643 - categorical_accuracy: 0.7834
 5216/60000 [=>............................] - ETA: 1:34 - loss: 0.6630 - categorical_accuracy: 0.7837
 5248/60000 [=>............................] - ETA: 1:34 - loss: 0.6629 - categorical_accuracy: 0.7843
 5280/60000 [=>............................] - ETA: 1:34 - loss: 0.6616 - categorical_accuracy: 0.7847
 5312/60000 [=>............................] - ETA: 1:33 - loss: 0.6584 - categorical_accuracy: 0.7858
 5344/60000 [=>............................] - ETA: 1:33 - loss: 0.6561 - categorical_accuracy: 0.7863
 5376/60000 [=>............................] - ETA: 1:33 - loss: 0.6533 - categorical_accuracy: 0.7874
 5408/60000 [=>............................] - ETA: 1:33 - loss: 0.6517 - categorical_accuracy: 0.7883
 5440/60000 [=>............................] - ETA: 1:33 - loss: 0.6499 - categorical_accuracy: 0.7890
 5472/60000 [=>............................] - ETA: 1:33 - loss: 0.6483 - categorical_accuracy: 0.7893
 5504/60000 [=>............................] - ETA: 1:33 - loss: 0.6457 - categorical_accuracy: 0.7900
 5536/60000 [=>............................] - ETA: 1:33 - loss: 0.6432 - categorical_accuracy: 0.7905
 5568/60000 [=>............................] - ETA: 1:33 - loss: 0.6417 - categorical_accuracy: 0.7906
 5600/60000 [=>............................] - ETA: 1:33 - loss: 0.6408 - categorical_accuracy: 0.7909
 5632/60000 [=>............................] - ETA: 1:33 - loss: 0.6395 - categorical_accuracy: 0.7917
 5664/60000 [=>............................] - ETA: 1:33 - loss: 0.6372 - categorical_accuracy: 0.7922
 5728/60000 [=>............................] - ETA: 1:33 - loss: 0.6333 - categorical_accuracy: 0.7933
 5760/60000 [=>............................] - ETA: 1:33 - loss: 0.6309 - categorical_accuracy: 0.7943
 5792/60000 [=>............................] - ETA: 1:32 - loss: 0.6287 - categorical_accuracy: 0.7951
 5824/60000 [=>............................] - ETA: 1:32 - loss: 0.6261 - categorical_accuracy: 0.7958
 5856/60000 [=>............................] - ETA: 1:32 - loss: 0.6248 - categorical_accuracy: 0.7963
 5888/60000 [=>............................] - ETA: 1:32 - loss: 0.6231 - categorical_accuracy: 0.7970
 5920/60000 [=>............................] - ETA: 1:32 - loss: 0.6206 - categorical_accuracy: 0.7978
 5952/60000 [=>............................] - ETA: 1:32 - loss: 0.6181 - categorical_accuracy: 0.7987
 5984/60000 [=>............................] - ETA: 1:32 - loss: 0.6155 - categorical_accuracy: 0.7995
 6016/60000 [==>...........................] - ETA: 1:32 - loss: 0.6130 - categorical_accuracy: 0.8005
 6048/60000 [==>...........................] - ETA: 1:32 - loss: 0.6104 - categorical_accuracy: 0.8016
 6080/60000 [==>...........................] - ETA: 1:32 - loss: 0.6103 - categorical_accuracy: 0.8018
 6112/60000 [==>...........................] - ETA: 1:32 - loss: 0.6091 - categorical_accuracy: 0.8024
 6144/60000 [==>...........................] - ETA: 1:32 - loss: 0.6074 - categorical_accuracy: 0.8027
 6176/60000 [==>...........................] - ETA: 1:32 - loss: 0.6069 - categorical_accuracy: 0.8028
 6208/60000 [==>...........................] - ETA: 1:32 - loss: 0.6052 - categorical_accuracy: 0.8033
 6240/60000 [==>...........................] - ETA: 1:32 - loss: 0.6033 - categorical_accuracy: 0.8042
 6272/60000 [==>...........................] - ETA: 1:32 - loss: 0.6009 - categorical_accuracy: 0.8050
 6304/60000 [==>...........................] - ETA: 1:31 - loss: 0.5998 - categorical_accuracy: 0.8055
 6336/60000 [==>...........................] - ETA: 1:31 - loss: 0.5989 - categorical_accuracy: 0.8059
 6368/60000 [==>...........................] - ETA: 1:31 - loss: 0.5976 - categorical_accuracy: 0.8061
 6400/60000 [==>...........................] - ETA: 1:31 - loss: 0.5957 - categorical_accuracy: 0.8066
 6432/60000 [==>...........................] - ETA: 1:31 - loss: 0.5933 - categorical_accuracy: 0.8074
 6464/60000 [==>...........................] - ETA: 1:31 - loss: 0.5916 - categorical_accuracy: 0.8079
 6496/60000 [==>...........................] - ETA: 1:31 - loss: 0.5896 - categorical_accuracy: 0.8082
 6528/60000 [==>...........................] - ETA: 1:31 - loss: 0.5884 - categorical_accuracy: 0.8088
 6592/60000 [==>...........................] - ETA: 1:31 - loss: 0.5852 - categorical_accuracy: 0.8099
 6624/60000 [==>...........................] - ETA: 1:31 - loss: 0.5833 - categorical_accuracy: 0.8105
 6656/60000 [==>...........................] - ETA: 1:31 - loss: 0.5819 - categorical_accuracy: 0.8111
 6688/60000 [==>...........................] - ETA: 1:31 - loss: 0.5824 - categorical_accuracy: 0.8115
 6720/60000 [==>...........................] - ETA: 1:31 - loss: 0.5815 - categorical_accuracy: 0.8119
 6752/60000 [==>...........................] - ETA: 1:31 - loss: 0.5809 - categorical_accuracy: 0.8119
 6784/60000 [==>...........................] - ETA: 1:30 - loss: 0.5792 - categorical_accuracy: 0.8125
 6816/60000 [==>...........................] - ETA: 1:30 - loss: 0.5774 - categorical_accuracy: 0.8131
 6848/60000 [==>...........................] - ETA: 1:30 - loss: 0.5766 - categorical_accuracy: 0.8134
 6880/60000 [==>...........................] - ETA: 1:30 - loss: 0.5768 - categorical_accuracy: 0.8137
 6912/60000 [==>...........................] - ETA: 1:30 - loss: 0.5757 - categorical_accuracy: 0.8141
 6944/60000 [==>...........................] - ETA: 1:30 - loss: 0.5736 - categorical_accuracy: 0.8148
 6976/60000 [==>...........................] - ETA: 1:30 - loss: 0.5717 - categorical_accuracy: 0.8155
 7008/60000 [==>...........................] - ETA: 1:30 - loss: 0.5697 - categorical_accuracy: 0.8161
 7040/60000 [==>...........................] - ETA: 1:30 - loss: 0.5678 - categorical_accuracy: 0.8168
 7072/60000 [==>...........................] - ETA: 1:30 - loss: 0.5666 - categorical_accuracy: 0.8172
 7104/60000 [==>...........................] - ETA: 1:30 - loss: 0.5668 - categorical_accuracy: 0.8173
 7136/60000 [==>...........................] - ETA: 1:30 - loss: 0.5661 - categorical_accuracy: 0.8177
 7168/60000 [==>...........................] - ETA: 1:30 - loss: 0.5648 - categorical_accuracy: 0.8181
 7200/60000 [==>...........................] - ETA: 1:30 - loss: 0.5640 - categorical_accuracy: 0.8183
 7232/60000 [==>...........................] - ETA: 1:30 - loss: 0.5621 - categorical_accuracy: 0.8189
 7264/60000 [==>...........................] - ETA: 1:30 - loss: 0.5607 - categorical_accuracy: 0.8194
 7296/60000 [==>...........................] - ETA: 1:30 - loss: 0.5599 - categorical_accuracy: 0.8196
 7328/60000 [==>...........................] - ETA: 1:29 - loss: 0.5581 - categorical_accuracy: 0.8201
 7360/60000 [==>...........................] - ETA: 1:29 - loss: 0.5565 - categorical_accuracy: 0.8207
 7392/60000 [==>...........................] - ETA: 1:29 - loss: 0.5575 - categorical_accuracy: 0.8208
 7424/60000 [==>...........................] - ETA: 1:29 - loss: 0.5574 - categorical_accuracy: 0.8211
 7456/60000 [==>...........................] - ETA: 1:29 - loss: 0.5565 - categorical_accuracy: 0.8214
 7488/60000 [==>...........................] - ETA: 1:29 - loss: 0.5551 - categorical_accuracy: 0.8220
 7520/60000 [==>...........................] - ETA: 1:29 - loss: 0.5535 - categorical_accuracy: 0.8226
 7552/60000 [==>...........................] - ETA: 1:29 - loss: 0.5534 - categorical_accuracy: 0.8227
 7584/60000 [==>...........................] - ETA: 1:29 - loss: 0.5534 - categorical_accuracy: 0.8230
 7616/60000 [==>...........................] - ETA: 1:29 - loss: 0.5522 - categorical_accuracy: 0.8234
 7648/60000 [==>...........................] - ETA: 1:29 - loss: 0.5503 - categorical_accuracy: 0.8240
 7680/60000 [==>...........................] - ETA: 1:29 - loss: 0.5507 - categorical_accuracy: 0.8242
 7712/60000 [==>...........................] - ETA: 1:29 - loss: 0.5504 - categorical_accuracy: 0.8244
 7744/60000 [==>...........................] - ETA: 1:29 - loss: 0.5495 - categorical_accuracy: 0.8248
 7776/60000 [==>...........................] - ETA: 1:29 - loss: 0.5488 - categorical_accuracy: 0.8251
 7840/60000 [==>...........................] - ETA: 1:29 - loss: 0.5461 - categorical_accuracy: 0.8260
 7904/60000 [==>...........................] - ETA: 1:28 - loss: 0.5434 - categorical_accuracy: 0.8270
 7936/60000 [==>...........................] - ETA: 1:28 - loss: 0.5415 - categorical_accuracy: 0.8277
 7968/60000 [==>...........................] - ETA: 1:28 - loss: 0.5401 - categorical_accuracy: 0.8282
 8000/60000 [===>..........................] - ETA: 1:28 - loss: 0.5399 - categorical_accuracy: 0.8282
 8032/60000 [===>..........................] - ETA: 1:28 - loss: 0.5387 - categorical_accuracy: 0.8284
 8064/60000 [===>..........................] - ETA: 1:28 - loss: 0.5376 - categorical_accuracy: 0.8287
 8096/60000 [===>..........................] - ETA: 1:28 - loss: 0.5377 - categorical_accuracy: 0.8291
 8128/60000 [===>..........................] - ETA: 1:28 - loss: 0.5359 - categorical_accuracy: 0.8297
 8160/60000 [===>..........................] - ETA: 1:28 - loss: 0.5349 - categorical_accuracy: 0.8301
 8192/60000 [===>..........................] - ETA: 1:28 - loss: 0.5334 - categorical_accuracy: 0.8307
 8224/60000 [===>..........................] - ETA: 1:28 - loss: 0.5321 - categorical_accuracy: 0.8312
 8256/60000 [===>..........................] - ETA: 1:28 - loss: 0.5309 - categorical_accuracy: 0.8318
 8288/60000 [===>..........................] - ETA: 1:28 - loss: 0.5306 - categorical_accuracy: 0.8319
 8320/60000 [===>..........................] - ETA: 1:28 - loss: 0.5291 - categorical_accuracy: 0.8323
 8352/60000 [===>..........................] - ETA: 1:28 - loss: 0.5286 - categorical_accuracy: 0.8327
 8384/60000 [===>..........................] - ETA: 1:27 - loss: 0.5272 - categorical_accuracy: 0.8330
 8416/60000 [===>..........................] - ETA: 1:27 - loss: 0.5265 - categorical_accuracy: 0.8332
 8448/60000 [===>..........................] - ETA: 1:27 - loss: 0.5254 - categorical_accuracy: 0.8335
 8480/60000 [===>..........................] - ETA: 1:27 - loss: 0.5245 - categorical_accuracy: 0.8338
 8512/60000 [===>..........................] - ETA: 1:27 - loss: 0.5230 - categorical_accuracy: 0.8344
 8544/60000 [===>..........................] - ETA: 1:27 - loss: 0.5220 - categorical_accuracy: 0.8346
 8608/60000 [===>..........................] - ETA: 1:27 - loss: 0.5193 - categorical_accuracy: 0.8355
 8640/60000 [===>..........................] - ETA: 1:27 - loss: 0.5177 - categorical_accuracy: 0.8361
 8672/60000 [===>..........................] - ETA: 1:27 - loss: 0.5167 - categorical_accuracy: 0.8364
 8704/60000 [===>..........................] - ETA: 1:27 - loss: 0.5154 - categorical_accuracy: 0.8366
 8736/60000 [===>..........................] - ETA: 1:27 - loss: 0.5145 - categorical_accuracy: 0.8369
 8768/60000 [===>..........................] - ETA: 1:27 - loss: 0.5136 - categorical_accuracy: 0.8371
 8800/60000 [===>..........................] - ETA: 1:27 - loss: 0.5124 - categorical_accuracy: 0.8374
 8832/60000 [===>..........................] - ETA: 1:27 - loss: 0.5120 - categorical_accuracy: 0.8375
 8864/60000 [===>..........................] - ETA: 1:27 - loss: 0.5108 - categorical_accuracy: 0.8380
 8896/60000 [===>..........................] - ETA: 1:26 - loss: 0.5098 - categorical_accuracy: 0.8384
 8928/60000 [===>..........................] - ETA: 1:27 - loss: 0.5087 - categorical_accuracy: 0.8385
 8960/60000 [===>..........................] - ETA: 1:26 - loss: 0.5071 - categorical_accuracy: 0.8391
 8992/60000 [===>..........................] - ETA: 1:26 - loss: 0.5054 - categorical_accuracy: 0.8395
 9024/60000 [===>..........................] - ETA: 1:26 - loss: 0.5042 - categorical_accuracy: 0.8398
 9056/60000 [===>..........................] - ETA: 1:26 - loss: 0.5031 - categorical_accuracy: 0.8402
 9088/60000 [===>..........................] - ETA: 1:26 - loss: 0.5017 - categorical_accuracy: 0.8407
 9120/60000 [===>..........................] - ETA: 1:26 - loss: 0.5003 - categorical_accuracy: 0.8411
 9152/60000 [===>..........................] - ETA: 1:26 - loss: 0.5001 - categorical_accuracy: 0.8413
 9184/60000 [===>..........................] - ETA: 1:26 - loss: 0.4994 - categorical_accuracy: 0.8417
 9216/60000 [===>..........................] - ETA: 1:26 - loss: 0.4984 - categorical_accuracy: 0.8421
 9248/60000 [===>..........................] - ETA: 1:26 - loss: 0.4975 - categorical_accuracy: 0.8425
 9280/60000 [===>..........................] - ETA: 1:26 - loss: 0.4960 - categorical_accuracy: 0.8430
 9312/60000 [===>..........................] - ETA: 1:26 - loss: 0.4952 - categorical_accuracy: 0.8431
 9344/60000 [===>..........................] - ETA: 1:26 - loss: 0.4947 - categorical_accuracy: 0.8433
 9376/60000 [===>..........................] - ETA: 1:26 - loss: 0.4932 - categorical_accuracy: 0.8439
 9408/60000 [===>..........................] - ETA: 1:26 - loss: 0.4935 - categorical_accuracy: 0.8441
 9440/60000 [===>..........................] - ETA: 1:26 - loss: 0.4928 - categorical_accuracy: 0.8443
 9472/60000 [===>..........................] - ETA: 1:26 - loss: 0.4916 - categorical_accuracy: 0.8447
 9504/60000 [===>..........................] - ETA: 1:26 - loss: 0.4909 - categorical_accuracy: 0.8448
 9536/60000 [===>..........................] - ETA: 1:25 - loss: 0.4899 - categorical_accuracy: 0.8450
 9568/60000 [===>..........................] - ETA: 1:25 - loss: 0.4889 - categorical_accuracy: 0.8454
 9632/60000 [===>..........................] - ETA: 1:25 - loss: 0.4868 - categorical_accuracy: 0.8460
 9664/60000 [===>..........................] - ETA: 1:25 - loss: 0.4860 - categorical_accuracy: 0.8463
 9696/60000 [===>..........................] - ETA: 1:25 - loss: 0.4851 - categorical_accuracy: 0.8465
 9728/60000 [===>..........................] - ETA: 1:25 - loss: 0.4850 - categorical_accuracy: 0.8465
 9760/60000 [===>..........................] - ETA: 1:25 - loss: 0.4842 - categorical_accuracy: 0.8469
 9792/60000 [===>..........................] - ETA: 1:25 - loss: 0.4829 - categorical_accuracy: 0.8473
 9824/60000 [===>..........................] - ETA: 1:25 - loss: 0.4815 - categorical_accuracy: 0.8478
 9856/60000 [===>..........................] - ETA: 1:25 - loss: 0.4802 - categorical_accuracy: 0.8481
 9888/60000 [===>..........................] - ETA: 1:25 - loss: 0.4793 - categorical_accuracy: 0.8484
 9920/60000 [===>..........................] - ETA: 1:25 - loss: 0.4787 - categorical_accuracy: 0.8485
 9952/60000 [===>..........................] - ETA: 1:25 - loss: 0.4782 - categorical_accuracy: 0.8488
 9984/60000 [===>..........................] - ETA: 1:25 - loss: 0.4776 - categorical_accuracy: 0.8490
10016/60000 [====>.........................] - ETA: 1:25 - loss: 0.4768 - categorical_accuracy: 0.8492
10048/60000 [====>.........................] - ETA: 1:25 - loss: 0.4756 - categorical_accuracy: 0.8495
10080/60000 [====>.........................] - ETA: 1:24 - loss: 0.4743 - categorical_accuracy: 0.8500
10112/60000 [====>.........................] - ETA: 1:24 - loss: 0.4741 - categorical_accuracy: 0.8502
10144/60000 [====>.........................] - ETA: 1:24 - loss: 0.4729 - categorical_accuracy: 0.8506
10208/60000 [====>.........................] - ETA: 1:24 - loss: 0.4706 - categorical_accuracy: 0.8513
10240/60000 [====>.........................] - ETA: 1:24 - loss: 0.4693 - categorical_accuracy: 0.8518
10272/60000 [====>.........................] - ETA: 1:24 - loss: 0.4682 - categorical_accuracy: 0.8521
10304/60000 [====>.........................] - ETA: 1:24 - loss: 0.4676 - categorical_accuracy: 0.8523
10336/60000 [====>.........................] - ETA: 1:24 - loss: 0.4672 - categorical_accuracy: 0.8525
10368/60000 [====>.........................] - ETA: 1:24 - loss: 0.4681 - categorical_accuracy: 0.8523
10400/60000 [====>.........................] - ETA: 1:24 - loss: 0.4672 - categorical_accuracy: 0.8526
10432/60000 [====>.........................] - ETA: 1:24 - loss: 0.4666 - categorical_accuracy: 0.8528
10464/60000 [====>.........................] - ETA: 1:24 - loss: 0.4657 - categorical_accuracy: 0.8529
10496/60000 [====>.........................] - ETA: 1:24 - loss: 0.4651 - categorical_accuracy: 0.8531
10560/60000 [====>.........................] - ETA: 1:23 - loss: 0.4634 - categorical_accuracy: 0.8536
10592/60000 [====>.........................] - ETA: 1:23 - loss: 0.4622 - categorical_accuracy: 0.8540
10624/60000 [====>.........................] - ETA: 1:23 - loss: 0.4618 - categorical_accuracy: 0.8542
10656/60000 [====>.........................] - ETA: 1:23 - loss: 0.4606 - categorical_accuracy: 0.8546
10688/60000 [====>.........................] - ETA: 1:23 - loss: 0.4601 - categorical_accuracy: 0.8549
10720/60000 [====>.........................] - ETA: 1:23 - loss: 0.4602 - categorical_accuracy: 0.8550
10752/60000 [====>.........................] - ETA: 1:23 - loss: 0.4603 - categorical_accuracy: 0.8550
10784/60000 [====>.........................] - ETA: 1:23 - loss: 0.4593 - categorical_accuracy: 0.8553
10816/60000 [====>.........................] - ETA: 1:23 - loss: 0.4584 - categorical_accuracy: 0.8557
10848/60000 [====>.........................] - ETA: 1:23 - loss: 0.4577 - categorical_accuracy: 0.8559
10880/60000 [====>.........................] - ETA: 1:23 - loss: 0.4567 - categorical_accuracy: 0.8562
10912/60000 [====>.........................] - ETA: 1:23 - loss: 0.4562 - categorical_accuracy: 0.8565
10944/60000 [====>.........................] - ETA: 1:23 - loss: 0.4554 - categorical_accuracy: 0.8567
10976/60000 [====>.........................] - ETA: 1:23 - loss: 0.4545 - categorical_accuracy: 0.8569
11008/60000 [====>.........................] - ETA: 1:23 - loss: 0.4536 - categorical_accuracy: 0.8571
11040/60000 [====>.........................] - ETA: 1:23 - loss: 0.4528 - categorical_accuracy: 0.8572
11072/60000 [====>.........................] - ETA: 1:23 - loss: 0.4523 - categorical_accuracy: 0.8574
11104/60000 [====>.........................] - ETA: 1:23 - loss: 0.4516 - categorical_accuracy: 0.8576
11136/60000 [====>.........................] - ETA: 1:23 - loss: 0.4514 - categorical_accuracy: 0.8578
11168/60000 [====>.........................] - ETA: 1:22 - loss: 0.4507 - categorical_accuracy: 0.8579
11200/60000 [====>.........................] - ETA: 1:22 - loss: 0.4501 - categorical_accuracy: 0.8581
11232/60000 [====>.........................] - ETA: 1:22 - loss: 0.4497 - categorical_accuracy: 0.8581
11264/60000 [====>.........................] - ETA: 1:22 - loss: 0.4488 - categorical_accuracy: 0.8583
11296/60000 [====>.........................] - ETA: 1:22 - loss: 0.4476 - categorical_accuracy: 0.8587
11328/60000 [====>.........................] - ETA: 1:22 - loss: 0.4467 - categorical_accuracy: 0.8590
11360/60000 [====>.........................] - ETA: 1:22 - loss: 0.4459 - categorical_accuracy: 0.8592
11392/60000 [====>.........................] - ETA: 1:22 - loss: 0.4453 - categorical_accuracy: 0.8595
11424/60000 [====>.........................] - ETA: 1:22 - loss: 0.4443 - categorical_accuracy: 0.8598
11456/60000 [====>.........................] - ETA: 1:22 - loss: 0.4434 - categorical_accuracy: 0.8601
11488/60000 [====>.........................] - ETA: 1:22 - loss: 0.4425 - categorical_accuracy: 0.8602
11520/60000 [====>.........................] - ETA: 1:22 - loss: 0.4419 - categorical_accuracy: 0.8604
11552/60000 [====>.........................] - ETA: 1:22 - loss: 0.4411 - categorical_accuracy: 0.8605
11584/60000 [====>.........................] - ETA: 1:22 - loss: 0.4402 - categorical_accuracy: 0.8609
11616/60000 [====>.........................] - ETA: 1:22 - loss: 0.4393 - categorical_accuracy: 0.8612
11648/60000 [====>.........................] - ETA: 1:22 - loss: 0.4384 - categorical_accuracy: 0.8615
11680/60000 [====>.........................] - ETA: 1:21 - loss: 0.4375 - categorical_accuracy: 0.8617
11712/60000 [====>.........................] - ETA: 1:21 - loss: 0.4369 - categorical_accuracy: 0.8619
11744/60000 [====>.........................] - ETA: 1:21 - loss: 0.4363 - categorical_accuracy: 0.8621
11776/60000 [====>.........................] - ETA: 1:21 - loss: 0.4356 - categorical_accuracy: 0.8623
11808/60000 [====>.........................] - ETA: 1:21 - loss: 0.4347 - categorical_accuracy: 0.8626
11872/60000 [====>.........................] - ETA: 1:21 - loss: 0.4329 - categorical_accuracy: 0.8632
11904/60000 [====>.........................] - ETA: 1:21 - loss: 0.4323 - categorical_accuracy: 0.8634
11936/60000 [====>.........................] - ETA: 1:21 - loss: 0.4314 - categorical_accuracy: 0.8637
11968/60000 [====>.........................] - ETA: 1:21 - loss: 0.4306 - categorical_accuracy: 0.8639
12000/60000 [=====>........................] - ETA: 1:21 - loss: 0.4297 - categorical_accuracy: 0.8642
12032/60000 [=====>........................] - ETA: 1:21 - loss: 0.4292 - categorical_accuracy: 0.8644
12064/60000 [=====>........................] - ETA: 1:21 - loss: 0.4286 - categorical_accuracy: 0.8646
12096/60000 [=====>........................] - ETA: 1:21 - loss: 0.4277 - categorical_accuracy: 0.8648
12160/60000 [=====>........................] - ETA: 1:21 - loss: 0.4266 - categorical_accuracy: 0.8653
12192/60000 [=====>........................] - ETA: 1:21 - loss: 0.4259 - categorical_accuracy: 0.8656
12224/60000 [=====>........................] - ETA: 1:20 - loss: 0.4253 - categorical_accuracy: 0.8658
12256/60000 [=====>........................] - ETA: 1:20 - loss: 0.4244 - categorical_accuracy: 0.8661
12288/60000 [=====>........................] - ETA: 1:20 - loss: 0.4240 - categorical_accuracy: 0.8663
12320/60000 [=====>........................] - ETA: 1:20 - loss: 0.4235 - categorical_accuracy: 0.8665
12352/60000 [=====>........................] - ETA: 1:20 - loss: 0.4230 - categorical_accuracy: 0.8667
12416/60000 [=====>........................] - ETA: 1:20 - loss: 0.4215 - categorical_accuracy: 0.8672
12448/60000 [=====>........................] - ETA: 1:20 - loss: 0.4210 - categorical_accuracy: 0.8673
12480/60000 [=====>........................] - ETA: 1:20 - loss: 0.4223 - categorical_accuracy: 0.8673
12512/60000 [=====>........................] - ETA: 1:20 - loss: 0.4230 - categorical_accuracy: 0.8672
12544/60000 [=====>........................] - ETA: 1:20 - loss: 0.4223 - categorical_accuracy: 0.8674
12576/60000 [=====>........................] - ETA: 1:20 - loss: 0.4217 - categorical_accuracy: 0.8676
12608/60000 [=====>........................] - ETA: 1:20 - loss: 0.4209 - categorical_accuracy: 0.8679
12640/60000 [=====>........................] - ETA: 1:20 - loss: 0.4199 - categorical_accuracy: 0.8683
12672/60000 [=====>........................] - ETA: 1:20 - loss: 0.4193 - categorical_accuracy: 0.8685
12704/60000 [=====>........................] - ETA: 1:20 - loss: 0.4190 - categorical_accuracy: 0.8685
12736/60000 [=====>........................] - ETA: 1:20 - loss: 0.4185 - categorical_accuracy: 0.8687
12768/60000 [=====>........................] - ETA: 1:20 - loss: 0.4185 - categorical_accuracy: 0.8687
12800/60000 [=====>........................] - ETA: 1:20 - loss: 0.4178 - categorical_accuracy: 0.8689
12832/60000 [=====>........................] - ETA: 1:20 - loss: 0.4172 - categorical_accuracy: 0.8690
12864/60000 [=====>........................] - ETA: 1:19 - loss: 0.4167 - categorical_accuracy: 0.8692
12896/60000 [=====>........................] - ETA: 1:19 - loss: 0.4158 - categorical_accuracy: 0.8695
12928/60000 [=====>........................] - ETA: 1:19 - loss: 0.4151 - categorical_accuracy: 0.8697
12960/60000 [=====>........................] - ETA: 1:19 - loss: 0.4145 - categorical_accuracy: 0.8698
12992/60000 [=====>........................] - ETA: 1:19 - loss: 0.4137 - categorical_accuracy: 0.8702
13024/60000 [=====>........................] - ETA: 1:19 - loss: 0.4128 - categorical_accuracy: 0.8705
13056/60000 [=====>........................] - ETA: 1:19 - loss: 0.4121 - categorical_accuracy: 0.8706
13088/60000 [=====>........................] - ETA: 1:19 - loss: 0.4112 - categorical_accuracy: 0.8710
13120/60000 [=====>........................] - ETA: 1:19 - loss: 0.4105 - categorical_accuracy: 0.8711
13152/60000 [=====>........................] - ETA: 1:19 - loss: 0.4103 - categorical_accuracy: 0.8712
13184/60000 [=====>........................] - ETA: 1:19 - loss: 0.4096 - categorical_accuracy: 0.8713
13216/60000 [=====>........................] - ETA: 1:19 - loss: 0.4089 - categorical_accuracy: 0.8714
13248/60000 [=====>........................] - ETA: 1:19 - loss: 0.4082 - categorical_accuracy: 0.8717
13280/60000 [=====>........................] - ETA: 1:19 - loss: 0.4081 - categorical_accuracy: 0.8718
13312/60000 [=====>........................] - ETA: 1:19 - loss: 0.4078 - categorical_accuracy: 0.8719
13344/60000 [=====>........................] - ETA: 1:19 - loss: 0.4072 - categorical_accuracy: 0.8721
13376/60000 [=====>........................] - ETA: 1:19 - loss: 0.4068 - categorical_accuracy: 0.8722
13408/60000 [=====>........................] - ETA: 1:19 - loss: 0.4063 - categorical_accuracy: 0.8724
13440/60000 [=====>........................] - ETA: 1:18 - loss: 0.4056 - categorical_accuracy: 0.8726
13472/60000 [=====>........................] - ETA: 1:18 - loss: 0.4057 - categorical_accuracy: 0.8728
13504/60000 [=====>........................] - ETA: 1:18 - loss: 0.4056 - categorical_accuracy: 0.8728
13536/60000 [=====>........................] - ETA: 1:18 - loss: 0.4049 - categorical_accuracy: 0.8730
13568/60000 [=====>........................] - ETA: 1:18 - loss: 0.4046 - categorical_accuracy: 0.8731
13600/60000 [=====>........................] - ETA: 1:18 - loss: 0.4039 - categorical_accuracy: 0.8733
13632/60000 [=====>........................] - ETA: 1:18 - loss: 0.4032 - categorical_accuracy: 0.8735
13664/60000 [=====>........................] - ETA: 1:18 - loss: 0.4027 - categorical_accuracy: 0.8737
13696/60000 [=====>........................] - ETA: 1:18 - loss: 0.4019 - categorical_accuracy: 0.8739
13728/60000 [=====>........................] - ETA: 1:18 - loss: 0.4013 - categorical_accuracy: 0.8741
13760/60000 [=====>........................] - ETA: 1:18 - loss: 0.4006 - categorical_accuracy: 0.8743
13792/60000 [=====>........................] - ETA: 1:18 - loss: 0.4000 - categorical_accuracy: 0.8744
13824/60000 [=====>........................] - ETA: 1:18 - loss: 0.3993 - categorical_accuracy: 0.8747
13856/60000 [=====>........................] - ETA: 1:18 - loss: 0.3988 - categorical_accuracy: 0.8749
13888/60000 [=====>........................] - ETA: 1:18 - loss: 0.3987 - categorical_accuracy: 0.8749
13920/60000 [=====>........................] - ETA: 1:18 - loss: 0.3979 - categorical_accuracy: 0.8752
13952/60000 [=====>........................] - ETA: 1:18 - loss: 0.3978 - categorical_accuracy: 0.8753
13984/60000 [=====>........................] - ETA: 1:18 - loss: 0.3973 - categorical_accuracy: 0.8754
14016/60000 [======>.......................] - ETA: 1:17 - loss: 0.3975 - categorical_accuracy: 0.8754
14048/60000 [======>.......................] - ETA: 1:17 - loss: 0.3967 - categorical_accuracy: 0.8757
14080/60000 [======>.......................] - ETA: 1:17 - loss: 0.3966 - categorical_accuracy: 0.8758
14112/60000 [======>.......................] - ETA: 1:17 - loss: 0.3967 - categorical_accuracy: 0.8759
14144/60000 [======>.......................] - ETA: 1:17 - loss: 0.3964 - categorical_accuracy: 0.8758
14176/60000 [======>.......................] - ETA: 1:17 - loss: 0.3958 - categorical_accuracy: 0.8760
14208/60000 [======>.......................] - ETA: 1:17 - loss: 0.3959 - categorical_accuracy: 0.8761
14240/60000 [======>.......................] - ETA: 1:17 - loss: 0.3954 - categorical_accuracy: 0.8761
14272/60000 [======>.......................] - ETA: 1:17 - loss: 0.3948 - categorical_accuracy: 0.8763
14304/60000 [======>.......................] - ETA: 1:17 - loss: 0.3944 - categorical_accuracy: 0.8763
14336/60000 [======>.......................] - ETA: 1:17 - loss: 0.3940 - categorical_accuracy: 0.8765
14368/60000 [======>.......................] - ETA: 1:17 - loss: 0.3933 - categorical_accuracy: 0.8767
14400/60000 [======>.......................] - ETA: 1:17 - loss: 0.3925 - categorical_accuracy: 0.8769
14432/60000 [======>.......................] - ETA: 1:17 - loss: 0.3919 - categorical_accuracy: 0.8771
14464/60000 [======>.......................] - ETA: 1:17 - loss: 0.3914 - categorical_accuracy: 0.8772
14496/60000 [======>.......................] - ETA: 1:17 - loss: 0.3906 - categorical_accuracy: 0.8775
14528/60000 [======>.......................] - ETA: 1:17 - loss: 0.3905 - categorical_accuracy: 0.8775
14560/60000 [======>.......................] - ETA: 1:17 - loss: 0.3897 - categorical_accuracy: 0.8778
14592/60000 [======>.......................] - ETA: 1:17 - loss: 0.3890 - categorical_accuracy: 0.8780
14624/60000 [======>.......................] - ETA: 1:16 - loss: 0.3884 - categorical_accuracy: 0.8783
14656/60000 [======>.......................] - ETA: 1:16 - loss: 0.3878 - categorical_accuracy: 0.8784
14688/60000 [======>.......................] - ETA: 1:16 - loss: 0.3872 - categorical_accuracy: 0.8787
14720/60000 [======>.......................] - ETA: 1:16 - loss: 0.3871 - categorical_accuracy: 0.8787
14752/60000 [======>.......................] - ETA: 1:16 - loss: 0.3864 - categorical_accuracy: 0.8789
14784/60000 [======>.......................] - ETA: 1:16 - loss: 0.3857 - categorical_accuracy: 0.8791
14816/60000 [======>.......................] - ETA: 1:16 - loss: 0.3851 - categorical_accuracy: 0.8793
14848/60000 [======>.......................] - ETA: 1:16 - loss: 0.3846 - categorical_accuracy: 0.8792
14880/60000 [======>.......................] - ETA: 1:16 - loss: 0.3844 - categorical_accuracy: 0.8793
14912/60000 [======>.......................] - ETA: 1:16 - loss: 0.3845 - categorical_accuracy: 0.8793
14944/60000 [======>.......................] - ETA: 1:16 - loss: 0.3843 - categorical_accuracy: 0.8795
14976/60000 [======>.......................] - ETA: 1:16 - loss: 0.3838 - categorical_accuracy: 0.8797
15008/60000 [======>.......................] - ETA: 1:16 - loss: 0.3835 - categorical_accuracy: 0.8798
15040/60000 [======>.......................] - ETA: 1:16 - loss: 0.3832 - categorical_accuracy: 0.8799
15104/60000 [======>.......................] - ETA: 1:16 - loss: 0.3825 - categorical_accuracy: 0.8801
15136/60000 [======>.......................] - ETA: 1:16 - loss: 0.3824 - categorical_accuracy: 0.8802
15168/60000 [======>.......................] - ETA: 1:16 - loss: 0.3816 - categorical_accuracy: 0.8805
15200/60000 [======>.......................] - ETA: 1:15 - loss: 0.3811 - categorical_accuracy: 0.8806
15232/60000 [======>.......................] - ETA: 1:15 - loss: 0.3812 - categorical_accuracy: 0.8805
15264/60000 [======>.......................] - ETA: 1:15 - loss: 0.3808 - categorical_accuracy: 0.8806
15296/60000 [======>.......................] - ETA: 1:15 - loss: 0.3801 - categorical_accuracy: 0.8808
15328/60000 [======>.......................] - ETA: 1:15 - loss: 0.3796 - categorical_accuracy: 0.8809
15360/60000 [======>.......................] - ETA: 1:15 - loss: 0.3790 - categorical_accuracy: 0.8811
15392/60000 [======>.......................] - ETA: 1:15 - loss: 0.3794 - categorical_accuracy: 0.8811
15424/60000 [======>.......................] - ETA: 1:15 - loss: 0.3788 - categorical_accuracy: 0.8813
15456/60000 [======>.......................] - ETA: 1:15 - loss: 0.3781 - categorical_accuracy: 0.8815
15488/60000 [======>.......................] - ETA: 1:15 - loss: 0.3775 - categorical_accuracy: 0.8817
15520/60000 [======>.......................] - ETA: 1:15 - loss: 0.3771 - categorical_accuracy: 0.8818
15552/60000 [======>.......................] - ETA: 1:15 - loss: 0.3766 - categorical_accuracy: 0.8819
15584/60000 [======>.......................] - ETA: 1:15 - loss: 0.3763 - categorical_accuracy: 0.8821
15616/60000 [======>.......................] - ETA: 1:15 - loss: 0.3761 - categorical_accuracy: 0.8821
15648/60000 [======>.......................] - ETA: 1:15 - loss: 0.3754 - categorical_accuracy: 0.8823
15712/60000 [======>.......................] - ETA: 1:15 - loss: 0.3742 - categorical_accuracy: 0.8827
15744/60000 [======>.......................] - ETA: 1:15 - loss: 0.3738 - categorical_accuracy: 0.8829
15808/60000 [======>.......................] - ETA: 1:14 - loss: 0.3730 - categorical_accuracy: 0.8832
15840/60000 [======>.......................] - ETA: 1:14 - loss: 0.3729 - categorical_accuracy: 0.8833
15872/60000 [======>.......................] - ETA: 1:14 - loss: 0.3722 - categorical_accuracy: 0.8835
15904/60000 [======>.......................] - ETA: 1:14 - loss: 0.3716 - categorical_accuracy: 0.8837
15936/60000 [======>.......................] - ETA: 1:14 - loss: 0.3713 - categorical_accuracy: 0.8838
15968/60000 [======>.......................] - ETA: 1:14 - loss: 0.3709 - categorical_accuracy: 0.8840
16000/60000 [=======>......................] - ETA: 1:14 - loss: 0.3703 - categorical_accuracy: 0.8841
16032/60000 [=======>......................] - ETA: 1:14 - loss: 0.3700 - categorical_accuracy: 0.8842
16064/60000 [=======>......................] - ETA: 1:14 - loss: 0.3699 - categorical_accuracy: 0.8842
16096/60000 [=======>......................] - ETA: 1:14 - loss: 0.3694 - categorical_accuracy: 0.8843
16128/60000 [=======>......................] - ETA: 1:14 - loss: 0.3688 - categorical_accuracy: 0.8844
16160/60000 [=======>......................] - ETA: 1:14 - loss: 0.3687 - categorical_accuracy: 0.8845
16192/60000 [=======>......................] - ETA: 1:14 - loss: 0.3682 - categorical_accuracy: 0.8846
16224/60000 [=======>......................] - ETA: 1:14 - loss: 0.3677 - categorical_accuracy: 0.8847
16256/60000 [=======>......................] - ETA: 1:14 - loss: 0.3671 - categorical_accuracy: 0.8850
16288/60000 [=======>......................] - ETA: 1:14 - loss: 0.3664 - categorical_accuracy: 0.8852
16320/60000 [=======>......................] - ETA: 1:14 - loss: 0.3660 - categorical_accuracy: 0.8853
16352/60000 [=======>......................] - ETA: 1:14 - loss: 0.3656 - categorical_accuracy: 0.8854
16384/60000 [=======>......................] - ETA: 1:14 - loss: 0.3650 - categorical_accuracy: 0.8856
16416/60000 [=======>......................] - ETA: 1:13 - loss: 0.3644 - categorical_accuracy: 0.8858
16448/60000 [=======>......................] - ETA: 1:13 - loss: 0.3646 - categorical_accuracy: 0.8859
16480/60000 [=======>......................] - ETA: 1:13 - loss: 0.3640 - categorical_accuracy: 0.8861
16512/60000 [=======>......................] - ETA: 1:13 - loss: 0.3638 - categorical_accuracy: 0.8861
16544/60000 [=======>......................] - ETA: 1:13 - loss: 0.3634 - categorical_accuracy: 0.8862
16576/60000 [=======>......................] - ETA: 1:13 - loss: 0.3632 - categorical_accuracy: 0.8863
16608/60000 [=======>......................] - ETA: 1:13 - loss: 0.3627 - categorical_accuracy: 0.8865
16640/60000 [=======>......................] - ETA: 1:13 - loss: 0.3621 - categorical_accuracy: 0.8867
16672/60000 [=======>......................] - ETA: 1:13 - loss: 0.3615 - categorical_accuracy: 0.8869
16736/60000 [=======>......................] - ETA: 1:13 - loss: 0.3622 - categorical_accuracy: 0.8870
16768/60000 [=======>......................] - ETA: 1:13 - loss: 0.3617 - categorical_accuracy: 0.8872
16800/60000 [=======>......................] - ETA: 1:13 - loss: 0.3610 - categorical_accuracy: 0.8874
16832/60000 [=======>......................] - ETA: 1:13 - loss: 0.3604 - categorical_accuracy: 0.8877
16864/60000 [=======>......................] - ETA: 1:13 - loss: 0.3599 - categorical_accuracy: 0.8878
16896/60000 [=======>......................] - ETA: 1:13 - loss: 0.3593 - categorical_accuracy: 0.8880
16928/60000 [=======>......................] - ETA: 1:13 - loss: 0.3589 - categorical_accuracy: 0.8881
16960/60000 [=======>......................] - ETA: 1:12 - loss: 0.3586 - categorical_accuracy: 0.8883
16992/60000 [=======>......................] - ETA: 1:12 - loss: 0.3580 - categorical_accuracy: 0.8885
17024/60000 [=======>......................] - ETA: 1:12 - loss: 0.3574 - categorical_accuracy: 0.8887
17056/60000 [=======>......................] - ETA: 1:12 - loss: 0.3570 - categorical_accuracy: 0.8888
17088/60000 [=======>......................] - ETA: 1:12 - loss: 0.3570 - categorical_accuracy: 0.8889
17120/60000 [=======>......................] - ETA: 1:12 - loss: 0.3564 - categorical_accuracy: 0.8891
17152/60000 [=======>......................] - ETA: 1:12 - loss: 0.3560 - categorical_accuracy: 0.8892
17184/60000 [=======>......................] - ETA: 1:12 - loss: 0.3555 - categorical_accuracy: 0.8894
17216/60000 [=======>......................] - ETA: 1:12 - loss: 0.3549 - categorical_accuracy: 0.8896
17248/60000 [=======>......................] - ETA: 1:12 - loss: 0.3546 - categorical_accuracy: 0.8897
17280/60000 [=======>......................] - ETA: 1:12 - loss: 0.3540 - categorical_accuracy: 0.8899
17312/60000 [=======>......................] - ETA: 1:12 - loss: 0.3534 - categorical_accuracy: 0.8901
17344/60000 [=======>......................] - ETA: 1:12 - loss: 0.3532 - categorical_accuracy: 0.8901
17376/60000 [=======>......................] - ETA: 1:12 - loss: 0.3529 - categorical_accuracy: 0.8902
17408/60000 [=======>......................] - ETA: 1:12 - loss: 0.3525 - categorical_accuracy: 0.8903
17440/60000 [=======>......................] - ETA: 1:12 - loss: 0.3527 - categorical_accuracy: 0.8904
17472/60000 [=======>......................] - ETA: 1:12 - loss: 0.3522 - categorical_accuracy: 0.8905
17504/60000 [=======>......................] - ETA: 1:12 - loss: 0.3516 - categorical_accuracy: 0.8907
17536/60000 [=======>......................] - ETA: 1:11 - loss: 0.3511 - categorical_accuracy: 0.8909
17568/60000 [=======>......................] - ETA: 1:11 - loss: 0.3508 - categorical_accuracy: 0.8909
17600/60000 [=======>......................] - ETA: 1:11 - loss: 0.3504 - categorical_accuracy: 0.8911
17632/60000 [=======>......................] - ETA: 1:11 - loss: 0.3499 - categorical_accuracy: 0.8912
17664/60000 [=======>......................] - ETA: 1:11 - loss: 0.3495 - categorical_accuracy: 0.8914
17696/60000 [=======>......................] - ETA: 1:11 - loss: 0.3496 - categorical_accuracy: 0.8915
17728/60000 [=======>......................] - ETA: 1:11 - loss: 0.3493 - categorical_accuracy: 0.8916
17760/60000 [=======>......................] - ETA: 1:11 - loss: 0.3488 - categorical_accuracy: 0.8918
17792/60000 [=======>......................] - ETA: 1:11 - loss: 0.3488 - categorical_accuracy: 0.8919
17824/60000 [=======>......................] - ETA: 1:11 - loss: 0.3483 - categorical_accuracy: 0.8921
17856/60000 [=======>......................] - ETA: 1:11 - loss: 0.3479 - categorical_accuracy: 0.8922
17888/60000 [=======>......................] - ETA: 1:11 - loss: 0.3478 - categorical_accuracy: 0.8923
17920/60000 [=======>......................] - ETA: 1:11 - loss: 0.3476 - categorical_accuracy: 0.8923
17952/60000 [=======>......................] - ETA: 1:11 - loss: 0.3472 - categorical_accuracy: 0.8924
17984/60000 [=======>......................] - ETA: 1:11 - loss: 0.3468 - categorical_accuracy: 0.8925
18016/60000 [========>.....................] - ETA: 1:11 - loss: 0.3474 - categorical_accuracy: 0.8923
18048/60000 [========>.....................] - ETA: 1:11 - loss: 0.3471 - categorical_accuracy: 0.8924
18080/60000 [========>.....................] - ETA: 1:10 - loss: 0.3469 - categorical_accuracy: 0.8924
18112/60000 [========>.....................] - ETA: 1:10 - loss: 0.3470 - categorical_accuracy: 0.8923
18144/60000 [========>.....................] - ETA: 1:10 - loss: 0.3466 - categorical_accuracy: 0.8925
18176/60000 [========>.....................] - ETA: 1:10 - loss: 0.3462 - categorical_accuracy: 0.8926
18208/60000 [========>.....................] - ETA: 1:10 - loss: 0.3457 - categorical_accuracy: 0.8927
18240/60000 [========>.....................] - ETA: 1:10 - loss: 0.3459 - categorical_accuracy: 0.8928
18272/60000 [========>.....................] - ETA: 1:10 - loss: 0.3455 - categorical_accuracy: 0.8928
18304/60000 [========>.....................] - ETA: 1:10 - loss: 0.3452 - categorical_accuracy: 0.8929
18336/60000 [========>.....................] - ETA: 1:10 - loss: 0.3449 - categorical_accuracy: 0.8929
18368/60000 [========>.....................] - ETA: 1:10 - loss: 0.3446 - categorical_accuracy: 0.8930
18400/60000 [========>.....................] - ETA: 1:10 - loss: 0.3441 - categorical_accuracy: 0.8932
18432/60000 [========>.....................] - ETA: 1:10 - loss: 0.3437 - categorical_accuracy: 0.8932
18464/60000 [========>.....................] - ETA: 1:10 - loss: 0.3432 - categorical_accuracy: 0.8934
18496/60000 [========>.....................] - ETA: 1:10 - loss: 0.3431 - categorical_accuracy: 0.8934
18528/60000 [========>.....................] - ETA: 1:10 - loss: 0.3429 - categorical_accuracy: 0.8935
18560/60000 [========>.....................] - ETA: 1:10 - loss: 0.3427 - categorical_accuracy: 0.8935
18592/60000 [========>.....................] - ETA: 1:10 - loss: 0.3424 - categorical_accuracy: 0.8936
18624/60000 [========>.....................] - ETA: 1:10 - loss: 0.3421 - categorical_accuracy: 0.8936
18656/60000 [========>.....................] - ETA: 1:10 - loss: 0.3417 - categorical_accuracy: 0.8938
18688/60000 [========>.....................] - ETA: 1:09 - loss: 0.3416 - categorical_accuracy: 0.8938
18720/60000 [========>.....................] - ETA: 1:09 - loss: 0.3412 - categorical_accuracy: 0.8940
18752/60000 [========>.....................] - ETA: 1:09 - loss: 0.3409 - categorical_accuracy: 0.8940
18816/60000 [========>.....................] - ETA: 1:09 - loss: 0.3402 - categorical_accuracy: 0.8943
18848/60000 [========>.....................] - ETA: 1:09 - loss: 0.3397 - categorical_accuracy: 0.8945
18880/60000 [========>.....................] - ETA: 1:09 - loss: 0.3397 - categorical_accuracy: 0.8945
18912/60000 [========>.....................] - ETA: 1:09 - loss: 0.3391 - categorical_accuracy: 0.8947
18944/60000 [========>.....................] - ETA: 1:09 - loss: 0.3386 - categorical_accuracy: 0.8948
18976/60000 [========>.....................] - ETA: 1:09 - loss: 0.3384 - categorical_accuracy: 0.8950
19008/60000 [========>.....................] - ETA: 1:09 - loss: 0.3384 - categorical_accuracy: 0.8950
19040/60000 [========>.....................] - ETA: 1:09 - loss: 0.3379 - categorical_accuracy: 0.8952
19072/60000 [========>.....................] - ETA: 1:09 - loss: 0.3375 - categorical_accuracy: 0.8953
19104/60000 [========>.....................] - ETA: 1:09 - loss: 0.3371 - categorical_accuracy: 0.8954
19136/60000 [========>.....................] - ETA: 1:09 - loss: 0.3370 - categorical_accuracy: 0.8955
19168/60000 [========>.....................] - ETA: 1:09 - loss: 0.3367 - categorical_accuracy: 0.8956
19232/60000 [========>.....................] - ETA: 1:09 - loss: 0.3360 - categorical_accuracy: 0.8959
19264/60000 [========>.....................] - ETA: 1:08 - loss: 0.3356 - categorical_accuracy: 0.8959
19296/60000 [========>.....................] - ETA: 1:08 - loss: 0.3354 - categorical_accuracy: 0.8959
19328/60000 [========>.....................] - ETA: 1:08 - loss: 0.3350 - categorical_accuracy: 0.8960
19360/60000 [========>.....................] - ETA: 1:08 - loss: 0.3347 - categorical_accuracy: 0.8961
19392/60000 [========>.....................] - ETA: 1:08 - loss: 0.3342 - categorical_accuracy: 0.8962
19424/60000 [========>.....................] - ETA: 1:08 - loss: 0.3338 - categorical_accuracy: 0.8963
19456/60000 [========>.....................] - ETA: 1:08 - loss: 0.3333 - categorical_accuracy: 0.8965
19488/60000 [========>.....................] - ETA: 1:08 - loss: 0.3333 - categorical_accuracy: 0.8964
19520/60000 [========>.....................] - ETA: 1:08 - loss: 0.3329 - categorical_accuracy: 0.8966
19552/60000 [========>.....................] - ETA: 1:08 - loss: 0.3328 - categorical_accuracy: 0.8965
19584/60000 [========>.....................] - ETA: 1:08 - loss: 0.3325 - categorical_accuracy: 0.8967
19616/60000 [========>.....................] - ETA: 1:08 - loss: 0.3321 - categorical_accuracy: 0.8968
19648/60000 [========>.....................] - ETA: 1:08 - loss: 0.3322 - categorical_accuracy: 0.8967
19680/60000 [========>.....................] - ETA: 1:08 - loss: 0.3318 - categorical_accuracy: 0.8968
19712/60000 [========>.....................] - ETA: 1:08 - loss: 0.3316 - categorical_accuracy: 0.8969
19744/60000 [========>.....................] - ETA: 1:08 - loss: 0.3316 - categorical_accuracy: 0.8969
19776/60000 [========>.....................] - ETA: 1:08 - loss: 0.3314 - categorical_accuracy: 0.8970
19808/60000 [========>.....................] - ETA: 1:07 - loss: 0.3309 - categorical_accuracy: 0.8972
19872/60000 [========>.....................] - ETA: 1:07 - loss: 0.3308 - categorical_accuracy: 0.8972
19904/60000 [========>.....................] - ETA: 1:07 - loss: 0.3304 - categorical_accuracy: 0.8974
19936/60000 [========>.....................] - ETA: 1:07 - loss: 0.3300 - categorical_accuracy: 0.8975
19968/60000 [========>.....................] - ETA: 1:07 - loss: 0.3300 - categorical_accuracy: 0.8974
20000/60000 [=========>....................] - ETA: 1:07 - loss: 0.3297 - categorical_accuracy: 0.8974
20032/60000 [=========>....................] - ETA: 1:07 - loss: 0.3296 - categorical_accuracy: 0.8976
20064/60000 [=========>....................] - ETA: 1:07 - loss: 0.3292 - categorical_accuracy: 0.8977
20096/60000 [=========>....................] - ETA: 1:07 - loss: 0.3287 - categorical_accuracy: 0.8978
20128/60000 [=========>....................] - ETA: 1:07 - loss: 0.3284 - categorical_accuracy: 0.8979
20160/60000 [=========>....................] - ETA: 1:07 - loss: 0.3284 - categorical_accuracy: 0.8979
20192/60000 [=========>....................] - ETA: 1:07 - loss: 0.3280 - categorical_accuracy: 0.8981
20224/60000 [=========>....................] - ETA: 1:07 - loss: 0.3277 - categorical_accuracy: 0.8982
20256/60000 [=========>....................] - ETA: 1:07 - loss: 0.3272 - categorical_accuracy: 0.8984
20288/60000 [=========>....................] - ETA: 1:07 - loss: 0.3270 - categorical_accuracy: 0.8985
20320/60000 [=========>....................] - ETA: 1:07 - loss: 0.3268 - categorical_accuracy: 0.8986
20352/60000 [=========>....................] - ETA: 1:07 - loss: 0.3263 - categorical_accuracy: 0.8987
20384/60000 [=========>....................] - ETA: 1:07 - loss: 0.3259 - categorical_accuracy: 0.8989
20416/60000 [=========>....................] - ETA: 1:06 - loss: 0.3258 - categorical_accuracy: 0.8990
20480/60000 [=========>....................] - ETA: 1:06 - loss: 0.3251 - categorical_accuracy: 0.8992
20512/60000 [=========>....................] - ETA: 1:06 - loss: 0.3249 - categorical_accuracy: 0.8992
20544/60000 [=========>....................] - ETA: 1:06 - loss: 0.3249 - categorical_accuracy: 0.8993
20576/60000 [=========>....................] - ETA: 1:06 - loss: 0.3249 - categorical_accuracy: 0.8993
20608/60000 [=========>....................] - ETA: 1:06 - loss: 0.3245 - categorical_accuracy: 0.8995
20640/60000 [=========>....................] - ETA: 1:06 - loss: 0.3242 - categorical_accuracy: 0.8996
20672/60000 [=========>....................] - ETA: 1:06 - loss: 0.3240 - categorical_accuracy: 0.8996
20704/60000 [=========>....................] - ETA: 1:06 - loss: 0.3238 - categorical_accuracy: 0.8996
20736/60000 [=========>....................] - ETA: 1:06 - loss: 0.3234 - categorical_accuracy: 0.8997
20768/60000 [=========>....................] - ETA: 1:06 - loss: 0.3230 - categorical_accuracy: 0.8999
20800/60000 [=========>....................] - ETA: 1:06 - loss: 0.3226 - categorical_accuracy: 0.9000
20832/60000 [=========>....................] - ETA: 1:06 - loss: 0.3223 - categorical_accuracy: 0.9001
20864/60000 [=========>....................] - ETA: 1:06 - loss: 0.3220 - categorical_accuracy: 0.9002
20896/60000 [=========>....................] - ETA: 1:06 - loss: 0.3216 - categorical_accuracy: 0.9003
20960/60000 [=========>....................] - ETA: 1:06 - loss: 0.3214 - categorical_accuracy: 0.9005
20992/60000 [=========>....................] - ETA: 1:05 - loss: 0.3213 - categorical_accuracy: 0.9006
21024/60000 [=========>....................] - ETA: 1:05 - loss: 0.3209 - categorical_accuracy: 0.9007
21056/60000 [=========>....................] - ETA: 1:05 - loss: 0.3207 - categorical_accuracy: 0.9007
21088/60000 [=========>....................] - ETA: 1:05 - loss: 0.3204 - categorical_accuracy: 0.9008
21152/60000 [=========>....................] - ETA: 1:05 - loss: 0.3200 - categorical_accuracy: 0.9009
21184/60000 [=========>....................] - ETA: 1:05 - loss: 0.3203 - categorical_accuracy: 0.9010
21216/60000 [=========>....................] - ETA: 1:05 - loss: 0.3201 - categorical_accuracy: 0.9010
21248/60000 [=========>....................] - ETA: 1:05 - loss: 0.3200 - categorical_accuracy: 0.9010
21280/60000 [=========>....................] - ETA: 1:05 - loss: 0.3197 - categorical_accuracy: 0.9011
21312/60000 [=========>....................] - ETA: 1:05 - loss: 0.3193 - categorical_accuracy: 0.9013
21344/60000 [=========>....................] - ETA: 1:05 - loss: 0.3190 - categorical_accuracy: 0.9013
21376/60000 [=========>....................] - ETA: 1:05 - loss: 0.3186 - categorical_accuracy: 0.9015
21408/60000 [=========>....................] - ETA: 1:05 - loss: 0.3181 - categorical_accuracy: 0.9016
21440/60000 [=========>....................] - ETA: 1:05 - loss: 0.3178 - categorical_accuracy: 0.9017
21472/60000 [=========>....................] - ETA: 1:05 - loss: 0.3175 - categorical_accuracy: 0.9018
21504/60000 [=========>....................] - ETA: 1:05 - loss: 0.3171 - categorical_accuracy: 0.9018
21536/60000 [=========>....................] - ETA: 1:05 - loss: 0.3170 - categorical_accuracy: 0.9019
21568/60000 [=========>....................] - ETA: 1:05 - loss: 0.3167 - categorical_accuracy: 0.9020
21600/60000 [=========>....................] - ETA: 1:04 - loss: 0.3166 - categorical_accuracy: 0.9020
21632/60000 [=========>....................] - ETA: 1:04 - loss: 0.3165 - categorical_accuracy: 0.9020
21664/60000 [=========>....................] - ETA: 1:04 - loss: 0.3163 - categorical_accuracy: 0.9021
21696/60000 [=========>....................] - ETA: 1:04 - loss: 0.3162 - categorical_accuracy: 0.9022
21728/60000 [=========>....................] - ETA: 1:04 - loss: 0.3159 - categorical_accuracy: 0.9022
21760/60000 [=========>....................] - ETA: 1:04 - loss: 0.3156 - categorical_accuracy: 0.9023
21792/60000 [=========>....................] - ETA: 1:04 - loss: 0.3156 - categorical_accuracy: 0.9023
21824/60000 [=========>....................] - ETA: 1:04 - loss: 0.3154 - categorical_accuracy: 0.9023
21856/60000 [=========>....................] - ETA: 1:04 - loss: 0.3154 - categorical_accuracy: 0.9023
21888/60000 [=========>....................] - ETA: 1:04 - loss: 0.3153 - categorical_accuracy: 0.9023
21920/60000 [=========>....................] - ETA: 1:04 - loss: 0.3150 - categorical_accuracy: 0.9024
21952/60000 [=========>....................] - ETA: 1:04 - loss: 0.3146 - categorical_accuracy: 0.9026
21984/60000 [=========>....................] - ETA: 1:04 - loss: 0.3143 - categorical_accuracy: 0.9026
22016/60000 [==========>...................] - ETA: 1:04 - loss: 0.3143 - categorical_accuracy: 0.9027
22048/60000 [==========>...................] - ETA: 1:04 - loss: 0.3141 - categorical_accuracy: 0.9028
22080/60000 [==========>...................] - ETA: 1:04 - loss: 0.3138 - categorical_accuracy: 0.9029
22112/60000 [==========>...................] - ETA: 1:04 - loss: 0.3135 - categorical_accuracy: 0.9030
22144/60000 [==========>...................] - ETA: 1:04 - loss: 0.3132 - categorical_accuracy: 0.9031
22176/60000 [==========>...................] - ETA: 1:04 - loss: 0.3130 - categorical_accuracy: 0.9032
22240/60000 [==========>...................] - ETA: 1:03 - loss: 0.3124 - categorical_accuracy: 0.9034
22304/60000 [==========>...................] - ETA: 1:03 - loss: 0.3118 - categorical_accuracy: 0.9036
22336/60000 [==========>...................] - ETA: 1:03 - loss: 0.3116 - categorical_accuracy: 0.9037
22368/60000 [==========>...................] - ETA: 1:03 - loss: 0.3115 - categorical_accuracy: 0.9037
22400/60000 [==========>...................] - ETA: 1:03 - loss: 0.3115 - categorical_accuracy: 0.9038
22432/60000 [==========>...................] - ETA: 1:03 - loss: 0.3111 - categorical_accuracy: 0.9039
22464/60000 [==========>...................] - ETA: 1:03 - loss: 0.3111 - categorical_accuracy: 0.9040
22496/60000 [==========>...................] - ETA: 1:03 - loss: 0.3110 - categorical_accuracy: 0.9040
22528/60000 [==========>...................] - ETA: 1:03 - loss: 0.3107 - categorical_accuracy: 0.9041
22560/60000 [==========>...................] - ETA: 1:03 - loss: 0.3105 - categorical_accuracy: 0.9042
22592/60000 [==========>...................] - ETA: 1:03 - loss: 0.3104 - categorical_accuracy: 0.9043
22624/60000 [==========>...................] - ETA: 1:03 - loss: 0.3101 - categorical_accuracy: 0.9043
22656/60000 [==========>...................] - ETA: 1:03 - loss: 0.3100 - categorical_accuracy: 0.9044
22688/60000 [==========>...................] - ETA: 1:03 - loss: 0.3097 - categorical_accuracy: 0.9044
22720/60000 [==========>...................] - ETA: 1:03 - loss: 0.3094 - categorical_accuracy: 0.9045
22752/60000 [==========>...................] - ETA: 1:03 - loss: 0.3091 - categorical_accuracy: 0.9045
22784/60000 [==========>...................] - ETA: 1:03 - loss: 0.3091 - categorical_accuracy: 0.9045
22816/60000 [==========>...................] - ETA: 1:03 - loss: 0.3089 - categorical_accuracy: 0.9046
22848/60000 [==========>...................] - ETA: 1:02 - loss: 0.3087 - categorical_accuracy: 0.9047
22880/60000 [==========>...................] - ETA: 1:02 - loss: 0.3085 - categorical_accuracy: 0.9048
22912/60000 [==========>...................] - ETA: 1:02 - loss: 0.3083 - categorical_accuracy: 0.9048
22944/60000 [==========>...................] - ETA: 1:02 - loss: 0.3080 - categorical_accuracy: 0.9049
22976/60000 [==========>...................] - ETA: 1:02 - loss: 0.3078 - categorical_accuracy: 0.9050
23008/60000 [==========>...................] - ETA: 1:02 - loss: 0.3074 - categorical_accuracy: 0.9051
23040/60000 [==========>...................] - ETA: 1:02 - loss: 0.3072 - categorical_accuracy: 0.9052
23072/60000 [==========>...................] - ETA: 1:02 - loss: 0.3074 - categorical_accuracy: 0.9053
23104/60000 [==========>...................] - ETA: 1:02 - loss: 0.3071 - categorical_accuracy: 0.9053
23136/60000 [==========>...................] - ETA: 1:02 - loss: 0.3069 - categorical_accuracy: 0.9053
23168/60000 [==========>...................] - ETA: 1:02 - loss: 0.3068 - categorical_accuracy: 0.9054
23200/60000 [==========>...................] - ETA: 1:02 - loss: 0.3067 - categorical_accuracy: 0.9054
23232/60000 [==========>...................] - ETA: 1:02 - loss: 0.3065 - categorical_accuracy: 0.9055
23264/60000 [==========>...................] - ETA: 1:02 - loss: 0.3062 - categorical_accuracy: 0.9056
23296/60000 [==========>...................] - ETA: 1:02 - loss: 0.3059 - categorical_accuracy: 0.9057
23328/60000 [==========>...................] - ETA: 1:02 - loss: 0.3057 - categorical_accuracy: 0.9057
23360/60000 [==========>...................] - ETA: 1:02 - loss: 0.3055 - categorical_accuracy: 0.9058
23392/60000 [==========>...................] - ETA: 1:02 - loss: 0.3052 - categorical_accuracy: 0.9059
23424/60000 [==========>...................] - ETA: 1:02 - loss: 0.3053 - categorical_accuracy: 0.9060
23456/60000 [==========>...................] - ETA: 1:01 - loss: 0.3051 - categorical_accuracy: 0.9060
23488/60000 [==========>...................] - ETA: 1:01 - loss: 0.3049 - categorical_accuracy: 0.9060
23520/60000 [==========>...................] - ETA: 1:01 - loss: 0.3047 - categorical_accuracy: 0.9061
23552/60000 [==========>...................] - ETA: 1:01 - loss: 0.3044 - categorical_accuracy: 0.9062
23584/60000 [==========>...................] - ETA: 1:01 - loss: 0.3040 - categorical_accuracy: 0.9063
23616/60000 [==========>...................] - ETA: 1:01 - loss: 0.3037 - categorical_accuracy: 0.9064
23648/60000 [==========>...................] - ETA: 1:01 - loss: 0.3034 - categorical_accuracy: 0.9065
23680/60000 [==========>...................] - ETA: 1:01 - loss: 0.3030 - categorical_accuracy: 0.9066
23712/60000 [==========>...................] - ETA: 1:01 - loss: 0.3031 - categorical_accuracy: 0.9067
23744/60000 [==========>...................] - ETA: 1:01 - loss: 0.3031 - categorical_accuracy: 0.9067
23776/60000 [==========>...................] - ETA: 1:01 - loss: 0.3028 - categorical_accuracy: 0.9068
23808/60000 [==========>...................] - ETA: 1:01 - loss: 0.3024 - categorical_accuracy: 0.9069
23840/60000 [==========>...................] - ETA: 1:01 - loss: 0.3022 - categorical_accuracy: 0.9069
23872/60000 [==========>...................] - ETA: 1:01 - loss: 0.3019 - categorical_accuracy: 0.9070
23904/60000 [==========>...................] - ETA: 1:01 - loss: 0.3017 - categorical_accuracy: 0.9070
23936/60000 [==========>...................] - ETA: 1:01 - loss: 0.3015 - categorical_accuracy: 0.9071
23968/60000 [==========>...................] - ETA: 1:01 - loss: 0.3013 - categorical_accuracy: 0.9072
24000/60000 [===========>..................] - ETA: 1:01 - loss: 0.3014 - categorical_accuracy: 0.9072
24032/60000 [===========>..................] - ETA: 1:01 - loss: 0.3011 - categorical_accuracy: 0.9073
24064/60000 [===========>..................] - ETA: 1:00 - loss: 0.3008 - categorical_accuracy: 0.9073
24096/60000 [===========>..................] - ETA: 1:00 - loss: 0.3004 - categorical_accuracy: 0.9075
24128/60000 [===========>..................] - ETA: 1:00 - loss: 0.3003 - categorical_accuracy: 0.9075
24160/60000 [===========>..................] - ETA: 1:00 - loss: 0.3001 - categorical_accuracy: 0.9075
24192/60000 [===========>..................] - ETA: 1:00 - loss: 0.3001 - categorical_accuracy: 0.9074
24224/60000 [===========>..................] - ETA: 1:00 - loss: 0.3003 - categorical_accuracy: 0.9073
24256/60000 [===========>..................] - ETA: 1:00 - loss: 0.3008 - categorical_accuracy: 0.9071
24288/60000 [===========>..................] - ETA: 1:00 - loss: 0.3006 - categorical_accuracy: 0.9072
24320/60000 [===========>..................] - ETA: 1:00 - loss: 0.3004 - categorical_accuracy: 0.9072
24352/60000 [===========>..................] - ETA: 1:00 - loss: 0.3001 - categorical_accuracy: 0.9074
24384/60000 [===========>..................] - ETA: 1:00 - loss: 0.2999 - categorical_accuracy: 0.9074
24416/60000 [===========>..................] - ETA: 1:00 - loss: 0.2997 - categorical_accuracy: 0.9075
24448/60000 [===========>..................] - ETA: 1:00 - loss: 0.2994 - categorical_accuracy: 0.9076
24480/60000 [===========>..................] - ETA: 1:00 - loss: 0.2994 - categorical_accuracy: 0.9076
24512/60000 [===========>..................] - ETA: 1:00 - loss: 0.2991 - categorical_accuracy: 0.9076
24544/60000 [===========>..................] - ETA: 1:00 - loss: 0.2988 - categorical_accuracy: 0.9078
24576/60000 [===========>..................] - ETA: 1:00 - loss: 0.2985 - categorical_accuracy: 0.9079
24608/60000 [===========>..................] - ETA: 1:00 - loss: 0.2984 - categorical_accuracy: 0.9079
24640/60000 [===========>..................] - ETA: 59s - loss: 0.2985 - categorical_accuracy: 0.9079 
24672/60000 [===========>..................] - ETA: 59s - loss: 0.2983 - categorical_accuracy: 0.9080
24704/60000 [===========>..................] - ETA: 59s - loss: 0.2979 - categorical_accuracy: 0.9081
24736/60000 [===========>..................] - ETA: 59s - loss: 0.2976 - categorical_accuracy: 0.9082
24768/60000 [===========>..................] - ETA: 59s - loss: 0.2976 - categorical_accuracy: 0.9082
24800/60000 [===========>..................] - ETA: 59s - loss: 0.2976 - categorical_accuracy: 0.9083
24832/60000 [===========>..................] - ETA: 59s - loss: 0.2973 - categorical_accuracy: 0.9083
24864/60000 [===========>..................] - ETA: 59s - loss: 0.2971 - categorical_accuracy: 0.9084
24896/60000 [===========>..................] - ETA: 59s - loss: 0.2968 - categorical_accuracy: 0.9085
24928/60000 [===========>..................] - ETA: 59s - loss: 0.2965 - categorical_accuracy: 0.9086
24960/60000 [===========>..................] - ETA: 59s - loss: 0.2962 - categorical_accuracy: 0.9087
24992/60000 [===========>..................] - ETA: 59s - loss: 0.2959 - categorical_accuracy: 0.9088
25024/60000 [===========>..................] - ETA: 59s - loss: 0.2957 - categorical_accuracy: 0.9088
25056/60000 [===========>..................] - ETA: 59s - loss: 0.2954 - categorical_accuracy: 0.9089
25088/60000 [===========>..................] - ETA: 59s - loss: 0.2951 - categorical_accuracy: 0.9090
25120/60000 [===========>..................] - ETA: 59s - loss: 0.2949 - categorical_accuracy: 0.9091
25152/60000 [===========>..................] - ETA: 59s - loss: 0.2947 - categorical_accuracy: 0.9091
25216/60000 [===========>..................] - ETA: 59s - loss: 0.2943 - categorical_accuracy: 0.9092
25248/60000 [===========>..................] - ETA: 58s - loss: 0.2942 - categorical_accuracy: 0.9091
25280/60000 [===========>..................] - ETA: 58s - loss: 0.2944 - categorical_accuracy: 0.9091
25312/60000 [===========>..................] - ETA: 58s - loss: 0.2941 - categorical_accuracy: 0.9093
25344/60000 [===========>..................] - ETA: 58s - loss: 0.2939 - categorical_accuracy: 0.9093
25376/60000 [===========>..................] - ETA: 58s - loss: 0.2936 - categorical_accuracy: 0.9094
25408/60000 [===========>..................] - ETA: 58s - loss: 0.2935 - categorical_accuracy: 0.9095
25440/60000 [===========>..................] - ETA: 58s - loss: 0.2934 - categorical_accuracy: 0.9095
25472/60000 [===========>..................] - ETA: 58s - loss: 0.2933 - categorical_accuracy: 0.9095
25504/60000 [===========>..................] - ETA: 58s - loss: 0.2930 - categorical_accuracy: 0.9097
25536/60000 [===========>..................] - ETA: 58s - loss: 0.2928 - categorical_accuracy: 0.9097
25568/60000 [===========>..................] - ETA: 58s - loss: 0.2926 - categorical_accuracy: 0.9098
25632/60000 [===========>..................] - ETA: 58s - loss: 0.2920 - categorical_accuracy: 0.9100
25664/60000 [===========>..................] - ETA: 58s - loss: 0.2918 - categorical_accuracy: 0.9100
25696/60000 [===========>..................] - ETA: 58s - loss: 0.2916 - categorical_accuracy: 0.9101
25728/60000 [===========>..................] - ETA: 58s - loss: 0.2915 - categorical_accuracy: 0.9101
25760/60000 [===========>..................] - ETA: 58s - loss: 0.2920 - categorical_accuracy: 0.9100
25792/60000 [===========>..................] - ETA: 58s - loss: 0.2919 - categorical_accuracy: 0.9100
25824/60000 [===========>..................] - ETA: 57s - loss: 0.2917 - categorical_accuracy: 0.9101
25856/60000 [===========>..................] - ETA: 57s - loss: 0.2914 - categorical_accuracy: 0.9102
25888/60000 [===========>..................] - ETA: 57s - loss: 0.2914 - categorical_accuracy: 0.9102
25920/60000 [===========>..................] - ETA: 57s - loss: 0.2912 - categorical_accuracy: 0.9102
25952/60000 [===========>..................] - ETA: 57s - loss: 0.2909 - categorical_accuracy: 0.9103
25984/60000 [===========>..................] - ETA: 57s - loss: 0.2908 - categorical_accuracy: 0.9103
26016/60000 [============>.................] - ETA: 57s - loss: 0.2906 - categorical_accuracy: 0.9104
26048/60000 [============>.................] - ETA: 57s - loss: 0.2903 - categorical_accuracy: 0.9105
26080/60000 [============>.................] - ETA: 57s - loss: 0.2900 - categorical_accuracy: 0.9106
26112/60000 [============>.................] - ETA: 57s - loss: 0.2897 - categorical_accuracy: 0.9107
26144/60000 [============>.................] - ETA: 57s - loss: 0.2896 - categorical_accuracy: 0.9107
26176/60000 [============>.................] - ETA: 57s - loss: 0.2893 - categorical_accuracy: 0.9108
26208/60000 [============>.................] - ETA: 57s - loss: 0.2892 - categorical_accuracy: 0.9108
26240/60000 [============>.................] - ETA: 57s - loss: 0.2890 - categorical_accuracy: 0.9109
26304/60000 [============>.................] - ETA: 57s - loss: 0.2887 - categorical_accuracy: 0.9110
26336/60000 [============>.................] - ETA: 57s - loss: 0.2885 - categorical_accuracy: 0.9110
26368/60000 [============>.................] - ETA: 57s - loss: 0.2883 - categorical_accuracy: 0.9111
26400/60000 [============>.................] - ETA: 56s - loss: 0.2881 - categorical_accuracy: 0.9111
26432/60000 [============>.................] - ETA: 56s - loss: 0.2882 - categorical_accuracy: 0.9111
26464/60000 [============>.................] - ETA: 56s - loss: 0.2878 - categorical_accuracy: 0.9112
26496/60000 [============>.................] - ETA: 56s - loss: 0.2879 - categorical_accuracy: 0.9112
26528/60000 [============>.................] - ETA: 56s - loss: 0.2876 - categorical_accuracy: 0.9112
26560/60000 [============>.................] - ETA: 56s - loss: 0.2874 - categorical_accuracy: 0.9113
26592/60000 [============>.................] - ETA: 56s - loss: 0.2873 - categorical_accuracy: 0.9114
26624/60000 [============>.................] - ETA: 56s - loss: 0.2870 - categorical_accuracy: 0.9114
26656/60000 [============>.................] - ETA: 56s - loss: 0.2867 - categorical_accuracy: 0.9115
26720/60000 [============>.................] - ETA: 56s - loss: 0.2866 - categorical_accuracy: 0.9116
26752/60000 [============>.................] - ETA: 56s - loss: 0.2864 - categorical_accuracy: 0.9117
26784/60000 [============>.................] - ETA: 56s - loss: 0.2861 - categorical_accuracy: 0.9118
26848/60000 [============>.................] - ETA: 56s - loss: 0.2858 - categorical_accuracy: 0.9119
26880/60000 [============>.................] - ETA: 56s - loss: 0.2856 - categorical_accuracy: 0.9120
26912/60000 [============>.................] - ETA: 56s - loss: 0.2854 - categorical_accuracy: 0.9120
26944/60000 [============>.................] - ETA: 56s - loss: 0.2854 - categorical_accuracy: 0.9120
26976/60000 [============>.................] - ETA: 55s - loss: 0.2854 - categorical_accuracy: 0.9120
27008/60000 [============>.................] - ETA: 55s - loss: 0.2852 - categorical_accuracy: 0.9120
27040/60000 [============>.................] - ETA: 55s - loss: 0.2849 - categorical_accuracy: 0.9121
27072/60000 [============>.................] - ETA: 55s - loss: 0.2846 - categorical_accuracy: 0.9122
27104/60000 [============>.................] - ETA: 55s - loss: 0.2845 - categorical_accuracy: 0.9122
27136/60000 [============>.................] - ETA: 55s - loss: 0.2842 - categorical_accuracy: 0.9123
27168/60000 [============>.................] - ETA: 55s - loss: 0.2839 - categorical_accuracy: 0.9124
27200/60000 [============>.................] - ETA: 55s - loss: 0.2836 - categorical_accuracy: 0.9125
27232/60000 [============>.................] - ETA: 55s - loss: 0.2833 - categorical_accuracy: 0.9126
27264/60000 [============>.................] - ETA: 55s - loss: 0.2831 - categorical_accuracy: 0.9126
27296/60000 [============>.................] - ETA: 55s - loss: 0.2829 - categorical_accuracy: 0.9127
27328/60000 [============>.................] - ETA: 55s - loss: 0.2826 - categorical_accuracy: 0.9128
27360/60000 [============>.................] - ETA: 55s - loss: 0.2824 - categorical_accuracy: 0.9128
27392/60000 [============>.................] - ETA: 55s - loss: 0.2821 - categorical_accuracy: 0.9129
27456/60000 [============>.................] - ETA: 55s - loss: 0.2816 - categorical_accuracy: 0.9131
27520/60000 [============>.................] - ETA: 55s - loss: 0.2811 - categorical_accuracy: 0.9132
27584/60000 [============>.................] - ETA: 54s - loss: 0.2805 - categorical_accuracy: 0.9134
27616/60000 [============>.................] - ETA: 54s - loss: 0.2802 - categorical_accuracy: 0.9135
27648/60000 [============>.................] - ETA: 54s - loss: 0.2799 - categorical_accuracy: 0.9136
27680/60000 [============>.................] - ETA: 54s - loss: 0.2797 - categorical_accuracy: 0.9137
27712/60000 [============>.................] - ETA: 54s - loss: 0.2794 - categorical_accuracy: 0.9138
27744/60000 [============>.................] - ETA: 54s - loss: 0.2793 - categorical_accuracy: 0.9137
27776/60000 [============>.................] - ETA: 54s - loss: 0.2792 - categorical_accuracy: 0.9138
27808/60000 [============>.................] - ETA: 54s - loss: 0.2789 - categorical_accuracy: 0.9139
27840/60000 [============>.................] - ETA: 54s - loss: 0.2787 - categorical_accuracy: 0.9140
27872/60000 [============>.................] - ETA: 54s - loss: 0.2784 - categorical_accuracy: 0.9141
27904/60000 [============>.................] - ETA: 54s - loss: 0.2782 - categorical_accuracy: 0.9141
27936/60000 [============>.................] - ETA: 54s - loss: 0.2781 - categorical_accuracy: 0.9142
27968/60000 [============>.................] - ETA: 54s - loss: 0.2780 - categorical_accuracy: 0.9142
28000/60000 [=============>................] - ETA: 54s - loss: 0.2777 - categorical_accuracy: 0.9143
28032/60000 [=============>................] - ETA: 54s - loss: 0.2778 - categorical_accuracy: 0.9143
28064/60000 [=============>................] - ETA: 54s - loss: 0.2777 - categorical_accuracy: 0.9143
28096/60000 [=============>................] - ETA: 54s - loss: 0.2775 - categorical_accuracy: 0.9144
28160/60000 [=============>................] - ETA: 53s - loss: 0.2771 - categorical_accuracy: 0.9145
28192/60000 [=============>................] - ETA: 53s - loss: 0.2771 - categorical_accuracy: 0.9146
28224/60000 [=============>................] - ETA: 53s - loss: 0.2768 - categorical_accuracy: 0.9146
28256/60000 [=============>................] - ETA: 53s - loss: 0.2767 - categorical_accuracy: 0.9146
28288/60000 [=============>................] - ETA: 53s - loss: 0.2765 - categorical_accuracy: 0.9147
28352/60000 [=============>................] - ETA: 53s - loss: 0.2760 - categorical_accuracy: 0.9148
28384/60000 [=============>................] - ETA: 53s - loss: 0.2759 - categorical_accuracy: 0.9148
28416/60000 [=============>................] - ETA: 53s - loss: 0.2756 - categorical_accuracy: 0.9149
28448/60000 [=============>................] - ETA: 53s - loss: 0.2754 - categorical_accuracy: 0.9150
28480/60000 [=============>................] - ETA: 53s - loss: 0.2753 - categorical_accuracy: 0.9151
28512/60000 [=============>................] - ETA: 53s - loss: 0.2750 - categorical_accuracy: 0.9152
28544/60000 [=============>................] - ETA: 53s - loss: 0.2749 - categorical_accuracy: 0.9152
28576/60000 [=============>................] - ETA: 53s - loss: 0.2746 - categorical_accuracy: 0.9153
28608/60000 [=============>................] - ETA: 53s - loss: 0.2744 - categorical_accuracy: 0.9154
28640/60000 [=============>................] - ETA: 53s - loss: 0.2742 - categorical_accuracy: 0.9155
28672/60000 [=============>................] - ETA: 53s - loss: 0.2743 - categorical_accuracy: 0.9155
28736/60000 [=============>................] - ETA: 52s - loss: 0.2739 - categorical_accuracy: 0.9156
28768/60000 [=============>................] - ETA: 52s - loss: 0.2737 - categorical_accuracy: 0.9157
28800/60000 [=============>................] - ETA: 52s - loss: 0.2737 - categorical_accuracy: 0.9157
28832/60000 [=============>................] - ETA: 52s - loss: 0.2736 - categorical_accuracy: 0.9158
28864/60000 [=============>................] - ETA: 52s - loss: 0.2733 - categorical_accuracy: 0.9158
28896/60000 [=============>................] - ETA: 52s - loss: 0.2732 - categorical_accuracy: 0.9159
28928/60000 [=============>................] - ETA: 52s - loss: 0.2734 - categorical_accuracy: 0.9158
28960/60000 [=============>................] - ETA: 52s - loss: 0.2733 - categorical_accuracy: 0.9159
28992/60000 [=============>................] - ETA: 52s - loss: 0.2731 - categorical_accuracy: 0.9159
29024/60000 [=============>................] - ETA: 52s - loss: 0.2729 - categorical_accuracy: 0.9160
29056/60000 [=============>................] - ETA: 52s - loss: 0.2726 - categorical_accuracy: 0.9161
29088/60000 [=============>................] - ETA: 52s - loss: 0.2724 - categorical_accuracy: 0.9161
29120/60000 [=============>................] - ETA: 52s - loss: 0.2723 - categorical_accuracy: 0.9161
29152/60000 [=============>................] - ETA: 52s - loss: 0.2720 - categorical_accuracy: 0.9162
29184/60000 [=============>................] - ETA: 52s - loss: 0.2718 - categorical_accuracy: 0.9162
29216/60000 [=============>................] - ETA: 52s - loss: 0.2716 - categorical_accuracy: 0.9163
29280/60000 [=============>................] - ETA: 52s - loss: 0.2713 - categorical_accuracy: 0.9164
29312/60000 [=============>................] - ETA: 51s - loss: 0.2712 - categorical_accuracy: 0.9164
29344/60000 [=============>................] - ETA: 51s - loss: 0.2709 - categorical_accuracy: 0.9165
29376/60000 [=============>................] - ETA: 51s - loss: 0.2709 - categorical_accuracy: 0.9165
29408/60000 [=============>................] - ETA: 51s - loss: 0.2710 - categorical_accuracy: 0.9165
29440/60000 [=============>................] - ETA: 51s - loss: 0.2708 - categorical_accuracy: 0.9165
29472/60000 [=============>................] - ETA: 51s - loss: 0.2706 - categorical_accuracy: 0.9166
29504/60000 [=============>................] - ETA: 51s - loss: 0.2703 - categorical_accuracy: 0.9167
29536/60000 [=============>................] - ETA: 51s - loss: 0.2701 - categorical_accuracy: 0.9168
29568/60000 [=============>................] - ETA: 51s - loss: 0.2698 - categorical_accuracy: 0.9169
29600/60000 [=============>................] - ETA: 51s - loss: 0.2698 - categorical_accuracy: 0.9169
29632/60000 [=============>................] - ETA: 51s - loss: 0.2695 - categorical_accuracy: 0.9169
29664/60000 [=============>................] - ETA: 51s - loss: 0.2694 - categorical_accuracy: 0.9170
29696/60000 [=============>................] - ETA: 51s - loss: 0.2695 - categorical_accuracy: 0.9170
29728/60000 [=============>................] - ETA: 51s - loss: 0.2692 - categorical_accuracy: 0.9170
29760/60000 [=============>................] - ETA: 51s - loss: 0.2690 - categorical_accuracy: 0.9171
29792/60000 [=============>................] - ETA: 51s - loss: 0.2688 - categorical_accuracy: 0.9172
29824/60000 [=============>................] - ETA: 51s - loss: 0.2686 - categorical_accuracy: 0.9172
29856/60000 [=============>................] - ETA: 51s - loss: 0.2684 - categorical_accuracy: 0.9173
29888/60000 [=============>................] - ETA: 50s - loss: 0.2682 - categorical_accuracy: 0.9174
29952/60000 [=============>................] - ETA: 50s - loss: 0.2678 - categorical_accuracy: 0.9175
29984/60000 [=============>................] - ETA: 50s - loss: 0.2676 - categorical_accuracy: 0.9176
30016/60000 [==============>...............] - ETA: 50s - loss: 0.2675 - categorical_accuracy: 0.9176
30048/60000 [==============>...............] - ETA: 50s - loss: 0.2674 - categorical_accuracy: 0.9176
30080/60000 [==============>...............] - ETA: 50s - loss: 0.2673 - categorical_accuracy: 0.9177
30112/60000 [==============>...............] - ETA: 50s - loss: 0.2671 - categorical_accuracy: 0.9177
30144/60000 [==============>...............] - ETA: 50s - loss: 0.2669 - categorical_accuracy: 0.9178
30176/60000 [==============>...............] - ETA: 50s - loss: 0.2667 - categorical_accuracy: 0.9178
30208/60000 [==============>...............] - ETA: 50s - loss: 0.2668 - categorical_accuracy: 0.9178
30240/60000 [==============>...............] - ETA: 50s - loss: 0.2666 - categorical_accuracy: 0.9178
30272/60000 [==============>...............] - ETA: 50s - loss: 0.2664 - categorical_accuracy: 0.9179
30304/60000 [==============>...............] - ETA: 50s - loss: 0.2663 - categorical_accuracy: 0.9180
30336/60000 [==============>...............] - ETA: 50s - loss: 0.2663 - categorical_accuracy: 0.9180
30368/60000 [==============>...............] - ETA: 50s - loss: 0.2660 - categorical_accuracy: 0.9180
30400/60000 [==============>...............] - ETA: 50s - loss: 0.2660 - categorical_accuracy: 0.9181
30432/60000 [==============>...............] - ETA: 50s - loss: 0.2657 - categorical_accuracy: 0.9182
30464/60000 [==============>...............] - ETA: 49s - loss: 0.2655 - categorical_accuracy: 0.9182
30496/60000 [==============>...............] - ETA: 49s - loss: 0.2654 - categorical_accuracy: 0.9182
30528/60000 [==============>...............] - ETA: 49s - loss: 0.2654 - categorical_accuracy: 0.9182
30560/60000 [==============>...............] - ETA: 49s - loss: 0.2652 - categorical_accuracy: 0.9183
30624/60000 [==============>...............] - ETA: 49s - loss: 0.2650 - categorical_accuracy: 0.9183
30656/60000 [==============>...............] - ETA: 49s - loss: 0.2648 - categorical_accuracy: 0.9184
30688/60000 [==============>...............] - ETA: 49s - loss: 0.2645 - categorical_accuracy: 0.9185
30720/60000 [==============>...............] - ETA: 49s - loss: 0.2643 - categorical_accuracy: 0.9186
30752/60000 [==============>...............] - ETA: 49s - loss: 0.2641 - categorical_accuracy: 0.9186
30816/60000 [==============>...............] - ETA: 49s - loss: 0.2638 - categorical_accuracy: 0.9187
30848/60000 [==============>...............] - ETA: 49s - loss: 0.2635 - categorical_accuracy: 0.9188
30880/60000 [==============>...............] - ETA: 49s - loss: 0.2634 - categorical_accuracy: 0.9188
30912/60000 [==============>...............] - ETA: 49s - loss: 0.2631 - categorical_accuracy: 0.9189
30944/60000 [==============>...............] - ETA: 49s - loss: 0.2632 - categorical_accuracy: 0.9190
30976/60000 [==============>...............] - ETA: 49s - loss: 0.2629 - categorical_accuracy: 0.9190
31008/60000 [==============>...............] - ETA: 49s - loss: 0.2627 - categorical_accuracy: 0.9191
31040/60000 [==============>...............] - ETA: 48s - loss: 0.2625 - categorical_accuracy: 0.9192
31104/60000 [==============>...............] - ETA: 48s - loss: 0.2621 - categorical_accuracy: 0.9193
31168/60000 [==============>...............] - ETA: 48s - loss: 0.2618 - categorical_accuracy: 0.9193
31232/60000 [==============>...............] - ETA: 48s - loss: 0.2618 - categorical_accuracy: 0.9194
31264/60000 [==============>...............] - ETA: 48s - loss: 0.2616 - categorical_accuracy: 0.9195
31296/60000 [==============>...............] - ETA: 48s - loss: 0.2614 - categorical_accuracy: 0.9195
31328/60000 [==============>...............] - ETA: 48s - loss: 0.2614 - categorical_accuracy: 0.9195
31360/60000 [==============>...............] - ETA: 48s - loss: 0.2612 - categorical_accuracy: 0.9196
31392/60000 [==============>...............] - ETA: 48s - loss: 0.2609 - categorical_accuracy: 0.9197
31424/60000 [==============>...............] - ETA: 48s - loss: 0.2608 - categorical_accuracy: 0.9197
31456/60000 [==============>...............] - ETA: 48s - loss: 0.2606 - categorical_accuracy: 0.9197
31488/60000 [==============>...............] - ETA: 48s - loss: 0.2605 - categorical_accuracy: 0.9198
31520/60000 [==============>...............] - ETA: 48s - loss: 0.2603 - categorical_accuracy: 0.9198
31552/60000 [==============>...............] - ETA: 48s - loss: 0.2601 - categorical_accuracy: 0.9199
31584/60000 [==============>...............] - ETA: 48s - loss: 0.2599 - categorical_accuracy: 0.9200
31616/60000 [==============>...............] - ETA: 48s - loss: 0.2597 - categorical_accuracy: 0.9200
31648/60000 [==============>...............] - ETA: 47s - loss: 0.2599 - categorical_accuracy: 0.9200
31680/60000 [==============>...............] - ETA: 47s - loss: 0.2599 - categorical_accuracy: 0.9200
31744/60000 [==============>...............] - ETA: 47s - loss: 0.2596 - categorical_accuracy: 0.9201
31776/60000 [==============>...............] - ETA: 47s - loss: 0.2595 - categorical_accuracy: 0.9201
31808/60000 [==============>...............] - ETA: 47s - loss: 0.2593 - categorical_accuracy: 0.9202
31840/60000 [==============>...............] - ETA: 47s - loss: 0.2590 - categorical_accuracy: 0.9203
31872/60000 [==============>...............] - ETA: 47s - loss: 0.2588 - categorical_accuracy: 0.9203
31904/60000 [==============>...............] - ETA: 47s - loss: 0.2587 - categorical_accuracy: 0.9203
31936/60000 [==============>...............] - ETA: 47s - loss: 0.2585 - categorical_accuracy: 0.9204
31968/60000 [==============>...............] - ETA: 47s - loss: 0.2584 - categorical_accuracy: 0.9204
32000/60000 [===============>..............] - ETA: 47s - loss: 0.2582 - categorical_accuracy: 0.9205
32032/60000 [===============>..............] - ETA: 47s - loss: 0.2579 - categorical_accuracy: 0.9205
32064/60000 [===============>..............] - ETA: 47s - loss: 0.2580 - categorical_accuracy: 0.9206
32096/60000 [===============>..............] - ETA: 47s - loss: 0.2579 - categorical_accuracy: 0.9206
32128/60000 [===============>..............] - ETA: 47s - loss: 0.2582 - categorical_accuracy: 0.9205
32160/60000 [===============>..............] - ETA: 47s - loss: 0.2580 - categorical_accuracy: 0.9206
32192/60000 [===============>..............] - ETA: 47s - loss: 0.2579 - categorical_accuracy: 0.9207
32224/60000 [===============>..............] - ETA: 46s - loss: 0.2577 - categorical_accuracy: 0.9207
32256/60000 [===============>..............] - ETA: 46s - loss: 0.2575 - categorical_accuracy: 0.9208
32288/60000 [===============>..............] - ETA: 46s - loss: 0.2574 - categorical_accuracy: 0.9208
32320/60000 [===============>..............] - ETA: 46s - loss: 0.2572 - categorical_accuracy: 0.9209
32352/60000 [===============>..............] - ETA: 46s - loss: 0.2569 - categorical_accuracy: 0.9210
32384/60000 [===============>..............] - ETA: 46s - loss: 0.2569 - categorical_accuracy: 0.9210
32416/60000 [===============>..............] - ETA: 46s - loss: 0.2566 - categorical_accuracy: 0.9211
32448/60000 [===============>..............] - ETA: 46s - loss: 0.2566 - categorical_accuracy: 0.9211
32480/60000 [===============>..............] - ETA: 46s - loss: 0.2565 - categorical_accuracy: 0.9211
32512/60000 [===============>..............] - ETA: 46s - loss: 0.2565 - categorical_accuracy: 0.9211
32544/60000 [===============>..............] - ETA: 46s - loss: 0.2562 - categorical_accuracy: 0.9212
32576/60000 [===============>..............] - ETA: 46s - loss: 0.2561 - categorical_accuracy: 0.9212
32608/60000 [===============>..............] - ETA: 46s - loss: 0.2559 - categorical_accuracy: 0.9213
32640/60000 [===============>..............] - ETA: 46s - loss: 0.2558 - categorical_accuracy: 0.9214
32672/60000 [===============>..............] - ETA: 46s - loss: 0.2556 - categorical_accuracy: 0.9214
32704/60000 [===============>..............] - ETA: 46s - loss: 0.2555 - categorical_accuracy: 0.9214
32736/60000 [===============>..............] - ETA: 46s - loss: 0.2554 - categorical_accuracy: 0.9214
32768/60000 [===============>..............] - ETA: 46s - loss: 0.2551 - categorical_accuracy: 0.9215
32800/60000 [===============>..............] - ETA: 46s - loss: 0.2550 - categorical_accuracy: 0.9215
32832/60000 [===============>..............] - ETA: 45s - loss: 0.2548 - categorical_accuracy: 0.9216
32864/60000 [===============>..............] - ETA: 45s - loss: 0.2547 - categorical_accuracy: 0.9216
32896/60000 [===============>..............] - ETA: 45s - loss: 0.2545 - categorical_accuracy: 0.9217
32928/60000 [===============>..............] - ETA: 45s - loss: 0.2543 - categorical_accuracy: 0.9217
32960/60000 [===============>..............] - ETA: 45s - loss: 0.2542 - categorical_accuracy: 0.9218
32992/60000 [===============>..............] - ETA: 45s - loss: 0.2540 - categorical_accuracy: 0.9218
33024/60000 [===============>..............] - ETA: 45s - loss: 0.2538 - categorical_accuracy: 0.9219
33056/60000 [===============>..............] - ETA: 45s - loss: 0.2537 - categorical_accuracy: 0.9220
33088/60000 [===============>..............] - ETA: 45s - loss: 0.2537 - categorical_accuracy: 0.9220
33120/60000 [===============>..............] - ETA: 45s - loss: 0.2537 - categorical_accuracy: 0.9220
33152/60000 [===============>..............] - ETA: 45s - loss: 0.2535 - categorical_accuracy: 0.9221
33184/60000 [===============>..............] - ETA: 45s - loss: 0.2534 - categorical_accuracy: 0.9221
33216/60000 [===============>..............] - ETA: 45s - loss: 0.2535 - categorical_accuracy: 0.9221
33248/60000 [===============>..............] - ETA: 45s - loss: 0.2533 - categorical_accuracy: 0.9222
33280/60000 [===============>..............] - ETA: 45s - loss: 0.2531 - categorical_accuracy: 0.9222
33312/60000 [===============>..............] - ETA: 45s - loss: 0.2530 - categorical_accuracy: 0.9223
33344/60000 [===============>..............] - ETA: 45s - loss: 0.2529 - categorical_accuracy: 0.9223
33376/60000 [===============>..............] - ETA: 45s - loss: 0.2528 - categorical_accuracy: 0.9223
33408/60000 [===============>..............] - ETA: 45s - loss: 0.2526 - categorical_accuracy: 0.9223
33440/60000 [===============>..............] - ETA: 44s - loss: 0.2525 - categorical_accuracy: 0.9224
33472/60000 [===============>..............] - ETA: 44s - loss: 0.2525 - categorical_accuracy: 0.9224
33504/60000 [===============>..............] - ETA: 44s - loss: 0.2523 - categorical_accuracy: 0.9225
33536/60000 [===============>..............] - ETA: 44s - loss: 0.2522 - categorical_accuracy: 0.9225
33568/60000 [===============>..............] - ETA: 44s - loss: 0.2521 - categorical_accuracy: 0.9226
33600/60000 [===============>..............] - ETA: 44s - loss: 0.2520 - categorical_accuracy: 0.9226
33632/60000 [===============>..............] - ETA: 44s - loss: 0.2519 - categorical_accuracy: 0.9226
33664/60000 [===============>..............] - ETA: 44s - loss: 0.2517 - categorical_accuracy: 0.9227
33696/60000 [===============>..............] - ETA: 44s - loss: 0.2515 - categorical_accuracy: 0.9227
33728/60000 [===============>..............] - ETA: 44s - loss: 0.2515 - categorical_accuracy: 0.9228
33760/60000 [===============>..............] - ETA: 44s - loss: 0.2513 - categorical_accuracy: 0.9228
33792/60000 [===============>..............] - ETA: 44s - loss: 0.2511 - categorical_accuracy: 0.9229
33824/60000 [===============>..............] - ETA: 44s - loss: 0.2511 - categorical_accuracy: 0.9229
33856/60000 [===============>..............] - ETA: 44s - loss: 0.2511 - categorical_accuracy: 0.9229
33888/60000 [===============>..............] - ETA: 44s - loss: 0.2510 - categorical_accuracy: 0.9229
33920/60000 [===============>..............] - ETA: 44s - loss: 0.2510 - categorical_accuracy: 0.9229
33952/60000 [===============>..............] - ETA: 44s - loss: 0.2508 - categorical_accuracy: 0.9230
33984/60000 [===============>..............] - ETA: 44s - loss: 0.2508 - categorical_accuracy: 0.9230
34016/60000 [================>.............] - ETA: 44s - loss: 0.2508 - categorical_accuracy: 0.9230
34048/60000 [================>.............] - ETA: 43s - loss: 0.2506 - categorical_accuracy: 0.9231
34080/60000 [================>.............] - ETA: 43s - loss: 0.2504 - categorical_accuracy: 0.9232
34144/60000 [================>.............] - ETA: 43s - loss: 0.2503 - categorical_accuracy: 0.9232
34176/60000 [================>.............] - ETA: 43s - loss: 0.2502 - categorical_accuracy: 0.9232
34208/60000 [================>.............] - ETA: 43s - loss: 0.2501 - categorical_accuracy: 0.9232
34240/60000 [================>.............] - ETA: 43s - loss: 0.2500 - categorical_accuracy: 0.9232
34272/60000 [================>.............] - ETA: 43s - loss: 0.2499 - categorical_accuracy: 0.9233
34304/60000 [================>.............] - ETA: 43s - loss: 0.2498 - categorical_accuracy: 0.9233
34336/60000 [================>.............] - ETA: 43s - loss: 0.2498 - categorical_accuracy: 0.9233
34368/60000 [================>.............] - ETA: 43s - loss: 0.2497 - categorical_accuracy: 0.9233
34400/60000 [================>.............] - ETA: 43s - loss: 0.2497 - categorical_accuracy: 0.9233
34432/60000 [================>.............] - ETA: 43s - loss: 0.2495 - categorical_accuracy: 0.9234
34464/60000 [================>.............] - ETA: 43s - loss: 0.2495 - categorical_accuracy: 0.9234
34496/60000 [================>.............] - ETA: 43s - loss: 0.2494 - categorical_accuracy: 0.9234
34528/60000 [================>.............] - ETA: 43s - loss: 0.2493 - categorical_accuracy: 0.9234
34560/60000 [================>.............] - ETA: 43s - loss: 0.2492 - categorical_accuracy: 0.9235
34592/60000 [================>.............] - ETA: 43s - loss: 0.2491 - categorical_accuracy: 0.9235
34624/60000 [================>.............] - ETA: 43s - loss: 0.2491 - categorical_accuracy: 0.9236
34656/60000 [================>.............] - ETA: 42s - loss: 0.2489 - categorical_accuracy: 0.9236
34688/60000 [================>.............] - ETA: 42s - loss: 0.2489 - categorical_accuracy: 0.9236
34720/60000 [================>.............] - ETA: 42s - loss: 0.2490 - categorical_accuracy: 0.9236
34752/60000 [================>.............] - ETA: 42s - loss: 0.2488 - categorical_accuracy: 0.9236
34784/60000 [================>.............] - ETA: 42s - loss: 0.2486 - categorical_accuracy: 0.9237
34816/60000 [================>.............] - ETA: 42s - loss: 0.2486 - categorical_accuracy: 0.9237
34848/60000 [================>.............] - ETA: 42s - loss: 0.2485 - categorical_accuracy: 0.9237
34880/60000 [================>.............] - ETA: 42s - loss: 0.2484 - categorical_accuracy: 0.9237
34912/60000 [================>.............] - ETA: 42s - loss: 0.2483 - categorical_accuracy: 0.9238
34944/60000 [================>.............] - ETA: 42s - loss: 0.2482 - categorical_accuracy: 0.9238
34976/60000 [================>.............] - ETA: 42s - loss: 0.2480 - categorical_accuracy: 0.9239
35008/60000 [================>.............] - ETA: 42s - loss: 0.2478 - categorical_accuracy: 0.9239
35040/60000 [================>.............] - ETA: 42s - loss: 0.2476 - categorical_accuracy: 0.9240
35072/60000 [================>.............] - ETA: 42s - loss: 0.2474 - categorical_accuracy: 0.9240
35104/60000 [================>.............] - ETA: 42s - loss: 0.2473 - categorical_accuracy: 0.9241
35136/60000 [================>.............] - ETA: 42s - loss: 0.2471 - categorical_accuracy: 0.9242
35168/60000 [================>.............] - ETA: 42s - loss: 0.2473 - categorical_accuracy: 0.9242
35200/60000 [================>.............] - ETA: 42s - loss: 0.2472 - categorical_accuracy: 0.9242
35232/60000 [================>.............] - ETA: 42s - loss: 0.2471 - categorical_accuracy: 0.9242
35264/60000 [================>.............] - ETA: 41s - loss: 0.2469 - categorical_accuracy: 0.9243
35296/60000 [================>.............] - ETA: 41s - loss: 0.2469 - categorical_accuracy: 0.9242
35328/60000 [================>.............] - ETA: 41s - loss: 0.2468 - categorical_accuracy: 0.9243
35360/60000 [================>.............] - ETA: 41s - loss: 0.2467 - categorical_accuracy: 0.9243
35392/60000 [================>.............] - ETA: 41s - loss: 0.2466 - categorical_accuracy: 0.9243
35424/60000 [================>.............] - ETA: 41s - loss: 0.2464 - categorical_accuracy: 0.9244
35456/60000 [================>.............] - ETA: 41s - loss: 0.2463 - categorical_accuracy: 0.9244
35488/60000 [================>.............] - ETA: 41s - loss: 0.2465 - categorical_accuracy: 0.9244
35520/60000 [================>.............] - ETA: 41s - loss: 0.2464 - categorical_accuracy: 0.9244
35552/60000 [================>.............] - ETA: 41s - loss: 0.2464 - categorical_accuracy: 0.9244
35584/60000 [================>.............] - ETA: 41s - loss: 0.2463 - categorical_accuracy: 0.9244
35616/60000 [================>.............] - ETA: 41s - loss: 0.2463 - categorical_accuracy: 0.9244
35648/60000 [================>.............] - ETA: 41s - loss: 0.2462 - categorical_accuracy: 0.9244
35680/60000 [================>.............] - ETA: 41s - loss: 0.2460 - categorical_accuracy: 0.9244
35712/60000 [================>.............] - ETA: 41s - loss: 0.2459 - categorical_accuracy: 0.9245
35744/60000 [================>.............] - ETA: 41s - loss: 0.2458 - categorical_accuracy: 0.9245
35776/60000 [================>.............] - ETA: 41s - loss: 0.2456 - categorical_accuracy: 0.9246
35808/60000 [================>.............] - ETA: 41s - loss: 0.2457 - categorical_accuracy: 0.9245
35840/60000 [================>.............] - ETA: 41s - loss: 0.2455 - categorical_accuracy: 0.9246
35872/60000 [================>.............] - ETA: 40s - loss: 0.2453 - categorical_accuracy: 0.9246
35904/60000 [================>.............] - ETA: 40s - loss: 0.2451 - categorical_accuracy: 0.9247
35936/60000 [================>.............] - ETA: 40s - loss: 0.2452 - categorical_accuracy: 0.9247
35968/60000 [================>.............] - ETA: 40s - loss: 0.2450 - categorical_accuracy: 0.9247
36000/60000 [=================>............] - ETA: 40s - loss: 0.2449 - categorical_accuracy: 0.9248
36032/60000 [=================>............] - ETA: 40s - loss: 0.2449 - categorical_accuracy: 0.9248
36064/60000 [=================>............] - ETA: 40s - loss: 0.2449 - categorical_accuracy: 0.9248
36096/60000 [=================>............] - ETA: 40s - loss: 0.2448 - categorical_accuracy: 0.9249
36128/60000 [=================>............] - ETA: 40s - loss: 0.2448 - categorical_accuracy: 0.9249
36160/60000 [=================>............] - ETA: 40s - loss: 0.2447 - categorical_accuracy: 0.9249
36192/60000 [=================>............] - ETA: 40s - loss: 0.2446 - categorical_accuracy: 0.9249
36224/60000 [=================>............] - ETA: 40s - loss: 0.2445 - categorical_accuracy: 0.9249
36256/60000 [=================>............] - ETA: 40s - loss: 0.2443 - categorical_accuracy: 0.9250
36288/60000 [=================>............] - ETA: 40s - loss: 0.2443 - categorical_accuracy: 0.9250
36320/60000 [=================>............] - ETA: 40s - loss: 0.2441 - categorical_accuracy: 0.9251
36352/60000 [=================>............] - ETA: 40s - loss: 0.2439 - categorical_accuracy: 0.9252
36384/60000 [=================>............] - ETA: 40s - loss: 0.2437 - categorical_accuracy: 0.9252
36416/60000 [=================>............] - ETA: 40s - loss: 0.2436 - categorical_accuracy: 0.9253
36448/60000 [=================>............] - ETA: 39s - loss: 0.2434 - categorical_accuracy: 0.9253
36480/60000 [=================>............] - ETA: 39s - loss: 0.2433 - categorical_accuracy: 0.9254
36512/60000 [=================>............] - ETA: 39s - loss: 0.2431 - categorical_accuracy: 0.9254
36544/60000 [=================>............] - ETA: 39s - loss: 0.2431 - categorical_accuracy: 0.9254
36576/60000 [=================>............] - ETA: 39s - loss: 0.2430 - categorical_accuracy: 0.9255
36608/60000 [=================>............] - ETA: 39s - loss: 0.2430 - categorical_accuracy: 0.9255
36640/60000 [=================>............] - ETA: 39s - loss: 0.2428 - categorical_accuracy: 0.9255
36672/60000 [=================>............] - ETA: 39s - loss: 0.2428 - categorical_accuracy: 0.9255
36704/60000 [=================>............] - ETA: 39s - loss: 0.2427 - categorical_accuracy: 0.9256
36736/60000 [=================>............] - ETA: 39s - loss: 0.2426 - categorical_accuracy: 0.9256
36768/60000 [=================>............] - ETA: 39s - loss: 0.2424 - categorical_accuracy: 0.9256
36800/60000 [=================>............] - ETA: 39s - loss: 0.2422 - categorical_accuracy: 0.9257
36832/60000 [=================>............] - ETA: 39s - loss: 0.2420 - categorical_accuracy: 0.9257
36864/60000 [=================>............] - ETA: 39s - loss: 0.2419 - categorical_accuracy: 0.9258
36896/60000 [=================>............] - ETA: 39s - loss: 0.2417 - categorical_accuracy: 0.9258
36928/60000 [=================>............] - ETA: 39s - loss: 0.2417 - categorical_accuracy: 0.9258
36960/60000 [=================>............] - ETA: 39s - loss: 0.2415 - categorical_accuracy: 0.9259
36992/60000 [=================>............] - ETA: 39s - loss: 0.2413 - categorical_accuracy: 0.9260
37024/60000 [=================>............] - ETA: 39s - loss: 0.2412 - categorical_accuracy: 0.9260
37056/60000 [=================>............] - ETA: 38s - loss: 0.2410 - categorical_accuracy: 0.9261
37088/60000 [=================>............] - ETA: 38s - loss: 0.2408 - categorical_accuracy: 0.9261
37120/60000 [=================>............] - ETA: 38s - loss: 0.2407 - categorical_accuracy: 0.9261
37152/60000 [=================>............] - ETA: 38s - loss: 0.2407 - categorical_accuracy: 0.9261
37184/60000 [=================>............] - ETA: 38s - loss: 0.2405 - categorical_accuracy: 0.9262
37216/60000 [=================>............] - ETA: 38s - loss: 0.2404 - categorical_accuracy: 0.9263
37248/60000 [=================>............] - ETA: 38s - loss: 0.2403 - categorical_accuracy: 0.9263
37280/60000 [=================>............] - ETA: 38s - loss: 0.2402 - categorical_accuracy: 0.9263
37312/60000 [=================>............] - ETA: 38s - loss: 0.2400 - categorical_accuracy: 0.9264
37344/60000 [=================>............] - ETA: 38s - loss: 0.2398 - categorical_accuracy: 0.9264
37376/60000 [=================>............] - ETA: 38s - loss: 0.2397 - categorical_accuracy: 0.9265
37408/60000 [=================>............] - ETA: 38s - loss: 0.2396 - categorical_accuracy: 0.9265
37440/60000 [=================>............] - ETA: 38s - loss: 0.2394 - categorical_accuracy: 0.9266
37472/60000 [=================>............] - ETA: 38s - loss: 0.2394 - categorical_accuracy: 0.9266
37504/60000 [=================>............] - ETA: 38s - loss: 0.2392 - categorical_accuracy: 0.9266
37536/60000 [=================>............] - ETA: 38s - loss: 0.2390 - categorical_accuracy: 0.9267
37568/60000 [=================>............] - ETA: 38s - loss: 0.2389 - categorical_accuracy: 0.9267
37600/60000 [=================>............] - ETA: 38s - loss: 0.2387 - categorical_accuracy: 0.9268
37632/60000 [=================>............] - ETA: 37s - loss: 0.2386 - categorical_accuracy: 0.9268
37664/60000 [=================>............] - ETA: 37s - loss: 0.2384 - categorical_accuracy: 0.9269
37696/60000 [=================>............] - ETA: 37s - loss: 0.2382 - categorical_accuracy: 0.9269
37728/60000 [=================>............] - ETA: 37s - loss: 0.2380 - categorical_accuracy: 0.9270
37760/60000 [=================>............] - ETA: 37s - loss: 0.2379 - categorical_accuracy: 0.9270
37792/60000 [=================>............] - ETA: 37s - loss: 0.2377 - categorical_accuracy: 0.9271
37824/60000 [=================>............] - ETA: 37s - loss: 0.2378 - categorical_accuracy: 0.9270
37856/60000 [=================>............] - ETA: 37s - loss: 0.2377 - categorical_accuracy: 0.9271
37888/60000 [=================>............] - ETA: 37s - loss: 0.2378 - categorical_accuracy: 0.9271
37920/60000 [=================>............] - ETA: 37s - loss: 0.2377 - categorical_accuracy: 0.9271
37952/60000 [=================>............] - ETA: 37s - loss: 0.2376 - categorical_accuracy: 0.9272
37984/60000 [=================>............] - ETA: 37s - loss: 0.2374 - categorical_accuracy: 0.9272
38016/60000 [==================>...........] - ETA: 37s - loss: 0.2372 - categorical_accuracy: 0.9273
38048/60000 [==================>...........] - ETA: 37s - loss: 0.2372 - categorical_accuracy: 0.9273
38080/60000 [==================>...........] - ETA: 37s - loss: 0.2370 - categorical_accuracy: 0.9273
38112/60000 [==================>...........] - ETA: 37s - loss: 0.2371 - categorical_accuracy: 0.9273
38144/60000 [==================>...........] - ETA: 37s - loss: 0.2373 - categorical_accuracy: 0.9273
38176/60000 [==================>...........] - ETA: 37s - loss: 0.2372 - categorical_accuracy: 0.9273
38208/60000 [==================>...........] - ETA: 37s - loss: 0.2371 - categorical_accuracy: 0.9273
38240/60000 [==================>...........] - ETA: 36s - loss: 0.2371 - categorical_accuracy: 0.9273
38272/60000 [==================>...........] - ETA: 36s - loss: 0.2370 - categorical_accuracy: 0.9273
38304/60000 [==================>...........] - ETA: 36s - loss: 0.2368 - categorical_accuracy: 0.9274
38336/60000 [==================>...........] - ETA: 36s - loss: 0.2367 - categorical_accuracy: 0.9274
38368/60000 [==================>...........] - ETA: 36s - loss: 0.2365 - categorical_accuracy: 0.9275
38400/60000 [==================>...........] - ETA: 36s - loss: 0.2364 - categorical_accuracy: 0.9275
38432/60000 [==================>...........] - ETA: 36s - loss: 0.2363 - categorical_accuracy: 0.9275
38464/60000 [==================>...........] - ETA: 36s - loss: 0.2361 - categorical_accuracy: 0.9275
38496/60000 [==================>...........] - ETA: 36s - loss: 0.2360 - categorical_accuracy: 0.9276
38528/60000 [==================>...........] - ETA: 36s - loss: 0.2358 - categorical_accuracy: 0.9276
38560/60000 [==================>...........] - ETA: 36s - loss: 0.2357 - categorical_accuracy: 0.9277
38592/60000 [==================>...........] - ETA: 36s - loss: 0.2355 - categorical_accuracy: 0.9277
38624/60000 [==================>...........] - ETA: 36s - loss: 0.2357 - categorical_accuracy: 0.9277
38656/60000 [==================>...........] - ETA: 36s - loss: 0.2356 - categorical_accuracy: 0.9277
38688/60000 [==================>...........] - ETA: 36s - loss: 0.2354 - categorical_accuracy: 0.9278
38720/60000 [==================>...........] - ETA: 36s - loss: 0.2352 - categorical_accuracy: 0.9278
38752/60000 [==================>...........] - ETA: 36s - loss: 0.2351 - categorical_accuracy: 0.9278
38784/60000 [==================>...........] - ETA: 36s - loss: 0.2350 - categorical_accuracy: 0.9279
38816/60000 [==================>...........] - ETA: 35s - loss: 0.2349 - categorical_accuracy: 0.9279
38848/60000 [==================>...........] - ETA: 35s - loss: 0.2347 - categorical_accuracy: 0.9279
38880/60000 [==================>...........] - ETA: 35s - loss: 0.2346 - categorical_accuracy: 0.9280
38912/60000 [==================>...........] - ETA: 35s - loss: 0.2345 - categorical_accuracy: 0.9280
38944/60000 [==================>...........] - ETA: 35s - loss: 0.2343 - categorical_accuracy: 0.9281
38976/60000 [==================>...........] - ETA: 35s - loss: 0.2341 - categorical_accuracy: 0.9281
39008/60000 [==================>...........] - ETA: 35s - loss: 0.2340 - categorical_accuracy: 0.9281
39040/60000 [==================>...........] - ETA: 35s - loss: 0.2338 - categorical_accuracy: 0.9282
39072/60000 [==================>...........] - ETA: 35s - loss: 0.2338 - categorical_accuracy: 0.9282
39104/60000 [==================>...........] - ETA: 35s - loss: 0.2336 - categorical_accuracy: 0.9282
39136/60000 [==================>...........] - ETA: 35s - loss: 0.2335 - categorical_accuracy: 0.9283
39168/60000 [==================>...........] - ETA: 35s - loss: 0.2334 - categorical_accuracy: 0.9283
39200/60000 [==================>...........] - ETA: 35s - loss: 0.2332 - categorical_accuracy: 0.9283
39232/60000 [==================>...........] - ETA: 35s - loss: 0.2331 - categorical_accuracy: 0.9284
39264/60000 [==================>...........] - ETA: 35s - loss: 0.2330 - categorical_accuracy: 0.9284
39296/60000 [==================>...........] - ETA: 35s - loss: 0.2328 - categorical_accuracy: 0.9285
39328/60000 [==================>...........] - ETA: 35s - loss: 0.2326 - categorical_accuracy: 0.9285
39360/60000 [==================>...........] - ETA: 35s - loss: 0.2325 - categorical_accuracy: 0.9286
39392/60000 [==================>...........] - ETA: 34s - loss: 0.2327 - categorical_accuracy: 0.9285
39424/60000 [==================>...........] - ETA: 34s - loss: 0.2327 - categorical_accuracy: 0.9286
39456/60000 [==================>...........] - ETA: 34s - loss: 0.2325 - categorical_accuracy: 0.9286
39488/60000 [==================>...........] - ETA: 34s - loss: 0.2324 - categorical_accuracy: 0.9287
39520/60000 [==================>...........] - ETA: 34s - loss: 0.2324 - categorical_accuracy: 0.9287
39552/60000 [==================>...........] - ETA: 34s - loss: 0.2322 - categorical_accuracy: 0.9287
39584/60000 [==================>...........] - ETA: 34s - loss: 0.2321 - categorical_accuracy: 0.9288
39616/60000 [==================>...........] - ETA: 34s - loss: 0.2323 - categorical_accuracy: 0.9288
39648/60000 [==================>...........] - ETA: 34s - loss: 0.2322 - categorical_accuracy: 0.9288
39680/60000 [==================>...........] - ETA: 34s - loss: 0.2321 - categorical_accuracy: 0.9288
39712/60000 [==================>...........] - ETA: 34s - loss: 0.2322 - categorical_accuracy: 0.9288
39744/60000 [==================>...........] - ETA: 34s - loss: 0.2322 - categorical_accuracy: 0.9288
39776/60000 [==================>...........] - ETA: 34s - loss: 0.2322 - categorical_accuracy: 0.9288
39808/60000 [==================>...........] - ETA: 34s - loss: 0.2321 - categorical_accuracy: 0.9288
39840/60000 [==================>...........] - ETA: 34s - loss: 0.2319 - categorical_accuracy: 0.9289
39872/60000 [==================>...........] - ETA: 34s - loss: 0.2319 - categorical_accuracy: 0.9289
39904/60000 [==================>...........] - ETA: 34s - loss: 0.2317 - categorical_accuracy: 0.9290
39936/60000 [==================>...........] - ETA: 34s - loss: 0.2315 - categorical_accuracy: 0.9290
39968/60000 [==================>...........] - ETA: 33s - loss: 0.2315 - categorical_accuracy: 0.9290
40000/60000 [===================>..........] - ETA: 33s - loss: 0.2314 - categorical_accuracy: 0.9291
40032/60000 [===================>..........] - ETA: 33s - loss: 0.2312 - categorical_accuracy: 0.9291
40064/60000 [===================>..........] - ETA: 33s - loss: 0.2311 - categorical_accuracy: 0.9291
40096/60000 [===================>..........] - ETA: 33s - loss: 0.2310 - categorical_accuracy: 0.9292
40128/60000 [===================>..........] - ETA: 33s - loss: 0.2308 - categorical_accuracy: 0.9292
40160/60000 [===================>..........] - ETA: 33s - loss: 0.2308 - categorical_accuracy: 0.9292
40192/60000 [===================>..........] - ETA: 33s - loss: 0.2307 - categorical_accuracy: 0.9293
40224/60000 [===================>..........] - ETA: 33s - loss: 0.2306 - categorical_accuracy: 0.9293
40256/60000 [===================>..........] - ETA: 33s - loss: 0.2307 - categorical_accuracy: 0.9292
40288/60000 [===================>..........] - ETA: 33s - loss: 0.2306 - categorical_accuracy: 0.9293
40320/60000 [===================>..........] - ETA: 33s - loss: 0.2304 - categorical_accuracy: 0.9293
40352/60000 [===================>..........] - ETA: 33s - loss: 0.2303 - categorical_accuracy: 0.9293
40384/60000 [===================>..........] - ETA: 33s - loss: 0.2302 - categorical_accuracy: 0.9294
40416/60000 [===================>..........] - ETA: 33s - loss: 0.2300 - categorical_accuracy: 0.9294
40448/60000 [===================>..........] - ETA: 33s - loss: 0.2299 - categorical_accuracy: 0.9295
40480/60000 [===================>..........] - ETA: 33s - loss: 0.2297 - categorical_accuracy: 0.9295
40512/60000 [===================>..........] - ETA: 33s - loss: 0.2296 - categorical_accuracy: 0.9296
40576/60000 [===================>..........] - ETA: 32s - loss: 0.2293 - categorical_accuracy: 0.9296
40608/60000 [===================>..........] - ETA: 32s - loss: 0.2293 - categorical_accuracy: 0.9296
40640/60000 [===================>..........] - ETA: 32s - loss: 0.2293 - categorical_accuracy: 0.9297
40672/60000 [===================>..........] - ETA: 32s - loss: 0.2292 - categorical_accuracy: 0.9297
40704/60000 [===================>..........] - ETA: 32s - loss: 0.2290 - categorical_accuracy: 0.9297
40736/60000 [===================>..........] - ETA: 32s - loss: 0.2289 - categorical_accuracy: 0.9298
40768/60000 [===================>..........] - ETA: 32s - loss: 0.2288 - categorical_accuracy: 0.9298
40800/60000 [===================>..........] - ETA: 32s - loss: 0.2286 - categorical_accuracy: 0.9299
40832/60000 [===================>..........] - ETA: 32s - loss: 0.2284 - categorical_accuracy: 0.9299
40864/60000 [===================>..........] - ETA: 32s - loss: 0.2283 - categorical_accuracy: 0.9300
40896/60000 [===================>..........] - ETA: 32s - loss: 0.2281 - categorical_accuracy: 0.9300
40928/60000 [===================>..........] - ETA: 32s - loss: 0.2280 - categorical_accuracy: 0.9301
40960/60000 [===================>..........] - ETA: 32s - loss: 0.2279 - categorical_accuracy: 0.9301
40992/60000 [===================>..........] - ETA: 32s - loss: 0.2278 - categorical_accuracy: 0.9301
41024/60000 [===================>..........] - ETA: 32s - loss: 0.2277 - categorical_accuracy: 0.9301
41056/60000 [===================>..........] - ETA: 32s - loss: 0.2276 - categorical_accuracy: 0.9301
41088/60000 [===================>..........] - ETA: 32s - loss: 0.2274 - categorical_accuracy: 0.9302
41120/60000 [===================>..........] - ETA: 32s - loss: 0.2272 - categorical_accuracy: 0.9303
41152/60000 [===================>..........] - ETA: 31s - loss: 0.2272 - categorical_accuracy: 0.9303
41184/60000 [===================>..........] - ETA: 31s - loss: 0.2271 - categorical_accuracy: 0.9303
41216/60000 [===================>..........] - ETA: 31s - loss: 0.2270 - categorical_accuracy: 0.9304
41248/60000 [===================>..........] - ETA: 31s - loss: 0.2269 - categorical_accuracy: 0.9304
41280/60000 [===================>..........] - ETA: 31s - loss: 0.2268 - categorical_accuracy: 0.9304
41312/60000 [===================>..........] - ETA: 31s - loss: 0.2266 - categorical_accuracy: 0.9305
41344/60000 [===================>..........] - ETA: 31s - loss: 0.2265 - categorical_accuracy: 0.9305
41376/60000 [===================>..........] - ETA: 31s - loss: 0.2267 - categorical_accuracy: 0.9305
41408/60000 [===================>..........] - ETA: 31s - loss: 0.2266 - categorical_accuracy: 0.9305
41472/60000 [===================>..........] - ETA: 31s - loss: 0.2266 - categorical_accuracy: 0.9306
41504/60000 [===================>..........] - ETA: 31s - loss: 0.2265 - categorical_accuracy: 0.9306
41536/60000 [===================>..........] - ETA: 31s - loss: 0.2264 - categorical_accuracy: 0.9306
41568/60000 [===================>..........] - ETA: 31s - loss: 0.2264 - categorical_accuracy: 0.9306
41600/60000 [===================>..........] - ETA: 31s - loss: 0.2263 - categorical_accuracy: 0.9306
41632/60000 [===================>..........] - ETA: 31s - loss: 0.2264 - categorical_accuracy: 0.9306
41664/60000 [===================>..........] - ETA: 31s - loss: 0.2264 - categorical_accuracy: 0.9307
41696/60000 [===================>..........] - ETA: 31s - loss: 0.2262 - categorical_accuracy: 0.9307
41760/60000 [===================>..........] - ETA: 30s - loss: 0.2259 - categorical_accuracy: 0.9308
41792/60000 [===================>..........] - ETA: 30s - loss: 0.2260 - categorical_accuracy: 0.9308
41824/60000 [===================>..........] - ETA: 30s - loss: 0.2259 - categorical_accuracy: 0.9308
41856/60000 [===================>..........] - ETA: 30s - loss: 0.2258 - categorical_accuracy: 0.9308
41888/60000 [===================>..........] - ETA: 30s - loss: 0.2257 - categorical_accuracy: 0.9308
41920/60000 [===================>..........] - ETA: 30s - loss: 0.2257 - categorical_accuracy: 0.9308
41952/60000 [===================>..........] - ETA: 30s - loss: 0.2255 - categorical_accuracy: 0.9309
41984/60000 [===================>..........] - ETA: 30s - loss: 0.2256 - categorical_accuracy: 0.9309
42016/60000 [====================>.........] - ETA: 30s - loss: 0.2254 - categorical_accuracy: 0.9309
42048/60000 [====================>.........] - ETA: 30s - loss: 0.2253 - categorical_accuracy: 0.9310
42080/60000 [====================>.........] - ETA: 30s - loss: 0.2252 - categorical_accuracy: 0.9310
42112/60000 [====================>.........] - ETA: 30s - loss: 0.2251 - categorical_accuracy: 0.9311
42176/60000 [====================>.........] - ETA: 30s - loss: 0.2248 - categorical_accuracy: 0.9312
42208/60000 [====================>.........] - ETA: 30s - loss: 0.2246 - categorical_accuracy: 0.9312
42240/60000 [====================>.........] - ETA: 30s - loss: 0.2246 - categorical_accuracy: 0.9312
42272/60000 [====================>.........] - ETA: 30s - loss: 0.2245 - categorical_accuracy: 0.9313
42304/60000 [====================>.........] - ETA: 30s - loss: 0.2243 - categorical_accuracy: 0.9313
42368/60000 [====================>.........] - ETA: 29s - loss: 0.2241 - categorical_accuracy: 0.9314
42400/60000 [====================>.........] - ETA: 29s - loss: 0.2240 - categorical_accuracy: 0.9314
42464/60000 [====================>.........] - ETA: 29s - loss: 0.2237 - categorical_accuracy: 0.9315
42496/60000 [====================>.........] - ETA: 29s - loss: 0.2238 - categorical_accuracy: 0.9315
42528/60000 [====================>.........] - ETA: 29s - loss: 0.2238 - categorical_accuracy: 0.9315
42560/60000 [====================>.........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9316
42592/60000 [====================>.........] - ETA: 29s - loss: 0.2235 - categorical_accuracy: 0.9316
42624/60000 [====================>.........] - ETA: 29s - loss: 0.2233 - categorical_accuracy: 0.9317
42656/60000 [====================>.........] - ETA: 29s - loss: 0.2232 - categorical_accuracy: 0.9317
42688/60000 [====================>.........] - ETA: 29s - loss: 0.2231 - categorical_accuracy: 0.9317
42720/60000 [====================>.........] - ETA: 29s - loss: 0.2231 - categorical_accuracy: 0.9318
42752/60000 [====================>.........] - ETA: 29s - loss: 0.2230 - categorical_accuracy: 0.9318
42784/60000 [====================>.........] - ETA: 29s - loss: 0.2228 - categorical_accuracy: 0.9318
42816/60000 [====================>.........] - ETA: 29s - loss: 0.2227 - categorical_accuracy: 0.9318
42848/60000 [====================>.........] - ETA: 29s - loss: 0.2227 - categorical_accuracy: 0.9319
42880/60000 [====================>.........] - ETA: 29s - loss: 0.2226 - categorical_accuracy: 0.9319
42944/60000 [====================>.........] - ETA: 28s - loss: 0.2225 - categorical_accuracy: 0.9319
42976/60000 [====================>.........] - ETA: 28s - loss: 0.2225 - categorical_accuracy: 0.9319
43008/60000 [====================>.........] - ETA: 28s - loss: 0.2223 - categorical_accuracy: 0.9319
43040/60000 [====================>.........] - ETA: 28s - loss: 0.2222 - categorical_accuracy: 0.9320
43072/60000 [====================>.........] - ETA: 28s - loss: 0.2221 - categorical_accuracy: 0.9320
43104/60000 [====================>.........] - ETA: 28s - loss: 0.2219 - categorical_accuracy: 0.9320
43136/60000 [====================>.........] - ETA: 28s - loss: 0.2218 - categorical_accuracy: 0.9321
43168/60000 [====================>.........] - ETA: 28s - loss: 0.2216 - categorical_accuracy: 0.9321
43200/60000 [====================>.........] - ETA: 28s - loss: 0.2215 - categorical_accuracy: 0.9322
43264/60000 [====================>.........] - ETA: 28s - loss: 0.2213 - categorical_accuracy: 0.9323
43296/60000 [====================>.........] - ETA: 28s - loss: 0.2211 - categorical_accuracy: 0.9323
43328/60000 [====================>.........] - ETA: 28s - loss: 0.2213 - categorical_accuracy: 0.9323
43392/60000 [====================>.........] - ETA: 28s - loss: 0.2212 - categorical_accuracy: 0.9323
43424/60000 [====================>.........] - ETA: 28s - loss: 0.2211 - categorical_accuracy: 0.9323
43456/60000 [====================>.........] - ETA: 28s - loss: 0.2210 - categorical_accuracy: 0.9324
43488/60000 [====================>.........] - ETA: 27s - loss: 0.2208 - categorical_accuracy: 0.9324
43520/60000 [====================>.........] - ETA: 27s - loss: 0.2207 - categorical_accuracy: 0.9324
43552/60000 [====================>.........] - ETA: 27s - loss: 0.2206 - categorical_accuracy: 0.9324
43584/60000 [====================>.........] - ETA: 27s - loss: 0.2205 - categorical_accuracy: 0.9325
43616/60000 [====================>.........] - ETA: 27s - loss: 0.2205 - categorical_accuracy: 0.9325
43648/60000 [====================>.........] - ETA: 27s - loss: 0.2203 - categorical_accuracy: 0.9326
43680/60000 [====================>.........] - ETA: 27s - loss: 0.2202 - categorical_accuracy: 0.9326
43712/60000 [====================>.........] - ETA: 27s - loss: 0.2201 - categorical_accuracy: 0.9326
43744/60000 [====================>.........] - ETA: 27s - loss: 0.2200 - categorical_accuracy: 0.9327
43776/60000 [====================>.........] - ETA: 27s - loss: 0.2199 - categorical_accuracy: 0.9327
43808/60000 [====================>.........] - ETA: 27s - loss: 0.2199 - categorical_accuracy: 0.9327
43840/60000 [====================>.........] - ETA: 27s - loss: 0.2197 - categorical_accuracy: 0.9328
43872/60000 [====================>.........] - ETA: 27s - loss: 0.2198 - categorical_accuracy: 0.9328
43904/60000 [====================>.........] - ETA: 27s - loss: 0.2197 - categorical_accuracy: 0.9328
43936/60000 [====================>.........] - ETA: 27s - loss: 0.2196 - categorical_accuracy: 0.9328
43968/60000 [====================>.........] - ETA: 27s - loss: 0.2194 - categorical_accuracy: 0.9329
44032/60000 [=====================>........] - ETA: 27s - loss: 0.2191 - categorical_accuracy: 0.9329
44064/60000 [=====================>........] - ETA: 27s - loss: 0.2190 - categorical_accuracy: 0.9330
44128/60000 [=====================>........] - ETA: 26s - loss: 0.2189 - categorical_accuracy: 0.9330
44160/60000 [=====================>........] - ETA: 26s - loss: 0.2189 - categorical_accuracy: 0.9330
44192/60000 [=====================>........] - ETA: 26s - loss: 0.2188 - categorical_accuracy: 0.9330
44224/60000 [=====================>........] - ETA: 26s - loss: 0.2187 - categorical_accuracy: 0.9331
44256/60000 [=====================>........] - ETA: 26s - loss: 0.2186 - categorical_accuracy: 0.9331
44288/60000 [=====================>........] - ETA: 26s - loss: 0.2185 - categorical_accuracy: 0.9331
44320/60000 [=====================>........] - ETA: 26s - loss: 0.2184 - categorical_accuracy: 0.9331
44352/60000 [=====================>........] - ETA: 26s - loss: 0.2183 - categorical_accuracy: 0.9332
44384/60000 [=====================>........] - ETA: 26s - loss: 0.2181 - categorical_accuracy: 0.9332
44416/60000 [=====================>........] - ETA: 26s - loss: 0.2180 - categorical_accuracy: 0.9333
44448/60000 [=====================>........] - ETA: 26s - loss: 0.2179 - categorical_accuracy: 0.9333
44512/60000 [=====================>........] - ETA: 26s - loss: 0.2176 - categorical_accuracy: 0.9334
44576/60000 [=====================>........] - ETA: 26s - loss: 0.2174 - categorical_accuracy: 0.9335
44608/60000 [=====================>........] - ETA: 26s - loss: 0.2172 - categorical_accuracy: 0.9335
44640/60000 [=====================>........] - ETA: 26s - loss: 0.2171 - categorical_accuracy: 0.9336
44672/60000 [=====================>........] - ETA: 25s - loss: 0.2171 - categorical_accuracy: 0.9336
44704/60000 [=====================>........] - ETA: 25s - loss: 0.2170 - categorical_accuracy: 0.9336
44736/60000 [=====================>........] - ETA: 25s - loss: 0.2168 - categorical_accuracy: 0.9337
44768/60000 [=====================>........] - ETA: 25s - loss: 0.2172 - categorical_accuracy: 0.9337
44800/60000 [=====================>........] - ETA: 25s - loss: 0.2170 - categorical_accuracy: 0.9337
44832/60000 [=====================>........] - ETA: 25s - loss: 0.2169 - categorical_accuracy: 0.9338
44864/60000 [=====================>........] - ETA: 25s - loss: 0.2167 - categorical_accuracy: 0.9338
44896/60000 [=====================>........] - ETA: 25s - loss: 0.2167 - categorical_accuracy: 0.9338
44928/60000 [=====================>........] - ETA: 25s - loss: 0.2167 - categorical_accuracy: 0.9338
44960/60000 [=====================>........] - ETA: 25s - loss: 0.2166 - categorical_accuracy: 0.9339
44992/60000 [=====================>........] - ETA: 25s - loss: 0.2164 - categorical_accuracy: 0.9339
45024/60000 [=====================>........] - ETA: 25s - loss: 0.2163 - categorical_accuracy: 0.9339
45088/60000 [=====================>........] - ETA: 25s - loss: 0.2161 - categorical_accuracy: 0.9340
45120/60000 [=====================>........] - ETA: 25s - loss: 0.2159 - categorical_accuracy: 0.9340
45152/60000 [=====================>........] - ETA: 25s - loss: 0.2159 - categorical_accuracy: 0.9340
45184/60000 [=====================>........] - ETA: 25s - loss: 0.2159 - categorical_accuracy: 0.9340
45216/60000 [=====================>........] - ETA: 25s - loss: 0.2158 - categorical_accuracy: 0.9341
45248/60000 [=====================>........] - ETA: 24s - loss: 0.2158 - categorical_accuracy: 0.9341
45280/60000 [=====================>........] - ETA: 24s - loss: 0.2156 - categorical_accuracy: 0.9341
45312/60000 [=====================>........] - ETA: 24s - loss: 0.2155 - categorical_accuracy: 0.9341
45376/60000 [=====================>........] - ETA: 24s - loss: 0.2154 - categorical_accuracy: 0.9342
45408/60000 [=====================>........] - ETA: 24s - loss: 0.2154 - categorical_accuracy: 0.9342
45440/60000 [=====================>........] - ETA: 24s - loss: 0.2153 - categorical_accuracy: 0.9342
45472/60000 [=====================>........] - ETA: 24s - loss: 0.2152 - categorical_accuracy: 0.9342
45504/60000 [=====================>........] - ETA: 24s - loss: 0.2152 - categorical_accuracy: 0.9342
45536/60000 [=====================>........] - ETA: 24s - loss: 0.2152 - categorical_accuracy: 0.9343
45568/60000 [=====================>........] - ETA: 24s - loss: 0.2150 - categorical_accuracy: 0.9343
45600/60000 [=====================>........] - ETA: 24s - loss: 0.2150 - categorical_accuracy: 0.9343
45632/60000 [=====================>........] - ETA: 24s - loss: 0.2150 - categorical_accuracy: 0.9343
45664/60000 [=====================>........] - ETA: 24s - loss: 0.2149 - categorical_accuracy: 0.9344
45696/60000 [=====================>........] - ETA: 24s - loss: 0.2148 - categorical_accuracy: 0.9344
45728/60000 [=====================>........] - ETA: 24s - loss: 0.2147 - categorical_accuracy: 0.9344
45760/60000 [=====================>........] - ETA: 24s - loss: 0.2145 - categorical_accuracy: 0.9345
45792/60000 [=====================>........] - ETA: 24s - loss: 0.2144 - categorical_accuracy: 0.9345
45824/60000 [=====================>........] - ETA: 24s - loss: 0.2143 - categorical_accuracy: 0.9346
45856/60000 [=====================>........] - ETA: 23s - loss: 0.2142 - categorical_accuracy: 0.9346
45888/60000 [=====================>........] - ETA: 23s - loss: 0.2142 - categorical_accuracy: 0.9346
45920/60000 [=====================>........] - ETA: 23s - loss: 0.2140 - categorical_accuracy: 0.9346
45984/60000 [=====================>........] - ETA: 23s - loss: 0.2139 - categorical_accuracy: 0.9347
46016/60000 [======================>.......] - ETA: 23s - loss: 0.2139 - categorical_accuracy: 0.9347
46048/60000 [======================>.......] - ETA: 23s - loss: 0.2139 - categorical_accuracy: 0.9347
46080/60000 [======================>.......] - ETA: 23s - loss: 0.2137 - categorical_accuracy: 0.9347
46144/60000 [======================>.......] - ETA: 23s - loss: 0.2136 - categorical_accuracy: 0.9347
46176/60000 [======================>.......] - ETA: 23s - loss: 0.2135 - categorical_accuracy: 0.9348
46208/60000 [======================>.......] - ETA: 23s - loss: 0.2134 - categorical_accuracy: 0.9348
46240/60000 [======================>.......] - ETA: 23s - loss: 0.2132 - categorical_accuracy: 0.9349
46272/60000 [======================>.......] - ETA: 23s - loss: 0.2131 - categorical_accuracy: 0.9349
46304/60000 [======================>.......] - ETA: 23s - loss: 0.2131 - categorical_accuracy: 0.9349
46336/60000 [======================>.......] - ETA: 23s - loss: 0.2130 - categorical_accuracy: 0.9349
46368/60000 [======================>.......] - ETA: 23s - loss: 0.2129 - categorical_accuracy: 0.9350
46400/60000 [======================>.......] - ETA: 23s - loss: 0.2128 - categorical_accuracy: 0.9350
46432/60000 [======================>.......] - ETA: 22s - loss: 0.2127 - categorical_accuracy: 0.9350
46464/60000 [======================>.......] - ETA: 22s - loss: 0.2126 - categorical_accuracy: 0.9350
46496/60000 [======================>.......] - ETA: 22s - loss: 0.2125 - categorical_accuracy: 0.9351
46528/60000 [======================>.......] - ETA: 22s - loss: 0.2124 - categorical_accuracy: 0.9351
46560/60000 [======================>.......] - ETA: 22s - loss: 0.2123 - categorical_accuracy: 0.9351
46592/60000 [======================>.......] - ETA: 22s - loss: 0.2122 - categorical_accuracy: 0.9352
46624/60000 [======================>.......] - ETA: 22s - loss: 0.2120 - categorical_accuracy: 0.9352
46656/60000 [======================>.......] - ETA: 22s - loss: 0.2119 - categorical_accuracy: 0.9353
46688/60000 [======================>.......] - ETA: 22s - loss: 0.2118 - categorical_accuracy: 0.9353
46720/60000 [======================>.......] - ETA: 22s - loss: 0.2117 - categorical_accuracy: 0.9353
46752/60000 [======================>.......] - ETA: 22s - loss: 0.2115 - categorical_accuracy: 0.9354
46784/60000 [======================>.......] - ETA: 22s - loss: 0.2115 - categorical_accuracy: 0.9354
46816/60000 [======================>.......] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9354
46848/60000 [======================>.......] - ETA: 22s - loss: 0.2112 - categorical_accuracy: 0.9355
46880/60000 [======================>.......] - ETA: 22s - loss: 0.2112 - categorical_accuracy: 0.9355
46912/60000 [======================>.......] - ETA: 22s - loss: 0.2112 - categorical_accuracy: 0.9355
46944/60000 [======================>.......] - ETA: 22s - loss: 0.2110 - categorical_accuracy: 0.9356
46976/60000 [======================>.......] - ETA: 22s - loss: 0.2110 - categorical_accuracy: 0.9356
47008/60000 [======================>.......] - ETA: 21s - loss: 0.2109 - categorical_accuracy: 0.9356
47040/60000 [======================>.......] - ETA: 21s - loss: 0.2108 - categorical_accuracy: 0.9356
47072/60000 [======================>.......] - ETA: 21s - loss: 0.2107 - categorical_accuracy: 0.9357
47104/60000 [======================>.......] - ETA: 21s - loss: 0.2107 - categorical_accuracy: 0.9356
47136/60000 [======================>.......] - ETA: 21s - loss: 0.2107 - categorical_accuracy: 0.9356
47168/60000 [======================>.......] - ETA: 21s - loss: 0.2106 - categorical_accuracy: 0.9357
47200/60000 [======================>.......] - ETA: 21s - loss: 0.2108 - categorical_accuracy: 0.9356
47232/60000 [======================>.......] - ETA: 21s - loss: 0.2107 - categorical_accuracy: 0.9357
47264/60000 [======================>.......] - ETA: 21s - loss: 0.2106 - categorical_accuracy: 0.9357
47296/60000 [======================>.......] - ETA: 21s - loss: 0.2105 - categorical_accuracy: 0.9357
47328/60000 [======================>.......] - ETA: 21s - loss: 0.2106 - categorical_accuracy: 0.9357
47360/60000 [======================>.......] - ETA: 21s - loss: 0.2105 - categorical_accuracy: 0.9357
47392/60000 [======================>.......] - ETA: 21s - loss: 0.2104 - categorical_accuracy: 0.9358
47424/60000 [======================>.......] - ETA: 21s - loss: 0.2104 - categorical_accuracy: 0.9358
47456/60000 [======================>.......] - ETA: 21s - loss: 0.2102 - categorical_accuracy: 0.9358
47488/60000 [======================>.......] - ETA: 21s - loss: 0.2102 - categorical_accuracy: 0.9359
47520/60000 [======================>.......] - ETA: 21s - loss: 0.2101 - categorical_accuracy: 0.9359
47552/60000 [======================>.......] - ETA: 21s - loss: 0.2100 - categorical_accuracy: 0.9359
47584/60000 [======================>.......] - ETA: 21s - loss: 0.2099 - categorical_accuracy: 0.9359
47616/60000 [======================>.......] - ETA: 20s - loss: 0.2098 - categorical_accuracy: 0.9360
47648/60000 [======================>.......] - ETA: 20s - loss: 0.2098 - categorical_accuracy: 0.9360
47680/60000 [======================>.......] - ETA: 20s - loss: 0.2099 - categorical_accuracy: 0.9360
47712/60000 [======================>.......] - ETA: 20s - loss: 0.2098 - categorical_accuracy: 0.9360
47744/60000 [======================>.......] - ETA: 20s - loss: 0.2096 - categorical_accuracy: 0.9360
47776/60000 [======================>.......] - ETA: 20s - loss: 0.2095 - categorical_accuracy: 0.9361
47808/60000 [======================>.......] - ETA: 20s - loss: 0.2095 - categorical_accuracy: 0.9361
47840/60000 [======================>.......] - ETA: 20s - loss: 0.2095 - categorical_accuracy: 0.9361
47872/60000 [======================>.......] - ETA: 20s - loss: 0.2094 - categorical_accuracy: 0.9361
47904/60000 [======================>.......] - ETA: 20s - loss: 0.2093 - categorical_accuracy: 0.9361
47936/60000 [======================>.......] - ETA: 20s - loss: 0.2092 - categorical_accuracy: 0.9361
47968/60000 [======================>.......] - ETA: 20s - loss: 0.2091 - categorical_accuracy: 0.9362
48000/60000 [=======================>......] - ETA: 20s - loss: 0.2090 - categorical_accuracy: 0.9362
48032/60000 [=======================>......] - ETA: 20s - loss: 0.2089 - categorical_accuracy: 0.9362
48064/60000 [=======================>......] - ETA: 20s - loss: 0.2088 - categorical_accuracy: 0.9363
48096/60000 [=======================>......] - ETA: 20s - loss: 0.2088 - categorical_accuracy: 0.9363
48128/60000 [=======================>......] - ETA: 20s - loss: 0.2087 - categorical_accuracy: 0.9363
48160/60000 [=======================>......] - ETA: 20s - loss: 0.2086 - categorical_accuracy: 0.9363
48224/60000 [=======================>......] - ETA: 19s - loss: 0.2085 - categorical_accuracy: 0.9363
48288/60000 [=======================>......] - ETA: 19s - loss: 0.2084 - categorical_accuracy: 0.9364
48320/60000 [=======================>......] - ETA: 19s - loss: 0.2084 - categorical_accuracy: 0.9363
48352/60000 [=======================>......] - ETA: 19s - loss: 0.2083 - categorical_accuracy: 0.9363
48384/60000 [=======================>......] - ETA: 19s - loss: 0.2082 - categorical_accuracy: 0.9364
48416/60000 [=======================>......] - ETA: 19s - loss: 0.2082 - categorical_accuracy: 0.9364
48448/60000 [=======================>......] - ETA: 19s - loss: 0.2080 - categorical_accuracy: 0.9364
48480/60000 [=======================>......] - ETA: 19s - loss: 0.2079 - categorical_accuracy: 0.9364
48512/60000 [=======================>......] - ETA: 19s - loss: 0.2078 - categorical_accuracy: 0.9365
48544/60000 [=======================>......] - ETA: 19s - loss: 0.2079 - categorical_accuracy: 0.9365
48576/60000 [=======================>......] - ETA: 19s - loss: 0.2078 - categorical_accuracy: 0.9366
48608/60000 [=======================>......] - ETA: 19s - loss: 0.2077 - categorical_accuracy: 0.9366
48640/60000 [=======================>......] - ETA: 19s - loss: 0.2077 - categorical_accuracy: 0.9366
48672/60000 [=======================>......] - ETA: 19s - loss: 0.2076 - categorical_accuracy: 0.9366
48736/60000 [=======================>......] - ETA: 19s - loss: 0.2075 - categorical_accuracy: 0.9366
48768/60000 [=======================>......] - ETA: 19s - loss: 0.2073 - categorical_accuracy: 0.9367
48800/60000 [=======================>......] - ETA: 18s - loss: 0.2075 - categorical_accuracy: 0.9367
48832/60000 [=======================>......] - ETA: 18s - loss: 0.2074 - categorical_accuracy: 0.9367
48864/60000 [=======================>......] - ETA: 18s - loss: 0.2073 - categorical_accuracy: 0.9367
48896/60000 [=======================>......] - ETA: 18s - loss: 0.2072 - categorical_accuracy: 0.9368
48928/60000 [=======================>......] - ETA: 18s - loss: 0.2072 - categorical_accuracy: 0.9367
48960/60000 [=======================>......] - ETA: 18s - loss: 0.2071 - categorical_accuracy: 0.9368
48992/60000 [=======================>......] - ETA: 18s - loss: 0.2070 - categorical_accuracy: 0.9368
49024/60000 [=======================>......] - ETA: 18s - loss: 0.2069 - categorical_accuracy: 0.9369
49056/60000 [=======================>......] - ETA: 18s - loss: 0.2072 - categorical_accuracy: 0.9369
49088/60000 [=======================>......] - ETA: 18s - loss: 0.2071 - categorical_accuracy: 0.9369
49120/60000 [=======================>......] - ETA: 18s - loss: 0.2071 - categorical_accuracy: 0.9369
49152/60000 [=======================>......] - ETA: 18s - loss: 0.2070 - categorical_accuracy: 0.9369
49184/60000 [=======================>......] - ETA: 18s - loss: 0.2070 - categorical_accuracy: 0.9369
49216/60000 [=======================>......] - ETA: 18s - loss: 0.2069 - categorical_accuracy: 0.9370
49248/60000 [=======================>......] - ETA: 18s - loss: 0.2068 - categorical_accuracy: 0.9370
49280/60000 [=======================>......] - ETA: 18s - loss: 0.2068 - categorical_accuracy: 0.9370
49312/60000 [=======================>......] - ETA: 18s - loss: 0.2067 - categorical_accuracy: 0.9370
49344/60000 [=======================>......] - ETA: 18s - loss: 0.2066 - categorical_accuracy: 0.9371
49376/60000 [=======================>......] - ETA: 18s - loss: 0.2065 - categorical_accuracy: 0.9371
49408/60000 [=======================>......] - ETA: 17s - loss: 0.2064 - categorical_accuracy: 0.9371
49440/60000 [=======================>......] - ETA: 17s - loss: 0.2064 - categorical_accuracy: 0.9371
49472/60000 [=======================>......] - ETA: 17s - loss: 0.2063 - categorical_accuracy: 0.9372
49504/60000 [=======================>......] - ETA: 17s - loss: 0.2062 - categorical_accuracy: 0.9372
49536/60000 [=======================>......] - ETA: 17s - loss: 0.2061 - categorical_accuracy: 0.9372
49568/60000 [=======================>......] - ETA: 17s - loss: 0.2061 - categorical_accuracy: 0.9373
49600/60000 [=======================>......] - ETA: 17s - loss: 0.2061 - categorical_accuracy: 0.9373
49632/60000 [=======================>......] - ETA: 17s - loss: 0.2060 - categorical_accuracy: 0.9373
49664/60000 [=======================>......] - ETA: 17s - loss: 0.2059 - categorical_accuracy: 0.9373
49696/60000 [=======================>......] - ETA: 17s - loss: 0.2058 - categorical_accuracy: 0.9374
49728/60000 [=======================>......] - ETA: 17s - loss: 0.2057 - categorical_accuracy: 0.9374
49760/60000 [=======================>......] - ETA: 17s - loss: 0.2056 - categorical_accuracy: 0.9374
49792/60000 [=======================>......] - ETA: 17s - loss: 0.2055 - categorical_accuracy: 0.9375
49824/60000 [=======================>......] - ETA: 17s - loss: 0.2055 - categorical_accuracy: 0.9375
49856/60000 [=======================>......] - ETA: 17s - loss: 0.2054 - categorical_accuracy: 0.9375
49888/60000 [=======================>......] - ETA: 17s - loss: 0.2053 - categorical_accuracy: 0.9376
49920/60000 [=======================>......] - ETA: 17s - loss: 0.2052 - categorical_accuracy: 0.9376
49952/60000 [=======================>......] - ETA: 17s - loss: 0.2051 - categorical_accuracy: 0.9376
49984/60000 [=======================>......] - ETA: 16s - loss: 0.2050 - categorical_accuracy: 0.9376
50016/60000 [========================>.....] - ETA: 16s - loss: 0.2050 - categorical_accuracy: 0.9377
50048/60000 [========================>.....] - ETA: 16s - loss: 0.2049 - categorical_accuracy: 0.9377
50080/60000 [========================>.....] - ETA: 16s - loss: 0.2048 - categorical_accuracy: 0.9377
50112/60000 [========================>.....] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9377
50144/60000 [========================>.....] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9377
50176/60000 [========================>.....] - ETA: 16s - loss: 0.2046 - categorical_accuracy: 0.9378
50208/60000 [========================>.....] - ETA: 16s - loss: 0.2046 - categorical_accuracy: 0.9378
50240/60000 [========================>.....] - ETA: 16s - loss: 0.2047 - categorical_accuracy: 0.9377
50272/60000 [========================>.....] - ETA: 16s - loss: 0.2045 - categorical_accuracy: 0.9378
50336/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9378
50368/60000 [========================>.....] - ETA: 16s - loss: 0.2045 - categorical_accuracy: 0.9378
50400/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9378
50432/60000 [========================>.....] - ETA: 16s - loss: 0.2044 - categorical_accuracy: 0.9378
50464/60000 [========================>.....] - ETA: 16s - loss: 0.2043 - categorical_accuracy: 0.9379
50496/60000 [========================>.....] - ETA: 16s - loss: 0.2042 - categorical_accuracy: 0.9379
50560/60000 [========================>.....] - ETA: 15s - loss: 0.2041 - categorical_accuracy: 0.9379
50592/60000 [========================>.....] - ETA: 15s - loss: 0.2040 - categorical_accuracy: 0.9380
50624/60000 [========================>.....] - ETA: 15s - loss: 0.2039 - categorical_accuracy: 0.9380
50656/60000 [========================>.....] - ETA: 15s - loss: 0.2039 - categorical_accuracy: 0.9380
50688/60000 [========================>.....] - ETA: 15s - loss: 0.2039 - categorical_accuracy: 0.9380
50720/60000 [========================>.....] - ETA: 15s - loss: 0.2037 - categorical_accuracy: 0.9381
50752/60000 [========================>.....] - ETA: 15s - loss: 0.2037 - categorical_accuracy: 0.9381
50784/60000 [========================>.....] - ETA: 15s - loss: 0.2036 - categorical_accuracy: 0.9381
50816/60000 [========================>.....] - ETA: 15s - loss: 0.2035 - categorical_accuracy: 0.9381
50848/60000 [========================>.....] - ETA: 15s - loss: 0.2034 - categorical_accuracy: 0.9381
50880/60000 [========================>.....] - ETA: 15s - loss: 0.2033 - categorical_accuracy: 0.9381
50912/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9382
50944/60000 [========================>.....] - ETA: 15s - loss: 0.2032 - categorical_accuracy: 0.9382
50976/60000 [========================>.....] - ETA: 15s - loss: 0.2031 - categorical_accuracy: 0.9382
51008/60000 [========================>.....] - ETA: 15s - loss: 0.2030 - categorical_accuracy: 0.9382
51040/60000 [========================>.....] - ETA: 15s - loss: 0.2030 - categorical_accuracy: 0.9383
51072/60000 [========================>.....] - ETA: 15s - loss: 0.2030 - categorical_accuracy: 0.9383
51104/60000 [========================>.....] - ETA: 15s - loss: 0.2030 - categorical_accuracy: 0.9382
51136/60000 [========================>.....] - ETA: 15s - loss: 0.2029 - categorical_accuracy: 0.9383
51168/60000 [========================>.....] - ETA: 14s - loss: 0.2028 - categorical_accuracy: 0.9383
51232/60000 [========================>.....] - ETA: 14s - loss: 0.2026 - categorical_accuracy: 0.9384
51264/60000 [========================>.....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9384
51296/60000 [========================>.....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9384
51328/60000 [========================>.....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9384
51392/60000 [========================>.....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9384
51424/60000 [========================>.....] - ETA: 14s - loss: 0.2025 - categorical_accuracy: 0.9384
51456/60000 [========================>.....] - ETA: 14s - loss: 0.2024 - categorical_accuracy: 0.9384
51488/60000 [========================>.....] - ETA: 14s - loss: 0.2023 - categorical_accuracy: 0.9385
51520/60000 [========================>.....] - ETA: 14s - loss: 0.2022 - categorical_accuracy: 0.9385
51552/60000 [========================>.....] - ETA: 14s - loss: 0.2021 - categorical_accuracy: 0.9385
51584/60000 [========================>.....] - ETA: 14s - loss: 0.2020 - categorical_accuracy: 0.9385
51616/60000 [========================>.....] - ETA: 14s - loss: 0.2019 - categorical_accuracy: 0.9386
51648/60000 [========================>.....] - ETA: 14s - loss: 0.2018 - categorical_accuracy: 0.9386
51680/60000 [========================>.....] - ETA: 14s - loss: 0.2017 - categorical_accuracy: 0.9386
51712/60000 [========================>.....] - ETA: 14s - loss: 0.2016 - categorical_accuracy: 0.9387
51744/60000 [========================>.....] - ETA: 13s - loss: 0.2016 - categorical_accuracy: 0.9387
51776/60000 [========================>.....] - ETA: 13s - loss: 0.2017 - categorical_accuracy: 0.9386
51808/60000 [========================>.....] - ETA: 13s - loss: 0.2017 - categorical_accuracy: 0.9386
51840/60000 [========================>.....] - ETA: 13s - loss: 0.2016 - categorical_accuracy: 0.9387
51872/60000 [========================>.....] - ETA: 13s - loss: 0.2016 - categorical_accuracy: 0.9387
51904/60000 [========================>.....] - ETA: 13s - loss: 0.2015 - categorical_accuracy: 0.9387
51936/60000 [========================>.....] - ETA: 13s - loss: 0.2014 - categorical_accuracy: 0.9388
51968/60000 [========================>.....] - ETA: 13s - loss: 0.2013 - categorical_accuracy: 0.9388
52000/60000 [=========================>....] - ETA: 13s - loss: 0.2013 - categorical_accuracy: 0.9388
52032/60000 [=========================>....] - ETA: 13s - loss: 0.2012 - categorical_accuracy: 0.9388
52064/60000 [=========================>....] - ETA: 13s - loss: 0.2011 - categorical_accuracy: 0.9389
52096/60000 [=========================>....] - ETA: 13s - loss: 0.2010 - categorical_accuracy: 0.9389
52128/60000 [=========================>....] - ETA: 13s - loss: 0.2010 - categorical_accuracy: 0.9389
52160/60000 [=========================>....] - ETA: 13s - loss: 0.2008 - categorical_accuracy: 0.9389
52192/60000 [=========================>....] - ETA: 13s - loss: 0.2008 - categorical_accuracy: 0.9389
52224/60000 [=========================>....] - ETA: 13s - loss: 0.2007 - categorical_accuracy: 0.9390
52256/60000 [=========================>....] - ETA: 13s - loss: 0.2007 - categorical_accuracy: 0.9390
52288/60000 [=========================>....] - ETA: 13s - loss: 0.2007 - categorical_accuracy: 0.9390
52320/60000 [=========================>....] - ETA: 13s - loss: 0.2006 - categorical_accuracy: 0.9390
52352/60000 [=========================>....] - ETA: 12s - loss: 0.2006 - categorical_accuracy: 0.9390
52384/60000 [=========================>....] - ETA: 12s - loss: 0.2005 - categorical_accuracy: 0.9390
52416/60000 [=========================>....] - ETA: 12s - loss: 0.2004 - categorical_accuracy: 0.9390
52448/60000 [=========================>....] - ETA: 12s - loss: 0.2003 - categorical_accuracy: 0.9391
52480/60000 [=========================>....] - ETA: 12s - loss: 0.2002 - categorical_accuracy: 0.9391
52512/60000 [=========================>....] - ETA: 12s - loss: 0.2001 - categorical_accuracy: 0.9391
52544/60000 [=========================>....] - ETA: 12s - loss: 0.2000 - categorical_accuracy: 0.9391
52576/60000 [=========================>....] - ETA: 12s - loss: 0.1999 - categorical_accuracy: 0.9392
52608/60000 [=========================>....] - ETA: 12s - loss: 0.1999 - categorical_accuracy: 0.9392
52640/60000 [=========================>....] - ETA: 12s - loss: 0.1999 - categorical_accuracy: 0.9392
52672/60000 [=========================>....] - ETA: 12s - loss: 0.1998 - categorical_accuracy: 0.9392
52704/60000 [=========================>....] - ETA: 12s - loss: 0.1997 - categorical_accuracy: 0.9392
52736/60000 [=========================>....] - ETA: 12s - loss: 0.1996 - categorical_accuracy: 0.9393
52768/60000 [=========================>....] - ETA: 12s - loss: 0.1995 - categorical_accuracy: 0.9393
52800/60000 [=========================>....] - ETA: 12s - loss: 0.1994 - categorical_accuracy: 0.9393
52832/60000 [=========================>....] - ETA: 12s - loss: 0.1993 - categorical_accuracy: 0.9394
52896/60000 [=========================>....] - ETA: 12s - loss: 0.1994 - categorical_accuracy: 0.9394
52928/60000 [=========================>....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9394
52960/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9394
52992/60000 [=========================>....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9394
53024/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9394
53056/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9395
53088/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9395
53120/60000 [=========================>....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9395
53152/60000 [=========================>....] - ETA: 11s - loss: 0.1992 - categorical_accuracy: 0.9395
53216/60000 [=========================>....] - ETA: 11s - loss: 0.1990 - categorical_accuracy: 0.9395
53248/60000 [=========================>....] - ETA: 11s - loss: 0.1989 - categorical_accuracy: 0.9396
53280/60000 [=========================>....] - ETA: 11s - loss: 0.1988 - categorical_accuracy: 0.9396
53312/60000 [=========================>....] - ETA: 11s - loss: 0.1987 - categorical_accuracy: 0.9396
53344/60000 [=========================>....] - ETA: 11s - loss: 0.1986 - categorical_accuracy: 0.9397
53376/60000 [=========================>....] - ETA: 11s - loss: 0.1986 - categorical_accuracy: 0.9397
53408/60000 [=========================>....] - ETA: 11s - loss: 0.1985 - categorical_accuracy: 0.9397
53440/60000 [=========================>....] - ETA: 11s - loss: 0.1984 - categorical_accuracy: 0.9397
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1984 - categorical_accuracy: 0.9398
53536/60000 [=========================>....] - ETA: 10s - loss: 0.1985 - categorical_accuracy: 0.9397
53568/60000 [=========================>....] - ETA: 10s - loss: 0.1984 - categorical_accuracy: 0.9397
53600/60000 [=========================>....] - ETA: 10s - loss: 0.1984 - categorical_accuracy: 0.9398
53632/60000 [=========================>....] - ETA: 10s - loss: 0.1983 - categorical_accuracy: 0.9398
53664/60000 [=========================>....] - ETA: 10s - loss: 0.1982 - categorical_accuracy: 0.9398
53696/60000 [=========================>....] - ETA: 10s - loss: 0.1981 - categorical_accuracy: 0.9398
53728/60000 [=========================>....] - ETA: 10s - loss: 0.1981 - categorical_accuracy: 0.9398
53760/60000 [=========================>....] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9398
53792/60000 [=========================>....] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9398
53824/60000 [=========================>....] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9399
53856/60000 [=========================>....] - ETA: 10s - loss: 0.1978 - categorical_accuracy: 0.9399
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9399
53920/60000 [=========================>....] - ETA: 10s - loss: 0.1977 - categorical_accuracy: 0.9399
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1976 - categorical_accuracy: 0.9400
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1975 - categorical_accuracy: 0.9400
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1975 - categorical_accuracy: 0.9400
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1974 - categorical_accuracy: 0.9400
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1974 - categorical_accuracy: 0.9400
54112/60000 [==========================>...] - ETA: 9s - loss: 0.1973 - categorical_accuracy: 0.9401 
54144/60000 [==========================>...] - ETA: 9s - loss: 0.1973 - categorical_accuracy: 0.9401
54208/60000 [==========================>...] - ETA: 9s - loss: 0.1971 - categorical_accuracy: 0.9401
54240/60000 [==========================>...] - ETA: 9s - loss: 0.1972 - categorical_accuracy: 0.9401
54272/60000 [==========================>...] - ETA: 9s - loss: 0.1972 - categorical_accuracy: 0.9401
54304/60000 [==========================>...] - ETA: 9s - loss: 0.1971 - categorical_accuracy: 0.9401
54336/60000 [==========================>...] - ETA: 9s - loss: 0.1970 - categorical_accuracy: 0.9402
54368/60000 [==========================>...] - ETA: 9s - loss: 0.1969 - categorical_accuracy: 0.9402
54400/60000 [==========================>...] - ETA: 9s - loss: 0.1968 - categorical_accuracy: 0.9402
54432/60000 [==========================>...] - ETA: 9s - loss: 0.1967 - categorical_accuracy: 0.9403
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1966 - categorical_accuracy: 0.9403
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1966 - categorical_accuracy: 0.9403
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1966 - categorical_accuracy: 0.9403
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1967 - categorical_accuracy: 0.9403
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1966 - categorical_accuracy: 0.9403
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1965 - categorical_accuracy: 0.9403
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1964 - categorical_accuracy: 0.9404
54688/60000 [==========================>...] - ETA: 8s - loss: 0.1964 - categorical_accuracy: 0.9404
54720/60000 [==========================>...] - ETA: 8s - loss: 0.1963 - categorical_accuracy: 0.9404
54752/60000 [==========================>...] - ETA: 8s - loss: 0.1962 - categorical_accuracy: 0.9404
54784/60000 [==========================>...] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9404
54816/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9404
54848/60000 [==========================>...] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9405
54880/60000 [==========================>...] - ETA: 8s - loss: 0.1959 - categorical_accuracy: 0.9405
54912/60000 [==========================>...] - ETA: 8s - loss: 0.1959 - categorical_accuracy: 0.9405
54944/60000 [==========================>...] - ETA: 8s - loss: 0.1958 - categorical_accuracy: 0.9405
54976/60000 [==========================>...] - ETA: 8s - loss: 0.1959 - categorical_accuracy: 0.9405
55008/60000 [==========================>...] - ETA: 8s - loss: 0.1958 - categorical_accuracy: 0.9405
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1957 - categorical_accuracy: 0.9405
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1956 - categorical_accuracy: 0.9406
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1956 - categorical_accuracy: 0.9406
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1955 - categorical_accuracy: 0.9406
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9407
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9407
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9407
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9407
55296/60000 [==========================>...] - ETA: 7s - loss: 0.1954 - categorical_accuracy: 0.9407
55328/60000 [==========================>...] - ETA: 7s - loss: 0.1953 - categorical_accuracy: 0.9407
55360/60000 [==========================>...] - ETA: 7s - loss: 0.1953 - categorical_accuracy: 0.9407
55392/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9407
55424/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9407
55456/60000 [==========================>...] - ETA: 7s - loss: 0.1953 - categorical_accuracy: 0.9407
55488/60000 [==========================>...] - ETA: 7s - loss: 0.1953 - categorical_accuracy: 0.9408
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9408
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9408
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1951 - categorical_accuracy: 0.9408
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1951 - categorical_accuracy: 0.9408
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1950 - categorical_accuracy: 0.9408
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1950 - categorical_accuracy: 0.9408
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9408
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1948 - categorical_accuracy: 0.9408
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1948 - categorical_accuracy: 0.9409
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9409
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9409
55872/60000 [==========================>...] - ETA: 6s - loss: 0.1946 - categorical_accuracy: 0.9409
55904/60000 [==========================>...] - ETA: 6s - loss: 0.1945 - categorical_accuracy: 0.9409
55936/60000 [==========================>...] - ETA: 6s - loss: 0.1946 - categorical_accuracy: 0.9409
55968/60000 [==========================>...] - ETA: 6s - loss: 0.1945 - categorical_accuracy: 0.9410
56000/60000 [===========================>..] - ETA: 6s - loss: 0.1944 - categorical_accuracy: 0.9410
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1943 - categorical_accuracy: 0.9410
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1942 - categorical_accuracy: 0.9410
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1942 - categorical_accuracy: 0.9410
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1941 - categorical_accuracy: 0.9411
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1941 - categorical_accuracy: 0.9411
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1940 - categorical_accuracy: 0.9411
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1940 - categorical_accuracy: 0.9411
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9412
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9412
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9412
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1936 - categorical_accuracy: 0.9413
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1936 - categorical_accuracy: 0.9412
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9413
56480/60000 [===========================>..] - ETA: 5s - loss: 0.1934 - categorical_accuracy: 0.9413
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1933 - categorical_accuracy: 0.9413
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9414
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9414
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1932 - categorical_accuracy: 0.9414
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9414
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9415
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9414
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9415
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9415
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1929 - categorical_accuracy: 0.9415
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1928 - categorical_accuracy: 0.9415
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1927 - categorical_accuracy: 0.9415
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1927 - categorical_accuracy: 0.9415
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1926 - categorical_accuracy: 0.9415
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1925 - categorical_accuracy: 0.9416
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9416
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9416
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1923 - categorical_accuracy: 0.9416
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1924 - categorical_accuracy: 0.9416
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1923 - categorical_accuracy: 0.9417
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1923 - categorical_accuracy: 0.9417
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1922 - categorical_accuracy: 0.9417
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1921 - categorical_accuracy: 0.9417
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1920 - categorical_accuracy: 0.9418
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1919 - categorical_accuracy: 0.9418
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1919 - categorical_accuracy: 0.9418
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1919 - categorical_accuracy: 0.9418
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1919 - categorical_accuracy: 0.9418
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1919 - categorical_accuracy: 0.9417
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1918 - categorical_accuracy: 0.9418
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1917 - categorical_accuracy: 0.9418
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1917 - categorical_accuracy: 0.9418
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1916 - categorical_accuracy: 0.9418
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1916 - categorical_accuracy: 0.9418
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1916 - categorical_accuracy: 0.9418
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1915 - categorical_accuracy: 0.9418
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1914 - categorical_accuracy: 0.9419
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1913 - categorical_accuracy: 0.9419
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1912 - categorical_accuracy: 0.9419
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1911 - categorical_accuracy: 0.9419
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1911 - categorical_accuracy: 0.9420
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1912 - categorical_accuracy: 0.9420
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1910 - categorical_accuracy: 0.9420
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1909 - categorical_accuracy: 0.9420
58016/60000 [============================>.] - ETA: 3s - loss: 0.1909 - categorical_accuracy: 0.9421
58048/60000 [============================>.] - ETA: 3s - loss: 0.1908 - categorical_accuracy: 0.9421
58080/60000 [============================>.] - ETA: 3s - loss: 0.1908 - categorical_accuracy: 0.9421
58112/60000 [============================>.] - ETA: 3s - loss: 0.1908 - categorical_accuracy: 0.9421
58144/60000 [============================>.] - ETA: 3s - loss: 0.1907 - categorical_accuracy: 0.9421
58176/60000 [============================>.] - ETA: 3s - loss: 0.1907 - categorical_accuracy: 0.9421
58208/60000 [============================>.] - ETA: 3s - loss: 0.1906 - categorical_accuracy: 0.9421
58240/60000 [============================>.] - ETA: 2s - loss: 0.1906 - categorical_accuracy: 0.9421
58272/60000 [============================>.] - ETA: 2s - loss: 0.1905 - categorical_accuracy: 0.9422
58336/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9422
58368/60000 [============================>.] - ETA: 2s - loss: 0.1903 - categorical_accuracy: 0.9422
58400/60000 [============================>.] - ETA: 2s - loss: 0.1902 - categorical_accuracy: 0.9422
58432/60000 [============================>.] - ETA: 2s - loss: 0.1901 - categorical_accuracy: 0.9423
58464/60000 [============================>.] - ETA: 2s - loss: 0.1901 - categorical_accuracy: 0.9423
58496/60000 [============================>.] - ETA: 2s - loss: 0.1900 - categorical_accuracy: 0.9423
58528/60000 [============================>.] - ETA: 2s - loss: 0.1900 - categorical_accuracy: 0.9423
58560/60000 [============================>.] - ETA: 2s - loss: 0.1901 - categorical_accuracy: 0.9423
58624/60000 [============================>.] - ETA: 2s - loss: 0.1900 - categorical_accuracy: 0.9423
58656/60000 [============================>.] - ETA: 2s - loss: 0.1899 - categorical_accuracy: 0.9424
58688/60000 [============================>.] - ETA: 2s - loss: 0.1898 - categorical_accuracy: 0.9424
58720/60000 [============================>.] - ETA: 2s - loss: 0.1897 - categorical_accuracy: 0.9424
58752/60000 [============================>.] - ETA: 2s - loss: 0.1896 - categorical_accuracy: 0.9424
58784/60000 [============================>.] - ETA: 2s - loss: 0.1896 - categorical_accuracy: 0.9425
58816/60000 [============================>.] - ETA: 2s - loss: 0.1895 - categorical_accuracy: 0.9425
58848/60000 [============================>.] - ETA: 1s - loss: 0.1896 - categorical_accuracy: 0.9424
58880/60000 [============================>.] - ETA: 1s - loss: 0.1896 - categorical_accuracy: 0.9424
58912/60000 [============================>.] - ETA: 1s - loss: 0.1895 - categorical_accuracy: 0.9425
58944/60000 [============================>.] - ETA: 1s - loss: 0.1895 - categorical_accuracy: 0.9425
58976/60000 [============================>.] - ETA: 1s - loss: 0.1894 - categorical_accuracy: 0.9425
59008/60000 [============================>.] - ETA: 1s - loss: 0.1894 - categorical_accuracy: 0.9425
59040/60000 [============================>.] - ETA: 1s - loss: 0.1893 - categorical_accuracy: 0.9425
59104/60000 [============================>.] - ETA: 1s - loss: 0.1892 - categorical_accuracy: 0.9426
59136/60000 [============================>.] - ETA: 1s - loss: 0.1891 - categorical_accuracy: 0.9426
59168/60000 [============================>.] - ETA: 1s - loss: 0.1891 - categorical_accuracy: 0.9426
59200/60000 [============================>.] - ETA: 1s - loss: 0.1891 - categorical_accuracy: 0.9426
59232/60000 [============================>.] - ETA: 1s - loss: 0.1890 - categorical_accuracy: 0.9426
59264/60000 [============================>.] - ETA: 1s - loss: 0.1890 - categorical_accuracy: 0.9426
59296/60000 [============================>.] - ETA: 1s - loss: 0.1889 - categorical_accuracy: 0.9426
59328/60000 [============================>.] - ETA: 1s - loss: 0.1889 - categorical_accuracy: 0.9427
59360/60000 [============================>.] - ETA: 1s - loss: 0.1888 - categorical_accuracy: 0.9427
59392/60000 [============================>.] - ETA: 1s - loss: 0.1888 - categorical_accuracy: 0.9427
59424/60000 [============================>.] - ETA: 0s - loss: 0.1888 - categorical_accuracy: 0.9427
59456/60000 [============================>.] - ETA: 0s - loss: 0.1887 - categorical_accuracy: 0.9427
59488/60000 [============================>.] - ETA: 0s - loss: 0.1886 - categorical_accuracy: 0.9427
59520/60000 [============================>.] - ETA: 0s - loss: 0.1885 - categorical_accuracy: 0.9428
59552/60000 [============================>.] - ETA: 0s - loss: 0.1886 - categorical_accuracy: 0.9428
59584/60000 [============================>.] - ETA: 0s - loss: 0.1885 - categorical_accuracy: 0.9428
59616/60000 [============================>.] - ETA: 0s - loss: 0.1885 - categorical_accuracy: 0.9428
59648/60000 [============================>.] - ETA: 0s - loss: 0.1885 - categorical_accuracy: 0.9428
59680/60000 [============================>.] - ETA: 0s - loss: 0.1884 - categorical_accuracy: 0.9428
59712/60000 [============================>.] - ETA: 0s - loss: 0.1883 - categorical_accuracy: 0.9429
59744/60000 [============================>.] - ETA: 0s - loss: 0.1883 - categorical_accuracy: 0.9429
59776/60000 [============================>.] - ETA: 0s - loss: 0.1882 - categorical_accuracy: 0.9429
59808/60000 [============================>.] - ETA: 0s - loss: 0.1881 - categorical_accuracy: 0.9429
59840/60000 [============================>.] - ETA: 0s - loss: 0.1880 - categorical_accuracy: 0.9429
59872/60000 [============================>.] - ETA: 0s - loss: 0.1880 - categorical_accuracy: 0.9429
59904/60000 [============================>.] - ETA: 0s - loss: 0.1880 - categorical_accuracy: 0.9430
59936/60000 [============================>.] - ETA: 0s - loss: 0.1879 - categorical_accuracy: 0.9430
59968/60000 [============================>.] - ETA: 0s - loss: 0.1878 - categorical_accuracy: 0.9430
60000/60000 [==============================] - 105s 2ms/step - loss: 0.1878 - categorical_accuracy: 0.9430 - val_loss: 0.0462 - val_categorical_accuracy: 0.9848

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 14s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 3s
  672/10000 [=>............................] - ETA: 3s
  832/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 3s
 1312/10000 [==>...........................] - ETA: 3s
 1472/10000 [===>..........................] - ETA: 3s
 1632/10000 [===>..........................] - ETA: 2s
 1792/10000 [====>.........................] - ETA: 2s
 1952/10000 [====>.........................] - ETA: 2s
 2112/10000 [=====>........................] - ETA: 2s
 2272/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2592/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2944/10000 [=======>......................] - ETA: 2s
 3104/10000 [========>.....................] - ETA: 2s
 3296/10000 [========>.....................] - ETA: 2s
 3456/10000 [=========>....................] - ETA: 2s
 3648/10000 [=========>....................] - ETA: 2s
 3808/10000 [==========>...................] - ETA: 2s
 4000/10000 [===========>..................] - ETA: 2s
 4160/10000 [===========>..................] - ETA: 1s
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
 6112/10000 [=================>............] - ETA: 1s
 6272/10000 [=================>............] - ETA: 1s
 6464/10000 [==================>...........] - ETA: 1s
 6624/10000 [==================>...........] - ETA: 1s
 6784/10000 [===================>..........] - ETA: 1s
 6944/10000 [===================>..........] - ETA: 1s
 7136/10000 [====================>.........] - ETA: 0s
 7296/10000 [====================>.........] - ETA: 0s
 7456/10000 [=====================>........] - ETA: 0s
 7616/10000 [=====================>........] - ETA: 0s
 7776/10000 [======================>.......] - ETA: 0s
 7936/10000 [======================>.......] - ETA: 0s
 8128/10000 [=======================>......] - ETA: 0s
 8288/10000 [=======================>......] - ETA: 0s
 8448/10000 [========================>.....] - ETA: 0s
 8608/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9120/10000 [==========================>...] - ETA: 0s
 9280/10000 [==========================>...] - ETA: 0s
 9472/10000 [===========================>..] - ETA: 0s
 9664/10000 [===========================>..] - ETA: 0s
 9856/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 329us/step
[[2.2631867e-09 5.1741957e-09 5.6641676e-07 ... 9.9999833e-01
  2.4409816e-09 3.2236358e-07]
 [4.4372853e-05 5.5653421e-05 9.9988055e-01 ... 1.7566530e-08
  1.5949050e-05 1.3606212e-09]
 [6.1882037e-07 9.9984872e-01 7.9228121e-06 ... 6.0796468e-05
  1.8743532e-06 2.6002809e-07]
 ...
 [6.8998709e-09 7.8489848e-07 5.0091153e-08 ... 2.1085985e-05
  7.5720220e-07 4.3770713e-05]
 [1.6093823e-06 2.0352832e-07 5.9900276e-09 ... 2.3944062e-07
  8.4765919e-04 3.9885545e-06]
 [9.2721558e-07 1.2569302e-07 4.2488730e-07 ... 4.5586596e-10
  7.1771360e-08 3.6728679e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04620787980241003, 'accuracy_test:': 0.9847999811172485}

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
[master f3d23dd] ml_store  && git pull --all
 1 file changed, 1951 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 8d99965...f3d23dd master -> master (forced update)





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
{'loss': 0.45605727285146713, 'loss_history': []}

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
