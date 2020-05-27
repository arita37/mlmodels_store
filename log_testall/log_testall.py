
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
[master ae918ed] ml_store  && git pull --all
 2 files changed, 61 insertions(+), 9895 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
 + 699f27c...ae918ed master -> master (forced update)





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
[master 1fe25eb] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   ae918ed..1fe25eb  master -> master





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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-27 12:15:22.239341: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-27 12:15:22.244165: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-27 12:15:22.244906: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562c605e5b60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 12:15:22.244922: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 1ms/sample - loss: 0.2499 - binary_crossentropy: 0.6928 - val_loss: 0.2498 - val_binary_crossentropy: 0.6927

  #### metrics   #################################################### 
{'MSE': 0.24957098304336808}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
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
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
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
Total params: 438
Trainable params: 438
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2489 - binary_crossentropy: 0.6910500/500 [==============================] - 1s 1ms/sample - loss: 0.2508 - binary_crossentropy: 0.6948 - val_loss: 0.2503 - val_binary_crossentropy: 0.6937

  #### metrics   #################################################### 
{'MSE': 0.2504099798490631}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
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
Total params: 438
Trainable params: 438
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2499 - binary_crossentropy: 0.6930 - val_loss: 0.2498 - val_binary_crossentropy: 0.6928

  #### metrics   #################################################### 
{'MSE': 0.24971226403949623}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
Total params: 453
Trainable params: 453
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2722 - binary_crossentropy: 1.0038500/500 [==============================] - 1s 3ms/sample - loss: 0.2702 - binary_crossentropy: 0.8678 - val_loss: 0.2681 - val_binary_crossentropy: 0.8361

  #### metrics   #################################################### 
{'MSE': 0.26695548707522593}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 188
Trainable params: 188
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2468 - binary_crossentropy: 0.6867500/500 [==============================] - 2s 3ms/sample - loss: 0.2499 - binary_crossentropy: 0.6929 - val_loss: 0.2495 - val_binary_crossentropy: 0.6922

  #### metrics   #################################################### 
{'MSE': 0.24968693683062948}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
Total params: 188
Trainable params: 188
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-27 12:16:41.017942: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:16:41.019937: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:16:41.025382: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 12:16:41.034952: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 12:16:41.036585: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 12:16:41.038233: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:16:41.039703: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2459 - val_binary_crossentropy: 0.6850
2020-05-27 12:16:42.343342: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:16:42.344947: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:16:42.348721: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 12:16:42.357031: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-27 12:16:42.358427: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 12:16:42.359695: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:16:42.360865: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24409215918748353}

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
2020-05-27 12:17:06.486341: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:06.487670: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:06.491303: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 12:17:06.497180: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 12:17:06.498206: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 12:17:06.499172: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:06.500009: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2511 - val_binary_crossentropy: 0.6953
2020-05-27 12:17:08.122685: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:08.123823: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:08.126278: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 12:17:08.131261: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-27 12:17:08.132250: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 12:17:08.133129: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:08.133958: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25131655051042223}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-27 12:17:42.474000: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:42.479521: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:42.495958: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 12:17:42.523275: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 12:17:42.527963: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 12:17:42.532220: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:42.536332: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.4141 - binary_crossentropy: 1.0315 - val_loss: 0.2667 - val_binary_crossentropy: 0.7276
2020-05-27 12:17:44.851812: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:44.855991: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:44.868372: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 12:17:44.893077: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-27 12:17:44.897652: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-27 12:17:44.902065: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-27 12:17:44.906100: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.3094179004123066}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
Total params: 660
Trainable params: 660
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2598 - binary_crossentropy: 0.7256500/500 [==============================] - 4s 9ms/sample - loss: 0.2852 - binary_crossentropy: 0.8562 - val_loss: 0.3053 - val_binary_crossentropy: 1.0271

  #### metrics   #################################################### 
{'MSE': 0.2945731445402514}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         14          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         18          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         2           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_5[0][0]           
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
Total params: 272
Trainable params: 272
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2655 - binary_crossentropy: 0.7284500/500 [==============================] - 5s 9ms/sample - loss: 0.2672 - binary_crossentropy: 0.7325 - val_loss: 0.2736 - val_binary_crossentropy: 0.7451

  #### metrics   #################################################### 
{'MSE': 0.2670465067721657}

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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         18          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         14          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         18          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         2           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_5[0][0]           
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
Total params: 272
Trainable params: 272
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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 5s - loss: 0.2674 - binary_crossentropy: 0.8587500/500 [==============================] - 4s 9ms/sample - loss: 0.2612 - binary_crossentropy: 0.8461 - val_loss: 0.2615 - val_binary_crossentropy: 0.9528

  #### metrics   #################################################### 
{'MSE': 0.25916629240502037}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
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
regionsequence_sum (InputLayer) [(None, 5)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 9)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 8)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         1           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 8s - loss: 0.4501 - binary_crossentropy: 6.9412500/500 [==============================] - 6s 11ms/sample - loss: 0.4741 - binary_crossentropy: 7.3114 - val_loss: 0.4721 - val_binary_crossentropy: 7.2806

  #### metrics   #################################################### 
{'MSE': 0.473}

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
regionsequence_mean (InputLayer [(None, 9)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 8)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         1           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
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
Total params: 1,442
Trainable params: 1,442
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3001 - binary_crossentropy: 0.8172500/500 [==============================] - 6s 11ms/sample - loss: 0.2910 - binary_crossentropy: 0.7975 - val_loss: 0.2887 - val_binary_crossentropy: 0.7872

  #### metrics   #################################################### 
{'MSE': 0.2862964686746584}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
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
Total params: 1,442
Trainable params: 1,442
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           hash_11[0][0]                    
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
Total params: 3,171
Trainable params: 3,091
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2797 - binary_crossentropy: 0.7573500/500 [==============================] - 6s 12ms/sample - loss: 0.2635 - binary_crossentropy: 0.7221 - val_loss: 0.2618 - val_binary_crossentropy: 0.7174

  #### metrics   #################################################### 
{'MSE': 0.260052059824161}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           hash_11[0][0]                    
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
Total params: 3,171
Trainable params: 3,091
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
[master 6f60aba] ml_store  && git pull --all
 1 file changed, 4946 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 9926bdc...6f60aba master -> master (forced update)





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
[master 5039afc] ml_store  && git pull --all
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   6f60aba..5039afc  master -> master





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
[master 220ef9a] ml_store  && git pull --all
 1 file changed, 45 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   5039afc..220ef9a  master -> master





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
[master d7f733f] ml_store  && git pull --all
 1 file changed, 35 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   220ef9a..d7f733f  master -> master





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
[master ddc8749] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   d7f733f..ddc8749  master -> master





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
[master 9b54004] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   ddc8749..9b54004  master -> master





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
[master 0f5a779] ml_store  && git pull --all
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   9b54004..0f5a779  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 11s
  122880/17464789 [..............................] - ETA: 9s 
  245760/17464789 [..............................] - ETA: 8s
  491520/17464789 [..............................] - ETA: 5s
  753664/17464789 [>.............................] - ETA: 4s
 1097728/17464789 [>.............................] - ETA: 4s
 1466368/17464789 [=>............................] - ETA: 3s
 1916928/17464789 [==>...........................] - ETA: 3s
 2318336/17464789 [==>...........................] - ETA: 2s
 2924544/17464789 [====>.........................] - ETA: 2s
 3538944/17464789 [=====>........................] - ETA: 2s
 4284416/17464789 [======>.......................] - ETA: 1s
 4997120/17464789 [=======>......................] - ETA: 1s
 5890048/17464789 [=========>....................] - ETA: 1s
 6774784/17464789 [==========>...................] - ETA: 1s
 7610368/17464789 [============>.................] - ETA: 1s
 8691712/17464789 [=============>................] - ETA: 0s
 9789440/17464789 [===============>..............] - ETA: 0s
11001856/17464789 [=================>............] - ETA: 0s
12255232/17464789 [====================>.........] - ETA: 0s
13615104/17464789 [======================>.......] - ETA: 0s
15024128/17464789 [========================>.....] - ETA: 0s
16506880/17464789 [===========================>..] - ETA: 0s
17465344/17464789 [==============================] - 1s 0us/step
Pad sequences (samples x time)...

  #### Model init, fit   ############################################# 
Using TensorFlow backend.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
2020-05-27 12:27:21.957659: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-27 12:27:21.961987: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-27 12:27:21.962148: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5648c01bc2d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-27 12:27:21.962178: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 11s - loss: 7.9273 - accuracy: 0.4830
 2000/25000 [=>............................] - ETA: 7s - loss: 7.8200 - accuracy: 0.4900 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7280 - accuracy: 0.4960
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7970 - accuracy: 0.4915
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.8016 - accuracy: 0.4912
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7765 - accuracy: 0.4928
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7214 - accuracy: 0.4964
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6685 - accuracy: 0.4999
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6768 - accuracy: 0.4993
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7065 - accuracy: 0.4974
11000/25000 [============>.................] - ETA: 3s - loss: 7.7154 - accuracy: 0.4968
12000/25000 [=============>................] - ETA: 2s - loss: 7.7433 - accuracy: 0.4950
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7468 - accuracy: 0.4948
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7291 - accuracy: 0.4959
15000/25000 [=================>............] - ETA: 2s - loss: 7.7208 - accuracy: 0.4965
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7308 - accuracy: 0.4958
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7180 - accuracy: 0.4966
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7126 - accuracy: 0.4970
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7167 - accuracy: 0.4967
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7264 - accuracy: 0.4961
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7221 - accuracy: 0.4964
22000/25000 [=========================>....] - ETA: 0s - loss: 7.7036 - accuracy: 0.4976
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6933 - accuracy: 0.4983
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 7s 261us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fa2ef4281d0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fa2ebdf9470> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 10s - loss: 7.5746 - accuracy: 0.5060
 2000/25000 [=>............................] - ETA: 7s - loss: 7.6206 - accuracy: 0.5030 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.5785 - accuracy: 0.5058
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6951 - accuracy: 0.4981
 8000/25000 [========>.....................] - ETA: 3s - loss: 7.7069 - accuracy: 0.4974
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6605 - accuracy: 0.5004
11000/25000 [============>.................] - ETA: 3s - loss: 7.6875 - accuracy: 0.4986
12000/25000 [=============>................] - ETA: 2s - loss: 7.6679 - accuracy: 0.4999
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6690 - accuracy: 0.4998
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6677 - accuracy: 0.4999
15000/25000 [=================>............] - ETA: 2s - loss: 7.6503 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 1s - loss: 7.6714 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6729 - accuracy: 0.4996
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6544 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6418 - accuracy: 0.5016
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6367 - accuracy: 0.5020
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6573 - accuracy: 0.5006
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 6s 258us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 10s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 7s - loss: 7.7280 - accuracy: 0.4960 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7024 - accuracy: 0.4977
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6896 - accuracy: 0.4985
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6912 - accuracy: 0.4984
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.6768 - accuracy: 0.4993
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7104 - accuracy: 0.4971
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6628 - accuracy: 0.5002
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.6700 - accuracy: 0.4998
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6436 - accuracy: 0.5015
11000/25000 [============>.................] - ETA: 3s - loss: 7.6583 - accuracy: 0.5005
12000/25000 [=============>................] - ETA: 3s - loss: 7.6487 - accuracy: 0.5012
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6489 - accuracy: 0.5012
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6469 - accuracy: 0.5013
15000/25000 [=================>............] - ETA: 2s - loss: 7.6319 - accuracy: 0.5023
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6091 - accuracy: 0.5038
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6125 - accuracy: 0.5035
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6036 - accuracy: 0.5041
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6029 - accuracy: 0.5042
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6229 - accuracy: 0.5028
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6192 - accuracy: 0.5031
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6353 - accuracy: 0.5020
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6546 - accuracy: 0.5008
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6590 - accuracy: 0.5005
25000/25000 [==============================] - 7s 270us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master 1dc63df] ml_store  && git pull --all
 1 file changed, 335 insertions(+)
To github.com:arita37/mlmodels_store.git
 + defe74f...1dc63df master -> master (forced update)





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

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 02064d4] ml_store  && git pull --all
 1 file changed, 124 insertions(+)
To github.com:arita37/mlmodels_store.git
   1dc63df..02064d4  master -> master





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
 3481600/11490434 [========>.....................] - ETA: 0s
10305536/11490434 [=========================>....] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:50 - loss: 2.3197 - categorical_accuracy: 0.1250
   96/60000 [..............................] - ETA: 3:33 - loss: 2.2635 - categorical_accuracy: 0.1771
  160/60000 [..............................] - ETA: 2:41 - loss: 2.2073 - categorical_accuracy: 0.2062
  224/60000 [..............................] - ETA: 2:20 - loss: 2.1914 - categorical_accuracy: 0.2232
  288/60000 [..............................] - ETA: 2:06 - loss: 2.1133 - categorical_accuracy: 0.2812
  352/60000 [..............................] - ETA: 1:57 - loss: 2.0472 - categorical_accuracy: 0.2926
  416/60000 [..............................] - ETA: 1:52 - loss: 1.9876 - categorical_accuracy: 0.3101
  448/60000 [..............................] - ETA: 1:50 - loss: 1.9643 - categorical_accuracy: 0.3125
  512/60000 [..............................] - ETA: 1:47 - loss: 1.8995 - categorical_accuracy: 0.3359
  576/60000 [..............................] - ETA: 1:43 - loss: 1.8309 - categorical_accuracy: 0.3646
  640/60000 [..............................] - ETA: 1:41 - loss: 1.7838 - categorical_accuracy: 0.3812
  704/60000 [..............................] - ETA: 1:39 - loss: 1.7318 - categorical_accuracy: 0.4020
  768/60000 [..............................] - ETA: 1:37 - loss: 1.6665 - categorical_accuracy: 0.4245
  832/60000 [..............................] - ETA: 1:35 - loss: 1.5988 - categorical_accuracy: 0.4459
  896/60000 [..............................] - ETA: 1:34 - loss: 1.5457 - categorical_accuracy: 0.4654
  960/60000 [..............................] - ETA: 1:33 - loss: 1.4984 - categorical_accuracy: 0.4812
 1024/60000 [..............................] - ETA: 1:33 - loss: 1.4568 - categorical_accuracy: 0.4951
 1088/60000 [..............................] - ETA: 1:31 - loss: 1.4236 - categorical_accuracy: 0.5064
 1152/60000 [..............................] - ETA: 1:30 - loss: 1.3930 - categorical_accuracy: 0.5165
 1216/60000 [..............................] - ETA: 1:29 - loss: 1.3826 - categorical_accuracy: 0.5206
 1280/60000 [..............................] - ETA: 1:29 - loss: 1.3576 - categorical_accuracy: 0.5328
 1344/60000 [..............................] - ETA: 1:28 - loss: 1.3306 - categorical_accuracy: 0.5439
 1408/60000 [..............................] - ETA: 1:28 - loss: 1.3179 - categorical_accuracy: 0.5518
 1472/60000 [..............................] - ETA: 1:27 - loss: 1.2939 - categorical_accuracy: 0.5598
 1536/60000 [..............................] - ETA: 1:27 - loss: 1.2665 - categorical_accuracy: 0.5697
 1600/60000 [..............................] - ETA: 1:26 - loss: 1.2393 - categorical_accuracy: 0.5794
 1664/60000 [..............................] - ETA: 1:26 - loss: 1.2160 - categorical_accuracy: 0.5895
 1728/60000 [..............................] - ETA: 1:25 - loss: 1.2027 - categorical_accuracy: 0.5961
 1792/60000 [..............................] - ETA: 1:25 - loss: 1.1828 - categorical_accuracy: 0.6032
 1856/60000 [..............................] - ETA: 1:25 - loss: 1.1628 - categorical_accuracy: 0.6094
 1920/60000 [..............................] - ETA: 1:25 - loss: 1.1383 - categorical_accuracy: 0.6172
 1984/60000 [..............................] - ETA: 1:24 - loss: 1.1282 - categorical_accuracy: 0.6210
 2048/60000 [>.............................] - ETA: 1:24 - loss: 1.1110 - categorical_accuracy: 0.6260
 2112/60000 [>.............................] - ETA: 1:23 - loss: 1.0975 - categorical_accuracy: 0.6316
 2176/60000 [>.............................] - ETA: 1:23 - loss: 1.0827 - categorical_accuracy: 0.6379
 2240/60000 [>.............................] - ETA: 1:23 - loss: 1.0660 - categorical_accuracy: 0.6433
 2304/60000 [>.............................] - ETA: 1:23 - loss: 1.0578 - categorical_accuracy: 0.6454
 2368/60000 [>.............................] - ETA: 1:22 - loss: 1.0391 - categorical_accuracy: 0.6516
 2432/60000 [>.............................] - ETA: 1:22 - loss: 1.0265 - categorical_accuracy: 0.6567
 2496/60000 [>.............................] - ETA: 1:22 - loss: 1.0144 - categorical_accuracy: 0.6607
 2560/60000 [>.............................] - ETA: 1:22 - loss: 1.0035 - categorical_accuracy: 0.6645
 2624/60000 [>.............................] - ETA: 1:21 - loss: 0.9938 - categorical_accuracy: 0.6684
 2688/60000 [>.............................] - ETA: 1:21 - loss: 0.9838 - categorical_accuracy: 0.6737
 2752/60000 [>.............................] - ETA: 1:21 - loss: 0.9678 - categorical_accuracy: 0.6795
 2816/60000 [>.............................] - ETA: 1:21 - loss: 0.9540 - categorical_accuracy: 0.6839
 2880/60000 [>.............................] - ETA: 1:21 - loss: 0.9473 - categorical_accuracy: 0.6872
 2944/60000 [>.............................] - ETA: 1:21 - loss: 0.9361 - categorical_accuracy: 0.6916
 3008/60000 [>.............................] - ETA: 1:21 - loss: 0.9284 - categorical_accuracy: 0.6951
 3072/60000 [>.............................] - ETA: 1:21 - loss: 0.9164 - categorical_accuracy: 0.6992
 3136/60000 [>.............................] - ETA: 1:20 - loss: 0.9063 - categorical_accuracy: 0.7015
 3200/60000 [>.............................] - ETA: 1:20 - loss: 0.8957 - categorical_accuracy: 0.7059
 3264/60000 [>.............................] - ETA: 1:20 - loss: 0.8845 - categorical_accuracy: 0.7096
 3328/60000 [>.............................] - ETA: 1:20 - loss: 0.8768 - categorical_accuracy: 0.7124
 3392/60000 [>.............................] - ETA: 1:20 - loss: 0.8673 - categorical_accuracy: 0.7155
 3456/60000 [>.............................] - ETA: 1:20 - loss: 0.8581 - categorical_accuracy: 0.7185
 3520/60000 [>.............................] - ETA: 1:19 - loss: 0.8479 - categorical_accuracy: 0.7213
 3584/60000 [>.............................] - ETA: 1:19 - loss: 0.8420 - categorical_accuracy: 0.7235
 3648/60000 [>.............................] - ETA: 1:19 - loss: 0.8377 - categorical_accuracy: 0.7259
 3712/60000 [>.............................] - ETA: 1:19 - loss: 0.8321 - categorical_accuracy: 0.7279
 3776/60000 [>.............................] - ETA: 1:19 - loss: 0.8233 - categorical_accuracy: 0.7307
 3840/60000 [>.............................] - ETA: 1:19 - loss: 0.8158 - categorical_accuracy: 0.7333
 3904/60000 [>.............................] - ETA: 1:19 - loss: 0.8103 - categorical_accuracy: 0.7354
 3968/60000 [>.............................] - ETA: 1:19 - loss: 0.8024 - categorical_accuracy: 0.7379
 4032/60000 [=>............................] - ETA: 1:19 - loss: 0.7978 - categorical_accuracy: 0.7403
 4096/60000 [=>............................] - ETA: 1:18 - loss: 0.7900 - categorical_accuracy: 0.7437
 4160/60000 [=>............................] - ETA: 1:18 - loss: 0.7838 - categorical_accuracy: 0.7459
 4224/60000 [=>............................] - ETA: 1:18 - loss: 0.7758 - categorical_accuracy: 0.7483
 4288/60000 [=>............................] - ETA: 1:18 - loss: 0.7684 - categorical_accuracy: 0.7505
 4352/60000 [=>............................] - ETA: 1:18 - loss: 0.7608 - categorical_accuracy: 0.7534
 4416/60000 [=>............................] - ETA: 1:18 - loss: 0.7531 - categorical_accuracy: 0.7566
 4480/60000 [=>............................] - ETA: 1:18 - loss: 0.7457 - categorical_accuracy: 0.7592
 4544/60000 [=>............................] - ETA: 1:18 - loss: 0.7404 - categorical_accuracy: 0.7612
 4608/60000 [=>............................] - ETA: 1:18 - loss: 0.7346 - categorical_accuracy: 0.7632
 4672/60000 [=>............................] - ETA: 1:17 - loss: 0.7277 - categorical_accuracy: 0.7652
 4736/60000 [=>............................] - ETA: 1:17 - loss: 0.7226 - categorical_accuracy: 0.7673
 4800/60000 [=>............................] - ETA: 1:17 - loss: 0.7180 - categorical_accuracy: 0.7690
 4864/60000 [=>............................] - ETA: 1:17 - loss: 0.7136 - categorical_accuracy: 0.7697
 4928/60000 [=>............................] - ETA: 1:17 - loss: 0.7068 - categorical_accuracy: 0.7723
 4960/60000 [=>............................] - ETA: 1:17 - loss: 0.7049 - categorical_accuracy: 0.7728
 5024/60000 [=>............................] - ETA: 1:17 - loss: 0.6984 - categorical_accuracy: 0.7751
 5088/60000 [=>............................] - ETA: 1:17 - loss: 0.6932 - categorical_accuracy: 0.7773
 5152/60000 [=>............................] - ETA: 1:17 - loss: 0.6888 - categorical_accuracy: 0.7785
 5184/60000 [=>............................] - ETA: 1:17 - loss: 0.6863 - categorical_accuracy: 0.7791
 5248/60000 [=>............................] - ETA: 1:17 - loss: 0.6815 - categorical_accuracy: 0.7805
 5312/60000 [=>............................] - ETA: 1:16 - loss: 0.6768 - categorical_accuracy: 0.7820
 5376/60000 [=>............................] - ETA: 1:16 - loss: 0.6737 - categorical_accuracy: 0.7831
 5440/60000 [=>............................] - ETA: 1:16 - loss: 0.6695 - categorical_accuracy: 0.7851
 5504/60000 [=>............................] - ETA: 1:16 - loss: 0.6641 - categorical_accuracy: 0.7869
 5568/60000 [=>............................] - ETA: 1:16 - loss: 0.6593 - categorical_accuracy: 0.7886
 5632/60000 [=>............................] - ETA: 1:16 - loss: 0.6559 - categorical_accuracy: 0.7894
 5696/60000 [=>............................] - ETA: 1:16 - loss: 0.6496 - categorical_accuracy: 0.7918
 5760/60000 [=>............................] - ETA: 1:16 - loss: 0.6461 - categorical_accuracy: 0.7927
 5824/60000 [=>............................] - ETA: 1:15 - loss: 0.6416 - categorical_accuracy: 0.7941
 5888/60000 [=>............................] - ETA: 1:15 - loss: 0.6379 - categorical_accuracy: 0.7957
 5952/60000 [=>............................] - ETA: 1:15 - loss: 0.6336 - categorical_accuracy: 0.7972
 6016/60000 [==>...........................] - ETA: 1:15 - loss: 0.6287 - categorical_accuracy: 0.7989
 6080/60000 [==>...........................] - ETA: 1:15 - loss: 0.6250 - categorical_accuracy: 0.8005
 6144/60000 [==>...........................] - ETA: 1:15 - loss: 0.6206 - categorical_accuracy: 0.8019
 6208/60000 [==>...........................] - ETA: 1:15 - loss: 0.6162 - categorical_accuracy: 0.8033
 6272/60000 [==>...........................] - ETA: 1:15 - loss: 0.6131 - categorical_accuracy: 0.8044
 6336/60000 [==>...........................] - ETA: 1:14 - loss: 0.6098 - categorical_accuracy: 0.8056
 6400/60000 [==>...........................] - ETA: 1:14 - loss: 0.6056 - categorical_accuracy: 0.8069
 6464/60000 [==>...........................] - ETA: 1:14 - loss: 0.6035 - categorical_accuracy: 0.8079
 6528/60000 [==>...........................] - ETA: 1:14 - loss: 0.6023 - categorical_accuracy: 0.8082
 6592/60000 [==>...........................] - ETA: 1:14 - loss: 0.5993 - categorical_accuracy: 0.8092
 6656/60000 [==>...........................] - ETA: 1:14 - loss: 0.5969 - categorical_accuracy: 0.8102
 6720/60000 [==>...........................] - ETA: 1:14 - loss: 0.5932 - categorical_accuracy: 0.8113
 6784/60000 [==>...........................] - ETA: 1:13 - loss: 0.5916 - categorical_accuracy: 0.8122
 6848/60000 [==>...........................] - ETA: 1:13 - loss: 0.5891 - categorical_accuracy: 0.8131
 6912/60000 [==>...........................] - ETA: 1:13 - loss: 0.5858 - categorical_accuracy: 0.8138
 6976/60000 [==>...........................] - ETA: 1:13 - loss: 0.5824 - categorical_accuracy: 0.8149
 7040/60000 [==>...........................] - ETA: 1:13 - loss: 0.5793 - categorical_accuracy: 0.8162
 7104/60000 [==>...........................] - ETA: 1:13 - loss: 0.5766 - categorical_accuracy: 0.8170
 7168/60000 [==>...........................] - ETA: 1:13 - loss: 0.5732 - categorical_accuracy: 0.8181
 7232/60000 [==>...........................] - ETA: 1:13 - loss: 0.5705 - categorical_accuracy: 0.8189
 7296/60000 [==>...........................] - ETA: 1:13 - loss: 0.5666 - categorical_accuracy: 0.8202
 7360/60000 [==>...........................] - ETA: 1:13 - loss: 0.5642 - categorical_accuracy: 0.8209
 7424/60000 [==>...........................] - ETA: 1:13 - loss: 0.5610 - categorical_accuracy: 0.8219
 7488/60000 [==>...........................] - ETA: 1:12 - loss: 0.5590 - categorical_accuracy: 0.8225
 7552/60000 [==>...........................] - ETA: 1:12 - loss: 0.5552 - categorical_accuracy: 0.8236
 7616/60000 [==>...........................] - ETA: 1:12 - loss: 0.5519 - categorical_accuracy: 0.8247
 7680/60000 [==>...........................] - ETA: 1:12 - loss: 0.5487 - categorical_accuracy: 0.8257
 7744/60000 [==>...........................] - ETA: 1:12 - loss: 0.5456 - categorical_accuracy: 0.8267
 7808/60000 [==>...........................] - ETA: 1:12 - loss: 0.5428 - categorical_accuracy: 0.8277
 7872/60000 [==>...........................] - ETA: 1:12 - loss: 0.5418 - categorical_accuracy: 0.8285
 7936/60000 [==>...........................] - ETA: 1:12 - loss: 0.5398 - categorical_accuracy: 0.8293
 8000/60000 [===>..........................] - ETA: 1:12 - loss: 0.5380 - categorical_accuracy: 0.8299
 8064/60000 [===>..........................] - ETA: 1:12 - loss: 0.5361 - categorical_accuracy: 0.8306
 8128/60000 [===>..........................] - ETA: 1:11 - loss: 0.5336 - categorical_accuracy: 0.8312
 8192/60000 [===>..........................] - ETA: 1:11 - loss: 0.5300 - categorical_accuracy: 0.8324
 8256/60000 [===>..........................] - ETA: 1:11 - loss: 0.5287 - categorical_accuracy: 0.8327
 8320/60000 [===>..........................] - ETA: 1:11 - loss: 0.5267 - categorical_accuracy: 0.8335
 8384/60000 [===>..........................] - ETA: 1:11 - loss: 0.5256 - categorical_accuracy: 0.8337
 8448/60000 [===>..........................] - ETA: 1:11 - loss: 0.5241 - categorical_accuracy: 0.8344
 8512/60000 [===>..........................] - ETA: 1:11 - loss: 0.5219 - categorical_accuracy: 0.8349
 8576/60000 [===>..........................] - ETA: 1:11 - loss: 0.5206 - categorical_accuracy: 0.8351
 8640/60000 [===>..........................] - ETA: 1:11 - loss: 0.5189 - categorical_accuracy: 0.8355
 8704/60000 [===>..........................] - ETA: 1:10 - loss: 0.5168 - categorical_accuracy: 0.8362
 8768/60000 [===>..........................] - ETA: 1:10 - loss: 0.5146 - categorical_accuracy: 0.8370
 8832/60000 [===>..........................] - ETA: 1:10 - loss: 0.5114 - categorical_accuracy: 0.8381
 8896/60000 [===>..........................] - ETA: 1:10 - loss: 0.5085 - categorical_accuracy: 0.8389
 8960/60000 [===>..........................] - ETA: 1:10 - loss: 0.5071 - categorical_accuracy: 0.8396
 9024/60000 [===>..........................] - ETA: 1:10 - loss: 0.5046 - categorical_accuracy: 0.8405
 9088/60000 [===>..........................] - ETA: 1:10 - loss: 0.5025 - categorical_accuracy: 0.8413
 9152/60000 [===>..........................] - ETA: 1:10 - loss: 0.5003 - categorical_accuracy: 0.8420
 9216/60000 [===>..........................] - ETA: 1:10 - loss: 0.4985 - categorical_accuracy: 0.8423
 9280/60000 [===>..........................] - ETA: 1:10 - loss: 0.4961 - categorical_accuracy: 0.8429
 9344/60000 [===>..........................] - ETA: 1:10 - loss: 0.4938 - categorical_accuracy: 0.8435
 9408/60000 [===>..........................] - ETA: 1:09 - loss: 0.4912 - categorical_accuracy: 0.8444
 9472/60000 [===>..........................] - ETA: 1:09 - loss: 0.4900 - categorical_accuracy: 0.8447
 9536/60000 [===>..........................] - ETA: 1:09 - loss: 0.4897 - categorical_accuracy: 0.8450
 9600/60000 [===>..........................] - ETA: 1:09 - loss: 0.4871 - categorical_accuracy: 0.8459
 9664/60000 [===>..........................] - ETA: 1:09 - loss: 0.4849 - categorical_accuracy: 0.8464
 9728/60000 [===>..........................] - ETA: 1:09 - loss: 0.4839 - categorical_accuracy: 0.8469
 9792/60000 [===>..........................] - ETA: 1:09 - loss: 0.4825 - categorical_accuracy: 0.8472
 9856/60000 [===>..........................] - ETA: 1:09 - loss: 0.4817 - categorical_accuracy: 0.8475
 9920/60000 [===>..........................] - ETA: 1:09 - loss: 0.4792 - categorical_accuracy: 0.8484
 9984/60000 [===>..........................] - ETA: 1:08 - loss: 0.4777 - categorical_accuracy: 0.8490
10048/60000 [====>.........................] - ETA: 1:08 - loss: 0.4761 - categorical_accuracy: 0.8495
10112/60000 [====>.........................] - ETA: 1:08 - loss: 0.4748 - categorical_accuracy: 0.8498
10176/60000 [====>.........................] - ETA: 1:08 - loss: 0.4733 - categorical_accuracy: 0.8503
10208/60000 [====>.........................] - ETA: 1:08 - loss: 0.4722 - categorical_accuracy: 0.8506
10272/60000 [====>.........................] - ETA: 1:08 - loss: 0.4715 - categorical_accuracy: 0.8511
10336/60000 [====>.........................] - ETA: 1:08 - loss: 0.4695 - categorical_accuracy: 0.8518
10400/60000 [====>.........................] - ETA: 1:08 - loss: 0.4676 - categorical_accuracy: 0.8523
10464/60000 [====>.........................] - ETA: 1:08 - loss: 0.4650 - categorical_accuracy: 0.8532
10528/60000 [====>.........................] - ETA: 1:08 - loss: 0.4630 - categorical_accuracy: 0.8538
10592/60000 [====>.........................] - ETA: 1:08 - loss: 0.4610 - categorical_accuracy: 0.8544
10656/60000 [====>.........................] - ETA: 1:08 - loss: 0.4592 - categorical_accuracy: 0.8551
10720/60000 [====>.........................] - ETA: 1:08 - loss: 0.4579 - categorical_accuracy: 0.8555
10784/60000 [====>.........................] - ETA: 1:07 - loss: 0.4581 - categorical_accuracy: 0.8555
10848/60000 [====>.........................] - ETA: 1:07 - loss: 0.4570 - categorical_accuracy: 0.8559
10912/60000 [====>.........................] - ETA: 1:07 - loss: 0.4553 - categorical_accuracy: 0.8566
10976/60000 [====>.........................] - ETA: 1:07 - loss: 0.4537 - categorical_accuracy: 0.8568
11040/60000 [====>.........................] - ETA: 1:07 - loss: 0.4522 - categorical_accuracy: 0.8572
11104/60000 [====>.........................] - ETA: 1:07 - loss: 0.4504 - categorical_accuracy: 0.8577
11168/60000 [====>.........................] - ETA: 1:07 - loss: 0.4498 - categorical_accuracy: 0.8580
11232/60000 [====>.........................] - ETA: 1:07 - loss: 0.4481 - categorical_accuracy: 0.8584
11296/60000 [====>.........................] - ETA: 1:07 - loss: 0.4461 - categorical_accuracy: 0.8592
11360/60000 [====>.........................] - ETA: 1:07 - loss: 0.4456 - categorical_accuracy: 0.8596
11424/60000 [====>.........................] - ETA: 1:07 - loss: 0.4441 - categorical_accuracy: 0.8599
11488/60000 [====>.........................] - ETA: 1:07 - loss: 0.4426 - categorical_accuracy: 0.8605
11552/60000 [====>.........................] - ETA: 1:06 - loss: 0.4415 - categorical_accuracy: 0.8608
11616/60000 [====>.........................] - ETA: 1:06 - loss: 0.4398 - categorical_accuracy: 0.8612
11680/60000 [====>.........................] - ETA: 1:06 - loss: 0.4386 - categorical_accuracy: 0.8617
11712/60000 [====>.........................] - ETA: 1:06 - loss: 0.4394 - categorical_accuracy: 0.8616
11776/60000 [====>.........................] - ETA: 1:06 - loss: 0.4388 - categorical_accuracy: 0.8619
11840/60000 [====>.........................] - ETA: 1:06 - loss: 0.4386 - categorical_accuracy: 0.8617
11904/60000 [====>.........................] - ETA: 1:06 - loss: 0.4374 - categorical_accuracy: 0.8622
11968/60000 [====>.........................] - ETA: 1:06 - loss: 0.4353 - categorical_accuracy: 0.8630
12032/60000 [=====>........................] - ETA: 1:06 - loss: 0.4334 - categorical_accuracy: 0.8636
12096/60000 [=====>........................] - ETA: 1:06 - loss: 0.4318 - categorical_accuracy: 0.8640
12160/60000 [=====>........................] - ETA: 1:06 - loss: 0.4309 - categorical_accuracy: 0.8644
12224/60000 [=====>........................] - ETA: 1:06 - loss: 0.4304 - categorical_accuracy: 0.8644
12288/60000 [=====>........................] - ETA: 1:05 - loss: 0.4286 - categorical_accuracy: 0.8651
12352/60000 [=====>........................] - ETA: 1:05 - loss: 0.4273 - categorical_accuracy: 0.8654
12416/60000 [=====>........................] - ETA: 1:05 - loss: 0.4262 - categorical_accuracy: 0.8658
12480/60000 [=====>........................] - ETA: 1:05 - loss: 0.4257 - categorical_accuracy: 0.8659
12544/60000 [=====>........................] - ETA: 1:05 - loss: 0.4243 - categorical_accuracy: 0.8664
12608/60000 [=====>........................] - ETA: 1:05 - loss: 0.4230 - categorical_accuracy: 0.8667
12672/60000 [=====>........................] - ETA: 1:05 - loss: 0.4216 - categorical_accuracy: 0.8671
12736/60000 [=====>........................] - ETA: 1:05 - loss: 0.4199 - categorical_accuracy: 0.8676
12800/60000 [=====>........................] - ETA: 1:05 - loss: 0.4198 - categorical_accuracy: 0.8677
12864/60000 [=====>........................] - ETA: 1:05 - loss: 0.4181 - categorical_accuracy: 0.8683
12928/60000 [=====>........................] - ETA: 1:05 - loss: 0.4164 - categorical_accuracy: 0.8690
12992/60000 [=====>........................] - ETA: 1:04 - loss: 0.4146 - categorical_accuracy: 0.8696
13056/60000 [=====>........................] - ETA: 1:04 - loss: 0.4133 - categorical_accuracy: 0.8700
13120/60000 [=====>........................] - ETA: 1:04 - loss: 0.4118 - categorical_accuracy: 0.8704
13184/60000 [=====>........................] - ETA: 1:04 - loss: 0.4115 - categorical_accuracy: 0.8707
13248/60000 [=====>........................] - ETA: 1:04 - loss: 0.4100 - categorical_accuracy: 0.8712
13312/60000 [=====>........................] - ETA: 1:04 - loss: 0.4087 - categorical_accuracy: 0.8714
13376/60000 [=====>........................] - ETA: 1:04 - loss: 0.4071 - categorical_accuracy: 0.8719
13440/60000 [=====>........................] - ETA: 1:04 - loss: 0.4057 - categorical_accuracy: 0.8723
13504/60000 [=====>........................] - ETA: 1:04 - loss: 0.4042 - categorical_accuracy: 0.8729
13568/60000 [=====>........................] - ETA: 1:04 - loss: 0.4033 - categorical_accuracy: 0.8730
13600/60000 [=====>........................] - ETA: 1:04 - loss: 0.4025 - categorical_accuracy: 0.8733
13664/60000 [=====>........................] - ETA: 1:04 - loss: 0.4025 - categorical_accuracy: 0.8735
13728/60000 [=====>........................] - ETA: 1:03 - loss: 0.4016 - categorical_accuracy: 0.8739
13792/60000 [=====>........................] - ETA: 1:03 - loss: 0.4004 - categorical_accuracy: 0.8741
13856/60000 [=====>........................] - ETA: 1:03 - loss: 0.3998 - categorical_accuracy: 0.8743
13920/60000 [=====>........................] - ETA: 1:03 - loss: 0.3983 - categorical_accuracy: 0.8747
13984/60000 [=====>........................] - ETA: 1:03 - loss: 0.3970 - categorical_accuracy: 0.8751
14048/60000 [======>.......................] - ETA: 1:03 - loss: 0.3956 - categorical_accuracy: 0.8756
14112/60000 [======>.......................] - ETA: 1:03 - loss: 0.3942 - categorical_accuracy: 0.8759
14176/60000 [======>.......................] - ETA: 1:03 - loss: 0.3939 - categorical_accuracy: 0.8760
14240/60000 [======>.......................] - ETA: 1:03 - loss: 0.3926 - categorical_accuracy: 0.8765
14304/60000 [======>.......................] - ETA: 1:03 - loss: 0.3913 - categorical_accuracy: 0.8768
14368/60000 [======>.......................] - ETA: 1:03 - loss: 0.3905 - categorical_accuracy: 0.8769
14432/60000 [======>.......................] - ETA: 1:02 - loss: 0.3898 - categorical_accuracy: 0.8771
14496/60000 [======>.......................] - ETA: 1:02 - loss: 0.3890 - categorical_accuracy: 0.8774
14560/60000 [======>.......................] - ETA: 1:02 - loss: 0.3879 - categorical_accuracy: 0.8776
14624/60000 [======>.......................] - ETA: 1:02 - loss: 0.3871 - categorical_accuracy: 0.8779
14688/60000 [======>.......................] - ETA: 1:02 - loss: 0.3859 - categorical_accuracy: 0.8783
14752/60000 [======>.......................] - ETA: 1:02 - loss: 0.3844 - categorical_accuracy: 0.8788
14816/60000 [======>.......................] - ETA: 1:02 - loss: 0.3833 - categorical_accuracy: 0.8791
14880/60000 [======>.......................] - ETA: 1:02 - loss: 0.3824 - categorical_accuracy: 0.8794
14944/60000 [======>.......................] - ETA: 1:02 - loss: 0.3819 - categorical_accuracy: 0.8796
15008/60000 [======>.......................] - ETA: 1:02 - loss: 0.3806 - categorical_accuracy: 0.8800
15072/60000 [======>.......................] - ETA: 1:02 - loss: 0.3799 - categorical_accuracy: 0.8804
15104/60000 [======>.......................] - ETA: 1:02 - loss: 0.3794 - categorical_accuracy: 0.8806
15168/60000 [======>.......................] - ETA: 1:01 - loss: 0.3789 - categorical_accuracy: 0.8807
15232/60000 [======>.......................] - ETA: 1:01 - loss: 0.3784 - categorical_accuracy: 0.8810
15296/60000 [======>.......................] - ETA: 1:01 - loss: 0.3778 - categorical_accuracy: 0.8811
15360/60000 [======>.......................] - ETA: 1:01 - loss: 0.3764 - categorical_accuracy: 0.8816
15424/60000 [======>.......................] - ETA: 1:01 - loss: 0.3750 - categorical_accuracy: 0.8821
15488/60000 [======>.......................] - ETA: 1:01 - loss: 0.3742 - categorical_accuracy: 0.8823
15552/60000 [======>.......................] - ETA: 1:01 - loss: 0.3732 - categorical_accuracy: 0.8826
15616/60000 [======>.......................] - ETA: 1:01 - loss: 0.3727 - categorical_accuracy: 0.8829
15680/60000 [======>.......................] - ETA: 1:01 - loss: 0.3719 - categorical_accuracy: 0.8831
15744/60000 [======>.......................] - ETA: 1:01 - loss: 0.3709 - categorical_accuracy: 0.8834
15808/60000 [======>.......................] - ETA: 1:01 - loss: 0.3702 - categorical_accuracy: 0.8836
15872/60000 [======>.......................] - ETA: 1:00 - loss: 0.3690 - categorical_accuracy: 0.8840
15936/60000 [======>.......................] - ETA: 1:00 - loss: 0.3688 - categorical_accuracy: 0.8841
16000/60000 [=======>......................] - ETA: 1:00 - loss: 0.3683 - categorical_accuracy: 0.8842
16064/60000 [=======>......................] - ETA: 1:00 - loss: 0.3671 - categorical_accuracy: 0.8846
16128/60000 [=======>......................] - ETA: 1:00 - loss: 0.3661 - categorical_accuracy: 0.8849
16192/60000 [=======>......................] - ETA: 1:00 - loss: 0.3653 - categorical_accuracy: 0.8852
16256/60000 [=======>......................] - ETA: 1:00 - loss: 0.3640 - categorical_accuracy: 0.8855
16320/60000 [=======>......................] - ETA: 1:00 - loss: 0.3637 - categorical_accuracy: 0.8857
16384/60000 [=======>......................] - ETA: 1:00 - loss: 0.3628 - categorical_accuracy: 0.8860
16448/60000 [=======>......................] - ETA: 1:00 - loss: 0.3627 - categorical_accuracy: 0.8861
16512/60000 [=======>......................] - ETA: 1:00 - loss: 0.3616 - categorical_accuracy: 0.8864
16576/60000 [=======>......................] - ETA: 59s - loss: 0.3608 - categorical_accuracy: 0.8868 
16640/60000 [=======>......................] - ETA: 59s - loss: 0.3600 - categorical_accuracy: 0.8870
16704/60000 [=======>......................] - ETA: 59s - loss: 0.3592 - categorical_accuracy: 0.8873
16768/60000 [=======>......................] - ETA: 59s - loss: 0.3583 - categorical_accuracy: 0.8876
16832/60000 [=======>......................] - ETA: 59s - loss: 0.3576 - categorical_accuracy: 0.8878
16896/60000 [=======>......................] - ETA: 59s - loss: 0.3570 - categorical_accuracy: 0.8880
16960/60000 [=======>......................] - ETA: 59s - loss: 0.3561 - categorical_accuracy: 0.8882
17024/60000 [=======>......................] - ETA: 59s - loss: 0.3552 - categorical_accuracy: 0.8884
17088/60000 [=======>......................] - ETA: 59s - loss: 0.3544 - categorical_accuracy: 0.8886
17152/60000 [=======>......................] - ETA: 59s - loss: 0.3537 - categorical_accuracy: 0.8889
17216/60000 [=======>......................] - ETA: 59s - loss: 0.3532 - categorical_accuracy: 0.8890
17280/60000 [=======>......................] - ETA: 58s - loss: 0.3529 - categorical_accuracy: 0.8891
17344/60000 [=======>......................] - ETA: 58s - loss: 0.3523 - categorical_accuracy: 0.8894
17408/60000 [=======>......................] - ETA: 58s - loss: 0.3515 - categorical_accuracy: 0.8896
17472/60000 [=======>......................] - ETA: 58s - loss: 0.3504 - categorical_accuracy: 0.8901
17536/60000 [=======>......................] - ETA: 58s - loss: 0.3504 - categorical_accuracy: 0.8901
17600/60000 [=======>......................] - ETA: 58s - loss: 0.3502 - categorical_accuracy: 0.8902
17664/60000 [=======>......................] - ETA: 58s - loss: 0.3494 - categorical_accuracy: 0.8905
17728/60000 [=======>......................] - ETA: 58s - loss: 0.3486 - categorical_accuracy: 0.8907
17792/60000 [=======>......................] - ETA: 58s - loss: 0.3476 - categorical_accuracy: 0.8910
17856/60000 [=======>......................] - ETA: 58s - loss: 0.3475 - categorical_accuracy: 0.8912
17920/60000 [=======>......................] - ETA: 57s - loss: 0.3469 - categorical_accuracy: 0.8914
17984/60000 [=======>......................] - ETA: 57s - loss: 0.3461 - categorical_accuracy: 0.8916
18048/60000 [========>.....................] - ETA: 57s - loss: 0.3451 - categorical_accuracy: 0.8920
18112/60000 [========>.....................] - ETA: 57s - loss: 0.3440 - categorical_accuracy: 0.8924
18176/60000 [========>.....................] - ETA: 57s - loss: 0.3436 - categorical_accuracy: 0.8926
18240/60000 [========>.....................] - ETA: 57s - loss: 0.3426 - categorical_accuracy: 0.8929
18304/60000 [========>.....................] - ETA: 57s - loss: 0.3419 - categorical_accuracy: 0.8931
18368/60000 [========>.....................] - ETA: 57s - loss: 0.3410 - categorical_accuracy: 0.8934
18432/60000 [========>.....................] - ETA: 57s - loss: 0.3401 - categorical_accuracy: 0.8937
18496/60000 [========>.....................] - ETA: 57s - loss: 0.3395 - categorical_accuracy: 0.8939
18560/60000 [========>.....................] - ETA: 57s - loss: 0.3389 - categorical_accuracy: 0.8940
18624/60000 [========>.....................] - ETA: 56s - loss: 0.3378 - categorical_accuracy: 0.8944
18688/60000 [========>.....................] - ETA: 56s - loss: 0.3372 - categorical_accuracy: 0.8945
18752/60000 [========>.....................] - ETA: 56s - loss: 0.3365 - categorical_accuracy: 0.8947
18816/60000 [========>.....................] - ETA: 56s - loss: 0.3357 - categorical_accuracy: 0.8950
18880/60000 [========>.....................] - ETA: 56s - loss: 0.3349 - categorical_accuracy: 0.8953
18944/60000 [========>.....................] - ETA: 56s - loss: 0.3343 - categorical_accuracy: 0.8955
19008/60000 [========>.....................] - ETA: 56s - loss: 0.3339 - categorical_accuracy: 0.8957
19072/60000 [========>.....................] - ETA: 56s - loss: 0.3336 - categorical_accuracy: 0.8958
19136/60000 [========>.....................] - ETA: 56s - loss: 0.3330 - categorical_accuracy: 0.8959
19200/60000 [========>.....................] - ETA: 56s - loss: 0.3331 - categorical_accuracy: 0.8959
19264/60000 [========>.....................] - ETA: 56s - loss: 0.3322 - categorical_accuracy: 0.8962
19328/60000 [========>.....................] - ETA: 55s - loss: 0.3319 - categorical_accuracy: 0.8964
19392/60000 [========>.....................] - ETA: 55s - loss: 0.3312 - categorical_accuracy: 0.8967
19456/60000 [========>.....................] - ETA: 55s - loss: 0.3310 - categorical_accuracy: 0.8968
19520/60000 [========>.....................] - ETA: 55s - loss: 0.3302 - categorical_accuracy: 0.8970
19584/60000 [========>.....................] - ETA: 55s - loss: 0.3299 - categorical_accuracy: 0.8972
19648/60000 [========>.....................] - ETA: 55s - loss: 0.3292 - categorical_accuracy: 0.8973
19712/60000 [========>.....................] - ETA: 55s - loss: 0.3286 - categorical_accuracy: 0.8975
19776/60000 [========>.....................] - ETA: 55s - loss: 0.3282 - categorical_accuracy: 0.8976
19840/60000 [========>.....................] - ETA: 55s - loss: 0.3275 - categorical_accuracy: 0.8978
19904/60000 [========>.....................] - ETA: 55s - loss: 0.3266 - categorical_accuracy: 0.8981
19968/60000 [========>.....................] - ETA: 55s - loss: 0.3260 - categorical_accuracy: 0.8982
20032/60000 [=========>....................] - ETA: 54s - loss: 0.3251 - categorical_accuracy: 0.8986
20096/60000 [=========>....................] - ETA: 54s - loss: 0.3245 - categorical_accuracy: 0.8987
20160/60000 [=========>....................] - ETA: 54s - loss: 0.3237 - categorical_accuracy: 0.8990
20224/60000 [=========>....................] - ETA: 54s - loss: 0.3233 - categorical_accuracy: 0.8991
20288/60000 [=========>....................] - ETA: 54s - loss: 0.3228 - categorical_accuracy: 0.8993
20352/60000 [=========>....................] - ETA: 54s - loss: 0.3219 - categorical_accuracy: 0.8995
20416/60000 [=========>....................] - ETA: 54s - loss: 0.3213 - categorical_accuracy: 0.8996
20480/60000 [=========>....................] - ETA: 54s - loss: 0.3207 - categorical_accuracy: 0.8998
20544/60000 [=========>....................] - ETA: 54s - loss: 0.3201 - categorical_accuracy: 0.9000
20608/60000 [=========>....................] - ETA: 54s - loss: 0.3193 - categorical_accuracy: 0.9003
20640/60000 [=========>....................] - ETA: 54s - loss: 0.3189 - categorical_accuracy: 0.9004
20704/60000 [=========>....................] - ETA: 54s - loss: 0.3185 - categorical_accuracy: 0.9005
20768/60000 [=========>....................] - ETA: 53s - loss: 0.3185 - categorical_accuracy: 0.9004
20832/60000 [=========>....................] - ETA: 53s - loss: 0.3183 - categorical_accuracy: 0.9006
20896/60000 [=========>....................] - ETA: 53s - loss: 0.3178 - categorical_accuracy: 0.9008
20960/60000 [=========>....................] - ETA: 53s - loss: 0.3170 - categorical_accuracy: 0.9011
21024/60000 [=========>....................] - ETA: 53s - loss: 0.3163 - categorical_accuracy: 0.9014
21088/60000 [=========>....................] - ETA: 53s - loss: 0.3159 - categorical_accuracy: 0.9015
21152/60000 [=========>....................] - ETA: 53s - loss: 0.3153 - categorical_accuracy: 0.9017
21216/60000 [=========>....................] - ETA: 53s - loss: 0.3150 - categorical_accuracy: 0.9017
21280/60000 [=========>....................] - ETA: 53s - loss: 0.3145 - categorical_accuracy: 0.9019
21344/60000 [=========>....................] - ETA: 53s - loss: 0.3143 - categorical_accuracy: 0.9020
21408/60000 [=========>....................] - ETA: 52s - loss: 0.3136 - categorical_accuracy: 0.9022
21472/60000 [=========>....................] - ETA: 52s - loss: 0.3132 - categorical_accuracy: 0.9023
21536/60000 [=========>....................] - ETA: 52s - loss: 0.3127 - categorical_accuracy: 0.9025
21600/60000 [=========>....................] - ETA: 52s - loss: 0.3119 - categorical_accuracy: 0.9028
21664/60000 [=========>....................] - ETA: 52s - loss: 0.3110 - categorical_accuracy: 0.9031
21728/60000 [=========>....................] - ETA: 52s - loss: 0.3103 - categorical_accuracy: 0.9032
21792/60000 [=========>....................] - ETA: 52s - loss: 0.3100 - categorical_accuracy: 0.9033
21856/60000 [=========>....................] - ETA: 52s - loss: 0.3098 - categorical_accuracy: 0.9035
21920/60000 [=========>....................] - ETA: 52s - loss: 0.3093 - categorical_accuracy: 0.9036
21984/60000 [=========>....................] - ETA: 52s - loss: 0.3088 - categorical_accuracy: 0.9037
22048/60000 [==========>...................] - ETA: 52s - loss: 0.3092 - categorical_accuracy: 0.9037
22080/60000 [==========>...................] - ETA: 52s - loss: 0.3088 - categorical_accuracy: 0.9038
22144/60000 [==========>...................] - ETA: 51s - loss: 0.3084 - categorical_accuracy: 0.9040
22208/60000 [==========>...................] - ETA: 51s - loss: 0.3078 - categorical_accuracy: 0.9041
22272/60000 [==========>...................] - ETA: 51s - loss: 0.3077 - categorical_accuracy: 0.9042
22336/60000 [==========>...................] - ETA: 51s - loss: 0.3069 - categorical_accuracy: 0.9045
22400/60000 [==========>...................] - ETA: 51s - loss: 0.3063 - categorical_accuracy: 0.9047
22464/60000 [==========>...................] - ETA: 51s - loss: 0.3057 - categorical_accuracy: 0.9049
22528/60000 [==========>...................] - ETA: 51s - loss: 0.3054 - categorical_accuracy: 0.9050
22592/60000 [==========>...................] - ETA: 51s - loss: 0.3047 - categorical_accuracy: 0.9051
22656/60000 [==========>...................] - ETA: 51s - loss: 0.3040 - categorical_accuracy: 0.9054
22720/60000 [==========>...................] - ETA: 51s - loss: 0.3035 - categorical_accuracy: 0.9055
22784/60000 [==========>...................] - ETA: 51s - loss: 0.3031 - categorical_accuracy: 0.9057
22848/60000 [==========>...................] - ETA: 50s - loss: 0.3024 - categorical_accuracy: 0.9059
22912/60000 [==========>...................] - ETA: 50s - loss: 0.3024 - categorical_accuracy: 0.9059
22976/60000 [==========>...................] - ETA: 50s - loss: 0.3020 - categorical_accuracy: 0.9060
23040/60000 [==========>...................] - ETA: 50s - loss: 0.3020 - categorical_accuracy: 0.9061
23104/60000 [==========>...................] - ETA: 50s - loss: 0.3013 - categorical_accuracy: 0.9063
23168/60000 [==========>...................] - ETA: 50s - loss: 0.3008 - categorical_accuracy: 0.9063
23232/60000 [==========>...................] - ETA: 50s - loss: 0.3003 - categorical_accuracy: 0.9065
23296/60000 [==========>...................] - ETA: 50s - loss: 0.2998 - categorical_accuracy: 0.9066
23360/60000 [==========>...................] - ETA: 50s - loss: 0.2994 - categorical_accuracy: 0.9068
23424/60000 [==========>...................] - ETA: 50s - loss: 0.2995 - categorical_accuracy: 0.9069
23488/60000 [==========>...................] - ETA: 50s - loss: 0.2994 - categorical_accuracy: 0.9070
23552/60000 [==========>...................] - ETA: 49s - loss: 0.2991 - categorical_accuracy: 0.9071
23616/60000 [==========>...................] - ETA: 49s - loss: 0.2987 - categorical_accuracy: 0.9073
23680/60000 [==========>...................] - ETA: 49s - loss: 0.2988 - categorical_accuracy: 0.9073
23744/60000 [==========>...................] - ETA: 49s - loss: 0.2984 - categorical_accuracy: 0.9074
23808/60000 [==========>...................] - ETA: 49s - loss: 0.2981 - categorical_accuracy: 0.9076
23872/60000 [==========>...................] - ETA: 49s - loss: 0.2976 - categorical_accuracy: 0.9077
23936/60000 [==========>...................] - ETA: 49s - loss: 0.2975 - categorical_accuracy: 0.9078
24000/60000 [===========>..................] - ETA: 49s - loss: 0.2972 - categorical_accuracy: 0.9080
24064/60000 [===========>..................] - ETA: 49s - loss: 0.2970 - categorical_accuracy: 0.9080
24128/60000 [===========>..................] - ETA: 49s - loss: 0.2964 - categorical_accuracy: 0.9082
24192/60000 [===========>..................] - ETA: 49s - loss: 0.2960 - categorical_accuracy: 0.9083
24256/60000 [===========>..................] - ETA: 48s - loss: 0.2954 - categorical_accuracy: 0.9084
24320/60000 [===========>..................] - ETA: 48s - loss: 0.2949 - categorical_accuracy: 0.9086
24384/60000 [===========>..................] - ETA: 48s - loss: 0.2945 - categorical_accuracy: 0.9088
24448/60000 [===========>..................] - ETA: 48s - loss: 0.2943 - categorical_accuracy: 0.9088
24512/60000 [===========>..................] - ETA: 48s - loss: 0.2940 - categorical_accuracy: 0.9089
24576/60000 [===========>..................] - ETA: 48s - loss: 0.2935 - categorical_accuracy: 0.9090
24640/60000 [===========>..................] - ETA: 48s - loss: 0.2932 - categorical_accuracy: 0.9091
24704/60000 [===========>..................] - ETA: 48s - loss: 0.2926 - categorical_accuracy: 0.9093
24768/60000 [===========>..................] - ETA: 48s - loss: 0.2921 - categorical_accuracy: 0.9094
24832/60000 [===========>..................] - ETA: 48s - loss: 0.2915 - categorical_accuracy: 0.9096
24896/60000 [===========>..................] - ETA: 48s - loss: 0.2909 - categorical_accuracy: 0.9098
24960/60000 [===========>..................] - ETA: 47s - loss: 0.2904 - categorical_accuracy: 0.9100
25024/60000 [===========>..................] - ETA: 47s - loss: 0.2902 - categorical_accuracy: 0.9100
25088/60000 [===========>..................] - ETA: 47s - loss: 0.2896 - categorical_accuracy: 0.9102
25152/60000 [===========>..................] - ETA: 47s - loss: 0.2889 - categorical_accuracy: 0.9104
25184/60000 [===========>..................] - ETA: 47s - loss: 0.2886 - categorical_accuracy: 0.9105
25248/60000 [===========>..................] - ETA: 47s - loss: 0.2882 - categorical_accuracy: 0.9107
25312/60000 [===========>..................] - ETA: 47s - loss: 0.2877 - categorical_accuracy: 0.9109
25376/60000 [===========>..................] - ETA: 47s - loss: 0.2871 - categorical_accuracy: 0.9111
25440/60000 [===========>..................] - ETA: 47s - loss: 0.2865 - categorical_accuracy: 0.9113
25504/60000 [===========>..................] - ETA: 47s - loss: 0.2865 - categorical_accuracy: 0.9114
25568/60000 [===========>..................] - ETA: 47s - loss: 0.2858 - categorical_accuracy: 0.9116
25632/60000 [===========>..................] - ETA: 47s - loss: 0.2857 - categorical_accuracy: 0.9117
25696/60000 [===========>..................] - ETA: 47s - loss: 0.2852 - categorical_accuracy: 0.9118
25760/60000 [===========>..................] - ETA: 46s - loss: 0.2846 - categorical_accuracy: 0.9120
25824/60000 [===========>..................] - ETA: 46s - loss: 0.2841 - categorical_accuracy: 0.9121
25888/60000 [===========>..................] - ETA: 46s - loss: 0.2841 - categorical_accuracy: 0.9122
25952/60000 [===========>..................] - ETA: 46s - loss: 0.2837 - categorical_accuracy: 0.9123
26016/60000 [============>.................] - ETA: 46s - loss: 0.2832 - categorical_accuracy: 0.9125
26080/60000 [============>.................] - ETA: 46s - loss: 0.2829 - categorical_accuracy: 0.9126
26144/60000 [============>.................] - ETA: 46s - loss: 0.2826 - categorical_accuracy: 0.9126
26208/60000 [============>.................] - ETA: 46s - loss: 0.2821 - categorical_accuracy: 0.9128
26272/60000 [============>.................] - ETA: 46s - loss: 0.2823 - categorical_accuracy: 0.9129
26336/60000 [============>.................] - ETA: 46s - loss: 0.2820 - categorical_accuracy: 0.9130
26368/60000 [============>.................] - ETA: 46s - loss: 0.2819 - categorical_accuracy: 0.9130
26400/60000 [============>.................] - ETA: 46s - loss: 0.2817 - categorical_accuracy: 0.9131
26464/60000 [============>.................] - ETA: 45s - loss: 0.2812 - categorical_accuracy: 0.9132
26528/60000 [============>.................] - ETA: 45s - loss: 0.2807 - categorical_accuracy: 0.9134
26592/60000 [============>.................] - ETA: 45s - loss: 0.2805 - categorical_accuracy: 0.9134
26656/60000 [============>.................] - ETA: 45s - loss: 0.2800 - categorical_accuracy: 0.9135
26720/60000 [============>.................] - ETA: 45s - loss: 0.2797 - categorical_accuracy: 0.9136
26784/60000 [============>.................] - ETA: 45s - loss: 0.2794 - categorical_accuracy: 0.9137
26848/60000 [============>.................] - ETA: 45s - loss: 0.2791 - categorical_accuracy: 0.9138
26912/60000 [============>.................] - ETA: 45s - loss: 0.2792 - categorical_accuracy: 0.9137
26976/60000 [============>.................] - ETA: 45s - loss: 0.2787 - categorical_accuracy: 0.9139
27040/60000 [============>.................] - ETA: 45s - loss: 0.2783 - categorical_accuracy: 0.9140
27104/60000 [============>.................] - ETA: 45s - loss: 0.2777 - categorical_accuracy: 0.9142
27168/60000 [============>.................] - ETA: 44s - loss: 0.2774 - categorical_accuracy: 0.9143
27232/60000 [============>.................] - ETA: 44s - loss: 0.2769 - categorical_accuracy: 0.9145
27296/60000 [============>.................] - ETA: 44s - loss: 0.2764 - categorical_accuracy: 0.9146
27360/60000 [============>.................] - ETA: 44s - loss: 0.2763 - categorical_accuracy: 0.9147
27424/60000 [============>.................] - ETA: 44s - loss: 0.2758 - categorical_accuracy: 0.9148
27488/60000 [============>.................] - ETA: 44s - loss: 0.2753 - categorical_accuracy: 0.9150
27552/60000 [============>.................] - ETA: 44s - loss: 0.2748 - categorical_accuracy: 0.9151
27616/60000 [============>.................] - ETA: 44s - loss: 0.2747 - categorical_accuracy: 0.9152
27680/60000 [============>.................] - ETA: 44s - loss: 0.2742 - categorical_accuracy: 0.9154
27744/60000 [============>.................] - ETA: 44s - loss: 0.2739 - categorical_accuracy: 0.9155
27808/60000 [============>.................] - ETA: 44s - loss: 0.2735 - categorical_accuracy: 0.9156
27872/60000 [============>.................] - ETA: 44s - loss: 0.2730 - categorical_accuracy: 0.9158
27936/60000 [============>.................] - ETA: 43s - loss: 0.2727 - categorical_accuracy: 0.9159
28000/60000 [=============>................] - ETA: 43s - loss: 0.2723 - categorical_accuracy: 0.9160
28064/60000 [=============>................] - ETA: 43s - loss: 0.2724 - categorical_accuracy: 0.9160
28128/60000 [=============>................] - ETA: 43s - loss: 0.2720 - categorical_accuracy: 0.9161
28192/60000 [=============>................] - ETA: 43s - loss: 0.2717 - categorical_accuracy: 0.9163
28256/60000 [=============>................] - ETA: 43s - loss: 0.2713 - categorical_accuracy: 0.9164
28320/60000 [=============>................] - ETA: 43s - loss: 0.2710 - categorical_accuracy: 0.9165
28384/60000 [=============>................] - ETA: 43s - loss: 0.2707 - categorical_accuracy: 0.9166
28448/60000 [=============>................] - ETA: 43s - loss: 0.2704 - categorical_accuracy: 0.9167
28512/60000 [=============>................] - ETA: 43s - loss: 0.2700 - categorical_accuracy: 0.9168
28576/60000 [=============>................] - ETA: 43s - loss: 0.2697 - categorical_accuracy: 0.9168
28640/60000 [=============>................] - ETA: 42s - loss: 0.2692 - categorical_accuracy: 0.9170
28704/60000 [=============>................] - ETA: 42s - loss: 0.2689 - categorical_accuracy: 0.9170
28768/60000 [=============>................] - ETA: 42s - loss: 0.2688 - categorical_accuracy: 0.9170
28832/60000 [=============>................] - ETA: 42s - loss: 0.2684 - categorical_accuracy: 0.9171
28864/60000 [=============>................] - ETA: 42s - loss: 0.2682 - categorical_accuracy: 0.9172
28928/60000 [=============>................] - ETA: 42s - loss: 0.2678 - categorical_accuracy: 0.9174
28992/60000 [=============>................] - ETA: 42s - loss: 0.2675 - categorical_accuracy: 0.9175
29056/60000 [=============>................] - ETA: 42s - loss: 0.2674 - categorical_accuracy: 0.9176
29120/60000 [=============>................] - ETA: 42s - loss: 0.2670 - categorical_accuracy: 0.9177
29184/60000 [=============>................] - ETA: 42s - loss: 0.2667 - categorical_accuracy: 0.9178
29248/60000 [=============>................] - ETA: 42s - loss: 0.2664 - categorical_accuracy: 0.9178
29312/60000 [=============>................] - ETA: 41s - loss: 0.2659 - categorical_accuracy: 0.9180
29376/60000 [=============>................] - ETA: 41s - loss: 0.2660 - categorical_accuracy: 0.9180
29440/60000 [=============>................] - ETA: 41s - loss: 0.2657 - categorical_accuracy: 0.9180
29504/60000 [=============>................] - ETA: 41s - loss: 0.2653 - categorical_accuracy: 0.9182
29568/60000 [=============>................] - ETA: 41s - loss: 0.2650 - categorical_accuracy: 0.9182
29632/60000 [=============>................] - ETA: 41s - loss: 0.2647 - categorical_accuracy: 0.9183
29696/60000 [=============>................] - ETA: 41s - loss: 0.2642 - categorical_accuracy: 0.9185
29760/60000 [=============>................] - ETA: 41s - loss: 0.2642 - categorical_accuracy: 0.9186
29824/60000 [=============>................] - ETA: 41s - loss: 0.2638 - categorical_accuracy: 0.9187
29888/60000 [=============>................] - ETA: 41s - loss: 0.2635 - categorical_accuracy: 0.9188
29920/60000 [=============>................] - ETA: 41s - loss: 0.2634 - categorical_accuracy: 0.9188
29984/60000 [=============>................] - ETA: 41s - loss: 0.2628 - categorical_accuracy: 0.9190
30048/60000 [==============>...............] - ETA: 40s - loss: 0.2624 - categorical_accuracy: 0.9191
30112/60000 [==============>...............] - ETA: 40s - loss: 0.2622 - categorical_accuracy: 0.9191
30176/60000 [==============>...............] - ETA: 40s - loss: 0.2619 - categorical_accuracy: 0.9192
30240/60000 [==============>...............] - ETA: 40s - loss: 0.2617 - categorical_accuracy: 0.9193
30304/60000 [==============>...............] - ETA: 40s - loss: 0.2613 - categorical_accuracy: 0.9194
30368/60000 [==============>...............] - ETA: 40s - loss: 0.2610 - categorical_accuracy: 0.9196
30432/60000 [==============>...............] - ETA: 40s - loss: 0.2608 - categorical_accuracy: 0.9197
30496/60000 [==============>...............] - ETA: 40s - loss: 0.2603 - categorical_accuracy: 0.9198
30560/60000 [==============>...............] - ETA: 40s - loss: 0.2600 - categorical_accuracy: 0.9199
30624/60000 [==============>...............] - ETA: 40s - loss: 0.2596 - categorical_accuracy: 0.9200
30688/60000 [==============>...............] - ETA: 40s - loss: 0.2593 - categorical_accuracy: 0.9201
30752/60000 [==============>...............] - ETA: 39s - loss: 0.2594 - categorical_accuracy: 0.9202
30816/60000 [==============>...............] - ETA: 39s - loss: 0.2590 - categorical_accuracy: 0.9202
30880/60000 [==============>...............] - ETA: 39s - loss: 0.2590 - categorical_accuracy: 0.9202
30944/60000 [==============>...............] - ETA: 39s - loss: 0.2591 - categorical_accuracy: 0.9203
31008/60000 [==============>...............] - ETA: 39s - loss: 0.2590 - categorical_accuracy: 0.9204
31072/60000 [==============>...............] - ETA: 39s - loss: 0.2586 - categorical_accuracy: 0.9205
31136/60000 [==============>...............] - ETA: 39s - loss: 0.2584 - categorical_accuracy: 0.9205
31200/60000 [==============>...............] - ETA: 39s - loss: 0.2581 - categorical_accuracy: 0.9206
31264/60000 [==============>...............] - ETA: 39s - loss: 0.2581 - categorical_accuracy: 0.9207
31328/60000 [==============>...............] - ETA: 39s - loss: 0.2577 - categorical_accuracy: 0.9208
31392/60000 [==============>...............] - ETA: 39s - loss: 0.2574 - categorical_accuracy: 0.9209
31456/60000 [==============>...............] - ETA: 39s - loss: 0.2573 - categorical_accuracy: 0.9210
31520/60000 [==============>...............] - ETA: 38s - loss: 0.2570 - categorical_accuracy: 0.9211
31584/60000 [==============>...............] - ETA: 38s - loss: 0.2569 - categorical_accuracy: 0.9211
31648/60000 [==============>...............] - ETA: 38s - loss: 0.2566 - categorical_accuracy: 0.9212
31712/60000 [==============>...............] - ETA: 38s - loss: 0.2562 - categorical_accuracy: 0.9213
31776/60000 [==============>...............] - ETA: 38s - loss: 0.2562 - categorical_accuracy: 0.9213
31840/60000 [==============>...............] - ETA: 38s - loss: 0.2561 - categorical_accuracy: 0.9214
31904/60000 [==============>...............] - ETA: 38s - loss: 0.2558 - categorical_accuracy: 0.9214
31968/60000 [==============>...............] - ETA: 38s - loss: 0.2554 - categorical_accuracy: 0.9215
32032/60000 [===============>..............] - ETA: 38s - loss: 0.2550 - categorical_accuracy: 0.9216
32096/60000 [===============>..............] - ETA: 38s - loss: 0.2545 - categorical_accuracy: 0.9218
32160/60000 [===============>..............] - ETA: 38s - loss: 0.2543 - categorical_accuracy: 0.9218
32192/60000 [===============>..............] - ETA: 38s - loss: 0.2541 - categorical_accuracy: 0.9219
32256/60000 [===============>..............] - ETA: 37s - loss: 0.2538 - categorical_accuracy: 0.9220
32320/60000 [===============>..............] - ETA: 37s - loss: 0.2534 - categorical_accuracy: 0.9222
32384/60000 [===============>..............] - ETA: 37s - loss: 0.2530 - categorical_accuracy: 0.9222
32448/60000 [===============>..............] - ETA: 37s - loss: 0.2528 - categorical_accuracy: 0.9223
32512/60000 [===============>..............] - ETA: 37s - loss: 0.2525 - categorical_accuracy: 0.9223
32576/60000 [===============>..............] - ETA: 37s - loss: 0.2525 - categorical_accuracy: 0.9224
32640/60000 [===============>..............] - ETA: 37s - loss: 0.2525 - categorical_accuracy: 0.9225
32704/60000 [===============>..............] - ETA: 37s - loss: 0.2522 - categorical_accuracy: 0.9225
32768/60000 [===============>..............] - ETA: 37s - loss: 0.2519 - categorical_accuracy: 0.9226
32832/60000 [===============>..............] - ETA: 37s - loss: 0.2515 - categorical_accuracy: 0.9228
32896/60000 [===============>..............] - ETA: 37s - loss: 0.2513 - categorical_accuracy: 0.9228
32960/60000 [===============>..............] - ETA: 36s - loss: 0.2512 - categorical_accuracy: 0.9229
33024/60000 [===============>..............] - ETA: 36s - loss: 0.2509 - categorical_accuracy: 0.9230
33088/60000 [===============>..............] - ETA: 36s - loss: 0.2507 - categorical_accuracy: 0.9231
33152/60000 [===============>..............] - ETA: 36s - loss: 0.2504 - categorical_accuracy: 0.9232
33216/60000 [===============>..............] - ETA: 36s - loss: 0.2502 - categorical_accuracy: 0.9233
33280/60000 [===============>..............] - ETA: 36s - loss: 0.2500 - categorical_accuracy: 0.9233
33344/60000 [===============>..............] - ETA: 36s - loss: 0.2496 - categorical_accuracy: 0.9234
33408/60000 [===============>..............] - ETA: 36s - loss: 0.2493 - categorical_accuracy: 0.9235
33472/60000 [===============>..............] - ETA: 36s - loss: 0.2489 - categorical_accuracy: 0.9236
33536/60000 [===============>..............] - ETA: 36s - loss: 0.2488 - categorical_accuracy: 0.9236
33600/60000 [===============>..............] - ETA: 36s - loss: 0.2486 - categorical_accuracy: 0.9237
33664/60000 [===============>..............] - ETA: 35s - loss: 0.2482 - categorical_accuracy: 0.9238
33728/60000 [===============>..............] - ETA: 35s - loss: 0.2480 - categorical_accuracy: 0.9239
33792/60000 [===============>..............] - ETA: 35s - loss: 0.2478 - categorical_accuracy: 0.9239
33856/60000 [===============>..............] - ETA: 35s - loss: 0.2476 - categorical_accuracy: 0.9240
33920/60000 [===============>..............] - ETA: 35s - loss: 0.2475 - categorical_accuracy: 0.9241
33984/60000 [===============>..............] - ETA: 35s - loss: 0.2473 - categorical_accuracy: 0.9242
34048/60000 [================>.............] - ETA: 35s - loss: 0.2471 - categorical_accuracy: 0.9242
34112/60000 [================>.............] - ETA: 35s - loss: 0.2469 - categorical_accuracy: 0.9242
34176/60000 [================>.............] - ETA: 35s - loss: 0.2469 - categorical_accuracy: 0.9243
34240/60000 [================>.............] - ETA: 35s - loss: 0.2465 - categorical_accuracy: 0.9244
34304/60000 [================>.............] - ETA: 35s - loss: 0.2463 - categorical_accuracy: 0.9245
34368/60000 [================>.............] - ETA: 34s - loss: 0.2461 - categorical_accuracy: 0.9245
34432/60000 [================>.............] - ETA: 34s - loss: 0.2459 - categorical_accuracy: 0.9245
34496/60000 [================>.............] - ETA: 34s - loss: 0.2456 - categorical_accuracy: 0.9246
34560/60000 [================>.............] - ETA: 34s - loss: 0.2452 - categorical_accuracy: 0.9248
34624/60000 [================>.............] - ETA: 34s - loss: 0.2450 - categorical_accuracy: 0.9248
34688/60000 [================>.............] - ETA: 34s - loss: 0.2448 - categorical_accuracy: 0.9249
34752/60000 [================>.............] - ETA: 34s - loss: 0.2447 - categorical_accuracy: 0.9250
34816/60000 [================>.............] - ETA: 34s - loss: 0.2446 - categorical_accuracy: 0.9249
34880/60000 [================>.............] - ETA: 34s - loss: 0.2443 - categorical_accuracy: 0.9250
34944/60000 [================>.............] - ETA: 34s - loss: 0.2441 - categorical_accuracy: 0.9250
35008/60000 [================>.............] - ETA: 34s - loss: 0.2438 - categorical_accuracy: 0.9250
35072/60000 [================>.............] - ETA: 34s - loss: 0.2436 - categorical_accuracy: 0.9251
35136/60000 [================>.............] - ETA: 33s - loss: 0.2435 - categorical_accuracy: 0.9251
35200/60000 [================>.............] - ETA: 33s - loss: 0.2432 - categorical_accuracy: 0.9253
35264/60000 [================>.............] - ETA: 33s - loss: 0.2432 - categorical_accuracy: 0.9253
35328/60000 [================>.............] - ETA: 33s - loss: 0.2430 - categorical_accuracy: 0.9253
35392/60000 [================>.............] - ETA: 33s - loss: 0.2426 - categorical_accuracy: 0.9254
35456/60000 [================>.............] - ETA: 33s - loss: 0.2422 - categorical_accuracy: 0.9255
35520/60000 [================>.............] - ETA: 33s - loss: 0.2422 - categorical_accuracy: 0.9256
35584/60000 [================>.............] - ETA: 33s - loss: 0.2419 - categorical_accuracy: 0.9257
35648/60000 [================>.............] - ETA: 33s - loss: 0.2416 - categorical_accuracy: 0.9258
35712/60000 [================>.............] - ETA: 33s - loss: 0.2414 - categorical_accuracy: 0.9258
35776/60000 [================>.............] - ETA: 33s - loss: 0.2411 - categorical_accuracy: 0.9259
35840/60000 [================>.............] - ETA: 32s - loss: 0.2407 - categorical_accuracy: 0.9261
35904/60000 [================>.............] - ETA: 32s - loss: 0.2406 - categorical_accuracy: 0.9261
35968/60000 [================>.............] - ETA: 32s - loss: 0.2402 - categorical_accuracy: 0.9262
36032/60000 [=================>............] - ETA: 32s - loss: 0.2402 - categorical_accuracy: 0.9262
36096/60000 [=================>............] - ETA: 32s - loss: 0.2400 - categorical_accuracy: 0.9263
36160/60000 [=================>............] - ETA: 32s - loss: 0.2398 - categorical_accuracy: 0.9264
36224/60000 [=================>............] - ETA: 32s - loss: 0.2397 - categorical_accuracy: 0.9264
36288/60000 [=================>............] - ETA: 32s - loss: 0.2394 - categorical_accuracy: 0.9265
36352/60000 [=================>............] - ETA: 32s - loss: 0.2391 - categorical_accuracy: 0.9266
36416/60000 [=================>............] - ETA: 32s - loss: 0.2391 - categorical_accuracy: 0.9267
36480/60000 [=================>............] - ETA: 32s - loss: 0.2388 - categorical_accuracy: 0.9267
36544/60000 [=================>............] - ETA: 32s - loss: 0.2385 - categorical_accuracy: 0.9269
36608/60000 [=================>............] - ETA: 31s - loss: 0.2383 - categorical_accuracy: 0.9269
36672/60000 [=================>............] - ETA: 31s - loss: 0.2380 - categorical_accuracy: 0.9270
36736/60000 [=================>............] - ETA: 31s - loss: 0.2377 - categorical_accuracy: 0.9271
36800/60000 [=================>............] - ETA: 31s - loss: 0.2374 - categorical_accuracy: 0.9272
36864/60000 [=================>............] - ETA: 31s - loss: 0.2372 - categorical_accuracy: 0.9272
36928/60000 [=================>............] - ETA: 31s - loss: 0.2373 - categorical_accuracy: 0.9272
36992/60000 [=================>............] - ETA: 31s - loss: 0.2370 - categorical_accuracy: 0.9273
37056/60000 [=================>............] - ETA: 31s - loss: 0.2367 - categorical_accuracy: 0.9274
37120/60000 [=================>............] - ETA: 31s - loss: 0.2363 - categorical_accuracy: 0.9275
37184/60000 [=================>............] - ETA: 31s - loss: 0.2360 - categorical_accuracy: 0.9276
37248/60000 [=================>............] - ETA: 31s - loss: 0.2357 - categorical_accuracy: 0.9276
37312/60000 [=================>............] - ETA: 30s - loss: 0.2354 - categorical_accuracy: 0.9277
37376/60000 [=================>............] - ETA: 30s - loss: 0.2350 - categorical_accuracy: 0.9279
37440/60000 [=================>............] - ETA: 30s - loss: 0.2349 - categorical_accuracy: 0.9279
37472/60000 [=================>............] - ETA: 30s - loss: 0.2348 - categorical_accuracy: 0.9280
37536/60000 [=================>............] - ETA: 30s - loss: 0.2347 - categorical_accuracy: 0.9280
37600/60000 [=================>............] - ETA: 30s - loss: 0.2346 - categorical_accuracy: 0.9280
37664/60000 [=================>............] - ETA: 30s - loss: 0.2345 - categorical_accuracy: 0.9281
37728/60000 [=================>............] - ETA: 30s - loss: 0.2343 - categorical_accuracy: 0.9281
37792/60000 [=================>............] - ETA: 30s - loss: 0.2339 - categorical_accuracy: 0.9283
37856/60000 [=================>............] - ETA: 30s - loss: 0.2336 - categorical_accuracy: 0.9284
37920/60000 [=================>............] - ETA: 30s - loss: 0.2334 - categorical_accuracy: 0.9285
37984/60000 [=================>............] - ETA: 30s - loss: 0.2330 - categorical_accuracy: 0.9286
38048/60000 [==================>...........] - ETA: 29s - loss: 0.2327 - categorical_accuracy: 0.9286
38112/60000 [==================>...........] - ETA: 29s - loss: 0.2325 - categorical_accuracy: 0.9287
38176/60000 [==================>...........] - ETA: 29s - loss: 0.2323 - categorical_accuracy: 0.9288
38240/60000 [==================>...........] - ETA: 29s - loss: 0.2320 - categorical_accuracy: 0.9289
38304/60000 [==================>...........] - ETA: 29s - loss: 0.2323 - categorical_accuracy: 0.9289
38368/60000 [==================>...........] - ETA: 29s - loss: 0.2320 - categorical_accuracy: 0.9290
38432/60000 [==================>...........] - ETA: 29s - loss: 0.2318 - categorical_accuracy: 0.9291
38496/60000 [==================>...........] - ETA: 29s - loss: 0.2317 - categorical_accuracy: 0.9291
38560/60000 [==================>...........] - ETA: 29s - loss: 0.2315 - categorical_accuracy: 0.9292
38624/60000 [==================>...........] - ETA: 29s - loss: 0.2312 - categorical_accuracy: 0.9293
38688/60000 [==================>...........] - ETA: 29s - loss: 0.2309 - categorical_accuracy: 0.9294
38752/60000 [==================>...........] - ETA: 28s - loss: 0.2309 - categorical_accuracy: 0.9294
38816/60000 [==================>...........] - ETA: 28s - loss: 0.2308 - categorical_accuracy: 0.9294
38880/60000 [==================>...........] - ETA: 28s - loss: 0.2307 - categorical_accuracy: 0.9295
38944/60000 [==================>...........] - ETA: 28s - loss: 0.2304 - categorical_accuracy: 0.9295
39008/60000 [==================>...........] - ETA: 28s - loss: 0.2304 - categorical_accuracy: 0.9296
39072/60000 [==================>...........] - ETA: 28s - loss: 0.2304 - categorical_accuracy: 0.9295
39136/60000 [==================>...........] - ETA: 28s - loss: 0.2303 - categorical_accuracy: 0.9295
39200/60000 [==================>...........] - ETA: 28s - loss: 0.2301 - categorical_accuracy: 0.9296
39264/60000 [==================>...........] - ETA: 28s - loss: 0.2299 - categorical_accuracy: 0.9297
39328/60000 [==================>...........] - ETA: 28s - loss: 0.2296 - categorical_accuracy: 0.9297
39392/60000 [==================>...........] - ETA: 28s - loss: 0.2293 - categorical_accuracy: 0.9298
39456/60000 [==================>...........] - ETA: 28s - loss: 0.2293 - categorical_accuracy: 0.9298
39520/60000 [==================>...........] - ETA: 27s - loss: 0.2291 - categorical_accuracy: 0.9299
39584/60000 [==================>...........] - ETA: 27s - loss: 0.2289 - categorical_accuracy: 0.9299
39648/60000 [==================>...........] - ETA: 27s - loss: 0.2286 - categorical_accuracy: 0.9300
39712/60000 [==================>...........] - ETA: 27s - loss: 0.2284 - categorical_accuracy: 0.9301
39776/60000 [==================>...........] - ETA: 27s - loss: 0.2282 - categorical_accuracy: 0.9302
39840/60000 [==================>...........] - ETA: 27s - loss: 0.2279 - categorical_accuracy: 0.9303
39904/60000 [==================>...........] - ETA: 27s - loss: 0.2277 - categorical_accuracy: 0.9303
39968/60000 [==================>...........] - ETA: 27s - loss: 0.2274 - categorical_accuracy: 0.9304
40032/60000 [===================>..........] - ETA: 27s - loss: 0.2271 - categorical_accuracy: 0.9305
40096/60000 [===================>..........] - ETA: 27s - loss: 0.2270 - categorical_accuracy: 0.9305
40160/60000 [===================>..........] - ETA: 27s - loss: 0.2267 - categorical_accuracy: 0.9306
40224/60000 [===================>..........] - ETA: 26s - loss: 0.2266 - categorical_accuracy: 0.9306
40288/60000 [===================>..........] - ETA: 26s - loss: 0.2267 - categorical_accuracy: 0.9305
40352/60000 [===================>..........] - ETA: 26s - loss: 0.2265 - categorical_accuracy: 0.9306
40416/60000 [===================>..........] - ETA: 26s - loss: 0.2263 - categorical_accuracy: 0.9306
40480/60000 [===================>..........] - ETA: 26s - loss: 0.2261 - categorical_accuracy: 0.9307
40544/60000 [===================>..........] - ETA: 26s - loss: 0.2258 - categorical_accuracy: 0.9308
40608/60000 [===================>..........] - ETA: 26s - loss: 0.2257 - categorical_accuracy: 0.9308
40672/60000 [===================>..........] - ETA: 26s - loss: 0.2256 - categorical_accuracy: 0.9308
40736/60000 [===================>..........] - ETA: 26s - loss: 0.2255 - categorical_accuracy: 0.9309
40800/60000 [===================>..........] - ETA: 26s - loss: 0.2254 - categorical_accuracy: 0.9309
40864/60000 [===================>..........] - ETA: 26s - loss: 0.2252 - categorical_accuracy: 0.9309
40928/60000 [===================>..........] - ETA: 26s - loss: 0.2249 - categorical_accuracy: 0.9310
40992/60000 [===================>..........] - ETA: 25s - loss: 0.2248 - categorical_accuracy: 0.9311
41056/60000 [===================>..........] - ETA: 25s - loss: 0.2248 - categorical_accuracy: 0.9311
41120/60000 [===================>..........] - ETA: 25s - loss: 0.2247 - categorical_accuracy: 0.9311
41184/60000 [===================>..........] - ETA: 25s - loss: 0.2246 - categorical_accuracy: 0.9312
41248/60000 [===================>..........] - ETA: 25s - loss: 0.2244 - categorical_accuracy: 0.9312
41312/60000 [===================>..........] - ETA: 25s - loss: 0.2241 - categorical_accuracy: 0.9314
41376/60000 [===================>..........] - ETA: 25s - loss: 0.2238 - categorical_accuracy: 0.9314
41440/60000 [===================>..........] - ETA: 25s - loss: 0.2237 - categorical_accuracy: 0.9314
41504/60000 [===================>..........] - ETA: 25s - loss: 0.2236 - categorical_accuracy: 0.9315
41568/60000 [===================>..........] - ETA: 25s - loss: 0.2233 - categorical_accuracy: 0.9315
41632/60000 [===================>..........] - ETA: 25s - loss: 0.2232 - categorical_accuracy: 0.9316
41696/60000 [===================>..........] - ETA: 24s - loss: 0.2235 - categorical_accuracy: 0.9315
41760/60000 [===================>..........] - ETA: 24s - loss: 0.2232 - categorical_accuracy: 0.9316
41824/60000 [===================>..........] - ETA: 24s - loss: 0.2230 - categorical_accuracy: 0.9317
41888/60000 [===================>..........] - ETA: 24s - loss: 0.2228 - categorical_accuracy: 0.9318
41952/60000 [===================>..........] - ETA: 24s - loss: 0.2226 - categorical_accuracy: 0.9318
42016/60000 [====================>.........] - ETA: 24s - loss: 0.2224 - categorical_accuracy: 0.9319
42080/60000 [====================>.........] - ETA: 24s - loss: 0.2223 - categorical_accuracy: 0.9319
42144/60000 [====================>.........] - ETA: 24s - loss: 0.2222 - categorical_accuracy: 0.9320
42208/60000 [====================>.........] - ETA: 24s - loss: 0.2220 - categorical_accuracy: 0.9321
42272/60000 [====================>.........] - ETA: 24s - loss: 0.2219 - categorical_accuracy: 0.9321
42336/60000 [====================>.........] - ETA: 24s - loss: 0.2216 - categorical_accuracy: 0.9322
42400/60000 [====================>.........] - ETA: 24s - loss: 0.2214 - categorical_accuracy: 0.9322
42464/60000 [====================>.........] - ETA: 23s - loss: 0.2211 - categorical_accuracy: 0.9323
42528/60000 [====================>.........] - ETA: 23s - loss: 0.2210 - categorical_accuracy: 0.9323
42560/60000 [====================>.........] - ETA: 23s - loss: 0.2208 - categorical_accuracy: 0.9324
42624/60000 [====================>.........] - ETA: 23s - loss: 0.2208 - categorical_accuracy: 0.9324
42688/60000 [====================>.........] - ETA: 23s - loss: 0.2206 - categorical_accuracy: 0.9325
42752/60000 [====================>.........] - ETA: 23s - loss: 0.2203 - categorical_accuracy: 0.9326
42816/60000 [====================>.........] - ETA: 23s - loss: 0.2201 - categorical_accuracy: 0.9326
42880/60000 [====================>.........] - ETA: 23s - loss: 0.2200 - categorical_accuracy: 0.9327
42944/60000 [====================>.........] - ETA: 23s - loss: 0.2198 - categorical_accuracy: 0.9327
43008/60000 [====================>.........] - ETA: 23s - loss: 0.2195 - categorical_accuracy: 0.9328
43072/60000 [====================>.........] - ETA: 23s - loss: 0.2192 - categorical_accuracy: 0.9329
43136/60000 [====================>.........] - ETA: 22s - loss: 0.2189 - categorical_accuracy: 0.9330
43200/60000 [====================>.........] - ETA: 22s - loss: 0.2187 - categorical_accuracy: 0.9330
43264/60000 [====================>.........] - ETA: 22s - loss: 0.2185 - categorical_accuracy: 0.9331
43328/60000 [====================>.........] - ETA: 22s - loss: 0.2182 - categorical_accuracy: 0.9332
43392/60000 [====================>.........] - ETA: 22s - loss: 0.2180 - categorical_accuracy: 0.9333
43456/60000 [====================>.........] - ETA: 22s - loss: 0.2177 - categorical_accuracy: 0.9334
43520/60000 [====================>.........] - ETA: 22s - loss: 0.2177 - categorical_accuracy: 0.9333
43584/60000 [====================>.........] - ETA: 22s - loss: 0.2178 - categorical_accuracy: 0.9333
43648/60000 [====================>.........] - ETA: 22s - loss: 0.2175 - categorical_accuracy: 0.9334
43712/60000 [====================>.........] - ETA: 22s - loss: 0.2174 - categorical_accuracy: 0.9335
43776/60000 [====================>.........] - ETA: 22s - loss: 0.2173 - categorical_accuracy: 0.9335
43840/60000 [====================>.........] - ETA: 22s - loss: 0.2170 - categorical_accuracy: 0.9336
43904/60000 [====================>.........] - ETA: 21s - loss: 0.2168 - categorical_accuracy: 0.9337
43968/60000 [====================>.........] - ETA: 21s - loss: 0.2170 - categorical_accuracy: 0.9337
44032/60000 [=====================>........] - ETA: 21s - loss: 0.2170 - categorical_accuracy: 0.9336
44096/60000 [=====================>........] - ETA: 21s - loss: 0.2168 - categorical_accuracy: 0.9337
44160/60000 [=====================>........] - ETA: 21s - loss: 0.2167 - categorical_accuracy: 0.9337
44224/60000 [=====================>........] - ETA: 21s - loss: 0.2165 - categorical_accuracy: 0.9337
44288/60000 [=====================>........] - ETA: 21s - loss: 0.2163 - categorical_accuracy: 0.9338
44352/60000 [=====================>........] - ETA: 21s - loss: 0.2162 - categorical_accuracy: 0.9338
44416/60000 [=====================>........] - ETA: 21s - loss: 0.2160 - categorical_accuracy: 0.9339
44480/60000 [=====================>........] - ETA: 21s - loss: 0.2158 - categorical_accuracy: 0.9340
44544/60000 [=====================>........] - ETA: 21s - loss: 0.2156 - categorical_accuracy: 0.9340
44608/60000 [=====================>........] - ETA: 20s - loss: 0.2155 - categorical_accuracy: 0.9341
44672/60000 [=====================>........] - ETA: 20s - loss: 0.2153 - categorical_accuracy: 0.9341
44704/60000 [=====================>........] - ETA: 20s - loss: 0.2152 - categorical_accuracy: 0.9342
44768/60000 [=====================>........] - ETA: 20s - loss: 0.2150 - categorical_accuracy: 0.9342
44832/60000 [=====================>........] - ETA: 20s - loss: 0.2149 - categorical_accuracy: 0.9342
44896/60000 [=====================>........] - ETA: 20s - loss: 0.2147 - categorical_accuracy: 0.9343
44960/60000 [=====================>........] - ETA: 20s - loss: 0.2147 - categorical_accuracy: 0.9343
45024/60000 [=====================>........] - ETA: 20s - loss: 0.2145 - categorical_accuracy: 0.9344
45088/60000 [=====================>........] - ETA: 20s - loss: 0.2143 - categorical_accuracy: 0.9344
45152/60000 [=====================>........] - ETA: 20s - loss: 0.2141 - categorical_accuracy: 0.9345
45216/60000 [=====================>........] - ETA: 20s - loss: 0.2138 - categorical_accuracy: 0.9346
45280/60000 [=====================>........] - ETA: 20s - loss: 0.2136 - categorical_accuracy: 0.9346
45344/60000 [=====================>........] - ETA: 19s - loss: 0.2135 - categorical_accuracy: 0.9346
45408/60000 [=====================>........] - ETA: 19s - loss: 0.2135 - categorical_accuracy: 0.9347
45472/60000 [=====================>........] - ETA: 19s - loss: 0.2133 - categorical_accuracy: 0.9347
45536/60000 [=====================>........] - ETA: 19s - loss: 0.2131 - categorical_accuracy: 0.9348
45600/60000 [=====================>........] - ETA: 19s - loss: 0.2129 - categorical_accuracy: 0.9349
45664/60000 [=====================>........] - ETA: 19s - loss: 0.2127 - categorical_accuracy: 0.9349
45728/60000 [=====================>........] - ETA: 19s - loss: 0.2127 - categorical_accuracy: 0.9350
45792/60000 [=====================>........] - ETA: 19s - loss: 0.2126 - categorical_accuracy: 0.9350
45856/60000 [=====================>........] - ETA: 19s - loss: 0.2123 - categorical_accuracy: 0.9351
45920/60000 [=====================>........] - ETA: 19s - loss: 0.2122 - categorical_accuracy: 0.9351
45984/60000 [=====================>........] - ETA: 19s - loss: 0.2120 - categorical_accuracy: 0.9352
46048/60000 [======================>.......] - ETA: 19s - loss: 0.2119 - categorical_accuracy: 0.9352
46112/60000 [======================>.......] - ETA: 18s - loss: 0.2122 - categorical_accuracy: 0.9352
46176/60000 [======================>.......] - ETA: 18s - loss: 0.2120 - categorical_accuracy: 0.9352
46240/60000 [======================>.......] - ETA: 18s - loss: 0.2119 - categorical_accuracy: 0.9353
46304/60000 [======================>.......] - ETA: 18s - loss: 0.2117 - categorical_accuracy: 0.9353
46368/60000 [======================>.......] - ETA: 18s - loss: 0.2116 - categorical_accuracy: 0.9354
46432/60000 [======================>.......] - ETA: 18s - loss: 0.2113 - categorical_accuracy: 0.9355
46496/60000 [======================>.......] - ETA: 18s - loss: 0.2111 - categorical_accuracy: 0.9355
46560/60000 [======================>.......] - ETA: 18s - loss: 0.2110 - categorical_accuracy: 0.9356
46624/60000 [======================>.......] - ETA: 18s - loss: 0.2108 - categorical_accuracy: 0.9356
46688/60000 [======================>.......] - ETA: 18s - loss: 0.2108 - categorical_accuracy: 0.9356
46752/60000 [======================>.......] - ETA: 18s - loss: 0.2107 - categorical_accuracy: 0.9356
46816/60000 [======================>.......] - ETA: 17s - loss: 0.2107 - categorical_accuracy: 0.9356
46880/60000 [======================>.......] - ETA: 17s - loss: 0.2107 - categorical_accuracy: 0.9356
46944/60000 [======================>.......] - ETA: 17s - loss: 0.2104 - categorical_accuracy: 0.9357
47008/60000 [======================>.......] - ETA: 17s - loss: 0.2104 - categorical_accuracy: 0.9357
47072/60000 [======================>.......] - ETA: 17s - loss: 0.2102 - categorical_accuracy: 0.9358
47136/60000 [======================>.......] - ETA: 17s - loss: 0.2099 - categorical_accuracy: 0.9359
47200/60000 [======================>.......] - ETA: 17s - loss: 0.2098 - categorical_accuracy: 0.9359
47264/60000 [======================>.......] - ETA: 17s - loss: 0.2096 - categorical_accuracy: 0.9360
47328/60000 [======================>.......] - ETA: 17s - loss: 0.2094 - categorical_accuracy: 0.9360
47392/60000 [======================>.......] - ETA: 17s - loss: 0.2092 - categorical_accuracy: 0.9361
47456/60000 [======================>.......] - ETA: 17s - loss: 0.2093 - categorical_accuracy: 0.9361
47520/60000 [======================>.......] - ETA: 16s - loss: 0.2091 - categorical_accuracy: 0.9361
47584/60000 [======================>.......] - ETA: 16s - loss: 0.2090 - categorical_accuracy: 0.9362
47648/60000 [======================>.......] - ETA: 16s - loss: 0.2088 - categorical_accuracy: 0.9362
47712/60000 [======================>.......] - ETA: 16s - loss: 0.2086 - categorical_accuracy: 0.9363
47776/60000 [======================>.......] - ETA: 16s - loss: 0.2085 - categorical_accuracy: 0.9363
47840/60000 [======================>.......] - ETA: 16s - loss: 0.2083 - categorical_accuracy: 0.9364
47904/60000 [======================>.......] - ETA: 16s - loss: 0.2081 - categorical_accuracy: 0.9365
47968/60000 [======================>.......] - ETA: 16s - loss: 0.2079 - categorical_accuracy: 0.9365
48032/60000 [=======================>......] - ETA: 16s - loss: 0.2076 - categorical_accuracy: 0.9366
48096/60000 [=======================>......] - ETA: 16s - loss: 0.2074 - categorical_accuracy: 0.9366
48160/60000 [=======================>......] - ETA: 16s - loss: 0.2072 - categorical_accuracy: 0.9367
48224/60000 [=======================>......] - ETA: 16s - loss: 0.2070 - categorical_accuracy: 0.9368
48288/60000 [=======================>......] - ETA: 15s - loss: 0.2069 - categorical_accuracy: 0.9368
48352/60000 [=======================>......] - ETA: 15s - loss: 0.2067 - categorical_accuracy: 0.9369
48416/60000 [=======================>......] - ETA: 15s - loss: 0.2065 - categorical_accuracy: 0.9369
48480/60000 [=======================>......] - ETA: 15s - loss: 0.2065 - categorical_accuracy: 0.9369
48544/60000 [=======================>......] - ETA: 15s - loss: 0.2064 - categorical_accuracy: 0.9370
48608/60000 [=======================>......] - ETA: 15s - loss: 0.2062 - categorical_accuracy: 0.9370
48672/60000 [=======================>......] - ETA: 15s - loss: 0.2060 - categorical_accuracy: 0.9371
48736/60000 [=======================>......] - ETA: 15s - loss: 0.2057 - categorical_accuracy: 0.9372
48800/60000 [=======================>......] - ETA: 15s - loss: 0.2056 - categorical_accuracy: 0.9372
48864/60000 [=======================>......] - ETA: 15s - loss: 0.2054 - categorical_accuracy: 0.9372
48928/60000 [=======================>......] - ETA: 15s - loss: 0.2053 - categorical_accuracy: 0.9373
48992/60000 [=======================>......] - ETA: 14s - loss: 0.2051 - categorical_accuracy: 0.9373
49056/60000 [=======================>......] - ETA: 14s - loss: 0.2050 - categorical_accuracy: 0.9374
49120/60000 [=======================>......] - ETA: 14s - loss: 0.2048 - categorical_accuracy: 0.9374
49184/60000 [=======================>......] - ETA: 14s - loss: 0.2046 - categorical_accuracy: 0.9375
49248/60000 [=======================>......] - ETA: 14s - loss: 0.2047 - categorical_accuracy: 0.9375
49312/60000 [=======================>......] - ETA: 14s - loss: 0.2046 - categorical_accuracy: 0.9375
49376/60000 [=======================>......] - ETA: 14s - loss: 0.2046 - categorical_accuracy: 0.9375
49440/60000 [=======================>......] - ETA: 14s - loss: 0.2044 - categorical_accuracy: 0.9376
49504/60000 [=======================>......] - ETA: 14s - loss: 0.2043 - categorical_accuracy: 0.9377
49568/60000 [=======================>......] - ETA: 14s - loss: 0.2042 - categorical_accuracy: 0.9377
49632/60000 [=======================>......] - ETA: 14s - loss: 0.2042 - categorical_accuracy: 0.9377
49696/60000 [=======================>......] - ETA: 14s - loss: 0.2042 - categorical_accuracy: 0.9377
49760/60000 [=======================>......] - ETA: 13s - loss: 0.2040 - categorical_accuracy: 0.9378
49824/60000 [=======================>......] - ETA: 13s - loss: 0.2038 - categorical_accuracy: 0.9378
49888/60000 [=======================>......] - ETA: 13s - loss: 0.2037 - categorical_accuracy: 0.9379
49952/60000 [=======================>......] - ETA: 13s - loss: 0.2035 - categorical_accuracy: 0.9379
50016/60000 [========================>.....] - ETA: 13s - loss: 0.2034 - categorical_accuracy: 0.9379
50080/60000 [========================>.....] - ETA: 13s - loss: 0.2033 - categorical_accuracy: 0.9380
50144/60000 [========================>.....] - ETA: 13s - loss: 0.2031 - categorical_accuracy: 0.9380
50176/60000 [========================>.....] - ETA: 13s - loss: 0.2033 - categorical_accuracy: 0.9380
50240/60000 [========================>.....] - ETA: 13s - loss: 0.2031 - categorical_accuracy: 0.9381
50304/60000 [========================>.....] - ETA: 13s - loss: 0.2030 - categorical_accuracy: 0.9381
50368/60000 [========================>.....] - ETA: 13s - loss: 0.2027 - categorical_accuracy: 0.9382
50432/60000 [========================>.....] - ETA: 13s - loss: 0.2025 - categorical_accuracy: 0.9383
50496/60000 [========================>.....] - ETA: 12s - loss: 0.2024 - categorical_accuracy: 0.9383
50560/60000 [========================>.....] - ETA: 12s - loss: 0.2023 - categorical_accuracy: 0.9383
50624/60000 [========================>.....] - ETA: 12s - loss: 0.2022 - categorical_accuracy: 0.9384
50688/60000 [========================>.....] - ETA: 12s - loss: 0.2021 - categorical_accuracy: 0.9384
50752/60000 [========================>.....] - ETA: 12s - loss: 0.2019 - categorical_accuracy: 0.9384
50816/60000 [========================>.....] - ETA: 12s - loss: 0.2019 - categorical_accuracy: 0.9385
50880/60000 [========================>.....] - ETA: 12s - loss: 0.2017 - categorical_accuracy: 0.9386
50944/60000 [========================>.....] - ETA: 12s - loss: 0.2017 - categorical_accuracy: 0.9386
51008/60000 [========================>.....] - ETA: 12s - loss: 0.2016 - categorical_accuracy: 0.9386
51072/60000 [========================>.....] - ETA: 12s - loss: 0.2014 - categorical_accuracy: 0.9387
51136/60000 [========================>.....] - ETA: 12s - loss: 0.2012 - categorical_accuracy: 0.9387
51200/60000 [========================>.....] - ETA: 11s - loss: 0.2011 - categorical_accuracy: 0.9388
51264/60000 [========================>.....] - ETA: 11s - loss: 0.2009 - categorical_accuracy: 0.9388
51328/60000 [========================>.....] - ETA: 11s - loss: 0.2007 - categorical_accuracy: 0.9388
51360/60000 [========================>.....] - ETA: 11s - loss: 0.2008 - categorical_accuracy: 0.9388
51424/60000 [========================>.....] - ETA: 11s - loss: 0.2006 - categorical_accuracy: 0.9388
51488/60000 [========================>.....] - ETA: 11s - loss: 0.2004 - categorical_accuracy: 0.9389
51552/60000 [========================>.....] - ETA: 11s - loss: 0.2002 - categorical_accuracy: 0.9389
51616/60000 [========================>.....] - ETA: 11s - loss: 0.2000 - categorical_accuracy: 0.9390
51680/60000 [========================>.....] - ETA: 11s - loss: 0.1999 - categorical_accuracy: 0.9390
51744/60000 [========================>.....] - ETA: 11s - loss: 0.1997 - categorical_accuracy: 0.9391
51808/60000 [========================>.....] - ETA: 11s - loss: 0.1996 - categorical_accuracy: 0.9391
51872/60000 [========================>.....] - ETA: 11s - loss: 0.1996 - categorical_accuracy: 0.9391
51936/60000 [========================>.....] - ETA: 10s - loss: 0.1994 - categorical_accuracy: 0.9392
52000/60000 [=========================>....] - ETA: 10s - loss: 0.1994 - categorical_accuracy: 0.9392
52064/60000 [=========================>....] - ETA: 10s - loss: 0.1993 - categorical_accuracy: 0.9392
52128/60000 [=========================>....] - ETA: 10s - loss: 0.1993 - categorical_accuracy: 0.9393
52192/60000 [=========================>....] - ETA: 10s - loss: 0.1991 - categorical_accuracy: 0.9393
52256/60000 [=========================>....] - ETA: 10s - loss: 0.1989 - categorical_accuracy: 0.9394
52320/60000 [=========================>....] - ETA: 10s - loss: 0.1988 - categorical_accuracy: 0.9394
52384/60000 [=========================>....] - ETA: 10s - loss: 0.1986 - categorical_accuracy: 0.9395
52448/60000 [=========================>....] - ETA: 10s - loss: 0.1984 - categorical_accuracy: 0.9396
52512/60000 [=========================>....] - ETA: 10s - loss: 0.1983 - categorical_accuracy: 0.9396
52576/60000 [=========================>....] - ETA: 10s - loss: 0.1982 - categorical_accuracy: 0.9396
52640/60000 [=========================>....] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9397
52704/60000 [=========================>....] - ETA: 9s - loss: 0.1979 - categorical_accuracy: 0.9398 
52768/60000 [=========================>....] - ETA: 9s - loss: 0.1978 - categorical_accuracy: 0.9398
52832/60000 [=========================>....] - ETA: 9s - loss: 0.1976 - categorical_accuracy: 0.9398
52896/60000 [=========================>....] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9399
52960/60000 [=========================>....] - ETA: 9s - loss: 0.1973 - categorical_accuracy: 0.9399
53024/60000 [=========================>....] - ETA: 9s - loss: 0.1972 - categorical_accuracy: 0.9400
53088/60000 [=========================>....] - ETA: 9s - loss: 0.1970 - categorical_accuracy: 0.9400
53152/60000 [=========================>....] - ETA: 9s - loss: 0.1969 - categorical_accuracy: 0.9400
53216/60000 [=========================>....] - ETA: 9s - loss: 0.1967 - categorical_accuracy: 0.9401
53280/60000 [=========================>....] - ETA: 9s - loss: 0.1965 - categorical_accuracy: 0.9401
53312/60000 [=========================>....] - ETA: 9s - loss: 0.1964 - categorical_accuracy: 0.9402
53376/60000 [=========================>....] - ETA: 9s - loss: 0.1962 - categorical_accuracy: 0.9403
53440/60000 [=========================>....] - ETA: 8s - loss: 0.1961 - categorical_accuracy: 0.9403
53504/60000 [=========================>....] - ETA: 8s - loss: 0.1962 - categorical_accuracy: 0.9403
53568/60000 [=========================>....] - ETA: 8s - loss: 0.1962 - categorical_accuracy: 0.9403
53632/60000 [=========================>....] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9404
53664/60000 [=========================>....] - ETA: 8s - loss: 0.1960 - categorical_accuracy: 0.9404
53728/60000 [=========================>....] - ETA: 8s - loss: 0.1958 - categorical_accuracy: 0.9405
53792/60000 [=========================>....] - ETA: 8s - loss: 0.1958 - categorical_accuracy: 0.9405
53856/60000 [=========================>....] - ETA: 8s - loss: 0.1956 - categorical_accuracy: 0.9405
53920/60000 [=========================>....] - ETA: 8s - loss: 0.1955 - categorical_accuracy: 0.9406
53984/60000 [=========================>....] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9406
54048/60000 [==========================>...] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9406
54112/60000 [==========================>...] - ETA: 8s - loss: 0.1953 - categorical_accuracy: 0.9406
54176/60000 [==========================>...] - ETA: 7s - loss: 0.1952 - categorical_accuracy: 0.9407
54240/60000 [==========================>...] - ETA: 7s - loss: 0.1950 - categorical_accuracy: 0.9407
54272/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9408
54336/60000 [==========================>...] - ETA: 7s - loss: 0.1949 - categorical_accuracy: 0.9408
54400/60000 [==========================>...] - ETA: 7s - loss: 0.1947 - categorical_accuracy: 0.9408
54464/60000 [==========================>...] - ETA: 7s - loss: 0.1946 - categorical_accuracy: 0.9409
54528/60000 [==========================>...] - ETA: 7s - loss: 0.1945 - categorical_accuracy: 0.9409
54592/60000 [==========================>...] - ETA: 7s - loss: 0.1945 - categorical_accuracy: 0.9410
54656/60000 [==========================>...] - ETA: 7s - loss: 0.1943 - categorical_accuracy: 0.9410
54720/60000 [==========================>...] - ETA: 7s - loss: 0.1943 - categorical_accuracy: 0.9410
54784/60000 [==========================>...] - ETA: 7s - loss: 0.1941 - categorical_accuracy: 0.9411
54848/60000 [==========================>...] - ETA: 7s - loss: 0.1940 - categorical_accuracy: 0.9411
54912/60000 [==========================>...] - ETA: 6s - loss: 0.1939 - categorical_accuracy: 0.9411
54976/60000 [==========================>...] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9412
55040/60000 [==========================>...] - ETA: 6s - loss: 0.1936 - categorical_accuracy: 0.9412
55104/60000 [==========================>...] - ETA: 6s - loss: 0.1937 - categorical_accuracy: 0.9413
55168/60000 [==========================>...] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9413
55232/60000 [==========================>...] - ETA: 6s - loss: 0.1938 - categorical_accuracy: 0.9413
55296/60000 [==========================>...] - ETA: 6s - loss: 0.1936 - categorical_accuracy: 0.9414
55360/60000 [==========================>...] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9414
55424/60000 [==========================>...] - ETA: 6s - loss: 0.1936 - categorical_accuracy: 0.9413
55488/60000 [==========================>...] - ETA: 6s - loss: 0.1936 - categorical_accuracy: 0.9413
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1935 - categorical_accuracy: 0.9414
55616/60000 [==========================>...] - ETA: 5s - loss: 0.1933 - categorical_accuracy: 0.9414
55680/60000 [==========================>...] - ETA: 5s - loss: 0.1933 - categorical_accuracy: 0.9415
55744/60000 [==========================>...] - ETA: 5s - loss: 0.1931 - categorical_accuracy: 0.9415
55808/60000 [==========================>...] - ETA: 5s - loss: 0.1930 - categorical_accuracy: 0.9415
55872/60000 [==========================>...] - ETA: 5s - loss: 0.1928 - categorical_accuracy: 0.9416
55936/60000 [==========================>...] - ETA: 5s - loss: 0.1926 - categorical_accuracy: 0.9416
56000/60000 [===========================>..] - ETA: 5s - loss: 0.1925 - categorical_accuracy: 0.9417
56064/60000 [===========================>..] - ETA: 5s - loss: 0.1925 - categorical_accuracy: 0.9417
56128/60000 [===========================>..] - ETA: 5s - loss: 0.1923 - categorical_accuracy: 0.9418
56192/60000 [===========================>..] - ETA: 5s - loss: 0.1922 - categorical_accuracy: 0.9418
56256/60000 [===========================>..] - ETA: 5s - loss: 0.1921 - categorical_accuracy: 0.9418
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1920 - categorical_accuracy: 0.9419
56384/60000 [===========================>..] - ETA: 4s - loss: 0.1919 - categorical_accuracy: 0.9419
56448/60000 [===========================>..] - ETA: 4s - loss: 0.1918 - categorical_accuracy: 0.9419
56512/60000 [===========================>..] - ETA: 4s - loss: 0.1917 - categorical_accuracy: 0.9419
56576/60000 [===========================>..] - ETA: 4s - loss: 0.1915 - categorical_accuracy: 0.9420
56640/60000 [===========================>..] - ETA: 4s - loss: 0.1914 - categorical_accuracy: 0.9420
56704/60000 [===========================>..] - ETA: 4s - loss: 0.1912 - categorical_accuracy: 0.9421
56768/60000 [===========================>..] - ETA: 4s - loss: 0.1910 - categorical_accuracy: 0.9421
56832/60000 [===========================>..] - ETA: 4s - loss: 0.1909 - categorical_accuracy: 0.9421
56896/60000 [===========================>..] - ETA: 4s - loss: 0.1908 - categorical_accuracy: 0.9422
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1907 - categorical_accuracy: 0.9422
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9423
57088/60000 [===========================>..] - ETA: 3s - loss: 0.1905 - categorical_accuracy: 0.9423
57152/60000 [===========================>..] - ETA: 3s - loss: 0.1905 - categorical_accuracy: 0.9423
57216/60000 [===========================>..] - ETA: 3s - loss: 0.1904 - categorical_accuracy: 0.9423
57280/60000 [===========================>..] - ETA: 3s - loss: 0.1902 - categorical_accuracy: 0.9424
57344/60000 [===========================>..] - ETA: 3s - loss: 0.1900 - categorical_accuracy: 0.9425
57408/60000 [===========================>..] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9425
57472/60000 [===========================>..] - ETA: 3s - loss: 0.1898 - categorical_accuracy: 0.9425
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1897 - categorical_accuracy: 0.9426
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1895 - categorical_accuracy: 0.9426
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1894 - categorical_accuracy: 0.9427
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1892 - categorical_accuracy: 0.9427
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1890 - categorical_accuracy: 0.9428
57856/60000 [===========================>..] - ETA: 2s - loss: 0.1888 - categorical_accuracy: 0.9429
57920/60000 [===========================>..] - ETA: 2s - loss: 0.1887 - categorical_accuracy: 0.9429
57984/60000 [===========================>..] - ETA: 2s - loss: 0.1886 - categorical_accuracy: 0.9429
58048/60000 [============================>.] - ETA: 2s - loss: 0.1885 - categorical_accuracy: 0.9430
58112/60000 [============================>.] - ETA: 2s - loss: 0.1883 - categorical_accuracy: 0.9430
58176/60000 [============================>.] - ETA: 2s - loss: 0.1883 - categorical_accuracy: 0.9430
58240/60000 [============================>.] - ETA: 2s - loss: 0.1882 - categorical_accuracy: 0.9430
58304/60000 [============================>.] - ETA: 2s - loss: 0.1880 - categorical_accuracy: 0.9431
58368/60000 [============================>.] - ETA: 2s - loss: 0.1879 - categorical_accuracy: 0.9432
58432/60000 [============================>.] - ETA: 2s - loss: 0.1877 - categorical_accuracy: 0.9432
58496/60000 [============================>.] - ETA: 2s - loss: 0.1877 - categorical_accuracy: 0.9432
58560/60000 [============================>.] - ETA: 1s - loss: 0.1875 - categorical_accuracy: 0.9433
58624/60000 [============================>.] - ETA: 1s - loss: 0.1873 - categorical_accuracy: 0.9434
58688/60000 [============================>.] - ETA: 1s - loss: 0.1871 - categorical_accuracy: 0.9434
58752/60000 [============================>.] - ETA: 1s - loss: 0.1870 - categorical_accuracy: 0.9435
58816/60000 [============================>.] - ETA: 1s - loss: 0.1868 - categorical_accuracy: 0.9435
58880/60000 [============================>.] - ETA: 1s - loss: 0.1866 - categorical_accuracy: 0.9435
58944/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9436
59008/60000 [============================>.] - ETA: 1s - loss: 0.1863 - categorical_accuracy: 0.9436
59072/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9437
59136/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9437
59200/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9437
59264/60000 [============================>.] - ETA: 1s - loss: 0.1859 - categorical_accuracy: 0.9438
59328/60000 [============================>.] - ETA: 0s - loss: 0.1858 - categorical_accuracy: 0.9438
59392/60000 [============================>.] - ETA: 0s - loss: 0.1857 - categorical_accuracy: 0.9438
59456/60000 [============================>.] - ETA: 0s - loss: 0.1855 - categorical_accuracy: 0.9439
59520/60000 [============================>.] - ETA: 0s - loss: 0.1854 - categorical_accuracy: 0.9439
59584/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9440
59648/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9440
59712/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9441
59776/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9441
59840/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9441
59904/60000 [============================>.] - ETA: 0s - loss: 0.1848 - categorical_accuracy: 0.9442
59968/60000 [============================>.] - ETA: 0s - loss: 0.1847 - categorical_accuracy: 0.9442
60000/60000 [==============================] - 85s 1ms/step - loss: 0.1848 - categorical_accuracy: 0.9442 - val_loss: 0.0520 - val_categorical_accuracy: 0.9844

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 14s
  256/10000 [..............................] - ETA: 3s 
  480/10000 [>.............................] - ETA: 3s
  672/10000 [=>............................] - ETA: 2s
  864/10000 [=>............................] - ETA: 2s
 1056/10000 [==>...........................] - ETA: 2s
 1248/10000 [==>...........................] - ETA: 2s
 1440/10000 [===>..........................] - ETA: 2s
 1632/10000 [===>..........................] - ETA: 2s
 1824/10000 [====>.........................] - ETA: 2s
 2016/10000 [=====>........................] - ETA: 2s
 2208/10000 [=====>........................] - ETA: 2s
 2432/10000 [======>.......................] - ETA: 2s
 2656/10000 [======>.......................] - ETA: 2s
 2848/10000 [=======>......................] - ETA: 2s
 3040/10000 [========>.....................] - ETA: 1s
 3264/10000 [========>.....................] - ETA: 1s
 3456/10000 [=========>....................] - ETA: 1s
 3680/10000 [==========>...................] - ETA: 1s
 3904/10000 [==========>...................] - ETA: 1s
 4096/10000 [===========>..................] - ETA: 1s
 4320/10000 [===========>..................] - ETA: 1s
 4544/10000 [============>.................] - ETA: 1s
 4768/10000 [=============>................] - ETA: 1s
 4992/10000 [=============>................] - ETA: 1s
 5184/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5600/10000 [===============>..............] - ETA: 1s
 5824/10000 [================>.............] - ETA: 1s
 6048/10000 [=================>............] - ETA: 1s
 6240/10000 [=================>............] - ETA: 1s
 6432/10000 [==================>...........] - ETA: 0s
 6656/10000 [==================>...........] - ETA: 0s
 6848/10000 [===================>..........] - ETA: 0s
 7072/10000 [====================>.........] - ETA: 0s
 7264/10000 [====================>.........] - ETA: 0s
 7456/10000 [=====================>........] - ETA: 0s
 7680/10000 [======================>.......] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8096/10000 [=======================>......] - ETA: 0s
 8320/10000 [=======================>......] - ETA: 0s
 8544/10000 [========================>.....] - ETA: 0s
 8768/10000 [=========================>....] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9184/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9632/10000 [===========================>..] - ETA: 0s
 9856/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 264us/step
[[6.9686408e-09 5.2330265e-09 3.3731297e-07 ... 9.9999940e-01
  1.6561983e-09 1.1937047e-07]
 [8.7461312e-06 2.2787235e-05 9.9996579e-01 ... 1.2962749e-08
  1.0753724e-06 6.2354844e-10]
 [6.8209042e-06 9.9988914e-01 1.5324336e-05 ... 3.4299279e-05
  2.2027956e-05 6.8454489e-07]
 ...
 [1.0634787e-10 6.2324240e-08 3.1234371e-10 ... 3.3389824e-07
  3.0846982e-07 2.9505236e-06]
 [5.1188118e-07 1.8309408e-09 6.6183595e-08 ... 1.6253757e-07
  1.7886698e-03 1.2393429e-06]
 [5.3694072e-07 1.4484450e-08 2.7035237e-06 ... 5.1442656e-10
  5.3673693e-07 1.7596895e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05204916457508225, 'accuracy_test:': 0.9843999743461609}

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
[master d9973c0] ml_store  && git pull --all
 1 file changed, 1096 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 22b6a94...d9973c0 master -> master (forced update)





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
{'loss': 0.500198382884264, 'loss_history': []}

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
[master 3a5be66] ml_store  && git pull --all
 1 file changed, 111 insertions(+)
To github.com:arita37/mlmodels_store.git
   d9973c0..3a5be66  master -> master





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
[master 0fca876] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   3a5be66..0fca876  master -> master





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
 40%|████      | 2/5 [00:19<00:28,  9.56s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.933663686602709, 'learning_rate': 0.09255745331825321, 'min_data_in_leaf': 11, 'num_leaves': 45} and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xe0\x92\xaa\xed|XX\r\x00\x00\x00learning_rateq\x02G?\xb7\xb1\xd8c\x00\xc0\xf4X\x10\x00\x00\x00min_data_in_leafq\x03K\x0bX\n\x00\x00\x00num_leavesq\x04K-u.' and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\xe0\x92\xaa\xed|XX\r\x00\x00\x00learning_rateq\x02G?\xb7\xb1\xd8c\x00\xc0\xf4X\x10\x00\x00\x00min_data_in_leafq\x03K\x0bX\n\x00\x00\x00num_leavesq\x04K-u.' and reward: 0.3902
 60%|██████    | 3/5 [00:40<00:26, 13.18s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9227227654644483, 'learning_rate': 0.019555635173495768, 'min_data_in_leaf': 17, 'num_leaves': 40} and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x86\xf1\xe4\x9e96X\r\x00\x00\x00learning_rateq\x02G?\x94\x06dv\x177:X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K(u.' and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x86\xf1\xe4\x9e96X\r\x00\x00\x00learning_rateq\x02G?\x94\x06dv\x177:X\x10\x00\x00\x00min_data_in_leafq\x03K\x11X\n\x00\x00\x00num_leavesq\x04K(u.' and reward: 0.3902
 80%|████████  | 4/5 [01:01<00:15, 15.43s/it] 80%|████████  | 4/5 [01:01<00:15, 15.35s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8961595163279996, 'learning_rate': 0.008196136230774238, 'min_data_in_leaf': 28, 'num_leaves': 58} and reward: 0.386
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xadV\xb8\xd4\x18\x0eX\r\x00\x00\x00learning_rateq\x02G?\x80\xc9"\xc8\x84\x94WX\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K:u.' and reward: 0.386
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xadV\xb8\xd4\x18\x0eX\r\x00\x00\x00learning_rateq\x02G?\x80\xc9"\xc8\x84\x94WX\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K:u.' and reward: 0.386
Time for Gradient Boosting hyperparameter optimization: 88.31942987442017
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:49<01:14, 24.74s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.4705479893717549, 'embedding_size_factor': 0.6138204183186308, 'layers.choice': 1, 'learning_rate': 0.0008414800971345956, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 1.5099520436047005e-08} and reward: 0.3674
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xde\x1duPc7.X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3\xa4j\xb7\xc9u7X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?K\x92\xd8\xbf\xab\xb8nX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>P6\x86Jo\xe0hu.' and reward: 0.3674
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xde\x1duPc7.X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe3\xa4j\xb7\xc9u7X\r\x00\x00\x00layers.choiceq\x04K\x01X\r\x00\x00\x00learning_rateq\x05G?K\x92\xd8\xbf\xab\xb8nX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>P6\x86Jo\xe0hu.' and reward: 0.3674
 60%|██████    | 3/5 [02:45<01:44, 52.03s/it] 60%|██████    | 3/5 [02:45<01:50, 55.06s/it]
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
distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.03942176466745124, 'embedding_size_factor': 0.8464152612238471, 'layers.choice': 0, 'learning_rate': 0.005891727470011709, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 3.257543447505317e-12} and reward: 0.3848
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xa4/\x16\xeb\xfe\x9f\x99X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\x15\xd5u9U\xc6X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?x!\xec\x8c\xcfq?X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x8c\xa7U\xf2G\xed\nu.' and reward: 0.3848
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xa4/\x16\xeb\xfe\x9f\x99X\x15\x00\x00\x00embedding_size_factorq\x03G?\xeb\x15\xd5u9U\xc6X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?x!\xec\x8c\xcfq?X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x8c\xa7U\xf2G\xed\nu.' and reward: 0.3848
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 217.1435101032257
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.78s of the -189.69s of remaining time.
Ensemble size: 80
Ensemble weights: 
[0.1875 0.0875 0.0625 0.1875 0.175  0.1875 0.1125]
	0.3982	 = Validation accuracy score
	1.57s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 311.31s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f17541a9be0>

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
[master 384af21] ml_store  && git pull --all
 1 file changed, 283 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + cb2189d...384af21 master -> master (forced update)





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
[master ea1c422] ml_store  && git pull --all
 1 file changed, 35 insertions(+)
To github.com:arita37/mlmodels_store.git
   384af21..ea1c422  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  4.06it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 2.465 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.254002
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.254001808166504 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f27901c03c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f27901c03c8>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 108.23it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1145.3619791666667,
    "abs_error": 386.05780029296875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.5579996130773344,
    "sMAPE": 0.528148457103983,
    "MSIS": 102.31998128776631,
    "QuantileLoss[0.5]": 386.0577850341797,
    "Coverage[0.5]": 1.0,
    "RMSE": 33.84319694069499,
    "NRMSE": 0.7124883566462102,
    "ND": 0.6772943864788925,
    "wQuantileLoss[0.5]": 0.6772943597090871,
    "mean_wQuantileLoss": 0.6772943597090871,
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
100%|██████████| 10/10 [00:01<00:00,  7.80it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.283 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2763c50240>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2763c50240>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 160.81it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.26it/s, avg_epoch_loss=5.2]
INFO:root:Epoch[0] Elapsed time 1.903 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.197623
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.197622776031494 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2760c41128>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2760c41128>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 150.67it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 354.009765625,
    "abs_error": 204.21319580078125,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.3531063883368406,
    "sMAPE": 0.3251443729700797,
    "MSIS": 54.12425553347364,
    "QuantileLoss[0.5]": 204.21319198608398,
    "Coverage[0.5]": 0.75,
    "RMSE": 18.815147238993376,
    "NRMSE": 0.39610836292617635,
    "ND": 0.3582687645627741,
    "wQuantileLoss[0.5]": 0.3582687578703228,
    "mean_wQuantileLoss": 0.3582687578703228,
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
 30%|███       | 3/10 [00:12<00:29,  4.27s/it, avg_epoch_loss=6.92] 60%|██████    | 6/10 [00:24<00:16,  4.13s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:35<00:04,  4.06s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:39<00:00,  3.98s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 39.773 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.860541
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.860541248321534 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2760bc5c88>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2760bc5c88>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 116.98it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 53525.661458333336,
    "abs_error": 2733.185546875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.109950287415934,
    "sMAPE": 1.4144968254601302,
    "MSIS": 724.3980114966373,
    "QuantileLoss[0.5]": 2733.1853637695312,
    "Coverage[0.5]": 1.0,
    "RMSE": 231.35613555368124,
    "NRMSE": 4.870655485340658,
    "ND": 4.795062362938596,
    "wQuantileLoss[0.5]": 4.795062041700932,
    "mean_wQuantileLoss": 4.795062041700932,
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
100%|██████████| 10/10 [00:00<00:00, 59.74it/s, avg_epoch_loss=5.28]
INFO:root:Epoch[0] Elapsed time 0.168 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.284914
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.284914350509643 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2763ce6d30>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2763ce6d30>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 166.94it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 331.3861897786458,
    "abs_error": 181.73013305664062,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.204134742752524,
    "sMAPE": 0.31507365179630115,
    "MSIS": 48.16539132776448,
    "QuantileLoss[0.5]": 181.73012924194336,
    "Coverage[0.5]": 0.5833333333333334,
    "RMSE": 18.20401575967912,
    "NRMSE": 0.3832424370458762,
    "ND": 0.3188247948362116,
    "wQuantileLoss[0.5]": 0.3188247881437603,
    "mean_wQuantileLoss": 0.3188247881437603,
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
100%|██████████| 10/10 [00:01<00:00,  7.79it/s, avg_epoch_loss=123]
INFO:root:Epoch[0] Elapsed time 1.285 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=122.866774
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 122.86677375545166 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2760bdb5f8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2760bdb5f8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 164.16it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [02:00<18:07, 120.86s/it, avg_epoch_loss=0.703] 20%|██        | 2/10 [04:55<18:16, 137.04s/it, avg_epoch_loss=0.686] 30%|███       | 3/10 [07:58<17:35, 150.74s/it, avg_epoch_loss=0.669] 40%|████      | 4/10 [10:55<15:52, 158.75s/it, avg_epoch_loss=0.652] 50%|█████     | 5/10 [14:34<14:44, 176.82s/it, avg_epoch_loss=0.634] 60%|██████    | 6/10 [18:01<12:23, 185.76s/it, avg_epoch_loss=0.617] 70%|███████   | 7/10 [21:16<09:25, 188.43s/it, avg_epoch_loss=0.599] 80%|████████  | 8/10 [24:47<06:30, 195.23s/it, avg_epoch_loss=0.581] 90%|█████████ | 9/10 [28:05<03:16, 196.01s/it, avg_epoch_loss=0.563]100%|██████████| 10/10 [31:39<00:00, 201.65s/it, avg_epoch_loss=0.546]100%|██████████| 10/10 [31:39<00:00, 189.99s/it, avg_epoch_loss=0.546]
INFO:root:Epoch[0] Elapsed time 1899.876 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.546187
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5461866021156311 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2760c32f28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2760c32f28>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 14.46it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
[master 4c71783] ml_store  && git pull --all
 1 file changed, 498 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 900b9fe...4c71783 master -> master (forced update)





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
{'roc_auc_score': 0.9583333333333333}

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_sklearn' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_sklearn.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f271c596748> 

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
[master b16747a] ml_store  && git pull --all
 1 file changed, 107 insertions(+)
To github.com:arita37/mlmodels_store.git
   4c71783..b16747a  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 1.34728643 -0.36453805  0.08075099 -0.45971768 -0.8894876   1.70548352
   0.09499611  0.24050555 -0.9994265  -0.76780375]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 1.02817479 -0.50845713  1.7653351   0.77741921  0.61771419 -0.11877117
   0.45015551 -0.19899818  1.86647138  0.8709698 ]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 0.55853873 -0.51634791 -0.51814555  0.3511169   0.82550695 -0.06877046
  -0.9520621  -1.34776494  1.47073986 -1.4614036 ]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f87fae6bd68>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f881bee7fd0> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.18559003  0.08646441  1.23289919 -2.14246673  1.033341   -0.83016886
   0.36723181  0.45161595  1.10417433 -0.42285696]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.99785516 -0.6001388   0.45794708  0.14676526 -0.93355729  0.57180488
   0.57296273 -0.03681766  0.11236849 -0.01781755]
 [ 0.89551051  0.92061512  0.79452824 -0.03536792  0.8780991   2.11060505
  -1.02188594 -1.30653407  0.07638048 -1.87316098]
 [ 0.78344054 -0.05118845  0.82458463 -0.72559712  0.9317172  -0.86776868
   3.03085711 -0.13597733 -0.79726979  0.65458015]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 1.16777676 -0.66575452 -1.23312074 -1.67419581  1.01313574  0.82502982
  -0.12046457 -0.49821356 -0.31098498 -1.18231813]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 0.87874071 -0.01923163  0.31965694  0.15001628 -1.46662161  0.46353432
  -0.89868319  0.39788042 -0.99601089  0.3181542 ]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]]
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
[[ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.88838944  0.28299553  0.01795589  0.10803082 -0.84967187  0.02941762
  -0.50397395 -0.13479313  1.04921829 -1.27046078]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
   0.59921324 -0.21465384  1.02124813  0.60640394]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]]
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
[master 7aeb59e] ml_store  && git pull --all
 1 file changed, 245 insertions(+)
To github.com:arita37/mlmodels_store.git
   b16747a..7aeb59e  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140461479239632
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140461479239408
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140461479238176
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140461479237728
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140461479237224
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140461479236888

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
grad_step = 000000, loss = 0.323738
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.250997
grad_step = 000002, loss = 0.198455
grad_step = 000003, loss = 0.146177
grad_step = 000004, loss = 0.093792
grad_step = 000005, loss = 0.054333
grad_step = 000006, loss = 0.047422
grad_step = 000007, loss = 0.055296
grad_step = 000008, loss = 0.046028
grad_step = 000009, loss = 0.028259
grad_step = 000010, loss = 0.016684
grad_step = 000011, loss = 0.013651
grad_step = 000012, loss = 0.015740
grad_step = 000013, loss = 0.018613
grad_step = 000014, loss = 0.019618
grad_step = 000015, loss = 0.018037
grad_step = 000016, loss = 0.014515
grad_step = 000017, loss = 0.010435
grad_step = 000018, loss = 0.007416
grad_step = 000019, loss = 0.006640
grad_step = 000020, loss = 0.007973
grad_step = 000021, loss = 0.009686
grad_step = 000022, loss = 0.010094
grad_step = 000023, loss = 0.009191
grad_step = 000024, loss = 0.008094
grad_step = 000025, loss = 0.007610
grad_step = 000026, loss = 0.007678
grad_step = 000027, loss = 0.007816
grad_step = 000028, loss = 0.007700
grad_step = 000029, loss = 0.007257
grad_step = 000030, loss = 0.006636
grad_step = 000031, loss = 0.006170
grad_step = 000032, loss = 0.006038
grad_step = 000033, loss = 0.006121
grad_step = 000034, loss = 0.006114
grad_step = 000035, loss = 0.005832
grad_step = 000036, loss = 0.005378
grad_step = 000037, loss = 0.005025
grad_step = 000038, loss = 0.004938
grad_step = 000039, loss = 0.005061
grad_step = 000040, loss = 0.005196
grad_step = 000041, loss = 0.005190
grad_step = 000042, loss = 0.005016
grad_step = 000043, loss = 0.004773
grad_step = 000044, loss = 0.004597
grad_step = 000045, loss = 0.004551
grad_step = 000046, loss = 0.004570
grad_step = 000047, loss = 0.004542
grad_step = 000048, loss = 0.004427
grad_step = 000049, loss = 0.004282
grad_step = 000050, loss = 0.004179
grad_step = 000051, loss = 0.004123
grad_step = 000052, loss = 0.004065
grad_step = 000053, loss = 0.003965
grad_step = 000054, loss = 0.003830
grad_step = 000055, loss = 0.003713
grad_step = 000056, loss = 0.003642
grad_step = 000057, loss = 0.003606
grad_step = 000058, loss = 0.003559
grad_step = 000059, loss = 0.003476
grad_step = 000060, loss = 0.003378
grad_step = 000061, loss = 0.003306
grad_step = 000062, loss = 0.003264
grad_step = 000063, loss = 0.003222
grad_step = 000064, loss = 0.003160
grad_step = 000065, loss = 0.003087
grad_step = 000066, loss = 0.003023
grad_step = 000067, loss = 0.002974
grad_step = 000068, loss = 0.002920
grad_step = 000069, loss = 0.002852
grad_step = 000070, loss = 0.002778
grad_step = 000071, loss = 0.002714
grad_step = 000072, loss = 0.002658
grad_step = 000073, loss = 0.002593
grad_step = 000074, loss = 0.002518
grad_step = 000075, loss = 0.002445
grad_step = 000076, loss = 0.002381
grad_step = 000077, loss = 0.002318
grad_step = 000078, loss = 0.002246
grad_step = 000079, loss = 0.002174
grad_step = 000080, loss = 0.002108
grad_step = 000081, loss = 0.002040
grad_step = 000082, loss = 0.001970
grad_step = 000083, loss = 0.001905
grad_step = 000084, loss = 0.001842
grad_step = 000085, loss = 0.001775
grad_step = 000086, loss = 0.001713
grad_step = 000087, loss = 0.001657
grad_step = 000088, loss = 0.001602
grad_step = 000089, loss = 0.001547
grad_step = 000090, loss = 0.001496
grad_step = 000091, loss = 0.001449
grad_step = 000092, loss = 0.001407
grad_step = 000093, loss = 0.001368
grad_step = 000094, loss = 0.001331
grad_step = 000095, loss = 0.001298
grad_step = 000096, loss = 0.001269
grad_step = 000097, loss = 0.001250
grad_step = 000098, loss = 0.001254
grad_step = 000099, loss = 0.001253
grad_step = 000100, loss = 0.001191
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.001165
grad_step = 000102, loss = 0.001181
grad_step = 000103, loss = 0.001136
grad_step = 000104, loss = 0.001116
grad_step = 000105, loss = 0.001123
grad_step = 000106, loss = 0.001083
grad_step = 000107, loss = 0.001071
grad_step = 000108, loss = 0.001069
grad_step = 000109, loss = 0.001036
grad_step = 000110, loss = 0.001026
grad_step = 000111, loss = 0.001022
grad_step = 000112, loss = 0.000993
grad_step = 000113, loss = 0.000987
grad_step = 000114, loss = 0.000978
grad_step = 000115, loss = 0.000952
grad_step = 000116, loss = 0.000947
grad_step = 000117, loss = 0.000938
grad_step = 000118, loss = 0.000915
grad_step = 000119, loss = 0.000910
grad_step = 000120, loss = 0.000902
grad_step = 000121, loss = 0.000881
grad_step = 000122, loss = 0.000875
grad_step = 000123, loss = 0.000868
grad_step = 000124, loss = 0.000850
grad_step = 000125, loss = 0.000842
grad_step = 000126, loss = 0.000836
grad_step = 000127, loss = 0.000820
grad_step = 000128, loss = 0.000811
grad_step = 000129, loss = 0.000805
grad_step = 000130, loss = 0.000793
grad_step = 000131, loss = 0.000782
grad_step = 000132, loss = 0.000776
grad_step = 000133, loss = 0.000768
grad_step = 000134, loss = 0.000756
grad_step = 000135, loss = 0.000748
grad_step = 000136, loss = 0.000741
grad_step = 000137, loss = 0.000734
grad_step = 000138, loss = 0.000724
grad_step = 000139, loss = 0.000716
grad_step = 000140, loss = 0.000710
grad_step = 000141, loss = 0.000703
grad_step = 000142, loss = 0.000693
grad_step = 000143, loss = 0.000684
grad_step = 000144, loss = 0.000678
grad_step = 000145, loss = 0.000672
grad_step = 000146, loss = 0.000664
grad_step = 000147, loss = 0.000656
grad_step = 000148, loss = 0.000650
grad_step = 000149, loss = 0.000644
grad_step = 000150, loss = 0.000638
grad_step = 000151, loss = 0.000633
grad_step = 000152, loss = 0.000629
grad_step = 000153, loss = 0.000624
grad_step = 000154, loss = 0.000619
grad_step = 000155, loss = 0.000613
grad_step = 000156, loss = 0.000607
grad_step = 000157, loss = 0.000599
grad_step = 000158, loss = 0.000592
grad_step = 000159, loss = 0.000585
grad_step = 000160, loss = 0.000582
grad_step = 000161, loss = 0.000578
grad_step = 000162, loss = 0.000574
grad_step = 000163, loss = 0.000573
grad_step = 000164, loss = 0.000579
grad_step = 000165, loss = 0.000584
grad_step = 000166, loss = 0.000582
grad_step = 000167, loss = 0.000576
grad_step = 000168, loss = 0.000558
grad_step = 000169, loss = 0.000541
grad_step = 000170, loss = 0.000541
grad_step = 000171, loss = 0.000549
grad_step = 000172, loss = 0.000552
grad_step = 000173, loss = 0.000544
grad_step = 000174, loss = 0.000531
grad_step = 000175, loss = 0.000520
grad_step = 000176, loss = 0.000517
grad_step = 000177, loss = 0.000522
grad_step = 000178, loss = 0.000523
grad_step = 000179, loss = 0.000515
grad_step = 000180, loss = 0.000507
grad_step = 000181, loss = 0.000502
grad_step = 000182, loss = 0.000495
grad_step = 000183, loss = 0.000495
grad_step = 000184, loss = 0.000495
grad_step = 000185, loss = 0.000493
grad_step = 000186, loss = 0.000491
grad_step = 000187, loss = 0.000490
grad_step = 000188, loss = 0.000490
grad_step = 000189, loss = 0.000485
grad_step = 000190, loss = 0.000482
grad_step = 000191, loss = 0.000477
grad_step = 000192, loss = 0.000472
grad_step = 000193, loss = 0.000466
grad_step = 000194, loss = 0.000462
grad_step = 000195, loss = 0.000460
grad_step = 000196, loss = 0.000458
grad_step = 000197, loss = 0.000454
grad_step = 000198, loss = 0.000451
grad_step = 000199, loss = 0.000450
grad_step = 000200, loss = 0.000450
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000452
grad_step = 000202, loss = 0.000460
grad_step = 000203, loss = 0.000478
grad_step = 000204, loss = 0.000503
grad_step = 000205, loss = 0.000515
grad_step = 000206, loss = 0.000492
grad_step = 000207, loss = 0.000448
grad_step = 000208, loss = 0.000427
grad_step = 000209, loss = 0.000441
grad_step = 000210, loss = 0.000463
grad_step = 000211, loss = 0.000461
grad_step = 000212, loss = 0.000434
grad_step = 000213, loss = 0.000416
grad_step = 000214, loss = 0.000428
grad_step = 000215, loss = 0.000442
grad_step = 000216, loss = 0.000437
grad_step = 000217, loss = 0.000416
grad_step = 000218, loss = 0.000407
grad_step = 000219, loss = 0.000415
grad_step = 000220, loss = 0.000424
grad_step = 000221, loss = 0.000418
grad_step = 000222, loss = 0.000403
grad_step = 000223, loss = 0.000397
grad_step = 000224, loss = 0.000401
grad_step = 000225, loss = 0.000403
grad_step = 000226, loss = 0.000399
grad_step = 000227, loss = 0.000391
grad_step = 000228, loss = 0.000386
grad_step = 000229, loss = 0.000387
grad_step = 000230, loss = 0.000390
grad_step = 000231, loss = 0.000391
grad_step = 000232, loss = 0.000386
grad_step = 000233, loss = 0.000380
grad_step = 000234, loss = 0.000375
grad_step = 000235, loss = 0.000373
grad_step = 000236, loss = 0.000372
grad_step = 000237, loss = 0.000372
grad_step = 000238, loss = 0.000371
grad_step = 000239, loss = 0.000371
grad_step = 000240, loss = 0.000371
grad_step = 000241, loss = 0.000372
grad_step = 000242, loss = 0.000373
grad_step = 000243, loss = 0.000365
grad_step = 000244, loss = 0.000358
grad_step = 000245, loss = 0.000357
grad_step = 000246, loss = 0.000359
grad_step = 000247, loss = 0.000361
grad_step = 000248, loss = 0.000358
grad_step = 000249, loss = 0.000354
grad_step = 000250, loss = 0.000349
grad_step = 000251, loss = 0.000347
grad_step = 000252, loss = 0.000347
grad_step = 000253, loss = 0.000351
grad_step = 000254, loss = 0.000355
grad_step = 000255, loss = 0.000358
grad_step = 000256, loss = 0.000358
grad_step = 000257, loss = 0.000359
grad_step = 000258, loss = 0.000360
grad_step = 000259, loss = 0.000375
grad_step = 000260, loss = 0.000383
grad_step = 000261, loss = 0.000380
grad_step = 000262, loss = 0.000341
grad_step = 000263, loss = 0.000332
grad_step = 000264, loss = 0.000357
grad_step = 000265, loss = 0.000355
grad_step = 000266, loss = 0.000332
grad_step = 000267, loss = 0.000328
grad_step = 000268, loss = 0.000342
grad_step = 000269, loss = 0.000338
grad_step = 000270, loss = 0.000321
grad_step = 000271, loss = 0.000320
grad_step = 000272, loss = 0.000331
grad_step = 000273, loss = 0.000329
grad_step = 000274, loss = 0.000318
grad_step = 000275, loss = 0.000317
grad_step = 000276, loss = 0.000325
grad_step = 000277, loss = 0.000331
grad_step = 000278, loss = 0.000325
grad_step = 000279, loss = 0.000321
grad_step = 000280, loss = 0.000325
grad_step = 000281, loss = 0.000330
grad_step = 000282, loss = 0.000325
grad_step = 000283, loss = 0.000322
grad_step = 000284, loss = 0.000322
grad_step = 000285, loss = 0.000324
grad_step = 000286, loss = 0.000313
grad_step = 000287, loss = 0.000303
grad_step = 000288, loss = 0.000299
grad_step = 000289, loss = 0.000301
grad_step = 000290, loss = 0.000302
grad_step = 000291, loss = 0.000301
grad_step = 000292, loss = 0.000302
grad_step = 000293, loss = 0.000305
grad_step = 000294, loss = 0.000309
grad_step = 000295, loss = 0.000309
grad_step = 000296, loss = 0.000307
grad_step = 000297, loss = 0.000303
grad_step = 000298, loss = 0.000305
grad_step = 000299, loss = 0.000304
grad_step = 000300, loss = 0.000312
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000315
grad_step = 000302, loss = 0.000315
grad_step = 000303, loss = 0.000304
grad_step = 000304, loss = 0.000292
grad_step = 000305, loss = 0.000283
grad_step = 000306, loss = 0.000287
grad_step = 000307, loss = 0.000296
grad_step = 000308, loss = 0.000293
grad_step = 000309, loss = 0.000287
grad_step = 000310, loss = 0.000280
grad_step = 000311, loss = 0.000281
grad_step = 000312, loss = 0.000285
grad_step = 000313, loss = 0.000286
grad_step = 000314, loss = 0.000286
grad_step = 000315, loss = 0.000282
grad_step = 000316, loss = 0.000279
grad_step = 000317, loss = 0.000277
grad_step = 000318, loss = 0.000278
grad_step = 000319, loss = 0.000283
grad_step = 000320, loss = 0.000292
grad_step = 000321, loss = 0.000303
grad_step = 000322, loss = 0.000322
grad_step = 000323, loss = 0.000339
grad_step = 000324, loss = 0.000364
grad_step = 000325, loss = 0.000351
grad_step = 000326, loss = 0.000326
grad_step = 000327, loss = 0.000283
grad_step = 000328, loss = 0.000270
grad_step = 000329, loss = 0.000287
grad_step = 000330, loss = 0.000310
grad_step = 000331, loss = 0.000313
grad_step = 000332, loss = 0.000298
grad_step = 000333, loss = 0.000287
grad_step = 000334, loss = 0.000303
grad_step = 000335, loss = 0.000320
grad_step = 000336, loss = 0.000322
grad_step = 000337, loss = 0.000286
grad_step = 000338, loss = 0.000267
grad_step = 000339, loss = 0.000278
grad_step = 000340, loss = 0.000287
grad_step = 000341, loss = 0.000284
grad_step = 000342, loss = 0.000266
grad_step = 000343, loss = 0.000264
grad_step = 000344, loss = 0.000278
grad_step = 000345, loss = 0.000274
grad_step = 000346, loss = 0.000264
grad_step = 000347, loss = 0.000260
grad_step = 000348, loss = 0.000264
grad_step = 000349, loss = 0.000268
grad_step = 000350, loss = 0.000262
grad_step = 000351, loss = 0.000257
grad_step = 000352, loss = 0.000260
grad_step = 000353, loss = 0.000264
grad_step = 000354, loss = 0.000266
grad_step = 000355, loss = 0.000263
grad_step = 000356, loss = 0.000258
grad_step = 000357, loss = 0.000256
grad_step = 000358, loss = 0.000257
grad_step = 000359, loss = 0.000259
grad_step = 000360, loss = 0.000259
grad_step = 000361, loss = 0.000257
grad_step = 000362, loss = 0.000254
grad_step = 000363, loss = 0.000257
grad_step = 000364, loss = 0.000263
grad_step = 000365, loss = 0.000275
grad_step = 000366, loss = 0.000291
grad_step = 000367, loss = 0.000318
grad_step = 000368, loss = 0.000336
grad_step = 000369, loss = 0.000337
grad_step = 000370, loss = 0.000296
grad_step = 000371, loss = 0.000255
grad_step = 000372, loss = 0.000261
grad_step = 000373, loss = 0.000286
grad_step = 000374, loss = 0.000275
grad_step = 000375, loss = 0.000250
grad_step = 000376, loss = 0.000261
grad_step = 000377, loss = 0.000277
grad_step = 000378, loss = 0.000258
grad_step = 000379, loss = 0.000248
grad_step = 000380, loss = 0.000264
grad_step = 000381, loss = 0.000262
grad_step = 000382, loss = 0.000249
grad_step = 000383, loss = 0.000250
grad_step = 000384, loss = 0.000258
grad_step = 000385, loss = 0.000252
grad_step = 000386, loss = 0.000244
grad_step = 000387, loss = 0.000247
grad_step = 000388, loss = 0.000249
grad_step = 000389, loss = 0.000244
grad_step = 000390, loss = 0.000242
grad_step = 000391, loss = 0.000246
grad_step = 000392, loss = 0.000249
grad_step = 000393, loss = 0.000246
grad_step = 000394, loss = 0.000245
grad_step = 000395, loss = 0.000245
grad_step = 000396, loss = 0.000247
grad_step = 000397, loss = 0.000245
grad_step = 000398, loss = 0.000239
grad_step = 000399, loss = 0.000238
grad_step = 000400, loss = 0.000242
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000243
grad_step = 000402, loss = 0.000242
grad_step = 000403, loss = 0.000238
grad_step = 000404, loss = 0.000237
grad_step = 000405, loss = 0.000238
grad_step = 000406, loss = 0.000241
grad_step = 000407, loss = 0.000243
grad_step = 000408, loss = 0.000245
grad_step = 000409, loss = 0.000247
grad_step = 000410, loss = 0.000250
grad_step = 000411, loss = 0.000254
grad_step = 000412, loss = 0.000262
grad_step = 000413, loss = 0.000285
grad_step = 000414, loss = 0.000287
grad_step = 000415, loss = 0.000287
grad_step = 000416, loss = 0.000266
grad_step = 000417, loss = 0.000241
grad_step = 000418, loss = 0.000236
grad_step = 000419, loss = 0.000251
grad_step = 000420, loss = 0.000258
grad_step = 000421, loss = 0.000246
grad_step = 000422, loss = 0.000232
grad_step = 000423, loss = 0.000235
grad_step = 000424, loss = 0.000242
grad_step = 000425, loss = 0.000241
grad_step = 000426, loss = 0.000236
grad_step = 000427, loss = 0.000231
grad_step = 000428, loss = 0.000231
grad_step = 000429, loss = 0.000235
grad_step = 000430, loss = 0.000240
grad_step = 000431, loss = 0.000238
grad_step = 000432, loss = 0.000236
grad_step = 000433, loss = 0.000230
grad_step = 000434, loss = 0.000226
grad_step = 000435, loss = 0.000225
grad_step = 000436, loss = 0.000226
grad_step = 000437, loss = 0.000229
grad_step = 000438, loss = 0.000230
grad_step = 000439, loss = 0.000230
grad_step = 000440, loss = 0.000230
grad_step = 000441, loss = 0.000233
grad_step = 000442, loss = 0.000239
grad_step = 000443, loss = 0.000244
grad_step = 000444, loss = 0.000248
grad_step = 000445, loss = 0.000249
grad_step = 000446, loss = 0.000251
grad_step = 000447, loss = 0.000250
grad_step = 000448, loss = 0.000250
grad_step = 000449, loss = 0.000252
grad_step = 000450, loss = 0.000252
grad_step = 000451, loss = 0.000254
grad_step = 000452, loss = 0.000248
grad_step = 000453, loss = 0.000243
grad_step = 000454, loss = 0.000236
grad_step = 000455, loss = 0.000236
grad_step = 000456, loss = 0.000231
grad_step = 000457, loss = 0.000237
grad_step = 000458, loss = 0.000247
grad_step = 000459, loss = 0.000247
grad_step = 000460, loss = 0.000240
grad_step = 000461, loss = 0.000229
grad_step = 000462, loss = 0.000225
grad_step = 000463, loss = 0.000237
grad_step = 000464, loss = 0.000245
grad_step = 000465, loss = 0.000254
grad_step = 000466, loss = 0.000239
grad_step = 000467, loss = 0.000221
grad_step = 000468, loss = 0.000218
grad_step = 000469, loss = 0.000229
grad_step = 000470, loss = 0.000237
grad_step = 000471, loss = 0.000231
grad_step = 000472, loss = 0.000221
grad_step = 000473, loss = 0.000222
grad_step = 000474, loss = 0.000233
grad_step = 000475, loss = 0.000232
grad_step = 000476, loss = 0.000233
grad_step = 000477, loss = 0.000234
grad_step = 000478, loss = 0.000225
grad_step = 000479, loss = 0.000217
grad_step = 000480, loss = 0.000219
grad_step = 000481, loss = 0.000221
grad_step = 000482, loss = 0.000224
grad_step = 000483, loss = 0.000219
grad_step = 000484, loss = 0.000216
grad_step = 000485, loss = 0.000212
grad_step = 000486, loss = 0.000206
grad_step = 000487, loss = 0.000207
grad_step = 000488, loss = 0.000211
grad_step = 000489, loss = 0.000214
grad_step = 000490, loss = 0.000214
grad_step = 000491, loss = 0.000215
grad_step = 000492, loss = 0.000216
grad_step = 000493, loss = 0.000224
grad_step = 000494, loss = 0.000225
grad_step = 000495, loss = 0.000236
grad_step = 000496, loss = 0.000244
grad_step = 000497, loss = 0.000250
grad_step = 000498, loss = 0.000249
grad_step = 000499, loss = 0.000245
grad_step = 000500, loss = 0.000238
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000223
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
[[0.83991057 0.8627319  0.9196799  0.9316666  1.0113498 ]
 [0.8488115  0.9081571  0.94529855 1.0084819  0.9899069 ]
 [0.88880426 0.93286157 0.99578005 0.9761874  0.9590038 ]
 [0.9369675  0.9737226  0.9977362  0.9541292  0.92549354]
 [1.0045257  0.99844575 0.94302666 0.9193171  0.8505996 ]
 [0.9821861  0.9480674  0.91067153 0.8516679  0.84630704]
 [0.9311803  0.9014349  0.8404416  0.84998775 0.80295885]
 [0.88449967 0.8393994  0.85289514 0.8089142  0.8371328 ]
 [0.82917625 0.82337916 0.81212837 0.8504609  0.8475144 ]
 [0.81085277 0.8066238  0.8325626  0.8466443  0.820735  ]
 [0.8006745  0.8045491  0.8673486  0.8175148  0.92682093]
 [0.8319478  0.8500715  0.82412827 0.9291664  0.94032466]
 [0.8333381  0.855999   0.9145007  0.92996645 1.0057945 ]
 [0.8529213  0.9109384  0.945018   1.0035017  0.97884786]
 [0.9082789  0.9481506  1.0032077  0.9723645  0.94371355]
 [0.94847864 0.97962505 0.9903326  0.93887395 0.90348744]
 [1.0077275  0.9905205  0.928631   0.89535844 0.8298893 ]
 [0.97356725 0.93261623 0.89512503 0.83325696 0.83328897]
 [0.9331038  0.89625245 0.82920504 0.8417464  0.80466676]
 [0.898954   0.8484365  0.8460475  0.81042856 0.8391798 ]
 [0.8483548  0.8371635  0.8136193  0.85624903 0.85769135]
 [0.8331859  0.82305324 0.8371505  0.8560157  0.82922083]
 [0.814926   0.818608   0.8764907  0.82614124 0.9272326 ]
 [0.8428298  0.8600198  0.83087283 0.93175155 0.9428556 ]
 [0.8468115  0.86894953 0.9220057  0.9350924  1.0182521 ]
 [0.85725296 0.9162814  0.9516408  1.0182272  1.0016088 ]
 [0.8995765  0.943875   1.0045377  0.98682535 0.96986693]
 [0.9486484  0.9864907  1.0137949  0.96898997 0.9393481 ]
 [1.0127579  1.0098506  0.955563   0.93083024 0.85527956]
 [0.99116874 0.9569297  0.92224145 0.8561508  0.8505447 ]
 [0.94018126 0.91027594 0.8483628  0.8559401  0.8106034 ]]

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
[master 160da43] ml_store  && git pull --all
 1 file changed, 1121 insertions(+)
To github.com:arita37/mlmodels_store.git
   7aeb59e..160da43  master -> master





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
[master 8c5b5ee] ml_store  && git pull --all
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   160da43..8c5b5ee  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 10662643.22B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 219658.16B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|          | 4968448/440473133 [00:00<00:08, 49680381.72B/s]  3%|▎         | 11530240/440473133 [00:00<00:08, 53584613.65B/s]  4%|▍         | 18227200/440473133 [00:00<00:07, 57002286.98B/s]  6%|▌         | 25022464/440473133 [00:00<00:06, 59897749.66B/s]  7%|▋         | 31732736/440473133 [00:00<00:06, 61890492.67B/s]  9%|▉         | 38558720/440473133 [00:00<00:06, 63670914.45B/s] 10%|█         | 45447168/440473133 [00:00<00:06, 65147964.58B/s] 12%|█▏        | 52282368/440473133 [00:00<00:05, 66075421.71B/s] 13%|█▎        | 59113472/440473133 [00:00<00:05, 66730329.07B/s] 15%|█▍        | 65878016/440473133 [00:01<00:05, 66999755.29B/s] 16%|█▋        | 72464384/440473133 [00:01<00:05, 66334447.03B/s] 18%|█▊        | 79020032/440473133 [00:01<00:05, 65354689.96B/s] 19%|█▉        | 85604352/440473133 [00:01<00:05, 65498355.00B/s] 21%|██        | 92434432/440473133 [00:01<00:05, 66310882.63B/s] 23%|██▎       | 99166208/440473133 [00:01<00:05, 66608373.62B/s] 24%|██▍       | 105811968/440473133 [00:01<00:05, 66070877.26B/s] 26%|██▌       | 112663552/440473133 [00:01<00:04, 66783489.82B/s] 27%|██▋       | 119550976/440473133 [00:01<00:04, 67395585.03B/s] 29%|██▊       | 126313472/440473133 [00:01<00:04, 67460835.50B/s] 30%|███       | 133058560/440473133 [00:02<00:04, 67088390.31B/s] 32%|███▏      | 139827200/440473133 [00:02<00:04, 67266173.92B/s] 33%|███▎      | 146677760/440473133 [00:02<00:04, 67631519.05B/s] 35%|███▍      | 153510912/440473133 [00:02<00:04, 67837667.62B/s] 36%|███▋      | 160356352/440473133 [00:02<00:04, 68020763.16B/s] 38%|███▊      | 167158784/440473133 [00:02<00:04, 67996025.62B/s] 40%|███▉      | 174010368/440473133 [00:02<00:03, 68150383.49B/s] 41%|████      | 180846592/440473133 [00:02<00:03, 68212166.69B/s] 43%|████▎     | 187668480/440473133 [00:02<00:03, 67205538.21B/s] 44%|████▍     | 194437120/440473133 [00:02<00:03, 67348090.44B/s] 46%|████▌     | 201175040/440473133 [00:03<00:03, 66816336.13B/s] 47%|████▋     | 208030720/440473133 [00:03<00:03, 67327725.22B/s] 49%|████▉     | 214893568/440473133 [00:03<00:03, 67712381.55B/s] 50%|█████     | 221733888/440473133 [00:03<00:03, 67917115.67B/s] 52%|█████▏    | 228586496/440473133 [00:03<00:03, 68096361.53B/s] 53%|█████▎    | 235398144/440473133 [00:03<00:03, 68021279.51B/s] 55%|█████▌    | 242275328/440473133 [00:03<00:02, 68244208.36B/s] 57%|█████▋    | 249111552/440473133 [00:03<00:02, 68278131.23B/s] 58%|█████▊    | 255940608/440473133 [00:03<00:02, 68279779.17B/s] 60%|█████▉    | 262854656/440473133 [00:03<00:02, 68535264.46B/s] 61%|██████    | 269709312/440473133 [00:04<00:02, 68223964.15B/s] 63%|██████▎   | 276571136/440473133 [00:04<00:02, 68340377.98B/s] 64%|██████▍   | 283454464/440473133 [00:04<00:02, 68486696.70B/s] 66%|██████▌   | 290304000/440473133 [00:04<00:02, 67525183.37B/s] 67%|██████▋   | 297154560/440473133 [00:04<00:02, 67815407.96B/s] 69%|██████▉   | 303939584/440473133 [00:04<00:02, 64655161.93B/s] 71%|███████   | 310634496/440473133 [00:04<00:01, 65323863.38B/s] 72%|███████▏  | 317447168/440473133 [00:04<00:01, 66139714.85B/s] 74%|███████▎  | 324282368/440473133 [00:04<00:01, 66787394.82B/s] 75%|███████▌  | 331105280/440473133 [00:04<00:01, 67212024.84B/s] 77%|███████▋  | 337838080/440473133 [00:05<00:01, 67239157.03B/s] 78%|███████▊  | 344578048/440473133 [00:05<00:01, 67285571.19B/s] 80%|███████▉  | 351349760/440473133 [00:05<00:01, 67414332.38B/s] 81%|████████▏ | 358159360/440473133 [00:05<00:01, 67616117.95B/s] 83%|████████▎ | 364998656/440473133 [00:05<00:01, 67844875.39B/s] 84%|████████▍ | 371785728/440473133 [00:05<00:01, 67601666.89B/s] 86%|████████▌ | 378660864/440473133 [00:05<00:00, 67941047.55B/s] 88%|████████▊ | 385494016/440473133 [00:05<00:00, 68057545.36B/s] 89%|████████▉ | 392301568/440473133 [00:05<00:00, 67799150.53B/s] 91%|█████████ | 399083520/440473133 [00:05<00:00, 67800188.63B/s] 92%|█████████▏| 405864448/440473133 [00:06<00:00, 67190297.93B/s] 94%|█████████▎| 412605440/440473133 [00:06<00:00, 67255429.41B/s] 95%|█████████▌| 419367936/440473133 [00:06<00:00, 67365067.06B/s] 97%|█████████▋| 426207232/440473133 [00:06<00:00, 67668758.57B/s] 98%|█████████▊| 432975872/440473133 [00:06<00:00, 65899340.26B/s]100%|█████████▉| 439666688/440473133 [00:06<00:00, 66198468.24B/s]100%|██████████| 440473133/440473133 [00:06<00:00, 67055505.72B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
1064960/7094233 [===>..........................] - ETA: 0s
2105344/7094233 [=======>......................] - ETA: 0s
2883584/7094233 [===========>..................] - ETA: 0s
3670016/7094233 [==============>...............] - ETA: 0s
4497408/7094233 [==================>...........] - ETA: 0s
4988928/7094233 [====================>.........] - ETA: 0s
5750784/7094233 [=======================>......] - ETA: 0s
6275072/7094233 [=========================>....] - ETA: 0s
7094272/7094233 [==============================] - 1s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   6%|▋         | 135/2118 [00:00<00:01, 1349.70it/s]Processing text_left with encode:  26%|██▋       | 561/2118 [00:00<00:00, 1697.35it/s]Processing text_left with encode:  45%|████▍     | 943/2118 [00:00<00:00, 1981.25it/s]Processing text_left with encode:  65%|██████▌   | 1383/2118 [00:00<00:00, 2371.36it/s]Processing text_left with encode:  87%|████████▋ | 1842/2118 [00:00<00:00, 2772.68it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3651.71it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 153/18841 [00:00<00:12, 1524.84it/s]Processing text_right with encode:   2%|▏         | 325/18841 [00:00<00:11, 1576.95it/s]Processing text_right with encode:   3%|▎         | 492/18841 [00:00<00:11, 1602.92it/s]Processing text_right with encode:   4%|▎         | 660/18841 [00:00<00:11, 1623.72it/s]Processing text_right with encode:   4%|▍         | 825/18841 [00:00<00:11, 1630.34it/s]Processing text_right with encode:   5%|▌         | 983/18841 [00:00<00:11, 1614.11it/s]Processing text_right with encode:   6%|▌         | 1136/18841 [00:00<00:11, 1582.93it/s]Processing text_right with encode:   7%|▋         | 1301/18841 [00:00<00:10, 1600.90it/s]Processing text_right with encode:   8%|▊         | 1457/18841 [00:00<00:10, 1587.94it/s]Processing text_right with encode:   9%|▊         | 1624/18841 [00:01<00:10, 1609.93it/s]Processing text_right with encode:  10%|▉         | 1796/18841 [00:01<00:10, 1640.07it/s]Processing text_right with encode:  10%|█         | 1958/18841 [00:01<00:10, 1603.42it/s]Processing text_right with encode:  11%|█         | 2117/18841 [00:01<00:10, 1588.03it/s]Processing text_right with encode:  12%|█▏        | 2290/18841 [00:01<00:10, 1626.68it/s]Processing text_right with encode:  13%|█▎        | 2460/18841 [00:01<00:09, 1644.42it/s]Processing text_right with encode:  14%|█▍        | 2625/18841 [00:01<00:09, 1645.34it/s]Processing text_right with encode:  15%|█▍        | 2812/18841 [00:01<00:09, 1705.40it/s]Processing text_right with encode:  16%|█▌        | 2983/18841 [00:01<00:09, 1677.65it/s]Processing text_right with encode:  17%|█▋        | 3152/18841 [00:01<00:09, 1604.46it/s]Processing text_right with encode:  18%|█▊        | 3314/18841 [00:02<00:09, 1597.73it/s]Processing text_right with encode:  18%|█▊        | 3475/18841 [00:02<00:09, 1581.22it/s]Processing text_right with encode:  19%|█▉        | 3642/18841 [00:02<00:09, 1605.44it/s]Processing text_right with encode:  20%|██        | 3804/18841 [00:02<00:09, 1607.59it/s]Processing text_right with encode:  21%|██        | 3975/18841 [00:02<00:09, 1636.11it/s]Processing text_right with encode:  22%|██▏       | 4141/18841 [00:02<00:08, 1641.11it/s]Processing text_right with encode:  23%|██▎       | 4306/18841 [00:02<00:08, 1617.15it/s]Processing text_right with encode:  24%|██▎       | 4468/18841 [00:02<00:08, 1602.41it/s]Processing text_right with encode:  25%|██▍       | 4629/18841 [00:02<00:08, 1599.66it/s]Processing text_right with encode:  25%|██▌       | 4790/18841 [00:02<00:08, 1596.88it/s]Processing text_right with encode:  26%|██▋       | 4964/18841 [00:03<00:08, 1636.14it/s]Processing text_right with encode:  27%|██▋       | 5138/18841 [00:03<00:08, 1665.42it/s]Processing text_right with encode:  28%|██▊       | 5318/18841 [00:03<00:07, 1703.20it/s]Processing text_right with encode:  29%|██▉       | 5489/18841 [00:03<00:07, 1670.39it/s]Processing text_right with encode:  30%|███       | 5657/18841 [00:03<00:07, 1653.62it/s]Processing text_right with encode:  31%|███       | 5824/18841 [00:03<00:07, 1655.45it/s]Processing text_right with encode:  32%|███▏      | 5994/18841 [00:03<00:07, 1668.45it/s]Processing text_right with encode:  33%|███▎      | 6162/18841 [00:03<00:07, 1663.35it/s]Processing text_right with encode:  34%|███▎      | 6329/18841 [00:03<00:07, 1638.80it/s]Processing text_right with encode:  35%|███▍      | 6506/18841 [00:03<00:07, 1671.32it/s]Processing text_right with encode:  35%|███▌      | 6681/18841 [00:04<00:07, 1690.56it/s]Processing text_right with encode:  36%|███▋      | 6851/18841 [00:04<00:07, 1679.35it/s]Processing text_right with encode:  37%|███▋      | 7020/18841 [00:04<00:07, 1659.15it/s]Processing text_right with encode:  38%|███▊      | 7187/18841 [00:04<00:07, 1639.95it/s]Processing text_right with encode:  39%|███▉      | 7368/18841 [00:04<00:06, 1685.92it/s]Processing text_right with encode:  40%|████      | 7538/18841 [00:04<00:06, 1674.96it/s]Processing text_right with encode:  41%|████      | 7706/18841 [00:04<00:06, 1666.96it/s]Processing text_right with encode:  42%|████▏     | 7874/18841 [00:04<00:06, 1668.60it/s]Processing text_right with encode:  43%|████▎     | 8042/18841 [00:04<00:06, 1615.91it/s]Processing text_right with encode:  44%|████▎     | 8205/18841 [00:05<00:06, 1611.76it/s]Processing text_right with encode:  44%|████▍     | 8381/18841 [00:05<00:06, 1651.26it/s]Processing text_right with encode:  45%|████▌     | 8547/18841 [00:05<00:06, 1617.66it/s]Processing text_right with encode:  46%|████▋     | 8718/18841 [00:05<00:06, 1643.78it/s]Processing text_right with encode:  47%|████▋     | 8893/18841 [00:05<00:05, 1672.88it/s]Processing text_right with encode:  48%|████▊     | 9061/18841 [00:05<00:06, 1629.58it/s]Processing text_right with encode:  49%|████▉     | 9236/18841 [00:05<00:05, 1661.45it/s]Processing text_right with encode:  50%|████▉     | 9403/18841 [00:05<00:05, 1643.24it/s]Processing text_right with encode:  51%|█████     | 9579/18841 [00:05<00:05, 1674.53it/s]Processing text_right with encode:  52%|█████▏    | 9747/18841 [00:05<00:05, 1627.11it/s]Processing text_right with encode:  53%|█████▎    | 9914/18841 [00:06<00:05, 1639.47it/s]Processing text_right with encode:  54%|█████▎    | 10093/18841 [00:06<00:05, 1681.61it/s]Processing text_right with encode:  54%|█████▍    | 10262/18841 [00:06<00:05, 1616.34it/s]Processing text_right with encode:  55%|█████▌    | 10446/18841 [00:06<00:05, 1673.87it/s]Processing text_right with encode:  56%|█████▋    | 10615/18841 [00:06<00:04, 1675.86it/s]Processing text_right with encode:  57%|█████▋    | 10784/18841 [00:06<00:04, 1678.56it/s]Processing text_right with encode:  58%|█████▊    | 10953/18841 [00:06<00:04, 1645.28it/s]Processing text_right with encode:  59%|█████▉    | 11119/18841 [00:06<00:04, 1621.86it/s]Processing text_right with encode:  60%|█████▉    | 11282/18841 [00:06<00:04, 1623.79it/s]Processing text_right with encode:  61%|██████    | 11445/18841 [00:06<00:04, 1582.19it/s]Processing text_right with encode:  62%|██████▏   | 11604/18841 [00:07<00:04, 1578.95it/s]Processing text_right with encode:  62%|██████▏   | 11773/18841 [00:07<00:04, 1610.53it/s]Processing text_right with encode:  63%|██████▎   | 11940/18841 [00:07<00:04, 1624.73it/s]Processing text_right with encode:  64%|██████▍   | 12110/18841 [00:07<00:04, 1646.03it/s]Processing text_right with encode:  65%|██████▌   | 12276/18841 [00:07<00:03, 1648.53it/s]Processing text_right with encode:  66%|██████▌   | 12442/18841 [00:07<00:03, 1647.00it/s]Processing text_right with encode:  67%|██████▋   | 12607/18841 [00:07<00:03, 1631.09it/s]Processing text_right with encode:  68%|██████▊   | 12778/18841 [00:07<00:03, 1652.91it/s]Processing text_right with encode:  69%|██████▊   | 12946/18841 [00:07<00:03, 1659.42it/s]Processing text_right with encode:  70%|██████▉   | 13113/18841 [00:07<00:03, 1649.84it/s]Processing text_right with encode:  70%|███████   | 13280/18841 [00:08<00:03, 1653.49it/s]Processing text_right with encode:  71%|███████▏  | 13449/18841 [00:08<00:03, 1663.19it/s]Processing text_right with encode:  72%|███████▏  | 13619/18841 [00:08<00:03, 1672.65it/s]Processing text_right with encode:  73%|███████▎  | 13796/18841 [00:08<00:02, 1700.19it/s]Processing text_right with encode:  74%|███████▍  | 13971/18841 [00:08<00:02, 1713.04it/s]Processing text_right with encode:  75%|███████▌  | 14143/18841 [00:08<00:02, 1676.59it/s]Processing text_right with encode:  76%|███████▌  | 14311/18841 [00:08<00:02, 1650.51it/s]Processing text_right with encode:  77%|███████▋  | 14477/18841 [00:08<00:02, 1651.17it/s]Processing text_right with encode:  78%|███████▊  | 14653/18841 [00:08<00:02, 1679.04it/s]Processing text_right with encode:  79%|███████▊  | 14828/18841 [00:09<00:02, 1697.69it/s]Processing text_right with encode:  80%|███████▉  | 14998/18841 [00:09<00:02, 1616.41it/s]Processing text_right with encode:  80%|████████  | 15167/18841 [00:09<00:02, 1635.73it/s]Processing text_right with encode:  81%|████████▏ | 15333/18841 [00:09<00:02, 1640.37it/s]Processing text_right with encode:  82%|████████▏ | 15498/18841 [00:09<00:02, 1630.05it/s]Processing text_right with encode:  83%|████████▎ | 15668/18841 [00:09<00:01, 1648.70it/s]Processing text_right with encode:  84%|████████▍ | 15844/18841 [00:09<00:01, 1678.09it/s]Processing text_right with encode:  85%|████████▍ | 16013/18841 [00:09<00:01, 1611.76it/s]Processing text_right with encode:  86%|████████▌ | 16178/18841 [00:09<00:01, 1621.27it/s]Processing text_right with encode:  87%|████████▋ | 16343/18841 [00:09<00:01, 1625.19it/s]Processing text_right with encode:  88%|████████▊ | 16508/18841 [00:10<00:01, 1628.30it/s]Processing text_right with encode:  88%|████████▊ | 16672/18841 [00:10<00:01, 1617.65it/s]Processing text_right with encode:  89%|████████▉ | 16838/18841 [00:10<00:01, 1627.30it/s]Processing text_right with encode:  90%|█████████ | 17001/18841 [00:10<00:01, 1623.49it/s]Processing text_right with encode:  91%|█████████ | 17164/18841 [00:10<00:01, 1622.94it/s]Processing text_right with encode:  92%|█████████▏| 17330/18841 [00:10<00:00, 1633.67it/s]Processing text_right with encode:  93%|█████████▎| 17494/18841 [00:10<00:00, 1617.87it/s]Processing text_right with encode:  94%|█████████▍| 17666/18841 [00:10<00:00, 1643.00it/s]Processing text_right with encode:  95%|█████████▍| 17834/18841 [00:10<00:00, 1653.16it/s]Processing text_right with encode:  96%|█████████▌| 18009/18841 [00:10<00:00, 1680.65it/s]Processing text_right with encode:  97%|█████████▋| 18190/18841 [00:11<00:00, 1714.41it/s]Processing text_right with encode:  97%|█████████▋| 18362/18841 [00:11<00:00, 1677.42it/s]Processing text_right with encode:  98%|█████████▊| 18538/18841 [00:11<00:00, 1700.72it/s]Processing text_right with encode:  99%|█████████▉| 18709/18841 [00:11<00:00, 1675.71it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1645.81it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 492217.19it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 767708.88it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  73%|███████▎  | 459/633 [00:00<00:00, 4583.58it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4522.82it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 174/5961 [00:00<00:03, 1735.45it/s]Processing text_right with encode:   6%|▌         | 343/5961 [00:00<00:03, 1719.63it/s]Processing text_right with encode:   8%|▊         | 501/5961 [00:00<00:03, 1673.91it/s]Processing text_right with encode:  11%|█▏        | 676/5961 [00:00<00:03, 1691.81it/s]Processing text_right with encode:  14%|█▍        | 838/5961 [00:00<00:03, 1668.63it/s]Processing text_right with encode:  17%|█▋        | 1012/5961 [00:00<00:02, 1688.37it/s]Processing text_right with encode:  20%|█▉        | 1179/5961 [00:00<00:02, 1680.18it/s]Processing text_right with encode:  23%|██▎       | 1354/5961 [00:00<00:02, 1697.38it/s]Processing text_right with encode:  26%|██▌       | 1525/5961 [00:00<00:02, 1699.17it/s]Processing text_right with encode:  28%|██▊       | 1692/5961 [00:01<00:02, 1690.11it/s]Processing text_right with encode:  31%|███       | 1857/5961 [00:01<00:02, 1656.41it/s]Processing text_right with encode:  34%|███▍      | 2027/5961 [00:01<00:02, 1668.27it/s]Processing text_right with encode:  37%|███▋      | 2203/5961 [00:01<00:02, 1692.66it/s]Processing text_right with encode:  40%|███▉      | 2371/5961 [00:01<00:02, 1654.41it/s]Processing text_right with encode:  43%|████▎     | 2537/5961 [00:01<00:02, 1656.06it/s]Processing text_right with encode:  45%|████▌     | 2710/5961 [00:01<00:01, 1677.26it/s]Processing text_right with encode:  49%|████▊     | 2895/5961 [00:01<00:01, 1724.41it/s]Processing text_right with encode:  51%|█████▏    | 3068/5961 [00:01<00:01, 1721.39it/s]Processing text_right with encode:  54%|█████▍    | 3243/5961 [00:01<00:01, 1726.77it/s]Processing text_right with encode:  57%|█████▋    | 3416/5961 [00:02<00:01, 1666.80it/s]Processing text_right with encode:  60%|██████    | 3584/5961 [00:02<00:01, 1653.84it/s]Processing text_right with encode:  63%|██████▎   | 3750/5961 [00:02<00:01, 1641.72it/s]Processing text_right with encode:  66%|██████▌   | 3934/5961 [00:02<00:01, 1691.55it/s]Processing text_right with encode:  69%|██████▉   | 4104/5961 [00:02<00:01, 1686.09it/s]Processing text_right with encode:  72%|███████▏  | 4274/5961 [00:02<00:01, 1669.85it/s]Processing text_right with encode:  75%|███████▍  | 4442/5961 [00:02<00:00, 1662.64it/s]Processing text_right with encode:  77%|███████▋  | 4609/5961 [00:02<00:00, 1661.21it/s]Processing text_right with encode:  80%|████████  | 4776/5961 [00:02<00:00, 1636.15it/s]Processing text_right with encode:  83%|████████▎ | 4940/5961 [00:02<00:00, 1619.22it/s]Processing text_right with encode:  86%|████████▌ | 5106/5961 [00:03<00:00, 1629.94it/s]Processing text_right with encode:  89%|████████▊ | 5278/5961 [00:03<00:00, 1655.25it/s]Processing text_right with encode:  91%|█████████▏| 5444/5961 [00:03<00:00, 1618.05it/s]Processing text_right with encode:  94%|█████████▍| 5607/5961 [00:03<00:00, 1595.51it/s]Processing text_right with encode:  97%|█████████▋| 5767/5961 [00:03<00:00, 1580.13it/s]Processing text_right with encode: 100%|█████████▉| 5957/5961 [00:03<00:00, 1663.35it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1667.15it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 281517.81it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 685518.92it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:20<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:20<?, ?it/s, loss=1.044]Epoch 1/1:   1%|          | 1/102 [00:20<34:27, 20.47s/it, loss=1.044]Epoch 1/1:   1%|          | 1/102 [00:38<34:27, 20.47s/it, loss=1.044]Epoch 1/1:   1%|          | 1/102 [00:38<34:27, 20.47s/it, loss=0.965]Epoch 1/1:   2%|▏         | 2/102 [00:38<32:42, 19.63s/it, loss=0.965]Epoch 1/1:   2%|▏         | 2/102 [01:55<32:42, 19.63s/it, loss=0.965]Epoch 1/1:   2%|▏         | 2/102 [01:55<32:42, 19.63s/it, loss=1.143]Epoch 1/1:   3%|▎         | 3/102 [01:55<1:01:06, 37.03s/it, loss=1.143]Epoch 1/1:   3%|▎         | 3/102 [02:57<1:01:06, 37.03s/it, loss=1.143]Epoch 1/1:   3%|▎         | 3/102 [02:57<1:01:06, 37.03s/it, loss=0.935]Epoch 1/1:   4%|▍         | 4/102 [02:57<1:12:44, 44.54s/it, loss=0.935]Epoch 1/1:   4%|▍         | 4/102 [03:56<1:12:44, 44.54s/it, loss=0.935]Epoch 1/1:   4%|▍         | 4/102 [03:56<1:12:44, 44.54s/it, loss=0.935]Epoch 1/1:   5%|▍         | 5/102 [03:56<1:18:52, 48.79s/it, loss=0.935]Epoch 1/1:   5%|▍         | 5/102 [04:21<1:18:52, 48.79s/it, loss=0.935]Epoch 1/1:   5%|▍         | 5/102 [04:21<1:18:52, 48.79s/it, loss=0.903]Epoch 1/1:   6%|▌         | 6/102 [04:21<1:06:30, 41.57s/it, loss=0.903]Epoch 1/1:   6%|▌         | 6/102 [05:29<1:06:30, 41.57s/it, loss=0.903]Epoch 1/1:   6%|▌         | 6/102 [05:29<1:06:30, 41.57s/it, loss=0.948]Epoch 1/1:   7%|▋         | 7/102 [05:29<1:18:39, 49.68s/it, loss=0.948]Epoch 1/1:   7%|▋         | 7/102 [05:54<1:18:39, 49.68s/it, loss=0.948]Epoch 1/1:   7%|▋         | 7/102 [05:54<1:18:39, 49.68s/it, loss=0.776]Epoch 1/1:   8%|▊         | 8/102 [05:54<1:05:56, 42.09s/it, loss=0.776]Epoch 1/1:   8%|▊         | 8/102 [06:20<1:05:56, 42.09s/it, loss=0.776]Epoch 1/1:   8%|▊         | 8/102 [06:20<1:05:56, 42.09s/it, loss=0.750]Epoch 1/1:   9%|▉         | 9/102 [06:20<57:57, 37.39s/it, loss=0.750]  Killed

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
[master cc71aa4] ml_store  && git pull --all
 1 file changed, 64 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 3da4df4...cc71aa4 master -> master (forced update)





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'dataset/vision/MNIST/', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/MNIST/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/MNIST/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1c11332e18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1c11332e18>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f1c11332e18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:27, 113260.17it/s] 91%|█████████▏| 9060352/9912422 [00:00<00:05, 161712.43it/s]9920512it [00:00, 39178721.06it/s]                           
0it [00:00, ?it/s]32768it [00:00, 460856.58it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 141389.08it/s]1654784it [00:00, 10491412.15it/s]                         
0it [00:00, ?it/s]8192it [00:00, 114955.50it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1c1066ab70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1c1066ab70>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f1c1066ab70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0032497462232907613 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.021001387715339662 	 Accuracy: 3
model saves at 3 accuracy
Train Epoch: 2 	 Loss: 0.0026965127090613048 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.014290825605392457 	 Accuracy: 5
model saves at 5 accuracy
Train Epoch: 3 	 Loss: 0.0023638905882835386 	 Accuracy: 0
Train Epoch: 3 	 Loss: 0.015163586378097534 	 Accuracy: 5
Train Epoch: 4 	 Loss: 0.0022793659865856172 	 Accuracy: 1
Train Epoch: 4 	 Loss: 0.017272705256938935 	 Accuracy: 5
Train Epoch: 5 	 Loss: 0.0014684903770685197 	 Accuracy: 1
Train Epoch: 5 	 Loss: 0.011285552471876145 	 Accuracy: 6
model saves at 6 accuracy

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1c1066a950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1c1066a950>
>>>>input_tmp:  None

  function with postional parmater data_info <function get_dataset_torch at 0x7f1c1066a950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/MNIST/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7f1c110e8ac8>

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
<__main__.Model object at 0x7f1c089b1668>

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
[master 75de5d1] ml_store  && git pull --all
 1 file changed, 156 insertions(+)
To github.com:arita37/mlmodels_store.git
   cc71aa4..75de5d1  master -> master





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
[master aa3a71a] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   75de5d1..aa3a71a  master -> master





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
[master 9aeaa29] ml_store  && git pull --all
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   aa3a71a..9aeaa29  master -> master





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
[master 8191955] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   9aeaa29..8191955  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pplm.py 
 Generating text ... 
= Prefix of sentence =
<|endoftext|>The potato

 Unperturbed generated text :

<|endoftext|>The potato-shaped, potato-eating insect of modern times (Ophiocordyceps elegans) has a unique ability to adapt quickly to a wide range of environments. It is able to survive in many different environments, including the Arctic, deserts

 Perturbed generated text :

<|endoftext|>The potato bomb is nothing new. It's been on the news a lot since 9/11. In fact, since the bombing in Paris last November, a bomb has been detonated in every major European country in the European Union.

The bomb in Brussels


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
