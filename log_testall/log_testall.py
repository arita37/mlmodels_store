
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
[master 084aaff] ml_store  && git pull --all
 2 files changed, 63 insertions(+), 7373 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
 + 94ce497...084aaff master -> master (forced update)





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
[master 7caefab] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   084aaff..7caefab  master -> master





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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-26 16:14:16.895973: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-26 16:14:16.911682: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-26 16:14:16.911897: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5637b6721320 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 16:14:16.911915: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 2s - loss: 0.4900 - binary_crossentropy: 7.5582500/500 [==============================] - 1s 1ms/sample - loss: 0.5180 - binary_crossentropy: 7.9901 - val_loss: 0.4880 - val_binary_crossentropy: 7.5274

  #### metrics   #################################################### 
{'MSE': 0.503}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
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
Total params: 478
Trainable params: 478
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2512 - binary_crossentropy: 0.6956500/500 [==============================] - 1s 2ms/sample - loss: 0.2533 - binary_crossentropy: 0.6999 - val_loss: 0.2544 - val_binary_crossentropy: 0.7021

  #### metrics   #################################################### 
{'MSE': 0.25342116950019383}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
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
Total params: 478
Trainable params: 478
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
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
Total params: 652
Trainable params: 652
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2501 - binary_crossentropy: 0.6933 - val_loss: 0.2498 - val_binary_crossentropy: 0.6928

  #### metrics   #################################################### 
{'MSE': 0.24984702687012192}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
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
Total params: 652
Trainable params: 652
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
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2554 - binary_crossentropy: 0.7046500/500 [==============================] - 1s 3ms/sample - loss: 0.2609 - binary_crossentropy: 0.7167 - val_loss: 0.2483 - val_binary_crossentropy: 0.6898

  #### metrics   #################################################### 
{'MSE': 0.24940712939201642}

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
weighted_sequence_layer_9 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
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
Total params: 148
Trainable params: 148
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2492 - binary_crossentropy: 0.6915500/500 [==============================] - 2s 3ms/sample - loss: 0.2528 - binary_crossentropy: 0.6989 - val_loss: 0.2509 - val_binary_crossentropy: 0.6949

  #### metrics   #################################################### 
{'MSE': 0.2515254150755777}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-26 16:15:37.745635: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:15:37.747602: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:15:37.753060: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 16:15:37.762819: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 16:15:37.764627: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 16:15:37.766368: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:15:37.767977: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2519 - val_binary_crossentropy: 0.6970
2020-05-26 16:15:39.030527: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:15:39.032096: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:15:39.036179: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 16:15:39.044108: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-26 16:15:39.045496: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 16:15:39.046785: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:15:39.047942: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2524302365602509}

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
2020-05-26 16:16:02.263382: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:02.264728: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:02.268276: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 16:16:02.274666: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 16:16:02.275699: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 16:16:02.276707: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:02.277631: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2493 - val_binary_crossentropy: 0.6917
2020-05-26 16:16:03.834003: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:03.835135: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:03.837678: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 16:16:03.843131: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-26 16:16:03.843989: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 16:16:03.844803: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:03.845529: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24890417472217882}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-26 16:16:37.230814: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:37.235536: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:37.249884: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 16:16:37.274362: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 16:16:37.279142: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 16:16:37.283386: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:37.287609: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.4828 - binary_crossentropy: 1.1870 - val_loss: 0.2576 - val_binary_crossentropy: 0.7087
2020-05-26 16:16:39.546965: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:39.551299: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:39.562264: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 16:16:39.586658: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-26 16:16:39.590724: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-26 16:16:39.594395: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-26 16:16:39.597750: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2872469898871944}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 740
Trainable params: 740
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3065 - binary_crossentropy: 0.8213500/500 [==============================] - 4s 9ms/sample - loss: 0.2928 - binary_crossentropy: 0.7906 - val_loss: 0.2781 - val_binary_crossentropy: 0.7588

  #### metrics   #################################################### 
{'MSE': 0.2843578324920228}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 740
Trainable params: 740
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
Total params: 257
Trainable params: 257
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2816 - binary_crossentropy: 0.7585500/500 [==============================] - 5s 9ms/sample - loss: 0.2715 - binary_crossentropy: 0.7384 - val_loss: 0.2590 - val_binary_crossentropy: 0.7122

  #### metrics   #################################################### 
{'MSE': 0.2638639028182721}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 2)         10          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_2[0][0]           
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
Total params: 257
Trainable params: 257
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
100/500 [=====>........................] - ETA: 6s - loss: 0.2440 - binary_crossentropy: 0.6807500/500 [==============================] - 5s 9ms/sample - loss: 0.2529 - binary_crossentropy: 0.6991 - val_loss: 0.2535 - val_binary_crossentropy: 0.7001

  #### metrics   #################################################### 
{'MSE': 0.25119394892359226}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         1           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 9s - loss: 0.4902 - binary_crossentropy: 7.5582500/500 [==============================] - 6s 12ms/sample - loss: 0.4742 - binary_crossentropy: 7.3114 - val_loss: 0.4762 - val_binary_crossentropy: 7.3423

  #### metrics   #################################################### 
{'MSE': 0.475}

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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         1           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         1           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
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
Total params: 1,397
Trainable params: 1,397
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2570 - binary_crossentropy: 0.7073500/500 [==============================] - 6s 11ms/sample - loss: 0.2519 - binary_crossentropy: 0.6971 - val_loss: 0.2498 - val_binary_crossentropy: 0.6928

  #### metrics   #################################################### 
{'MSE': 0.25011860214146237}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         28          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_10[0][0]                    
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
Total params: 3,035
Trainable params: 2,955
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2602 - binary_crossentropy: 0.8432500/500 [==============================] - 6s 13ms/sample - loss: 0.2683 - binary_crossentropy: 0.8883 - val_loss: 0.2589 - val_binary_crossentropy: 0.8427

  #### metrics   #################################################### 
{'MSE': 0.26156427217655276}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         28          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         28          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           hash_10[0][0]                    
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
Total params: 3,035
Trainable params: 2,955
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
[master 3b8df70] ml_store  && git pull --all
 1 file changed, 4946 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.118.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + 7308901...3b8df70 master -> master (forced update)





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
[master 2e18d30] ml_store  && git pull --all
 1 file changed, 50 insertions(+)
To github.com:arita37/mlmodels_store.git
   3b8df70..2e18d30  master -> master





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
[master 5c5f81e] ml_store  && git pull --all
 1 file changed, 45 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   2e18d30..5c5f81e  master -> master





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
[master 756c718] ml_store  && git pull --all
 1 file changed, 35 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   5c5f81e..756c718  master -> master





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
[master 85a9dd7] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   756c718..85a9dd7  master -> master





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
[master 61971d8] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   85a9dd7..61971d8  master -> master





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
[master 75e16cb] ml_store  && git pull --all
 1 file changed, 43 insertions(+)
To github.com:arita37/mlmodels_store.git
   61971d8..75e16cb  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2514944/17464789 [===>..........................] - ETA: 0s
11337728/17464789 [==================>...........] - ETA: 0s
16310272/17464789 [===========================>..] - ETA: 0s
16867328/17464789 [===========================>..] - ETA: 0s
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
2020-05-26 16:26:23.409261: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-26 16:26:23.413832: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-05-26 16:26:23.414057: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56469cdb5d20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-26 16:26:23.414077: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7740 - accuracy: 0.4930 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6360 - accuracy: 0.5020
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6398 - accuracy: 0.5017
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6728 - accuracy: 0.4996
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6294 - accuracy: 0.5024
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6321 - accuracy: 0.5023
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6411 - accuracy: 0.5017
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6697 - accuracy: 0.4998
11000/25000 [============>.................] - ETA: 3s - loss: 7.6638 - accuracy: 0.5002
12000/25000 [=============>................] - ETA: 3s - loss: 7.7152 - accuracy: 0.4968
13000/25000 [==============>...............] - ETA: 2s - loss: 7.6820 - accuracy: 0.4990
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7061 - accuracy: 0.4974
15000/25000 [=================>............] - ETA: 2s - loss: 7.7034 - accuracy: 0.4976
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7203 - accuracy: 0.4965
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7253 - accuracy: 0.4962
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7177 - accuracy: 0.4967
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7094 - accuracy: 0.4972
20000/25000 [=======================>......] - ETA: 1s - loss: 7.7234 - accuracy: 0.4963
21000/25000 [========================>.....] - ETA: 0s - loss: 7.7097 - accuracy: 0.4972
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6952 - accuracy: 0.4981
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6733 - accuracy: 0.4996
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
25000/25000 [==============================] - 7s 278us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fd6e396b9b0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fd6e49b1940> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 7s - loss: 7.5056 - accuracy: 0.5105 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.5900 - accuracy: 0.5050
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.6053 - accuracy: 0.5040
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5900 - accuracy: 0.5050
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6027 - accuracy: 0.5042
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5681 - accuracy: 0.5064
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.5976 - accuracy: 0.5045
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.5678 - accuracy: 0.5064
10000/25000 [===========>..................] - ETA: 3s - loss: 7.5624 - accuracy: 0.5068
11000/25000 [============>.................] - ETA: 3s - loss: 7.5649 - accuracy: 0.5066
12000/25000 [=============>................] - ETA: 3s - loss: 7.5976 - accuracy: 0.5045
13000/25000 [==============>...............] - ETA: 2s - loss: 7.5994 - accuracy: 0.5044
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6250 - accuracy: 0.5027
15000/25000 [=================>............] - ETA: 2s - loss: 7.6636 - accuracy: 0.5002
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6580 - accuracy: 0.5006
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6621 - accuracy: 0.5003
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6658 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6594 - accuracy: 0.5005
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6754 - accuracy: 0.4994
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6882 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6706 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6692 - accuracy: 0.4998
25000/25000 [==============================] - 7s 280us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 10s - loss: 7.8966 - accuracy: 0.4850
 2000/25000 [=>............................] - ETA: 7s - loss: 7.9580 - accuracy: 0.4810 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7688 - accuracy: 0.4933
 4000/25000 [===>..........................] - ETA: 5s - loss: 7.7586 - accuracy: 0.4940
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7954 - accuracy: 0.4916
 6000/25000 [======>.......................] - ETA: 4s - loss: 7.7458 - accuracy: 0.4948
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7367 - accuracy: 0.4954
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7720 - accuracy: 0.4931
 9000/25000 [=========>....................] - ETA: 3s - loss: 7.8131 - accuracy: 0.4904
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7970 - accuracy: 0.4915
11000/25000 [============>.................] - ETA: 3s - loss: 7.7809 - accuracy: 0.4925
12000/25000 [=============>................] - ETA: 3s - loss: 7.7395 - accuracy: 0.4952
13000/25000 [==============>...............] - ETA: 2s - loss: 7.7303 - accuracy: 0.4958
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7477 - accuracy: 0.4947
15000/25000 [=================>............] - ETA: 2s - loss: 7.7464 - accuracy: 0.4948
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7385 - accuracy: 0.4953
17000/25000 [===================>..........] - ETA: 1s - loss: 7.7072 - accuracy: 0.4974
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7143 - accuracy: 0.4969
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6868 - accuracy: 0.4987
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6919 - accuracy: 0.4983
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6885 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6750 - accuracy: 0.4995
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6773 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
25000/25000 [==============================] - 7s 272us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master 04069ba] ml_store  && git pull --all
 1 file changed, 317 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 367ddfc...04069ba master -> master (forced update)





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

13/13 [==============================] - 2s 136ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master f37ea74] ml_store  && git pull --all
 1 file changed, 124 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   04069ba..f37ea74  master -> master





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
 1417216/11490434 [==>...........................] - ETA: 0s
 5455872/11490434 [=============>................] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:55 - loss: 2.3051 - categorical_accuracy: 0.0000e+00
   96/60000 [..............................] - ETA: 3:38 - loss: 2.2531 - categorical_accuracy: 0.1771    
  160/60000 [..............................] - ETA: 2:47 - loss: 2.1885 - categorical_accuracy: 0.2000
  224/60000 [..............................] - ETA: 2:23 - loss: 2.1621 - categorical_accuracy: 0.2188
  288/60000 [..............................] - ETA: 2:09 - loss: 2.0505 - categorical_accuracy: 0.2812
  320/60000 [..............................] - ETA: 2:06 - loss: 2.0258 - categorical_accuracy: 0.2937
  384/60000 [..............................] - ETA: 1:58 - loss: 1.9767 - categorical_accuracy: 0.3203
  448/60000 [..............................] - ETA: 1:52 - loss: 1.9361 - categorical_accuracy: 0.3415
  512/60000 [..............................] - ETA: 1:48 - loss: 1.9045 - categorical_accuracy: 0.3535
  576/60000 [..............................] - ETA: 1:46 - loss: 1.8327 - categorical_accuracy: 0.3837
  640/60000 [..............................] - ETA: 1:43 - loss: 1.7790 - categorical_accuracy: 0.4078
  704/60000 [..............................] - ETA: 1:41 - loss: 1.7312 - categorical_accuracy: 0.4205
  768/60000 [..............................] - ETA: 1:40 - loss: 1.6812 - categorical_accuracy: 0.4349
  832/60000 [..............................] - ETA: 1:39 - loss: 1.6292 - categorical_accuracy: 0.4519
  896/60000 [..............................] - ETA: 1:37 - loss: 1.6019 - categorical_accuracy: 0.4609
  960/60000 [..............................] - ETA: 1:36 - loss: 1.5449 - categorical_accuracy: 0.4792
 1024/60000 [..............................] - ETA: 1:35 - loss: 1.5116 - categorical_accuracy: 0.4922
 1088/60000 [..............................] - ETA: 1:34 - loss: 1.4837 - categorical_accuracy: 0.5009
 1152/60000 [..............................] - ETA: 1:33 - loss: 1.4669 - categorical_accuracy: 0.5043
 1216/60000 [..............................] - ETA: 1:32 - loss: 1.4343 - categorical_accuracy: 0.5206
 1280/60000 [..............................] - ETA: 1:32 - loss: 1.3869 - categorical_accuracy: 0.5391
 1344/60000 [..............................] - ETA: 1:31 - loss: 1.3504 - categorical_accuracy: 0.5499
 1408/60000 [..............................] - ETA: 1:30 - loss: 1.3244 - categorical_accuracy: 0.5582
 1472/60000 [..............................] - ETA: 1:30 - loss: 1.3019 - categorical_accuracy: 0.5645
 1536/60000 [..............................] - ETA: 1:29 - loss: 1.2844 - categorical_accuracy: 0.5723
 1600/60000 [..............................] - ETA: 1:29 - loss: 1.2554 - categorical_accuracy: 0.5844
 1664/60000 [..............................] - ETA: 1:28 - loss: 1.2268 - categorical_accuracy: 0.5925
 1728/60000 [..............................] - ETA: 1:28 - loss: 1.2090 - categorical_accuracy: 0.5978
 1792/60000 [..............................] - ETA: 1:28 - loss: 1.1911 - categorical_accuracy: 0.6044
 1856/60000 [..............................] - ETA: 1:28 - loss: 1.1635 - categorical_accuracy: 0.6148
 1920/60000 [..............................] - ETA: 1:27 - loss: 1.1459 - categorical_accuracy: 0.6214
 1984/60000 [..............................] - ETA: 1:27 - loss: 1.1274 - categorical_accuracy: 0.6265
 2048/60000 [>.............................] - ETA: 1:27 - loss: 1.1217 - categorical_accuracy: 0.6313
 2112/60000 [>.............................] - ETA: 1:27 - loss: 1.1060 - categorical_accuracy: 0.6373
 2176/60000 [>.............................] - ETA: 1:26 - loss: 1.0981 - categorical_accuracy: 0.6397
 2240/60000 [>.............................] - ETA: 1:26 - loss: 1.0821 - categorical_accuracy: 0.6455
 2304/60000 [>.............................] - ETA: 1:26 - loss: 1.0655 - categorical_accuracy: 0.6510
 2336/60000 [>.............................] - ETA: 1:26 - loss: 1.0572 - categorical_accuracy: 0.6537
 2368/60000 [>.............................] - ETA: 1:26 - loss: 1.0505 - categorical_accuracy: 0.6567
 2432/60000 [>.............................] - ETA: 1:26 - loss: 1.0345 - categorical_accuracy: 0.6620
 2496/60000 [>.............................] - ETA: 1:26 - loss: 1.0145 - categorical_accuracy: 0.6695
 2560/60000 [>.............................] - ETA: 1:25 - loss: 0.9968 - categorical_accuracy: 0.6754
 2624/60000 [>.............................] - ETA: 1:25 - loss: 0.9877 - categorical_accuracy: 0.6791
 2656/60000 [>.............................] - ETA: 1:25 - loss: 0.9817 - categorical_accuracy: 0.6811
 2720/60000 [>.............................] - ETA: 1:25 - loss: 0.9737 - categorical_accuracy: 0.6827
 2784/60000 [>.............................] - ETA: 1:25 - loss: 0.9608 - categorical_accuracy: 0.6853
 2848/60000 [>.............................] - ETA: 1:25 - loss: 0.9451 - categorical_accuracy: 0.6903
 2912/60000 [>.............................] - ETA: 1:24 - loss: 0.9371 - categorical_accuracy: 0.6923
 2976/60000 [>.............................] - ETA: 1:24 - loss: 0.9242 - categorical_accuracy: 0.6969
 3040/60000 [>.............................] - ETA: 1:24 - loss: 0.9164 - categorical_accuracy: 0.7007
 3104/60000 [>.............................] - ETA: 1:24 - loss: 0.9050 - categorical_accuracy: 0.7046
 3168/60000 [>.............................] - ETA: 1:24 - loss: 0.8926 - categorical_accuracy: 0.7090
 3232/60000 [>.............................] - ETA: 1:23 - loss: 0.8816 - categorical_accuracy: 0.7123
 3296/60000 [>.............................] - ETA: 1:23 - loss: 0.8730 - categorical_accuracy: 0.7154
 3360/60000 [>.............................] - ETA: 1:23 - loss: 0.8642 - categorical_accuracy: 0.7193
 3424/60000 [>.............................] - ETA: 1:23 - loss: 0.8509 - categorical_accuracy: 0.7237
 3488/60000 [>.............................] - ETA: 1:23 - loss: 0.8410 - categorical_accuracy: 0.7282
 3552/60000 [>.............................] - ETA: 1:22 - loss: 0.8353 - categorical_accuracy: 0.7300
 3616/60000 [>.............................] - ETA: 1:22 - loss: 0.8283 - categorical_accuracy: 0.7323
 3680/60000 [>.............................] - ETA: 1:22 - loss: 0.8193 - categorical_accuracy: 0.7348
 3744/60000 [>.............................] - ETA: 1:22 - loss: 0.8096 - categorical_accuracy: 0.7380
 3808/60000 [>.............................] - ETA: 1:22 - loss: 0.8011 - categorical_accuracy: 0.7405
 3872/60000 [>.............................] - ETA: 1:22 - loss: 0.7915 - categorical_accuracy: 0.7438
 3936/60000 [>.............................] - ETA: 1:22 - loss: 0.7830 - categorical_accuracy: 0.7467
 4000/60000 [=>............................] - ETA: 1:22 - loss: 0.7761 - categorical_accuracy: 0.7487
 4064/60000 [=>............................] - ETA: 1:21 - loss: 0.7729 - categorical_accuracy: 0.7502
 4128/60000 [=>............................] - ETA: 1:21 - loss: 0.7663 - categorical_accuracy: 0.7519
 4192/60000 [=>............................] - ETA: 1:21 - loss: 0.7589 - categorical_accuracy: 0.7543
 4256/60000 [=>............................] - ETA: 1:21 - loss: 0.7520 - categorical_accuracy: 0.7563
 4320/60000 [=>............................] - ETA: 1:21 - loss: 0.7447 - categorical_accuracy: 0.7588
 4384/60000 [=>............................] - ETA: 1:21 - loss: 0.7370 - categorical_accuracy: 0.7614
 4448/60000 [=>............................] - ETA: 1:20 - loss: 0.7321 - categorical_accuracy: 0.7628
 4512/60000 [=>............................] - ETA: 1:20 - loss: 0.7248 - categorical_accuracy: 0.7653
 4544/60000 [=>............................] - ETA: 1:20 - loss: 0.7214 - categorical_accuracy: 0.7663
 4608/60000 [=>............................] - ETA: 1:20 - loss: 0.7141 - categorical_accuracy: 0.7687
 4672/60000 [=>............................] - ETA: 1:20 - loss: 0.7080 - categorical_accuracy: 0.7708
 4736/60000 [=>............................] - ETA: 1:20 - loss: 0.7055 - categorical_accuracy: 0.7728
 4800/60000 [=>............................] - ETA: 1:20 - loss: 0.7008 - categorical_accuracy: 0.7746
 4864/60000 [=>............................] - ETA: 1:20 - loss: 0.6938 - categorical_accuracy: 0.7771
 4928/60000 [=>............................] - ETA: 1:20 - loss: 0.6871 - categorical_accuracy: 0.7794
 4992/60000 [=>............................] - ETA: 1:19 - loss: 0.6832 - categorical_accuracy: 0.7810
 5056/60000 [=>............................] - ETA: 1:19 - loss: 0.6779 - categorical_accuracy: 0.7830
 5120/60000 [=>............................] - ETA: 1:19 - loss: 0.6729 - categorical_accuracy: 0.7848
 5184/60000 [=>............................] - ETA: 1:19 - loss: 0.6675 - categorical_accuracy: 0.7865
 5216/60000 [=>............................] - ETA: 1:19 - loss: 0.6649 - categorical_accuracy: 0.7872
 5280/60000 [=>............................] - ETA: 1:19 - loss: 0.6590 - categorical_accuracy: 0.7890
 5344/60000 [=>............................] - ETA: 1:19 - loss: 0.6571 - categorical_accuracy: 0.7902
 5408/60000 [=>............................] - ETA: 1:19 - loss: 0.6533 - categorical_accuracy: 0.7916
 5472/60000 [=>............................] - ETA: 1:18 - loss: 0.6478 - categorical_accuracy: 0.7935
 5536/60000 [=>............................] - ETA: 1:18 - loss: 0.6428 - categorical_accuracy: 0.7946
 5600/60000 [=>............................] - ETA: 1:18 - loss: 0.6386 - categorical_accuracy: 0.7959
 5664/60000 [=>............................] - ETA: 1:18 - loss: 0.6342 - categorical_accuracy: 0.7975
 5728/60000 [=>............................] - ETA: 1:18 - loss: 0.6296 - categorical_accuracy: 0.7989
 5792/60000 [=>............................] - ETA: 1:18 - loss: 0.6255 - categorical_accuracy: 0.8001
 5856/60000 [=>............................] - ETA: 1:18 - loss: 0.6209 - categorical_accuracy: 0.8014
 5888/60000 [=>............................] - ETA: 1:18 - loss: 0.6194 - categorical_accuracy: 0.8020
 5952/60000 [=>............................] - ETA: 1:18 - loss: 0.6149 - categorical_accuracy: 0.8034
 6016/60000 [==>...........................] - ETA: 1:18 - loss: 0.6145 - categorical_accuracy: 0.8042
 6080/60000 [==>...........................] - ETA: 1:17 - loss: 0.6137 - categorical_accuracy: 0.8049
 6144/60000 [==>...........................] - ETA: 1:17 - loss: 0.6090 - categorical_accuracy: 0.8065
 6208/60000 [==>...........................] - ETA: 1:17 - loss: 0.6051 - categorical_accuracy: 0.8080
 6272/60000 [==>...........................] - ETA: 1:17 - loss: 0.6006 - categorical_accuracy: 0.8093
 6336/60000 [==>...........................] - ETA: 1:17 - loss: 0.5982 - categorical_accuracy: 0.8104
 6400/60000 [==>...........................] - ETA: 1:17 - loss: 0.5949 - categorical_accuracy: 0.8116
 6464/60000 [==>...........................] - ETA: 1:17 - loss: 0.5922 - categorical_accuracy: 0.8127
 6528/60000 [==>...........................] - ETA: 1:16 - loss: 0.5882 - categorical_accuracy: 0.8137
 6592/60000 [==>...........................] - ETA: 1:16 - loss: 0.5856 - categorical_accuracy: 0.8142
 6656/60000 [==>...........................] - ETA: 1:16 - loss: 0.5810 - categorical_accuracy: 0.8155
 6720/60000 [==>...........................] - ETA: 1:16 - loss: 0.5767 - categorical_accuracy: 0.8168
 6784/60000 [==>...........................] - ETA: 1:16 - loss: 0.5737 - categorical_accuracy: 0.8178
 6848/60000 [==>...........................] - ETA: 1:16 - loss: 0.5720 - categorical_accuracy: 0.8182
 6912/60000 [==>...........................] - ETA: 1:16 - loss: 0.5694 - categorical_accuracy: 0.8190
 6976/60000 [==>...........................] - ETA: 1:16 - loss: 0.5670 - categorical_accuracy: 0.8197
 7040/60000 [==>...........................] - ETA: 1:15 - loss: 0.5657 - categorical_accuracy: 0.8202
 7104/60000 [==>...........................] - ETA: 1:15 - loss: 0.5620 - categorical_accuracy: 0.8214
 7168/60000 [==>...........................] - ETA: 1:15 - loss: 0.5587 - categorical_accuracy: 0.8221
 7232/60000 [==>...........................] - ETA: 1:15 - loss: 0.5556 - categorical_accuracy: 0.8229
 7296/60000 [==>...........................] - ETA: 1:15 - loss: 0.5527 - categorical_accuracy: 0.8239
 7360/60000 [==>...........................] - ETA: 1:15 - loss: 0.5504 - categorical_accuracy: 0.8243
 7424/60000 [==>...........................] - ETA: 1:15 - loss: 0.5490 - categorical_accuracy: 0.8249
 7488/60000 [==>...........................] - ETA: 1:14 - loss: 0.5456 - categorical_accuracy: 0.8259
 7552/60000 [==>...........................] - ETA: 1:14 - loss: 0.5425 - categorical_accuracy: 0.8265
 7616/60000 [==>...........................] - ETA: 1:14 - loss: 0.5400 - categorical_accuracy: 0.8275
 7680/60000 [==>...........................] - ETA: 1:14 - loss: 0.5374 - categorical_accuracy: 0.8283
 7744/60000 [==>...........................] - ETA: 1:14 - loss: 0.5352 - categorical_accuracy: 0.8290
 7808/60000 [==>...........................] - ETA: 1:14 - loss: 0.5317 - categorical_accuracy: 0.8303
 7872/60000 [==>...........................] - ETA: 1:14 - loss: 0.5285 - categorical_accuracy: 0.8314
 7936/60000 [==>...........................] - ETA: 1:14 - loss: 0.5266 - categorical_accuracy: 0.8320
 8000/60000 [===>..........................] - ETA: 1:14 - loss: 0.5241 - categorical_accuracy: 0.8329
 8064/60000 [===>..........................] - ETA: 1:14 - loss: 0.5218 - categorical_accuracy: 0.8337
 8128/60000 [===>..........................] - ETA: 1:14 - loss: 0.5192 - categorical_accuracy: 0.8346
 8192/60000 [===>..........................] - ETA: 1:13 - loss: 0.5166 - categorical_accuracy: 0.8356
 8256/60000 [===>..........................] - ETA: 1:13 - loss: 0.5140 - categorical_accuracy: 0.8365
 8320/60000 [===>..........................] - ETA: 1:13 - loss: 0.5110 - categorical_accuracy: 0.8374
 8384/60000 [===>..........................] - ETA: 1:13 - loss: 0.5093 - categorical_accuracy: 0.8381
 8448/60000 [===>..........................] - ETA: 1:13 - loss: 0.5062 - categorical_accuracy: 0.8390
 8512/60000 [===>..........................] - ETA: 1:13 - loss: 0.5037 - categorical_accuracy: 0.8398
 8576/60000 [===>..........................] - ETA: 1:13 - loss: 0.5015 - categorical_accuracy: 0.8404
 8640/60000 [===>..........................] - ETA: 1:13 - loss: 0.4993 - categorical_accuracy: 0.8412
 8704/60000 [===>..........................] - ETA: 1:13 - loss: 0.4977 - categorical_accuracy: 0.8418
 8768/60000 [===>..........................] - ETA: 1:13 - loss: 0.4950 - categorical_accuracy: 0.8428
 8832/60000 [===>..........................] - ETA: 1:12 - loss: 0.4940 - categorical_accuracy: 0.8434
 8896/60000 [===>..........................] - ETA: 1:12 - loss: 0.4925 - categorical_accuracy: 0.8439
 8928/60000 [===>..........................] - ETA: 1:12 - loss: 0.4919 - categorical_accuracy: 0.8440
 8992/60000 [===>..........................] - ETA: 1:12 - loss: 0.4896 - categorical_accuracy: 0.8444
 9056/60000 [===>..........................] - ETA: 1:12 - loss: 0.4877 - categorical_accuracy: 0.8449
 9120/60000 [===>..........................] - ETA: 1:12 - loss: 0.4857 - categorical_accuracy: 0.8456
 9184/60000 [===>..........................] - ETA: 1:12 - loss: 0.4832 - categorical_accuracy: 0.8464
 9248/60000 [===>..........................] - ETA: 1:12 - loss: 0.4813 - categorical_accuracy: 0.8470
 9312/60000 [===>..........................] - ETA: 1:12 - loss: 0.4807 - categorical_accuracy: 0.8473
 9376/60000 [===>..........................] - ETA: 1:12 - loss: 0.4798 - categorical_accuracy: 0.8474
 9440/60000 [===>..........................] - ETA: 1:11 - loss: 0.4781 - categorical_accuracy: 0.8479
 9504/60000 [===>..........................] - ETA: 1:11 - loss: 0.4762 - categorical_accuracy: 0.8485
 9568/60000 [===>..........................] - ETA: 1:11 - loss: 0.4756 - categorical_accuracy: 0.8485
 9632/60000 [===>..........................] - ETA: 1:11 - loss: 0.4736 - categorical_accuracy: 0.8488
 9696/60000 [===>..........................] - ETA: 1:11 - loss: 0.4716 - categorical_accuracy: 0.8496
 9760/60000 [===>..........................] - ETA: 1:11 - loss: 0.4698 - categorical_accuracy: 0.8504
 9792/60000 [===>..........................] - ETA: 1:11 - loss: 0.4690 - categorical_accuracy: 0.8508
 9856/60000 [===>..........................] - ETA: 1:11 - loss: 0.4667 - categorical_accuracy: 0.8516
 9920/60000 [===>..........................] - ETA: 1:11 - loss: 0.4650 - categorical_accuracy: 0.8522
 9984/60000 [===>..........................] - ETA: 1:11 - loss: 0.4631 - categorical_accuracy: 0.8529
10048/60000 [====>.........................] - ETA: 1:11 - loss: 0.4608 - categorical_accuracy: 0.8537
10112/60000 [====>.........................] - ETA: 1:10 - loss: 0.4596 - categorical_accuracy: 0.8540
10176/60000 [====>.........................] - ETA: 1:10 - loss: 0.4581 - categorical_accuracy: 0.8545
10240/60000 [====>.........................] - ETA: 1:10 - loss: 0.4565 - categorical_accuracy: 0.8549
10304/60000 [====>.........................] - ETA: 1:10 - loss: 0.4545 - categorical_accuracy: 0.8556
10368/60000 [====>.........................] - ETA: 1:10 - loss: 0.4523 - categorical_accuracy: 0.8564
10432/60000 [====>.........................] - ETA: 1:10 - loss: 0.4499 - categorical_accuracy: 0.8572
10496/60000 [====>.........................] - ETA: 1:10 - loss: 0.4483 - categorical_accuracy: 0.8579
10560/60000 [====>.........................] - ETA: 1:10 - loss: 0.4467 - categorical_accuracy: 0.8582
10624/60000 [====>.........................] - ETA: 1:10 - loss: 0.4455 - categorical_accuracy: 0.8588
10688/60000 [====>.........................] - ETA: 1:10 - loss: 0.4450 - categorical_accuracy: 0.8592
10752/60000 [====>.........................] - ETA: 1:09 - loss: 0.4440 - categorical_accuracy: 0.8597
10816/60000 [====>.........................] - ETA: 1:09 - loss: 0.4419 - categorical_accuracy: 0.8604
10880/60000 [====>.........................] - ETA: 1:09 - loss: 0.4400 - categorical_accuracy: 0.8610
10944/60000 [====>.........................] - ETA: 1:09 - loss: 0.4388 - categorical_accuracy: 0.8614
11008/60000 [====>.........................] - ETA: 1:09 - loss: 0.4368 - categorical_accuracy: 0.8620
11072/60000 [====>.........................] - ETA: 1:09 - loss: 0.4355 - categorical_accuracy: 0.8623
11136/60000 [====>.........................] - ETA: 1:09 - loss: 0.4338 - categorical_accuracy: 0.8628
11200/60000 [====>.........................] - ETA: 1:09 - loss: 0.4328 - categorical_accuracy: 0.8631
11264/60000 [====>.........................] - ETA: 1:09 - loss: 0.4325 - categorical_accuracy: 0.8635
11328/60000 [====>.........................] - ETA: 1:09 - loss: 0.4314 - categorical_accuracy: 0.8639
11392/60000 [====>.........................] - ETA: 1:08 - loss: 0.4297 - categorical_accuracy: 0.8644
11456/60000 [====>.........................] - ETA: 1:08 - loss: 0.4291 - categorical_accuracy: 0.8646
11520/60000 [====>.........................] - ETA: 1:08 - loss: 0.4278 - categorical_accuracy: 0.8652
11552/60000 [====>.........................] - ETA: 1:08 - loss: 0.4267 - categorical_accuracy: 0.8656
11616/60000 [====>.........................] - ETA: 1:08 - loss: 0.4247 - categorical_accuracy: 0.8662
11680/60000 [====>.........................] - ETA: 1:08 - loss: 0.4233 - categorical_accuracy: 0.8667
11744/60000 [====>.........................] - ETA: 1:08 - loss: 0.4216 - categorical_accuracy: 0.8671
11808/60000 [====>.........................] - ETA: 1:08 - loss: 0.4200 - categorical_accuracy: 0.8676
11872/60000 [====>.........................] - ETA: 1:08 - loss: 0.4184 - categorical_accuracy: 0.8681
11936/60000 [====>.........................] - ETA: 1:08 - loss: 0.4170 - categorical_accuracy: 0.8685
11968/60000 [====>.........................] - ETA: 1:08 - loss: 0.4161 - categorical_accuracy: 0.8689
12032/60000 [=====>........................] - ETA: 1:07 - loss: 0.4139 - categorical_accuracy: 0.8696
12096/60000 [=====>........................] - ETA: 1:07 - loss: 0.4136 - categorical_accuracy: 0.8699
12160/60000 [=====>........................] - ETA: 1:07 - loss: 0.4122 - categorical_accuracy: 0.8702
12224/60000 [=====>........................] - ETA: 1:07 - loss: 0.4121 - categorical_accuracy: 0.8703
12288/60000 [=====>........................] - ETA: 1:07 - loss: 0.4109 - categorical_accuracy: 0.8707
12352/60000 [=====>........................] - ETA: 1:07 - loss: 0.4097 - categorical_accuracy: 0.8710
12416/60000 [=====>........................] - ETA: 1:07 - loss: 0.4086 - categorical_accuracy: 0.8715
12480/60000 [=====>........................] - ETA: 1:07 - loss: 0.4069 - categorical_accuracy: 0.8720
12544/60000 [=====>........................] - ETA: 1:07 - loss: 0.4058 - categorical_accuracy: 0.8724
12608/60000 [=====>........................] - ETA: 1:07 - loss: 0.4046 - categorical_accuracy: 0.8729
12672/60000 [=====>........................] - ETA: 1:07 - loss: 0.4031 - categorical_accuracy: 0.8733
12736/60000 [=====>........................] - ETA: 1:06 - loss: 0.4032 - categorical_accuracy: 0.8732
12800/60000 [=====>........................] - ETA: 1:06 - loss: 0.4020 - categorical_accuracy: 0.8735
12864/60000 [=====>........................] - ETA: 1:06 - loss: 0.4010 - categorical_accuracy: 0.8739
12928/60000 [=====>........................] - ETA: 1:06 - loss: 0.3999 - categorical_accuracy: 0.8741
12992/60000 [=====>........................] - ETA: 1:06 - loss: 0.3986 - categorical_accuracy: 0.8745
13056/60000 [=====>........................] - ETA: 1:06 - loss: 0.3984 - categorical_accuracy: 0.8748
13120/60000 [=====>........................] - ETA: 1:06 - loss: 0.3972 - categorical_accuracy: 0.8752
13184/60000 [=====>........................] - ETA: 1:06 - loss: 0.3956 - categorical_accuracy: 0.8757
13248/60000 [=====>........................] - ETA: 1:06 - loss: 0.3946 - categorical_accuracy: 0.8761
13312/60000 [=====>........................] - ETA: 1:06 - loss: 0.3935 - categorical_accuracy: 0.8763
13344/60000 [=====>........................] - ETA: 1:05 - loss: 0.3927 - categorical_accuracy: 0.8766
13408/60000 [=====>........................] - ETA: 1:05 - loss: 0.3918 - categorical_accuracy: 0.8768
13440/60000 [=====>........................] - ETA: 1:05 - loss: 0.3918 - categorical_accuracy: 0.8769
13504/60000 [=====>........................] - ETA: 1:05 - loss: 0.3905 - categorical_accuracy: 0.8773
13568/60000 [=====>........................] - ETA: 1:05 - loss: 0.3896 - categorical_accuracy: 0.8775
13632/60000 [=====>........................] - ETA: 1:05 - loss: 0.3884 - categorical_accuracy: 0.8779
13696/60000 [=====>........................] - ETA: 1:05 - loss: 0.3869 - categorical_accuracy: 0.8784
13760/60000 [=====>........................] - ETA: 1:05 - loss: 0.3866 - categorical_accuracy: 0.8786
13824/60000 [=====>........................] - ETA: 1:05 - loss: 0.3855 - categorical_accuracy: 0.8790
13888/60000 [=====>........................] - ETA: 1:05 - loss: 0.3840 - categorical_accuracy: 0.8795
13952/60000 [=====>........................] - ETA: 1:05 - loss: 0.3825 - categorical_accuracy: 0.8801
14016/60000 [======>.......................] - ETA: 1:05 - loss: 0.3820 - categorical_accuracy: 0.8803
14080/60000 [======>.......................] - ETA: 1:04 - loss: 0.3812 - categorical_accuracy: 0.8805
14144/60000 [======>.......................] - ETA: 1:04 - loss: 0.3807 - categorical_accuracy: 0.8807
14208/60000 [======>.......................] - ETA: 1:04 - loss: 0.3801 - categorical_accuracy: 0.8810
14272/60000 [======>.......................] - ETA: 1:04 - loss: 0.3790 - categorical_accuracy: 0.8812
14336/60000 [======>.......................] - ETA: 1:04 - loss: 0.3780 - categorical_accuracy: 0.8816
14400/60000 [======>.......................] - ETA: 1:04 - loss: 0.3770 - categorical_accuracy: 0.8819
14464/60000 [======>.......................] - ETA: 1:04 - loss: 0.3763 - categorical_accuracy: 0.8821
14528/60000 [======>.......................] - ETA: 1:04 - loss: 0.3759 - categorical_accuracy: 0.8822
14592/60000 [======>.......................] - ETA: 1:04 - loss: 0.3747 - categorical_accuracy: 0.8827
14656/60000 [======>.......................] - ETA: 1:04 - loss: 0.3735 - categorical_accuracy: 0.8831
14720/60000 [======>.......................] - ETA: 1:03 - loss: 0.3724 - categorical_accuracy: 0.8835
14784/60000 [======>.......................] - ETA: 1:03 - loss: 0.3714 - categorical_accuracy: 0.8837
14848/60000 [======>.......................] - ETA: 1:03 - loss: 0.3703 - categorical_accuracy: 0.8841
14912/60000 [======>.......................] - ETA: 1:03 - loss: 0.3707 - categorical_accuracy: 0.8841
14976/60000 [======>.......................] - ETA: 1:03 - loss: 0.3699 - categorical_accuracy: 0.8844
15040/60000 [======>.......................] - ETA: 1:03 - loss: 0.3691 - categorical_accuracy: 0.8846
15104/60000 [======>.......................] - ETA: 1:03 - loss: 0.3683 - categorical_accuracy: 0.8849
15168/60000 [======>.......................] - ETA: 1:03 - loss: 0.3669 - categorical_accuracy: 0.8853
15232/60000 [======>.......................] - ETA: 1:03 - loss: 0.3660 - categorical_accuracy: 0.8856
15296/60000 [======>.......................] - ETA: 1:03 - loss: 0.3654 - categorical_accuracy: 0.8859
15360/60000 [======>.......................] - ETA: 1:02 - loss: 0.3645 - categorical_accuracy: 0.8861
15424/60000 [======>.......................] - ETA: 1:02 - loss: 0.3636 - categorical_accuracy: 0.8864
15488/60000 [======>.......................] - ETA: 1:02 - loss: 0.3626 - categorical_accuracy: 0.8868
15552/60000 [======>.......................] - ETA: 1:02 - loss: 0.3618 - categorical_accuracy: 0.8870
15616/60000 [======>.......................] - ETA: 1:02 - loss: 0.3608 - categorical_accuracy: 0.8872
15680/60000 [======>.......................] - ETA: 1:02 - loss: 0.3601 - categorical_accuracy: 0.8874
15744/60000 [======>.......................] - ETA: 1:02 - loss: 0.3589 - categorical_accuracy: 0.8877
15808/60000 [======>.......................] - ETA: 1:02 - loss: 0.3578 - categorical_accuracy: 0.8880
15872/60000 [======>.......................] - ETA: 1:02 - loss: 0.3569 - categorical_accuracy: 0.8884
15936/60000 [======>.......................] - ETA: 1:02 - loss: 0.3561 - categorical_accuracy: 0.8887
16000/60000 [=======>......................] - ETA: 1:01 - loss: 0.3553 - categorical_accuracy: 0.8890
16064/60000 [=======>......................] - ETA: 1:01 - loss: 0.3548 - categorical_accuracy: 0.8893
16128/60000 [=======>......................] - ETA: 1:01 - loss: 0.3541 - categorical_accuracy: 0.8894
16192/60000 [=======>......................] - ETA: 1:01 - loss: 0.3537 - categorical_accuracy: 0.8895
16256/60000 [=======>......................] - ETA: 1:01 - loss: 0.3530 - categorical_accuracy: 0.8896
16320/60000 [=======>......................] - ETA: 1:01 - loss: 0.3522 - categorical_accuracy: 0.8898
16384/60000 [=======>......................] - ETA: 1:01 - loss: 0.3516 - categorical_accuracy: 0.8900
16448/60000 [=======>......................] - ETA: 1:01 - loss: 0.3508 - categorical_accuracy: 0.8903
16512/60000 [=======>......................] - ETA: 1:01 - loss: 0.3499 - categorical_accuracy: 0.8906
16576/60000 [=======>......................] - ETA: 1:01 - loss: 0.3489 - categorical_accuracy: 0.8908
16640/60000 [=======>......................] - ETA: 1:01 - loss: 0.3478 - categorical_accuracy: 0.8912
16704/60000 [=======>......................] - ETA: 1:00 - loss: 0.3467 - categorical_accuracy: 0.8916
16736/60000 [=======>......................] - ETA: 1:00 - loss: 0.3464 - categorical_accuracy: 0.8917
16800/60000 [=======>......................] - ETA: 1:00 - loss: 0.3456 - categorical_accuracy: 0.8918
16864/60000 [=======>......................] - ETA: 1:00 - loss: 0.3449 - categorical_accuracy: 0.8921
16928/60000 [=======>......................] - ETA: 1:00 - loss: 0.3440 - categorical_accuracy: 0.8924
16992/60000 [=======>......................] - ETA: 1:00 - loss: 0.3434 - categorical_accuracy: 0.8925
17056/60000 [=======>......................] - ETA: 1:00 - loss: 0.3431 - categorical_accuracy: 0.8926
17120/60000 [=======>......................] - ETA: 1:00 - loss: 0.3430 - categorical_accuracy: 0.8928
17184/60000 [=======>......................] - ETA: 1:00 - loss: 0.3421 - categorical_accuracy: 0.8930
17248/60000 [=======>......................] - ETA: 1:00 - loss: 0.3411 - categorical_accuracy: 0.8932
17312/60000 [=======>......................] - ETA: 1:00 - loss: 0.3402 - categorical_accuracy: 0.8934
17376/60000 [=======>......................] - ETA: 59s - loss: 0.3397 - categorical_accuracy: 0.8935 
17440/60000 [=======>......................] - ETA: 59s - loss: 0.3392 - categorical_accuracy: 0.8936
17504/60000 [=======>......................] - ETA: 59s - loss: 0.3391 - categorical_accuracy: 0.8938
17568/60000 [=======>......................] - ETA: 59s - loss: 0.3384 - categorical_accuracy: 0.8940
17632/60000 [=======>......................] - ETA: 59s - loss: 0.3381 - categorical_accuracy: 0.8940
17696/60000 [=======>......................] - ETA: 59s - loss: 0.3372 - categorical_accuracy: 0.8942
17760/60000 [=======>......................] - ETA: 59s - loss: 0.3362 - categorical_accuracy: 0.8946
17824/60000 [=======>......................] - ETA: 59s - loss: 0.3355 - categorical_accuracy: 0.8948
17888/60000 [=======>......................] - ETA: 59s - loss: 0.3347 - categorical_accuracy: 0.8951
17952/60000 [=======>......................] - ETA: 59s - loss: 0.3341 - categorical_accuracy: 0.8953
18016/60000 [========>.....................] - ETA: 59s - loss: 0.3342 - categorical_accuracy: 0.8953
18048/60000 [========>.....................] - ETA: 59s - loss: 0.3340 - categorical_accuracy: 0.8953
18112/60000 [========>.....................] - ETA: 58s - loss: 0.3333 - categorical_accuracy: 0.8956
18176/60000 [========>.....................] - ETA: 58s - loss: 0.3330 - categorical_accuracy: 0.8956
18240/60000 [========>.....................] - ETA: 58s - loss: 0.3324 - categorical_accuracy: 0.8958
18304/60000 [========>.....................] - ETA: 58s - loss: 0.3316 - categorical_accuracy: 0.8961
18368/60000 [========>.....................] - ETA: 58s - loss: 0.3310 - categorical_accuracy: 0.8964
18432/60000 [========>.....................] - ETA: 58s - loss: 0.3312 - categorical_accuracy: 0.8964
18496/60000 [========>.....................] - ETA: 58s - loss: 0.3304 - categorical_accuracy: 0.8967
18560/60000 [========>.....................] - ETA: 58s - loss: 0.3295 - categorical_accuracy: 0.8969
18624/60000 [========>.....................] - ETA: 58s - loss: 0.3286 - categorical_accuracy: 0.8972
18688/60000 [========>.....................] - ETA: 58s - loss: 0.3278 - categorical_accuracy: 0.8975
18752/60000 [========>.....................] - ETA: 58s - loss: 0.3274 - categorical_accuracy: 0.8976
18816/60000 [========>.....................] - ETA: 57s - loss: 0.3268 - categorical_accuracy: 0.8977
18880/60000 [========>.....................] - ETA: 57s - loss: 0.3262 - categorical_accuracy: 0.8978
18912/60000 [========>.....................] - ETA: 57s - loss: 0.3257 - categorical_accuracy: 0.8980
18976/60000 [========>.....................] - ETA: 57s - loss: 0.3247 - categorical_accuracy: 0.8983
19040/60000 [========>.....................] - ETA: 57s - loss: 0.3243 - categorical_accuracy: 0.8985
19104/60000 [========>.....................] - ETA: 57s - loss: 0.3237 - categorical_accuracy: 0.8987
19168/60000 [========>.....................] - ETA: 57s - loss: 0.3228 - categorical_accuracy: 0.8989
19232/60000 [========>.....................] - ETA: 57s - loss: 0.3218 - categorical_accuracy: 0.8993
19296/60000 [========>.....................] - ETA: 57s - loss: 0.3214 - categorical_accuracy: 0.8995
19360/60000 [========>.....................] - ETA: 57s - loss: 0.3206 - categorical_accuracy: 0.8998
19424/60000 [========>.....................] - ETA: 57s - loss: 0.3198 - categorical_accuracy: 0.9000
19456/60000 [========>.....................] - ETA: 57s - loss: 0.3195 - categorical_accuracy: 0.9000
19520/60000 [========>.....................] - ETA: 57s - loss: 0.3192 - categorical_accuracy: 0.9001
19584/60000 [========>.....................] - ETA: 56s - loss: 0.3187 - categorical_accuracy: 0.9003
19648/60000 [========>.....................] - ETA: 56s - loss: 0.3177 - categorical_accuracy: 0.9006
19712/60000 [========>.....................] - ETA: 56s - loss: 0.3180 - categorical_accuracy: 0.9007
19776/60000 [========>.....................] - ETA: 56s - loss: 0.3172 - categorical_accuracy: 0.9009
19840/60000 [========>.....................] - ETA: 56s - loss: 0.3165 - categorical_accuracy: 0.9012
19904/60000 [========>.....................] - ETA: 56s - loss: 0.3168 - categorical_accuracy: 0.9013
19968/60000 [========>.....................] - ETA: 56s - loss: 0.3164 - categorical_accuracy: 0.9014
20032/60000 [=========>....................] - ETA: 56s - loss: 0.3164 - categorical_accuracy: 0.9015
20096/60000 [=========>....................] - ETA: 56s - loss: 0.3158 - categorical_accuracy: 0.9017
20160/60000 [=========>....................] - ETA: 56s - loss: 0.3155 - categorical_accuracy: 0.9018
20224/60000 [=========>....................] - ETA: 56s - loss: 0.3152 - categorical_accuracy: 0.9019
20288/60000 [=========>....................] - ETA: 55s - loss: 0.3147 - categorical_accuracy: 0.9022
20352/60000 [=========>....................] - ETA: 55s - loss: 0.3140 - categorical_accuracy: 0.9024
20416/60000 [=========>....................] - ETA: 55s - loss: 0.3134 - categorical_accuracy: 0.9026
20448/60000 [=========>....................] - ETA: 55s - loss: 0.3130 - categorical_accuracy: 0.9028
20480/60000 [=========>....................] - ETA: 55s - loss: 0.3126 - categorical_accuracy: 0.9029
20512/60000 [=========>....................] - ETA: 55s - loss: 0.3123 - categorical_accuracy: 0.9030
20576/60000 [=========>....................] - ETA: 55s - loss: 0.3118 - categorical_accuracy: 0.9031
20640/60000 [=========>....................] - ETA: 55s - loss: 0.3110 - categorical_accuracy: 0.9034
20704/60000 [=========>....................] - ETA: 55s - loss: 0.3104 - categorical_accuracy: 0.9036
20768/60000 [=========>....................] - ETA: 55s - loss: 0.3103 - categorical_accuracy: 0.9036
20832/60000 [=========>....................] - ETA: 55s - loss: 0.3103 - categorical_accuracy: 0.9036
20896/60000 [=========>....................] - ETA: 55s - loss: 0.3095 - categorical_accuracy: 0.9038
20960/60000 [=========>....................] - ETA: 54s - loss: 0.3091 - categorical_accuracy: 0.9039
21024/60000 [=========>....................] - ETA: 54s - loss: 0.3083 - categorical_accuracy: 0.9042
21088/60000 [=========>....................] - ETA: 54s - loss: 0.3076 - categorical_accuracy: 0.9044
21152/60000 [=========>....................] - ETA: 54s - loss: 0.3071 - categorical_accuracy: 0.9046
21216/60000 [=========>....................] - ETA: 54s - loss: 0.3067 - categorical_accuracy: 0.9046
21280/60000 [=========>....................] - ETA: 54s - loss: 0.3060 - categorical_accuracy: 0.9048
21344/60000 [=========>....................] - ETA: 54s - loss: 0.3055 - categorical_accuracy: 0.9050
21408/60000 [=========>....................] - ETA: 54s - loss: 0.3053 - categorical_accuracy: 0.9050
21472/60000 [=========>....................] - ETA: 54s - loss: 0.3050 - categorical_accuracy: 0.9050
21536/60000 [=========>....................] - ETA: 54s - loss: 0.3047 - categorical_accuracy: 0.9051
21600/60000 [=========>....................] - ETA: 54s - loss: 0.3044 - categorical_accuracy: 0.9052
21664/60000 [=========>....................] - ETA: 54s - loss: 0.3037 - categorical_accuracy: 0.9054
21728/60000 [=========>....................] - ETA: 53s - loss: 0.3032 - categorical_accuracy: 0.9055
21792/60000 [=========>....................] - ETA: 53s - loss: 0.3026 - categorical_accuracy: 0.9057
21856/60000 [=========>....................] - ETA: 53s - loss: 0.3023 - categorical_accuracy: 0.9058
21920/60000 [=========>....................] - ETA: 53s - loss: 0.3022 - categorical_accuracy: 0.9058
21984/60000 [=========>....................] - ETA: 53s - loss: 0.3014 - categorical_accuracy: 0.9061
22016/60000 [==========>...................] - ETA: 53s - loss: 0.3011 - categorical_accuracy: 0.9062
22048/60000 [==========>...................] - ETA: 53s - loss: 0.3008 - categorical_accuracy: 0.9062
22112/60000 [==========>...................] - ETA: 53s - loss: 0.3003 - categorical_accuracy: 0.9063
22176/60000 [==========>...................] - ETA: 53s - loss: 0.2998 - categorical_accuracy: 0.9064
22240/60000 [==========>...................] - ETA: 53s - loss: 0.2995 - categorical_accuracy: 0.9065
22304/60000 [==========>...................] - ETA: 53s - loss: 0.2989 - categorical_accuracy: 0.9067
22368/60000 [==========>...................] - ETA: 53s - loss: 0.2982 - categorical_accuracy: 0.9069
22432/60000 [==========>...................] - ETA: 52s - loss: 0.2987 - categorical_accuracy: 0.9069
22496/60000 [==========>...................] - ETA: 52s - loss: 0.2982 - categorical_accuracy: 0.9071
22560/60000 [==========>...................] - ETA: 52s - loss: 0.2983 - categorical_accuracy: 0.9072
22624/60000 [==========>...................] - ETA: 52s - loss: 0.2977 - categorical_accuracy: 0.9074
22688/60000 [==========>...................] - ETA: 52s - loss: 0.2970 - categorical_accuracy: 0.9076
22752/60000 [==========>...................] - ETA: 52s - loss: 0.2964 - categorical_accuracy: 0.9078
22816/60000 [==========>...................] - ETA: 52s - loss: 0.2960 - categorical_accuracy: 0.9080
22848/60000 [==========>...................] - ETA: 52s - loss: 0.2956 - categorical_accuracy: 0.9081
22912/60000 [==========>...................] - ETA: 52s - loss: 0.2950 - categorical_accuracy: 0.9083
22976/60000 [==========>...................] - ETA: 52s - loss: 0.2946 - categorical_accuracy: 0.9084
23040/60000 [==========>...................] - ETA: 52s - loss: 0.2944 - categorical_accuracy: 0.9085
23072/60000 [==========>...................] - ETA: 52s - loss: 0.2940 - categorical_accuracy: 0.9086
23136/60000 [==========>...................] - ETA: 51s - loss: 0.2935 - categorical_accuracy: 0.9088
23200/60000 [==========>...................] - ETA: 51s - loss: 0.2933 - categorical_accuracy: 0.9088
23264/60000 [==========>...................] - ETA: 51s - loss: 0.2932 - categorical_accuracy: 0.9088
23328/60000 [==========>...................] - ETA: 51s - loss: 0.2926 - categorical_accuracy: 0.9090
23392/60000 [==========>...................] - ETA: 51s - loss: 0.2923 - categorical_accuracy: 0.9091
23456/60000 [==========>...................] - ETA: 51s - loss: 0.2921 - categorical_accuracy: 0.9092
23520/60000 [==========>...................] - ETA: 51s - loss: 0.2918 - categorical_accuracy: 0.9094
23584/60000 [==========>...................] - ETA: 51s - loss: 0.2913 - categorical_accuracy: 0.9096
23648/60000 [==========>...................] - ETA: 51s - loss: 0.2909 - categorical_accuracy: 0.9097
23680/60000 [==========>...................] - ETA: 51s - loss: 0.2907 - categorical_accuracy: 0.9098
23744/60000 [==========>...................] - ETA: 51s - loss: 0.2904 - categorical_accuracy: 0.9098
23808/60000 [==========>...................] - ETA: 51s - loss: 0.2898 - categorical_accuracy: 0.9100
23872/60000 [==========>...................] - ETA: 50s - loss: 0.2894 - categorical_accuracy: 0.9101
23936/60000 [==========>...................] - ETA: 50s - loss: 0.2891 - categorical_accuracy: 0.9102
24000/60000 [===========>..................] - ETA: 50s - loss: 0.2886 - categorical_accuracy: 0.9104
24064/60000 [===========>..................] - ETA: 50s - loss: 0.2881 - categorical_accuracy: 0.9105
24128/60000 [===========>..................] - ETA: 50s - loss: 0.2880 - categorical_accuracy: 0.9106
24192/60000 [===========>..................] - ETA: 50s - loss: 0.2878 - categorical_accuracy: 0.9107
24224/60000 [===========>..................] - ETA: 50s - loss: 0.2876 - categorical_accuracy: 0.9107
24288/60000 [===========>..................] - ETA: 50s - loss: 0.2870 - categorical_accuracy: 0.9109
24352/60000 [===========>..................] - ETA: 50s - loss: 0.2868 - categorical_accuracy: 0.9110
24416/60000 [===========>..................] - ETA: 50s - loss: 0.2862 - categorical_accuracy: 0.9112
24480/60000 [===========>..................] - ETA: 50s - loss: 0.2856 - categorical_accuracy: 0.9114
24544/60000 [===========>..................] - ETA: 49s - loss: 0.2853 - categorical_accuracy: 0.9115
24608/60000 [===========>..................] - ETA: 49s - loss: 0.2848 - categorical_accuracy: 0.9117
24672/60000 [===========>..................] - ETA: 49s - loss: 0.2846 - categorical_accuracy: 0.9118
24736/60000 [===========>..................] - ETA: 49s - loss: 0.2845 - categorical_accuracy: 0.9118
24800/60000 [===========>..................] - ETA: 49s - loss: 0.2841 - categorical_accuracy: 0.9120
24832/60000 [===========>..................] - ETA: 49s - loss: 0.2839 - categorical_accuracy: 0.9120
24896/60000 [===========>..................] - ETA: 49s - loss: 0.2834 - categorical_accuracy: 0.9122
24960/60000 [===========>..................] - ETA: 49s - loss: 0.2828 - categorical_accuracy: 0.9124
25024/60000 [===========>..................] - ETA: 49s - loss: 0.2822 - categorical_accuracy: 0.9126
25088/60000 [===========>..................] - ETA: 49s - loss: 0.2816 - categorical_accuracy: 0.9128
25152/60000 [===========>..................] - ETA: 49s - loss: 0.2813 - categorical_accuracy: 0.9128
25216/60000 [===========>..................] - ETA: 49s - loss: 0.2810 - categorical_accuracy: 0.9130
25280/60000 [===========>..................] - ETA: 48s - loss: 0.2803 - categorical_accuracy: 0.9132
25344/60000 [===========>..................] - ETA: 48s - loss: 0.2808 - categorical_accuracy: 0.9131
25376/60000 [===========>..................] - ETA: 48s - loss: 0.2805 - categorical_accuracy: 0.9132
25440/60000 [===========>..................] - ETA: 48s - loss: 0.2805 - categorical_accuracy: 0.9132
25504/60000 [===========>..................] - ETA: 48s - loss: 0.2803 - categorical_accuracy: 0.9132
25568/60000 [===========>..................] - ETA: 48s - loss: 0.2798 - categorical_accuracy: 0.9134
25632/60000 [===========>..................] - ETA: 48s - loss: 0.2793 - categorical_accuracy: 0.9135
25696/60000 [===========>..................] - ETA: 48s - loss: 0.2793 - categorical_accuracy: 0.9136
25760/60000 [===========>..................] - ETA: 48s - loss: 0.2788 - categorical_accuracy: 0.9137
25824/60000 [===========>..................] - ETA: 48s - loss: 0.2783 - categorical_accuracy: 0.9139
25888/60000 [===========>..................] - ETA: 48s - loss: 0.2780 - categorical_accuracy: 0.9140
25952/60000 [===========>..................] - ETA: 47s - loss: 0.2776 - categorical_accuracy: 0.9141
26016/60000 [============>.................] - ETA: 47s - loss: 0.2772 - categorical_accuracy: 0.9143
26080/60000 [============>.................] - ETA: 47s - loss: 0.2772 - categorical_accuracy: 0.9143
26144/60000 [============>.................] - ETA: 47s - loss: 0.2771 - categorical_accuracy: 0.9144
26208/60000 [============>.................] - ETA: 47s - loss: 0.2770 - categorical_accuracy: 0.9145
26272/60000 [============>.................] - ETA: 47s - loss: 0.2766 - categorical_accuracy: 0.9146
26336/60000 [============>.................] - ETA: 47s - loss: 0.2762 - categorical_accuracy: 0.9147
26368/60000 [============>.................] - ETA: 47s - loss: 0.2759 - categorical_accuracy: 0.9148
26400/60000 [============>.................] - ETA: 47s - loss: 0.2757 - categorical_accuracy: 0.9148
26432/60000 [============>.................] - ETA: 47s - loss: 0.2757 - categorical_accuracy: 0.9149
26496/60000 [============>.................] - ETA: 47s - loss: 0.2757 - categorical_accuracy: 0.9149
26560/60000 [============>.................] - ETA: 47s - loss: 0.2755 - categorical_accuracy: 0.9150
26624/60000 [============>.................] - ETA: 47s - loss: 0.2752 - categorical_accuracy: 0.9151
26688/60000 [============>.................] - ETA: 46s - loss: 0.2751 - categorical_accuracy: 0.9151
26752/60000 [============>.................] - ETA: 46s - loss: 0.2751 - categorical_accuracy: 0.9151
26816/60000 [============>.................] - ETA: 46s - loss: 0.2747 - categorical_accuracy: 0.9153
26880/60000 [============>.................] - ETA: 46s - loss: 0.2743 - categorical_accuracy: 0.9154
26944/60000 [============>.................] - ETA: 46s - loss: 0.2738 - categorical_accuracy: 0.9156
27008/60000 [============>.................] - ETA: 46s - loss: 0.2737 - categorical_accuracy: 0.9156
27072/60000 [============>.................] - ETA: 46s - loss: 0.2734 - categorical_accuracy: 0.9157
27136/60000 [============>.................] - ETA: 46s - loss: 0.2731 - categorical_accuracy: 0.9158
27200/60000 [============>.................] - ETA: 46s - loss: 0.2728 - categorical_accuracy: 0.9159
27264/60000 [============>.................] - ETA: 46s - loss: 0.2723 - categorical_accuracy: 0.9160
27328/60000 [============>.................] - ETA: 45s - loss: 0.2721 - categorical_accuracy: 0.9160
27360/60000 [============>.................] - ETA: 45s - loss: 0.2718 - categorical_accuracy: 0.9161
27392/60000 [============>.................] - ETA: 45s - loss: 0.2716 - categorical_accuracy: 0.9162
27456/60000 [============>.................] - ETA: 45s - loss: 0.2711 - categorical_accuracy: 0.9163
27520/60000 [============>.................] - ETA: 45s - loss: 0.2707 - categorical_accuracy: 0.9165
27584/60000 [============>.................] - ETA: 45s - loss: 0.2704 - categorical_accuracy: 0.9165
27648/60000 [============>.................] - ETA: 45s - loss: 0.2700 - categorical_accuracy: 0.9166
27712/60000 [============>.................] - ETA: 45s - loss: 0.2700 - categorical_accuracy: 0.9167
27776/60000 [============>.................] - ETA: 45s - loss: 0.2696 - categorical_accuracy: 0.9167
27840/60000 [============>.................] - ETA: 45s - loss: 0.2693 - categorical_accuracy: 0.9168
27904/60000 [============>.................] - ETA: 45s - loss: 0.2690 - categorical_accuracy: 0.9169
27968/60000 [============>.................] - ETA: 45s - loss: 0.2685 - categorical_accuracy: 0.9170
28032/60000 [=============>................] - ETA: 44s - loss: 0.2683 - categorical_accuracy: 0.9171
28096/60000 [=============>................] - ETA: 44s - loss: 0.2679 - categorical_accuracy: 0.9172
28160/60000 [=============>................] - ETA: 44s - loss: 0.2678 - categorical_accuracy: 0.9173
28224/60000 [=============>................] - ETA: 44s - loss: 0.2676 - categorical_accuracy: 0.9173
28288/60000 [=============>................] - ETA: 44s - loss: 0.2674 - categorical_accuracy: 0.9175
28352/60000 [=============>................] - ETA: 44s - loss: 0.2672 - categorical_accuracy: 0.9175
28416/60000 [=============>................] - ETA: 44s - loss: 0.2667 - categorical_accuracy: 0.9177
28480/60000 [=============>................] - ETA: 44s - loss: 0.2665 - categorical_accuracy: 0.9178
28544/60000 [=============>................] - ETA: 44s - loss: 0.2661 - categorical_accuracy: 0.9179
28608/60000 [=============>................] - ETA: 44s - loss: 0.2661 - categorical_accuracy: 0.9179
28640/60000 [=============>................] - ETA: 44s - loss: 0.2659 - categorical_accuracy: 0.9180
28704/60000 [=============>................] - ETA: 43s - loss: 0.2654 - categorical_accuracy: 0.9181
28768/60000 [=============>................] - ETA: 43s - loss: 0.2652 - categorical_accuracy: 0.9182
28832/60000 [=============>................] - ETA: 43s - loss: 0.2648 - categorical_accuracy: 0.9183
28896/60000 [=============>................] - ETA: 43s - loss: 0.2645 - categorical_accuracy: 0.9184
28960/60000 [=============>................] - ETA: 43s - loss: 0.2640 - categorical_accuracy: 0.9186
29024/60000 [=============>................] - ETA: 43s - loss: 0.2638 - categorical_accuracy: 0.9187
29088/60000 [=============>................] - ETA: 43s - loss: 0.2635 - categorical_accuracy: 0.9188
29152/60000 [=============>................] - ETA: 43s - loss: 0.2631 - categorical_accuracy: 0.9189
29184/60000 [=============>................] - ETA: 43s - loss: 0.2630 - categorical_accuracy: 0.9189
29248/60000 [=============>................] - ETA: 43s - loss: 0.2632 - categorical_accuracy: 0.9189
29280/60000 [=============>................] - ETA: 43s - loss: 0.2629 - categorical_accuracy: 0.9190
29344/60000 [=============>................] - ETA: 43s - loss: 0.2627 - categorical_accuracy: 0.9191
29408/60000 [=============>................] - ETA: 42s - loss: 0.2624 - categorical_accuracy: 0.9191
29472/60000 [=============>................] - ETA: 42s - loss: 0.2621 - categorical_accuracy: 0.9192
29536/60000 [=============>................] - ETA: 42s - loss: 0.2623 - categorical_accuracy: 0.9193
29600/60000 [=============>................] - ETA: 42s - loss: 0.2619 - categorical_accuracy: 0.9194
29664/60000 [=============>................] - ETA: 42s - loss: 0.2617 - categorical_accuracy: 0.9194
29728/60000 [=============>................] - ETA: 42s - loss: 0.2613 - categorical_accuracy: 0.9195
29792/60000 [=============>................] - ETA: 42s - loss: 0.2613 - categorical_accuracy: 0.9195
29856/60000 [=============>................] - ETA: 42s - loss: 0.2609 - categorical_accuracy: 0.9196
29920/60000 [=============>................] - ETA: 42s - loss: 0.2608 - categorical_accuracy: 0.9197
29984/60000 [=============>................] - ETA: 42s - loss: 0.2604 - categorical_accuracy: 0.9198
30048/60000 [==============>...............] - ETA: 42s - loss: 0.2601 - categorical_accuracy: 0.9199
30080/60000 [==============>...............] - ETA: 42s - loss: 0.2598 - categorical_accuracy: 0.9200
30144/60000 [==============>...............] - ETA: 41s - loss: 0.2596 - categorical_accuracy: 0.9201
30208/60000 [==============>...............] - ETA: 41s - loss: 0.2594 - categorical_accuracy: 0.9202
30272/60000 [==============>...............] - ETA: 41s - loss: 0.2593 - categorical_accuracy: 0.9202
30336/60000 [==============>...............] - ETA: 41s - loss: 0.2588 - categorical_accuracy: 0.9203
30400/60000 [==============>...............] - ETA: 41s - loss: 0.2586 - categorical_accuracy: 0.9204
30464/60000 [==============>...............] - ETA: 41s - loss: 0.2584 - categorical_accuracy: 0.9204
30528/60000 [==============>...............] - ETA: 41s - loss: 0.2580 - categorical_accuracy: 0.9205
30592/60000 [==============>...............] - ETA: 41s - loss: 0.2576 - categorical_accuracy: 0.9207
30656/60000 [==============>...............] - ETA: 41s - loss: 0.2573 - categorical_accuracy: 0.9207
30720/60000 [==============>...............] - ETA: 41s - loss: 0.2569 - categorical_accuracy: 0.9208
30752/60000 [==============>...............] - ETA: 41s - loss: 0.2568 - categorical_accuracy: 0.9208
30816/60000 [==============>...............] - ETA: 41s - loss: 0.2566 - categorical_accuracy: 0.9209
30848/60000 [==============>...............] - ETA: 40s - loss: 0.2564 - categorical_accuracy: 0.9209
30912/60000 [==============>...............] - ETA: 40s - loss: 0.2562 - categorical_accuracy: 0.9210
30944/60000 [==============>...............] - ETA: 40s - loss: 0.2559 - categorical_accuracy: 0.9211
31008/60000 [==============>...............] - ETA: 40s - loss: 0.2560 - categorical_accuracy: 0.9211
31072/60000 [==============>...............] - ETA: 40s - loss: 0.2556 - categorical_accuracy: 0.9212
31136/60000 [==============>...............] - ETA: 40s - loss: 0.2553 - categorical_accuracy: 0.9213
31200/60000 [==============>...............] - ETA: 40s - loss: 0.2551 - categorical_accuracy: 0.9214
31264/60000 [==============>...............] - ETA: 40s - loss: 0.2546 - categorical_accuracy: 0.9215
31328/60000 [==============>...............] - ETA: 40s - loss: 0.2543 - categorical_accuracy: 0.9217
31392/60000 [==============>...............] - ETA: 40s - loss: 0.2541 - categorical_accuracy: 0.9217
31456/60000 [==============>...............] - ETA: 40s - loss: 0.2537 - categorical_accuracy: 0.9218
31520/60000 [==============>...............] - ETA: 40s - loss: 0.2535 - categorical_accuracy: 0.9219
31584/60000 [==============>...............] - ETA: 39s - loss: 0.2531 - categorical_accuracy: 0.9220
31648/60000 [==============>...............] - ETA: 39s - loss: 0.2530 - categorical_accuracy: 0.9221
31712/60000 [==============>...............] - ETA: 39s - loss: 0.2525 - categorical_accuracy: 0.9222
31776/60000 [==============>...............] - ETA: 39s - loss: 0.2524 - categorical_accuracy: 0.9222
31840/60000 [==============>...............] - ETA: 39s - loss: 0.2523 - categorical_accuracy: 0.9222
31904/60000 [==============>...............] - ETA: 39s - loss: 0.2520 - categorical_accuracy: 0.9223
31968/60000 [==============>...............] - ETA: 39s - loss: 0.2517 - categorical_accuracy: 0.9224
32032/60000 [===============>..............] - ETA: 39s - loss: 0.2514 - categorical_accuracy: 0.9225
32096/60000 [===============>..............] - ETA: 39s - loss: 0.2511 - categorical_accuracy: 0.9226
32128/60000 [===============>..............] - ETA: 39s - loss: 0.2509 - categorical_accuracy: 0.9227
32192/60000 [===============>..............] - ETA: 39s - loss: 0.2505 - categorical_accuracy: 0.9228
32256/60000 [===============>..............] - ETA: 38s - loss: 0.2502 - categorical_accuracy: 0.9229
32320/60000 [===============>..............] - ETA: 38s - loss: 0.2498 - categorical_accuracy: 0.9230
32384/60000 [===============>..............] - ETA: 38s - loss: 0.2495 - categorical_accuracy: 0.9231
32448/60000 [===============>..............] - ETA: 38s - loss: 0.2492 - categorical_accuracy: 0.9232
32512/60000 [===============>..............] - ETA: 38s - loss: 0.2487 - categorical_accuracy: 0.9234
32576/60000 [===============>..............] - ETA: 38s - loss: 0.2487 - categorical_accuracy: 0.9234
32640/60000 [===============>..............] - ETA: 38s - loss: 0.2484 - categorical_accuracy: 0.9235
32704/60000 [===============>..............] - ETA: 38s - loss: 0.2481 - categorical_accuracy: 0.9235
32768/60000 [===============>..............] - ETA: 38s - loss: 0.2481 - categorical_accuracy: 0.9236
32832/60000 [===============>..............] - ETA: 38s - loss: 0.2480 - categorical_accuracy: 0.9236
32896/60000 [===============>..............] - ETA: 38s - loss: 0.2476 - categorical_accuracy: 0.9237
32960/60000 [===============>..............] - ETA: 37s - loss: 0.2473 - categorical_accuracy: 0.9238
33024/60000 [===============>..............] - ETA: 37s - loss: 0.2471 - categorical_accuracy: 0.9239
33088/60000 [===============>..............] - ETA: 37s - loss: 0.2468 - categorical_accuracy: 0.9239
33152/60000 [===============>..............] - ETA: 37s - loss: 0.2466 - categorical_accuracy: 0.9240
33216/60000 [===============>..............] - ETA: 37s - loss: 0.2462 - categorical_accuracy: 0.9241
33280/60000 [===============>..............] - ETA: 37s - loss: 0.2460 - categorical_accuracy: 0.9242
33344/60000 [===============>..............] - ETA: 37s - loss: 0.2456 - categorical_accuracy: 0.9242
33408/60000 [===============>..............] - ETA: 37s - loss: 0.2454 - categorical_accuracy: 0.9244
33472/60000 [===============>..............] - ETA: 37s - loss: 0.2451 - categorical_accuracy: 0.9245
33536/60000 [===============>..............] - ETA: 37s - loss: 0.2450 - categorical_accuracy: 0.9246
33600/60000 [===============>..............] - ETA: 37s - loss: 0.2448 - categorical_accuracy: 0.9247
33664/60000 [===============>..............] - ETA: 36s - loss: 0.2449 - categorical_accuracy: 0.9247
33728/60000 [===============>..............] - ETA: 36s - loss: 0.2446 - categorical_accuracy: 0.9248
33792/60000 [===============>..............] - ETA: 36s - loss: 0.2447 - categorical_accuracy: 0.9248
33856/60000 [===============>..............] - ETA: 36s - loss: 0.2445 - categorical_accuracy: 0.9248
33920/60000 [===============>..............] - ETA: 36s - loss: 0.2442 - categorical_accuracy: 0.9249
33984/60000 [===============>..............] - ETA: 36s - loss: 0.2440 - categorical_accuracy: 0.9250
34048/60000 [================>.............] - ETA: 36s - loss: 0.2438 - categorical_accuracy: 0.9251
34112/60000 [================>.............] - ETA: 36s - loss: 0.2434 - categorical_accuracy: 0.9252
34176/60000 [================>.............] - ETA: 36s - loss: 0.2432 - categorical_accuracy: 0.9252
34240/60000 [================>.............] - ETA: 36s - loss: 0.2428 - categorical_accuracy: 0.9254
34304/60000 [================>.............] - ETA: 36s - loss: 0.2427 - categorical_accuracy: 0.9253
34368/60000 [================>.............] - ETA: 35s - loss: 0.2424 - categorical_accuracy: 0.9254
34432/60000 [================>.............] - ETA: 35s - loss: 0.2420 - categorical_accuracy: 0.9255
34496/60000 [================>.............] - ETA: 35s - loss: 0.2420 - categorical_accuracy: 0.9255
34560/60000 [================>.............] - ETA: 35s - loss: 0.2419 - categorical_accuracy: 0.9255
34624/60000 [================>.............] - ETA: 35s - loss: 0.2417 - categorical_accuracy: 0.9256
34688/60000 [================>.............] - ETA: 35s - loss: 0.2415 - categorical_accuracy: 0.9257
34752/60000 [================>.............] - ETA: 35s - loss: 0.2412 - categorical_accuracy: 0.9257
34816/60000 [================>.............] - ETA: 35s - loss: 0.2409 - categorical_accuracy: 0.9258
34880/60000 [================>.............] - ETA: 35s - loss: 0.2406 - categorical_accuracy: 0.9259
34944/60000 [================>.............] - ETA: 35s - loss: 0.2402 - categorical_accuracy: 0.9260
35008/60000 [================>.............] - ETA: 35s - loss: 0.2401 - categorical_accuracy: 0.9260
35072/60000 [================>.............] - ETA: 34s - loss: 0.2397 - categorical_accuracy: 0.9262
35136/60000 [================>.............] - ETA: 34s - loss: 0.2397 - categorical_accuracy: 0.9261
35200/60000 [================>.............] - ETA: 34s - loss: 0.2396 - categorical_accuracy: 0.9262
35232/60000 [================>.............] - ETA: 34s - loss: 0.2395 - categorical_accuracy: 0.9262
35296/60000 [================>.............] - ETA: 34s - loss: 0.2392 - categorical_accuracy: 0.9263
35360/60000 [================>.............] - ETA: 34s - loss: 0.2388 - categorical_accuracy: 0.9264
35424/60000 [================>.............] - ETA: 34s - loss: 0.2385 - categorical_accuracy: 0.9265
35488/60000 [================>.............] - ETA: 34s - loss: 0.2384 - categorical_accuracy: 0.9265
35552/60000 [================>.............] - ETA: 34s - loss: 0.2383 - categorical_accuracy: 0.9266
35616/60000 [================>.............] - ETA: 34s - loss: 0.2380 - categorical_accuracy: 0.9266
35680/60000 [================>.............] - ETA: 34s - loss: 0.2377 - categorical_accuracy: 0.9267
35744/60000 [================>.............] - ETA: 34s - loss: 0.2374 - categorical_accuracy: 0.9268
35808/60000 [================>.............] - ETA: 33s - loss: 0.2372 - categorical_accuracy: 0.9269
35872/60000 [================>.............] - ETA: 33s - loss: 0.2370 - categorical_accuracy: 0.9270
35936/60000 [================>.............] - ETA: 33s - loss: 0.2368 - categorical_accuracy: 0.9270
36000/60000 [=================>............] - ETA: 33s - loss: 0.2365 - categorical_accuracy: 0.9271
36064/60000 [=================>............] - ETA: 33s - loss: 0.2365 - categorical_accuracy: 0.9272
36128/60000 [=================>............] - ETA: 33s - loss: 0.2361 - categorical_accuracy: 0.9273
36192/60000 [=================>............] - ETA: 33s - loss: 0.2358 - categorical_accuracy: 0.9274
36256/60000 [=================>............] - ETA: 33s - loss: 0.2356 - categorical_accuracy: 0.9275
36320/60000 [=================>............] - ETA: 33s - loss: 0.2352 - categorical_accuracy: 0.9276
36384/60000 [=================>............] - ETA: 33s - loss: 0.2350 - categorical_accuracy: 0.9277
36448/60000 [=================>............] - ETA: 33s - loss: 0.2347 - categorical_accuracy: 0.9278
36512/60000 [=================>............] - ETA: 32s - loss: 0.2345 - categorical_accuracy: 0.9278
36576/60000 [=================>............] - ETA: 32s - loss: 0.2342 - categorical_accuracy: 0.9279
36640/60000 [=================>............] - ETA: 32s - loss: 0.2339 - categorical_accuracy: 0.9280
36704/60000 [=================>............] - ETA: 32s - loss: 0.2337 - categorical_accuracy: 0.9280
36768/60000 [=================>............] - ETA: 32s - loss: 0.2337 - categorical_accuracy: 0.9281
36832/60000 [=================>............] - ETA: 32s - loss: 0.2335 - categorical_accuracy: 0.9282
36896/60000 [=================>............] - ETA: 32s - loss: 0.2333 - categorical_accuracy: 0.9282
36960/60000 [=================>............] - ETA: 32s - loss: 0.2331 - categorical_accuracy: 0.9282
37024/60000 [=================>............] - ETA: 32s - loss: 0.2331 - categorical_accuracy: 0.9283
37088/60000 [=================>............] - ETA: 32s - loss: 0.2328 - categorical_accuracy: 0.9284
37152/60000 [=================>............] - ETA: 32s - loss: 0.2325 - categorical_accuracy: 0.9285
37216/60000 [=================>............] - ETA: 31s - loss: 0.2326 - categorical_accuracy: 0.9285
37280/60000 [=================>............] - ETA: 31s - loss: 0.2323 - categorical_accuracy: 0.9286
37344/60000 [=================>............] - ETA: 31s - loss: 0.2321 - categorical_accuracy: 0.9287
37408/60000 [=================>............] - ETA: 31s - loss: 0.2320 - categorical_accuracy: 0.9287
37440/60000 [=================>............] - ETA: 31s - loss: 0.2318 - categorical_accuracy: 0.9288
37504/60000 [=================>............] - ETA: 31s - loss: 0.2316 - categorical_accuracy: 0.9289
37568/60000 [=================>............] - ETA: 31s - loss: 0.2316 - categorical_accuracy: 0.9289
37632/60000 [=================>............] - ETA: 31s - loss: 0.2313 - categorical_accuracy: 0.9290
37696/60000 [=================>............] - ETA: 31s - loss: 0.2310 - categorical_accuracy: 0.9291
37760/60000 [=================>............] - ETA: 31s - loss: 0.2308 - categorical_accuracy: 0.9291
37824/60000 [=================>............] - ETA: 31s - loss: 0.2306 - categorical_accuracy: 0.9291
37888/60000 [=================>............] - ETA: 31s - loss: 0.2304 - categorical_accuracy: 0.9292
37952/60000 [=================>............] - ETA: 30s - loss: 0.2303 - categorical_accuracy: 0.9293
38016/60000 [==================>...........] - ETA: 30s - loss: 0.2301 - categorical_accuracy: 0.9294
38080/60000 [==================>...........] - ETA: 30s - loss: 0.2299 - categorical_accuracy: 0.9294
38144/60000 [==================>...........] - ETA: 30s - loss: 0.2296 - categorical_accuracy: 0.9295
38208/60000 [==================>...........] - ETA: 30s - loss: 0.2293 - categorical_accuracy: 0.9296
38272/60000 [==================>...........] - ETA: 30s - loss: 0.2294 - categorical_accuracy: 0.9296
38336/60000 [==================>...........] - ETA: 30s - loss: 0.2291 - categorical_accuracy: 0.9296
38400/60000 [==================>...........] - ETA: 30s - loss: 0.2292 - categorical_accuracy: 0.9297
38464/60000 [==================>...........] - ETA: 30s - loss: 0.2290 - categorical_accuracy: 0.9298
38528/60000 [==================>...........] - ETA: 30s - loss: 0.2288 - categorical_accuracy: 0.9298
38592/60000 [==================>...........] - ETA: 30s - loss: 0.2285 - categorical_accuracy: 0.9299
38656/60000 [==================>...........] - ETA: 29s - loss: 0.2283 - categorical_accuracy: 0.9300
38720/60000 [==================>...........] - ETA: 29s - loss: 0.2280 - categorical_accuracy: 0.9300
38784/60000 [==================>...........] - ETA: 29s - loss: 0.2280 - categorical_accuracy: 0.9300
38848/60000 [==================>...........] - ETA: 29s - loss: 0.2277 - categorical_accuracy: 0.9301
38912/60000 [==================>...........] - ETA: 29s - loss: 0.2273 - categorical_accuracy: 0.9303
38976/60000 [==================>...........] - ETA: 29s - loss: 0.2270 - categorical_accuracy: 0.9303
39040/60000 [==================>...........] - ETA: 29s - loss: 0.2269 - categorical_accuracy: 0.9304
39104/60000 [==================>...........] - ETA: 29s - loss: 0.2268 - categorical_accuracy: 0.9305
39168/60000 [==================>...........] - ETA: 29s - loss: 0.2267 - categorical_accuracy: 0.9305
39232/60000 [==================>...........] - ETA: 29s - loss: 0.2266 - categorical_accuracy: 0.9305
39296/60000 [==================>...........] - ETA: 29s - loss: 0.2264 - categorical_accuracy: 0.9306
39360/60000 [==================>...........] - ETA: 28s - loss: 0.2261 - categorical_accuracy: 0.9307
39424/60000 [==================>...........] - ETA: 28s - loss: 0.2261 - categorical_accuracy: 0.9307
39488/60000 [==================>...........] - ETA: 28s - loss: 0.2261 - categorical_accuracy: 0.9307
39552/60000 [==================>...........] - ETA: 28s - loss: 0.2261 - categorical_accuracy: 0.9307
39616/60000 [==================>...........] - ETA: 28s - loss: 0.2258 - categorical_accuracy: 0.9308
39648/60000 [==================>...........] - ETA: 28s - loss: 0.2257 - categorical_accuracy: 0.9308
39680/60000 [==================>...........] - ETA: 28s - loss: 0.2258 - categorical_accuracy: 0.9308
39744/60000 [==================>...........] - ETA: 28s - loss: 0.2255 - categorical_accuracy: 0.9309
39808/60000 [==================>...........] - ETA: 28s - loss: 0.2256 - categorical_accuracy: 0.9309
39872/60000 [==================>...........] - ETA: 28s - loss: 0.2256 - categorical_accuracy: 0.9310
39936/60000 [==================>...........] - ETA: 28s - loss: 0.2255 - categorical_accuracy: 0.9310
40000/60000 [===================>..........] - ETA: 28s - loss: 0.2253 - categorical_accuracy: 0.9311
40064/60000 [===================>..........] - ETA: 27s - loss: 0.2251 - categorical_accuracy: 0.9311
40128/60000 [===================>..........] - ETA: 27s - loss: 0.2249 - categorical_accuracy: 0.9312
40192/60000 [===================>..........] - ETA: 27s - loss: 0.2246 - categorical_accuracy: 0.9313
40256/60000 [===================>..........] - ETA: 27s - loss: 0.2245 - categorical_accuracy: 0.9313
40320/60000 [===================>..........] - ETA: 27s - loss: 0.2242 - categorical_accuracy: 0.9314
40384/60000 [===================>..........] - ETA: 27s - loss: 0.2239 - categorical_accuracy: 0.9315
40448/60000 [===================>..........] - ETA: 27s - loss: 0.2238 - categorical_accuracy: 0.9316
40512/60000 [===================>..........] - ETA: 27s - loss: 0.2237 - categorical_accuracy: 0.9316
40576/60000 [===================>..........] - ETA: 27s - loss: 0.2235 - categorical_accuracy: 0.9317
40640/60000 [===================>..........] - ETA: 27s - loss: 0.2233 - categorical_accuracy: 0.9317
40704/60000 [===================>..........] - ETA: 27s - loss: 0.2231 - categorical_accuracy: 0.9318
40768/60000 [===================>..........] - ETA: 26s - loss: 0.2229 - categorical_accuracy: 0.9318
40832/60000 [===================>..........] - ETA: 26s - loss: 0.2230 - categorical_accuracy: 0.9318
40864/60000 [===================>..........] - ETA: 26s - loss: 0.2228 - categorical_accuracy: 0.9319
40928/60000 [===================>..........] - ETA: 26s - loss: 0.2227 - categorical_accuracy: 0.9319
40992/60000 [===================>..........] - ETA: 26s - loss: 0.2227 - categorical_accuracy: 0.9319
41056/60000 [===================>..........] - ETA: 26s - loss: 0.2225 - categorical_accuracy: 0.9319
41120/60000 [===================>..........] - ETA: 26s - loss: 0.2224 - categorical_accuracy: 0.9320
41184/60000 [===================>..........] - ETA: 26s - loss: 0.2224 - categorical_accuracy: 0.9320
41248/60000 [===================>..........] - ETA: 26s - loss: 0.2221 - categorical_accuracy: 0.9321
41312/60000 [===================>..........] - ETA: 26s - loss: 0.2221 - categorical_accuracy: 0.9321
41376/60000 [===================>..........] - ETA: 26s - loss: 0.2222 - categorical_accuracy: 0.9321
41440/60000 [===================>..........] - ETA: 26s - loss: 0.2222 - categorical_accuracy: 0.9321
41504/60000 [===================>..........] - ETA: 25s - loss: 0.2219 - categorical_accuracy: 0.9322
41568/60000 [===================>..........] - ETA: 25s - loss: 0.2217 - categorical_accuracy: 0.9322
41632/60000 [===================>..........] - ETA: 25s - loss: 0.2216 - categorical_accuracy: 0.9323
41696/60000 [===================>..........] - ETA: 25s - loss: 0.2214 - categorical_accuracy: 0.9323
41760/60000 [===================>..........] - ETA: 25s - loss: 0.2214 - categorical_accuracy: 0.9323
41792/60000 [===================>..........] - ETA: 25s - loss: 0.2214 - categorical_accuracy: 0.9323
41824/60000 [===================>..........] - ETA: 25s - loss: 0.2213 - categorical_accuracy: 0.9323
41888/60000 [===================>..........] - ETA: 25s - loss: 0.2210 - categorical_accuracy: 0.9324
41952/60000 [===================>..........] - ETA: 25s - loss: 0.2209 - categorical_accuracy: 0.9325
42016/60000 [====================>.........] - ETA: 25s - loss: 0.2207 - categorical_accuracy: 0.9325
42080/60000 [====================>.........] - ETA: 25s - loss: 0.2205 - categorical_accuracy: 0.9326
42144/60000 [====================>.........] - ETA: 25s - loss: 0.2203 - categorical_accuracy: 0.9327
42208/60000 [====================>.........] - ETA: 24s - loss: 0.2201 - categorical_accuracy: 0.9327
42272/60000 [====================>.........] - ETA: 24s - loss: 0.2199 - categorical_accuracy: 0.9328
42336/60000 [====================>.........] - ETA: 24s - loss: 0.2197 - categorical_accuracy: 0.9329
42400/60000 [====================>.........] - ETA: 24s - loss: 0.2194 - categorical_accuracy: 0.9330
42464/60000 [====================>.........] - ETA: 24s - loss: 0.2192 - categorical_accuracy: 0.9330
42528/60000 [====================>.........] - ETA: 24s - loss: 0.2190 - categorical_accuracy: 0.9331
42592/60000 [====================>.........] - ETA: 24s - loss: 0.2187 - categorical_accuracy: 0.9332
42656/60000 [====================>.........] - ETA: 24s - loss: 0.2185 - categorical_accuracy: 0.9332
42720/60000 [====================>.........] - ETA: 24s - loss: 0.2183 - categorical_accuracy: 0.9333
42784/60000 [====================>.........] - ETA: 24s - loss: 0.2182 - categorical_accuracy: 0.9333
42848/60000 [====================>.........] - ETA: 24s - loss: 0.2180 - categorical_accuracy: 0.9334
42912/60000 [====================>.........] - ETA: 23s - loss: 0.2177 - categorical_accuracy: 0.9335
42976/60000 [====================>.........] - ETA: 23s - loss: 0.2176 - categorical_accuracy: 0.9335
43040/60000 [====================>.........] - ETA: 23s - loss: 0.2177 - categorical_accuracy: 0.9335
43104/60000 [====================>.........] - ETA: 23s - loss: 0.2176 - categorical_accuracy: 0.9336
43168/60000 [====================>.........] - ETA: 23s - loss: 0.2175 - categorical_accuracy: 0.9337
43232/60000 [====================>.........] - ETA: 23s - loss: 0.2174 - categorical_accuracy: 0.9337
43296/60000 [====================>.........] - ETA: 23s - loss: 0.2172 - categorical_accuracy: 0.9338
43360/60000 [====================>.........] - ETA: 23s - loss: 0.2171 - categorical_accuracy: 0.9338
43424/60000 [====================>.........] - ETA: 23s - loss: 0.2169 - categorical_accuracy: 0.9338
43488/60000 [====================>.........] - ETA: 23s - loss: 0.2167 - categorical_accuracy: 0.9338
43552/60000 [====================>.........] - ETA: 23s - loss: 0.2166 - categorical_accuracy: 0.9339
43616/60000 [====================>.........] - ETA: 22s - loss: 0.2164 - categorical_accuracy: 0.9340
43680/60000 [====================>.........] - ETA: 22s - loss: 0.2162 - categorical_accuracy: 0.9340
43744/60000 [====================>.........] - ETA: 22s - loss: 0.2160 - categorical_accuracy: 0.9341
43808/60000 [====================>.........] - ETA: 22s - loss: 0.2157 - categorical_accuracy: 0.9342
43872/60000 [====================>.........] - ETA: 22s - loss: 0.2156 - categorical_accuracy: 0.9342
43936/60000 [====================>.........] - ETA: 22s - loss: 0.2154 - categorical_accuracy: 0.9343
44000/60000 [=====================>........] - ETA: 22s - loss: 0.2153 - categorical_accuracy: 0.9344
44064/60000 [=====================>........] - ETA: 22s - loss: 0.2151 - categorical_accuracy: 0.9344
44096/60000 [=====================>........] - ETA: 22s - loss: 0.2151 - categorical_accuracy: 0.9344
44160/60000 [=====================>........] - ETA: 22s - loss: 0.2150 - categorical_accuracy: 0.9345
44192/60000 [=====================>........] - ETA: 22s - loss: 0.2148 - categorical_accuracy: 0.9345
44256/60000 [=====================>........] - ETA: 22s - loss: 0.2149 - categorical_accuracy: 0.9345
44320/60000 [=====================>........] - ETA: 21s - loss: 0.2146 - categorical_accuracy: 0.9346
44384/60000 [=====================>........] - ETA: 21s - loss: 0.2146 - categorical_accuracy: 0.9346
44448/60000 [=====================>........] - ETA: 21s - loss: 0.2145 - categorical_accuracy: 0.9347
44480/60000 [=====================>........] - ETA: 21s - loss: 0.2147 - categorical_accuracy: 0.9346
44544/60000 [=====================>........] - ETA: 21s - loss: 0.2144 - categorical_accuracy: 0.9347
44608/60000 [=====================>........] - ETA: 21s - loss: 0.2143 - categorical_accuracy: 0.9348
44672/60000 [=====================>........] - ETA: 21s - loss: 0.2141 - categorical_accuracy: 0.9349
44736/60000 [=====================>........] - ETA: 21s - loss: 0.2139 - categorical_accuracy: 0.9349
44800/60000 [=====================>........] - ETA: 21s - loss: 0.2138 - categorical_accuracy: 0.9349
44864/60000 [=====================>........] - ETA: 21s - loss: 0.2137 - categorical_accuracy: 0.9350
44928/60000 [=====================>........] - ETA: 21s - loss: 0.2135 - categorical_accuracy: 0.9350
44992/60000 [=====================>........] - ETA: 21s - loss: 0.2134 - categorical_accuracy: 0.9351
45056/60000 [=====================>........] - ETA: 20s - loss: 0.2132 - categorical_accuracy: 0.9351
45120/60000 [=====================>........] - ETA: 20s - loss: 0.2129 - categorical_accuracy: 0.9352
45184/60000 [=====================>........] - ETA: 20s - loss: 0.2130 - categorical_accuracy: 0.9352
45248/60000 [=====================>........] - ETA: 20s - loss: 0.2131 - categorical_accuracy: 0.9352
45312/60000 [=====================>........] - ETA: 20s - loss: 0.2130 - categorical_accuracy: 0.9352
45376/60000 [=====================>........] - ETA: 20s - loss: 0.2129 - categorical_accuracy: 0.9352
45440/60000 [=====================>........] - ETA: 20s - loss: 0.2129 - categorical_accuracy: 0.9353
45504/60000 [=====================>........] - ETA: 20s - loss: 0.2129 - categorical_accuracy: 0.9353
45568/60000 [=====================>........] - ETA: 20s - loss: 0.2128 - categorical_accuracy: 0.9353
45632/60000 [=====================>........] - ETA: 20s - loss: 0.2128 - categorical_accuracy: 0.9353
45696/60000 [=====================>........] - ETA: 20s - loss: 0.2128 - categorical_accuracy: 0.9354
45760/60000 [=====================>........] - ETA: 19s - loss: 0.2130 - categorical_accuracy: 0.9353
45824/60000 [=====================>........] - ETA: 19s - loss: 0.2130 - categorical_accuracy: 0.9354
45888/60000 [=====================>........] - ETA: 19s - loss: 0.2129 - categorical_accuracy: 0.9354
45952/60000 [=====================>........] - ETA: 19s - loss: 0.2128 - categorical_accuracy: 0.9355
46016/60000 [======================>.......] - ETA: 19s - loss: 0.2126 - categorical_accuracy: 0.9355
46080/60000 [======================>.......] - ETA: 19s - loss: 0.2125 - categorical_accuracy: 0.9356
46144/60000 [======================>.......] - ETA: 19s - loss: 0.2123 - categorical_accuracy: 0.9356
46176/60000 [======================>.......] - ETA: 19s - loss: 0.2124 - categorical_accuracy: 0.9356
46240/60000 [======================>.......] - ETA: 19s - loss: 0.2122 - categorical_accuracy: 0.9356
46304/60000 [======================>.......] - ETA: 19s - loss: 0.2121 - categorical_accuracy: 0.9356
46368/60000 [======================>.......] - ETA: 19s - loss: 0.2120 - categorical_accuracy: 0.9357
46432/60000 [======================>.......] - ETA: 19s - loss: 0.2118 - categorical_accuracy: 0.9357
46496/60000 [======================>.......] - ETA: 18s - loss: 0.2117 - categorical_accuracy: 0.9358
46560/60000 [======================>.......] - ETA: 18s - loss: 0.2116 - categorical_accuracy: 0.9358
46624/60000 [======================>.......] - ETA: 18s - loss: 0.2114 - categorical_accuracy: 0.9359
46688/60000 [======================>.......] - ETA: 18s - loss: 0.2112 - categorical_accuracy: 0.9359
46752/60000 [======================>.......] - ETA: 18s - loss: 0.2111 - categorical_accuracy: 0.9360
46816/60000 [======================>.......] - ETA: 18s - loss: 0.2109 - categorical_accuracy: 0.9360
46880/60000 [======================>.......] - ETA: 18s - loss: 0.2107 - categorical_accuracy: 0.9361
46944/60000 [======================>.......] - ETA: 18s - loss: 0.2107 - categorical_accuracy: 0.9362
47008/60000 [======================>.......] - ETA: 18s - loss: 0.2105 - categorical_accuracy: 0.9362
47072/60000 [======================>.......] - ETA: 18s - loss: 0.2103 - categorical_accuracy: 0.9363
47136/60000 [======================>.......] - ETA: 18s - loss: 0.2101 - categorical_accuracy: 0.9363
47200/60000 [======================>.......] - ETA: 17s - loss: 0.2098 - categorical_accuracy: 0.9364
47264/60000 [======================>.......] - ETA: 17s - loss: 0.2097 - categorical_accuracy: 0.9365
47328/60000 [======================>.......] - ETA: 17s - loss: 0.2096 - categorical_accuracy: 0.9365
47392/60000 [======================>.......] - ETA: 17s - loss: 0.2094 - categorical_accuracy: 0.9366
47424/60000 [======================>.......] - ETA: 17s - loss: 0.2093 - categorical_accuracy: 0.9366
47488/60000 [======================>.......] - ETA: 17s - loss: 0.2090 - categorical_accuracy: 0.9367
47552/60000 [======================>.......] - ETA: 17s - loss: 0.2090 - categorical_accuracy: 0.9367
47616/60000 [======================>.......] - ETA: 17s - loss: 0.2091 - categorical_accuracy: 0.9367
47680/60000 [======================>.......] - ETA: 17s - loss: 0.2089 - categorical_accuracy: 0.9367
47744/60000 [======================>.......] - ETA: 17s - loss: 0.2087 - categorical_accuracy: 0.9368
47808/60000 [======================>.......] - ETA: 17s - loss: 0.2086 - categorical_accuracy: 0.9369
47872/60000 [======================>.......] - ETA: 17s - loss: 0.2084 - categorical_accuracy: 0.9369
47936/60000 [======================>.......] - ETA: 16s - loss: 0.2082 - categorical_accuracy: 0.9370
48000/60000 [=======================>......] - ETA: 16s - loss: 0.2081 - categorical_accuracy: 0.9370
48064/60000 [=======================>......] - ETA: 16s - loss: 0.2079 - categorical_accuracy: 0.9370
48128/60000 [=======================>......] - ETA: 16s - loss: 0.2077 - categorical_accuracy: 0.9371
48192/60000 [=======================>......] - ETA: 16s - loss: 0.2075 - categorical_accuracy: 0.9372
48256/60000 [=======================>......] - ETA: 16s - loss: 0.2073 - categorical_accuracy: 0.9373
48320/60000 [=======================>......] - ETA: 16s - loss: 0.2070 - categorical_accuracy: 0.9373
48352/60000 [=======================>......] - ETA: 16s - loss: 0.2070 - categorical_accuracy: 0.9373
48416/60000 [=======================>......] - ETA: 16s - loss: 0.2069 - categorical_accuracy: 0.9374
48480/60000 [=======================>......] - ETA: 16s - loss: 0.2069 - categorical_accuracy: 0.9374
48544/60000 [=======================>......] - ETA: 16s - loss: 0.2068 - categorical_accuracy: 0.9374
48608/60000 [=======================>......] - ETA: 15s - loss: 0.2066 - categorical_accuracy: 0.9374
48672/60000 [=======================>......] - ETA: 15s - loss: 0.2066 - categorical_accuracy: 0.9374
48736/60000 [=======================>......] - ETA: 15s - loss: 0.2063 - categorical_accuracy: 0.9375
48800/60000 [=======================>......] - ETA: 15s - loss: 0.2061 - categorical_accuracy: 0.9376
48864/60000 [=======================>......] - ETA: 15s - loss: 0.2059 - categorical_accuracy: 0.9376
48896/60000 [=======================>......] - ETA: 15s - loss: 0.2058 - categorical_accuracy: 0.9376
48960/60000 [=======================>......] - ETA: 15s - loss: 0.2056 - categorical_accuracy: 0.9377
49024/60000 [=======================>......] - ETA: 15s - loss: 0.2054 - categorical_accuracy: 0.9377
49088/60000 [=======================>......] - ETA: 15s - loss: 0.2052 - categorical_accuracy: 0.9378
49152/60000 [=======================>......] - ETA: 15s - loss: 0.2052 - categorical_accuracy: 0.9378
49216/60000 [=======================>......] - ETA: 15s - loss: 0.2051 - categorical_accuracy: 0.9378
49248/60000 [=======================>......] - ETA: 15s - loss: 0.2050 - categorical_accuracy: 0.9378
49280/60000 [=======================>......] - ETA: 15s - loss: 0.2052 - categorical_accuracy: 0.9378
49312/60000 [=======================>......] - ETA: 15s - loss: 0.2051 - categorical_accuracy: 0.9379
49344/60000 [=======================>......] - ETA: 14s - loss: 0.2052 - categorical_accuracy: 0.9379
49408/60000 [=======================>......] - ETA: 14s - loss: 0.2050 - categorical_accuracy: 0.9379
49472/60000 [=======================>......] - ETA: 14s - loss: 0.2049 - categorical_accuracy: 0.9379
49536/60000 [=======================>......] - ETA: 14s - loss: 0.2047 - categorical_accuracy: 0.9380
49600/60000 [=======================>......] - ETA: 14s - loss: 0.2046 - categorical_accuracy: 0.9380
49664/60000 [=======================>......] - ETA: 14s - loss: 0.2045 - categorical_accuracy: 0.9381
49728/60000 [=======================>......] - ETA: 14s - loss: 0.2043 - categorical_accuracy: 0.9381
49792/60000 [=======================>......] - ETA: 14s - loss: 0.2041 - categorical_accuracy: 0.9382
49856/60000 [=======================>......] - ETA: 14s - loss: 0.2039 - categorical_accuracy: 0.9383
49920/60000 [=======================>......] - ETA: 14s - loss: 0.2037 - categorical_accuracy: 0.9383
49984/60000 [=======================>......] - ETA: 14s - loss: 0.2035 - categorical_accuracy: 0.9384
50048/60000 [========================>.....] - ETA: 13s - loss: 0.2035 - categorical_accuracy: 0.9384
50112/60000 [========================>.....] - ETA: 13s - loss: 0.2033 - categorical_accuracy: 0.9385
50176/60000 [========================>.....] - ETA: 13s - loss: 0.2031 - categorical_accuracy: 0.9385
50240/60000 [========================>.....] - ETA: 13s - loss: 0.2029 - categorical_accuracy: 0.9386
50304/60000 [========================>.....] - ETA: 13s - loss: 0.2029 - categorical_accuracy: 0.9386
50368/60000 [========================>.....] - ETA: 13s - loss: 0.2026 - categorical_accuracy: 0.9387
50432/60000 [========================>.....] - ETA: 13s - loss: 0.2024 - categorical_accuracy: 0.9387
50496/60000 [========================>.....] - ETA: 13s - loss: 0.2025 - categorical_accuracy: 0.9387
50560/60000 [========================>.....] - ETA: 13s - loss: 0.2023 - categorical_accuracy: 0.9387
50624/60000 [========================>.....] - ETA: 13s - loss: 0.2021 - categorical_accuracy: 0.9388
50688/60000 [========================>.....] - ETA: 13s - loss: 0.2020 - categorical_accuracy: 0.9388
50752/60000 [========================>.....] - ETA: 12s - loss: 0.2018 - categorical_accuracy: 0.9389
50816/60000 [========================>.....] - ETA: 12s - loss: 0.2018 - categorical_accuracy: 0.9389
50880/60000 [========================>.....] - ETA: 12s - loss: 0.2018 - categorical_accuracy: 0.9389
50944/60000 [========================>.....] - ETA: 12s - loss: 0.2018 - categorical_accuracy: 0.9389
51008/60000 [========================>.....] - ETA: 12s - loss: 0.2016 - categorical_accuracy: 0.9389
51072/60000 [========================>.....] - ETA: 12s - loss: 0.2015 - categorical_accuracy: 0.9390
51136/60000 [========================>.....] - ETA: 12s - loss: 0.2013 - categorical_accuracy: 0.9390
51200/60000 [========================>.....] - ETA: 12s - loss: 0.2011 - categorical_accuracy: 0.9391
51264/60000 [========================>.....] - ETA: 12s - loss: 0.2010 - categorical_accuracy: 0.9391
51328/60000 [========================>.....] - ETA: 12s - loss: 0.2009 - categorical_accuracy: 0.9392
51392/60000 [========================>.....] - ETA: 12s - loss: 0.2007 - categorical_accuracy: 0.9392
51456/60000 [========================>.....] - ETA: 11s - loss: 0.2005 - categorical_accuracy: 0.9392
51520/60000 [========================>.....] - ETA: 11s - loss: 0.2004 - categorical_accuracy: 0.9393
51584/60000 [========================>.....] - ETA: 11s - loss: 0.2002 - categorical_accuracy: 0.9394
51648/60000 [========================>.....] - ETA: 11s - loss: 0.2001 - categorical_accuracy: 0.9394
51712/60000 [========================>.....] - ETA: 11s - loss: 0.1999 - categorical_accuracy: 0.9395
51744/60000 [========================>.....] - ETA: 11s - loss: 0.1998 - categorical_accuracy: 0.9395
51808/60000 [========================>.....] - ETA: 11s - loss: 0.1996 - categorical_accuracy: 0.9395
51872/60000 [========================>.....] - ETA: 11s - loss: 0.1995 - categorical_accuracy: 0.9396
51936/60000 [========================>.....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9396
52000/60000 [=========================>....] - ETA: 11s - loss: 0.1993 - categorical_accuracy: 0.9396
52064/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9397
52128/60000 [=========================>....] - ETA: 11s - loss: 0.1991 - categorical_accuracy: 0.9397
52192/60000 [=========================>....] - ETA: 10s - loss: 0.1991 - categorical_accuracy: 0.9397
52256/60000 [=========================>....] - ETA: 10s - loss: 0.1989 - categorical_accuracy: 0.9398
52320/60000 [=========================>....] - ETA: 10s - loss: 0.1988 - categorical_accuracy: 0.9398
52384/60000 [=========================>....] - ETA: 10s - loss: 0.1987 - categorical_accuracy: 0.9399
52448/60000 [=========================>....] - ETA: 10s - loss: 0.1986 - categorical_accuracy: 0.9399
52512/60000 [=========================>....] - ETA: 10s - loss: 0.1986 - categorical_accuracy: 0.9399
52576/60000 [=========================>....] - ETA: 10s - loss: 0.1985 - categorical_accuracy: 0.9399
52640/60000 [=========================>....] - ETA: 10s - loss: 0.1983 - categorical_accuracy: 0.9400
52704/60000 [=========================>....] - ETA: 10s - loss: 0.1982 - categorical_accuracy: 0.9400
52768/60000 [=========================>....] - ETA: 10s - loss: 0.1980 - categorical_accuracy: 0.9400
52832/60000 [=========================>....] - ETA: 10s - loss: 0.1979 - categorical_accuracy: 0.9401
52896/60000 [=========================>....] - ETA: 9s - loss: 0.1978 - categorical_accuracy: 0.9401 
52960/60000 [=========================>....] - ETA: 9s - loss: 0.1975 - categorical_accuracy: 0.9402
53024/60000 [=========================>....] - ETA: 9s - loss: 0.1974 - categorical_accuracy: 0.9402
53088/60000 [=========================>....] - ETA: 9s - loss: 0.1972 - categorical_accuracy: 0.9402
53120/60000 [=========================>....] - ETA: 9s - loss: 0.1971 - categorical_accuracy: 0.9403
53184/60000 [=========================>....] - ETA: 9s - loss: 0.1969 - categorical_accuracy: 0.9403
53248/60000 [=========================>....] - ETA: 9s - loss: 0.1968 - categorical_accuracy: 0.9404
53312/60000 [=========================>....] - ETA: 9s - loss: 0.1966 - categorical_accuracy: 0.9404
53376/60000 [=========================>....] - ETA: 9s - loss: 0.1964 - categorical_accuracy: 0.9405
53440/60000 [=========================>....] - ETA: 9s - loss: 0.1963 - categorical_accuracy: 0.9405
53504/60000 [=========================>....] - ETA: 9s - loss: 0.1961 - categorical_accuracy: 0.9405
53568/60000 [=========================>....] - ETA: 9s - loss: 0.1960 - categorical_accuracy: 0.9406
53632/60000 [=========================>....] - ETA: 8s - loss: 0.1959 - categorical_accuracy: 0.9406
53696/60000 [=========================>....] - ETA: 8s - loss: 0.1957 - categorical_accuracy: 0.9407
53760/60000 [=========================>....] - ETA: 8s - loss: 0.1954 - categorical_accuracy: 0.9407
53824/60000 [=========================>....] - ETA: 8s - loss: 0.1953 - categorical_accuracy: 0.9408
53888/60000 [=========================>....] - ETA: 8s - loss: 0.1952 - categorical_accuracy: 0.9408
53952/60000 [=========================>....] - ETA: 8s - loss: 0.1951 - categorical_accuracy: 0.9409
54016/60000 [==========================>...] - ETA: 8s - loss: 0.1949 - categorical_accuracy: 0.9409
54080/60000 [==========================>...] - ETA: 8s - loss: 0.1948 - categorical_accuracy: 0.9410
54144/60000 [==========================>...] - ETA: 8s - loss: 0.1947 - categorical_accuracy: 0.9410
54208/60000 [==========================>...] - ETA: 8s - loss: 0.1946 - categorical_accuracy: 0.9410
54272/60000 [==========================>...] - ETA: 8s - loss: 0.1943 - categorical_accuracy: 0.9411
54336/60000 [==========================>...] - ETA: 7s - loss: 0.1943 - categorical_accuracy: 0.9411
54400/60000 [==========================>...] - ETA: 7s - loss: 0.1942 - categorical_accuracy: 0.9411
54464/60000 [==========================>...] - ETA: 7s - loss: 0.1941 - categorical_accuracy: 0.9411
54528/60000 [==========================>...] - ETA: 7s - loss: 0.1940 - categorical_accuracy: 0.9412
54592/60000 [==========================>...] - ETA: 7s - loss: 0.1939 - categorical_accuracy: 0.9412
54656/60000 [==========================>...] - ETA: 7s - loss: 0.1938 - categorical_accuracy: 0.9412
54720/60000 [==========================>...] - ETA: 7s - loss: 0.1938 - categorical_accuracy: 0.9412
54784/60000 [==========================>...] - ETA: 7s - loss: 0.1936 - categorical_accuracy: 0.9413
54816/60000 [==========================>...] - ETA: 7s - loss: 0.1937 - categorical_accuracy: 0.9413
54880/60000 [==========================>...] - ETA: 7s - loss: 0.1935 - categorical_accuracy: 0.9414
54944/60000 [==========================>...] - ETA: 7s - loss: 0.1935 - categorical_accuracy: 0.9414
55008/60000 [==========================>...] - ETA: 7s - loss: 0.1933 - categorical_accuracy: 0.9414
55072/60000 [==========================>...] - ETA: 6s - loss: 0.1932 - categorical_accuracy: 0.9415
55136/60000 [==========================>...] - ETA: 6s - loss: 0.1931 - categorical_accuracy: 0.9415
55200/60000 [==========================>...] - ETA: 6s - loss: 0.1930 - categorical_accuracy: 0.9415
55264/60000 [==========================>...] - ETA: 6s - loss: 0.1929 - categorical_accuracy: 0.9415
55328/60000 [==========================>...] - ETA: 6s - loss: 0.1927 - categorical_accuracy: 0.9416
55360/60000 [==========================>...] - ETA: 6s - loss: 0.1926 - categorical_accuracy: 0.9416
55424/60000 [==========================>...] - ETA: 6s - loss: 0.1925 - categorical_accuracy: 0.9417
55488/60000 [==========================>...] - ETA: 6s - loss: 0.1923 - categorical_accuracy: 0.9417
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1921 - categorical_accuracy: 0.9418
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1920 - categorical_accuracy: 0.9418
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1919 - categorical_accuracy: 0.9418
55744/60000 [==========================>...] - ETA: 5s - loss: 0.1919 - categorical_accuracy: 0.9419
55808/60000 [==========================>...] - ETA: 5s - loss: 0.1918 - categorical_accuracy: 0.9419
55872/60000 [==========================>...] - ETA: 5s - loss: 0.1916 - categorical_accuracy: 0.9419
55936/60000 [==========================>...] - ETA: 5s - loss: 0.1915 - categorical_accuracy: 0.9420
56000/60000 [===========================>..] - ETA: 5s - loss: 0.1915 - categorical_accuracy: 0.9420
56064/60000 [===========================>..] - ETA: 5s - loss: 0.1914 - categorical_accuracy: 0.9420
56128/60000 [===========================>..] - ETA: 5s - loss: 0.1913 - categorical_accuracy: 0.9420
56192/60000 [===========================>..] - ETA: 5s - loss: 0.1913 - categorical_accuracy: 0.9420
56256/60000 [===========================>..] - ETA: 5s - loss: 0.1911 - categorical_accuracy: 0.9421
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1909 - categorical_accuracy: 0.9421
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1907 - categorical_accuracy: 0.9422
56448/60000 [===========================>..] - ETA: 4s - loss: 0.1907 - categorical_accuracy: 0.9422
56512/60000 [===========================>..] - ETA: 4s - loss: 0.1906 - categorical_accuracy: 0.9422
56576/60000 [===========================>..] - ETA: 4s - loss: 0.1905 - categorical_accuracy: 0.9422
56640/60000 [===========================>..] - ETA: 4s - loss: 0.1904 - categorical_accuracy: 0.9423
56704/60000 [===========================>..] - ETA: 4s - loss: 0.1904 - categorical_accuracy: 0.9423
56768/60000 [===========================>..] - ETA: 4s - loss: 0.1902 - categorical_accuracy: 0.9424
56832/60000 [===========================>..] - ETA: 4s - loss: 0.1900 - categorical_accuracy: 0.9424
56896/60000 [===========================>..] - ETA: 4s - loss: 0.1899 - categorical_accuracy: 0.9425
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1898 - categorical_accuracy: 0.9425
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1897 - categorical_accuracy: 0.9425
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1895 - categorical_accuracy: 0.9425
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1894 - categorical_accuracy: 0.9426
57184/60000 [===========================>..] - ETA: 3s - loss: 0.1893 - categorical_accuracy: 0.9426
57248/60000 [===========================>..] - ETA: 3s - loss: 0.1893 - categorical_accuracy: 0.9426
57312/60000 [===========================>..] - ETA: 3s - loss: 0.1892 - categorical_accuracy: 0.9426
57376/60000 [===========================>..] - ETA: 3s - loss: 0.1890 - categorical_accuracy: 0.9427
57440/60000 [===========================>..] - ETA: 3s - loss: 0.1888 - categorical_accuracy: 0.9427
57504/60000 [===========================>..] - ETA: 3s - loss: 0.1887 - categorical_accuracy: 0.9428
57568/60000 [===========================>..] - ETA: 3s - loss: 0.1885 - categorical_accuracy: 0.9428
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1885 - categorical_accuracy: 0.9428
57696/60000 [===========================>..] - ETA: 3s - loss: 0.1884 - categorical_accuracy: 0.9428
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1884 - categorical_accuracy: 0.9428
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1883 - categorical_accuracy: 0.9429
57888/60000 [===========================>..] - ETA: 2s - loss: 0.1881 - categorical_accuracy: 0.9429
57952/60000 [===========================>..] - ETA: 2s - loss: 0.1880 - categorical_accuracy: 0.9430
58016/60000 [============================>.] - ETA: 2s - loss: 0.1878 - categorical_accuracy: 0.9430
58080/60000 [============================>.] - ETA: 2s - loss: 0.1876 - categorical_accuracy: 0.9431
58144/60000 [============================>.] - ETA: 2s - loss: 0.1875 - categorical_accuracy: 0.9431
58208/60000 [============================>.] - ETA: 2s - loss: 0.1873 - categorical_accuracy: 0.9432
58272/60000 [============================>.] - ETA: 2s - loss: 0.1871 - categorical_accuracy: 0.9432
58336/60000 [============================>.] - ETA: 2s - loss: 0.1870 - categorical_accuracy: 0.9433
58400/60000 [============================>.] - ETA: 2s - loss: 0.1869 - categorical_accuracy: 0.9433
58464/60000 [============================>.] - ETA: 2s - loss: 0.1868 - categorical_accuracy: 0.9433
58528/60000 [============================>.] - ETA: 2s - loss: 0.1866 - categorical_accuracy: 0.9433
58592/60000 [============================>.] - ETA: 1s - loss: 0.1865 - categorical_accuracy: 0.9434
58656/60000 [============================>.] - ETA: 1s - loss: 0.1864 - categorical_accuracy: 0.9434
58720/60000 [============================>.] - ETA: 1s - loss: 0.1864 - categorical_accuracy: 0.9434
58784/60000 [============================>.] - ETA: 1s - loss: 0.1863 - categorical_accuracy: 0.9435
58816/60000 [============================>.] - ETA: 1s - loss: 0.1862 - categorical_accuracy: 0.9435
58880/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9435
58944/60000 [============================>.] - ETA: 1s - loss: 0.1861 - categorical_accuracy: 0.9435
59008/60000 [============================>.] - ETA: 1s - loss: 0.1860 - categorical_accuracy: 0.9436
59072/60000 [============================>.] - ETA: 1s - loss: 0.1858 - categorical_accuracy: 0.9436
59136/60000 [============================>.] - ETA: 1s - loss: 0.1857 - categorical_accuracy: 0.9437
59200/60000 [============================>.] - ETA: 1s - loss: 0.1856 - categorical_accuracy: 0.9437
59264/60000 [============================>.] - ETA: 1s - loss: 0.1855 - categorical_accuracy: 0.9437
59296/60000 [============================>.] - ETA: 0s - loss: 0.1855 - categorical_accuracy: 0.9437
59328/60000 [============================>.] - ETA: 0s - loss: 0.1855 - categorical_accuracy: 0.9437
59392/60000 [============================>.] - ETA: 0s - loss: 0.1854 - categorical_accuracy: 0.9438
59456/60000 [============================>.] - ETA: 0s - loss: 0.1852 - categorical_accuracy: 0.9438
59520/60000 [============================>.] - ETA: 0s - loss: 0.1850 - categorical_accuracy: 0.9439
59584/60000 [============================>.] - ETA: 0s - loss: 0.1849 - categorical_accuracy: 0.9439
59648/60000 [============================>.] - ETA: 0s - loss: 0.1848 - categorical_accuracy: 0.9439
59712/60000 [============================>.] - ETA: 0s - loss: 0.1846 - categorical_accuracy: 0.9440
59744/60000 [============================>.] - ETA: 0s - loss: 0.1845 - categorical_accuracy: 0.9440
59808/60000 [============================>.] - ETA: 0s - loss: 0.1844 - categorical_accuracy: 0.9440
59872/60000 [============================>.] - ETA: 0s - loss: 0.1843 - categorical_accuracy: 0.9441
59936/60000 [============================>.] - ETA: 0s - loss: 0.1842 - categorical_accuracy: 0.9441
60000/60000 [==============================] - 87s 1ms/step - loss: 0.1840 - categorical_accuracy: 0.9441 - val_loss: 0.0591 - val_categorical_accuracy: 0.9813

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
 2560/10000 [======>.......................] - ETA: 2s
 2752/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3136/10000 [========>.....................] - ETA: 1s
 3328/10000 [========>.....................] - ETA: 1s
 3520/10000 [=========>....................] - ETA: 1s
 3712/10000 [==========>...................] - ETA: 1s
 3936/10000 [==========>...................] - ETA: 1s
 4128/10000 [===========>..................] - ETA: 1s
 4352/10000 [============>.................] - ETA: 1s
 4544/10000 [============>.................] - ETA: 1s
 4768/10000 [=============>................] - ETA: 1s
 4992/10000 [=============>................] - ETA: 1s
 5184/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5600/10000 [===============>..............] - ETA: 1s
 5792/10000 [================>.............] - ETA: 1s
 6016/10000 [=================>............] - ETA: 1s
 6240/10000 [=================>............] - ETA: 1s
 6464/10000 [==================>...........] - ETA: 0s
 6656/10000 [==================>...........] - ETA: 0s
 6848/10000 [===================>..........] - ETA: 0s
 7040/10000 [====================>.........] - ETA: 0s
 7232/10000 [====================>.........] - ETA: 0s
 7424/10000 [=====================>........] - ETA: 0s
 7648/10000 [=====================>........] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8064/10000 [=======================>......] - ETA: 0s
 8256/10000 [=======================>......] - ETA: 0s
 8448/10000 [========================>.....] - ETA: 0s
 8640/10000 [========================>.....] - ETA: 0s
 8832/10000 [=========================>....] - ETA: 0s
 9024/10000 [==========================>...] - ETA: 0s
 9184/10000 [==========================>...] - ETA: 0s
 9376/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9792/10000 [============================>.] - ETA: 0s
 9984/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 275us/step
[[3.1922702e-09 5.9316907e-10 1.5213975e-07 ... 9.9999893e-01
  1.8556316e-09 1.9201568e-07]
 [9.7215889e-07 1.9519126e-05 9.9997377e-01 ... 1.7533445e-09
  6.5936763e-07 1.1053297e-08]
 [8.8218707e-07 9.9951279e-01 1.5423726e-04 ... 8.5544394e-05
  8.1709513e-05 1.4920223e-05]
 ...
 [3.1350624e-09 8.3840160e-07 3.4653434e-07 ... 1.0598834e-06
  3.8736791e-05 7.7619166e-05]
 [2.2387376e-07 8.7844754e-09 1.3261057e-09 ... 7.5125817e-10
  3.1653463e-04 1.0578295e-07]
 [1.9002299e-07 1.4468372e-08 2.0387026e-06 ... 8.2166642e-11
  3.5902596e-07 2.2696720e-10]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.05910472284681164, 'accuracy_test:': 0.9812999963760376}

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
