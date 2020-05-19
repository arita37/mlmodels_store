
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '69b309ad857428cc5a734b8afd99842edf9b2a42', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/69b309ad857428cc5a734b8afd99842edf9b2a42

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/69b309ad857428cc5a734b8afd99842edf9b2a42

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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
Already up to date.
[master 6d530eb] ml_store
 2 files changed, 64 insertions(+), 9738 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   305a25c..6d530eb  master -> master





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
[master 3e445db] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   6d530eb..3e445db  master -> master





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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
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
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         20          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-19 20:13:33.029897: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 20:13:33.042636: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-19 20:13:33.042867: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555a01000910 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 20:13:33.042886: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 253
Trainable params: 253
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2675 - binary_crossentropy: 1.2478500/500 [==============================] - 1s 1ms/sample - loss: 0.2822 - binary_crossentropy: 1.6176 - val_loss: 0.2771 - val_binary_crossentropy: 1.5012

  #### metrics   #################################################### 
{'MSE': 0.2791903231887307}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
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
Total params: 253
Trainable params: 253
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
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
Total params: 473
Trainable params: 473
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2541 - binary_crossentropy: 0.8279500/500 [==============================] - 1s 1ms/sample - loss: 0.2615 - binary_crossentropy: 0.9255 - val_loss: 0.2577 - val_binary_crossentropy: 0.8132

  #### metrics   #################################################### 
{'MSE': 0.25933163420297706}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_max[0][0]               
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
Total params: 473
Trainable params: 473
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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
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
Total params: 602
Trainable params: 602
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2499 - binary_crossentropy: 0.6930 - val_loss: 0.2500 - val_binary_crossentropy: 0.6931

  #### metrics   #################################################### 
{'MSE': 0.2497007934275207}

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
weighted_sequence_layer_6 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
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
Total params: 602
Trainable params: 602
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
Total params: 408
Trainable params: 408
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2958 - binary_crossentropy: 0.8025500/500 [==============================] - 1s 2ms/sample - loss: 0.2801 - binary_crossentropy: 0.7696 - val_loss: 0.2811 - val_binary_crossentropy: 0.7622

  #### metrics   #################################################### 
{'MSE': 0.27901578819755957}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
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
Total params: 123
Trainable params: 123
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.5200 - binary_crossentropy: 8.0210500/500 [==============================] - 2s 3ms/sample - loss: 0.5040 - binary_crossentropy: 7.7742 - val_loss: 0.5140 - val_binary_crossentropy: 7.9284

  #### metrics   #################################################### 
{'MSE': 0.509}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
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
Total params: 123
Trainable params: 123
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-19 20:14:48.658447: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:14:48.660375: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:14:48.666933: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-19 20:14:48.677369: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-19 20:14:48.679245: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 20:14:48.681140: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:14:48.682633: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2491 - val_binary_crossentropy: 0.6914
2020-05-19 20:14:49.867857: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:14:49.869809: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:14:49.875173: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-19 20:14:49.885092: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-19 20:14:49.886673: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 20:14:49.888024: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:14:49.889278: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24875320478987817}

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
2020-05-19 20:15:12.321544: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:12.323205: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:12.327093: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-19 20:15:12.332788: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-19 20:15:12.333893: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 20:15:12.334843: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:12.335700: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2507 - val_binary_crossentropy: 0.6945
2020-05-19 20:15:13.743590: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:13.744712: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:13.747486: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-19 20:15:13.752585: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-19 20:15:13.753514: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 20:15:13.754675: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:13.755748: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25078994440326774}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-19 20:15:46.426482: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:46.431211: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:46.446734: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-19 20:15:46.471768: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-19 20:15:46.476474: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 20:15:46.480034: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:46.483943: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.4688 - binary_crossentropy: 1.1541 - val_loss: 0.2627 - val_binary_crossentropy: 0.7193
2020-05-19 20:15:48.566730: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:48.571001: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:48.581701: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-19 20:15:48.605736: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-19 20:15:48.610089: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-19 20:15:48.614149: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-19 20:15:48.617900: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.22523572343746068}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 6s - loss: 0.3115 - binary_crossentropy: 0.8281500/500 [==============================] - 4s 8ms/sample - loss: 0.2827 - binary_crossentropy: 0.7655 - val_loss: 0.2722 - val_binary_crossentropy: 0.7423

  #### metrics   #################################################### 
{'MSE': 0.2762564599002984}

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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 251
Trainable params: 251
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2773 - binary_crossentropy: 0.7531500/500 [==============================] - 4s 9ms/sample - loss: 0.2711 - binary_crossentropy: 0.7415 - val_loss: 0.2569 - val_binary_crossentropy: 0.7598

  #### metrics   #################################################### 
{'MSE': 0.2591697896574908}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 2)         14          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 2)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         14          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         12          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
Total params: 251
Trainable params: 251
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
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
Total params: 1,889
Trainable params: 1,889
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2619 - binary_crossentropy: 0.8493500/500 [==============================] - 4s 9ms/sample - loss: 0.2552 - binary_crossentropy: 0.7565 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934

  #### metrics   #################################################### 
{'MSE': 0.2516031134773982}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_max[0][0]               
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
Total params: 1,889
Trainable params: 1,889
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
regionsequence_sum (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 4)]          0                                            
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
region_10sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 8s - loss: 0.2671 - binary_crossentropy: 0.7282500/500 [==============================] - 6s 11ms/sample - loss: 0.2670 - binary_crossentropy: 0.7281 - val_loss: 0.2606 - val_binary_crossentropy: 0.7150

  #### metrics   #################################################### 
{'MSE': 0.26291642780936025}

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
regionsequence_sum (InputLayer) [(None, 3)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 4)]          0                                            
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
region_10sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 3, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 1,402
Trainable params: 1,402
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3110 - binary_crossentropy: 0.8327500/500 [==============================] - 5s 11ms/sample - loss: 0.2702 - binary_crossentropy: 0.7395 - val_loss: 0.2662 - val_binary_crossentropy: 0.7278

  #### metrics   #################################################### 
{'MSE': 0.26380998187530225}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         5           sequence_max[0][0]               
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
Total params: 1,402
Trainable params: 1,402
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
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
Total params: 2,950
Trainable params: 2,870
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3280 - binary_crossentropy: 0.9078500/500 [==============================] - 6s 12ms/sample - loss: 0.3290 - binary_crossentropy: 0.9014 - val_loss: 0.3213 - val_binary_crossentropy: 0.8830

  #### metrics   #################################################### 
{'MSE': 0.32180159212133785}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         8           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         4           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         8           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         4           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 7, 4)         24          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           hash_10[0][0]                    
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
Total params: 2,950
Trainable params: 2,870
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
   3e445db..7b21c3d  master     -> origin/master
Updating 3e445db..7b21c3d
Fast-forward
 .../20200519/list_log_pullrequest_20200519.md      |   2 +-
 error_list/20200519/list_log_testall_20200519.md   | 727 ---------------------
 ...-12_69b309ad857428cc5a734b8afd99842edf9b2a42.py | 627 ++++++++++++++++++
 3 files changed, 628 insertions(+), 728 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-19-20-12_69b309ad857428cc5a734b8afd99842edf9b2a42.py
[master a1cc4f9] ml_store
 1 file changed, 4955 insertions(+)
To github.com:arita37/mlmodels_store.git
   7b21c3d..a1cc4f9  master -> master





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
[master ba8eb9f] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   a1cc4f9..ba8eb9f  master -> master





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
[master 3386708] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   ba8eb9f..3386708  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
Already up to date.
[master 30d2aea] ml_store
 1 file changed, 37 insertions(+)
To github.com:arita37/mlmodels_store.git
   3386708..30d2aea  master -> master





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

2020-05-19 20:24:21.624011: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 20:24:21.628527: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-19 20:24:21.628679: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e6fadca260 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 20:24:21.628694: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3877
256/354 [====================>.........] - ETA: 3s - loss: 1.2969
354/354 [==============================] - 15s 42ms/step - loss: 1.4635 - val_loss: 1.7364

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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
Already up to date.
[master 49734c7] ml_store
 1 file changed, 151 insertions(+)
To github.com:arita37/mlmodels_store.git
   30d2aea..49734c7  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
Already up to date.
[master 4ac20bc] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   49734c7..4ac20bc  master -> master





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
[master e64a6a8] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   4ac20bc..e64a6a8  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2228224/17464789 [==>...........................] - ETA: 0s
 7421952/17464789 [===========>..................] - ETA: 0s
11542528/17464789 [==================>...........] - ETA: 0s
15163392/17464789 [=========================>....] - ETA: 0s
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
2020-05-19 20:25:26.569740: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 20:25:26.573956: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-19 20:25:26.574096: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55eae1175830 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-19 20:25:26.574109: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7893 - accuracy: 0.4920
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8046 - accuracy: 0.4910 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8660 - accuracy: 0.4870
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8046 - accuracy: 0.4910
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8261 - accuracy: 0.4896
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6922 - accuracy: 0.4983
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.7148 - accuracy: 0.4969
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7529 - accuracy: 0.4944
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7314 - accuracy: 0.4958
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7188 - accuracy: 0.4966
11000/25000 [============>.................] - ETA: 4s - loss: 7.7349 - accuracy: 0.4955
12000/25000 [=============>................] - ETA: 4s - loss: 7.7088 - accuracy: 0.4972
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7221 - accuracy: 0.4964
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7258 - accuracy: 0.4961
15000/25000 [=================>............] - ETA: 3s - loss: 7.7249 - accuracy: 0.4962
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7260 - accuracy: 0.4961
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7180 - accuracy: 0.4966
18000/25000 [====================>.........] - ETA: 2s - loss: 7.7143 - accuracy: 0.4969
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6900 - accuracy: 0.4985
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6827 - accuracy: 0.4990
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6878 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6766 - accuracy: 0.4993
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6717 - accuracy: 0.4997
25000/25000 [==============================] - 9s 366us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f0cc2a439b0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f0ca4cd1160> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.2646 - accuracy: 0.4610
 2000/25000 [=>............................] - ETA: 9s - loss: 7.8123 - accuracy: 0.4905 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8251 - accuracy: 0.4897
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6858 - accuracy: 0.4988
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.7218 - accuracy: 0.4964
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7612 - accuracy: 0.4938
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7258 - accuracy: 0.4961
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7107 - accuracy: 0.4971
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6871 - accuracy: 0.4987
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6896 - accuracy: 0.4985
11000/25000 [============>.................] - ETA: 4s - loss: 7.6638 - accuracy: 0.5002
12000/25000 [=============>................] - ETA: 4s - loss: 7.6436 - accuracy: 0.5015
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6654 - accuracy: 0.5001
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6579 - accuracy: 0.5006
15000/25000 [=================>............] - ETA: 3s - loss: 7.6574 - accuracy: 0.5006
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6360 - accuracy: 0.5020
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6305 - accuracy: 0.5024
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6377 - accuracy: 0.5019
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6545 - accuracy: 0.5008
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6544 - accuracy: 0.5008
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6637 - accuracy: 0.5002
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6457 - accuracy: 0.5014
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6613 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6737 - accuracy: 0.4995
25000/25000 [==============================] - 9s 373us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.4366 - accuracy: 0.5150
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5593 - accuracy: 0.5070 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5900 - accuracy: 0.5050
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7510 - accuracy: 0.4945
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6482 - accuracy: 0.5012
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6538 - accuracy: 0.5008
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6885 - accuracy: 0.4986
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6781 - accuracy: 0.4992
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6768 - accuracy: 0.4993
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6973 - accuracy: 0.4980
11000/25000 [============>.................] - ETA: 4s - loss: 7.6903 - accuracy: 0.4985
12000/25000 [=============>................] - ETA: 4s - loss: 7.7216 - accuracy: 0.4964
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7221 - accuracy: 0.4964
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7225 - accuracy: 0.4964
15000/25000 [=================>............] - ETA: 3s - loss: 7.7239 - accuracy: 0.4963
16000/25000 [==================>...........] - ETA: 2s - loss: 7.7193 - accuracy: 0.4966
17000/25000 [===================>..........] - ETA: 2s - loss: 7.7063 - accuracy: 0.4974
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6981 - accuracy: 0.4979
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7021 - accuracy: 0.4977
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6958 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6841 - accuracy: 0.4989
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6875 - accuracy: 0.4986
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6686 - accuracy: 0.4999
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6602 - accuracy: 0.5004
25000/25000 [==============================] - 9s 370us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   e64a6a8..a8427b7  master     -> origin/master
Updating e64a6a8..a8427b7
Fast-forward
 .../20200519/list_log_pullrequest_20200519.md      |   2 +-
 error_list/20200519/list_log_testall_20200519.md   | 103 +++++++++++++++++++++
 2 files changed, 104 insertions(+), 1 deletion(-)
[master 9e2b244] ml_store
 1 file changed, 325 insertions(+)
To github.com:arita37/mlmodels_store.git
   a8427b7..9e2b244  master -> master





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

13/13 [==============================] - 1s 106ms/step - loss: nan
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
[master a28fd8e] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   9e2b244..a28fd8e  master -> master





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

    8192/11490434 [..............................] - ETA: 2s
 1294336/11490434 [==>...........................] - ETA: 0s
 4145152/11490434 [=========>....................] - ETA: 0s
 8257536/11490434 [====================>.........] - ETA: 0s
10690560/11490434 [==========================>...] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:17 - loss: 2.3013 - categorical_accuracy: 0.1250
   64/60000 [..............................] - ETA: 4:32 - loss: 2.3355 - categorical_accuracy: 0.1250
   96/60000 [..............................] - ETA: 3:38 - loss: 2.3065 - categorical_accuracy: 0.1562
  128/60000 [..............................] - ETA: 3:09 - loss: 2.2944 - categorical_accuracy: 0.1875
  160/60000 [..............................] - ETA: 2:52 - loss: 2.2785 - categorical_accuracy: 0.2313
  192/60000 [..............................] - ETA: 2:44 - loss: 2.2484 - categorical_accuracy: 0.2448
  224/60000 [..............................] - ETA: 2:38 - loss: 2.2298 - categorical_accuracy: 0.2589
  256/60000 [..............................] - ETA: 2:31 - loss: 2.1960 - categorical_accuracy: 0.2539
  288/60000 [..............................] - ETA: 2:27 - loss: 2.1770 - categorical_accuracy: 0.2569
  320/60000 [..............................] - ETA: 2:23 - loss: 2.1364 - categorical_accuracy: 0.2812
  352/60000 [..............................] - ETA: 2:20 - loss: 2.0734 - categorical_accuracy: 0.3097
  384/60000 [..............................] - ETA: 2:17 - loss: 2.0475 - categorical_accuracy: 0.3151
  416/60000 [..............................] - ETA: 2:14 - loss: 2.0178 - categorical_accuracy: 0.3269
  448/60000 [..............................] - ETA: 2:12 - loss: 1.9802 - categorical_accuracy: 0.3460
  480/60000 [..............................] - ETA: 2:09 - loss: 1.9560 - categorical_accuracy: 0.3542
  512/60000 [..............................] - ETA: 2:07 - loss: 1.9351 - categorical_accuracy: 0.3633
  544/60000 [..............................] - ETA: 2:05 - loss: 1.9216 - categorical_accuracy: 0.3676
  576/60000 [..............................] - ETA: 2:04 - loss: 1.8906 - categorical_accuracy: 0.3767
  608/60000 [..............................] - ETA: 2:03 - loss: 1.8505 - categorical_accuracy: 0.3882
  640/60000 [..............................] - ETA: 2:02 - loss: 1.8312 - categorical_accuracy: 0.3953
  672/60000 [..............................] - ETA: 2:01 - loss: 1.7942 - categorical_accuracy: 0.4092
  704/60000 [..............................] - ETA: 2:00 - loss: 1.7694 - categorical_accuracy: 0.4134
  736/60000 [..............................] - ETA: 1:59 - loss: 1.7452 - categorical_accuracy: 0.4198
  768/60000 [..............................] - ETA: 1:59 - loss: 1.7258 - categorical_accuracy: 0.4258
  800/60000 [..............................] - ETA: 1:58 - loss: 1.7099 - categorical_accuracy: 0.4300
  832/60000 [..............................] - ETA: 1:57 - loss: 1.6948 - categorical_accuracy: 0.4279
  864/60000 [..............................] - ETA: 1:57 - loss: 1.6658 - categorical_accuracy: 0.4363
  896/60000 [..............................] - ETA: 1:57 - loss: 1.6517 - categorical_accuracy: 0.4420
  928/60000 [..............................] - ETA: 1:57 - loss: 1.6410 - categorical_accuracy: 0.4461
  960/60000 [..............................] - ETA: 1:56 - loss: 1.6161 - categorical_accuracy: 0.4563
  992/60000 [..............................] - ETA: 1:56 - loss: 1.5961 - categorical_accuracy: 0.4627
 1024/60000 [..............................] - ETA: 1:55 - loss: 1.5643 - categorical_accuracy: 0.4746
 1056/60000 [..............................] - ETA: 1:55 - loss: 1.5316 - categorical_accuracy: 0.4877
 1088/60000 [..............................] - ETA: 1:54 - loss: 1.5066 - categorical_accuracy: 0.4945
 1120/60000 [..............................] - ETA: 1:54 - loss: 1.4888 - categorical_accuracy: 0.5045
 1152/60000 [..............................] - ETA: 1:54 - loss: 1.4672 - categorical_accuracy: 0.5113
 1184/60000 [..............................] - ETA: 1:53 - loss: 1.4512 - categorical_accuracy: 0.5160
 1216/60000 [..............................] - ETA: 1:53 - loss: 1.4334 - categorical_accuracy: 0.5230
 1248/60000 [..............................] - ETA: 1:53 - loss: 1.4203 - categorical_accuracy: 0.5264
 1280/60000 [..............................] - ETA: 1:53 - loss: 1.3988 - categorical_accuracy: 0.5344
 1312/60000 [..............................] - ETA: 1:53 - loss: 1.3923 - categorical_accuracy: 0.5366
 1344/60000 [..............................] - ETA: 1:52 - loss: 1.3735 - categorical_accuracy: 0.5432
 1376/60000 [..............................] - ETA: 1:52 - loss: 1.3636 - categorical_accuracy: 0.5451
 1408/60000 [..............................] - ETA: 1:52 - loss: 1.3485 - categorical_accuracy: 0.5504
 1440/60000 [..............................] - ETA: 1:52 - loss: 1.3333 - categorical_accuracy: 0.5542
 1472/60000 [..............................] - ETA: 1:51 - loss: 1.3193 - categorical_accuracy: 0.5591
 1504/60000 [..............................] - ETA: 1:51 - loss: 1.3089 - categorical_accuracy: 0.5618
 1536/60000 [..............................] - ETA: 1:51 - loss: 1.2975 - categorical_accuracy: 0.5671
 1568/60000 [..............................] - ETA: 1:51 - loss: 1.2797 - categorical_accuracy: 0.5727
 1600/60000 [..............................] - ETA: 1:51 - loss: 1.2697 - categorical_accuracy: 0.5756
 1632/60000 [..............................] - ETA: 1:51 - loss: 1.2630 - categorical_accuracy: 0.5784
 1664/60000 [..............................] - ETA: 1:50 - loss: 1.2499 - categorical_accuracy: 0.5829
 1696/60000 [..............................] - ETA: 1:51 - loss: 1.2384 - categorical_accuracy: 0.5873
 1728/60000 [..............................] - ETA: 1:51 - loss: 1.2274 - categorical_accuracy: 0.5909
 1760/60000 [..............................] - ETA: 1:50 - loss: 1.2226 - categorical_accuracy: 0.5938
 1792/60000 [..............................] - ETA: 1:50 - loss: 1.2135 - categorical_accuracy: 0.5960
 1824/60000 [..............................] - ETA: 1:50 - loss: 1.2049 - categorical_accuracy: 0.5987
 1856/60000 [..............................] - ETA: 1:50 - loss: 1.1933 - categorical_accuracy: 0.6029
 1888/60000 [..............................] - ETA: 1:50 - loss: 1.1814 - categorical_accuracy: 0.6059
 1920/60000 [..............................] - ETA: 1:50 - loss: 1.1740 - categorical_accuracy: 0.6089
 1952/60000 [..............................] - ETA: 1:50 - loss: 1.1647 - categorical_accuracy: 0.6117
 1984/60000 [..............................] - ETA: 1:50 - loss: 1.1507 - categorical_accuracy: 0.6169
 2016/60000 [>.............................] - ETA: 1:50 - loss: 1.1398 - categorical_accuracy: 0.6200
 2048/60000 [>.............................] - ETA: 1:50 - loss: 1.1414 - categorical_accuracy: 0.6211
 2080/60000 [>.............................] - ETA: 1:50 - loss: 1.1346 - categorical_accuracy: 0.6231
 2112/60000 [>.............................] - ETA: 1:50 - loss: 1.1244 - categorical_accuracy: 0.6274
 2144/60000 [>.............................] - ETA: 1:50 - loss: 1.1177 - categorical_accuracy: 0.6297
 2176/60000 [>.............................] - ETA: 1:49 - loss: 1.1065 - categorical_accuracy: 0.6337
 2208/60000 [>.............................] - ETA: 1:49 - loss: 1.0960 - categorical_accuracy: 0.6372
 2240/60000 [>.............................] - ETA: 1:49 - loss: 1.0908 - categorical_accuracy: 0.6406
 2272/60000 [>.............................] - ETA: 1:49 - loss: 1.0789 - categorical_accuracy: 0.6448
 2304/60000 [>.............................] - ETA: 1:49 - loss: 1.0733 - categorical_accuracy: 0.6471
 2336/60000 [>.............................] - ETA: 1:49 - loss: 1.0687 - categorical_accuracy: 0.6498
 2368/60000 [>.............................] - ETA: 1:49 - loss: 1.0612 - categorical_accuracy: 0.6520
 2400/60000 [>.............................] - ETA: 1:48 - loss: 1.0554 - categorical_accuracy: 0.6546
 2432/60000 [>.............................] - ETA: 1:48 - loss: 1.0465 - categorical_accuracy: 0.6575
 2464/60000 [>.............................] - ETA: 1:48 - loss: 1.0388 - categorical_accuracy: 0.6599
 2496/60000 [>.............................] - ETA: 1:48 - loss: 1.0351 - categorical_accuracy: 0.6611
 2528/60000 [>.............................] - ETA: 1:48 - loss: 1.0268 - categorical_accuracy: 0.6646
 2560/60000 [>.............................] - ETA: 1:47 - loss: 1.0163 - categorical_accuracy: 0.6680
 2592/60000 [>.............................] - ETA: 1:47 - loss: 1.0085 - categorical_accuracy: 0.6709
 2624/60000 [>.............................] - ETA: 1:47 - loss: 1.0002 - categorical_accuracy: 0.6734
 2656/60000 [>.............................] - ETA: 1:47 - loss: 0.9950 - categorical_accuracy: 0.6755
 2688/60000 [>.............................] - ETA: 1:47 - loss: 0.9871 - categorical_accuracy: 0.6786
 2720/60000 [>.............................] - ETA: 1:47 - loss: 0.9790 - categorical_accuracy: 0.6812
 2752/60000 [>.............................] - ETA: 1:47 - loss: 0.9703 - categorical_accuracy: 0.6846
 2784/60000 [>.............................] - ETA: 1:46 - loss: 0.9642 - categorical_accuracy: 0.6864
 2816/60000 [>.............................] - ETA: 1:46 - loss: 0.9568 - categorical_accuracy: 0.6889
 2848/60000 [>.............................] - ETA: 1:46 - loss: 0.9526 - categorical_accuracy: 0.6903
 2880/60000 [>.............................] - ETA: 1:46 - loss: 0.9450 - categorical_accuracy: 0.6931
 2912/60000 [>.............................] - ETA: 1:45 - loss: 0.9389 - categorical_accuracy: 0.6954
 2944/60000 [>.............................] - ETA: 1:45 - loss: 0.9355 - categorical_accuracy: 0.6970
 2976/60000 [>.............................] - ETA: 1:45 - loss: 0.9288 - categorical_accuracy: 0.6989
 3008/60000 [>.............................] - ETA: 1:45 - loss: 0.9230 - categorical_accuracy: 0.6998
 3040/60000 [>.............................] - ETA: 1:45 - loss: 0.9150 - categorical_accuracy: 0.7023
 3072/60000 [>.............................] - ETA: 1:45 - loss: 0.9075 - categorical_accuracy: 0.7051
 3104/60000 [>.............................] - ETA: 1:45 - loss: 0.9042 - categorical_accuracy: 0.7062
 3136/60000 [>.............................] - ETA: 1:44 - loss: 0.8962 - categorical_accuracy: 0.7092
 3168/60000 [>.............................] - ETA: 1:44 - loss: 0.8925 - categorical_accuracy: 0.7105
 3200/60000 [>.............................] - ETA: 1:44 - loss: 0.8895 - categorical_accuracy: 0.7122
 3232/60000 [>.............................] - ETA: 1:44 - loss: 0.8821 - categorical_accuracy: 0.7147
 3264/60000 [>.............................] - ETA: 1:44 - loss: 0.8785 - categorical_accuracy: 0.7157
 3296/60000 [>.............................] - ETA: 1:44 - loss: 0.8750 - categorical_accuracy: 0.7169
 3328/60000 [>.............................] - ETA: 1:44 - loss: 0.8731 - categorical_accuracy: 0.7166
 3360/60000 [>.............................] - ETA: 1:44 - loss: 0.8660 - categorical_accuracy: 0.7190
 3392/60000 [>.............................] - ETA: 1:44 - loss: 0.8608 - categorical_accuracy: 0.7205
 3424/60000 [>.............................] - ETA: 1:44 - loss: 0.8543 - categorical_accuracy: 0.7228
 3456/60000 [>.............................] - ETA: 1:44 - loss: 0.8500 - categorical_accuracy: 0.7245
 3488/60000 [>.............................] - ETA: 1:44 - loss: 0.8441 - categorical_accuracy: 0.7265
 3520/60000 [>.............................] - ETA: 1:44 - loss: 0.8389 - categorical_accuracy: 0.7276
 3552/60000 [>.............................] - ETA: 1:44 - loss: 0.8365 - categorical_accuracy: 0.7280
 3584/60000 [>.............................] - ETA: 1:43 - loss: 0.8322 - categorical_accuracy: 0.7294
 3616/60000 [>.............................] - ETA: 1:43 - loss: 0.8299 - categorical_accuracy: 0.7301
 3648/60000 [>.............................] - ETA: 1:43 - loss: 0.8280 - categorical_accuracy: 0.7305
 3680/60000 [>.............................] - ETA: 1:43 - loss: 0.8262 - categorical_accuracy: 0.7315
 3712/60000 [>.............................] - ETA: 1:43 - loss: 0.8236 - categorical_accuracy: 0.7328
 3744/60000 [>.............................] - ETA: 1:43 - loss: 0.8201 - categorical_accuracy: 0.7337
 3776/60000 [>.............................] - ETA: 1:43 - loss: 0.8167 - categorical_accuracy: 0.7354
 3808/60000 [>.............................] - ETA: 1:43 - loss: 0.8125 - categorical_accuracy: 0.7363
 3840/60000 [>.............................] - ETA: 1:43 - loss: 0.8115 - categorical_accuracy: 0.7367
 3872/60000 [>.............................] - ETA: 1:43 - loss: 0.8136 - categorical_accuracy: 0.7371
 3904/60000 [>.............................] - ETA: 1:43 - loss: 0.8095 - categorical_accuracy: 0.7387
 3936/60000 [>.............................] - ETA: 1:43 - loss: 0.8054 - categorical_accuracy: 0.7406
 3968/60000 [>.............................] - ETA: 1:43 - loss: 0.8026 - categorical_accuracy: 0.7414
 4000/60000 [=>............................] - ETA: 1:42 - loss: 0.7980 - categorical_accuracy: 0.7427
 4032/60000 [=>............................] - ETA: 1:42 - loss: 0.7948 - categorical_accuracy: 0.7438
 4064/60000 [=>............................] - ETA: 1:42 - loss: 0.7925 - categorical_accuracy: 0.7446
 4096/60000 [=>............................] - ETA: 1:42 - loss: 0.7884 - categorical_accuracy: 0.7461
 4128/60000 [=>............................] - ETA: 1:42 - loss: 0.7878 - categorical_accuracy: 0.7469
 4160/60000 [=>............................] - ETA: 1:42 - loss: 0.7831 - categorical_accuracy: 0.7483
 4192/60000 [=>............................] - ETA: 1:42 - loss: 0.7799 - categorical_accuracy: 0.7493
 4224/60000 [=>............................] - ETA: 1:42 - loss: 0.7766 - categorical_accuracy: 0.7502
 4256/60000 [=>............................] - ETA: 1:41 - loss: 0.7738 - categorical_accuracy: 0.7509
 4288/60000 [=>............................] - ETA: 1:41 - loss: 0.7701 - categorical_accuracy: 0.7521
 4320/60000 [=>............................] - ETA: 1:41 - loss: 0.7674 - categorical_accuracy: 0.7528
 4352/60000 [=>............................] - ETA: 1:41 - loss: 0.7626 - categorical_accuracy: 0.7544
 4384/60000 [=>............................] - ETA: 1:41 - loss: 0.7591 - categorical_accuracy: 0.7555
 4416/60000 [=>............................] - ETA: 1:41 - loss: 0.7550 - categorical_accuracy: 0.7568
 4448/60000 [=>............................] - ETA: 1:41 - loss: 0.7513 - categorical_accuracy: 0.7579
 4480/60000 [=>............................] - ETA: 1:41 - loss: 0.7498 - categorical_accuracy: 0.7589
 4512/60000 [=>............................] - ETA: 1:41 - loss: 0.7463 - categorical_accuracy: 0.7602
 4544/60000 [=>............................] - ETA: 1:41 - loss: 0.7425 - categorical_accuracy: 0.7614
 4576/60000 [=>............................] - ETA: 1:41 - loss: 0.7397 - categorical_accuracy: 0.7620
 4608/60000 [=>............................] - ETA: 1:41 - loss: 0.7368 - categorical_accuracy: 0.7630
 4640/60000 [=>............................] - ETA: 1:41 - loss: 0.7328 - categorical_accuracy: 0.7642
 4672/60000 [=>............................] - ETA: 1:41 - loss: 0.7284 - categorical_accuracy: 0.7658
 4704/60000 [=>............................] - ETA: 1:41 - loss: 0.7253 - categorical_accuracy: 0.7670
 4736/60000 [=>............................] - ETA: 1:41 - loss: 0.7219 - categorical_accuracy: 0.7682
 4768/60000 [=>............................] - ETA: 1:41 - loss: 0.7180 - categorical_accuracy: 0.7693
 4800/60000 [=>............................] - ETA: 1:40 - loss: 0.7146 - categorical_accuracy: 0.7706
 4832/60000 [=>............................] - ETA: 1:40 - loss: 0.7113 - categorical_accuracy: 0.7715
 4864/60000 [=>............................] - ETA: 1:40 - loss: 0.7094 - categorical_accuracy: 0.7722
 4896/60000 [=>............................] - ETA: 1:40 - loss: 0.7050 - categorical_accuracy: 0.7737
 4928/60000 [=>............................] - ETA: 1:40 - loss: 0.7023 - categorical_accuracy: 0.7748
 4960/60000 [=>............................] - ETA: 1:40 - loss: 0.6996 - categorical_accuracy: 0.7756
 4992/60000 [=>............................] - ETA: 1:40 - loss: 0.6987 - categorical_accuracy: 0.7762
 5024/60000 [=>............................] - ETA: 1:40 - loss: 0.6977 - categorical_accuracy: 0.7771
 5056/60000 [=>............................] - ETA: 1:40 - loss: 0.6946 - categorical_accuracy: 0.7781
 5088/60000 [=>............................] - ETA: 1:40 - loss: 0.6927 - categorical_accuracy: 0.7785
 5120/60000 [=>............................] - ETA: 1:40 - loss: 0.6900 - categorical_accuracy: 0.7795
 5152/60000 [=>............................] - ETA: 1:40 - loss: 0.6868 - categorical_accuracy: 0.7807
 5184/60000 [=>............................] - ETA: 1:40 - loss: 0.6841 - categorical_accuracy: 0.7816
 5216/60000 [=>............................] - ETA: 1:39 - loss: 0.6815 - categorical_accuracy: 0.7826
 5248/60000 [=>............................] - ETA: 1:39 - loss: 0.6787 - categorical_accuracy: 0.7837
 5280/60000 [=>............................] - ETA: 1:39 - loss: 0.6789 - categorical_accuracy: 0.7841
 5312/60000 [=>............................] - ETA: 1:39 - loss: 0.6764 - categorical_accuracy: 0.7850
 5344/60000 [=>............................] - ETA: 1:39 - loss: 0.6736 - categorical_accuracy: 0.7859
 5376/60000 [=>............................] - ETA: 1:39 - loss: 0.6706 - categorical_accuracy: 0.7868
 5408/60000 [=>............................] - ETA: 1:39 - loss: 0.6678 - categorical_accuracy: 0.7875
 5440/60000 [=>............................] - ETA: 1:39 - loss: 0.6644 - categorical_accuracy: 0.7886
 5472/60000 [=>............................] - ETA: 1:39 - loss: 0.6615 - categorical_accuracy: 0.7897
 5504/60000 [=>............................] - ETA: 1:39 - loss: 0.6591 - categorical_accuracy: 0.7902
 5536/60000 [=>............................] - ETA: 1:39 - loss: 0.6566 - categorical_accuracy: 0.7912
 5568/60000 [=>............................] - ETA: 1:39 - loss: 0.6536 - categorical_accuracy: 0.7920
 5600/60000 [=>............................] - ETA: 1:39 - loss: 0.6514 - categorical_accuracy: 0.7927
 5632/60000 [=>............................] - ETA: 1:39 - loss: 0.6495 - categorical_accuracy: 0.7935
 5664/60000 [=>............................] - ETA: 1:39 - loss: 0.6471 - categorical_accuracy: 0.7941
 5696/60000 [=>............................] - ETA: 1:39 - loss: 0.6463 - categorical_accuracy: 0.7944
 5728/60000 [=>............................] - ETA: 1:39 - loss: 0.6436 - categorical_accuracy: 0.7950
 5760/60000 [=>............................] - ETA: 1:38 - loss: 0.6413 - categorical_accuracy: 0.7957
 5792/60000 [=>............................] - ETA: 1:38 - loss: 0.6402 - categorical_accuracy: 0.7959
 5824/60000 [=>............................] - ETA: 1:38 - loss: 0.6375 - categorical_accuracy: 0.7969
 5856/60000 [=>............................] - ETA: 1:38 - loss: 0.6347 - categorical_accuracy: 0.7978
 5888/60000 [=>............................] - ETA: 1:38 - loss: 0.6339 - categorical_accuracy: 0.7982
 5920/60000 [=>............................] - ETA: 1:38 - loss: 0.6322 - categorical_accuracy: 0.7988
 5952/60000 [=>............................] - ETA: 1:38 - loss: 0.6302 - categorical_accuracy: 0.7992
 5984/60000 [=>............................] - ETA: 1:38 - loss: 0.6276 - categorical_accuracy: 0.8000
 6016/60000 [==>...........................] - ETA: 1:38 - loss: 0.6273 - categorical_accuracy: 0.8002
 6048/60000 [==>...........................] - ETA: 1:38 - loss: 0.6259 - categorical_accuracy: 0.8008
 6080/60000 [==>...........................] - ETA: 1:38 - loss: 0.6267 - categorical_accuracy: 0.8010
 6112/60000 [==>...........................] - ETA: 1:37 - loss: 0.6250 - categorical_accuracy: 0.8015
 6144/60000 [==>...........................] - ETA: 1:37 - loss: 0.6232 - categorical_accuracy: 0.8019
 6176/60000 [==>...........................] - ETA: 1:37 - loss: 0.6217 - categorical_accuracy: 0.8023
 6208/60000 [==>...........................] - ETA: 1:37 - loss: 0.6200 - categorical_accuracy: 0.8028
 6240/60000 [==>...........................] - ETA: 1:37 - loss: 0.6189 - categorical_accuracy: 0.8032
 6272/60000 [==>...........................] - ETA: 1:37 - loss: 0.6165 - categorical_accuracy: 0.8040
 6304/60000 [==>...........................] - ETA: 1:37 - loss: 0.6147 - categorical_accuracy: 0.8047
 6336/60000 [==>...........................] - ETA: 1:37 - loss: 0.6119 - categorical_accuracy: 0.8057
 6368/60000 [==>...........................] - ETA: 1:37 - loss: 0.6108 - categorical_accuracy: 0.8059
 6400/60000 [==>...........................] - ETA: 1:37 - loss: 0.6106 - categorical_accuracy: 0.8061
 6432/60000 [==>...........................] - ETA: 1:37 - loss: 0.6095 - categorical_accuracy: 0.8063
 6464/60000 [==>...........................] - ETA: 1:37 - loss: 0.6068 - categorical_accuracy: 0.8072
 6496/60000 [==>...........................] - ETA: 1:37 - loss: 0.6071 - categorical_accuracy: 0.8073
 6528/60000 [==>...........................] - ETA: 1:36 - loss: 0.6054 - categorical_accuracy: 0.8079
 6560/60000 [==>...........................] - ETA: 1:36 - loss: 0.6035 - categorical_accuracy: 0.8085
 6592/60000 [==>...........................] - ETA: 1:36 - loss: 0.6037 - categorical_accuracy: 0.8086
 6624/60000 [==>...........................] - ETA: 1:36 - loss: 0.6015 - categorical_accuracy: 0.8095
 6656/60000 [==>...........................] - ETA: 1:36 - loss: 0.6005 - categorical_accuracy: 0.8095
 6688/60000 [==>...........................] - ETA: 1:36 - loss: 0.5984 - categorical_accuracy: 0.8103
 6720/60000 [==>...........................] - ETA: 1:36 - loss: 0.5961 - categorical_accuracy: 0.8110
 6752/60000 [==>...........................] - ETA: 1:36 - loss: 0.5949 - categorical_accuracy: 0.8115
 6784/60000 [==>...........................] - ETA: 1:36 - loss: 0.5940 - categorical_accuracy: 0.8116
 6816/60000 [==>...........................] - ETA: 1:36 - loss: 0.5917 - categorical_accuracy: 0.8124
 6848/60000 [==>...........................] - ETA: 1:36 - loss: 0.5900 - categorical_accuracy: 0.8128
 6880/60000 [==>...........................] - ETA: 1:36 - loss: 0.5889 - categorical_accuracy: 0.8134
 6912/60000 [==>...........................] - ETA: 1:36 - loss: 0.5870 - categorical_accuracy: 0.8141
 6944/60000 [==>...........................] - ETA: 1:36 - loss: 0.5848 - categorical_accuracy: 0.8149
 6976/60000 [==>...........................] - ETA: 1:36 - loss: 0.5831 - categorical_accuracy: 0.8155
 7008/60000 [==>...........................] - ETA: 1:35 - loss: 0.5811 - categorical_accuracy: 0.8162
 7040/60000 [==>...........................] - ETA: 1:35 - loss: 0.5789 - categorical_accuracy: 0.8170
 7072/60000 [==>...........................] - ETA: 1:35 - loss: 0.5771 - categorical_accuracy: 0.8176
 7104/60000 [==>...........................] - ETA: 1:35 - loss: 0.5759 - categorical_accuracy: 0.8180
 7136/60000 [==>...........................] - ETA: 1:35 - loss: 0.5740 - categorical_accuracy: 0.8187
 7168/60000 [==>...........................] - ETA: 1:35 - loss: 0.5718 - categorical_accuracy: 0.8195
 7200/60000 [==>...........................] - ETA: 1:35 - loss: 0.5697 - categorical_accuracy: 0.8200
 7232/60000 [==>...........................] - ETA: 1:35 - loss: 0.5684 - categorical_accuracy: 0.8205
 7264/60000 [==>...........................] - ETA: 1:35 - loss: 0.5665 - categorical_accuracy: 0.8210
 7296/60000 [==>...........................] - ETA: 1:35 - loss: 0.5653 - categorical_accuracy: 0.8214
 7328/60000 [==>...........................] - ETA: 1:35 - loss: 0.5639 - categorical_accuracy: 0.8218
 7360/60000 [==>...........................] - ETA: 1:35 - loss: 0.5623 - categorical_accuracy: 0.8223
 7392/60000 [==>...........................] - ETA: 1:35 - loss: 0.5604 - categorical_accuracy: 0.8229
 7424/60000 [==>...........................] - ETA: 1:35 - loss: 0.5587 - categorical_accuracy: 0.8233
 7456/60000 [==>...........................] - ETA: 1:35 - loss: 0.5567 - categorical_accuracy: 0.8239
 7488/60000 [==>...........................] - ETA: 1:34 - loss: 0.5554 - categorical_accuracy: 0.8244
 7520/60000 [==>...........................] - ETA: 1:34 - loss: 0.5549 - categorical_accuracy: 0.8246
 7552/60000 [==>...........................] - ETA: 1:34 - loss: 0.5532 - categorical_accuracy: 0.8252
 7584/60000 [==>...........................] - ETA: 1:34 - loss: 0.5517 - categorical_accuracy: 0.8258
 7616/60000 [==>...........................] - ETA: 1:34 - loss: 0.5498 - categorical_accuracy: 0.8264
 7648/60000 [==>...........................] - ETA: 1:34 - loss: 0.5487 - categorical_accuracy: 0.8266
 7680/60000 [==>...........................] - ETA: 1:34 - loss: 0.5474 - categorical_accuracy: 0.8270
 7712/60000 [==>...........................] - ETA: 1:34 - loss: 0.5473 - categorical_accuracy: 0.8270
 7744/60000 [==>...........................] - ETA: 1:34 - loss: 0.5474 - categorical_accuracy: 0.8272
 7776/60000 [==>...........................] - ETA: 1:34 - loss: 0.5457 - categorical_accuracy: 0.8278
 7808/60000 [==>...........................] - ETA: 1:34 - loss: 0.5438 - categorical_accuracy: 0.8284
 7840/60000 [==>...........................] - ETA: 1:34 - loss: 0.5420 - categorical_accuracy: 0.8290
 7872/60000 [==>...........................] - ETA: 1:33 - loss: 0.5407 - categorical_accuracy: 0.8290
 7904/60000 [==>...........................] - ETA: 1:33 - loss: 0.5388 - categorical_accuracy: 0.8296
 7936/60000 [==>...........................] - ETA: 1:33 - loss: 0.5380 - categorical_accuracy: 0.8300
 7968/60000 [==>...........................] - ETA: 1:33 - loss: 0.5368 - categorical_accuracy: 0.8303
 8000/60000 [===>..........................] - ETA: 1:33 - loss: 0.5353 - categorical_accuracy: 0.8306
 8032/60000 [===>..........................] - ETA: 1:33 - loss: 0.5337 - categorical_accuracy: 0.8312
 8064/60000 [===>..........................] - ETA: 1:33 - loss: 0.5337 - categorical_accuracy: 0.8312
 8096/60000 [===>..........................] - ETA: 1:33 - loss: 0.5329 - categorical_accuracy: 0.8314
 8128/60000 [===>..........................] - ETA: 1:33 - loss: 0.5319 - categorical_accuracy: 0.8318
 8160/60000 [===>..........................] - ETA: 1:33 - loss: 0.5315 - categorical_accuracy: 0.8321
 8192/60000 [===>..........................] - ETA: 1:33 - loss: 0.5309 - categorical_accuracy: 0.8324
 8224/60000 [===>..........................] - ETA: 1:33 - loss: 0.5298 - categorical_accuracy: 0.8327
 8256/60000 [===>..........................] - ETA: 1:33 - loss: 0.5292 - categorical_accuracy: 0.8327
 8288/60000 [===>..........................] - ETA: 1:33 - loss: 0.5279 - categorical_accuracy: 0.8329
 8320/60000 [===>..........................] - ETA: 1:33 - loss: 0.5266 - categorical_accuracy: 0.8334
 8352/60000 [===>..........................] - ETA: 1:33 - loss: 0.5250 - categorical_accuracy: 0.8341
 8384/60000 [===>..........................] - ETA: 1:33 - loss: 0.5247 - categorical_accuracy: 0.8342
 8416/60000 [===>..........................] - ETA: 1:33 - loss: 0.5233 - categorical_accuracy: 0.8347
 8448/60000 [===>..........................] - ETA: 1:32 - loss: 0.5225 - categorical_accuracy: 0.8351
 8480/60000 [===>..........................] - ETA: 1:32 - loss: 0.5209 - categorical_accuracy: 0.8356
 8512/60000 [===>..........................] - ETA: 1:32 - loss: 0.5203 - categorical_accuracy: 0.8358
 8544/60000 [===>..........................] - ETA: 1:32 - loss: 0.5196 - categorical_accuracy: 0.8361
 8576/60000 [===>..........................] - ETA: 1:32 - loss: 0.5183 - categorical_accuracy: 0.8365
 8608/60000 [===>..........................] - ETA: 1:32 - loss: 0.5168 - categorical_accuracy: 0.8371
 8640/60000 [===>..........................] - ETA: 1:32 - loss: 0.5167 - categorical_accuracy: 0.8373
 8672/60000 [===>..........................] - ETA: 1:32 - loss: 0.5156 - categorical_accuracy: 0.8376
 8704/60000 [===>..........................] - ETA: 1:32 - loss: 0.5148 - categorical_accuracy: 0.8379
 8736/60000 [===>..........................] - ETA: 1:32 - loss: 0.5132 - categorical_accuracy: 0.8385
 8768/60000 [===>..........................] - ETA: 1:32 - loss: 0.5123 - categorical_accuracy: 0.8388
 8800/60000 [===>..........................] - ETA: 1:32 - loss: 0.5107 - categorical_accuracy: 0.8394
 8832/60000 [===>..........................] - ETA: 1:32 - loss: 0.5111 - categorical_accuracy: 0.8396
 8864/60000 [===>..........................] - ETA: 1:32 - loss: 0.5097 - categorical_accuracy: 0.8400
 8896/60000 [===>..........................] - ETA: 1:32 - loss: 0.5088 - categorical_accuracy: 0.8405
 8928/60000 [===>..........................] - ETA: 1:32 - loss: 0.5079 - categorical_accuracy: 0.8407
 8960/60000 [===>..........................] - ETA: 1:32 - loss: 0.5065 - categorical_accuracy: 0.8412
 8992/60000 [===>..........................] - ETA: 1:32 - loss: 0.5050 - categorical_accuracy: 0.8416
 9024/60000 [===>..........................] - ETA: 1:32 - loss: 0.5036 - categorical_accuracy: 0.8421
 9056/60000 [===>..........................] - ETA: 1:31 - loss: 0.5019 - categorical_accuracy: 0.8426
 9088/60000 [===>..........................] - ETA: 1:31 - loss: 0.5011 - categorical_accuracy: 0.8430
 9120/60000 [===>..........................] - ETA: 1:31 - loss: 0.5018 - categorical_accuracy: 0.8429
 9152/60000 [===>..........................] - ETA: 1:31 - loss: 0.5008 - categorical_accuracy: 0.8433
 9184/60000 [===>..........................] - ETA: 1:31 - loss: 0.5002 - categorical_accuracy: 0.8435
 9216/60000 [===>..........................] - ETA: 1:31 - loss: 0.4994 - categorical_accuracy: 0.8439
 9248/60000 [===>..........................] - ETA: 1:31 - loss: 0.4989 - categorical_accuracy: 0.8440
 9280/60000 [===>..........................] - ETA: 1:31 - loss: 0.4982 - categorical_accuracy: 0.8442
 9312/60000 [===>..........................] - ETA: 1:31 - loss: 0.4978 - categorical_accuracy: 0.8442
 9344/60000 [===>..........................] - ETA: 1:31 - loss: 0.4974 - categorical_accuracy: 0.8443
 9376/60000 [===>..........................] - ETA: 1:31 - loss: 0.4964 - categorical_accuracy: 0.8445
 9408/60000 [===>..........................] - ETA: 1:31 - loss: 0.4951 - categorical_accuracy: 0.8448
 9440/60000 [===>..........................] - ETA: 1:31 - loss: 0.4938 - categorical_accuracy: 0.8451
 9472/60000 [===>..........................] - ETA: 1:31 - loss: 0.4932 - categorical_accuracy: 0.8453
 9504/60000 [===>..........................] - ETA: 1:31 - loss: 0.4919 - categorical_accuracy: 0.8457
 9536/60000 [===>..........................] - ETA: 1:31 - loss: 0.4910 - categorical_accuracy: 0.8461
 9568/60000 [===>..........................] - ETA: 1:30 - loss: 0.4903 - categorical_accuracy: 0.8463
 9600/60000 [===>..........................] - ETA: 1:30 - loss: 0.4887 - categorical_accuracy: 0.8468
 9632/60000 [===>..........................] - ETA: 1:30 - loss: 0.4874 - categorical_accuracy: 0.8473
 9664/60000 [===>..........................] - ETA: 1:30 - loss: 0.4870 - categorical_accuracy: 0.8475
 9696/60000 [===>..........................] - ETA: 1:30 - loss: 0.4859 - categorical_accuracy: 0.8478
 9728/60000 [===>..........................] - ETA: 1:30 - loss: 0.4848 - categorical_accuracy: 0.8482
 9760/60000 [===>..........................] - ETA: 1:30 - loss: 0.4840 - categorical_accuracy: 0.8484
 9792/60000 [===>..........................] - ETA: 1:30 - loss: 0.4831 - categorical_accuracy: 0.8485
 9824/60000 [===>..........................] - ETA: 1:30 - loss: 0.4820 - categorical_accuracy: 0.8488
 9856/60000 [===>..........................] - ETA: 1:30 - loss: 0.4817 - categorical_accuracy: 0.8489
 9888/60000 [===>..........................] - ETA: 1:30 - loss: 0.4814 - categorical_accuracy: 0.8490
 9920/60000 [===>..........................] - ETA: 1:30 - loss: 0.4805 - categorical_accuracy: 0.8493
 9952/60000 [===>..........................] - ETA: 1:30 - loss: 0.4792 - categorical_accuracy: 0.8497
 9984/60000 [===>..........................] - ETA: 1:30 - loss: 0.4783 - categorical_accuracy: 0.8498
10016/60000 [====>.........................] - ETA: 1:29 - loss: 0.4782 - categorical_accuracy: 0.8499
10048/60000 [====>.........................] - ETA: 1:29 - loss: 0.4771 - categorical_accuracy: 0.8502
10080/60000 [====>.........................] - ETA: 1:29 - loss: 0.4763 - categorical_accuracy: 0.8504
10112/60000 [====>.........................] - ETA: 1:29 - loss: 0.4755 - categorical_accuracy: 0.8507
10144/60000 [====>.........................] - ETA: 1:29 - loss: 0.4742 - categorical_accuracy: 0.8511
10176/60000 [====>.........................] - ETA: 1:29 - loss: 0.4730 - categorical_accuracy: 0.8515
10208/60000 [====>.........................] - ETA: 1:29 - loss: 0.4721 - categorical_accuracy: 0.8519
10240/60000 [====>.........................] - ETA: 1:29 - loss: 0.4708 - categorical_accuracy: 0.8522
10272/60000 [====>.........................] - ETA: 1:29 - loss: 0.4712 - categorical_accuracy: 0.8525
10304/60000 [====>.........................] - ETA: 1:29 - loss: 0.4710 - categorical_accuracy: 0.8528
10336/60000 [====>.........................] - ETA: 1:29 - loss: 0.4699 - categorical_accuracy: 0.8531
10368/60000 [====>.........................] - ETA: 1:29 - loss: 0.4687 - categorical_accuracy: 0.8534
10400/60000 [====>.........................] - ETA: 1:29 - loss: 0.4678 - categorical_accuracy: 0.8537
10432/60000 [====>.........................] - ETA: 1:29 - loss: 0.4668 - categorical_accuracy: 0.8540
10464/60000 [====>.........................] - ETA: 1:28 - loss: 0.4665 - categorical_accuracy: 0.8542
10496/60000 [====>.........................] - ETA: 1:28 - loss: 0.4657 - categorical_accuracy: 0.8543
10528/60000 [====>.........................] - ETA: 1:28 - loss: 0.4651 - categorical_accuracy: 0.8547
10560/60000 [====>.........................] - ETA: 1:28 - loss: 0.4638 - categorical_accuracy: 0.8551
10592/60000 [====>.........................] - ETA: 1:28 - loss: 0.4631 - categorical_accuracy: 0.8553
10624/60000 [====>.........................] - ETA: 1:28 - loss: 0.4629 - categorical_accuracy: 0.8556
10656/60000 [====>.........................] - ETA: 1:28 - loss: 0.4622 - categorical_accuracy: 0.8559
10688/60000 [====>.........................] - ETA: 1:28 - loss: 0.4617 - categorical_accuracy: 0.8560
10720/60000 [====>.........................] - ETA: 1:28 - loss: 0.4604 - categorical_accuracy: 0.8564
10752/60000 [====>.........................] - ETA: 1:28 - loss: 0.4593 - categorical_accuracy: 0.8569
10784/60000 [====>.........................] - ETA: 1:28 - loss: 0.4582 - categorical_accuracy: 0.8573
10816/60000 [====>.........................] - ETA: 1:28 - loss: 0.4569 - categorical_accuracy: 0.8577
10848/60000 [====>.........................] - ETA: 1:28 - loss: 0.4560 - categorical_accuracy: 0.8580
10880/60000 [====>.........................] - ETA: 1:28 - loss: 0.4549 - categorical_accuracy: 0.8584
10912/60000 [====>.........................] - ETA: 1:28 - loss: 0.4539 - categorical_accuracy: 0.8587
10944/60000 [====>.........................] - ETA: 1:28 - loss: 0.4533 - categorical_accuracy: 0.8588
10976/60000 [====>.........................] - ETA: 1:28 - loss: 0.4522 - categorical_accuracy: 0.8591
11008/60000 [====>.........................] - ETA: 1:28 - loss: 0.4511 - categorical_accuracy: 0.8594
11040/60000 [====>.........................] - ETA: 1:28 - loss: 0.4504 - categorical_accuracy: 0.8595
11072/60000 [====>.........................] - ETA: 1:28 - loss: 0.4500 - categorical_accuracy: 0.8597
11104/60000 [====>.........................] - ETA: 1:27 - loss: 0.4490 - categorical_accuracy: 0.8601
11136/60000 [====>.........................] - ETA: 1:27 - loss: 0.4484 - categorical_accuracy: 0.8603
11168/60000 [====>.........................] - ETA: 1:27 - loss: 0.4476 - categorical_accuracy: 0.8604
11200/60000 [====>.........................] - ETA: 1:27 - loss: 0.4471 - categorical_accuracy: 0.8606
11232/60000 [====>.........................] - ETA: 1:27 - loss: 0.4466 - categorical_accuracy: 0.8608
11264/60000 [====>.........................] - ETA: 1:27 - loss: 0.4459 - categorical_accuracy: 0.8610
11296/60000 [====>.........................] - ETA: 1:27 - loss: 0.4450 - categorical_accuracy: 0.8613
11328/60000 [====>.........................] - ETA: 1:27 - loss: 0.4445 - categorical_accuracy: 0.8615
11360/60000 [====>.........................] - ETA: 1:27 - loss: 0.4436 - categorical_accuracy: 0.8617
11392/60000 [====>.........................] - ETA: 1:27 - loss: 0.4428 - categorical_accuracy: 0.8619
11424/60000 [====>.........................] - ETA: 1:27 - loss: 0.4420 - categorical_accuracy: 0.8621
11456/60000 [====>.........................] - ETA: 1:27 - loss: 0.4420 - categorical_accuracy: 0.8619
11488/60000 [====>.........................] - ETA: 1:27 - loss: 0.4410 - categorical_accuracy: 0.8622
11520/60000 [====>.........................] - ETA: 1:27 - loss: 0.4404 - categorical_accuracy: 0.8624
11552/60000 [====>.........................] - ETA: 1:27 - loss: 0.4393 - categorical_accuracy: 0.8628
11584/60000 [====>.........................] - ETA: 1:27 - loss: 0.4387 - categorical_accuracy: 0.8631
11616/60000 [====>.........................] - ETA: 1:27 - loss: 0.4386 - categorical_accuracy: 0.8630
11648/60000 [====>.........................] - ETA: 1:27 - loss: 0.4378 - categorical_accuracy: 0.8632
11680/60000 [====>.........................] - ETA: 1:26 - loss: 0.4371 - categorical_accuracy: 0.8635
11712/60000 [====>.........................] - ETA: 1:26 - loss: 0.4367 - categorical_accuracy: 0.8637
11744/60000 [====>.........................] - ETA: 1:26 - loss: 0.4358 - categorical_accuracy: 0.8641
11776/60000 [====>.........................] - ETA: 1:26 - loss: 0.4347 - categorical_accuracy: 0.8645
11808/60000 [====>.........................] - ETA: 1:26 - loss: 0.4338 - categorical_accuracy: 0.8647
11840/60000 [====>.........................] - ETA: 1:26 - loss: 0.4330 - categorical_accuracy: 0.8649
11872/60000 [====>.........................] - ETA: 1:26 - loss: 0.4324 - categorical_accuracy: 0.8651
11904/60000 [====>.........................] - ETA: 1:26 - loss: 0.4319 - categorical_accuracy: 0.8653
11936/60000 [====>.........................] - ETA: 1:26 - loss: 0.4317 - categorical_accuracy: 0.8653
11968/60000 [====>.........................] - ETA: 1:26 - loss: 0.4312 - categorical_accuracy: 0.8654
12000/60000 [=====>........................] - ETA: 1:26 - loss: 0.4305 - categorical_accuracy: 0.8655
12032/60000 [=====>........................] - ETA: 1:26 - loss: 0.4296 - categorical_accuracy: 0.8659
12064/60000 [=====>........................] - ETA: 1:26 - loss: 0.4293 - categorical_accuracy: 0.8660
12096/60000 [=====>........................] - ETA: 1:26 - loss: 0.4282 - categorical_accuracy: 0.8663
12128/60000 [=====>........................] - ETA: 1:26 - loss: 0.4280 - categorical_accuracy: 0.8664
12160/60000 [=====>........................] - ETA: 1:26 - loss: 0.4280 - categorical_accuracy: 0.8664
12192/60000 [=====>........................] - ETA: 1:26 - loss: 0.4272 - categorical_accuracy: 0.8666
12224/60000 [=====>........................] - ETA: 1:25 - loss: 0.4271 - categorical_accuracy: 0.8667
12256/60000 [=====>........................] - ETA: 1:25 - loss: 0.4260 - categorical_accuracy: 0.8671
12288/60000 [=====>........................] - ETA: 1:25 - loss: 0.4251 - categorical_accuracy: 0.8674
12320/60000 [=====>........................] - ETA: 1:25 - loss: 0.4243 - categorical_accuracy: 0.8676
12352/60000 [=====>........................] - ETA: 1:25 - loss: 0.4236 - categorical_accuracy: 0.8679
12384/60000 [=====>........................] - ETA: 1:25 - loss: 0.4230 - categorical_accuracy: 0.8681
12416/60000 [=====>........................] - ETA: 1:25 - loss: 0.4220 - categorical_accuracy: 0.8684
12448/60000 [=====>........................] - ETA: 1:25 - loss: 0.4216 - categorical_accuracy: 0.8684
12480/60000 [=====>........................] - ETA: 1:25 - loss: 0.4211 - categorical_accuracy: 0.8684
12512/60000 [=====>........................] - ETA: 1:25 - loss: 0.4203 - categorical_accuracy: 0.8687
12544/60000 [=====>........................] - ETA: 1:25 - loss: 0.4196 - categorical_accuracy: 0.8689
12576/60000 [=====>........................] - ETA: 1:25 - loss: 0.4191 - categorical_accuracy: 0.8691
12608/60000 [=====>........................] - ETA: 1:25 - loss: 0.4186 - categorical_accuracy: 0.8691
12640/60000 [=====>........................] - ETA: 1:25 - loss: 0.4178 - categorical_accuracy: 0.8693
12672/60000 [=====>........................] - ETA: 1:25 - loss: 0.4169 - categorical_accuracy: 0.8696
12704/60000 [=====>........................] - ETA: 1:25 - loss: 0.4159 - categorical_accuracy: 0.8699
12736/60000 [=====>........................] - ETA: 1:25 - loss: 0.4149 - categorical_accuracy: 0.8702
12768/60000 [=====>........................] - ETA: 1:24 - loss: 0.4141 - categorical_accuracy: 0.8705
12800/60000 [=====>........................] - ETA: 1:24 - loss: 0.4134 - categorical_accuracy: 0.8706
12832/60000 [=====>........................] - ETA: 1:24 - loss: 0.4124 - categorical_accuracy: 0.8709
12864/60000 [=====>........................] - ETA: 1:24 - loss: 0.4118 - categorical_accuracy: 0.8711
12896/60000 [=====>........................] - ETA: 1:24 - loss: 0.4111 - categorical_accuracy: 0.8713
12928/60000 [=====>........................] - ETA: 1:24 - loss: 0.4106 - categorical_accuracy: 0.8714
12960/60000 [=====>........................] - ETA: 1:24 - loss: 0.4097 - categorical_accuracy: 0.8717
12992/60000 [=====>........................] - ETA: 1:24 - loss: 0.4093 - categorical_accuracy: 0.8718
13024/60000 [=====>........................] - ETA: 1:24 - loss: 0.4085 - categorical_accuracy: 0.8719
13056/60000 [=====>........................] - ETA: 1:24 - loss: 0.4083 - categorical_accuracy: 0.8720
13088/60000 [=====>........................] - ETA: 1:24 - loss: 0.4077 - categorical_accuracy: 0.8722
13120/60000 [=====>........................] - ETA: 1:24 - loss: 0.4073 - categorical_accuracy: 0.8723
13152/60000 [=====>........................] - ETA: 1:24 - loss: 0.4066 - categorical_accuracy: 0.8726
13184/60000 [=====>........................] - ETA: 1:24 - loss: 0.4057 - categorical_accuracy: 0.8730
13216/60000 [=====>........................] - ETA: 1:24 - loss: 0.4052 - categorical_accuracy: 0.8730
13248/60000 [=====>........................] - ETA: 1:24 - loss: 0.4046 - categorical_accuracy: 0.8733
13280/60000 [=====>........................] - ETA: 1:24 - loss: 0.4041 - categorical_accuracy: 0.8733
13312/60000 [=====>........................] - ETA: 1:23 - loss: 0.4040 - categorical_accuracy: 0.8733
13344/60000 [=====>........................] - ETA: 1:23 - loss: 0.4032 - categorical_accuracy: 0.8735
13376/60000 [=====>........................] - ETA: 1:23 - loss: 0.4028 - categorical_accuracy: 0.8737
13408/60000 [=====>........................] - ETA: 1:23 - loss: 0.4022 - categorical_accuracy: 0.8738
13440/60000 [=====>........................] - ETA: 1:23 - loss: 0.4020 - categorical_accuracy: 0.8740
13472/60000 [=====>........................] - ETA: 1:23 - loss: 0.4011 - categorical_accuracy: 0.8743
13504/60000 [=====>........................] - ETA: 1:23 - loss: 0.4004 - categorical_accuracy: 0.8744
13536/60000 [=====>........................] - ETA: 1:23 - loss: 0.3997 - categorical_accuracy: 0.8746
13568/60000 [=====>........................] - ETA: 1:23 - loss: 0.4002 - categorical_accuracy: 0.8746
13600/60000 [=====>........................] - ETA: 1:23 - loss: 0.3994 - categorical_accuracy: 0.8749
13632/60000 [=====>........................] - ETA: 1:23 - loss: 0.3986 - categorical_accuracy: 0.8751
13664/60000 [=====>........................] - ETA: 1:23 - loss: 0.3980 - categorical_accuracy: 0.8754
13696/60000 [=====>........................] - ETA: 1:23 - loss: 0.3977 - categorical_accuracy: 0.8755
13728/60000 [=====>........................] - ETA: 1:23 - loss: 0.3974 - categorical_accuracy: 0.8757
13760/60000 [=====>........................] - ETA: 1:23 - loss: 0.3967 - categorical_accuracy: 0.8759
13792/60000 [=====>........................] - ETA: 1:23 - loss: 0.3958 - categorical_accuracy: 0.8762
13824/60000 [=====>........................] - ETA: 1:23 - loss: 0.3952 - categorical_accuracy: 0.8762
13856/60000 [=====>........................] - ETA: 1:23 - loss: 0.3944 - categorical_accuracy: 0.8764
13888/60000 [=====>........................] - ETA: 1:23 - loss: 0.3939 - categorical_accuracy: 0.8765
13920/60000 [=====>........................] - ETA: 1:22 - loss: 0.3931 - categorical_accuracy: 0.8768
13952/60000 [=====>........................] - ETA: 1:22 - loss: 0.3925 - categorical_accuracy: 0.8769
13984/60000 [=====>........................] - ETA: 1:22 - loss: 0.3918 - categorical_accuracy: 0.8771
14016/60000 [======>.......................] - ETA: 1:22 - loss: 0.3912 - categorical_accuracy: 0.8773
14048/60000 [======>.......................] - ETA: 1:22 - loss: 0.3912 - categorical_accuracy: 0.8773
14080/60000 [======>.......................] - ETA: 1:22 - loss: 0.3907 - categorical_accuracy: 0.8774
14112/60000 [======>.......................] - ETA: 1:22 - loss: 0.3900 - categorical_accuracy: 0.8776
14144/60000 [======>.......................] - ETA: 1:22 - loss: 0.3896 - categorical_accuracy: 0.8778
14176/60000 [======>.......................] - ETA: 1:22 - loss: 0.3889 - categorical_accuracy: 0.8780
14208/60000 [======>.......................] - ETA: 1:22 - loss: 0.3888 - categorical_accuracy: 0.8780
14240/60000 [======>.......................] - ETA: 1:22 - loss: 0.3885 - categorical_accuracy: 0.8781
14272/60000 [======>.......................] - ETA: 1:22 - loss: 0.3878 - categorical_accuracy: 0.8783
14304/60000 [======>.......................] - ETA: 1:22 - loss: 0.3871 - categorical_accuracy: 0.8786
14336/60000 [======>.......................] - ETA: 1:22 - loss: 0.3867 - categorical_accuracy: 0.8787
14368/60000 [======>.......................] - ETA: 1:22 - loss: 0.3861 - categorical_accuracy: 0.8789
14400/60000 [======>.......................] - ETA: 1:22 - loss: 0.3856 - categorical_accuracy: 0.8790
14432/60000 [======>.......................] - ETA: 1:22 - loss: 0.3851 - categorical_accuracy: 0.8792
14464/60000 [======>.......................] - ETA: 1:22 - loss: 0.3846 - categorical_accuracy: 0.8794
14496/60000 [======>.......................] - ETA: 1:22 - loss: 0.3842 - categorical_accuracy: 0.8796
14528/60000 [======>.......................] - ETA: 1:21 - loss: 0.3836 - categorical_accuracy: 0.8797
14560/60000 [======>.......................] - ETA: 1:21 - loss: 0.3830 - categorical_accuracy: 0.8799
14592/60000 [======>.......................] - ETA: 1:21 - loss: 0.3829 - categorical_accuracy: 0.8799
14624/60000 [======>.......................] - ETA: 1:21 - loss: 0.3832 - categorical_accuracy: 0.8800
14656/60000 [======>.......................] - ETA: 1:21 - loss: 0.3828 - categorical_accuracy: 0.8801
14688/60000 [======>.......................] - ETA: 1:21 - loss: 0.3820 - categorical_accuracy: 0.8804
14720/60000 [======>.......................] - ETA: 1:21 - loss: 0.3819 - categorical_accuracy: 0.8804
14752/60000 [======>.......................] - ETA: 1:21 - loss: 0.3814 - categorical_accuracy: 0.8806
14784/60000 [======>.......................] - ETA: 1:21 - loss: 0.3806 - categorical_accuracy: 0.8808
14816/60000 [======>.......................] - ETA: 1:21 - loss: 0.3807 - categorical_accuracy: 0.8808
14848/60000 [======>.......................] - ETA: 1:21 - loss: 0.3801 - categorical_accuracy: 0.8811
14880/60000 [======>.......................] - ETA: 1:21 - loss: 0.3794 - categorical_accuracy: 0.8813
14912/60000 [======>.......................] - ETA: 1:21 - loss: 0.3789 - categorical_accuracy: 0.8814
14944/60000 [======>.......................] - ETA: 1:21 - loss: 0.3784 - categorical_accuracy: 0.8815
14976/60000 [======>.......................] - ETA: 1:21 - loss: 0.3782 - categorical_accuracy: 0.8816
15008/60000 [======>.......................] - ETA: 1:21 - loss: 0.3776 - categorical_accuracy: 0.8818
15040/60000 [======>.......................] - ETA: 1:21 - loss: 0.3770 - categorical_accuracy: 0.8820
15072/60000 [======>.......................] - ETA: 1:21 - loss: 0.3769 - categorical_accuracy: 0.8822
15104/60000 [======>.......................] - ETA: 1:20 - loss: 0.3766 - categorical_accuracy: 0.8823
15136/60000 [======>.......................] - ETA: 1:20 - loss: 0.3762 - categorical_accuracy: 0.8825
15168/60000 [======>.......................] - ETA: 1:20 - loss: 0.3756 - categorical_accuracy: 0.8826
15200/60000 [======>.......................] - ETA: 1:20 - loss: 0.3750 - categorical_accuracy: 0.8828
15232/60000 [======>.......................] - ETA: 1:20 - loss: 0.3749 - categorical_accuracy: 0.8829
15264/60000 [======>.......................] - ETA: 1:20 - loss: 0.3743 - categorical_accuracy: 0.8831
15296/60000 [======>.......................] - ETA: 1:20 - loss: 0.3738 - categorical_accuracy: 0.8832
15328/60000 [======>.......................] - ETA: 1:20 - loss: 0.3734 - categorical_accuracy: 0.8834
15360/60000 [======>.......................] - ETA: 1:20 - loss: 0.3727 - categorical_accuracy: 0.8835
15392/60000 [======>.......................] - ETA: 1:20 - loss: 0.3724 - categorical_accuracy: 0.8836
15424/60000 [======>.......................] - ETA: 1:20 - loss: 0.3724 - categorical_accuracy: 0.8838
15456/60000 [======>.......................] - ETA: 1:20 - loss: 0.3718 - categorical_accuracy: 0.8839
15488/60000 [======>.......................] - ETA: 1:20 - loss: 0.3715 - categorical_accuracy: 0.8840
15520/60000 [======>.......................] - ETA: 1:20 - loss: 0.3708 - categorical_accuracy: 0.8843
15552/60000 [======>.......................] - ETA: 1:20 - loss: 0.3703 - categorical_accuracy: 0.8845
15584/60000 [======>.......................] - ETA: 1:20 - loss: 0.3700 - categorical_accuracy: 0.8846
15616/60000 [======>.......................] - ETA: 1:20 - loss: 0.3693 - categorical_accuracy: 0.8849
15648/60000 [======>.......................] - ETA: 1:20 - loss: 0.3689 - categorical_accuracy: 0.8850
15680/60000 [======>.......................] - ETA: 1:19 - loss: 0.3683 - categorical_accuracy: 0.8851
15712/60000 [======>.......................] - ETA: 1:19 - loss: 0.3677 - categorical_accuracy: 0.8853
15744/60000 [======>.......................] - ETA: 1:19 - loss: 0.3671 - categorical_accuracy: 0.8855
15776/60000 [======>.......................] - ETA: 1:19 - loss: 0.3665 - categorical_accuracy: 0.8856
15808/60000 [======>.......................] - ETA: 1:19 - loss: 0.3659 - categorical_accuracy: 0.8858
15840/60000 [======>.......................] - ETA: 1:19 - loss: 0.3661 - categorical_accuracy: 0.8859
15872/60000 [======>.......................] - ETA: 1:19 - loss: 0.3657 - categorical_accuracy: 0.8860
15904/60000 [======>.......................] - ETA: 1:19 - loss: 0.3655 - categorical_accuracy: 0.8861
15936/60000 [======>.......................] - ETA: 1:19 - loss: 0.3651 - categorical_accuracy: 0.8861
15968/60000 [======>.......................] - ETA: 1:19 - loss: 0.3645 - categorical_accuracy: 0.8863
16000/60000 [=======>......................] - ETA: 1:19 - loss: 0.3640 - categorical_accuracy: 0.8864
16032/60000 [=======>......................] - ETA: 1:19 - loss: 0.3638 - categorical_accuracy: 0.8865
16064/60000 [=======>......................] - ETA: 1:19 - loss: 0.3636 - categorical_accuracy: 0.8865
16096/60000 [=======>......................] - ETA: 1:19 - loss: 0.3632 - categorical_accuracy: 0.8866
16128/60000 [=======>......................] - ETA: 1:19 - loss: 0.3631 - categorical_accuracy: 0.8867
16160/60000 [=======>......................] - ETA: 1:19 - loss: 0.3628 - categorical_accuracy: 0.8867
16192/60000 [=======>......................] - ETA: 1:19 - loss: 0.3626 - categorical_accuracy: 0.8869
16224/60000 [=======>......................] - ETA: 1:18 - loss: 0.3626 - categorical_accuracy: 0.8870
16256/60000 [=======>......................] - ETA: 1:18 - loss: 0.3621 - categorical_accuracy: 0.8872
16288/60000 [=======>......................] - ETA: 1:18 - loss: 0.3616 - categorical_accuracy: 0.8873
16320/60000 [=======>......................] - ETA: 1:18 - loss: 0.3613 - categorical_accuracy: 0.8874
16352/60000 [=======>......................] - ETA: 1:18 - loss: 0.3607 - categorical_accuracy: 0.8876
16384/60000 [=======>......................] - ETA: 1:18 - loss: 0.3604 - categorical_accuracy: 0.8876
16416/60000 [=======>......................] - ETA: 1:18 - loss: 0.3602 - categorical_accuracy: 0.8877
16448/60000 [=======>......................] - ETA: 1:18 - loss: 0.3600 - categorical_accuracy: 0.8878
16480/60000 [=======>......................] - ETA: 1:18 - loss: 0.3595 - categorical_accuracy: 0.8879
16512/60000 [=======>......................] - ETA: 1:18 - loss: 0.3589 - categorical_accuracy: 0.8881
16544/60000 [=======>......................] - ETA: 1:18 - loss: 0.3589 - categorical_accuracy: 0.8882
16576/60000 [=======>......................] - ETA: 1:18 - loss: 0.3585 - categorical_accuracy: 0.8884
16608/60000 [=======>......................] - ETA: 1:18 - loss: 0.3587 - categorical_accuracy: 0.8884
16640/60000 [=======>......................] - ETA: 1:18 - loss: 0.3583 - categorical_accuracy: 0.8886
16672/60000 [=======>......................] - ETA: 1:18 - loss: 0.3581 - categorical_accuracy: 0.8887
16704/60000 [=======>......................] - ETA: 1:18 - loss: 0.3577 - categorical_accuracy: 0.8888
16736/60000 [=======>......................] - ETA: 1:18 - loss: 0.3571 - categorical_accuracy: 0.8890
16768/60000 [=======>......................] - ETA: 1:18 - loss: 0.3568 - categorical_accuracy: 0.8890
16800/60000 [=======>......................] - ETA: 1:18 - loss: 0.3562 - categorical_accuracy: 0.8892
16832/60000 [=======>......................] - ETA: 1:17 - loss: 0.3556 - categorical_accuracy: 0.8894
16864/60000 [=======>......................] - ETA: 1:17 - loss: 0.3551 - categorical_accuracy: 0.8895
16896/60000 [=======>......................] - ETA: 1:17 - loss: 0.3546 - categorical_accuracy: 0.8896
16928/60000 [=======>......................] - ETA: 1:17 - loss: 0.3541 - categorical_accuracy: 0.8898
16960/60000 [=======>......................] - ETA: 1:17 - loss: 0.3536 - categorical_accuracy: 0.8899
16992/60000 [=======>......................] - ETA: 1:17 - loss: 0.3532 - categorical_accuracy: 0.8901
17024/60000 [=======>......................] - ETA: 1:17 - loss: 0.3525 - categorical_accuracy: 0.8903
17056/60000 [=======>......................] - ETA: 1:17 - loss: 0.3525 - categorical_accuracy: 0.8904
17088/60000 [=======>......................] - ETA: 1:17 - loss: 0.3522 - categorical_accuracy: 0.8905
17120/60000 [=======>......................] - ETA: 1:17 - loss: 0.3521 - categorical_accuracy: 0.8905
17152/60000 [=======>......................] - ETA: 1:17 - loss: 0.3520 - categorical_accuracy: 0.8905
17184/60000 [=======>......................] - ETA: 1:17 - loss: 0.3515 - categorical_accuracy: 0.8906
17216/60000 [=======>......................] - ETA: 1:17 - loss: 0.3511 - categorical_accuracy: 0.8907
17248/60000 [=======>......................] - ETA: 1:17 - loss: 0.3507 - categorical_accuracy: 0.8908
17280/60000 [=======>......................] - ETA: 1:17 - loss: 0.3501 - categorical_accuracy: 0.8910
17312/60000 [=======>......................] - ETA: 1:17 - loss: 0.3502 - categorical_accuracy: 0.8909
17344/60000 [=======>......................] - ETA: 1:16 - loss: 0.3499 - categorical_accuracy: 0.8910
17376/60000 [=======>......................] - ETA: 1:16 - loss: 0.3495 - categorical_accuracy: 0.8911
17408/60000 [=======>......................] - ETA: 1:16 - loss: 0.3492 - categorical_accuracy: 0.8911
17440/60000 [=======>......................] - ETA: 1:16 - loss: 0.3487 - categorical_accuracy: 0.8913
17472/60000 [=======>......................] - ETA: 1:16 - loss: 0.3481 - categorical_accuracy: 0.8915
17504/60000 [=======>......................] - ETA: 1:16 - loss: 0.3476 - categorical_accuracy: 0.8916
17536/60000 [=======>......................] - ETA: 1:16 - loss: 0.3472 - categorical_accuracy: 0.8917
17568/60000 [=======>......................] - ETA: 1:16 - loss: 0.3470 - categorical_accuracy: 0.8918
17600/60000 [=======>......................] - ETA: 1:16 - loss: 0.3467 - categorical_accuracy: 0.8919
17632/60000 [=======>......................] - ETA: 1:16 - loss: 0.3464 - categorical_accuracy: 0.8918
17664/60000 [=======>......................] - ETA: 1:16 - loss: 0.3465 - categorical_accuracy: 0.8919
17696/60000 [=======>......................] - ETA: 1:16 - loss: 0.3463 - categorical_accuracy: 0.8920
17728/60000 [=======>......................] - ETA: 1:16 - loss: 0.3460 - categorical_accuracy: 0.8921
17760/60000 [=======>......................] - ETA: 1:16 - loss: 0.3455 - categorical_accuracy: 0.8923
17792/60000 [=======>......................] - ETA: 1:16 - loss: 0.3450 - categorical_accuracy: 0.8925
17824/60000 [=======>......................] - ETA: 1:16 - loss: 0.3447 - categorical_accuracy: 0.8926
17856/60000 [=======>......................] - ETA: 1:15 - loss: 0.3445 - categorical_accuracy: 0.8926
17888/60000 [=======>......................] - ETA: 1:15 - loss: 0.3447 - categorical_accuracy: 0.8925
17920/60000 [=======>......................] - ETA: 1:15 - loss: 0.3441 - categorical_accuracy: 0.8927
17952/60000 [=======>......................] - ETA: 1:15 - loss: 0.3439 - categorical_accuracy: 0.8928
17984/60000 [=======>......................] - ETA: 1:15 - loss: 0.3435 - categorical_accuracy: 0.8929
18016/60000 [========>.....................] - ETA: 1:15 - loss: 0.3431 - categorical_accuracy: 0.8930
18048/60000 [========>.....................] - ETA: 1:15 - loss: 0.3426 - categorical_accuracy: 0.8931
18080/60000 [========>.....................] - ETA: 1:15 - loss: 0.3423 - categorical_accuracy: 0.8932
18112/60000 [========>.....................] - ETA: 1:15 - loss: 0.3419 - categorical_accuracy: 0.8933
18144/60000 [========>.....................] - ETA: 1:15 - loss: 0.3414 - categorical_accuracy: 0.8935
18176/60000 [========>.....................] - ETA: 1:15 - loss: 0.3410 - categorical_accuracy: 0.8935
18208/60000 [========>.....................] - ETA: 1:15 - loss: 0.3408 - categorical_accuracy: 0.8936
18240/60000 [========>.....................] - ETA: 1:15 - loss: 0.3406 - categorical_accuracy: 0.8937
18272/60000 [========>.....................] - ETA: 1:15 - loss: 0.3407 - categorical_accuracy: 0.8938
18304/60000 [========>.....................] - ETA: 1:15 - loss: 0.3402 - categorical_accuracy: 0.8940
18336/60000 [========>.....................] - ETA: 1:15 - loss: 0.3396 - categorical_accuracy: 0.8941
18368/60000 [========>.....................] - ETA: 1:14 - loss: 0.3391 - categorical_accuracy: 0.8943
18400/60000 [========>.....................] - ETA: 1:14 - loss: 0.3388 - categorical_accuracy: 0.8943
18432/60000 [========>.....................] - ETA: 1:14 - loss: 0.3385 - categorical_accuracy: 0.8944
18464/60000 [========>.....................] - ETA: 1:14 - loss: 0.3381 - categorical_accuracy: 0.8945
18496/60000 [========>.....................] - ETA: 1:14 - loss: 0.3378 - categorical_accuracy: 0.8946
18528/60000 [========>.....................] - ETA: 1:14 - loss: 0.3375 - categorical_accuracy: 0.8946
18560/60000 [========>.....................] - ETA: 1:14 - loss: 0.3370 - categorical_accuracy: 0.8948
18592/60000 [========>.....................] - ETA: 1:14 - loss: 0.3367 - categorical_accuracy: 0.8949
18624/60000 [========>.....................] - ETA: 1:14 - loss: 0.3362 - categorical_accuracy: 0.8950
18656/60000 [========>.....................] - ETA: 1:14 - loss: 0.3358 - categorical_accuracy: 0.8952
18688/60000 [========>.....................] - ETA: 1:14 - loss: 0.3359 - categorical_accuracy: 0.8952
18720/60000 [========>.....................] - ETA: 1:14 - loss: 0.3358 - categorical_accuracy: 0.8952
18752/60000 [========>.....................] - ETA: 1:14 - loss: 0.3356 - categorical_accuracy: 0.8953
18784/60000 [========>.....................] - ETA: 1:14 - loss: 0.3354 - categorical_accuracy: 0.8953
18816/60000 [========>.....................] - ETA: 1:14 - loss: 0.3353 - categorical_accuracy: 0.8954
18848/60000 [========>.....................] - ETA: 1:14 - loss: 0.3348 - categorical_accuracy: 0.8955
18880/60000 [========>.....................] - ETA: 1:14 - loss: 0.3345 - categorical_accuracy: 0.8956
18912/60000 [========>.....................] - ETA: 1:13 - loss: 0.3342 - categorical_accuracy: 0.8957
18944/60000 [========>.....................] - ETA: 1:13 - loss: 0.3340 - categorical_accuracy: 0.8958
18976/60000 [========>.....................] - ETA: 1:13 - loss: 0.3337 - categorical_accuracy: 0.8959
19008/60000 [========>.....................] - ETA: 1:13 - loss: 0.3337 - categorical_accuracy: 0.8959
19040/60000 [========>.....................] - ETA: 1:13 - loss: 0.3337 - categorical_accuracy: 0.8958
19072/60000 [========>.....................] - ETA: 1:13 - loss: 0.3335 - categorical_accuracy: 0.8959
19104/60000 [========>.....................] - ETA: 1:13 - loss: 0.3332 - categorical_accuracy: 0.8960
19136/60000 [========>.....................] - ETA: 1:13 - loss: 0.3328 - categorical_accuracy: 0.8962
19168/60000 [========>.....................] - ETA: 1:13 - loss: 0.3323 - categorical_accuracy: 0.8963
19200/60000 [========>.....................] - ETA: 1:13 - loss: 0.3318 - categorical_accuracy: 0.8965
19232/60000 [========>.....................] - ETA: 1:13 - loss: 0.3314 - categorical_accuracy: 0.8966
19264/60000 [========>.....................] - ETA: 1:13 - loss: 0.3310 - categorical_accuracy: 0.8967
19296/60000 [========>.....................] - ETA: 1:13 - loss: 0.3306 - categorical_accuracy: 0.8969
19328/60000 [========>.....................] - ETA: 1:13 - loss: 0.3301 - categorical_accuracy: 0.8970
19360/60000 [========>.....................] - ETA: 1:13 - loss: 0.3297 - categorical_accuracy: 0.8972
19392/60000 [========>.....................] - ETA: 1:13 - loss: 0.3292 - categorical_accuracy: 0.8973
19424/60000 [========>.....................] - ETA: 1:13 - loss: 0.3287 - categorical_accuracy: 0.8975
19456/60000 [========>.....................] - ETA: 1:12 - loss: 0.3282 - categorical_accuracy: 0.8977
19488/60000 [========>.....................] - ETA: 1:12 - loss: 0.3277 - categorical_accuracy: 0.8978
19520/60000 [========>.....................] - ETA: 1:12 - loss: 0.3273 - categorical_accuracy: 0.8980
19552/60000 [========>.....................] - ETA: 1:12 - loss: 0.3268 - categorical_accuracy: 0.8981
19584/60000 [========>.....................] - ETA: 1:12 - loss: 0.3266 - categorical_accuracy: 0.8981
19616/60000 [========>.....................] - ETA: 1:12 - loss: 0.3262 - categorical_accuracy: 0.8982
19648/60000 [========>.....................] - ETA: 1:12 - loss: 0.3262 - categorical_accuracy: 0.8983
19680/60000 [========>.....................] - ETA: 1:12 - loss: 0.3261 - categorical_accuracy: 0.8984
19712/60000 [========>.....................] - ETA: 1:12 - loss: 0.3257 - categorical_accuracy: 0.8985
19744/60000 [========>.....................] - ETA: 1:12 - loss: 0.3252 - categorical_accuracy: 0.8987
19776/60000 [========>.....................] - ETA: 1:12 - loss: 0.3252 - categorical_accuracy: 0.8987
19808/60000 [========>.....................] - ETA: 1:12 - loss: 0.3247 - categorical_accuracy: 0.8989
19840/60000 [========>.....................] - ETA: 1:12 - loss: 0.3251 - categorical_accuracy: 0.8988
19872/60000 [========>.....................] - ETA: 1:12 - loss: 0.3246 - categorical_accuracy: 0.8990
19904/60000 [========>.....................] - ETA: 1:12 - loss: 0.3248 - categorical_accuracy: 0.8991
19936/60000 [========>.....................] - ETA: 1:12 - loss: 0.3246 - categorical_accuracy: 0.8992
19968/60000 [========>.....................] - ETA: 1:11 - loss: 0.3243 - categorical_accuracy: 0.8993
20000/60000 [=========>....................] - ETA: 1:11 - loss: 0.3239 - categorical_accuracy: 0.8995
20032/60000 [=========>....................] - ETA: 1:11 - loss: 0.3234 - categorical_accuracy: 0.8996
20064/60000 [=========>....................] - ETA: 1:11 - loss: 0.3231 - categorical_accuracy: 0.8997
20096/60000 [=========>....................] - ETA: 1:11 - loss: 0.3228 - categorical_accuracy: 0.8998
20128/60000 [=========>....................] - ETA: 1:11 - loss: 0.3223 - categorical_accuracy: 0.9000
20160/60000 [=========>....................] - ETA: 1:11 - loss: 0.3218 - categorical_accuracy: 0.9001
20192/60000 [=========>....................] - ETA: 1:11 - loss: 0.3214 - categorical_accuracy: 0.9003
20224/60000 [=========>....................] - ETA: 1:11 - loss: 0.3212 - categorical_accuracy: 0.9004
20256/60000 [=========>....................] - ETA: 1:11 - loss: 0.3211 - categorical_accuracy: 0.9004
20288/60000 [=========>....................] - ETA: 1:11 - loss: 0.3207 - categorical_accuracy: 0.9006
20320/60000 [=========>....................] - ETA: 1:11 - loss: 0.3203 - categorical_accuracy: 0.9007
20352/60000 [=========>....................] - ETA: 1:11 - loss: 0.3199 - categorical_accuracy: 0.9008
20384/60000 [=========>....................] - ETA: 1:11 - loss: 0.3196 - categorical_accuracy: 0.9009
20416/60000 [=========>....................] - ETA: 1:11 - loss: 0.3195 - categorical_accuracy: 0.9009
20448/60000 [=========>....................] - ETA: 1:11 - loss: 0.3190 - categorical_accuracy: 0.9011
20480/60000 [=========>....................] - ETA: 1:10 - loss: 0.3186 - categorical_accuracy: 0.9012
20512/60000 [=========>....................] - ETA: 1:10 - loss: 0.3184 - categorical_accuracy: 0.9012
20544/60000 [=========>....................] - ETA: 1:10 - loss: 0.3181 - categorical_accuracy: 0.9013
20576/60000 [=========>....................] - ETA: 1:10 - loss: 0.3179 - categorical_accuracy: 0.9014
20608/60000 [=========>....................] - ETA: 1:10 - loss: 0.3174 - categorical_accuracy: 0.9015
20640/60000 [=========>....................] - ETA: 1:10 - loss: 0.3170 - categorical_accuracy: 0.9017
20672/60000 [=========>....................] - ETA: 1:10 - loss: 0.3170 - categorical_accuracy: 0.9017
20704/60000 [=========>....................] - ETA: 1:10 - loss: 0.3166 - categorical_accuracy: 0.9018
20736/60000 [=========>....................] - ETA: 1:10 - loss: 0.3164 - categorical_accuracy: 0.9018
20768/60000 [=========>....................] - ETA: 1:10 - loss: 0.3160 - categorical_accuracy: 0.9019
20800/60000 [=========>....................] - ETA: 1:10 - loss: 0.3161 - categorical_accuracy: 0.9019
20832/60000 [=========>....................] - ETA: 1:10 - loss: 0.3157 - categorical_accuracy: 0.9020
20864/60000 [=========>....................] - ETA: 1:10 - loss: 0.3154 - categorical_accuracy: 0.9022
20896/60000 [=========>....................] - ETA: 1:10 - loss: 0.3151 - categorical_accuracy: 0.9023
20928/60000 [=========>....................] - ETA: 1:10 - loss: 0.3147 - categorical_accuracy: 0.9024
20960/60000 [=========>....................] - ETA: 1:10 - loss: 0.3143 - categorical_accuracy: 0.9025
20992/60000 [=========>....................] - ETA: 1:09 - loss: 0.3139 - categorical_accuracy: 0.9027
21024/60000 [=========>....................] - ETA: 1:09 - loss: 0.3137 - categorical_accuracy: 0.9027
21056/60000 [=========>....................] - ETA: 1:09 - loss: 0.3134 - categorical_accuracy: 0.9028
21088/60000 [=========>....................] - ETA: 1:09 - loss: 0.3133 - categorical_accuracy: 0.9028
21120/60000 [=========>....................] - ETA: 1:09 - loss: 0.3134 - categorical_accuracy: 0.9028
21152/60000 [=========>....................] - ETA: 1:09 - loss: 0.3135 - categorical_accuracy: 0.9028
21184/60000 [=========>....................] - ETA: 1:09 - loss: 0.3131 - categorical_accuracy: 0.9029
21216/60000 [=========>....................] - ETA: 1:09 - loss: 0.3129 - categorical_accuracy: 0.9030
21248/60000 [=========>....................] - ETA: 1:09 - loss: 0.3126 - categorical_accuracy: 0.9030
21280/60000 [=========>....................] - ETA: 1:09 - loss: 0.3122 - categorical_accuracy: 0.9032
21312/60000 [=========>....................] - ETA: 1:09 - loss: 0.3118 - categorical_accuracy: 0.9033
21344/60000 [=========>....................] - ETA: 1:09 - loss: 0.3116 - categorical_accuracy: 0.9034
21376/60000 [=========>....................] - ETA: 1:09 - loss: 0.3112 - categorical_accuracy: 0.9035
21408/60000 [=========>....................] - ETA: 1:09 - loss: 0.3110 - categorical_accuracy: 0.9036
21440/60000 [=========>....................] - ETA: 1:09 - loss: 0.3105 - categorical_accuracy: 0.9037
21472/60000 [=========>....................] - ETA: 1:09 - loss: 0.3104 - categorical_accuracy: 0.9037
21504/60000 [=========>....................] - ETA: 1:09 - loss: 0.3103 - categorical_accuracy: 0.9038
21536/60000 [=========>....................] - ETA: 1:09 - loss: 0.3106 - categorical_accuracy: 0.9036
21568/60000 [=========>....................] - ETA: 1:08 - loss: 0.3104 - categorical_accuracy: 0.9037
21600/60000 [=========>....................] - ETA: 1:08 - loss: 0.3100 - categorical_accuracy: 0.9038
21632/60000 [=========>....................] - ETA: 1:08 - loss: 0.3098 - categorical_accuracy: 0.9039
21664/60000 [=========>....................] - ETA: 1:08 - loss: 0.3095 - categorical_accuracy: 0.9039
21696/60000 [=========>....................] - ETA: 1:08 - loss: 0.3093 - categorical_accuracy: 0.9040
21728/60000 [=========>....................] - ETA: 1:08 - loss: 0.3089 - categorical_accuracy: 0.9041
21760/60000 [=========>....................] - ETA: 1:08 - loss: 0.3086 - categorical_accuracy: 0.9042
21792/60000 [=========>....................] - ETA: 1:08 - loss: 0.3084 - categorical_accuracy: 0.9043
21824/60000 [=========>....................] - ETA: 1:08 - loss: 0.3080 - categorical_accuracy: 0.9044
21856/60000 [=========>....................] - ETA: 1:08 - loss: 0.3076 - categorical_accuracy: 0.9045
21888/60000 [=========>....................] - ETA: 1:08 - loss: 0.3073 - categorical_accuracy: 0.9047
21920/60000 [=========>....................] - ETA: 1:08 - loss: 0.3070 - categorical_accuracy: 0.9047
21952/60000 [=========>....................] - ETA: 1:08 - loss: 0.3067 - categorical_accuracy: 0.9047
21984/60000 [=========>....................] - ETA: 1:08 - loss: 0.3063 - categorical_accuracy: 0.9049
22016/60000 [==========>...................] - ETA: 1:08 - loss: 0.3060 - categorical_accuracy: 0.9050
22048/60000 [==========>...................] - ETA: 1:08 - loss: 0.3058 - categorical_accuracy: 0.9050
22080/60000 [==========>...................] - ETA: 1:08 - loss: 0.3055 - categorical_accuracy: 0.9051
22112/60000 [==========>...................] - ETA: 1:07 - loss: 0.3052 - categorical_accuracy: 0.9052
22144/60000 [==========>...................] - ETA: 1:07 - loss: 0.3049 - categorical_accuracy: 0.9053
22176/60000 [==========>...................] - ETA: 1:07 - loss: 0.3046 - categorical_accuracy: 0.9054
22208/60000 [==========>...................] - ETA: 1:07 - loss: 0.3043 - categorical_accuracy: 0.9055
22240/60000 [==========>...................] - ETA: 1:07 - loss: 0.3042 - categorical_accuracy: 0.9056
22272/60000 [==========>...................] - ETA: 1:07 - loss: 0.3039 - categorical_accuracy: 0.9057
22304/60000 [==========>...................] - ETA: 1:07 - loss: 0.3036 - categorical_accuracy: 0.9058
22336/60000 [==========>...................] - ETA: 1:07 - loss: 0.3033 - categorical_accuracy: 0.9059
22368/60000 [==========>...................] - ETA: 1:07 - loss: 0.3032 - categorical_accuracy: 0.9059
22400/60000 [==========>...................] - ETA: 1:07 - loss: 0.3029 - categorical_accuracy: 0.9060
22432/60000 [==========>...................] - ETA: 1:07 - loss: 0.3027 - categorical_accuracy: 0.9061
22464/60000 [==========>...................] - ETA: 1:07 - loss: 0.3023 - categorical_accuracy: 0.9062
22496/60000 [==========>...................] - ETA: 1:07 - loss: 0.3019 - categorical_accuracy: 0.9064
22528/60000 [==========>...................] - ETA: 1:07 - loss: 0.3017 - categorical_accuracy: 0.9065
22560/60000 [==========>...................] - ETA: 1:07 - loss: 0.3014 - categorical_accuracy: 0.9066
22592/60000 [==========>...................] - ETA: 1:07 - loss: 0.3016 - categorical_accuracy: 0.9066
22624/60000 [==========>...................] - ETA: 1:07 - loss: 0.3015 - categorical_accuracy: 0.9066
22656/60000 [==========>...................] - ETA: 1:06 - loss: 0.3013 - categorical_accuracy: 0.9067
22688/60000 [==========>...................] - ETA: 1:06 - loss: 0.3013 - categorical_accuracy: 0.9066
22720/60000 [==========>...................] - ETA: 1:06 - loss: 0.3010 - categorical_accuracy: 0.9067
22752/60000 [==========>...................] - ETA: 1:06 - loss: 0.3005 - categorical_accuracy: 0.9069
22784/60000 [==========>...................] - ETA: 1:06 - loss: 0.3003 - categorical_accuracy: 0.9070
22816/60000 [==========>...................] - ETA: 1:06 - loss: 0.3001 - categorical_accuracy: 0.9070
22848/60000 [==========>...................] - ETA: 1:06 - loss: 0.2999 - categorical_accuracy: 0.9070
22880/60000 [==========>...................] - ETA: 1:06 - loss: 0.2995 - categorical_accuracy: 0.9072
22912/60000 [==========>...................] - ETA: 1:06 - loss: 0.2992 - categorical_accuracy: 0.9073
22944/60000 [==========>...................] - ETA: 1:06 - loss: 0.2988 - categorical_accuracy: 0.9073
22976/60000 [==========>...................] - ETA: 1:06 - loss: 0.2987 - categorical_accuracy: 0.9074
23008/60000 [==========>...................] - ETA: 1:06 - loss: 0.2983 - categorical_accuracy: 0.9076
23040/60000 [==========>...................] - ETA: 1:06 - loss: 0.2981 - categorical_accuracy: 0.9076
23072/60000 [==========>...................] - ETA: 1:06 - loss: 0.2978 - categorical_accuracy: 0.9077
23104/60000 [==========>...................] - ETA: 1:06 - loss: 0.2975 - categorical_accuracy: 0.9078
23136/60000 [==========>...................] - ETA: 1:06 - loss: 0.2972 - categorical_accuracy: 0.9078
23168/60000 [==========>...................] - ETA: 1:06 - loss: 0.2970 - categorical_accuracy: 0.9079
23200/60000 [==========>...................] - ETA: 1:05 - loss: 0.2967 - categorical_accuracy: 0.9080
23232/60000 [==========>...................] - ETA: 1:05 - loss: 0.2965 - categorical_accuracy: 0.9081
23264/60000 [==========>...................] - ETA: 1:05 - loss: 0.2961 - categorical_accuracy: 0.9082
23296/60000 [==========>...................] - ETA: 1:05 - loss: 0.2959 - categorical_accuracy: 0.9083
23328/60000 [==========>...................] - ETA: 1:05 - loss: 0.2955 - categorical_accuracy: 0.9084
23360/60000 [==========>...................] - ETA: 1:05 - loss: 0.2952 - categorical_accuracy: 0.9085
23392/60000 [==========>...................] - ETA: 1:05 - loss: 0.2955 - categorical_accuracy: 0.9085
23424/60000 [==========>...................] - ETA: 1:05 - loss: 0.2953 - categorical_accuracy: 0.9085
23456/60000 [==========>...................] - ETA: 1:05 - loss: 0.2953 - categorical_accuracy: 0.9085
23488/60000 [==========>...................] - ETA: 1:05 - loss: 0.2953 - categorical_accuracy: 0.9085
23520/60000 [==========>...................] - ETA: 1:05 - loss: 0.2950 - categorical_accuracy: 0.9086
23552/60000 [==========>...................] - ETA: 1:05 - loss: 0.2947 - categorical_accuracy: 0.9087
23584/60000 [==========>...................] - ETA: 1:05 - loss: 0.2948 - categorical_accuracy: 0.9087
23616/60000 [==========>...................] - ETA: 1:05 - loss: 0.2945 - categorical_accuracy: 0.9088
23648/60000 [==========>...................] - ETA: 1:05 - loss: 0.2943 - categorical_accuracy: 0.9088
23680/60000 [==========>...................] - ETA: 1:05 - loss: 0.2941 - categorical_accuracy: 0.9089
23712/60000 [==========>...................] - ETA: 1:05 - loss: 0.2938 - categorical_accuracy: 0.9090
23744/60000 [==========>...................] - ETA: 1:04 - loss: 0.2936 - categorical_accuracy: 0.9090
23776/60000 [==========>...................] - ETA: 1:04 - loss: 0.2932 - categorical_accuracy: 0.9091
23808/60000 [==========>...................] - ETA: 1:04 - loss: 0.2931 - categorical_accuracy: 0.9092
23840/60000 [==========>...................] - ETA: 1:04 - loss: 0.2931 - categorical_accuracy: 0.9092
23872/60000 [==========>...................] - ETA: 1:04 - loss: 0.2930 - categorical_accuracy: 0.9093
23904/60000 [==========>...................] - ETA: 1:04 - loss: 0.2928 - categorical_accuracy: 0.9093
23936/60000 [==========>...................] - ETA: 1:04 - loss: 0.2928 - categorical_accuracy: 0.9093
23968/60000 [==========>...................] - ETA: 1:04 - loss: 0.2927 - categorical_accuracy: 0.9094
24000/60000 [===========>..................] - ETA: 1:04 - loss: 0.2928 - categorical_accuracy: 0.9094
24032/60000 [===========>..................] - ETA: 1:04 - loss: 0.2925 - categorical_accuracy: 0.9095
24064/60000 [===========>..................] - ETA: 1:04 - loss: 0.2924 - categorical_accuracy: 0.9095
24096/60000 [===========>..................] - ETA: 1:04 - loss: 0.2923 - categorical_accuracy: 0.9094
24128/60000 [===========>..................] - ETA: 1:04 - loss: 0.2920 - categorical_accuracy: 0.9096
24160/60000 [===========>..................] - ETA: 1:04 - loss: 0.2917 - categorical_accuracy: 0.9096
24192/60000 [===========>..................] - ETA: 1:04 - loss: 0.2916 - categorical_accuracy: 0.9097
24224/60000 [===========>..................] - ETA: 1:04 - loss: 0.2914 - categorical_accuracy: 0.9098
24256/60000 [===========>..................] - ETA: 1:04 - loss: 0.2912 - categorical_accuracy: 0.9098
24288/60000 [===========>..................] - ETA: 1:03 - loss: 0.2909 - categorical_accuracy: 0.9099
24320/60000 [===========>..................] - ETA: 1:03 - loss: 0.2906 - categorical_accuracy: 0.9100
24352/60000 [===========>..................] - ETA: 1:03 - loss: 0.2904 - categorical_accuracy: 0.9100
24384/60000 [===========>..................] - ETA: 1:03 - loss: 0.2901 - categorical_accuracy: 0.9101
24416/60000 [===========>..................] - ETA: 1:03 - loss: 0.2898 - categorical_accuracy: 0.9102
24448/60000 [===========>..................] - ETA: 1:03 - loss: 0.2896 - categorical_accuracy: 0.9102
24480/60000 [===========>..................] - ETA: 1:03 - loss: 0.2898 - categorical_accuracy: 0.9102
24512/60000 [===========>..................] - ETA: 1:03 - loss: 0.2896 - categorical_accuracy: 0.9102
24544/60000 [===========>..................] - ETA: 1:03 - loss: 0.2894 - categorical_accuracy: 0.9103
24576/60000 [===========>..................] - ETA: 1:03 - loss: 0.2892 - categorical_accuracy: 0.9104
24608/60000 [===========>..................] - ETA: 1:03 - loss: 0.2889 - categorical_accuracy: 0.9105
24640/60000 [===========>..................] - ETA: 1:03 - loss: 0.2889 - categorical_accuracy: 0.9105
24672/60000 [===========>..................] - ETA: 1:03 - loss: 0.2888 - categorical_accuracy: 0.9105
24704/60000 [===========>..................] - ETA: 1:03 - loss: 0.2885 - categorical_accuracy: 0.9107
24736/60000 [===========>..................] - ETA: 1:03 - loss: 0.2882 - categorical_accuracy: 0.9108
24768/60000 [===========>..................] - ETA: 1:03 - loss: 0.2881 - categorical_accuracy: 0.9108
24800/60000 [===========>..................] - ETA: 1:03 - loss: 0.2880 - categorical_accuracy: 0.9108
24832/60000 [===========>..................] - ETA: 1:02 - loss: 0.2879 - categorical_accuracy: 0.9109
24864/60000 [===========>..................] - ETA: 1:02 - loss: 0.2877 - categorical_accuracy: 0.9110
24896/60000 [===========>..................] - ETA: 1:02 - loss: 0.2874 - categorical_accuracy: 0.9110
24928/60000 [===========>..................] - ETA: 1:02 - loss: 0.2871 - categorical_accuracy: 0.9111
24960/60000 [===========>..................] - ETA: 1:02 - loss: 0.2869 - categorical_accuracy: 0.9112
24992/60000 [===========>..................] - ETA: 1:02 - loss: 0.2865 - categorical_accuracy: 0.9113
25024/60000 [===========>..................] - ETA: 1:02 - loss: 0.2862 - categorical_accuracy: 0.9114
25056/60000 [===========>..................] - ETA: 1:02 - loss: 0.2859 - categorical_accuracy: 0.9115
25088/60000 [===========>..................] - ETA: 1:02 - loss: 0.2862 - categorical_accuracy: 0.9116
25120/60000 [===========>..................] - ETA: 1:02 - loss: 0.2861 - categorical_accuracy: 0.9116
25152/60000 [===========>..................] - ETA: 1:02 - loss: 0.2860 - categorical_accuracy: 0.9117
25184/60000 [===========>..................] - ETA: 1:02 - loss: 0.2857 - categorical_accuracy: 0.9117
25216/60000 [===========>..................] - ETA: 1:02 - loss: 0.2854 - categorical_accuracy: 0.9118
25248/60000 [===========>..................] - ETA: 1:02 - loss: 0.2852 - categorical_accuracy: 0.9119
25280/60000 [===========>..................] - ETA: 1:02 - loss: 0.2850 - categorical_accuracy: 0.9120
25312/60000 [===========>..................] - ETA: 1:02 - loss: 0.2847 - categorical_accuracy: 0.9121
25344/60000 [===========>..................] - ETA: 1:02 - loss: 0.2844 - categorical_accuracy: 0.9122
25376/60000 [===========>..................] - ETA: 1:01 - loss: 0.2841 - categorical_accuracy: 0.9123
25408/60000 [===========>..................] - ETA: 1:01 - loss: 0.2841 - categorical_accuracy: 0.9123
25440/60000 [===========>..................] - ETA: 1:01 - loss: 0.2838 - categorical_accuracy: 0.9124
25472/60000 [===========>..................] - ETA: 1:01 - loss: 0.2837 - categorical_accuracy: 0.9125
25504/60000 [===========>..................] - ETA: 1:01 - loss: 0.2834 - categorical_accuracy: 0.9126
25536/60000 [===========>..................] - ETA: 1:01 - loss: 0.2832 - categorical_accuracy: 0.9126
25568/60000 [===========>..................] - ETA: 1:01 - loss: 0.2829 - categorical_accuracy: 0.9127
25600/60000 [===========>..................] - ETA: 1:01 - loss: 0.2826 - categorical_accuracy: 0.9128
25632/60000 [===========>..................] - ETA: 1:01 - loss: 0.2825 - categorical_accuracy: 0.9128
25664/60000 [===========>..................] - ETA: 1:01 - loss: 0.2822 - categorical_accuracy: 0.9129
25696/60000 [===========>..................] - ETA: 1:01 - loss: 0.2819 - categorical_accuracy: 0.9129
25728/60000 [===========>..................] - ETA: 1:01 - loss: 0.2818 - categorical_accuracy: 0.9130
25760/60000 [===========>..................] - ETA: 1:01 - loss: 0.2815 - categorical_accuracy: 0.9131
25792/60000 [===========>..................] - ETA: 1:01 - loss: 0.2813 - categorical_accuracy: 0.9132
25824/60000 [===========>..................] - ETA: 1:01 - loss: 0.2810 - categorical_accuracy: 0.9133
25856/60000 [===========>..................] - ETA: 1:01 - loss: 0.2807 - categorical_accuracy: 0.9134
25888/60000 [===========>..................] - ETA: 1:00 - loss: 0.2804 - categorical_accuracy: 0.9135
25920/60000 [===========>..................] - ETA: 1:00 - loss: 0.2803 - categorical_accuracy: 0.9135
25952/60000 [===========>..................] - ETA: 1:00 - loss: 0.2801 - categorical_accuracy: 0.9136
25984/60000 [===========>..................] - ETA: 1:00 - loss: 0.2800 - categorical_accuracy: 0.9136
26016/60000 [============>.................] - ETA: 1:00 - loss: 0.2798 - categorical_accuracy: 0.9137
26048/60000 [============>.................] - ETA: 1:00 - loss: 0.2795 - categorical_accuracy: 0.9137
26080/60000 [============>.................] - ETA: 1:00 - loss: 0.2793 - categorical_accuracy: 0.9138
26112/60000 [============>.................] - ETA: 1:00 - loss: 0.2790 - categorical_accuracy: 0.9139
26144/60000 [============>.................] - ETA: 1:00 - loss: 0.2787 - categorical_accuracy: 0.9140
26176/60000 [============>.................] - ETA: 1:00 - loss: 0.2785 - categorical_accuracy: 0.9140
26208/60000 [============>.................] - ETA: 1:00 - loss: 0.2782 - categorical_accuracy: 0.9141
26240/60000 [============>.................] - ETA: 1:00 - loss: 0.2779 - categorical_accuracy: 0.9142
26272/60000 [============>.................] - ETA: 1:00 - loss: 0.2777 - categorical_accuracy: 0.9142
26304/60000 [============>.................] - ETA: 1:00 - loss: 0.2774 - categorical_accuracy: 0.9143
26336/60000 [============>.................] - ETA: 1:00 - loss: 0.2771 - categorical_accuracy: 0.9144
26368/60000 [============>.................] - ETA: 1:00 - loss: 0.2774 - categorical_accuracy: 0.9144
26400/60000 [============>.................] - ETA: 1:00 - loss: 0.2771 - categorical_accuracy: 0.9145
26432/60000 [============>.................] - ETA: 1:00 - loss: 0.2769 - categorical_accuracy: 0.9145
26464/60000 [============>.................] - ETA: 59s - loss: 0.2770 - categorical_accuracy: 0.9146 
26496/60000 [============>.................] - ETA: 59s - loss: 0.2767 - categorical_accuracy: 0.9146
26528/60000 [============>.................] - ETA: 59s - loss: 0.2764 - categorical_accuracy: 0.9147
26560/60000 [============>.................] - ETA: 59s - loss: 0.2761 - categorical_accuracy: 0.9148
26592/60000 [============>.................] - ETA: 59s - loss: 0.2761 - categorical_accuracy: 0.9148
26624/60000 [============>.................] - ETA: 59s - loss: 0.2765 - categorical_accuracy: 0.9148
26656/60000 [============>.................] - ETA: 59s - loss: 0.2763 - categorical_accuracy: 0.9149
26688/60000 [============>.................] - ETA: 59s - loss: 0.2759 - categorical_accuracy: 0.9150
26720/60000 [============>.................] - ETA: 59s - loss: 0.2758 - categorical_accuracy: 0.9150
26752/60000 [============>.................] - ETA: 59s - loss: 0.2755 - categorical_accuracy: 0.9151
26784/60000 [============>.................] - ETA: 59s - loss: 0.2752 - categorical_accuracy: 0.9152
26816/60000 [============>.................] - ETA: 59s - loss: 0.2752 - categorical_accuracy: 0.9152
26848/60000 [============>.................] - ETA: 59s - loss: 0.2749 - categorical_accuracy: 0.9153
26880/60000 [============>.................] - ETA: 59s - loss: 0.2746 - categorical_accuracy: 0.9154
26912/60000 [============>.................] - ETA: 59s - loss: 0.2743 - categorical_accuracy: 0.9155
26944/60000 [============>.................] - ETA: 59s - loss: 0.2743 - categorical_accuracy: 0.9155
26976/60000 [============>.................] - ETA: 59s - loss: 0.2741 - categorical_accuracy: 0.9155
27008/60000 [============>.................] - ETA: 59s - loss: 0.2738 - categorical_accuracy: 0.9156
27040/60000 [============>.................] - ETA: 59s - loss: 0.2738 - categorical_accuracy: 0.9156
27072/60000 [============>.................] - ETA: 58s - loss: 0.2735 - categorical_accuracy: 0.9157
27104/60000 [============>.................] - ETA: 58s - loss: 0.2734 - categorical_accuracy: 0.9157
27136/60000 [============>.................] - ETA: 58s - loss: 0.2732 - categorical_accuracy: 0.9158
27168/60000 [============>.................] - ETA: 58s - loss: 0.2732 - categorical_accuracy: 0.9158
27200/60000 [============>.................] - ETA: 58s - loss: 0.2730 - categorical_accuracy: 0.9158
27232/60000 [============>.................] - ETA: 58s - loss: 0.2731 - categorical_accuracy: 0.9158
27264/60000 [============>.................] - ETA: 58s - loss: 0.2729 - categorical_accuracy: 0.9158
27296/60000 [============>.................] - ETA: 58s - loss: 0.2728 - categorical_accuracy: 0.9158
27328/60000 [============>.................] - ETA: 58s - loss: 0.2729 - categorical_accuracy: 0.9158
27360/60000 [============>.................] - ETA: 58s - loss: 0.2726 - categorical_accuracy: 0.9159
27392/60000 [============>.................] - ETA: 58s - loss: 0.2724 - categorical_accuracy: 0.9160
27424/60000 [============>.................] - ETA: 58s - loss: 0.2725 - categorical_accuracy: 0.9160
27456/60000 [============>.................] - ETA: 58s - loss: 0.2722 - categorical_accuracy: 0.9161
27488/60000 [============>.................] - ETA: 58s - loss: 0.2719 - categorical_accuracy: 0.9162
27520/60000 [============>.................] - ETA: 58s - loss: 0.2716 - categorical_accuracy: 0.9163
27552/60000 [============>.................] - ETA: 58s - loss: 0.2713 - categorical_accuracy: 0.9163
27584/60000 [============>.................] - ETA: 58s - loss: 0.2712 - categorical_accuracy: 0.9164
27616/60000 [============>.................] - ETA: 58s - loss: 0.2711 - categorical_accuracy: 0.9164
27648/60000 [============>.................] - ETA: 57s - loss: 0.2710 - categorical_accuracy: 0.9164
27680/60000 [============>.................] - ETA: 57s - loss: 0.2709 - categorical_accuracy: 0.9165
27712/60000 [============>.................] - ETA: 57s - loss: 0.2710 - categorical_accuracy: 0.9165
27744/60000 [============>.................] - ETA: 57s - loss: 0.2711 - categorical_accuracy: 0.9165
27776/60000 [============>.................] - ETA: 57s - loss: 0.2708 - categorical_accuracy: 0.9166
27808/60000 [============>.................] - ETA: 57s - loss: 0.2706 - categorical_accuracy: 0.9166
27840/60000 [============>.................] - ETA: 57s - loss: 0.2704 - categorical_accuracy: 0.9167
27872/60000 [============>.................] - ETA: 57s - loss: 0.2703 - categorical_accuracy: 0.9167
27904/60000 [============>.................] - ETA: 57s - loss: 0.2701 - categorical_accuracy: 0.9168
27936/60000 [============>.................] - ETA: 57s - loss: 0.2699 - categorical_accuracy: 0.9168
27968/60000 [============>.................] - ETA: 57s - loss: 0.2697 - categorical_accuracy: 0.9168
28000/60000 [=============>................] - ETA: 57s - loss: 0.2695 - categorical_accuracy: 0.9169
28032/60000 [=============>................] - ETA: 57s - loss: 0.2693 - categorical_accuracy: 0.9170
28064/60000 [=============>................] - ETA: 57s - loss: 0.2691 - categorical_accuracy: 0.9170
28096/60000 [=============>................] - ETA: 57s - loss: 0.2691 - categorical_accuracy: 0.9170
28128/60000 [=============>................] - ETA: 57s - loss: 0.2688 - categorical_accuracy: 0.9171
28160/60000 [=============>................] - ETA: 57s - loss: 0.2685 - categorical_accuracy: 0.9172
28192/60000 [=============>................] - ETA: 57s - loss: 0.2682 - categorical_accuracy: 0.9173
28224/60000 [=============>................] - ETA: 56s - loss: 0.2680 - categorical_accuracy: 0.9173
28256/60000 [=============>................] - ETA: 56s - loss: 0.2679 - categorical_accuracy: 0.9174
28288/60000 [=============>................] - ETA: 56s - loss: 0.2677 - categorical_accuracy: 0.9175
28320/60000 [=============>................] - ETA: 56s - loss: 0.2674 - categorical_accuracy: 0.9175
28352/60000 [=============>................] - ETA: 56s - loss: 0.2672 - categorical_accuracy: 0.9176
28384/60000 [=============>................] - ETA: 56s - loss: 0.2670 - categorical_accuracy: 0.9176
28416/60000 [=============>................] - ETA: 56s - loss: 0.2669 - categorical_accuracy: 0.9177
28448/60000 [=============>................] - ETA: 56s - loss: 0.2667 - categorical_accuracy: 0.9177
28480/60000 [=============>................] - ETA: 56s - loss: 0.2664 - categorical_accuracy: 0.9178
28512/60000 [=============>................] - ETA: 56s - loss: 0.2662 - categorical_accuracy: 0.9179
28544/60000 [=============>................] - ETA: 56s - loss: 0.2660 - categorical_accuracy: 0.9180
28576/60000 [=============>................] - ETA: 56s - loss: 0.2658 - categorical_accuracy: 0.9180
28608/60000 [=============>................] - ETA: 56s - loss: 0.2656 - categorical_accuracy: 0.9180
28640/60000 [=============>................] - ETA: 56s - loss: 0.2656 - categorical_accuracy: 0.9180
28672/60000 [=============>................] - ETA: 56s - loss: 0.2655 - categorical_accuracy: 0.9180
28704/60000 [=============>................] - ETA: 56s - loss: 0.2654 - categorical_accuracy: 0.9181
28736/60000 [=============>................] - ETA: 56s - loss: 0.2653 - categorical_accuracy: 0.9181
28768/60000 [=============>................] - ETA: 55s - loss: 0.2650 - categorical_accuracy: 0.9182
28800/60000 [=============>................] - ETA: 55s - loss: 0.2648 - categorical_accuracy: 0.9183
28832/60000 [=============>................] - ETA: 55s - loss: 0.2647 - categorical_accuracy: 0.9183
28864/60000 [=============>................] - ETA: 55s - loss: 0.2645 - categorical_accuracy: 0.9184
28896/60000 [=============>................] - ETA: 55s - loss: 0.2643 - categorical_accuracy: 0.9184
28928/60000 [=============>................] - ETA: 55s - loss: 0.2642 - categorical_accuracy: 0.9184
28960/60000 [=============>................] - ETA: 55s - loss: 0.2640 - categorical_accuracy: 0.9184
28992/60000 [=============>................] - ETA: 55s - loss: 0.2638 - categorical_accuracy: 0.9185
29024/60000 [=============>................] - ETA: 55s - loss: 0.2637 - categorical_accuracy: 0.9186
29056/60000 [=============>................] - ETA: 55s - loss: 0.2636 - categorical_accuracy: 0.9186
29088/60000 [=============>................] - ETA: 55s - loss: 0.2633 - categorical_accuracy: 0.9187
29120/60000 [=============>................] - ETA: 55s - loss: 0.2632 - categorical_accuracy: 0.9187
29152/60000 [=============>................] - ETA: 55s - loss: 0.2631 - categorical_accuracy: 0.9187
29184/60000 [=============>................] - ETA: 55s - loss: 0.2629 - categorical_accuracy: 0.9188
29216/60000 [=============>................] - ETA: 55s - loss: 0.2627 - categorical_accuracy: 0.9188
29248/60000 [=============>................] - ETA: 55s - loss: 0.2625 - categorical_accuracy: 0.9189
29280/60000 [=============>................] - ETA: 55s - loss: 0.2623 - categorical_accuracy: 0.9190
29312/60000 [=============>................] - ETA: 55s - loss: 0.2620 - categorical_accuracy: 0.9190
29344/60000 [=============>................] - ETA: 54s - loss: 0.2618 - categorical_accuracy: 0.9191
29376/60000 [=============>................] - ETA: 54s - loss: 0.2617 - categorical_accuracy: 0.9191
29408/60000 [=============>................] - ETA: 54s - loss: 0.2617 - categorical_accuracy: 0.9191
29440/60000 [=============>................] - ETA: 54s - loss: 0.2615 - categorical_accuracy: 0.9191
29472/60000 [=============>................] - ETA: 54s - loss: 0.2613 - categorical_accuracy: 0.9192
29504/60000 [=============>................] - ETA: 54s - loss: 0.2610 - categorical_accuracy: 0.9193
29536/60000 [=============>................] - ETA: 54s - loss: 0.2608 - categorical_accuracy: 0.9194
29568/60000 [=============>................] - ETA: 54s - loss: 0.2607 - categorical_accuracy: 0.9193
29600/60000 [=============>................] - ETA: 54s - loss: 0.2605 - categorical_accuracy: 0.9194
29632/60000 [=============>................] - ETA: 54s - loss: 0.2603 - categorical_accuracy: 0.9195
29664/60000 [=============>................] - ETA: 54s - loss: 0.2601 - categorical_accuracy: 0.9195
29696/60000 [=============>................] - ETA: 54s - loss: 0.2599 - categorical_accuracy: 0.9196
29728/60000 [=============>................] - ETA: 54s - loss: 0.2602 - categorical_accuracy: 0.9196
29760/60000 [=============>................] - ETA: 54s - loss: 0.2600 - categorical_accuracy: 0.9196
29792/60000 [=============>................] - ETA: 54s - loss: 0.2598 - categorical_accuracy: 0.9197
29824/60000 [=============>................] - ETA: 54s - loss: 0.2596 - categorical_accuracy: 0.9198
29856/60000 [=============>................] - ETA: 54s - loss: 0.2597 - categorical_accuracy: 0.9197
29888/60000 [=============>................] - ETA: 54s - loss: 0.2596 - categorical_accuracy: 0.9197
29920/60000 [=============>................] - ETA: 53s - loss: 0.2600 - categorical_accuracy: 0.9197
29952/60000 [=============>................] - ETA: 53s - loss: 0.2597 - categorical_accuracy: 0.9198
30016/60000 [==============>...............] - ETA: 53s - loss: 0.2596 - categorical_accuracy: 0.9199
30048/60000 [==============>...............] - ETA: 53s - loss: 0.2594 - categorical_accuracy: 0.9199
30080/60000 [==============>...............] - ETA: 53s - loss: 0.2591 - categorical_accuracy: 0.9200
30112/60000 [==============>...............] - ETA: 53s - loss: 0.2589 - categorical_accuracy: 0.9201
30144/60000 [==============>...............] - ETA: 53s - loss: 0.2586 - categorical_accuracy: 0.9202
30176/60000 [==============>...............] - ETA: 53s - loss: 0.2585 - categorical_accuracy: 0.9202
30208/60000 [==============>...............] - ETA: 53s - loss: 0.2584 - categorical_accuracy: 0.9202
30240/60000 [==============>...............] - ETA: 53s - loss: 0.2582 - categorical_accuracy: 0.9203
30272/60000 [==============>...............] - ETA: 53s - loss: 0.2581 - categorical_accuracy: 0.9203
30304/60000 [==============>...............] - ETA: 53s - loss: 0.2578 - categorical_accuracy: 0.9204
30336/60000 [==============>...............] - ETA: 53s - loss: 0.2576 - categorical_accuracy: 0.9205
30368/60000 [==============>...............] - ETA: 53s - loss: 0.2574 - categorical_accuracy: 0.9206
30400/60000 [==============>...............] - ETA: 53s - loss: 0.2572 - categorical_accuracy: 0.9206
30432/60000 [==============>...............] - ETA: 53s - loss: 0.2569 - categorical_accuracy: 0.9207
30464/60000 [==============>...............] - ETA: 52s - loss: 0.2567 - categorical_accuracy: 0.9208
30496/60000 [==============>...............] - ETA: 52s - loss: 0.2566 - categorical_accuracy: 0.9208
30528/60000 [==============>...............] - ETA: 52s - loss: 0.2565 - categorical_accuracy: 0.9208
30592/60000 [==============>...............] - ETA: 52s - loss: 0.2561 - categorical_accuracy: 0.9210
30624/60000 [==============>...............] - ETA: 52s - loss: 0.2559 - categorical_accuracy: 0.9210
30656/60000 [==============>...............] - ETA: 52s - loss: 0.2557 - categorical_accuracy: 0.9211
30688/60000 [==============>...............] - ETA: 52s - loss: 0.2555 - categorical_accuracy: 0.9211
30720/60000 [==============>...............] - ETA: 52s - loss: 0.2555 - categorical_accuracy: 0.9211
30752/60000 [==============>...............] - ETA: 52s - loss: 0.2553 - categorical_accuracy: 0.9212
30784/60000 [==============>...............] - ETA: 52s - loss: 0.2552 - categorical_accuracy: 0.9212
30816/60000 [==============>...............] - ETA: 52s - loss: 0.2553 - categorical_accuracy: 0.9212
30848/60000 [==============>...............] - ETA: 52s - loss: 0.2551 - categorical_accuracy: 0.9213
30880/60000 [==============>...............] - ETA: 52s - loss: 0.2550 - categorical_accuracy: 0.9212
30912/60000 [==============>...............] - ETA: 52s - loss: 0.2548 - categorical_accuracy: 0.9213
30944/60000 [==============>...............] - ETA: 52s - loss: 0.2548 - categorical_accuracy: 0.9213
30976/60000 [==============>...............] - ETA: 52s - loss: 0.2546 - categorical_accuracy: 0.9214
31008/60000 [==============>...............] - ETA: 51s - loss: 0.2543 - categorical_accuracy: 0.9215
31040/60000 [==============>...............] - ETA: 51s - loss: 0.2541 - categorical_accuracy: 0.9216
31072/60000 [==============>...............] - ETA: 51s - loss: 0.2540 - categorical_accuracy: 0.9216
31104/60000 [==============>...............] - ETA: 51s - loss: 0.2538 - categorical_accuracy: 0.9216
31136/60000 [==============>...............] - ETA: 51s - loss: 0.2536 - categorical_accuracy: 0.9217
31168/60000 [==============>...............] - ETA: 51s - loss: 0.2534 - categorical_accuracy: 0.9217
31200/60000 [==============>...............] - ETA: 51s - loss: 0.2532 - categorical_accuracy: 0.9217
31232/60000 [==============>...............] - ETA: 51s - loss: 0.2532 - categorical_accuracy: 0.9218
31264/60000 [==============>...............] - ETA: 51s - loss: 0.2531 - categorical_accuracy: 0.9218
31296/60000 [==============>...............] - ETA: 51s - loss: 0.2529 - categorical_accuracy: 0.9219
31328/60000 [==============>...............] - ETA: 51s - loss: 0.2528 - categorical_accuracy: 0.9219
31360/60000 [==============>...............] - ETA: 51s - loss: 0.2528 - categorical_accuracy: 0.9218
31392/60000 [==============>...............] - ETA: 51s - loss: 0.2528 - categorical_accuracy: 0.9219
31424/60000 [==============>...............] - ETA: 51s - loss: 0.2526 - categorical_accuracy: 0.9219
31456/60000 [==============>...............] - ETA: 51s - loss: 0.2524 - categorical_accuracy: 0.9220
31488/60000 [==============>...............] - ETA: 51s - loss: 0.2523 - categorical_accuracy: 0.9220
31520/60000 [==============>...............] - ETA: 51s - loss: 0.2521 - categorical_accuracy: 0.9221
31552/60000 [==============>...............] - ETA: 50s - loss: 0.2521 - categorical_accuracy: 0.9221
31584/60000 [==============>...............] - ETA: 50s - loss: 0.2521 - categorical_accuracy: 0.9221
31616/60000 [==============>...............] - ETA: 50s - loss: 0.2521 - categorical_accuracy: 0.9221
31648/60000 [==============>...............] - ETA: 50s - loss: 0.2520 - categorical_accuracy: 0.9221
31680/60000 [==============>...............] - ETA: 50s - loss: 0.2519 - categorical_accuracy: 0.9221
31712/60000 [==============>...............] - ETA: 50s - loss: 0.2517 - categorical_accuracy: 0.9222
31744/60000 [==============>...............] - ETA: 50s - loss: 0.2514 - categorical_accuracy: 0.9223
31776/60000 [==============>...............] - ETA: 50s - loss: 0.2513 - categorical_accuracy: 0.9223
31808/60000 [==============>...............] - ETA: 50s - loss: 0.2513 - categorical_accuracy: 0.9223
31840/60000 [==============>...............] - ETA: 50s - loss: 0.2514 - categorical_accuracy: 0.9223
31872/60000 [==============>...............] - ETA: 50s - loss: 0.2512 - categorical_accuracy: 0.9224
31904/60000 [==============>...............] - ETA: 50s - loss: 0.2510 - categorical_accuracy: 0.9224
31936/60000 [==============>...............] - ETA: 50s - loss: 0.2510 - categorical_accuracy: 0.9224
31968/60000 [==============>...............] - ETA: 50s - loss: 0.2509 - categorical_accuracy: 0.9225
32000/60000 [===============>..............] - ETA: 50s - loss: 0.2507 - categorical_accuracy: 0.9225
32032/60000 [===============>..............] - ETA: 50s - loss: 0.2505 - categorical_accuracy: 0.9225
32064/60000 [===============>..............] - ETA: 50s - loss: 0.2506 - categorical_accuracy: 0.9225
32096/60000 [===============>..............] - ETA: 50s - loss: 0.2507 - categorical_accuracy: 0.9225
32128/60000 [===============>..............] - ETA: 49s - loss: 0.2505 - categorical_accuracy: 0.9225
32160/60000 [===============>..............] - ETA: 49s - loss: 0.2502 - categorical_accuracy: 0.9226
32192/60000 [===============>..............] - ETA: 49s - loss: 0.2501 - categorical_accuracy: 0.9226
32224/60000 [===============>..............] - ETA: 49s - loss: 0.2500 - categorical_accuracy: 0.9227
32256/60000 [===============>..............] - ETA: 49s - loss: 0.2498 - categorical_accuracy: 0.9227
32288/60000 [===============>..............] - ETA: 49s - loss: 0.2497 - categorical_accuracy: 0.9228
32320/60000 [===============>..............] - ETA: 49s - loss: 0.2495 - categorical_accuracy: 0.9228
32352/60000 [===============>..............] - ETA: 49s - loss: 0.2494 - categorical_accuracy: 0.9228
32384/60000 [===============>..............] - ETA: 49s - loss: 0.2496 - categorical_accuracy: 0.9228
32416/60000 [===============>..............] - ETA: 49s - loss: 0.2494 - categorical_accuracy: 0.9228
32448/60000 [===============>..............] - ETA: 49s - loss: 0.2496 - categorical_accuracy: 0.9228
32480/60000 [===============>..............] - ETA: 49s - loss: 0.2495 - categorical_accuracy: 0.9228
32512/60000 [===============>..............] - ETA: 49s - loss: 0.2493 - categorical_accuracy: 0.9229
32544/60000 [===============>..............] - ETA: 49s - loss: 0.2492 - categorical_accuracy: 0.9229
32576/60000 [===============>..............] - ETA: 49s - loss: 0.2492 - categorical_accuracy: 0.9230
32608/60000 [===============>..............] - ETA: 49s - loss: 0.2491 - categorical_accuracy: 0.9231
32640/60000 [===============>..............] - ETA: 49s - loss: 0.2489 - categorical_accuracy: 0.9231
32672/60000 [===============>..............] - ETA: 48s - loss: 0.2486 - categorical_accuracy: 0.9232
32704/60000 [===============>..............] - ETA: 48s - loss: 0.2485 - categorical_accuracy: 0.9233
32736/60000 [===============>..............] - ETA: 48s - loss: 0.2483 - categorical_accuracy: 0.9233
32768/60000 [===============>..............] - ETA: 48s - loss: 0.2483 - categorical_accuracy: 0.9233
32800/60000 [===============>..............] - ETA: 48s - loss: 0.2481 - categorical_accuracy: 0.9234
32832/60000 [===============>..............] - ETA: 48s - loss: 0.2480 - categorical_accuracy: 0.9234
32864/60000 [===============>..............] - ETA: 48s - loss: 0.2478 - categorical_accuracy: 0.9235
32896/60000 [===============>..............] - ETA: 48s - loss: 0.2476 - categorical_accuracy: 0.9235
32928/60000 [===============>..............] - ETA: 48s - loss: 0.2474 - categorical_accuracy: 0.9236
32960/60000 [===============>..............] - ETA: 48s - loss: 0.2473 - categorical_accuracy: 0.9236
32992/60000 [===============>..............] - ETA: 48s - loss: 0.2471 - categorical_accuracy: 0.9237
33024/60000 [===============>..............] - ETA: 48s - loss: 0.2468 - categorical_accuracy: 0.9238
33056/60000 [===============>..............] - ETA: 48s - loss: 0.2469 - categorical_accuracy: 0.9238
33088/60000 [===============>..............] - ETA: 48s - loss: 0.2467 - categorical_accuracy: 0.9239
33120/60000 [===============>..............] - ETA: 48s - loss: 0.2469 - categorical_accuracy: 0.9238
33152/60000 [===============>..............] - ETA: 48s - loss: 0.2467 - categorical_accuracy: 0.9239
33184/60000 [===============>..............] - ETA: 48s - loss: 0.2466 - categorical_accuracy: 0.9239
33216/60000 [===============>..............] - ETA: 48s - loss: 0.2464 - categorical_accuracy: 0.9240
33248/60000 [===============>..............] - ETA: 47s - loss: 0.2462 - categorical_accuracy: 0.9240
33280/60000 [===============>..............] - ETA: 47s - loss: 0.2462 - categorical_accuracy: 0.9240
33312/60000 [===============>..............] - ETA: 47s - loss: 0.2460 - categorical_accuracy: 0.9241
33344/60000 [===============>..............] - ETA: 47s - loss: 0.2459 - categorical_accuracy: 0.9241
33376/60000 [===============>..............] - ETA: 47s - loss: 0.2457 - categorical_accuracy: 0.9242
33408/60000 [===============>..............] - ETA: 47s - loss: 0.2455 - categorical_accuracy: 0.9242
33440/60000 [===============>..............] - ETA: 47s - loss: 0.2457 - categorical_accuracy: 0.9242
33472/60000 [===============>..............] - ETA: 47s - loss: 0.2456 - categorical_accuracy: 0.9242
33504/60000 [===============>..............] - ETA: 47s - loss: 0.2454 - categorical_accuracy: 0.9242
33536/60000 [===============>..............] - ETA: 47s - loss: 0.2456 - categorical_accuracy: 0.9243
33568/60000 [===============>..............] - ETA: 47s - loss: 0.2454 - categorical_accuracy: 0.9243
33600/60000 [===============>..............] - ETA: 47s - loss: 0.2453 - categorical_accuracy: 0.9244
33632/60000 [===============>..............] - ETA: 47s - loss: 0.2451 - categorical_accuracy: 0.9244
33664/60000 [===============>..............] - ETA: 47s - loss: 0.2451 - categorical_accuracy: 0.9245
33696/60000 [===============>..............] - ETA: 47s - loss: 0.2450 - categorical_accuracy: 0.9245
33728/60000 [===============>..............] - ETA: 47s - loss: 0.2449 - categorical_accuracy: 0.9245
33760/60000 [===============>..............] - ETA: 47s - loss: 0.2447 - categorical_accuracy: 0.9246
33792/60000 [===============>..............] - ETA: 46s - loss: 0.2448 - categorical_accuracy: 0.9245
33824/60000 [===============>..............] - ETA: 46s - loss: 0.2447 - categorical_accuracy: 0.9246
33856/60000 [===============>..............] - ETA: 46s - loss: 0.2447 - categorical_accuracy: 0.9246
33888/60000 [===============>..............] - ETA: 46s - loss: 0.2447 - categorical_accuracy: 0.9246
33920/60000 [===============>..............] - ETA: 46s - loss: 0.2447 - categorical_accuracy: 0.9246
33952/60000 [===============>..............] - ETA: 46s - loss: 0.2446 - categorical_accuracy: 0.9246
33984/60000 [===============>..............] - ETA: 46s - loss: 0.2445 - categorical_accuracy: 0.9246
34016/60000 [================>.............] - ETA: 46s - loss: 0.2444 - categorical_accuracy: 0.9247
34048/60000 [================>.............] - ETA: 46s - loss: 0.2443 - categorical_accuracy: 0.9247
34080/60000 [================>.............] - ETA: 46s - loss: 0.2440 - categorical_accuracy: 0.9248
34112/60000 [================>.............] - ETA: 46s - loss: 0.2439 - categorical_accuracy: 0.9248
34144/60000 [================>.............] - ETA: 46s - loss: 0.2438 - categorical_accuracy: 0.9248
34176/60000 [================>.............] - ETA: 46s - loss: 0.2436 - categorical_accuracy: 0.9249
34208/60000 [================>.............] - ETA: 46s - loss: 0.2435 - categorical_accuracy: 0.9249
34240/60000 [================>.............] - ETA: 46s - loss: 0.2433 - categorical_accuracy: 0.9250
34272/60000 [================>.............] - ETA: 46s - loss: 0.2431 - categorical_accuracy: 0.9250
34304/60000 [================>.............] - ETA: 46s - loss: 0.2429 - categorical_accuracy: 0.9251
34336/60000 [================>.............] - ETA: 45s - loss: 0.2427 - categorical_accuracy: 0.9252
34368/60000 [================>.............] - ETA: 45s - loss: 0.2426 - categorical_accuracy: 0.9252
34400/60000 [================>.............] - ETA: 45s - loss: 0.2424 - categorical_accuracy: 0.9253
34432/60000 [================>.............] - ETA: 45s - loss: 0.2422 - categorical_accuracy: 0.9253
34464/60000 [================>.............] - ETA: 45s - loss: 0.2420 - categorical_accuracy: 0.9254
34496/60000 [================>.............] - ETA: 45s - loss: 0.2419 - categorical_accuracy: 0.9254
34528/60000 [================>.............] - ETA: 45s - loss: 0.2418 - categorical_accuracy: 0.9255
34560/60000 [================>.............] - ETA: 45s - loss: 0.2416 - categorical_accuracy: 0.9255
34592/60000 [================>.............] - ETA: 45s - loss: 0.2415 - categorical_accuracy: 0.9256
34624/60000 [================>.............] - ETA: 45s - loss: 0.2413 - categorical_accuracy: 0.9256
34656/60000 [================>.............] - ETA: 45s - loss: 0.2412 - categorical_accuracy: 0.9257
34688/60000 [================>.............] - ETA: 45s - loss: 0.2412 - categorical_accuracy: 0.9257
34720/60000 [================>.............] - ETA: 45s - loss: 0.2410 - categorical_accuracy: 0.9257
34752/60000 [================>.............] - ETA: 45s - loss: 0.2410 - categorical_accuracy: 0.9258
34784/60000 [================>.............] - ETA: 45s - loss: 0.2409 - categorical_accuracy: 0.9257
34816/60000 [================>.............] - ETA: 45s - loss: 0.2408 - categorical_accuracy: 0.9258
34848/60000 [================>.............] - ETA: 45s - loss: 0.2407 - categorical_accuracy: 0.9258
34880/60000 [================>.............] - ETA: 45s - loss: 0.2405 - categorical_accuracy: 0.9259
34912/60000 [================>.............] - ETA: 44s - loss: 0.2403 - categorical_accuracy: 0.9259
34944/60000 [================>.............] - ETA: 44s - loss: 0.2403 - categorical_accuracy: 0.9259
34976/60000 [================>.............] - ETA: 44s - loss: 0.2400 - categorical_accuracy: 0.9260
35008/60000 [================>.............] - ETA: 44s - loss: 0.2399 - categorical_accuracy: 0.9260
35040/60000 [================>.............] - ETA: 44s - loss: 0.2397 - categorical_accuracy: 0.9261
35072/60000 [================>.............] - ETA: 44s - loss: 0.2395 - categorical_accuracy: 0.9262
35104/60000 [================>.............] - ETA: 44s - loss: 0.2396 - categorical_accuracy: 0.9261
35136/60000 [================>.............] - ETA: 44s - loss: 0.2394 - categorical_accuracy: 0.9262
35168/60000 [================>.............] - ETA: 44s - loss: 0.2392 - categorical_accuracy: 0.9263
35200/60000 [================>.............] - ETA: 44s - loss: 0.2391 - categorical_accuracy: 0.9263
35232/60000 [================>.............] - ETA: 44s - loss: 0.2389 - categorical_accuracy: 0.9263
35264/60000 [================>.............] - ETA: 44s - loss: 0.2390 - categorical_accuracy: 0.9264
35296/60000 [================>.............] - ETA: 44s - loss: 0.2389 - categorical_accuracy: 0.9263
35328/60000 [================>.............] - ETA: 44s - loss: 0.2388 - categorical_accuracy: 0.9264
35360/60000 [================>.............] - ETA: 44s - loss: 0.2387 - categorical_accuracy: 0.9264
35392/60000 [================>.............] - ETA: 44s - loss: 0.2385 - categorical_accuracy: 0.9264
35424/60000 [================>.............] - ETA: 44s - loss: 0.2385 - categorical_accuracy: 0.9264
35456/60000 [================>.............] - ETA: 43s - loss: 0.2383 - categorical_accuracy: 0.9265
35488/60000 [================>.............] - ETA: 43s - loss: 0.2381 - categorical_accuracy: 0.9265
35520/60000 [================>.............] - ETA: 43s - loss: 0.2379 - categorical_accuracy: 0.9265
35552/60000 [================>.............] - ETA: 43s - loss: 0.2381 - categorical_accuracy: 0.9266
35584/60000 [================>.............] - ETA: 43s - loss: 0.2381 - categorical_accuracy: 0.9265
35616/60000 [================>.............] - ETA: 43s - loss: 0.2379 - categorical_accuracy: 0.9265
35648/60000 [================>.............] - ETA: 43s - loss: 0.2381 - categorical_accuracy: 0.9265
35680/60000 [================>.............] - ETA: 43s - loss: 0.2380 - categorical_accuracy: 0.9265
35712/60000 [================>.............] - ETA: 43s - loss: 0.2380 - categorical_accuracy: 0.9265
35744/60000 [================>.............] - ETA: 43s - loss: 0.2380 - categorical_accuracy: 0.9265
35776/60000 [================>.............] - ETA: 43s - loss: 0.2379 - categorical_accuracy: 0.9266
35808/60000 [================>.............] - ETA: 43s - loss: 0.2377 - categorical_accuracy: 0.9266
35840/60000 [================>.............] - ETA: 43s - loss: 0.2376 - categorical_accuracy: 0.9267
35872/60000 [================>.............] - ETA: 43s - loss: 0.2374 - categorical_accuracy: 0.9267
35904/60000 [================>.............] - ETA: 43s - loss: 0.2373 - categorical_accuracy: 0.9267
35936/60000 [================>.............] - ETA: 43s - loss: 0.2372 - categorical_accuracy: 0.9268
35968/60000 [================>.............] - ETA: 43s - loss: 0.2370 - categorical_accuracy: 0.9269
36000/60000 [=================>............] - ETA: 42s - loss: 0.2369 - categorical_accuracy: 0.9269
36032/60000 [=================>............] - ETA: 42s - loss: 0.2367 - categorical_accuracy: 0.9270
36064/60000 [=================>............] - ETA: 42s - loss: 0.2365 - categorical_accuracy: 0.9270
36096/60000 [=================>............] - ETA: 42s - loss: 0.2363 - categorical_accuracy: 0.9271
36128/60000 [=================>............] - ETA: 42s - loss: 0.2361 - categorical_accuracy: 0.9271
36160/60000 [=================>............] - ETA: 42s - loss: 0.2359 - categorical_accuracy: 0.9272
36192/60000 [=================>............] - ETA: 42s - loss: 0.2358 - categorical_accuracy: 0.9273
36224/60000 [=================>............] - ETA: 42s - loss: 0.2356 - categorical_accuracy: 0.9273
36256/60000 [=================>............] - ETA: 42s - loss: 0.2354 - categorical_accuracy: 0.9274
36288/60000 [=================>............] - ETA: 42s - loss: 0.2354 - categorical_accuracy: 0.9274
36320/60000 [=================>............] - ETA: 42s - loss: 0.2355 - categorical_accuracy: 0.9274
36352/60000 [=================>............] - ETA: 42s - loss: 0.2353 - categorical_accuracy: 0.9275
36384/60000 [=================>............] - ETA: 42s - loss: 0.2351 - categorical_accuracy: 0.9275
36416/60000 [=================>............] - ETA: 42s - loss: 0.2351 - categorical_accuracy: 0.9275
36448/60000 [=================>............] - ETA: 42s - loss: 0.2349 - categorical_accuracy: 0.9276
36480/60000 [=================>............] - ETA: 42s - loss: 0.2348 - categorical_accuracy: 0.9276
36512/60000 [=================>............] - ETA: 42s - loss: 0.2347 - categorical_accuracy: 0.9276
36544/60000 [=================>............] - ETA: 42s - loss: 0.2345 - categorical_accuracy: 0.9277
36576/60000 [=================>............] - ETA: 41s - loss: 0.2346 - categorical_accuracy: 0.9277
36608/60000 [=================>............] - ETA: 41s - loss: 0.2345 - categorical_accuracy: 0.9277
36640/60000 [=================>............] - ETA: 41s - loss: 0.2344 - categorical_accuracy: 0.9278
36672/60000 [=================>............] - ETA: 41s - loss: 0.2344 - categorical_accuracy: 0.9278
36704/60000 [=================>............] - ETA: 41s - loss: 0.2342 - categorical_accuracy: 0.9279
36736/60000 [=================>............] - ETA: 41s - loss: 0.2341 - categorical_accuracy: 0.9279
36768/60000 [=================>............] - ETA: 41s - loss: 0.2340 - categorical_accuracy: 0.9279
36800/60000 [=================>............] - ETA: 41s - loss: 0.2338 - categorical_accuracy: 0.9280
36832/60000 [=================>............] - ETA: 41s - loss: 0.2338 - categorical_accuracy: 0.9279
36864/60000 [=================>............] - ETA: 41s - loss: 0.2337 - categorical_accuracy: 0.9280
36896/60000 [=================>............] - ETA: 41s - loss: 0.2338 - categorical_accuracy: 0.9279
36928/60000 [=================>............] - ETA: 41s - loss: 0.2336 - categorical_accuracy: 0.9280
36960/60000 [=================>............] - ETA: 41s - loss: 0.2335 - categorical_accuracy: 0.9280
36992/60000 [=================>............] - ETA: 41s - loss: 0.2333 - categorical_accuracy: 0.9281
37024/60000 [=================>............] - ETA: 41s - loss: 0.2332 - categorical_accuracy: 0.9281
37056/60000 [=================>............] - ETA: 41s - loss: 0.2330 - categorical_accuracy: 0.9282
37088/60000 [=================>............] - ETA: 41s - loss: 0.2329 - categorical_accuracy: 0.9282
37120/60000 [=================>............] - ETA: 40s - loss: 0.2327 - categorical_accuracy: 0.9283
37152/60000 [=================>............] - ETA: 40s - loss: 0.2327 - categorical_accuracy: 0.9283
37184/60000 [=================>............] - ETA: 40s - loss: 0.2326 - categorical_accuracy: 0.9283
37216/60000 [=================>............] - ETA: 40s - loss: 0.2325 - categorical_accuracy: 0.9283
37248/60000 [=================>............] - ETA: 40s - loss: 0.2324 - categorical_accuracy: 0.9283
37280/60000 [=================>............] - ETA: 40s - loss: 0.2322 - categorical_accuracy: 0.9284
37312/60000 [=================>............] - ETA: 40s - loss: 0.2321 - categorical_accuracy: 0.9284
37344/60000 [=================>............] - ETA: 40s - loss: 0.2320 - categorical_accuracy: 0.9284
37376/60000 [=================>............] - ETA: 40s - loss: 0.2318 - categorical_accuracy: 0.9285
37408/60000 [=================>............] - ETA: 40s - loss: 0.2318 - categorical_accuracy: 0.9285
37440/60000 [=================>............] - ETA: 40s - loss: 0.2316 - categorical_accuracy: 0.9286
37472/60000 [=================>............] - ETA: 40s - loss: 0.2316 - categorical_accuracy: 0.9286
37504/60000 [=================>............] - ETA: 40s - loss: 0.2315 - categorical_accuracy: 0.9285
37536/60000 [=================>............] - ETA: 40s - loss: 0.2316 - categorical_accuracy: 0.9285
37568/60000 [=================>............] - ETA: 40s - loss: 0.2314 - categorical_accuracy: 0.9286
37600/60000 [=================>............] - ETA: 40s - loss: 0.2313 - categorical_accuracy: 0.9287
37632/60000 [=================>............] - ETA: 40s - loss: 0.2312 - categorical_accuracy: 0.9287
37664/60000 [=================>............] - ETA: 39s - loss: 0.2311 - categorical_accuracy: 0.9287
37696/60000 [=================>............] - ETA: 39s - loss: 0.2310 - categorical_accuracy: 0.9287
37728/60000 [=================>............] - ETA: 39s - loss: 0.2309 - categorical_accuracy: 0.9288
37760/60000 [=================>............] - ETA: 39s - loss: 0.2307 - categorical_accuracy: 0.9289
37792/60000 [=================>............] - ETA: 39s - loss: 0.2306 - categorical_accuracy: 0.9289
37824/60000 [=================>............] - ETA: 39s - loss: 0.2304 - categorical_accuracy: 0.9289
37856/60000 [=================>............] - ETA: 39s - loss: 0.2302 - categorical_accuracy: 0.9290
37888/60000 [=================>............] - ETA: 39s - loss: 0.2301 - categorical_accuracy: 0.9290
37920/60000 [=================>............] - ETA: 39s - loss: 0.2301 - categorical_accuracy: 0.9290
37952/60000 [=================>............] - ETA: 39s - loss: 0.2301 - categorical_accuracy: 0.9291
37984/60000 [=================>............] - ETA: 39s - loss: 0.2299 - categorical_accuracy: 0.9291
38016/60000 [==================>...........] - ETA: 39s - loss: 0.2299 - categorical_accuracy: 0.9291
38048/60000 [==================>...........] - ETA: 39s - loss: 0.2298 - categorical_accuracy: 0.9291
38080/60000 [==================>...........] - ETA: 39s - loss: 0.2297 - categorical_accuracy: 0.9291
38112/60000 [==================>...........] - ETA: 39s - loss: 0.2296 - categorical_accuracy: 0.9292
38144/60000 [==================>...........] - ETA: 39s - loss: 0.2294 - categorical_accuracy: 0.9293
38176/60000 [==================>...........] - ETA: 39s - loss: 0.2293 - categorical_accuracy: 0.9293
38208/60000 [==================>...........] - ETA: 38s - loss: 0.2292 - categorical_accuracy: 0.9293
38240/60000 [==================>...........] - ETA: 38s - loss: 0.2291 - categorical_accuracy: 0.9293
38272/60000 [==================>...........] - ETA: 38s - loss: 0.2291 - categorical_accuracy: 0.9294
38304/60000 [==================>...........] - ETA: 38s - loss: 0.2291 - categorical_accuracy: 0.9294
38336/60000 [==================>...........] - ETA: 38s - loss: 0.2289 - categorical_accuracy: 0.9295
38368/60000 [==================>...........] - ETA: 38s - loss: 0.2288 - categorical_accuracy: 0.9295
38400/60000 [==================>...........] - ETA: 38s - loss: 0.2288 - categorical_accuracy: 0.9295
38432/60000 [==================>...........] - ETA: 38s - loss: 0.2286 - categorical_accuracy: 0.9296
38464/60000 [==================>...........] - ETA: 38s - loss: 0.2284 - categorical_accuracy: 0.9296
38496/60000 [==================>...........] - ETA: 38s - loss: 0.2283 - categorical_accuracy: 0.9297
38528/60000 [==================>...........] - ETA: 38s - loss: 0.2282 - categorical_accuracy: 0.9297
38560/60000 [==================>...........] - ETA: 38s - loss: 0.2280 - categorical_accuracy: 0.9298
38592/60000 [==================>...........] - ETA: 38s - loss: 0.2279 - categorical_accuracy: 0.9298
38624/60000 [==================>...........] - ETA: 38s - loss: 0.2277 - categorical_accuracy: 0.9299
38656/60000 [==================>...........] - ETA: 38s - loss: 0.2276 - categorical_accuracy: 0.9299
38688/60000 [==================>...........] - ETA: 38s - loss: 0.2274 - categorical_accuracy: 0.9300
38720/60000 [==================>...........] - ETA: 38s - loss: 0.2273 - categorical_accuracy: 0.9300
38752/60000 [==================>...........] - ETA: 38s - loss: 0.2271 - categorical_accuracy: 0.9301
38784/60000 [==================>...........] - ETA: 37s - loss: 0.2270 - categorical_accuracy: 0.9301
38816/60000 [==================>...........] - ETA: 37s - loss: 0.2268 - categorical_accuracy: 0.9302
38848/60000 [==================>...........] - ETA: 37s - loss: 0.2267 - categorical_accuracy: 0.9302
38880/60000 [==================>...........] - ETA: 37s - loss: 0.2266 - categorical_accuracy: 0.9302
38912/60000 [==================>...........] - ETA: 37s - loss: 0.2266 - categorical_accuracy: 0.9303
38944/60000 [==================>...........] - ETA: 37s - loss: 0.2264 - categorical_accuracy: 0.9303
38976/60000 [==================>...........] - ETA: 37s - loss: 0.2263 - categorical_accuracy: 0.9303
39008/60000 [==================>...........] - ETA: 37s - loss: 0.2262 - categorical_accuracy: 0.9304
39040/60000 [==================>...........] - ETA: 37s - loss: 0.2260 - categorical_accuracy: 0.9304
39072/60000 [==================>...........] - ETA: 37s - loss: 0.2259 - categorical_accuracy: 0.9304
39104/60000 [==================>...........] - ETA: 37s - loss: 0.2258 - categorical_accuracy: 0.9305
39136/60000 [==================>...........] - ETA: 37s - loss: 0.2256 - categorical_accuracy: 0.9305
39168/60000 [==================>...........] - ETA: 37s - loss: 0.2255 - categorical_accuracy: 0.9305
39200/60000 [==================>...........] - ETA: 37s - loss: 0.2256 - categorical_accuracy: 0.9305
39232/60000 [==================>...........] - ETA: 37s - loss: 0.2255 - categorical_accuracy: 0.9305
39264/60000 [==================>...........] - ETA: 37s - loss: 0.2253 - categorical_accuracy: 0.9306
39296/60000 [==================>...........] - ETA: 37s - loss: 0.2253 - categorical_accuracy: 0.9306
39328/60000 [==================>...........] - ETA: 37s - loss: 0.2251 - categorical_accuracy: 0.9307
39360/60000 [==================>...........] - ETA: 36s - loss: 0.2251 - categorical_accuracy: 0.9307
39392/60000 [==================>...........] - ETA: 36s - loss: 0.2249 - categorical_accuracy: 0.9307
39424/60000 [==================>...........] - ETA: 36s - loss: 0.2252 - categorical_accuracy: 0.9307
39456/60000 [==================>...........] - ETA: 36s - loss: 0.2250 - categorical_accuracy: 0.9308
39488/60000 [==================>...........] - ETA: 36s - loss: 0.2251 - categorical_accuracy: 0.9307
39520/60000 [==================>...........] - ETA: 36s - loss: 0.2251 - categorical_accuracy: 0.9308
39552/60000 [==================>...........] - ETA: 36s - loss: 0.2249 - categorical_accuracy: 0.9308
39584/60000 [==================>...........] - ETA: 36s - loss: 0.2248 - categorical_accuracy: 0.9308
39616/60000 [==================>...........] - ETA: 36s - loss: 0.2249 - categorical_accuracy: 0.9308
39648/60000 [==================>...........] - ETA: 36s - loss: 0.2247 - categorical_accuracy: 0.9309
39680/60000 [==================>...........] - ETA: 36s - loss: 0.2247 - categorical_accuracy: 0.9308
39712/60000 [==================>...........] - ETA: 36s - loss: 0.2246 - categorical_accuracy: 0.9309
39744/60000 [==================>...........] - ETA: 36s - loss: 0.2245 - categorical_accuracy: 0.9309
39776/60000 [==================>...........] - ETA: 36s - loss: 0.2246 - categorical_accuracy: 0.9309
39808/60000 [==================>...........] - ETA: 36s - loss: 0.2244 - categorical_accuracy: 0.9310
39840/60000 [==================>...........] - ETA: 36s - loss: 0.2243 - categorical_accuracy: 0.9310
39872/60000 [==================>...........] - ETA: 36s - loss: 0.2241 - categorical_accuracy: 0.9311
39904/60000 [==================>...........] - ETA: 35s - loss: 0.2240 - categorical_accuracy: 0.9311
39936/60000 [==================>...........] - ETA: 35s - loss: 0.2241 - categorical_accuracy: 0.9311
39968/60000 [==================>...........] - ETA: 35s - loss: 0.2240 - categorical_accuracy: 0.9311
40000/60000 [===================>..........] - ETA: 35s - loss: 0.2239 - categorical_accuracy: 0.9312
40032/60000 [===================>..........] - ETA: 35s - loss: 0.2239 - categorical_accuracy: 0.9312
40064/60000 [===================>..........] - ETA: 35s - loss: 0.2238 - categorical_accuracy: 0.9312
40096/60000 [===================>..........] - ETA: 35s - loss: 0.2237 - categorical_accuracy: 0.9312
40128/60000 [===================>..........] - ETA: 35s - loss: 0.2236 - categorical_accuracy: 0.9313
40160/60000 [===================>..........] - ETA: 35s - loss: 0.2235 - categorical_accuracy: 0.9313
40192/60000 [===================>..........] - ETA: 35s - loss: 0.2234 - categorical_accuracy: 0.9314
40224/60000 [===================>..........] - ETA: 35s - loss: 0.2232 - categorical_accuracy: 0.9314
40256/60000 [===================>..........] - ETA: 35s - loss: 0.2231 - categorical_accuracy: 0.9315
40288/60000 [===================>..........] - ETA: 35s - loss: 0.2230 - categorical_accuracy: 0.9315
40320/60000 [===================>..........] - ETA: 35s - loss: 0.2228 - categorical_accuracy: 0.9315
40352/60000 [===================>..........] - ETA: 35s - loss: 0.2227 - categorical_accuracy: 0.9316
40384/60000 [===================>..........] - ETA: 35s - loss: 0.2226 - categorical_accuracy: 0.9316
40416/60000 [===================>..........] - ETA: 35s - loss: 0.2224 - categorical_accuracy: 0.9317
40448/60000 [===================>..........] - ETA: 35s - loss: 0.2223 - categorical_accuracy: 0.9317
40480/60000 [===================>..........] - ETA: 34s - loss: 0.2222 - categorical_accuracy: 0.9317
40512/60000 [===================>..........] - ETA: 34s - loss: 0.2221 - categorical_accuracy: 0.9318
40544/60000 [===================>..........] - ETA: 34s - loss: 0.2219 - categorical_accuracy: 0.9318
40576/60000 [===================>..........] - ETA: 34s - loss: 0.2218 - categorical_accuracy: 0.9319
40608/60000 [===================>..........] - ETA: 34s - loss: 0.2217 - categorical_accuracy: 0.9319
40640/60000 [===================>..........] - ETA: 34s - loss: 0.2217 - categorical_accuracy: 0.9319
40672/60000 [===================>..........] - ETA: 34s - loss: 0.2217 - categorical_accuracy: 0.9319
40704/60000 [===================>..........] - ETA: 34s - loss: 0.2216 - categorical_accuracy: 0.9319
40736/60000 [===================>..........] - ETA: 34s - loss: 0.2215 - categorical_accuracy: 0.9320
40768/60000 [===================>..........] - ETA: 34s - loss: 0.2214 - categorical_accuracy: 0.9320
40800/60000 [===================>..........] - ETA: 34s - loss: 0.2212 - categorical_accuracy: 0.9321
40832/60000 [===================>..........] - ETA: 34s - loss: 0.2211 - categorical_accuracy: 0.9321
40864/60000 [===================>..........] - ETA: 34s - loss: 0.2212 - categorical_accuracy: 0.9321
40896/60000 [===================>..........] - ETA: 34s - loss: 0.2213 - categorical_accuracy: 0.9320
40928/60000 [===================>..........] - ETA: 34s - loss: 0.2213 - categorical_accuracy: 0.9321
40960/60000 [===================>..........] - ETA: 34s - loss: 0.2212 - categorical_accuracy: 0.9321
40992/60000 [===================>..........] - ETA: 34s - loss: 0.2213 - categorical_accuracy: 0.9321
41024/60000 [===================>..........] - ETA: 33s - loss: 0.2211 - categorical_accuracy: 0.9322
41056/60000 [===================>..........] - ETA: 33s - loss: 0.2211 - categorical_accuracy: 0.9322
41088/60000 [===================>..........] - ETA: 33s - loss: 0.2209 - categorical_accuracy: 0.9322
41120/60000 [===================>..........] - ETA: 33s - loss: 0.2208 - categorical_accuracy: 0.9323
41152/60000 [===================>..........] - ETA: 33s - loss: 0.2208 - categorical_accuracy: 0.9323
41184/60000 [===================>..........] - ETA: 33s - loss: 0.2207 - categorical_accuracy: 0.9323
41216/60000 [===================>..........] - ETA: 33s - loss: 0.2206 - categorical_accuracy: 0.9324
41248/60000 [===================>..........] - ETA: 33s - loss: 0.2205 - categorical_accuracy: 0.9324
41280/60000 [===================>..........] - ETA: 33s - loss: 0.2204 - categorical_accuracy: 0.9324
41312/60000 [===================>..........] - ETA: 33s - loss: 0.2203 - categorical_accuracy: 0.9325
41344/60000 [===================>..........] - ETA: 33s - loss: 0.2201 - categorical_accuracy: 0.9325
41376/60000 [===================>..........] - ETA: 33s - loss: 0.2201 - categorical_accuracy: 0.9325
41408/60000 [===================>..........] - ETA: 33s - loss: 0.2201 - categorical_accuracy: 0.9325
41440/60000 [===================>..........] - ETA: 33s - loss: 0.2200 - categorical_accuracy: 0.9325
41472/60000 [===================>..........] - ETA: 33s - loss: 0.2200 - categorical_accuracy: 0.9325
41504/60000 [===================>..........] - ETA: 33s - loss: 0.2201 - categorical_accuracy: 0.9325
41536/60000 [===================>..........] - ETA: 33s - loss: 0.2201 - categorical_accuracy: 0.9325
41568/60000 [===================>..........] - ETA: 33s - loss: 0.2200 - categorical_accuracy: 0.9326
41600/60000 [===================>..........] - ETA: 32s - loss: 0.2198 - categorical_accuracy: 0.9326
41632/60000 [===================>..........] - ETA: 32s - loss: 0.2197 - categorical_accuracy: 0.9326
41664/60000 [===================>..........] - ETA: 32s - loss: 0.2196 - categorical_accuracy: 0.9327
41696/60000 [===================>..........] - ETA: 32s - loss: 0.2196 - categorical_accuracy: 0.9327
41728/60000 [===================>..........] - ETA: 32s - loss: 0.2196 - categorical_accuracy: 0.9327
41760/60000 [===================>..........] - ETA: 32s - loss: 0.2195 - categorical_accuracy: 0.9327
41792/60000 [===================>..........] - ETA: 32s - loss: 0.2195 - categorical_accuracy: 0.9327
41824/60000 [===================>..........] - ETA: 32s - loss: 0.2196 - categorical_accuracy: 0.9327
41856/60000 [===================>..........] - ETA: 32s - loss: 0.2194 - categorical_accuracy: 0.9327
41888/60000 [===================>..........] - ETA: 32s - loss: 0.2193 - categorical_accuracy: 0.9328
41920/60000 [===================>..........] - ETA: 32s - loss: 0.2194 - categorical_accuracy: 0.9328
41952/60000 [===================>..........] - ETA: 32s - loss: 0.2193 - categorical_accuracy: 0.9328
41984/60000 [===================>..........] - ETA: 32s - loss: 0.2192 - categorical_accuracy: 0.9329
42016/60000 [====================>.........] - ETA: 32s - loss: 0.2190 - categorical_accuracy: 0.9329
42048/60000 [====================>.........] - ETA: 32s - loss: 0.2190 - categorical_accuracy: 0.9329
42080/60000 [====================>.........] - ETA: 32s - loss: 0.2189 - categorical_accuracy: 0.9329
42112/60000 [====================>.........] - ETA: 32s - loss: 0.2188 - categorical_accuracy: 0.9329
42144/60000 [====================>.........] - ETA: 31s - loss: 0.2187 - categorical_accuracy: 0.9329
42176/60000 [====================>.........] - ETA: 31s - loss: 0.2188 - categorical_accuracy: 0.9329
42208/60000 [====================>.........] - ETA: 31s - loss: 0.2188 - categorical_accuracy: 0.9329
42240/60000 [====================>.........] - ETA: 31s - loss: 0.2187 - categorical_accuracy: 0.9329
42272/60000 [====================>.........] - ETA: 31s - loss: 0.2185 - categorical_accuracy: 0.9330
42304/60000 [====================>.........] - ETA: 31s - loss: 0.2184 - categorical_accuracy: 0.9330
42336/60000 [====================>.........] - ETA: 31s - loss: 0.2183 - categorical_accuracy: 0.9331
42368/60000 [====================>.........] - ETA: 31s - loss: 0.2182 - categorical_accuracy: 0.9331
42400/60000 [====================>.........] - ETA: 31s - loss: 0.2180 - categorical_accuracy: 0.9331
42432/60000 [====================>.........] - ETA: 31s - loss: 0.2179 - categorical_accuracy: 0.9332
42464/60000 [====================>.........] - ETA: 31s - loss: 0.2179 - categorical_accuracy: 0.9332
42496/60000 [====================>.........] - ETA: 31s - loss: 0.2178 - categorical_accuracy: 0.9332
42528/60000 [====================>.........] - ETA: 31s - loss: 0.2178 - categorical_accuracy: 0.9332
42560/60000 [====================>.........] - ETA: 31s - loss: 0.2178 - categorical_accuracy: 0.9332
42592/60000 [====================>.........] - ETA: 31s - loss: 0.2176 - categorical_accuracy: 0.9332
42624/60000 [====================>.........] - ETA: 31s - loss: 0.2177 - categorical_accuracy: 0.9332
42656/60000 [====================>.........] - ETA: 31s - loss: 0.2176 - categorical_accuracy: 0.9333
42688/60000 [====================>.........] - ETA: 31s - loss: 0.2175 - categorical_accuracy: 0.9333
42720/60000 [====================>.........] - ETA: 30s - loss: 0.2174 - categorical_accuracy: 0.9333
42752/60000 [====================>.........] - ETA: 30s - loss: 0.2173 - categorical_accuracy: 0.9333
42784/60000 [====================>.........] - ETA: 30s - loss: 0.2171 - categorical_accuracy: 0.9334
42816/60000 [====================>.........] - ETA: 30s - loss: 0.2170 - categorical_accuracy: 0.9334
42848/60000 [====================>.........] - ETA: 30s - loss: 0.2169 - categorical_accuracy: 0.9334
42880/60000 [====================>.........] - ETA: 30s - loss: 0.2169 - categorical_accuracy: 0.9334
42912/60000 [====================>.........] - ETA: 30s - loss: 0.2168 - categorical_accuracy: 0.9335
42944/60000 [====================>.........] - ETA: 30s - loss: 0.2167 - categorical_accuracy: 0.9335
42976/60000 [====================>.........] - ETA: 30s - loss: 0.2168 - categorical_accuracy: 0.9335
43008/60000 [====================>.........] - ETA: 30s - loss: 0.2167 - categorical_accuracy: 0.9335
43040/60000 [====================>.........] - ETA: 30s - loss: 0.2166 - categorical_accuracy: 0.9335
43072/60000 [====================>.........] - ETA: 30s - loss: 0.2165 - categorical_accuracy: 0.9336
43104/60000 [====================>.........] - ETA: 30s - loss: 0.2164 - categorical_accuracy: 0.9336
43136/60000 [====================>.........] - ETA: 30s - loss: 0.2164 - categorical_accuracy: 0.9336
43168/60000 [====================>.........] - ETA: 30s - loss: 0.2163 - categorical_accuracy: 0.9336
43200/60000 [====================>.........] - ETA: 30s - loss: 0.2162 - categorical_accuracy: 0.9336
43232/60000 [====================>.........] - ETA: 30s - loss: 0.2161 - categorical_accuracy: 0.9337
43264/60000 [====================>.........] - ETA: 29s - loss: 0.2161 - categorical_accuracy: 0.9336
43296/60000 [====================>.........] - ETA: 29s - loss: 0.2160 - categorical_accuracy: 0.9336
43328/60000 [====================>.........] - ETA: 29s - loss: 0.2160 - categorical_accuracy: 0.9336
43360/60000 [====================>.........] - ETA: 29s - loss: 0.2158 - categorical_accuracy: 0.9337
43392/60000 [====================>.........] - ETA: 29s - loss: 0.2157 - categorical_accuracy: 0.9337
43424/60000 [====================>.........] - ETA: 29s - loss: 0.2155 - categorical_accuracy: 0.9338
43456/60000 [====================>.........] - ETA: 29s - loss: 0.2154 - categorical_accuracy: 0.9338
43488/60000 [====================>.........] - ETA: 29s - loss: 0.2153 - categorical_accuracy: 0.9338
43520/60000 [====================>.........] - ETA: 29s - loss: 0.2152 - categorical_accuracy: 0.9339
43552/60000 [====================>.........] - ETA: 29s - loss: 0.2151 - categorical_accuracy: 0.9339
43584/60000 [====================>.........] - ETA: 29s - loss: 0.2150 - categorical_accuracy: 0.9340
43616/60000 [====================>.........] - ETA: 29s - loss: 0.2149 - categorical_accuracy: 0.9340
43648/60000 [====================>.........] - ETA: 29s - loss: 0.2148 - categorical_accuracy: 0.9340
43680/60000 [====================>.........] - ETA: 29s - loss: 0.2147 - categorical_accuracy: 0.9340
43712/60000 [====================>.........] - ETA: 29s - loss: 0.2145 - categorical_accuracy: 0.9341
43744/60000 [====================>.........] - ETA: 29s - loss: 0.2145 - categorical_accuracy: 0.9341
43776/60000 [====================>.........] - ETA: 29s - loss: 0.2144 - categorical_accuracy: 0.9341
43808/60000 [====================>.........] - ETA: 28s - loss: 0.2143 - categorical_accuracy: 0.9342
43840/60000 [====================>.........] - ETA: 28s - loss: 0.2143 - categorical_accuracy: 0.9342
43872/60000 [====================>.........] - ETA: 28s - loss: 0.2142 - categorical_accuracy: 0.9342
43904/60000 [====================>.........] - ETA: 28s - loss: 0.2141 - categorical_accuracy: 0.9342
43936/60000 [====================>.........] - ETA: 28s - loss: 0.2140 - categorical_accuracy: 0.9342
43968/60000 [====================>.........] - ETA: 28s - loss: 0.2139 - categorical_accuracy: 0.9343
44000/60000 [=====================>........] - ETA: 28s - loss: 0.2138 - categorical_accuracy: 0.9343
44032/60000 [=====================>........] - ETA: 28s - loss: 0.2138 - categorical_accuracy: 0.9343
44064/60000 [=====================>........] - ETA: 28s - loss: 0.2137 - categorical_accuracy: 0.9344
44096/60000 [=====================>........] - ETA: 28s - loss: 0.2136 - categorical_accuracy: 0.9344
44128/60000 [=====================>........] - ETA: 28s - loss: 0.2134 - categorical_accuracy: 0.9344
44160/60000 [=====================>........] - ETA: 28s - loss: 0.2134 - categorical_accuracy: 0.9344
44192/60000 [=====================>........] - ETA: 28s - loss: 0.2133 - categorical_accuracy: 0.9344
44224/60000 [=====================>........] - ETA: 28s - loss: 0.2133 - categorical_accuracy: 0.9344
44256/60000 [=====================>........] - ETA: 28s - loss: 0.2132 - categorical_accuracy: 0.9344
44288/60000 [=====================>........] - ETA: 28s - loss: 0.2132 - categorical_accuracy: 0.9344
44320/60000 [=====================>........] - ETA: 28s - loss: 0.2131 - categorical_accuracy: 0.9345
44352/60000 [=====================>........] - ETA: 28s - loss: 0.2130 - categorical_accuracy: 0.9345
44384/60000 [=====================>........] - ETA: 27s - loss: 0.2128 - categorical_accuracy: 0.9345
44416/60000 [=====================>........] - ETA: 27s - loss: 0.2127 - categorical_accuracy: 0.9346
44448/60000 [=====================>........] - ETA: 27s - loss: 0.2126 - categorical_accuracy: 0.9346
44480/60000 [=====================>........] - ETA: 27s - loss: 0.2127 - categorical_accuracy: 0.9346
44512/60000 [=====================>........] - ETA: 27s - loss: 0.2126 - categorical_accuracy: 0.9346
44544/60000 [=====================>........] - ETA: 27s - loss: 0.2125 - categorical_accuracy: 0.9347
44576/60000 [=====================>........] - ETA: 27s - loss: 0.2124 - categorical_accuracy: 0.9347
44608/60000 [=====================>........] - ETA: 27s - loss: 0.2123 - categorical_accuracy: 0.9347
44640/60000 [=====================>........] - ETA: 27s - loss: 0.2122 - categorical_accuracy: 0.9348
44672/60000 [=====================>........] - ETA: 27s - loss: 0.2121 - categorical_accuracy: 0.9348
44704/60000 [=====================>........] - ETA: 27s - loss: 0.2120 - categorical_accuracy: 0.9348
44736/60000 [=====================>........] - ETA: 27s - loss: 0.2118 - categorical_accuracy: 0.9349
44768/60000 [=====================>........] - ETA: 27s - loss: 0.2117 - categorical_accuracy: 0.9349
44800/60000 [=====================>........] - ETA: 27s - loss: 0.2115 - categorical_accuracy: 0.9350
44832/60000 [=====================>........] - ETA: 27s - loss: 0.2115 - categorical_accuracy: 0.9350
44864/60000 [=====================>........] - ETA: 27s - loss: 0.2114 - categorical_accuracy: 0.9350
44896/60000 [=====================>........] - ETA: 27s - loss: 0.2113 - categorical_accuracy: 0.9350
44928/60000 [=====================>........] - ETA: 26s - loss: 0.2113 - categorical_accuracy: 0.9351
44960/60000 [=====================>........] - ETA: 26s - loss: 0.2113 - categorical_accuracy: 0.9350
44992/60000 [=====================>........] - ETA: 26s - loss: 0.2112 - categorical_accuracy: 0.9350
45024/60000 [=====================>........] - ETA: 26s - loss: 0.2111 - categorical_accuracy: 0.9351
45056/60000 [=====================>........] - ETA: 26s - loss: 0.2112 - categorical_accuracy: 0.9350
45088/60000 [=====================>........] - ETA: 26s - loss: 0.2111 - categorical_accuracy: 0.9351
45120/60000 [=====================>........] - ETA: 26s - loss: 0.2110 - categorical_accuracy: 0.9351
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2109 - categorical_accuracy: 0.9351
45184/60000 [=====================>........] - ETA: 26s - loss: 0.2109 - categorical_accuracy: 0.9351
45216/60000 [=====================>........] - ETA: 26s - loss: 0.2108 - categorical_accuracy: 0.9352
45248/60000 [=====================>........] - ETA: 26s - loss: 0.2107 - categorical_accuracy: 0.9352
45280/60000 [=====================>........] - ETA: 26s - loss: 0.2105 - categorical_accuracy: 0.9352
45312/60000 [=====================>........] - ETA: 26s - loss: 0.2105 - categorical_accuracy: 0.9352
45344/60000 [=====================>........] - ETA: 26s - loss: 0.2105 - categorical_accuracy: 0.9352
45376/60000 [=====================>........] - ETA: 26s - loss: 0.2105 - categorical_accuracy: 0.9352
45408/60000 [=====================>........] - ETA: 26s - loss: 0.2104 - categorical_accuracy: 0.9353
45440/60000 [=====================>........] - ETA: 26s - loss: 0.2104 - categorical_accuracy: 0.9353
45472/60000 [=====================>........] - ETA: 26s - loss: 0.2103 - categorical_accuracy: 0.9353
45504/60000 [=====================>........] - ETA: 25s - loss: 0.2103 - categorical_accuracy: 0.9353
45536/60000 [=====================>........] - ETA: 25s - loss: 0.2103 - categorical_accuracy: 0.9353
45568/60000 [=====================>........] - ETA: 25s - loss: 0.2102 - categorical_accuracy: 0.9353
45600/60000 [=====================>........] - ETA: 25s - loss: 0.2101 - categorical_accuracy: 0.9354
45632/60000 [=====================>........] - ETA: 25s - loss: 0.2100 - categorical_accuracy: 0.9354
45664/60000 [=====================>........] - ETA: 25s - loss: 0.2098 - categorical_accuracy: 0.9354
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2097 - categorical_accuracy: 0.9354
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2096 - categorical_accuracy: 0.9355
45760/60000 [=====================>........] - ETA: 25s - loss: 0.2095 - categorical_accuracy: 0.9355
45792/60000 [=====================>........] - ETA: 25s - loss: 0.2093 - categorical_accuracy: 0.9356
45824/60000 [=====================>........] - ETA: 25s - loss: 0.2092 - categorical_accuracy: 0.9356
45856/60000 [=====================>........] - ETA: 25s - loss: 0.2092 - categorical_accuracy: 0.9356
45888/60000 [=====================>........] - ETA: 25s - loss: 0.2090 - categorical_accuracy: 0.9357
45920/60000 [=====================>........] - ETA: 25s - loss: 0.2089 - categorical_accuracy: 0.9357
45952/60000 [=====================>........] - ETA: 25s - loss: 0.2088 - categorical_accuracy: 0.9358
45984/60000 [=====================>........] - ETA: 25s - loss: 0.2087 - categorical_accuracy: 0.9358
46016/60000 [======================>.......] - ETA: 25s - loss: 0.2086 - categorical_accuracy: 0.9358
46048/60000 [======================>.......] - ETA: 24s - loss: 0.2085 - categorical_accuracy: 0.9358
46080/60000 [======================>.......] - ETA: 24s - loss: 0.2084 - categorical_accuracy: 0.9359
46112/60000 [======================>.......] - ETA: 24s - loss: 0.2083 - categorical_accuracy: 0.9359
46144/60000 [======================>.......] - ETA: 24s - loss: 0.2083 - categorical_accuracy: 0.9359
46176/60000 [======================>.......] - ETA: 24s - loss: 0.2083 - categorical_accuracy: 0.9359
46208/60000 [======================>.......] - ETA: 24s - loss: 0.2082 - categorical_accuracy: 0.9359
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2081 - categorical_accuracy: 0.9359
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2080 - categorical_accuracy: 0.9360
46304/60000 [======================>.......] - ETA: 24s - loss: 0.2079 - categorical_accuracy: 0.9360
46336/60000 [======================>.......] - ETA: 24s - loss: 0.2077 - categorical_accuracy: 0.9361
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2076 - categorical_accuracy: 0.9361
46400/60000 [======================>.......] - ETA: 24s - loss: 0.2075 - categorical_accuracy: 0.9361
46432/60000 [======================>.......] - ETA: 24s - loss: 0.2074 - categorical_accuracy: 0.9362
46464/60000 [======================>.......] - ETA: 24s - loss: 0.2073 - categorical_accuracy: 0.9362
46496/60000 [======================>.......] - ETA: 24s - loss: 0.2072 - categorical_accuracy: 0.9362
46528/60000 [======================>.......] - ETA: 24s - loss: 0.2071 - categorical_accuracy: 0.9362
46560/60000 [======================>.......] - ETA: 24s - loss: 0.2070 - categorical_accuracy: 0.9363
46592/60000 [======================>.......] - ETA: 23s - loss: 0.2069 - categorical_accuracy: 0.9363
46624/60000 [======================>.......] - ETA: 23s - loss: 0.2069 - categorical_accuracy: 0.9363
46656/60000 [======================>.......] - ETA: 23s - loss: 0.2068 - categorical_accuracy: 0.9363
46688/60000 [======================>.......] - ETA: 23s - loss: 0.2067 - categorical_accuracy: 0.9364
46720/60000 [======================>.......] - ETA: 23s - loss: 0.2066 - categorical_accuracy: 0.9364
46752/60000 [======================>.......] - ETA: 23s - loss: 0.2065 - categorical_accuracy: 0.9364
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2064 - categorical_accuracy: 0.9365
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2063 - categorical_accuracy: 0.9365
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2062 - categorical_accuracy: 0.9365
46880/60000 [======================>.......] - ETA: 23s - loss: 0.2060 - categorical_accuracy: 0.9366
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2059 - categorical_accuracy: 0.9366
46944/60000 [======================>.......] - ETA: 23s - loss: 0.2060 - categorical_accuracy: 0.9366
46976/60000 [======================>.......] - ETA: 23s - loss: 0.2059 - categorical_accuracy: 0.9366
47008/60000 [======================>.......] - ETA: 23s - loss: 0.2059 - categorical_accuracy: 0.9366
47040/60000 [======================>.......] - ETA: 23s - loss: 0.2058 - categorical_accuracy: 0.9366
47072/60000 [======================>.......] - ETA: 23s - loss: 0.2058 - categorical_accuracy: 0.9366
47104/60000 [======================>.......] - ETA: 23s - loss: 0.2058 - categorical_accuracy: 0.9367
47136/60000 [======================>.......] - ETA: 23s - loss: 0.2057 - categorical_accuracy: 0.9367
47168/60000 [======================>.......] - ETA: 22s - loss: 0.2056 - categorical_accuracy: 0.9367
47200/60000 [======================>.......] - ETA: 22s - loss: 0.2055 - categorical_accuracy: 0.9368
47232/60000 [======================>.......] - ETA: 22s - loss: 0.2054 - categorical_accuracy: 0.9368
47264/60000 [======================>.......] - ETA: 22s - loss: 0.2053 - categorical_accuracy: 0.9368
47296/60000 [======================>.......] - ETA: 22s - loss: 0.2052 - categorical_accuracy: 0.9368
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2051 - categorical_accuracy: 0.9369
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9369
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2050 - categorical_accuracy: 0.9369
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2049 - categorical_accuracy: 0.9369
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2049 - categorical_accuracy: 0.9369
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2048 - categorical_accuracy: 0.9369
47520/60000 [======================>.......] - ETA: 22s - loss: 0.2048 - categorical_accuracy: 0.9369
47552/60000 [======================>.......] - ETA: 22s - loss: 0.2047 - categorical_accuracy: 0.9369
47584/60000 [======================>.......] - ETA: 22s - loss: 0.2046 - categorical_accuracy: 0.9369
47616/60000 [======================>.......] - ETA: 22s - loss: 0.2045 - categorical_accuracy: 0.9369
47648/60000 [======================>.......] - ETA: 22s - loss: 0.2045 - categorical_accuracy: 0.9369
47680/60000 [======================>.......] - ETA: 22s - loss: 0.2043 - categorical_accuracy: 0.9370
47712/60000 [======================>.......] - ETA: 22s - loss: 0.2042 - categorical_accuracy: 0.9370
47744/60000 [======================>.......] - ETA: 21s - loss: 0.2041 - categorical_accuracy: 0.9370
47776/60000 [======================>.......] - ETA: 21s - loss: 0.2042 - categorical_accuracy: 0.9371
47808/60000 [======================>.......] - ETA: 21s - loss: 0.2040 - categorical_accuracy: 0.9371
47840/60000 [======================>.......] - ETA: 21s - loss: 0.2040 - categorical_accuracy: 0.9371
47872/60000 [======================>.......] - ETA: 21s - loss: 0.2039 - categorical_accuracy: 0.9372
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2037 - categorical_accuracy: 0.9372
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2036 - categorical_accuracy: 0.9372
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2036 - categorical_accuracy: 0.9372
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2036 - categorical_accuracy: 0.9373
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2034 - categorical_accuracy: 0.9373
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2033 - categorical_accuracy: 0.9373
48096/60000 [=======================>......] - ETA: 21s - loss: 0.2032 - categorical_accuracy: 0.9374
48128/60000 [=======================>......] - ETA: 21s - loss: 0.2031 - categorical_accuracy: 0.9374
48160/60000 [=======================>......] - ETA: 21s - loss: 0.2030 - categorical_accuracy: 0.9375
48192/60000 [=======================>......] - ETA: 21s - loss: 0.2028 - categorical_accuracy: 0.9375
48224/60000 [=======================>......] - ETA: 21s - loss: 0.2027 - categorical_accuracy: 0.9375
48256/60000 [=======================>......] - ETA: 21s - loss: 0.2028 - categorical_accuracy: 0.9375
48288/60000 [=======================>......] - ETA: 20s - loss: 0.2028 - categorical_accuracy: 0.9375
48320/60000 [=======================>......] - ETA: 20s - loss: 0.2027 - categorical_accuracy: 0.9376
48352/60000 [=======================>......] - ETA: 20s - loss: 0.2026 - categorical_accuracy: 0.9376
48384/60000 [=======================>......] - ETA: 20s - loss: 0.2025 - categorical_accuracy: 0.9376
48416/60000 [=======================>......] - ETA: 20s - loss: 0.2024 - categorical_accuracy: 0.9377
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2024 - categorical_accuracy: 0.9376
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2023 - categorical_accuracy: 0.9377
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2022 - categorical_accuracy: 0.9377
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2025 - categorical_accuracy: 0.9377
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2024 - categorical_accuracy: 0.9377
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2023 - categorical_accuracy: 0.9377
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2022 - categorical_accuracy: 0.9378
48672/60000 [=======================>......] - ETA: 20s - loss: 0.2021 - categorical_accuracy: 0.9378
48704/60000 [=======================>......] - ETA: 20s - loss: 0.2022 - categorical_accuracy: 0.9378
48736/60000 [=======================>......] - ETA: 20s - loss: 0.2021 - categorical_accuracy: 0.9379
48768/60000 [=======================>......] - ETA: 20s - loss: 0.2020 - categorical_accuracy: 0.9379
48800/60000 [=======================>......] - ETA: 20s - loss: 0.2019 - categorical_accuracy: 0.9379
48832/60000 [=======================>......] - ETA: 20s - loss: 0.2018 - categorical_accuracy: 0.9380
48864/60000 [=======================>......] - ETA: 19s - loss: 0.2017 - categorical_accuracy: 0.9380
48896/60000 [=======================>......] - ETA: 19s - loss: 0.2016 - categorical_accuracy: 0.9381
48928/60000 [=======================>......] - ETA: 19s - loss: 0.2015 - categorical_accuracy: 0.9381
48960/60000 [=======================>......] - ETA: 19s - loss: 0.2014 - categorical_accuracy: 0.9381
48992/60000 [=======================>......] - ETA: 19s - loss: 0.2013 - categorical_accuracy: 0.9381
49024/60000 [=======================>......] - ETA: 19s - loss: 0.2012 - categorical_accuracy: 0.9382
49056/60000 [=======================>......] - ETA: 19s - loss: 0.2011 - categorical_accuracy: 0.9382
49088/60000 [=======================>......] - ETA: 19s - loss: 0.2011 - categorical_accuracy: 0.9382
49120/60000 [=======================>......] - ETA: 19s - loss: 0.2009 - categorical_accuracy: 0.9383
49152/60000 [=======================>......] - ETA: 19s - loss: 0.2009 - categorical_accuracy: 0.9383
49184/60000 [=======================>......] - ETA: 19s - loss: 0.2010 - categorical_accuracy: 0.9383
49216/60000 [=======================>......] - ETA: 19s - loss: 0.2010 - categorical_accuracy: 0.9382
49248/60000 [=======================>......] - ETA: 19s - loss: 0.2010 - categorical_accuracy: 0.9382
49280/60000 [=======================>......] - ETA: 19s - loss: 0.2009 - categorical_accuracy: 0.9383
49312/60000 [=======================>......] - ETA: 19s - loss: 0.2008 - categorical_accuracy: 0.9383
49344/60000 [=======================>......] - ETA: 19s - loss: 0.2007 - categorical_accuracy: 0.9383
49376/60000 [=======================>......] - ETA: 19s - loss: 0.2006 - categorical_accuracy: 0.9383
49408/60000 [=======================>......] - ETA: 18s - loss: 0.2009 - categorical_accuracy: 0.9383
49440/60000 [=======================>......] - ETA: 18s - loss: 0.2008 - categorical_accuracy: 0.9383
49472/60000 [=======================>......] - ETA: 18s - loss: 0.2007 - categorical_accuracy: 0.9384
49504/60000 [=======================>......] - ETA: 18s - loss: 0.2006 - categorical_accuracy: 0.9384
49536/60000 [=======================>......] - ETA: 18s - loss: 0.2005 - categorical_accuracy: 0.9384
49568/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9384
49600/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9384
49632/60000 [=======================>......] - ETA: 18s - loss: 0.2004 - categorical_accuracy: 0.9384
49664/60000 [=======================>......] - ETA: 18s - loss: 0.2003 - categorical_accuracy: 0.9384
49696/60000 [=======================>......] - ETA: 18s - loss: 0.2002 - categorical_accuracy: 0.9385
49728/60000 [=======================>......] - ETA: 18s - loss: 0.2001 - categorical_accuracy: 0.9385
49760/60000 [=======================>......] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9385
49792/60000 [=======================>......] - ETA: 18s - loss: 0.1999 - categorical_accuracy: 0.9386
49824/60000 [=======================>......] - ETA: 18s - loss: 0.2000 - categorical_accuracy: 0.9386
49856/60000 [=======================>......] - ETA: 18s - loss: 0.1999 - categorical_accuracy: 0.9386
49888/60000 [=======================>......] - ETA: 18s - loss: 0.1998 - categorical_accuracy: 0.9386
49920/60000 [=======================>......] - ETA: 18s - loss: 0.1997 - categorical_accuracy: 0.9387
49952/60000 [=======================>......] - ETA: 17s - loss: 0.1999 - categorical_accuracy: 0.9386
49984/60000 [=======================>......] - ETA: 17s - loss: 0.1998 - categorical_accuracy: 0.9386
50016/60000 [========================>.....] - ETA: 17s - loss: 0.1997 - categorical_accuracy: 0.9386
50048/60000 [========================>.....] - ETA: 17s - loss: 0.1997 - categorical_accuracy: 0.9387
50080/60000 [========================>.....] - ETA: 17s - loss: 0.1996 - categorical_accuracy: 0.9387
50112/60000 [========================>.....] - ETA: 17s - loss: 0.1995 - categorical_accuracy: 0.9387
50144/60000 [========================>.....] - ETA: 17s - loss: 0.1996 - categorical_accuracy: 0.9387
50176/60000 [========================>.....] - ETA: 17s - loss: 0.1995 - categorical_accuracy: 0.9388
50208/60000 [========================>.....] - ETA: 17s - loss: 0.1994 - categorical_accuracy: 0.9388
50240/60000 [========================>.....] - ETA: 17s - loss: 0.1993 - categorical_accuracy: 0.9388
50272/60000 [========================>.....] - ETA: 17s - loss: 0.1993 - categorical_accuracy: 0.9389
50304/60000 [========================>.....] - ETA: 17s - loss: 0.1992 - categorical_accuracy: 0.9389
50336/60000 [========================>.....] - ETA: 17s - loss: 0.1991 - categorical_accuracy: 0.9389
50368/60000 [========================>.....] - ETA: 17s - loss: 0.1991 - categorical_accuracy: 0.9389
50400/60000 [========================>.....] - ETA: 17s - loss: 0.1990 - categorical_accuracy: 0.9389
50432/60000 [========================>.....] - ETA: 17s - loss: 0.1990 - categorical_accuracy: 0.9389
50464/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9390
50496/60000 [========================>.....] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9390
50528/60000 [========================>.....] - ETA: 16s - loss: 0.1987 - categorical_accuracy: 0.9390
50560/60000 [========================>.....] - ETA: 16s - loss: 0.1985 - categorical_accuracy: 0.9390
50592/60000 [========================>.....] - ETA: 16s - loss: 0.1985 - categorical_accuracy: 0.9391
50624/60000 [========================>.....] - ETA: 16s - loss: 0.1984 - categorical_accuracy: 0.9391
50656/60000 [========================>.....] - ETA: 16s - loss: 0.1983 - categorical_accuracy: 0.9391
50688/60000 [========================>.....] - ETA: 16s - loss: 0.1982 - categorical_accuracy: 0.9391
50720/60000 [========================>.....] - ETA: 16s - loss: 0.1983 - categorical_accuracy: 0.9391
50752/60000 [========================>.....] - ETA: 16s - loss: 0.1982 - categorical_accuracy: 0.9391
50784/60000 [========================>.....] - ETA: 16s - loss: 0.1981 - categorical_accuracy: 0.9392
50816/60000 [========================>.....] - ETA: 16s - loss: 0.1981 - categorical_accuracy: 0.9391
50848/60000 [========================>.....] - ETA: 16s - loss: 0.1980 - categorical_accuracy: 0.9392
50880/60000 [========================>.....] - ETA: 16s - loss: 0.1979 - categorical_accuracy: 0.9392
50912/60000 [========================>.....] - ETA: 16s - loss: 0.1979 - categorical_accuracy: 0.9392
50944/60000 [========================>.....] - ETA: 16s - loss: 0.1978 - categorical_accuracy: 0.9392
50976/60000 [========================>.....] - ETA: 16s - loss: 0.1976 - categorical_accuracy: 0.9393
51008/60000 [========================>.....] - ETA: 16s - loss: 0.1975 - categorical_accuracy: 0.9393
51040/60000 [========================>.....] - ETA: 16s - loss: 0.1974 - categorical_accuracy: 0.9393
51072/60000 [========================>.....] - ETA: 15s - loss: 0.1973 - categorical_accuracy: 0.9393
51104/60000 [========================>.....] - ETA: 15s - loss: 0.1973 - categorical_accuracy: 0.9393
51136/60000 [========================>.....] - ETA: 15s - loss: 0.1973 - categorical_accuracy: 0.9394
51168/60000 [========================>.....] - ETA: 15s - loss: 0.1972 - categorical_accuracy: 0.9394
51200/60000 [========================>.....] - ETA: 15s - loss: 0.1973 - categorical_accuracy: 0.9394
51232/60000 [========================>.....] - ETA: 15s - loss: 0.1972 - categorical_accuracy: 0.9394
51264/60000 [========================>.....] - ETA: 15s - loss: 0.1971 - categorical_accuracy: 0.9395
51296/60000 [========================>.....] - ETA: 15s - loss: 0.1970 - categorical_accuracy: 0.9395
51328/60000 [========================>.....] - ETA: 15s - loss: 0.1969 - categorical_accuracy: 0.9395
51360/60000 [========================>.....] - ETA: 15s - loss: 0.1968 - categorical_accuracy: 0.9395
51392/60000 [========================>.....] - ETA: 15s - loss: 0.1968 - categorical_accuracy: 0.9395
51424/60000 [========================>.....] - ETA: 15s - loss: 0.1967 - categorical_accuracy: 0.9396
51456/60000 [========================>.....] - ETA: 15s - loss: 0.1967 - categorical_accuracy: 0.9396
51488/60000 [========================>.....] - ETA: 15s - loss: 0.1967 - categorical_accuracy: 0.9396
51520/60000 [========================>.....] - ETA: 15s - loss: 0.1966 - categorical_accuracy: 0.9396
51552/60000 [========================>.....] - ETA: 15s - loss: 0.1965 - categorical_accuracy: 0.9396
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1965 - categorical_accuracy: 0.9397
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1964 - categorical_accuracy: 0.9397
51648/60000 [========================>.....] - ETA: 14s - loss: 0.1963 - categorical_accuracy: 0.9397
51680/60000 [========================>.....] - ETA: 14s - loss: 0.1963 - categorical_accuracy: 0.9397
51712/60000 [========================>.....] - ETA: 14s - loss: 0.1962 - categorical_accuracy: 0.9397
51744/60000 [========================>.....] - ETA: 14s - loss: 0.1961 - categorical_accuracy: 0.9398
51776/60000 [========================>.....] - ETA: 14s - loss: 0.1963 - categorical_accuracy: 0.9397
51808/60000 [========================>.....] - ETA: 14s - loss: 0.1963 - categorical_accuracy: 0.9398
51840/60000 [========================>.....] - ETA: 14s - loss: 0.1962 - categorical_accuracy: 0.9398
51872/60000 [========================>.....] - ETA: 14s - loss: 0.1961 - categorical_accuracy: 0.9398
51904/60000 [========================>.....] - ETA: 14s - loss: 0.1961 - categorical_accuracy: 0.9398
51936/60000 [========================>.....] - ETA: 14s - loss: 0.1960 - categorical_accuracy: 0.9398
51968/60000 [========================>.....] - ETA: 14s - loss: 0.1959 - categorical_accuracy: 0.9399
52000/60000 [=========================>....] - ETA: 14s - loss: 0.1958 - categorical_accuracy: 0.9399
52032/60000 [=========================>....] - ETA: 14s - loss: 0.1957 - categorical_accuracy: 0.9399
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1957 - categorical_accuracy: 0.9399
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1956 - categorical_accuracy: 0.9400
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1955 - categorical_accuracy: 0.9400
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1954 - categorical_accuracy: 0.9400
52192/60000 [=========================>....] - ETA: 13s - loss: 0.1953 - categorical_accuracy: 0.9400
52224/60000 [=========================>....] - ETA: 13s - loss: 0.1952 - categorical_accuracy: 0.9401
52256/60000 [=========================>....] - ETA: 13s - loss: 0.1952 - categorical_accuracy: 0.9401
52288/60000 [=========================>....] - ETA: 13s - loss: 0.1951 - categorical_accuracy: 0.9401
52320/60000 [=========================>....] - ETA: 13s - loss: 0.1951 - categorical_accuracy: 0.9401
52352/60000 [=========================>....] - ETA: 13s - loss: 0.1950 - categorical_accuracy: 0.9402
52384/60000 [=========================>....] - ETA: 13s - loss: 0.1949 - categorical_accuracy: 0.9402
52416/60000 [=========================>....] - ETA: 13s - loss: 0.1948 - categorical_accuracy: 0.9402
52448/60000 [=========================>....] - ETA: 13s - loss: 0.1947 - categorical_accuracy: 0.9403
52480/60000 [=========================>....] - ETA: 13s - loss: 0.1946 - categorical_accuracy: 0.9403
52512/60000 [=========================>....] - ETA: 13s - loss: 0.1945 - categorical_accuracy: 0.9403
52544/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9404
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9404
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9404
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9404
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9404
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9404
52736/60000 [=========================>....] - ETA: 12s - loss: 0.1941 - categorical_accuracy: 0.9405
52768/60000 [=========================>....] - ETA: 12s - loss: 0.1941 - categorical_accuracy: 0.9405
52800/60000 [=========================>....] - ETA: 12s - loss: 0.1941 - categorical_accuracy: 0.9405
52832/60000 [=========================>....] - ETA: 12s - loss: 0.1940 - categorical_accuracy: 0.9405
52864/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9405
52896/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9406
52928/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9406
52960/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9406
52992/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9406
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1935 - categorical_accuracy: 0.9406
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9407
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1933 - categorical_accuracy: 0.9407
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1933 - categorical_accuracy: 0.9407
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1932 - categorical_accuracy: 0.9407
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1930 - categorical_accuracy: 0.9408
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1929 - categorical_accuracy: 0.9408
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9407
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1933 - categorical_accuracy: 0.9408
53312/60000 [=========================>....] - ETA: 11s - loss: 0.1934 - categorical_accuracy: 0.9408
53344/60000 [=========================>....] - ETA: 11s - loss: 0.1932 - categorical_accuracy: 0.9408
53376/60000 [=========================>....] - ETA: 11s - loss: 0.1932 - categorical_accuracy: 0.9408
53408/60000 [=========================>....] - ETA: 11s - loss: 0.1931 - categorical_accuracy: 0.9408
53440/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9409
53472/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9409
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1929 - categorical_accuracy: 0.9409
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1929 - categorical_accuracy: 0.9409
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1929 - categorical_accuracy: 0.9409
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1929 - categorical_accuracy: 0.9409
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1928 - categorical_accuracy: 0.9409
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9410
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9410
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1928 - categorical_accuracy: 0.9410
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9410
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9410
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9410
53856/60000 [=========================>....] - ETA: 10s - loss: 0.1925 - categorical_accuracy: 0.9410
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1924 - categorical_accuracy: 0.9411
53920/60000 [=========================>....] - ETA: 10s - loss: 0.1923 - categorical_accuracy: 0.9411
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1922 - categorical_accuracy: 0.9411
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1921 - categorical_accuracy: 0.9411
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1920 - categorical_accuracy: 0.9412
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1919 - categorical_accuracy: 0.9412
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1919 - categorical_accuracy: 0.9412
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1918 - categorical_accuracy: 0.9412
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1917 - categorical_accuracy: 0.9412
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1916 - categorical_accuracy: 0.9413
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9413
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9413
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1914 - categorical_accuracy: 0.9414
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1913 - categorical_accuracy: 0.9414
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1912 - categorical_accuracy: 0.9414
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1911 - categorical_accuracy: 0.9415
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1910 - categorical_accuracy: 0.9415
54432/60000 [==========================>...] - ETA: 9s - loss: 0.1909 - categorical_accuracy: 0.9415 
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1908 - categorical_accuracy: 0.9415
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1908 - categorical_accuracy: 0.9415
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9416
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9416
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9416
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1908 - categorical_accuracy: 0.9416
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9416
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9416
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1906 - categorical_accuracy: 0.9417
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9417
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9417
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9417
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1904 - categorical_accuracy: 0.9417
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1903 - categorical_accuracy: 0.9418
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1902 - categorical_accuracy: 0.9418
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1902 - categorical_accuracy: 0.9418
54976/60000 [==========================>...] - ETA: 8s - loss: 0.1902 - categorical_accuracy: 0.9418
55008/60000 [==========================>...] - ETA: 8s - loss: 0.1901 - categorical_accuracy: 0.9418
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1900 - categorical_accuracy: 0.9419
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1902 - categorical_accuracy: 0.9418
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1901 - categorical_accuracy: 0.9419
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1900 - categorical_accuracy: 0.9419
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1900 - categorical_accuracy: 0.9419
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1900 - categorical_accuracy: 0.9419
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1900 - categorical_accuracy: 0.9419
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1899 - categorical_accuracy: 0.9420
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1899 - categorical_accuracy: 0.9420
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1901 - categorical_accuracy: 0.9419
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1900 - categorical_accuracy: 0.9419
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1900 - categorical_accuracy: 0.9420
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1899 - categorical_accuracy: 0.9420
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1899 - categorical_accuracy: 0.9419
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1899 - categorical_accuracy: 0.9420
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1898 - categorical_accuracy: 0.9420
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1897 - categorical_accuracy: 0.9420
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1897 - categorical_accuracy: 0.9420
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1897 - categorical_accuracy: 0.9420
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1896 - categorical_accuracy: 0.9420
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1896 - categorical_accuracy: 0.9421
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1895 - categorical_accuracy: 0.9421
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1894 - categorical_accuracy: 0.9421
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1894 - categorical_accuracy: 0.9421
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1893 - categorical_accuracy: 0.9421
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1893 - categorical_accuracy: 0.9422
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1892 - categorical_accuracy: 0.9422
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1891 - categorical_accuracy: 0.9422
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1891 - categorical_accuracy: 0.9422
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1891 - categorical_accuracy: 0.9422
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1890 - categorical_accuracy: 0.9422
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1891 - categorical_accuracy: 0.9422
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1890 - categorical_accuracy: 0.9422
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1889 - categorical_accuracy: 0.9422
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1888 - categorical_accuracy: 0.9423
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1888 - categorical_accuracy: 0.9423
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1887 - categorical_accuracy: 0.9423
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1887 - categorical_accuracy: 0.9423
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1886 - categorical_accuracy: 0.9423
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1885 - categorical_accuracy: 0.9424
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1885 - categorical_accuracy: 0.9424
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1886 - categorical_accuracy: 0.9424
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1885 - categorical_accuracy: 0.9424
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1884 - categorical_accuracy: 0.9424
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1883 - categorical_accuracy: 0.9424
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1883 - categorical_accuracy: 0.9424
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1882 - categorical_accuracy: 0.9425
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1881 - categorical_accuracy: 0.9425
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1880 - categorical_accuracy: 0.9425
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1880 - categorical_accuracy: 0.9425
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1879 - categorical_accuracy: 0.9425
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1879 - categorical_accuracy: 0.9425
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1878 - categorical_accuracy: 0.9426
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1877 - categorical_accuracy: 0.9426
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1876 - categorical_accuracy: 0.9426
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1875 - categorical_accuracy: 0.9427
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1875 - categorical_accuracy: 0.9427
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1875 - categorical_accuracy: 0.9427
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1874 - categorical_accuracy: 0.9427
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1873 - categorical_accuracy: 0.9427
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1872 - categorical_accuracy: 0.9427
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1873 - categorical_accuracy: 0.9427
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1872 - categorical_accuracy: 0.9428
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1871 - categorical_accuracy: 0.9428
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1870 - categorical_accuracy: 0.9428
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1869 - categorical_accuracy: 0.9428
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1869 - categorical_accuracy: 0.9429
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1868 - categorical_accuracy: 0.9429
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1868 - categorical_accuracy: 0.9429
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9429
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9429
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1866 - categorical_accuracy: 0.9429
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1866 - categorical_accuracy: 0.9429
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1865 - categorical_accuracy: 0.9430
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1865 - categorical_accuracy: 0.9430
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9430
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9430
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1866 - categorical_accuracy: 0.9430
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1865 - categorical_accuracy: 0.9430
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1865 - categorical_accuracy: 0.9430
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1865 - categorical_accuracy: 0.9431
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1864 - categorical_accuracy: 0.9431
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1863 - categorical_accuracy: 0.9431
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1862 - categorical_accuracy: 0.9431
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1861 - categorical_accuracy: 0.9431
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1862 - categorical_accuracy: 0.9431
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1861 - categorical_accuracy: 0.9431
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1861 - categorical_accuracy: 0.9431
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1860 - categorical_accuracy: 0.9432
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1860 - categorical_accuracy: 0.9432
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1859 - categorical_accuracy: 0.9432
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1858 - categorical_accuracy: 0.9432
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1858 - categorical_accuracy: 0.9432
58016/60000 [============================>.] - ETA: 3s - loss: 0.1857 - categorical_accuracy: 0.9432
58048/60000 [============================>.] - ETA: 3s - loss: 0.1856 - categorical_accuracy: 0.9433
58080/60000 [============================>.] - ETA: 3s - loss: 0.1856 - categorical_accuracy: 0.9433
58112/60000 [============================>.] - ETA: 3s - loss: 0.1856 - categorical_accuracy: 0.9433
58144/60000 [============================>.] - ETA: 3s - loss: 0.1855 - categorical_accuracy: 0.9433
58176/60000 [============================>.] - ETA: 3s - loss: 0.1855 - categorical_accuracy: 0.9433
58208/60000 [============================>.] - ETA: 3s - loss: 0.1854 - categorical_accuracy: 0.9433
58240/60000 [============================>.] - ETA: 3s - loss: 0.1853 - categorical_accuracy: 0.9433
58272/60000 [============================>.] - ETA: 3s - loss: 0.1853 - categorical_accuracy: 0.9434
58304/60000 [============================>.] - ETA: 3s - loss: 0.1852 - categorical_accuracy: 0.9434
58336/60000 [============================>.] - ETA: 2s - loss: 0.1852 - categorical_accuracy: 0.9434
58368/60000 [============================>.] - ETA: 2s - loss: 0.1852 - categorical_accuracy: 0.9434
58400/60000 [============================>.] - ETA: 2s - loss: 0.1852 - categorical_accuracy: 0.9434
58432/60000 [============================>.] - ETA: 2s - loss: 0.1851 - categorical_accuracy: 0.9434
58464/60000 [============================>.] - ETA: 2s - loss: 0.1850 - categorical_accuracy: 0.9434
58496/60000 [============================>.] - ETA: 2s - loss: 0.1850 - categorical_accuracy: 0.9434
58528/60000 [============================>.] - ETA: 2s - loss: 0.1849 - categorical_accuracy: 0.9434
58560/60000 [============================>.] - ETA: 2s - loss: 0.1850 - categorical_accuracy: 0.9434
58592/60000 [============================>.] - ETA: 2s - loss: 0.1849 - categorical_accuracy: 0.9434
58624/60000 [============================>.] - ETA: 2s - loss: 0.1848 - categorical_accuracy: 0.9435
58656/60000 [============================>.] - ETA: 2s - loss: 0.1848 - categorical_accuracy: 0.9435
58688/60000 [============================>.] - ETA: 2s - loss: 0.1847 - categorical_accuracy: 0.9435
58720/60000 [============================>.] - ETA: 2s - loss: 0.1847 - categorical_accuracy: 0.9435
58752/60000 [============================>.] - ETA: 2s - loss: 0.1846 - categorical_accuracy: 0.9436
58784/60000 [============================>.] - ETA: 2s - loss: 0.1845 - categorical_accuracy: 0.9436
58816/60000 [============================>.] - ETA: 2s - loss: 0.1844 - categorical_accuracy: 0.9436
58848/60000 [============================>.] - ETA: 2s - loss: 0.1844 - categorical_accuracy: 0.9436
58880/60000 [============================>.] - ETA: 2s - loss: 0.1844 - categorical_accuracy: 0.9436
58912/60000 [============================>.] - ETA: 1s - loss: 0.1843 - categorical_accuracy: 0.9437
58944/60000 [============================>.] - ETA: 1s - loss: 0.1842 - categorical_accuracy: 0.9437
58976/60000 [============================>.] - ETA: 1s - loss: 0.1841 - categorical_accuracy: 0.9437
59008/60000 [============================>.] - ETA: 1s - loss: 0.1841 - categorical_accuracy: 0.9438
59040/60000 [============================>.] - ETA: 1s - loss: 0.1840 - categorical_accuracy: 0.9438
59072/60000 [============================>.] - ETA: 1s - loss: 0.1839 - categorical_accuracy: 0.9438
59104/60000 [============================>.] - ETA: 1s - loss: 0.1839 - categorical_accuracy: 0.9438
59136/60000 [============================>.] - ETA: 1s - loss: 0.1838 - categorical_accuracy: 0.9438
59168/60000 [============================>.] - ETA: 1s - loss: 0.1837 - categorical_accuracy: 0.9438
59200/60000 [============================>.] - ETA: 1s - loss: 0.1836 - categorical_accuracy: 0.9439
59232/60000 [============================>.] - ETA: 1s - loss: 0.1836 - categorical_accuracy: 0.9439
59264/60000 [============================>.] - ETA: 1s - loss: 0.1835 - categorical_accuracy: 0.9439
59296/60000 [============================>.] - ETA: 1s - loss: 0.1836 - categorical_accuracy: 0.9439
59328/60000 [============================>.] - ETA: 1s - loss: 0.1835 - categorical_accuracy: 0.9439
59360/60000 [============================>.] - ETA: 1s - loss: 0.1834 - categorical_accuracy: 0.9440
59392/60000 [============================>.] - ETA: 1s - loss: 0.1833 - categorical_accuracy: 0.9440
59424/60000 [============================>.] - ETA: 1s - loss: 0.1833 - categorical_accuracy: 0.9440
59456/60000 [============================>.] - ETA: 0s - loss: 0.1832 - categorical_accuracy: 0.9440
59488/60000 [============================>.] - ETA: 0s - loss: 0.1833 - categorical_accuracy: 0.9440
59520/60000 [============================>.] - ETA: 0s - loss: 0.1833 - categorical_accuracy: 0.9440
59552/60000 [============================>.] - ETA: 0s - loss: 0.1832 - categorical_accuracy: 0.9440
59584/60000 [============================>.] - ETA: 0s - loss: 0.1832 - categorical_accuracy: 0.9440
59616/60000 [============================>.] - ETA: 0s - loss: 0.1831 - categorical_accuracy: 0.9440
59648/60000 [============================>.] - ETA: 0s - loss: 0.1830 - categorical_accuracy: 0.9441
59680/60000 [============================>.] - ETA: 0s - loss: 0.1830 - categorical_accuracy: 0.9441
59712/60000 [============================>.] - ETA: 0s - loss: 0.1831 - categorical_accuracy: 0.9440
59744/60000 [============================>.] - ETA: 0s - loss: 0.1832 - categorical_accuracy: 0.9440
59776/60000 [============================>.] - ETA: 0s - loss: 0.1831 - categorical_accuracy: 0.9441
59808/60000 [============================>.] - ETA: 0s - loss: 0.1830 - categorical_accuracy: 0.9441
59840/60000 [============================>.] - ETA: 0s - loss: 0.1831 - categorical_accuracy: 0.9440
59872/60000 [============================>.] - ETA: 0s - loss: 0.1830 - categorical_accuracy: 0.9440
59904/60000 [============================>.] - ETA: 0s - loss: 0.1830 - categorical_accuracy: 0.9441
59936/60000 [============================>.] - ETA: 0s - loss: 0.1829 - categorical_accuracy: 0.9441
59968/60000 [============================>.] - ETA: 0s - loss: 0.1828 - categorical_accuracy: 0.9441
60000/60000 [==============================] - 111s 2ms/step - loss: 0.1827 - categorical_accuracy: 0.9441 - val_loss: 0.0451 - val_categorical_accuracy: 0.9855

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  672/10000 [=>............................] - ETA: 3s
  832/10000 [=>............................] - ETA: 3s
  992/10000 [=>............................] - ETA: 3s
 1120/10000 [==>...........................] - ETA: 3s
 1280/10000 [==>...........................] - ETA: 3s
 1408/10000 [===>..........................] - ETA: 3s
 1536/10000 [===>..........................] - ETA: 3s
 1696/10000 [====>.........................] - ETA: 3s
 1856/10000 [====>.........................] - ETA: 3s
 2016/10000 [=====>........................] - ETA: 3s
 2176/10000 [=====>........................] - ETA: 3s
 2336/10000 [======>.......................] - ETA: 2s
 2496/10000 [======>.......................] - ETA: 2s
 2656/10000 [======>.......................] - ETA: 2s
 2816/10000 [=======>......................] - ETA: 2s
 2976/10000 [=======>......................] - ETA: 2s
 3136/10000 [========>.....................] - ETA: 2s
 3296/10000 [========>.....................] - ETA: 2s
 3456/10000 [=========>....................] - ETA: 2s
 3616/10000 [=========>....................] - ETA: 2s
 3776/10000 [==========>...................] - ETA: 2s
 3936/10000 [==========>...................] - ETA: 2s
 4096/10000 [===========>..................] - ETA: 2s
 4256/10000 [===========>..................] - ETA: 2s
 4416/10000 [============>.................] - ETA: 2s
 4576/10000 [============>.................] - ETA: 1s
 4736/10000 [=============>................] - ETA: 1s
 4896/10000 [=============>................] - ETA: 1s
 5056/10000 [==============>...............] - ETA: 1s
 5216/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5536/10000 [===============>..............] - ETA: 1s
 5696/10000 [================>.............] - ETA: 1s
 5856/10000 [================>.............] - ETA: 1s
 6016/10000 [=================>............] - ETA: 1s
 6176/10000 [=================>............] - ETA: 1s
 6336/10000 [==================>...........] - ETA: 1s
 6496/10000 [==================>...........] - ETA: 1s
 6656/10000 [==================>...........] - ETA: 1s
 6816/10000 [===================>..........] - ETA: 1s
 6976/10000 [===================>..........] - ETA: 1s
 7136/10000 [====================>.........] - ETA: 1s
 7296/10000 [====================>.........] - ETA: 0s
 7456/10000 [=====================>........] - ETA: 0s
 7616/10000 [=====================>........] - ETA: 0s
 7776/10000 [======================>.......] - ETA: 0s
 7936/10000 [======================>.......] - ETA: 0s
 8096/10000 [=======================>......] - ETA: 0s
 8256/10000 [=======================>......] - ETA: 0s
 8416/10000 [========================>.....] - ETA: 0s
 8576/10000 [========================>.....] - ETA: 0s
 8736/10000 [=========================>....] - ETA: 0s
 8896/10000 [=========================>....] - ETA: 0s
 9056/10000 [==========================>...] - ETA: 0s
 9216/10000 [==========================>...] - ETA: 0s
 9376/10000 [===========================>..] - ETA: 0s
 9536/10000 [===========================>..] - ETA: 0s
 9696/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 349us/step
[[3.73432485e-08 3.00969951e-08 4.99398311e-06 ... 9.99970913e-01
  1.01544721e-07 1.24235221e-05]
 [2.72625584e-05 2.13065214e-05 9.99922872e-01 ... 1.25676065e-08
  5.54412054e-06 5.97609739e-09]
 [1.52980795e-06 9.99850392e-01 1.88725389e-05 ... 3.05385001e-05
  2.82831697e-05 3.24300458e-06]
 ...
 [2.98514884e-08 1.67587802e-06 9.86455717e-09 ... 7.67560869e-06
  6.04480874e-06 9.20131424e-05]
 [2.06582922e-06 1.53608653e-07 1.14965468e-08 ... 2.33720954e-07
  6.13423821e-04 2.21582104e-06]
 [1.22330557e-05 1.55300404e-06 4.78283000e-05 ... 4.41827375e-08
  6.85456780e-06 1.12423301e-07]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04512540909376694, 'accuracy_test:': 0.9854999780654907}

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
   a28fd8e..94f70d3  master     -> origin/master
Updating a28fd8e..94f70d3
Fast-forward
 .../20200519/list_log_pullrequest_20200519.md      |   2 +-
 error_list/20200519/list_log_testall_20200519.md   | 434 +++++++++++++++++++++
 2 files changed, 435 insertions(+), 1 deletion(-)
[master abbb2b5] ml_store
 1 file changed, 2044 insertions(+)
To github.com:arita37/mlmodels_store.git
   94f70d3..abbb2b5  master -> master





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
{'loss': 0.42722031846642494, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-19 20:29:09.949053: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
