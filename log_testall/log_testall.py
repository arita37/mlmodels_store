
  test_all /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_all', folder=None, log_file=None, save_folder='ztest/') 

  ml_test --do test_all 





 ************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '51b64e342c7b2661e79b8abaa33db92672ae95c7', 'workflow': 'test_all'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_all

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/51b64e342c7b2661e79b8abaa33db92672ae95c7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/51b64e342c7b2661e79b8abaa33db92672ae95c7

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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
Already up to date.
[master be70fff] ml_store
 2 files changed, 64 insertions(+), 9931 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   2409ba1..be70fff  master -> master





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
[master e682ea3] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   be70fff..e682ea3  master -> master





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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-21 08:12:00.144111: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-21 08:12:00.149781: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-21 08:12:00.150004: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562ca4f23e40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 08:12:00.150023: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 183
Trainable params: 183
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2499 - binary_crossentropy: 0.6928 - val_loss: 0.2508 - val_binary_crossentropy: 0.6948

  #### metrics   #################################################### 
{'MSE': 0.25004193805778885}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
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
Total params: 183
Trainable params: 183
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
100/500 [=====>........................] - ETA: 1s - loss: 0.2580 - binary_crossentropy: 0.7093500/500 [==============================] - 1s 2ms/sample - loss: 0.2599 - binary_crossentropy: 0.7134 - val_loss: 0.2508 - val_binary_crossentropy: 0.6949

  #### metrics   #################################################### 
{'MSE': 0.2547399383051787}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 687
Trainable params: 687
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2499 - binary_crossentropy: 0.6929 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.24973141686913955}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
Total params: 687
Trainable params: 687
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 443
Trainable params: 443
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3382 - binary_crossentropy: 1.9319500/500 [==============================] - 2s 3ms/sample - loss: 0.3224 - binary_crossentropy: 2.0764 - val_loss: 0.3016 - val_binary_crossentropy: 1.7363

  #### metrics   #################################################### 
{'MSE': 0.30927069124727596}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 443
Trainable params: 443
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
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
Total params: 108
Trainable params: 108
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2549 - binary_crossentropy: 0.8355500/500 [==============================] - 2s 4ms/sample - loss: 0.2601 - binary_crossentropy: 0.8703 - val_loss: 0.2525 - val_binary_crossentropy: 0.8024

  #### metrics   #################################################### 
{'MSE': 0.2562511739845571}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_12 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_max[0][0]               
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
Total params: 108
Trainable params: 108
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-21 08:13:26.496990: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:26.499266: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:26.505120: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 08:13:26.516504: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 08:13:26.518418: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 08:13:26.520139: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:26.521669: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2487 - val_binary_crossentropy: 0.6906
2020-05-21 08:13:27.998241: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:27.999953: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:28.004207: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 08:13:28.013287: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 08:13:28.014957: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 08:13:28.016395: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:28.017672: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2481573287126686}

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
2020-05-21 08:13:53.659131: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:53.660750: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:53.664502: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 08:13:53.671563: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 08:13:53.672689: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 08:13:53.673689: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:53.674598: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2510 - val_binary_crossentropy: 0.6952
2020-05-21 08:13:55.323932: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:55.325125: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:55.327748: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 08:13:55.332947: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 08:13:55.333856: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 08:13:55.334718: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:13:55.335479: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2512783386566613}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-21 08:14:31.717035: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:14:31.722535: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:14:31.739022: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 08:14:31.768446: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 08:14:31.773456: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 08:14:31.778133: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:14:31.783071: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 6s 6s/sample - loss: 0.4428 - binary_crossentropy: 1.0949 - val_loss: 0.2700 - val_binary_crossentropy: 0.7348
2020-05-21 08:14:34.279533: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:14:34.285256: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:14:34.299947: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 08:14:34.328447: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 08:14:34.333365: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 08:14:34.337710: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 08:14:34.342058: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.3173913631396503}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 695
Trainable params: 695
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.3255 - binary_crossentropy: 0.8833500/500 [==============================] - 5s 9ms/sample - loss: 0.3311 - binary_crossentropy: 0.9056 - val_loss: 0.2940 - val_binary_crossentropy: 0.8144

  #### metrics   #################################################### 
{'MSE': 0.3110304235550422}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
Total params: 272
Trainable params: 272
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2551 - binary_crossentropy: 0.7036500/500 [==============================] - 5s 10ms/sample - loss: 0.2745 - binary_crossentropy: 0.9033 - val_loss: 0.2735 - val_binary_crossentropy: 0.7953

  #### metrics   #################################################### 
{'MSE': 0.2718433759825442}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 2)         18          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_0 (Em (None, 1, 2)         2           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         14          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         16          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         16          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
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
Total params: 1,864
Trainable params: 1,864
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2583 - binary_crossentropy: 0.8421500/500 [==============================] - 5s 10ms/sample - loss: 0.2568 - binary_crossentropy: 0.7585 - val_loss: 0.2516 - val_binary_crossentropy: 0.7212

  #### metrics   #################################################### 
{'MSE': 0.2524207916216176}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         5           sequence_max[0][0]               
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
Total params: 1,864
Trainable params: 1,864
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
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         1           regionsequence_mean[0][0]        
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
Total params: 152
Trainable params: 152
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2815 - binary_crossentropy: 0.7606500/500 [==============================] - 6s 13ms/sample - loss: 0.2677 - binary_crossentropy: 0.7311 - val_loss: 0.2667 - val_binary_crossentropy: 0.7293

  #### metrics   #################################################### 
{'MSE': 0.2660385682831168}

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
region_10sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         1           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         9           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         1           regionsequence_mean[0][0]        
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
Total params: 152
Trainable params: 152
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
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
Total params: 1,352
Trainable params: 1,352
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.5100 - binary_crossentropy: 7.8667500/500 [==============================] - 6s 13ms/sample - loss: 0.4860 - binary_crossentropy: 7.4965 - val_loss: 0.5320 - val_binary_crossentropy: 8.2061

  #### metrics   #################################################### 
{'MSE': 0.509}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_max[0][0]               
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
Total params: 1,352
Trainable params: 1,352
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
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
Total params: 3,052
Trainable params: 2,972
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 7s 14ms/sample - loss: 0.2539 - binary_crossentropy: 0.7010 - val_loss: 0.2498 - val_binary_crossentropy: 0.6927

  #### metrics   #################################################### 
{'MSE': 0.25000468337806203}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         4           hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         36          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         4           hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         36          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 2, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 7, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 6, 4)         16          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           hash_10[0][0]                    
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
   e682ea3..e26bba1  master     -> origin/master
Updating e682ea3..e26bba1
Fast-forward
 error_list/20200521/list_log_testall_20200521.md   | 742 +--------------------
 ...-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py | 627 +++++++++++++++++
 2 files changed, 631 insertions(+), 738 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-21-08-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py
[master a681b34] ml_store
 1 file changed, 4954 insertions(+)
To github.com:arita37/mlmodels_store.git
   e26bba1..a681b34  master -> master





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
[master 1eb11cf] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   a681b34..1eb11cf  master -> master





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
[master fbff9a3] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   1eb11cf..fbff9a3  master -> master





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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
Already up to date.
[master c817e94] ml_store
 1 file changed, 37 insertions(+)
To github.com:arita37/mlmodels_store.git
   fbff9a3..c817e94  master -> master





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

2020-05-21 08:24:16.968283: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-21 08:24:16.974097: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-21 08:24:16.974350: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556586a3ea60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 08:24:16.974370: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 5s - loss: 1.3852
256/354 [====================>.........] - ETA: 2s - loss: 1.2481
354/354 [==============================] - 10s 28ms/step - loss: 1.3775 - val_loss: 1.9294

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
[master 342f5ac] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   c817e94..342f5ac  master -> master





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
[master 2f85723] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   342f5ac..2f85723  master -> master





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
[master a875e63] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   2f85723..a875e63  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2252800/17464789 [==>...........................] - ETA: 0s
 8970240/17464789 [==============>...............] - ETA: 0s
16408576/17464789 [===========================>..] - ETA: 0s
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
2020-05-21 08:25:14.352713: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-21 08:25:14.357378: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-21 08:25:14.357550: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56393a2ca680 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 08:25:14.357566: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.7126 - accuracy: 0.4970
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7433 - accuracy: 0.4950 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.8046 - accuracy: 0.4910
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.8430 - accuracy: 0.4885
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7924 - accuracy: 0.4918
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.8174 - accuracy: 0.4902
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7389 - accuracy: 0.4953
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7356 - accuracy: 0.4955
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6990 - accuracy: 0.4979
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6881 - accuracy: 0.4986
11000/25000 [============>.................] - ETA: 3s - loss: 7.7015 - accuracy: 0.4977
12000/25000 [=============>................] - ETA: 3s - loss: 7.7062 - accuracy: 0.4974
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7209 - accuracy: 0.4965
14000/25000 [===============>..............] - ETA: 2s - loss: 7.7104 - accuracy: 0.4971
15000/25000 [=================>............] - ETA: 2s - loss: 7.6973 - accuracy: 0.4980
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6848 - accuracy: 0.4988
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6874 - accuracy: 0.4986
18000/25000 [====================>.........] - ETA: 1s - loss: 7.7058 - accuracy: 0.4974
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6924 - accuracy: 0.4983
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6950 - accuracy: 0.4981
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6805 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6638 - accuracy: 0.5002
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6600 - accuracy: 0.5004
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
25000/25000 [==============================] - 7s 289us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fd69aa4f400>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fd6c016b9b0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 8.0500 - accuracy: 0.4750
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9580 - accuracy: 0.4810 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.7740 - accuracy: 0.4930
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6513 - accuracy: 0.5010
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5961 - accuracy: 0.5046
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6027 - accuracy: 0.5042
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6294 - accuracy: 0.5024
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6302 - accuracy: 0.5024
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6445 - accuracy: 0.5014
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6436 - accuracy: 0.5015
11000/25000 [============>.................] - ETA: 3s - loss: 7.6847 - accuracy: 0.4988
12000/25000 [=============>................] - ETA: 3s - loss: 7.6628 - accuracy: 0.5002
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6454 - accuracy: 0.5014
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6436 - accuracy: 0.5015
15000/25000 [=================>............] - ETA: 2s - loss: 7.6595 - accuracy: 0.5005
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6398 - accuracy: 0.5017
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6405 - accuracy: 0.5017
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6598 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6618 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6505 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6571 - accuracy: 0.5006
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6680 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6685 - accuracy: 0.4999
25000/25000 [==============================] - 7s 289us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 8.1726 - accuracy: 0.4670
 2000/25000 [=>............................] - ETA: 8s - loss: 7.7970 - accuracy: 0.4915 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6053 - accuracy: 0.5040
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5930 - accuracy: 0.5048
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6206 - accuracy: 0.5030
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6272 - accuracy: 0.5026
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6110 - accuracy: 0.5036
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6326 - accuracy: 0.5022
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6651 - accuracy: 0.5001
11000/25000 [============>.................] - ETA: 3s - loss: 7.6820 - accuracy: 0.4990
12000/25000 [=============>................] - ETA: 3s - loss: 7.6768 - accuracy: 0.4993
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6595 - accuracy: 0.5005
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6513 - accuracy: 0.5010
15000/25000 [=================>............] - ETA: 2s - loss: 7.6503 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6455 - accuracy: 0.5014
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6522 - accuracy: 0.5009
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6615 - accuracy: 0.5003
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6368 - accuracy: 0.5019
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6467 - accuracy: 0.5013
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6608 - accuracy: 0.5004
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6645 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6620 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6666 - accuracy: 0.5000
25000/25000 [==============================] - 7s 297us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   a875e63..cf4e0dd  master     -> origin/master
Updating a875e63..cf4e0dd
Fast-forward
 error_list/20200521/list_log_testall_20200521.md | 103 +++++++++++++++++++++++
 1 file changed, 103 insertions(+)
[master 6bba9e4] ml_store
 1 file changed, 324 insertions(+)
To github.com:arita37/mlmodels_store.git
   cf4e0dd..6bba9e4  master -> master





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
[master c4cba42] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   6bba9e4..c4cba42  master -> master





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
 3203072/11490434 [=======>......................] - ETA: 0s
11116544/11490434 [============================>.] - ETA: 0s
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

   32/60000 [..............................] - ETA: 8:02 - loss: 2.3459 - categorical_accuracy: 0.0625
   64/60000 [..............................] - ETA: 4:48 - loss: 2.3066 - categorical_accuracy: 0.1406
   96/60000 [..............................] - ETA: 3:44 - loss: 2.2542 - categorical_accuracy: 0.1667
  160/60000 [..............................] - ETA: 2:51 - loss: 2.1916 - categorical_accuracy: 0.2375
  224/60000 [..............................] - ETA: 2:28 - loss: 2.0896 - categorical_accuracy: 0.2768
  256/60000 [..............................] - ETA: 2:22 - loss: 2.0650 - categorical_accuracy: 0.2930
  320/60000 [..............................] - ETA: 2:12 - loss: 1.9888 - categorical_accuracy: 0.3219
  384/60000 [..............................] - ETA: 2:04 - loss: 1.9365 - categorical_accuracy: 0.3307
  448/60000 [..............................] - ETA: 1:58 - loss: 1.8924 - categorical_accuracy: 0.3527
  512/60000 [..............................] - ETA: 1:55 - loss: 1.8368 - categorical_accuracy: 0.3770
  544/60000 [..............................] - ETA: 1:54 - loss: 1.8105 - categorical_accuracy: 0.3842
  608/60000 [..............................] - ETA: 1:51 - loss: 1.7456 - categorical_accuracy: 0.4046
  672/60000 [..............................] - ETA: 1:48 - loss: 1.7117 - categorical_accuracy: 0.4062
  736/60000 [..............................] - ETA: 1:46 - loss: 1.6747 - categorical_accuracy: 0.4226
  800/60000 [..............................] - ETA: 1:45 - loss: 1.6100 - categorical_accuracy: 0.4462
  864/60000 [..............................] - ETA: 1:43 - loss: 1.5769 - categorical_accuracy: 0.4572
  896/60000 [..............................] - ETA: 1:43 - loss: 1.5607 - categorical_accuracy: 0.4632
  960/60000 [..............................] - ETA: 1:42 - loss: 1.5330 - categorical_accuracy: 0.4760
 1024/60000 [..............................] - ETA: 1:41 - loss: 1.4865 - categorical_accuracy: 0.4971
 1088/60000 [..............................] - ETA: 1:40 - loss: 1.4550 - categorical_accuracy: 0.5028
 1152/60000 [..............................] - ETA: 1:39 - loss: 1.4198 - categorical_accuracy: 0.5156
 1216/60000 [..............................] - ETA: 1:39 - loss: 1.3954 - categorical_accuracy: 0.5238
 1280/60000 [..............................] - ETA: 1:38 - loss: 1.3681 - categorical_accuracy: 0.5336
 1344/60000 [..............................] - ETA: 1:37 - loss: 1.3425 - categorical_accuracy: 0.5417
 1408/60000 [..............................] - ETA: 1:36 - loss: 1.3163 - categorical_accuracy: 0.5497
 1472/60000 [..............................] - ETA: 1:36 - loss: 1.2874 - categorical_accuracy: 0.5605
 1504/60000 [..............................] - ETA: 1:36 - loss: 1.2829 - categorical_accuracy: 0.5638
 1568/60000 [..............................] - ETA: 1:35 - loss: 1.2559 - categorical_accuracy: 0.5746
 1632/60000 [..............................] - ETA: 1:35 - loss: 1.2314 - categorical_accuracy: 0.5833
 1664/60000 [..............................] - ETA: 1:35 - loss: 1.2174 - categorical_accuracy: 0.5871
 1728/60000 [..............................] - ETA: 1:34 - loss: 1.1955 - categorical_accuracy: 0.5966
 1792/60000 [..............................] - ETA: 1:34 - loss: 1.1726 - categorical_accuracy: 0.6038
 1856/60000 [..............................] - ETA: 1:33 - loss: 1.1549 - categorical_accuracy: 0.6110
 1920/60000 [..............................] - ETA: 1:33 - loss: 1.1332 - categorical_accuracy: 0.6193
 1984/60000 [..............................] - ETA: 1:32 - loss: 1.1146 - categorical_accuracy: 0.6255
 2016/60000 [>.............................] - ETA: 1:32 - loss: 1.1050 - categorical_accuracy: 0.6290
 2080/60000 [>.............................] - ETA: 1:32 - loss: 1.0849 - categorical_accuracy: 0.6346
 2144/60000 [>.............................] - ETA: 1:31 - loss: 1.0611 - categorical_accuracy: 0.6418
 2208/60000 [>.............................] - ETA: 1:31 - loss: 1.0457 - categorical_accuracy: 0.6458
 2272/60000 [>.............................] - ETA: 1:31 - loss: 1.0343 - categorical_accuracy: 0.6505
 2336/60000 [>.............................] - ETA: 1:31 - loss: 1.0222 - categorical_accuracy: 0.6554
 2400/60000 [>.............................] - ETA: 1:30 - loss: 1.0167 - categorical_accuracy: 0.6575
 2464/60000 [>.............................] - ETA: 1:30 - loss: 1.0037 - categorical_accuracy: 0.6631
 2528/60000 [>.............................] - ETA: 1:30 - loss: 0.9983 - categorical_accuracy: 0.6650
 2560/60000 [>.............................] - ETA: 1:30 - loss: 0.9952 - categorical_accuracy: 0.6660
 2624/60000 [>.............................] - ETA: 1:29 - loss: 0.9818 - categorical_accuracy: 0.6704
 2688/60000 [>.............................] - ETA: 1:29 - loss: 0.9680 - categorical_accuracy: 0.6760
 2752/60000 [>.............................] - ETA: 1:29 - loss: 0.9556 - categorical_accuracy: 0.6799
 2816/60000 [>.............................] - ETA: 1:29 - loss: 0.9406 - categorical_accuracy: 0.6847
 2848/60000 [>.............................] - ETA: 1:29 - loss: 0.9394 - categorical_accuracy: 0.6850
 2912/60000 [>.............................] - ETA: 1:28 - loss: 0.9301 - categorical_accuracy: 0.6892
 2976/60000 [>.............................] - ETA: 1:28 - loss: 0.9180 - categorical_accuracy: 0.6939
 3040/60000 [>.............................] - ETA: 1:28 - loss: 0.9093 - categorical_accuracy: 0.6970
 3104/60000 [>.............................] - ETA: 1:28 - loss: 0.8998 - categorical_accuracy: 0.7010
 3168/60000 [>.............................] - ETA: 1:28 - loss: 0.8920 - categorical_accuracy: 0.7042
 3232/60000 [>.............................] - ETA: 1:27 - loss: 0.8828 - categorical_accuracy: 0.7076
 3296/60000 [>.............................] - ETA: 1:27 - loss: 0.8719 - categorical_accuracy: 0.7115
 3360/60000 [>.............................] - ETA: 1:27 - loss: 0.8599 - categorical_accuracy: 0.7161
 3424/60000 [>.............................] - ETA: 1:27 - loss: 0.8494 - categorical_accuracy: 0.7190
 3488/60000 [>.............................] - ETA: 1:26 - loss: 0.8474 - categorical_accuracy: 0.7199
 3552/60000 [>.............................] - ETA: 1:26 - loss: 0.8418 - categorical_accuracy: 0.7224
 3616/60000 [>.............................] - ETA: 1:26 - loss: 0.8334 - categorical_accuracy: 0.7254
 3680/60000 [>.............................] - ETA: 1:26 - loss: 0.8242 - categorical_accuracy: 0.7293
 3744/60000 [>.............................] - ETA: 1:26 - loss: 0.8161 - categorical_accuracy: 0.7324
 3808/60000 [>.............................] - ETA: 1:26 - loss: 0.8085 - categorical_accuracy: 0.7350
 3872/60000 [>.............................] - ETA: 1:25 - loss: 0.8028 - categorical_accuracy: 0.7363
 3904/60000 [>.............................] - ETA: 1:25 - loss: 0.8037 - categorical_accuracy: 0.7364
 3968/60000 [>.............................] - ETA: 1:25 - loss: 0.7981 - categorical_accuracy: 0.7384
 4032/60000 [=>............................] - ETA: 1:25 - loss: 0.7900 - categorical_accuracy: 0.7416
 4096/60000 [=>............................] - ETA: 1:25 - loss: 0.7875 - categorical_accuracy: 0.7424
 4160/60000 [=>............................] - ETA: 1:25 - loss: 0.7800 - categorical_accuracy: 0.7447
 4224/60000 [=>............................] - ETA: 1:25 - loss: 0.7727 - categorical_accuracy: 0.7467
 4288/60000 [=>............................] - ETA: 1:25 - loss: 0.7646 - categorical_accuracy: 0.7491
 4320/60000 [=>............................] - ETA: 1:25 - loss: 0.7639 - categorical_accuracy: 0.7500
 4352/60000 [=>............................] - ETA: 1:24 - loss: 0.7628 - categorical_accuracy: 0.7505
 4416/60000 [=>............................] - ETA: 1:24 - loss: 0.7579 - categorical_accuracy: 0.7523
 4480/60000 [=>............................] - ETA: 1:24 - loss: 0.7544 - categorical_accuracy: 0.7533
 4544/60000 [=>............................] - ETA: 1:24 - loss: 0.7472 - categorical_accuracy: 0.7557
 4608/60000 [=>............................] - ETA: 1:24 - loss: 0.7405 - categorical_accuracy: 0.7582
 4672/60000 [=>............................] - ETA: 1:24 - loss: 0.7333 - categorical_accuracy: 0.7609
 4736/60000 [=>............................] - ETA: 1:23 - loss: 0.7276 - categorical_accuracy: 0.7625
 4800/60000 [=>............................] - ETA: 1:23 - loss: 0.7226 - categorical_accuracy: 0.7644
 4864/60000 [=>............................] - ETA: 1:23 - loss: 0.7164 - categorical_accuracy: 0.7667
 4928/60000 [=>............................] - ETA: 1:23 - loss: 0.7116 - categorical_accuracy: 0.7683
 4992/60000 [=>............................] - ETA: 1:23 - loss: 0.7055 - categorical_accuracy: 0.7698
 5056/60000 [=>............................] - ETA: 1:23 - loss: 0.7002 - categorical_accuracy: 0.7714
 5120/60000 [=>............................] - ETA: 1:22 - loss: 0.6942 - categorical_accuracy: 0.7736
 5184/60000 [=>............................] - ETA: 1:22 - loss: 0.6887 - categorical_accuracy: 0.7755
 5248/60000 [=>............................] - ETA: 1:22 - loss: 0.6876 - categorical_accuracy: 0.7761
 5312/60000 [=>............................] - ETA: 1:22 - loss: 0.6833 - categorical_accuracy: 0.7773
 5344/60000 [=>............................] - ETA: 1:22 - loss: 0.6808 - categorical_accuracy: 0.7779
 5408/60000 [=>............................] - ETA: 1:22 - loss: 0.6764 - categorical_accuracy: 0.7788
 5472/60000 [=>............................] - ETA: 1:22 - loss: 0.6714 - categorical_accuracy: 0.7807
 5536/60000 [=>............................] - ETA: 1:22 - loss: 0.6686 - categorical_accuracy: 0.7816
 5600/60000 [=>............................] - ETA: 1:22 - loss: 0.6634 - categorical_accuracy: 0.7836
 5664/60000 [=>............................] - ETA: 1:21 - loss: 0.6599 - categorical_accuracy: 0.7848
 5728/60000 [=>............................] - ETA: 1:21 - loss: 0.6597 - categorical_accuracy: 0.7858
 5760/60000 [=>............................] - ETA: 1:21 - loss: 0.6569 - categorical_accuracy: 0.7868
 5824/60000 [=>............................] - ETA: 1:21 - loss: 0.6522 - categorical_accuracy: 0.7886
 5856/60000 [=>............................] - ETA: 1:21 - loss: 0.6501 - categorical_accuracy: 0.7891
 5920/60000 [=>............................] - ETA: 1:21 - loss: 0.6467 - categorical_accuracy: 0.7902
 5984/60000 [=>............................] - ETA: 1:21 - loss: 0.6420 - categorical_accuracy: 0.7919
 6048/60000 [==>...........................] - ETA: 1:21 - loss: 0.6382 - categorical_accuracy: 0.7930
 6112/60000 [==>...........................] - ETA: 1:21 - loss: 0.6355 - categorical_accuracy: 0.7938
 6176/60000 [==>...........................] - ETA: 1:21 - loss: 0.6309 - categorical_accuracy: 0.7953
 6240/60000 [==>...........................] - ETA: 1:20 - loss: 0.6296 - categorical_accuracy: 0.7960
 6272/60000 [==>...........................] - ETA: 1:20 - loss: 0.6281 - categorical_accuracy: 0.7966
 6336/60000 [==>...........................] - ETA: 1:20 - loss: 0.6246 - categorical_accuracy: 0.7978
 6400/60000 [==>...........................] - ETA: 1:20 - loss: 0.6225 - categorical_accuracy: 0.7987
 6432/60000 [==>...........................] - ETA: 1:20 - loss: 0.6204 - categorical_accuracy: 0.7996
 6496/60000 [==>...........................] - ETA: 1:20 - loss: 0.6181 - categorical_accuracy: 0.8006
 6560/60000 [==>...........................] - ETA: 1:20 - loss: 0.6145 - categorical_accuracy: 0.8018
 6624/60000 [==>...........................] - ETA: 1:20 - loss: 0.6096 - categorical_accuracy: 0.8033
 6688/60000 [==>...........................] - ETA: 1:20 - loss: 0.6053 - categorical_accuracy: 0.8049
 6752/60000 [==>...........................] - ETA: 1:19 - loss: 0.6022 - categorical_accuracy: 0.8060
 6816/60000 [==>...........................] - ETA: 1:19 - loss: 0.5980 - categorical_accuracy: 0.8069
 6848/60000 [==>...........................] - ETA: 1:19 - loss: 0.5956 - categorical_accuracy: 0.8077
 6912/60000 [==>...........................] - ETA: 1:19 - loss: 0.5920 - categorical_accuracy: 0.8090
 6976/60000 [==>...........................] - ETA: 1:19 - loss: 0.5883 - categorical_accuracy: 0.8102
 7040/60000 [==>...........................] - ETA: 1:19 - loss: 0.5845 - categorical_accuracy: 0.8116
 7104/60000 [==>...........................] - ETA: 1:19 - loss: 0.5815 - categorical_accuracy: 0.8126
 7168/60000 [==>...........................] - ETA: 1:19 - loss: 0.5774 - categorical_accuracy: 0.8140
 7232/60000 [==>...........................] - ETA: 1:19 - loss: 0.5755 - categorical_accuracy: 0.8146
 7264/60000 [==>...........................] - ETA: 1:19 - loss: 0.5736 - categorical_accuracy: 0.8153
 7328/60000 [==>...........................] - ETA: 1:19 - loss: 0.5706 - categorical_accuracy: 0.8165
 7392/60000 [==>...........................] - ETA: 1:19 - loss: 0.5694 - categorical_accuracy: 0.8174
 7456/60000 [==>...........................] - ETA: 1:18 - loss: 0.5661 - categorical_accuracy: 0.8184
 7520/60000 [==>...........................] - ETA: 1:18 - loss: 0.5635 - categorical_accuracy: 0.8193
 7584/60000 [==>...........................] - ETA: 1:18 - loss: 0.5621 - categorical_accuracy: 0.8199
 7648/60000 [==>...........................] - ETA: 1:18 - loss: 0.5605 - categorical_accuracy: 0.8207
 7712/60000 [==>...........................] - ETA: 1:18 - loss: 0.5585 - categorical_accuracy: 0.8213
 7744/60000 [==>...........................] - ETA: 1:18 - loss: 0.5569 - categorical_accuracy: 0.8217
 7808/60000 [==>...........................] - ETA: 1:18 - loss: 0.5549 - categorical_accuracy: 0.8225
 7872/60000 [==>...........................] - ETA: 1:18 - loss: 0.5511 - categorical_accuracy: 0.8238
 7936/60000 [==>...........................] - ETA: 1:18 - loss: 0.5493 - categorical_accuracy: 0.8241
 7968/60000 [==>...........................] - ETA: 1:18 - loss: 0.5476 - categorical_accuracy: 0.8247
 8032/60000 [===>..........................] - ETA: 1:17 - loss: 0.5447 - categorical_accuracy: 0.8256
 8096/60000 [===>..........................] - ETA: 1:17 - loss: 0.5420 - categorical_accuracy: 0.8265
 8160/60000 [===>..........................] - ETA: 1:17 - loss: 0.5388 - categorical_accuracy: 0.8276
 8224/60000 [===>..........................] - ETA: 1:17 - loss: 0.5367 - categorical_accuracy: 0.8283
 8288/60000 [===>..........................] - ETA: 1:17 - loss: 0.5338 - categorical_accuracy: 0.8292
 8352/60000 [===>..........................] - ETA: 1:17 - loss: 0.5312 - categorical_accuracy: 0.8299
 8416/60000 [===>..........................] - ETA: 1:17 - loss: 0.5293 - categorical_accuracy: 0.8307
 8480/60000 [===>..........................] - ETA: 1:17 - loss: 0.5262 - categorical_accuracy: 0.8315
 8544/60000 [===>..........................] - ETA: 1:17 - loss: 0.5239 - categorical_accuracy: 0.8323
 8608/60000 [===>..........................] - ETA: 1:16 - loss: 0.5211 - categorical_accuracy: 0.8333
 8672/60000 [===>..........................] - ETA: 1:16 - loss: 0.5187 - categorical_accuracy: 0.8342
 8736/60000 [===>..........................] - ETA: 1:16 - loss: 0.5162 - categorical_accuracy: 0.8348
 8800/60000 [===>..........................] - ETA: 1:16 - loss: 0.5147 - categorical_accuracy: 0.8353
 8864/60000 [===>..........................] - ETA: 1:16 - loss: 0.5120 - categorical_accuracy: 0.8362
 8928/60000 [===>..........................] - ETA: 1:16 - loss: 0.5114 - categorical_accuracy: 0.8369
 8992/60000 [===>..........................] - ETA: 1:16 - loss: 0.5087 - categorical_accuracy: 0.8379
 9056/60000 [===>..........................] - ETA: 1:16 - loss: 0.5063 - categorical_accuracy: 0.8387
 9120/60000 [===>..........................] - ETA: 1:15 - loss: 0.5042 - categorical_accuracy: 0.8394
 9184/60000 [===>..........................] - ETA: 1:15 - loss: 0.5032 - categorical_accuracy: 0.8396
 9248/60000 [===>..........................] - ETA: 1:15 - loss: 0.5027 - categorical_accuracy: 0.8399
 9312/60000 [===>..........................] - ETA: 1:15 - loss: 0.5011 - categorical_accuracy: 0.8404
 9376/60000 [===>..........................] - ETA: 1:15 - loss: 0.5004 - categorical_accuracy: 0.8408
 9440/60000 [===>..........................] - ETA: 1:15 - loss: 0.4991 - categorical_accuracy: 0.8413
 9504/60000 [===>..........................] - ETA: 1:15 - loss: 0.4966 - categorical_accuracy: 0.8421
 9568/60000 [===>..........................] - ETA: 1:15 - loss: 0.4947 - categorical_accuracy: 0.8427
 9632/60000 [===>..........................] - ETA: 1:14 - loss: 0.4932 - categorical_accuracy: 0.8432
 9696/60000 [===>..........................] - ETA: 1:14 - loss: 0.4907 - categorical_accuracy: 0.8441
 9760/60000 [===>..........................] - ETA: 1:14 - loss: 0.4898 - categorical_accuracy: 0.8444
 9824/60000 [===>..........................] - ETA: 1:14 - loss: 0.4879 - categorical_accuracy: 0.8451
 9888/60000 [===>..........................] - ETA: 1:14 - loss: 0.4862 - categorical_accuracy: 0.8455
 9952/60000 [===>..........................] - ETA: 1:14 - loss: 0.4853 - categorical_accuracy: 0.8458
 9984/60000 [===>..........................] - ETA: 1:14 - loss: 0.4847 - categorical_accuracy: 0.8461
10048/60000 [====>.........................] - ETA: 1:14 - loss: 0.4833 - categorical_accuracy: 0.8468
10112/60000 [====>.........................] - ETA: 1:14 - loss: 0.4818 - categorical_accuracy: 0.8472
10176/60000 [====>.........................] - ETA: 1:14 - loss: 0.4823 - categorical_accuracy: 0.8473
10240/60000 [====>.........................] - ETA: 1:14 - loss: 0.4808 - categorical_accuracy: 0.8478
10304/60000 [====>.........................] - ETA: 1:13 - loss: 0.4789 - categorical_accuracy: 0.8484
10336/60000 [====>.........................] - ETA: 1:13 - loss: 0.4776 - categorical_accuracy: 0.8488
10400/60000 [====>.........................] - ETA: 1:13 - loss: 0.4765 - categorical_accuracy: 0.8490
10464/60000 [====>.........................] - ETA: 1:13 - loss: 0.4762 - categorical_accuracy: 0.8492
10528/60000 [====>.........................] - ETA: 1:13 - loss: 0.4749 - categorical_accuracy: 0.8497
10592/60000 [====>.........................] - ETA: 1:13 - loss: 0.4729 - categorical_accuracy: 0.8504
10656/60000 [====>.........................] - ETA: 1:13 - loss: 0.4709 - categorical_accuracy: 0.8508
10720/60000 [====>.........................] - ETA: 1:13 - loss: 0.4705 - categorical_accuracy: 0.8512
10784/60000 [====>.........................] - ETA: 1:13 - loss: 0.4695 - categorical_accuracy: 0.8514
10848/60000 [====>.........................] - ETA: 1:13 - loss: 0.4687 - categorical_accuracy: 0.8517
10912/60000 [====>.........................] - ETA: 1:13 - loss: 0.4670 - categorical_accuracy: 0.8523
10976/60000 [====>.........................] - ETA: 1:12 - loss: 0.4659 - categorical_accuracy: 0.8527
11040/60000 [====>.........................] - ETA: 1:12 - loss: 0.4639 - categorical_accuracy: 0.8533
11104/60000 [====>.........................] - ETA: 1:12 - loss: 0.4625 - categorical_accuracy: 0.8536
11168/60000 [====>.........................] - ETA: 1:12 - loss: 0.4606 - categorical_accuracy: 0.8542
11232/60000 [====>.........................] - ETA: 1:12 - loss: 0.4594 - categorical_accuracy: 0.8545
11296/60000 [====>.........................] - ETA: 1:12 - loss: 0.4575 - categorical_accuracy: 0.8551
11360/60000 [====>.........................] - ETA: 1:12 - loss: 0.4565 - categorical_accuracy: 0.8553
11424/60000 [====>.........................] - ETA: 1:12 - loss: 0.4544 - categorical_accuracy: 0.8560
11488/60000 [====>.........................] - ETA: 1:12 - loss: 0.4532 - categorical_accuracy: 0.8565
11520/60000 [====>.........................] - ETA: 1:12 - loss: 0.4523 - categorical_accuracy: 0.8567
11584/60000 [====>.........................] - ETA: 1:11 - loss: 0.4513 - categorical_accuracy: 0.8571
11648/60000 [====>.........................] - ETA: 1:11 - loss: 0.4503 - categorical_accuracy: 0.8576
11680/60000 [====>.........................] - ETA: 1:11 - loss: 0.4493 - categorical_accuracy: 0.8579
11744/60000 [====>.........................] - ETA: 1:11 - loss: 0.4479 - categorical_accuracy: 0.8583
11808/60000 [====>.........................] - ETA: 1:11 - loss: 0.4474 - categorical_accuracy: 0.8586
11872/60000 [====>.........................] - ETA: 1:11 - loss: 0.4459 - categorical_accuracy: 0.8592
11936/60000 [====>.........................] - ETA: 1:11 - loss: 0.4443 - categorical_accuracy: 0.8597
12000/60000 [=====>........................] - ETA: 1:11 - loss: 0.4433 - categorical_accuracy: 0.8601
12064/60000 [=====>........................] - ETA: 1:11 - loss: 0.4421 - categorical_accuracy: 0.8605
12096/60000 [=====>........................] - ETA: 1:11 - loss: 0.4419 - categorical_accuracy: 0.8605
12160/60000 [=====>........................] - ETA: 1:11 - loss: 0.4412 - categorical_accuracy: 0.8609
12224/60000 [=====>........................] - ETA: 1:10 - loss: 0.4394 - categorical_accuracy: 0.8616
12288/60000 [=====>........................] - ETA: 1:10 - loss: 0.4391 - categorical_accuracy: 0.8617
12352/60000 [=====>........................] - ETA: 1:10 - loss: 0.4380 - categorical_accuracy: 0.8620
12416/60000 [=====>........................] - ETA: 1:10 - loss: 0.4367 - categorical_accuracy: 0.8623
12480/60000 [=====>........................] - ETA: 1:10 - loss: 0.4350 - categorical_accuracy: 0.8628
12544/60000 [=====>........................] - ETA: 1:10 - loss: 0.4346 - categorical_accuracy: 0.8630
12608/60000 [=====>........................] - ETA: 1:10 - loss: 0.4333 - categorical_accuracy: 0.8636
12672/60000 [=====>........................] - ETA: 1:10 - loss: 0.4318 - categorical_accuracy: 0.8640
12736/60000 [=====>........................] - ETA: 1:10 - loss: 0.4309 - categorical_accuracy: 0.8643
12800/60000 [=====>........................] - ETA: 1:10 - loss: 0.4292 - categorical_accuracy: 0.8649
12864/60000 [=====>........................] - ETA: 1:09 - loss: 0.4280 - categorical_accuracy: 0.8653
12896/60000 [=====>........................] - ETA: 1:09 - loss: 0.4279 - categorical_accuracy: 0.8654
12960/60000 [=====>........................] - ETA: 1:09 - loss: 0.4272 - categorical_accuracy: 0.8657
13024/60000 [=====>........................] - ETA: 1:09 - loss: 0.4260 - categorical_accuracy: 0.8660
13056/60000 [=====>........................] - ETA: 1:09 - loss: 0.4257 - categorical_accuracy: 0.8662
13120/60000 [=====>........................] - ETA: 1:09 - loss: 0.4240 - categorical_accuracy: 0.8668
13184/60000 [=====>........................] - ETA: 1:09 - loss: 0.4227 - categorical_accuracy: 0.8671
13248/60000 [=====>........................] - ETA: 1:09 - loss: 0.4213 - categorical_accuracy: 0.8675
13312/60000 [=====>........................] - ETA: 1:09 - loss: 0.4209 - categorical_accuracy: 0.8677
13376/60000 [=====>........................] - ETA: 1:09 - loss: 0.4198 - categorical_accuracy: 0.8680
13440/60000 [=====>........................] - ETA: 1:09 - loss: 0.4193 - categorical_accuracy: 0.8681
13504/60000 [=====>........................] - ETA: 1:08 - loss: 0.4182 - categorical_accuracy: 0.8684
13568/60000 [=====>........................] - ETA: 1:08 - loss: 0.4172 - categorical_accuracy: 0.8687
13632/60000 [=====>........................] - ETA: 1:08 - loss: 0.4161 - categorical_accuracy: 0.8689
13696/60000 [=====>........................] - ETA: 1:08 - loss: 0.4144 - categorical_accuracy: 0.8695
13760/60000 [=====>........................] - ETA: 1:08 - loss: 0.4139 - categorical_accuracy: 0.8695
13824/60000 [=====>........................] - ETA: 1:08 - loss: 0.4127 - categorical_accuracy: 0.8699
13888/60000 [=====>........................] - ETA: 1:08 - loss: 0.4116 - categorical_accuracy: 0.8702
13952/60000 [=====>........................] - ETA: 1:08 - loss: 0.4110 - categorical_accuracy: 0.8705
14016/60000 [======>.......................] - ETA: 1:08 - loss: 0.4101 - categorical_accuracy: 0.8708
14080/60000 [======>.......................] - ETA: 1:07 - loss: 0.4089 - categorical_accuracy: 0.8711
14144/60000 [======>.......................] - ETA: 1:07 - loss: 0.4077 - categorical_accuracy: 0.8716
14176/60000 [======>.......................] - ETA: 1:07 - loss: 0.4077 - categorical_accuracy: 0.8716
14240/60000 [======>.......................] - ETA: 1:07 - loss: 0.4065 - categorical_accuracy: 0.8719
14304/60000 [======>.......................] - ETA: 1:07 - loss: 0.4058 - categorical_accuracy: 0.8722
14368/60000 [======>.......................] - ETA: 1:07 - loss: 0.4042 - categorical_accuracy: 0.8727
14432/60000 [======>.......................] - ETA: 1:07 - loss: 0.4034 - categorical_accuracy: 0.8730
14496/60000 [======>.......................] - ETA: 1:07 - loss: 0.4022 - categorical_accuracy: 0.8733
14560/60000 [======>.......................] - ETA: 1:07 - loss: 0.4022 - categorical_accuracy: 0.8736
14624/60000 [======>.......................] - ETA: 1:07 - loss: 0.4008 - categorical_accuracy: 0.8740
14688/60000 [======>.......................] - ETA: 1:06 - loss: 0.3995 - categorical_accuracy: 0.8744
14752/60000 [======>.......................] - ETA: 1:06 - loss: 0.3986 - categorical_accuracy: 0.8746
14816/60000 [======>.......................] - ETA: 1:06 - loss: 0.3977 - categorical_accuracy: 0.8750
14880/60000 [======>.......................] - ETA: 1:06 - loss: 0.3970 - categorical_accuracy: 0.8753
14944/60000 [======>.......................] - ETA: 1:06 - loss: 0.3960 - categorical_accuracy: 0.8757
15008/60000 [======>.......................] - ETA: 1:06 - loss: 0.3952 - categorical_accuracy: 0.8760
15072/60000 [======>.......................] - ETA: 1:06 - loss: 0.3946 - categorical_accuracy: 0.8763
15136/60000 [======>.......................] - ETA: 1:06 - loss: 0.3946 - categorical_accuracy: 0.8765
15200/60000 [======>.......................] - ETA: 1:06 - loss: 0.3937 - categorical_accuracy: 0.8768
15232/60000 [======>.......................] - ETA: 1:06 - loss: 0.3929 - categorical_accuracy: 0.8770
15296/60000 [======>.......................] - ETA: 1:05 - loss: 0.3922 - categorical_accuracy: 0.8772
15360/60000 [======>.......................] - ETA: 1:05 - loss: 0.3915 - categorical_accuracy: 0.8773
15424/60000 [======>.......................] - ETA: 1:05 - loss: 0.3906 - categorical_accuracy: 0.8777
15488/60000 [======>.......................] - ETA: 1:05 - loss: 0.3895 - categorical_accuracy: 0.8781
15552/60000 [======>.......................] - ETA: 1:05 - loss: 0.3889 - categorical_accuracy: 0.8782
15616/60000 [======>.......................] - ETA: 1:05 - loss: 0.3879 - categorical_accuracy: 0.8787
15648/60000 [======>.......................] - ETA: 1:05 - loss: 0.3875 - categorical_accuracy: 0.8788
15712/60000 [======>.......................] - ETA: 1:05 - loss: 0.3861 - categorical_accuracy: 0.8792
15776/60000 [======>.......................] - ETA: 1:05 - loss: 0.3853 - categorical_accuracy: 0.8793
15840/60000 [======>.......................] - ETA: 1:05 - loss: 0.3848 - categorical_accuracy: 0.8794
15904/60000 [======>.......................] - ETA: 1:04 - loss: 0.3841 - categorical_accuracy: 0.8798
15968/60000 [======>.......................] - ETA: 1:04 - loss: 0.3835 - categorical_accuracy: 0.8799
16032/60000 [=======>......................] - ETA: 1:04 - loss: 0.3826 - categorical_accuracy: 0.8802
16096/60000 [=======>......................] - ETA: 1:04 - loss: 0.3818 - categorical_accuracy: 0.8805
16160/60000 [=======>......................] - ETA: 1:04 - loss: 0.3807 - categorical_accuracy: 0.8808
16224/60000 [=======>......................] - ETA: 1:04 - loss: 0.3798 - categorical_accuracy: 0.8809
16288/60000 [=======>......................] - ETA: 1:04 - loss: 0.3785 - categorical_accuracy: 0.8814
16352/60000 [=======>......................] - ETA: 1:04 - loss: 0.3783 - categorical_accuracy: 0.8816
16416/60000 [=======>......................] - ETA: 1:04 - loss: 0.3774 - categorical_accuracy: 0.8819
16480/60000 [=======>......................] - ETA: 1:04 - loss: 0.3766 - categorical_accuracy: 0.8822
16544/60000 [=======>......................] - ETA: 1:04 - loss: 0.3757 - categorical_accuracy: 0.8826
16608/60000 [=======>......................] - ETA: 1:03 - loss: 0.3750 - categorical_accuracy: 0.8828
16672/60000 [=======>......................] - ETA: 1:03 - loss: 0.3739 - categorical_accuracy: 0.8832
16704/60000 [=======>......................] - ETA: 1:03 - loss: 0.3738 - categorical_accuracy: 0.8832
16768/60000 [=======>......................] - ETA: 1:03 - loss: 0.3727 - categorical_accuracy: 0.8835
16800/60000 [=======>......................] - ETA: 1:03 - loss: 0.3721 - categorical_accuracy: 0.8836
16832/60000 [=======>......................] - ETA: 1:03 - loss: 0.3717 - categorical_accuracy: 0.8838
16896/60000 [=======>......................] - ETA: 1:03 - loss: 0.3709 - categorical_accuracy: 0.8840
16928/60000 [=======>......................] - ETA: 1:03 - loss: 0.3707 - categorical_accuracy: 0.8842
16992/60000 [=======>......................] - ETA: 1:03 - loss: 0.3699 - categorical_accuracy: 0.8844
17056/60000 [=======>......................] - ETA: 1:03 - loss: 0.3698 - categorical_accuracy: 0.8847
17120/60000 [=======>......................] - ETA: 1:03 - loss: 0.3690 - categorical_accuracy: 0.8850
17184/60000 [=======>......................] - ETA: 1:03 - loss: 0.3679 - categorical_accuracy: 0.8854
17248/60000 [=======>......................] - ETA: 1:03 - loss: 0.3668 - categorical_accuracy: 0.8857
17312/60000 [=======>......................] - ETA: 1:02 - loss: 0.3661 - categorical_accuracy: 0.8859
17376/60000 [=======>......................] - ETA: 1:02 - loss: 0.3652 - categorical_accuracy: 0.8861
17440/60000 [=======>......................] - ETA: 1:02 - loss: 0.3646 - categorical_accuracy: 0.8863
17472/60000 [=======>......................] - ETA: 1:02 - loss: 0.3641 - categorical_accuracy: 0.8864
17536/60000 [=======>......................] - ETA: 1:02 - loss: 0.3633 - categorical_accuracy: 0.8866
17600/60000 [=======>......................] - ETA: 1:02 - loss: 0.3626 - categorical_accuracy: 0.8868
17664/60000 [=======>......................] - ETA: 1:02 - loss: 0.3617 - categorical_accuracy: 0.8872
17696/60000 [=======>......................] - ETA: 1:02 - loss: 0.3614 - categorical_accuracy: 0.8873
17760/60000 [=======>......................] - ETA: 1:02 - loss: 0.3605 - categorical_accuracy: 0.8874
17824/60000 [=======>......................] - ETA: 1:02 - loss: 0.3596 - categorical_accuracy: 0.8878
17888/60000 [=======>......................] - ETA: 1:02 - loss: 0.3590 - categorical_accuracy: 0.8881
17952/60000 [=======>......................] - ETA: 1:01 - loss: 0.3581 - categorical_accuracy: 0.8885
18016/60000 [========>.....................] - ETA: 1:01 - loss: 0.3574 - categorical_accuracy: 0.8887
18080/60000 [========>.....................] - ETA: 1:01 - loss: 0.3564 - categorical_accuracy: 0.8890
18144/60000 [========>.....................] - ETA: 1:01 - loss: 0.3560 - categorical_accuracy: 0.8892
18176/60000 [========>.....................] - ETA: 1:01 - loss: 0.3560 - categorical_accuracy: 0.8891
18240/60000 [========>.....................] - ETA: 1:01 - loss: 0.3557 - categorical_accuracy: 0.8893
18304/60000 [========>.....................] - ETA: 1:01 - loss: 0.3547 - categorical_accuracy: 0.8896
18336/60000 [========>.....................] - ETA: 1:01 - loss: 0.3546 - categorical_accuracy: 0.8896
18400/60000 [========>.....................] - ETA: 1:01 - loss: 0.3539 - categorical_accuracy: 0.8897
18464/60000 [========>.....................] - ETA: 1:01 - loss: 0.3529 - categorical_accuracy: 0.8901
18528/60000 [========>.....................] - ETA: 1:01 - loss: 0.3523 - categorical_accuracy: 0.8903
18592/60000 [========>.....................] - ETA: 1:01 - loss: 0.3515 - categorical_accuracy: 0.8905
18656/60000 [========>.....................] - ETA: 1:00 - loss: 0.3505 - categorical_accuracy: 0.8909
18720/60000 [========>.....................] - ETA: 1:00 - loss: 0.3500 - categorical_accuracy: 0.8910
18784/60000 [========>.....................] - ETA: 1:00 - loss: 0.3490 - categorical_accuracy: 0.8912
18848/60000 [========>.....................] - ETA: 1:00 - loss: 0.3479 - categorical_accuracy: 0.8916
18912/60000 [========>.....................] - ETA: 1:00 - loss: 0.3469 - categorical_accuracy: 0.8919
18976/60000 [========>.....................] - ETA: 1:00 - loss: 0.3463 - categorical_accuracy: 0.8920
19040/60000 [========>.....................] - ETA: 1:00 - loss: 0.3459 - categorical_accuracy: 0.8921
19104/60000 [========>.....................] - ETA: 1:00 - loss: 0.3449 - categorical_accuracy: 0.8924
19168/60000 [========>.....................] - ETA: 1:00 - loss: 0.3443 - categorical_accuracy: 0.8926
19232/60000 [========>.....................] - ETA: 59s - loss: 0.3433 - categorical_accuracy: 0.8930 
19296/60000 [========>.....................] - ETA: 59s - loss: 0.3422 - categorical_accuracy: 0.8933
19360/60000 [========>.....................] - ETA: 59s - loss: 0.3419 - categorical_accuracy: 0.8934
19424/60000 [========>.....................] - ETA: 59s - loss: 0.3411 - categorical_accuracy: 0.8937
19488/60000 [========>.....................] - ETA: 59s - loss: 0.3403 - categorical_accuracy: 0.8939
19552/60000 [========>.....................] - ETA: 59s - loss: 0.3394 - categorical_accuracy: 0.8942
19616/60000 [========>.....................] - ETA: 59s - loss: 0.3388 - categorical_accuracy: 0.8944
19680/60000 [========>.....................] - ETA: 59s - loss: 0.3382 - categorical_accuracy: 0.8947
19744/60000 [========>.....................] - ETA: 59s - loss: 0.3378 - categorical_accuracy: 0.8948
19808/60000 [========>.....................] - ETA: 59s - loss: 0.3371 - categorical_accuracy: 0.8950
19872/60000 [========>.....................] - ETA: 58s - loss: 0.3365 - categorical_accuracy: 0.8952
19936/60000 [========>.....................] - ETA: 58s - loss: 0.3357 - categorical_accuracy: 0.8954
20000/60000 [=========>....................] - ETA: 58s - loss: 0.3351 - categorical_accuracy: 0.8957
20064/60000 [=========>....................] - ETA: 58s - loss: 0.3344 - categorical_accuracy: 0.8959
20128/60000 [=========>....................] - ETA: 58s - loss: 0.3345 - categorical_accuracy: 0.8960
20160/60000 [=========>....................] - ETA: 58s - loss: 0.3341 - categorical_accuracy: 0.8961
20224/60000 [=========>....................] - ETA: 58s - loss: 0.3334 - categorical_accuracy: 0.8963
20288/60000 [=========>....................] - ETA: 58s - loss: 0.3326 - categorical_accuracy: 0.8966
20352/60000 [=========>....................] - ETA: 58s - loss: 0.3320 - categorical_accuracy: 0.8967
20416/60000 [=========>....................] - ETA: 58s - loss: 0.3312 - categorical_accuracy: 0.8969
20480/60000 [=========>....................] - ETA: 58s - loss: 0.3310 - categorical_accuracy: 0.8970
20512/60000 [=========>....................] - ETA: 58s - loss: 0.3314 - categorical_accuracy: 0.8969
20576/60000 [=========>....................] - ETA: 58s - loss: 0.3308 - categorical_accuracy: 0.8971
20608/60000 [=========>....................] - ETA: 57s - loss: 0.3303 - categorical_accuracy: 0.8973
20672/60000 [=========>....................] - ETA: 57s - loss: 0.3297 - categorical_accuracy: 0.8974
20736/60000 [=========>....................] - ETA: 57s - loss: 0.3289 - categorical_accuracy: 0.8977
20800/60000 [=========>....................] - ETA: 57s - loss: 0.3286 - categorical_accuracy: 0.8979
20864/60000 [=========>....................] - ETA: 57s - loss: 0.3279 - categorical_accuracy: 0.8981
20928/60000 [=========>....................] - ETA: 57s - loss: 0.3273 - categorical_accuracy: 0.8982
20992/60000 [=========>....................] - ETA: 57s - loss: 0.3270 - categorical_accuracy: 0.8983
21056/60000 [=========>....................] - ETA: 57s - loss: 0.3261 - categorical_accuracy: 0.8987
21088/60000 [=========>....................] - ETA: 57s - loss: 0.3258 - categorical_accuracy: 0.8988
21152/60000 [=========>....................] - ETA: 57s - loss: 0.3254 - categorical_accuracy: 0.8990
21216/60000 [=========>....................] - ETA: 57s - loss: 0.3248 - categorical_accuracy: 0.8992
21280/60000 [=========>....................] - ETA: 56s - loss: 0.3243 - categorical_accuracy: 0.8993
21344/60000 [=========>....................] - ETA: 56s - loss: 0.3236 - categorical_accuracy: 0.8996
21408/60000 [=========>....................] - ETA: 56s - loss: 0.3227 - categorical_accuracy: 0.8999
21472/60000 [=========>....................] - ETA: 56s - loss: 0.3220 - categorical_accuracy: 0.9001
21504/60000 [=========>....................] - ETA: 56s - loss: 0.3216 - categorical_accuracy: 0.9002
21568/60000 [=========>....................] - ETA: 56s - loss: 0.3211 - categorical_accuracy: 0.9004
21632/60000 [=========>....................] - ETA: 56s - loss: 0.3209 - categorical_accuracy: 0.9004
21696/60000 [=========>....................] - ETA: 56s - loss: 0.3203 - categorical_accuracy: 0.9007
21760/60000 [=========>....................] - ETA: 56s - loss: 0.3197 - categorical_accuracy: 0.9009
21824/60000 [=========>....................] - ETA: 56s - loss: 0.3193 - categorical_accuracy: 0.9010
21856/60000 [=========>....................] - ETA: 56s - loss: 0.3193 - categorical_accuracy: 0.9010
21920/60000 [=========>....................] - ETA: 56s - loss: 0.3187 - categorical_accuracy: 0.9011
21984/60000 [=========>....................] - ETA: 55s - loss: 0.3182 - categorical_accuracy: 0.9013
22048/60000 [==========>...................] - ETA: 55s - loss: 0.3176 - categorical_accuracy: 0.9014
22112/60000 [==========>...................] - ETA: 55s - loss: 0.3180 - categorical_accuracy: 0.9014
22176/60000 [==========>...................] - ETA: 55s - loss: 0.3174 - categorical_accuracy: 0.9015
22240/60000 [==========>...................] - ETA: 55s - loss: 0.3170 - categorical_accuracy: 0.9017
22304/60000 [==========>...................] - ETA: 55s - loss: 0.3163 - categorical_accuracy: 0.9019
22368/60000 [==========>...................] - ETA: 55s - loss: 0.3157 - categorical_accuracy: 0.9021
22432/60000 [==========>...................] - ETA: 55s - loss: 0.3152 - categorical_accuracy: 0.9023
22496/60000 [==========>...................] - ETA: 55s - loss: 0.3148 - categorical_accuracy: 0.9024
22560/60000 [==========>...................] - ETA: 55s - loss: 0.3143 - categorical_accuracy: 0.9025
22592/60000 [==========>...................] - ETA: 55s - loss: 0.3140 - categorical_accuracy: 0.9026
22656/60000 [==========>...................] - ETA: 54s - loss: 0.3134 - categorical_accuracy: 0.9028
22720/60000 [==========>...................] - ETA: 54s - loss: 0.3127 - categorical_accuracy: 0.9030
22784/60000 [==========>...................] - ETA: 54s - loss: 0.3123 - categorical_accuracy: 0.9031
22848/60000 [==========>...................] - ETA: 54s - loss: 0.3119 - categorical_accuracy: 0.9033
22912/60000 [==========>...................] - ETA: 54s - loss: 0.3114 - categorical_accuracy: 0.9035
22976/60000 [==========>...................] - ETA: 54s - loss: 0.3111 - categorical_accuracy: 0.9036
23040/60000 [==========>...................] - ETA: 54s - loss: 0.3104 - categorical_accuracy: 0.9038
23104/60000 [==========>...................] - ETA: 54s - loss: 0.3099 - categorical_accuracy: 0.9040
23168/60000 [==========>...................] - ETA: 54s - loss: 0.3098 - categorical_accuracy: 0.9040
23232/60000 [==========>...................] - ETA: 54s - loss: 0.3093 - categorical_accuracy: 0.9042
23296/60000 [==========>...................] - ETA: 53s - loss: 0.3091 - categorical_accuracy: 0.9043
23360/60000 [==========>...................] - ETA: 53s - loss: 0.3086 - categorical_accuracy: 0.9045
23424/60000 [==========>...................] - ETA: 53s - loss: 0.3082 - categorical_accuracy: 0.9046
23488/60000 [==========>...................] - ETA: 53s - loss: 0.3077 - categorical_accuracy: 0.9048
23552/60000 [==========>...................] - ETA: 53s - loss: 0.3074 - categorical_accuracy: 0.9048
23616/60000 [==========>...................] - ETA: 53s - loss: 0.3066 - categorical_accuracy: 0.9051
23680/60000 [==========>...................] - ETA: 53s - loss: 0.3059 - categorical_accuracy: 0.9053
23744/60000 [==========>...................] - ETA: 53s - loss: 0.3055 - categorical_accuracy: 0.9054
23808/60000 [==========>...................] - ETA: 53s - loss: 0.3047 - categorical_accuracy: 0.9057
23872/60000 [==========>...................] - ETA: 53s - loss: 0.3044 - categorical_accuracy: 0.9057
23936/60000 [==========>...................] - ETA: 52s - loss: 0.3038 - categorical_accuracy: 0.9060
24000/60000 [===========>..................] - ETA: 52s - loss: 0.3033 - categorical_accuracy: 0.9061
24032/60000 [===========>..................] - ETA: 52s - loss: 0.3033 - categorical_accuracy: 0.9061
24096/60000 [===========>..................] - ETA: 52s - loss: 0.3031 - categorical_accuracy: 0.9062
24160/60000 [===========>..................] - ETA: 52s - loss: 0.3027 - categorical_accuracy: 0.9064
24224/60000 [===========>..................] - ETA: 52s - loss: 0.3021 - categorical_accuracy: 0.9066
24288/60000 [===========>..................] - ETA: 52s - loss: 0.3014 - categorical_accuracy: 0.9068
24352/60000 [===========>..................] - ETA: 52s - loss: 0.3007 - categorical_accuracy: 0.9070
24416/60000 [===========>..................] - ETA: 52s - loss: 0.3003 - categorical_accuracy: 0.9072
24480/60000 [===========>..................] - ETA: 52s - loss: 0.2998 - categorical_accuracy: 0.9074
24544/60000 [===========>..................] - ETA: 52s - loss: 0.2992 - categorical_accuracy: 0.9076
24608/60000 [===========>..................] - ETA: 51s - loss: 0.2988 - categorical_accuracy: 0.9076
24640/60000 [===========>..................] - ETA: 51s - loss: 0.2986 - categorical_accuracy: 0.9076
24704/60000 [===========>..................] - ETA: 51s - loss: 0.2983 - categorical_accuracy: 0.9078
24768/60000 [===========>..................] - ETA: 51s - loss: 0.2985 - categorical_accuracy: 0.9078
24832/60000 [===========>..................] - ETA: 51s - loss: 0.2986 - categorical_accuracy: 0.9079
24864/60000 [===========>..................] - ETA: 51s - loss: 0.2987 - categorical_accuracy: 0.9079
24928/60000 [===========>..................] - ETA: 51s - loss: 0.2985 - categorical_accuracy: 0.9080
24992/60000 [===========>..................] - ETA: 51s - loss: 0.2984 - categorical_accuracy: 0.9080
25056/60000 [===========>..................] - ETA: 51s - loss: 0.2978 - categorical_accuracy: 0.9082
25120/60000 [===========>..................] - ETA: 51s - loss: 0.2974 - categorical_accuracy: 0.9084
25184/60000 [===========>..................] - ETA: 51s - loss: 0.2970 - categorical_accuracy: 0.9084
25248/60000 [===========>..................] - ETA: 51s - loss: 0.2965 - categorical_accuracy: 0.9085
25312/60000 [===========>..................] - ETA: 50s - loss: 0.2959 - categorical_accuracy: 0.9087
25376/60000 [===========>..................] - ETA: 50s - loss: 0.2955 - categorical_accuracy: 0.9089
25440/60000 [===========>..................] - ETA: 50s - loss: 0.2950 - categorical_accuracy: 0.9090
25504/60000 [===========>..................] - ETA: 50s - loss: 0.2944 - categorical_accuracy: 0.9092
25568/60000 [===========>..................] - ETA: 50s - loss: 0.2939 - categorical_accuracy: 0.9094
25632/60000 [===========>..................] - ETA: 50s - loss: 0.2935 - categorical_accuracy: 0.9095
25696/60000 [===========>..................] - ETA: 50s - loss: 0.2929 - categorical_accuracy: 0.9097
25760/60000 [===========>..................] - ETA: 50s - loss: 0.2926 - categorical_accuracy: 0.9098
25824/60000 [===========>..................] - ETA: 50s - loss: 0.2925 - categorical_accuracy: 0.9099
25888/60000 [===========>..................] - ETA: 50s - loss: 0.2920 - categorical_accuracy: 0.9100
25952/60000 [===========>..................] - ETA: 49s - loss: 0.2915 - categorical_accuracy: 0.9102
26016/60000 [============>.................] - ETA: 49s - loss: 0.2914 - categorical_accuracy: 0.9102
26080/60000 [============>.................] - ETA: 49s - loss: 0.2911 - categorical_accuracy: 0.9104
26144/60000 [============>.................] - ETA: 49s - loss: 0.2910 - categorical_accuracy: 0.9103
26208/60000 [============>.................] - ETA: 49s - loss: 0.2904 - categorical_accuracy: 0.9105
26272/60000 [============>.................] - ETA: 49s - loss: 0.2900 - categorical_accuracy: 0.9106
26336/60000 [============>.................] - ETA: 49s - loss: 0.2897 - categorical_accuracy: 0.9108
26400/60000 [============>.................] - ETA: 49s - loss: 0.2894 - categorical_accuracy: 0.9109
26464/60000 [============>.................] - ETA: 49s - loss: 0.2894 - categorical_accuracy: 0.9110
26528/60000 [============>.................] - ETA: 49s - loss: 0.2890 - categorical_accuracy: 0.9111
26592/60000 [============>.................] - ETA: 48s - loss: 0.2885 - categorical_accuracy: 0.9113
26656/60000 [============>.................] - ETA: 48s - loss: 0.2880 - categorical_accuracy: 0.9114
26720/60000 [============>.................] - ETA: 48s - loss: 0.2876 - categorical_accuracy: 0.9115
26784/60000 [============>.................] - ETA: 48s - loss: 0.2871 - categorical_accuracy: 0.9117
26816/60000 [============>.................] - ETA: 48s - loss: 0.2870 - categorical_accuracy: 0.9118
26848/60000 [============>.................] - ETA: 48s - loss: 0.2867 - categorical_accuracy: 0.9119
26912/60000 [============>.................] - ETA: 48s - loss: 0.2861 - categorical_accuracy: 0.9120
26976/60000 [============>.................] - ETA: 48s - loss: 0.2856 - categorical_accuracy: 0.9122
27040/60000 [============>.................] - ETA: 48s - loss: 0.2853 - categorical_accuracy: 0.9123
27104/60000 [============>.................] - ETA: 48s - loss: 0.2849 - categorical_accuracy: 0.9124
27136/60000 [============>.................] - ETA: 48s - loss: 0.2849 - categorical_accuracy: 0.9124
27200/60000 [============>.................] - ETA: 48s - loss: 0.2845 - categorical_accuracy: 0.9125
27232/60000 [============>.................] - ETA: 48s - loss: 0.2844 - categorical_accuracy: 0.9126
27264/60000 [============>.................] - ETA: 48s - loss: 0.2843 - categorical_accuracy: 0.9126
27328/60000 [============>.................] - ETA: 47s - loss: 0.2841 - categorical_accuracy: 0.9127
27392/60000 [============>.................] - ETA: 47s - loss: 0.2837 - categorical_accuracy: 0.9128
27456/60000 [============>.................] - ETA: 47s - loss: 0.2836 - categorical_accuracy: 0.9128
27520/60000 [============>.................] - ETA: 47s - loss: 0.2831 - categorical_accuracy: 0.9130
27584/60000 [============>.................] - ETA: 47s - loss: 0.2828 - categorical_accuracy: 0.9131
27648/60000 [============>.................] - ETA: 47s - loss: 0.2827 - categorical_accuracy: 0.9131
27712/60000 [============>.................] - ETA: 47s - loss: 0.2822 - categorical_accuracy: 0.9133
27776/60000 [============>.................] - ETA: 47s - loss: 0.2819 - categorical_accuracy: 0.9133
27840/60000 [============>.................] - ETA: 47s - loss: 0.2818 - categorical_accuracy: 0.9133
27872/60000 [============>.................] - ETA: 47s - loss: 0.2821 - categorical_accuracy: 0.9133
27936/60000 [============>.................] - ETA: 47s - loss: 0.2819 - categorical_accuracy: 0.9133
28000/60000 [=============>................] - ETA: 46s - loss: 0.2815 - categorical_accuracy: 0.9135
28032/60000 [=============>................] - ETA: 46s - loss: 0.2817 - categorical_accuracy: 0.9135
28096/60000 [=============>................] - ETA: 46s - loss: 0.2814 - categorical_accuracy: 0.9135
28160/60000 [=============>................] - ETA: 46s - loss: 0.2813 - categorical_accuracy: 0.9136
28192/60000 [=============>................] - ETA: 46s - loss: 0.2811 - categorical_accuracy: 0.9137
28256/60000 [=============>................] - ETA: 46s - loss: 0.2808 - categorical_accuracy: 0.9138
28320/60000 [=============>................] - ETA: 46s - loss: 0.2807 - categorical_accuracy: 0.9138
28384/60000 [=============>................] - ETA: 46s - loss: 0.2804 - categorical_accuracy: 0.9140
28416/60000 [=============>................] - ETA: 46s - loss: 0.2801 - categorical_accuracy: 0.9141
28480/60000 [=============>................] - ETA: 46s - loss: 0.2795 - categorical_accuracy: 0.9142
28544/60000 [=============>................] - ETA: 46s - loss: 0.2792 - categorical_accuracy: 0.9143
28608/60000 [=============>................] - ETA: 46s - loss: 0.2789 - categorical_accuracy: 0.9144
28640/60000 [=============>................] - ETA: 46s - loss: 0.2787 - categorical_accuracy: 0.9144
28672/60000 [=============>................] - ETA: 46s - loss: 0.2785 - categorical_accuracy: 0.9145
28736/60000 [=============>................] - ETA: 45s - loss: 0.2781 - categorical_accuracy: 0.9146
28800/60000 [=============>................] - ETA: 45s - loss: 0.2776 - categorical_accuracy: 0.9148
28832/60000 [=============>................] - ETA: 45s - loss: 0.2776 - categorical_accuracy: 0.9148
28864/60000 [=============>................] - ETA: 45s - loss: 0.2774 - categorical_accuracy: 0.9148
28896/60000 [=============>................] - ETA: 45s - loss: 0.2772 - categorical_accuracy: 0.9149
28960/60000 [=============>................] - ETA: 45s - loss: 0.2767 - categorical_accuracy: 0.9150
28992/60000 [=============>................] - ETA: 45s - loss: 0.2766 - categorical_accuracy: 0.9150
29056/60000 [=============>................] - ETA: 45s - loss: 0.2762 - categorical_accuracy: 0.9152
29120/60000 [=============>................] - ETA: 45s - loss: 0.2758 - categorical_accuracy: 0.9152
29184/60000 [=============>................] - ETA: 45s - loss: 0.2754 - categorical_accuracy: 0.9153
29248/60000 [=============>................] - ETA: 45s - loss: 0.2750 - categorical_accuracy: 0.9155
29312/60000 [=============>................] - ETA: 45s - loss: 0.2747 - categorical_accuracy: 0.9156
29376/60000 [=============>................] - ETA: 45s - loss: 0.2741 - categorical_accuracy: 0.9158
29440/60000 [=============>................] - ETA: 44s - loss: 0.2737 - categorical_accuracy: 0.9160
29472/60000 [=============>................] - ETA: 44s - loss: 0.2734 - categorical_accuracy: 0.9160
29536/60000 [=============>................] - ETA: 44s - loss: 0.2736 - categorical_accuracy: 0.9161
29600/60000 [=============>................] - ETA: 44s - loss: 0.2735 - categorical_accuracy: 0.9162
29664/60000 [=============>................] - ETA: 44s - loss: 0.2733 - categorical_accuracy: 0.9163
29728/60000 [=============>................] - ETA: 44s - loss: 0.2729 - categorical_accuracy: 0.9163
29792/60000 [=============>................] - ETA: 44s - loss: 0.2729 - categorical_accuracy: 0.9164
29856/60000 [=============>................] - ETA: 44s - loss: 0.2727 - categorical_accuracy: 0.9164
29920/60000 [=============>................] - ETA: 44s - loss: 0.2724 - categorical_accuracy: 0.9165
29984/60000 [=============>................] - ETA: 44s - loss: 0.2722 - categorical_accuracy: 0.9165
30048/60000 [==============>...............] - ETA: 44s - loss: 0.2721 - categorical_accuracy: 0.9165
30112/60000 [==============>...............] - ETA: 43s - loss: 0.2716 - categorical_accuracy: 0.9167
30176/60000 [==============>...............] - ETA: 43s - loss: 0.2712 - categorical_accuracy: 0.9168
30208/60000 [==============>...............] - ETA: 43s - loss: 0.2710 - categorical_accuracy: 0.9168
30272/60000 [==============>...............] - ETA: 43s - loss: 0.2708 - categorical_accuracy: 0.9169
30336/60000 [==============>...............] - ETA: 43s - loss: 0.2706 - categorical_accuracy: 0.9170
30400/60000 [==============>...............] - ETA: 43s - loss: 0.2706 - categorical_accuracy: 0.9171
30432/60000 [==============>...............] - ETA: 43s - loss: 0.2704 - categorical_accuracy: 0.9172
30496/60000 [==============>...............] - ETA: 43s - loss: 0.2701 - categorical_accuracy: 0.9173
30560/60000 [==============>...............] - ETA: 43s - loss: 0.2699 - categorical_accuracy: 0.9172
30624/60000 [==============>...............] - ETA: 43s - loss: 0.2695 - categorical_accuracy: 0.9173
30688/60000 [==============>...............] - ETA: 43s - loss: 0.2691 - categorical_accuracy: 0.9174
30752/60000 [==============>...............] - ETA: 42s - loss: 0.2689 - categorical_accuracy: 0.9175
30816/60000 [==============>...............] - ETA: 42s - loss: 0.2687 - categorical_accuracy: 0.9176
30880/60000 [==============>...............] - ETA: 42s - loss: 0.2684 - categorical_accuracy: 0.9177
30944/60000 [==============>...............] - ETA: 42s - loss: 0.2682 - categorical_accuracy: 0.9177
30976/60000 [==============>...............] - ETA: 42s - loss: 0.2681 - categorical_accuracy: 0.9178
31040/60000 [==============>...............] - ETA: 42s - loss: 0.2678 - categorical_accuracy: 0.9179
31104/60000 [==============>...............] - ETA: 42s - loss: 0.2674 - categorical_accuracy: 0.9180
31168/60000 [==============>...............] - ETA: 42s - loss: 0.2672 - categorical_accuracy: 0.9180
31232/60000 [==============>...............] - ETA: 42s - loss: 0.2669 - categorical_accuracy: 0.9181
31296/60000 [==============>...............] - ETA: 42s - loss: 0.2667 - categorical_accuracy: 0.9182
31360/60000 [==============>...............] - ETA: 42s - loss: 0.2664 - categorical_accuracy: 0.9183
31424/60000 [==============>...............] - ETA: 41s - loss: 0.2660 - categorical_accuracy: 0.9184
31488/60000 [==============>...............] - ETA: 41s - loss: 0.2657 - categorical_accuracy: 0.9184
31552/60000 [==============>...............] - ETA: 41s - loss: 0.2653 - categorical_accuracy: 0.9185
31616/60000 [==============>...............] - ETA: 41s - loss: 0.2650 - categorical_accuracy: 0.9186
31680/60000 [==============>...............] - ETA: 41s - loss: 0.2649 - categorical_accuracy: 0.9186
31744/60000 [==============>...............] - ETA: 41s - loss: 0.2645 - categorical_accuracy: 0.9187
31776/60000 [==============>...............] - ETA: 41s - loss: 0.2643 - categorical_accuracy: 0.9188
31840/60000 [==============>...............] - ETA: 41s - loss: 0.2639 - categorical_accuracy: 0.9189
31904/60000 [==============>...............] - ETA: 41s - loss: 0.2635 - categorical_accuracy: 0.9191
31936/60000 [==============>...............] - ETA: 41s - loss: 0.2634 - categorical_accuracy: 0.9191
31968/60000 [==============>...............] - ETA: 41s - loss: 0.2632 - categorical_accuracy: 0.9191
32032/60000 [===============>..............] - ETA: 41s - loss: 0.2629 - categorical_accuracy: 0.9192
32096/60000 [===============>..............] - ETA: 40s - loss: 0.2624 - categorical_accuracy: 0.9193
32128/60000 [===============>..............] - ETA: 40s - loss: 0.2622 - categorical_accuracy: 0.9194
32160/60000 [===============>..............] - ETA: 40s - loss: 0.2621 - categorical_accuracy: 0.9194
32224/60000 [===============>..............] - ETA: 40s - loss: 0.2621 - categorical_accuracy: 0.9195
32288/60000 [===============>..............] - ETA: 40s - loss: 0.2621 - categorical_accuracy: 0.9195
32320/60000 [===============>..............] - ETA: 40s - loss: 0.2619 - categorical_accuracy: 0.9196
32352/60000 [===============>..............] - ETA: 40s - loss: 0.2617 - categorical_accuracy: 0.9197
32384/60000 [===============>..............] - ETA: 40s - loss: 0.2615 - categorical_accuracy: 0.9197
32448/60000 [===============>..............] - ETA: 40s - loss: 0.2613 - categorical_accuracy: 0.9198
32512/60000 [===============>..............] - ETA: 40s - loss: 0.2609 - categorical_accuracy: 0.9199
32576/60000 [===============>..............] - ETA: 40s - loss: 0.2608 - categorical_accuracy: 0.9199
32640/60000 [===============>..............] - ETA: 40s - loss: 0.2606 - categorical_accuracy: 0.9200
32704/60000 [===============>..............] - ETA: 40s - loss: 0.2606 - categorical_accuracy: 0.9200
32736/60000 [===============>..............] - ETA: 40s - loss: 0.2604 - categorical_accuracy: 0.9201
32800/60000 [===============>..............] - ETA: 39s - loss: 0.2600 - categorical_accuracy: 0.9202
32864/60000 [===============>..............] - ETA: 39s - loss: 0.2597 - categorical_accuracy: 0.9203
32928/60000 [===============>..............] - ETA: 39s - loss: 0.2593 - categorical_accuracy: 0.9205
32992/60000 [===============>..............] - ETA: 39s - loss: 0.2590 - categorical_accuracy: 0.9206
33024/60000 [===============>..............] - ETA: 39s - loss: 0.2589 - categorical_accuracy: 0.9206
33088/60000 [===============>..............] - ETA: 39s - loss: 0.2586 - categorical_accuracy: 0.9207
33152/60000 [===============>..............] - ETA: 39s - loss: 0.2585 - categorical_accuracy: 0.9207
33216/60000 [===============>..............] - ETA: 39s - loss: 0.2582 - categorical_accuracy: 0.9208
33248/60000 [===============>..............] - ETA: 39s - loss: 0.2580 - categorical_accuracy: 0.9208
33312/60000 [===============>..............] - ETA: 39s - loss: 0.2576 - categorical_accuracy: 0.9210
33376/60000 [===============>..............] - ETA: 39s - loss: 0.2577 - categorical_accuracy: 0.9210
33440/60000 [===============>..............] - ETA: 39s - loss: 0.2573 - categorical_accuracy: 0.9211
33504/60000 [===============>..............] - ETA: 38s - loss: 0.2569 - categorical_accuracy: 0.9213
33568/60000 [===============>..............] - ETA: 38s - loss: 0.2567 - categorical_accuracy: 0.9213
33632/60000 [===============>..............] - ETA: 38s - loss: 0.2564 - categorical_accuracy: 0.9214
33696/60000 [===============>..............] - ETA: 38s - loss: 0.2562 - categorical_accuracy: 0.9215
33728/60000 [===============>..............] - ETA: 38s - loss: 0.2561 - categorical_accuracy: 0.9215
33792/60000 [===============>..............] - ETA: 38s - loss: 0.2558 - categorical_accuracy: 0.9215
33856/60000 [===============>..............] - ETA: 38s - loss: 0.2556 - categorical_accuracy: 0.9216
33920/60000 [===============>..............] - ETA: 38s - loss: 0.2553 - categorical_accuracy: 0.9217
33984/60000 [===============>..............] - ETA: 38s - loss: 0.2550 - categorical_accuracy: 0.9218
34048/60000 [================>.............] - ETA: 38s - loss: 0.2549 - categorical_accuracy: 0.9219
34112/60000 [================>.............] - ETA: 38s - loss: 0.2549 - categorical_accuracy: 0.9219
34176/60000 [================>.............] - ETA: 37s - loss: 0.2546 - categorical_accuracy: 0.9220
34240/60000 [================>.............] - ETA: 37s - loss: 0.2542 - categorical_accuracy: 0.9221
34272/60000 [================>.............] - ETA: 37s - loss: 0.2540 - categorical_accuracy: 0.9221
34304/60000 [================>.............] - ETA: 37s - loss: 0.2541 - categorical_accuracy: 0.9221
34368/60000 [================>.............] - ETA: 37s - loss: 0.2539 - categorical_accuracy: 0.9222
34400/60000 [================>.............] - ETA: 37s - loss: 0.2538 - categorical_accuracy: 0.9222
34464/60000 [================>.............] - ETA: 37s - loss: 0.2535 - categorical_accuracy: 0.9223
34528/60000 [================>.............] - ETA: 37s - loss: 0.2533 - categorical_accuracy: 0.9224
34592/60000 [================>.............] - ETA: 37s - loss: 0.2530 - categorical_accuracy: 0.9224
34624/60000 [================>.............] - ETA: 37s - loss: 0.2529 - categorical_accuracy: 0.9224
34688/60000 [================>.............] - ETA: 37s - loss: 0.2528 - categorical_accuracy: 0.9225
34752/60000 [================>.............] - ETA: 37s - loss: 0.2525 - categorical_accuracy: 0.9226
34816/60000 [================>.............] - ETA: 37s - loss: 0.2522 - categorical_accuracy: 0.9227
34880/60000 [================>.............] - ETA: 36s - loss: 0.2520 - categorical_accuracy: 0.9227
34944/60000 [================>.............] - ETA: 36s - loss: 0.2516 - categorical_accuracy: 0.9228
35008/60000 [================>.............] - ETA: 36s - loss: 0.2513 - categorical_accuracy: 0.9229
35072/60000 [================>.............] - ETA: 36s - loss: 0.2509 - categorical_accuracy: 0.9230
35104/60000 [================>.............] - ETA: 36s - loss: 0.2507 - categorical_accuracy: 0.9231
35136/60000 [================>.............] - ETA: 36s - loss: 0.2506 - categorical_accuracy: 0.9231
35200/60000 [================>.............] - ETA: 36s - loss: 0.2503 - categorical_accuracy: 0.9232
35264/60000 [================>.............] - ETA: 36s - loss: 0.2500 - categorical_accuracy: 0.9233
35328/60000 [================>.............] - ETA: 36s - loss: 0.2497 - categorical_accuracy: 0.9234
35392/60000 [================>.............] - ETA: 36s - loss: 0.2495 - categorical_accuracy: 0.9235
35456/60000 [================>.............] - ETA: 36s - loss: 0.2492 - categorical_accuracy: 0.9236
35520/60000 [================>.............] - ETA: 35s - loss: 0.2490 - categorical_accuracy: 0.9236
35584/60000 [================>.............] - ETA: 35s - loss: 0.2487 - categorical_accuracy: 0.9237
35648/60000 [================>.............] - ETA: 35s - loss: 0.2487 - categorical_accuracy: 0.9238
35680/60000 [================>.............] - ETA: 35s - loss: 0.2486 - categorical_accuracy: 0.9238
35712/60000 [================>.............] - ETA: 35s - loss: 0.2484 - categorical_accuracy: 0.9238
35776/60000 [================>.............] - ETA: 35s - loss: 0.2481 - categorical_accuracy: 0.9239
35808/60000 [================>.............] - ETA: 35s - loss: 0.2479 - categorical_accuracy: 0.9240
35872/60000 [================>.............] - ETA: 35s - loss: 0.2479 - categorical_accuracy: 0.9240
35936/60000 [================>.............] - ETA: 35s - loss: 0.2475 - categorical_accuracy: 0.9241
36000/60000 [=================>............] - ETA: 35s - loss: 0.2473 - categorical_accuracy: 0.9241
36064/60000 [=================>............] - ETA: 35s - loss: 0.2471 - categorical_accuracy: 0.9242
36128/60000 [=================>............] - ETA: 35s - loss: 0.2468 - categorical_accuracy: 0.9243
36192/60000 [=================>............] - ETA: 34s - loss: 0.2466 - categorical_accuracy: 0.9244
36256/60000 [=================>............] - ETA: 34s - loss: 0.2465 - categorical_accuracy: 0.9244
36320/60000 [=================>............] - ETA: 34s - loss: 0.2463 - categorical_accuracy: 0.9245
36384/60000 [=================>............] - ETA: 34s - loss: 0.2461 - categorical_accuracy: 0.9246
36448/60000 [=================>............] - ETA: 34s - loss: 0.2458 - categorical_accuracy: 0.9246
36512/60000 [=================>............] - ETA: 34s - loss: 0.2457 - categorical_accuracy: 0.9247
36576/60000 [=================>............] - ETA: 34s - loss: 0.2455 - categorical_accuracy: 0.9247
36640/60000 [=================>............] - ETA: 34s - loss: 0.2452 - categorical_accuracy: 0.9248
36704/60000 [=================>............] - ETA: 34s - loss: 0.2450 - categorical_accuracy: 0.9249
36768/60000 [=================>............] - ETA: 34s - loss: 0.2446 - categorical_accuracy: 0.9250
36832/60000 [=================>............] - ETA: 34s - loss: 0.2444 - categorical_accuracy: 0.9250
36896/60000 [=================>............] - ETA: 33s - loss: 0.2441 - categorical_accuracy: 0.9251
36960/60000 [=================>............] - ETA: 33s - loss: 0.2439 - categorical_accuracy: 0.9252
37024/60000 [=================>............] - ETA: 33s - loss: 0.2436 - categorical_accuracy: 0.9253
37088/60000 [=================>............] - ETA: 33s - loss: 0.2433 - categorical_accuracy: 0.9254
37120/60000 [=================>............] - ETA: 33s - loss: 0.2431 - categorical_accuracy: 0.9254
37152/60000 [=================>............] - ETA: 33s - loss: 0.2430 - categorical_accuracy: 0.9255
37216/60000 [=================>............] - ETA: 33s - loss: 0.2426 - categorical_accuracy: 0.9256
37280/60000 [=================>............] - ETA: 33s - loss: 0.2423 - categorical_accuracy: 0.9257
37344/60000 [=================>............] - ETA: 33s - loss: 0.2422 - categorical_accuracy: 0.9257
37408/60000 [=================>............] - ETA: 33s - loss: 0.2420 - categorical_accuracy: 0.9257
37472/60000 [=================>............] - ETA: 33s - loss: 0.2418 - categorical_accuracy: 0.9258
37536/60000 [=================>............] - ETA: 33s - loss: 0.2415 - categorical_accuracy: 0.9258
37600/60000 [=================>............] - ETA: 32s - loss: 0.2412 - categorical_accuracy: 0.9259
37664/60000 [=================>............] - ETA: 32s - loss: 0.2410 - categorical_accuracy: 0.9260
37728/60000 [=================>............] - ETA: 32s - loss: 0.2408 - categorical_accuracy: 0.9260
37792/60000 [=================>............] - ETA: 32s - loss: 0.2405 - categorical_accuracy: 0.9261
37856/60000 [=================>............] - ETA: 32s - loss: 0.2404 - categorical_accuracy: 0.9262
37920/60000 [=================>............] - ETA: 32s - loss: 0.2400 - categorical_accuracy: 0.9263
37984/60000 [=================>............] - ETA: 32s - loss: 0.2397 - categorical_accuracy: 0.9264
38048/60000 [==================>...........] - ETA: 32s - loss: 0.2395 - categorical_accuracy: 0.9265
38080/60000 [==================>...........] - ETA: 32s - loss: 0.2394 - categorical_accuracy: 0.9265
38144/60000 [==================>...........] - ETA: 32s - loss: 0.2392 - categorical_accuracy: 0.9266
38208/60000 [==================>...........] - ETA: 32s - loss: 0.2389 - categorical_accuracy: 0.9267
38272/60000 [==================>...........] - ETA: 31s - loss: 0.2388 - categorical_accuracy: 0.9268
38336/60000 [==================>...........] - ETA: 31s - loss: 0.2386 - categorical_accuracy: 0.9269
38400/60000 [==================>...........] - ETA: 31s - loss: 0.2384 - categorical_accuracy: 0.9270
38464/60000 [==================>...........] - ETA: 31s - loss: 0.2381 - categorical_accuracy: 0.9270
38528/60000 [==================>...........] - ETA: 31s - loss: 0.2381 - categorical_accuracy: 0.9271
38592/60000 [==================>...........] - ETA: 31s - loss: 0.2382 - categorical_accuracy: 0.9271
38624/60000 [==================>...........] - ETA: 31s - loss: 0.2380 - categorical_accuracy: 0.9272
38688/60000 [==================>...........] - ETA: 31s - loss: 0.2378 - categorical_accuracy: 0.9273
38752/60000 [==================>...........] - ETA: 31s - loss: 0.2375 - categorical_accuracy: 0.9273
38816/60000 [==================>...........] - ETA: 31s - loss: 0.2374 - categorical_accuracy: 0.9273
38880/60000 [==================>...........] - ETA: 31s - loss: 0.2370 - categorical_accuracy: 0.9275
38944/60000 [==================>...........] - ETA: 30s - loss: 0.2368 - categorical_accuracy: 0.9276
39008/60000 [==================>...........] - ETA: 30s - loss: 0.2366 - categorical_accuracy: 0.9276
39072/60000 [==================>...........] - ETA: 30s - loss: 0.2364 - categorical_accuracy: 0.9277
39136/60000 [==================>...........] - ETA: 30s - loss: 0.2363 - categorical_accuracy: 0.9278
39200/60000 [==================>...........] - ETA: 30s - loss: 0.2360 - categorical_accuracy: 0.9279
39264/60000 [==================>...........] - ETA: 30s - loss: 0.2357 - categorical_accuracy: 0.9279
39296/60000 [==================>...........] - ETA: 30s - loss: 0.2356 - categorical_accuracy: 0.9280
39328/60000 [==================>...........] - ETA: 30s - loss: 0.2355 - categorical_accuracy: 0.9280
39392/60000 [==================>...........] - ETA: 30s - loss: 0.2353 - categorical_accuracy: 0.9280
39456/60000 [==================>...........] - ETA: 30s - loss: 0.2350 - categorical_accuracy: 0.9281
39520/60000 [==================>...........] - ETA: 30s - loss: 0.2347 - categorical_accuracy: 0.9282
39584/60000 [==================>...........] - ETA: 29s - loss: 0.2345 - categorical_accuracy: 0.9283
39648/60000 [==================>...........] - ETA: 29s - loss: 0.2343 - categorical_accuracy: 0.9283
39712/60000 [==================>...........] - ETA: 29s - loss: 0.2343 - categorical_accuracy: 0.9284
39776/60000 [==================>...........] - ETA: 29s - loss: 0.2342 - categorical_accuracy: 0.9284
39840/60000 [==================>...........] - ETA: 29s - loss: 0.2339 - categorical_accuracy: 0.9285
39904/60000 [==================>...........] - ETA: 29s - loss: 0.2336 - categorical_accuracy: 0.9286
39936/60000 [==================>...........] - ETA: 29s - loss: 0.2334 - categorical_accuracy: 0.9287
40000/60000 [===================>..........] - ETA: 29s - loss: 0.2331 - categorical_accuracy: 0.9287
40064/60000 [===================>..........] - ETA: 29s - loss: 0.2329 - categorical_accuracy: 0.9288
40128/60000 [===================>..........] - ETA: 29s - loss: 0.2327 - categorical_accuracy: 0.9289
40192/60000 [===================>..........] - ETA: 29s - loss: 0.2327 - categorical_accuracy: 0.9289
40256/60000 [===================>..........] - ETA: 28s - loss: 0.2325 - categorical_accuracy: 0.9290
40288/60000 [===================>..........] - ETA: 28s - loss: 0.2323 - categorical_accuracy: 0.9291
40352/60000 [===================>..........] - ETA: 28s - loss: 0.2320 - categorical_accuracy: 0.9292
40384/60000 [===================>..........] - ETA: 28s - loss: 0.2318 - categorical_accuracy: 0.9293
40448/60000 [===================>..........] - ETA: 28s - loss: 0.2315 - categorical_accuracy: 0.9293
40512/60000 [===================>..........] - ETA: 28s - loss: 0.2316 - categorical_accuracy: 0.9294
40544/60000 [===================>..........] - ETA: 28s - loss: 0.2315 - categorical_accuracy: 0.9294
40608/60000 [===================>..........] - ETA: 28s - loss: 0.2312 - categorical_accuracy: 0.9294
40672/60000 [===================>..........] - ETA: 28s - loss: 0.2309 - categorical_accuracy: 0.9296
40736/60000 [===================>..........] - ETA: 28s - loss: 0.2306 - categorical_accuracy: 0.9296
40800/60000 [===================>..........] - ETA: 28s - loss: 0.2303 - categorical_accuracy: 0.9297
40864/60000 [===================>..........] - ETA: 28s - loss: 0.2303 - categorical_accuracy: 0.9298
40928/60000 [===================>..........] - ETA: 28s - loss: 0.2301 - categorical_accuracy: 0.9299
40992/60000 [===================>..........] - ETA: 27s - loss: 0.2298 - categorical_accuracy: 0.9299
41056/60000 [===================>..........] - ETA: 27s - loss: 0.2297 - categorical_accuracy: 0.9299
41120/60000 [===================>..........] - ETA: 27s - loss: 0.2298 - categorical_accuracy: 0.9299
41184/60000 [===================>..........] - ETA: 27s - loss: 0.2296 - categorical_accuracy: 0.9300
41248/60000 [===================>..........] - ETA: 27s - loss: 0.2295 - categorical_accuracy: 0.9300
41312/60000 [===================>..........] - ETA: 27s - loss: 0.2295 - categorical_accuracy: 0.9300
41376/60000 [===================>..........] - ETA: 27s - loss: 0.2292 - categorical_accuracy: 0.9301
41440/60000 [===================>..........] - ETA: 27s - loss: 0.2291 - categorical_accuracy: 0.9301
41504/60000 [===================>..........] - ETA: 27s - loss: 0.2289 - categorical_accuracy: 0.9302
41568/60000 [===================>..........] - ETA: 27s - loss: 0.2287 - categorical_accuracy: 0.9303
41632/60000 [===================>..........] - ETA: 26s - loss: 0.2285 - categorical_accuracy: 0.9304
41696/60000 [===================>..........] - ETA: 26s - loss: 0.2281 - categorical_accuracy: 0.9305
41760/60000 [===================>..........] - ETA: 26s - loss: 0.2279 - categorical_accuracy: 0.9305
41792/60000 [===================>..........] - ETA: 26s - loss: 0.2278 - categorical_accuracy: 0.9306
41856/60000 [===================>..........] - ETA: 26s - loss: 0.2276 - categorical_accuracy: 0.9306
41920/60000 [===================>..........] - ETA: 26s - loss: 0.2274 - categorical_accuracy: 0.9307
41984/60000 [===================>..........] - ETA: 26s - loss: 0.2272 - categorical_accuracy: 0.9307
42048/60000 [====================>.........] - ETA: 26s - loss: 0.2272 - categorical_accuracy: 0.9308
42080/60000 [====================>.........] - ETA: 26s - loss: 0.2271 - categorical_accuracy: 0.9308
42112/60000 [====================>.........] - ETA: 26s - loss: 0.2269 - categorical_accuracy: 0.9308
42176/60000 [====================>.........] - ETA: 26s - loss: 0.2268 - categorical_accuracy: 0.9309
42240/60000 [====================>.........] - ETA: 26s - loss: 0.2267 - categorical_accuracy: 0.9309
42304/60000 [====================>.........] - ETA: 25s - loss: 0.2264 - categorical_accuracy: 0.9310
42368/60000 [====================>.........] - ETA: 25s - loss: 0.2263 - categorical_accuracy: 0.9311
42432/60000 [====================>.........] - ETA: 25s - loss: 0.2260 - categorical_accuracy: 0.9312
42496/60000 [====================>.........] - ETA: 25s - loss: 0.2258 - categorical_accuracy: 0.9312
42560/60000 [====================>.........] - ETA: 25s - loss: 0.2257 - categorical_accuracy: 0.9312
42624/60000 [====================>.........] - ETA: 25s - loss: 0.2254 - categorical_accuracy: 0.9313
42688/60000 [====================>.........] - ETA: 25s - loss: 0.2253 - categorical_accuracy: 0.9313
42752/60000 [====================>.........] - ETA: 25s - loss: 0.2251 - categorical_accuracy: 0.9314
42816/60000 [====================>.........] - ETA: 25s - loss: 0.2248 - categorical_accuracy: 0.9315
42880/60000 [====================>.........] - ETA: 25s - loss: 0.2245 - categorical_accuracy: 0.9316
42944/60000 [====================>.........] - ETA: 25s - loss: 0.2243 - categorical_accuracy: 0.9317
43008/60000 [====================>.........] - ETA: 24s - loss: 0.2240 - categorical_accuracy: 0.9317
43072/60000 [====================>.........] - ETA: 24s - loss: 0.2238 - categorical_accuracy: 0.9318
43136/60000 [====================>.........] - ETA: 24s - loss: 0.2236 - categorical_accuracy: 0.9319
43200/60000 [====================>.........] - ETA: 24s - loss: 0.2234 - categorical_accuracy: 0.9319
43264/60000 [====================>.........] - ETA: 24s - loss: 0.2234 - categorical_accuracy: 0.9320
43328/60000 [====================>.........] - ETA: 24s - loss: 0.2232 - categorical_accuracy: 0.9320
43392/60000 [====================>.........] - ETA: 24s - loss: 0.2229 - categorical_accuracy: 0.9321
43456/60000 [====================>.........] - ETA: 24s - loss: 0.2227 - categorical_accuracy: 0.9322
43488/60000 [====================>.........] - ETA: 24s - loss: 0.2226 - categorical_accuracy: 0.9322
43520/60000 [====================>.........] - ETA: 24s - loss: 0.2224 - categorical_accuracy: 0.9322
43584/60000 [====================>.........] - ETA: 24s - loss: 0.2222 - categorical_accuracy: 0.9323
43648/60000 [====================>.........] - ETA: 23s - loss: 0.2221 - categorical_accuracy: 0.9323
43712/60000 [====================>.........] - ETA: 23s - loss: 0.2218 - categorical_accuracy: 0.9324
43776/60000 [====================>.........] - ETA: 23s - loss: 0.2218 - categorical_accuracy: 0.9324
43840/60000 [====================>.........] - ETA: 23s - loss: 0.2216 - categorical_accuracy: 0.9325
43904/60000 [====================>.........] - ETA: 23s - loss: 0.2215 - categorical_accuracy: 0.9325
43968/60000 [====================>.........] - ETA: 23s - loss: 0.2213 - categorical_accuracy: 0.9326
44032/60000 [=====================>........] - ETA: 23s - loss: 0.2212 - categorical_accuracy: 0.9326
44096/60000 [=====================>........] - ETA: 23s - loss: 0.2211 - categorical_accuracy: 0.9326
44160/60000 [=====================>........] - ETA: 23s - loss: 0.2210 - categorical_accuracy: 0.9327
44224/60000 [=====================>........] - ETA: 23s - loss: 0.2208 - categorical_accuracy: 0.9327
44288/60000 [=====================>........] - ETA: 23s - loss: 0.2205 - categorical_accuracy: 0.9328
44352/60000 [=====================>........] - ETA: 22s - loss: 0.2202 - categorical_accuracy: 0.9329
44416/60000 [=====================>........] - ETA: 22s - loss: 0.2202 - categorical_accuracy: 0.9330
44480/60000 [=====================>........] - ETA: 22s - loss: 0.2200 - categorical_accuracy: 0.9330
44544/60000 [=====================>........] - ETA: 22s - loss: 0.2197 - categorical_accuracy: 0.9331
44608/60000 [=====================>........] - ETA: 22s - loss: 0.2195 - categorical_accuracy: 0.9332
44672/60000 [=====================>........] - ETA: 22s - loss: 0.2193 - categorical_accuracy: 0.9332
44736/60000 [=====================>........] - ETA: 22s - loss: 0.2195 - categorical_accuracy: 0.9333
44800/60000 [=====================>........] - ETA: 22s - loss: 0.2193 - categorical_accuracy: 0.9333
44864/60000 [=====================>........] - ETA: 22s - loss: 0.2192 - categorical_accuracy: 0.9334
44928/60000 [=====================>........] - ETA: 22s - loss: 0.2191 - categorical_accuracy: 0.9334
44992/60000 [=====================>........] - ETA: 22s - loss: 0.2191 - categorical_accuracy: 0.9334
45056/60000 [=====================>........] - ETA: 21s - loss: 0.2189 - categorical_accuracy: 0.9335
45120/60000 [=====================>........] - ETA: 21s - loss: 0.2186 - categorical_accuracy: 0.9336
45184/60000 [=====================>........] - ETA: 21s - loss: 0.2184 - categorical_accuracy: 0.9336
45248/60000 [=====================>........] - ETA: 21s - loss: 0.2182 - categorical_accuracy: 0.9337
45312/60000 [=====================>........] - ETA: 21s - loss: 0.2179 - categorical_accuracy: 0.9338
45376/60000 [=====================>........] - ETA: 21s - loss: 0.2178 - categorical_accuracy: 0.9338
45440/60000 [=====================>........] - ETA: 21s - loss: 0.2176 - categorical_accuracy: 0.9338
45504/60000 [=====================>........] - ETA: 21s - loss: 0.2176 - categorical_accuracy: 0.9338
45568/60000 [=====================>........] - ETA: 21s - loss: 0.2176 - categorical_accuracy: 0.9339
45632/60000 [=====================>........] - ETA: 21s - loss: 0.2176 - categorical_accuracy: 0.9339
45696/60000 [=====================>........] - ETA: 20s - loss: 0.2173 - categorical_accuracy: 0.9340
45760/60000 [=====================>........] - ETA: 20s - loss: 0.2170 - categorical_accuracy: 0.9341
45824/60000 [=====================>........] - ETA: 20s - loss: 0.2169 - categorical_accuracy: 0.9341
45888/60000 [=====================>........] - ETA: 20s - loss: 0.2167 - categorical_accuracy: 0.9342
45952/60000 [=====================>........] - ETA: 20s - loss: 0.2165 - categorical_accuracy: 0.9342
46016/60000 [======================>.......] - ETA: 20s - loss: 0.2164 - categorical_accuracy: 0.9342
46048/60000 [======================>.......] - ETA: 20s - loss: 0.2164 - categorical_accuracy: 0.9342
46112/60000 [======================>.......] - ETA: 20s - loss: 0.2162 - categorical_accuracy: 0.9342
46176/60000 [======================>.......] - ETA: 20s - loss: 0.2160 - categorical_accuracy: 0.9343
46240/60000 [======================>.......] - ETA: 20s - loss: 0.2158 - categorical_accuracy: 0.9344
46304/60000 [======================>.......] - ETA: 20s - loss: 0.2157 - categorical_accuracy: 0.9344
46368/60000 [======================>.......] - ETA: 19s - loss: 0.2156 - categorical_accuracy: 0.9344
46432/60000 [======================>.......] - ETA: 19s - loss: 0.2154 - categorical_accuracy: 0.9344
46496/60000 [======================>.......] - ETA: 19s - loss: 0.2153 - categorical_accuracy: 0.9345
46560/60000 [======================>.......] - ETA: 19s - loss: 0.2151 - categorical_accuracy: 0.9346
46624/60000 [======================>.......] - ETA: 19s - loss: 0.2152 - categorical_accuracy: 0.9346
46688/60000 [======================>.......] - ETA: 19s - loss: 0.2152 - categorical_accuracy: 0.9347
46752/60000 [======================>.......] - ETA: 19s - loss: 0.2151 - categorical_accuracy: 0.9347
46816/60000 [======================>.......] - ETA: 19s - loss: 0.2150 - categorical_accuracy: 0.9347
46880/60000 [======================>.......] - ETA: 19s - loss: 0.2148 - categorical_accuracy: 0.9348
46912/60000 [======================>.......] - ETA: 19s - loss: 0.2147 - categorical_accuracy: 0.9348
46976/60000 [======================>.......] - ETA: 19s - loss: 0.2146 - categorical_accuracy: 0.9349
47040/60000 [======================>.......] - ETA: 18s - loss: 0.2143 - categorical_accuracy: 0.9349
47104/60000 [======================>.......] - ETA: 18s - loss: 0.2141 - categorical_accuracy: 0.9350
47168/60000 [======================>.......] - ETA: 18s - loss: 0.2139 - categorical_accuracy: 0.9351
47232/60000 [======================>.......] - ETA: 18s - loss: 0.2136 - categorical_accuracy: 0.9351
47296/60000 [======================>.......] - ETA: 18s - loss: 0.2134 - categorical_accuracy: 0.9352
47360/60000 [======================>.......] - ETA: 18s - loss: 0.2134 - categorical_accuracy: 0.9352
47424/60000 [======================>.......] - ETA: 18s - loss: 0.2134 - categorical_accuracy: 0.9353
47488/60000 [======================>.......] - ETA: 18s - loss: 0.2133 - categorical_accuracy: 0.9353
47520/60000 [======================>.......] - ETA: 18s - loss: 0.2132 - categorical_accuracy: 0.9353
47584/60000 [======================>.......] - ETA: 18s - loss: 0.2130 - categorical_accuracy: 0.9354
47648/60000 [======================>.......] - ETA: 18s - loss: 0.2128 - categorical_accuracy: 0.9354
47712/60000 [======================>.......] - ETA: 18s - loss: 0.2127 - categorical_accuracy: 0.9354
47744/60000 [======================>.......] - ETA: 17s - loss: 0.2126 - categorical_accuracy: 0.9354
47808/60000 [======================>.......] - ETA: 17s - loss: 0.2124 - categorical_accuracy: 0.9355
47872/60000 [======================>.......] - ETA: 17s - loss: 0.2123 - categorical_accuracy: 0.9356
47936/60000 [======================>.......] - ETA: 17s - loss: 0.2121 - categorical_accuracy: 0.9356
48000/60000 [=======================>......] - ETA: 17s - loss: 0.2121 - categorical_accuracy: 0.9356
48064/60000 [=======================>......] - ETA: 17s - loss: 0.2118 - categorical_accuracy: 0.9357
48128/60000 [=======================>......] - ETA: 17s - loss: 0.2116 - categorical_accuracy: 0.9358
48192/60000 [=======================>......] - ETA: 17s - loss: 0.2116 - categorical_accuracy: 0.9357
48224/60000 [=======================>......] - ETA: 17s - loss: 0.2114 - categorical_accuracy: 0.9358
48288/60000 [=======================>......] - ETA: 17s - loss: 0.2113 - categorical_accuracy: 0.9358
48352/60000 [=======================>......] - ETA: 17s - loss: 0.2112 - categorical_accuracy: 0.9358
48416/60000 [=======================>......] - ETA: 16s - loss: 0.2111 - categorical_accuracy: 0.9359
48480/60000 [=======================>......] - ETA: 16s - loss: 0.2108 - categorical_accuracy: 0.9360
48544/60000 [=======================>......] - ETA: 16s - loss: 0.2107 - categorical_accuracy: 0.9360
48608/60000 [=======================>......] - ETA: 16s - loss: 0.2108 - categorical_accuracy: 0.9360
48672/60000 [=======================>......] - ETA: 16s - loss: 0.2108 - categorical_accuracy: 0.9360
48736/60000 [=======================>......] - ETA: 16s - loss: 0.2107 - categorical_accuracy: 0.9360
48800/60000 [=======================>......] - ETA: 16s - loss: 0.2106 - categorical_accuracy: 0.9361
48864/60000 [=======================>......] - ETA: 16s - loss: 0.2104 - categorical_accuracy: 0.9361
48928/60000 [=======================>......] - ETA: 16s - loss: 0.2103 - categorical_accuracy: 0.9362
48992/60000 [=======================>......] - ETA: 16s - loss: 0.2101 - categorical_accuracy: 0.9362
49056/60000 [=======================>......] - ETA: 16s - loss: 0.2101 - categorical_accuracy: 0.9362
49120/60000 [=======================>......] - ETA: 15s - loss: 0.2100 - categorical_accuracy: 0.9362
49184/60000 [=======================>......] - ETA: 15s - loss: 0.2100 - categorical_accuracy: 0.9363
49248/60000 [=======================>......] - ETA: 15s - loss: 0.2098 - categorical_accuracy: 0.9363
49312/60000 [=======================>......] - ETA: 15s - loss: 0.2096 - categorical_accuracy: 0.9364
49376/60000 [=======================>......] - ETA: 15s - loss: 0.2095 - categorical_accuracy: 0.9364
49440/60000 [=======================>......] - ETA: 15s - loss: 0.2093 - categorical_accuracy: 0.9365
49504/60000 [=======================>......] - ETA: 15s - loss: 0.2091 - categorical_accuracy: 0.9365
49568/60000 [=======================>......] - ETA: 15s - loss: 0.2090 - categorical_accuracy: 0.9365
49600/60000 [=======================>......] - ETA: 15s - loss: 0.2089 - categorical_accuracy: 0.9365
49664/60000 [=======================>......] - ETA: 15s - loss: 0.2088 - categorical_accuracy: 0.9366
49728/60000 [=======================>......] - ETA: 15s - loss: 0.2086 - categorical_accuracy: 0.9366
49760/60000 [=======================>......] - ETA: 15s - loss: 0.2085 - categorical_accuracy: 0.9367
49792/60000 [=======================>......] - ETA: 14s - loss: 0.2084 - categorical_accuracy: 0.9367
49856/60000 [=======================>......] - ETA: 14s - loss: 0.2083 - categorical_accuracy: 0.9367
49920/60000 [=======================>......] - ETA: 14s - loss: 0.2081 - categorical_accuracy: 0.9367
49984/60000 [=======================>......] - ETA: 14s - loss: 0.2079 - categorical_accuracy: 0.9368
50048/60000 [========================>.....] - ETA: 14s - loss: 0.2078 - categorical_accuracy: 0.9368
50112/60000 [========================>.....] - ETA: 14s - loss: 0.2077 - categorical_accuracy: 0.9368
50176/60000 [========================>.....] - ETA: 14s - loss: 0.2075 - categorical_accuracy: 0.9369
50240/60000 [========================>.....] - ETA: 14s - loss: 0.2073 - categorical_accuracy: 0.9369
50304/60000 [========================>.....] - ETA: 14s - loss: 0.2072 - categorical_accuracy: 0.9370
50368/60000 [========================>.....] - ETA: 14s - loss: 0.2070 - categorical_accuracy: 0.9370
50432/60000 [========================>.....] - ETA: 14s - loss: 0.2070 - categorical_accuracy: 0.9369
50496/60000 [========================>.....] - ETA: 13s - loss: 0.2069 - categorical_accuracy: 0.9370
50560/60000 [========================>.....] - ETA: 13s - loss: 0.2069 - categorical_accuracy: 0.9370
50624/60000 [========================>.....] - ETA: 13s - loss: 0.2068 - categorical_accuracy: 0.9371
50688/60000 [========================>.....] - ETA: 13s - loss: 0.2066 - categorical_accuracy: 0.9371
50752/60000 [========================>.....] - ETA: 13s - loss: 0.2064 - categorical_accuracy: 0.9371
50816/60000 [========================>.....] - ETA: 13s - loss: 0.2062 - categorical_accuracy: 0.9372
50880/60000 [========================>.....] - ETA: 13s - loss: 0.2060 - categorical_accuracy: 0.9372
50944/60000 [========================>.....] - ETA: 13s - loss: 0.2058 - categorical_accuracy: 0.9373
51008/60000 [========================>.....] - ETA: 13s - loss: 0.2057 - categorical_accuracy: 0.9373
51072/60000 [========================>.....] - ETA: 13s - loss: 0.2057 - categorical_accuracy: 0.9373
51136/60000 [========================>.....] - ETA: 12s - loss: 0.2055 - categorical_accuracy: 0.9374
51200/60000 [========================>.....] - ETA: 12s - loss: 0.2054 - categorical_accuracy: 0.9375
51264/60000 [========================>.....] - ETA: 12s - loss: 0.2053 - categorical_accuracy: 0.9375
51328/60000 [========================>.....] - ETA: 12s - loss: 0.2051 - categorical_accuracy: 0.9376
51392/60000 [========================>.....] - ETA: 12s - loss: 0.2051 - categorical_accuracy: 0.9376
51456/60000 [========================>.....] - ETA: 12s - loss: 0.2049 - categorical_accuracy: 0.9376
51488/60000 [========================>.....] - ETA: 12s - loss: 0.2048 - categorical_accuracy: 0.9377
51552/60000 [========================>.....] - ETA: 12s - loss: 0.2047 - categorical_accuracy: 0.9377
51616/60000 [========================>.....] - ETA: 12s - loss: 0.2045 - categorical_accuracy: 0.9377
51680/60000 [========================>.....] - ETA: 12s - loss: 0.2044 - categorical_accuracy: 0.9378
51744/60000 [========================>.....] - ETA: 12s - loss: 0.2042 - categorical_accuracy: 0.9378
51808/60000 [========================>.....] - ETA: 11s - loss: 0.2041 - categorical_accuracy: 0.9379
51872/60000 [========================>.....] - ETA: 11s - loss: 0.2040 - categorical_accuracy: 0.9379
51936/60000 [========================>.....] - ETA: 11s - loss: 0.2040 - categorical_accuracy: 0.9379
52000/60000 [=========================>....] - ETA: 11s - loss: 0.2038 - categorical_accuracy: 0.9380
52064/60000 [=========================>....] - ETA: 11s - loss: 0.2037 - categorical_accuracy: 0.9380
52128/60000 [=========================>....] - ETA: 11s - loss: 0.2035 - categorical_accuracy: 0.9380
52192/60000 [=========================>....] - ETA: 11s - loss: 0.2034 - categorical_accuracy: 0.9381
52256/60000 [=========================>....] - ETA: 11s - loss: 0.2032 - categorical_accuracy: 0.9381
52320/60000 [=========================>....] - ETA: 11s - loss: 0.2033 - categorical_accuracy: 0.9381
52384/60000 [=========================>....] - ETA: 11s - loss: 0.2031 - categorical_accuracy: 0.9381
52448/60000 [=========================>....] - ETA: 11s - loss: 0.2030 - categorical_accuracy: 0.9382
52512/60000 [=========================>....] - ETA: 10s - loss: 0.2029 - categorical_accuracy: 0.9382
52576/60000 [=========================>....] - ETA: 10s - loss: 0.2028 - categorical_accuracy: 0.9383
52640/60000 [=========================>....] - ETA: 10s - loss: 0.2027 - categorical_accuracy: 0.9383
52704/60000 [=========================>....] - ETA: 10s - loss: 0.2025 - categorical_accuracy: 0.9384
52768/60000 [=========================>....] - ETA: 10s - loss: 0.2023 - categorical_accuracy: 0.9384
52800/60000 [=========================>....] - ETA: 10s - loss: 0.2022 - categorical_accuracy: 0.9384
52864/60000 [=========================>....] - ETA: 10s - loss: 0.2021 - categorical_accuracy: 0.9384
52928/60000 [=========================>....] - ETA: 10s - loss: 0.2019 - categorical_accuracy: 0.9385
52992/60000 [=========================>....] - ETA: 10s - loss: 0.2017 - categorical_accuracy: 0.9386
53056/60000 [=========================>....] - ETA: 10s - loss: 0.2015 - categorical_accuracy: 0.9386
53120/60000 [=========================>....] - ETA: 10s - loss: 0.2014 - categorical_accuracy: 0.9386
53184/60000 [=========================>....] - ETA: 9s - loss: 0.2012 - categorical_accuracy: 0.9387 
53248/60000 [=========================>....] - ETA: 9s - loss: 0.2010 - categorical_accuracy: 0.9388
53312/60000 [=========================>....] - ETA: 9s - loss: 0.2009 - categorical_accuracy: 0.9388
53376/60000 [=========================>....] - ETA: 9s - loss: 0.2007 - categorical_accuracy: 0.9389
53440/60000 [=========================>....] - ETA: 9s - loss: 0.2004 - categorical_accuracy: 0.9390
53504/60000 [=========================>....] - ETA: 9s - loss: 0.2004 - categorical_accuracy: 0.9390
53568/60000 [=========================>....] - ETA: 9s - loss: 0.2003 - categorical_accuracy: 0.9390
53632/60000 [=========================>....] - ETA: 9s - loss: 0.2001 - categorical_accuracy: 0.9391
53696/60000 [=========================>....] - ETA: 9s - loss: 0.1999 - categorical_accuracy: 0.9391
53760/60000 [=========================>....] - ETA: 9s - loss: 0.1999 - categorical_accuracy: 0.9392
53824/60000 [=========================>....] - ETA: 9s - loss: 0.1999 - categorical_accuracy: 0.9391
53888/60000 [=========================>....] - ETA: 8s - loss: 0.1998 - categorical_accuracy: 0.9392
53952/60000 [=========================>....] - ETA: 8s - loss: 0.1996 - categorical_accuracy: 0.9392
54016/60000 [==========================>...] - ETA: 8s - loss: 0.1994 - categorical_accuracy: 0.9393
54048/60000 [==========================>...] - ETA: 8s - loss: 0.1995 - categorical_accuracy: 0.9393
54080/60000 [==========================>...] - ETA: 8s - loss: 0.1994 - categorical_accuracy: 0.9393
54144/60000 [==========================>...] - ETA: 8s - loss: 0.1992 - categorical_accuracy: 0.9394
54208/60000 [==========================>...] - ETA: 8s - loss: 0.1990 - categorical_accuracy: 0.9395
54240/60000 [==========================>...] - ETA: 8s - loss: 0.1989 - categorical_accuracy: 0.9395
54304/60000 [==========================>...] - ETA: 8s - loss: 0.1989 - categorical_accuracy: 0.9395
54368/60000 [==========================>...] - ETA: 8s - loss: 0.1989 - categorical_accuracy: 0.9395
54400/60000 [==========================>...] - ETA: 8s - loss: 0.1989 - categorical_accuracy: 0.9395
54464/60000 [==========================>...] - ETA: 8s - loss: 0.1987 - categorical_accuracy: 0.9396
54528/60000 [==========================>...] - ETA: 8s - loss: 0.1986 - categorical_accuracy: 0.9396
54592/60000 [==========================>...] - ETA: 7s - loss: 0.1984 - categorical_accuracy: 0.9397
54656/60000 [==========================>...] - ETA: 7s - loss: 0.1982 - categorical_accuracy: 0.9397
54720/60000 [==========================>...] - ETA: 7s - loss: 0.1981 - categorical_accuracy: 0.9398
54784/60000 [==========================>...] - ETA: 7s - loss: 0.1979 - categorical_accuracy: 0.9399
54848/60000 [==========================>...] - ETA: 7s - loss: 0.1978 - categorical_accuracy: 0.9399
54912/60000 [==========================>...] - ETA: 7s - loss: 0.1978 - categorical_accuracy: 0.9399
54976/60000 [==========================>...] - ETA: 7s - loss: 0.1977 - categorical_accuracy: 0.9399
55040/60000 [==========================>...] - ETA: 7s - loss: 0.1976 - categorical_accuracy: 0.9399
55072/60000 [==========================>...] - ETA: 7s - loss: 0.1975 - categorical_accuracy: 0.9400
55136/60000 [==========================>...] - ETA: 7s - loss: 0.1975 - categorical_accuracy: 0.9400
55200/60000 [==========================>...] - ETA: 7s - loss: 0.1974 - categorical_accuracy: 0.9400
55264/60000 [==========================>...] - ETA: 6s - loss: 0.1973 - categorical_accuracy: 0.9401
55328/60000 [==========================>...] - ETA: 6s - loss: 0.1971 - categorical_accuracy: 0.9401
55392/60000 [==========================>...] - ETA: 6s - loss: 0.1970 - categorical_accuracy: 0.9401
55456/60000 [==========================>...] - ETA: 6s - loss: 0.1969 - categorical_accuracy: 0.9402
55520/60000 [==========================>...] - ETA: 6s - loss: 0.1969 - categorical_accuracy: 0.9402
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1968 - categorical_accuracy: 0.9402
55584/60000 [==========================>...] - ETA: 6s - loss: 0.1967 - categorical_accuracy: 0.9403
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1966 - categorical_accuracy: 0.9403
55648/60000 [==========================>...] - ETA: 6s - loss: 0.1966 - categorical_accuracy: 0.9403
55712/60000 [==========================>...] - ETA: 6s - loss: 0.1966 - categorical_accuracy: 0.9402
55776/60000 [==========================>...] - ETA: 6s - loss: 0.1964 - categorical_accuracy: 0.9403
55840/60000 [==========================>...] - ETA: 6s - loss: 0.1962 - categorical_accuracy: 0.9403
55904/60000 [==========================>...] - ETA: 5s - loss: 0.1960 - categorical_accuracy: 0.9404
55968/60000 [==========================>...] - ETA: 5s - loss: 0.1961 - categorical_accuracy: 0.9404
56032/60000 [===========================>..] - ETA: 5s - loss: 0.1961 - categorical_accuracy: 0.9404
56096/60000 [===========================>..] - ETA: 5s - loss: 0.1960 - categorical_accuracy: 0.9404
56128/60000 [===========================>..] - ETA: 5s - loss: 0.1960 - categorical_accuracy: 0.9404
56160/60000 [===========================>..] - ETA: 5s - loss: 0.1960 - categorical_accuracy: 0.9404
56192/60000 [===========================>..] - ETA: 5s - loss: 0.1959 - categorical_accuracy: 0.9405
56256/60000 [===========================>..] - ETA: 5s - loss: 0.1959 - categorical_accuracy: 0.9404
56320/60000 [===========================>..] - ETA: 5s - loss: 0.1958 - categorical_accuracy: 0.9404
56384/60000 [===========================>..] - ETA: 5s - loss: 0.1957 - categorical_accuracy: 0.9405
56448/60000 [===========================>..] - ETA: 5s - loss: 0.1955 - categorical_accuracy: 0.9406
56512/60000 [===========================>..] - ETA: 5s - loss: 0.1955 - categorical_accuracy: 0.9406
56576/60000 [===========================>..] - ETA: 5s - loss: 0.1953 - categorical_accuracy: 0.9406
56640/60000 [===========================>..] - ETA: 4s - loss: 0.1951 - categorical_accuracy: 0.9407
56704/60000 [===========================>..] - ETA: 4s - loss: 0.1949 - categorical_accuracy: 0.9407
56768/60000 [===========================>..] - ETA: 4s - loss: 0.1949 - categorical_accuracy: 0.9408
56832/60000 [===========================>..] - ETA: 4s - loss: 0.1947 - categorical_accuracy: 0.9408
56896/60000 [===========================>..] - ETA: 4s - loss: 0.1945 - categorical_accuracy: 0.9409
56960/60000 [===========================>..] - ETA: 4s - loss: 0.1943 - categorical_accuracy: 0.9409
57024/60000 [===========================>..] - ETA: 4s - loss: 0.1942 - categorical_accuracy: 0.9409
57088/60000 [===========================>..] - ETA: 4s - loss: 0.1942 - categorical_accuracy: 0.9410
57152/60000 [===========================>..] - ETA: 4s - loss: 0.1942 - categorical_accuracy: 0.9410
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1941 - categorical_accuracy: 0.9410
57280/60000 [===========================>..] - ETA: 3s - loss: 0.1940 - categorical_accuracy: 0.9410
57344/60000 [===========================>..] - ETA: 3s - loss: 0.1939 - categorical_accuracy: 0.9411
57408/60000 [===========================>..] - ETA: 3s - loss: 0.1938 - categorical_accuracy: 0.9411
57472/60000 [===========================>..] - ETA: 3s - loss: 0.1936 - categorical_accuracy: 0.9412
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1935 - categorical_accuracy: 0.9412
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1935 - categorical_accuracy: 0.9412
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1933 - categorical_accuracy: 0.9413
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1932 - categorical_accuracy: 0.9414
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1930 - categorical_accuracy: 0.9414
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9414
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1929 - categorical_accuracy: 0.9414
57952/60000 [===========================>..] - ETA: 2s - loss: 0.1927 - categorical_accuracy: 0.9415
58016/60000 [============================>.] - ETA: 2s - loss: 0.1927 - categorical_accuracy: 0.9415
58080/60000 [============================>.] - ETA: 2s - loss: 0.1925 - categorical_accuracy: 0.9416
58144/60000 [============================>.] - ETA: 2s - loss: 0.1923 - categorical_accuracy: 0.9416
58208/60000 [============================>.] - ETA: 2s - loss: 0.1922 - categorical_accuracy: 0.9417
58272/60000 [============================>.] - ETA: 2s - loss: 0.1920 - categorical_accuracy: 0.9417
58304/60000 [============================>.] - ETA: 2s - loss: 0.1919 - categorical_accuracy: 0.9418
58368/60000 [============================>.] - ETA: 2s - loss: 0.1919 - categorical_accuracy: 0.9418
58432/60000 [============================>.] - ETA: 2s - loss: 0.1917 - categorical_accuracy: 0.9419
58496/60000 [============================>.] - ETA: 2s - loss: 0.1915 - categorical_accuracy: 0.9419
58560/60000 [============================>.] - ETA: 2s - loss: 0.1914 - categorical_accuracy: 0.9419
58624/60000 [============================>.] - ETA: 2s - loss: 0.1913 - categorical_accuracy: 0.9420
58688/60000 [============================>.] - ETA: 1s - loss: 0.1912 - categorical_accuracy: 0.9420
58752/60000 [============================>.] - ETA: 1s - loss: 0.1910 - categorical_accuracy: 0.9420
58816/60000 [============================>.] - ETA: 1s - loss: 0.1910 - categorical_accuracy: 0.9420
58880/60000 [============================>.] - ETA: 1s - loss: 0.1909 - categorical_accuracy: 0.9421
58912/60000 [============================>.] - ETA: 1s - loss: 0.1908 - categorical_accuracy: 0.9421
58976/60000 [============================>.] - ETA: 1s - loss: 0.1907 - categorical_accuracy: 0.9421
59040/60000 [============================>.] - ETA: 1s - loss: 0.1906 - categorical_accuracy: 0.9421
59104/60000 [============================>.] - ETA: 1s - loss: 0.1905 - categorical_accuracy: 0.9422
59168/60000 [============================>.] - ETA: 1s - loss: 0.1903 - categorical_accuracy: 0.9422
59232/60000 [============================>.] - ETA: 1s - loss: 0.1902 - categorical_accuracy: 0.9423
59296/60000 [============================>.] - ETA: 1s - loss: 0.1900 - categorical_accuracy: 0.9423
59360/60000 [============================>.] - ETA: 0s - loss: 0.1898 - categorical_accuracy: 0.9424
59392/60000 [============================>.] - ETA: 0s - loss: 0.1897 - categorical_accuracy: 0.9424
59456/60000 [============================>.] - ETA: 0s - loss: 0.1897 - categorical_accuracy: 0.9424
59520/60000 [============================>.] - ETA: 0s - loss: 0.1896 - categorical_accuracy: 0.9425
59584/60000 [============================>.] - ETA: 0s - loss: 0.1895 - categorical_accuracy: 0.9425
59648/60000 [============================>.] - ETA: 0s - loss: 0.1894 - categorical_accuracy: 0.9425
59712/60000 [============================>.] - ETA: 0s - loss: 0.1893 - categorical_accuracy: 0.9426
59776/60000 [============================>.] - ETA: 0s - loss: 0.1892 - categorical_accuracy: 0.9426
59840/60000 [============================>.] - ETA: 0s - loss: 0.1891 - categorical_accuracy: 0.9426
59904/60000 [============================>.] - ETA: 0s - loss: 0.1889 - categorical_accuracy: 0.9427
59968/60000 [============================>.] - ETA: 0s - loss: 0.1889 - categorical_accuracy: 0.9427
60000/60000 [==============================] - 91s 2ms/step - loss: 0.1889 - categorical_accuracy: 0.9427 - val_loss: 0.0505 - val_categorical_accuracy: 0.9842

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  224/10000 [..............................] - ETA: 5s 
  384/10000 [>.............................] - ETA: 4s
  576/10000 [>.............................] - ETA: 3s
  768/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 3s
 1152/10000 [==>...........................] - ETA: 3s
 1344/10000 [===>..........................] - ETA: 2s
 1568/10000 [===>..........................] - ETA: 2s
 1760/10000 [====>.........................] - ETA: 2s
 1952/10000 [====>.........................] - ETA: 2s
 2144/10000 [=====>........................] - ETA: 2s
 2336/10000 [======>.......................] - ETA: 2s
 2528/10000 [======>.......................] - ETA: 2s
 2688/10000 [=======>......................] - ETA: 2s
 2880/10000 [=======>......................] - ETA: 2s
 3072/10000 [========>.....................] - ETA: 2s
 3296/10000 [========>.....................] - ETA: 1s
 3488/10000 [=========>....................] - ETA: 1s
 3680/10000 [==========>...................] - ETA: 1s
 3872/10000 [==========>...................] - ETA: 1s
 4064/10000 [===========>..................] - ETA: 1s
 4256/10000 [===========>..................] - ETA: 1s
 4448/10000 [============>.................] - ETA: 1s
 4640/10000 [============>.................] - ETA: 1s
 4832/10000 [=============>................] - ETA: 1s
 5024/10000 [==============>...............] - ETA: 1s
 5216/10000 [==============>...............] - ETA: 1s
 5408/10000 [===============>..............] - ETA: 1s
 5568/10000 [===============>..............] - ETA: 1s
 5760/10000 [================>.............] - ETA: 1s
 5952/10000 [================>.............] - ETA: 1s
 6112/10000 [=================>............] - ETA: 1s
 6304/10000 [=================>............] - ETA: 1s
 6496/10000 [==================>...........] - ETA: 1s
 6688/10000 [===================>..........] - ETA: 0s
 6880/10000 [===================>..........] - ETA: 0s
 7072/10000 [====================>.........] - ETA: 0s
 7296/10000 [====================>.........] - ETA: 0s
 7488/10000 [=====================>........] - ETA: 0s
 7680/10000 [======================>.......] - ETA: 0s
 7872/10000 [======================>.......] - ETA: 0s
 8064/10000 [=======================>......] - ETA: 0s
 8288/10000 [=======================>......] - ETA: 0s
 8512/10000 [========================>.....] - ETA: 0s
 8704/10000 [=========================>....] - ETA: 0s
 8928/10000 [=========================>....] - ETA: 0s
 9120/10000 [==========================>...] - ETA: 0s
 9280/10000 [==========================>...] - ETA: 0s
 9472/10000 [===========================>..] - ETA: 0s
 9664/10000 [===========================>..] - ETA: 0s
 9856/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 281us/step
[[1.5001920e-08 2.4444600e-07 3.3586607e-06 ... 9.9997842e-01
  2.9826229e-08 3.0445452e-07]
 [8.9966952e-06 9.7666407e-06 9.9997628e-01 ... 1.3982070e-08
  9.3410466e-07 2.8054675e-10]
 [7.6193751e-06 9.9981171e-01 2.5263987e-05 ... 2.7181364e-05
  2.9753070e-05 1.5146226e-06]
 ...
 [5.3931095e-09 1.4367649e-07 1.8160218e-08 ... 6.9087747e-07
  5.0080803e-06 1.8550612e-05]
 [8.7719519e-07 7.3472641e-08 2.7361510e-08 ... 4.3292032e-09
  6.8231882e-04 4.8206437e-07]
 [2.9862969e-05 1.4047889e-06 7.3149036e-05 ... 3.5334460e-09
  1.0660858e-06 4.1673122e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.050488999673677606, 'accuracy_test:': 0.9842000007629395}

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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   c4cba42..2ac15bd  master     -> origin/master
Updating c4cba42..2ac15bd
Fast-forward
 .../20200521/list_log_pullrequest_20200521.md      |   2 +-
 error_list/20200521/list_log_testall_20200521.md   | 433 +++++++++++++++++++++
 2 files changed, 434 insertions(+), 1 deletion(-)
[master 3d56805] ml_store
 1 file changed, 1163 insertions(+)
To github.com:arita37/mlmodels_store.git
   2ac15bd..3d56805  master -> master





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
{'loss': 0.39020708203315735, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-21 08:28:32.327597: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master ca698c8] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   3d56805..ca698c8  master -> master





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
[master 0ab9e59] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   ca698c8..0ab9e59  master -> master





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
 40%|████      | 2/5 [00:21<00:32, 10.92s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8381788077651547, 'learning_rate': 0.019355040998387397, 'min_data_in_leaf': 27, 'num_leaves': 29} and reward: 0.3914
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xd2\\\\\xf1\xa6\xcbX\r\x00\x00\x00learning_rateq\x02G?\x93\xd1\xce\xd0g;\x8aX\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K\x1du.' and reward: 0.3914
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xea\xd2\\\\\xf1\xa6\xcbX\r\x00\x00\x00learning_rateq\x02G?\x93\xd1\xce\xd0g;\x8aX\x10\x00\x00\x00min_data_in_leafq\x03K\x1bX\n\x00\x00\x00num_leavesq\x04K\x1du.' and reward: 0.3914
 60%|██████    | 3/5 [00:41<00:27, 13.54s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.8963646086483914, 'learning_rate': 0.01514947246036236, 'min_data_in_leaf': 3, 'num_leaves': 38} and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xaf\x04\xd4\xed\xf9HX\r\x00\x00\x00learning_rateq\x02G?\x8f\x06\xaf\xc6&\xb9\xc2X\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K&u.' and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xaf\x04\xd4\xed\xf9HX\r\x00\x00\x00learning_rateq\x02G?\x8f\x06\xaf\xc6&\xb9\xc2X\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K&u.' and reward: 0.3888
 80%|████████  | 4/5 [01:04<00:16, 16.50s/it] 80%|████████  | 4/5 [01:04<00:16, 16.23s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9924367796068528, 'learning_rate': 0.01090267003327339, 'min_data_in_leaf': 10, 'num_leaves': 56} and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\xc2\n\xc6\xf8I\xabX\r\x00\x00\x00learning_rateq\x02G?\x86T#\x99\xdb\x1c\xc2X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K8u.' and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\xc2\n\xc6\xf8I\xabX\r\x00\x00\x00learning_rateq\x02G?\x86T#\x99\xdb\x1c\xc2X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K8u.' and reward: 0.3888
Time for Gradient Boosting hyperparameter optimization: 94.94735193252563
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8381788077651547, 'learning_rate': 0.019355040998387397, 'min_data_in_leaf': 27, 'num_leaves': 29}
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:51<01:17, 25.80s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
distributed.utils_perf - WARNING - full garbage collections took 11% CPU time recently (threshold: 10%)
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.007841696730120475, 'embedding_size_factor': 1.1159908691926712, 'layers.choice': 3, 'learning_rate': 0.005450262791298418, 'network_type.choice': 1, 'use_batchnorm.choice': 0, 'weight_decay': 0.00011286865826644362} and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\x80\x0fN\xb8\x02\x15\x81X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xdb\x19=\xdd\x12\xfeX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?vS\x03\xc7\x17z\xd6X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\x1d\x96|\xc8\xb3\xa4\x0eu.' and reward: 0.3888
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\x80\x0fN\xb8\x02\x15\x81X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf1\xdb\x19=\xdd\x12\xfeX\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?vS\x03\xc7\x17z\xd6X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G?\x1d\x96|\xc8\xb3\xa4\x0eu.' and reward: 0.3888
 60%|██████    | 3/5 [01:47<01:09, 34.75s/it] 60%|██████    | 3/5 [01:47<01:11, 35.75s/it]
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4651141323492256, 'embedding_size_factor': 0.6740145120712027, 'layers.choice': 2, 'learning_rate': 0.005119713588303506, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 2.3746291581740143e-11} and reward: 0.3728
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\xc4n\x10\xd6:\xd0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x91\x86\xe1\xcc\x01\xbeX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?t\xf8h\xa6\xd3\x8fVX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xba\x1b\xfc\xa3\x82\x0f,u.' and reward: 0.3728
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdd\xc4n\x10\xd6:\xd0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xe5\x91\x86\xe1\xcc\x01\xbeX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?t\xf8h\xa6\xd3\x8fVX\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xba\x1b\xfc\xa3\x82\x0f,u.' and reward: 0.3728
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 160.36514520645142
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.76s of the -139.96s of remaining time.
Ensemble size: 47
Ensemble weights: 
[0.0212766  0.10638298 0.21276596 0.         0.29787234 0.36170213
 0.        ]
	0.4018	 = Validation accuracy score
	1.6s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 261.61s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fd7c861ada0>

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
   0ab9e59..d1b0efb  master     -> origin/master
Updating 0ab9e59..d1b0efb
Fast-forward
 error_list/20200521/list_log_testall_20200521.md | 175 +++++++++++++++++++++++
 1 file changed, 175 insertions(+)
[master 6fc04c3] ml_store
 1 file changed, 309 insertions(+)
To github.com:arita37/mlmodels_store.git
   d1b0efb..6fc04c3  master -> master





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
[master 519a3a0] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   6fc04c3..519a3a0  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.87it/s, avg_epoch_loss=5.26]
INFO:root:Epoch[0] Elapsed time 2.586 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.257501
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.257501459121704 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f25469aa3c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f25469aa3c8>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 108.79it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1030.39208984375,
    "abs_error": 363.9076843261719,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.4112340561330456,
    "sMAPE": 0.5064963519328787,
    "MSIS": 96.44937033363946,
    "QuantileLoss[0.5]": 363.9076843261719,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.099721024391314,
    "NRMSE": 0.6757836005135014,
    "ND": 0.6384345339055647,
    "wQuantileLoss[0.5]": 0.6384345339055647,
    "mean_wQuantileLoss": 0.6384345339055647,
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
100%|██████████| 10/10 [00:01<00:00,  7.95it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.259 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f251ab7aeb8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f251ab7aeb8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 153.71it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.38it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 1.861 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.226043
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.226042938232422 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f251ac80f60>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f251ac80f60>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 159.32it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 307.28985595703125,
    "abs_error": 185.22018432617188,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.227259647340143,
    "sMAPE": 0.3020596624405546,
    "MSIS": 49.09038751126925,
    "QuantileLoss[0.5]": 185.2201805114746,
    "Coverage[0.5]": 0.75,
    "RMSE": 17.529684993091898,
    "NRMSE": 0.36904599985456626,
    "ND": 0.32494769180030153,
    "wQuantileLoss[0.5]": 0.3249476851078502,
    "mean_wQuantileLoss": 0.3249476851078502,
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
 30%|███       | 3/10 [00:12<00:29,  4.25s/it, avg_epoch_loss=6.92] 60%|██████    | 6/10 [00:24<00:16,  4.10s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:35<00:04,  4.00s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:39<00:00,  3.91s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 39.121 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.869233
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.869232606887818 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f251ac71f98>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f251ac71f98>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 130.88it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 54135.171875,
    "abs_error": 2746.7880859375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.200080028693893,
    "sMAPE": 1.4164307753050391,
    "MSIS": 728.0032011477558,
    "QuantileLoss[0.5]": 2746.7880249023438,
    "Coverage[0.5]": 1.0,
    "RMSE": 232.66966255831463,
    "NRMSE": 4.898308685438202,
    "ND": 4.818926466557017,
    "wQuantileLoss[0.5]": 4.818926359477796,
    "mean_wQuantileLoss": 4.818926359477796,
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
100%|██████████| 10/10 [00:00<00:00, 56.27it/s, avg_epoch_loss=5.09]
INFO:root:Epoch[0] Elapsed time 0.178 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.093898
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.093898487091065 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f251acbd898>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f251acbd898>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 158.10it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 346.9827067057292,
    "abs_error": 205.4649200439453,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.3614002503628466,
    "sMAPE": 0.3345723281661513,
    "MSIS": 54.4560092056821,
    "QuantileLoss[0.5]": 205.4649200439453,
    "Coverage[0.5]": 0.75,
    "RMSE": 18.627471828075038,
    "NRMSE": 0.392157301643685,
    "ND": 0.3604647720069216,
    "wQuantileLoss[0.5]": 0.3604647720069216,
    "mean_wQuantileLoss": 0.3604647720069216,
    "MAE_Coverage": 0.25
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
100%|██████████| 10/10 [00:01<00:00,  8.28it/s, avg_epoch_loss=144]
INFO:root:Epoch[0] Elapsed time 1.208 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=144.412806
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 144.41280616614463 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f25181ab748>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f25181ab748>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 157.41it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 2434.5670123443265,
    "abs_error": 573.7377910258112,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 3.8015578142395565,
    "sMAPE": 1.9586154749802402,
    "MSIS": 152.0623125695823,
    "QuantileLoss[0.5]": 573.7377910258112,
    "Coverage[0.5]": 0.0,
    "RMSE": 49.34133168393742,
    "NRMSE": 1.0387648775565772,
    "ND": 1.0065575281154584,
    "wQuantileLoss[0.5]": 1.0065575281154584,
    "mean_wQuantileLoss": 1.0065575281154584,
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
 10%|█         | 1/10 [01:53<17:05, 113.92s/it, avg_epoch_loss=0.488] 20%|██        | 2/10 [04:50<17:41, 132.69s/it, avg_epoch_loss=0.47]  30%|███       | 3/10 [08:11<17:51, 153.07s/it, avg_epoch_loss=0.453] 40%|████      | 4/10 [11:29<16:39, 166.63s/it, avg_epoch_loss=0.437] 50%|█████     | 5/10 [15:02<15:02, 180.57s/it, avg_epoch_loss=0.424] 60%|██████    | 6/10 [18:06<12:06, 181.63s/it, avg_epoch_loss=0.414] 70%|███████   | 7/10 [21:28<09:23, 187.75s/it, avg_epoch_loss=0.407] 80%|████████  | 8/10 [25:05<06:33, 196.55s/it, avg_epoch_loss=0.404] 90%|█████████ | 9/10 [28:25<03:17, 197.67s/it, avg_epoch_loss=0.401]100%|██████████| 10/10 [31:34<00:00, 194.96s/it, avg_epoch_loss=0.399]100%|██████████| 10/10 [31:34<00:00, 189.46s/it, avg_epoch_loss=0.399]
INFO:root:Epoch[0] Elapsed time 1894.573 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.398938
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3989381194114685 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f2518241160>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f2518241160>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 20.07it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 137.49435424804688,
    "abs_error": 102.26634979248047,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 0.6776117021893543,
    "sMAPE": 0.177204681090722,
    "MSIS": 27.104465256662994,
    "QuantileLoss[0.5]": 102.26634216308594,
    "Coverage[0.5]": 0.4166666666666667,
    "RMSE": 11.725798661415217,
    "NRMSE": 0.2468589191876888,
    "ND": 0.17941464875873767,
    "wQuantileLoss[0.5]": 0.17941463537383498,
    "mean_wQuantileLoss": 0.17941463537383498,
    "MAE_Coverage": 0.08333333333333331
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
   519a3a0..83810ef  master     -> origin/master
Updating 519a3a0..83810ef
Fast-forward
 error_list/20200521/list_log_pullrequest_20200521.md | 2 +-
 error_list/20200521/list_log_testall_20200521.md     | 7 +++++++
 2 files changed, 8 insertions(+), 1 deletion(-)
[master 3d79822] ml_store
 1 file changed, 506 insertions(+)
To github.com:arita37/mlmodels_store.git
   83810ef..3d79822  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f4aa36f0748> 

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
[master 4488d8c] ml_store
 1 file changed, 109 insertions(+)
To github.com:arita37/mlmodels_store.git
   3d79822..4488d8c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 1.09488485e+00 -6.96245395e-02 -1.16444148e-01  3.53870427e-01
  -1.44189096e+00 -1.86955017e-01  1.29118890e+00 -1.53236162e-01
  -2.43250851e+00 -2.27729800e+00]
 [ 7.90323893e-01  1.61336137e+00 -2.09424782e+00 -3.74804687e-01
   9.15884042e-01 -7.49969617e-01  3.10272288e-01  2.05462410e+00
   5.34095368e-02 -2.28765829e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 1.16755486e+00  3.53600971e-02  7.14789597e-01 -1.53879325e+00
   1.10863359e+00 -4.47895185e-01 -1.75592564e+00  6.17985534e-01
  -1.84176326e-01  8.52704062e-01]
 [ 9.83799588e-01 -4.07240024e-01  9.32721414e-01  1.60564992e-01
  -1.27861800e+00 -1.20149976e-01  1.99759555e-01  3.85602292e-01
   7.18290736e-01 -5.30119800e-01]
 [ 1.07258847e+00 -5.86523939e-01 -1.34267579e+00 -1.23685338e+00
   1.24328724e+00  8.75838928e-01 -3.26499498e-01  6.23362177e-01
  -4.34956683e-01  1.11438298e+00]
 [ 1.27991386e+00 -8.71422066e-01 -3.24032329e-01 -8.64829941e-01
  -9.68539694e-01  6.08749082e-01  5.07984337e-01  5.61638097e-01
   1.51475038e+00 -1.51107661e+00]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 7.75285326e-01  1.47016034e+00  1.03298378e+00 -8.70008223e-01
   7.86556511e-01  3.69190470e-01 -1.43195745e-01  8.53282186e-01
  -1.39711730e-01 -2.22414029e-01]
 [ 4.41189807e-01  4.79852371e-01 -1.92003697e-01 -1.55269878e+00
  -1.88873982e+00  5.78464420e-01  3.98598388e-01 -9.61263599e-01
  -1.45832446e+00 -3.05376438e+00]
 [ 1.02242019e+00  1.85300949e+00  6.44353666e-01  1.42251373e-01
   1.15080755e+00  5.13505480e-01 -4.59942831e-01  3.72456852e-01
  -1.48489803e-01  3.71670291e-01]
 [ 1.18468624e+00 -1.00016919e+00 -5.93843067e-01  1.04499441e+00
   9.65482331e-01  6.08514698e-01 -6.25342001e-01 -6.93286967e-02
  -1.08392067e-01 -3.43900709e-01]
 [ 6.23629500e-01  9.86352180e-01  1.45391758e+00 -4.66154857e-01
   9.36403332e-01  1.38499134e+00  3.49435894e-02 -1.07296428e+00
   4.95158611e-01  6.61681076e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f7688636da0>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f76a29a55f8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 1.66752297  1.22372221 -0.4599301  -0.0593679  -0.493857    1.4489894
  -1.18110317 -0.47758085  0.02599999 -0.79079995]
 [ 0.47330777 -0.97326759 -0.22814069  0.17516773 -1.01366961 -0.05348369
   0.39378773 -0.18306199 -0.2210289   0.58033011]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 0.84806927  0.45194604  0.63019567 -1.57915629  0.82798737 -0.82862798
  -0.10534471  0.52887975 -2.23708651 -0.4148469 ]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.10593645 -0.73728963  0.65032321  0.16466507 -1.53556118  0.77817418
   0.05031709  0.30981676  1.05132077  0.6065484 ]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 1.09488485 -0.06962454 -0.11644415  0.35387043 -1.44189096 -0.18695502
   1.2911889  -0.15323616 -2.43250851 -2.277298  ]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 0.62153099 -1.50957268 -0.10193204 -1.08071069 -1.13742855  0.725474
   0.7980638  -0.03917826 -0.22875417  0.74335654]]
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
[[ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 9.71395338e-01  7.13049050e-01  1.76041518e+00  1.30620607e+00
   1.05765490e+00 -6.04602969e-01  1.28376990e-01  6.36583409e-01
   1.40925339e+00  9.66539250e-01]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 1.02242019e+00  1.85300949e+00  6.44353666e-01  1.42251373e-01
   1.15080755e+00  5.13505480e-01 -4.59942831e-01  3.72456852e-01
  -1.48489803e-01  3.71670291e-01]
 [ 1.34740825e+00  7.33023232e-01  8.38634747e-01 -1.89881206e+00
  -5.42459922e-01 -1.11711069e+00 -1.09715436e+00 -5.08972278e-01
  -1.66485955e-01 -1.03918232e+00]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 8.59823751e-01  1.71957132e-01 -3.48984191e-01  4.90561044e-01
  -1.15649503e+00 -1.39528303e+00  6.14726276e-01 -5.22356465e-01
  -3.69255902e-01 -9.77773002e-01]
 [ 6.18390447e-01 -7.25214926e-01  4.00084198e-03  1.53653633e+00
  -1.03048932e+00 -3.75008758e-04  5.31163793e-01  1.29354962e+00
  -4.38997664e-01  3.21265914e-01]
 [ 1.25704434e+00 -1.82391985e+00 -6.12406973e-01  1.16707517e+00
  -6.23732812e-01 -3.96687001e-02  8.16043684e-01  8.85825799e-01
   1.89861649e-01  3.93109245e-01]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
  -7.97269785e-01  6.54580153e-01]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 9.29250600e-01 -1.10657307e+00 -1.95816909e+00 -3.59224096e-01
  -1.21258781e+00  5.05381903e-01  5.42645295e-01  1.21794090e+00
  -1.94068096e+00  6.77807571e-01]
 [ 1.12062155e+00 -7.02920403e-01 -1.22957425e+00  7.25550518e-01
  -1.18013412e+00 -3.24204219e-01  1.10223673e+00  8.14343129e-01
   7.80469930e-01  1.10861676e+00]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]
 [ 9.26869810e-01  3.92334911e-01 -4.23478297e-01  4.48380651e-01
  -1.09230828e+00  1.12532350e+00 -9.48439656e-01  1.04053390e-01
   5.28003422e-01  1.00796648e+00]
 [ 9.64572049e-01 -1.06793987e-01  1.12232832e+00  1.45142926e+00
   1.21828168e+00 -6.18036848e-01  4.38166347e-01 -2.03720123e+00
  -1.94258918e+00 -9.97019796e-01]
 [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
   9.10423603e-01  1.21987438e+00]
 [ 1.24549398e+00 -7.22391905e-01  1.11813340e+00  1.09899633e+00
   1.00277655e+00 -9.01634490e-01 -5.32234021e-01 -8.22467189e-01
   7.21711292e-01  6.74396105e-01]
 [ 1.66752297e+00  1.22372221e+00 -4.59930104e-01 -5.93679025e-02
  -4.93856997e-01  1.44898940e+00 -1.18110317e+00 -4.77580855e-01
   2.59999942e-02 -7.90799954e-01]
 [ 8.88838813e-01  1.03368687e+00 -4.97025792e-02  8.08844360e-01
   8.14051347e-01  1.78975468e+00  1.14690038e+00  4.51284016e-01
  -1.68405999e+00  4.66643267e-01]]
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
[master 3308113] ml_store
 1 file changed, 297 insertions(+)
To github.com:arita37/mlmodels_store.git
   4488d8c..3308113  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140438518251472
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140438518251248
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140438518250016
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140438518249568
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140438518249064
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140438518248728

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
grad_step = 000000, loss = 0.672147
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.526282
grad_step = 000002, loss = 0.404421
grad_step = 000003, loss = 0.271820
grad_step = 000004, loss = 0.134848
grad_step = 000005, loss = 0.047265
grad_step = 000006, loss = 0.085127
grad_step = 000007, loss = 0.116155
grad_step = 000008, loss = 0.069892
grad_step = 000009, loss = 0.023379
grad_step = 000010, loss = 0.009923
grad_step = 000011, loss = 0.021654
grad_step = 000012, loss = 0.038222
grad_step = 000013, loss = 0.047557
grad_step = 000014, loss = 0.047047
grad_step = 000015, loss = 0.038761
grad_step = 000016, loss = 0.026828
grad_step = 000017, loss = 0.016099
grad_step = 000018, loss = 0.010651
grad_step = 000019, loss = 0.011801
grad_step = 000020, loss = 0.016228
grad_step = 000021, loss = 0.018219
grad_step = 000022, loss = 0.015431
grad_step = 000023, loss = 0.010267
grad_step = 000024, loss = 0.006532
grad_step = 000025, loss = 0.006040
grad_step = 000026, loss = 0.008056
grad_step = 000027, loss = 0.010594
grad_step = 000028, loss = 0.012021
grad_step = 000029, loss = 0.011701
grad_step = 000030, loss = 0.009945
grad_step = 000031, loss = 0.007676
grad_step = 000032, loss = 0.005945
grad_step = 000033, loss = 0.005420
grad_step = 000034, loss = 0.005981
grad_step = 000035, loss = 0.006785
grad_step = 000036, loss = 0.006971
grad_step = 000037, loss = 0.006342
grad_step = 000038, loss = 0.005388
grad_step = 000039, loss = 0.004748
grad_step = 000040, loss = 0.004715
grad_step = 000041, loss = 0.005142
grad_step = 000042, loss = 0.005633
grad_step = 000043, loss = 0.005831
grad_step = 000044, loss = 0.005614
grad_step = 000045, loss = 0.005119
grad_step = 000046, loss = 0.004623
grad_step = 000047, loss = 0.004365
grad_step = 000048, loss = 0.004392
grad_step = 000049, loss = 0.004556
grad_step = 000050, loss = 0.004648
grad_step = 000051, loss = 0.004556
grad_step = 000052, loss = 0.004332
grad_step = 000053, loss = 0.004125
grad_step = 000054, loss = 0.004050
grad_step = 000055, loss = 0.004106
grad_step = 000056, loss = 0.004200
grad_step = 000057, loss = 0.004228
grad_step = 000058, loss = 0.004144
grad_step = 000059, loss = 0.003988
grad_step = 000060, loss = 0.003842
grad_step = 000061, loss = 0.003768
grad_step = 000062, loss = 0.003766
grad_step = 000063, loss = 0.003783
grad_step = 000064, loss = 0.003759
grad_step = 000065, loss = 0.003682
grad_step = 000066, loss = 0.003589
grad_step = 000067, loss = 0.003525
grad_step = 000068, loss = 0.003508
grad_step = 000069, loss = 0.003512
grad_step = 000070, loss = 0.003496
grad_step = 000071, loss = 0.003441
grad_step = 000072, loss = 0.003362
grad_step = 000073, loss = 0.003294
grad_step = 000074, loss = 0.003252
grad_step = 000075, loss = 0.003228
grad_step = 000076, loss = 0.003197
grad_step = 000077, loss = 0.003149
grad_step = 000078, loss = 0.003093
grad_step = 000079, loss = 0.003044
grad_step = 000080, loss = 0.003011
grad_step = 000081, loss = 0.002986
grad_step = 000082, loss = 0.002953
grad_step = 000083, loss = 0.002909
grad_step = 000084, loss = 0.002863
grad_step = 000085, loss = 0.002824
grad_step = 000086, loss = 0.002797
grad_step = 000087, loss = 0.002770
grad_step = 000088, loss = 0.002737
grad_step = 000089, loss = 0.002704
grad_step = 000090, loss = 0.002678
grad_step = 000091, loss = 0.002657
grad_step = 000092, loss = 0.002634
grad_step = 000093, loss = 0.002607
grad_step = 000094, loss = 0.002577
grad_step = 000095, loss = 0.002546
grad_step = 000096, loss = 0.002521
grad_step = 000097, loss = 0.002491
grad_step = 000098, loss = 0.002457
grad_step = 000099, loss = 0.002425
grad_step = 000100, loss = 0.002389
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.002357
grad_step = 000102, loss = 0.002322
grad_step = 000103, loss = 0.002285
grad_step = 000104, loss = 0.002251
grad_step = 000105, loss = 0.002214
grad_step = 000106, loss = 0.002177
grad_step = 000107, loss = 0.002141
grad_step = 000108, loss = 0.002104
grad_step = 000109, loss = 0.002066
grad_step = 000110, loss = 0.002029
grad_step = 000111, loss = 0.001990
grad_step = 000112, loss = 0.001951
grad_step = 000113, loss = 0.001911
grad_step = 000114, loss = 0.001871
grad_step = 000115, loss = 0.001830
grad_step = 000116, loss = 0.001790
grad_step = 000117, loss = 0.001750
grad_step = 000118, loss = 0.001710
grad_step = 000119, loss = 0.001671
grad_step = 000120, loss = 0.001633
grad_step = 000121, loss = 0.001596
grad_step = 000122, loss = 0.001559
grad_step = 000123, loss = 0.001523
grad_step = 000124, loss = 0.001489
grad_step = 000125, loss = 0.001457
grad_step = 000126, loss = 0.001425
grad_step = 000127, loss = 0.001443
grad_step = 000128, loss = 0.001436
grad_step = 000129, loss = 0.001393
grad_step = 000130, loss = 0.001340
grad_step = 000131, loss = 0.001360
grad_step = 000132, loss = 0.001306
grad_step = 000133, loss = 0.001301
grad_step = 000134, loss = 0.001295
grad_step = 000135, loss = 0.001262
grad_step = 000136, loss = 0.001262
grad_step = 000137, loss = 0.001244
grad_step = 000138, loss = 0.001235
grad_step = 000139, loss = 0.001225
grad_step = 000140, loss = 0.001211
grad_step = 000141, loss = 0.001198
grad_step = 000142, loss = 0.001190
grad_step = 000143, loss = 0.001170
grad_step = 000144, loss = 0.001166
grad_step = 000145, loss = 0.001144
grad_step = 000146, loss = 0.001138
grad_step = 000147, loss = 0.001121
grad_step = 000148, loss = 0.001109
grad_step = 000149, loss = 0.001093
grad_step = 000150, loss = 0.001081
grad_step = 000151, loss = 0.001066
grad_step = 000152, loss = 0.001051
grad_step = 000153, loss = 0.001036
grad_step = 000154, loss = 0.001020
grad_step = 000155, loss = 0.001008
grad_step = 000156, loss = 0.000989
grad_step = 000157, loss = 0.000977
grad_step = 000158, loss = 0.000958
grad_step = 000159, loss = 0.000946
grad_step = 000160, loss = 0.000927
grad_step = 000161, loss = 0.000912
grad_step = 000162, loss = 0.000896
grad_step = 000163, loss = 0.000881
grad_step = 000164, loss = 0.000863
grad_step = 000165, loss = 0.000848
grad_step = 000166, loss = 0.000831
grad_step = 000167, loss = 0.000816
grad_step = 000168, loss = 0.000801
grad_step = 000169, loss = 0.000785
grad_step = 000170, loss = 0.000771
grad_step = 000171, loss = 0.000755
grad_step = 000172, loss = 0.000740
grad_step = 000173, loss = 0.000726
grad_step = 000174, loss = 0.000714
grad_step = 000175, loss = 0.000702
grad_step = 000176, loss = 0.000692
grad_step = 000177, loss = 0.000690
grad_step = 000178, loss = 0.000716
grad_step = 000179, loss = 0.000766
grad_step = 000180, loss = 0.000723
grad_step = 000181, loss = 0.000651
grad_step = 000182, loss = 0.000700
grad_step = 000183, loss = 0.000680
grad_step = 000184, loss = 0.000634
grad_step = 000185, loss = 0.000674
grad_step = 000186, loss = 0.000635
grad_step = 000187, loss = 0.000631
grad_step = 000188, loss = 0.000643
grad_step = 000189, loss = 0.000613
grad_step = 000190, loss = 0.000620
grad_step = 000191, loss = 0.000619
grad_step = 000192, loss = 0.000594
grad_step = 000193, loss = 0.000611
grad_step = 000194, loss = 0.000592
grad_step = 000195, loss = 0.000586
grad_step = 000196, loss = 0.000591
grad_step = 000197, loss = 0.000573
grad_step = 000198, loss = 0.000575
grad_step = 000199, loss = 0.000571
grad_step = 000200, loss = 0.000560
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000561
grad_step = 000202, loss = 0.000557
grad_step = 000203, loss = 0.000547
grad_step = 000204, loss = 0.000550
grad_step = 000205, loss = 0.000542
grad_step = 000206, loss = 0.000537
grad_step = 000207, loss = 0.000536
grad_step = 000208, loss = 0.000531
grad_step = 000209, loss = 0.000525
grad_step = 000210, loss = 0.000524
grad_step = 000211, loss = 0.000521
grad_step = 000212, loss = 0.000514
grad_step = 000213, loss = 0.000514
grad_step = 000214, loss = 0.000511
grad_step = 000215, loss = 0.000505
grad_step = 000216, loss = 0.000500
grad_step = 000217, loss = 0.000499
grad_step = 000218, loss = 0.000497
grad_step = 000219, loss = 0.000492
grad_step = 000220, loss = 0.000486
grad_step = 000221, loss = 0.000482
grad_step = 000222, loss = 0.000480
grad_step = 000223, loss = 0.000477
grad_step = 000224, loss = 0.000475
grad_step = 000225, loss = 0.000472
grad_step = 000226, loss = 0.000467
grad_step = 000227, loss = 0.000462
grad_step = 000228, loss = 0.000459
grad_step = 000229, loss = 0.000456
grad_step = 000230, loss = 0.000454
grad_step = 000231, loss = 0.000454
grad_step = 000232, loss = 0.000453
grad_step = 000233, loss = 0.000457
grad_step = 000234, loss = 0.000460
grad_step = 000235, loss = 0.000466
grad_step = 000236, loss = 0.000462
grad_step = 000237, loss = 0.000454
grad_step = 000238, loss = 0.000440
grad_step = 000239, loss = 0.000428
grad_step = 000240, loss = 0.000424
grad_step = 000241, loss = 0.000429
grad_step = 000242, loss = 0.000435
grad_step = 000243, loss = 0.000442
grad_step = 000244, loss = 0.000441
grad_step = 000245, loss = 0.000431
grad_step = 000246, loss = 0.000416
grad_step = 000247, loss = 0.000407
grad_step = 000248, loss = 0.000406
grad_step = 000249, loss = 0.000410
grad_step = 000250, loss = 0.000414
grad_step = 000251, loss = 0.000413
grad_step = 000252, loss = 0.000409
grad_step = 000253, loss = 0.000402
grad_step = 000254, loss = 0.000394
grad_step = 000255, loss = 0.000389
grad_step = 000256, loss = 0.000387
grad_step = 000257, loss = 0.000385
grad_step = 000258, loss = 0.000385
grad_step = 000259, loss = 0.000385
grad_step = 000260, loss = 0.000386
grad_step = 000261, loss = 0.000389
grad_step = 000262, loss = 0.000396
grad_step = 000263, loss = 0.000403
grad_step = 000264, loss = 0.000409
grad_step = 000265, loss = 0.000409
grad_step = 000266, loss = 0.000402
grad_step = 000267, loss = 0.000383
grad_step = 000268, loss = 0.000366
grad_step = 000269, loss = 0.000360
grad_step = 000270, loss = 0.000366
grad_step = 000271, loss = 0.000372
grad_step = 000272, loss = 0.000374
grad_step = 000273, loss = 0.000370
grad_step = 000274, loss = 0.000361
grad_step = 000275, loss = 0.000352
grad_step = 000276, loss = 0.000348
grad_step = 000277, loss = 0.000349
grad_step = 000278, loss = 0.000351
grad_step = 000279, loss = 0.000354
grad_step = 000280, loss = 0.000355
grad_step = 000281, loss = 0.000353
grad_step = 000282, loss = 0.000350
grad_step = 000283, loss = 0.000345
grad_step = 000284, loss = 0.000341
grad_step = 000285, loss = 0.000337
grad_step = 000286, loss = 0.000333
grad_step = 000287, loss = 0.000331
grad_step = 000288, loss = 0.000329
grad_step = 000289, loss = 0.000327
grad_step = 000290, loss = 0.000326
grad_step = 000291, loss = 0.000325
grad_step = 000292, loss = 0.000326
grad_step = 000293, loss = 0.000331
grad_step = 000294, loss = 0.000345
grad_step = 000295, loss = 0.000384
grad_step = 000296, loss = 0.000461
grad_step = 000297, loss = 0.000589
grad_step = 000298, loss = 0.000569
grad_step = 000299, loss = 0.000406
grad_step = 000300, loss = 0.000319
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000424
grad_step = 000302, loss = 0.000443
grad_step = 000303, loss = 0.000327
grad_step = 000304, loss = 0.000354
grad_step = 000305, loss = 0.000405
grad_step = 000306, loss = 0.000328
grad_step = 000307, loss = 0.000335
grad_step = 000308, loss = 0.000376
grad_step = 000309, loss = 0.000322
grad_step = 000310, loss = 0.000325
grad_step = 000311, loss = 0.000353
grad_step = 000312, loss = 0.000312
grad_step = 000313, loss = 0.000321
grad_step = 000314, loss = 0.000337
grad_step = 000315, loss = 0.000305
grad_step = 000316, loss = 0.000317
grad_step = 000317, loss = 0.000323
grad_step = 000318, loss = 0.000300
grad_step = 000319, loss = 0.000311
grad_step = 000320, loss = 0.000314
grad_step = 000321, loss = 0.000297
grad_step = 000322, loss = 0.000306
grad_step = 000323, loss = 0.000308
grad_step = 000324, loss = 0.000293
grad_step = 000325, loss = 0.000302
grad_step = 000326, loss = 0.000305
grad_step = 000327, loss = 0.000291
grad_step = 000328, loss = 0.000294
grad_step = 000329, loss = 0.000300
grad_step = 000330, loss = 0.000290
grad_step = 000331, loss = 0.000288
grad_step = 000332, loss = 0.000294
grad_step = 000333, loss = 0.000288
grad_step = 000334, loss = 0.000284
grad_step = 000335, loss = 0.000287
grad_step = 000336, loss = 0.000286
grad_step = 000337, loss = 0.000281
grad_step = 000338, loss = 0.000281
grad_step = 000339, loss = 0.000283
grad_step = 000340, loss = 0.000281
grad_step = 000341, loss = 0.000277
grad_step = 000342, loss = 0.000278
grad_step = 000343, loss = 0.000279
grad_step = 000344, loss = 0.000276
grad_step = 000345, loss = 0.000274
grad_step = 000346, loss = 0.000275
grad_step = 000347, loss = 0.000275
grad_step = 000348, loss = 0.000273
grad_step = 000349, loss = 0.000271
grad_step = 000350, loss = 0.000271
grad_step = 000351, loss = 0.000271
grad_step = 000352, loss = 0.000270
grad_step = 000353, loss = 0.000268
grad_step = 000354, loss = 0.000267
grad_step = 000355, loss = 0.000267
grad_step = 000356, loss = 0.000266
grad_step = 000357, loss = 0.000266
grad_step = 000358, loss = 0.000265
grad_step = 000359, loss = 0.000264
grad_step = 000360, loss = 0.000263
grad_step = 000361, loss = 0.000262
grad_step = 000362, loss = 0.000261
grad_step = 000363, loss = 0.000260
grad_step = 000364, loss = 0.000260
grad_step = 000365, loss = 0.000259
grad_step = 000366, loss = 0.000259
grad_step = 000367, loss = 0.000258
grad_step = 000368, loss = 0.000257
grad_step = 000369, loss = 0.000257
grad_step = 000370, loss = 0.000257
grad_step = 000371, loss = 0.000257
grad_step = 000372, loss = 0.000258
grad_step = 000373, loss = 0.000261
grad_step = 000374, loss = 0.000266
grad_step = 000375, loss = 0.000276
grad_step = 000376, loss = 0.000287
grad_step = 000377, loss = 0.000297
grad_step = 000378, loss = 0.000297
grad_step = 000379, loss = 0.000292
grad_step = 000380, loss = 0.000276
grad_step = 000381, loss = 0.000261
grad_step = 000382, loss = 0.000251
grad_step = 000383, loss = 0.000248
grad_step = 000384, loss = 0.000252
grad_step = 000385, loss = 0.000257
grad_step = 000386, loss = 0.000263
grad_step = 000387, loss = 0.000264
grad_step = 000388, loss = 0.000262
grad_step = 000389, loss = 0.000256
grad_step = 000390, loss = 0.000250
grad_step = 000391, loss = 0.000245
grad_step = 000392, loss = 0.000243
grad_step = 000393, loss = 0.000244
grad_step = 000394, loss = 0.000246
grad_step = 000395, loss = 0.000249
grad_step = 000396, loss = 0.000251
grad_step = 000397, loss = 0.000252
grad_step = 000398, loss = 0.000251
grad_step = 000399, loss = 0.000248
grad_step = 000400, loss = 0.000245
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000242
grad_step = 000402, loss = 0.000240
grad_step = 000403, loss = 0.000238
grad_step = 000404, loss = 0.000237
grad_step = 000405, loss = 0.000236
grad_step = 000406, loss = 0.000235
grad_step = 000407, loss = 0.000235
grad_step = 000408, loss = 0.000236
grad_step = 000409, loss = 0.000238
grad_step = 000410, loss = 0.000241
grad_step = 000411, loss = 0.000246
grad_step = 000412, loss = 0.000254
grad_step = 000413, loss = 0.000263
grad_step = 000414, loss = 0.000272
grad_step = 000415, loss = 0.000280
grad_step = 000416, loss = 0.000283
grad_step = 000417, loss = 0.000277
grad_step = 000418, loss = 0.000260
grad_step = 000419, loss = 0.000242
grad_step = 000420, loss = 0.000230
grad_step = 000421, loss = 0.000231
grad_step = 000422, loss = 0.000242
grad_step = 000423, loss = 0.000252
grad_step = 000424, loss = 0.000252
grad_step = 000425, loss = 0.000244
grad_step = 000426, loss = 0.000232
grad_step = 000427, loss = 0.000226
grad_step = 000428, loss = 0.000228
grad_step = 000429, loss = 0.000235
grad_step = 000430, loss = 0.000240
grad_step = 000431, loss = 0.000240
grad_step = 000432, loss = 0.000234
grad_step = 000433, loss = 0.000228
grad_step = 000434, loss = 0.000223
grad_step = 000435, loss = 0.000223
grad_step = 000436, loss = 0.000225
grad_step = 000437, loss = 0.000228
grad_step = 000438, loss = 0.000231
grad_step = 000439, loss = 0.000234
grad_step = 000440, loss = 0.000234
grad_step = 000441, loss = 0.000234
grad_step = 000442, loss = 0.000234
grad_step = 000443, loss = 0.000231
grad_step = 000444, loss = 0.000227
grad_step = 000445, loss = 0.000224
grad_step = 000446, loss = 0.000221
grad_step = 000447, loss = 0.000219
grad_step = 000448, loss = 0.000217
grad_step = 000449, loss = 0.000217
grad_step = 000450, loss = 0.000217
grad_step = 000451, loss = 0.000217
grad_step = 000452, loss = 0.000220
grad_step = 000453, loss = 0.000223
grad_step = 000454, loss = 0.000225
grad_step = 000455, loss = 0.000229
grad_step = 000456, loss = 0.000235
grad_step = 000457, loss = 0.000243
grad_step = 000458, loss = 0.000253
grad_step = 000459, loss = 0.000259
grad_step = 000460, loss = 0.000262
grad_step = 000461, loss = 0.000256
grad_step = 000462, loss = 0.000245
grad_step = 000463, loss = 0.000228
grad_step = 000464, loss = 0.000215
grad_step = 000465, loss = 0.000212
grad_step = 000466, loss = 0.000217
grad_step = 000467, loss = 0.000225
grad_step = 000468, loss = 0.000230
grad_step = 000469, loss = 0.000230
grad_step = 000470, loss = 0.000224
grad_step = 000471, loss = 0.000217
grad_step = 000472, loss = 0.000211
grad_step = 000473, loss = 0.000209
grad_step = 000474, loss = 0.000210
grad_step = 000475, loss = 0.000214
grad_step = 000476, loss = 0.000217
grad_step = 000477, loss = 0.000219
grad_step = 000478, loss = 0.000221
grad_step = 000479, loss = 0.000221
grad_step = 000480, loss = 0.000221
grad_step = 000481, loss = 0.000217
grad_step = 000482, loss = 0.000215
grad_step = 000483, loss = 0.000213
grad_step = 000484, loss = 0.000211
grad_step = 000485, loss = 0.000207
grad_step = 000486, loss = 0.000205
grad_step = 000487, loss = 0.000204
grad_step = 000488, loss = 0.000204
grad_step = 000489, loss = 0.000205
grad_step = 000490, loss = 0.000208
grad_step = 000491, loss = 0.000213
grad_step = 000492, loss = 0.000220
grad_step = 000493, loss = 0.000230
grad_step = 000494, loss = 0.000240
grad_step = 000495, loss = 0.000249
grad_step = 000496, loss = 0.000251
grad_step = 000497, loss = 0.000247
grad_step = 000498, loss = 0.000234
grad_step = 000499, loss = 0.000218
grad_step = 000500, loss = 0.000205
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000201
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
[[0.8591575  0.84778893 0.9429711  0.9460547  1.0270914 ]
 [0.85075814 0.91579795 0.9537507  1.0216904  0.9975203 ]
 [0.8872391  0.9118347  1.0063266  0.98608756 0.9610118 ]
 [0.92500037 0.98915386 0.9882232  0.9626597  0.9279645 ]
 [0.98601305 0.98326313 0.9407413  0.91983914 0.85390574]
 [0.9728794  0.962515   0.9172846  0.86329    0.86129653]
 [0.9480675  0.9099957  0.85987663 0.8573346  0.8259483 ]
 [0.89690053 0.8392874  0.8656454  0.8122035  0.8605762 ]
 [0.82790416 0.83036697 0.8161051  0.8420778  0.8523193 ]
 [0.844663   0.80073595 0.83633506 0.8592391  0.83759975]
 [0.8040541  0.8234445  0.8581189  0.8275596  0.9307425 ]
 [0.81260407 0.84878016 0.8284449  0.9202129  0.94337803]
 [0.8516207  0.8413974  0.9387351  0.9495504  1.0215212 ]
 [0.8483529  0.9209728  0.959713   1.0259087  0.9912156 ]
 [0.8960583  0.92059934 1.0018698  0.9803337  0.9400738 ]
 [0.93360066 0.99310875 0.97747993 0.9481431  0.90758616]
 [0.9903213  0.97781706 0.92795336 0.9053606  0.8366493 ]
 [0.9705686  0.94382095 0.8988891  0.8408251  0.84843665]
 [0.938457   0.897822   0.8345953  0.84430003 0.82193893]
 [0.9009013  0.8420143  0.84994483 0.8109106  0.85527456]
 [0.8396827  0.8416903  0.8148092  0.8431312  0.8578833 ]
 [0.8613839  0.81613576 0.8364208  0.8677132  0.84043396]
 [0.8148687  0.83682895 0.86854506 0.8327409  0.94054246]
 [0.82064414 0.86375904 0.8346895  0.92299056 0.9470001 ]
 [0.86704206 0.8497633  0.9403075  0.94532645 1.0335076 ]
 [0.8618558  0.9212012  0.9582456  1.0301806  1.0099429 ]
 [0.8946942  0.921186   1.0152961  0.998438   0.9755751 ]
 [0.93451285 0.99923456 1.0003288  0.9773332  0.94095653]
 [0.9963792  0.99314153 0.95138395 0.93316233 0.86346537]
 [0.98450696 0.9735925  0.9255368  0.8732488  0.8650166 ]
 [0.9565506  0.91805565 0.86541307 0.86191976 0.83420205]]

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
[master 91e48ef] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   3308113..91e48ef  master -> master





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
[master aca141b] ml_store
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   91e48ef..aca141b  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s]100%|██████████| 231508/231508 [00:00<00:00, 26237264.73B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 243547.49B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  1%|▏         | 5779456/440473133 [00:00<00:07, 57781252.26B/s]  3%|▎         | 12145664/440473133 [00:00<00:07, 59426913.58B/s]  4%|▍         | 18599936/440473133 [00:00<00:06, 60871857.67B/s]  6%|▌         | 24889344/440473133 [00:00<00:06, 61462176.60B/s]  7%|▋         | 31320064/440473133 [00:00<00:06, 62288309.11B/s]  8%|▊         | 37274624/440473133 [00:00<00:06, 61438031.70B/s] 10%|▉         | 43764736/440473133 [00:00<00:06, 62435691.11B/s] 11%|█▏        | 50289664/440473133 [00:00<00:06, 63252833.12B/s] 13%|█▎        | 56788992/440473133 [00:00<00:06, 63762299.40B/s] 14%|█▍        | 63294464/440473133 [00:01<00:05, 64139191.27B/s] 16%|█▌        | 69840896/440473133 [00:01<00:05, 64529852.36B/s] 17%|█▋        | 76390400/440473133 [00:01<00:05, 64815509.16B/s] 19%|█▉        | 82796544/440473133 [00:01<00:05, 63557171.81B/s] 20%|██        | 89346048/440473133 [00:01<00:05, 64124784.82B/s] 22%|██▏       | 95856640/440473133 [00:01<00:05, 64413289.52B/s] 23%|██▎       | 102276096/440473133 [00:01<00:05, 64106463.23B/s] 25%|██▍       | 108672000/440473133 [00:01<00:05, 63429771.57B/s] 26%|██▌       | 115084288/440473133 [00:01<00:05, 63633162.64B/s] 28%|██▊       | 121442304/440473133 [00:01<00:05, 63244354.77B/s] 29%|██▉       | 127910912/440473133 [00:02<00:04, 63668303.91B/s] 30%|███       | 134276096/440473133 [00:02<00:04, 62182016.74B/s] 32%|███▏      | 140849152/440473133 [00:02<00:04, 63205139.50B/s] 33%|███▎      | 147408896/440473133 [00:02<00:04, 63902052.31B/s] 35%|███▍      | 153940992/440473133 [00:02<00:04, 64319143.91B/s] 36%|███▋      | 160502784/440473133 [00:02<00:04, 64701949.71B/s] 38%|███▊      | 166978560/440473133 [00:02<00:04, 64286644.91B/s] 39%|███▉      | 173512704/440473133 [00:02<00:04, 64596539.46B/s] 41%|████      | 179976192/440473133 [00:02<00:04, 63958991.38B/s] 42%|████▏     | 186509312/440473133 [00:02<00:03, 64362268.10B/s] 44%|████▍     | 193058816/440473133 [00:03<00:03, 64695468.11B/s] 45%|████▌     | 199603200/440473133 [00:03<00:03, 64917043.28B/s] 47%|████▋     | 206179328/440473133 [00:03<00:03, 65166015.59B/s] 48%|████▊     | 212698112/440473133 [00:03<00:03, 64321887.16B/s] 50%|████▉     | 219217920/440473133 [00:03<00:03, 64581456.93B/s] 51%|█████     | 225679360/440473133 [00:03<00:03, 63518587.46B/s] 53%|█████▎    | 232037376/440473133 [00:03<00:03, 62142353.85B/s] 54%|█████▍    | 238573568/440473133 [00:03<00:03, 63072479.11B/s] 56%|█████▌    | 245156864/440473133 [00:03<00:03, 63871014.80B/s] 57%|█████▋    | 251554816/440473133 [00:03<00:03, 62082448.81B/s] 59%|█████▊    | 257781760/440473133 [00:04<00:02, 61112566.85B/s] 60%|█████▉    | 264171520/440473133 [00:04<00:02, 61921062.95B/s] 61%|██████▏   | 270718976/440473133 [00:04<00:02, 62945612.44B/s] 63%|██████▎   | 277267456/440473133 [00:04<00:02, 63685891.08B/s] 64%|██████▍   | 283648000/440473133 [00:04<00:02, 63171344.77B/s] 66%|██████▌   | 290006016/440473133 [00:04<00:02, 63289939.26B/s] 67%|██████▋   | 296341504/440473133 [00:04<00:02, 62906208.67B/s] 69%|██████▉   | 302828544/440473133 [00:04<00:02, 63480569.44B/s] 70%|███████   | 309181440/440473133 [00:04<00:02, 62764632.40B/s] 72%|███████▏  | 315463680/440473133 [00:04<00:01, 62617302.73B/s] 73%|███████▎  | 321968128/440473133 [00:05<00:01, 63323327.83B/s] 75%|███████▍  | 328459264/440473133 [00:05<00:01, 63790684.51B/s] 76%|███████▌  | 335046656/440473133 [00:05<00:01, 64399183.73B/s] 78%|███████▊  | 341605376/440473133 [00:05<00:01, 64750697.40B/s] 79%|███████▉  | 348218368/440473133 [00:05<00:01, 65156291.51B/s] 81%|████████  | 354793472/440473133 [00:05<00:01, 65329562.34B/s] 82%|████████▏ | 361328640/440473133 [00:05<00:01, 63528704.89B/s] 84%|████████▎ | 367955968/440473133 [00:05<00:01, 64327836.87B/s] 85%|████████▌ | 374584320/440473133 [00:05<00:01, 64901016.19B/s] 87%|████████▋ | 381209600/440473133 [00:05<00:00, 65299821.36B/s] 88%|████████▊ | 387746816/440473133 [00:06<00:00, 64685435.34B/s] 90%|████████▉ | 394276864/440473133 [00:06<00:00, 64868424.58B/s] 91%|█████████ | 400858112/440473133 [00:06<00:00, 65148111.61B/s] 92%|█████████▏| 407409664/440473133 [00:06<00:00, 65257872.81B/s] 94%|█████████▍| 413988864/440473133 [00:06<00:00, 65414704.57B/s] 95%|█████████▌| 420552704/440473133 [00:06<00:00, 65479615.82B/s] 97%|█████████▋| 427102208/440473133 [00:06<00:00, 65046302.70B/s] 98%|█████████▊| 433678336/440473133 [00:06<00:00, 65258018.65B/s]100%|█████████▉| 440326144/440473133 [00:06<00:00, 65617057.31B/s]100%|██████████| 440473133/440473133 [00:06<00:00, 63952450.79B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
6586368/7094233 [==========================>...] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   1%|          | 16/2118 [00:00<00:13, 152.98it/s]Processing text_left with encode:  21%|██▏       | 451/2118 [00:00<00:07, 215.30it/s]Processing text_left with encode:  43%|████▎     | 911/2118 [00:00<00:04, 301.51it/s]Processing text_left with encode:  60%|█████▉    | 1266/2118 [00:00<00:02, 415.58it/s]Processing text_left with encode:  81%|████████▏ | 1726/2118 [00:00<00:00, 571.53it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3580.64it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 157/18841 [00:00<00:11, 1566.03it/s]Processing text_right with encode:   2%|▏         | 331/18841 [00:00<00:11, 1614.29it/s]Processing text_right with encode:   3%|▎         | 506/18841 [00:00<00:11, 1652.47it/s]Processing text_right with encode:   4%|▎         | 674/18841 [00:00<00:10, 1656.84it/s]Processing text_right with encode:   4%|▍         | 847/18841 [00:00<00:10, 1677.59it/s]Processing text_right with encode:   5%|▌         | 1018/18841 [00:00<00:10, 1683.33it/s]Processing text_right with encode:   6%|▋         | 1178/18841 [00:00<00:10, 1655.92it/s]Processing text_right with encode:   7%|▋         | 1339/18841 [00:00<00:10, 1640.68it/s]Processing text_right with encode:   8%|▊         | 1510/18841 [00:00<00:10, 1659.31it/s]Processing text_right with encode:   9%|▉         | 1689/18841 [00:01<00:10, 1696.26it/s]Processing text_right with encode:  10%|▉         | 1855/18841 [00:01<00:10, 1680.40it/s]Processing text_right with encode:  11%|█         | 2021/18841 [00:01<00:10, 1627.87it/s]Processing text_right with encode:  12%|█▏        | 2189/18841 [00:01<00:10, 1641.44it/s]Processing text_right with encode:  13%|█▎        | 2361/18841 [00:01<00:09, 1660.43it/s]Processing text_right with encode:  13%|█▎        | 2528/18841 [00:01<00:09, 1662.44it/s]Processing text_right with encode:  14%|█▍        | 2699/18841 [00:01<00:09, 1674.21it/s]Processing text_right with encode:  15%|█▌        | 2892/18841 [00:01<00:09, 1743.06it/s]Processing text_right with encode:  16%|█▋        | 3067/18841 [00:01<00:09, 1694.62it/s]Processing text_right with encode:  17%|█▋        | 3238/18841 [00:01<00:09, 1672.89it/s]Processing text_right with encode:  18%|█▊        | 3406/18841 [00:02<00:09, 1658.16it/s]Processing text_right with encode:  19%|█▉        | 3579/18841 [00:02<00:09, 1676.86it/s]Processing text_right with encode:  20%|█▉        | 3749/18841 [00:02<00:08, 1683.68it/s]Processing text_right with encode:  21%|██        | 3919/18841 [00:02<00:08, 1686.40it/s]Processing text_right with encode:  22%|██▏       | 4096/18841 [00:02<00:08, 1710.59it/s]Processing text_right with encode:  23%|██▎       | 4268/18841 [00:02<00:08, 1711.92it/s]Processing text_right with encode:  24%|██▎       | 4440/18841 [00:02<00:08, 1688.81it/s]Processing text_right with encode:  24%|██▍       | 4616/18841 [00:02<00:08, 1706.47it/s]Processing text_right with encode:  25%|██▌       | 4789/18841 [00:02<00:08, 1711.99it/s]Processing text_right with encode:  26%|██▋       | 4971/18841 [00:02<00:07, 1741.76it/s]Processing text_right with encode:  27%|██▋       | 5153/18841 [00:03<00:07, 1763.87it/s]Processing text_right with encode:  28%|██▊       | 5335/18841 [00:03<00:07, 1776.02it/s]Processing text_right with encode:  29%|██▉       | 5513/18841 [00:03<00:07, 1764.67it/s]Processing text_right with encode:  30%|███       | 5690/18841 [00:03<00:07, 1752.64it/s]Processing text_right with encode:  31%|███       | 5866/18841 [00:03<00:07, 1743.88it/s]Processing text_right with encode:  32%|███▏      | 6041/18841 [00:03<00:07, 1699.67it/s]Processing text_right with encode:  33%|███▎      | 6212/18841 [00:03<00:07, 1687.04it/s]Processing text_right with encode:  34%|███▍      | 6387/18841 [00:03<00:07, 1704.56it/s]Processing text_right with encode:  35%|███▍      | 6558/18841 [00:03<00:07, 1692.92it/s]Processing text_right with encode:  36%|███▌      | 6728/18841 [00:03<00:07, 1689.33it/s]Processing text_right with encode:  37%|███▋      | 6899/18841 [00:04<00:07, 1692.91it/s]Processing text_right with encode:  38%|███▊      | 7071/18841 [00:04<00:06, 1700.00it/s]Processing text_right with encode:  38%|███▊      | 7244/18841 [00:04<00:06, 1707.69it/s]Processing text_right with encode:  39%|███▉      | 7417/18841 [00:04<00:06, 1712.45it/s]Processing text_right with encode:  40%|████      | 7589/18841 [00:04<00:06, 1688.71it/s]Processing text_right with encode:  41%|████      | 7761/18841 [00:04<00:06, 1697.65it/s]Processing text_right with encode:  42%|████▏     | 7931/18841 [00:04<00:06, 1683.81it/s]Processing text_right with encode:  43%|████▎     | 8100/18841 [00:04<00:06, 1667.03it/s]Processing text_right with encode:  44%|████▍     | 8268/18841 [00:04<00:06, 1669.95it/s]Processing text_right with encode:  45%|████▍     | 8442/18841 [00:04<00:06, 1688.50it/s]Processing text_right with encode:  46%|████▌     | 8611/18841 [00:05<00:06, 1645.04it/s]Processing text_right with encode:  47%|████▋     | 8790/18841 [00:05<00:05, 1683.02it/s]Processing text_right with encode:  48%|████▊     | 8959/18841 [00:05<00:05, 1667.34it/s]Processing text_right with encode:  48%|████▊     | 9127/18841 [00:05<00:05, 1654.39it/s]Processing text_right with encode:  49%|████▉     | 9299/18841 [00:05<00:05, 1670.56it/s]Processing text_right with encode:  50%|█████     | 9467/18841 [00:05<00:05, 1649.72it/s]Processing text_right with encode:  51%|█████     | 9633/18841 [00:05<00:05, 1628.45it/s]Processing text_right with encode:  52%|█████▏    | 9797/18841 [00:05<00:05, 1616.25it/s]Processing text_right with encode:  53%|█████▎    | 9978/18841 [00:05<00:05, 1668.50it/s]Processing text_right with encode:  54%|█████▍    | 10147/18841 [00:06<00:05, 1666.80it/s]Processing text_right with encode:  55%|█████▍    | 10315/18841 [00:06<00:05, 1651.47it/s]Processing text_right with encode:  56%|█████▌    | 10507/18841 [00:06<00:04, 1721.08it/s]Processing text_right with encode:  57%|█████▋    | 10681/18841 [00:06<00:04, 1687.96it/s]Processing text_right with encode:  58%|█████▊    | 10851/18841 [00:06<00:04, 1660.38it/s]Processing text_right with encode:  58%|█████▊    | 11018/18841 [00:06<00:04, 1634.49it/s]Processing text_right with encode:  59%|█████▉    | 11183/18841 [00:06<00:04, 1617.05it/s]Processing text_right with encode:  60%|██████    | 11346/18841 [00:06<00:04, 1571.65it/s]Processing text_right with encode:  61%|██████    | 11511/18841 [00:06<00:04, 1593.51it/s]Processing text_right with encode:  62%|██████▏   | 11683/18841 [00:06<00:04, 1627.61it/s]Processing text_right with encode:  63%|██████▎   | 11855/18841 [00:07<00:04, 1652.30it/s]Processing text_right with encode:  64%|██████▍   | 12024/18841 [00:07<00:04, 1661.39it/s]Processing text_right with encode:  65%|██████▍   | 12191/18841 [00:07<00:04, 1644.24it/s]Processing text_right with encode:  66%|██████▌   | 12360/18841 [00:07<00:03, 1655.34it/s]Processing text_right with encode:  66%|██████▋   | 12526/18841 [00:07<00:03, 1646.05it/s]Processing text_right with encode:  67%|██████▋   | 12691/18841 [00:07<00:03, 1641.72it/s]Processing text_right with encode:  68%|██████▊   | 12856/18841 [00:07<00:03, 1635.87it/s]Processing text_right with encode:  69%|██████▉   | 13023/18841 [00:07<00:03, 1645.63it/s]Processing text_right with encode:  70%|███████   | 13192/18841 [00:07<00:03, 1655.61it/s]Processing text_right with encode:  71%|███████   | 13371/18841 [00:07<00:03, 1692.38it/s]Processing text_right with encode:  72%|███████▏  | 13544/18841 [00:08<00:03, 1703.24it/s]Processing text_right with encode:  73%|███████▎  | 13726/18841 [00:08<00:02, 1734.17it/s]Processing text_right with encode:  74%|███████▍  | 13900/18841 [00:08<00:02, 1718.79it/s]Processing text_right with encode:  75%|███████▍  | 14073/18841 [00:08<00:02, 1654.21it/s]Processing text_right with encode:  76%|███████▌  | 14240/18841 [00:08<00:02, 1647.70it/s]Processing text_right with encode:  77%|███████▋  | 14417/18841 [00:08<00:02, 1679.72it/s]Processing text_right with encode:  77%|███████▋  | 14594/18841 [00:08<00:02, 1705.44it/s]Processing text_right with encode:  78%|███████▊  | 14785/18841 [00:08<00:02, 1760.19it/s]Processing text_right with encode:  79%|███████▉  | 14962/18841 [00:08<00:02, 1738.72it/s]Processing text_right with encode:  80%|████████  | 15137/18841 [00:09<00:02, 1658.50it/s]Processing text_right with encode:  81%|████████  | 15306/18841 [00:09<00:02, 1663.67it/s]Processing text_right with encode:  82%|████████▏ | 15477/18841 [00:09<00:02, 1676.86it/s]Processing text_right with encode:  83%|████████▎ | 15654/18841 [00:09<00:01, 1697.51it/s]Processing text_right with encode:  84%|████████▍ | 15834/18841 [00:09<00:01, 1726.41it/s]Processing text_right with encode:  85%|████████▍ | 16008/18841 [00:09<00:01, 1707.12it/s]Processing text_right with encode:  86%|████████▌ | 16180/18841 [00:09<00:01, 1693.68it/s]Processing text_right with encode:  87%|████████▋ | 16350/18841 [00:09<00:01, 1679.26it/s]Processing text_right with encode:  88%|████████▊ | 16519/18841 [00:09<00:01, 1659.07it/s]Processing text_right with encode:  89%|████████▊ | 16686/18841 [00:09<00:01, 1658.05it/s]Processing text_right with encode:  89%|████████▉ | 16853/18841 [00:10<00:01, 1658.84it/s]Processing text_right with encode:  90%|█████████ | 17019/18841 [00:10<00:01, 1651.31it/s]Processing text_right with encode:  91%|█████████ | 17185/18841 [00:10<00:01, 1644.11it/s]Processing text_right with encode:  92%|█████████▏| 17352/18841 [00:10<00:00, 1648.00it/s]Processing text_right with encode:  93%|█████████▎| 17517/18841 [00:10<00:00, 1587.20it/s]Processing text_right with encode:  94%|█████████▍| 17693/18841 [00:10<00:00, 1633.85it/s]Processing text_right with encode:  95%|█████████▍| 17860/18841 [00:10<00:00, 1641.98it/s]Processing text_right with encode:  96%|█████████▌| 18047/18841 [00:10<00:00, 1703.09it/s]Processing text_right with encode:  97%|█████████▋| 18223/18841 [00:10<00:00, 1719.05it/s]Processing text_right with encode:  98%|█████████▊| 18396/18841 [00:10<00:00, 1719.69it/s]Processing text_right with encode:  99%|█████████▊| 18583/18841 [00:11<00:00, 1759.82it/s]Processing text_right with encode: 100%|█████████▉| 18760/18841 [00:11<00:00, 1742.83it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1682.63it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 614224.98it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 752002.00it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  76%|███████▋  | 483/633 [00:00<00:00, 4826.52it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4796.09it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 182/5961 [00:00<00:03, 1812.08it/s]Processing text_right with encode:   6%|▌         | 355/5961 [00:00<00:03, 1785.51it/s]Processing text_right with encode:   9%|▉         | 523/5961 [00:00<00:03, 1751.60it/s]Processing text_right with encode:  12%|█▏        | 705/5961 [00:00<00:02, 1770.78it/s]Processing text_right with encode:  15%|█▍        | 891/5961 [00:00<00:02, 1795.18it/s]Processing text_right with encode:  18%|█▊        | 1070/5961 [00:00<00:02, 1790.35it/s]Processing text_right with encode:  21%|██        | 1248/5961 [00:00<00:02, 1785.45it/s]Processing text_right with encode:  24%|██▍       | 1428/5961 [00:00<00:02, 1786.27it/s]Processing text_right with encode:  27%|██▋       | 1604/5961 [00:00<00:02, 1776.23it/s]Processing text_right with encode:  30%|██▉       | 1775/5961 [00:01<00:02, 1716.96it/s]Processing text_right with encode:  33%|███▎      | 1951/5961 [00:01<00:02, 1725.83it/s]Processing text_right with encode:  36%|███▌      | 2131/5961 [00:01<00:02, 1746.91it/s]Processing text_right with encode:  39%|███▊      | 2304/5961 [00:01<00:02, 1733.53it/s]Processing text_right with encode:  42%|████▏     | 2476/5961 [00:01<00:02, 1699.23it/s]Processing text_right with encode:  45%|████▍     | 2654/5961 [00:01<00:01, 1719.88it/s]Processing text_right with encode:  48%|████▊     | 2847/5961 [00:01<00:01, 1776.38it/s]Processing text_right with encode:  51%|█████     | 3025/5961 [00:01<00:01, 1739.36it/s]Processing text_right with encode:  54%|█████▍    | 3208/5961 [00:01<00:01, 1764.39it/s]Processing text_right with encode:  57%|█████▋    | 3385/5961 [00:01<00:01, 1701.65it/s]Processing text_right with encode:  60%|█████▉    | 3556/5961 [00:02<00:01, 1700.65it/s]Processing text_right with encode:  63%|██████▎   | 3727/5961 [00:02<00:01, 1662.99it/s]Processing text_right with encode:  65%|██████▌   | 3903/5961 [00:02<00:01, 1686.55it/s]Processing text_right with encode:  68%|██████▊   | 4080/5961 [00:02<00:01, 1710.09it/s]Processing text_right with encode:  71%|███████▏  | 4252/5961 [00:02<00:01, 1697.14it/s]Processing text_right with encode:  74%|███████▍  | 4423/5961 [00:02<00:00, 1675.61it/s]Processing text_right with encode:  77%|███████▋  | 4602/5961 [00:02<00:00, 1705.88it/s]Processing text_right with encode:  80%|████████  | 4773/5961 [00:02<00:00, 1676.96it/s]Processing text_right with encode:  83%|████████▎ | 4942/5961 [00:02<00:00, 1663.92it/s]Processing text_right with encode:  86%|████████▌ | 5114/5961 [00:02<00:00, 1678.70it/s]Processing text_right with encode:  89%|████████▊ | 5289/5961 [00:03<00:00, 1698.63it/s]Processing text_right with encode:  92%|█████████▏| 5460/5961 [00:03<00:00, 1669.98it/s]Processing text_right with encode:  94%|█████████▍| 5628/5961 [00:03<00:00, 1627.05it/s]Processing text_right with encode:  97%|█████████▋| 5792/5961 [00:03<00:00, 1612.39it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1715.79it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 411563.24it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 624665.74it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:20<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:20<?, ?it/s, loss=0.916]Epoch 1/1:   1%|          | 1/102 [00:20<34:40, 20.60s/it, loss=0.916]Epoch 1/1:   1%|          | 1/102 [00:35<34:40, 20.60s/it, loss=0.916]Epoch 1/1:   1%|          | 1/102 [00:35<34:40, 20.60s/it, loss=0.997]Epoch 1/1:   2%|▏         | 2/102 [00:35<31:33, 18.94s/it, loss=0.997]Epoch 1/1:   2%|▏         | 2/102 [00:43<31:33, 18.94s/it, loss=0.997]Epoch 1/1:   2%|▏         | 2/102 [00:43<31:33, 18.94s/it, loss=0.770]Epoch 1/1:   3%|▎         | 3/102 [00:43<25:53, 15.69s/it, loss=0.770]Epoch 1/1:   3%|▎         | 3/102 [01:42<25:53, 15.69s/it, loss=0.770]Epoch 1/1:   3%|▎         | 3/102 [01:42<25:53, 15.69s/it, loss=0.798]Epoch 1/1:   4%|▍         | 4/102 [01:42<46:37, 28.55s/it, loss=0.798]Epoch 1/1:   4%|▍         | 4/102 [02:46<46:37, 28.55s/it, loss=0.798]Epoch 1/1:   4%|▍         | 4/102 [02:46<46:37, 28.55s/it, loss=0.623]Epoch 1/1:   5%|▍         | 5/102 [02:46<1:03:18, 39.16s/it, loss=0.623]Epoch 1/1:   5%|▍         | 5/102 [03:48<1:03:18, 39.16s/it, loss=0.623]Epoch 1/1:   5%|▍         | 5/102 [03:48<1:03:18, 39.16s/it, loss=0.818]Epoch 1/1:   6%|▌         | 6/102 [03:48<1:13:53, 46.18s/it, loss=0.818]Epoch 1/1:   6%|▌         | 6/102 [05:07<1:13:53, 46.18s/it, loss=0.818]Epoch 1/1:   6%|▌         | 6/102 [05:07<1:13:53, 46.18s/it, loss=0.973]Epoch 1/1:   7%|▋         | 7/102 [05:07<1:28:45, 56.05s/it, loss=0.973]Epoch 1/1:   7%|▋         | 7/102 [05:55<1:28:45, 56.05s/it, loss=0.973]Epoch 1/1:   7%|▋         | 7/102 [05:55<1:28:45, 56.05s/it, loss=0.895]Epoch 1/1:   8%|▊         | 8/102 [05:55<1:23:56, 53.58s/it, loss=0.895]Epoch 1/1:   8%|▊         | 8/102 [06:36<1:23:56, 53.58s/it, loss=0.895]Epoch 1/1:   8%|▊         | 8/102 [06:36<1:23:56, 53.58s/it, loss=0.798]Epoch 1/1:   9%|▉         | 9/102 [06:36<1:16:55, 49.63s/it, loss=0.798]Epoch 1/1:   9%|▉         | 9/102 [07:02<1:16:55, 49.63s/it, loss=0.798]Epoch 1/1:   9%|▉         | 9/102 [07:02<1:16:55, 49.63s/it, loss=0.298]Epoch 1/1:  10%|▉         | 10/102 [07:02<1:05:26, 42.67s/it, loss=0.298]Epoch 1/1:  10%|▉         | 10/102 [08:20<1:05:26, 42.67s/it, loss=0.298]Epoch 1/1:  10%|▉         | 10/102 [08:20<1:05:26, 42.67s/it, loss=0.575]Epoch 1/1:  11%|█         | 11/102 [08:20<1:20:57, 53.37s/it, loss=0.575]Epoch 1/1:  11%|█         | 11/102 [09:05<1:20:57, 53.37s/it, loss=0.575]Epoch 1/1:  11%|█         | 11/102 [09:05<1:20:57, 53.37s/it, loss=0.721]Epoch 1/1:  12%|█▏        | 12/102 [09:05<1:15:57, 50.64s/it, loss=0.721]Epoch 1/1:  12%|█▏        | 12/102 [10:16<1:15:57, 50.64s/it, loss=0.721]Epoch 1/1:  12%|█▏        | 12/102 [10:16<1:15:57, 50.64s/it, loss=0.840]Epoch 1/1:  13%|█▎        | 13/102 [10:16<1:24:08, 56.73s/it, loss=0.840]Killed

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
Warning: Permanently added the RSA host key for IP address '140.82.118.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   aca141b..78579bd  master     -> origin/master
Updating aca141b..78579bd
Fast-forward
 deps.txt                                           |   6 +-
 .../20200521/list_log_pullrequest_20200521.md      |   2 +-
 error_list/20200521/list_log_testall_20200521.md   |   9 +
 ...-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py | 626 +++++++++++++++++++++
 4 files changed, 641 insertions(+), 2 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-21-09-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py
[master 9256b62] ml_store
 1 file changed, 69 insertions(+)
To github.com:arita37/mlmodels_store.git
   78579bd..9256b62  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f30911efe18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f30911efe18>

  function with postional parmater data_info <function get_dataset_torch at 0x7f30911efe18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 134209.60it/s] 82%|████████▏ | 8093696/9912422 [00:00<00:09, 191591.38it/s]9920512it [00:00, 41747954.14it/s]                           
0it [00:00, ?it/s]32768it [00:00, 719805.56it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1030324.54it/s]1654784it [00:00, 12515786.86it/s]                           
0it [00:00, ?it/s]8192it [00:00, 238453.64it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting mlmodels/dataset/vision/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw
Processing...
Done!

  #### Model init, fit   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3090528b70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3090528b70>

  function with postional parmater data_info <function get_dataset_torch at 0x7f3090528b70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.0018511305451393127 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.011109930992126465 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0013602466384569805 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.009833815097808838 	 Accuracy: 1
model saves at 1 accuracy

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f3090528950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f3090528950>

  function with postional parmater data_info <function get_dataset_torch at 0x7f3090528950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7f3090fa49e8>

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels.preprocess.generic::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/pytorch_GAN_zoo/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/pytorch_GAN_zoo/'} 

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
<__main__.Model object at 0x7f3088eb2b00>

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
[master f9047e7] ml_store
 2 files changed, 149 insertions(+), 5 deletions(-)
To github.com:arita37/mlmodels_store.git
   9256b62..f9047e7  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//03_nbeats_dataloader.py", line 9, in <module>
    from dataloader import DataLoader
ModuleNotFoundError: No module named 'dataloader'

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
[master 0fc6f4c] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   f9047e7..0fc6f4c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//transformer_sentence.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 
Epoch:   0%|          | 0/1 [00:00<?, ?it/s]
Iteration:   0%|          | 0/29440 [00:00<?, ?it/s][A
Iteration:   0%|          | 1/29440 [00:34<281:05:28, 34.37s/it][A
Iteration:   0%|          | 2/29440 [00:44<222:45:38, 27.24s/it][A
Iteration:   0%|          | 3/29440 [01:55<328:21:18, 40.16s/it][A
Iteration:   0%|          | 4/29440 [03:00<388:38:48, 47.53s/it][A
Iteration:   0%|          | 5/29440 [04:17<461:39:45, 56.46s/it][A
Iteration:   0%|          | 6/29440 [04:52<408:37:26, 49.98s/it][A
Iteration:   0%|          | 7/29440 [05:30<380:49:54, 46.58s/it][A
Iteration:   0%|          | 8/29440 [07:56<623:58:47, 76.32s/it][A
Iteration:   0%|          | 9/29440 [09:14<627:50:33, 76.80s/it][A
Iteration:   0%|          | 10/29440 [12:50<969:22:15, 118.58s/it][A
Iteration:   0%|          | 11/29440 [13:08<722:53:43, 88.43s/it] [A
Iteration:   0%|          | 12/29440 [14:46<746:22:31, 91.31s/it][A
Iteration:   0%|          | 13/29440 [15:43<662:01:16, 80.99s/it][A
Iteration:   0%|          | 14/29440 [17:23<709:05:03, 86.75s/it][AKilled

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
   0fc6f4c..e14b0a0  master     -> origin/master
Updating 0fc6f4c..e14b0a0
Fast-forward
 deps.txt                                           |   28 +-
 .../20200521/list_log_dataloader_20200521.md       |   29 +-
 .../20200521/list_log_pullrequest_20200521.md      |    2 +-
 error_list/20200521/list_log_testall_20200521.md   |    7 +
 log_dataloader/log_dataloader.py                   | 2466 ++------------------
 5 files changed, 165 insertions(+), 2367 deletions(-)
[master 8176fb4] ml_store
 1 file changed, 65 insertions(+)
To github.com:arita37/mlmodels_store.git
   e14b0a0..8176fb4  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//pytorch_vae.py", line 34, in <module>
    "beta_vae": md.model.beta_vae,
AttributeError: module 'mlmodels.model_tch.raw.pytorch_vae' has no attribute 'model'

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
[master 3e641ff] ml_store
 2 files changed, 50 insertions(+), 14 deletions(-)
To github.com:arita37/mlmodels_store.git
   8176fb4..3e641ff  master -> master





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
   3e641ff..3d11e13  master     -> origin/master
Updating 3e641ff..3d11e13
Fast-forward
 error_list/20200521/list_log_testall_20200521.md | 7 +++++++
 1 file changed, 7 insertions(+)
[master edccb8c] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   3d11e13..edccb8c  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/textcnn/model', 'checkpointdir': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/textcnn//checkpoint/'}

  #### Loading dataset   ############################################# 
>>>>> load:  {'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_sample.txt', 'train_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_train.csv', 'valid_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/recommender/IMDB_valid.csv', 'split_if_exists': True, 'frac': 0.99, 'lang': 'en', 'pretrained_emb': 'glove.6B.300d', 'batch_size': 64, 'val_batch_size': 64}
Spliting original file to train/valid set...

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.4)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.1)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=5d479efbcdae2218f12145a3b49c66ddaaa66bc575149ffb90d6e85564aa4c77
  Stored in directory: /tmp/pip-ephem-wheel-cache-4drdc8ob/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
WARNING: You are using pip version 20.1; however, version 20.1.1 is available.
You should consider upgrading via the '/opt/hostedtoolcache/Python/3.6.10/x64/bin/python -m pip install --upgrade pip' command.
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 153, in create_tabular_dataset
    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
    return util.load_model(name, **overrides)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
    raise IOError(Errors.E050.format(name=name))
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 477, in <module>
    test( data_path="model_tch/textcnn.json", pars_choice = "test01" )
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 442, in test
    Xtuple = get_dataset(data_pars)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 334, in get_dataset
    trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//textcnn.py", line 159, in create_tabular_dataset
    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/__init__.py", line 30, in load
    return util.load_model(name, **overrides)
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/util.py", line 169, in load_model
    raise IOError(Errors.E050.format(name=name))
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.

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
[master f3fa613] ml_store
 2 files changed, 108 insertions(+)
To github.com:arita37/mlmodels_store.git
   edccb8c..f3fa613  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//mlp.py 

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
