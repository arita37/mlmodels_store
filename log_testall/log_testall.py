
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
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
Already up to date.
[master d37ef06] ml_store
 2 files changed, 64 insertions(+), 10782 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   8d12117..d37ef06  master -> master





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
[master e17d984] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   d37ef06..e17d984  master -> master





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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-21 20:12:07.954356: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 20:12:07.959393: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-21 20:12:07.959590: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e1e20bb580 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 20:12:07.959606: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 218
Trainable params: 218
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2775 - binary_crossentropy: 1.2755500/500 [==============================] - 1s 1ms/sample - loss: 0.2887 - binary_crossentropy: 1.6605 - val_loss: 0.2824 - val_binary_crossentropy: 1.5169

  #### metrics   #################################################### 
{'MSE': 0.2852524459475077}

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
weighted_sequence_layer_1 (Weig (None, 3, 1)         0           linear0sparse_seq_emb_weighted_se
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         12          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         28          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         24          sparse_feature_2[0][0]           
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
Total params: 218
Trainable params: 218
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
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
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
Total params: 473
Trainable params: 473
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2503 - binary_crossentropy: 0.6937500/500 [==============================] - 1s 2ms/sample - loss: 0.2503 - binary_crossentropy: 0.6937 - val_loss: 0.2503 - val_binary_crossentropy: 0.6937

  #### metrics   #################################################### 
{'MSE': 0.25015426393671836}

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
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 592
Trainable params: 592
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2750 - binary_crossentropy: 1.7746500/500 [==============================] - 1s 2ms/sample - loss: 0.2821 - binary_crossentropy: 1.9207 - val_loss: 0.3318 - val_binary_crossentropy: 2.7927

  #### metrics   #################################################### 
{'MSE': 0.30676718518266854}

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
sequence_mean (InputLayer)      [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         28          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         16          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_1[0][0]           
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
Total params: 592
Trainable params: 592
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
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 468
Trainable params: 468
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3730 - binary_crossentropy: 1.0205500/500 [==============================] - 1s 3ms/sample - loss: 0.3445 - binary_crossentropy: 0.9791 - val_loss: 0.3683 - val_binary_crossentropy: 1.0278

  #### metrics   #################################################### 
{'MSE': 0.3542329107853333}

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
sequence_max (InputLayer)       [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
Total params: 468
Trainable params: 468
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
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
Total params: 153
Trainable params: 153
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3118 - binary_crossentropy: 0.8354500/500 [==============================] - 2s 3ms/sample - loss: 0.2922 - binary_crossentropy: 0.7931 - val_loss: 0.3030 - val_binary_crossentropy: 0.8187

  #### metrics   #################################################### 
{'MSE': 0.2970083665442015}

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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         7           sequence_max[0][0]               
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-21 20:13:31.452622: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:31.454721: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:31.460660: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 20:13:31.470978: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 20:13:31.472758: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 20:13:31.474314: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:31.475730: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2485 - val_binary_crossentropy: 0.6901
2020-05-21 20:13:32.822014: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:32.823937: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:32.828737: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 20:13:32.838177: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-21 20:13:32.839865: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 20:13:32.841396: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:32.842833: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24782409595764085}

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
2020-05-21 20:13:57.185737: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:57.187828: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:57.191838: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 20:13:57.197921: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 20:13:57.199247: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 20:13:57.200384: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:57.201652: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2482 - val_binary_crossentropy: 0.6896
2020-05-21 20:13:58.769570: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:58.770750: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:58.773904: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 20:13:58.779705: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-21 20:13:58.780690: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 20:13:58.781521: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:13:58.782313: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24748510339081284}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-21 20:14:34.938001: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:14:34.943682: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:14:34.959850: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 20:14:34.987806: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 20:14:34.992633: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 20:14:34.997238: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:14:35.002014: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.1410 - binary_crossentropy: 0.4707 - val_loss: 0.2506 - val_binary_crossentropy: 0.6943
2020-05-21 20:14:37.481028: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:14:37.485822: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:14:37.498411: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 20:14:37.524522: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-21 20:14:37.529479: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-21 20:14:37.533744: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-21 20:14:37.537623: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2410053557999158}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 675
Trainable params: 675
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2558 - binary_crossentropy: 0.7049500/500 [==============================] - 5s 9ms/sample - loss: 0.2575 - binary_crossentropy: 0.7085 - val_loss: 0.2535 - val_binary_crossentropy: 0.7002

  #### metrics   #################################################### 
{'MSE': 0.25532543784025186}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         10          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         10          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         18          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         10          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_5[0][0]           
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
Total params: 248
Trainable params: 248
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2496 - binary_crossentropy: 0.6993500/500 [==============================] - 5s 10ms/sample - loss: 0.2801 - binary_crossentropy: 0.7944 - val_loss: 0.2909 - val_binary_crossentropy: 0.7848

  #### metrics   #################################################### 
{'MSE': 0.2819999309050434}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         10          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 2)         10          sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         18          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         10          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         10          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_5[0][0]           
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
Total params: 248
Trainable params: 248
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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 1,894
Trainable params: 1,894
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2756 - binary_crossentropy: 0.7473500/500 [==============================] - 5s 10ms/sample - loss: 0.2647 - binary_crossentropy: 0.7242 - val_loss: 0.2639 - val_binary_crossentropy: 0.7217

  #### metrics   #################################################### 
{'MSE': 0.26253160871024545}

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
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 1,894
Trainable params: 1,894
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
regionsequence_mean (InputLayer [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         7           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 9s - loss: 0.2480 - binary_crossentropy: 0.6887500/500 [==============================] - 6s 12ms/sample - loss: 0.2527 - binary_crossentropy: 0.6985 - val_loss: 0.2537 - val_binary_crossentropy: 0.7004

  #### metrics   #################################################### 
{'MSE': 0.2529530672843991}

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
regionsequence_mean (InputLayer [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 9, 1)         7           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         4           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         2           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 9, 1)         7           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 2)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
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
Total params: 1,377
Trainable params: 1,377
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2590 - binary_crossentropy: 0.7115500/500 [==============================] - 6s 12ms/sample - loss: 0.2546 - binary_crossentropy: 0.7026 - val_loss: 0.2499 - val_binary_crossentropy: 0.6925

  #### metrics   #################################################### 
{'MSE': 0.25000018645176825}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
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
Total params: 2,967
Trainable params: 2,887
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2594 - binary_crossentropy: 0.8418500/500 [==============================] - 7s 13ms/sample - loss: 0.2853 - binary_crossentropy: 1.0344 - val_loss: 0.2928 - val_binary_crossentropy: 0.9718

  #### metrics   #################################################### 
{'MSE': 0.2894256876704941}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         8           hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         8           hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 2, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 6, 4)         16          sequence_mean[0][0]              
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
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
Total params: 2,967
Trainable params: 2,887
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
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
From github.com:arita37/mlmodels_store
   e17d984..68327b1  master     -> origin/master
Updating e17d984..68327b1
Fast-forward
 error_list/20200521/list_log_testall_20200521.md   | 734 ---------------------
 ...-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py | 627 ++++++++++++++++++
 2 files changed, 627 insertions(+), 734 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-21-20-10_51b64e342c7b2661e79b8abaa33db92672ae95c7.py
[master fd260e7] ml_store
 1 file changed, 4955 insertions(+)
To github.com:arita37/mlmodels_store.git
   68327b1..fd260e7  master -> master





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
[master a594731] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   fd260e7..a594731  master -> master





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
[master bec5272] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   a594731..bec5272  master -> master





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
[master 4527dc5] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   bec5272..4527dc5  master -> master





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

2020-05-21 20:23:56.434689: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 20:23:56.440664: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-21 20:23:56.440937: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557de0431e20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 20:23:56.440983: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 8s - loss: 1.3847
256/354 [====================>.........] - ETA: 3s - loss: 1.2437
354/354 [==============================] - 16s 44ms/step - loss: 1.2688 - val_loss: 2.1184

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
[master c6c2e31] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   4527dc5..c6c2e31  master -> master





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
[master eca3763] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   c6c2e31..eca3763  master -> master





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
[master f625ef8] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   eca3763..f625ef8  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2580480/17464789 [===>..........................] - ETA: 0s
10584064/17464789 [=================>............] - ETA: 0s
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
2020-05-21 20:25:00.630938: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-21 20:25:00.635541: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-21 20:25:00.635683: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5561f0abc680 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-21 20:25:00.635699: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.7740 - accuracy: 0.4930
 2000/25000 [=>............................] - ETA: 10s - loss: 7.8046 - accuracy: 0.4910
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.8711 - accuracy: 0.4867 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.8928 - accuracy: 0.4852
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.8230 - accuracy: 0.4898
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8123 - accuracy: 0.4905
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7696 - accuracy: 0.4933
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7625 - accuracy: 0.4938
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7382 - accuracy: 0.4953
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7065 - accuracy: 0.4974
11000/25000 [============>.................] - ETA: 4s - loss: 7.7098 - accuracy: 0.4972
12000/25000 [=============>................] - ETA: 4s - loss: 7.6858 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6855 - accuracy: 0.4988
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6929 - accuracy: 0.4983
15000/25000 [=================>............] - ETA: 3s - loss: 7.6850 - accuracy: 0.4988
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6848 - accuracy: 0.4988
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6793 - accuracy: 0.4992
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6964 - accuracy: 0.4981
19000/25000 [=====================>........] - ETA: 1s - loss: 7.7094 - accuracy: 0.4972
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6889 - accuracy: 0.4985
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6805 - accuracy: 0.4991
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6931 - accuracy: 0.4983
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6793 - accuracy: 0.4992
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6762 - accuracy: 0.4994
25000/25000 [==============================] - 10s 397us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fcbacc41240>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fcb94417ef0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6513 - accuracy: 0.5010
 2000/25000 [=>............................] - ETA: 10s - loss: 7.7510 - accuracy: 0.4945
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6820 - accuracy: 0.4990 
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6168 - accuracy: 0.5033
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6084 - accuracy: 0.5038
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.6538 - accuracy: 0.5008
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6184 - accuracy: 0.5031
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.6302 - accuracy: 0.5024
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6598 - accuracy: 0.5004
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6835 - accuracy: 0.4989
11000/25000 [============>.................] - ETA: 4s - loss: 7.6778 - accuracy: 0.4993
12000/25000 [=============>................] - ETA: 4s - loss: 7.6730 - accuracy: 0.4996
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7020 - accuracy: 0.4977
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6765 - accuracy: 0.4994
15000/25000 [=================>............] - ETA: 3s - loss: 7.6922 - accuracy: 0.4983
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6896 - accuracy: 0.4985
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6892 - accuracy: 0.4985
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6649 - accuracy: 0.5001
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6658 - accuracy: 0.5001
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6758 - accuracy: 0.4994
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6900 - accuracy: 0.4985
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6952 - accuracy: 0.4981
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6800 - accuracy: 0.4991
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6768 - accuracy: 0.4993
25000/25000 [==============================] - 10s 393us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.8813 - accuracy: 0.4860
 2000/25000 [=>............................] - ETA: 9s - loss: 7.7280 - accuracy: 0.4960 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6717 - accuracy: 0.4997
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6245 - accuracy: 0.5027
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6482 - accuracy: 0.5012
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7126 - accuracy: 0.4970
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7258 - accuracy: 0.4961
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7260 - accuracy: 0.4961
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7535 - accuracy: 0.4943
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6988 - accuracy: 0.4979
11000/25000 [============>.................] - ETA: 4s - loss: 7.6847 - accuracy: 0.4988
12000/25000 [=============>................] - ETA: 4s - loss: 7.6705 - accuracy: 0.4997
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6690 - accuracy: 0.4998
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6743 - accuracy: 0.4995
15000/25000 [=================>............] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6603 - accuracy: 0.5004
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6573 - accuracy: 0.5006
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6448 - accuracy: 0.5014
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6521 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6433 - accuracy: 0.5015
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6485 - accuracy: 0.5012
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6626 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6622 - accuracy: 0.5003
25000/25000 [==============================] - 10s 394us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   f625ef8..a7e091c  master     -> origin/master
Updating f625ef8..a7e091c
Fast-forward
 .../20200521/list_log_pullrequest_20200521.md      |   2 +-
 error_list/20200521/list_log_testall_20200521.md   | 103 +++++++++++++++++++++
 2 files changed, 104 insertions(+), 1 deletion(-)
[master 7f6f8b2] ml_store
 1 file changed, 323 insertions(+)
To github.com:arita37/mlmodels_store.git
   a7e091c..7f6f8b2  master -> master





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

13/13 [==============================] - 2s 125ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 48cf09d] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   7f6f8b2..48cf09d  master -> master





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
 1392640/11490434 [==>...........................] - ETA: 0s
 3760128/11490434 [========>.....................] - ETA: 0s
 6316032/11490434 [===============>..............] - ETA: 0s
 8871936/11490434 [======================>.......] - ETA: 0s
11313152/11490434 [============================>.] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:50 - loss: 2.3128 - categorical_accuracy: 0.2188
   64/60000 [..............................] - ETA: 4:54 - loss: 2.2909 - categorical_accuracy: 0.2188
   96/60000 [..............................] - ETA: 3:57 - loss: 2.2911 - categorical_accuracy: 0.1979
  128/60000 [..............................] - ETA: 3:27 - loss: 2.2665 - categorical_accuracy: 0.1719
  160/60000 [..............................] - ETA: 3:13 - loss: 2.2362 - categorical_accuracy: 0.1937
  192/60000 [..............................] - ETA: 2:59 - loss: 2.2018 - categorical_accuracy: 0.2135
  224/60000 [..............................] - ETA: 2:48 - loss: 2.1932 - categorical_accuracy: 0.2009
  256/60000 [..............................] - ETA: 2:41 - loss: 2.2238 - categorical_accuracy: 0.2031
  288/60000 [..............................] - ETA: 2:36 - loss: 2.2035 - categorical_accuracy: 0.2222
  320/60000 [..............................] - ETA: 2:31 - loss: 2.1743 - categorical_accuracy: 0.2469
  352/60000 [..............................] - ETA: 2:27 - loss: 2.1203 - categorical_accuracy: 0.2812
  384/60000 [..............................] - ETA: 2:25 - loss: 2.0720 - categorical_accuracy: 0.2995
  416/60000 [..............................] - ETA: 2:22 - loss: 2.0154 - categorical_accuracy: 0.3245
  448/60000 [..............................] - ETA: 2:20 - loss: 1.9902 - categorical_accuracy: 0.3304
  480/60000 [..............................] - ETA: 2:17 - loss: 1.9864 - categorical_accuracy: 0.3354
  512/60000 [..............................] - ETA: 2:15 - loss: 1.9697 - categorical_accuracy: 0.3457
  544/60000 [..............................] - ETA: 2:14 - loss: 1.9228 - categorical_accuracy: 0.3695
  576/60000 [..............................] - ETA: 2:12 - loss: 1.8913 - categorical_accuracy: 0.3819
  608/60000 [..............................] - ETA: 2:11 - loss: 1.8572 - categorical_accuracy: 0.3898
  640/60000 [..............................] - ETA: 2:10 - loss: 1.8078 - categorical_accuracy: 0.4078
  672/60000 [..............................] - ETA: 2:09 - loss: 1.7753 - categorical_accuracy: 0.4167
  704/60000 [..............................] - ETA: 2:08 - loss: 1.7733 - categorical_accuracy: 0.4190
  736/60000 [..............................] - ETA: 2:07 - loss: 1.7445 - categorical_accuracy: 0.4307
  768/60000 [..............................] - ETA: 2:07 - loss: 1.7240 - categorical_accuracy: 0.4388
  800/60000 [..............................] - ETA: 2:06 - loss: 1.6921 - categorical_accuracy: 0.4550
  832/60000 [..............................] - ETA: 2:05 - loss: 1.6552 - categorical_accuracy: 0.4651
  864/60000 [..............................] - ETA: 2:05 - loss: 1.6286 - categorical_accuracy: 0.4745
  896/60000 [..............................] - ETA: 2:04 - loss: 1.6102 - categorical_accuracy: 0.4788
  928/60000 [..............................] - ETA: 2:04 - loss: 1.5904 - categorical_accuracy: 0.4828
  960/60000 [..............................] - ETA: 2:03 - loss: 1.5704 - categorical_accuracy: 0.4896
  992/60000 [..............................] - ETA: 2:03 - loss: 1.5469 - categorical_accuracy: 0.4970
 1024/60000 [..............................] - ETA: 2:02 - loss: 1.5239 - categorical_accuracy: 0.5020
 1056/60000 [..............................] - ETA: 2:02 - loss: 1.5209 - categorical_accuracy: 0.5038
 1088/60000 [..............................] - ETA: 2:01 - loss: 1.5027 - categorical_accuracy: 0.5101
 1120/60000 [..............................] - ETA: 2:01 - loss: 1.4773 - categorical_accuracy: 0.5196
 1152/60000 [..............................] - ETA: 2:01 - loss: 1.4577 - categorical_accuracy: 0.5260
 1184/60000 [..............................] - ETA: 2:01 - loss: 1.4363 - categorical_accuracy: 0.5321
 1216/60000 [..............................] - ETA: 2:01 - loss: 1.4190 - categorical_accuracy: 0.5378
 1248/60000 [..............................] - ETA: 2:00 - loss: 1.3988 - categorical_accuracy: 0.5441
 1280/60000 [..............................] - ETA: 2:00 - loss: 1.3910 - categorical_accuracy: 0.5484
 1312/60000 [..............................] - ETA: 2:00 - loss: 1.3716 - categorical_accuracy: 0.5549
 1344/60000 [..............................] - ETA: 1:59 - loss: 1.3588 - categorical_accuracy: 0.5595
 1376/60000 [..............................] - ETA: 1:59 - loss: 1.3501 - categorical_accuracy: 0.5618
 1408/60000 [..............................] - ETA: 1:59 - loss: 1.3329 - categorical_accuracy: 0.5653
 1440/60000 [..............................] - ETA: 1:58 - loss: 1.3213 - categorical_accuracy: 0.5701
 1472/60000 [..............................] - ETA: 1:58 - loss: 1.3045 - categorical_accuracy: 0.5761
 1504/60000 [..............................] - ETA: 1:58 - loss: 1.2885 - categorical_accuracy: 0.5824
 1536/60000 [..............................] - ETA: 1:58 - loss: 1.2732 - categorical_accuracy: 0.5866
 1568/60000 [..............................] - ETA: 1:57 - loss: 1.2580 - categorical_accuracy: 0.5912
 1600/60000 [..............................] - ETA: 1:57 - loss: 1.2441 - categorical_accuracy: 0.5944
 1632/60000 [..............................] - ETA: 1:57 - loss: 1.2329 - categorical_accuracy: 0.5993
 1664/60000 [..............................] - ETA: 1:57 - loss: 1.2168 - categorical_accuracy: 0.6034
 1696/60000 [..............................] - ETA: 1:57 - loss: 1.2083 - categorical_accuracy: 0.6061
 1728/60000 [..............................] - ETA: 1:56 - loss: 1.1978 - categorical_accuracy: 0.6088
 1760/60000 [..............................] - ETA: 1:56 - loss: 1.1881 - categorical_accuracy: 0.6114
 1792/60000 [..............................] - ETA: 1:56 - loss: 1.1747 - categorical_accuracy: 0.6155
 1824/60000 [..............................] - ETA: 1:56 - loss: 1.1610 - categorical_accuracy: 0.6201
 1856/60000 [..............................] - ETA: 1:56 - loss: 1.1484 - categorical_accuracy: 0.6234
 1888/60000 [..............................] - ETA: 1:56 - loss: 1.1443 - categorical_accuracy: 0.6250
 1920/60000 [..............................] - ETA: 1:56 - loss: 1.1348 - categorical_accuracy: 0.6281
 1952/60000 [..............................] - ETA: 1:56 - loss: 1.1242 - categorical_accuracy: 0.6306
 1984/60000 [..............................] - ETA: 1:55 - loss: 1.1110 - categorical_accuracy: 0.6361
 2016/60000 [>.............................] - ETA: 1:55 - loss: 1.1021 - categorical_accuracy: 0.6389
 2048/60000 [>.............................] - ETA: 1:55 - loss: 1.0929 - categorical_accuracy: 0.6421
 2080/60000 [>.............................] - ETA: 1:55 - loss: 1.0870 - categorical_accuracy: 0.6442
 2112/60000 [>.............................] - ETA: 1:54 - loss: 1.0786 - categorical_accuracy: 0.6473
 2144/60000 [>.............................] - ETA: 1:54 - loss: 1.0682 - categorical_accuracy: 0.6507
 2176/60000 [>.............................] - ETA: 1:54 - loss: 1.0618 - categorical_accuracy: 0.6526
 2208/60000 [>.............................] - ETA: 1:54 - loss: 1.0530 - categorical_accuracy: 0.6562
 2240/60000 [>.............................] - ETA: 1:54 - loss: 1.0423 - categorical_accuracy: 0.6594
 2272/60000 [>.............................] - ETA: 1:54 - loss: 1.0355 - categorical_accuracy: 0.6615
 2304/60000 [>.............................] - ETA: 1:53 - loss: 1.0282 - categorical_accuracy: 0.6636
 2336/60000 [>.............................] - ETA: 1:53 - loss: 1.0198 - categorical_accuracy: 0.6665
 2368/60000 [>.............................] - ETA: 1:53 - loss: 1.0135 - categorical_accuracy: 0.6693
 2400/60000 [>.............................] - ETA: 1:53 - loss: 1.0062 - categorical_accuracy: 0.6717
 2432/60000 [>.............................] - ETA: 1:53 - loss: 0.9995 - categorical_accuracy: 0.6735
 2464/60000 [>.............................] - ETA: 1:53 - loss: 0.9953 - categorical_accuracy: 0.6753
 2496/60000 [>.............................] - ETA: 1:53 - loss: 0.9849 - categorical_accuracy: 0.6791
 2528/60000 [>.............................] - ETA: 1:52 - loss: 0.9767 - categorical_accuracy: 0.6820
 2560/60000 [>.............................] - ETA: 1:52 - loss: 0.9710 - categorical_accuracy: 0.6844
 2592/60000 [>.............................] - ETA: 1:52 - loss: 0.9714 - categorical_accuracy: 0.6860
 2624/60000 [>.............................] - ETA: 1:52 - loss: 0.9662 - categorical_accuracy: 0.6875
 2656/60000 [>.............................] - ETA: 1:52 - loss: 0.9581 - categorical_accuracy: 0.6901
 2688/60000 [>.............................] - ETA: 1:52 - loss: 0.9510 - categorical_accuracy: 0.6920
 2720/60000 [>.............................] - ETA: 1:52 - loss: 0.9471 - categorical_accuracy: 0.6930
 2752/60000 [>.............................] - ETA: 1:52 - loss: 0.9402 - categorical_accuracy: 0.6951
 2784/60000 [>.............................] - ETA: 1:51 - loss: 0.9335 - categorical_accuracy: 0.6972
 2816/60000 [>.............................] - ETA: 1:51 - loss: 0.9259 - categorical_accuracy: 0.6992
 2848/60000 [>.............................] - ETA: 1:51 - loss: 0.9186 - categorical_accuracy: 0.7019
 2880/60000 [>.............................] - ETA: 1:51 - loss: 0.9158 - categorical_accuracy: 0.7031
 2912/60000 [>.............................] - ETA: 1:51 - loss: 0.9107 - categorical_accuracy: 0.7054
 2944/60000 [>.............................] - ETA: 1:51 - loss: 0.9085 - categorical_accuracy: 0.7062
 2976/60000 [>.............................] - ETA: 1:51 - loss: 0.9027 - categorical_accuracy: 0.7080
 3008/60000 [>.............................] - ETA: 1:50 - loss: 0.8981 - categorical_accuracy: 0.7098
 3040/60000 [>.............................] - ETA: 1:50 - loss: 0.8965 - categorical_accuracy: 0.7109
 3072/60000 [>.............................] - ETA: 1:50 - loss: 0.8910 - categorical_accuracy: 0.7132
 3104/60000 [>.............................] - ETA: 1:50 - loss: 0.8852 - categorical_accuracy: 0.7152
 3136/60000 [>.............................] - ETA: 1:50 - loss: 0.8823 - categorical_accuracy: 0.7162
 3168/60000 [>.............................] - ETA: 1:50 - loss: 0.8760 - categorical_accuracy: 0.7184
 3200/60000 [>.............................] - ETA: 1:50 - loss: 0.8705 - categorical_accuracy: 0.7197
 3232/60000 [>.............................] - ETA: 1:50 - loss: 0.8678 - categorical_accuracy: 0.7206
 3264/60000 [>.............................] - ETA: 1:50 - loss: 0.8632 - categorical_accuracy: 0.7224
 3296/60000 [>.............................] - ETA: 1:49 - loss: 0.8586 - categorical_accuracy: 0.7233
 3328/60000 [>.............................] - ETA: 1:49 - loss: 0.8519 - categorical_accuracy: 0.7254
 3360/60000 [>.............................] - ETA: 1:49 - loss: 0.8484 - categorical_accuracy: 0.7265
 3392/60000 [>.............................] - ETA: 1:49 - loss: 0.8438 - categorical_accuracy: 0.7279
 3424/60000 [>.............................] - ETA: 1:49 - loss: 0.8394 - categorical_accuracy: 0.7293
 3456/60000 [>.............................] - ETA: 1:49 - loss: 0.8378 - categorical_accuracy: 0.7303
 3488/60000 [>.............................] - ETA: 1:49 - loss: 0.8341 - categorical_accuracy: 0.7314
 3520/60000 [>.............................] - ETA: 1:49 - loss: 0.8298 - categorical_accuracy: 0.7330
 3552/60000 [>.............................] - ETA: 1:49 - loss: 0.8255 - categorical_accuracy: 0.7345
 3584/60000 [>.............................] - ETA: 1:49 - loss: 0.8201 - categorical_accuracy: 0.7360
 3616/60000 [>.............................] - ETA: 1:49 - loss: 0.8146 - categorical_accuracy: 0.7378
 3648/60000 [>.............................] - ETA: 1:49 - loss: 0.8138 - categorical_accuracy: 0.7379
 3680/60000 [>.............................] - ETA: 1:49 - loss: 0.8108 - categorical_accuracy: 0.7389
 3712/60000 [>.............................] - ETA: 1:49 - loss: 0.8049 - categorical_accuracy: 0.7411
 3744/60000 [>.............................] - ETA: 1:48 - loss: 0.8009 - categorical_accuracy: 0.7425
 3776/60000 [>.............................] - ETA: 1:48 - loss: 0.7972 - categorical_accuracy: 0.7434
 3808/60000 [>.............................] - ETA: 1:48 - loss: 0.7920 - categorical_accuracy: 0.7453
 3840/60000 [>.............................] - ETA: 1:48 - loss: 0.7874 - categorical_accuracy: 0.7469
 3872/60000 [>.............................] - ETA: 1:48 - loss: 0.7838 - categorical_accuracy: 0.7485
 3904/60000 [>.............................] - ETA: 1:48 - loss: 0.7806 - categorical_accuracy: 0.7487
 3936/60000 [>.............................] - ETA: 1:48 - loss: 0.7756 - categorical_accuracy: 0.7505
 3968/60000 [>.............................] - ETA: 1:48 - loss: 0.7721 - categorical_accuracy: 0.7518
 4000/60000 [=>............................] - ETA: 1:48 - loss: 0.7678 - categorical_accuracy: 0.7533
 4032/60000 [=>............................] - ETA: 1:48 - loss: 0.7641 - categorical_accuracy: 0.7545
 4064/60000 [=>............................] - ETA: 1:47 - loss: 0.7613 - categorical_accuracy: 0.7552
 4096/60000 [=>............................] - ETA: 1:47 - loss: 0.7567 - categorical_accuracy: 0.7566
 4128/60000 [=>............................] - ETA: 1:47 - loss: 0.7525 - categorical_accuracy: 0.7582
 4160/60000 [=>............................] - ETA: 1:47 - loss: 0.7485 - categorical_accuracy: 0.7594
 4192/60000 [=>............................] - ETA: 1:47 - loss: 0.7477 - categorical_accuracy: 0.7595
 4224/60000 [=>............................] - ETA: 1:47 - loss: 0.7452 - categorical_accuracy: 0.7604
 4256/60000 [=>............................] - ETA: 1:47 - loss: 0.7418 - categorical_accuracy: 0.7613
 4288/60000 [=>............................] - ETA: 1:47 - loss: 0.7384 - categorical_accuracy: 0.7624
 4320/60000 [=>............................] - ETA: 1:47 - loss: 0.7351 - categorical_accuracy: 0.7637
 4352/60000 [=>............................] - ETA: 1:47 - loss: 0.7320 - categorical_accuracy: 0.7642
 4384/60000 [=>............................] - ETA: 1:47 - loss: 0.7293 - categorical_accuracy: 0.7655
 4416/60000 [=>............................] - ETA: 1:46 - loss: 0.7257 - categorical_accuracy: 0.7668
 4448/60000 [=>............................] - ETA: 1:46 - loss: 0.7210 - categorical_accuracy: 0.7684
 4480/60000 [=>............................] - ETA: 1:46 - loss: 0.7189 - categorical_accuracy: 0.7692
 4512/60000 [=>............................] - ETA: 1:46 - loss: 0.7168 - categorical_accuracy: 0.7697
 4544/60000 [=>............................] - ETA: 1:46 - loss: 0.7176 - categorical_accuracy: 0.7700
 4576/60000 [=>............................] - ETA: 1:46 - loss: 0.7144 - categorical_accuracy: 0.7712
 4608/60000 [=>............................] - ETA: 1:46 - loss: 0.7106 - categorical_accuracy: 0.7724
 4640/60000 [=>............................] - ETA: 1:46 - loss: 0.7065 - categorical_accuracy: 0.7735
 4672/60000 [=>............................] - ETA: 1:46 - loss: 0.7038 - categorical_accuracy: 0.7744
 4704/60000 [=>............................] - ETA: 1:46 - loss: 0.7013 - categorical_accuracy: 0.7749
 4736/60000 [=>............................] - ETA: 1:46 - loss: 0.6979 - categorical_accuracy: 0.7760
 4768/60000 [=>............................] - ETA: 1:46 - loss: 0.6966 - categorical_accuracy: 0.7766
 4800/60000 [=>............................] - ETA: 1:45 - loss: 0.6957 - categorical_accuracy: 0.7775
 4832/60000 [=>............................] - ETA: 1:45 - loss: 0.6937 - categorical_accuracy: 0.7781
 4864/60000 [=>............................] - ETA: 1:45 - loss: 0.6897 - categorical_accuracy: 0.7794
 4896/60000 [=>............................] - ETA: 1:45 - loss: 0.6888 - categorical_accuracy: 0.7800
 4928/60000 [=>............................] - ETA: 1:45 - loss: 0.6881 - categorical_accuracy: 0.7804
 4960/60000 [=>............................] - ETA: 1:45 - loss: 0.6858 - categorical_accuracy: 0.7815
 4992/60000 [=>............................] - ETA: 1:45 - loss: 0.6824 - categorical_accuracy: 0.7827
 5024/60000 [=>............................] - ETA: 1:45 - loss: 0.6808 - categorical_accuracy: 0.7828
 5056/60000 [=>............................] - ETA: 1:45 - loss: 0.6784 - categorical_accuracy: 0.7832
 5088/60000 [=>............................] - ETA: 1:45 - loss: 0.6760 - categorical_accuracy: 0.7840
 5120/60000 [=>............................] - ETA: 1:45 - loss: 0.6742 - categorical_accuracy: 0.7846
 5152/60000 [=>............................] - ETA: 1:45 - loss: 0.6719 - categorical_accuracy: 0.7853
 5184/60000 [=>............................] - ETA: 1:45 - loss: 0.6690 - categorical_accuracy: 0.7863
 5216/60000 [=>............................] - ETA: 1:45 - loss: 0.6662 - categorical_accuracy: 0.7872
 5248/60000 [=>............................] - ETA: 1:44 - loss: 0.6644 - categorical_accuracy: 0.7877
 5280/60000 [=>............................] - ETA: 1:44 - loss: 0.6630 - categorical_accuracy: 0.7879
 5312/60000 [=>............................] - ETA: 1:44 - loss: 0.6599 - categorical_accuracy: 0.7888
 5344/60000 [=>............................] - ETA: 1:44 - loss: 0.6578 - categorical_accuracy: 0.7895
 5376/60000 [=>............................] - ETA: 1:44 - loss: 0.6549 - categorical_accuracy: 0.7902
 5408/60000 [=>............................] - ETA: 1:44 - loss: 0.6534 - categorical_accuracy: 0.7905
 5440/60000 [=>............................] - ETA: 1:44 - loss: 0.6510 - categorical_accuracy: 0.7912
 5472/60000 [=>............................] - ETA: 1:44 - loss: 0.6500 - categorical_accuracy: 0.7917
 5504/60000 [=>............................] - ETA: 1:44 - loss: 0.6480 - categorical_accuracy: 0.7925
 5536/60000 [=>............................] - ETA: 1:44 - loss: 0.6456 - categorical_accuracy: 0.7934
 5568/60000 [=>............................] - ETA: 1:44 - loss: 0.6428 - categorical_accuracy: 0.7944
 5600/60000 [=>............................] - ETA: 1:44 - loss: 0.6394 - categorical_accuracy: 0.7955
 5632/60000 [=>............................] - ETA: 1:43 - loss: 0.6377 - categorical_accuracy: 0.7962
 5664/60000 [=>............................] - ETA: 1:43 - loss: 0.6344 - categorical_accuracy: 0.7973
 5696/60000 [=>............................] - ETA: 1:43 - loss: 0.6316 - categorical_accuracy: 0.7983
 5728/60000 [=>............................] - ETA: 1:43 - loss: 0.6305 - categorical_accuracy: 0.7985
 5760/60000 [=>............................] - ETA: 1:43 - loss: 0.6288 - categorical_accuracy: 0.7990
 5792/60000 [=>............................] - ETA: 1:43 - loss: 0.6261 - categorical_accuracy: 0.7997
 5824/60000 [=>............................] - ETA: 1:43 - loss: 0.6243 - categorical_accuracy: 0.8001
 5856/60000 [=>............................] - ETA: 1:43 - loss: 0.6213 - categorical_accuracy: 0.8011
 5888/60000 [=>............................] - ETA: 1:43 - loss: 0.6198 - categorical_accuracy: 0.8018
 5920/60000 [=>............................] - ETA: 1:43 - loss: 0.6173 - categorical_accuracy: 0.8025
 5952/60000 [=>............................] - ETA: 1:43 - loss: 0.6158 - categorical_accuracy: 0.8028
 5984/60000 [=>............................] - ETA: 1:43 - loss: 0.6135 - categorical_accuracy: 0.8033
 6016/60000 [==>...........................] - ETA: 1:43 - loss: 0.6110 - categorical_accuracy: 0.8039
 6048/60000 [==>...........................] - ETA: 1:43 - loss: 0.6080 - categorical_accuracy: 0.8049
 6080/60000 [==>...........................] - ETA: 1:42 - loss: 0.6053 - categorical_accuracy: 0.8056
 6112/60000 [==>...........................] - ETA: 1:42 - loss: 0.6030 - categorical_accuracy: 0.8063
 6144/60000 [==>...........................] - ETA: 1:42 - loss: 0.6002 - categorical_accuracy: 0.8071
 6176/60000 [==>...........................] - ETA: 1:42 - loss: 0.5979 - categorical_accuracy: 0.8078
 6208/60000 [==>...........................] - ETA: 1:42 - loss: 0.5969 - categorical_accuracy: 0.8082
 6240/60000 [==>...........................] - ETA: 1:42 - loss: 0.5950 - categorical_accuracy: 0.8088
 6272/60000 [==>...........................] - ETA: 1:42 - loss: 0.5935 - categorical_accuracy: 0.8095
 6304/60000 [==>...........................] - ETA: 1:42 - loss: 0.5912 - categorical_accuracy: 0.8103
 6336/60000 [==>...........................] - ETA: 1:42 - loss: 0.5902 - categorical_accuracy: 0.8106
 6368/60000 [==>...........................] - ETA: 1:42 - loss: 0.5897 - categorical_accuracy: 0.8108
 6400/60000 [==>...........................] - ETA: 1:42 - loss: 0.5873 - categorical_accuracy: 0.8116
 6432/60000 [==>...........................] - ETA: 1:42 - loss: 0.5867 - categorical_accuracy: 0.8120
 6464/60000 [==>...........................] - ETA: 1:42 - loss: 0.5846 - categorical_accuracy: 0.8127
 6496/60000 [==>...........................] - ETA: 1:42 - loss: 0.5828 - categorical_accuracy: 0.8133
 6528/60000 [==>...........................] - ETA: 1:41 - loss: 0.5807 - categorical_accuracy: 0.8140
 6560/60000 [==>...........................] - ETA: 1:41 - loss: 0.5790 - categorical_accuracy: 0.8143
 6592/60000 [==>...........................] - ETA: 1:41 - loss: 0.5770 - categorical_accuracy: 0.8149
 6624/60000 [==>...........................] - ETA: 1:41 - loss: 0.5747 - categorical_accuracy: 0.8157
 6656/60000 [==>...........................] - ETA: 1:41 - loss: 0.5739 - categorical_accuracy: 0.8160
 6688/60000 [==>...........................] - ETA: 1:41 - loss: 0.5728 - categorical_accuracy: 0.8164
 6720/60000 [==>...........................] - ETA: 1:41 - loss: 0.5706 - categorical_accuracy: 0.8173
 6752/60000 [==>...........................] - ETA: 1:41 - loss: 0.5689 - categorical_accuracy: 0.8177
 6784/60000 [==>...........................] - ETA: 1:41 - loss: 0.5679 - categorical_accuracy: 0.8181
 6816/60000 [==>...........................] - ETA: 1:41 - loss: 0.5659 - categorical_accuracy: 0.8187
 6848/60000 [==>...........................] - ETA: 1:41 - loss: 0.5642 - categorical_accuracy: 0.8192
 6880/60000 [==>...........................] - ETA: 1:41 - loss: 0.5636 - categorical_accuracy: 0.8190
 6912/60000 [==>...........................] - ETA: 1:41 - loss: 0.5615 - categorical_accuracy: 0.8197
 6944/60000 [==>...........................] - ETA: 1:41 - loss: 0.5596 - categorical_accuracy: 0.8203
 6976/60000 [==>...........................] - ETA: 1:41 - loss: 0.5578 - categorical_accuracy: 0.8208
 7008/60000 [==>...........................] - ETA: 1:41 - loss: 0.5557 - categorical_accuracy: 0.8216
 7040/60000 [==>...........................] - ETA: 1:40 - loss: 0.5535 - categorical_accuracy: 0.8222
 7072/60000 [==>...........................] - ETA: 1:40 - loss: 0.5519 - categorical_accuracy: 0.8227
 7104/60000 [==>...........................] - ETA: 1:40 - loss: 0.5502 - categorical_accuracy: 0.8232
 7136/60000 [==>...........................] - ETA: 1:40 - loss: 0.5484 - categorical_accuracy: 0.8239
 7168/60000 [==>...........................] - ETA: 1:40 - loss: 0.5466 - categorical_accuracy: 0.8245
 7200/60000 [==>...........................] - ETA: 1:40 - loss: 0.5453 - categorical_accuracy: 0.8250
 7232/60000 [==>...........................] - ETA: 1:40 - loss: 0.5448 - categorical_accuracy: 0.8256
 7264/60000 [==>...........................] - ETA: 1:40 - loss: 0.5433 - categorical_accuracy: 0.8259
 7296/60000 [==>...........................] - ETA: 1:40 - loss: 0.5416 - categorical_accuracy: 0.8263
 7328/60000 [==>...........................] - ETA: 1:40 - loss: 0.5395 - categorical_accuracy: 0.8271
 7360/60000 [==>...........................] - ETA: 1:40 - loss: 0.5386 - categorical_accuracy: 0.8274
 7392/60000 [==>...........................] - ETA: 1:40 - loss: 0.5378 - categorical_accuracy: 0.8278
 7424/60000 [==>...........................] - ETA: 1:40 - loss: 0.5356 - categorical_accuracy: 0.8285
 7456/60000 [==>...........................] - ETA: 1:40 - loss: 0.5353 - categorical_accuracy: 0.8287
 7488/60000 [==>...........................] - ETA: 1:40 - loss: 0.5342 - categorical_accuracy: 0.8291
 7520/60000 [==>...........................] - ETA: 1:39 - loss: 0.5326 - categorical_accuracy: 0.8294
 7552/60000 [==>...........................] - ETA: 1:39 - loss: 0.5316 - categorical_accuracy: 0.8298
 7584/60000 [==>...........................] - ETA: 1:39 - loss: 0.5306 - categorical_accuracy: 0.8300
 7616/60000 [==>...........................] - ETA: 1:39 - loss: 0.5297 - categorical_accuracy: 0.8305
 7648/60000 [==>...........................] - ETA: 1:39 - loss: 0.5280 - categorical_accuracy: 0.8311
 7680/60000 [==>...........................] - ETA: 1:39 - loss: 0.5280 - categorical_accuracy: 0.8315
 7712/60000 [==>...........................] - ETA: 1:39 - loss: 0.5267 - categorical_accuracy: 0.8320
 7744/60000 [==>...........................] - ETA: 1:39 - loss: 0.5253 - categorical_accuracy: 0.8325
 7776/60000 [==>...........................] - ETA: 1:39 - loss: 0.5240 - categorical_accuracy: 0.8329
 7808/60000 [==>...........................] - ETA: 1:39 - loss: 0.5230 - categorical_accuracy: 0.8332
 7840/60000 [==>...........................] - ETA: 1:39 - loss: 0.5221 - categorical_accuracy: 0.8337
 7872/60000 [==>...........................] - ETA: 1:39 - loss: 0.5206 - categorical_accuracy: 0.8342
 7904/60000 [==>...........................] - ETA: 1:39 - loss: 0.5193 - categorical_accuracy: 0.8348
 7936/60000 [==>...........................] - ETA: 1:38 - loss: 0.5182 - categorical_accuracy: 0.8352
 7968/60000 [==>...........................] - ETA: 1:38 - loss: 0.5172 - categorical_accuracy: 0.8355
 8000/60000 [===>..........................] - ETA: 1:38 - loss: 0.5160 - categorical_accuracy: 0.8359
 8032/60000 [===>..........................] - ETA: 1:38 - loss: 0.5144 - categorical_accuracy: 0.8364
 8064/60000 [===>..........................] - ETA: 1:38 - loss: 0.5136 - categorical_accuracy: 0.8367
 8096/60000 [===>..........................] - ETA: 1:38 - loss: 0.5136 - categorical_accuracy: 0.8368
 8128/60000 [===>..........................] - ETA: 1:38 - loss: 0.5123 - categorical_accuracy: 0.8374
 8160/60000 [===>..........................] - ETA: 1:38 - loss: 0.5119 - categorical_accuracy: 0.8377
 8192/60000 [===>..........................] - ETA: 1:38 - loss: 0.5110 - categorical_accuracy: 0.8381
 8224/60000 [===>..........................] - ETA: 1:38 - loss: 0.5104 - categorical_accuracy: 0.8384
 8256/60000 [===>..........................] - ETA: 1:38 - loss: 0.5090 - categorical_accuracy: 0.8388
 8288/60000 [===>..........................] - ETA: 1:38 - loss: 0.5088 - categorical_accuracy: 0.8388
 8320/60000 [===>..........................] - ETA: 1:38 - loss: 0.5077 - categorical_accuracy: 0.8392
 8352/60000 [===>..........................] - ETA: 1:38 - loss: 0.5075 - categorical_accuracy: 0.8393
 8384/60000 [===>..........................] - ETA: 1:38 - loss: 0.5064 - categorical_accuracy: 0.8396
 8416/60000 [===>..........................] - ETA: 1:37 - loss: 0.5058 - categorical_accuracy: 0.8398
 8448/60000 [===>..........................] - ETA: 1:37 - loss: 0.5044 - categorical_accuracy: 0.8403
 8480/60000 [===>..........................] - ETA: 1:37 - loss: 0.5032 - categorical_accuracy: 0.8407
 8512/60000 [===>..........................] - ETA: 1:37 - loss: 0.5031 - categorical_accuracy: 0.8408
 8544/60000 [===>..........................] - ETA: 1:37 - loss: 0.5017 - categorical_accuracy: 0.8413
 8576/60000 [===>..........................] - ETA: 1:37 - loss: 0.5012 - categorical_accuracy: 0.8415
 8608/60000 [===>..........................] - ETA: 1:37 - loss: 0.4998 - categorical_accuracy: 0.8421
 8640/60000 [===>..........................] - ETA: 1:37 - loss: 0.4995 - categorical_accuracy: 0.8422
 8672/60000 [===>..........................] - ETA: 1:37 - loss: 0.4988 - categorical_accuracy: 0.8425
 8704/60000 [===>..........................] - ETA: 1:37 - loss: 0.4986 - categorical_accuracy: 0.8426
 8736/60000 [===>..........................] - ETA: 1:37 - loss: 0.4973 - categorical_accuracy: 0.8432
 8768/60000 [===>..........................] - ETA: 1:37 - loss: 0.4968 - categorical_accuracy: 0.8433
 8800/60000 [===>..........................] - ETA: 1:37 - loss: 0.4963 - categorical_accuracy: 0.8438
 8832/60000 [===>..........................] - ETA: 1:37 - loss: 0.4950 - categorical_accuracy: 0.8442
 8864/60000 [===>..........................] - ETA: 1:37 - loss: 0.4944 - categorical_accuracy: 0.8445
 8896/60000 [===>..........................] - ETA: 1:37 - loss: 0.4930 - categorical_accuracy: 0.8450
 8928/60000 [===>..........................] - ETA: 1:37 - loss: 0.4917 - categorical_accuracy: 0.8453
 8960/60000 [===>..........................] - ETA: 1:36 - loss: 0.4903 - categorical_accuracy: 0.8459
 8992/60000 [===>..........................] - ETA: 1:36 - loss: 0.4894 - categorical_accuracy: 0.8461
 9024/60000 [===>..........................] - ETA: 1:36 - loss: 0.4888 - categorical_accuracy: 0.8464
 9056/60000 [===>..........................] - ETA: 1:36 - loss: 0.4879 - categorical_accuracy: 0.8466
 9088/60000 [===>..........................] - ETA: 1:36 - loss: 0.4871 - categorical_accuracy: 0.8468
 9120/60000 [===>..........................] - ETA: 1:36 - loss: 0.4858 - categorical_accuracy: 0.8473
 9152/60000 [===>..........................] - ETA: 1:36 - loss: 0.4847 - categorical_accuracy: 0.8477
 9184/60000 [===>..........................] - ETA: 1:36 - loss: 0.4844 - categorical_accuracy: 0.8477
 9216/60000 [===>..........................] - ETA: 1:36 - loss: 0.4841 - categorical_accuracy: 0.8480
 9248/60000 [===>..........................] - ETA: 1:36 - loss: 0.4828 - categorical_accuracy: 0.8483
 9280/60000 [===>..........................] - ETA: 1:36 - loss: 0.4813 - categorical_accuracy: 0.8487
 9312/60000 [===>..........................] - ETA: 1:36 - loss: 0.4809 - categorical_accuracy: 0.8488
 9344/60000 [===>..........................] - ETA: 1:36 - loss: 0.4799 - categorical_accuracy: 0.8492
 9376/60000 [===>..........................] - ETA: 1:36 - loss: 0.4789 - categorical_accuracy: 0.8495
 9408/60000 [===>..........................] - ETA: 1:36 - loss: 0.4784 - categorical_accuracy: 0.8497
 9440/60000 [===>..........................] - ETA: 1:36 - loss: 0.4779 - categorical_accuracy: 0.8498
 9472/60000 [===>..........................] - ETA: 1:35 - loss: 0.4770 - categorical_accuracy: 0.8501
 9504/60000 [===>..........................] - ETA: 1:35 - loss: 0.4775 - categorical_accuracy: 0.8497
 9536/60000 [===>..........................] - ETA: 1:35 - loss: 0.4768 - categorical_accuracy: 0.8500
 9568/60000 [===>..........................] - ETA: 1:35 - loss: 0.4760 - categorical_accuracy: 0.8502
 9600/60000 [===>..........................] - ETA: 1:35 - loss: 0.4754 - categorical_accuracy: 0.8505
 9632/60000 [===>..........................] - ETA: 1:35 - loss: 0.4744 - categorical_accuracy: 0.8508
 9664/60000 [===>..........................] - ETA: 1:35 - loss: 0.4735 - categorical_accuracy: 0.8510
 9696/60000 [===>..........................] - ETA: 1:35 - loss: 0.4727 - categorical_accuracy: 0.8512
 9728/60000 [===>..........................] - ETA: 1:35 - loss: 0.4714 - categorical_accuracy: 0.8516
 9760/60000 [===>..........................] - ETA: 1:35 - loss: 0.4711 - categorical_accuracy: 0.8518
 9792/60000 [===>..........................] - ETA: 1:35 - loss: 0.4700 - categorical_accuracy: 0.8522
 9824/60000 [===>..........................] - ETA: 1:35 - loss: 0.4688 - categorical_accuracy: 0.8525
 9856/60000 [===>..........................] - ETA: 1:35 - loss: 0.4679 - categorical_accuracy: 0.8527
 9888/60000 [===>..........................] - ETA: 1:35 - loss: 0.4670 - categorical_accuracy: 0.8530
 9920/60000 [===>..........................] - ETA: 1:35 - loss: 0.4667 - categorical_accuracy: 0.8533
 9952/60000 [===>..........................] - ETA: 1:34 - loss: 0.4655 - categorical_accuracy: 0.8536
 9984/60000 [===>..........................] - ETA: 1:34 - loss: 0.4649 - categorical_accuracy: 0.8539
10016/60000 [====>.........................] - ETA: 1:34 - loss: 0.4636 - categorical_accuracy: 0.8543
10048/60000 [====>.........................] - ETA: 1:34 - loss: 0.4626 - categorical_accuracy: 0.8547
10080/60000 [====>.........................] - ETA: 1:34 - loss: 0.4616 - categorical_accuracy: 0.8549
10112/60000 [====>.........................] - ETA: 1:34 - loss: 0.4620 - categorical_accuracy: 0.8549
10144/60000 [====>.........................] - ETA: 1:34 - loss: 0.4613 - categorical_accuracy: 0.8552
10176/60000 [====>.........................] - ETA: 1:34 - loss: 0.4609 - categorical_accuracy: 0.8553
10208/60000 [====>.........................] - ETA: 1:34 - loss: 0.4600 - categorical_accuracy: 0.8556
10240/60000 [====>.........................] - ETA: 1:34 - loss: 0.4593 - categorical_accuracy: 0.8559
10272/60000 [====>.........................] - ETA: 1:34 - loss: 0.4583 - categorical_accuracy: 0.8563
10304/60000 [====>.........................] - ETA: 1:34 - loss: 0.4576 - categorical_accuracy: 0.8566
10336/60000 [====>.........................] - ETA: 1:34 - loss: 0.4568 - categorical_accuracy: 0.8569
10368/60000 [====>.........................] - ETA: 1:34 - loss: 0.4556 - categorical_accuracy: 0.8573
10400/60000 [====>.........................] - ETA: 1:34 - loss: 0.4546 - categorical_accuracy: 0.8576
10432/60000 [====>.........................] - ETA: 1:34 - loss: 0.4539 - categorical_accuracy: 0.8577
10464/60000 [====>.........................] - ETA: 1:33 - loss: 0.4532 - categorical_accuracy: 0.8580
10496/60000 [====>.........................] - ETA: 1:33 - loss: 0.4521 - categorical_accuracy: 0.8583
10528/60000 [====>.........................] - ETA: 1:33 - loss: 0.4514 - categorical_accuracy: 0.8586
10560/60000 [====>.........................] - ETA: 1:33 - loss: 0.4505 - categorical_accuracy: 0.8589
10592/60000 [====>.........................] - ETA: 1:33 - loss: 0.4499 - categorical_accuracy: 0.8590
10624/60000 [====>.........................] - ETA: 1:33 - loss: 0.4494 - categorical_accuracy: 0.8592
10656/60000 [====>.........................] - ETA: 1:33 - loss: 0.4494 - categorical_accuracy: 0.8591
10688/60000 [====>.........................] - ETA: 1:33 - loss: 0.4483 - categorical_accuracy: 0.8595
10720/60000 [====>.........................] - ETA: 1:33 - loss: 0.4477 - categorical_accuracy: 0.8597
10752/60000 [====>.........................] - ETA: 1:33 - loss: 0.4468 - categorical_accuracy: 0.8598
10784/60000 [====>.........................] - ETA: 1:33 - loss: 0.4471 - categorical_accuracy: 0.8601
10816/60000 [====>.........................] - ETA: 1:33 - loss: 0.4466 - categorical_accuracy: 0.8603
10848/60000 [====>.........................] - ETA: 1:33 - loss: 0.4458 - categorical_accuracy: 0.8606
10880/60000 [====>.........................] - ETA: 1:33 - loss: 0.4450 - categorical_accuracy: 0.8609
10912/60000 [====>.........................] - ETA: 1:33 - loss: 0.4442 - categorical_accuracy: 0.8612
10944/60000 [====>.........................] - ETA: 1:33 - loss: 0.4434 - categorical_accuracy: 0.8614
10976/60000 [====>.........................] - ETA: 1:32 - loss: 0.4424 - categorical_accuracy: 0.8618
11008/60000 [====>.........................] - ETA: 1:32 - loss: 0.4416 - categorical_accuracy: 0.8621
11040/60000 [====>.........................] - ETA: 1:32 - loss: 0.4404 - categorical_accuracy: 0.8625
11072/60000 [====>.........................] - ETA: 1:32 - loss: 0.4395 - categorical_accuracy: 0.8628
11104/60000 [====>.........................] - ETA: 1:32 - loss: 0.4391 - categorical_accuracy: 0.8630
11136/60000 [====>.........................] - ETA: 1:32 - loss: 0.4382 - categorical_accuracy: 0.8633
11168/60000 [====>.........................] - ETA: 1:32 - loss: 0.4375 - categorical_accuracy: 0.8635
11200/60000 [====>.........................] - ETA: 1:32 - loss: 0.4365 - categorical_accuracy: 0.8638
11232/60000 [====>.........................] - ETA: 1:32 - loss: 0.4356 - categorical_accuracy: 0.8641
11264/60000 [====>.........................] - ETA: 1:32 - loss: 0.4351 - categorical_accuracy: 0.8641
11296/60000 [====>.........................] - ETA: 1:32 - loss: 0.4341 - categorical_accuracy: 0.8644
11328/60000 [====>.........................] - ETA: 1:32 - loss: 0.4334 - categorical_accuracy: 0.8645
11360/60000 [====>.........................] - ETA: 1:32 - loss: 0.4322 - categorical_accuracy: 0.8649
11392/60000 [====>.........................] - ETA: 1:32 - loss: 0.4318 - categorical_accuracy: 0.8650
11424/60000 [====>.........................] - ETA: 1:32 - loss: 0.4307 - categorical_accuracy: 0.8654
11456/60000 [====>.........................] - ETA: 1:32 - loss: 0.4299 - categorical_accuracy: 0.8656
11488/60000 [====>.........................] - ETA: 1:31 - loss: 0.4289 - categorical_accuracy: 0.8659
11520/60000 [====>.........................] - ETA: 1:31 - loss: 0.4285 - categorical_accuracy: 0.8661
11552/60000 [====>.........................] - ETA: 1:31 - loss: 0.4281 - categorical_accuracy: 0.8662
11584/60000 [====>.........................] - ETA: 1:31 - loss: 0.4278 - categorical_accuracy: 0.8662
11616/60000 [====>.........................] - ETA: 1:31 - loss: 0.4272 - categorical_accuracy: 0.8665
11648/60000 [====>.........................] - ETA: 1:31 - loss: 0.4270 - categorical_accuracy: 0.8665
11680/60000 [====>.........................] - ETA: 1:31 - loss: 0.4263 - categorical_accuracy: 0.8668
11712/60000 [====>.........................] - ETA: 1:31 - loss: 0.4253 - categorical_accuracy: 0.8671
11744/60000 [====>.........................] - ETA: 1:31 - loss: 0.4252 - categorical_accuracy: 0.8673
11776/60000 [====>.........................] - ETA: 1:31 - loss: 0.4243 - categorical_accuracy: 0.8675
11808/60000 [====>.........................] - ETA: 1:31 - loss: 0.4234 - categorical_accuracy: 0.8678
11840/60000 [====>.........................] - ETA: 1:31 - loss: 0.4227 - categorical_accuracy: 0.8679
11872/60000 [====>.........................] - ETA: 1:31 - loss: 0.4227 - categorical_accuracy: 0.8679
11904/60000 [====>.........................] - ETA: 1:31 - loss: 0.4228 - categorical_accuracy: 0.8679
11936/60000 [====>.........................] - ETA: 1:31 - loss: 0.4226 - categorical_accuracy: 0.8679
11968/60000 [====>.........................] - ETA: 1:31 - loss: 0.4217 - categorical_accuracy: 0.8681
12000/60000 [=====>........................] - ETA: 1:30 - loss: 0.4207 - categorical_accuracy: 0.8685
12032/60000 [=====>........................] - ETA: 1:30 - loss: 0.4199 - categorical_accuracy: 0.8688
12064/60000 [=====>........................] - ETA: 1:30 - loss: 0.4197 - categorical_accuracy: 0.8689
12096/60000 [=====>........................] - ETA: 1:30 - loss: 0.4191 - categorical_accuracy: 0.8690
12128/60000 [=====>........................] - ETA: 1:30 - loss: 0.4189 - categorical_accuracy: 0.8693
12160/60000 [=====>........................] - ETA: 1:30 - loss: 0.4179 - categorical_accuracy: 0.8697
12192/60000 [=====>........................] - ETA: 1:30 - loss: 0.4175 - categorical_accuracy: 0.8698
12224/60000 [=====>........................] - ETA: 1:30 - loss: 0.4171 - categorical_accuracy: 0.8698
12256/60000 [=====>........................] - ETA: 1:30 - loss: 0.4162 - categorical_accuracy: 0.8701
12288/60000 [=====>........................] - ETA: 1:30 - loss: 0.4156 - categorical_accuracy: 0.8703
12320/60000 [=====>........................] - ETA: 1:30 - loss: 0.4146 - categorical_accuracy: 0.8706
12352/60000 [=====>........................] - ETA: 1:30 - loss: 0.4142 - categorical_accuracy: 0.8706
12384/60000 [=====>........................] - ETA: 1:30 - loss: 0.4144 - categorical_accuracy: 0.8707
12416/60000 [=====>........................] - ETA: 1:30 - loss: 0.4141 - categorical_accuracy: 0.8707
12448/60000 [=====>........................] - ETA: 1:30 - loss: 0.4135 - categorical_accuracy: 0.8709
12480/60000 [=====>........................] - ETA: 1:30 - loss: 0.4126 - categorical_accuracy: 0.8712
12512/60000 [=====>........................] - ETA: 1:29 - loss: 0.4118 - categorical_accuracy: 0.8715
12544/60000 [=====>........................] - ETA: 1:29 - loss: 0.4112 - categorical_accuracy: 0.8717
12576/60000 [=====>........................] - ETA: 1:29 - loss: 0.4104 - categorical_accuracy: 0.8719
12608/60000 [=====>........................] - ETA: 1:29 - loss: 0.4098 - categorical_accuracy: 0.8721
12640/60000 [=====>........................] - ETA: 1:29 - loss: 0.4090 - categorical_accuracy: 0.8723
12672/60000 [=====>........................] - ETA: 1:29 - loss: 0.4085 - categorical_accuracy: 0.8725
12704/60000 [=====>........................] - ETA: 1:29 - loss: 0.4083 - categorical_accuracy: 0.8724
12736/60000 [=====>........................] - ETA: 1:29 - loss: 0.4076 - categorical_accuracy: 0.8726
12768/60000 [=====>........................] - ETA: 1:29 - loss: 0.4070 - categorical_accuracy: 0.8728
12800/60000 [=====>........................] - ETA: 1:29 - loss: 0.4068 - categorical_accuracy: 0.8730
12832/60000 [=====>........................] - ETA: 1:29 - loss: 0.4059 - categorical_accuracy: 0.8732
12864/60000 [=====>........................] - ETA: 1:29 - loss: 0.4054 - categorical_accuracy: 0.8734
12896/60000 [=====>........................] - ETA: 1:29 - loss: 0.4054 - categorical_accuracy: 0.8734
12928/60000 [=====>........................] - ETA: 1:29 - loss: 0.4046 - categorical_accuracy: 0.8735
12960/60000 [=====>........................] - ETA: 1:29 - loss: 0.4039 - categorical_accuracy: 0.8738
12992/60000 [=====>........................] - ETA: 1:28 - loss: 0.4035 - categorical_accuracy: 0.8738
13024/60000 [=====>........................] - ETA: 1:28 - loss: 0.4034 - categorical_accuracy: 0.8738
13056/60000 [=====>........................] - ETA: 1:28 - loss: 0.4029 - categorical_accuracy: 0.8739
13088/60000 [=====>........................] - ETA: 1:28 - loss: 0.4022 - categorical_accuracy: 0.8742
13120/60000 [=====>........................] - ETA: 1:28 - loss: 0.4018 - categorical_accuracy: 0.8742
13152/60000 [=====>........................] - ETA: 1:28 - loss: 0.4009 - categorical_accuracy: 0.8745
13184/60000 [=====>........................] - ETA: 1:28 - loss: 0.4003 - categorical_accuracy: 0.8746
13216/60000 [=====>........................] - ETA: 1:28 - loss: 0.4003 - categorical_accuracy: 0.8745
13248/60000 [=====>........................] - ETA: 1:28 - loss: 0.3997 - categorical_accuracy: 0.8747
13280/60000 [=====>........................] - ETA: 1:28 - loss: 0.3990 - categorical_accuracy: 0.8750
13312/60000 [=====>........................] - ETA: 1:28 - loss: 0.3987 - categorical_accuracy: 0.8752
13344/60000 [=====>........................] - ETA: 1:28 - loss: 0.3978 - categorical_accuracy: 0.8754
13376/60000 [=====>........................] - ETA: 1:28 - loss: 0.3977 - categorical_accuracy: 0.8755
13408/60000 [=====>........................] - ETA: 1:28 - loss: 0.3970 - categorical_accuracy: 0.8758
13440/60000 [=====>........................] - ETA: 1:28 - loss: 0.3960 - categorical_accuracy: 0.8761
13472/60000 [=====>........................] - ETA: 1:28 - loss: 0.3952 - categorical_accuracy: 0.8764
13504/60000 [=====>........................] - ETA: 1:28 - loss: 0.3951 - categorical_accuracy: 0.8765
13536/60000 [=====>........................] - ETA: 1:27 - loss: 0.3950 - categorical_accuracy: 0.8763
13568/60000 [=====>........................] - ETA: 1:27 - loss: 0.3942 - categorical_accuracy: 0.8766
13600/60000 [=====>........................] - ETA: 1:27 - loss: 0.3941 - categorical_accuracy: 0.8768
13632/60000 [=====>........................] - ETA: 1:27 - loss: 0.3933 - categorical_accuracy: 0.8771
13664/60000 [=====>........................] - ETA: 1:27 - loss: 0.3926 - categorical_accuracy: 0.8773
13696/60000 [=====>........................] - ETA: 1:27 - loss: 0.3919 - categorical_accuracy: 0.8775
13728/60000 [=====>........................] - ETA: 1:27 - loss: 0.3912 - categorical_accuracy: 0.8777
13760/60000 [=====>........................] - ETA: 1:27 - loss: 0.3905 - categorical_accuracy: 0.8779
13792/60000 [=====>........................] - ETA: 1:27 - loss: 0.3899 - categorical_accuracy: 0.8781
13824/60000 [=====>........................] - ETA: 1:27 - loss: 0.3894 - categorical_accuracy: 0.8781
13856/60000 [=====>........................] - ETA: 1:27 - loss: 0.3887 - categorical_accuracy: 0.8782
13888/60000 [=====>........................] - ETA: 1:27 - loss: 0.3884 - categorical_accuracy: 0.8784
13920/60000 [=====>........................] - ETA: 1:27 - loss: 0.3876 - categorical_accuracy: 0.8787
13952/60000 [=====>........................] - ETA: 1:27 - loss: 0.3869 - categorical_accuracy: 0.8789
13984/60000 [=====>........................] - ETA: 1:27 - loss: 0.3863 - categorical_accuracy: 0.8791
14016/60000 [======>.......................] - ETA: 1:26 - loss: 0.3860 - categorical_accuracy: 0.8791
14048/60000 [======>.......................] - ETA: 1:26 - loss: 0.3854 - categorical_accuracy: 0.8793
14080/60000 [======>.......................] - ETA: 1:26 - loss: 0.3846 - categorical_accuracy: 0.8796
14112/60000 [======>.......................] - ETA: 1:26 - loss: 0.3845 - categorical_accuracy: 0.8798
14144/60000 [======>.......................] - ETA: 1:26 - loss: 0.3844 - categorical_accuracy: 0.8800
14176/60000 [======>.......................] - ETA: 1:26 - loss: 0.3838 - categorical_accuracy: 0.8802
14208/60000 [======>.......................] - ETA: 1:26 - loss: 0.3835 - categorical_accuracy: 0.8804
14240/60000 [======>.......................] - ETA: 1:26 - loss: 0.3827 - categorical_accuracy: 0.8807
14272/60000 [======>.......................] - ETA: 1:26 - loss: 0.3819 - categorical_accuracy: 0.8810
14304/60000 [======>.......................] - ETA: 1:26 - loss: 0.3813 - categorical_accuracy: 0.8811
14336/60000 [======>.......................] - ETA: 1:26 - loss: 0.3806 - categorical_accuracy: 0.8813
14368/60000 [======>.......................] - ETA: 1:26 - loss: 0.3800 - categorical_accuracy: 0.8816
14400/60000 [======>.......................] - ETA: 1:26 - loss: 0.3793 - categorical_accuracy: 0.8819
14432/60000 [======>.......................] - ETA: 1:26 - loss: 0.3787 - categorical_accuracy: 0.8821
14464/60000 [======>.......................] - ETA: 1:26 - loss: 0.3782 - categorical_accuracy: 0.8821
14496/60000 [======>.......................] - ETA: 1:26 - loss: 0.3777 - categorical_accuracy: 0.8822
14528/60000 [======>.......................] - ETA: 1:26 - loss: 0.3774 - categorical_accuracy: 0.8823
14560/60000 [======>.......................] - ETA: 1:26 - loss: 0.3769 - categorical_accuracy: 0.8823
14592/60000 [======>.......................] - ETA: 1:25 - loss: 0.3765 - categorical_accuracy: 0.8824
14624/60000 [======>.......................] - ETA: 1:25 - loss: 0.3759 - categorical_accuracy: 0.8827
14656/60000 [======>.......................] - ETA: 1:25 - loss: 0.3756 - categorical_accuracy: 0.8828
14688/60000 [======>.......................] - ETA: 1:25 - loss: 0.3753 - categorical_accuracy: 0.8830
14720/60000 [======>.......................] - ETA: 1:25 - loss: 0.3746 - categorical_accuracy: 0.8832
14752/60000 [======>.......................] - ETA: 1:25 - loss: 0.3739 - categorical_accuracy: 0.8834
14784/60000 [======>.......................] - ETA: 1:25 - loss: 0.3737 - categorical_accuracy: 0.8835
14816/60000 [======>.......................] - ETA: 1:25 - loss: 0.3731 - categorical_accuracy: 0.8836
14848/60000 [======>.......................] - ETA: 1:25 - loss: 0.3728 - categorical_accuracy: 0.8838
14880/60000 [======>.......................] - ETA: 1:25 - loss: 0.3726 - categorical_accuracy: 0.8838
14912/60000 [======>.......................] - ETA: 1:25 - loss: 0.3726 - categorical_accuracy: 0.8837
14944/60000 [======>.......................] - ETA: 1:25 - loss: 0.3722 - categorical_accuracy: 0.8838
14976/60000 [======>.......................] - ETA: 1:25 - loss: 0.3719 - categorical_accuracy: 0.8838
15008/60000 [======>.......................] - ETA: 1:25 - loss: 0.3714 - categorical_accuracy: 0.8839
15040/60000 [======>.......................] - ETA: 1:25 - loss: 0.3713 - categorical_accuracy: 0.8840
15072/60000 [======>.......................] - ETA: 1:25 - loss: 0.3710 - categorical_accuracy: 0.8841
15104/60000 [======>.......................] - ETA: 1:25 - loss: 0.3703 - categorical_accuracy: 0.8843
15136/60000 [======>.......................] - ETA: 1:24 - loss: 0.3697 - categorical_accuracy: 0.8846
15168/60000 [======>.......................] - ETA: 1:24 - loss: 0.3692 - categorical_accuracy: 0.8847
15200/60000 [======>.......................] - ETA: 1:24 - loss: 0.3688 - categorical_accuracy: 0.8849
15232/60000 [======>.......................] - ETA: 1:24 - loss: 0.3686 - categorical_accuracy: 0.8849
15264/60000 [======>.......................] - ETA: 1:24 - loss: 0.3688 - categorical_accuracy: 0.8849
15296/60000 [======>.......................] - ETA: 1:24 - loss: 0.3684 - categorical_accuracy: 0.8850
15328/60000 [======>.......................] - ETA: 1:24 - loss: 0.3680 - categorical_accuracy: 0.8851
15360/60000 [======>.......................] - ETA: 1:24 - loss: 0.3683 - categorical_accuracy: 0.8852
15392/60000 [======>.......................] - ETA: 1:24 - loss: 0.3677 - categorical_accuracy: 0.8854
15424/60000 [======>.......................] - ETA: 1:24 - loss: 0.3670 - categorical_accuracy: 0.8856
15456/60000 [======>.......................] - ETA: 1:24 - loss: 0.3665 - categorical_accuracy: 0.8858
15488/60000 [======>.......................] - ETA: 1:24 - loss: 0.3659 - categorical_accuracy: 0.8860
15520/60000 [======>.......................] - ETA: 1:24 - loss: 0.3654 - categorical_accuracy: 0.8861
15552/60000 [======>.......................] - ETA: 1:24 - loss: 0.3647 - categorical_accuracy: 0.8863
15584/60000 [======>.......................] - ETA: 1:24 - loss: 0.3642 - categorical_accuracy: 0.8865
15616/60000 [======>.......................] - ETA: 1:24 - loss: 0.3640 - categorical_accuracy: 0.8866
15648/60000 [======>.......................] - ETA: 1:23 - loss: 0.3638 - categorical_accuracy: 0.8867
15680/60000 [======>.......................] - ETA: 1:23 - loss: 0.3632 - categorical_accuracy: 0.8869
15712/60000 [======>.......................] - ETA: 1:23 - loss: 0.3629 - categorical_accuracy: 0.8870
15744/60000 [======>.......................] - ETA: 1:23 - loss: 0.3622 - categorical_accuracy: 0.8873
15776/60000 [======>.......................] - ETA: 1:23 - loss: 0.3616 - categorical_accuracy: 0.8874
15808/60000 [======>.......................] - ETA: 1:23 - loss: 0.3614 - categorical_accuracy: 0.8874
15840/60000 [======>.......................] - ETA: 1:23 - loss: 0.3608 - categorical_accuracy: 0.8876
15872/60000 [======>.......................] - ETA: 1:23 - loss: 0.3603 - categorical_accuracy: 0.8877
15904/60000 [======>.......................] - ETA: 1:23 - loss: 0.3601 - categorical_accuracy: 0.8879
15936/60000 [======>.......................] - ETA: 1:23 - loss: 0.3595 - categorical_accuracy: 0.8881
15968/60000 [======>.......................] - ETA: 1:23 - loss: 0.3589 - categorical_accuracy: 0.8883
16000/60000 [=======>......................] - ETA: 1:23 - loss: 0.3584 - categorical_accuracy: 0.8885
16032/60000 [=======>......................] - ETA: 1:23 - loss: 0.3578 - categorical_accuracy: 0.8887
16064/60000 [=======>......................] - ETA: 1:23 - loss: 0.3575 - categorical_accuracy: 0.8888
16096/60000 [=======>......................] - ETA: 1:23 - loss: 0.3569 - categorical_accuracy: 0.8889
16128/60000 [=======>......................] - ETA: 1:23 - loss: 0.3564 - categorical_accuracy: 0.8890
16160/60000 [=======>......................] - ETA: 1:22 - loss: 0.3560 - categorical_accuracy: 0.8891
16192/60000 [=======>......................] - ETA: 1:22 - loss: 0.3553 - categorical_accuracy: 0.8893
16224/60000 [=======>......................] - ETA: 1:22 - loss: 0.3550 - categorical_accuracy: 0.8894
16256/60000 [=======>......................] - ETA: 1:22 - loss: 0.3544 - categorical_accuracy: 0.8896
16288/60000 [=======>......................] - ETA: 1:22 - loss: 0.3543 - categorical_accuracy: 0.8897
16320/60000 [=======>......................] - ETA: 1:22 - loss: 0.3542 - categorical_accuracy: 0.8898
16352/60000 [=======>......................] - ETA: 1:22 - loss: 0.3537 - categorical_accuracy: 0.8899
16384/60000 [=======>......................] - ETA: 1:22 - loss: 0.3532 - categorical_accuracy: 0.8900
16416/60000 [=======>......................] - ETA: 1:22 - loss: 0.3530 - categorical_accuracy: 0.8901
16448/60000 [=======>......................] - ETA: 1:22 - loss: 0.3526 - categorical_accuracy: 0.8902
16480/60000 [=======>......................] - ETA: 1:22 - loss: 0.3522 - categorical_accuracy: 0.8902
16512/60000 [=======>......................] - ETA: 1:22 - loss: 0.3517 - categorical_accuracy: 0.8904
16544/60000 [=======>......................] - ETA: 1:22 - loss: 0.3514 - categorical_accuracy: 0.8905
16576/60000 [=======>......................] - ETA: 1:22 - loss: 0.3511 - categorical_accuracy: 0.8906
16608/60000 [=======>......................] - ETA: 1:22 - loss: 0.3508 - categorical_accuracy: 0.8907
16640/60000 [=======>......................] - ETA: 1:22 - loss: 0.3505 - categorical_accuracy: 0.8908
16672/60000 [=======>......................] - ETA: 1:21 - loss: 0.3506 - categorical_accuracy: 0.8908
16704/60000 [=======>......................] - ETA: 1:21 - loss: 0.3502 - categorical_accuracy: 0.8909
16736/60000 [=======>......................] - ETA: 1:21 - loss: 0.3502 - categorical_accuracy: 0.8910
16768/60000 [=======>......................] - ETA: 1:21 - loss: 0.3502 - categorical_accuracy: 0.8909
16800/60000 [=======>......................] - ETA: 1:21 - loss: 0.3496 - categorical_accuracy: 0.8911
16832/60000 [=======>......................] - ETA: 1:21 - loss: 0.3491 - categorical_accuracy: 0.8913
16864/60000 [=======>......................] - ETA: 1:21 - loss: 0.3492 - categorical_accuracy: 0.8914
16896/60000 [=======>......................] - ETA: 1:21 - loss: 0.3487 - categorical_accuracy: 0.8916
16928/60000 [=======>......................] - ETA: 1:21 - loss: 0.3483 - categorical_accuracy: 0.8918
16960/60000 [=======>......................] - ETA: 1:21 - loss: 0.3481 - categorical_accuracy: 0.8919
16992/60000 [=======>......................] - ETA: 1:21 - loss: 0.3475 - categorical_accuracy: 0.8921
17024/60000 [=======>......................] - ETA: 1:21 - loss: 0.3471 - categorical_accuracy: 0.8922
17056/60000 [=======>......................] - ETA: 1:21 - loss: 0.3465 - categorical_accuracy: 0.8924
17088/60000 [=======>......................] - ETA: 1:21 - loss: 0.3460 - categorical_accuracy: 0.8926
17120/60000 [=======>......................] - ETA: 1:21 - loss: 0.3459 - categorical_accuracy: 0.8926
17152/60000 [=======>......................] - ETA: 1:21 - loss: 0.3454 - categorical_accuracy: 0.8927
17184/60000 [=======>......................] - ETA: 1:21 - loss: 0.3450 - categorical_accuracy: 0.8928
17216/60000 [=======>......................] - ETA: 1:20 - loss: 0.3445 - categorical_accuracy: 0.8930
17248/60000 [=======>......................] - ETA: 1:20 - loss: 0.3444 - categorical_accuracy: 0.8930
17280/60000 [=======>......................] - ETA: 1:20 - loss: 0.3445 - categorical_accuracy: 0.8929
17312/60000 [=======>......................] - ETA: 1:20 - loss: 0.3441 - categorical_accuracy: 0.8930
17344/60000 [=======>......................] - ETA: 1:20 - loss: 0.3440 - categorical_accuracy: 0.8931
17376/60000 [=======>......................] - ETA: 1:20 - loss: 0.3438 - categorical_accuracy: 0.8932
17408/60000 [=======>......................] - ETA: 1:20 - loss: 0.3436 - categorical_accuracy: 0.8933
17440/60000 [=======>......................] - ETA: 1:20 - loss: 0.3434 - categorical_accuracy: 0.8933
17472/60000 [=======>......................] - ETA: 1:20 - loss: 0.3428 - categorical_accuracy: 0.8935
17504/60000 [=======>......................] - ETA: 1:20 - loss: 0.3422 - categorical_accuracy: 0.8937
17536/60000 [=======>......................] - ETA: 1:20 - loss: 0.3424 - categorical_accuracy: 0.8939
17568/60000 [=======>......................] - ETA: 1:20 - loss: 0.3418 - categorical_accuracy: 0.8941
17600/60000 [=======>......................] - ETA: 1:20 - loss: 0.3418 - categorical_accuracy: 0.8941
17632/60000 [=======>......................] - ETA: 1:20 - loss: 0.3415 - categorical_accuracy: 0.8942
17664/60000 [=======>......................] - ETA: 1:20 - loss: 0.3415 - categorical_accuracy: 0.8942
17696/60000 [=======>......................] - ETA: 1:20 - loss: 0.3410 - categorical_accuracy: 0.8944
17728/60000 [=======>......................] - ETA: 1:19 - loss: 0.3410 - categorical_accuracy: 0.8944
17760/60000 [=======>......................] - ETA: 1:19 - loss: 0.3407 - categorical_accuracy: 0.8945
17792/60000 [=======>......................] - ETA: 1:19 - loss: 0.3402 - categorical_accuracy: 0.8947
17824/60000 [=======>......................] - ETA: 1:19 - loss: 0.3397 - categorical_accuracy: 0.8948
17856/60000 [=======>......................] - ETA: 1:19 - loss: 0.3392 - categorical_accuracy: 0.8949
17888/60000 [=======>......................] - ETA: 1:19 - loss: 0.3389 - categorical_accuracy: 0.8951
17920/60000 [=======>......................] - ETA: 1:19 - loss: 0.3383 - categorical_accuracy: 0.8953
17952/60000 [=======>......................] - ETA: 1:19 - loss: 0.3379 - categorical_accuracy: 0.8953
17984/60000 [=======>......................] - ETA: 1:19 - loss: 0.3382 - categorical_accuracy: 0.8954
18016/60000 [========>.....................] - ETA: 1:19 - loss: 0.3382 - categorical_accuracy: 0.8954
18048/60000 [========>.....................] - ETA: 1:19 - loss: 0.3377 - categorical_accuracy: 0.8956
18080/60000 [========>.....................] - ETA: 1:19 - loss: 0.3378 - categorical_accuracy: 0.8957
18112/60000 [========>.....................] - ETA: 1:19 - loss: 0.3379 - categorical_accuracy: 0.8955
18144/60000 [========>.....................] - ETA: 1:19 - loss: 0.3376 - categorical_accuracy: 0.8956
18176/60000 [========>.....................] - ETA: 1:19 - loss: 0.3376 - categorical_accuracy: 0.8956
18208/60000 [========>.....................] - ETA: 1:19 - loss: 0.3371 - categorical_accuracy: 0.8958
18240/60000 [========>.....................] - ETA: 1:18 - loss: 0.3367 - categorical_accuracy: 0.8959
18272/60000 [========>.....................] - ETA: 1:18 - loss: 0.3362 - categorical_accuracy: 0.8961
18304/60000 [========>.....................] - ETA: 1:18 - loss: 0.3357 - categorical_accuracy: 0.8963
18336/60000 [========>.....................] - ETA: 1:18 - loss: 0.3355 - categorical_accuracy: 0.8963
18368/60000 [========>.....................] - ETA: 1:18 - loss: 0.3355 - categorical_accuracy: 0.8963
18400/60000 [========>.....................] - ETA: 1:18 - loss: 0.3352 - categorical_accuracy: 0.8964
18432/60000 [========>.....................] - ETA: 1:18 - loss: 0.3349 - categorical_accuracy: 0.8965
18464/60000 [========>.....................] - ETA: 1:18 - loss: 0.3347 - categorical_accuracy: 0.8966
18496/60000 [========>.....................] - ETA: 1:18 - loss: 0.3342 - categorical_accuracy: 0.8967
18528/60000 [========>.....................] - ETA: 1:18 - loss: 0.3338 - categorical_accuracy: 0.8969
18560/60000 [========>.....................] - ETA: 1:18 - loss: 0.3336 - categorical_accuracy: 0.8970
18592/60000 [========>.....................] - ETA: 1:18 - loss: 0.3335 - categorical_accuracy: 0.8969
18624/60000 [========>.....................] - ETA: 1:18 - loss: 0.3331 - categorical_accuracy: 0.8971
18656/60000 [========>.....................] - ETA: 1:18 - loss: 0.3328 - categorical_accuracy: 0.8971
18688/60000 [========>.....................] - ETA: 1:18 - loss: 0.3325 - categorical_accuracy: 0.8972
18720/60000 [========>.....................] - ETA: 1:18 - loss: 0.3320 - categorical_accuracy: 0.8973
18752/60000 [========>.....................] - ETA: 1:17 - loss: 0.3316 - categorical_accuracy: 0.8975
18784/60000 [========>.....................] - ETA: 1:17 - loss: 0.3312 - categorical_accuracy: 0.8975
18816/60000 [========>.....................] - ETA: 1:17 - loss: 0.3308 - categorical_accuracy: 0.8976
18848/60000 [========>.....................] - ETA: 1:17 - loss: 0.3306 - categorical_accuracy: 0.8978
18880/60000 [========>.....................] - ETA: 1:17 - loss: 0.3302 - categorical_accuracy: 0.8979
18912/60000 [========>.....................] - ETA: 1:17 - loss: 0.3298 - categorical_accuracy: 0.8981
18944/60000 [========>.....................] - ETA: 1:17 - loss: 0.3297 - categorical_accuracy: 0.8982
18976/60000 [========>.....................] - ETA: 1:17 - loss: 0.3292 - categorical_accuracy: 0.8983
19008/60000 [========>.....................] - ETA: 1:17 - loss: 0.3288 - categorical_accuracy: 0.8985
19040/60000 [========>.....................] - ETA: 1:17 - loss: 0.3285 - categorical_accuracy: 0.8985
19072/60000 [========>.....................] - ETA: 1:17 - loss: 0.3282 - categorical_accuracy: 0.8985
19104/60000 [========>.....................] - ETA: 1:17 - loss: 0.3278 - categorical_accuracy: 0.8986
19136/60000 [========>.....................] - ETA: 1:17 - loss: 0.3279 - categorical_accuracy: 0.8986
19168/60000 [========>.....................] - ETA: 1:17 - loss: 0.3276 - categorical_accuracy: 0.8987
19200/60000 [========>.....................] - ETA: 1:17 - loss: 0.3274 - categorical_accuracy: 0.8987
19232/60000 [========>.....................] - ETA: 1:17 - loss: 0.3274 - categorical_accuracy: 0.8987
19264/60000 [========>.....................] - ETA: 1:16 - loss: 0.3275 - categorical_accuracy: 0.8987
19296/60000 [========>.....................] - ETA: 1:16 - loss: 0.3273 - categorical_accuracy: 0.8987
19328/60000 [========>.....................] - ETA: 1:16 - loss: 0.3272 - categorical_accuracy: 0.8988
19360/60000 [========>.....................] - ETA: 1:16 - loss: 0.3267 - categorical_accuracy: 0.8990
19392/60000 [========>.....................] - ETA: 1:16 - loss: 0.3266 - categorical_accuracy: 0.8990
19424/60000 [========>.....................] - ETA: 1:16 - loss: 0.3264 - categorical_accuracy: 0.8991
19456/60000 [========>.....................] - ETA: 1:16 - loss: 0.3260 - categorical_accuracy: 0.8992
19488/60000 [========>.....................] - ETA: 1:16 - loss: 0.3257 - categorical_accuracy: 0.8993
19520/60000 [========>.....................] - ETA: 1:16 - loss: 0.3254 - categorical_accuracy: 0.8994
19552/60000 [========>.....................] - ETA: 1:16 - loss: 0.3253 - categorical_accuracy: 0.8994
19584/60000 [========>.....................] - ETA: 1:16 - loss: 0.3250 - categorical_accuracy: 0.8995
19616/60000 [========>.....................] - ETA: 1:16 - loss: 0.3251 - categorical_accuracy: 0.8994
19648/60000 [========>.....................] - ETA: 1:16 - loss: 0.3249 - categorical_accuracy: 0.8994
19680/60000 [========>.....................] - ETA: 1:16 - loss: 0.3247 - categorical_accuracy: 0.8994
19712/60000 [========>.....................] - ETA: 1:16 - loss: 0.3242 - categorical_accuracy: 0.8996
19744/60000 [========>.....................] - ETA: 1:16 - loss: 0.3239 - categorical_accuracy: 0.8997
19776/60000 [========>.....................] - ETA: 1:16 - loss: 0.3236 - categorical_accuracy: 0.8997
19808/60000 [========>.....................] - ETA: 1:15 - loss: 0.3231 - categorical_accuracy: 0.8999
19840/60000 [========>.....................] - ETA: 1:15 - loss: 0.3226 - categorical_accuracy: 0.9001
19872/60000 [========>.....................] - ETA: 1:15 - loss: 0.3223 - categorical_accuracy: 0.9002
19904/60000 [========>.....................] - ETA: 1:15 - loss: 0.3219 - categorical_accuracy: 0.9002
19936/60000 [========>.....................] - ETA: 1:15 - loss: 0.3217 - categorical_accuracy: 0.9003
19968/60000 [========>.....................] - ETA: 1:15 - loss: 0.3214 - categorical_accuracy: 0.9004
20000/60000 [=========>....................] - ETA: 1:15 - loss: 0.3214 - categorical_accuracy: 0.9004
20032/60000 [=========>....................] - ETA: 1:15 - loss: 0.3210 - categorical_accuracy: 0.9005
20064/60000 [=========>....................] - ETA: 1:15 - loss: 0.3207 - categorical_accuracy: 0.9006
20096/60000 [=========>....................] - ETA: 1:15 - loss: 0.3203 - categorical_accuracy: 0.9007
20128/60000 [=========>....................] - ETA: 1:15 - loss: 0.3201 - categorical_accuracy: 0.9007
20160/60000 [=========>....................] - ETA: 1:15 - loss: 0.3200 - categorical_accuracy: 0.9008
20192/60000 [=========>....................] - ETA: 1:15 - loss: 0.3196 - categorical_accuracy: 0.9010
20224/60000 [=========>....................] - ETA: 1:15 - loss: 0.3192 - categorical_accuracy: 0.9011
20256/60000 [=========>....................] - ETA: 1:15 - loss: 0.3187 - categorical_accuracy: 0.9013
20288/60000 [=========>....................] - ETA: 1:15 - loss: 0.3185 - categorical_accuracy: 0.9012
20320/60000 [=========>....................] - ETA: 1:14 - loss: 0.3183 - categorical_accuracy: 0.9012
20352/60000 [=========>....................] - ETA: 1:14 - loss: 0.3180 - categorical_accuracy: 0.9013
20384/60000 [=========>....................] - ETA: 1:14 - loss: 0.3178 - categorical_accuracy: 0.9014
20416/60000 [=========>....................] - ETA: 1:14 - loss: 0.3177 - categorical_accuracy: 0.9014
20448/60000 [=========>....................] - ETA: 1:14 - loss: 0.3176 - categorical_accuracy: 0.9015
20480/60000 [=========>....................] - ETA: 1:14 - loss: 0.3176 - categorical_accuracy: 0.9015
20512/60000 [=========>....................] - ETA: 1:14 - loss: 0.3173 - categorical_accuracy: 0.9016
20544/60000 [=========>....................] - ETA: 1:14 - loss: 0.3169 - categorical_accuracy: 0.9017
20576/60000 [=========>....................] - ETA: 1:14 - loss: 0.3165 - categorical_accuracy: 0.9019
20608/60000 [=========>....................] - ETA: 1:14 - loss: 0.3161 - categorical_accuracy: 0.9020
20640/60000 [=========>....................] - ETA: 1:14 - loss: 0.3159 - categorical_accuracy: 0.9021
20672/60000 [=========>....................] - ETA: 1:14 - loss: 0.3156 - categorical_accuracy: 0.9022
20704/60000 [=========>....................] - ETA: 1:14 - loss: 0.3152 - categorical_accuracy: 0.9023
20736/60000 [=========>....................] - ETA: 1:14 - loss: 0.3150 - categorical_accuracy: 0.9024
20768/60000 [=========>....................] - ETA: 1:14 - loss: 0.3154 - categorical_accuracy: 0.9024
20800/60000 [=========>....................] - ETA: 1:14 - loss: 0.3150 - categorical_accuracy: 0.9026
20832/60000 [=========>....................] - ETA: 1:13 - loss: 0.3147 - categorical_accuracy: 0.9027
20864/60000 [=========>....................] - ETA: 1:13 - loss: 0.3144 - categorical_accuracy: 0.9028
20896/60000 [=========>....................] - ETA: 1:13 - loss: 0.3140 - categorical_accuracy: 0.9029
20928/60000 [=========>....................] - ETA: 1:13 - loss: 0.3136 - categorical_accuracy: 0.9030
20960/60000 [=========>....................] - ETA: 1:13 - loss: 0.3133 - categorical_accuracy: 0.9031
20992/60000 [=========>....................] - ETA: 1:13 - loss: 0.3129 - categorical_accuracy: 0.9033
21024/60000 [=========>....................] - ETA: 1:13 - loss: 0.3125 - categorical_accuracy: 0.9034
21056/60000 [=========>....................] - ETA: 1:13 - loss: 0.3123 - categorical_accuracy: 0.9034
21088/60000 [=========>....................] - ETA: 1:13 - loss: 0.3121 - categorical_accuracy: 0.9035
21120/60000 [=========>....................] - ETA: 1:13 - loss: 0.3119 - categorical_accuracy: 0.9036
21152/60000 [=========>....................] - ETA: 1:13 - loss: 0.3118 - categorical_accuracy: 0.9036
21184/60000 [=========>....................] - ETA: 1:13 - loss: 0.3121 - categorical_accuracy: 0.9037
21216/60000 [=========>....................] - ETA: 1:13 - loss: 0.3118 - categorical_accuracy: 0.9038
21248/60000 [=========>....................] - ETA: 1:13 - loss: 0.3114 - categorical_accuracy: 0.9039
21280/60000 [=========>....................] - ETA: 1:13 - loss: 0.3110 - categorical_accuracy: 0.9040
21312/60000 [=========>....................] - ETA: 1:13 - loss: 0.3107 - categorical_accuracy: 0.9041
21344/60000 [=========>....................] - ETA: 1:12 - loss: 0.3102 - categorical_accuracy: 0.9043
21376/60000 [=========>....................] - ETA: 1:12 - loss: 0.3100 - categorical_accuracy: 0.9043
21408/60000 [=========>....................] - ETA: 1:12 - loss: 0.3097 - categorical_accuracy: 0.9045
21440/60000 [=========>....................] - ETA: 1:12 - loss: 0.3096 - categorical_accuracy: 0.9045
21472/60000 [=========>....................] - ETA: 1:12 - loss: 0.3092 - categorical_accuracy: 0.9046
21504/60000 [=========>....................] - ETA: 1:12 - loss: 0.3090 - categorical_accuracy: 0.9047
21536/60000 [=========>....................] - ETA: 1:12 - loss: 0.3086 - categorical_accuracy: 0.9048
21568/60000 [=========>....................] - ETA: 1:12 - loss: 0.3081 - categorical_accuracy: 0.9050
21600/60000 [=========>....................] - ETA: 1:12 - loss: 0.3080 - categorical_accuracy: 0.9050
21632/60000 [=========>....................] - ETA: 1:12 - loss: 0.3076 - categorical_accuracy: 0.9050
21664/60000 [=========>....................] - ETA: 1:12 - loss: 0.3075 - categorical_accuracy: 0.9051
21696/60000 [=========>....................] - ETA: 1:12 - loss: 0.3072 - categorical_accuracy: 0.9052
21728/60000 [=========>....................] - ETA: 1:12 - loss: 0.3069 - categorical_accuracy: 0.9052
21760/60000 [=========>....................] - ETA: 1:12 - loss: 0.3065 - categorical_accuracy: 0.9054
21792/60000 [=========>....................] - ETA: 1:12 - loss: 0.3064 - categorical_accuracy: 0.9054
21824/60000 [=========>....................] - ETA: 1:12 - loss: 0.3060 - categorical_accuracy: 0.9055
21856/60000 [=========>....................] - ETA: 1:11 - loss: 0.3057 - categorical_accuracy: 0.9056
21888/60000 [=========>....................] - ETA: 1:11 - loss: 0.3055 - categorical_accuracy: 0.9056
21920/60000 [=========>....................] - ETA: 1:11 - loss: 0.3053 - categorical_accuracy: 0.9057
21952/60000 [=========>....................] - ETA: 1:11 - loss: 0.3050 - categorical_accuracy: 0.9058
21984/60000 [=========>....................] - ETA: 1:11 - loss: 0.3048 - categorical_accuracy: 0.9059
22016/60000 [==========>...................] - ETA: 1:11 - loss: 0.3045 - categorical_accuracy: 0.9060
22048/60000 [==========>...................] - ETA: 1:11 - loss: 0.3041 - categorical_accuracy: 0.9061
22080/60000 [==========>...................] - ETA: 1:11 - loss: 0.3037 - categorical_accuracy: 0.9062
22112/60000 [==========>...................] - ETA: 1:11 - loss: 0.3033 - categorical_accuracy: 0.9063
22144/60000 [==========>...................] - ETA: 1:11 - loss: 0.3029 - categorical_accuracy: 0.9065
22176/60000 [==========>...................] - ETA: 1:11 - loss: 0.3028 - categorical_accuracy: 0.9066
22208/60000 [==========>...................] - ETA: 1:11 - loss: 0.3025 - categorical_accuracy: 0.9066
22240/60000 [==========>...................] - ETA: 1:11 - loss: 0.3026 - categorical_accuracy: 0.9066
22272/60000 [==========>...................] - ETA: 1:11 - loss: 0.3023 - categorical_accuracy: 0.9067
22304/60000 [==========>...................] - ETA: 1:11 - loss: 0.3022 - categorical_accuracy: 0.9067
22336/60000 [==========>...................] - ETA: 1:11 - loss: 0.3023 - categorical_accuracy: 0.9068
22368/60000 [==========>...................] - ETA: 1:11 - loss: 0.3022 - categorical_accuracy: 0.9068
22400/60000 [==========>...................] - ETA: 1:10 - loss: 0.3020 - categorical_accuracy: 0.9069
22432/60000 [==========>...................] - ETA: 1:10 - loss: 0.3019 - categorical_accuracy: 0.9070
22464/60000 [==========>...................] - ETA: 1:10 - loss: 0.3017 - categorical_accuracy: 0.9070
22496/60000 [==========>...................] - ETA: 1:10 - loss: 0.3014 - categorical_accuracy: 0.9070
22528/60000 [==========>...................] - ETA: 1:10 - loss: 0.3012 - categorical_accuracy: 0.9071
22560/60000 [==========>...................] - ETA: 1:10 - loss: 0.3010 - categorical_accuracy: 0.9071
22592/60000 [==========>...................] - ETA: 1:10 - loss: 0.3013 - categorical_accuracy: 0.9071
22624/60000 [==========>...................] - ETA: 1:10 - loss: 0.3010 - categorical_accuracy: 0.9072
22656/60000 [==========>...................] - ETA: 1:10 - loss: 0.3007 - categorical_accuracy: 0.9073
22688/60000 [==========>...................] - ETA: 1:10 - loss: 0.3004 - categorical_accuracy: 0.9074
22720/60000 [==========>...................] - ETA: 1:10 - loss: 0.3001 - categorical_accuracy: 0.9075
22752/60000 [==========>...................] - ETA: 1:10 - loss: 0.3000 - categorical_accuracy: 0.9076
22784/60000 [==========>...................] - ETA: 1:10 - loss: 0.2997 - categorical_accuracy: 0.9077
22816/60000 [==========>...................] - ETA: 1:10 - loss: 0.2994 - categorical_accuracy: 0.9077
22848/60000 [==========>...................] - ETA: 1:10 - loss: 0.2990 - categorical_accuracy: 0.9078
22880/60000 [==========>...................] - ETA: 1:10 - loss: 0.2990 - categorical_accuracy: 0.9079
22912/60000 [==========>...................] - ETA: 1:09 - loss: 0.2986 - categorical_accuracy: 0.9080
22944/60000 [==========>...................] - ETA: 1:09 - loss: 0.2983 - categorical_accuracy: 0.9081
22976/60000 [==========>...................] - ETA: 1:09 - loss: 0.2980 - categorical_accuracy: 0.9082
23008/60000 [==========>...................] - ETA: 1:09 - loss: 0.2980 - categorical_accuracy: 0.9082
23040/60000 [==========>...................] - ETA: 1:09 - loss: 0.2980 - categorical_accuracy: 0.9081
23072/60000 [==========>...................] - ETA: 1:09 - loss: 0.2980 - categorical_accuracy: 0.9081
23104/60000 [==========>...................] - ETA: 1:09 - loss: 0.2979 - categorical_accuracy: 0.9081
23136/60000 [==========>...................] - ETA: 1:09 - loss: 0.2976 - categorical_accuracy: 0.9082
23168/60000 [==========>...................] - ETA: 1:09 - loss: 0.2972 - categorical_accuracy: 0.9083
23200/60000 [==========>...................] - ETA: 1:09 - loss: 0.2970 - categorical_accuracy: 0.9083
23232/60000 [==========>...................] - ETA: 1:09 - loss: 0.2970 - categorical_accuracy: 0.9083
23264/60000 [==========>...................] - ETA: 1:09 - loss: 0.2967 - categorical_accuracy: 0.9084
23296/60000 [==========>...................] - ETA: 1:09 - loss: 0.2966 - categorical_accuracy: 0.9084
23328/60000 [==========>...................] - ETA: 1:09 - loss: 0.2964 - categorical_accuracy: 0.9084
23360/60000 [==========>...................] - ETA: 1:09 - loss: 0.2962 - categorical_accuracy: 0.9085
23392/60000 [==========>...................] - ETA: 1:09 - loss: 0.2959 - categorical_accuracy: 0.9086
23424/60000 [==========>...................] - ETA: 1:09 - loss: 0.2958 - categorical_accuracy: 0.9086
23456/60000 [==========>...................] - ETA: 1:08 - loss: 0.2955 - categorical_accuracy: 0.9087
23488/60000 [==========>...................] - ETA: 1:08 - loss: 0.2952 - categorical_accuracy: 0.9088
23520/60000 [==========>...................] - ETA: 1:08 - loss: 0.2950 - categorical_accuracy: 0.9088
23552/60000 [==========>...................] - ETA: 1:08 - loss: 0.2949 - categorical_accuracy: 0.9088
23584/60000 [==========>...................] - ETA: 1:08 - loss: 0.2946 - categorical_accuracy: 0.9088
23616/60000 [==========>...................] - ETA: 1:08 - loss: 0.2944 - categorical_accuracy: 0.9089
23648/60000 [==========>...................] - ETA: 1:08 - loss: 0.2941 - categorical_accuracy: 0.9090
23680/60000 [==========>...................] - ETA: 1:08 - loss: 0.2938 - categorical_accuracy: 0.9091
23712/60000 [==========>...................] - ETA: 1:08 - loss: 0.2940 - categorical_accuracy: 0.9091
23744/60000 [==========>...................] - ETA: 1:08 - loss: 0.2937 - categorical_accuracy: 0.9092
23776/60000 [==========>...................] - ETA: 1:08 - loss: 0.2935 - categorical_accuracy: 0.9093
23808/60000 [==========>...................] - ETA: 1:08 - loss: 0.2933 - categorical_accuracy: 0.9094
23840/60000 [==========>...................] - ETA: 1:08 - loss: 0.2931 - categorical_accuracy: 0.9094
23872/60000 [==========>...................] - ETA: 1:08 - loss: 0.2928 - categorical_accuracy: 0.9095
23904/60000 [==========>...................] - ETA: 1:08 - loss: 0.2926 - categorical_accuracy: 0.9096
23936/60000 [==========>...................] - ETA: 1:08 - loss: 0.2923 - categorical_accuracy: 0.9097
23968/60000 [==========>...................] - ETA: 1:07 - loss: 0.2920 - categorical_accuracy: 0.9098
24000/60000 [===========>..................] - ETA: 1:07 - loss: 0.2916 - categorical_accuracy: 0.9099
24032/60000 [===========>..................] - ETA: 1:07 - loss: 0.2913 - categorical_accuracy: 0.9100
24064/60000 [===========>..................] - ETA: 1:07 - loss: 0.2911 - categorical_accuracy: 0.9101
24096/60000 [===========>..................] - ETA: 1:07 - loss: 0.2909 - categorical_accuracy: 0.9102
24128/60000 [===========>..................] - ETA: 1:07 - loss: 0.2907 - categorical_accuracy: 0.9102
24160/60000 [===========>..................] - ETA: 1:07 - loss: 0.2904 - categorical_accuracy: 0.9102
24192/60000 [===========>..................] - ETA: 1:07 - loss: 0.2902 - categorical_accuracy: 0.9103
24224/60000 [===========>..................] - ETA: 1:07 - loss: 0.2899 - categorical_accuracy: 0.9104
24256/60000 [===========>..................] - ETA: 1:07 - loss: 0.2897 - categorical_accuracy: 0.9104
24288/60000 [===========>..................] - ETA: 1:07 - loss: 0.2896 - categorical_accuracy: 0.9104
24320/60000 [===========>..................] - ETA: 1:07 - loss: 0.2893 - categorical_accuracy: 0.9106
24352/60000 [===========>..................] - ETA: 1:07 - loss: 0.2889 - categorical_accuracy: 0.9107
24384/60000 [===========>..................] - ETA: 1:07 - loss: 0.2888 - categorical_accuracy: 0.9107
24416/60000 [===========>..................] - ETA: 1:07 - loss: 0.2885 - categorical_accuracy: 0.9108
24448/60000 [===========>..................] - ETA: 1:07 - loss: 0.2886 - categorical_accuracy: 0.9109
24480/60000 [===========>..................] - ETA: 1:07 - loss: 0.2883 - categorical_accuracy: 0.9109
24512/60000 [===========>..................] - ETA: 1:06 - loss: 0.2880 - categorical_accuracy: 0.9111
24544/60000 [===========>..................] - ETA: 1:06 - loss: 0.2877 - categorical_accuracy: 0.9112
24576/60000 [===========>..................] - ETA: 1:06 - loss: 0.2874 - categorical_accuracy: 0.9113
24608/60000 [===========>..................] - ETA: 1:06 - loss: 0.2871 - categorical_accuracy: 0.9114
24640/60000 [===========>..................] - ETA: 1:06 - loss: 0.2870 - categorical_accuracy: 0.9114
24672/60000 [===========>..................] - ETA: 1:06 - loss: 0.2867 - categorical_accuracy: 0.9115
24704/60000 [===========>..................] - ETA: 1:06 - loss: 0.2864 - categorical_accuracy: 0.9116
24736/60000 [===========>..................] - ETA: 1:06 - loss: 0.2861 - categorical_accuracy: 0.9117
24768/60000 [===========>..................] - ETA: 1:06 - loss: 0.2860 - categorical_accuracy: 0.9117
24800/60000 [===========>..................] - ETA: 1:06 - loss: 0.2860 - categorical_accuracy: 0.9117
24832/60000 [===========>..................] - ETA: 1:06 - loss: 0.2858 - categorical_accuracy: 0.9118
24864/60000 [===========>..................] - ETA: 1:06 - loss: 0.2857 - categorical_accuracy: 0.9118
24896/60000 [===========>..................] - ETA: 1:06 - loss: 0.2855 - categorical_accuracy: 0.9118
24928/60000 [===========>..................] - ETA: 1:06 - loss: 0.2851 - categorical_accuracy: 0.9119
24960/60000 [===========>..................] - ETA: 1:06 - loss: 0.2848 - categorical_accuracy: 0.9121
24992/60000 [===========>..................] - ETA: 1:06 - loss: 0.2846 - categorical_accuracy: 0.9121
25024/60000 [===========>..................] - ETA: 1:06 - loss: 0.2843 - categorical_accuracy: 0.9122
25056/60000 [===========>..................] - ETA: 1:05 - loss: 0.2840 - categorical_accuracy: 0.9123
25088/60000 [===========>..................] - ETA: 1:05 - loss: 0.2837 - categorical_accuracy: 0.9123
25120/60000 [===========>..................] - ETA: 1:05 - loss: 0.2834 - categorical_accuracy: 0.9125
25152/60000 [===========>..................] - ETA: 1:05 - loss: 0.2831 - categorical_accuracy: 0.9126
25184/60000 [===========>..................] - ETA: 1:05 - loss: 0.2830 - categorical_accuracy: 0.9126
25216/60000 [===========>..................] - ETA: 1:05 - loss: 0.2827 - categorical_accuracy: 0.9128
25248/60000 [===========>..................] - ETA: 1:05 - loss: 0.2826 - categorical_accuracy: 0.9128
25280/60000 [===========>..................] - ETA: 1:05 - loss: 0.2823 - categorical_accuracy: 0.9129
25312/60000 [===========>..................] - ETA: 1:05 - loss: 0.2822 - categorical_accuracy: 0.9130
25344/60000 [===========>..................] - ETA: 1:05 - loss: 0.2819 - categorical_accuracy: 0.9131
25376/60000 [===========>..................] - ETA: 1:05 - loss: 0.2819 - categorical_accuracy: 0.9131
25408/60000 [===========>..................] - ETA: 1:05 - loss: 0.2816 - categorical_accuracy: 0.9133
25440/60000 [===========>..................] - ETA: 1:05 - loss: 0.2814 - categorical_accuracy: 0.9133
25472/60000 [===========>..................] - ETA: 1:05 - loss: 0.2811 - categorical_accuracy: 0.9133
25504/60000 [===========>..................] - ETA: 1:05 - loss: 0.2814 - categorical_accuracy: 0.9133
25536/60000 [===========>..................] - ETA: 1:05 - loss: 0.2811 - categorical_accuracy: 0.9135
25568/60000 [===========>..................] - ETA: 1:04 - loss: 0.2812 - categorical_accuracy: 0.9134
25600/60000 [===========>..................] - ETA: 1:04 - loss: 0.2811 - categorical_accuracy: 0.9134
25632/60000 [===========>..................] - ETA: 1:04 - loss: 0.2808 - categorical_accuracy: 0.9135
25664/60000 [===========>..................] - ETA: 1:04 - loss: 0.2806 - categorical_accuracy: 0.9137
25696/60000 [===========>..................] - ETA: 1:04 - loss: 0.2802 - categorical_accuracy: 0.9138
25728/60000 [===========>..................] - ETA: 1:04 - loss: 0.2799 - categorical_accuracy: 0.9138
25760/60000 [===========>..................] - ETA: 1:04 - loss: 0.2796 - categorical_accuracy: 0.9139
25792/60000 [===========>..................] - ETA: 1:04 - loss: 0.2793 - categorical_accuracy: 0.9140
25824/60000 [===========>..................] - ETA: 1:04 - loss: 0.2790 - categorical_accuracy: 0.9141
25856/60000 [===========>..................] - ETA: 1:04 - loss: 0.2787 - categorical_accuracy: 0.9142
25888/60000 [===========>..................] - ETA: 1:04 - loss: 0.2785 - categorical_accuracy: 0.9143
25920/60000 [===========>..................] - ETA: 1:04 - loss: 0.2784 - categorical_accuracy: 0.9144
25952/60000 [===========>..................] - ETA: 1:04 - loss: 0.2782 - categorical_accuracy: 0.9143
25984/60000 [===========>..................] - ETA: 1:04 - loss: 0.2780 - categorical_accuracy: 0.9144
26016/60000 [============>.................] - ETA: 1:04 - loss: 0.2776 - categorical_accuracy: 0.9145
26048/60000 [============>.................] - ETA: 1:04 - loss: 0.2774 - categorical_accuracy: 0.9146
26080/60000 [============>.................] - ETA: 1:03 - loss: 0.2772 - categorical_accuracy: 0.9146
26112/60000 [============>.................] - ETA: 1:03 - loss: 0.2773 - categorical_accuracy: 0.9147
26144/60000 [============>.................] - ETA: 1:03 - loss: 0.2770 - categorical_accuracy: 0.9147
26176/60000 [============>.................] - ETA: 1:03 - loss: 0.2770 - categorical_accuracy: 0.9147
26208/60000 [============>.................] - ETA: 1:03 - loss: 0.2767 - categorical_accuracy: 0.9148
26240/60000 [============>.................] - ETA: 1:03 - loss: 0.2765 - categorical_accuracy: 0.9148
26272/60000 [============>.................] - ETA: 1:03 - loss: 0.2763 - categorical_accuracy: 0.9149
26304/60000 [============>.................] - ETA: 1:03 - loss: 0.2760 - categorical_accuracy: 0.9150
26336/60000 [============>.................] - ETA: 1:03 - loss: 0.2757 - categorical_accuracy: 0.9151
26368/60000 [============>.................] - ETA: 1:03 - loss: 0.2754 - categorical_accuracy: 0.9152
26400/60000 [============>.................] - ETA: 1:03 - loss: 0.2752 - categorical_accuracy: 0.9152
26432/60000 [============>.................] - ETA: 1:03 - loss: 0.2752 - categorical_accuracy: 0.9152
26464/60000 [============>.................] - ETA: 1:03 - loss: 0.2749 - categorical_accuracy: 0.9153
26496/60000 [============>.................] - ETA: 1:03 - loss: 0.2746 - categorical_accuracy: 0.9153
26528/60000 [============>.................] - ETA: 1:03 - loss: 0.2744 - categorical_accuracy: 0.9154
26560/60000 [============>.................] - ETA: 1:03 - loss: 0.2742 - categorical_accuracy: 0.9155
26592/60000 [============>.................] - ETA: 1:03 - loss: 0.2743 - categorical_accuracy: 0.9155
26624/60000 [============>.................] - ETA: 1:02 - loss: 0.2741 - categorical_accuracy: 0.9156
26656/60000 [============>.................] - ETA: 1:02 - loss: 0.2738 - categorical_accuracy: 0.9157
26688/60000 [============>.................] - ETA: 1:02 - loss: 0.2736 - categorical_accuracy: 0.9158
26720/60000 [============>.................] - ETA: 1:02 - loss: 0.2733 - categorical_accuracy: 0.9158
26752/60000 [============>.................] - ETA: 1:02 - loss: 0.2730 - categorical_accuracy: 0.9159
26784/60000 [============>.................] - ETA: 1:02 - loss: 0.2729 - categorical_accuracy: 0.9160
26816/60000 [============>.................] - ETA: 1:02 - loss: 0.2726 - categorical_accuracy: 0.9161
26848/60000 [============>.................] - ETA: 1:02 - loss: 0.2724 - categorical_accuracy: 0.9161
26880/60000 [============>.................] - ETA: 1:02 - loss: 0.2725 - categorical_accuracy: 0.9161
26912/60000 [============>.................] - ETA: 1:02 - loss: 0.2726 - categorical_accuracy: 0.9161
26944/60000 [============>.................] - ETA: 1:02 - loss: 0.2723 - categorical_accuracy: 0.9162
26976/60000 [============>.................] - ETA: 1:02 - loss: 0.2724 - categorical_accuracy: 0.9162
27008/60000 [============>.................] - ETA: 1:02 - loss: 0.2721 - categorical_accuracy: 0.9163
27040/60000 [============>.................] - ETA: 1:02 - loss: 0.2719 - categorical_accuracy: 0.9163
27072/60000 [============>.................] - ETA: 1:02 - loss: 0.2720 - categorical_accuracy: 0.9163
27104/60000 [============>.................] - ETA: 1:02 - loss: 0.2719 - categorical_accuracy: 0.9164
27136/60000 [============>.................] - ETA: 1:01 - loss: 0.2721 - categorical_accuracy: 0.9163
27168/60000 [============>.................] - ETA: 1:01 - loss: 0.2719 - categorical_accuracy: 0.9164
27200/60000 [============>.................] - ETA: 1:01 - loss: 0.2718 - categorical_accuracy: 0.9164
27232/60000 [============>.................] - ETA: 1:01 - loss: 0.2718 - categorical_accuracy: 0.9164
27264/60000 [============>.................] - ETA: 1:01 - loss: 0.2716 - categorical_accuracy: 0.9165
27296/60000 [============>.................] - ETA: 1:01 - loss: 0.2716 - categorical_accuracy: 0.9165
27328/60000 [============>.................] - ETA: 1:01 - loss: 0.2713 - categorical_accuracy: 0.9166
27360/60000 [============>.................] - ETA: 1:01 - loss: 0.2713 - categorical_accuracy: 0.9166
27392/60000 [============>.................] - ETA: 1:01 - loss: 0.2710 - categorical_accuracy: 0.9167
27424/60000 [============>.................] - ETA: 1:01 - loss: 0.2709 - categorical_accuracy: 0.9167
27456/60000 [============>.................] - ETA: 1:01 - loss: 0.2706 - categorical_accuracy: 0.9168
27488/60000 [============>.................] - ETA: 1:01 - loss: 0.2705 - categorical_accuracy: 0.9168
27520/60000 [============>.................] - ETA: 1:01 - loss: 0.2704 - categorical_accuracy: 0.9168
27552/60000 [============>.................] - ETA: 1:01 - loss: 0.2702 - categorical_accuracy: 0.9169
27584/60000 [============>.................] - ETA: 1:01 - loss: 0.2700 - categorical_accuracy: 0.9169
27616/60000 [============>.................] - ETA: 1:01 - loss: 0.2697 - categorical_accuracy: 0.9170
27648/60000 [============>.................] - ETA: 1:00 - loss: 0.2695 - categorical_accuracy: 0.9171
27680/60000 [============>.................] - ETA: 1:00 - loss: 0.2694 - categorical_accuracy: 0.9172
27712/60000 [============>.................] - ETA: 1:00 - loss: 0.2693 - categorical_accuracy: 0.9171
27744/60000 [============>.................] - ETA: 1:00 - loss: 0.2691 - categorical_accuracy: 0.9172
27776/60000 [============>.................] - ETA: 1:00 - loss: 0.2691 - categorical_accuracy: 0.9173
27808/60000 [============>.................] - ETA: 1:00 - loss: 0.2690 - categorical_accuracy: 0.9173
27840/60000 [============>.................] - ETA: 1:00 - loss: 0.2691 - categorical_accuracy: 0.9172
27872/60000 [============>.................] - ETA: 1:00 - loss: 0.2690 - categorical_accuracy: 0.9173
27904/60000 [============>.................] - ETA: 1:00 - loss: 0.2687 - categorical_accuracy: 0.9174
27936/60000 [============>.................] - ETA: 1:00 - loss: 0.2686 - categorical_accuracy: 0.9174
27968/60000 [============>.................] - ETA: 1:00 - loss: 0.2686 - categorical_accuracy: 0.9174
28000/60000 [=============>................] - ETA: 1:00 - loss: 0.2686 - categorical_accuracy: 0.9173
28032/60000 [=============>................] - ETA: 1:00 - loss: 0.2683 - categorical_accuracy: 0.9174
28064/60000 [=============>................] - ETA: 1:00 - loss: 0.2681 - categorical_accuracy: 0.9175
28096/60000 [=============>................] - ETA: 1:00 - loss: 0.2679 - categorical_accuracy: 0.9175
28128/60000 [=============>................] - ETA: 1:00 - loss: 0.2678 - categorical_accuracy: 0.9176
28160/60000 [=============>................] - ETA: 1:00 - loss: 0.2677 - categorical_accuracy: 0.9176
28192/60000 [=============>................] - ETA: 59s - loss: 0.2676 - categorical_accuracy: 0.9176 
28224/60000 [=============>................] - ETA: 59s - loss: 0.2674 - categorical_accuracy: 0.9177
28256/60000 [=============>................] - ETA: 59s - loss: 0.2671 - categorical_accuracy: 0.9178
28288/60000 [=============>................] - ETA: 59s - loss: 0.2669 - categorical_accuracy: 0.9178
28320/60000 [=============>................] - ETA: 59s - loss: 0.2666 - categorical_accuracy: 0.9179
28352/60000 [=============>................] - ETA: 59s - loss: 0.2666 - categorical_accuracy: 0.9180
28384/60000 [=============>................] - ETA: 59s - loss: 0.2665 - categorical_accuracy: 0.9180
28416/60000 [=============>................] - ETA: 59s - loss: 0.2663 - categorical_accuracy: 0.9181
28448/60000 [=============>................] - ETA: 59s - loss: 0.2660 - categorical_accuracy: 0.9182
28480/60000 [=============>................] - ETA: 59s - loss: 0.2658 - categorical_accuracy: 0.9182
28512/60000 [=============>................] - ETA: 59s - loss: 0.2656 - categorical_accuracy: 0.9183
28544/60000 [=============>................] - ETA: 59s - loss: 0.2654 - categorical_accuracy: 0.9183
28576/60000 [=============>................] - ETA: 59s - loss: 0.2652 - categorical_accuracy: 0.9184
28608/60000 [=============>................] - ETA: 59s - loss: 0.2650 - categorical_accuracy: 0.9184
28640/60000 [=============>................] - ETA: 59s - loss: 0.2649 - categorical_accuracy: 0.9184
28672/60000 [=============>................] - ETA: 59s - loss: 0.2646 - categorical_accuracy: 0.9185
28704/60000 [=============>................] - ETA: 58s - loss: 0.2644 - categorical_accuracy: 0.9186
28736/60000 [=============>................] - ETA: 58s - loss: 0.2642 - categorical_accuracy: 0.9187
28768/60000 [=============>................] - ETA: 58s - loss: 0.2641 - categorical_accuracy: 0.9187
28800/60000 [=============>................] - ETA: 58s - loss: 0.2639 - categorical_accuracy: 0.9187
28832/60000 [=============>................] - ETA: 58s - loss: 0.2641 - categorical_accuracy: 0.9187
28864/60000 [=============>................] - ETA: 58s - loss: 0.2641 - categorical_accuracy: 0.9187
28896/60000 [=============>................] - ETA: 58s - loss: 0.2639 - categorical_accuracy: 0.9188
28928/60000 [=============>................] - ETA: 58s - loss: 0.2639 - categorical_accuracy: 0.9188
28960/60000 [=============>................] - ETA: 58s - loss: 0.2637 - categorical_accuracy: 0.9189
28992/60000 [=============>................] - ETA: 58s - loss: 0.2635 - categorical_accuracy: 0.9189
29024/60000 [=============>................] - ETA: 58s - loss: 0.2634 - categorical_accuracy: 0.9190
29056/60000 [=============>................] - ETA: 58s - loss: 0.2631 - categorical_accuracy: 0.9190
29088/60000 [=============>................] - ETA: 58s - loss: 0.2630 - categorical_accuracy: 0.9190
29120/60000 [=============>................] - ETA: 58s - loss: 0.2629 - categorical_accuracy: 0.9190
29152/60000 [=============>................] - ETA: 58s - loss: 0.2627 - categorical_accuracy: 0.9191
29184/60000 [=============>................] - ETA: 58s - loss: 0.2625 - categorical_accuracy: 0.9191
29216/60000 [=============>................] - ETA: 58s - loss: 0.2622 - categorical_accuracy: 0.9192
29248/60000 [=============>................] - ETA: 57s - loss: 0.2622 - categorical_accuracy: 0.9192
29280/60000 [=============>................] - ETA: 57s - loss: 0.2620 - categorical_accuracy: 0.9193
29312/60000 [=============>................] - ETA: 57s - loss: 0.2618 - categorical_accuracy: 0.9194
29344/60000 [=============>................] - ETA: 57s - loss: 0.2615 - categorical_accuracy: 0.9194
29376/60000 [=============>................] - ETA: 57s - loss: 0.2616 - categorical_accuracy: 0.9194
29408/60000 [=============>................] - ETA: 57s - loss: 0.2614 - categorical_accuracy: 0.9195
29440/60000 [=============>................] - ETA: 57s - loss: 0.2612 - categorical_accuracy: 0.9195
29472/60000 [=============>................] - ETA: 57s - loss: 0.2609 - categorical_accuracy: 0.9196
29504/60000 [=============>................] - ETA: 57s - loss: 0.2610 - categorical_accuracy: 0.9196
29536/60000 [=============>................] - ETA: 57s - loss: 0.2609 - categorical_accuracy: 0.9196
29568/60000 [=============>................] - ETA: 57s - loss: 0.2607 - categorical_accuracy: 0.9196
29600/60000 [=============>................] - ETA: 57s - loss: 0.2605 - categorical_accuracy: 0.9197
29632/60000 [=============>................] - ETA: 57s - loss: 0.2603 - categorical_accuracy: 0.9198
29664/60000 [=============>................] - ETA: 57s - loss: 0.2601 - categorical_accuracy: 0.9199
29696/60000 [=============>................] - ETA: 57s - loss: 0.2598 - categorical_accuracy: 0.9199
29728/60000 [=============>................] - ETA: 57s - loss: 0.2600 - categorical_accuracy: 0.9199
29760/60000 [=============>................] - ETA: 56s - loss: 0.2599 - categorical_accuracy: 0.9199
29792/60000 [=============>................] - ETA: 56s - loss: 0.2600 - categorical_accuracy: 0.9199
29824/60000 [=============>................] - ETA: 56s - loss: 0.2599 - categorical_accuracy: 0.9199
29856/60000 [=============>................] - ETA: 56s - loss: 0.2597 - categorical_accuracy: 0.9200
29888/60000 [=============>................] - ETA: 56s - loss: 0.2597 - categorical_accuracy: 0.9200
29920/60000 [=============>................] - ETA: 56s - loss: 0.2596 - categorical_accuracy: 0.9201
29952/60000 [=============>................] - ETA: 56s - loss: 0.2595 - categorical_accuracy: 0.9201
29984/60000 [=============>................] - ETA: 56s - loss: 0.2593 - categorical_accuracy: 0.9201
30016/60000 [==============>...............] - ETA: 56s - loss: 0.2591 - categorical_accuracy: 0.9202
30048/60000 [==============>...............] - ETA: 56s - loss: 0.2589 - categorical_accuracy: 0.9203
30080/60000 [==============>...............] - ETA: 56s - loss: 0.2589 - categorical_accuracy: 0.9202
30112/60000 [==============>...............] - ETA: 56s - loss: 0.2588 - categorical_accuracy: 0.9203
30144/60000 [==============>...............] - ETA: 56s - loss: 0.2585 - categorical_accuracy: 0.9204
30176/60000 [==============>...............] - ETA: 56s - loss: 0.2584 - categorical_accuracy: 0.9204
30208/60000 [==============>...............] - ETA: 56s - loss: 0.2582 - categorical_accuracy: 0.9205
30240/60000 [==============>...............] - ETA: 56s - loss: 0.2580 - categorical_accuracy: 0.9205
30272/60000 [==============>...............] - ETA: 56s - loss: 0.2578 - categorical_accuracy: 0.9206
30304/60000 [==============>...............] - ETA: 55s - loss: 0.2577 - categorical_accuracy: 0.9207
30336/60000 [==============>...............] - ETA: 55s - loss: 0.2575 - categorical_accuracy: 0.9207
30368/60000 [==============>...............] - ETA: 55s - loss: 0.2575 - categorical_accuracy: 0.9207
30400/60000 [==============>...............] - ETA: 55s - loss: 0.2573 - categorical_accuracy: 0.9208
30432/60000 [==============>...............] - ETA: 55s - loss: 0.2571 - categorical_accuracy: 0.9209
30464/60000 [==============>...............] - ETA: 55s - loss: 0.2571 - categorical_accuracy: 0.9209
30496/60000 [==============>...............] - ETA: 55s - loss: 0.2569 - categorical_accuracy: 0.9209
30528/60000 [==============>...............] - ETA: 55s - loss: 0.2569 - categorical_accuracy: 0.9209
30560/60000 [==============>...............] - ETA: 55s - loss: 0.2567 - categorical_accuracy: 0.9209
30592/60000 [==============>...............] - ETA: 55s - loss: 0.2566 - categorical_accuracy: 0.9209
30624/60000 [==============>...............] - ETA: 55s - loss: 0.2564 - categorical_accuracy: 0.9210
30656/60000 [==============>...............] - ETA: 55s - loss: 0.2561 - categorical_accuracy: 0.9211
30688/60000 [==============>...............] - ETA: 55s - loss: 0.2559 - categorical_accuracy: 0.9211
30720/60000 [==============>...............] - ETA: 55s - loss: 0.2558 - categorical_accuracy: 0.9212
30752/60000 [==============>...............] - ETA: 55s - loss: 0.2557 - categorical_accuracy: 0.9212
30784/60000 [==============>...............] - ETA: 55s - loss: 0.2556 - categorical_accuracy: 0.9213
30816/60000 [==============>...............] - ETA: 54s - loss: 0.2554 - categorical_accuracy: 0.9213
30848/60000 [==============>...............] - ETA: 54s - loss: 0.2555 - categorical_accuracy: 0.9214
30880/60000 [==============>...............] - ETA: 54s - loss: 0.2553 - categorical_accuracy: 0.9214
30912/60000 [==============>...............] - ETA: 54s - loss: 0.2550 - categorical_accuracy: 0.9215
30944/60000 [==============>...............] - ETA: 54s - loss: 0.2548 - categorical_accuracy: 0.9215
30976/60000 [==============>...............] - ETA: 54s - loss: 0.2546 - categorical_accuracy: 0.9216
31008/60000 [==============>...............] - ETA: 54s - loss: 0.2545 - categorical_accuracy: 0.9216
31040/60000 [==============>...............] - ETA: 54s - loss: 0.2543 - categorical_accuracy: 0.9216
31072/60000 [==============>...............] - ETA: 54s - loss: 0.2542 - categorical_accuracy: 0.9217
31104/60000 [==============>...............] - ETA: 54s - loss: 0.2540 - categorical_accuracy: 0.9217
31136/60000 [==============>...............] - ETA: 54s - loss: 0.2539 - categorical_accuracy: 0.9217
31168/60000 [==============>...............] - ETA: 54s - loss: 0.2539 - categorical_accuracy: 0.9217
31200/60000 [==============>...............] - ETA: 54s - loss: 0.2539 - categorical_accuracy: 0.9217
31232/60000 [==============>...............] - ETA: 54s - loss: 0.2537 - categorical_accuracy: 0.9217
31264/60000 [==============>...............] - ETA: 54s - loss: 0.2537 - categorical_accuracy: 0.9218
31296/60000 [==============>...............] - ETA: 54s - loss: 0.2535 - categorical_accuracy: 0.9218
31328/60000 [==============>...............] - ETA: 54s - loss: 0.2535 - categorical_accuracy: 0.9218
31360/60000 [==============>...............] - ETA: 53s - loss: 0.2533 - categorical_accuracy: 0.9218
31392/60000 [==============>...............] - ETA: 53s - loss: 0.2531 - categorical_accuracy: 0.9219
31424/60000 [==============>...............] - ETA: 53s - loss: 0.2529 - categorical_accuracy: 0.9219
31456/60000 [==============>...............] - ETA: 53s - loss: 0.2527 - categorical_accuracy: 0.9220
31488/60000 [==============>...............] - ETA: 53s - loss: 0.2525 - categorical_accuracy: 0.9221
31520/60000 [==============>...............] - ETA: 53s - loss: 0.2524 - categorical_accuracy: 0.9221
31552/60000 [==============>...............] - ETA: 53s - loss: 0.2522 - categorical_accuracy: 0.9222
31584/60000 [==============>...............] - ETA: 53s - loss: 0.2520 - categorical_accuracy: 0.9222
31616/60000 [==============>...............] - ETA: 53s - loss: 0.2520 - categorical_accuracy: 0.9222
31648/60000 [==============>...............] - ETA: 53s - loss: 0.2519 - categorical_accuracy: 0.9222
31680/60000 [==============>...............] - ETA: 53s - loss: 0.2517 - categorical_accuracy: 0.9223
31712/60000 [==============>...............] - ETA: 53s - loss: 0.2519 - categorical_accuracy: 0.9222
31744/60000 [==============>...............] - ETA: 53s - loss: 0.2517 - categorical_accuracy: 0.9223
31776/60000 [==============>...............] - ETA: 53s - loss: 0.2516 - categorical_accuracy: 0.9223
31808/60000 [==============>...............] - ETA: 53s - loss: 0.2514 - categorical_accuracy: 0.9224
31840/60000 [==============>...............] - ETA: 53s - loss: 0.2512 - categorical_accuracy: 0.9225
31872/60000 [==============>...............] - ETA: 52s - loss: 0.2510 - categorical_accuracy: 0.9225
31904/60000 [==============>...............] - ETA: 52s - loss: 0.2510 - categorical_accuracy: 0.9225
31936/60000 [==============>...............] - ETA: 52s - loss: 0.2511 - categorical_accuracy: 0.9225
31968/60000 [==============>...............] - ETA: 52s - loss: 0.2510 - categorical_accuracy: 0.9225
32000/60000 [===============>..............] - ETA: 52s - loss: 0.2508 - categorical_accuracy: 0.9226
32032/60000 [===============>..............] - ETA: 52s - loss: 0.2508 - categorical_accuracy: 0.9226
32064/60000 [===============>..............] - ETA: 52s - loss: 0.2507 - categorical_accuracy: 0.9227
32096/60000 [===============>..............] - ETA: 52s - loss: 0.2504 - categorical_accuracy: 0.9227
32128/60000 [===============>..............] - ETA: 52s - loss: 0.2503 - categorical_accuracy: 0.9228
32160/60000 [===============>..............] - ETA: 52s - loss: 0.2501 - categorical_accuracy: 0.9228
32192/60000 [===============>..............] - ETA: 52s - loss: 0.2499 - categorical_accuracy: 0.9229
32224/60000 [===============>..............] - ETA: 52s - loss: 0.2498 - categorical_accuracy: 0.9229
32256/60000 [===============>..............] - ETA: 52s - loss: 0.2497 - categorical_accuracy: 0.9230
32288/60000 [===============>..............] - ETA: 52s - loss: 0.2498 - categorical_accuracy: 0.9230
32320/60000 [===============>..............] - ETA: 52s - loss: 0.2499 - categorical_accuracy: 0.9230
32352/60000 [===============>..............] - ETA: 52s - loss: 0.2497 - categorical_accuracy: 0.9231
32384/60000 [===============>..............] - ETA: 52s - loss: 0.2496 - categorical_accuracy: 0.9231
32416/60000 [===============>..............] - ETA: 51s - loss: 0.2494 - categorical_accuracy: 0.9232
32448/60000 [===============>..............] - ETA: 51s - loss: 0.2493 - categorical_accuracy: 0.9232
32480/60000 [===============>..............] - ETA: 51s - loss: 0.2491 - categorical_accuracy: 0.9232
32512/60000 [===============>..............] - ETA: 51s - loss: 0.2490 - categorical_accuracy: 0.9233
32544/60000 [===============>..............] - ETA: 51s - loss: 0.2489 - categorical_accuracy: 0.9233
32576/60000 [===============>..............] - ETA: 51s - loss: 0.2487 - categorical_accuracy: 0.9233
32608/60000 [===============>..............] - ETA: 51s - loss: 0.2485 - categorical_accuracy: 0.9234
32640/60000 [===============>..............] - ETA: 51s - loss: 0.2483 - categorical_accuracy: 0.9235
32672/60000 [===============>..............] - ETA: 51s - loss: 0.2482 - categorical_accuracy: 0.9235
32704/60000 [===============>..............] - ETA: 51s - loss: 0.2480 - categorical_accuracy: 0.9236
32736/60000 [===============>..............] - ETA: 51s - loss: 0.2479 - categorical_accuracy: 0.9237
32768/60000 [===============>..............] - ETA: 51s - loss: 0.2477 - categorical_accuracy: 0.9237
32800/60000 [===============>..............] - ETA: 51s - loss: 0.2475 - categorical_accuracy: 0.9237
32832/60000 [===============>..............] - ETA: 51s - loss: 0.2474 - categorical_accuracy: 0.9238
32864/60000 [===============>..............] - ETA: 51s - loss: 0.2473 - categorical_accuracy: 0.9238
32896/60000 [===============>..............] - ETA: 51s - loss: 0.2472 - categorical_accuracy: 0.9238
32928/60000 [===============>..............] - ETA: 50s - loss: 0.2471 - categorical_accuracy: 0.9239
32960/60000 [===============>..............] - ETA: 50s - loss: 0.2470 - categorical_accuracy: 0.9239
32992/60000 [===============>..............] - ETA: 50s - loss: 0.2468 - categorical_accuracy: 0.9239
33024/60000 [===============>..............] - ETA: 50s - loss: 0.2467 - categorical_accuracy: 0.9239
33056/60000 [===============>..............] - ETA: 50s - loss: 0.2466 - categorical_accuracy: 0.9239
33088/60000 [===============>..............] - ETA: 50s - loss: 0.2467 - categorical_accuracy: 0.9239
33120/60000 [===============>..............] - ETA: 50s - loss: 0.2466 - categorical_accuracy: 0.9240
33152/60000 [===============>..............] - ETA: 50s - loss: 0.2464 - categorical_accuracy: 0.9240
33184/60000 [===============>..............] - ETA: 50s - loss: 0.2462 - categorical_accuracy: 0.9241
33216/60000 [===============>..............] - ETA: 50s - loss: 0.2462 - categorical_accuracy: 0.9241
33248/60000 [===============>..............] - ETA: 50s - loss: 0.2461 - categorical_accuracy: 0.9242
33280/60000 [===============>..............] - ETA: 50s - loss: 0.2461 - categorical_accuracy: 0.9242
33312/60000 [===============>..............] - ETA: 50s - loss: 0.2463 - categorical_accuracy: 0.9241
33344/60000 [===============>..............] - ETA: 50s - loss: 0.2461 - categorical_accuracy: 0.9242
33376/60000 [===============>..............] - ETA: 50s - loss: 0.2459 - categorical_accuracy: 0.9242
33408/60000 [===============>..............] - ETA: 50s - loss: 0.2458 - categorical_accuracy: 0.9243
33440/60000 [===============>..............] - ETA: 50s - loss: 0.2456 - categorical_accuracy: 0.9243
33472/60000 [===============>..............] - ETA: 49s - loss: 0.2454 - categorical_accuracy: 0.9244
33504/60000 [===============>..............] - ETA: 49s - loss: 0.2452 - categorical_accuracy: 0.9244
33536/60000 [===============>..............] - ETA: 49s - loss: 0.2453 - categorical_accuracy: 0.9244
33568/60000 [===============>..............] - ETA: 49s - loss: 0.2451 - categorical_accuracy: 0.9245
33600/60000 [===============>..............] - ETA: 49s - loss: 0.2450 - categorical_accuracy: 0.9245
33632/60000 [===============>..............] - ETA: 49s - loss: 0.2449 - categorical_accuracy: 0.9245
33664/60000 [===============>..............] - ETA: 49s - loss: 0.2447 - categorical_accuracy: 0.9246
33696/60000 [===============>..............] - ETA: 49s - loss: 0.2445 - categorical_accuracy: 0.9246
33728/60000 [===============>..............] - ETA: 49s - loss: 0.2444 - categorical_accuracy: 0.9247
33760/60000 [===============>..............] - ETA: 49s - loss: 0.2442 - categorical_accuracy: 0.9247
33792/60000 [===============>..............] - ETA: 49s - loss: 0.2441 - categorical_accuracy: 0.9247
33824/60000 [===============>..............] - ETA: 49s - loss: 0.2443 - categorical_accuracy: 0.9247
33856/60000 [===============>..............] - ETA: 49s - loss: 0.2441 - categorical_accuracy: 0.9247
33888/60000 [===============>..............] - ETA: 49s - loss: 0.2440 - categorical_accuracy: 0.9248
33920/60000 [===============>..............] - ETA: 49s - loss: 0.2438 - categorical_accuracy: 0.9248
33952/60000 [===============>..............] - ETA: 49s - loss: 0.2437 - categorical_accuracy: 0.9249
33984/60000 [===============>..............] - ETA: 48s - loss: 0.2440 - categorical_accuracy: 0.9248
34016/60000 [================>.............] - ETA: 48s - loss: 0.2438 - categorical_accuracy: 0.9249
34048/60000 [================>.............] - ETA: 48s - loss: 0.2437 - categorical_accuracy: 0.9249
34080/60000 [================>.............] - ETA: 48s - loss: 0.2435 - categorical_accuracy: 0.9250
34112/60000 [================>.............] - ETA: 48s - loss: 0.2433 - categorical_accuracy: 0.9250
34144/60000 [================>.............] - ETA: 48s - loss: 0.2433 - categorical_accuracy: 0.9251
34176/60000 [================>.............] - ETA: 48s - loss: 0.2431 - categorical_accuracy: 0.9251
34208/60000 [================>.............] - ETA: 48s - loss: 0.2429 - categorical_accuracy: 0.9252
34240/60000 [================>.............] - ETA: 48s - loss: 0.2428 - categorical_accuracy: 0.9253
34272/60000 [================>.............] - ETA: 48s - loss: 0.2426 - categorical_accuracy: 0.9253
34304/60000 [================>.............] - ETA: 48s - loss: 0.2425 - categorical_accuracy: 0.9253
34336/60000 [================>.............] - ETA: 48s - loss: 0.2424 - categorical_accuracy: 0.9254
34368/60000 [================>.............] - ETA: 48s - loss: 0.2423 - categorical_accuracy: 0.9254
34400/60000 [================>.............] - ETA: 48s - loss: 0.2423 - categorical_accuracy: 0.9254
34432/60000 [================>.............] - ETA: 48s - loss: 0.2421 - categorical_accuracy: 0.9255
34464/60000 [================>.............] - ETA: 48s - loss: 0.2419 - categorical_accuracy: 0.9255
34496/60000 [================>.............] - ETA: 48s - loss: 0.2418 - categorical_accuracy: 0.9255
34528/60000 [================>.............] - ETA: 47s - loss: 0.2417 - categorical_accuracy: 0.9255
34560/60000 [================>.............] - ETA: 47s - loss: 0.2417 - categorical_accuracy: 0.9255
34592/60000 [================>.............] - ETA: 47s - loss: 0.2415 - categorical_accuracy: 0.9256
34624/60000 [================>.............] - ETA: 47s - loss: 0.2413 - categorical_accuracy: 0.9256
34656/60000 [================>.............] - ETA: 47s - loss: 0.2413 - categorical_accuracy: 0.9256
34688/60000 [================>.............] - ETA: 47s - loss: 0.2411 - categorical_accuracy: 0.9257
34720/60000 [================>.............] - ETA: 47s - loss: 0.2410 - categorical_accuracy: 0.9257
34752/60000 [================>.............] - ETA: 47s - loss: 0.2408 - categorical_accuracy: 0.9258
34784/60000 [================>.............] - ETA: 47s - loss: 0.2406 - categorical_accuracy: 0.9258
34816/60000 [================>.............] - ETA: 47s - loss: 0.2405 - categorical_accuracy: 0.9258
34848/60000 [================>.............] - ETA: 47s - loss: 0.2403 - categorical_accuracy: 0.9259
34880/60000 [================>.............] - ETA: 47s - loss: 0.2401 - categorical_accuracy: 0.9259
34912/60000 [================>.............] - ETA: 47s - loss: 0.2400 - categorical_accuracy: 0.9260
34944/60000 [================>.............] - ETA: 47s - loss: 0.2398 - categorical_accuracy: 0.9261
34976/60000 [================>.............] - ETA: 47s - loss: 0.2396 - categorical_accuracy: 0.9261
35008/60000 [================>.............] - ETA: 47s - loss: 0.2396 - categorical_accuracy: 0.9262
35040/60000 [================>.............] - ETA: 46s - loss: 0.2394 - categorical_accuracy: 0.9262
35072/60000 [================>.............] - ETA: 46s - loss: 0.2392 - categorical_accuracy: 0.9263
35104/60000 [================>.............] - ETA: 46s - loss: 0.2390 - categorical_accuracy: 0.9263
35136/60000 [================>.............] - ETA: 46s - loss: 0.2389 - categorical_accuracy: 0.9263
35168/60000 [================>.............] - ETA: 46s - loss: 0.2388 - categorical_accuracy: 0.9264
35200/60000 [================>.............] - ETA: 46s - loss: 0.2386 - categorical_accuracy: 0.9264
35232/60000 [================>.............] - ETA: 46s - loss: 0.2386 - categorical_accuracy: 0.9264
35264/60000 [================>.............] - ETA: 46s - loss: 0.2385 - categorical_accuracy: 0.9264
35296/60000 [================>.............] - ETA: 46s - loss: 0.2383 - categorical_accuracy: 0.9264
35328/60000 [================>.............] - ETA: 46s - loss: 0.2384 - categorical_accuracy: 0.9264
35360/60000 [================>.............] - ETA: 46s - loss: 0.2382 - categorical_accuracy: 0.9265
35392/60000 [================>.............] - ETA: 46s - loss: 0.2380 - categorical_accuracy: 0.9265
35424/60000 [================>.............] - ETA: 46s - loss: 0.2378 - categorical_accuracy: 0.9266
35456/60000 [================>.............] - ETA: 46s - loss: 0.2376 - categorical_accuracy: 0.9267
35488/60000 [================>.............] - ETA: 46s - loss: 0.2375 - categorical_accuracy: 0.9267
35520/60000 [================>.............] - ETA: 46s - loss: 0.2375 - categorical_accuracy: 0.9267
35552/60000 [================>.............] - ETA: 46s - loss: 0.2373 - categorical_accuracy: 0.9267
35584/60000 [================>.............] - ETA: 45s - loss: 0.2372 - categorical_accuracy: 0.9268
35616/60000 [================>.............] - ETA: 45s - loss: 0.2370 - categorical_accuracy: 0.9268
35648/60000 [================>.............] - ETA: 45s - loss: 0.2369 - categorical_accuracy: 0.9268
35680/60000 [================>.............] - ETA: 45s - loss: 0.2369 - categorical_accuracy: 0.9268
35712/60000 [================>.............] - ETA: 45s - loss: 0.2368 - categorical_accuracy: 0.9269
35744/60000 [================>.............] - ETA: 45s - loss: 0.2366 - categorical_accuracy: 0.9270
35776/60000 [================>.............] - ETA: 45s - loss: 0.2364 - categorical_accuracy: 0.9270
35808/60000 [================>.............] - ETA: 45s - loss: 0.2363 - categorical_accuracy: 0.9271
35840/60000 [================>.............] - ETA: 45s - loss: 0.2362 - categorical_accuracy: 0.9271
35872/60000 [================>.............] - ETA: 45s - loss: 0.2360 - categorical_accuracy: 0.9271
35904/60000 [================>.............] - ETA: 45s - loss: 0.2358 - categorical_accuracy: 0.9272
35936/60000 [================>.............] - ETA: 45s - loss: 0.2357 - categorical_accuracy: 0.9272
35968/60000 [================>.............] - ETA: 45s - loss: 0.2359 - categorical_accuracy: 0.9272
36000/60000 [=================>............] - ETA: 45s - loss: 0.2357 - categorical_accuracy: 0.9272
36032/60000 [=================>............] - ETA: 45s - loss: 0.2356 - categorical_accuracy: 0.9273
36064/60000 [=================>............] - ETA: 45s - loss: 0.2356 - categorical_accuracy: 0.9273
36096/60000 [=================>............] - ETA: 45s - loss: 0.2354 - categorical_accuracy: 0.9273
36128/60000 [=================>............] - ETA: 44s - loss: 0.2352 - categorical_accuracy: 0.9274
36160/60000 [=================>............] - ETA: 44s - loss: 0.2351 - categorical_accuracy: 0.9274
36192/60000 [=================>............] - ETA: 44s - loss: 0.2349 - categorical_accuracy: 0.9275
36224/60000 [=================>............] - ETA: 44s - loss: 0.2348 - categorical_accuracy: 0.9276
36256/60000 [=================>............] - ETA: 44s - loss: 0.2347 - categorical_accuracy: 0.9276
36288/60000 [=================>............] - ETA: 44s - loss: 0.2346 - categorical_accuracy: 0.9276
36320/60000 [=================>............] - ETA: 44s - loss: 0.2345 - categorical_accuracy: 0.9276
36352/60000 [=================>............] - ETA: 44s - loss: 0.2346 - categorical_accuracy: 0.9277
36384/60000 [=================>............] - ETA: 44s - loss: 0.2345 - categorical_accuracy: 0.9277
36416/60000 [=================>............] - ETA: 44s - loss: 0.2344 - categorical_accuracy: 0.9278
36448/60000 [=================>............] - ETA: 44s - loss: 0.2343 - categorical_accuracy: 0.9278
36480/60000 [=================>............] - ETA: 44s - loss: 0.2342 - categorical_accuracy: 0.9278
36512/60000 [=================>............] - ETA: 44s - loss: 0.2340 - categorical_accuracy: 0.9278
36544/60000 [=================>............] - ETA: 44s - loss: 0.2340 - categorical_accuracy: 0.9279
36576/60000 [=================>............] - ETA: 44s - loss: 0.2340 - categorical_accuracy: 0.9279
36608/60000 [=================>............] - ETA: 44s - loss: 0.2338 - categorical_accuracy: 0.9279
36640/60000 [=================>............] - ETA: 43s - loss: 0.2336 - categorical_accuracy: 0.9280
36672/60000 [=================>............] - ETA: 43s - loss: 0.2335 - categorical_accuracy: 0.9280
36704/60000 [=================>............] - ETA: 43s - loss: 0.2333 - categorical_accuracy: 0.9281
36736/60000 [=================>............] - ETA: 43s - loss: 0.2331 - categorical_accuracy: 0.9281
36768/60000 [=================>............] - ETA: 43s - loss: 0.2331 - categorical_accuracy: 0.9281
36800/60000 [=================>............] - ETA: 43s - loss: 0.2330 - categorical_accuracy: 0.9282
36832/60000 [=================>............] - ETA: 43s - loss: 0.2329 - categorical_accuracy: 0.9282
36864/60000 [=================>............] - ETA: 43s - loss: 0.2327 - categorical_accuracy: 0.9282
36896/60000 [=================>............] - ETA: 43s - loss: 0.2326 - categorical_accuracy: 0.9283
36928/60000 [=================>............] - ETA: 43s - loss: 0.2324 - categorical_accuracy: 0.9283
36960/60000 [=================>............] - ETA: 43s - loss: 0.2323 - categorical_accuracy: 0.9284
36992/60000 [=================>............] - ETA: 43s - loss: 0.2322 - categorical_accuracy: 0.9284
37024/60000 [=================>............] - ETA: 43s - loss: 0.2320 - categorical_accuracy: 0.9284
37056/60000 [=================>............] - ETA: 43s - loss: 0.2320 - categorical_accuracy: 0.9284
37088/60000 [=================>............] - ETA: 43s - loss: 0.2318 - categorical_accuracy: 0.9284
37120/60000 [=================>............] - ETA: 43s - loss: 0.2317 - categorical_accuracy: 0.9285
37152/60000 [=================>............] - ETA: 43s - loss: 0.2316 - categorical_accuracy: 0.9285
37184/60000 [=================>............] - ETA: 42s - loss: 0.2314 - categorical_accuracy: 0.9285
37216/60000 [=================>............] - ETA: 42s - loss: 0.2313 - categorical_accuracy: 0.9286
37248/60000 [=================>............] - ETA: 42s - loss: 0.2314 - categorical_accuracy: 0.9286
37280/60000 [=================>............] - ETA: 42s - loss: 0.2313 - categorical_accuracy: 0.9286
37312/60000 [=================>............] - ETA: 42s - loss: 0.2312 - categorical_accuracy: 0.9287
37344/60000 [=================>............] - ETA: 42s - loss: 0.2310 - categorical_accuracy: 0.9287
37376/60000 [=================>............] - ETA: 42s - loss: 0.2308 - categorical_accuracy: 0.9288
37408/60000 [=================>............] - ETA: 42s - loss: 0.2310 - categorical_accuracy: 0.9288
37440/60000 [=================>............] - ETA: 42s - loss: 0.2308 - categorical_accuracy: 0.9288
37472/60000 [=================>............] - ETA: 42s - loss: 0.2307 - categorical_accuracy: 0.9289
37504/60000 [=================>............] - ETA: 42s - loss: 0.2305 - categorical_accuracy: 0.9289
37536/60000 [=================>............] - ETA: 42s - loss: 0.2303 - categorical_accuracy: 0.9290
37568/60000 [=================>............] - ETA: 42s - loss: 0.2303 - categorical_accuracy: 0.9290
37600/60000 [=================>............] - ETA: 42s - loss: 0.2302 - categorical_accuracy: 0.9290
37632/60000 [=================>............] - ETA: 42s - loss: 0.2300 - categorical_accuracy: 0.9291
37664/60000 [=================>............] - ETA: 42s - loss: 0.2298 - categorical_accuracy: 0.9291
37696/60000 [=================>............] - ETA: 41s - loss: 0.2297 - categorical_accuracy: 0.9292
37728/60000 [=================>............] - ETA: 41s - loss: 0.2295 - categorical_accuracy: 0.9292
37760/60000 [=================>............] - ETA: 41s - loss: 0.2294 - categorical_accuracy: 0.9293
37792/60000 [=================>............] - ETA: 41s - loss: 0.2292 - categorical_accuracy: 0.9293
37824/60000 [=================>............] - ETA: 41s - loss: 0.2291 - categorical_accuracy: 0.9293
37856/60000 [=================>............] - ETA: 41s - loss: 0.2290 - categorical_accuracy: 0.9293
37888/60000 [=================>............] - ETA: 41s - loss: 0.2288 - categorical_accuracy: 0.9294
37920/60000 [=================>............] - ETA: 41s - loss: 0.2287 - categorical_accuracy: 0.9295
37952/60000 [=================>............] - ETA: 41s - loss: 0.2285 - categorical_accuracy: 0.9295
37984/60000 [=================>............] - ETA: 41s - loss: 0.2284 - categorical_accuracy: 0.9295
38016/60000 [==================>...........] - ETA: 41s - loss: 0.2286 - categorical_accuracy: 0.9296
38048/60000 [==================>...........] - ETA: 41s - loss: 0.2284 - categorical_accuracy: 0.9296
38080/60000 [==================>...........] - ETA: 41s - loss: 0.2283 - categorical_accuracy: 0.9296
38112/60000 [==================>...........] - ETA: 41s - loss: 0.2281 - categorical_accuracy: 0.9297
38144/60000 [==================>...........] - ETA: 41s - loss: 0.2280 - categorical_accuracy: 0.9297
38176/60000 [==================>...........] - ETA: 41s - loss: 0.2279 - categorical_accuracy: 0.9297
38208/60000 [==================>...........] - ETA: 41s - loss: 0.2277 - categorical_accuracy: 0.9298
38240/60000 [==================>...........] - ETA: 40s - loss: 0.2276 - categorical_accuracy: 0.9298
38272/60000 [==================>...........] - ETA: 40s - loss: 0.2274 - categorical_accuracy: 0.9299
38304/60000 [==================>...........] - ETA: 40s - loss: 0.2273 - categorical_accuracy: 0.9299
38336/60000 [==================>...........] - ETA: 40s - loss: 0.2272 - categorical_accuracy: 0.9300
38368/60000 [==================>...........] - ETA: 40s - loss: 0.2271 - categorical_accuracy: 0.9300
38400/60000 [==================>...........] - ETA: 40s - loss: 0.2269 - categorical_accuracy: 0.9300
38432/60000 [==================>...........] - ETA: 40s - loss: 0.2267 - categorical_accuracy: 0.9301
38464/60000 [==================>...........] - ETA: 40s - loss: 0.2267 - categorical_accuracy: 0.9301
38496/60000 [==================>...........] - ETA: 40s - loss: 0.2266 - categorical_accuracy: 0.9301
38528/60000 [==================>...........] - ETA: 40s - loss: 0.2265 - categorical_accuracy: 0.9302
38560/60000 [==================>...........] - ETA: 40s - loss: 0.2264 - categorical_accuracy: 0.9302
38592/60000 [==================>...........] - ETA: 40s - loss: 0.2262 - categorical_accuracy: 0.9302
38624/60000 [==================>...........] - ETA: 40s - loss: 0.2260 - categorical_accuracy: 0.9303
38656/60000 [==================>...........] - ETA: 40s - loss: 0.2259 - categorical_accuracy: 0.9304
38688/60000 [==================>...........] - ETA: 40s - loss: 0.2259 - categorical_accuracy: 0.9304
38720/60000 [==================>...........] - ETA: 40s - loss: 0.2259 - categorical_accuracy: 0.9304
38752/60000 [==================>...........] - ETA: 39s - loss: 0.2261 - categorical_accuracy: 0.9304
38784/60000 [==================>...........] - ETA: 39s - loss: 0.2259 - categorical_accuracy: 0.9304
38816/60000 [==================>...........] - ETA: 39s - loss: 0.2258 - categorical_accuracy: 0.9305
38848/60000 [==================>...........] - ETA: 39s - loss: 0.2256 - categorical_accuracy: 0.9305
38880/60000 [==================>...........] - ETA: 39s - loss: 0.2255 - categorical_accuracy: 0.9306
38912/60000 [==================>...........] - ETA: 39s - loss: 0.2253 - categorical_accuracy: 0.9306
38944/60000 [==================>...........] - ETA: 39s - loss: 0.2252 - categorical_accuracy: 0.9307
38976/60000 [==================>...........] - ETA: 39s - loss: 0.2250 - categorical_accuracy: 0.9307
39008/60000 [==================>...........] - ETA: 39s - loss: 0.2251 - categorical_accuracy: 0.9307
39040/60000 [==================>...........] - ETA: 39s - loss: 0.2250 - categorical_accuracy: 0.9308
39072/60000 [==================>...........] - ETA: 39s - loss: 0.2248 - categorical_accuracy: 0.9308
39104/60000 [==================>...........] - ETA: 39s - loss: 0.2247 - categorical_accuracy: 0.9309
39136/60000 [==================>...........] - ETA: 39s - loss: 0.2247 - categorical_accuracy: 0.9309
39168/60000 [==================>...........] - ETA: 39s - loss: 0.2246 - categorical_accuracy: 0.9309
39200/60000 [==================>...........] - ETA: 39s - loss: 0.2244 - categorical_accuracy: 0.9309
39232/60000 [==================>...........] - ETA: 39s - loss: 0.2243 - categorical_accuracy: 0.9309
39264/60000 [==================>...........] - ETA: 39s - loss: 0.2242 - categorical_accuracy: 0.9310
39296/60000 [==================>...........] - ETA: 38s - loss: 0.2242 - categorical_accuracy: 0.9310
39328/60000 [==================>...........] - ETA: 38s - loss: 0.2242 - categorical_accuracy: 0.9310
39360/60000 [==================>...........] - ETA: 38s - loss: 0.2241 - categorical_accuracy: 0.9310
39392/60000 [==================>...........] - ETA: 38s - loss: 0.2239 - categorical_accuracy: 0.9311
39424/60000 [==================>...........] - ETA: 38s - loss: 0.2238 - categorical_accuracy: 0.9311
39456/60000 [==================>...........] - ETA: 38s - loss: 0.2237 - categorical_accuracy: 0.9311
39488/60000 [==================>...........] - ETA: 38s - loss: 0.2236 - categorical_accuracy: 0.9312
39520/60000 [==================>...........] - ETA: 38s - loss: 0.2234 - categorical_accuracy: 0.9312
39552/60000 [==================>...........] - ETA: 38s - loss: 0.2234 - categorical_accuracy: 0.9313
39584/60000 [==================>...........] - ETA: 38s - loss: 0.2232 - categorical_accuracy: 0.9313
39616/60000 [==================>...........] - ETA: 38s - loss: 0.2231 - categorical_accuracy: 0.9314
39648/60000 [==================>...........] - ETA: 38s - loss: 0.2229 - categorical_accuracy: 0.9314
39680/60000 [==================>...........] - ETA: 38s - loss: 0.2227 - categorical_accuracy: 0.9315
39712/60000 [==================>...........] - ETA: 38s - loss: 0.2227 - categorical_accuracy: 0.9315
39744/60000 [==================>...........] - ETA: 38s - loss: 0.2227 - categorical_accuracy: 0.9315
39776/60000 [==================>...........] - ETA: 38s - loss: 0.2225 - categorical_accuracy: 0.9315
39808/60000 [==================>...........] - ETA: 37s - loss: 0.2226 - categorical_accuracy: 0.9315
39840/60000 [==================>...........] - ETA: 37s - loss: 0.2225 - categorical_accuracy: 0.9315
39872/60000 [==================>...........] - ETA: 37s - loss: 0.2224 - categorical_accuracy: 0.9316
39904/60000 [==================>...........] - ETA: 37s - loss: 0.2223 - categorical_accuracy: 0.9316
39936/60000 [==================>...........] - ETA: 37s - loss: 0.2222 - categorical_accuracy: 0.9316
39968/60000 [==================>...........] - ETA: 37s - loss: 0.2223 - categorical_accuracy: 0.9316
40000/60000 [===================>..........] - ETA: 37s - loss: 0.2221 - categorical_accuracy: 0.9316
40032/60000 [===================>..........] - ETA: 37s - loss: 0.2221 - categorical_accuracy: 0.9316
40064/60000 [===================>..........] - ETA: 37s - loss: 0.2220 - categorical_accuracy: 0.9317
40096/60000 [===================>..........] - ETA: 37s - loss: 0.2219 - categorical_accuracy: 0.9317
40128/60000 [===================>..........] - ETA: 37s - loss: 0.2218 - categorical_accuracy: 0.9317
40160/60000 [===================>..........] - ETA: 37s - loss: 0.2217 - categorical_accuracy: 0.9318
40192/60000 [===================>..........] - ETA: 37s - loss: 0.2216 - categorical_accuracy: 0.9318
40224/60000 [===================>..........] - ETA: 37s - loss: 0.2214 - categorical_accuracy: 0.9319
40256/60000 [===================>..........] - ETA: 37s - loss: 0.2213 - categorical_accuracy: 0.9319
40288/60000 [===================>..........] - ETA: 37s - loss: 0.2213 - categorical_accuracy: 0.9319
40320/60000 [===================>..........] - ETA: 37s - loss: 0.2213 - categorical_accuracy: 0.9319
40352/60000 [===================>..........] - ETA: 36s - loss: 0.2211 - categorical_accuracy: 0.9320
40384/60000 [===================>..........] - ETA: 36s - loss: 0.2211 - categorical_accuracy: 0.9320
40416/60000 [===================>..........] - ETA: 36s - loss: 0.2211 - categorical_accuracy: 0.9320
40448/60000 [===================>..........] - ETA: 36s - loss: 0.2210 - categorical_accuracy: 0.9320
40480/60000 [===================>..........] - ETA: 36s - loss: 0.2208 - categorical_accuracy: 0.9321
40512/60000 [===================>..........] - ETA: 36s - loss: 0.2207 - categorical_accuracy: 0.9321
40544/60000 [===================>..........] - ETA: 36s - loss: 0.2207 - categorical_accuracy: 0.9320
40576/60000 [===================>..........] - ETA: 36s - loss: 0.2207 - categorical_accuracy: 0.9320
40608/60000 [===================>..........] - ETA: 36s - loss: 0.2205 - categorical_accuracy: 0.9321
40640/60000 [===================>..........] - ETA: 36s - loss: 0.2204 - categorical_accuracy: 0.9321
40672/60000 [===================>..........] - ETA: 36s - loss: 0.2202 - categorical_accuracy: 0.9322
40704/60000 [===================>..........] - ETA: 36s - loss: 0.2203 - categorical_accuracy: 0.9322
40736/60000 [===================>..........] - ETA: 36s - loss: 0.2201 - categorical_accuracy: 0.9322
40768/60000 [===================>..........] - ETA: 36s - loss: 0.2200 - categorical_accuracy: 0.9323
40800/60000 [===================>..........] - ETA: 36s - loss: 0.2199 - categorical_accuracy: 0.9323
40832/60000 [===================>..........] - ETA: 36s - loss: 0.2197 - categorical_accuracy: 0.9324
40864/60000 [===================>..........] - ETA: 35s - loss: 0.2196 - categorical_accuracy: 0.9324
40896/60000 [===================>..........] - ETA: 35s - loss: 0.2194 - categorical_accuracy: 0.9325
40928/60000 [===================>..........] - ETA: 35s - loss: 0.2194 - categorical_accuracy: 0.9325
40960/60000 [===================>..........] - ETA: 35s - loss: 0.2193 - categorical_accuracy: 0.9325
40992/60000 [===================>..........] - ETA: 35s - loss: 0.2194 - categorical_accuracy: 0.9325
41024/60000 [===================>..........] - ETA: 35s - loss: 0.2193 - categorical_accuracy: 0.9325
41056/60000 [===================>..........] - ETA: 35s - loss: 0.2192 - categorical_accuracy: 0.9325
41088/60000 [===================>..........] - ETA: 35s - loss: 0.2191 - categorical_accuracy: 0.9326
41120/60000 [===================>..........] - ETA: 35s - loss: 0.2192 - categorical_accuracy: 0.9326
41152/60000 [===================>..........] - ETA: 35s - loss: 0.2190 - categorical_accuracy: 0.9326
41184/60000 [===================>..........] - ETA: 35s - loss: 0.2190 - categorical_accuracy: 0.9326
41216/60000 [===================>..........] - ETA: 35s - loss: 0.2190 - categorical_accuracy: 0.9326
41248/60000 [===================>..........] - ETA: 35s - loss: 0.2189 - categorical_accuracy: 0.9327
41280/60000 [===================>..........] - ETA: 35s - loss: 0.2189 - categorical_accuracy: 0.9327
41312/60000 [===================>..........] - ETA: 35s - loss: 0.2189 - categorical_accuracy: 0.9327
41344/60000 [===================>..........] - ETA: 35s - loss: 0.2188 - categorical_accuracy: 0.9327
41376/60000 [===================>..........] - ETA: 35s - loss: 0.2187 - categorical_accuracy: 0.9327
41408/60000 [===================>..........] - ETA: 34s - loss: 0.2186 - categorical_accuracy: 0.9327
41440/60000 [===================>..........] - ETA: 34s - loss: 0.2185 - categorical_accuracy: 0.9328
41472/60000 [===================>..........] - ETA: 34s - loss: 0.2183 - categorical_accuracy: 0.9328
41504/60000 [===================>..........] - ETA: 34s - loss: 0.2182 - categorical_accuracy: 0.9329
41536/60000 [===================>..........] - ETA: 34s - loss: 0.2182 - categorical_accuracy: 0.9329
41568/60000 [===================>..........] - ETA: 34s - loss: 0.2181 - categorical_accuracy: 0.9329
41600/60000 [===================>..........] - ETA: 34s - loss: 0.2179 - categorical_accuracy: 0.9330
41632/60000 [===================>..........] - ETA: 34s - loss: 0.2179 - categorical_accuracy: 0.9330
41664/60000 [===================>..........] - ETA: 34s - loss: 0.2178 - categorical_accuracy: 0.9330
41696/60000 [===================>..........] - ETA: 34s - loss: 0.2178 - categorical_accuracy: 0.9330
41728/60000 [===================>..........] - ETA: 34s - loss: 0.2177 - categorical_accuracy: 0.9331
41760/60000 [===================>..........] - ETA: 34s - loss: 0.2175 - categorical_accuracy: 0.9331
41792/60000 [===================>..........] - ETA: 34s - loss: 0.2176 - categorical_accuracy: 0.9332
41824/60000 [===================>..........] - ETA: 34s - loss: 0.2174 - categorical_accuracy: 0.9332
41856/60000 [===================>..........] - ETA: 34s - loss: 0.2173 - categorical_accuracy: 0.9333
41888/60000 [===================>..........] - ETA: 34s - loss: 0.2171 - categorical_accuracy: 0.9333
41920/60000 [===================>..........] - ETA: 33s - loss: 0.2170 - categorical_accuracy: 0.9333
41952/60000 [===================>..........] - ETA: 33s - loss: 0.2169 - categorical_accuracy: 0.9334
41984/60000 [===================>..........] - ETA: 33s - loss: 0.2168 - categorical_accuracy: 0.9334
42016/60000 [====================>.........] - ETA: 33s - loss: 0.2168 - categorical_accuracy: 0.9335
42048/60000 [====================>.........] - ETA: 33s - loss: 0.2167 - categorical_accuracy: 0.9335
42080/60000 [====================>.........] - ETA: 33s - loss: 0.2166 - categorical_accuracy: 0.9335
42112/60000 [====================>.........] - ETA: 33s - loss: 0.2165 - categorical_accuracy: 0.9335
42144/60000 [====================>.........] - ETA: 33s - loss: 0.2164 - categorical_accuracy: 0.9335
42176/60000 [====================>.........] - ETA: 33s - loss: 0.2163 - categorical_accuracy: 0.9336
42208/60000 [====================>.........] - ETA: 33s - loss: 0.2162 - categorical_accuracy: 0.9336
42240/60000 [====================>.........] - ETA: 33s - loss: 0.2161 - categorical_accuracy: 0.9336
42272/60000 [====================>.........] - ETA: 33s - loss: 0.2159 - categorical_accuracy: 0.9337
42304/60000 [====================>.........] - ETA: 33s - loss: 0.2158 - categorical_accuracy: 0.9337
42336/60000 [====================>.........] - ETA: 33s - loss: 0.2158 - categorical_accuracy: 0.9337
42368/60000 [====================>.........] - ETA: 33s - loss: 0.2157 - categorical_accuracy: 0.9338
42400/60000 [====================>.........] - ETA: 33s - loss: 0.2156 - categorical_accuracy: 0.9338
42432/60000 [====================>.........] - ETA: 33s - loss: 0.2155 - categorical_accuracy: 0.9339
42464/60000 [====================>.........] - ETA: 32s - loss: 0.2154 - categorical_accuracy: 0.9339
42496/60000 [====================>.........] - ETA: 32s - loss: 0.2153 - categorical_accuracy: 0.9339
42528/60000 [====================>.........] - ETA: 32s - loss: 0.2151 - categorical_accuracy: 0.9340
42560/60000 [====================>.........] - ETA: 32s - loss: 0.2150 - categorical_accuracy: 0.9340
42592/60000 [====================>.........] - ETA: 32s - loss: 0.2149 - categorical_accuracy: 0.9340
42624/60000 [====================>.........] - ETA: 32s - loss: 0.2148 - categorical_accuracy: 0.9341
42656/60000 [====================>.........] - ETA: 32s - loss: 0.2146 - categorical_accuracy: 0.9341
42688/60000 [====================>.........] - ETA: 32s - loss: 0.2146 - categorical_accuracy: 0.9341
42720/60000 [====================>.........] - ETA: 32s - loss: 0.2145 - categorical_accuracy: 0.9342
42752/60000 [====================>.........] - ETA: 32s - loss: 0.2144 - categorical_accuracy: 0.9342
42784/60000 [====================>.........] - ETA: 32s - loss: 0.2144 - categorical_accuracy: 0.9342
42816/60000 [====================>.........] - ETA: 32s - loss: 0.2142 - categorical_accuracy: 0.9342
42848/60000 [====================>.........] - ETA: 32s - loss: 0.2143 - categorical_accuracy: 0.9342
42880/60000 [====================>.........] - ETA: 32s - loss: 0.2142 - categorical_accuracy: 0.9342
42912/60000 [====================>.........] - ETA: 32s - loss: 0.2142 - categorical_accuracy: 0.9342
42944/60000 [====================>.........] - ETA: 32s - loss: 0.2141 - categorical_accuracy: 0.9342
42976/60000 [====================>.........] - ETA: 32s - loss: 0.2141 - categorical_accuracy: 0.9342
43008/60000 [====================>.........] - ETA: 31s - loss: 0.2143 - categorical_accuracy: 0.9342
43040/60000 [====================>.........] - ETA: 31s - loss: 0.2142 - categorical_accuracy: 0.9342
43072/60000 [====================>.........] - ETA: 31s - loss: 0.2141 - categorical_accuracy: 0.9342
43104/60000 [====================>.........] - ETA: 31s - loss: 0.2139 - categorical_accuracy: 0.9343
43136/60000 [====================>.........] - ETA: 31s - loss: 0.2138 - categorical_accuracy: 0.9343
43168/60000 [====================>.........] - ETA: 31s - loss: 0.2137 - categorical_accuracy: 0.9343
43200/60000 [====================>.........] - ETA: 31s - loss: 0.2136 - categorical_accuracy: 0.9344
43232/60000 [====================>.........] - ETA: 31s - loss: 0.2135 - categorical_accuracy: 0.9344
43264/60000 [====================>.........] - ETA: 31s - loss: 0.2135 - categorical_accuracy: 0.9344
43296/60000 [====================>.........] - ETA: 31s - loss: 0.2134 - categorical_accuracy: 0.9345
43328/60000 [====================>.........] - ETA: 31s - loss: 0.2132 - categorical_accuracy: 0.9345
43360/60000 [====================>.........] - ETA: 31s - loss: 0.2131 - categorical_accuracy: 0.9345
43392/60000 [====================>.........] - ETA: 31s - loss: 0.2130 - categorical_accuracy: 0.9346
43424/60000 [====================>.........] - ETA: 31s - loss: 0.2129 - categorical_accuracy: 0.9346
43456/60000 [====================>.........] - ETA: 31s - loss: 0.2128 - categorical_accuracy: 0.9346
43488/60000 [====================>.........] - ETA: 31s - loss: 0.2127 - categorical_accuracy: 0.9346
43520/60000 [====================>.........] - ETA: 30s - loss: 0.2126 - categorical_accuracy: 0.9346
43552/60000 [====================>.........] - ETA: 30s - loss: 0.2125 - categorical_accuracy: 0.9347
43584/60000 [====================>.........] - ETA: 30s - loss: 0.2125 - categorical_accuracy: 0.9347
43616/60000 [====================>.........] - ETA: 30s - loss: 0.2124 - categorical_accuracy: 0.9347
43648/60000 [====================>.........] - ETA: 30s - loss: 0.2123 - categorical_accuracy: 0.9348
43680/60000 [====================>.........] - ETA: 30s - loss: 0.2122 - categorical_accuracy: 0.9348
43712/60000 [====================>.........] - ETA: 30s - loss: 0.2121 - categorical_accuracy: 0.9348
43744/60000 [====================>.........] - ETA: 30s - loss: 0.2121 - categorical_accuracy: 0.9348
43776/60000 [====================>.........] - ETA: 30s - loss: 0.2121 - categorical_accuracy: 0.9348
43808/60000 [====================>.........] - ETA: 30s - loss: 0.2121 - categorical_accuracy: 0.9348
43840/60000 [====================>.........] - ETA: 30s - loss: 0.2121 - categorical_accuracy: 0.9348
43872/60000 [====================>.........] - ETA: 30s - loss: 0.2120 - categorical_accuracy: 0.9348
43904/60000 [====================>.........] - ETA: 30s - loss: 0.2119 - categorical_accuracy: 0.9349
43936/60000 [====================>.........] - ETA: 30s - loss: 0.2117 - categorical_accuracy: 0.9349
43968/60000 [====================>.........] - ETA: 30s - loss: 0.2116 - categorical_accuracy: 0.9350
44000/60000 [=====================>........] - ETA: 30s - loss: 0.2115 - categorical_accuracy: 0.9350
44032/60000 [=====================>........] - ETA: 30s - loss: 0.2114 - categorical_accuracy: 0.9350
44064/60000 [=====================>........] - ETA: 29s - loss: 0.2112 - categorical_accuracy: 0.9350
44096/60000 [=====================>........] - ETA: 29s - loss: 0.2111 - categorical_accuracy: 0.9351
44128/60000 [=====================>........] - ETA: 29s - loss: 0.2111 - categorical_accuracy: 0.9351
44160/60000 [=====================>........] - ETA: 29s - loss: 0.2109 - categorical_accuracy: 0.9352
44192/60000 [=====================>........] - ETA: 29s - loss: 0.2108 - categorical_accuracy: 0.9352
44224/60000 [=====================>........] - ETA: 29s - loss: 0.2107 - categorical_accuracy: 0.9352
44256/60000 [=====================>........] - ETA: 29s - loss: 0.2106 - categorical_accuracy: 0.9353
44288/60000 [=====================>........] - ETA: 29s - loss: 0.2106 - categorical_accuracy: 0.9353
44320/60000 [=====================>........] - ETA: 29s - loss: 0.2105 - categorical_accuracy: 0.9353
44352/60000 [=====================>........] - ETA: 29s - loss: 0.2104 - categorical_accuracy: 0.9354
44384/60000 [=====================>........] - ETA: 29s - loss: 0.2102 - categorical_accuracy: 0.9354
44416/60000 [=====================>........] - ETA: 29s - loss: 0.2101 - categorical_accuracy: 0.9354
44448/60000 [=====================>........] - ETA: 29s - loss: 0.2100 - categorical_accuracy: 0.9355
44480/60000 [=====================>........] - ETA: 29s - loss: 0.2099 - categorical_accuracy: 0.9355
44512/60000 [=====================>........] - ETA: 29s - loss: 0.2097 - categorical_accuracy: 0.9355
44544/60000 [=====================>........] - ETA: 29s - loss: 0.2097 - categorical_accuracy: 0.9356
44576/60000 [=====================>........] - ETA: 28s - loss: 0.2096 - categorical_accuracy: 0.9356
44608/60000 [=====================>........] - ETA: 28s - loss: 0.2094 - categorical_accuracy: 0.9357
44640/60000 [=====================>........] - ETA: 28s - loss: 0.2093 - categorical_accuracy: 0.9357
44672/60000 [=====================>........] - ETA: 28s - loss: 0.2092 - categorical_accuracy: 0.9358
44704/60000 [=====================>........] - ETA: 28s - loss: 0.2091 - categorical_accuracy: 0.9358
44736/60000 [=====================>........] - ETA: 28s - loss: 0.2089 - categorical_accuracy: 0.9358
44768/60000 [=====================>........] - ETA: 28s - loss: 0.2090 - categorical_accuracy: 0.9358
44800/60000 [=====================>........] - ETA: 28s - loss: 0.2089 - categorical_accuracy: 0.9359
44832/60000 [=====================>........] - ETA: 28s - loss: 0.2087 - categorical_accuracy: 0.9359
44864/60000 [=====================>........] - ETA: 28s - loss: 0.2086 - categorical_accuracy: 0.9359
44896/60000 [=====================>........] - ETA: 28s - loss: 0.2085 - categorical_accuracy: 0.9360
44928/60000 [=====================>........] - ETA: 28s - loss: 0.2084 - categorical_accuracy: 0.9360
44960/60000 [=====================>........] - ETA: 28s - loss: 0.2083 - categorical_accuracy: 0.9360
44992/60000 [=====================>........] - ETA: 28s - loss: 0.2083 - categorical_accuracy: 0.9360
45024/60000 [=====================>........] - ETA: 28s - loss: 0.2082 - categorical_accuracy: 0.9360
45056/60000 [=====================>........] - ETA: 28s - loss: 0.2082 - categorical_accuracy: 0.9360
45088/60000 [=====================>........] - ETA: 28s - loss: 0.2084 - categorical_accuracy: 0.9360
45120/60000 [=====================>........] - ETA: 27s - loss: 0.2082 - categorical_accuracy: 0.9361
45152/60000 [=====================>........] - ETA: 27s - loss: 0.2082 - categorical_accuracy: 0.9361
45184/60000 [=====================>........] - ETA: 27s - loss: 0.2081 - categorical_accuracy: 0.9361
45216/60000 [=====================>........] - ETA: 27s - loss: 0.2080 - categorical_accuracy: 0.9362
45248/60000 [=====================>........] - ETA: 27s - loss: 0.2080 - categorical_accuracy: 0.9362
45280/60000 [=====================>........] - ETA: 27s - loss: 0.2079 - categorical_accuracy: 0.9362
45312/60000 [=====================>........] - ETA: 27s - loss: 0.2079 - categorical_accuracy: 0.9362
45344/60000 [=====================>........] - ETA: 27s - loss: 0.2078 - categorical_accuracy: 0.9362
45376/60000 [=====================>........] - ETA: 27s - loss: 0.2076 - categorical_accuracy: 0.9363
45408/60000 [=====================>........] - ETA: 27s - loss: 0.2076 - categorical_accuracy: 0.9363
45440/60000 [=====================>........] - ETA: 27s - loss: 0.2075 - categorical_accuracy: 0.9363
45472/60000 [=====================>........] - ETA: 27s - loss: 0.2074 - categorical_accuracy: 0.9363
45504/60000 [=====================>........] - ETA: 27s - loss: 0.2073 - categorical_accuracy: 0.9363
45536/60000 [=====================>........] - ETA: 27s - loss: 0.2072 - categorical_accuracy: 0.9364
45568/60000 [=====================>........] - ETA: 27s - loss: 0.2071 - categorical_accuracy: 0.9364
45600/60000 [=====================>........] - ETA: 27s - loss: 0.2070 - categorical_accuracy: 0.9364
45632/60000 [=====================>........] - ETA: 27s - loss: 0.2070 - categorical_accuracy: 0.9364
45664/60000 [=====================>........] - ETA: 26s - loss: 0.2069 - categorical_accuracy: 0.9364
45696/60000 [=====================>........] - ETA: 26s - loss: 0.2068 - categorical_accuracy: 0.9364
45728/60000 [=====================>........] - ETA: 26s - loss: 0.2068 - categorical_accuracy: 0.9365
45760/60000 [=====================>........] - ETA: 26s - loss: 0.2066 - categorical_accuracy: 0.9365
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2066 - categorical_accuracy: 0.9365
45824/60000 [=====================>........] - ETA: 26s - loss: 0.2065 - categorical_accuracy: 0.9365
45856/60000 [=====================>........] - ETA: 26s - loss: 0.2064 - categorical_accuracy: 0.9365
45888/60000 [=====================>........] - ETA: 26s - loss: 0.2063 - categorical_accuracy: 0.9366
45920/60000 [=====================>........] - ETA: 26s - loss: 0.2062 - categorical_accuracy: 0.9366
45952/60000 [=====================>........] - ETA: 26s - loss: 0.2062 - categorical_accuracy: 0.9366
45984/60000 [=====================>........] - ETA: 26s - loss: 0.2061 - categorical_accuracy: 0.9366
46016/60000 [======================>.......] - ETA: 26s - loss: 0.2060 - categorical_accuracy: 0.9367
46048/60000 [======================>.......] - ETA: 26s - loss: 0.2062 - categorical_accuracy: 0.9367
46080/60000 [======================>.......] - ETA: 26s - loss: 0.2061 - categorical_accuracy: 0.9367
46112/60000 [======================>.......] - ETA: 26s - loss: 0.2060 - categorical_accuracy: 0.9367
46144/60000 [======================>.......] - ETA: 26s - loss: 0.2059 - categorical_accuracy: 0.9367
46176/60000 [======================>.......] - ETA: 25s - loss: 0.2059 - categorical_accuracy: 0.9367
46208/60000 [======================>.......] - ETA: 25s - loss: 0.2059 - categorical_accuracy: 0.9367
46240/60000 [======================>.......] - ETA: 25s - loss: 0.2058 - categorical_accuracy: 0.9368
46272/60000 [======================>.......] - ETA: 25s - loss: 0.2057 - categorical_accuracy: 0.9368
46304/60000 [======================>.......] - ETA: 25s - loss: 0.2057 - categorical_accuracy: 0.9368
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2056 - categorical_accuracy: 0.9368
46368/60000 [======================>.......] - ETA: 25s - loss: 0.2055 - categorical_accuracy: 0.9369
46400/60000 [======================>.......] - ETA: 25s - loss: 0.2054 - categorical_accuracy: 0.9369
46432/60000 [======================>.......] - ETA: 25s - loss: 0.2052 - categorical_accuracy: 0.9369
46464/60000 [======================>.......] - ETA: 25s - loss: 0.2051 - categorical_accuracy: 0.9370
46496/60000 [======================>.......] - ETA: 25s - loss: 0.2052 - categorical_accuracy: 0.9370
46528/60000 [======================>.......] - ETA: 25s - loss: 0.2051 - categorical_accuracy: 0.9370
46560/60000 [======================>.......] - ETA: 25s - loss: 0.2050 - categorical_accuracy: 0.9370
46592/60000 [======================>.......] - ETA: 25s - loss: 0.2049 - categorical_accuracy: 0.9371
46624/60000 [======================>.......] - ETA: 25s - loss: 0.2049 - categorical_accuracy: 0.9371
46656/60000 [======================>.......] - ETA: 25s - loss: 0.2048 - categorical_accuracy: 0.9371
46688/60000 [======================>.......] - ETA: 25s - loss: 0.2047 - categorical_accuracy: 0.9371
46720/60000 [======================>.......] - ETA: 24s - loss: 0.2046 - categorical_accuracy: 0.9372
46752/60000 [======================>.......] - ETA: 24s - loss: 0.2047 - categorical_accuracy: 0.9372
46784/60000 [======================>.......] - ETA: 24s - loss: 0.2047 - categorical_accuracy: 0.9372
46816/60000 [======================>.......] - ETA: 24s - loss: 0.2046 - categorical_accuracy: 0.9372
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2045 - categorical_accuracy: 0.9372
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2045 - categorical_accuracy: 0.9373
46912/60000 [======================>.......] - ETA: 24s - loss: 0.2044 - categorical_accuracy: 0.9373
46944/60000 [======================>.......] - ETA: 24s - loss: 0.2043 - categorical_accuracy: 0.9373
46976/60000 [======================>.......] - ETA: 24s - loss: 0.2042 - categorical_accuracy: 0.9374
47008/60000 [======================>.......] - ETA: 24s - loss: 0.2041 - categorical_accuracy: 0.9374
47040/60000 [======================>.......] - ETA: 24s - loss: 0.2040 - categorical_accuracy: 0.9374
47072/60000 [======================>.......] - ETA: 24s - loss: 0.2039 - categorical_accuracy: 0.9375
47104/60000 [======================>.......] - ETA: 24s - loss: 0.2038 - categorical_accuracy: 0.9375
47136/60000 [======================>.......] - ETA: 24s - loss: 0.2037 - categorical_accuracy: 0.9375
47168/60000 [======================>.......] - ETA: 24s - loss: 0.2035 - categorical_accuracy: 0.9376
47200/60000 [======================>.......] - ETA: 24s - loss: 0.2034 - categorical_accuracy: 0.9376
47232/60000 [======================>.......] - ETA: 23s - loss: 0.2034 - categorical_accuracy: 0.9376
47264/60000 [======================>.......] - ETA: 23s - loss: 0.2032 - categorical_accuracy: 0.9377
47296/60000 [======================>.......] - ETA: 23s - loss: 0.2032 - categorical_accuracy: 0.9377
47328/60000 [======================>.......] - ETA: 23s - loss: 0.2031 - categorical_accuracy: 0.9377
47360/60000 [======================>.......] - ETA: 23s - loss: 0.2033 - categorical_accuracy: 0.9377
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2032 - categorical_accuracy: 0.9377
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2030 - categorical_accuracy: 0.9378
47456/60000 [======================>.......] - ETA: 23s - loss: 0.2029 - categorical_accuracy: 0.9378
47488/60000 [======================>.......] - ETA: 23s - loss: 0.2028 - categorical_accuracy: 0.9379
47520/60000 [======================>.......] - ETA: 23s - loss: 0.2027 - categorical_accuracy: 0.9379
47552/60000 [======================>.......] - ETA: 23s - loss: 0.2026 - categorical_accuracy: 0.9379
47584/60000 [======================>.......] - ETA: 23s - loss: 0.2025 - categorical_accuracy: 0.9380
47616/60000 [======================>.......] - ETA: 23s - loss: 0.2025 - categorical_accuracy: 0.9380
47648/60000 [======================>.......] - ETA: 23s - loss: 0.2024 - categorical_accuracy: 0.9380
47680/60000 [======================>.......] - ETA: 23s - loss: 0.2023 - categorical_accuracy: 0.9380
47712/60000 [======================>.......] - ETA: 23s - loss: 0.2023 - categorical_accuracy: 0.9380
47744/60000 [======================>.......] - ETA: 23s - loss: 0.2021 - categorical_accuracy: 0.9380
47776/60000 [======================>.......] - ETA: 22s - loss: 0.2021 - categorical_accuracy: 0.9381
47808/60000 [======================>.......] - ETA: 22s - loss: 0.2020 - categorical_accuracy: 0.9381
47840/60000 [======================>.......] - ETA: 22s - loss: 0.2018 - categorical_accuracy: 0.9381
47872/60000 [======================>.......] - ETA: 22s - loss: 0.2018 - categorical_accuracy: 0.9381
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2017 - categorical_accuracy: 0.9381
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2018 - categorical_accuracy: 0.9381
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2016 - categorical_accuracy: 0.9381
48000/60000 [=======================>......] - ETA: 22s - loss: 0.2016 - categorical_accuracy: 0.9382
48032/60000 [=======================>......] - ETA: 22s - loss: 0.2014 - categorical_accuracy: 0.9382
48064/60000 [=======================>......] - ETA: 22s - loss: 0.2014 - categorical_accuracy: 0.9382
48096/60000 [=======================>......] - ETA: 22s - loss: 0.2013 - categorical_accuracy: 0.9383
48128/60000 [=======================>......] - ETA: 22s - loss: 0.2012 - categorical_accuracy: 0.9383
48160/60000 [=======================>......] - ETA: 22s - loss: 0.2011 - categorical_accuracy: 0.9383
48192/60000 [=======================>......] - ETA: 22s - loss: 0.2009 - categorical_accuracy: 0.9384
48224/60000 [=======================>......] - ETA: 22s - loss: 0.2008 - categorical_accuracy: 0.9384
48256/60000 [=======================>......] - ETA: 22s - loss: 0.2009 - categorical_accuracy: 0.9384
48288/60000 [=======================>......] - ETA: 22s - loss: 0.2010 - categorical_accuracy: 0.9383
48320/60000 [=======================>......] - ETA: 21s - loss: 0.2009 - categorical_accuracy: 0.9384
48352/60000 [=======================>......] - ETA: 21s - loss: 0.2007 - categorical_accuracy: 0.9384
48384/60000 [=======================>......] - ETA: 21s - loss: 0.2008 - categorical_accuracy: 0.9384
48416/60000 [=======================>......] - ETA: 21s - loss: 0.2007 - categorical_accuracy: 0.9385
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2006 - categorical_accuracy: 0.9385
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2007 - categorical_accuracy: 0.9385
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2005 - categorical_accuracy: 0.9386
48544/60000 [=======================>......] - ETA: 21s - loss: 0.2005 - categorical_accuracy: 0.9386
48576/60000 [=======================>......] - ETA: 21s - loss: 0.2004 - categorical_accuracy: 0.9386
48608/60000 [=======================>......] - ETA: 21s - loss: 0.2003 - categorical_accuracy: 0.9386
48640/60000 [=======================>......] - ETA: 21s - loss: 0.2002 - categorical_accuracy: 0.9387
48672/60000 [=======================>......] - ETA: 21s - loss: 0.2001 - categorical_accuracy: 0.9387
48704/60000 [=======================>......] - ETA: 21s - loss: 0.2000 - categorical_accuracy: 0.9387
48736/60000 [=======================>......] - ETA: 21s - loss: 0.1999 - categorical_accuracy: 0.9388
48768/60000 [=======================>......] - ETA: 21s - loss: 0.1998 - categorical_accuracy: 0.9388
48800/60000 [=======================>......] - ETA: 21s - loss: 0.1997 - categorical_accuracy: 0.9388
48832/60000 [=======================>......] - ETA: 20s - loss: 0.1996 - categorical_accuracy: 0.9389
48864/60000 [=======================>......] - ETA: 20s - loss: 0.1995 - categorical_accuracy: 0.9389
48896/60000 [=======================>......] - ETA: 20s - loss: 0.1994 - categorical_accuracy: 0.9390
48928/60000 [=======================>......] - ETA: 20s - loss: 0.1992 - categorical_accuracy: 0.9390
48960/60000 [=======================>......] - ETA: 20s - loss: 0.1993 - categorical_accuracy: 0.9390
48992/60000 [=======================>......] - ETA: 20s - loss: 0.1992 - categorical_accuracy: 0.9390
49024/60000 [=======================>......] - ETA: 20s - loss: 0.1992 - categorical_accuracy: 0.9391
49056/60000 [=======================>......] - ETA: 20s - loss: 0.1991 - categorical_accuracy: 0.9391
49088/60000 [=======================>......] - ETA: 20s - loss: 0.1990 - categorical_accuracy: 0.9391
49120/60000 [=======================>......] - ETA: 20s - loss: 0.1989 - categorical_accuracy: 0.9391
49152/60000 [=======================>......] - ETA: 20s - loss: 0.1988 - categorical_accuracy: 0.9392
49184/60000 [=======================>......] - ETA: 20s - loss: 0.1987 - categorical_accuracy: 0.9392
49216/60000 [=======================>......] - ETA: 20s - loss: 0.1987 - categorical_accuracy: 0.9392
49248/60000 [=======================>......] - ETA: 20s - loss: 0.1986 - categorical_accuracy: 0.9392
49280/60000 [=======================>......] - ETA: 20s - loss: 0.1985 - categorical_accuracy: 0.9393
49312/60000 [=======================>......] - ETA: 20s - loss: 0.1984 - categorical_accuracy: 0.9393
49344/60000 [=======================>......] - ETA: 20s - loss: 0.1984 - categorical_accuracy: 0.9393
49376/60000 [=======================>......] - ETA: 19s - loss: 0.1984 - categorical_accuracy: 0.9393
49408/60000 [=======================>......] - ETA: 19s - loss: 0.1983 - categorical_accuracy: 0.9393
49440/60000 [=======================>......] - ETA: 19s - loss: 0.1982 - categorical_accuracy: 0.9394
49472/60000 [=======================>......] - ETA: 19s - loss: 0.1981 - categorical_accuracy: 0.9394
49504/60000 [=======================>......] - ETA: 19s - loss: 0.1980 - categorical_accuracy: 0.9394
49536/60000 [=======================>......] - ETA: 19s - loss: 0.1982 - categorical_accuracy: 0.9394
49568/60000 [=======================>......] - ETA: 19s - loss: 0.1982 - categorical_accuracy: 0.9394
49600/60000 [=======================>......] - ETA: 19s - loss: 0.1981 - categorical_accuracy: 0.9394
49632/60000 [=======================>......] - ETA: 19s - loss: 0.1980 - categorical_accuracy: 0.9394
49664/60000 [=======================>......] - ETA: 19s - loss: 0.1980 - categorical_accuracy: 0.9395
49696/60000 [=======================>......] - ETA: 19s - loss: 0.1980 - categorical_accuracy: 0.9394
49728/60000 [=======================>......] - ETA: 19s - loss: 0.1980 - categorical_accuracy: 0.9395
49760/60000 [=======================>......] - ETA: 19s - loss: 0.1979 - categorical_accuracy: 0.9395
49792/60000 [=======================>......] - ETA: 19s - loss: 0.1978 - categorical_accuracy: 0.9395
49824/60000 [=======================>......] - ETA: 19s - loss: 0.1978 - categorical_accuracy: 0.9395
49856/60000 [=======================>......] - ETA: 19s - loss: 0.1978 - categorical_accuracy: 0.9395
49888/60000 [=======================>......] - ETA: 19s - loss: 0.1977 - categorical_accuracy: 0.9395
49920/60000 [=======================>......] - ETA: 18s - loss: 0.1976 - categorical_accuracy: 0.9396
49952/60000 [=======================>......] - ETA: 18s - loss: 0.1975 - categorical_accuracy: 0.9396
49984/60000 [=======================>......] - ETA: 18s - loss: 0.1974 - categorical_accuracy: 0.9396
50016/60000 [========================>.....] - ETA: 18s - loss: 0.1973 - categorical_accuracy: 0.9396
50048/60000 [========================>.....] - ETA: 18s - loss: 0.1973 - categorical_accuracy: 0.9396
50080/60000 [========================>.....] - ETA: 18s - loss: 0.1972 - categorical_accuracy: 0.9397
50112/60000 [========================>.....] - ETA: 18s - loss: 0.1971 - categorical_accuracy: 0.9397
50144/60000 [========================>.....] - ETA: 18s - loss: 0.1970 - categorical_accuracy: 0.9397
50176/60000 [========================>.....] - ETA: 18s - loss: 0.1969 - categorical_accuracy: 0.9397
50208/60000 [========================>.....] - ETA: 18s - loss: 0.1968 - categorical_accuracy: 0.9398
50240/60000 [========================>.....] - ETA: 18s - loss: 0.1968 - categorical_accuracy: 0.9398
50272/60000 [========================>.....] - ETA: 18s - loss: 0.1966 - categorical_accuracy: 0.9398
50304/60000 [========================>.....] - ETA: 18s - loss: 0.1966 - categorical_accuracy: 0.9398
50336/60000 [========================>.....] - ETA: 18s - loss: 0.1967 - categorical_accuracy: 0.9398
50368/60000 [========================>.....] - ETA: 18s - loss: 0.1967 - categorical_accuracy: 0.9398
50400/60000 [========================>.....] - ETA: 18s - loss: 0.1966 - categorical_accuracy: 0.9398
50432/60000 [========================>.....] - ETA: 17s - loss: 0.1965 - categorical_accuracy: 0.9399
50464/60000 [========================>.....] - ETA: 17s - loss: 0.1965 - categorical_accuracy: 0.9399
50496/60000 [========================>.....] - ETA: 17s - loss: 0.1964 - categorical_accuracy: 0.9399
50528/60000 [========================>.....] - ETA: 17s - loss: 0.1964 - categorical_accuracy: 0.9399
50560/60000 [========================>.....] - ETA: 17s - loss: 0.1963 - categorical_accuracy: 0.9399
50592/60000 [========================>.....] - ETA: 17s - loss: 0.1964 - categorical_accuracy: 0.9399
50624/60000 [========================>.....] - ETA: 17s - loss: 0.1962 - categorical_accuracy: 0.9399
50656/60000 [========================>.....] - ETA: 17s - loss: 0.1961 - categorical_accuracy: 0.9400
50688/60000 [========================>.....] - ETA: 17s - loss: 0.1961 - categorical_accuracy: 0.9400
50720/60000 [========================>.....] - ETA: 17s - loss: 0.1960 - categorical_accuracy: 0.9400
50752/60000 [========================>.....] - ETA: 17s - loss: 0.1959 - categorical_accuracy: 0.9400
50784/60000 [========================>.....] - ETA: 17s - loss: 0.1958 - categorical_accuracy: 0.9400
50816/60000 [========================>.....] - ETA: 17s - loss: 0.1957 - categorical_accuracy: 0.9401
50848/60000 [========================>.....] - ETA: 17s - loss: 0.1958 - categorical_accuracy: 0.9401
50880/60000 [========================>.....] - ETA: 17s - loss: 0.1958 - categorical_accuracy: 0.9400
50912/60000 [========================>.....] - ETA: 17s - loss: 0.1957 - categorical_accuracy: 0.9401
50944/60000 [========================>.....] - ETA: 17s - loss: 0.1956 - categorical_accuracy: 0.9401
50976/60000 [========================>.....] - ETA: 16s - loss: 0.1955 - categorical_accuracy: 0.9401
51008/60000 [========================>.....] - ETA: 16s - loss: 0.1955 - categorical_accuracy: 0.9401
51040/60000 [========================>.....] - ETA: 16s - loss: 0.1954 - categorical_accuracy: 0.9402
51072/60000 [========================>.....] - ETA: 16s - loss: 0.1953 - categorical_accuracy: 0.9402
51104/60000 [========================>.....] - ETA: 16s - loss: 0.1953 - categorical_accuracy: 0.9402
51136/60000 [========================>.....] - ETA: 16s - loss: 0.1952 - categorical_accuracy: 0.9402
51168/60000 [========================>.....] - ETA: 16s - loss: 0.1953 - categorical_accuracy: 0.9401
51200/60000 [========================>.....] - ETA: 16s - loss: 0.1952 - categorical_accuracy: 0.9402
51232/60000 [========================>.....] - ETA: 16s - loss: 0.1951 - categorical_accuracy: 0.9402
51264/60000 [========================>.....] - ETA: 16s - loss: 0.1951 - categorical_accuracy: 0.9402
51296/60000 [========================>.....] - ETA: 16s - loss: 0.1950 - categorical_accuracy: 0.9402
51328/60000 [========================>.....] - ETA: 16s - loss: 0.1950 - categorical_accuracy: 0.9402
51360/60000 [========================>.....] - ETA: 16s - loss: 0.1949 - categorical_accuracy: 0.9403
51392/60000 [========================>.....] - ETA: 16s - loss: 0.1949 - categorical_accuracy: 0.9403
51424/60000 [========================>.....] - ETA: 16s - loss: 0.1948 - categorical_accuracy: 0.9403
51456/60000 [========================>.....] - ETA: 16s - loss: 0.1948 - categorical_accuracy: 0.9403
51488/60000 [========================>.....] - ETA: 16s - loss: 0.1947 - categorical_accuracy: 0.9404
51520/60000 [========================>.....] - ETA: 15s - loss: 0.1947 - categorical_accuracy: 0.9404
51552/60000 [========================>.....] - ETA: 15s - loss: 0.1946 - categorical_accuracy: 0.9404
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1945 - categorical_accuracy: 0.9404
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1944 - categorical_accuracy: 0.9405
51648/60000 [========================>.....] - ETA: 15s - loss: 0.1943 - categorical_accuracy: 0.9405
51680/60000 [========================>.....] - ETA: 15s - loss: 0.1943 - categorical_accuracy: 0.9405
51712/60000 [========================>.....] - ETA: 15s - loss: 0.1942 - categorical_accuracy: 0.9405
51744/60000 [========================>.....] - ETA: 15s - loss: 0.1941 - categorical_accuracy: 0.9405
51776/60000 [========================>.....] - ETA: 15s - loss: 0.1941 - categorical_accuracy: 0.9406
51808/60000 [========================>.....] - ETA: 15s - loss: 0.1939 - categorical_accuracy: 0.9406
51840/60000 [========================>.....] - ETA: 15s - loss: 0.1939 - categorical_accuracy: 0.9406
51872/60000 [========================>.....] - ETA: 15s - loss: 0.1940 - categorical_accuracy: 0.9406
51904/60000 [========================>.....] - ETA: 15s - loss: 0.1941 - categorical_accuracy: 0.9406
51936/60000 [========================>.....] - ETA: 15s - loss: 0.1942 - categorical_accuracy: 0.9406
51968/60000 [========================>.....] - ETA: 15s - loss: 0.1941 - categorical_accuracy: 0.9406
52000/60000 [=========================>....] - ETA: 15s - loss: 0.1942 - categorical_accuracy: 0.9405
52032/60000 [=========================>....] - ETA: 14s - loss: 0.1943 - categorical_accuracy: 0.9405
52064/60000 [=========================>....] - ETA: 14s - loss: 0.1942 - categorical_accuracy: 0.9406
52096/60000 [=========================>....] - ETA: 14s - loss: 0.1942 - categorical_accuracy: 0.9406
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1941 - categorical_accuracy: 0.9406
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1940 - categorical_accuracy: 0.9406
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1940 - categorical_accuracy: 0.9407
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1939 - categorical_accuracy: 0.9407
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9407
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9407
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1937 - categorical_accuracy: 0.9407
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1936 - categorical_accuracy: 0.9408
52384/60000 [=========================>....] - ETA: 14s - loss: 0.1935 - categorical_accuracy: 0.9408
52416/60000 [=========================>....] - ETA: 14s - loss: 0.1934 - categorical_accuracy: 0.9408
52448/60000 [=========================>....] - ETA: 14s - loss: 0.1933 - categorical_accuracy: 0.9409
52480/60000 [=========================>....] - ETA: 14s - loss: 0.1933 - categorical_accuracy: 0.9409
52512/60000 [=========================>....] - ETA: 14s - loss: 0.1932 - categorical_accuracy: 0.9409
52544/60000 [=========================>....] - ETA: 14s - loss: 0.1931 - categorical_accuracy: 0.9409
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1931 - categorical_accuracy: 0.9409
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1931 - categorical_accuracy: 0.9409
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1931 - categorical_accuracy: 0.9409
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1932 - categorical_accuracy: 0.9409
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1931 - categorical_accuracy: 0.9409
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1930 - categorical_accuracy: 0.9410
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1931 - categorical_accuracy: 0.9409
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1930 - categorical_accuracy: 0.9410
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9410
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1931 - categorical_accuracy: 0.9410
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1930 - categorical_accuracy: 0.9410
52928/60000 [=========================>....] - ETA: 13s - loss: 0.1930 - categorical_accuracy: 0.9410
52960/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9410
52992/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9410
53024/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9411
53056/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9411
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1927 - categorical_accuracy: 0.9411
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1927 - categorical_accuracy: 0.9411
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1927 - categorical_accuracy: 0.9411
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1926 - categorical_accuracy: 0.9411
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1925 - categorical_accuracy: 0.9411
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1923 - categorical_accuracy: 0.9412
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1923 - categorical_accuracy: 0.9412
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9412
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1923 - categorical_accuracy: 0.9412
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9412
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9412
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9413
53472/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9413
53504/60000 [=========================>....] - ETA: 12s - loss: 0.1920 - categorical_accuracy: 0.9413
53536/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9413
53568/60000 [=========================>....] - ETA: 12s - loss: 0.1920 - categorical_accuracy: 0.9413
53600/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9414
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1918 - categorical_accuracy: 0.9414
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1918 - categorical_accuracy: 0.9414
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1917 - categorical_accuracy: 0.9414
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1917 - categorical_accuracy: 0.9414
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1917 - categorical_accuracy: 0.9414
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1916 - categorical_accuracy: 0.9415
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1915 - categorical_accuracy: 0.9415
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1915 - categorical_accuracy: 0.9415
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1914 - categorical_accuracy: 0.9415
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1914 - categorical_accuracy: 0.9415
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1914 - categorical_accuracy: 0.9415
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1913 - categorical_accuracy: 0.9415
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1912 - categorical_accuracy: 0.9416
54048/60000 [==========================>...] - ETA: 11s - loss: 0.1911 - categorical_accuracy: 0.9416
54080/60000 [==========================>...] - ETA: 11s - loss: 0.1911 - categorical_accuracy: 0.9416
54112/60000 [==========================>...] - ETA: 11s - loss: 0.1910 - categorical_accuracy: 0.9416
54144/60000 [==========================>...] - ETA: 11s - loss: 0.1910 - categorical_accuracy: 0.9416
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1910 - categorical_accuracy: 0.9416
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1909 - categorical_accuracy: 0.9417
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1909 - categorical_accuracy: 0.9417
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1909 - categorical_accuracy: 0.9417
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1908 - categorical_accuracy: 0.9417
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9417
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1906 - categorical_accuracy: 0.9418
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1906 - categorical_accuracy: 0.9418
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1905 - categorical_accuracy: 0.9418
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1905 - categorical_accuracy: 0.9418
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9418
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9419
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1902 - categorical_accuracy: 0.9419
54592/60000 [==========================>...] - ETA: 10s - loss: 0.1901 - categorical_accuracy: 0.9419
54624/60000 [==========================>...] - ETA: 10s - loss: 0.1900 - categorical_accuracy: 0.9420
54656/60000 [==========================>...] - ETA: 10s - loss: 0.1899 - categorical_accuracy: 0.9420
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9420 
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1898 - categorical_accuracy: 0.9420
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9421
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9421
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1895 - categorical_accuracy: 0.9421
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1894 - categorical_accuracy: 0.9421
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1894 - categorical_accuracy: 0.9421
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1893 - categorical_accuracy: 0.9422
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1892 - categorical_accuracy: 0.9422
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1891 - categorical_accuracy: 0.9422
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1890 - categorical_accuracy: 0.9423
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1889 - categorical_accuracy: 0.9423
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1889 - categorical_accuracy: 0.9423
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1888 - categorical_accuracy: 0.9423
55136/60000 [==========================>...] - ETA: 9s - loss: 0.1888 - categorical_accuracy: 0.9424
55168/60000 [==========================>...] - ETA: 9s - loss: 0.1887 - categorical_accuracy: 0.9424
55200/60000 [==========================>...] - ETA: 9s - loss: 0.1887 - categorical_accuracy: 0.9424
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9424
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9424
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1885 - categorical_accuracy: 0.9425
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1885 - categorical_accuracy: 0.9425
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1884 - categorical_accuracy: 0.9425
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1884 - categorical_accuracy: 0.9425
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1884 - categorical_accuracy: 0.9425
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1883 - categorical_accuracy: 0.9425
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1882 - categorical_accuracy: 0.9425
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1882 - categorical_accuracy: 0.9426
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1881 - categorical_accuracy: 0.9426
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1880 - categorical_accuracy: 0.9426
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1879 - categorical_accuracy: 0.9426
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1879 - categorical_accuracy: 0.9426
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1878 - categorical_accuracy: 0.9427
55712/60000 [==========================>...] - ETA: 8s - loss: 0.1877 - categorical_accuracy: 0.9427
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1878 - categorical_accuracy: 0.9427
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1878 - categorical_accuracy: 0.9427
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1878 - categorical_accuracy: 0.9427
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9427
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9427
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9428
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9428
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9428
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1874 - categorical_accuracy: 0.9428
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1874 - categorical_accuracy: 0.9428
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1874 - categorical_accuracy: 0.9428
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1873 - categorical_accuracy: 0.9429
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1872 - categorical_accuracy: 0.9429
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1872 - categorical_accuracy: 0.9429
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1872 - categorical_accuracy: 0.9429
56224/60000 [===========================>..] - ETA: 7s - loss: 0.1871 - categorical_accuracy: 0.9429
56256/60000 [===========================>..] - ETA: 7s - loss: 0.1871 - categorical_accuracy: 0.9429
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9430
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9430
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9430
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9430
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9430
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9431
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9431
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9431
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9431
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9431
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9431
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1864 - categorical_accuracy: 0.9431
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1864 - categorical_accuracy: 0.9432
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1863 - categorical_accuracy: 0.9432
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1862 - categorical_accuracy: 0.9432
56768/60000 [===========================>..] - ETA: 6s - loss: 0.1861 - categorical_accuracy: 0.9432
56800/60000 [===========================>..] - ETA: 6s - loss: 0.1860 - categorical_accuracy: 0.9433
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1861 - categorical_accuracy: 0.9433
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9433
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9433
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9433
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9433
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9434
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9434
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9434
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1857 - categorical_accuracy: 0.9434
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1857 - categorical_accuracy: 0.9434
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1856 - categorical_accuracy: 0.9435
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1856 - categorical_accuracy: 0.9435
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1856 - categorical_accuracy: 0.9435
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1855 - categorical_accuracy: 0.9435
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1855 - categorical_accuracy: 0.9435
57312/60000 [===========================>..] - ETA: 5s - loss: 0.1854 - categorical_accuracy: 0.9435
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9435
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9435
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9435
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9436
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9436
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9436
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9436
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9436
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9436
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9436
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9436
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9436
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9436
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9436
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9436
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1849 - categorical_accuracy: 0.9436
57856/60000 [===========================>..] - ETA: 4s - loss: 0.1849 - categorical_accuracy: 0.9436
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9436
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9437
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9437
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9437
58016/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9437
58048/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9437
58080/60000 [============================>.] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9437
58112/60000 [============================>.] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9437
58144/60000 [============================>.] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9437
58176/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9437
58208/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9437
58240/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9437
58272/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9438
58304/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9438
58336/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9438
58368/60000 [============================>.] - ETA: 3s - loss: 0.1842 - categorical_accuracy: 0.9438
58400/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9439
58432/60000 [============================>.] - ETA: 2s - loss: 0.1840 - categorical_accuracy: 0.9439
58464/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9439
58496/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9439
58528/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9439
58560/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9439
58592/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9440
58624/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9440
58656/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9440
58688/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9440
58720/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9440
58752/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9440
58784/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9440
58816/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9440
58848/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9441
58880/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9441
58912/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9441
58944/60000 [============================>.] - ETA: 1s - loss: 0.1833 - categorical_accuracy: 0.9441
58976/60000 [============================>.] - ETA: 1s - loss: 0.1832 - categorical_accuracy: 0.9441
59008/60000 [============================>.] - ETA: 1s - loss: 0.1832 - categorical_accuracy: 0.9441
59040/60000 [============================>.] - ETA: 1s - loss: 0.1833 - categorical_accuracy: 0.9441
59072/60000 [============================>.] - ETA: 1s - loss: 0.1832 - categorical_accuracy: 0.9442
59104/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9442
59136/60000 [============================>.] - ETA: 1s - loss: 0.1830 - categorical_accuracy: 0.9442
59168/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9442
59200/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9443
59232/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9443
59264/60000 [============================>.] - ETA: 1s - loss: 0.1827 - categorical_accuracy: 0.9443
59296/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9443
59328/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9443
59360/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9444
59392/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9444
59424/60000 [============================>.] - ETA: 1s - loss: 0.1824 - categorical_accuracy: 0.9444
59456/60000 [============================>.] - ETA: 1s - loss: 0.1824 - categorical_accuracy: 0.9444
59488/60000 [============================>.] - ETA: 0s - loss: 0.1823 - categorical_accuracy: 0.9444
59520/60000 [============================>.] - ETA: 0s - loss: 0.1823 - categorical_accuracy: 0.9444
59552/60000 [============================>.] - ETA: 0s - loss: 0.1822 - categorical_accuracy: 0.9444
59584/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9445
59616/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9445
59648/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9445
59680/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9445
59712/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9445
59744/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9445
59776/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9445
59808/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9445
59840/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9446
59872/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9446
59904/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9446
59936/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9446
59968/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9446
60000/60000 [==============================] - 117s 2ms/step - loss: 0.1815 - categorical_accuracy: 0.9447 - val_loss: 0.0496 - val_categorical_accuracy: 0.9844

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  160/10000 [..............................] - ETA: 6s 
  320/10000 [..............................] - ETA: 4s
  480/10000 [>.............................] - ETA: 4s
  640/10000 [>.............................] - ETA: 4s
  800/10000 [=>............................] - ETA: 3s
  960/10000 [=>............................] - ETA: 3s
 1120/10000 [==>...........................] - ETA: 3s
 1280/10000 [==>...........................] - ETA: 3s
 1440/10000 [===>..........................] - ETA: 3s
 1568/10000 [===>..........................] - ETA: 3s
 1728/10000 [====>.........................] - ETA: 3s
 1888/10000 [====>.........................] - ETA: 3s
 2048/10000 [=====>........................] - ETA: 3s
 2208/10000 [=====>........................] - ETA: 3s
 2368/10000 [======>.......................] - ETA: 2s
 2528/10000 [======>.......................] - ETA: 2s
 2688/10000 [=======>......................] - ETA: 2s
 2848/10000 [=======>......................] - ETA: 2s
 3008/10000 [========>.....................] - ETA: 2s
 3168/10000 [========>.....................] - ETA: 2s
 3328/10000 [========>.....................] - ETA: 2s
 3488/10000 [=========>....................] - ETA: 2s
 3648/10000 [=========>....................] - ETA: 2s
 3808/10000 [==========>...................] - ETA: 2s
 3968/10000 [==========>...................] - ETA: 2s
 4128/10000 [===========>..................] - ETA: 2s
 4288/10000 [===========>..................] - ETA: 2s
 4448/10000 [============>.................] - ETA: 2s
 4608/10000 [============>.................] - ETA: 2s
 4768/10000 [=============>................] - ETA: 1s
 4928/10000 [=============>................] - ETA: 1s
 5088/10000 [==============>...............] - ETA: 1s
 5216/10000 [==============>...............] - ETA: 1s
 5376/10000 [===============>..............] - ETA: 1s
 5536/10000 [===============>..............] - ETA: 1s
 5696/10000 [================>.............] - ETA: 1s
 5824/10000 [================>.............] - ETA: 1s
 5984/10000 [================>.............] - ETA: 1s
 6144/10000 [=================>............] - ETA: 1s
 6304/10000 [=================>............] - ETA: 1s
 6464/10000 [==================>...........] - ETA: 1s
 6624/10000 [==================>...........] - ETA: 1s
 6784/10000 [===================>..........] - ETA: 1s
 6944/10000 [===================>..........] - ETA: 1s
 7104/10000 [====================>.........] - ETA: 1s
 7264/10000 [====================>.........] - ETA: 1s
 7424/10000 [=====================>........] - ETA: 0s
 7584/10000 [=====================>........] - ETA: 0s
 7744/10000 [======================>.......] - ETA: 0s
 7904/10000 [======================>.......] - ETA: 0s
 8064/10000 [=======================>......] - ETA: 0s
 8192/10000 [=======================>......] - ETA: 0s
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
10000/10000 [==============================] - 4s 377us/step
[[1.23936612e-08 6.21056842e-08 3.13724013e-06 ... 9.99985456e-01
  1.70417884e-08 8.75758815e-06]
 [4.24852551e-05 6.27452027e-05 9.99863982e-01 ... 1.72363116e-07
  2.05794231e-06 1.78269470e-08]
 [8.17313833e-07 9.99880552e-01 2.17542420e-05 ... 2.62474823e-05
  7.34269042e-06 8.72895839e-07]
 ...
 [1.94973193e-09 3.69860800e-07 1.05520197e-08 ... 5.25664575e-07
  2.52361383e-06 1.51145809e-06]
 [5.01952818e-05 8.53683559e-07 1.24357439e-06 ... 1.77746116e-07
  1.44809205e-02 1.11482059e-05]
 [1.04234495e-07 5.01874133e-08 9.27205917e-07 ... 2.45137444e-09
  3.85969201e-07 1.86517757e-09]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04961153943864629, 'accuracy_test:': 0.9843999743461609}

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
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
From github.com:arita37/mlmodels_store
   48cf09d..d8d3e30  master     -> origin/master
Updating 48cf09d..d8d3e30
Fast-forward
 .../20200521/list_log_pullrequest_20200521.md      |   2 +-
 error_list/20200521/list_log_testall_20200521.md   | 432 +++++++++++++++++++++
 2 files changed, 433 insertions(+), 1 deletion(-)
[master d039e2c] ml_store
 1 file changed, 2049 insertions(+)
To github.com:arita37/mlmodels_store.git
   d8d3e30..d039e2c  master -> master





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
{'loss': 0.4738125242292881, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-21 20:28:56.740421: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 8c79dc8] ml_store
 1 file changed, 234 insertions(+)
To github.com:arita37/mlmodels_store.git
   d039e2c..8c79dc8  master -> master





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
[master 794bf95] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   8c79dc8..794bf95  master -> master





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
 40%|████      | 2/5 [00:21<00:32, 10.78s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.88892270329116, 'learning_rate': 0.09616945472741677, 'min_data_in_leaf': 16, 'num_leaves': 66} and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xecr\x0e\x06i\xd6\x94X\r\x00\x00\x00learning_rateq\x02G?\xb8\x9e\x8f\xb6\xed\xac\x1fX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04KBu.' and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xecr\x0e\x06i\xd6\x94X\r\x00\x00\x00learning_rateq\x02G?\xb8\x9e\x8f\xb6\xed\xac\x1fX\x10\x00\x00\x00min_data_in_leafq\x03K\x10X\n\x00\x00\x00num_leavesq\x04KBu.' and reward: 0.3896
 60%|██████    | 3/5 [00:54<00:34, 17.48s/it] 60%|██████    | 3/5 [00:54<00:36, 18.23s/it]
Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9933701301639678, 'learning_rate': 0.14673510289339836, 'min_data_in_leaf': 21, 'num_leaves': 39} and reward: 0.3878
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\xc9\xb0'\xbc\x14\x85X\r\x00\x00\x00learning_rateq\x02G?\xc2\xc87B\r\x19\x82X\x10\x00\x00\x00min_data_in_leafq\x03K\x15X\n\x00\x00\x00num_leavesq\x04K'u." and reward: 0.3878
Finished Task with config: b"\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\xc9\xb0'\xbc\x14\x85X\r\x00\x00\x00learning_rateq\x02G?\xc2\xc87B\r\x19\x82X\x10\x00\x00\x00min_data_in_leafq\x03K\x15X\n\x00\x00\x00num_leavesq\x04K'u." and reward: 0.3878
Time for Gradient Boosting hyperparameter optimization: 77.07219457626343
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
Saving dataset/models/NeuralNetClassifier/trial_3_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.39
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.39
 40%|████      | 2/5 [00:54<01:21, 27.23s/it] 40%|████      | 2/5 [00:54<01:21, 27.23s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.06001986507357861, 'embedding_size_factor': 1.1388888358984905, 'layers.choice': 2, 'learning_rate': 0.005122345761115565, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 3.040707813381702e-12} and reward: 0.3904
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xae\xba\xec{3\xe4\x82X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf28\xe3\x7f\xffj\x9cX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?t\xfb+8]^\xe6X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x8a\xbf\x10\xb6\xb0\xe8\xa6u.' and reward: 0.3904
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xae\xba\xec{3\xe4\x82X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf28\xe3\x7f\xffj\x9cX\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?t\xfb+8]^\xe6X\x13\x00\x00\x00network_type.choiceq\x06K\x01X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\x8a\xbf\x10\xb6\xb0\xe8\xa6u.' and reward: 0.3904
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 109.67870831489563
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 2, 'dropout_prob': 0.06001986507357861, 'embedding_size_factor': 1.1388888358984905, 'layers.choice': 2, 'learning_rate': 0.005122345761115565, 'network_type.choice': 1, 'use_batchnorm.choice': 1, 'weight_decay': 3.040707813381702e-12}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_3_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -69.44s of remaining time.
Ensemble size: 85
Ensemble weights: 
[0.23529412 0.51764706 0.03529412 0.12941176 0.08235294]
	0.3996	 = Validation accuracy score
	1.37s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 190.86s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_3_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f842b06d048>

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
   794bf95..dab1400  master     -> origin/master
Updating 794bf95..dab1400
Fast-forward
 error_list/20200521/list_log_testall_20200521.md | 175 +++++++++++++++++++++++
 1 file changed, 175 insertions(+)
[master 1e6e3f9] ml_store
 1 file changed, 200 insertions(+)
To github.com:arita37/mlmodels_store.git
   dab1400..1e6e3f9  master -> master





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
[master 61aca2d] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   1e6e3f9..61aca2d  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.53it/s, avg_epoch_loss=5.25]
INFO:root:Epoch[0] Elapsed time 2.839 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.245047
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.245046997070313 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fb587f9f400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fb587f9f400>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 88.65it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 989.07470703125,
    "abs_error": 355.06005859375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.352610131987456,
    "sMAPE": 0.49765127422640454,
    "MSIS": 94.10441822080648,
    "QuantileLoss[0.5]": 355.06009674072266,
    "Coverage[0.5]": 1.0,
    "RMSE": 31.449558137297416,
    "NRMSE": 0.6620959607852087,
    "ND": 0.622912383497807,
    "wQuantileLoss[0.5]": 0.6229124504223205,
    "mean_wQuantileLoss": 0.6229124504223205,
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
100%|██████████| 10/10 [00:01<00:00,  7.27it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.377 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55c16ff28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55c16ff28>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 130.33it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:01<00:00,  5.15it/s, avg_epoch_loss=5.17]
INFO:root:Epoch[0] Elapsed time 1.944 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.165545
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.165544939041138 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55c2b1eb8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55c2b1eb8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 150.70it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 263.7734375,
    "abs_error": 170.58670043945312,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.1302989173878055,
    "sMAPE": 0.2836224667213057,
    "MSIS": 45.21195184252163,
    "QuantileLoss[0.5]": 170.58667373657227,
    "Coverage[0.5]": 0.75,
    "RMSE": 16.241103333825567,
    "NRMSE": 0.34191796492264354,
    "ND": 0.29927491305167214,
    "wQuantileLoss[0.5]": 0.29927486620451277,
    "mean_wQuantileLoss": 0.29927486620451277,
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
 30%|███       | 3/10 [00:13<00:30,  4.37s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:24<00:16,  4.23s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:36<00:04,  4.13s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:40<00:00,  4.04s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 40.394 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.865718
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.865717792510987 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55c19b668>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55c19b668>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 143.03it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 54263.755208333336,
    "abs_error": 2756.42333984375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.263922737597547,
    "sMAPE": 1.417917220418886,
    "MSIS": 730.5568318560524,
    "QuantileLoss[0.5]": 2756.4232482910156,
    "Coverage[0.5]": 1.0,
    "RMSE": 232.94582032810405,
    "NRMSE": 4.904122533223243,
    "ND": 4.8358304207785086,
    "wQuantileLoss[0.5]": 4.8358302601596765,
    "mean_wQuantileLoss": 4.8358302601596765,
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
100%|██████████| 10/10 [00:00<00:00, 49.24it/s, avg_epoch_loss=5.08]
INFO:root:Epoch[0] Elapsed time 0.204 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.077965
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.077965354919433 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55482c278>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55482c278>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 141.01it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 331.14377848307294,
    "abs_error": 199.930419921875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.3247289302649528,
    "sMAPE": 0.32750870265778415,
    "MSIS": 52.989160445925165,
    "QuantileLoss[0.5]": 199.93042755126953,
    "Coverage[0.5]": 0.75,
    "RMSE": 18.19735635973184,
    "NRMSE": 0.3831022391522493,
    "ND": 0.3507551226699561,
    "wQuantileLoss[0.5]": 0.3507551360548588,
    "mean_wQuantileLoss": 0.3507551360548588,
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
100%|██████████| 10/10 [00:01<00:00,  8.05it/s, avg_epoch_loss=144]
INFO:root:Epoch[0] Elapsed time 1.243 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=144.412806
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 144.41280616614463 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55c16ff28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fb55c16ff28>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 144.26it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [02:01<18:14, 121.61s/it, avg_epoch_loss=0.488] 20%|██        | 2/10 [05:03<18:38, 139.78s/it, avg_epoch_loss=0.47]  30%|███       | 3/10 [08:44<19:07, 163.98s/it, avg_epoch_loss=0.453] 40%|████      | 4/10 [12:18<17:54, 179.13s/it, avg_epoch_loss=0.437] 50%|█████     | 5/10 [16:07<16:09, 193.96s/it, avg_epoch_loss=0.424] 60%|██████    | 6/10 [19:38<13:16, 199.09s/it, avg_epoch_loss=0.414] 70%|███████   | 7/10 [23:18<10:16, 205.38s/it, avg_epoch_loss=0.407] 80%|████████  | 8/10 [26:56<06:58, 209.09s/it, avg_epoch_loss=0.404] 90%|█████████ | 9/10 [30:16<03:26, 206.45s/it, avg_epoch_loss=0.401]100%|██████████| 10/10 [33:33<00:00, 203.52s/it, avg_epoch_loss=0.399]100%|██████████| 10/10 [33:33<00:00, 201.33s/it, avg_epoch_loss=0.399]
INFO:root:Epoch[0] Elapsed time 2013.357 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.398938
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.3989381194114685 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7fb5548f0048>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7fb5548f0048>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 13.68it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
   61aca2d..be1e011  master     -> origin/master
Updating 61aca2d..be1e011
Fast-forward
 error_list/20200521/list_log_pullrequest_20200521.md | 2 +-
 error_list/20200521/list_log_testall_20200521.md     | 7 +++++++
 2 files changed, 8 insertions(+), 1 deletion(-)
[master 14fe54e] ml_store
 1 file changed, 506 insertions(+)
To github.com:arita37/mlmodels_store.git
   be1e011..14fe54e  master -> master





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7fd7e1414668> 

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
[master 8a55723] ml_store
 1 file changed, 109 insertions(+)
To github.com:arita37/mlmodels_store.git
   14fe54e..8a55723  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 1.64661853 -1.52568032 -0.6069984   0.79502609  1.08480038 -0.37443832
   0.42952614  0.1340482   1.20205486  0.10622272]
 [ 0.98379959 -0.40724002  0.93272141  0.16056499 -1.278618   -0.12014998
   0.19975956  0.38560229  0.71829074 -0.5301198 ]
 [ 0.9292506  -1.10657307 -1.95816909 -0.3592241  -1.21258781  0.5053819
   0.54264529  1.2179409  -1.94068096  0.67780757]
 [ 1.17867274 -0.59980453 -0.6946936   1.12341216  1.17899425  0.30526704
   0.01335268  1.3887794  -0.66134424  0.6218035 ]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 1.66752297  1.22372221 -0.4599301  -0.0593679  -0.493857    1.4489894
  -1.18110317 -0.47758085  0.02599999 -0.79079995]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.86146256  0.07432055 -1.34501002 -0.19956072 -1.47533915 -0.65460317
  -0.31456386  0.3180143  -0.89027155 -1.29525789]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 0.81583612 -1.39169388  2.50598029  0.45021774 -0.88286982  0.62743708
  -1.19586151  0.75133724  0.14039544  1.91979229]
 [ 0.46739791 -0.23787527 -0.15449119 -0.75566277 -0.54706224  1.85143789
  -1.46405357  0.20909668  1.55501599 -0.09243232]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f6a9b34edd8>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f6ab56c15f8> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 8.95510508e-01  9.20615118e-01  7.94528240e-01 -3.53679249e-02
   8.78099103e-01  2.11060505e+00 -1.02188594e+00 -1.30653407e+00
   7.63804802e-02 -1.87316098e+00]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 8.57719529e-01  9.81122462e-02 -2.60466059e-01  1.06032751e+00
  -1.39003042e+00 -1.71116766e+00  2.65642403e-01  1.65712464e+00
   1.41767401e+00  4.45096710e-01]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 7.00175710e-01  5.56073510e-01  8.96864073e-02  1.69380911e+00
   8.82393314e-01  1.96869779e-01 -5.63788735e-01  1.69869255e-01
  -1.16400797e+00 -6.01156801e-01]
 [ 7.88018455e-01  3.01960045e-01  7.00982122e-01 -3.94689681e-01
  -1.20376927e+00 -1.17181338e+00  7.55392029e-01  9.84012237e-01
  -5.59681422e-01 -1.98937450e-01]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 1.18559003e+00  8.64644065e-02  1.23289919e+00 -2.14246673e+00
   1.03334100e+00 -8.30168864e-01  3.67231814e-01  4.51615951e-01
   1.10417433e+00 -4.22856961e-01]
 [ 1.39198128e+00 -1.90221025e-01 -5.37223024e-01 -4.48738033e-01
   7.04557071e-01 -6.72448039e-01 -7.01344426e-01 -5.57494722e-01
   9.39168744e-01  1.56263850e-01]
 [ 9.47814113e-01 -1.13379204e+00  6.40985866e-01 -1.90548298e-01
  -1.23912256e+00  2.33339126e-01 -3.16901197e-01  4.34998324e-01
   9.10423603e-01  1.21987438e+00]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]
 [ 4.73307772e-01 -9.73267585e-01 -2.28140691e-01  1.75167729e-01
  -1.01366961e+00 -5.34836927e-02  3.93787731e-01 -1.83061987e-01
  -2.21028902e-01  5.80330113e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 8.71225789e-01 -2.09752935e-01 -4.56987858e-01  9.35147780e-01
  -8.73535822e-01  1.81252782e+00  9.25501215e-01  1.40109881e-01
  -1.41914878e+00  1.06898597e+00]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 7.61706684e-01 -1.48515645e+00  1.30253554e+00 -5.92461285e-01
  -1.64162479e+00 -2.30490794e+00 -1.34869645e+00 -3.18171727e-02
   1.12487742e-01 -3.62612088e-01]
 [ 8.15836116e-01 -1.39169388e+00  2.50598029e+00  4.50217742e-01
  -8.82869820e-01  6.27437083e-01 -1.19586151e+00  7.51337235e-01
   1.40395436e-01  1.91979229e+00]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 5.69983848e-01 -5.33020326e-01 -1.75458969e-01 -1.42655542e+00
   6.06604307e-01  1.76795995e+00 -1.15985185e-01 -4.75372875e-01
   4.77610182e-01 -9.33914656e-01]
 [ 8.88611457e-01  8.49586845e-01 -3.09114176e-02 -1.22154015e-01
  -1.14722826e+00 -6.80851574e-01 -3.26061306e-01 -1.06787658e+00
  -7.66793627e-02  3.55717262e-01]
 [ 6.13636707e-01  3.16658895e-01  1.34710546e+00 -1.89526695e+00
  -7.60458095e-01  8.97291174e-02 -3.29051549e-01  4.10265745e-01
   8.59870972e-01 -1.04906775e+00]]
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
[[ 1.37661405 -0.60022533  0.72591685 -0.37951752 -0.62754626 -1.01480369
   0.96622086  0.4359862  -0.68748739  3.32107876]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.62368852  1.2066079   0.90399917 -0.28286355 -1.18913787 -0.26632688
   1.42361443  1.06897162  0.04037143  1.57546791]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 1.01177337  0.09574677  0.73140252  1.0334508  -1.42203164 -0.14627327
  -0.01745495 -0.85749682 -0.93418184  0.95449567]
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.87699465  1.23225307 -0.86778722 -0.25417987  0.89189141  1.39984394
  -0.87728152 -0.78191168 -0.43750898 -1.44087602]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]]
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
[master 584d002] ml_store
 1 file changed, 272 insertions(+)
To github.com:arita37/mlmodels_store.git
   8a55723..584d002  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140288475238352
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140288475238128
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140288475236896
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140288475236448
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140288475235944
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140288475235608

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
grad_step = 000000, loss = 1.099550
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.962363
grad_step = 000002, loss = 0.855704
grad_step = 000003, loss = 0.738709
grad_step = 000004, loss = 0.594400
grad_step = 000005, loss = 0.434929
grad_step = 000006, loss = 0.289018
grad_step = 000007, loss = 0.221532
grad_step = 000008, loss = 0.258875
grad_step = 000009, loss = 0.237764
grad_step = 000010, loss = 0.170830
grad_step = 000011, loss = 0.102168
grad_step = 000012, loss = 0.063976
grad_step = 000013, loss = 0.050811
grad_step = 000014, loss = 0.045849
grad_step = 000015, loss = 0.040124
grad_step = 000016, loss = 0.032321
grad_step = 000017, loss = 0.026322
grad_step = 000018, loss = 0.025759
grad_step = 000019, loss = 0.028312
grad_step = 000020, loss = 0.026932
grad_step = 000021, loss = 0.019698
grad_step = 000022, loss = 0.011746
grad_step = 000023, loss = 0.008057
grad_step = 000024, loss = 0.008880
grad_step = 000025, loss = 0.011218
grad_step = 000026, loss = 0.012213
grad_step = 000027, loss = 0.011495
grad_step = 000028, loss = 0.010671
grad_step = 000029, loss = 0.011259
grad_step = 000030, loss = 0.012873
grad_step = 000031, loss = 0.013635
grad_step = 000032, loss = 0.012637
grad_step = 000033, loss = 0.010960
grad_step = 000034, loss = 0.010046
grad_step = 000035, loss = 0.009973
grad_step = 000036, loss = 0.009790
grad_step = 000037, loss = 0.008915
grad_step = 000038, loss = 0.007729
grad_step = 000039, loss = 0.007002
grad_step = 000040, loss = 0.006951
grad_step = 000041, loss = 0.006946
grad_step = 000042, loss = 0.006472
grad_step = 000043, loss = 0.005782
grad_step = 000044, loss = 0.005376
grad_step = 000045, loss = 0.005336
grad_step = 000046, loss = 0.005351
grad_step = 000047, loss = 0.005181
grad_step = 000048, loss = 0.004954
grad_step = 000049, loss = 0.004923
grad_step = 000050, loss = 0.005121
grad_step = 000051, loss = 0.005306
grad_step = 000052, loss = 0.005291
grad_step = 000053, loss = 0.005172
grad_step = 000054, loss = 0.005128
grad_step = 000055, loss = 0.005164
grad_step = 000056, loss = 0.005135
grad_step = 000057, loss = 0.004980
grad_step = 000058, loss = 0.004804
grad_step = 000059, loss = 0.004718
grad_step = 000060, loss = 0.004692
grad_step = 000061, loss = 0.004624
grad_step = 000062, loss = 0.004503
grad_step = 000063, loss = 0.004411
grad_step = 000064, loss = 0.004387
grad_step = 000065, loss = 0.004374
grad_step = 000066, loss = 0.004318
grad_step = 000067, loss = 0.004250
grad_step = 000068, loss = 0.004220
grad_step = 000069, loss = 0.004217
grad_step = 000070, loss = 0.004193
grad_step = 000071, loss = 0.004146
grad_step = 000072, loss = 0.004117
grad_step = 000073, loss = 0.004113
grad_step = 000074, loss = 0.004100
grad_step = 000075, loss = 0.004062
grad_step = 000076, loss = 0.004023
grad_step = 000077, loss = 0.003999
grad_step = 000078, loss = 0.003971
grad_step = 000079, loss = 0.003925
grad_step = 000080, loss = 0.003879
grad_step = 000081, loss = 0.003847
grad_step = 000082, loss = 0.003818
grad_step = 000083, loss = 0.003780
grad_step = 000084, loss = 0.003743
grad_step = 000085, loss = 0.003718
grad_step = 000086, loss = 0.003694
grad_step = 000087, loss = 0.003662
grad_step = 000088, loss = 0.003632
grad_step = 000089, loss = 0.003610
grad_step = 000090, loss = 0.003585
grad_step = 000091, loss = 0.003557
grad_step = 000092, loss = 0.003533
grad_step = 000093, loss = 0.003512
grad_step = 000094, loss = 0.003487
grad_step = 000095, loss = 0.003462
grad_step = 000096, loss = 0.003440
grad_step = 000097, loss = 0.003417
grad_step = 000098, loss = 0.003391
grad_step = 000099, loss = 0.003369
grad_step = 000100, loss = 0.003348
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.003325
grad_step = 000102, loss = 0.003304
grad_step = 000103, loss = 0.003286
grad_step = 000104, loss = 0.003265
grad_step = 000105, loss = 0.003247
grad_step = 000106, loss = 0.003230
grad_step = 000107, loss = 0.003211
grad_step = 000108, loss = 0.003195
grad_step = 000109, loss = 0.003178
grad_step = 000110, loss = 0.003160
grad_step = 000111, loss = 0.003145
grad_step = 000112, loss = 0.003128
grad_step = 000113, loss = 0.003113
grad_step = 000114, loss = 0.003098
grad_step = 000115, loss = 0.003083
grad_step = 000116, loss = 0.003068
grad_step = 000117, loss = 0.003053
grad_step = 000118, loss = 0.003039
grad_step = 000119, loss = 0.003025
grad_step = 000120, loss = 0.003011
grad_step = 000121, loss = 0.002997
grad_step = 000122, loss = 0.002984
grad_step = 000123, loss = 0.002971
grad_step = 000124, loss = 0.002957
grad_step = 000125, loss = 0.002944
grad_step = 000126, loss = 0.002931
grad_step = 000127, loss = 0.002918
grad_step = 000128, loss = 0.002905
grad_step = 000129, loss = 0.002892
grad_step = 000130, loss = 0.002878
grad_step = 000131, loss = 0.002866
grad_step = 000132, loss = 0.002853
grad_step = 000133, loss = 0.002840
grad_step = 000134, loss = 0.002827
grad_step = 000135, loss = 0.002813
grad_step = 000136, loss = 0.002800
grad_step = 000137, loss = 0.002787
grad_step = 000138, loss = 0.002773
grad_step = 000139, loss = 0.002760
grad_step = 000140, loss = 0.002746
grad_step = 000141, loss = 0.002732
grad_step = 000142, loss = 0.002717
grad_step = 000143, loss = 0.002702
grad_step = 000144, loss = 0.002688
grad_step = 000145, loss = 0.002673
grad_step = 000146, loss = 0.002659
grad_step = 000147, loss = 0.002645
grad_step = 000148, loss = 0.002632
grad_step = 000149, loss = 0.002618
grad_step = 000150, loss = 0.002602
grad_step = 000151, loss = 0.002587
grad_step = 000152, loss = 0.002573
grad_step = 000153, loss = 0.002560
grad_step = 000154, loss = 0.002544
grad_step = 000155, loss = 0.002529
grad_step = 000156, loss = 0.002514
grad_step = 000157, loss = 0.002501
grad_step = 000158, loss = 0.002486
grad_step = 000159, loss = 0.002470
grad_step = 000160, loss = 0.002455
grad_step = 000161, loss = 0.002441
grad_step = 000162, loss = 0.002427
grad_step = 000163, loss = 0.002411
grad_step = 000164, loss = 0.002394
grad_step = 000165, loss = 0.002378
grad_step = 000166, loss = 0.002363
grad_step = 000167, loss = 0.002347
grad_step = 000168, loss = 0.002330
grad_step = 000169, loss = 0.002312
grad_step = 000170, loss = 0.002295
grad_step = 000171, loss = 0.002279
grad_step = 000172, loss = 0.002263
grad_step = 000173, loss = 0.002246
grad_step = 000174, loss = 0.002228
grad_step = 000175, loss = 0.002209
grad_step = 000176, loss = 0.002189
grad_step = 000177, loss = 0.002170
grad_step = 000178, loss = 0.002152
grad_step = 000179, loss = 0.002137
grad_step = 000180, loss = 0.002125
grad_step = 000181, loss = 0.002110
grad_step = 000182, loss = 0.002083
grad_step = 000183, loss = 0.002055
grad_step = 000184, loss = 0.002039
grad_step = 000185, loss = 0.002025
grad_step = 000186, loss = 0.002001
grad_step = 000187, loss = 0.001976
grad_step = 000188, loss = 0.001960
grad_step = 000189, loss = 0.001943
grad_step = 000190, loss = 0.001920
grad_step = 000191, loss = 0.001898
grad_step = 000192, loss = 0.001881
grad_step = 000193, loss = 0.001865
grad_step = 000194, loss = 0.001845
grad_step = 000195, loss = 0.001824
grad_step = 000196, loss = 0.001803
grad_step = 000197, loss = 0.001786
grad_step = 000198, loss = 0.001772
grad_step = 000199, loss = 0.001756
grad_step = 000200, loss = 0.001737
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.001718
grad_step = 000202, loss = 0.001699
grad_step = 000203, loss = 0.001685
grad_step = 000204, loss = 0.001671
grad_step = 000205, loss = 0.001658
grad_step = 000206, loss = 0.001647
grad_step = 000207, loss = 0.001636
grad_step = 000208, loss = 0.001620
grad_step = 000209, loss = 0.001598
grad_step = 000210, loss = 0.001568
grad_step = 000211, loss = 0.001545
grad_step = 000212, loss = 0.001530
grad_step = 000213, loss = 0.001520
grad_step = 000214, loss = 0.001516
grad_step = 000215, loss = 0.001504
grad_step = 000216, loss = 0.001481
grad_step = 000217, loss = 0.001441
grad_step = 000218, loss = 0.001415
grad_step = 000219, loss = 0.001403
grad_step = 000220, loss = 0.001400
grad_step = 000221, loss = 0.001382
grad_step = 000222, loss = 0.001349
grad_step = 000223, loss = 0.001311
grad_step = 000224, loss = 0.001293
grad_step = 000225, loss = 0.001284
grad_step = 000226, loss = 0.001270
grad_step = 000227, loss = 0.001242
grad_step = 000228, loss = 0.001208
grad_step = 000229, loss = 0.001178
grad_step = 000230, loss = 0.001159
grad_step = 000231, loss = 0.001145
grad_step = 000232, loss = 0.001130
grad_step = 000233, loss = 0.001112
grad_step = 000234, loss = 0.001088
grad_step = 000235, loss = 0.001059
grad_step = 000236, loss = 0.001023
grad_step = 000237, loss = 0.000990
grad_step = 000238, loss = 0.000961
grad_step = 000239, loss = 0.000938
grad_step = 000240, loss = 0.000918
grad_step = 000241, loss = 0.000913
grad_step = 000242, loss = 0.000941
grad_step = 000243, loss = 0.001012
grad_step = 000244, loss = 0.001015
grad_step = 000245, loss = 0.000852
grad_step = 000246, loss = 0.000735
grad_step = 000247, loss = 0.000768
grad_step = 000248, loss = 0.000792
grad_step = 000249, loss = 0.000685
grad_step = 000250, loss = 0.000662
grad_step = 000251, loss = 0.000704
grad_step = 000252, loss = 0.000656
grad_step = 000253, loss = 0.000611
grad_step = 000254, loss = 0.000642
grad_step = 000255, loss = 0.000635
grad_step = 000256, loss = 0.000585
grad_step = 000257, loss = 0.000591
grad_step = 000258, loss = 0.000613
grad_step = 000259, loss = 0.000585
grad_step = 000260, loss = 0.000567
grad_step = 000261, loss = 0.000587
grad_step = 000262, loss = 0.000578
grad_step = 000263, loss = 0.000553
grad_step = 000264, loss = 0.000563
grad_step = 000265, loss = 0.000571
grad_step = 000266, loss = 0.000544
grad_step = 000267, loss = 0.000535
grad_step = 000268, loss = 0.000540
grad_step = 000269, loss = 0.000532
grad_step = 000270, loss = 0.000519
grad_step = 000271, loss = 0.000523
grad_step = 000272, loss = 0.000519
grad_step = 000273, loss = 0.000508
grad_step = 000274, loss = 0.000506
grad_step = 000275, loss = 0.000507
grad_step = 000276, loss = 0.000497
grad_step = 000277, loss = 0.000493
grad_step = 000278, loss = 0.000493
grad_step = 000279, loss = 0.000488
grad_step = 000280, loss = 0.000482
grad_step = 000281, loss = 0.000479
grad_step = 000282, loss = 0.000478
grad_step = 000283, loss = 0.000472
grad_step = 000284, loss = 0.000469
grad_step = 000285, loss = 0.000467
grad_step = 000286, loss = 0.000464
grad_step = 000287, loss = 0.000459
grad_step = 000288, loss = 0.000457
grad_step = 000289, loss = 0.000455
grad_step = 000290, loss = 0.000451
grad_step = 000291, loss = 0.000448
grad_step = 000292, loss = 0.000445
grad_step = 000293, loss = 0.000443
grad_step = 000294, loss = 0.000440
grad_step = 000295, loss = 0.000436
grad_step = 000296, loss = 0.000433
grad_step = 000297, loss = 0.000431
grad_step = 000298, loss = 0.000428
grad_step = 000299, loss = 0.000426
grad_step = 000300, loss = 0.000423
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000420
grad_step = 000302, loss = 0.000417
grad_step = 000303, loss = 0.000415
grad_step = 000304, loss = 0.000413
grad_step = 000305, loss = 0.000411
grad_step = 000306, loss = 0.000409
grad_step = 000307, loss = 0.000406
grad_step = 000308, loss = 0.000403
grad_step = 000309, loss = 0.000401
grad_step = 000310, loss = 0.000399
grad_step = 000311, loss = 0.000397
grad_step = 000312, loss = 0.000395
grad_step = 000313, loss = 0.000393
grad_step = 000314, loss = 0.000391
grad_step = 000315, loss = 0.000390
grad_step = 000316, loss = 0.000389
grad_step = 000317, loss = 0.000391
grad_step = 000318, loss = 0.000396
grad_step = 000319, loss = 0.000402
grad_step = 000320, loss = 0.000405
grad_step = 000321, loss = 0.000399
grad_step = 000322, loss = 0.000386
grad_step = 000323, loss = 0.000376
grad_step = 000324, loss = 0.000374
grad_step = 000325, loss = 0.000379
grad_step = 000326, loss = 0.000384
grad_step = 000327, loss = 0.000384
grad_step = 000328, loss = 0.000379
grad_step = 000329, loss = 0.000371
grad_step = 000330, loss = 0.000365
grad_step = 000331, loss = 0.000364
grad_step = 000332, loss = 0.000366
grad_step = 000333, loss = 0.000370
grad_step = 000334, loss = 0.000370
grad_step = 000335, loss = 0.000367
grad_step = 000336, loss = 0.000361
grad_step = 000337, loss = 0.000356
grad_step = 000338, loss = 0.000353
grad_step = 000339, loss = 0.000354
grad_step = 000340, loss = 0.000356
grad_step = 000341, loss = 0.000357
grad_step = 000342, loss = 0.000357
grad_step = 000343, loss = 0.000355
grad_step = 000344, loss = 0.000351
grad_step = 000345, loss = 0.000347
grad_step = 000346, loss = 0.000344
grad_step = 000347, loss = 0.000342
grad_step = 000348, loss = 0.000342
grad_step = 000349, loss = 0.000342
grad_step = 000350, loss = 0.000343
grad_step = 000351, loss = 0.000343
grad_step = 000352, loss = 0.000343
grad_step = 000353, loss = 0.000343
grad_step = 000354, loss = 0.000342
grad_step = 000355, loss = 0.000341
grad_step = 000356, loss = 0.000340
grad_step = 000357, loss = 0.000337
grad_step = 000358, loss = 0.000334
grad_step = 000359, loss = 0.000331
grad_step = 000360, loss = 0.000330
grad_step = 000361, loss = 0.000328
grad_step = 000362, loss = 0.000327
grad_step = 000363, loss = 0.000325
grad_step = 000364, loss = 0.000324
grad_step = 000365, loss = 0.000323
grad_step = 000366, loss = 0.000321
grad_step = 000367, loss = 0.000320
grad_step = 000368, loss = 0.000320
grad_step = 000369, loss = 0.000319
grad_step = 000370, loss = 0.000319
grad_step = 000371, loss = 0.000320
grad_step = 000372, loss = 0.000322
grad_step = 000373, loss = 0.000329
grad_step = 000374, loss = 0.000342
grad_step = 000375, loss = 0.000363
grad_step = 000376, loss = 0.000391
grad_step = 000377, loss = 0.000399
grad_step = 000378, loss = 0.000367
grad_step = 000379, loss = 0.000321
grad_step = 000380, loss = 0.000316
grad_step = 000381, loss = 0.000335
grad_step = 000382, loss = 0.000340
grad_step = 000383, loss = 0.000332
grad_step = 000384, loss = 0.000327
grad_step = 000385, loss = 0.000320
grad_step = 000386, loss = 0.000307
grad_step = 000387, loss = 0.000315
grad_step = 000388, loss = 0.000330
grad_step = 000389, loss = 0.000313
grad_step = 000390, loss = 0.000300
grad_step = 000391, loss = 0.000305
grad_step = 000392, loss = 0.000307
grad_step = 000393, loss = 0.000303
grad_step = 000394, loss = 0.000301
grad_step = 000395, loss = 0.000300
grad_step = 000396, loss = 0.000296
grad_step = 000397, loss = 0.000293
grad_step = 000398, loss = 0.000297
grad_step = 000399, loss = 0.000298
grad_step = 000400, loss = 0.000293
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000288
grad_step = 000402, loss = 0.000289
grad_step = 000403, loss = 0.000290
grad_step = 000404, loss = 0.000288
grad_step = 000405, loss = 0.000285
grad_step = 000406, loss = 0.000285
grad_step = 000407, loss = 0.000285
grad_step = 000408, loss = 0.000283
grad_step = 000409, loss = 0.000281
grad_step = 000410, loss = 0.000279
grad_step = 000411, loss = 0.000279
grad_step = 000412, loss = 0.000279
grad_step = 000413, loss = 0.000278
grad_step = 000414, loss = 0.000277
grad_step = 000415, loss = 0.000275
grad_step = 000416, loss = 0.000275
grad_step = 000417, loss = 0.000275
grad_step = 000418, loss = 0.000274
grad_step = 000419, loss = 0.000272
grad_step = 000420, loss = 0.000271
grad_step = 000421, loss = 0.000271
grad_step = 000422, loss = 0.000271
grad_step = 000423, loss = 0.000273
grad_step = 000424, loss = 0.000274
grad_step = 000425, loss = 0.000275
grad_step = 000426, loss = 0.000276
grad_step = 000427, loss = 0.000275
grad_step = 000428, loss = 0.000277
grad_step = 000429, loss = 0.000281
grad_step = 000430, loss = 0.000287
grad_step = 000431, loss = 0.000291
grad_step = 000432, loss = 0.000293
grad_step = 000433, loss = 0.000290
grad_step = 000434, loss = 0.000285
grad_step = 000435, loss = 0.000274
grad_step = 000436, loss = 0.000265
grad_step = 000437, loss = 0.000259
grad_step = 000438, loss = 0.000257
grad_step = 000439, loss = 0.000259
grad_step = 000440, loss = 0.000262
grad_step = 000441, loss = 0.000267
grad_step = 000442, loss = 0.000271
grad_step = 000443, loss = 0.000274
grad_step = 000444, loss = 0.000274
grad_step = 000445, loss = 0.000271
grad_step = 000446, loss = 0.000265
grad_step = 000447, loss = 0.000257
grad_step = 000448, loss = 0.000252
grad_step = 000449, loss = 0.000250
grad_step = 000450, loss = 0.000253
grad_step = 000451, loss = 0.000254
grad_step = 000452, loss = 0.000255
grad_step = 000453, loss = 0.000253
grad_step = 000454, loss = 0.000253
grad_step = 000455, loss = 0.000256
grad_step = 000456, loss = 0.000259
grad_step = 000457, loss = 0.000264
grad_step = 000458, loss = 0.000261
grad_step = 000459, loss = 0.000255
grad_step = 000460, loss = 0.000248
grad_step = 000461, loss = 0.000248
grad_step = 000462, loss = 0.000250
grad_step = 000463, loss = 0.000248
grad_step = 000464, loss = 0.000243
grad_step = 000465, loss = 0.000238
grad_step = 000466, loss = 0.000237
grad_step = 000467, loss = 0.000239
grad_step = 000468, loss = 0.000241
grad_step = 000469, loss = 0.000242
grad_step = 000470, loss = 0.000239
grad_step = 000471, loss = 0.000236
grad_step = 000472, loss = 0.000236
grad_step = 000473, loss = 0.000240
grad_step = 000474, loss = 0.000248
grad_step = 000475, loss = 0.000256
grad_step = 000476, loss = 0.000269
grad_step = 000477, loss = 0.000283
grad_step = 000478, loss = 0.000302
grad_step = 000479, loss = 0.000330
grad_step = 000480, loss = 0.000332
grad_step = 000481, loss = 0.000293
grad_step = 000482, loss = 0.000250
grad_step = 000483, loss = 0.000241
grad_step = 000484, loss = 0.000256
grad_step = 000485, loss = 0.000260
grad_step = 000486, loss = 0.000270
grad_step = 000487, loss = 0.000278
grad_step = 000488, loss = 0.000257
grad_step = 000489, loss = 0.000231
grad_step = 000490, loss = 0.000224
grad_step = 000491, loss = 0.000239
grad_step = 000492, loss = 0.000249
grad_step = 000493, loss = 0.000243
grad_step = 000494, loss = 0.000238
grad_step = 000495, loss = 0.000231
grad_step = 000496, loss = 0.000222
grad_step = 000497, loss = 0.000221
grad_step = 000498, loss = 0.000229
grad_step = 000499, loss = 0.000236
grad_step = 000500, loss = 0.000232
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000221
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
[[0.855744   0.8380061  0.930659   0.95225257 1.0330184 ]
 [0.8486302  0.90539116 0.9422161  0.9971072  0.9959437 ]
 [0.90159273 0.921127   0.9946149  0.98514396 0.96681625]
 [0.9210743  0.98029965 0.9929113  0.94855404 0.90718085]
 [0.9797653  0.9911056  0.9630988  0.9174226  0.8622895 ]
 [0.97534204 0.94257385 0.91502994 0.85735226 0.8589467 ]
 [0.9283693  0.9013191  0.84020275 0.8468347  0.818738  ]
 [0.8965081  0.8320962  0.8421592  0.8094809  0.8372803 ]
 [0.8178468  0.83036035 0.8063819  0.8356983  0.8467927 ]
 [0.82108474 0.82268065 0.8265119  0.83584243 0.83024734]
 [0.81841296 0.8153632  0.8499201  0.85076994 0.9223798 ]
 [0.81230783 0.8418731  0.83465713 0.9164707  0.94142884]
 [0.8501241  0.83289766 0.93060964 0.9501924  1.0308043 ]
 [0.84947014 0.91296107 0.94245416 0.99736255 0.9876152 ]
 [0.91568756 0.93175226 1.0029101  0.9734865  0.94062173]
 [0.9281304  0.99222225 0.98816854 0.9344911  0.8886377 ]
 [0.9846426  0.98506635 0.94587296 0.89957625 0.850191  ]
 [0.96939516 0.92942697 0.90154165 0.8409844  0.84908426]
 [0.9261006  0.8908406  0.83198184 0.8341178  0.81368506]
 [0.9067019  0.83740026 0.8512512  0.8067698  0.84658736]
 [0.8353684  0.844263   0.82209426 0.84119064 0.8571649 ]
 [0.83951414 0.83505607 0.83475333 0.84129864 0.83776134]
 [0.83401287 0.8274579  0.85824007 0.8554328  0.92765623]
 [0.8226687  0.85526997 0.83716017 0.9208799  0.93873197]
 [0.8615968  0.84010404 0.93520397 0.9538601  1.0383248 ]
 [0.8574641  0.9083864  0.9518134  1.0019109  1.0064898 ]
 [0.91114295 0.92950934 0.9996467  0.9955559  0.98057574]
 [0.93174195 0.9924324  1.0049336  0.95869005 0.9193657 ]
 [0.9855404  0.99808866 0.9731818  0.92427975 0.8708032 ]
 [0.9825535  0.9510173  0.92542535 0.8643992  0.8669264 ]
 [0.9357418  0.90709025 0.84639543 0.85175604 0.82608974]]

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
