
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
Already up to date.
[master 11cf607] ml_store
 2 files changed, 63 insertions(+), 10226 deletions(-)
 rewrite log_testall/log_testall.py (99%)
To github.com:arita37/mlmodels_store.git
   6617275..11cf607  master -> master





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
[master 9aabce8] ml_store
 1 file changed, 49 insertions(+)
To github.com:arita37/mlmodels_store.git
   11cf607..9aabce8  master -> master





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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-20 08:11:52.003412: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-20 08:11:52.018932: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-20 08:11:52.019158: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556bdad21e80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 08:11:52.019177: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2500 - binary_crossentropy: 0.6930 - val_loss: 0.2502 - val_binary_crossentropy: 0.6935

  #### metrics   #################################################### 
{'MSE': 0.24990341520282827}

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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 2, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         24          sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
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
Total params: 458
Trainable params: 458
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2667 - binary_crossentropy: 0.7332500/500 [==============================] - 1s 2ms/sample - loss: 0.2724 - binary_crossentropy: 0.7465 - val_loss: 0.2690 - val_binary_crossentropy: 0.7366

  #### metrics   #################################################### 
{'MSE': 0.26997577919639937}

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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_max[0][0]               
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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 652
Trainable params: 652
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 2ms/sample - loss: 0.2502 - binary_crossentropy: 0.6935 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.24995596578472568}

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
sequence_mean (InputLayer)      [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 7, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
100/500 [=====>........................] - ETA: 2s - loss: 0.2581 - binary_crossentropy: 0.7160500/500 [==============================] - 1s 3ms/sample - loss: 0.2670 - binary_crossentropy: 0.7347 - val_loss: 0.2683 - val_binary_crossentropy: 0.7342

  #### metrics   #################################################### 
{'MSE': 0.2653252951403009}

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
sequence_sum (InputLayer)       [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 3, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         36          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         1           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
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
Total params: 133
Trainable params: 133
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2502 - binary_crossentropy: 0.6935500/500 [==============================] - 2s 4ms/sample - loss: 0.2517 - binary_crossentropy: 0.6966 - val_loss: 0.2497 - val_binary_crossentropy: 0.6926

  #### metrics   #################################################### 
{'MSE': 0.25008179792715907}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_max[0][0]               
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
Total params: 133
Trainable params: 133
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-20 08:13:16.944095: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:16.946135: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:16.951446: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-20 08:13:16.961306: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-20 08:13:16.963034: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 08:13:16.964530: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:16.965897: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2549 - val_binary_crossentropy: 0.7030
2020-05-20 08:13:18.343016: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:18.344759: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:18.349147: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-20 08:13:18.358174: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-20 08:13:18.359725: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 08:13:18.361137: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:18.362402: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.25616800517235944}

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
2020-05-20 08:13:42.840642: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:42.842006: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:42.845459: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-20 08:13:42.851751: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-20 08:13:42.852787: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 08:13:42.853735: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:42.854633: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2516 - val_binary_crossentropy: 0.6963
2020-05-20 08:13:44.441810: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:44.443068: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:44.445596: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-20 08:13:44.450563: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-20 08:13:44.451416: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 08:13:44.452200: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:13:44.452924: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2519931032377576}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-20 08:14:18.830266: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:14:18.835551: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:14:18.851149: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-20 08:14:18.878104: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-20 08:14:18.882923: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 08:14:18.887279: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:14:18.891529: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.0547 - binary_crossentropy: 0.2664 - val_loss: 0.3065 - val_binary_crossentropy: 0.8213
2020-05-20 08:14:21.270303: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:14:21.275398: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:14:21.288097: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-20 08:14:21.316812: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-20 08:14:21.321615: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-20 08:14:21.325718: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-20 08:14:21.329714: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.22726935253040756}

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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 735
Trainable params: 735
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2580 - binary_crossentropy: 0.7092500/500 [==============================] - 4s 9ms/sample - loss: 0.2538 - binary_crossentropy: 0.7273 - val_loss: 0.2559 - val_binary_crossentropy: 0.7314

  #### metrics   #################################################### 
{'MSE': 0.2541078010148135}

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
sparse_seq_emb_sequence_sum (Em (None, 2, 4)         32          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 2, 1)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 735
Trainable params: 735
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         2           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         12          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         18          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
Total params: 233
Trainable params: 233
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.5700 - binary_crossentropy: 8.7465500/500 [==============================] - 5s 9ms/sample - loss: 0.5260 - binary_crossentropy: 8.0703 - val_loss: 0.5020 - val_binary_crossentropy: 7.7433

  #### metrics   #################################################### 
{'MSE': 0.506}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 2)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 2)         2           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         12          sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         18          sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_2[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
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
Total params: 1,869
Trainable params: 1,869
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2535 - binary_crossentropy: 0.7002500/500 [==============================] - 5s 9ms/sample - loss: 0.2526 - binary_crossentropy: 0.6985 - val_loss: 0.2530 - val_binary_crossentropy: 0.6992

  #### metrics   #################################################### 
{'MSE': 0.2522373632094695}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 4, 4)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         2           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_max[0][0]               
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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 9s - loss: 0.2752 - binary_crossentropy: 0.7478500/500 [==============================] - 6s 12ms/sample - loss: 0.2774 - binary_crossentropy: 0.7516 - val_loss: 0.2606 - val_binary_crossentropy: 0.7161

  #### metrics   #################################################### 
{'MSE': 0.2678734561164383}

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
regionsequence_mean (InputLayer [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         7           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         7           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 5, 1)         5           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 1,362
Trainable params: 1,362
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2446 - binary_crossentropy: 0.6822500/500 [==============================] - 6s 11ms/sample - loss: 0.2539 - binary_crossentropy: 0.7014 - val_loss: 0.2585 - val_binary_crossentropy: 0.7102

  #### metrics   #################################################### 
{'MSE': 0.2548825130707922}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_101 (NoMask)            (None, 1, 4)         0           bi_interaction_pooling[0][0]     
__________________________________________________________________________________________________
no_mask_102 (NoMask)            (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 1,362
Trainable params: 1,362
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
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         32          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 8, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_11[0][0]                    
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
Total params: 3,239
Trainable params: 3,159
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2653 - binary_crossentropy: 0.7251500/500 [==============================] - 6s 13ms/sample - loss: 0.2611 - binary_crossentropy: 0.7162 - val_loss: 0.2517 - val_binary_crossentropy: 0.6967

  #### metrics   #################################################### 
{'MSE': 0.2529275608965124}

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
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         16          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         32          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         16          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         32          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 6, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 8, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           hash_10[0][0]                    
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           hash_11[0][0]                    
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
Total params: 3,239
Trainable params: 3,159
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
Warning: Permanently added the RSA host key for IP address '192.30.255.112' to the list of known hosts.
From github.com:arita37/mlmodels_store
   9aabce8..6f6e60b  master     -> origin/master
Updating 9aabce8..6f6e60b
Fast-forward
 error_list/20200520/list_log_testall_20200520.md   | 773 ---------------------
 ...-10_69b309ad857428cc5a734b8afd99842edf9b2a42.py | 635 +++++++++++++++++
 2 files changed, 635 insertions(+), 773 deletions(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-20-08-10_69b309ad857428cc5a734b8afd99842edf9b2a42.py
[master d65f0b4] ml_store
 1 file changed, 4955 insertions(+)
To github.com:arita37/mlmodels_store.git
   6f6e60b..d65f0b4  master -> master





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
[master 694c4f9] ml_store
 1 file changed, 51 insertions(+)
To github.com:arita37/mlmodels_store.git
   d65f0b4..694c4f9  master -> master





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
[master 4f95992] ml_store
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   694c4f9..4f95992  master -> master





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
[master ecba73a] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   4f95992..ecba73a  master -> master





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

2020-05-20 08:23:19.700395: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-20 08:23:19.705398: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-20 08:23:19.705555: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5570bf41b6c0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 08:23:19.705569: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

128/354 [=========>....................] - ETA: 6s - loss: 1.3820
256/354 [====================>.........] - ETA: 2s - loss: 1.1685
354/354 [==============================] - 10s 28ms/step - loss: 1.5537 - val_loss: 3.4955

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
[master b43ddfa] ml_store
 1 file changed, 150 insertions(+)
To github.com:arita37/mlmodels_store.git
   ecba73a..b43ddfa  master -> master





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
[master 894e49f] ml_store
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   b43ddfa..894e49f  master -> master





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
[master 1af244d] ml_store
 1 file changed, 45 insertions(+)
To github.com:arita37/mlmodels_store.git
   894e49f..1af244d  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
   24576/17464789 [..............................] - ETA: 47s
   57344/17464789 [..............................] - ETA: 40s
   90112/17464789 [..............................] - ETA: 38s
  180224/17464789 [..............................] - ETA: 25s
  368640/17464789 [..............................] - ETA: 15s
  737280/17464789 [>.............................] - ETA: 9s 
 1466368/17464789 [=>............................] - ETA: 5s
 2908160/17464789 [===>..........................] - ETA: 2s
 5627904/17464789 [========>.....................] - ETA: 1s
 8708096/17464789 [=============>................] - ETA: 0s
11706368/17464789 [===================>..........] - ETA: 0s
14737408/17464789 [========================>.....] - ETA: 0s
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
2020-05-20 08:24:21.982074: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-05-20 08:24:21.986376: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-05-20 08:24:21.986541: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ec51706010 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-20 08:24:21.986555: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 12s - loss: 7.9426 - accuracy: 0.4820
 2000/25000 [=>............................] - ETA: 8s - loss: 7.8506 - accuracy: 0.4880 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6973 - accuracy: 0.4980
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6053 - accuracy: 0.5040
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.5992 - accuracy: 0.5044
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6385 - accuracy: 0.5018
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.6513 - accuracy: 0.5010
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6935 - accuracy: 0.4983
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7484 - accuracy: 0.4947
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7356 - accuracy: 0.4955
11000/25000 [============>.................] - ETA: 3s - loss: 7.7879 - accuracy: 0.4921
12000/25000 [=============>................] - ETA: 3s - loss: 7.7625 - accuracy: 0.4938
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7197 - accuracy: 0.4965
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6929 - accuracy: 0.4983
15000/25000 [=================>............] - ETA: 2s - loss: 7.6728 - accuracy: 0.4996
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6743 - accuracy: 0.4995
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6531 - accuracy: 0.5009
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6538 - accuracy: 0.5008
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6440 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6605 - accuracy: 0.5004
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6878 - accuracy: 0.4986
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6924 - accuracy: 0.4983
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6713 - accuracy: 0.4997
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6634 - accuracy: 0.5002
25000/25000 [==============================] - 7s 291us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fd1d3ac0160>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fd1db850860> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 11s - loss: 7.8200 - accuracy: 0.4900
 2000/25000 [=>............................] - ETA: 8s - loss: 7.5976 - accuracy: 0.5045 
 3000/25000 [==>...........................] - ETA: 6s - loss: 7.6871 - accuracy: 0.4987
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6475 - accuracy: 0.5013
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.7341 - accuracy: 0.4956
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.7203 - accuracy: 0.4965
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.7893 - accuracy: 0.4920
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.7759 - accuracy: 0.4929
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.7194 - accuracy: 0.4966
10000/25000 [===========>..................] - ETA: 3s - loss: 7.7356 - accuracy: 0.4955
11000/25000 [============>.................] - ETA: 3s - loss: 7.7210 - accuracy: 0.4965
12000/25000 [=============>................] - ETA: 3s - loss: 7.7024 - accuracy: 0.4977
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6419 - accuracy: 0.5016
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6666 - accuracy: 0.5000
15000/25000 [=================>............] - ETA: 2s - loss: 7.6503 - accuracy: 0.5011
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6503 - accuracy: 0.5011
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6360 - accuracy: 0.5020
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6147 - accuracy: 0.5034
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6319 - accuracy: 0.5023
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6528 - accuracy: 0.5009
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6272 - accuracy: 0.5026
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6276 - accuracy: 0.5025
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6393 - accuracy: 0.5018
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6564 - accuracy: 0.5007
25000/25000 [==============================] - 7s 297us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 11s - loss: 8.0806 - accuracy: 0.4730
 2000/25000 [=>............................] - ETA: 8s - loss: 7.9350 - accuracy: 0.4825 
 3000/25000 [==>...........................] - ETA: 7s - loss: 7.6564 - accuracy: 0.5007
 4000/25000 [===>..........................] - ETA: 6s - loss: 7.6781 - accuracy: 0.4992
 5000/25000 [=====>........................] - ETA: 5s - loss: 7.6360 - accuracy: 0.5020
 6000/25000 [======>.......................] - ETA: 5s - loss: 7.6078 - accuracy: 0.5038
 7000/25000 [=======>......................] - ETA: 4s - loss: 7.5943 - accuracy: 0.5047
 8000/25000 [========>.....................] - ETA: 4s - loss: 7.6168 - accuracy: 0.5033
 9000/25000 [=========>....................] - ETA: 4s - loss: 7.6547 - accuracy: 0.5008
10000/25000 [===========>..................] - ETA: 3s - loss: 7.6712 - accuracy: 0.4997
11000/25000 [============>.................] - ETA: 3s - loss: 7.6959 - accuracy: 0.4981
12000/25000 [=============>................] - ETA: 3s - loss: 7.6730 - accuracy: 0.4996
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6784 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 2s - loss: 7.6732 - accuracy: 0.4996
15000/25000 [=================>............] - ETA: 2s - loss: 7.6707 - accuracy: 0.4997
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6685 - accuracy: 0.4999
17000/25000 [===================>..........] - ETA: 1s - loss: 7.6549 - accuracy: 0.5008
18000/25000 [====================>.........] - ETA: 1s - loss: 7.6470 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6569 - accuracy: 0.5006
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6682 - accuracy: 0.4999
21000/25000 [========================>.....] - ETA: 0s - loss: 7.6681 - accuracy: 0.4999
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6799 - accuracy: 0.4991
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6940 - accuracy: 0.4982
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6781 - accuracy: 0.4992
25000/25000 [==============================] - 7s 294us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
   1af244d..e2be3ff  master     -> origin/master
Updating 1af244d..e2be3ff
Fast-forward
 error_list/20200520/list_log_testall_20200520.md | 103 +++++++++++++++++++++++
 1 file changed, 103 insertions(+)
[master 8f4328a] ml_store
 1 file changed, 332 insertions(+)
To github.com:arita37/mlmodels_store.git
   e2be3ff..8f4328a  master -> master





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

13/13 [==============================] - 2s 143ms/step - loss: nan
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
[master cfff6ef] ml_store
 1 file changed, 126 insertions(+)
To github.com:arita37/mlmodels_store.git
   8f4328a..cfff6ef  master -> master





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
   24576/11490434 [..............................] - ETA: 31s
   57344/11490434 [..............................] - ETA: 26s
   90112/11490434 [..............................] - ETA: 25s
  180224/11490434 [..............................] - ETA: 16s
  262144/11490434 [..............................] - ETA: 14s
  524288/11490434 [>.............................] - ETA: 8s 
 1064960/11490434 [=>............................] - ETA: 4s
 2105344/11490434 [====>.........................] - ETA: 2s
 4186112/11490434 [=========>....................] - ETA: 1s
 7249920/11490434 [=================>............] - ETA: 0s
10248192/11490434 [=========================>....] - ETA: 0s
11493376/11490434 [==============================] - 1s 0us/step

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

   32/60000 [..............................] - ETA: 8:12 - loss: 2.2928 - categorical_accuracy: 0.0938
   64/60000 [..............................] - ETA: 4:56 - loss: 2.3105 - categorical_accuracy: 0.1094
   96/60000 [..............................] - ETA: 3:49 - loss: 2.2828 - categorical_accuracy: 0.1562
  160/60000 [..............................] - ETA: 2:52 - loss: 2.2052 - categorical_accuracy: 0.2188
  224/60000 [..............................] - ETA: 2:27 - loss: 2.1109 - categorical_accuracy: 0.2545
  288/60000 [..............................] - ETA: 2:14 - loss: 2.0744 - categorical_accuracy: 0.2708
  320/60000 [..............................] - ETA: 2:10 - loss: 2.0448 - categorical_accuracy: 0.2812
  384/60000 [..............................] - ETA: 2:02 - loss: 1.9599 - categorical_accuracy: 0.3229
  448/60000 [..............................] - ETA: 1:58 - loss: 1.9053 - categorical_accuracy: 0.3415
  512/60000 [..............................] - ETA: 1:54 - loss: 1.8966 - categorical_accuracy: 0.3477
  576/60000 [..............................] - ETA: 1:51 - loss: 1.8377 - categorical_accuracy: 0.3785
  640/60000 [..............................] - ETA: 1:48 - loss: 1.8079 - categorical_accuracy: 0.3828
  704/60000 [..............................] - ETA: 1:46 - loss: 1.7615 - categorical_accuracy: 0.4006
  736/60000 [..............................] - ETA: 1:45 - loss: 1.7262 - categorical_accuracy: 0.4130
  768/60000 [..............................] - ETA: 1:45 - loss: 1.6914 - categorical_accuracy: 0.4219
  832/60000 [..............................] - ETA: 1:43 - loss: 1.6280 - categorical_accuracy: 0.4423
  896/60000 [..............................] - ETA: 1:42 - loss: 1.5699 - categorical_accuracy: 0.4643
  960/60000 [..............................] - ETA: 1:41 - loss: 1.5378 - categorical_accuracy: 0.4771
 1024/60000 [..............................] - ETA: 1:40 - loss: 1.4946 - categorical_accuracy: 0.4971
 1088/60000 [..............................] - ETA: 1:39 - loss: 1.4602 - categorical_accuracy: 0.5129
 1152/60000 [..............................] - ETA: 1:38 - loss: 1.4279 - categorical_accuracy: 0.5269
 1184/60000 [..............................] - ETA: 1:38 - loss: 1.4075 - categorical_accuracy: 0.5312
 1248/60000 [..............................] - ETA: 1:37 - loss: 1.3760 - categorical_accuracy: 0.5377
 1312/60000 [..............................] - ETA: 1:37 - loss: 1.3722 - categorical_accuracy: 0.5396
 1376/60000 [..............................] - ETA: 1:36 - loss: 1.3415 - categorical_accuracy: 0.5516
 1440/60000 [..............................] - ETA: 1:35 - loss: 1.3112 - categorical_accuracy: 0.5618
 1504/60000 [..............................] - ETA: 1:35 - loss: 1.2826 - categorical_accuracy: 0.5725
 1568/60000 [..............................] - ETA: 1:35 - loss: 1.2853 - categorical_accuracy: 0.5727
 1632/60000 [..............................] - ETA: 1:34 - loss: 1.2638 - categorical_accuracy: 0.5803
 1696/60000 [..............................] - ETA: 1:34 - loss: 1.2431 - categorical_accuracy: 0.5867
 1728/60000 [..............................] - ETA: 1:34 - loss: 1.2311 - categorical_accuracy: 0.5920
 1760/60000 [..............................] - ETA: 1:34 - loss: 1.2171 - categorical_accuracy: 0.5972
 1824/60000 [..............................] - ETA: 1:33 - loss: 1.1966 - categorical_accuracy: 0.6036
 1888/60000 [..............................] - ETA: 1:33 - loss: 1.1832 - categorical_accuracy: 0.6075
 1952/60000 [..............................] - ETA: 1:32 - loss: 1.1619 - categorical_accuracy: 0.6153
 2016/60000 [>.............................] - ETA: 1:32 - loss: 1.1497 - categorical_accuracy: 0.6220
 2080/60000 [>.............................] - ETA: 1:31 - loss: 1.1326 - categorical_accuracy: 0.6269
 2144/60000 [>.............................] - ETA: 1:31 - loss: 1.1120 - categorical_accuracy: 0.6339
 2208/60000 [>.............................] - ETA: 1:31 - loss: 1.0944 - categorical_accuracy: 0.6404
 2272/60000 [>.............................] - ETA: 1:30 - loss: 1.0796 - categorical_accuracy: 0.6452
 2336/60000 [>.............................] - ETA: 1:30 - loss: 1.0670 - categorical_accuracy: 0.6503
 2400/60000 [>.............................] - ETA: 1:30 - loss: 1.0520 - categorical_accuracy: 0.6550
 2464/60000 [>.............................] - ETA: 1:30 - loss: 1.0386 - categorical_accuracy: 0.6595
 2496/60000 [>.............................] - ETA: 1:30 - loss: 1.0319 - categorical_accuracy: 0.6623
 2560/60000 [>.............................] - ETA: 1:29 - loss: 1.0202 - categorical_accuracy: 0.6676
 2592/60000 [>.............................] - ETA: 1:29 - loss: 1.0125 - categorical_accuracy: 0.6698
 2656/60000 [>.............................] - ETA: 1:29 - loss: 0.9976 - categorical_accuracy: 0.6743
 2720/60000 [>.............................] - ETA: 1:29 - loss: 0.9869 - categorical_accuracy: 0.6790
 2784/60000 [>.............................] - ETA: 1:29 - loss: 0.9743 - categorical_accuracy: 0.6835
 2816/60000 [>.............................] - ETA: 1:29 - loss: 0.9703 - categorical_accuracy: 0.6839
 2880/60000 [>.............................] - ETA: 1:28 - loss: 0.9609 - categorical_accuracy: 0.6878
 2944/60000 [>.............................] - ETA: 1:28 - loss: 0.9489 - categorical_accuracy: 0.6916
 3008/60000 [>.............................] - ETA: 1:28 - loss: 0.9365 - categorical_accuracy: 0.6955
 3072/60000 [>.............................] - ETA: 1:28 - loss: 0.9287 - categorical_accuracy: 0.6982
 3104/60000 [>.............................] - ETA: 1:28 - loss: 0.9237 - categorical_accuracy: 0.6994
 3168/60000 [>.............................] - ETA: 1:28 - loss: 0.9123 - categorical_accuracy: 0.7033
 3232/60000 [>.............................] - ETA: 1:28 - loss: 0.9038 - categorical_accuracy: 0.7070
 3296/60000 [>.............................] - ETA: 1:27 - loss: 0.8962 - categorical_accuracy: 0.7103
 3360/60000 [>.............................] - ETA: 1:27 - loss: 0.8899 - categorical_accuracy: 0.7128
 3424/60000 [>.............................] - ETA: 1:27 - loss: 0.8823 - categorical_accuracy: 0.7158
 3488/60000 [>.............................] - ETA: 1:27 - loss: 0.8723 - categorical_accuracy: 0.7196
 3520/60000 [>.............................] - ETA: 1:27 - loss: 0.8662 - categorical_accuracy: 0.7216
 3552/60000 [>.............................] - ETA: 1:27 - loss: 0.8611 - categorical_accuracy: 0.7233
 3616/60000 [>.............................] - ETA: 1:27 - loss: 0.8554 - categorical_accuracy: 0.7248
 3648/60000 [>.............................] - ETA: 1:27 - loss: 0.8505 - categorical_accuracy: 0.7267
 3712/60000 [>.............................] - ETA: 1:26 - loss: 0.8427 - categorical_accuracy: 0.7298
 3776/60000 [>.............................] - ETA: 1:26 - loss: 0.8353 - categorical_accuracy: 0.7325
 3840/60000 [>.............................] - ETA: 1:26 - loss: 0.8249 - categorical_accuracy: 0.7359
 3904/60000 [>.............................] - ETA: 1:26 - loss: 0.8163 - categorical_accuracy: 0.7387
 3968/60000 [>.............................] - ETA: 1:26 - loss: 0.8069 - categorical_accuracy: 0.7412
 4032/60000 [=>............................] - ETA: 1:26 - loss: 0.7972 - categorical_accuracy: 0.7443
 4096/60000 [=>............................] - ETA: 1:26 - loss: 0.7886 - categorical_accuracy: 0.7473
 4128/60000 [=>............................] - ETA: 1:26 - loss: 0.7840 - categorical_accuracy: 0.7488
 4160/60000 [=>............................] - ETA: 1:26 - loss: 0.7790 - categorical_accuracy: 0.7505
 4224/60000 [=>............................] - ETA: 1:25 - loss: 0.7732 - categorical_accuracy: 0.7526
 4256/60000 [=>............................] - ETA: 1:25 - loss: 0.7698 - categorical_accuracy: 0.7540
 4320/60000 [=>............................] - ETA: 1:25 - loss: 0.7620 - categorical_accuracy: 0.7565
 4384/60000 [=>............................] - ETA: 1:25 - loss: 0.7562 - categorical_accuracy: 0.7578
 4416/60000 [=>............................] - ETA: 1:25 - loss: 0.7540 - categorical_accuracy: 0.7582
 4480/60000 [=>............................] - ETA: 1:25 - loss: 0.7483 - categorical_accuracy: 0.7600
 4544/60000 [=>............................] - ETA: 1:25 - loss: 0.7424 - categorical_accuracy: 0.7617
 4608/60000 [=>............................] - ETA: 1:25 - loss: 0.7374 - categorical_accuracy: 0.7632
 4672/60000 [=>............................] - ETA: 1:25 - loss: 0.7330 - categorical_accuracy: 0.7643
 4736/60000 [=>............................] - ETA: 1:24 - loss: 0.7273 - categorical_accuracy: 0.7660
 4800/60000 [=>............................] - ETA: 1:24 - loss: 0.7214 - categorical_accuracy: 0.7677
 4864/60000 [=>............................] - ETA: 1:24 - loss: 0.7161 - categorical_accuracy: 0.7691
 4928/60000 [=>............................] - ETA: 1:24 - loss: 0.7115 - categorical_accuracy: 0.7709
 4992/60000 [=>............................] - ETA: 1:24 - loss: 0.7063 - categorical_accuracy: 0.7728
 5056/60000 [=>............................] - ETA: 1:24 - loss: 0.7006 - categorical_accuracy: 0.7747
 5120/60000 [=>............................] - ETA: 1:24 - loss: 0.6939 - categorical_accuracy: 0.7770
 5184/60000 [=>............................] - ETA: 1:24 - loss: 0.6883 - categorical_accuracy: 0.7791
 5216/60000 [=>............................] - ETA: 1:24 - loss: 0.6869 - categorical_accuracy: 0.7795
 5280/60000 [=>............................] - ETA: 1:23 - loss: 0.6809 - categorical_accuracy: 0.7816
 5344/60000 [=>............................] - ETA: 1:23 - loss: 0.6744 - categorical_accuracy: 0.7837
 5408/60000 [=>............................] - ETA: 1:23 - loss: 0.6696 - categorical_accuracy: 0.7853
 5472/60000 [=>............................] - ETA: 1:23 - loss: 0.6639 - categorical_accuracy: 0.7873
 5536/60000 [=>............................] - ETA: 1:23 - loss: 0.6598 - categorical_accuracy: 0.7890
 5600/60000 [=>............................] - ETA: 1:23 - loss: 0.6559 - categorical_accuracy: 0.7900
 5664/60000 [=>............................] - ETA: 1:23 - loss: 0.6517 - categorical_accuracy: 0.7913
 5728/60000 [=>............................] - ETA: 1:22 - loss: 0.6477 - categorical_accuracy: 0.7919
 5792/60000 [=>............................] - ETA: 1:22 - loss: 0.6452 - categorical_accuracy: 0.7933
 5856/60000 [=>............................] - ETA: 1:22 - loss: 0.6419 - categorical_accuracy: 0.7947
 5920/60000 [=>............................] - ETA: 1:22 - loss: 0.6383 - categorical_accuracy: 0.7959
 5984/60000 [=>............................] - ETA: 1:22 - loss: 0.6369 - categorical_accuracy: 0.7958
 6048/60000 [==>...........................] - ETA: 1:22 - loss: 0.6314 - categorical_accuracy: 0.7976
 6112/60000 [==>...........................] - ETA: 1:21 - loss: 0.6283 - categorical_accuracy: 0.7986
 6176/60000 [==>...........................] - ETA: 1:21 - loss: 0.6236 - categorical_accuracy: 0.8004
 6240/60000 [==>...........................] - ETA: 1:21 - loss: 0.6198 - categorical_accuracy: 0.8013
 6272/60000 [==>...........................] - ETA: 1:21 - loss: 0.6187 - categorical_accuracy: 0.8018
 6336/60000 [==>...........................] - ETA: 1:21 - loss: 0.6150 - categorical_accuracy: 0.8029
 6400/60000 [==>...........................] - ETA: 1:21 - loss: 0.6102 - categorical_accuracy: 0.8044
 6464/60000 [==>...........................] - ETA: 1:21 - loss: 0.6075 - categorical_accuracy: 0.8052
 6528/60000 [==>...........................] - ETA: 1:21 - loss: 0.6051 - categorical_accuracy: 0.8059
 6592/60000 [==>...........................] - ETA: 1:21 - loss: 0.6013 - categorical_accuracy: 0.8073
 6624/60000 [==>...........................] - ETA: 1:21 - loss: 0.6013 - categorical_accuracy: 0.8074
 6688/60000 [==>...........................] - ETA: 1:20 - loss: 0.5979 - categorical_accuracy: 0.8085
 6720/60000 [==>...........................] - ETA: 1:20 - loss: 0.5965 - categorical_accuracy: 0.8092
 6784/60000 [==>...........................] - ETA: 1:20 - loss: 0.5928 - categorical_accuracy: 0.8101
 6816/60000 [==>...........................] - ETA: 1:20 - loss: 0.5908 - categorical_accuracy: 0.8104
 6880/60000 [==>...........................] - ETA: 1:20 - loss: 0.5884 - categorical_accuracy: 0.8115
 6944/60000 [==>...........................] - ETA: 1:20 - loss: 0.5863 - categorical_accuracy: 0.8122
 7008/60000 [==>...........................] - ETA: 1:20 - loss: 0.5830 - categorical_accuracy: 0.8132
 7072/60000 [==>...........................] - ETA: 1:20 - loss: 0.5805 - categorical_accuracy: 0.8143
 7136/60000 [==>...........................] - ETA: 1:20 - loss: 0.5779 - categorical_accuracy: 0.8153
 7200/60000 [==>...........................] - ETA: 1:19 - loss: 0.5743 - categorical_accuracy: 0.8165
 7264/60000 [==>...........................] - ETA: 1:19 - loss: 0.5733 - categorical_accuracy: 0.8168
 7328/60000 [==>...........................] - ETA: 1:19 - loss: 0.5706 - categorical_accuracy: 0.8178
 7392/60000 [==>...........................] - ETA: 1:19 - loss: 0.5679 - categorical_accuracy: 0.8189
 7456/60000 [==>...........................] - ETA: 1:19 - loss: 0.5647 - categorical_accuracy: 0.8200
 7520/60000 [==>...........................] - ETA: 1:19 - loss: 0.5622 - categorical_accuracy: 0.8206
 7584/60000 [==>...........................] - ETA: 1:19 - loss: 0.5606 - categorical_accuracy: 0.8212
 7648/60000 [==>...........................] - ETA: 1:19 - loss: 0.5565 - categorical_accuracy: 0.8226
 7712/60000 [==>...........................] - ETA: 1:18 - loss: 0.5526 - categorical_accuracy: 0.8240
 7776/60000 [==>...........................] - ETA: 1:18 - loss: 0.5505 - categorical_accuracy: 0.8246
 7840/60000 [==>...........................] - ETA: 1:18 - loss: 0.5486 - categorical_accuracy: 0.8251
 7904/60000 [==>...........................] - ETA: 1:18 - loss: 0.5455 - categorical_accuracy: 0.8262
 7936/60000 [==>...........................] - ETA: 1:18 - loss: 0.5443 - categorical_accuracy: 0.8266
 8000/60000 [===>..........................] - ETA: 1:18 - loss: 0.5417 - categorical_accuracy: 0.8275
 8064/60000 [===>..........................] - ETA: 1:18 - loss: 0.5395 - categorical_accuracy: 0.8284
 8128/60000 [===>..........................] - ETA: 1:18 - loss: 0.5371 - categorical_accuracy: 0.8290
 8192/60000 [===>..........................] - ETA: 1:18 - loss: 0.5345 - categorical_accuracy: 0.8297
 8256/60000 [===>..........................] - ETA: 1:17 - loss: 0.5313 - categorical_accuracy: 0.8305
 8320/60000 [===>..........................] - ETA: 1:17 - loss: 0.5293 - categorical_accuracy: 0.8314
 8384/60000 [===>..........................] - ETA: 1:17 - loss: 0.5272 - categorical_accuracy: 0.8322
 8448/60000 [===>..........................] - ETA: 1:17 - loss: 0.5242 - categorical_accuracy: 0.8332
 8512/60000 [===>..........................] - ETA: 1:17 - loss: 0.5236 - categorical_accuracy: 0.8338
 8576/60000 [===>..........................] - ETA: 1:17 - loss: 0.5208 - categorical_accuracy: 0.8347
 8640/60000 [===>..........................] - ETA: 1:17 - loss: 0.5194 - categorical_accuracy: 0.8351
 8704/60000 [===>..........................] - ETA: 1:17 - loss: 0.5163 - categorical_accuracy: 0.8362
 8768/60000 [===>..........................] - ETA: 1:16 - loss: 0.5136 - categorical_accuracy: 0.8368
 8800/60000 [===>..........................] - ETA: 1:16 - loss: 0.5130 - categorical_accuracy: 0.8373
 8864/60000 [===>..........................] - ETA: 1:16 - loss: 0.5117 - categorical_accuracy: 0.8379
 8928/60000 [===>..........................] - ETA: 1:16 - loss: 0.5110 - categorical_accuracy: 0.8380
 8992/60000 [===>..........................] - ETA: 1:16 - loss: 0.5087 - categorical_accuracy: 0.8387
 9056/60000 [===>..........................] - ETA: 1:16 - loss: 0.5068 - categorical_accuracy: 0.8394
 9120/60000 [===>..........................] - ETA: 1:16 - loss: 0.5047 - categorical_accuracy: 0.8399
 9184/60000 [===>..........................] - ETA: 1:16 - loss: 0.5029 - categorical_accuracy: 0.8405
 9216/60000 [===>..........................] - ETA: 1:16 - loss: 0.5022 - categorical_accuracy: 0.8407
 9280/60000 [===>..........................] - ETA: 1:16 - loss: 0.4994 - categorical_accuracy: 0.8415
 9344/60000 [===>..........................] - ETA: 1:15 - loss: 0.4978 - categorical_accuracy: 0.8419
 9408/60000 [===>..........................] - ETA: 1:15 - loss: 0.4953 - categorical_accuracy: 0.8428
 9472/60000 [===>..........................] - ETA: 1:15 - loss: 0.4950 - categorical_accuracy: 0.8431
 9536/60000 [===>..........................] - ETA: 1:15 - loss: 0.4926 - categorical_accuracy: 0.8440
 9568/60000 [===>..........................] - ETA: 1:15 - loss: 0.4916 - categorical_accuracy: 0.8443
 9632/60000 [===>..........................] - ETA: 1:15 - loss: 0.4893 - categorical_accuracy: 0.8450
 9696/60000 [===>..........................] - ETA: 1:15 - loss: 0.4867 - categorical_accuracy: 0.8459
 9760/60000 [===>..........................] - ETA: 1:15 - loss: 0.4852 - categorical_accuracy: 0.8463
 9824/60000 [===>..........................] - ETA: 1:15 - loss: 0.4837 - categorical_accuracy: 0.8468
 9888/60000 [===>..........................] - ETA: 1:14 - loss: 0.4817 - categorical_accuracy: 0.8474
 9952/60000 [===>..........................] - ETA: 1:14 - loss: 0.4799 - categorical_accuracy: 0.8478
10016/60000 [====>.........................] - ETA: 1:14 - loss: 0.4783 - categorical_accuracy: 0.8483
10080/60000 [====>.........................] - ETA: 1:14 - loss: 0.4769 - categorical_accuracy: 0.8487
10144/60000 [====>.........................] - ETA: 1:14 - loss: 0.4749 - categorical_accuracy: 0.8495
10208/60000 [====>.........................] - ETA: 1:14 - loss: 0.4737 - categorical_accuracy: 0.8499
10272/60000 [====>.........................] - ETA: 1:14 - loss: 0.4715 - categorical_accuracy: 0.8508
10304/60000 [====>.........................] - ETA: 1:14 - loss: 0.4711 - categorical_accuracy: 0.8511
10368/60000 [====>.........................] - ETA: 1:14 - loss: 0.4689 - categorical_accuracy: 0.8518
10432/60000 [====>.........................] - ETA: 1:13 - loss: 0.4669 - categorical_accuracy: 0.8523
10496/60000 [====>.........................] - ETA: 1:13 - loss: 0.4650 - categorical_accuracy: 0.8529
10560/60000 [====>.........................] - ETA: 1:13 - loss: 0.4637 - categorical_accuracy: 0.8534
10624/60000 [====>.........................] - ETA: 1:13 - loss: 0.4617 - categorical_accuracy: 0.8540
10688/60000 [====>.........................] - ETA: 1:13 - loss: 0.4596 - categorical_accuracy: 0.8547
10752/60000 [====>.........................] - ETA: 1:13 - loss: 0.4577 - categorical_accuracy: 0.8553
10816/60000 [====>.........................] - ETA: 1:13 - loss: 0.4565 - categorical_accuracy: 0.8558
10880/60000 [====>.........................] - ETA: 1:13 - loss: 0.4554 - categorical_accuracy: 0.8562
10944/60000 [====>.........................] - ETA: 1:13 - loss: 0.4546 - categorical_accuracy: 0.8565
11008/60000 [====>.........................] - ETA: 1:12 - loss: 0.4542 - categorical_accuracy: 0.8566
11072/60000 [====>.........................] - ETA: 1:12 - loss: 0.4533 - categorical_accuracy: 0.8569
11136/60000 [====>.........................] - ETA: 1:12 - loss: 0.4519 - categorical_accuracy: 0.8575
11200/60000 [====>.........................] - ETA: 1:12 - loss: 0.4505 - categorical_accuracy: 0.8581
11264/60000 [====>.........................] - ETA: 1:12 - loss: 0.4494 - categorical_accuracy: 0.8585
11328/60000 [====>.........................] - ETA: 1:12 - loss: 0.4481 - categorical_accuracy: 0.8589
11392/60000 [====>.........................] - ETA: 1:12 - loss: 0.4473 - categorical_accuracy: 0.8593
11456/60000 [====>.........................] - ETA: 1:12 - loss: 0.4456 - categorical_accuracy: 0.8597
11520/60000 [====>.........................] - ETA: 1:12 - loss: 0.4446 - categorical_accuracy: 0.8602
11584/60000 [====>.........................] - ETA: 1:11 - loss: 0.4429 - categorical_accuracy: 0.8608
11648/60000 [====>.........................] - ETA: 1:11 - loss: 0.4410 - categorical_accuracy: 0.8613
11712/60000 [====>.........................] - ETA: 1:11 - loss: 0.4403 - categorical_accuracy: 0.8615
11776/60000 [====>.........................] - ETA: 1:11 - loss: 0.4384 - categorical_accuracy: 0.8621
11840/60000 [====>.........................] - ETA: 1:11 - loss: 0.4365 - categorical_accuracy: 0.8627
11904/60000 [====>.........................] - ETA: 1:11 - loss: 0.4346 - categorical_accuracy: 0.8632
11968/60000 [====>.........................] - ETA: 1:11 - loss: 0.4336 - categorical_accuracy: 0.8636
12032/60000 [=====>........................] - ETA: 1:11 - loss: 0.4322 - categorical_accuracy: 0.8639
12096/60000 [=====>........................] - ETA: 1:11 - loss: 0.4305 - categorical_accuracy: 0.8645
12160/60000 [=====>........................] - ETA: 1:11 - loss: 0.4295 - categorical_accuracy: 0.8649
12224/60000 [=====>........................] - ETA: 1:10 - loss: 0.4284 - categorical_accuracy: 0.8651
12288/60000 [=====>........................] - ETA: 1:10 - loss: 0.4267 - categorical_accuracy: 0.8656
12352/60000 [=====>........................] - ETA: 1:10 - loss: 0.4259 - categorical_accuracy: 0.8658
12416/60000 [=====>........................] - ETA: 1:10 - loss: 0.4241 - categorical_accuracy: 0.8663
12480/60000 [=====>........................] - ETA: 1:10 - loss: 0.4235 - categorical_accuracy: 0.8666
12544/60000 [=====>........................] - ETA: 1:10 - loss: 0.4216 - categorical_accuracy: 0.8672
12608/60000 [=====>........................] - ETA: 1:10 - loss: 0.4208 - categorical_accuracy: 0.8675
12672/60000 [=====>........................] - ETA: 1:10 - loss: 0.4195 - categorical_accuracy: 0.8677
12704/60000 [=====>........................] - ETA: 1:10 - loss: 0.4187 - categorical_accuracy: 0.8681
12768/60000 [=====>........................] - ETA: 1:10 - loss: 0.4178 - categorical_accuracy: 0.8682
12832/60000 [=====>........................] - ETA: 1:09 - loss: 0.4165 - categorical_accuracy: 0.8686
12896/60000 [=====>........................] - ETA: 1:09 - loss: 0.4153 - categorical_accuracy: 0.8691
12928/60000 [=====>........................] - ETA: 1:09 - loss: 0.4145 - categorical_accuracy: 0.8694
12992/60000 [=====>........................] - ETA: 1:09 - loss: 0.4131 - categorical_accuracy: 0.8698
13056/60000 [=====>........................] - ETA: 1:09 - loss: 0.4115 - categorical_accuracy: 0.8704
13120/60000 [=====>........................] - ETA: 1:09 - loss: 0.4101 - categorical_accuracy: 0.8708
13184/60000 [=====>........................] - ETA: 1:09 - loss: 0.4088 - categorical_accuracy: 0.8712
13248/60000 [=====>........................] - ETA: 1:09 - loss: 0.4073 - categorical_accuracy: 0.8717
13312/60000 [=====>........................] - ETA: 1:09 - loss: 0.4073 - categorical_accuracy: 0.8718
13376/60000 [=====>........................] - ETA: 1:09 - loss: 0.4069 - categorical_accuracy: 0.8721
13408/60000 [=====>........................] - ETA: 1:09 - loss: 0.4066 - categorical_accuracy: 0.8720
13472/60000 [=====>........................] - ETA: 1:09 - loss: 0.4053 - categorical_accuracy: 0.8723
13536/60000 [=====>........................] - ETA: 1:08 - loss: 0.4039 - categorical_accuracy: 0.8727
13600/60000 [=====>........................] - ETA: 1:08 - loss: 0.4027 - categorical_accuracy: 0.8729
13664/60000 [=====>........................] - ETA: 1:08 - loss: 0.4010 - categorical_accuracy: 0.8735
13728/60000 [=====>........................] - ETA: 1:08 - loss: 0.4001 - categorical_accuracy: 0.8738
13792/60000 [=====>........................] - ETA: 1:08 - loss: 0.3988 - categorical_accuracy: 0.8741
13856/60000 [=====>........................] - ETA: 1:08 - loss: 0.3974 - categorical_accuracy: 0.8746
13920/60000 [=====>........................] - ETA: 1:08 - loss: 0.3962 - categorical_accuracy: 0.8749
13984/60000 [=====>........................] - ETA: 1:08 - loss: 0.3947 - categorical_accuracy: 0.8754
14048/60000 [======>.......................] - ETA: 1:08 - loss: 0.3933 - categorical_accuracy: 0.8758
14112/60000 [======>.......................] - ETA: 1:07 - loss: 0.3921 - categorical_accuracy: 0.8762
14176/60000 [======>.......................] - ETA: 1:07 - loss: 0.3911 - categorical_accuracy: 0.8765
14240/60000 [======>.......................] - ETA: 1:07 - loss: 0.3897 - categorical_accuracy: 0.8768
14304/60000 [======>.......................] - ETA: 1:07 - loss: 0.3890 - categorical_accuracy: 0.8770
14368/60000 [======>.......................] - ETA: 1:07 - loss: 0.3882 - categorical_accuracy: 0.8773
14400/60000 [======>.......................] - ETA: 1:07 - loss: 0.3875 - categorical_accuracy: 0.8775
14464/60000 [======>.......................] - ETA: 1:07 - loss: 0.3867 - categorical_accuracy: 0.8777
14528/60000 [======>.......................] - ETA: 1:07 - loss: 0.3857 - categorical_accuracy: 0.8780
14592/60000 [======>.......................] - ETA: 1:07 - loss: 0.3853 - categorical_accuracy: 0.8782
14656/60000 [======>.......................] - ETA: 1:07 - loss: 0.3849 - categorical_accuracy: 0.8785
14720/60000 [======>.......................] - ETA: 1:06 - loss: 0.3839 - categorical_accuracy: 0.8787
14784/60000 [======>.......................] - ETA: 1:06 - loss: 0.3828 - categorical_accuracy: 0.8790
14848/60000 [======>.......................] - ETA: 1:06 - loss: 0.3823 - categorical_accuracy: 0.8791
14912/60000 [======>.......................] - ETA: 1:06 - loss: 0.3814 - categorical_accuracy: 0.8793
14976/60000 [======>.......................] - ETA: 1:06 - loss: 0.3803 - categorical_accuracy: 0.8797
15040/60000 [======>.......................] - ETA: 1:06 - loss: 0.3799 - categorical_accuracy: 0.8799
15104/60000 [======>.......................] - ETA: 1:06 - loss: 0.3791 - categorical_accuracy: 0.8801
15168/60000 [======>.......................] - ETA: 1:06 - loss: 0.3782 - categorical_accuracy: 0.8804
15232/60000 [======>.......................] - ETA: 1:06 - loss: 0.3776 - categorical_accuracy: 0.8807
15296/60000 [======>.......................] - ETA: 1:06 - loss: 0.3764 - categorical_accuracy: 0.8811
15360/60000 [======>.......................] - ETA: 1:05 - loss: 0.3750 - categorical_accuracy: 0.8816
15424/60000 [======>.......................] - ETA: 1:05 - loss: 0.3739 - categorical_accuracy: 0.8819
15488/60000 [======>.......................] - ETA: 1:05 - loss: 0.3734 - categorical_accuracy: 0.8820
15552/60000 [======>.......................] - ETA: 1:05 - loss: 0.3731 - categorical_accuracy: 0.8821
15616/60000 [======>.......................] - ETA: 1:05 - loss: 0.3728 - categorical_accuracy: 0.8823
15680/60000 [======>.......................] - ETA: 1:05 - loss: 0.3719 - categorical_accuracy: 0.8826
15744/60000 [======>.......................] - ETA: 1:05 - loss: 0.3706 - categorical_accuracy: 0.8830
15808/60000 [======>.......................] - ETA: 1:05 - loss: 0.3694 - categorical_accuracy: 0.8834
15872/60000 [======>.......................] - ETA: 1:05 - loss: 0.3686 - categorical_accuracy: 0.8836
15936/60000 [======>.......................] - ETA: 1:05 - loss: 0.3676 - categorical_accuracy: 0.8839
15968/60000 [======>.......................] - ETA: 1:05 - loss: 0.3671 - categorical_accuracy: 0.8841
16032/60000 [=======>......................] - ETA: 1:04 - loss: 0.3659 - categorical_accuracy: 0.8844
16096/60000 [=======>......................] - ETA: 1:04 - loss: 0.3648 - categorical_accuracy: 0.8848
16160/60000 [=======>......................] - ETA: 1:04 - loss: 0.3641 - categorical_accuracy: 0.8850
16224/60000 [=======>......................] - ETA: 1:04 - loss: 0.3639 - categorical_accuracy: 0.8852
16288/60000 [=======>......................] - ETA: 1:04 - loss: 0.3628 - categorical_accuracy: 0.8855
16352/60000 [=======>......................] - ETA: 1:04 - loss: 0.3616 - categorical_accuracy: 0.8859
16416/60000 [=======>......................] - ETA: 1:04 - loss: 0.3608 - categorical_accuracy: 0.8860
16480/60000 [=======>......................] - ETA: 1:04 - loss: 0.3600 - categorical_accuracy: 0.8863
16544/60000 [=======>......................] - ETA: 1:04 - loss: 0.3593 - categorical_accuracy: 0.8864
16608/60000 [=======>......................] - ETA: 1:04 - loss: 0.3592 - categorical_accuracy: 0.8864
16672/60000 [=======>......................] - ETA: 1:03 - loss: 0.3583 - categorical_accuracy: 0.8868
16736/60000 [=======>......................] - ETA: 1:03 - loss: 0.3581 - categorical_accuracy: 0.8868
16800/60000 [=======>......................] - ETA: 1:03 - loss: 0.3578 - categorical_accuracy: 0.8870
16864/60000 [=======>......................] - ETA: 1:03 - loss: 0.3568 - categorical_accuracy: 0.8873
16928/60000 [=======>......................] - ETA: 1:03 - loss: 0.3559 - categorical_accuracy: 0.8875
16992/60000 [=======>......................] - ETA: 1:03 - loss: 0.3560 - categorical_accuracy: 0.8875
17056/60000 [=======>......................] - ETA: 1:03 - loss: 0.3552 - categorical_accuracy: 0.8877
17120/60000 [=======>......................] - ETA: 1:03 - loss: 0.3543 - categorical_accuracy: 0.8880
17184/60000 [=======>......................] - ETA: 1:03 - loss: 0.3535 - categorical_accuracy: 0.8883
17248/60000 [=======>......................] - ETA: 1:03 - loss: 0.3526 - categorical_accuracy: 0.8885
17312/60000 [=======>......................] - ETA: 1:02 - loss: 0.3516 - categorical_accuracy: 0.8889
17376/60000 [=======>......................] - ETA: 1:02 - loss: 0.3511 - categorical_accuracy: 0.8891
17440/60000 [=======>......................] - ETA: 1:02 - loss: 0.3504 - categorical_accuracy: 0.8893
17504/60000 [=======>......................] - ETA: 1:02 - loss: 0.3497 - categorical_accuracy: 0.8896
17568/60000 [=======>......................] - ETA: 1:02 - loss: 0.3488 - categorical_accuracy: 0.8899
17632/60000 [=======>......................] - ETA: 1:02 - loss: 0.3478 - categorical_accuracy: 0.8902
17696/60000 [=======>......................] - ETA: 1:02 - loss: 0.3468 - categorical_accuracy: 0.8905
17760/60000 [=======>......................] - ETA: 1:02 - loss: 0.3457 - categorical_accuracy: 0.8909
17824/60000 [=======>......................] - ETA: 1:02 - loss: 0.3451 - categorical_accuracy: 0.8910
17888/60000 [=======>......................] - ETA: 1:02 - loss: 0.3442 - categorical_accuracy: 0.8913
17952/60000 [=======>......................] - ETA: 1:01 - loss: 0.3432 - categorical_accuracy: 0.8917
18016/60000 [========>.....................] - ETA: 1:01 - loss: 0.3422 - categorical_accuracy: 0.8920
18080/60000 [========>.....................] - ETA: 1:01 - loss: 0.3411 - categorical_accuracy: 0.8924
18144/60000 [========>.....................] - ETA: 1:01 - loss: 0.3407 - categorical_accuracy: 0.8925
18208/60000 [========>.....................] - ETA: 1:01 - loss: 0.3406 - categorical_accuracy: 0.8925
18272/60000 [========>.....................] - ETA: 1:01 - loss: 0.3396 - categorical_accuracy: 0.8928
18336/60000 [========>.....................] - ETA: 1:01 - loss: 0.3387 - categorical_accuracy: 0.8931
18400/60000 [========>.....................] - ETA: 1:01 - loss: 0.3384 - categorical_accuracy: 0.8934
18464/60000 [========>.....................] - ETA: 1:01 - loss: 0.3379 - categorical_accuracy: 0.8936
18528/60000 [========>.....................] - ETA: 1:01 - loss: 0.3370 - categorical_accuracy: 0.8939
18592/60000 [========>.....................] - ETA: 1:00 - loss: 0.3367 - categorical_accuracy: 0.8940
18656/60000 [========>.....................] - ETA: 1:00 - loss: 0.3358 - categorical_accuracy: 0.8944
18688/60000 [========>.....................] - ETA: 1:00 - loss: 0.3353 - categorical_accuracy: 0.8945
18752/60000 [========>.....................] - ETA: 1:00 - loss: 0.3345 - categorical_accuracy: 0.8948
18816/60000 [========>.....................] - ETA: 1:00 - loss: 0.3336 - categorical_accuracy: 0.8951
18880/60000 [========>.....................] - ETA: 1:00 - loss: 0.3327 - categorical_accuracy: 0.8953
18944/60000 [========>.....................] - ETA: 1:00 - loss: 0.3318 - categorical_accuracy: 0.8957
19008/60000 [========>.....................] - ETA: 1:00 - loss: 0.3311 - categorical_accuracy: 0.8960
19040/60000 [========>.....................] - ETA: 1:00 - loss: 0.3307 - categorical_accuracy: 0.8961
19072/60000 [========>.....................] - ETA: 1:00 - loss: 0.3304 - categorical_accuracy: 0.8961
19136/60000 [========>.....................] - ETA: 1:00 - loss: 0.3299 - categorical_accuracy: 0.8963
19168/60000 [========>.....................] - ETA: 1:00 - loss: 0.3295 - categorical_accuracy: 0.8963
19232/60000 [========>.....................] - ETA: 1:00 - loss: 0.3288 - categorical_accuracy: 0.8965
19296/60000 [========>.....................] - ETA: 59s - loss: 0.3286 - categorical_accuracy: 0.8966 
19360/60000 [========>.....................] - ETA: 59s - loss: 0.3278 - categorical_accuracy: 0.8968
19424/60000 [========>.....................] - ETA: 59s - loss: 0.3273 - categorical_accuracy: 0.8970
19488/60000 [========>.....................] - ETA: 59s - loss: 0.3269 - categorical_accuracy: 0.8972
19552/60000 [========>.....................] - ETA: 59s - loss: 0.3267 - categorical_accuracy: 0.8972
19616/60000 [========>.....................] - ETA: 59s - loss: 0.3264 - categorical_accuracy: 0.8974
19680/60000 [========>.....................] - ETA: 59s - loss: 0.3260 - categorical_accuracy: 0.8975
19744/60000 [========>.....................] - ETA: 59s - loss: 0.3254 - categorical_accuracy: 0.8976
19808/60000 [========>.....................] - ETA: 59s - loss: 0.3250 - categorical_accuracy: 0.8977
19872/60000 [========>.....................] - ETA: 59s - loss: 0.3243 - categorical_accuracy: 0.8978
19936/60000 [========>.....................] - ETA: 58s - loss: 0.3234 - categorical_accuracy: 0.8982
20000/60000 [=========>....................] - ETA: 58s - loss: 0.3230 - categorical_accuracy: 0.8982
20064/60000 [=========>....................] - ETA: 58s - loss: 0.3222 - categorical_accuracy: 0.8985
20128/60000 [=========>....................] - ETA: 58s - loss: 0.3216 - categorical_accuracy: 0.8987
20192/60000 [=========>....................] - ETA: 58s - loss: 0.3209 - categorical_accuracy: 0.8989
20256/60000 [=========>....................] - ETA: 58s - loss: 0.3203 - categorical_accuracy: 0.8991
20320/60000 [=========>....................] - ETA: 58s - loss: 0.3196 - categorical_accuracy: 0.8993
20384/60000 [=========>....................] - ETA: 58s - loss: 0.3192 - categorical_accuracy: 0.8995
20448/60000 [=========>....................] - ETA: 58s - loss: 0.3186 - categorical_accuracy: 0.8996
20512/60000 [=========>....................] - ETA: 58s - loss: 0.3180 - categorical_accuracy: 0.8998
20576/60000 [=========>....................] - ETA: 57s - loss: 0.3173 - categorical_accuracy: 0.9000
20640/60000 [=========>....................] - ETA: 57s - loss: 0.3168 - categorical_accuracy: 0.9001
20672/60000 [=========>....................] - ETA: 57s - loss: 0.3164 - categorical_accuracy: 0.9003
20704/60000 [=========>....................] - ETA: 57s - loss: 0.3162 - categorical_accuracy: 0.9003
20768/60000 [=========>....................] - ETA: 57s - loss: 0.3154 - categorical_accuracy: 0.9006
20832/60000 [=========>....................] - ETA: 57s - loss: 0.3154 - categorical_accuracy: 0.9006
20896/60000 [=========>....................] - ETA: 57s - loss: 0.3147 - categorical_accuracy: 0.9008
20960/60000 [=========>....................] - ETA: 57s - loss: 0.3142 - categorical_accuracy: 0.9010
21024/60000 [=========>....................] - ETA: 57s - loss: 0.3137 - categorical_accuracy: 0.9011
21088/60000 [=========>....................] - ETA: 57s - loss: 0.3130 - categorical_accuracy: 0.9013
21152/60000 [=========>....................] - ETA: 57s - loss: 0.3125 - categorical_accuracy: 0.9014
21216/60000 [=========>....................] - ETA: 56s - loss: 0.3117 - categorical_accuracy: 0.9017
21280/60000 [=========>....................] - ETA: 56s - loss: 0.3110 - categorical_accuracy: 0.9019
21344/60000 [=========>....................] - ETA: 56s - loss: 0.3102 - categorical_accuracy: 0.9022
21408/60000 [=========>....................] - ETA: 56s - loss: 0.3104 - categorical_accuracy: 0.9021
21472/60000 [=========>....................] - ETA: 56s - loss: 0.3098 - categorical_accuracy: 0.9023
21536/60000 [=========>....................] - ETA: 56s - loss: 0.3093 - categorical_accuracy: 0.9025
21600/60000 [=========>....................] - ETA: 56s - loss: 0.3088 - categorical_accuracy: 0.9026
21664/60000 [=========>....................] - ETA: 56s - loss: 0.3087 - categorical_accuracy: 0.9027
21728/60000 [=========>....................] - ETA: 56s - loss: 0.3079 - categorical_accuracy: 0.9030
21792/60000 [=========>....................] - ETA: 56s - loss: 0.3081 - categorical_accuracy: 0.9030
21856/60000 [=========>....................] - ETA: 56s - loss: 0.3076 - categorical_accuracy: 0.9031
21920/60000 [=========>....................] - ETA: 55s - loss: 0.3071 - categorical_accuracy: 0.9032
21952/60000 [=========>....................] - ETA: 55s - loss: 0.3069 - categorical_accuracy: 0.9033
22016/60000 [==========>...................] - ETA: 55s - loss: 0.3065 - categorical_accuracy: 0.9034
22080/60000 [==========>...................] - ETA: 55s - loss: 0.3060 - categorical_accuracy: 0.9036
22144/60000 [==========>...................] - ETA: 55s - loss: 0.3056 - categorical_accuracy: 0.9038
22208/60000 [==========>...................] - ETA: 55s - loss: 0.3054 - categorical_accuracy: 0.9038
22272/60000 [==========>...................] - ETA: 55s - loss: 0.3049 - categorical_accuracy: 0.9040
22336/60000 [==========>...................] - ETA: 55s - loss: 0.3042 - categorical_accuracy: 0.9042
22368/60000 [==========>...................] - ETA: 55s - loss: 0.3039 - categorical_accuracy: 0.9043
22432/60000 [==========>...................] - ETA: 55s - loss: 0.3032 - categorical_accuracy: 0.9045
22496/60000 [==========>...................] - ETA: 55s - loss: 0.3031 - categorical_accuracy: 0.9046
22560/60000 [==========>...................] - ETA: 55s - loss: 0.3024 - categorical_accuracy: 0.9048
22624/60000 [==========>...................] - ETA: 54s - loss: 0.3017 - categorical_accuracy: 0.9050
22688/60000 [==========>...................] - ETA: 54s - loss: 0.3011 - categorical_accuracy: 0.9052
22720/60000 [==========>...................] - ETA: 54s - loss: 0.3008 - categorical_accuracy: 0.9053
22784/60000 [==========>...................] - ETA: 54s - loss: 0.3006 - categorical_accuracy: 0.9055
22848/60000 [==========>...................] - ETA: 54s - loss: 0.2998 - categorical_accuracy: 0.9058
22912/60000 [==========>...................] - ETA: 54s - loss: 0.2994 - categorical_accuracy: 0.9059
22976/60000 [==========>...................] - ETA: 54s - loss: 0.2988 - categorical_accuracy: 0.9061
23040/60000 [==========>...................] - ETA: 54s - loss: 0.2982 - categorical_accuracy: 0.9062
23072/60000 [==========>...................] - ETA: 54s - loss: 0.2978 - categorical_accuracy: 0.9064
23136/60000 [==========>...................] - ETA: 54s - loss: 0.2978 - categorical_accuracy: 0.9065
23200/60000 [==========>...................] - ETA: 54s - loss: 0.2971 - categorical_accuracy: 0.9066
23264/60000 [==========>...................] - ETA: 53s - loss: 0.2964 - categorical_accuracy: 0.9069
23328/60000 [==========>...................] - ETA: 53s - loss: 0.2957 - categorical_accuracy: 0.9070
23392/60000 [==========>...................] - ETA: 53s - loss: 0.2952 - categorical_accuracy: 0.9071
23456/60000 [==========>...................] - ETA: 53s - loss: 0.2949 - categorical_accuracy: 0.9073
23520/60000 [==========>...................] - ETA: 53s - loss: 0.2944 - categorical_accuracy: 0.9074
23552/60000 [==========>...................] - ETA: 53s - loss: 0.2941 - categorical_accuracy: 0.9075
23616/60000 [==========>...................] - ETA: 53s - loss: 0.2937 - categorical_accuracy: 0.9076
23680/60000 [==========>...................] - ETA: 53s - loss: 0.2932 - categorical_accuracy: 0.9078
23744/60000 [==========>...................] - ETA: 53s - loss: 0.2927 - categorical_accuracy: 0.9080
23808/60000 [==========>...................] - ETA: 53s - loss: 0.2922 - categorical_accuracy: 0.9081
23872/60000 [==========>...................] - ETA: 53s - loss: 0.2920 - categorical_accuracy: 0.9082
23936/60000 [==========>...................] - ETA: 52s - loss: 0.2917 - categorical_accuracy: 0.9082
24000/60000 [===========>..................] - ETA: 52s - loss: 0.2912 - categorical_accuracy: 0.9084
24064/60000 [===========>..................] - ETA: 52s - loss: 0.2907 - categorical_accuracy: 0.9086
24128/60000 [===========>..................] - ETA: 52s - loss: 0.2902 - categorical_accuracy: 0.9087
24192/60000 [===========>..................] - ETA: 52s - loss: 0.2897 - categorical_accuracy: 0.9089
24224/60000 [===========>..................] - ETA: 52s - loss: 0.2897 - categorical_accuracy: 0.9089
24256/60000 [===========>..................] - ETA: 52s - loss: 0.2896 - categorical_accuracy: 0.9089
24288/60000 [===========>..................] - ETA: 52s - loss: 0.2892 - categorical_accuracy: 0.9090
24352/60000 [===========>..................] - ETA: 52s - loss: 0.2888 - categorical_accuracy: 0.9092
24416/60000 [===========>..................] - ETA: 52s - loss: 0.2883 - categorical_accuracy: 0.9093
24480/60000 [===========>..................] - ETA: 52s - loss: 0.2880 - categorical_accuracy: 0.9094
24544/60000 [===========>..................] - ETA: 52s - loss: 0.2875 - categorical_accuracy: 0.9096
24608/60000 [===========>..................] - ETA: 52s - loss: 0.2872 - categorical_accuracy: 0.9097
24672/60000 [===========>..................] - ETA: 51s - loss: 0.2870 - categorical_accuracy: 0.9099
24736/60000 [===========>..................] - ETA: 51s - loss: 0.2865 - categorical_accuracy: 0.9100
24800/60000 [===========>..................] - ETA: 51s - loss: 0.2860 - categorical_accuracy: 0.9102
24864/60000 [===========>..................] - ETA: 51s - loss: 0.2854 - categorical_accuracy: 0.9104
24928/60000 [===========>..................] - ETA: 51s - loss: 0.2850 - categorical_accuracy: 0.9105
24992/60000 [===========>..................] - ETA: 51s - loss: 0.2847 - categorical_accuracy: 0.9106
25056/60000 [===========>..................] - ETA: 51s - loss: 0.2848 - categorical_accuracy: 0.9107
25120/60000 [===========>..................] - ETA: 51s - loss: 0.2843 - categorical_accuracy: 0.9109
25184/60000 [===========>..................] - ETA: 51s - loss: 0.2840 - categorical_accuracy: 0.9110
25248/60000 [===========>..................] - ETA: 51s - loss: 0.2834 - categorical_accuracy: 0.9112
25312/60000 [===========>..................] - ETA: 50s - loss: 0.2830 - categorical_accuracy: 0.9113
25376/60000 [===========>..................] - ETA: 50s - loss: 0.2826 - categorical_accuracy: 0.9115
25440/60000 [===========>..................] - ETA: 50s - loss: 0.2823 - categorical_accuracy: 0.9116
25504/60000 [===========>..................] - ETA: 50s - loss: 0.2820 - categorical_accuracy: 0.9117
25568/60000 [===========>..................] - ETA: 50s - loss: 0.2816 - categorical_accuracy: 0.9118
25632/60000 [===========>..................] - ETA: 50s - loss: 0.2811 - categorical_accuracy: 0.9120
25696/60000 [===========>..................] - ETA: 50s - loss: 0.2807 - categorical_accuracy: 0.9121
25728/60000 [===========>..................] - ETA: 50s - loss: 0.2805 - categorical_accuracy: 0.9122
25760/60000 [===========>..................] - ETA: 50s - loss: 0.2802 - categorical_accuracy: 0.9123
25824/60000 [===========>..................] - ETA: 50s - loss: 0.2797 - categorical_accuracy: 0.9124
25888/60000 [===========>..................] - ETA: 50s - loss: 0.2792 - categorical_accuracy: 0.9125
25952/60000 [===========>..................] - ETA: 49s - loss: 0.2788 - categorical_accuracy: 0.9126
26016/60000 [============>.................] - ETA: 49s - loss: 0.2784 - categorical_accuracy: 0.9127
26080/60000 [============>.................] - ETA: 49s - loss: 0.2778 - categorical_accuracy: 0.9129
26144/60000 [============>.................] - ETA: 49s - loss: 0.2776 - categorical_accuracy: 0.9129
26208/60000 [============>.................] - ETA: 49s - loss: 0.2777 - categorical_accuracy: 0.9130
26272/60000 [============>.................] - ETA: 49s - loss: 0.2771 - categorical_accuracy: 0.9131
26336/60000 [============>.................] - ETA: 49s - loss: 0.2773 - categorical_accuracy: 0.9132
26400/60000 [============>.................] - ETA: 49s - loss: 0.2775 - categorical_accuracy: 0.9132
26464/60000 [============>.................] - ETA: 49s - loss: 0.2776 - categorical_accuracy: 0.9133
26528/60000 [============>.................] - ETA: 49s - loss: 0.2776 - categorical_accuracy: 0.9133
26592/60000 [============>.................] - ETA: 49s - loss: 0.2770 - categorical_accuracy: 0.9135
26656/60000 [============>.................] - ETA: 48s - loss: 0.2765 - categorical_accuracy: 0.9136
26720/60000 [============>.................] - ETA: 48s - loss: 0.2763 - categorical_accuracy: 0.9137
26784/60000 [============>.................] - ETA: 48s - loss: 0.2759 - categorical_accuracy: 0.9138
26848/60000 [============>.................] - ETA: 48s - loss: 0.2755 - categorical_accuracy: 0.9139
26912/60000 [============>.................] - ETA: 48s - loss: 0.2751 - categorical_accuracy: 0.9140
26976/60000 [============>.................] - ETA: 48s - loss: 0.2749 - categorical_accuracy: 0.9140
27040/60000 [============>.................] - ETA: 48s - loss: 0.2747 - categorical_accuracy: 0.9141
27104/60000 [============>.................] - ETA: 48s - loss: 0.2743 - categorical_accuracy: 0.9143
27168/60000 [============>.................] - ETA: 48s - loss: 0.2739 - categorical_accuracy: 0.9144
27232/60000 [============>.................] - ETA: 48s - loss: 0.2734 - categorical_accuracy: 0.9145
27296/60000 [============>.................] - ETA: 47s - loss: 0.2731 - categorical_accuracy: 0.9146
27360/60000 [============>.................] - ETA: 47s - loss: 0.2726 - categorical_accuracy: 0.9147
27424/60000 [============>.................] - ETA: 47s - loss: 0.2723 - categorical_accuracy: 0.9148
27488/60000 [============>.................] - ETA: 47s - loss: 0.2719 - categorical_accuracy: 0.9149
27552/60000 [============>.................] - ETA: 47s - loss: 0.2715 - categorical_accuracy: 0.9151
27584/60000 [============>.................] - ETA: 47s - loss: 0.2714 - categorical_accuracy: 0.9151
27648/60000 [============>.................] - ETA: 47s - loss: 0.2710 - categorical_accuracy: 0.9152
27712/60000 [============>.................] - ETA: 47s - loss: 0.2707 - categorical_accuracy: 0.9152
27776/60000 [============>.................] - ETA: 47s - loss: 0.2704 - categorical_accuracy: 0.9153
27840/60000 [============>.................] - ETA: 47s - loss: 0.2699 - categorical_accuracy: 0.9154
27904/60000 [============>.................] - ETA: 47s - loss: 0.2696 - categorical_accuracy: 0.9156
27968/60000 [============>.................] - ETA: 46s - loss: 0.2692 - categorical_accuracy: 0.9157
28032/60000 [=============>................] - ETA: 46s - loss: 0.2690 - categorical_accuracy: 0.9158
28096/60000 [=============>................] - ETA: 46s - loss: 0.2686 - categorical_accuracy: 0.9159
28160/60000 [=============>................] - ETA: 46s - loss: 0.2683 - categorical_accuracy: 0.9161
28224/60000 [=============>................] - ETA: 46s - loss: 0.2681 - categorical_accuracy: 0.9160
28256/60000 [=============>................] - ETA: 46s - loss: 0.2680 - categorical_accuracy: 0.9161
28320/60000 [=============>................] - ETA: 46s - loss: 0.2677 - categorical_accuracy: 0.9162
28384/60000 [=============>................] - ETA: 46s - loss: 0.2673 - categorical_accuracy: 0.9163
28448/60000 [=============>................] - ETA: 46s - loss: 0.2667 - categorical_accuracy: 0.9165
28512/60000 [=============>................] - ETA: 46s - loss: 0.2664 - categorical_accuracy: 0.9166
28576/60000 [=============>................] - ETA: 46s - loss: 0.2660 - categorical_accuracy: 0.9167
28640/60000 [=============>................] - ETA: 45s - loss: 0.2655 - categorical_accuracy: 0.9169
28704/60000 [=============>................] - ETA: 45s - loss: 0.2652 - categorical_accuracy: 0.9170
28768/60000 [=============>................] - ETA: 45s - loss: 0.2650 - categorical_accuracy: 0.9171
28832/60000 [=============>................] - ETA: 45s - loss: 0.2647 - categorical_accuracy: 0.9172
28896/60000 [=============>................] - ETA: 45s - loss: 0.2642 - categorical_accuracy: 0.9174
28960/60000 [=============>................] - ETA: 45s - loss: 0.2643 - categorical_accuracy: 0.9174
29024/60000 [=============>................] - ETA: 45s - loss: 0.2641 - categorical_accuracy: 0.9175
29088/60000 [=============>................] - ETA: 45s - loss: 0.2641 - categorical_accuracy: 0.9176
29152/60000 [=============>................] - ETA: 45s - loss: 0.2636 - categorical_accuracy: 0.9177
29216/60000 [=============>................] - ETA: 45s - loss: 0.2633 - categorical_accuracy: 0.9178
29280/60000 [=============>................] - ETA: 44s - loss: 0.2628 - categorical_accuracy: 0.9179
29344/60000 [=============>................] - ETA: 44s - loss: 0.2625 - categorical_accuracy: 0.9181
29408/60000 [=============>................] - ETA: 44s - loss: 0.2620 - categorical_accuracy: 0.9182
29472/60000 [=============>................] - ETA: 44s - loss: 0.2617 - categorical_accuracy: 0.9183
29536/60000 [=============>................] - ETA: 44s - loss: 0.2615 - categorical_accuracy: 0.9184
29600/60000 [=============>................] - ETA: 44s - loss: 0.2611 - categorical_accuracy: 0.9185
29664/60000 [=============>................] - ETA: 44s - loss: 0.2609 - categorical_accuracy: 0.9186
29728/60000 [=============>................] - ETA: 44s - loss: 0.2606 - categorical_accuracy: 0.9186
29792/60000 [=============>................] - ETA: 44s - loss: 0.2606 - categorical_accuracy: 0.9186
29856/60000 [=============>................] - ETA: 44s - loss: 0.2607 - categorical_accuracy: 0.9186
29920/60000 [=============>................] - ETA: 44s - loss: 0.2602 - categorical_accuracy: 0.9188
29984/60000 [=============>................] - ETA: 43s - loss: 0.2598 - categorical_accuracy: 0.9189
30048/60000 [==============>...............] - ETA: 43s - loss: 0.2594 - categorical_accuracy: 0.9190
30112/60000 [==============>...............] - ETA: 43s - loss: 0.2592 - categorical_accuracy: 0.9191
30176/60000 [==============>...............] - ETA: 43s - loss: 0.2592 - categorical_accuracy: 0.9192
30240/60000 [==============>...............] - ETA: 43s - loss: 0.2588 - categorical_accuracy: 0.9193
30304/60000 [==============>...............] - ETA: 43s - loss: 0.2590 - categorical_accuracy: 0.9192
30368/60000 [==============>...............] - ETA: 43s - loss: 0.2588 - categorical_accuracy: 0.9192
30400/60000 [==============>...............] - ETA: 43s - loss: 0.2586 - categorical_accuracy: 0.9193
30464/60000 [==============>...............] - ETA: 43s - loss: 0.2588 - categorical_accuracy: 0.9193
30528/60000 [==============>...............] - ETA: 43s - loss: 0.2585 - categorical_accuracy: 0.9194
30592/60000 [==============>...............] - ETA: 43s - loss: 0.2582 - categorical_accuracy: 0.9195
30656/60000 [==============>...............] - ETA: 42s - loss: 0.2581 - categorical_accuracy: 0.9195
30720/60000 [==============>...............] - ETA: 42s - loss: 0.2578 - categorical_accuracy: 0.9195
30784/60000 [==============>...............] - ETA: 42s - loss: 0.2574 - categorical_accuracy: 0.9197
30816/60000 [==============>...............] - ETA: 42s - loss: 0.2573 - categorical_accuracy: 0.9197
30880/60000 [==============>...............] - ETA: 42s - loss: 0.2568 - categorical_accuracy: 0.9199
30944/60000 [==============>...............] - ETA: 42s - loss: 0.2567 - categorical_accuracy: 0.9200
31008/60000 [==============>...............] - ETA: 42s - loss: 0.2566 - categorical_accuracy: 0.9201
31072/60000 [==============>...............] - ETA: 42s - loss: 0.2564 - categorical_accuracy: 0.9202
31136/60000 [==============>...............] - ETA: 42s - loss: 0.2561 - categorical_accuracy: 0.9202
31200/60000 [==============>...............] - ETA: 42s - loss: 0.2560 - categorical_accuracy: 0.9204
31232/60000 [==============>...............] - ETA: 42s - loss: 0.2560 - categorical_accuracy: 0.9203
31296/60000 [==============>...............] - ETA: 41s - loss: 0.2557 - categorical_accuracy: 0.9204
31360/60000 [==============>...............] - ETA: 41s - loss: 0.2554 - categorical_accuracy: 0.9205
31424/60000 [==============>...............] - ETA: 41s - loss: 0.2551 - categorical_accuracy: 0.9206
31488/60000 [==============>...............] - ETA: 41s - loss: 0.2551 - categorical_accuracy: 0.9206
31552/60000 [==============>...............] - ETA: 41s - loss: 0.2549 - categorical_accuracy: 0.9207
31616/60000 [==============>...............] - ETA: 41s - loss: 0.2545 - categorical_accuracy: 0.9209
31680/60000 [==============>...............] - ETA: 41s - loss: 0.2542 - categorical_accuracy: 0.9209
31744/60000 [==============>...............] - ETA: 41s - loss: 0.2542 - categorical_accuracy: 0.9210
31808/60000 [==============>...............] - ETA: 41s - loss: 0.2539 - categorical_accuracy: 0.9211
31872/60000 [==============>...............] - ETA: 41s - loss: 0.2538 - categorical_accuracy: 0.9211
31936/60000 [==============>...............] - ETA: 41s - loss: 0.2537 - categorical_accuracy: 0.9211
32000/60000 [===============>..............] - ETA: 40s - loss: 0.2533 - categorical_accuracy: 0.9212
32064/60000 [===============>..............] - ETA: 40s - loss: 0.2531 - categorical_accuracy: 0.9213
32128/60000 [===============>..............] - ETA: 40s - loss: 0.2527 - categorical_accuracy: 0.9215
32192/60000 [===============>..............] - ETA: 40s - loss: 0.2525 - categorical_accuracy: 0.9215
32256/60000 [===============>..............] - ETA: 40s - loss: 0.2522 - categorical_accuracy: 0.9216
32320/60000 [===============>..............] - ETA: 40s - loss: 0.2518 - categorical_accuracy: 0.9218
32384/60000 [===============>..............] - ETA: 40s - loss: 0.2514 - categorical_accuracy: 0.9219
32448/60000 [===============>..............] - ETA: 40s - loss: 0.2513 - categorical_accuracy: 0.9220
32512/60000 [===============>..............] - ETA: 40s - loss: 0.2509 - categorical_accuracy: 0.9220
32544/60000 [===============>..............] - ETA: 40s - loss: 0.2509 - categorical_accuracy: 0.9220
32608/60000 [===============>..............] - ETA: 40s - loss: 0.2508 - categorical_accuracy: 0.9221
32672/60000 [===============>..............] - ETA: 39s - loss: 0.2506 - categorical_accuracy: 0.9221
32736/60000 [===============>..............] - ETA: 39s - loss: 0.2504 - categorical_accuracy: 0.9222
32800/60000 [===============>..............] - ETA: 39s - loss: 0.2499 - categorical_accuracy: 0.9224
32864/60000 [===============>..............] - ETA: 39s - loss: 0.2495 - categorical_accuracy: 0.9225
32928/60000 [===============>..............] - ETA: 39s - loss: 0.2493 - categorical_accuracy: 0.9226
32992/60000 [===============>..............] - ETA: 39s - loss: 0.2489 - categorical_accuracy: 0.9228
33056/60000 [===============>..............] - ETA: 39s - loss: 0.2485 - categorical_accuracy: 0.9229
33120/60000 [===============>..............] - ETA: 39s - loss: 0.2483 - categorical_accuracy: 0.9229
33184/60000 [===============>..............] - ETA: 39s - loss: 0.2481 - categorical_accuracy: 0.9230
33248/60000 [===============>..............] - ETA: 39s - loss: 0.2477 - categorical_accuracy: 0.9231
33312/60000 [===============>..............] - ETA: 38s - loss: 0.2474 - categorical_accuracy: 0.9232
33376/60000 [===============>..............] - ETA: 38s - loss: 0.2473 - categorical_accuracy: 0.9232
33440/60000 [===============>..............] - ETA: 38s - loss: 0.2470 - categorical_accuracy: 0.9233
33504/60000 [===============>..............] - ETA: 38s - loss: 0.2467 - categorical_accuracy: 0.9233
33568/60000 [===============>..............] - ETA: 38s - loss: 0.2464 - categorical_accuracy: 0.9234
33600/60000 [===============>..............] - ETA: 38s - loss: 0.2463 - categorical_accuracy: 0.9235
33664/60000 [===============>..............] - ETA: 38s - loss: 0.2459 - categorical_accuracy: 0.9236
33728/60000 [===============>..............] - ETA: 38s - loss: 0.2458 - categorical_accuracy: 0.9236
33792/60000 [===============>..............] - ETA: 38s - loss: 0.2455 - categorical_accuracy: 0.9237
33856/60000 [===============>..............] - ETA: 38s - loss: 0.2456 - categorical_accuracy: 0.9237
33920/60000 [===============>..............] - ETA: 38s - loss: 0.2454 - categorical_accuracy: 0.9239
33984/60000 [===============>..............] - ETA: 37s - loss: 0.2450 - categorical_accuracy: 0.9240
34048/60000 [================>.............] - ETA: 37s - loss: 0.2446 - categorical_accuracy: 0.9241
34112/60000 [================>.............] - ETA: 37s - loss: 0.2442 - categorical_accuracy: 0.9242
34176/60000 [================>.............] - ETA: 37s - loss: 0.2440 - categorical_accuracy: 0.9243
34240/60000 [================>.............] - ETA: 37s - loss: 0.2436 - categorical_accuracy: 0.9244
34304/60000 [================>.............] - ETA: 37s - loss: 0.2431 - categorical_accuracy: 0.9246
34368/60000 [================>.............] - ETA: 37s - loss: 0.2427 - categorical_accuracy: 0.9247
34432/60000 [================>.............] - ETA: 37s - loss: 0.2427 - categorical_accuracy: 0.9248
34496/60000 [================>.............] - ETA: 37s - loss: 0.2424 - categorical_accuracy: 0.9249
34560/60000 [================>.............] - ETA: 37s - loss: 0.2421 - categorical_accuracy: 0.9250
34624/60000 [================>.............] - ETA: 37s - loss: 0.2418 - categorical_accuracy: 0.9251
34688/60000 [================>.............] - ETA: 36s - loss: 0.2417 - categorical_accuracy: 0.9251
34752/60000 [================>.............] - ETA: 36s - loss: 0.2413 - categorical_accuracy: 0.9252
34816/60000 [================>.............] - ETA: 36s - loss: 0.2412 - categorical_accuracy: 0.9252
34880/60000 [================>.............] - ETA: 36s - loss: 0.2408 - categorical_accuracy: 0.9253
34944/60000 [================>.............] - ETA: 36s - loss: 0.2408 - categorical_accuracy: 0.9254
35008/60000 [================>.............] - ETA: 36s - loss: 0.2405 - categorical_accuracy: 0.9255
35072/60000 [================>.............] - ETA: 36s - loss: 0.2402 - categorical_accuracy: 0.9256
35136/60000 [================>.............] - ETA: 36s - loss: 0.2399 - categorical_accuracy: 0.9257
35200/60000 [================>.............] - ETA: 36s - loss: 0.2396 - categorical_accuracy: 0.9258
35264/60000 [================>.............] - ETA: 36s - loss: 0.2392 - categorical_accuracy: 0.9258
35328/60000 [================>.............] - ETA: 36s - loss: 0.2390 - categorical_accuracy: 0.9259
35392/60000 [================>.............] - ETA: 35s - loss: 0.2387 - categorical_accuracy: 0.9260
35456/60000 [================>.............] - ETA: 35s - loss: 0.2383 - categorical_accuracy: 0.9261
35520/60000 [================>.............] - ETA: 35s - loss: 0.2382 - categorical_accuracy: 0.9262
35584/60000 [================>.............] - ETA: 35s - loss: 0.2382 - categorical_accuracy: 0.9261
35648/60000 [================>.............] - ETA: 35s - loss: 0.2380 - categorical_accuracy: 0.9262
35712/60000 [================>.............] - ETA: 35s - loss: 0.2379 - categorical_accuracy: 0.9262
35776/60000 [================>.............] - ETA: 35s - loss: 0.2375 - categorical_accuracy: 0.9263
35840/60000 [================>.............] - ETA: 35s - loss: 0.2375 - categorical_accuracy: 0.9264
35872/60000 [================>.............] - ETA: 35s - loss: 0.2374 - categorical_accuracy: 0.9264
35904/60000 [================>.............] - ETA: 35s - loss: 0.2374 - categorical_accuracy: 0.9264
35968/60000 [================>.............] - ETA: 35s - loss: 0.2371 - categorical_accuracy: 0.9265
36000/60000 [=================>............] - ETA: 35s - loss: 0.2370 - categorical_accuracy: 0.9265
36064/60000 [=================>............] - ETA: 34s - loss: 0.2367 - categorical_accuracy: 0.9266
36096/60000 [=================>............] - ETA: 34s - loss: 0.2366 - categorical_accuracy: 0.9266
36160/60000 [=================>............] - ETA: 34s - loss: 0.2364 - categorical_accuracy: 0.9266
36224/60000 [=================>............] - ETA: 34s - loss: 0.2360 - categorical_accuracy: 0.9268
36288/60000 [=================>............] - ETA: 34s - loss: 0.2356 - categorical_accuracy: 0.9269
36352/60000 [=================>............] - ETA: 34s - loss: 0.2354 - categorical_accuracy: 0.9269
36416/60000 [=================>............] - ETA: 34s - loss: 0.2351 - categorical_accuracy: 0.9270
36480/60000 [=================>............] - ETA: 34s - loss: 0.2349 - categorical_accuracy: 0.9271
36544/60000 [=================>............] - ETA: 34s - loss: 0.2347 - categorical_accuracy: 0.9271
36608/60000 [=================>............] - ETA: 34s - loss: 0.2344 - categorical_accuracy: 0.9272
36672/60000 [=================>............] - ETA: 34s - loss: 0.2342 - categorical_accuracy: 0.9272
36736/60000 [=================>............] - ETA: 33s - loss: 0.2340 - categorical_accuracy: 0.9273
36768/60000 [=================>............] - ETA: 33s - loss: 0.2338 - categorical_accuracy: 0.9274
36832/60000 [=================>............] - ETA: 33s - loss: 0.2334 - categorical_accuracy: 0.9275
36896/60000 [=================>............] - ETA: 33s - loss: 0.2330 - categorical_accuracy: 0.9276
36960/60000 [=================>............] - ETA: 33s - loss: 0.2328 - categorical_accuracy: 0.9277
36992/60000 [=================>............] - ETA: 33s - loss: 0.2329 - categorical_accuracy: 0.9277
37056/60000 [=================>............] - ETA: 33s - loss: 0.2326 - categorical_accuracy: 0.9278
37120/60000 [=================>............] - ETA: 33s - loss: 0.2322 - categorical_accuracy: 0.9279
37184/60000 [=================>............] - ETA: 33s - loss: 0.2320 - categorical_accuracy: 0.9279
37248/60000 [=================>............] - ETA: 33s - loss: 0.2318 - categorical_accuracy: 0.9280
37280/60000 [=================>............] - ETA: 33s - loss: 0.2317 - categorical_accuracy: 0.9280
37344/60000 [=================>............] - ETA: 33s - loss: 0.2314 - categorical_accuracy: 0.9281
37408/60000 [=================>............] - ETA: 33s - loss: 0.2311 - categorical_accuracy: 0.9282
37472/60000 [=================>............] - ETA: 32s - loss: 0.2311 - categorical_accuracy: 0.9283
37536/60000 [=================>............] - ETA: 32s - loss: 0.2307 - categorical_accuracy: 0.9284
37600/60000 [=================>............] - ETA: 32s - loss: 0.2305 - categorical_accuracy: 0.9285
37664/60000 [=================>............] - ETA: 32s - loss: 0.2306 - categorical_accuracy: 0.9285
37728/60000 [=================>............] - ETA: 32s - loss: 0.2304 - categorical_accuracy: 0.9286
37792/60000 [=================>............] - ETA: 32s - loss: 0.2301 - categorical_accuracy: 0.9287
37856/60000 [=================>............] - ETA: 32s - loss: 0.2299 - categorical_accuracy: 0.9287
37920/60000 [=================>............] - ETA: 32s - loss: 0.2296 - categorical_accuracy: 0.9287
37984/60000 [=================>............] - ETA: 32s - loss: 0.2295 - categorical_accuracy: 0.9288
38048/60000 [==================>...........] - ETA: 32s - loss: 0.2292 - categorical_accuracy: 0.9289
38112/60000 [==================>...........] - ETA: 31s - loss: 0.2289 - categorical_accuracy: 0.9290
38176/60000 [==================>...........] - ETA: 31s - loss: 0.2290 - categorical_accuracy: 0.9290
38240/60000 [==================>...........] - ETA: 31s - loss: 0.2287 - categorical_accuracy: 0.9291
38304/60000 [==================>...........] - ETA: 31s - loss: 0.2284 - categorical_accuracy: 0.9292
38368/60000 [==================>...........] - ETA: 31s - loss: 0.2282 - categorical_accuracy: 0.9293
38432/60000 [==================>...........] - ETA: 31s - loss: 0.2280 - categorical_accuracy: 0.9293
38496/60000 [==================>...........] - ETA: 31s - loss: 0.2278 - categorical_accuracy: 0.9293
38560/60000 [==================>...........] - ETA: 31s - loss: 0.2276 - categorical_accuracy: 0.9294
38624/60000 [==================>...........] - ETA: 31s - loss: 0.2275 - categorical_accuracy: 0.9294
38688/60000 [==================>...........] - ETA: 31s - loss: 0.2276 - categorical_accuracy: 0.9294
38752/60000 [==================>...........] - ETA: 31s - loss: 0.2273 - categorical_accuracy: 0.9295
38816/60000 [==================>...........] - ETA: 30s - loss: 0.2272 - categorical_accuracy: 0.9295
38880/60000 [==================>...........] - ETA: 30s - loss: 0.2271 - categorical_accuracy: 0.9295
38944/60000 [==================>...........] - ETA: 30s - loss: 0.2270 - categorical_accuracy: 0.9295
39008/60000 [==================>...........] - ETA: 30s - loss: 0.2270 - categorical_accuracy: 0.9296
39072/60000 [==================>...........] - ETA: 30s - loss: 0.2269 - categorical_accuracy: 0.9296
39136/60000 [==================>...........] - ETA: 30s - loss: 0.2266 - categorical_accuracy: 0.9297
39200/60000 [==================>...........] - ETA: 30s - loss: 0.2266 - categorical_accuracy: 0.9297
39264/60000 [==================>...........] - ETA: 30s - loss: 0.2263 - categorical_accuracy: 0.9298
39328/60000 [==================>...........] - ETA: 30s - loss: 0.2260 - categorical_accuracy: 0.9299
39392/60000 [==================>...........] - ETA: 30s - loss: 0.2258 - categorical_accuracy: 0.9300
39456/60000 [==================>...........] - ETA: 29s - loss: 0.2257 - categorical_accuracy: 0.9300
39520/60000 [==================>...........] - ETA: 29s - loss: 0.2254 - categorical_accuracy: 0.9301
39584/60000 [==================>...........] - ETA: 29s - loss: 0.2251 - categorical_accuracy: 0.9302
39648/60000 [==================>...........] - ETA: 29s - loss: 0.2250 - categorical_accuracy: 0.9303
39712/60000 [==================>...........] - ETA: 29s - loss: 0.2246 - categorical_accuracy: 0.9304
39776/60000 [==================>...........] - ETA: 29s - loss: 0.2243 - categorical_accuracy: 0.9305
39840/60000 [==================>...........] - ETA: 29s - loss: 0.2242 - categorical_accuracy: 0.9305
39872/60000 [==================>...........] - ETA: 29s - loss: 0.2242 - categorical_accuracy: 0.9306
39936/60000 [==================>...........] - ETA: 29s - loss: 0.2240 - categorical_accuracy: 0.9306
39968/60000 [==================>...........] - ETA: 29s - loss: 0.2238 - categorical_accuracy: 0.9307
40032/60000 [===================>..........] - ETA: 29s - loss: 0.2236 - categorical_accuracy: 0.9308
40096/60000 [===================>..........] - ETA: 29s - loss: 0.2235 - categorical_accuracy: 0.9308
40160/60000 [===================>..........] - ETA: 28s - loss: 0.2233 - categorical_accuracy: 0.9309
40224/60000 [===================>..........] - ETA: 28s - loss: 0.2232 - categorical_accuracy: 0.9310
40288/60000 [===================>..........] - ETA: 28s - loss: 0.2228 - categorical_accuracy: 0.9311
40320/60000 [===================>..........] - ETA: 28s - loss: 0.2227 - categorical_accuracy: 0.9311
40384/60000 [===================>..........] - ETA: 28s - loss: 0.2229 - categorical_accuracy: 0.9311
40448/60000 [===================>..........] - ETA: 28s - loss: 0.2227 - categorical_accuracy: 0.9311
40512/60000 [===================>..........] - ETA: 28s - loss: 0.2224 - categorical_accuracy: 0.9312
40544/60000 [===================>..........] - ETA: 28s - loss: 0.2223 - categorical_accuracy: 0.9313
40576/60000 [===================>..........] - ETA: 28s - loss: 0.2223 - categorical_accuracy: 0.9313
40640/60000 [===================>..........] - ETA: 28s - loss: 0.2220 - categorical_accuracy: 0.9314
40704/60000 [===================>..........] - ETA: 28s - loss: 0.2223 - categorical_accuracy: 0.9313
40768/60000 [===================>..........] - ETA: 28s - loss: 0.2223 - categorical_accuracy: 0.9312
40832/60000 [===================>..........] - ETA: 27s - loss: 0.2222 - categorical_accuracy: 0.9313
40896/60000 [===================>..........] - ETA: 27s - loss: 0.2222 - categorical_accuracy: 0.9313
40928/60000 [===================>..........] - ETA: 27s - loss: 0.2221 - categorical_accuracy: 0.9313
40960/60000 [===================>..........] - ETA: 27s - loss: 0.2219 - categorical_accuracy: 0.9314
41024/60000 [===================>..........] - ETA: 27s - loss: 0.2216 - categorical_accuracy: 0.9315
41088/60000 [===================>..........] - ETA: 27s - loss: 0.2215 - categorical_accuracy: 0.9316
41152/60000 [===================>..........] - ETA: 27s - loss: 0.2212 - categorical_accuracy: 0.9316
41216/60000 [===================>..........] - ETA: 27s - loss: 0.2210 - categorical_accuracy: 0.9317
41280/60000 [===================>..........] - ETA: 27s - loss: 0.2208 - categorical_accuracy: 0.9318
41344/60000 [===================>..........] - ETA: 27s - loss: 0.2207 - categorical_accuracy: 0.9318
41408/60000 [===================>..........] - ETA: 27s - loss: 0.2204 - categorical_accuracy: 0.9319
41472/60000 [===================>..........] - ETA: 27s - loss: 0.2201 - categorical_accuracy: 0.9320
41536/60000 [===================>..........] - ETA: 26s - loss: 0.2200 - categorical_accuracy: 0.9321
41600/60000 [===================>..........] - ETA: 26s - loss: 0.2198 - categorical_accuracy: 0.9321
41664/60000 [===================>..........] - ETA: 26s - loss: 0.2196 - categorical_accuracy: 0.9321
41728/60000 [===================>..........] - ETA: 26s - loss: 0.2194 - categorical_accuracy: 0.9322
41792/60000 [===================>..........] - ETA: 26s - loss: 0.2193 - categorical_accuracy: 0.9322
41856/60000 [===================>..........] - ETA: 26s - loss: 0.2190 - categorical_accuracy: 0.9323
41888/60000 [===================>..........] - ETA: 26s - loss: 0.2190 - categorical_accuracy: 0.9323
41952/60000 [===================>..........] - ETA: 26s - loss: 0.2188 - categorical_accuracy: 0.9324
42016/60000 [====================>.........] - ETA: 26s - loss: 0.2186 - categorical_accuracy: 0.9324
42080/60000 [====================>.........] - ETA: 26s - loss: 0.2184 - categorical_accuracy: 0.9325
42144/60000 [====================>.........] - ETA: 26s - loss: 0.2181 - categorical_accuracy: 0.9326
42208/60000 [====================>.........] - ETA: 25s - loss: 0.2180 - categorical_accuracy: 0.9326
42272/60000 [====================>.........] - ETA: 25s - loss: 0.2177 - categorical_accuracy: 0.9327
42336/60000 [====================>.........] - ETA: 25s - loss: 0.2174 - categorical_accuracy: 0.9328
42368/60000 [====================>.........] - ETA: 25s - loss: 0.2173 - categorical_accuracy: 0.9329
42400/60000 [====================>.........] - ETA: 25s - loss: 0.2173 - categorical_accuracy: 0.9328
42464/60000 [====================>.........] - ETA: 25s - loss: 0.2170 - categorical_accuracy: 0.9329
42496/60000 [====================>.........] - ETA: 25s - loss: 0.2170 - categorical_accuracy: 0.9329
42560/60000 [====================>.........] - ETA: 25s - loss: 0.2170 - categorical_accuracy: 0.9329
42624/60000 [====================>.........] - ETA: 25s - loss: 0.2168 - categorical_accuracy: 0.9330
42688/60000 [====================>.........] - ETA: 25s - loss: 0.2166 - categorical_accuracy: 0.9330
42752/60000 [====================>.........] - ETA: 25s - loss: 0.2164 - categorical_accuracy: 0.9331
42816/60000 [====================>.........] - ETA: 25s - loss: 0.2162 - categorical_accuracy: 0.9332
42880/60000 [====================>.........] - ETA: 24s - loss: 0.2160 - categorical_accuracy: 0.9332
42944/60000 [====================>.........] - ETA: 24s - loss: 0.2158 - categorical_accuracy: 0.9333
43008/60000 [====================>.........] - ETA: 24s - loss: 0.2156 - categorical_accuracy: 0.9333
43040/60000 [====================>.........] - ETA: 24s - loss: 0.2155 - categorical_accuracy: 0.9333
43104/60000 [====================>.........] - ETA: 24s - loss: 0.2152 - categorical_accuracy: 0.9334
43168/60000 [====================>.........] - ETA: 24s - loss: 0.2150 - categorical_accuracy: 0.9334
43232/60000 [====================>.........] - ETA: 24s - loss: 0.2147 - categorical_accuracy: 0.9335
43296/60000 [====================>.........] - ETA: 24s - loss: 0.2147 - categorical_accuracy: 0.9335
43360/60000 [====================>.........] - ETA: 24s - loss: 0.2147 - categorical_accuracy: 0.9336
43424/60000 [====================>.........] - ETA: 24s - loss: 0.2146 - categorical_accuracy: 0.9335
43488/60000 [====================>.........] - ETA: 24s - loss: 0.2145 - categorical_accuracy: 0.9336
43552/60000 [====================>.........] - ETA: 24s - loss: 0.2145 - categorical_accuracy: 0.9336
43584/60000 [====================>.........] - ETA: 23s - loss: 0.2145 - categorical_accuracy: 0.9336
43616/60000 [====================>.........] - ETA: 23s - loss: 0.2144 - categorical_accuracy: 0.9336
43680/60000 [====================>.........] - ETA: 23s - loss: 0.2145 - categorical_accuracy: 0.9336
43744/60000 [====================>.........] - ETA: 23s - loss: 0.2143 - categorical_accuracy: 0.9337
43808/60000 [====================>.........] - ETA: 23s - loss: 0.2141 - categorical_accuracy: 0.9337
43872/60000 [====================>.........] - ETA: 23s - loss: 0.2140 - categorical_accuracy: 0.9337
43936/60000 [====================>.........] - ETA: 23s - loss: 0.2140 - categorical_accuracy: 0.9337
43968/60000 [====================>.........] - ETA: 23s - loss: 0.2138 - categorical_accuracy: 0.9337
44000/60000 [=====================>........] - ETA: 23s - loss: 0.2137 - categorical_accuracy: 0.9338
44064/60000 [=====================>........] - ETA: 23s - loss: 0.2135 - categorical_accuracy: 0.9338
44128/60000 [=====================>........] - ETA: 23s - loss: 0.2133 - categorical_accuracy: 0.9339
44192/60000 [=====================>........] - ETA: 23s - loss: 0.2131 - categorical_accuracy: 0.9339
44224/60000 [=====================>........] - ETA: 23s - loss: 0.2130 - categorical_accuracy: 0.9340
44256/60000 [=====================>........] - ETA: 22s - loss: 0.2129 - categorical_accuracy: 0.9340
44288/60000 [=====================>........] - ETA: 22s - loss: 0.2129 - categorical_accuracy: 0.9340
44352/60000 [=====================>........] - ETA: 22s - loss: 0.2126 - categorical_accuracy: 0.9341
44416/60000 [=====================>........] - ETA: 22s - loss: 0.2124 - categorical_accuracy: 0.9342
44480/60000 [=====================>........] - ETA: 22s - loss: 0.2123 - categorical_accuracy: 0.9343
44544/60000 [=====================>........] - ETA: 22s - loss: 0.2121 - categorical_accuracy: 0.9343
44608/60000 [=====================>........] - ETA: 22s - loss: 0.2119 - categorical_accuracy: 0.9344
44672/60000 [=====================>........] - ETA: 22s - loss: 0.2117 - categorical_accuracy: 0.9344
44736/60000 [=====================>........] - ETA: 22s - loss: 0.2116 - categorical_accuracy: 0.9345
44800/60000 [=====================>........] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9345
44864/60000 [=====================>........] - ETA: 22s - loss: 0.2114 - categorical_accuracy: 0.9345
44928/60000 [=====================>........] - ETA: 22s - loss: 0.2112 - categorical_accuracy: 0.9345
44992/60000 [=====================>........] - ETA: 21s - loss: 0.2109 - categorical_accuracy: 0.9346
45056/60000 [=====================>........] - ETA: 21s - loss: 0.2107 - categorical_accuracy: 0.9347
45088/60000 [=====================>........] - ETA: 21s - loss: 0.2107 - categorical_accuracy: 0.9347
45152/60000 [=====================>........] - ETA: 21s - loss: 0.2105 - categorical_accuracy: 0.9348
45216/60000 [=====================>........] - ETA: 21s - loss: 0.2106 - categorical_accuracy: 0.9347
45280/60000 [=====================>........] - ETA: 21s - loss: 0.2104 - categorical_accuracy: 0.9348
45344/60000 [=====================>........] - ETA: 21s - loss: 0.2102 - categorical_accuracy: 0.9348
45408/60000 [=====================>........] - ETA: 21s - loss: 0.2100 - categorical_accuracy: 0.9349
45472/60000 [=====================>........] - ETA: 21s - loss: 0.2099 - categorical_accuracy: 0.9349
45536/60000 [=====================>........] - ETA: 21s - loss: 0.2097 - categorical_accuracy: 0.9350
45600/60000 [=====================>........] - ETA: 21s - loss: 0.2096 - categorical_accuracy: 0.9350
45664/60000 [=====================>........] - ETA: 20s - loss: 0.2094 - categorical_accuracy: 0.9351
45728/60000 [=====================>........] - ETA: 20s - loss: 0.2093 - categorical_accuracy: 0.9351
45792/60000 [=====================>........] - ETA: 20s - loss: 0.2091 - categorical_accuracy: 0.9352
45856/60000 [=====================>........] - ETA: 20s - loss: 0.2089 - categorical_accuracy: 0.9353
45920/60000 [=====================>........] - ETA: 20s - loss: 0.2086 - categorical_accuracy: 0.9354
45984/60000 [=====================>........] - ETA: 20s - loss: 0.2084 - categorical_accuracy: 0.9354
46016/60000 [======================>.......] - ETA: 20s - loss: 0.2085 - categorical_accuracy: 0.9354
46080/60000 [======================>.......] - ETA: 20s - loss: 0.2083 - categorical_accuracy: 0.9355
46144/60000 [======================>.......] - ETA: 20s - loss: 0.2080 - categorical_accuracy: 0.9356
46208/60000 [======================>.......] - ETA: 20s - loss: 0.2078 - categorical_accuracy: 0.9356
46272/60000 [======================>.......] - ETA: 20s - loss: 0.2077 - categorical_accuracy: 0.9357
46336/60000 [======================>.......] - ETA: 19s - loss: 0.2077 - categorical_accuracy: 0.9357
46400/60000 [======================>.......] - ETA: 19s - loss: 0.2076 - categorical_accuracy: 0.9357
46464/60000 [======================>.......] - ETA: 19s - loss: 0.2074 - categorical_accuracy: 0.9358
46528/60000 [======================>.......] - ETA: 19s - loss: 0.2072 - categorical_accuracy: 0.9359
46560/60000 [======================>.......] - ETA: 19s - loss: 0.2071 - categorical_accuracy: 0.9358
46624/60000 [======================>.......] - ETA: 19s - loss: 0.2071 - categorical_accuracy: 0.9359
46656/60000 [======================>.......] - ETA: 19s - loss: 0.2070 - categorical_accuracy: 0.9359
46688/60000 [======================>.......] - ETA: 19s - loss: 0.2069 - categorical_accuracy: 0.9359
46752/60000 [======================>.......] - ETA: 19s - loss: 0.2067 - categorical_accuracy: 0.9360
46816/60000 [======================>.......] - ETA: 19s - loss: 0.2066 - categorical_accuracy: 0.9361
46848/60000 [======================>.......] - ETA: 19s - loss: 0.2066 - categorical_accuracy: 0.9361
46912/60000 [======================>.......] - ETA: 19s - loss: 0.2063 - categorical_accuracy: 0.9362
46976/60000 [======================>.......] - ETA: 19s - loss: 0.2062 - categorical_accuracy: 0.9362
47040/60000 [======================>.......] - ETA: 18s - loss: 0.2060 - categorical_accuracy: 0.9362
47104/60000 [======================>.......] - ETA: 18s - loss: 0.2058 - categorical_accuracy: 0.9363
47168/60000 [======================>.......] - ETA: 18s - loss: 0.2055 - categorical_accuracy: 0.9364
47200/60000 [======================>.......] - ETA: 18s - loss: 0.2054 - categorical_accuracy: 0.9364
47264/60000 [======================>.......] - ETA: 18s - loss: 0.2053 - categorical_accuracy: 0.9365
47296/60000 [======================>.......] - ETA: 18s - loss: 0.2052 - categorical_accuracy: 0.9365
47360/60000 [======================>.......] - ETA: 18s - loss: 0.2051 - categorical_accuracy: 0.9365
47424/60000 [======================>.......] - ETA: 18s - loss: 0.2050 - categorical_accuracy: 0.9366
47456/60000 [======================>.......] - ETA: 18s - loss: 0.2049 - categorical_accuracy: 0.9366
47520/60000 [======================>.......] - ETA: 18s - loss: 0.2049 - categorical_accuracy: 0.9366
47584/60000 [======================>.......] - ETA: 18s - loss: 0.2049 - categorical_accuracy: 0.9366
47648/60000 [======================>.......] - ETA: 18s - loss: 0.2048 - categorical_accuracy: 0.9366
47712/60000 [======================>.......] - ETA: 17s - loss: 0.2046 - categorical_accuracy: 0.9367
47776/60000 [======================>.......] - ETA: 17s - loss: 0.2044 - categorical_accuracy: 0.9367
47840/60000 [======================>.......] - ETA: 17s - loss: 0.2043 - categorical_accuracy: 0.9368
47904/60000 [======================>.......] - ETA: 17s - loss: 0.2041 - categorical_accuracy: 0.9369
47968/60000 [======================>.......] - ETA: 17s - loss: 0.2039 - categorical_accuracy: 0.9369
48032/60000 [=======================>......] - ETA: 17s - loss: 0.2037 - categorical_accuracy: 0.9370
48096/60000 [=======================>......] - ETA: 17s - loss: 0.2038 - categorical_accuracy: 0.9370
48160/60000 [=======================>......] - ETA: 17s - loss: 0.2037 - categorical_accuracy: 0.9370
48224/60000 [=======================>......] - ETA: 17s - loss: 0.2036 - categorical_accuracy: 0.9371
48288/60000 [=======================>......] - ETA: 17s - loss: 0.2034 - categorical_accuracy: 0.9371
48352/60000 [=======================>......] - ETA: 17s - loss: 0.2032 - categorical_accuracy: 0.9372
48416/60000 [=======================>......] - ETA: 16s - loss: 0.2032 - categorical_accuracy: 0.9371
48480/60000 [=======================>......] - ETA: 16s - loss: 0.2031 - categorical_accuracy: 0.9372
48544/60000 [=======================>......] - ETA: 16s - loss: 0.2030 - categorical_accuracy: 0.9372
48608/60000 [=======================>......] - ETA: 16s - loss: 0.2028 - categorical_accuracy: 0.9373
48672/60000 [=======================>......] - ETA: 16s - loss: 0.2026 - categorical_accuracy: 0.9374
48736/60000 [=======================>......] - ETA: 16s - loss: 0.2026 - categorical_accuracy: 0.9374
48800/60000 [=======================>......] - ETA: 16s - loss: 0.2027 - categorical_accuracy: 0.9374
48864/60000 [=======================>......] - ETA: 16s - loss: 0.2024 - categorical_accuracy: 0.9375
48928/60000 [=======================>......] - ETA: 16s - loss: 0.2023 - categorical_accuracy: 0.9375
48992/60000 [=======================>......] - ETA: 16s - loss: 0.2021 - categorical_accuracy: 0.9375
49056/60000 [=======================>......] - ETA: 15s - loss: 0.2019 - categorical_accuracy: 0.9376
49120/60000 [=======================>......] - ETA: 15s - loss: 0.2020 - categorical_accuracy: 0.9376
49184/60000 [=======================>......] - ETA: 15s - loss: 0.2017 - categorical_accuracy: 0.9377
49248/60000 [=======================>......] - ETA: 15s - loss: 0.2015 - categorical_accuracy: 0.9378
49280/60000 [=======================>......] - ETA: 15s - loss: 0.2014 - categorical_accuracy: 0.9378
49312/60000 [=======================>......] - ETA: 15s - loss: 0.2013 - categorical_accuracy: 0.9378
49376/60000 [=======================>......] - ETA: 15s - loss: 0.2012 - categorical_accuracy: 0.9378
49408/60000 [=======================>......] - ETA: 15s - loss: 0.2012 - categorical_accuracy: 0.9378
49472/60000 [=======================>......] - ETA: 15s - loss: 0.2010 - categorical_accuracy: 0.9379
49536/60000 [=======================>......] - ETA: 15s - loss: 0.2008 - categorical_accuracy: 0.9379
49600/60000 [=======================>......] - ETA: 15s - loss: 0.2007 - categorical_accuracy: 0.9380
49664/60000 [=======================>......] - ETA: 15s - loss: 0.2005 - categorical_accuracy: 0.9380
49728/60000 [=======================>......] - ETA: 15s - loss: 0.2005 - categorical_accuracy: 0.9381
49792/60000 [=======================>......] - ETA: 14s - loss: 0.2004 - categorical_accuracy: 0.9381
49856/60000 [=======================>......] - ETA: 14s - loss: 0.2002 - categorical_accuracy: 0.9381
49920/60000 [=======================>......] - ETA: 14s - loss: 0.2001 - categorical_accuracy: 0.9382
49984/60000 [=======================>......] - ETA: 14s - loss: 0.2001 - categorical_accuracy: 0.9382
50048/60000 [========================>.....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9382
50112/60000 [========================>.....] - ETA: 14s - loss: 0.2000 - categorical_accuracy: 0.9383
50144/60000 [========================>.....] - ETA: 14s - loss: 0.2000 - categorical_accuracy: 0.9383
50208/60000 [========================>.....] - ETA: 14s - loss: 0.1999 - categorical_accuracy: 0.9383
50272/60000 [========================>.....] - ETA: 14s - loss: 0.1997 - categorical_accuracy: 0.9384
50336/60000 [========================>.....] - ETA: 14s - loss: 0.1994 - categorical_accuracy: 0.9385
50400/60000 [========================>.....] - ETA: 14s - loss: 0.1994 - categorical_accuracy: 0.9385
50464/60000 [========================>.....] - ETA: 13s - loss: 0.1993 - categorical_accuracy: 0.9385
50528/60000 [========================>.....] - ETA: 13s - loss: 0.1991 - categorical_accuracy: 0.9386
50592/60000 [========================>.....] - ETA: 13s - loss: 0.1988 - categorical_accuracy: 0.9387
50656/60000 [========================>.....] - ETA: 13s - loss: 0.1986 - categorical_accuracy: 0.9387
50720/60000 [========================>.....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9387
50784/60000 [========================>.....] - ETA: 13s - loss: 0.1982 - categorical_accuracy: 0.9388
50848/60000 [========================>.....] - ETA: 13s - loss: 0.1982 - categorical_accuracy: 0.9388
50912/60000 [========================>.....] - ETA: 13s - loss: 0.1984 - categorical_accuracy: 0.9388
50944/60000 [========================>.....] - ETA: 13s - loss: 0.1984 - categorical_accuracy: 0.9389
51008/60000 [========================>.....] - ETA: 13s - loss: 0.1985 - categorical_accuracy: 0.9389
51072/60000 [========================>.....] - ETA: 13s - loss: 0.1984 - categorical_accuracy: 0.9389
51136/60000 [========================>.....] - ETA: 12s - loss: 0.1982 - categorical_accuracy: 0.9390
51200/60000 [========================>.....] - ETA: 12s - loss: 0.1980 - categorical_accuracy: 0.9390
51264/60000 [========================>.....] - ETA: 12s - loss: 0.1978 - categorical_accuracy: 0.9391
51328/60000 [========================>.....] - ETA: 12s - loss: 0.1977 - categorical_accuracy: 0.9391
51360/60000 [========================>.....] - ETA: 12s - loss: 0.1976 - categorical_accuracy: 0.9392
51424/60000 [========================>.....] - ETA: 12s - loss: 0.1975 - categorical_accuracy: 0.9392
51488/60000 [========================>.....] - ETA: 12s - loss: 0.1973 - categorical_accuracy: 0.9393
51552/60000 [========================>.....] - ETA: 12s - loss: 0.1971 - categorical_accuracy: 0.9393
51616/60000 [========================>.....] - ETA: 12s - loss: 0.1971 - categorical_accuracy: 0.9393
51680/60000 [========================>.....] - ETA: 12s - loss: 0.1972 - categorical_accuracy: 0.9393
51744/60000 [========================>.....] - ETA: 12s - loss: 0.1971 - categorical_accuracy: 0.9394
51808/60000 [========================>.....] - ETA: 11s - loss: 0.1969 - categorical_accuracy: 0.9394
51872/60000 [========================>.....] - ETA: 11s - loss: 0.1967 - categorical_accuracy: 0.9395
51936/60000 [========================>.....] - ETA: 11s - loss: 0.1966 - categorical_accuracy: 0.9395
52000/60000 [=========================>....] - ETA: 11s - loss: 0.1965 - categorical_accuracy: 0.9396
52064/60000 [=========================>....] - ETA: 11s - loss: 0.1966 - categorical_accuracy: 0.9396
52128/60000 [=========================>....] - ETA: 11s - loss: 0.1964 - categorical_accuracy: 0.9396
52160/60000 [=========================>....] - ETA: 11s - loss: 0.1964 - categorical_accuracy: 0.9396
52224/60000 [=========================>....] - ETA: 11s - loss: 0.1962 - categorical_accuracy: 0.9397
52288/60000 [=========================>....] - ETA: 11s - loss: 0.1960 - categorical_accuracy: 0.9398
52352/60000 [=========================>....] - ETA: 11s - loss: 0.1958 - categorical_accuracy: 0.9398
52416/60000 [=========================>....] - ETA: 11s - loss: 0.1958 - categorical_accuracy: 0.9398
52480/60000 [=========================>....] - ETA: 10s - loss: 0.1956 - categorical_accuracy: 0.9398
52544/60000 [=========================>....] - ETA: 10s - loss: 0.1954 - categorical_accuracy: 0.9399
52608/60000 [=========================>....] - ETA: 10s - loss: 0.1952 - categorical_accuracy: 0.9400
52672/60000 [=========================>....] - ETA: 10s - loss: 0.1950 - categorical_accuracy: 0.9400
52736/60000 [=========================>....] - ETA: 10s - loss: 0.1949 - categorical_accuracy: 0.9400
52800/60000 [=========================>....] - ETA: 10s - loss: 0.1946 - categorical_accuracy: 0.9401
52864/60000 [=========================>....] - ETA: 10s - loss: 0.1945 - categorical_accuracy: 0.9402
52928/60000 [=========================>....] - ETA: 10s - loss: 0.1944 - categorical_accuracy: 0.9402
52992/60000 [=========================>....] - ETA: 10s - loss: 0.1942 - categorical_accuracy: 0.9403
53056/60000 [=========================>....] - ETA: 10s - loss: 0.1941 - categorical_accuracy: 0.9403
53120/60000 [=========================>....] - ETA: 10s - loss: 0.1941 - categorical_accuracy: 0.9403
53152/60000 [=========================>....] - ETA: 10s - loss: 0.1940 - categorical_accuracy: 0.9403
53216/60000 [=========================>....] - ETA: 9s - loss: 0.1938 - categorical_accuracy: 0.9404 
53280/60000 [=========================>....] - ETA: 9s - loss: 0.1936 - categorical_accuracy: 0.9405
53344/60000 [=========================>....] - ETA: 9s - loss: 0.1937 - categorical_accuracy: 0.9404
53408/60000 [=========================>....] - ETA: 9s - loss: 0.1936 - categorical_accuracy: 0.9405
53472/60000 [=========================>....] - ETA: 9s - loss: 0.1934 - categorical_accuracy: 0.9405
53504/60000 [=========================>....] - ETA: 9s - loss: 0.1933 - categorical_accuracy: 0.9405
53568/60000 [=========================>....] - ETA: 9s - loss: 0.1931 - categorical_accuracy: 0.9406
53632/60000 [=========================>....] - ETA: 9s - loss: 0.1929 - categorical_accuracy: 0.9407
53696/60000 [=========================>....] - ETA: 9s - loss: 0.1929 - categorical_accuracy: 0.9407
53760/60000 [=========================>....] - ETA: 9s - loss: 0.1927 - categorical_accuracy: 0.9407
53824/60000 [=========================>....] - ETA: 9s - loss: 0.1926 - categorical_accuracy: 0.9407
53888/60000 [=========================>....] - ETA: 8s - loss: 0.1924 - categorical_accuracy: 0.9408
53952/60000 [=========================>....] - ETA: 8s - loss: 0.1923 - categorical_accuracy: 0.9408
54016/60000 [==========================>...] - ETA: 8s - loss: 0.1921 - categorical_accuracy: 0.9409
54080/60000 [==========================>...] - ETA: 8s - loss: 0.1919 - categorical_accuracy: 0.9409
54144/60000 [==========================>...] - ETA: 8s - loss: 0.1917 - categorical_accuracy: 0.9410
54208/60000 [==========================>...] - ETA: 8s - loss: 0.1916 - categorical_accuracy: 0.9411
54272/60000 [==========================>...] - ETA: 8s - loss: 0.1914 - categorical_accuracy: 0.9411
54336/60000 [==========================>...] - ETA: 8s - loss: 0.1914 - categorical_accuracy: 0.9411
54400/60000 [==========================>...] - ETA: 8s - loss: 0.1912 - categorical_accuracy: 0.9412
54464/60000 [==========================>...] - ETA: 8s - loss: 0.1911 - categorical_accuracy: 0.9412
54528/60000 [==========================>...] - ETA: 7s - loss: 0.1910 - categorical_accuracy: 0.9412
54592/60000 [==========================>...] - ETA: 7s - loss: 0.1908 - categorical_accuracy: 0.9413
54656/60000 [==========================>...] - ETA: 7s - loss: 0.1908 - categorical_accuracy: 0.9413
54720/60000 [==========================>...] - ETA: 7s - loss: 0.1907 - categorical_accuracy: 0.9413
54784/60000 [==========================>...] - ETA: 7s - loss: 0.1905 - categorical_accuracy: 0.9414
54848/60000 [==========================>...] - ETA: 7s - loss: 0.1904 - categorical_accuracy: 0.9414
54912/60000 [==========================>...] - ETA: 7s - loss: 0.1903 - categorical_accuracy: 0.9414
54976/60000 [==========================>...] - ETA: 7s - loss: 0.1901 - categorical_accuracy: 0.9415
55040/60000 [==========================>...] - ETA: 7s - loss: 0.1900 - categorical_accuracy: 0.9415
55104/60000 [==========================>...] - ETA: 7s - loss: 0.1898 - categorical_accuracy: 0.9416
55136/60000 [==========================>...] - ETA: 7s - loss: 0.1897 - categorical_accuracy: 0.9416
55200/60000 [==========================>...] - ETA: 7s - loss: 0.1896 - categorical_accuracy: 0.9417
55264/60000 [==========================>...] - ETA: 6s - loss: 0.1896 - categorical_accuracy: 0.9417
55328/60000 [==========================>...] - ETA: 6s - loss: 0.1895 - categorical_accuracy: 0.9417
55392/60000 [==========================>...] - ETA: 6s - loss: 0.1893 - categorical_accuracy: 0.9418
55456/60000 [==========================>...] - ETA: 6s - loss: 0.1891 - categorical_accuracy: 0.9418
55520/60000 [==========================>...] - ETA: 6s - loss: 0.1890 - categorical_accuracy: 0.9419
55552/60000 [==========================>...] - ETA: 6s - loss: 0.1889 - categorical_accuracy: 0.9419
55616/60000 [==========================>...] - ETA: 6s - loss: 0.1888 - categorical_accuracy: 0.9419
55680/60000 [==========================>...] - ETA: 6s - loss: 0.1886 - categorical_accuracy: 0.9420
55744/60000 [==========================>...] - ETA: 6s - loss: 0.1885 - categorical_accuracy: 0.9420
55776/60000 [==========================>...] - ETA: 6s - loss: 0.1885 - categorical_accuracy: 0.9420
55840/60000 [==========================>...] - ETA: 6s - loss: 0.1884 - categorical_accuracy: 0.9420
55904/60000 [==========================>...] - ETA: 5s - loss: 0.1882 - categorical_accuracy: 0.9421
55968/60000 [==========================>...] - ETA: 5s - loss: 0.1881 - categorical_accuracy: 0.9422
56032/60000 [===========================>..] - ETA: 5s - loss: 0.1879 - categorical_accuracy: 0.9422
56096/60000 [===========================>..] - ETA: 5s - loss: 0.1877 - categorical_accuracy: 0.9423
56160/60000 [===========================>..] - ETA: 5s - loss: 0.1877 - categorical_accuracy: 0.9423
56224/60000 [===========================>..] - ETA: 5s - loss: 0.1875 - categorical_accuracy: 0.9423
56288/60000 [===========================>..] - ETA: 5s - loss: 0.1874 - categorical_accuracy: 0.9424
56352/60000 [===========================>..] - ETA: 5s - loss: 0.1874 - categorical_accuracy: 0.9424
56416/60000 [===========================>..] - ETA: 5s - loss: 0.1874 - categorical_accuracy: 0.9424
56480/60000 [===========================>..] - ETA: 5s - loss: 0.1874 - categorical_accuracy: 0.9424
56544/60000 [===========================>..] - ETA: 5s - loss: 0.1873 - categorical_accuracy: 0.9425
56608/60000 [===========================>..] - ETA: 4s - loss: 0.1873 - categorical_accuracy: 0.9425
56672/60000 [===========================>..] - ETA: 4s - loss: 0.1873 - categorical_accuracy: 0.9425
56736/60000 [===========================>..] - ETA: 4s - loss: 0.1871 - categorical_accuracy: 0.9425
56800/60000 [===========================>..] - ETA: 4s - loss: 0.1870 - categorical_accuracy: 0.9426
56864/60000 [===========================>..] - ETA: 4s - loss: 0.1870 - categorical_accuracy: 0.9426
56928/60000 [===========================>..] - ETA: 4s - loss: 0.1869 - categorical_accuracy: 0.9426
56992/60000 [===========================>..] - ETA: 4s - loss: 0.1869 - categorical_accuracy: 0.9427
57056/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9427
57120/60000 [===========================>..] - ETA: 4s - loss: 0.1868 - categorical_accuracy: 0.9427
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1867 - categorical_accuracy: 0.9427
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1865 - categorical_accuracy: 0.9428
57280/60000 [===========================>..] - ETA: 3s - loss: 0.1865 - categorical_accuracy: 0.9428
57344/60000 [===========================>..] - ETA: 3s - loss: 0.1864 - categorical_accuracy: 0.9428
57408/60000 [===========================>..] - ETA: 3s - loss: 0.1863 - categorical_accuracy: 0.9429
57472/60000 [===========================>..] - ETA: 3s - loss: 0.1861 - categorical_accuracy: 0.9429
57536/60000 [===========================>..] - ETA: 3s - loss: 0.1861 - categorical_accuracy: 0.9429
57568/60000 [===========================>..] - ETA: 3s - loss: 0.1860 - categorical_accuracy: 0.9430
57600/60000 [===========================>..] - ETA: 3s - loss: 0.1859 - categorical_accuracy: 0.9430
57632/60000 [===========================>..] - ETA: 3s - loss: 0.1859 - categorical_accuracy: 0.9430
57664/60000 [===========================>..] - ETA: 3s - loss: 0.1859 - categorical_accuracy: 0.9430
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1859 - categorical_accuracy: 0.9430
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1858 - categorical_accuracy: 0.9431
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1857 - categorical_accuracy: 0.9431
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1856 - categorical_accuracy: 0.9431
57984/60000 [===========================>..] - ETA: 2s - loss: 0.1856 - categorical_accuracy: 0.9431
58048/60000 [============================>.] - ETA: 2s - loss: 0.1855 - categorical_accuracy: 0.9432
58080/60000 [============================>.] - ETA: 2s - loss: 0.1854 - categorical_accuracy: 0.9432
58144/60000 [============================>.] - ETA: 2s - loss: 0.1854 - categorical_accuracy: 0.9432
58208/60000 [============================>.] - ETA: 2s - loss: 0.1853 - categorical_accuracy: 0.9432
58272/60000 [============================>.] - ETA: 2s - loss: 0.1851 - categorical_accuracy: 0.9433
58336/60000 [============================>.] - ETA: 2s - loss: 0.1850 - categorical_accuracy: 0.9433
58400/60000 [============================>.] - ETA: 2s - loss: 0.1848 - categorical_accuracy: 0.9434
58464/60000 [============================>.] - ETA: 2s - loss: 0.1847 - categorical_accuracy: 0.9434
58528/60000 [============================>.] - ETA: 2s - loss: 0.1845 - categorical_accuracy: 0.9434
58592/60000 [============================>.] - ETA: 2s - loss: 0.1846 - categorical_accuracy: 0.9435
58656/60000 [============================>.] - ETA: 1s - loss: 0.1845 - categorical_accuracy: 0.9435
58720/60000 [============================>.] - ETA: 1s - loss: 0.1843 - categorical_accuracy: 0.9435
58784/60000 [============================>.] - ETA: 1s - loss: 0.1843 - categorical_accuracy: 0.9435
58848/60000 [============================>.] - ETA: 1s - loss: 0.1842 - categorical_accuracy: 0.9436
58912/60000 [============================>.] - ETA: 1s - loss: 0.1840 - categorical_accuracy: 0.9436
58976/60000 [============================>.] - ETA: 1s - loss: 0.1840 - categorical_accuracy: 0.9436
59040/60000 [============================>.] - ETA: 1s - loss: 0.1839 - categorical_accuracy: 0.9436
59072/60000 [============================>.] - ETA: 1s - loss: 0.1839 - categorical_accuracy: 0.9436
59136/60000 [============================>.] - ETA: 1s - loss: 0.1838 - categorical_accuracy: 0.9437
59200/60000 [============================>.] - ETA: 1s - loss: 0.1837 - categorical_accuracy: 0.9437
59264/60000 [============================>.] - ETA: 1s - loss: 0.1836 - categorical_accuracy: 0.9437
59328/60000 [============================>.] - ETA: 0s - loss: 0.1836 - categorical_accuracy: 0.9437
59392/60000 [============================>.] - ETA: 0s - loss: 0.1837 - categorical_accuracy: 0.9437
59456/60000 [============================>.] - ETA: 0s - loss: 0.1835 - categorical_accuracy: 0.9437
59520/60000 [============================>.] - ETA: 0s - loss: 0.1834 - categorical_accuracy: 0.9438
59584/60000 [============================>.] - ETA: 0s - loss: 0.1834 - categorical_accuracy: 0.9438
59616/60000 [============================>.] - ETA: 0s - loss: 0.1834 - categorical_accuracy: 0.9438
59680/60000 [============================>.] - ETA: 0s - loss: 0.1832 - categorical_accuracy: 0.9439
59744/60000 [============================>.] - ETA: 0s - loss: 0.1830 - categorical_accuracy: 0.9439
59808/60000 [============================>.] - ETA: 0s - loss: 0.1830 - categorical_accuracy: 0.9440
59872/60000 [============================>.] - ETA: 0s - loss: 0.1829 - categorical_accuracy: 0.9440
59936/60000 [============================>.] - ETA: 0s - loss: 0.1827 - categorical_accuracy: 0.9440
60000/60000 [==============================] - 91s 2ms/step - loss: 0.1827 - categorical_accuracy: 0.9441 - val_loss: 0.0479 - val_categorical_accuracy: 0.9843

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 15s
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
 2528/10000 [======>.......................] - ETA: 2s
 2720/10000 [=======>......................] - ETA: 2s
 2912/10000 [=======>......................] - ETA: 2s
 3104/10000 [========>.....................] - ETA: 2s
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
 6272/10000 [=================>............] - ETA: 1s
 6464/10000 [==================>...........] - ETA: 1s
 6656/10000 [==================>...........] - ETA: 0s
 6816/10000 [===================>..........] - ETA: 0s
 7008/10000 [====================>.........] - ETA: 0s
 7200/10000 [====================>.........] - ETA: 0s
 7392/10000 [=====================>........] - ETA: 0s
 7552/10000 [=====================>........] - ETA: 0s
 7712/10000 [======================>.......] - ETA: 0s
 7904/10000 [======================>.......] - ETA: 0s
 8096/10000 [=======================>......] - ETA: 0s
 8256/10000 [=======================>......] - ETA: 0s
 8448/10000 [========================>.....] - ETA: 0s
 8640/10000 [========================>.....] - ETA: 0s
 8832/10000 [=========================>....] - ETA: 0s
 9024/10000 [==========================>...] - ETA: 0s
 9216/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9600/10000 [===========================>..] - ETA: 0s
 9792/10000 [============================>.] - ETA: 0s
 9984/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 3s 296us/step
[[3.0900875e-09 2.3153145e-08 5.4942511e-08 ... 9.9999964e-01
  9.7641484e-10 2.5777956e-07]
 [9.3386880e-06 9.9400488e-05 9.9988163e-01 ... 3.9236059e-09
  4.0398340e-06 1.2708671e-09]
 [5.9458807e-07 9.9974519e-01 2.5762305e-05 ... 5.7322941e-05
  1.2428004e-05 6.9337347e-07]
 ...
 [3.7555492e-09 5.8641410e-07 3.5484196e-09 ... 1.2676744e-06
  1.1893344e-06 1.8290706e-06]
 [6.3490047e-06 1.6795470e-06 2.7229922e-07 ... 1.9082577e-06
  4.2041470e-04 9.0009308e-07]
 [2.0781392e-05 9.9552528e-07 1.7548153e-05 ... 8.1187101e-09
  1.2385980e-06 1.8682108e-08]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04789356698349584, 'accuracy_test:': 0.9843000173568726}

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
   cfff6ef..f9461a3  master     -> origin/master
Updating cfff6ef..f9461a3
Fast-forward
 deps.txt                                         |    5 +-
 error_list/20200520/list_log_json_20200520.md    | 1036 +++++++++++-----------
 error_list/20200520/list_log_testall_20200520.md |  441 +++++++++
 log_json/log_json.py                             |  346 ++++----
 4 files changed, 1131 insertions(+), 697 deletions(-)
[master dbf53a3] ml_store
 1 file changed, 1165 insertions(+)
To github.com:arita37/mlmodels_store.git
   f9461a3..dbf53a3  master -> master





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
{'loss': 0.42481420189142227, 'loss_history': []}

  #### Plot   ######################################################## 

  #### Save   ######################################################## 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm/'}
Model saved in path: /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tf/1_lstm//model//model.ckpt

  #### Load   ######################################################## 
2020-05-20 08:27:42.257039: W tensorflow/core/framework/op_kernel.cc:1651] OP_REQUIRES failed at save_restore_v2_ops.cc:184 : Not found: Key Variable not found in checkpoint
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
[master 57a9108] ml_store
 2 files changed, 236 insertions(+), 3 deletions(-)
To github.com:arita37/mlmodels_store.git
   dbf53a3..57a9108  master -> master





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
[master 8bd51d4] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   57a9108..8bd51d4  master -> master





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
 40%|████      | 2/5 [00:21<00:31, 10.63s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.8631359016621127, 'learning_rate': 0.032261780278328205, 'min_data_in_leaf': 30, 'num_leaves': 35} and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\x9e\xcf.\xb4\x8dHX\r\x00\x00\x00learning_rateq\x02G?\xa0\x84\x9d\xb6i\x8dwX\x10\x00\x00\x00min_data_in_leafq\x03K\x1eX\n\x00\x00\x00num_leavesq\x04K#u.' and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeb\x9e\xcf.\xb4\x8dHX\r\x00\x00\x00learning_rateq\x02G?\xa0\x84\x9d\xb6i\x8dwX\x10\x00\x00\x00min_data_in_leafq\x03K\x1eX\n\x00\x00\x00num_leavesq\x04K#u.' and reward: 0.3924
 60%|██████    | 3/5 [00:42<00:27, 13.87s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.7828554377919761, 'learning_rate': 0.019053989998400033, 'min_data_in_leaf': 10, 'num_leaves': 26} and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\r&\xd8\xd9\xfe\\X\r\x00\x00\x00learning_rateq\x02G?\x93\x82\xe3\x9f\x9b\x12\xe3X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K\x1au.' and reward: 0.3896
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9\r&\xd8\xd9\xfe\\X\r\x00\x00\x00learning_rateq\x02G?\x93\x82\xe3\x9f\x9b\x12\xe3X\x10\x00\x00\x00min_data_in_leafq\x03K\nX\n\x00\x00\x00num_leavesq\x04K\x1au.' and reward: 0.3896
 80%|████████  | 4/5 [01:00<00:14, 14.98s/it] 80%|████████  | 4/5 [01:00<00:15, 15.07s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.8217101381417626, 'learning_rate': 0.01350220414946531, 'min_data_in_leaf': 28, 'num_leaves': 62} and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaKs\x0fC\x89RX\r\x00\x00\x00learning_rateq\x02G?\x8b\xa7\x0b)\xf7\x8a+X\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K>u.' and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xeaKs\x0fC\x89RX\r\x00\x00\x00learning_rateq\x02G?\x8b\xa7\x0b)\xf7\x8a+X\x10\x00\x00\x00min_data_in_leafq\x03K\x1cX\n\x00\x00\x00num_leavesq\x04K>u.' and reward: 0.3902
Time for Gradient Boosting hyperparameter optimization: 91.2778582572937
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.8631359016621127, 'learning_rate': 0.032261780278328205, 'min_data_in_leaf': 30, 'num_leaves': 35}
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
Saving dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06} and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xb9\x99\x99\x99\x99\x99\x9aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xf0\x00\x00\x00\x00\x00\x00X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?@bM\xd2\xf1\xa9\xfcX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G>\xb0\xc6\xf7\xa0\xb5\xed\x8du.' and reward: 0.3894
 40%|████      | 2/5 [00:51<01:16, 25.59s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
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
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.2880312581762441, 'embedding_size_factor': 1.3045524981225554, 'layers.choice': 0, 'learning_rate': 0.0014989074826879973, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 1.1315515944102306e-05} and reward: 0.3718
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd2o\x1a\xa8\x85\xee\xd0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xdfrp\xb5\x9f\xa6X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?X\x8e\xdf\xa7\x8bB\rX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xe7\xba\xf8\xabZ\xbaeu.' and reward: 0.3718
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xd2o\x1a\xa8\x85\xee\xd0X\x15\x00\x00\x00embedding_size_factorq\x03G?\xf4\xdfrp\xb5\x9f\xa6X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?X\x8e\xdf\xa7\x8bB\rX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xe7\xba\xf8\xabZ\xbaeu.' and reward: 0.3718
 60%|██████    | 3/5 [01:42<01:06, 33.37s/it] 60%|██████    | 3/5 [01:42<01:08, 34.23s/it]
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
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.4383417055382513, 'embedding_size_factor': 0.9236098632420893, 'layers.choice': 3, 'learning_rate': 0.0032221387515240624, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 3.5823943523582016e-10} and reward: 0.377
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdc\r\xca^p\x9e!X\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\x8e6E\x9ch\xe8X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?jeP\x91\xf3w\x9bX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xf8\x9e6\xfc\xae|\x9cu.' and reward: 0.377
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xdc\r\xca^p\x9e!X\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\x8e6E\x9ch\xe8X\r\x00\x00\x00layers.choiceq\x04K\x03X\r\x00\x00\x00learning_rateq\x05G?jeP\x91\xf3w\x9bX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G=\xf8\x9e6\xfc\xae|\x9cu.' and reward: 0.377
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 157.48457956314087
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -133.63s of remaining time.
Ensemble size: 58
Ensemble weights: 
[0.13793103 0.12068966 0.03448276 0.12068966 0.34482759 0.17241379
 0.06896552]
	0.3998	 = Validation accuracy score
	1.61s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 255.29s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f3fc24c6668>

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
   8bd51d4..c794c80  master     -> origin/master
Updating 8bd51d4..c794c80
Fast-forward
 error_list/20200520/list_log_testall_20200520.md | 175 +++++++++++++++++++++++
 1 file changed, 175 insertions(+)
[master 286c004] ml_store
 1 file changed, 294 insertions(+)
To github.com:arita37/mlmodels_store.git
   c794c80..286c004  master -> master





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
[master ca3cac9] ml_store
 1 file changed, 36 insertions(+)
To github.com:arita37/mlmodels_store.git
   286c004..ca3cac9  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.86it/s, avg_epoch_loss=5.24]
INFO:root:Epoch[0] Elapsed time 2.590 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.241890
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.241890382766724 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7d0649e400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f7d0649e400>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 92.83it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1160.1620279947917,
    "abs_error": 388.24432373046875,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.572487407658521,
    "sMAPE": 0.5300597097278044,
    "MSIS": 102.89948660035968,
    "QuantileLoss[0.5]": 388.2442855834961,
    "Coverage[0.5]": 1.0,
    "RMSE": 34.06115130166319,
    "NRMSE": 0.7170768695086986,
    "ND": 0.6811303925095943,
    "wQuantileLoss[0.5]": 0.6811303255850809,
    "mean_wQuantileLoss": 0.6811303255850809,
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
100%|██████████| 10/10 [00:01<00:00,  7.91it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.265 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7d0649e400>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f7d0649e400>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 165.12it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:02<00:00,  4.72it/s, avg_epoch_loss=5.22]
INFO:root:Epoch[0] Elapsed time 2.118 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.223855
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.2238551616668705 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd9eb9be0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd9eb9be0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 162.10it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 407.6198323567708,
    "abs_error": 218.07424926757812,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.4449490330859007,
    "sMAPE": 0.34121586971558426,
    "MSIS": 57.7979597057725,
    "QuantileLoss[0.5]": 218.07424926757812,
    "Coverage[0.5]": 0.9166666666666666,
    "RMSE": 20.189597132106694,
    "NRMSE": 0.4250441501496146,
    "ND": 0.3825864022238213,
    "wQuantileLoss[0.5]": 0.3825864022238213,
    "mean_wQuantileLoss": 0.3825864022238213,
    "MAE_Coverage": 0.41666666666666663
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
 30%|███       | 3/10 [00:13<00:30,  4.36s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:24<00:16,  4.20s/it, avg_epoch_loss=6.9]  90%|█████████ | 9/10 [00:36<00:04,  4.09s/it, avg_epoch_loss=6.87]100%|██████████| 10/10 [00:39<00:00,  3.98s/it, avg_epoch_loss=6.86]
INFO:root:Epoch[0] Elapsed time 39.842 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.861526
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.861526012420654 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd81696a0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd81696a0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 134.93it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 53297.869791666664,
    "abs_error": 2714.48193359375,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 17.986021084326968,
    "sMAPE": 1.4116101907088774,
    "MSIS": 719.4407139599963,
    "QuantileLoss[0.5]": 2714.481674194336,
    "Coverage[0.5]": 1.0,
    "RMSE": 230.86331408793964,
    "NRMSE": 4.860280296588203,
    "ND": 4.762249006304825,
    "wQuantileLoss[0.5]": 4.762248551218133,
    "mean_wQuantileLoss": 4.762248551218133,
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
100%|██████████| 10/10 [00:00<00:00, 56.82it/s, avg_epoch_loss=5.24]
INFO:root:Epoch[0] Elapsed time 0.177 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.243184
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.24318380355835 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd2da2c50>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd2da2c50>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 153.47it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 340.46339925130206,
    "abs_error": 186.51275634765625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2358241646956334,
    "sMAPE": 0.32099082190179035,
    "MSIS": 49.43296335249827,
    "QuantileLoss[0.5]": 186.51274871826172,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 18.451650312405718,
    "NRMSE": 0.3884557960506467,
    "ND": 0.32721536201343204,
    "wQuantileLoss[0.5]": 0.32721534862852936,
    "mean_wQuantileLoss": 0.32721534862852936,
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
100%|██████████| 10/10 [00:01<00:00,  7.83it/s, avg_epoch_loss=123]
INFO:root:Epoch[0] Elapsed time 1.278 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=122.866774
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 122.86677375545166 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd8130588>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd8130588>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 150.85it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [02:00<18:03, 120.34s/it, avg_epoch_loss=0.703] 20%|██        | 2/10 [05:01<18:28, 138.51s/it, avg_epoch_loss=0.686] 30%|███       | 3/10 [08:25<18:26, 158.12s/it, avg_epoch_loss=0.669] 40%|████      | 4/10 [11:37<16:49, 168.26s/it, avg_epoch_loss=0.652] 50%|█████     | 5/10 [15:18<15:20, 184.07s/it, avg_epoch_loss=0.634] 60%|██████    | 6/10 [18:36<12:33, 188.49s/it, avg_epoch_loss=0.617] 70%|███████   | 7/10 [21:59<09:38, 192.89s/it, avg_epoch_loss=0.599] 80%|████████  | 8/10 [25:44<06:45, 202.53s/it, avg_epoch_loss=0.581] 90%|█████████ | 9/10 [29:05<03:22, 202.07s/it, avg_epoch_loss=0.563]100%|██████████| 10/10 [32:39<00:00, 205.47s/it, avg_epoch_loss=0.546]100%|██████████| 10/10 [32:39<00:00, 195.95s/it, avg_epoch_loss=0.546]
INFO:root:Epoch[0] Elapsed time 1959.483 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.546187
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5461866021156311 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd2dad710>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f7cd2dad710>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 15.06it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
   ca3cac9..cbdea93  master     -> origin/master
Updating ca3cac9..cbdea93
Fast-forward
 error_list/20200520/list_log_testall_20200520.md | 7 +++++++
 1 file changed, 7 insertions(+)
[master 05b8b71] ml_store
 1 file changed, 505 insertions(+)
To github.com:arita37/mlmodels_store.git
   cbdea93..05b8b71  master -> master





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
{'roc_auc_score': 0.9642857142857143}

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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7fad385ba518> 

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
[master 694d261] ml_store
 1 file changed, 109 insertions(+)
To github.com:arita37/mlmodels_store.git
   05b8b71..694d261  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 0.78801845  0.30196005  0.70098212 -0.39468968 -1.20376927 -1.17181338
   0.75539203  0.98401224 -0.55968142 -0.19893745]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.96457205 -0.10679399  1.12232832  1.45142926  1.21828168 -0.61803685
   0.43816635 -2.03720123 -1.94258918 -0.9970198 ]
 [ 1.12062155 -0.7029204  -1.22957425  0.72555052 -1.18013412 -0.32420422
   1.10223673  0.81434313  0.78046993  1.10861676]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 1.22867367  0.13437312 -0.18242041 -0.2683713  -1.73963799 -0.13167563
  -0.92687194  1.01855247  1.2305582  -0.49112514]
 [ 0.62567337  0.5924728   0.67457071  1.19783084  1.23187251  1.70459417
  -0.76730983  1.04008915 -0.91844004  1.46089238]
 [ 0.5630779  -1.17598267 -0.17418034  1.01012718  1.06796368  0.92001793
  -0.16819884 -0.19505734  0.80539342  0.4611641 ]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 0.76170668 -1.48515645  1.30253554 -0.59246129 -1.64162479 -2.30490794
  -1.34869645 -0.03181717  0.11248774 -0.36261209]
 [ 0.87226739 -2.51630386 -0.77507029 -0.59566788  1.02600767 -0.30912132
   1.74643509  0.51093777  1.71066184  0.14164054]
 [ 0.44118981  0.47985237 -0.1920037  -1.55269878 -1.88873982  0.57846442
   0.39859839 -0.9612636  -1.45832446 -3.05376438]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.44689516  0.38653915  1.35010682 -0.85145566  0.85063796  1.00088142
  -1.1601701  -0.38483225  1.45810824 -0.33128317]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 1.14809657 -0.7332716   0.26246745  0.83600472  1.17353145  1.54335911
   0.28474811  0.75880566  0.88490881  0.2764993 ]
 [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
   0.59921324 -0.21465384  1.02124813  0.60640394]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7f5849242d68>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7f58635be630> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 1.06523311 -0.66486777  1.00806543 -1.94504696 -1.23017555 -0.91542437
   0.33722094  1.22515585 -1.05354607  0.78522692]
 [ 0.85771953  0.09811225 -0.26046606  1.06032751 -1.39003042 -1.71116766
   0.2656424   1.65712464  1.41767401  0.44509671]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]
 [ 1.16755486  0.0353601   0.7147896  -1.53879325  1.10863359 -0.44789518
  -1.75592564  0.61798553 -0.18417633  0.85270406]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 1.03967316 -0.73153098  0.36184732 -1.56573815  0.95928819  1.01382247
  -1.78791289 -2.22711263 -1.6993336  -0.42449279]
 [ 1.12641981 -0.6294416   1.1010002  -1.1134361   0.94459507 -0.06741002
  -0.1834002   1.16143998 -0.02752939  0.78002714]
 [ 1.18947778 -0.68067814 -0.05682448 -0.08450803  0.82178321 -0.29736188
  -0.18657899  0.417302    0.78477065  0.49233656]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 1.02817479 -0.50845713  1.7653351   0.77741921  0.61771419 -0.11877117
   0.45015551 -0.19899818  1.86647138  0.8709698 ]
 [ 0.69174373  1.00978733 -1.21333813 -1.55694156 -1.20257258 -0.61244213
  -2.69836174 -0.13935181 -0.72853749  0.0722519 ]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.68188934 -1.15498263  1.22895559 -0.1776322   0.99854519 -1.51045638
  -0.27584606  1.01120706 -1.47656266  1.30970591]
 [ 0.35413361  0.21112476  0.92145007  0.01652757  0.90394545  0.17718772
   0.09542509 -1.11647002  0.0809271   0.0607502 ]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]
 [ 0.345716   -0.41302931 -0.46867382  1.83471763  0.77151441  0.56438286
   0.02186284  2.13782807 -0.785534    0.85328122]
 [ 0.56998385 -0.53302033 -0.17545897 -1.42655542  0.60660431  1.76795995
  -0.11598519 -0.47537288  0.47761018 -0.93391466]
 [ 0.87122579 -0.20975294 -0.45698786  0.93514778 -0.87353582  1.81252782
   0.92550121  0.14010988 -1.41914878  1.06898597]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]]
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
[[ 7.83440538e-01 -5.11884476e-02  8.24584625e-01 -7.25597119e-01
   9.31717197e-01 -8.67768678e-01  3.03085711e+00 -1.35977326e-01
  -7.97269785e-01  6.54580153e-01]
 [ 8.57296491e-01  9.56121704e-01 -8.26097432e-01 -7.05840507e-01
   1.13872896e+00  1.19268607e+00  2.82675712e-01 -2.37941936e-01
   1.15528789e+00  6.21082701e-01]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 1.06040861e+00  5.10307597e-01  5.01725109e-01 -9.15791849e-01
  -9.07318361e-01 -4.07252043e-01 -1.79612295e-01  9.84951672e-01
   1.07125243e+00 -5.93343754e-01]
 [ 1.02817479e+00 -5.08457134e-01  1.76533510e+00  7.77419205e-01
   6.17714185e-01 -1.18771172e-01  4.50155513e-01 -1.98998184e-01
   1.86647138e+00  8.70969803e-01]
 [ 7.00175710e-01  5.56073510e-01  8.96864073e-02  1.69380911e+00
   8.82393314e-01  1.96869779e-01 -5.63788735e-01  1.69869255e-01
  -1.16400797e+00 -6.01156801e-01]
 [ 1.12641981e+00 -6.29441604e-01  1.10100020e+00 -1.11343610e+00
   9.44595066e-01 -6.74100249e-02 -1.83400197e-01  1.16143998e+00
  -2.75293863e-02  7.80027135e-01]
 [ 8.95623122e-01 -2.29820588e+00 -1.95225583e-02  1.45652739e+00
  -1.85064099e+00  3.16637236e-01  1.11337266e-01 -2.66412594e+00
  -4.26428618e-01 -8.39988915e-01]
 [ 7.73703613e-01  1.27852808e+00 -2.11416392e+00 -4.42229280e-01
   1.06821044e+00  3.23527354e-01 -2.50644065e+00 -1.09991490e-01
   8.54894544e-03 -4.11639163e-01]
 [ 4.67397905e-01 -2.37875265e-01 -1.54491194e-01 -7.55662765e-01
  -5.47062239e-01  1.85143789e+00 -1.46405357e+00  2.09096677e-01
   1.55501599e+00 -9.24323185e-02]
 [ 8.72267394e-01 -2.51630386e+00 -7.75070287e-01 -5.95667881e-01
   1.02600767e+00 -3.09121319e-01  1.74643509e+00  5.10937774e-01
   1.71066184e+00  1.41640538e-01]
 [ 6.92114488e-01 -6.06524918e-02  2.05635552e+00 -2.41350300e+00
   1.17456965e+00 -1.77756638e+00 -2.81736269e-01 -7.77858827e-01
   1.11584111e+00  1.76024923e+00]
 [ 6.91743730e-01  1.00978733e+00 -1.21333813e+00 -1.55694156e+00
  -1.20257258e+00 -6.12442128e-01 -2.69836174e+00 -1.39351805e-01
  -7.28537489e-01  7.22518992e-02]
 [ 8.61462558e-01  7.43205537e-02 -1.34501002e+00 -1.99560718e-01
  -1.47533915e+00 -6.54603169e-01 -3.14563862e-01  3.18014296e-01
  -8.90271552e-01 -1.29525789e+00]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 9.80427414e-01  1.93752881e+00 -2.30839743e-01  3.66332015e-01
   1.10018476e+00 -1.04458938e+00 -3.44987210e-01  2.05117344e+00
   5.85662000e-01 -2.79308500e+00]
 [ 3.45715997e-01 -4.13029310e-01 -4.68673816e-01  1.83471763e+00
   7.71514409e-01  5.64382855e-01  2.18628366e-02  2.13782807e+00
  -7.85533997e-01  8.53281222e-01]
 [ 9.36211246e-01  2.04377395e-01 -1.49419377e+00  6.12232523e-01
  -9.84377246e-01  7.44884536e-01  4.94341651e-01 -3.62812886e-02
  -8.32395348e-01 -4.46699203e-01]
 [ 6.25673373e-01  5.92472801e-01  6.74570707e-01  1.19783084e+00
   1.23187251e+00  1.70459417e+00 -7.67309826e-01  1.04008915e+00
  -9.18440038e-01  1.46089238e+00]
 [ 1.01177337e+00  9.57467711e-02  7.31402517e-01  1.03345080e+00
  -1.42203164e+00 -1.46273275e-01 -1.74549518e-02 -8.57496825e-01
  -9.34181843e-01  9.54495667e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 2.07582971e+00 -1.40232915e+00 -4.79184915e-01  4.51122939e-01
   1.03436581e+00 -6.94920901e-01 -4.18937898e-01  5.15413802e-01
  -1.11487105e+00 -1.95210529e+00]
 [ 6.81889336e-01 -1.15498263e+00  1.22895559e+00 -1.77632196e-01
   9.98545187e-01 -1.51045638e+00 -2.75846063e-01  1.01120706e+00
  -1.47656266e+00  1.30970591e+00]
 [ 1.18947778e+00 -6.80678141e-01 -5.68244809e-02 -8.45080274e-02
   8.21783210e-01 -2.97361883e-01 -1.86578994e-01  4.17302005e-01
   7.84770651e-01  4.92336556e-01]]
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
[master 295c870] ml_store
 1 file changed, 272 insertions(+)
To github.com:arita37/mlmodels_store.git
   694d261..295c870  master -> master





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
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140126632288208
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140126632287984
     | -- GenericBlock(units=256, thetas_dim=7, backcast_length=10, forecast_length=5, share_thetas=False) at @140126632286752
| --  Stack Generic (#1) (share_weights_in_stack=0)
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140126632286304
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140126632285800
     | -- GenericBlock(units=256, thetas_dim=8, backcast_length=10, forecast_length=5, share_thetas=False) at @140126632285464

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
grad_step = 000000, loss = 0.814159
plot()
Saved image to .//n_beats_0.png.
grad_step = 000001, loss = 0.695032
grad_step = 000002, loss = 0.594556
grad_step = 000003, loss = 0.489043
grad_step = 000004, loss = 0.378772
grad_step = 000005, loss = 0.280433
grad_step = 000006, loss = 0.221571
grad_step = 000007, loss = 0.191723
grad_step = 000008, loss = 0.154179
grad_step = 000009, loss = 0.106513
grad_step = 000010, loss = 0.072851
grad_step = 000011, loss = 0.060345
grad_step = 000012, loss = 0.052496
grad_step = 000013, loss = 0.038973
grad_step = 000014, loss = 0.022476
grad_step = 000015, loss = 0.010907
grad_step = 000016, loss = 0.008907
grad_step = 000017, loss = 0.016076
grad_step = 000018, loss = 0.024499
grad_step = 000019, loss = 0.025811
grad_step = 000020, loss = 0.020653
grad_step = 000021, loss = 0.015312
grad_step = 000022, loss = 0.013777
grad_step = 000023, loss = 0.014567
grad_step = 000024, loss = 0.014995
grad_step = 000025, loss = 0.013555
grad_step = 000026, loss = 0.010725
grad_step = 000027, loss = 0.008078
grad_step = 000028, loss = 0.006931
grad_step = 000029, loss = 0.007325
grad_step = 000030, loss = 0.008030
grad_step = 000031, loss = 0.007836
grad_step = 000032, loss = 0.006811
grad_step = 000033, loss = 0.005926
grad_step = 000034, loss = 0.005835
grad_step = 000035, loss = 0.006356
grad_step = 000036, loss = 0.006826
grad_step = 000037, loss = 0.006785
grad_step = 000038, loss = 0.006277
grad_step = 000039, loss = 0.005704
grad_step = 000040, loss = 0.005443
grad_step = 000041, loss = 0.005527
grad_step = 000042, loss = 0.005623
grad_step = 000043, loss = 0.005394
grad_step = 000044, loss = 0.004861
grad_step = 000045, loss = 0.004365
grad_step = 000046, loss = 0.004191
grad_step = 000047, loss = 0.004316
grad_step = 000048, loss = 0.004495
grad_step = 000049, loss = 0.004519
grad_step = 000050, loss = 0.004377
grad_step = 000051, loss = 0.004195
grad_step = 000052, loss = 0.004096
grad_step = 000053, loss = 0.004081
grad_step = 000054, loss = 0.004050
grad_step = 000055, loss = 0.003943
grad_step = 000056, loss = 0.003822
grad_step = 000057, loss = 0.003769
grad_step = 000058, loss = 0.003771
grad_step = 000059, loss = 0.003742
grad_step = 000060, loss = 0.003631
grad_step = 000061, loss = 0.003477
grad_step = 000062, loss = 0.003368
grad_step = 000063, loss = 0.003338
grad_step = 000064, loss = 0.003345
grad_step = 000065, loss = 0.003325
grad_step = 000066, loss = 0.003261
grad_step = 000067, loss = 0.003194
grad_step = 000068, loss = 0.003152
grad_step = 000069, loss = 0.003118
grad_step = 000070, loss = 0.003062
grad_step = 000071, loss = 0.002990
grad_step = 000072, loss = 0.002938
grad_step = 000073, loss = 0.002913
grad_step = 000074, loss = 0.002881
grad_step = 000075, loss = 0.002821
grad_step = 000076, loss = 0.002757
grad_step = 000077, loss = 0.002711
grad_step = 000078, loss = 0.002673
grad_step = 000079, loss = 0.002628
grad_step = 000080, loss = 0.002582
grad_step = 000081, loss = 0.002545
grad_step = 000082, loss = 0.002513
grad_step = 000083, loss = 0.002466
grad_step = 000084, loss = 0.002413
grad_step = 000085, loss = 0.002371
grad_step = 000086, loss = 0.002331
grad_step = 000087, loss = 0.002285
grad_step = 000088, loss = 0.002240
grad_step = 000089, loss = 0.002201
grad_step = 000090, loss = 0.002152
grad_step = 000091, loss = 0.002104
grad_step = 000092, loss = 0.002063
grad_step = 000093, loss = 0.002017
grad_step = 000094, loss = 0.001975
grad_step = 000095, loss = 0.001936
grad_step = 000096, loss = 0.001890
grad_step = 000097, loss = 0.001849
grad_step = 000098, loss = 0.001803
grad_step = 000099, loss = 0.001758
grad_step = 000100, loss = 0.001712
plot()
Saved image to .//n_beats_100.png.
grad_step = 000101, loss = 0.001661
grad_step = 000102, loss = 0.001609
grad_step = 000103, loss = 0.001554
grad_step = 000104, loss = 0.001499
grad_step = 000105, loss = 0.001440
grad_step = 000106, loss = 0.001382
grad_step = 000107, loss = 0.001325
grad_step = 000108, loss = 0.001267
grad_step = 000109, loss = 0.001211
grad_step = 000110, loss = 0.001158
grad_step = 000111, loss = 0.001112
grad_step = 000112, loss = 0.001074
grad_step = 000113, loss = 0.001053
grad_step = 000114, loss = 0.001025
grad_step = 000115, loss = 0.000984
grad_step = 000116, loss = 0.000963
grad_step = 000117, loss = 0.000962
grad_step = 000118, loss = 0.000940
grad_step = 000119, loss = 0.000915
grad_step = 000120, loss = 0.000912
grad_step = 000121, loss = 0.000904
grad_step = 000122, loss = 0.000881
grad_step = 000123, loss = 0.000875
grad_step = 000124, loss = 0.000871
grad_step = 000125, loss = 0.000854
grad_step = 000126, loss = 0.000845
grad_step = 000127, loss = 0.000843
grad_step = 000128, loss = 0.000830
grad_step = 000129, loss = 0.000818
grad_step = 000130, loss = 0.000814
grad_step = 000131, loss = 0.000806
grad_step = 000132, loss = 0.000795
grad_step = 000133, loss = 0.000788
grad_step = 000134, loss = 0.000783
grad_step = 000135, loss = 0.000775
grad_step = 000136, loss = 0.000765
grad_step = 000137, loss = 0.000758
grad_step = 000138, loss = 0.000752
grad_step = 000139, loss = 0.000744
grad_step = 000140, loss = 0.000736
grad_step = 000141, loss = 0.000728
grad_step = 000142, loss = 0.000721
grad_step = 000143, loss = 0.000716
grad_step = 000144, loss = 0.000709
grad_step = 000145, loss = 0.000702
grad_step = 000146, loss = 0.000694
grad_step = 000147, loss = 0.000686
grad_step = 000148, loss = 0.000679
grad_step = 000149, loss = 0.000673
grad_step = 000150, loss = 0.000667
grad_step = 000151, loss = 0.000663
grad_step = 000152, loss = 0.000659
grad_step = 000153, loss = 0.000655
grad_step = 000154, loss = 0.000651
grad_step = 000155, loss = 0.000644
grad_step = 000156, loss = 0.000635
grad_step = 000157, loss = 0.000626
grad_step = 000158, loss = 0.000619
grad_step = 000159, loss = 0.000613
grad_step = 000160, loss = 0.000609
grad_step = 000161, loss = 0.000606
grad_step = 000162, loss = 0.000605
grad_step = 000163, loss = 0.000608
grad_step = 000164, loss = 0.000614
grad_step = 000165, loss = 0.000621
grad_step = 000166, loss = 0.000610
grad_step = 000167, loss = 0.000588
grad_step = 000168, loss = 0.000572
grad_step = 000169, loss = 0.000575
grad_step = 000170, loss = 0.000585
grad_step = 000171, loss = 0.000579
grad_step = 000172, loss = 0.000564
grad_step = 000173, loss = 0.000554
grad_step = 000174, loss = 0.000558
grad_step = 000175, loss = 0.000561
grad_step = 000176, loss = 0.000554
grad_step = 000177, loss = 0.000542
grad_step = 000178, loss = 0.000538
grad_step = 000179, loss = 0.000540
grad_step = 000180, loss = 0.000541
grad_step = 000181, loss = 0.000534
grad_step = 000182, loss = 0.000526
grad_step = 000183, loss = 0.000522
grad_step = 000184, loss = 0.000523
grad_step = 000185, loss = 0.000523
grad_step = 000186, loss = 0.000518
grad_step = 000187, loss = 0.000512
grad_step = 000188, loss = 0.000508
grad_step = 000189, loss = 0.000506
grad_step = 000190, loss = 0.000506
grad_step = 000191, loss = 0.000506
grad_step = 000192, loss = 0.000502
grad_step = 000193, loss = 0.000497
grad_step = 000194, loss = 0.000493
grad_step = 000195, loss = 0.000490
grad_step = 000196, loss = 0.000488
grad_step = 000197, loss = 0.000488
grad_step = 000198, loss = 0.000486
grad_step = 000199, loss = 0.000483
grad_step = 000200, loss = 0.000480
plot()
Saved image to .//n_beats_200.png.
grad_step = 000201, loss = 0.000476
grad_step = 000202, loss = 0.000473
grad_step = 000203, loss = 0.000471
grad_step = 000204, loss = 0.000469
grad_step = 000205, loss = 0.000468
grad_step = 000206, loss = 0.000467
grad_step = 000207, loss = 0.000466
grad_step = 000208, loss = 0.000464
grad_step = 000209, loss = 0.000462
grad_step = 000210, loss = 0.000461
grad_step = 000211, loss = 0.000459
grad_step = 000212, loss = 0.000457
grad_step = 000213, loss = 0.000454
grad_step = 000214, loss = 0.000451
grad_step = 000215, loss = 0.000448
grad_step = 000216, loss = 0.000445
grad_step = 000217, loss = 0.000442
grad_step = 000218, loss = 0.000440
grad_step = 000219, loss = 0.000437
grad_step = 000220, loss = 0.000435
grad_step = 000221, loss = 0.000433
grad_step = 000222, loss = 0.000431
grad_step = 000223, loss = 0.000429
grad_step = 000224, loss = 0.000427
grad_step = 000225, loss = 0.000426
grad_step = 000226, loss = 0.000424
grad_step = 000227, loss = 0.000423
grad_step = 000228, loss = 0.000424
grad_step = 000229, loss = 0.000427
grad_step = 000230, loss = 0.000435
grad_step = 000231, loss = 0.000455
grad_step = 000232, loss = 0.000478
grad_step = 000233, loss = 0.000500
grad_step = 000234, loss = 0.000469
grad_step = 000235, loss = 0.000423
grad_step = 000236, loss = 0.000407
grad_step = 000237, loss = 0.000433
grad_step = 000238, loss = 0.000451
grad_step = 000239, loss = 0.000422
grad_step = 000240, loss = 0.000400
grad_step = 000241, loss = 0.000412
grad_step = 000242, loss = 0.000425
grad_step = 000243, loss = 0.000411
grad_step = 000244, loss = 0.000393
grad_step = 000245, loss = 0.000401
grad_step = 000246, loss = 0.000411
grad_step = 000247, loss = 0.000398
grad_step = 000248, loss = 0.000387
grad_step = 000249, loss = 0.000393
grad_step = 000250, loss = 0.000396
grad_step = 000251, loss = 0.000387
grad_step = 000252, loss = 0.000380
grad_step = 000253, loss = 0.000384
grad_step = 000254, loss = 0.000386
grad_step = 000255, loss = 0.000379
grad_step = 000256, loss = 0.000374
grad_step = 000257, loss = 0.000376
grad_step = 000258, loss = 0.000377
grad_step = 000259, loss = 0.000373
grad_step = 000260, loss = 0.000368
grad_step = 000261, loss = 0.000368
grad_step = 000262, loss = 0.000369
grad_step = 000263, loss = 0.000366
grad_step = 000264, loss = 0.000362
grad_step = 000265, loss = 0.000360
grad_step = 000266, loss = 0.000360
grad_step = 000267, loss = 0.000360
grad_step = 000268, loss = 0.000357
grad_step = 000269, loss = 0.000354
grad_step = 000270, loss = 0.000353
grad_step = 000271, loss = 0.000352
grad_step = 000272, loss = 0.000351
grad_step = 000273, loss = 0.000349
grad_step = 000274, loss = 0.000347
grad_step = 000275, loss = 0.000345
grad_step = 000276, loss = 0.000343
grad_step = 000277, loss = 0.000342
grad_step = 000278, loss = 0.000341
grad_step = 000279, loss = 0.000340
grad_step = 000280, loss = 0.000338
grad_step = 000281, loss = 0.000336
grad_step = 000282, loss = 0.000334
grad_step = 000283, loss = 0.000333
grad_step = 000284, loss = 0.000331
grad_step = 000285, loss = 0.000330
grad_step = 000286, loss = 0.000328
grad_step = 000287, loss = 0.000327
grad_step = 000288, loss = 0.000326
grad_step = 000289, loss = 0.000326
grad_step = 000290, loss = 0.000326
grad_step = 000291, loss = 0.000326
grad_step = 000292, loss = 0.000328
grad_step = 000293, loss = 0.000331
grad_step = 000294, loss = 0.000334
grad_step = 000295, loss = 0.000337
grad_step = 000296, loss = 0.000339
grad_step = 000297, loss = 0.000335
grad_step = 000298, loss = 0.000328
grad_step = 000299, loss = 0.000318
grad_step = 000300, loss = 0.000311
plot()
Saved image to .//n_beats_300.png.
grad_step = 000301, loss = 0.000309
grad_step = 000302, loss = 0.000310
grad_step = 000303, loss = 0.000314
grad_step = 000304, loss = 0.000317
grad_step = 000305, loss = 0.000317
grad_step = 000306, loss = 0.000315
grad_step = 000307, loss = 0.000311
grad_step = 000308, loss = 0.000305
grad_step = 000309, loss = 0.000300
grad_step = 000310, loss = 0.000298
grad_step = 000311, loss = 0.000298
grad_step = 000312, loss = 0.000299
grad_step = 000313, loss = 0.000300
grad_step = 000314, loss = 0.000302
grad_step = 000315, loss = 0.000304
grad_step = 000316, loss = 0.000304
grad_step = 000317, loss = 0.000304
grad_step = 000318, loss = 0.000302
grad_step = 000319, loss = 0.000299
grad_step = 000320, loss = 0.000296
grad_step = 000321, loss = 0.000292
grad_step = 000322, loss = 0.000288
grad_step = 000323, loss = 0.000285
grad_step = 000324, loss = 0.000283
grad_step = 000325, loss = 0.000282
grad_step = 000326, loss = 0.000281
grad_step = 000327, loss = 0.000281
grad_step = 000328, loss = 0.000281
grad_step = 000329, loss = 0.000281
grad_step = 000330, loss = 0.000284
grad_step = 000331, loss = 0.000289
grad_step = 000332, loss = 0.000300
grad_step = 000333, loss = 0.000319
grad_step = 000334, loss = 0.000341
grad_step = 000335, loss = 0.000359
grad_step = 000336, loss = 0.000348
grad_step = 000337, loss = 0.000314
grad_step = 000338, loss = 0.000278
grad_step = 000339, loss = 0.000274
grad_step = 000340, loss = 0.000295
grad_step = 000341, loss = 0.000309
grad_step = 000342, loss = 0.000300
grad_step = 000343, loss = 0.000275
grad_step = 000344, loss = 0.000267
grad_step = 000345, loss = 0.000277
grad_step = 000346, loss = 0.000288
grad_step = 000347, loss = 0.000283
grad_step = 000348, loss = 0.000269
grad_step = 000349, loss = 0.000262
grad_step = 000350, loss = 0.000267
grad_step = 000351, loss = 0.000275
grad_step = 000352, loss = 0.000273
grad_step = 000353, loss = 0.000265
grad_step = 000354, loss = 0.000259
grad_step = 000355, loss = 0.000259
grad_step = 000356, loss = 0.000262
grad_step = 000357, loss = 0.000264
grad_step = 000358, loss = 0.000262
grad_step = 000359, loss = 0.000258
grad_step = 000360, loss = 0.000255
grad_step = 000361, loss = 0.000254
grad_step = 000362, loss = 0.000254
grad_step = 000363, loss = 0.000256
grad_step = 000364, loss = 0.000255
grad_step = 000365, loss = 0.000253
grad_step = 000366, loss = 0.000250
grad_step = 000367, loss = 0.000249
grad_step = 000368, loss = 0.000248
grad_step = 000369, loss = 0.000249
grad_step = 000370, loss = 0.000249
grad_step = 000371, loss = 0.000249
grad_step = 000372, loss = 0.000249
grad_step = 000373, loss = 0.000247
grad_step = 000374, loss = 0.000246
grad_step = 000375, loss = 0.000244
grad_step = 000376, loss = 0.000243
grad_step = 000377, loss = 0.000242
grad_step = 000378, loss = 0.000241
grad_step = 000379, loss = 0.000241
grad_step = 000380, loss = 0.000241
grad_step = 000381, loss = 0.000241
grad_step = 000382, loss = 0.000241
grad_step = 000383, loss = 0.000243
grad_step = 000384, loss = 0.000245
grad_step = 000385, loss = 0.000247
grad_step = 000386, loss = 0.000250
grad_step = 000387, loss = 0.000253
grad_step = 000388, loss = 0.000256
grad_step = 000389, loss = 0.000258
grad_step = 000390, loss = 0.000257
grad_step = 000391, loss = 0.000253
grad_step = 000392, loss = 0.000245
grad_step = 000393, loss = 0.000238
grad_step = 000394, loss = 0.000233
grad_step = 000395, loss = 0.000233
grad_step = 000396, loss = 0.000234
grad_step = 000397, loss = 0.000236
grad_step = 000398, loss = 0.000238
grad_step = 000399, loss = 0.000239
grad_step = 000400, loss = 0.000240
plot()
Saved image to .//n_beats_400.png.
grad_step = 000401, loss = 0.000241
grad_step = 000402, loss = 0.000240
grad_step = 000403, loss = 0.000239
grad_step = 000404, loss = 0.000235
grad_step = 000405, loss = 0.000231
grad_step = 000406, loss = 0.000227
grad_step = 000407, loss = 0.000225
grad_step = 000408, loss = 0.000225
grad_step = 000409, loss = 0.000227
grad_step = 000410, loss = 0.000228
grad_step = 000411, loss = 0.000229
grad_step = 000412, loss = 0.000228
grad_step = 000413, loss = 0.000228
grad_step = 000414, loss = 0.000227
grad_step = 000415, loss = 0.000227
grad_step = 000416, loss = 0.000228
grad_step = 000417, loss = 0.000230
grad_step = 000418, loss = 0.000232
grad_step = 000419, loss = 0.000236
grad_step = 000420, loss = 0.000239
grad_step = 000421, loss = 0.000242
grad_step = 000422, loss = 0.000243
grad_step = 000423, loss = 0.000243
grad_step = 000424, loss = 0.000243
grad_step = 000425, loss = 0.000242
grad_step = 000426, loss = 0.000241
grad_step = 000427, loss = 0.000235
grad_step = 000428, loss = 0.000227
grad_step = 000429, loss = 0.000219
grad_step = 000430, loss = 0.000215
grad_step = 000431, loss = 0.000216
grad_step = 000432, loss = 0.000220
grad_step = 000433, loss = 0.000224
grad_step = 000434, loss = 0.000226
grad_step = 000435, loss = 0.000225
grad_step = 000436, loss = 0.000221
grad_step = 000437, loss = 0.000217
grad_step = 000438, loss = 0.000215
grad_step = 000439, loss = 0.000214
grad_step = 000440, loss = 0.000216
grad_step = 000441, loss = 0.000218
grad_step = 000442, loss = 0.000222
grad_step = 000443, loss = 0.000223
grad_step = 000444, loss = 0.000224
grad_step = 000445, loss = 0.000222
grad_step = 000446, loss = 0.000220
grad_step = 000447, loss = 0.000218
grad_step = 000448, loss = 0.000218
grad_step = 000449, loss = 0.000219
grad_step = 000450, loss = 0.000222
grad_step = 000451, loss = 0.000225
grad_step = 000452, loss = 0.000227
grad_step = 000453, loss = 0.000228
grad_step = 000454, loss = 0.000226
grad_step = 000455, loss = 0.000227
grad_step = 000456, loss = 0.000226
grad_step = 000457, loss = 0.000221
grad_step = 000458, loss = 0.000214
grad_step = 000459, loss = 0.000208
grad_step = 000460, loss = 0.000208
grad_step = 000461, loss = 0.000212
grad_step = 000462, loss = 0.000214
grad_step = 000463, loss = 0.000212
grad_step = 000464, loss = 0.000207
grad_step = 000465, loss = 0.000203
grad_step = 000466, loss = 0.000203
grad_step = 000467, loss = 0.000206
grad_step = 000468, loss = 0.000210
grad_step = 000469, loss = 0.000211
grad_step = 000470, loss = 0.000210
grad_step = 000471, loss = 0.000208
grad_step = 000472, loss = 0.000209
grad_step = 000473, loss = 0.000213
grad_step = 000474, loss = 0.000221
grad_step = 000475, loss = 0.000232
grad_step = 000476, loss = 0.000242
grad_step = 000477, loss = 0.000249
grad_step = 000478, loss = 0.000256
grad_step = 000479, loss = 0.000255
grad_step = 000480, loss = 0.000243
grad_step = 000481, loss = 0.000219
grad_step = 000482, loss = 0.000200
grad_step = 000483, loss = 0.000199
grad_step = 000484, loss = 0.000212
grad_step = 000485, loss = 0.000221
grad_step = 000486, loss = 0.000217
grad_step = 000487, loss = 0.000207
grad_step = 000488, loss = 0.000201
grad_step = 000489, loss = 0.000202
grad_step = 000490, loss = 0.000204
grad_step = 000491, loss = 0.000205
grad_step = 000492, loss = 0.000204
grad_step = 000493, loss = 0.000204
grad_step = 000494, loss = 0.000204
grad_step = 000495, loss = 0.000201
grad_step = 000496, loss = 0.000196
grad_step = 000497, loss = 0.000192
grad_step = 000498, loss = 0.000193
grad_step = 000499, loss = 0.000196
grad_step = 000500, loss = 0.000199
plot()
Saved image to .//n_beats_500.png.
grad_step = 000501, loss = 0.000198
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
[[0.8676643  0.8334873  0.9321882  0.9429578  1.0052848 ]
 [0.8341052  0.9127953  0.9552809  1.0174506  0.99464226]
 [0.90266013 0.9238919  0.9949776  0.97542846 0.9556117 ]
 [0.91598034 0.999385   0.9957871  0.94398975 0.9053129 ]
 [1.0092709  0.9927901  0.9514295  0.91235244 0.85954636]
 [0.98753893 0.9375574  0.9192883  0.87019235 0.84965605]
 [0.9428687  0.9066776  0.85007393 0.85541755 0.8216622 ]
 [0.88914293 0.833228   0.8588287  0.80160105 0.8294238 ]
 [0.8319     0.83466816 0.8067765  0.83951443 0.857018  ]
 [0.83232963 0.7993008  0.8197278  0.82801217 0.8486692 ]
 [0.7876748  0.82908803 0.86332357 0.8211688  0.8967978 ]
 [0.83158815 0.8518843  0.82324284 0.91518366 0.9368344 ]
 [0.86105484 0.8296182  0.9293012  0.94846153 1.0057484 ]
 [0.8403678  0.9274512  0.9590682  1.0200016  0.9834279 ]
 [0.9238864  0.9413867  0.99643314 0.9653363  0.93727946]
 [0.93281806 1.0077966  0.9865291  0.9254241  0.8833974 ]
 [1.0143832  0.9893973  0.9349091  0.890073   0.8367256 ]
 [0.98282444 0.92036045 0.9019785  0.8456115  0.8317504 ]
 [0.93129927 0.89408994 0.8355135  0.8392143  0.8133791 ]
 [0.8963909  0.84419703 0.85367626 0.8046273  0.8296042 ]
 [0.84901744 0.8513407  0.8086989  0.84544647 0.86009276]
 [0.8525384  0.8200702  0.8228364  0.8373608  0.8519676 ]
 [0.80433166 0.83774626 0.87114406 0.8257834  0.8990246 ]
 [0.84157777 0.8668668  0.8273372  0.9110229  0.9303386 ]
 [0.8713435  0.84344137 0.9330677  0.94564676 1.0076334 ]
 [0.8470559  0.9188252  0.95956945 1.0278171  1.0077877 ]
 [0.9138087  0.93683517 1.0038055  0.98885494 0.9661452 ]
 [0.92855453 1.0169836  1.0081897  0.9539838  0.91267073]
 [1.0212203  1.0103469  0.9631343  0.91885686 0.86362875]
 [0.999689   0.948323   0.9296932  0.8726275  0.85246253]
 [0.9488955  0.91378975 0.8570139  0.85977566 0.82716715]]

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
[master 34ded35] ml_store
 1 file changed, 1123 insertions(+)
To github.com:arita37/mlmodels_store.git
   295c870..34ded35  master -> master





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
[master a2fc9d9] ml_store
 1 file changed, 38 insertions(+)
To github.com:arita37/mlmodels_store.git
   34ded35..a2fc9d9  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//matchzoo_models.py 

  #### Loading params   ############################################## 

  {'dataset': 'WIKI_QA', 'data_path': '/home/runner/work/mlmodels/mlmodels/mlmodels/dataset/nlp/', 'dataset_pars': {'data_pack': '', 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'resample': True, 'sort': False, 'callbacks': 'PADDING'}, 'dataloader': '', 'dataloader_pars': {'device': 'cpu', 'dataset': 'None', 'stage': 'train', 'callback': 'PADDING'}, 'preprocess': {'train': {'transform': True, 'mode': 'pair', 'num_dup': 2, 'num_neg': 1, 'batch_size': 20, 'stage': 'train', 'resample': True, 'sort': False, 'dataloader_callback': 'PADDING'}, 'test': {'transform': True, 'batch_size': 20, 'stage': 'dev', 'dataloader_callback': 'PADDING'}}} {'checkpointdir': 'ztest/model_tch/MATCHZOO/BERT/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/MATCHZOO/BERT/'} 

  #### Loading dataset   ############################################# 

  #### Model init   ################################################## 
  0%|          | 0/231508 [00:00<?, ?B/s] 23%|██▎       | 52224/231508 [00:00<00:00, 372418.17B/s] 83%|████████▎ | 191488/231508 [00:00<00:00, 457186.39B/s]100%|██████████| 231508/231508 [00:00<00:00, 808464.96B/s]
  0%|          | 0/433 [00:00<?, ?B/s]100%|██████████| 433/433 [00:00<00:00, 178947.05B/s]
  0%|          | 0/440473133 [00:00<?, ?B/s]  0%|          | 52224/440473133 [00:00<19:48, 370619.78B/s]  0%|          | 121856/440473133 [00:00<18:21, 399678.46B/s]  0%|          | 400384/440473133 [00:00<13:58, 525027.08B/s]  0%|          | 1479680/440473133 [00:00<10:02, 728405.07B/s]  1%|          | 4539392/440473133 [00:00<07:03, 1030069.06B/s]  2%|▏         | 7272448/440473133 [00:00<04:59, 1444782.90B/s]  3%|▎         | 11208704/440473133 [00:00<03:31, 2032005.88B/s]  3%|▎         | 13793280/440473133 [00:00<02:32, 2795998.69B/s]  4%|▍         | 17680384/440473133 [00:01<01:49, 3874824.58B/s]  5%|▍         | 20416512/440473133 [00:01<01:21, 5166937.52B/s]  6%|▌         | 24455168/440473133 [00:01<00:59, 6997610.15B/s]  6%|▌         | 27496448/440473133 [00:01<00:46, 8959221.41B/s]  7%|▋         | 31782912/440473133 [00:01<00:35, 11421450.24B/s]  8%|▊         | 36015104/440473133 [00:01<00:27, 14624722.34B/s]  9%|▉         | 39377920/440473133 [00:01<00:23, 17032556.08B/s] 10%|▉         | 43219968/440473133 [00:01<00:19, 20447302.92B/s] 11%|█         | 46601216/440473133 [00:02<00:17, 22208898.79B/s] 11%|█▏        | 50296832/440473133 [00:02<00:16, 23935200.21B/s] 12%|█▏        | 54141952/440473133 [00:02<00:14, 26991699.83B/s] 13%|█▎        | 57478144/440473133 [00:02<00:14, 27017477.39B/s] 14%|█▍        | 61519872/440473133 [00:02<00:13, 27984122.73B/s] 15%|█▍        | 65580032/440473133 [00:02<00:12, 30861008.57B/s] 16%|█▌        | 68981760/440473133 [00:02<00:12, 29598988.74B/s] 17%|█▋        | 72808448/440473133 [00:02<00:12, 29543060.39B/s] 17%|█▋        | 76681216/440473133 [00:02<00:11, 31805635.88B/s] 18%|█▊        | 80024576/440473133 [00:03<00:11, 30324606.46B/s] 19%|█▉        | 84146176/440473133 [00:03<00:11, 30448662.94B/s] 20%|██        | 88326144/440473133 [00:03<00:10, 33148608.73B/s] 21%|██        | 91774976/440473133 [00:03<00:11, 31105118.47B/s] 22%|██▏       | 95516672/440473133 [00:03<00:11, 30379150.71B/s] 23%|██▎       | 99705856/440473133 [00:03<00:10, 33108225.42B/s] 23%|██▎       | 103150592/440473133 [00:03<00:10, 32141241.65B/s] 24%|██▍       | 106971136/440473133 [00:03<00:09, 33747522.25B/s] 25%|██▌       | 110437376/440473133 [00:03<00:10, 32211553.55B/s] 26%|██▌       | 114702336/440473133 [00:04<00:09, 32614763.47B/s] 27%|██▋       | 119246848/440473133 [00:04<00:09, 35632394.53B/s] 28%|██▊       | 122933248/440473133 [00:04<00:09, 33798260.97B/s] 29%|██▉       | 127608832/440473133 [00:04<00:08, 36862080.84B/s] 30%|██▉       | 131451904/440473133 [00:04<00:08, 34865467.32B/s] 31%|███       | 135411712/440473133 [00:04<00:09, 33856736.01B/s] 32%|███▏      | 139458560/440473133 [00:04<00:08, 35599757.38B/s] 32%|███▏      | 143112192/440473133 [00:04<00:08, 33860323.11B/s] 33%|███▎      | 147535872/440473133 [00:05<00:08, 36421968.69B/s] 34%|███▍      | 151290880/440473133 [00:05<00:08, 33947260.90B/s] 35%|███▌      | 155105280/440473133 [00:05<00:08, 32963962.54B/s] 36%|███▌      | 158997504/440473133 [00:05<00:08, 34549596.37B/s] 37%|███▋      | 162532352/440473133 [00:05<00:08, 33259174.51B/s] 38%|███▊      | 167299072/440473133 [00:05<00:07, 36575295.28B/s] 39%|███▉      | 171104256/440473133 [00:05<00:07, 34739444.30B/s] 40%|███▉      | 175306752/440473133 [00:05<00:07, 34302193.75B/s] 41%|████      | 179683328/440473133 [00:05<00:07, 36680699.97B/s] 42%|████▏     | 183458816/440473133 [00:06<00:07, 34792766.92B/s] 43%|████▎     | 187916288/440473133 [00:06<00:06, 37244482.34B/s] 44%|████▎     | 191756288/440473133 [00:06<00:07, 35484702.25B/s] 44%|████▍     | 195844096/440473133 [00:06<00:06, 36946224.55B/s] 45%|████▌     | 199623680/440473133 [00:06<00:06, 34730759.18B/s] 46%|████▌     | 203454464/440473133 [00:06<00:06, 34033826.75B/s] 47%|████▋     | 207845376/440473133 [00:06<00:06, 36493776.21B/s] 48%|████▊     | 211589120/440473133 [00:06<00:06, 34817656.52B/s] 49%|████▉     | 215862272/440473133 [00:06<00:06, 36865091.01B/s] 50%|████▉     | 219639808/440473133 [00:07<00:06, 35110546.15B/s] 51%|█████     | 224124928/440473133 [00:07<00:05, 37556646.57B/s] 52%|█████▏    | 227985408/440473133 [00:07<00:05, 35586693.50B/s] 53%|█████▎    | 232044544/440473133 [00:07<00:05, 34843815.39B/s] 54%|█████▎    | 236408832/440473133 [00:07<00:05, 37086766.81B/s] 55%|█████▍    | 240205824/440473133 [00:07<00:05, 35221486.57B/s] 56%|█████▌    | 244676608/440473133 [00:07<00:05, 37614106.98B/s] 56%|█████▋    | 248540160/440473133 [00:07<00:05, 35764573.24B/s] 57%|█████▋    | 252868608/440473133 [00:07<00:05, 35118403.84B/s] 58%|█████▊    | 257200128/440473133 [00:08<00:04, 37231193.76B/s] 59%|█████▉    | 261004288/440473133 [00:08<00:05, 35238917.90B/s] 60%|██████    | 265522176/440473133 [00:08<00:04, 37728169.76B/s] 61%|██████    | 269400064/440473133 [00:08<00:04, 35596809.71B/s] 62%|██████▏   | 273643520/440473133 [00:08<00:04, 34970327.53B/s] 63%|██████▎   | 278093824/440473133 [00:08<00:04, 37371670.04B/s] 64%|██████▍   | 281926656/440473133 [00:08<00:04, 35372854.44B/s] 65%|██████▍   | 285923328/440473133 [00:08<00:04, 36633102.97B/s] 66%|██████▌   | 289660928/440473133 [00:08<00:04, 34495411.35B/s] 67%|██████▋   | 293861376/440473133 [00:09<00:04, 34187588.83B/s] 68%|██████▊   | 298098688/440473133 [00:09<00:03, 36290564.30B/s] 69%|██████▊   | 301801472/440473133 [00:09<00:04, 34413371.75B/s] 70%|██████▉   | 306268160/440473133 [00:09<00:03, 36958261.14B/s] 70%|███████   | 310068224/440473133 [00:09<00:03, 35158112.73B/s] 71%|███████▏  | 314210304/440473133 [00:09<00:03, 34483189.20B/s] 72%|███████▏  | 318482432/440473133 [00:09<00:03, 36599418.54B/s] 73%|███████▎  | 322225152/440473133 [00:09<00:03, 34787624.46B/s] 74%|███████▍  | 326658048/440473133 [00:09<00:03, 37187921.85B/s] 75%|███████▌  | 330476544/440473133 [00:10<00:03, 35286652.83B/s] 76%|███████▌  | 334772224/440473133 [00:10<00:03, 34752716.37B/s] 77%|███████▋  | 339106816/440473133 [00:10<00:02, 36949681.91B/s] 78%|███████▊  | 342885376/440473133 [00:10<00:02, 34961981.53B/s] 79%|███████▉  | 347612160/440473133 [00:10<00:02, 37923483.98B/s] 80%|███████▉  | 351534080/440473133 [00:10<00:02, 35829263.25B/s] 81%|████████  | 355645440/440473133 [00:10<00:02, 34831894.80B/s] 82%|████████▏ | 359972864/440473133 [00:10<00:02, 36997075.31B/s] 83%|████████▎ | 363769856/440473133 [00:11<00:02, 35226533.16B/s] 83%|████████▎ | 367759360/440473133 [00:11<00:01, 36507574.88B/s] 84%|████████▍ | 371482624/440473133 [00:11<00:01, 34606483.75B/s] 85%|████████▌ | 376010752/440473133 [00:11<00:01, 34901530.60B/s] 86%|████████▋ | 380407808/440473133 [00:11<00:01, 37203060.62B/s] 87%|████████▋ | 384205824/440473133 [00:11<00:01, 35400677.63B/s] 88%|████████▊ | 388416512/440473133 [00:11<00:01, 37176927.81B/s] 89%|████████▉ | 392208384/440473133 [00:11<00:01, 35232147.31B/s] 90%|█████████ | 396587008/440473133 [00:11<00:01, 37336509.58B/s] 91%|█████████ | 400406528/440473133 [00:12<00:01, 35936729.00B/s] 92%|█████████▏| 404617216/440473133 [00:12<00:01, 35113070.49B/s] 93%|█████████▎| 408987648/440473133 [00:12<00:00, 37312752.07B/s] 94%|█████████▎| 412794880/440473133 [00:12<00:00, 35353010.77B/s] 95%|█████████▍| 417078272/440473133 [00:12<00:00, 37307424.64B/s] 96%|█████████▌| 420890624/440473133 [00:12<00:00, 35127291.12B/s] 96%|█████████▋| 424998912/440473133 [00:12<00:00, 34087268.38B/s] 97%|█████████▋| 429029376/440473133 [00:12<00:00, 35741103.33B/s] 98%|█████████▊| 432670720/440473133 [00:12<00:00, 33300570.30B/s] 99%|█████████▉| 436762624/440473133 [00:13<00:00, 32714012.38B/s]100%|██████████| 440473133/440473133 [00:13<00:00, 33405810.00B/s]Downloading data from https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip

   8192/7094233 [..............................] - ETA: 0s
1064960/7094233 [===>..........................] - ETA: 0s
2105344/7094233 [=======>......................] - ETA: 0s
3145728/7094233 [============>.................] - ETA: 0s
4186112/7094233 [================>.............] - ETA: 0s
5234688/7094233 [=====================>........] - ETA: 0s
6275072/7094233 [=========================>....] - ETA: 0s
7094272/7094233 [==============================] - 0s 0us/step

Processing text_left with encode:   0%|          | 0/2118 [00:00<?, ?it/s]Processing text_left with encode:   4%|▍         | 80/2118 [00:00<00:02, 799.64it/s]Processing text_left with encode:  24%|██▍       | 518/2118 [00:00<00:01, 1059.38it/s]Processing text_left with encode:  45%|████▍     | 947/2118 [00:00<00:00, 1349.99it/s]Processing text_left with encode:  66%|██████▋   | 1408/2118 [00:00<00:00, 1713.33it/s]Processing text_left with encode:  88%|████████▊ | 1865/2118 [00:00<00:00, 2108.01it/s]Processing text_left with encode: 100%|██████████| 2118/2118 [00:00<00:00, 3712.82it/s]
Processing text_right with encode:   0%|          | 0/18841 [00:00<?, ?it/s]Processing text_right with encode:   1%|          | 150/18841 [00:00<00:16, 1140.52it/s]Processing text_right with encode:   2%|▏         | 317/18841 [00:00<00:14, 1259.73it/s]Processing text_right with encode:   3%|▎         | 482/18841 [00:00<00:13, 1355.78it/s]Processing text_right with encode:   4%|▎         | 662/18841 [00:00<00:12, 1460.67it/s]Processing text_right with encode:   4%|▍         | 827/18841 [00:00<00:11, 1507.28it/s]Processing text_right with encode:   5%|▌         | 989/18841 [00:00<00:11, 1537.88it/s]Processing text_right with encode:   6%|▌         | 1152/18841 [00:00<00:11, 1562.11it/s]Processing text_right with encode:   7%|▋         | 1315/18841 [00:00<00:11, 1581.71it/s]Processing text_right with encode:   8%|▊         | 1472/18841 [00:00<00:11, 1577.55it/s]Processing text_right with encode:   9%|▊         | 1638/18841 [00:01<00:10, 1600.53it/s]Processing text_right with encode:  10%|▉         | 1807/18841 [00:01<00:10, 1624.64it/s]Processing text_right with encode:  10%|█         | 1971/18841 [00:01<00:10, 1628.38it/s]Processing text_right with encode:  11%|█▏        | 2142/18841 [00:01<00:10, 1651.85it/s]Processing text_right with encode:  12%|█▏        | 2310/18841 [00:01<00:09, 1657.69it/s]Processing text_right with encode:  13%|█▎        | 2476/18841 [00:01<00:09, 1648.82it/s]Processing text_right with encode:  14%|█▍        | 2641/18841 [00:01<00:09, 1640.39it/s]Processing text_right with encode:  15%|█▌        | 2834/18841 [00:01<00:09, 1717.69it/s]Processing text_right with encode:  16%|█▌        | 3007/18841 [00:01<00:09, 1685.38it/s]Processing text_right with encode:  17%|█▋        | 3177/18841 [00:01<00:09, 1668.87it/s]Processing text_right with encode:  18%|█▊        | 3347/18841 [00:02<00:09, 1676.85it/s]Processing text_right with encode:  19%|█▊        | 3516/18841 [00:02<00:09, 1609.59it/s]Processing text_right with encode:  20%|█▉        | 3678/18841 [00:02<00:09, 1587.21it/s]Processing text_right with encode:  20%|██        | 3844/18841 [00:02<00:09, 1606.92it/s]Processing text_right with encode:  21%|██▏       | 4025/18841 [00:02<00:08, 1662.51it/s]Processing text_right with encode:  22%|██▏       | 4193/18841 [00:02<00:08, 1638.08it/s]Processing text_right with encode:  23%|██▎       | 4358/18841 [00:02<00:09, 1603.53it/s]Processing text_right with encode:  24%|██▍       | 4519/18841 [00:02<00:08, 1599.81it/s]Processing text_right with encode:  25%|██▍       | 4692/18841 [00:02<00:08, 1630.99it/s]Processing text_right with encode:  26%|██▌       | 4871/18841 [00:02<00:08, 1669.07it/s]Processing text_right with encode:  27%|██▋       | 5045/18841 [00:03<00:08, 1688.67it/s]Processing text_right with encode:  28%|██▊       | 5226/18841 [00:03<00:07, 1722.91it/s]Processing text_right with encode:  29%|██▊       | 5399/18841 [00:03<00:07, 1724.03it/s]Processing text_right with encode:  30%|██▉       | 5572/18841 [00:03<00:07, 1683.83it/s]Processing text_right with encode:  30%|███       | 5741/18841 [00:03<00:07, 1676.92it/s]Processing text_right with encode:  31%|███▏      | 5910/18841 [00:03<00:07, 1674.53it/s]Processing text_right with encode:  32%|███▏      | 6078/18841 [00:03<00:07, 1673.27it/s]Processing text_right with encode:  33%|███▎      | 6246/18841 [00:03<00:07, 1655.77it/s]Processing text_right with encode:  34%|███▍      | 6420/18841 [00:03<00:07, 1678.75it/s]Processing text_right with encode:  35%|███▍      | 6590/18841 [00:04<00:07, 1683.36it/s]Processing text_right with encode:  36%|███▌      | 6762/18841 [00:04<00:07, 1686.70it/s]Processing text_right with encode:  37%|███▋      | 6931/18841 [00:04<00:07, 1675.68it/s]Processing text_right with encode:  38%|███▊      | 7099/18841 [00:04<00:07, 1631.08it/s]Processing text_right with encode:  39%|███▊      | 7272/18841 [00:04<00:06, 1657.74it/s]Processing text_right with encode:  40%|███▉      | 7452/18841 [00:04<00:06, 1696.07it/s]Processing text_right with encode:  40%|████      | 7623/18841 [00:04<00:06, 1652.10it/s]Processing text_right with encode:  41%|████▏     | 7793/18841 [00:04<00:06, 1663.47it/s]Processing text_right with encode:  42%|████▏     | 7962/18841 [00:04<00:06, 1668.19it/s]Processing text_right with encode:  43%|████▎     | 8131/18841 [00:04<00:06, 1674.07it/s]Processing text_right with encode:  44%|████▍     | 8299/18841 [00:05<00:06, 1666.71it/s]Processing text_right with encode:  45%|████▍     | 8466/18841 [00:05<00:06, 1659.77it/s]Processing text_right with encode:  46%|████▌     | 8633/18841 [00:05<00:06, 1648.10it/s]Processing text_right with encode:  47%|████▋     | 8800/18841 [00:05<00:06, 1650.70it/s]Processing text_right with encode:  48%|████▊     | 8966/18841 [00:05<00:06, 1631.90it/s]Processing text_right with encode:  48%|████▊     | 9130/18841 [00:05<00:05, 1630.87it/s]Processing text_right with encode:  49%|████▉     | 9296/18841 [00:05<00:05, 1638.07it/s]Processing text_right with encode:  50%|█████     | 9463/18841 [00:05<00:05, 1645.50it/s]Processing text_right with encode:  51%|█████     | 9631/18841 [00:05<00:05, 1653.07it/s]Processing text_right with encode:  52%|█████▏    | 9797/18841 [00:05<00:05, 1636.31it/s]Processing text_right with encode:  53%|█████▎    | 9972/18841 [00:06<00:05, 1668.80it/s]Processing text_right with encode:  54%|█████▍    | 10145/18841 [00:06<00:05, 1683.03it/s]Processing text_right with encode:  55%|█████▍    | 10314/18841 [00:06<00:05, 1645.12it/s]Processing text_right with encode:  56%|█████▌    | 10513/18841 [00:06<00:04, 1734.75it/s]Processing text_right with encode:  57%|█████▋    | 10688/18841 [00:06<00:04, 1718.77it/s]Processing text_right with encode:  58%|█████▊    | 10861/18841 [00:06<00:04, 1711.68it/s]Processing text_right with encode:  59%|█████▊    | 11033/18841 [00:06<00:04, 1665.18it/s]Processing text_right with encode:  59%|█████▉    | 11201/18841 [00:06<00:04, 1614.70it/s]Processing text_right with encode:  60%|██████    | 11367/18841 [00:06<00:04, 1627.66it/s]Processing text_right with encode:  61%|██████    | 11533/18841 [00:06<00:04, 1633.96it/s]Processing text_right with encode:  62%|██████▏   | 11701/18841 [00:07<00:04, 1645.49it/s]Processing text_right with encode:  63%|██████▎   | 11866/18841 [00:07<00:04, 1644.65it/s]Processing text_right with encode:  64%|██████▍   | 12038/18841 [00:07<00:04, 1663.71it/s]Processing text_right with encode:  65%|██████▍   | 12205/18841 [00:07<00:04, 1654.51it/s]Processing text_right with encode:  66%|██████▌   | 12381/18841 [00:07<00:03, 1683.24it/s]Processing text_right with encode:  67%|██████▋   | 12550/18841 [00:07<00:03, 1669.27it/s]Processing text_right with encode:  68%|██████▊   | 12719/18841 [00:07<00:03, 1673.35it/s]Processing text_right with encode:  68%|██████▊   | 12887/18841 [00:07<00:03, 1660.92it/s]Processing text_right with encode:  69%|██████▉   | 13064/18841 [00:07<00:03, 1690.41it/s]Processing text_right with encode:  70%|███████   | 13234/18841 [00:08<00:03, 1666.32it/s]Processing text_right with encode:  71%|███████   | 13407/18841 [00:08<00:03, 1684.34it/s]Processing text_right with encode:  72%|███████▏  | 13582/18841 [00:08<00:03, 1701.90it/s]Processing text_right with encode:  73%|███████▎  | 13767/18841 [00:08<00:02, 1742.84it/s]Processing text_right with encode:  74%|███████▍  | 13942/18841 [00:08<00:02, 1724.79it/s]Processing text_right with encode:  75%|███████▍  | 14115/18841 [00:08<00:02, 1691.97it/s]Processing text_right with encode:  76%|███████▌  | 14285/18841 [00:08<00:02, 1683.88it/s]Processing text_right with encode:  77%|███████▋  | 14454/18841 [00:08<00:02, 1650.92it/s]Processing text_right with encode:  78%|███████▊  | 14629/18841 [00:08<00:02, 1677.75it/s]Processing text_right with encode:  79%|███████▊  | 14810/18841 [00:08<00:02, 1711.37it/s]Processing text_right with encode:  80%|███████▉  | 14982/18841 [00:09<00:02, 1692.63it/s]Processing text_right with encode:  80%|████████  | 15157/18841 [00:09<00:02, 1706.26it/s]Processing text_right with encode:  81%|████████▏ | 15328/18841 [00:09<00:02, 1683.31it/s]Processing text_right with encode:  82%|████████▏ | 15497/18841 [00:09<00:01, 1680.12it/s]Processing text_right with encode:  83%|████████▎ | 15672/18841 [00:09<00:01, 1699.15it/s]Processing text_right with encode:  84%|████████▍ | 15847/18841 [00:09<00:01, 1710.99it/s]Processing text_right with encode:  85%|████████▌ | 16019/18841 [00:09<00:01, 1686.27it/s]Processing text_right with encode:  86%|████████▌ | 16188/18841 [00:09<00:01, 1669.27it/s]Processing text_right with encode:  87%|████████▋ | 16356/18841 [00:09<00:01, 1671.49it/s]Processing text_right with encode:  88%|████████▊ | 16525/18841 [00:09<00:01, 1676.38it/s]Processing text_right with encode:  89%|████████▊ | 16693/18841 [00:10<00:01, 1663.78it/s]Processing text_right with encode:  89%|████████▉ | 16860/18841 [00:10<00:01, 1650.80it/s]Processing text_right with encode:  90%|█████████ | 17026/18841 [00:10<00:01, 1645.45it/s]Processing text_right with encode:  91%|█████████ | 17191/18841 [00:10<00:01, 1643.28it/s]Processing text_right with encode:  92%|█████████▏| 17364/18841 [00:10<00:00, 1665.46it/s]Processing text_right with encode:  93%|█████████▎| 17531/18841 [00:10<00:00, 1623.15it/s]Processing text_right with encode:  94%|█████████▍| 17714/18841 [00:10<00:00, 1677.98it/s]Processing text_right with encode:  95%|█████████▍| 17883/18841 [00:10<00:00, 1663.48it/s]Processing text_right with encode:  96%|█████████▌| 18077/18841 [00:10<00:00, 1736.79it/s]Processing text_right with encode:  97%|█████████▋| 18252/18841 [00:10<00:00, 1711.71it/s]Processing text_right with encode:  98%|█████████▊| 18437/18841 [00:11<00:00, 1749.58it/s]Processing text_right with encode:  99%|█████████▉| 18615/18841 [00:11<00:00, 1757.91it/s]Processing text_right with encode: 100%|█████████▉| 18792/18841 [00:11<00:00, 1736.06it/s]Processing text_right with encode: 100%|██████████| 18841/18841 [00:11<00:00, 1666.05it/s]
Processing length_left with len:   0%|          | 0/2118 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 2118/2118 [00:00<00:00, 583636.81it/s]
Processing length_right with len:   0%|          | 0/18841 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 18841/18841 [00:00<00:00, 747018.84it/s]
Processing text_left with encode:   0%|          | 0/633 [00:00<?, ?it/s]Processing text_left with encode:  72%|███████▏  | 456/633 [00:00<00:00, 4554.73it/s]Processing text_left with encode: 100%|██████████| 633/633 [00:00<00:00, 4507.80it/s]
Processing text_right with encode:   0%|          | 0/5961 [00:00<?, ?it/s]Processing text_right with encode:   3%|▎         | 175/5961 [00:00<00:03, 1745.93it/s]Processing text_right with encode:   6%|▌         | 346/5961 [00:00<00:03, 1731.39it/s]Processing text_right with encode:   9%|▊         | 512/5961 [00:00<00:03, 1706.05it/s]Processing text_right with encode:  11%|█▏        | 685/5961 [00:00<00:03, 1711.71it/s]Processing text_right with encode:  14%|█▍        | 862/5961 [00:00<00:02, 1728.13it/s]Processing text_right with encode:  17%|█▋        | 1041/5961 [00:00<00:02, 1744.29it/s]Processing text_right with encode:  20%|██        | 1215/5961 [00:00<00:02, 1742.73it/s]Processing text_right with encode:  23%|██▎       | 1395/5961 [00:00<00:02, 1759.22it/s]Processing text_right with encode:  26%|██▋       | 1573/5961 [00:00<00:02, 1764.69it/s]Processing text_right with encode:  29%|██▉       | 1743/5961 [00:01<00:02, 1676.44it/s]Processing text_right with encode:  32%|███▏      | 1922/5961 [00:01<00:02, 1705.46it/s]Processing text_right with encode:  35%|███▌      | 2095/5961 [00:01<00:02, 1710.76it/s]Processing text_right with encode:  38%|███▊      | 2271/5961 [00:01<00:02, 1723.85it/s]Processing text_right with encode:  41%|████      | 2443/5961 [00:01<00:02, 1680.41it/s]Processing text_right with encode:  44%|████▍     | 2628/5961 [00:01<00:01, 1727.54it/s]Processing text_right with encode:  47%|████▋     | 2821/5961 [00:01<00:01, 1781.18it/s]Processing text_right with encode:  50%|█████     | 3000/5961 [00:01<00:01, 1723.48it/s]Processing text_right with encode:  53%|█████▎    | 3182/5961 [00:01<00:01, 1749.29it/s]Processing text_right with encode:  56%|█████▋    | 3358/5961 [00:01<00:01, 1695.24it/s]Processing text_right with encode:  59%|█████▉    | 3529/5961 [00:02<00:01, 1694.11it/s]Processing text_right with encode:  62%|██████▏   | 3703/5961 [00:02<00:01, 1704.72it/s]Processing text_right with encode:  65%|██████▌   | 3877/5961 [00:02<00:01, 1714.37it/s]Processing text_right with encode:  68%|██████▊   | 4064/5961 [00:02<00:01, 1755.90it/s]Processing text_right with encode:  71%|███████   | 4241/5961 [00:02<00:00, 1734.41it/s]Processing text_right with encode:  74%|███████▍  | 4415/5961 [00:02<00:00, 1727.24it/s]Processing text_right with encode:  77%|███████▋  | 4597/5961 [00:02<00:00, 1753.89it/s]Processing text_right with encode:  80%|████████  | 4773/5961 [00:02<00:00, 1718.03it/s]Processing text_right with encode:  83%|████████▎ | 4946/5961 [00:02<00:00, 1697.97it/s]Processing text_right with encode:  86%|████████▌ | 5117/5961 [00:02<00:00, 1697.48it/s]Processing text_right with encode:  89%|████████▉ | 5295/5961 [00:03<00:00, 1718.70it/s]Processing text_right with encode:  92%|█████████▏| 5468/5961 [00:03<00:00, 1674.79it/s]Processing text_right with encode:  95%|█████████▍| 5636/5961 [00:03<00:00, 1648.51it/s]Processing text_right with encode:  97%|█████████▋| 5802/5961 [00:03<00:00, 1632.56it/s]Processing text_right with encode: 100%|██████████| 5961/5961 [00:03<00:00, 1717.60it/s]
Processing length_left with len:   0%|          | 0/633 [00:00<?, ?it/s]Processing length_left with len: 100%|██████████| 633/633 [00:00<00:00, 426162.83it/s]
Processing length_right with len:   0%|          | 0/5961 [00:00<?, ?it/s]Processing length_right with len: 100%|██████████| 5961/5961 [00:00<00:00, 623435.22it/s]
  #### Model  fit   ############################################# 

  0%|          | 0/102 [00:00<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:48<?, ?it/s]Epoch 1/1:   0%|          | 0/102 [00:48<?, ?it/s, loss=0.789]Epoch 1/1:   1%|          | 1/102 [00:48<1:20:55, 48.08s/it, loss=0.789]Killed

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
   a2fc9d9..5a6dbc8  master     -> origin/master
Updating a2fc9d9..5a6dbc8
Fast-forward
 deps.txt                                           |   6 +-
 error_list/20200520/list_log_testall_20200520.md   |   9 +
 ...-10_69b309ad857428cc5a734b8afd99842edf9b2a42.py | 625 +++++++++++++++++++++
 3 files changed, 639 insertions(+), 1 deletion(-)
 create mode 100644 log_pullrequest/log_pr_2020-05-20-09-10_69b309ad857428cc5a734b8afd99842edf9b2a42.py
[master 19cf9d5] ml_store
 1 file changed, 72 insertions(+)
To github.com:arita37/mlmodels_store.git
   5a6dbc8..19cf9d5  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_tch//torchhub.py 

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels/preprocess/generic.py::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb5d03fde18>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb5d03fde18>

  function with postional parmater data_info <function get_dataset_torch at 0x7fb5d03fde18> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s]  0%|          | 49152/9912422 [00:00<00:34, 288624.11it/s]  2%|▏         | 212992/9912422 [00:00<00:25, 375503.76it/s]  9%|▉         | 876544/9912422 [00:00<00:17, 519910.46it/s] 35%|███▌      | 3473408/9912422 [00:00<00:08, 735715.27it/s] 69%|██████▉   | 6873088/9912422 [00:00<00:02, 1041343.29it/s] 95%|█████████▌| 9420800/9912422 [00:01<00:00, 1462011.59it/s]9920512it [00:01, 9693053.73it/s]                             
0it [00:00, ?it/s]32768it [00:00, 368088.13it/s]
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:05, 294013.48it/s] 13%|█▎        | 212992/1648877 [00:00<00:03, 381060.09it/s] 53%|█████▎    | 876544/1648877 [00:00<00:01, 527183.88it/s]1654784it [00:00, 2651022.99it/s]                           
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 20884.26it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to mlmodels/dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb5d0173b70>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb5d0173b70>

  function with postional parmater data_info <function get_dataset_torch at 0x7fb5d0173b70> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Train Epoch: 1 	 Loss: 0.002041345477104187 	 Accuracy: 0
Train Epoch: 1 	 Loss: 0.011165170669555665 	 Accuracy: 0
model saves at 0 accuracy
Train Epoch: 2 	 Loss: 0.0015532450874646505 	 Accuracy: 0
Train Epoch: 2 	 Loss: 0.010967649817466736 	 Accuracy: 1
model saves at 1 accuracy

  #### Predict   ##################################################### 

  URL:  mlmodels/preprocess/generic.py::get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb5d0173950>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb5d0173950>

  function with postional parmater data_info <function get_dataset_torch at 0x7fb5d0173950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  #### metrics   ##################################################### 
None

  #### Plot   ######################################################## 

  #### Save  ######################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 

  #### Load   ######################################################## 
<__main__.Model object at 0x7fb5d01b39e8>

  #### Loading params   ############################################## 

  {'data_info': {'data_path': 'mlmodels/dataset/vision/MNIST', 'dataset': 'MNIST', 'data_type': 'tch_dataset', 'batch_size': 10, 'train': True}, 'preprocessors': [{'name': 'tch_dataset_start', 'uri': 'mlmodels.preprocess.generic::get_dataset_torch', 'args': {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}}, 'shuffle': True, 'download': True}}]} {'checkpointdir': 'ztest/model_tch/torchhub/restnet18/checkpoints/', 'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18/'} 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### Predict   ##################################################### 
img_01.png
torch_model

  #### metrics   ##################################################### 

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 

  /home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_tch/torchhub/restnet18//torch_model/ ['model.pb', 'torch_model_pars.pkl'] 
img_01.png
torch_model

Downloading: "https://github.com/pytorch/vision/archive/master.zip" to /home/runner/.cache/torch/hub/master.zip
Using cache found in /home/runner/.cache/torch/hub/pytorch_vision_master
Downloading: "https://github.com/facebookresearch/pytorch_GAN_zoo/archive/hub.zip" to /home/runner/.cache/torch/hub/hub.zip
Using cache found in /home/runner/.cache/torch/hub/facebookresearch_pytorch_GAN_zoo_hub
<__main__.Model object at 0x7fb5c7ef2320>

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
[master a39ad62] ml_store
 2 files changed, 150 insertions(+), 5 deletions(-)
To github.com:arita37/mlmodels_store.git
   19cf9d5..a39ad62  master -> master





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
