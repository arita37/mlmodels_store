
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
[master 2d805c3] ml_store  && git pull --all
 2 files changed, 62 insertions(+), 10325 deletions(-)
 rewrite log_testall/log_testall.py (99%)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + 5923b89...2d805c3 master -> master (forced update)





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
[master 7b810d4] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   2d805c3..7b810d4  master -> master





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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         8           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         8           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-25 16:14:41.062685: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-25 16:14:41.068589: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-25 16:14:41.068773: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f7ede61f70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 16:14:41.068791: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6931500/500 [==============================] - 1s 1ms/sample - loss: 0.2510 - binary_crossentropy: 0.7212 - val_loss: 0.2492 - val_binary_crossentropy: 0.6906

  #### metrics   #################################################### 
{'MSE': 0.24982472731104174}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 5, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 1, 7)         0           no_mask[0][0]                    
                                                                 no_mask[1][0]                    
                                                                 no_mask[2][0]                    
                                                                 no_mask[3][0]                    
                                                                 no_mask[4][0]                    
                                                                 no_mask[5][0]                    
                                                                 no_mask[6][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
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
Total params: 453
Trainable params: 453
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.5100 - binary_crossentropy: 7.8667500/500 [==============================] - 1s 2ms/sample - loss: 0.5220 - binary_crossentropy: 8.0518 - val_loss: 0.4560 - val_binary_crossentropy: 7.0338

  #### metrics   #################################################### 
{'MSE': 0.489}

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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         4           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         1           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 672
Trainable params: 672
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 2ms/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2501 - val_binary_crossentropy: 0.6933

  #### metrics   #################################################### 
{'MSE': 0.24984335928218518}

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
sequence_sum (InputLayer)       [(None, 5)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 672
Trainable params: 672
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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 483
Trainable params: 483
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3000 - binary_crossentropy: 0.8524500/500 [==============================] - 1s 3ms/sample - loss: 0.3233 - binary_crossentropy: 0.8961 - val_loss: 0.2996 - val_binary_crossentropy: 0.8279

  #### metrics   #################################################### 
{'MSE': 0.3087391651799283}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 4, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         16          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         32          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_0[0][0]           
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
Total params: 483
Trainable params: 483
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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
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
Total params: 178
Trainable params: 178
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2548 - binary_crossentropy: 0.8303500/500 [==============================] - 2s 3ms/sample - loss: 0.2577 - binary_crossentropy: 0.7863 - val_loss: 0.2715 - val_binary_crossentropy: 0.8939

  #### metrics   #################################################### 
{'MSE': 0.2646359259726304}

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
sequence_sum (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 2, 4)         28          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 2, 1)         7           sequence_max[0][0]               
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
Total params: 178
Trainable params: 178
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-25 16:16:06.707807: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:06.710051: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:06.716014: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 16:16:06.727298: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 16:16:06.728996: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 16:16:06.731137: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:06.732886: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2472 - val_binary_crossentropy: 0.6875
2020-05-25 16:16:08.110593: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:08.112443: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:08.117125: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 16:16:08.129576: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 16:16:08.132502: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 16:16:08.134419: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:08.136029: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24598195753835364}

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
2020-05-25 16:16:33.548587: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:33.550153: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:33.554058: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 16:16:33.562565: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 16:16:33.563794: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 16:16:33.564859: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:33.565973: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2491 - val_binary_crossentropy: 0.6913
2020-05-25 16:16:35.231129: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:35.232442: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:35.235925: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 16:16:35.242432: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 16:16:35.243426: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 16:16:35.244339: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:16:35.245302: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24869030033635653}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-25 16:17:12.324444: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:17:12.330409: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:17:12.347707: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 16:17:12.374846: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 16:17:12.379598: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 16:17:12.383960: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:17:12.388441: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.1195 - binary_crossentropy: 0.4241 - val_loss: 0.2682 - val_binary_crossentropy: 0.7310
2020-05-25 16:17:14.796644: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:17:14.801935: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:17:14.817265: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 16:17:14.845863: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 16:17:14.850593: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 16:17:14.856244: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 16:17:14.860778: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2232774706897954}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
100/500 [=====>........................] - ETA: 7s - loss: 0.2576 - binary_crossentropy: 0.7089500/500 [==============================] - 5s 9ms/sample - loss: 0.2620 - binary_crossentropy: 0.7181 - val_loss: 0.2576 - val_binary_crossentropy: 0.7088

  #### metrics   #################################################### 
{'MSE': 0.25916142975044076}

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
sequence_mean (InputLayer)      [(None, 4)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 4)]          0                                            
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
sparse_seq_emb_sequence_mean (E (None, 4, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 4, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         32          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 4, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         8           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         10          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 2)         8           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         8           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_5[0][0]           
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
Total params: 209
Trainable params: 209
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2875 - binary_crossentropy: 0.7751500/500 [==============================] - 5s 10ms/sample - loss: 0.2717 - binary_crossentropy: 0.7403 - val_loss: 0.2546 - val_binary_crossentropy: 0.7026

  #### metrics   #################################################### 
{'MSE': 0.2547906703553367}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 9, 2)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         10          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 2)         8           sequence_max[0][0]               
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
sparse_emb_sparse_feature_3 (Em (None, 1, 2)         8           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         8           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         12          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         8           sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_3[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_5[0][0]           
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
Total params: 209
Trainable params: 209
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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_max[0][0]               
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
Total params: 1,924
Trainable params: 1,924
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2465 - binary_crossentropy: 0.6866500/500 [==============================] - 5s 10ms/sample - loss: 0.2527 - binary_crossentropy: 0.6987 - val_loss: 0.2512 - val_binary_crossentropy: 0.7220

  #### metrics   #################################################### 
{'MSE': 0.2502737437185591}

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
sequence_mean (InputLayer)      [(None, 3)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 8)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_max[0][0]               
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
regionsequence_sum (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         8           regionsequence_max[0][0]         
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
100/500 [=====>........................] - ETA: 9s - loss: 0.2452 - binary_crossentropy: 0.6833500/500 [==============================] - 6s 13ms/sample - loss: 0.2512 - binary_crossentropy: 0.6953 - val_loss: 0.2549 - val_binary_crossentropy: 0.7029

  #### metrics   #################################################### 
{'MSE': 0.2528388745871153}

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
regionsequence_sum (InputLayer) [(None, 4)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 8)]          0                                            
__________________________________________________________________________________________________
regionsequence_max (InputLayer) [(None, 6)]          0                                            
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
region_10sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 6, 1)         8           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 4, 1)         1           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 8, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 6, 1)         8           regionsequence_max[0][0]         
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
sequence_sum (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
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
Total params: 1,397
Trainable params: 1,397
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.3164 - binary_crossentropy: 0.8674500/500 [==============================] - 6s 13ms/sample - loss: 0.3309 - binary_crossentropy: 0.9548 - val_loss: 0.3035 - val_binary_crossentropy: 0.8325

  #### metrics   #################################################### 
{'MSE': 0.3139787233166116}

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
sequence_mean (InputLayer)      [(None, 9)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sparse_seq_emb_weighted_seq (Em (None, 3, 4)         8           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_40 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 7, 4)         16          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 9, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         8           sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 7, 1)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_max[0][0]               
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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
Total params: 3,052
Trainable params: 2,972
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2607 - binary_crossentropy: 0.7149500/500 [==============================] - 7s 14ms/sample - loss: 0.2591 - binary_crossentropy: 0.7117 - val_loss: 0.2536 - val_binary_crossentropy: 0.7003

  #### metrics   #################################################### 
{'MSE': 0.25557959629501853}

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
sequence_sum (InputLayer)       [(None, 4)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
hash_18 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_19 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_20 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
hash_21 (Hash)                  (None, 1)            0           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         24          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         24          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 4, 4)         12          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         12          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 1, 4)         16          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 9, 4)         12          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 4, 1)         3           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         3           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
[master bbf5bd1] ml_store  && git pull --all
 1 file changed, 4945 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + fe46582...bbf5bd1 master -> master (forced update)





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
[master 2246b6e] ml_store  && git pull --all
 1 file changed, 50 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   bbf5bd1..2246b6e  master -> master





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
[master 1f185d6] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   2246b6e..1f185d6  master -> master





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
[master 66aeed1] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   1f185d6..66aeed1  master -> master





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
[master 6777345] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   66aeed1..6777345  master -> master





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
[master 6cebaa9] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   6777345..6cebaa9  master -> master





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
[master 2302197] ml_store  && git pull --all
 1 file changed, 44 insertions(+)
To github.com:arita37/mlmodels_store.git
   6cebaa9..2302197  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 2842624/17464789 [===>..........................] - ETA: 0s
11010048/17464789 [=================>............] - ETA: 0s
16605184/17464789 [===========================>..] - ETA: 0s
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
2020-05-25 16:28:03.403454: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-25 16:28:03.408515: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-25 16:28:03.408680: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561c01dd5980 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 16:28:03.408698: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 14s - loss: 7.3600 - accuracy: 0.5200
 2000/25000 [=>............................] - ETA: 10s - loss: 7.4366 - accuracy: 0.5150
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4724 - accuracy: 0.5127 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5133 - accuracy: 0.5100
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5440 - accuracy: 0.5080
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5312 - accuracy: 0.5088
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.4848 - accuracy: 0.5119
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5765 - accuracy: 0.5059
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5797 - accuracy: 0.5057
10000/25000 [===========>..................] - ETA: 5s - loss: 7.5424 - accuracy: 0.5081
11000/25000 [============>.................] - ETA: 4s - loss: 7.5816 - accuracy: 0.5055
12000/25000 [=============>................] - ETA: 4s - loss: 7.5657 - accuracy: 0.5066
13000/25000 [==============>...............] - ETA: 4s - loss: 7.5569 - accuracy: 0.5072
14000/25000 [===============>..............] - ETA: 3s - loss: 7.5779 - accuracy: 0.5058
15000/25000 [=================>............] - ETA: 3s - loss: 7.6043 - accuracy: 0.5041
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6130 - accuracy: 0.5035
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6342 - accuracy: 0.5021
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6445 - accuracy: 0.5014
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6618 - accuracy: 0.5003
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6513 - accuracy: 0.5010
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6659 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6687 - accuracy: 0.4999
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6673 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6628 - accuracy: 0.5002
25000/25000 [==============================] - 10s 396us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7fbcfc6a81d0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7fbcffcdc470> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 13s - loss: 7.5440 - accuracy: 0.5080
 2000/25000 [=>............................] - ETA: 9s - loss: 7.6206 - accuracy: 0.5030 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.6768 - accuracy: 0.4993
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6436 - accuracy: 0.5015
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.6666 - accuracy: 0.5000
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7101 - accuracy: 0.4972
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.7061 - accuracy: 0.4974
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7088 - accuracy: 0.4972
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.7331 - accuracy: 0.4957
10000/25000 [===========>..................] - ETA: 5s - loss: 7.7280 - accuracy: 0.4960
11000/25000 [============>.................] - ETA: 4s - loss: 7.7043 - accuracy: 0.4975
12000/25000 [=============>................] - ETA: 4s - loss: 7.7228 - accuracy: 0.4963
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7126 - accuracy: 0.4970
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7115 - accuracy: 0.4971
15000/25000 [=================>............] - ETA: 3s - loss: 7.6901 - accuracy: 0.4985
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6657 - accuracy: 0.5001
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6504 - accuracy: 0.5011
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6462 - accuracy: 0.5013
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6432 - accuracy: 0.5015
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6383 - accuracy: 0.5019
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6411 - accuracy: 0.5017
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6415 - accuracy: 0.5016
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6460 - accuracy: 0.5013
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6698 - accuracy: 0.4998
25000/25000 [==============================] - 10s 396us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
 2000/25000 [=>............................] - ETA: 10s - loss: 7.6896 - accuracy: 0.4985
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.5491 - accuracy: 0.5077 
 4000/25000 [===>..........................] - ETA: 8s - loss: 7.5248 - accuracy: 0.5092
 5000/25000 [=====>........................] - ETA: 7s - loss: 7.5685 - accuracy: 0.5064
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.5951 - accuracy: 0.5047
 7000/25000 [=======>......................] - ETA: 6s - loss: 7.6184 - accuracy: 0.5031
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5689 - accuracy: 0.5064
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.5883 - accuracy: 0.5051
10000/25000 [===========>..................] - ETA: 5s - loss: 7.6421 - accuracy: 0.5016
11000/25000 [============>.................] - ETA: 4s - loss: 7.6527 - accuracy: 0.5009
12000/25000 [=============>................] - ETA: 4s - loss: 7.6590 - accuracy: 0.5005
13000/25000 [==============>...............] - ETA: 4s - loss: 7.6867 - accuracy: 0.4987
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6940 - accuracy: 0.4982
15000/25000 [=================>............] - ETA: 3s - loss: 7.6615 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 3s - loss: 7.6666 - accuracy: 0.5000
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6630 - accuracy: 0.5002
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6598 - accuracy: 0.5004
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6723 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6605 - accuracy: 0.5004
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6593 - accuracy: 0.5005
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6792 - accuracy: 0.4992
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6620 - accuracy: 0.5003
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6558 - accuracy: 0.5007
25000/25000 [==============================] - 10s 401us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master 53c5720] ml_store  && git pull --all
 1 file changed, 316 insertions(+)
To github.com:arita37/mlmodels_store.git
   2302197..53c5720  master -> master





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

13/13 [==============================] - 2s 131ms/step - loss: nan
Epoch 2/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 3/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 4/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 5/10

13/13 [==============================] - 0s 5ms/step - loss: nan
Epoch 6/10

13/13 [==============================] - 0s 4ms/step - loss: nan
Epoch 7/10

13/13 [==============================] - 0s 5ms/step - loss: nan
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
[master 1ca7415] ml_store  && git pull --all
 1 file changed, 124 insertions(+)
To github.com:arita37/mlmodels_store.git
   53c5720..1ca7415  master -> master





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
 1572864/11490434 [===>..........................] - ETA: 0s
 6447104/11490434 [===============>..............] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:53 - loss: 2.2865 - categorical_accuracy: 0.1562
   64/60000 [..............................] - ETA: 4:57 - loss: 2.3594 - categorical_accuracy: 0.1250
   96/60000 [..............................] - ETA: 3:58 - loss: 2.3085 - categorical_accuracy: 0.1458
  128/60000 [..............................] - ETA: 3:26 - loss: 2.2714 - categorical_accuracy: 0.1797
  160/60000 [..............................] - ETA: 3:06 - loss: 2.2349 - categorical_accuracy: 0.2000
  192/60000 [..............................] - ETA: 2:53 - loss: 2.1976 - categorical_accuracy: 0.2292
  224/60000 [..............................] - ETA: 2:45 - loss: 2.1862 - categorical_accuracy: 0.2277
  256/60000 [..............................] - ETA: 2:41 - loss: 2.1492 - categorical_accuracy: 0.2422
  288/60000 [..............................] - ETA: 2:35 - loss: 2.1288 - categorical_accuracy: 0.2535
  320/60000 [..............................] - ETA: 2:31 - loss: 2.1038 - categorical_accuracy: 0.2719
  352/60000 [..............................] - ETA: 2:27 - loss: 2.0837 - categorical_accuracy: 0.2784
  384/60000 [..............................] - ETA: 2:24 - loss: 2.0262 - categorical_accuracy: 0.3099
  416/60000 [..............................] - ETA: 2:21 - loss: 2.0285 - categorical_accuracy: 0.3125
  448/60000 [..............................] - ETA: 2:20 - loss: 1.9839 - categorical_accuracy: 0.3237
  480/60000 [..............................] - ETA: 2:18 - loss: 1.9500 - categorical_accuracy: 0.3417
  512/60000 [..............................] - ETA: 2:17 - loss: 1.9861 - categorical_accuracy: 0.3379
  544/60000 [..............................] - ETA: 2:15 - loss: 1.9647 - categorical_accuracy: 0.3474
  576/60000 [..............................] - ETA: 2:14 - loss: 1.9372 - categorical_accuracy: 0.3542
  608/60000 [..............................] - ETA: 2:13 - loss: 1.9111 - categorical_accuracy: 0.3602
  640/60000 [..............................] - ETA: 2:12 - loss: 1.8852 - categorical_accuracy: 0.3656
  672/60000 [..............................] - ETA: 2:11 - loss: 1.8651 - categorical_accuracy: 0.3765
  704/60000 [..............................] - ETA: 2:10 - loss: 1.8341 - categorical_accuracy: 0.3906
  736/60000 [..............................] - ETA: 2:09 - loss: 1.7994 - categorical_accuracy: 0.4049
  768/60000 [..............................] - ETA: 2:09 - loss: 1.7810 - categorical_accuracy: 0.4076
  800/60000 [..............................] - ETA: 2:08 - loss: 1.7520 - categorical_accuracy: 0.4175
  832/60000 [..............................] - ETA: 2:07 - loss: 1.7215 - categorical_accuracy: 0.4279
  864/60000 [..............................] - ETA: 2:06 - loss: 1.6994 - categorical_accuracy: 0.4387
  896/60000 [..............................] - ETA: 2:06 - loss: 1.6713 - categorical_accuracy: 0.4487
  928/60000 [..............................] - ETA: 2:06 - loss: 1.6362 - categorical_accuracy: 0.4601
  960/60000 [..............................] - ETA: 2:05 - loss: 1.6183 - categorical_accuracy: 0.4656
  992/60000 [..............................] - ETA: 2:05 - loss: 1.5971 - categorical_accuracy: 0.4728
 1024/60000 [..............................] - ETA: 2:04 - loss: 1.5731 - categorical_accuracy: 0.4814
 1056/60000 [..............................] - ETA: 2:04 - loss: 1.5728 - categorical_accuracy: 0.4811
 1088/60000 [..............................] - ETA: 2:04 - loss: 1.5627 - categorical_accuracy: 0.4825
 1120/60000 [..............................] - ETA: 2:03 - loss: 1.5422 - categorical_accuracy: 0.4920
 1152/60000 [..............................] - ETA: 2:03 - loss: 1.5213 - categorical_accuracy: 0.4965
 1184/60000 [..............................] - ETA: 2:03 - loss: 1.5034 - categorical_accuracy: 0.5042
 1216/60000 [..............................] - ETA: 2:02 - loss: 1.4885 - categorical_accuracy: 0.5082
 1248/60000 [..............................] - ETA: 2:02 - loss: 1.4738 - categorical_accuracy: 0.5144
 1280/60000 [..............................] - ETA: 2:01 - loss: 1.4554 - categorical_accuracy: 0.5203
 1312/60000 [..............................] - ETA: 2:01 - loss: 1.4412 - categorical_accuracy: 0.5252
 1344/60000 [..............................] - ETA: 2:01 - loss: 1.4291 - categorical_accuracy: 0.5283
 1376/60000 [..............................] - ETA: 2:00 - loss: 1.4159 - categorical_accuracy: 0.5320
 1408/60000 [..............................] - ETA: 2:00 - loss: 1.3967 - categorical_accuracy: 0.5376
 1440/60000 [..............................] - ETA: 2:00 - loss: 1.3790 - categorical_accuracy: 0.5437
 1472/60000 [..............................] - ETA: 2:00 - loss: 1.3723 - categorical_accuracy: 0.5476
 1504/60000 [..............................] - ETA: 1:59 - loss: 1.3586 - categorical_accuracy: 0.5525
 1536/60000 [..............................] - ETA: 1:59 - loss: 1.3443 - categorical_accuracy: 0.5579
 1568/60000 [..............................] - ETA: 1:59 - loss: 1.3315 - categorical_accuracy: 0.5625
 1600/60000 [..............................] - ETA: 1:59 - loss: 1.3157 - categorical_accuracy: 0.5675
 1632/60000 [..............................] - ETA: 1:59 - loss: 1.3000 - categorical_accuracy: 0.5717
 1664/60000 [..............................] - ETA: 1:58 - loss: 1.2855 - categorical_accuracy: 0.5769
 1696/60000 [..............................] - ETA: 1:58 - loss: 1.2766 - categorical_accuracy: 0.5796
 1728/60000 [..............................] - ETA: 1:58 - loss: 1.2618 - categorical_accuracy: 0.5851
 1760/60000 [..............................] - ETA: 1:57 - loss: 1.2470 - categorical_accuracy: 0.5898
 1792/60000 [..............................] - ETA: 1:57 - loss: 1.2314 - categorical_accuracy: 0.5943
 1824/60000 [..............................] - ETA: 1:57 - loss: 1.2170 - categorical_accuracy: 0.5987
 1856/60000 [..............................] - ETA: 1:56 - loss: 1.2092 - categorical_accuracy: 0.6024
 1888/60000 [..............................] - ETA: 1:56 - loss: 1.2030 - categorical_accuracy: 0.6059
 1920/60000 [..............................] - ETA: 1:56 - loss: 1.1925 - categorical_accuracy: 0.6099
 1952/60000 [..............................] - ETA: 1:56 - loss: 1.1816 - categorical_accuracy: 0.6132
 1984/60000 [..............................] - ETA: 1:56 - loss: 1.1739 - categorical_accuracy: 0.6149
 2016/60000 [>.............................] - ETA: 1:56 - loss: 1.1648 - categorical_accuracy: 0.6166
 2048/60000 [>.............................] - ETA: 1:56 - loss: 1.1546 - categorical_accuracy: 0.6196
 2080/60000 [>.............................] - ETA: 1:56 - loss: 1.1501 - categorical_accuracy: 0.6226
 2112/60000 [>.............................] - ETA: 1:56 - loss: 1.1375 - categorical_accuracy: 0.6269
 2144/60000 [>.............................] - ETA: 1:55 - loss: 1.1285 - categorical_accuracy: 0.6292
 2176/60000 [>.............................] - ETA: 1:55 - loss: 1.1165 - categorical_accuracy: 0.6337
 2208/60000 [>.............................] - ETA: 1:55 - loss: 1.1076 - categorical_accuracy: 0.6359
 2240/60000 [>.............................] - ETA: 1:55 - loss: 1.0975 - categorical_accuracy: 0.6388
 2272/60000 [>.............................] - ETA: 1:55 - loss: 1.0871 - categorical_accuracy: 0.6417
 2304/60000 [>.............................] - ETA: 1:54 - loss: 1.0836 - categorical_accuracy: 0.6424
 2336/60000 [>.............................] - ETA: 1:54 - loss: 1.0781 - categorical_accuracy: 0.6451
 2368/60000 [>.............................] - ETA: 1:54 - loss: 1.0713 - categorical_accuracy: 0.6470
 2400/60000 [>.............................] - ETA: 1:54 - loss: 1.0627 - categorical_accuracy: 0.6500
 2432/60000 [>.............................] - ETA: 1:54 - loss: 1.0553 - categorical_accuracy: 0.6530
 2464/60000 [>.............................] - ETA: 1:54 - loss: 1.0497 - categorical_accuracy: 0.6538
 2496/60000 [>.............................] - ETA: 1:54 - loss: 1.0422 - categorical_accuracy: 0.6567
 2528/60000 [>.............................] - ETA: 1:54 - loss: 1.0333 - categorical_accuracy: 0.6598
 2560/60000 [>.............................] - ETA: 1:54 - loss: 1.0252 - categorical_accuracy: 0.6633
 2592/60000 [>.............................] - ETA: 1:53 - loss: 1.0171 - categorical_accuracy: 0.6655
 2624/60000 [>.............................] - ETA: 1:53 - loss: 1.0076 - categorical_accuracy: 0.6688
 2656/60000 [>.............................] - ETA: 1:53 - loss: 1.0009 - categorical_accuracy: 0.6713
 2688/60000 [>.............................] - ETA: 1:53 - loss: 0.9951 - categorical_accuracy: 0.6722
 2720/60000 [>.............................] - ETA: 1:53 - loss: 0.9908 - categorical_accuracy: 0.6735
 2752/60000 [>.............................] - ETA: 1:53 - loss: 0.9836 - categorical_accuracy: 0.6751
 2784/60000 [>.............................] - ETA: 1:53 - loss: 0.9762 - categorical_accuracy: 0.6771
 2816/60000 [>.............................] - ETA: 1:53 - loss: 0.9672 - categorical_accuracy: 0.6808
 2848/60000 [>.............................] - ETA: 1:52 - loss: 0.9597 - categorical_accuracy: 0.6836
 2880/60000 [>.............................] - ETA: 1:52 - loss: 0.9550 - categorical_accuracy: 0.6847
 2912/60000 [>.............................] - ETA: 1:52 - loss: 0.9549 - categorical_accuracy: 0.6858
 2944/60000 [>.............................] - ETA: 1:52 - loss: 0.9479 - categorical_accuracy: 0.6885
 2976/60000 [>.............................] - ETA: 1:52 - loss: 0.9416 - categorical_accuracy: 0.6902
 3008/60000 [>.............................] - ETA: 1:52 - loss: 0.9332 - categorical_accuracy: 0.6932
 3040/60000 [>.............................] - ETA: 1:52 - loss: 0.9309 - categorical_accuracy: 0.6944
 3072/60000 [>.............................] - ETA: 1:52 - loss: 0.9249 - categorical_accuracy: 0.6969
 3104/60000 [>.............................] - ETA: 1:52 - loss: 0.9197 - categorical_accuracy: 0.6988
 3136/60000 [>.............................] - ETA: 1:52 - loss: 0.9149 - categorical_accuracy: 0.7006
 3168/60000 [>.............................] - ETA: 1:51 - loss: 0.9114 - categorical_accuracy: 0.7014
 3200/60000 [>.............................] - ETA: 1:51 - loss: 0.9090 - categorical_accuracy: 0.7025
 3232/60000 [>.............................] - ETA: 1:51 - loss: 0.9009 - categorical_accuracy: 0.7054
 3264/60000 [>.............................] - ETA: 1:51 - loss: 0.8953 - categorical_accuracy: 0.7077
 3296/60000 [>.............................] - ETA: 1:51 - loss: 0.8917 - categorical_accuracy: 0.7093
 3328/60000 [>.............................] - ETA: 1:51 - loss: 0.8863 - categorical_accuracy: 0.7106
 3360/60000 [>.............................] - ETA: 1:51 - loss: 0.8836 - categorical_accuracy: 0.7122
 3392/60000 [>.............................] - ETA: 1:51 - loss: 0.8781 - categorical_accuracy: 0.7146
 3424/60000 [>.............................] - ETA: 1:51 - loss: 0.8747 - categorical_accuracy: 0.7158
 3456/60000 [>.............................] - ETA: 1:50 - loss: 0.8697 - categorical_accuracy: 0.7179
 3488/60000 [>.............................] - ETA: 1:50 - loss: 0.8640 - categorical_accuracy: 0.7199
 3520/60000 [>.............................] - ETA: 1:50 - loss: 0.8609 - categorical_accuracy: 0.7216
 3552/60000 [>.............................] - ETA: 1:50 - loss: 0.8608 - categorical_accuracy: 0.7216
 3584/60000 [>.............................] - ETA: 1:50 - loss: 0.8551 - categorical_accuracy: 0.7238
 3616/60000 [>.............................] - ETA: 1:50 - loss: 0.8507 - categorical_accuracy: 0.7251
 3648/60000 [>.............................] - ETA: 1:50 - loss: 0.8476 - categorical_accuracy: 0.7259
 3680/60000 [>.............................] - ETA: 1:50 - loss: 0.8432 - categorical_accuracy: 0.7280
 3712/60000 [>.............................] - ETA: 1:50 - loss: 0.8392 - categorical_accuracy: 0.7293
 3744/60000 [>.............................] - ETA: 1:50 - loss: 0.8345 - categorical_accuracy: 0.7305
 3776/60000 [>.............................] - ETA: 1:50 - loss: 0.8316 - categorical_accuracy: 0.7315
 3808/60000 [>.............................] - ETA: 1:50 - loss: 0.8278 - categorical_accuracy: 0.7327
 3840/60000 [>.............................] - ETA: 1:49 - loss: 0.8244 - categorical_accuracy: 0.7339
 3872/60000 [>.............................] - ETA: 1:49 - loss: 0.8195 - categorical_accuracy: 0.7350
 3904/60000 [>.............................] - ETA: 1:49 - loss: 0.8153 - categorical_accuracy: 0.7367
 3936/60000 [>.............................] - ETA: 1:49 - loss: 0.8138 - categorical_accuracy: 0.7373
 3968/60000 [>.............................] - ETA: 1:49 - loss: 0.8091 - categorical_accuracy: 0.7392
 4000/60000 [=>............................] - ETA: 1:49 - loss: 0.8078 - categorical_accuracy: 0.7395
 4032/60000 [=>............................] - ETA: 1:49 - loss: 0.8051 - categorical_accuracy: 0.7403
 4064/60000 [=>............................] - ETA: 1:49 - loss: 0.8008 - categorical_accuracy: 0.7416
 4096/60000 [=>............................] - ETA: 1:49 - loss: 0.7987 - categorical_accuracy: 0.7422
 4128/60000 [=>............................] - ETA: 1:49 - loss: 0.7944 - categorical_accuracy: 0.7437
 4160/60000 [=>............................] - ETA: 1:48 - loss: 0.7909 - categorical_accuracy: 0.7447
 4192/60000 [=>............................] - ETA: 1:48 - loss: 0.7869 - categorical_accuracy: 0.7457
 4224/60000 [=>............................] - ETA: 1:48 - loss: 0.7831 - categorical_accuracy: 0.7472
 4256/60000 [=>............................] - ETA: 1:48 - loss: 0.7809 - categorical_accuracy: 0.7474
 4288/60000 [=>............................] - ETA: 1:48 - loss: 0.7769 - categorical_accuracy: 0.7486
 4320/60000 [=>............................] - ETA: 1:48 - loss: 0.7733 - categorical_accuracy: 0.7502
 4352/60000 [=>............................] - ETA: 1:48 - loss: 0.7687 - categorical_accuracy: 0.7518
 4384/60000 [=>............................] - ETA: 1:48 - loss: 0.7655 - categorical_accuracy: 0.7527
 4416/60000 [=>............................] - ETA: 1:48 - loss: 0.7617 - categorical_accuracy: 0.7538
 4448/60000 [=>............................] - ETA: 1:48 - loss: 0.7574 - categorical_accuracy: 0.7554
 4480/60000 [=>............................] - ETA: 1:48 - loss: 0.7549 - categorical_accuracy: 0.7563
 4512/60000 [=>............................] - ETA: 1:48 - loss: 0.7528 - categorical_accuracy: 0.7573
 4544/60000 [=>............................] - ETA: 1:47 - loss: 0.7497 - categorical_accuracy: 0.7584
 4576/60000 [=>............................] - ETA: 1:47 - loss: 0.7455 - categorical_accuracy: 0.7598
 4608/60000 [=>............................] - ETA: 1:47 - loss: 0.7431 - categorical_accuracy: 0.7604
 4640/60000 [=>............................] - ETA: 1:47 - loss: 0.7402 - categorical_accuracy: 0.7614
 4672/60000 [=>............................] - ETA: 1:47 - loss: 0.7361 - categorical_accuracy: 0.7631
 4704/60000 [=>............................] - ETA: 1:47 - loss: 0.7334 - categorical_accuracy: 0.7636
 4736/60000 [=>............................] - ETA: 1:47 - loss: 0.7312 - categorical_accuracy: 0.7644
 4768/60000 [=>............................] - ETA: 1:47 - loss: 0.7303 - categorical_accuracy: 0.7649
 4800/60000 [=>............................] - ETA: 1:47 - loss: 0.7273 - categorical_accuracy: 0.7660
 4832/60000 [=>............................] - ETA: 1:47 - loss: 0.7242 - categorical_accuracy: 0.7670
 4864/60000 [=>............................] - ETA: 1:47 - loss: 0.7238 - categorical_accuracy: 0.7673
 4896/60000 [=>............................] - ETA: 1:47 - loss: 0.7217 - categorical_accuracy: 0.7678
 4928/60000 [=>............................] - ETA: 1:47 - loss: 0.7193 - categorical_accuracy: 0.7687
 4960/60000 [=>............................] - ETA: 1:47 - loss: 0.7165 - categorical_accuracy: 0.7694
 4992/60000 [=>............................] - ETA: 1:46 - loss: 0.7133 - categorical_accuracy: 0.7704
 5024/60000 [=>............................] - ETA: 1:46 - loss: 0.7127 - categorical_accuracy: 0.7709
 5056/60000 [=>............................] - ETA: 1:46 - loss: 0.7093 - categorical_accuracy: 0.7722
 5088/60000 [=>............................] - ETA: 1:46 - loss: 0.7053 - categorical_accuracy: 0.7734
 5120/60000 [=>............................] - ETA: 1:46 - loss: 0.7015 - categorical_accuracy: 0.7746
 5152/60000 [=>............................] - ETA: 1:46 - loss: 0.6984 - categorical_accuracy: 0.7754
 5184/60000 [=>............................] - ETA: 1:46 - loss: 0.6959 - categorical_accuracy: 0.7760
 5216/60000 [=>............................] - ETA: 1:46 - loss: 0.6941 - categorical_accuracy: 0.7766
 5248/60000 [=>............................] - ETA: 1:46 - loss: 0.6940 - categorical_accuracy: 0.7767
 5280/60000 [=>............................] - ETA: 1:46 - loss: 0.6925 - categorical_accuracy: 0.7771
 5312/60000 [=>............................] - ETA: 1:46 - loss: 0.6916 - categorical_accuracy: 0.7773
 5344/60000 [=>............................] - ETA: 1:46 - loss: 0.6902 - categorical_accuracy: 0.7777
 5376/60000 [=>............................] - ETA: 1:45 - loss: 0.6876 - categorical_accuracy: 0.7785
 5408/60000 [=>............................] - ETA: 1:45 - loss: 0.6848 - categorical_accuracy: 0.7794
 5440/60000 [=>............................] - ETA: 1:45 - loss: 0.6816 - categorical_accuracy: 0.7805
 5472/60000 [=>............................] - ETA: 1:45 - loss: 0.6794 - categorical_accuracy: 0.7811
 5504/60000 [=>............................] - ETA: 1:45 - loss: 0.6767 - categorical_accuracy: 0.7820
 5536/60000 [=>............................] - ETA: 1:45 - loss: 0.6744 - categorical_accuracy: 0.7825
 5568/60000 [=>............................] - ETA: 1:45 - loss: 0.6733 - categorical_accuracy: 0.7829
 5600/60000 [=>............................] - ETA: 1:45 - loss: 0.6706 - categorical_accuracy: 0.7839
 5632/60000 [=>............................] - ETA: 1:45 - loss: 0.6681 - categorical_accuracy: 0.7848
 5664/60000 [=>............................] - ETA: 1:45 - loss: 0.6656 - categorical_accuracy: 0.7857
 5696/60000 [=>............................] - ETA: 1:45 - loss: 0.6641 - categorical_accuracy: 0.7865
 5728/60000 [=>............................] - ETA: 1:45 - loss: 0.6630 - categorical_accuracy: 0.7870
 5760/60000 [=>............................] - ETA: 1:44 - loss: 0.6603 - categorical_accuracy: 0.7878
 5792/60000 [=>............................] - ETA: 1:44 - loss: 0.6573 - categorical_accuracy: 0.7887
 5824/60000 [=>............................] - ETA: 1:44 - loss: 0.6553 - categorical_accuracy: 0.7893
 5856/60000 [=>............................] - ETA: 1:44 - loss: 0.6537 - categorical_accuracy: 0.7900
 5888/60000 [=>............................] - ETA: 1:44 - loss: 0.6512 - categorical_accuracy: 0.7908
 5920/60000 [=>............................] - ETA: 1:44 - loss: 0.6485 - categorical_accuracy: 0.7916
 5952/60000 [=>............................] - ETA: 1:44 - loss: 0.6477 - categorical_accuracy: 0.7920
 5984/60000 [=>............................] - ETA: 1:44 - loss: 0.6452 - categorical_accuracy: 0.7926
 6016/60000 [==>...........................] - ETA: 1:44 - loss: 0.6435 - categorical_accuracy: 0.7931
 6048/60000 [==>...........................] - ETA: 1:44 - loss: 0.6416 - categorical_accuracy: 0.7935
 6080/60000 [==>...........................] - ETA: 1:44 - loss: 0.6391 - categorical_accuracy: 0.7942
 6112/60000 [==>...........................] - ETA: 1:44 - loss: 0.6375 - categorical_accuracy: 0.7948
 6144/60000 [==>...........................] - ETA: 1:44 - loss: 0.6353 - categorical_accuracy: 0.7957
 6176/60000 [==>...........................] - ETA: 1:44 - loss: 0.6330 - categorical_accuracy: 0.7963
 6208/60000 [==>...........................] - ETA: 1:44 - loss: 0.6305 - categorical_accuracy: 0.7972
 6240/60000 [==>...........................] - ETA: 1:44 - loss: 0.6279 - categorical_accuracy: 0.7981
 6272/60000 [==>...........................] - ETA: 1:44 - loss: 0.6263 - categorical_accuracy: 0.7988
 6304/60000 [==>...........................] - ETA: 1:44 - loss: 0.6260 - categorical_accuracy: 0.7990
 6336/60000 [==>...........................] - ETA: 1:43 - loss: 0.6244 - categorical_accuracy: 0.7996
 6368/60000 [==>...........................] - ETA: 1:43 - loss: 0.6230 - categorical_accuracy: 0.7999
 6400/60000 [==>...........................] - ETA: 1:43 - loss: 0.6203 - categorical_accuracy: 0.8009
 6432/60000 [==>...........................] - ETA: 1:43 - loss: 0.6176 - categorical_accuracy: 0.8018
 6464/60000 [==>...........................] - ETA: 1:43 - loss: 0.6153 - categorical_accuracy: 0.8026
 6496/60000 [==>...........................] - ETA: 1:43 - loss: 0.6127 - categorical_accuracy: 0.8034
 6528/60000 [==>...........................] - ETA: 1:43 - loss: 0.6101 - categorical_accuracy: 0.8044
 6560/60000 [==>...........................] - ETA: 1:43 - loss: 0.6075 - categorical_accuracy: 0.8052
 6592/60000 [==>...........................] - ETA: 1:43 - loss: 0.6053 - categorical_accuracy: 0.8060
 6624/60000 [==>...........................] - ETA: 1:43 - loss: 0.6025 - categorical_accuracy: 0.8069
 6656/60000 [==>...........................] - ETA: 1:43 - loss: 0.5999 - categorical_accuracy: 0.8078
 6688/60000 [==>...........................] - ETA: 1:43 - loss: 0.5980 - categorical_accuracy: 0.8085
 6720/60000 [==>...........................] - ETA: 1:42 - loss: 0.5961 - categorical_accuracy: 0.8091
 6752/60000 [==>...........................] - ETA: 1:42 - loss: 0.5937 - categorical_accuracy: 0.8100
 6784/60000 [==>...........................] - ETA: 1:42 - loss: 0.5923 - categorical_accuracy: 0.8104
 6816/60000 [==>...........................] - ETA: 1:42 - loss: 0.5906 - categorical_accuracy: 0.8109
 6848/60000 [==>...........................] - ETA: 1:42 - loss: 0.5896 - categorical_accuracy: 0.8109
 6880/60000 [==>...........................] - ETA: 1:42 - loss: 0.5872 - categorical_accuracy: 0.8116
 6912/60000 [==>...........................] - ETA: 1:42 - loss: 0.5862 - categorical_accuracy: 0.8121
 6944/60000 [==>...........................] - ETA: 1:42 - loss: 0.5837 - categorical_accuracy: 0.8129
 6976/60000 [==>...........................] - ETA: 1:42 - loss: 0.5821 - categorical_accuracy: 0.8134
 7008/60000 [==>...........................] - ETA: 1:42 - loss: 0.5805 - categorical_accuracy: 0.8138
 7040/60000 [==>...........................] - ETA: 1:42 - loss: 0.5787 - categorical_accuracy: 0.8143
 7072/60000 [==>...........................] - ETA: 1:42 - loss: 0.5772 - categorical_accuracy: 0.8148
 7104/60000 [==>...........................] - ETA: 1:41 - loss: 0.5758 - categorical_accuracy: 0.8150
 7136/60000 [==>...........................] - ETA: 1:41 - loss: 0.5741 - categorical_accuracy: 0.8156
 7168/60000 [==>...........................] - ETA: 1:41 - loss: 0.5724 - categorical_accuracy: 0.8161
 7200/60000 [==>...........................] - ETA: 1:41 - loss: 0.5717 - categorical_accuracy: 0.8165
 7232/60000 [==>...........................] - ETA: 1:41 - loss: 0.5703 - categorical_accuracy: 0.8171
 7264/60000 [==>...........................] - ETA: 1:41 - loss: 0.5681 - categorical_accuracy: 0.8179
 7296/60000 [==>...........................] - ETA: 1:41 - loss: 0.5663 - categorical_accuracy: 0.8183
 7328/60000 [==>...........................] - ETA: 1:41 - loss: 0.5649 - categorical_accuracy: 0.8186
 7360/60000 [==>...........................] - ETA: 1:41 - loss: 0.5630 - categorical_accuracy: 0.8193
 7392/60000 [==>...........................] - ETA: 1:41 - loss: 0.5618 - categorical_accuracy: 0.8197
 7424/60000 [==>...........................] - ETA: 1:41 - loss: 0.5599 - categorical_accuracy: 0.8203
 7456/60000 [==>...........................] - ETA: 1:41 - loss: 0.5608 - categorical_accuracy: 0.8201
 7488/60000 [==>...........................] - ETA: 1:41 - loss: 0.5589 - categorical_accuracy: 0.8208
 7520/60000 [==>...........................] - ETA: 1:41 - loss: 0.5576 - categorical_accuracy: 0.8209
 7552/60000 [==>...........................] - ETA: 1:41 - loss: 0.5562 - categorical_accuracy: 0.8212
 7584/60000 [==>...........................] - ETA: 1:41 - loss: 0.5545 - categorical_accuracy: 0.8219
 7616/60000 [==>...........................] - ETA: 1:41 - loss: 0.5528 - categorical_accuracy: 0.8223
 7648/60000 [==>...........................] - ETA: 1:41 - loss: 0.5518 - categorical_accuracy: 0.8226
 7680/60000 [==>...........................] - ETA: 1:40 - loss: 0.5514 - categorical_accuracy: 0.8229
 7712/60000 [==>...........................] - ETA: 1:40 - loss: 0.5506 - categorical_accuracy: 0.8231
 7744/60000 [==>...........................] - ETA: 1:40 - loss: 0.5504 - categorical_accuracy: 0.8232
 7776/60000 [==>...........................] - ETA: 1:40 - loss: 0.5488 - categorical_accuracy: 0.8237
 7808/60000 [==>...........................] - ETA: 1:40 - loss: 0.5477 - categorical_accuracy: 0.8240
 7840/60000 [==>...........................] - ETA: 1:40 - loss: 0.5468 - categorical_accuracy: 0.8242
 7872/60000 [==>...........................] - ETA: 1:40 - loss: 0.5450 - categorical_accuracy: 0.8248
 7904/60000 [==>...........................] - ETA: 1:40 - loss: 0.5436 - categorical_accuracy: 0.8253
 7936/60000 [==>...........................] - ETA: 1:40 - loss: 0.5426 - categorical_accuracy: 0.8256
 7968/60000 [==>...........................] - ETA: 1:40 - loss: 0.5409 - categorical_accuracy: 0.8262
 8000/60000 [===>..........................] - ETA: 1:40 - loss: 0.5403 - categorical_accuracy: 0.8266
 8032/60000 [===>..........................] - ETA: 1:40 - loss: 0.5386 - categorical_accuracy: 0.8271
 8064/60000 [===>..........................] - ETA: 1:40 - loss: 0.5369 - categorical_accuracy: 0.8276
 8096/60000 [===>..........................] - ETA: 1:40 - loss: 0.5355 - categorical_accuracy: 0.8282
 8128/60000 [===>..........................] - ETA: 1:40 - loss: 0.5342 - categorical_accuracy: 0.8286
 8160/60000 [===>..........................] - ETA: 1:39 - loss: 0.5337 - categorical_accuracy: 0.8292
 8192/60000 [===>..........................] - ETA: 1:39 - loss: 0.5321 - categorical_accuracy: 0.8297
 8224/60000 [===>..........................] - ETA: 1:39 - loss: 0.5309 - categorical_accuracy: 0.8299
 8256/60000 [===>..........................] - ETA: 1:39 - loss: 0.5295 - categorical_accuracy: 0.8303
 8288/60000 [===>..........................] - ETA: 1:39 - loss: 0.5288 - categorical_accuracy: 0.8307
 8320/60000 [===>..........................] - ETA: 1:39 - loss: 0.5290 - categorical_accuracy: 0.8309
 8352/60000 [===>..........................] - ETA: 1:39 - loss: 0.5274 - categorical_accuracy: 0.8315
 8384/60000 [===>..........................] - ETA: 1:39 - loss: 0.5264 - categorical_accuracy: 0.8319
 8416/60000 [===>..........................] - ETA: 1:39 - loss: 0.5251 - categorical_accuracy: 0.8322
 8448/60000 [===>..........................] - ETA: 1:39 - loss: 0.5241 - categorical_accuracy: 0.8326
 8480/60000 [===>..........................] - ETA: 1:39 - loss: 0.5233 - categorical_accuracy: 0.8329
 8512/60000 [===>..........................] - ETA: 1:39 - loss: 0.5222 - categorical_accuracy: 0.8334
 8544/60000 [===>..........................] - ETA: 1:39 - loss: 0.5220 - categorical_accuracy: 0.8336
 8576/60000 [===>..........................] - ETA: 1:39 - loss: 0.5206 - categorical_accuracy: 0.8340
 8608/60000 [===>..........................] - ETA: 1:39 - loss: 0.5189 - categorical_accuracy: 0.8346
 8640/60000 [===>..........................] - ETA: 1:38 - loss: 0.5172 - categorical_accuracy: 0.8352
 8672/60000 [===>..........................] - ETA: 1:38 - loss: 0.5162 - categorical_accuracy: 0.8354
 8704/60000 [===>..........................] - ETA: 1:38 - loss: 0.5148 - categorical_accuracy: 0.8358
 8736/60000 [===>..........................] - ETA: 1:38 - loss: 0.5134 - categorical_accuracy: 0.8362
 8768/60000 [===>..........................] - ETA: 1:38 - loss: 0.5126 - categorical_accuracy: 0.8366
 8800/60000 [===>..........................] - ETA: 1:38 - loss: 0.5108 - categorical_accuracy: 0.8372
 8832/60000 [===>..........................] - ETA: 1:38 - loss: 0.5106 - categorical_accuracy: 0.8373
 8864/60000 [===>..........................] - ETA: 1:38 - loss: 0.5090 - categorical_accuracy: 0.8378
 8896/60000 [===>..........................] - ETA: 1:38 - loss: 0.5082 - categorical_accuracy: 0.8380
 8928/60000 [===>..........................] - ETA: 1:38 - loss: 0.5068 - categorical_accuracy: 0.8385
 8960/60000 [===>..........................] - ETA: 1:38 - loss: 0.5057 - categorical_accuracy: 0.8388
 8992/60000 [===>..........................] - ETA: 1:38 - loss: 0.5053 - categorical_accuracy: 0.8390
 9024/60000 [===>..........................] - ETA: 1:38 - loss: 0.5038 - categorical_accuracy: 0.8394
 9056/60000 [===>..........................] - ETA: 1:37 - loss: 0.5024 - categorical_accuracy: 0.8399
 9088/60000 [===>..........................] - ETA: 1:37 - loss: 0.5011 - categorical_accuracy: 0.8402
 9120/60000 [===>..........................] - ETA: 1:37 - loss: 0.5005 - categorical_accuracy: 0.8407
 9152/60000 [===>..........................] - ETA: 1:37 - loss: 0.4992 - categorical_accuracy: 0.8411
 9184/60000 [===>..........................] - ETA: 1:37 - loss: 0.4979 - categorical_accuracy: 0.8416
 9216/60000 [===>..........................] - ETA: 1:37 - loss: 0.4964 - categorical_accuracy: 0.8421
 9248/60000 [===>..........................] - ETA: 1:37 - loss: 0.4964 - categorical_accuracy: 0.8423
 9280/60000 [===>..........................] - ETA: 1:37 - loss: 0.4950 - categorical_accuracy: 0.8429
 9312/60000 [===>..........................] - ETA: 1:37 - loss: 0.4939 - categorical_accuracy: 0.8432
 9344/60000 [===>..........................] - ETA: 1:37 - loss: 0.4935 - categorical_accuracy: 0.8434
 9376/60000 [===>..........................] - ETA: 1:37 - loss: 0.4923 - categorical_accuracy: 0.8438
 9408/60000 [===>..........................] - ETA: 1:37 - loss: 0.4913 - categorical_accuracy: 0.8441
 9440/60000 [===>..........................] - ETA: 1:37 - loss: 0.4898 - categorical_accuracy: 0.8446
 9472/60000 [===>..........................] - ETA: 1:37 - loss: 0.4889 - categorical_accuracy: 0.8449
 9504/60000 [===>..........................] - ETA: 1:37 - loss: 0.4879 - categorical_accuracy: 0.8452
 9536/60000 [===>..........................] - ETA: 1:37 - loss: 0.4865 - categorical_accuracy: 0.8456
 9568/60000 [===>..........................] - ETA: 1:36 - loss: 0.4861 - categorical_accuracy: 0.8458
 9600/60000 [===>..........................] - ETA: 1:36 - loss: 0.4848 - categorical_accuracy: 0.8462
 9632/60000 [===>..........................] - ETA: 1:36 - loss: 0.4840 - categorical_accuracy: 0.8464
 9664/60000 [===>..........................] - ETA: 1:36 - loss: 0.4833 - categorical_accuracy: 0.8466
 9696/60000 [===>..........................] - ETA: 1:36 - loss: 0.4820 - categorical_accuracy: 0.8471
 9728/60000 [===>..........................] - ETA: 1:36 - loss: 0.4813 - categorical_accuracy: 0.8472
 9760/60000 [===>..........................] - ETA: 1:36 - loss: 0.4803 - categorical_accuracy: 0.8476
 9792/60000 [===>..........................] - ETA: 1:36 - loss: 0.4790 - categorical_accuracy: 0.8479
 9824/60000 [===>..........................] - ETA: 1:36 - loss: 0.4777 - categorical_accuracy: 0.8484
 9856/60000 [===>..........................] - ETA: 1:36 - loss: 0.4769 - categorical_accuracy: 0.8487
 9888/60000 [===>..........................] - ETA: 1:36 - loss: 0.4760 - categorical_accuracy: 0.8490
 9920/60000 [===>..........................] - ETA: 1:36 - loss: 0.4755 - categorical_accuracy: 0.8492
 9952/60000 [===>..........................] - ETA: 1:36 - loss: 0.4743 - categorical_accuracy: 0.8496
 9984/60000 [===>..........................] - ETA: 1:36 - loss: 0.4733 - categorical_accuracy: 0.8500
10016/60000 [====>.........................] - ETA: 1:36 - loss: 0.4728 - categorical_accuracy: 0.8502
10048/60000 [====>.........................] - ETA: 1:36 - loss: 0.4721 - categorical_accuracy: 0.8504
10080/60000 [====>.........................] - ETA: 1:35 - loss: 0.4707 - categorical_accuracy: 0.8509
10112/60000 [====>.........................] - ETA: 1:35 - loss: 0.4697 - categorical_accuracy: 0.8513
10144/60000 [====>.........................] - ETA: 1:35 - loss: 0.4691 - categorical_accuracy: 0.8514
10176/60000 [====>.........................] - ETA: 1:35 - loss: 0.4677 - categorical_accuracy: 0.8519
10208/60000 [====>.........................] - ETA: 1:35 - loss: 0.4670 - categorical_accuracy: 0.8522
10240/60000 [====>.........................] - ETA: 1:35 - loss: 0.4658 - categorical_accuracy: 0.8525
10272/60000 [====>.........................] - ETA: 1:35 - loss: 0.4645 - categorical_accuracy: 0.8530
10304/60000 [====>.........................] - ETA: 1:35 - loss: 0.4632 - categorical_accuracy: 0.8535
10336/60000 [====>.........................] - ETA: 1:35 - loss: 0.4619 - categorical_accuracy: 0.8539
10368/60000 [====>.........................] - ETA: 1:35 - loss: 0.4610 - categorical_accuracy: 0.8542
10400/60000 [====>.........................] - ETA: 1:35 - loss: 0.4602 - categorical_accuracy: 0.8545
10432/60000 [====>.........................] - ETA: 1:35 - loss: 0.4596 - categorical_accuracy: 0.8547
10464/60000 [====>.........................] - ETA: 1:35 - loss: 0.4583 - categorical_accuracy: 0.8551
10496/60000 [====>.........................] - ETA: 1:35 - loss: 0.4574 - categorical_accuracy: 0.8552
10528/60000 [====>.........................] - ETA: 1:34 - loss: 0.4563 - categorical_accuracy: 0.8555
10560/60000 [====>.........................] - ETA: 1:34 - loss: 0.4554 - categorical_accuracy: 0.8558
10592/60000 [====>.........................] - ETA: 1:34 - loss: 0.4547 - categorical_accuracy: 0.8558
10624/60000 [====>.........................] - ETA: 1:34 - loss: 0.4541 - categorical_accuracy: 0.8560
10656/60000 [====>.........................] - ETA: 1:34 - loss: 0.4539 - categorical_accuracy: 0.8560
10688/60000 [====>.........................] - ETA: 1:34 - loss: 0.4531 - categorical_accuracy: 0.8563
10720/60000 [====>.........................] - ETA: 1:34 - loss: 0.4523 - categorical_accuracy: 0.8564
10752/60000 [====>.........................] - ETA: 1:34 - loss: 0.4515 - categorical_accuracy: 0.8566
10784/60000 [====>.........................] - ETA: 1:34 - loss: 0.4505 - categorical_accuracy: 0.8569
10816/60000 [====>.........................] - ETA: 1:34 - loss: 0.4497 - categorical_accuracy: 0.8572
10848/60000 [====>.........................] - ETA: 1:34 - loss: 0.4489 - categorical_accuracy: 0.8575
10880/60000 [====>.........................] - ETA: 1:34 - loss: 0.4485 - categorical_accuracy: 0.8577
10912/60000 [====>.........................] - ETA: 1:34 - loss: 0.4474 - categorical_accuracy: 0.8581
10944/60000 [====>.........................] - ETA: 1:34 - loss: 0.4464 - categorical_accuracy: 0.8585
10976/60000 [====>.........................] - ETA: 1:34 - loss: 0.4463 - categorical_accuracy: 0.8584
11008/60000 [====>.........................] - ETA: 1:33 - loss: 0.4458 - categorical_accuracy: 0.8586
11040/60000 [====>.........................] - ETA: 1:33 - loss: 0.4448 - categorical_accuracy: 0.8590
11072/60000 [====>.........................] - ETA: 1:33 - loss: 0.4439 - categorical_accuracy: 0.8592
11104/60000 [====>.........................] - ETA: 1:33 - loss: 0.4428 - categorical_accuracy: 0.8596
11136/60000 [====>.........................] - ETA: 1:33 - loss: 0.4418 - categorical_accuracy: 0.8600
11168/60000 [====>.........................] - ETA: 1:33 - loss: 0.4413 - categorical_accuracy: 0.8601
11200/60000 [====>.........................] - ETA: 1:33 - loss: 0.4407 - categorical_accuracy: 0.8602
11232/60000 [====>.........................] - ETA: 1:33 - loss: 0.4400 - categorical_accuracy: 0.8605
11264/60000 [====>.........................] - ETA: 1:33 - loss: 0.4394 - categorical_accuracy: 0.8607
11296/60000 [====>.........................] - ETA: 1:33 - loss: 0.4389 - categorical_accuracy: 0.8608
11328/60000 [====>.........................] - ETA: 1:33 - loss: 0.4383 - categorical_accuracy: 0.8611
11360/60000 [====>.........................] - ETA: 1:33 - loss: 0.4375 - categorical_accuracy: 0.8612
11392/60000 [====>.........................] - ETA: 1:33 - loss: 0.4368 - categorical_accuracy: 0.8615
11424/60000 [====>.........................] - ETA: 1:33 - loss: 0.4359 - categorical_accuracy: 0.8617
11456/60000 [====>.........................] - ETA: 1:33 - loss: 0.4351 - categorical_accuracy: 0.8620
11488/60000 [====>.........................] - ETA: 1:33 - loss: 0.4345 - categorical_accuracy: 0.8622
11520/60000 [====>.........................] - ETA: 1:32 - loss: 0.4334 - categorical_accuracy: 0.8626
11552/60000 [====>.........................] - ETA: 1:32 - loss: 0.4324 - categorical_accuracy: 0.8629
11584/60000 [====>.........................] - ETA: 1:32 - loss: 0.4320 - categorical_accuracy: 0.8631
11616/60000 [====>.........................] - ETA: 1:32 - loss: 0.4315 - categorical_accuracy: 0.8633
11648/60000 [====>.........................] - ETA: 1:32 - loss: 0.4308 - categorical_accuracy: 0.8635
11680/60000 [====>.........................] - ETA: 1:32 - loss: 0.4307 - categorical_accuracy: 0.8635
11712/60000 [====>.........................] - ETA: 1:32 - loss: 0.4303 - categorical_accuracy: 0.8636
11744/60000 [====>.........................] - ETA: 1:32 - loss: 0.4294 - categorical_accuracy: 0.8638
11776/60000 [====>.........................] - ETA: 1:32 - loss: 0.4284 - categorical_accuracy: 0.8641
11808/60000 [====>.........................] - ETA: 1:32 - loss: 0.4277 - categorical_accuracy: 0.8644
11840/60000 [====>.........................] - ETA: 1:32 - loss: 0.4273 - categorical_accuracy: 0.8646
11872/60000 [====>.........................] - ETA: 1:32 - loss: 0.4274 - categorical_accuracy: 0.8647
11904/60000 [====>.........................] - ETA: 1:32 - loss: 0.4264 - categorical_accuracy: 0.8650
11936/60000 [====>.........................] - ETA: 1:32 - loss: 0.4260 - categorical_accuracy: 0.8651
11968/60000 [====>.........................] - ETA: 1:32 - loss: 0.4254 - categorical_accuracy: 0.8654
12000/60000 [=====>........................] - ETA: 1:31 - loss: 0.4254 - categorical_accuracy: 0.8655
12032/60000 [=====>........................] - ETA: 1:31 - loss: 0.4244 - categorical_accuracy: 0.8659
12064/60000 [=====>........................] - ETA: 1:31 - loss: 0.4240 - categorical_accuracy: 0.8660
12096/60000 [=====>........................] - ETA: 1:31 - loss: 0.4231 - categorical_accuracy: 0.8662
12128/60000 [=====>........................] - ETA: 1:31 - loss: 0.4220 - categorical_accuracy: 0.8666
12160/60000 [=====>........................] - ETA: 1:31 - loss: 0.4213 - categorical_accuracy: 0.8668
12192/60000 [=====>........................] - ETA: 1:31 - loss: 0.4213 - categorical_accuracy: 0.8667
12224/60000 [=====>........................] - ETA: 1:31 - loss: 0.4203 - categorical_accuracy: 0.8671
12256/60000 [=====>........................] - ETA: 1:31 - loss: 0.4202 - categorical_accuracy: 0.8671
12288/60000 [=====>........................] - ETA: 1:31 - loss: 0.4199 - categorical_accuracy: 0.8672
12320/60000 [=====>........................] - ETA: 1:31 - loss: 0.4191 - categorical_accuracy: 0.8674
12352/60000 [=====>........................] - ETA: 1:31 - loss: 0.4183 - categorical_accuracy: 0.8676
12384/60000 [=====>........................] - ETA: 1:31 - loss: 0.4177 - categorical_accuracy: 0.8679
12416/60000 [=====>........................] - ETA: 1:31 - loss: 0.4168 - categorical_accuracy: 0.8682
12448/60000 [=====>........................] - ETA: 1:31 - loss: 0.4160 - categorical_accuracy: 0.8685
12480/60000 [=====>........................] - ETA: 1:31 - loss: 0.4155 - categorical_accuracy: 0.8687
12512/60000 [=====>........................] - ETA: 1:30 - loss: 0.4151 - categorical_accuracy: 0.8688
12544/60000 [=====>........................] - ETA: 1:30 - loss: 0.4143 - categorical_accuracy: 0.8691
12576/60000 [=====>........................] - ETA: 1:30 - loss: 0.4141 - categorical_accuracy: 0.8692
12608/60000 [=====>........................] - ETA: 1:30 - loss: 0.4132 - categorical_accuracy: 0.8695
12640/60000 [=====>........................] - ETA: 1:30 - loss: 0.4124 - categorical_accuracy: 0.8698
12672/60000 [=====>........................] - ETA: 1:30 - loss: 0.4121 - categorical_accuracy: 0.8700
12704/60000 [=====>........................] - ETA: 1:30 - loss: 0.4113 - categorical_accuracy: 0.8704
12736/60000 [=====>........................] - ETA: 1:30 - loss: 0.4116 - categorical_accuracy: 0.8704
12768/60000 [=====>........................] - ETA: 1:30 - loss: 0.4109 - categorical_accuracy: 0.8707
12800/60000 [=====>........................] - ETA: 1:30 - loss: 0.4101 - categorical_accuracy: 0.8709
12832/60000 [=====>........................] - ETA: 1:30 - loss: 0.4093 - categorical_accuracy: 0.8713
12864/60000 [=====>........................] - ETA: 1:30 - loss: 0.4084 - categorical_accuracy: 0.8716
12896/60000 [=====>........................] - ETA: 1:30 - loss: 0.4075 - categorical_accuracy: 0.8719
12928/60000 [=====>........................] - ETA: 1:30 - loss: 0.4068 - categorical_accuracy: 0.8721
12960/60000 [=====>........................] - ETA: 1:30 - loss: 0.4061 - categorical_accuracy: 0.8723
12992/60000 [=====>........................] - ETA: 1:30 - loss: 0.4054 - categorical_accuracy: 0.8725
13024/60000 [=====>........................] - ETA: 1:29 - loss: 0.4045 - categorical_accuracy: 0.8727
13056/60000 [=====>........................] - ETA: 1:29 - loss: 0.4043 - categorical_accuracy: 0.8729
13088/60000 [=====>........................] - ETA: 1:29 - loss: 0.4035 - categorical_accuracy: 0.8732
13120/60000 [=====>........................] - ETA: 1:29 - loss: 0.4028 - categorical_accuracy: 0.8733
13152/60000 [=====>........................] - ETA: 1:29 - loss: 0.4023 - categorical_accuracy: 0.8736
13184/60000 [=====>........................] - ETA: 1:29 - loss: 0.4020 - categorical_accuracy: 0.8736
13216/60000 [=====>........................] - ETA: 1:29 - loss: 0.4016 - categorical_accuracy: 0.8739
13248/60000 [=====>........................] - ETA: 1:29 - loss: 0.4011 - categorical_accuracy: 0.8740
13280/60000 [=====>........................] - ETA: 1:29 - loss: 0.4004 - categorical_accuracy: 0.8742
13312/60000 [=====>........................] - ETA: 1:29 - loss: 0.4002 - categorical_accuracy: 0.8742
13344/60000 [=====>........................] - ETA: 1:29 - loss: 0.3996 - categorical_accuracy: 0.8744
13376/60000 [=====>........................] - ETA: 1:29 - loss: 0.3992 - categorical_accuracy: 0.8744
13408/60000 [=====>........................] - ETA: 1:29 - loss: 0.3990 - categorical_accuracy: 0.8746
13440/60000 [=====>........................] - ETA: 1:29 - loss: 0.3985 - categorical_accuracy: 0.8747
13472/60000 [=====>........................] - ETA: 1:29 - loss: 0.3977 - categorical_accuracy: 0.8750
13504/60000 [=====>........................] - ETA: 1:28 - loss: 0.3970 - categorical_accuracy: 0.8752
13536/60000 [=====>........................] - ETA: 1:28 - loss: 0.3967 - categorical_accuracy: 0.8754
13568/60000 [=====>........................] - ETA: 1:28 - loss: 0.3961 - categorical_accuracy: 0.8755
13600/60000 [=====>........................] - ETA: 1:28 - loss: 0.3952 - categorical_accuracy: 0.8758
13632/60000 [=====>........................] - ETA: 1:28 - loss: 0.3944 - categorical_accuracy: 0.8761
13664/60000 [=====>........................] - ETA: 1:28 - loss: 0.3938 - categorical_accuracy: 0.8763
13696/60000 [=====>........................] - ETA: 1:28 - loss: 0.3934 - categorical_accuracy: 0.8765
13728/60000 [=====>........................] - ETA: 1:28 - loss: 0.3933 - categorical_accuracy: 0.8766
13760/60000 [=====>........................] - ETA: 1:28 - loss: 0.3931 - categorical_accuracy: 0.8767
13792/60000 [=====>........................] - ETA: 1:28 - loss: 0.3925 - categorical_accuracy: 0.8770
13824/60000 [=====>........................] - ETA: 1:28 - loss: 0.3918 - categorical_accuracy: 0.8772
13856/60000 [=====>........................] - ETA: 1:28 - loss: 0.3915 - categorical_accuracy: 0.8772
13888/60000 [=====>........................] - ETA: 1:28 - loss: 0.3909 - categorical_accuracy: 0.8774
13920/60000 [=====>........................] - ETA: 1:28 - loss: 0.3902 - categorical_accuracy: 0.8777
13952/60000 [=====>........................] - ETA: 1:28 - loss: 0.3897 - categorical_accuracy: 0.8777
13984/60000 [=====>........................] - ETA: 1:28 - loss: 0.3894 - categorical_accuracy: 0.8778
14016/60000 [======>.......................] - ETA: 1:27 - loss: 0.3892 - categorical_accuracy: 0.8778
14048/60000 [======>.......................] - ETA: 1:27 - loss: 0.3888 - categorical_accuracy: 0.8779
14080/60000 [======>.......................] - ETA: 1:27 - loss: 0.3883 - categorical_accuracy: 0.8781
14112/60000 [======>.......................] - ETA: 1:27 - loss: 0.3876 - categorical_accuracy: 0.8783
14144/60000 [======>.......................] - ETA: 1:27 - loss: 0.3873 - categorical_accuracy: 0.8784
14176/60000 [======>.......................] - ETA: 1:27 - loss: 0.3871 - categorical_accuracy: 0.8785
14208/60000 [======>.......................] - ETA: 1:27 - loss: 0.3863 - categorical_accuracy: 0.8787
14240/60000 [======>.......................] - ETA: 1:27 - loss: 0.3858 - categorical_accuracy: 0.8789
14272/60000 [======>.......................] - ETA: 1:27 - loss: 0.3852 - categorical_accuracy: 0.8790
14304/60000 [======>.......................] - ETA: 1:27 - loss: 0.3844 - categorical_accuracy: 0.8793
14336/60000 [======>.......................] - ETA: 1:27 - loss: 0.3837 - categorical_accuracy: 0.8795
14368/60000 [======>.......................] - ETA: 1:27 - loss: 0.3831 - categorical_accuracy: 0.8797
14400/60000 [======>.......................] - ETA: 1:27 - loss: 0.3824 - categorical_accuracy: 0.8799
14432/60000 [======>.......................] - ETA: 1:27 - loss: 0.3816 - categorical_accuracy: 0.8801
14464/60000 [======>.......................] - ETA: 1:27 - loss: 0.3812 - categorical_accuracy: 0.8803
14496/60000 [======>.......................] - ETA: 1:26 - loss: 0.3808 - categorical_accuracy: 0.8804
14528/60000 [======>.......................] - ETA: 1:26 - loss: 0.3809 - categorical_accuracy: 0.8806
14560/60000 [======>.......................] - ETA: 1:26 - loss: 0.3804 - categorical_accuracy: 0.8808
14592/60000 [======>.......................] - ETA: 1:26 - loss: 0.3799 - categorical_accuracy: 0.8809
14624/60000 [======>.......................] - ETA: 1:26 - loss: 0.3795 - categorical_accuracy: 0.8810
14656/60000 [======>.......................] - ETA: 1:26 - loss: 0.3788 - categorical_accuracy: 0.8812
14688/60000 [======>.......................] - ETA: 1:26 - loss: 0.3783 - categorical_accuracy: 0.8814
14720/60000 [======>.......................] - ETA: 1:26 - loss: 0.3775 - categorical_accuracy: 0.8817
14752/60000 [======>.......................] - ETA: 1:26 - loss: 0.3769 - categorical_accuracy: 0.8818
14784/60000 [======>.......................] - ETA: 1:26 - loss: 0.3770 - categorical_accuracy: 0.8819
14816/60000 [======>.......................] - ETA: 1:26 - loss: 0.3777 - categorical_accuracy: 0.8820
14848/60000 [======>.......................] - ETA: 1:26 - loss: 0.3774 - categorical_accuracy: 0.8820
14880/60000 [======>.......................] - ETA: 1:26 - loss: 0.3769 - categorical_accuracy: 0.8821
14912/60000 [======>.......................] - ETA: 1:26 - loss: 0.3767 - categorical_accuracy: 0.8822
14944/60000 [======>.......................] - ETA: 1:26 - loss: 0.3763 - categorical_accuracy: 0.8823
14976/60000 [======>.......................] - ETA: 1:26 - loss: 0.3757 - categorical_accuracy: 0.8824
15008/60000 [======>.......................] - ETA: 1:25 - loss: 0.3751 - categorical_accuracy: 0.8826
15040/60000 [======>.......................] - ETA: 1:25 - loss: 0.3745 - categorical_accuracy: 0.8828
15072/60000 [======>.......................] - ETA: 1:25 - loss: 0.3742 - categorical_accuracy: 0.8829
15104/60000 [======>.......................] - ETA: 1:25 - loss: 0.3735 - categorical_accuracy: 0.8831
15136/60000 [======>.......................] - ETA: 1:25 - loss: 0.3735 - categorical_accuracy: 0.8832
15168/60000 [======>.......................] - ETA: 1:25 - loss: 0.3729 - categorical_accuracy: 0.8834
15200/60000 [======>.......................] - ETA: 1:25 - loss: 0.3722 - categorical_accuracy: 0.8837
15232/60000 [======>.......................] - ETA: 1:25 - loss: 0.3722 - categorical_accuracy: 0.8838
15264/60000 [======>.......................] - ETA: 1:25 - loss: 0.3717 - categorical_accuracy: 0.8839
15296/60000 [======>.......................] - ETA: 1:25 - loss: 0.3710 - categorical_accuracy: 0.8842
15328/60000 [======>.......................] - ETA: 1:25 - loss: 0.3705 - categorical_accuracy: 0.8843
15360/60000 [======>.......................] - ETA: 1:25 - loss: 0.3699 - categorical_accuracy: 0.8845
15392/60000 [======>.......................] - ETA: 1:25 - loss: 0.3695 - categorical_accuracy: 0.8846
15424/60000 [======>.......................] - ETA: 1:25 - loss: 0.3690 - categorical_accuracy: 0.8847
15456/60000 [======>.......................] - ETA: 1:25 - loss: 0.3687 - categorical_accuracy: 0.8848
15488/60000 [======>.......................] - ETA: 1:25 - loss: 0.3688 - categorical_accuracy: 0.8849
15520/60000 [======>.......................] - ETA: 1:25 - loss: 0.3686 - categorical_accuracy: 0.8850
15552/60000 [======>.......................] - ETA: 1:24 - loss: 0.3687 - categorical_accuracy: 0.8850
15584/60000 [======>.......................] - ETA: 1:24 - loss: 0.3681 - categorical_accuracy: 0.8851
15616/60000 [======>.......................] - ETA: 1:24 - loss: 0.3676 - categorical_accuracy: 0.8853
15648/60000 [======>.......................] - ETA: 1:24 - loss: 0.3670 - categorical_accuracy: 0.8855
15680/60000 [======>.......................] - ETA: 1:24 - loss: 0.3663 - categorical_accuracy: 0.8857
15712/60000 [======>.......................] - ETA: 1:24 - loss: 0.3664 - categorical_accuracy: 0.8858
15744/60000 [======>.......................] - ETA: 1:24 - loss: 0.3663 - categorical_accuracy: 0.8857
15776/60000 [======>.......................] - ETA: 1:24 - loss: 0.3664 - categorical_accuracy: 0.8856
15808/60000 [======>.......................] - ETA: 1:24 - loss: 0.3663 - categorical_accuracy: 0.8856
15840/60000 [======>.......................] - ETA: 1:24 - loss: 0.3658 - categorical_accuracy: 0.8858
15872/60000 [======>.......................] - ETA: 1:24 - loss: 0.3655 - categorical_accuracy: 0.8858
15904/60000 [======>.......................] - ETA: 1:24 - loss: 0.3652 - categorical_accuracy: 0.8859
15936/60000 [======>.......................] - ETA: 1:24 - loss: 0.3649 - categorical_accuracy: 0.8860
15968/60000 [======>.......................] - ETA: 1:24 - loss: 0.3645 - categorical_accuracy: 0.8860
16000/60000 [=======>......................] - ETA: 1:24 - loss: 0.3644 - categorical_accuracy: 0.8861
16032/60000 [=======>......................] - ETA: 1:24 - loss: 0.3640 - categorical_accuracy: 0.8862
16064/60000 [=======>......................] - ETA: 1:24 - loss: 0.3634 - categorical_accuracy: 0.8865
16096/60000 [=======>......................] - ETA: 1:23 - loss: 0.3630 - categorical_accuracy: 0.8866
16128/60000 [=======>......................] - ETA: 1:23 - loss: 0.3627 - categorical_accuracy: 0.8867
16160/60000 [=======>......................] - ETA: 1:23 - loss: 0.3620 - categorical_accuracy: 0.8869
16192/60000 [=======>......................] - ETA: 1:23 - loss: 0.3619 - categorical_accuracy: 0.8870
16224/60000 [=======>......................] - ETA: 1:23 - loss: 0.3615 - categorical_accuracy: 0.8871
16256/60000 [=======>......................] - ETA: 1:23 - loss: 0.3609 - categorical_accuracy: 0.8873
16288/60000 [=======>......................] - ETA: 1:23 - loss: 0.3605 - categorical_accuracy: 0.8873
16320/60000 [=======>......................] - ETA: 1:23 - loss: 0.3599 - categorical_accuracy: 0.8876
16352/60000 [=======>......................] - ETA: 1:23 - loss: 0.3593 - categorical_accuracy: 0.8878
16384/60000 [=======>......................] - ETA: 1:23 - loss: 0.3593 - categorical_accuracy: 0.8878
16416/60000 [=======>......................] - ETA: 1:23 - loss: 0.3592 - categorical_accuracy: 0.8879
16448/60000 [=======>......................] - ETA: 1:23 - loss: 0.3588 - categorical_accuracy: 0.8880
16480/60000 [=======>......................] - ETA: 1:23 - loss: 0.3582 - categorical_accuracy: 0.8882
16512/60000 [=======>......................] - ETA: 1:23 - loss: 0.3579 - categorical_accuracy: 0.8883
16544/60000 [=======>......................] - ETA: 1:23 - loss: 0.3576 - categorical_accuracy: 0.8885
16576/60000 [=======>......................] - ETA: 1:23 - loss: 0.3574 - categorical_accuracy: 0.8885
16608/60000 [=======>......................] - ETA: 1:22 - loss: 0.3569 - categorical_accuracy: 0.8886
16640/60000 [=======>......................] - ETA: 1:22 - loss: 0.3567 - categorical_accuracy: 0.8886
16672/60000 [=======>......................] - ETA: 1:22 - loss: 0.3565 - categorical_accuracy: 0.8887
16704/60000 [=======>......................] - ETA: 1:22 - loss: 0.3560 - categorical_accuracy: 0.8888
16736/60000 [=======>......................] - ETA: 1:22 - loss: 0.3556 - categorical_accuracy: 0.8889
16768/60000 [=======>......................] - ETA: 1:22 - loss: 0.3553 - categorical_accuracy: 0.8890
16800/60000 [=======>......................] - ETA: 1:22 - loss: 0.3548 - categorical_accuracy: 0.8892
16832/60000 [=======>......................] - ETA: 1:22 - loss: 0.3546 - categorical_accuracy: 0.8893
16864/60000 [=======>......................] - ETA: 1:22 - loss: 0.3541 - categorical_accuracy: 0.8895
16896/60000 [=======>......................] - ETA: 1:22 - loss: 0.3538 - categorical_accuracy: 0.8895
16928/60000 [=======>......................] - ETA: 1:22 - loss: 0.3537 - categorical_accuracy: 0.8896
16960/60000 [=======>......................] - ETA: 1:22 - loss: 0.3535 - categorical_accuracy: 0.8897
16992/60000 [=======>......................] - ETA: 1:22 - loss: 0.3533 - categorical_accuracy: 0.8898
17024/60000 [=======>......................] - ETA: 1:22 - loss: 0.3527 - categorical_accuracy: 0.8900
17056/60000 [=======>......................] - ETA: 1:22 - loss: 0.3524 - categorical_accuracy: 0.8902
17088/60000 [=======>......................] - ETA: 1:22 - loss: 0.3521 - categorical_accuracy: 0.8903
17120/60000 [=======>......................] - ETA: 1:22 - loss: 0.3517 - categorical_accuracy: 0.8904
17152/60000 [=======>......................] - ETA: 1:21 - loss: 0.3513 - categorical_accuracy: 0.8906
17184/60000 [=======>......................] - ETA: 1:21 - loss: 0.3509 - categorical_accuracy: 0.8907
17216/60000 [=======>......................] - ETA: 1:21 - loss: 0.3503 - categorical_accuracy: 0.8909
17248/60000 [=======>......................] - ETA: 1:21 - loss: 0.3499 - categorical_accuracy: 0.8910
17280/60000 [=======>......................] - ETA: 1:21 - loss: 0.3493 - categorical_accuracy: 0.8912
17312/60000 [=======>......................] - ETA: 1:21 - loss: 0.3491 - categorical_accuracy: 0.8913
17344/60000 [=======>......................] - ETA: 1:21 - loss: 0.3485 - categorical_accuracy: 0.8915
17376/60000 [=======>......................] - ETA: 1:21 - loss: 0.3486 - categorical_accuracy: 0.8915
17408/60000 [=======>......................] - ETA: 1:21 - loss: 0.3481 - categorical_accuracy: 0.8917
17440/60000 [=======>......................] - ETA: 1:21 - loss: 0.3476 - categorical_accuracy: 0.8919
17472/60000 [=======>......................] - ETA: 1:21 - loss: 0.3473 - categorical_accuracy: 0.8919
17504/60000 [=======>......................] - ETA: 1:21 - loss: 0.3470 - categorical_accuracy: 0.8921
17536/60000 [=======>......................] - ETA: 1:21 - loss: 0.3467 - categorical_accuracy: 0.8921
17568/60000 [=======>......................] - ETA: 1:21 - loss: 0.3464 - categorical_accuracy: 0.8922
17600/60000 [=======>......................] - ETA: 1:21 - loss: 0.3460 - categorical_accuracy: 0.8923
17632/60000 [=======>......................] - ETA: 1:21 - loss: 0.3457 - categorical_accuracy: 0.8924
17664/60000 [=======>......................] - ETA: 1:20 - loss: 0.3452 - categorical_accuracy: 0.8925
17696/60000 [=======>......................] - ETA: 1:20 - loss: 0.3450 - categorical_accuracy: 0.8926
17728/60000 [=======>......................] - ETA: 1:20 - loss: 0.3452 - categorical_accuracy: 0.8925
17760/60000 [=======>......................] - ETA: 1:20 - loss: 0.3454 - categorical_accuracy: 0.8925
17792/60000 [=======>......................] - ETA: 1:20 - loss: 0.3452 - categorical_accuracy: 0.8925
17824/60000 [=======>......................] - ETA: 1:20 - loss: 0.3448 - categorical_accuracy: 0.8926
17856/60000 [=======>......................] - ETA: 1:20 - loss: 0.3444 - categorical_accuracy: 0.8928
17888/60000 [=======>......................] - ETA: 1:20 - loss: 0.3438 - categorical_accuracy: 0.8929
17920/60000 [=======>......................] - ETA: 1:20 - loss: 0.3435 - categorical_accuracy: 0.8931
17952/60000 [=======>......................] - ETA: 1:20 - loss: 0.3431 - categorical_accuracy: 0.8932
17984/60000 [=======>......................] - ETA: 1:20 - loss: 0.3426 - categorical_accuracy: 0.8933
18016/60000 [========>.....................] - ETA: 1:20 - loss: 0.3421 - categorical_accuracy: 0.8935
18048/60000 [========>.....................] - ETA: 1:20 - loss: 0.3415 - categorical_accuracy: 0.8937
18080/60000 [========>.....................] - ETA: 1:20 - loss: 0.3412 - categorical_accuracy: 0.8939
18112/60000 [========>.....................] - ETA: 1:20 - loss: 0.3410 - categorical_accuracy: 0.8940
18144/60000 [========>.....................] - ETA: 1:20 - loss: 0.3404 - categorical_accuracy: 0.8942
18176/60000 [========>.....................] - ETA: 1:20 - loss: 0.3402 - categorical_accuracy: 0.8943
18208/60000 [========>.....................] - ETA: 1:19 - loss: 0.3399 - categorical_accuracy: 0.8944
18240/60000 [========>.....................] - ETA: 1:19 - loss: 0.3399 - categorical_accuracy: 0.8945
18272/60000 [========>.....................] - ETA: 1:19 - loss: 0.3398 - categorical_accuracy: 0.8946
18304/60000 [========>.....................] - ETA: 1:19 - loss: 0.3393 - categorical_accuracy: 0.8947
18336/60000 [========>.....................] - ETA: 1:19 - loss: 0.3393 - categorical_accuracy: 0.8947
18368/60000 [========>.....................] - ETA: 1:19 - loss: 0.3392 - categorical_accuracy: 0.8948
18400/60000 [========>.....................] - ETA: 1:19 - loss: 0.3389 - categorical_accuracy: 0.8949
18432/60000 [========>.....................] - ETA: 1:19 - loss: 0.3387 - categorical_accuracy: 0.8950
18464/60000 [========>.....................] - ETA: 1:19 - loss: 0.3383 - categorical_accuracy: 0.8951
18496/60000 [========>.....................] - ETA: 1:19 - loss: 0.3378 - categorical_accuracy: 0.8953
18528/60000 [========>.....................] - ETA: 1:19 - loss: 0.3374 - categorical_accuracy: 0.8954
18560/60000 [========>.....................] - ETA: 1:19 - loss: 0.3371 - categorical_accuracy: 0.8955
18592/60000 [========>.....................] - ETA: 1:19 - loss: 0.3370 - categorical_accuracy: 0.8956
18624/60000 [========>.....................] - ETA: 1:19 - loss: 0.3365 - categorical_accuracy: 0.8958
18656/60000 [========>.....................] - ETA: 1:19 - loss: 0.3361 - categorical_accuracy: 0.8959
18688/60000 [========>.....................] - ETA: 1:18 - loss: 0.3356 - categorical_accuracy: 0.8961
18720/60000 [========>.....................] - ETA: 1:18 - loss: 0.3351 - categorical_accuracy: 0.8962
18752/60000 [========>.....................] - ETA: 1:18 - loss: 0.3347 - categorical_accuracy: 0.8963
18784/60000 [========>.....................] - ETA: 1:18 - loss: 0.3343 - categorical_accuracy: 0.8964
18816/60000 [========>.....................] - ETA: 1:18 - loss: 0.3340 - categorical_accuracy: 0.8965
18848/60000 [========>.....................] - ETA: 1:18 - loss: 0.3338 - categorical_accuracy: 0.8966
18880/60000 [========>.....................] - ETA: 1:18 - loss: 0.3336 - categorical_accuracy: 0.8967
18912/60000 [========>.....................] - ETA: 1:18 - loss: 0.3336 - categorical_accuracy: 0.8967
18944/60000 [========>.....................] - ETA: 1:18 - loss: 0.3333 - categorical_accuracy: 0.8968
18976/60000 [========>.....................] - ETA: 1:18 - loss: 0.3330 - categorical_accuracy: 0.8968
19008/60000 [========>.....................] - ETA: 1:18 - loss: 0.3327 - categorical_accuracy: 0.8968
19040/60000 [========>.....................] - ETA: 1:18 - loss: 0.3324 - categorical_accuracy: 0.8970
19072/60000 [========>.....................] - ETA: 1:18 - loss: 0.3324 - categorical_accuracy: 0.8970
19104/60000 [========>.....................] - ETA: 1:18 - loss: 0.3324 - categorical_accuracy: 0.8969
19136/60000 [========>.....................] - ETA: 1:18 - loss: 0.3321 - categorical_accuracy: 0.8970
19168/60000 [========>.....................] - ETA: 1:18 - loss: 0.3317 - categorical_accuracy: 0.8971
19200/60000 [========>.....................] - ETA: 1:18 - loss: 0.3313 - categorical_accuracy: 0.8972
19232/60000 [========>.....................] - ETA: 1:17 - loss: 0.3308 - categorical_accuracy: 0.8974
19264/60000 [========>.....................] - ETA: 1:17 - loss: 0.3304 - categorical_accuracy: 0.8975
19296/60000 [========>.....................] - ETA: 1:17 - loss: 0.3303 - categorical_accuracy: 0.8975
19328/60000 [========>.....................] - ETA: 1:17 - loss: 0.3299 - categorical_accuracy: 0.8977
19360/60000 [========>.....................] - ETA: 1:17 - loss: 0.3294 - categorical_accuracy: 0.8979
19392/60000 [========>.....................] - ETA: 1:17 - loss: 0.3291 - categorical_accuracy: 0.8979
19424/60000 [========>.....................] - ETA: 1:17 - loss: 0.3286 - categorical_accuracy: 0.8981
19456/60000 [========>.....................] - ETA: 1:17 - loss: 0.3281 - categorical_accuracy: 0.8983
19488/60000 [========>.....................] - ETA: 1:17 - loss: 0.3279 - categorical_accuracy: 0.8984
19520/60000 [========>.....................] - ETA: 1:17 - loss: 0.3275 - categorical_accuracy: 0.8985
19552/60000 [========>.....................] - ETA: 1:17 - loss: 0.3274 - categorical_accuracy: 0.8986
19584/60000 [========>.....................] - ETA: 1:17 - loss: 0.3270 - categorical_accuracy: 0.8988
19616/60000 [========>.....................] - ETA: 1:17 - loss: 0.3268 - categorical_accuracy: 0.8989
19648/60000 [========>.....................] - ETA: 1:17 - loss: 0.3263 - categorical_accuracy: 0.8991
19680/60000 [========>.....................] - ETA: 1:17 - loss: 0.3259 - categorical_accuracy: 0.8992
19712/60000 [========>.....................] - ETA: 1:17 - loss: 0.3255 - categorical_accuracy: 0.8994
19744/60000 [========>.....................] - ETA: 1:16 - loss: 0.3250 - categorical_accuracy: 0.8995
19776/60000 [========>.....................] - ETA: 1:16 - loss: 0.3246 - categorical_accuracy: 0.8997
19808/60000 [========>.....................] - ETA: 1:16 - loss: 0.3243 - categorical_accuracy: 0.8997
19840/60000 [========>.....................] - ETA: 1:16 - loss: 0.3239 - categorical_accuracy: 0.8998
19872/60000 [========>.....................] - ETA: 1:16 - loss: 0.3242 - categorical_accuracy: 0.8997
19904/60000 [========>.....................] - ETA: 1:16 - loss: 0.3239 - categorical_accuracy: 0.8997
19936/60000 [========>.....................] - ETA: 1:16 - loss: 0.3237 - categorical_accuracy: 0.8998
19968/60000 [========>.....................] - ETA: 1:16 - loss: 0.3234 - categorical_accuracy: 0.8999
20000/60000 [=========>....................] - ETA: 1:16 - loss: 0.3229 - categorical_accuracy: 0.9000
20032/60000 [=========>....................] - ETA: 1:16 - loss: 0.3227 - categorical_accuracy: 0.9002
20064/60000 [=========>....................] - ETA: 1:16 - loss: 0.3224 - categorical_accuracy: 0.9003
20096/60000 [=========>....................] - ETA: 1:16 - loss: 0.3221 - categorical_accuracy: 0.9004
20128/60000 [=========>....................] - ETA: 1:16 - loss: 0.3216 - categorical_accuracy: 0.9005
20160/60000 [=========>....................] - ETA: 1:16 - loss: 0.3211 - categorical_accuracy: 0.9007
20192/60000 [=========>....................] - ETA: 1:16 - loss: 0.3206 - categorical_accuracy: 0.9009
20224/60000 [=========>....................] - ETA: 1:15 - loss: 0.3202 - categorical_accuracy: 0.9010
20256/60000 [=========>....................] - ETA: 1:15 - loss: 0.3200 - categorical_accuracy: 0.9011
20288/60000 [=========>....................] - ETA: 1:15 - loss: 0.3196 - categorical_accuracy: 0.9012
20320/60000 [=========>....................] - ETA: 1:15 - loss: 0.3198 - categorical_accuracy: 0.9012
20352/60000 [=========>....................] - ETA: 1:15 - loss: 0.3194 - categorical_accuracy: 0.9013
20384/60000 [=========>....................] - ETA: 1:15 - loss: 0.3192 - categorical_accuracy: 0.9013
20416/60000 [=========>....................] - ETA: 1:15 - loss: 0.3187 - categorical_accuracy: 0.9015
20448/60000 [=========>....................] - ETA: 1:15 - loss: 0.3183 - categorical_accuracy: 0.9017
20480/60000 [=========>....................] - ETA: 1:15 - loss: 0.3180 - categorical_accuracy: 0.9017
20512/60000 [=========>....................] - ETA: 1:15 - loss: 0.3176 - categorical_accuracy: 0.9018
20544/60000 [=========>....................] - ETA: 1:15 - loss: 0.3172 - categorical_accuracy: 0.9020
20576/60000 [=========>....................] - ETA: 1:15 - loss: 0.3167 - categorical_accuracy: 0.9021
20608/60000 [=========>....................] - ETA: 1:15 - loss: 0.3167 - categorical_accuracy: 0.9022
20640/60000 [=========>....................] - ETA: 1:15 - loss: 0.3167 - categorical_accuracy: 0.9022
20672/60000 [=========>....................] - ETA: 1:15 - loss: 0.3166 - categorical_accuracy: 0.9022
20704/60000 [=========>....................] - ETA: 1:15 - loss: 0.3162 - categorical_accuracy: 0.9022
20736/60000 [=========>....................] - ETA: 1:14 - loss: 0.3161 - categorical_accuracy: 0.9022
20768/60000 [=========>....................] - ETA: 1:14 - loss: 0.3157 - categorical_accuracy: 0.9024
20800/60000 [=========>....................] - ETA: 1:14 - loss: 0.3156 - categorical_accuracy: 0.9025
20832/60000 [=========>....................] - ETA: 1:14 - loss: 0.3153 - categorical_accuracy: 0.9026
20864/60000 [=========>....................] - ETA: 1:14 - loss: 0.3149 - categorical_accuracy: 0.9028
20896/60000 [=========>....................] - ETA: 1:14 - loss: 0.3153 - categorical_accuracy: 0.9027
20928/60000 [=========>....................] - ETA: 1:14 - loss: 0.3150 - categorical_accuracy: 0.9028
20960/60000 [=========>....................] - ETA: 1:14 - loss: 0.3146 - categorical_accuracy: 0.9029
20992/60000 [=========>....................] - ETA: 1:14 - loss: 0.3142 - categorical_accuracy: 0.9030
21024/60000 [=========>....................] - ETA: 1:14 - loss: 0.3141 - categorical_accuracy: 0.9030
21056/60000 [=========>....................] - ETA: 1:14 - loss: 0.3142 - categorical_accuracy: 0.9030
21088/60000 [=========>....................] - ETA: 1:14 - loss: 0.3137 - categorical_accuracy: 0.9032
21120/60000 [=========>....................] - ETA: 1:14 - loss: 0.3134 - categorical_accuracy: 0.9033
21152/60000 [=========>....................] - ETA: 1:14 - loss: 0.3130 - categorical_accuracy: 0.9034
21184/60000 [=========>....................] - ETA: 1:14 - loss: 0.3127 - categorical_accuracy: 0.9035
21216/60000 [=========>....................] - ETA: 1:14 - loss: 0.3123 - categorical_accuracy: 0.9036
21248/60000 [=========>....................] - ETA: 1:13 - loss: 0.3123 - categorical_accuracy: 0.9037
21280/60000 [=========>....................] - ETA: 1:13 - loss: 0.3121 - categorical_accuracy: 0.9037
21312/60000 [=========>....................] - ETA: 1:13 - loss: 0.3119 - categorical_accuracy: 0.9038
21344/60000 [=========>....................] - ETA: 1:13 - loss: 0.3115 - categorical_accuracy: 0.9039
21376/60000 [=========>....................] - ETA: 1:13 - loss: 0.3113 - categorical_accuracy: 0.9039
21408/60000 [=========>....................] - ETA: 1:13 - loss: 0.3110 - categorical_accuracy: 0.9040
21440/60000 [=========>....................] - ETA: 1:13 - loss: 0.3107 - categorical_accuracy: 0.9041
21472/60000 [=========>....................] - ETA: 1:13 - loss: 0.3106 - categorical_accuracy: 0.9041
21504/60000 [=========>....................] - ETA: 1:13 - loss: 0.3103 - categorical_accuracy: 0.9043
21536/60000 [=========>....................] - ETA: 1:13 - loss: 0.3099 - categorical_accuracy: 0.9043
21568/60000 [=========>....................] - ETA: 1:13 - loss: 0.3096 - categorical_accuracy: 0.9044
21600/60000 [=========>....................] - ETA: 1:13 - loss: 0.3092 - categorical_accuracy: 0.9045
21632/60000 [=========>....................] - ETA: 1:13 - loss: 0.3090 - categorical_accuracy: 0.9046
21664/60000 [=========>....................] - ETA: 1:13 - loss: 0.3088 - categorical_accuracy: 0.9046
21696/60000 [=========>....................] - ETA: 1:13 - loss: 0.3085 - categorical_accuracy: 0.9047
21728/60000 [=========>....................] - ETA: 1:13 - loss: 0.3084 - categorical_accuracy: 0.9048
21760/60000 [=========>....................] - ETA: 1:12 - loss: 0.3083 - categorical_accuracy: 0.9048
21792/60000 [=========>....................] - ETA: 1:12 - loss: 0.3084 - categorical_accuracy: 0.9048
21824/60000 [=========>....................] - ETA: 1:12 - loss: 0.3081 - categorical_accuracy: 0.9049
21856/60000 [=========>....................] - ETA: 1:12 - loss: 0.3077 - categorical_accuracy: 0.9050
21888/60000 [=========>....................] - ETA: 1:12 - loss: 0.3075 - categorical_accuracy: 0.9051
21920/60000 [=========>....................] - ETA: 1:12 - loss: 0.3072 - categorical_accuracy: 0.9052
21952/60000 [=========>....................] - ETA: 1:12 - loss: 0.3070 - categorical_accuracy: 0.9052
21984/60000 [=========>....................] - ETA: 1:12 - loss: 0.3067 - categorical_accuracy: 0.9052
22016/60000 [==========>...................] - ETA: 1:12 - loss: 0.3065 - categorical_accuracy: 0.9053
22048/60000 [==========>...................] - ETA: 1:12 - loss: 0.3063 - categorical_accuracy: 0.9054
22080/60000 [==========>...................] - ETA: 1:12 - loss: 0.3059 - categorical_accuracy: 0.9055
22112/60000 [==========>...................] - ETA: 1:12 - loss: 0.3055 - categorical_accuracy: 0.9057
22144/60000 [==========>...................] - ETA: 1:12 - loss: 0.3052 - categorical_accuracy: 0.9058
22176/60000 [==========>...................] - ETA: 1:12 - loss: 0.3048 - categorical_accuracy: 0.9059
22208/60000 [==========>...................] - ETA: 1:12 - loss: 0.3046 - categorical_accuracy: 0.9060
22240/60000 [==========>...................] - ETA: 1:12 - loss: 0.3042 - categorical_accuracy: 0.9061
22272/60000 [==========>...................] - ETA: 1:12 - loss: 0.3040 - categorical_accuracy: 0.9062
22304/60000 [==========>...................] - ETA: 1:11 - loss: 0.3036 - categorical_accuracy: 0.9063
22336/60000 [==========>...................] - ETA: 1:11 - loss: 0.3035 - categorical_accuracy: 0.9063
22368/60000 [==========>...................] - ETA: 1:11 - loss: 0.3033 - categorical_accuracy: 0.9064
22400/60000 [==========>...................] - ETA: 1:11 - loss: 0.3030 - categorical_accuracy: 0.9065
22432/60000 [==========>...................] - ETA: 1:11 - loss: 0.3030 - categorical_accuracy: 0.9065
22464/60000 [==========>...................] - ETA: 1:11 - loss: 0.3027 - categorical_accuracy: 0.9065
22496/60000 [==========>...................] - ETA: 1:11 - loss: 0.3024 - categorical_accuracy: 0.9066
22528/60000 [==========>...................] - ETA: 1:11 - loss: 0.3023 - categorical_accuracy: 0.9066
22560/60000 [==========>...................] - ETA: 1:11 - loss: 0.3019 - categorical_accuracy: 0.9067
22592/60000 [==========>...................] - ETA: 1:11 - loss: 0.3017 - categorical_accuracy: 0.9067
22624/60000 [==========>...................] - ETA: 1:11 - loss: 0.3014 - categorical_accuracy: 0.9069
22656/60000 [==========>...................] - ETA: 1:11 - loss: 0.3010 - categorical_accuracy: 0.9070
22688/60000 [==========>...................] - ETA: 1:11 - loss: 0.3007 - categorical_accuracy: 0.9071
22720/60000 [==========>...................] - ETA: 1:11 - loss: 0.3003 - categorical_accuracy: 0.9072
22752/60000 [==========>...................] - ETA: 1:11 - loss: 0.3002 - categorical_accuracy: 0.9072
22784/60000 [==========>...................] - ETA: 1:11 - loss: 0.2998 - categorical_accuracy: 0.9073
22816/60000 [==========>...................] - ETA: 1:10 - loss: 0.2994 - categorical_accuracy: 0.9075
22848/60000 [==========>...................] - ETA: 1:10 - loss: 0.2991 - categorical_accuracy: 0.9075
22880/60000 [==========>...................] - ETA: 1:10 - loss: 0.2988 - categorical_accuracy: 0.9076
22912/60000 [==========>...................] - ETA: 1:10 - loss: 0.2985 - categorical_accuracy: 0.9077
22944/60000 [==========>...................] - ETA: 1:10 - loss: 0.2984 - categorical_accuracy: 0.9078
22976/60000 [==========>...................] - ETA: 1:10 - loss: 0.2980 - categorical_accuracy: 0.9079
23008/60000 [==========>...................] - ETA: 1:10 - loss: 0.2978 - categorical_accuracy: 0.9079
23040/60000 [==========>...................] - ETA: 1:10 - loss: 0.2974 - categorical_accuracy: 0.9080
23072/60000 [==========>...................] - ETA: 1:10 - loss: 0.2973 - categorical_accuracy: 0.9080
23104/60000 [==========>...................] - ETA: 1:10 - loss: 0.2969 - categorical_accuracy: 0.9082
23136/60000 [==========>...................] - ETA: 1:10 - loss: 0.2968 - categorical_accuracy: 0.9082
23168/60000 [==========>...................] - ETA: 1:10 - loss: 0.2965 - categorical_accuracy: 0.9083
23200/60000 [==========>...................] - ETA: 1:10 - loss: 0.2965 - categorical_accuracy: 0.9083
23232/60000 [==========>...................] - ETA: 1:10 - loss: 0.2961 - categorical_accuracy: 0.9084
23264/60000 [==========>...................] - ETA: 1:10 - loss: 0.2958 - categorical_accuracy: 0.9085
23296/60000 [==========>...................] - ETA: 1:10 - loss: 0.2958 - categorical_accuracy: 0.9085
23328/60000 [==========>...................] - ETA: 1:09 - loss: 0.2955 - categorical_accuracy: 0.9086
23360/60000 [==========>...................] - ETA: 1:09 - loss: 0.2953 - categorical_accuracy: 0.9086
23392/60000 [==========>...................] - ETA: 1:09 - loss: 0.2949 - categorical_accuracy: 0.9088
23424/60000 [==========>...................] - ETA: 1:09 - loss: 0.2946 - categorical_accuracy: 0.9088
23456/60000 [==========>...................] - ETA: 1:09 - loss: 0.2944 - categorical_accuracy: 0.9089
23488/60000 [==========>...................] - ETA: 1:09 - loss: 0.2941 - categorical_accuracy: 0.9090
23520/60000 [==========>...................] - ETA: 1:09 - loss: 0.2940 - categorical_accuracy: 0.9090
23552/60000 [==========>...................] - ETA: 1:09 - loss: 0.2937 - categorical_accuracy: 0.9091
23584/60000 [==========>...................] - ETA: 1:09 - loss: 0.2936 - categorical_accuracy: 0.9091
23616/60000 [==========>...................] - ETA: 1:09 - loss: 0.2933 - categorical_accuracy: 0.9092
23648/60000 [==========>...................] - ETA: 1:09 - loss: 0.2931 - categorical_accuracy: 0.9093
23680/60000 [==========>...................] - ETA: 1:09 - loss: 0.2927 - categorical_accuracy: 0.9094
23712/60000 [==========>...................] - ETA: 1:09 - loss: 0.2925 - categorical_accuracy: 0.9094
23744/60000 [==========>...................] - ETA: 1:09 - loss: 0.2921 - categorical_accuracy: 0.9095
23776/60000 [==========>...................] - ETA: 1:09 - loss: 0.2917 - categorical_accuracy: 0.9097
23808/60000 [==========>...................] - ETA: 1:09 - loss: 0.2916 - categorical_accuracy: 0.9097
23840/60000 [==========>...................] - ETA: 1:08 - loss: 0.2913 - categorical_accuracy: 0.9098
23872/60000 [==========>...................] - ETA: 1:08 - loss: 0.2912 - categorical_accuracy: 0.9098
23904/60000 [==========>...................] - ETA: 1:08 - loss: 0.2909 - categorical_accuracy: 0.9099
23936/60000 [==========>...................] - ETA: 1:08 - loss: 0.2907 - categorical_accuracy: 0.9099
23968/60000 [==========>...................] - ETA: 1:08 - loss: 0.2903 - categorical_accuracy: 0.9100
24000/60000 [===========>..................] - ETA: 1:08 - loss: 0.2900 - categorical_accuracy: 0.9102
24032/60000 [===========>..................] - ETA: 1:08 - loss: 0.2901 - categorical_accuracy: 0.9102
24064/60000 [===========>..................] - ETA: 1:08 - loss: 0.2901 - categorical_accuracy: 0.9102
24096/60000 [===========>..................] - ETA: 1:08 - loss: 0.2899 - categorical_accuracy: 0.9102
24128/60000 [===========>..................] - ETA: 1:08 - loss: 0.2896 - categorical_accuracy: 0.9103
24160/60000 [===========>..................] - ETA: 1:08 - loss: 0.2893 - categorical_accuracy: 0.9104
24192/60000 [===========>..................] - ETA: 1:08 - loss: 0.2889 - categorical_accuracy: 0.9105
24224/60000 [===========>..................] - ETA: 1:08 - loss: 0.2887 - categorical_accuracy: 0.9106
24256/60000 [===========>..................] - ETA: 1:08 - loss: 0.2885 - categorical_accuracy: 0.9106
24288/60000 [===========>..................] - ETA: 1:08 - loss: 0.2881 - categorical_accuracy: 0.9107
24320/60000 [===========>..................] - ETA: 1:08 - loss: 0.2878 - categorical_accuracy: 0.9109
24352/60000 [===========>..................] - ETA: 1:07 - loss: 0.2875 - categorical_accuracy: 0.9110
24384/60000 [===========>..................] - ETA: 1:07 - loss: 0.2872 - categorical_accuracy: 0.9110
24416/60000 [===========>..................] - ETA: 1:07 - loss: 0.2871 - categorical_accuracy: 0.9111
24448/60000 [===========>..................] - ETA: 1:07 - loss: 0.2871 - categorical_accuracy: 0.9111
24480/60000 [===========>..................] - ETA: 1:07 - loss: 0.2870 - categorical_accuracy: 0.9112
24512/60000 [===========>..................] - ETA: 1:07 - loss: 0.2867 - categorical_accuracy: 0.9113
24544/60000 [===========>..................] - ETA: 1:07 - loss: 0.2866 - categorical_accuracy: 0.9113
24576/60000 [===========>..................] - ETA: 1:07 - loss: 0.2863 - categorical_accuracy: 0.9114
24608/60000 [===========>..................] - ETA: 1:07 - loss: 0.2862 - categorical_accuracy: 0.9115
24640/60000 [===========>..................] - ETA: 1:07 - loss: 0.2858 - categorical_accuracy: 0.9116
24672/60000 [===========>..................] - ETA: 1:07 - loss: 0.2857 - categorical_accuracy: 0.9116
24704/60000 [===========>..................] - ETA: 1:07 - loss: 0.2854 - categorical_accuracy: 0.9116
24736/60000 [===========>..................] - ETA: 1:07 - loss: 0.2852 - categorical_accuracy: 0.9117
24768/60000 [===========>..................] - ETA: 1:07 - loss: 0.2850 - categorical_accuracy: 0.9117
24800/60000 [===========>..................] - ETA: 1:07 - loss: 0.2850 - categorical_accuracy: 0.9117
24832/60000 [===========>..................] - ETA: 1:07 - loss: 0.2848 - categorical_accuracy: 0.9118
24864/60000 [===========>..................] - ETA: 1:07 - loss: 0.2845 - categorical_accuracy: 0.9118
24896/60000 [===========>..................] - ETA: 1:06 - loss: 0.2842 - categorical_accuracy: 0.9119
24928/60000 [===========>..................] - ETA: 1:06 - loss: 0.2840 - categorical_accuracy: 0.9120
24960/60000 [===========>..................] - ETA: 1:06 - loss: 0.2837 - categorical_accuracy: 0.9121
24992/60000 [===========>..................] - ETA: 1:06 - loss: 0.2835 - categorical_accuracy: 0.9121
25024/60000 [===========>..................] - ETA: 1:06 - loss: 0.2833 - categorical_accuracy: 0.9122
25056/60000 [===========>..................] - ETA: 1:06 - loss: 0.2830 - categorical_accuracy: 0.9123
25088/60000 [===========>..................] - ETA: 1:06 - loss: 0.2827 - categorical_accuracy: 0.9124
25120/60000 [===========>..................] - ETA: 1:06 - loss: 0.2826 - categorical_accuracy: 0.9125
25152/60000 [===========>..................] - ETA: 1:06 - loss: 0.2826 - categorical_accuracy: 0.9125
25184/60000 [===========>..................] - ETA: 1:06 - loss: 0.2826 - categorical_accuracy: 0.9125
25216/60000 [===========>..................] - ETA: 1:06 - loss: 0.2825 - categorical_accuracy: 0.9126
25248/60000 [===========>..................] - ETA: 1:06 - loss: 0.2824 - categorical_accuracy: 0.9126
25280/60000 [===========>..................] - ETA: 1:06 - loss: 0.2823 - categorical_accuracy: 0.9126
25312/60000 [===========>..................] - ETA: 1:06 - loss: 0.2827 - categorical_accuracy: 0.9125
25344/60000 [===========>..................] - ETA: 1:06 - loss: 0.2823 - categorical_accuracy: 0.9126
25376/60000 [===========>..................] - ETA: 1:06 - loss: 0.2821 - categorical_accuracy: 0.9126
25408/60000 [===========>..................] - ETA: 1:05 - loss: 0.2819 - categorical_accuracy: 0.9127
25440/60000 [===========>..................] - ETA: 1:05 - loss: 0.2820 - categorical_accuracy: 0.9127
25472/60000 [===========>..................] - ETA: 1:05 - loss: 0.2818 - categorical_accuracy: 0.9128
25504/60000 [===========>..................] - ETA: 1:05 - loss: 0.2816 - categorical_accuracy: 0.9128
25536/60000 [===========>..................] - ETA: 1:05 - loss: 0.2814 - categorical_accuracy: 0.9128
25568/60000 [===========>..................] - ETA: 1:05 - loss: 0.2815 - categorical_accuracy: 0.9129
25600/60000 [===========>..................] - ETA: 1:05 - loss: 0.2813 - categorical_accuracy: 0.9129
25632/60000 [===========>..................] - ETA: 1:05 - loss: 0.2810 - categorical_accuracy: 0.9130
25664/60000 [===========>..................] - ETA: 1:05 - loss: 0.2809 - categorical_accuracy: 0.9130
25696/60000 [===========>..................] - ETA: 1:05 - loss: 0.2807 - categorical_accuracy: 0.9131
25728/60000 [===========>..................] - ETA: 1:05 - loss: 0.2805 - categorical_accuracy: 0.9132
25760/60000 [===========>..................] - ETA: 1:05 - loss: 0.2805 - categorical_accuracy: 0.9132
25792/60000 [===========>..................] - ETA: 1:05 - loss: 0.2803 - categorical_accuracy: 0.9132
25824/60000 [===========>..................] - ETA: 1:05 - loss: 0.2800 - categorical_accuracy: 0.9133
25856/60000 [===========>..................] - ETA: 1:05 - loss: 0.2803 - categorical_accuracy: 0.9133
25888/60000 [===========>..................] - ETA: 1:05 - loss: 0.2800 - categorical_accuracy: 0.9134
25920/60000 [===========>..................] - ETA: 1:04 - loss: 0.2798 - categorical_accuracy: 0.9135
25952/60000 [===========>..................] - ETA: 1:04 - loss: 0.2795 - categorical_accuracy: 0.9136
25984/60000 [===========>..................] - ETA: 1:04 - loss: 0.2792 - categorical_accuracy: 0.9138
26016/60000 [============>.................] - ETA: 1:04 - loss: 0.2789 - categorical_accuracy: 0.9139
26048/60000 [============>.................] - ETA: 1:04 - loss: 0.2786 - categorical_accuracy: 0.9140
26080/60000 [============>.................] - ETA: 1:04 - loss: 0.2783 - categorical_accuracy: 0.9141
26112/60000 [============>.................] - ETA: 1:04 - loss: 0.2781 - categorical_accuracy: 0.9141
26144/60000 [============>.................] - ETA: 1:04 - loss: 0.2780 - categorical_accuracy: 0.9142
26176/60000 [============>.................] - ETA: 1:04 - loss: 0.2779 - categorical_accuracy: 0.9142
26208/60000 [============>.................] - ETA: 1:04 - loss: 0.2777 - categorical_accuracy: 0.9143
26240/60000 [============>.................] - ETA: 1:04 - loss: 0.2778 - categorical_accuracy: 0.9143
26272/60000 [============>.................] - ETA: 1:04 - loss: 0.2776 - categorical_accuracy: 0.9144
26304/60000 [============>.................] - ETA: 1:04 - loss: 0.2773 - categorical_accuracy: 0.9145
26336/60000 [============>.................] - ETA: 1:04 - loss: 0.2772 - categorical_accuracy: 0.9145
26368/60000 [============>.................] - ETA: 1:04 - loss: 0.2771 - categorical_accuracy: 0.9146
26400/60000 [============>.................] - ETA: 1:04 - loss: 0.2768 - categorical_accuracy: 0.9147
26432/60000 [============>.................] - ETA: 1:03 - loss: 0.2768 - categorical_accuracy: 0.9147
26464/60000 [============>.................] - ETA: 1:03 - loss: 0.2765 - categorical_accuracy: 0.9148
26496/60000 [============>.................] - ETA: 1:03 - loss: 0.2762 - categorical_accuracy: 0.9149
26528/60000 [============>.................] - ETA: 1:03 - loss: 0.2763 - categorical_accuracy: 0.9148
26560/60000 [============>.................] - ETA: 1:03 - loss: 0.2761 - categorical_accuracy: 0.9149
26592/60000 [============>.................] - ETA: 1:03 - loss: 0.2761 - categorical_accuracy: 0.9149
26624/60000 [============>.................] - ETA: 1:03 - loss: 0.2759 - categorical_accuracy: 0.9150
26656/60000 [============>.................] - ETA: 1:03 - loss: 0.2757 - categorical_accuracy: 0.9151
26688/60000 [============>.................] - ETA: 1:03 - loss: 0.2755 - categorical_accuracy: 0.9151
26720/60000 [============>.................] - ETA: 1:03 - loss: 0.2752 - categorical_accuracy: 0.9152
26752/60000 [============>.................] - ETA: 1:03 - loss: 0.2751 - categorical_accuracy: 0.9152
26784/60000 [============>.................] - ETA: 1:03 - loss: 0.2748 - categorical_accuracy: 0.9153
26816/60000 [============>.................] - ETA: 1:03 - loss: 0.2746 - categorical_accuracy: 0.9153
26848/60000 [============>.................] - ETA: 1:03 - loss: 0.2744 - categorical_accuracy: 0.9154
26880/60000 [============>.................] - ETA: 1:03 - loss: 0.2744 - categorical_accuracy: 0.9154
26912/60000 [============>.................] - ETA: 1:03 - loss: 0.2741 - categorical_accuracy: 0.9155
26944/60000 [============>.................] - ETA: 1:03 - loss: 0.2739 - categorical_accuracy: 0.9156
26976/60000 [============>.................] - ETA: 1:02 - loss: 0.2739 - categorical_accuracy: 0.9156
27008/60000 [============>.................] - ETA: 1:02 - loss: 0.2736 - categorical_accuracy: 0.9157
27040/60000 [============>.................] - ETA: 1:02 - loss: 0.2734 - categorical_accuracy: 0.9158
27072/60000 [============>.................] - ETA: 1:02 - loss: 0.2735 - categorical_accuracy: 0.9158
27104/60000 [============>.................] - ETA: 1:02 - loss: 0.2734 - categorical_accuracy: 0.9158
27136/60000 [============>.................] - ETA: 1:02 - loss: 0.2732 - categorical_accuracy: 0.9158
27168/60000 [============>.................] - ETA: 1:02 - loss: 0.2729 - categorical_accuracy: 0.9159
27200/60000 [============>.................] - ETA: 1:02 - loss: 0.2727 - categorical_accuracy: 0.9160
27232/60000 [============>.................] - ETA: 1:02 - loss: 0.2724 - categorical_accuracy: 0.9161
27264/60000 [============>.................] - ETA: 1:02 - loss: 0.2722 - categorical_accuracy: 0.9162
27296/60000 [============>.................] - ETA: 1:02 - loss: 0.2723 - categorical_accuracy: 0.9161
27328/60000 [============>.................] - ETA: 1:02 - loss: 0.2721 - categorical_accuracy: 0.9162
27360/60000 [============>.................] - ETA: 1:02 - loss: 0.2719 - categorical_accuracy: 0.9162
27392/60000 [============>.................] - ETA: 1:02 - loss: 0.2717 - categorical_accuracy: 0.9163
27424/60000 [============>.................] - ETA: 1:02 - loss: 0.2714 - categorical_accuracy: 0.9164
27456/60000 [============>.................] - ETA: 1:02 - loss: 0.2713 - categorical_accuracy: 0.9164
27488/60000 [============>.................] - ETA: 1:01 - loss: 0.2712 - categorical_accuracy: 0.9164
27520/60000 [============>.................] - ETA: 1:01 - loss: 0.2709 - categorical_accuracy: 0.9165
27552/60000 [============>.................] - ETA: 1:01 - loss: 0.2707 - categorical_accuracy: 0.9165
27584/60000 [============>.................] - ETA: 1:01 - loss: 0.2705 - categorical_accuracy: 0.9166
27616/60000 [============>.................] - ETA: 1:01 - loss: 0.2702 - categorical_accuracy: 0.9167
27648/60000 [============>.................] - ETA: 1:01 - loss: 0.2700 - categorical_accuracy: 0.9167
27680/60000 [============>.................] - ETA: 1:01 - loss: 0.2698 - categorical_accuracy: 0.9168
27712/60000 [============>.................] - ETA: 1:01 - loss: 0.2699 - categorical_accuracy: 0.9168
27744/60000 [============>.................] - ETA: 1:01 - loss: 0.2696 - categorical_accuracy: 0.9169
27776/60000 [============>.................] - ETA: 1:01 - loss: 0.2693 - categorical_accuracy: 0.9170
27808/60000 [============>.................] - ETA: 1:01 - loss: 0.2692 - categorical_accuracy: 0.9170
27840/60000 [============>.................] - ETA: 1:01 - loss: 0.2691 - categorical_accuracy: 0.9171
27872/60000 [============>.................] - ETA: 1:01 - loss: 0.2689 - categorical_accuracy: 0.9172
27904/60000 [============>.................] - ETA: 1:01 - loss: 0.2688 - categorical_accuracy: 0.9172
27936/60000 [============>.................] - ETA: 1:01 - loss: 0.2685 - categorical_accuracy: 0.9173
27968/60000 [============>.................] - ETA: 1:01 - loss: 0.2683 - categorical_accuracy: 0.9174
28000/60000 [=============>................] - ETA: 1:00 - loss: 0.2681 - categorical_accuracy: 0.9174
28032/60000 [=============>................] - ETA: 1:00 - loss: 0.2678 - categorical_accuracy: 0.9175
28064/60000 [=============>................] - ETA: 1:00 - loss: 0.2676 - categorical_accuracy: 0.9176
28096/60000 [=============>................] - ETA: 1:00 - loss: 0.2673 - categorical_accuracy: 0.9176
28128/60000 [=============>................] - ETA: 1:00 - loss: 0.2671 - categorical_accuracy: 0.9177
28160/60000 [=============>................] - ETA: 1:00 - loss: 0.2670 - categorical_accuracy: 0.9176
28192/60000 [=============>................] - ETA: 1:00 - loss: 0.2668 - categorical_accuracy: 0.9177
28224/60000 [=============>................] - ETA: 1:00 - loss: 0.2665 - categorical_accuracy: 0.9178
28256/60000 [=============>................] - ETA: 1:00 - loss: 0.2662 - categorical_accuracy: 0.9179
28288/60000 [=============>................] - ETA: 1:00 - loss: 0.2661 - categorical_accuracy: 0.9180
28320/60000 [=============>................] - ETA: 1:00 - loss: 0.2659 - categorical_accuracy: 0.9180
28352/60000 [=============>................] - ETA: 1:00 - loss: 0.2660 - categorical_accuracy: 0.9180
28384/60000 [=============>................] - ETA: 1:00 - loss: 0.2659 - categorical_accuracy: 0.9181
28416/60000 [=============>................] - ETA: 1:00 - loss: 0.2658 - categorical_accuracy: 0.9181
28448/60000 [=============>................] - ETA: 1:00 - loss: 0.2656 - categorical_accuracy: 0.9182
28480/60000 [=============>................] - ETA: 1:00 - loss: 0.2653 - categorical_accuracy: 0.9183
28512/60000 [=============>................] - ETA: 59s - loss: 0.2652 - categorical_accuracy: 0.9184 
28544/60000 [=============>................] - ETA: 59s - loss: 0.2649 - categorical_accuracy: 0.9184
28576/60000 [=============>................] - ETA: 59s - loss: 0.2648 - categorical_accuracy: 0.9185
28608/60000 [=============>................] - ETA: 59s - loss: 0.2649 - categorical_accuracy: 0.9185
28640/60000 [=============>................] - ETA: 59s - loss: 0.2647 - categorical_accuracy: 0.9186
28672/60000 [=============>................] - ETA: 59s - loss: 0.2645 - categorical_accuracy: 0.9186
28704/60000 [=============>................] - ETA: 59s - loss: 0.2642 - categorical_accuracy: 0.9187
28736/60000 [=============>................] - ETA: 59s - loss: 0.2640 - categorical_accuracy: 0.9188
28768/60000 [=============>................] - ETA: 59s - loss: 0.2637 - categorical_accuracy: 0.9188
28800/60000 [=============>................] - ETA: 59s - loss: 0.2639 - categorical_accuracy: 0.9188
28832/60000 [=============>................] - ETA: 59s - loss: 0.2637 - categorical_accuracy: 0.9189
28864/60000 [=============>................] - ETA: 59s - loss: 0.2635 - categorical_accuracy: 0.9189
28896/60000 [=============>................] - ETA: 59s - loss: 0.2633 - categorical_accuracy: 0.9190
28928/60000 [=============>................] - ETA: 59s - loss: 0.2631 - categorical_accuracy: 0.9190
28960/60000 [=============>................] - ETA: 59s - loss: 0.2630 - categorical_accuracy: 0.9190
28992/60000 [=============>................] - ETA: 59s - loss: 0.2628 - categorical_accuracy: 0.9191
29024/60000 [=============>................] - ETA: 58s - loss: 0.2626 - categorical_accuracy: 0.9192
29056/60000 [=============>................] - ETA: 58s - loss: 0.2624 - categorical_accuracy: 0.9192
29088/60000 [=============>................] - ETA: 58s - loss: 0.2624 - categorical_accuracy: 0.9193
29120/60000 [=============>................] - ETA: 58s - loss: 0.2621 - categorical_accuracy: 0.9194
29152/60000 [=============>................] - ETA: 58s - loss: 0.2620 - categorical_accuracy: 0.9194
29184/60000 [=============>................] - ETA: 58s - loss: 0.2617 - categorical_accuracy: 0.9195
29216/60000 [=============>................] - ETA: 58s - loss: 0.2615 - categorical_accuracy: 0.9195
29248/60000 [=============>................] - ETA: 58s - loss: 0.2613 - categorical_accuracy: 0.9196
29280/60000 [=============>................] - ETA: 58s - loss: 0.2611 - categorical_accuracy: 0.9196
29312/60000 [=============>................] - ETA: 58s - loss: 0.2610 - categorical_accuracy: 0.9197
29344/60000 [=============>................] - ETA: 58s - loss: 0.2607 - categorical_accuracy: 0.9197
29376/60000 [=============>................] - ETA: 58s - loss: 0.2604 - categorical_accuracy: 0.9198
29408/60000 [=============>................] - ETA: 58s - loss: 0.2602 - categorical_accuracy: 0.9199
29440/60000 [=============>................] - ETA: 58s - loss: 0.2599 - categorical_accuracy: 0.9200
29472/60000 [=============>................] - ETA: 58s - loss: 0.2597 - categorical_accuracy: 0.9201
29504/60000 [=============>................] - ETA: 58s - loss: 0.2597 - categorical_accuracy: 0.9201
29536/60000 [=============>................] - ETA: 57s - loss: 0.2595 - categorical_accuracy: 0.9201
29568/60000 [=============>................] - ETA: 57s - loss: 0.2593 - categorical_accuracy: 0.9202
29600/60000 [=============>................] - ETA: 57s - loss: 0.2591 - categorical_accuracy: 0.9203
29632/60000 [=============>................] - ETA: 57s - loss: 0.2589 - categorical_accuracy: 0.9204
29664/60000 [=============>................] - ETA: 57s - loss: 0.2587 - categorical_accuracy: 0.9204
29696/60000 [=============>................] - ETA: 57s - loss: 0.2584 - categorical_accuracy: 0.9205
29728/60000 [=============>................] - ETA: 57s - loss: 0.2582 - categorical_accuracy: 0.9205
29760/60000 [=============>................] - ETA: 57s - loss: 0.2582 - categorical_accuracy: 0.9205
29792/60000 [=============>................] - ETA: 57s - loss: 0.2579 - categorical_accuracy: 0.9206
29824/60000 [=============>................] - ETA: 57s - loss: 0.2577 - categorical_accuracy: 0.9207
29856/60000 [=============>................] - ETA: 57s - loss: 0.2575 - categorical_accuracy: 0.9208
29888/60000 [=============>................] - ETA: 57s - loss: 0.2573 - categorical_accuracy: 0.9208
29920/60000 [=============>................] - ETA: 57s - loss: 0.2571 - categorical_accuracy: 0.9209
29952/60000 [=============>................] - ETA: 57s - loss: 0.2571 - categorical_accuracy: 0.9209
29984/60000 [=============>................] - ETA: 57s - loss: 0.2570 - categorical_accuracy: 0.9209
30016/60000 [==============>...............] - ETA: 57s - loss: 0.2572 - categorical_accuracy: 0.9209
30048/60000 [==============>...............] - ETA: 56s - loss: 0.2570 - categorical_accuracy: 0.9209
30080/60000 [==============>...............] - ETA: 56s - loss: 0.2569 - categorical_accuracy: 0.9210
30112/60000 [==============>...............] - ETA: 56s - loss: 0.2567 - categorical_accuracy: 0.9211
30144/60000 [==============>...............] - ETA: 56s - loss: 0.2564 - categorical_accuracy: 0.9211
30176/60000 [==============>...............] - ETA: 56s - loss: 0.2562 - categorical_accuracy: 0.9212
30208/60000 [==============>...............] - ETA: 56s - loss: 0.2562 - categorical_accuracy: 0.9212
30240/60000 [==============>...............] - ETA: 56s - loss: 0.2560 - categorical_accuracy: 0.9213
30272/60000 [==============>...............] - ETA: 56s - loss: 0.2557 - categorical_accuracy: 0.9214
30304/60000 [==============>...............] - ETA: 56s - loss: 0.2555 - categorical_accuracy: 0.9215
30336/60000 [==============>...............] - ETA: 56s - loss: 0.2554 - categorical_accuracy: 0.9215
30368/60000 [==============>...............] - ETA: 56s - loss: 0.2554 - categorical_accuracy: 0.9216
30400/60000 [==============>...............] - ETA: 56s - loss: 0.2554 - categorical_accuracy: 0.9215
30432/60000 [==============>...............] - ETA: 56s - loss: 0.2552 - categorical_accuracy: 0.9216
30464/60000 [==============>...............] - ETA: 56s - loss: 0.2549 - categorical_accuracy: 0.9217
30496/60000 [==============>...............] - ETA: 56s - loss: 0.2550 - categorical_accuracy: 0.9217
30528/60000 [==============>...............] - ETA: 56s - loss: 0.2550 - categorical_accuracy: 0.9217
30560/60000 [==============>...............] - ETA: 56s - loss: 0.2551 - categorical_accuracy: 0.9216
30592/60000 [==============>...............] - ETA: 55s - loss: 0.2549 - categorical_accuracy: 0.9216
30624/60000 [==============>...............] - ETA: 55s - loss: 0.2547 - categorical_accuracy: 0.9217
30656/60000 [==============>...............] - ETA: 55s - loss: 0.2548 - categorical_accuracy: 0.9217
30688/60000 [==============>...............] - ETA: 55s - loss: 0.2550 - categorical_accuracy: 0.9217
30720/60000 [==============>...............] - ETA: 55s - loss: 0.2548 - categorical_accuracy: 0.9217
30752/60000 [==============>...............] - ETA: 55s - loss: 0.2547 - categorical_accuracy: 0.9218
30784/60000 [==============>...............] - ETA: 55s - loss: 0.2550 - categorical_accuracy: 0.9217
30816/60000 [==============>...............] - ETA: 55s - loss: 0.2548 - categorical_accuracy: 0.9218
30848/60000 [==============>...............] - ETA: 55s - loss: 0.2546 - categorical_accuracy: 0.9219
30880/60000 [==============>...............] - ETA: 55s - loss: 0.2545 - categorical_accuracy: 0.9219
30912/60000 [==============>...............] - ETA: 55s - loss: 0.2543 - categorical_accuracy: 0.9220
30944/60000 [==============>...............] - ETA: 55s - loss: 0.2541 - categorical_accuracy: 0.9221
30976/60000 [==============>...............] - ETA: 55s - loss: 0.2539 - categorical_accuracy: 0.9221
31008/60000 [==============>...............] - ETA: 55s - loss: 0.2537 - categorical_accuracy: 0.9222
31040/60000 [==============>...............] - ETA: 55s - loss: 0.2535 - categorical_accuracy: 0.9223
31072/60000 [==============>...............] - ETA: 55s - loss: 0.2534 - categorical_accuracy: 0.9223
31104/60000 [==============>...............] - ETA: 54s - loss: 0.2533 - categorical_accuracy: 0.9223
31136/60000 [==============>...............] - ETA: 54s - loss: 0.2531 - categorical_accuracy: 0.9224
31168/60000 [==============>...............] - ETA: 54s - loss: 0.2530 - categorical_accuracy: 0.9225
31200/60000 [==============>...............] - ETA: 54s - loss: 0.2531 - categorical_accuracy: 0.9225
31232/60000 [==============>...............] - ETA: 54s - loss: 0.2529 - categorical_accuracy: 0.9225
31264/60000 [==============>...............] - ETA: 54s - loss: 0.2529 - categorical_accuracy: 0.9225
31296/60000 [==============>...............] - ETA: 54s - loss: 0.2527 - categorical_accuracy: 0.9226
31328/60000 [==============>...............] - ETA: 54s - loss: 0.2525 - categorical_accuracy: 0.9227
31360/60000 [==============>...............] - ETA: 54s - loss: 0.2523 - categorical_accuracy: 0.9227
31392/60000 [==============>...............] - ETA: 54s - loss: 0.2522 - categorical_accuracy: 0.9228
31424/60000 [==============>...............] - ETA: 54s - loss: 0.2520 - categorical_accuracy: 0.9228
31456/60000 [==============>...............] - ETA: 54s - loss: 0.2518 - categorical_accuracy: 0.9229
31488/60000 [==============>...............] - ETA: 54s - loss: 0.2517 - categorical_accuracy: 0.9229
31520/60000 [==============>...............] - ETA: 54s - loss: 0.2517 - categorical_accuracy: 0.9229
31552/60000 [==============>...............] - ETA: 54s - loss: 0.2516 - categorical_accuracy: 0.9230
31584/60000 [==============>...............] - ETA: 54s - loss: 0.2513 - categorical_accuracy: 0.9230
31616/60000 [==============>...............] - ETA: 53s - loss: 0.2511 - categorical_accuracy: 0.9231
31648/60000 [==============>...............] - ETA: 53s - loss: 0.2509 - categorical_accuracy: 0.9232
31680/60000 [==============>...............] - ETA: 53s - loss: 0.2506 - categorical_accuracy: 0.9232
31712/60000 [==============>...............] - ETA: 53s - loss: 0.2504 - categorical_accuracy: 0.9233
31744/60000 [==============>...............] - ETA: 53s - loss: 0.2503 - categorical_accuracy: 0.9234
31776/60000 [==============>...............] - ETA: 53s - loss: 0.2500 - categorical_accuracy: 0.9234
31808/60000 [==============>...............] - ETA: 53s - loss: 0.2501 - categorical_accuracy: 0.9235
31840/60000 [==============>...............] - ETA: 53s - loss: 0.2500 - categorical_accuracy: 0.9235
31872/60000 [==============>...............] - ETA: 53s - loss: 0.2499 - categorical_accuracy: 0.9235
31904/60000 [==============>...............] - ETA: 53s - loss: 0.2496 - categorical_accuracy: 0.9236
31936/60000 [==============>...............] - ETA: 53s - loss: 0.2494 - categorical_accuracy: 0.9236
31968/60000 [==============>...............] - ETA: 53s - loss: 0.2492 - categorical_accuracy: 0.9237
32000/60000 [===============>..............] - ETA: 53s - loss: 0.2491 - categorical_accuracy: 0.9237
32032/60000 [===============>..............] - ETA: 53s - loss: 0.2489 - categorical_accuracy: 0.9238
32064/60000 [===============>..............] - ETA: 53s - loss: 0.2488 - categorical_accuracy: 0.9238
32096/60000 [===============>..............] - ETA: 53s - loss: 0.2487 - categorical_accuracy: 0.9239
32128/60000 [===============>..............] - ETA: 53s - loss: 0.2485 - categorical_accuracy: 0.9240
32160/60000 [===============>..............] - ETA: 52s - loss: 0.2483 - categorical_accuracy: 0.9240
32192/60000 [===============>..............] - ETA: 52s - loss: 0.2481 - categorical_accuracy: 0.9241
32224/60000 [===============>..............] - ETA: 52s - loss: 0.2479 - categorical_accuracy: 0.9241
32256/60000 [===============>..............] - ETA: 52s - loss: 0.2480 - categorical_accuracy: 0.9241
32288/60000 [===============>..............] - ETA: 52s - loss: 0.2478 - categorical_accuracy: 0.9242
32320/60000 [===============>..............] - ETA: 52s - loss: 0.2477 - categorical_accuracy: 0.9242
32352/60000 [===============>..............] - ETA: 52s - loss: 0.2474 - categorical_accuracy: 0.9243
32384/60000 [===============>..............] - ETA: 52s - loss: 0.2472 - categorical_accuracy: 0.9244
32416/60000 [===============>..............] - ETA: 52s - loss: 0.2470 - categorical_accuracy: 0.9245
32448/60000 [===============>..............] - ETA: 52s - loss: 0.2469 - categorical_accuracy: 0.9245
32480/60000 [===============>..............] - ETA: 52s - loss: 0.2470 - categorical_accuracy: 0.9245
32512/60000 [===============>..............] - ETA: 52s - loss: 0.2468 - categorical_accuracy: 0.9245
32544/60000 [===============>..............] - ETA: 52s - loss: 0.2466 - categorical_accuracy: 0.9246
32576/60000 [===============>..............] - ETA: 52s - loss: 0.2464 - categorical_accuracy: 0.9247
32608/60000 [===============>..............] - ETA: 52s - loss: 0.2462 - categorical_accuracy: 0.9247
32640/60000 [===============>..............] - ETA: 52s - loss: 0.2460 - categorical_accuracy: 0.9248
32672/60000 [===============>..............] - ETA: 51s - loss: 0.2459 - categorical_accuracy: 0.9248
32704/60000 [===============>..............] - ETA: 51s - loss: 0.2461 - categorical_accuracy: 0.9248
32736/60000 [===============>..............] - ETA: 51s - loss: 0.2461 - categorical_accuracy: 0.9248
32768/60000 [===============>..............] - ETA: 51s - loss: 0.2460 - categorical_accuracy: 0.9248
32800/60000 [===============>..............] - ETA: 51s - loss: 0.2458 - categorical_accuracy: 0.9249
32832/60000 [===============>..............] - ETA: 51s - loss: 0.2457 - categorical_accuracy: 0.9250
32864/60000 [===============>..............] - ETA: 51s - loss: 0.2455 - categorical_accuracy: 0.9250
32896/60000 [===============>..............] - ETA: 51s - loss: 0.2453 - categorical_accuracy: 0.9251
32928/60000 [===============>..............] - ETA: 51s - loss: 0.2451 - categorical_accuracy: 0.9251
32960/60000 [===============>..............] - ETA: 51s - loss: 0.2449 - categorical_accuracy: 0.9252
32992/60000 [===============>..............] - ETA: 51s - loss: 0.2448 - categorical_accuracy: 0.9252
33024/60000 [===============>..............] - ETA: 51s - loss: 0.2450 - categorical_accuracy: 0.9252
33056/60000 [===============>..............] - ETA: 51s - loss: 0.2448 - categorical_accuracy: 0.9252
33088/60000 [===============>..............] - ETA: 51s - loss: 0.2447 - categorical_accuracy: 0.9253
33120/60000 [===============>..............] - ETA: 51s - loss: 0.2445 - categorical_accuracy: 0.9254
33152/60000 [===============>..............] - ETA: 51s - loss: 0.2444 - categorical_accuracy: 0.9253
33184/60000 [===============>..............] - ETA: 51s - loss: 0.2443 - categorical_accuracy: 0.9254
33216/60000 [===============>..............] - ETA: 50s - loss: 0.2442 - categorical_accuracy: 0.9254
33248/60000 [===============>..............] - ETA: 50s - loss: 0.2441 - categorical_accuracy: 0.9254
33280/60000 [===============>..............] - ETA: 50s - loss: 0.2439 - categorical_accuracy: 0.9255
33312/60000 [===============>..............] - ETA: 50s - loss: 0.2438 - categorical_accuracy: 0.9255
33344/60000 [===============>..............] - ETA: 50s - loss: 0.2437 - categorical_accuracy: 0.9255
33376/60000 [===============>..............] - ETA: 50s - loss: 0.2436 - categorical_accuracy: 0.9255
33408/60000 [===============>..............] - ETA: 50s - loss: 0.2434 - categorical_accuracy: 0.9256
33440/60000 [===============>..............] - ETA: 50s - loss: 0.2432 - categorical_accuracy: 0.9257
33472/60000 [===============>..............] - ETA: 50s - loss: 0.2430 - categorical_accuracy: 0.9257
33504/60000 [===============>..............] - ETA: 50s - loss: 0.2428 - categorical_accuracy: 0.9258
33536/60000 [===============>..............] - ETA: 50s - loss: 0.2427 - categorical_accuracy: 0.9258
33568/60000 [===============>..............] - ETA: 50s - loss: 0.2426 - categorical_accuracy: 0.9259
33600/60000 [===============>..............] - ETA: 50s - loss: 0.2425 - categorical_accuracy: 0.9259
33632/60000 [===============>..............] - ETA: 50s - loss: 0.2424 - categorical_accuracy: 0.9259
33664/60000 [===============>..............] - ETA: 50s - loss: 0.2424 - categorical_accuracy: 0.9259
33696/60000 [===============>..............] - ETA: 50s - loss: 0.2422 - categorical_accuracy: 0.9260
33728/60000 [===============>..............] - ETA: 49s - loss: 0.2420 - categorical_accuracy: 0.9261
33760/60000 [===============>..............] - ETA: 49s - loss: 0.2418 - categorical_accuracy: 0.9262
33792/60000 [===============>..............] - ETA: 49s - loss: 0.2416 - categorical_accuracy: 0.9262
33824/60000 [===============>..............] - ETA: 49s - loss: 0.2414 - categorical_accuracy: 0.9263
33856/60000 [===============>..............] - ETA: 49s - loss: 0.2413 - categorical_accuracy: 0.9263
33888/60000 [===============>..............] - ETA: 49s - loss: 0.2411 - categorical_accuracy: 0.9264
33920/60000 [===============>..............] - ETA: 49s - loss: 0.2410 - categorical_accuracy: 0.9264
33952/60000 [===============>..............] - ETA: 49s - loss: 0.2408 - categorical_accuracy: 0.9265
33984/60000 [===============>..............] - ETA: 49s - loss: 0.2407 - categorical_accuracy: 0.9265
34016/60000 [================>.............] - ETA: 49s - loss: 0.2405 - categorical_accuracy: 0.9266
34048/60000 [================>.............] - ETA: 49s - loss: 0.2404 - categorical_accuracy: 0.9266
34080/60000 [================>.............] - ETA: 49s - loss: 0.2402 - categorical_accuracy: 0.9267
34112/60000 [================>.............] - ETA: 49s - loss: 0.2400 - categorical_accuracy: 0.9267
34144/60000 [================>.............] - ETA: 49s - loss: 0.2399 - categorical_accuracy: 0.9268
34176/60000 [================>.............] - ETA: 49s - loss: 0.2397 - categorical_accuracy: 0.9268
34208/60000 [================>.............] - ETA: 49s - loss: 0.2403 - categorical_accuracy: 0.9268
34240/60000 [================>.............] - ETA: 49s - loss: 0.2401 - categorical_accuracy: 0.9269
34272/60000 [================>.............] - ETA: 48s - loss: 0.2400 - categorical_accuracy: 0.9269
34304/60000 [================>.............] - ETA: 48s - loss: 0.2398 - categorical_accuracy: 0.9270
34336/60000 [================>.............] - ETA: 48s - loss: 0.2399 - categorical_accuracy: 0.9270
34368/60000 [================>.............] - ETA: 48s - loss: 0.2397 - categorical_accuracy: 0.9270
34400/60000 [================>.............] - ETA: 48s - loss: 0.2396 - categorical_accuracy: 0.9271
34432/60000 [================>.............] - ETA: 48s - loss: 0.2396 - categorical_accuracy: 0.9271
34464/60000 [================>.............] - ETA: 48s - loss: 0.2396 - categorical_accuracy: 0.9271
34496/60000 [================>.............] - ETA: 48s - loss: 0.2394 - categorical_accuracy: 0.9271
34528/60000 [================>.............] - ETA: 48s - loss: 0.2392 - categorical_accuracy: 0.9272
34560/60000 [================>.............] - ETA: 48s - loss: 0.2393 - categorical_accuracy: 0.9272
34592/60000 [================>.............] - ETA: 48s - loss: 0.2392 - categorical_accuracy: 0.9272
34624/60000 [================>.............] - ETA: 48s - loss: 0.2390 - categorical_accuracy: 0.9273
34656/60000 [================>.............] - ETA: 48s - loss: 0.2387 - categorical_accuracy: 0.9274
34688/60000 [================>.............] - ETA: 48s - loss: 0.2385 - categorical_accuracy: 0.9274
34720/60000 [================>.............] - ETA: 48s - loss: 0.2385 - categorical_accuracy: 0.9274
34752/60000 [================>.............] - ETA: 48s - loss: 0.2384 - categorical_accuracy: 0.9275
34784/60000 [================>.............] - ETA: 47s - loss: 0.2382 - categorical_accuracy: 0.9275
34816/60000 [================>.............] - ETA: 47s - loss: 0.2381 - categorical_accuracy: 0.9275
34848/60000 [================>.............] - ETA: 47s - loss: 0.2379 - categorical_accuracy: 0.9276
34880/60000 [================>.............] - ETA: 47s - loss: 0.2378 - categorical_accuracy: 0.9276
34912/60000 [================>.............] - ETA: 47s - loss: 0.2379 - categorical_accuracy: 0.9276
34944/60000 [================>.............] - ETA: 47s - loss: 0.2378 - categorical_accuracy: 0.9277
34976/60000 [================>.............] - ETA: 47s - loss: 0.2376 - categorical_accuracy: 0.9278
35008/60000 [================>.............] - ETA: 47s - loss: 0.2375 - categorical_accuracy: 0.9278
35040/60000 [================>.............] - ETA: 47s - loss: 0.2377 - categorical_accuracy: 0.9278
35072/60000 [================>.............] - ETA: 47s - loss: 0.2378 - categorical_accuracy: 0.9278
35104/60000 [================>.............] - ETA: 47s - loss: 0.2379 - categorical_accuracy: 0.9277
35136/60000 [================>.............] - ETA: 47s - loss: 0.2377 - categorical_accuracy: 0.9278
35168/60000 [================>.............] - ETA: 47s - loss: 0.2375 - categorical_accuracy: 0.9279
35200/60000 [================>.............] - ETA: 47s - loss: 0.2375 - categorical_accuracy: 0.9278
35232/60000 [================>.............] - ETA: 47s - loss: 0.2374 - categorical_accuracy: 0.9279
35264/60000 [================>.............] - ETA: 47s - loss: 0.2372 - categorical_accuracy: 0.9279
35296/60000 [================>.............] - ETA: 46s - loss: 0.2370 - categorical_accuracy: 0.9280
35328/60000 [================>.............] - ETA: 46s - loss: 0.2369 - categorical_accuracy: 0.9280
35360/60000 [================>.............] - ETA: 46s - loss: 0.2368 - categorical_accuracy: 0.9281
35392/60000 [================>.............] - ETA: 46s - loss: 0.2366 - categorical_accuracy: 0.9281
35424/60000 [================>.............] - ETA: 46s - loss: 0.2365 - categorical_accuracy: 0.9282
35456/60000 [================>.............] - ETA: 46s - loss: 0.2363 - categorical_accuracy: 0.9282
35488/60000 [================>.............] - ETA: 46s - loss: 0.2362 - categorical_accuracy: 0.9282
35520/60000 [================>.............] - ETA: 46s - loss: 0.2361 - categorical_accuracy: 0.9282
35552/60000 [================>.............] - ETA: 46s - loss: 0.2360 - categorical_accuracy: 0.9282
35584/60000 [================>.............] - ETA: 46s - loss: 0.2359 - categorical_accuracy: 0.9283
35616/60000 [================>.............] - ETA: 46s - loss: 0.2357 - categorical_accuracy: 0.9283
35648/60000 [================>.............] - ETA: 46s - loss: 0.2356 - categorical_accuracy: 0.9284
35680/60000 [================>.............] - ETA: 46s - loss: 0.2355 - categorical_accuracy: 0.9284
35712/60000 [================>.............] - ETA: 46s - loss: 0.2354 - categorical_accuracy: 0.9285
35744/60000 [================>.............] - ETA: 46s - loss: 0.2353 - categorical_accuracy: 0.9285
35776/60000 [================>.............] - ETA: 46s - loss: 0.2352 - categorical_accuracy: 0.9285
35808/60000 [================>.............] - ETA: 46s - loss: 0.2351 - categorical_accuracy: 0.9285
35840/60000 [================>.............] - ETA: 45s - loss: 0.2349 - categorical_accuracy: 0.9286
35872/60000 [================>.............] - ETA: 45s - loss: 0.2347 - categorical_accuracy: 0.9286
35904/60000 [================>.............] - ETA: 45s - loss: 0.2347 - categorical_accuracy: 0.9286
35936/60000 [================>.............] - ETA: 45s - loss: 0.2346 - categorical_accuracy: 0.9287
35968/60000 [================>.............] - ETA: 45s - loss: 0.2344 - categorical_accuracy: 0.9287
36000/60000 [=================>............] - ETA: 45s - loss: 0.2342 - categorical_accuracy: 0.9288
36032/60000 [=================>............] - ETA: 45s - loss: 0.2341 - categorical_accuracy: 0.9288
36064/60000 [=================>............] - ETA: 45s - loss: 0.2341 - categorical_accuracy: 0.9288
36096/60000 [=================>............] - ETA: 45s - loss: 0.2339 - categorical_accuracy: 0.9289
36128/60000 [=================>............] - ETA: 45s - loss: 0.2337 - categorical_accuracy: 0.9289
36160/60000 [=================>............] - ETA: 45s - loss: 0.2336 - categorical_accuracy: 0.9290
36192/60000 [=================>............] - ETA: 45s - loss: 0.2335 - categorical_accuracy: 0.9290
36224/60000 [=================>............] - ETA: 45s - loss: 0.2333 - categorical_accuracy: 0.9290
36256/60000 [=================>............] - ETA: 45s - loss: 0.2331 - categorical_accuracy: 0.9291
36288/60000 [=================>............] - ETA: 45s - loss: 0.2331 - categorical_accuracy: 0.9291
36320/60000 [=================>............] - ETA: 45s - loss: 0.2330 - categorical_accuracy: 0.9291
36352/60000 [=================>............] - ETA: 44s - loss: 0.2328 - categorical_accuracy: 0.9292
36384/60000 [=================>............] - ETA: 44s - loss: 0.2326 - categorical_accuracy: 0.9293
36416/60000 [=================>............] - ETA: 44s - loss: 0.2325 - categorical_accuracy: 0.9293
36448/60000 [=================>............] - ETA: 44s - loss: 0.2327 - categorical_accuracy: 0.9293
36480/60000 [=================>............] - ETA: 44s - loss: 0.2325 - categorical_accuracy: 0.9293
36512/60000 [=================>............] - ETA: 44s - loss: 0.2325 - categorical_accuracy: 0.9292
36544/60000 [=================>............] - ETA: 44s - loss: 0.2323 - categorical_accuracy: 0.9293
36576/60000 [=================>............] - ETA: 44s - loss: 0.2322 - categorical_accuracy: 0.9293
36608/60000 [=================>............] - ETA: 44s - loss: 0.2321 - categorical_accuracy: 0.9294
36640/60000 [=================>............] - ETA: 44s - loss: 0.2320 - categorical_accuracy: 0.9294
36672/60000 [=================>............] - ETA: 44s - loss: 0.2325 - categorical_accuracy: 0.9294
36704/60000 [=================>............] - ETA: 44s - loss: 0.2324 - categorical_accuracy: 0.9294
36736/60000 [=================>............] - ETA: 44s - loss: 0.2325 - categorical_accuracy: 0.9293
36768/60000 [=================>............] - ETA: 44s - loss: 0.2324 - categorical_accuracy: 0.9293
36800/60000 [=================>............] - ETA: 44s - loss: 0.2323 - categorical_accuracy: 0.9293
36832/60000 [=================>............] - ETA: 44s - loss: 0.2322 - categorical_accuracy: 0.9294
36864/60000 [=================>............] - ETA: 43s - loss: 0.2320 - categorical_accuracy: 0.9294
36896/60000 [=================>............] - ETA: 43s - loss: 0.2318 - categorical_accuracy: 0.9295
36928/60000 [=================>............] - ETA: 43s - loss: 0.2317 - categorical_accuracy: 0.9295
36960/60000 [=================>............] - ETA: 43s - loss: 0.2316 - categorical_accuracy: 0.9295
36992/60000 [=================>............] - ETA: 43s - loss: 0.2314 - categorical_accuracy: 0.9296
37024/60000 [=================>............] - ETA: 43s - loss: 0.2312 - categorical_accuracy: 0.9297
37056/60000 [=================>............] - ETA: 43s - loss: 0.2312 - categorical_accuracy: 0.9297
37088/60000 [=================>............] - ETA: 43s - loss: 0.2312 - categorical_accuracy: 0.9297
37120/60000 [=================>............] - ETA: 43s - loss: 0.2314 - categorical_accuracy: 0.9297
37152/60000 [=================>............] - ETA: 43s - loss: 0.2313 - categorical_accuracy: 0.9297
37184/60000 [=================>............] - ETA: 43s - loss: 0.2311 - categorical_accuracy: 0.9298
37216/60000 [=================>............] - ETA: 43s - loss: 0.2309 - categorical_accuracy: 0.9299
37248/60000 [=================>............] - ETA: 43s - loss: 0.2308 - categorical_accuracy: 0.9299
37280/60000 [=================>............] - ETA: 43s - loss: 0.2306 - categorical_accuracy: 0.9300
37312/60000 [=================>............] - ETA: 43s - loss: 0.2306 - categorical_accuracy: 0.9300
37344/60000 [=================>............] - ETA: 43s - loss: 0.2307 - categorical_accuracy: 0.9300
37376/60000 [=================>............] - ETA: 43s - loss: 0.2306 - categorical_accuracy: 0.9300
37408/60000 [=================>............] - ETA: 42s - loss: 0.2305 - categorical_accuracy: 0.9301
37440/60000 [=================>............] - ETA: 42s - loss: 0.2303 - categorical_accuracy: 0.9301
37472/60000 [=================>............] - ETA: 42s - loss: 0.2302 - categorical_accuracy: 0.9302
37504/60000 [=================>............] - ETA: 42s - loss: 0.2301 - categorical_accuracy: 0.9302
37536/60000 [=================>............] - ETA: 42s - loss: 0.2301 - categorical_accuracy: 0.9302
37568/60000 [=================>............] - ETA: 42s - loss: 0.2302 - categorical_accuracy: 0.9302
37600/60000 [=================>............] - ETA: 42s - loss: 0.2301 - categorical_accuracy: 0.9302
37632/60000 [=================>............] - ETA: 42s - loss: 0.2301 - categorical_accuracy: 0.9302
37664/60000 [=================>............] - ETA: 42s - loss: 0.2300 - categorical_accuracy: 0.9303
37696/60000 [=================>............] - ETA: 42s - loss: 0.2301 - categorical_accuracy: 0.9303
37728/60000 [=================>............] - ETA: 42s - loss: 0.2299 - categorical_accuracy: 0.9303
37760/60000 [=================>............] - ETA: 42s - loss: 0.2298 - categorical_accuracy: 0.9303
37792/60000 [=================>............] - ETA: 42s - loss: 0.2299 - categorical_accuracy: 0.9303
37824/60000 [=================>............] - ETA: 42s - loss: 0.2299 - categorical_accuracy: 0.9303
37856/60000 [=================>............] - ETA: 42s - loss: 0.2298 - categorical_accuracy: 0.9303
37888/60000 [=================>............] - ETA: 42s - loss: 0.2296 - categorical_accuracy: 0.9304
37920/60000 [=================>............] - ETA: 41s - loss: 0.2295 - categorical_accuracy: 0.9305
37952/60000 [=================>............] - ETA: 41s - loss: 0.2293 - categorical_accuracy: 0.9305
37984/60000 [=================>............] - ETA: 41s - loss: 0.2292 - categorical_accuracy: 0.9306
38016/60000 [==================>...........] - ETA: 41s - loss: 0.2292 - categorical_accuracy: 0.9306
38048/60000 [==================>...........] - ETA: 41s - loss: 0.2292 - categorical_accuracy: 0.9306
38080/60000 [==================>...........] - ETA: 41s - loss: 0.2292 - categorical_accuracy: 0.9306
38112/60000 [==================>...........] - ETA: 41s - loss: 0.2291 - categorical_accuracy: 0.9306
38144/60000 [==================>...........] - ETA: 41s - loss: 0.2290 - categorical_accuracy: 0.9307
38176/60000 [==================>...........] - ETA: 41s - loss: 0.2289 - categorical_accuracy: 0.9307
38208/60000 [==================>...........] - ETA: 41s - loss: 0.2289 - categorical_accuracy: 0.9307
38240/60000 [==================>...........] - ETA: 41s - loss: 0.2288 - categorical_accuracy: 0.9308
38272/60000 [==================>...........] - ETA: 41s - loss: 0.2286 - categorical_accuracy: 0.9308
38304/60000 [==================>...........] - ETA: 41s - loss: 0.2284 - categorical_accuracy: 0.9309
38336/60000 [==================>...........] - ETA: 41s - loss: 0.2285 - categorical_accuracy: 0.9308
38368/60000 [==================>...........] - ETA: 41s - loss: 0.2285 - categorical_accuracy: 0.9309
38400/60000 [==================>...........] - ETA: 41s - loss: 0.2285 - categorical_accuracy: 0.9308
38432/60000 [==================>...........] - ETA: 41s - loss: 0.2283 - categorical_accuracy: 0.9309
38464/60000 [==================>...........] - ETA: 40s - loss: 0.2282 - categorical_accuracy: 0.9309
38496/60000 [==================>...........] - ETA: 40s - loss: 0.2280 - categorical_accuracy: 0.9310
38528/60000 [==================>...........] - ETA: 40s - loss: 0.2280 - categorical_accuracy: 0.9310
38560/60000 [==================>...........] - ETA: 40s - loss: 0.2279 - categorical_accuracy: 0.9311
38592/60000 [==================>...........] - ETA: 40s - loss: 0.2278 - categorical_accuracy: 0.9311
38624/60000 [==================>...........] - ETA: 40s - loss: 0.2277 - categorical_accuracy: 0.9312
38656/60000 [==================>...........] - ETA: 40s - loss: 0.2275 - categorical_accuracy: 0.9312
38688/60000 [==================>...........] - ETA: 40s - loss: 0.2274 - categorical_accuracy: 0.9312
38720/60000 [==================>...........] - ETA: 40s - loss: 0.2272 - categorical_accuracy: 0.9313
38752/60000 [==================>...........] - ETA: 40s - loss: 0.2271 - categorical_accuracy: 0.9313
38784/60000 [==================>...........] - ETA: 40s - loss: 0.2270 - categorical_accuracy: 0.9314
38816/60000 [==================>...........] - ETA: 40s - loss: 0.2269 - categorical_accuracy: 0.9314
38848/60000 [==================>...........] - ETA: 40s - loss: 0.2267 - categorical_accuracy: 0.9315
38880/60000 [==================>...........] - ETA: 40s - loss: 0.2266 - categorical_accuracy: 0.9315
38912/60000 [==================>...........] - ETA: 40s - loss: 0.2264 - categorical_accuracy: 0.9315
38944/60000 [==================>...........] - ETA: 40s - loss: 0.2263 - categorical_accuracy: 0.9316
38976/60000 [==================>...........] - ETA: 39s - loss: 0.2261 - categorical_accuracy: 0.9317
39008/60000 [==================>...........] - ETA: 39s - loss: 0.2261 - categorical_accuracy: 0.9317
39040/60000 [==================>...........] - ETA: 39s - loss: 0.2259 - categorical_accuracy: 0.9317
39072/60000 [==================>...........] - ETA: 39s - loss: 0.2257 - categorical_accuracy: 0.9318
39104/60000 [==================>...........] - ETA: 39s - loss: 0.2256 - categorical_accuracy: 0.9318
39136/60000 [==================>...........] - ETA: 39s - loss: 0.2255 - categorical_accuracy: 0.9318
39168/60000 [==================>...........] - ETA: 39s - loss: 0.2254 - categorical_accuracy: 0.9318
39200/60000 [==================>...........] - ETA: 39s - loss: 0.2253 - categorical_accuracy: 0.9319
39232/60000 [==================>...........] - ETA: 39s - loss: 0.2253 - categorical_accuracy: 0.9319
39264/60000 [==================>...........] - ETA: 39s - loss: 0.2252 - categorical_accuracy: 0.9319
39296/60000 [==================>...........] - ETA: 39s - loss: 0.2252 - categorical_accuracy: 0.9319
39328/60000 [==================>...........] - ETA: 39s - loss: 0.2250 - categorical_accuracy: 0.9320
39360/60000 [==================>...........] - ETA: 39s - loss: 0.2249 - categorical_accuracy: 0.9320
39392/60000 [==================>...........] - ETA: 39s - loss: 0.2248 - categorical_accuracy: 0.9320
39424/60000 [==================>...........] - ETA: 39s - loss: 0.2247 - categorical_accuracy: 0.9320
39456/60000 [==================>...........] - ETA: 39s - loss: 0.2246 - categorical_accuracy: 0.9321
39488/60000 [==================>...........] - ETA: 38s - loss: 0.2244 - categorical_accuracy: 0.9322
39520/60000 [==================>...........] - ETA: 38s - loss: 0.2243 - categorical_accuracy: 0.9322
39552/60000 [==================>...........] - ETA: 38s - loss: 0.2242 - categorical_accuracy: 0.9322
39584/60000 [==================>...........] - ETA: 38s - loss: 0.2241 - categorical_accuracy: 0.9322
39616/60000 [==================>...........] - ETA: 38s - loss: 0.2241 - categorical_accuracy: 0.9322
39648/60000 [==================>...........] - ETA: 38s - loss: 0.2239 - categorical_accuracy: 0.9323
39680/60000 [==================>...........] - ETA: 38s - loss: 0.2238 - categorical_accuracy: 0.9323
39712/60000 [==================>...........] - ETA: 38s - loss: 0.2236 - categorical_accuracy: 0.9324
39744/60000 [==================>...........] - ETA: 38s - loss: 0.2235 - categorical_accuracy: 0.9324
39776/60000 [==================>...........] - ETA: 38s - loss: 0.2234 - categorical_accuracy: 0.9324
39808/60000 [==================>...........] - ETA: 38s - loss: 0.2232 - categorical_accuracy: 0.9325
39840/60000 [==================>...........] - ETA: 38s - loss: 0.2231 - categorical_accuracy: 0.9325
39872/60000 [==================>...........] - ETA: 38s - loss: 0.2229 - categorical_accuracy: 0.9326
39904/60000 [==================>...........] - ETA: 38s - loss: 0.2229 - categorical_accuracy: 0.9326
39936/60000 [==================>...........] - ETA: 38s - loss: 0.2228 - categorical_accuracy: 0.9326
39968/60000 [==================>...........] - ETA: 38s - loss: 0.2226 - categorical_accuracy: 0.9327
40000/60000 [===================>..........] - ETA: 38s - loss: 0.2226 - categorical_accuracy: 0.9327
40032/60000 [===================>..........] - ETA: 37s - loss: 0.2224 - categorical_accuracy: 0.9328
40064/60000 [===================>..........] - ETA: 37s - loss: 0.2223 - categorical_accuracy: 0.9328
40096/60000 [===================>..........] - ETA: 37s - loss: 0.2223 - categorical_accuracy: 0.9328
40128/60000 [===================>..........] - ETA: 37s - loss: 0.2223 - categorical_accuracy: 0.9328
40160/60000 [===================>..........] - ETA: 37s - loss: 0.2222 - categorical_accuracy: 0.9328
40192/60000 [===================>..........] - ETA: 37s - loss: 0.2223 - categorical_accuracy: 0.9328
40224/60000 [===================>..........] - ETA: 37s - loss: 0.2221 - categorical_accuracy: 0.9329
40256/60000 [===================>..........] - ETA: 37s - loss: 0.2219 - categorical_accuracy: 0.9329
40288/60000 [===================>..........] - ETA: 37s - loss: 0.2219 - categorical_accuracy: 0.9330
40320/60000 [===================>..........] - ETA: 37s - loss: 0.2218 - categorical_accuracy: 0.9330
40352/60000 [===================>..........] - ETA: 37s - loss: 0.2216 - categorical_accuracy: 0.9331
40384/60000 [===================>..........] - ETA: 37s - loss: 0.2214 - categorical_accuracy: 0.9331
40416/60000 [===================>..........] - ETA: 37s - loss: 0.2213 - categorical_accuracy: 0.9331
40448/60000 [===================>..........] - ETA: 37s - loss: 0.2212 - categorical_accuracy: 0.9332
40480/60000 [===================>..........] - ETA: 37s - loss: 0.2210 - categorical_accuracy: 0.9332
40512/60000 [===================>..........] - ETA: 37s - loss: 0.2209 - categorical_accuracy: 0.9333
40544/60000 [===================>..........] - ETA: 36s - loss: 0.2208 - categorical_accuracy: 0.9333
40576/60000 [===================>..........] - ETA: 36s - loss: 0.2207 - categorical_accuracy: 0.9333
40608/60000 [===================>..........] - ETA: 36s - loss: 0.2206 - categorical_accuracy: 0.9333
40640/60000 [===================>..........] - ETA: 36s - loss: 0.2206 - categorical_accuracy: 0.9333
40672/60000 [===================>..........] - ETA: 36s - loss: 0.2204 - categorical_accuracy: 0.9334
40704/60000 [===================>..........] - ETA: 36s - loss: 0.2203 - categorical_accuracy: 0.9334
40736/60000 [===================>..........] - ETA: 36s - loss: 0.2202 - categorical_accuracy: 0.9335
40768/60000 [===================>..........] - ETA: 36s - loss: 0.2201 - categorical_accuracy: 0.9335
40800/60000 [===================>..........] - ETA: 36s - loss: 0.2201 - categorical_accuracy: 0.9335
40832/60000 [===================>..........] - ETA: 36s - loss: 0.2199 - categorical_accuracy: 0.9335
40864/60000 [===================>..........] - ETA: 36s - loss: 0.2199 - categorical_accuracy: 0.9335
40896/60000 [===================>..........] - ETA: 36s - loss: 0.2198 - categorical_accuracy: 0.9336
40928/60000 [===================>..........] - ETA: 36s - loss: 0.2197 - categorical_accuracy: 0.9336
40960/60000 [===================>..........] - ETA: 36s - loss: 0.2195 - categorical_accuracy: 0.9336
40992/60000 [===================>..........] - ETA: 36s - loss: 0.2195 - categorical_accuracy: 0.9336
41024/60000 [===================>..........] - ETA: 36s - loss: 0.2194 - categorical_accuracy: 0.9337
41056/60000 [===================>..........] - ETA: 35s - loss: 0.2192 - categorical_accuracy: 0.9337
41088/60000 [===================>..........] - ETA: 35s - loss: 0.2192 - categorical_accuracy: 0.9338
41120/60000 [===================>..........] - ETA: 35s - loss: 0.2191 - categorical_accuracy: 0.9338
41152/60000 [===================>..........] - ETA: 35s - loss: 0.2190 - categorical_accuracy: 0.9338
41184/60000 [===================>..........] - ETA: 35s - loss: 0.2189 - categorical_accuracy: 0.9339
41216/60000 [===================>..........] - ETA: 35s - loss: 0.2187 - categorical_accuracy: 0.9339
41248/60000 [===================>..........] - ETA: 35s - loss: 0.2187 - categorical_accuracy: 0.9339
41280/60000 [===================>..........] - ETA: 35s - loss: 0.2186 - categorical_accuracy: 0.9339
41312/60000 [===================>..........] - ETA: 35s - loss: 0.2186 - categorical_accuracy: 0.9339
41344/60000 [===================>..........] - ETA: 35s - loss: 0.2185 - categorical_accuracy: 0.9340
41376/60000 [===================>..........] - ETA: 35s - loss: 0.2185 - categorical_accuracy: 0.9340
41408/60000 [===================>..........] - ETA: 35s - loss: 0.2184 - categorical_accuracy: 0.9340
41440/60000 [===================>..........] - ETA: 35s - loss: 0.2183 - categorical_accuracy: 0.9340
41472/60000 [===================>..........] - ETA: 35s - loss: 0.2182 - categorical_accuracy: 0.9341
41504/60000 [===================>..........] - ETA: 35s - loss: 0.2181 - categorical_accuracy: 0.9341
41536/60000 [===================>..........] - ETA: 35s - loss: 0.2181 - categorical_accuracy: 0.9342
41568/60000 [===================>..........] - ETA: 35s - loss: 0.2180 - categorical_accuracy: 0.9342
41600/60000 [===================>..........] - ETA: 34s - loss: 0.2178 - categorical_accuracy: 0.9343
41632/60000 [===================>..........] - ETA: 34s - loss: 0.2177 - categorical_accuracy: 0.9343
41664/60000 [===================>..........] - ETA: 34s - loss: 0.2175 - categorical_accuracy: 0.9343
41696/60000 [===================>..........] - ETA: 34s - loss: 0.2174 - categorical_accuracy: 0.9344
41728/60000 [===================>..........] - ETA: 34s - loss: 0.2173 - categorical_accuracy: 0.9344
41760/60000 [===================>..........] - ETA: 34s - loss: 0.2171 - categorical_accuracy: 0.9344
41792/60000 [===================>..........] - ETA: 34s - loss: 0.2170 - categorical_accuracy: 0.9345
41824/60000 [===================>..........] - ETA: 34s - loss: 0.2169 - categorical_accuracy: 0.9345
41856/60000 [===================>..........] - ETA: 34s - loss: 0.2167 - categorical_accuracy: 0.9346
41888/60000 [===================>..........] - ETA: 34s - loss: 0.2167 - categorical_accuracy: 0.9346
41920/60000 [===================>..........] - ETA: 34s - loss: 0.2166 - categorical_accuracy: 0.9346
41952/60000 [===================>..........] - ETA: 34s - loss: 0.2165 - categorical_accuracy: 0.9346
41984/60000 [===================>..........] - ETA: 34s - loss: 0.2164 - categorical_accuracy: 0.9347
42016/60000 [====================>.........] - ETA: 34s - loss: 0.2163 - categorical_accuracy: 0.9347
42048/60000 [====================>.........] - ETA: 34s - loss: 0.2163 - categorical_accuracy: 0.9347
42080/60000 [====================>.........] - ETA: 34s - loss: 0.2163 - categorical_accuracy: 0.9347
42112/60000 [====================>.........] - ETA: 33s - loss: 0.2162 - categorical_accuracy: 0.9348
42144/60000 [====================>.........] - ETA: 33s - loss: 0.2160 - categorical_accuracy: 0.9348
42176/60000 [====================>.........] - ETA: 33s - loss: 0.2159 - categorical_accuracy: 0.9349
42208/60000 [====================>.........] - ETA: 33s - loss: 0.2158 - categorical_accuracy: 0.9349
42240/60000 [====================>.........] - ETA: 33s - loss: 0.2157 - categorical_accuracy: 0.9349
42272/60000 [====================>.........] - ETA: 33s - loss: 0.2155 - categorical_accuracy: 0.9350
42304/60000 [====================>.........] - ETA: 33s - loss: 0.2154 - categorical_accuracy: 0.9350
42336/60000 [====================>.........] - ETA: 33s - loss: 0.2153 - categorical_accuracy: 0.9350
42368/60000 [====================>.........] - ETA: 33s - loss: 0.2152 - categorical_accuracy: 0.9351
42400/60000 [====================>.........] - ETA: 33s - loss: 0.2151 - categorical_accuracy: 0.9351
42432/60000 [====================>.........] - ETA: 33s - loss: 0.2150 - categorical_accuracy: 0.9351
42464/60000 [====================>.........] - ETA: 33s - loss: 0.2148 - categorical_accuracy: 0.9352
42496/60000 [====================>.........] - ETA: 33s - loss: 0.2148 - categorical_accuracy: 0.9352
42528/60000 [====================>.........] - ETA: 33s - loss: 0.2148 - categorical_accuracy: 0.9352
42560/60000 [====================>.........] - ETA: 33s - loss: 0.2147 - categorical_accuracy: 0.9352
42592/60000 [====================>.........] - ETA: 33s - loss: 0.2147 - categorical_accuracy: 0.9352
42624/60000 [====================>.........] - ETA: 33s - loss: 0.2147 - categorical_accuracy: 0.9352
42656/60000 [====================>.........] - ETA: 32s - loss: 0.2146 - categorical_accuracy: 0.9352
42688/60000 [====================>.........] - ETA: 32s - loss: 0.2145 - categorical_accuracy: 0.9353
42720/60000 [====================>.........] - ETA: 32s - loss: 0.2146 - categorical_accuracy: 0.9353
42752/60000 [====================>.........] - ETA: 32s - loss: 0.2145 - categorical_accuracy: 0.9353
42784/60000 [====================>.........] - ETA: 32s - loss: 0.2143 - categorical_accuracy: 0.9353
42816/60000 [====================>.........] - ETA: 32s - loss: 0.2142 - categorical_accuracy: 0.9354
42848/60000 [====================>.........] - ETA: 32s - loss: 0.2141 - categorical_accuracy: 0.9354
42880/60000 [====================>.........] - ETA: 32s - loss: 0.2140 - categorical_accuracy: 0.9354
42912/60000 [====================>.........] - ETA: 32s - loss: 0.2139 - categorical_accuracy: 0.9354
42944/60000 [====================>.........] - ETA: 32s - loss: 0.2139 - categorical_accuracy: 0.9355
42976/60000 [====================>.........] - ETA: 32s - loss: 0.2138 - categorical_accuracy: 0.9355
43008/60000 [====================>.........] - ETA: 32s - loss: 0.2140 - categorical_accuracy: 0.9355
43040/60000 [====================>.........] - ETA: 32s - loss: 0.2139 - categorical_accuracy: 0.9355
43072/60000 [====================>.........] - ETA: 32s - loss: 0.2138 - categorical_accuracy: 0.9355
43104/60000 [====================>.........] - ETA: 32s - loss: 0.2138 - categorical_accuracy: 0.9355
43136/60000 [====================>.........] - ETA: 32s - loss: 0.2137 - categorical_accuracy: 0.9355
43168/60000 [====================>.........] - ETA: 31s - loss: 0.2137 - categorical_accuracy: 0.9355
43200/60000 [====================>.........] - ETA: 31s - loss: 0.2135 - categorical_accuracy: 0.9356
43232/60000 [====================>.........] - ETA: 31s - loss: 0.2134 - categorical_accuracy: 0.9356
43264/60000 [====================>.........] - ETA: 31s - loss: 0.2133 - categorical_accuracy: 0.9356
43296/60000 [====================>.........] - ETA: 31s - loss: 0.2132 - categorical_accuracy: 0.9357
43328/60000 [====================>.........] - ETA: 31s - loss: 0.2132 - categorical_accuracy: 0.9357
43360/60000 [====================>.........] - ETA: 31s - loss: 0.2131 - categorical_accuracy: 0.9357
43392/60000 [====================>.........] - ETA: 31s - loss: 0.2131 - categorical_accuracy: 0.9357
43424/60000 [====================>.........] - ETA: 31s - loss: 0.2131 - categorical_accuracy: 0.9357
43456/60000 [====================>.........] - ETA: 31s - loss: 0.2130 - categorical_accuracy: 0.9357
43488/60000 [====================>.........] - ETA: 31s - loss: 0.2130 - categorical_accuracy: 0.9357
43520/60000 [====================>.........] - ETA: 31s - loss: 0.2130 - categorical_accuracy: 0.9358
43552/60000 [====================>.........] - ETA: 31s - loss: 0.2130 - categorical_accuracy: 0.9358
43584/60000 [====================>.........] - ETA: 31s - loss: 0.2128 - categorical_accuracy: 0.9358
43616/60000 [====================>.........] - ETA: 31s - loss: 0.2127 - categorical_accuracy: 0.9359
43648/60000 [====================>.........] - ETA: 31s - loss: 0.2126 - categorical_accuracy: 0.9359
43680/60000 [====================>.........] - ETA: 31s - loss: 0.2126 - categorical_accuracy: 0.9359
43712/60000 [====================>.........] - ETA: 30s - loss: 0.2125 - categorical_accuracy: 0.9359
43744/60000 [====================>.........] - ETA: 30s - loss: 0.2124 - categorical_accuracy: 0.9359
43776/60000 [====================>.........] - ETA: 30s - loss: 0.2123 - categorical_accuracy: 0.9359
43808/60000 [====================>.........] - ETA: 30s - loss: 0.2123 - categorical_accuracy: 0.9359
43840/60000 [====================>.........] - ETA: 30s - loss: 0.2122 - categorical_accuracy: 0.9359
43872/60000 [====================>.........] - ETA: 30s - loss: 0.2120 - categorical_accuracy: 0.9360
43904/60000 [====================>.........] - ETA: 30s - loss: 0.2120 - categorical_accuracy: 0.9360
43936/60000 [====================>.........] - ETA: 30s - loss: 0.2119 - categorical_accuracy: 0.9360
43968/60000 [====================>.........] - ETA: 30s - loss: 0.2118 - categorical_accuracy: 0.9360
44000/60000 [=====================>........] - ETA: 30s - loss: 0.2116 - categorical_accuracy: 0.9361
44032/60000 [=====================>........] - ETA: 30s - loss: 0.2115 - categorical_accuracy: 0.9361
44064/60000 [=====================>........] - ETA: 30s - loss: 0.2114 - categorical_accuracy: 0.9362
44096/60000 [=====================>........] - ETA: 30s - loss: 0.2113 - categorical_accuracy: 0.9362
44128/60000 [=====================>........] - ETA: 30s - loss: 0.2114 - categorical_accuracy: 0.9362
44160/60000 [=====================>........] - ETA: 30s - loss: 0.2114 - categorical_accuracy: 0.9362
44192/60000 [=====================>........] - ETA: 30s - loss: 0.2113 - categorical_accuracy: 0.9362
44224/60000 [=====================>........] - ETA: 29s - loss: 0.2112 - categorical_accuracy: 0.9363
44256/60000 [=====================>........] - ETA: 29s - loss: 0.2110 - categorical_accuracy: 0.9363
44288/60000 [=====================>........] - ETA: 29s - loss: 0.2110 - categorical_accuracy: 0.9363
44320/60000 [=====================>........] - ETA: 29s - loss: 0.2109 - categorical_accuracy: 0.9363
44352/60000 [=====================>........] - ETA: 29s - loss: 0.2108 - categorical_accuracy: 0.9364
44384/60000 [=====================>........] - ETA: 29s - loss: 0.2107 - categorical_accuracy: 0.9364
44416/60000 [=====================>........] - ETA: 29s - loss: 0.2106 - categorical_accuracy: 0.9364
44448/60000 [=====================>........] - ETA: 29s - loss: 0.2105 - categorical_accuracy: 0.9364
44480/60000 [=====================>........] - ETA: 29s - loss: 0.2103 - categorical_accuracy: 0.9365
44512/60000 [=====================>........] - ETA: 29s - loss: 0.2103 - categorical_accuracy: 0.9365
44544/60000 [=====================>........] - ETA: 29s - loss: 0.2102 - categorical_accuracy: 0.9365
44576/60000 [=====================>........] - ETA: 29s - loss: 0.2101 - categorical_accuracy: 0.9365
44608/60000 [=====================>........] - ETA: 29s - loss: 0.2099 - categorical_accuracy: 0.9366
44640/60000 [=====================>........] - ETA: 29s - loss: 0.2099 - categorical_accuracy: 0.9366
44672/60000 [=====================>........] - ETA: 29s - loss: 0.2100 - categorical_accuracy: 0.9366
44704/60000 [=====================>........] - ETA: 29s - loss: 0.2099 - categorical_accuracy: 0.9366
44736/60000 [=====================>........] - ETA: 29s - loss: 0.2097 - categorical_accuracy: 0.9366
44768/60000 [=====================>........] - ETA: 28s - loss: 0.2096 - categorical_accuracy: 0.9367
44800/60000 [=====================>........] - ETA: 28s - loss: 0.2095 - categorical_accuracy: 0.9367
44832/60000 [=====================>........] - ETA: 28s - loss: 0.2094 - categorical_accuracy: 0.9367
44864/60000 [=====================>........] - ETA: 28s - loss: 0.2093 - categorical_accuracy: 0.9367
44896/60000 [=====================>........] - ETA: 28s - loss: 0.2093 - categorical_accuracy: 0.9367
44928/60000 [=====================>........] - ETA: 28s - loss: 0.2092 - categorical_accuracy: 0.9368
44960/60000 [=====================>........] - ETA: 28s - loss: 0.2092 - categorical_accuracy: 0.9368
44992/60000 [=====================>........] - ETA: 28s - loss: 0.2093 - categorical_accuracy: 0.9368
45024/60000 [=====================>........] - ETA: 28s - loss: 0.2094 - categorical_accuracy: 0.9368
45056/60000 [=====================>........] - ETA: 28s - loss: 0.2094 - categorical_accuracy: 0.9368
45088/60000 [=====================>........] - ETA: 28s - loss: 0.2093 - categorical_accuracy: 0.9368
45120/60000 [=====================>........] - ETA: 28s - loss: 0.2092 - categorical_accuracy: 0.9369
45152/60000 [=====================>........] - ETA: 28s - loss: 0.2090 - categorical_accuracy: 0.9369
45184/60000 [=====================>........] - ETA: 28s - loss: 0.2089 - categorical_accuracy: 0.9369
45216/60000 [=====================>........] - ETA: 28s - loss: 0.2088 - categorical_accuracy: 0.9370
45248/60000 [=====================>........] - ETA: 28s - loss: 0.2087 - categorical_accuracy: 0.9370
45280/60000 [=====================>........] - ETA: 27s - loss: 0.2086 - categorical_accuracy: 0.9371
45312/60000 [=====================>........] - ETA: 27s - loss: 0.2085 - categorical_accuracy: 0.9371
45344/60000 [=====================>........] - ETA: 27s - loss: 0.2084 - categorical_accuracy: 0.9371
45376/60000 [=====================>........] - ETA: 27s - loss: 0.2083 - categorical_accuracy: 0.9371
45408/60000 [=====================>........] - ETA: 27s - loss: 0.2082 - categorical_accuracy: 0.9372
45440/60000 [=====================>........] - ETA: 27s - loss: 0.2082 - categorical_accuracy: 0.9372
45472/60000 [=====================>........] - ETA: 27s - loss: 0.2081 - categorical_accuracy: 0.9372
45504/60000 [=====================>........] - ETA: 27s - loss: 0.2080 - categorical_accuracy: 0.9372
45536/60000 [=====================>........] - ETA: 27s - loss: 0.2080 - categorical_accuracy: 0.9372
45568/60000 [=====================>........] - ETA: 27s - loss: 0.2079 - categorical_accuracy: 0.9373
45600/60000 [=====================>........] - ETA: 27s - loss: 0.2078 - categorical_accuracy: 0.9373
45632/60000 [=====================>........] - ETA: 27s - loss: 0.2077 - categorical_accuracy: 0.9373
45664/60000 [=====================>........] - ETA: 27s - loss: 0.2077 - categorical_accuracy: 0.9373
45696/60000 [=====================>........] - ETA: 27s - loss: 0.2076 - categorical_accuracy: 0.9373
45728/60000 [=====================>........] - ETA: 27s - loss: 0.2074 - categorical_accuracy: 0.9374
45760/60000 [=====================>........] - ETA: 27s - loss: 0.2073 - categorical_accuracy: 0.9374
45792/60000 [=====================>........] - ETA: 26s - loss: 0.2073 - categorical_accuracy: 0.9375
45824/60000 [=====================>........] - ETA: 26s - loss: 0.2072 - categorical_accuracy: 0.9375
45856/60000 [=====================>........] - ETA: 26s - loss: 0.2071 - categorical_accuracy: 0.9375
45888/60000 [=====================>........] - ETA: 26s - loss: 0.2070 - categorical_accuracy: 0.9375
45920/60000 [=====================>........] - ETA: 26s - loss: 0.2069 - categorical_accuracy: 0.9376
45952/60000 [=====================>........] - ETA: 26s - loss: 0.2068 - categorical_accuracy: 0.9376
45984/60000 [=====================>........] - ETA: 26s - loss: 0.2067 - categorical_accuracy: 0.9377
46016/60000 [======================>.......] - ETA: 26s - loss: 0.2067 - categorical_accuracy: 0.9377
46048/60000 [======================>.......] - ETA: 26s - loss: 0.2066 - categorical_accuracy: 0.9377
46080/60000 [======================>.......] - ETA: 26s - loss: 0.2065 - categorical_accuracy: 0.9377
46112/60000 [======================>.......] - ETA: 26s - loss: 0.2065 - categorical_accuracy: 0.9377
46144/60000 [======================>.......] - ETA: 26s - loss: 0.2065 - categorical_accuracy: 0.9377
46176/60000 [======================>.......] - ETA: 26s - loss: 0.2063 - categorical_accuracy: 0.9377
46208/60000 [======================>.......] - ETA: 26s - loss: 0.2064 - categorical_accuracy: 0.9377
46240/60000 [======================>.......] - ETA: 26s - loss: 0.2065 - categorical_accuracy: 0.9377
46272/60000 [======================>.......] - ETA: 26s - loss: 0.2065 - categorical_accuracy: 0.9377
46304/60000 [======================>.......] - ETA: 26s - loss: 0.2063 - categorical_accuracy: 0.9377
46336/60000 [======================>.......] - ETA: 25s - loss: 0.2062 - categorical_accuracy: 0.9378
46368/60000 [======================>.......] - ETA: 25s - loss: 0.2061 - categorical_accuracy: 0.9378
46400/60000 [======================>.......] - ETA: 25s - loss: 0.2060 - categorical_accuracy: 0.9378
46432/60000 [======================>.......] - ETA: 25s - loss: 0.2060 - categorical_accuracy: 0.9378
46464/60000 [======================>.......] - ETA: 25s - loss: 0.2059 - categorical_accuracy: 0.9379
46496/60000 [======================>.......] - ETA: 25s - loss: 0.2060 - categorical_accuracy: 0.9378
46528/60000 [======================>.......] - ETA: 25s - loss: 0.2058 - categorical_accuracy: 0.9379
46560/60000 [======================>.......] - ETA: 25s - loss: 0.2057 - categorical_accuracy: 0.9379
46592/60000 [======================>.......] - ETA: 25s - loss: 0.2056 - categorical_accuracy: 0.9380
46624/60000 [======================>.......] - ETA: 25s - loss: 0.2056 - categorical_accuracy: 0.9380
46656/60000 [======================>.......] - ETA: 25s - loss: 0.2055 - categorical_accuracy: 0.9380
46688/60000 [======================>.......] - ETA: 25s - loss: 0.2054 - categorical_accuracy: 0.9380
46720/60000 [======================>.......] - ETA: 25s - loss: 0.2053 - categorical_accuracy: 0.9380
46752/60000 [======================>.......] - ETA: 25s - loss: 0.2051 - categorical_accuracy: 0.9381
46784/60000 [======================>.......] - ETA: 25s - loss: 0.2051 - categorical_accuracy: 0.9381
46816/60000 [======================>.......] - ETA: 25s - loss: 0.2050 - categorical_accuracy: 0.9381
46848/60000 [======================>.......] - ETA: 24s - loss: 0.2048 - categorical_accuracy: 0.9382
46880/60000 [======================>.......] - ETA: 24s - loss: 0.2048 - categorical_accuracy: 0.9382
46912/60000 [======================>.......] - ETA: 24s - loss: 0.2047 - categorical_accuracy: 0.9382
46944/60000 [======================>.......] - ETA: 24s - loss: 0.2047 - categorical_accuracy: 0.9382
46976/60000 [======================>.......] - ETA: 24s - loss: 0.2046 - categorical_accuracy: 0.9382
47008/60000 [======================>.......] - ETA: 24s - loss: 0.2045 - categorical_accuracy: 0.9383
47040/60000 [======================>.......] - ETA: 24s - loss: 0.2044 - categorical_accuracy: 0.9383
47072/60000 [======================>.......] - ETA: 24s - loss: 0.2043 - categorical_accuracy: 0.9383
47104/60000 [======================>.......] - ETA: 24s - loss: 0.2042 - categorical_accuracy: 0.9384
47136/60000 [======================>.......] - ETA: 24s - loss: 0.2041 - categorical_accuracy: 0.9384
47168/60000 [======================>.......] - ETA: 24s - loss: 0.2040 - categorical_accuracy: 0.9384
47200/60000 [======================>.......] - ETA: 24s - loss: 0.2040 - categorical_accuracy: 0.9385
47232/60000 [======================>.......] - ETA: 24s - loss: 0.2039 - categorical_accuracy: 0.9385
47264/60000 [======================>.......] - ETA: 24s - loss: 0.2038 - categorical_accuracy: 0.9385
47296/60000 [======================>.......] - ETA: 24s - loss: 0.2037 - categorical_accuracy: 0.9386
47328/60000 [======================>.......] - ETA: 24s - loss: 0.2036 - categorical_accuracy: 0.9386
47360/60000 [======================>.......] - ETA: 24s - loss: 0.2037 - categorical_accuracy: 0.9386
47392/60000 [======================>.......] - ETA: 23s - loss: 0.2038 - categorical_accuracy: 0.9386
47424/60000 [======================>.......] - ETA: 23s - loss: 0.2037 - categorical_accuracy: 0.9386
47456/60000 [======================>.......] - ETA: 23s - loss: 0.2037 - categorical_accuracy: 0.9386
47488/60000 [======================>.......] - ETA: 23s - loss: 0.2036 - categorical_accuracy: 0.9387
47520/60000 [======================>.......] - ETA: 23s - loss: 0.2034 - categorical_accuracy: 0.9387
47552/60000 [======================>.......] - ETA: 23s - loss: 0.2034 - categorical_accuracy: 0.9387
47584/60000 [======================>.......] - ETA: 23s - loss: 0.2033 - categorical_accuracy: 0.9387
47616/60000 [======================>.......] - ETA: 23s - loss: 0.2032 - categorical_accuracy: 0.9388
47648/60000 [======================>.......] - ETA: 23s - loss: 0.2033 - categorical_accuracy: 0.9387
47680/60000 [======================>.......] - ETA: 23s - loss: 0.2033 - categorical_accuracy: 0.9388
47712/60000 [======================>.......] - ETA: 23s - loss: 0.2032 - categorical_accuracy: 0.9388
47744/60000 [======================>.......] - ETA: 23s - loss: 0.2032 - categorical_accuracy: 0.9388
47776/60000 [======================>.......] - ETA: 23s - loss: 0.2031 - categorical_accuracy: 0.9388
47808/60000 [======================>.......] - ETA: 23s - loss: 0.2030 - categorical_accuracy: 0.9389
47840/60000 [======================>.......] - ETA: 23s - loss: 0.2029 - categorical_accuracy: 0.9389
47872/60000 [======================>.......] - ETA: 23s - loss: 0.2028 - categorical_accuracy: 0.9389
47904/60000 [======================>.......] - ETA: 22s - loss: 0.2027 - categorical_accuracy: 0.9389
47936/60000 [======================>.......] - ETA: 22s - loss: 0.2026 - categorical_accuracy: 0.9390
47968/60000 [======================>.......] - ETA: 22s - loss: 0.2026 - categorical_accuracy: 0.9390
48000/60000 [=======================>......] - ETA: 22s - loss: 0.2025 - categorical_accuracy: 0.9390
48032/60000 [=======================>......] - ETA: 22s - loss: 0.2024 - categorical_accuracy: 0.9390
48064/60000 [=======================>......] - ETA: 22s - loss: 0.2023 - categorical_accuracy: 0.9391
48096/60000 [=======================>......] - ETA: 22s - loss: 0.2022 - categorical_accuracy: 0.9391
48128/60000 [=======================>......] - ETA: 22s - loss: 0.2021 - categorical_accuracy: 0.9391
48160/60000 [=======================>......] - ETA: 22s - loss: 0.2020 - categorical_accuracy: 0.9391
48192/60000 [=======================>......] - ETA: 22s - loss: 0.2019 - categorical_accuracy: 0.9392
48224/60000 [=======================>......] - ETA: 22s - loss: 0.2018 - categorical_accuracy: 0.9392
48256/60000 [=======================>......] - ETA: 22s - loss: 0.2018 - categorical_accuracy: 0.9392
48288/60000 [=======================>......] - ETA: 22s - loss: 0.2016 - categorical_accuracy: 0.9392
48320/60000 [=======================>......] - ETA: 22s - loss: 0.2016 - categorical_accuracy: 0.9392
48352/60000 [=======================>......] - ETA: 22s - loss: 0.2015 - categorical_accuracy: 0.9392
48384/60000 [=======================>......] - ETA: 22s - loss: 0.2014 - categorical_accuracy: 0.9393
48416/60000 [=======================>......] - ETA: 22s - loss: 0.2013 - categorical_accuracy: 0.9393
48448/60000 [=======================>......] - ETA: 21s - loss: 0.2013 - categorical_accuracy: 0.9393
48480/60000 [=======================>......] - ETA: 21s - loss: 0.2013 - categorical_accuracy: 0.9393
48512/60000 [=======================>......] - ETA: 21s - loss: 0.2012 - categorical_accuracy: 0.9393
48544/60000 [=======================>......] - ETA: 21s - loss: 0.2013 - categorical_accuracy: 0.9393
48576/60000 [=======================>......] - ETA: 21s - loss: 0.2012 - categorical_accuracy: 0.9394
48608/60000 [=======================>......] - ETA: 21s - loss: 0.2010 - categorical_accuracy: 0.9394
48640/60000 [=======================>......] - ETA: 21s - loss: 0.2010 - categorical_accuracy: 0.9394
48672/60000 [=======================>......] - ETA: 21s - loss: 0.2009 - categorical_accuracy: 0.9395
48704/60000 [=======================>......] - ETA: 21s - loss: 0.2008 - categorical_accuracy: 0.9395
48736/60000 [=======================>......] - ETA: 21s - loss: 0.2007 - categorical_accuracy: 0.9395
48768/60000 [=======================>......] - ETA: 21s - loss: 0.2007 - categorical_accuracy: 0.9395
48800/60000 [=======================>......] - ETA: 21s - loss: 0.2009 - categorical_accuracy: 0.9395
48832/60000 [=======================>......] - ETA: 21s - loss: 0.2008 - categorical_accuracy: 0.9395
48864/60000 [=======================>......] - ETA: 21s - loss: 0.2007 - categorical_accuracy: 0.9395
48896/60000 [=======================>......] - ETA: 21s - loss: 0.2006 - categorical_accuracy: 0.9396
48928/60000 [=======================>......] - ETA: 21s - loss: 0.2005 - categorical_accuracy: 0.9396
48960/60000 [=======================>......] - ETA: 20s - loss: 0.2005 - categorical_accuracy: 0.9396
48992/60000 [=======================>......] - ETA: 20s - loss: 0.2004 - categorical_accuracy: 0.9396
49024/60000 [=======================>......] - ETA: 20s - loss: 0.2003 - categorical_accuracy: 0.9396
49056/60000 [=======================>......] - ETA: 20s - loss: 0.2003 - categorical_accuracy: 0.9397
49088/60000 [=======================>......] - ETA: 20s - loss: 0.2002 - categorical_accuracy: 0.9397
49120/60000 [=======================>......] - ETA: 20s - loss: 0.2001 - categorical_accuracy: 0.9397
49152/60000 [=======================>......] - ETA: 20s - loss: 0.2001 - categorical_accuracy: 0.9397
49184/60000 [=======================>......] - ETA: 20s - loss: 0.2000 - categorical_accuracy: 0.9397
49216/60000 [=======================>......] - ETA: 20s - loss: 0.1999 - categorical_accuracy: 0.9397
49248/60000 [=======================>......] - ETA: 20s - loss: 0.1998 - categorical_accuracy: 0.9397
49280/60000 [=======================>......] - ETA: 20s - loss: 0.1997 - categorical_accuracy: 0.9398
49312/60000 [=======================>......] - ETA: 20s - loss: 0.1997 - categorical_accuracy: 0.9398
49344/60000 [=======================>......] - ETA: 20s - loss: 0.1997 - categorical_accuracy: 0.9398
49376/60000 [=======================>......] - ETA: 20s - loss: 0.1995 - categorical_accuracy: 0.9398
49408/60000 [=======================>......] - ETA: 20s - loss: 0.1995 - categorical_accuracy: 0.9398
49440/60000 [=======================>......] - ETA: 20s - loss: 0.1994 - categorical_accuracy: 0.9399
49472/60000 [=======================>......] - ETA: 19s - loss: 0.1993 - categorical_accuracy: 0.9399
49504/60000 [=======================>......] - ETA: 19s - loss: 0.1993 - categorical_accuracy: 0.9399
49536/60000 [=======================>......] - ETA: 19s - loss: 0.1992 - categorical_accuracy: 0.9399
49568/60000 [=======================>......] - ETA: 19s - loss: 0.1991 - categorical_accuracy: 0.9400
49600/60000 [=======================>......] - ETA: 19s - loss: 0.1990 - categorical_accuracy: 0.9400
49632/60000 [=======================>......] - ETA: 19s - loss: 0.1989 - categorical_accuracy: 0.9400
49664/60000 [=======================>......] - ETA: 19s - loss: 0.1989 - categorical_accuracy: 0.9401
49696/60000 [=======================>......] - ETA: 19s - loss: 0.1988 - categorical_accuracy: 0.9401
49728/60000 [=======================>......] - ETA: 19s - loss: 0.1986 - categorical_accuracy: 0.9401
49760/60000 [=======================>......] - ETA: 19s - loss: 0.1985 - categorical_accuracy: 0.9402
49792/60000 [=======================>......] - ETA: 19s - loss: 0.1985 - categorical_accuracy: 0.9402
49824/60000 [=======================>......] - ETA: 19s - loss: 0.1985 - categorical_accuracy: 0.9402
49856/60000 [=======================>......] - ETA: 19s - loss: 0.1984 - categorical_accuracy: 0.9402
49888/60000 [=======================>......] - ETA: 19s - loss: 0.1984 - categorical_accuracy: 0.9402
49920/60000 [=======================>......] - ETA: 19s - loss: 0.1983 - categorical_accuracy: 0.9403
49952/60000 [=======================>......] - ETA: 19s - loss: 0.1982 - categorical_accuracy: 0.9403
49984/60000 [=======================>......] - ETA: 19s - loss: 0.1981 - categorical_accuracy: 0.9403
50016/60000 [========================>.....] - ETA: 18s - loss: 0.1982 - categorical_accuracy: 0.9403
50048/60000 [========================>.....] - ETA: 18s - loss: 0.1981 - categorical_accuracy: 0.9404
50080/60000 [========================>.....] - ETA: 18s - loss: 0.1980 - categorical_accuracy: 0.9404
50112/60000 [========================>.....] - ETA: 18s - loss: 0.1979 - categorical_accuracy: 0.9404
50144/60000 [========================>.....] - ETA: 18s - loss: 0.1978 - categorical_accuracy: 0.9404
50176/60000 [========================>.....] - ETA: 18s - loss: 0.1977 - categorical_accuracy: 0.9404
50208/60000 [========================>.....] - ETA: 18s - loss: 0.1976 - categorical_accuracy: 0.9405
50240/60000 [========================>.....] - ETA: 18s - loss: 0.1975 - categorical_accuracy: 0.9405
50272/60000 [========================>.....] - ETA: 18s - loss: 0.1976 - categorical_accuracy: 0.9405
50304/60000 [========================>.....] - ETA: 18s - loss: 0.1976 - categorical_accuracy: 0.9405
50336/60000 [========================>.....] - ETA: 18s - loss: 0.1975 - categorical_accuracy: 0.9405
50368/60000 [========================>.....] - ETA: 18s - loss: 0.1975 - categorical_accuracy: 0.9405
50400/60000 [========================>.....] - ETA: 18s - loss: 0.1975 - categorical_accuracy: 0.9405
50432/60000 [========================>.....] - ETA: 18s - loss: 0.1974 - categorical_accuracy: 0.9406
50464/60000 [========================>.....] - ETA: 18s - loss: 0.1973 - categorical_accuracy: 0.9406
50496/60000 [========================>.....] - ETA: 18s - loss: 0.1971 - categorical_accuracy: 0.9406
50528/60000 [========================>.....] - ETA: 17s - loss: 0.1971 - categorical_accuracy: 0.9406
50560/60000 [========================>.....] - ETA: 17s - loss: 0.1970 - categorical_accuracy: 0.9406
50592/60000 [========================>.....] - ETA: 17s - loss: 0.1970 - categorical_accuracy: 0.9406
50624/60000 [========================>.....] - ETA: 17s - loss: 0.1969 - categorical_accuracy: 0.9406
50656/60000 [========================>.....] - ETA: 17s - loss: 0.1970 - categorical_accuracy: 0.9406
50688/60000 [========================>.....] - ETA: 17s - loss: 0.1969 - categorical_accuracy: 0.9407
50720/60000 [========================>.....] - ETA: 17s - loss: 0.1968 - categorical_accuracy: 0.9407
50752/60000 [========================>.....] - ETA: 17s - loss: 0.1968 - categorical_accuracy: 0.9407
50784/60000 [========================>.....] - ETA: 17s - loss: 0.1967 - categorical_accuracy: 0.9407
50816/60000 [========================>.....] - ETA: 17s - loss: 0.1966 - categorical_accuracy: 0.9408
50848/60000 [========================>.....] - ETA: 17s - loss: 0.1967 - categorical_accuracy: 0.9408
50880/60000 [========================>.....] - ETA: 17s - loss: 0.1965 - categorical_accuracy: 0.9408
50912/60000 [========================>.....] - ETA: 17s - loss: 0.1964 - categorical_accuracy: 0.9409
50944/60000 [========================>.....] - ETA: 17s - loss: 0.1963 - categorical_accuracy: 0.9409
50976/60000 [========================>.....] - ETA: 17s - loss: 0.1963 - categorical_accuracy: 0.9409
51008/60000 [========================>.....] - ETA: 17s - loss: 0.1962 - categorical_accuracy: 0.9410
51040/60000 [========================>.....] - ETA: 17s - loss: 0.1961 - categorical_accuracy: 0.9410
51072/60000 [========================>.....] - ETA: 16s - loss: 0.1959 - categorical_accuracy: 0.9410
51104/60000 [========================>.....] - ETA: 16s - loss: 0.1959 - categorical_accuracy: 0.9410
51136/60000 [========================>.....] - ETA: 16s - loss: 0.1960 - categorical_accuracy: 0.9410
51168/60000 [========================>.....] - ETA: 16s - loss: 0.1958 - categorical_accuracy: 0.9411
51200/60000 [========================>.....] - ETA: 16s - loss: 0.1958 - categorical_accuracy: 0.9411
51232/60000 [========================>.....] - ETA: 16s - loss: 0.1957 - categorical_accuracy: 0.9411
51264/60000 [========================>.....] - ETA: 16s - loss: 0.1956 - categorical_accuracy: 0.9411
51296/60000 [========================>.....] - ETA: 16s - loss: 0.1956 - categorical_accuracy: 0.9411
51328/60000 [========================>.....] - ETA: 16s - loss: 0.1957 - categorical_accuracy: 0.9411
51360/60000 [========================>.....] - ETA: 16s - loss: 0.1955 - categorical_accuracy: 0.9411
51392/60000 [========================>.....] - ETA: 16s - loss: 0.1955 - categorical_accuracy: 0.9411
51424/60000 [========================>.....] - ETA: 16s - loss: 0.1954 - categorical_accuracy: 0.9411
51456/60000 [========================>.....] - ETA: 16s - loss: 0.1953 - categorical_accuracy: 0.9412
51488/60000 [========================>.....] - ETA: 16s - loss: 0.1952 - categorical_accuracy: 0.9412
51520/60000 [========================>.....] - ETA: 16s - loss: 0.1952 - categorical_accuracy: 0.9412
51552/60000 [========================>.....] - ETA: 16s - loss: 0.1952 - categorical_accuracy: 0.9412
51584/60000 [========================>.....] - ETA: 15s - loss: 0.1951 - categorical_accuracy: 0.9412
51616/60000 [========================>.....] - ETA: 15s - loss: 0.1950 - categorical_accuracy: 0.9412
51648/60000 [========================>.....] - ETA: 15s - loss: 0.1952 - categorical_accuracy: 0.9412
51680/60000 [========================>.....] - ETA: 15s - loss: 0.1951 - categorical_accuracy: 0.9412
51712/60000 [========================>.....] - ETA: 15s - loss: 0.1951 - categorical_accuracy: 0.9412
51744/60000 [========================>.....] - ETA: 15s - loss: 0.1950 - categorical_accuracy: 0.9413
51776/60000 [========================>.....] - ETA: 15s - loss: 0.1949 - categorical_accuracy: 0.9413
51808/60000 [========================>.....] - ETA: 15s - loss: 0.1948 - categorical_accuracy: 0.9413
51840/60000 [========================>.....] - ETA: 15s - loss: 0.1947 - categorical_accuracy: 0.9413
51872/60000 [========================>.....] - ETA: 15s - loss: 0.1946 - categorical_accuracy: 0.9414
51904/60000 [========================>.....] - ETA: 15s - loss: 0.1946 - categorical_accuracy: 0.9414
51936/60000 [========================>.....] - ETA: 15s - loss: 0.1945 - categorical_accuracy: 0.9414
51968/60000 [========================>.....] - ETA: 15s - loss: 0.1944 - categorical_accuracy: 0.9414
52000/60000 [=========================>....] - ETA: 15s - loss: 0.1943 - categorical_accuracy: 0.9415
52032/60000 [=========================>....] - ETA: 15s - loss: 0.1943 - categorical_accuracy: 0.9415
52064/60000 [=========================>....] - ETA: 15s - loss: 0.1942 - categorical_accuracy: 0.9415
52096/60000 [=========================>....] - ETA: 15s - loss: 0.1943 - categorical_accuracy: 0.9415
52128/60000 [=========================>....] - ETA: 14s - loss: 0.1942 - categorical_accuracy: 0.9415
52160/60000 [=========================>....] - ETA: 14s - loss: 0.1943 - categorical_accuracy: 0.9415
52192/60000 [=========================>....] - ETA: 14s - loss: 0.1942 - categorical_accuracy: 0.9415
52224/60000 [=========================>....] - ETA: 14s - loss: 0.1942 - categorical_accuracy: 0.9415
52256/60000 [=========================>....] - ETA: 14s - loss: 0.1941 - categorical_accuracy: 0.9415
52288/60000 [=========================>....] - ETA: 14s - loss: 0.1940 - categorical_accuracy: 0.9416
52320/60000 [=========================>....] - ETA: 14s - loss: 0.1939 - categorical_accuracy: 0.9416
52352/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9416
52384/60000 [=========================>....] - ETA: 14s - loss: 0.1938 - categorical_accuracy: 0.9416
52416/60000 [=========================>....] - ETA: 14s - loss: 0.1937 - categorical_accuracy: 0.9417
52448/60000 [=========================>....] - ETA: 14s - loss: 0.1936 - categorical_accuracy: 0.9417
52480/60000 [=========================>....] - ETA: 14s - loss: 0.1935 - categorical_accuracy: 0.9417
52512/60000 [=========================>....] - ETA: 14s - loss: 0.1935 - categorical_accuracy: 0.9417
52544/60000 [=========================>....] - ETA: 14s - loss: 0.1934 - categorical_accuracy: 0.9417
52576/60000 [=========================>....] - ETA: 14s - loss: 0.1933 - categorical_accuracy: 0.9418
52608/60000 [=========================>....] - ETA: 14s - loss: 0.1932 - categorical_accuracy: 0.9418
52640/60000 [=========================>....] - ETA: 13s - loss: 0.1932 - categorical_accuracy: 0.9418
52672/60000 [=========================>....] - ETA: 13s - loss: 0.1931 - categorical_accuracy: 0.9418
52704/60000 [=========================>....] - ETA: 13s - loss: 0.1930 - categorical_accuracy: 0.9418
52736/60000 [=========================>....] - ETA: 13s - loss: 0.1930 - categorical_accuracy: 0.9418
52768/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9419
52800/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9419
52832/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9419
52864/60000 [=========================>....] - ETA: 13s - loss: 0.1927 - categorical_accuracy: 0.9419
52896/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9419
52928/60000 [=========================>....] - ETA: 13s - loss: 0.1929 - categorical_accuracy: 0.9419
52960/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9419
52992/60000 [=========================>....] - ETA: 13s - loss: 0.1928 - categorical_accuracy: 0.9419
53024/60000 [=========================>....] - ETA: 13s - loss: 0.1927 - categorical_accuracy: 0.9419
53056/60000 [=========================>....] - ETA: 13s - loss: 0.1926 - categorical_accuracy: 0.9419
53088/60000 [=========================>....] - ETA: 13s - loss: 0.1925 - categorical_accuracy: 0.9420
53120/60000 [=========================>....] - ETA: 13s - loss: 0.1924 - categorical_accuracy: 0.9420
53152/60000 [=========================>....] - ETA: 13s - loss: 0.1924 - categorical_accuracy: 0.9420
53184/60000 [=========================>....] - ETA: 12s - loss: 0.1923 - categorical_accuracy: 0.9420
53216/60000 [=========================>....] - ETA: 12s - loss: 0.1922 - categorical_accuracy: 0.9420
53248/60000 [=========================>....] - ETA: 12s - loss: 0.1921 - categorical_accuracy: 0.9421
53280/60000 [=========================>....] - ETA: 12s - loss: 0.1920 - categorical_accuracy: 0.9421
53312/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9421
53344/60000 [=========================>....] - ETA: 12s - loss: 0.1919 - categorical_accuracy: 0.9421
53376/60000 [=========================>....] - ETA: 12s - loss: 0.1918 - categorical_accuracy: 0.9422
53408/60000 [=========================>....] - ETA: 12s - loss: 0.1918 - categorical_accuracy: 0.9422
53440/60000 [=========================>....] - ETA: 12s - loss: 0.1918 - categorical_accuracy: 0.9422
53472/60000 [=========================>....] - ETA: 12s - loss: 0.1918 - categorical_accuracy: 0.9422
53504/60000 [=========================>....] - ETA: 12s - loss: 0.1917 - categorical_accuracy: 0.9422
53536/60000 [=========================>....] - ETA: 12s - loss: 0.1917 - categorical_accuracy: 0.9422
53568/60000 [=========================>....] - ETA: 12s - loss: 0.1917 - categorical_accuracy: 0.9422
53600/60000 [=========================>....] - ETA: 12s - loss: 0.1916 - categorical_accuracy: 0.9422
53632/60000 [=========================>....] - ETA: 12s - loss: 0.1916 - categorical_accuracy: 0.9422
53664/60000 [=========================>....] - ETA: 12s - loss: 0.1917 - categorical_accuracy: 0.9422
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1916 - categorical_accuracy: 0.9423
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1916 - categorical_accuracy: 0.9423
53760/60000 [=========================>....] - ETA: 11s - loss: 0.1915 - categorical_accuracy: 0.9423
53792/60000 [=========================>....] - ETA: 11s - loss: 0.1914 - categorical_accuracy: 0.9423
53824/60000 [=========================>....] - ETA: 11s - loss: 0.1914 - categorical_accuracy: 0.9423
53856/60000 [=========================>....] - ETA: 11s - loss: 0.1913 - categorical_accuracy: 0.9423
53888/60000 [=========================>....] - ETA: 11s - loss: 0.1913 - categorical_accuracy: 0.9423
53920/60000 [=========================>....] - ETA: 11s - loss: 0.1912 - categorical_accuracy: 0.9423
53952/60000 [=========================>....] - ETA: 11s - loss: 0.1912 - categorical_accuracy: 0.9424
53984/60000 [=========================>....] - ETA: 11s - loss: 0.1911 - categorical_accuracy: 0.9424
54016/60000 [==========================>...] - ETA: 11s - loss: 0.1909 - categorical_accuracy: 0.9424
54048/60000 [==========================>...] - ETA: 11s - loss: 0.1909 - categorical_accuracy: 0.9424
54080/60000 [==========================>...] - ETA: 11s - loss: 0.1909 - categorical_accuracy: 0.9424
54112/60000 [==========================>...] - ETA: 11s - loss: 0.1909 - categorical_accuracy: 0.9425
54144/60000 [==========================>...] - ETA: 11s - loss: 0.1909 - categorical_accuracy: 0.9424
54176/60000 [==========================>...] - ETA: 11s - loss: 0.1908 - categorical_accuracy: 0.9424
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9425
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9425
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1908 - categorical_accuracy: 0.9425
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1908 - categorical_accuracy: 0.9425
54336/60000 [==========================>...] - ETA: 10s - loss: 0.1907 - categorical_accuracy: 0.9425
54368/60000 [==========================>...] - ETA: 10s - loss: 0.1906 - categorical_accuracy: 0.9426
54400/60000 [==========================>...] - ETA: 10s - loss: 0.1905 - categorical_accuracy: 0.9426
54432/60000 [==========================>...] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9426
54464/60000 [==========================>...] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9426
54496/60000 [==========================>...] - ETA: 10s - loss: 0.1904 - categorical_accuracy: 0.9427
54528/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9427
54560/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9427
54592/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9427
54624/60000 [==========================>...] - ETA: 10s - loss: 0.1902 - categorical_accuracy: 0.9427
54656/60000 [==========================>...] - ETA: 10s - loss: 0.1901 - categorical_accuracy: 0.9428
54688/60000 [==========================>...] - ETA: 10s - loss: 0.1903 - categorical_accuracy: 0.9427
54720/60000 [==========================>...] - ETA: 10s - loss: 0.1902 - categorical_accuracy: 0.9428
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1902 - categorical_accuracy: 0.9428 
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1902 - categorical_accuracy: 0.9428
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1901 - categorical_accuracy: 0.9428
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1900 - categorical_accuracy: 0.9429
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9429
54912/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9429
54944/60000 [==========================>...] - ETA: 9s - loss: 0.1898 - categorical_accuracy: 0.9429
54976/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9429
55008/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9429
55040/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9429
55072/60000 [==========================>...] - ETA: 9s - loss: 0.1898 - categorical_accuracy: 0.9429
55104/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9430
55136/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9430
55168/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9430
55200/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9430
55232/60000 [==========================>...] - ETA: 9s - loss: 0.1896 - categorical_accuracy: 0.9430
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1896 - categorical_accuracy: 0.9430
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1896 - categorical_accuracy: 0.9430
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1895 - categorical_accuracy: 0.9430
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1894 - categorical_accuracy: 0.9430
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1894 - categorical_accuracy: 0.9430
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1893 - categorical_accuracy: 0.9430
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1892 - categorical_accuracy: 0.9431
55488/60000 [==========================>...] - ETA: 8s - loss: 0.1891 - categorical_accuracy: 0.9431
55520/60000 [==========================>...] - ETA: 8s - loss: 0.1890 - categorical_accuracy: 0.9431
55552/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9432
55584/60000 [==========================>...] - ETA: 8s - loss: 0.1888 - categorical_accuracy: 0.9432
55616/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9432
55648/60000 [==========================>...] - ETA: 8s - loss: 0.1888 - categorical_accuracy: 0.9432
55680/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9432
55712/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9432
55744/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9432
55776/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9433
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1885 - categorical_accuracy: 0.9433
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1884 - categorical_accuracy: 0.9433
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1883 - categorical_accuracy: 0.9433
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1882 - categorical_accuracy: 0.9434
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1883 - categorical_accuracy: 0.9434
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1882 - categorical_accuracy: 0.9434
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9434
56032/60000 [===========================>..] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9434
56064/60000 [===========================>..] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9435
56096/60000 [===========================>..] - ETA: 7s - loss: 0.1879 - categorical_accuracy: 0.9435
56128/60000 [===========================>..] - ETA: 7s - loss: 0.1878 - categorical_accuracy: 0.9435
56160/60000 [===========================>..] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9435
56192/60000 [===========================>..] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9436
56224/60000 [===========================>..] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9435
56256/60000 [===========================>..] - ETA: 7s - loss: 0.1875 - categorical_accuracy: 0.9436
56288/60000 [===========================>..] - ETA: 7s - loss: 0.1874 - categorical_accuracy: 0.9436
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1873 - categorical_accuracy: 0.9436
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1872 - categorical_accuracy: 0.9437
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1872 - categorical_accuracy: 0.9437
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9437
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9437
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1872 - categorical_accuracy: 0.9437
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9437
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9437
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9437
56608/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9438
56640/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9438
56672/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9438
56704/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9438
56736/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9438
56768/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9438
56800/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9439
56832/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9439
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1866 - categorical_accuracy: 0.9439
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1865 - categorical_accuracy: 0.9439
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9439
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9440
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9440
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1861 - categorical_accuracy: 0.9440
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1861 - categorical_accuracy: 0.9440
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9441
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9441
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9441
57184/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9441
57216/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9442
57248/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9442
57280/60000 [===========================>..] - ETA: 5s - loss: 0.1858 - categorical_accuracy: 0.9442
57312/60000 [===========================>..] - ETA: 5s - loss: 0.1857 - categorical_accuracy: 0.9442
57344/60000 [===========================>..] - ETA: 5s - loss: 0.1856 - categorical_accuracy: 0.9442
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1856 - categorical_accuracy: 0.9442
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1855 - categorical_accuracy: 0.9442
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9443
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9443
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9443
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9443
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1851 - categorical_accuracy: 0.9444
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9444
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9444
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1850 - categorical_accuracy: 0.9444
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1849 - categorical_accuracy: 0.9445
57728/60000 [===========================>..] - ETA: 4s - loss: 0.1849 - categorical_accuracy: 0.9445
57760/60000 [===========================>..] - ETA: 4s - loss: 0.1848 - categorical_accuracy: 0.9445
57792/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9445
57824/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9446
57856/60000 [===========================>..] - ETA: 4s - loss: 0.1847 - categorical_accuracy: 0.9446
57888/60000 [===========================>..] - ETA: 4s - loss: 0.1846 - categorical_accuracy: 0.9446
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9446
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9446
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9446
58016/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9446
58048/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9446
58080/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9446
58112/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9447
58144/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9447
58176/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9447
58208/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9447
58240/60000 [============================>.] - ETA: 3s - loss: 0.1843 - categorical_accuracy: 0.9447
58272/60000 [============================>.] - ETA: 3s - loss: 0.1842 - categorical_accuracy: 0.9447
58304/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9448
58336/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9448
58368/60000 [============================>.] - ETA: 3s - loss: 0.1841 - categorical_accuracy: 0.9448
58400/60000 [============================>.] - ETA: 3s - loss: 0.1840 - categorical_accuracy: 0.9448
58432/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9448
58464/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9449
58496/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9449
58528/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9449
58560/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9449
58592/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9450
58624/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9450
58656/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9450
58688/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9450
58720/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9450
58752/60000 [============================>.] - ETA: 2s - loss: 0.1832 - categorical_accuracy: 0.9451
58784/60000 [============================>.] - ETA: 2s - loss: 0.1831 - categorical_accuracy: 0.9451
58816/60000 [============================>.] - ETA: 2s - loss: 0.1830 - categorical_accuracy: 0.9451
58848/60000 [============================>.] - ETA: 2s - loss: 0.1829 - categorical_accuracy: 0.9452
58880/60000 [============================>.] - ETA: 2s - loss: 0.1829 - categorical_accuracy: 0.9452
58912/60000 [============================>.] - ETA: 2s - loss: 0.1828 - categorical_accuracy: 0.9452
58944/60000 [============================>.] - ETA: 2s - loss: 0.1827 - categorical_accuracy: 0.9452
58976/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9453
59008/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9453
59040/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9453
59072/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9453
59104/60000 [============================>.] - ETA: 1s - loss: 0.1824 - categorical_accuracy: 0.9453
59136/60000 [============================>.] - ETA: 1s - loss: 0.1823 - categorical_accuracy: 0.9453
59168/60000 [============================>.] - ETA: 1s - loss: 0.1822 - categorical_accuracy: 0.9453
59200/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9454
59232/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9454
59264/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9454
59296/60000 [============================>.] - ETA: 1s - loss: 0.1820 - categorical_accuracy: 0.9454
59328/60000 [============================>.] - ETA: 1s - loss: 0.1820 - categorical_accuracy: 0.9454
59360/60000 [============================>.] - ETA: 1s - loss: 0.1819 - categorical_accuracy: 0.9455
59392/60000 [============================>.] - ETA: 1s - loss: 0.1819 - categorical_accuracy: 0.9455
59424/60000 [============================>.] - ETA: 1s - loss: 0.1820 - categorical_accuracy: 0.9454
59456/60000 [============================>.] - ETA: 1s - loss: 0.1819 - categorical_accuracy: 0.9454
59488/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9455
59520/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9455
59552/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9455
59584/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9455
59616/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9455
59648/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9455
59680/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9455
59712/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9455
59744/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9456
59776/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9456
59808/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9456
59840/60000 [============================>.] - ETA: 0s - loss: 0.1815 - categorical_accuracy: 0.9456
59872/60000 [============================>.] - ETA: 0s - loss: 0.1814 - categorical_accuracy: 0.9456
59904/60000 [============================>.] - ETA: 0s - loss: 0.1815 - categorical_accuracy: 0.9456
59936/60000 [============================>.] - ETA: 0s - loss: 0.1814 - categorical_accuracy: 0.9456
59968/60000 [============================>.] - ETA: 0s - loss: 0.1813 - categorical_accuracy: 0.9457
60000/60000 [==============================] - 118s 2ms/step - loss: 0.1813 - categorical_accuracy: 0.9457 - val_loss: 0.0470 - val_categorical_accuracy: 0.9841

  ('#### Predict   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 

   32/10000 [..............................] - ETA: 17s
  192/10000 [..............................] - ETA: 5s 
  352/10000 [>.............................] - ETA: 4s
  512/10000 [>.............................] - ETA: 4s
  640/10000 [>.............................] - ETA: 4s
  768/10000 [=>............................] - ETA: 4s
  928/10000 [=>............................] - ETA: 3s
 1088/10000 [==>...........................] - ETA: 3s
 1248/10000 [==>...........................] - ETA: 3s
 1408/10000 [===>..........................] - ETA: 3s
 1568/10000 [===>..........................] - ETA: 3s
 1728/10000 [====>.........................] - ETA: 3s
 1888/10000 [====>.........................] - ETA: 3s
 2048/10000 [=====>........................] - ETA: 3s
 2208/10000 [=====>........................] - ETA: 3s
 2336/10000 [======>.......................] - ETA: 3s
 2496/10000 [======>.......................] - ETA: 2s
 2656/10000 [======>.......................] - ETA: 2s
 2784/10000 [=======>......................] - ETA: 2s
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
 4512/10000 [============>.................] - ETA: 2s
 4672/10000 [=============>................] - ETA: 2s
 4832/10000 [=============>................] - ETA: 1s
 4992/10000 [=============>................] - ETA: 1s
 5152/10000 [==============>...............] - ETA: 1s
 5312/10000 [==============>...............] - ETA: 1s
 5472/10000 [===============>..............] - ETA: 1s
 5632/10000 [===============>..............] - ETA: 1s
 5792/10000 [================>.............] - ETA: 1s
 5952/10000 [================>.............] - ETA: 1s
 6112/10000 [=================>............] - ETA: 1s
 6272/10000 [=================>............] - ETA: 1s
 6432/10000 [==================>...........] - ETA: 1s
 6592/10000 [==================>...........] - ETA: 1s
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
 8928/10000 [=========================>....] - ETA: 0s
 9088/10000 [==========================>...] - ETA: 0s
 9248/10000 [==========================>...] - ETA: 0s
 9408/10000 [===========================>..] - ETA: 0s
 9568/10000 [===========================>..] - ETA: 0s
 9728/10000 [============================>.] - ETA: 0s
 9888/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 4s 375us/step
[[1.26881687e-07 8.33129903e-08 1.00652767e-06 ... 9.99995947e-01
  1.06886811e-08 8.51068251e-07]
 [1.82445965e-05 2.79993747e-05 9.99898434e-01 ... 1.74024422e-07
  1.47867161e-06 1.11388516e-08]
 [3.20438926e-06 9.99706805e-01 9.78685348e-05 ... 1.09123306e-04
  2.53069829e-05 3.07991513e-06]
 ...
 [4.47089299e-08 3.54932376e-06 2.45428197e-07 ... 2.11896040e-05
  8.45864633e-06 1.01631660e-04]
 [8.59787008e-07 1.11138502e-06 2.58340940e-08 ... 1.03429343e-07
  2.94744415e-04 3.00559418e-06]
 [9.61363639e-06 7.33268905e-07 3.75812706e-05 ... 3.55130361e-08
  2.97223255e-06 1.14141997e-07]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.04702493347926065, 'accuracy_test:': 0.9840999841690063}

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
[master e72beb0] ml_store  && git pull --all
 1 file changed, 2037 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 13509ea...e72beb0 master -> master (forced update)





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
{'loss': 0.4145846627652645, 'loss_history': []}

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
[master 0503972] ml_store  && git pull --all
 1 file changed, 111 insertions(+)
To github.com:arita37/mlmodels_store.git
   e72beb0..0503972  master -> master





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
[master 6a9d9a4] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   0503972..6a9d9a4  master -> master





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
 40%|████      | 2/5 [00:22<00:33, 11.14s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9534175653409722, 'learning_rate': 0.08567585987855597, 'min_data_in_leaf': 18, 'num_leaves': 38} and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee\x82e\x8d\xd2I\x10X\r\x00\x00\x00learning_rateq\x02G?\xb5\xee\xdah<-\xf8X\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K&u.' and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xee\x82e\x8d\xd2I\x10X\r\x00\x00\x00learning_rateq\x02G?\xb5\xee\xdah<-\xf8X\x10\x00\x00\x00min_data_in_leafq\x03K\x12X\n\x00\x00\x00num_leavesq\x04K&u.' and reward: 0.3924
 60%|██████    | 3/5 [00:45<00:29, 14.66s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.7685166877943908, 'learning_rate': 0.15533310322694127, 'min_data_in_leaf': 20, 'num_leaves': 35} and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\x97\xb0O\x10:\x94X\r\x00\x00\x00learning_rateq\x02G?\xc3\xe1\xf4\x83,F\x9cX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K#u.' and reward: 0.3912
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe8\x97\xb0O\x10:\x94X\r\x00\x00\x00learning_rateq\x02G?\xc3\xe1\xf4\x83,F\x9cX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K#u.' and reward: 0.3912
 80%|████████  | 4/5 [01:05<00:16, 16.47s/it] 80%|████████  | 4/5 [01:05<00:16, 16.46s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.7923239753376374, 'learning_rate': 0.18944708895673518, 'min_data_in_leaf': 7, 'num_leaves': 58} and reward: 0.3852
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9Z\xb7\xcf=-\xfcX\r\x00\x00\x00learning_rateq\x02G?\xc8?\xcd]\xb2\x1fNX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K:u.' and reward: 0.3852
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xe9Z\xb7\xcf=-\xfcX\r\x00\x00\x00learning_rateq\x02G?\xc8?\xcd]\xb2\x1fNX\x10\x00\x00\x00min_data_in_leafq\x03K\x07X\n\x00\x00\x00num_leavesq\x04K:u.' and reward: 0.3852
Time for Gradient Boosting hyperparameter optimization: 93.91906404495239
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9534175653409722, 'learning_rate': 0.08567585987855597, 'min_data_in_leaf': 18, 'num_leaves': 38}
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
 40%|████      | 2/5 [00:57<01:25, 28.59s/it] 40%|████      | 2/5 [00:57<01:25, 28.59s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 1, 'dropout_prob': 0.18244024982551935, 'embedding_size_factor': 0.9927039218211279, 'layers.choice': 0, 'learning_rate': 0.0015151825166530536, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 0.0017896674886805053} and reward: 0.367
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc7Z3\xbd<\xc2)X\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xc4;\x03\xda\xa5[X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?X\xd3"\xd6\xd0\xd3\xeeX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?]Rh\xd5j\nvu.' and reward: 0.367
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x01X\x0c\x00\x00\x00dropout_probq\x02G?\xc7Z3\xbd<\xc2)X\x15\x00\x00\x00embedding_size_factorq\x03G?\xef\xc4;\x03\xda\xa5[X\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?X\xd3"\xd6\xd0\xd3\xeeX\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G?]Rh\xd5j\nvu.' and reward: 0.367
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 116.04186224937439
Best hyperparameter configuration for Tabular Neural Network: 
{'activation.choice': 0, 'dropout_prob': 0.1, 'embedding_size_factor': 1.0, 'layers.choice': 0, 'learning_rate': 0.0005, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1e-06}
Saving dataset/models/trainer.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.74s of the -93.53s of remaining time.
Ensemble size: 79
Ensemble weights: 
[0.25316456 0.03797468 0.16455696 0.29113924 0.01265823 0.24050633]
	0.3994	 = Validation accuracy score
	1.72s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 215.31s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7fd87c430c50>

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
[master 48d93ad] ml_store  && git pull --all
 1 file changed, 199 insertions(+)
To github.com:arita37/mlmodels_store.git
 + ca95378...48d93ad master -> master (forced update)





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
[master 3699487] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   48d93ad..3699487  master -> master





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
100%|██████████| 10/10 [00:02<00:00,  3.54it/s, avg_epoch_loss=5.22]
INFO:root:Epoch[0] Elapsed time 2.831 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.217863
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.2178631782531735 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f3546bcc3c8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f3546bcc3c8>

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
Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 98.51it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 1072.1932779947917,
    "abs_error": 372.0683898925781,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 2.4653064817269947,
    "sMAPE": 0.5145933960190471,
    "MSIS": 98.6122705927245,
    "QuantileLoss[0.5]": 372.0684127807617,
    "Coverage[0.5]": 1.0,
    "RMSE": 32.744362537615416,
    "NRMSE": 0.6893550007919035,
    "ND": 0.6527515612150493,
    "wQuantileLoss[0.5]": 0.6527516013697574,
    "mean_wQuantileLoss": 0.6527516013697574,
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
100%|██████████| 10/10 [00:01<00:00,  6.91it/s, avg_epoch_loss=2.71e+3]
INFO:root:Epoch[0] Elapsed time 1.448 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=2713.411247
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 2713.4112467447917 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f351a59cef0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f351a59cef0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 136.91it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
100%|██████████| 10/10 [00:02<00:00,  4.64it/s, avg_epoch_loss=5.23]
INFO:root:Epoch[0] Elapsed time 2.157 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.233665
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.233664846420288 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f351a5e4828>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f351a5e4828>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 124.36it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 291.73179117838544,
    "abs_error": 187.0802459716797,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2395843224688368,
    "sMAPE": 0.304835361576234,
    "MSIS": 49.58337694291229,
    "QuantileLoss[0.5]": 187.0802459716797,
    "Coverage[0.5]": 0.75,
    "RMSE": 17.08015782065217,
    "NRMSE": 0.3595822699084667,
    "ND": 0.3282109578450521,
    "wQuantileLoss[0.5]": 0.3282109578450521,
    "mean_wQuantileLoss": 0.3282109578450521,
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
 30%|███       | 3/10 [00:13<00:32,  4.65s/it, avg_epoch_loss=6.93] 60%|██████    | 6/10 [00:26<00:17,  4.47s/it, avg_epoch_loss=6.91] 90%|█████████ | 9/10 [00:38<00:04,  4.35s/it, avg_epoch_loss=6.88]100%|██████████| 10/10 [00:42<00:00,  4.24s/it, avg_epoch_loss=6.87]
INFO:root:Epoch[0] Elapsed time 42.427 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=6.869909
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 6.869909334182739 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f351a5d58d0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f351a5d58d0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 139.36it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 54145.973958333336,
    "abs_error": 2751.322509765625,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 18.23012489345008,
    "sMAPE": 1.41721095039309,
    "MSIS": 729.2049827966948,
    "QuantileLoss[0.5]": 2751.322296142578,
    "Coverage[0.5]": 1.0,
    "RMSE": 232.69287474766676,
    "NRMSE": 4.898797363108774,
    "ND": 4.826881596080044,
    "wQuantileLoss[0.5]": 4.826881221302768,
    "mean_wQuantileLoss": 4.826881221302768,
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
100%|██████████| 10/10 [00:00<00:00, 47.02it/s, avg_epoch_loss=5.27]
INFO:root:Epoch[0] Elapsed time 0.213 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=5.272513
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 5.272512674331665 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f35184e54e0>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f35184e54e0>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 125.25it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
{
    "MSE": 343.7166748046875,
    "abs_error": 187.41859436035156,
    "abs_target_sum": 570.0,
    "abs_target_mean": 47.5,
    "seasonal_error": 12.576813222830921,
    "MASE": 1.2418262019144823,
    "sMAPE": 0.32268854442006256,
    "MSIS": 49.67304726774753,
    "QuantileLoss[0.5]": 187.4185905456543,
    "Coverage[0.5]": 0.6666666666666666,
    "RMSE": 18.539597482272573,
    "NRMSE": 0.3903073154162647,
    "ND": 0.3288045515093887,
    "wQuantileLoss[0.5]": 0.3288045448169374,
    "mean_wQuantileLoss": 0.3288045448169374,
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
100%|██████████| 10/10 [00:01<00:00,  7.45it/s, avg_epoch_loss=123]
INFO:root:Epoch[0] Elapsed time 1.343 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=122.866774
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 122.86677375545166 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f351855ef28>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f351855ef28>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 122.58it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
 10%|█         | 1/10 [02:09<19:27, 129.67s/it, avg_epoch_loss=0.703] 20%|██        | 2/10 [05:21<19:46, 148.33s/it, avg_epoch_loss=0.686] 30%|███       | 3/10 [08:47<19:19, 165.59s/it, avg_epoch_loss=0.669] 40%|████      | 4/10 [12:11<17:43, 177.19s/it, avg_epoch_loss=0.652] 50%|█████     | 5/10 [15:47<15:44, 188.87s/it, avg_epoch_loss=0.634] 60%|██████    | 6/10 [19:31<13:16, 199.18s/it, avg_epoch_loss=0.617] 70%|███████   | 7/10 [23:09<10:14, 204.91s/it, avg_epoch_loss=0.599] 80%|████████  | 8/10 [27:00<07:05, 212.83s/it, avg_epoch_loss=0.581] 90%|█████████ | 9/10 [30:24<03:30, 210.28s/it, avg_epoch_loss=0.563]100%|██████████| 10/10 [33:37<00:00, 204.97s/it, avg_epoch_loss=0.546]100%|██████████| 10/10 [33:37<00:00, 201.78s/it, avg_epoch_loss=0.546]
INFO:root:Epoch[0] Elapsed time 2017.844 seconds
INFO:root:Epoch[0] Evaluation metric 'epoch_loss'=0.546187
INFO:root:Loading parameters from best epoch (0)
INFO:root:Final loss: 0.5461866021156311 (occurred at epoch 0)
INFO:root:End model training
<module 'mlmodels.model_gluon.gluonts_model' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_gluon/gluonts_model.py'> <mlmodels.model_gluon.gluonts_model.Model object at 0x7f35183ba4a8>
[array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
{'target': array([57., 43., 55., ..., 44., 61., 59.]), 'start': Timestamp('2015-02-26 21:42:53', freq='5T')}
learning rate from ``lr_scheduler`` has been overwritten by ``learning_rate`` in optimizer.
<mlmodels.model_gluon.gluonts_model.Model object at 0x7f35183ba4a8>

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
Running evaluation:   0%|          | 0/1 [00:00<?, ?it/s]Running evaluation: 100%|██████████| 1/1 [00:00<00:00, 19.93it/s][array([57., 43., 55., ..., 44., 61., 59.])] [Timestamp('2015-02-26 21:42:53', freq='5T')] [] []
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
[master 4beb65c] ml_store  && git pull --all
 1 file changed, 498 insertions(+)
To github.com:arita37/mlmodels_store.git
 + f79d38d...4beb65c master -> master (forced update)





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

  <mlmodels.model_sklearn.model_sklearn.Model object at 0x7f3fd3336470> 

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
[master 8ee9291] ml_store  && git pull --all
 1 file changed, 107 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.118.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   4beb65c..8ee9291  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn//model_lightgbm.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 

  #### Model init, fit   ############################################# 

  #### save the trained model  ####################################### 

  #### Predict   ##################################################### 
[[ 1.64661853e+00 -1.52568032e+00 -6.06998398e-01  7.95026094e-01
   1.08480038e+00 -3.74438319e-01  4.29526140e-01  1.34048197e-01
   1.20205486e+00  1.06222724e-01]
 [ 8.76994650e-01  1.23225307e+00 -8.67787223e-01 -2.54179868e-01
   8.91891405e-01  1.39984394e+00 -8.77281519e-01 -7.81911683e-01
  -4.37508983e-01 -1.44087602e+00]
 [ 8.48069266e-01  4.51946037e-01  6.30195671e-01 -1.57915629e+00
   8.27987371e-01 -8.28627979e-01 -1.05344713e-01  5.28879746e-01
  -2.23708651e+00 -4.14846901e-01]
 [ 1.46893146e+00 -1.47115693e+00  5.85910431e-01 -8.30171895e-01
   1.03345052e+00 -8.80577600e-01 -9.55425262e-01 -2.79097722e-01
   1.62284909e+00  2.06578332e+00]
 [ 1.05936450e-01 -7.37289628e-01  6.50323214e-01  1.64665066e-01
  -1.53556118e+00  7.78174179e-01  5.03170861e-02  3.09816759e-01
   1.05132077e+00  6.06548400e-01]
 [ 1.03967316e+00 -7.31530982e-01  3.61847316e-01 -1.56573815e+00
   9.59288190e-01  1.01382247e+00 -1.78791289e+00 -2.22711263e+00
  -1.69933360e+00 -4.24492791e-01]
 [ 1.14809657e+00 -7.33271604e-01  2.62467445e-01  8.36004719e-01
   1.17353145e+00  1.54335911e+00  2.84748111e-01  7.58805660e-01
   8.84908814e-01  2.76499305e-01]
 [ 8.98917161e-01  5.57439453e-01 -7.58067329e-01  1.81038744e-01
   8.41467206e-01  1.10717545e+00  6.93366226e-01  1.44287693e+00
  -5.39681562e-01 -8.08847196e-01]
 [ 9.67037267e-01  3.82715174e-01 -8.06184817e-01 -2.88997343e-01
   9.08526041e-01 -3.91816240e-01  1.62091229e+00  6.84001328e-01
  -3.53409983e-01 -2.51674208e-01]
 [ 4.46895161e-01  3.86539145e-01  1.35010682e+00 -8.51455657e-01
   8.50637963e-01  1.00088142e+00 -1.16017010e+00 -3.84832249e-01
   1.45810824e+00 -3.31283170e-01]
 [ 1.32720112e+00 -1.61198320e-01  6.02450901e-01 -2.86384915e-01
  -5.78962302e-01 -8.70887650e-01  1.37975819e+00  5.01429590e-01
  -4.78614074e-01 -8.92646674e-01]
 [ 1.14377130e+00  7.27813500e-01  3.52494364e-01  5.15073614e-01
   1.17718111e+00 -2.78253447e+00 -1.94332341e+00  5.84646610e-01
   3.24274243e-01 -2.36436952e-01]
 [ 1.32857949e+00 -5.63236604e-01 -1.06179676e+00  2.39014596e+00
  -1.68450770e+00  2.45422849e-01 -5.69148654e-01  1.15259914e+00
  -2.24235772e-01  1.32247779e-01]
 [ 1.77547698e+00 -2.03394449e-01 -1.98837863e-01  2.42669441e-01
   9.64350564e-01  2.01830179e-01 -5.45774168e-01  6.61020288e-01
   1.79215821e+00 -7.00398505e-01]
 [ 8.58774962e-01  2.29371761e+00 -1.47023709e+00 -8.30010986e-01
  -6.72049816e-01 -1.01951985e+00  5.99213235e-01 -2.14653842e-01
   1.02124813e+00  6.06403944e-01]
 [ 1.07258847e+00 -5.86523939e-01 -1.34267579e+00 -1.23685338e+00
   1.24328724e+00  8.75838928e-01 -3.26499498e-01  6.23362177e-01
  -4.34956683e-01  1.11438298e+00]
 [ 7.22978007e-01  1.85535621e-01  9.15499268e-01  3.94428030e-01
  -8.49830738e-01  7.25522558e-01 -1.50504326e-01  1.49588477e+00
   6.75453809e-01 -4.38200267e-01]
 [ 1.21619061e+00 -1.90005215e-02  8.60891241e-01 -2.26760192e-01
  -1.36419132e+00 -1.56450785e+00  1.63169151e+00  9.31255679e-01
   9.49808815e-01 -8.80189065e-01]
 [ 9.64572049e-01 -1.06793987e-01  1.12232832e+00  1.45142926e+00
   1.21828168e+00 -6.18036848e-01  4.38166347e-01 -2.03720123e+00
  -1.94258918e+00 -9.97019796e-01]
 [ 1.06702918e+00 -4.29142278e-01  3.50167159e-01  1.20845633e+00
   7.51480619e-01  1.11570180e+00 -4.79157099e-01  8.40861558e-01
  -1.02887218e-01  1.71647264e-02]
 [ 9.97855163e-01 -6.00138799e-01  4.57947076e-01  1.46765263e-01
  -9.33557290e-01  5.71804879e-01  5.72962726e-01 -3.68176565e-02
   1.12368489e-01 -1.78175491e-02]
 [ 1.13545112e+00  8.61623101e-01  4.90616924e-02 -2.08639057e+00
  -1.11469020e+00  3.61801641e-01 -8.06178212e-01  4.25920177e-01
   4.90803971e-02 -5.96086335e-01]
 [ 1.98519313e+00  6.74711526e-01 -1.39662042e+00  6.18539131e-01
   1.22382712e+00 -4.43171931e-01 -1.89148284e-03  1.81053491e+00
  -1.30572692e+00 -8.61316361e-01]
 [ 8.53355545e-01 -7.04350332e-01 -6.79383783e-01 -4.58666861e-02
  -1.29936179e+00 -2.18733459e-01  5.90039464e-01  1.53920701e+00
  -1.14870423e+00 -9.50909251e-01]
 [ 1.44682180e+00  8.07455917e-01  1.49810818e+00  3.12238689e-01
  -6.82430193e-01 -1.93321640e-01  2.88078167e-01 -2.07680202e+00
   9.47501167e-01 -3.00976154e-01]]

  #### metrics   ##################################################### 
{}

  #### Plot   ######################################################## 

  #### Save/Load   ################################################### 
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
{'path': '/home/runner/work/mlmodels/mlmodels/mlmodels/ztest/model_sklearn/model_lightgbm/model.pkl'}
<__main__.Model object at 0x7fb600349d68>

  #### Module init   ############################################ 

  <module 'mlmodels.model_sklearn.model_lightgbm' from '/home/runner/work/mlmodels/mlmodels/mlmodels/model_sklearn/model_lightgbm.py'> 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Model init   ############################################ 

  <mlmodels.model_sklearn.model_lightgbm.Model object at 0x7fb6213b7fd0> 

  #### Fit   ######################################################## 

  #### Predict   #################################################### 
[[ 1.32720112 -0.16119832  0.6024509  -0.28638492 -0.5789623  -0.87088765
   1.37975819  0.50142959 -0.47861407 -0.89264667]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 2.07582971 -1.40232915 -0.47918492  0.45112294  1.03436581 -0.6949209
  -0.4189379   0.5154138  -1.11487105 -1.95210529]
 [ 0.93621125  0.20437739 -1.49419377  0.61223252 -0.98437725  0.74488454
   0.49434165 -0.03628129 -0.83239535 -0.4466992 ]
 [ 0.6236295   0.98635218  1.45391758 -0.46615486  0.93640333  1.38499134
   0.03494359 -1.07296428  0.49515861  0.66168108]
 [ 1.18468624 -1.00016919 -0.59384307  1.04499441  0.96548233  0.6085147
  -0.625342   -0.0693287  -0.10839207 -0.34390071]
 [ 1.39198128 -0.19022103 -0.53722302 -0.44873803  0.70455707 -0.67244804
  -0.70134443 -0.55749472  0.93916874  0.15626385]
 [ 1.34728643 -0.36453805  0.08075099 -0.45971768 -0.8894876   1.70548352
   0.09499611  0.24050555 -0.9994265  -0.76780375]
 [ 0.89562312 -2.29820588 -0.01952256  1.45652739 -1.85064099  0.31663724
   0.11133727 -2.66412594 -0.42642862 -0.83998891]
 [ 1.25704434 -1.82391985 -0.61240697  1.16707517 -0.62373281 -0.0396687
   0.81604368  0.8858258   0.18986165  0.39310924]
 [ 0.72297801  0.18553562  0.91549927  0.39442803 -0.84983074  0.72552256
  -0.15050433  1.49588477  0.67545381 -0.43820027]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 1.32857949 -0.5632366  -1.06179676  2.39014596 -1.6845077   0.24542285
  -0.56914865  1.15259914 -0.22423577  0.13224778]
 [ 1.58463774  0.057121   -0.01771832 -0.79954749  1.32970299 -0.2915946
  -1.1077125  -0.25898285  0.1892932  -1.71939447]
 [ 1.77547698 -0.20339445 -0.19883786  0.24266944  0.96435056  0.20183018
  -0.54577417  0.66102029  1.79215821 -0.7003985 ]
 [ 0.69211449 -0.06065249  2.05635552 -2.413503    1.17456965 -1.77756638
  -0.28173627 -0.77785883  1.11584111  1.76024923]
 [ 1.07258847 -0.58652394 -1.34267579 -1.23685338  1.24328724  0.87583893
  -0.3264995   0.62336218 -0.43495668  1.11438298]
 [ 0.77370361  1.27852808 -2.11416392 -0.44222928  1.06821044  0.32352735
  -2.50644065 -0.10999149  0.00854895 -0.41163916]
 [ 0.85771953  0.09811225 -0.26046606  1.06032751 -1.39003042 -1.71116766
   0.2656424   1.65712464  1.41767401  0.44509671]
 [ 1.36586461  3.9586027   0.54812958  0.64864364  0.84917607  0.10734329
   1.38631426 -1.39881282  0.08176782 -1.63744959]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.92686981  0.39233491 -0.4234783   0.44838065 -1.09230828  1.1253235
  -0.94843966  0.10405339  0.52800342  1.00796648]
 [ 0.61363671  0.3166589   1.34710546 -1.89526695 -0.76045809  0.08972912
  -0.32905155  0.41026575  0.85987097 -1.04906775]
 [ 0.6675918  -0.45252497 -0.60598132  1.16128569 -1.44620987  1.06996554
   1.92381543 -1.04553425  0.35528451  1.80358898]]
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
 [ 1.24549398 -0.72239191  1.1181334   1.09899633  1.00277655 -0.90163449
  -0.53223402 -0.82246719  0.72171129  0.6743961 ]
 [ 0.96703727  0.38271517 -0.80618482 -0.28899734  0.90852604 -0.39181624
   1.62091229  0.68400133 -0.35340998 -0.25167421]
 [ 1.27991386 -0.87142207 -0.32403233 -0.86482994 -0.96853969  0.60874908
   0.50798434  0.5616381   1.51475038 -1.51107661]
 [ 0.89891716  0.55743945 -0.75806733  0.18103874  0.84146721  1.10717545
   0.69336623  1.44287693 -0.53968156 -0.8088472 ]
 [ 0.85877496  2.29371761 -1.47023709 -0.83001099 -0.67204982 -1.01951985
   0.59921324 -0.21465384  1.02124813  0.60640394]
 [ 1.06040861  0.5103076   0.50172511 -0.91579185 -0.90731836 -0.40725204
  -0.17961229  0.98495167  1.07125243 -0.59334375]
 [ 0.85982375  0.17195713 -0.34898419  0.49056104 -1.15649503 -1.39528303
   0.61472628 -0.52235647 -0.3692559  -0.977773  ]
 [ 1.46893146 -1.47115693  0.58591043 -0.8301719   1.03345052 -0.8805776
  -0.95542526 -0.27909772  1.62284909  2.06578332]
 [ 0.8786438   1.03703898 -0.47712421  0.67261975 -1.04948638  2.42887697
   0.52475049  1.00568668  0.35356722 -0.03599018]
 [ 1.34740825  0.73302323  0.83863475 -1.89881206 -0.54245992 -1.11711069
  -1.09715436 -0.50897228 -0.16648595 -1.03918232]
 [ 0.88883881  1.03368687 -0.04970258  0.80884436  0.81405135  1.78975468
   1.14690038  0.45128402 -1.68405999  0.46664327]
 [ 0.97139534  0.71304905  1.76041518  1.30620607  1.0576549  -0.60460297
   0.12837699  0.63658341  1.40925339  0.96653925]
 [ 0.88861146  0.84958685 -0.03091142 -0.12215402 -1.14722826 -0.68085157
  -0.32606131 -1.06787658 -0.07667936  0.35571726]
 [ 1.4468218   0.80745592  1.49810818  0.31223869 -0.68243019 -0.19332164
   0.28807817 -2.07680202  0.94750117 -0.30097615]
 [ 1.06702918 -0.42914228  0.35016716  1.20845633  0.75148062  1.1157018
  -0.4791571   0.84086156 -0.10288722  0.01716473]
 [ 0.70017571  0.55607351  0.08968641  1.69380911  0.88239331  0.19686978
  -0.56378873  0.16986926 -1.16400797 -0.6011568 ]
 [ 1.13545112  0.8616231   0.04906169 -2.08639057 -1.1146902   0.36180164
  -0.80617821  0.42592018  0.0490804  -0.59608633]
 [ 0.85729649  0.9561217  -0.82609743 -0.70584051  1.13872896  1.19268607
   0.28267571 -0.23794194  1.15528789  0.6210827 ]
 [ 1.21619061 -0.01900052  0.86089124 -0.22676019 -1.36419132 -1.56450785
   1.63169151  0.93125568  0.94980882 -0.88018906]
 [ 0.79032389  1.61336137 -2.09424782 -0.37480469  0.91588404 -0.74996962
   0.31027229  2.0546241   0.05340954 -0.22876583]
 [ 1.838294    0.50274088  0.12910158  1.55880554  1.32551412  0.1094027
   1.40754    -1.2197444   2.44936865  1.6169496 ]
 [ 0.94781411 -1.13379204  0.64098587 -0.1905483  -1.23912256  0.23333913
  -0.3169012   0.43499832  0.9104236   1.21987438]
 [ 0.6109426  -2.79099641 -1.33520272 -0.45611756 -0.94495995 -0.97989025
  -0.15699367  0.69257435 -0.47867236 -0.10646012]
 [ 0.85335555 -0.70435033 -0.67938378 -0.04586669 -1.29936179 -0.21873346
   0.59003946  1.53920701 -1.14870423 -0.95090925]]
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
