
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
[master acdd55f] ml_store  && git pull --all
 1 file changed, 60 insertions(+), 9933 deletions(-)
 rewrite log_testall/log_testall.py (99%)
Warning: Permanently added the RSA host key for IP address '140.82.112.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
 + eb4db41...acdd55f master -> master (forced update)





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
[master c7f9e39] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
To github.com:arita37/mlmodels_store.git
   acdd55f..c7f9e39  master -> master





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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sequence_pooling_layer (Sequenc (None, 1, 4)         0           weighted_sequence_layer[0][0]    2020-05-25 00:23:26.324987: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-25 00:23:26.329725: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-25 00:23:26.330562: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561bd1b65cd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 00:23:26.330582: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

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
Total params: 238
Trainable params: 238
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2500 - binary_crossentropy: 0.6932500/500 [==============================] - 1s 1ms/sample - loss: 0.2502 - binary_crossentropy: 0.6935 - val_loss: 0.2501 - val_binary_crossentropy: 0.6934

  #### metrics   #################################################### 
{'MSE': 0.24997753189870758}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         7           sparse_feature_2[0][0]           
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
sparse_seq_emb_sequence_mean (E (None, 6, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 8, 4)         28          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         20          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         28          sparse_feature_2[0][0]           
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
Total params: 238
Trainable params: 238
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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 433
Trainable params: 433
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 1s - loss: 0.2693 - binary_crossentropy: 0.7329500/500 [==============================] - 1s 2ms/sample - loss: 0.2645 - binary_crossentropy: 0.7231 - val_loss: 0.2594 - val_binary_crossentropy: 0.7127

  #### metrics   #################################################### 
{'MSE': 0.2608797588265834}

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
sequence_mean (InputLayer)      [(None, 8)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_3 (Weig (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 8, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 8, 4)         12          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         20          sequence_max[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_weighted_ (None, 3, 1)         2           weighted_seq[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         8           sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 8, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 8, 1)         3           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         5           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_5 (NoMask)              (None, 1, 4)         0           sparse_emb_sparse_feature_0[0][0]
                                                                 sequence_pooling_layer_12[0][0]  
                                                                 sequence_pooling_layer_13[0][0]  
                                                                 sequence_pooling_layer_14[0][0]  
                                                                 sequence_pooling_layer_15[0][0]  
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_0[0][0]           
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
Total params: 433
Trainable params: 433
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 572
Trainable params: 572
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.3700 - binary_crossentropy: 5.7072500/500 [==============================] - 1s 2ms/sample - loss: 0.4620 - binary_crossentropy: 7.1263 - val_loss: 0.4880 - val_binary_crossentropy: 7.5274

  #### metrics   #################################################### 
{'MSE': 0.475}

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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         4           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         20          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 4)         4           sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         36          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         12          sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 4)         4           sparse_feature_2[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         1           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         1           sequence_max[0][0]               
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 3, 4, 1)      5           k_max_pooling[0][0]              
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         1           sparse_feature_2[0][0]           
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
Total params: 572
Trainable params: 572
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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
Total params: 483
Trainable params: 483
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2763 - binary_crossentropy: 0.7502500/500 [==============================] - 1s 3ms/sample - loss: 0.2659 - binary_crossentropy: 0.7287 - val_loss: 0.2709 - val_binary_crossentropy: 0.7392

  #### metrics   #################################################### 
{'MSE': 0.26729616480233337}

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
sparse_seq_emb_sequence_sum (Em (None, 9, 4)         28          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 5, 4)         36          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 9, 4)         24          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         12          sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1 (Em (None, 1, 4)         36          sparse_feature_1[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         9           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 9, 1)         6           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_4 (Flatten)             (None, 28)           0           concatenate_9[0][0]              
__________________________________________________________________________________________________
flatten_5 (Flatten)             (None, 3)            0           concatenate_10[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_1[0][0]           
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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 183
Trainable params: 183
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 2s - loss: 0.2744 - binary_crossentropy: 1.3901500/500 [==============================] - 2s 3ms/sample - loss: 0.2757 - binary_crossentropy: 1.3959 - val_loss: 0.2664 - val_binary_crossentropy: 1.2417

  #### metrics   #################################################### 
{'MSE': 0.27034481362128154}

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
sequence_sum (InputLayer)       [(None, 6)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 3)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 3, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 1, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         16          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
concatenate_14 (Concatenate)    (None, 1, 20)        0           no_mask_22[0][0]                 
                                                                 no_mask_22[1][0]                 
                                                                 no_mask_22[2][0]                 
                                                                 no_mask_22[3][0]                 
                                                                 no_mask_22[4][0]                 
__________________________________________________________________________________________________
no_mask_23 (NoMask)             (None, 1)            0           dense_feature_0[0][0]            
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         4           sparse_feature_0[0][0]           
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
Total params: 183
Trainable params: 183
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
dnn_4 (DNN)                     (None, 4)            152         concatenate_20[0][0]             2020-05-25 00:24:45.727215: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:24:45.729522: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:24:45.735893: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 00:24:45.746797: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 00:24:45.748720: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 00:24:45.750510: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:24:45.752203: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 2s 2s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2556 - val_binary_crossentropy: 0.7045
2020-05-25 00:24:46.962153: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:24:46.964110: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:24:46.968720: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 00:24:46.979167: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer/local_activation_unit/concat' has self cycle fanin 'attention_sequence_pooling_layer/local_activation_unit/concat'.
2020-05-25 00:24:46.980845: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 00:24:46.982538: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:24:46.984072: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.2570666552177017}

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
2020-05-25 00:25:09.888426: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:09.889920: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:09.894076: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 00:25:09.901127: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 00:25:09.902287: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 00:25:09.903401: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:09.904429: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
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
1/1 [==============================] - 3s 3s/sample - loss: 0.2500 - binary_crossentropy: 0.6931 - val_loss: 0.2486 - val_binary_crossentropy: 0.6904
2020-05-25 00:25:11.401814: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:11.403531: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:11.406599: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 00:25:11.412362: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat' has self cycle fanin 'attention_sequence_pooling_layer_1_1/local_activation_unit_2/concat'.
2020-05-25 00:25:11.413356: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 00:25:11.414258: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:11.415096: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.24807064827616068}

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
concatenate_27 (Concatenate)    (None, 1, 16)        0           no_mask_36[0][0]                 2020-05-25 00:25:45.483292: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:45.488719: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:45.504393: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 00:25:45.531742: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 00:25:45.536430: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 00:25:45.540833: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:45.545084: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

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
1/1 [==============================] - 5s 5s/sample - loss: 0.0356 - binary_crossentropy: 0.2092 - val_loss: 0.2585 - val_binary_crossentropy: 0.7105
2020-05-25 00:25:47.712487: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:47.717542: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:47.730155: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] model_pruner failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 00:25:47.755971: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] remapper failed: Invalid argument: MutableGraphView::MutableGraphView error: node 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat' has self cycle fanin 'attention_sequence_pooling_layer_3/local_activation_unit_5/concat'.
2020-05-25 00:25:47.760405: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:533] arithmetic_optimizer failed: Invalid argument: The graph couldn't be sorted in topological order.
2020-05-25 00:25:47.764508: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-05-25 00:25:47.768713: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:697] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.

  #### metrics   #################################################### 
{'MSE': 0.29012730083226107}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
Total params: 720
Trainable params: 720
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2794 - binary_crossentropy: 0.7563500/500 [==============================] - 4s 9ms/sample - loss: 0.2649 - binary_crossentropy: 0.7257 - val_loss: 0.2652 - val_binary_crossentropy: 0.7267

  #### metrics   #################################################### 
{'MSE': 0.2644822856243612}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 6, 4)         8           sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         28          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         36          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         20          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 6, 1)         2           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         9           sequence_max[0][0]               
__________________________________________________________________________________________________
no_mask_48 (NoMask)             (None, 120)          0           flatten_19[0][0]                 
__________________________________________________________________________________________________
concatenate_39 (Concatenate)    (None, 2)            0           no_mask_49[0][0]                 
                                                                 no_mask_49[1][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         5           sparse_feature_0[0][0]           
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
sequence_sum (InputLayer)       [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         18          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_5[0][0]           
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
Total params: 269
Trainable params: 269
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2933 - binary_crossentropy: 0.7907500/500 [==============================] - 4s 9ms/sample - loss: 0.2777 - binary_crossentropy: 0.7572 - val_loss: 0.2713 - val_binary_crossentropy: 0.7396

  #### metrics   #################################################### 
{'MSE': 0.2689421281697339}

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
sequence_mean (InputLayer)      [(None, 6)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 1, 2)         14          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 6, 2)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 6, 2)         18          sequence_max[0][0]               
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
sparse_emb_sparse_feature_1 (Em (None, 1, 2)         4           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_4 (Em (None, 1, 2)         18          sparse_feature_4[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_2 (Em (None, 1, 2)         6           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
sparse_emb_sparse_feature_5 (Em (None, 1, 2)         18          sparse_feature_5[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         7           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         4           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 6, 1)         9           sequence_max[0][0]               
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
linear0sparse_emb_sparse_featur (None, 1, 1)         2           sparse_feature_1[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_4[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         3           sparse_feature_2[0][0]           
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           sparse_feature_5[0][0]           
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
Total params: 269
Trainable params: 269
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,939
Trainable params: 1,939
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 6s - loss: 0.2698 - binary_crossentropy: 0.7342500/500 [==============================] - 5s 9ms/sample - loss: 0.2582 - binary_crossentropy: 0.7102 - val_loss: 0.2563 - val_binary_crossentropy: 0.7061

  #### metrics   #################################################### 
{'MSE': 0.254721689511482}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
__________________________________________________________________________________________________
sequence_max (InputLayer)       [(None, 7)]          0                                            
__________________________________________________________________________________________________
sparse_feature_0 (InputLayer)   [(None, 1)]          0                                            
__________________________________________________________________________________________________
weighted_sequence_layer_21 (Wei (None, 3, 4)         0           sparse_seq_emb_weighted_seq[0][0]
                                                                 weighted_seq_seq_length[0][0]    
                                                                 weight[0][0]                     
__________________________________________________________________________________________________
sparse_seq_emb_sequence_sum (Em (None, 1, 4)         24          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         28          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0 (Em (None, 1, 4)         24          sparse_feature_0[0][0]           
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
linear0sparse_seq_emb_sequence_ (None, 1, 1)         6           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         7           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_24 (Flatten)            (None, 20)           0           concatenate_55[0][0]             
__________________________________________________________________________________________________
flatten_25 (Flatten)            (None, 1)            0           no_mask_69[0][0]                 
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         6           sparse_feature_0[0][0]           
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
Total params: 1,939
Trainable params: 1,939
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
regionsequence_sum (InputLayer) [(None, 2)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
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
Total params: 144
Trainable params: 144
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 8s - loss: 0.2668 - binary_crossentropy: 0.7289500/500 [==============================] - 6s 12ms/sample - loss: 0.2612 - binary_crossentropy: 0.7170 - val_loss: 0.2716 - val_binary_crossentropy: 0.7380

  #### metrics   #################################################### 
{'MSE': 0.266018277390903}

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
regionsequence_sum (InputLayer) [(None, 2)]          0                                            
__________________________________________________________________________________________________
regionsequence_mean (InputLayer [(None, 7)]          0                                            
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
region_10sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_10sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_26 (Wei (None, 3, 1)         0           region_20sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_20sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_28 (Wei (None, 3, 1)         0           region_30sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_30sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_30 (Wei (None, 3, 1)         0           region_40sparse_seq_emb_regionwei
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
region_40sparse_seq_emb_regions (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_32 (Wei (None, 3, 1)         0           learner_10sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_10sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_34 (Wei (None, 3, 1)         0           learner_20sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_20sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_36 (Wei (None, 3, 1)         0           learner_30sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
__________________________________________________________________________________________________
learner_30sparse_seq_emb_region (None, 8, 1)         5           regionsequence_max[0][0]         
__________________________________________________________________________________________________
weighted_sequence_layer_38 (Wei (None, 3, 1)         0           learner_40sparse_seq_emb_regionwe
                                                                 regionweighted_seq_seq_length[0][
                                                                 regionweight[0][0]               
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 2, 1)         3           regionsequence_sum[0][0]         
__________________________________________________________________________________________________
learner_40sparse_seq_emb_region (None, 7, 1)         6           regionsequence_mean[0][0]        
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
Total params: 144
Trainable params: 144
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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
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
Total params: 1,387
Trainable params: 1,387
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 7s - loss: 0.2539 - binary_crossentropy: 0.7010500/500 [==============================] - 6s 12ms/sample - loss: 0.2541 - binary_crossentropy: 0.7014 - val_loss: 0.2495 - val_binary_crossentropy: 0.6921

  #### metrics   #################################################### 
{'MSE': 0.24985569952887265}

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
sequence_mean (InputLayer)      [(None, 1)]          0                                            
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
sparse_seq_emb_sequence_sum (Em (None, 5, 4)         20          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_seq_emb_sequence_mean (E (None, 1, 4)         32          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_seq_emb_sequence_max (Em (None, 7, 4)         20          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 5, 1)         5           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 1, 1)         8           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 7, 1)         5           sequence_max[0][0]               
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
Total params: 1,387
Trainable params: 1,387
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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         20          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
Total params: 3,256
Trainable params: 3,176
Non-trainable params: 80
__________________________________________________________________________________________________
Train on 500 samples, validate on 500 samples
100/500 [=====>........................] - ETA: 9s - loss: 0.2751 - binary_crossentropy: 0.7559500/500 [==============================] - 7s 13ms/sample - loss: 0.2786 - binary_crossentropy: 0.7890 - val_loss: 0.2769 - val_binary_crossentropy: 0.7804

  #### metrics   #################################################### 
{'MSE': 0.27539682363665363}

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
sequence_sum (InputLayer)       [(None, 9)]          0                                            
__________________________________________________________________________________________________
hash_17 (Hash)                  (None, 1)            0           sparse_feature_0[0][0]           
__________________________________________________________________________________________________
sequence_mean (InputLayer)      [(None, 5)]          0                                            
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
sparse_emb_sparse_feature_0_spa (None, 1, 4)         36          hash_14[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_spa (None, 1, 4)         20          hash_15[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_16[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_17[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_0_seq (None, 1, 4)         36          hash_18[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_19[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sparse_ (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_20[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sparse (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sparse_feature_1_seq (None, 1, 4)         20          hash_21[0][0]                    
__________________________________________________________________________________________________
sparse_emb_sequence_max_sparse_ (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_sum_sequenc (None, 9, 4)         36          sequence_sum[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         32          sequence_max[0][0]               
__________________________________________________________________________________________________
sparse_emb_sequence_mean_sequen (None, 5, 4)         24          sequence_mean[0][0]              
__________________________________________________________________________________________________
sparse_emb_sequence_max_sequenc (None, 3, 4)         32          sequence_max[0][0]               
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
linear0sparse_seq_emb_sequence_ (None, 9, 1)         9           sequence_sum[0][0]               
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 5, 1)         6           sequence_mean[0][0]              
__________________________________________________________________________________________________
linear0sparse_seq_emb_sequence_ (None, 3, 1)         8           sequence_max[0][0]               
__________________________________________________________________________________________________
flatten_29 (Flatten)            (None, 40)           0           no_mask_116[0][0]                
__________________________________________________________________________________________________
flatten_30 (Flatten)            (None, 2)            0           concatenate_81[0][0]             
__________________________________________________________________________________________________
linear0sparse_emb_sparse_featur (None, 1, 1)         9           hash_10[0][0]                    
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
Total params: 3,256
Trainable params: 3,176
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
[master 37a0f4f] ml_store  && git pull --all
 1 file changed, 4945 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 2cf9ea2...37a0f4f master -> master (forced update)





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
[master 2889ba2] ml_store  && git pull --all
 1 file changed, 49 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.112.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   37a0f4f..2889ba2  master -> master





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
[master b97a401] ml_store  && git pull --all
 1 file changed, 46 insertions(+)
To github.com:arita37/mlmodels_store.git
   2889ba2..b97a401  master -> master





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
[master 5b2941d] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.3' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   b97a401..5b2941d  master -> master





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
[master 86ee5a4] ml_store  && git pull --all
 1 file changed, 48 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.114.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   5b2941d..86ee5a4  master -> master





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
[master dc91b33] ml_store  && git pull --all
 1 file changed, 47 insertions(+)
To github.com:arita37/mlmodels_store.git
   86ee5a4..dc91b33  master -> master





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
[master 4878a45] ml_store  && git pull --all
 1 file changed, 43 insertions(+)
To github.com:arita37/mlmodels_store.git
   dc91b33..4878a45  master -> master





 ************************************************************************************************************************

  python /home/runner/work/mlmodels/mlmodels/mlmodels/model_keras//textcnn.py 

  #### Loading params   ############################################## 

  #### Path params   ########################################## 

  #### Loading dataset   ############################################# 
Loading data...
Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz

    8192/17464789 [..............................] - ETA: 0s
 4407296/17464789 [======>.......................] - ETA: 0s
11526144/17464789 [==================>...........] - ETA: 0s
16482304/17464789 [===========================>..] - ETA: 0s
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
2020-05-25 00:35:38.797030: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-25 00:35:38.801866: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-05-25 00:35:38.802044: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56514a1f7090 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-25 00:35:38.802062: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

 1000/25000 [>.............................] - ETA: 13s - loss: 7.6820 - accuracy: 0.4990
 2000/25000 [=>............................] - ETA: 9s - loss: 7.5363 - accuracy: 0.5085 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7228 - accuracy: 0.4963
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.7241 - accuracy: 0.4963
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.8230 - accuracy: 0.4898
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.8532 - accuracy: 0.4878
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.8616 - accuracy: 0.4873
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.8200 - accuracy: 0.4900
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.8012 - accuracy: 0.4912
10000/25000 [===========>..................] - ETA: 4s - loss: 7.7602 - accuracy: 0.4939
11000/25000 [============>.................] - ETA: 4s - loss: 7.7684 - accuracy: 0.4934
12000/25000 [=============>................] - ETA: 4s - loss: 7.7573 - accuracy: 0.4941
13000/25000 [==============>...............] - ETA: 3s - loss: 7.7433 - accuracy: 0.4950
14000/25000 [===============>..............] - ETA: 3s - loss: 7.7104 - accuracy: 0.4971
15000/25000 [=================>............] - ETA: 3s - loss: 7.6993 - accuracy: 0.4979
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6954 - accuracy: 0.4981
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6847 - accuracy: 0.4988
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6871 - accuracy: 0.4987
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6836 - accuracy: 0.4989
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6781 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6469 - accuracy: 0.5013
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6436 - accuracy: 0.5015
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6560 - accuracy: 0.5007
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6570 - accuracy: 0.5006
25000/25000 [==============================] - 9s 365us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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
(<mlmodels.util.Model_empty object at 0x7f0ef21a71d0>, None)

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

  <mlmodels.model_keras.textcnn.Model object at 0x7f0ef225d8d0> 

  #### Fit   ######################################################## 
Loading data...
Pad sequences (samples x time)...
Train on 25000 samples, validate on 25000 samples
Epoch 1/1

 1000/25000 [>.............................] - ETA: 12s - loss: 8.3106 - accuracy: 0.4580
 2000/25000 [=>............................] - ETA: 9s - loss: 7.9733 - accuracy: 0.4800 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.7382 - accuracy: 0.4953
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.6551 - accuracy: 0.5008
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.6022 - accuracy: 0.5042
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.7075 - accuracy: 0.4973
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.6885 - accuracy: 0.4986
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.7011 - accuracy: 0.4978
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6803 - accuracy: 0.4991
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6559 - accuracy: 0.5007
11000/25000 [============>.................] - ETA: 4s - loss: 7.6959 - accuracy: 0.4981
12000/25000 [=============>................] - ETA: 4s - loss: 7.6845 - accuracy: 0.4988
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6796 - accuracy: 0.4992
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6984 - accuracy: 0.4979
15000/25000 [=================>............] - ETA: 3s - loss: 7.6789 - accuracy: 0.4992
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6714 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6576 - accuracy: 0.5006
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6726 - accuracy: 0.4996
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6731 - accuracy: 0.4996
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6781 - accuracy: 0.4992
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6666 - accuracy: 0.5000
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6729 - accuracy: 0.4996
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6660 - accuracy: 0.5000
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6641 - accuracy: 0.5002
25000/25000 [==============================] - 9s 366us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000

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

 1000/25000 [>.............................] - ETA: 12s - loss: 6.8846 - accuracy: 0.5510
 2000/25000 [=>............................] - ETA: 9s - loss: 7.2910 - accuracy: 0.5245 
 3000/25000 [==>...........................] - ETA: 8s - loss: 7.4520 - accuracy: 0.5140
 4000/25000 [===>..........................] - ETA: 7s - loss: 7.4558 - accuracy: 0.5138
 5000/25000 [=====>........................] - ETA: 6s - loss: 7.4857 - accuracy: 0.5118
 6000/25000 [======>.......................] - ETA: 6s - loss: 7.4928 - accuracy: 0.5113
 7000/25000 [=======>......................] - ETA: 5s - loss: 7.5396 - accuracy: 0.5083
 8000/25000 [========>.....................] - ETA: 5s - loss: 7.5880 - accuracy: 0.5051
 9000/25000 [=========>....................] - ETA: 5s - loss: 7.6155 - accuracy: 0.5033
10000/25000 [===========>..................] - ETA: 4s - loss: 7.6390 - accuracy: 0.5018
11000/25000 [============>.................] - ETA: 4s - loss: 7.6638 - accuracy: 0.5002
12000/25000 [=============>................] - ETA: 4s - loss: 7.6679 - accuracy: 0.4999
13000/25000 [==============>...............] - ETA: 3s - loss: 7.6501 - accuracy: 0.5011
14000/25000 [===============>..............] - ETA: 3s - loss: 7.6754 - accuracy: 0.4994
15000/25000 [=================>............] - ETA: 3s - loss: 7.6625 - accuracy: 0.5003
16000/25000 [==================>...........] - ETA: 2s - loss: 7.6714 - accuracy: 0.4997
17000/25000 [===================>..........] - ETA: 2s - loss: 7.6558 - accuracy: 0.5007
18000/25000 [====================>.........] - ETA: 2s - loss: 7.6479 - accuracy: 0.5012
19000/25000 [=====================>........] - ETA: 1s - loss: 7.6553 - accuracy: 0.5007
20000/25000 [=======================>......] - ETA: 1s - loss: 7.6574 - accuracy: 0.5006
21000/25000 [========================>.....] - ETA: 1s - loss: 7.6571 - accuracy: 0.5006
22000/25000 [=========================>....] - ETA: 0s - loss: 7.6652 - accuracy: 0.5001
23000/25000 [==========================>...] - ETA: 0s - loss: 7.6820 - accuracy: 0.4990
24000/25000 [===========================>..] - ETA: 0s - loss: 7.6711 - accuracy: 0.4997
25000/25000 [==============================] - 9s 368us/step - loss: 7.6666 - accuracy: 0.5000 - val_loss: 7.6246 - val_accuracy: 0.5000
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
[master fe6ea2d] ml_store  && git pull --all
 1 file changed, 316 insertions(+)
To github.com:arita37/mlmodels_store.git
 + 4908409...fe6ea2d master -> master (forced update)





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

13/13 [==============================] - 2s 132ms/step - loss: nan
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
[master 0740752] ml_store  && git pull --all
 1 file changed, 124 insertions(+)
Warning: Permanently added the RSA host key for IP address '140.82.113.4' to the list of known hosts.
To github.com:arita37/mlmodels_store.git
   fe6ea2d..0740752  master -> master





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
 4677632/11490434 [===========>..................] - ETA: 0s
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

   32/60000 [..............................] - ETA: 7:33 - loss: 2.3184 - categorical_accuracy: 0.0000e+00
   64/60000 [..............................] - ETA: 4:42 - loss: 2.3190 - categorical_accuracy: 0.0469    
   96/60000 [..............................] - ETA: 3:43 - loss: 2.3060 - categorical_accuracy: 0.0729
  128/60000 [..............................] - ETA: 3:14 - loss: 2.2827 - categorical_accuracy: 0.1016
  160/60000 [..............................] - ETA: 2:56 - loss: 2.2456 - categorical_accuracy: 0.1500
  192/60000 [..............................] - ETA: 2:47 - loss: 2.2190 - categorical_accuracy: 0.1615
  224/60000 [..............................] - ETA: 2:38 - loss: 2.1918 - categorical_accuracy: 0.1786
  256/60000 [..............................] - ETA: 2:32 - loss: 2.1703 - categorical_accuracy: 0.1797
  288/60000 [..............................] - ETA: 2:29 - loss: 2.1477 - categorical_accuracy: 0.2014
  320/60000 [..............................] - ETA: 2:24 - loss: 2.1027 - categorical_accuracy: 0.2281
  352/60000 [..............................] - ETA: 2:21 - loss: 2.0681 - categorical_accuracy: 0.2443
  384/60000 [..............................] - ETA: 2:18 - loss: 2.0615 - categorical_accuracy: 0.2500
  416/60000 [..............................] - ETA: 2:16 - loss: 2.0286 - categorical_accuracy: 0.2812
  448/60000 [..............................] - ETA: 2:14 - loss: 2.0017 - categorical_accuracy: 0.2902
  480/60000 [..............................] - ETA: 2:12 - loss: 1.9704 - categorical_accuracy: 0.3000
  512/60000 [..............................] - ETA: 2:10 - loss: 1.9236 - categorical_accuracy: 0.3145
  544/60000 [..............................] - ETA: 2:08 - loss: 1.8834 - categorical_accuracy: 0.3235
  576/60000 [..............................] - ETA: 2:07 - loss: 1.8387 - categorical_accuracy: 0.3438
  608/60000 [..............................] - ETA: 2:06 - loss: 1.8011 - categorical_accuracy: 0.3569
  640/60000 [..............................] - ETA: 2:05 - loss: 1.7623 - categorical_accuracy: 0.3766
  672/60000 [..............................] - ETA: 2:04 - loss: 1.7313 - categorical_accuracy: 0.3899
  704/60000 [..............................] - ETA: 2:03 - loss: 1.7112 - categorical_accuracy: 0.3991
  736/60000 [..............................] - ETA: 2:03 - loss: 1.6892 - categorical_accuracy: 0.4090
  768/60000 [..............................] - ETA: 2:02 - loss: 1.6709 - categorical_accuracy: 0.4180
  800/60000 [..............................] - ETA: 2:01 - loss: 1.6343 - categorical_accuracy: 0.4338
  832/60000 [..............................] - ETA: 2:01 - loss: 1.6016 - categorical_accuracy: 0.4483
  864/60000 [..............................] - ETA: 2:00 - loss: 1.5776 - categorical_accuracy: 0.4572
  896/60000 [..............................] - ETA: 1:59 - loss: 1.5726 - categorical_accuracy: 0.4576
  928/60000 [..............................] - ETA: 1:59 - loss: 1.5527 - categorical_accuracy: 0.4634
  960/60000 [..............................] - ETA: 1:58 - loss: 1.5260 - categorical_accuracy: 0.4750
  992/60000 [..............................] - ETA: 1:58 - loss: 1.5011 - categorical_accuracy: 0.4829
 1024/60000 [..............................] - ETA: 1:57 - loss: 1.4844 - categorical_accuracy: 0.4912
 1056/60000 [..............................] - ETA: 1:57 - loss: 1.4599 - categorical_accuracy: 0.4981
 1088/60000 [..............................] - ETA: 1:56 - loss: 1.4364 - categorical_accuracy: 0.5064
 1120/60000 [..............................] - ETA: 1:56 - loss: 1.4174 - categorical_accuracy: 0.5134
 1152/60000 [..............................] - ETA: 1:55 - loss: 1.3958 - categorical_accuracy: 0.5234
 1184/60000 [..............................] - ETA: 1:55 - loss: 1.3884 - categorical_accuracy: 0.5253
 1216/60000 [..............................] - ETA: 1:55 - loss: 1.3714 - categorical_accuracy: 0.5329
 1248/60000 [..............................] - ETA: 1:54 - loss: 1.3456 - categorical_accuracy: 0.5425
 1280/60000 [..............................] - ETA: 1:54 - loss: 1.3184 - categorical_accuracy: 0.5523
 1312/60000 [..............................] - ETA: 1:54 - loss: 1.3021 - categorical_accuracy: 0.5595
 1344/60000 [..............................] - ETA: 1:53 - loss: 1.2816 - categorical_accuracy: 0.5662
 1376/60000 [..............................] - ETA: 1:53 - loss: 1.2665 - categorical_accuracy: 0.5712
 1408/60000 [..............................] - ETA: 1:53 - loss: 1.2542 - categorical_accuracy: 0.5753
 1440/60000 [..............................] - ETA: 1:53 - loss: 1.2515 - categorical_accuracy: 0.5771
 1472/60000 [..............................] - ETA: 1:52 - loss: 1.2385 - categorical_accuracy: 0.5822
 1504/60000 [..............................] - ETA: 1:52 - loss: 1.2261 - categorical_accuracy: 0.5858
 1536/60000 [..............................] - ETA: 1:52 - loss: 1.2206 - categorical_accuracy: 0.5872
 1568/60000 [..............................] - ETA: 1:52 - loss: 1.2142 - categorical_accuracy: 0.5906
 1600/60000 [..............................] - ETA: 1:51 - loss: 1.2041 - categorical_accuracy: 0.5944
 1632/60000 [..............................] - ETA: 1:51 - loss: 1.1882 - categorical_accuracy: 0.6011
 1664/60000 [..............................] - ETA: 1:51 - loss: 1.1742 - categorical_accuracy: 0.6070
 1696/60000 [..............................] - ETA: 1:51 - loss: 1.1596 - categorical_accuracy: 0.6114
 1728/60000 [..............................] - ETA: 1:50 - loss: 1.1546 - categorical_accuracy: 0.6128
 1760/60000 [..............................] - ETA: 1:50 - loss: 1.1430 - categorical_accuracy: 0.6159
 1792/60000 [..............................] - ETA: 1:50 - loss: 1.1371 - categorical_accuracy: 0.6189
 1824/60000 [..............................] - ETA: 1:50 - loss: 1.1267 - categorical_accuracy: 0.6234
 1856/60000 [..............................] - ETA: 1:50 - loss: 1.1228 - categorical_accuracy: 0.6239
 1888/60000 [..............................] - ETA: 1:49 - loss: 1.1100 - categorical_accuracy: 0.6287
 1920/60000 [..............................] - ETA: 1:49 - loss: 1.1045 - categorical_accuracy: 0.6318
 1952/60000 [..............................] - ETA: 1:49 - loss: 1.0941 - categorical_accuracy: 0.6358
 1984/60000 [..............................] - ETA: 1:49 - loss: 1.0879 - categorical_accuracy: 0.6366
 2016/60000 [>.............................] - ETA: 1:49 - loss: 1.0754 - categorical_accuracy: 0.6414
 2048/60000 [>.............................] - ETA: 1:49 - loss: 1.0627 - categorical_accuracy: 0.6465
 2080/60000 [>.............................] - ETA: 1:49 - loss: 1.0578 - categorical_accuracy: 0.6495
 2112/60000 [>.............................] - ETA: 1:48 - loss: 1.0561 - categorical_accuracy: 0.6501
 2144/60000 [>.............................] - ETA: 1:48 - loss: 1.0479 - categorical_accuracy: 0.6530
 2176/60000 [>.............................] - ETA: 1:48 - loss: 1.0406 - categorical_accuracy: 0.6558
 2208/60000 [>.............................] - ETA: 1:48 - loss: 1.0370 - categorical_accuracy: 0.6576
 2240/60000 [>.............................] - ETA: 1:48 - loss: 1.0285 - categorical_accuracy: 0.6612
 2272/60000 [>.............................] - ETA: 1:48 - loss: 1.0200 - categorical_accuracy: 0.6642
 2304/60000 [>.............................] - ETA: 1:48 - loss: 1.0134 - categorical_accuracy: 0.6654
 2336/60000 [>.............................] - ETA: 1:47 - loss: 1.0052 - categorical_accuracy: 0.6682
 2368/60000 [>.............................] - ETA: 1:47 - loss: 0.9991 - categorical_accuracy: 0.6698
 2400/60000 [>.............................] - ETA: 1:47 - loss: 0.9919 - categorical_accuracy: 0.6725
 2432/60000 [>.............................] - ETA: 1:47 - loss: 0.9831 - categorical_accuracy: 0.6752
 2464/60000 [>.............................] - ETA: 1:47 - loss: 0.9756 - categorical_accuracy: 0.6782
 2496/60000 [>.............................] - ETA: 1:47 - loss: 0.9672 - categorical_accuracy: 0.6807
 2528/60000 [>.............................] - ETA: 1:47 - loss: 0.9572 - categorical_accuracy: 0.6847
 2560/60000 [>.............................] - ETA: 1:47 - loss: 0.9488 - categorical_accuracy: 0.6875
 2592/60000 [>.............................] - ETA: 1:47 - loss: 0.9414 - categorical_accuracy: 0.6902
 2624/60000 [>.............................] - ETA: 1:46 - loss: 0.9337 - categorical_accuracy: 0.6928
 2656/60000 [>.............................] - ETA: 1:46 - loss: 0.9283 - categorical_accuracy: 0.6958
 2688/60000 [>.............................] - ETA: 1:46 - loss: 0.9217 - categorical_accuracy: 0.6972
 2720/60000 [>.............................] - ETA: 1:46 - loss: 0.9149 - categorical_accuracy: 0.6993
 2752/60000 [>.............................] - ETA: 1:46 - loss: 0.9107 - categorical_accuracy: 0.7006
 2784/60000 [>.............................] - ETA: 1:46 - loss: 0.9038 - categorical_accuracy: 0.7022
 2816/60000 [>.............................] - ETA: 1:46 - loss: 0.8980 - categorical_accuracy: 0.7045
 2848/60000 [>.............................] - ETA: 1:46 - loss: 0.8897 - categorical_accuracy: 0.7075
 2880/60000 [>.............................] - ETA: 1:46 - loss: 0.8847 - categorical_accuracy: 0.7087
 2912/60000 [>.............................] - ETA: 1:45 - loss: 0.8816 - categorical_accuracy: 0.7102
 2944/60000 [>.............................] - ETA: 1:45 - loss: 0.8750 - categorical_accuracy: 0.7126
 2976/60000 [>.............................] - ETA: 1:45 - loss: 0.8688 - categorical_accuracy: 0.7144
 3008/60000 [>.............................] - ETA: 1:45 - loss: 0.8622 - categorical_accuracy: 0.7161
 3040/60000 [>.............................] - ETA: 1:45 - loss: 0.8549 - categorical_accuracy: 0.7184
 3072/60000 [>.............................] - ETA: 1:45 - loss: 0.8495 - categorical_accuracy: 0.7201
 3104/60000 [>.............................] - ETA: 1:45 - loss: 0.8446 - categorical_accuracy: 0.7216
 3136/60000 [>.............................] - ETA: 1:45 - loss: 0.8388 - categorical_accuracy: 0.7235
 3168/60000 [>.............................] - ETA: 1:45 - loss: 0.8326 - categorical_accuracy: 0.7257
 3200/60000 [>.............................] - ETA: 1:44 - loss: 0.8261 - categorical_accuracy: 0.7278
 3232/60000 [>.............................] - ETA: 1:44 - loss: 0.8216 - categorical_accuracy: 0.7293
 3264/60000 [>.............................] - ETA: 1:44 - loss: 0.8148 - categorical_accuracy: 0.7313
 3296/60000 [>.............................] - ETA: 1:44 - loss: 0.8121 - categorical_accuracy: 0.7327
 3328/60000 [>.............................] - ETA: 1:44 - loss: 0.8072 - categorical_accuracy: 0.7341
 3360/60000 [>.............................] - ETA: 1:44 - loss: 0.8054 - categorical_accuracy: 0.7342
 3392/60000 [>.............................] - ETA: 1:44 - loss: 0.8024 - categorical_accuracy: 0.7353
 3424/60000 [>.............................] - ETA: 1:44 - loss: 0.7978 - categorical_accuracy: 0.7369
 3456/60000 [>.............................] - ETA: 1:44 - loss: 0.7915 - categorical_accuracy: 0.7393
 3488/60000 [>.............................] - ETA: 1:43 - loss: 0.7859 - categorical_accuracy: 0.7411
 3520/60000 [>.............................] - ETA: 1:43 - loss: 0.7815 - categorical_accuracy: 0.7423
 3552/60000 [>.............................] - ETA: 1:43 - loss: 0.7777 - categorical_accuracy: 0.7438
 3584/60000 [>.............................] - ETA: 1:43 - loss: 0.7740 - categorical_accuracy: 0.7453
 3616/60000 [>.............................] - ETA: 1:43 - loss: 0.7706 - categorical_accuracy: 0.7461
 3648/60000 [>.............................] - ETA: 1:43 - loss: 0.7700 - categorical_accuracy: 0.7464
 3680/60000 [>.............................] - ETA: 1:43 - loss: 0.7653 - categorical_accuracy: 0.7481
 3712/60000 [>.............................] - ETA: 1:43 - loss: 0.7607 - categorical_accuracy: 0.7497
 3744/60000 [>.............................] - ETA: 1:43 - loss: 0.7570 - categorical_accuracy: 0.7511
 3776/60000 [>.............................] - ETA: 1:43 - loss: 0.7518 - categorical_accuracy: 0.7529
 3808/60000 [>.............................] - ETA: 1:43 - loss: 0.7475 - categorical_accuracy: 0.7542
 3840/60000 [>.............................] - ETA: 1:43 - loss: 0.7440 - categorical_accuracy: 0.7552
 3872/60000 [>.............................] - ETA: 1:43 - loss: 0.7405 - categorical_accuracy: 0.7565
 3904/60000 [>.............................] - ETA: 1:42 - loss: 0.7360 - categorical_accuracy: 0.7582
 3936/60000 [>.............................] - ETA: 1:42 - loss: 0.7320 - categorical_accuracy: 0.7597
 3968/60000 [>.............................] - ETA: 1:42 - loss: 0.7294 - categorical_accuracy: 0.7608
 4000/60000 [=>............................] - ETA: 1:42 - loss: 0.7254 - categorical_accuracy: 0.7620
 4032/60000 [=>............................] - ETA: 1:42 - loss: 0.7205 - categorical_accuracy: 0.7639
 4064/60000 [=>............................] - ETA: 1:42 - loss: 0.7157 - categorical_accuracy: 0.7653
 4096/60000 [=>............................] - ETA: 1:42 - loss: 0.7116 - categorical_accuracy: 0.7666
 4128/60000 [=>............................] - ETA: 1:42 - loss: 0.7075 - categorical_accuracy: 0.7679
 4160/60000 [=>............................] - ETA: 1:42 - loss: 0.7051 - categorical_accuracy: 0.7692
 4192/60000 [=>............................] - ETA: 1:42 - loss: 0.7020 - categorical_accuracy: 0.7703
 4224/60000 [=>............................] - ETA: 1:42 - loss: 0.6990 - categorical_accuracy: 0.7711
 4256/60000 [=>............................] - ETA: 1:41 - loss: 0.6946 - categorical_accuracy: 0.7726
 4288/60000 [=>............................] - ETA: 1:41 - loss: 0.6921 - categorical_accuracy: 0.7733
 4320/60000 [=>............................] - ETA: 1:41 - loss: 0.6894 - categorical_accuracy: 0.7745
 4352/60000 [=>............................] - ETA: 1:41 - loss: 0.6866 - categorical_accuracy: 0.7757
 4384/60000 [=>............................] - ETA: 1:41 - loss: 0.6830 - categorical_accuracy: 0.7767
 4416/60000 [=>............................] - ETA: 1:41 - loss: 0.6799 - categorical_accuracy: 0.7776
 4448/60000 [=>............................] - ETA: 1:41 - loss: 0.6765 - categorical_accuracy: 0.7788
 4480/60000 [=>............................] - ETA: 1:41 - loss: 0.6738 - categorical_accuracy: 0.7792
 4512/60000 [=>............................] - ETA: 1:41 - loss: 0.6728 - categorical_accuracy: 0.7799
 4544/60000 [=>............................] - ETA: 1:41 - loss: 0.6688 - categorical_accuracy: 0.7815
 4576/60000 [=>............................] - ETA: 1:41 - loss: 0.6650 - categorical_accuracy: 0.7828
 4608/60000 [=>............................] - ETA: 1:41 - loss: 0.6613 - categorical_accuracy: 0.7841
 4640/60000 [=>............................] - ETA: 1:41 - loss: 0.6590 - categorical_accuracy: 0.7849
 4672/60000 [=>............................] - ETA: 1:40 - loss: 0.6557 - categorical_accuracy: 0.7860
 4704/60000 [=>............................] - ETA: 1:40 - loss: 0.6540 - categorical_accuracy: 0.7866
 4736/60000 [=>............................] - ETA: 1:40 - loss: 0.6510 - categorical_accuracy: 0.7876
 4768/60000 [=>............................] - ETA: 1:40 - loss: 0.6495 - categorical_accuracy: 0.7880
 4800/60000 [=>............................] - ETA: 1:40 - loss: 0.6481 - categorical_accuracy: 0.7890
 4832/60000 [=>............................] - ETA: 1:40 - loss: 0.6459 - categorical_accuracy: 0.7899
 4864/60000 [=>............................] - ETA: 1:40 - loss: 0.6440 - categorical_accuracy: 0.7901
 4896/60000 [=>............................] - ETA: 1:40 - loss: 0.6408 - categorical_accuracy: 0.7911
 4928/60000 [=>............................] - ETA: 1:40 - loss: 0.6392 - categorical_accuracy: 0.7918
 4960/60000 [=>............................] - ETA: 1:40 - loss: 0.6359 - categorical_accuracy: 0.7929
 4992/60000 [=>............................] - ETA: 1:40 - loss: 0.6332 - categorical_accuracy: 0.7937
 5024/60000 [=>............................] - ETA: 1:39 - loss: 0.6305 - categorical_accuracy: 0.7946
 5056/60000 [=>............................] - ETA: 1:39 - loss: 0.6296 - categorical_accuracy: 0.7949
 5088/60000 [=>............................] - ETA: 1:39 - loss: 0.6287 - categorical_accuracy: 0.7952
 5120/60000 [=>............................] - ETA: 1:39 - loss: 0.6265 - categorical_accuracy: 0.7963
 5152/60000 [=>............................] - ETA: 1:39 - loss: 0.6247 - categorical_accuracy: 0.7968
 5184/60000 [=>............................] - ETA: 1:39 - loss: 0.6238 - categorical_accuracy: 0.7973
 5216/60000 [=>............................] - ETA: 1:39 - loss: 0.6207 - categorical_accuracy: 0.7983
 5248/60000 [=>............................] - ETA: 1:39 - loss: 0.6185 - categorical_accuracy: 0.7990
 5280/60000 [=>............................] - ETA: 1:39 - loss: 0.6165 - categorical_accuracy: 0.7996
 5312/60000 [=>............................] - ETA: 1:39 - loss: 0.6140 - categorical_accuracy: 0.8005
 5344/60000 [=>............................] - ETA: 1:39 - loss: 0.6114 - categorical_accuracy: 0.8013
 5376/60000 [=>............................] - ETA: 1:39 - loss: 0.6084 - categorical_accuracy: 0.8021
 5408/60000 [=>............................] - ETA: 1:38 - loss: 0.6054 - categorical_accuracy: 0.8031
 5440/60000 [=>............................] - ETA: 1:39 - loss: 0.6034 - categorical_accuracy: 0.8039
 5472/60000 [=>............................] - ETA: 1:38 - loss: 0.6023 - categorical_accuracy: 0.8043
 5504/60000 [=>............................] - ETA: 1:38 - loss: 0.6021 - categorical_accuracy: 0.8047
 5536/60000 [=>............................] - ETA: 1:38 - loss: 0.6009 - categorical_accuracy: 0.8049
 5568/60000 [=>............................] - ETA: 1:38 - loss: 0.6001 - categorical_accuracy: 0.8057
 5600/60000 [=>............................] - ETA: 1:38 - loss: 0.5978 - categorical_accuracy: 0.8062
 5632/60000 [=>............................] - ETA: 1:38 - loss: 0.5974 - categorical_accuracy: 0.8065
 5664/60000 [=>............................] - ETA: 1:38 - loss: 0.5963 - categorical_accuracy: 0.8070
 5696/60000 [=>............................] - ETA: 1:38 - loss: 0.5944 - categorical_accuracy: 0.8078
 5728/60000 [=>............................] - ETA: 1:38 - loss: 0.5915 - categorical_accuracy: 0.8088
 5760/60000 [=>............................] - ETA: 1:38 - loss: 0.5885 - categorical_accuracy: 0.8099
 5792/60000 [=>............................] - ETA: 1:38 - loss: 0.5866 - categorical_accuracy: 0.8106
 5824/60000 [=>............................] - ETA: 1:38 - loss: 0.5843 - categorical_accuracy: 0.8113
 5856/60000 [=>............................] - ETA: 1:38 - loss: 0.5823 - categorical_accuracy: 0.8120
 5888/60000 [=>............................] - ETA: 1:37 - loss: 0.5808 - categorical_accuracy: 0.8125
 5920/60000 [=>............................] - ETA: 1:37 - loss: 0.5800 - categorical_accuracy: 0.8125
 5952/60000 [=>............................] - ETA: 1:37 - loss: 0.5777 - categorical_accuracy: 0.8133
 5984/60000 [=>............................] - ETA: 1:37 - loss: 0.5762 - categorical_accuracy: 0.8138
 6016/60000 [==>...........................] - ETA: 1:37 - loss: 0.5762 - categorical_accuracy: 0.8145
 6048/60000 [==>...........................] - ETA: 1:37 - loss: 0.5745 - categorical_accuracy: 0.8151
 6080/60000 [==>...........................] - ETA: 1:37 - loss: 0.5760 - categorical_accuracy: 0.8148
 6112/60000 [==>...........................] - ETA: 1:37 - loss: 0.5737 - categorical_accuracy: 0.8156
 6144/60000 [==>...........................] - ETA: 1:37 - loss: 0.5722 - categorical_accuracy: 0.8161
 6176/60000 [==>...........................] - ETA: 1:37 - loss: 0.5699 - categorical_accuracy: 0.8169
 6208/60000 [==>...........................] - ETA: 1:37 - loss: 0.5681 - categorical_accuracy: 0.8175
 6240/60000 [==>...........................] - ETA: 1:37 - loss: 0.5663 - categorical_accuracy: 0.8178
 6272/60000 [==>...........................] - ETA: 1:37 - loss: 0.5674 - categorical_accuracy: 0.8178
 6304/60000 [==>...........................] - ETA: 1:37 - loss: 0.5653 - categorical_accuracy: 0.8185
 6336/60000 [==>...........................] - ETA: 1:37 - loss: 0.5635 - categorical_accuracy: 0.8190
 6368/60000 [==>...........................] - ETA: 1:36 - loss: 0.5612 - categorical_accuracy: 0.8197
 6400/60000 [==>...........................] - ETA: 1:36 - loss: 0.5596 - categorical_accuracy: 0.8202
 6432/60000 [==>...........................] - ETA: 1:36 - loss: 0.5582 - categorical_accuracy: 0.8209
 6464/60000 [==>...........................] - ETA: 1:36 - loss: 0.5568 - categorical_accuracy: 0.8210
 6496/60000 [==>...........................] - ETA: 1:36 - loss: 0.5549 - categorical_accuracy: 0.8217
 6528/60000 [==>...........................] - ETA: 1:36 - loss: 0.5531 - categorical_accuracy: 0.8223
 6560/60000 [==>...........................] - ETA: 1:36 - loss: 0.5509 - categorical_accuracy: 0.8230
 6592/60000 [==>...........................] - ETA: 1:36 - loss: 0.5487 - categorical_accuracy: 0.8237
 6624/60000 [==>...........................] - ETA: 1:36 - loss: 0.5476 - categorical_accuracy: 0.8238
 6656/60000 [==>...........................] - ETA: 1:36 - loss: 0.5465 - categorical_accuracy: 0.8242
 6688/60000 [==>...........................] - ETA: 1:36 - loss: 0.5447 - categorical_accuracy: 0.8249
 6720/60000 [==>...........................] - ETA: 1:36 - loss: 0.5436 - categorical_accuracy: 0.8250
 6752/60000 [==>...........................] - ETA: 1:36 - loss: 0.5429 - categorical_accuracy: 0.8254
 6784/60000 [==>...........................] - ETA: 1:35 - loss: 0.5428 - categorical_accuracy: 0.8256
 6816/60000 [==>...........................] - ETA: 1:35 - loss: 0.5423 - categorical_accuracy: 0.8257
 6848/60000 [==>...........................] - ETA: 1:35 - loss: 0.5420 - categorical_accuracy: 0.8258
 6880/60000 [==>...........................] - ETA: 1:35 - loss: 0.5399 - categorical_accuracy: 0.8265
 6912/60000 [==>...........................] - ETA: 1:35 - loss: 0.5381 - categorical_accuracy: 0.8271
 6944/60000 [==>...........................] - ETA: 1:35 - loss: 0.5363 - categorical_accuracy: 0.8276
 6976/60000 [==>...........................] - ETA: 1:35 - loss: 0.5348 - categorical_accuracy: 0.8281
 7008/60000 [==>...........................] - ETA: 1:35 - loss: 0.5347 - categorical_accuracy: 0.8285
 7040/60000 [==>...........................] - ETA: 1:35 - loss: 0.5333 - categorical_accuracy: 0.8290
 7072/60000 [==>...........................] - ETA: 1:35 - loss: 0.5312 - categorical_accuracy: 0.8298
 7104/60000 [==>...........................] - ETA: 1:35 - loss: 0.5293 - categorical_accuracy: 0.8305
 7136/60000 [==>...........................] - ETA: 1:35 - loss: 0.5274 - categorical_accuracy: 0.8311
 7168/60000 [==>...........................] - ETA: 1:35 - loss: 0.5261 - categorical_accuracy: 0.8315
 7200/60000 [==>...........................] - ETA: 1:35 - loss: 0.5260 - categorical_accuracy: 0.8315
 7232/60000 [==>...........................] - ETA: 1:34 - loss: 0.5244 - categorical_accuracy: 0.8319
 7264/60000 [==>...........................] - ETA: 1:34 - loss: 0.5228 - categorical_accuracy: 0.8323
 7296/60000 [==>...........................] - ETA: 1:34 - loss: 0.5211 - categorical_accuracy: 0.8329
 7328/60000 [==>...........................] - ETA: 1:34 - loss: 0.5201 - categorical_accuracy: 0.8334
 7360/60000 [==>...........................] - ETA: 1:34 - loss: 0.5184 - categorical_accuracy: 0.8340
 7392/60000 [==>...........................] - ETA: 1:34 - loss: 0.5182 - categorical_accuracy: 0.8340
 7424/60000 [==>...........................] - ETA: 1:34 - loss: 0.5167 - categorical_accuracy: 0.8345
 7456/60000 [==>...........................] - ETA: 1:34 - loss: 0.5152 - categorical_accuracy: 0.8349
 7488/60000 [==>...........................] - ETA: 1:34 - loss: 0.5138 - categorical_accuracy: 0.8355
 7520/60000 [==>...........................] - ETA: 1:34 - loss: 0.5128 - categorical_accuracy: 0.8356
 7552/60000 [==>...........................] - ETA: 1:34 - loss: 0.5122 - categorical_accuracy: 0.8359
 7584/60000 [==>...........................] - ETA: 1:34 - loss: 0.5125 - categorical_accuracy: 0.8364
 7616/60000 [==>...........................] - ETA: 1:34 - loss: 0.5118 - categorical_accuracy: 0.8367
 7648/60000 [==>...........................] - ETA: 1:34 - loss: 0.5106 - categorical_accuracy: 0.8371
 7680/60000 [==>...........................] - ETA: 1:34 - loss: 0.5103 - categorical_accuracy: 0.8371
 7712/60000 [==>...........................] - ETA: 1:33 - loss: 0.5091 - categorical_accuracy: 0.8375
 7744/60000 [==>...........................] - ETA: 1:33 - loss: 0.5080 - categorical_accuracy: 0.8378
 7776/60000 [==>...........................] - ETA: 1:33 - loss: 0.5063 - categorical_accuracy: 0.8383
 7808/60000 [==>...........................] - ETA: 1:33 - loss: 0.5048 - categorical_accuracy: 0.8388
 7840/60000 [==>...........................] - ETA: 1:33 - loss: 0.5037 - categorical_accuracy: 0.8389
 7872/60000 [==>...........................] - ETA: 1:33 - loss: 0.5026 - categorical_accuracy: 0.8393
 7904/60000 [==>...........................] - ETA: 1:33 - loss: 0.5012 - categorical_accuracy: 0.8397
 7936/60000 [==>...........................] - ETA: 1:33 - loss: 0.4997 - categorical_accuracy: 0.8400
 7968/60000 [==>...........................] - ETA: 1:33 - loss: 0.4986 - categorical_accuracy: 0.8404
 8000/60000 [===>..........................] - ETA: 1:33 - loss: 0.4971 - categorical_accuracy: 0.8410
 8032/60000 [===>..........................] - ETA: 1:33 - loss: 0.4966 - categorical_accuracy: 0.8414
 8064/60000 [===>..........................] - ETA: 1:33 - loss: 0.4950 - categorical_accuracy: 0.8420
 8096/60000 [===>..........................] - ETA: 1:33 - loss: 0.4932 - categorical_accuracy: 0.8426
 8128/60000 [===>..........................] - ETA: 1:33 - loss: 0.4923 - categorical_accuracy: 0.8430
 8160/60000 [===>..........................] - ETA: 1:33 - loss: 0.4907 - categorical_accuracy: 0.8434
 8192/60000 [===>..........................] - ETA: 1:32 - loss: 0.4901 - categorical_accuracy: 0.8436
 8224/60000 [===>..........................] - ETA: 1:32 - loss: 0.4888 - categorical_accuracy: 0.8440
 8256/60000 [===>..........................] - ETA: 1:32 - loss: 0.4890 - categorical_accuracy: 0.8439
 8288/60000 [===>..........................] - ETA: 1:32 - loss: 0.4878 - categorical_accuracy: 0.8444
 8320/60000 [===>..........................] - ETA: 1:32 - loss: 0.4867 - categorical_accuracy: 0.8448
 8352/60000 [===>..........................] - ETA: 1:32 - loss: 0.4854 - categorical_accuracy: 0.8452
 8384/60000 [===>..........................] - ETA: 1:32 - loss: 0.4845 - categorical_accuracy: 0.8453
 8416/60000 [===>..........................] - ETA: 1:32 - loss: 0.4852 - categorical_accuracy: 0.8454
 8448/60000 [===>..........................] - ETA: 1:32 - loss: 0.4842 - categorical_accuracy: 0.8456
 8480/60000 [===>..........................] - ETA: 1:32 - loss: 0.4829 - categorical_accuracy: 0.8462
 8512/60000 [===>..........................] - ETA: 1:32 - loss: 0.4825 - categorical_accuracy: 0.8466
 8544/60000 [===>..........................] - ETA: 1:32 - loss: 0.4813 - categorical_accuracy: 0.8470
 8576/60000 [===>..........................] - ETA: 1:32 - loss: 0.4815 - categorical_accuracy: 0.8471
 8608/60000 [===>..........................] - ETA: 1:32 - loss: 0.4800 - categorical_accuracy: 0.8477
 8640/60000 [===>..........................] - ETA: 1:31 - loss: 0.4789 - categorical_accuracy: 0.8480
 8672/60000 [===>..........................] - ETA: 1:31 - loss: 0.4781 - categorical_accuracy: 0.8484
 8704/60000 [===>..........................] - ETA: 1:31 - loss: 0.4766 - categorical_accuracy: 0.8489
 8736/60000 [===>..........................] - ETA: 1:31 - loss: 0.4750 - categorical_accuracy: 0.8495
 8768/60000 [===>..........................] - ETA: 1:31 - loss: 0.4742 - categorical_accuracy: 0.8498
 8800/60000 [===>..........................] - ETA: 1:31 - loss: 0.4732 - categorical_accuracy: 0.8499
 8832/60000 [===>..........................] - ETA: 1:31 - loss: 0.4725 - categorical_accuracy: 0.8501
 8864/60000 [===>..........................] - ETA: 1:31 - loss: 0.4719 - categorical_accuracy: 0.8503
 8896/60000 [===>..........................] - ETA: 1:31 - loss: 0.4707 - categorical_accuracy: 0.8508
 8928/60000 [===>..........................] - ETA: 1:31 - loss: 0.4695 - categorical_accuracy: 0.8511
 8960/60000 [===>..........................] - ETA: 1:31 - loss: 0.4682 - categorical_accuracy: 0.8516
 8992/60000 [===>..........................] - ETA: 1:31 - loss: 0.4669 - categorical_accuracy: 0.8520
 9024/60000 [===>..........................] - ETA: 1:31 - loss: 0.4661 - categorical_accuracy: 0.8523
 9056/60000 [===>..........................] - ETA: 1:31 - loss: 0.4650 - categorical_accuracy: 0.8527
 9088/60000 [===>..........................] - ETA: 1:31 - loss: 0.4641 - categorical_accuracy: 0.8531
 9120/60000 [===>..........................] - ETA: 1:31 - loss: 0.4630 - categorical_accuracy: 0.8534
 9152/60000 [===>..........................] - ETA: 1:31 - loss: 0.4620 - categorical_accuracy: 0.8538
 9184/60000 [===>..........................] - ETA: 1:30 - loss: 0.4612 - categorical_accuracy: 0.8541
 9216/60000 [===>..........................] - ETA: 1:30 - loss: 0.4604 - categorical_accuracy: 0.8544
 9248/60000 [===>..........................] - ETA: 1:30 - loss: 0.4590 - categorical_accuracy: 0.8549
 9280/60000 [===>..........................] - ETA: 1:30 - loss: 0.4578 - categorical_accuracy: 0.8552
 9312/60000 [===>..........................] - ETA: 1:30 - loss: 0.4580 - categorical_accuracy: 0.8552
 9344/60000 [===>..........................] - ETA: 1:30 - loss: 0.4572 - categorical_accuracy: 0.8556
 9376/60000 [===>..........................] - ETA: 1:30 - loss: 0.4558 - categorical_accuracy: 0.8560
 9408/60000 [===>..........................] - ETA: 1:30 - loss: 0.4561 - categorical_accuracy: 0.8560
 9440/60000 [===>..........................] - ETA: 1:30 - loss: 0.4555 - categorical_accuracy: 0.8560
 9472/60000 [===>..........................] - ETA: 1:30 - loss: 0.4544 - categorical_accuracy: 0.8565
 9504/60000 [===>..........................] - ETA: 1:30 - loss: 0.4531 - categorical_accuracy: 0.8569
 9536/60000 [===>..........................] - ETA: 1:30 - loss: 0.4520 - categorical_accuracy: 0.8572
 9568/60000 [===>..........................] - ETA: 1:30 - loss: 0.4511 - categorical_accuracy: 0.8574
 9600/60000 [===>..........................] - ETA: 1:30 - loss: 0.4501 - categorical_accuracy: 0.8577
 9632/60000 [===>..........................] - ETA: 1:29 - loss: 0.4489 - categorical_accuracy: 0.8582
 9664/60000 [===>..........................] - ETA: 1:29 - loss: 0.4478 - categorical_accuracy: 0.8585
 9696/60000 [===>..........................] - ETA: 1:29 - loss: 0.4478 - categorical_accuracy: 0.8586
 9728/60000 [===>..........................] - ETA: 1:29 - loss: 0.4471 - categorical_accuracy: 0.8588
 9760/60000 [===>..........................] - ETA: 1:29 - loss: 0.4464 - categorical_accuracy: 0.8590
 9792/60000 [===>..........................] - ETA: 1:29 - loss: 0.4455 - categorical_accuracy: 0.8594
 9824/60000 [===>..........................] - ETA: 1:29 - loss: 0.4445 - categorical_accuracy: 0.8596
 9856/60000 [===>..........................] - ETA: 1:29 - loss: 0.4438 - categorical_accuracy: 0.8599
 9888/60000 [===>..........................] - ETA: 1:29 - loss: 0.4430 - categorical_accuracy: 0.8601
 9920/60000 [===>..........................] - ETA: 1:29 - loss: 0.4420 - categorical_accuracy: 0.8606
 9952/60000 [===>..........................] - ETA: 1:29 - loss: 0.4411 - categorical_accuracy: 0.8609
 9984/60000 [===>..........................] - ETA: 1:29 - loss: 0.4402 - categorical_accuracy: 0.8613
10016/60000 [====>.........................] - ETA: 1:29 - loss: 0.4402 - categorical_accuracy: 0.8614
10048/60000 [====>.........................] - ETA: 1:29 - loss: 0.4392 - categorical_accuracy: 0.8618
10080/60000 [====>.........................] - ETA: 1:29 - loss: 0.4380 - categorical_accuracy: 0.8621
10112/60000 [====>.........................] - ETA: 1:29 - loss: 0.4372 - categorical_accuracy: 0.8624
10144/60000 [====>.........................] - ETA: 1:28 - loss: 0.4360 - categorical_accuracy: 0.8628
10176/60000 [====>.........................] - ETA: 1:28 - loss: 0.4351 - categorical_accuracy: 0.8631
10208/60000 [====>.........................] - ETA: 1:28 - loss: 0.4343 - categorical_accuracy: 0.8633
10240/60000 [====>.........................] - ETA: 1:28 - loss: 0.4340 - categorical_accuracy: 0.8635
10272/60000 [====>.........................] - ETA: 1:28 - loss: 0.4334 - categorical_accuracy: 0.8637
10304/60000 [====>.........................] - ETA: 1:28 - loss: 0.4330 - categorical_accuracy: 0.8637
10336/60000 [====>.........................] - ETA: 1:28 - loss: 0.4319 - categorical_accuracy: 0.8641
10368/60000 [====>.........................] - ETA: 1:28 - loss: 0.4312 - categorical_accuracy: 0.8643
10400/60000 [====>.........................] - ETA: 1:28 - loss: 0.4305 - categorical_accuracy: 0.8645
10432/60000 [====>.........................] - ETA: 1:28 - loss: 0.4313 - categorical_accuracy: 0.8646
10464/60000 [====>.........................] - ETA: 1:28 - loss: 0.4304 - categorical_accuracy: 0.8650
10496/60000 [====>.........................] - ETA: 1:28 - loss: 0.4296 - categorical_accuracy: 0.8653
10528/60000 [====>.........................] - ETA: 1:28 - loss: 0.4292 - categorical_accuracy: 0.8654
10560/60000 [====>.........................] - ETA: 1:28 - loss: 0.4281 - categorical_accuracy: 0.8658
10592/60000 [====>.........................] - ETA: 1:28 - loss: 0.4272 - categorical_accuracy: 0.8659
10624/60000 [====>.........................] - ETA: 1:28 - loss: 0.4269 - categorical_accuracy: 0.8660
10656/60000 [====>.........................] - ETA: 1:27 - loss: 0.4264 - categorical_accuracy: 0.8663
10688/60000 [====>.........................] - ETA: 1:27 - loss: 0.4258 - categorical_accuracy: 0.8665
10720/60000 [====>.........................] - ETA: 1:27 - loss: 0.4249 - categorical_accuracy: 0.8668
10752/60000 [====>.........................] - ETA: 1:27 - loss: 0.4244 - categorical_accuracy: 0.8669
10784/60000 [====>.........................] - ETA: 1:27 - loss: 0.4234 - categorical_accuracy: 0.8673
10816/60000 [====>.........................] - ETA: 1:27 - loss: 0.4227 - categorical_accuracy: 0.8675
10848/60000 [====>.........................] - ETA: 1:27 - loss: 0.4218 - categorical_accuracy: 0.8677
10880/60000 [====>.........................] - ETA: 1:27 - loss: 0.4220 - categorical_accuracy: 0.8678
10912/60000 [====>.........................] - ETA: 1:27 - loss: 0.4209 - categorical_accuracy: 0.8682
10944/60000 [====>.........................] - ETA: 1:27 - loss: 0.4203 - categorical_accuracy: 0.8683
10976/60000 [====>.........................] - ETA: 1:27 - loss: 0.4201 - categorical_accuracy: 0.8683
11008/60000 [====>.........................] - ETA: 1:27 - loss: 0.4194 - categorical_accuracy: 0.8686
11040/60000 [====>.........................] - ETA: 1:27 - loss: 0.4189 - categorical_accuracy: 0.8687
11072/60000 [====>.........................] - ETA: 1:27 - loss: 0.4181 - categorical_accuracy: 0.8689
11104/60000 [====>.........................] - ETA: 1:27 - loss: 0.4172 - categorical_accuracy: 0.8692
11136/60000 [====>.........................] - ETA: 1:27 - loss: 0.4170 - categorical_accuracy: 0.8693
11168/60000 [====>.........................] - ETA: 1:26 - loss: 0.4166 - categorical_accuracy: 0.8695
11200/60000 [====>.........................] - ETA: 1:26 - loss: 0.4165 - categorical_accuracy: 0.8696
11232/60000 [====>.........................] - ETA: 1:26 - loss: 0.4155 - categorical_accuracy: 0.8699
11264/60000 [====>.........................] - ETA: 1:26 - loss: 0.4147 - categorical_accuracy: 0.8702
11296/60000 [====>.........................] - ETA: 1:26 - loss: 0.4148 - categorical_accuracy: 0.8703
11328/60000 [====>.........................] - ETA: 1:26 - loss: 0.4139 - categorical_accuracy: 0.8706
11360/60000 [====>.........................] - ETA: 1:26 - loss: 0.4135 - categorical_accuracy: 0.8708
11392/60000 [====>.........................] - ETA: 1:26 - loss: 0.4133 - categorical_accuracy: 0.8709
11424/60000 [====>.........................] - ETA: 1:26 - loss: 0.4127 - categorical_accuracy: 0.8711
11456/60000 [====>.........................] - ETA: 1:26 - loss: 0.4129 - categorical_accuracy: 0.8711
11488/60000 [====>.........................] - ETA: 1:26 - loss: 0.4120 - categorical_accuracy: 0.8714
11520/60000 [====>.........................] - ETA: 1:26 - loss: 0.4111 - categorical_accuracy: 0.8717
11552/60000 [====>.........................] - ETA: 1:26 - loss: 0.4102 - categorical_accuracy: 0.8720
11584/60000 [====>.........................] - ETA: 1:26 - loss: 0.4102 - categorical_accuracy: 0.8721
11616/60000 [====>.........................] - ETA: 1:26 - loss: 0.4101 - categorical_accuracy: 0.8721
11648/60000 [====>.........................] - ETA: 1:26 - loss: 0.4093 - categorical_accuracy: 0.8724
11680/60000 [====>.........................] - ETA: 1:25 - loss: 0.4090 - categorical_accuracy: 0.8725
11712/60000 [====>.........................] - ETA: 1:25 - loss: 0.4084 - categorical_accuracy: 0.8726
11744/60000 [====>.........................] - ETA: 1:25 - loss: 0.4078 - categorical_accuracy: 0.8729
11776/60000 [====>.........................] - ETA: 1:25 - loss: 0.4075 - categorical_accuracy: 0.8730
11808/60000 [====>.........................] - ETA: 1:25 - loss: 0.4069 - categorical_accuracy: 0.8731
11840/60000 [====>.........................] - ETA: 1:25 - loss: 0.4064 - categorical_accuracy: 0.8733
11872/60000 [====>.........................] - ETA: 1:25 - loss: 0.4058 - categorical_accuracy: 0.8735
11904/60000 [====>.........................] - ETA: 1:25 - loss: 0.4053 - categorical_accuracy: 0.8736
11936/60000 [====>.........................] - ETA: 1:25 - loss: 0.4053 - categorical_accuracy: 0.8735
11968/60000 [====>.........................] - ETA: 1:25 - loss: 0.4046 - categorical_accuracy: 0.8737
12000/60000 [=====>........................] - ETA: 1:25 - loss: 0.4047 - categorical_accuracy: 0.8736
12032/60000 [=====>........................] - ETA: 1:25 - loss: 0.4040 - categorical_accuracy: 0.8738
12064/60000 [=====>........................] - ETA: 1:25 - loss: 0.4036 - categorical_accuracy: 0.8738
12096/60000 [=====>........................] - ETA: 1:25 - loss: 0.4029 - categorical_accuracy: 0.8738
12128/60000 [=====>........................] - ETA: 1:25 - loss: 0.4038 - categorical_accuracy: 0.8738
12160/60000 [=====>........................] - ETA: 1:25 - loss: 0.4030 - categorical_accuracy: 0.8741
12192/60000 [=====>........................] - ETA: 1:24 - loss: 0.4021 - categorical_accuracy: 0.8744
12224/60000 [=====>........................] - ETA: 1:24 - loss: 0.4012 - categorical_accuracy: 0.8748
12256/60000 [=====>........................] - ETA: 1:24 - loss: 0.4008 - categorical_accuracy: 0.8749
12288/60000 [=====>........................] - ETA: 1:24 - loss: 0.4000 - categorical_accuracy: 0.8752
12320/60000 [=====>........................] - ETA: 1:24 - loss: 0.3994 - categorical_accuracy: 0.8753
12352/60000 [=====>........................] - ETA: 1:24 - loss: 0.3985 - categorical_accuracy: 0.8756
12384/60000 [=====>........................] - ETA: 1:24 - loss: 0.3979 - categorical_accuracy: 0.8759
12416/60000 [=====>........................] - ETA: 1:24 - loss: 0.3973 - categorical_accuracy: 0.8760
12448/60000 [=====>........................] - ETA: 1:24 - loss: 0.3967 - categorical_accuracy: 0.8761
12480/60000 [=====>........................] - ETA: 1:24 - loss: 0.3965 - categorical_accuracy: 0.8762
12512/60000 [=====>........................] - ETA: 1:24 - loss: 0.3959 - categorical_accuracy: 0.8764
12544/60000 [=====>........................] - ETA: 1:24 - loss: 0.3958 - categorical_accuracy: 0.8764
12576/60000 [=====>........................] - ETA: 1:24 - loss: 0.3952 - categorical_accuracy: 0.8764
12608/60000 [=====>........................] - ETA: 1:24 - loss: 0.3949 - categorical_accuracy: 0.8764
12640/60000 [=====>........................] - ETA: 1:24 - loss: 0.3942 - categorical_accuracy: 0.8766
12672/60000 [=====>........................] - ETA: 1:24 - loss: 0.3935 - categorical_accuracy: 0.8767
12704/60000 [=====>........................] - ETA: 1:23 - loss: 0.3939 - categorical_accuracy: 0.8767
12736/60000 [=====>........................] - ETA: 1:23 - loss: 0.3933 - categorical_accuracy: 0.8770
12768/60000 [=====>........................] - ETA: 1:23 - loss: 0.3925 - categorical_accuracy: 0.8772
12800/60000 [=====>........................] - ETA: 1:23 - loss: 0.3927 - categorical_accuracy: 0.8772
12832/60000 [=====>........................] - ETA: 1:23 - loss: 0.3920 - categorical_accuracy: 0.8774
12864/60000 [=====>........................] - ETA: 1:23 - loss: 0.3915 - categorical_accuracy: 0.8775
12896/60000 [=====>........................] - ETA: 1:23 - loss: 0.3909 - categorical_accuracy: 0.8777
12928/60000 [=====>........................] - ETA: 1:23 - loss: 0.3900 - categorical_accuracy: 0.8780
12960/60000 [=====>........................] - ETA: 1:23 - loss: 0.3896 - categorical_accuracy: 0.8781
12992/60000 [=====>........................] - ETA: 1:23 - loss: 0.3889 - categorical_accuracy: 0.8783
13024/60000 [=====>........................] - ETA: 1:23 - loss: 0.3882 - categorical_accuracy: 0.8785
13056/60000 [=====>........................] - ETA: 1:23 - loss: 0.3875 - categorical_accuracy: 0.8786
13088/60000 [=====>........................] - ETA: 1:23 - loss: 0.3872 - categorical_accuracy: 0.8787
13120/60000 [=====>........................] - ETA: 1:23 - loss: 0.3870 - categorical_accuracy: 0.8788
13152/60000 [=====>........................] - ETA: 1:23 - loss: 0.3868 - categorical_accuracy: 0.8787
13184/60000 [=====>........................] - ETA: 1:23 - loss: 0.3860 - categorical_accuracy: 0.8790
13216/60000 [=====>........................] - ETA: 1:23 - loss: 0.3853 - categorical_accuracy: 0.8792
13248/60000 [=====>........................] - ETA: 1:22 - loss: 0.3848 - categorical_accuracy: 0.8793
13280/60000 [=====>........................] - ETA: 1:22 - loss: 0.3843 - categorical_accuracy: 0.8795
13312/60000 [=====>........................] - ETA: 1:22 - loss: 0.3842 - categorical_accuracy: 0.8797
13344/60000 [=====>........................] - ETA: 1:22 - loss: 0.3836 - categorical_accuracy: 0.8799
13376/60000 [=====>........................] - ETA: 1:22 - loss: 0.3833 - categorical_accuracy: 0.8799
13408/60000 [=====>........................] - ETA: 1:22 - loss: 0.3828 - categorical_accuracy: 0.8801
13440/60000 [=====>........................] - ETA: 1:22 - loss: 0.3832 - categorical_accuracy: 0.8800
13472/60000 [=====>........................] - ETA: 1:22 - loss: 0.3826 - categorical_accuracy: 0.8802
13504/60000 [=====>........................] - ETA: 1:22 - loss: 0.3818 - categorical_accuracy: 0.8804
13536/60000 [=====>........................] - ETA: 1:22 - loss: 0.3810 - categorical_accuracy: 0.8806
13568/60000 [=====>........................] - ETA: 1:22 - loss: 0.3804 - categorical_accuracy: 0.8807
13600/60000 [=====>........................] - ETA: 1:22 - loss: 0.3796 - categorical_accuracy: 0.8810
13632/60000 [=====>........................] - ETA: 1:22 - loss: 0.3793 - categorical_accuracy: 0.8812
13664/60000 [=====>........................] - ETA: 1:22 - loss: 0.3787 - categorical_accuracy: 0.8814
13696/60000 [=====>........................] - ETA: 1:22 - loss: 0.3779 - categorical_accuracy: 0.8816
13728/60000 [=====>........................] - ETA: 1:22 - loss: 0.3775 - categorical_accuracy: 0.8818
13760/60000 [=====>........................] - ETA: 1:22 - loss: 0.3781 - categorical_accuracy: 0.8820
13792/60000 [=====>........................] - ETA: 1:21 - loss: 0.3775 - categorical_accuracy: 0.8821
13824/60000 [=====>........................] - ETA: 1:21 - loss: 0.3767 - categorical_accuracy: 0.8824
13856/60000 [=====>........................] - ETA: 1:21 - loss: 0.3761 - categorical_accuracy: 0.8826
13888/60000 [=====>........................] - ETA: 1:21 - loss: 0.3754 - categorical_accuracy: 0.8828
13920/60000 [=====>........................] - ETA: 1:21 - loss: 0.3751 - categorical_accuracy: 0.8830
13952/60000 [=====>........................] - ETA: 1:21 - loss: 0.3744 - categorical_accuracy: 0.8832
13984/60000 [=====>........................] - ETA: 1:21 - loss: 0.3742 - categorical_accuracy: 0.8834
14016/60000 [======>.......................] - ETA: 1:21 - loss: 0.3742 - categorical_accuracy: 0.8834
14048/60000 [======>.......................] - ETA: 1:21 - loss: 0.3737 - categorical_accuracy: 0.8835
14080/60000 [======>.......................] - ETA: 1:21 - loss: 0.3731 - categorical_accuracy: 0.8837
14112/60000 [======>.......................] - ETA: 1:21 - loss: 0.3730 - categorical_accuracy: 0.8838
14144/60000 [======>.......................] - ETA: 1:21 - loss: 0.3726 - categorical_accuracy: 0.8839
14176/60000 [======>.......................] - ETA: 1:21 - loss: 0.3718 - categorical_accuracy: 0.8842
14208/60000 [======>.......................] - ETA: 1:21 - loss: 0.3717 - categorical_accuracy: 0.8841
14240/60000 [======>.......................] - ETA: 1:21 - loss: 0.3713 - categorical_accuracy: 0.8843
14272/60000 [======>.......................] - ETA: 1:21 - loss: 0.3706 - categorical_accuracy: 0.8845
14304/60000 [======>.......................] - ETA: 1:21 - loss: 0.3701 - categorical_accuracy: 0.8847
14336/60000 [======>.......................] - ETA: 1:21 - loss: 0.3694 - categorical_accuracy: 0.8849
14368/60000 [======>.......................] - ETA: 1:20 - loss: 0.3688 - categorical_accuracy: 0.8851
14400/60000 [======>.......................] - ETA: 1:20 - loss: 0.3681 - categorical_accuracy: 0.8853
14432/60000 [======>.......................] - ETA: 1:20 - loss: 0.3679 - categorical_accuracy: 0.8854
14464/60000 [======>.......................] - ETA: 1:20 - loss: 0.3681 - categorical_accuracy: 0.8853
14496/60000 [======>.......................] - ETA: 1:20 - loss: 0.3679 - categorical_accuracy: 0.8853
14528/60000 [======>.......................] - ETA: 1:20 - loss: 0.3674 - categorical_accuracy: 0.8855
14560/60000 [======>.......................] - ETA: 1:20 - loss: 0.3668 - categorical_accuracy: 0.8857
14592/60000 [======>.......................] - ETA: 1:20 - loss: 0.3661 - categorical_accuracy: 0.8860
14624/60000 [======>.......................] - ETA: 1:20 - loss: 0.3658 - categorical_accuracy: 0.8861
14656/60000 [======>.......................] - ETA: 1:20 - loss: 0.3652 - categorical_accuracy: 0.8863
14688/60000 [======>.......................] - ETA: 1:20 - loss: 0.3646 - categorical_accuracy: 0.8866
14720/60000 [======>.......................] - ETA: 1:20 - loss: 0.3643 - categorical_accuracy: 0.8866
14752/60000 [======>.......................] - ETA: 1:20 - loss: 0.3639 - categorical_accuracy: 0.8867
14784/60000 [======>.......................] - ETA: 1:20 - loss: 0.3633 - categorical_accuracy: 0.8868
14816/60000 [======>.......................] - ETA: 1:20 - loss: 0.3636 - categorical_accuracy: 0.8868
14848/60000 [======>.......................] - ETA: 1:20 - loss: 0.3629 - categorical_accuracy: 0.8871
14880/60000 [======>.......................] - ETA: 1:19 - loss: 0.3625 - categorical_accuracy: 0.8872
14912/60000 [======>.......................] - ETA: 1:19 - loss: 0.3619 - categorical_accuracy: 0.8873
14944/60000 [======>.......................] - ETA: 1:19 - loss: 0.3615 - categorical_accuracy: 0.8875
14976/60000 [======>.......................] - ETA: 1:19 - loss: 0.3608 - categorical_accuracy: 0.8878
15008/60000 [======>.......................] - ETA: 1:19 - loss: 0.3601 - categorical_accuracy: 0.8880
15040/60000 [======>.......................] - ETA: 1:19 - loss: 0.3598 - categorical_accuracy: 0.8880
15072/60000 [======>.......................] - ETA: 1:19 - loss: 0.3592 - categorical_accuracy: 0.8883
15104/60000 [======>.......................] - ETA: 1:19 - loss: 0.3591 - categorical_accuracy: 0.8884
15136/60000 [======>.......................] - ETA: 1:19 - loss: 0.3585 - categorical_accuracy: 0.8885
15168/60000 [======>.......................] - ETA: 1:19 - loss: 0.3579 - categorical_accuracy: 0.8886
15200/60000 [======>.......................] - ETA: 1:19 - loss: 0.3579 - categorical_accuracy: 0.8886
15232/60000 [======>.......................] - ETA: 1:19 - loss: 0.3577 - categorical_accuracy: 0.8887
15264/60000 [======>.......................] - ETA: 1:19 - loss: 0.3576 - categorical_accuracy: 0.8888
15296/60000 [======>.......................] - ETA: 1:19 - loss: 0.3573 - categorical_accuracy: 0.8889
15328/60000 [======>.......................] - ETA: 1:19 - loss: 0.3568 - categorical_accuracy: 0.8891
15360/60000 [======>.......................] - ETA: 1:19 - loss: 0.3563 - categorical_accuracy: 0.8892
15392/60000 [======>.......................] - ETA: 1:19 - loss: 0.3557 - categorical_accuracy: 0.8894
15424/60000 [======>.......................] - ETA: 1:18 - loss: 0.3553 - categorical_accuracy: 0.8895
15456/60000 [======>.......................] - ETA: 1:18 - loss: 0.3554 - categorical_accuracy: 0.8895
15488/60000 [======>.......................] - ETA: 1:18 - loss: 0.3554 - categorical_accuracy: 0.8896
15520/60000 [======>.......................] - ETA: 1:18 - loss: 0.3553 - categorical_accuracy: 0.8896
15552/60000 [======>.......................] - ETA: 1:18 - loss: 0.3546 - categorical_accuracy: 0.8899
15584/60000 [======>.......................] - ETA: 1:18 - loss: 0.3541 - categorical_accuracy: 0.8901
15616/60000 [======>.......................] - ETA: 1:18 - loss: 0.3536 - categorical_accuracy: 0.8902
15648/60000 [======>.......................] - ETA: 1:18 - loss: 0.3543 - categorical_accuracy: 0.8902
15680/60000 [======>.......................] - ETA: 1:18 - loss: 0.3540 - categorical_accuracy: 0.8904
15712/60000 [======>.......................] - ETA: 1:18 - loss: 0.3537 - categorical_accuracy: 0.8905
15744/60000 [======>.......................] - ETA: 1:18 - loss: 0.3531 - categorical_accuracy: 0.8907
15776/60000 [======>.......................] - ETA: 1:18 - loss: 0.3525 - categorical_accuracy: 0.8909
15808/60000 [======>.......................] - ETA: 1:18 - loss: 0.3521 - categorical_accuracy: 0.8911
15840/60000 [======>.......................] - ETA: 1:18 - loss: 0.3518 - categorical_accuracy: 0.8912
15872/60000 [======>.......................] - ETA: 1:18 - loss: 0.3514 - categorical_accuracy: 0.8913
15904/60000 [======>.......................] - ETA: 1:18 - loss: 0.3511 - categorical_accuracy: 0.8914
15936/60000 [======>.......................] - ETA: 1:18 - loss: 0.3505 - categorical_accuracy: 0.8916
15968/60000 [======>.......................] - ETA: 1:17 - loss: 0.3504 - categorical_accuracy: 0.8916
16000/60000 [=======>......................] - ETA: 1:17 - loss: 0.3500 - categorical_accuracy: 0.8917
16032/60000 [=======>......................] - ETA: 1:17 - loss: 0.3499 - categorical_accuracy: 0.8917
16064/60000 [=======>......................] - ETA: 1:17 - loss: 0.3495 - categorical_accuracy: 0.8917
16096/60000 [=======>......................] - ETA: 1:17 - loss: 0.3491 - categorical_accuracy: 0.8918
16128/60000 [=======>......................] - ETA: 1:17 - loss: 0.3487 - categorical_accuracy: 0.8919
16160/60000 [=======>......................] - ETA: 1:17 - loss: 0.3482 - categorical_accuracy: 0.8921
16192/60000 [=======>......................] - ETA: 1:17 - loss: 0.3476 - categorical_accuracy: 0.8923
16224/60000 [=======>......................] - ETA: 1:17 - loss: 0.3471 - categorical_accuracy: 0.8924
16256/60000 [=======>......................] - ETA: 1:17 - loss: 0.3466 - categorical_accuracy: 0.8925
16288/60000 [=======>......................] - ETA: 1:17 - loss: 0.3461 - categorical_accuracy: 0.8926
16320/60000 [=======>......................] - ETA: 1:17 - loss: 0.3457 - categorical_accuracy: 0.8928
16352/60000 [=======>......................] - ETA: 1:17 - loss: 0.3452 - categorical_accuracy: 0.8929
16384/60000 [=======>......................] - ETA: 1:17 - loss: 0.3447 - categorical_accuracy: 0.8931
16416/60000 [=======>......................] - ETA: 1:17 - loss: 0.3445 - categorical_accuracy: 0.8933
16448/60000 [=======>......................] - ETA: 1:17 - loss: 0.3439 - categorical_accuracy: 0.8935
16480/60000 [=======>......................] - ETA: 1:17 - loss: 0.3435 - categorical_accuracy: 0.8936
16512/60000 [=======>......................] - ETA: 1:16 - loss: 0.3430 - categorical_accuracy: 0.8938
16544/60000 [=======>......................] - ETA: 1:16 - loss: 0.3426 - categorical_accuracy: 0.8939
16576/60000 [=======>......................] - ETA: 1:16 - loss: 0.3423 - categorical_accuracy: 0.8940
16608/60000 [=======>......................] - ETA: 1:16 - loss: 0.3419 - categorical_accuracy: 0.8941
16640/60000 [=======>......................] - ETA: 1:16 - loss: 0.3413 - categorical_accuracy: 0.8943
16672/60000 [=======>......................] - ETA: 1:16 - loss: 0.3407 - categorical_accuracy: 0.8945
16704/60000 [=======>......................] - ETA: 1:16 - loss: 0.3403 - categorical_accuracy: 0.8946
16736/60000 [=======>......................] - ETA: 1:16 - loss: 0.3398 - categorical_accuracy: 0.8948
16768/60000 [=======>......................] - ETA: 1:16 - loss: 0.3395 - categorical_accuracy: 0.8949
16800/60000 [=======>......................] - ETA: 1:16 - loss: 0.3390 - categorical_accuracy: 0.8951
16832/60000 [=======>......................] - ETA: 1:16 - loss: 0.3385 - categorical_accuracy: 0.8953
16864/60000 [=======>......................] - ETA: 1:16 - loss: 0.3382 - categorical_accuracy: 0.8954
16896/60000 [=======>......................] - ETA: 1:16 - loss: 0.3380 - categorical_accuracy: 0.8955
16928/60000 [=======>......................] - ETA: 1:16 - loss: 0.3376 - categorical_accuracy: 0.8956
16960/60000 [=======>......................] - ETA: 1:16 - loss: 0.3372 - categorical_accuracy: 0.8956
16992/60000 [=======>......................] - ETA: 1:16 - loss: 0.3366 - categorical_accuracy: 0.8958
17024/60000 [=======>......................] - ETA: 1:16 - loss: 0.3362 - categorical_accuracy: 0.8959
17056/60000 [=======>......................] - ETA: 1:15 - loss: 0.3360 - categorical_accuracy: 0.8960
17088/60000 [=======>......................] - ETA: 1:15 - loss: 0.3354 - categorical_accuracy: 0.8962
17120/60000 [=======>......................] - ETA: 1:15 - loss: 0.3350 - categorical_accuracy: 0.8963
17152/60000 [=======>......................] - ETA: 1:15 - loss: 0.3344 - categorical_accuracy: 0.8965
17184/60000 [=======>......................] - ETA: 1:15 - loss: 0.3339 - categorical_accuracy: 0.8966
17216/60000 [=======>......................] - ETA: 1:15 - loss: 0.3336 - categorical_accuracy: 0.8968
17248/60000 [=======>......................] - ETA: 1:15 - loss: 0.3332 - categorical_accuracy: 0.8969
17280/60000 [=======>......................] - ETA: 1:15 - loss: 0.3327 - categorical_accuracy: 0.8970
17312/60000 [=======>......................] - ETA: 1:15 - loss: 0.3327 - categorical_accuracy: 0.8971
17344/60000 [=======>......................] - ETA: 1:15 - loss: 0.3325 - categorical_accuracy: 0.8972
17376/60000 [=======>......................] - ETA: 1:15 - loss: 0.3320 - categorical_accuracy: 0.8974
17408/60000 [=======>......................] - ETA: 1:15 - loss: 0.3318 - categorical_accuracy: 0.8975
17440/60000 [=======>......................] - ETA: 1:15 - loss: 0.3314 - categorical_accuracy: 0.8975
17472/60000 [=======>......................] - ETA: 1:15 - loss: 0.3318 - categorical_accuracy: 0.8976
17504/60000 [=======>......................] - ETA: 1:15 - loss: 0.3313 - categorical_accuracy: 0.8976
17536/60000 [=======>......................] - ETA: 1:15 - loss: 0.3307 - categorical_accuracy: 0.8978
17568/60000 [=======>......................] - ETA: 1:15 - loss: 0.3306 - categorical_accuracy: 0.8978
17600/60000 [=======>......................] - ETA: 1:15 - loss: 0.3302 - categorical_accuracy: 0.8980
17632/60000 [=======>......................] - ETA: 1:14 - loss: 0.3307 - categorical_accuracy: 0.8980
17664/60000 [=======>......................] - ETA: 1:14 - loss: 0.3302 - categorical_accuracy: 0.8982
17696/60000 [=======>......................] - ETA: 1:14 - loss: 0.3309 - categorical_accuracy: 0.8980
17728/60000 [=======>......................] - ETA: 1:14 - loss: 0.3306 - categorical_accuracy: 0.8981
17760/60000 [=======>......................] - ETA: 1:14 - loss: 0.3307 - categorical_accuracy: 0.8980
17792/60000 [=======>......................] - ETA: 1:14 - loss: 0.3303 - categorical_accuracy: 0.8982
17824/60000 [=======>......................] - ETA: 1:14 - loss: 0.3301 - categorical_accuracy: 0.8982
17856/60000 [=======>......................] - ETA: 1:14 - loss: 0.3298 - categorical_accuracy: 0.8984
17888/60000 [=======>......................] - ETA: 1:14 - loss: 0.3294 - categorical_accuracy: 0.8985
17920/60000 [=======>......................] - ETA: 1:14 - loss: 0.3290 - categorical_accuracy: 0.8987
17952/60000 [=======>......................] - ETA: 1:14 - loss: 0.3285 - categorical_accuracy: 0.8988
17984/60000 [=======>......................] - ETA: 1:14 - loss: 0.3280 - categorical_accuracy: 0.8990
18016/60000 [========>.....................] - ETA: 1:14 - loss: 0.3275 - categorical_accuracy: 0.8991
18048/60000 [========>.....................] - ETA: 1:14 - loss: 0.3274 - categorical_accuracy: 0.8992
18080/60000 [========>.....................] - ETA: 1:14 - loss: 0.3271 - categorical_accuracy: 0.8992
18112/60000 [========>.....................] - ETA: 1:14 - loss: 0.3268 - categorical_accuracy: 0.8993
18144/60000 [========>.....................] - ETA: 1:14 - loss: 0.3267 - categorical_accuracy: 0.8994
18176/60000 [========>.....................] - ETA: 1:14 - loss: 0.3268 - categorical_accuracy: 0.8993
18208/60000 [========>.....................] - ETA: 1:13 - loss: 0.3264 - categorical_accuracy: 0.8994
18240/60000 [========>.....................] - ETA: 1:13 - loss: 0.3259 - categorical_accuracy: 0.8996
18272/60000 [========>.....................] - ETA: 1:13 - loss: 0.3255 - categorical_accuracy: 0.8998
18304/60000 [========>.....................] - ETA: 1:13 - loss: 0.3250 - categorical_accuracy: 0.9000
18336/60000 [========>.....................] - ETA: 1:13 - loss: 0.3247 - categorical_accuracy: 0.9001
18368/60000 [========>.....................] - ETA: 1:13 - loss: 0.3242 - categorical_accuracy: 0.9003
18400/60000 [========>.....................] - ETA: 1:13 - loss: 0.3238 - categorical_accuracy: 0.9004
18432/60000 [========>.....................] - ETA: 1:13 - loss: 0.3239 - categorical_accuracy: 0.9003
18464/60000 [========>.....................] - ETA: 1:13 - loss: 0.3238 - categorical_accuracy: 0.9003
18496/60000 [========>.....................] - ETA: 1:13 - loss: 0.3237 - categorical_accuracy: 0.9004
18528/60000 [========>.....................] - ETA: 1:13 - loss: 0.3235 - categorical_accuracy: 0.9004
18560/60000 [========>.....................] - ETA: 1:13 - loss: 0.3233 - categorical_accuracy: 0.9004
18592/60000 [========>.....................] - ETA: 1:13 - loss: 0.3229 - categorical_accuracy: 0.9005
18624/60000 [========>.....................] - ETA: 1:13 - loss: 0.3226 - categorical_accuracy: 0.9006
18656/60000 [========>.....................] - ETA: 1:13 - loss: 0.3226 - categorical_accuracy: 0.9006
18688/60000 [========>.....................] - ETA: 1:13 - loss: 0.3221 - categorical_accuracy: 0.9008
18720/60000 [========>.....................] - ETA: 1:12 - loss: 0.3218 - categorical_accuracy: 0.9009
18752/60000 [========>.....................] - ETA: 1:12 - loss: 0.3216 - categorical_accuracy: 0.9009
18784/60000 [========>.....................] - ETA: 1:12 - loss: 0.3211 - categorical_accuracy: 0.9011
18816/60000 [========>.....................] - ETA: 1:12 - loss: 0.3207 - categorical_accuracy: 0.9013
18848/60000 [========>.....................] - ETA: 1:12 - loss: 0.3204 - categorical_accuracy: 0.9014
18880/60000 [========>.....................] - ETA: 1:12 - loss: 0.3202 - categorical_accuracy: 0.9014
18912/60000 [========>.....................] - ETA: 1:12 - loss: 0.3202 - categorical_accuracy: 0.9014
18944/60000 [========>.....................] - ETA: 1:12 - loss: 0.3199 - categorical_accuracy: 0.9016
18976/60000 [========>.....................] - ETA: 1:12 - loss: 0.3195 - categorical_accuracy: 0.9017
19008/60000 [========>.....................] - ETA: 1:12 - loss: 0.3191 - categorical_accuracy: 0.9018
19040/60000 [========>.....................] - ETA: 1:12 - loss: 0.3187 - categorical_accuracy: 0.9019
19072/60000 [========>.....................] - ETA: 1:12 - loss: 0.3184 - categorical_accuracy: 0.9021
19104/60000 [========>.....................] - ETA: 1:12 - loss: 0.3183 - categorical_accuracy: 0.9021
19136/60000 [========>.....................] - ETA: 1:12 - loss: 0.3178 - categorical_accuracy: 0.9022
19168/60000 [========>.....................] - ETA: 1:12 - loss: 0.3175 - categorical_accuracy: 0.9023
19200/60000 [========>.....................] - ETA: 1:12 - loss: 0.3173 - categorical_accuracy: 0.9023
19232/60000 [========>.....................] - ETA: 1:12 - loss: 0.3171 - categorical_accuracy: 0.9024
19264/60000 [========>.....................] - ETA: 1:12 - loss: 0.3174 - categorical_accuracy: 0.9024
19296/60000 [========>.....................] - ETA: 1:11 - loss: 0.3173 - categorical_accuracy: 0.9023
19328/60000 [========>.....................] - ETA: 1:11 - loss: 0.3169 - categorical_accuracy: 0.9025
19360/60000 [========>.....................] - ETA: 1:11 - loss: 0.3165 - categorical_accuracy: 0.9026
19392/60000 [========>.....................] - ETA: 1:11 - loss: 0.3167 - categorical_accuracy: 0.9025
19424/60000 [========>.....................] - ETA: 1:11 - loss: 0.3163 - categorical_accuracy: 0.9026
19456/60000 [========>.....................] - ETA: 1:11 - loss: 0.3160 - categorical_accuracy: 0.9027
19488/60000 [========>.....................] - ETA: 1:11 - loss: 0.3156 - categorical_accuracy: 0.9028
19520/60000 [========>.....................] - ETA: 1:11 - loss: 0.3154 - categorical_accuracy: 0.9029
19552/60000 [========>.....................] - ETA: 1:11 - loss: 0.3152 - categorical_accuracy: 0.9029
19584/60000 [========>.....................] - ETA: 1:11 - loss: 0.3148 - categorical_accuracy: 0.9029
19616/60000 [========>.....................] - ETA: 1:11 - loss: 0.3150 - categorical_accuracy: 0.9029
19648/60000 [========>.....................] - ETA: 1:11 - loss: 0.3146 - categorical_accuracy: 0.9030
19680/60000 [========>.....................] - ETA: 1:11 - loss: 0.3142 - categorical_accuracy: 0.9032
19712/60000 [========>.....................] - ETA: 1:11 - loss: 0.3138 - categorical_accuracy: 0.9033
19744/60000 [========>.....................] - ETA: 1:11 - loss: 0.3137 - categorical_accuracy: 0.9034
19776/60000 [========>.....................] - ETA: 1:11 - loss: 0.3133 - categorical_accuracy: 0.9035
19808/60000 [========>.....................] - ETA: 1:11 - loss: 0.3131 - categorical_accuracy: 0.9036
19840/60000 [========>.....................] - ETA: 1:11 - loss: 0.3130 - categorical_accuracy: 0.9035
19872/60000 [========>.....................] - ETA: 1:10 - loss: 0.3127 - categorical_accuracy: 0.9036
19904/60000 [========>.....................] - ETA: 1:10 - loss: 0.3123 - categorical_accuracy: 0.9037
19936/60000 [========>.....................] - ETA: 1:10 - loss: 0.3124 - categorical_accuracy: 0.9037
19968/60000 [========>.....................] - ETA: 1:10 - loss: 0.3121 - categorical_accuracy: 0.9038
20000/60000 [=========>....................] - ETA: 1:10 - loss: 0.3119 - categorical_accuracy: 0.9037
20032/60000 [=========>....................] - ETA: 1:10 - loss: 0.3115 - categorical_accuracy: 0.9039
20064/60000 [=========>....................] - ETA: 1:10 - loss: 0.3114 - categorical_accuracy: 0.9039
20096/60000 [=========>....................] - ETA: 1:10 - loss: 0.3111 - categorical_accuracy: 0.9040
20128/60000 [=========>....................] - ETA: 1:10 - loss: 0.3107 - categorical_accuracy: 0.9041
20160/60000 [=========>....................] - ETA: 1:10 - loss: 0.3104 - categorical_accuracy: 0.9042
20192/60000 [=========>....................] - ETA: 1:10 - loss: 0.3100 - categorical_accuracy: 0.9043
20224/60000 [=========>....................] - ETA: 1:10 - loss: 0.3096 - categorical_accuracy: 0.9045
20256/60000 [=========>....................] - ETA: 1:10 - loss: 0.3094 - categorical_accuracy: 0.9045
20288/60000 [=========>....................] - ETA: 1:10 - loss: 0.3089 - categorical_accuracy: 0.9047
20320/60000 [=========>....................] - ETA: 1:10 - loss: 0.3085 - categorical_accuracy: 0.9048
20352/60000 [=========>....................] - ETA: 1:10 - loss: 0.3090 - categorical_accuracy: 0.9047
20384/60000 [=========>....................] - ETA: 1:09 - loss: 0.3087 - categorical_accuracy: 0.9048
20416/60000 [=========>....................] - ETA: 1:09 - loss: 0.3086 - categorical_accuracy: 0.9047
20448/60000 [=========>....................] - ETA: 1:09 - loss: 0.3086 - categorical_accuracy: 0.9047
20480/60000 [=========>....................] - ETA: 1:09 - loss: 0.3084 - categorical_accuracy: 0.9048
20512/60000 [=========>....................] - ETA: 1:09 - loss: 0.3081 - categorical_accuracy: 0.9049
20544/60000 [=========>....................] - ETA: 1:09 - loss: 0.3080 - categorical_accuracy: 0.9049
20576/60000 [=========>....................] - ETA: 1:09 - loss: 0.3076 - categorical_accuracy: 0.9050
20608/60000 [=========>....................] - ETA: 1:09 - loss: 0.3073 - categorical_accuracy: 0.9051
20640/60000 [=========>....................] - ETA: 1:09 - loss: 0.3072 - categorical_accuracy: 0.9051
20672/60000 [=========>....................] - ETA: 1:09 - loss: 0.3069 - categorical_accuracy: 0.9051
20704/60000 [=========>....................] - ETA: 1:09 - loss: 0.3066 - categorical_accuracy: 0.9052
20736/60000 [=========>....................] - ETA: 1:09 - loss: 0.3063 - categorical_accuracy: 0.9053
20768/60000 [=========>....................] - ETA: 1:09 - loss: 0.3061 - categorical_accuracy: 0.9053
20800/60000 [=========>....................] - ETA: 1:09 - loss: 0.3065 - categorical_accuracy: 0.9052
20832/60000 [=========>....................] - ETA: 1:09 - loss: 0.3065 - categorical_accuracy: 0.9052
20864/60000 [=========>....................] - ETA: 1:09 - loss: 0.3069 - categorical_accuracy: 0.9052
20896/60000 [=========>....................] - ETA: 1:09 - loss: 0.3066 - categorical_accuracy: 0.9053
20928/60000 [=========>....................] - ETA: 1:09 - loss: 0.3067 - categorical_accuracy: 0.9053
20960/60000 [=========>....................] - ETA: 1:08 - loss: 0.3063 - categorical_accuracy: 0.9055
20992/60000 [=========>....................] - ETA: 1:08 - loss: 0.3059 - categorical_accuracy: 0.9056
21024/60000 [=========>....................] - ETA: 1:08 - loss: 0.3056 - categorical_accuracy: 0.9057
21056/60000 [=========>....................] - ETA: 1:08 - loss: 0.3053 - categorical_accuracy: 0.9058
21088/60000 [=========>....................] - ETA: 1:08 - loss: 0.3049 - categorical_accuracy: 0.9059
21120/60000 [=========>....................] - ETA: 1:08 - loss: 0.3046 - categorical_accuracy: 0.9060
21152/60000 [=========>....................] - ETA: 1:08 - loss: 0.3042 - categorical_accuracy: 0.9061
21184/60000 [=========>....................] - ETA: 1:08 - loss: 0.3038 - categorical_accuracy: 0.9062
21216/60000 [=========>....................] - ETA: 1:08 - loss: 0.3037 - categorical_accuracy: 0.9062
21248/60000 [=========>....................] - ETA: 1:08 - loss: 0.3034 - categorical_accuracy: 0.9063
21280/60000 [=========>....................] - ETA: 1:08 - loss: 0.3030 - categorical_accuracy: 0.9064
21312/60000 [=========>....................] - ETA: 1:08 - loss: 0.3033 - categorical_accuracy: 0.9063
21344/60000 [=========>....................] - ETA: 1:08 - loss: 0.3030 - categorical_accuracy: 0.9064
21376/60000 [=========>....................] - ETA: 1:08 - loss: 0.3027 - categorical_accuracy: 0.9065
21408/60000 [=========>....................] - ETA: 1:08 - loss: 0.3024 - categorical_accuracy: 0.9066
21440/60000 [=========>....................] - ETA: 1:08 - loss: 0.3020 - categorical_accuracy: 0.9067
21472/60000 [=========>....................] - ETA: 1:08 - loss: 0.3019 - categorical_accuracy: 0.9068
21504/60000 [=========>....................] - ETA: 1:08 - loss: 0.3016 - categorical_accuracy: 0.9069
21536/60000 [=========>....................] - ETA: 1:07 - loss: 0.3012 - categorical_accuracy: 0.9070
21568/60000 [=========>....................] - ETA: 1:07 - loss: 0.3008 - categorical_accuracy: 0.9071
21600/60000 [=========>....................] - ETA: 1:07 - loss: 0.3005 - categorical_accuracy: 0.9073
21632/60000 [=========>....................] - ETA: 1:07 - loss: 0.3000 - categorical_accuracy: 0.9074
21664/60000 [=========>....................] - ETA: 1:07 - loss: 0.3002 - categorical_accuracy: 0.9074
21696/60000 [=========>....................] - ETA: 1:07 - loss: 0.3000 - categorical_accuracy: 0.9074
21728/60000 [=========>....................] - ETA: 1:07 - loss: 0.3001 - categorical_accuracy: 0.9074
21760/60000 [=========>....................] - ETA: 1:07 - loss: 0.2999 - categorical_accuracy: 0.9074
21792/60000 [=========>....................] - ETA: 1:07 - loss: 0.2995 - categorical_accuracy: 0.9076
21824/60000 [=========>....................] - ETA: 1:07 - loss: 0.2993 - categorical_accuracy: 0.9077
21856/60000 [=========>....................] - ETA: 1:07 - loss: 0.2990 - categorical_accuracy: 0.9077
21888/60000 [=========>....................] - ETA: 1:07 - loss: 0.2987 - categorical_accuracy: 0.9078
21920/60000 [=========>....................] - ETA: 1:07 - loss: 0.2983 - categorical_accuracy: 0.9079
21952/60000 [=========>....................] - ETA: 1:07 - loss: 0.2980 - categorical_accuracy: 0.9081
21984/60000 [=========>....................] - ETA: 1:07 - loss: 0.2976 - categorical_accuracy: 0.9082
22016/60000 [==========>...................] - ETA: 1:07 - loss: 0.2974 - categorical_accuracy: 0.9082
22048/60000 [==========>...................] - ETA: 1:07 - loss: 0.2973 - categorical_accuracy: 0.9083
22080/60000 [==========>...................] - ETA: 1:06 - loss: 0.2971 - categorical_accuracy: 0.9084
22112/60000 [==========>...................] - ETA: 1:06 - loss: 0.2968 - categorical_accuracy: 0.9085
22144/60000 [==========>...................] - ETA: 1:06 - loss: 0.2965 - categorical_accuracy: 0.9086
22176/60000 [==========>...................] - ETA: 1:06 - loss: 0.2961 - categorical_accuracy: 0.9087
22208/60000 [==========>...................] - ETA: 1:06 - loss: 0.2963 - categorical_accuracy: 0.9088
22240/60000 [==========>...................] - ETA: 1:06 - loss: 0.2959 - categorical_accuracy: 0.9089
22272/60000 [==========>...................] - ETA: 1:06 - loss: 0.2955 - categorical_accuracy: 0.9090
22304/60000 [==========>...................] - ETA: 1:06 - loss: 0.2956 - categorical_accuracy: 0.9091
22336/60000 [==========>...................] - ETA: 1:06 - loss: 0.2953 - categorical_accuracy: 0.9092
22368/60000 [==========>...................] - ETA: 1:06 - loss: 0.2954 - categorical_accuracy: 0.9091
22400/60000 [==========>...................] - ETA: 1:06 - loss: 0.2951 - categorical_accuracy: 0.9092
22432/60000 [==========>...................] - ETA: 1:06 - loss: 0.2951 - categorical_accuracy: 0.9092
22464/60000 [==========>...................] - ETA: 1:06 - loss: 0.2948 - categorical_accuracy: 0.9093
22496/60000 [==========>...................] - ETA: 1:06 - loss: 0.2945 - categorical_accuracy: 0.9094
22528/60000 [==========>...................] - ETA: 1:06 - loss: 0.2944 - categorical_accuracy: 0.9094
22560/60000 [==========>...................] - ETA: 1:06 - loss: 0.2943 - categorical_accuracy: 0.9095
22592/60000 [==========>...................] - ETA: 1:05 - loss: 0.2939 - categorical_accuracy: 0.9096
22624/60000 [==========>...................] - ETA: 1:05 - loss: 0.2936 - categorical_accuracy: 0.9097
22656/60000 [==========>...................] - ETA: 1:05 - loss: 0.2938 - categorical_accuracy: 0.9096
22688/60000 [==========>...................] - ETA: 1:05 - loss: 0.2934 - categorical_accuracy: 0.9098
22720/60000 [==========>...................] - ETA: 1:05 - loss: 0.2937 - categorical_accuracy: 0.9098
22752/60000 [==========>...................] - ETA: 1:05 - loss: 0.2933 - categorical_accuracy: 0.9099
22784/60000 [==========>...................] - ETA: 1:05 - loss: 0.2931 - categorical_accuracy: 0.9100
22816/60000 [==========>...................] - ETA: 1:05 - loss: 0.2929 - categorical_accuracy: 0.9101
22848/60000 [==========>...................] - ETA: 1:05 - loss: 0.2928 - categorical_accuracy: 0.9101
22880/60000 [==========>...................] - ETA: 1:05 - loss: 0.2928 - categorical_accuracy: 0.9101
22912/60000 [==========>...................] - ETA: 1:05 - loss: 0.2928 - categorical_accuracy: 0.9100
22944/60000 [==========>...................] - ETA: 1:05 - loss: 0.2925 - categorical_accuracy: 0.9101
22976/60000 [==========>...................] - ETA: 1:05 - loss: 0.2922 - categorical_accuracy: 0.9102
23008/60000 [==========>...................] - ETA: 1:05 - loss: 0.2920 - categorical_accuracy: 0.9102
23040/60000 [==========>...................] - ETA: 1:05 - loss: 0.2918 - categorical_accuracy: 0.9103
23072/60000 [==========>...................] - ETA: 1:05 - loss: 0.2915 - categorical_accuracy: 0.9104
23104/60000 [==========>...................] - ETA: 1:05 - loss: 0.2914 - categorical_accuracy: 0.9104
23136/60000 [==========>...................] - ETA: 1:05 - loss: 0.2915 - categorical_accuracy: 0.9104
23168/60000 [==========>...................] - ETA: 1:04 - loss: 0.2913 - categorical_accuracy: 0.9104
23200/60000 [==========>...................] - ETA: 1:04 - loss: 0.2910 - categorical_accuracy: 0.9105
23232/60000 [==========>...................] - ETA: 1:04 - loss: 0.2907 - categorical_accuracy: 0.9106
23264/60000 [==========>...................] - ETA: 1:04 - loss: 0.2907 - categorical_accuracy: 0.9106
23296/60000 [==========>...................] - ETA: 1:04 - loss: 0.2906 - categorical_accuracy: 0.9107
23328/60000 [==========>...................] - ETA: 1:04 - loss: 0.2903 - categorical_accuracy: 0.9108
23360/60000 [==========>...................] - ETA: 1:04 - loss: 0.2901 - categorical_accuracy: 0.9108
23392/60000 [==========>...................] - ETA: 1:04 - loss: 0.2898 - categorical_accuracy: 0.9109
23424/60000 [==========>...................] - ETA: 1:04 - loss: 0.2895 - categorical_accuracy: 0.9110
23456/60000 [==========>...................] - ETA: 1:04 - loss: 0.2894 - categorical_accuracy: 0.9110
23488/60000 [==========>...................] - ETA: 1:04 - loss: 0.2893 - categorical_accuracy: 0.9110
23520/60000 [==========>...................] - ETA: 1:04 - loss: 0.2890 - categorical_accuracy: 0.9111
23552/60000 [==========>...................] - ETA: 1:04 - loss: 0.2887 - categorical_accuracy: 0.9112
23584/60000 [==========>...................] - ETA: 1:04 - loss: 0.2884 - categorical_accuracy: 0.9113
23616/60000 [==========>...................] - ETA: 1:04 - loss: 0.2881 - categorical_accuracy: 0.9114
23648/60000 [==========>...................] - ETA: 1:04 - loss: 0.2878 - categorical_accuracy: 0.9115
23680/60000 [==========>...................] - ETA: 1:04 - loss: 0.2877 - categorical_accuracy: 0.9115
23712/60000 [==========>...................] - ETA: 1:04 - loss: 0.2877 - categorical_accuracy: 0.9116
23744/60000 [==========>...................] - ETA: 1:03 - loss: 0.2874 - categorical_accuracy: 0.9116
23776/60000 [==========>...................] - ETA: 1:03 - loss: 0.2873 - categorical_accuracy: 0.9117
23808/60000 [==========>...................] - ETA: 1:03 - loss: 0.2870 - categorical_accuracy: 0.9118
23840/60000 [==========>...................] - ETA: 1:03 - loss: 0.2868 - categorical_accuracy: 0.9119
23872/60000 [==========>...................] - ETA: 1:03 - loss: 0.2865 - categorical_accuracy: 0.9119
23904/60000 [==========>...................] - ETA: 1:03 - loss: 0.2862 - categorical_accuracy: 0.9119
23936/60000 [==========>...................] - ETA: 1:03 - loss: 0.2860 - categorical_accuracy: 0.9120
23968/60000 [==========>...................] - ETA: 1:03 - loss: 0.2858 - categorical_accuracy: 0.9121
24000/60000 [===========>..................] - ETA: 1:03 - loss: 0.2854 - categorical_accuracy: 0.9122
24032/60000 [===========>..................] - ETA: 1:03 - loss: 0.2853 - categorical_accuracy: 0.9122
24064/60000 [===========>..................] - ETA: 1:03 - loss: 0.2850 - categorical_accuracy: 0.9123
24096/60000 [===========>..................] - ETA: 1:03 - loss: 0.2851 - categorical_accuracy: 0.9124
24128/60000 [===========>..................] - ETA: 1:03 - loss: 0.2849 - categorical_accuracy: 0.9124
24160/60000 [===========>..................] - ETA: 1:03 - loss: 0.2847 - categorical_accuracy: 0.9125
24192/60000 [===========>..................] - ETA: 1:03 - loss: 0.2848 - categorical_accuracy: 0.9125
24224/60000 [===========>..................] - ETA: 1:03 - loss: 0.2847 - categorical_accuracy: 0.9125
24256/60000 [===========>..................] - ETA: 1:03 - loss: 0.2845 - categorical_accuracy: 0.9126
24288/60000 [===========>..................] - ETA: 1:03 - loss: 0.2842 - categorical_accuracy: 0.9127
24320/60000 [===========>..................] - ETA: 1:02 - loss: 0.2840 - categorical_accuracy: 0.9127
24352/60000 [===========>..................] - ETA: 1:02 - loss: 0.2840 - categorical_accuracy: 0.9127
24384/60000 [===========>..................] - ETA: 1:02 - loss: 0.2838 - categorical_accuracy: 0.9127
24416/60000 [===========>..................] - ETA: 1:02 - loss: 0.2837 - categorical_accuracy: 0.9128
24448/60000 [===========>..................] - ETA: 1:02 - loss: 0.2833 - categorical_accuracy: 0.9129
24480/60000 [===========>..................] - ETA: 1:02 - loss: 0.2831 - categorical_accuracy: 0.9129
24512/60000 [===========>..................] - ETA: 1:02 - loss: 0.2829 - categorical_accuracy: 0.9129
24544/60000 [===========>..................] - ETA: 1:02 - loss: 0.2828 - categorical_accuracy: 0.9130
24576/60000 [===========>..................] - ETA: 1:02 - loss: 0.2825 - categorical_accuracy: 0.9130
24608/60000 [===========>..................] - ETA: 1:02 - loss: 0.2825 - categorical_accuracy: 0.9130
24640/60000 [===========>..................] - ETA: 1:02 - loss: 0.2822 - categorical_accuracy: 0.9131
24672/60000 [===========>..................] - ETA: 1:02 - loss: 0.2819 - categorical_accuracy: 0.9132
24704/60000 [===========>..................] - ETA: 1:02 - loss: 0.2820 - categorical_accuracy: 0.9131
24736/60000 [===========>..................] - ETA: 1:02 - loss: 0.2819 - categorical_accuracy: 0.9131
24768/60000 [===========>..................] - ETA: 1:02 - loss: 0.2816 - categorical_accuracy: 0.9132
24800/60000 [===========>..................] - ETA: 1:02 - loss: 0.2815 - categorical_accuracy: 0.9133
24832/60000 [===========>..................] - ETA: 1:02 - loss: 0.2814 - categorical_accuracy: 0.9133
24864/60000 [===========>..................] - ETA: 1:02 - loss: 0.2812 - categorical_accuracy: 0.9133
24896/60000 [===========>..................] - ETA: 1:01 - loss: 0.2810 - categorical_accuracy: 0.9134
24928/60000 [===========>..................] - ETA: 1:01 - loss: 0.2808 - categorical_accuracy: 0.9135
24960/60000 [===========>..................] - ETA: 1:01 - loss: 0.2805 - categorical_accuracy: 0.9136
24992/60000 [===========>..................] - ETA: 1:01 - loss: 0.2805 - categorical_accuracy: 0.9136
25024/60000 [===========>..................] - ETA: 1:01 - loss: 0.2802 - categorical_accuracy: 0.9136
25056/60000 [===========>..................] - ETA: 1:01 - loss: 0.2803 - categorical_accuracy: 0.9136
25088/60000 [===========>..................] - ETA: 1:01 - loss: 0.2799 - categorical_accuracy: 0.9137
25120/60000 [===========>..................] - ETA: 1:01 - loss: 0.2798 - categorical_accuracy: 0.9138
25152/60000 [===========>..................] - ETA: 1:01 - loss: 0.2795 - categorical_accuracy: 0.9139
25184/60000 [===========>..................] - ETA: 1:01 - loss: 0.2794 - categorical_accuracy: 0.9139
25216/60000 [===========>..................] - ETA: 1:01 - loss: 0.2791 - categorical_accuracy: 0.9140
25248/60000 [===========>..................] - ETA: 1:01 - loss: 0.2792 - categorical_accuracy: 0.9139
25280/60000 [===========>..................] - ETA: 1:01 - loss: 0.2789 - categorical_accuracy: 0.9140
25312/60000 [===========>..................] - ETA: 1:01 - loss: 0.2786 - categorical_accuracy: 0.9141
25344/60000 [===========>..................] - ETA: 1:01 - loss: 0.2784 - categorical_accuracy: 0.9142
25376/60000 [===========>..................] - ETA: 1:01 - loss: 0.2784 - categorical_accuracy: 0.9142
25408/60000 [===========>..................] - ETA: 1:01 - loss: 0.2782 - categorical_accuracy: 0.9143
25440/60000 [===========>..................] - ETA: 1:01 - loss: 0.2779 - categorical_accuracy: 0.9143
25472/60000 [===========>..................] - ETA: 1:00 - loss: 0.2779 - categorical_accuracy: 0.9143
25504/60000 [===========>..................] - ETA: 1:00 - loss: 0.2776 - categorical_accuracy: 0.9144
25536/60000 [===========>..................] - ETA: 1:00 - loss: 0.2774 - categorical_accuracy: 0.9144
25568/60000 [===========>..................] - ETA: 1:00 - loss: 0.2772 - categorical_accuracy: 0.9145
25600/60000 [===========>..................] - ETA: 1:00 - loss: 0.2771 - categorical_accuracy: 0.9145
25632/60000 [===========>..................] - ETA: 1:00 - loss: 0.2768 - categorical_accuracy: 0.9146
25664/60000 [===========>..................] - ETA: 1:00 - loss: 0.2768 - categorical_accuracy: 0.9147
25696/60000 [===========>..................] - ETA: 1:00 - loss: 0.2767 - categorical_accuracy: 0.9147
25728/60000 [===========>..................] - ETA: 1:00 - loss: 0.2764 - categorical_accuracy: 0.9148
25760/60000 [===========>..................] - ETA: 1:00 - loss: 0.2762 - categorical_accuracy: 0.9148
25792/60000 [===========>..................] - ETA: 1:00 - loss: 0.2760 - categorical_accuracy: 0.9149
25824/60000 [===========>..................] - ETA: 1:00 - loss: 0.2758 - categorical_accuracy: 0.9149
25856/60000 [===========>..................] - ETA: 1:00 - loss: 0.2756 - categorical_accuracy: 0.9150
25888/60000 [===========>..................] - ETA: 1:00 - loss: 0.2753 - categorical_accuracy: 0.9151
25920/60000 [===========>..................] - ETA: 1:00 - loss: 0.2750 - categorical_accuracy: 0.9152
25952/60000 [===========>..................] - ETA: 1:00 - loss: 0.2748 - categorical_accuracy: 0.9152
25984/60000 [===========>..................] - ETA: 1:00 - loss: 0.2745 - categorical_accuracy: 0.9153
26016/60000 [============>.................] - ETA: 59s - loss: 0.2747 - categorical_accuracy: 0.9153 
26048/60000 [============>.................] - ETA: 59s - loss: 0.2746 - categorical_accuracy: 0.9153
26080/60000 [============>.................] - ETA: 59s - loss: 0.2743 - categorical_accuracy: 0.9154
26112/60000 [============>.................] - ETA: 59s - loss: 0.2743 - categorical_accuracy: 0.9154
26144/60000 [============>.................] - ETA: 59s - loss: 0.2740 - categorical_accuracy: 0.9155
26176/60000 [============>.................] - ETA: 59s - loss: 0.2739 - categorical_accuracy: 0.9155
26208/60000 [============>.................] - ETA: 59s - loss: 0.2740 - categorical_accuracy: 0.9155
26240/60000 [============>.................] - ETA: 59s - loss: 0.2741 - categorical_accuracy: 0.9155
26272/60000 [============>.................] - ETA: 59s - loss: 0.2739 - categorical_accuracy: 0.9155
26304/60000 [============>.................] - ETA: 59s - loss: 0.2736 - categorical_accuracy: 0.9156
26336/60000 [============>.................] - ETA: 59s - loss: 0.2733 - categorical_accuracy: 0.9157
26368/60000 [============>.................] - ETA: 59s - loss: 0.2730 - categorical_accuracy: 0.9158
26400/60000 [============>.................] - ETA: 59s - loss: 0.2729 - categorical_accuracy: 0.9158
26432/60000 [============>.................] - ETA: 59s - loss: 0.2728 - categorical_accuracy: 0.9158
26464/60000 [============>.................] - ETA: 59s - loss: 0.2730 - categorical_accuracy: 0.9158
26496/60000 [============>.................] - ETA: 59s - loss: 0.2727 - categorical_accuracy: 0.9159
26528/60000 [============>.................] - ETA: 59s - loss: 0.2729 - categorical_accuracy: 0.9159
26560/60000 [============>.................] - ETA: 59s - loss: 0.2727 - categorical_accuracy: 0.9160
26592/60000 [============>.................] - ETA: 58s - loss: 0.2725 - categorical_accuracy: 0.9160
26624/60000 [============>.................] - ETA: 58s - loss: 0.2722 - categorical_accuracy: 0.9161
26656/60000 [============>.................] - ETA: 58s - loss: 0.2720 - categorical_accuracy: 0.9162
26688/60000 [============>.................] - ETA: 58s - loss: 0.2720 - categorical_accuracy: 0.9162
26720/60000 [============>.................] - ETA: 58s - loss: 0.2721 - categorical_accuracy: 0.9161
26752/60000 [============>.................] - ETA: 58s - loss: 0.2718 - categorical_accuracy: 0.9162
26784/60000 [============>.................] - ETA: 58s - loss: 0.2715 - categorical_accuracy: 0.9163
26816/60000 [============>.................] - ETA: 58s - loss: 0.2713 - categorical_accuracy: 0.9164
26848/60000 [============>.................] - ETA: 58s - loss: 0.2711 - categorical_accuracy: 0.9165
26880/60000 [============>.................] - ETA: 58s - loss: 0.2708 - categorical_accuracy: 0.9166
26912/60000 [============>.................] - ETA: 58s - loss: 0.2705 - categorical_accuracy: 0.9167
26944/60000 [============>.................] - ETA: 58s - loss: 0.2705 - categorical_accuracy: 0.9166
26976/60000 [============>.................] - ETA: 58s - loss: 0.2708 - categorical_accuracy: 0.9166
27008/60000 [============>.................] - ETA: 58s - loss: 0.2706 - categorical_accuracy: 0.9166
27040/60000 [============>.................] - ETA: 58s - loss: 0.2704 - categorical_accuracy: 0.9166
27072/60000 [============>.................] - ETA: 58s - loss: 0.2703 - categorical_accuracy: 0.9167
27104/60000 [============>.................] - ETA: 58s - loss: 0.2701 - categorical_accuracy: 0.9168
27136/60000 [============>.................] - ETA: 58s - loss: 0.2699 - categorical_accuracy: 0.9168
27168/60000 [============>.................] - ETA: 57s - loss: 0.2697 - categorical_accuracy: 0.9169
27200/60000 [============>.................] - ETA: 57s - loss: 0.2694 - categorical_accuracy: 0.9169
27232/60000 [============>.................] - ETA: 57s - loss: 0.2694 - categorical_accuracy: 0.9170
27264/60000 [============>.................] - ETA: 57s - loss: 0.2692 - categorical_accuracy: 0.9170
27296/60000 [============>.................] - ETA: 57s - loss: 0.2689 - categorical_accuracy: 0.9171
27328/60000 [============>.................] - ETA: 57s - loss: 0.2688 - categorical_accuracy: 0.9171
27360/60000 [============>.................] - ETA: 57s - loss: 0.2688 - categorical_accuracy: 0.9170
27392/60000 [============>.................] - ETA: 57s - loss: 0.2687 - categorical_accuracy: 0.9171
27424/60000 [============>.................] - ETA: 57s - loss: 0.2686 - categorical_accuracy: 0.9170
27456/60000 [============>.................] - ETA: 57s - loss: 0.2684 - categorical_accuracy: 0.9170
27488/60000 [============>.................] - ETA: 57s - loss: 0.2683 - categorical_accuracy: 0.9171
27520/60000 [============>.................] - ETA: 57s - loss: 0.2681 - categorical_accuracy: 0.9172
27552/60000 [============>.................] - ETA: 57s - loss: 0.2680 - categorical_accuracy: 0.9171
27584/60000 [============>.................] - ETA: 57s - loss: 0.2678 - categorical_accuracy: 0.9172
27616/60000 [============>.................] - ETA: 57s - loss: 0.2675 - categorical_accuracy: 0.9173
27648/60000 [============>.................] - ETA: 57s - loss: 0.2675 - categorical_accuracy: 0.9174
27680/60000 [============>.................] - ETA: 57s - loss: 0.2673 - categorical_accuracy: 0.9174
27712/60000 [============>.................] - ETA: 57s - loss: 0.2670 - categorical_accuracy: 0.9175
27744/60000 [============>.................] - ETA: 56s - loss: 0.2668 - categorical_accuracy: 0.9176
27776/60000 [============>.................] - ETA: 56s - loss: 0.2667 - categorical_accuracy: 0.9176
27808/60000 [============>.................] - ETA: 56s - loss: 0.2664 - categorical_accuracy: 0.9177
27840/60000 [============>.................] - ETA: 56s - loss: 0.2663 - categorical_accuracy: 0.9177
27872/60000 [============>.................] - ETA: 56s - loss: 0.2661 - categorical_accuracy: 0.9178
27904/60000 [============>.................] - ETA: 56s - loss: 0.2659 - categorical_accuracy: 0.9178
27936/60000 [============>.................] - ETA: 56s - loss: 0.2660 - categorical_accuracy: 0.9178
27968/60000 [============>.................] - ETA: 56s - loss: 0.2660 - categorical_accuracy: 0.9178
28000/60000 [=============>................] - ETA: 56s - loss: 0.2658 - categorical_accuracy: 0.9179
28032/60000 [=============>................] - ETA: 56s - loss: 0.2655 - categorical_accuracy: 0.9180
28064/60000 [=============>................] - ETA: 56s - loss: 0.2653 - categorical_accuracy: 0.9180
28096/60000 [=============>................] - ETA: 56s - loss: 0.2650 - categorical_accuracy: 0.9181
28128/60000 [=============>................] - ETA: 56s - loss: 0.2648 - categorical_accuracy: 0.9182
28160/60000 [=============>................] - ETA: 56s - loss: 0.2645 - categorical_accuracy: 0.9183
28192/60000 [=============>................] - ETA: 56s - loss: 0.2643 - categorical_accuracy: 0.9183
28224/60000 [=============>................] - ETA: 56s - loss: 0.2643 - categorical_accuracy: 0.9184
28256/60000 [=============>................] - ETA: 56s - loss: 0.2641 - categorical_accuracy: 0.9185
28288/60000 [=============>................] - ETA: 55s - loss: 0.2638 - categorical_accuracy: 0.9186
28320/60000 [=============>................] - ETA: 55s - loss: 0.2635 - categorical_accuracy: 0.9186
28352/60000 [=============>................] - ETA: 55s - loss: 0.2634 - categorical_accuracy: 0.9187
28384/60000 [=============>................] - ETA: 55s - loss: 0.2632 - categorical_accuracy: 0.9188
28416/60000 [=============>................] - ETA: 55s - loss: 0.2632 - categorical_accuracy: 0.9188
28448/60000 [=============>................] - ETA: 55s - loss: 0.2629 - categorical_accuracy: 0.9189
28480/60000 [=============>................] - ETA: 55s - loss: 0.2626 - categorical_accuracy: 0.9190
28512/60000 [=============>................] - ETA: 55s - loss: 0.2625 - categorical_accuracy: 0.9190
28544/60000 [=============>................] - ETA: 55s - loss: 0.2625 - categorical_accuracy: 0.9191
28576/60000 [=============>................] - ETA: 55s - loss: 0.2623 - categorical_accuracy: 0.9192
28608/60000 [=============>................] - ETA: 55s - loss: 0.2620 - categorical_accuracy: 0.9193
28640/60000 [=============>................] - ETA: 55s - loss: 0.2619 - categorical_accuracy: 0.9193
28672/60000 [=============>................] - ETA: 55s - loss: 0.2617 - categorical_accuracy: 0.9193
28704/60000 [=============>................] - ETA: 55s - loss: 0.2616 - categorical_accuracy: 0.9193
28736/60000 [=============>................] - ETA: 55s - loss: 0.2617 - categorical_accuracy: 0.9193
28768/60000 [=============>................] - ETA: 55s - loss: 0.2614 - categorical_accuracy: 0.9194
28800/60000 [=============>................] - ETA: 55s - loss: 0.2611 - categorical_accuracy: 0.9195
28832/60000 [=============>................] - ETA: 55s - loss: 0.2610 - categorical_accuracy: 0.9195
28864/60000 [=============>................] - ETA: 54s - loss: 0.2609 - categorical_accuracy: 0.9196
28896/60000 [=============>................] - ETA: 54s - loss: 0.2607 - categorical_accuracy: 0.9196
28928/60000 [=============>................] - ETA: 54s - loss: 0.2604 - categorical_accuracy: 0.9197
28960/60000 [=============>................] - ETA: 54s - loss: 0.2602 - categorical_accuracy: 0.9198
28992/60000 [=============>................] - ETA: 54s - loss: 0.2600 - categorical_accuracy: 0.9199
29024/60000 [=============>................] - ETA: 54s - loss: 0.2597 - categorical_accuracy: 0.9200
29056/60000 [=============>................] - ETA: 54s - loss: 0.2596 - categorical_accuracy: 0.9200
29088/60000 [=============>................] - ETA: 54s - loss: 0.2595 - categorical_accuracy: 0.9201
29120/60000 [=============>................] - ETA: 54s - loss: 0.2595 - categorical_accuracy: 0.9201
29152/60000 [=============>................] - ETA: 54s - loss: 0.2593 - categorical_accuracy: 0.9201
29184/60000 [=============>................] - ETA: 54s - loss: 0.2590 - categorical_accuracy: 0.9202
29216/60000 [=============>................] - ETA: 54s - loss: 0.2589 - categorical_accuracy: 0.9202
29248/60000 [=============>................] - ETA: 54s - loss: 0.2587 - categorical_accuracy: 0.9202
29280/60000 [=============>................] - ETA: 54s - loss: 0.2585 - categorical_accuracy: 0.9203
29312/60000 [=============>................] - ETA: 54s - loss: 0.2590 - categorical_accuracy: 0.9203
29344/60000 [=============>................] - ETA: 54s - loss: 0.2590 - categorical_accuracy: 0.9203
29376/60000 [=============>................] - ETA: 54s - loss: 0.2589 - categorical_accuracy: 0.9203
29408/60000 [=============>................] - ETA: 54s - loss: 0.2587 - categorical_accuracy: 0.9204
29440/60000 [=============>................] - ETA: 53s - loss: 0.2587 - categorical_accuracy: 0.9204
29472/60000 [=============>................] - ETA: 53s - loss: 0.2585 - categorical_accuracy: 0.9205
29504/60000 [=============>................] - ETA: 53s - loss: 0.2584 - categorical_accuracy: 0.9206
29536/60000 [=============>................] - ETA: 53s - loss: 0.2582 - categorical_accuracy: 0.9206
29568/60000 [=============>................] - ETA: 53s - loss: 0.2580 - categorical_accuracy: 0.9207
29600/60000 [=============>................] - ETA: 53s - loss: 0.2578 - categorical_accuracy: 0.9207
29632/60000 [=============>................] - ETA: 53s - loss: 0.2577 - categorical_accuracy: 0.9207
29664/60000 [=============>................] - ETA: 53s - loss: 0.2578 - categorical_accuracy: 0.9207
29696/60000 [=============>................] - ETA: 53s - loss: 0.2578 - categorical_accuracy: 0.9207
29728/60000 [=============>................] - ETA: 53s - loss: 0.2575 - categorical_accuracy: 0.9208
29760/60000 [=============>................] - ETA: 53s - loss: 0.2576 - categorical_accuracy: 0.9208
29792/60000 [=============>................] - ETA: 53s - loss: 0.2574 - categorical_accuracy: 0.9209
29824/60000 [=============>................] - ETA: 53s - loss: 0.2572 - categorical_accuracy: 0.9210
29856/60000 [=============>................] - ETA: 53s - loss: 0.2570 - categorical_accuracy: 0.9210
29888/60000 [=============>................] - ETA: 53s - loss: 0.2569 - categorical_accuracy: 0.9210
29920/60000 [=============>................] - ETA: 53s - loss: 0.2567 - categorical_accuracy: 0.9211
29952/60000 [=============>................] - ETA: 53s - loss: 0.2564 - categorical_accuracy: 0.9211
29984/60000 [=============>................] - ETA: 53s - loss: 0.2563 - categorical_accuracy: 0.9212
30016/60000 [==============>...............] - ETA: 52s - loss: 0.2561 - categorical_accuracy: 0.9212
30048/60000 [==============>...............] - ETA: 52s - loss: 0.2561 - categorical_accuracy: 0.9212
30080/60000 [==============>...............] - ETA: 52s - loss: 0.2559 - categorical_accuracy: 0.9212
30112/60000 [==============>...............] - ETA: 52s - loss: 0.2557 - categorical_accuracy: 0.9213
30144/60000 [==============>...............] - ETA: 52s - loss: 0.2555 - categorical_accuracy: 0.9214
30176/60000 [==============>...............] - ETA: 52s - loss: 0.2554 - categorical_accuracy: 0.9214
30208/60000 [==============>...............] - ETA: 52s - loss: 0.2552 - categorical_accuracy: 0.9214
30240/60000 [==============>...............] - ETA: 52s - loss: 0.2550 - categorical_accuracy: 0.9215
30272/60000 [==============>...............] - ETA: 52s - loss: 0.2550 - categorical_accuracy: 0.9215
30304/60000 [==============>...............] - ETA: 52s - loss: 0.2548 - categorical_accuracy: 0.9215
30336/60000 [==============>...............] - ETA: 52s - loss: 0.2547 - categorical_accuracy: 0.9215
30368/60000 [==============>...............] - ETA: 52s - loss: 0.2547 - categorical_accuracy: 0.9215
30400/60000 [==============>...............] - ETA: 52s - loss: 0.2545 - categorical_accuracy: 0.9215
30432/60000 [==============>...............] - ETA: 52s - loss: 0.2544 - categorical_accuracy: 0.9216
30464/60000 [==============>...............] - ETA: 52s - loss: 0.2542 - categorical_accuracy: 0.9216
30496/60000 [==============>...............] - ETA: 52s - loss: 0.2541 - categorical_accuracy: 0.9217
30528/60000 [==============>...............] - ETA: 52s - loss: 0.2539 - categorical_accuracy: 0.9217
30560/60000 [==============>...............] - ETA: 51s - loss: 0.2539 - categorical_accuracy: 0.9217
30592/60000 [==============>...............] - ETA: 51s - loss: 0.2537 - categorical_accuracy: 0.9218
30624/60000 [==============>...............] - ETA: 51s - loss: 0.2535 - categorical_accuracy: 0.9219
30656/60000 [==============>...............] - ETA: 51s - loss: 0.2532 - categorical_accuracy: 0.9219
30688/60000 [==============>...............] - ETA: 51s - loss: 0.2533 - categorical_accuracy: 0.9219
30720/60000 [==============>...............] - ETA: 51s - loss: 0.2530 - categorical_accuracy: 0.9219
30752/60000 [==============>...............] - ETA: 51s - loss: 0.2529 - categorical_accuracy: 0.9220
30784/60000 [==============>...............] - ETA: 51s - loss: 0.2526 - categorical_accuracy: 0.9220
30816/60000 [==============>...............] - ETA: 51s - loss: 0.2525 - categorical_accuracy: 0.9221
30848/60000 [==============>...............] - ETA: 51s - loss: 0.2524 - categorical_accuracy: 0.9221
30880/60000 [==============>...............] - ETA: 51s - loss: 0.2522 - categorical_accuracy: 0.9222
30912/60000 [==============>...............] - ETA: 51s - loss: 0.2522 - categorical_accuracy: 0.9222
30944/60000 [==============>...............] - ETA: 51s - loss: 0.2519 - categorical_accuracy: 0.9222
30976/60000 [==============>...............] - ETA: 51s - loss: 0.2517 - categorical_accuracy: 0.9223
31008/60000 [==============>...............] - ETA: 51s - loss: 0.2515 - categorical_accuracy: 0.9224
31040/60000 [==============>...............] - ETA: 51s - loss: 0.2514 - categorical_accuracy: 0.9224
31072/60000 [==============>...............] - ETA: 51s - loss: 0.2512 - categorical_accuracy: 0.9225
31104/60000 [==============>...............] - ETA: 51s - loss: 0.2511 - categorical_accuracy: 0.9225
31136/60000 [==============>...............] - ETA: 50s - loss: 0.2509 - categorical_accuracy: 0.9226
31168/60000 [==============>...............] - ETA: 50s - loss: 0.2506 - categorical_accuracy: 0.9227
31200/60000 [==============>...............] - ETA: 50s - loss: 0.2507 - categorical_accuracy: 0.9227
31232/60000 [==============>...............] - ETA: 50s - loss: 0.2505 - categorical_accuracy: 0.9228
31264/60000 [==============>...............] - ETA: 50s - loss: 0.2504 - categorical_accuracy: 0.9228
31296/60000 [==============>...............] - ETA: 50s - loss: 0.2502 - categorical_accuracy: 0.9229
31328/60000 [==============>...............] - ETA: 50s - loss: 0.2502 - categorical_accuracy: 0.9229
31360/60000 [==============>...............] - ETA: 50s - loss: 0.2500 - categorical_accuracy: 0.9230
31392/60000 [==============>...............] - ETA: 50s - loss: 0.2499 - categorical_accuracy: 0.9230
31424/60000 [==============>...............] - ETA: 50s - loss: 0.2500 - categorical_accuracy: 0.9230
31456/60000 [==============>...............] - ETA: 50s - loss: 0.2498 - categorical_accuracy: 0.9230
31488/60000 [==============>...............] - ETA: 50s - loss: 0.2497 - categorical_accuracy: 0.9231
31520/60000 [==============>...............] - ETA: 50s - loss: 0.2495 - categorical_accuracy: 0.9231
31552/60000 [==============>...............] - ETA: 50s - loss: 0.2495 - categorical_accuracy: 0.9231
31584/60000 [==============>...............] - ETA: 50s - loss: 0.2493 - categorical_accuracy: 0.9232
31616/60000 [==============>...............] - ETA: 50s - loss: 0.2491 - categorical_accuracy: 0.9233
31648/60000 [==============>...............] - ETA: 50s - loss: 0.2490 - categorical_accuracy: 0.9233
31680/60000 [==============>...............] - ETA: 49s - loss: 0.2489 - categorical_accuracy: 0.9233
31712/60000 [==============>...............] - ETA: 49s - loss: 0.2487 - categorical_accuracy: 0.9233
31744/60000 [==============>...............] - ETA: 49s - loss: 0.2487 - categorical_accuracy: 0.9234
31776/60000 [==============>...............] - ETA: 49s - loss: 0.2485 - categorical_accuracy: 0.9235
31808/60000 [==============>...............] - ETA: 49s - loss: 0.2483 - categorical_accuracy: 0.9235
31840/60000 [==============>...............] - ETA: 49s - loss: 0.2482 - categorical_accuracy: 0.9236
31872/60000 [==============>...............] - ETA: 49s - loss: 0.2481 - categorical_accuracy: 0.9236
31904/60000 [==============>...............] - ETA: 49s - loss: 0.2478 - categorical_accuracy: 0.9237
31936/60000 [==============>...............] - ETA: 49s - loss: 0.2477 - categorical_accuracy: 0.9237
31968/60000 [==============>...............] - ETA: 49s - loss: 0.2476 - categorical_accuracy: 0.9237
32000/60000 [===============>..............] - ETA: 49s - loss: 0.2474 - categorical_accuracy: 0.9238
32032/60000 [===============>..............] - ETA: 49s - loss: 0.2474 - categorical_accuracy: 0.9238
32064/60000 [===============>..............] - ETA: 49s - loss: 0.2472 - categorical_accuracy: 0.9239
32096/60000 [===============>..............] - ETA: 49s - loss: 0.2470 - categorical_accuracy: 0.9239
32128/60000 [===============>..............] - ETA: 49s - loss: 0.2468 - categorical_accuracy: 0.9240
32160/60000 [===============>..............] - ETA: 49s - loss: 0.2466 - categorical_accuracy: 0.9241
32192/60000 [===============>..............] - ETA: 49s - loss: 0.2465 - categorical_accuracy: 0.9241
32224/60000 [===============>..............] - ETA: 49s - loss: 0.2463 - categorical_accuracy: 0.9242
32256/60000 [===============>..............] - ETA: 48s - loss: 0.2463 - categorical_accuracy: 0.9242
32288/60000 [===============>..............] - ETA: 48s - loss: 0.2463 - categorical_accuracy: 0.9242
32320/60000 [===============>..............] - ETA: 48s - loss: 0.2461 - categorical_accuracy: 0.9243
32352/60000 [===============>..............] - ETA: 48s - loss: 0.2459 - categorical_accuracy: 0.9243
32384/60000 [===============>..............] - ETA: 48s - loss: 0.2458 - categorical_accuracy: 0.9243
32416/60000 [===============>..............] - ETA: 48s - loss: 0.2456 - categorical_accuracy: 0.9244
32448/60000 [===============>..............] - ETA: 48s - loss: 0.2459 - categorical_accuracy: 0.9244
32480/60000 [===============>..............] - ETA: 48s - loss: 0.2459 - categorical_accuracy: 0.9243
32512/60000 [===============>..............] - ETA: 48s - loss: 0.2458 - categorical_accuracy: 0.9244
32544/60000 [===============>..............] - ETA: 48s - loss: 0.2456 - categorical_accuracy: 0.9244
32576/60000 [===============>..............] - ETA: 48s - loss: 0.2454 - categorical_accuracy: 0.9245
32608/60000 [===============>..............] - ETA: 48s - loss: 0.2452 - categorical_accuracy: 0.9246
32640/60000 [===============>..............] - ETA: 48s - loss: 0.2450 - categorical_accuracy: 0.9247
32672/60000 [===============>..............] - ETA: 48s - loss: 0.2448 - categorical_accuracy: 0.9247
32704/60000 [===============>..............] - ETA: 48s - loss: 0.2446 - categorical_accuracy: 0.9248
32736/60000 [===============>..............] - ETA: 48s - loss: 0.2445 - categorical_accuracy: 0.9248
32768/60000 [===============>..............] - ETA: 48s - loss: 0.2443 - categorical_accuracy: 0.9249
32800/60000 [===============>..............] - ETA: 48s - loss: 0.2444 - categorical_accuracy: 0.9249
32832/60000 [===============>..............] - ETA: 47s - loss: 0.2443 - categorical_accuracy: 0.9250
32864/60000 [===============>..............] - ETA: 47s - loss: 0.2441 - categorical_accuracy: 0.9250
32896/60000 [===============>..............] - ETA: 47s - loss: 0.2439 - categorical_accuracy: 0.9251
32928/60000 [===============>..............] - ETA: 47s - loss: 0.2440 - categorical_accuracy: 0.9250
32960/60000 [===============>..............] - ETA: 47s - loss: 0.2438 - categorical_accuracy: 0.9251
32992/60000 [===============>..............] - ETA: 47s - loss: 0.2436 - categorical_accuracy: 0.9251
33024/60000 [===============>..............] - ETA: 47s - loss: 0.2435 - categorical_accuracy: 0.9251
33056/60000 [===============>..............] - ETA: 47s - loss: 0.2435 - categorical_accuracy: 0.9251
33088/60000 [===============>..............] - ETA: 47s - loss: 0.2435 - categorical_accuracy: 0.9251
33120/60000 [===============>..............] - ETA: 47s - loss: 0.2433 - categorical_accuracy: 0.9252
33152/60000 [===============>..............] - ETA: 47s - loss: 0.2431 - categorical_accuracy: 0.9253
33184/60000 [===============>..............] - ETA: 47s - loss: 0.2429 - categorical_accuracy: 0.9253
33216/60000 [===============>..............] - ETA: 47s - loss: 0.2428 - categorical_accuracy: 0.9253
33248/60000 [===============>..............] - ETA: 47s - loss: 0.2426 - categorical_accuracy: 0.9254
33280/60000 [===============>..............] - ETA: 47s - loss: 0.2425 - categorical_accuracy: 0.9254
33312/60000 [===============>..............] - ETA: 47s - loss: 0.2423 - categorical_accuracy: 0.9255
33344/60000 [===============>..............] - ETA: 47s - loss: 0.2422 - categorical_accuracy: 0.9255
33376/60000 [===============>..............] - ETA: 47s - loss: 0.2420 - categorical_accuracy: 0.9256
33408/60000 [===============>..............] - ETA: 46s - loss: 0.2418 - categorical_accuracy: 0.9256
33440/60000 [===============>..............] - ETA: 46s - loss: 0.2417 - categorical_accuracy: 0.9256
33472/60000 [===============>..............] - ETA: 46s - loss: 0.2415 - categorical_accuracy: 0.9257
33504/60000 [===============>..............] - ETA: 46s - loss: 0.2413 - categorical_accuracy: 0.9257
33536/60000 [===============>..............] - ETA: 46s - loss: 0.2411 - categorical_accuracy: 0.9258
33568/60000 [===============>..............] - ETA: 46s - loss: 0.2410 - categorical_accuracy: 0.9258
33600/60000 [===============>..............] - ETA: 46s - loss: 0.2408 - categorical_accuracy: 0.9259
33632/60000 [===============>..............] - ETA: 46s - loss: 0.2405 - categorical_accuracy: 0.9260
33664/60000 [===============>..............] - ETA: 46s - loss: 0.2404 - categorical_accuracy: 0.9260
33696/60000 [===============>..............] - ETA: 46s - loss: 0.2402 - categorical_accuracy: 0.9261
33728/60000 [===============>..............] - ETA: 46s - loss: 0.2402 - categorical_accuracy: 0.9261
33760/60000 [===============>..............] - ETA: 46s - loss: 0.2402 - categorical_accuracy: 0.9261
33792/60000 [===============>..............] - ETA: 46s - loss: 0.2401 - categorical_accuracy: 0.9261
33824/60000 [===============>..............] - ETA: 46s - loss: 0.2399 - categorical_accuracy: 0.9261
33856/60000 [===============>..............] - ETA: 46s - loss: 0.2398 - categorical_accuracy: 0.9262
33888/60000 [===============>..............] - ETA: 46s - loss: 0.2400 - categorical_accuracy: 0.9262
33920/60000 [===============>..............] - ETA: 46s - loss: 0.2399 - categorical_accuracy: 0.9262
33952/60000 [===============>..............] - ETA: 45s - loss: 0.2399 - categorical_accuracy: 0.9262
33984/60000 [===============>..............] - ETA: 45s - loss: 0.2397 - categorical_accuracy: 0.9262
34016/60000 [================>.............] - ETA: 45s - loss: 0.2395 - categorical_accuracy: 0.9263
34048/60000 [================>.............] - ETA: 45s - loss: 0.2396 - categorical_accuracy: 0.9263
34080/60000 [================>.............] - ETA: 45s - loss: 0.2394 - categorical_accuracy: 0.9264
34112/60000 [================>.............] - ETA: 45s - loss: 0.2393 - categorical_accuracy: 0.9264
34144/60000 [================>.............] - ETA: 45s - loss: 0.2391 - categorical_accuracy: 0.9264
34176/60000 [================>.............] - ETA: 45s - loss: 0.2390 - categorical_accuracy: 0.9264
34208/60000 [================>.............] - ETA: 45s - loss: 0.2389 - categorical_accuracy: 0.9264
34240/60000 [================>.............] - ETA: 45s - loss: 0.2388 - categorical_accuracy: 0.9265
34272/60000 [================>.............] - ETA: 45s - loss: 0.2386 - categorical_accuracy: 0.9266
34304/60000 [================>.............] - ETA: 45s - loss: 0.2385 - categorical_accuracy: 0.9266
34336/60000 [================>.............] - ETA: 45s - loss: 0.2383 - categorical_accuracy: 0.9267
34368/60000 [================>.............] - ETA: 45s - loss: 0.2383 - categorical_accuracy: 0.9267
34400/60000 [================>.............] - ETA: 45s - loss: 0.2381 - categorical_accuracy: 0.9267
34432/60000 [================>.............] - ETA: 45s - loss: 0.2381 - categorical_accuracy: 0.9268
34464/60000 [================>.............] - ETA: 45s - loss: 0.2379 - categorical_accuracy: 0.9268
34496/60000 [================>.............] - ETA: 45s - loss: 0.2377 - categorical_accuracy: 0.9269
34528/60000 [================>.............] - ETA: 44s - loss: 0.2377 - categorical_accuracy: 0.9268
34560/60000 [================>.............] - ETA: 44s - loss: 0.2375 - categorical_accuracy: 0.9269
34592/60000 [================>.............] - ETA: 44s - loss: 0.2376 - categorical_accuracy: 0.9269
34624/60000 [================>.............] - ETA: 44s - loss: 0.2374 - categorical_accuracy: 0.9269
34656/60000 [================>.............] - ETA: 44s - loss: 0.2373 - categorical_accuracy: 0.9269
34688/60000 [================>.............] - ETA: 44s - loss: 0.2372 - categorical_accuracy: 0.9269
34720/60000 [================>.............] - ETA: 44s - loss: 0.2371 - categorical_accuracy: 0.9270
34752/60000 [================>.............] - ETA: 44s - loss: 0.2370 - categorical_accuracy: 0.9270
34784/60000 [================>.............] - ETA: 44s - loss: 0.2370 - categorical_accuracy: 0.9269
34816/60000 [================>.............] - ETA: 44s - loss: 0.2368 - categorical_accuracy: 0.9270
34848/60000 [================>.............] - ETA: 44s - loss: 0.2366 - categorical_accuracy: 0.9270
34880/60000 [================>.............] - ETA: 44s - loss: 0.2366 - categorical_accuracy: 0.9270
34912/60000 [================>.............] - ETA: 44s - loss: 0.2365 - categorical_accuracy: 0.9271
34944/60000 [================>.............] - ETA: 44s - loss: 0.2365 - categorical_accuracy: 0.9270
34976/60000 [================>.............] - ETA: 44s - loss: 0.2364 - categorical_accuracy: 0.9270
35008/60000 [================>.............] - ETA: 44s - loss: 0.2362 - categorical_accuracy: 0.9271
35040/60000 [================>.............] - ETA: 44s - loss: 0.2360 - categorical_accuracy: 0.9272
35072/60000 [================>.............] - ETA: 44s - loss: 0.2357 - categorical_accuracy: 0.9272
35104/60000 [================>.............] - ETA: 43s - loss: 0.2357 - categorical_accuracy: 0.9272
35136/60000 [================>.............] - ETA: 43s - loss: 0.2355 - categorical_accuracy: 0.9273
35168/60000 [================>.............] - ETA: 43s - loss: 0.2353 - categorical_accuracy: 0.9274
35200/60000 [================>.............] - ETA: 43s - loss: 0.2353 - categorical_accuracy: 0.9274
35232/60000 [================>.............] - ETA: 43s - loss: 0.2352 - categorical_accuracy: 0.9274
35264/60000 [================>.............] - ETA: 43s - loss: 0.2350 - categorical_accuracy: 0.9274
35296/60000 [================>.............] - ETA: 43s - loss: 0.2349 - categorical_accuracy: 0.9274
35328/60000 [================>.............] - ETA: 43s - loss: 0.2347 - categorical_accuracy: 0.9275
35360/60000 [================>.............] - ETA: 43s - loss: 0.2345 - categorical_accuracy: 0.9276
35392/60000 [================>.............] - ETA: 43s - loss: 0.2344 - categorical_accuracy: 0.9276
35424/60000 [================>.............] - ETA: 43s - loss: 0.2342 - categorical_accuracy: 0.9276
35456/60000 [================>.............] - ETA: 43s - loss: 0.2341 - categorical_accuracy: 0.9277
35488/60000 [================>.............] - ETA: 43s - loss: 0.2339 - categorical_accuracy: 0.9278
35520/60000 [================>.............] - ETA: 43s - loss: 0.2338 - categorical_accuracy: 0.9278
35552/60000 [================>.............] - ETA: 43s - loss: 0.2336 - categorical_accuracy: 0.9279
35584/60000 [================>.............] - ETA: 43s - loss: 0.2334 - categorical_accuracy: 0.9279
35616/60000 [================>.............] - ETA: 43s - loss: 0.2333 - categorical_accuracy: 0.9280
35648/60000 [================>.............] - ETA: 42s - loss: 0.2333 - categorical_accuracy: 0.9280
35680/60000 [================>.............] - ETA: 42s - loss: 0.2332 - categorical_accuracy: 0.9280
35712/60000 [================>.............] - ETA: 42s - loss: 0.2331 - categorical_accuracy: 0.9280
35744/60000 [================>.............] - ETA: 42s - loss: 0.2330 - categorical_accuracy: 0.9281
35776/60000 [================>.............] - ETA: 42s - loss: 0.2328 - categorical_accuracy: 0.9281
35808/60000 [================>.............] - ETA: 42s - loss: 0.2327 - categorical_accuracy: 0.9282
35840/60000 [================>.............] - ETA: 42s - loss: 0.2326 - categorical_accuracy: 0.9282
35872/60000 [================>.............] - ETA: 42s - loss: 0.2325 - categorical_accuracy: 0.9283
35904/60000 [================>.............] - ETA: 42s - loss: 0.2323 - categorical_accuracy: 0.9283
35936/60000 [================>.............] - ETA: 42s - loss: 0.2322 - categorical_accuracy: 0.9283
35968/60000 [================>.............] - ETA: 42s - loss: 0.2322 - categorical_accuracy: 0.9284
36000/60000 [=================>............] - ETA: 42s - loss: 0.2321 - categorical_accuracy: 0.9284
36032/60000 [=================>............] - ETA: 42s - loss: 0.2320 - categorical_accuracy: 0.9284
36064/60000 [=================>............] - ETA: 42s - loss: 0.2319 - categorical_accuracy: 0.9284
36096/60000 [=================>............] - ETA: 42s - loss: 0.2317 - categorical_accuracy: 0.9285
36128/60000 [=================>............] - ETA: 42s - loss: 0.2317 - categorical_accuracy: 0.9285
36160/60000 [=================>............] - ETA: 42s - loss: 0.2316 - categorical_accuracy: 0.9286
36192/60000 [=================>............] - ETA: 42s - loss: 0.2315 - categorical_accuracy: 0.9286
36224/60000 [=================>............] - ETA: 41s - loss: 0.2315 - categorical_accuracy: 0.9286
36256/60000 [=================>............] - ETA: 41s - loss: 0.2315 - categorical_accuracy: 0.9286
36288/60000 [=================>............] - ETA: 41s - loss: 0.2314 - categorical_accuracy: 0.9287
36320/60000 [=================>............] - ETA: 41s - loss: 0.2313 - categorical_accuracy: 0.9287
36352/60000 [=================>............] - ETA: 41s - loss: 0.2313 - categorical_accuracy: 0.9287
36384/60000 [=================>............] - ETA: 41s - loss: 0.2311 - categorical_accuracy: 0.9288
36416/60000 [=================>............] - ETA: 41s - loss: 0.2309 - categorical_accuracy: 0.9288
36448/60000 [=================>............] - ETA: 41s - loss: 0.2309 - categorical_accuracy: 0.9289
36480/60000 [=================>............] - ETA: 41s - loss: 0.2308 - categorical_accuracy: 0.9289
36512/60000 [=================>............] - ETA: 41s - loss: 0.2308 - categorical_accuracy: 0.9289
36544/60000 [=================>............] - ETA: 41s - loss: 0.2306 - categorical_accuracy: 0.9289
36576/60000 [=================>............] - ETA: 41s - loss: 0.2305 - categorical_accuracy: 0.9289
36608/60000 [=================>............] - ETA: 41s - loss: 0.2304 - categorical_accuracy: 0.9289
36640/60000 [=================>............] - ETA: 41s - loss: 0.2303 - categorical_accuracy: 0.9289
36672/60000 [=================>............] - ETA: 41s - loss: 0.2302 - categorical_accuracy: 0.9290
36704/60000 [=================>............] - ETA: 41s - loss: 0.2301 - categorical_accuracy: 0.9290
36736/60000 [=================>............] - ETA: 41s - loss: 0.2300 - categorical_accuracy: 0.9290
36768/60000 [=================>............] - ETA: 40s - loss: 0.2299 - categorical_accuracy: 0.9290
36800/60000 [=================>............] - ETA: 40s - loss: 0.2297 - categorical_accuracy: 0.9291
36832/60000 [=================>............] - ETA: 40s - loss: 0.2296 - categorical_accuracy: 0.9291
36864/60000 [=================>............] - ETA: 40s - loss: 0.2295 - categorical_accuracy: 0.9292
36896/60000 [=================>............] - ETA: 40s - loss: 0.2295 - categorical_accuracy: 0.9292
36928/60000 [=================>............] - ETA: 40s - loss: 0.2295 - categorical_accuracy: 0.9292
36960/60000 [=================>............] - ETA: 40s - loss: 0.2295 - categorical_accuracy: 0.9291
36992/60000 [=================>............] - ETA: 40s - loss: 0.2293 - categorical_accuracy: 0.9292
37024/60000 [=================>............] - ETA: 40s - loss: 0.2291 - categorical_accuracy: 0.9293
37056/60000 [=================>............] - ETA: 40s - loss: 0.2290 - categorical_accuracy: 0.9293
37088/60000 [=================>............] - ETA: 40s - loss: 0.2289 - categorical_accuracy: 0.9293
37120/60000 [=================>............] - ETA: 40s - loss: 0.2288 - categorical_accuracy: 0.9294
37152/60000 [=================>............] - ETA: 40s - loss: 0.2289 - categorical_accuracy: 0.9294
37184/60000 [=================>............] - ETA: 40s - loss: 0.2287 - categorical_accuracy: 0.9295
37216/60000 [=================>............] - ETA: 40s - loss: 0.2285 - categorical_accuracy: 0.9295
37248/60000 [=================>............] - ETA: 40s - loss: 0.2284 - categorical_accuracy: 0.9296
37280/60000 [=================>............] - ETA: 40s - loss: 0.2283 - categorical_accuracy: 0.9296
37312/60000 [=================>............] - ETA: 40s - loss: 0.2282 - categorical_accuracy: 0.9296
37344/60000 [=================>............] - ETA: 39s - loss: 0.2282 - categorical_accuracy: 0.9296
37376/60000 [=================>............] - ETA: 39s - loss: 0.2281 - categorical_accuracy: 0.9297
37408/60000 [=================>............] - ETA: 39s - loss: 0.2280 - categorical_accuracy: 0.9297
37440/60000 [=================>............] - ETA: 39s - loss: 0.2280 - categorical_accuracy: 0.9297
37472/60000 [=================>............] - ETA: 39s - loss: 0.2279 - categorical_accuracy: 0.9297
37504/60000 [=================>............] - ETA: 39s - loss: 0.2278 - categorical_accuracy: 0.9297
37536/60000 [=================>............] - ETA: 39s - loss: 0.2277 - categorical_accuracy: 0.9297
37568/60000 [=================>............] - ETA: 39s - loss: 0.2278 - categorical_accuracy: 0.9297
37600/60000 [=================>............] - ETA: 39s - loss: 0.2276 - categorical_accuracy: 0.9298
37632/60000 [=================>............] - ETA: 39s - loss: 0.2276 - categorical_accuracy: 0.9298
37664/60000 [=================>............] - ETA: 39s - loss: 0.2274 - categorical_accuracy: 0.9299
37696/60000 [=================>............] - ETA: 39s - loss: 0.2273 - categorical_accuracy: 0.9299
37728/60000 [=================>............] - ETA: 39s - loss: 0.2271 - categorical_accuracy: 0.9299
37760/60000 [=================>............] - ETA: 39s - loss: 0.2269 - categorical_accuracy: 0.9300
37792/60000 [=================>............] - ETA: 39s - loss: 0.2268 - categorical_accuracy: 0.9301
37824/60000 [=================>............] - ETA: 39s - loss: 0.2267 - categorical_accuracy: 0.9301
37856/60000 [=================>............] - ETA: 39s - loss: 0.2267 - categorical_accuracy: 0.9301
37888/60000 [=================>............] - ETA: 39s - loss: 0.2265 - categorical_accuracy: 0.9301
37920/60000 [=================>............] - ETA: 38s - loss: 0.2264 - categorical_accuracy: 0.9302
37952/60000 [=================>............] - ETA: 38s - loss: 0.2264 - categorical_accuracy: 0.9302
37984/60000 [=================>............] - ETA: 38s - loss: 0.2263 - categorical_accuracy: 0.9302
38016/60000 [==================>...........] - ETA: 38s - loss: 0.2262 - categorical_accuracy: 0.9302
38048/60000 [==================>...........] - ETA: 38s - loss: 0.2261 - categorical_accuracy: 0.9302
38080/60000 [==================>...........] - ETA: 38s - loss: 0.2261 - categorical_accuracy: 0.9302
38112/60000 [==================>...........] - ETA: 38s - loss: 0.2260 - categorical_accuracy: 0.9303
38144/60000 [==================>...........] - ETA: 38s - loss: 0.2259 - categorical_accuracy: 0.9303
38176/60000 [==================>...........] - ETA: 38s - loss: 0.2260 - categorical_accuracy: 0.9303
38208/60000 [==================>...........] - ETA: 38s - loss: 0.2258 - categorical_accuracy: 0.9304
38240/60000 [==================>...........] - ETA: 38s - loss: 0.2257 - categorical_accuracy: 0.9304
38272/60000 [==================>...........] - ETA: 38s - loss: 0.2255 - categorical_accuracy: 0.9305
38304/60000 [==================>...........] - ETA: 38s - loss: 0.2253 - categorical_accuracy: 0.9305
38336/60000 [==================>...........] - ETA: 38s - loss: 0.2252 - categorical_accuracy: 0.9306
38368/60000 [==================>...........] - ETA: 38s - loss: 0.2251 - categorical_accuracy: 0.9306
38400/60000 [==================>...........] - ETA: 38s - loss: 0.2250 - categorical_accuracy: 0.9306
38432/60000 [==================>...........] - ETA: 38s - loss: 0.2249 - categorical_accuracy: 0.9307
38464/60000 [==================>...........] - ETA: 37s - loss: 0.2248 - categorical_accuracy: 0.9307
38496/60000 [==================>...........] - ETA: 37s - loss: 0.2247 - categorical_accuracy: 0.9307
38528/60000 [==================>...........] - ETA: 37s - loss: 0.2246 - categorical_accuracy: 0.9308
38560/60000 [==================>...........] - ETA: 37s - loss: 0.2245 - categorical_accuracy: 0.9308
38592/60000 [==================>...........] - ETA: 37s - loss: 0.2243 - categorical_accuracy: 0.9308
38624/60000 [==================>...........] - ETA: 37s - loss: 0.2243 - categorical_accuracy: 0.9309
38656/60000 [==================>...........] - ETA: 37s - loss: 0.2245 - categorical_accuracy: 0.9309
38688/60000 [==================>...........] - ETA: 37s - loss: 0.2244 - categorical_accuracy: 0.9309
38720/60000 [==================>...........] - ETA: 37s - loss: 0.2242 - categorical_accuracy: 0.9310
38752/60000 [==================>...........] - ETA: 37s - loss: 0.2245 - categorical_accuracy: 0.9309
38784/60000 [==================>...........] - ETA: 37s - loss: 0.2243 - categorical_accuracy: 0.9310
38816/60000 [==================>...........] - ETA: 37s - loss: 0.2242 - categorical_accuracy: 0.9310
38848/60000 [==================>...........] - ETA: 37s - loss: 0.2240 - categorical_accuracy: 0.9311
38880/60000 [==================>...........] - ETA: 37s - loss: 0.2240 - categorical_accuracy: 0.9311
38912/60000 [==================>...........] - ETA: 37s - loss: 0.2239 - categorical_accuracy: 0.9311
38944/60000 [==================>...........] - ETA: 37s - loss: 0.2239 - categorical_accuracy: 0.9311
38976/60000 [==================>...........] - ETA: 37s - loss: 0.2238 - categorical_accuracy: 0.9312
39008/60000 [==================>...........] - ETA: 37s - loss: 0.2236 - categorical_accuracy: 0.9312
39040/60000 [==================>...........] - ETA: 36s - loss: 0.2236 - categorical_accuracy: 0.9312
39072/60000 [==================>...........] - ETA: 36s - loss: 0.2234 - categorical_accuracy: 0.9313
39104/60000 [==================>...........] - ETA: 36s - loss: 0.2233 - categorical_accuracy: 0.9313
39136/60000 [==================>...........] - ETA: 36s - loss: 0.2232 - categorical_accuracy: 0.9314
39168/60000 [==================>...........] - ETA: 36s - loss: 0.2231 - categorical_accuracy: 0.9314
39200/60000 [==================>...........] - ETA: 36s - loss: 0.2229 - categorical_accuracy: 0.9315
39232/60000 [==================>...........] - ETA: 36s - loss: 0.2227 - categorical_accuracy: 0.9315
39264/60000 [==================>...........] - ETA: 36s - loss: 0.2226 - categorical_accuracy: 0.9315
39296/60000 [==================>...........] - ETA: 36s - loss: 0.2226 - categorical_accuracy: 0.9315
39328/60000 [==================>...........] - ETA: 36s - loss: 0.2225 - categorical_accuracy: 0.9316
39360/60000 [==================>...........] - ETA: 36s - loss: 0.2223 - categorical_accuracy: 0.9316
39392/60000 [==================>...........] - ETA: 36s - loss: 0.2222 - categorical_accuracy: 0.9317
39424/60000 [==================>...........] - ETA: 36s - loss: 0.2221 - categorical_accuracy: 0.9317
39456/60000 [==================>...........] - ETA: 36s - loss: 0.2220 - categorical_accuracy: 0.9317
39488/60000 [==================>...........] - ETA: 36s - loss: 0.2219 - categorical_accuracy: 0.9318
39520/60000 [==================>...........] - ETA: 36s - loss: 0.2218 - categorical_accuracy: 0.9318
39552/60000 [==================>...........] - ETA: 36s - loss: 0.2217 - categorical_accuracy: 0.9318
39584/60000 [==================>...........] - ETA: 36s - loss: 0.2217 - categorical_accuracy: 0.9318
39616/60000 [==================>...........] - ETA: 35s - loss: 0.2216 - categorical_accuracy: 0.9318
39648/60000 [==================>...........] - ETA: 35s - loss: 0.2215 - categorical_accuracy: 0.9319
39680/60000 [==================>...........] - ETA: 35s - loss: 0.2215 - categorical_accuracy: 0.9319
39712/60000 [==================>...........] - ETA: 35s - loss: 0.2213 - categorical_accuracy: 0.9319
39744/60000 [==================>...........] - ETA: 35s - loss: 0.2215 - categorical_accuracy: 0.9319
39776/60000 [==================>...........] - ETA: 35s - loss: 0.2215 - categorical_accuracy: 0.9318
39808/60000 [==================>...........] - ETA: 35s - loss: 0.2214 - categorical_accuracy: 0.9319
39840/60000 [==================>...........] - ETA: 35s - loss: 0.2216 - categorical_accuracy: 0.9319
39872/60000 [==================>...........] - ETA: 35s - loss: 0.2215 - categorical_accuracy: 0.9319
39904/60000 [==================>...........] - ETA: 35s - loss: 0.2214 - categorical_accuracy: 0.9320
39936/60000 [==================>...........] - ETA: 35s - loss: 0.2213 - categorical_accuracy: 0.9320
39968/60000 [==================>...........] - ETA: 35s - loss: 0.2211 - categorical_accuracy: 0.9320
40000/60000 [===================>..........] - ETA: 35s - loss: 0.2209 - categorical_accuracy: 0.9321
40032/60000 [===================>..........] - ETA: 35s - loss: 0.2209 - categorical_accuracy: 0.9321
40064/60000 [===================>..........] - ETA: 35s - loss: 0.2208 - categorical_accuracy: 0.9321
40096/60000 [===================>..........] - ETA: 35s - loss: 0.2207 - categorical_accuracy: 0.9321
40128/60000 [===================>..........] - ETA: 35s - loss: 0.2207 - categorical_accuracy: 0.9321
40160/60000 [===================>..........] - ETA: 35s - loss: 0.2206 - categorical_accuracy: 0.9321
40192/60000 [===================>..........] - ETA: 34s - loss: 0.2204 - categorical_accuracy: 0.9322
40224/60000 [===================>..........] - ETA: 34s - loss: 0.2203 - categorical_accuracy: 0.9322
40256/60000 [===================>..........] - ETA: 34s - loss: 0.2201 - categorical_accuracy: 0.9323
40288/60000 [===================>..........] - ETA: 34s - loss: 0.2200 - categorical_accuracy: 0.9323
40320/60000 [===================>..........] - ETA: 34s - loss: 0.2199 - categorical_accuracy: 0.9324
40352/60000 [===================>..........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9324
40384/60000 [===================>..........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9324
40416/60000 [===================>..........] - ETA: 34s - loss: 0.2198 - categorical_accuracy: 0.9324
40448/60000 [===================>..........] - ETA: 34s - loss: 0.2196 - categorical_accuracy: 0.9325
40480/60000 [===================>..........] - ETA: 34s - loss: 0.2195 - categorical_accuracy: 0.9325
40512/60000 [===================>..........] - ETA: 34s - loss: 0.2193 - categorical_accuracy: 0.9325
40544/60000 [===================>..........] - ETA: 34s - loss: 0.2192 - categorical_accuracy: 0.9326
40576/60000 [===================>..........] - ETA: 34s - loss: 0.2191 - categorical_accuracy: 0.9326
40608/60000 [===================>..........] - ETA: 34s - loss: 0.2191 - categorical_accuracy: 0.9326
40640/60000 [===================>..........] - ETA: 34s - loss: 0.2191 - categorical_accuracy: 0.9326
40672/60000 [===================>..........] - ETA: 34s - loss: 0.2192 - categorical_accuracy: 0.9326
40704/60000 [===================>..........] - ETA: 34s - loss: 0.2192 - categorical_accuracy: 0.9326
40736/60000 [===================>..........] - ETA: 33s - loss: 0.2191 - categorical_accuracy: 0.9326
40768/60000 [===================>..........] - ETA: 33s - loss: 0.2190 - categorical_accuracy: 0.9326
40800/60000 [===================>..........] - ETA: 33s - loss: 0.2189 - categorical_accuracy: 0.9327
40832/60000 [===================>..........] - ETA: 33s - loss: 0.2187 - categorical_accuracy: 0.9327
40864/60000 [===================>..........] - ETA: 33s - loss: 0.2187 - categorical_accuracy: 0.9327
40896/60000 [===================>..........] - ETA: 33s - loss: 0.2186 - categorical_accuracy: 0.9327
40928/60000 [===================>..........] - ETA: 33s - loss: 0.2185 - categorical_accuracy: 0.9328
40960/60000 [===================>..........] - ETA: 33s - loss: 0.2185 - categorical_accuracy: 0.9328
40992/60000 [===================>..........] - ETA: 33s - loss: 0.2184 - categorical_accuracy: 0.9328
41024/60000 [===================>..........] - ETA: 33s - loss: 0.2182 - categorical_accuracy: 0.9328
41056/60000 [===================>..........] - ETA: 33s - loss: 0.2182 - categorical_accuracy: 0.9329
41088/60000 [===================>..........] - ETA: 33s - loss: 0.2180 - categorical_accuracy: 0.9329
41120/60000 [===================>..........] - ETA: 33s - loss: 0.2179 - categorical_accuracy: 0.9330
41152/60000 [===================>..........] - ETA: 33s - loss: 0.2179 - categorical_accuracy: 0.9330
41184/60000 [===================>..........] - ETA: 33s - loss: 0.2178 - categorical_accuracy: 0.9330
41216/60000 [===================>..........] - ETA: 33s - loss: 0.2177 - categorical_accuracy: 0.9330
41248/60000 [===================>..........] - ETA: 33s - loss: 0.2176 - categorical_accuracy: 0.9331
41280/60000 [===================>..........] - ETA: 33s - loss: 0.2176 - categorical_accuracy: 0.9331
41312/60000 [===================>..........] - ETA: 32s - loss: 0.2174 - categorical_accuracy: 0.9331
41344/60000 [===================>..........] - ETA: 32s - loss: 0.2176 - categorical_accuracy: 0.9331
41376/60000 [===================>..........] - ETA: 32s - loss: 0.2175 - categorical_accuracy: 0.9332
41408/60000 [===================>..........] - ETA: 32s - loss: 0.2176 - categorical_accuracy: 0.9332
41440/60000 [===================>..........] - ETA: 32s - loss: 0.2175 - categorical_accuracy: 0.9332
41472/60000 [===================>..........] - ETA: 32s - loss: 0.2173 - categorical_accuracy: 0.9333
41504/60000 [===================>..........] - ETA: 32s - loss: 0.2172 - categorical_accuracy: 0.9333
41536/60000 [===================>..........] - ETA: 32s - loss: 0.2171 - categorical_accuracy: 0.9333
41568/60000 [===================>..........] - ETA: 32s - loss: 0.2171 - categorical_accuracy: 0.9333
41600/60000 [===================>..........] - ETA: 32s - loss: 0.2169 - categorical_accuracy: 0.9334
41632/60000 [===================>..........] - ETA: 32s - loss: 0.2168 - categorical_accuracy: 0.9334
41664/60000 [===================>..........] - ETA: 32s - loss: 0.2166 - categorical_accuracy: 0.9334
41696/60000 [===================>..........] - ETA: 32s - loss: 0.2165 - categorical_accuracy: 0.9335
41728/60000 [===================>..........] - ETA: 32s - loss: 0.2164 - categorical_accuracy: 0.9335
41760/60000 [===================>..........] - ETA: 32s - loss: 0.2163 - categorical_accuracy: 0.9335
41792/60000 [===================>..........] - ETA: 32s - loss: 0.2162 - categorical_accuracy: 0.9336
41824/60000 [===================>..........] - ETA: 32s - loss: 0.2161 - categorical_accuracy: 0.9336
41856/60000 [===================>..........] - ETA: 31s - loss: 0.2160 - categorical_accuracy: 0.9336
41888/60000 [===================>..........] - ETA: 31s - loss: 0.2159 - categorical_accuracy: 0.9337
41920/60000 [===================>..........] - ETA: 31s - loss: 0.2159 - categorical_accuracy: 0.9336
41952/60000 [===================>..........] - ETA: 31s - loss: 0.2159 - categorical_accuracy: 0.9336
41984/60000 [===================>..........] - ETA: 31s - loss: 0.2158 - categorical_accuracy: 0.9337
42016/60000 [====================>.........] - ETA: 31s - loss: 0.2156 - categorical_accuracy: 0.9337
42048/60000 [====================>.........] - ETA: 31s - loss: 0.2156 - categorical_accuracy: 0.9337
42080/60000 [====================>.........] - ETA: 31s - loss: 0.2154 - categorical_accuracy: 0.9338
42112/60000 [====================>.........] - ETA: 31s - loss: 0.2153 - categorical_accuracy: 0.9338
42144/60000 [====================>.........] - ETA: 31s - loss: 0.2152 - categorical_accuracy: 0.9339
42176/60000 [====================>.........] - ETA: 31s - loss: 0.2151 - categorical_accuracy: 0.9338
42208/60000 [====================>.........] - ETA: 31s - loss: 0.2150 - categorical_accuracy: 0.9339
42240/60000 [====================>.........] - ETA: 31s - loss: 0.2149 - categorical_accuracy: 0.9339
42272/60000 [====================>.........] - ETA: 31s - loss: 0.2147 - categorical_accuracy: 0.9340
42304/60000 [====================>.........] - ETA: 31s - loss: 0.2146 - categorical_accuracy: 0.9340
42336/60000 [====================>.........] - ETA: 31s - loss: 0.2144 - categorical_accuracy: 0.9341
42368/60000 [====================>.........] - ETA: 31s - loss: 0.2143 - categorical_accuracy: 0.9341
42400/60000 [====================>.........] - ETA: 31s - loss: 0.2142 - categorical_accuracy: 0.9342
42432/60000 [====================>.........] - ETA: 30s - loss: 0.2141 - categorical_accuracy: 0.9342
42464/60000 [====================>.........] - ETA: 30s - loss: 0.2140 - categorical_accuracy: 0.9342
42496/60000 [====================>.........] - ETA: 30s - loss: 0.2139 - categorical_accuracy: 0.9342
42528/60000 [====================>.........] - ETA: 30s - loss: 0.2140 - categorical_accuracy: 0.9342
42560/60000 [====================>.........] - ETA: 30s - loss: 0.2139 - categorical_accuracy: 0.9343
42592/60000 [====================>.........] - ETA: 30s - loss: 0.2138 - categorical_accuracy: 0.9343
42624/60000 [====================>.........] - ETA: 30s - loss: 0.2137 - categorical_accuracy: 0.9343
42656/60000 [====================>.........] - ETA: 30s - loss: 0.2136 - categorical_accuracy: 0.9344
42688/60000 [====================>.........] - ETA: 30s - loss: 0.2134 - categorical_accuracy: 0.9344
42720/60000 [====================>.........] - ETA: 30s - loss: 0.2133 - categorical_accuracy: 0.9345
42752/60000 [====================>.........] - ETA: 30s - loss: 0.2133 - categorical_accuracy: 0.9345
42784/60000 [====================>.........] - ETA: 30s - loss: 0.2131 - categorical_accuracy: 0.9345
42816/60000 [====================>.........] - ETA: 30s - loss: 0.2131 - categorical_accuracy: 0.9346
42848/60000 [====================>.........] - ETA: 30s - loss: 0.2130 - categorical_accuracy: 0.9346
42880/60000 [====================>.........] - ETA: 30s - loss: 0.2128 - categorical_accuracy: 0.9346
42912/60000 [====================>.........] - ETA: 30s - loss: 0.2128 - categorical_accuracy: 0.9346
42944/60000 [====================>.........] - ETA: 30s - loss: 0.2126 - categorical_accuracy: 0.9347
42976/60000 [====================>.........] - ETA: 30s - loss: 0.2125 - categorical_accuracy: 0.9347
43008/60000 [====================>.........] - ETA: 29s - loss: 0.2124 - categorical_accuracy: 0.9348
43040/60000 [====================>.........] - ETA: 29s - loss: 0.2125 - categorical_accuracy: 0.9347
43072/60000 [====================>.........] - ETA: 29s - loss: 0.2124 - categorical_accuracy: 0.9348
43104/60000 [====================>.........] - ETA: 29s - loss: 0.2123 - categorical_accuracy: 0.9348
43136/60000 [====================>.........] - ETA: 29s - loss: 0.2123 - categorical_accuracy: 0.9348
43168/60000 [====================>.........] - ETA: 29s - loss: 0.2121 - categorical_accuracy: 0.9348
43200/60000 [====================>.........] - ETA: 29s - loss: 0.2120 - categorical_accuracy: 0.9348
43232/60000 [====================>.........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9349
43264/60000 [====================>.........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9349
43296/60000 [====================>.........] - ETA: 29s - loss: 0.2119 - categorical_accuracy: 0.9349
43328/60000 [====================>.........] - ETA: 29s - loss: 0.2118 - categorical_accuracy: 0.9349
43360/60000 [====================>.........] - ETA: 29s - loss: 0.2117 - categorical_accuracy: 0.9350
43392/60000 [====================>.........] - ETA: 29s - loss: 0.2115 - categorical_accuracy: 0.9350
43424/60000 [====================>.........] - ETA: 29s - loss: 0.2115 - categorical_accuracy: 0.9350
43456/60000 [====================>.........] - ETA: 29s - loss: 0.2114 - categorical_accuracy: 0.9351
43488/60000 [====================>.........] - ETA: 29s - loss: 0.2114 - categorical_accuracy: 0.9350
43520/60000 [====================>.........] - ETA: 29s - loss: 0.2114 - categorical_accuracy: 0.9351
43552/60000 [====================>.........] - ETA: 28s - loss: 0.2113 - categorical_accuracy: 0.9351
43584/60000 [====================>.........] - ETA: 28s - loss: 0.2112 - categorical_accuracy: 0.9351
43616/60000 [====================>.........] - ETA: 28s - loss: 0.2111 - categorical_accuracy: 0.9352
43648/60000 [====================>.........] - ETA: 28s - loss: 0.2110 - categorical_accuracy: 0.9352
43680/60000 [====================>.........] - ETA: 28s - loss: 0.2109 - categorical_accuracy: 0.9352
43712/60000 [====================>.........] - ETA: 28s - loss: 0.2109 - categorical_accuracy: 0.9352
43744/60000 [====================>.........] - ETA: 28s - loss: 0.2107 - categorical_accuracy: 0.9353
43776/60000 [====================>.........] - ETA: 28s - loss: 0.2106 - categorical_accuracy: 0.9353
43808/60000 [====================>.........] - ETA: 28s - loss: 0.2104 - categorical_accuracy: 0.9354
43840/60000 [====================>.........] - ETA: 28s - loss: 0.2103 - categorical_accuracy: 0.9354
43872/60000 [====================>.........] - ETA: 28s - loss: 0.2102 - categorical_accuracy: 0.9354
43904/60000 [====================>.........] - ETA: 28s - loss: 0.2102 - categorical_accuracy: 0.9354
43936/60000 [====================>.........] - ETA: 28s - loss: 0.2101 - categorical_accuracy: 0.9354
43968/60000 [====================>.........] - ETA: 28s - loss: 0.2101 - categorical_accuracy: 0.9354
44000/60000 [=====================>........] - ETA: 28s - loss: 0.2100 - categorical_accuracy: 0.9355
44032/60000 [=====================>........] - ETA: 28s - loss: 0.2098 - categorical_accuracy: 0.9355
44064/60000 [=====================>........] - ETA: 28s - loss: 0.2097 - categorical_accuracy: 0.9356
44096/60000 [=====================>........] - ETA: 28s - loss: 0.2096 - categorical_accuracy: 0.9356
44128/60000 [=====================>........] - ETA: 27s - loss: 0.2095 - categorical_accuracy: 0.9356
44160/60000 [=====================>........] - ETA: 27s - loss: 0.2096 - categorical_accuracy: 0.9356
44192/60000 [=====================>........] - ETA: 27s - loss: 0.2096 - categorical_accuracy: 0.9356
44224/60000 [=====================>........] - ETA: 27s - loss: 0.2094 - categorical_accuracy: 0.9357
44256/60000 [=====================>........] - ETA: 27s - loss: 0.2094 - categorical_accuracy: 0.9357
44288/60000 [=====================>........] - ETA: 27s - loss: 0.2093 - categorical_accuracy: 0.9358
44320/60000 [=====================>........] - ETA: 27s - loss: 0.2095 - categorical_accuracy: 0.9357
44352/60000 [=====================>........] - ETA: 27s - loss: 0.2094 - categorical_accuracy: 0.9358
44384/60000 [=====================>........] - ETA: 27s - loss: 0.2093 - categorical_accuracy: 0.9358
44416/60000 [=====================>........] - ETA: 27s - loss: 0.2091 - categorical_accuracy: 0.9359
44448/60000 [=====================>........] - ETA: 27s - loss: 0.2092 - categorical_accuracy: 0.9358
44480/60000 [=====================>........] - ETA: 27s - loss: 0.2092 - categorical_accuracy: 0.9358
44512/60000 [=====================>........] - ETA: 27s - loss: 0.2091 - categorical_accuracy: 0.9359
44544/60000 [=====================>........] - ETA: 27s - loss: 0.2090 - categorical_accuracy: 0.9359
44576/60000 [=====================>........] - ETA: 27s - loss: 0.2088 - categorical_accuracy: 0.9360
44608/60000 [=====================>........] - ETA: 27s - loss: 0.2087 - categorical_accuracy: 0.9360
44640/60000 [=====================>........] - ETA: 27s - loss: 0.2086 - categorical_accuracy: 0.9360
44672/60000 [=====================>........] - ETA: 27s - loss: 0.2086 - categorical_accuracy: 0.9360
44704/60000 [=====================>........] - ETA: 26s - loss: 0.2084 - categorical_accuracy: 0.9361
44736/60000 [=====================>........] - ETA: 26s - loss: 0.2083 - categorical_accuracy: 0.9361
44768/60000 [=====================>........] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9361
44800/60000 [=====================>........] - ETA: 26s - loss: 0.2082 - categorical_accuracy: 0.9362
44832/60000 [=====================>........] - ETA: 26s - loss: 0.2080 - categorical_accuracy: 0.9362
44864/60000 [=====================>........] - ETA: 26s - loss: 0.2079 - categorical_accuracy: 0.9362
44896/60000 [=====================>........] - ETA: 26s - loss: 0.2078 - categorical_accuracy: 0.9363
44928/60000 [=====================>........] - ETA: 26s - loss: 0.2079 - categorical_accuracy: 0.9362
44960/60000 [=====================>........] - ETA: 26s - loss: 0.2080 - categorical_accuracy: 0.9362
44992/60000 [=====================>........] - ETA: 26s - loss: 0.2080 - categorical_accuracy: 0.9362
45024/60000 [=====================>........] - ETA: 26s - loss: 0.2079 - categorical_accuracy: 0.9363
45056/60000 [=====================>........] - ETA: 26s - loss: 0.2078 - categorical_accuracy: 0.9363
45088/60000 [=====================>........] - ETA: 26s - loss: 0.2077 - categorical_accuracy: 0.9363
45120/60000 [=====================>........] - ETA: 26s - loss: 0.2076 - categorical_accuracy: 0.9363
45152/60000 [=====================>........] - ETA: 26s - loss: 0.2075 - categorical_accuracy: 0.9364
45184/60000 [=====================>........] - ETA: 26s - loss: 0.2075 - categorical_accuracy: 0.9364
45216/60000 [=====================>........] - ETA: 26s - loss: 0.2074 - categorical_accuracy: 0.9364
45248/60000 [=====================>........] - ETA: 26s - loss: 0.2074 - categorical_accuracy: 0.9364
45280/60000 [=====================>........] - ETA: 25s - loss: 0.2074 - categorical_accuracy: 0.9364
45312/60000 [=====================>........] - ETA: 25s - loss: 0.2074 - categorical_accuracy: 0.9364
45344/60000 [=====================>........] - ETA: 25s - loss: 0.2073 - categorical_accuracy: 0.9364
45376/60000 [=====================>........] - ETA: 25s - loss: 0.2072 - categorical_accuracy: 0.9364
45408/60000 [=====================>........] - ETA: 25s - loss: 0.2072 - categorical_accuracy: 0.9364
45440/60000 [=====================>........] - ETA: 25s - loss: 0.2071 - categorical_accuracy: 0.9365
45472/60000 [=====================>........] - ETA: 25s - loss: 0.2072 - categorical_accuracy: 0.9364
45504/60000 [=====================>........] - ETA: 25s - loss: 0.2071 - categorical_accuracy: 0.9365
45536/60000 [=====================>........] - ETA: 25s - loss: 0.2070 - categorical_accuracy: 0.9365
45568/60000 [=====================>........] - ETA: 25s - loss: 0.2069 - categorical_accuracy: 0.9365
45600/60000 [=====================>........] - ETA: 25s - loss: 0.2068 - categorical_accuracy: 0.9365
45632/60000 [=====================>........] - ETA: 25s - loss: 0.2068 - categorical_accuracy: 0.9365
45664/60000 [=====================>........] - ETA: 25s - loss: 0.2066 - categorical_accuracy: 0.9366
45696/60000 [=====================>........] - ETA: 25s - loss: 0.2067 - categorical_accuracy: 0.9366
45728/60000 [=====================>........] - ETA: 25s - loss: 0.2068 - categorical_accuracy: 0.9366
45760/60000 [=====================>........] - ETA: 25s - loss: 0.2067 - categorical_accuracy: 0.9366
45792/60000 [=====================>........] - ETA: 25s - loss: 0.2066 - categorical_accuracy: 0.9366
45824/60000 [=====================>........] - ETA: 24s - loss: 0.2066 - categorical_accuracy: 0.9366
45856/60000 [=====================>........] - ETA: 24s - loss: 0.2065 - categorical_accuracy: 0.9366
45888/60000 [=====================>........] - ETA: 24s - loss: 0.2064 - categorical_accuracy: 0.9367
45920/60000 [=====================>........] - ETA: 24s - loss: 0.2063 - categorical_accuracy: 0.9367
45952/60000 [=====================>........] - ETA: 24s - loss: 0.2062 - categorical_accuracy: 0.9367
45984/60000 [=====================>........] - ETA: 24s - loss: 0.2062 - categorical_accuracy: 0.9367
46016/60000 [======================>.......] - ETA: 24s - loss: 0.2060 - categorical_accuracy: 0.9368
46048/60000 [======================>.......] - ETA: 24s - loss: 0.2059 - categorical_accuracy: 0.9368
46080/60000 [======================>.......] - ETA: 24s - loss: 0.2058 - categorical_accuracy: 0.9368
46112/60000 [======================>.......] - ETA: 24s - loss: 0.2058 - categorical_accuracy: 0.9368
46144/60000 [======================>.......] - ETA: 24s - loss: 0.2059 - categorical_accuracy: 0.9368
46176/60000 [======================>.......] - ETA: 24s - loss: 0.2058 - categorical_accuracy: 0.9369
46208/60000 [======================>.......] - ETA: 24s - loss: 0.2057 - categorical_accuracy: 0.9369
46240/60000 [======================>.......] - ETA: 24s - loss: 0.2056 - categorical_accuracy: 0.9369
46272/60000 [======================>.......] - ETA: 24s - loss: 0.2055 - categorical_accuracy: 0.9369
46304/60000 [======================>.......] - ETA: 24s - loss: 0.2054 - categorical_accuracy: 0.9370
46336/60000 [======================>.......] - ETA: 24s - loss: 0.2053 - categorical_accuracy: 0.9370
46368/60000 [======================>.......] - ETA: 24s - loss: 0.2052 - categorical_accuracy: 0.9370
46400/60000 [======================>.......] - ETA: 23s - loss: 0.2051 - categorical_accuracy: 0.9370
46432/60000 [======================>.......] - ETA: 23s - loss: 0.2050 - categorical_accuracy: 0.9370
46464/60000 [======================>.......] - ETA: 23s - loss: 0.2049 - categorical_accuracy: 0.9370
46496/60000 [======================>.......] - ETA: 23s - loss: 0.2048 - categorical_accuracy: 0.9371
46528/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9371
46560/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9372
46592/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9372
46624/60000 [======================>.......] - ETA: 23s - loss: 0.2048 - categorical_accuracy: 0.9371
46656/60000 [======================>.......] - ETA: 23s - loss: 0.2047 - categorical_accuracy: 0.9372
46688/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9372
46720/60000 [======================>.......] - ETA: 23s - loss: 0.2046 - categorical_accuracy: 0.9372
46752/60000 [======================>.......] - ETA: 23s - loss: 0.2045 - categorical_accuracy: 0.9372
46784/60000 [======================>.......] - ETA: 23s - loss: 0.2045 - categorical_accuracy: 0.9372
46816/60000 [======================>.......] - ETA: 23s - loss: 0.2044 - categorical_accuracy: 0.9373
46848/60000 [======================>.......] - ETA: 23s - loss: 0.2044 - categorical_accuracy: 0.9373
46880/60000 [======================>.......] - ETA: 23s - loss: 0.2043 - categorical_accuracy: 0.9373
46912/60000 [======================>.......] - ETA: 23s - loss: 0.2042 - categorical_accuracy: 0.9374
46944/60000 [======================>.......] - ETA: 22s - loss: 0.2042 - categorical_accuracy: 0.9374
46976/60000 [======================>.......] - ETA: 22s - loss: 0.2041 - categorical_accuracy: 0.9374
47008/60000 [======================>.......] - ETA: 22s - loss: 0.2041 - categorical_accuracy: 0.9374
47040/60000 [======================>.......] - ETA: 22s - loss: 0.2040 - categorical_accuracy: 0.9374
47072/60000 [======================>.......] - ETA: 22s - loss: 0.2039 - categorical_accuracy: 0.9375
47104/60000 [======================>.......] - ETA: 22s - loss: 0.2038 - categorical_accuracy: 0.9375
47136/60000 [======================>.......] - ETA: 22s - loss: 0.2037 - categorical_accuracy: 0.9376
47168/60000 [======================>.......] - ETA: 22s - loss: 0.2035 - categorical_accuracy: 0.9376
47200/60000 [======================>.......] - ETA: 22s - loss: 0.2035 - categorical_accuracy: 0.9376
47232/60000 [======================>.......] - ETA: 22s - loss: 0.2034 - categorical_accuracy: 0.9376
47264/60000 [======================>.......] - ETA: 22s - loss: 0.2033 - categorical_accuracy: 0.9377
47296/60000 [======================>.......] - ETA: 22s - loss: 0.2032 - categorical_accuracy: 0.9377
47328/60000 [======================>.......] - ETA: 22s - loss: 0.2031 - categorical_accuracy: 0.9377
47360/60000 [======================>.......] - ETA: 22s - loss: 0.2030 - categorical_accuracy: 0.9378
47392/60000 [======================>.......] - ETA: 22s - loss: 0.2029 - categorical_accuracy: 0.9378
47424/60000 [======================>.......] - ETA: 22s - loss: 0.2028 - categorical_accuracy: 0.9378
47456/60000 [======================>.......] - ETA: 22s - loss: 0.2027 - categorical_accuracy: 0.9379
47488/60000 [======================>.......] - ETA: 22s - loss: 0.2025 - categorical_accuracy: 0.9379
47520/60000 [======================>.......] - ETA: 21s - loss: 0.2026 - categorical_accuracy: 0.9379
47552/60000 [======================>.......] - ETA: 21s - loss: 0.2025 - categorical_accuracy: 0.9380
47584/60000 [======================>.......] - ETA: 21s - loss: 0.2024 - categorical_accuracy: 0.9380
47616/60000 [======================>.......] - ETA: 21s - loss: 0.2024 - categorical_accuracy: 0.9380
47648/60000 [======================>.......] - ETA: 21s - loss: 0.2023 - categorical_accuracy: 0.9380
47680/60000 [======================>.......] - ETA: 21s - loss: 0.2024 - categorical_accuracy: 0.9380
47712/60000 [======================>.......] - ETA: 21s - loss: 0.2023 - categorical_accuracy: 0.9380
47744/60000 [======================>.......] - ETA: 21s - loss: 0.2022 - categorical_accuracy: 0.9380
47776/60000 [======================>.......] - ETA: 21s - loss: 0.2021 - categorical_accuracy: 0.9381
47808/60000 [======================>.......] - ETA: 21s - loss: 0.2020 - categorical_accuracy: 0.9381
47840/60000 [======================>.......] - ETA: 21s - loss: 0.2021 - categorical_accuracy: 0.9381
47872/60000 [======================>.......] - ETA: 21s - loss: 0.2021 - categorical_accuracy: 0.9381
47904/60000 [======================>.......] - ETA: 21s - loss: 0.2020 - categorical_accuracy: 0.9381
47936/60000 [======================>.......] - ETA: 21s - loss: 0.2019 - categorical_accuracy: 0.9381
47968/60000 [======================>.......] - ETA: 21s - loss: 0.2018 - categorical_accuracy: 0.9382
48000/60000 [=======================>......] - ETA: 21s - loss: 0.2017 - categorical_accuracy: 0.9382
48032/60000 [=======================>......] - ETA: 21s - loss: 0.2016 - categorical_accuracy: 0.9383
48064/60000 [=======================>......] - ETA: 21s - loss: 0.2016 - categorical_accuracy: 0.9383
48096/60000 [=======================>......] - ETA: 20s - loss: 0.2014 - categorical_accuracy: 0.9383
48128/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9384
48160/60000 [=======================>......] - ETA: 20s - loss: 0.2013 - categorical_accuracy: 0.9384
48192/60000 [=======================>......] - ETA: 20s - loss: 0.2012 - categorical_accuracy: 0.9384
48224/60000 [=======================>......] - ETA: 20s - loss: 0.2011 - categorical_accuracy: 0.9384
48256/60000 [=======================>......] - ETA: 20s - loss: 0.2010 - categorical_accuracy: 0.9384
48288/60000 [=======================>......] - ETA: 20s - loss: 0.2009 - categorical_accuracy: 0.9385
48320/60000 [=======================>......] - ETA: 20s - loss: 0.2009 - categorical_accuracy: 0.9385
48352/60000 [=======================>......] - ETA: 20s - loss: 0.2008 - categorical_accuracy: 0.9385
48384/60000 [=======================>......] - ETA: 20s - loss: 0.2006 - categorical_accuracy: 0.9386
48416/60000 [=======================>......] - ETA: 20s - loss: 0.2006 - categorical_accuracy: 0.9386
48448/60000 [=======================>......] - ETA: 20s - loss: 0.2006 - categorical_accuracy: 0.9386
48480/60000 [=======================>......] - ETA: 20s - loss: 0.2005 - categorical_accuracy: 0.9386
48512/60000 [=======================>......] - ETA: 20s - loss: 0.2004 - categorical_accuracy: 0.9386
48544/60000 [=======================>......] - ETA: 20s - loss: 0.2004 - categorical_accuracy: 0.9387
48576/60000 [=======================>......] - ETA: 20s - loss: 0.2003 - categorical_accuracy: 0.9387
48608/60000 [=======================>......] - ETA: 20s - loss: 0.2002 - categorical_accuracy: 0.9387
48640/60000 [=======================>......] - ETA: 20s - loss: 0.2002 - categorical_accuracy: 0.9387
48672/60000 [=======================>......] - ETA: 19s - loss: 0.2001 - categorical_accuracy: 0.9388
48704/60000 [=======================>......] - ETA: 19s - loss: 0.2000 - categorical_accuracy: 0.9388
48736/60000 [=======================>......] - ETA: 19s - loss: 0.2000 - categorical_accuracy: 0.9388
48768/60000 [=======================>......] - ETA: 19s - loss: 0.1999 - categorical_accuracy: 0.9388
48800/60000 [=======================>......] - ETA: 19s - loss: 0.1998 - categorical_accuracy: 0.9388
48832/60000 [=======================>......] - ETA: 19s - loss: 0.1997 - categorical_accuracy: 0.9389
48864/60000 [=======================>......] - ETA: 19s - loss: 0.1998 - categorical_accuracy: 0.9389
48896/60000 [=======================>......] - ETA: 19s - loss: 0.1998 - categorical_accuracy: 0.9388
48928/60000 [=======================>......] - ETA: 19s - loss: 0.1996 - categorical_accuracy: 0.9389
48960/60000 [=======================>......] - ETA: 19s - loss: 0.1995 - categorical_accuracy: 0.9389
48992/60000 [=======================>......] - ETA: 19s - loss: 0.1995 - categorical_accuracy: 0.9389
49024/60000 [=======================>......] - ETA: 19s - loss: 0.1995 - categorical_accuracy: 0.9389
49056/60000 [=======================>......] - ETA: 19s - loss: 0.1995 - categorical_accuracy: 0.9389
49088/60000 [=======================>......] - ETA: 19s - loss: 0.1995 - categorical_accuracy: 0.9389
49120/60000 [=======================>......] - ETA: 19s - loss: 0.1994 - categorical_accuracy: 0.9390
49152/60000 [=======================>......] - ETA: 19s - loss: 0.1997 - categorical_accuracy: 0.9390
49184/60000 [=======================>......] - ETA: 19s - loss: 0.1996 - categorical_accuracy: 0.9390
49216/60000 [=======================>......] - ETA: 18s - loss: 0.1995 - categorical_accuracy: 0.9390
49248/60000 [=======================>......] - ETA: 18s - loss: 0.1995 - categorical_accuracy: 0.9390
49280/60000 [=======================>......] - ETA: 18s - loss: 0.1994 - categorical_accuracy: 0.9391
49312/60000 [=======================>......] - ETA: 18s - loss: 0.1993 - categorical_accuracy: 0.9391
49344/60000 [=======================>......] - ETA: 18s - loss: 0.1992 - categorical_accuracy: 0.9391
49376/60000 [=======================>......] - ETA: 18s - loss: 0.1991 - categorical_accuracy: 0.9391
49408/60000 [=======================>......] - ETA: 18s - loss: 0.1990 - categorical_accuracy: 0.9392
49440/60000 [=======================>......] - ETA: 18s - loss: 0.1989 - categorical_accuracy: 0.9392
49472/60000 [=======================>......] - ETA: 18s - loss: 0.1989 - categorical_accuracy: 0.9392
49504/60000 [=======================>......] - ETA: 18s - loss: 0.1994 - categorical_accuracy: 0.9392
49536/60000 [=======================>......] - ETA: 18s - loss: 0.1993 - categorical_accuracy: 0.9392
49568/60000 [=======================>......] - ETA: 18s - loss: 0.1992 - categorical_accuracy: 0.9392
49600/60000 [=======================>......] - ETA: 18s - loss: 0.1992 - categorical_accuracy: 0.9392
49632/60000 [=======================>......] - ETA: 18s - loss: 0.1991 - categorical_accuracy: 0.9393
49664/60000 [=======================>......] - ETA: 18s - loss: 0.1991 - categorical_accuracy: 0.9393
49696/60000 [=======================>......] - ETA: 18s - loss: 0.1990 - categorical_accuracy: 0.9393
49728/60000 [=======================>......] - ETA: 18s - loss: 0.1989 - categorical_accuracy: 0.9393
49760/60000 [=======================>......] - ETA: 18s - loss: 0.1989 - categorical_accuracy: 0.9393
49792/60000 [=======================>......] - ETA: 17s - loss: 0.1988 - categorical_accuracy: 0.9394
49824/60000 [=======================>......] - ETA: 17s - loss: 0.1987 - categorical_accuracy: 0.9394
49856/60000 [=======================>......] - ETA: 17s - loss: 0.1986 - categorical_accuracy: 0.9394
49888/60000 [=======================>......] - ETA: 17s - loss: 0.1985 - categorical_accuracy: 0.9395
49920/60000 [=======================>......] - ETA: 17s - loss: 0.1984 - categorical_accuracy: 0.9394
49952/60000 [=======================>......] - ETA: 17s - loss: 0.1984 - categorical_accuracy: 0.9395
49984/60000 [=======================>......] - ETA: 17s - loss: 0.1983 - categorical_accuracy: 0.9395
50016/60000 [========================>.....] - ETA: 17s - loss: 0.1982 - categorical_accuracy: 0.9395
50048/60000 [========================>.....] - ETA: 17s - loss: 0.1981 - categorical_accuracy: 0.9395
50080/60000 [========================>.....] - ETA: 17s - loss: 0.1980 - categorical_accuracy: 0.9395
50112/60000 [========================>.....] - ETA: 17s - loss: 0.1979 - categorical_accuracy: 0.9396
50144/60000 [========================>.....] - ETA: 17s - loss: 0.1978 - categorical_accuracy: 0.9396
50176/60000 [========================>.....] - ETA: 17s - loss: 0.1977 - categorical_accuracy: 0.9396
50208/60000 [========================>.....] - ETA: 17s - loss: 0.1976 - categorical_accuracy: 0.9397
50240/60000 [========================>.....] - ETA: 17s - loss: 0.1975 - categorical_accuracy: 0.9397
50272/60000 [========================>.....] - ETA: 17s - loss: 0.1975 - categorical_accuracy: 0.9396
50304/60000 [========================>.....] - ETA: 17s - loss: 0.1974 - categorical_accuracy: 0.9397
50336/60000 [========================>.....] - ETA: 17s - loss: 0.1973 - categorical_accuracy: 0.9397
50368/60000 [========================>.....] - ETA: 16s - loss: 0.1973 - categorical_accuracy: 0.9397
50400/60000 [========================>.....] - ETA: 16s - loss: 0.1973 - categorical_accuracy: 0.9397
50432/60000 [========================>.....] - ETA: 16s - loss: 0.1972 - categorical_accuracy: 0.9398
50464/60000 [========================>.....] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9398
50496/60000 [========================>.....] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9398
50528/60000 [========================>.....] - ETA: 16s - loss: 0.1971 - categorical_accuracy: 0.9398
50560/60000 [========================>.....] - ETA: 16s - loss: 0.1970 - categorical_accuracy: 0.9398
50592/60000 [========================>.....] - ETA: 16s - loss: 0.1969 - categorical_accuracy: 0.9398
50624/60000 [========================>.....] - ETA: 16s - loss: 0.1970 - categorical_accuracy: 0.9398
50656/60000 [========================>.....] - ETA: 16s - loss: 0.1969 - categorical_accuracy: 0.9398
50688/60000 [========================>.....] - ETA: 16s - loss: 0.1968 - categorical_accuracy: 0.9399
50720/60000 [========================>.....] - ETA: 16s - loss: 0.1967 - categorical_accuracy: 0.9399
50752/60000 [========================>.....] - ETA: 16s - loss: 0.1966 - categorical_accuracy: 0.9399
50784/60000 [========================>.....] - ETA: 16s - loss: 0.1965 - categorical_accuracy: 0.9400
50816/60000 [========================>.....] - ETA: 16s - loss: 0.1964 - categorical_accuracy: 0.9400
50848/60000 [========================>.....] - ETA: 16s - loss: 0.1963 - categorical_accuracy: 0.9401
50880/60000 [========================>.....] - ETA: 16s - loss: 0.1962 - categorical_accuracy: 0.9401
50912/60000 [========================>.....] - ETA: 16s - loss: 0.1961 - categorical_accuracy: 0.9401
50944/60000 [========================>.....] - ETA: 15s - loss: 0.1960 - categorical_accuracy: 0.9401
50976/60000 [========================>.....] - ETA: 15s - loss: 0.1959 - categorical_accuracy: 0.9401
51008/60000 [========================>.....] - ETA: 15s - loss: 0.1959 - categorical_accuracy: 0.9401
51040/60000 [========================>.....] - ETA: 15s - loss: 0.1960 - categorical_accuracy: 0.9401
51072/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9402
51104/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9402
51136/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9402
51168/60000 [========================>.....] - ETA: 15s - loss: 0.1958 - categorical_accuracy: 0.9402
51200/60000 [========================>.....] - ETA: 15s - loss: 0.1957 - categorical_accuracy: 0.9403
51232/60000 [========================>.....] - ETA: 15s - loss: 0.1956 - categorical_accuracy: 0.9403
51264/60000 [========================>.....] - ETA: 15s - loss: 0.1955 - categorical_accuracy: 0.9403
51296/60000 [========================>.....] - ETA: 15s - loss: 0.1954 - categorical_accuracy: 0.9403
51328/60000 [========================>.....] - ETA: 15s - loss: 0.1953 - categorical_accuracy: 0.9404
51360/60000 [========================>.....] - ETA: 15s - loss: 0.1957 - categorical_accuracy: 0.9403
51392/60000 [========================>.....] - ETA: 15s - loss: 0.1956 - categorical_accuracy: 0.9403
51424/60000 [========================>.....] - ETA: 15s - loss: 0.1956 - categorical_accuracy: 0.9403
51456/60000 [========================>.....] - ETA: 15s - loss: 0.1955 - categorical_accuracy: 0.9404
51488/60000 [========================>.....] - ETA: 14s - loss: 0.1955 - categorical_accuracy: 0.9404
51520/60000 [========================>.....] - ETA: 14s - loss: 0.1954 - categorical_accuracy: 0.9404
51552/60000 [========================>.....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9404
51584/60000 [========================>.....] - ETA: 14s - loss: 0.1953 - categorical_accuracy: 0.9404
51616/60000 [========================>.....] - ETA: 14s - loss: 0.1952 - categorical_accuracy: 0.9404
51648/60000 [========================>.....] - ETA: 14s - loss: 0.1951 - categorical_accuracy: 0.9404
51680/60000 [========================>.....] - ETA: 14s - loss: 0.1950 - categorical_accuracy: 0.9405
51712/60000 [========================>.....] - ETA: 14s - loss: 0.1949 - categorical_accuracy: 0.9405
51744/60000 [========================>.....] - ETA: 14s - loss: 0.1949 - categorical_accuracy: 0.9405
51776/60000 [========================>.....] - ETA: 14s - loss: 0.1948 - categorical_accuracy: 0.9405
51808/60000 [========================>.....] - ETA: 14s - loss: 0.1947 - categorical_accuracy: 0.9405
51840/60000 [========================>.....] - ETA: 14s - loss: 0.1946 - categorical_accuracy: 0.9406
51872/60000 [========================>.....] - ETA: 14s - loss: 0.1945 - categorical_accuracy: 0.9406
51904/60000 [========================>.....] - ETA: 14s - loss: 0.1944 - categorical_accuracy: 0.9406
51936/60000 [========================>.....] - ETA: 14s - loss: 0.1943 - categorical_accuracy: 0.9407
51968/60000 [========================>.....] - ETA: 14s - loss: 0.1943 - categorical_accuracy: 0.9407
52000/60000 [=========================>....] - ETA: 14s - loss: 0.1943 - categorical_accuracy: 0.9407
52032/60000 [=========================>....] - ETA: 14s - loss: 0.1944 - categorical_accuracy: 0.9407
52064/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9407
52096/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9407
52128/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9407
52160/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9407
52192/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9407
52224/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9407
52256/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9407
52288/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9408
52320/60000 [=========================>....] - ETA: 13s - loss: 0.1943 - categorical_accuracy: 0.9407
52352/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9408
52384/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9408
52416/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9408
52448/60000 [=========================>....] - ETA: 13s - loss: 0.1940 - categorical_accuracy: 0.9408
52480/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9408
52512/60000 [=========================>....] - ETA: 13s - loss: 0.1942 - categorical_accuracy: 0.9408
52544/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9408
52576/60000 [=========================>....] - ETA: 13s - loss: 0.1941 - categorical_accuracy: 0.9408
52608/60000 [=========================>....] - ETA: 13s - loss: 0.1940 - categorical_accuracy: 0.9408
52640/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9409
52672/60000 [=========================>....] - ETA: 12s - loss: 0.1939 - categorical_accuracy: 0.9409
52704/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9409
52736/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9409
52768/60000 [=========================>....] - ETA: 12s - loss: 0.1938 - categorical_accuracy: 0.9409
52800/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9409
52832/60000 [=========================>....] - ETA: 12s - loss: 0.1937 - categorical_accuracy: 0.9409
52864/60000 [=========================>....] - ETA: 12s - loss: 0.1936 - categorical_accuracy: 0.9410
52896/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9410
52928/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9410
52960/60000 [=========================>....] - ETA: 12s - loss: 0.1935 - categorical_accuracy: 0.9410
52992/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9411
53024/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9411
53056/60000 [=========================>....] - ETA: 12s - loss: 0.1934 - categorical_accuracy: 0.9411
53088/60000 [=========================>....] - ETA: 12s - loss: 0.1933 - categorical_accuracy: 0.9411
53120/60000 [=========================>....] - ETA: 12s - loss: 0.1932 - categorical_accuracy: 0.9411
53152/60000 [=========================>....] - ETA: 12s - loss: 0.1932 - categorical_accuracy: 0.9411
53184/60000 [=========================>....] - ETA: 11s - loss: 0.1931 - categorical_accuracy: 0.9412
53216/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9412
53248/60000 [=========================>....] - ETA: 11s - loss: 0.1931 - categorical_accuracy: 0.9412
53280/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9412
53312/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9413
53344/60000 [=========================>....] - ETA: 11s - loss: 0.1929 - categorical_accuracy: 0.9413
53376/60000 [=========================>....] - ETA: 11s - loss: 0.1930 - categorical_accuracy: 0.9413
53408/60000 [=========================>....] - ETA: 11s - loss: 0.1929 - categorical_accuracy: 0.9413
53440/60000 [=========================>....] - ETA: 11s - loss: 0.1928 - categorical_accuracy: 0.9413
53472/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9413
53504/60000 [=========================>....] - ETA: 11s - loss: 0.1927 - categorical_accuracy: 0.9414
53536/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9414
53568/60000 [=========================>....] - ETA: 11s - loss: 0.1926 - categorical_accuracy: 0.9414
53600/60000 [=========================>....] - ETA: 11s - loss: 0.1925 - categorical_accuracy: 0.9414
53632/60000 [=========================>....] - ETA: 11s - loss: 0.1924 - categorical_accuracy: 0.9414
53664/60000 [=========================>....] - ETA: 11s - loss: 0.1924 - categorical_accuracy: 0.9414
53696/60000 [=========================>....] - ETA: 11s - loss: 0.1923 - categorical_accuracy: 0.9414
53728/60000 [=========================>....] - ETA: 11s - loss: 0.1922 - categorical_accuracy: 0.9415
53760/60000 [=========================>....] - ETA: 10s - loss: 0.1921 - categorical_accuracy: 0.9415
53792/60000 [=========================>....] - ETA: 10s - loss: 0.1920 - categorical_accuracy: 0.9415
53824/60000 [=========================>....] - ETA: 10s - loss: 0.1919 - categorical_accuracy: 0.9416
53856/60000 [=========================>....] - ETA: 10s - loss: 0.1918 - categorical_accuracy: 0.9416
53888/60000 [=========================>....] - ETA: 10s - loss: 0.1917 - categorical_accuracy: 0.9416
53920/60000 [=========================>....] - ETA: 10s - loss: 0.1916 - categorical_accuracy: 0.9417
53952/60000 [=========================>....] - ETA: 10s - loss: 0.1917 - categorical_accuracy: 0.9417
53984/60000 [=========================>....] - ETA: 10s - loss: 0.1917 - categorical_accuracy: 0.9417
54016/60000 [==========================>...] - ETA: 10s - loss: 0.1917 - categorical_accuracy: 0.9417
54048/60000 [==========================>...] - ETA: 10s - loss: 0.1916 - categorical_accuracy: 0.9418
54080/60000 [==========================>...] - ETA: 10s - loss: 0.1915 - categorical_accuracy: 0.9418
54112/60000 [==========================>...] - ETA: 10s - loss: 0.1914 - categorical_accuracy: 0.9418
54144/60000 [==========================>...] - ETA: 10s - loss: 0.1913 - categorical_accuracy: 0.9419
54176/60000 [==========================>...] - ETA: 10s - loss: 0.1912 - categorical_accuracy: 0.9419
54208/60000 [==========================>...] - ETA: 10s - loss: 0.1911 - categorical_accuracy: 0.9419
54240/60000 [==========================>...] - ETA: 10s - loss: 0.1910 - categorical_accuracy: 0.9419
54272/60000 [==========================>...] - ETA: 10s - loss: 0.1909 - categorical_accuracy: 0.9420
54304/60000 [==========================>...] - ETA: 10s - loss: 0.1908 - categorical_accuracy: 0.9420
54336/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9420 
54368/60000 [==========================>...] - ETA: 9s - loss: 0.1907 - categorical_accuracy: 0.9420
54400/60000 [==========================>...] - ETA: 9s - loss: 0.1906 - categorical_accuracy: 0.9421
54432/60000 [==========================>...] - ETA: 9s - loss: 0.1906 - categorical_accuracy: 0.9421
54464/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9421
54496/60000 [==========================>...] - ETA: 9s - loss: 0.1905 - categorical_accuracy: 0.9421
54528/60000 [==========================>...] - ETA: 9s - loss: 0.1904 - categorical_accuracy: 0.9421
54560/60000 [==========================>...] - ETA: 9s - loss: 0.1903 - categorical_accuracy: 0.9422
54592/60000 [==========================>...] - ETA: 9s - loss: 0.1903 - categorical_accuracy: 0.9422
54624/60000 [==========================>...] - ETA: 9s - loss: 0.1902 - categorical_accuracy: 0.9422
54656/60000 [==========================>...] - ETA: 9s - loss: 0.1902 - categorical_accuracy: 0.9422
54688/60000 [==========================>...] - ETA: 9s - loss: 0.1901 - categorical_accuracy: 0.9422
54720/60000 [==========================>...] - ETA: 9s - loss: 0.1901 - categorical_accuracy: 0.9423
54752/60000 [==========================>...] - ETA: 9s - loss: 0.1900 - categorical_accuracy: 0.9423
54784/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9423
54816/60000 [==========================>...] - ETA: 9s - loss: 0.1899 - categorical_accuracy: 0.9423
54848/60000 [==========================>...] - ETA: 9s - loss: 0.1898 - categorical_accuracy: 0.9423
54880/60000 [==========================>...] - ETA: 9s - loss: 0.1897 - categorical_accuracy: 0.9423
54912/60000 [==========================>...] - ETA: 8s - loss: 0.1896 - categorical_accuracy: 0.9424
54944/60000 [==========================>...] - ETA: 8s - loss: 0.1895 - categorical_accuracy: 0.9424
54976/60000 [==========================>...] - ETA: 8s - loss: 0.1895 - categorical_accuracy: 0.9424
55008/60000 [==========================>...] - ETA: 8s - loss: 0.1895 - categorical_accuracy: 0.9424
55040/60000 [==========================>...] - ETA: 8s - loss: 0.1894 - categorical_accuracy: 0.9424
55072/60000 [==========================>...] - ETA: 8s - loss: 0.1894 - categorical_accuracy: 0.9424
55104/60000 [==========================>...] - ETA: 8s - loss: 0.1893 - categorical_accuracy: 0.9425
55136/60000 [==========================>...] - ETA: 8s - loss: 0.1892 - categorical_accuracy: 0.9425
55168/60000 [==========================>...] - ETA: 8s - loss: 0.1892 - categorical_accuracy: 0.9425
55200/60000 [==========================>...] - ETA: 8s - loss: 0.1892 - categorical_accuracy: 0.9425
55232/60000 [==========================>...] - ETA: 8s - loss: 0.1891 - categorical_accuracy: 0.9425
55264/60000 [==========================>...] - ETA: 8s - loss: 0.1890 - categorical_accuracy: 0.9425
55296/60000 [==========================>...] - ETA: 8s - loss: 0.1889 - categorical_accuracy: 0.9425
55328/60000 [==========================>...] - ETA: 8s - loss: 0.1888 - categorical_accuracy: 0.9426
55360/60000 [==========================>...] - ETA: 8s - loss: 0.1888 - categorical_accuracy: 0.9426
55392/60000 [==========================>...] - ETA: 8s - loss: 0.1888 - categorical_accuracy: 0.9426
55424/60000 [==========================>...] - ETA: 8s - loss: 0.1887 - categorical_accuracy: 0.9426
55456/60000 [==========================>...] - ETA: 8s - loss: 0.1886 - categorical_accuracy: 0.9427
55488/60000 [==========================>...] - ETA: 7s - loss: 0.1884 - categorical_accuracy: 0.9427
55520/60000 [==========================>...] - ETA: 7s - loss: 0.1883 - categorical_accuracy: 0.9427
55552/60000 [==========================>...] - ETA: 7s - loss: 0.1883 - categorical_accuracy: 0.9427
55584/60000 [==========================>...] - ETA: 7s - loss: 0.1883 - categorical_accuracy: 0.9427
55616/60000 [==========================>...] - ETA: 7s - loss: 0.1882 - categorical_accuracy: 0.9428
55648/60000 [==========================>...] - ETA: 7s - loss: 0.1882 - categorical_accuracy: 0.9428
55680/60000 [==========================>...] - ETA: 7s - loss: 0.1881 - categorical_accuracy: 0.9428
55712/60000 [==========================>...] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9428
55744/60000 [==========================>...] - ETA: 7s - loss: 0.1880 - categorical_accuracy: 0.9428
55776/60000 [==========================>...] - ETA: 7s - loss: 0.1879 - categorical_accuracy: 0.9429
55808/60000 [==========================>...] - ETA: 7s - loss: 0.1878 - categorical_accuracy: 0.9429
55840/60000 [==========================>...] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9429
55872/60000 [==========================>...] - ETA: 7s - loss: 0.1878 - categorical_accuracy: 0.9429
55904/60000 [==========================>...] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9429
55936/60000 [==========================>...] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9429
55968/60000 [==========================>...] - ETA: 7s - loss: 0.1877 - categorical_accuracy: 0.9429
56000/60000 [===========================>..] - ETA: 7s - loss: 0.1876 - categorical_accuracy: 0.9429
56032/60000 [===========================>..] - ETA: 6s - loss: 0.1876 - categorical_accuracy: 0.9429
56064/60000 [===========================>..] - ETA: 6s - loss: 0.1875 - categorical_accuracy: 0.9430
56096/60000 [===========================>..] - ETA: 6s - loss: 0.1874 - categorical_accuracy: 0.9430
56128/60000 [===========================>..] - ETA: 6s - loss: 0.1873 - categorical_accuracy: 0.9430
56160/60000 [===========================>..] - ETA: 6s - loss: 0.1872 - categorical_accuracy: 0.9430
56192/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9431
56224/60000 [===========================>..] - ETA: 6s - loss: 0.1871 - categorical_accuracy: 0.9431
56256/60000 [===========================>..] - ETA: 6s - loss: 0.1870 - categorical_accuracy: 0.9431
56288/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9431
56320/60000 [===========================>..] - ETA: 6s - loss: 0.1869 - categorical_accuracy: 0.9431
56352/60000 [===========================>..] - ETA: 6s - loss: 0.1868 - categorical_accuracy: 0.9431
56384/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9432
56416/60000 [===========================>..] - ETA: 6s - loss: 0.1867 - categorical_accuracy: 0.9432
56448/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9432
56480/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9432
56512/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9433
56544/60000 [===========================>..] - ETA: 6s - loss: 0.1866 - categorical_accuracy: 0.9432
56576/60000 [===========================>..] - ETA: 6s - loss: 0.1865 - categorical_accuracy: 0.9433
56608/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9433
56640/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9433
56672/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9433
56704/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9433
56736/60000 [===========================>..] - ETA: 5s - loss: 0.1864 - categorical_accuracy: 0.9433
56768/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9433
56800/60000 [===========================>..] - ETA: 5s - loss: 0.1863 - categorical_accuracy: 0.9434
56832/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9434
56864/60000 [===========================>..] - ETA: 5s - loss: 0.1862 - categorical_accuracy: 0.9434
56896/60000 [===========================>..] - ETA: 5s - loss: 0.1861 - categorical_accuracy: 0.9434
56928/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9435
56960/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9434
56992/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9434
57024/60000 [===========================>..] - ETA: 5s - loss: 0.1859 - categorical_accuracy: 0.9435
57056/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9435
57088/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9435
57120/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9435
57152/60000 [===========================>..] - ETA: 5s - loss: 0.1860 - categorical_accuracy: 0.9435
57184/60000 [===========================>..] - ETA: 4s - loss: 0.1860 - categorical_accuracy: 0.9436
57216/60000 [===========================>..] - ETA: 4s - loss: 0.1859 - categorical_accuracy: 0.9435
57248/60000 [===========================>..] - ETA: 4s - loss: 0.1859 - categorical_accuracy: 0.9435
57280/60000 [===========================>..] - ETA: 4s - loss: 0.1859 - categorical_accuracy: 0.9435
57312/60000 [===========================>..] - ETA: 4s - loss: 0.1858 - categorical_accuracy: 0.9435
57344/60000 [===========================>..] - ETA: 4s - loss: 0.1858 - categorical_accuracy: 0.9435
57376/60000 [===========================>..] - ETA: 4s - loss: 0.1859 - categorical_accuracy: 0.9435
57408/60000 [===========================>..] - ETA: 4s - loss: 0.1858 - categorical_accuracy: 0.9436
57440/60000 [===========================>..] - ETA: 4s - loss: 0.1857 - categorical_accuracy: 0.9436
57472/60000 [===========================>..] - ETA: 4s - loss: 0.1856 - categorical_accuracy: 0.9436
57504/60000 [===========================>..] - ETA: 4s - loss: 0.1856 - categorical_accuracy: 0.9437
57536/60000 [===========================>..] - ETA: 4s - loss: 0.1855 - categorical_accuracy: 0.9437
57568/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9437
57600/60000 [===========================>..] - ETA: 4s - loss: 0.1854 - categorical_accuracy: 0.9437
57632/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9437
57664/60000 [===========================>..] - ETA: 4s - loss: 0.1853 - categorical_accuracy: 0.9438
57696/60000 [===========================>..] - ETA: 4s - loss: 0.1852 - categorical_accuracy: 0.9438
57728/60000 [===========================>..] - ETA: 3s - loss: 0.1853 - categorical_accuracy: 0.9438
57760/60000 [===========================>..] - ETA: 3s - loss: 0.1852 - categorical_accuracy: 0.9438
57792/60000 [===========================>..] - ETA: 3s - loss: 0.1851 - categorical_accuracy: 0.9438
57824/60000 [===========================>..] - ETA: 3s - loss: 0.1850 - categorical_accuracy: 0.9438
57856/60000 [===========================>..] - ETA: 3s - loss: 0.1850 - categorical_accuracy: 0.9439
57888/60000 [===========================>..] - ETA: 3s - loss: 0.1849 - categorical_accuracy: 0.9439
57920/60000 [===========================>..] - ETA: 3s - loss: 0.1849 - categorical_accuracy: 0.9439
57952/60000 [===========================>..] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9439
57984/60000 [===========================>..] - ETA: 3s - loss: 0.1848 - categorical_accuracy: 0.9439
58016/60000 [============================>.] - ETA: 3s - loss: 0.1847 - categorical_accuracy: 0.9440
58048/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9440
58080/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9440
58112/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9440
58144/60000 [============================>.] - ETA: 3s - loss: 0.1846 - categorical_accuracy: 0.9441
58176/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9441
58208/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9441
58240/60000 [============================>.] - ETA: 3s - loss: 0.1845 - categorical_accuracy: 0.9441
58272/60000 [============================>.] - ETA: 3s - loss: 0.1844 - categorical_accuracy: 0.9441
58304/60000 [============================>.] - ETA: 2s - loss: 0.1843 - categorical_accuracy: 0.9442
58336/60000 [============================>.] - ETA: 2s - loss: 0.1842 - categorical_accuracy: 0.9442
58368/60000 [============================>.] - ETA: 2s - loss: 0.1841 - categorical_accuracy: 0.9442
58400/60000 [============================>.] - ETA: 2s - loss: 0.1841 - categorical_accuracy: 0.9442
58432/60000 [============================>.] - ETA: 2s - loss: 0.1840 - categorical_accuracy: 0.9442
58464/60000 [============================>.] - ETA: 2s - loss: 0.1839 - categorical_accuracy: 0.9443
58496/60000 [============================>.] - ETA: 2s - loss: 0.1838 - categorical_accuracy: 0.9443
58528/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9443
58560/60000 [============================>.] - ETA: 2s - loss: 0.1837 - categorical_accuracy: 0.9443
58592/60000 [============================>.] - ETA: 2s - loss: 0.1836 - categorical_accuracy: 0.9444
58624/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9444
58656/60000 [============================>.] - ETA: 2s - loss: 0.1835 - categorical_accuracy: 0.9444
58688/60000 [============================>.] - ETA: 2s - loss: 0.1834 - categorical_accuracy: 0.9445
58720/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9445
58752/60000 [============================>.] - ETA: 2s - loss: 0.1832 - categorical_accuracy: 0.9445
58784/60000 [============================>.] - ETA: 2s - loss: 0.1833 - categorical_accuracy: 0.9445
58816/60000 [============================>.] - ETA: 2s - loss: 0.1832 - categorical_accuracy: 0.9445
58848/60000 [============================>.] - ETA: 2s - loss: 0.1831 - categorical_accuracy: 0.9446
58880/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9446
58912/60000 [============================>.] - ETA: 1s - loss: 0.1831 - categorical_accuracy: 0.9445
58944/60000 [============================>.] - ETA: 1s - loss: 0.1830 - categorical_accuracy: 0.9446
58976/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9446
59008/60000 [============================>.] - ETA: 1s - loss: 0.1829 - categorical_accuracy: 0.9446
59040/60000 [============================>.] - ETA: 1s - loss: 0.1828 - categorical_accuracy: 0.9446
59072/60000 [============================>.] - ETA: 1s - loss: 0.1827 - categorical_accuracy: 0.9447
59104/60000 [============================>.] - ETA: 1s - loss: 0.1826 - categorical_accuracy: 0.9447
59136/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9447
59168/60000 [============================>.] - ETA: 1s - loss: 0.1825 - categorical_accuracy: 0.9447
59200/60000 [============================>.] - ETA: 1s - loss: 0.1824 - categorical_accuracy: 0.9448
59232/60000 [============================>.] - ETA: 1s - loss: 0.1824 - categorical_accuracy: 0.9448
59264/60000 [============================>.] - ETA: 1s - loss: 0.1823 - categorical_accuracy: 0.9448
59296/60000 [============================>.] - ETA: 1s - loss: 0.1823 - categorical_accuracy: 0.9448
59328/60000 [============================>.] - ETA: 1s - loss: 0.1823 - categorical_accuracy: 0.9448
59360/60000 [============================>.] - ETA: 1s - loss: 0.1822 - categorical_accuracy: 0.9448
59392/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9448
59424/60000 [============================>.] - ETA: 1s - loss: 0.1821 - categorical_accuracy: 0.9449
59456/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9448
59488/60000 [============================>.] - ETA: 0s - loss: 0.1821 - categorical_accuracy: 0.9448
59520/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9449
59552/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9449
59584/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9449
59616/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9449
59648/60000 [============================>.] - ETA: 0s - loss: 0.1820 - categorical_accuracy: 0.9449
59680/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9449
59712/60000 [============================>.] - ETA: 0s - loss: 0.1819 - categorical_accuracy: 0.9449
59744/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9449
59776/60000 [============================>.] - ETA: 0s - loss: 0.1818 - categorical_accuracy: 0.9449
59808/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9450
59840/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9450
59872/60000 [============================>.] - ETA: 0s - loss: 0.1817 - categorical_accuracy: 0.9450
59904/60000 [============================>.] - ETA: 0s - loss: 0.1816 - categorical_accuracy: 0.9450
59936/60000 [============================>.] - ETA: 0s - loss: 0.1815 - categorical_accuracy: 0.9450
59968/60000 [============================>.] - ETA: 0s - loss: 0.1814 - categorical_accuracy: 0.9451
60000/60000 [==============================] - 110s 2ms/step - loss: 0.1814 - categorical_accuracy: 0.9451 - val_loss: 0.0504 - val_categorical_accuracy: 0.9844

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
 1152/10000 [==>...........................] - ETA: 3s
 1312/10000 [==>...........................] - ETA: 3s
 1472/10000 [===>..........................] - ETA: 3s
 1632/10000 [===>..........................] - ETA: 3s
 1792/10000 [====>.........................] - ETA: 2s
 1952/10000 [====>.........................] - ETA: 2s
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
 4320/10000 [===========>..................] - ETA: 2s
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
 6080/10000 [=================>............] - ETA: 1s
 6240/10000 [=================>............] - ETA: 1s
 6400/10000 [==================>...........] - ETA: 1s
 6560/10000 [==================>...........] - ETA: 1s
 6720/10000 [===================>..........] - ETA: 1s
 6880/10000 [===================>..........] - ETA: 1s
 7040/10000 [====================>.........] - ETA: 1s
 7200/10000 [====================>.........] - ETA: 0s
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
10000/10000 [==============================] - 3s 344us/step
[[1.0077069e-08 1.2208661e-08 1.6824991e-06 ... 9.9999678e-01
  2.0227512e-08 5.3742838e-07]
 [8.1161517e-05 1.0675905e-05 9.9988365e-01 ... 1.7567766e-07
  1.1399361e-05 3.8139842e-08]
 [1.2174197e-06 9.9985826e-01 1.4674407e-05 ... 4.5685345e-05
  4.0555766e-05 6.9726690e-07]
 ...
 [3.0948235e-09 9.3625704e-07 3.2196564e-08 ... 7.2214607e-07
  1.2121243e-06 4.5048495e-05]
 [2.4700396e-06 2.7816023e-07 3.3424122e-08 ... 6.5922698e-08
  2.0906373e-03 1.7470587e-05]
 [2.6470534e-06 3.0539934e-07 3.2797598e-06 ... 1.2411323e-09
  8.3897692e-07 3.4566008e-07]]

  ('#### metrics   ####################################################',) 

  ('#### Path params   ################################################',) 

  ('/home/runner/work/mlmodels/mlmodels/mlmodels/model_keras/', '/home/runner/work/mlmodels/mlmodels/keras_deepAR/') 
{'loss_test:': 0.050350028115848544, 'accuracy_test:': 0.9843999743461609}

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
[master d458017] ml_store  && git pull --all
 1 file changed, 2036 insertions(+)
To github.com:arita37/mlmodels_store.git
 + d83769d...d458017 master -> master (forced update)





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
{'loss': 0.5808363854885101, 'loss_history': []}

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
[master ec08557] ml_store  && git pull --all
 1 file changed, 111 insertions(+)
To github.com:arita37/mlmodels_store.git
   d458017..ec08557  master -> master





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
[master a703d0b] ml_store  && git pull --all
 1 file changed, 34 insertions(+)
To github.com:arita37/mlmodels_store.git
   ec08557..a703d0b  master -> master





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
 40%|████      | 2/5 [00:17<00:26,  8.68s/it]Saving dataset/models/LightGBMClassifier/trial_1_model.pkl
Finished Task with config: {'feature_fraction': 0.9053644758201441, 'learning_rate': 0.00974234984367248, 'min_data_in_leaf': 3, 'num_leaves': 36} and reward: 0.3882
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xf8\xbe\xeb\xd3rKX\r\x00\x00\x00learning_rateq\x02G?\x83\xf3\xcc\x0f\xb7\xd62X\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3882
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xec\xf8\xbe\xeb\xd3rKX\r\x00\x00\x00learning_rateq\x02G?\x83\xf3\xcc\x0f\xb7\xd62X\x10\x00\x00\x00min_data_in_leafq\x03K\x03X\n\x00\x00\x00num_leavesq\x04K$u.' and reward: 0.3882
 60%|██████    | 3/5 [00:35<00:22, 11.39s/it]Saving dataset/models/LightGBMClassifier/trial_2_model.pkl
Finished Task with config: {'feature_fraction': 0.9988838141966151, 'learning_rate': 0.18565226230391002, 'min_data_in_leaf': 23, 'num_leaves': 57} and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\xf6\xdb0OMoX\r\x00\x00\x00learning_rateq\x02G?\xc7\xc3t\r\x83\x08\xd6X\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3924
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xef\xf6\xdb0OMoX\r\x00\x00\x00learning_rateq\x02G?\xc7\xc3t\r\x83\x08\xd6X\x10\x00\x00\x00min_data_in_leafq\x03K\x17X\n\x00\x00\x00num_leavesq\x04K9u.' and reward: 0.3924
 80%|████████  | 4/5 [00:56<00:14, 14.28s/it] 80%|████████  | 4/5 [00:56<00:14, 14.02s/it]
Saving dataset/models/LightGBMClassifier/trial_3_model.pkl
Finished Task with config: {'feature_fraction': 0.9231805418288291, 'learning_rate': 0.16026700516780276, 'min_data_in_leaf': 20, 'num_leaves': 44} and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x8a\xb1\xebn\xab\x0fX\r\x00\x00\x00learning_rateq\x02G?\xc4\x83\xa1\x14\xe9j\xefX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K,u.' and reward: 0.3902
Finished Task with config: b'\x80\x03}q\x00(X\x10\x00\x00\x00feature_fractionq\x01G?\xed\x8a\xb1\xebn\xab\x0fX\r\x00\x00\x00learning_rateq\x02G?\xc4\x83\xa1\x14\xe9j\xefX\x10\x00\x00\x00min_data_in_leafq\x03K\x14X\n\x00\x00\x00num_leavesq\x04K,u.' and reward: 0.3902
Time for Gradient Boosting hyperparameter optimization: 74.3324830532074
Best hyperparameter configuration for Gradient Boosting Model: 
{'feature_fraction': 0.9988838141966151, 'learning_rate': 0.18565226230391002, 'min_data_in_leaf': 23, 'num_leaves': 57}
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
 40%|████      | 2/5 [00:53<01:20, 26.94s/it]Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_5_tabularNN.pkl
Finished Task with config: {'activation.choice': 0, 'dropout_prob': 0.3289309573677045, 'embedding_size_factor': 0.9354298767249828, 'layers.choice': 0, 'learning_rate': 0.0010415068444658434, 'network_type.choice': 0, 'use_batchnorm.choice': 0, 'weight_decay': 1.3621258755668652e-10} and reward: 0.3848
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\r4n"R\x1aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\xef\n\xa3\x07\x86\x0bX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?Q\x10euy\x80\xc6X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xe2\xb8\x8d\xea\xc8\xd4\x96u.' and reward: 0.3848
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x00X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\r4n"R\x1aX\x15\x00\x00\x00embedding_size_factorq\x03G?\xed\xef\n\xa3\x07\x86\x0bX\r\x00\x00\x00layers.choiceq\x04K\x00X\r\x00\x00\x00learning_rateq\x05G?Q\x10euy\x80\xc6X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x00X\x0c\x00\x00\x00weight_decayq\x08G=\xe2\xb8\x8d\xea\xc8\xd4\x96u.' and reward: 0.3848
 60%|██████    | 3/5 [01:46<01:09, 34.73s/it] 60%|██████    | 3/5 [01:46<01:11, 35.60s/it]
Loading: dataset/models/NeuralNetClassifier/train_tabNNdataset.pkl
Loading: dataset/models/NeuralNetClassifier/validation_tabNNdataset.pkl
Saving dataset/models/NeuralNetClassifier/trial_6_tabularNN.pkl
Finished Task with config: {'activation.choice': 2, 'dropout_prob': 0.3435911380193319, 'embedding_size_factor': 0.8910141523128791, 'layers.choice': 2, 'learning_rate': 0.008241124015060441, 'network_type.choice': 0, 'use_batchnorm.choice': 1, 'weight_decay': 8.805496091833857e-07} and reward: 0.379
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\xfde\xaf?B\xcfX\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\x830\x1c\x8e\x9f\x91X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x80\xe0\xb8\xf1\x04\x06 X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xad\x8b\xdd\x11y\x95\xc4u.' and reward: 0.379
Finished Task with config: b'\x80\x03}q\x00(X\x11\x00\x00\x00activation.choiceq\x01K\x02X\x0c\x00\x00\x00dropout_probq\x02G?\xd5\xfde\xaf?B\xcfX\x15\x00\x00\x00embedding_size_factorq\x03G?\xec\x830\x1c\x8e\x9f\x91X\r\x00\x00\x00layers.choiceq\x04K\x02X\r\x00\x00\x00learning_rateq\x05G?\x80\xe0\xb8\xf1\x04\x06 X\x13\x00\x00\x00network_type.choiceq\x06K\x00X\x14\x00\x00\x00use_batchnorm.choiceq\x07K\x01X\x0c\x00\x00\x00weight_decayq\x08G>\xad\x8b\xdd\x11y\x95\xc4u.' and reward: 0.379
Please either provide filename or allow plot in get_training_curves
Time for Neural Network hyperparameter optimization: 161.63429617881775
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
Fitting model: weighted_ensemble_k0_l1 ... Training model for up to 119.75s of the -119.8s of remaining time.
Ensemble size: 18
Ensemble weights: 
[0.77777778 0.05555556 0.05555556 0.05555556 0.05555556 0.
 0.        ]
	0.3936	 = Validation accuracy score
	1.72s	 = Training runtime
	0.0s	 = Validation runtime
Saving dataset/models/weighted_ensemble_k0_l1/model.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
Saving dataset/models/trainer.pkl
AutoGluon training complete, total runtime = 241.57s ...
Loading: dataset/models/trainer.pkl

  #### save the trained model  ####################################### 

  #### Predict   #################################################### 
Loaded data from: https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv | Columns = 15 / 15 | Rows = 9769 -> 9769
Loading: dataset/models/trainer.pkl
Loading: dataset/models/weighted_ensemble_k0_l1/model.pkl
Loading: dataset/models/LightGBMClassifier/trial_2_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_0_model.pkl
Loading: dataset/models/LightGBMClassifier/trial_3_model.pkl
Loading: dataset/models/NeuralNetClassifier/trial_4_tabularNN.pkl
Loading: dataset/models/LightGBMClassifier/trial_1_model.pkl

  #### Plot   ####################################################### 

  #### Save/Load   ################################################## 
Saving dataset/learner.pkl
TabularPredictor saved. To load, use: TabularPredictor.load(dataset/)
<mlmodels.model_gluon.util_autogluon.Model_empty object at 0x7f4d0e09f470>

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
[master 8234975] ml_store  && git pull --all
 1 file changed, 205 insertions(+)
To github.com:arita37/mlmodels_store.git
 + f9dbaa0...8234975 master -> master (forced update)





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
